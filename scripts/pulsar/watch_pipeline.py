#!/usr/bin/env python3.11
"""Server-owned checks, bounded daily catch-up, deduplicated GitHub Issues alerts.

No model chooses shell commands. Only fixed scripts can run. Default is read-only.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from _health import atomic
from run_pipeline import execute, code_digest

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / 'data/health'
ZONE = ZoneInfo('Asia/Shanghai')
REPO = 'sou350121/Trading-Handbook'
MARKER = '<!-- trading-pipeline-watch-v1 -->'
TITLE = '[Trading Handbook] automatic pipeline alert'


def read(path):
    try:
        value = path.read_text(encoding='utf-8')
        return json.loads(value)
    except (OSError, ValueError):
        return {}


def due(now, weekly=False):
    value = now.replace(hour=15 if weekly else 21, minute=0, second=0, microsecond=0)
    while value > now or weekly and value.weekday() != 5:
        value -= dt.timedelta(days=1)
    return value


def valid_receipt(receipt, expected, now):
    try:
        start = dt.datetime.fromisoformat(receipt['started_at'])
        end = dt.datetime.fromisoformat(receipt['finished_at'])
        return (receipt.get('status') == 'success' and receipt.get('published') is True
                and receipt.get('sources_only') is False and receipt.get('code_sha256') == code_digest()
                and expected <= start <= end <= now + dt.timedelta(minutes=5))
    except (KeyError, TypeError, ValueError):
        return False


def checks(now):
    rows = []
    for job in ['daily', 'weekly']:
        expected = due(now, job == 'weekly')
        receipt = read(STATE / ('pipeline-' + job + '.json'))
        grace = 150 if job == 'weekly' else 25
        status = 'healthy' if valid_receipt(receipt, expected, now) else 'grace' if now < expected + dt.timedelta(minutes=grace) else 'needs_recovery'
        rows.append({'component': job, 'status': status, 'due': expected.isoformat()})
    try:
        from local_index import check
        check()
        status = 'healthy'
    except Exception:
        status = 'index_stale_or_missing'
    rows.append({'component': 'local-index', 'status': status})
    return rows


def notify(rows):
    bad = [x for x in rows if x['status'] not in ('healthy', 'grace')]
    if not bad and any(x['status'] != 'healthy' for x in rows):
        return 'pending'
    # Sanitize by construction. No paths, host data, source text or exception strings.
    body = MARKER + '\n' + ('Automatic checks require review.\n' if bad else 'Recovery verified.\n')
    body += '\n'.join('- ' + x['component'] + ': ' + x['status'] for x in bad)
    fingerprint = hashlib.sha256(body.encode()).hexdigest()
    history = read(STATE / 'notification.json')
    if history.get('fingerprint') == fingerprint:
        return 'unchanged'
    from gh_quant_issues import gh_token
    key = gh_token()
    if not key:
        return 'notification_credential_missing'
    def request(method, suffix, data=None):
        req = urllib.request.Request('https://api.github.com/repos/' + REPO + suffix,
            headers={'Authorization': 'Bearer ' + key, 'Accept': 'application/vnd.github+json',
                     'User-Agent': 'Trading-Handbook monitor'},
            data=json.dumps(data).encode() if data is not None else None, method=method)
        # No blind retries of a write with an uncertain outcome.
        with urllib.request.urlopen(req, timeout=25) as response:
            return json.load(response)
    try:
        issues = request('GET', '/issues?state=open&per_page=100')
        issue = next((x for x in issues if not x.get('pull_request') and MARKER in (x.get('body') or '')), None)
        if bad:
            if issue:
                marker = '<!-- trading-watch:' + fingerprint + ' -->'
                comments = request('GET', '/issues/' + str(issue['number']) + '/comments?per_page=100')
                if not any(marker in x.get('body', '') for x in comments):
                    request('POST', '/issues/' + str(issue['number']) + '/comments', {'body': marker + '\n' + body})
                result = request('PATCH', '/issues/' + str(issue['number']), {'body': body})
            else:
                result = request('POST', '/issues', {'title': TITLE, 'body': body})
            verified = request('GET', '/issues/' + str(result['number']))
            if verified.get('body') != body:
                return 'delivery_unverified'
        elif issue:
            request('POST', '/issues/' + str(issue['number']) + '/comments', {'body': body})
            request('PATCH', '/issues/' + str(issue['number']), {'state': 'closed'})
        atomic(STATE / 'notification.json', {'fingerprint': fingerprint})
        return 'delivered'
    except Exception:
        return 'delivery_failed'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--notify', action='store_true')
    args = parser.parse_args()
    now = dt.datetime.now(ZONE)
    STATE.mkdir(parents=True, exist_ok=True)
    lock = (STATE / '.watch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return 75
    rows = checks(now)
    attempts = read(STATE / 'attempts.json')
    if args.apply:
        for row in rows:
            # Daily catch-up has reserved bounded windows before other content jobs.
            # Weekly failures remain visible instead of launching overlapping heavy work.
            if row['component'] != 'daily' or row['status'] != 'needs_recovery' or (now.hour, now.minute) not in [(9, 0), (21, 30)]:
                continue
            slot = row['due']; item = attempts.get(slot, {'count': 0})
            if item['count'] >= 2:
                row['status'] = 'recovery_exhausted'
                continue
            item['count'] += 1; attempts[slot] = item
            atomic(STATE / 'attempts.json', attempts)
            try:
                execute([sys.executable, 'scripts/pulsar/run_pipeline.py', '--job', 'daily', '--publish', '--allow-verified-partial'], 1200)
            except RuntimeError as error:
                if str(error) == 'step_failed_75':
                    item['count'] -= 1
                    atomic(STATE / 'attempts.json', attempts)
        # Index rebuilding is offline and never launches an LLM or changes Git.
        if any(x['component'] == 'local-index' and x['status'] != 'healthy' for x in rows):
            pipeline_lock = (ROOT / 'data/.pipeline.lock').open('a')
            try:
                fcntl.flock(pipeline_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                execute([sys.executable, 'scripts/pulsar/local_index.py'], 300)
            except (BlockingIOError, RuntimeError):
                pass
            finally:
                pipeline_lock.close()
        rows = checks(dt.datetime.now(ZONE))
        for row in rows:
            if row['component'] == 'daily' and row['status'] == 'needs_recovery' and attempts.get(row['due'], {}).get('count', 0) >= 2:
                row['status'] = 'recovery_exhausted'
    result = {'checked_at': now.isoformat(), 'rows': rows, 'notification': notify(rows) if args.notify else 'not_requested'}
    atomic(STATE / 'watch-latest.json', result)
    print(json.dumps(result))
    return 1 if any(x['status'] not in ('healthy', 'grace') for x in rows) else 0


if __name__ == '__main__':
    sys.exit(main())
