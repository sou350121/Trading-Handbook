#!/usr/bin/env python3.11
"""One locked Trading runner. Default is preparation; publishing is explicit.

All subprocesses have process-group deadlines. Successful source receipts, a clean
quality gate, push readback and a current offline index are required for success.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from _health import atomic

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
ALLOWED = ('foundations/', 'crossing/', 'cheat-sheet/', 'bridge-to-vla/', 'radar/', 'deployment/', 'guides/')
SINGLES = {'docs.json', 'data/index.json', 'data/corpus_meta.json'}


class PipelineCancelled(BaseException):
    """Cancellation must pass through partial-mode RuntimeError handlers."""


def code_digest():
    digest = hashlib.sha256()
    paths = [p for p in (ROOT / 'scripts').rglob('*') if p.is_file() and p.suffix in ('.py', '.sh', '.json', '.md')]
    paths.append(ROOT / 'data/taxonomy.json')
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def allowed(path):
    return path in SINGLES or path.startswith(ALLOWED)


def execute(command, seconds=600):
    print('STEP', ' '.join(command), flush=True)
    proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
    try:
        code = proc.wait(timeout=seconds)
    except BaseException as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        if not isinstance(exc, subprocess.TimeoutExpired):
            raise
        raise RuntimeError('step_timeout') from None
    if code:
        raise RuntimeError('step_failed_' + str(code))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode('utf-8')


def dirty_paths():
    rows = git('status', '--porcelain', '-z').split('\0')
    # Renames are never part of the automated output contract.
    if any(x and (x[0] in 'RC' or x[1] in 'RC') for x in rows):
        raise RuntimeError('rename_requires_review')
    return [x[3:] for x in rows if x]


def pending():
    import distill_pass_b as pb
    index = json.loads((ROOT / 'data/index.json').read_text())
    absent = [a for a in index if not (ROOT / 'data/distill' / (str(a['msgid']) + '.json')).exists()]
    missing = []
    for path in (ROOT / 'data/distill').glob('*.json'):
        d = json.loads(path.read_text())
        if d.get('page_worthy') and d.get('rating') in ('⚡', '🔧') and not d.get('dup_of'):
            if not (ROOT / 'foundations' / d['zone'] / (pb.slugify(d) + '.md')).exists():
                missing.append(d['msgid'])
    if absent or missing:
        raise RuntimeError('content_backlog_remaining')


def validate_outputs(paths):
    import distill_pass_b as pb
    from _content_gate import validate_page
    records = {}
    for record in (ROOT / 'data/distill').glob('*.json'):
        d = json.loads(record.read_text())
        if d.get('zone'):
            records['foundations/' + d['zone'] + '/' + pb.slugify(d) + '.md'] = d
    for path in paths:
        if not path.startswith('foundations/') or not path.endswith('.md') or path.endswith('/overview.md') or path == 'foundations/overview.md':
            continue
        if subprocess.run(['git', 'cat-file', '-e', 'HEAD:' + path], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('existing_article_changed_requires_review')
        d = records.get(path)
        if not d:
            raise RuntimeError('new_article_metadata_missing')
        raws = list((ROOT / 'data/raw').glob(str(d['msgid']) + '_*.json'))
        if not raws:
            raise RuntimeError('new_article_source_missing')
        validate_page((ROOT / path).read_text(), json.loads(raws[0].read_text()), d)


def process(job, publish=False, days=8, sources_only=False, allow_verified_partial=False):
    incomplete = []
    if publish:
        if git('diff', '--cached', '--name-only').strip():
            raise RuntimeError('preexisting_staged_changes')
        if any(not allowed(p) for p in dirty_paths()):
            raise RuntimeError('unmanaged_changes_require_review')
        if not dirty_paths():
            execute(['git', 'pull', '--rebase', '--quiet'])
    step = lambda name, *args, seconds=900: execute([PYTHON, 'scripts/pulsar/' + name, *args], seconds)
    if job == 'daily':
        # Keep the normal 1200s cron envelope. Failed legacy sources must not
        # consume the whole run before writing/audit/publication can start.
        step('enumerate_album.py', seconds=60)
        for name, args, seconds in [('fetch_article.py', ['--workers', '1', '--budget-seconds', '180'], 210),
                                    ('distill_pass_a.py', ['--workers', '1'], 120),
                                    ('fix_collisions.py', [], 30),
                                    ('distill_pass_b.py', ['--workers', '1'], 600)]:
            try:
                step(name, *args, seconds=seconds)
            except RuntimeError:
                incomplete.append(name)
                if not allow_verified_partial:
                    raise
        try:
            pending()
        except RuntimeError:
            incomplete.append('content_backlog_remaining')
            if not allow_verified_partial:
                raise
    else:
        # A failed upstream remains failed; never run dissections of an empty scan.
        failures = []
        for name, args in [
            ('arxiv_radar.py', ['--days', str(days), '--max-classify', '300']),
            ('journal_radar.py', ['--days', str(days + 1)]),
            ('practitioner_radar.py', ['--days', str(days), '--cap', '300'] +
             (['--allow-verified-partial'] if allow_verified_partial else [])),
            ('gh_quant_issues.py', [])]:
            try:
                step(name, *args, seconds=2400)
            except RuntimeError:
                failures.append(name)
        incomplete.extend(failures)
        if failures and not allow_verified_partial:
            raise RuntimeError('source_scan_incomplete')
        if not sources_only:
            for args in [[], ['--oa']]:
                try:
                    step('arxiv_dissect.py', *args, '--limit', '6', '--workers', '1', seconds=2400)
                except RuntimeError:
                    incomplete.append('dissection_incomplete')
                    if not allow_verified_partial:
                        raise
            try:
                pending()
            except RuntimeError:
                incomplete.append('content_backlog_remaining')
                if not allow_verified_partial:
                    raise
    if not sources_only:
        for name in ['add_fingerprint.py', 'gen_overviews.py', 'gen_nav.py', 'export_corpus_meta.py']:
            execute([PYTHON, 'scripts/' + name])
        step('fix_github_render.py')
    markdown_outputs = [p for p in dirty_paths() if p.endswith('.md') and allowed(p)]
    if markdown_outputs:
        step('github_links.py', '--paths', *markdown_outputs)
    execute([PYTHON, 'scripts/handbook_audit.py'])
    if publish:
        paths = dirty_paths()
        if any(not allowed(p) for p in paths):
            raise RuntimeError('unexpected_output_requires_review')
        if paths:
            validate_outputs(paths)
            execute(['git', 'add', '--', *paths])
            staged = git('diff', '--cached', '--name-only').splitlines()
            if any(not allowed(p) for p in staged):
                raise RuntimeError('staged_scope_violation')
            env = {'GIT_AUTHOR_NAME': 'Pulsar Bot', 'GIT_AUTHOR_EMAIL': 'noreply@users.noreply.github.com',
                   'GIT_COMMITTER_NAME': 'Pulsar Bot', 'GIT_COMMITTER_EMAIL': 'noreply@users.noreply.github.com'}
            os.environ.update(env)
            execute(['git', 'commit', '--quiet', '-m', 'auto: verified Trading ' + job + ' update'])
        execute(['git', 'push', '--quiet'])
        if git('ls-remote', 'origin', 'refs/heads/main').split()[0] != git('rev-parse', 'HEAD').strip():
            raise RuntimeError('remote_head_unverified')
    # Index committed pages only. Drafts never leak into search as published content.
    step('local_index.py')
    step('local_index.py', '--check')
    return incomplete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', choices=['daily', 'weekly'], required=True)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--days', type=int, default=8)
    parser.add_argument('--sources-only', action='store_true')
    parser.add_argument('--allow-verified-partial', action='store_true', help='publish only passed outputs while retaining failed health status')
    args = parser.parse_args()
    def terminated(*_):
        raise PipelineCancelled()
    signal.signal(signal.SIGTERM, terminated)
    if not 1 <= args.days <= 90:
        parser.error('days must be 1..90')
    # One common lock covers daily, weekly, recovery and index writes.
    lock = (ROOT / 'data/.pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        return 75
    receipt = {'job': args.job, 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
               'status': 'running', 'published': args.publish, 'sources_only': args.sources_only,
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'code_sha256': code_digest()}
    path = ROOT / 'data/health' / ('pipeline-' + args.job + '.json')
    atomic(path, receipt)
    try:
        incomplete = process(args.job, args.publish, args.days, args.sources_only, args.allow_verified_partial)
        receipt['status'] = ('partial' if incomplete else 'success') if args.publish else 'prepared'
        receipt['incomplete'] = incomplete
        code = 2 if incomplete else 0
    except PipelineCancelled:
        receipt.update(status='failed', reason='runner_terminated')
        code = 143
    except Exception as exc:
        receipt.update(status='failed', reason=str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__)
        code = 1
    receipt.update(finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), head=git('rev-parse', 'HEAD').strip())
    atomic(path, receipt)
    print(json.dumps(receipt))
    lock.close()
    return code


if __name__ == '__main__':
    sys.exit(main())
