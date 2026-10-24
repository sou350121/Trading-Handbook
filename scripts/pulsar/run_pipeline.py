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
from signal import getsignal, signal as set_signal
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
    return (isinstance(path, str) and not any(c in path for c in ('\\', '\0'))
            and all(part not in ('', '.', '..') for part in path.split('/'))
            and (path in SINGLES or path.startswith(ALLOWED)))


def _cleanup_process_group(process):
    """Bound cleanup and kill descendants even when the leader exits first."""
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


class _OwnedExecutionCancellation:
    """Defer SIGTERM until spawn owns a handle and while cleanup runs."""

    def __enter__(self):
        self.deferred = True
        self.pending = None
        self.previous = getsignal(signal.SIGTERM)
        set_signal(signal.SIGTERM, self.terminated)
        return self

    def terminated(self, signum, frame):
        try:
            if self.previous == signal.SIG_DFL:
                raise SystemExit(128 + signum)
            if callable(self.previous):
                self.previous(signum, frame)
        except BaseException as error:
            if self.pending is None:
                self.pending = error
            if not self.deferred:
                self.deferred = True
                raise self.pending

    def resume(self):
        self.deferred = False
        if self.pending is not None:
            self.deferred = True
            raise self.pending

    def __exit__(self, error_type, error, traceback):
        set_signal(signal.SIGTERM, self.previous)
        if error_type is None and self.pending is not None:
            raise self.pending


class _OwnedExecutionExceptionCancellation(_OwnedExecutionCancellation):
    """Defer SIGTERM while an exception is already unwinding this execute."""

    def __enter__(self):
        self.owner = sys._getframe(1)
        return super().__enter__()

    def terminated(self, signum, frame):
        error = sys.exception()
        traceback = error.__traceback__ if error is not None else None
        while traceback is not None:
            if traceback.tb_frame is self.owner:
                self.deferred = True
                break
            traceback = traceback.tb_next
        super().terminated(signum, frame)

    def __exit__(self, error_type, error, traceback):
        try:
            return super().__exit__(error_type, error, traceback)
        finally:
            self.owner = None


def execute(command, seconds=600):
    print('STEP', ' '.join(command), flush=True)
    proc = None
    with _OwnedExecutionExceptionCancellation() as cancellation:
        try:
            proc = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
            cancellation.resume()
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            if proc is None:
                raise
            cancellation.deferred = True
            _cleanup_process_group(proc)
            cancellation.resume()
            raise RuntimeError('step_timeout') from None
        except BaseException:
            cancellation.deferred = True
            if proc is not None:
                try:
                    _cleanup_process_group(proc)
                except BaseException:
                    # Cleanup must not replace the original cancellation.
                    pass
            raise
    if code:
        raise RuntimeError('step_failed_' + str(code))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.DEVNULL).decode('utf-8')


def dirty_paths():
    rows = git('status', '--porcelain', '-z', '--untracked-files=all').split('\0')
    # Renames are never part of the automated output contract.
    if any(x and any(c in x[:2] for c in 'RC') for x in rows):
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


def validate_outputs(paths, base='HEAD'):
    import distill_pass_b as pb
    from _content_gate import validate_page
    existing = set(git('ls-tree', '-r', '--name-only', '-z', base).split('\0'))
    records = {}
    for record in (ROOT / 'data/distill').glob('*.json'):
        d = json.loads(record.read_text())
        if d.get('zone'):
            records['foundations/' + d['zone'] + '/' + pb.slugify(d) + '.md'] = d
    for path in paths:
        if not path.startswith('foundations/') or not path.endswith('.md') or path.endswith('/overview.md') or path == 'foundations/overview.md':
            continue
        if path in existing:
            raise RuntimeError('existing_article_changed_requires_review')
        d = records.get(path)
        if not d:
            raise RuntimeError('new_article_metadata_missing')
        raws = list((ROOT / 'data/raw').glob(str(d['msgid']) + '_*.json'))
        if not raws:
            raise RuntimeError('new_article_source_missing')
        validate_page((ROOT / path).read_text(), json.loads(raws[0].read_text()), d)


def album_status():
    from _album import _index_digest, _validated_records
    receipt = json.loads((ROOT / 'data/health/album.json').read_text(encoding='utf-8'))
    if not isinstance(receipt, dict) or receipt.get('current_enumeration_complete') is not True:
        raise RuntimeError('album_receipt_invalid')
    current = receipt.get('current_count')
    retained = receipt.get('retained_count')
    missing = receipt.get('missing_ids')
    if (type(current) is not int or current < 1 or type(retained) is not int or retained < 0
            or not isinstance(missing, list) or len(missing) != retained):
        raise RuntimeError('album_receipt_invalid')
    missing_ids = []
    for identity in missing:
        if (not isinstance(identity, dict) or set(identity) != {'msgid', 'itemidx'}
                or any(not isinstance(identity[k], str) or not identity[k].isascii()
                       or not identity[k].isdigit() or int(identity[k]) < 1 for k in ('msgid', 'itemidx'))):
            raise RuntimeError('album_receipt_invalid')
        missing_ids.append((identity['msgid'], identity['itemidx']))
    if missing_ids != sorted(set(missing_ids)):
        raise RuntimeError('album_receipt_invalid')
    index = json.loads((ROOT / 'data/index.json').read_text(encoding='utf-8'))
    identities = _validated_records(index)
    expected = 'partial' if retained else 'success'
    # save_complete writes current records first, then unchanged history.
    retained_ids = sorted((str(x['msgid']), str(x['itemidx'])) for x in index[current:])
    if (receipt.get('status') != expected or current + retained != len(identities)
            or missing_ids != retained_ids
            or ('index_sha256' in receipt and receipt['index_sha256'] != _index_digest(index))):
        raise RuntimeError('album_receipt_inconsistent')
    return expected


def publication_path():
    return ROOT / 'data/health/pending-publication.json'


def fetch_main():
    # An unsuccessful fetch must never authorize use of an old tracking ref.
    execute(['git', 'fetch', '--quiet', '--no-tags', 'origin',
             'refs/heads/main:refs/remotes/origin/main'])
    return git('rev-parse', '--verify', 'refs/remotes/origin/main^{commit}').strip()


def publication_branch():
    if git('symbolic-ref', '--quiet', '--short', 'HEAD').strip() != 'main':
        raise RuntimeError('publication_requires_main')


def verify_pending(receipt, job):
    fields = {'version', 'job', 'base', 'head', 'tree', 'paths', 'code_sha256', 'incomplete'}
    if (not isinstance(receipt, dict) or set(receipt) != fields
            or type(receipt['version']) is not int or receipt['version'] != 1
            or receipt['job'] not in ('daily', 'weekly') or receipt['job'] != job
            or not isinstance(receipt['paths'], list) or not receipt['paths']
            or any(not allowed(p) for p in receipt['paths'])
            or receipt['paths'] != sorted(set(receipt['paths']))
            or not isinstance(receipt['incomplete'], list)
            or any(not isinstance(p, str) for p in receipt['incomplete'])):
        raise RuntimeError('pending_publication_schema_invalid')
    for field in ('base', 'head', 'tree', 'code_sha256'):
        value = receipt[field]
        lengths = (64,) if field == 'code_sha256' else (40, 64)
        if not isinstance(value, str) or len(value) not in lengths or any(c not in '0123456789abcdef' for c in value):
            raise RuntimeError('pending_publication_schema_invalid')
    publication_branch()
    if dirty_paths() or git('diff', '--cached', '--name-only', '-z'):
        raise RuntimeError('pending_publication_worktree_changed')
    head = git('rev-parse', '--verify', 'HEAD^{commit}').strip()
    paths = sorted(p for p in git('diff', '--no-renames', '--name-only', '-z',
                                 receipt['base'], head, '--').split('\0') if p)
    if (head != receipt['head']
            or git('rev-list', '--parents', '-n', '1', head).split() != [head, receipt['base']]
            or git('rev-parse', head + '^{tree}').strip() != receipt['tree']
            or paths != receipt['paths'] or code_digest() != receipt['code_sha256']):
        raise RuntimeError('pending_publication_mismatch')


def unique_receipt_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError('pending_publication_schema_invalid')
        result[key] = value
    return result


def read_pending():
    path = publication_path()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('pending_publication_file_invalid')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_receipt_fields)


def publication_preflight(job):
    publication_branch()
    if git('diff', '--cached', '--name-only', '-z'):
        raise RuntimeError('preexisting_staged_changes')
    paths = dirty_paths()
    if any(not allowed(p) for p in paths):
        raise RuntimeError('unmanaged_changes_require_review')
    # The authorization receipt must stay outside the Git snapshot.
    git('check-ignore', '--', 'data/health/pending-publication.json')
    remote = fetch_main()
    head = git('rev-parse', '--verify', 'HEAD^{commit}').strip()
    path = publication_path()
    if path.exists() or path.is_symlink():
        receipt = read_pending()
        verify_pending(receipt, job)
        if remote not in (receipt['base'], receipt['head']):
            raise RuntimeError('pending_publication_remote_changed')
        return remote, receipt
    if head != remote:
        if paths or git('merge-base', head, remote).strip() != head:
            raise RuntimeError('unreviewed_local_history')
        execute(['git', 'merge', '--ff-only', '--quiet', remote])
        if dirty_paths() or git('rev-parse', 'HEAD').strip() != remote:
            raise RuntimeError('publication_fast_forward_failed')
    return remote, None


def write_pending(receipt):
    path = publication_path()
    if path.exists() or path.is_symlink():
        raise RuntimeError('pending_publication_already_exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    descriptor = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def complete_publication(receipt, job):
    if read_pending() != receipt:
        raise RuntimeError('pending_publication_receipt_changed')
    verify_pending(receipt, job)
    remote = fetch_main()
    if remote == receipt['base']:
        # Pin the refspec to the authorized object; never push other local refs.
        execute(['git', 'push', '--quiet', 'origin', receipt['head'] + ':refs/heads/main'])
    elif remote != receipt['head']:
        raise RuntimeError('pending_publication_remote_changed')
    if fetch_main() != receipt['head']:
        raise RuntimeError('remote_head_unverified')
    verify_pending(receipt, job)
    execute([PYTHON, 'scripts/pulsar/local_index.py'])
    execute([PYTHON, 'scripts/pulsar/local_index.py', '--check'])
    verify_pending(receipt, job)
    if git('rev-parse', 'refs/remotes/origin/main').strip() != receipt['head']:
        raise RuntimeError('remote_head_unverified')
    if read_pending() != receipt:
        raise RuntimeError('pending_publication_receipt_changed')
    publication_path().unlink()


def process(job, publish=False, days=8, sources_only=False, allow_verified_partial=False):
    incomplete = []
    if publish:
        base, receipt = publication_preflight(job)
        digest = code_digest()
        if receipt is not None:
            incomplete = list(receipt['incomplete'])
            if incomplete and not allow_verified_partial:
                raise RuntimeError('pending_publication_requires_partial_opt_in')
            if job == 'daily':
                if album_status() == 'partial' and 'album_history_missing' not in incomplete:
                    raise RuntimeError('album_receipt_inconsistent')
            if job == 'daily' or not sources_only:
                try:
                    pending()
                except RuntimeError:
                    if 'content_backlog_remaining' not in incomplete:
                        raise
            validate_outputs(receipt['paths'], base=receipt['base'])
            execute([PYTHON, 'scripts/handbook_audit.py'])
            complete_publication(receipt, job)
            return incomplete
    step = lambda name, *args, seconds=900: execute([PYTHON, 'scripts/pulsar/' + name, *args], seconds)
    if job == 'daily':
        # Keep the normal 1200s cron envelope. Failed legacy sources must not
        # consume the whole run before writing/audit/publication can start.
        # A successful child must create a receipt during this enumeration.
        # Never accept a receipt left by an earlier run or a failed replacement.
        (ROOT / 'data/health/album.json').unlink(missing_ok=True)
        step('enumerate_album.py', seconds=60)
        if album_status() == 'partial':
            if not allow_verified_partial:
                raise RuntimeError('album_history_missing')
            incomplete.append('album_history_missing')
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
        publication_branch()
        if (git('rev-parse', 'HEAD').strip() != base or code_digest() != digest
                or git('diff', '--cached', '--name-only', '-z')):
            raise RuntimeError('publication_state_changed')
        paths = sorted(dirty_paths())
        if any(not allowed(p) for p in paths):
            raise RuntimeError('unexpected_output_requires_review')
        if paths:
            validate_outputs(paths)
            execute(['git', '--literal-pathspecs', 'add', '--', *paths])
            staged = sorted(p for p in git('diff', '--cached', '--name-only', '-z').split('\0') if p)
            if staged != paths or git('diff', '--name-only', '-z'):
                raise RuntimeError('staged_scope_violation')
            env = {'GIT_AUTHOR_NAME': 'Pulsar Bot', 'GIT_AUTHOR_EMAIL': 'noreply@users.noreply.github.com',
                   'GIT_COMMITTER_NAME': 'Pulsar Bot', 'GIT_COMMITTER_EMAIL': 'noreply@users.noreply.github.com'}
            os.environ.update(env)
            execute(['git', 'commit', '--quiet', '-m', 'auto: verified Trading ' + job + ' update'])
            receipt = {'version': 1, 'job': job, 'base': base,
                       'head': git('rev-parse', 'HEAD').strip(),
                       'tree': git('rev-parse', 'HEAD^{tree}').strip(),
                       'paths': paths, 'code_sha256': digest, 'incomplete': incomplete}
            verify_pending(receipt, job)
            write_pending(receipt)
            complete_publication(receipt, job)
            return incomplete
        if fetch_main() != base or git('rev-parse', 'HEAD').strip() != base:
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
