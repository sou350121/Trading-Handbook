"""Receipts and rollback: an incomplete scan must not publish an empty page."""
import datetime as dt
import json
import os
import signal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERRORS = []


class SourceCancelled(BaseException):
    pass


def fail(component, reason):
    ERRORS.append({'component': component, 'reason': reason})


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def run(component, main):
    ERRORS.clear()
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    roots = [ROOT / 'radar', ROOT / 'deployment']
    def paths():
        out = [p for root in roots for p in root.rglob('*.md')]
        for pattern in ['*seen.json', '*state.json', '*queue.json']:
            out += list((ROOT / 'data').glob(pattern))
        return out
    before = {p: p.read_bytes() for p in paths()}
    verified_partial = False
    cancelled = False
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(*_):
        raise SourceCancelled()
    signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            result = main()
        finally:
            # Complete rollback/receipt without a second SIGTERM interrupting it.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Only an explicitly opted-in source collector may retain its independently
        # verified outputs. An exception never gets this completion marker.
        verified_partial = component == 'practitioner' and result == 'verified_partial_outputs'
    except SourceCancelled:
        cancelled = True
        fail(component, 'cancelled')
    except Exception as exc:
        fail(component, 'exception_' + type(exc).__name__)
    if ERRORS and not verified_partial:
        for path in set(paths()) | set(before):
            if path in before:
                path.write_bytes(before[path])
            elif path.exists():
                path.unlink()  # Only new files created by this failed transaction.
    receipt = {'component': component, 'started_at': started,
               'finished_at': dt.datetime.now(dt.timezone.utc).isoformat(),
               'status': ('partial' if verified_partial else 'failed') if ERRORS else 'success', 'errors': list(ERRORS)}
    atomic(ROOT / 'data/health' / (component + '.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    signal.signal(signal.SIGTERM, previous)
    if cancelled:
        return 143
    return (2 if verified_partial else 1) if ERRORS else 0
