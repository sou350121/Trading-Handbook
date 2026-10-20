"""Receipts and rollback: an incomplete scan must not publish an empty page."""
import datetime as dt
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ERRORS = []


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
    try:
        main()
    except Exception as exc:
        fail(component, 'exception_' + type(exc).__name__)
    if ERRORS:
        for path in set(paths()) | set(before):
            if path in before:
                path.write_bytes(before[path])
            elif path.exists():
                path.unlink()  # Only new files created by this failed transaction.
    receipt = {'component': component, 'started_at': started,
               'finished_at': dt.datetime.now(dt.timezone.utc).isoformat(),
               'status': 'failed' if ERRORS else 'success', 'errors': list(ERRORS)}
    atomic(ROOT / 'data/health' / (component + '.json'), receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 1 if ERRORS else 0
