#!/usr/bin/env python3.11
"""Index committed Handbook Markdown with the already deployed local BM25 engine."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, '/home/admin/clawd/scripts')
from _local_memory_search import make_index, load_index, search_index, MAX_INDEX_BYTES
from _health import atomic

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'data/trading-search-index.json'
MANIFEST = ROOT / 'data/trading-search-manifest.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')


def published_head(expected=None):
    head = git('rev-parse', '--verify', 'HEAD^{commit}').strip()
    remote = git('rev-parse', '--verify', 'refs/remotes/origin/main^{commit}').strip()
    if (head != remote or (expected is not None and head != expected)
            or git('rev-parse', '--verify', 'HEAD^{commit}').strip() != head
            or git('rev-parse', '--verify', 'refs/remotes/origin/main^{commit}').strip() != remote):
        raise ValueError('index_head_unpublished_or_changed')
    return head


def build():
    head = published_head()
    day = git('show', '-s', '--format=%cI', head).strip()[:10]
    paths = sorted(p for p in git('ls-tree', '-r', '--name-only', head).splitlines()
                   if p.endswith('.md') and not p.startswith(('scripts/', 'data/')))
    chunks = []
    for path in paths:
        body = git('show', head + ':' + path)
        for section_no, section in enumerate(re.split(r'\n(?=## )', body)):
            for offset in range(0, len(section), 2200):
                text = section[offset:offset + 2200].strip()
                if not text:
                    continue
                chunks.append({'id': hashlib.sha256(f'{path}:{section_no}:{offset}'.encode()).hexdigest(),
                    'text': text, 'date': day, 'source_type': 'trading_handbook',
                    'source': path, 'section': section.splitlines()[0].lstrip('# ').strip()})
    if not paths or not chunks:
        raise ValueError('published_content_empty')
    index, reused = make_index(chunks)
    if len(json.dumps(index, ensure_ascii=False, indent=2).encode('utf-8')) > MAX_INDEX_BYTES:
        raise ValueError('index_too_large')
    PATH.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='trading-index-', dir=PATH.parent) as staging:
        staging = Path(staging)
        candidate = staging / 'index.json'
        atomic(candidate, index)
        loaded = load_index(str(candidate), fresh=True)
        if len({c['source'] for c in loaded['chunks']}) != len(paths):
            raise ValueError('index_document_missing')
        manifest = staging / 'manifest.json'
        atomic(manifest, {'head': head, 'documents': len(paths), 'chunks': len(chunks),
            'built_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'engine': index['engine'],
            'index_sha256': hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'date_semantics': 'repository snapshot commit date, not paper publication date'})
        backups = {}
        for target in (PATH, MANIFEST):
            backup = staging / (target.name + '.previous')
            if target.exists():
                backup.write_bytes(target.read_bytes())
            backups[target] = backup
        published_head(head)
        replaced = []
        try:
            for source, target in ((candidate, PATH), (manifest, MANIFEST)):
                replaced.append(target)
                os.replace(source, target)
            published_head(head)
        except BaseException:
            for target in reversed(replaced):
                if backups[target].exists():
                    os.replace(backups[target], target)
                else:
                    target.unlink(missing_ok=True)
            raise
    return {'documents': len(paths), 'chunks': len(chunks), 'reused': reused}


def check():
    head = published_head()
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    index = load_index(str(PATH), fresh=True)
    if manifest['head'] != head:
        raise ValueError('index_head_stale')
    if manifest['index_sha256'] != hashlib.sha256(PATH.read_bytes()).hexdigest():
        raise ValueError('index_hash_mismatch')
    published_head(head)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--query')
    args = parser.parse_args()
    if args.query:
        head = published_head()
        results = search_index(check(), args.query, days=365, top=5)
        published_head(head)
        print(json.dumps(results, ensure_ascii=False))
    elif args.check:
        print(json.dumps({'status': 'success', 'chunks': check()['chunk_count']}))
    else:
        print(json.dumps(build()))


if __name__ == '__main__':
    main()
