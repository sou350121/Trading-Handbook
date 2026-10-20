#!/usr/bin/env python3.11
"""Index committed Handbook Markdown with the already deployed local BM25 engine."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, '/home/admin/clawd/scripts')
from _local_memory_search import make_index, load_index, search_index, MAX_INDEX_BYTES
from _health import atomic

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'data/trading-search-index.json'
MANIFEST = ROOT / 'data/trading-search-manifest.json'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')


def build():
    head = git('rev-parse', 'HEAD').strip()
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
    atomic(PATH, index)
    loaded = load_index(str(PATH), fresh=True)
    if len({c['source'] for c in loaded['chunks']}) != len(paths):
        raise ValueError('index_document_missing')
    if git('rev-parse', 'HEAD').strip() != head:
        raise ValueError('index_head_changed')
    atomic(MANIFEST, {'head': head, 'documents': len(paths), 'chunks': len(chunks),
        'built_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'engine': index['engine'],
        'index_sha256': hashlib.sha256(PATH.read_bytes()).hexdigest(),
        'date_semantics': 'repository snapshot commit date, not paper publication date'})
    return {'documents': len(paths), 'chunks': len(chunks), 'reused': reused}


def check():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    index = load_index(str(PATH), fresh=True)
    if manifest['head'] != git('rev-parse', 'HEAD').strip():
        raise ValueError('index_head_stale')
    if manifest['index_sha256'] != hashlib.sha256(PATH.read_bytes()).hexdigest():
        raise ValueError('index_hash_mismatch')
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--query')
    args = parser.parse_args()
    if args.query:
        print(json.dumps(search_index(check(), args.query, days=365, top=5), ensure_ascii=False))
    elif args.check:
        print(json.dumps({'status': 'success', 'chunks': check()['chunk_count']}))
    else:
        print(json.dumps(build()))


if __name__ == '__main__':
    main()
