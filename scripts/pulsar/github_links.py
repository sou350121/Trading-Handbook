#!/usr/bin/env python3.11
"""Convert resolvable Mintlify links to equivalent GitHub relative file links.

Runtime callers pass their own changed outputs. --all is a one-time reviewed repair.
Labels, prose, anchors, queries and fenced examples are preserved.
"""
import argparse
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import handbook_audit as audit


def rewrite(text, path):
    protected = list(re.finditer(r'```.*?```|~~~.*?~~~', text, re.S))
    changes = []
    for match in audit.LINK_RE.finditer(text):
        if any(block.start() <= match.start() < block.end() for block in protected):
            continue
        target = match.group(1)
        if target.startswith(('http://', 'https://', 'mailto:', 'tel:', '#')):
            continue
        plain, suffix = re.match(r'([^?#]*)(.*)', target).groups()
        resolved = audit.resolve_target(path, target)
        if not isinstance(resolved, Path) or not resolved.is_file():
            continue
        if not target.startswith('/') and Path(plain).suffix:
            continue
        updated = os.path.relpath(resolved, path.parent).replace('\\', '/') + suffix
        if updated != target:
            changes.append((match.start(1), match.end(1), updated))
    for start, end, updated in reversed(changes):
        text = text[:start] + updated + text[end:]
    return text, len(changes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--paths', nargs='*', default=[])
    args = parser.parse_args()
    paths = audit.all_content_files() if args.all else [ROOT / value for value in args.paths]
    files = links = 0
    for path in paths:
        if not path.is_file() or path.suffix != '.md' or not path.resolve().is_relative_to(ROOT):
            continue
        before = path.read_text(encoding='utf-8')
        after, count = rewrite(before, path)
        if count:
            # Formatting-only preservation check, independent of the new destination.
            mask = lambda text: audit.LINK_RE.sub(lambda m: m.group(0).replace(m.group(1), '<TARGET>'), text)
            if mask(before) != mask(after):
                raise ValueError('link_rewrite_changed_content')
            path.write_text(after, encoding='utf-8')
            files += 1; links += count
    print(f'github_links: files={files} links={links}')


if __name__ == '__main__':
    main()
