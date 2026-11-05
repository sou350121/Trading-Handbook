"""Derive numeric evidence from visible CommonMark without rewriting the source."""
from functools import lru_cache
from pathlib import Path
import sys


@lru_cache(maxsize=4)
def _visible(body):
    dependencies = Path(__file__).resolve().parents[2] / '.pulsar-deps'
    if dependencies.is_dir() and str(dependencies) not in sys.path:
        sys.path.insert(0, str(dependencies))
    try:
        import markdown_it
        if markdown_it.__version__ != '4.2.0':
            raise ImportError('pinned_parser_required')
    except ImportError:
        raise ValueError('source_markdown_parser_dependency_missing') from None
    if len(body.encode('utf-8')) > 600_000:
        raise ValueError('source_markdown_outside_bounds')
    tokens = markdown_it.MarkdownIt('commonmark', {'html': True, 'maxNesting': 32}).parse(body)
    parts = []
    for token in tokens:
        if token.type == 'html_block':
            # Unreviewed raw HTML cannot supply visible numerical evidence.
            raise ValueError('source_markdown_raw_html_requires_review')
        if token.type in ('fence', 'code_block'):
            parts.append(token.content)
        elif token.type == 'inline':
            for child in token.children or []:
                if child.type == 'html_inline':
                    raise ValueError('source_markdown_raw_html_requires_review')
                if child.type in ('text', 'code_inline'):
                    parts.append(child.content)
                elif child.type in ('softbreak', 'hardbreak'):
                    parts.append('\n')
                # Image alt text and URL destinations are not inspected evidence.
    # Preserve boundaries: separate nodes must not invent a joined decimal.
    return '\n'.join(parts)


def numeric_source_text(raw):
    body = raw['body']
    return _visible(body) if raw.get('source_format') == 'markdown' else body
