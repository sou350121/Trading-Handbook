"""Fail closed for NEW pages; never rewrite previously published articles."""
import re
from decimal import Decimal

NUMBER = re.compile(r'(?<![\w])[-−]?\d+(?:,\d{3})*(?:\.\d+)?%?')
HTML_TEXT_VERSION = 3


def validate_metadata(d, axes, zones):
    if d.get('zone') not in zones or d.get('zone') == 'unsorted' and d.get('page_worthy'):
        raise ValueError('classification_zone_invalid')
    if any(d.get('axes', {}).get(k) not in values for k, values in axes.items()):
        raise ValueError('classification_axes_invalid')
    if d.get('rating') not in ('⚡', '🔧', '📖', '❌') or type(d.get('page_worthy')) is not bool:
        raise ValueError('classification_schema_invalid')
    if not isinstance(d.get('source'), dict) or any(v is not None and not isinstance(v, str) for v in d.get('source', {}).values()) or any(not isinstance(d.get(k), str) or not d[k].strip() for k in ('tldr', 'key_trick')):
        raise ValueError('classification_metadata_missing')
    arxiv = d['source'].get('arxiv')
    if arxiv not in (None, '', 'null') and not re.fullmatch(r'\d{4}\.\d{4,5}(?:v\d+)?', arxiv):
        raise ValueError('classification_arxiv_invalid')


def validate_page(text, raw, d):
    if len(raw.get('body', '')) < 300:
        raise ValueError('source_too_thin')
    origin = (d.get('source') or {}).get('origin')
    if origin in ('arxiv', 'oa') and not raw.get('source_kind', '').startswith('fulltext'):
        raise ValueError('fulltext_provenance_missing')
    if origin in ('arxiv', 'oa') and raw.get('source_kind') == 'fulltext-html' and raw.get('extraction_version') != HTML_TEXT_VERSION:
        raise ValueError('source_extraction_requires_refresh')
    if len(text) < 1800 or not re.search(r'<!--\s*ontology-5axis\s', text[:600]):
        raise ValueError('page_structure_invalid')
    if d.get('axes'):
        expected = '<!-- ontology-5axis ' + ' '.join(k + '=' + d['axes'][k] for k in ('data', 'horizon', 'paradigm', 'alpha', 'autonomy')) + ' -->'
        if expected not in text[:600]:
            raise ValueError('ontology_metadata_mismatch')
    for section in range(1, 9):
        if not re.search(r'^##\s*§' + str(section) + r'\s*[·.／/ ]', text, re.M):
            raise ValueError('required_section_missing')
    if not d.get('url') or d['url'] not in text:
        raise ValueError('source_link_missing')
    # Empirical claims outside the explicitly hypothetical worked example must be
    # verbatim source tokens. Delta is allowed only in its own table column with
    # the two source-backed operands in the same row, at matching units.
    source = set(NUMBER.findall(raw['body']))
    known_arxiv = set(re.findall(r'\d{4}\.\d{4,5}(?:v\d+)?', raw['body'] + ' ' + raw.get('url', '')))
    verified = d.get('bibliographic_verification', {}).get('arxiv')
    if isinstance(verified, str):
        known_arxiv.add(verified)
    empirical = re.sub(r'##\s*§2\.5[^\n]*\n.*?(?=\n##\s|\Z)', '', text, flags=re.S)
    empirical = re.sub(r'```.*?```', '', empirical, flags=re.S)
    for line in empirical.splitlines():
        if line.startswith(('#', '> **發布', '> **QuantML', '> **arXiv', '> **期刊', '<!--')):
            continue
        # URLs, math exponents, reference years and ordered heading markers are
        # not empirical measurements. Percentages and decimals remain checked.
        line = re.sub(r'https?://[^\s)]+|\$[^$]+\$|\b(?:19|20)\d{2}(?:[-./]\d{1,2}){0,2}\b', '', line)
        # A reference phrase plus an explicit section marker is navigation, not a
        # measurement. Keep bare decimals, percentages and metric-like uses gated.
        line = re.sub(r'(?:參見|参见|詳見|详见|參照|参照|參考|参考|見|见|照|\bsee|\bcf\.?|\bsection)\s*§\s*[1-8](?:\.\d+)+(?![\d.%])', '', line, flags=re.I)
        # The frozen template's 7.1 subheading is also sometimes rendered bold.
        # Strip its section number only; any measurements later on stay checked.
        line = re.sub(r'^\s*\*\*7\.1(?=\s+可(?:證偽預測|证伪预测))', '**', line)
        for token in NUMBER.findall(line):
            if '.' not in token and '%' not in token:
                continue
            if token in source or token in known_arxiv and 'arxiv' in line.lower():
                continue
            cells = line.split('|')
            if len(cells) >= 7 and token in cells[-2]:
                left = NUMBER.findall(cells[-4]); right = NUMBER.findall(cells[-3])
                if len(left) == len(right) == 1 and left[0] in source and right[0] in source:
                    a, b = left[0], right[0]
                    if a.endswith('%') == b.endswith('%'):
                        val = lambda n: Decimal(n.replace(',', '').replace('−', '-').rstrip('%'))
                        if val(token) == val(b) - val(a):
                            continue
            raise ValueError('unverified_empirical_number:' + token)
