"""NBER's public official metadata supplies dates absent from its RSS.

Only the documented w-series is collected. Both official TSV tables must agree,
and every current RSS identity/title must occur in the full metadata snapshot.
No retrieval timestamp, DOI registration date or paper-number estimate is used.
"""
import datetime as dt
import hashlib
import html
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BASE = 'https://data.nber.org/nber_paper_chapter_metadata/tsv/'
REF = BASE + 'ref.tsv'
DATES = BASE + 'date.tsv'
RSS = 'https://back.nber.org/rss/new.xml'
DOCS = 'https://www.nber.org/research/data/nber-working-papers-and-chapters-metadata'
ID = re.compile(r'w\d{4,5}')


def get(url, limit):
    request = urllib.request.Request(url, headers={'User-Agent': 'Trading-Handbook source verifier'})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.geturl() != url:
            raise ValueError('nber_unexpected_redirect')
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError('nber_response_too_large')
    return body


def table(body, header):
    # These TSV files contain literal quotes. CSV quote processing can silently
    # merge hundreds of rows into one title; split tabs without interpreting quotes.
    lines = body.decode('utf-8-sig').splitlines()
    if not lines or lines[0].split('\t') != header:
        raise ValueError('nber_header_invalid')
    records = {}
    for line in lines[1:]:
        fields = line.split('\t')
        if not ID.fullmatch(fields[0]):
            continue  # f/c/h/t records are outside the documented active w-series.
        if len(fields) != len(header) or fields[0] in records:
            raise ValueError('nber_record_invalid_or_duplicate')
        records[fields[0]] = dict(zip(header, fields))
    if not records:
        raise ValueError('nber_metadata_empty')
    return records


def normalize(title):
    return re.sub(r'\W+', '', html.unescape(title)).casefold()


def reconcile(ref_body, date_body, rss_body, days, today=None):
    today = today or dt.date.today()
    refs = table(ref_body, ['paper', 'author', 'title', 'issue_date', 'doi'])
    dates = table(date_body, ['paper', 'issue_date'])
    if set(refs) != set(dates):
        raise ValueError('nber_metadata_identity_mismatch')
    for paper, row in refs.items():
        if not row['title'].strip() or row['issue_date'] != dates[paper]['issue_date']:
            raise ValueError('nber_title_or_date_mismatch')
        try:
            day = dt.date.fromisoformat(row['issue_date'])
        except ValueError:
            raise ValueError('nber_publication_date_invalid') from None
        if day > today:
            raise ValueError('nber_publication_date_future')
    root = ET.fromstring(rss_body)
    if root.tag != 'rss':
        raise ValueError('nber_rss_invalid')
    entries = root.findall('./channel/item')
    if not entries:
        raise ValueError('nber_rss_empty')
    rss_ids = set()
    for entry in entries:
        link = urllib.parse.urlsplit(entry.findtext('link', ''))
        match = re.fullmatch(r'/papers/(w\d{4,5})', link.path)
        if link.scheme != 'https' or link.hostname != 'www.nber.org' or not match:
            raise ValueError('nber_rss_identity_invalid')
        paper = match[1]
        if paper in rss_ids or paper not in refs:
            raise ValueError('nber_metadata_missing_current_rss_identity')
        rss_ids.add(paper)
        rss_title = entry.findtext('title', '')
        # NBER's RSS appends its author list using this explicit separator.
        if ' -- by ' in rss_title:
            rss_title, rss_authors = rss_title.rsplit(' -- by ', 1)
            if normalize(rss_authors) != normalize(refs[paper]['author']):
                raise ValueError('nber_rss_author_mismatch')
        if normalize(rss_title) != normalize(refs[paper]['title']):
            raise ValueError('nber_rss_title_mismatch')
    cutoff = today - dt.timedelta(days=days)
    recent = [row for row in refs.values() if cutoff.isoformat() <= row['issue_date'] <= today.isoformat()]
    proof = {'metadata_source': REF, 'date_source': DATES, 'identity_crosscheck': RSS,
             'documentation': DOCS, 'metadata_sha256': hashlib.sha256(ref_body).hexdigest(),
             'date_sha256': hashlib.sha256(date_body).hexdigest(),
             'rss_sha256': hashlib.sha256(rss_body).hexdigest(),
             'working_papers': len(refs), 'rss_identities_verified': len(rss_ids),
             'window_start': cutoff.isoformat(), 'window_end': today.isoformat(),
             'recent': len(recent), 'scope': 'official w-series metadata, not full text'}
    return sorted(recent, key=lambda row: (row['issue_date'], row['paper'])), proof


def fetch(feed, days):
    rows, proof = reconcile(get(REF, 25_000_000), get(DATES, 5_000_000), get(RSS, 2_000_000), days)
    # Guard against a valid-looking truncated official snapshot.
    if proof['working_papers'] < 10_000:
        raise ValueError('nber_metadata_snapshot_incomplete')
    items = [{'feed_id': feed['id'], 'feed_name': feed['name'], 'title': row['title'],
              # Preserve the historical RSS URL identity for seen-set compatibility.
              'link': 'https://www.nber.org/papers/' + row['paper'] + '#fromrss',
              'date': row['issue_date'], 'date_source': DATES,
              'metadata_source': REF, 'metadata_sha256': proof['metadata_sha256']}
             for row in rows]
    return items, proof
