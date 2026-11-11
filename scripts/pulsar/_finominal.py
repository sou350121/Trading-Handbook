"""Read the Finominal publisher sitemap and verify each candidate at its canonical page."""
import datetime as dt
import hashlib
from html.parser import HTMLParser
import re
import urllib.request
import xml.etree.ElementTree as ET

ROOT = 'https://insights.finominal.com'
SITEMAP = ROOT + '/sitemap.xml'
NS = '{http://www.sitemaps.org/schemas/sitemap/0.9}'
ARTICLE = re.compile(re.escape(ROOT) + r'/[a-z0-9]+(?:-[a-z0-9]+)+$')


def read(url, limit, mime):
    request = urllib.request.Request(url, headers={'User-Agent': 'TradingHandbookSourceVerification/1.0'})
    with urllib.request.urlopen(request, timeout=9) as response:
        body = response.read(limit + 1)
        if response.geturl() != url or len(body) > limit or response.headers.get_content_type() not in mime:
            raise ValueError('finominal_unexpected_source_response')
        return body


class ArticleMeta(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.values = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        key, value = None, None
        if tag == 'link' and attrs.get('rel') == 'canonical':
            key, value = 'canonical', attrs.get('href')
        if tag == 'meta':
            key, value = attrs.get('property'), attrs.get('content')
        if key in ('canonical', 'og:title', 'og:type', 'article:published_time') and value:
            self.values.setdefault(key, set()).add(value.strip())


def article_metadata(raw, url):
    parser = ArticleMeta()
    parser.feed(raw.decode('utf-8'))
    fields = parser.values
    if (set(fields) != {'canonical', 'og:title', 'og:type', 'article:published_time'}
            or any(len(values) != 1 for values in fields.values())
            or fields['canonical'] != {url} or fields['og:type'] != {'article'}):
        raise ValueError('finominal_article_identity_unverified')
    title = next(iter(fields['og:title'])).removesuffix(' - Finominal')
    if not title:
        raise ValueError('finominal_article_title_missing')
    published = dt.datetime.fromisoformat(next(iter(fields['article:published_time']))).date()
    return title, published


def fetch(feed, days, reader=read, today=None):
    if feed['url'] != SITEMAP or not 1 <= days <= 90:
        raise ValueError('finominal_invalid_feed')
    today = today or dt.date.today()
    cutoff = today - dt.timedelta(days=days)
    sitemap = reader(SITEMAP, 512_000, ('application/xml', 'text/xml'))
    if b'<!DOCTYPE' in sitemap.upper():
        raise ValueError('finominal_invalid_sitemap')
    try:
        root = ET.fromstring(sitemap)
    except ET.ParseError as error:
        raise ValueError('finominal_invalid_sitemap') from error
    if root.tag != NS + 'urlset':
        raise ValueError('finominal_invalid_sitemap')
    rows = {}
    for node in root.findall(NS + 'url'):
        url = node.findtext(NS + 'loc', '')
        modified = node.findtext(NS + 'lastmod', '')
        if not modified or not ARTICLE.fullmatch(url):
            continue
        if url in rows:
            raise ValueError('finominal_duplicate_article')
        rows[url] = dt.datetime.fromisoformat(modified).date()
    if len(rows) < 2:
        raise ValueError('finominal_sitemap_insufficient')
    ordered = sorted(rows, key=rows.get, reverse=True)
    candidates = [url for i, url in enumerate(ordered) if i < 2 or rows[url] >= cutoff]
    if len(candidates) > 12:
        raise ValueError('finominal_candidate_limit')
    items, articles = [], []
    for url in candidates:
        raw = reader(url, 1_800_000, ('text/html',))
        title, published = article_metadata(raw, url)
        if not published <= rows[url] <= today:
            raise ValueError('finominal_article_date_conflict')
        articles.append({'url': url, 'published': published.isoformat(),
                         'sha256': hashlib.sha256(raw).hexdigest()})
        if published >= cutoff:
            items.append({'feed_id': feed['id'], 'feed_name': feed['name'], 'title': title,
                          'link': url, 'date': published.isoformat()})
    return items, {'status': 'parsed', 'source_url': SITEMAP,
                   'retrieved_at': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'sha256': hashlib.sha256(sitemap).hexdigest(), 'articles': articles}
