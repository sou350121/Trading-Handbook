"""Validate a complete public album before atomically advancing its index."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

BIZ = 'Mzg2MzAwNzM0NQ=='
ALBUM = '3485425931919884288'
OUT = Path(__file__).resolve().parents[2] / 'data/index.json'
PUBLISHER_TZ = dt.timezone(dt.timedelta(hours=8))


def _publisher_date(timestamp):
    return dt.datetime.fromtimestamp(_positive_integer(timestamp), PUBLISHER_TZ).date().isoformat()


def _positive_integer(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError('album_integer_invalid')
    text = str(value)
    if not text.isascii() or not text.isdigit() or int(text) < 1:
        raise ValueError('album_integer_invalid')
    return int(text)


def _article_url(value, allow_http=False):
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError('album_source_invalid')
    url = urllib.parse.urlsplit(value)
    if (url.scheme not in (('https', 'http') if allow_http else ('https',))
            or url.hostname != 'mp.weixin.qq.com' or url.username is not None
            or url.password is not None or url.port not in (None, 80 if url.scheme == 'http' else 443)
            or not (url.path.startswith('/s/') and len(url.path) > 3
                    or url.path == '/s' and url.query)):
        raise ValueError('album_source_invalid')
    if url.scheme == 'http':
        return urllib.parse.urlunsplit(('https', 'mp.weixin.qq.com', url.path, url.query, url.fragment))
    return value


def _index_digest(records):
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _validated_records(records, allow_empty=False):
    if not isinstance(records, list) or (not records and not allow_empty):
        raise ValueError('album_records_invalid')
    identities = set()
    required = ('msgid', 'itemidx', 'title', 'url', 'pos', 'create_time', 'date')
    for record in records:
        if not isinstance(record, dict) or not all(k in record for k in required):
            raise ValueError('album_item_invalid')
        for name in ('msgid', 'itemidx', 'pos', 'create_time'):
            _positive_integer(record[name])
        key = (str(record['msgid']), str(record['itemidx']))
        if key in identities:
            raise ValueError('album_duplicate_identity')
        identities.add(key)
        if not isinstance(record['title'], str) or not record['title'].strip():
            raise ValueError('album_metadata_invalid')
        _article_url(record['url'])
        dt.datetime.fromtimestamp(int(record['create_time']), dt.timezone.utc)
        date = record['date']
        if not isinstance(date, str) or dt.date.fromisoformat(date).isoformat() != date:
            raise ValueError('album_date_invalid')
    return identities


def fetch(begin_msgid=None, begin_itemidx=None):
    query = {'action': 'getalbum', '__biz': BIZ, 'album_id': ALBUM, 'f': 'json', 'count': '20'}
    if begin_msgid is not None:
        query.update(begin_msgid=begin_msgid, begin_itemidx=begin_itemidx)
    req = urllib.request.Request('https://mp.weixin.qq.com/mp/appmsgalbum?' + urllib.parse.urlencode(query),
                                 headers={'User-Agent': 'Mozilla/5.0 MicroMessenger/8.0'})
    with urllib.request.urlopen(req, timeout=30) as response:
        obj = json.load(response)
    if obj.get('base_resp', {}).get('ret', 0) != 0 or not isinstance(obj.get('getalbum_resp'), dict):
        raise ValueError('album_response_invalid')
    return obj['getalbum_resp']


def enumerate_all(fetcher=fetch, pause=time.sleep):
    records, cursors = {}, set()
    cursor = (None, None)
    for page in range(100):
        response = fetcher(*cursor)
        if not isinstance(response, dict) or response.get('base_resp', {}).get('ret', 0) != 0:
            raise ValueError('album_response_invalid')
        articles = response.get('article_list')
        flag = str(response.get('continue_flag'))
        if not isinstance(articles, list) or flag not in ('0', '1'):
            raise ValueError('album_page_invalid')
        if not articles and flag == '1':
            raise ValueError('album_incomplete_page')
        for item in articles:
            if not isinstance(item, dict) or not all(k in item for k in ('msgid', 'itemidx', 'title', 'url', 'pos_num', 'create_time')):
                raise ValueError('album_item_invalid')
            key = (str(item['msgid']), str(item['itemidx']))
            _positive_integer(item['msgid'])
            _positive_integer(item['itemidx'])
            timestamp = _positive_integer(item['create_time'])
            record = {'pos': _positive_integer(item['pos_num']), 'title': item['title'],
                'url': _article_url(item['url'], allow_http=True), 'msgid': key[0], 'itemidx': key[1],
                'create_time': timestamp, 'date': _publisher_date(timestamp)}
            _validated_records([record])
            if key in records:
                raise ValueError('album_duplicate_identity')
            records[key] = record
        print(f'page={page + 1} total={len(records)} continue={flag}', flush=True)
        if flag == '0':
            if not records:
                raise ValueError('album_empty_unverified')
            return sorted(records.values(), key=lambda x: x['pos'])
        last = articles[-1]
        cursor = (str(last['msgid']), str(last['itemidx']))
        if cursor in cursors:
            raise ValueError('album_pagination_stalled')
        cursors.add(cursor)
        pause(1)
    raise ValueError('album_pagination_limit')


def save_complete(records, path=OUT):
    new_ids = _validated_records(records)
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    old_ids = _validated_records(old, allow_empty=True)
    missing = old_ids - new_ids
    retained = [x for x in old if (str(x['msgid']), str(x['itemidx'])) in missing]
    merged = records + retained
    receipt = {'status': 'partial' if missing else 'success',
               'current_enumeration_complete': True, 'current_count': len(records),
               'retained_count': len(retained),
               'index_sha256': _index_digest(merged),
               'missing_ids': [{'msgid': msgid, 'itemidx': itemidx} for msgid, itemidx in sorted(missing)]}
    # Validate both inputs before touching either output. Each replacement is atomic.
    health = path.parent / 'health/album.json'
    health.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    receipt_tmp = health.with_name(health.name + '.tmp')
    tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt_tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
    receipt_tmp.replace(health)


def main():
    records = enumerate_all()
    save_complete(records)
    print(f'WROTE {len(records)} articles')
