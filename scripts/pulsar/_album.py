"""Validate a complete public album before atomically advancing its index."""
import datetime as dt
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

BIZ = 'Mzg2MzAwNzM0NQ=='
ALBUM = '3485425931919884288'
OUT = Path(__file__).resolve().parents[2] / 'data/index.json'


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
            if not all(x.isdigit() for x in key) or int(key[1]) < 1 or int(item['pos_num']) < 1:
                raise ValueError('album_identity_invalid')
            timestamp = int(item['create_time'])
            if timestamp <= 0 or not item['title'].strip() or urllib.parse.urlsplit(item['url']).hostname != 'mp.weixin.qq.com':
                raise ValueError('album_metadata_invalid')
            records[key] = {'pos': int(item['pos_num']), 'title': item['title'],
                'url': item['url'].replace('http://', 'https://'), 'msgid': key[0], 'itemidx': key[1],
                'create_time': timestamp, 'date': dt.datetime.fromtimestamp(timestamp, dt.timezone.utc).strftime('%Y-%m-%d')}
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
    old = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
    old_ids = {(str(x['msgid']), str(x['itemidx'])) for x in old}
    new_ids = {(str(x['msgid']), str(x['itemidx'])) for x in records}
    if not old_ids <= new_ids:
        raise ValueError('album_history_missing')
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    records = enumerate_all()
    save_complete(records)
    print(f'WROTE {len(records)} articles')
