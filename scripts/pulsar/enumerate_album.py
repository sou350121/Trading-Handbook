#!/usr/bin/env python3
"""Enumerate ALL articles in QuantML's WeChat album via the public getalbum JSON.
Writes data/index.json: [{pos,title,url,msgid,itemidx,create_time,date}].
Pagination: pass begin_msgid/begin_itemidx of the last item until continue_flag==0.
"""
import json, time, urllib.request, urllib.parse, pathlib, datetime

BIZ = "Mzg2MzAwNzM0NQ=="
ALBUM = "3485425931919884288"
UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0")
OUT = pathlib.Path(__file__).resolve().parents[2] / "data" / "index.json"

def fetch(begin_msgid=None, begin_itemidx=None):
    q = {"action":"getalbum","__biz":BIZ,"album_id":ALBUM,"f":"json","count":"20"}
    if begin_msgid:
        q["begin_msgid"]=begin_msgid; q["begin_itemidx"]=begin_itemidx
    url = "https://mp.weixin.qq.com/mp/appmsgalbum?"+urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent":UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))["getalbum_resp"]

seen = {}
order = []
begin_msgid = begin_itemidx = None
for page in range(40):  # safety cap; 399/20 ~ 20 pages
    resp = fetch(begin_msgid, begin_itemidx)
    al = resp.get("article_list", [])
    if not al:
        break
    new = 0
    for a in al:
        k = (a["msgid"], a["itemidx"])
        if k in seen:
            continue
        seen[k] = True
        ct = int(a.get("create_time", 0))
        order.append({
            "pos": int(a.get("pos_num", 0)),
            "title": a["title"],
            "url": a["url"].replace("http://","https://"),
            "msgid": a["msgid"],
            "itemidx": a["itemidx"],
            "create_time": ct,
            "date": datetime.datetime.utcfromtimestamp(ct).strftime("%Y-%m-%d") if ct else "",
        })
        new += 1
    last = al[-1]
    begin_msgid, begin_itemidx = last["msgid"], last["itemidx"]
    print(f"page {page+1}: +{new} (total {len(order)}) continue={resp.get('continue_flag')}")
    if str(resp.get("continue_flag")) != "1" or new == 0:
        break
    time.sleep(1.0)

order.sort(key=lambda x: x["pos"])
OUT.write_text(json.dumps(order, ensure_ascii=False, indent=2))
print(f"\nWROTE {len(order)} articles -> {OUT}")
if order:
    print("date range:", order[0]["date"], "->", order[-1]["date"])
