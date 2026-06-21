#!/usr/bin/env python3
"""Fetch WeChat article body via allorigins proxy (WeChat blocks our direct IP).
Caches data/raw/<msgid>_<itemidx>.json. Resumable: skips already-fetched.

Usage:
    python3 fetch_article.py [--limit N] [--start POS] [--only msgid] [--workers W]
"""
import json, time, urllib.request, urllib.parse, pathlib, argparse, datetime, sys, re, html, threading
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "index.json"
RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122 Safari/537.36")


class ContentExtractor(HTMLParser):
    """Capture text inside the div#js_content subtree, depth-tracked."""
    def __init__(self):
        super().__init__()
        self.cap = False; self.depth = 0; self.parts = []
    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if not self.cap and tag == "div" and d.get("id") == "js_content":
            self.cap = True; self.depth = 1; return
        if self.cap:
            if tag in ("p", "br", "section", "li", "h1", "h2", "h3", "div", "tr"):
                self.parts.append("\n")
            if tag == "div":
                self.depth += 1
            if tag == "img" and d.get("data-src"):
                self.parts.append("[img]")
    def handle_endtag(self, tag):
        if self.cap and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                self.cap = False
    def handle_data(self, data):
        if self.cap and data.strip():
            self.parts.append(data)


def extract(htmltext):
    m = (re.search(r'var\s+msg_title\s*=\s*[\'"](.*?)[\'"]\s*\.html', htmltext)
         or re.search(r'<h1[^>]*class="rich_media_title"[^>]*>(.*?)</h1>', htmltext, re.S)
         or re.search(r'<meta property="og:title" content="(.*?)"', htmltext))
    title = html.unescape(re.sub('<[^>]+>', '', m.group(1)).strip()) if m else ""
    dm = re.search(r'var\s+ct\s*=\s*[\'"](\d+)[\'"]', htmltext)
    date = datetime.datetime.utcfromtimestamp(int(dm.group(1))).strftime("%Y-%m-%d") if dm else ""
    p = ContentExtractor(); p.feed(htmltext)
    body = html.unescape("".join(p.parts))
    body = re.sub(r'\n[ \t]+', '\n', body); body = re.sub(r'\n{3,}', '\n\n', body)
    body = re.sub(r'[ \t]{2,}', ' ', body).strip()
    return title, date, body


def fetch(article_url, retries=6):
    enc = urllib.parse.quote(article_url, safe="")
    proxy = f"https://api.allorigins.win/raw?url={enc}"
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(proxy, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=100) as r:
                data = r.read().decode("utf-8", errors="ignore")
            if len(data) > 40000 and "js_content" in data:
                return data
            last = f"short/no-content ({len(data)}B)"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(2.5 * (i + 1))
    raise RuntimeError(f"fetch failed after {retries}: {last}")


_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0}


def process_one(a):
    out = RAW / f"{a['msgid']}_{a['itemidx']}.json"
    if out.exists():
        try:
            if len(json.loads(out.read_text()).get("body", "")) > 200:
                with _lock: _stats["skip"] += 1
                return
        except Exception:
            pass
    try:
        raw = fetch(a["url"])
        title, date, body = extract(raw)
        rec = {"pos": a["pos"], "msgid": a["msgid"], "itemidx": a["itemidx"],
               "title": title or a["title"], "wechat_title": a["title"],
               "date": date or a["date"], "url": a["url"],
               "chars": len(body), "body": body}
        out.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        with _lock:
            _stats["ok"] += 1
            print(f"  OK pos={a['pos']:>3} {rec['chars']:>5}c  {a['title'][:34]}", flush=True)
    except Exception as e:
        with _lock:
            _stats["fail"] += 1
            print(f"  FAIL pos={a['pos']:>3} {a['title'][:30]} :: {e}", file=sys.stderr, flush=True)
    time.sleep(1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="min pos")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    idx = json.loads(INDEX.read_text())
    todo = [a for a in idx if a["pos"] >= args.start]
    if args.only:
        todo = [a for a in idx if a["msgid"] == args.only]
    if args.limit:
        todo = todo[:args.limit]
    print(f"start: {len(todo)} articles, workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed(ex.submit(process_one, a) for a in todo))
    print(f"\nDONE ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']} | raw dir: {RAW}", flush=True)


if __name__ == "__main__":
    main()
