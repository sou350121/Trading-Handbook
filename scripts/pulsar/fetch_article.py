#!/usr/bin/env python3
"""Fetch WeChat article body via allorigins proxy (WeChat blocks our direct IP).
Caches data/raw/<msgid>_<itemidx>.json. Resumable: skips already-fetched.

Usage:
    python3 fetch_article.py [--limit N] [--start POS] [--only msgid] [--workers W]
"""
import json, time, urllib.request, urllib.parse, pathlib, argparse, datetime, sys, re, html, threading, hashlib, signal
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
    # Album dates are publisher-local calendar days, not UTC calendar days.
    date = datetime.datetime.fromtimestamp(int(dm.group(1)), datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d") if dm else ""
    p = ContentExtractor(); p.feed(htmltext)
    body = html.unescape("".join(p.parts))
    body = re.sub(r'\n[ \t]+', '\n', body); body = re.sub(r'\n{3,}', '\n\n', body)
    body = re.sub(r'[ \t]{2,}', ' ', body).strip()
    return title, date, body


def fetch(article_url, retries=2):
    # Try the public original first. A CAPTCHA/HTML shell has no valid article body;
    # the existing public proxy remains a bounded fallback, never a success by size.
    enc = urllib.parse.quote(article_url, safe="")
    urls = [article_url] + [f"https://api.allorigins.win/raw?url={enc}"] * retries
    reasons = []
    for i, url in enumerate(urls):
        label = 'original' if i == 0 else 'proxy'
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as response:
                data = response.read(4_000_000).decode("utf-8", errors="replace")
                effective_path = urllib.parse.urlsplit(response.geturl()).path
            title, date, body = extract(data)
            challenge = any(x in effective_path.lower() or x in data.lower()
                            for x in ('appmsgcaptcha', 'wappoc_appmsgcaptcha'))
            if not challenge and "js_content" in data and title and date and len(body) >= 300:
                return data
            reason = 'challenge' if challenge else 'article_structure_missing'
            reasons.append(label + '_' + reason)
        except urllib.error.HTTPError as error:
            reasons.append(label + '_http_' + str(error.code))
        except (urllib.error.URLError, TimeoutError, OSError):
            reasons.append(label + '_network_error')
        if i + 1 < len(urls):
            time.sleep(2)
    # No URLs, challenge tokens or response bodies in operational diagnostics.
    raise RuntimeError('article_body_unavailable:' + ','.join(dict.fromkeys(reasons)))


_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0}
_failures = []
_deferred = []


class BatchBudgetExpired(BaseException):
    """Do not turn the batch deadline into an ordinary item retry."""


def cached(a):
    out = RAW / f"{a['msgid']}_{a['itemidx']}.json"
    try:
        return len(json.loads(out.read_text()).get('body', '')) >= 300
    except (OSError, ValueError):
        return False


def process_one(a):
    out = RAW / f"{a['msgid']}_{a['itemidx']}.json"
    if out.exists():
        try:
            if len(json.loads(out.read_text()).get("body", "")) >= 300:
                with _lock: _stats["skip"] += 1
                return
        except Exception:
            pass
    try:
        raw = fetch(a["url"])
        title, date, body = extract(raw)
        if len(body) < 300 or not date or date != a["date"]:
            raise ValueError("article_provenance_invalid")
        if re.sub(r"\W+", "", title).casefold() != re.sub(r"\W+", "", a["title"]).casefold():
            raise ValueError("article_title_mismatch")
        rec = {"pos": a["pos"], "msgid": a["msgid"], "itemidx": a["itemidx"],
               "title": title or a["title"], "wechat_title": a["title"],
               "date": date or a["date"], "url": a["url"],
               "chars": len(body), "body": body,
               "source_kind": "fulltext_quantml",
               "source_html_sha256": hashlib.sha256(raw.encode('utf-8')).hexdigest(),
               "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        from _health import atomic
        atomic(out, rec)
        with _lock:
            _stats["ok"] += 1
            print(f"  OK pos={a['pos']:>3} {rec['chars']:>5}c  {a['title'][:34]}", flush=True)
    except Exception as e:
        with _lock:
            _stats["fail"] += 1
            _failures.append({'msgid': a['msgid'], 'itemidx': a['itemidx'],
                              'pos': a['pos'], 'reason': str(e) if isinstance(e, (RuntimeError, ValueError)) else type(e).__name__})
            print(f"  FAIL pos={a['pos']:>3} {a['title'][:30]} :: {e}", file=sys.stderr, flush=True)
    time.sleep(1.5)


def bounded_fetch(todo, seconds):
    """One process, a hard wall deadline, and fair retries of missing bodies.

    Unattempted items go first (newest first on ties). Failed or interrupted
    items rotate behind older attempts. Deferral never means a verified source.
    """
    from _health import atomic
    attempts_path = ROOT / 'data/health/quantml-attempts.json'
    try:
        attempts = json.loads(attempts_path.read_text())
    except (OSError, ValueError):
        attempts = {}
    key = lambda a: str(a['msgid']) + '_' + str(a['itemidx'])
    ready = []
    for item in todo:
        if cached(item):
            _stats['skip'] += 1
        else:
            ready.append(item)
    ready.sort(key=lambda a: (attempts.get(key(a), 0), -a['pos']))
    if not ready:
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer != (0.0, 0.0):
        raise RuntimeError('preexisting_alarm_requires_review')
    def deadline(*_):
        raise BatchBudgetExpired()
    signal.signal(signal.SIGALRM, deadline)
    attempted = set()
    expired = False
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        for item in ready:
            identity = key(item)
            attempts[identity] = time.time()
            # Persist before the request, so an interrupted slow item also rotates.
            atomic(attempts_path, attempts)
            attempted.add(identity)
            process_one(item)
    except BatchBudgetExpired:
        expired = True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    if expired:
        failures = {(str(row['msgid']), str(row['itemidx'])) for row in _failures}
        for item in ready:
            if cached(item) or (str(item['msgid']), str(item['itemidx'])) in failures:
                continue
            _deferred.append({'msgid': item['msgid'], 'itemidx': item['itemidx'], 'pos': item['pos'],
                              'reason': 'batch_budget_interrupted' if key(item) in attempted else 'batch_budget_deferred'})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="min pos")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--poslist", type=str, default="", help="comma-separated pos list (disjoint shard)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-seconds", type=float, default=0, help="hard serial fetch budget; deferred items remain failed/unverified")
    args = ap.parse_args()
    if not 0 <= args.budget_seconds <= 3600 or args.budget_seconds and args.workers != 1:
        ap.error('budget must be 0..3600 seconds; positive budget requires workers=1')
    if args.budget_seconds and not hasattr(signal, 'setitimer'):
        ap.error('hard batch deadlines require POSIX setitimer')
    idx = json.loads(INDEX.read_text())
    todo = [a for a in idx if a["pos"] >= args.start]
    if args.poslist:
        want = {int(x) for x in args.poslist.split(",") if x.strip()}
        todo = [a for a in idx if a["pos"] in want]
    if args.only:
        todo = [a for a in idx if a["msgid"] == args.only]
    if args.limit:
        todo = todo[:args.limit]
    print(f"start: {len(todo)} articles, workers={args.workers}", flush=True)
    if args.budget_seconds:
        bounded_fetch(todo, args.budget_seconds)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(as_completed(ex.submit(process_one, a) for a in todo))
    from _health import atomic
    atomic(ROOT / 'data/health/quantml-fetch.json', {
        'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'failed' if _stats['fail'] or _deferred else 'success',
        'stats': dict(_stats), 'failures': sorted(_failures, key=lambda item: item['pos']),
        'deferred': sorted(_deferred, key=lambda item: item['pos']), 'budget_seconds': args.budget_seconds})
    print(f"\nDONE ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']} deferred={len(_deferred)} | raw dir: {RAW}", flush=True)


if __name__ == "__main__":
    main()
    sys.exit(1 if _stats["fail"] or _deferred else 0)
