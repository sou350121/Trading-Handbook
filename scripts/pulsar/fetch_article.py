#!/usr/bin/env python3
"""Fetch WeChat article body via allorigins proxy (WeChat blocks our direct IP).
Caches data/raw/<msgid>_<itemidx>.json. Resumable: skips already-fetched.

Usage:
    python3 fetch_article.py [--limit N] [--start POS] [--only msgid] [--workers W]
"""
import json, time, urllib.request, urllib.parse, pathlib, argparse, datetime, sys, re, html, threading, hashlib, signal
from contextlib import contextmanager
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[2]
INDEX = ROOT / "data" / "index.json"
RAW = ROOT / "data" / "raw"; RAW.mkdir(parents=True, exist_ok=True)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122 Safari/537.36")
FETCH_ENDPOINT = 'https://api.deepseek.com/v1/tools/fetch'
FETCH_MAX_BYTES = 2_000_000
FETCH_TIMEOUT = 20


class ArticleUnavailable(RuntimeError):
    """Only unavailable HTML permits the separately verified Fetch fallback."""


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


def fetch(article_url, retries=2, *, expected_title=None, expected_date=None):
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
            # Recognized metadata remains evidence even on a thin/challenge page.
            # Reject conflicts before trying another source or the paid fallback.
            if date and expected_date is not None and date != expected_date:
                raise ValueError('article_provenance_invalid')
            if (title and expected_title is not None
                    and re.sub(r'\W+', '', title).casefold()
                    != re.sub(r'\W+', '', expected_title).casefold()):
                raise ValueError('article_title_mismatch')
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
    raise ArticleUnavailable('article_body_unavailable:' + ','.join(dict.fromkeys(reasons)))


def markdown_visible_text(body):
    """Count readable text without changing the stored Markdown or its hash."""
    # Only complete reference definitions are metadata. A destination cannot
    # contain bare spaces, and indented/fenced code is literal visible text.
    label = re.compile(r' {0,3}\[((?:\\[^\n]|[^\[\]\\\n]){1,999})\]:[ \t]*(?:\n[ \t]*)?')
    destination = re.compile(r'<(?:\\[^\n]|[^<>\\\n])*>|(?!<)(?:\\[^\x00-\x20\x7f]|[^\x00-\x20\x7f])+')
    title = re.compile(r'''[ \t]*(?:\n[ \t]*)?(?:"(?:\\[^\n]|[^"\\])*"|'(?:\\[^\n]|[^'\\])*'|\((?:\\[^\n]|[^()\\])*\))[ \t]*(?=\n|$)''')
    line_end = re.compile(r'[ \t]*(?:\n|$)')
    text, fence, pos = [], None, 0
    source = body.replace('\r\n', '\n').replace('\r', '\n')
    while pos < len(source):
        end = source.find('\n', pos)
        end = len(source) if end < 0 else end + 1
        line = source[pos:end]
        marker = re.match(r' {0,3}(`{3,}|~{3,})([^\n]*)', line)
        if fence:
            if re.fullmatch(r' {0,3}' + re.escape(fence[0]) + '{' + str(len(fence)) + r',}[ \t]*\n?', line):
                fence = None
        elif marker and not (marker[1][0] == '`' and '`' in marker[2]):
            fence = marker[1]
        elif (match := label.match(source, pos)) and match[1].strip():
            dest = destination.match(source, match.end())
            if dest:
                value, depth, valid, escaped = dest[0], 0, True, False
                if not value.startswith('<'):
                    for char in value:
                        if escaped:
                            escaped = False
                        elif char == '\\':
                            escaped = True
                        elif char == '(':
                            depth += 1
                        elif char == ')':
                            depth -= 1
                            if depth < 0:
                                valid = False
                        elif ord(char) < 32 or ord(char) == 127:
                            valid = False
                    valid = valid and depth == 0
                tail = dest.end()
                # A title needs whitespace after the destination. A malformed
                # next-line title leaves that line as prose, not metadata.
                if tail < len(source) and source[tail] in ' \t\n':
                    quoted = title.match(source, tail)
                    if quoted and not re.search(r'\n[ \t]*\n', quoted[0]):
                        tail = quoted.end()
                ending = line_end.match(source, tail)
                if valid and ending:
                    pos = ending.end()
                    continue
        text.append(line)
        pos = end
    text = ''.join(text)
    # Match balanced delimiters so nested/escaped URL parentheses cannot leak
    # into the count. Keep link labels, but discard entire images (including alt).
    ends, stacks = {}, {'[': [], '(': []}
    escaped = False
    for i, char in enumerate(text):
        if escaped:
            escaped = False
        elif char == '\\':
            escaped = True
        elif char in stacks:
            stacks[char].append(i)
        elif char in (']', ')'):
            stack = stacks['[' if char == ']' else '(']
            if stack:
                ends[stack.pop()] = i
    parts, skip, i = [], {}, 0
    while i < len(text):
        if i in skip:
            i = skip[i]
            continue
        if text[i] == '\\':
            parts.append(text[i:i + 2]); i += 2
            continue
        image = text.startswith('![', i)
        opening = i + 1 if image else i
        closing = ends.get(opening) if text[opening] == '[' else None
        if closing is not None:
            after = closing + 1
            destination = ends.get(after) if after < len(text) and text[after] in '([' else None
            if image:
                i = (destination if destination is not None else closing) + 1
                continue
            if destination is not None:
                skip[closing] = destination + 1
        parts.append(text[i]); i += 1

    class VisibleText(HTMLParser):
        def __init__(self):
            super().__init__(); self.parts = []; self.hidden = []
        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            style = re.sub(r'/\*.*?\*/', '', d.get('style') or '', flags=re.S)
            display, important = '', False
            # Only top-level semicolons split declarations, including custom values.
            declarations, start, quote, blocks, escaped = [], 0, '', [], False
            for i, char in enumerate(style):
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif quote:
                    if char == quote:
                        quote = ''
                elif char in ('"', "'"):
                    quote = char
                elif char in '([{':
                    blocks.append({'(': ')', '[': ']', '{': '}'}[char])
                elif blocks and char == blocks[-1]:
                    blocks.pop()
                elif char == ';' and not blocks:
                    declarations.append(style[start:i]); start = i + 1
            declarations.append(style[start:])
            for declaration in declarations:
                match = re.fullmatch(r'\s*display\s*:\s*([a-z-]+(?:\s+[a-z-]+)*)\s*(!\s*important\s*)?', declaration, re.I)
                if match:
                    value, priority = ' '.join(match[1].lower().split()), bool(match[2])
                    words = value.split()
                    compound = (len(words) == 2
                                and sum(w in ('block', 'inline', 'run-in') for w in words) == 1
                                and sum(w in ('flow', 'flow-root', 'table', 'flex', 'grid', 'ruby') for w in words) == 1)
                    # Ignore invalid values, which cannot override display:none.
                    if compound or value in ('none', 'block', 'inline', 'run-in', 'flow', 'inline-block', 'flow-root',
                                 'flex', 'inline-flex', 'grid', 'inline-grid', 'contents',
                                 'table', 'inline-table', 'table-row', 'table-cell',
                                 'table-caption', 'table-row-group', 'table-header-group',
                                 'table-footer-group', 'table-column', 'table-column-group',
                                 'list-item', 'ruby', 'ruby-base', 'ruby-text',
                                 'ruby-base-container', 'ruby-text-container',
                                 'initial', 'inherit', 'unset', 'revert', 'revert-layer'):
                        if priority or not important:
                            display, important = value, priority
            hidden = ('hidden' in d or tag in ('script', 'style', 'template', 'head')
                      or display == 'none')
            # Track all descendants so closing a nested tag cannot expose the
            # rest of its hidden parent. Void elements never open a subtree.
            if (self.hidden or hidden) and tag not in (
                    'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                    'link', 'meta', 'param', 'source', 'track', 'wbr'):
                self.hidden.append(tag)
        def handle_endtag(self, tag):
            if tag in self.hidden:
                while self.hidden.pop() != tag:
                    pass
        def handle_data(self, data):
            if not self.hidden:
                self.parts.append(data)
    parser = VisibleText(); parser.feed(html.unescape(''.join(parts))); parser.close()
    return ''.join(parser.parts)


_lock = threading.Lock()
_stats = {"ok": 0, "skip": 0, "fail": 0}
_failures = []
_deferred = []
_fetch_attempted = set()
_fetch_calls = 0


class BatchBudgetExpired(BaseException):
    """Do not turn the batch deadline into an ordinary item retry."""


def official_album_evidence(article):
    """Bind the requested record to the complete, current part of the receipt."""
    from _album import _index_digest, _validated_records, _publisher_date
    try:
        records = json.loads(INDEX.read_text(encoding='utf-8'))
        proof = json.loads((INDEX.parent / 'health/album.json').read_text(encoding='utf-8'))
        _validated_records(records)
        current, retained = proof['current_count'], proof['retained_count']
        if (type(current) is not int or type(retained) is not int or current < 1 or retained < 0
                or current + retained != len(records)
                or proof['current_enumeration_complete'] is not True
                or proof['status'] != ('partial' if retained else 'success')
                or proof['index_sha256'] != _index_digest(records)):
            raise ValueError()
        missing = proof['missing_ids']
        if (not isinstance(missing, list) or len(missing) != retained
                or any(not isinstance(x, dict) or set(x) != {'msgid', 'itemidx'} for x in missing)):
            raise ValueError()
        missing_ids = {(str(x['msgid']), str(x['itemidx'])) for x in missing}
        if missing_ids != {(str(x['msgid']), str(x['itemidx'])) for x in records[current:]}:
            raise ValueError()
        fields = ('msgid', 'itemidx', 'pos', 'url', 'title', 'create_time', 'date')
        matches = [row for row in records[:current]
                   if all(article.get(k) == row[k] for k in fields)]
        if len(matches) != 1 or matches[0]['date'] != _publisher_date(matches[0]['create_time']):
            raise ValueError()
        return {'official_album_index_sha256': proof['index_sha256'],
                'create_time': matches[0]['create_time'],
                'date_source': 'official_album_create_time'}
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        raise ValueError('official_album_evidence_invalid') from None


@contextmanager
def fetch_deadline():
    """Keep the earlier batch alarm, or temporarily impose the 20-second wall."""
    # Daily serial POSIX runs also get a wall deadline; other callers retain
    # the HTTP timeout and bounded streaming checks below.
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[1]:
        raise RuntimeError('deepseek_fetch_periodic_alarm_requires_review')
    if 0 < previous_timer[0] <= FETCH_TIMEOUT:
        yield  # The existing batch wall is earlier and must keep its handler.
        return
    started = time.monotonic()
    def expired(*_):
        raise RuntimeError('deepseek_fetch_timeout')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, FETCH_TIMEOUT)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0]:
            remaining = previous_timer[0] - (time.monotonic() - started)
            if remaining <= 0:
                raise BatchBudgetExpired()
            signal.setitimer(signal.ITIMER_REAL, remaining)


def fetch_markdown(article):
    """One paid Fetch attempt, never Search, with independently bound metadata."""
    evidence = official_album_evidence(article)
    identity = (str(article['msgid']), str(article['itemidx']))
    global _fetch_calls
    with _lock:
        if identity in _fetch_attempted or _fetch_calls >= 20:
            raise RuntimeError('deepseek_fetch_budget_exhausted')
        _fetch_attempted.add(identity)
        _fetch_calls += 1
    try:
        with fetch_deadline():
            started = time.monotonic()
            from _deepseek import _key
            req = urllib.request.Request(FETCH_ENDPOINT,
                data=json.dumps({'url': article['url']}).encode('utf-8'),
                headers={'Content-Type': 'application/json', 'Accept': 'application/json'}, method='POST')
            # Do not forward credentials if an HTTP redirect is attempted.
            req.add_unredirected_header('Authorization', 'Bearer ' + _key())
            with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as response:
                if response.getcode() != 200:
                    raise ValueError('deepseek_fetch_status_invalid')
                # read1 avoids an unbounded trickle inside a single buffered read.
                if hasattr(response, 'read1'):
                    chunks, size = [], 0
                    while size <= FETCH_MAX_BYTES:
                        if response.fp is None:
                            break
                        remaining = FETCH_TIMEOUT - (time.monotonic() - started)
                        if remaining <= 0:
                            raise ValueError('deepseek_fetch_timeout')
                        response.fp.raw._sock.settimeout(remaining)
                        chunk = response.read1(min(65536, FETCH_MAX_BYTES + 1 - size))
                        if not chunk:
                            break
                        chunks.append(chunk)
                        size += len(chunk)
                    payload = b''.join(chunks)
                else:
                    payload = response.read(FETCH_MAX_BYTES + 1)
            if time.monotonic() - started > FETCH_TIMEOUT:
                raise ValueError('deepseek_fetch_timeout')
            if not payload or len(payload) > FETCH_MAX_BYTES:
                raise ValueError('deepseek_fetch_size_invalid')
            data = json.loads(payload.decode('utf-8'))
            if not isinstance(data, dict):
                raise ValueError('deepseek_fetch_json_invalid')
            # The endpoint extracts the one submitted URL; any returned URL must agree.
            if 'url' in data and data['url'] != article['url']:
                raise ValueError('deepseek_fetch_url_mismatch')
            title, body = data.get('title'), data.get('content')
            if not isinstance(title, str) or not isinstance(body, str):
                raise ValueError('deepseek_fetch_content_invalid')
            normalized = lambda s: re.sub(r'\W+', '', html.unescape(s)).casefold()
            if not normalized(title) or normalized(title) != normalized(article['title']):
                raise ValueError('deepseek_fetch_title_mismatch')
            body = body.strip()
            visible = re.sub(r'[\\_*`#]+', '', html.unescape(body)).casefold()
            if any(marker in visible for marker in (
                    'appmsgcaptcha', '环境异常', '请完成验证', '访问过于频繁', '安全验证',
                    'captcha', 'verify your identity', 'access denied', 'enable javascript')):
                raise ValueError('deepseek_fetch_challenge')
            if (len(body) < 300 or len(normalized(markdown_visible_text(body))) < 300
                    or re.search(r'<(?:!doctype|html|body)\b', body, re.I)):
                raise ValueError('deepseek_fetch_body_invalid')
            evidence.update(source_format='markdown', fetch_provider='deepseek_fetch',
                            body_sha256=hashlib.sha256(body.encode('utf-8')).hexdigest())
            # Recheck before saving, including concurrent changes to index/receipt.
            if official_album_evidence(article)['official_album_index_sha256'] != evidence['official_album_index_sha256']:
                raise ValueError('official_album_evidence_changed')
            return title, article['date'], body, evidence
    except urllib.error.HTTPError as error:
        raise RuntimeError('deepseek_fetch_http_' + str(error.code)) from None
    except Exception:
        # Neither provider bodies, key-loader errors nor URLs belong in the log.
        raise RuntimeError('deepseek_fetch_failed') from None


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
        from _album import _article_url
        _article_url(a['url'])
        try:
            raw = fetch(a["url"], expected_title=a['title'], expected_date=a['date'])
        except ArticleUnavailable:
            title, date, body, provenance = fetch_markdown(a)
        else:
            title, date, body = extract(raw)
            provenance = {'source_html_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest()}
        if len(body) < 300 or not date or date != a["date"]:
            raise ValueError("article_provenance_invalid")
        if re.sub(r"\W+", "", title).casefold() != re.sub(r"\W+", "", a["title"]).casefold():
            raise ValueError("article_title_mismatch")
        rec = {"pos": a["pos"], "msgid": a["msgid"], "itemidx": a["itemidx"],
               "title": title or a["title"], "wechat_title": a["title"],
               "date": date or a["date"], "url": a["url"],
               "chars": len(body), "body": body,
               "source_kind": "fulltext_quantml",
               **provenance,
               "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        from _health import atomic
        atomic(out, rec)
        with _lock:
            _stats["ok"] += 1
            print(f"  OK pos={a['pos']:>3} {rec['chars']:>5}c  {a['title'][:34]}", flush=True)
    except Exception as e:
        reason = (str(e) if isinstance(e, (RuntimeError, ValueError))
                  and re.fullmatch(r'[a-z][a-z0-9_]*(?::[a-z0-9_,]+)?', str(e))
                  else type(e).__name__)
        with _lock:
            _stats["fail"] += 1
            _failures.append({'msgid': a['msgid'], 'itemidx': a['itemidx'],
                              'pos': a['pos'], 'reason': reason})
            print(f"  FAIL pos={a['pos']:>3} {a['title'][:30]} :: {reason}", file=sys.stderr, flush=True)
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
