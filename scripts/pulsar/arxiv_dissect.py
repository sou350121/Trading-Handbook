#!/usr/bin/env python3
"""#1  arXiv ⚡ -> 一手全文自主解構 pipeline.

Reads the hot (⚡) items the weekly radar appended to data/radar_queue.json, fetches each
paper's FULL TEXT (arxiv.org/html -> ar5iv -> abs-abstract fallback), ingests it as a QuantML
raw record (data/raw/arxiv_<id>_1.json), then REUSES the existing pipeline unchanged:
  Pass A (qwen classify -> zone/5-axis/rating) -> patch source.origin=arxiv -> Pass B (writes a
  first-hand dissection page; header + body-length are origin-aware in distill_pass_b.py).

This is the leverage move: QuantML gives second-hand digests (+~3/day); this gives first-hand,
primary-source dissections straight off arXiv -- the same intake shape that lets VLA-Handbook grow.

Downstream (fingerprint / mermaid / overviews / nav / resolver / gate / commit) is run by the
caller (radar_weekly.sh) so this script stays a pure content-producer and is trivially resumable:
raw existence skips re-fetch, distill existence skips Pass A, page existence skips Pass B.

Usage:
  python3.11 arxiv_dissect.py [--limit N] [--ids 2607.15195,2607.13929] [--ingest-only] [--workers W]
"""
import json, re, sys, html, time, pathlib, argparse, urllib.request
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import distill_pass_a as PA
import distill_pass_b as PB

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
DIS = ROOT / "data" / "distill"
QUEUE = ROOT / "data" / "radar_queue.json"
POS_BASE = 100000  # arXiv pos band, far above QuantML (~419) -> never collides with corpus pos
UA = {"User-Agent": "Mozilla/5.0 (Trading-Handbook research bot)"}


class _Text(HTMLParser):
    """Generic HTML -> readable text (arXiv/ar5iv have no js_content div, so we can't reuse
    fetch_article's WeChat extractor). Drops script/style/nav chrome, keeps paragraph breaks."""
    SKIP = {"script", "style", "head", "nav", "footer", "noscript"}

    def __init__(self):
        super().__init__(); self.parts = []; self.skip = 0

    def handle_starttag(self, t, a):
        if t in self.SKIP:
            self.skip += 1
        elif t in ("p", "br", "div", "section", "li", "h1", "h2", "h3", "h4", "tr", "table"):
            self.parts.append("\n")

    def handle_endtag(self, t):
        if t in self.SKIP and self.skip > 0:
            self.skip -= 1

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.parts.append(d)


# arXiv / ar5iv page chrome that leaks into the text extract -- strip so qwen sees only the paper.
_BOILER = [
    r"Skip to main content", r"Press Enter to search[^\n]*", r"Advanced search",
    r"Search arXiv", r"Report GitHub Issue.?", r"Content selection saved[^\n]*",
    r"arXiv is now an independent nonprofit.*?Learn more.?", r"Back to arXiv",
    r"View a PDF of the paper titled.*?View PDF", r"Bibliographic.*?Tools", r"Bibliographic Explorer.*",
    r"License:\s*arXiv\.org perpetual[^\n]*", r"These authors contributed equally\.",
    r"\d+footnotemark:\s*\d+", r"Learn more.?",
]
_BOILER_RE = re.compile("|".join(_BOILER), re.S)


def html_to_text(h):
    p = _Text(); p.feed(h)
    s = html.unescape("".join(p.parts))
    s = _BOILER_RE.sub("", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def _fetch(url, timeout=30, retries=3):
    """(bytes, final_url). ar5iv is flaky under load -> retry with backoff. final_url matters:
    ar5iv redirects to arxiv.org/abs/<id> when no HTML render exists yet (paper too new)."""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            r = urllib.request.urlopen(req, timeout=timeout)
            return r.read(), r.geturl()
        except Exception as e:
            last = e
            if i < retries - 1:
                time.sleep(2.5 * (i + 1))
    raise last


def _pdf_text(pdf_bytes):
    """PDF -> reading-order text via poppler's pdftotext (primary-source path for HTML-less papers)."""
    import subprocess
    p = subprocess.run(["pdftotext", "-q", "-nopgbrk", "-", "-"],
                       input=pdf_bytes, capture_output=True, timeout=150)
    txt = p.stdout.decode("utf-8", "replace")
    txt = _BOILER_RE.sub("", txt)
    txt = re.sub(r"-\n(?=[a-z])", "", txt)       # de-hyphenate line-wrapped words
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


def fetch_fulltext(item):
    """(body, kind). kind in {fulltext-html, fulltext-pdf, abstract-only, none}.
    Order: HTML mirror (best reading text) -> PDF (reliable for HTML-less new papers) -> abstract."""
    tid = item["id"]
    vurl = item.get("abs_url", "")
    m = re.search(r"/(\d{4}\.\d{4,5}(?:v\d+)?)$", vurl)
    vid = m.group(1) if m else tid
    best = ""
    for u in (f"https://ar5iv.org/html/{tid}", f"https://arxiv.org/html/{vid}",
              f"https://arxiv.org/html/{tid}", f"https://ar5iv.labs.arxiv.org/html/{tid}"):
        try:
            data, final = _fetch(u)
            if "/abs/" in final:        # ar5iv bounced to the abstract page -> no HTML render, skip
                continue
            txt = html_to_text(data.decode("utf-8", "replace"))
            if len(txt) > len(best):
                best = txt
            if len(best) > 8000:
                break
        except Exception:
            pass
    if len(best) > 3000:
        return best, "fulltext-html"
    # PDF fallback -- first-hand full text when no HTML exists (common for brand-new q-fin papers)
    try:
        data, _ = _fetch(f"https://arxiv.org/pdf/{tid}", timeout=60)
        if data[:5] == b"%PDF-":
            txt = _pdf_text(data)
            if len(txt) > 2500:
                return txt, "fulltext-pdf"
    except Exception:
        pass
    # abstract-only last resort
    try:
        h, _ = _fetch(f"https://arxiv.org/abs/{tid}")
        h = h.decode("utf-8", "replace")
        m = re.search(r'<blockquote class="abstract[^"]*">(.*?)</blockquote>', h, re.S)
        ab = html_to_text(m.group(1)) if m else ""
        if len(ab) > 120:
            return "（僅取得摘要，正文未抓到）\n\n" + ab, "abstract-only"
    except Exception:
        pass
    return "", "none"


def ingest(item, seq):
    """Write data/raw/arxiv_<id>_1.json in QuantML raw schema. Returns (msgid, status)."""
    tid = item["id"]
    msgid = f"arxiv_{tid}"
    out = RAW / f"{msgid}_1.json"
    if out.exists() and len(json.loads(out.read_text()).get("body", "")) > 800:
        return msgid, "cached"
    body, kind = fetch_fulltext(item)
    if kind == "none" or len(body) < 400:
        return msgid, "fetch-fail"
    rec = {"pos": POS_BASE + seq, "msgid": msgid, "itemidx": "1",
           "title": item["title"], "wechat_title": item["title"],
           "date": item.get("date", ""),
           "url": item.get("abs_url") or f"https://arxiv.org/abs/{tid}",
           "chars": len(body), "body": body}
    RAW.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return msgid, kind


def patch_origin(msgid, tid):
    """Pass A rewrites `source` from qwen output (no origin) -> stamp it back so Pass B renders the
    arXiv-origin header/body-length. Returns the distill dict, or None if Pass A produced none."""
    f = DIS / f"{msgid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    src = d.setdefault("source", {})
    src["origin"] = "arxiv"
    src["arxiv"] = tid
    f.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="max papers per run (0 = all queued)")
    ap.add_argument("--ids", type=str, default="", help="comma arXiv ids; overrides queue")
    ap.add_argument("--ingest-only", action="store_true", help="fetch+raw only, skip qwen")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    if args.ids:
        wanted = [i.strip() for i in args.ids.split(",") if i.strip()]
        q = json.load(open(QUEUE)) if QUEUE.exists() else []
        by_id = {it["id"]: it for it in q}
        items = [by_id.get(i, {"id": i, "title": i, "abs_url": f"https://arxiv.org/abs/{i}", "date": ""})
                 for i in wanted]
    else:
        q = json.load(open(QUEUE)) if QUEUE.exists() else []
        # skip ones already turned into a distill record (resume); newest first
        items = [it for it in q if not (DIS / f"arxiv_{it['id']}.json").exists()]
        items.sort(key=lambda it: it.get("date", ""), reverse=True)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print("arxiv_dissect: nothing to do (queue drained)"); return

    print(f"arxiv_dissect: {len(items)} paper(s)", flush=True)
    ingested = []  # (item, msgid)
    for seq, it in enumerate(items):
        msgid, st = ingest(it, seq)
        print(f"  ingest {it['id']:>14} [{st}]  {it['title'][:52]}", flush=True)
        if st.startswith("fulltext") or st in ("abstract-only", "cached"):
            ingested.append((it, msgid))
    if args.ingest_only or not ingested:
        print(f"ingested {len(ingested)} raw record(s); qwen skipped"); return

    # Pass A (classify) on just the new arXiv raws -- do_one skips any that already have a distill.
    print("Pass A (classify) ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(PA.do_one, RAW / f"{mid}_1.json") for _, mid in ingested]
        for fu in as_completed(futs):
            fu.result()

    # Stamp origin=arxiv, then Pass B on page-worthy ⚡/🔧 (mirror the daily gate; junk stays out).
    pages = []
    for it, mid in ingested:
        d = patch_origin(mid, it["id"])
        if d and d.get("page_worthy") and d.get("rating") in ("⚡", "🔧"):
            pages.append(d)
        else:
            r = (d or {}).get("rating", "?")
            print(f"  gate: {it['id']} rating={r} page_worthy={(d or {}).get('page_worthy')} -> no page", flush=True)
    print(f"Pass B (dissect) on {len(pages)} page(s) ...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(PB.do_one, d) for d in pages]
        for fu in as_completed(futs):
            fu.result()
    print(f"arxiv_dissect done: ingested={len(ingested)} pages_attempted={len(pages)} "
          f"pageB_ok={PB._n['ok']} pageB_skip={PB._n['skip']} pageB_fail={PB._n['fail']}", flush=True)


if __name__ == "__main__":
    main()
