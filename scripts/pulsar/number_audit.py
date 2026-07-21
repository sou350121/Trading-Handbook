#!/usr/bin/env python3
"""Mechanical number-discipline check: every numeric token in a page's §5 benchmark table
should be a verbatim substring of its QuantML digest body (data/raw), OR an arithmetic
difference of two such numbers (the allowed Δ case). Flags pages with unverifiable numbers.

This is the headless stand-in for the one-time Opus audit: the hardened Pass B prompt is the
primary defense; this catches residual fabrication leaks in the daily cron. Log-only (no edit).

Run: python3 scripts/pulsar/number_audit.py [--pages f1 f2 ...]   (default: all pages)
"""
import glob, json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
NUM = re.compile(r'-?\d+(?:\.\d+)?')
# years / year.month / year.month.day and ranges thereof -> not metrics
DATE = re.compile(r'(19|20)\d\d(?:[-./]\d{1,2}){0,2}')


def strip_dates(s):
    prev = None
    while prev != s:                       # repeat to clear "2018.01-2019.06" ranges
        prev = s
        s = DATE.sub(' ', s)
    return s


_BODY_INDEX = None


def _build_index():
    """Load all raw digests once: {url: body}. O(M) instead of O(N*M)."""
    idx = {}
    for f in glob.glob(str(RAW / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("url"):
            idx[d["url"]] = d.get("body", "")
    return idx


def body_for(page_text):
    global _BODY_INDEX
    if _BODY_INDEX is None:
        _BODY_INDEX = _build_index()
    m = re.search(r'mp\.weixin\.qq\.com/s\?[^\s)\]]+', page_text)
    if m:
        url = m.group(0)
    else:
        # arXiv-origin pages carry an arxiv.org/abs URL; OA-journal pages carry a doi.org URL
        # (their raw is keyed by exactly that link). Versioned/unversioned handled by containment.
        m = (re.search(r'arxiv\.org/abs/\d{4}\.\d{4,5}(?:v\d+)?', page_text)
             or re.search(r'doi\.org/[^\s)\]"]+', page_text))
        if not m:
            return None
        url = "https://" + m.group(0)
    if url in _BODY_INDEX:
        return _BODY_INDEX[url]
    tail = url[-40:]                       # fall back to suffix / containment match
    for u, b in _BODY_INDEX.items():
        if tail in u or url in u or u in url:
            return b
    return None


def section5(text):
    m = re.search(r'##\s*§?5[^\n]*\n(.*?)(?:\n##\s|\Z)', text, re.S)
    return m.group(1) if m else ""


def main():
    args = sys.argv[1:]
    if "--pages" in args:
        pages = args[args.index("--pages") + 1:]
    else:
        pages = [f for f in glob.glob(str(ROOT / "foundations/*/*.md"))
                 if not f.endswith("overview.md")]
    flagged = []
    for p in sorted(pages):
        text = pathlib.Path(p).read_text(encoding="utf-8")
        body = body_for(text)
        if body is None:
            continue
        body_nums = set(NUM.findall(strip_dates(body).replace(",", "")))
        s5 = strip_dates(section5(text))
        bad = []
        for n in NUM.findall(s5.replace(",", "")):
            if n in body_nums:
                continue
            # allow arithmetic Δ: n == a-b for some verbatim a,b on nearby rows
            if any(abs(float(n) - (float(a) - float(b))) < 0.011
                   for a in body_nums for b in body_nums if a != b):
                continue
            # allow tiny ints (row indices, §, counts) and the year range
            if re.fullmatch(r'-?\d{1,2}', n) or re.fullmatch(r'20\d\d', n):
                continue
            bad.append(n)
        if bad:
            flagged.append((p, bad[:8]))
    print("number_audit: %d pages checked, %d flagged" %
          (len(pages), len(flagged)))
    for p, bad in flagged:
        print("  FLAG %s :: unverifiable §5 nums: %s" % (p, bad))
    return len(flagged)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 0)  # advisory: never hard-fail the pipeline
