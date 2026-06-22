#!/usr/bin/env python3
"""Idempotent: add the 五軸座標 fingerprint table to any dissection page missing it.

GitHub-primary (no frontmatter). Used by the daily cron on freshly generated pages.
Skip rule keys on the fingerprint marker itself, so re-running never churns existing pages.

Run: python3 scripts/add_fingerprint.py [--dry-run]
"""
import glob, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
AXES = [("data", "數據模態"), ("horizon", "時間尺度"), ("paradigm", "學習範式"),
        ("alpha", "Alpha機制"), ("autonomy", "人機協作")]


def fingerprint(kv):
    head = "| " + " | ".join(l for _, l in AXES) + " |"
    sep = "|" + "|".join([":-:"] * len(AXES)) + "|"
    row = "| " + " | ".join("`%s`" % kv.get(k, "—") for k, _ in AXES) + " |"
    return "**五軸座標**\n\n" + "\n".join([head, sep, row]) + "\n"


def fix(text):
    if "五軸座標" in text:
        return None
    m = re.search(r'<!--\s*ontology-5axis\s+(.*?)-->', text)
    if not m:
        return None
    kv = dict(re.findall(r'(\w+)=([^\s]+)', m.group(1)))
    fp = "\n" + fingerprint(kv)
    bq = re.search(r'((?:^>.*\n)+)', text, re.M)   # header blockquote block
    if bq:
        return text[:bq.end()] + fp + text[bq.end():]
    return None


def main():
    dry = "--dry-run" in sys.argv
    done = skip = 0
    for f in sorted(glob.glob(str(ROOT / "foundations/*/*.md"))):
        if f.endswith("overview.md"):
            continue
        t = pathlib.Path(f).read_text(encoding="utf-8")
        new = fix(t)
        if new is None:
            skip += 1
            continue
        if dry:
            print("WOULD ADD fingerprint:", f)
        else:
            pathlib.Path(f).write_text(new, encoding="utf-8")
        done += 1
    print("fingerprinted=%d skipped(has/none)=%d" % (done, skip))


if __name__ == "__main__":
    main()
