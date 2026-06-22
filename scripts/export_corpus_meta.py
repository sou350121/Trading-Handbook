#!/usr/bin/env python3
"""Export a copyright-safe corpus manifest from data/distill -> data/corpus_meta.json.

Ships in the repo (the slug/classification mapping is the cron's critical state, otherwise
only in gitignored data/distill -> lost if the host dies). Contains ONLY our own labels and
factual metadata (zone/rating/slug/dup_of/framework/title) -- NO QuantML article body text.

Run: python3 scripts/export_corpus_meta.py
"""
import json, glob, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "corpus_meta.json"


def slug(d):
    if d.get("slug_override"):
        return d["slug_override"]
    fw = (d.get("source", {}) or {}).get("framework")
    b = fw if fw and fw.lower() not in ("null", "none", "") else d["title"]
    s = re.sub(r'[^a-zA-Z0-9]+', '-', (b or "")).strip('-').lower()
    return s or "art-%s" % d["pos"]


def main():
    rows = []
    for f in glob.glob(str(ROOT / "data" / "distill" / "*.json")):
        d = json.load(open(f))
        src = d.get("source", {}) or {}
        rows.append({
            "pos": d.get("pos"),
            "msgid": d.get("msgid"),
            "title": d.get("title"),
            "zone": d.get("zone"),
            "rating": d.get("rating"),
            "page_worthy": bool(d.get("page_worthy")),
            "slug": slug(d),
            "dup_of": d.get("dup_of"),
            "framework": src.get("framework"),
            "venue": src.get("venue"),
            "arxiv": src.get("arxiv"),
            "axes": d.get("axes", {}),
        })
    rows.sort(key=lambda r: (r["pos"] is None, r["pos"]))
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    pw = sum(1 for r in rows if r["page_worthy"] and r["rating"] in ("⚡", "🔧") and not r["dup_of"])
    print("wrote %d records (%d page-worthy non-dup) -> %s" % (len(rows), pw, OUT))


if __name__ == "__main__":
    main()
