#!/usr/bin/env python3
"""Enforce unique (zone, slug) across page-worthy non-dup records.

Slug collisions (generic Chinese titles -> same ascii slug) make Pass B silently drop the
2nd page (skip-existing). This scans for collisions and disambiguates: the lowest-pos record
keeps the slug; the others get slug_override="<slug>-<pos>". Run Pass B afterwards to generate
the newly-named pages. Idempotent.

Run: python3 scripts/pulsar/fix_collisions.py [--dry-run]
"""
import glob, json, re, sys, pathlib, collections

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIS = ROOT / "data" / "distill"


def slugify(d):
    if d.get("slug_override"):
        return d["slug_override"]
    fw = (d.get("source", {}) or {}).get("framework")
    base = fw if fw and fw.lower() not in ("null", "none", "") else d["title"]
    s = re.sub(r'[^a-zA-Z0-9]+', '-', (base or "")).strip('-').lower()
    return s or "art-%s" % d["pos"]


def main():
    dry = "--dry-run" in sys.argv
    recs = []
    for f in glob.glob(str(DIS / "*.json")):
        d = json.load(open(f))
        if d.get("page_worthy") and d.get("rating") in ("⚡", "🔧") and not d.get("dup_of"):
            recs.append((f, d))
    groups = collections.defaultdict(list)
    for f, d in recs:
        groups[(d["zone"], slugify(d))].append((f, d))
    changed = 0
    for (zone, slug), members in sorted(groups.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda fd: fd[1]["pos"])   # lowest pos keeps the slug
        keep = members[0]
        print("COLLISION %s/%s.md (%d records); keep pos%s" %
              (zone, slug, len(members), keep[1]["pos"]))
        for f, d in members[1:]:
            d["slug_override"] = "%s-%s" % (slug, d["pos"])
            print("   -> pos%s slug_override=%s" % (d["pos"], d["slug_override"]))
            if not dry:
                json.dump(d, open(f, "w"), ensure_ascii=False, indent=2)
            changed += 1
    print("disambiguated=%d" % changed)
    return changed


if __name__ == "__main__":
    main()
