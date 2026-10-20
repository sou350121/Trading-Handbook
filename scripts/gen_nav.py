#!/usr/bin/env python3
"""Rebuild docs.json Foundations tab to include all generated dissection pages.
Mechanical, Opus-owned."""
import json, pathlib
from pathlib import Path
ROOT = pathlib.Path(__file__).resolve().parents[1]
TAX = json.loads((ROOT/"data"/"taxonomy.json").read_text())
ZONES = TAX["foundations_zones"]
docs = json.loads((ROOT/"docs.json").read_text())

def zone_pages(slug):
    d = ROOT/"foundations"/slug
    pages = [f"foundations/{slug}/overview.md"]
    for f in sorted(d.glob("*.md")):
        if f.name != "overview.md":
            pages.append(f"foundations/{slug}/{f.name}")
    return pages

for tab in docs["navigation"]["tabs"]:
    if tab["tab"] == "Foundations":
        groups = [{"group":"Overview","pages":["foundations/overview.md"]}]
        for z in ZONES:
            groups.append({"group": z["name"], "pages": zone_pages(z["slug"])})
        tab["groups"] = groups
(ROOT/"docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
npages = sum(len(zone_pages(z["slug"]))-1 for z in ZONES)
print(f"docs.json nav rebuilt: {npages} dissection pages across {len(ZONES)} zones")

# Keep all generated radar weeks discoverable in the navigation manifest.
for tab in docs["navigation"]["tabs"]:
    for group in tab.get("groups", []):
        pages = group.get("pages", [])
        roots = {str(Path(p).parent).replace("\\", "/") for p in pages if isinstance(p, str) and p.startswith("radar/")}
        for root in roots:
            for page in sorted((ROOT / root).glob("*-W*.md"), reverse=True):
                rel = page.relative_to(ROOT).as_posix()
                if rel not in pages and rel.removesuffix(".md") not in pages:
                    pages.append(rel)
(ROOT/"docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
