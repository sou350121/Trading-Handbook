#!/usr/bin/env python3
"""Regenerate each foundations/<zone>/overview.md with a 解構清單 table from
Pass A distill data. Page-worthy ⚡/🔧 with a generated page -> link; others -> registry line.
Mechanical, Opus-owned (no qwen). Overwrites overview.md (generated artifact)."""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent/"pulsar"))
from distill_pass_b import slugify  # shared slug logic

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAX = json.loads((ROOT/"data"/"taxonomy.json").read_text())
ONT = TAX["ontology"]
ZONES = {z["slug"]: z for z in TAX["foundations_zones"]}
AX_LEGEND = " · ".join(a["name"] for a in ONT["axes"])
RANK = {"⚡":0,"🔧":1,"📖":2,"❌":3}

def src_str(d):
    s = d.get("source",{}) or {}
    bits = []
    if s.get("venue") and s["venue"] not in ("null","None"): bits.append(s["venue"])
    ax = s.get("arxiv")
    if ax and ax not in ("null","None"): bits.append(f"[arXiv]({'https://arxiv.org/abs/'+ax})")
    fw = s.get("framework")
    if fw and fw not in ("null","None") and not bits: bits.append(fw)
    return " ".join(bits) or "—"

def main():
    dis = [json.loads(f.read_text()) for f in (ROOT/"data"/"distill").glob("*.json")]
    by = {}
    for d in dis: by.setdefault(d.get("zone","unsorted"), []).append(d)
    for slug, z in ZONES.items():
        items = by.get(slug, [])
        items.sort(key=lambda d:(RANK.get(d.get("rating"),9), -d.get("pos",0)))
        n_page = sum(1 for d in items if d.get("page_worthy") and d.get("rating") in ("⚡","🔧"))
        rows = []
        for d in items:
            slugn = slugify(d)
            page = ROOT/"foundations"/slug/f"{slugn}.md"
            name = (d.get("source",{}) or {}).get("framework") or d["title"][:30]
            label = f"[{name}]({slugn})" if page.exists() else name
            rows.append(f"| {label} | {d.get('rating','')} | {src_str(d)} | {d.get('tldr','')[:48]} |")
        table = ("\n".join(rows) if rows else "| _（待 Pass A 填充）_ | | | |")
        md = f"""# {z['name']}

> **收錄**：{z.get('scope','')}
> **五軸**：{AX_LEGEND}
> **本 zone 現有**：{len(items)} 篇（{n_page} 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
{table}

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
"""
        (ROOT/"foundations"/slug/"overview.md").write_text(md)
    # foundations index with counts
    lines = [f"- [{z['name']}](/foundations/{slug}/overview) — {len(by.get(slug,[]))} 篇"
             for slug,z in ZONES.items()]
    (ROOT/"foundations"/"overview.md").write_text("# Foundations\n\n按方法族組織的解構區。\n\n"+"\n".join(lines)+"\n")
    # report
    print("zone counts:")
    for slug,z in ZONES.items():
        its=by.get(slug,[]); pw=sum(1 for d in its if d.get('page_worthy') and d.get('rating') in('⚡','🔧'))
        print(f"  {slug:>26}: {len(its):>3}  (page-worthy {pw})")
    uns=by.get("unsorted",[])
    if uns: print(f"  {'unsorted':>26}: {len(uns)}")

if __name__=="__main__":
    main()
