#!/usr/bin/env python3
"""Scaffold Trading-Handbook skeleton + docs.json + ontology cheat-sheet from taxonomy.json.
Opus-authored architecture; mirrors VLA/Spatial/Physics handbook structure. Idempotent."""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
TAX = json.loads((ROOT/"data"/"taxonomy.json").read_text())
ONT = TAX["ontology"]; ZONES = TAX["foundations_zones"]; CROSS = TAX["crossing_atlases"]

def w(rel, text):
    p = ROOT/rel; p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():           # never clobber hand-edited content
        p.write_text(text)

# ---- cheat-sheet: the 5-axis ontology (soul doc) ----
axrows = "\n".join(f"| {i+1} | **{a['name']}** | {a['poles']} | {a.get('what_it_discriminates','')} |"
                   for i,a in enumerate(ONT["axes"]))
zrows = "\n".join(f"| [`{z['slug']}`](/foundations/{z['slug']}/overview) | {z['name']} | {z.get('scope','')} | ~{z.get('est_articles','?')} |"
                  for z in ZONES)
w("cheat-sheet/ontology.md", f"""# {ONT['name']}

> 手冊的靈魂。任意兩個量化方法都能投到這五軸上做 Pareto 對比。
> **設計理由**：{ONT.get('rationale','')}

## 五軸

| # | 軸 | 兩端 | 它在語料裡區分了什麼 |
|---|---|---|---|
{axrows}

## Foundations 方法族 → 軸的落點

| Zone | 名稱 | 收什麼 | 容量估計 |
|---|---|---|---|
{zrows}

> 容量估計是設計時的配額，不是實際數；實際歸類以 Pass A（`data/distill/`）為準。
""")

w("cheat-sheet/overview.md", f"""# Cheat Sheet

- [五軸本體論](/cheat-sheet/ontology) — {ONT['name']}
- [時間線](/cheat-sheet/timeline) — QuantML 兩年語料的演進

本手冊（照見四冊之四 · Trading 端）把 QuantML 公眾號 2024-06 ~ 2026-06 的 399 篇論文導讀，
按方法族解構、按五軸對比。讀者畫像：做過實盤/回測的量化研究員，要的是對比、失效模式、如何組合。
""")
w("cheat-sheet/timeline.md", "# 時間線\n\n> 自動生成中（待 Pass A 完成後由 gen 腳本填充）。\n")

# ---- foundations zones ----
ax_legend = " · ".join(a["name"] for a in ONT["axes"])
for z in ZONES:
    w(f"foundations/{z['slug']}/overview.md", f"""# {z['name']}

> **收錄**：{z.get('scope','')}
> **五軸**：{ax_legend}

## 解構清單

> 待 Pass A/B 填充。⚡/🔧 評級的方法寫成整頁解構；📖 邊際/綜述降級為下方 registry 一行。

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| _（pending）_ | | | |
""")
w("foundations/overview.md", "# Foundations\n\n按方法族組織的解構區。\n\n" +
  "\n".join(f"- [{z['name']}](/foundations/{z['slug']}/overview)" for z in ZONES) + "\n")

# ---- crossing atlases ----
for c in CROSS:
    w(f"crossing/{c['slug']}/overview.md", f"""# {c['name']}

> **本質衝突**：{c.get('tension','')}

跨 zone 張力圖（Opus 手寫綜合，非摘要）。待 Pass A 完成後，由 Opus 從各 zone 抽代表方法填這張對比。
""")
w("crossing/overview.md", "# Crossing — 跨方法族張力圖\n\n" +
  "\n".join(f"- [{c['name']}](/crossing/{c['slug']}/overview)" for c in CROSS) + "\n")

# ---- other tabs ----
for slug,name in [("benchmarks","評測基準"),("deployment","落地與失效"),
                  ("companies","機構與團隊"),("use-cases","應用場景"),("frontier","前沿")]:
    w(f"{slug}/overview.md", f"# {name}\n\n> 待填充。\n")
w("reports/overview.md", "# Reports\n\n`reports/trading-daily/` 為 Pulsar Phase 1 自動產出區（每日 arxiv q-fin + ML→qwen 評級）。\n")
w("bridge-to-vla/overview.md", "# Bridge to VLA / 姊妹手冊\n\n量化里的 RL/Agent/世界模型與 VLA/Physics 手冊的接口。\n")

# ---- top-level ----
w("ONBOARDING.md", """# Onboarding

照見系列第四冊 · **Trading-Handbook（量化交易手冊）**。語料源：QuantML 公眾號（2024-06~2026-06，399 篇論文導讀）。

- 先讀 [五軸本體論](/cheat-sheet/ontology)。
- 方法解構在 [Foundations](/foundations/overview)，按方法族。
- 跨方法族對比在 [Crossing](/crossing/overview)。
- AI agent 編輯規範見 [AGENTS.md](/AGENTS.md)；qwen 使用合約見 `scripts/pulsar/QWEN_USAGE.md`。
""")
w("CONTRIBUTING.md", "# Contributing\n\n沿用照見系列協定。新增解構走 `foundations/<zone>/<method>.md`，遵循 AGENTS.md 模板。\n")
w("MAINTAINER.md", "# Maintainer\n\n維護者：Pulsar / 照見。語料抓取見 `scripts/pulsar/`（enumerate→fetch→distill）。\n")
w("LICENSE", "CC-BY-4.0. 解構為二手分析，版權歸原論文與 QuantML 公眾號所有。\n")
w(".gitignore", "data/raw/\ndata/distill/\n__pycache__/\n*.pyc\n.DS_Store\n")

# ---- docs.json (Mintlify) ----
def grp(group, pages): return {"group": group, "pages": pages}
docs = {
 "$schema":"https://mintlify.com/docs.json","theme":"mint",
 "name":"Trading Handbook",
 "description":f"ML-for-Quant 方法解構 · {ONT['name']} · 照見四冊之四（Trading 端）· 語料 QuantML 399 篇",
 "colors":{"primary":"#0E7C66","light":"#34D399","dark":"#065F46"},
 "favicon":"/favicon.svg",
 "navigation":{"tabs":[
   {"tab":"Get Started","icon":"rocket","groups":[grp("Start here",["ONBOARDING.md","AGENTS.md","CONTRIBUTING.md","MAINTAINER.md"])]},
   {"tab":"Cheat Sheet","icon":"map","groups":[grp("Overview",["cheat-sheet/overview.md","cheat-sheet/ontology.md","cheat-sheet/timeline.md"])]},
   {"tab":"Foundations","icon":"layer-group","groups":[grp("Overview",["foundations/overview.md"])]+[grp(z["name"],[f"foundations/{z['slug']}/overview.md"]) for z in ZONES]},
   {"tab":"Crossing","icon":"shuffle","groups":[grp("Atlases",["crossing/overview.md"]+[f"crossing/{c['slug']}/overview.md" for c in CROSS])]},
   {"tab":"Benchmarks","icon":"gauge","groups":[grp("Overview",["benchmarks/overview.md"])]},
   {"tab":"Deployment","icon":"server","groups":[grp("Overview",["deployment/overview.md"])]},
   {"tab":"Use Cases","icon":"briefcase","groups":[grp("Overview",["use-cases/overview.md"])]},
   {"tab":"Companies","icon":"building","groups":[grp("Overview",["companies/overview.md"])]},
   {"tab":"Frontier","icon":"telescope","groups":[grp("Overview",["frontier/overview.md"])]},
   {"tab":"Reports","icon":"newspaper","groups":[grp("Overview",["reports/overview.md"])]},
 ]}}
(ROOT/"docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2))
print("scaffold done. tree:")
