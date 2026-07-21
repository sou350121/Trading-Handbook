# QWEN_USAGE.md — Opus 主導下「如何使用 qwen」的操作合約

> 原則由 Peter 於 2026-06-21 設定：**智能要求不高的大量任務交給 qwen；放 qwen 上規模之前，先由 Opus 4.8 ultrathink 把怎麼用 qwen 定死。** 本文件就是那份契約。改 prompt / schema / 路由前先改本文件。

## 0. 分工總則

| 角色 | 定位 | 負責 |
|---|---|---|
| **Opus（我）** | 架構師 / 大腦 | 本體論、prompt 模板、JSON schema、品質閘、跨篇綜合（Crossing）、誤分類裁決、docs.json/audit/腳手架 |
| **qwen**（qwen3.5-plus, DashScope CodingPlan, thinking on） | 主力勞動 | 399× 的逐篇蒸餾、分類打標、來源論文抽取、評級、套模板成 markdown、zone overview 初稿 |

鐵律：**Opus 先想清楚並凍結合約 → qwen 在合約內無人值守批量跑 → Opus 抽樣校準、裁決長尾。** qwen 永遠不准自創 zone slug；超出注入清單的一律歸 `unsorted` 交 Opus。

## 1. 任務路由表

| 任務 | Owner | 理由 |
|---|---|---|
| 本體論 / 五軸 / zone 劃分 | Opus（qwen 出草稿） | 手冊靈魂，高判斷 |
| prompt 模板 + JSON schema | Opus | 定義契約本身 |
| 逐篇蒸餾（399×） | **qwen** | 對乾淨輸入做摘要/抽取 |
| 逐篇分類 → zone + 五軸座標 | **qwen** | 在固定本體內打標 |
| 來源論文抽取（venue/arXiv/框架名） | **qwen** | 抽取 |
| 評級 ⚡/🔧/📖/❌ | **qwen** | rubric 驅動 |
| dissection markdown（被選中的篇） | **qwen** | 模板化寫作 |
| zone overview 綜述 | qwen 初稿 → Opus 改 | 輕綜合 |
| Crossing 張力圖（跨篇本質衝突） | **Opus** | 真綜合，非摘要 |
| 去重裁決 / 誤分類修正 | **Opus** | 判斷 |
| audit / docs.json / 腳手架 | **Opus** | 架構 |

## 2. 兩段式 qwen 管線

### Pass A — Router + Metadata（全 399，便宜，temp 0.1，JSON-out）
- **輸入**：`title` + `body`(截 ≤7000 字) + 注入的 taxonomy（zone slug 清單 + 五軸 poles）
- **輸出 schema**（嚴格 JSON）：
```json
{
  "zone": "<taxonomy 裡的 slug，否則 unsorted>",
  "axes": {"data":"...","horizon":"...","paradigm":"...","alpha":"...","autonomy":"..."},
  "rating": "⚡|🔧|📖|❌",
  "tldr": "≤60 字一句話",
  "key_trick": "≤80 字核心 trick",
  "source": {"venue":"ICLR26|KDD|JFE|null","arxiv":"NNNN.NNNNN|null","framework":"名字|null","confidence":"high|mid|low"},
  "tags": ["≤5 個"],
  "dup_of": "重複的 pos 或 null",
  "page_worthy": true
}
```
- **Opus 閘**：跑完先抽 20 篇人工校準 prompt → 凍結 → 再跑全 399。

### Pass B — Dissection 全頁（只給 `page_worthy && rating∈{⚡,🔧} && dup_of==null`，估 ~120-180 篇；其餘變成 zone overview 裡的 registry 一行）
- **輸入**：`body` + Pass A metadata + dissection 模板（量化版：§架構 / §數學 napkin / §數據 / §代碼 / §評測 / §失效與隱含假設 / §對比 / §For Reader）
- **輸出**：依模板的完整 markdown。temp 0.4。
- **不寫**：paper abstract 摘要、主觀讚美。要的是 **對比 + 失效模式 + 如何組合**（沿用 sister handbook AGENTS 反 anti-pattern 規則）。

## 3. 品質閘（機械，零 LLM）
- JSON schema 合法；`zone` ∈ taxonomy slugs；`axes` 各值 ∈ 允許 poles。
- body floor：<300 字標 `thin`，不進 Pass B。
- 去重：同 zone 內 title Jaccard ≥0.6 → flag 給 Opus 裁決（不自動刪）。
- commit 前過 `scripts/handbook_audit.py`（斷鏈/孤兒頁/nav 一致性）。

## 4. qwen 調用紀律
- 走 `_qwen.py`（DashScope coding endpoint，BAILIAN key，thinking on）。
- temp：分類 0.1 / 寫作 0.4。retry 4×（429/5xx，backoff 5*(i+1)s）。
- 並發 ≤4（host 3.5GB RAM 無 swap，序列或小並發）。
- 全程**可續傳**：每篇輸出落 `data/distill/<msgid>.json`，重跑跳過已完成。

## 5. Opus 留在環裡的「縫」
1. 定稿本體論（已：審 qwen 草稿 → `data/taxonomy.json`）。
2. Pass A 跑前抽 20 樣校準。
3. 定 Pass B 入選政策（page_worthy 標準）。
4. 手寫 Crossing atlases + cheat-sheet 五軸總圖。
5. 跑 audit、定 docs.json、修長尾。

## 6. 多源情報層的 qwen 偵察角色（2026-07-22 增）
- **頂刊雷達分類**（`journal_radar.py`）：12 本頂刊新文 → {relevance, zone, rating, 一句話}；temp 0.1、thinking off、json_mode、signal.alarm(85) 硬牆。分類失敗不記 seen → 下週自動重試。
- **GitHub 工程蒸餾**（`gh_quant_issues.py`）：每庫一批高互動 issues → 「坑→解法」JSON；**只依據 issue 原文，數字須逐字在文本中**；thinking on、alarm(170)。
- **正典一句話**（`build_seminal_canon.py`）：開山論文各 ≤40 字「開創了什麼」；引用計數一律 Crossref 機械統計，qwen 不碰數字。
- **OA 一手解構**：復用 Pass A/B（同 §2-4 紀律），origin=oa 由管線機械蓋章，qwen 無權決定來源標籤。
