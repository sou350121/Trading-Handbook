# AGENTS.md — 給 AI agents（Pulsar / Opus / qwen）的編輯指南

照見系列第四冊 · **Trading-Handbook（量化交易手冊）**。姊妹倉：VLA-Handbook · Spatial-Intelligence-Handbook · Physics-Generation-Handbook。四倉共用同一套寫作協定。

**分工**：架構/裁決/綜合由 Opus 主導，399 篇的蒸餾/分類/套模板由 qwen 批量執行。完整契約見 [`scripts/pulsar/QWEN_USAGE.md`](scripts/pulsar/QWEN_USAGE.md)。

## 寫作對象（Reader Persona）

不是學生、不是審稿人。是一位**做過實盤或嚴肅回測**的量化研究員 / ML 工程師。他想知道：

- 這個方法跟我已知的另一個方法在**五軸**上差在哪？
- 哪些 paper claim 在真實市場會崩？什麼工況崩（換 regime、加成本、樣本外、容量）？
- 我要怎麼**組合**它跟另一條線（factor mining × RL、TS 基座 × LLM agent、microstructure × portfolio opt）？

不需要科普什麼是 Transformer / 強化學習 / 因子。直接進「對比 / 取捨 / 失效模式」。

## 五軸本體論（手冊的靈魂）

任意方法都投到這五軸上：**數據模態 · 時間尺度 · 學習範式 · Alpha生成機制 · 人機協作度**。完整定義見 [`cheat-sheet/ontology.md`](cheat-sheet/ontology.md)。**zone 按方法族切；跨族資訊由五軸座標 + tags 承載，在 Crossing 撈出。**

## Dissection 寫作模板（量化版）

每篇解構（`foundations/<zone>/<method>.md`）開頭加 ontology 註解，再依下列骨架：

```
<!-- ontology-5axis data=.. horizon=.. paradigm=.. alpha=.. autonomy=.. -->

# <Method Name> 解構（<English>）

> **發布**：YYYY-MM · <venue> · arXiv [NNNN.NNNNN](url) · [code](url)
> **QuantML 導讀**：[標題](原文url)（YYYY-MM-DD）
> **核心定位**：1-2 句說它在五軸落點 + 解了什麼 prior gap。

**Status:** v0.5 — 解構基於 QuantML 導讀 + 原論文摘要 +（若有）代碼/issues。完整 benchmark 數字待升 v1。
**TL;DR:** ≤4 句：① 做什麼新事 ② 核心 trick ③ 為什麼這對某條軸★ ④ 一個關鍵實證數字（IR/Sharpe 提升、回撤、容量、成本敏感度）。

**X-Ray.**（150-300 字，分析性論斷，不是 summary）把它放回五軸 Pareto、標它解了哪些舊工程坑、預測它打不開的 envelope（什麼會超出論文）、對量化讀者的意義。

## §1 · 架構 / Core Mechanism
1.1 三大改動 vs 前作（表）｜1.2 ⚡ Eureka（一句話 trick + 直覺）｜1.3 信息流 ASCII 圖

## §2 · 數學層
📌 Napkin Formula（1-3 行最重要 equation + 複雜度 O(..) vs prior）+ 直覺 2-3 句；loss / 訓練細節。

## §3 · 數據層
資料規模/頻率/市場/時段、怎麼來的、樣本外與容量假設。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前 SOTA | 本方法 | Δ |
解讀：**哪部分 Δ 是真 capability，哪部分是 過擬合 / 前瞻偏差 / 成本未計 / benchmark Goodhart**。

## §6 · 失效與隱含假設
6.1 論文自述 limitations｜6.2 我們推斷的隱含假設（regime 依賴、容量、成本、數據泄漏、survivorship）｜6.x（若有）代碼/issue 驗證的失敗模式。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
🎤 Interview Tip：典型問題的正確答 vs 錯答。
7.1 可證偽預測（帶日期）。

## §8 · For the Reader（按 persona 分流，至少 3 條）
因子研究員 / 高頻執行 / 組合配置 / LLM-agent 工程 / RL 策略 / 研究學生。

## References + Boundary + 維護者升 v1 清單
```

## 不寫的東西（anti-pattern）

- 不複製 paper / 導讀的摘要 —— 手冊價值在 **對比 + 失效 + 組合**，不在 summary。
- 不寫「我覺得很 cool」—— 對比一定落到五軸 / 失效模式 / 成本。
- 不假裝確定 —— TBD / UNVERIFIED 是設計特性，留給維護者升 v1。
- 不把回測曲線當結論 —— 永遠追問成本、容量、樣本外、regime。

## 與三個 sister handbook 的分工

| Handbook | 視角 |
|---|---|
| VLA-Handbook | Action 端 |
| Spatial-Intelligence-Handbook | Perception 端 |
| Physics-Generation-Handbook | Generation 端 |
| **本倉** | **Decision / Trading 端 — 從市場觀測輸出倉位/因子/執行** |

跨倉引用走 `bridge-to-*/`（量化的 RL/Agent/世界模型 ↔ VLA/Physics）。

## Pulsar pipeline 接口

`reports/trading-daily/` 是 Pulsar Phase 1 自動產出區：arxiv **q-fin.* + cs.LG/cs.AI**（交易/因子/組合關鍵詞）→ qwen3.5-plus 評 ⚡/🔧/📖/❌ → git push（Mintlify rebuild）。歷史語料回填見 `scripts/pulsar/`（`enumerate_album` → `fetch_article` → `distill_pass_a` → Pass B）。
