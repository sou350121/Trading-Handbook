<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# Trading-Handbook · 量化交易手冊

**照見（Pulsar）系列第四冊 — Decision / Trading 端**

把 ML-for-Quant 的方法按「方法族」解構、按**五軸本體論**對比：對比 · 失效模式 · 如何組合。

</div>

---

## 這是什麼

一本給**做過實盤或嚴肅回測**的量化研究員 / ML 工程師的方法解構手冊。不科普「什麼是 Transformer / 強化學習 / 因子」，直接進「這個方法跟另一個在五軸上差在哪、哪些 claim 在真實市場會崩、怎麼跟另一條線組合」。

語料源：**QuantML 公眾號**「論文」合集（2024-06 起，每日增量）作為**選題雷達與發現源**。每一頁解構是我們**自己對公開論文（arXiv / KDD / ICML / JFE 等）的二階分析**，多數頁面已回鏈到 `原始論文`（arXiv/DOI + venue + 被引數，經 Crossref 核驗）；QuantML 作為策展來源致謝並回鏈，不轉載其正文。

**多源情報層（[Radar](radar/overview.md)）**：除 QuantML 外，每週自動掃描更多來源並依五軸/十區分類——[arXiv q-fin 雷達](radar/arxiv/overview.md)（q-fin.* + cs.LG/stat.ML 量化新論文）與[实践者雷达](radar/practitioner/overview.md)（Quantocracy / Alpha Architect / AQR / Quantpedia / CXO 等從業者博客）。雷達只出「標題＋鏈接＋我們的一句話機制解讀」，⚡ 標記候選深度解構。

## 五軸本體論（手冊的靈魂）

任意兩個量化方法都能投到這五軸上做 Pareto 對比：

| 軸 | 兩端 |
|---|---|
| **數據模態** | 量價/表格 ↔ 另類/文本/圖結構/微觀盤口 |
| **時間尺度** | 高頻/日內 ↔ 日頻/波段 ↔ 宏觀/長週期 |
| **學習範式** | 監督回歸 ↔ 強化學習 ↔ 生成式/大模型 ↔ 因果/結構 |
| **Alpha 生成機制** | 顯式因子挖掘 ↔ 端到端表徵 ↔ 組合/執行優化 ↔ 多智能體博弈 |
| **人機協作度** | 全自動黑盒 ↔ 人機協同/可解釋 ↔ Agent/自主演進 |

詳見 [`cheat-sheet/ontology.md`](cheat-sheet/ontology.md)。

## Foundations — 10 個方法族 zone（351 頁，點進去逐頁解構）

| Zone 方法族 | 頁數 | 焦點 |
|---|--:|---|
| [時序預測](foundations/time-series-forecasting/overview.md) `time-series-forecasting` | 87 | 預測基座/基礎模型 |
| [因子挖掘](foundations/factor-mining/overview.md) `factor-mining` | 60 | Alpha 因子/擁擠度 |
| [LLM 智能體](foundations/llm-agentic/overview.md) `llm-agentic` | 48 | 大模型/多智能體交易 |
| [強化學習](foundations/reinforcement-learning/overview.md) `reinforcement-learning` | 39 | 執行/做市/組合 RL |
| [組合優化](foundations/portfolio-optimization/overview.md) `portfolio-optimization` | 33 | 配置/端到端優化 |
| [市場微結構](foundations/market-microstructure/overview.md) `market-microstructure` | 26 | 訂單流/LOB/日內 |
| [圖網絡](foundations/graph-networks/overview.md) `graph-networks` | 24 | 關係圖/動態圖 |
| [評測基準](foundations/evaluation-benchmarks/overview.md) `evaluation-benchmarks` | 19 | 基準/打假/穩健性 |
| [因果結構](foundations/causal-structural/overview.md) `causal-structural` | 9 | 機制/regime/定價 |
| [數據生成](foundations/data-generation-augmentation/overview.md) `data-generation-augmentation` | 6 | 合成/增強/模擬 |

> zone 按**方法族**切；跨族資訊由五軸座標 + tags 承載，在 [Crossing 張力圖](crossing/overview.md) 撈出（預測 vs 策略驅動、可解釋因子 vs 端到端黑盒、大模型 vs 專用時序基座 …）。時間維度見 [兩年演進 timeline](cheat-sheet/timeline.md)。

## 怎麼建的（Pulsar 管線 · Opus/qwen 分工）

```
enumerate_album → fetch_article(allorigins 代理) → distill_pass_a(qwen 分類) → distill_pass_b(qwen 寫頁) → Opus 綜合/audit
```

- **Opus = 架構/本體論/裁決/Crossing 綜合**；**qwen = 399× 蒸餾/分類/套模板**。完整契約見 [`scripts/pulsar/QWEN_USAGE.md`](scripts/pulsar/QWEN_USAGE.md)。
- 抓來的原文正文（版權屬原作者 / QuantML）**不入庫**（`data/raw`、`data/distill` 已 gitignore）；庫內只有解構頁、骨架、`index.json`、`taxonomy.json`（公開元數據）。

## 狀態

**v0.5 · 建設中。** 骨架 + 五軸本體 + 管線就緒；399 篇解構頁正分批生成。每頁標 `Status: v0.5`，未驗證細節標 `TBD`（這是設計特性，不假裝確定）。

## 姊妹手冊（照見四冊）

| 冊 | 視角 |
|---|---|
| [VLA-Handbook](https://github.com/sou350121/VLA-Handbook) | Action 端 |
| [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook) | Perception 端 |
| [Physics-Controllable-Generation-Handbook](https://github.com/sou350121/Physics-Controllable-Generation-Handbook) | Generation 端 |
| **Trading-Handbook**（本冊） | **Decision / Trading 端** |

## 致謝 & License

- **選題與發現源**：[QuantML 公眾號](https://mp.weixin.qq.com/) — 兩年 399 篇論文導讀的策展。
- 解構為對公開論文的二手分析，CC-BY-4.0；版權歸原論文與 QuantML 所有。
