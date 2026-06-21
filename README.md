<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# Trading-Handbook · 量化交易手冊

**照見（Pulsar）系列第四冊 — Decision / Trading 端**

把 ML-for-Quant 的方法按「方法族」解構、按**五軸本體論**對比：對比 · 失效模式 · 如何組合。

</div>

---

## 這是什麼

一本給**做過實盤或嚴肅回測**的量化研究員 / ML 工程師的方法解構手冊。不科普「什麼是 Transformer / 強化學習 / 因子」，直接進「這個方法跟另一個在五軸上差在哪、哪些 claim 在真實市場會崩、怎麼跟另一條線組合」。

語料源：**QuantML 公眾號**「論文」合集（2024-06 ~ 2026-06，399 篇論文導讀）作為**選題雷達與發現源**。每一頁解構是我們**自己對公開論文（arXiv / KDD / ICML / JFE 等）的二階分析**，QuantML 作為策展來源致謝並回鏈；不轉載其正文。

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

## Foundations — 10 個方法族 zone

`time-series-forecasting` · `graph-networks` · `reinforcement-learning` · `llm-agentic` · `market-microstructure` · `factor-mining` · `portfolio-optimization` · `causal-structural` · `evaluation-benchmarks` · `data-generation-augmentation`

> zone 按**方法族**切；跨族資訊由五軸座標 + tags 承載，在 [Crossing](crossing/overview.md) 張力圖撈出（預測 vs 策略驅動、可解釋因子 vs 端到端黑盒、大模型 vs 專用時序基座 …）。

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
