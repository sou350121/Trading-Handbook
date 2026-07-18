<!-- markdownlint-disable MD033 MD041 -->
<div align="center">

# Trading-Handbook · 量化交易手冊

**照見（Pulsar）系列 · Decision / Trading 端**

給做過實盤的人的一張「量化 ML 方法活地圖」——不科普概念，只回答<br>
**這個方法跟另一個差在哪 · 哪些 claim 在真實市場會崩 · 誰在持續付錢**

`351 頁解構` · `10 方法族` · `6 張張力圖` · `每日自動增長`

</div>

---

## 這是什麼

不是教科書、不是論文摘要，是一張**可導航的方法地圖**。每一頁把一個量化 ML 方法拆成「架構 → 數學 → 數據 → 評測 → **失效模式** → 怎麼跟別的線組合」，再投到**五軸本體論**上跟其他方法做 Pareto 對比。多數頁已回鏈 `原始論文`（arXiv/DOI＋被引數，Crossref 核）。

> [!TIP]
> **想在裡面找靈感？三條路——**
> 1. 要**策略角度** → 往下的 **⚔️ 六個策略問題**：每個都是一場「誰持續付錢」的博弈
> 2. 查**某類方法** → **📂 10 區地圖**，每區 `⚡` = 最值得先讀
> 3. 追**前沿** → **📡 雷達**，每週自動掃 arXiv ＋ 從業者博客

## ⚔️ 六個策略問題（張力圖 · 靈感從這裡來）

每張張力圖不是「介紹兩類方法」，而是逼你回答一個真實的設計選擇——並點出**誰因為結構性理由持續付錢**。

| 張力 | 它逼你回答的問題 |
|---|---|
| [預測驅動 vs 策略驅動](crossing/supervised-vs-rl/overview.md) | Alpha 該長在**信號**裡（監督預測）還是**策略**裡（RL）？交易成本內化在哪？ |
| [可解釋因子 vs 端到端黑盒](crossing/explicit-factors-vs-e2e/overview.md) | 因子要**看得懂、能歸因**，還是**榨乾非線性**？擁擠與沉默退化怎麼防？ |
| [大模型邏輯 vs 專用時序基座](crossing/llm-reasoning-vs-ts-baselines/overview.md) | 讀**敘事**還是算**數值**？LLM 的幻覺與延遲關在哪一層？ |
| [靜態關聯 vs 動態微觀流](crossing/static-graph-vs-dynamic-flow/overview.md) | 用**慢的關係圖**還是**快的訂單流**？圖定權重、流定時機的接縫在哪？ |
| [兩步法 vs 聯合優化](crossing/predict-then-optimize-vs-end-to-end/overview.md) | 先預測再優化（模組可單測）還是端到端（全局最優但梯度塌縮）？ |
| [人機協同 vs 全自動智能體](crossing/human-in-loop-vs-autonomous-agent/overview.md) | co-pilot 可問責 vs 全自治更快——護欄放哪、誰扛責？ |

> 全部入口與代表方法見 [Crossing 總覽](crossing/overview.md)。

## 五軸本體論（手冊的靈魂）

任意兩個方法都能投到這五軸上做對比——這是全書的座標系：

| 軸 | 兩端 |
|---|---|
| **數據模態** | 量價/表格 ↔ 另類/文本/圖/微觀盤口 |
| **時間尺度** | 高頻/日內 ↔ 日頻/波段 ↔ 宏觀/長週期 |
| **學習範式** | 監督回歸 ↔ 強化學習 ↔ 生成式/大模型 ↔ 因果/結構 |
| **Alpha 機制** | 顯式因子 ↔ 端到端表徵 ↔ 組合/執行優化 ↔ 多智能體博弈 |
| **人機協作** | 全自動黑盒 ↔ 人機協同/可解釋 ↔ Agent/自主演進 |

詳見 [五軸速查](cheat-sheet/ontology.md) · 時間維度看 [兩年演進 timeline](cheat-sheet/timeline.md)。

## 📂 10 方法族地圖（351 頁）

| 方法族 Zone | 頁數 | 焦點 |
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

> zone 按**方法族**切；跨族的比較由五軸 + tags 承載，在上面的張力圖撈出。

## 📡 雷達（多源前瞻情報）

除 QuantML 策展外，每週自動掃更多來源、依十區分類，只出「標題＋鏈接＋我們的一句話機制解讀」，`⚡` = 候選深度解構：

- [arXiv q-fin 雷達](radar/arxiv/overview.md) — q-fin.* ＋ cs.LG/stat.ML 量化新論文
- [實踐者雷達](radar/practitioner/overview.md) — Quantocracy / Alpha Architect / AQR / Quantpedia / CXO 等從業者博客

## 狀態（誠實）

**v0.5 · 每日自動增長。** 351 頁已生成、質量閘門全綠（0 斷鏈/孤頁/版權洩漏）。內容是對公開論文的**二階分析**（多數已回鏈原文、約 1/4 帶被引數）——是一張**強地圖**，不是逐頁核過原文的權威參考；未驗證細節標 `TBD`（設計特性，不假裝確定）。

<details>
<summary><b>怎麼建的 · 版權</b></summary>

```
enumerate_album → fetch(allorigins) → distill_pass_a(qwen 分類) → distill_pass_b(qwen 寫頁) → Opus 綜合/audit
```

- **Opus** 定架構/本體論/裁決/Crossing 綜合；**qwen** 跑蒸餾/分類/套模板。契約見 [`QWEN_USAGE.md`](scripts/pulsar/QWEN_USAGE.md)。
- 抓來的原文正文（版權屬原作者 / QuantML）**不入庫**（`data/raw`、`data/distill` 已 gitignore）；庫內只有解構頁與公開元數據。每日 cron 自動增量、自愈、推送。
</details>

## 姊妹手冊（照見四冊）

| 冊 | 視角 |
|---|---|
| [VLA-Handbook](https://github.com/sou350121/VLA-Handbook) | Action 端 |
| [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook) | Perception 端 |
| [Physics-Controllable-Generation-Handbook](https://github.com/sou350121/Physics-Controllable-Generation-Handbook) | Generation 端 |
| **Trading-Handbook**（本冊） | **Decision / Trading 端** |

## 致謝 & License

選題與發現源：[QuantML 公眾號](https://mp.weixin.qq.com/) 的策展。解構為對公開論文的二手分析，CC-BY-4.0；版權歸原論文與 QuantML 所有。
