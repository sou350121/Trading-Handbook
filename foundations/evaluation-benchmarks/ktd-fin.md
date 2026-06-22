---
title: "KTD-FIN"
description: "落點於「生成式大模型 × 因子挖掘 × Agent自主演进」軸，解決 LLM 交易評測中預訓練記憶洩漏與風格暴露混淆選股能力的 prior gap，將評測標準從「淨值神話」強制拉回「殘差歸因」。"
---
<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=生成式大模型 alpha=因子挖掘 autonomy=Agent自主演进 -->

> **發布**：2026-06-18 · （無 venue）
> **QuantML 導讀**：[清华 X 阶跃星辰 ｜ 剥离风格和记忆，LLM还剩下多少Alpha？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247494094&idx=1&sn=f04a150a17f3b98c4d7b4a8fc22ae31f&chksm=ce7d8ed0f90a07c69e5e37ecb5129b007cc84ae3cfd3214dd13f8ff576a3734a20494d6d9284#rd)
> **核心定位**：落點於「生成式大模型 × 因子挖掘 × Agent自主演进」軸，解決 LLM 交易評測中預訓練記憶洩漏與風格暴露混淆選股能力的 prior gap，將評測標準從「淨值神話」強制拉回「殘差歸因」。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `生成式大模型` | `因子挖掘` | `Agent自主演进` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 建立首個帶物理級數據掩碼與 Barra 截面歸因的 LLM 交易 Agent 基準。② 核心 trick 為四階段掩碼協議切斷記憶通道，並用 WLS 回歸將收益拆解為市場β、風格與選股殘差。③ 對「Agent自主演进」軸而言，它廢止了 Prompt 工程驱动的淨值幻覺，強制模型在匿名化因子表格上進行結構化推理。④ 實證顯示 10 款主流 LLM 中 9 款選股 Alpha 為負，淨值收益實為風格暴露。

**X-Ray.** KTD-FIN 將 LLM 評測從「淨值對標」推向「歸因解構」的 Pareto 前沿。它填補了生成式 Agent 在量化回測中缺乏記憶隔離與風格剝離的基礎設施空白，直接證偽了「高累積收益率=高 Alpha」的學術慣性。然而，其 envelope 仍受限於 A 股日頻波段與單一模態，未觸及多模態新聞/訂單簿的高頻博弈。對量化讀者而言，它不是模型，而是一把「照妖鏡」：任何未通過 Blinded × Open-research 測試的 LLM 交易論文，其收益曲線皆應預設為風格漂移的產物，直至證明殘差 Alpha 的穩健性。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作 (無掩碼 LLM 框架) | KTD-FIN |
|---|---|---|
| 數據輸入 | 真實代碼/日期/新聞文本 | 四階段掩碼 (Blinded 模式) |
| 決策依賴 | 預訓練記憶與語義推理 | 結構化因子表格 + 6 只讀工具 |
| 評測指標 | 累積收益率 / 淨值曲線 | Barra 三層歸因 (Common / Style / Selection α) |

**1.2 ⚡ Eureka 一句話 trick**
用「物理掩碼+相對時間偏移」強制 LLM 放棄文本記憶，僅依賴截面因子表格進行權重分配。

**1.3 信息流 ASCII 圖**
```
[Blinded Data] → (Day_-3 / asset_NNNN)
       ↓
[LLM Agent] → ReAct Loop → {get_market_context, screen_candidates, get_stock_snapshot, compare_candidates, portfolio_state, risk_check}
       ↓
[Qlib Engine] → T+1 / ±9.5%~29.5% / 5bps-15bps 費率
       ↓
[Barra WLS] → 截面回歸 → 殘差 α (Selection Alpha)
```

## §2 · 數學層
📌 **Napkin Formula**
$$r_{i,t} = \alpha_t + \sum_{k=1}^{9} \beta_{i,t-1}^k f_{k,t} + \epsilon_{i,t}$$
**複雜度**：$O(N \cdot K)$ 每日截面 WLS 回歸（$N$ 為可投股票數，$K=9$ 因子）。
**直覺**：截距 $\alpha_t$ 捕捉市場公共收益，$\beta f$ 剝離風格暴露，殘差 $\epsilon$ 即為剔除所有被動載荷後的選股 Alpha。
**Loss/訓練細節**：屬評測框架而非端到端訓練模型，無傳統 DL loss；依賴 Qlib 引擎模擬執行與 6 個只讀工具調用，模型參數凍結或僅作推理。

## §3 · 數據層
**規模/頻率/市場/時段**：A 股全市場，日頻，2024 年 1 月 1 日至 2026 年 4 月 10 日（共 548 個交易日）。
**來源與處理**：Qlib 引擎計算 9 個低互相關經典因子（MOM_12_1, RV_60, ILLIQ 等），截面去極值與標準化。
**樣本外與容量假設**：單一樣本外長窗口測試；起步資金 100 萬人民幣，未披露機構級容量壓力測試與滑點模型。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 高（需 Qlib 環境+LLM API+Barra 因子庫） | 中（需 A 股日頻量價與 Barra 因子，導讀未公開原始數據包） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| A股 (2024-2026) | 總收益率 (Total ret.) | CSI300 +36.92% | qwen3.6-plus +85.29% | +48.37% |

**解讀**：Δ 為 +48.37%，但 Barra 歸因顯示此 Δ 完全由 +41.8% 的大盤公共收益與 +29.2% 的風格暴露驅動，選股 Alpha 實為 -0.7%。該 Δ 屬「風格運氣」而非真 capability。若計入 5 bps 買入/15 bps 賣出費率與 T+1 鎖倉，頻繁調倉的 LLM 策略淨值將進一步侵蝕，導讀未披露成本調整後數據。傳統 ML 基線（如 SFM 總收益 +86.58%，回撤 -7.41%）在風格剝離後展現更強魯棒性，凸顯 LLM 在截面選股上的結構性劣勢。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：僅覆蓋 A 股日頻波段；未測試多模態/訂單簿數據；Barra 9 因子可能未完全捕捉所有風格風險；攻擊者去匿名化成功率雖低（top-1 3.0%），但極端行情下量價特徵仍可能洩漏身份。
**6.2 推斷的隱含假設**：Regime 依賴強（2024 至 2026 高波動投機風格利好 LLM 偏好）；容量假設為散户型（100 萬起步），未驗證機構級滑點；數據泄漏風險已通過掩碼大幅降低，但因子計算邏輯若與預訓練語料重疊仍存隱患。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 ML 因子模型 | 歸因剝離 vs 淨值對標 | TBD | v0.5 |
| 無掩碼 LLM 交易框架 | 記憶隔離 vs 文本依賴 | TBD | v0.5 |

🎤 **Interview Tip**
**正確答**：「LLM 交易收益需經 Barra 截面歸因拆解，剔除 β 與風格暴露後，殘差 Alpha 才是泛化能力的唯一指標。未通過 Blinded 測試的淨值皆應視為風格漂移。」
**錯答**：「LLM 憑藉強大的上下文理解與歷史記憶，能直接預測股價走勢並實現超額收益。」

**7.1 可證偽預測帶日期**：若 2026 年 Q3 市場風格切換至低波價值，當前高暴露 MOM_ID 與 RV_60 的 LLM 策略選股 Alpha 將進一步惡化，導讀未給具體閾值，但邏輯上其殘差收益無法維持正區間。

## §8 · For the Reader
* **因子研究員**：將 Barra 殘差作為 LLM 因子挖掘的終極驗收標準，拒絕淨值神話；優先驗證因子正交性與截面穩定性。
* **組合配置**：LLM Agent 目前適合作為風格輪動或 β 增強工具，不可直接替代傳統截面選股模型；需嚴格監控風格暴露集中度。
* **LLM-Agent 開發者**：優先實現四階段掩碼協議與工具調用隔離，將評測重心從 Prompt 工程轉向因子結構化推理與風險預檢。

## References
* 原論文：From Knowing to Doing: A Memory-Controlled Benchmark for LLM Trading Agents on Stock Markets
* QuantML 導讀：[清华 X 阶跃星辰 ｜ 剥离风格和记忆，LLM还剩下多少Alpha？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247494094&idx=1&sn=f04a150a17f3b98c4d7b4a8fc22ae31f&chksm=ce7d8ed0f90a07c69e5e37ecb5129b007cc84ae3cfd3214dd13f8ff576a3734a20494d6d9284#rd)
* Lineage: Barra 截面歸因框架 / Qlib 量化引擎 / ReAct Agent 架構