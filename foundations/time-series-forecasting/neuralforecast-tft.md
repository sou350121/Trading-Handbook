---
title: "NeuralForecast/TFT"
description: "聚焦日頻波段的風險擇時，以監督回歸範式處理量價表格。解決了傳統深度學習以 MSE/RMSE 為驗證指標時，與實際交易風險收益結構脫節的 prior gap。"
---
<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=人机协同可解释 -->

> **發布**：2024-11-03 · （無 venue）
> **QuantML 導讀**：[增强深度学习模型评估以进行股市预测--Part I](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487443&idx=1&sn=c6c555e9b90e263f3ed083ba4166ed44&chksm=ce7e68cdf909e1db7adde95df86e8bcd92e30642b492d3f9e4de4267d29ab9027f0de83abcd5#rd)
> **核心定位**：聚焦日頻波段的風險擇時，以監督回歸範式處理量價表格。解決了傳統深度學習以 MSE/RMSE 為驗證指標時，與實際交易風險收益結構脫節的 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ①將不可微的商業交易指標（如 Sharpe/Sortino）直接嵌入 TFT 驗證循環。②核心 trick 是利用分位數損失輸出概率分佈，並透過置信閾值過濾低質量信號。③這對「風險擇時」軸★至關重要，因為它強制模型優化目標與實際交易的風險調整回報對齊，而非單純最小化點預測誤差。④導讀未給量化結果。

**X-Ray.** 本文本質是「驗證指標工程化」的示範，而非新架構發明。在量價表格與日頻波段的 Pareto 前沿上，它切中了深度學習實盤中最致命的工程坑：訓練損失（可微）與實盤評估（不可微）的目標錯配。透過將 Sharpe 等指標硬編碼為 `custom validation metric`，模型被迫在驗證集上直接優化風險調整收益，這比事後對 MSE 預測結果做線性映射或閾值截斷更符合交易直覺。然而，其 envelope 受限於單變量 SPY 與極簡特徵工程，未處理滑點/佣金/流動性衝擊，且分位數閾值與最小交易次數的設定帶有強主觀性。對量化讀者而言，價值不在於直接 copy-paste 策略，而在於提供了一套可複用的 `NeuralForecast` 回調架構，將「交易邏輯」前置到模型收斂階段，為後續接入 RL 或組合優化層鋪平了指標對齊的路徑。

## §1 · 架構 / Core Mechanism
| 維度 | 傳統 TFT / 時序模型 | 本文改動 | 工程意義 |
|---|---|---|---|
| 驗證指標 | MSE / RMSE / MAE | Sharpe / Sortino (Custom) | 消除訓練目標與實盤風險調整收益的錯配 |
| 輸出形態 | 點預測 (Point Forecast) | 多分位數概率分佈 (Quantile) | 支援機率推斷與動態閾值過濾 |
| 信號過濾 | 靜態閾值或無 | 置信水平閾值 + 最小交易次數 | 避免低質量信號頻繁觸發，降低偽正例 |

⚡ **Eureka:** 用分位數損失的機率分佈替代點預測，直接對齊不可微的商業驗證指標。
```text
[Exogenous/History] → TFT Encoder → Quantile Decoder → P(Return > 0)
          ↓                                      ↓
    [Custom Metric: Sharpe] ← [Threshold Filter (Conf=0.6)] ← [Quantiles]
          ↓
[EarlyStopping / ModelCheckpoint]
```

## §2 · 數學層
📌 **Napkin Formula:**
`L = HuberMQLoss(y, ŷ_q)` for `q ∈ [0.05,0.4,0.5,0.6,0.95]`
複雜度: `O(T * hidden_size * n_head)` per step，驗證階段需 `valid_batch_size=2000` 一次性計算全局交易指標。
**直覺:** 多分位數回歸強制模型學習條件分佈而非均值，使 `P(R_t > 0)` 可計算。驗證階段將分位數推斷轉化為頭寸決策，並計算 Sharpe/Sortino 作為 `monitor` 指標觸發 `EarlyStopping`。
**Loss/訓練細節:** 使用 `HuberMQLoss` 抗異常值，`learning_rate:0.0005`，`gradient_clip val:1`，`batch size:32`。驗證集一次性過 batch 以計算全局交易指標。

## §3 · 數據層
- **資料規模/頻率/市場/時段:** SPY 日頻回報，`2005-07-01` 至 `2024-05-23`。
- **怎麼來:** Yahoo Finance (SPY, GC=F, MSFT, GOOGL, AAPL, AMZN) + FRED (VIX, T5YIE, T10YIE, T10Y3M, DGS10, DGS2, DTB3, DEXUSNZ, T10Y2Y, NASDAQCOM, DCOILWTICO)。
- **樣本外與容量假設:** `train test split:[0.9,0.10]`，`val proportion size:0.15`。未披露具體樣本量與容量限制，假設為單資產/低頻容量，未計滑點與佣金。`max missing data:0.02`，`min nb trades:60`。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | `NeuralForecast` 框架內建 TFT |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 低（YAML 配置完整，CPU 可跑） |
| 數據可得性 | 高（FRED/Yahoo 公開 API） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| SPY 日頻 | IR / Sharpe / AR / MDD | 未披露 | 未披露 | 未披露 |
**解讀:** 導讀未給量化結果，無法計算 Δ。此階段屬方法論驗證，未涵蓋交易成本與實盤滑點，Sharpe 提升可能僅反映驗證指標對齊的理論優勢，實盤有效性待回測檢驗。分位數閾值 `0.6` 與 `min nb trades:60` 的設定帶有強主觀性，未進行超參優化或交叉驗證，當前 Δ 僅代表「指標對齊」的架構可行性，非策略 Alpha。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 省略佣金/滑點，特徵工程極簡，無超參優化，單變量目標，驗證指標計算需一次性過全量 batch 導致擴展性受限。
**6.2 推斷的隱含假設:** 市場微結構穩定（無流動性衝擊），分位數閾值 `0.6` 與 `min nb trades:60` 為靜態先驗，數據無前瞻偏差（嚴格按日切分），容量假設極低（僅 SPY），`max missing data:0.02` 假設外生變量缺失為隨機且可前值填充。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 LSTM/GRU 回歸 | 驗證指標對齊方式 (MSE vs Custom Trading Metric) | Open | 基線 |
| RL 執行層 (PPO/SAC) | 優化階段 (訓練期嵌入 vs 推理期策略搜索) | Open | 互補 |
| 因子模型 + 閾值截斷 | 信號生成邏輯 (概率分佈推斷 vs 線性加權) | Open | 替代 |
🎤 **Interview Tip:**
- ✅ 正確答：「深度學習的可微訓練損失與不可微的實盤交易指標存在目標錯配。本文透過分位數損失輸出概率，並將 Sharpe 硬編碼為驗證回調，使模型在收斂階段直接優化風險調整收益，而非事後映射。」
- ❌ 錯答：「把 MSE 換成 Sharpe 就能直接提升策略表現，因為 Sharpe 更貼合交易。」（忽略不可微性、分位數推斷機制與驗證集一次性計算的工程限制）
**7.1 可證偽預測:** 若接入真實滑點模型（>2bp），該靜態閾值策略的 Sharpe 將衰減至 1 以下（預測日期：2024-12-31）。

## §8 · For the Reader
- **因子研究員:** 將此架構作為 `validation callback` 模板，替換自研因子組合的 Sharpe/IR 計算邏輯，避免因子過度擬合 MSE。
- **高頻執行:** 注意 `valid_batch_size:2000` 的記憶體瓶頸與 `min nb trades:60` 的頻率限制，需改寫為流式 (streaming) 指標計算以適應高頻。
- **組合配置:** 當前單變量 SPY 設定無法直接用於多資產權重優化，需擴展 TFT 的 `unique_id` 維度並引入組合層 Sharpe 作為全局驗證指標。
- **研究學生:** 重點複現 `HuberMQLoss` 與分位數閾值過濾的對接代碼，理解可微訓練與不可微評估的橋接工程。

## References
- Framework: `NeuralForecast` / TFT (Time Series Transformer)
- QuantML 導讀: [增强深度学习模型评估以进行股市预测--Part I](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487443&idx=1&sn=c6c555e9b90e263f3ed083ba4166ed44&chksm=ce7e68cdf909e1db7adde95df86e8bcd92e30642b492d3f9e4de4267d29ab9027f0de83abcd5#rd)
- Lineage: Quantile Regression → TFT Architecture → Custom Validation Metric Engineering