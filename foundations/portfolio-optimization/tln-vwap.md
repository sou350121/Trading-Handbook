<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=监督回归 alpha=组合执行优化 autonomy=全自动黑盒 -->

# TLN-VWAP 解構

> **發布**：2025-04-18 · （無 venue）
> **QuantML 導讀**：[基于深度学习的VWAP执行策略](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490093&idx=1&sn=e45a68c581147bf29fa124f06a57c1c8&chksm=ce7e7d33f909f425bad6b0a4cb32f036c69f361d54be658c8dacfe4b86d4cbd290340af89127#rd)
> **核心定位**：落點於「組合執行優化」軸，解構傳統 VWAP 執行中「預測成交量曲線→分配訂單」的兩步法斷裂，將執行目標直接嵌入監督回歸的損失函數。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `高频日内` | `监督回归` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出端到端 VWAP 執行框架，繞過成交量預測中間步驟。② 核心 trick 為利用自動微分與自定義 VWAP 滑點損失，直接優化訂單分配權重。③ 對「組合執行優化」軸★ 的意義在於證明架構複雜度非必需，簡單線性模型對齊真實交易目標即可降滑點。④ 導讀未給量化結果。

**X-Ray.** 傳統執行框架的 Pareto 劣勢在於「代理目標錯配」：預測成交量曲線（MSE/MAE）與實際執行目標（VWAP 滑點）存在數學斷裂，波動 regime 下預測誤差會沿分配路徑非線性放大。TLN-VWAP 的工程價值在於將 Loss 設計從「特徵拟合」轉向「目標對齊」，用 Softmax 約束保證分配合法性，用自動微分打通滑點梯度。這解了「預測完美但執行虧錢」的舊坑。但它打不開的 envelope 很明確：靜態回顧窗無法處理訂單生命週期內的流動性枯竭與微觀結構反饋；小時級數據掩蓋了盤中滑價與手續費摩擦。對量化讀者的意義是示範了「Loss Engineering > Architecture Hunting」在執行任務上的有效性，但實盤必須疊加動態再平衡與成本約束。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 傳統兩步法 | TLN-VWAP |
|---|---|---|
| 優化目標 | 預測成交量曲線（代理變量） | 直接最小化 VWAP 滑點損失 |
| 模型架構 | 計量/ML 預測器 + 規則分配 | TLN + Softmax 約束輸出 |
| 訓練機制 | 最小化預測誤差 (MSE/MAE) | 自動微分優化自定義執行 Loss |

**1.2 ⚡ Eureka**
放棄「猜準成交量」的代理目標，用自動微分把「滑點」本身變成可導的 Loss，讓模型直接學「怎麼分倉最省錢」。

**1.3 信息流 ASCII**
```
Input (Vol, Hour, Day, Ret) → TLN Layers → Raw Weights → Softmax (Σ=1, ≥0)
       ↓
Allocation Curve → Custom VWAP Loss (Abs/Quad) → Backprop (Adam)
```

## §2 · 數學層
📌 **Napkin Formula**：$L_{VWAP} = \frac{1}{T} \sum_{t=1}^T |P_{exec}^t - VWAP^t|$ （Absolute）或二次形式。複雜度：$O(T \cdot F \cdot L)$ 每步前向。
**直覺**：滑點可分解為價格偏差加權與成交量分佈差異。模型透過 Softmax 輸出 $\omega_t$ 直接壓低分佈差異項，避開預測誤差累積。
**Loss/訓練**：Adam 優化器，初始 LR=0.001，最大輪數 1000，早停 patience=10，LR 降頻 patience=5。

## §3 · 數據層
Binance 永久合約，BTC/ETH/BNB/ADA/XRP 五種加密貨幣。頻率：小時級。時段：2020-01-01 至 2024-07-01。劃分：前 80% 訓練（含 20% 驗證集），最後 20% 測試。樣本外假設：靜態回顧窗（12步/120步），未披露具體樣本量與容量限制。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 低（TLN+Softmax+自定義Loss） | 中（需 Binance 小時級歷史數據） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Binance 加密貨幣 | Absolute VWAP Loss | 未披露 | 未披露 | 未披露 |
| Binance 加密貨幣 | Quadratic VWAP Loss | 未披露 | 未披露 | 未披露 |
| Binance 加密貨幣 | Volume Curve R² | 未披露 | 未披露 | 未披露 |
*解讀*：導讀正文於表2前截斷，所有數值嚴格標註為未披露。若實測，需警惕小時級數據對高頻執行的降頻偏差，以及未計入滑價/手續費的純 VWAP 偏差可能低估真實成本。基線包含 Flat、Fixed Optimal (DE)、Linear Regression、Dynamic VWAP，應逐項對比而非合成區間。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：靜態框架未納入訂單生命週期內的動態更新；未處理市場微觀結構與交易成本。
**6.2 推斷的隱含假設**：依賴小時級數據的穩定性（加密貨幣波動性高，小時級可能掩蓋盤中流動性枯竭）；容量假設未驗證（大單執行可能改變 VWAP 本身，但模型未建模 impact）；數據泄漏風險低（嚴格時間順序劃分），但回顧窗固定可能無法適應 regime shift。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統成交量預測執行 | 代理目標 vs 直接目標 | TBD | 工業界主流 |
| RL 動態執行 | 離散決策 vs 連續分配 | TBD | 學術探索 |
🎤 **Interview Tip**：正確答：「VWAP 執行的核心矛盾是預測誤差與執行目標的錯配，TLN-VWAP 用自動微分將滑點 Loss 直接嵌入訓練，繞過成交量預測的代理變量，適合波動市場但需警惕靜態分配的流動性假設。」錯答：「用 Transformer 預測成交量曲線就能完美執行 VWAP。」（忽略預測誤差累積與直接優化目標的必要性）。
**7.1 可證偽預測**：若 2025-Q3 加密貨幣市場出現連續小時級流動性斷層，該靜態分配策略的 VWAP 滑點將顯著劣於動態再平衡策略（預測日期：2025-09-30）。

## §8 · For the Reader
- **因子研究員**：將自定義 Loss 思維移植至 Alpha 生成，別再迷信 MSE；執行任務的 Loss 設計應直接映射交易成本。
- **高頻執行**：小時級框架需降頻至 Tick/Orderbook 層級驗證，警惕微觀結構摩擦與盤中流動性枯竭。
- **組合配置**：可作為大單拆單的基礎模塊，但實盤必須疊加風險預算、滑價模型與跨資產流動性約束。

## References
- 原論文：TLN-VWAP（無 venue/arxiv，標註為 QuantML 導讀來源）
- Lineage：Konishi[19], McCulloch & Kazakov[23]
- QuantML 導讀鏈接：[基于深度学习的VWAP执行策略](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490093&idx=1&sn=e45a68c581147bf29fa124f06a57c1c8&chksm=ce7e7d33f909f425bad6b0a4cb32f036c69f361d54be658c8dacfe4b86d4cbd290340af89127#rd)