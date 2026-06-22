<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# TimePro 解構

> **發布**：2025-08-07 · ICML 25
> **QuantML 導讀**：[ICML 25 | 华为诺亚方舟：TimePro高效多元时序预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491281&idx=2&sn=c431c36f81ccc98cef4112b920e58024&chksm=ce7e79cff909f0d98baef327c37bd6d4d7b10f0ff2e6dd807bb4dfd8eccb427cd91cf8d5a1d4#rd)
> **核心定位**：落點於「端到端表征 × 监督回归」軸，針對多元長序列預測中「不同變數對目標影響存在時間滯後差異」（多延遲問題）的 prior gap，以硬體感知的線性複雜度架構取代 Transformer 的二次方瓶頸。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出基於 Mamba 的 TimePro 模型，專攻多元長序列的「多延遲問題」。② 核心 trick 為 HyperMamba 模組：沿變數維度掃描獲取初始狀態後，透過可學習偏移量與線性插值，自適應採樣關鍵時間點重構「變數與時間感知超狀態」。③ 這對「自動黑盒 × 長週期」軸★的意義在於，將狀態空間模型的選擇性機制從純時間維度擴展至「變數-時間」雙維度，兼顧細粒度依賴與線性擴展性。④ 導讀未給量化結果。

**X-Ray.** 放回五軸 Pareto，TimePro 實質是將 SSM 的選擇性掃描從「單一時序流」升級為「變數-時間雙流形」。它解了舊工程坑：Transformer 的二次方記憶體牆與通道獨立 MLP 的變數割裂問題。預測其打不開的 envelope：對極端 regime 切換（如流動性枯竭或結構性斷裂）的魯棒性未經驗證，因超狀態重構依賴歷史相關性分佈的平穩性；對量化讀者而言，其價值不在於單點預測精度，而在於提供一種低延遲、線性複雜度的多因子狀態壓縮器，可作為高維因子池的預處理層或 RL 策略的觀測空間編碼器。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作 (Bi-Mamba+, iTransformer, PatchTST) | TimePro 改動 | 工程意圖 |
|---|---|---|---|
| 狀態掃描路徑 | 純時間維度或通道獨立 | 變數維度雙向掃描 + 時間偏移採樣 | 解耦變數依賴與時間動態，避免統一處理導致的信息稀釋 |
| 狀態更新機制 | 靜態 SSM 參數或全局注意力 | HyperMamba 超狀態重構 (Time-tune) | 將可微插值引入狀態空間，實現關鍵時間點的自適應聚焦 |
| 歸一化與嵌入 | 標準 LayerNorm / 序列嵌入 | RevIN + 時間/變數保留嵌入 | 緩解訓練/測試分佈偏移，保留多變數結構以便後續雙維度交互 |

⚡ **Eureka:** 變數維度掃描定基調，時間偏移量採樣抓脈衝，超狀態重構合一。

```text
Input → RevIN → Time/Var Preserved Embedding
       ↓
[ProBlock] → HyperMamba (Bi-scan → Time-tune w/ learnable offset → Linear Interp → HyperState)
       ↓                + TimeFFN (捕捉變數內部非線性)
       ↓
Linear Projection → Output
```

## §2 · 數學層
📌 **Napkin Formula:**
離散化 SSM 迭代：$h_t = \bar{A} h_{t-1} + \bar{B} x_t$
Time-tune 採樣：$\hat{h} = \text{Interp}(h, \text{ref} + \Delta_{\text{learn}})$
複雜度：$O(L)$（線性），對比 iTransformer/PatchTST 的二次方。

**直覺:** 傳統 SSM 的參數靜態或僅隨時間變化；TimePro 將狀態更新解耦為「變數空間的基礎隱狀態」與「時間軸的可學習偏移採樣」。透過線性插值保證可微性，使模型能跳過冗餘時間步，直接對齊變數間的真实滯後關係。
**Loss/訓練:** 導讀未披露具體 loss 函數與訓練超參，標 TBD。

## §3 · 數據層
- **資料規模/頻率/市場/時段:** 導讀未披露具體頻率、時段與樣本量。
- **來源:** 8 個公開真實世界長期預測基準數據集（ETTh1/h2, ETTm1/m2, Exchange, Weather, Solar-Energy, ECL）。
- **樣本外與容量假設:** 採用標準 benchmark 劃分；容量假設依原論文設定，導讀未披露具體訓練/驗證/測試比例或序列長度上限。

## §4 · 代碼層
| 欄位 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（需實現 Hyper-Scan 雙向掃描與 Time-tune 可微線性插值） |
| 數據可得性 | 高（均為公開 benchmark） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA (SOFTS) | 前SOTA (iTransformer) | 本方法 | Δ |
|---|---|---|---|---|---|
| Weather | MSE | 未披露 | 未披露 | 未披露 | 低2.0% / 低3.1% |
| Exchange | MSE/MAE | 未披露 | 未披露 | 未披露 | 未披露 |
| ETTh1 | MSE/MAE | 未披露 | 未披露 | 未披露 | 未披露 |
| 全局 | 排名 | 未披露 | 未披露 | 12個第一 / 2個第二 | 未披露 |
| 全局 | 推理速度 | 未披露 | 未披露 | 未披露 | 未披露 |
| 全局 | 參數量/GFLOPs | 未披露 | 未披露 | 未披露 | 未披露 |

**解讀:** 導讀僅披露 Weather 上的相對 MSE 降幅（低2.0% / 低3.1%）與全局排名分佈，未給絕對數值。該 Δ 反映的是對多延遲結構的捕捉能力，屬真 capability；但 Weather/Energy 數據具強季節性與物理先驗，增益在金融噪聲環境中可能衰減。推理速度（2.7倍 / 14.4倍）與資源佔用（67% / 78%）的優勢源於硬體感知設計與線性複雜度，屬架構紅利，非過擬合。成本與前瞻偏差未計入，純預測導向。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 導讀未明確列出 limitations。消融顯示移除深度卷積與掃描後線性投影後性能微升，暗示模型對變數空間局部性假設較弱，且 TimeFFN 已覆蓋部分輸出映射需求。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 超狀態重構依賴歷史相關性分佈的平穩性；若變數間滯後結構發生結構性斷裂（如政策轉向），線性插值採樣可能跟蹤滯後。
- **容量/成本:** 線性複雜度適合長序列，但未計入交易成本、滑點與再平衡頻率；純點預測無法直接轉化為 PnL。
- **數據泄漏:** RevIN 依賴全局均值/方差統計量，若樣本外劃分未嚴格隔離統計量計算，存在潛在泄漏風險。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| iTransformer | 變數交互方式（注意力 vs 變數掃描+超狀態） | 是 | 成熟 benchmark |
| PatchTST | 時間建模（通道獨立 vs 變數-時間雙維度） | 是 | 成熟 benchmark |
| S-Mamba | 複雜度/資源佔用（線性 vs 線性但參數更少） | 是 | 成熟 benchmark |

🎤 **Interview Tip:** 
- **正確答:** TimePro 將狀態更新從純時間維度解耦為「變數掃描定基 + 時間偏移採樣」，用可微插值實現超狀態重構，兼顧線性複雜度與多延遲捕捉。
- **錯答:** TimePro 只是把 Mamba 的掃描方向從時間改成變數，本質沒變。（錯在忽略 Time-tune 的自適應採樣與超狀態重構機制，以及硬體感知的 SRAM/HBM 優化設計）

**7.1 可證偽預測帶日期:** 至 2025-12-31，若將 TimePro 直接套用於高頻 tick 數據，其線性插值時間調整機制將因微結構噪聲過大而失效，預測誤差將顯著高於基於圖神經網絡或純注意力架構的模型。

## §8 · For the Reader
- **因子研究員:** 可將 HyperMamba 的超狀態輸出作為高維因子的壓縮表征，替代 PCA/VAE，保留變數間滯後關係與非線性動態。
- **高頻執行:** 模型設計為長週期預測，線性複雜度適合分鐘級重構，但需驗證微結構跳躍下的插值穩定性與延遲分佈。
- **組合配置:** TimePro 提供的是點預測，需外接風險模型或 RL 策略層；其低 FLOPs 特性便於多場景蒙特卡羅壓力測試與資產配置優化。
- **LLM-agent/RL 策略:** 可作為觀測空間的時序編碼器，將多元市場狀態壓縮為固定維度超狀態，供下游策略網絡使用，降低狀態空間維度災難。

## References
- ICML 25 原論文 (TBD)
- QuantML 導讀鏈接 (見上方)
- Lineage: Mamba (S6), iTransformer, PatchTST, S-Mamba, Bi-Mamba+, TimeMachine, DLinear, LightTS, SOFTS