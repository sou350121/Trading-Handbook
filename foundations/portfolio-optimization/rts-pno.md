---
title: "RTS-PnO"
description: "落點於「監督回歸 × 組合執行優化」軸，解決傳統 PtO 架構中預測目標與決策目標的結構性錯配。透過端到端 PnO 範式與動態保形風險約束，將不確定性內生化為執行預算，而非事後過濾。"
---
<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=监督回归 alpha=组合执行优化 autonomy=全自动黑盒 -->

> **發布**：2025-07-13 · KDD25
> **QuantML 導讀**：[KDD 25 |  时间至关重要：基于风险感知的时序预测与分配框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491002&idx=1&sn=ef72c0d6683b25e8132cdb7512fb8f5b&chksm=ce7e7aa4f909f3b2ab88937ed2014b01a45ad98e4060869f571d473aaf60b829a3dd26d3210b#rd)
> **核心定位**：落點於「監督回歸 × 組合執行優化」軸，解決傳統 PtO 架構中預測目標（MSE）與決策目標（Regret）的結構性錯配。透過端到端 PnO 範式與動態保形風險約束，將不確定性內生化為執行預算，而非事後過濾。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `高频日内` | `监督回归` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出端到端 RTS-PnO 框架，直接以決策遺憾值優化預測模型。② 核心 trick 為 SPO+ 代理損失回傳不可微優化層梯度，並結合保形預測動態校準風險約束。③ 對執行優化軸的關鍵價值在於打破「預測精度越高決策越好」的幻覺，以決策損失重塑特徵表示。④ 實盤 A/B 測試將平均決策成本降低 8.4%。

**X-Ray.** RTS-PnO 在 Pareto 前緣上明確選擇「犧牲預測像素級準確度（MSE/MAE 普遍下降），換取關鍵極值點的決策洞察」。它填補了 DFL（Decision-Focused Learning）在金融執行場景的工程坑：傳統 PtO 將不確定性視為靜態濾網，導致模型訓練後約束過時；RTS-PnO 以 epoch 級別動態重算分位數閾值，使風險預算與模型置信度同步收斂。該框架打不開的 envelope 在於推理延遲（依賴 LP 求解器），不適用於毫秒級 HFT，但完全覆蓋分鐘級 intraday 執行。對量化讀者的意義在於：Alpha 生成不應再孤立於損失函數設計之外，執行成本應直接編碼為訓練目標；保形預測提供了一條無需分佈假設的實用風險控制路徑。

## §1 · 架構 / Core Mechanism
| 改動維度 | 傳統 PtO / Forecasting-Only | RTS-PnO |
|---|---|---|
| 訓練目標 | 最小化 MSE/MAE（預測誤差） | 最小化 SPO+ Regret（決策成本） |
| 不確定性處理 | 靜態約束或完全忽略 | Epoch 級別動態校準（保形分位數） |
| 優化層互動 | 兩階段解耦（預測→求解器） | 端到端梯度回傳（代理損失橋接） |

⚡ **Eureka:** SPO+ 代理損失讓不可微的 LP 求解器「開口說話」，將真實決策成本梯度回傳至預測網絡；同時保形預測將風險約束從「固定閾值」升級為「隨模型置信度收縮的動態預算」。

```
[History] → [Forecaster] → [Predictions]
                      ↓
[Calibration Set] → [Conformal Error Calc] → [Uncertainty Vector σ]
                      ↓
[Adaptive Threshold τ_k] → [LP Solver] → [Allocation α]
                      ↓
[SPO+ Loss + λ·L_pred] → [Backprop] → [Update Forecaster]
```

## §2 · 數學層
📌 **Napkin Formula:**
`min_w  L_pred(w) + λ·L_SPO+(w)`  s.t. `Σ (α_t · σ_t) ≤ τ_k`
`τ_k = Quantile_k({σ_t^(cal)})`  |  複雜度：`O(E · T · LP_solve)`

**直覺:** 損失函數不再懲罰「每個時間點的價格猜錯」，而是直接懲罰「基於預測做出的買入比例與真實最優比例的決策差距」。約束項中的風險預算 `τ_k` 隨訓練推進自動收緊，避免初期高不確定性導致過度保守、後期低不確定性導致過度激進。訓練細節：每輪 epoch 結束後重新跑一次保形校準集，更新 `σ` 與 `τ_k`，再進行下一輪梯度下降。

## §3 · 數據層
- **規模/頻率/市場:** 8 個數據集（貨幣 USD2CNY/USD2JPY、股票 S&P 500/Dow Jones、加密 BTC/ETH）；分鐘級決策場景。
- **來源與處理:** 公開市場數據 + 騰訊跨境支付實盤流水；保形預測需嚴格劃分 Calibration Set 與 Test Set。
- **樣本外與容量假設:** 單變量時間序列驗證；假設校準集與測試集滿足交換性（exchangeability）；未披露多變量擴展與滑點/衝擊成本建模。

## §4 · 代碼層
| 項目 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | Medium（需集成 LP 求解器如 Gurobi/COPT + 保形校準循環） |
| 數據可得性 | TBD |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA (逐行列出) | 本方法 | Δ |
|---|---|---|---|---|
| USD2CNY | Regret | Forecasting-Only / Risk-Avoiding / RTS-PtO / Fixed-PnO | RTS-PnO | 超過12% |
| USD2CNY (FEDformer backbone) | Regret | Fixed-PnO | RTS-PnO | 超過11% |
| 騰訊跨境支付 (2024年12個時段) | Regret | 生產線 PtO 基準 | RTS-PnO | -8.4% |
| 全數據集 | MSE/MAE | 未披露 | 未披露 | 未披露 |

**解讀:** Δ 在 Regret 上的改善屬真實 capability，證明 SPO+ 損失成功將優化目標錨定至執行成本。但導讀明確指出 MSE/MAE 普遍下降，這是 PnO 的預期 trade-off：模型學會忽略對決策無害的預測誤差，專注極值點。需注意 Δ 未計入推理延遲成本與市場衝擊，實盤 8.4% 成本降低已涵蓋真實業務摩擦，但回測環境的 Regret 計算假設無滑點執行。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 推理時間較長，不適用於毫秒級高頻交易；僅在單變量時間序列驗證，多變量擴展待探索。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 保形預測依賴校準集與測試集的分佈穩定性；若波動率結構突變（如閃崩/流動性枯竭），動態 `τ_k` 可能滯後 1 個 epoch 才收斂，導致短期風險敞口失控。
- **容量/成本:** LP 求解器複雜度隨時間步長 `T` 線性/超線性增長；未建模訂單簿深度與衝擊成本，極端行情下 `α_t` 可能超出可執行容量。
- **數據泄漏:** 若 Calibration Set 與訓練集時間重疊或存在前瞻偏差，保形分位數將被低估，風險約束形同虛設。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 PtO (Gurobi/COPT) | 目標錯配 vs 端到端對齊 | Open | 工業界主流基線 |
| RL-based Execution (PPO/SAC) | 模型無關代理損失 vs 環境交互策略梯度 | Open | 研究熱點，樣本效率低 |
| 純預測模型 (PatchTST/TimesNet) | 決策損失優化 vs 像素級 MSE | Open | 預測軸 SOTA |

🎤 **Interview Tip:** 
- ✅ 正確答：「PtO 的致命傷是 MSE 最小化不等價於 Regret 最小化。RTS-PnO 用 SPO+ 代理損失將 LP 求解器的反饋轉為可微梯度，並用保形預測動態縮放風險預算，使預測網絡直接學習『對決策有幫助的誤差分佈』。」
- ❌ 錯答：「只要預測模型夠準（如換成 SOTA Transformer），PtO 的決策成本自然會下降。」（忽略目標錯配與不確定性靜態化問題）

**7.1 可證偽預測:** 若市場進入高頻波動 regime（如宏觀數據發布窗口），保形校準集的誤差分佈將偏離測試集，`τ_k` 動態調整滯後將導致 Regret 劣於靜態 Risk-Avoiding 策略；驗證窗口：2025-Q4 宏觀事件密集期。

## §8 · For the Reader
- **因子研究員:** 將 SPO+ 損失引入 Alpha 訓練循環，測試是否能在高换手策略中降低滑點敏感度；注意 λ 係數的穩健性掃描。
- **高頻執行:** 該框架推理瓶頸明確，不適合 tick 級訂單路由；可降頻至分鐘級 VWAP/TWAP 執行引擎，或將 LP 求解器替換為近似可微優化層（如 OptNet）以提速。
- **組合配置/RL 策略:** 保形預測提供了一條無需假設收益分佈的風險控制路徑；可與 RL 的 Constraint MDP 結合，用動態 `τ_k` 替代固定 Lagrange 乘子。

## References
- KDD 2025: *Risk-aware Time-Series Predict-and-Allocate (RTS-PnO)*
- Lineage: SPO+ Loss (Mandi et al., 2020) · Conformal Prediction (Vovk et al.) · Decision-Focused Learning (Pogančić et al.)
- QuantML 導讀: [KDD 25 | 时间至关重要：基于风险感知的时序预测与分配框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491002&idx=1&sn=ef72c0d6683b25e8132cdb7512fb8f5b&chksm=ce7e7aa4f909f3b2ab88937ed2014b01a45ad98e4060869f571d473aaf60b829a3dd26d3210b#rd)