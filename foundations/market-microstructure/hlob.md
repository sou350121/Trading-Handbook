---
title: "HLOB"
description: "落點於「微觀盤口×高頻日內」的端到端監督回歸黑盒。解了傳統 CNN/LSTM 將 LOB 視為規則網格、忽略價格層級間非線性耦合的 prior gap，用 TMFG 拓撲先驗將非歐空間結構轉化為可卷積的高階同調特徵。"
---
<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

> **發布**：2024-09-06 · （無 venue）
> **QuantML 導讀**：[HLOB：限价订单簿中的信息持久性和结构](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486124&idx=1&sn=5234e96240ce90a76152992af88c59c4&chksm=ce7e6db2f909e4a48e1494a97eaf8e8378d71dabc387c0f7d123e598f923358a207b6af2bda1#rd)
> **核心定位**：落點於「微觀盤口×高頻日內」的端到端監督回歸黑盒。解了傳統 CNN/LSTM 將 LOB 視為規則網格、忽略價格層級間非線性耦合的 prior gap，用 TMFG 拓撲先驗將非歐空間結構轉化為可卷積的高階同調特徵。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 用 TMFG 提取 LOB 量價層的空間拓撲先驗，經同卷積(HCNN)捕捉高階依賴，再串 LSTM 聯合建模時空動態。② 核心 trick 在於將互信息矩陣轉化為三角最大化過濾圖，把非結構化盤口快照轉為可微的幾何圖信號。③ 對「端到端表征」軸★：避免人工因子工程，直接從原始 tick 數據學習空間依賴退化規律。④ 關鍵實證：在 NASDAQ 15 檔股票上，於 10/50/100 tick 預測視窗下，分別以 73.3% / 60% / 33% 的勝率超越 9 種 SOTA 深度架構。

**X-Ray.** 放回五軸 Pareto，HLOB 將「空間拓撲先驗」硬編入卷積核，本質是將 LOB 的平面圖結構降維為可計算的同調群。它解了舊工程坑：傳統 CNN 將 LOB 視為規則網格，忽略價格層級間的異質性與非線性耦合；HLOB 用 TMFG 過濾掉冗余邊，保留四面體/三角形/邊三類高階結構，使卷積操作具備幾木不變性。預測其打不開的 envelope：TMFG 依賴訓練期平均互信息矩陣，對 regime shift（如波動率跳升、流動性枯竭）的拓撲重構滯後；且同卷積計算複雜度隨節點數呈多項式增長，難以直接擴展至全市場或更深盤口（>10 levels）。對量化讀者意義：提供了一條「圖拓撲先驗 + 時序記憶」的因子生成路徑，但需警惕其監督回歸標籤（Up/Down/Stable）與真實執行滑點/成本的非對稱性，實盤需搭配執行算法與風險預算。

## §1 · 架構 / Core Mechanism
| 維度 | 前作 (CNN/LSTM/Transformer) | HLOB 改動 | 工程意圖 |
|---|---|---|---|
| 空間表征 | 規則網格卷積 / 全連接注意力 | TMFG 拓撲過濾 + 同調卷積 | 將非歐盤口結構轉為可微幾何信號 |
| 時序建模 | 獨立 LSTM / 自回歸 | 空間特徵展平後串聯 LSTM | 解耦空間依賴與時間動態，降低梯度消失 |
| 標籤設計 | 連續價格回歸 / 二分類 | 三態分類 (Up/Down/Stable) + Tick 閾值 | 維持高頻序列平穩性，對齊微觀結構 |

⚡ **Eureka:** 用訓練期平均互信息構建 TMFG，把 LOB 的「價格-成交量」耦合關係提煉為四面體/三角形/邊三類高階圖結構，讓卷積核直接對齊市場微觀幾何。

```text
Raw LOB Ticks (10 levels)
          │
          ▼
[MI Matrix] → Bootstrapping → Avg MI Matrix
          │
          ▼
[TMFG Builder] → Tetrahedra / Triangles / Edges
          │
          ▼
[HCNN Heads] → Spatial Embeddings (×3) + Price Levels
          │
          ▼
[LSTM] → Temporal Dynamics → FC → Softmax (Up/Down/Stable)
```

## §2 · 數學層
📌 **Napkin Formula:**
$M_{ij} = \frac{1}{T}\sum_{t=1}^T I(X_i^t; X_j^t)$  (Avg MI Matrix)
$G_{TMFG} = \text{argmax}_{G \in \mathcal{P}} \sum_{(i,j)\in E} M_{ij}$  (Planar/Maximal Triangulation)
$H^{(l+1)} = \sigma(\Theta^{(l)} *_{\mathcal{H}} H^{(l)})$  (HCNN on Homology Groups)

**複雜度:** TMFG 構建 $O(N^2)$，HCNN 卷積 $O(|V|+|E|+|F|)$，LSTM 前向 $O(T \cdot d^2)$。
**直覺:** 互信息矩陣量化量價層級的統計依賴，TMFG 施加平面圖約束過濾雜訊邊，同卷積在簡化同調群（0/1/2 維）上執行局部聚合，最後由 LSTM 吸收時間殘差。
**Loss/訓練:** 分類交叉熵 (Cross-Entropy)，AdamW ($lr=6\times10^{-5}, \beta_1=0.90, \beta_2=0.95$)，按年份劃分 Train/Val/Test，超參繼承自 LOBFrame 基線。

## §3 · 數據層
- **市場/標的:** NASDAQ 15 檔股票（覆蓋 6 部門/13 子行業，市值 >$100B）。
- **頻率/時段:** 逐筆 (Tick-by-tick) LOB 數據，2017-01 至 2019-12。
- **來源/預處理:** LOBSTER 提供商；保留 10 檔價格/成交量層級；按 tick size 分三組 (Small/Mid/Large)；成交量分箱降噪；訓練期逐日計算成對 MI 並平均。
- **樣本外/容量:** 按年份嚴格劃分（訓練連續日/驗證隨機日/測試連續日）；未披露單日最大訂單流容量與滑點假設，推測適用於機構級流動性標的，小盤/低流動性股票拓撲穩定性存疑（未驗證）。

## §4 · 代碼層
| 維度 | 狀態 |
|---|---|
| Repo | TBD (QuantML 導讀提及「論文及代碼下載見星球」，未公開 GitHub) |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中高 (需 LOBSTER 數據 + TMFG 圖構建 + 同卷積自實現/調用 PyTorch Geometric) |
| 數據可得性 | 低 (LOBSTER 商業授權，15 檔 NASDAQ 數據需採購) |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| NASDAQ 15 stocks | F1 (HΔτ=10) | 未披露 | 未披露 | 未披露 |
| NASDAQ 15 stocks | MCC (HΔτ=50) | 未披露 | 未披露 | 未披露 |
| NASDAQ 15 stocks | pT (HΔτ=100) | 未披露 | 未披露 | 未披露 |

**解讀:** 導讀僅給出 HLOB 超越 SOTA 的「案例佔比」(73.3%/60%/33%)，未披露絕對 Metric 數值與 Δ。此 Δ 反映模型在特定股票/視窗的相對優勢，但 F1/MCC/pT 屬預測/模擬指標，未計入真實交易成本、滑點與訂單簿衝擊。高勝率可能部分源於訓練期 MI 矩陣的靜態平均（前瞻/過擬合風險），實盤 Sharpe/IR 需經執行層過濾後重估。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 相似性矩陣計算方法待改進；未引入時間演化的 IFN（拓撲靜態）；預測能力隨視窗延長退化。
**6.2 推斷的隱含假設:**
- **Regime 依賴:** TMFG 基於訓練期平均 MI，假設微觀結構依賴關係在測試期保持平穩。波動率跳升或流動性驟降時，拓撲重構滯後將導致特徵失效。
- **容量/成本:** 監督回歸標籤 (Up/Down/Stable) 忽略買賣價差與訂單簿深度非對稱性；未計入 TCA 成本，高頻信號需搭配智能路由/拆單算法。
- **數據泄漏/Survivorship:** 僅選取 NASDAQ 大型股 (>$100B)，存在倖存者偏差；按年劃分雖嚴謹，但 MI 計算若未嚴格隔離測試期數據，可能引入輕微前瞻。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| LobTransformer / iTransformer | 注意力機制 vs 圖拓撲卷積 | 部分開源 | 活躍迭代 |
| CNN1/CNN2 (LOBFrame) | 規則網格 vs 非歐同調群 | 開源 | 基線穩定 |
| DLA / BoF | 特徵工程/雙線性投影 vs 端到端圖信號 | 開源 | 維護中 |

🎤 **Interview Tip:**
- ✅ **正確答:** 「HLOB 的核心不在於 LSTM 的時序能力，而在於用 TMFG 將 LOB 的非歐空間依賴轉化為可卷積的同調特徵。實盤需解決靜態拓撲對 Regime Shift 的滯後，並對齊真實執行成本。」
- ❌ **錯答:** 「HLOB 就是用圖神經網絡預測股價，比 Transformer 準是因為注意力機制太慢。」（忽略同卷積幾何不變性與 MI 拓撲先驗的工程價值，混淆預測指標與交易收益）

**7.1 可證偽預測:** 若 2025-06-30 前，HLOB 的靜態 TMFG 拓撲未擴展為動態時變圖 (Dynamic IFN)，其在高波動 regime（如 VIX > 30）下的 pT 勝率將較平穩期下降 >15%。

## §8 · For the Reader
- **因子研究員:** 將 TMFG 提取的「四面體/三角形」權重視為高階盤口因子，可與傳統 Order Flow Imbalance 正交化，構建多頻因子庫。
- **高頻執行:** 模型輸出為方向概率，需接入執行算法（如 POV/TWAP）與風險預算；注意 Stable 標籤的閾值設定需對齊實際 Tick Size 與滑點容忍度。
- **RL 策略/研究學生:** 將 HLOB 的時空表征作為 RL Agent 的 State Space，替代原始 LOB 快照，可加速策略收斂；但需自行實現同卷積層或調用 PyTorch Geometric 擴展包。

## References
- 原論文: HLOB: Information Persistence and Structure in Limit Order Books (2024)
- Lineage: LOBFrame (Prata et al.), TMFG (Massara et al.), HCNN (Gao et al.)
- QuantML 導讀鏈接: [HLOB：限价订单簿中的信息持久性和结构](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486124&idx=1&sn=5234e96240ce90a76152992af88c59c4&chksm=ce7e6db2f909e4a48e1494a97eaf8e8378d71dabc387c0f7d123e598f923358a207b6af2bda1#rd)