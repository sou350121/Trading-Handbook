---
title: "ML-GAT"
description: "落點於多模態與結構化注意力的交叉軸，解了傳統金融圖模型「關係雜訊淹沒信號」與「人工構圖滯後」的 prior gap。"
---
<!-- ontology-5axis data=多模态 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=人机协同可解释 -->

> **發布**：2025-07-12 · （無 venue）
> **QuantML 導讀**：[ML-GAT：用于股票预测的多层图注意力模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490991&idx=1&sn=ad76da6ffa4b4b4568443d0199dd22ed&chksm=ce7e7ab1f909f3a7fe6607bd26225bc207a437167c6c1ede8a963dab02f230bea6358afee06a#rd)
> **核心定位**：落點於多模態與結構化注意力的交叉軸，解了傳統金融圖模型「關係雜訊淹沒信號」與「人工構圖滯後」的 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `多模态` | `日频波段` | `监督回归` | `端到端表征` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 將價格（LSTM）、新聞（BERT）與知識圖譜關係（Wikidata）注入同構股票圖，透過狀態與關係雙層注意力動態篩選鄰居與關係類型。② 核心 trick 在於用元路徑（meta-paths）將異構關係轉為同構，並以雙層 GAT 過濾無效連接，解決傳統 GNN 對關係類型「一視同仁」的聚合盲區。③ 這對「端到端表征」軸★的意義在於：將離散商業邏輯轉為可微的圖權重，使因子挖掘從人工規則轉向數據驅動的結構化注意力。④ 導讀指出其模擬交易夏普比率較最佳基線提升 94.81%。

**X-Ray.** 本模型在五軸 Pareto 中偏向「高維多模態 × 結構化注意力」，解了傳統 GNN 在金融圖上「關係雜訊淹沒信號」的工程坑。它不依賴頻繁調參的因子工程，而是讓模型自學何種商業關係在當下 regime 有效。然而，其 envelope 打不開高頻與微結構層級：日頻波段注定無法捕捉訂單流衝擊，且全倉買賣策略忽略交易成本與滑點，實盤夏普必遭侵蝕。對量化讀者而言，其價值不在直接實盤，而在提供「關係注意力權重」作為可解釋的結構因子，可與動量因子正交組合，或作為 RL agent 的狀態編碼器。

## §1 · 架構 / Core Mechanism
| 維度 | 前作 (GCN/TGC) | ML-GAT 改動 | 工程意圖 |
|---|---|---|---|
| 圖結構 | 人工構建或單一關係 | 元路徑（meta-paths）轉異構為同構 | 解決 Wikidata 稀疏與關係類型混雜 |
| 聚合機制 | 同等權重聚合鄰居 | 狀態注意力層（State Attention） | 動態篩選同關係下的有效鄰居節點 |
| 信息融合 | 單層或靜態加權 | 關係注意力層（Relation Attention） | 按當下市場 regime 動態分配關係類型權重 |

⚡ **Eureka:** 用「關係摘要向量」作為中間態，先按關係類型聚合鄰居，再按類型重要性聚合摘要，將圖卷積的「空間平滑」轉為「語義篩選」。
**信息流 ASCII:**
```
[Price] -> LSTM -> h_price
[News]  -> BERT -> h_news
[Wikidata] -> Meta-paths -> Homogeneous Graph
h_price, h_news -> Node Init -> h_i
h_i + Graph -> State Attn -> r_rel_type (relation summary)
r_rel_type + h_i -> Relation Attn -> h_agg
h_agg + h_price + h_news -> h_final -> Linear/Softmax -> Trend
```

## §2 · 數學層
📌 **Napkin Formula:**
$h_i^{(l+1)} = \sigma\left( \sum_{r \in \mathcal{R}} \alpha_{i,r} \cdot \text{Attn}_{state}(h_i, \{h_j\}_{j \in \mathcal{N}_r}, r) \right)$
複雜度：$O(|\mathcal{E}| \cdot d + |\mathcal{R}| \cdot d)$，其中 $|\mathcal{E}|$ 為邊數，$d$ 為隱層維度，$|\mathcal{R}|$ 為關係類型數（導讀提及使用篩選後的 10 種）。
直覺：第一層注意力計算節點間係數 $\beta_{ij}^r$，第二層計算關係類型權重 $\alpha_{i,r}$。雙層設計將「誰重要」與「什麼關係重要」解耦，避免梯度在異構邊上發散。
Loss/訓練：交叉熵損失（Cross-Entropy），Adam 優化器，lr=5e-4，wd=5e-5，batch=32，epochs=100，每層 Dropout=0.5。端到端聯合訓練。

## §3 · 數據層
*   **規模/頻率/市場**：S&P 500 423 支 / CSI 300 286 支，日頻（Daily）。
*   **來源**：價格（Yahoo Finance/CSMAR）、新聞（Yahoo Finance 等，約 15 萬篇）、關係（Wikidata 知識圖譜）。
*   **樣本外與容量假設**：導讀未披露具體回測區間與 train/val/test 劃分比例。假設為標準時間序列切分。容量受限於圖結構稀疏性與日頻調倉，單策略理論容量約為數千萬至億級美元（未驗證），且全倉策略在實盤中需考慮流動性約束。

## §4 · 代碼層
| 項目 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（需處理 Wikidata 元路徑提取與異構圖轉同構，BERT/LSTM 特徵對齊） |
| 數據可得性 | 價格/新聞公開；Wikidata 需自行爬取/清洗關係邊 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| S&P 500 / CSI 300 | F1-score | 未披露 | 未披露 | 11.82% |
| S&P 500 / CSI 300 | Accuracy | 未披露 | 未披露 | 12.6% |
| S&P 500 / CSI 300 | Avg daily return | 未披露 | 未披露 | 5.06% |
| S&P 500 / CSI 300 | Sharpe ratio | 未披露 | 未披露 | 94.81% |

*解讀*：Δ 欄數字均為導讀逐字給出的「較最佳基線提升」比例。F1 與 Accuracy 的雙位數提升反映模型在分類邊界上的確學到了結構化信號，但金融分類任務的基線通常較低，提升空間大。Sharpe 提升 94.81% 極具吸引力，但導讀明確指出策略為「全倉買賣」且「未計交易成本與滑點」。實盤中，日頻調倉的佣金與衝擊成本將直接侵蝕日均回報的 5.06% 提升，真實 Sharpe 可能回落至基線水平。此外，GCN 的 F1 尚可但 Sharpe 遠低於 LSTM，證明「有效聚合」比「引入關係」更重要，Δ 中的夏普增益主要來自注意力過濾了導致過度交易的雜訊關係。

## §6 · 失效與隱含假設
*   **6.1 論文自述 limitations**：新聞數據稀疏性導致夏普比率在不含新聞模塊時反而稍好；未來需探索更先進 NLP 與多源關係網絡（如 multi-Hawkes Process）。
*   **6.2 推斷的隱含假設**：
    *   **Regime 依賴**：雙層注意力權重高度依賴訓練期分佈，若市場結構性斷裂（如政策轉向、流動性枯竭），Wikidata 的靜態商業關係將失效，注意力機制可能鎖定過時連接。
    *   **數據泄漏/前瞻**：新聞抓取時間戳若未嚴格對齊交易時點，易引入盤中信息泄漏；標籤定義（漲/盤/跌）的閾值未披露，若閾值過窄會導致樣本不平衡與過擬合。
    *   **Survivorship**：僅包含指數成分股（423/286 支），剔除無關係節點，隱含生存者偏差，實盤需處理退市/停牌股票的圖節點動態更新。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| TGC (Time-Graph Conv) | 關係聚合權重（靜態/同等 vs 動態雙層注意力） | 未披露 | 基線 |
| GCN | 圖卷積平滑 vs 語義篩選 | 未披露 | 基線 |
| LSTM (純價格) | 單模態時間序列 vs 多模態圖結構 | 未披露 | 基線 |

🎤 **Interview Tip:**
*   **正確答**：「ML-GAT 的核心不在於堆疊 GNN 層，而在於將『關係類型』作為可學習的注意力維度。它解決了金融圖中異構邊權重難以先驗設定的問題，但實盤需警惕靜態知識圖譜的滯後性與全倉策略的成本摩擦。」
*   **錯答**：「它用 BERT 和 LSTM 提取特徵所以效果好，圖注意力只是把鄰居加權平均。」（忽略元路徑轉同構與雙層解耦設計，未觸及結構化信號篩選的本質）
*   **7.1 可證偽預測**：若將 Wikidata 關係替換為高頻共跳躍（co-jump）圖，ML-GAT 的日頻分類 F1 提升將不顯著（預測日期：2025-12-31）。

## §8 · For the Reader
*   **因子研究員**：提取 Relation Attention Layer 的 $\alpha_{i,r}$ 權重作為「結構相關性因子」，與傳統動量/價值因子正交，可構建多層面 Alpha 組合。
*   **高頻執行**：本模型為日頻波段，不適用。但其「狀態注意力」機制可降維移植至盤中訂單流圖，用於預測短期流動性黑洞。
*   **組合配置**：將模型輸出概率轉為風險預算權重，而非直接全倉買賣。利用其對新聞衝擊的敏感性（如 2018 年底土耳其危機回撤），構建事件驅動型尾部風險對沖模塊。
*   **LLM-agent / RL 策略**：將 ML-GAT 的 $h_{final}$ 作為 RL agent 的狀態編碼器（State Encoder），替代原始價格序列，可加速策略在稀疏獎勵環境下的收斂。

## References
*   原論文：ML-GAT：用于股票预测的多层图注意力模型（無 venue/arXiv）
*   Lineage：GCN (Kipf & Welling, 2016) → GAT (Veličković et al., 2018) → TGC (金融圖時序基線)
*   QuantML 導讀鏈接：[ML-GAT：用于股票预测的多层图注意力模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490991&idx=1&sn=ad76da6ffa4b4b4568443d0199dd22ed&chksm=ce7e7ab1f909f3a7fe6607bd26225bc207a437167c6c1ede8a963dab02f230bea6358afee06a#rd)