<!-- ontology-5axis data=多模态 horizon=跨周期 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# ViL (Vision-LSTM) 解構（ViL (Vision-LSTM)）

> **發布**：2024-06-11 · （無 venue）
> **QuantML 導讀**：[ViL：通用图像识别框架的xLSTM应用](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247484694&idx=1&sn=749b65c0987fc7f47ec79ec780f4a690&chksm=ce7e6208f909eb1eb8e0719cefa63ea745716a3b5a401f78341eee5ac57ba09118187bf21742#rd)
> **核心定位**：將 xLSTM 的並行化矩陣記憶與交替掃描機制適配至視覺骨架，試圖以線性複雜度替代 ViT 的二次注意力，同時解決傳統 LSTM 在長序列建模中的梯度瓶頸。對量化而言，其「奇偶層交替方向」設計為跨周期序列的雙向依賴捕獲提供了一種低延遲的端到端表征路徑，但需跨越 CV→金融的域偏移鴻溝。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 將 xLSTM 的 mLSTM 塊引入視覺任務，以奇偶層交替上下掃描替換標準 ViT 的全局注意力。② 核心 trick 是並行化矩陣記憶與協方差更新規則，配合 224x224 分塊與可學習位置嵌入。③ 對「跨周期/端到端表征」軸★的關鍵在於：線性計算複雜度下保留長程依賴，且運行時比 SSM 基線 Vim 快最多 69%。④ 關鍵實證：在 ImageNet-1K 上以 400/800 epochs 訓練，長序列微調（729 patches, 50% overlap）進一步推升性能，具體 Acc 數值未披露。

**X-Ray.** 放回五軸 Pareto，ViL 本質是「線性注意力替代方案」在視覺域的變體。它解了 ViT 的 $O(N^2)$ 計算坑與傳統 LSTM 的序列化瓶頸，但代價是犧牲了自注意力對全局稀疏交互的動態路由能力。奇偶層交替掃描（Top-Down / Bottom-Up）在圖像上等效於雙向 RNN 的並行化實現，這對量化跨周期序列（如日頻+分鐘頻 K 線拼接、或 Level2 訂單簿切片）具有直觀的遷移價值：無需因果掩碼即可捕獲前後文依賴，且矩陣記憶的協方差更新規則天然適合處理非平穩的金融時序。然而，ViL 打不開的 Envelope 很明確：它仍是監督分類頭架構，缺乏對多模態異步數據的對齊機制；其位置嵌入依賴固定網格，直接平移至非均勻採樣的金融時序需重構 Patch 策略。對量化讀者而言，ViL 的價值不在於直接套用於股價預測，而在於其 mLSTM 塊的並行化訓練范式與線性複雜度，為高頻/日頻混合頻段的端到端表征提供了低延遲推理的硬件友好型基底。

## §1 · 架構 / Core Mechanism
### 1.1 三大改動 vs 前作
| 維度 | ViT (Transformer) | Vim (SSM) | ViL (xLSTM) |
|---|---|---|---|
| 序列建模核心 | 全局自注意力 $O(N^2)$ | 狀態空間模型 (S6) | 並行化 mLSTM 塊 + 矩陣記憶 |
| 掃描/路由策略 | 全向並行 | 因果/雙向 SSM 卷積 | 奇偶層交替：奇數 Top-Down，偶數 Bottom-Up |
| 位置編碼 | 絕對/相對 PE | 隱式/可學習 | 可學習 Patch 位置嵌入 (Learnable Patch PE) |

### 1.2 ⚡ Eureka 一句話 trick
用奇偶層交替的雙向掃描路徑，將傳統 RNN 的序列依賴轉化為可完全並行化的矩陣記憶更新，兼顧長程建模與訓練吞吐。

### 1.3 信息流 ASCII 圖
```
Image -> Patchify (224x224) -> Learnable PE -> [mLSTM Block (Top-Down)] -> [mLSTM Block (Bottom-Up)] -> ... -> [mLSTM Block] -> Pool/Head -> Output
(奇數層: ↓)          (偶數層: ↑)          (交替堆疊)
```

## §2 · 數學層
📌 **Napkin Formula:**（概念簡化，非原論文完整推導）
$$C_{t} = \sigma(g_t) \odot (C_{t-1} + x_t \otimes x_t) \quad ; \quad h_t = \text{GLU}(x_t W) \odot \sigma(C_t h_{t-1})$$
複雜度：訓練 $O(N \cdot d^2)$ 並行，推理 $O(N \cdot d)$ 遞歸（$N$=序列長，$d$=隱層維度）
直覺：矩陣記憶 $C_t$ 通過協方差形式累積歷史 patch 交互，指數門控 $\sigma(g_t)$ 動態過濾無關信息，避免傳統 LSTM 的梯度消失。奇偶層交替確保序列雙向信息在並行訓練中充分混合。
Loss/訓練：標準 Cross-Entropy 分類損失。訓練 400/800 epochs，LR=1e-3 + Cosine Decay。長序列微調階段固定 backbone，僅更新頭部與位置嵌入，30 epochs。

## §3 · 數據層
- **資料規模/頻率/市場/時段**：ImageNet-1K（1.3M 訓練 / 50K 驗證 / 1000 類別），靜態自然圖像，無時間頻率概念。
- **怎麼來**：標準公開數據集劃分，224x224 分辨率裁剪/縮放。
- **樣本外與容量假設**：論文未披露金融時序適配細節。假設直接遷移：需將 K 線/訂單簿重構為 2D Patch 序列，樣本外依賴於市場 Regime 穩定性與 Patch 滑窗的獨立同分佈假設。容量未驗證，但線性複雜度暗示可擴展至長序列（如 729 patches 微調）。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| **Repo** | TBD（QuantML 導讀提及「論文及代碼下載見星球」，未公開 GitHub） |
| **Checkpoint** | TBD |
| **License** | TBD |
| **複現難度** | 中低（基於 xLSTM 官方實現改動，核心為 mLSTM 塊與交替掃描邏輯） |
| **數據可得性** | 高（ImageNet-1K 標準數據） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前 SOTA | 本方法 | Δ |
|---|---|---|---|---|
| ImageNet-1K | Top-1 Acc | Vim / DeiT-III-S | ViL-S+ / ViL-Tiny | 未披露 |
| ImageNet-1K | 推理延遲/吞吐 | Vim (CUDA 優化) | ViL | 運行時快最多 69%（未披露絕對 ms） |
| ImageNet-1K | 訓練 Epochs | DeiT-III-S (未披露) | ViL-S+ | 400 epochs（性能相當） |

**解讀**：
- **真 Capability**：69% 推理加速與 400 epochs 達到競爭性能，證明 mLSTM 並行化與交替掃描在計算圖優化上確實降低了訓練/推理開銷，線性複雜度優勢成立。
- **潛在偏差/未計成本**：① 未披露具體 Top-1 Acc 數值與標準差，無法判斷統計顯著性；② 對比 Vim 時雖匹配參數量，但未說明是否統一了數據增強/正則化策略，可能存在實驗設置不對稱；③ 長序列微調（729 patches）提升性能，但金融序列若直接套用此長度，可能引入過擬合或前瞻偏差（若滑窗未嚴格因果）。

## §6 · 失效與隱含假設
### 6.1 論文自述 limitations
- 預訓練方案與超參設置仍有優化空間（如未充分探索 LayerScale 等 Transformer 遷移技術）。
- 當前驗證僅限於圖像分類，未覆蓋檢測/分割等密集預測任務。

### 6.2 推斷的隱含假設
- **Regime 依賴**：奇偶層交替掃描依賴序列的局部平滑性與雙向依賴結構。金融市場在極端波動（如閃崩、流動性枯竭）時，序列統計特性突變，矩陣記憶的協方差累積可能滯後或發散。
- **容量/成本**：矩陣記憶 $C_t$ 維度為 $d \times d$，隱層擴大時顯存佔用呈平方增長，高頻實盤部署需嚴格控制 $d$ 或採用分塊近似。
- **數據泄漏/Survivorship**：若直接將 ImageNet 預訓練權重遷移至金融數據，需警惕域偏移（Domain Shift）；金融數據的幸存者偏差與前視偏差若未在 Patch 構建階段嚴格隔離，模型易學習到數據清洗痕跡而非真實 Alpha。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| ViT / DeiT | 注意力機制 ($O(N^2)$ vs $O(N)$) / 全局路由 vs 局部交替掃描 | ✅ | 成熟/基準 |
| Vim / Mamba | SSM 狀態轉移 vs xLSTM 矩陣記憶 / 因果卷積 vs 並行 RNN | ✅ | 活躍/SSM 陣營 |
| PatchTST / iTransformer | 時序專用 Patch/逆變換 vs 通用視覺 Patch / 單變量獨立 vs 多變量交互 | ✅ | 金融時序主流 |

🎤 **Interview Tip**：
- **正確答**：「ViL 的本質是將 xLSTM 的並行化矩陣記憶與雙向交替掃描引入視覺骨架，用線性複雜度替代 ViT 的二次注意力。其核心優勢在於訓練吞吐與長序列推理延遲的平衡，但金融遷移需重構 Patch 策略與位置編碼，且矩陣記憶的 $O(d^2)$ 顯存開銷需評估。」
- **錯答**：「ViL 就是 LSTM 的改進版，直接拿它跑股票預測比 Transformer 好，因為它沒有注意力機制所以不會過擬合。」（忽略並行化設計、域偏移問題與矩陣記憶複雜度）

### 7.1 可證偽預測
若 2024-Q4 前無開源實現將 ViL 的交替掃描機制適配至 Level2 訂單簿預測，且未解決 $d>512$ 時的顯存瓶頸，則該架構在實盤高頻場景的落地價值將被降級為「學術對比基線」。

## §8 · For the Reader
- **因子研究員**：勿直接套用 ViL 做股價回歸。提取其 `mLSTM` 塊替換現有 LSTM/GRU 的時序編碼器，利用並行化訓練加速因子挖掘循環；注意將 2D 圖像 Patch 重構為 1D/2D 金融特徵滑窗，並加入因果掩碼或嚴格時間劃分。
- **高頻執行/系統架構**：關注 69% 推理加速的來源（並行化訓練 vs 遞推推理）。實盤部署時，訓練可用並行模式，但低延遲推理必須切換至遞推模式（$O(N)$），需評估 CUDA kernel 優化與顯存佔用（$d \times d$ 矩陣）對 TPS 的影響。
- **組合配置/多模態策略**：ViL 的「奇偶層交替掃描」啟發了多頻段數據對齊新思路。可嘗試將日頻宏觀與分鐘頻量價數據拼接為長序列，用交替層分別捕獲長週期趨勢與短週期動量，但需嚴格驗證跨周期 Patch 的獨立性假設。

## References
- **原論文**：TBD（QuantML 導讀提及「論文及代碼下載見星球」，未提供 arXiv/DOI）
- **Lineage**：LSTM (Hochreiter & Schmidhuber, 1997) → xLSTM (Beck et al., 2024) → ViL (Vision-LSTM, 2024)
- **QuantML 導讀**：[ViL：通用图像识别框架的xLSTM应用](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247484694&idx=1&sn=749b65c0987fc7f47ec79ec780f4a690&chksm=ce7e6208f909eb1eb8e0719cefa63ea745716a3b5a401f78341eee5ac57ba09118187bf21742#rd)