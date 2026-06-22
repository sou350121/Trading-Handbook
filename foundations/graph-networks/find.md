<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# FinD³ 解構

> **發布**：2025-12-23 · CIKM 2025
> **QuantML 導讀**：[CIKM 25 | “3D状态空间+演化超图”挖掘Alpha](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492772&idx=1&sn=93f437ede205d1cdec4c1cf1041d7b8f&chksm=ce7d83baf90a0aac8624b62823c70355545d105681478b3aca007a29c218ada48534b5a8f778#rd)
> **核心定位**：落點於日频波段监督回归，解了傳統MTS坍缩特徵與靜態GNN無法捕捉Regime Shift的Prior Gap。將SSM狀態轉移與超圖拓撲更新統一為端到端可微流程，繞過因子工程瓶頸。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** 提出FinD³框架，將SSM擴展至3D立方體並引入演化超圖，端到端建模量價時序與動態關聯。核心Trick是Selective CubeSSM並行捕捉跨股/跨特徵依賴，配合Gumbel-Softmax可微二值化動態更新先驗超邊。此設計對「端到端表征」軸★，直接繞過因子工程瓶頸。導讀未給IRR具體數值，僅披露NASDAQ上SR達2.32。

**X-Ray.** 放回五軸Pareto，FinD³在「日频波段」與「端到端表征」的交集中，用線性複雜度的3D-SSM替代二次方Transformer，並用動態超圖補齊靜態圖的Regime盲區。解了舊工程坑：MTS通道獨立假設與GNN拓撲僵化。預測打不開的Envelope：高頻微結構（SSM狀態更新延遲與超圖稀疏化開銷不適合毫秒級）與極端流動性枯竭（先驗超邊依賴行業/股權數據，危機時關聯斷裂）。對量化讀者意義：提供一套可並行推論的3D特徵萃取器，但需警惕超圖正則化對尾部風險的掩蓋。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作/基線 | FinD³ 改動 |
|---|---|---|
| 時序建模 | UTS/MTS 坍缩為1D向量或通道獨立 | 3D-MTS 立方體建模，雙分支正交掃描 |
| 參數化機制 | 固定狀態轉移矩陣 (LTI) | Selective CubeSSM：FConv/SConv 動態生成參數 |
| 關聯結構 | 靜態圖或純數據驅動動態圖 | 先驗知識初始化 + Gumbel-Softmax 可微二值化演化 |

⚡ **Eureka:** 將3D張量視為立方體，用兩條正交SSM分支並行掃描，再以可微二值化超圖動態重權。
**信息流:**
```mermaid
flowchart TD
    A["Input (Stock×Feature×Time)"] --> B["Cross-Stock Branch"]
    A["Input (Stock×Feature×Time)"] --> E["Cross-Feature Branch"]
    B["Cross-Stock Branch"] --> C["Selective CubeSSM"]
    C["Selective CubeSSM"] --> D["h_stock"]
    E["Cross-Feature Branch"] --> F["Flip"]
    F["Flip"] --> G["Selective CubeSSM"]
    G["Selective CubeSSM"] --> H["h_feat"]
    D["h_stock"] -->|↓ (Concat/Flip back)| I["Unified Embedding"]
    H["h_feat"] -->|↓ (Concat/Flip back)| I["Unified Embedding"]
    I["Unified Embedding"] --> J["EHA (HAGC + Gumbel Topology)"]
    J["EHA (HAGC + Gumbel Topology)"] --> K["Output"]
```

## §2 · 數學層
📌 **Napkin Formula:**
$h_t = \bar{A}h_{t-1} + \bar{B}x_t$，$\bar{A},\bar{B}$ 由 FConv/SConv 動態生成。複雜度 $O(N \cdot F \cdot T)$。
$\mathcal{L} = \mathcal{L}_{MSE} + \lambda_1 \mathcal{L}_{Rank} + \lambda_2 \mathcal{L}_{Lap}$

**直覺:** 狀態轉移矩陣由卷積動態生成，超圖權重經Gumbel-Softmax稀疏化，回歸與排序損失雙軌優化。訓練時端到端反傳，超圖拓撲與SSM參數同步更新。

## §3 · 數據層
- **規模/頻率/市場:** NASDAQ (1,026只) / NYSE (1,737只)，日频，2013-2017。
- **特徵:** 收盤價 + 5/10/20/30日均線，最大絕對值標準化。
- **來源/假設:** 導讀未披露數據供應商與清洗細節。樣本外僅覆蓋2013-2017，未見2018後驗證。容量假設 TBD。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| QuantML知识星球 | TBD | TBD | 高（需自實作Selective CubeSSM與Gumbel超圖） | 中（需自備日频量價與行業/股權先驗圖） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| NASDAQ | SR | PatchTST 1.78 | FinD³ 2.32 | 0.54 |
| NASDAQ | 推論耗時 | PatchTST 159.84秒 | FinD³ 14.89秒 | 144.95秒 |
| NASDAQ | 推論耗時 | iTransformer 138.20秒 | FinD³ 14.89秒 | 123.31秒 |

**解讀:** Δ 0.54 的SR提升源於動態超圖對Regime Shift的適應，而非單純特徵記憶。推論耗時Δ 144.95秒/123.31秒證實SSM線性複雜度優勢，但導讀未披露訓練成本與實盤滑點/手續費扣除後的淨值，當前Δ僅反映離線推論效率。需警惕2013-2017低波動環境下的過擬合，以及Top-K買入持有策略未計入換手成本。

## §6 · 失效與隱含假設
- **6.1 論文自述 limitations:** 導讀未明確列出。
- **6.2 推斷的隱含假設:** 
  - **Regime 依賴:** 超圖演化假設關聯連續變化，黑天鵝時可能滯後。
  - **容量/成本:** 未計交易成本與流動性衝擊，實盤淨值可能顯著低於離線SR。
  - **數據泄漏:** DTW鄰近度與行業分組若使用未來數據構建先驗圖，將產生前瞻偏差。
  - **Survivorship:** 導讀未說明是否處理退市股票，存在生存者偏差風險。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| PatchTST | 3D立方體SSM vs 2D Patching | 開源 | SOTA基線 |
| iTransformer | 動態演化超圖 vs 靜態/數據驅動圖 | 開源 | SOTA基線 |
| CI-STHPAN | 可微二值化拓撲 vs 靜態圖 | 未披露 | 關係感知基線 |

🎤 **Interview Tip:** 
- **正確答:** FinD³用Selective CubeSSM解決3D-MTS坍缩問題，並以Gumbel-Softmax實現超圖拓撲的端到端可微更新，將狀態轉移與圖結構學習統一為單次反傳。
- **錯答:** 認為它是傳統Transformer的變體或僅靠靜態行業圖做卷積。

**7.1 可證偽預測:** 若將訓練窗口外推至後續高波動期，SR表現將顯著劣於導讀披露的2.32，且超圖正則化項會抑制對新興概念股的捕捉。

## §8 · For the Reader
- **因子研究員:** 可直接將DCSSM輸出作為高維隱因子輸入組合優化器，但需驗證Ranking Loss與傳統IC的對齊度。
- **高頻執行:** 不適用，SSM狀態更新與超圖稀疏化開銷不適合毫秒級決策，僅適合日频再平衡。
- **組合配置:** 關注EHA的Regime切換信號，可作為權重調整的宏觀過濾器。
- **LLM-agent:** 可將先驗超邊構建邏輯轉為知識圖譜Prompt，輔助Agent理解市場結構。
- **研究學生:** 重點拆解Selective CubeSSM的參數化策略，這是將SSM從1D擴展至3D的關鍵工程創新。

## References
- FinD³: Financial 3D model using Dual cubic state spaces and Dynamic hypergraphs. CIKM 2025.
- Lineage: Mamba (SSM), PatchTST, iTransformer, HyperGNN.
- QuantML 導讀鏈接: [CIKM 25 | “3D状态空间+演化超图”挖掘Alpha](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492772&idx=1&sn=93f437ede205d1cdec4c1cf1041d7b8f&chksm=ce7d83baf90a0aac8624b62823c70355545d105681478b3aca007a29c218ada48534b5a8f778#rd)