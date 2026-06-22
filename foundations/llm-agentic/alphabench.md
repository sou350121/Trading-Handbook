---
title: "AlphaBench"
description: "首個嵌入 Qlib 回測引擎的 LLM 因子挖掘系統性基準。解決了「純文本公式無法承載執行上下文」的評估斷層，將 LLM 從不可靠的零樣本裁判重構為 SFT 成對比較器與 EA 變異算子。"
---
<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=生成式大模型 alpha=因子挖掘 autonomy=Agent自主演进 -->

> **發布**：2026-05-25 · （無 venue）
> **QuantML 導讀**：[哪款大模型更适合因子挖掘？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247493934&idx=1&sn=6dbb6e8846d1cf5186b7a52de93f783d&chksm=ce7d8e30f90a07262bcece160866ce33dbf9fde5066d2e6e2908bfc17e05b7299219ab1d3b00#rd)
> **核心定位**：首個嵌入 Qlib 回測引擎的 LLM 因子挖掘系統性基準。解決了「純文本公式無法承載執行上下文」的評估斷層，將 LLM 從不可靠的零樣本裁判重構為 SFT 成對比較器與 EA 變異算子。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `生成式大模型` | `因子挖掘` | `Agent自主演进` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 建立 Text2Alpha/評估/搜索三維基準，驗證零樣本因子評分在數學上屬欠定問題。② 核心 trick 為放棄絕對打分，改用 SFT 微調成對比較器，並強制輸入 AST 結構與執行元數據。③ 對「Agent自主演进」軸★：將 LLM 降維為可控的符號變異/雜交生成器，回測引擎接管物理約束。④ 導讀給出關鍵實證：GPT-4.1-Mini 成對選擇準確率從 0.44 提升至 0.83，且跨市場 SP500 維持 0.64。

**X-Ray.** 本框架切中 LLM 量化落地的核心 Pareto 邊界：語義生成力 vs 物理執行約束。舊工程坑在於試圖用 Prompt 讓模型直接輸出 RankIC 或絕對評分，忽略調倉頻率、滯後結構等隱變量，導致零樣本準確率淪為擲硬幣。AlphaBench 的解法不是堆參數，而是「表徵重構」：用 AST 補齊算子圖拓撲，用成對損失函數強迫模型學習結構魯棒性而非記憶市場分佈。預測其打不開的 envelope 在於高頻微結構與非線性訂單流因子，因 AST 目前僅覆蓋日頻量價算子集。對量化讀者的意義：徹底終結「Prompt 即策略」的幻想，確立「LLM 生成 + 硬代碼過濾 + 回測約束」的標準 Agent 流水線。

## §1 · 架構 / Core Mechanism
| 維度 | 前作/基線 (Symbol Regression / Zero-shot LLM) | AlphaBench 改動 |
|---|---|---|
| **評估範式** | 零樣本絕對評分 (1-5分) 或單次生成 | SFT 成對比較器 (Pairwise Selection) + 高 contrast 訓練集 |
| **輸入表徵** | 純文本公式字符串 | AST 抽象語法樹 JSON + 執行元數據 (調倉頻/滯後/NaN處理) |
| **搜索與約束** | CoE 串行 / GP 符號回歸 | EA 並行演化 + 硬代碼過濾器 (AST深度≤5 / Lookahead 強懲罰 / 計算時限) |

⚡ **Eureka:** 放棄讓 LLM 當「絕對裁判」，改讓它當「相對排序器」；用 AST 與元數據補齊公式背後的物理執行上下文，使比較任務從欠定問題轉為可學習的結構特徵匹配。

**信息流 ASCII:**
```
User Intent / Taxonomy
       │
       ▼
[LLM Agent] (EA/CoE/ToT) ──生成候選公式──►
       │                              │
       ▼                              ▼
[AST Parser & Hard Filter] ◄──語法/前瞻/深度/NaN/時效校驗──┘
       │
       ▼
[Qlib Backtest] ──計算 IC/RankIC/IR ──►
       │
       ▼
[SFT Pairwise Comparator] ──相對排序/篩選 Top-K ──► Factor Pool
```

## §2 · 數學層
📌 **Napkin Formula:**
`L_pairwise = -log(σ(s(f_i) - s(f_j)))`，其中 `s(·)` 為 SFT 評分網絡，`f_i` 為 RankIC 顯著優於 `f_j` 的因子對。
**複雜度:** 訓練 O(B·D)，推理 O(1) per pair。B 為 batch size，D 為 AST 節點深度。
**直覺:** 相對損失強制模型提取算子單調性、分母防零溢出、窗口交互等跨市場通用結構特徵，而非擬合單一市場的收益分佈漂移。
**Loss/訓練細節:** 使用二分類/排序損失；訓練集採用高 contrast 設計（確定 Signal vs 確定 Noise 強行配對），防止邊界模糊導致過擬合。

## §3 · 數據層
- **規模/頻率/市場:** CSI300 股票池，日頻波段，2020-2025 真實數據。
- **來源/處理:** 內嵌 Qlib 回測框架，因子計算依賴標準量價截面。
- **樣本外假設:** 驗證跨市場泛化能力（直接遷移至 SP500 未見數據）；假設算子物理意義與微結構在跨市場間具備結構同構性。

## §4 · 代碼層
| 欄位 | 狀態/細節 |
|---|---|
| **Repo** | TBD（導讀提及「代碼見QuantML知識星球」，未公開 GitHub） |
| **Checkpoint** | TBD |
| **License** | TBD |
| **複現難度** | 中（需 Qlib 環境 + 自備高 contrast 因子標註數據 + AST 解析器） |
| **數據可得性** | 需自備日頻量價數據與算子規範清單 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA (Zero-shot) | 本方法 (SFT/優化) | Δ |
|---|---|---|---|---|
| CSI300 | Pairwise Accuracy (GPT-4.1-Mini) | 0.44 | 0.83 | 0.39 |
| CSI300 | Pairwise Accuracy (CoT) | 未披露 | 0.86 | 未披露 |
| SP500 | Pairwise Accuracy (Cross-market) | 0.52 (GPT-5 Zero-shot) | 0.64 (GPT-4.1-Mini SFT) | 0.12 |
| CSI300 | Signal Classification (DeepSeek-V3) | 0.46 | 未披露 | 未披露 |

**解讀:** 
- `0.44 → 0.83` 的 Δ 屬真 capability 躍升，源於 SFT 將任務從「絕對數值預測」降維為「結構相對排序」，避開了金融低信噪比下的欠定問題。
- 跨市場 `0.52 → 0.64` 證明成對比較學習到的算子拓撲特徵具備市場不變性，非過擬合。
- 零樣本分類/成對準確率徘徊於 `0.46-0.52`，證實純文本公式缺乏執行上下文時，LLM 判斷等同隨機基線。CoT 在評分任務引入方差，但在成對選擇微調後可達 `0.86`，屬有效增益而非腦補。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 
- 依賴 Qlib 內建算子集，無法處理自定義複雜衍生品或訂單流數據。
- SFT 需人工構建高 contrast 因子對，標註成本未量化。
- EA 搜索中若 Temperature 過高會觸發 Word Salad（JSON CoT 字段亂碼），需 Parser 端異常攔截。

**6.2 推斷的隱含假設:**
- **Regime 依賴:** 假設算子結構特徵（如窗口交互、防零邏輯）在不同市場/週期下保持穩定，未驗證極端波動率 regime 下的衰減。
- **容量/成本:** 假設 Token 成本可透過 Prompt Caching（命中率 85%）與輕量模型（Flash/Mini）覆蓋；未計入 AST 解析與 Qlib 截面回測的延遲成本。
- **數據泄漏:** Lookahead 懲罰僅覆蓋顯性未來函數（如 Ref 算子），未涵蓋隱性前瞻（如使用未來截面統計量進行當前排序）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| AlphaGen / GP | 符號回歸 vs 語義對齊；黑盒冗雜 vs AST 可解釋 | Open | 成熟但算力消耗大 |
| AutoAlpha | 遺傳算法搜索 vs LLM EA 演化；缺乏金融常識 vs 內建 Taxonomy | Open | 迭代慢，易過擬合 |
| AlphaBench | 成對比較 SFT + 硬代碼過濾器 + 執行元數據補齊 | TBD | v0.5 驗證期 |

🎤 **Interview Tip:** 
- **正確答法:** LLM 不適合零樣本絕對評分，因純公式是欠定問題。實盤應將 LLM 定位為「變異算子生成器」，配合 AST 結構輸入與 SFT 成對比較器進行初篩，最終約束必須交由 C++/Qlib 截面回測引擎。
- **錯答法:** 直接寫 Prompt 讓模型輸出 RankIC 預測值，或依賴 CoT 推理鏈進行因子打分。

**7.1 可證偽預測帶日期:** 2026-Q3 前，基於 AST 表徵的 LLM 因子挖掘將取代純符號回歸成為中頻因子庫主力；若實盤驗證顯示跨市場 Pairwise 準確率跌破 0.55，則證明算子結構特徵存在隱性市場依賴，需引入宏觀狀態變量。

## §8 · For the Reader
- **因子研究員:** 徹底放棄零樣本打分幻想。重構 Prompt 為 AST JSON + 執行元數據（調倉頻/滯後/NaN處理），將 LLM 輸出對齊至算子拓撲而非自然語言。
- **LLM-Agent 開發者:** 實現 Prompt Caching 控制 Token 成本（導讀顯示命中率達 85%）；嚴格限制 Temperature 於 0.75 以平衡收斂速度與合規率；Parser 端必須寫死 AST 深度與前瞻詞彙硬過濾。
- **組合配置/高頻執行:** 將 LLM 視為「創造力注入層」，回測引擎為「物理約束層」。因子入池前必須通過計算時限（如兩週數據 ≤30秒）與 NaN 比例（≤1%）過濾器，避免算子爆炸拖垮實盤延遲。

## References
- 原論文/框架：AlphaBench (2026-05-25, 無 venue)
- Lineage: Alpha101/Alpha158 (人工挖掘) → AutoAlpha/AlphaGen (GP/符號回歸) → AlphaBench (LLM+AST+SFT)
- QuantML 導讀鏈接：[哪款大模型更适合因子挖掘？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247493934&idx=1&sn=6dbb6e8846d1cf5186b7a52de93f783d&chksm=ce7d8e30f90a07262bcece160866ce33dbf9fde5066d2e6e2908bfc17e05b7299219ab1d3b00#rd)