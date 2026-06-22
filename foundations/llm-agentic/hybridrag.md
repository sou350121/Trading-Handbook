<!-- ontology-5axis data=文本另类 horizon=跨周期 paradigm=生成式大模型 alpha=因子挖掘 autonomy=人机协同可解释 -->

# HybridRAG 解構

> **發布**：2024-08-15 · （無 venue）
> **QuantML 導讀**：[贝莱德&英伟达：HybridRAG结合GraphRAG以及VectorRAG的新型RAG系统](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247485798&idx=1&sn=04b56d6bd5864325ee5b4195b88007c1&chksm=ce7e6e78f909e76e3ed448882a568bfb2c453d91fa76339713dbf5e89c3d23353107c4b57b34#rd)
> **核心定位**：落點於文本另类與生成式大模型軸，解決單一檢索在金融非結構化文本中「廣度與深度不可兼得」的 prior gap，透過雙路上下文拼接提升 LLM 財務問答的結構化推理與語義覆蓋。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `跨周期` | `生成式大模型` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 融合 GraphRAG 與 VectorRAG 構建雙路檢索管道，專攻金融財報與電話會議問答。② 核心 trick 為「向量語義抓廣度、圖譜結構抓深度」，按序拼接上下文後輸入 LLM 生成。③ 對「因子挖掘」軸的價值在於將非結構化管理層語氣與關聯實體轉化為可程式化提取的結構化特徵。④ 導讀給出 HybridRAG 在忠實度與答案相關性均達 0.96，上下文召回率達 1。

**X-Ray.** HybridRAG 將 RAG 的檢索邊界從「語義相似度」推至「知識圖譜三元組」，本質上是把 LLM 的幻覺風險轉嫁給圖資料庫的結構約束。在量化五軸 Pareto 中，它犧牲了部分上下文精確度（0.79），換取了召回率與答案相關性的頂格表現。這解了傳統 VectorRAG 在財務專有名詞與跨段落因果鏈上的斷裂坑，但打不開「實時數值計算」與「多模態圖表解析」的 envelope。對量化讀者而言，此架構不是直接產出 Alpha 的模型，而是「非結構化因子」的標準化預處理器：它能把電話會議中的管理層指引、供應鏈關聯與風險提示，穩定轉譯為可回測的文本特徵矩陣。後續需警惕 LLM 鏈式提取的延遲成本與提示工程漂移。

## §1 · 架構 / Core Mechanism
| 改動維度 | VectorRAG (前作) | GraphRAG (前作) | HybridRAG (本法) |
|---|---|---|---|
| 檢索機制 | 向量語義相似度搜索 | 圖節點與邊的關係遍歷 | 雙路並行檢索，上下文按序拼接 |
| 上下文來源 | 文本塊 (Chunks) | 知識圖譜三元組 (Subgraph) | 向量上下文 + 圖譜上下文 |
| 生成依賴 | LLM 內部知識 + 檢索文本 | LLM 內部知識 + 圖結構 | LLM 內部知識 + 融合上下文 |

**⚡ Eureka:** 向量檢索負責「找全」，圖譜檢索負責「找準」，兩路上下文不經複雜權重學習，直接按 `Vector → Graph` 順序拼接輸入 LLM，以結構化提示工程替代端到端訓練。

**信息流:**
```
Query → [VectorDB: Pinecone] → Context_V → \
                                  → [Concatenation] → [LLM: GPT-3.5-turbo] → Answer
Query → [KG: Networkx]    → Context_G → /
```

## §2 · 數學層
📌 Napkin Formula:
`C_final = Concat(Context_V(q), Context_G(q))`
`P(y|q, C_final) = LLM(q, C_final)`
複雜度：檢索階段 O(N) 線性掃描或圖遍歷，生成階段 O(L) 依賴 LLM 推理。無端到端梯度下降，依賴兩階段 LLM 鏈 (摘要生成 → 實體/關係提取) 與提示工程。
直覺：將檢索問題解耦為「語義覆蓋」與「結構推理」兩個正交子空間，透過上下文拼接強制 LLM 在生成時同時參考廣義語料與精確關聯，降低財務專有名詞的錯配率。

## §3 · 數據層
- **市場/時段**：印度 Nifty 50 指數成分股，2024年第一季度收益電話會議記錄。
- **來源/規模**：自研爬蟲抓取公司官網財報與問答環節。精選 400 個 Q&A 對用於評估。
- **樣本外假設**：導讀未披露跨季度或跨市場驗證，預設為單一市場單一季度截面，容量假設受限於 LLM 上下文窗口與圖構建延遲。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | TBD |
| Checkpoint | 依賴 OpenAI GPT-3.5-turbo 與 text-embedding-ada-002 |
| License | TBD |
| 複現難度 | 中 (需配置 Pinecone, LangChain, Networkx 與 LLM API) |
| 數據可得性 | 低 (自研爬蟲，未開源原始 PDF 與 Q&A 集) |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Nifty 50 Q1 2024 Q&A | Faithfulness | 未披露 | 0.96 | 未披露 |
| Nifty 50 Q1 2024 Q&A | Answer Relevance | 未披露 | 0.96 | 未披露 |
| Nifty 50 Q1 2024 Q&A | Context Precision | 未披露 | 0.79 | 未披露 |
| Nifty 50 Q1 2024 Q&A | Context Recall | 未披露 | 1 | 未披露 |

**解讀：** 導讀僅提供 HybridRAG 的絕對得分，未給出基線具體數值，故 Δ 欄留白。0.96 的忠實度與相關性反映 LLM 在給定上下文後能穩定遵循指令，但 Context Precision 僅 0.79 且導讀明確指出「低於 GraphRAG」，暗示向量檢索引入的語義廣度可能混入無關文本塊，產生前置偏差。此為檢索-生成解耦架構的常見 trade-off，非模型過擬合，但實盤需計入 LLM API 調用成本與上下文窗口浪費。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：導讀提及未來需整合數值數據分析、多模態輸入與實時數據流，暗示當前系統僅處理靜態文本，缺乏財務報表數值計算能力，且評估指標未覆蓋實時性與計算準確度。
**6.2 推斷的隱含假設**：
- **Regime 依賴**：圖譜三元組提取高度依賴提示工程與 LLM 版本，模型更新可能導致實體消歧結果漂移。
- **容量/成本**：雙路檢索 + 兩階段 LLM 鏈 + 最終生成，API 成本與延遲顯著高於單路 VectorRAG，不適合高頻或低延遲場景。
- **數據泄漏/倖存者偏差**：僅覆蓋 Nifty 50 成分股，未處理退市或財務異常公司，回測樣本存在倖存者偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 VectorRAG | 檢索維度 (語義 vs 語義+結構) | Open | 成熟 |
| 純 GraphRAG | 上下文覆蓋率 (結構推理 vs 廣度+深度) | Open | 實驗 |
| 微調 LLM (SFT) | 訓練範式 (提示拼接 vs 權重更新) | 閉源/開源皆有 | 生產 |

**🎤 Interview Tip**：
- ✅ 正確答：「HybridRAG 的核心價值不在於替代 LLM，而在於將非結構化金融文本的檢索問題轉化為結構化約束問題。向量抓召回，圖譜抓精確，拼接順序決定了 LLM 的注意力偏向。實盤需權衡 API 成本與上下文精確度的 trade-off。」
- ❌ 錯答：「它比 VectorRAG 好是因為用了知識圖譜，所以什麼指標都更高。」（忽略 Context Precision 下降的 trade-off 與成本結構）

**7.1 可證偽預測**：若 2025-Q2 前未引入自動化數值解析模塊，該架構在處理財報數字問答（如 EPS 預測、毛利率計算）時，Context Precision 將持續低於 0.8，無法直接替代傳統數值因子模型。

## §8 · For the Reader
- **因子研究員**：將電話會議中的管理層語氣與供應鏈關係轉化為二值/連續文本因子，需先跑通實體消歧管道，避免提示漂移污染因子 IC。
- **高頻執行**：此架構延遲過高，不適用。僅適合日頻/週頻的資訊驅動型 Alpha 預處理。
- **LLM-agent / 研究學生**：學習雙路檢索的上下文拼接策略，嘗試將 Networkx 圖遍歷替換為自研圖神經網路嵌入，降低對閉源 LLM 的依賴。

## References
- QuantML 導讀：[贝莱德&英伟达：HybridRAG结合GraphRAG以及VectorRAG的新型RAG系统](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247485798&idx=1&sn=04b56d6bd5864325ee5b4195b88007c1&chksm=ce7e6e78f909e76e3ed448882a568bfb2c453d91fa76339713dbf5e89c3d23353107c4b57b34#rd)
- Lineage：VectorRAG (Lewis et al., 2020) → GraphRAG (Microsoft, 2024) → HybridRAG (本文)