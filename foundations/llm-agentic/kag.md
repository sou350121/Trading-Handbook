<!-- ontology-5axis data=多模态 horizon=跨周期 paradigm=生成式大模型 alpha=端到端表征 autonomy=Agent自主演进 -->

# KAG 解構

> **發布**：2025-03-06 · （無 venue）
> **QuantML 導讀**：[KAG + 多模态 RAG + LLM Agents：构建强大的AI推理系统](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489565&idx=1&sn=ee48872ce61ab0e5cd53e99dc108c61e&chksm=ce7e7f03f909f61594b7b1b248716b4260b63a81667725a5432961d6f79d998120aabf66b073#rd)
> **核心定位**：落點於「生成式大模型 × Agent自主演进」軸，解了傳統 RAG 依賴向量相似度導致邏輯鏈断裂與幻覺放大的 prior gap。將符號邏輯（KG）與神經生成（LLM）透過互索引與混合算子鏈硬編碼，提供可追蹤的推理路徑而非黑盒機率輸出。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `多模态` | `跨周期` | `生成式大模型` | `端到端表征` | `Agent自主演进` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① KAG 將知識圖譜、多模態 RAG 與 LLM Agents 串接為統一推理服務框架。② 核心 trick 在於圖塊互索引結構與邏輯形式引導的混合推理引擎，將自然語言轉譯為可組合的規劃/推理/檢索算子。③ 對端到端表征軸的價值在於以符號約束替換純語義匹配，降低特徵提取的隨機方差。④ 導讀未給量化結果。

**X-Ray.** KAG 不是直接產生 Alpha 的預測模型，而是特徵工程與邏輯驗證的基礎設施層。它把 RAG 從「機率檢索」推向「確定性圖遍歷」，用專家規則與三元組約束 LLM 的候選空間，直接命中量化研究中「非結構化資訊轉結構化因子」的痛點。其 Pareto 邊界卡在混合推理的延遲疊加與 KG 維護成本：圖結構越密，檢索越準，但 API 呼叫與算子調度時間呈線性膨脹。對實盤而言，它打不開高頻 envelope，但極適合作為中低頻信號的邏輯過濾層或事件驅動因子的自動建構管道。框架能力不取決於模型參數量，而取決於領域 Schema 的完備度與對齊品質。

## §1 · 架構 / Core Mechanism
| 改動維度 | 傳統 RAG | KAG 架構 |
|---|---|---|
| 檢索機制 | 向量相似度匹配 | 知識圖譜與文本塊互索引倒排 |
| 推理路徑 | 純語義生成/提示詞鏈 | 邏輯形式引導的混合算子（規劃/推理/檢索/計算） |
| 系統閉環 | 靜態問答 | Agent 自主任務規劃 + 動態知識反饋更新 |

⚡ **Eureka:** 用符號邏輯將自然語言問題轉譯為可執行圖遍歷路徑，讓 LLM 從「猜詞補全」變成「解題演算」。

**信息流 ASCII:**
```
User Query → Intent/Entity Parse → Logic Form Generation
       ↓
Hybrid Engine [Plan | Reason | Retrieve | Compute]
       ↓
KG ↔ Text Mutual Index → Multimodal Alignment
       ↓
Agent Execution → Answer Generation → Feedback Loop (KG Update)
```

## §2 · 數學層
📌 **Napkin Formula:** 導讀未披露具體損失函數或梯度更新公式。框架運作可抽象為算子組合映射：
$$Q \xrightarrow{\text{Parse}} \mathcal{L}(Q) \xrightarrow{\text{Hybrid}} \bigoplus_{i \in \{R, KG, L, C\}} \mathcal{O}_i(\mathcal{L}, \mathcal{G}, \mathcal{T}) \rightarrow \hat{A}$$
**直覺:** 將檢索與生成解耦為可組合的算子鏈，透過圖結構約束候選空間，降低 LLM 的幻覺方差。複雜度取決於跳數與算子調度次數，非端到端反向傳播。
**Loss/訓練細節:** 導讀未披露微調目標或對比損失。系統依賴預訓練 LLM 與 OpenSPG 引擎的管道串接，屬推理時動態路由而非訓練時聯合優化。

## §3 · 數據層
- **規模/頻率/市場:** 導讀未披露具體數據規模、頻率或適用市場。僅提及支援非結構化（新聞/日誌）、結構化（交易/統計）與專家規則，並可整合圖像/音頻/視頻。
- **來源與樣本外:** 資料由使用者上傳（如含圖表的 PDF）或對接業務庫；樣本外驗證、容量假設與數據切分策略均未披露。框架設計為垂直領域私有庫，非公開金融 benchmark。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | `https://github.com/OpenSPG/openspg`（導讀 docker-compose 來源） |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（需 Docker 環境 + 自建/對接 KG + 外部 LLM API） |
| 數據可得性 | 私有/業務庫（導讀未提供公開金融數據集） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 未披露 | IR/Sharpe/MDD/AR | 未披露 | 未披露 | 未披露 |

**解讀:** 導讀未給任何量化指標或基線對比數字。此框架屬基礎設施層，其「能力」體現在推理路徑的確定性與多模態對齊率，而非直接交易收益。Δ 欄全數標未披露，嚴格遵循數字紀律。若未來接入實盤，需警惕：圖遍歷與多模態對齊帶來的延遲可能吃掉中低頻 Alpha 的滑點預算；且邏輯形式生成若依賴特定 Prompt 模板，易產生前瞻偏差或樣本內過擬合。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 導讀明確指出框架「目前處於早期開發階段」，邏輯推理路徑與對話任務支援待優化，kag-model 尚未完全開源，前端與分布式構建仍處規劃中。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 高度依賴領域專家構建的 KG Schema（模式約束）。市場結構或會計準則變更時，圖規則需手動重構，否則推理路徑失效。
- **容量/成本:** 混合推理引擎含規劃+圖遍歷+LLM生成，API 成本與延遲隨跳數膨脹，不適合高頻或低預算場景。
- **數據泄漏/生存者偏差:** 互索引機制若未嚴格隔離訓練/推理時間窗口，易將未來事件節點提前注入圖結構，產生隱性前瞻。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Traditional RAG | 向量檢索 vs 圖塊互索引+邏輯算子鏈 | 多數開源 | 成熟但幻覺率高 |
| FinRobot / TradingAgents | 純 LLM 工具調度 vs KG 約束的混合推理 | 開源 | 活躍但缺乏符號邏輯層 |
| 傳統因子庫 (BARRA/TA-Lib) | 靜態數學公式 vs 動態多模態知識對齊 | 閉源/開源混合 | 穩定但泛化弱 |

🎤 **Interview Tip:** 
- **正確答:** KAG 的核心不是替換 LLM，而是用 KG 的倒排互索引與邏輯形式將非結構化資訊轉為可驗證的推理路徑。它解決的是特徵提取的確定性問題，而非直接預測收益率。
- **錯答:** 「KAG 比傳統 RAG 準確率更高，所以能直接拿來做选股模型。」（混淆了推理基礎設施與預測模型，忽略延遲與 Schema 維護成本）

**7.1 可證偽預測帶日期:** 若 2025Q3 前 kag-model 未開源或混合推理端到端延遲未降至 <2s，則該框架在實盤 Alpha 提取場景的機構採用率將低於預期。

## §8 · For the Reader
- **因子研究員:** 用 KG 互索引替代純文本 Embedding，可將財務報表/新聞事件自動轉為結構化三元組，直接輸出為可回測的因子庫，減少人工清洗偏差。
- **高頻執行:** 框架含 LLM 生成與圖遍歷，延遲過高，不適用；僅適合盤後邏輯驗證或中低頻信號過濾層。
- **LLM-Agent/RL 策略:** 將 KAG 的混合推理引擎作為 Agent 的「世界模型」或工具調度器，可大幅降低 RL 探索空間的隨機性，提升策略收斂穩定度。

## References
- QuantML 導讀: [KAG + 多模态 RAG + LLM Agents：构建强大的AI推理系统](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489565&idx=1&sn=ee48872ce61ab0e5cd53e99dc108c61e&chksm=ce7e7f03f909f61594b7b1b248716b4260b63a81667725a5432961d6f79d998120aabf66b073#rd)
- Framework Repo: `https://github.com/OpenSPG/openspg`
- Lineage: Retrieval-Augmented Generation (RAG) → Knowledge Graph Augmentation → Multimodal RAG → LLM Agents with Tool/Logic Routing