<!-- ontology-5axis data=文本另类 horizon=跨周期 paradigm=生成式大模型 alpha=端到端表征 autonomy=人机协同可解释 -->

# SusGen-GPT 解構

> **發布**：2024-12-19 · （無 venue）
> **QuantML 導讀**：[SusGen-GPT：大模型生成ESG报告](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247488448&idx=1&sn=86125b3ac559b59faff52a751134c2f8&chksm=ce7e74def909fdc8082cd6b688d1687836fa3df6d6544eba26a69b3d57c2e8c3db4fbd6e1af6#rd)
> **核心定位**：落點於「文本另类 × 生成式大模型」軸，解決金融/ESG領域開源模型缺乏領域知識與數據類別不平衡的 prior gap，以數據中心策略實現 7-8B 模型逼近閉源 1,700B 模型效能。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `跨周期` | `生成式大模型` | `端到端表征` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 構建均衡多任務指令集 SusGen-30K 與 TCFD-Bench 基準。② 核心 trick 為 QLoRA 高效微調結合 RAG 檢索非結構化年報。③ 對「端到端表征」軸的關鍵在於以極低算力代價維持通用能力並注入金融/ESG 專項知識。④ 關鍵實證數字：在 FiQASA 情感分析上 F1 達 0.72，TCFD-Bench 報告生成 Rouge-L 達 0.20（均為 NLP 指標，非交易 Alpha）。

**X-Ray.** 在參數規模與領域效能的 Pareto 前沿上，該框架選擇犧牲通用推理極限（如 NER 僅 0.35 vs GPT-4 0.83），換取 ESG 報告生成的結構化合規能力。解了舊工程坑：數據類別不平衡與非結構化年報提取的幻覺問題。打不開的 envelope：無法處理高頻/低延遲交易信號，僅限於低頻文檔解析與合規披露。對量化讀者意義：提供可嵌入因子流水線的 ESG 文本特徵提取模組，但需警惕其作為生成式模型的確定性偏差與檢索延遲。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作/基線 (通用LLM/FinLLM/ChatReport) | SusGen-GPT 改動 |
|---|---|---|
| 數據構建 | 原始爬蟲/混合語料/類別失衡 | 數據中心主義：類別均衡下採樣 + 多語言翻譯增強 |
| 微調策略 | 全參數微調或標準 LoRA | QLoRA 4-bit 量化 + Paged AdamW，凍結基座保留通用能力 |
| 生成機制 | 端到端黑盒生成 | RAG 檢索非結構化年報 + 預定義 TCFD 提示模板組合 |

**1.2 ⚡ Eureka 一句話 trick + 直覺**
以數據中心主義的類別均衡下採樣，強制模型在 7-8B 容量內重構金融語義流形，將生成任務轉化為條件機率最大化而非純粹參數記憶。

**1.3 信息流 ASCII 圖**
```
Input(年報/金融文本) 
  → RAG Chunk(1024 tok) 
  → Vector DB (all-mpnet-base-v2) 
  → Top-10 Context Retrieval 
  → Prompt(Alpaca Template) 
  → QLoRA 7-8B (4-bit) 
  → Output(TCFD Report / Label)
```

## §2 · 數學層
📌 **Napkin Formula**
`L_total = L_SFT(θ_0 + Δθ_LoRA) + λ * L_RAG(Context, Query)`
複雜度：`O(N * d^2 * r)`，其中 `r=16` (lora rank)，`d` 為 hidden dim。
直覺：QLoRA 凍結基座權重，僅更新低秩投影矩陣，結合 RAG 的交叉注意力機制，將生成任務轉化為條件機率最大化。Loss/訓練細節：Paged AdamW, cosine LR, 3 epochs, lr=2e-5, batch=8, grad_acc=8, max_len=2048, dropout=0.1, 4-bit quant, bfloat16。

## §3 · 數據層
資料規模：SusGen-30K（30,000 條指令）。頻率/市場：跨市場金融文本與 ESG 年報。時段：未披露。怎麼來：HuggingFace 開源集、TCFDHub 年報、網路爬蟲、GPT-4o 輔助生成與人工校對。樣本外與容量假設：TCFD-Bench 為獨立測試集；文本處理為主，無交易頻率限制，但受 RAG 檢索延遲與生成帶寬限制，僅適用低頻週期。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 中（需兩個NVIDIA RTX 24GB 3090 Ti GPU，約10小時） | 高（SusGen-30K 與 TCFD-Bench 已發布） |

## §5 · 評測 / Benchmark
| 數據集/任務 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| FinQA (金融問答) | Exact Match/F1 | GPT-4 0.63 | 0.57 | -0.06 |
| TATQA (表格問答) | Exact Match/F1 | 未披露 | 0.80 | 未披露 |
| FiQASA (情感) | F1 | GPT-4 0.70 | 0.72 | +0.02 |
| FOMC (情感) | F1 | GPT-4 0.71 | 0.70 | -0.01 |
| MultiFin (標題分類) | MicroF1 | GPT-4 0.65 | 0.52 | -0.13 |
| MLESG (ESG分類) | MicroF1 | 未披露 | 0.51 | 未披露 |
| NER (實體識別) | Entity F1 | GPT-4 0.83 | 0.35 | -0.48 |
| FINER-ORD | Entity F1 | GPT-4 0.77 | 0.18 | -0.59 |
| SC (關係提取) | F1 | 未披露 | 0.96 | 未披露 |
| FinRED | F1 | 未披露 | 0.23 | 未披露 |
| TCFD-Bench | Rouge-L | ChatReport 0.14 | 0.20 | +0.06 |
| TCFD-Bench | BERTScore | ChatReport 0.32 | 0.40 | +0.08 |
| TCFD-Bench | METEOR | ChatReport 0.12 | 0.27 | +0.15 |
| TCFD-Bench | BLEU-1 | ChatReport 0.41 | 0.39 | -0.02 |

**解讀**：Δ 為正的多見於情感分類與 ESG 報告生成，反映模型在特定領域指令微調後確實捕捉到領域語義。Δ 為負且幅度大（如 NER -0.48/-0.59）顯示 7-8B 模型在複雜結構化抽取上仍受容量限制，屬模型架構瓶頸而非過擬合。TCFD-Bench 的 BLEU-1 略低但 Rouge-L/BERTScore 顯著領先，表明生成內容在語義重疊與長程連貫性上優於 ChatReport，但精確詞彙匹配稍弱。所有測試均為靜態基準，未計入 RAG 檢索延遲與推理成本，實戰中需評估延遲對低頻因子週期的影響。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：缺乏針對高頻/低延遲場景優化；開源模型在複雜推理（如 NER）仍落後閉源；數據依賴於現有公開報告與爬蟲。
**6.2 推斷的隱含假設**：Regime 依賴於企業披露合規性與文本格式穩定性；容量受限於 RAG 上下文窗口與生成帶寬；成本隱含於 GPU 推理與向量資料庫維護；數據泄漏風險低（明確區分訓練/測試集）；無 survivorship bias 但依賴 TCFDHub 收錄公司，可能偏向已披露 ESG 的大型企業。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| BloombergGPT | 參數規模(50B)/閉源/通用金融 | 否 | 商用 |
| FinGPT | 數據工具鏈/開源/多任務 | 是 | 活躍 |
| ChatReport | 任務聚焦(ESG總結)/閉源基座 | 否 | 靜態 |

🎤 **Interview Tip**：正確答法應指出 QLoRA+RAG 是「參數效率與領域知識注入的權衡」，錯答法將生成式文本相似度指標直接等同於 Alpha 預測能力。
**7.1 可證偽預測帶日期**：若 2025-Q3 前未引入動態圖神經網絡處理供應鏈 ESG 關聯，該框架在跨公司風險傳導分析上的 F1 將停滯於 0.23 附近。

## §8 · For the Reader
- **因子研究員**：將 TCFD-Bench 輸出轉化為結構化 ESG 因子，需校準文本生成與財務報表的時間戳錯配。
- **高頻執行**：不適用，RAG 檢索與生成延遲無法滿足毫秒級需求。
- **組合配置**：用於低頻再平衡前的合規審查與風險披露解析，可作為輔助決策模組。
- **LLM-agent**：參考其數據均衡縮放策略，解決多任務指令微調時的災難性遺忘。
- **研究學生**：複現時優先驗證 10k/20k/30k 消融實驗，理解數據質量 vs 數量的邊際效應。

## References
- 原論文：SusGen-GPT (2024)
- Lineage：QLoRA (Dettmers et al., 2024) / RAG (Lewis et al., 2020) / ChatReport (Ni et al., 2023)
- QuantML 導讀鏈接：[SusGen-GPT：大模型生成ESG报告](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247488448&idx=1&sn=86125b3ac559b59faff52a751134c2f8&chksm=ce7e74def909fdc8082cd6b688d1687836fa3df6d6544eba26a69b3d57c2e8c3db4fbd6e1af6#rd)