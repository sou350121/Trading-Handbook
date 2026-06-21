<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=因子挖掘 autonomy=人机协同可解释 -->

# FinBERT & LLM Prompting 解構（FinBERT & LLM Prompting）

> **發布**：2024-10-06 · （無 venue）
> **QuantML 導讀**：[使用LLMs和FinBERT对新闻和报告进行金融情绪分析](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486930&idx=1&sn=6e733991443d5f9cf1d3c83424620307&chksm=ce7e6accf909e3dabe5f604aaa6ab70dfed6e0af3faa809bb57dc96d327a8b51e52388f71e3f#rd)
> **核心定位**：落點於「文本另类 × 生成式大模型 × 人机协同可解释」。解了 prior gap：傳統金融 NLP 依賴高昂的領域微調（如 FinBERT）或脆弱的詞典規則，本文驗證了透過精心設計的少樣本提示（Few-shot Prompting），通用 LLM 在金融情緒分類上可逼近領域專用模型，將垂直領域的 Alpha 挖掘從「重微調」轉向「輕提示」。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 驗證 Few-shot Prompting 能否讓通用 LLM（GPT-4o/3.5）在金融新聞情緒分類上媲美領域微調的 FinBERT。② 核心 trick 是注入 9 條高質量金融語境示例（3正/3負/3中）的結構化提示模板。③ 對「生成式大模型」軸★ 意義在於證明垂直領域的 Alpha 挖掘可從「重微調」轉向「輕提示」，大幅降低部署與維護成本。④ 關鍵實證：微調版 FinBERT 準確率達 0.88，GPT-4o Few-shot 性能逼近該基準。

**X-Ray.** 本文本質是 NLP 工程實踐而非新算法發明。它將「提示工程」從黑盒調參拉回可解釋的因子挖掘工作流：金融語境注入直接對應於因子構建中的「領域知識先驗」。在五軸 Pareto 上，它犧牲了 FinBERT 的絕對精度上限，換取了極低的遷移成本與 API 級的可解釋性。解了舊坑：免去了金融語料預訓練與梯度更新的算力開銷，且 Few-shot 示例天然構成可審計的決策路徑。預測其打不開的 Envelope：Financial PhraseBank 僅 4.8k 句，屬靜態、單語種、低頻快照，無法支撐實盤所需的動態流式處理與跨市場泛化；LLM API 的延遲與成本在日頻波段以上會迅速侵蝕 Sharpe。對量化讀者的意義不在於直接上線，而在於驗證了「提示模板=可編程因子」的可行性，為後續構建動態 Few-shot 檢索或 LLM-as-a-Judge 的 Alpha 管道提供基線。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作 (FinBERT / 傳統詞典) | 本文方法 (LLM + Prompt) | 量化影響 |
|---|---|---|---|
| 知識注入 | 領域語料預訓練 + 全量微調 | 靜態 Few-shot 示例注入提示 | 免梯度更新，部署成本↓ |
| 決策路徑 | 黑盒注意力權重 | 提示模板 + 示例對齊 | 可審計，符合合規 |
| 泛化機制 | 依賴訓練集分佈 | 上下文學習 (In-context Learning) | 跨任務遷移快，但需人工設計 |

**1.2 ⚡ Eureka** 用 9 條標註清晰的金融語句作為「上下文錨點」，強制 LLM 在推理時對齊金融語義空間，而非依賴通用語料。

**1.3 信息流**
```
[金融新聞/公告] -> [Prompt 模板] -> [Few-shot 示例庫 (3+/3-/3=)] -> [LLM Inference] -> [情緒標籤/置信度]
        ^                      |
        |_______________________| (人工審計/動態替換示例)
```

## §2 · 數學層
📌 **Napkin Formula**: $P(y|x, \mathcal{D}_{prompt}) = \text{Softmax}(W \cdot \text{LLM}_{\theta}(x \oplus \mathcal{D}_{prompt}))$，其中 $\mathcal{D}_{prompt}$ 為靜態示例集。複雜度：推理 $O(L \cdot d)$，無訓練梯度計算。
**直覺**: 將領域知識編碼為輸入序列的條件分佈，而非更新模型權重 $\theta$。
**Loss/訓練**: 無。LLM 採用 API 調用（Zero/Few-shot）；FinBERT 使用標準 Cross-Entropy Loss 在 Financial PhraseBank 上微調至收斂。

## §3 · 數據層
- **資料規模/頻率/市場/時段**: Financial PhraseBank，4,845 條英文金融新聞句子，靜態快照（來源 LexisNexis，具體年份未披露）。
- **怎麼來**: 16 位金融背景專家手動標註（4類：正/負/中/不確定），含一致性分數。
- **樣本外與容量假設**: 20% Test / 20% Val / ~3,101 Train。屬典型 NLP 分類小樣本設定，不具備實盤時間序列外推能力；容量假設極低，僅驗證分類準確率，未計入交易成本與流動性。

## §4 · 代碼層
| 項目 | 狀態/詳情 |
|---|---|
| Repo | TBD |
| Checkpoint | FinBERT (公開) / GPT-3.5/4o (API) |
| License | FinBERT (Apache 2.0) / OpenAI API (商業授權) |
| 複現難度 | 低（Prompt 模板公開，API 調用即可） |
| 數據可得性 | Financial PhraseBank (公開) |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Financial PhraseBank | Accuracy | FinBERT (0.88) | GPT-4o Few-shot (未披露) | 未披露 |
| Financial PhraseBank | Precision/Recall/F1 | 未披露 | 未披露 | 未披露 |
| Financial PhraseBank | Zero-shot vs Few-shot | 未披露 | Few-shot 顯著優於 Zero-shot | 定性提升 |

**解讀**: Δ 主要來自上下文學習對金融歧義的消解，屬真實 capability 提升；但樣本僅 4.8k 且為靜態分類，未計入 LLM API 延遲與成本，實盤 Sharpe/IR 無法從該表推斷。存在樣本選擇偏差（僅高共識句子）與前瞻偏差風險（若示例庫含未來信息）。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**: 依賴特定提示設計；LLM 版本更新會導致結果波動；零樣本缺乏上下文導致精度不足；FinBERT 仍為精度上限。
**6.2 推斷的隱含假設**: Regime 依賴高（提示模板需手動維護以適應新金融術語）；容量極低（API 限流與成本不支援高頻/大規模掃描）；數據泄漏風險（Few-shot 示例若與測試集語義重疊會虛高準確率）；Survivorship bias（LexisNexis 歷史數據可能已過濾破產/退市公司）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| FinBERT (Domain Fine-tune) | 權重更新 vs 上下文學習 | 是 | 成熟/基準 |
| LLM-as-a-Judge (Chain-of-Thought) | 靜態分類 vs 動態推理評分 | 部分 | 實驗/高成本 |
| 傳統詞典/ML (Loughran-McDonald) | 規則驅動 vs 語義驅動 | 是 | 過時/低維 |

🎤 **Interview Tip**: 正確答：Few-shot 的本質是 In-context Learning，它不改變模型參數，而是通過輸入序列的條件分佈引導推理，適合低頻、高解釋性要求的因子挖掘；錯答：認為 Few-shot 能完全替代微調，或忽略 API 成本與延遲對實盤容量的毀滅性影響。
**7.1 可證偽預測**: 若 2025-Q3 前 OpenAI 推出原生金融垂直模型或 Few-shot 檢索自動化框架成熟，本文手動 Prompt 設計的工作流將被動態 RAG 管道取代，靜態 Few-shot 準確率優勢將歸零。

## §8 · For the Reader
- **因子研究員**: 將 Few-shot 示例視為「可編程因子」，建立示例版本控制與績效歸因，避免提示漂移。
- **高頻執行**: 無直接適用性；LLM 推理延遲（>500ms）與成本不支援日頻以下頻次，僅適合盤後情緒快照。
- **組合配置**: 可將 LLM 輸出情緒分數作為風險預算調整的宏觀濾鏡，但需對齊交易成本與滑點模型。
- **LLM-agent/RL 策略**: 探索動態示例檢索（Dynamic Few-shot）或將情緒分類作為 Reward Model 的基礎信號。
- **研究學生**: 專注於提示模板的自動化搜索（Prompt Optimization）與跨語言/跨市場泛化實驗。

## References
- QuantML 導讀：[使用LLMs和FinBERT对新闻和报告进行金融情绪分析](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486930&idx=1&sn=6e733991443d5f9cf1d3c83424620307&chksm=ce7e6accf909e3dabe5f604aaa6ab70dfed6e0af3faa809bb57dc96d327a8b51e52388f71e3f#rd)
- 原論文/框架：FinBERT (Araci, 2019) / OpenAI GPT-3.5/4o API Docs
- 數據集：Financial PhraseBank (Malo et al., 2014)