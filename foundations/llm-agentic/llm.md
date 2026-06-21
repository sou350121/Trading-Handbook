<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=端到端表征 autonomy=Agent自主演进 -->

# 从新闻到预测：基于LLM的时间序列预测 解構（从新闻到预测：基于LLM的时间序列预测）

> **發布**：2024-10-15 · （無 venue）
> **QuantML 導讀**：[从新闻到预测：基于LLM的时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487049&idx=1&sn=7337916648f291ef40e325faa83b0aca&chksm=ce7e6957f909e0414601ededa9a7c3957a1c31f0360a79398c097b4e9db731345fa970f93ea1#rd)
> **核心定位**：將時間序列預測從「純數值自回歸」推向「條件生成式建模」，以雙Agent閉環替代靜態NLP特徵工程，解了非結構化文本入模時的對齊損耗與動態衰減Prior Gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 用推理與評估雙Agent閉環自動篩選新聞，以預測殘差反饋迭代優化過濾邏輯，將文本轉為條件輸入。② 核心trick是「預測誤差驅動的新聞選擇動態演化」，打破傳統靜態情緒因子的高衰減坑。③ 對`Agent自主演进`軸★，證明LLM可作為動態特徵選擇器而非僅是預測頭。④ 關鍵實證：在電力/匯率/加密貨幣上顯著優於純數值基線，但交通場景因新聞粒度不匹配改善有限（具體IR/Sharpe未披露）。

**X-Ray.** 本文在五軸Pareto中明確卡位`文本另类`與`Agent自主演进`。傳統量化工程常困於靜態NLP因子（情緒得分、TF-IDF）的過擬合與快速衰減，本文以雙Agent閉環替代人工特徵工程：推理Agent負責新聞分類與邏輯生成，評估Agent以預測誤差為信號反饋迭代過濾規則。這實質上構建了一個動態的「事件-序列」對齊器，解決了非結構化數據入模時的對齊損耗與前瞻偏差控制問題。然而，其Envelope受限於LLM上下文窗口與新聞來源的地理/行業粒度匹配度。對量化讀者而言，此架構的價值不在於直接替換預測頭，而在於提供一套可插拔的`Event-Driven Feature Selection`範式；若結合高頻訂單簿或盤中微結構數據，需警惕LLM推理延遲與API成本對實盤滑點的侵蝕，且該方法僅在人類/市場活動驅動Regime下有效，純物理序列將觸發失效邊界。

## §1 · 架構 / Core Mechanism
| 維度 | 前作/傳統基線 | 本方法改動 | 量化意義 |
|:---|:---|:---|:---|
| 特徵來源 | 靜態NLP情緒/關鍵詞/TF-IDF | 動態新聞流 + 雙Agent閉環篩選 | 解決特徵衰減快、需人工重構的維護成本 |
| 對齊機制 | 人工標註/規則匹配/靜態權重 | 預測殘差反饋驅動的新聞邏輯迭代 | 將「事件-序列」對齊轉為可優化的條件生成問題 |
| 建模範式 | 數值自回歸/統計或CNN/LSTM | LLM條件生成（數值Token化 + 新聞Prompt） | 保留物理尺度（無歸一化），利用LLM先驗處理非線性衝擊 |

⚡ **Eureka 一句話 trick**：預測誤差驅動新聞過濾邏輯的閉環迭代（Prediction-Feedback Loop），讓模型自己決定「哪些新聞該被記住」。
🔀 **信息流 ASCII**：
```
Raw News → [Reasoning Agent] → Filtered/Classified News (JSON)
                                      ↓
[LLM Predictor] ← Conditioned Generation (Num Tokens ⊕ News Tokens)
                                      ↓
[Eval Agent] → Compare Pred vs Actual → Identify Omission/Error
                                      ↓
Feedback Signal → Refine Reasoning Logic → Loop until Convergence
```

## §2 · 數學層
📌 **Napkin Formula**：
$$P(x_{t+1} \mid x_{1:t}, \mathcal{N}_{sel}) = \text{LLM}_{\phi}\big(\text{tokenize}(x_{1:t}) \oplus \text{prompt}(\mathcal{N}_{sel})\big)$$
$$\mathcal{N}_{sel}^{(k+1)} = \text{Agent}_{reason}\big(\mathcal{N}_{raw}, \text{Feedback}(\text{Loss}^{(k)})\big)$$
**複雜度**：單步前向 $O(L \cdot V)$（$L$為上下文長度，$V$為詞表），但實際瓶頸在LLM推理吞吐與Agent迭代次數。
**直覺**：將數值序列與篩選新聞統一映射為Token序列，以條件概率生成下一時刻數值。無歸一化保留物理尺度，避免新聞事件與原始數據值之間的尺度依賴關係被抹平。
**Loss/訓練細節**：標準Next-Token Prediction Loss (Cross-Entropy)。新聞作為條件Prompt注入，微調階段未披露具體優化器、學習率調度與凍結策略（TBD）。

## §3 · 數據層
- **規模/頻率/市場**：交通(CA)、匯率、比特幣、澳洲電力(AEMO)。頻率涵蓋日/小時/半小時。時段更新至2022年。
- **來源與處理**：公開新聞API/網絡爬取（未披露具體來源管道、清洗規則與時間戳對齊精度）。
- **樣本外與容量假設**：假設新聞與序列存在因果/相關對齊，但未驗證樣本外穩定性與長週期Regime切換下的容量限制。上下文窗口限制直接制約多序列並行處理能力。

## §4 · 代碼層
| 欄位 | 狀態 |
|:---|:---|
| Repo | `TBD` |
| Checkpoint | LLaMA 2 (Base) |
| License | `TBD` |
| 複現難度 | 高（需Agent框架+LLM推理+數據對齊+閉環調試） |
| 數據可得性 | 中（公開TS數據易得，新聞數據質量與API成本`未驗證`） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|:---|:---|:---|:---|:---|
| 澳洲電力 | MSE/MAE | 未披露 | 未披露 | 未披露 |
| 匯率/比特幣 | MSE/MAE | 未披露 | 未披露 | 未披露 |
| 交通(CA) | MSE/MAE | 未披露 | 未披露 | 未披露 |

**解讀**：Δ主要來自「事件衝擊期」的條件補償，非穩態下的持續Alpha。交通場景改善有限暴露新聞粒度與目標序列空間分辨率錯配，屬數據對齊偏差風險。未計API推理成本、延遲與算力開銷，實盤Sharpe需扣除執行摩擦後重算。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：上下文窗口限制長序列/多序列處理；不適合純物理/氣象數據（人類活動影響小）；增強而非取代傳統分類/插值任務。
**6.2 推斷的隱含假設**：
- **Regime依賴**：僅在人類/市場活動顯著驅動趨勢的市場有效，低流動性或純算法驅動Regime下新聞信號將退化為噪聲。
- **容量/成本**：Agent迭代與LLM推理引入高延遲與API成本，不適合高頻或低延遲執行。
- **數據泄漏**：新聞發布時間與序列時間戳對齊若不嚴謹，評估Agent的反饋可能引入前瞻偏差。
- **過擬合風險**：Agent過濾邏輯可能過度適配特定新聞源或歷史事件分佈，跨市場遷移需重新訓練閉環。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|:---|:---|:---|:---|
| FinBERT/LLM-as-Feature | 靜態情緒得分 vs 動態殘差反饋閉環 | 部分開源 | 成熟但衰減快 |
| Time-LLM / Timer | 純數值Token化 vs 新聞條件生成 | 開源 | 基線強，缺事件對齊 |
| 本方法 | 預測誤差驅動的新聞邏輯演化 | `TBD` | v0.5 驗證中 |

🎤 **Interview Tip**：
- ✅ **正確答法**：強調Agent閉環解決的是「特徵選擇的動態衰減」與「事件-序列對齊損耗」，LLM在此是條件生成器與動態過濾器，而非直接替換統計預測頭。實盤需離線預計算新聞條件以避開推理延遲。
- ❌ **錯答法**：認為LLM能處理所有高頻數據、或該框架已解決前瞻偏差與API成本問題。

**7.1 可證偽預測**：若於 `2025-12-31` 前引入實時新聞流與低延遲推理部署，該框架在流動性較低的加密貨幣/外匯市場將因滑點與API成本導致實盤Sharpe下降>20%（`未驗證`，需回測扣除執行成本後確認）。

## §8 · For the Reader
- **因子研究員**：將評估Agent的殘差反饋機制抽象為動態權重因子，替代靜態情緒得分；構建「新聞覆蓋率×預測誤差」的交互因子。
- **高頻執行**：警惕LLM推理延遲（通常>100ms），此架構僅適用於日頻/波段。實盤需離線預計算新聞條件，線上僅做條件查詢與快速預測。
- **組合配置/LLM-agent**：將雙Agent解耦為獨立的「事件篩選器」與「預測頭」，便於接入現有RL或Portfolio優化器；利用新聞條件做動態風險預算調整。
- **研究學生**：聚焦「新聞粒度-序列空間分辨率」匹配問題，嘗試用圖神經網絡或局部空間特徵修正交通場景的對齊偏差；探索多模態新聞（圖表/短視頻）入模的Token化方案。

## References
- 原論文：《从新闻到预测：基于LLM的时间序列预测》（無 venue/arXiv，框架名如上）
- Lineage：LLaMA → Time-LLM/Timer → Agent-based TS Forecasting
- QuantML 導讀：[从新闻到预测：基于LLM的时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487049&idx=1&sn=7337916648f291ef40e325faa83b0aca&chksm=ce7e6957f909e0414601ededa9a7c3957a1c31f0360a79398c097b4e9db731345fa970f93ea1#rd)