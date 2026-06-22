<!-- ontology-5axis data=多模态 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# CryptoPulse 解構（CryptoPulse）

> **發布**：2025-03-01 · （無 venue）
> **QuantML 導讀**：[CryptoPulse：基于双重预测与跨相关市场指标的短期加密货币预测模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489490&idx=1&sn=9a8a90ea0a1c3a0b7f2f0f03d7eafe40&chksm=ce7e70ccf909f9dabeab4b27b0094b4a13ec43cf20988cc684ad267ae1fe805383fb23413bf5#rd)
> **核心定位**：多模态监督回归框架，解決傳統量價模型在加密貨幣市場中忽略宏觀代理與新聞情緒的 prior gap，以雙路預測解耦系統性波動與個體價格動態。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出 CryptoPulse 框架，融合 LLM 情緒、宏觀代理指標與技術指標預測次日收盤價。② 核心 trick 為雙路預測解耦（宏觀波動 vs 個體動態），並透過 LLM 少樣本一致性校準提取情緒向量，重標定與融合兩路輸出。③ 對「端到端表徵」軸★，它繞過傳統因子工程，直接將非結構化新聞與多維量價序列映射為預測張量。④ 導讀給出在排名前 5 加密貨幣上 MAE 降低 10.4% 到 63.8%、MSE 降低 17.2% 到 69.0%。

**X-Ray.** 該架構將「多模态」與「日频波段」強綁定，用 LLM 一致性校準替代人工標註情緒，實質是將非結構化文本降維為低頻調控信號。它解了傳統 LSTM/Transformer 在加密貨幣極端波動下易過擬合量價殘差的工程坑，但雙路融合依賴 Tanh 激活與固定權重，意味著模型對 regime switch 的適應性受限於情緒向量的平滑度。對量化讀者而言，其價值不在於絕對預測精度，而在於提供了一套可插拔的「宏觀-個體-情緒」三維解耦範式；然而，7 天觀察窗與次日預測設定，註定它打不開高頻或週期大於 1 個月的 envelope，且 LLM API 延遲與成本在實盤中將成為隱形摩擦。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作 (SVM/RF/LSTM/CNN-LSTM) | CryptoPulse |
|:---|:---|:---|
| 特徵輸入 | 單一歷史量價或手動標註情緒 | 多模态：Top N 宏觀代理 + 7 種技術指標 + LLM 情緒向量 |
| 預測路徑 | 單路端到端回歸 | 雙路解耦：宏觀環境波動路徑 + 個體價格動態路徑 |
| 融合機制 | 靜態加權或無融合 | 動態重標定：情緒張量經 Tanh 生成縮放因子，規範波動範圍並融合兩路輸出 |

⚡ **Eureka:** 用「智囊團討論」式少樣本提示讓 LLM 投票出情緒標籤，再以一致性分數校準，避免單次調用噪聲。

**信息流 ASCII:**
```
News ──▶ LLM Few-Shot (m=3) ──▶ Sentiment Vector ──▶ Embed ──▶ Tanh(α) & Fuse
Macro (Top N Crypto) ──▶ 1D Conv + Pos Enc + Cross-Attn ──▶ V̂_macro
Target Price + 7 Tech Indicators ──▶ Modified NLinear ──▶ V̂_indiv
V̂_macro, V̂_indiv, α ──▶ Fusion ──▶ P̂_{t+1}
```

## §2 · 數學層
📌 **Napkin Formula:**
$$ \hat{V}_{macro} = \text{Attn}(E_{target}, E_{macro}) \cdot W_{ffn} $$
$$ \hat{V}_{indiv} = \text{NLinear}(X_{price}, X_{tech}) $$
$$ \alpha = \text{Tanh}(\text{MLP}(E_{sent})) $$
$$ \hat{P}_{t+1} = P_t \cdot (1 + \text{Fusion}(\hat{V}_{macro}, \hat{V}_{indiv}, \alpha)) $$
**直覺:** 先預測波動率而非直接預測價格，规避極端波動導致的模型發散；情緒向量不直接參與回歸，而是作為動態閾值與融合權重，起到「減震器」作用。
**Loss/訓練:** 導讀未披露具體 loss 函數，僅以 MAE/MSE 作為評估指標。訓練採用時間順序切分，結果為五次實驗平均值。複雜度主要來自 1D Conv 與 Cross-Attention 的 $O(T \cdot d)$ 序列處理，LLM 調用為離線/批處理 $O(1)$。

## §3 · 數據層
- **規模/頻率/市場:** 雅虎財經與 Cointelegraph 編譯的真實世界數據集。頻率為日频，觀察窗口固定為 7 天。市場覆蓋市值排名前 5、10、15、20 的加密貨幣。
- **來源與處理:** 價格數據來自雅虎財經，新聞來自 Cointelegraph。技術指標預處理包含 %K/%D、Williams %R、A/D 振盪器、動量、Diff 7、ROC 等 7 類。
- **樣本外與容量假設:** 數據按時間順序以 7:1:2 比例拆分訓練/驗證/測試集。隱含假設為 Top 20 加密貨幣具備足夠流動性與新聞覆蓋度，低流動性山寨幣被明確排除。

## §4 · 代碼層
| 項目 | 狀態 |
|:---|:---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | Medium（需 GPT-3.5-Turbo API 權限，需手動修改 NLinear 結構與實現 Cross-Attn 宏觀路徑） |
| 數據可得性 | 導讀註明「將在獲得接受後公開下載」，當前為 TBD |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|:---|:---|:---|:---|:---|
| Top 5 Crypto | MAE | 未披露 | 降低 10.4% 到 63.8% | 未披露 |
| Top 5 Crypto | MSE | 未披露 | 降低 17.2% 到 69.0% | 未披露 |
| Top 10/15/20 Crypto | MAE | 未披露 | 降低 42.2% 到 46.9% | 未披露 |
| Top 10/15/20 Crypto | MSE | 未披露 | 降低 41.8% 到 47.9% | 未披露 |

**解讀:** 導讀僅給出相對「最佳比較方法」的提升區間，未提供絕對基線數值，故 Δ 欄標記未披露。MAE/MSE 的顯著降低主要來自雙路解耦對極端波動的平滑作用，以及 LLM 情緒對新聞衝擊的提前定價。但該 Δ 純屬預測精度指標，未計入交易成本、滑點與 LLM API 延遲。Top 5 到 Top 20 的提升區間收窄，暗示模型對流動性較低資產的泛化能力已接近瓶頸，部分 Δ 可能來自 7 天短窗口內的過擬合或新聞時間戳與收盤價的潛在前瞻對齊。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** LLM 響應不穩定且易受新聞噪聲干擾；傳統情緒分析依賴手動標註無法擴展至實時多幣種；直接使用情緒向量會引入波動性噪聲。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 模型假設新聞情緒與價格波動存在穩定線性/非線性映射，在「純鏈上巨鲸拋售」或「無新聞真空期」regime 下，情緒融合權重將失效。
- **容量與成本:** 僅覆蓋 Top 20 幣種，隱含容量假設為中高流動性市場；實盤中 GPT-3.5-Turbo 的 API 成本與延遲將侵蝕日频波段利潤。
- **數據泄漏風險:** 新聞發布時間若與雅虎財經收盤價時間戳未嚴格錯開，易產生 look-ahead bias。
- **Survivorship:** 僅追蹤當前 Top N 幣種，忽略已退市或歸零資產，高估實戰存活率。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|:---|:---|:---|:---|
| NLinear / DLinear | 單路量價回歸，無情緒/宏觀解耦 | Open | SOTA Baseline |
| LSTM / CNN-LSTM | 序列建模但忽略跨市場宏觀代理 | Open | Legacy |
| FinRLlama / TradingAgents | LLM 驅動但偏向 RL/多智能體決策 | Open | Research |

🎤 **Interview Tip:** 
- ✅ 正確答：「CryptoPulse 的核心不在於預測器本身，而在於將情緒作為動態縮放因子（Tanh 閾值）而非直接特徵輸入，這降低了非結構化文本的噪聲傳播。實盤需重點驗證 API 延遲對 7 天窗口的滑點影響。」
- ❌ 錯答：「它用 LLM 直接預測價格，所以比 LSTM 準。」（混淆了特徵工程與預測頭，且忽略雙路解耦設計）

**7.1 可證偽預測帶日期:** 若 2025-Q3 加密貨幣市場轉為純鏈上數據（如 On-chain Flow/MEV）主導定價，情緒融合路徑的 ablation 權重將衰減至統計不顯著水平（p>0.05）。

## §8 · For the Reader
- **因子研究員:** 將 LLM 一致性校準視為「情緒因子構建器」，可替換為開源 LLM（如 Qwen/GLM）進行本地化部署，降低實盤摩擦成本。
- **高頻執行:** 該框架為日频波段設計，不適用 HFT。若需降頻使用，需將 7 天窗口壓縮至 1-3 天，並重校 Tanh 縮放係數以防過平滑。
- **組合配置:** 可將 CryptoPulse 的雙路輸出作為資產權重調整的先行指標，但需對 Top 20 以外的長尾資產進行流動性濾波，避免容量陷阱。

## References
- 原論文/框架: CryptoPulse (Venue: TBD, arxiv: None)
- Lineage: NLinear, DLinear, LSTM/bi-LSTM, CNN-LSTM, FinRLlama, TradingAgents
- QuantML 導讀鏈接: [CryptoPulse：基于双重预测与跨相关市场指标的短期加密货币预测模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489490&idx=1&sn=9a8a90ea0a1c3a0b7f2f0f03d7eafe40&chksm=ce7e70ccf909f9dabeab4b27b0094b4a13ec43cf20988cc684ad267ae1fe805383fb23413bf5#rd)