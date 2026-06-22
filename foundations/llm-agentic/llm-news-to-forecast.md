<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=端到端表征 autonomy=Agent自主演进 -->

# 从新闻到预测：基于LLM的时间序列预测 解構

> **發布**：2024-10-15 · （無 venue）
> **QuantML 導讀**：[从新闻到预测：基于LLM的时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487049&idx=1&sn=7337916648f291ef40e325faa83b0aca&chksm=ce7e6957f909e0414601ededa9a7c3957a1c31f0360a79398c097b4e9db731345fa970f93ea1#rd)
> **核心定位**：落點於「文本另类 × 生成式大模型」軸，解了傳統TS模型無法系統性內化非結構化社會事件（Regime Shift）的Prior Gap，將新聞過濾與序列預測耦合為可迭代的Agent閉環。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `端到端表征` | `Agent自主演进` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ①將新聞事件轉化為條件Token，微調預訓練LLM執行自回歸時間序列預測。②核心Trick是設計「推理Agent（過濾/分類新聞）+評估Agent（基於預測誤差反饋優化過濾邏輯）」的雙閉環迭代。③對「Agent自主演进」軸★，它用預測結果驅動數據清洗，實現了非結構化Alpha的自動對齊與降噪。④導讀未給量化結果。

**X-Ray.** 本方法實質是將TS預測重構為條件生成任務，繞開了傳統統計模型對外部衝擊的滯後響應。其工程價值不在於模型架構的創新，而在於用LLM Agent替代了人工因子挖掘中的「新聞篩選-權重分配」黑盒，實現了Alpha來源的動態對齊。然而，生成式建模天然對Token長度與序列對齊敏感，日頻波段下的長窗口預測必然面臨上下文截斷與幻覺風險。對量化讀者而言，此框架的真實Alpha不來自LLM本身的生成能力，而來自評估Agent反饋所構建的「新聞-波動」因果過濾鏈；若無法解決粗粒度新聞與本地化資產的錯配，該方法僅能作為事件驅動策略的輔助特徵管道，而非端到端交易信號。

## §1 · 架構 / Core Mechanism
| 維度 | 傳統統計/深度TS模型 | 標準LLM-TS微調 | 本方法 (News-to-Pred) |
|---|---|---|---|
| **輸入模態** | 純數值序列 | 數值序列 + 靜態文本Prompt | 數值序列 + 動態過濾新聞JSON |
| **數據清洗** | 人工特徵工程/靜態規則 | 無/一次性對齊 | 推理Agent自動分類+評估Agent反饋迭代 |
| **訓練範式** | 監督回歸/自回歸 | Next-Token Prediction | 條件生成 + 誤差驅動的新聞邏輯閉環更新 |

⚡ **Eureka:** 用預測誤差反饋驅動新聞過濾邏輯的迭代，而非一次性靜態對齊。
📊 **信息流 ASCII:**
```
Raw News → Reasoning Agent (Filter/Classify) → JSON Context
                                      ↓
                              LLM Predictor (Fine-tune)
                                      ↓
                              Prediction Output → Evaluation Agent (Check Error)
                                      ↓
                              Feedback → Reasoning Agent (Update Logic) → Loop
```

## §2 · 數學層
📌 **Napkin Formula:** $P(y_t | y_{<t}, \text{News}_{filtered}) = \text{LLM}_\theta(y_{<t} \oplus \text{News}_{filtered})$
**複雜度:** $O(N \cdot L_{ctx} \cdot d^2)$ per step，主要瓶頸在於Attention對拼接Token的計算。
**直覺:** 將數字序列離散化為Token，新聞作為條件前綴注入Attention Mask，自回歸生成下一時刻數值分佈。Loss為標準Next-Token Prediction Loss (Cross-Entropy)，但輸出層直接映射回數值空間（無歸一化以保留物理尺度）。訓練非端到端梯度更新，而是依賴評估Agent根據MSE/MAE誤差調整新聞過濾Prompt與邏輯，屬離散優化閉環。

## §3 · 數據層
**規模/頻率:** 交通(車流量)、匯率、比特幣、澳洲電力需求。頻率涵蓋日頻、小時級、半小時級。
**來源/時段:** 公開API與AEMO數據，為避免預訓練偏見，資料更新至2022年。
**樣本外/容量:** 導讀未披露具體OOS劃分策略與樣本容量假設，僅提及跨頻率評估以驗證算法有效性。

## §4 · 代碼層
| 欄位 | 狀態/細節 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 高（需LLM API調用+Agent邏輯編排+多源數據時間戳對齊） |
| 數據可得性 | 中（電力/匯率/比特幣公開，交通新聞需本地化爬取與清洗） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 電力需求/匯率/比特幣 | 未披露 | 未披露 | 未披露 | 未披露 |
| 交通(加州道路) | 未披露 | 未披露 | 未披露 | 未披露 |

**解讀:** 導讀僅定性描述「顯著優於傳統方法」與「交通改進幅度較小」，未披露IR/Sharpe/MSE等具體數值，Δ欄全為未披露。定性差異指向「新聞粒度與資產本地化程度的匹配度」是真實Capability的邊界。交通案例的MSE劣化與MAE穩定，暗示模型對極端值敏感，可能混入了前瞻偏差或新聞發布時間戳與交易執行時間的錯配。未計入API調用成本與Agent迭代算力開銷前，該框架在實盤中的淨Alpha需打折扣。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 預訓練LLM最大Token長度限制導致長序列/多序列處理複雜；不適用於氣象/物理等人類活動影響小的領域；僅增強而非取代傳統TS任務（分類/插值）。
**6.2 推斷的隱含假設:** 新聞發布時間與市場反應存在穩定滯後（Regime依賴）；LLM過濾邏輯具備跨市場泛化性（未驗證）；評估Agent的誤差反饋能準確歸因於新聞遺漏而非模型容量瓶頸（數據泄漏風險：若新聞包含未來信息或時間戳未嚴格對齊，Agent會學習到偽因果）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統統計模型(ARIMA/GARCH) | 外部衝擊內化能力 | 是 | 成熟基線 |
| 標準LLM-TS微調(如Time-LLM) | 新聞過濾自動化與反饋閉環 | 部分 | 快速迭代 |
| 事件驅動因子庫(如RavenPack) | 端到端生成 vs 預定義標籤 | 否(商業) | 工業標準 |

🎤 **Interview Tip:** 
正確答：「該方法的核心不是LLM的預測能力，而是Agent閉環解決了非結構化數據的動態對齊與降噪問題；實盤需嚴格處理新聞時間戳與交易執行的錯配，並評估API成本。」
錯答：「LLM直接讀新聞就能預測股價，比傳統模型準。」
**7.1 可證偽預測:** 若未來實盤驗證中，該雙Agent架構在公開日頻資產的淨收益未顯著高於靜態新聞因子基線，則「Agent自主演進帶來淨Alpha」的命題失效。

## §8 · For the Reader
- **因子研究員:** 將評估Agent的誤差反饋機制抽象為「新聞權重動態調整」因子，脫離LLM生成層以降低延遲，聚焦於過濾邏輯的離散特徵提取。
- **高頻執行:** 此方法為日頻波段設計，Token生成與Agent迭代延遲過高，不適合HFT；可轉為盤後新聞情緒特徵管道，供中低頻策略使用。
- **組合配置:** 利用新聞過濾邏輯識別Regime Shift，作為戰術資產配置(TAA)的開關信號，而非直接生成收益預測，規避生成模型的幻覺風險。

## References
- 原論文：从新闻到预测：基于LLM的时间序列预测（無 venue, 2024）
- Lineage: LLaMA (Touvron et al., 2023) / Time-LLM (Jin et al., 2023) / Agent-based TS forecasting
- QuantML 導讀鏈接：[从新闻到预测：基于LLM的时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487049&idx=1&sn=7337916648f291ef40e325faa83b0aca&chksm=ce7e6957f909e0414601ededa9a7c3957a1c31f0360a79398c097b4e9db731345fa970f93ea1#rd)