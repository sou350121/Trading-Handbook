<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=组合执行优化 autonomy=人机协同可解释 -->

# OU-LLM配对交易框架 解構（OU-LLM配对交易框架）

> **發布**：2025-12-21 · JP Morgan
> **QuantML 導讀**：[JP Morgan | 融合NLP情绪与LLM洞察的配对交易策略](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492744&idx=1&sn=baa6d786bba61e1fe46580d0101b788a&chksm=ce7d8396f90a0a80ef6258181dbb6cc758ff1c6f43431182c417e00d899c96049bb1768c3e2a#rd)
> **核心定位**：落點於日频波段與生成式大模型的交叉軸，解構傳統Z-score對均值回歸速度與波動率恆定的錯誤假設，以OU過程重構統計套利基線，並透過LLM語義分類剝離永久性結構衝擊。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ①以OU均值回歸模型動態建模價差回歸速度與波動率生成S-score，取代靜態Z-score。②融合多層級條件過濾（NLP情緒/同業限制）與LLM衝擊分類（暫時vs永久）。③對「組合執行優化」軸★，將黑箱統計套利升級為具備語義理解能力的量化取證，顯著改善非美市場風險調整後收益與收益偏度。④歐洲市場95%置信水平下夏普比率達0.86，總收益躍升至115.42%。

**X-Ray.** 該框架本質是「統計地基+語義斷路器」的解耦架構。它解構了傳統Z-score對常數均值與異方差的錯誤假設，以OU過程動態重定價差風險。對量化讀者而言，其工程價值不在於預測精度，而在於將黑箱統計套利升級為可解釋的風險過濾系統：LLM不生成信號，僅擔任結構斷點檢測器，剝離永久性衝擊。然而，該架構打不開高頻執行與跨資產宏觀的envelope，且日频參數估計在體制切換時存在滯後。其核心權衡在於周轉率與交易成本的摩擦，實盤需嚴格對齊新聞時間戳以防前瞻偏差。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作 (Z-score Stat Arb) | 本框架 (OU-LLM) | 工程意圖 |
|---|---|---|---|
| 基線模型 | 滾動窗口標準差倍數 (常數假設) | OU均值回歸過程 (動態S-score) | 解決異方差與回歸速度漂移 |
| 信號過濾 | 無/靜態閾值 | 多層級條件過濾 (同業/NLP/Q-Score) | 剝離行業結構性風險與尾部情緒 |
| 決策邏輯 | 純數學統計 (黑箱) | LLM語義衝擊分類 (暫時vs永久) | 引入量化取證，規避基本面永久斷裂 |

**1.2 ⚡ Eureka**
S-score將價差偏離度除以「均衡波動率」，本質是讓閾值隨回歸速度與波動率體制自適應；LLM在此不產α，僅作語義電路斷路器，過濾永久衝擊。

**1.3 信息流 ASCII**
```
Raw Prices -> Correlation/Hurst/ADF/Cointegration -> OU Parameter Est. -> S-score Calc
      |
      v
Multi-Level Conditioning (Sector Limit / SmartBuzz NLP / Q-Score)
      |
      v
LLM Shock Classification (5-day News -> Temp/Perm) -> Execution / Stop-loss (2% Cap)
```

## §2 · 數學層
**📌 Napkin Formula**
$$S\text{-}score = \frac{X_t - \mu}{\sigma_{eq}}, \quad \sigma_{eq} = \sqrt{\frac{\sigma^2}{2\kappa}}$$
**複雜度**：$O(N)$ 滾動參數估計 + LLM API 調用（非端到端梯度下降）。
**直覺**：Z-score 假設方差恆定，OU 將價差視為隨機微分方程。S-score 實質是「均衡標準差」倍數，對波動率體制切換自適應。Loss/訓練：無監督統計估計 + Prompt Engineering，依賴協整檢驗與ADF測試的頻繁重估。

## §3 · 數據層
**規模/頻率/市場/時段**：2015-2024 日频开盘价/全收益序列。覆蓋 MSCI World、歐洲、美國、亞洲（除日本）。
**來源**：公開市場數據 + J.P. Morgan SmartBuzz 新聞流 + LLM API。
**樣本外與容量假設**：閉環交易計數已披露，但滑點與交易成本未明確披露。容量假設未披露，日频波段依賴協整對數量，隱含流動性瓶頸假設。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 中高 (需自實現OU參數估計與協整篩選) | 低 (SmartBuzz為機構內部數據，LLM需商業API) |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 全球 (MSCI World) | Sharpe | 0.58 | 1.20 | +107% |
| 歐洲 (95%置信) | Sharpe | 0.71 | 0.86 | +0.15 |
| 歐洲 (95%置信) | Total PnL | 47.53% | 115.42% | +67.89% |
| 歐洲 (95%置信) | Max Drawdown | 未披露 | -19.41% | 未披露 |
| 亞洲除日本 (95%置信) | Sharpe | 0.42 | 0.67 | +0.25 |
| 亞洲除日本 (95%置信) | Total PnL | 17.70% | 56.83% | +39.13% |
| 美國 (95%置信) | Sharpe | 0.61 | 0.44 | -0.17 |
| 美國 (90%置信) | Max Drawdown | -22.74% | -17.00% | +5.74% |
| 美國 (同業限制) | Sharpe | 未披露 | 提升40% | +40% |
| 美國 (同業限制) | Total PnL | 43.29% | 106.76% | +63.47% |
| 美國 (SmartBuzz 2021-2024) | Sharpe | 0.06 | 0.36 | +0.30 |
| 美國 (SmartBuzz 2021-2024) | Max Drawdown | -16.23% | -8.18% | +8.05% |
| 美國 (SmartBuzz 2021-2024) | Skewness | -3.79 | -0.50 | +3.29 |

**解讀**：Δ 中的 Sharpe 與 PnL 提升主要來自風險結構重塑（負偏度改善）與過濾永久性衝擊，屬真實 capability。但歐洲市場交易數翻倍（6950筆 vs 3089筆）暗示高周轉，原始 Total PnL 極可能未計入滑點與手續費，實盤 Δ 將被成本摩擦大幅稀釋。美國市場 95% 置信下的負 Δ 反映該框架對高度集中動態的 regime 依賴。案例研究中的異常 Sharpe 屬樣本內倖存者偏差，不可作為泛化指標。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**
美國市場統計均值回歸難以捕捉高度集中的動態特徵；體制轉換時高敏感度可能引發回撤；LLM分類依賴新聞時效與提示詞穩定性。
**6.2 推斷的隱含假設**
假設日频數據足以捕捉OU參數穩定性（未驗證高頻噪聲影響）；假設SmartBuzz/Q-Score數據無前瞻偏差（實盤需嚴格對齊發布時間戳）；容量假設未披露，但交易數翻倍暗示可能面臨流動性瓶頸；隱含假設LLM推理延遲不影響日频波段執行。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統Z-score配對交易 | 靜態標準差 vs 動態OU過程 | Open | Baseline |
| 純LLM驅動交易 | 黑箱語義生成 vs 統計地基+語義過濾 | Closed/Proprietary | Emerging |

**🎤 Interview Tip**
*正確答*：OU模型將價差建模為隨機微分方程，S-score動態調整波動率與回歸速度，本質是解決Z-score的異方差與常數均值假設失效；LLM在此架構中擔任「結構斷點檢測器」而非信號生成器，用於過濾永久衝擊。
*錯答*：認為LLM直接輸出買賣點或替代統計檢驗；忽略OU模型對體制切換的敏感性與高周轉帶來的成本摩擦。

**7.1 可證偽預測帶日期**
若2026年Q2全球波動率體制持續高頻切換且新聞流延遲超過1小時，該框架在非美市場的Sharpe將回落至0.6以下，且SmartBuzz過濾的勝率將顯著低於同業限制策略。

## §8 · For the Reader
* **因子研究員**：將S-score視為動態因子，檢驗其與傳統動能/價值因子的正交性；注意協整檢驗的樣本內過擬合風險。
* **高頻執行**：日频波段框架不適用HFT，但OU參數估計可下沉至分鐘频；需實測高周轉率下的滑點模型。
* **組合配置**：歐洲市場95%置信下的Sharpe 0.86與-19.41% MDD適合做低相關衛星策略；美國市場需疊加行業中性約束。
* **LLM-agent**：此架構示範了「統計過濾+LLM分類」的解耦設計，避免端到端大模型直接控制倉位的合規與延遲風險。
* **RL策略**：OU過程的連續狀態空間可作為RL的環境動力學先驗，將S-score閾值優化轉化為MDP中的策略搜索。

## References
* JP Morgan arxiv=None framework=OU-LLM配对交易框架
* QuantML 導讀鏈接：[JP Morgan | 融合NLP情绪与LLM洞察的配对交易策略](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492744&idx=1&sn=baa6d786bba61e1fe46580d0101b788a&chksm=ce7d8396f90a0a80ef6258181dbb6cc758ff1c6f43431182c417e00d899c96049bb1768c3e2a#rd)
* Lineage: Z-score Stat Arb -> OU Process (Vasicek) -> NLP Sentiment Filtering -> LLM Semantic Classification