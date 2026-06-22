<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# Zookeeper 解構

> **發布**：2026-05-20 · （無 venue）
> **QuantML 導讀**：[Zookeeper：CTA 统一因子框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247493897&idx=1&sn=0b2fc47a22541864e2f0d49f974d0c94&chksm=ce7d8e17f90a0701f1a6c466a7a2dfcecf185bc3286daccb0efc840f75fdcee92e2369df811f#rd)
> **核心定位**：落點於「量价表格 × 中长周期 × 监督回归 × 因子挖掘 × 人机协同可解释」五軸。解了 CTA 截面極窄與低信噪比下，傳統單因子理論各自為戰且非線性 ML 易過擬合的 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 用等權線性集成（L-En）統一三大商品風險溢價理論，構建抗過擬合的 Zookeeper 因子。② 核心 trick 是棄用非線性 ML，改用 120 個月滾動窗口防穿越，並結合 SHAP 實現理論歸因。③ 對「因子挖掘 × 人機協作」軸★，將黑盒預測轉譯為清晰經濟學解釋，解決實盤因子維護冗餘問題。④ 導讀給出樣本外（2005-2022）年化收益率 5.8%，Sharpe 比率 0.64。

**X-Ray.** 放回五軸 Pareto，本法在「可解釋性 × 穩健性」象限取得壓倒性優勢，但主動放棄了「非線性捕捉能力 × 高頻響應」的 envelope。它解了實盤兩大工程坑：一是小截面下 ML 邊界擬合淪為噪聲的過擬合陷阱；二是多因子疊加導致的共線性與維護成本。預測其打不開的邊界在於：無法處理非線性 regime shift（如極端流動性枯竭或政策干預導致的基差斷裂），且等權集成本質是方差懲罰器，在趨勢單邊發散時會產生截斷損益。對量化讀者的意義不在於直接搬磚，而在於提供了一套「降維打擊」的因子收斂範式：當截面寬度 < 50 時，優先驗證線性集成的穩健性，再考慮非線性升級。

## §1 · 架構 / Core Mechanism
| 維度 | 前作/傳統做法 | Zookeeper 改動 | 工程意圖 |
|---|---|---|---|
| 模型族 | 單一 OLS 或複雜非線性 ML | 等權線性集成（L-En） | 規避小截面下非線性邊界的過擬合方差 |
| 維度對齊 | 直接將宏觀/情緒變量塞入截面 | 滾動 OLS 載荷（Beta）轉換 | 解決 TS 與 CS 預測的維度不對齊問題 |
| 歸因機制 | 黑盒係數或單因子回歸 | SHAP 博弈論拆解 | 將預測信號映射至三大經濟理論，實現可解釋收斂 |

⚡ Eureka: 等權平均 5 種線性模型預測值，以二次估計誤差為代價，換取橫截面排序的極端穩健性。
ASCII Info Flow:
Raw Features (C/M/S) → Standardize → [Stepwise, PLS, Ridge, LASSO, Elastic Net] → Predictions → Equal-Weight Avg → Cross-Sectional Sort → Q4 Long / Q1 Short → Monthly Rebalance

## §2 · 數學層
📌 Napkin Formula:
$r_{i,t+1} = \alpha + \beta^C c_{i,t} + \beta^M m_{t} + \beta^S s_{t} + \epsilon_{i,t+1}$
(CS mode: $m_t, s_t$ replaced by rolling 60 個月 OLS betas $\hat{\beta}^M_{i,t}, \hat{\beta}^S_{i,t}$)
複雜度：$O(T \times N \times F)$ 滾動回歸，集成計算 $O(1)$。
直覺：用滾動載荷將全市場變量「個性化」為品種暴露度，使宏觀/情緒因子能參與截面定價。
Loss/訓練：MSE loss；10-fold temporal CV 鎖定超參，120 個月滾動窗口重訓係數，嚴禁前瞻。

## §3 · 數據層
規模/頻率：1988-04 至 2022-01，月頻。25 種主流大宗商品（軟商品/穀物/油籽/家畜/貴金屬/工業金屬/能源）。
來源：Group C (37 個商品特定變量) / Group M (40 個宏觀變量) / Group S (4 個情緒變量，含 FinBERT 提取 WSJ 822,595 篇新聞得分)。
樣本外與容量假設：嚴格劃分訓練 (1988-2004) 與測試 (2005-2022)；假設實盤可承受月度再平衡與完全保證金抵押，未計入滑點與衝擊成本。

## §4 · 代碼層
| 欄位 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（需處理宏觀數據發布滯後與文本情緒重構） |
| 數據可得性 | 低（WSJ 歷史新聞語料與精確宏觀實時修訂數據難獲取） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA (逐列) | 本方法 | Δ |
|---|---|---|---|---|
| TS 預測 (AVG 等權組合) | $R^2$ | OLS -101.68% / Ridge 2.80% / Elastic Net 3.00% / MLP 0.33% | L-En (C+M+S) 3.55% | 未披露 |
| CS 回測 (2005-2022) | 年化收益率 | 未披露 | 5.8% | 未披露 |
| CS 回測 (2005-2022) | Sharpe 比率 | 未披露 | 0.64 | 未披露 |
| CS 回測 (2005-2022) | t 統計量 | 未披露 | 3.2 | 未披露 |

解讀散文：TS 預測的 $R^2$ 提升（從 OLS 的 -101.68% 至 L-En 的 3.55%）主要來自 L1/L2 正則化與集成對小樣本方差的壓制，屬真實 capability。CS 回測的 5.8% 年化與 0.64 Sharpe 需警惕：導讀明確指出未扣除交易摩擦，且月度換手達 40%-80%，實盤 Sharpe 極可能因 15-20 bps 滑點與佣金大幅縮水。此外，宏觀與文本數據的實時發布滯後若未嚴格對齊，$R^2$ 與收益指標將被嚴重高估。

## §6 · 失效與隱含假設
6.1 論文自述 limitations：非線性 ML 在低樣本量下無法有效降低偏差，僅增加預測方差；等權集成無法動態適應不同宏觀環境下的理論權重切換。
6.2 推斷的隱含假設：
- Regime 依賴：假設三大理論（庫存/貼水/流動性）的線性加權關係在樣本外保持穩定，未處理結構性斷點。
- 容量/成本：假設 25 個品種的 Q1/Q4 分位組合可無摩擦執行，忽略窄截面排序劇烈變動導致的 200% 雙邊方向性換手衝擊。
- 數據泄漏：回測使用修正後宏觀數值與歷史新聞語料，實盤需嚴格對齊 Real-time Publication Date，否則存在隱蔽的前瞻偏差。
- Survivorship：樣本覆蓋 1988 年以來的 25 種主流商品，未明確說明是否包含已退市合約。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統單因子 (Basis/Momentum) | 因子收斂 vs 獨立維護 | 是 | 成熟但冗餘 |
| 非線性 ML CTA (RF/XGBoost) | 線性集成穩健性 vs 非線性過擬合風險 | 是 | 學術熱點/實盤謹慎 |
| 動態權重因子模型 | 等權靜態 vs 時變參數估計 | 是 | 高維護成本 |

🎤 Interview Tip:
正確答：「在截面寬度 < 50 且信噪比極低時，非線性模型的邊界擬合會將噪聲誤判為信號，導致樣本外方差爆炸。Zookeeper 用等權線性集成做方差懲罰器，並用 SHAP 將黑盒輸出對齊經濟理論，本質是『用可解釋性換取穩健性』。」
錯答：「因為 ML 模型太複雜所以不用，線性模型永遠更好。」（忽略小截面與低 SNR 的約束條件）

7.1 可證偽預測：若未來實盤嚴格對齊宏觀數據發布滯後並扣除 15-20 bps 成本，Zookeeper 多空組合的實盤 Sharpe 比率將顯著低於導讀給出的 0.64，證明等權集成在窄截面月度換手下的成本脆弱性。

## §8 · For the Reader
- 因子研究員：將本法視為「因子收斂基線」。在引入新商品因子前，先跑 L-En 集成與 Spanning Test，若傳統因子截距不顯著，直接棄用冗餘因子。
- 高頻執行/組合配置：警惕月度再平衡的換手衝擊。實盤需加入換手率懲罰約束與精確的跨期滑點模型，否則 5.8% 年化在扣除成本後可能轉負。
- LLM-agent/研究學生：Group S 的情緒提取可作為 LLM 金融語料應用的標準 benchmark。注意國內市場需替換為本土財經媒體語料，並嚴格處理宏觀數據的發布滯後（1-2 個月）。

## References
- 原論文：Call the Zookeeper: A Unified Framework for Commodity Risk Premiums
- Lineage：Theory of Storage / Theory of Normal Backwardation / Liquidity Demand and Provision → SHAP Attribution → Linear Ensemble
- QuantML 導讀鏈接：[Zookeeper：CTA 统一因子框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247493897&idx=1&sn=0b2fc47a22541864e2f0d49f974d0c94&chksm=ce7d8e17f90a0701f1a6c466a7a2dfcecf185bc3286daccb0efc840f75fdcee92e2369df811f#rd)