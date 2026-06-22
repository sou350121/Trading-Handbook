<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# DeePM 解構

> **發布**：2026-01-12 · （無 venue）
> **QuantML 導讀**：[Oxford-Man ｜ 端到端深度学习宏观量化交易框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492979&idx=1&sn=0750044bd65444f2fca58dcf237caeee&chksm=ce7d826df90a0b7b322e5c0de1cd731cbcded9d67301ea5c4ae3b61d82a7e342307141ad1025#rd)
> **核心定位**：落點於端到端表征與全自動黑盒軸，解決傳統兩階段宏觀配置中 MSE 損失與最終效用錯配、以及異步市場閉盤導致的前視偏差（Look-ahead Bias）問題。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 端到端學習宏觀資產配置，跳過獨立預測階段直接優化淨夏普。② 核心 trick 為 Directed Delay 強制因果過濾、Macro Graph Prior 正則化截面依賴、SoftMin 代理 EVaR 目標。③ 對端到端表征軸★，將經濟學第一性原理硬編碼進注意力機制，打破純數據驅動在低信噪比下的過擬合。④ 導讀給出 OOS 2010-2025 淨夏普 0.93，較 Momentum Transformer 基線提升 0.27。

**X-Ray.** 放回五軸 Pareto，DeePM 在「結構先驗 vs 數據驅動」光譜上明確偏向結構化正則化。它解了兩階段流水線的損失錯配坑，以及異步數據混合產生的偽相關前視偏差。預測其打不開的 envelope 在於：固定宏觀圖譜無法適應結構性斷裂（如供應鏈重組或貨幣體系脫鉤），且日頻波段在流動性枯竭時仍受 Structural Floor 限制。對量化讀者的意義在於提供了一套可部署的「因果過濾+分佈魯棒優化」工程藍圖，證明在宏觀層面，犧牲信息新鮮度換取因果穩健性遠勝於追逐高頻微結構。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作/基線 (Momentum Transformer / 兩階段法) | DeePM 改動 | 工程意義 |
|---|---|---|---|
| 信息流動 | 混合當日數據 / 獨立時間序列 | Directed Delay (Key/Value 嚴格設為 t-1) | 強制因果過濾，消滅異步閉盤引入的 Look-ahead Bias |
| 截面依賴 | 隱式數據驅動 / 無結構 | Macro Graph Prior 注入 GAT 注意力分數 | 以經濟學第一性原理正則化跨資產溢出，抑制低信噪比過擬合 |
| 優化目標 | MSE 或標準 Sharpe | Pooled Sharpe + SoftMin (EVaR 代理) | 對抗最差歷史窗口，防止模型為省成本退化成低頻 Beta |

⚡ **Eureka:** 用定向延遲強制注意力機制學習預測性的脈衝響應函數（近似傳遞熵），而非同期共運動；信息新鮮度讓位於因果穩健性。
**信息流:**
`Static Asset Embedding + Cost Params` → `V-VSN (FiLM Feature Weighting)` → `LSTM (Nonlinear Low-pass)` → `Temporal Attn (ResSwiGLU)` → `Cross-Sectional MHA (t-1 Filter)` → `GAT (Graph Prior Bias)` → `tanh → Vol-Scaled Weights`

## §2 · 數學層
📌 **Napkin Formula:**
`Loss = Pooled_Sharpe(θ) + λ·SoftMin_{w∈Windows}(Sharpe_w(θ))`
`SoftMin(x) ≈ -τ·log(Σ exp(-x/τ))` → 對偶於 Entropic Value-at-Risk (EVaR)
**直覺:** SoftMin 構建隱式對手重加權歷史窗口，強迫模型在最具不利子時期保持活躍交易，避免陷入「惰性陷阱」。
**訓練細節:** AdamW 優化器；Exact Gradient Accumulation 解決 GPU 顯存限制下的全局 Sharpe 統計計算；Top-K 集成利用 Jensen 不等式使集成交易成本嚴格低於成員平均，隱式正則化換手率。

## §3 · 數據層
- **規模/頻率/市場:** 50 種高流動性期貨與外匯合約。日頻收盤價。
- **時段/協議:** 1990-2025 全樣本；嚴格 Walk-Forward Validation（每 5 年訓練/測試塊）；報告 2010-2025 樣本外表現。
- **特徵來源:** 僅日收盤價，不引入外部宏觀指標。計算 1, 21, 63, 252 天波動率縮放回報、多尺度 MACD、價格 Z-Score、MAD 截斷離群值。
- **容量假設:** 依賴 63 天 EWMA 波動率估計與 Structural Minimum Cost Model（含 Tick Size Floor 與 Liquidity Scalar）。

## §4 · 代碼層
| 欄位 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | High（需實現 Exact Gradient Accumulation 與 GAT 結構偏差注入） |
| 數據可得性 | 需 50 個流動性期貨/外匯日收盤價（具體合約列表未披露） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 50 Futures/FX (OOS 2010-2025) | Net SR | Risk Parity 0.50 | 0.93 | +0.43 |
| 50 Futures/FX (OOS 2010-2025) | Net SR | TSMOM 0.45 | 0.93 | +0.48 |
| 50 Futures/FX (OOS 2010-2025) | Net SR | Momentum Transformer 0.66 | 0.93 | +0.27 |
| 50 Futures/FX (OOS 2010-2025) | MDD | No Graph 19.8% | 16.0% | -3.8% |
| 50 Futures/FX (OOS 2010-2025) | IR | 未披露 | 0.44 | 未披露 |
| 50 Futures/FX (OOS 2010-2025) | Avg Hold | 未披露 | 7.1天 | 未披露 |

**解讀:** Δ 在 Net SR 與 MDD 上的提升屬真 capability，源於 Macro Graph Prior 的截面去噪與 Directed Delay 的因果過濾。Post-2020 體制下 DeePM 維持 0.79 淨夏普（經典趨勢 0.38），證明表示學習成功泛化至高利率環境。需警惕：成本模型為線性+結構底限，未計入訂單簿深度與排隊位置；實際執行中「蟑螂屋」市場的 Liquidity Scalar 可能放大衝擊成本，導致實盤 Δ 收斂。無前視偏差已通過嚴格 t-1 過濾驗證。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 固定宏觀圖譜先驗無法隨時間演變，可能滯後於供應鏈重組或貨幣體系脫鉤；Exact Gradient Accumulation 增加訓練複雜度；框架目前限於日頻，未測試高頻內信息新鮮度與因果有效性的權衡反轉點。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 假設經濟學第一性原理（部門同質性、供應鏈、利差交易）在樣本外仍具結構穩定性。
- **容量/成本:** 假設 50 資產宇宙流動性足以容納日頻再平衡；Structural Floor 與線性衝擊模型在極端波動下可能低估滑點。
- **數據泄漏:** 特徵工程（MACD/Z-Score）使用回溯窗口，若實盤數據推送延遲或修正，可能破壞因果過濾假設。
- **Survivorship:** 導讀未明確說明 50 合約是否含已退市/流動性枯竭標的，需驗證生存偏差控制。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Momentum Transformer | 無截面圖譜先驗 / 無定向延遲 / 標準 Sharpe | 開源 | 穩定 |
| Deep RL Portfolio (PPO/SAC) | 強化學習 vs 監督回歸 / 狀態空間定義不同 / 無 EVaR 代理 | 部分開源 | 實驗性 |
| Rolling MVO / ERC | 兩階段參數估計 / 協方差誤差最大化 / 無端到端優化 | 開源 | 工業標準 |

🎤 **Interview Tip:** 
- **正確答:** 「DeePM 用 SoftMin 代理 EVaR 強制模型在最差窗口保持活躍，避免為省成本而退化成低頻 Beta；同時用 Directed Delay 犧牲信息新鮮度換取因果穩健性。」
- **錯答:** 「DeePM 用強化學習直接優化夏普比率，並通過高頻數據捕捉微結構 Alpha。」

**7.1 可證偽預測:** 若 2026-2027 全球供應鏈發生非線性重構，固定圖譜先驗將導致截面正則化失效，淨夏普應回落至 0.70 以下。

## §8 · For the Reader
- **因子研究員:** 關注 Macro Graph Prior 的鄰接矩陣構建邏輯，可替代傳統行業分類因子，注入截面正則化模塊。
- **高頻執行:** Structural Floor 與 Liquidity Scalar 模型可借鑒至衝擊成本預估，但日頻框架不適用微結構排隊博弈。
- **組合配置:** Top-K 集成利用 Jensen 不等式隱式降低換手率，是低成本控制 turnover 的工程技巧，可移植至多模型投票系統。
- **LLM-agent:** 定向延遲機制可移植至多模態 Agent 的因果推理模塊，防止信息新鮮度陷阱與幻覺共運動。
- **研究學生:** 精確梯度累積（Exact Gradient Accumulation）是解決全局統計量不可微/顯存瓶頸的標準範式，值得復現。

## References
- QuantML 導讀：[Oxford-Man ｜ 端到端深度学习宏观量化交易框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492979&idx=1&sn=0750044bd65444f2fca58dcf237caeee&chksm=ce7d826df90a0b7b322e5c0de1cd731cbcded9d67301ea5c4ae3b61d82a7e342307141ad1025#rd)
- Lineage: Momentum Transformer (Time-series Attn) → Deep Learning for Portfolio Optimization (End-to-End) → EVaR / Distributionally Robust Optimization Literature.
- 原論文: TBD（導讀未提供 arXiv/DOI）