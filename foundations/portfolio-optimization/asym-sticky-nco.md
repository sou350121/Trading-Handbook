<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=组合执行优化 autonomy=全自动黑盒 -->

# Asym-Sticky-NCO 解構（Asym-Sticky-NCO）

> **來源發布**：2026-11-03 · （無 venue） · arXiv [2611.05310](https://arxiv.org/abs/2611.05310)
> **arXiv 原文**：[Mitigating Return Drag and Regime Whipsaws in Sector Rotation: An Asymmetric Downside-Aware Nested Clustered Optimization Framework](https://arxiv.org/abs/2611.05310v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 中長周期 × 監督回歸 × 組合執行優化 × 全自動黑盒」；它不解行業輪動預測，而是把 NCO 的對稱協方差假設與 HMM 的狀態抖動兩個工程坑，用 downside semi-covariance 與 sticky regime 兩道閘門拆掉。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ① 提出 Asym-Sticky-NCO：把 Bawa-Lindenberg downside semi-covariance 貫穿 NCO 的樹距離與兩階段配置，並把 2D（SPY 收益 + VIX）Gaussian HMM 加上 sticky 自轉移先驗 κ=0.30 與置信閾值 δ=0.65。② 核心 trick 是「不對稱性要一路走到聚類拓撲」＋「regime 持續性用 prior 與 threshold 兩道硬閘門」，而不只是換最終目標函數。③ 對組合執行軸：65 個月 walk-forward 上 CAGR 13.41% vs Plain NCO 10.65%，Max DD -14.22 vs S&P 500 23.93% 跌幅，月度換手 30.48% vs Baseline 60.78%，regime switches 16→4。④ 但 Sharpe 0.63 / Sortino 1.07 低於被動 S&P 500 的 0.86 / 1.47，CAGR 13.41% 也低於 SPY 16.36%——edge 在 downside 與 execution 維度，不在波動調整後收益。

**X-Ray.** 放回五軸 Pareto：N=11、月度、全自動、無 ML 收益預測——它的位置是「組合執行優化」而非 alpha 生成。解了兩個具體工程坑：(a) 對稱協方差把大漲與大跌同罰，NCO 在成長股主導的牛市裡系統性低配；(b) 無約束 Baum-Welch 在月度頻率反覆翻狀態，換手爆表。它的 envelope 很窄：11 個美國 sector ETF、65 個月 OOS、只做多、月度調倉、線性 bps 成本；regime 只有 2 態、特徵只有 SPY 收益與 VIX，不解決行業輪動的預測問題，也不解決容量與衝擊成本。對量化讀者，這篇的價值在兩處：一是 ablation 清楚顯示 downside 與 regime 互補而非替代（單用 downside 把 MDD 從 -14.48 惡化到 -20.32）；二是「聚類距離也換成 downside」這一步——本頁推斷，多數 NCO 變體只換最終目標函數，不換樹拓撲。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| # | 前作（NCO / Baseline Regime-NCO） | 本文改動 | 作用 |
|---|---|---|---|
| 1 | 對稱協方差 Σ；樹距離與兩階段配置都用它 | Bawa-Lindenberg downside semi-covariance Σ⁻（Eq 2–4），貫穿 ρ⁻（Eq 5）、d⁻（Eq 6）與 intra/inter 配置 | 上行離散不再被當風險；聚類拓撲從「一起動」變成「一起跌」 |
| 2 | 1D Gaussian HMM on SPY returns，無約束 Baum-Welch | 2D 特徵 [R_SPY, z_VIX] + sticky 自轉移先驗 κ=0.30（Eq 7）+ 置信閾值 δ=0.65（Eq 8） | 壓制月度抖動；regime 持續性用 prior 與 threshold 雙閘門 |
| 3 | 單一 Sharpe-max 或 min-variance 目標 | Bull：Sortino utility（γ=1.2，Eq 9）；Bear：min downside variance（Eq 10）；λ_turn=0.015 只在 intra-cluster | 牛市讓動能跑、熊市保本；換手正則分層（inter-cluster 不懲罰） |

### 1.2 ⚡ Eureka

**一句話 trick**：把「風險」的定義從「波動」換成「下跌」，並且讓這個定義一路走到聚類距離；regime 那側則用 prior 把 a_jj 往 1 推、用 δ 把不確定後驗直接忽略。

**直覺**：本文的差異在於把尺本身換掉：Σ⁻ 只累積 min(R−τ,0) 的共動，分組結果變成「誰會跟我一起跌」。若只換最終目標函數、樹仍用對稱相關建，等於先用「大漲也算風險」的尺把資產分組，再在組內用 downside 目標，兩步互相打架（本頁推斷）。HMM 那邊，未約束轉移矩陣允許 a_jj 很小，一個月度小噪聲就翻狀態；κ 把 a_jj 拉向 1，δ 把 [0.35,0.65] 區間的後驗概率直接忽略，兩者疊加把「抖動」壓成「持續」。

### 1.3 信息流

R (11 sectors, rolling 24m) ──► Σ⁻ (Eq 2–4) ──► ρ⁻ (Eq 5) ──► d⁻ (Eq 6)  
　　　　　　　　　　　　　　　　　　　　　　　　　　　│  
　　　　　　　　　　　　　　　　　　　　　　　Ward 聚類 → K=3 clusters  
　　　　　　　　　　　　　　　　　　　　　　　　　　　│  
X = [R_SPY, z_VIX] (expanding) ──► Sticky 2-state HMM (κ=0.30) ──► p_bull  
　　　　　　　　　　　　　　　　　　　　　　　　　　　│  
　　　　　　　　　　　　　　　　　　　　　δ=0.65 置信規則 (Eq 8)  
　　　　　　　　　　　　　　　　　　　　　　　　　　　│  
　　　　　　　　　　　　　　　　　　　　　　　S_t ∈ {Bull, Bear}  
　　　　　　　　　　　　　　　　　　　　　　　　　　　│  
　　　　　　　　　　┌─────────────────────────────────┘  
　　　　　　　　　　▼  
Bull: min ½w'Σ⁻w − γ(w'μ−r_f) + λ_turn‖Δw‖₁　(γ=1.2)  
Bear: min w'Σ⁻w + λ_turn‖Δw‖₁  
　　　　　　　　　　│  
intra-cluster (λ_turn=0.015) → inter-cluster (無懲罰)  
　　　　　　　　　　│  
w_t = w_intra ⊙ w_inter，重整為 simplex

## §2 · 數學層

**📌 Napkin Formula**

- Σ⁻_ij = (1/T) Σ_t min(R_i,t − τ_t, 0) · min(R_j,t − τ_t, 0)
- d⁻_ij = sqrt( ½(1 − ρ⁻_ij) )，ρ⁻_ij = Σ⁻_ij / sqrt(Σ⁻_ii · Σ⁻_jj)
- ã_jj = (a_jj + κ)/(1 + κ)，ã_jk = a_jk/(1 + κ)；S_t = k if P(S_t=k | x_1:t) ≥ δ else S_{t−1}

**複雜度**：Σ⁻ 需對 T 期、N×N 配對累積；Ward 聚類與每期 QP 是主要計算；N=11 很小，計算不是瓶頸。原文未給複雜度階數；瓶頸在 HMM 每期重擬合與 QP solver 的穩定性。

**直覺**：Eq 3 只累積「同時下跌」的乘積，所以單邊大漲不會進入 Σ⁻；Eq 4 用 spectral ridge（ε=10⁻⁶）保證正半定，讓 QP 能解。Eq 7 是 Dirichlet 式偽計數，把對角轉移往 1 拉；Eq 8 是硬閘門，把不確定區間 [0.35,0.65] 的概率變動直接丟掉。

**loss / 訓練細節**：不是梯度訓練。HMM 用 Baum-Welch（EM）在 expanding window 上擬合；Σ⁻ 與 μ 在 24 個月 rolling window 上估；配置用 QP（solver 名稱所讀文本未見）。待核。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為**假設/示意**，不對應論文任何實證數字；τ 簡化為 0（實務用 13-week T-bill）。論文真實數字一律留在 §5。

**假設**：4 個月、3 個 sector，月收益如下：

| 月 | A | B | C |
|---|---|---|---|
| 1 | -0.10 | +0.12 | -0.12 |
| 2 | +0.30 | +0.28 | -0.05 |
| 3 | -0.05 | -0.01 | -0.08 |
| 4 | +0.25 | +0.22 | +0.02 |

**Step 1 — 取 downside deviation min(R−τ,0)**

| 月 | A | B | C |
|---|---|---|---|
| 1 | -0.10 | 0 | -0.12 |
| 2 | 0 | 0 | -0.05 |
| 3 | -0.05 | -0.01 | -0.08 |
| 4 | 0 | 0 | 0 |

**Step 2 — 算 Σ⁻（Eq 3，除以 T=4）**
- Σ⁻_AA = (0.01 + 0 + 0.0025 + 0)/4 = 0.003125
- Σ⁻_BB = (0 + 0 + 0.0001 + 0)/4 = 0.000025
- Σ⁻_CC = (0.0144 + 0.0025 + 0.0064 + 0)/4 = 0.005825
- Σ⁻_AB = 0.000125；Σ⁻_AC = 0.004；Σ⁻_BC = 0.0002

**Step 3 — 算 ρ⁻ 與 d⁻（Eq 5–6）**
- ρ⁻_AC = 0.004 / sqrt(0.003125 × 0.005825) ≈ 0.9375 → d⁻_AC = sqrt(½ × 0.0625) ≈ 0.1768
- ρ⁻_AB = 0.000125 / sqrt(0.003125 × 0.000025) ≈ 0.4472 → d⁻_AB ≈ 0.5257
- ρ⁻_BC ≈ 0.5241 → d⁻_BC ≈ 0.4878

**Step 4 — Ward 聚類**：最小距離是 A-C（0.1768）→ 先併 {A, C}，B 最後併。**樹拓撲 = ({A,C}, B)**。

**Step 5 — 兩個 regime 給出相反的配置**
- Bull（Eq 9，γ=1.2）：A 的上行大，被獎勵；C 的下行方差最大，被懲罰 → 偏向 A。
- Bear（Eq 10）：純 min w'Σ⁻w → Σ⁻_BB 最小（0.000025），權重推向 B。這是「downside 單獨用會集中到低下行方差資產」的機制示意，對應 Table II 中 Asymmetric NCO (No Regime) 的 MDD -20.32。

**Step 6 — 對稱協方差下同一組數據的 ρ_AB 會不同**（本文 Fig 4 的定性對比；具體拓撲翻轉在 11 資產上出現，此處不展開）。要點：換的是**距離尺度**，不只是目標函數。

## §3 · 數據層

- **資產池**：11 個 GICS Select Sector SPDR ETFs——XLK、XLF、XLV、XLE、XLI、XLY、XLP、XLU、XLB、XLRE、XLC。
- **樣本期**：2018 年 8 月至 2025 年 12 月（89 個月）；**OOS 回測**：2020 年 8 月至 2025 年 12 月（65 個月）。
- **頻率**：月度。
- **輔助序列**：SPY（基準）、CBOE VIX（波動因子）、13-week T-bill（ˆIRX，risk-free）。全部來自 Yahoo Finance historical data API。
- **時間邊界分列**：Σ⁻ 與 μ 用 rolling 24 個月窗口估；HMM 在 expanding window 上擬合（特徵只用 t−1 以前的資料）。兩者都是每期重估，不是單次預訓練後凍結。
- **樣本外與容量**：walk-forward，宣稱嚴格非前視；容量／AUM 假設所讀文本未見（TBD）。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| https://github.com/obo-cloud/sp500-asymmetric-nco | N/A（統計模型，無神經網絡權重） | TBD（所讀文本未見） | 中：需 QP solver + HMM/EM 實作，具體依賴未披露 | 公開（Yahoo Finance；作者稱全部歷史 ETF 價格、walk-forward pipeline 與繪圖腳本已釋出） |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄逐行標明**基線本身**（Plain NCO / Baseline Regime-NCO / S&P 500 被動基準）；三者不是同一條 SOTA，Δ 只做「本方法 − 該行基線」。

| 數據集/市場 | Metric | 前SOTA（基線） | 本方法 | Δ |
|---|---|---|---|---|
| S&P 500 11 GICS sectors · 65 OOS months | CAGR (%) — plain NCO | 10.65 | 13.41 | +2.76pp |
| 同上 | Total Ret (%) — plain NCO | 72.99 | 97.74 | +24.75pp |
| 同上 | Max DD (%) — plain NCO | -14.48 | -14.22 | +0.26pp |
| 同上 | Calmar Ratio — plain NCO | 0.74 | 0.94 | +0.20 |
| 同上 | Sharpe Ratio — plain NCO | 0.58 | 0.63 | +0.05 |
| 同上 | Sortino Ratio — plain NCO | 0.97 | 1.07 | +0.10 |
| 同上 | Ann. Vol (%) — plain NCO | 14.03 | 17.71 | +3.68pp |
| 同上 | Downside Vol (%) — plain NCO | 8.40 | 10.51 | +2.11pp |
| 同上 | CAGR (%) — Baseline Regime-NCO | 11.05 | 13.41 | +2.36pp |
| 同上 | Monthly Turnover (%/mo) — Baseline Regime-NCO | 60.78 | 30.48 | -30.30pp |
| 同上 | CAGR (%) — S&P 500 (SPY) | 16.36 | 13.41 | -2.95pp |
| 同上 | Max DD (%) — S&P 500 (SPY) | -23.93 | -14.22 | +9.71pp |
| 同上 | Sharpe Ratio — S&P 500 (SPY) | 0.86 | 0.63 | -0.23 |
| 同上 | Calmar Ratio — S&P 500 (SPY) | 0.68 | 0.94 | +0.26 |

**Table II 消融（逐字）**

| 架構 | CAGR (%) | Max DD (%) | Calmar | Switches |
|---|---|---|---|---|
| Plain NCO (Symmetric, No Regime) | 10.65 | -14.48 | 0.74 | 0 |
| Baseline Regime-NCO (Symmetric, Naive HMM) | 11.05 | -21.81 | 0.51 | 16 |
| Asymmetric NCO (Downside, No Regime) | 11.09 | -20.32 | 0.55 | 0 |
| Proposed Asym-Sticky-NCO | 13.41 | -14.22 | 0.94 | 4 |

**Table III 成本敏感（淨 CAGR %，逐字）**：Plain NCO 10.65 / 10.47 / 10.29 / 10.12 / 9.94（0–20 bps）；Baseline 11.05 / 10.65 / 10.24 / 9.84 / 9.43；Proposed 13.41 / 13.21 / 13.01 / 12.80 / 12.60。

**Table V 統計（逐字）**：Proposed Sharpe 0.63、PSR 1.0000、DSR 0.9976（K=15）；PBO 0.85（S=16，12{,}870 個 split）。

**解讀**

- **真 capability 在哪**：換手從 60.78% 降到 30.48%、regime switches 從 16 降到 4、Max DD 從 SPY 的 23.93% 收到 14.22%、Calmar 0.94 全場最高。這些都在 downside／execution 維度，機制上可解釋：downside 不對上行收費，sticky HMM 不亂翻。
- **不是什麼**：CAGR 13.41% 低於 SPY 16.36%，Total Ret 97.74% 低於 SPY 的總報酬（Table I），Sharpe 0.63 與 Sortino 1.07 低於 SPY 的 0.86 與 1.47。作者在 §VI 自述「its edge is therefore not volatility-scaled」。所以它贏的是 Calmar／回撤／換手，不是波動調整後收益。
- **可能是過擬合或偽穩健**：PBO 0.85 明說 in-sample 排名不穩；作者用 Table IV 的「六配置同軌跡」plateau 論證經濟上無害。但這個 plateau 也可能是狀態機離散跳變造成的 artifact——六個 (κ,δ) 點落入同一條 regime 序列，不等於參數不敏感（本頁推斷）。另有兩處待核：Table I 的 SPY Sharpe 是 0.86、Table V 是 0.87，原文內部不一致；z_VIX 的標準化窗口未披露（若用全樣本 z-score 就有泄漏，推斷）。
- **成本未計的部分**：Table III 只做線性 0–20 bps，所讀文本未見衝擊模型、容量約束、借券／放空成本（策略只做多）。30.48% 月度換手在美股大型 sector ETF 上是否可持續，屬市場特定推斷，不是普遍結論。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§VI caveat 1**：本策略年化波動 17.71%，是全部策略中最高；Sharpe 0.63、Sortino 1.07 低於被動 S&P 500 的 0.86、1.47。作者明說 edge 不在波動調整後。
- **§VI caveat 2 / §V-D**：PBO 0.85 表示 in-sample 排名不可靠；可辯護的主張是 plateau 層級的穩健性，不是單一參數最優。
- **§VI Limitations**：長期只多、月度再平衡；在寬買賣價差或印花稅市場，λ_turn 應調更高以壓制換手；單一市場、65 個月樣本限制統計功效；作者把 PSR／DSR／PBO 重估到更長、多市場樣本列為 future work。
- **§VII**：future work 擴到 multi-asset global factor 與 deep RL 做動態閾值。

### 6.2 本頁推斷的隱含假設（標「推斷」，非作者自述）

- **Regime 依賴（推斷）**：2 態 Bull/Bear 在 2020–2025 樣本內有效（含 2020 疫情衝擊與 2022 Fed tightening 熊市），但未覆蓋更早的系統性信用危機；「慢熊」或「震盪市」可能被二態誤判。限制：只在含一次明確熊市的樣本上驗證過。
- **VIX 標準化窗口（待核）**：所讀文本只寫「standardized CBOE Volatility Index level」，未說 expanding 還是全樣本 z-score。若後者，expanding HMM 的特徵就有前視。圖表／附件待核。
- **容量與成本（推斷）**：線性 bps 成本假設換手與衝擊成比例，所讀文本未見 square-root impact 或容量模型；11 個美股大型 ETF 的高流動性使 30.48% 月度換手可行，但這是市場特定條件，不是普遍結論。
- **downside 單獨使用會加深回撤（原文 ablation 已證）**：Asymmetric NCO (No Regime) 的 Max DD -20.32 比 Plain NCO 的 -14.48 更差、Calmar 從 0.74 掉到 0.55。所以「不對稱協方差」與「sticky regime」是互補而非替代；任何只取前者的推廣都可能複製這個失效。
- **多重檢定（原文已做診斷）**：K=15 trials 的 DSR 0.9976 看起來高，但 PBO 0.85 說明 in-sample 選參不可靠；兩者不矛盾——DSR 是對單一策略的選擇偏差修正，PBO 是對「選哪個配置」的穩定性診斷。
- **數據來源（推斷）**：Yahoo Finance 的 ETF 調整價可能有分割／股息處理問題；原文未討論數據清洗與生存者偏差，11 個 ETF 是否含已退市／重分類需核。此為推斷，圖表／附件待核。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| NCO（López de Prado 2016） | 對稱協方差、無 regime | TBD | 成熟基線 |
| Hierarchical clustering-based asset allocation（Raffinot 2017） | 層次聚類配置、無顯式 sticky regime | TBD | 成熟基線 |
| Baseline Regime-NCO（Supriadi & Siswoyo 2026, arXiv:2610.27751） | 1D HMM on SPY、無 sticky、λ_turn=0.01 | TBD（所讀文本未提代碼） | 本文主要對照 |
| Persistent HMM（Nystrup et al. 2020） | penalize jumps 的 prior 思路，未用於 sector NCO | TBD | 方法 lineage |
| 經典 MVO（Markowitz 1952） | 對稱、error maximizer | TBD | 理論基線 |
| **Asym-Sticky-NCO（本文）** | downside 貫穿樹距離 + 2D sticky HMM | 開源（GitHub） | v0.5 預印本 |

**🎤 Interview Tip**

- **正確答**：「這篇把 downside semi-covariance 貫穿到聚類距離，不只是最終目標函數；並用 κ prior + δ threshold 兩道閘門壓 HMM 抖動。但 ablation 顯示 downside 單獨用會把 MDD 從 -14.48 惡化到 -20.32，必須配 regime；而且 CAGR 13.41% 仍低於被動 SPY 16.36%，贏在 Calmar 0.94 與換手 30.48%。」
- **錯答**：「它用 downside 協方差就解決了 return drag，Sharpe 0.63 打敗 SPY 0.86」——兩處都錯：Sharpe 低於 SPY，且 downside 單獨用回撤更深。

### 7.1 可證偽預測

1. **2026 年內**若把同一 pipeline 擴到非美市場或全球 factor 且不重調 κ／δ，預期 whipsaw 降幅會小於 75.0%——因為 75.0% 是在 65 個月單一市場上測出的，跨市場的 regime 切換頻率不同（本頁推斷，尚待驗證）。
2. 若用 square-root impact 模型替代線性 bps，30.48% 月度換手的淨 CAGR 預期低於 Table III 的 12.60%@20bps（本頁推斷，尚待驗證）。
3. 若把 z_VIX 改為 expanding-window 標準化後結果顯著改變，則原結果存在前視泄漏；此檢驗可在 2026 年內完成。
4. 若樣本延長涵蓋更早的系統性信用危機，預期 Bull/Bear 二態會出現第三態需求或誤判——這是 2 態假設的直接否證點（本頁推斷，尚待驗證）。

## §8 · For the Reader

- **因子研究員**：把 downside semi-covariance 當聚類距離是一個可測變體；重點測「換距離」vs「只換目標函數」的增量，而不是只看最終 CAGR。注意 downside 單獨用會加深回撤（Table II）。
- **高頻執行**：不是你的菜——月度、30.48% 換手。但兩層 turnover regularization（intra 有 λ_turn=0.015、inter 無）值得借用：自適應地在「組內調倉」加懲罰、在「組間配置」放行，可減少不必要的跨簇搬倉。
- **組合配置**：Calmar 0.94、Max DD -14.22 適合 downside-averse mandate；但 Sharpe 0.63 低於 SPY 0.86，不是波動調整後的贏家。若委託人看 Calmar／回撤，這是可考慮的；若看 Sharpe，不是。
- **LLM-agent**：SPY + VIX 二維 HMM 可當 macro regime gate；κ 與 δ 是兩個可解釋旋鈕，比黑箱分類器更容易在 agent pipeline 裡做審計。
- **RL 策略**：作者自己在 §VII 指出把 δ 換成 learned threshold 是 future work；但 PBO 0.85 提醒——在 65 個月單一樣本上學閾值，很容易學到樣本內的路徑。
- **研究學生**：這篇是「ablation 顯示 component 交互」的好教材——單一 component 的邊際貢獻可能為負（downside 單獨 +44 bps CAGR 但 MDD 惡化），必須看組合而非孤立指標。同時注意 Table I 與 Table V 的 SPY Sharpe 0.86 vs 0.87 內部不一致，是讀預印本時要標記的核對點。

## References

- Supriadi (2026). *Mitigating Return Drag and Regime Whipsaws in Sector Rotation: An Asymmetric Downside-Aware Nested Clustered Optimization Framework.* arXiv:2611.05310. https://arxiv.org/abs/2611.05310v1
- López de Prado, M. (2016). Building diversified portfolios that outperform out of sample. *The Journal of Portfolio Management* 42(4), 59–69.（NCO 與層次聚類 lineage）
- Bawa, V. S. & Lindenberg, E. B. (1977). Capital market equilibrium in a mean-lower partial moment framework. *Journal of Financial Economics* 5(2), 189–200.（downside semi-covariance）
- Nystrup, P., Lindström, E. & Madsen, H. (2020). Learning hidden Markov models with persistent states by penalizing jumps. *Expert Systems with Applications* 150, 113307.（sticky prior）
- Supriadi & Siswoyo, B. (2026). Market regime-based multi-asset portfolio optimization using nested clustered optimization for tactical asset allocation. arXiv:2610.27751.（Baseline Regime-NCO）
- Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio. *The Journal of Portfolio Management* 40(5), 94–107.
- Bailey, D. H., Borwein, J. M., López de Prado, M. & Zhu, Q. J. (2017). The probability of backtest overfitting. *The Journal of Computational Finance* 20(4), 39–69.
- 代碼：https://github.com/obo-cloud/sp500-asymmetric-nco
- 數據：Yahoo Finance historical data API（https://finance.yahoo.com）