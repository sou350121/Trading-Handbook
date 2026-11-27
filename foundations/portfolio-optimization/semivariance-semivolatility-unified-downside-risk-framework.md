<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Semivariance/Semivolatility Unified Downside-Risk Framework 解構（Semivariance/Semivolatility Unified Downside-Risk Framework）

> **來源發布**：2026-11-24 · （無 venue） · arXiv [2611.32638](https://arxiv.org/abs/2611.32638)
> **arXiv 原文**：[Semivariance and Semivolatility in Portfolio Selection: A Unified Framework for Downside-Risk Portfolio Models](https://arxiv.org/abs/2611.32638v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 中長周期 × 監督回歸（μ 作為輸入）× 組合執行優化 × 人機協同可解釋」。它解的 prior gap 不是「downside 比 variance 好」，而是：**downside 風險的齊次度（degree）決定了哪一類組合模型可解**；把 threshold 顯式化後，risk parity、最大分散、Sortino 三類看似各自為政的非凸問題，塌縮成同一個凸規劃。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把 downside 風險按「對權重的正齊次度」分類：semivariance 是 degree 2（配最小風險），semivolatility 是 degree 1（配比值型與風險預算型），並對三個 threshold（μᵀx、0、R^I）給出非退化條件。②核心 trick 是 degree-1 齊次 + 可微性 ⇒ Euler 分解的風險貢獻可加、且 ratio 問題可用 Schaible 變換降為凸規劃；log-barrier 風險預算程式因此真的達到指定 budget。③對「組合執行優化」軸★的意義：downside 版的 risk parity／MDP／Sortino 不再需要非凸最小平方或全域優化機器。④關鍵實證：把波動率換成半波動率，Sharpe 差在 risk parity 上為 DowJones - 0.0010、Nasdaq100 - 0.0033，Holm p 皆為 1.000 —— 即**沒有顯著差異**。

**X-Ray.** 這篇的真正貢獻在「可解性分類學」，不在 alpha。它把 downside 文獻裡長年被當成「variance 的直接延伸」的鬆散說法，換成一句可操作的判準：degree 決定模型類。放回五軸 Pareto，它讓「組合執行優化」這一軸上的 downside 分支從 heuristic 區（Bai et al. 2016 的非凸最小平方、Barkhagen et al. 2023 的全域優化、Kapsos et al. 2014 的逐案 reformulation）搬進凸錐區，代價是必須接受 non-degeneracy 假設與 threshold 的模型選擇自由。它打不開的 envelope 很清楚：這是**給定 μ 與情境矩陣之後的配置層**，完全不處理報酬預測、regime、容量與市場衝擊；實證只用美股兩個指數的現行成分股，作者自己承認 survivorship 與 look-ahead。對量化讀者最值錢的一點其實是負面結果：在 18.6 與 14.1 年的樣本上，downside 化本身沒有帶來可檢出的 Sharpe 改善，真正動 Sharpe 的是 threshold 慣例。這意味著把「換 downside 風險度量」當成改進賣點的策略提案，需要先過 equivalence margin 這一關。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作做法 | 本文改動 | 帶來的直接後果 |
|---|---|---|---|
| 風險度量的分類軸 | downside 模型被當成 variance 對應物的「直接延伸」，不區分結構性質 | 以**正齊次度**分類：semivariance degree 2、semivolatility degree 1（Table 1） | degree 2 配最小風險，風險預算亦支援 degree 2（但風險貢獻加總為 2 S_P²、且與其根號同解）；degree 1 才讓風險貢獻加總為風險、且比值對權重縮放不變 |
| threshold 的處理 | τ 多半隱含在模型裡，未檢查退化 | 顯式給 Assumption 2（相對 τ 的非退化），並證明**不**由 below-mean 版（Assumption 1）推出 | benchmark 可完全複製（R^I = Rᵀx̄）時 ρ_{R^I}(x̄)=0，Sortino／分散率可無界、風險預算程式退化 |
| 比值模型的可解性 | risk parity 非凸最小平方、MDP 需全域優化、Omega 逐案 reformulate | Lemma 2（Schaible 型 reduction）：degree-1 凸、在錐上為正 ⇒ 分數規劃等價於凸規劃 | MDP 與 Sortino 共用同一個 `min ρ(y) s.t. cᵀy = 1` 模板，threshold 變成「建模選擇」而非「計算障礙」 |
| 風險預算的可達性 | 對任意凸風險度量「宣稱」log 形式可用，但未處理不可微 | Lemma 1：Assumption 1 下 S_P 連續可微、給出顯式梯度，且**不要求報酬分佈連續** | L¹ 類比（CVaR）在離散分佈上不可微的坑被繞過；log 程式確實達到 b_i 指定的 budget |

### 1.2 ⚡ Eureka

**同一個 downside 風險，取平方是 degree 2、取根號是 degree 1；degree 決定 Euler 分解能不能加總，於是「風險平價／最大分散／Sortino」三類非凸問題塌縮成同一個凸規劃。**

直覺：degree 2 的量（變異數、半變異數）在縮放權重時以平方成長，比值型目標失去尺度不變性；degree 1 的量（波動率、半波動率）才讓「每單位風險的報酬」與「風險貢獻佔比」這兩個概念同時良定義。風險平價要的是後者，最小風險要的只是 minimizer 相同（半變異數與其根號同解）。

### 1.3 信息流

```
daily total-return panel R  (n assets × D_L scenarios, D_L = 504)
        │
        ├─ threshold τ ∈ { μᵀx , 0 , R^I_t }        ← 本文認為「該被寫出來」的旋鈕
        ▼
  ρ_τ(x) = ‖ ((R − τ·1)ᵀx)_− ‖_{L²}          convex, positively homogeneous of degree 1
        │
   ┌────┴───────────────────┬───────────────────────────┬──────────────────────────┐
   ▼ degree 2               ▼ degree 1 (ratio)          ▼ degree 1 (risk budgeting)
 min S²_τ(x)               max cᵀx / ρ_τ(x)            min ρ(y) − Σ b_i log y_i
 quadprog                  ── Lemma 2 (Schaible) ──►    ── KKT + Euler ──►
                           min ρ(y)  s.t.  cᵀy = 1      RC_i(x*) = b_i · ρ(x*)
                                  │                            │
                                  └──────── x = y / 1ᵀy ────────┘
                                               ▼
                         weights held 21 days, drifted; 50 bp on turnover
```

## §2 · 數學層

📌 **Napkin Formula**

1. 定義（degree 區分）：`S²_τ(X) = E[(X−τ)_−²]`、`S_τ(X) = (E[(X−τ)_−²])^{1/2}`；在 τ≠0 時 S_τ 對 portfolio return 並非正齊次，齊次度是對權重而言：portfolio 層 `ρ_τ(x) = ‖((R−τ1)ᵀx)_−‖_{L²}` 對權重為 degree 1，而 `S²_τ` 對權重為 degree 2。
2. 梯度（Lemma 1）：`∂S_P/∂x_i = −E[(R_i−μ_i)((R−μ)ᵀx)_−] / S_P(x)`；在 Assumption 1 下對所有非零 long-only 組合成立。
3. Log 風險預算（Problem 4）：`min_y S_P(y) s.t. Σ b_i log y_i ≥ γ` ⇒ `RC_i(x*) = b_i S_P(x*)`，且 x* 與 γ 無關。
4. 比值降維（Lemma 2）：`max{cᵀx/ρ(x)}` ⟺ `min{ρ(y) : cᵀy = 1}`，`x* = y*/1ᵀy*`。

**直覺**：degree-1 凸函數的 Euler 恆等式把「總風險」拆成可加的資產貢獻；log 約束則把「貢獻佔比 = budget」這個非線性要求，轉成一個有唯一解、且對尺度不敏感的凸問題。ratio 那邊則是觀察到：在齊次度 1 之下，把分母歸一化即可消去分母，分數規劃變成純風險最小化。

**複雜度／求解細節**：無訓練、無 loss；全部是凸規劃。原文實作在 MATLAB R2025b：MV、MaxSharpe、MDP-Vol 用 `quadprog`（interior-point-convex），其餘用 `fmincon`（SQP + 解析梯度），起點為等權或（MDP-SVol 的）反半波動率權重；tolerance 對二次與比值程式為 `10^{-12}`、其餘 `10^{-10}`；數值時間複雜度未披露。Lemma 1 的證明附帶說明梯度是 Lipschitz 的。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部是假設／示意數字，不是論文實證結果。** 目的是把「degree-1 ⇒ Euler 貢獻可加 ⇒ 風險平價」手算一遍。

**假設**：2 個資產、4 個等機率情境（機率各 1/4）。

| 情境 | A | B |
|---|---|---|
| 1 | −10% | −4% |
| 2 | −2% | 2% |
| 3 | 6% | 4% |
| 4 | 10% | 6% |

**Step 1 — 去均值**：μ_A = 1%、μ_B = 2%。ξ_A = (−11, −3, 5, 9)、ξ_B = (−6, 0, 2, 4)（單位 %）。

**Step 2 — 資產層半波動率**：A 的負部平方和 = 121 + 9 = 130，除以 4 = 32.5，開根號 S_μ(A) ≈ 5.70%；B 的負部平方和 = 36 + 0 = 36，除以 4 = 9，S_μ(B) = 3.00%。

**Step 3 — 組合層（x = 0.5/0.5）**：組合報酬 = (−7, 0, 5, 8)，均值 1.5%；z = ξᵀx = (−8.5, −1.5, 3.5, 6.5)，z_− = (8.5, 1.5, 0, 0)。S_P = √((72.25+2.25)/4) = √18.625 ≈ 4.32%。

**Step 4 — 檢查 subadditivity**：0.5×5.70 + 0.5×3.00 = 4.35 > 4.32。分散率 DR = 4.35/4.32 ≈ 1.008 —— 這例子**不滿足** additivity（Theorem 1 的條件），所以 Proposition 4 的閉式解不適用。

**Step 5 — 風險貢獻（Euler）**：E[ξ_A z_−] = (1/4)[(−11)(8.5) + (−3)(1.5)] = −24.5 ⇒ ∂_A ≈ 24.5/4.32 ≈ 5.68，RC_A ≈ 2.84；E[ξ_B z_−] = (1/4)[(−6)(8.5)] = −12.75 ⇒ RC_B ≈ 1.48。RC_A + RC_B ≈ 4.32 = S_P ✔（degree 1 的 Euler 恆等式成立；若換成 degree 2 的半變異數，貢獻會加總到 2 倍）。

**Step 6 — 風險平價解**：50/50 時 A 佔風險 2.84/4.32 ≈ 66%，遠超 50% budget。把 x_A 往下壓：x_A = 0.30 時 A 佔 44.8%，x_A = 0.40 時佔 56.0%，故 1:1 風險預算的解約在 **x ≈ (0.35, 0.65)**。順帶一提，反半波動率權重 (1/5.70 : 1/3.00) 正規化後約為 (0.345, 0.655)，與此玩具解幾乎重合——這純屬本例巧合，additivity 不成立時沒有定理保證。

## §3 · 數據層

| 指數 | 全名 | n | 國家 | 時段 |
|---|---|---|---|---|
| DowJones | Dow Jones Industrial Average | 30 | USA | Mar 2008 – Nov 2026 |
| Nasdaq100 | Nasdaq-100 | 84 | USA | Oct 2012 – Nov 2026 |

- **來源**：FactSet，2026 年 11 月 6 日下載；日頻 total-return 價格，已調整拆股、股利、spin-off；指數本身也用 total-return 版本。資料與結果公開於 francescocesarone.com，受數據商授權條款約束。
- **樣本建構**：簡單報酬 `r_it = (P_it − P_i,t−1)/P_i,t−1`。宇宙為**下載日的現行成分股**中價格序列完整者，保留 30 of 30 與 84 of 101 檔；Nasdaq100 的 FER-US 因 2020 前多數日子價格不變而被手工剔除。**宇宙是靜態的**。
- **滾動窗口**：學習窗 D_L = 504 trading days、持有窗 D_H = 21 trading days；持有期內不交易、權重隨價格漂移；產生 Q = 199（DowJones，約 2010 年 3 月起）與 Q = 144（Nasdaq100，約 2014 年 10 月起）個窗口，皆於 2026 年 10 月結束；out-of-sample 觀測數 N^out = 4179 與 3024。
- **成本與其他假設**：換手按 50 basis points 比例成本（c_tc = 0.0050），首窗不計成本；r_f = 0；基準配置無個別權重上限（ub = 1）。補充材料另報 0、25、50、100 bp 下的 Sharpe。
- **已知偏誤（作者自述）**：現行成分股 ⇒ survivorship 與 look-ahead bias，且該偏誤**對所有策略相對指數都有利**，故指數只作參考，結論只在策略間比較；四個以 R^I 為 threshold 的策略更需謹慎，因為指數進入了它們的優化本身。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 所讀文本未見公開 repo；作者稱 MATLAB code「available from the corresponding author on reasonable request」 |
| Checkpoint | 不適用（無學習權重）；但有「已存權重」用於補充材料 S.7.2 的殘差核算 |
| License | 程式：未披露；數據：FactSet 授權條款 |
| 複現難度 | 中——需 MATLAB R2025b 與 Optimization Toolbox（`quadprog` / `fmincon`）；理論部分無隨機性，數值部分窗口與 solver 設定均明列，可重現性高 |
| 數據可得性 | 價格需 FactSet 授權；作者提供整理後的資料集（fetched 2026-11-06） |

## §5 · 評測 / Benchmark

> 表格的「前SOTA」欄是**基線**（同模型類的 variance-based 或 below-mean 版本，逐字取自原文 Table 4／5），不是領域 SOTA；本文不做跨方法 SOTA 排名。Δ 為原文 Table 6 逐字報告的 within-family 差值（該表以 1/N^out 變異數估計量計算，「the differences may deviate by 0.0001 from Tables 4–5」）。

| 數據集/市場 | Metric | 前SOTA（基線） | 本方法 | Δ |
|---|---|---|---|---|
| DowJones | Sharpe | MDP-Vol 1.028 | MDP-SVolμ 1.071 | 0.0429 |
| Nasdaq100 | Sharpe | MDP-Vol 1.109 | MDP-SVolμ 0.997 | - 0.1120 |
| DowJones | Sharpe | RP-Vol 1.040 | RP-SVolμ 1.039 | - 0.0010 |
| Nasdaq100 | Sharpe | RP-Vol 1.127 | RP-SVolμ 1.124 | - 0.0033 |
| DowJones | Sharpe | MV 0.774 | MSVμ 0.767 | - 0.0077 |
| Nasdaq100 | Sharpe | MV 0.869 | MSVμ 0.848 | - 0.0209 |
| DowJones | Sharpe | MaxSharpe 0.885 | MaxSortinoμ 0.860 | - 0.0247 |
| Nasdaq100 | Sharpe | MaxSharpe 0.824 | MaxSortinoμ 0.811 | - 0.0133 |
| DowJones | Sharpe | RP-SVolμ 1.039 | RP-SVol_RI 0.891 | - 0.1481 |

**顯著性（原文 Table 6，Holm 調整後）**：Panel A 各列的 p_Holm 多為 1.000，只有 Nasdaq100 的 MDP 對為 0.292；Panel B（τ=0 減 τ=μᵀx）全部不顯著；Panel C 唯一通過 Holm 的是 DowJones risk parity 的 benchmark threshold 效應，p_Holm = 0.006（MDP 對應為 0.137）。

**哪些 Δ 是真 capability？** 幾乎沒有。Panel A 的訊息是**等價而非改進**：risk parity 上兩者差距窄到可以宣告等價——原文以成本換算的 margin δ = 0.018（DowJones）與 0.021（Nasdaq100）做 two one-sided tests，結論是 RP-Vol 與 RP-SVolμ 在兩個資料集上等價；但作者明說這條規則「chosen after the analysis and not pre-specified, so the equivalence is exploratory」。其他三類的置信區間「too wide either to separate the two risk measures or to establish their equivalence」，即**既不能說有效也不能說無效**。

**哪些 Δ 可能是過擬合／偏誤／成本未計？**（1）MDP 家族的方向在兩個資料集上反號（DowJones 為正、Nasdaq100 為負），且 Nasdaq100 的點估計絕對值最大正是最不顯著的一列，典型的樣本內排序在樣本外翻轉。（2）DowJones 的描述性頭部（MDP-SVol0 與 MDP-SVolμ 的 Sharpe 1.089 與 1.071 高於 EW 的 1.049 與 MDP-Vol 的 1.028）；MDP-SVol0 未見對應的 within-family 檢定，MDP-SVolμ 雖有檢定但 p_Holm 1.000，且 EW 在 Nasdaq100 直接排第一（1.156），不宜當作能力證據。（3）成本只計 50 bp 比例換手成本，無市場衝擊；高換手族（DowJones MaxSortinoμ 與 MaxSortino0 的 Turn 為 44.50 與 44.99；Nasdaq100 為 50.04 與 50.58）在真實衝擊下會更差，這方向只會讓 downside 化的比較更不顯著。（4）survivorship 對所有策略有利，作者明示指數只作參考。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位至原文章節）

- **§4.1**：宇宙為現行成分股，存在 survivorship 與 look-ahead bias（引 Brown et al. 1992；Elton et al. 1996），對所有策略相對指數有利；因此指數只作參考、結論取自策略間比較；以指數報酬為 threshold 的四個策略「should be read with particular caution」，因為指數進入了它們的優化。
- **§4.4.3**：等價檢定的 margin 規則是分析後才選定、非預先指定，故等價結論屬 exploratory；且檢定「conditional on the out-of-sample return series produced by the backtest」。
- **§4.4.3**：設計上無法把「benchmark 本身」與「其單邊處理」分離；對 MaxSortino_RI，分子同時變成 expected active return，故收斂現象不能全歸因於共同基準。
- **§5**：結論建立於現行成分股資料集；未來研究應改用 point-in-time 成分股，並把分類學推廣到更高階 lower partial moments。
- **§S.7.1**：Proposition 4 的閉式解只適用於「no downside diversification」的極限情形，作者自評「structural relevance only」。
- **§2.2**：若報酬對每個 long-only 組合都對稱於均值（例如橢圓分佈），below-mean 模型與 variance 版重合——即方法在對稱分佈下**結構上不可能**有差異；§2.3 與 §3.5 的比值與風險預算結果則掛在 Assumption 1／2 上。
- **§S.2**：對 EW／MV／指數的十七個比較使用未調整 p 值，需考慮多重性；deflated Sharpe ratio 與 SPA 類檢定未實作。

### 6.2 推斷的隱含假設（本頁推斷，非作者主張）

- **Regime 依賴（推斷）**：兩個資料集皆為美股、皆為大型權值股、且都終止於同一月份；無法區分「downside 化無效」與「這段期間 downside 與 variance 的相關結構特別高」。跨 regime（如 2008 前後分開、或非美市場）的行為未驗證。
- **容量與成本（推斷）**：僅有比例換手成本，無衝擊模型、無借券／放空限制以外的微觀結構；Nasdaq100 成分股流動性較好，結論對小市值或高衝擊資產不可外推。容量上限未披露、未估算。
- **估計誤差集中於 μ（推斷）**：Mean-based 家族（MaxSharpe、MaxSortinoμ、MaxSortino0）在兩個資料集上都是報酬最高、風險指標最差、換手最高的一群，這是均值估計誤差被優化放大的典型指紋，而非 downside 度量本身的問題。
- **近似恆等可能來自資產層相關結構（推斷）**：RP-Vol 與 RP-SVolμ 的平均 L₁ 權重距離只有 0.030（DowJones）與 0.042（Nasdaq100），作者明說這是「a feature of the observed returns, not an algebraic identity」；推論是：在低偏態、高共同因子的權值股宇宙裡，downside 與兩側風險幾乎同構。這一條若成立，則本文的負面結果不能外推到高偏態標的（如期權賣方、小型股、危機資產）。
- **靜態宇宙不是生存者偏差的替代（推斷）**：作者以「只在策略間比較」緩解偏誤，但靜態宇宙同時抹掉了成分股變動的換手與流動性事件，對換手估計（Turn）與成本敏感度有獨立影響。
- **輔助數據缺失**：圖片／附件（Figure S.1 財富曲線、原文色彩排序表）未在本次輸入中提取，相關視覺結論待核。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Markowitz MV／波動率版 risk parity（Maillard et al. 2010） | degree 2 vs degree 1；本文的直接基線 | 古典、演算法公開 | 本文視為基線，Panel A 顯示差異不顯著 |
| MAD risk parity（Ararat et al. 2024） | additivity 條件強弱：MAD 弱、below-mean semivolatility 嚴（需 downside proportionality） | 論文公開 | 本文 Lemma 3／Theorem 1 為嚴格強化，並給出共同的 downside factor 形式 |
| CVaR risk budgeting（Boudt et al. 2013；Cesarone & Colucci 2018） | L¹ 型在離散分佈上不可微 vs L² 型可微且不需分佈連續 | 演算法公開 | 本文 Lemma 1 針對此坑 |
| MDP 需全域優化（Barkhagen et al. 2023） | 等齊次度凸函數比值 vs degree-1 凸函數比值 | 論文公開 | Lemma 2 讓本文免於全域優化機器 |
| Omega ratio 的 LP 解（Kapsos et al. 2014） | 個案 reformulation vs 統一 reduction | 論文公開 | 本文視 Omega 為 Lemma 2 的特例 |
| 非凸最小平方 risk parity（Bai et al. 2016） | 非凸 vs 凸 | — | 本文在 long-only + Assumption 1 下給出凸形式 |

🎤 **Interview Tip**

- **正確答**：「關鍵不是 downside 比 variance 好，而是 degree-1 齊次性讓 Euler 貢獻可加、比值尺度不變，於是 risk parity／MDP／Sortino 共用同一個凸模板。實證上把波動率換成半波動率在 Holm 調整後沒有顯著差異，risk parity 那對甚至在成本 margin 下可宣告等價；真正動 Sharpe 的是 threshold 慣例——benchmark threshold 在 DowJones 的 risk parity 上把 Sharpe 拉低 0.148，這是唯一通過 Holm 的差異。所以做 downside 模型要先聲明 threshold。」
- **錯答**：「半方差只罰下行所以一定優於方差」——（a）minimum-risk 下 semivariance 與其根號同解，與 variance 版的差在兩個資料集上都不顯著（例如 DowJones 的 MSVμ - MV 為 - 0.0077，p_Holm 1.000）；（b）若報酬對每個 long-only 組合對稱於均值（如橢圓分佈），below-mean 模型與 variance 版數學上重合，方法根本沒有自由度。

### 7.1 可證偽預測（本頁研究假說，尚待驗證）

1. **2027-06 前**：若把宇宙改為 point-in-time 成分股，MDP-SVolμ 相對 MDP-Vol 的 Sharpe 差在 Nasdaq100 上的負號會消失或縮小到不顯著（目前為 - 0.1120、p_Holm 0.292，且與 DowJones 反號）。判準：同一套 block bootstrap 下 p_Holm > 0.05。
2. **2027-06 前**：把持有窗改為 D_H = 63 天重跑，RP-SVolμ 與 RP-Vol 的平均 L₁ 權重距離仍與基線的 0.030 與 0.042 同量級。若顯著放大，則「近似恆等」是月度再平衡的產物而非資產結構。
3. **2027-12 前**：在高偏態宇宙（如期權賣方或小型股）重複同一實驗，semivolatility 與 volatility 的權重距離會**大於**上述 0.030／0.042，且 downside 化的 Sharpe 差可能轉為顯著。此為本文結論的主要外部效度風險。
4. **2027-12 前**：若加入平方根型市場衝擊成本，高換手族的排序會進一步下沉（DowJones 的 MaxSortinoμ／MaxSortino0 換手為 44.50／44.99，Nasdaq100 為 50.04／50.58），使 Panel A 更難拒絕等價虛無。

## §8 · For the Reader

- **組合配置／風險預算**：可以直接抄的是 Lemma 2 的模板與 log 風險預算的 KKT 條件——它讓 downside 版 risk parity 用 `quadprog`/`fmincon` 就能跑到 solver 精度（原文 S.7.2 報風險貢獻相對偏差不超過 `3\times 10^{-6}`，benchmark threshold 為 `1.6\times 10^{-4}`）。但別把「換 downside 度量」當賣點，先量 equivalence margin。
- **因子研究員**：這篇提供了一個乾淨的負面對照——同一個 alpha 訊號下，換風險度量不改變 Sharpe，改變 threshold 才改變 Sharpe。做 downside 因子時應把 threshold 當成一個顯式的策略參數做敏感度表，而不是埋在實作裡。
- **高頻執行／交易成本**：唯一可用的是換手向量，以及 S.1.1 的成本敏感度（0→100 bp 時 EW 的 Sharpe 降約 0.03／0.04，MV 降 0.12 到 0.15，而 RP-SVolμ 與 RP-Vol 的差變化不超過 0.001）。這給了一個判斷：低換手的 risk parity 家族對成本不敏感，高換手的 max-ratio 家族才是執行層要優化的對象。
- **RL／ML 策略研究者**：把這篇當成「動作層」的候選。你的模型輸出 μ 或分數後，這個框架可以作為一個可微、凸、且對 threshold 顯式的配置頭；它與監督回歸解耦，不會偷你的預測容量。注意 Assumption 1／2 是硬約束，需要在每個窗口用線性規劃檢查（原文用 `min_{x∈Δ} max_t (τ_t − r_tᵀx)` 驗證）。
- **研究學生**：讀 §2.2 的 Table 1 與 §3.3 的 Lemma 2 就夠掌握骨架；真正值得模仿的是 §S.7.1 那張「定理—假設—最近前驅—status（recalled / specialized / extended / new）」對照表，這是把綜述型貢獻寫清楚的方法論範本。

## References

- 原論文：Cesarone, F., Corradini, M., Giunta, N. (2026). *Semivariance and Semivolatility in Portfolio Selection: A Unified Framework for Downside-Risk Portfolio Models.* arXiv:2611.32638v1 [q-fin.PM], 24 Nov 2026. https://arxiv.org/abs/2611.32638v1
- 補充材料：同文 Supplementary Material（Sections S.1–S.7、Appendix S.A），本次輸入含其可讀文本。
- Lineage — 下行風險：Markowitz (1959); Hogan & Warren (1974); Bawa (1975, 1976); Fishburn (1977); Ogryczak & Ruszczyński (1999, 2001); Fischer (2003); Estrada (2006, 2008); Cesarone et al. (2026).
- Lineage — 風險預算與分散：Choueifaty & Coignard (2008); Maillard et al. (2010); Bruder & Roncalli (2012); Spinu (2013); Bai et al. (2016); Cetingoz et al. (2024); Ararat et al. (2024); Barkhagen et al. (2023); Cesarone et al. (2025).
- Lineage — 分數規劃與統計推論：Schaible (1974); Stoyanov et al. (2007); Kapsos et al. (2014); Politis & Romano (1992); Ledoit & Wolf (2008); Holm (1979); DeMiguel et al. (2009).
- 資料與程式：FactSet (2026)；資料集 https://www.francescocesarone.com/data-sets（accessed 6 November 2026）。