<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=人机协同可解释 -->

# DRP-ES (Diversified-Risk-Perception Expected Shortfall) 解構

> **來源發布**：2026-11-09 · （無 venue） · arXiv [2611.11803](https://arxiv.org/abs/2611.11803)
> **arXiv 原文**：[Diversified-Risk-Perception Risk Measures and Portfolio Selection](https://arxiv.org/abs/2611.11803v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 model risk 從 worst-case 的單點保守估計，改成由模型 disagreement 驅動的 KL 模糊集，並用 exponential aggregation 把 asset diversification 與 model diversification 接在同一個風險度量；五軸落在日頻、量價表格、風險擇時、可解釋人機協同。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 把多個風險模型與多資產損失放進同一個 DRP 風險度量，給出 DRP-ES 並用於日頻組合選擇。② 核心 trick 是 exponential aggregation 的 θ 連續連接 linear 與 max，並用 Wasserstein 不一致度 D_t 調 KL 半徑，讓模型權重 q 從均分滑向悲觀模型。③ 對「風險擇時」軸★，它把 model risk 變成可調、可解釋的風險預算，而不是單模型硬切。④ 中國 A 股 April 1, 2020 至 August 12, 2026、α=0.95：DRP Sharpe 0.9506、MDD 12.2735 %，對 CSI 300 的 0.2065、45.6025 %；但 April 1, 2013 至 August 12, 2019 中國 DRP Sharpe 0.8225 低於 HS 1.0362，穩定優勢不是全域。

**X-Ray.** DRP-ES 在五軸上不是新 alpha 模型，而是把 model risk 塞進風險度量的風險擇時層：日頻、量價、可解釋權重。它解了兩個舊坑：線性聚合對大損失不夠敏感、max 聚合忽略其他資產；worst-case ES 只看最悲觀模型、equal weights 忽略 disagreement。Eureka 是用 exponential aggregation 的 θ 把線性與 max 連成連續譜，再用 Wasserstein 不一致度 D_t 調 KL 半徑，讓 q 從均分滑向悲觀模型。它打不開的 envelope：候選模型集之外的新尾部機制、非日頻／非股票資產、交易成本與市場衝擊、以及以驗證集 Sharpe 調參帶來的選擇偏差；Table 5 南韓 worst-case 權重奪冠就是反例。對量化讀者，它的價值是可審計的 risk overlay，不是保證打敗單模型。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 構件 | 前作／基線 | DRP-ES 改動 | 讀者意義 |
|---|---|---|---|
| 損失聚合 | linear weighted aggregation；maximum aggregation | exponential aggregation \(F_{\theta,\pi}(\mathbf{X})=\theta\log(\sum_i \pi_i e^{X_i/\theta})\)；θ↓0 趨近 max，θ→∞ 趨近 linear | 用一個 θ 控制「重視大損失」與「保留其他資產貢獻」的平衡 |
| 模型權重 | worst-case ES；thick modelling 的 equal weights | 以 KL 模糊集 \(\mathcal{U}_t(D_t)\) 包住均分權重 \(\omega_0\)，半徑 \(\delta_0+\delta_1D_t\)，再優化 q | 模型分歧小則近均分，分歧大則偏悲觀；不是一刀切 worst-case |
| 風險度量 | GRM 的一般理論框架 | 具體化為 DRP-ES，給出凸性、cash invariance、參數敏感性上界，並做三市場回測 | 從「存在性／公理化」走到「可數值求解＋可回測」 |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

**用 exponential aggregation 的 θ 把 linear 與 max 連成一條連續譜；再用 Wasserstein 不一致度 D_t 調 KL 半徑，讓模型權重從均分滑向最悲觀模型。** 直覺是：資產層面用 θ 決定「大損失是否主導」，模型層面用 D_t 決定「分歧時是否更保守」；兩個旋鈕都不需要事先選定單一模型或單一聚合函數。

### 1.3 信息流 ASCII 圖

股票日損失 X_{i,t}
   │
   ├─ HS / GARCH / EVT / GARCH-EVT / FHS → M 組 N 個情境
   │        └─ 聚合 F_{θ,π}(X) → Y_{m,t}^{(n)}
   │                 └─ 兩兩 Wasserstein → D_t
   │
   └─ KL 半徑 ρ(D_t)=δ0+δ1D_t
            └─ q∈U_t(D_t) 優化
                     └─ DRP-ES = sup_q ES_α(Y|P_t^q)
                              ├─ 個股風險排序 → 每產業選 M_stock
                              └─ 產業風險 Ψ_j → 反比配置 w_j

## §2 · 數學層

📌 **Napkin Formula**

\[
F_{\theta,\boldsymbol{\pi}}(\mathbf{X})=\theta\log\left(\sum_{i=1}^{d}\pi_i e^{X_i/\theta}\right),
\qquad
\Psi_{t,\alpha}(D,\mathbf{X})=\sup_{\mathbf{q}\in\mathcal{U}_t(D)}\operatorname{ES}_\alpha\left(\Lambda_\theta(\mathbf{X})\mid P_t^{\mathbf{q}}\right)
\]

\[
\mathcal{U}_t(D)=\left\{\mathbf{q}\in\Delta_M:\operatorname{KL}(\mathbf{q}\mid\boldsymbol{\omega}_0)\le \delta_0+\delta_1D_t\right\},
\qquad
D_t=\frac{\overline{W}_t}{s_t+\varepsilon}
\]

**直覺**：第一式把 d 維損失壓成純量，θ 小則接近 max、θ 大則接近線性加權；第二式在 KL 球內找最壞的模型混合，但球的半徑由模型間 Wasserstein 分歧 D_t 決定。**複雜度**：每個時點需對 M 個模型各抽 N 個情境，再解一個含 q 與 ES 對偶變數 u 的凸優化；原文未披露 Big-O 或求解時間。**Loss／訓練細節**：沒有神經網絡式訓練損失；δ0、δ1 與各候選模型參數在驗證集以最大化 Sharpe 調優，測試期固定。DRP-ES 本身是 sup over q 的凸優化，不是梯度訓練。

理論性質：exponential aggregation 滿足 (L0)-(L3)，不一般滿足 (L4)、(L5)，所以 DRP-ES 是 normalized convex monetary risk measure，但不一般 coherent；d=1 時退化為 coherent 的單維 DRP-ES。Theorem 2 給出 DRP-ES 對 D 與 α 變化的上界，形式涉及 \(\overline{y}-\underline{y}\)、\(1-\alpha\)、\(\delta_1(D_2-D_1)\)；這是敏感性界，不是實證績效保證。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下只做**符號化示意**，不引入論文未給的數值；論文真實數字留在 §5。假設兩個資產、兩個候選模型，損失向量 \(\mathbf{X}=(X_1,X_2)\)。

1. **資產聚合**：取權重 \(\pi_1,\pi_2\) 與溫度 \(\theta\)，算 \(F_{\theta,\pi}(\mathbf{X})=\theta\log(\pi_1e^{X_1/\theta}+\pi_2e^{X_2/\theta})\)。當 \(\theta\) 很小，\(F\) 由 \(\max(X_1,X_2)\) 主導；當 \(\theta\) 很大，\(F\) 趨近 \(\pi_1X_1+\pi_2X_2\)。
2. **情境生成**：模型 A 與模型 B 各抽 N 個損失情境，分別聚合成 \(Y_A^{(n)}\)、\(Y_B^{(n)}\)。
3. **分歧度**：對兩組排序後情境算 Wasserstein 距離 \(\overline{W}_t\)，再除以所有模擬損失的標準差 \(s_t\)（加 \(\varepsilon\) 避免除零），得 \(D_t\)。
4. **模糊集**：由 \(D_t\) 得 KL 半徑 \(\rho=\delta_0+\delta_1D_t\)，限制 \(\mathbf{q}=(q_A,q_B)\) 不能離均分 \(\omega_0\) 太遠。\(D_t\) 小則 q 近均分；\(D_t\) 大則 q 可偏向尾部較悲觀的模型。
5. **輸出**：解 \(\sup_{\mathbf{q}}\operatorname{ES}_\alpha(F_{\theta,\pi}(\mathbf{X})\mid P_t^{\mathbf{q}})\)。θ 管「資產層面大損失權重」，D_t 與 ρ 管「模型層面偏悲觀程度」；兩者分開，不會把聚合函數的選擇與模型權重的選擇混成一個黑箱。

## §3 · 數據層

- **市場／頻率**：美國、中國 A 股、韓國股市；以交易日為單位（日頻），用日損失與日報酬計算。
- **股票池**：每個市場依 CITIC Tier-1 分七個產業：Financials、Consumer Discretionary、Consumer Staples、Information Technology、Upstream Cyclicals、Midstream Cyclicals、Downstream Cyclicals；選大型市值股票以確保流動性，價格調整股票分割與現金股利。
- **時段**：兩個樣本期間為 January 1, 2019 to August 12, 2026，以及 January 1, 2012 to August 12, 2019；Table 2／3 的測試期分別為 April 1, 2020 to August 12, 2026、April 1, 2013 to August 12, 2019。
- **切分**：一年訓練集、下一季驗證集（以 Sharpe 調參）、再下一季測試集；完整滾動方式與重估頻率所讀文本未見，待核。
- **情境生成**：每個候選模型生成 N=10000 個損失情境；五個模型為 HS、GARCH、EVT、GARCH-EVT、FHS。
- **組合建構**：每產業選 M_stock 檔最小 DRP-ES 股票；全文設定持有 21 檔、每產業 3 檔，產業內等權，產業間按風險倒數配置；再平衡頻率未披露。
- **成本／容量**：作者稱納入各市場實際交易成本、按交易價值比例計算；具體費率、換手、市場衝擊與容量假設未披露。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 所讀文本未見 | 不適用（無神經權重；為風險度量＋凸優化） | 未披露 | 中高：需自建五類風險模型、情境模擬、KL/ES 凸優化與三市場回測 | 市場行情公開；具體股票池、清洗規則、成本明細、再平衡頻率 TBD |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄依模板命名，實際填入所讀原文的**基線**數值；本文未宣稱這些基線是 SOTA。Δ 固定定義為「本方法 - 前SOTA」。MDD 越低越好，Δ 負值代表本方法回撤較淺，不把負號解讀為能力下降。未列 IR，因所讀文本未見 IR 數值。Δ 只在本表內有效，不可跨市場或跨配置轉移。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 中國 A 股 April 1, 2020 至 August 12, 2026（基線 CSI 300） | Annualized Return | 3.8592 % | 14.1200 % | +10.2608pp |
| 中國 A 股 April 1, 2020 至 August 12, 2026（基線 CSI 300） | Annualized Volatility | 18.0675 % | 13.0139 % | -5.0536pp |
| 中國 A 股 April 1, 2020 至 August 12, 2026（基線 CSI 300） | Maximum Drawdown | 45.6025 % | 12.2735 % | -33.3290pp |
| 中國 A 股 April 1, 2020 至 August 12, 2026（基線 CSI 300） | Sharpe Ratio | 0.2065 | 0.9506 | +0.7441 |
| 中國 A 股 April 1, 2020 至 August 12, 2026（基線 CSI 300） | Win Rate | 50.58 % | 51.81 % | +1.23pp |
| 中國 A 股 April 1, 2020 至 August 12, 2026（基線 HS） | Sharpe Ratio | 0.7597 | 0.9506 | +0.1909 |
| 韓國 April 1, 2020 至 August 12, 2026（基線 KOSPI） | Annualized Return | 23.0743 % | 21.5700 % | -1.5043pp |
| 韓國 April 1, 2020 至 August 12, 2026（基線 KOSPI） | Maximum Drawdown | 38.6304 % | 13.3408 % | -25.2896pp |
| 韓國 April 1, 2020 至 August 12, 2026（基線 KOSPI） | Sharpe Ratio | 0.7554 | 1.2295 | +0.4741 |
| 美國 April 1, 2020 至 August 12, 2026（基線 S&P 500） | Annualized Return | 18.8245 % | 15.9037 % | -2.9208pp |
| 美國 April 1, 2020 至 August 12, 2026（基線 S&P 500） | Maximum Drawdown | 25.4257 % | 18.1458 % | -7.2799pp |
| 美國 April 1, 2020 至 August 12, 2026（基線 S&P 500） | Sharpe Ratio | 0.7958 | 0.8593 | +0.0635 |
| 美國 April 1, 2020 至 August 12, 2026（基線 GARCH-EVT） | Sharpe Ratio | 0.8795 | 0.8593 | -0.0202 |
| 中國 A 股 April 1, 2013 至 August 12, 2019（基線 HS） | Sharpe Ratio | 1.0362 | 0.8225 | -0.2137 |
| 中國 A 股 April 1, 2013 至 August 12, 2019（基線 CSI 300） | Maximum Drawdown | 46.6960 % | 38.7882 % | -7.9078pp |
| 美國 April 1, 2013 至 August 12, 2019（基線 FHS） | Sharpe Ratio | 0.7157 | 0.6588 | -0.0569 |
| 美國 April 1, 2013 至 August 12, 2019（基線 S&P 500） | Maximum Drawdown | 19.7795 % | 11.8271 % | -7.9524pp |
| 韓國 April 1, 2013 至 August 12, 2019（基線 KOSPI） | Sharpe Ratio | -0.3344 | 0.2303 | +0.5647 |
| 韓國 April 1, 2013 至 August 12, 2019（基線 KOSPI） | Maximum Drawdown | 26.4984 % | 13.7979 % | -12.7005pp |
| 韓國 April 1, 2020 至 August 12, 2026（消融：linear aggregation 基線） | Sharpe Ratio | 1.2318 | 1.2295 | -0.0023 |
| 韓國 April 1, 2020 至 August 12, 2026（消融：worst-case weights 基線） | Sharpe Ratio | 1.2477 | 1.2295 | -0.0182 |

**解讀**：Table 2 期間，DRP 對市場基準的 Sharpe 與 MDD 方向大致支持「風險分散 overlay」；但這不是無條件的 capability。Annualized Return 在韓國與美國低於基準（KOSPI 23.0743 % 對 DRP 21.5700 %；S&P 500 18.8245 % 對 DRP 15.9037 %），所以「較高 Sharpe、較低 MDD」不等於「較高報酬」。Table 3 期間，中國 HS 的 Sharpe 1.0362 高於 DRP 0.8225；美國 FHS 的 0.7157 與 GARCH 的 0.6865 高於 DRP 0.6588。這些是直接反證：「多模型分散」在部分 regime 會輸給單一模型，不能寫成所有模型／市場的必然因果。

Table 5 更關鍵：南韓 linear aggregation 的 Sharpe 1.2318 高於 exponential 的 1.2295，worst-case weights 的 1.2477 亦高於 optimized weights 的 1.2295，反證「優化權重必然更優」。成本面，作者稱已計實際交易成本，但費率、換手、市場衝擊未披露，不能把全部 Δ 歸因於風險模型；驗證集以 Sharpe 調 δ0、δ1 與模型參數，也有選擇偏差與多重檢驗風險。這些 Δ 應視為「同一回測配置下的樣本內外混合證據」，不是跨所有模型／市場的必然因果。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

所讀文本未見獨立的 Limitations 小節；以下僅列可定位的作者自述或明示約束。

| 位置 | 作者自述／可定位限制 | 對本頁判讀 |
|---|---|---|
| §6 Conclusion | Future work 明列使用 modern reinforcement learning 來 elicit risk aversion、刻畫不同 risk perceptions；目前框架未涵蓋 RL | 這是作者承認的延伸邊界，不是已驗證能力 |
| §1 Introduction | 指 GRM 一般框架「practical implementation in risk management remains challenging」，未直接指定如何組合模型與配權 | 本文是對該缺口的具體化，但未證明所有 GRM 實作都可行 |
| §2.1 | 不允許 short-selling，\(\pi_i\in(0,\infty)\) | 策略是 long-only；空頭／槓桿市場的外推未驗證 |
| §5 | 只考慮 \(\Psi_j>0\) 的產業組合 | 若風險估計為負或極端不穩定，配置規則需另處理 |
| §5 | 股票池選大型市值以確保流動性 | 小型股、低流動性市場的容量與交易成本未驗證 |

### 6.2 推斷的隱含假設（本頁推斷）

1. **候選模型集包含尾部相關機制**：DRP 只在給定的 M 個模型中分散；若五個模型共同漏掉同一尾部機制，D_t 再大也無法補回。這是模型集依賴，不是普遍穩健。
2. **D_t 是可信的 disagreement proxy**：D_t 由模擬情境的 pairwise Wasserstein 標準化而來；N=10000 的 Monte Carlo 噪聲、模型模擬誤差與 ε 的影響，所讀文本未給敏感度表。
3. **KL 半徑與驗證集 Sharpe 調參不會過擬合**：δ0、δ1 在驗證集以 Sharpe 最大化選定，測試期固定；若驗證季與測試季 regime 斷裂，最優半徑可能失效。此為本頁推斷，原文未給正式的樣本外參數穩定性檢驗。
4. **成本模型足夠**：作者只說交易成本按交易價值比例計入；未見換手率、買賣價差、市場衝擊、借券成本。若成本上升，日頻風險擇時的優勢可能被侵蝕。
5. **股票池無 survivorship／look-ahead**：大型股池的選定時點、退市處理、財務資料發布延遲未披露；若池子是事後選定，回測可能有 survivorship bias。輔助數據缺失本身不是 survivorship 的證據，但需列為待核。
6. **基準可投資性**：CSI 300、KOSPI、S&P 500 是否含股利、是否可直接交易、再平衡與費用處理未披露；跨基準的 Δ 不宜直接當成可實現 alpha。
7. **單一樣本排序不等於普遍因果**：Table 2／3 的策略排名會隨市場與期間變動；不能把某市場某期間的 rank 推成「DRP 必然優於所有單模型」。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Worst-case ES／robust CVaR（Zhu & Fukushima 2009；Fadina et al. 2024） | 只看最悲觀模型 vs KL 模糊集＋D_t 調半徑 | 未見 | 概念基線；本文稱 D→∞ 時可收斂到 worst-case |
| Thick modelling／equal weights（Pesaran et al. 2009） | 等權子集 vs 以 disagreement 優化 q | 未見 | 文獻基線；本文 Table 5 有 equal weights 消融 |
| Linear weighted aggregation（De Vecchi et al. 2026） | 線性加權 vs exponential；θ→∞ 退化為 linear | 未見 | 文獻基線；Table 5 有對照 |
| Maximum aggregation（Lux & Papapantoleon 2019） | 只看最大損失 vs exponential；θ↓0 退化為 max | 未見 | 文獻基線；Table 5 有對照 |
| Single-scenario ES（HS／GARCH／EVT／GARCH-EVT／FHS） | 單一機率模型 vs 多模型混合 | 部分模型有公開定義／套件；本文代碼未見 | 實證基線；Table 2／3 逐項列出 |

🎤 **Interview Tip**

- **正確答**：DRP-ES 不是新的報酬預測模型，而是把 model risk 放進風險度量的 robust aggregation；θ 控制資產層面的「大損失權重」，D_t 與 KL 半徑控制模型層面的「偏悲觀程度」。它的賣點是 rank stability，不是保證每個市場都第一；Table 5 南韓 worst-case 權重奪冠就是反例。
- **錯答**：說「它用五個模型投票，所以一定打敗單模型」或「Sharpe 高就是 alpha」。前者忽略模型集依賴與消融反例；後者把風險 overlay 與報酬預測混為一談。

### 7.1 可證偽預測帶日期

以下為本頁研究假說，非作者已驗證結論；日期為本頁預測日期。

1. **2027 年底前**：若在未參與的三市場外樣本重跑，DRP-ES 的 Sharpe rank 分散度應低於單模型策略；若沒有，則「rank stability」主張受挑戰。
2. **2027 年底前**：若把交易成本提高或加入市場衝擊，DRP 相對 HS／FHS 的 Sharpe 優勢應縮小；若成本明細未披露，應先要求作者提供換手與費率表。
3. **2027 年底前**：若 D_t 長時間接近 0，優化 q 應接近均分；若偏離，KL 半徑或數值求解可能有誤。
4. **2027 年底前**：若移除 GARCH-EVT 或加入一個主導尾部的新模型，DRP 的結果應對模型集敏感；若完全不敏感，反而說明 D_t 沒有捕捉到模型分歧。

## §8 · For the Reader

- **因子研究員**：把 DRP-ES 當 risk overlay，不是 alpha 因子；先重現 HS／FHS 基線，再做 θ、δ0、δ1 的敏感度與換手分析。若只報 Sharpe 不報 turnover／成本，結論不完整。
- **高頻執行**：本文是日頻、下一交易日收盤建立部位；不要直接外推到盤中。成本按交易價值比例計入，但沒有市場衝擊模型；高頻執行者應先補 slippage 與容量測試。
- **組合配置**：產業內等權、產業間風險倒數配置，long-only 且要求 \(\Psi_j>0\)。南韓 worst-case weights 的 Sharpe 高於 optimized weights，說明「保守權重」在部分市場可能更優；配置前應做 regime 分解。
- **LLM-agent／可解釋 AI**：q 權重、D_t、θ、產業權重都是可審計的中介量，適合做「風險感知」的解釋層；但本文沒有 agent 實驗，任何 LLM 整合都是本頁推斷。
- **RL 策略**：作者在 §6 明列 future work 用 RL 來 elicit risk aversion；目前 DRP-ES 不是 RL，也沒有 reward shaping 實驗。若要做 RL，應把 DRP-ES 當 risk penalty 或 reward 的風險項，而非已驗證的 RL 方法。
- **研究學生**：先讀 Theorem 1／2 與 Proposition 2 的假設，再復現五個候選模型與凸優化；最容易出錯的是情境對齊、ES 對偶約束、以及驗證集 Sharpe 調參造成的選擇偏差。

## References

- 原論文：Yang Liu, Yuhao Liu, Jiani Shen, “Diversified-Risk-Perception Risk Measures and Portfolio Selection,” arXiv:2611.11803v1 [q-fin.RM], 09 Nov 2026. [arXiv](https://arxiv.org/abs/2611.11803)
- Fadina, Liu, Wang (2024), “A framework for measures of risk under uncertainty,” *Finance and Stochastics* 28(2), 363–390.
- Artzner et al. (1999), “Coherent measures of risk,” *Mathematical Finance* 9(3), 203–228.
- Föllmer & Schied (2016), *Stochastic Finance: An Introduction in Discrete Time*.
- Zhu & Fukushima (2009), “Worst-case conditional value-at-risk with application to robust portfolio management,” *Operations Research* 57(5), 1155–1168.
- Pesaran, Schleicher, Zaffaroni (2009), “Model averaging in risk management with an application to futures markets,” *Journal of Empirical Finance* 16(2), 280–305.
- De Vecchi, Nendel, Streicher (2026), “Upper comonotonicity and risk aggregation under dependence uncertainty,” *Mathematical Finance* 36(1), 118–139.
- Lux & Papapantoleon (2019), “Model-free bounds on value-at-risk using extreme value information and statistical distances,” *Insurance: Mathematics and Economics* 86, 73–83.
- McNeil (1997), “Estimating the tails of loss severity distributions using extreme value theory,” *ASTIN Bulletin* 27(1), 117–137.
- McNeil & Frey (2000), “Estimation of tail-related risk measures for heteroscedastic financial time series: an extreme value approach,” *Journal of Empirical Finance* 7(3–4), 271–300.
- Bollerslev (1986), “Generalized autoregressive conditional heteroskedasticity,” *Journal of Econometrics* 31(3), 307–327.
- Barone-Adesi, Giannopoulos, Vosper (1999), “VaR without correlations for portfolios of derivative securities,” *Journal of Futures Markets* 19(5), 583–602.