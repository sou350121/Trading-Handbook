<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=监督回归 alpha=组合执行优化 autonomy=人机协同可解释 -->

# SPIDE 解構（SPIDE）

> **來源發布**：2026-10-21 · （無 venue） · arXiv [2610.30539](https://arxiv.org/abs/2610.30539)
> **arXiv 原文**：[Dynamics of the Liquidity Profile in Concentrated Liquidity Provision: An SPIDE Approach](https://arxiv.org/abs/2610.30539v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 Uniswap V3 的聚合流動性剖面視為 tick space 上的 infinite-dimensional field，用退化 Sturm–Liouville 算子加 Hilbert–Schmidt 非局部積分算子寫成 SPIDE；再用 Gegenbauer 特徵展開降成有限維線性 SDE，讓 level／slope／curvature 模態可校準、可預測，並接到 pool reserves、pool value 與 LVR。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `监督回归` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①它把 AMM 流動性剖面建模成 SPIDE，而不是逐 tick 的獨立 SDE 或只做 PCA 因子。②核心 trick 是用 Gegenbauer 特徵函數把退化 Sturm–Liouville 與非局部 relocation 算子一起投影成有限維線性 OU／VAR，低階模態直接對應 level、slope、curvature。③對「微觀盤口 × 高頻日內 × 監督回歸 × 組合執行優化 × 人機協同可解釋」這條軸，它提供的是可解釋的 operator 先驗，不是黑箱 alpha。④關鍵實證：ARB5 Full 的 active-band ratio \(R_A\) 為 Sparse SPIDE 0.926，martingale null 1.000（定義值）；但原文未給前 SOTA，且 value-weighted 與 ETH30 的結果不普遍佔優。

**X-Ray.** SPIDE 的 Pareto 位置很特別：它不像 LOB SPDE 那樣建模集中式訂單簿的 mid-price 附近深度，也不像 functional PCA 只做曲線降維；它把「LP 重新配置區間」造成的 cross-tick redistribution 寫成非局部積分算子，這是 AMM 場域才有的結構。對量化讀者，價值在於把流動性剖面變成低維狀態空間：level mode 的均值回歸不能由 \(\mathcal{L}\) 提供，因 \(\lambda_0=0\)，只能來自 \(V\)；這使「剖面水平持久性」成為可檢驗的 operator 證據。它打不開的 envelope 也清楚：校準在 rank grid 而非 log-moneyness，stationary additive-OU 分支對 ETH30 出現非平穩壓力，且 noise operator 未校準、成本／容量未進模型。因此它是「結構化診斷與模擬器」，不是可直接部署的執行策略。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作／舊典範 | SPIDE 的改法 | 讀者要盯的風險 |
|---|---|---|---|
| 建模對象 | LOB SPDE 描述 centralized venue 的 supply–demand schedule；functional PCA 描述流動性曲線 | 直接建模 AMM pool 的 aggregate liquidity profile，tick space 上的 spot-centered log-liquidity \(u_t(x)\) | rank grid 不是價格距離；跨池、跨 fee tier 的比較需另做坐標映射 |
| 算子結構 | 局部擴散或多重噪聲；PCA 只給靜態因子 | 退化 Sturm–Liouville \(\mathcal{L}\) 編碼向市價集中與均值回歸；Hilbert–Schmidt \(V\) 編碼非局部 repositioning | \(V\) 在有限秩被截斷；\(\alpha\) 與 \(V\) 會競爭同一批低階模態 |
| 降維與校準 | Legendre／PCA 低秩近似，或 SPDE 的 spectral estimator | Gegenbauer 特徵基底把 SPIDE 降成耦合線性 SDE；在固定 Legendre 參考框架中校準，避免 \(\alpha\) 同時移動殘差度量 | 固定框架是為了識別，不是證明 \(\alpha=1/2\)；objective 對 \(\alpha\) 可能很平 |

### 1.2 ⚡ Eureka

**一句話：** 把「流動性剖面如何被 LP 重新配置」寫成一個退化 Sturm–Liouville 加非局部積分算子的 SPIDE，再用 Gegenbauer 特徵函數把它投影成有限維線性 SDE——於是無限維的剖面動力學，變成 level／slope／curvature 模態的 VAR 式轉移。

**直覺：** \(\mathcal{L}\) 負責「向市價集中」的局部回復，\(V\) 負責「某一 tick 的流動性被搬到遠處 tick」的非局部再分配。Gegenbauer 基底同時對角化 \(\mathcal{L}\)，所以 \(\lambda_n=a n(n+2\alpha)\) 是現成的二次特徵值階梯；剩下的 \(V\) 在有限秩裡只是一個低階矩陣，兩者相加就是可校準的 drift。

### 1.3 信息流

```text
tick-level liquidity snapshots
        │
        ▼
spot-centered log-liquidity u_t(x) on rank grid
        │  Gegenbauer / Legendre projection
        ▼
modal coefficients c_t = (c_t^0, c_t^1, c_t^2, ...)
        │  finite-rank closure: A_K = Λ_K + V_K
        ▼
linear SDE / VAR:  c_{j+1} = Φ_K c_j + ε_{j+1},  Φ_K = exp(-A_K Δ)
        │
        ├──► one-step forecast of u_t(x)
        ├──► active-band / value-weighted diagnostics
        └──► reserves, pool value, LVR under OU or BS noise
```

## §2 · 數學層

📌 **Napkin Formula**

\[
du_t(x)=\{\mathcal{L}u_t-V[u_t]+h(t,x)\}\,dt+\sum_m\{\sigma_m(t,x)+\tilde\sigma_m(t,x)[u_t]\}\,dW_t^m
\]

\[
\mathcal{L}=a(L^2-x^2)\partial_x^2-bx\partial_x,\qquad
\lambda_n=a n(n+2\alpha)
\]

\[
dc_t=(-\Lambda_K-V_K)c_t\,dt+\sum_m \sigma_m^{(K)}dW_t^m
\]

**直覺：** \(\mathcal{L}\) 是退化擴散，端點 \(x=\pm L\) 不是均勻橢圓；\(V\) 是積分算子，讓遠端 tick 互相影響。投影後，drift 是「對角特徵值階梯 + 低秩 relocation 矩陣」，所以單步轉移是指數矩陣 \(\exp(-A_K\Delta)\)，可直接做 Gaussian quasi-likelihood。

**Loss／訓練細節：** 原文用 profiled Gaussian quasi-likelihood，對 innovation covariance 做 profile，並加 \(\lambda_V\|V_\Delta\|_F^2\) 的 Frobenius penalty；\(\alpha\) 在 compact grid 上 profile，並加 stability margin 約束 \(\min_k \operatorname{Re}\operatorname{eig}_k(A_\Delta^{(r)})\ge \eta_{\mathrm{stab}}\)。每次目標評估涉及矩陣指數與行列式；原文未給計算複雜度。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為**假設／示意**的符號化玩具例，不引入原文沒有的數值；目的只示範「照著算一遍」的順序。

1. **假設**一個 toy profile，投影到低階 Gegenbauer 模態，得到 level、tilt、curvature 三個係數 \(c_0,c_1,c_2\)。這一步只做投影，不做交易。
2. 給定 toy 參數 \(a,\alpha\) 與 toy relocation 矩陣 \(V\)，組出 \(A=\operatorname{diag}(\lambda_n)+V\)。注意 \(\lambda_0=0\)：level 的衰減只能由 \(V_{00}\) 或 level 與其他模態的交叉項提供。
3. 給定單步 \(\Delta\)，算轉移 \(\Phi=\exp(-A\Delta)\)。若 \(A\) 的特徵值實部為正，\(\Phi\) 的譜半徑小於一，預測器才平穩。
4. 給定一個 toy innovation \(\varepsilon\)，算下一期模態 \(c'=\Phi c+\varepsilon\)。這一步就是 VAR 式單步預測。
5. 用 \(u'(x)=\sum_n c'_n e_n(x)\) 重構剖面；再代入 reserves／value 的積分式，檢查 active-band 預測誤差是否低於 martingale null。若 level 模態被當成隨機漫步，\(\Phi\) 的譜半徑會被推到一，這是 \(V=0\) 分支的已知退化。

## §3 · 數據層

- **市場／池子：** WETH–USDC 5 bps on Arbitrum（ARB5）、WETH–USDC 5 bps on Ethereum Mainnet（ETH5）、WETH–USDC 30 bps on Mainnet（ETH30）。原文明說 selection 是 illustrative，不是定義好 universe 的抽樣。
- **資料來源：** tick-level liquidity snapshots 透過 Teahouse Finance API；price histories 另取。未見公開原始數據附件。
- **網格：** 每張 snapshot 放在 \(N=201\) 的 uniform grid，\(L=1\)，坐標是 sampled liquidity ticks 的 normalized rank；rank zero 不必然正好是當前 tick，rank 差也不等於價格距離。
- **頻率／窗口：** snapshots 以固定 block stride 抽樣；Mainnet 與 Arbitrum 各自設定（原文寫 `2{,}400` blocks on Mainnet、`115{,}200` on Arbitrum，nominal 為 8 hours）。共同 `1200`-observation Full block 切成三個 contiguous `400`-observation windows W1–W3；報告 W1 與 Full。
- **樣本外協議：** 在 window 的前半段 fit，對後半段做 rolling one-step-ahead forecast；in-sample protocol 則在整個 window 重 fit 看 residual whiteness。
- **容量／成本假設：** 所讀文本未見 pool capacity、gas、slippage、交易成本或 breakeven 成本的校準；LVR 只取價格套利成分，排除 LP 添加／移除流動性的現金流。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD；所讀文本未見公開 code repository |
| Checkpoint | TBD；所讀文本未見 model checkpoint |
| License | TBD；所讀文本未見 license |
| 複現難度 | 高：需要實作 Gegenbauer／Legendre 基底、有限秩 Galerkin、矩陣指數、profiled likelihood、stability constraint 與 multistart 優化 |
| 數據可得性 | 需 Teahouse Finance API 的 tick-level snapshots；原始數據與附件未見公開連結 |
| 關鍵實作細節 | 固定 Legendre 參考框架、finite-volume quadrature masses、Frobenius penalty、\(\alpha\)-grid profile；細節在 Appendix B |

## §5 · 評測 / Benchmark

> 原文未給「前 SOTA」；下表「前SOTA」欄取原文最強的非 SPIDE 對照 **Factor VAR** 的逐字數值，僅作對照，**不代表 SOTA**。本方法欄取 **Dense-3 + penalty** 的逐字數值；Δ 僅在末欄計算，定義為「本方法 − 前SOTA」。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ（本方法−前SOTA） |
|---|---:|---:|---:|---:|
| ARB5 Full | \(R_A\)（active-band RMSE ratio） | 0.966 | 0.929 | -0.037 |
| ETH5 Full | \(R_A\)（active-band RMSE ratio） | 0.881 | 0.847 | -0.034 |
| ETH30 Full | \(R_A\)（active-band RMSE ratio） | 1.369 | 1.394 | +0.025 |
| ARB5 Full | \(R_V\)（value-weighted profile ratio） | 1.214 | 0.807 | -0.407 |
| ETH5 Full | \(R_V\)（value-weighted profile ratio） | 0.956 | 0.957 | +0.001 |
| ETH30 Full | \(R_V\)（value-weighted profile ratio） | 2.663 | 3.051 | +0.388 |
| ARB5 Full | \(S_H^{\mathrm{OOS}}\)（Hosking severity） | 2.879 | 2.018 | -0.861 |

**解讀：** Δ 為負只表示在該池、該窗口、該指標下相對 Factor VAR 較低；不是所有模型／市場的必然因果。原文自己的讀法是：ARB5 與 ETH5 的 active-band gains 較一致，consistent with out-of-sample mean reversion carried by low modes；但 value-weighted gains 取決於窗口與配置，且 ETH30 的 \(R_A\)、\(R_V\) 與 out-of-sample severity 都顯示 stationarity 受壓。這些 diagnostics 是「fitted structure 的診斷」，不是 model competition；也沒有把交易成本、capacity 或 price-distance 映射納入，因此不能把 \(R_A\) 或 \(R_V\) 的改善直接解讀成可實現 alpha。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位者）

- **Stationarity／centering 失效：** §5.2 明說若存在 structural drift，\(\bar u\) 不再是 conditional mean 的代理，會出現系統性樣本外偏差；ETH30 的 held-out diagnostics 與此機制一致。
- **\(\alpha\) 與 \(V\) 的識別競爭：** §5.3 與 §5.4 明說若 \(V_\Delta\) 在每個 retained mode 都自由，\(\alpha\) 的 differential damping 可被逐項抵銷，\(\alpha\) 不可分識別；finite-rank 的 Hilbert–Schmidt 限制是靠 penalty 與 support restriction 「imposed rather than observed」。原文也明說 objective profile 對 \(\alpha\) 近 flat，不能建立 unique concentration order，也不能證明 \(\alpha=1/2\)。
- **Noise operator 未校準：** §5.4 明說 scheme 不校準 noise operator；用平滑 rank-\(K\) truncation 重現 full-resolution quadratic covariation 會被高頻與 boundary structure 支配。
- **Rank grid 不是價格距離：** §5.1 與 §5.3 明說 fitted rates、\(\hat\alpha\)、relocation matrices、active band 都是在 rank coordinates 中估計，描述的是 rank grid 的剖面，不是 price distances；price-aware diagnostics 需要 ranks 到 tick distances 的 map。
- **Branch-specific 限制：** §4.2.1 Remark 4.1 明說 case study 的 \(V\) 是在 log-liquidity 的 additive OU 模型、normalized tick-rank grid 上估；用到 log-price coordinates 或 raw-liquidity model 需要 coordinate map 與 separate calibration，本文未提供。§4.2.2 的 BS/raw-liquidity branch 需要假設 \(U_t\ge 0\)，而這不是 §3 的結論。
- **Log 坐標的 back-transform bias：** §5.4 明說 \(\mathbb{E}[e^u]=e^{m+\sigma^2/2}>e^{\mathbb{E}[u]}\)，\(R_V\) 的 lognormal correction 只在 Gaussian approximation 下移除。
- **樣本選擇：** §5.1 明說三池選擇是 illustrative，不是 defined universe 的樣本。
- **獨立噪聲假設：** §4.2 Remark 4.1 明說 price 與 liquidity 的 Brownian motions 獨立是為了方便； correlated driving noises 可納入但表達更長。

### 6.2 推斷的隱含假設（本頁推斷，待核）

- **Regime 依賴：**  Stationary additive-OU 分支只在「剖面圍繞平穩均值波動」時合理；ETH30 的結果支持「非平穩／結構漂移」是主要失效模式，但不能推論所有 fee tier 或所有 AMM 都如此。此為本頁推斷，尚待更多池與窗口驗證。
- **容量與成本：** 模型与診斷未內生化 pool depth 對交易的影響、LP 進入／退出成本、gas 與 rebalancing 成本；因此不能把 \(R_A\) 的改善直接當成可交易策略的净收益。原文未給 capacity／cost 參數，所讀文本未見。
- **數據泄漏風險：** 預測協議是前半 fit、後半 holdout；但 evaluation 的 PCA space 是「fitted once per window」，是否只用訓練段、是否含 holdout 資訊，所讀文本未明言，圖表／附件待核。此為本頁推斷，不是已證實的泄漏。
- **Survivorship／selection：** 三池是 illustrative selection，不是 universe；因此不能把結果當成對所有 Uniswap V3 pools 的無偏估計。輔助數據缺失或未提取圖片不是 survivorship bias 的證據；此處只說本頁未核。
- **Rank-to-price 映射：** 若要把 fitted \(V\) 或 \(\alpha\) 用於價格坐標的執行模型，需另外校準 coordinate map；未做此映射前，不能把 rank-grid 的均值回歸速度直接套到 log-moneyness。此為本頁推斷。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| LOB SPDE（Hambly et al. 2020；Cont & Müller 2021） | centralized LOB 的 mid-price 附近深度 vs AMM 的 aggregate liquidity profile；局部擴散／多重噪聲 vs 非局部 relocation | 論文公開 | 概念對手；SPIDE 的 \(V\) 與 rank-grid 校準是差異點 |
| Functional PCA／dynamic factor（Risk et al. 2025） | 曲線降維與模態解釋 vs 生成元算子與可校準 SPIDE | arXiv 公開 | 互補：前者給模態語言，後者給 operator 結構 |
| Factor VAR | 通用低秩 VAR vs 有 \(\mathcal{L}+V\) 結構的 finite-rank OU | 本文 baseline | 在 ARB5／ETH5 的 active-band 上 SPIDE 較低，但不是所有指標與池 |
| Martingale null | 無均值回歸 vs 低階模態均值回歸 | 定義基準 | 所有 ratio 的分母；不是可訓練模型 |
| OU-type vs BS-type SPIDE | log-liquidity 的 additive OU vs raw-liquidity 的 multiplicative BS | 同框架分支 | OU 分支被校準；BS 分支需 nonnegativity 假設，未校準 |

🎤 **Interview Tip**

- **正確答：** 「SPIDE 的關鍵不是『用 SPDE 擬合流動性』，而是用退化 Sturm–Liouville 的 Gegenbauer 特徵函數，把非局部積分算子一起有限維化成線性 SDE。因為 \(\lambda_0=0\)，level mode 的均值回歸只能來自 \(V\)，不是 \(\mathcal{L}\)。校準必須在固定參考框架中做，否則 \(\alpha\) 會同時移動殘差度量；所以 \(\alpha\) 與 \(V\) 要聯合讀，不能單獨宣稱 \(\alpha=1/2\) 最優。」
- **錯答：** 「它證明 SPIDE 全面勝過 martingale 與 Factor VAR，且 \(\alpha=1/2\) 是資料選出的最佳值；rank-grid 的 \(V\) 可以直接套到價格坐標做執行。」錯在：原文只說 active-band gains 較一致、value-weighted gains 取決於窗口與配置；\(\alpha\) profile 近 flat，不能建立 unique order；rank 坐標與價格距離之間沒有給 mapping。

### 7.1 可證偽預測（帶日期）

在 **2026-10-21 之後的下一版（v2）**，若作者新增 price-aware 擴展並重跑 ETH30：若樣本外 active-band ratio 仍高於一，則支持「非平穩／rank-to-price 映射不足」是主要瓶頸；若降至一以下，則支持「stationary branch 的失效主要來自缺少 price-aware 項，而非框架本身」。此預測為本頁研究假說，尚待驗證。

## §8 · For the Reader

- **因子研究員：** 把低階 Gegenbauer 模態當成 level／slope／curvature 的狀態變量，先检验它們的均值回歸與 cross-mode 耦合；但不要用 rank-grid 的 \(\hat V\) 直接推價格距離的 alpha。可先做 fixed-frame 的 modal VAR，再逐步加 operator 限制。
- **高頻執行：** LVR 公式與 active-band 診斷可用來評估「剖面預測改善是否集中在市價附近」；但本文沒有 gas、slippage、庫存成本與 capacity，不能直接把 \(R_A\) 當成執行淨收益。ETH30 的結果提醒：非平穩池的剖面預測會系統性失準。
- **組合配置：** OU／BS 分支給出 pool value 與 LVR 的 mean／covariance 表達式，適合做情境模擬與風險分解；但 BS 分支需要 nonnegativity 假設，且 value 的 modal decomposition 只在 raw-liquidity 坐標線性。配置用途應把它當結構化模擬器，不是預測收益的因子。
- **LLM-agent／研究工程：** 這篇的工程價值在「operator family + fixed-frame calibration」：可把 \(\alpha\)、\(a\)、\(V\) 的支持集當成可配置的結構先驗，讓 agent 在有限秩假設下做模型選擇；但要明示 rank grid 與 price distance 的差異，避免把診斷結果誤讀成跨市場定律。
- **RL 策略：** finite-rank linear SDE 可當成流動性剖面的模擬環境，用於測試 rebalancing／LP 策略；但 stationarity、nonnegativity、noise operator 未校準這三點會限制 sim-to-real。ETH30 可作為 domain randomization 的反例池。
- **研究學生：** 先讀 §3 的 semigroup／mild solution 與 Appendix A 的 Picard iteration，再讀 Appendix B 的 Legendre reference algebra；復現時先用符號化 toy 例確認 \(\lambda_0=0\) 與 \(V_{00}\) 的角色，再進真實數據。

## References

- Lee, C.-Y., Risk, J., Tung, S.-N., & Wang, T.-H. (2026). *Dynamics of the Liquidity Profile in Concentrated Liquidity Provision: A Stochastic Partial Integro-Differential Equation Approach.* arXiv:2610.30539. https://arxiv.org/abs/2610.30539
- Risk, J., Tung, S.-N., & Wang, T.-H. (2025). *Dynamics of liquidity surfaces in Uniswap v3.* arXiv:2509.05013.
- Tung, S.-N., & Wang, T.-H. (2026). *A mathematical framework for modelling CLMM dynamics in continuous time.* Digital Finance.
- Milionis, J., Moallemi, C. C., Roughgarden, T., & Zhang, A. L. (2022). *Automated market making and Loss-Versus-Rebalancing.* arXiv:2208.06046.
- Hambly, B., Kalsi, J., & Newbury, J. (2020). *Limit order books, diffusion approximations and reflected SPDEs.* Applied Mathematical Finance.
- Cont, R., & Müller, M. S. (2021). *A stochastic partial differential equation model for limit order book dynamics.* SIAM Journal on Financial Mathematics.
- Da Prato, G., & Zabczyk, J. (2014). *Stochastic Equations in Infinite Dimensions.* Cambridge University Press.
- Abramowitz, M., & Stegun, I. A. (2012). *Handbook of Mathematical Functions.* Dover Publications.