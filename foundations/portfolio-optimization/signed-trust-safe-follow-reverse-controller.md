<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=强化学习 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Signed Trust (Safe/Follow/Reverse) Controller 解構（Signed Trust (Safe/Follow/Reverse) Controller）

> **來源發布**：03 Nov 2026 ·（無 venue）· arXiv [2611.04826](https://arxiv.org/abs/2611.04826)
> **arXiv 原文**：[Safe Portfolio Learning from Untrusted Predictions: Signed Trust, Abstention, and Cross-Universe Polarity Failure](https://arxiv.org/abs/2611.04826v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：在五軸上把「不可信預測」放進 Safe/Follow/Reverse 三動作線上決策，並把預測生成與資本授權分離；解的是 online portfolio selection 與 LLM-advised allocation 長期把訊號方向當給定、缺少 capital-authorization layer 的 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `强化学习` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ①把訊號方向本身變成可學習動作：Safe、Follow、Reverse，並用 evidence gate 與 delayed collateral budget 決定是否、以及能資助多少方向性曝險。②核心 trick 是「單向安全干預」：gate 只把不支持的方向概率轉移到 Safe，collateral 先預留未結算最壞損失，split account 再用保護核心封頂可觸碰資本。③對五軸★的意義在 alpha=組合執行優化與 autonomy=人機協同可解釋：不是造新預測，而是給預測一層可審計的資本授權。④關鍵實證是 pre-specified Reverse 在 82-stock holdout 未跑贏 Equal Weight，post-holdout controller 的 protected wealth 全期最大差距為 15.9 basis points；作者明言貢獻是 safe adaptive use，不是 demonstrated alpha。

**X-Ray.** 放回五軸 Pareto，這篇不是把 alpha 前沿往前推，而是把 execution/authorization 前沿往安全端推：資料仍是低維量價＋會計表格，horizon 是日頻波段，學習實質是 online expert tracking（Fixed-Share＋Hedge），不是 deep RL policy gradient；alpha 機制落在組合執行與風險預算，人機協作則體現在可解釋的 Safe anchor、abstention 與逐路徑抵押憑證。它解掉舊工程坑：Fixed-Share 讓被貶抑方向仍可回復、gate 把「不預測」變成仍全額投資的 Safe 動作、collateral 把延遲回饋下的未結算損失先融資。打不開的 envelope 也很清楚：沒有 demonstrated alpha、holdout 是同日曆的 cross-universe 而非 temporal regime shift、成本模型是比例成本，capacity／impact 未建模；因此對量化讀者的價值是「安全 overlay／abstention baseline／資本授權審計」，不是可直接上線的 alpha 訊號。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作預設 | 本文做法 | 讀者含義 |
|---|---|---|---|
| 方向不再給定 | Follow-the-Winner／Follow-the-Loser 與 universal portfolio 類方法固定訊號到倉位的方向映射 | 動作集擴成 Safe、Follow、Reverse；Reverse 是訊號的否定，不是事後翻符號 | 把「極性不穩」變成可學習、可棄權的決策 |
| 切換強度可學習 | 單一 Fixed-Share share rate 會把「最佳動作多常變」寫死 | 多個 Fixed-Share base 跑 share-rate grid，外層 Hedge 聚合 | 不需要先相信極性持續多久 |
| 預測與資本授權分離 | RL finance 多半把 reward 當可信、風險只進 magnitude 約束；LLM advice 缺資本授權層 | master 只給學習機率；gate 單向棄權到 Safe；collateral budget 預留未結算最壞損失；split account 封頂 adaptive capital | 預測品質只從候選動作進入，安全層不假設預測正確 |

### 1.2 ⚡ Eureka

一句話：**把「訊號方向」當成可學習量，並把「預測」與「允許風險」拆開；gate 只單向把不支持的方向概率轉移到 Safe，collateral 先為未結算 cohort 預留最壞損失。** 直覺是：與其猜 Follow 還是 Reverse，不如讓 matured Safe-relative reward 決定兩者概率；當證據不足時，最安全的動作不是現金，而是仍然全額投資、但不押方向的 Safe。

### 1.3 信息流 ASCII 圖

預測 → 中心化方向 d_t → 三候選：Safe／Follow／Reverse  
　→ share-adaptive master 給 q_raw  
　→ evidence gate：χ 只減不增，將不支持方向 mass 轉 Safe → q_gate  
　→ collateral budget：以可用 reserve 共同縮放 active mass → q_exec  
　→ split account：ρ 保護核心跟 Safe；satellite 執行 Σ q_exec,a w^a  
　→ H=5 staggered cohorts 到期後：先 settle collateral，再 update master，再 update gate  
　→ 同一日期 maturity 不得影響同日較早 opening 的 cohort。

## §2 · 數學層

📌 **Napkin Formula**

\[
\kappa(s)=\max\{1,\,s-H+1\},\quad q_s^{\mathrm{raw}}=\bar q[\kappa(s)]
\]
\[
q_s^{\mathrm{exec}}=\operatorname{CollateralAllocate}\bigl(\operatorname{EvidenceGate}(q_s^{\mathrm{raw}})\bigr),\quad
w_s^{\mathrm{exec}}=\sum_a q_{s,a}^{\mathrm{exec}}w_s^a
\]
\[
x_{s,a}=\frac{\ell_{s,a}}{B},\quad \ell_{s,a}=\log\frac{F_{s,a}}{F_{s,\mathrm{Safe}}},\quad
(q_s^{\mathrm{exec}})^\top x_s\ge -a(q_s^{\mathrm{exec}})
\]

直覺：master 只回答「哪個方向目前較好」，gate 回答「證據是否足以授權」，collateral 回答「未結算損失容量能資助多少」。loss／訓練細節不是監督式 loss，而是 Fixed-Share 的 exponential reward update＋share transition，外層 Hedge 由每個 base 的 pre-update reward 更新；gate 用 last 50 matured rewards、至少 25、one-sided normal test at 5%，名義水準不修正 overlapping cohort 依賴，安全憑證不依賴 gate。所讀文本未給統一 Big-O；只給路徑複雜度價格 \(O(K\log(AT/K))\) 與 grid overhead，未見彙總複雜度。

## §2.5 · 帶數字走一遍（Worked Example）

以下配置數值取自 Table 2；機率與財富路徑是**假設／示意**，不是論文實證結果。論文真實數字留在 §5。

1. 假設某 decision date 的 raw master 給出 \(q^{\mathrm{raw}}\)；方向 \(d_t\) 由預測中心化，\(\mathbf{1}^\top d_t=0\)。三候選以 Safe 為 anchor，Follow／Reverse 對稱加減 \(\delta=0.10\) 的 overlay，且 \(w_{\max}=0.20\)。
2. Gate 只看已 matured 的 Safe-relative rewards：last 50、至少 25、one-sided normal lower bound at 5%。若 Follow 或 Reverse 的 lower bound 未過，\(\chi=0\)，該方向 mass 轉到 Safe；gate 不會創造 active mass。
3. Collateral budget 以 \(a(q)=q_{\mathrm{Follow}}+q_{\mathrm{Reverse}}\) 計 active mass，檢查 \(q^\top x_s\ge -a(q)\)；可用預算 \(b^{\mathrm{avail}}=R-P\)，以共同因子 \(\zeta\) 縮放 Follow／Reverse，承諾 \(c_s=a_s^{\mathrm{exec}}\)。初始 reserve \(R_0=1\)、排程 replenishment \(J_T=1\)、cap 為 2 normalized units；在 \(B=0.05\) 下對應 0.05、0.05、0.10 in log wealth。
4. Split account 以 \(\rho=0.80\) 把 core 綁在 Safe，其餘進 satellite 執行 \(w_s^{\mathrm{exec}}\)；protected wealth \(W_t^{\mathrm{prot}}=\rho S_t+V_t\)。這一步回答「adaptive 層最多能碰多少資本」，不是 master 的 Safe prior。
5. 到期時以 \(H=5\) 計算 cohort 的 \(F_{s,a}\) 與 \(\ell_{s,a}\)，驗證 \(|\ell_{s,a}|\le B\)，設 \(x_{s,a}=\ell_{s,a}/B\)；先 settle collateral，再 update master，再 update gate。

## §3 · 數據層

| 層 | 內容 | 來源定位 |
|---|---|---|
| Development universe | 24 stocks，主要 energy、materials、mining、utilities；daily market／accounting data from February 2014 through September 2025；out-of-sample development 為 February 7, 2017 through September 18, 2025，2,166 sequential decisions | §4.1 |
| Frozen holdout | 82 cross-sector stocks，disjoint from development；daily panel from February 20, 2014 through September 19, 2025；2,166 daily observations beginning February 7, 2017；H=5 下 2,162 completed cohorts，decisions through September 12, 2025 | §4.1 |
| Features | 同一 ordered dictionary 共 40 features：17 price-derived、23 technical／accounting；accounting values lagged；跨 universe dictionary identity 由 audit 強制 | §4.1 |
| Predictor | expanding walk-forward Ridge on 40 features＋asset fixed effects；每 126 trading days retrain，min history 504 days，756-day recency half-life；split-conformal directional uncertainty at 75%；validation mean IC 0.148 | §4.2 |
| Cost／frequency | daily decisions；H=5 overlapping cohorts；5 bp per unit drift-relative turnover | §4.2、§4.4 |

樣本外與容量假設：holdout 是 disjoint assets over substantially overlapping calendar history，測 cross-universe transfer，不測未來 regime；成本只含 proportional 5 bp，nonlinear impact、bid–ask variation、order-book depth、liquidity shocks 明列 omitted；capacity 未建模。數據來自 commercial vendors under license，不能重發；feature definitions 可重建 panel。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| Controller code available from corresponding author on reasonable request；所讀文本未見公開 URL | walk-forward caches／target-validity masks 有存檔描述，未見公開下載；Checkpoint 未披露 | 未披露 | 高：需商業日頻＋會計數據、重建 40-feature panel、重跑 walk-forward 與凍結協議 | 結果表以 ancillary files 附 arXiv，每表有 SHA-256 hash；底層 market／accounting data under license 不能 redistribute |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄只放**比較基線本身**的逐字數值；本文沒有給出可稱為 SOTA 的單一前作，故不合成 SOTA 區間。Development 的基線是 Daily Equal Weight，「本方法」欄在該列是 Safe OPMD／Certified execution（兩者重合），不是完整 signed controller；82-stock frozen 的「本方法」是 pre-specified Legacy Full-Ridge Reverse，secondary 是 Clean Full-Ridge Reverse；完整 signed controller 的 CAGR／Sharpe 所讀文本未披露。Δ 固定為「本方法 − 前SOTA」；MDD 為負值，Δ 為正表示回撤較淺。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 24-stock development | CAGR | 16.86% | 16.71% | -0.15pp |
| 24-stock development | Sharpe | 0.775 | 0.779 | +0.004 |
| 24-stock development | Avg. turn. | 1.383% | 0.104% | -1.279pp |
| 82-stock frozen | CAGR | 14.38% | 13.88% | -0.50pp |
| 82-stock frozen | Sharpe | 0.813 | 0.803 | -0.010 |
| 82-stock frozen | Max. DD | -38.58% | -38.03% | +0.55pp |
| 82-stock frozen | Avg. turn. | 1.070% | 0.965% | -0.105pp |
| 82-stock frozen | Rel. EW（ann. log growth） | 0.00% | -0.43% | -0.43pp |
| 82-stock frozen secondary | CAGR | 14.38% | 13.74% | -0.64pp |
| 82-stock post-holdout safety | min \(W_t^{\mathrm{prot}}/S_t\) vs Remark 1 floor (ratio) | 0.80 | 0.998406 | +0.198406 |

前SOTA 欄在此僅代表基線：development 是 Daily Equal Weight；82-stock frozen 的「本方法」是 pre-specified Legacy Full-Ridge Reverse 規格，不是完整 signed controller；secondary 是 Clean Full-Ridge Reverse。完整 signed controller 的 CAGR／Sharpe 所讀文本未披露，不能把 exploratory variants 或 safety ratio 當成 alpha 結果。Development 的 Certified execution 因 directional lower bound 從未轉正而與 Safe OPMD 重合；這是 non-result，不是獨立確認。Frozen primary 的 pre-registered one-sided test 為 p=0.83，95% interval for annual log-growth gap from -1.33% to +0.44%，confirmatory hypothesis 未支持；低換手貼近基準可轉移，directional edge 不轉移。Post-holdout 中 Consensus Abstention 的 CAGR 14.17%、Rel. EW -0.18%、turnover 0.793%，仍低於 Equal Weight；full controller 的 terminal gain -15.9 basis points、worst shortfall 15.9 basis points、mean active prob 0.108、abstention 62.0%。Intervention penalty \(\mathcal{I}_T=1{,}131\) normalized units over 2,162 cohorts、mean 0.52，這是 gate 與 collateral budget 合計移除的 active probability（不是只歸給 gate）；delay penalty \(\Delta_T^{\mathrm{exact}}=2.74\)、mean 0.0013；numerics 到 machine precision，split-account identity 到 \(10^{-16}\)。成本敏感度：Consensus Abstention 約在 31 basis points 與 Equal Weight 持平，但 baseline 成本下仍約落後 21 basis points per year；這是低換手 implementation resilience，不是 forecasting alpha。Protected wealth 的 realized minimum 0.998406，對照 certified worst case 2,000 basis points；Safe 自身在 audited path 曾到 roughly 0.942，故保護是相對 Safe 的增量損害，不是絕對保本。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§6.1 evidential scope**：只有兩個 universes；24-stock 建構架構、不能再驗證；82-stock 給出 frozen verdict 後，所有 signed variants 都是 exploratory。Holdout 是 cross-sectional、disjoint assets under substantially overlapping calendar history，不是 temporal holdout，不能說未來 regime。
- **§6.1 signal side**：預測證據弱，適合研究「不可信預測的安全使用」，不適合量測「真正 informative predictor 的價值保存」；唯一 engagement under informative signal 是 synthetic ladder，且 signal strength 是 generator input；training randomness 未表徵，需 multi-seed 才能對 neural predictor 下結論。
- **§6.1 model scope**：proportional cost model 使 31-basis-point crossover 只是 assumed linear frictions 的 sensitivity，不是 impact／spread／depth／capacity 的 execution simulation；Table 2 的架構常數界定 certified domain，換 mandate／horizon 會改經濟行為與憑證緊度。
- **§6.1 protection boundary**：保護是 relative to endogenous fallback；Safe policy 不好不會因保護 satellite 而被救；選 fallback 仍是學習保證之外的投資決策。
- **§6.1 下一步**：把完整 controller、candidates、master、gate、collateral、protected fraction、reporting protocol 凍結，在第三個 untouched universe（最好不同 calendar regime、richer cost／liquidity、repeated predictor training、alternative fallback anchors）做 confirmatory evaluation；sharper data-dependent bounds 是理論方向。

### 6.2 推斷的隱含假設（本頁推斷，尚待驗證）

- **Regime 依賴**：master 可在 Follow／Reverse 間回復，但 holdout 同日曆、非 regime shift；不能把 82-stock 的失敗外推成所有市場的極性必然不穩。
- **容量／成本**：5 bp 比例成本與 drift-relative turnover 是簡化；若 impact 非線性、depth 有限，低換手優勢可能縮水；31-basis-point crossover 不是通用 breakeven。
- **數據泄漏**：two-clock 與 accounting lag 有 audit，但推斷仍依賴 vendor timestamp 與 universe 建構；所讀文本未見對 universe survivorship 的專門處理，圖表／附件待核，不能把缺失當成已證明偏誤。
- **Safe anchor 假設**：所有安全陳述相對於 Safe；若 Safe 本身是高回撤 fallback，保護只限制 incremental damage，不保證 nominal wealth。
- **單一路徑推論**：Table 8 的「no gate 的 shortfall 低於 full controller」是該 reference path 的機制診斷；不能寫成所有模型／市場的必然因果，需第三 universe 與 multi-seed 隔離。所讀文本的 supplement 亦顯示 legacy signal 上 gate／budget 排序反轉，支持「非普遍必然」的讀法。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Follow-the-Winner／Follow-the-Loser（Li and Hoi, 2014） | 固定方向映射 vs 學習 Follow／Reverse／Safe | 概念公開；代碼未核 | lineage／contrast |
| Universal portfolios（Cover, 1991；Blum and Kalai, 1999） | worst-case growth＋costs vs capital authorization＋abstention | 概念公開；代碼未核 | lineage |
| Fixed-Share／Hedge（Herbster and Warmuth, 1998；Freund and Schapire, 1997） | expert tracking vs 動作集含 negation 與 Safe，外加 gate／budget | 概念公開；代碼未核 | machinery |
| Algorithms with predictions（Lykouris and Vassilvitskii, 2018；Mitzenmacher and Vassilvitskii, 2020） | consistency／robustness vs pathwise collateral＋split account | 概念公開；代碼未核 | closest contract |
| Safe RL／constrained policy（Achiam et al., 2017） | expectation／distribution 約束 vs pathwise accounting invariant | 概念公開；代碼未核 | contrast |
| LLM-advised allocation（Zhang et al., 2024） | advice as prior vs capital-authorization layer | 未核 | gap |

🎤 **Interview Tip**  
- **正確答**：這篇的貢獻是 safe adaptive use of untrusted predictions，不是 demonstrated alpha；pre-specified Reverse 在 82-stock holdout 未跑贏 Equal Weight，永久翻成 Follow 也沒有恢復 return edge；完整 signed controller 是 post-holdout exploratory；15.9 basis points 是 protected wealth 相對 induced Safe 的最大差距，不是絕對回撤保證；方法實質是 online expert tracking，不是 deep RL policy gradient。  
- **錯答**：把它說成新 alpha 模型、把 24-stock development 當獨立確認、把 Reverse 失敗說成「改 Follow 就修好」、把 15.9 basis points 當成所有市場的下限、或把 gate／budget 當成可互相替代的同一層；也別把 \(\mathcal{I}_T\) 只歸給 gate，它是 gate 與 collateral budget 合計移除的 active probability。

### 7.1 可證偽預測（本頁假說，尚待驗證）

在 03 Nov 2026 之後的首次第三個 untouched universe 凍結評估（確切日期 TBD；所讀文本未給）：以 pre-registered one-sided test 對 Equal Weight 的 annualized net log growth 為主要終點。若完整 controller 在 richer cost／liquidity 與 repeated predictor training 下仍顯著為正，則本頁「safety-not-alpha」讀法被推翻；若仍不顯著，則支持。另一個可證偽點：若在第三 universe 中「移除 gate 的 shortfall 低於 full controller」的排序反轉，則「gate 主要買 abstention、不是買較小 shortfall」的機制讀法需修正。

## §8 · For the Reader

- **因子研究員**：把這篇當「方向極性不穩」的診斷框架，不是因子 alpha 論文；可借用 Safe-relative reward、Follow／Reverse 對稱候選與 gate 的 abstention 設計，但不要把 development 的 24-stock non-result 當確認。
- **高頻執行**：collateral budget 與 drift-relative turnover 是可審計的 execution accounting；但 5 bp 比例成本、capacity 缺失，不能直接搬到 impact／depth 敏感的執行問題。
- **組合配置**：split account 的 \(\rho=0.80\) 保護核心是可解釋的 capital-authorization 結構；保護只相對 Safe，選 fallback 仍是投資決策。
- **LLM-agent**：這篇最適合當「advice → permission」層的參考：LLM 給方向，controller 決定是否、以及多少資本可資助；不要讓 agent 的預測直接變成曝險。
- **RL 策略**：五軸的 RL 應讀作 online decision learning；沒有 actor-critic／policy gradient 的訓練細節，reward 是 matured Safe-relative bounded reward，安全是 pathwise accounting 而非 expected constraint。
- **研究學生**：重點學「negative result 如何保留」：pre-registered holdout 失敗後，post-hoc 翻符號只能標 exploratory；dual certificate 的價值在分解與審計，不在保證 alpha。

## References

- Amirhossein MohammadpourAzari, Mohammad Zarei, MohammadAmin Fazli. _Safe Portfolio Learning from Untrusted Predictions: Signed Trust, Abstention, and Cross-Universe Polarity Failure_. arXiv:2611.04826v1, 03 Nov 2026. https://arxiv.org/abs/2611.04826v1
- Lineage：Cover (1991) Universal Portfolios；Blum and Kalai (1999) universal portfolios with transaction costs；Herbster and Warmuth (1998) Fixed-Share；Freund and Schapire (1997) Hedge；Lykouris and Vassilvitskii (2018) competitive caching with ML advice；Mitzenmacher and Vassilvitskii (2020) algorithms with predictions；Bailey et al. (2016) backtest overfitting；Pippas et al. (2025) RL finance evaluation survey。
- 來源鏈接：https://arxiv.org/abs/2611.04826