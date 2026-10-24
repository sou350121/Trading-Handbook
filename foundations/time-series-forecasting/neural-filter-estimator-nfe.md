<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=全自动黑盒 -->

# Neural Filter Estimator (NFE) 解構（Neural Filter Estimator (NFE)）

> **來源發布**：2026-10-19 · ICAIF26 · arXiv [2610.25802](https://arxiv.org/abs/2610.25802)
> **arXiv 原文**：[Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage](https://arxiv.org/abs/2610.25802v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 日頻波段 × 監督回歸 × 風險擇時 × 全自動黑盒」。它處理的是統計套利中的 prior gap：殘差價差的隱狀態原本由手工插件點估計承擔，NFE 改為一次前向輸出**校準後驗**，並讓不確定性進入預先固定的下單規則。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ①在**模擬價差路徑**上以 neural posterior estimation 訓練 GRU＋混合密度頭，一次前向輸出橫截面每檔殘差半衰期的後驗；②核心 trick 是先用 classifier two-sample test 否證候選新息律，只有隨機波動率＋非對稱新息通過，再在通過檢驗的模擬器上監督訓練；③★軸在「風險擇時」——式 (7) 讓後驗不確定性進入倉位分母，但作者自陳後驗**不加 alpha**，且真實資料上「寬度隨窗口變化」與「懲罰依賴 d²」兩條通道尚未分離；④Sharpe 上 AFM 1.45、NFE 1.49，觀察到 +0.04，但本研究無法據此判定能力優勢；預測落差上 OU 點估計 1084 bp、amortized OU-form 588 bp、amortized direct 137 bp，後者屬 PCA 殘差預測診斷且非部署濾波器。

**X-Ray.** 這篇在五軸 Pareto 上的位置反直覺：名義上是監督回歸，量出來的卻是**估計器換不出識別力**。在真值已知、高斯固定尺度特例的 regime-AUC 實驗裡，所有實用估計器與強參考偵測器落在 0.61 – 0.66 AUC 的同一窄帶；看得到真實 level 與 scale 的偵測器到 0.93，但該比較另有「窗內狀態固定」條件。該實驗中未知 level 是最大的已識別損失來源：level 已知時分離度由 0.62 – 0.64 升到 0.79 – 0.81，而餵入未去均值價差只加 +0.0002±0.0005 AUC。這不等於「任何改進都不可能」：panel form 在真值已知抽樣上已改善半衰期後驗，NLL 由 2.088 降到 1.762，但真實資料上的價值尚未測試。工程價值因此集中在**校準**與**預測落差披露**，不是 alpha。真正可帶走的是否證紀律：label-free 診斷必須附零模型與正控制；作者就因此撤回 250 日窗口的增益。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 對照 | 前作做法 | NFE 的改動 | 論文自述代價／限制 |
|---|---|---|---|
| vs Avellaneda–Lee (PCA 殘差) | 對累積殘差擬合均值回歸，用 κ 的**點估計**進出場 | 保留殘差價差與均值回歸核，改輸出 τ 的**五成分混合後驗** | 識別力不變；換來不確定性與區間，不加 alpha |
| vs Epstein et al. (AFM) | 端到端學因子＋端到端交易網路 | **沿用 AFM 因子不自己學**，只接後驗濾波＋預先固定決策層 | 固定規則才能隔離後驗貢獻；AFM 交易網路在本論文 60d/20d 窗口下 SR 0.25，原文明言非復現 |
| vs TabPFN / TabICL / TabFM | 行視為可交換集合，一次前向做上下文學習 | 時序步改成 **GRU 遞歸**保留日序；橫截面步留在外部因子模型 | panel form 只在真值已知模擬上測，未進組合；真實資料價值未測試 |

### 1.2 ⚡ Eureka trick

**「先把模擬器否證掉，再相信從模擬器學來的後驗。」** 真實殘差沒有直接隱狀態標籤；模擬器提供已知 τ 標籤，但標籤能否用於真實資料，取決於生成模型是否近似資料。NFE 用 classifier two-sample test 檢查新息律：Gaussian 常數波動 0.903±.002（crypto） / 0.904±.001（KRX） 未通過，Student-t 0.809±.002（crypto） / 0.788±.002（KRX） 未通過，只有 SV＋lognormal 以 0.686±.003（crypto） / 0.678±.002（KRX） 在 AUC＋兩個標準誤低於 0.75 的判準下通過。檢驗約束新息律與尺度，**不驗證 kernel 或 level**；選定模型與資料仍可分辨（0.678），作者要求解讀 KRX 組合後驗聲明時保留此折扣。

### 1.3 信息流

- 面板（日 × 股）→ 橫截面步：因子模型 F_t = Ω_{t−1}R_t，ε_t = R_t − β_{t−1}⊤F_t；NFE 不學這一步，直接用 AFM 殘差，PCA 只作對照。
- 價差 S_{i,t} = Σ_{s≤t} ε_{i,s} → 窗口 w_{i,t}（L=60，窗內標準化）。
- 時序步：GRU_ψ（隱藏維度 64）→ h_t。
- 後驗頭：MDN_ψ → q_ψ(τ | w)，τ = ln2/κ，log₂τ 上的五成分高斯混合。
- 尾質量 ∫_{τ*}^∞ q_ψ dτ 對應 P(z=D | w)；**直接校準檢查的是 τ 的覆蓋率**，尾機率的邊際校準由作者據此推論；每股先驗重加權後未另核。
- 部署重加權：訓練用平坦先驗（τ∈[3,400] 天，log₂τ 均勻），上線換成每股 log₂τ 上的窄高斯先驗；只重加權後驗抽樣，網路不重訓。
- 決策層（預先固定）：ω ∝ −d·E[1−e^{−κH}] / (E[1−e^{−2κH}] + d²·Var(e^{−κH}))；去均值、歸一化到 unit gross 後形成組合 P&L。
- 訓練分支：模擬器抽樣 (w, θ, z) 真值已知 → NLL。Conformal 層：outcome-fitted 方法在訓練集前 80% 擬合、後 20% 校準並隔 L 天；模擬器方法用完整訓練切分；寬度按分位數乘法校準。

## §2 · 數學層

📌 **Napkin Formula**

生成核（OU，regime-indexed 速度）：

> S_{i,t+1} − S_{i,t} = κ(z_{i,t})(m_{i,t} − S_{i,t}) + σ_{i,t} ζ_{i,t}

後驗與決策：

> q_ψ(τ | w_{i,t}) ← GRU_ψ(w_{i,t}) → MDN_ψ，τ = ln2/κ
>
> ω_{i,t} ∝ −d·E[1 − e^{−κH}] / (E[1 − e^{−2κH}] + d²·Var(e^{−κH}))

**複雜度**：訓練一次性攤銷（原文：訓練 NFE 一次 104 s）；推論 2.5 ms per cross-section，對比未攤銷 particle filter（25×256 particles）的 7.9 s。

**直覺**：式 (7) 的分子、分母第一項與方差項都取決於後驗。固定 d 與另兩個矩時，增加 `d²·Var(e^{−κH})` 會降低未正規化權重；點估計使 Var 項恆等於零，所以點估計版本退化為古典插件規則。固定 gross 後，跨資產去均值與縮放會重新分配曝險，不能把單一資產的未正規化權重直接當成最終倉位。此懲罰只涵蓋收斂速度不確定性；level 的不確定性留在 d 裡不被罰，而論文找到的最大已識別損失來源正是 level。

**Loss / 訓練**：L(ψ) = −E_{(τ,w)~p_train}[log q_ψ(τ | w)]。訓練時半衰期在窗內固定、τ∈[3,400] 天，log₂τ 均勻，故 θ 退化成 τ、頭部輸出呈 likelihood 形狀，這是部署時可用先驗重加權的前提。GRU-64、五成分 MDN、Adam、4,000 steps、batch 512、lr 3×10⁻³ cosine、60,000 draws。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ 以下全部是**假設／示意**的玩具數字，用來手算式 (7) 的機制，**不是論文實證結果**。論文真實數字一律見 §5。

**步 1｜訊號**：假設某日某檔的標準化偏離 d = −2（價差低於窗均值 2 個窗內標準差，規則要做多；原文 clip 範圍 d ∈ [−3,3]，−2 只是範圍內一個值），持有期 H = 20 天。

**步 2｜後驗**：假設後驗把質量對半分給快態 τ = 10 天、慢態 τ = 120 天。κ = ln2/τ：

- 快態 κ ≈ 0.0693 → e^{−κH} ≈ 0.25，1−e^{−κH} ≈ 0.75，1−e^{−2κH} ≈ 0.9375。
- 慢態 κ ≈ 0.00578 → e^{−κH} ≈ 0.8909，1−e^{−κH} ≈ 0.1091，1−e^{−2κH} ≈ 0.2063。

**步 3｜後驗矩**（各 0.5 權重）：

- E[1−e^{−κH}] ≈ 0.5×0.75 + 0.5×0.1091 ≈ 0.4296。
- E[1−e^{−2κH}] ≈ 0.5×0.9375 + 0.5×0.2063 ≈ 0.5719。
- Var(e^{−κH})：兩值 0.25 / 0.8909，均值 0.5704 → 方差 ≈ 0.1027。

**步 4｜代入式 (7)**：

- 分子 −d·E[1−e^{−κH}] = 2×0.4296 ≈ 0.8592。
- 分母 0.5719 + (−2)²×0.1027 ≈ 0.5719 + 0.4108 ≈ 0.9827。
- ω ∝ 0.8592 / 0.9827 ≈ 0.874。

**步 5｜代數消融：刪除方差項**：這不是「把同一混合後驗當成點估計」；同一分布的方差不能任意設零。這裡僅固定 E[1−e^{−κH}] ≈ 0.4296、E[1−e^{−2κH}] ≈ 0.5719，把 Var 項設為 0：

- ω ∝ 2×0.4296 / 0.5719 ≈ 0.8592 / 0.5719 ≈ 1.502。

僅就這一步，方差項把未正規化權重由 ≈1.502 壓到 ≈0.874。這是代數隔離，不是有效後驗的寬度變化。

**步 6｜真正的單一 τ 點估計會同時改動三個矩**：若改用 τ̂ = 60 天（κ ≈ 0.01155），e^{−κH} ≈ 0.7937：

- 1−e^{−κH} ≈ 0.2063，1−e^{−2κH} ≈ 0.3700。
- ω_plugin ∝ 2×0.2063 / 0.3700 ≈ 0.4126 / 0.3700 ≈ 1.115。

這裡變動的不只是方差項：期望收斂幅度由 0.4296 掉到 0.2063、分母第一項由 0.5719 掉到 0.3700。混合後驗 0.874 vs 單一 τ̂ 點估計 1.115 的差 −0.241，不能全歸給方差懲罰；步 5 的 1.502 → 0.874 是 −0.628。兩者方向相同、量級不同，且比較對象不同。

**步 7｜讀數**：在固定 d 下，d² 會放大方差懲罰，這是式 (7) 的機械性質；但真實資料上「寬度隨窗口變化」與「懲罰依賴 d²（即使寬度固定也會重新加權大偏離）」兩條通道**未被分離**（原文 §4.2 明說 has not been separated on real data），所以本頁不把 §5 的倉位結果歸因於其中任一條。

## §3 · 數據層

| 項目 | 內容（原文表格數值／敘述） |
|---|---|
| 市場 / 樣本 | 120 檔 Korea Exchange (KRX) 股票，2010 – 2026（4,072 d） |
| Universe 規則 | top-200 KRX by median traded value；取 120 檔有完整歷史者；流動性排名用樣本末資訊，delisted stocks dropped |
| 特徵 | ROE, D/E, net margin, revenue growth, B/P, E/P（DART, point-in-time, from 2016） |
| 頻率 / 視窗 | 日頻；look-back L=60、持有 H=20（non-overlapping）；τ* = 60 d |
| 殘差構造 | PCA（K=3 on 252 d）、FF3 on [1, size, B/P]；NFE 用 AFM 的因子（K=5, annual refits） |
| 評估期 | 2018-04 至 2026-07（evaluation days），net of costs |
| 成本 | 5 bp per unit of turnover；1 bp per day on the short notional；因子對沖腿未單獨交易、未單獨計成本 |
| 切分 | split 2018-04；每個期間的前 30% 作訓練 |

**時間邊界分列**（本方法沒有 tokenizer、沒有文字語料、沒有適配後微調）：

| 階段 | 資料來源 | 時間邊界 |
|---|---|---|
| 網路訓練（唯一訓練階段） | **純模擬路徑**，真實報酬不進入 | 原文：No real returns enter the training of the network |
| 生成模型選擇（新息律） | 真實殘差窗口 | full-sample windows |
| 宇宙篩選／下市剔除 | 樣本末流動性排名與完整歷史 | end-of-sample information |
| 超參矩匹配 (φ, γ) | 訓練集真實窗口 | (0.09,0.08) in 2010 – 18；(0.28,0.12) in 2018 – 26 |
| 每股 τ 先驗（部署適配） | 訓練集 L 天塊＋empirical Bayes 向面板均值收縮 | 在訓練集估；上線只重加權，網路不重訓 |
| Conformal（outcome-fitted） | 訓練集 | 前 80% fit / 後 20% calibrate，隔 L 天 |
| Conformal（模擬器方法） | 完整訓練切分 | 寬度按分位數乘法校準 |

**樣本外與容量假設**：宇宙篩選使用樣本末資訊（流動性排名在樣本末、剔除下市股），作者自述。因子對沖腿未單獨交易、未單獨計成本。韓國交易稅與賣空限制在窗口部分期間生效，**未建模**，作者明言因此 levels are not implementable returns。【推斷】這些限制使水準不可直接視為可實現報酬；其對各項結果的淨影響方向未量化，不宜宣稱全部同向。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | https://github.com/minkeymouse/nfe（原文 §4 明示：package、experiment scripts、data pipeline） |
| Checkpoint | 未披露（所讀文本未見權重發布說明） |
| License | 未披露（所讀文本未見；論文本身為 CC BY 4.0） |
| 複現難度 | 高——需 KRX 日頻＋DART point-in-time 基本面；FF3 因 size/B-P 聯合覆蓋僅從 2017 起，前期弱有一部分是覆蓋問題 |
| 數據可得性 | 公開日頻市場資料；DART 為公開揭露源；作者聲明不再散布供應商價格資料 |

## §5 · 評測 / Benchmark

**表 A — 主結果（Table 1，KRX 面板 2018-04 至 2026-07，net of costs）**

| 數據集/市場 | Metric | 比較基線 | 本方法 | Δ |
|---|---|---|---|---|
| KRX 2018-04 至 2026-07 | SR | 1.45（四代主比較中的 AFM 因子基線，作者自行實作） | 1.49（NFE，後驗倉位） | +0.04 |
| KRX 2018-04 至 2026-07 | μ (%) | 10.4（AFM 基線） | 10.6（NFE） | +0.2 |
| KRX 2018-04 至 2026-07 | σ (%) | 7.2（AFM 基線） | 7.1（NFE） | -0.1 |
| KRX 2018-04 至 2026-07 | MDD (%) | 4.2（AFM 基線） | 4.1（NFE） | -0.1 |
| KRX PCA 殘差 | SR | 0.57（PCA 基線，base rule） | 0.68（PCA 殘差＋後驗倉位） | +0.11 |

**基線逐一列出（不合成區間）**：FF3 SR 為 -0.12、market-adjusted 對照亦為 -0.12、random benchmark 為 -0.23；PCA 0.57、AFM 1.45、NFE 1.49。作者明言 FF3 earn no excess return over the random benchmark。FF3 與 market-adjusted 的 SR 同為 -0.12，但兩者是不同列、不同語義，不可互代。

**表 B — 預測分布的預測–實現落差（Table 2，held-out KRX PCA 殘差，2018-04 至 2026-07）**

| 數據集/市場 | Metric | 比較基線 | 本方法 | Δ |
|---|---|---|---|---|
| KRX PCA 殘差 | gap (bp / 持有期) | 1084（OU 點估計） | 588（amortized OU-form，ours） | -496 |
| KRX PCA 殘差 | gap (bp / 持有期) | 1084（OU 點估計） | 137（amortized direct，ours；功能形式控制，非部署濾波器） | -947 |
| KRX PCA 殘差 | CRPSS | 0.00（unconditional baseline） | -0.11（amortized OU-form） | -0.11 |
| KRX PCA 殘差 | CRPSS | 0.00（unconditional baseline） | 0.00（amortized direct） | 0.00 |

**判讀——哪部分是能力，哪部分不是**

1. **AFM → NFE 的 +0.04 不可據此判定能力優勢**。原文自己寫 ΔSR 的標準誤是 0.41，並直言 a difference of this size is not resolvable here；0.41 是各策略相對隨機基準之 ΔSR 的標準誤，原文沒有提供 AFM–NFE 配對差的標準誤。這是「本研究無法分辨」，不是「已證明沒有差異」。任何把它讀成「後驗提升 Sharpe」的結論都是誤讀。
2. **真正的落差在世代之間，不在 NFE 與 AFM 之間**：PCA 0.57 → AFM 1.45 的跨度遠大於 AFM 1.45 → NFE 1.49。這支持論文自己的結論句——殘差構造比交易策略更重要。但作者同時警告 the ordering therefore depends on the period，且面板是在樣本末端選出的，因此**水準值是上界**。較早期間的評估結果則顯示 no strategy outperforms the benchmark。
3. **gap 的縮小是真的，但預測落差降低可由均勻縮放預測重現**。原文明寫：把點預測乘一個在訓練集上定的常數，就能在抽樣誤差內達到後驗的 gap；在 admitted process 的真值已知抽樣上，OU-form 後驗 gap 減去縮放點預測 gap 為 -1 to -5 bp；在第二價格源重建的 KRX 面板兩期為 +1±17 與 -13±12 bp；在 no-reversion null 上為 -38±10 與 -42±10 bp。**這不是說縮放可複製後驗倉位、區間或風險**：控制組保留點估計的倉位，只縮放其預測；實驗沒有證明決策層倉位機制可被常數縮放複製。也不等於條件性不確定性定價能力。
4. **可能被高估的部分**：宇宙篩選用樣本末資訊（倖存者偏差）、因子對沖腿未計成本、韓國稅與賣空限制未建模——三項都由作者在 Appendix B 自陳；【推斷】這些限制使水準偏樂觀，但各項淨影響方向未量化，不能宣稱全部同向。
5. **無任何方法跑贏無條件基準**：CRPSS 全表 ≤ 0，原文：no method has positive skill against the baseline。
6. **direct 變體的角色要限定**：它是功能形式控制，用來檢查 OU 形式是否限制預測；決策層需要半衰期後驗，direct 不是可部署濾波器。137 bp 屬 PCA 殘差預測診斷，不是組合中 NFE 濾波器的部署成績。

**輔助證據（逐項轉述）**：識別力上，真值已知高斯固定尺度特例的所有實用估計器落在 0.61 – 0.66 AUC；看得到真實 level/scale 的 Bayes bound 到 0.93，但該比較在窗內狀態固定下計算。level 已知時分離度由 0.62 – 0.64 升到 0.79 – 0.81。未攤銷 particle filter 的 rank correlation 0.74 對 NFE 的 0.59。panel form 在共享因子 panel 上 NLL 由 2.088 降到 1.762，其中 0.179±0.003 是對獨立因子控制的對比，並在 disjoint seeds 復現 0.170±0.002；真實資料上 panel form 未測試。250 日窗口的增益被作者主動撤回——直接交易 250 日偏離的對照在同期 SR 達 2.77、另一期為 -0.91；作者不 claim 它，保留 60 日窗口。其他變體：學出來的橫截面步 SR -0.03；threshold rule 在 PCA 殘差上 0.75；AFM 交易網路在本論文窗口下 0.25，非復現。Overshoot finding：此處 regime 標籤由特徵學得的 level g(X) 計算，不是式(6)後驗，式(7)也不使用該標籤；base rule 在 diverging 窗口後 +118 bp、converging 窗口後 +27 bp，差 -92±25 bp；正控制 +81±20、random-walk null +8±21、按倉位權重歸一化後排序仍在、disjoint period -90±37，作者列為 confirmed，且與預先預測相反，留待後續。每股先驗在三個 disjoint training seeds 上 confirmed：training prior 平均 SR 1.424 / -0.259，per-asset prior 1.470 / -0.193，per-asset centres permuted 1.458 / -0.215；增益方向一致但相對噪聲小，多數來自先驗中心與寬度對面板的校準，而非每股中心。

**覆蓋率校準與控制**：Figure 2(b) 在 truth-known draws、training prior 下、per-asset reweighting 前檢查半衰期後驗；其覆蓋率貼合 nominal，故作者推論同一後驗尾質量在相同邊際意義下校準。每股先驗重加權後的校準未單獨檢查。原後驗在所示 seed 的最大覆蓋偏差為 0.022，另兩個 seeds 至多 0.010；半寬後驗則為 0.08 – 0.12。雙樣本檢驗控制：same-process windows 約 0.50、wrong process 約 0.90；改 kernel 只移動 AUC 至多 0.014，這是對 kernel 低敏感度的結果，不是正控制。其他零模型與正控制包括 random benchmark、market-adjusted negative control、no-reversion null、injected positive controls。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **宇宙篩選含樣本末資訊**：Appendix B 明言 Both universe filters use end-of-sample information (liquidity ranking at the sample end; delisted stocks dropped)。
- **水準值不可實現**：Appendix B 明言韓國交易稅與賣空限制在窗口部分期間生效，levels are therefore not implementable returns。
- **因子對沖腿未計成本**：Appendix B 明言 the implied factor-hedge legs are not separately traded or costed。
- **生成模型仍與資料可分辨**：Appendix A 明言選定模型 AUC 0.678 remains distinguishable from the data，並要求解讀 KRX 組合後驗聲明時保留此殘差差異。
- **雙樣本檢驗對 kernel 敏感度低**：Appendix A 明言 kernel moves the AUC by at most 0.014；檢驗約束新息律與尺度，不驗證 kernel 或 level。
- **regime 數是建模選擇**：Appendix A 明言 the number of regimes is a modeling choice；level 未經檢驗，每個候選以 m ≡ 0 模擬；特徵學習的 g(X) 不勝 250 日移動平均，只作 overshoot 檢定的預註冊錨點。
- **250 日窗口增益被撤回**：§4.1 明言該增益是窗口效應，主動放棄。
- **主結果單一種子**：§4 明言 Results come from one training seed and are exploratory unless stated；只有四項過了 confirmation run（overshoot、panel-form pooling、per-asset prior、gap 排序）。
- **後驗的兩條通道未分離**：§4.2 明言 whether the sizing results … come from the width varying across windows or from the dependence of the penalty on d², has not been separated on real data。
- **per-asset 先驗後的校準未單獨檢查**：§4.2 明言 before the per-asset reweighting, which is not separately checked。
- **panel form 不是上線濾波器**：§3.3／§5 明言只在 truth-known 抽樣上測；真實資料上 its value there is untested。
- **direct 變體不是部署濾波器**：§3.3 明言它是功能形式控制，決策層需要半衰期後驗。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **【推斷】兩態切分是決策驅動的近似**：定義 3.1 用 ρ* 把 κ 譜切成 C/D，等於假設「夠快 vs 不夠快」的二分足以承載決策；若真實 τ 在 τ* 附近有質量，後驗尾質量會被切點位置主導。論文的敏感性分析所讀文本未見，圖表／附件待核。
- **【推斷】在本論文的 60 日窗／20 日持有設定下，level 識別是主要已識別損失來源**：level-known 對照把分離度由 0.62 – 0.64 抬到 0.79 – 0.81，而餵入未去均值價差只加 +0.0002±0.0005 AUC。這是論文量到的比較，不是「任何窗口內改良都不可能」的證明；panel form 在真值已知抽樣上已改善半衰期後驗，真實資料效力尚未測試。
- **【推斷】容量約束被成本假設局部掩蓋**：5 bp 換手＋1 bp/日空頭名義只計在被交易的殘差組合上，因子腿的容量約束不在成本內；真實資金規模的可承載量所讀文本未見，未披露。此為成本定義上的推斷，不是論文給出的容量數字。
- **【推斷】regime 依賴不可外推**：較早期間所有策略都不勝基準、250 日對照在兩期之間由 2.77 擺到 -0.91，說明此類方法表現高度依賴期間；任何單一期間的排序都不能當成跨所有模型／市場的必然因果。
- **【推斷】label-free 診斷的陷阱是最可轉移的教訓**：作者自陳在補上零模型與正控制之前，好幾個結論 lacked power or were biased upward。復現時應把零模型與正控制視為方法的一部分，而非附錄。
- **【推斷】panel form 未測不等於無效**：它在共享因子真值已知 panel 上改善 NLL 並保持校準，但真實資料上兩條橫截面路線（學出來的橫截面步、特徵學的 level）未帶來幫助；不能由此斷言 panel form 在真實資料必然無效。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Avellaneda–Lee (2010) | 點估計 κ vs τ 的完整後驗 | 前作程式位置未核 | 古典基線；PCA 殘差 SR 0.57 |
| Epstein et al. (2025) AFM | 端到端交易網路 vs 預先固定決策層 | 實作細節待核 | 本論文四代主比較中的 AFM 因子基線 SR 1.45；其交易網路在本論文 60d/20d 窗口下 0.25，原文明言非復現 |
| TabPFN / TabICL / TabFM | 行可交換的上下文學習 vs 保序遞歸 | 授權待核 | 不在金融面板上；NFE 借 row-embedding 類比但改成 GRU |
| Deep Kalman Filter / amortized VI (Krishnan 2015; Marino 2018) | 學習狀態空間模型（Krishnan）／以變分網路攤銷濾波（Marino） vs 在檢定過的模擬器上攤銷濾波 | — | 方法系譜上游 |
| NPE / SBI (Papamakarios & Murray 2016; Cranmer 2020; Talts 2018) | 通用 SBI vs 金融價差專用＋新息律否證 | — | 方法系譜上游 |

🎤 **Interview Tip**

**正確答**：「這篇的貢獻不是 alpha，是把模型知道自己多不確定變成可披露、可進入規則的量。證據是 gap：OU 點估計預測超過實現 1084 bp / 持有期，amortized OU-form 588 bp，direct 變體 137 bp；但作者自己補了一刀——把點預測乘一個常數就能達到同樣的 gap，所以這是預測落差的均勻縮放重現，不是條件定價，也不證明可複製倉位、區間或風險。作者判斷 Sharpe 上 AFM 1.45 → NFE 1.49 的 +0.04 在本期間不可分辨；0.41 是各策略相對隨機基準之 ΔSR 的標準誤，原文沒有提供 AFM–NFE 配對差的標準誤；這是無法判定，不是已證明無差異。真正大的落差在殘差構造：PCA 0.57 → AFM 1.45。」

**錯答**：「NFE 用貝氏後驗取代點估計，所以 Sharpe 從 1.45 提升到 1.49，證明不確定性建模能加 alpha。」——原文兩處直接打臉：後驗 adds no alpha，且該差距 not resolvable。

**次級陷阱**：把 0.61 – 0.66 AUC 說成「NFE 識別力優於插件估計」。原文是所有人**在同一窄帶內**，後驗 gains no accuracy over a point estimate；且該比較限於高斯固定尺度特例，0.93 另有窗內狀態固定條件。

### 7.1 可證偽預測（本頁提出，尚待驗證）

1. **到 2027-06-30**：若有人把 panel form NFE 在真實 KRX（或同類市場）上訓練，並跑出相對 AFM 的**跨不重疊期間同號**增益，則本頁「panel form 在真實資料上價值尚未顯示」的判讀被推翻。本頁的條件式預期（僅為待驗證假說）是增益不易出現，理由是論文在真實資料上試的兩條橫截面路線——學出來的橫截面步 SR -0.03、特徵學的 level 不勝 250 日移動平均——都沒帶來幫助；但 panel form 在真實資料上**未被測試**，本頁不宣稱它已被證明無效。
2. **到 2027-12-31**：預期後續工作會把「level 與 reversion 共用窗口」當成主要攻擊點，而非只改進濾波器本身。若屆時主流跟進方向仍是換更強的序列編碼器，則本頁判讀失效。
3. **到 2027-06-30**：若獨立復現顯示「後驗 gap 減去同設定縮放點預測 gap」在分期間落在預先設定的等效容忍範圍之外，且以不確定區間判斷，則均勻縮放等效性結論需下修。原文在 no-reversion null 給出 -38±10 與 -42±10 bp，這是該比較量的錨點；檢驗縮放等效性應比較後驗與同設定縮放控制，而不是只看差值符號或原始 gap 是否反轉。

## §8 · For the Reader

- **因子研究員**：把這篇當「殘差構造 > 交易策略」的實證背書讀。AFM 1.45 / NFE 1.49 的差距不可據此判定差異，而 PCA 0.57 → AFM 1.45 的跨度才是可操作的槓桿。若要抄一個東西，抄 Appendix A 的**雙樣本檢驗選新息律**流程：15 個候選、AUC＋2SE 低於 0.75 的預設判準（Table 3 兩欄依序為 crypto、KRX）、same-process 約 0.50 與 wrong process 約 0.90 的控制，並正確理解 kernel 只移動 AUC 至多 0.014 是低敏感度而非正控制。
- **高頻執行**：這不是日內方法的直接替代。L=60 / H=20 non-overlapping、成本只算到 5 bp 換手與 1 bp/日空頭名義，因子腿未計成本。若要拿去做日內，式 (7) 的 H 需重參數化，而 κ 在那個尺度上的可識別性所讀文本未見——本頁推斷，未驗證。
- **組合配置**：可取的是**風險披露與收縮機制**，不是把它當 alpha。式 (7) 的分母第二項讓不確定性進入權重；但作者自己的證據顯示預測落差降低可由常數縮放預測重現，而該控制保留點估計倉位、只縮放預測，**不能**推論可複製後驗倉位、區間或風險。用於配置時應把它視為校準層與不確定性披露層，而非收益來源。
- **LLM-agent / 表格基礎模型研究者**：這篇是「row embedding＋潛狀態動態」的金融特化。可借鑑的是時間邊界分列：網路訓練只吃模擬路徑（No real returns enter the training of the network）、超參矩匹配與每股 τ 先驗與 conformal 只在訓練集擬合、表格期間標為 2018-04 至 2026-07；評估使用訓練切分後日期，精確首日未能確認。但這不是一條乾淨的樣本外鏈——新息律的雙樣本檢定用 full-sample windows，宇宙篩選也用 end-of-sample information，且選定模型與真實資料仍可分辨（0.678）。解讀 KRX 組合上的後驗聲明時必須帶著這個折扣。panel form 則是本文尚未驗證真實資料價值的方向。
- **RL 策略研究者**：注意這裡的決策層是**預先固定、不學的**。作者的理由值得記住：若規則也學，績效差異就分不清是規則還是狀態估計帶來的。這是把 reward learning 換成 identifiability 的典型取捨。
- **研究學生**：把 §5 的 label-free 診斷紀律抄進 pipeline——每個診斷配零模型＋正控制，否則你會像作者一樣撤回 250 日窗口那個結論。另外，這篇是少見的「負面結果寫得比正面結果多」的論文，適合作為「如何誠實報告一個沒有 alpha 的方法」的範本；同時記得區分「不可分辨」與「已證明無差異」。

## References

- Chang, M. *Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage.* ICAIF '26. arXiv:2610.25802v1 [q-fin.GN], 19 Oct 2026. DOI: 10.1145/3858063.3858894 · https://arxiv.org/abs/2610.25802v1
- 代碼／資料：https://github.com/minkeymouse/nfe（原文 §4 明示）
- **Lineage — 統計套利與因子**：Fama & French (1993); Gatev et al. (2006); Avellaneda & Lee (2010), *Quantitative Finance* 10(7):761–782; Guijarro-Ordonez et al. (2021), arXiv:2106.04028; Epstein et al. (2025), ICAIF '25:396–404.
- **Lineage — 濾波與攤銷推論**：Kalman (1960); Hamilton (1989), *Econometrica* 57(2):357–384; Krishnan et al. (2015), arXiv:1511.05121; Marino et al. (2018), NeurIPS 31; Papamakarios & Murray (2016), NeurIPS 29:1028–1036; Cranmer et al. (2020), *PNAS* 117(48):30055–30062; Talts et al. (2018), arXiv:1804.06788; Gu & Dao (2023), arXiv:2312.00752.
- **Lineage — 表格基礎模型**：Müller et al. (2022); Hollmann et al. (2023); Hollmann et al. (2025), *Nature* 637:319–326; Qu et al. (2025), ICML 42; Kong & Das (2026), Google Research Blog.
- **方法論引用**：Lopez-Paz & Oquab (2017), ICLR; Romano et al. (2019), NeurIPS 32; Efron (2011), *JASA* 106(496):1602–1614; Smith & Winkler (2006), *Management Science* 52(3):311–322; Gneiting et al. (2007), *JRSS-B* 69(2):243–268.