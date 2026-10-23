<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=全自动黑盒 -->

# Neural Filter Estimator (NFE) 解構（Neural Filter Estimator (NFE)）

> **來源發布**：2026-10-19 · ICAIF26 · arXiv [2610.25802](https://arxiv.org/abs/2610.25802)
> **arXiv 原文**：[Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage](https://arxiv.org/abs/2610.25802v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 日頻波段 × 監督回歸 × 風險擇時 × 全自動黑盒」；它解的不是預測精度，而是**舊統計套利管線丟掉估計不確定性**這個 prior gap——把殘差價差的隱狀態後驗（而非點估計）餵進固定決策層。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①用 amortized neural posterior estimation 訓練一個 GRU + 混合密度頭的濾波器，一次前向就吐出橫截面每條 spread 的隱狀態（均值回歸半衰期）後驗。②核心 trick 是「以候選新息律對真實殘差做 two-sample 檢定選生成模型，再在模擬路徑上監督訓練」，因此真實報酬完全不進網路訓練。③對「風險择時」軸的意義是：把不確定性顯式寫進下單公式，直接對抗選股後的 winner's curse。④關鍵實證：KRX 上 OU 點估計的預測—實現落差為 1084 bp／持有期，amortized OU-form 為 588 bp；但組合 Sharpe 只與 AFM 打平（1.45 vs 1.49，差 +0.04）。

**X-Ray.** 這篇的價值不在「又一個深度學習打敗 PCA」，而在於它把統計套利裡的**估計器角色**重寫了：傳統管線把 `κ`、`m` 當真值代入，NFE 承認它們是後驗。放回五軸 Pareto，它在「日頻 × 量價表格」這一格上幾乎沒有換取任何 alpha（作者自己反覆說 posterior adds no alpha、no method shows skill against an unconditional baseline），換來的是**校準**——一個可用於倉位縮放的方差項。工程上它清掉了兩個老坑：（i）參數不確定性無法進入 mean–variance 權重；（ii）橫截面選股後 top-ranked spread 的估計誤差被系統性高估。它打不開的 envelope 也很清楚：**level 與 reversion 共用同一個 window**，這是識別上限（所有實用估計器都擠在 0.61 – 0.66 AUC 的窄帶），任何更好的推論都救不回來，只有更多資訊才行。對量化讀者，最該抄的不是網路結構，而是「label-free 診斷必須配 null model 與 positive control」這條方法論——作者正是靠它撤回了自己 250-day 窗口的 Sharpe 2.77。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | Avellaneda–Lee (2010) 點估計管線 | AFM (Epstein et al. 2025) | NFE（本文） |
|---|---|---|---|
| 殘差怎麼來 | PCA 因子，外生 | attention over characteristics，端到端學 | **沿用 AFM 因子**（不自己學；作者實測自學橫截面步只有 Sharpe -0.03） |
| 狀態怎麼估 | 滾動窗口最小平方點估計 | 端到端交易網路隱式吸收 | **GRU 編碼 60 日窗口 → 5 分量 MDN 給 log₂τ 後驗** |
| 不確定性 | 丟棄 | 未顯式建模 | **保留；經 (7) 式進入倉位** |
| 決策層 | 閾值規則（tuned） | 學習式交易網路 | **預先固定、永不擬合到已實現損益** |
| 訓練資料 | 真實殘差 | 真實殘差 | **純模擬路徑**（真實報酬不進訓練） |

### 1.2 ⚡ Eureka

> **一句話**：把「這條 spread 現在是在收斂還是發散、收斂多快」定義成生成模型的潛在變數，用 simulation-based inference 一次性攤銷成後驗，再讓 mean–variance 公式把後驗寬度當成懲罰項。

直覺：點估計給你一個 `κ`，你只能開一個倉；後驗給你一整條 `τ` 的分佈，寬的時候公式自動縮手、窄的時候原樣下手。**它不改善你猜得多準，它改善你知不知道自己猜得多不準。**

### 1.3 信息流

```
Panel（日 × 股票）
   │  cross-sectional step:  factor model（AFM 或 PCA）
   ▼
ε_{i,t} → S_{i,t} = Σ_{s≤t} ε_{i,s}   （spread）
   │  window w_{i,t}（L=60 日，窗內自身 mean/std 標準化）
   ▼
GRU_ψ (h=64) ──► MDN_ψ：log₂τ 上 5 分量高斯混合 q_ψ(τ|w)
   │                                   │
   │                          tail mass > τ*=60 ──► P(z=D|w)（僅報告）
   ▼
後驗動差 E[1−e^{−κH}], E[1−e^{−2κH}], Var(e^{−κH})
   ▼
固定決策層 (7)：ω ∝ −d·E[1−e^{−κH}] / (E[1−e^{−2κH}] + d²Var(e^{−κH}))
   ▼
橫截面去均值 → 單位 gross → 組合 P&L
   ▲
   └── 訓練端：模擬路徑 (w, θ, z) 真值已知 ⇒ NLL；部署端：per-asset prior 重加權、split-conformal
```

## §2 · 數學層

📌 **Napkin Formula**

```
ω_{i,t} ∝  −d · E_q[1 − e^{−κH}]  /  ( E_q[1 − e^{−2κH}] + d² · Var_q(e^{−κH}) ),   κ = ln2/τ
```

複雜度：訓練一次性 104 s（桌面 CPU）；推論 **2.5 ms／橫截面**（對比未攤銷粒子濾波 7.9 s）。難點在分母第二項——`d²Var(e^{−κH})` 對任何點估計恆等於零，所以 (7) 式退化即古典 plug-in 規則；這一項正是「後驗」與「點估計」在數學上唯一的接口。

**Loss / 訓練**：`L(ψ) = −E_{(τ,w)~p_train}[log q_ψ(τ|w)]`，於模擬窗口上取樣；訓練時窗內半衰期固定、`log₂τ ~ U(3,400)` 日均勻，先驗刻意做平，使 head 輸出呈 likelihood-shaped，部署時才能用 per-asset 高斯先驗重加權而不必重訓。梯度層面無特殊設計（Adam，4,000 steps，batch 512，3×10⁻³ cosine，60,000 draws）。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ **以下全部為假設/示意數字，用來演示 (7) 式的機制，不是論文實證結果。** 論文真實數字一律見 §5。

1. **假設**：標準化偏離 `d = 2`，持有期 `H = 20` 日。
2. **假設**：後驗對 `τ` 給出 50/50 的兩點質量，分別落在 `τ = 10` 與 `τ = 120` 日（即模型「不知道是快是慢」）。
3. 令 `X = 2^{−H/τ}`：`τ=10 → X = 0.25`，`τ=120 → X = 0.891`。
   - `E[X] = 0.570`、`E[X²] = 0.428` ⇒ `Var(X) = 0.428 − 0.570² = 0.103`
   - `E[1−e^{−κH}] = 1 − 0.570 = 0.430`；`E[1−e^{−2κH}] = 1 − 0.428 = 0.572`
4. **代回 (7)**：分子 `= −2 × 0.430 = −0.859`；分母 `= 0.572 + 2² × 0.103 = 0.983` ⇒ `ω ∝ −0.874`。
5. **對照點估計**：若窗口剛好只給出 `τ = 10`（`Var = 0`），同式退化為 `−1.5 / 0.938 = −1.60`。
6. **結論**：同一個訊號下，後驗把倉位縮到約 0.55 倍——**縮手不是因為訊號變差，而是因為後驗變寬**。這就是本文全部的交易學主張。

## §3 · 數據層

- **市場/樣本**：KRX，top-200 by median traded value 篩出的 120 檔完整歷史個股，2010 – 2026（4,072 d）；評估期 2018-04 至 2026-07（evaluation days）。
- **頻率**：日頻；`L = 60`、`H = 20`，non-overlapping；`τ* = 60` 日。
- **特徵**：ROE、D/E、net margin、revenue growth、B/P、E/P（DART，point-in-time，自 2016 起）。
- **殘差構造**：PCA `K=3` on 252 d；FF3 on `[1, size, B/P]`；AFM `K=5`，年度 refit。
- **成本假設**：5 bp per unit of turnover；short notional 1 bp/day。
- **樣本外協定**：walk-forward，每年 refit；split 2018-04。
- **容量/可得性紅線（作者自述）**：兩個 universe filter 都用了 end-of-sample 資訊（樣本末流動性排序、退市股剔除）；韓國交易稅與賣空限制未建模，故 **levels 不是可實現報酬**；FF3 需同日 size 與 B/P，聯合覆蓋僅自 2017 起。vendors 的價格未重散布。
- **兩條 cross-section 用於檢定**：crypto 與 KRX（雙樣本檢定同時報兩者）。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | `https://github.com/minkeymouse/nfe`（package + experiment scripts + data pipeline，原文 §4） |
| Checkpoint | 未披露（所讀文本未見權重釋出說明） |
| License | 論文 CC BY 4.0；代碼 license 所讀文本未見（TBD） |
| 複現難度 | 中高：模型本身輕（GRU-64 + 5 分量 MDN），但 KRX/DART point-in-time 特徵與 vendor 價格需自備，且 vendor 價格未重散布 |
| 數據可得性 | 模擬路徑可完全複現；真實面板需自建（作者另用第二家 vendor 重建面板做穩健性） |
| 計算量 | 訓練 104 s；推論 2.5 ms／橫截面 |

## §5 · 評測 / Benchmark

| 數據集/市場 | Metric | 前SOTA（AFM，本文重實作、同 base rule） | 本方法（NFE） | Δ |
|---|---|---|---|---|
| KRX 120 檔，2018-04 至 2026-07（evaluation days，淨成本） | 年化 Sharpe SR | 1.45 | 1.49 | +0.04 |

> 表中 Δ 為 `1.49 − 1.45`，方向定義為「本方法 − 前SOTA」；+0.04 亦逐字見於原文（"a difference in level of +0.04"）。

**逐代排序（同表、同 base rule，皆為本文實作）**：FF3 SR `-0.12`；PCA `0.57`；AFM `1.45`；NFE `1.49`；market-adjusted 對照 `-0.12`；random benchmark `-0.23`。作者明言只比排序不比絕對值（不與原文的美股數字對照），且 **levels 是 upper bound**（panel 以樣本末資訊篩選）。

**哪些 Δ 是真 capability、哪些不是**

- **不是**：+0.04 的 Sharpe。作者自陳 ΔSR 的 standard error 為 `0.41`，「a difference of this size is not resolvable here」。這在統計上等於零。任何把它讀成「NFE 打敗 AFM」的結論都是誤讀。
- **是**：**預測—實現落差的收斂**。表 2 中 OU 點估計 `1084` bp、amortized OU-form `588` bp、amortized direct `137` bp、mixture density net `798` bp、boosted quantiles `277` bp、unconditional baseline `0`；跨三個訓練種子 OU-form 落在 `572 – 593`、direct 落在 `120 – 150`，對照點估計 `1084` 與 mixture density net `658 – 838`。但作者自己拆了這個 capability：**這是均勻收縮而非條件收縮**——「a point forecast scaled by a single constant … reaches the posterior's gap within sampling error」。所以真正新增的是那個**區間**（點估計沒有區間），不是更準的均值。
- **不是**：alpha。全表 CRPSS 對 unconditional baseline 無一為正（最好的是 amortized direct `0.00`）；作者結論句是 "No method shows skill against an unconditional baseline"。
- **可能是過擬合/未計成本**：`250`-day 窗口的優勢被作者主動撤回——一個預先指定的控制組直接交易 250 日偏離，在同期拿到 SR `2.77`、在另一期 `-0.91`。這是本頁最該記住的失敗模式：**窗口長度是隱藏的自由度，會偽裝成模型能力。**
- **未計**：韓國交易稅、賣空限制；因子對沖腿未單獨交易或計費。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位到章節）

- **§4.1**：panel「selected at the end of the sample, the levels are an upper bound」。
- **§4.1**：較早期間**沒有任何策略**跑贏 benchmark；排序依期間而變。
- **§4**：結果來自單一訓練種子，「exploratory unless stated」。
- **§4.1**：250-day 窗口的增益是 window effect，主動不 claim。
- **§4.1**：overshoot 發現**與預先登記的預測相反**（发散窗後 +118 bp、收敛窗後 +27 bp，差 −92±25 bp），且「left open」；(7) 式的收縮方向與此發現相反。
- **§4.2**：per-asset prior 的重加權「not separately checked」。
- **§4.2**：sizing 效果究竟來自後驗寬度變化、還是來自 `d²` 對大偏離的重加權，「has not been separated on real data」。
- **§4.2 / §3.3**：panel form 只在 truth-known draws 上測，真實資料上價值未測。
- **Appendix A**：雙樣本檢定選出的過程 AUC 為 `0.678`，仍與真實資料可區分，「the posterior claims on the KRX portfolio should be interpreted in light of that residual discrepancy」；且該檢定對 kernel 幾乎無鑑別力（AUC 移動至多 `0.014`）。
- **Appendix B**：universe filter 用 end-of-sample 資訊、剔除退市股；稅與賣空限制未建模；levels 非可實現報酬。
- **§5 Conclusion**：真實資料上「whether the information is worth more than the cost of acting on it is the question this work leaves open」。

### 6.2 推斷的隱含假設（本頁推斷，非作者原文）

- **生成模型是全部的地基**：後驗只在「模擬器等於真實」時校準。作者用雙樣本檢定把門檻設在 AUC + 2SE < `0.75`，而選中的模型實際仍有 `0.678` 的可區分度——**校準保證只在 admitted process 內成立**，跨 regime 的外推沒有保證（本頁推斷）。
- **level 與 κ 共用同一個 window**：這是識別上限的來源（已知 level 時分離度由 `0.62 – 0.64` 升到 `0.79 – 0.81`）。任何假設「換更強的 encoder 就能突破」的期待都不成立，除非引入 window 外的 level 資訊；作者試過的兩條橫截面路線（自學橫截面步、characteristics 學 level）在真實資料上「did not help」。
- **成本結構固定且線性**：5 bp／turnover + 1 bp/day short notional，未含稅與借券費；在毛利極薄的均值回歸上，成本假設的小變動足以翻轉排序（作者自己說 mean-reverting structure「smaller than the cost of trading on it」）。
- **容量**：unit gross、dollar-neutral、120 檔 KRX，屬小型市場的相對小容量；未討論市場衝擊。
- **生存者偏差**：作者明示退市股被剔除且 universe 用樣本末排序——**這是已披露的偏誤，不是隱含假設**；但「levels 是 upper bound」的量化幅度未披露。
- **單一市場、單一期間**：KRX 2018-04 至 2026-07；crypto 只出現在雙樣本檢定的 AUC 欄，未做組合回測。不得外推到其他市場或所有模型。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Avellaneda–Lee (2010) | 同一訊號 `d` 與同一 OU 核，但只給點估計 → (7) 退化為 plug-in | 未披露 | 本文主要對照（PCA 殘差 SR `0.57`） |
| AFM (Epstein et al. 2025) | 因子同源；差別在「有無潛在狀態後驗」與「學習式 vs 固定決策層」 | 未披露 | 本文主基線（SR `1.45`）；其端到端交易網路在本協定下 SR `0.25`（作者聲明非復現） |
| Guijarro-Ordonez et al. (2021) | 端到端交易因子組合，非濾波 | 未披露 | 僅列於 lineage，未數值比較 |
| TabPFN / TabICL / TabFM | 同為「合成先驗預訓練 + 一次前向」，但 rows 視為 exchangeable，時間面板不是 | 未披露 | 概念對照；本文主張以遞歸時間步補上此缺口 |
| 未攤銷粒子濾波（25×256 particles） | 同一模型、更高保真 | — | 半衰期秩相關 `0.74` vs NFE `0.59`；`7.9` s vs `2.5` ms |

🎤 **Interview Tip**

- **問**：「NFE 比 AFM 好在哪？」
  **正確答**：Sharpe 上**沒有**可解析的優勢（`1.45` vs `1.49`，ΔSR 標準誤 `0.41`）。它換到的是**校準的預測區間**與更小的 forecast–realized gap（`1084` → `588` bp），而且這個收斂被作者證明可用「點預測乘一個常數」複製——所以真正增量是區間與不確定性，不是 alpha。
- **錯答**：「它用深度學習把 Sharpe 從 1.45 提升到 1.49，證明後驗優於點估計。」——把噪聲當因果，且忽略了作者自己說 posterior adds no alpha。
- **追問**：「那為什麼還要後驗？」
  **正確答**：古典管線選股後，被選中的 spread 恰好是估計誤差最有利的那批（winner's curse）；後驗的 `d²Var` 項在選股後做二次懲罰，把倉位往後驗窄的標的移。這是 decision-level 的修正，不是 forecast-level 的。
- **陷阱**：「所以換更強的 encoder 可以突破 0.61 – 0.66 AUC？」**錯**。那是 window 的識別上限，作者用已知 level 的對照（`0.79 – 0.81`）證明損失來自 level 估計而非估計器。

### 7.1 可證偽預測（帶日期）

1. **2027-12 前**：若 panel form（selective state-space encoder + 橫截面池化）在真實 KRX 上跑出顯著優於 per-asset NFE 的 ΔSR，且超過 `0.41` 的噪聲門檻，則「level 共用 window 是主要損失」的解釋被強化；若仍無差異，則識別上限的主因需重新定位到資料而非架構。作者自己在 §5 已把這條列為 open question。
2. **2027-06 前**：把韓國交易稅與賣空限制納入成本後，NFE 相對 AFM 的 ΔSR 若維持在噪聲內（不變號、不放大），則「殘差構造比交易政策重要」的結論穩健；若排序翻轉，則 §4 的「residual construction matters more than the trading policy」需下修。
3. **2027-12 前**：以第二家 vendor 重建面板並重跑 (7) 式的 width-vs-`d²` 分離實驗，若顯示 sizing 效果主要來自 `d²` 項而非後驗寬度，則本文「不確定性進入決策」的機制主張被降格為「二次型倉位規則」——這是作者 §4.2 自承尚未分離的缺口。

## §8 · For the Reader

- **因子研究員**：把注意力放在 §4.2 的 level-known 對照（`0.62 – 0.64` → `0.79 – 0.81`）。這告訴你：殘差價差的可識別性被「level 與 reversion 共用窗口」鎖死；與其換更強的序列模型，不如想辦法從橫截面或基本面取得 window 外的 level 資訊。作者試過的兩條橫截面路線在真實資料上都失敗（自學橫截面步 SR `-0.03`），這是有價值的負面結果。
- **高頻執行**：`2.5` ms／橫截面、`104` s 訓練，這個成本結構在日頻波段是免費的；但 5 bp／turnover 與 1 bp/day short notional 的假設下，作者的結論是「mean-reverting structure is real, but it is smaller than the cost of trading on it」。若你的成本更低，這篇的 alpha 才開始有空氣。
- **組合配置**：可直接借用的是**校準診斷流程**——truth-known draws 上畫 reliability diagram、用 null model（vol-matched random walk）與 positive control 檢驗每個 label-free 診斷。作者的 250-day 撤回就是沒做這一步的代價。
- **LLM-agent / 表格基礎模型讀者**：§2 把 tabular foundation model 的 row embedding 對應到 factor model，並指出其 in-context 階段把 rows 當 exchangeable set——時間面板不是。這是把 TabPFN 系方法搬進金融面板時最容易踩的假設錯誤，NFE 用遞歸時間步繞開它。
- **RL / 決策策略讀者**：本文的決策層是**預先固定、永不擬合**的，這是它與端到端 RL 交易網路最大的分歧點。理由是因果識別：規則若可學，績效差異就無法歸因到狀態估計。想把 (7) 式換成學出來的 policy，等於放棄這個識別策略——值得，但要想清楚你要回答哪個問題。
- **研究學生**：這是一篇方法論示範。三個測量（simulator 檢定 → 可識別性上界 → 決策層效果）層層收斂，且作者主動列出四項通過 confirmation run 的發現（overshoot、panel-form pooling、per-asset prior、forecast–realized gap 排序）。讀它的方式是讀「他們怎麼殺掉自己的結論」，不是讀數字。

## References

- 原文：Minkey Chang. *Amortized Neural Filtering for Calibrated Posteriors in Statistical Arbitrage.* ICAIF '26. arXiv:2610.25802v1 [q-fin.GN], 19 Oct 2026. DOI: 10.1145/3858063.3858894. Code: https://github.com/minkeymouse/nfe
- 直接前作：Epstein, Wang, Choi, Pelger. *Attention Factors for Statistical Arbitrage.* ICAIF '25, 396–404. doi:10.1145/3768292.3770398
- 古典管線：Avellaneda & Lee. *Statistical Arbitrage in the US Equities Market.* Quantitative Finance 10(7), 2010, 761–782. doi:10.1080/14697680903124632；Gatev, Goetzmann, Rouwenhorst. *Pairs Trading.* RFS 19(3), 2006, 797–827. doi:10.1093/rfs/hhj020
- 端到端深度統計套利：Guijarro-Ordonez, Pelger, Zanotti. *Deep Learning Statistical Arbitrage.* arXiv:2106.04028
- 濾波 / 攤銷推論：Kalman (1960)；Hamilton, *Econometrica* 57(2), 1989, 357–384；Krishnan, Shalit, Sontag, *Deep Kalman Filters*, arXiv:1511.05121；Marino, Cvitkovic, Yue, NeurIPS 31, 2018；Papamakarios & Murray, NeurIPS 29, 2016；Cranmer, Brehmer, Louppe, *PNAS* 117(48), 2020；Talts et al., arXiv:1804.06788
- 表格基礎模型：Müller et al., ICLR 2022；Hollmann et al., ICLR 2023 / *Nature* 637, 2025；Qu et al., ICML 2025（TabICL）；Kong & Das, *Introducing TabFM*, Google Research Blog, 2026
- 統計工具：Efron, *JASA* 106(496), 2011, 1602–1614（Tweedie / selection bias）；Smith & Winkler, *Management Science* 52(3), 2006, 311–322（optimizer's curse）；Lopez-Paz & Oquab, ICLR 2017（classifier two-sample test）；Romano, Patterson, Candès, NeurIPS 32, 2019；Gneiting, Balabdaoui, Raftery, *JRSS-B* 69(2), 2007, 243–268
- 來源鏈接：https://arxiv.org/abs/2610.25802v1