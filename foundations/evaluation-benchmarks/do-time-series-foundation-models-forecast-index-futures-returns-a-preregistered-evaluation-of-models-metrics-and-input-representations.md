<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=生成式大模型 alpha=端到端表征 autonomy=全自动黑盒 -->

# Do Time-Series Foundatio 解構（Do Time-Series Foundatio）

> **來源發布**：2026-11-09 · （無 venue） · arXiv [2611.12141](https://arxiv.org/abs/2611.12141)
> **arXiv 原文**：[Do Time-Series Foundation Models Forecast Index-Futures Returns? A Preregistered Evaluation of Models, Metrics, and Input Representations](https://arxiv.org/abs/2611.12141v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量价表格 × 高頻日內 × 生成式大模型 × 端到端表征 × 全自動黑盒」，但它的貢獻不是造 alpha，而是**把 TSFM 在低訊號域的表面技能拆成可檢驗的零假設**——預註冊盲測 + SPA 多重檢驗 + 成分控制（ridge／EWMA／sign-flip null）。解的 prior gap 是：多數 TSFM 財經評測只報 IC／RankIC 或與 random walk 的單點比較，沒有 fixed analysis plan、沒有 matched null、沒有「representation 注入的成分」對照。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `高频日内` | `生成式大模型` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 作者在 NQ／ES 一分鐘資料上預註冊盲測五個凍結 TSFM（Chronos-Bolt、Chronos-2、TimesFM 2.5、Moirai 2.0、TiRex）的四種輸入表征、5-120 分鐘共 160 格，另以 Kronos 做獨立實驗（僅 native R0）；核心 trick 是把「有沒有增量技能」定義成 MAE ratio 相對 no-change 的 SMI 門檻（≤ 0.995 且區間上界 < 1）＋ Hansen SPA 多重檢驗，並用 ridge／EWMA／sign-randomised null 對照。對「生成式大模型 × 端到端表征」這條軸★的意義是否定的：五模型 panel 盲測 0 / 160 格勝過 no-change，SPA p = 1.000（兩市場、T = 0），回溯 hold-out 0 / 120；Kronos 獨立實驗亦未勝 no-change。表面技能全部有機械解釋——price-level 相關 > 0.9996（no-change 更高）、日內 RankIC 靠 random walk 下的均值回歸 artifact、對 HAR 的波動率優勢被輸入自帶的 causal EWMA（0.955 NQ / 0.968 ES）蓋過。

**X-Ray.** 這篇不是方法論文，是一篇**評測方法論論文**，而它的方法論恰好是量化研究員最該抄的那一段。五軸 Pareto 上它站在「最先進表征 × 最難標的」的角落：把 2024-2026 一整批 zero-shot TSFM 丟進近乎不可預測的日內指數期貨收益，然後拒絕用方向分類率、IC 或 level correlation 這些「看起來會贏」的指標下結論。它解掉的舊工程坑有三個：（一）多模型 × 多 horizon × 多表征的 p-hacking——用固定 C1 家族（60 格／市場）＋ SPA 而不是逐格 t 檢定；（二）「表征改善了」與「模型有技能」混為一談——用 transformation-aware ridge 當同訊息集對照，0 / 80 格能加在 ridge 之上；（三）指標建構偏差——把日內 RankIC 在 random walk 下的期望共變異數寫成 closed form，並用 sign-flip null 重現 95.1% / 95.5%。它打不開的 envelope 要說清楚：只測凍結 zero-shot（Kronos 例外）、只測兩個最流動合約、只測 2023-2026、偵測極限在 δ ≈ 0.1 以下完全空白、且作者自承 points bound **無法排除**該固定策略存在高於成本的邊際（60% NQ cells、29% ES cells 的上界高於成本）。對量化讀者的意義：把它當成「採用新模型前的驗收清單」，而不是「TSFM 無用」的判決書。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（所讀文本所述） | 本研究改動 | 為什麼是關鍵 |
|---|---|---|---|
| 檢定框架 | Noguer i Alonso & Franklin (2026)、Konat & Sarwas (2026)、Zhong et al. (2026)、Brini (2026) 多為 rolling-origin 或單點對比 random walk／HAR | 預註冊（本地時間戳 + SHA-256，計畫 2026-11-02 凍結、2026-11-04 03:49 UTC 鎖定）＋ 盲測開一次 ＋ 回溯 hold-out 僅作描述性檢查 | 把「多模型多 horizon 把噪音變發現」的通道先封死 |
| 統計推論 | 逐格顯著性或 pooled IC／RankIC（Kronos 用 IC／RankIC） | C1 家族 {R1, R2, R4} × 5 模型 × {5, 30, 60, 120} min = 60 格／市場，用 Hansen SPA（stationary bootstrap、mean block 1 day、1,000 resamples）＋ 預設 SMI（ratio ≤ 0.995 且上界 < 1） | 檢定的是「有沒有一格贏」，不是「哪一格贏」；同時擋住 magnitude-insensitive 的指標 |
| 成分控制 | 多數評測只比 random walk／HAR | no-change、transformation-aware ridge、causal EWMA、placebo（隨機符號窗）、sign-randomised null、points bound | 把「表徵注入的簡單成分」與「模型技能」分離 |

資料／模型變體上的第三個改動是 **R0 native / R1 log returns / R2 volatility-standardised / R4 volume clock** 四表征同 protocol 對跑，且 R1／R2 有同訊息集的 ridge 對照（ridge 只用 t-1 以前的 bar，訊息集與模型完全相同；只有驗證階段的 B2 ridge／B3 LightGBM 用到 P(t)，不進盲測）。

### 1.2 ⚡ Eureka

**一句話 trick：** 不要問「模型準不準」，要問「把 no-change、ridge、EWMA 與隨機符號這四個零假設都扣掉之後，還剩什麼」——剩下的才叫增量技能。

**直覺：** 日內指數期貨收益的變異數絕大部分是噪音，任何「看起來有預測力」的指標，先檢查它是否只是把價格水平、日內共同路徑、或輸入自帶的波動率估計重新包裝一次。作者把這件事做成三個可執行檢定：MAE ratio（量級敏感）＋ matched null（符號隨機化）＋ ingredient control（ridge／EWMA）。

### 1.3 信息流

    1-min OHLCV (Databento CME Globex, NQ/ES front-month, roll 8 days before 3rd Friday)
            │
            ├─[R0 native price]──────────────┐
            ├─[R1 log returns]──┐            │   400 completed bars (t-400 .. t-1)
            ├─[R2 z_k = r_k/σ_k (EWMA α=0.05)]┤            │
            └─[R4 volume clock (median vol NQ 111 / ES 321)]│
                                              ▼
                         frozen zero-shot TSFM（點預測＝20 條抽樣路徑中位數 / 分位數中位數）
                                              │
                                              ▼
                                f = log return anchored at close(t-1)
                                              │
            ┌─────────────────────────────────┼──────────────────────────────┐
            ▼                                 ▼                              ▼
      no-change (f=0)                  ridge (同表徵、同訊息集)         EWMA / HAR（波動率）
            │                                 │                              │
            └──────────► MAE ratio Σ|y-f| / Σ|y|（day-block bootstrap）◄─────┘
                                              │
                              ┌───────────────┼────────────────┐
                              ▼               ▼                ▼
                        SMI 閘門        Hansen SPA (C1)    sign-flip / placebo null
                    (≤0.995, 上界<1)   (p=1.000, T=0)     (RankIC、forecast size)

## §2 · 數學層

📌 **Napkin Formula**

- 主損失：`ratio = Σ_t |y_t − f_t| / Σ_t |y_t|`，`y = log P(t+h)/P(t)`，h ∈ {5, 30, 60, 120} 分鐘；ratio < 1 才對模型有利。
- 判定：cell 勝 no-change ⇔ `ratio ≤ 0.995` 且 95% day-block bootstrap 上界 `< 1`（SMI）；表征改善 ⇔ `ratio_R − ratio_R0 ≤ −0.005` 且上界 `< 0`。
- 均值回歸 artifact：`E[cov_day(x, y)] = (σ²/n) tr(AᵀHB)`，`H = I − 11ᵀ/n`；特例 `x_t = −P_t` 時 `E[cov_day] = σ²(n−1)/(2n) > 0`。

**直覺：** 雖然每個 origin 上 `E[x_t y_t] = 0`，但日內中心化時，早期 origin 的預測器含有晚期 origin 才實現的增量，「共同路徑」把共變異數推到正值——這就是日內 RankIC 在 random walk 下為正、且一個 trivial revert-to-mean 規則就能拿 +0.348（30m）／+0.604（120m）的原因。

**loss／訓練細節：** 全部模型凍結 zero-shot；Kronos 另做從零訓練與 fine-tune 作為訓練控制。評估用絕對誤差，DM 檢定用平方誤差（Newey-West lag 5，one-sided），波動率用 QLIKE；文中明確聲明「direction DM p 值永不當成 MAE 優越性的檢定」。複雜度層面無訓練成本，主成本是推理 rollout（R4 需外推 200-1,100 步）。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部是示意玩具數字，不是論文實證結果**；論文真實數字一律見 §5。目的是把 SMI 閘門與「表征改變量級而非技能」手算一遍。

1. **示意輸入**：取 3 個 origin（真實 NQ 30 分鐘網格是每日 12 個）。實際報酬 `y = (+6, −4, −5)` bp（示意），no-change 的誤差和 `Σ|y| = 15`。
2. **模型輸出**：某 TSFM 給 `f = (+2, −1, +4)` bp（示意）。`Σ|y − f| = |6−2| + |−4+1| + |−5−4| = 4 + 3 + 9 = 16`。
3. **算 ratio**：`16 / 15 ≈ 1.067 > 1` → 這一格輸給 no-change；此示意 ratio > 1，與原文盲測所有 cell 皆未低於 1 的定性一致（原文範圍見 §5）。
4. **做表征實驗（縮放而非方向）**：把 `f` 乘 0.5 → `(+1, −0.5, +2)`，`Σ|y − f| = 5 + 3.5 + 7 = 15.5`，ratio `≈ 1.033`；把 `f` 乘 0 → ratio 恰好 1.000（就是 no-change）。**縮小預測量級只會把 ratio 推向 1，不會穿過 1** —— 這就是「representation changes magnitude, not skill」的手算版；要真的 < 1，方向與量級得同時對。
5. **過 SMI 閘門**：門檻是 `ratio ≤ 0.995` 且 bootstrap 上界 < 1。示意上需要把誤差和壓到 `0.995 × 15 = 14.925` 以下（示意算術），且日區塊重抽的 95% 上界也得 < 1；只達成一項不算贏。
6. **對照真實結論**：真實盲測 160 格裡最好的 per-cell 區間下界是 1.0048，hold-out 最好的點估計是 1.0113——連第 5 步的門檻邊都沒碰到。

## §3 · 數據層

| 項目 | 內容（所讀文本逐字） |
|---|---|
| 標的 | E-mini Nasdaq-100（NQ，primary）、E-mini S&P 500（ES，replication）；front-month，roll 於第三個週五前 8 天 |
| 頻率 | 1-minute OHLCV 全日 bar；volume clock 用 1-second bars |
| 來源 | Databento CME Globex；hold-out 用 TopstepX 1 分鐘歷史 |
| 目標 | `y = log P(t+h)/P(t)`，`P(t)` 為第 t 分鐘第一筆成交；h ∈ {5, 30, 60, 120} 分鐘 |
| Origin 網格 | 美東固定時鐘：30 分鐘 horizon 為 10:00-15:30 每 30 分鐘（每日 12 個）；120 分鐘為 10:00-13:30（每日 8 個）；5／60 分鐘類比；跨 roll 或跨 > 60 分鐘 session gap 的窗丟棄 |
| Context | 最後 400 根已完成 1 分鐘 bar；所有預測锚定 close(t-1)，不重新锚定 |
| 切割 | Train NQ 2018-01–2022-12／ES 2020-01–2022-12；Validation 2023-01–2024-12；Blind NQ 2025-01-01–2026-05-11（333 days）／ES 2025-01-01–2026-04-16（316 days）；Hold-out 2026-09-03–2026-10-30（39 days） |
| 日頻樣本 | Blind NQ 349 sessions（2025-01-02 to 2026-05-08）、ES 332；日頻 hold-out NQ 119（2026-05-12 to 2026-10-29）、ES 136（2026-04-17 to 2026-10-29） |
| 對齊細節 | close(t-1) 與 open(t) 在 NQ 有 26%、ES 有 55% 的 origin 重合；post-hoc close-anchored 重評（非預註冊，外審後加）仍 0/160，best ratio 1.0075 vs 1.0073，median change 0.0000 |
| 污染控制 | 模型發布日與 cutoff 於開跑前記錄；Blind 在 Kronos（June 2024）與 Chronos-Bolt 發布之後，2025 年發布的模型只有部分 Blind 保證乾淨；pre-cutoff vs post-cutoff 誤差相等（Chronos NQ 30m 1.086 vs 1.084） |

**樣本外與容量假設：** 盲測與 hold-out 皆為時間外樣本；但 hold-out 是「2026-09-03 至 2026-10-30 錄製、2026-11-04 才打開」的回溯性 hold-out，不是前瞻收集。容量／市場衝擊未建模，只用了固定 round-trip 成本（points bound 用 NQ 0.85 pt、ES 0.75 pt；H9/H11 對 ES 仍套用 NQ 的 0.85 pt），可承載規模 **未披露**。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | **TBD** — 所讀文本只提到內部檔名（`tsfm_infer.py`、`kronos_infer.py`、`score_rep.py`、`research_notes/rankic_theory.py`）、results log 與「release manifest」，未見公開連結 |
| Checkpoint | 各模型版本記於 release manifest；參數量：Kronos-small 24.7M、Chronos-Bolt-base 205M、Chronos-2 約 120M、TimesFM 2.5 200M、Moirai 2.0-R-small 11.4M、TiRex 約 35M；授權 **未披露** |
| License | **TBD**（所讀文本未見） |
| 複現難度 | 高：需付費 1 分鐘 + 1 秒 CME 資料、六個模型權重、Mac MPS 與 CPU 數值差（TimesFM 6e-9、Moirai 5e-9、Chronos-Bolt 2e-7、Chronos-2 7e-5、TiRex 1.8e-4 log-return 單位） |
| 數據可得性 | 商業資料（Databento／TopstepX），非公開；1-second bars 僅用於 volume clock |
| 可重算性 | 作者做了獨立 AI agent 審計：從存檔預測重算所有 headline 表到約 1e-14、獨立 Hansen 實作（T = 0, p = 1）、約 250 origins/model 的 CPU 重推理；Kronos 未重推理（僅重評分） |

## §5 · 評測 / Benchmark

> 表中「前SOTA」欄一律是**基線／控制組**本身的值，不是被合成的 SOTA 區間；所讀文本未提供可直接對齊的差值時，Δ 欄留空，不以其他位置補算。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| NQ 5m（Blind） | QLIKE / HAR ratio | causal EWMA 控制 0.955 | Moirai 2.0 R2 0.971 |  |
| ES 5m（Blind） | QLIKE / HAR ratio | causal EWMA 控制 0.968 | Moirai 2.0 R2 0.981 |  |
| NQ 日頻 H9（Blind） | 累積交易 P&L (USD) | always-long +\$13,587 | Kronos-small 訊號交易 +\$4,682 |  |
| ES 日頻 H9（hold-out） | 累積交易 P&L (USD) | always-long +\$418 | Kronos-small 訊號交易 +\$3,049 |  |

**主結果（無可配對 Δ 者，逐字列出）：** 盲測 `0 / 160` 格勝 no-change，SPA `p = 1.000`（NQ 與 ES、available-case 與 common-sample 皆同）、`T = 0`；MAE ratio 範圍 `R0 1.019-1.145`、`R1/R2 1.007-2.07`、`R4 1.07-139`；能加在 ridge 之上的 cell `0 / 80`；最低 per-cell 95% 區間下界 `1.0048`。Hold-out `0 / 120`，最佳 cell 為 Chronos-2 R1 ES 5m `1.0113 [1.0063, 1.0170]`，IC `-0.016`。

**波動率（H7）：** Moirai 2.0 + R2 @ 5m 通過註冊規則——NQ QLIKE/HAR `0.971 [0.957, 0.986]`、DM p `1.2e-4`、80% coverage `0.815`；ES `0.981 [0.965, 0.997]`、p `0.015`、coverage `0.814`。只有 NQ 也過更嚴的 volatility SMI（≤ 0.98）。對照 EWMA，模型 QLIKE ratio 為 `1.016 [1.011, 1.022]`（NQ）與 `1.014 [1.007, 1.020]`（ES），兩者區間都排除 1。30 分鐘以上無模型勝 HAR（best `1.08-1.12`），hold-out 任何 horizon 皆不勝 HAR。解讀：QLIKE 表面勝 HAR，但輸入自帶的 causal EWMA 更低（`0.955` NQ／`0.968` ES），模型對 EWMA 反而顯著更差，故沒有增量價值的證據。

**日頻：** H9 NQ Blind IC `+0.116`（p `0.015`，Validation 為 `+0.133`），但交易 `+$4,682` < random-direction 95th `$16,219`、< always-long `+$13,587` → 交易條件失敗；NQ hold-out IC `-0.008`。H11 未複製（IC `-0.032`、`-$13,576`）。ES hold-out H9 同時過兩道門檻（IC `+0.228`、p `0.004`、`+$3,049` vs random `$1,833`、always-long `+$418`），作者明示為二次市場、報告不宣稱。

**表徵實驗：** R1／R2 勝 R0 於 Blind `13/160`、hold-out `11/120`（最佳 Chronos-2 R1 `1.009` NQ / `1.007` ES），但 0 格勝 no-change；Moirai R1 @ 120m 約差 `2×`。R4 volume clock 最差：Blind 最高達 `15×`（Moirai）與 `139×`（TiRex 120m），Validation `0/40`，去 drift 後仍 `1.06-1.6`。decoder 審計（in-sample 最小平方縮放）best ratio `0.9965-1.002`（120 cells），transformation-aware ridge `0.994-1.002`。

**表面技能三件套（同一批預測）：** price-level 相關每個模型都 `≥ 0.9996`，但 no-change 更高（NQ 30m `0.99991` vs 模型 `0.99969-0.99991`）；日內 RankIC 由 trivial revert-to-mean 規則拿下 `+0.348`（30m）／`+0.604`（120m），sign-flip null 均值 `+0.331`／`+0.577`（null/real `95.1%`／`95.5%`，P(null ≥ real) = `0.09`／`0.06`），Gaussian RW 模擬 `+0.303`／`+0.515`；Kronos-small per-day RankIC `+0.188`（30m）但 pooled `-0.032`，與該規則相關 `0.857`。

**偵測極限（planted signal）：** δ = 0.3 時 5m 20 格 MAE `0.949-0.98`（一格 `1.0003`）、Spearman IC `0.19-0.29`、detected `13/20`；oracle `0.927 / 0.914`（NQ/ES）、ridge `0.944 / 0.941`。Chronos-2 與 TiRex 5m 11 次平均 `0.969-0.982`、IC `0.22-0.23`、`11/44` runs；30m 反而 `1.07-1.12`、IC `0.03`、`0/44`。偵測率表：NQ 5m δ=0.2 為 oracle `10/11`、ridge `9/11`、TiRex `0/11`、Chronos-2 `0/11`；δ=0.3 為 `11/11`、`11/11`、`3/11`、`0/11`；ES 5m δ=0.3 為 `11/11`、`11/11`、`6/11`、`2/11`；NQ 30m δ=0.3 為 `10/11`、`4/11`、`0/11`、`0/11`。

**經濟界線：** 固定策略 `sign(f)`、`t+h` 出場，per-cell 上界 95% 超過 round-trip 成本的比例為 NQ `60%`、ES `29%`；NQ per-cell 上界最大 `13.1` pt vs 成本 `0.85` pt；下界 > 0 的 cell 為 `3/80` NQ、`4/80` ES（≈ 隨機機率）。

**哪些 Δ 是真 capability？** 嚴格說，本頁沒有任何一格可以標成「真 capability」。QLIKE 的基線對照顯示：對 HAR 的表面優勢低於輸入自帶的 EWMA，且對 EWMA 的 ratio 區間排除 1；H9 NQ Blind 的日頻 P&L 低於 always-long 與 random-direction 95th（ES hold-out H9 則高於兩者，作者報告不宣稱），不是技能指標而是成本與部位方向的結果。可能的過擬合／泄漏通道：Validation 上看過後才註冊 H9／H11（作者自承 H9 在 Validation `+0.133` 而 Blind `+0.116`），intraday hold-out 僅 39 天（日頻 hold-out 為 NQ 119 sessions／ES 136 sessions）且由同一批作者錄製、先前用於無關交易策略工作；成本未含滑價與容量。唯一「乾淨正向」的訊號是 planted-signal 復原（5 分鐘部分恢復、30 分鐘退化），但那是合成資料，不能外推。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（§8，含 §3／§8 disclosures）

- 只涵蓋 2023-2026 一個資產類別的兩個指數期貨；結果可能不適用於其他資產、頻率或 regime。
- 只評估凍結模型，Kronos 例外（另做從零訓練與 fine-tune）。
- Blind 可能部分早於某些 2025 模型的訓練資料；hold-out 不會，但只有 39 天。
- R1／R2 解碼用 per-step 中位數之和，因為模型給的是 per-step 分位而非聯合路徑，median-of-sums 無法計算；decoder 審計只覆蓋線性縮放。
- δ ≈ 0.1 以下的效應即使 oracle 也偵測不到；points bound 無法排除該固定策略存在高於成本的邊際。
- 無 signed trades，價格與成交量以外的訊息不可檢。
- 預註冊弱於第三方 registry；審計由獨立 AI coding agent 而非人類審查執行，提出四項 disclosure：舊版計畫未保留（hash 依賴本地 session log）、凍結前已印出各 split 的聚合報酬統計、Blind 分多次而非單一腳本執行、波動率 pass 未在第一版結果 log 明列。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **【推斷】no-change 是充分的 hard benchmark。** 對兩個最流動合約的 5-120 分鐘收益，這個假設有該文資料支撐；但若存在 session／roll／macro 日曆型的條件漂移，no-change 就低估了「平凡但可交易」的成分——本文未測條件均值模型。圖表／附件待核。
- **【推斷】400 bar context 足夠。** 所有模型固定吃 400 根 1 分鐘 bar；更長記憶、跨資產（例如 NQ 對 ES 領先落後）、或 1-second 解析度輸入未被測，故「表徵沒救」只適用於被測的 R0／R1／R2／R4 與 400-bar 協定。
- **【推斷】凍結 zero-shot ≈ 採用場景。** 若實務上是 fine-tune 或全量重訓，本頁結論不覆蓋——文中 Kronos 從零訓練 `1.10` vs zero-shot `1.72`（微調後 `1.079`）、fine-tune `1.99` 正是提示「訓練會改量級但不改結論」的單點證據，且只此一例。
- **【推斷】成本與容量假設寬鬆。** 0.85／0.75 pt 是固定 round-trip；未建模滑價、部分成交、roll 前流動性、以及日內策略的容量衰減；因此 points bound 的「不能排除」應讀成「上界仍寬」，不是「存在可實現 alpha」。
- **【推斷】regime 依賴。** 2023-2026 的低利率後段／高波動段結構未必延續；H9 在 Validation 與 Blind 的 IC（`+0.133` 與 `+0.116`）到 hold-out 變成 `-0.008`，正是 regime 或樣本大小的雙重解釋，二者在此無法分離。
- **生存者偏差不是主問題**：NQ／ES 是刻意選的兩個最流動合約，輔助標的缺失不是 survivorship；真正的限制是**標的選擇本身即一種外部效度限制**（流動性溢價、日內微結構在流動性差的市場完全不同）。
- **數據泄漏**：sign-flip／placebo 窗與 pre-cutoff 檢查（`1.086` vs `1.084`）都指向無記憶化痕跡；但 Blind 與 2025 年模型的訓練窗重疊未被完全排除。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Kronos（Shi et al., 2026） | 財經專用 TSFM，宣稱 RankIC 大增益；本文用同一族模型在低訊號標的上 0/160 | 權重與 tokenizer 規格公開程度 **TBD**（所讀文本只給 12B K-line 預訓練與 Kronos-Tokenizer-base） | 本文的 Kronos-small zero-shot `1.72`（ES `1.69`）、從零訓練 `1.10`、微調 `1.99`，皆未勝 no-change |
| Chronos-Bolt／Chronos-2／TimesFM 2.5／Moirai 2.0／TiRex | 通用 zero-shot TSFM；本文把它們當被測對象而非對手 | 各模型 repo／license 未在文中列出（TBD） | 全數 0/160；表徵只改量級 |
| Konat & Sarwas (2026)（19 個固定收益與 FX 標的） | 同樣對 random walk，結論方向一致（所讀文本未述其是否預註冊或使用 SPA） | 否（文獻對照） | 相容 |
| Noguer i Alonso & Franklin (2026)（五檔美股） | 與 random walk＋訓練網路比，gain 小而稀疏；本文加盲測與 SMI | 否 | 相容 |
| Brini (2026)（realised volatility + recalibration） | 重新校準吃掉大部分對 HAR 的優勢；本文的 EWMA 控制扮演同一角色 | 否 | 相容 |
| ridge／HAR／EWMA（計量基線） | 同訊息集、同表徵的簡單對照；R1/R2 有 transformation-aware ridge，R4 無 ridge 對照 | 是（方法本身公開） | 0/80 格能加在 ridge 之上 |

🎤 **Interview Tip**

- **正確答：**「這篇的貢獻不是『TSFM 不能預測』，而是把『表面技能』操作化成三個可反證的檢定：量級敏感的 MAE ratio 對 no-change、SPA 對 60 格家族做多重檢驗、以及成分控制（ridge／EWMA）與符號隨機化 null。它的負面結論有明確邊界：凍結 zero-shot、兩個合約、2023-2026、δ ≈ 0.1 以下不可偵測，而且作者的 points bound 自己說不能排除高於成本的邊際。」
- **錯答：**「160 格全輸所以 TSFM 對金融沒用」——忽略作者 §1 明列的「不宣稱收益不可預測、不宣稱這些模型學不到金融結構、不宣稱任何已發表數字無效」；或把它說成「打敗了 SOTA」，實際上被比較的是 no-change／ridge／HAR／EWMA 這些**基線**，不是任何已發表的最佳模型。
- **加分答：** 指出日內 RankIC 的正值是 random walk 下的中心化 artifact，`E[cov_day] = (σ²/n) tr(AᵀHB)`，而 trivial revert-to-mean 就能拿 `+0.348`／`+0.604`——這是本頁最可直接移植到自家評測管線的一條。

### 7.1 可證偽預測（帶日期）

1. **2027-06-30 前**：若有人在 NQ／ES 5-120 分鐘、同 400-bar context、同 SMI（≤ 0.995 且上界 < 1）協定下，讓任一凍結 zero-shot TSFM 的某一格通過並使 SPA 在 NQ 拒絕「無 cell 勝 no-change」，本頁 §5 主結論即被推翻。（尚待驗證）
2. **2027-06-30 前**：若把同一批模型換成 1-second 解析度 context、或加入跨資產（ES→NQ）輸入後，仍無任何 cell 能加在 transformation-aware ridge 之上，則「表徵無效」可從 R0-R4 推廣到高頻微結構表徵一類。（尚待驗證）
3. **2027-12-31 前**：若有人在同協定下把 delta 拉到 δ ≈ 0.1 的 planted signal 上仍得到 `0/11` 級偵測率，則「偵測極限」的論述可收緊；若反而穩定偵測到，則本文的 power 分析偏保守。（尚待驗證）

## §8 · For the Reader

- **因子研究員**：把 §5 的三件套（no-change／ridge／sign-flip null）直接搬進你的特徵驗收流程——尤其「日內 RankIC 必須配 null」這條，任何以日內 origin 為橫截面算 rank 相關的因子報告都應重跑一次符號隨機化。
- **高頻執行**：注意 points bound 的語義——`60%` NQ cells／`29%` ES cells 的 per-cell 上界高於 round-trip 成本，代表**現有樣本無法排除**該固定 policy 的經濟邊際；但這是上界不是估計值，不能當成 alpha 存在性證據，且未含滑價與容量。
- **組合配置**：這篇給的是「採用新模型的驗收清單」——量級敏感損失、成分控制、偵測極限揭露。若供應商拿 IC／level correlation 給你看，請要求同時給 no-change 與 ridge 對照。
- **LLM-agent／自動化研究**：本文的獨立 AI coding agent 審計（重算 headline 表到約 1e-14、約 250 origins/model 重推理、四項 disclosure）是可複用的 agent-audit 範式；但注意它明確聲明不等於人類同行評審。
- **RL 策略**：若你的 reward 建立在日內方向，這裡的 null 校準提醒你：random walk 下的「看起來有訊號」足以讓 policy 學到均值回歸的假技能；先把 no-change 與 sign-flip baseline 寫進 reward 診斷。
- **研究學生**：可讀 §6 的 Proposition 與 Appendix A 作為「評測指標建構偏差」的入門範本——它把 Kendall (1954)、Dickey & Fuller (1979)、Nickell (1981) 的古典小樣本偏誤，映射到一個現代 metric 上。

## References

- 原論文：Rishabh Upadhyay, *Do Time-Series Foundation Models Forecast Index-Futures Returns? A Preregistered Evaluation of Models, Metrics, and Input Representations*, arXiv:2611.12141v1 [cs.LG], 9 Nov 2026. https://arxiv.org/abs/2611.12141v1
- Lineage（模型）：Kronos — Shi et al., AAAI 2026, doi:10.1609/aaai.v40i30.39730；Chronos — Ansari et al., TMLR 2024, arXiv:2403.07815；Chronos-2 — arXiv:2510.15821；TimesFM — Das et al., ICML 2024, arXiv:2310.10688；Moirai — Woo et al., ICML 2024, arXiv:2402.02592；Moirai 2.0 — arXiv:2511.11698；TiRex — arXiv:2505.23719
- Lineage（評測與統計）：Hansen (2005) SPA；Politis & Romano (1994) stationary bootstrap；Diebold & Mariano (1995)；Patton (2011) QLIKE；Corsi (2009) HAR；Kendall (1954)；Dickey & Fuller (1979)；Nickell (1981)；Wilson (1927)
- 同域對照：Konat & Sarwas (2026) PAEE, doi:10.1109/paee71887.2026.11660673；Noguer i Alonso & Franklin (2026) arXiv:2606.27100；Brini (2026) arXiv:2607.05291；Zhong et al. (2026) Finance Research Letters 110:110606；Rahimikia et al. (2025) arXiv:2511.18578；Wan et al. (2026) arXiv:2608.14106；FinTSB arXiv:2502.18834
- 來源鏈接：https://arxiv.org/abs/2611.12141