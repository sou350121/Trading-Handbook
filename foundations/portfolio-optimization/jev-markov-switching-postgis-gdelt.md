<!-- ontology-5axis data=文本另类 horizon=中长周期 paradigm=监督回归 alpha=风险择时 autonomy=人机协同可解释 -->

# Jev + Markov-switching + PostGIS/GDELT 解構（Jev + Markov-switching + PostGIS/GDELT）

> **來源發布**：2026-11-19 · （無 venue） · arXiv [2611.26716](https://arxiv.org/abs/2611.26716)
> **arXiv 原文**：[Are Calibrated Decision Models Calibrated on Markets? Geolocated News, Markov Switching and a System One Model for Nasdaq-100 Allocation](https://arxiv.org/abs/2611.26716v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文本另類 × 中长周期 × 监督回归 × 风险择时 × 人机协同可解释」。它解的不是預測精度，而是一個 prior gap：decision model 的 calibration 只在 base rate 穩定的標註 benchmark 上被檢驗過（Deußer et al. 2026），從未對已實現市場結果檢驗，更沒有被折算成 sizing 的經濟價值。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `中长周期` | `监督回归` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 每週五問 Jev「NDX 四週後是否收低」，並與 1990–2026 週報酬的 two-regime Markov-switching、PostGIS/GDELT 地理新聞特徵、以及兩者組合同台比較；核心 trick 是發現 raw Jev 的平均預測 0.48 對上實際下跌頻率 0.38、ECE 0.10，而一個只用歷史基率、不碰測試期結果的單參數 prior shift 就把 ECE 壓到 0.015，Platt／isotonic／與 Markov 集成反而更差。★ 對「风险择时」軸的意義是：calibration 的價值只在 sizing 上實現——half-Kelly 下 raw Jev 年報酬 3.1%、prior shift 後 14.5%（同期 QQQ 買進持有 15.3%）。方向完全不可預測，唯一有 skill 的是 Markov 對「>10% 下跌」的偵測（AUC 0.81）。

**X-Ray.** 這不是新模型論文，是一篇 calibration 的照妖鏡，而且是少見的「自曝其短」寫法。放在五軸 Pareto 上，它的文本模態幾乎全軍覆沒：headlines 對方向的增量為零（AUC 0.45 到 0.56，沒有一個輸入贏過 climatology，DM p ≥ 0.54），200 km 芯片廠／海峽緩衝區特徵也被作者自己判為負面結果。真正的 alpha 不在新聞、也不在 Jev，而在 Markov 的風險擇時——但它只對罕見事件有效（>10% 下跌 AUC 0.81），對方向退化為基率（AUC 0.50）。它解掉的舊工程坑有兩個：一是 prevalence shift 下 decision model 的 calibration 崩壞（Iqbal et al. 2026 的金融版實證），二是 LLM 回測最致命的前瞻偏差——作者用 knowledge probe 自己量出 Jev 的知識邊界（gap 由 0.88 掉到 -0.08），把樣本硬切在 2025-01 之後。打不開的 envelope 很明確：88 週已知結果、重疊四週窗，作者自估約 22 個獨立結果；單一指數、單一問題、封閉且會改版的模型、週頻且在同一根收盤成交。對量化讀者的刀口在兩個反例：prior shift 只是 Saerens 式一參數先驗校正，不是魔法，可貴的是「先驗錨定」這個動作本身；而 ECE 與經濟價值不同構——isotonic 把 ECE 從 0.031 壓到 0.014，Sharpe 卻只有 0.49，低於 raw Markov 的 0.60。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作做法 | 本文做法 | 為什麼會改變結論 |
|---|---|---|---|
| 檢驗對象 | decision model 的 calibration 在 base rate 穩定的標註問答集上報「良好」（Deußer et al. 2026；TypeSafe 2026） | 對已實現 NDX 報酬檢驗兩個事件：四週下跌（基率 0.367）、四週跌逾 10%（基率 0.041） | benchmark 的 prior 是平衡的，市場事件的 prior 不是；prevalence shift 直接把 ECE 從基準打壞 |
| 知識邊界 | 未處理；Jev 的 training cut-off 未公布 | knowledge probe 逐期量測，只用 2025-01 之後的 88 週做評估 | 沒有這一步，任何 2024 前的回測（含校正 map 的訓練）都是 look-ahead |
| 校正與定價 | Platt／isotonic 等 post-hoc map；組合模型 | 一參數 prior shift（錨在 1990 起歷史基率）× half-Kelly 配置，對照 climatology／raw Markov | 只有 26 到 87 對樣本時，學 map 的校正比不學更差；calibration 的價值只有在 sizing 上才看得見 |

### 1.2 ⚡ Eureka

> **raw Jev 對「下跌」的機率是一個近似固定的 log-odds 水平偏移，把它的平均 logit 平移到歷史基率的 logit，偏移消失，而週與週之間的相對排序一格不動。**

直覺：Jev 在 balanced benchmark 上學到的隱含先驗接近「五五波」，市場事件的先驗是 0.375；兩者差一個常數，減掉常數就對齊。這也是為什麼更貴的 isotonic 反而更差——它在 26 到 87 對樣本上擬合的是雜訊，不是偏移。

### 1.3 信息流

    [NDX 週線 1985-10 → 2026-10-23, 2142 週]        [GDELT 2.0: 15.3 million rows / 552 million events]
              │                                                  │ BigQuery: week × 0.1° cell × country
              │                                                  ▼
              │                                        PostgreSQL + PostGIS (SRID 4326)
              │                                          ├ ST_Within → 7 regions (buffered ~15 km)
              │                                          └ ST_DWithin(200 km) → 7 fabs + 5 chokepoints
              │                                                  │
              │                            ┌─────────────────────┴────────────────────┐
              ▼                            ▼                                          ▼
      two-regime Markov            conflict share z_t                   25 + 25 headlines / week (GDELT GKG)
      MLE, expanding window,       → logistic TVTP                      → OpenRouter Decisions API
      re-estimated every 13 weeks    (Diebold et al. 1994)                typesafe/jev-1.14-20261014
              │                            │                                          │
              └────────────┬───────────────┘                                        │
                           ▼                                                        ▼
                 p_MS = Σ_s Pr(s|π,P) Φ(...)                              p_Jev raw (0.45 to 0.54)
                           │                                                        │
                           │                                              prior shift (Eq. 3, 只用 t 以前)
                           └──────────────────┬─────────────────────────────────────┘
                                              ▼
                        half-Kelly w = clip(0.5·(μ̂(p) - r_f)/σ̂²(p), 0, 1) → QQQ / T-bill
                                    成本 5 bp per unit of turnover，週五收盤成交
                                              ▼
                        Brier / log-loss / ECE / AUC + Diebold–Mariano + block bootstrap

## §2 · 數學層

### 📌 Napkin Formula

    logit p̃_t = logit p_t - (1/t′) Σ_{s≤t} logit p_s + logit ȳ_{≤ t-H}          (Eq. 3, prior shift)
    w_t      = clip( f · (μ̂(p_t) - r_f,t) / σ̂²(p_t), 0, 1 ),  f = 0.5            (Eq. 4, half-Kelly)
    p_t^MS   = Σ_s Pr(s | π_t, P) · Φ( -Σ_k μ_{s_k} / √(Σ_k σ²_{s_k}) )           (Eq. 2, H=4 → 2^4 = 16 paths)

直覺：Eq. 3 只做兩件事——把本週 Jev 的 logit 減去「過去所有 Jev 預測的 logit 均值」，再加上「已知結果的歷史基率 logit」。它**不動相對排序**，只平移水平，且錨 `ȳ_{≤t-H}` 由 1990 起 35 年歷史主導，測試期結果一格都沒進來。Eq. 4 只是把這個矯正過的機率送進 fractional Kelly。

複雜度與訓練細節：prior shift 每週 O(1)；Markov 用 MLE 在 expanding window 上估計、每 13 週重估一次，中間由 Hamilton filter 以固定參數更新 π_t；Platt 對 logit p 做 logistic regression（Jev 版用 L2、C=1），isotonic 輸出 clip 到 [0.01, 0.99]，組合為 `logit p^C = a + b·logit p^MS + c·logit p^Jev`；Platt／isotonic 對 Markov 在 104 對已知結果後才啟動、對 Jev 在 26 對後啟動。顯著性用 Diebold–Mariano（Newey–West，lag = max{H-1, ⌊4(N/100)^{2/9}⌋}）與 circular block bootstrap（block 26 週、2000 抽）。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部是假設／示意的玩具數字，不是論文實證結果。** 論文真實數字一律留在 §5。

1. **錨**：假設截至本週、結果已知的歷史下跌頻率 `ȳ = 0.36` → `logit(0.36) = ln(0.36/0.64) ≈ -0.575`（示意）。
2. **本週 raw Jev**：`p_t = 0.50` → `logit = 0`；假設過去所有 Jev 預測的 logit 均值為 `0.05`（示意）。
3. **套 Eq. 3**：`logit p̃ = 0 - 0.05 + (-0.575) = -0.625` → `p̃ = 1/(1+e^{0.625}) ≈ 0.349`。水平從 0.50 下移到 0.349。
4. **下一週**：raw `p = 0.54` → `logit ≈ 0.160` → `logit p̃ = 0.160 - 0.05 - 0.575 = -0.465` → `p̃ ≈ 0.386`。兩週的機率差由 `0.04` 變成 `0.037`——**排序與相對間距基本保留，只有水平被搬移**，這就是「只修偏移、不動資訊」的算術樣貌。
5. **進 Kelly**：假設 `σ̂² = 0.0025`（四週）、`r_f = 0.001`、混合均值 `μ̂(p=0.50) = 0.003` → `w = 0.5 × (0.003 - 0.001)/0.0025 = 0.4`；同一組假設下 `μ̂(p̃=0.349) = 0.006` → `w = 0.5 × (0.006 - 0.001)/0.0025 = 1.0`，clip 後滿倉。
6. **讀出來的教訓**：兩個輸入的「資訊」一樣（排序不變），但一個讓你只放四成、一個讓你滿倉。§5 裡 raw Jev 平均只握 0.01 的 QQQ、prior shift 後 0.98，就是這個算術在真實樣本上的放大版。

## §3 · 數據層

**價格**：NDX 日收盤自 1985-10；QQQ（已調派息）自 1999-03；13 週 T-bill（^IRX），全部來自 Yahoo Finance。週以週五收盤界定，未結束的週丟棄。1985-10 至 2026-10-23 共 2142 個週對數報酬，四週後收低佔 0.367、跌逾 10% 佔 0.041；評估樣本（1992-10-23 至 2026-09-25、1771 週）分別為 0.365 與 0.043（Table 1 另列 0.043 與 0.045 兩處，逐字照錄）。週對數報酬 mean 0.26、std 3.29（全樣本）。

**地理事件**：GDELT 2.0 event table，取 2015-03-01 至 2026-10-23 20:00 UTC 之間加入座標的 root events，以 `DATEADDED`（而非事件發生週）歸週，週五 20:00 UTC（NDX 收盤後）加入者歸下一週——這是本文唯一真正的 point-in-time 防線。BigQuery 按 week × 0.1° cell × country × geocoding type 聚合，得 15.3 million rows、代表 552 million events。載入 PostgreSQL/PostGIS（SRID 4326）後接兩層：7 個區域多邊形（緩衝約 15 km，落在多區緩衝者歸最近的非緩衝區）與 key_sites（7 座晶圓廠 + 5 個海峽，`ST_DWithin` 200 km）；只 geocode 到國家的事件被排除（GDELT 把台灣放在國土質心，緊鄰 TSMC 廠）。GDELT 斷線以「小時覆蓋率 < 80%」刪週，影響 608 週中的 14 週。

**Headlines**：GDELT Global Knowledge Graph 標題，英文、按關鍵詞分 economy/markets 與 conflict/geopolitics 兩類，同標題合併、按不同來源數排名，剔除 `.uk`／`.ie` 網域（避免單一通訊社轉載洗榜），保留 ≥ 2 來源、每類每週前 25 則。合計 4,700 則、94 週（2025-01-03 至 2026-10-23），缺 2025-06-27 該週。

**樣本切分與容量**：測試期 95 個週五（2025-01-03 至 2026-10-23），結果已知 91 週（至 2026-09-25）；prior shift 需至少四個 Jev 預測，故前 3 週僅用於初始化，統一從 2025-01-24 起評：價格輸入 88 週、headlines 87 週、含 geo 特徵 85 週。容量假設：QQQ／T-bill 週頻、成本 5 bp per unit of turnover，未披露 AUM 或市場冲击模型。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 所讀文本只說「code, weekly GDELT features, headline metadata, all Jev requests/responses 可向作者索取」，無公開連結；Repo URL：TBD |
| Checkpoint | Jev 版本 `typesafe/jev-1.14-20261014`，經 OpenRouter Decisions API 呼叫（封閉模型，非本地權重）；Markov 由本文自行估計 |
| License | 未披露（文本未見；代碼釋出條款待核） |
| 複現難度 | 中高：Markov／PostGIS／Kelly 全可複現，但 Jev 是封閉且會改版的模型，只能靠快取重放；GDELT 斷線週與 headline 排名會隨抓取時間漂移 |
| 數據可得性 | GDELT 公開（https://www.gdeltproject.org/）；NDX/QQQ/^IRX 來自 Yahoo Finance；Jev 查詢成本：373 次查詢共 USD 0.017（27 October 2026），罕見事件 189 次查詢 USD 0.007（15 November 2026） |

## §5 · 評測 / Benchmark

> **基線說明**：本文不存在「前 SOTA」——這是一個先前無人檢驗過的問題。表內「前SOTA」欄一律是**基線**：climatology（歷史基率，即時點可用頻率）或 raw（未校正）模型。逐字列出，不合成任何區間。

**A. 方向（四週後收低）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| NDX-100, 1771 週（1992-10 至 2026-09） | Brier（Markov raw vs climatology） | 0.233 | 0.232 | -0.001 |
| 同上 | ECE（Markov raw） | 0.011 | 0.031 | +0.020 |
| 同上 | AUC（Markov raw） | 0.470 | 0.496 | +0.026 |
| NDX-100, 88 週（2025-01-24 至 2026-09-25，基率 0.375） | Brier（Jev headlines） | 0.235 | 0.247 | +0.012 |
| 同上 | ECE（Jev headlines） | 0.010 | 0.101 | +0.091 |
| 同上 | AUC（Jev headlines） | 0.352 | 0.457 | +0.105 |
| 同上 | Brier（Markov raw） | 0.235 | 0.251 | +0.016 |
| 同上 | ECE（Markov raw） | 0.010 | 0.129 | +0.119 |
| NDX-100, 87 週 | Brier（Jev headlines，baseline = raw Jev） | 0.247 | 0.236 | -0.011 |
| 同上 | ECE（Jev + prior shift，baseline = raw Jev） | 0.101 | 0.015 | -0.086 |
| 同上 | AUC（Jev + prior shift，baseline = raw Jev） | 0.457 | 0.511 | +0.054 |

**B. 罕見事件（四週後跌逾 10%）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| NDX-100, 1694 週, H=4（基率 0.045） | Brier（Markov raw） | 0.0432 | 0.0392 | -0.0040 |
| 同上 | Log-loss（Markov raw） | 0.191 | 0.152 | -0.039 |
| 同上 | AUC（Markov raw） | 未披露 | 0.808 | |
| NDX-100, 87 週測試期（歷史基率 0.042，僅 2 個正例） | Brier（Jev headlines） | 0.023 | 0.071 | +0.048 |
| 同上 | Brier（Jev prices only） | 0.023 | 0.042 | +0.019 |
| 同上 | ECE（Jev headlines） | 未披露 | 0.22 | |
| 同上 | Brier（Jev headlines + prior shift） | 0.0228 | 0.0225 | -0.0003 |
| 同上 | Brier（Jev prices only + prior shift） | 0.0228 | 0.0226 | -0.0002 |

**C. 配置（half-Kelly, f=0.5, 5 bp per unit turnover；baseline = buy and hold QQQ）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| QQQ/T-bill, 1999-03-12 至 2026-10-16 | CAGR (%)（Markov raw） | 10.7 | 12.2 | +1.5 |
| 同上 | Sharpe（Markov raw） | 0.46 | 0.60 | +0.14 |
| 同上 | Max DD (%)（Markov raw） | -82.7 | -55.6 | +27.1 |
| 同上 | Sharpe（Markov + isotonic） | 0.46 | 0.49 | +0.03 |
| QQQ/T-bill, 2025-01-24 至 2026-10-16（僅 88 週） | CAGR (%)（Jev headlines raw） | 15.3 | 3.1 | -12.2 |
| 同上 | CAGR (%)（Jev headlines + prior shift） | 15.3 | 14.5 | -0.8 |
| 同上 | Sharpe（Jev headlines raw） | 0.61 | -1.26 | -1.87 |
| 同上 | Sharpe（Jev headlines + prior shift） | 0.61 | 0.58 | -0.03 |
| 同上 | Avg. w（Jev headlines raw → + prior shift） | 0.01 | 0.98 | +0.97 |

**解讀——哪個 Δ 是真 capability，哪個不是**

- **方向：全部在雜訊內，且沒有一個贏過 climatology。** 測試期 Jev 四個輸入的 DM p ≥ 0.54、AUC 0.45 到 0.56；Markov 在測試期甚至顯著**更差**（Brier 0.251 vs 0.235，DM p=0.049），原因是 2025 年 4 月崩盤後它的預測跳到 0.54，正好趕上反彈。全樣本 Markov 對方向的 AUC 0.496（正文四捨五入為 0.50），Brier 0.232 vs 0.233（DM p=0.81）——**這一整欄的 Δ 都不能讀成能力差異**。
- **罕見事件：唯一像真 capability 的一格。** Markov 對 >10% 下跌的 Brier 0.0392 vs 0.0432（DM p=0.045）、log-loss 0.152 vs 0.191（p=0.006）、AUC 0.808（摘要與正文作 0.81）。但這是作者自述「我們檢查過的多個 horizon 與事件定義之一」，有選擇性風險；且測試期只有 2 個正例，AUC 0.45 到 0.86 跨模型「carries no information」，此能力在 2025–2026 **無法被驗證**。
- **校準改善不能直接當成 alpha。** 原文報告 prior shift 前後 ECE 0.101 → 0.015、AUC 0.457 → 0.511。所讀文本沒有給出 AUC 變動的機制說明，本頁不替作者補上因果解釋；這些是原實驗的校準與區分能力結果，仍須與可交易績效分開。
- **配置 Δ 是最容易被誤讀的一欄。** 全期 Markov raw 的 Sharpe +0.14（95% CI [-0.07, +0.38]，bootstrap 落在 0 以下的抽樣比例 0.11）——**統計上未建立**。測試期 Jev prices only + prior shift 的 CAGR 16.3% 高於買進持有 15.3%，作者自己的圖註寫明「Only 88 weeks: the numbers illustrate the effect of calibration, they are not a test」，且四週窗重疊使有效獨立結果約 22 個；把 16.3% 當 alpha 就是典型的樣本內誤讀。
- **成本與泄漏**：5 bp per unit of turnover 已計（Markov raw 年周轉 1.85、Jev headlines + prior shift 2.07，屬不同樣本，不可互比），但部位在「產生預測的那個週五收盤」成交，作者自己列為限制。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（§Discussion and Limitations，逐條可定位）

1. **Jev 測試期太短**：88 週已知結果、四週窗重疊，作者自估「約 22 個獨立結果」；DM 與 block bootstrap 已處理重疊但 power 低，2025–2026 的配置結果「只是示意」。
2. **樣本無法往前延伸**：knowledge probe 顯示 Jev 至少知道到 2024-11，更早的週會引入 look-ahead，**評估與校正 map 的訓練皆然**。
3. **Jev 是封閉且會改版的模型**：全部結果對應 `typesafe/jev-1.14-20261014`，所有請求與回應已快取。
4. **新聞輸入受 GDELT 覆蓋限制**：大型通訊社與付費牆媒體代表性不足、關鍵詞主題含噪音（提及大型科技公司的娛樂新聞）、覆蓋缺口刪掉部分週。
5. **罕見事件結果來自多個 horizon 與事件定義中的一個**；地圖反映的是少數 bear episode 主導的新聞，不是因果。
6. **外部效度**：只涵蓋一個指數、一個問題、一個 decision model；部位在預測當下的週五收盤成交。

### 6.2 本頁推斷的隱含假設（明標推斷，不跨系統斷言）

- **（推斷）prior shift 假設偏移是純 prevalence shift、在 log-odds 上是常數。** 若 Jev 的偏移在高低波動 regime 不同號或不同量級（即 miscalibration 有斜率項），單一常數修不乾淨；本文未報交互項或分 regime 的 ECE，此項 TBD／圖表待核。
- **（推斷）ECE 的解析度不足。** 十等寬 bin 用在 88 週上，平均每 bin 樣本極少；ECE 0.009 到 0.015 這一級差距不宜當成排序依據，本文亦未給 ECE 的信賴區間。
- **（推斷）Kelly 的 μ̂/σ̂² 是兩點混合假設。** 它假設未來的四週報酬只透過 Y（跌/不跌）與 p 相關；若 regime 改變的是幅度而非方向，σ̂² 會系統性偏移。作者只做了「用一週矩取代四週矩」的穩健性（結論：CAGR 變動 ≤ 0.1 個百分點），未測混合分布的設定錯誤。
- **（作者自述 + 推斷）同一根 bar 成交。** 資訊截止設在週五 20:00 UTC、部位也在該收盤建立，離線回測可接受，實盤需處理收盤瞬時流動性與延遲；成本模型只有 5 bp 線性項，無冲击／借券／稅。
- **（推斷）指數層級的連續性假設。** 用 1985 起的 NDX 歷史做基率錨與 Kelly 矩，隱含「指數編製規則與成分演化不影響週報酬分布」；這與輔助數據缺失無關，不應與生存者偏差混談。
- **容量**：本文未披露 AUM 或容量上限；週頻 QQQ 的容量約束在絕大多數用途下不緊，但這不是本文驗證過的結論。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Markov-switching allocation（Hamilton 1989；Ang–Bekaert 2002；Guidolin–Timmermann 2007；Nystrup et al. 2015） | 只看組合績效，不檢驗 regime 機率本身的 calibration | 是 | 本文把它當基線，並指出其方向機率退化為基率 |
| LLM／新聞預測報酬（Tetlock 2007；Lopez-Lira & Tang 2023；Araci 2019） | 評方向或報酬，不評機率 calibration | 部分 | 本文補上 calibration 與 sizing 的經濟價值 |
| Post-hoc calibration（Platt 1999；Zadrozny & Elkan 2002；Guo et al. 2017） | 需要大量 paired outcomes 才能學 map | 是 | 本文在 26 到 87 對樣本下顯示比不做更差 |
| Prevalence shift 校正（Saerens et al. 2002；Iqbal et al. 2026） | 只調先驗、不學 map | 是 | 本文的 prior shift 是此線的市場版實證 |
| Decision model benchmark（Deußer et al. 2026；TypeSafe 2026） | 固定 base rate 的標註問答 | Jev 封閉 | 本文是第一個對已實現市場結果的檢驗 |
| 地緣風險特徵（Caldara & Iacoviello 2022；Leetaru & Schrodt 2013 + PostGIS） | 國別／區域指數，多用於解釋而非配置 | 是 | 本文：區域 conflict share 只對 >10% 下跌有邊際貢獻（log-loss DM p=0.025），方向僅把基線缺口補回 climatology 水準（DM p=0.73） |

🎤 **Interview Tip**

> **問**：「一個宣稱 calibrated 的 decision model，在 benchmark 上 ECE 很低。你直接拿它的機率去做 Kelly sizing，會有什麼問題？」
>
> **正確答**：calibration 是**相對事件基率**的性質，不是模型的內稟屬性。benchmark 的 base rate 穩定且接近平衡，市場事件的 prior 是 0.375、罕見事件的 prior 是 0.042；prevalence 一移，輸出就帶固定偏移（本文 raw Jev 平均 0.48 對上實際 0.38），Kelly 會系統性縮倉——2025–2026 平均只握 0.01 的 QQQ，年報酬 3.1% vs 買進持有 15.3%。修法優先序是「先錨基率」（一參數 prior shift，不需測試期結果），而不是 Platt／isotonic——後者要學 map，在 26 到 87 對樣本上反而把 Brier 推到 0.246 到 0.280。同時要說清楚：ECE 下降**不等於**經濟價值上升，isotonic 的 ECE 0.014 比 raw Markov 的 0.031 好，Sharpe 卻是 0.49 vs 0.60。
>
> **錯答**：「ECE 越低配置越好」／「用 Platt scaling 一定修得回來」／「LLM 讀新聞 headline 就有 alpha」／「Markov 的 Sharpe +0.14 已經證明有效」（CI [-0.07, +0.38] 跨零）／「測試期 Jev prices only 校正後 16.3% 打敗買進持有，所以有 alpha」（88 週、重疊窗、約 22 個獨立結果，作者自述非檢驗）。

### 7.1 可證偽預測（本頁推斷，非原文結論；門檻未定者標 TBD）

- **P1**：若作者提出的 pre-registered live test 在 2026-10-23 之後啟動並累積到覆蓋多個獨立的「>10% 下跌」episode，則 raw Jev 在該事件上的 Brier 仍將顯著劣於 climatology，而 prior shift 後差距消失但 AUC 不會顯著偏離 0.50——因為偏移是水平位移，不含排序資訊。判定樣本量 TBD。
- **P2**：若把 prior shift 的錨由 1990 起全歷史改成短窗 rolling 基率，方向事件的 ECE 會上升，但配置績效不一定下降——即「錨的穩定性」比「錨的即時性」重要。檢驗窗口 TBD。
- **P3**：Table 3 的 Markov >10% 下跌優勢（AUC 0.808）若在新 episode 上掉回 0.50 附近，則應被重新歸因為 1998–2003 與 2008–2009 兩個 episode 的槓桿，而非可持續的 regime 偵測能力。判定日：新 episode 出現後 TBD。

## §8 · For the Reader

- **因子研究員**：把這篇當「calibration 不是 alpha」的乾淨反例讀。你要複製的不是 prior shift（它只是一行 logit 平移），而是它的檢驗協議：knowledge probe 定邊界 → 只用邊界後的樣本 → 基線用 climatology 而非隨便一個模型 → DM test 加 block bootstrap 處理重疊。任何自建 LLM 因子回測都該先跑這個流程。
- **高頻／執行**：本文對你幾乎沒有直接可用項——週頻、5 bp 線性成本、同一根收盤成交、無冲击模型。可取的是它的成本敏感度誠實度：turnover 1.85 與 2.07 的兩條策略在 5 bp 下仍能拉開 CAGR 差距，說明這條策略線對成本不敏感，也就意味著它**不是**容量受限的 alpha。
- **組合配置**：核心可搬移的是「先驗錨定 + fractional Kelly」這個兩步式流程，以及「用模型偵測風險 episode、不要用模型猜方向」的配置哲學。但請注意 Markov raw 的 Sharpe 增益 CI 跨零，實務上應把它當尾部風險覆蓋（Max DD -55.6 vs -82.7），而不是報酬增強。
- **LLM-agent 開發者**：Jev 的失效模式值得直接抄進 system prompt 設計——decision model 對「基準率未知的 yes/no 問題」會給出接近五五波的答案，且越罕見的事件偏得越兇（>10% 下跌：歷史基率 0.042，raw Jev 0.16 到 0.24）。正確做法是把外部基率當先驗注入，而不是期待模型自己校準。
- **RL 策略研究者**：把「reward 是報酬、constraint 是 calibration」拆開看。本文示範了 calibration 與經濟目標不同構的實證（isotonic 的 ECE 更低、Sharpe 更差），對 reward shaping 的啟示是：別把 proper scoring rule 直接當 utility。
- **研究學生**：這是「負面結果寫成好論文」的範本——地理特徵、headlines、模型集成都失敗，作者照樣發表，而且把 limitation 寫成六條。要學的是它的樣本切分紀律（t-H 之前才可用）與「88 週 ≈ 22 個獨立結果」這種把有效樣本量講清楚的習慣。

## References

**原論文**
- Zając, J. and Dyrek, M. (2026). *Are Calibrated Decision Models Calibrated on Markets? Geolocated News, Markov Switching and a System One Model for Nasdaq-100 Allocation.* arXiv:2611.26716v1 [q-fin.ST], 19 Nov 2026. https://arxiv.org/abs/2611.26716

**Lineage（依原文引用）**
- Hamilton (1989) *Econometrica* 57(2):357–384 — two-regime Markov switching。
- Diebold, Lee & Weinbach (1994)；Filardo (1994) — time-varying transition probabilities（本文 TVTP 的來源）。
- Ang & Bekaert (2002)；Guidolin & Timmermann (2007)；Nystrup et al. (2015) — regime-based allocation。
- Saerens, Latinne & Decaestecker (2002) — 先驗機率調整，prior shift 的理論祖先。
- Platt (1999)；Zadrozny & Elkan (2002)；Guo et al. (2017) — post-hoc calibration 與其失效。
- Iqbal, Nasim & Saleem (2026) arXiv:2610.26325 — prevalence shift 下的 typed decision models（本文假說的直接前作）。
- Deußer et al. (2026) arXiv:2609.37647；TypeSafe (2026) OpenRouter Jev docs — decision model 的 benchmark 與本體。
- Kelly (1956)；Thorp (2006) — fractional Kelly。
- Brier (1950)；Gneiting & Raftery (2007)；Diebold & Mariano (1995)；Newey & West (1987)；Politis & Romano (1992) — 評分規則與推論。
- Leetaru & Schrodt (2013) GDELT；Caldara & Iacoviello (2022) GPR — 地理事件資料與地緣風險。
- Tetlock (2007)；Araci (2019)；Lopez-Lira & Tang (2023) — 新聞／文本與報酬預測。

**來源鏈接**
- GDELT Project：https://www.gdeltproject.org/
- OpenRouter Jev 文件：https://openrouter.ai/docs/guides/community/jev

_未核項目：Figure 1（reliability diagram）、Figure 2（bear regime 機率時序）、Figure 3（測試期預測對照）、Figure 4（conflict 事件地圖）之圖像內容；Table 1 與 Table 3 之完整欄位對齊；代碼釋出條款與 repo 位址。上述皆為「所讀文本未見／圖表待核」，不構成論文未披露之斷言。_