<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=全自动黑盒 -->

# Rare replay / SMSCP 解構（Rare replay / SMSCP）

> **來源發布**：2026-11-03 · （無 venue） · arXiv [2611.04223](https://arxiv.org/abs/2611.04223)
> **arXiv 原文**：[Aggregate Calibration Can Mask Stress-State Undercoverage: Bounded-Memory Forecast Intervals under Recurrent Financial Stress](https://arxiv.org/abs/2611.04223v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 日頻 × 監督回歸 × 風險擇時 × 全自動」——但這裡的「擇時」是**區間寬度的狀態擇時**，不是部位擇時。它補的是 conformal 文獻的一個縫隙：weighted／localized／Mondrian 系列都在處理「已被保留的殘差怎麼用」，本文則在硬記憶預算下隔離**content-blind 驅逐讓哪些證據根本不存在**。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①在 HAR 點預測、conformity score、ACI 控制器與記憶預算 B=250 全部固定的條件下，只改「每次查詢可用的歷史殘差集」；②trick 是把預算切成近期佇列與按 pre-target VIX 狀態分庫的 reservoir，查詢時只開當前狀態那一庫（inverse-frequency 分配，η=−1）；③它擊中的是「聚合 coverage 合格即安全」這條運維假設——原文回報 FIFO 整體 coverage 0.9003（ETF11）／0.9002（Stock50），但 high-VIX 日分別只有 0.7576／0.7763；④Rare replay 在 high-VIX interval score 上為 0.0736／0.0925，優於 FIFO 的 0.0872／0.1058，代價是 high-VIX width 由 0.0261／0.0367 升至 0.0345／0.0473。

**X-Ray.** 這篇的正確讀法是「不確定性資訊設計」而非 alpha 論文：所讀文本全篇未見 Sharpe／AR／MDD／換手／成本數字，收益軸（風險擇時）之所以掛得上，只是因為輸出的區間寬度在壓力態被放寬、可能間接影響 downstream sizing，而作者並未驗證這條傳導。它真正殺掉的舊工程坑是**用聚合 coverage 做監控**：同一套在 2012–2025、每資產 3,520 個評估日上看起來貼近名義 0.90 的校準器，在 156 個 high-VIX 日上覆蓋率明顯掉下去，而這個落差被 calm dates 的樣本占比吸收。envelope 也很清楚：state 必須**事前可觀測且離散**（VIX_{t−1} 切 20/30，門檻評估前指定），13 個 stress entry 中僅 three 個滿足嚴格 FIFO-eviction，且 storage 與 retrieval 綁在一起改，所以「狀態分庫」與「只查當前庫」的因果貢獻未被識別（作者自述）。對量化讀者的意義：這是把「壓力期可靠性」從口號變成可審計 KPI 的一條低成本路徑；但 interval score 的改善是區間品質單位，不能兌換成收益，且須付更寬區間的代價——這一步本文沒做。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 對照：recent-only FIFO／weighted conformal | 本文：Rare replay | 隔離了什麼 |
|---|---|---|---|
| 記憶體切分 | 單一 FIFO 緩衝，B=250 全部按年齡 | B_rec=⌊B/2⌋ 近期佇列 + B_rep 歷史分庫；B=250 時近期 125 槽、(L,M,H) 分庫容量 (46,31,48) | 同一硬預算下的**組成**，不是容量 |
| 驅逐規則 | content-blind，最舊先出 | 被逐出殘差路由到其**儲存時** pre-target VIX 狀態分庫（各庫用 standard reservoir sampling） | 「留下什麼」 |
| 檢索 | 全部 stored 皆 active | 近期佇列 + **只**激活當前狀態那一庫 | 「拿什麼來算 cutoff」 |
| 不變量 | — | HAR 點預測、conformity score（絕對誤差）、ACI 更新、B、驗證期、評估日曆全固定 | 排除模型／控制器解釋 |

分庫容量不是手調：$\widetilde{B}_g(\eta)=B_{\mathrm{rep}}\hat p_g^{\eta}/\sum_h \hat p_h^{\eta}$，η=−1（inverse-frequency）在評估前選定，再以 largest-remainder 取整；$\hat p_g$ 由 ETF11 驗證期估計後**原封不動**套到 Stock50。附錄另重建了 Mondrian proportional（同總預算、按狀態分儲、查詢只用當前狀態緩衝、控制器 state-specific）作為同軸對手。

**釐清命名**：SMSCP（Han and Lin, 2026b, arXiv:2610.19484）在本文是**被引用的同作者前作／lineage**，不是本文提出的方法；本文提出並檢定的是 Rare replay（structured replay）這一組記憶建構。

### 1.2 ⚡ Eureka

> **availability ≠ use。** 在綁死的記憶預算下，先有「哪些殘差還活著」，才輪到「怎麼加權」；weighted／localized conformal 再聰明，也只能對倖存者加權。

直覺：高波動是**會回來的**（作者原文：a high-volatility state may return after a long calm period），而 recency 不是相關性的單調代理。FIFO 的問題不是它加權錯了，而是壓力態回來的那一天，它的緩衝裡可能已經沒有任何一次上次壓力態的誤差——作者稱之為 **FIFO forgetting**，並把「聚合 coverage 掩蓋狀態欠覆蓋」做成可檢定的量：high 減 non-high 覆蓋差，在同一評估日曆上配對計算。

### 1.3 信息流

```
Y_{t-1}, mean5, mean22 ──HAR (OLS, ≤756 obs, [·]_+)──► Ŷ_t
                                                          │
VIX_{t-1} ──► G_t ∈ {L,M,H} ──────────┐                   ▼
                                      │      q̂_t = inf{q : Σ a·1{R≤q} ≥ (1−α_t)(W+1)}
  recent queue (125 slots) ───────────┼──► A_t ──► C_t = [max(0,Ŷ−q̂), Ŷ+q̂]
  bank[G_t] (46 / 31 / 48) ───────────┘                   │
                                                          ▼
      Y_t 觀測 → V=1{Y_t ∉ C_t} → α_{t+1} = Π_[0.001,0.30][α_t + 0.005{0.10 − V}]
                                                          │
        R_t = |Y_t − Ŷ_t| → 進 recent queue；被逐出者 → bank[G_s]（依儲存時狀態）
```

順序固定：出區間 → 觀測 → 更新控制器 → 插入殘差。評估起點 α 對每個設計與隨機種子一律重置為 0.10，之後各路徑分歧。

## §2 · 數學層

📌 **Napkin Formula**

$$\hat q_{i,t}=\inf\Big\{q:\textstyle\sum_{s\in\mathcal A_{i,t}} a_{i,s,t}\,\mathbf 1\{R_{i,s}\le q\}\ \ge\ (1-\alpha_{i,t})(W_{i,t}+1)\Big\},\quad W_{i,t}=\textstyle\sum_{s\in\mathcal A_{i,t}}a_{i,s,t}$$

權重全 1 時退化為 active set 上的有限樣本分位數；若所需質量超過 W，取最大 active score。控制器：$\alpha_{i,t+1}=\Pi_{[0.001,0.30]}[\alpha_{i,t}+0.005\{0.10-V_{i,t}\}]$。

**複雜度**：每次查詢對 active set 取分位數，n = 125 + 當前庫容量（B=250 下最大 48）；儲存 O(B) 常數級；無梯度、無訓練，只有每日一次 HAR 的 OLS（至多 756 筆 origin–response 配對）。

**直覺**：cutoff 只看「活著的誤差有多肥」，不看它們多老；本文把「活著」這件事從時間函數改成狀態函數。**Loss**：沒有訓練損失——conformity score 是絕對誤差，品質由 interval score（Gneiting–Raftery，名義 miscoverage 0.10，越低越好）事後評估，寬度與未覆蓋同時受罰。

## §2.5 · 帶數字走一遍（示意，非論文實證）

**以下整段為假設的玩具數字，只示範 cutoff 與控制器的算法；論文真實數字一律在 §5，缺的寫「未披露」。**

1. 假設某日 α_t = 0.10，active set 有 9 筆殘差、權重皆 1（W=9），R 為 0.01 至 0.09 的等距序列。
2. 要求質量 ≥ (1−0.10)×(9+1)，取到第 9 小的殘差，故 q̂ = 0.09。
3. 若 Ŷ_t = 0.05，區間 = [max(0, 0.05−0.09), 0.05+0.09] = [0, 0.14]。
4. 若實際 Y_t = 0.20 → 未覆蓋，V=1 → α_{t+1} = 0.10 + 0.005×(0.10 − 1) = 0.0955（落在 [0.001, 0.30] 內，不截斷）。
5. 殘差 R_t = |0.20 − 0.05| = 0.15 進近期佇列；若已滿則最舊者被逐出，並依它**當時**的狀態標籤放進對應分庫。
6. 分配示意：設 B=8 → B_rec=4、B_rep=4；驗證期 p_L=0.75、p_M=0.25，η=−1 給出未取整容量 1 與 3 → (1, 3)。壓力態分到三倍槽位——這一步就是「Rare」的全部內容：它不改權重，只改**誰還在**。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 資產池 | ETF11：SPY, QQQ, IWM, XLK, XLF, XLE, XLV, XLY, XLP, TLT, GLD（共 11 檔 ETF）；Stock50：50 檔大型美股（附錄 A 逐字列出） |
| 頻率／標的 | 日頻 OHLC，$Y_{i,t}=\log(H_{i,t}/L_{i,t})$（隔日 high–low log range） |
| 原始區間 | 2005–2025；2005–2008 warm-up、2009–2011 驗證、January 3, 2012 through December 31, 2025 評估 |
| 評估規模 | 每資產 3,520 個評估日；評估日曆含 156 個 high-VIX 日；狀態日數 2,634 L / 730 M / 156 H |
| 狀態定義 | VIX 在自己的交易日曆上平移 1 期後併入資產目標日；G=L 若 VIX_{t−1}<20、M 若 20≤·<30、H 若 ≥30，門檻評估前指定 |
| 來源／版本 | Yahoo Finance（automatic price adjustment disabled）＋ Cboe VIX；ETF11 與 VIX 下載於 May 11, 2026、Stock50 於 June 12, 2026 |
| 樣本外紀律 | Tuning choices are based on the designated validation period rather than evaluation outcomes；η、半衰期、γ 皆在 2009–2011 選定後套用 Stock50；缺列直接刪除、不前填 |
| 容量說明 | Stock50 為**事前指定**、非機率抽樣、**非 survivorship-free**，作者明言不以代表美股市場為目的；兩面板共用同一市場日曆，the 61 securities 不當作獨立時間複現 |

## §4 · 代碼層

| 項目 | 值 |
|---|---|
| Repo | 未公開；原文 Data Availability：「available from the corresponding author on reasonable request」 |
| Checkpoint | 不適用（HAR 為每日 OLS 重估，無神經網路權重） |
| License | 論文 CC BY 4.0；程式碼授權未披露 |
| 複現難度 | 中：需逐日 OHLC + 平移後 VIX、20 個 prespecified reservoir seeds、5,000 次 paired circular block-bootstrap（主 block=20 交易日，另有 10 與 40 日敏感度） |
| 數據可得性 | Yahoo Finance／Cboe，受供應商條款約束；資料版本隨下載日漂移，原值不可永久對齊 |
| 超參可復現性 | 部分：TWC 選 h_{1/2}=62.5、RWC-adapted 選 (h_{1/2},γ)=(62.5,1)；η=−1 與 (46,31,48) 分庫容量已給，但 allocation 的 largest-remainder 細節、ACI 起點以外的路徑 TBD |

## §5 · 評測 / Benchmark

> 模板欄名為「前SOTA」，但原文**沒有任何 SOTA 宣稱**：主對照是 recent-only FIFO 基線。下表前SOTA 欄逐字放 FIFO 原值，**不合成任何 SOTA 區間**。指標為 high-VIX 日的 coverage／width／interval score（IS，越低越好），樣本為 156 個 high-VIX 評估日、B=250。

| 數據集/市場 | Metric | 前SOTA（recent-only FIFO 基線） | 本方法（Rare replay） | Δ |
|---|---|---|---|---|
| ETF11 | high-VIX coverage | 0.7576 | 0.8478 | 0.0902 |
| ETF11 | high-VIX interval score（越低越好） | 0.0872 | 0.0736 | -0.0136 |
| ETF11 | high-VIX width | 0.0261 | 0.0345 | 0.0084 |
| Stock50 | high-VIX coverage | 0.7763 | 0.8619 | 0.0856 |
| Stock50 | high-VIX interval score（越低越好） | 0.1058 | 0.0925 | —（見註） |
| Stock50 | high-VIX width | 0.0367 | 0.0473 | 0.0106 |

註：Stock50 的 IS 差，原文 Table 3 的配對估計逐字為 -0.0132（95% CI [-0.0286, -0.0009]）；以 Table 2 顯示精度直接相減與該配對值不完全一致，故本頁不填 Δ，只保留兩個來源原值。ETF11 對應的配對估計逐字為 -0.0136（95% CI [-0.0275, -0.0018]），與上表 Δ 同值。

**同條件下的其餘來源原值（逐字照抄 Table 2，非 Δ）**

| Design（B=250） | ETF11 high-VIX IS | Stock50 high-VIX IS | 診斷角色 |
|---|---|---|---|
| FIFO | 0.0872 | 0.1058 | recent-only 參考 |
| Recent-*n*（active size 對齊） | 0.0839 | 0.1028 | 是否只是 active set 變大 |
| Hybrid-*n*（regime-agnostic reservoir，active size 對齊） | 0.0855 | 0.1033 | 是否只是能碰到更老的殘差 |
| Rare replay | 0.0736 | 0.0925 | 狀態結構化保留＋檢索 |
| unbounded current-state reference | 0.0688 | 0.0878 | 資訊上限（非 bounded 競品） |

**FIFO 的聚合掩蓋（§5.1 逐字）**：整體 coverage 0.9003／0.9002，非高波動日 0.9070／0.9059，高波動日 0.7576／0.7763，落差 -0.1494／-0.1297；配對 block-bootstrap 區間 [-0.2778, -0.0376]／[-0.2440, -0.0318]（20-day block，10 與 40 日敏感度見附錄 C，區間皆不含零）。附錄 G 顯示各設計整體 coverage 落在 0.9001 至 0.9011（ETF11）與 0.8987 至 0.9008（Stock50）——**把所有設計的聚合指標壓成同一個數字，正是本文要攻擊的評估方式**。

**哪些是真 capability、哪些不是**
- Recent-*n* 與 Hybrid-*n* 都精確對齊 Rare 的 realized active size，卻都沒重現 Rare 的 stress-period IS（原文 Table 3 的配對估計 -0.0102／-0.0103 與 -0.0119／-0.0108，區間皆不含零）→ 「只是樣本變多」與「只是能碰到更老的殘差」兩個解釋被排除。這是本文最乾淨的一段。
- 但 unbounded 參照仍低於 Rare（0.0688 vs 0.0736；0.0878 vs 0.0925），且 unbounded−FIFO 的配對區間在 ETF11 排除零、**Stock50 不排除**（-0.0180 [-0.0420, 0.0021]）→ 狀態分庫是「有限預算下的折衷」，不是資訊上限。
- 代價明確且方向一致：Rare 的 high-VIX 區間**更寬**（0.0345 vs 0.0261；0.0473 vs 0.0367）。這是 coverage–width 交換，不是免費午餐；若下游只想要窄區間，這個交換未必划算。
- 過擬合與未計項：η=−1、分庫容量、半衰期、γ 皆在 ETF11 驗證期選定再套 Stock50，驗證期與評估期的落差診斷 `TBD`（所讀文本未見）。全篇無交易成本、無部位、無 PnL，所以不存在成本未計的問題，但也**不能**把 IS 改善讀成收益改善。
- 反向證據必須一起讀：Mondrian proportional 相對 FIFO 的點估計較低但 20-day 區間含零；Rare 對 Mondrian 在兩面板**變號**且所有 block-bootstrap 區間含零；附錄 F 以 SPY 22 日 realized-vol 重定義壓力（56 個評估壓力日，其中 43 個與 VIX-high 重疊）時，Rare−FIFO 的區間**跨零**。也就是說「狀態結構有效」在點估計上一致，在推論上只對 VIX-high 子群穩。
- 加權分支不可與保留策略混談：附錄 I 的 RWC-adapted−Rare 在兩面板區間皆含零，且 RWC-adapted 相對 RWC-MM 的對比同時改變總校準質量，原文自述「does not isolate relative state weighting alone」。原文結論是「These results do not identify a uniformly superior state-aware implementation.」

**壓力再進入（§5.3／Table 4 逐字）**：13 次 stress entry、12 次 recurrent、three 次滿足嚴格 FIFO-eviction；recurrent entry 當下 high-state 殘差占 FIFO active set 9.57%、占 Rare 34.44%，歷史回放分量有 48 筆 active、年齡中位數 approximately 2,914 個 prediction origins。Rare−FIFO 的 IS 差：recurrent entries -0.0167／-0.0199；first five high-state dates -0.0232／-0.0202；later high-state dates -0.0052／-0.0081；strict FIFO eviction -0.0386／-0.0505（僅 three dates，作者標為 descriptive）。分 episode 看，Rare 在 10/13（ETF11）與 9/13（Stock50）episode 勝出 → 作者自己把它降級為 timing diagnostic。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- §6：結論條件於「HAR-based range forecasts、兩個固定的美股面板、此處使用的可觀測狀態定義」；Stock50 是橫截面轉移分析，**不是**獨立時間複現（兩面板共用同一市場日曆）。
- §4.1：Stock50 非機率抽樣、非 survivorship-free，不以代表美股市場為目的。
- §6／§5.3：13 個 stress episode 太少，不足以做 episode 級推論（"too few for episode-level inference"）。
- §5.2／§6：replay **同時**改儲存與檢索，兩者因果貢獻未被識別；future work 明列「designs that vary storage and retrieval separately」。
- Appendix I：Mondrian 點估計低於 FIFO 但 20-day 區間含零；Rare vs Mondrian 跨面板變號；結論是不識別一致較優的 state-aware 實作。
- Appendix F：以 SPY realized-vol 重定義壓力時，壓力日僅 56 天、與 VIX-high 大量重疊，區間跨零 → 只算敏感性，不是獨立複現。
- Appendix J：B∈{100,250,500,1000} 對結構化設計**非單調**，容量本身不保證改善。
- AI Use Statement：使用 OpenAI ChatGPT 於語言潤飾與編碼協助，作者負全責。
- 運算成本／延遲分析：所讀文本未見，圖表／附件待核。

### 6.2 本頁推斷的隱含假設（標明推斷）

1. **狀態可事前觀測且離散**（推斷）：整套設計靠 VIX_{t−1} 的 20/30 切點；若狀態只能事後標記、或需要 learned representation（作者列入 future work），這條路不成立。門檻評估前指定降低了前視風險，但主分析只有一組切點；附錄 J 的 18/28 只測 ETF11，且 Rare−proportional 為 -0.00011（95% CI [-0.00085, 0.00051]）。
2. **regime 結構在評估窗內穩定**（推斷）：分庫容量由 2009–2011 頻率決定後不再更新；若壓力態相對頻率漂移，槽位配置會失配，本文未測此漂移。
3. **回放殘差與現時殘差可互換**（推斷）：主分析 cutoff 對 active set 一視同仁（權重全 1），等於假設 2,914 origins 前的壓力誤差與今日壓力誤差同分布。附錄 E 的 Oldest-*n* 對照（Rare−Oldest-*n* 為 -0.00956／-0.00882，區間皆不含零）只排除「年齡本身」的解釋，未排除分布差異。
4. **無成本／無容量約束**（事實＋推斷）：因不產生物位，交易成本不進場；但也因此沒有任何證據說這套區間在組合層面能提高風險調整報酬（所讀文本未見）。
5. **缺列機制**（推斷）：缺列直接刪除不前填，對 high-VIX 日可能非隨機缺失；作者未報告被刪列的狀態分布，圖表／附件待核。
6. **單一狀態軸足夠**（推斷）：狀態只由 VIX 三檔決定，未測「同 VIX 不同結構」（信用、流動性）的細分；非美股市場未測。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| FIFO（recent-only conformal，Xu–Xie／Gibbs–Candès 系） | 驅逐規則：年齡 vs 狀態 | 文獻方法；本文未綁定其程式碼 | 主基線；聚合 coverage 貼近 0.90，high-VIX 0.7576／0.7763 |
| Hybrid-*n*（regime-agnostic reservoir，ResCP 系） | 有無狀態結構 | 文獻方法 | active size 對齊後仍未達 Rare 水準（0.0855／0.1033 vs 0.0736／0.0925） |
| Mondrian / proportional 重建 | 分類內校準 vs 狀態分庫＋近期佇列 | 文獻方法 | 點估計低於 FIFO 但 20-day 區間含零；與 Rare 互有勝負且變號 |
| TWC / RWC（時間衰減＋regime 相似度重加權，Schmitt 2026 系） | 對倖存者加權 vs 讓證據存活 | 文獻方法 | 附錄 G：RWC-adapted high-VIX IS 0.0636／0.0851，但同時改變總校準質量，未隔離單一因子 |
| SMSCP（Han & Lin, 2026b） | 目標特定狀態相似度參考集 | 前作／lineage | 本文的鄰近方法，非本文提案 |
| unbounded current-state reference | 有無預算上限 | 非 bounded 競品 | 資訊上限參照；ETF11 區間排除零、Stock50 不排除 |

🎤 **Interview Tip**
- **正確答**：「這是 calibration information design 的隔離實驗：HAR、conformity score、ACI、B=250 全固定，只換 active residual set。它證明兩件事——聚合 coverage 0.9003 可以和 high-VIX 0.7576 並存；active-size 對齊（Recent-*n*）與 regime-agnostic 歷史訪問（Hybrid-*n*）都解釋不了 Rare 的 stress-period IS 表現（0.0872／0.1058 對 0.0736／0.0925）。但它同時承認 storage 與 retrieval 綁在一起改、13 個 episode 不足、替代壓力定義下區間跨零、Mondrian 與 Rare 互有勝負。」
- **錯答**：「它提出新的波動率模型／新的 conformal 覆蓋保證」——沒有，點預測是教科書 HAR，保證仍是既有 finite-sample cutoff；或「它證明 regime memory 一定有效」——作者在 Appendix I 明言未識別出一致較優的 state-aware 實作；或把它讀成交易策略——全篇無 PnL。

### 7.1 可證偽預測（本頁假說，尚待驗證）

1. **前提**：B 仍是 binding、評估窗仍含 recurrent stress。**預測**：若做 storage／retrieval 分離的 ablation（固定分庫、檢索改成按年齡，或反之），Rare 相對該對照的 high-VIX IS 優勢會**小於** Rare−FIFO 的配對估計 -0.0136（ETF11）。**檢驗時點**：TBD。
2. **前提**：換到獨立時間窗或非美股市場重跑，狀態仍用事前可觀測的離散切點。**預測**：Rare 對 Mondrian proportional 的 high-VIX IS 差仍不顯著（區間含零），與 Appendix I 一致；若顯著，則「狀態分庫＋近期佇列」的組合確有超出單純按狀態分儲的增益。**檢驗時點**：TBD。
3. **前提**：把 B 放大到壓力態殘差不再被驅逐。**預測**：FIFO 與 Rare 的 high-VIX 指標差會收斂——這是「availability 才是機制」的直接推論；原文只測到 B∈{100,250,500,1000} 且對結構化設計非單調，未測不 binding 區。**檢驗時點**：TBD。

## §8 · For the Reader

- **因子研究員**：把「狀態條件覆蓋率」加進模型監控面板，而不是只看聚合 coverage。本文的最小成本做法是：固定預測器與校準器，只換殘差保留策略（近期佇列＋狀態分庫），然後報 high 減 non-high 的覆蓋差。注意 interval score 是區間品質單位，不是 alpha；也注意 Rare 的區間在壓力態更寬。
- **高頻執行／風控**：這是哨兵不是訊號。壓力態區間變寬（0.0261→0.0345／0.0367→0.0473，逐字來源值）會直接影響 downstream 的 sizing／限額；先確認 sizing 規則對寬度是線性還是門檻式，再決定要不要付這個成本。文中沒有任何延遲或吞吐數字（未披露）。
- **組合配置**：跨資產共用日曆、the 61 securities 不當獨立時間複現——這個統計處理值得抄。但 Stock50 非 survivorship-free，且與 ETF11 共用同一市場日曆，橫截面結論要打折。
- **LLM-agent／continual learning**：把「FIFO forgetting」搬到 agent memory 是同一結構問題：content-blind eviction 讓久未出現但會復現的狀態失去證據。本文給的是可量化的診斷（狀態條件覆蓋差），不是記憶架構的通用解；作者也只主張在 observable recurrent states ＋ binding memory 的條件下成立。
- **RL 策略**：replay buffer 的優先級取樣文獻早有此爭論，本文的貢獻是把「保留」與「加權」分開談——你的 prioritized replay 若同時改取樣與 loss 權重，因果同樣不可識別（本文自己就栽在這點，作者明列為 limitation）。
- **研究學生**：這是「控制變量做減法」的範本——同一骨幹、同一預算、只換資訊集；同時也是「負面結果照登」的範本（Mondrian 區間含零、Rare vs Mondrian 變號、替代狀態定義跨零、B 掃描非單調）。想入門 conformal＋時間序列，讀它的 §1 文獻分層比讀公式更有用。

## References

- 原論文：Han, R., Lin, L. *Aggregate Calibration Can Mask Stress-State Undercoverage: Bounded-Memory Forecast Intervals under Recurrent Financial Stress.* arXiv:2611.04223v1 [stat.ME], 03 Nov 2026. https://arxiv.org/abs/2611.04223v1
- 方法 lineage：HAR（Corsi, 2009）· range-based volatility（Parkinson, 1980；Chou, 2005）· split/adaptive conformal（Vovk et al., 2005；Lei et al., 2018；Gibbs & Candès, 2021, 2024；Xu & Xie, 2021, 2023；Zaffran et al., 2022；Bhatnagar et al., 2023；Angelopoulos et al., 2023）· weighted/localized conformal（Tibshirani et al., 2019；Barber et al., 2023；Guan, 2023；Hore & Barber, 2025；Auer et al., 2023；Lee et al., 2025）· Mondrian（Boström et al., 2021）· reservoir sampling（Vitter, 1985）· replay（Rolnick et al., 2019；Aljundi et al., 2019）· 鄰近競品（Schmitt, 2026；Neglia et al., 2026；Gao & Liu, 2025；Shihab et al., 2026；Tumu et al., 2026）· 同作者前作 SMSCP（Han & Lin, 2026b, arXiv:2610.19484）與 StaLoP（Han & Lin, 2026a, arXiv:2607.00128）
- 評估準則：interval score（Gneiting & Raftery, 2007）；conditional calibration 分解（Allen et al., 2026）；rolling/expanding window 與波動預測視窗選擇（Feng et al., 2024；Feng & Zhang, 2025）
- 來源鏈接：arXiv 摘要頁 https://arxiv.org/abs/2611.04223 ｜ 全文（本頁依據）https://arxiv.org/abs/2611.04223v1