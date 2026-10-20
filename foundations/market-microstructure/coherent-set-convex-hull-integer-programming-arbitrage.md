<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=组合执行优化 autonomy=全自动黑盒 -->

# Coherent-set convex hull + integer programming arbitrage 解構（Coherent-set convex hull + integer programming arbitrage）

> **來源發布**：2026-10-16 · （無 venue） · arXiv [2610.23517](https://arxiv.org/abs/2610.23517)
> **arXiv 原文**：[Combinatorial Arbitrage in Prediction Markets](https://arxiv.org/abs/2610.23517v1)　·　_本頁由 arXiv 原文一手自主解構_
> **核心定位**：五軸落點在「微觀盤口 × 高頻日內 × 因果結構 × 組合執行優化 × 全自動黑盒」的交角。它補的 prior gap 不是「預測市場有沒有套利」，而是把 arbitrage existence（凸包幾何）與 arbitrage size（全深度盤口逐檔）縫成同一條可回放 pipeline，並把 relation discovery 的邊際成本當成解釋 alpha 衰減的一階變量。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把邏輯相關合約的 admissible payoff 向量取 convex hull，證明報價落在 hull 外即存套利、每單位最大保證利潤等於 ℓ1 距離，再用線性約束系統 + integer programming 在多關係聯合下求最優組合（避開 payoff 向量列舉爆炸）。關係用 venue metadata、match/type/line 欄位與 LLM 讀 settlement text 三層抽取。★ 對 alpha 軸的意義：套利半衰期由「關係被發現的難度」而非僅由資本競爭決定。實證上，已結算組合扣 spread 與 fee 後保證利潤為 committed capital 的 2.2%，到期利潤 3.2%。

**X-Ray.** 這篇的殺傷力不在 LLM，而在它把「關係」從資料前處理降級為 alpha 的一階變量。舊工程坑有三個：一是 pairwise 等價套利（Rothschild & Pennock 2014、Saguillo et al. 2025 路線）無法處理 mutually exclusive / nesting / partition / conjunction 的交互，二是有關係卻沒有規模估計，三是市場 maker 側的 coherence 求解（Dudík et al. 2012、Kroer et al. 2016）預設有人維持價格一致，而真實盤口是分裂的。本文的取捨很明確：放棄 cross-venue、放棄 maker 側、放棄 short 端的一般化，換取「taker 在單一 venue 全深度 L2 上到底能吃多少」這個可回放的答案。它打不開的 envelope 也清楚：只覆蓋 Polymarket、利潤以 displayed depth 為上限、LLM 抽取的 recall/FP 成本未在摘要層揭露、rebate 與做市激勵未見。對量化讀者，真正的可移植物是「把邏輯約束寫成 IP 的 constraint 集」這個模板，而非 2.2% 這個數字。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（逐篇列，不合成） | 本文 |
|---|---|---|
| 關係類型 | Saguillo et al. (2025)：以語言模型識別市場間依賴，測「市場內」與「等價 payoff 的市場對之間」的套利；Gebele & Matthes (2026)：跨 venue 的等價或子集合約對；Rothschild & Pennock (2014)：相同與邏輯相關合約間的 misalignment | 互斥、nesting、partition、conjunction 多型關係放進**同一個** constraint system 聯合求解 |
| 求解物件 | 上述多為 pair 或等價類的價差檢定 | convex hull 外的 ℓ1 距離 → integer programming 求最大保證利潤組合 |
| 規模測量 | Rothschild & Pennock (2014) 以 field trades 測流動性 | 全深度 order book 逐檔掃描 + on-chain fill events 對照實際參與 |

### 1.2 ⚡ Eureka

> **把「邏輯」寫成線性約束、把「套利存在」寫成凸包外、把「套利規模」寫成盤口逐檔累加——三個原本分屬幾何、優化、microstructure 的問題，被壓成同一條可回放的 pipeline。**

直覺：coherent set = 所有合法聯合結算 payoff 向量的 convex hull。報價在 hull 內 → 存在一組一致性定價信念讓它合理；在 hull 外 → 無論世界怎麼結算都賺，這就是保證利潤，而 ℓ1 距離正好量化「每單位部位賺多少」。合約一多，payoff 向量是指數級，不能列舉，於是反轉：不枚舉 payoff，而是把產生 payoff 的邏輯關係寫成約束，交給 IP。

### 1.3 信息流

```
Polymarket market catalog（~745,000 合約 + resolution text）
        │
        ├─ Layer 1 venue-asserted：官方 mutually exclusive 群組 + 群組規則
        ├─ Layer 2 venue-derived：match / contract type / line 欄位重建
        └─ Layer 3 semantic-inferred：structured parser + LLM 讀 settlement text
        │
        ▼  逐層篩檢（invalid relation → 假保證利潤；漏關係 → 只是漏機會）
   1.9 million relations ──► ~570,000 relations / >255,000 contracts
        │
        ▼  每層各自編譯成 linear constraint system
   觀測報價 p ──► 檢查 p 是否在 conv(payoff) 外 ──► ℓ1 距離 = 每單位保證利潤
        │
        ▼  integer programming：多層關係聯合下的最優組合方向
   組合方向（買哪幾檔、各多少）
        │
        ▼  逐檔掃 displayed depth（73.22 billion order-book messages 回放）
   扣 spread + taker fee 後的可執行規模與利潤
        │
        ▼  對照 227 million on-chain fill events
   實際交易者是否照套利方向交易（44% / 15% / 9%）
```

## §2 · 數學層

📌 **Napkin Formula**

無套利 ⟺ `p ∈ conv{ a⁽¹⁾, …, a⁽ᴹ⁾ }`，其中 `a⁽ʲ⁾ ∈ {0,1}ⁿ` 為第 j 種 admissible joint resolution 的 payoff 向量。

最大保證利潤 / 單位部位 = `dist_{ℓ1}( p , conv{a⁽ʲ⁾} )`。

求解：`max_x  (保證利潤)` s.t. `邏輯關係 → 線性約束`, `x` 整數 —— 不枚舉 `a⁽ʲ⁾`（數量隨合約數指數增長），只帶入約束。

**直覺**：hull 是「所有講得通的定價」的集合；報價跑到外面，代表市場對同一組互斥事件的隱含機率加總偏離可行域，偏離量就是無風險利潤。IP 的作用是在「這檔和那檔不能同時為真」這類組合約束下，找出最便宜的 payoff 打包方式。

**loss / 訓練細節**：無 supervised loss、無訓練階段；LLM 只做 relation extraction（抽取品質的 precision/recall 未披露）。求解器、time limit、gap tolerance、複雜度的實測數字均未披露。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ 以下全部為**示意假設數字**，用來手算核心機制，**不是**論文實證結果；論文真實數字一律見 §5，缺失者寫「未披露」。

**Step 0（假設）**：三檔合約 A / B / C 對同一事件構成 mutually exclusive 且 exhaustive 的分割，恰好一檔結算為 1。報價 `p_A = 0.30`、`p_B = 0.30`、`p_C = 0.30`（示意）。

**Step 1（建 payoff）**：admissible payoff 向量只有三個：`(1,0,0)`、`(0,1,0)`、`(0,0,1)`。convex hull 就是這三點的單純形，等價於約束 `p_A + p_B + p_C = 1`、各分量 ≥ 0。

**Step 2（判套利）**：報價和 `0.30 + 0.30 + 0.30 = 0.90 < 1`，落在 hull 外。ℓ1 距離 = `1 − 0.90 = 0.10` → **每單位組合的保證利潤 0.10**（示意）。

**Step 3（IP 定方向）**：目標是買入一個 payoff 恆等於 1 的組合且成本最小。約束系統給出的最優解是「A、B、C 各買 1 單位」，成本 0.90，到期 payoff 恆為 1.00。若再加一條 nesting 關係（例如 D ⊂ A），約束系統會多一條 `p_D ≤ p_A`，IP 便在這個多面體內重解——這正是它無法用 pairwise 價差替代的原因。

**Step 4（掃盤口定規模）**：假設盤口顯示 A 可買 100 單位、B 可買 200、C 可買 150（示意）。組合須三檔同比例，故規模上限 = `min(100, 200, 150) = 100`（示意），毛利 `100 × 0.10 = 10`（示意單位）。

**Step 5（扣成本）**：扣掉 spread 與 taker fee 後的淨利，示意層面留空——論文的對應真實測量見 §5。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 市場 / venue | Polymarket（單一 venue） |
| 期間 | 23 June – 3 August 2026 |
| 合約池 | 約 745,000 檔合約及其 resolution text，取自 Polymarket market catalog |
| 盤口 | 73.22 billion order-book messages（full-depth） |
| 成交 | 227 million on-chain fill events |
| 關係抽取 | 三層：venue-asserted（官方 mutually exclusive 群組＋群組規則）／venue-derived（match、contract type、line 欄位重建）／semantic-inferred（structured parser + LLM 讀 settlement text） |
| 關係規模 | 1.9 million relations → 篩檢後約 570,000 relations，覆蓋 >255,000 contracts；每層各自編譯成獨立 constraint system |
| 樣本外 | 未披露（未見 train/test split 或 walk-forward 設計；本文為測量型而非預測型，樣本外概念不直接適用） |
| 容量假設 | 以 displayed depth 為可交易上限；另測「re-posted depth 不算新流動性」與「只吃部分 displayed depth」兩種保守化 |
| 延遲假設 | 1-second order delay 情境 |

**解構**：這是「catalog + L2 全深度 + 鏈上成交」的三表對齊，而非特徵工程。關鍵風險在對齊本身——catalog 的 resolution text 是可變欄位，回放時取的是哪個版本未披露；LLM 抽取若用了事後修訂文本，會直接污染 relation 集合。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD（導讀未提） |
| Checkpoint | 未披露（LLM 具體模型與版本未披露） |
| License | 論文為 CC BY-NC-SA 4.0；代碼 license 未披露 |
| 複現難度 | 高：需 73.22 billion 訊息級 full-depth L2 回放 + 227 million 鏈上事件對齊 + 求解器排程，且 relation screening 規則細節未在導讀展開 |
| 數據可得性 | Polymarket catalog 與鏈上成交為公開來源（具體抓取與清洗流程未披露）；full-depth 歷史盤口重建方式未披露 |

## §5 · 評測 / Benchmark

> 本文非 ML benchmark 型論文，導讀未提供任何同任務基線的對應數值，故「前SOTA」一律為未披露；不做跨基線合成。

| 數據集 / 市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Polymarket，已結算組合 | 保證利潤 / committed capital（扣 spread 與 fee） | 未披露 | 2.2% | 未披露 |
| Polymarket，已結算組合 | 到期利潤 / committed capital | 未披露 | 3.2% | 未披露 |
| Polymarket，已結算組合 | 保證利潤（美元） | 未披露 | \$2.50 million | 未披露 |
| Polymarket，已結算組合 | 到期利潤（美元） | 未披露 | \$3.63 million | 未披露 |
| Polymarket，已結算組合 | committed capital | 未披露 | \$113.7 million | 未披露 |
| Polymarket | 每窗口首筆執行保證利潤 | 未披露 | \$1.10 million | 未披露 |
| Polymarket | 偵測到的套利窗口數（best quotes 下保證利潤為正） | 未披露 | 491,939 | 未披露 |
| Polymarket | 可實際交易的窗口數 | 未披露 | 約 125,000 | 未披露 |
| Polymarket | 執行假設敏感性：1-second order delay 下保留的到期利潤比例 | 未披露 | 79% | 未披露 |
| Polymarket | 中位套利窗口持續時間（≥0.3 秒窗口；三層分別） | 未披露 | venue-asserted 2.9 seconds / venue-derived 6.3 seconds / semantic-inferred 約 31 seconds | 未披露 |
| Polymarket | 保證利潤來源分解（三層分別） | 未披露 | venue-asserted 34% / venue-derived 59% / semantic-inferred 7% | 未披露 |
| Polymarket | 至少一交易者按套利方向交易每一 leg 的窗口佔比（三層分別） | 未披露 | venue-asserted 44% / venue-derived 15% / semantic-inferred 9% | 未披露 |
| Polymarket | 從窗口開啟到該類交易的中位延遲 | 未披露 | 約 3 seconds | 未披露 |
| Polymarket | 執行此類交易的錢包數 / 前 10 活躍地址佔比 | 未披露 | 854 addresses / 78% | 未披露 |

**哪部分是真 capability**：2.2% 這個數字之所以不只是 fitting，在於它是「扣 spread 與 fee 後、且對 re-posted depth 與 partial fill 保守化後仍成立」的量，且 34% / 59% / 7% 的來源分解與「關係越難找、持續越久、越少人吃」的單調排序互相印證——若是前瞻偏差驅動，這三個獨立信號不會同向。三層持續時間的排序在控制 portfolio 特徵、該 component 當日窗口數、day 與 hour fixed effects 後仍成立（venue-derived 約 2 times、semantic-inferred 約 10 times），這是本文最像「結構」而非「雜訊」的部分。

**哪部分可疑**：(a) 樣本條件於「合約全部已結算」，是對結算結果的選擇，不是全市場無條件期望；(b) 利潤以 displayed depth 為上限，若該 depth 本身由可撤單構成，回放會高估可成交量；(c) 491,939 個窗口對應僅約 125,000 個可交易，說明「偵測到」與「吃得到」之間有大量損耗，任何只報窗口數的轉述都會失真；(d) 7% 這層由 LLM 抽取，其 precision 未被量化，而作者的篩檢邏輯承認 invalid relation 會製造假保證利潤——這是唯一會單向膨脹利潤的誤差源。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

**未核原文，作者自述限制未驗證。** 導讀（摘要與 §1 全文）未見獨立的 limitations 小節；可引的作者自陳僅限於方法層面的保守化處理（重貼深度不計為新流動性、只吃部分 displayed depth、1-second order delay），這些是 robustness 設計而非限制陳述。具體失效閾值、成本上限、樣本量下限：未披露。

### 6.2 推斷的隱含假設（以下為本頁推斷，非作者陳述）

- **Regime 依賴**：全部證據落在 23 June – 3 August 2026 的單一 venue、單一市場結構。若 Polymarket 調整 fee 曲線或引入更強的一致性約束，2.2% 這類量級不具外推性。適用條件：prediction market 仍以「各合約獨立 order book」形式運作。
- **容量**：以 displayed depth 為可交易上限，隱含「resting orders 在回放中可被無衝擊吃掉」。對 micro 盘口而言這是強假設；真實 taker 會面對 queue priority 與撤單競速。作者已用 partial-fill 與 re-post 兩種保守化緩解，但未給衝擊模型。
- **成本**：只計 spread 與 taker fee，未見 gas、資金占用機會成本、resolution 前的保證金效率、以及 short 端不可行時的替代成本（Polymarket 無 naked short，組合須以買入或 No 腿實現，導讀未展開此映射）。
- **數據泄漏**：LLM 讀 settlement text 抽關係，若回放所用文本版本晚於報價時間戳，則 Layer 3 的 alpha 含事後資訊。文本版本控制未披露。
- **樣本選擇**：僅統計已結算組合（"Among settled portfolios"），條件於結算完成，會系統性排除未結算與爭議結算的合約；這不是輔助數據缺失，而是明確的結果條件化，需在引用 2.2% 時一併聲明。
- **關係完備性**：作者自陳「漏掉有效關係只會藏起機會、納入無效關係會製造假保證利潤」，因此篩檢必然偏保守（利潤偏低估）與偏激進（LLM 層偏膨脹）並存，淨方向未披露。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Saguillo et al. (2025) | LM 識別依賴；限「市場內」與「等價 payoff 的市場對」；無多型關係聯合 | 未披露 | 已發表（導讀引用） |
| Rothschild & Pennock (2014) | 相同／邏輯相關合約的 misalignment；以 field trades 測流動性，非全深度回放 | 未披露 | 已發表（導讀引用） |
| Gebele & Matthes (2026) | 跨 venue 的等價或子集合約對；關係型別較窄 | 未披露 | 已發表（導讀引用） |
| Capponi, Gliozzo & Zhu (2025) | 語意關係發現服務統計交易，非保證利潤的組合優化 | 未披露 | 已發表（導讀引用） |
| Dudík, Lahaie & Pennock (2012) / Kroer et al. (2016) | constraint generation + IP，但屬 **market maker 側**維持 coherence；本文為 **trader 側**，價格已形成且無人維持一致 | 未披露 | 已發表（導讀引用） |

🎤 **Interview Tip**

- **問**：「這篇跟『用 LLM 在新聞裡找配對交易』差在哪？」
- **正確答**：差在目標函數。這裡不是預測方向，而是求「在所有 admissible 結算下 payoff 恆正的組合」——存在性由 convex hull 幾何判定，大小由 ℓ1 距離給出，規模由盤口逐檔給出。LLM 只是 relation discovery 的其中一層（且只貢獻 7% 的保證利潤），可替換。真正的技術資產是「把多型邏輯關係編譯成線性約束、再交給 IP」這一步，它讓 payoff 向量不必被列舉。
- **錯答**：「用 LLM 讀結算條款做套利」——把手段當方法，且忽略 pairwise 價差無法處理 mutually exclusive / nesting 交互這一核心動機；或「保證利潤就是無風險利率」——忽略樣本條件於已結算、成本只計 spread 與 fee、以及 displayed depth 上限。

### 7.1 可證偽預測（判準日期：2027-06-30）

1. 若 Polymarket 在 2027-06-30 前對 mutually exclusive 群組引入自動一致性約束或調整 mint/merge 機制，則 semantic-inferred 層的中位窗口持續時間（現為約 31 seconds）應相對 venue-asserted 層（2.9 seconds）的倍數明顯壓縮；若倍數不變，則本文「發現成本決定半衰期」的解釋被削弱。
2. 若把 Layer 3 的 LLM 抽取換成更強模型並放寬 recall，保證利潤的來源分解中 semantic-inferred 佔比（現為 7%）應上升；若不上升，說明該層的瓶頸在執行成本而非發現成本。
3. 若以同一 constraint system 移植到 Kalshi 或跨 venue，2.2% 這類量級不應直接重現——跨 venue 需額外承擔結算規則差異與資金跨場轉移成本；若聲稱可直接重現，視為過擬合信號。

## §8 · For the Reader

- **因子研究員**：把「邏輯約束 → 線性不等式系統 → IP」抽出來，可移植到任何有結構約束的組合定價問題（日曆價差、同一標的的多重 barrier 產品）。別抄 2.2%，抄 constraint 編譯流程。
- **高頻執行**：本文最可操作的是「偵測窗口數 vs 可交易窗口數」的落差（491,939 → 約 125,000）與 1-second delay 下的衰減（保留 79%）。這是 queue 競速與撤單預測的入口，不是 alpha 入口。
- **組合配置**：注意樣本條件於已結算組合，且利潤以 displayed depth 為上限。任何把 2.2% 當成可配置收益率的做法，都需先補上資金占用、保證金效率與爭議結算風險。
- **LLM-agent 開發者**：本文把 LLM 定位為 relation extractor 且只貢獻 7%，這是清醒的 baseline。要做的是量測抽取 precision 對假保證利潤的單向膨脹效應，而非追求 recall。
- **RL / 最佳化策略**：IP 的形式（整數變量 + 邏輯約束）比 RL 更適合此問題，因為約束是硬性的、不需要學。可借鏡的是把 hard constraint 與 market depth 分層求解：先定方向（IP），再定規模（盤口掃描）。
- **研究學生**：這是「幾何刻畫 + 優化求解 + microstructure 回放」三段式論文的乾淨範本。讀法建議：先讀凸包與 ℓ1 距離的等價性命題，再讀三層關係的篩檢邏輯（為什麼 invalid relation 比 missing relation 危險），最後才讀實證表。

## References

- Shi, H., & Zhai, D. (2026). *Combinatorial Arbitrage in Prediction Markets.* arXiv:2610.23517v1 [q-fin.TR]. https://arxiv.org/abs/2610.23517v1
- Saguillo, et al. (2025). 語言模型識別 Polymarket 市場依賴並測算套利（導讀引用）。
- Rothschild & Pennock (2014). 相同與邏輯相關合約的 misalignment 與 field trades（導讀引用）。
- Gebele & Matthes (2026). 跨 venue 等價／子集合約對識別（導讀引用）。
- Capponi, Gliozzo & Zhu (2025). 語意關係發現於統計交易（導讀引用）。
- Dudík, Lahaie & Pennock (2012); Kroer et al. (2016). Coherent pricing 的 constraint generation 與 integer programming（導讀引用）。
- Hanson (2003); Abernethy, Chen & Wortman Vaughan (2013); Chen et al. (2008). 組合市場設計與 coherent price 計算複雜度（導讀引用）。
- Lamont & Thaler (2003); Gromb & Vayanos (2002). 一價定律違反與受約束套利（導讀引用）。
- 來源連結：https://arxiv.org/abs/2610.23517