<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=监督回归 alpha=因子挖掘 autonomy=全自动黑盒 -->

# Beyond Anonymous Order F 解構（Beyond Anonymous Order F）

> **來源發布**：2026-11-03 · （無 venue） · arXiv [2611.05218](https://arxiv.org/abs/2611.05218)
> **arXiv 原文**：[Beyond Anonymous Order Flow: Wallet Identity as a Feature for Short-Horizon Return Prediction](https://arxiv.org/abs/2611.05218v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：微觀盤口 × 高頻日內 × 監督回歸 × 因子挖掘 × 全自動黑盒。它解的是 anonymous LOB 文獻的結構性 prior gap——Cont/Sirignano/Kolm 一類模型只能條件於「市場在做什麼」，無法條件於「是誰在做」；本文把 Hyperliquid L4 的 persistent pseudonymous wallet address 當成可累積歷史的觀測單位，檢驗 wallet markout 排序的跨窗持久性，再測它對 1 秒收益預測的增量。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `监督回归` | `因子挖掘` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①用 Hyperliquid L4（2026-07-01–27，17.1 billion messages、\$84.3 billion taker notional）重建 full-depth book 並把 wallet address 掛回每一筆訂單，先以十秒 signed markout 對錢包打分，再把 top-decile 錢包的 11 個報價／流量特徵接到 10 個 literature-based anonymous 特徵上。②核心 trick 不是新模型，而是「凍結窗口的錢包分數 + 同構特徵映射」：identity block 用與 anonymous block 平行定義的 quote/flow 變數，讓 ridge/GBT 在同一條 100 ms 網格上做 A/B。③★ 在因子挖掘軸上這是把「交易對手異質性」變成可回測的橫截面因子，而非再挖一層盤口深度。④1 秒 ridge 的 out-of-sample \$R^2\$ 由 10.88% 升到 12.31%（+13.2% relative，\$t=9.2\$），但同一增量在 matched-cohort placebo 上最大只有 0.88 個百分點（本方法 1.43），且 GBT 下只剩 +6.0%。

**X-Ray.** 放回五軸 Pareto：它沒有換模型族，換的是**特徵的觀測單位**。在「匿名盤口 + ridge/GBT」這個已被榨乾的角落，作者押的是資料供給側的差異化——Hyperliquid 把 rejected、cancel、fill 與地址一起公開，等於把傳統只有 broker/監管方看得到的 account-level 面板搬到公開 feed 上。這解掉兩個舊工程坑：（a）多數交易所不給帳戶歷史，adverse selection 只能用 spread decomposition / PIN 這類匿名代理間接推，作者引 Collin-Dufresne–Fos 指出這類代理正好在識別最重要時失效；（b）「toxic flow」文獻（Cartea et al. 2026）在 dealer 內部做，本文把它搬到全公開、可被任何人重放的 tape 上。它打不開的 envelope 也很清楚：作者自陳這是 literature-based、non-maximal 的匿名基準，multi-level depth、cancellation、rejection、intensity、cross-venue 都可能吃掉增量；而 consensus-time 而非 receipt-time 索引、gap 期間 carried-forward state、以及「最終 top-decile 焦點／volatility scaling／July 評估窗是看過結果後選的」，三條合起來意味著**這是一個 chronological out-of-sample 比較，不是 prospectively locked test**。對量化讀者的意義：把這篇當成「identity 這條特徵線索的天花板估計的上界」，而不是「13.2% 可交易的 alpha」。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作（anonymous LOB 預測） | 本文 | 為何是改動 |
|---|---|---|---|
| 觀測單位 | aggregate book / order-flow imbalance，交易對手不可識別 | persistent pseudonymous wallet address，逐錢包累積 markout 歷史 | 把「誰在下單」變成可條件變數，而不只是「下了多少」 |
| 資料重建 | 公開 BBO / 逐筆成交，或 vendor 整理過的 L2 | 非驗證節點收 consensus-ordered messages，deterministic replay 重建 L4、maker/taker 配對、rejected attempts | 只有 L4 + 共識排序才能把 maker 與 taker 兩側、以及地址，正確接回時間軸 |
| 特徵映射 | 單一 10 變數 block | 10 anonymous + 11 identity；Panel A 為同構配對（imb↔hotImb、flow1s↔hotFlow1s…），Panel C 為無匿名對應的 4 個地址變數 | 讓 A/B 差異可歸因於「同一組量的 cohort 版本」，而不是換一組特徵族 |

驗證設計另有三件不是模型改動、但決定結論強度的東西：peer-adjusted score（對同 coin-minute 的其他錢包淨化）、200 個 activity-matched placebo cohort、以及 200/300 ms 的全特徵延遲 embargo。

### 1.2 ⚡ Eureka 一句話 trick + 直覺

**把「錢包」當成一個會持續留下痕跡的慢變量因子：先用過去十天的 aggressive-order markout 給錢包打分並凍結，再用這批高分錢包「當下的報價與流量」去預測未來一秒的 mid return。**

直覺：adverse selection 模型（Bagehot/Kyle/Glosten–Milgrom）說交易對手在成交後的价格漂移上不對稱；若這種不對稱是錢包的**屬性**而非一次性運氣，則（i）排序會跨窗持續，（ii）高分錢包此刻的動作比全市場聚合流量更接近「知情那一側」。這個 trick 的關鍵在**凍結**——分數只用 July 1–10，模型只用 July 11–20，評估用 July 21–27，避免分數與標的同期互相污染。代價是分數會 stale，且 July 10 之後才出現的錢包永遠沒有分（作者自陳）。

### 1.3 信息流 ASCII

    Hyperliquid validators
       │  consensus-ordered blocks（median inter-block gap 67.6 ms；91.3% ≤ 100 ms）
       ▼
    non-validating node ──► raw committed messages（July 1–27：17.1 bn msgs）
       │  deterministic replay（13 segments）
       ▼
    L4 book state + maker/taker fill pairing + wallet address
       │                                   │
       │ 10s oracle gate                   └──► aggressive-order markout x_e（Eq.1）
       │ 12 連續 10s tick 內 ≤ 25 bp              │  frozen July 1–10，notional 加權
       ▼                                          ▼
    100 ms 網格 state ────────────────────► α_w（Eq.2）──► top decile（231 wallets）
       │                                                     │
       ├─ 10 anonymous 特徵 ─┐                               │ 11 identity 特徵
       │                     ├──► ridge / LightGBM ◄─────────┘
       └─────────────────────┘             │
                                           ▼
                          y = 1s forward mid return（100 ms 網格）
                                           │
                          matched-cohort placebo ×200 / 200–300 ms embargo

## §2 · 數學層

📌 **Napkin Formula**

$$x_e = 10^{4}\,q_e\,\frac{m_{t_e+10s}-m_{t_e^{-}}}{m_{t_e^{-}}},\qquad \alpha_w=\frac{\sum_{e\in w} n_e x_e}{\sum_{e\in w} n_e},\qquad y_{c,t}(\Delta)=10^{4}\,\frac{m_{c,t+\Delta}-m_{c,t}}{m_{c,t}}$$

直覺：Eq.1 是「主動方視角」的十秒 signed markout，$q_e=+1$ 買、$-1$ 賣，正值代表後來的 mid 往侵略者方向走；它包含當塊衝擊與之後的漂移，所以不是純「資訊」而是「資訊 + 衝擊 + 後續同向流」的混合物。Eq.2 用 notional 加權把單筆壓成錢包級分數，門檻是至少 100 筆 qualifying aggressive orders、排除 TWAP 與 liquidation。Eq.3 的標的刻意落在固定 100 ms 網格而非 block-to-block，因為 block 間隔不規則，block-to-block 會把不同 elapsed time 的報酬混在一起。

**Loss / 訓練**：評估量是對零預測的 $R^2=1-\sum(y-\hat y)^2/\sum y^2$，每個保留網格點等權。ridge 掃 17 個 penalty $\lambda=10^{-7+0.5j},\ j=0,\ldots,16$，用 9 個 expanding fold（July 12–20 每天對其之前所有 fitting days 驗證），取日度驗證 MSE 和最小者重擬合 July 11–20；特徵用 fitting-window 的 pooled 均值／標準差標準化（含 validation fold 內）。LightGBM：learning rate 0.05、127 leaves、每葉至少 5,000 觀察、feature fraction 0.9、bagging fraction 0.8、最多 2,000 rounds、early stopping 50 rounds，seed 20251201；1 秒時 early stopping 選 192 棵（anonymous）與 415 棵（full）。推論用 7 個 July 日的 paired daily MSE 差做 $t$。

**複雜度**：所讀文本未給漸進複雜度；工程量級為 17.1 billion messages 的重放、44.0 million grid stamps × 201 cohorts = 8.8 billion cohort-observations、126 gigabytes 的 per-wallet quote tapes。

## §2.5 · 帶數字走一遍（Worked Example）

以下為**假設／示意**，數字借用原文既有值以便讀者照算；不代表任何真實錢包或論文實證結果。

1. **打分（Eq.2，示意）**：假設某錢包在 July 1–10 有 100 筆 qualifying aggressive orders（100 正是原文門檻）。對每筆算 Eq.1 的 $x_e$，再以 $n_e$ 加權平均得 $\alpha_w$。原文 top-decile 在 0.5 秒的 markout 是 1.25 bp、10 秒是 2.11 bp，所以一個「有料」錢包的 $\alpha_w$ 大致落在這兩個數所刻畫的區間內（示意，不求值）。
2. **凍結與排序**：對 231 個錢包（top decile）算完 $\alpha_w$ 後凍結，不隨評估期更新。驗證期（July 11–20）這批錢包的 markout 在 D10 為 2.20 bp，D9 為 1.09 bp，D8 及以下 ≤ 0.60 bp。
3. **造特徵**：把 top decile 當下的報價與流量做成 11 個 identity 變數，例如 hotOfi1s（cohort 1 秒報價更新 OFI）與 hotBreadth1s（100 ms bucket 內 distinct buyer 減 seller 的滾動和，錢包可重複計入）。
4. **A/B**：10 個 anonymous 變數的 1 秒 ridge $R^2$ = 10.88%；加上 11 個 identity 變數後 = 12.31%，差 1.43 個百分點（原文寫 "adds ... 1.43 points"），relative gain 13.2%，$t=9.2$。
5. **對照**：200 個 activity-matched cohort 中最好的 draw 只到 0.88 個百分點，故 1.43 是它的 1.6 倍（原文 "1.6 times"），best draw 只到真實增量的 61%。再把所有特徵延遲 300 ms：relative gain 由 13.2% 降到 9.1%，而 anonymous 的 $R^2$ 由 10.88% 降到 6.40%——增量比基準更耐延遲。
6. **判讀**：這條鏈條每一環都可手算，但第 4 步的 1.43 與第 5 步的 1.6 倍的差距正是全篇的爭點——它證明「不只是活性匹配」，不證明「這是資訊通道」。

## §3 · 數據層

| 維度 | 內容 |
|---|---|
| 樣本 | 2026-07-01–27；17.1 billion messages、14.3 million aggressive orders、147,113 wallets、\$84.3 billion taker notional、10 個 perpetual 市場（Table 1 全十行：17.09 / 14.27 / 27.90 / 147.1 / 84.3） |
| 估計市場 | btc、eth、sol（message 活動最高三個）；BTC 佔 taker notional \$46.8 bn（總 \$84.3 bn），ETH 18.7、SOL 4.9 |
| 頻率／時鐘 | 100 ms 網格；inter-block gap 中位數 67.6 ms、第 90 百分位 93.6 ms、91.3% ≤ 100 ms（22,734,866 個 gap）；所有時間索引為 **consensus timestamp，非本地 receipt time** |
| 來源 | 自建 non-validating node 連續擷取 + deterministic replay；另每十秒輪詢 metaAndAssetCtxs 取 mark/oracle/OI；December 2025 對照用 Albers et al. (2026a) 公開 release（26.3 billion messages、21.9 million taker fills、89,800 wallets） |
| 時間切分 | score July 1–10；fit July 11–20；eval July 21–27；trees 訓練 July 11–19、July 20 做 early stopping。December：score 12/1–10、fit 12/11–20、eval 12/21–31 |
| 覆蓋與品質 | gate 要求連續 12 個十秒 tick 與重建 mid 相差 ≤ 25 bp；gate 關閉時間佔 July 1–27 的 0.97%、評估窗 1.83%，其中 81% 為三市場同時（與 oracle 側中斷一致）；27,902,380 筆 crossed fills 的 maker 側配對零缺失；13 個 segment；July 12 獨立重放對 1.32 million blocks 的 bid/ask/mid/spread 一致率 99.994%；July 1–30 收集窗的 30 天中有 22 天日內最大 block gap ≤ 3 |
| 容量假設 | 所讀文本未給資金容量或成本曲線；可供推論的只有 cohort 的流量占比——top decile 佔 scoring-window discretionary taker notional 的 31.0%、validation-window 的 25.1%（這是集中度，不是容量） |

**缺口標記**：評估窗 July 25 有 16-block gap、July 27 有 28,326-block gap；scoring 窗 July 1、July 2 分別為 27,900 與 59,415-block gap，July 3、4、6 的最大值為 4、7、4 blocks。作者另提供去掉 July 25、27 的日度敏感性（不重擬合）：ridge 11.72%→13.18%，trees 19.75%→20.90%。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 所讀文本未見 code link（未披露；附件待核） |
| Checkpoint | 未披露（模型為 ridge 與 LightGBM，未給權重檔） |
| License | 論文頁標示 CC BY-NC-SA 4.0；代碼與 artifact 的 license 未披露 |
| 複現難度 | 高：需 Hyperliquid 非驗證節點連續擷取 + deterministic consensus replay + maker/taker 配對；placebo 一步要重建 200 個 cohort 的逐錢包 quote tapes（126 gigabytes）；另需 12 個十秒 tick 的 oracle gate 邏輯 |
| 數據可得性 | Level-4 公開數據見 Albers et al. (2026a)（SSRN Working Paper 6465720；dataset DOI 10.5281/zenodo.18184441），December 2025 樣本即來自此；July 2026 樣本為作者自建 collector，是否釋出未披露 |

## §5 · 評測 / Benchmark

前SOTA 欄實為原文的 **anonymous benchmark**（作者自述 literature-based、non-maximal），不是「先前最佳方法」；下表逐字保留其值，Δ = 本方法 − 基線，單位為 $R^2$ 百分點（pp）。所有列皆為同一資料切分（score July 1–10 / fit July 11–20 / eval July 21–27）、同一 100 ms 網格、三個市場 btc/eth/sol。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| btc+eth+sol, 0.2 s | ridge $R^2$ (%) | 6.91 | 8.20 | +1.29pp |
| btc+eth+sol, 0.5 s | ridge $R^2$ (%) | 10.16 | 11.76 | +1.60pp |
| btc+eth+sol, 1 s | ridge $R^2$ (%) | 10.88 | 12.31 | +1.43pp |
| btc+eth+sol, 2 s | ridge $R^2$ (%) | 10.60 | 11.66 | +1.06pp |
| btc+eth+sol, 5 s | ridge $R^2$ (%) | 9.17 | 9.79 | +0.62pp |
| btc+eth+sol, 10 s | ridge $R^2$ (%) | 7.18 | 7.56 | +0.38pp |
| btc+eth+sol, 30 s | ridge $R^2$ (%) | 3.56 | 3.67 | +0.11pp |
| btc+eth+sol, 1 s | GBT $R^2$ (%) | 19.48 | 20.65 | +1.17pp |

同一設計下的其他逐字結果（非固定五欄表，故不計 Δ）：
- **identity block 單獨** 1 秒 ridge $R^2$ = 10.06%，已接近 anonymous 的 10.88%；兩塊合併只加 1.43 點 → 兩塊高度互相 span。
- **top quintile** 1 秒 ridge relative gain 13.1%（與 top decile 幾乎相同）。
- **peer-adjusted score**（對同 coin-minute 其他錢包淨化）1 秒 gain 6.8%，約為 raw 13.2% 的一半（原文 "roughly half"）。
- **matched-wallet placebo**：200 個 cohort 最好者 0.88pp，本方法 1.43pp，比值 1.6 倍，best draw 達 61%；toxic cohort 在 10 秒內每個 horizon 都超過所有 draw，30 秒時 top decile 進入 placebo 區間而 top quintile 仍全部超過。
- **feature embargo**：全特徵延遲 200/300 ms，ridge relative gain 13.2% → 10.2% → 9.1%（$t=8.2$），trees 6.0% → 4.5% → 4.0%；300 ms 下 anonymous $R^2$ 由 10.88% 降到 6.40%（ridge）、19.48% 降到 9.64%（trees）。
- **December 2025 外部樣本**：$\rho=0.47$；volatility-scaled 規格下 1 秒 ridge $R^2$ 12.36% → 14.16%（+14.6% relative，$t=3.9$）；**不做 near-touch volatility scaling 時增量為 −1.17 個百分點**。
- **drop-one（1 秒 ridge，單位 pp，描述性且不可加）**：nearImb 1.93 → 0.24；ofi1s 1.64 → 1.25；imb 1.03 → 0.27；identity 側 hotOfi1s 0.67、hotBreadth1s 0.37。

**解讀**：真 capability 的部分我判為（i）1 秒 ridge 的 1.43pp 在 200/300 ms 延遲下仍保留大部分（10.2%/9.1%），比 anonymous 基準本身更耐延遲，這與「純粹靠最新 tick 的時間對齊泄漏」不符；（ii）matched-cohort placebo 顯示增量不能被「任何同樣大、同樣活躍的 cohort」複製（1.6 倍、61%）。可能是過擬合／選擇的部分：（a）top-decile 焦點、near-touch volatility scaling、July 評估窗三者都是看過結果後選的，作者自己列為 limitation；（b）7 個評估日、眾多未校正比較，$t$ 只在日度聚合層面處理依賴；（c）December 樣本一旦去掉 volatility scaling 就翻負（−1.17pp），說明 14.6% 不是穩健的規格不變量；（d）drop-one 顯示 anonymous nearImb 的貢獻在加入 identity 後從 1.93 塌到 0.24，identity 更像是「同一股流的低噪聲讀出」而非獨立通道——作者也承認這個替代解釋。成本未計：三個市場的 mean quoted spread 分別為 0.20 / 0.62 / 0.26 bps，但全文沒有任何 net-of-fee、queue priority、fill probability 的損益數字，作者明確說不證明 executable profit。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位到章節）

| 作者限制 | 位置 |
|---|---|
| 最終 top-decile 焦點、near-touch volatility scaling、July 評估窗的選擇都發生在看到結果之後；因此是 chronological out-of-sample 比較，不是 prospectively locked test | §4.5 "Selection and data limitations"；Abstract |
| 預測網格把 emitted state 帶過 coverage gap 與 gate closure，且不套用 markout 專屬的 staleness 篩選 | §4.2、§4.5 |
| 全部時間用 consensus timestamp 索引，而非另行審計的本地 receipt time；未證明交易者在該時點可取得該狀態 | Abstract、§2、§4.2、§4.5 |
| gate 的 25 bp 容忍度大於本文研究的 markout 效果；mid 一致時 resting-order 錯誤仍可能存活 | §2 |
| matched cohorts 因 exact-cell donor 不足而比 toxic cohort 小，排序為描述性、非等大小隨機化檢定 | §4.5 |
| December 2025 的 scaled 摘要不在既存 pre-scaling end-to-end audit 覆蓋內，需重新做 panel-to-output 審計 | §4.5 |
| 多個比較未校正，$t$ 統計為描述性；精確度仍依賴 between-day 依賴結構 | §4.2 |
| peer-adjusted 版本只控制 same coin-minute，未移除分鐘內的共同時序 | §3.1 |
| 不做 private information 識別、不證明可執行利潤；轉為決策仍需 latency、queue priority、fill probabilities、spreads、fees、inventory constraints | §1 "Controls and scope"、§5 |
| 匿名基準為 literature-based、non-maximal；multi-level depth、cancellation、rejection、intensity、cross-venue 可能壓縮增量；Albers et al. (2026b) 顯示 rejected orders 在此 venue 可預測報酬 | §4.1 footnote 3 |
| 完整 gap-masked 重跑仍待完成；需新一輪 prospectively locked、雙端 validity mask 的評估 | §4.5、§5 |

### 6.2 推斷的隱含假設（本頁推斷，非作者主張）

- **regime 依賴**：全部證據來自 2026-07 與 2025-12 兩段；錢包排序 $\rho=0.52$／0.47 是十個日曆日內的持久性。若市場結構、做市商組成或費用制度改變，凍結分數的排序會失效。此推斷不宣稱「所有 DEX 都如此」。
- **容量**：top decile 已佔 validation-window discretionary taker notional 的 25.1%。若真按此訊號加倉，cohort 本身的流量占比會被自己改動（特徵被策略反饋污染），這是本文完全未測的自我參照問題。
- **成本**：1 秒 horizon 的 $R^2$ 增量與可實現 bps 之間沒有換算；spread 0.20/0.62/0.26 bps 只是描述統計，未進入任何損益。
- **數據泄漏（最重要）**：consensus time 索引 + 未審計 receipt time，意味著「特徵在決策時刻是否可得」未證。200/300 ms embargo 是模擬延遲，不是可用性審計。
- **選擇效應而非 survivorship**：frozen window 讓 July 10 後新出現的錢包永不被評分，且 86.9% 的 scored wallets 在驗證窗繼續交易、top decile 續交易率 91.3%——這不是輔助數據缺失造成的生存者偏差，而是**設計上對「既有錢包」的條件化**；解讀時不可外推為「所有錢包」的性質。
- **單一樣本不可外推**：December 的 −1.17pp（去掉 volatility scaling）說明 14.6% 依賴一個事後選定的縮放；不可寫成「identity 在所有模型／市場必然有效」。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Cont et al. (2014)；Sirignano & Cont (2019)；Kolm et al. (2023) | 同樣做短 horizon LOB 預測，但交易對手不可識別、無 wallet history；本文的 anonymous block 直接沿用其變數 | 概念公開、資料不同 | 本文基準即此派的 literatures-based 子集 |
| Albers et al. (2026a) | Hyperliquid L4 資料本身；本文的資料層與 December 樣本來源于此 | 公開 dataset（DOI 10.5281/zenodo.18184441） | 基礎設施，非預測方法 |
| Albers et al. (2026b) | 同 venue，用 **rejected orders** 預測報酬 | 未披露 | 作者自陳可能壓縮本文增量 |
| Cartea et al. (2026) | 同樣「detect toxic flow」，但在 dealer 內部用 client identity | 未披露 | 本文把它搬到全公開 feed、以 markout 排序替代 dealer 內部資訊 |
| Barone & Lillo (2026) | 同 venue 的 adverse selection / 揭露執行意圖與成本 | 未披露 | 本文 markout 定義沿用其方法 |
| Zhang et al. (2026) ClusterLOB | 以聚類從 NASDAQ order events **反推** trader types | 未披露 | 無真實地址；本文是地址直接可觀測 |
| Foucault et al. (2007)、Meling (2021) | 匿名性對流動性的因果效應 | 否 | 市場結構證據，非預測增量 |

🎤 **Interview Tip**

- **正確答**：「增量是在一個作者自稱 non-maximal 的匿名特徵集上測的；ridge 1.43pp（13.2%、t=9.2）但 GBT 只剩 1.17pp（6.0%），說明非線性匿名特徵已吃掉一部分。drop-one 顯示 nearImb 的貢獻從 1.93 掉到 0.24，identity 更像同一股流的低噪聲讀出而非新通道；matched-cohort placebo 只排除了『活性匹配』這一種替代解釋。且全部時間是 consensus-time，receipt-time 未審計。」
- **錯答**：「錢包身份帶來 13.2% 的 alpha，可以直接上線」——把 relative $R^2$ 增量當收益、忽略 trees 下折半、忽略 December 去掉 scaling 後為 −1.17pp、忽略無成本與 fill probability。

### 7.1 可證偽預測（本頁推斷，待驗證）

1. **2027-06-30 前**若出現 prospectively locked、含 receipt-time 審計與雙端 validity mask 的重跑，1 秒 ridge 的 relative gain 會**低於** 13.2%（因 stale state 與不可得性被正確遮罩）。
2. 若匿名基準擴充到 multi-level depth + cancellation/rejection（Albers et al. 2026b 的方向），同一 identity block 的增量會被壓縮到低於 1.43pp 的水準。
3. 30 秒以上的 cohort 選擇不穩定：本就觀察到 top decile 在 30 秒進入 placebo 區間、top quintile 仍全部超過——預測更長 horizon 的「top-decile 優勢」不可複現。

## §8 · For the Reader

- **因子研究員**：把這篇當方法論模板而非 alpha 來源。可借的三件事：（a）以 signed markout 的 notional 加權均值做「交易者品質因子」，並用 split-half + Spearman–Brown（0.52/0.48 → 0.68/0.65）估可靠度上限；（b）drop-one 看新舊特徵塊的 span 程度，1.93 → 0.24 這種塌陷是「通道 vs 讀出」的判別訊號；（c）matched-cohort placebo 是比 shuffle 更硬的對照，但要注意 donor 不足導致的不等大小。
- **高頻執行**：本文沒有任何 net-of-cost 結果。若要碰，先自己做三件事：用 receipt-time 而非 consensus-time 重建特徵、把 gate 期間與 gap 期間的 state 遮成 missing、把 queue priority 與 fill probability 接進 1 秒 horizon。spread 0.20/0.62/0.26 bps 只說明這是個很緊的市場，不代表你的淨收益。
- **組合配置**：這是一個集中度極高的訊號——top decile（231 個錢包）佔 validation-window discretionary taker notional 的 25.1%。把它放進多策略組合前，先問「這部分流量本身是不是策略的容量上限」。
- **LLM-agent / 研究學生**：可複用的研究設計是「凍結打分窗 → 平行特徵映射 → 同 learner A/B → 活性匹配 placebo → 延遲 embargo」這條五步流水線；它的弱點也教你一件事：事後選規格（top-decile、volatility scaling、評估窗）會在單一外部樣本上以符號翻轉的方式現形（December 的 −1.17pp）。
- **RL 策略**：本文是監督回歸不是 RL；但 markout 可視為一個 dense reward shaping 的候選項。注意作者提醒的「persistent impact、repeated same-direction flow、latent metaorders」都還在替代解釋清單上，所以 markout 不應直接當成「資訊」的 reward。

## References

- Ding, J. & Zhai, D. (2026). *Beyond Anonymous Order Flow: Wallet Identity as a Feature for Short-Horizon Return Prediction.* arXiv:2611.05218v1 [q-fin.TR], 03 Nov 2026. https://arxiv.org/abs/2611.05218v1
- Albers, J., Cucuringu, M., Howison, S., & Shestopaloff, A. Y. (2026a). *An open book: Level 4 order book data from the Hyperliquid exchange.* SSRN Working Paper 6465720；dataset DOI 10.5281/zenodo.18184441.
- Albers et al. (2026b). *The price impact of nothing: rejected orders as predictors of future returns.* SSRN Working Paper 6250378.
- Barone, D. & Lillo, F. (2026). *Trading in the sunshine or in the shade: market impact and adverse selection on Hyperliquid.* arXiv:2606.15715.
- Cartea, Á., Duran-Martin, G., & Sánchez-Betancourt, L. (2026). *Detecting toxic flow.* Quantitative Finance 26(4), 541–561.
- Cont, R., Kukanov, A., & Stoikov, S. (2014). *The price impact of order book events.* J. Financial Econometrics 12(1), 47–88.
- Sirignano, J. & Cont, R. (2019). *Universal features of price formation in financial markets.* Quantitative Finance 19(9), 1449–1459.
- Kolm, P. N., Turiel, J., & Westray, N. (2023). *Deep order flow imbalance.* Mathematical Finance 33(4), 1044–1081.
- Bagehot (1971)；Kyle (1985)；Glosten & Milgrom (1985)；Glosten & Harris (1988)；Hasbrouck (1991)；Easley et al. (1996)；Collin-Dufresne & Fos (2015)；Foucault et al. (2007)；Meling (2021)；Linnainmaa & Saar (2012)；Kirilenko et al. (2017)；Baron et al. (2019)；Barbon et al. (2019)；Foucault et al. (2017)；Avellaneda & Stoikov (2008)；Stoikov (2018)；Xu et al. (2019)；Biais et al. (1995)；Andersen et al. (2003)；Makarov & Schoar (2020)；Daian et al. (2020)；He et al. (2024)；Félez-Viñas et al. (2022)；Zhang et al. (2026).（詳見原文 References）
- 本頁來源鏈接：https://arxiv.org/abs/2611.05218　·　圖片／附件未核，`TBD` 項待 v1 補。