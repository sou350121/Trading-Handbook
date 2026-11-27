<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=多智能体博弈 autonomy=人机协同可解释 -->

# Jev / emu 解構（Jev / emu）

> **來源發布**：2026-11-07 · （無 venue） · arXiv [2611.10414](https://arxiv.org/abs/2611.10414)
> **arXiv 原文**：[Track or Follow? A Jev-Compatible Multi-Agent Trading System When Web News Lags the Market](https://arxiv.org/abs/2611.10414v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 Web 新聞 reader 放進 point-in-time 多智能體交易系統，用 Jev schema 的 typed judgment 與 hash-committed 協議，分離「讀事件」與「跟隨已發生價格反應」。五軸落點是文本另類 × 日頻波段 × 生成式大模型 × 多智能體博弈 × 人機協同可解釋；它解的是 LLM reader 評測中 event-window 與 publication 混淆、item 本身報導價格、以及 text 可任意改倉的 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `多智能体博弈` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 建了一個 reader-agnostic、point-in-time 的多智能體交易系統，所有 agent 只共享 write-once stores、只回傳 typed judgments，code 決定倉位。② 核心 trick 是 trusted logit 為 offset、text logit 被 clip 到 2κ，band 只允許 trusted rank ≤ K+J 的名額進場，decision card 把每個倉位變動歸因到 item。③ 對「生成式大模型 × 多智能體博弈 × 日頻波段」這條軸，它把 alpha 問題改寫成 criterion validity 與可審計的影響上界，而不是再堆 agent 辯論。④ 關鍵實證：announcement-window weekly Spearman 0.119 [0.098, 0.140]、8-K earnings AUC 0.833 [0.826, 0.840]，但從 publication 後第一筆交易起關聯降到 0.023 [0.005, 0.041]，H3 對 code-only twin 的 ΔIC 為 -0.016 [-0.029, -0.002]。

**X-Ray.** 在五軸 Pareto 上，這篇的價值不在「多智能體找到 alpha」，而在「把 LLM reader 的語義判斷放進可審計的決策邊界」。它解了三個舊工程坑：backtest 中模型權重 recall 後見之明、item 本身報導價格移動、以及 event window 與 publication window 被混為一談。代價是 envelope 很窄：one feed 的 headlines/summaries、100 large caps、weekly decision、next-open fill，且 H3 在此 learner/decision layer 上沒有 rank 增量。對量化讀者，它更像 evaluation harness 與 risk-bound 設計：可以拿來測 reader、測 poisoning、測 publication-aligned 衰減，不應直接當成可交易 alpha 來源。它打不開的 envelope 是「band 內被 fabricated fact / planted press release 推動」與「reader 在 publication 後仍掌握非公開信息」這兩類問題；前者由 reader 決定，後者所讀文本只以 post-release 與 descriptive Jev 間接處理。

## §1 · 架構 / Core Mechanism

### 三大改動 vs 前作

| 改動 | 前作／常見做法 | 本文做法 | 對讀者的意義 |
|---|---|---|---|
| Typed question program | agent 交換 free text、模型輸出可帶指令或直接命名 action | 每個 agent 只回傳 caller-declared options 的機率；code 把答案轉成特徵（log-odds clip ±4、ordered choice 的 value-weighted mean、score 的 centred expected level）；abstention 設為 zero | 每個 probability 只有一個可審計來源；任何說 Jev schema 的 reader 都可插同一 slot |
| Point-in-time stores | 共享 memory／backtest 中以慣例防 look-ahead | 三個 store（weekly blackboard、event store、filing-profile store）皆 single writer、immutable、time-stamped、帶 taint；agent a 在 t 只讀 `τ_avail(r)<c_t`、writer 在授權集合、taint ≤ clearance 的 records | look-ahead 與 untrusted text 進入 trusted agent state 由 construction 排除，不靠 convention |
| Trust-partitioned learner + band bound + decision card | 把 text 與 code features 塞進一個 pooled model，模型輸出直接影響倉位 | trusted logit `L^T` 與 text logit `L^W` 分離，`L^W` 以 `L^T` 為 offset；decision logit `L=L^T+clip_{κ_t}(L^W)`，`κ_t=μ s^T_t`、`μ=2`；band 為 trusted rank ≤ K+J（K=20、J=10）；caps 10% per name、30% per industry、40% turnover | text 只能在 trusted data 已看好的邊緣 ranking 中起作用；decision card 把 moved position 歸因到 item，這是 holistic reader 做不到的 |

### ⚡ Eureka

一句話：**把「LLM 是否讀懂新聞」改成「在 publication 後還剩多少可交易信息」，再用 code 的影響上界把 text 的傷害與 reader 的能力解耦。** 直覺是：Web news 常常在價格反應之後才出現，announcement-window 的高相關可能只是「跟著已發生的 move 走」；真正要測的是從 publication 起算的 criterion，以及在任何 reader 下 text 最多能把倉位推多遠。這個設計讓「reader 好不好」與「text 能不能亂動倉位」變成兩個可分開檢定的問題。

### 信息流

> **信息流（ASCII）**  
> `untrusted Web news ──┐`  
> `semi-trusted 8-K/年報 ──┼──► point-in-time stores (single writer, immutable, taint)`  
> `trusted prices/filings/macro ──┘                │`  
> `                                                ▼`  
> `news / policy / profile / technical / fundamentals / macro agents`  
> `                                                │ typed questions, no free text`  
> `                                                ▼`  
> `Jev-schema interface (emu / Jev / NG / VERB)`  
> `                                                │ probabilities only`  
> `                                                ▼`  
> `code decision layer: trust split, capped text logit, band, sizing, risk rules`  
> `                                                │`  
> `                                                ▼`  
> `orders fill at next open; decision cards attribute every move`

## §2 · 數學層

📌 **Napkin Formula**：

\[
L_i = L^T_i + \operatorname{clip}_{\kappa_t}(L^W_i),\qquad \kappa_t=\mu s^T_t,\qquad \mu=2
\]

\[
|L_i(\mathcal{A}) - L^T_i| \le \kappa_t,\qquad |L_i(\mathcal{A}) - L_i(\varnothing)| \le 2\kappa_t
\]

`L^T` 由 24 個 trusted features（code factors + technical/fundamentals/macro judgments）組成；`L^W` 由 12 個 text features（news agent、sentinel、policy agent）組成，並以 `L^T` 為 offset 拟合，讓 text 只解釋 trusted data 未解釋的部分。`s^T` 是 `L^T` 的 cross-sectional standard deviation，有 0.01 下限。band 是 trusted rank ≤ K+J（K=20、J=10），protected 是 rank ≤ K−J；text 只能 admit band 內的名額，不能把 band 外拉進來，也不能在沒有 trusted anchor 時移除 protected holding。`clip_c` 在 news impact 中另有 `clip_3` 與 `clip_{3/2}`，用來防止單一 cluster 或 item flood 主導。訓練是 online logistic：預測 holding week open-to-open return 是否超過 universe median；features 以 trailing 52 weeks 標準化，slopes 有 zero-centred Gaussian priors、standard deviation 0.04，observations 以 half-life 52 weeks 衰減，每個 decision 從 prior refit。複雜度主要在每週重訓與 per-item LLM 呼叫；原文稱 full run 2.4 hours wall-clock、requests 在 1.84 s 內返回的比例為 nineteen in twenty、reading 成本 \$1.03 per weekly rebalance，每增加 ten names 加 390 requests 與 \$0.091 一週。

## §2.5 · 帶數字走一遍（Worked Example）

以下為機制走查（示意），不引入論文未披露數值；括號內參數均為原文逐字，未披露量保留符號。

1. 某週 cutoff `c_t` 為 16:00 ET，universe 為 100 名；code 凍結 point-in-time context：daily bars、news items created/last updated before `c_t`、XBRL facts、macro vintages、store records。
2. 對某名 `i`，trusted learner 給 `L^T_i`；text learner 以 `L^T` 為 offset 給 `L^W_i`。`κ_t=μ s^T_t`，`μ=2`；`s^T_t` 未披露，故 `κ_t` 不代入數值。決策 `L_i=L^T_i+clip_{κ_t}(L^W_i)`。
3. 若 `L^W_i` 超過上界，clip 後為 `κ_t`；因此 `|L_i−L^T_i|≤κ_t`，且 `|L_i(A)−L_i(∅)|≤2κ_t`。這是 P4 的 bounded logit。
4. 選股：band 是 trusted rank ≤ K+J，K=20、J=10；protected 是 rank ≤ K−J。文字只能讓 band 內的名額進入；protected 沒有 trusted anchor 不可移除。原文 attack run 中 entrant 可到 trusted rank 30 = K+J，74% 的 290 個 protected held names without anchor 被 halved、none excluded。
5. 權重：`v_i≤v^T_i`；caps 10% per name、30% per industry、40% turnover；sentinel 到 0.8 halves，earnings 在 2 sessions 內 ×0.75。若 attack 把 text logit 推到 cap，decision card 仍把變動歸因到 planted item；原文稱 decision-card ledgers 加總到 decision 的誤差為 `3×10^{-17}`。

## §3 · 數據層

- **Universe / 頻率**：100 個 S&P 500 members，以每月前 63 sessions 的 median dollar volume 取最大者，按月從 dated constituent lists 重排；weekly long-only portfolio，無 leverage、無 short，name/industry/turnover caps。cutoff 是每週最後 session 的 16:00 ET，orders fill at next open。
- **News**：Benzinga 經 Alpaca news API 的 headline + summary；evaluation window 100,906 items。publication time 用 vendor record 的 creation time，並在 §4.2 對 SEC acceptance times 做檢查。news agent 每 stock-week 讀 40 個最近 items（最多 3 tickers），screening 後 per-item 問 13 題。
- **Evaluation windows**：`W_clean` 為 2024-07-05 到 2026-09-11 的 115 weekly decisions，起點在 emu LLM 的 2024-06-01 cutoff 與 buffer 之後；`W_release` 為該 LLM release 後的 73 decisions。H2 frame 是 27,556 個 top event type 非 stock activity/opinion 的 kept items，over 115 weeks；74,314 item–firm pairs 進入 per-item questions，emu 保留 31,848。Jev descriptive panel 是 release 後 549 items、over 3 weeks。
- **成本與容量**：trades cost 15 bps per side；reading 成本 \$1.03 per weekly rebalance。市場衝擊、ADV 容量、borrow、滑價分布所讀文本未見，圖表／附件待核。
- **樣本外與泄漏控制**：hash-committed question bank、prompts、model pins、decision layer、attack templates、seeds、analysis code；blinded states（Company-h、relative times、bucketed numbers）、date-only probe、re-identification probe、post-release weeks。Jev 與 NG 的 cutoff 在 Table A1 標 undocumented，故 Jev rows 只作 descriptive；NG 是 registered reference。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | https://github.com/yebof/track-or-follow；含 code、question bank、prompts、model pins、hashed-identifier judgment tables、labels、pre-registration、scripts |
| Checkpoint / Model pin | emu/VERB/holistic reader：gpt-4.1-mini-2025-04-14；NG：flock-io/this-that-model-1.0@3d927195；Jev：jev-1.13.0 |
| License | 所讀文本未見；圖表／附件待核 |
| 複現難度 | 中高。演算法與決策層可從 repo 重跑，但 news 是 licensed 且 article text 不釋出；需要 Alpaca/Benzinga 與 pinned LLM API。 |
| 數據可得性 | 程式碼、question bank、prompts、model pins、hashed judgment tables、labels、pre-registration、scripts 可得；licensed news 與 article text 不釋出，decision cards 只帶 hashed item identifiers。 |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄實際多為基線／對照，不是作者宣稱的 SOTA；不同樣本、不同窗口、不同成本配置不直接算 Δ。Δ 只保留原文逐字給出的差值，或留空。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| H2 frame，27,556 items，115 weeks | weekly Spearman ρ(E[effect], AR[0,+1]) | 未披露（無同幀基線） | emu 0.119 |  |
| VERB 取樣 2,000 items | pooled Spearman ρ(E[effect], AR[0,+1]) | 基線 VERB 0.097 | emu 0.100 |  |
| 8-K earnings vs Item 2.02，emu 7,258 pos.／VERB 532 pos. | AUC | 基線 VERB 0.718 | emu 0.833 |  |
| 8-K earnings vs Item 2.02，NG 130 pos.／emu 7,258 pos. | AUC | 基線 NG 0.830 | emu 0.833 |  |
| Publication-aligned，first trade after publication | Spearman | 基線 FinBERT 0.004 | emu 0.023 |  |
| H3，113 learned weeks | weekly IC（combined logit vs holding week relative return） | code-only twin -0.008 | full system -0.024 | -0.016 |
| A1–A7 pooled，attacked stock-weeks | ASR (%) | holistic reader 0.1 | per-item reader 10.5 | 10.4 |
| Portfolio，system book vs equal-weight universe | Sharpe | equal-weight universe 0.97 | system book 0.38 |  |

**解讀**：H2 的 0.119 是 criterion validity，不是可交易 alpha；同一批 items 上 emu 0.100、VERB 0.097，原文的 paired difference 是 -0.003 [-0.029, 0.023]，不能說 typed 一定勝過 verbalized。真正決定交易視角的是 publication-aligned 0.023 [0.005, 0.041]；原文還給出 72% earnings items 在 filing 之後出現、named contributors 19.0% 報導價格移動（desk 0.6%），以及從 first trade after publication 起關聯落到 0.023。H3 的 ΔIC -0.016 [-0.029, -0.002] 表示在此 learner/decision layer、weekly horizon 上沒有偵測到 rank information over code-only features；full system IC -0.024、code-only twin -0.008，兩者都不可偵測。Portfolio 的 system book Sharpe 0.38 低於 equal-weight universe 0.97，MDD - 18.0 對 - 18.7；原文的 deflated Sharpe 是 0.41 over 73 registered trials（71 run），不能把 multi-agent 當成穩定 alpha。Attack 面，bound 確實 holds：A1/A3/A4 皆 0.0 [0.0, 2.1]*、A5 4.0、A2 22.5 [16.0, 29.3]、A6 31.5 [24.5, 38.6]；但 H1 未支持 per-item guarded decomposition 降低 ASR，per-item 10.5 反而高於 holistic 0.1（difference 0.104 [0.081, 0.127]），且 holistic 的低 ASR 部分反映其 learned text slope 只有 1% 週為正，slope flipped 後 ASR 21.9%（exploratory）。這是系統比較，不是「holistic reader 普遍更安全」的因果結論。此頁研究假說尚待外部樣本驗證。

## §6 · 失效與隱含假設

### 論文自述限制（可定位作者限制）

- **§6 Scope**：方法只 instantiate 一個 pinned LLM 作 confirmatory reader、Jev 作 descriptive reader、NG 作 registered reference；one feed 的 headlines/summaries、100 large caps、weekly decision with next-open fill。作者稱方法可延伸到任何有 public、time-stamped record 的 Web text，但本文只測金融新聞。
- **§4.4**：H3 bounds what this learner and decision layer extract, not what the judgments contain；frozen learner 在 commit 前只 recovered 0.37 of a planted IC of 0.10。
- **§4.5**：bound limits how far text can move a position, not whether a planted report can move a name inside the band；band 內仍由 reader 決定。
- **§4.3**：registered probe rule left window start unchanged（AUC 0.488, p=0.896）；有些 release 前窗口不同於 one half，作者把 claim 放在 post-release estimate，recall 不能到達。
- **Table A1**：只有 emu 的 LLM documented cutoff；NG 與 Jev 的 cutoff undocumented，Jev rows 是 descriptive。
- **§6 reporting rule**：作者主張 Web-text reader 評測應報告 placebo window、pre-open control、post-release estimate、re-identification with abstentions、share of items that report the outcome、以及 publication-aligned criterion next to announcement-window criterion。

### 推斷的隱含假設（本頁推斷）

- **Regime 依賴**：window 是 2024-07-05 到 2026-09-11；原文引用 fast-assimilation 研究，next-day drift of LLM news scores 到 early 2024 的 annualized Sharpe 1.22，早於本文 window。若 2027 後 Web news 的 publication lag 或市場反應速度改變，publication-aligned 0.023 不保證持續。
- **容量**：100 most traded S&P 500、weekly、next-open fill、15 bps per side；市場衝擊與 ADV 容量未披露，不能從 weekly 成本推大規模可容量。
- **成本**：15 bps per side 是固定假設；borrow、short、impact、queue priority 未測。long-only 與 caps 限制了結論外推。
- **數據泄漏**：hash-commit、blinding、post-release weeks 降低 recall，但 vendor timestamp 與 SEC acceptance 的差異、price-report items 的 undercount（audit precision 97% [92, 99]、miss 4%、recall 65% [48, 88]、true share near 10.6%）仍是測量誤差；Jev/NG cutoff undocumented 使跨 reader 比較只能是 descriptive。
- **Survivorship**：universe 從 dated constituent lists 每月重排，但 delisting 處理所讀文本未見；不能把 100 large caps 的結果直接推到小型股或退市股。
- **單一樣本／消融**：H1 是 system comparison，each arm keeps its own learned text slope，不能把 ASR 差異全部歸因於 per-item decomposition；leave-one-agent-out refits 的 rank IC 變化從 -0.007（policy）到 +0.015（fundamentals），區間都覆蓋 zero。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Lopez-Lira & Tang 2026（GPT headlines vs initial reaction） | 本文加 publication-aligned、post-release、influence bound；不是只測 announcement window | 未披露 | 期刊 2026 |
| Kirtac 2026（supervised LM probabilities, daily） | 本文 weekly decision horizon、typed reader、attack bound；不是 daily alpha | 未披露 | Preprint |
| TradingAgents（Xiao et al. 2024） | 本文 agents 不交換訊息、只共享 point-in-time stores；多智能體是為了審計與 bound，不是辯論 | 未披露 | Preprint |
| FinBERT tone | 本文用 event-type 8-K、typed judgments、publication-aligned；FinBERT 是 reference reader | 未披露 | 期刊 2023 |
| NG / Jev typed models | 本文把 reader 放進同一 interface，用 market records 與 8-K 評；Jev hosted、NG open non-generative | Jev hosted；NG 未披露 | Model pins |
| Code-only twin | 本文 H3 的直接對照；無 text features | repo 可重跑 | 本文實驗 |

🎤 **Interview Tip**：正確答是「announcement-window 0.119 只證明 reader 讀到事件；publication-aligned 0.023 才是決策 horizon 的候選信息；H3 在此 learner/decision layer 沒有 rank 增量；bound 限制倉位移動，不限制 reader 在 band 內被 fabricated fact / press release 騙；H1 未支持 per-item 降低 ASR。」錯答是「0.119 就是 alpha，可直接交易」、「multi-agent 必然抗攻擊」、「holistic reader ASR 0.1 所以 holistic 普遍更安全」——後者忽略其 learned slope 只有 1% 週為正，flipped 後 ASR 21.9%。

### 可證偽預測（帶日期）

在 2026-09-11 之後的樣本外重跑，若 announcement-window 與 publication-aligned 的差距消失，或 H3 的 ΔIC 轉正且區間排除零，則本文「Web news 多為跟隨、weekly horizon 無增量」的論斷被削弱；若差距仍在且 publication-aligned 接近零，則支持其 reporting rule。此為本頁預測，尚待驗證。

## §8 · For the Reader

- **因子研究員**：把 0.119 當 criterion validity，不要當 factor IC；先看 publication-aligned 0.023 與 price-report items 的 share（audit near 10.6%），再決定 text 是否值得進每週 model。8-K earnings AUC 0.833 可用來校準 event-type mapping，但 deal 0.611、management 0.557 顯示 coarse legal categories 的 limit。
- **高頻執行**：本文是 weekly next-open fill、15 bps per side、\$1.03 weekly reading cost；81% 的 AR[0,+1] window 在 next-open fill 前關閉，publication 後 first trade 的關聯 0.023。若要做 intraday，需另測 latency、queue、impact；所讀文本未見。
- **組合配置**：system book Sharpe 0.38 vs equal-weight universe 0.97、MDD - 18.0 vs - 18.7；不要用 multi-agent 故事覆蓋 portfolio 表現。可把 band、caps、decision cards 當 risk attribution 與 poisoning 壓力測試工具。
- **LLM-agent 研究者**：typed interface + no free text + single-writer stores 是可搬的架構；H1 未支持 per-item 降低 ASR，但 decision cards 追到 88% [80.8, 94.2] 的成功攻擊來源，這是 holistic reader 沒有的 attribution。Attack 中 A2 22.5、A6 31.5 說明「bound 內仍可被騙」。
- **RL 策略**：band-guarded selection 可視為 action mask；`2κ` 是 text 對 logit 的硬上界，caps 與 protected 規則是 environment constraint。Reward 設計、training stability 所讀文本未見。
- **研究學生**：從 repo 的 question bank、prompts、pre-registration、14 logged deviations 重跑；注意 licensed news 不釋出、Jev/NG cutoff undocumented、H2 frame 由 emu answers 選出（exploratory sensitivity 0.086 on all 53,698 item–firm pairs、0.062 without price reports、0.113 on 31,848 kept pairs），選樣會抬高估計但不創造關聯。

## References

- 原論文：Feng, Gan, Pan, Liang, Wu, Xu, Liu. *Track or Follow? A Jev-Compatible Multi-Agent Trading System When Web News Lags the Market.* arXiv:2611.10414v1, 2026-11-07. https://arxiv.org/abs/2611.10414v1
- Code / pre-registration / question bank：https://github.com/yebof/track-or-follow
- Lineage：Lopez-Lira & Tang 2026（ChatGPT headlines vs initial reaction）；Kirtac 2026（supervised LM probabilities, daily）；Xiao et al. 2024（TradingAgents）；Sun et al. 2026（Jev typed decision model）；Cheng et al. 2026（this-that-model-1.0 / NG）；Tang & Zheng 2026（typed-model evaluation audit）；Paleka et al. 2026（forecaster evaluation pitfalls）；Ye et al. 2026（Alpha Illusion）；Bailey & López de Prado 2014（Deflated Sharpe Ratio）。