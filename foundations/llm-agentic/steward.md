<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=多智能体博弈 autonomy=Agent自主演进 -->

# Steward 解構（Steward）

> **來源發布**：2026-10-22 · （無 venue） · arXiv [2610.30557](https://arxiv.org/abs/2610.30557)
> **arXiv 原文**：[Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core](https://arxiv.org/abs/2610.30557v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文本另類／日頻波段／生成式大模型／多智能體博弈／Agent 自主演進」；它解的不是預測 alpha 的 prior gap，而是「自演化 agent 如何在不改寫自身風控的前提下，把改動限制在可審計、可回放、可回滾的外圍」——把 self-evolution 邊界從 prompt 搬到 type system 與 deterministic gate。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `多智能体博弈` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** Steward 用九個角色化 LLM agent 做美股 paper account 的日頻交易，把 order arithmetic、blocking risk rules 與兩個 discipline agent 的 prompts 設為不可變核心，讓 quarterly reflector 只能提出 typed PromptLearning。核心 trick 是 proposal type 的 Literal 不能命名 Risk Manager／Position Reviewer，且每個 learning 要過 deterministic gate、再以 recorded inputs replay 驗證。這直接打在「Agent 自主演進」軸的痛點：自改 prompt 的 agent 也能改掉護欄；Steward 讓演化可逆、可記錄，但代價是只動外圍。關鍵實證：五個月 live deployment（102 trading days, 1,632 LLM calls）中，系統把 24.6% sector drawdown 控制到 0.3% loss；單模型在 replay 中五日超額 -1.35 points，聯合設計則不顯著。

**X-Ray.** 放回五軸 Pareto，Steward 不是把 LLM 選股推到更高 alpha，而是把「可自我演進」與「不可被改寫的風控」拆成兩個型別層：外圍 prompt 可 append／retract，核心 order arithmetic、blocking rules、discipline prompts 在 code 與 Literal 之外。這解了舊工程坑 F1–F4：自評樂觀、紀律侵蝕、改動不可審、learning 惰性；做法分別是 code grading、type-level allow-list、gate log + git revert、replay。它打不開的 envelope 也很清楚：single configuration、paper account、one rising window、overlapping horizons、simulations ignore fill risk、two quarterly cycles one applied；alpha not identifiable over 102 days。對量化讀者，可搬的是 evaluation harness 與 safety boundary，不是某個 signal；先補 execution cost、universe survivorship 與多 regime 驗證，再談 live alpha。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作典型 | Steward | 工程含義 |
|---|---|---|---|
| 自演化邊界 | Reflexion／Self-Refine／Promptbreeder 由模型自評、自改 prompt；無型別 allow-list | 提案是 PromptLearning；target 為六個 editable agents 的 Literal；Risk Manager／Position Reviewer 不在 Literal | 把「不能改什麼」變成 schema 不可表達，而非 prompt 勸阻 |
| 反饋信號 | 自評、verbal reinforcement 或 backtest | 每晚 code 用 realised fills／prices grade；quarterly reflector 只讀 deterministic digest | 降低 self-grading 樂觀與 backtest leakage |
| 驗證 | 以回測或主觀相信 learning 有效 | replay recorded inputs before／after，量測結構性輸出差異與 baseline re-run noise | 把「應該有幫助」換成「在記錄輸入上改變了什麼」 |
| 記憶 | FinMem 類 layered memory；FinCon 類 manager–analyst hierarchy | 七層共享記憶由 code 組裝；agents 不共享 conversation history | 每個引用數字來自 code，不是另一個 agent 的 prose |
| 風控 | LLM risk manager 或 prompt 約束 | deterministic hard risk engine 先擋；LLM Risk Manager 只能 reject／modify／scale，fail closed | 判斷與算術分離 |

### 1.2 ⚡ Eureka

**把自我演化的邊界寫進型別系統：proposal type 不能命名核心，所有演化信號由 code 從 outcomes 計算。** 直覺是：如果 agent 只能 append 一段短 prompt 到 allow-list 的 agent，且不能碰 order constructor、blocking rules、Risk Manager／Position Reviewer prompts，那它再怎麼「學壞」也改不了下單算術與硬風控；gate 再把長度、絕對詞、重複、FIFO、每季變更數、git revert 一層層鎖上。

### 1.3 信息流

> Web/API：nine RSS feeds／SEC EDGAR／nine FRED series／market data<br>
> → News／Earnings／Macro／Technical（typed reports + mandatory CoT）<br>
> → Shared memory L1–L7（code 從 record 組裝）<br>
> → Portfolio Manager → TargetPosition（weight ≤20%, thesis_invalid_if）<br>
> → Order constructor → Hard risk engine → LLM Risk Manager → Executor → Brokerage API<br>
> → fills／P&L → Evening Analyst（code grades + 7-step CoT）→ memory<br>
> → Quarterly Meta Reflector → PromptLearning[] → deterministic gate → git commit<br>
> → Replay harness（recorded inputs, before/after）→ verify／next invariant

## §2 · 數學層

📌 **Napkin Formula**  
`d_t = S_π(x_t)`；`π' = Gate(Reflect(R_T))`；`π_0 ⊂ π` 且 proposal type 不能命名 `π_0`。  
`R_T = {(x_t, d_t, y_t)}`；`y_t` 由市場事後給出，不由 `S` 自評。

**直覺**：決策是可檢核物件，演化只改外圍 prompt，核心在 code 與 type。 **複雜度／成本**：每日六 sessions；morning session median ten minutes、20-minute hard kill；live 102 trading days、1,632 LLM calls；logged API cost $455，約 US$4.4 per trading day。 **loss／訓練細節**：本文不是 fine-tuning，沒有 gradient 或 reward model；演化是 code grading + prompt learning，training loss 未披露。

## §2.5 · 帶數字走一遍（Worked Example）

以下均為**假設／示意**，只用原文規則數值展示機制，不是論文實證結果。

1. 假設 PM 對某 symbol 給 `target_weight = 20%`（原文 type cap），entry price 記為 `P`，equity 記為 `E`，stop 距 entry 為 `2 ATR`（原文 default）。
2. Constructor 算 risk-budget 股數：`0.5% × E / (2 ATR)`（原文 risk budget 0.5% of equity at stop）；target-weight 股數：`20% × E / P`；取小者。
3. 若 execution 時價格離 entry 超過 `5%`，skip（原文）；若 execution 後 R/R 低於 `1.2`，skip（原文）；Risk Manager 另以 floor `1.5` audit（原文）。
4. 硬規則先跑：cash negative、position 超過 `20%` equity、sector 超過 `40%`、net exposure 超過 `90%`、no stop、或 `3%` daily loss 後買入，任一觸發即 reject BUY（原文）。
5. 演化示意：假設 reflector 提出一段超過當時 `200` 字元上限的 learning，gate 以長度 invariant 拒絕並記錄 reason（原文 Q2 即此類）；若在 `300` ceiling 下通過，仍需 replay 看 trigger／control 是否有行為改變。
6. 示意結論：核心算術與硬規則先於 LLM；learning 只能改「想要什麼」，不能改「怎麼變成單」。

## §3 · 數據層

- **市場／頻率**：US equity portfolio，日頻；six sessions per trading day；只有 morning session 可開倉；intraday 每 30 分鐘 sell-only circuit breaker；evening grade；quarterly reflect。
- **資料來源**：nine RSS feeds、SEC EDGAR 10-Q／10-K、nine FRED series、daily price／sector／valuation data for 80 US stocks and 21 ETFs。
- **Live 窗口**：2026-04-08 至 2026-09-10，102 trading days；brokerage paper account funded with US$100,000；619 pipeline runs、1,632 LLM calls、95% 有 complete inputs、438 broker round trips、492 orders、746 fills；logged API cost $455，約 US$4.4 per trading day。
- **樣本外**：每個 decision 在 outcome 出現前記錄，live record 是 forward；原文稱此避開 backtest 中難以排除的 pre-training leakage。  
- **回測**：2015–2026 deterministic core only，LLM-free weekly entries，universe 的 80 stocks，5 bp per side；universe is from 2026，因此原文只主張 variant 之間比較有意义。
- **容量假設**：paper fills simulated against live quotes，market impact is not modeled；容量／流動性假設未披露。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| [github.com/yebof/quant-agent](https://github.com/yebof/quant-agent) | 未披露（所讀文本未見模型權重 bundle） | 未披露（所讀文本未見） | 中高（本頁推斷：需 brokerage paper API、兩家 LLM vendor、replay harness；原文未給逐步復現腳本細節） | 公開 Web data；code、prompts、replay harness 稱 open source；recorded trace 與 scripts 稱 released；實際檔案完整性待核 |

## §5 · 評測 / Benchmark

下表把所讀原文中可定位的基線／對照逐行列出；「前SOTA」欄只放基線本身數值，不合成區間。Δ 一律為「本方法 − 基線」，只放在末欄；正文不重複另算差。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| Live US equity, 2026-04-08–2026-09-10 | Return (%) | SPY +12.4 | Live portfolio +5.0 | -7.4pp |
| Live US equity, 2026-04-08–2026-09-10 | Sharpe | SPY 2.28 | Live portfolio 1.39 | -0.89 |
| Live US equity, 2026-04-08–2026-09-10 | Beta | SPY 1.00 | Live portfolio 0.16 | -0.84 |
| Backtest 2015–2026, random entries median of 50 seeds | Max DD (%) | No core -37 | Full core -28 | +9pp |
| Backtest 2015–2026, momentum entries | Max DD (%) | No core -61 | Full core -33 | +28pp |
| Backtest 2015–2026, momentum entries | CAGR (%) | No core +61 | Full core +41 | -20pp |
| Backtest 2018 crisis, momentum entries | Drawdown (%) | No core -25 | Full core -18 | +7pp |
| Backtest 2020 crisis, momentum entries | Drawdown (%) | No core -36 | Full core -30 | +6pp |
| Backtest 2022 crisis, momentum entries | Drawdown (%) | No core -27 | Full core -18 | +9pp |
| Backtest 2026 semiconductor window, momentum entries | Drawdown (%) | No core -46 | Full core -27 | +19pp |
| Replay 20 recorded mornings | Excess 5d (pp) | Single agent：低於 SPY 1.3 pp | Deployed design：低於 SPY 0.5 pp | +0.8pp |
| Replay 20 recorded mornings | Excess 20d (pp) | Single agent：低於 SPY 0.8 pp | Deployed design +1.3 | —（由前兩欄比較） |

**解讀**：live vs SPY 的 Return Δ 為負，但原文把 gap 主要歸給 execution 與 exposure，而非選股能力：realised gross exposure 56% vs intent 79%，180 of 492 orders cancelled or expired unfilled，hard rules trimmed buys，intraday stops fired where close-based simulation held；live beta 0.16 vs intent 0.64。論文自己也說 alpha is not identifiable over 102 days。半導體窗口 SMH fell 24.6%、QQQ 10.3%、SPY 2.0%，portfolio lost 0.3%——這是單一窗口的 containment 證據，不是跨 regime 因果。  
**真 capability 與可疑部分**：backtest core alone 在 50 random entry policies 下全部降低 max drawdown（median 37% → 28%），momentum 61% → 33%，2018／2020／2022 至少 49 of 50；這是 deterministic rules 的隔離較強者，但 universe from 2026、5 bp per side、LLM-free entries，仍有 selection 與成本假設。Architecture replay 的 single agent 是不同 trader：5.8 names vs 3.7、41% gross vs 19%、reversals 0.25 vs 0.10；其 5d 相對 SPY 超額分別為虧損 1.3 pp 與虧損 0.5 pp；20d 則為虧損 0.8 pp 與正 1.3 pp，但 intervals are wide，原文說 replays 不能按報酬排序變體。Grounding audit 的 99.9% 是 trace 品質，不是 alpha；strict 96.1%、placebo 22%、226 ungrounded、three candidate fabrications。Cost 端只看到 API $455、$4.4/day；paper fills 不計 market impact，live net-of-cost alpha 未隔離。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§6 Limitations and next steps**：one configuration traded on a paper account through one rising window；overlapping forward horizons；simulations that ignore fill risk；two quarterly cycles, one applied；replays of 14–20 inputs 只能暴露行為差異，不能按報酬排序變體。
- **§5.5**：alpha is not identifiable over 102 days；one window holds one drawdown。
- **§5.4**：Q3 learning 無法從 October record 隔離，因 operators 也改了其他 prompts 與 universe；只能靠 replay。
- **§5.1**：paper fills simulated against live quotes，market impact is not modeled。
- 其餘為本頁推斷；圖表／附件待核。

### 6.2 推斷的隱含假設（明標推斷）

- **Regime 依賴（推斷）**：live window 是 semiconductor-led rally；core backtest 覆蓋 2018／2020／2022／2026，但 2026 semiconductor window 同一 rules under momentum still lost 27%，所以 core 是 bound loss，不是跨 regime alpha。  
- **容量（推斷）**：paper account US$100,000；paper fills 無 market impact；容量／滑價未驗證，不能外推到更大資金。  
- **成本（推斷）**：API $455、$4.4/day 是 LLM 成本；backtest 5 bp per side；live paper fills 不計 impact，net-of-cost live alpha 未隔離。  
- **數據泄漏（推斷）**：live record forward 避開 live decision 的 pre-training leakage；但 2015–2026 core backtest 是 LLM-free，不能拿它證明 LLM 無泄漏。預訓練截止日、tokenizer、適配、topic 擬合的時間邊界：所讀文本未見，圖表／附件待核；不可把預訓練截止日套到適配後權重。  
- **Survivorship（推斷）**：universe 的 80 stocks／21 ETFs 以 2026 名單回看 2015–2026；原文只說 variant 比較有意义。這是 selection／survivorship 風險，不是「輔助數據缺失」；缺失值本身不等於 survivorship。  
- **單一樣本相關（推斷）**：evening loop 的 ρ=0.34（p=0.01）與 ρ=0.22（p=0.05）是單一窗口關聯，Holm-adjusted p=0.08 for the strongest；不能寫成所有市場的必然因果。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Reflexion／Self-Refine | 自評／自改 vs code-graded outcomes + gate | 未披露（所讀文本未見） | related work |
| Promptbreeder／OPRO | prompt 優化器 vs typed PromptLearning + deterministic gate | 未披露 | related work |
| FinMem／FinCon | layered memory／manager–analyst hierarchy；historical eval | 未披露 | related work |
| QuantAgent | self-improving knowledge base；backtest learning | 未披露 | related work |
| TradingAgents／FinAgent | trading-firm roles／tool use；historical eval | 未披露 | related work |
| AutoGen／MetaGPT | natural-language coordination vs schema-enforced typed CoT | 未披露 | related work |
| Steward | invariant core + live forward + replay | 是（code／prompts open source；license 未披露） | arXiv preprint |

🎤 **Interview Tip**  
**正確答**：Steward 的安全不是靠 prompt 說「不要改風控」，而是 (i) proposal schema 的 target Literal 排除 Risk Manager／Position Reviewer；(ii) order arithmetic 與 blocking rules 在 code；(iii) gate 的 twelve invariants + git revert；(iv) replay 驗證 learning 是否真的改變決策。  
**錯答**：把 Steward 說成「多 agent 投票選股」或「LLM 自己反思就會安全」；或把 live +5.0% 當 alpha 證明（beta 0.16、exposure 56%、alpha not identifiable over 102 days）。

### 7.1 可證偽預測（本頁研究假說，尚待驗證）

2026-12-31 前，若作者在下一個 quarterly cycle 把 outcome-scored replay 加進 gate invariant，則應觀察到：Q3 類 trigger evenings 的 prescribed TAKE/WAIT／Blocker 行為在 replay 中出現，control evenings 不出現；若 trigger 與 control 無差異，或 learning 仍因缺 input count 而 inert，則「semantic check 可由 replay 補上」的主張不成立。

## §8 · For the Reader

- **因子研究員**：把 Steward 當 evaluation harness 靈感——用 recorded decisions 做 layered policy simulation，而不是重跑回測；但別把 PM intent +12.2% 當可交易 alpha，因 live execution gap 是 56% vs 79%。
- **高頻執行**：本文 envelope 是日頻、six sessions、intraday 30-min sell-only；不解決 queue position、market impact、fill risk。要搬之前，先把 execution agent 放進 core 的建議與成本模型補上。
- **組合配置**：可借 invariant core（cash-only、20% position cap、40% sector cap、90% net、3% daily loss breaker）作 guardrail；但 2026 semiconductor window 同一 core under momentum still lost 27%，所以 core 是 bound loss，不是 alpha。
- **LLM-agent**：type-level allow-list + deterministic gate + replay 是可移植 pattern；注意 gate 是 syntactic，不是 semantic；Q3 learning 因缺 input count 而 recurring misses 未變（25 of 64 against 26 of 64）。
- **RL 策略**：Steward 不是 RL；沒有 reward model 或 gradient，只有 code grading + prompt learning。若把 replay 當 offline evaluation，注意 14–20 inputs 不能 rank variants by return。
- **研究學生**：先讀 §3 的 R1–R5 與 §4.3 的分層，再讀 §5 的 trace；limitations 在 §6，別把 single rising window 外推成通用結論。

## References

- Feng, Y., Dai, J., Pan, L., Lau, N., Xu, J., Liu, Y. 2026. *Governed Self-Evolution for Autonomous Investing: A Multi-Agent LLM System with an Invariant Core.* arXiv:2610.30557. https://arxiv.org/abs/2610.30557v1
- 代碼與 prompts：https://github.com/yebof/quant-agent
- Lineage：Shinn et al. 2023（Reflexion）；Madaan et al. 2023（Self-Refine）；Wang et al. 2024a（Voyager）；Fernando et al. 2024（Promptbreeder）；Yang et al. 2024（OPRO）；Park et al. 2023（Generative Agents）；Yu et al. 2024a（FinMem）；Yu et al. 2024b（FinCon）；Wang et al. 2024b（QuantAgent）；Xiao et al. 2024（TradingAgents）；Zhang et al. 2024（FinAgent）；Greshake et al. 2023；Debenedetti et al. 2024（AgentDojo）；Zhan et al. 2024（InjecAgent）；Zhang et al. 2025（Agent Security Bench）；Beurer-Kellner et al. 2025；Wu et al. 2025（IsolateGPT）；Costa et al. 2025；Liu et al. 2024。
