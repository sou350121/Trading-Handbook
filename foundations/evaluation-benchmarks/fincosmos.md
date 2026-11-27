<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=组合执行优化 autonomy=Agent自主演进 -->

# FinCosmos 解構（FinCosmos）

> **來源發布**：2026-11-05 · （無 venue） · arXiv [2611.07464](https://arxiv.org/abs/2611.07464)
> **arXiv 原文**：[FINCOSMOS: An Open-World Evaluation Paradigm for Agents in Finance](https://arxiv.org/abs/2611.07464v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文本另類 × 日頻波段 × 生成式大模型 × 組合執行優化 × Agent 自主演進」——它不產 alpha，它給 agent 一條可被程式否決的交付鏈（隱藏意圖 → 自取 Web 數據 → 自建時點回測 → 可重算交付物），補的是「既有金融 benchmark 各把意圖／數據／時間線關掉一端，於是靜默失敗被關掉而不是被測到」這個 gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `组合执行优化` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把金融 agent 評測從「給定任務＋給定數據＋給定截止日」改成三者全開，用組合語法採樣隱藏策略、由 simulated user 只在被問時披露。②核心 trick 是免參考答案的程式化驗收：數據溯源、point-in-time、指標可從 trade log 獨立重算，加上 15 項否決維度，只有「用戶簽核且零否決」才算 accept。③對 Agent 自主演進這條軸★最關鍵：它把「用戶滿意」與「交付正確」強制解耦，讓靜默失敗變成可觀測量。④關鍵數字（逐字）：150 sessions 中 no session is accepted；pre-veto quality ranges only from 20.2 to 37.2 out of 100。

**X-Ray.** 放回五軸 Pareto，FinCosmos 不是效率前緣上的點，是把評測座標系整個平移：它把「LLM 能不能做金融」換成「LLM 交付的金融研究能不能被第三方重建」，因此 alpha 那一軸從預測收益退化成組合執行鏈的完整性（六元件語法裡 execution 與 transaction cost 各佔一元）。它真正解決的舊工程坑有三個：一是 LLM-as-judge 讀報告讀不出「沒有 trade log」這種缺席型錯誤；二是時點正確性長期靠 pipeline 假設而非交付物驗證；三是固定題庫一旦被訓練污染就報廢。它的 envelope 也很清楚——語法只覆蓋 long-only US equity + 固定成本模型，題目難度以各選項難度 δ∈{0,…,3} 加總計（d(b)=Σ_k δ(o_k)），而非以可實現 alpha 計，這 150 場測試的 pre-veto quality 觀測範圍為 20.2 至 37.2，這意味著它擅長定位失敗、不擅長區分「好策略」與「沒那麼好的策略」。對量化讀者的意義：把它當成交付前的 CI gate（可重算性 + 時點 + 溯源）比當成選模型榜單更有價值；任何用它排名 agent「選股能力」的讀法都是誤讀。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 既有 benchmark 的封閉方式 | FinCosmos 的開放方式 | 對應被「移除」的靜默失敗 |
|---|---|---|---|
| 意圖（intent） | 任務在開場即完整陳述（QA、spec-to-strategy、web task） | simulated user 持有 belief b，依 Eq.2 只回答被問到的；開場是單一模糊想法（|O|≪|H|） | 交付了用戶從沒要的策略 |
| 數據（data） | 發放 curated documents 或 price series | agent 無數據，須自行從 Web / 247 個 FMP API 取得，且每條序列須可溯源 | 報告裡有從未檢索過的數字 |
| 時間線（time） | 環境內部 replay 或強制 cutoff | 只給 t_cut，look-ahead 的責任在 agent；只要一行日期晚於 t_cut 即 veto | 用了未來的回測 |
| 評分（grading） | 對答案、對 target state、或看 realized returns | 無參考答案：程式驗 invariant、重算指標、比對隱藏 spec；15 veto | 報告說服力與正確性脫鉤 |

來源依據：Table 1 明列 FinBen／MultiFinBen、FinDeepResearch、AlphaForgeBench、CLQT、InvestorBench、AMA、Herculean、τ-banking 在本框架的三欄（Hidden intent／Self-src. data／Point-in-time）上各自留空。**表中各格的勾選符號未在可讀文本中還原，僅能確認該表以這三欄做對比，逐格勾選狀態待圖表核對。**

### 1.2 ⚡ Eureka

> **把「正確」定義成可被另一支程式從交付物本身重算出來，而不是定義成跟標準答案一致。**

直覺：金融交付物的病徵不是「答錯」，而是「答得漂亮但無法驗」。一旦評分只依賴 D 本身（trade log、price series、report、code），任何缺席（沒有 trade log）與任何不一致（報告 MDD 與曲線不符）都變成可判定的布林值，於是不需要 ground-truth 策略，也就不需要凍結題庫——題目可以從語法無限量生成。

### 1.3 信息流

```
        Web / FMP MCP (247 APIs)          ← agent 自行取數，無 curated 數據
                 │
                 ▼
        ┌─────────────────┐   start / send / finalize   ┌──────────────────────────┐
        │      Agent      │ ──────────────────────────▶ │ User Simulator           │
        │ (harness/tools) │ ◀────────────────────────── │ persona + belief b       │
        └─────────────────┘   u_t（只披露被問到的）      │ 決策先過程式驗證          │
                 │                                      └──────────────────────────┘
                 │ 交付 D = {m_t} ∪ ⋃F_t                          │ h（整段 session）
                 ▼                                                ▼
        ┌───────────────────────────────────────────────────────────────────┐
        │ Evaluator：45 dimensions（37 scored, 15 veto）                     │
        │   program judge 17 / LLM judge 19 / both 6 / cost 3 measured       │
        │   LLM 分被程式上限約束：s_j = min(s_j^LLM, s_j^P)                  │
        └───────────────────────────────────────────────────────────────────┘
                 │
                 ▼
        Q = (25/|K|) Σ_k (1/|J_k|) Σ_j s_j ∈ [0,100]
        Score = Q · pass
        AC = pass · 1[e = satisfied] · 1[grading complete]
                 │
                 └── leak-free feedback → memory 編輯 → 同一任務重試
```

## §2 · 數學層

📌 **Napkin Formula**

```
Q      = (25/|K|) · Σ_{k∈K} (1/|J_k|) · Σ_{j∈J_k} s_j        ∈ [0,100]      (Eq.3)
pass   = Π_{v∈V} 1[ s_v ≥ θ_v ]                              |V| = 15     (Eq.4)
Score  = Q · pass
AC     = pass · 1[e = satisfied] · 1[grading complete]                     (Eq.5)

|B| ≈ 3.8 × 10^6   （六元件選項空間，參數值抽取之前）                        (Eq.1)
```

**複雜度 / 直覺**：Q 對五個類別等權（25 = 100/|K|，|K|=5），類別內再對「實際跑了的維度」平均——所以「沒交付任何可檢項目」不是略過，而是留在 J_k 內計 0。Score 是乘法而非加法：veto 一倒，部分分全歸零，但 Q 仍被保留作診斷。AC 再乘兩個布林（用戶簽核、評分完整），兩者皆是必要條件——簽核不充分，但缺簽核必拒，缺評分完整亦必拒。

**loss / 訓練細節**：本框架不訓練模型；所有 agent 的 memory features 在主實驗中關閉，只有 §5.5 的「Claude + Memory」配置在 session 之間保留並由 review step 編輯 memory。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ **以下全部數字為示意假設，非論文實證結果**；僅用來說明 Eq.3–5 的算法結構。論文真實數字見 §5。

假設某 session 有五個類別（|K|=5），每個類別恰好 2 個維度（|J_k|=2），分數如下（0–4 制）：

| 類別 k | s_j | 類別均值 |
|---|---|---|
| data | 4, 2 | 3.0 |
| strategy | 2, 0 | 1.0 |
| backtest | 0, 4 | 2.0 |
| assessment | 2, 2 | 2.0 |
| user experience | 4, 0 | 2.0 |

1. 類別均值相加：3.0 + 1.0 + 2.0 + 2.0 + 2.0 = 10.0
2. Q = (25/5) × 10.0 = **50.0**（示意）
3. 假設 15 個 veto 中有一個 strategy fidelity 維度得分 1，低於門檻 θ=2 → pass = 0
4. Score = 50.0 × 0 = **0**（示意）
5. 即使 e = satisfied（AC 的第二個因子為 1），因 pass = 0 → **AC = 0**（示意）

這就是「用戶簽核但零接受」的最小算術骨架：Q 保留 50 分的部分分用於診斷，但接受判定被 veto 完全截斷。論文實測中該落差是 35 個簽核 session 全部至少倒一個 veto。

## §3 · 數據層

| 項目 | 內容（逐字／原文可定位） |
|---|---|
| 題庫規模 | 10 frozen tasks；seeds 0–9；三 sessions per task per agent |
| 生成方式 | 六元件組合語法採樣，coherence rules 𝒞 排除不自洽組合；|B| ≈ 3.8×10^6（參數抽取前）；sampling is seeded，可按需再抽新題 |
| 市場 / 標的 | 全部十個任務為 U.S. stock trading；組合從 a small basket of stocks 到 a specific sector 或 the entire market |
| 頻率 | 原文未明確披露；只規定 dated rows 與 fills 晚於 signal 日期，不據此推成日頻或排除 tick |
| 數據來源 | agent 自行從 open Web / 「Financial Modeling Prep」MCP server（247 個金融 API）取得；無 curated 數據 |
| 互動預算 | 最多六輪，每輪 300 秒；`max_turns`／`resource_killed` 為無結果／零分；`timeout` 依截止前交付內容評分，仍不可接受 |
| 樣本外與容量假設 | 以 t_cut 為界，交付數據不得晚於 t_cut；**容量／衝擊成本假設未見於可讀文本**——成本元件選項僅 None 或 fixed basis points（Table 2），無 market impact 模型 |
| 洩漏控制 | 隱藏策略 sealed：僅 User Simulator 與 Evaluator 可見；data 與 strategy 類別的回饋只給分不給理由 |

**缺口（不得預設填補）**：每個任務的具體 t_cut 日期、價格頻率解析度（daily vs 其他）、單一任務的股票池規模，**所讀文本未見，圖表／附件待核**。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 原文宣稱 release「the task generator, the evaluator, and a public live leaderboard」；leaderboard 連結為 https://leaderboard.fincosmos.com.co。**程式碼倉庫 URL 所讀文本未見** |
| Checkpoint | 未披露（本框架不發布模型；受評 agent 為 Claude Code / Codex / Gemini CLI / Kimi Code 與 bare model） |
| License | 所讀文本未見 |
| 複現難度 | 中高：模擬器（Claude Opus 4.8, low reasoning effort）、judge（Claude Opus 5.5 via Claude CLI default settings）、evaluator 三者版本耦合；每個分數儲存 evaluator version 與 scoring code hash，使結果在 Evaluator 更新後仍保持可比 |
| 數據可得性 | 公開 Web 數據；ethics statement 明示只用 public Web data、不收個人資料。但具體 retrieval 結果不可複現（Web 會變），這是設計上的取捨而非缺陷 |
| 沙盒 | 一致 runtime，所有 agent 同一環境與 prompt 設定；每輪 300 秒上限 |

## §5 · 評測 / Benchmark

> **模板適用性說明**：原文是「評測範式」論文，**沒有**「本法 vs 前SOTA」的策略績效對照組，也未報告任何策略級 IR／Sharpe／AR／MDD 的跨方法比較。因此五欄模板中的「前SOTA」與「Δ」一律**不適用／不計算**；下表改列原文 Table 3 的逐字受評者結果，避免把 agent 之間的差值誤標為 SOTA 改善。

**Table 3（realistic condition，每 agent 30 sessions）逐字**

| 受評者 | AC (%) ↑ | Quality [95% CI] ↑ | Vetoes (#) ↓ | Cov. (%) ↑ | Cut (%) ↓ | Tokens (M) ↓ | USD ↓ |
|---|---|---|---|---|---|---|---|
| Claude Code | 0 | 37.1 [27.8, 45.0] | 7.4 | 54 | 65 | 1.13 | 2.24 |
| Codex | 0 | 37.2 [30.6, 43.3] | 7.2 | 64 | 69 | 3.20 | 0.80 |
| Gemini CLI | 0 | 22.2 [16.9, 27.4] | 9.3 | 57 | 75 | 3.51 | 0.80 |
| Kimi Code | 0 | 22.3 [14.3, 31.6] | 10.1 | 57 | 74 | 0.89 | 0.83 |
| Bare model | 0 | 20.2 [18.8, 21.5] | 9.1 | 41 | 0 | 0.01 | 0.24 |

*（Sat. 欄逐字：30 / 33 / 33 / 20 / 0；Quality 五類別分項（Data/Strategy/Backtest/Assess./UX）逐字見原文 Table 3，此處不重列以免混淆語義。）*

**Table 4（三種評分者下的「成功」session 數，每 agent 30）逐字**

| 受評者 | Signed off | + LLM-judged vetoes | All vetoes |
|---|---|---|---|
| Claude Code | 9 | 7 | 0 |
| Codex | 10 | 8 | 0 |
| Gemini CLI | 10 | 0 | 0 |
| Kimi Code | 6 | 2 | 0 |
| All 150 sessions | 35 (23%) | 17 (11%) | 0 |

**Table 5（工作流逐關通過數，每 agent 30）逐字**

| 步驟 | Claude | Codex | Gemini | Kimi | Bare | Agents | In seq. |
|---|---|---|---|---|---|---|---|
| (1) 交付的是用戶的策略 | 10 | 9 | 2 | 4 | 9 | 25 | 25 |
| (2) 數據在 cutoff 內 | 11 | 24 | 11 | 9 | 0 | 55 | 15 |
| (3) 無 look-ahead | 21 | 23 | 9 | 10 | 0 | 63 | 14 |
| (4) 交易可重現報告收益 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**哪些是 capability，哪些不是**

- **真 capability 的可信訊號**：(2)(3) 兩關單獨看通過率約一半（Agents 欄 55/120、63/120），且 Codex 在數據關（24）明顯高於其他——這是「會不會用 Web/API 取得合規數據序列」的可測能力，與報告文筆無關。
- **不是 capability、是格式病**：兩個完整性檢查（process completeness 3.8、verifier audit 3.7）在 120 與 119 of the 120 agent sessions 失敗，原文自述它們 do not separate the agents——這是交付模板門檻，不是能力差異，不宜當排名依據。
- **可能是過擬合/洩漏的部分**：原文自述 feedback scores can leak information about the hidden strategy across attempts，因此 leaderboard 結果應來自 frozen agents；本頁據此推斷，任何允許跨 attempt 累積記憶的排名都不可與固定 agent 排名互比。
- **不能反推「策略好壞」**：pre-veto quality 最高 37.2/100、AC 全 0——這組數字只說明交付鏈不完整，**不能用來說明某 agent 選股能力差**；原文未給任何策略層績效對照。
- **簽核 ≠ 正確（原文最硬的證據）**：35 個簽核 session 全部至少倒一個 veto，且全由 program check 抓到；29 delivered no trade log or no prices、15 used data after the cutoff or filled trades before their signal、15 delivered a strategy other than the user's，33 of the 35 至少命中三者之一。
- **圖 1 個案（逐字）**：Kimi Code session 用戶簽核、pre-veto quality 47.9，卻以 21-day momentum 取代 short-term reversal；報告 MDD 為 - 18.96%，同一條曲線上實為 - 21.91%，超出用戶要求的 20% stop。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（§6 Conclusion · Limitations，逐字可定位）

1. The benchmark freezes ten tasks with three sessions per agent，且其 grammar covers long-only US equity strategies with a fixed-cost model。
2. The simulator, the judge, and one evaluated agent come from the same model family；program checks and expert annotation limit, but do not remove, this risk。
3. Most agent turns hit the 300-second limit, so longer turns may let agents get further。
4. Program checks cannot prove that every figure is real。
5. Feedback scores can leak information about the hidden strategy across attempts, so leaderboard results should come from frozen agents。

### 6.2 推斷的隱含假設（**本頁推斷**，非原文主張）

- **Regime 依賴**：十題全為 U.S. stock、long-only、固定成本。**推斷**：對做空、衍生品、跨市場或非美市場的結論不成立；原文未測試，不等於已證明無效。
- **容量／成本**：成本元件只有 none 或 fixed basis points。**推斷**：本法不測 market impact、不測規模衰減；任何「策略在大資金下仍可行」的推論缺乏此框架支撐。
- **時間預算偏誤**：70% of agent turns run into the 300-second limit。**推斷**：需要大量抓取或多輪資料清洗的策略在此被結構性懲罰，這是預算假設而非能力上限。
- **評分者同源**：simulator 與 judge 與其中一個受評 agent 同家族，κ=0.87 僅說明「judge 對其被要求評的東西與專家一致」；原文自述這是 limited but not removed 的風險。
- **缺席型失敗的邊界**：program checks 不能證明每個數字為真，只能證明「與交付的 trade log 一致」。**推斷**：若 agent 同時偽造 trade log 與報告並使兩者自洽，本法測不到；原文未提供此類對抗測試。
- **用戶模擬器保真度**：60 個標註 session 中兩位專家同指 3 個 task-critical simulator error；41 個 sampled sessions 用戶離開，故至多 3 個 exit 屬過早。**推斷**：在更多 persona／更長對話下，模擬器保真度是首要的擴展風險，但這是外推，原文未測。
- **不可外推為普遍因果**：以上皆為單一 benchmark、單一組任務、單一時間窗下的觀察；**不得**寫成「所有 LLM agent 在金融交付上必然失敗」的普適結論。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| τ-bench / τ²-bench / τ-banking / τ-Voice | 同樣有 hidden intent + simulated user，但世界封閉（固定 DB、對 target state 評分） | 部分（意圖開、數據與時間不開） | 非金融／銀行客服域 |
| AlphaForgeBench、BacktestBench | 策略設計全 spec 化 + curated data，無用戶可問 | 不開 | 評「能否實作」 |
| InvestorBench、AMA（live multi-market trading arena） | 在歷史或實盤市場評決策，看 realized returns | 時間開、意圖不開 | 原文引 [28] 指 realized returns 有偏 |
| CLQT、Herculean | pipeline 內部強制 point-in-time，agent 不需自己守時 | 時間封閉 | 評分對象是決策非交付物 |
| FinDeepResearch | 按準確率或專家 ground truth 評分；不將它描述成 LLM judge | 具體邊界以原文所引方法為準 | 不把不同深度研究 benchmark 的評分方式混為一談 |
| FinTrace | 評 tool-call 軌跡，但任務 predefined | 不開 | 互補：可接在 FinCosmos 的 process transparency 上 |
| LiveBench / GDPevo（非金融） | 用刷新題目或重生成任務對抗污染 | — | FinCosmos 用語法採樣達成同目的 |

🎤 **Interview Tip**

- **正確答**：「FinCosmos 的貢獻不是讓 agent 賺更多，而是把『用戶簽核』與『交付可被重建』解耦——35 個簽核 session 全部倒至少一個程式 veto，而若只採用 LLM judge 決定的四項 veto，其中 17 個會被放行，這正是它主張『用程式判而非用 judge 判』的實證基礎。」＋「它的 envelope 是 long-only US equity 加固定成本，所以對執行成本建模與跨資產的結論要另外做。」
- **錯答**：「它是一個交易策略 benchmark，某 agent 的 Sharpe 最高所以最強。」——原文全 150 sessions 的 AC 都是 0，且未報告任何策略級 Sharpe／IR 對照；把 quality 分數當績效排名是類別錯誤。
- **加分答**：指出 completeness 類 veto（3.7、3.8）在 120 與 119 of the 120 agent sessions 失敗、原文自承 do not separate the agents，因此若有人拿它做模型排名，這兩項應先剝離再談。

### 7.1 可證偽預測（**本頁推斷，尚待驗證**）

1. **2027-06-30 前**：若把每輪 300 秒上限放寬，Cut 比例應顯著下降，但 Table 5 的 step (4)「trades reproduce reported returns」仍會是 0 或極接近 0——因為第 (4) 關卡住的是 trade log 完整性與重算一致性，不是時間預算。
2. **2027-06-30 前**：若把 simulator 與 judge 換成與受評 agent 不同模型家族重跑，接受率仍維持在 0 附近；若顯著上升，則 6.1 第 2 條的同源風險被證實為主要解釋。

## §8 · For the Reader

- **因子研究員**：把 §3.4 的三組程式檢查（溯源、point-in-time、獨立重算）抄成自己的交付前 CI；特別是「reported metrics 必須能從 trade log 重算」這一條，它比再跑一次回測便宜，卻能抓到報告與曲線不一致這類最貴的錯。注意容差是逐字固定的：returns 與 drawdowns 0.02、Sharpe 0.5。
- **高頻執行**：本框架對你幾乎沒有直接約束力——資料頻率未明確披露、long-only、成本只有 fixed basis points，沒有 impact 模型、沒有 queue、沒有部分成交。可借用的是它的 execution／transaction cost 兩元件語法當「必須明示假設」的清單，而不是當執行品質評測。
- **組合配置**：§5 的教訓可直接用於 due diligence：要求對方交付 trade log 與 price series，並自行重算 equity curve；35 個「看起來完整」的交付中，29 個根本沒有 trade log 或價格，這是實務上最容易漏檢的一類。
- **LLM-agent 研究者**：§5.5 是最有轉移價值的部分——leak-free feedback + per-turn program scores + memory 編輯，讓 agent 在 median 34.5 sessions per task（range 12–132、506 in total）內學會全部十題，且最後常卡在 strategy ambiguity（6 of 10 tasks 的最後一關）。這提示：對 agent 而言，「把模糊需求收斂成單一可執行定義」比「寫出回測」更難。
- **RL 策略 / 自演化系統**：注意原文的警告——feedback 跨 attempt 可能洩漏隱藏策略，因此自演化曲線必須在 frozen-agent 協定下比較；否則你量到的是「記憶裡有答案」而非「學會了方法」。
- **研究學生**：讀 §3.3 的 disclosure policy（Eq.2）與 Appendix D 的五條規則，這是把「多輪對話中的資訊揭露」形式化成可驗證約束的乾淨範例；比讀一般 simulated-user 論文更適合當方法章節的模板。

## References

- Cao, Peng, Qian, Li, Yu, Xu, Cao, Wang, Jiang, Wang, Xie, Chen, Li, Huang, Liu, Nie, Ananiadou. *FinCosmos: An Open-World Evaluation Paradigm for Agents in Finance.* arXiv:2611.07464v1 [cs.AI], 05 Nov 2026. https://arxiv.org/abs/2611.07464v1
- Leaderboard（原文脚注 1）：https://leaderboard.fincosmos.com.co
- Financial Modeling Prep MCP server（原文脚注 2）：https://site.financialmodelingprep.com/
- Lineage（原文 §2 所引，僅列本頁實際引用者）：τ-bench [60]；τ²-Bench [2]；τ-Knowledge [46]；τ-Voice [44]；AlphaForgeBench [65]；BacktestBench [53]；CLQT [43]；Herculean [40]；InvestorBench [27]；AMA [42]；FinDeepResearch [70]；FinTrace [4]；LiveBench [55]；GDPevo [68]；Bailey et al. 2017 [1]；Li et al. 2026 短窗高估 [28]；Zhang et al. 2026 one-switch leakage [63]。
- 來源狀態：本頁僅依據本次取得的 arXiv 可讀全文；圖片（Figure 1–4）、Appendix B 完整 rubric 表格之外的附件內容未提取，相關逐格數值待核。