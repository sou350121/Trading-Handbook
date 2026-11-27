<!-- ontology-5axis data=量价表格 horizon=跨周期 paradigm=元学习搜索 alpha=因子挖掘 autonomy=Agent自主演进 -->

# Autonomous Quant Researcher 解構（Autonomous Quant Researcher）

> **來源發布**：2026-11-04 · （無 venue） · arXiv [2611.06607](https://arxiv.org/abs/2611.06607)
> **arXiv 原文**：[Hard to Fool: A Bounded LLM Research Loop for Quantitative Finance, and What Two Dozen Campaigns Found](https://arxiv.org/abs/2611.06607v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點在「量價表格／跨周期／元學習搜索／因子挖掘／Agent 自主演進」，但它解的不是「如何生成更好因子」這個舊 gap，而是 **decider 端**的 gap：把「什麼算通過」從單一分數改成不可變分期資料倉＋一次性 holdout 收據＋多門禁合取，並公開全部 campaign（含拒絕）。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `跨周期` | `元学习搜索` | `因子挖掘` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①做的新事：agentic research loop 的 proposer 留給 LLM，decider 全部交給 trusted code；假設以 strict declarative JSON 宣告（schema version 3，僅 `hypothesis` 與 `rationale` 兩欄），模型不寫程式，因此沒有可沙箱的攻擊面。②核心 trick：分階段不可變 store ＋ 一次性 holdout（先落耐久收據才可開）＋ 絕對門禁合取 ＋ hash-chained append-only ledger，重複／近重複假設在耗 trial 前即被拒收。③對「Agent 自主演進」這條軸是反向命題：自主性被限制在提議面，驗收面完全不自主。④關鍵實證：一個 LLM survivor 在 discovery／validation 的 Net Sharpe 為 2.94／3.34，唯一一次 holdout 花掉後落到 0.26，被拒。

**X-Ray.** 多數 LLM-alpha 工作在優化 proposer（角色、記憶、反思、樹搜索、code evolution）；本文把工程預算整筆押在 decider。它解掉三個舊坑：(a)「模型寫可執行候選→沙箱」的安全與可定價性兩難——作者直接刪掉可執行欄位，讓「每個可准入假設都是 evaluator 會定價的」變成結構性事實；(b) 驗證集被反覆偷看——改成一次性 holdout＋耐久收據，收據後 crash 也不退還；(c) 暗計數——全部 campaign（含拒絕）入 hash-chained ledger，改序即 fail closed。預測它打不開的 envelope：詞彙表刻意小（defined-risk 價差、5 分鐘網格 entry、bounded Boolean signal），搜尋空間被釘死在「evaluator 已會定價」的結構上，產出上限是「把假策略被誤報的機率壓低」，而不是「找到好策略」。對量化讀者的意義：這是 research governance 的參考實作，最該記住的數字是失敗案例的 0.26，不是任何通過值。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（autoresearch／Loop Engineering／alpha-mining agents） | 本書改動 |
|---|---|---|
| 提議邊界 | autoresearch 讓 agent 編輯單一訓練腳本；Loop Engineering 把 loop 本身做成 first-class artifact（persistent contract、explicit budgets、deterministic gates、audit trail、maker/checker 分離）；本文早期設計曾讓模型回傳可執行 Python 候選（需 sandbox：credential stripping、import/capability denial、content-hashed evaluator） | 模型只輸出 strict declarative JSON（schema version 3）；多餘欄位、非法 JSON、未支援詞彙、時序錯誤、巢狀過深、參數越界一律在任何科學評估前拒收。原文的論證是「把問題移除，而不是把它圍起來」：沒有程式碼可沙箱 |
| 驗收 | autoresearch 以單一 validation metric 改善即保留；Loop Engineering 用 deterministic gates | 每階段須通過全部適用門禁的**合取**：base cost 後絕對 edge >0、base-cost 平均淨 P&L >0、weekly-cluster bootstrap 下界 >0（2,000 deterministic resamples，clustered by ISO week）、elevated cost 平均淨 P&L >0、指標有限且有定價列、樣本下限。ranking score 只在已通過者之間排序，**不能讓誰通過** |
| 樣本外 | 驗證集反覆使用 | discovery／validation／final holdout 三階段 store 由同一歷史來源一起 materialize，但每個 access capability 只解析自己的階段，每個 store 各帶 content hash 與 manifest hash；holdout 須先寫入不可逆收據才可開，收據後 crash 仍算已消耗、不可重試；「Budget exhaustion is a pause, never a success」 |
| 記錄 | autoresearch 只留改善的版本 | 每個 formulation／admission 決策／trial／crash／階段轉移／holdout 收據都入 hash-chained append-only ledger；finding 匯出為 portable JSON，帶 `research_only`、`execution_authorized: false` 的權威戳 |

### 1.2 ⚡ Eureka

**一句話**：把「模型提議、模型認定改善」翻轉為「模型提議、不可變程式認定為真」，而認定為真的路徑上只有一道一次性門。

直覺：ML 裡 lower validation loss 通常真是更好的模型，所以 loop 可以只看一個分數；交易裡「回測改善」的預設解釋是過擬合、look-ahead、多重比較、未計成本或市場漂移，所以分數只能是**必要條件之一**。作者把這條不對稱寫進架構：admission 先擋語意重複與近重複（同一個想法上轉一個數值旋鈕），family quota 每個 descriptor family 最多收兩個——「A 50-trial budget therefore buys 50 distinct ideas」，trial 數本身成為可審計量。

### 1.3 信息流

```
proposer（reference：本地 served 7B coding model；任何 OpenAI-compatible endpoint 皆可）
   │  JSON {hypothesis, rationale}（schema version 3：entry minute on the 5-min regular-session grid、
   │  bounded Boolean signal over approved causal observations、exactly one defined-risk 結構）
   ▼
trusted validator ── 不合規 → 記錄，不耗 trial
   ▼
admission ── 語意重複／近重複、family quota 超額 → 拒收，仍不耗 trial
   ▼
discovery store（SPY 參考 mission：2016–2021；traded-bar 0DTE：2024-01-19 至 2025-08-05）
   │  gates ∧（含 sample floors）
   ▼
shortlist（freezes at most three；依最弱 gate 分量排序、偏好不同 cluster、整表 hash）
   ▼
validation store（SPY：2022–2024；0DTE：2025-08-06 至 2026-02-05）
   │  gates ∧；5% 下尾機率除以 shortlist 大小
   ▼
[耐久收據寫入] ──► holdout store（SPY：2025 onward；0DTE：2026-02-06 至 2026-08-10）※僅一次
   ▼
hash-chained ledger ──► portable finding（contract/manifest hash、逐階段證據、限制與失效條件、authority stamp）
```

## §2 · 數學層

📌 **Napkin Formula**

```
Accept(h) = ∧_g  g(h) > 0
   g ∈ { edge_after_base_cost, mean_net_pnl_base,
         bootLB_weekly(2,000 deterministic resamples, clustered by ISO week),
         mean_net_pnl_elevated, finite&priced_rows, sample_floors }

Rank(h) = min(edge, bootLB, elevated_cost_pnl, worst_year)   # 只排序，不能准入
p_adj   = 5% / |shortlist|                                  # 下尾機率按 shortlist 大小除
ledger_i = H( ledger_{i-1} ‖ event_i )                       # 改序即 fail closed
```

**直覺**：這不是一個估計式，而是一組**否證條件**。單一 metric 可被 drift、成本假設或一個 outlier 日推過門檻；合取把「只有一個分量好看」的候選直接淘汰，並且把排序與准入徹底分離（control-relative metric 只作診斷，弱 control 不能救候選）。bootstrap 用 ISO week 聚類，對應每週僅少數交易的稀疏結構。

**複雜度**：evaluator 是 deterministic CPU code，無梯度訓練、無 GPU 需求；單一候選成本＝一次分階段 replay，總成本 ≈ trial budget × replay。真正的稀缺資源不是算力，而是 **trial 預算與 holdout 次數**。

**Loss／訓練細節**：所讀文本未見任何模型訓練、loss 或超參適配描述。三個時間邊界必須分列，不可互相代稱：(i) proposer 預訓練資料截止日——所讀文本未見；(ii) tokenizer／詞表——所讀文本未見；(iii) 微調或適配後權重的時間邊界——所讀文本未見（原文僅稱 evaluator 與 proposer 無關，換模型只換候選集、不會移動指標，§2.5 Model independence）；campaign 的資料切分時間邊界見 §3。

## §2.5 · 帶數字走一遍（Worked Example）

以下**全程為假設／示意**，數字非論文實證，只為讓機制定型；本文真實數值一律留在 §5。

1. **假設** proposer 提出候選 H：entry 在 5 分鐘網格上、訊號為「credit 超過 width 的 10%」、結構為 put credit spread（此形制描述取自原文 §3.1 的 campaign 敘述，非本頁發明；數值僅作排版示意）。
2. **admission 示意**：schema 檢查通過，但與早前已受理的某個 trial 只差一個數值旋鈕 → 判為近重複 → **拒收且不耗 trial**，trial 計數不變。這一步不產生任何績效數字。
3. **discovery 示意**：假設 H 的分量中 bootstrap 下界那一項 ≤0 → 合取斷開 → 不進 shortlist。注意即使它的 ranking score 其他分量再漂亮，ranking 也無法讓它准入。
4. **shortlist 示意**：假定有若干候選全數通過 discovery，coordinator 依「最弱 gate 分量」排序、偏好不同 cluster、凍結不超過三個並整體 hash。validation 只評這一批，5% 下尾機率除以 shortlist 大小——shortlist 越大，單一候選要過的門檻越嚴（此處刻意不給算術結果）。
5. **holdout 示意**：收據先落盤，才允許開窗；開窗後若行程 crash，holdout 已被消耗且不可重試。這就是「最貴的不可逆資源」的手感。

## §3 · 數據層

- **模態與頻率**：entry 決策點在 5 分鐘 regular-session 網格；訊號是 bounded Boolean expression over approved causal observations（return from the open、return between two earlier times、first touch before entry）。SPY／QQQ 的 0DTE 以 **actual traded option bars** 定價；actual OPRA evidence 是獨立 typed gate，狀態含 `unavailable`、`insufficient_coverage`、`failed_evidence`、`passed`，replay 只吃 hash-verified quote artifact、**無 fallback price source**。overnight SPX verticals 為 modeled prices；另有日頻 ETF 規則、Kalshi crypto binaries、earnings options、off-exchange volume（公開 FINRA 資料）與一個 pre-registered alternative-data campaign。
- **切分（逐字窗口）**：SPY 參考 mission（日頻與 modeled 資料）discovery 2016–2021、validation 2022–2024、final holdout 2025 onward；traded-bar 0DTE discovery 2024-01-19 至 2025-08-05、validation 2025-08-06 至 2026-02-05、holdout 2026-02-06 至 2026-08-10。每個 campaign 凍結自己的窗口。
- **樣本下限（reference SPY mission）**：discovery 120 trades / 40 effective weeks；validation 60 trades / 24 weeks，且 2022、2023、2024 每年至少 15 trades 與 8 weeks、不得有負年；holdout 30 trades / 12 weeks。traded-bar 0DTE 因窗口較短，三階段分別要求 40、15、15 trades。floors 隨窗口縮放並凍結於 campaign contract。
- **可得性與容量**：market-data 檔案**不隨 repo 散布**（原文：「Market-data files are not redistributed」）；campaign records 僅應作者請求提供。容量／市場衝擊曲線所讀文本未見，只有機制性描述（0DTE 溢價「earns its return by absorbing unforecastable first-day shocks」）。**未提取之圖表／附件待核**。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| `github.com/henryzhangpku/autonomous-quant-researcher`（engine code、reference mission、public pre-registration 與其 hash-chained ledger） | 無模型權重；reference 配置為本地 served 7B-parameter coding model，proposer 可換任何 OpenAI-compatible endpoint | repo 授權所讀文本未見（**論文本身**標 CC BY 4.0）；兩者不可混用 | 中：test suite 可離線無 key 跑；browser demo 以 Pyodide 在 synthetic tape 上跑（planted regime，discovery 找到、validation 駁回）；但 campaign records 需向作者索取 | 低：market-data 檔不釋出；OPRA quote artifact 需自備 |

## §5 · 評測 / Benchmark

本文**沒有「前SOTA vs 本方法」的同表對照設計**（比較對象是門禁，不是別的模型），下表「前SOTA」一律為「未披露」，Δ 欄依規則留空；且各階段窗口不同，**不可**把階段間差額當 Δ。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| SPY 0DTE put credit（w5/o5、10:00 ET、held to settlement、0.5% of width cost） | 每 session 淨 P&L（bps of width）· discovery | 未披露 | +55.7 | |
| 同上 | 每 session 淨 P&L（bps of width）· validation | 未披露 | +71.6 | |
| 同上 | gross win rate | 未披露 | 95.2% | |
| 同上 · drift-neutral 殘差 | gross P&L（bps of width）· discovery | 未披露 | +74 | |
| 同上 · drift-neutral 殘差 | gross P&L（bps of width）· validation | 未披露 | +100 | |
| QQQ 0DTE LLM survivor（traded-bar 窗口） | Net per trade（% of width）· discovery | 未披露 | +5.69 | |
| 同上 | Net per trade（% of width）· validation | 未披露 | +4.82 | |
| 同上 | Net per trade（% of width）· holdout（一次性） | 未披露 | +0.52 | |
| 同上 | Net Sharpe · discovery | 未披露 | 2.94 | |
| 同上 | Net Sharpe · validation | 未披露 | 3.34 | |
| 同上 | Net Sharpe · holdout | 未披露 | 0.26 | |
| 同上 | Trades · discovery | 未披露 | 221 | |
| 同上 | Trades · validation | 未披露 | 43 | |
| 同上 | Trades · holdout | 未披露 | 72 | |
| overnight SPX 1DTE ATM call debit spread | weekly Sharpe · validation | 未披露 | 2.27 | |

**解讀**：這張表要當**反面**讀。

- **真正的 capability 是拒絕能力**：232-cell 日頻網格「4 discovery passers, 0 promotions」、Kalshi 138 conditional hypotheses「0 cleared」、ETF flow 84 cells「0 cleared Bonferroni」。但這些 0 同時是 power 問題（作者自承短樣本），**不能**讀成「該市場無效」。
- **QQQ survivor 那組是一次性樣本外 holdout 證據**：in-sample 不是勉強過關——原文說「a loop that accepts on a validation Sharpe above 3 would have promoted this rule」；holdout 上 expectancy「fell by about 90%」、Sharpe 掉一個數量級、半個窗口淨負。這是全篇最可檢驗的一組數字，也是本文主張的最佳例證（且是失敗例）。
- **SPY 0DTE 溢價「真、但不夠」**：兩個標的形狀相同、drift 只解釋「13–30% of the payoff」、「The drift-neutral residual is positive in both splits」，但它的 weekly bootstrap 下界「marginally negative in both」；最佳幾何 w5/o2 只差「1.35 bps of width」未過 discovery raw bootstrap bound，QQQ 複現「fails exactly the same condition at a similar magnitude」。這讀成「結構真、門檻未達」，不是「發現 alpha」。
- **成本是未定量的關鍵**：原文「The breakeven entry cost for full gate compliance lies between 0.5% and 1.0% of width」——是否為策略取決於一個尚未實測的執行數字。任何把 +55.7／+71.6 bps 直接當淨 alpha 的解讀都越過這條線。
- **兩個通過自己 pre-registered bar 的結果不是策略**：off-exchange share 的資訊係數在 1、5、20 日皆負（原文「t between -3.1 and -5.8」、每個 2021–2026 日曆年皆負、集中於 top decile），而「the vendor-published 'dark pool index' predicts nothing」；idiosyncratic-gap campaign 則是延續假設被拒、crash-day 反轉存活。作者自稱 report as research findings with limitations, not as strategies。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **§5**：loop「makes it hard to report something that is not there; it does not make true things easy to find」；詞彙表刻意小，超出詞彙的假設在 trusted code 擴充前無法測試。
- **§5（power）**：多個 campaign 樣本短，原文舉例逐字為「46 days of Kalshi data, 60 weekends of ETF flow」，其 null「statements about power as much as about markets」。
- **§5（regime）**：原文「Option results from 2024 onward cover no bear market of 2022's kind」。
- **§5（定價）**：部分 modeled prices 用於 actual quotes 不可得處，相關 campaign 有標註。
- **§5（成本）**：數個 campaign 的成本是「assumptions with stated breakevens rather than measurements」。
- **§5（作者性）**：本文由系統作者撰寫；每個數字背後的完整 campaign records 僅應請求提供。
- **§2.3**：modeled option prices 不是市場價；OPRA 證據是獨立 typed gate，覆蓋不足即不可 replay。
- **§3.2**：門檻未達時「the bar was not moved」——作者拒絕事後鬆綁。
- **§4（門禁首輪漏掉的失效模式，作者自列）**：drift as alpha（原文引 overnight SPX 1DTE ATM call debit spread「passed every validation gate with a weekly Sharpe ratio of 2.27; 56% of its validation payoff was the underlying's own move, and the bootstrap bound on the residual was negative」，此後 drift control 成為該 family 的門禁之一）；one-direction discovery window（規則改為「a condition is read together with its opposite」，正反都過＝long-side artifact）；costs assumed not measured（now「measured entry costs precede any holdout spend」）；look-ahead through a feature（Kalshi 中唯一形式通過者以同窗成交量為條件，改標 descriptive-only）；a lost specification（凍結規格落在臨時路徑，重構後以 discovery Sharpe 對到六位小數與 manifest hash 才准開 holdout）；spending evidence for nothing（前提已全負時不讀 TA holdout）。
- **本頁未見**：proposer 預訓練資料截止、容量／市場衝擊分析、券商層執行成本實測。所讀文本未見，圖表／附件待核。

### 6.2 推斷的隱含假設（本頁推斷，非作者陳述）

- **權重記憶是 gate 覆蓋不到的泄漏面（推斷）**：staged store 約束的是**資料可見性**，約束不了 LLM 權重裡記得的新聞與已發表異象。本文引用 Li et al. 的「time travel」問題，但防線止於資料倉；對 pre-registration 類 campaign 可控（資料尚不存在），對回看 2016–2021 的 campaign 則否。預訓練截止未披露 → 此風險不可量化，**待核**。適用條件：任何以 LLM 作 proposer 且 discovery 窗早於模型訓練資料的設定。
- **成本前提是單點失效（推斷）**：0DTE family 的結論在 breakeven 落在原文所述的 0.5%–1.0% of width 區間時翻轉；若真實 round-trip 高於該帶，門禁通過與否全失效。僅適用於 10:00 ET entry、held to settlement 的 defined-risk 價差，**不可外推**到其他執行風格。
- **regime 依賴（推斷）**：traded-bar 窗口起自 2024-01-19，缺 2022 型熊市；「首日衝擊不可預測」的機制結論在波動結構改變時未必成立。
- **campaign 選擇仍有人為偏差（推斷）**：ledger 保證「跑過的全報」，不保證「該跑的都跑了」；mission 立項由作者決定，報告完整 ≠ 無 selection bias。
- **單一權威戳的邊界（推斷）**：`research_only` / `execution_authorized: false` 是好設計，但 hash chain 只證明「未被事後編輯」，不證明「evaluator 實作正確」；輔助資料缺失（如 OPRA 覆蓋不足）在本設計下是明示 typed state，不構成 survivorship 式的隱性偏差。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Karpathy `autoresearch` | 同為「跑固定實驗、改善就留」；差異在驗收是單一 validation metric，且 agent 直接改訓練腳本 | 文本載明為公開專案 | 被本文當出發點，非競爭者 |
| Greyling `Loop Engineering` | loop-as-artifact：persistent contract、budgets、deterministic gates、audit trail、maker/checker 分離；本文把契約落到金融 | 文本載明為公開專案 | 方法論同源 |
| TradingAgents／FinMem／AlphaAgent／R&D-Agent-Quant／MCTS factor mining／code-based alpha evolution／AlphaAgents | 全部優化 proposer（角色、記憶、反思、因子生成、RL）；本文不碰 proposer | 未披露 | 互補而非對立 |
| Gençay（leakage-safe、search-aware deflation） | 同樣結論：LLM 找到的策略在 agent 自己的 trial 數下過不了認證；差異在器械——他把 leakage safety 與 search-aware deflation 都建進策略發現系統（同時含結構性泄漏安全與統計校正），本文則以宣告式假設邊界＋一次性 holdout 為核心。原文並引其 planted look-ahead oracle（Sharpe 35）同時通過 DSR 與 PBO，說明統計校正不能替代結構控制 | 未披露 | 立場一致、器械不同 |
| Deep et al.（walk-forward、含成本） | 多段 OOS 驗證可解釋假設；本文用三階段＋一次性 holdout＋合取門禁 | 未披露 | 驗證強度更嚴 |
| Li et al. DeepFund（live benchmark，反 time travel） | 直接上實盤避免看過結果；本文以 pre-registration＋一次性 holdout 替代 | 未披露 | 互補 |
| Barton et al.（兩套 live fleet 六個月） | operating layer 比模型選擇更影響行為、且兩 fleet 皆無方向性 edge；與本文「decider 才是產品」同向 | 未披露 | 外部證據支持 |

🎤 **Interview Tip**
- **正確答**：「它的貢獻是 decider 的參考設計：假設限成 declarative JSON 消除沙箱面、一次性 holdout 收據把樣本外變成不可再生資源、絕對門禁的合取取代單一分數、全部拒絕公開。最有說服力的實證是那個反例——discovery／validation Sharpe 2.94／3.34，唯一一次 holdout 落到 0.26，被拒；也就是說『validation Sharpe 高於 3 就收』的 loop 會直接 promote 它。」
- **錯答**：「它找到 SPY 0DTE put credit 這個 alpha（+55.7／+71.6 bps）。」——那兩個是 base cost 下的階段均值；drift-neutral 殘差的 weekly bootstrap 下界兩邊皆 marginally negative，最佳幾何差 1.35 bps of width 未過門檻，且 full gate compliance 的 breakeven 落在 0.5% 與 1.0% of width 之間尚未實測；作者自己判定不 promote、門檻未移動。

### 7.1 可證偽預測（本頁提出，尚待驗證）

1. 若第三方在 2027-06-30 前實測到 SPY 0DTE put credit（10:00 ET entry、held to settlement）的 round-trip 執行成本低於 0.5% of width，則該 family 的 weekly bootstrap 下界應轉正——直接檢驗「溢價真、卡在執行」的讀法。
2. 若把 proposer 換成更強的 OpenAI-compatible 模型、重跑同一 pre-registered campaign，**既有候選的指標不應改變**（§2.5 model independence 的主張：evaluator 不依賴 proposer，換模型只換候選集）；通過率是否上升不在該主張範圍內，若上升可能只是候選集變好，不構成對 model independence 的反證。
3. 若在 2026-08-10 之後為已 spent 的 QQQ survivor 另立新 holdout 窗重測，預期 Net Sharpe 不會回到 discovery 水準；若回到，則 0.26 應改讀為窗口效應而非過擬合。

## §8 · For the Reader

- **因子研究員**：先把 §2.2 的「凍結合約＋admission 去重＋依最弱 gate 排序」抄進流程，比抄任何因子有用；特別是可執行的檢驗規則「條件與其反向一起讀」——§3.3 的 20 日跌／20 日漲同向通過，就是單向窗口 artifact 的標準長相。
- **高頻執行／做市**：本文把策略問題轉成執行問題的那一步值得記住——0DTE family 的生死線是 breakeven 落在 0.5% 與 1.0% of width 之間；你要提供的是實測 round-trip 成本，不是再一個回測。
- **組合配置**：把 finding 上的 `research_only` / `execution_authorized: false` 當下游合約範本；同時注意兩個通過 pre-registered bar 的結果（off-exchange share 的負 IC、crash-day 反轉）作者明說是研究發現不是策略，勿直接上倉。
- **LLM-agent 研究者**：這是「agent 能力不是瓶頸、loop 契約才是」的量化版本；若你的系統仍讓 LLM 寫可執行候選，先讀 §2.1 的刪欄位論證——移除問題，而不是圍住問題。
- **RL 策略研究者**：注意 trial 預算是比 step 數更硬的約束，且 holdout 不可重播；在此設計下「漸進式探索→逐步改善」的路徑被系統性壓縮，reward shaping 式敘事不適用。
- **研究學生**：§4 的六條失效模式清單是現成 method-section checklist；§5 的短樣本自承則是「null 結果要報 power」的示範。

## References

- 原論文：H. Zhang, *Hard to Fool: A Bounded LLM Research Loop for Quantitative Finance, and What Two Dozen Campaigns Found*, arXiv:2611.06607v1 [q-fin.CP], 04 Nov 2026（CC BY 4.0）。本頁數值取自其可讀全文；圖片／附件未核。
- 實作：[1] Autonomous Quant Researcher — https://github.com/henryzhangpku/autonomous-quant-researcher
- 方法論 lineage：[9] Karpathy, autoresearch（2026）；[5] Greyling, Loop Engineering（2026）；[14] Nosek et al., The preregistration revolution（PNAS 2018）；[10] Künsch, stationary bootstrap（1989）。
- 資料窺探／多重比較：[12] Lo & MacKinlay（1990）；[16] White（2000）；[6] Hansen（2005）；[24] Romano & Wolf（2005）；[7] Harvey, Liu & Zhu（2016）；[8] Hou, Xue & Zhang（2020）；[13] McLean & Pontiff（2016）；[2] Bailey & López de Prado（2014）；[3] Bailey et al.（2017）；[4] Bailey et al.（2014）。
- Proposer 側對手：[17] TradingAgents（2024）；[29] FinMem（2023）；[15] AlphaAgent（2025）；[11] R&D-Agent-Quant（2025）；[25] MCTS factor mining（2025）；[23] code-based alpha evolution（2025）；[27] AlphaAgents（2025）；[28] Choi et al.（2025）。
- 評估側對照：[20] Gençay（2026）；[21] Deep et al.（2025）；[22] Li et al., DeepFund（2025）；[26] Yao, Zheng & Li（2026）；[19] Chen & Pu（2026）；[18] Barton et al.（2026）。
- 來源鏈接：https://arxiv.org/abs/2611.06607v1