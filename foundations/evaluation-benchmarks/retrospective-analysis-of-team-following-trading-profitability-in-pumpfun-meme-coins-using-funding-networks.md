<!-- ontology-5axis data=图关系 horizon=高频日内 paradigm=元学习搜索 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Retrospective Analysis o 解構（Retrospective Analysis o）

> **來源發布**：2026-11-08 · （無 venue） · arXiv [2611.11436](https://arxiv.org/abs/2611.11436)
> **arXiv 原文**：[Retrospective Analysis of Team-Following Trading Profitability in Pumpfun Meme Coins Using Funding Networks](https://arxiv.org/abs/2611.11436v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點靠近「圖關係／高頻日內／元學習搜索／組合執行優化／人機協同可解釋」，但實作是 hindsight grid search + selection-matched null，不是 meta-RL。它補的 prior gap 是：Pump.fun/Solana 文獻多為操縱分類或早期詐騙預測，少見以有限資本帳戶跟隨 funding graph 候選團隊、並把 cash 與 marks 分開審計。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `图关系` | `高频日内` | `元学习搜索` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** 以 10 個獨立 Pump.fun 窗口（2024-04 至 2026-09）重建 directed SOL funding groups，對 1 SOL 起始的有限資本帳戶做事後團隊閾值搜索與現金／帳面拆分。主帳戶在 8 個有合格團隊的窗口 Equity 為 1.10 至 11.32 SOL，但 free cash 超過 1 SOL 僅 5 個窗口，marks 佔 Equity 17–94%。線上識別時 19 隊中有 9 隊只在最後一次發幣後才可識別；selection-matched null 僅 3 of 40 經 Holm 調整顯著。結論是可審計的條件式事後模型帳戶，不是可執行利潤，也不是 SOL 價格的因果證據。

**X-Ray.** 這篇的價值不在「找到 11.32 SOL 的策略」，而在把「事後可跟隨」拆成三層可審計對象：funding graph 的候選實體、全窗口閾值搜索、有限現金帳戶。它解掉舊工程坑——把 mark 當 cash、把 in-sample 排名當即時信號；但沒有解 execution、identity error、full-procedure selection。Pareto 上它把成本壓在資料重建與可解釋審計，換取跨 10 窗口的 failure-mode 對照；代價是窗口為 purposive、graph 規格一變合格團隊數可 1 到 5、線上結果仍用 full-window funding links。對量化讀者，這頁應被當成 backtest 失敗模式清單：先問 free cash 與 marks 比，再問 selection null 是否重跑 discovery，最後才看 equity。

## §1 · 架構 / Core Mechanism

### 三大改動 vs 前作

| 維度 | 前作／鄰近研究（所讀文本所述） | 本文改動 | 影響 |
|---|---|---|---|
| 任務物件 | Pump.fun/Solana 研究多分類操縱、偵測 rug pull 或預測詐騙 token；不跟隨已識別 creator groups 進有限資本帳戶 | 同一帳戶回報三種資訊制：hindsight team thresholds、cash instead of marks、online qualification | 把「事後大餘額」拆成可實現現金與不可融資 marks |
| 估值管線 | 泛用 AMM/Uniswap 論述常把報價當成交；本文不轉移其歷史 reserve 單位 | bonding-curve reserve-state pipeline：own deposit、units、closing states、strict 85 SOL boundary | 區分 quote/mark 與 finite liquidation proceeds |
| 樣本設計 | 單一窗口或單一市場的回測易把期間差異當策略能力 | 10 個 independently reconstructed windows，其中一個在結果計算前固定 | 顯示 apparent opportunity 的跨期差異與 failure modes |

### Eureka trick

**把「跟團隊」拆成識別→閾值搜索→現金可用性三段，用有限資本帳戶把 mark 與可再投入 cash 分開，再用 10 個獨立窗口與 selection-matched null 量化事後最優中有多少是不可實現浮盈。** 直覺是：ranking index 可以很大，但若現金被從未達閾值的倉位鎖住，那個大餘額只是模型 mark，不是下一個 entry 的資金。

### 信息流

```text
Solana getBlock API
  -> PostgreSQL decoded records
  -> reserve states / closing states, 0 <= R <= 85
  -> directed SOL funding graph (endpoint incidence, Jaccard pruning)
  -> candidate groups (11–399 launches; ranking gate >10)
  -> hindsight threshold grid (4, 6, ..., 84, 85 real SOL)
  -> finite-capital account: cash, marks, position caps, cash stop
  -> controls: random cohorts, selection-matched nulls, midpoint/frozen/expanding tests
```

## §2 · 數學層

📌 **Napkin Formula**

\[
I_g=\max_t \exp\left\{\sum_{i\in g}\log[1+w(\rho_{it}-1)]\right\},\quad w=0.1,\quad I_g>10
\]

其中 \(\rho_{it}\) 是標準化名目下 finite liquidation-or-mark value 除以 entry debit。Own deposit 以 \(X_e=30+R_e\)、\(u=q/k=a/[X_e(X_e+a)(1+\sigma)]\) 修正，再檢查 counterfactual reserve 不超過 85 SOL。

**直覺**：index 是「每筆押 \(w\) 財富、序列再投入」的成長乘積取 log 後可加；但它忽略現金鎖定與同時多倉，所以不能讀成有限資本投資人的財富。**複雜度／訓練**：未披露；本頁推斷為對 group × token × threshold × path state 的網格重放。非訓練式，無 loss/優化器；閾值為全窗口事後選擇，參數敏感度另做 one-at-a-time sweep。

## §2.5 · 帶數字走一遍（Worked Example）

以下不是本頁假設數字，而是原文 Table 3 的 worked example（避免用未披露數值）：

1. 取 \(w=0.1\)，第一個 token 的 \(\rho=1.64\)，因子 \(1+0.1(1.64-1)=1.064\)，log factor 0.062，running product 1.064。
2. 第二個 token \(\rho=1.81\)，因子 1.081，log factor 0.078，running product 1.150。
3. 第三個 token \(\rho=1.64\)，因子 1.064，log factor 0.062，running product 1.224。
4. 該隊全部 13 個 token 的 index 為 60.47，超過 gate 10；但這是 ranking statistic，不是有限現金帳戶的 wealth。

## §3 · 數據層

- **市場／時段**：Pump.fun on Solana；10 個獨立窗口，2024-04 至 2026-09；UTC 起訖與小時數逐窗見 Table 17（例：2024-04 23.68 小時、2025-06 79.27 小時、2026-09 48.00 小時）。
- **來源**：Solana blockchain `getBlock` API，解碼後存 PostgreSQL；decoded token price dynamics 可能含不準確。
- **頻率／狀態**：reserve states 以 completed second 為可用邊界；maximum quote age 30 s；observation horizon one hour；admitted real-reserve domain \(0\leq R\leq 85\)。
- **樣本損耗**：nine earlier windows 的 eligible paths 有 0.6% 在小時末前結束，2026-09 為 1%；full-hour archive coverage 不等於每筆交易都被收集；missing paths 不補零報酬。
- **樣本外／容量假設**：每個 window 內獨自重建 graph、ranking、controls；不跨窗口轉移身份。Midpoint/frozen/expanding 屬 post-inspection sensitivity，不是真正 prospective OOS；容量以 1 SOL 起始、finite cash、position caps 表示。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 未披露 | 未披露 | 論文 CC BY 4.0；代碼授權未披露 | 高：需 Solana 解碼、PostgreSQL、Euler trail 清洗、85 SOL 邊界與 finite-capital 模擬 | 原文稱 data and replication code available from the author upon reasonable request |

## §5 · 評測 / Benchmark

> 下表「前SOTA」僅為模板欄名；實際基線是 **random cohort median**，不是文獻 SOTA。Δ = 本方法 − random median，單位 SOL；不可讀成 alpha 或 SOTA 領先量。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 2024-04 | Equity (SOL) | 0.99 | 5.14 | 4.15 |
| 2024-07 | Equity (SOL) | 0.92 | 1.59 | 0.67 |
| 2024-10 | Equity (SOL) | 1.06 | 9.61 | 8.55 |
| 2025-02 | Equity (SOL) | 0.92 | 11.32 | 10.40 |
| 2025-05 | Equity (SOL) | 0.94 | 5.26 | 4.32 |
| 2025-06 | Equity (SOL) | 0.87 | 1.10 | 0.23 |
| 2025-09 | Equity (SOL) | 0.94 | 1.99 | 1.05 |
| 2026-09 | Equity (SOL) | 0.89 | 4.43 | 3.54 |

**解讀**：這些 Δ 不是乾淨 capability。主帳戶 Equity 含 marks；random cohort 只重跑同一 threshold search，不重跑 group discovery，所以 Δ 可能來自選中 token 的 seed 集中、事後閾值、full-window funding links 與 mark 估值。2024-10 的 random ≥ 為 0/10,000、2025-02 為 0/10,000，但 2025-06 仍有 194/10,000、2025-09 有 7/10,000 的 random 帳戶不小於主帳戶，顯示窗口差異大。更嚴格的 statistical statement 是 selection-matched nulls：只有 3 of 40 tests after Holm adjustment 被拒絕，且 nulls condition on the discovered funding graph，不校準整個 selection procedure。線上識別後，9 of 19 teams 只在最後一次 launch 後才可識別；其餘 10 隊的帳戶 free cash above one SOL in 7，且 9 of 10 accounts exceed at least 95% of random later launches，但 funding links 仍取自 full window，所以仍是樂觀上界。

## §6 · 失效與隱含假設

### 論文自述 limitations

- **§9**：funding links、ranking、thresholds、observation cohort 都用 full window；模型只加一個 follower deposit，未重建 rival responses、priority、failed inclusion、migration；marks 不是 cash。
- **§6.2**：online qualification 的 funding links 仍取自 full window，future edges／future wallet membership 可能影響 team composition，不能讀成 real-time tradable signal。
- **§7**：best/worst window 由自身結果選出，是 extreme cases；每個參數單獨變動，sweep 與 stop/graph variant 是看過結果後選的；team 與 threshold 由全窗口定義。
- **§6.5**：2026-09 最高 index 在 native SOL transfers 為 596.6，加入 owner-attributed WSOL edges 後為 5.9，低於 gate 10，selected teams 由 1 變 0；作者未隔離哪條 link 造成變化。endpoint-incidence cap 與 minimum transfer size 可使同窗口合格 group 數在 1 到 5 間變動。
- **§8 / Appendix D**：all thresholds chosen in hindsight；73% 的 opened positions 在小時末仍 open，持有 67% invested SOL；selling them at last admissible mark 把 geometric mean equity 由 3.82 提到 5.65，但依賴「可按 recorded reserve 賣出」假設。
- **§6.6 / §9**：窗口為 purposive、長度不同；不 annualize、不推 population trend、不主張 SOL-price causal effect。

### 推斷的隱含假設（本頁推斷）

- **Regime 依賴**：10 個窗口的 apparent opportunity 差異可能混合市場活躍度、creation 結構與 observation quality；不能把某窗口的 ranking 外推為所有市場狀態。此為本頁推斷，原文只說窗口 chosen for interest、differ in duration。
- **容量／流動性**：1 SOL 帳戶、10% cash fraction、0.1 SOL cap、1% reserve fraction 的結果不規模化；own deposit 只加一個 follower，未重建 pool 對其他人的回應。對大資金容量結論：未披露。
- **成本**：base fee 1% per side、slippage 0；2% fee + 5% slippage 情境下主帳戶 0.77 到 8.93 SOL，free cash 0.09 到 6.79 SOL，7 of 8 windows 仍 above one SOL。這說明成本敏感，但不是實際執行成本量測。
- **數據泄漏／選擇偏差**：funding graph 全窗口重建、team identity 事後排名、seed tokens 由 outcome 定義；Appendix A 移除 seeds 是 diagnostic，不是 trading scenario。線上 9 of 19 隊不可跟隨，已直接否證「所有事後大餘額都可交易」的讀法。
- **Survivorship**：窗口與團隊不是全體 Pump.fun 普查；missing paths 不補零，但 full-hour coverage 不等於完整交易。輔助數據缺失本身不是生存者偏差，然而用可用窗口比較期間差異仍有 selection 風險。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Szwajcok et al. [3] Pump.fun manipulation | 機制分類 vs 有限資本跟隨帳戶 | 未披露（本頁未核） | 最近平台文獻；不提供 creator-group following account |
| Chen et al. [16] / Li et al. [17] | rug-pull 偵測／早期詐騙預測 vs 事後現金帳 | 未披露 | 有明確 outcome/horizon；不是 launch 即刻可用的 self-financing signal |
| White [4] / Hansen [22] / Bailey et al. [5] | 多重搜索／backtest overfitting vs conditional randomization | 未披露 | 本文只做 conditional null，未跑 full-procedure null |
| Angeris [19] / Daian [20] | AMM 定價／ordering vs 單一 follower deposit | 未披露 | 承認 execution、latency、competition 未重建 |
| Gafarov [24] 前作 | team-level serial exits vs 10 窗口三資訊制 | 未披露 | 直接 lineage；本文加 cash/marks 拆分、online qualification、窗口比較 |

🎤 **Interview Tip**  
- **正確答**：這是 conditional retrospective model accounts；Equity 大不代表 cash 大，8 個有團隊窗口中 free cash >1 SOL 只有 5 個，marks 佔 17–94%；online 後 9 of 19 隊不可跟隨；selection-matched null 只有 3 of 40 經 Holm 顯著；不能說是可執行策略或 SOL 價格因果。  
- **錯答**：「11.32 SOL 是可複製 SOTA」「3 of 40 證明全流程顯著」「random 贏不了就是 alpha」——都混淆了 hindsight balance、conditional null 與 full-procedure selection。

### 可證偽預測（帶日期）

截止：來源發布日 **2026-11-08** 之後首個預先註冊樣本（具體日期 TBD）；以下為作者 §10 提出、本頁待驗的 hypothesis。

1. 若 funding graph 按事件時間逐邊重建，online 可跟隨比例會低於本文 §6.2 的 online 數字。  
2. full-procedure null（重跑 graph discovery、filtering、threshold optimisation、capital accounting）會比 conditional null 的 3 of 40 在更少窗口拒絕。  
3. compact fan-out groups 的帳戶表現會優於 long dispersal trees；若兩類無差異，則「團隊」語義假設被否證。

## §8 · For the Reader

- **因子研究員**：把 marks/cash 分開當第一公民；本文 17–94% marks share 是比 Equity 更重要的風險指標。random median 不是 SOTA，Δ 不能直接當 alpha。  
- **高頻執行**：base 4 s delay、1% fee、0 slippage 不是可執行假設；2% fee + 5% slippage 已使主帳戶 0.77 到 8.93 SOL。先做 latency／failed inclusion／pool liquidity haircut，再談策略。  
- **組合配置**：1 SOL 起始、10% cash fraction、0.1 SOL cap、1% reserve cap 的容量極小；marks 鎖住現金，73% positions 在小時末仍 open 且持有 67% invested SOL。  
- **LLM-agent／entity inference**：funding graph 是 candidate coordination group，不是 owner 或 fraud label；graph 規格可使合格團隊在 1 到 5 間變動，適合當 hypothesis generator，不適合當 identity ground truth。  
- **研究學生**：先讀 §9 的三種 statement（large model balance / above random / statistical confirmation）與 Appendix E 的 85 SOL boundary example；不要引用 11.32 SOL 當策略績效。

## References

- 原論文：Gafarov, F. *Retrospective Analysis of Team-Following Trading Profitability in Pumpfun Meme Coins Using Funding Networks*. arXiv:2611.11436v1, 2026. https://arxiv.org/abs/2611.11436v1  
- Lineage：Gafarov, F. *Serial Rug-Pull-Like Exits on Pump.fun: Team-Level Evidence and Its Sensitivity to Team Construction*. arXiv:2610.41597, 2026. https://arxiv.org/abs/2610.41597  
- 最近平台文獻：Szwajcok et al., *Meme Coin Factories: Uncovering Large-Scale Manipulations on pump.fun*, arXiv:2609.10246v1. https://arxiv.org/abs/2609.10246v1  
- 方法脈絡：White [4]；Hansen [22]；Bailey et al. [5]；Sullivan et al. [21]；Harvey et al. [23]；Angeris et al. [19]；Daian et al. [20]。  
- 來源鏈接：https://arxiv.org/abs/2611.11436