<!-- ontology-5axis data=图关系 horizon=高频日内 paradigm=监督回归 alpha=多智能体博弈 autonomy=人机协同可解释 -->

# Prospective multi-channel wallet-group discovery 解構（Prospective multi-channel wallet-group discovery）

> **來源發布**：2026-10-29 · （無 venue） · arXiv [2610.39458](https://arxiv.org/abs/2610.39458)
> **arXiv 原文**：[Early-Buyer Groups and Fast-Wallet Teams on a Bonding-Curve Launchpad: A Prospective Measurement Study of pump.fun on Solana](https://arxiv.org/abs/2610.39458v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 pump.fun 錢包群組從「人工標籤／token-level fraud」改成前瞻、多通道、無標籤驗證的群體測量問題；五軸雖標圖關係／日內／監督回歸／多智能體博弈／可解釋協同，但主貢獻是測量與否證，不是交易策略。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `图关系` | `高频日内` | `监督回归` | `多智能体博弈` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①用 88.3 小時 pump.fun 全量鏈上資料（200,443 launches、15,854,121 curve trades、51,025,958 SOL transfers）前瞻建出 23 710 個錢包群組。②trick 是多通道證據（共同買入、共同出資、聯合簽名、轉帳、合併）建群，並用樣本外連結預測、獨立非交易連結與「等量 flagged tokens」naive co-buy 基線做無標籤驗證。③關鍵負結果：等預算下 naive co-buy components 的 AP 0.585 高於 discovery groups 0.317（prospective 0.162 對 0.055），follower 在 32 個配置全部虧損，event costs 下 -7% 到 -4%。④團隊層在 hold-out 仍有約一半到四分之三精度，但 26% 到 40%（tuned）信號出現時 curve 已達 20 SOL，且對 60s 後增量無貢獻；所以這是描述性結構，不是可跟單 alpha。

**X-Ray.** 這篇的 Pareto 位置不是「更好的預測器」，而是「把群組結構的可驗證性與可交易性分開」：無標籤驗證只給內部一致性與獨立連結，等預算 naive 基線反而揭示 group score 多半在量 first-minute crowd size；精確儲備與持倉會計則把常見回測假利潤打回負值。它解了 label-free validation 與 execution leakage 兩個舊坑，但打不開的 envelope 很清楚：單一 88.3 小時窗口、無 ownership ground truth、秒級區塊解析度、無法表示同塊 submission／tips／bundle，且最大團隊信號集中、事後到達。對量化讀者，價值在 negative result 與 evaluation protocol；不是 alpha 因子。

## §1 · 架構 / Core Mechanism

### 三大改動 vs 前作

| 改動 | 前作常見做法 | 本文 |
|---|---|---|
| 前瞻多通道建群 | token-level manipulation／fraud 標籤、bundle traces、ownership heuristic clustering | 以 co-buying、shared funding、joint signing、transfers、consolidation 建 core+support 群；hub 錢包排除；決策只用 t 前資訊 |
| 無標籤驗證與等預算基線 | 依賴 labels／ownership heuristics，或不同 coverage 下比較 | 樣本外 link prediction、independent non-trading links、naive co-buy components／Louvain 在 equal numbers of flagged tokens 下對照 |
| 精確儲備執行與評估審計 | price path／return 常忽略 trader 自身持倉、mark-to-market 或 label leak | exact post-event reserves、exit state 加入 own position、兩種成本慣例、latency 1/2/3/5 s；明文記錄 label leak 與 own-position 造成的假利潤 |

### Eureka 一句話 trick + 直覺

**一句話：** 把「群」從標籤問題改成前瞻假設——多通道證據建群，無標籤用樣本外連結預測與等量 naive 基線驗證；再用精確儲備與持倉會計檢驗可交易性。  
**直覺：** 若群組真的帶資訊，它應該在 equal-budget 下贏過只數 crowd 的 naive score；若真的可跟，follower 在 latency 與成本後不應全負。兩個測試都用來否證，而不是用來堆 lift。

### 信息流 ASCII 圖

    Solana blocks
      -> block-by-block decoder
         -> {token creations, curve trades, graduations, AMM trades, SOL/SPL transfers}
            -> evidence channels: co-buy / co-sell / shared funding / joint signing / transfers / consolidation
               -> core+support group builder (prospective; hubs excluded)
                  -> earliest prospective signal
                     -> follower sim: exact reserves + own position + latency/costs
                  -> label-free validation:
                     link prediction / independent non-trading links /
                     naive co-buy & Louvain at equal flagged tokens
                  -> team layer: merge shared-wallet groups + fast-wallet Louvain
                     -> final hold-out block C + timing / concentration / incremental value

## §2 · 數學層

📌 **Napkin Formula**：follower 淨收益  
`r = (P_out - P_in - f_in - f_out - 2c) / P_in`  
其中 `c` 是 network fee；`P_out` 的 reserve state 必須加入 trader 自身 entry 的 position。  
**直覺：** 這條式子把「看到信號」與「吃到價格」分開；若不把 own position 加回 reserve，exit quote 會假設池裡有交易者自己剛存進去的 SOL，於是產生不存在的流動性與假利潤。  
**複雜度／訓練：** 群發現與驗證的計算複雜度所讀文本未見，圖表／附件待核；主流程是 streaming 解碼加 pair evidence，本頁推斷為事件量級，非原文複雜度分析。監督模型只用於次要的 entry-ranking 與 team-membership prediction：LightGBM 用 squared error、`n_estimators=400`、`learning_rate=0.03`、`num_leaves=31`、`min_child_samples=300`、`subsample=0.8`、`reg_lambda=10`、early stopping `40`；GNN 預設 `2` layers、`32` hidden units、`300` epochs。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為**假設／示意**，用原文規則門檻示範流程；不是論文實證結果，也不新增樣本統計。

1. **假設／示意**：token X 創建後，錢包 A、B 在前幾秒共同買入；A、B 在三個共同 token 上共買，對應原文「至少三個共享 token」的接受規則示例（此處僅為規則示例，不是樣本統計）。  
2. B 與 C 有共同出資（同一 funder 經少數 hops 在一日內），但 C 與 A 無直接邊。  
3. 核心＋支持規則：A-B 成 core；C 只有 B 一個 bridge，依原文「single bridge never merges two populations」不能併入；若 C 也獲 A 支持，才併入群組。  
4. 對 token X 取最早可用 attribution，記下當下 group size 與是否 creator-linked；follower 在 signal 與 creation 後 `4` 秒中較晚者之後的第一個 curve observation 買 `0.05` SOL，延遲 `1`、`2`、`3` 或 `5` s。  
5. 出場 quoted on reserve state 必須加入 own position；否則 exit 的 SOL 會少於 trader 存入，對應原文 own-position 審計。pipeline 自己的 earliest signal 只有約 `4%` 在偵測時已達 `20` SOL，這是它比 team signal 更即時的原因。

## §3 · 數據層

| 項目 | 數值 |
|---|---|
| 觀測窗口（UTC） | 2026-09-26T19:00:23 – 2026-09-30T11:16:00 |
| Duration | 88.3 hours |
| Tokens created / complete | 200,443 / 154,045 |
| Bonding-curve trades | 15,854,121 |
| System-program SOL transfers | 51,025,958 |
| PumpSwap trades / graduations | 5,565,871 / 5,862 |
| Protocol transactions | 21,183,236 |
| Tokens whose curve reached 20 / 60 SOL | 7.7% / 2.4% |
| Tokens with at most 3 trades | 26.3% |

資料來自在 Solana 上逐 block 解碼的自建 collector，涵蓋 token creations、curve buys/sells、graduations、AMM trades 與所有 native SOL／SPL token transfers；每筆事件帶 slot、block time 與 transaction 在 block 內的位置。樣本外設計分兩半、70/30 policy split、55/15/30 chronological split、block A/B/C 與 final hold-out；final hold-out 是 36,532 complete tokens、base rate 8.6%。容量假設很小：follower 用 `0.05` SOL，position sizes `0.01` 到 `0.5` SOL 的結果差異小於 one percentage point；這是小額相對 curve 的假設，不是大資金容量證明。本研究無預訓練權重、tokenizer、適配或 topic 擬合；時間邊界只有資料窗口與 block A/B/C，模型訓練截止未披露。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 未披露（原文只說 replication package；URL 所讀文本未見） |
| Checkpoint | 未披露（無模型 checkpoint；規則與 CSV 在 replication package） |
| License | CC BY 4.0 |
| 複現難度 | 高：需自建 block-by-block collector、解碼兩協議、完整 SOL/SPL transfers；本頁推斷 |
| 數據可得性 | 公開 release 以 salted hash 替換 wallet 與 mint、salt 不公開、raw database omitted；replication package 含 CSV 與單元測試 |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄是模板名；來源沒有 SOTA，只有 naive／reference 基線。逐行列出基線數值，不合成區間；Δ 固定為「本方法 − 基線」。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 第二半窗口；基線=co-buy components, K=3（非SOTA） | AP（reaches 20 SOL any time） | 0.585 | 0.317 | -0.268 |
| 同上 | P@2000（any time） | 88.2% | 53.4% | -34.8pp |
| 第二半窗口；基線=co-buy components, K=3（非SOTA） | AP（prospective: still below 20 SOL at 60 s, reaches later） | 0.162 | 0.055 | -0.107 |
| 同上 | P@250（prospective） | 32.4% | 12.8% | -19.6pp |
| 同上 | P@500（prospective） | 29.3% | 10.2% | -19.1pp |
| 同上 | P@2000（prospective） | 19.9% | 8.4% | -11.5pp |
| 第二半窗口；基線=team layer, agreement score（非SOTA） | P@500（prospective） | 18.5% | 10.2% | -8.3pp |
| 測試期 last 30% signals；基線=random entry, project costs, latency 1 s | mean net return per trade（buyers-only follower） | -7.9% | -7.8% | +0.1pp |
| 同上；event costs, latency 1 s | mean net return per trade（buyers-only follower） | -3.6% | -4.0% | -0.4pp |

**解讀。** 這不是「打敗 SOTA」的表：co-buy components 是 naive 基線，卻在 equal numbers of flagged tokens 下 rank 得更好，prospective outcome 尤其明顯（原文 P@500 約 three in ten 對 one in ten）。真正有 capability 的是 *結構*：群有有限大小、每條邊有 evidence，可檢查 persistence 與現金流；但 accepted edges 的樣本外共買 persistence 在 stratified on co-buy count 後接近 one，independent non-trading links 雖有 27% 對 0.6% 的共買率，equal co-buy counts 下也不加 persistence。價格路徑的關聯在 creation covariates 下大（Table 7 `+7.6 [+6.8, +8.2]`），但 trimmed 到 common support 只剩 `+0.7 [ - 0.3, +1.6]`；members cash flow 對 matched placebo 是 `+0.17 [+0.12, +0.23]`，late buyers 是 `- 2.34`，但這是 within-token cash flow，不是可實現 alpha。Follower 在 32 個配置全負；event costs 下 -7% 到 -4%，project costs 下 -11% 到 -8%，random entry 同量級，說「detectability ≠ exploitability」。Team hold-out 的 tuned rules 在 2,359／904／615 個 signal tokens 上給 50.4%／70.6%／73.0% reach 20，但 already at 20 SOL 的比例是 26%／37%／40%，top team share 62% 到 79%，且 group features 對 60s 後 outcome 的 incremental AP 是 `-0.015`（interval `-0.030` 到 `+0.004`）、AUC change `0.000`。

## §6 · 失效與隱含假設

### 論文自述 limitations

- **One window（Section 10）**：88.3 小時、單一 regime；群 lifetime 受窗口限制；20／60 SOL 是事先選定的 round numbers。
- **No ground truth（Section 10）**：驗證只給內部一致性與 independent-link agreement，不是 ownership；acceptance rules 在 equal co-buy counts 下不加 persistence，naive co-buy 在 equal budgets 至少一樣好。
- **Time resolution（Section 10）**：block time 以秒為單位；submission order、tips、bundle membership 不可見；同塊 front-run 可能無法表示。
- **Execution model（Section 10）**：reserve-based quotes 是近似；own position 只 first order 加入；未送真交易；confirmation delay、priority fees、bundle inclusion 未建模。
- **Few large teams, one dominant（Section 10）**：一個 126-wallet team 供應 most large-team signals；其餘候選 tokens 較少、控制平衡較差，結論是 case-based。
- **Residual confounding and overlap（Section 10；§6.4）**：group tokens 與 controls 的 covariate balance 不完整；部分 covariates 是 post-treatment；price-path contrast 在 common support 限制下不存活。
- **Single-wallet signals（Section 10）**：多數 earliest signals 是同時為 creator 的單錢包；協調 claims 只靠 multi-wallet subset。
- **Exploratory analyses（Section 10）**：team layer、refinements、case studies、parameter sweeps 都在同一窗口發展；§8.10 是時間內 hold-out，不是獨立樣本。
- **Signal timing（Section 10；§8.10）**：team signal 需 third member 已買；大比例 flagged tokens 在信號時已達 20 SOL。
- **Incomplete observation（Section 10）**：約四分之一 launches incomplete 且排除於 outcomes；group-control contrasts 不受差分影響，但 absolute base rates 條件化於有 recorded trading。
- **Concurrent work（Section 10）**：其他 pump.fun 測量樣本、定義、目標不同；直接可比的 cohort preprint 已撤回數值，無法 head-to-head。

### 推斷的隱含假設

- **「群」是協調假設，不是 ownership**：作者明說不主張共同所有權、不標記 manipulative；因此任何「群＝同一操作者」的解讀都超出資料。  
- **資金可見性假設**：多數 team wallets 在窗口內沒有可見 inbound SOL；作者明說這與 concealment 或窗口前 ordinary funding 都相容，不能分離（§8.8）。  
- **Hub 分類假設**：exchange flag 為空，靠 distinct senders 分 hub／mid／specific；這是 coarse proxy，不是交易所標籤（§8.8）。  
- **First-minute controls 是 post-treatment**：作者明說 events、buyers、curve SOL 受 members buying 影響，調整可能移除部分效果；因此 trimmed 後關聯消失不能讀成「無效果」，只能讀成「無法區分 selection 與 early activity」（§6.4）。  
- **Team 信號的可交易性假設**：規則需要 first minute 的交易；作者明說 none of the rules is usable by a follower（§8.6），且 final hold-out 多數信號太晚（§8.10）。  
- **容量假設**：follower 用 0.05 SOL、sizes 0.01 到 0.5 SOL；本頁推斷這不能外推到大型資金或同塊 bundle 玩家。  
- **Regime 外推無效**：四天屬同一 market state；作者明說 window too short for a test across regimes（§9）。  
- **Incomplete exclusion 不是 survivorship 斷言**：作者檢查 exclusion 幾乎由 creator initial buy 決定，group-control contrasts 不受差分影響；但 base rates 條件化於有 recorded trading。本頁不把它寫成「只留下贏家」的 survivorship bias。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Token-level manipulation／fraud detectors [2,4,5] | token labels vs wallet-group structure | 否（樣本／定義不同） | 本文不主張超越；補上 group-level validated structure 與 cash-flow 關聯 |
| Address clustering／Sybil detection [53-65] | ownership heuristics／labels vs behavioral evidence, no labels | 部分 | 群是 coordination hypothesis；channels 在 equal co-buy counts 下不加 persistence |
| Link prediction／GNN [77-82] | learned graph vs simple smoothing + boosting | 是（方法可復用） | 隨機 split 下 GCN 0.870、GraphSAGE 0.882，低於 SGC-style smoothing + LightGBM 0.919；whole-team hold-out 約 0.72 對 0.735 |
| Naive co-buy clustering | equal-budget signal quality | 是 | naive components 在 AP／P@K 至少一樣好；discovery 的價值是 descriptive structure |
| Pump.fun measurement [2,3,41,42] | sample／definition／goal | 否 | 作者說不能 head-to-head；[42] 數值已撤回，[43][44] 是 pipeline audit |

🎤 **Interview Tip**  
**正確答：** 本文把「發現群」與「可交易 alpha」分開；無標籤驗證只能給內部一致性與 independent-link agreement，不能證明 ownership；equal-budget naive co-buy 基線顯示 group score 多半在量 first-minute crowd size；follower 在 32 個配置全負，latency 1 s 已負；team hold-out 精度約一半到四分之三，但 26% 到 40%（tuned）信號在 curve 已達 20 SOL 後才出現，且對 60s 後增量無貢獻。  
**錯答：** 「多通道建群打敗 naive 聚類、所以有 alpha」「GNN 提升團隊預測」「group features 有增量資訊」——這三句都與原文的 negative／null result 相反。

### 可證偽預測（帶日期）

**2027-06-30 前（本頁預測日期，非原文數字）**：在至少一個獨立觀測窗口（非 2026-09-26–30）重跑 §8.10 的 frozen-rule hold-out；若 team features 在 curve SOL at 60s 之外仍無 AP／AUC 增量，且 signal timing 後置比例不降，則本文「描述性關聯、非可跟單 alpha」的讀法被支持；若出現穩定正增量，且 follower 在 `1 s` latency、event costs 下平均淨收益為正，則本頁推論被證偽。

## §8 · For the Reader

- **因子研究員**：把 group features 當 crowd／early-state proxy，先做 incremental value beyond curve SOL at 60s；不要只看 lift，必須用 equal-budget AP／P@K 與 naive co-buy 基線。  
- **高頻執行**：`1 s` latency 已把 gross edge 吃掉；本文秒級解析度不能表示同塊 submission、tips、bundle。若要真做，先補同塊可視性與 priority fee，而不是調策略參數。  
- **組合配置**：不要把 reach 20／60 SOL 當 graduation alpha；graduation 是 2.8%，team excess 集中在 20 SOL，60 SOL 與 graduation 弱且不穩；單窗口、單一 dominant team，不能當跨 regime 配置依據。  
- **LLM-agent／RL 策略**：evidence channels 可當 feature／reward shaping，但必須保留 label-free validation 與時間 hold-out；GNN 未勝簡單 smoothing，且 label leak／own-position 會制造假盈利。  
- **研究學生**：先讀 Table 3 與 §7.4 的 evaluation pitfalls；這篇的賣點是 negative result、E-value、equal-budget 基線與 exact-reserve 會計，不是又一個 lift 故事。

## References

- Gafarov, F. *Early-Buyer Groups and Fast-Wallet Teams on a Bonding-Curve Launchpad: A Prospective Measurement Study of pump.fun on Solana.* arXiv:2610.39458v1, 2026. https://arxiv.org/abs/2610.39458v1 （CC BY 4.0）
- Lineage：pump.fun／memecoin 測量 [2][3][4][5][40][41]；address clustering／Sybil [53]–[65]；link prediction／GNN [77]–[82]；empirical-Bayes shrinkage [100]；E-value [103]；execution／backtest pitfalls [83]–[90]。
- 來源鏈接：https://arxiv.org/abs/2610.39458