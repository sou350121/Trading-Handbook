<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=组合执行优化 autonomy=全自动黑盒 -->

# Coherent-set convex hull + integer programming arbitrage 解構（Coherent-set convex hull + integer programming arbitrage）

> **來源發布**：2026-10-16 · （無 venue） · arXiv [2610.23517](https://arxiv.org/abs/2610.23517)
> **arXiv 原文**：[Combinatorial Arbitrage in Prediction Markets](https://arxiv.org/abs/2610.23517v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點＝微觀盤口 × 高頻日內 × 因果（邏輯）結構 × 組合執行優化 × 全自動黑盒。它補的 prior gap 很具體：過去文獻只做「等價/相同合約」的 pairwise 錯價（Rothschild & Pennock 2014；Saguillo et al. 2025；Gebele & Matthes 2026），本文把 mutual exclusivity / partition / nesting / conjunction 四類關係**聯合**編進同一個 constraint system，並在 full-depth 盤口上量「可執行」而非「可觀察」的利潤。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把每個合約的 payoff 寫成 binary vector，邏輯關係寫成線性約束 `Fz ≤ g`，則「報價是否可套利」等價於報價向量是否落在 coherent set（payoff vectors 的 convex hull）之外，且最大保證利潤恰為其 ℓ1 距離；用 cutting-plane（Master LP + integer Oracle）避開 2^|I| 枚舉，再把線性成本換成逐檔盤口階梯得可執行規模。關鍵工程是關係發現的三層 scale-out：venue-asserted（negRisk group）／venue-derived（sports typed fields 映射到共同比分狀態）／semantic-inferred（parser + 5 個 proposal LLM + 2 個 auditor 雙審），共 1,942,133 條關係壓到 573,024 條、603,242 列約束。對「因果結構」這條軸的意義：alpha 不在預測方向，而在**關係圖的覆蓋率**——同一份盤口，誰的 F 更完整，誰的 payable set 更小、保證利潤更高；但這也立刻畫出它的天花板：來源未給 Sharpe/IR，只有 2.2% 的保證 ROIC 與 $2.50 million 的絕對規模。

**X-Ray.** 這篇不是 trading-strategy paper，是 market-microstructure measurement paper，讀它要換一把尺。在五軸 Pareto 上它犧牲了「時間尺度」的一致性去換「結構正確性」：五軸標「高頻日內」，但實測持有期中位在 venue-derived 是約 3 小時、semantic-inferred 是 7 天，資金加權平均持有期 1.6 天（accumulated take）——所以它不是 HFT，而是「高頻偵測 + 中頻持有」的混合體，把它塞進日內 slot 會錯估資本效率。它解掉的舊工程坑有兩個：一是把「關係發現」從人工/單點 pair 升級成可規模化的三層管線並用 Type I/Type II 不對稱性做風控（Proposition 4：漏關係只會少賺，不會假賺）；二是把「幾何判定」升級成 MILP，讓 32/48/16 檔的 active block 在 3.6 billion 個 best-quote state 上可算。它打不開的 envelope 也很清楚：**資金需求**——capital outstanding 峰值 $11.1 million、first take 峰值 $3.5 million；**可複現性**——作者自建 full-depth collector，本頁未取得其歷史深度資料，無法獨立重放同一樣本；**執行**——leg risk 未解，1 秒延遲下 22.1% 的 settled orders 在到期時虧錢。對量化讀者最該記住的一句：這篇的真正 asset 是「約束系統 + 三層關係圖」，不是那 2.2%。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（Rothschild & Pennock 2014；Saguillo et al. 2025；Gebele & Matthes 2026） | 本文 | 讀者為什麼在意 |
|---|---|---|---|
| 關係類型 | 以 identical／equivalent payoff 為主，pairwise | 四類聯合：mutual exclusivity、partition、nesting、conjunction，寫成 Table 2 的線性列，同一 constraint system 內 jointly 求解 | 「等價」只是 partition 的特例；真正的錯價大頭在 nesting（本文 reduced 系統中 351,233 條）與 exclusivity（193,934 條） |
| 關係來源 | 單一來源、常需人工核對 | 三層：venue-asserted（negRisk group + group rules）／venue-derived（match×contract type×line → 共同比分狀態，121–1,681 個 state）／semantic-inferred（parser + LLM 雙審） | 分層不是為了美觀，是為了**歸因與隔離**：三層各自獨立求解，一層的錯誤關係不會污染另一層的假保證 |
| 執行度量 | quote-level 或 field trades | full-depth ladder 的 cost function `K(y)=ΣK_i(y_i)`、fee-adjusted、shared liquidity ledger 防重複吃同一檔 | 「有價差」與「能吃下多少」是兩個問題；存在性由最佳檔決定，規模由深度決定 |
| 求解器 | 枚舉／幾何／pairwise 判斷 | Kelley (1960) cutting-plane：Master（LP/MILP）+ integer Oracle；5-leg cap 使 Master 成為 MILP | 直接枚舉 Φ 在 5 檔連通分量就有 2^5=32 個向量，乘上約 3.6 billion 個 quote state 不可行 |

### 1.2 ⚡ Eureka

**一句話**：套利的存在性與其最大保證利潤，可以完全寫成「報價向量到 coherent set 的 ℓ1 距離」；而那個距離的 separating direction 就是交易本身——所以你不需要知道真實機率，只需要一個 integer oracle 幫你找出「最壞狀態」。

直覺：coherent set M 是所有「某個信念 q 下的期望 payoff」的集合。報價落在 M 內 → 存在一個機率分布同時為所有合約定價 → 無套利。報價落在 M 外 → 沒有單一信念能合理化這組價格 → 分離超平面就是那個保證賺錢的組合。這個轉譯的威力在於：把「找尋套利」變成「求一個 ℓ1 投影」，再變成「一個 LP 加一個分離 oracle」。

### 1.3 信息流

```
Polymarket Gamma API (catalog + resolution text, 745,000 contracts)
  ├─ negRisk group id + group rules ──────────────→ Layer 1（20,118 partition / 748 exclusivity）
  ├─ typed sports fields (match / type / line) ───→ score-state grid → Layer 2（1,358,323 recovered）
  └─ question + resolution text
        ├─ 數值/時限 parser ─────────────────────→ Layer 3
        └─ pre-screen（277 billion pairs → 704,450 pairs / 13,894 groups）
              → 5 proposal LLMs（Qwen3-32B / Qwen3-235B-A22B / Llama-4-Maverick
                                 / DeepSeek-V4-Flash / GLM-5.2）
              → 2 auditors（Claude Fable 5.1 / GPT-5.6 Sol，後補 GPT-6 Astra），
                 僅在兩者對 type+direction 一致時保留
                        ↓
        每層各自 (F, g) → connected components（semantic 層：3,591 個，中位 4 檔，最大 19,188）
                        ↓
WSS full-depth books → 依 exchange timestamp 重建 best quotes（97.2% / 13.3 billion 次比對一致）
  → 每個 relevant update 跑 cutting-plane detector（block caps 32/48/16、leg cap 5）
  → arbitrage windows: 491,939
                        ↓
full-depth ladder replay + shared liquidity ledger（同一檔位不可重吃）
  → executed windows: 125,095（25.4%）
                        ↓
resolution records（僅 binary 0/1 才計 settled）
  → guaranteed profit 2.2% / profit at resolution 3.2%（settled, accumulated take）
                        ↓
on-chain OrderFilled（227 million events → 143.53 million trades）
  → 逐 leg 方向比對 + 反向 placebo → 44.1% / 14.5% / 8.8% 有同向 matched trade
```

## §2 · 數學層

📌 **Napkin Formula**

$$
\max_{\|y\|_\infty\le 1}\ \min_{\omega}\ y^\top\!\big(\phi(\omega)-p\big)\;=\;\min_{\mu\in M}\|\mu-p\|_1,\qquad M=\operatorname{conv}\{\phi(\omega)\}
$$

帶成本版本：

$$
\Pi^\star=\max_{y\in D}\ \min_{z\in\Phi}\big\{y^\top z-K(y)\big\},\qquad D=\prod_i\big[-Q^b_i,\ Q^a_i\big]
$$

偵測用 nested program（Master/Oracle）：`max_{y,η} η − y^⊤p  s.t.  y^⊤z ≥ η  ∀z ∈ Φ_work`，Oracle 解 `min_z y^⊤z s.t. Fz ≤ g, z ∈ {0,1}^I`。

**直覺**：最壞情況 payoff 是 y 的線性函數在所有可行 payoff vector 上的最小值；線性函數在點集與其 convex hull 上取到同一個最小值，所以「離 coherent set 多遠」＝「能保證賺多少」。**複雜度**：直接枚舉 Φ 最壞是指數（worst case may still require every vector）；cutting-plane 每次迭代新增一個未見過的 payoff vector，有限步終止。5-leg cap 引入 `x_i ∈ {0,1}`（`−Q^b_i x_i ≤ y_i ≤ Q^a_i x_i`、`Σx_i ≤ 5`），Master 變成 MILP；但 cap 只作用在 Step 1 選腿，Step 2 的數量重優化仍是 LP。**求解器耗時／迭代次數**：所讀文本未見，圖表／附件待核。

**Loss / 訓練**：無論文式訓練。LLM 部分無 fine-tune，只有 prompt + 雙審過濾；`r`（taker fee rate）在樣本內由 0.03 升至 0.05（2026-07-10 起，sports）。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ 以下全是**假設／示意**的玩具數字，用來把 §3.1–3.3 的機制手工跑一遍；**不是論文實證結果**。論文真實數字留在 §5。

1. **假設**：三檔互斥且窮盡的 partition 合約 A/B/C（例：主勝/和/客勝），fee-adjusted best bids 分別是 `0.34 / 0.34 / 0.35`。約束系統只有一列：`z_A + z_B + z_C = 1`。
2. **方向**：三個 bid 相加 `0.34+0.34+0.35 = 1.03 > 1`。取組合 `y = (−1,−1,−1)`（各賣 1 股），在任何可行頂點上 `y^⊤z = −1`，初始現金收入 `1.03`，最壞情況付 `1` → **保證利潤 0.03 / 股**。
3. **對照 ℓ1 距離**：`min_{μ∈M} ‖μ − p‖₁`，M 是三個頂點 `(1,0,0),(0,1,0),(0,0,1)` 的三角形；`p=(0.34,0.34,0.35)` 到三角形的 ℓ1 距離正是 `1.03−1 = 0.03`，與第 2 步一致。
4. **定規模（示意）**：假設三檔 bid 側深度分別 `500 / 300 / 200` 股，且**只看最佳檔**。規模受最淺那條腿限制 → 賣 200 組，保證利潤 `200 × 0.03 = 6.00`。真實情況會往更深檔走、賣價更低，這就是 `K_i(s)` 階梯在侵蝕利潤——「大價差 + 薄深度」可能不如「小價差 + 厚深度」。
5. **反面檢查**：三個 fee-adjusted bid 之和不超過 1，只排除了這個 partition 的賣出籃子機會；仍須核對 ask 側與其他有效關係。一般條件是 `M ∩ ∏[b̃_i1, ã_i1] ≠ ∅` 才無可執行套利（Proposition 3）。

**論文內的同型真實案例**（不是示意）：Algeria–Austria（2026-06-28），fee-adjusted bids 合計 `1.0222`，replay 鑄造 `68` 個 basket 花 `$68`、賣得 `$69.51`、當場賺 `$1.51`；同一場比賽 02:28:24 的 fee-adjusted asks 合計 `0.9285`，買 `136` 股各檔花 `$126.28`，到期賺 `$9.72`——**賣方能立刻兌現、買方要等結算**，這正是 negRisk conversion 的不對稱性。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 市場／venue | Polymarket 單一 venue（作者說明選它是因為報價全深度可串流、listing-time 問題鏈上可回收、每筆成交在 Polygon 上結算；Kalshi 為集中清算，published trades 無法連結帳戶） |
| 期間 | 23 June 至 3 August 2026，42 天 |
| Catalog | 約 745,000 個 binary contracts、178,000 個 event pages、20,866 個 negative-risk groups |
| 盤口 | 73.22 billion order-book messages，縮到約 3.6 billion 個 valid two-sided quote states（偵測用）；pooled median spread 8.13 cents，median best bid depth 40.20 股 / best ask 29.46 股 |
| 成交 | 227 million OrderFilled events → 143.53 million observed trades、$5.20 billion volume、677,000 個 distinct trader addresses |
| 頻率 | 事件驅動（每個 best-quote update 評估一次）；無抽樣，保留 exchange 與 receipt timestamps |
| 取得方式 | catalog 走 Gamma API；**full-depth 必須自建 live collector**（Polymarket 無歷史深度 feed），最多 600 tokens / WebSocket，每 3 小時刷新合約清單 |
| 樣本外 | **未披露**。單一 42 天窗口、無 holdout、無跨 venue 驗證 |
| 容量假設 | capital outstanding 峰值 $11.1 million（accumulated）/ $3.5 million（first take）；同一美元跨 execution 重複計入 |

**讀者要注意的數據層陷阱**：這份數據是**採集時的 catalog text**，不是 listing-time 規則。作者在 B.1.1 明說「We do not compare that text with the listing-time record」，所以關係是相對於檢索時的顯示規則稽核的——這是一個潛在的前瞻/事后偏誤入口，不是作者隱藏，而是資料本身如此。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露（所讀文本未見連結） |
| Checkpoint | 無自訓模型；proposal 用 5 個開源權重（Qwen3-32B、Qwen3-235B-A22B、Llama-4-Maverick、DeepSeek-V4-Flash、GLM-5.2），audit 用 Claude Fable 5.1 / GPT-5.6 Sol / GPT-6 Astra 的 agent 模式 |
| License | 論文 CC BY-NC-SA 4.0；程式碼授權未披露 |
| 複現難度 | 極高。三個獨立障礙：(1) full-depth 盤口無歷史 feed，必須自建 collector 且只覆蓋未來；(2) on-chain OrderFilled 解碼需對齊兩份 CTF Exchange V2 合約與 token id；(3) LLM 雙審依賴作者記載的特定模型與 reasoning-effort 設定，後續重跑需固定版本與輸入；本頁尚未獨立複現 |
| 數據可得性 | catalog 與 on-chain fills 公開可重建；**full-depth 私有**。作者提到 record code / solver 版本 / input checksums，但未給出取得途徑 |

## §5 · 評測 / Benchmark

> **這篇沒有外部 baseline 對照。** 全文的比較都是內部的：三層之間的 ablation、robustness settings、以及 matched-trade 的 placebo。因此「前SOTA」欄一律填未披露——硬把別的論文數字塞進來會製造假的 Δ。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Polymarket 2026-06-23–08-03（settled，accumulated take） | 保證利潤／投入資本 | 未披露 | 2.2% | — |
| 同上 | 到期利潤／投入資本 | 未披露 | 3.2% | — |
| 同上 | 保證利潤（千美元） | 未披露 | 2,498 | — |
| 同上 | 到期利潤（千美元） | 未披露 | 3,626 | — |
| 同上 | 投入資本（千美元） | 未披露 | 113,716 | — |
| 同上（first take，settled） | 保證利潤／投入資本 | 未披露 | 2.7% | — |
| 同上（first take，settled） | 保證利潤（千美元） | 未披露 | 1,104 | — |
| Layer 1 venue-asserted（settled，accumulated） | 保證 ROIC | 未披露 | 3.8% | — |
| Layer 2 venue-derived（settled，accumulated） | 保證 ROIC | 未披露 | 1.9% | — |
| Layer 3 semantic-inferred（settled，accumulated） | 保證 ROIC | 未披露 | 1.5% | — |
| Polymarket 全樣本，order delay 1,000 ms | 到期利潤（千美元） | 未披露 | 2,874 | — |
| Polymarket 全樣本，order delay 1,000 ms | 佔 baseline 到期利潤 | 未披露 | 79.3% | — |
| Polymarket 全樣本，partial participation 50% | 保證利潤佔 baseline | 未披露 | 56.5% | — |

**哪些是真 capability**：Layer 1 的 3.8% 與 Layer 2 的 1.9% 差距，不是模型好壞，而是**機制紅利**——negRisk 的 basket conversion 讓賣方可以鑄造後立刻賣出、當場取回抵押，不必等結算；作者在 §5.2 明說這類 legs「capital need not remain committed until resolution」。這部分是結構性優勢，換 venue 就消失。整體 2.2% 的保證利潤則建立在「displayed quotes 可成交、後續報價維持不變」的假設上（作者自己把 first take 當下界、accumulated take 當上界）。

**哪些可能是過擬合／偏差**：(1) 42 天單一窗口，sports 佔 listed contracts 67.5%、佔 reduced 關係 453,190 條，樣本期的世界盃賽程密度直接放大機會數；(2) settled-only（89.8%）把 semantic 層砍到 62.8% settled，因為它 resolve 得晚——這讓 Layer 3 的 1.5% 建立在偏小的成熟樣本上；(3) capital 跨 execution 可重複計入同一美元，ROIC 的分母不是 unique capital；(4) 未計自身市場衝擊、queue position、maker 反應，只做了 cancel-and-repost（10s/60s/120s 三檔，保證利潤維持在 100.0%/100.5%/100.7% 的 baseline）與 participation（90%/70%/50% → 92.4%/75.3%/56.5%）兩種壓力測試。

**成本是真實的**：order delay 1,000 ms 下只有 54.3% 的 orders 全額成交、成交 84.0% 的提交量、到期利潤掉到 baseline 的 79.3%，且 **22.1% 的 settled orders 在到期時虧錢**。這一行是整篇最該被引用的數字：leg fill 不均時，保證就不再是保證。

**Sharpe / IR / MDD**：所讀文本未見，圖表／附件待核。這是一篇 measurement paper，沒有風險調整後績效。

## §6 · 失效與隱含假設

### 6.1 論文自述的限制（可定位章節）

- **必須持有到結算**（§3.1 footnote 5、Appendix A.3）：提前平倉的利潤取決於未來的 spreads/depth/fees，形成時不可知；因此實證全部 hold to resolution，「early-exit profit cannot be guaranteed ex ante」。
- **Type II 誤差保守但不免費**（§4.1、Proposition 4）：漏掉有效關係只會讓 Φ 變大、保證利潤變小（lower bound），但會降低 detection power。semantic 層的覆蓋**明確不完備**：候選檢索、model abstention、有限算力都可能漏。
- **層間比較不是因果隔離**（§5.3、§6）：三層在合約、流動性、conversion 機制上都不同，「the evidence is consistent with both discovery and execution costs limiting arbitrage」，但無法把 discovery cost 與 execution cost 分開。
- **資本重複計入與上下界**（§5.1、Table 5 notes）：capital「can count the same dollar more than once」；first take 是下界（假設窗口內不再執行），accumulated take 是上界（假設後續報價維持觀察值）。
- **只評 binary settlement**（§5.1）：void 合約付半美元、open leg 歸 unsettled；只有 filled legs 全部 resolve 0/1 的 window 才計 settled。
- **事後更正**（Appendix B.3 "Corrective excision"）：2026-09-22 稽核發現 6 條 parser 生成的關係錯誤地施加了集體窮盡性，在 316 個受影響 window 中剔除 9,172 筆 execution，first take 減 $2,201、accumulated take 減 $17,404；其中 **82 個 window 在結算時是賺錢的**——作者據此指出「profitable settlement does not validate an ex-ante guarantee」。
- **關係稽核基準**（Appendix B.1.1）：用檢索時保存的 catalog text 稽核，未與鏈上 listing-time `QuestionInitialized` 對照。
- **深度完整性未驗證**（Appendix B.4）：best-quote 對賬只能驗證最佳檔，作者明說「cannot establish the completeness of depth at other price levels」。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **regime 依賴（推斷）**：機會分布與賽程密度共變。sports 佔 listed contracts 67.5%、crypto 22.4%，兩者合計佔 reduced 關係 94.6%。換一個沒有大型賽事的 42 天窗口，樣本結構會顯著不同——這是適用條件，不是必然結論。
- **容量（推斷）**：capital outstanding 峰值 $11.1 million 是回放中同時占用資金的觀察峰值，不是策略容量上限，也不是可擴張曲線。
- **成本結構可變（推斷）**：sports taker fee 在樣本內由 0.03 升到 0.05；作者另查了 20,866 個 group 的 conversion fee 全為零。若 conversion 開始收費，Layer 1 的 3.8% 直接受壓——這是本文最脆弱的單點依賴（輔助數據缺失不構成生存者偏差，但構成本文結論的機制前提）。
- **數據泄漏／前視（推斷）**：LLM 審計讀的是 retrieval-time 的 resolution text；若賽後文本被改寫，關係可能偏離 listing-time 規則。作者已聲明未對照，因此這是**已知未關閉**的風險口，不是我指控它作弊。
- **執行假設（推斷）**：shared liquidity ledger 只防「同一檔位重複吃」，未建模排隊位置、maker 撤單反應、以及自己下单造成的價格移動。cancel-and-repost 檢查的結果（三檔 retention 下保證利潤仍在 baseline 的 100.0%–100.7%）應理解為「對這一類摩擦不敏感」，不是「無摩擦」。
- **樣本選擇（推斷）**：Layer 3 的 62.8% settled 與較晚到期有關；settled-only 結果不等於全樣本無條件回報。這不足以斷言所有 survivorship／選擇偏差均已排除。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Rothschild & Pennock (2014), *Algorithmic Finance* | 只做 identical / logically related 的 pairwise 錯價，以 field trades 測流動性 | 未披露（所讀文本未見） | 2014 |
| Saguillo et al. (2025), AFT 2025 | 用 LM 找 dependency，量 in-market 與 pairwise equivalent-payoff 的套利 | 未披露 | 2025 |
| Gebele & Matthes (2026), arXiv:2601.01706 | 跨 venue 的 equivalent / subset 合約對 | 未披露 | 2026 |
| Capponi, Gliozzo & Zhu (2025), arXiv:2512.02436v2 | agentic clustering + semantic trading（統計交易，非保證利潤） | 未披露 | 2025 |
| Kroer et al. (2016), EC'16 | **maker 側**：用 IP 維持 combinatorial market 的 arbitrage-free 定價 | 未披露 | 2016 |
| Dudík, Lahaie & Pennock (2012), EC'12 | constraint generation 讓 coherent pricing tractable，無實證 | 未披露 | 2012 |
| **本文** | **taker 側**：四類關係聯合進同一 constraint system + full-depth 可執行量測 | 未披露 | 2026 |

**🎤 Interview Tip**

問：「預測市場的組合套利，你怎麼量『可執行』利潤？」

**正確答**：分兩步，且兩步用的資訊不同。第一步用 **best quotes** 判定存在性——等價於檢查 coherent set M 是否與 fee-adjusted 的 spread box `∏[b̃_i1, ã_i1]` 相交（Proposition 3）；不相交才有套利，且方向就是 separating hyperplane。第二步用 **full-depth ladder** 定規模——存在性只看最佳檔，規模要看 `K_i(s)` 階梯，因為大價差配薄深度可能不如小價差配厚深度。利潤的定義是 `min_{z∈Φ} y^⊤z − K(y)`，是**最壞情況**不是期望值；Φ 用 integer oracle 按需生成，不枚舉 2^|I|。最後一定要報 leg risk：作者自己的 1 秒延遲測試顯示 22.1% 的 settled orders 到期虧損。

**錯答**：「把同一事件所有 Yes 價格加起來，偏離 1 就套利。」——三個錯誤：(1) 忽略 partition 是否窮盡，20,866 個 negRisk group 裡只有 20,118 個在規則上窮盡，其餘 748 個只能下 mutual-exclusivity 不等式，買方套利不成立；(2) 忽略 taker fee 與逐檔深度，fee-adjusted 後價差可能消失；(3) 忽略 Type I 錯誤的方向性——加錯一條約束會**憑空造出**保證利潤（作者用 9,172 筆被剔除的 execution 展示了這個代價），而漏一條只會少賺。

### 7.1 待驗證的研究假說（本頁提出）

以下都需要新增實驗，不能當作本文已驗證的因果結論。以 2027-06-30 為覆核日期：

- 在關係集合與樣本固定時，測試原子多腿成交能否降低 leg mismatch 導致的虧損；只改執行條件，另報成本。
- 在相同 catalog 與稽核準則下重做 semantic 關係提案，對照錯誤類型、漏檢與保證利潤；雙審通過率不能當成獨立 precision／recall。
- 用另一個採集窗口重跑三層分析，檢查來源組成、結算成熟度與流動性改變後，持續時間和觀察參與率的關係是否仍成立。相關排序仍不足以證明發現成本的單一因果效果。

## §8 · For the Reader

- **因子研究員**：不要把 2.2% 當成 Sharpe 來比。這是扣 spread 與 taker fee、條件於已結算組合的投入資本回報，沒有 IR/MDD（未披露）。可借鏡的是**訊號構造**：coherent-set 的 ℓ1 距離是一個有理論保證的 cross-sectional 錯價度量，比「價格和減 1」穩健得多。
- **高頻執行**：真正的戰場是 leg risk。作者的 1 秒延遲實驗（54.3% 全額成交、84.0% 數量、79.3% 利潤、22.1% 虧損單）就是你的 baseline。分腿成交會留下方向曝險；應以自己的成交與成本模型評估尾部風險，不能由本文推定所有多腿策略的 PnL 分佈。
- **組合配置**：回放的同時占用資金峰值 $11.1 million、資金加權持有期 1.6 天、中位持有 3 小時到 7 天。這是 niche 中的 niche，配置前先問：這個 capacity 夠不夠覆蓋你的 infra 與 LLM API 成本（成本明細未披露）。
- **LLM-agent 研究者**：三層架構是很好的 **proposal-aggregate + dual-audit** 範本——5 個 proposer 的 union set 提案 239,653 條、只有 102,288 條（42.7%）過雙審。pre-screen 把候選空間由 277 billion pairs 收斂至 704,450；全文沒有完整分項成本，不能據此判定主要成本來自哪一步。
- **RL 策略**：cutting-plane + integer oracle 可以直接當 action space 的 **feasibility projector**：任何 policy 提出的組合，先以 oracle 評估最壞情況 payoff；如何把結果轉成 policy 約束，仍需額外設計與驗證。這是本頁提出的研究構想，本文未驗證其對 RL 的效果。
- **研究學生**：理論骨幹只有三個 proposition（§3.1 的 1、2 與 §3.3 的 3），Appendix A.1 的證明完整且自足；先讀這三條再看 §5 的實證，理解成本最低。

## References

- 原論文：Hongjian Shi, Daojing Zhai, *Combinatorial Arbitrage in Prediction Markets*, arXiv:2610.23517v1 [q-fin.TR], arXiv 收錄日期 2026-10-16；稿件頁首自述 first version 2026-09-15、this version 2026-10-10（分列兩種日期，不互相替代）。License: CC BY-NC-SA 4.0. <https://arxiv.org/abs/2610.23517v1>
- 理論 lineage：de Finetti (1937) Dutch book；Kelley (1960) cutting-plane；Grötschel, Lovász & Schrijver (1988) separation oracle；Rockafellar (1970) convex analysis；Jouini & Kallal (1995)、Pennanen (2011) 交易成本下的套利。
- 組合市場設計 lineage：Hanson (2003)；Chen et al. (2008)；Dudík, Lahaie & Pennock (2012)；Kroer et al. (2016)。
- 預測市場實證 lineage：Wolfers & Zitzewitz (2004)；Rothschild & Pennock (2014)；Sonnemann et al. (2013)；Saguillo et al. (2025)；Gebele & Matthes (2026)；Capponi, Gliozzo & Zhu (2025)。
- 限制套利 lineage：De Long et al. (1990)；Shleifer & Vishny (1997)；Pontiff (2006)；Grossman & Stiglitz (1980)；Cohen & Lou (2012)；Moskowitz (2021)；Stulz (2007)。
- 來源鏈接：arXiv abs 頁 <https://arxiv.org/abs/2610.23517>；本頁事實全部取自上述可讀全文文本，圖片（Figure 1–6、A.1–A.3、B.1）與附件未核。