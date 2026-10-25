<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=多智能体博弈 autonomy=Agent自主演进 -->

# ACE-Trader 解構（ACE-Trader）

> **來源發布**：2026-10-19 · （無 venue） · arXiv [2610.25847](https://arxiv.org/abs/2610.25847)
> **arXiv 原文**：[ACE-Trader: Adversarial Co-Evolution for Generator-Critic Trading Systems](https://arxiv.org/abs/2610.25847v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文本另類 × 日頻波段 × 生成式大模型 × 多智能體博弈 × Agent自主演進」。它解的不是「怎麼讓 LLM 選股」，而是 **generator–critic 系統裡「兩個角色怎麼各自選到對自己有用的經驗」**——把記憶檢索從 recency/importance/relevance 的靜態排序，換成一個由「有記憶 vs 無記憶整條流水線淨收益差」監督的學習式排序器。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `多智能体博弈` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 給 Trader 與 Red-Team 各配一個獨立的 context-dependent memory selector，用輕量 MLP 集成（K=3）對候選記憶打分。② 核心 trick 是把「有記憶流水線的淨收益 − 無記憶流水線的淨收益」當成共享二元標籤，近因加權在線訓練 selector，LLM 與 encoder 全程凍結；再用 shared case memory 把 proposal／critique／revision／outcome 串起來，讓兩個角色互相對齊。③ 對「Agent自主演進」這條軸★的意義是：演化發生在**檢索層**而非權重層，成本是每天多跑一條完整 pipeline。④ 關鍵實證：Cum. Ret. +12.44 對最強主動基線 FinRL-DQN 的 +9.59，原文自述領先 2.85 percentage points；Sharpe 2.023、MDD 5.21 為全場最佳。

**X-Ray.** 這篇的真正貢獻不在「多智能體」，而在**把記憶選擇變成一個可以被反事實監督的分類問題**。過去的 reflection agent（FinMem、FinAgent 系）只能從「這筆交易賺不賺」倒推經驗好壞，問題是賺錢可能掩蓋一次錯誤的修正、虧錢可能掩蓋一次正確的預警——原文 §1 自己把這個 pitfall 說得很清楚。ACE 的做法是把 credit assignment 從「單筆交易的盈虧」升格為「整條 pipeline 的 A/B 差」，這在方法論上是對的：標籤乾淨、不需標註、可 online 更新。但它的 envelope 也很清楚——這個標籤只回答「這條記憶在這一步整體有沒有用」，原文 Appendix B 明講「does not measure the separate contribution of an individual memory」，所以它不是 RL 的 credit assignment，而是一個 contextual bandit。對量化讀者最該警惕的是：Turnover 0.153 對比 LightGBM 的 1.341、XGBoost 的 1.329，而 Equal-Weight Buy & Hold 的 MDD 是 5.33、Sharpe 1.375，與 ACE 的 5.21／2.023 貼得很近——**低換手分散組合本身就能吃掉相當一部分風險指標**，這條軸上的歸因在原文裡沒有被隔離。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（FinMem / FinAgent / TradingAgents / FinCon 類） | ACE-Trader |
|---|---|---|
| 記憶檢索 | 按 recency、importance 或語義相關度排序（原文 §2 歸納） | 學習式 selector：輸入 = 語義特徵 + 當前 market/portfolio 特徵 + 該記憶的歷史結果；集成打分 `s = μ + ασ` |
| 監督信號 | 自身反思、單筆決策的實現盈虧 | **有記憶 vs 無記憶兩條完整 pipeline 的淨收益差**，二元標籤 ζ，落在 ±ε 內者剔除 |
| 角色耦合 | 各 agent 獨立反思，或 debate 後聚合 | shared case memory 記錄 proposal/critique/revision/outcome，雙向派生 role-specific 經驗並互相連結 |
| 更新對象 | 提示詞、記憶文本、（部分）policy | **只有 selector 的 MLP 權重**；trading LLM 與 semantic encoder 凍結（§4.4、E.1） |

### 1.2 ⚡ Eureka 一句話 trick

> **不要問「這條記憶過去靈不靈」，要問「在當前這個 account state 下，帶上它比不帶它多賺多少」——然後拿這個差當標籤去訓一個排序器。**

直覺：把「記憶有沒有用」從一個**生成問題**（讓 LLM 自己反思）降維成一個**排序問題**（給定 context，K 個小 MLP 打分），於是可以用幾百個樣本在線學，而不必微調 backbone。α 項（分歧紅利）則是給探索留的口子：三個網路吵得越凶，越值得試。

### 1.3 信息流

```
b_t, o_t ──┬─→ [Retrieve] 候選 M_t^u  (private memory + shared cases)
           │        │
           │        ▼
           │   [Selector: K=3 MLP] → s=μ+ασ → m_t^u
           │        │
           ▼        ▼
      Trader(Commit)  ──a_t^Com, d_t^Com──→  Red-Team(Challenge)
           │                                      │ c_t (challenge / PASS)
           │                                      ▼
           │                              Trader(Revise) → a_t^Rev  ══執行══╗
           │                                                              ║
           └──────── shadow execute（同一 b_t、同價格、同費率）───────────╫─→ 記憶輔助淨收益
                    無記憶參考流水線（同 workflow，不召回）─────────────╫─→ 無記憶淨收益
                                                                          ║
                             ζ_t = 1 / 0 / 剔除  ◄── 差值與 ε 比較 ────────╝
                                    │
                    ┌───────────────┴────────────────┐
                    ▼                                ▼
        [Trader selector 更新]            [Red-Team selector 更新]
                    ▲                                ▲
                    └──── shared case memory（風險標籤 + 修正收益）────┘
```

## §2 · 數學層

📌 **Napkin Formula**

檢索打分（Eq. 2）：`s_{t,m}^u = μ_{t,m}^u + α · σ_{t,m}^u`，取 `argmax_m` 得 `m_t^u`；候選集空則 `m_t^u = ∅`。
選擇器損失（Eq. 6–7）：`L_t(θ_{i,k}) = Σ_j pow(γ, d_{t,j}) · ℓ_{t_j}(θ_{i,k}) / Σ_j pow(γ, d_{t,j})`，`ℓ_{t_j}` 是該交易步所有已選記憶對各自 `ζ_{t_j}` 的 BCE 平均，防止同一步多選擇被額外加權；`j` 只掃最近 W 個已觀測結算日。

直覺：Eq. 2 是「平均預測 + 分歧加分」，σ 大代表三個網路對這條記憶的價值沒共識，α 大就是把不確定性當成探索紅利。Eq. 7 是 recency-weighted BCE——`d_{t,j}` 是結算日距離，`γ=0.8` 讓舊樣本按距離幂次衰減，但**不丟**，這與「只用最新反饋」形成對照（§5.4 觀察 `γ=1.0` 時退化為等權，表現下降）。

複雜度：selector 是 `42→32→16→1`（Trader）與 `43→32→16→1`（Red-Team），K=3，梯度更新成本可忽略；**真正的成本在推理**——每個 trading step 要跑「有記憶」與「無記憶」兩條完整 Commit–Challenge–Revise 流水線（§4.4、Appendix B），LLM 呼叫量約為單條流水線的兩倍，且只有記憶輔助那條會被真正執行。

訓練細節：Adam，lr 0.01，weight decay `10^{-4}`，betas (0.9, 0.999)，數值穩定項 `10^{-8}`，`BCEWithLogitsLoss`；每個有效反饋事件對每個網路觸發一次梯度更新（E.4）。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下為機制示意，數字僅取自原文披露的超參與結構，不含任何論文實證結果。**

1. **假設**當前 phase 為 Challenge，候選集非空。query 與候選 m 各編成 `384` 維語義嵌入；兩個嵌入連同 element-wise product 拼接成 `1,152` 維 → 線性層壓到 `16` 維，再與 phase 特徵、該記憶的歷史結果拼接 → Red-Team selector 輸入 `43` 維。
2. **打分**：`K=3` 個 MLP 各輸出一个機率，記均值 μ、標準差 σ，`s = μ + 0.8σ`（原文 α=0.8）。若兩個候選 μ 相同，σ 大者勝出——這是 α 項在起作用。
3. **選出** `m_t^Cha`，塞進 prompt，Red-Team 產出 `c_t`（challenge 或 PASS）。若 PASS，跳過 Revise，`a_t^Rev = a_t^Com`，該步不產生 Revise 選擇樣本。
4. **收盤後結算**：在 `b_t` 的副本上 shadow execute 最終計畫與無記憶參考計畫，價格與費率完全相同，得兩個淨收益；差 > ε=`0.0005`（即 `0.05 percentage points`）→ ζ=1；差 < −ε → ζ=0；`|差| ≤ ε` → 這一步不進訓練集。
5. **更新**：拿 ζ 對第 2 步存下的表徵做 BCE，按 `γ=0.8` 的 settlement-day 距離幂次加權，只看最近 `W=5` 個已觀測結算日；同一 ζ 同時餵給兩個 selector，但各自只在自己選過的候選上訓。
6. **關鍵限制**：這個 ζ 是**整條 pipeline 級的**，不是某條記憶的功勞。原文 Appendix B 逐字承認它「does not measure the separate contribution of an individual memory」——這就是說，第 2 步裡「吵得凶」的那個候選如果最後贏了，其實無法區分是它的功勞還是 Red-Team 那一步剛好救回來。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| Universe | 美股 `33` 檔、跨 `11` 個 sector，全程固定不變（C.1） |
| 頻率 | 日頻；每日**開盤**決策並執行，反饋以「下一個共同交易日的收盤」評估（§5.1） |
| 訓練/暖機 | `2021-01-01` 至 `2023-12-31`，**無獨立驗證集**（§5.1、C.1、E.2） |
| 測試 | `2026-01-01` 至 `2026-06-30`，資料源改用 FMP（§5.1、C.1） |
| 歷史資料源 | FNSPID：`29.7 million` 筆股價紀錄、`15.7 million` 篇時間對齊新聞；本 universe 子集為 `26,305` 筆 asset–article 紀錄（C.1、C.2） |
| 新聞截斷 | 前一日（曆日）`20:00` America/New_York；只保留 cutoff 前七日內最多兩則 asset-specific 紀錄（C.2） |
| 成本 | 單邊 `5 basis points`（traded notional）；**借券費、額外滑價、現金利息均未建模**，閒置現金零收益（C.4） |
| 約束 | 多空毛曝險上限 `100%` of NAV；帳戶連續承繼、不每日重置（§3、C.4） |

時序邊界需分列（不可混用）：**預訓練 cutoff** — 原文僅稱測試期「falls after the documented training-data cutoffs of all evaluated LLM checkpoints」（§5.1），逐個 checkpoint 的具體截止日**所讀文本未見**。**適配/暖機邊界** — `2021-01-01` 至 `2023-12-31`，selector 從 PyTorch 預設初始化、記憶庫初始為空，按時間序跑一遍（E.2）。**測試邊界** — `2026-01-01` 至 `2026-06-30`。三者互不推導；`2024–2025` 期間如何處理，所讀文本未見。樣本外交易日數、AUM 容量、market impact 假設：**未披露**。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 原文 Reproducibility Statement：「Anonymous source code is provided with the submission」；公開 URL **未披露** |
| Checkpoint | 未發布自有權重；Trader 與 Red-Team 均用 DeepSeek V4 Flash，記憶編碼用 MiniLM，三者皆凍結（§5.1、E.1） |
| License | **未披露** |
| 複現難度 | 高：作者實驗使用 `16` 張 NVIDIA A800-SXM4-80GB、Ubuntu `22.04`（E.1）；最低複現硬體需求所讀文本未見。每個 trading step 需雙跑 pipeline；另需 FNSPID + FMP 兩套商業/公開資料源 |
| 數據可得性 | FNSPID 為公開資料集（Dong et al., 2024）；FMP 需訂閱，測試期切片是否釋出**未披露** |
| 基線可執行性 | 原文 E.5 稱所有開源基線用官方實作或 faithful wrapper，保留原架構與超參；但各基線授權**未披露** |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄按規則填**單一基線的逐字數值**；此處多數基線並非前 SOTA，逐行標注來源模型。Δ = 本方法 − 該基線，僅置於末欄。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 美股 33 檔，2026-01-01–2026-06-30 | Cum. Ret. (%) ↑ | FinRL-DQN +9.59 | +12.44 | +2.85pp |
| 同上 | Sharpe ↑ | Nasdaq Buy & Hold 1.340 | 2.023 | +0.683 |
| 同上 | MDD. (%) ↓ | Nasdaq Buy & Hold 12.84 | 5.21 | −7.63pp |
| 同上 | Sharpe ↑ | Equal-Weight Buy & Hold 1.375 | 2.023 | +0.648 |
| 同上 | MDD. (%) ↓ | Equal-Weight Buy & Hold 5.33 | 5.21 | −0.12pp |
| 同上 | Cum. Ret. (%) ↑ | Nasdaq Buy & Hold +12.82 | +12.44 | −0.38pp |
| 同上 | Cum. Ret. (%) ↑ | DeepSeek V4 Flash 直推 −19.72 | +12.44 | +32.16pp |
| 同上（消融） | Cum. Ret. (%) ↑ | Variant 4 +10.07 | +12.44 | +2.37pp |
| 同上（消融，基線為 Variant 4） | Sharpe ↑ | 1.529 | 2.023 | +0.494 |

其餘主動基線逐字值（供讀者自行對照，不合成區間）：LightGBM +4.86 / 0.502；XGBoost −21.18 / −1.610；LSTM +2.81 / 0.399；GRU +0.53 / 0.148；MLP −1.64 / −0.124；TRA +6.47 / 0.662；HIST +5.29 / 0.619；FinRL-A2C −16.84 / −2.245；FinRL-PPO −7.40 / −0.820；GLM 5.3 Flash −6.30 / −0.453；Qwen3.6-35B-A3B −13.18 / −1.046；FinGPT −10.84 / −1.003；FinCast −1.30 / 0.040；Kronos −30.45 / −3.753；FinAgent −15.95 / −1.495；FLAG-Trader −2.62 / −0.159；TradingAgents −9.42 / −0.769；LLMoE −7.33 / −0.795；TradExpert +0.10 / 0.072；MM-DREX −5.27 / −0.615。ACE-Trader 之 Vol. 12.35、HR. 57.38、Turnover 0.153。

**是真 capability 的部分**：同一 backbone（DeepSeek V4 Flash）下，直推是 −19.72、套上框架是 +12.44，這個對比在方法論上乾淨——差異只來自 workflow 與記憶，不含模型能力。Sharpe 2.023 與 MDD 5.21 同時稱冠，且 Vol. 12.35 低於 Nasdaq 的 20.10，說明不是靠加槓桿換報酬（gross cap 100%）。消融 Variant 1（+5.53 / 0.823 / 7.81）→ Variant 4（+10.07 / 1.529 / 6.36）→ 完整（+12.44 / 2.023 / 5.21）單調改善，至少內部一致。

**可能是別的東西的部分**：（i）Cum. Ret. +12.44 實際**低於** Nasdaq Buy & Hold 的 +12.82，原文 §5.2 自承「slightly below」——即絕對報酬沒贏被動。（ii）ACE 的 Turnover 0.153，而 Equal-Weight Buy & Hold 的 MDD 5.33、Sharpe 1.375 與 ACE 的 5.21／2.023 距離很近；一個幾乎不調倉的 33 檔分散組合，本身就能拿到大部分的 MDD 與 Vol 優勢。**Turnover-matched 的低換手被動基準在原文中不存在**，因此「記憶機制貢獻了多少風險指標」無法從這張表分離（本頁推斷，需作者補實驗）。（iii）測試窗口僅 `2026-01-01` 至 `2026-06-30`，交易日數未披露，無 seed 變異、無顯著性檢定；單一窗口的相關排序不能外推為普遍因果。（iv）消融 Variant 4 的 +10.07 與完整的 +12.44 差 2.37pp，同樣是單一窗口單次觀測。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位章節）

- **§5.2**：作者主動承認累積報酬「slightly below NASDAQ Buy-and-Hold」，僅以較低回撤與波動主張風險報酬取捨更優。
- **Appendix C.4**：借券費、額外滑價、現金利息皆未建模，閒置現金零收益——空頭部位的持有成本被系統性低估。
- **§5.1 / C.1 / E.2**：`2021-01-01` 至 `2023-12-31` 用於訓練與暖機但**無獨立驗證集**；selector 超參也在同一段歷史期決定後凍結。這是作者明示的設定，代價是超參無法用 held-out 選。
- **§4.4 / E.1**：trading LLM 與 semantic encoder 全程凍結，改進只能來自檢索層與記憶內容。
- **Appendix B**：共享標籤「does not measure the separate contribution of an individual memory」——credit assignment 粒度受限，作者明說。
- **獨立 Limitations 章節**：所讀文本未見（§6 為 Conclusion，無另設 limitations）。
- **圖表／附件**：Figure 3 的 α、γ 曲線具體數值未在可讀文本中列出（僅有文字描述「drops sharply at 1.0」「highest return at 0.8」），**待核**。

### 6.2 推斷的隱含假設（本頁推斷，非作者陳述）

- **固定 universe 的存續偏誤**：`33` 檔在訓練與測試全程固定。若這批代號是以「2026 年仍可交易」為前提挑出的，測試期的橫截面就被前置篩選過；輔助資料缺失不是 survivorship 的免責理由，此處的問題是**標的清單的時點**，所讀文本未說明選樣時點。
- **Regime 依賴**：α 與 γ 在歷史期定為 0.8，測試只有半年。若 2026 上半年是低波動震盪市，低換手策略天然佔優；表中的 0.153 是交易名義金額／決策前 NAV 計算的平均每日換手，並非調倉速率上限；不能據此斷言趨勢市或流動性緊縮期幾乎無法轉向。這些 regime 的執行敏感度仍待另測。
- **成本假設**：`5 basis points` 單邊對大型股合理，但 universe 明含 mid/small strata（C.1）；ACE 自身低換手受影響小，**基線受影響大**——這會系統性有利於 ACE 的相對排名（推斷）。
- **反事實標籤的有效性**：ζ 假設「有記憶」與「無記憶」兩條 pipeline 的差異可歸因於記憶；兩條流水線每步都在當前 `b_t` 的副本上，以相同市場價格、執行與成本規則配對比較（Appendix B）。標籤衡量當前帳戶狀態下的整體記憶效益，不估計兩條獨立持續交易軌跡的長期差異。
- **LLM 知識邊界**：原文僅稱測試期在所有 checkpoint 的 documented cutoffs 之後，未逐一列出日期（所讀文本未見）；DeepSeek V4 Flash 的 cutoff 具體日期無從核對。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| FinMem (Yu et al., 2025) | 分層記憶 + 角色設定，檢索靠 recency/importance；**無學習式 selector、無 red-team、無反事實標籤** | 所讀文本未見授權，TBD | 被列為 §2 對照組 |
| FinAgent (Zhang et al., 2024) | 多模態 + 多樣化記憶 + 反思，**單 agent**；ACE 的差異是把 critique 外包給獨立角色並讓它也有記憶 | 未披露 | 表 1 為 −15.95 / −1.495 |
| TradingAgents (Xiao et al., 2024) | 多角色 debate + 風險管理；**無「有記憶 vs 無記憶」的訓練信號**，角色經驗不互通 | 未披露 | 表 1 為 −9.42 / −0.769 |
| FinCon (Yu et al., 2024) | conceptual verbal reinforcement 做多 agent 信念更新；ACE 的差異是更新選擇器權重而非只改文字信念 | 未披露 | §2 引用 |
| FLAG-Trader (Xiong et al., 2025) | LLM + 梯度式 RL 微調 policy；**ACE 凍結 LLM，Trader 與 Red-Team 各訓 3 個小 MLP** | 未披露 | 表 1 為 −2.62 / −0.159 |
| 直接 prompt 同一 backbone | 無 workflow、無記憶 | — | −19.72 vs +12.44 |

🎤 **Interview Tip**

**正確答**：「ACE 的關鍵不是多塞一個 red-team，而是把『記憶有沒有用』變成一個可用反事實差監督的二元分類問題——標籤來自同一個 account state 下、有記憶與無記憶兩條完整 pipeline 的淨收益比較，而且Trader 與 Red-Team 各訓 3 個 MLP（各自為 `42→32→16→1` 與 `43→32→16→1`），LLM 凍結。它的 credit assignment 停在 pipeline 層級，本質是 contextual bandit，不是 sequential RL。」

**錯答**：「因為它用了更強的 LLM／更多 agent／更大的記憶庫。」——同一個 DeepSeek V4 Flash 直推是 −19.72，加上框架才到 +12.44，模型能力不是解釋項；每個角色的短期記憶上限為 `24` 筆；長期 lessons 另存且不計入該上限，shared case memory 也不能與短期上限混同。因此此比較不能單獨排除記憶規模的影響。

### 7.1 可證偽預測（帶日期）

1. **2026-12-31 前**：若匿名代碼釋出，以同一 `33` 檔 universe 重跑 `2026-01-01` 至 `2026-06-30`，ACE 的 Turnover 0.153 與 MDD 5.21 應可複現。若不能，表 1 的風險指標整體可疑。
2. **本頁預測（尚待驗證）**：加入一個 turnover 與 ACE 同量級的低換手被動／簡單規則基準後，ACE 相對該基準的 Sharpe 優勢會顯著縮小（以 Equal-Weight Buy & Hold 的 1.375 與 5.33 為參照點）。若優勢仍完整保留，本頁關於「低換手分散組合解釋大部分風險指標」的質疑即被證偽。
3. **本頁預測（尚待驗證）**：把測試期延伸到 `2026-07-01` 之後，或把 universe 擴展到含測試期內退市／新上市的標的，ACE 的 Cum. Ret. 會低於 +12.44。若不降，survivorship 質疑被證偽。

## §8 · For the Reader

- **因子／組合研究員**：真正可搬的是那個監督範式——用「帶該特徵 vs 不帶該特徵」的完整 pipeline 差當標籤，去訓一個 meta-selector。把它從「記憶」換成「因子暴露」或「風控開關」同樣成立。但移植前先做 turnover-matched 基準，別重蹈本頁指出的歸因陷阱。
- **高頻／執行**：與你無關。日頻、開盤執行、`5 basis points` 單邊、無滑價模型、無借券費；`0.153` 是平均每日換手，單靠此值不能判定執行成本是否充分受檢驗；真實滑價與借券費敏感度仍待檢驗。
- **組合配置**：把 ACE 當成「低換手、多空對稱、gross 100% 的 33 檔配置器」來看更誠實。它的 MDD 5.21 與 Equal-Weight 的 5.33 差距有限，別為 Sharpe 2.023 付超額管理費。
- **LLM-agent 研究者**：值得抄的是 shared case memory 的 schema（proposal / critique / revision / outcome / 風險標籤 / 是否 hit），不是 agent 拓撲。原文 Appendix B 那句「a correct warning can still be followed by a revision that reduces net return」是整篇最好的設計洞察：**風險判對 ≠ 修正有益**，兩者必須分開記。
- **RL 策略**：注意它不是 RL——沒有 bootstrapping、沒有 policy gradient，只有一步的二元回饋 + recency 加權 BCE。想接 sequential 版本，得先解決跨步 credit assignment，而這正是作者自己標為未解的部分。
- **研究學生**：`33` 檔 × 半年 × 無驗證集 × 無顯著性檢定 × 單一 seed 未披露。把 2.023 當「這個方法有效」的證據太薄；把它當「這個監督信號值得一試」的證據剛好。

## References

- 原論文：Zhou, Zhou, Yu, Cheng, Zhang, Zhang. *ACE-Trader: Adversarial Co-Evolution for Generator–Critic Trading Systems.* arXiv:2610.25847v1 [cs.LG], 19 Oct 2026. https://arxiv.org/abs/2610.25847v1
- Lineage（記憶型交易 agent）：Yu et al., *FinMem*, IEEE Transactions on Big Data, 2025；Zhang et al., *FinAgent*, KDD 2024；Yu et al., *FinCon*, NeurIPS 2024.
- Lineage（多智能體／critic）：Xiao et al., *TradingAgents*, arXiv:2412.20138, 2024；Gong et al., IJCAI 2025（generator–critic LLM agents）。
- Lineage（policy 優化對照）：Xiong et al., *FLAG-Trader*, ACL Findings 2025；Liu et al., *FinRL*, ICAIF 2021。
- 資料源：Dong, Fan, Peng. *FNSPID*, KDD 2024；Financial Modeling Prep（測試期）。
- 文本依據：本次輸入之 arXiv 完整可讀文本（含 §1–§6、Appendix A–E）；圖片（Figure 1–3）與匿名代碼附件**未提取、待核**。
