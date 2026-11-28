<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# Point-in-Time Encoders 解構（Point-in-Time Encoders）

> **來源發布**：2026-11-10 · （無 venue） · arXiv [2611.14325](https://arxiv.org/abs/2611.14325)
> **arXiv 原文**：[Point-in-Time Encoders for Earnings Calls](https://arxiv.org/abs/2611.14325v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文字另類 × 日頻波段 × 監督回歸 × 端到端表徵 × 全自動黑箱」，但真正的支柱在 **檢索端**：它把 point-in-time（PIT）紀律從 LM 權重推進到 retriever 的 encoder 本身，補的 prior gap 是「ChronoBERT/DatedGPT 只定時了語言模型，餵給它／餵給回測的 encoder 仍可能讀過未來文本，而這條泄漏未被量測」。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①以 412,248 通財報電話會為語料，按 cutoff 2010–2024 逐年訓出四種配方的 encoder vintage，每條 vintage 只讀 cutoff 前的文本。②核心 trick 不是模型而是**任務來源與量測設計**：用電話會自帶的「分析師問→高管答」配對做 InfoNCE，免人工標註；再用「後一個 vintage 替換當時可用 vintage」的差值當前視溢價，並以 replay control 把「讀得更多」與「讀得更晚」拆開。③這對「Alpha 端到端表徵」軸是一個反面證據：PIT 溢價在**檢索**上真實存在、在**報酬預測**上量不到，端到端表徵的泄漏通道與它的收益通道不重疊。④from-scratch vintage 15 季平均 within-call recall@1 為 0.727\,844\,444\,444\,444\,4，對 e5-base 的 0.583\,711\,111\,111\,111\,1 差 0.144\,133\,333\,333\,333\,34，且每一季為正。

**X-Ray.** 這篇的價值不在「又一個更好的 retriever」——它 33.826\,176 M 參數打贏最高 335M 的一般檢索器，靠的是**域內對比監督 + 可審計的 cutoff**，不是表徵魔法；作者在 §V 承認一般檢索器 cutoff 未知、早期季度可能反被有利，並把配方結論全部建立在「固定對比微調、只動預訓練」的比較上——這是本文方法論上最誠實的一處。真正的 Pareto 位移在**可部署性**：它把「回測能不能用這個向量」從一句口頭保證，變成 registry 裡可強制執行的 `query_date ≥ cutoff` 失敗規則。它打不開的 envelope 也很清楚：有效的 PIT 監督需要**結構化的問答對**，作者在 §VIII 明說新聞、文件、券商報告多數沒有這種輪次結構，所以這套方法在「文字另類」軸上其實是**窄頻**的。對量化讀者最該記住的是那個負結果：把 2024 vintage 換上去，chained 家族在次日報酬預測的 2024-vintage 溢價為 -0.0022，from-scratch 家族為 -0.0023；六個目標 × 四家族 × 兩種後置 vintage × 有/無基線共 96 個預測溢價，用 moving-block bootstrap 只有 2 個過 Bonferroni 界，全在 earnings surprise 上——這意味著「文字 encoder 有前視泄漏」與「這個泄漏能轉成超額報酬」是兩個必須分開證的命題。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（ChronoBERT / DatedGPT / PALM / 金融 retriever） | 本文 PIT encoders |
|---|---|---|
| 被定時的對象 | 解碼器 LM 權重（DatedGPT、Kelly et al.）；PALM 不重訓、只掛低秩適配 | **檢索 encoder 本體**，逐年 vintage + registry 強制 cutoff |
| 監督信號 | 一般文本 MLM；金融 retriever 使用至少部分由語言模型生成的查詢或相關性標籤（cutoff 未聲明） | 電話會**自帶日期**的「分析師問→高管答」配對，免標註 |
| 訓練配方 | 通常單一（from scratch 或 chain） | **四配方並列** + replay control（同 token 數重讀舊文本）隔離「訓練量」與「新文本」 |
| 評測對象 | 組合信號、解碼器 forecast、凍結嵌入 IC | held-out within-call / cross-call recall@1 + 前視溢價 + 帶 reader 的 RAG 對數機率 |

### 1.2 ⚡ Eureka

> **把「同一批問題、換一條 vintage 來答」的差值定義成前視溢價，再用 replay 對照組證明這個差不是「多訓了幾步」——於是「encoder 讀過未來」第一次變成可測量的量，而不是一句假設。**

直覺：如果後一個 vintage 只是因為多讀了一年而變強，那讓它**重讀自己讀過的那一年、燒掉同樣的 token**，它應該也一樣強。實測 replay 對 vintage Y 是 -0.002\,738\,095\,238\,095\,238、14 季只有 3 季為正，可見「多訓」不漲；而 vintage Y+1 對 replay 是 0.006\,857\,142\,857\,142\,859、14 季有 13 季為正，可見漲的是**新文本**。

### 1.3 信息流

    Capital IQ 電話會 412,248 通 / >16,000 家 / 2004-09 – 2026-09
      │ 切塊 ~2,400 字元（~500 tokens）；丟 operator 輪；QA 配對兩側 ≥120 字元、每通 ≥5 答
      ├── 10% 依 SHA-1(event id) 整通留出 ──────────► 檢索題庫（除 2010 tokenizer 擬合外，任何階段不進）
      └── 可訓練電話會，按 event date 排序
            └── 窗口 Y := event_date ≤ Y-12-31
                  ├─[配方1] from-scratch：每 Y 獨立、1.2B tokens、MLM 30%
                  ├─[配方2] chained：Y-1 續訓 +1 pass over year Y（2024 累計 3.5B）
                  ├─[配方3] ChronoBERT vintage Y + call 對比微調
                  └─[配方4] ChronoBERT vintage Y + DAPT 1.2B + 對比微調（僅 2010/2017/2024）
                        │ InfoNCE τ=0.05；batch 32 通 × ≤16 對；4,000 steps；配對日期 ≤ Y-12-31
                        ▼
            registry(cutoff=Y)：查詢日期 d 只看得到 d 之前的通話；點名更晚 cutoff 的 vintage → 失敗
                        │
         ┌──────────────┼────────────────┬──────────────────────┐
         ▼              ▼                ▼                      ▼
     within/cross-call  ridge IC      50 近鄰（無參數）      RAG：vintage 取 3 段
     recall@1(held-out) 次日/5日/盤後/次季  next surprise        → ChronoGPT-Instruct 續寫對數機率
         └──────────────┴────────────────┴──────────────────────┘
                        ▼
       前視溢價 = 「後一個 vintage 的分數」 − 「當時可用 vintage 的分數」
       ＋ replay control（同段重讀同 token 數）→ 分離「訓練量」與「新文本」

## §2 · 數學層

📌 **Napkin Formula**

    L_InfoNCE = − log [ exp(sim(q,a⁺)/τ) / Σ_{j∈batch} exp(sim(q,a_j)/τ) ],  τ = 0.05
    sim(u,v)  = cos( mean_tok(u), mean_tok(v) )       # mean pooling over token vectors

**複雜度**：預訓練為 768-token 序列、30% 隨機遮罩的 MLM，固定 1.2B tokens／vintage；對比微調對**每個** vintage 都是固定 4,000 steps（batch 32 通 × ≤16 對，平均 370–405 對），因此步數與窗口大小脫鉤——同一預算下，2010 vintage 把每對讀約 3 遍，2024 vintage 只讀到 2/5 的對（表 III）。

**直覺**：分子把自己那條答案拉近問題，分母把所有 batch 內其他答案（同通電話 ≤15 條 + 其他通話的）推遠。負樣本的數量由 batch size 決定，所以他們用 gradient caching 把小 GPU 的梯度拼成全 batch。**關鍵取捨**：固定步數換來跨 vintage 的可比性，代價是晚 vintage 的窗口被「讀薄」——這正是 §7 裡 from-scratch 系列在 2025Q2 上 2024 反而低於 2023 的原因（作者自陳）。

**訓練細節**（表 III，逐字）：AdamW β=(0.9,0.98)、ε=10⁻⁶、weight decay 0.01、grad clip 1、線性 warmup 後 cosine 衰減到峰值 1/10；峰值學習率 預訓練 `6×10⁻⁴` / DAPT `5×10⁻⁵` / 微調 `2×10⁻⁵`，warmup 佔比 6% / 6% / 5%。問題截 256 tokens、答案截 512 tokens。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部為示意假設，不是論文實證；論文真實數字只出現在 §5。**

1. **假設**某 batch 只有 4 個候選答案（1 正 3 負，簡化），問題 q 與它們的餘弦相似度為 `s = [0.60, 0.45, 0.30, 0.10]`，τ = 0.05。
2. 縮放：`s/τ = [12, 9, 6, 2]`；取指數：`[162754.79, 8103.08, 403.43, 7.39]`；分母和 ≈ `171268.69`。
3. 正樣本機率 `p⁺ ≈ 162754.79 / 171268.69 ≈ 0.9503`；損失 `L = −ln(0.9503) ≈ 0.05096`。若某個負樣本被拉近到 0.58（`s/τ = 11.6`），`p⁺` 掉到約 0.83，損失升到約 0.19——這就是「同通電話內的其他高管答覆」作為 hard negative 的來源。
4. **溢價分解示意**：假設 vintage Y 在測試季 recall@1 = `0.500`，vintage Y+1 在同一批問題上 = `0.520` → 前視溢價 = `+0.020`。
5. 加 replay：把 vintage Y **重讀自己那一年**、燒掉與 Y+1 相同 token 數，得分 `0.495`。則「額外訓練」貢獻 = `0.495 − 0.500 = −0.005`（負，重讀有害或無益）；「更新的文本」貢獻 = `0.520 − 0.495 = +0.025`。**論文的整條因果鏈就靠這一步**：若 replay 也漲，溢價就不能歸因於「讀到晚的文本」。
6. **反例邊界**：若某家族是固定 token 預算的 from-scratch，晚 vintage 的窗口更長、每年被讀得更薄，第 4 步的 `+0.020` 有可能被讀薄抵銷成 0——這正是論文觀測到的「no net gain」，作者特意聲明這應讀作「無淨增益」而非「無泄漏」。

## §3 · 數據層

| 項目 | 內容（逐字） |
|---|---|
| 語料 | Capital IQ transcripts on WRDS，**412,248** 通電話會，**>16,000** 家，**2004-09 至 2026-09** |
| vintage 讀取範圍 | 讀到 2024 年；測試用到 2025 年 |
| cutoff 網格 | 2010–2024 每年一條（表 I：calls 39.4k → 358.8k；tokens 0.30B → 2.61B；pairs 0.49M → 3.81M） |
| 2010 窗口規模 | 0.302\,647\,296 B tokens（作者用以說明「遠小於一般 encoder 的數千億」） |
| 切塊 | ~2,400 字元 / ~500 tokens；QA 段以分析師提問開頭 |
| 配對過濾 | 兩側各 ≥120 字元；每通電話至少 5 條合格答覆 |
| 留出 | 10%，依 `SHA-1(event id)` 前四 byte 落在最低十分位——**by call**，任何 chunk／pair 都不進訓練 |
| 多版本問題 | Capital IQ 平均一通 3.2 個版本，保留字數最多者；作者量化其可能晚於 cutoff 的比例：2011–2024 各 cutoff 為 0.317\,939\,602\,631\,275\,6 % 至 1.394\,138\,149\,090\,427\,9 %，2010 cutoff 為 6.684\,182\,275\,334\,067 % |
| 市場資料 | CRSP daily stock file、Fama-French 日頻因子、Compustat 標準檔（含重編）算 standardized earnings surprise |

**三條時間邊界必須分列（不可互相套用）**：

- **預訓練截止**：vintage Y 只讀 event_date ≤ Y-12-31 的**可訓練**電話會（Y = 2010…2024）。
- **Tokenizer 擬合截止**：單一 byte-level BPE、32,768 tokens，**僅在 2010 年以前的通話上擬合（含 held-out）**——此後所有 vintage 共用、永不重擬合。作者另特意把每個數字拆成單獨 token，避免 2008/2015（單 token）與 2019/2024（雙 token）的切分本身洩漏年代。
- **適配（對比微調）截止**：每個 vintage 只用**日期 ≤ 該 vintage cutoff** 的電話會問答對；4,000 steps 固定。
- **測試截止**：主要檢索測試對每個 cutoff Y 在 Y+1 年第二季取樣，2010–2024 cutoff 對應 **2011Q2–2025Q2，共 15 季**；2025Q2 也是 2024 vintage 的主評測季，另有同季全 vintage 比較，不能只將它列成額外測試。

**容量假設**：所有權重皆為 33.8M（自有）或 149.0M（ChronoBERT 系）級別，單張共享叢集 GPU（多數抽到 RTX 2080 Ti）即可訓完；作者未給資料吞吐或延遲數字（未披露）。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| `github.com/Alessiobrini/point-in-time-encoders-for-earnings-calls`（§I 腳註 1，逐字）——含 code、call identifiers、result files、training records | 未披露（文本未明說是否發權重） | 未披露 | 中：硬體門檻低（單 GPU、from-scratch 預訓練約 20 小時／chained segment 約 1 小時／對比微調約 6 小時），難點在語料授權 | **transcripts 有授權、不再分發**；只能靠 Capital IQ/WRDS 自行取得，再用公開的 hash 規則重建留出集 |

留出集設計值得記一筆：規則是「`ecalls-holdout-v1:` 前綴 + event id 的 SHA-1 前四 byte 落在最低十分位」，**不需儲存清單**，任何人只要有 identifiers 就能復現同一分割——這是這篇在可審計性上最實的一手。

## §5 · 評測 / Benchmark

> **前SOTA 欄一律放來源逐字給出的基線數值。來源從未稱這些對手為 SOTA，它們是一般檢索器基線（e5-base / bge-small）與 encoder 基線（FinBERT / raw ChronoBERT），下表照此標示，不改稱 SOTA。** Δ 定義為「本方法 − 基線」，僅在能由同列兩個逐字數字驗算時填入。

| 數據集 / 市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 電話會 held-out 問答，15 季平均（基線 e5-base，general web，109.5M） | within-call recall@1 | 0.583\,711\,111\,111\,111\,1 | 0.727\,844\,444\,444\,444\,4 | 0.144\,133\,333\,333\,333\,34 |
| 同上（基線 bge-small，general web，33.4M） | within-call recall@1 | 0.538\,444\,444\,444\,444\,4 | 0.727\,844\,444\,444\,444\,4 |  |
| 同上（基線 e5-base） | Recall@5 | 0.879\,711\,111\,111\,111\,2 | 0.951\,711\,111\,111\,111\,1 |  |
| 電話會 held-out，6 季（基線 FinBERT，語料至 2019） | within-call recall@1 | 0.225\,644\,444\,444\,444\,45 | 未披露（from-scratch 的 0.727\,844\,444\,444\,444\,4 為 15 季均值，季集不同，不可比） |  |
| 電話會 held-out（基線 raw ChronoBERT，無對比階段） | within-call recall@1 | 0.200\,333\,333\,333\,333\,36 | 未披露（對比微調後僅報增量 0.558\,777\,777\,777\,777\,8，水平未直接給） |  |

**檢索解讀（逐字數字為主）**

- 對 e5-base 的 within-call 差 0.144\,133\,333\,333\,333\,34 **在每一季為正**；cross-call 對 e5-base 為 0.137\,488\,888\,888\,888\,9、對 bge-large 為 0.123\,311\,111\,111\,111\,11。cross-call 任務本身要打折看——作者引用 [27] 指出通話表徵大量編碼「公司身分」，跨通話辨識公司即可得分，所以他們**以 within-call 為主**。
- 表 IV 的成對差（同批 held-out 問題、逐題配對、以通話為重抽樣單位）：from-scratch 減 e5-base `+0.144 [ +0.139 , +0.149 ]`（15 季）；from-scratch 減 FinBERT `+0.508 [ +0.499 , +0.518 ]`（6 季，自 2020Q2 起）；ChronoBERT+對比微調減 raw ChronoBERT `+0.559 [ +0.554 , +0.564 ]`（15 季）——**這一條最乾淨地說明「對比階段才是 retriever 的本體」**，raw encoder 只到 0.200\,333\,333\,333\,333\,36。
- **反向的一條**：from-scratch 減 ChronoBERT+對比微調 = `-0.031 [ -0.035 , -0.028 ]`。即 33.8M／1.2B 域內 tokens 打不過 149.0M／460B–525B 一般 tokens + 域內對比微調。作者結論是「模型的規模與讀過的文字量，和文字領域的影響**差不多大**」——這是本頁認為最值得記住的實務論斷：**PIT 紀律不是免費的，代價就是你不能用那 460B tokens 的後見之明**。
- 種子敏感度：對比微調換三個種子、預訓練固定，from-scratch 的 recall 範圍是 0.001（平均）／0.006（單一 vintage 最大極差）。作者用 0.006\,333\,333\,333\,333\,413 這條線去檢定 chained 對 from-scratch 的 0.006\,928\,571\,428\,571\,429（自 2012Q2 起）與雙倍 tokens 的 0.018\,088\,888\,888\,888\,887（15 季）——**兩者都只是「略超過種子噪聲」的量級，不是壓倒性差距**。

**預測解讀（哪些是真 capability，哪些要打折）**

- 次日 IC：from-scratch 為價格基線的 IC 0.025\,096\,944\,235\,406\,68 增加 0.011\,838\,097\,802\,667\,544（t=3.021\,008\,615\,997\,043），與 chained、ChronoBERT+對比微調「大致相當」。**但注意 Sharpe**：次日多空 Sharpe 在基線為 1.730\,573\,778\,735\,763\,6，加入 encoder 向量後變動不大；盤後 session 那一欄在表 V 中 ChronoBERT+對比微調的 Sharpe 為 1.82，基線為 0.73。**這是本頁標記的第一個打折點**——盤後那欄只有 23.441\,860\,465\,116\,278 % 的交易日能湊到 ≥20 通電話進場，樣本薄且是「開盤到收盤」的日內收益，與日頻波段的宣稱有落差。
- 成本：作者給的是**損益兩平單邊成本**，次日 13.563\,686\,037\,766\,818 basis points、加 from-scratch 向量 11.871\,886\,479\,626\,934 basis points、盤後 23.391\,317\,630\,906\,04 basis points。三者都是「稅前 Sharpe 對應的 breakeven」，不是淨收益；且兩腿每個交易日全換手。**這是第二個打折點：可交易性未被證明，只被標了 breakeven 價格。**
- 基本面目標：earnings surprise 上，基線 IC 0.253\,834\,682\,640\,205\,64，單獨用 from-scratch 文本 0.101\,092\,154\,516\,601\,14，**合起來反而只有 0.241\,096\,885\,899\,600\,63**——低於基線。作者照實寫了，然後改成「文本不加基線」單獨擬合。這是一個明確的**共線／過擬合訊號**，不該被「nearest-neighbor IC 0.047\,874\,162\,876\,594\,725（t=3.254\,923\,433\,165\,685）」的漂亮數字蓋過。
- 前視溢價（表 VI，本頁最核心的一張）：2024 vintage 的 within-call recall@1 溢價，from-scratch 為 -0.002\,809\,523\,809\,523\,81（區間含零），chained 為 0.008\,976\,190\,476\,190\,476，ChronoBERT+對比微調為 0.006\,547\,619\,047\,619\,048；報酬類目標上「六欄有四欄近似零或負」。**判讀：retrieval 端的前視泄漏是真的、且配方依賴；報酬端量不到。**

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位到章節）

| # | 作者限制 | 出處 |
|---|---|---|
| 1 | 單一語料；encoder 最大 149.014\,272 M 參數 | §VIII |
| 2 | 訓練與測試配對**需要電話會的問答輪次**，新聞、文件、券商報告多數沒有此結構 | §VIII |
| 3 | ChronoBERT 系配方沿用 ChronoBERT 的 **2024 tokenizer**（作者僅稱其對 2010 與 2024 文本的切分率相近） | §VIII |
| 4 | replay control 在每個 segment 重啟學習率排程；把它推到常數學習率鏈（如 ChronoBERT）是**外推** | §VIII |
| 5 | 文本對預測的增益很小 | §VIII |
| 6 | 溢價是否隨模型規模、是否隨 reader 規模增長，留待未來 | §VIII |
| 7 | Capital IQ 一通多版本，保留的版本可能晚於 vintage cutoff（作者量化了比例：2011–2024 各 cutoff 為 0.317\,939\,602\,631\,275\,6 % 至 1.394\,138\,149\,090\,427\,9 %，2010 cutoff 為 6.684\,182\,275\,334\,067 %） | §III 腳註 2 |
| 8 | 一般檢索器的訓練 cutoff 未知，早期季度可能**有利於**它們（作者據此把配方結論限定在「固定對比微調」的比較上） | §V |

### 6.2 本頁推斷的隱含假設（明標推斷）

- **【推斷】有效監督綁定問答結構**：整套方法預設語料含「問—答」輪次，且答案輪次可被視為「該問題的真解」。新聞、10-K、券商報告無此結構，因而這條 PIT 路線在「文字另類」軸上是窄頻的。推廣前必須先證明替代監督（例如同文件內鄰近段落）不會摧毀 hard negative 的分布。
- **【推斷】單一通話內 negative 的語義**：within-call 任務把「同一通電話其他高管答覆」當負樣本，其中包含「第二位高管也回答同一問題」的情形——作者明說這對所有 retriever 一律算 miss。這使 within-call recall 的上界被結構性壓低，跨模型比較雖公平，但**不能用來宣稱絕對檢索能力**。
- **【推斷】capacity／regime**：全部結果出自一個語料、一個 cutoff 網格、一種市場（美股）。作者已處理倖存者偏差（「keep every firm, whether or not it is still listed」）與市場連動（CRSP link 在 call date 有效），但**未處理 regime**：2020 年後電話會型態（虛擬／AI 摘要）是否改變文本分布，文本未見討論。
- **【推斷】成本假設**：所有 Sharpe 為稅前，breakeven 為單邊 basis points；未見借券、滑價、盤後流動性的建模。盤後那欄只有 23.441\,860\,465\,116\,278 % 的日子可交易，實務上會進一步壓縮容量。
- **【推斷】held-out 的統計力**：2025Q2 上 from-scratch 的 2024 vintage 低於 2023 vintage，作者自陳「95% 區間含零」，並主動把 from-scratch 讀作「no net gain rather than no leak」——這個讓步是對的，但也意味著**from-scratch 家族的負結果本身缺乏統計力去否定泄漏存在**。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| ChronoBERT（He, Lv, Manela, Wu 2026） | 鏈式年編碼器 + 一般文本，**無檢索微調**，用凍結嵌入測組合 IC | 未披露（本頁未核 repo） | 配方 3（對比微調）使用 2010–2024 各 cutoff 的對應 vintage；僅配方 4（DAPT + 對比微調）限 2010/2017/2024 三個 cutoff。本文補上「encoder 端的 PIT + 檢索監督」 |
| DatedGPT（Yan et al. 2026） | 每年**從零**重訓、等預算的解碼器 | 未披露 | 本文配方 1 的直接先例；本文把它搬到 encoder 並加上 replay 對照 |
| PALM（Lee et al. 2026） | **不重訓**：凍結預訓練權重，只對決策日前的文本掛低秩適配 | 未披露 | 與本文結論張力最大：本文在 encoder 上量到「後一個 vintage 檢索更好」（chained +0.008\,976\,190\,476\,190\,476），與「年度重訓不必要」相反 |
| Kelly et al. 2026 | 單次跑 1T 時序過濾 tokens，發佈**月度** checkpoint | 未披露 | 交付粒度不同（月 vs 年）；本文未比較 |
| 一般檢索器 e5-base / bge-* | 無已知 cutoff、general web 預訓練 | 未披露（本頁未核） | 本文在其未訓練的任務上評分，作者承認早期季度可能對其有利 |
| BM25 | 純詞重疊、零訓練 | 未披露 | 參考下界，表 II 列為 n/a |

🎤 **Interview Tip**

- **問**：point-in-time encoder 解決的是什麼別的方法解決不了的問題？
  - **對的答**：ChronoBERT/DatedGPT 只定時了**語言模型權重**，但回測用的向量來自 retriever 的 encoder，encoder 若讀過未來文本，PIT LM 也救不回來（[7][8] 都把 embedding 列為必須版控的元件）。本文量的是這條通道，並用 registry 在查詢時強制 `query_date ≥ cutoff`。
  - **錯的答**：「就是每年重訓一次模型避免 look-ahead」——這只描述了配方，沒說出**為什麼檢索端是獨立通道**，也忽略了本文的核心發現「溢價存在但配方依賴」。
- **問**：既然 2024 vintage 檢索更好，是不是該用它回測？
  - **對的答**：不能。chained 家族的同年溢價 0.004\,119\,047\,619\,047\,619、2024 vintage 溢價 0.008\,976\,190\,476\,190\,476，且 replay 證明這不是訓練量造成的；作者 §VIII 明說「for encoders of this size on call text, the vintage available at the time is the one to use」。
  - **錯的答**：「溢價很小所以無所謂」——chained same-year premium 在 14 季中 11 季為正；13/14 則是 Y+1 vintage 對 replay 的分解結果，兩者不能互換，也不能據此直接宣稱可交易收益。

### 7.1 可證偽預測（帶日期）

1. **到 2027-06-30**：任何沿用「registry + replay control」這套量測、把 encoder 放大到明顯大於 149.014\,272 M 的後續工作，若配方為 chained 且監督仍為電話會問答對，應仍在 within-call recall@1 上報出**正的同年前視溢價**（點估計 > 0，且多數測試季為正）；若配方為固定 token 預算 from-scratch，則應仍報不出淨增益。**若 chained 的溢價在放大後消失，則本文「泄漏取決於配方」的解釋被推翻。**
2. **到 2027-06-30**：把同一套 encoder 向量接到**淨成本**（含滑價與借券）的日頻多空回測上，次日 Sharpe 的增量應小於本文稅前報的幅度；若有人在宣稱淨 Sharpe 大幅提升而無法重現 breakeven 11.871\,886\,479\,626\,934 basis points 的算法，視為不可比。
3. **可立即證偽**：若有人能在**不含問答輪次結構**的語料（純新聞、純 10-K）上，用本文的方法（對比微調 + 逐年 vintage）復現同樣量級的 within-call 對應指標，則 「論文自述限制」第 2 條「監督需要問答結構」的限制被推翻。**目前尚待驗證。**

## §8 · For the Reader

- **因子研究員**：把這篇當「泄漏審計」的清單來讀，不要當 alpha 論文。可立即拿去用的是兩件工具：(a) registry 式 cutoff 強制（查詢日期 → 可用版本的硬失敗）；(b) replay control（同 token 數重讀舊文本）——這是分辨「模型變強」與「模型讀到晚文本」的最低成本設計。本文對報酬 IC 的增量是 0.011\,838\,097\,802\,667\,544 配上 t=3.021\,008\,615\,997\,043，拿去和你的現有文本因子邊際貢獻比，先問「這點增量扣掉 breakeven 11.871\,886\,479\,626\,934 basis points 還剩多少」。
- **高頻執行**：這篇對你幾乎沒有直接可用的訊號——它最漂亮的那欄是**盤後 open-to-close**（表 V 的 1.82 對應 ChronoBERT+對比微調 session after），但那欄只覆蓋 23.441\,860\,465\,116\,278 % 的交易日，且要求 ≥20 通電話同日進場。真正值得注意的是它的**入場時點定義**：15:00 前開始的通話在當日收盤進場，否則次日收盤；通話前的 08:30 或 16:00 後才加盤後 session。這套時點紀律比它的模型更值得抄。
- **組合配置**：主要訊息是負向的——把 encoder 換成「更晚的 vintage」在報酬目標上量不到增益（次日預測 2024-vintage 溢價 from-scratch -0.0023、chained -0.0022），只有在 earnings surprise 上兩個 bootstrap 檢定才勉強站住（t 為 3.574\,794\,594\,188\,621 與 5.427\,796\,540\,630\,767，對 Bonferroni 界 3.469\,806\,555\,161\,574\,7）。**不要把「檢索變好」當成「預測變好」的代理。**
- **LLM-agent**：表 VII 是這篇對 agent 建構者最直接的一節——用晚一年的 retriever 取 3 段餵給 ChronoGPT-Instruct，答案的對數機率升 0.076\,411\,291\,267\,722\,85；但**把真答案從候選裡拿掉後，溢價掉到 0.001\,846\,001\,225\,390\,605、區間含零**。意思是：RAG 系統繼承的是 retriever「把那條答案撈出來」的泄漏，而不是一般性的「知道更多」。你的 as-of 審計要一路審到 retriever 的權重，不是只審 reader 的訓練截止。
- **RL 策略**：本頁未見 RL 相關實驗（未披露）。若要套用，唯一可遷移的是「vintage 化環境」的想法：把檢索器當環境的一部分，環境版本與決策日期必須綁定。
- **研究學生**：這篇是「怎麼把一個工程衛生問題寫成可測量命題」的範本。值得逐字讀的是 §VII 的多重檢定處理（96 個溢價、Bonferroni、moving-block bootstrap、最後只認 2 個）與 §V 對「一般檢索器 cutoff 未知」的主動讓步——兩者都是審稿人會問、而多數論文不寫的東西。**可做的 follow-up**：檢驗「文本+基線 IC 低於基線 IC」（earnings surprise 上 0.101\,092\,154\,516\,601\,14 vs 合併後 0.241\,096\,885\,899\,600\,63，低於基線的 0.253\,834\,682\,640\,205\,64）這個現象；共線、訊號冗餘與估計噪聲只是待驗證解釋，不能由這些 IC 值直接確定因果。

## References

- **原論文**：Alessio Brini, *Point-in-Time Encoders for Earnings Calls*, arXiv:2611.14325v1 [cs.CL], 10 Nov 2026. https://arxiv.org/abs/2611.14325v1
- **代碼／結果**：`https://github.com/Alessiobrini/point-in-time-encoders-for-earnings-calls`（§I 腳註 1；含 code、call identifiers、result files、training records；transcripts 有授權、不再分發）
- **Lineage（本文直接承接）**：
  - He, Lv, Manela, Wu, *Chronologically consistent large language models*, JFE 185 (2026) — ChronoBERT vintages，本文配方 2/3/4 的骨架
  - Yan, Tang, Gao, Jiang, Lu, *DatedGPT*, arXiv:2603.11838 (2026) — 配方 1 的先例
  - Lee et al., *PALM: Point-in-time adaptation for financial language models*, arXiv:2609.30316 (2026) — 反面對照
  - Kelly, Malamud, Schwab, Xu, *Scaling point-in-time language models*, arXiv:2607.11889 (2026)
  - Kong et al., *Evaluating LLMs in finance requires explicit bias consideration*, arXiv:2602.14233 (2026) — 把 retriever 列為第二泄漏源
  - Gururangan et al., *Don't stop pretraining* (DAPT), ACL 2020；van den Oord et al., InfoNCE, arXiv:1807.03748；Gao et al., gradient caching, RepL4NLP 2021
- **來源鏈接**：arXiv abs 頁 https://arxiv.org/abs/2611.14325　·　本次輸入為可讀全文文本（62,915 字元）；**圖片／附件未提取，圖 1–3 的座標軸細節待核**（正文已給出圖 2/3 的定性結論與部分數值）。