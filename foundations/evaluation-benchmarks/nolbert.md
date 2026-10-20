<!-- ontology-5axis data=文本另类 horizon=中长周期 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# NoLBERT 解構（NoLBERT）

> **來源發布**：2026-10-15 · （無 venue） · arXiv [2610.22745](https://arxiv.org/abs/2610.22745)
> **arXiv 原文**：[Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns](https://arxiv.org/abs/2610.22745v1)　·　_本頁由 arXiv 原文一手自主解構_
> **核心定位**：五軸落點＝（文本另类 / 中长周期 / 监督回归 / 因子挖掘 / 人机协同可解释）。它處理的 prior gap 是「表示端的前瞻偏差」（pretraining look-ahead），並非準確率：把 encoder 的預訓練語料鎖在 1976–1995，並用同文段落對比學習把它改造成句編碼器。**但真正的貢獻落在反面**——它同時證明「衝擊怎麼量」比「誰來讀」更重要，且單一最佳驗證實驗會直接虧錢。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `中长周期` | `监督回归` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（摘要＋部分正文）。§3 之後的數據構造細節、§4 全節、§5 的 IC 數值皆在截斷範圍外，標 TBD。
**TL;DR:** ①用 1976–1995 預訓練的 NoLBERT，經同文段落對比微調（無標籤、無報酬）成句編碼器，聚類成新聞主題，與 LDA / MiniLM / FinBERT 在同一 1.26 million 篇（2012–2024）語料上對照。②核心 trick 是「切斷預訓練穿越」，而非換更強模型；附帶發現是**把注意力衝擊的基準從 5 日改成 63 日**，對每一個 reader 都提升 test rank IC。③對 `文本另类 × 因子挖掘` 這條軸★的意義：alpha 的瓶頸在量測定義而非表示能力——NoLBERT 只「about as well as LDA」。④關鍵實證：全 reader × 全模型等權組合月頻淨 Sharpe **1.12**（decile）/ **1.19**（buffered），對零顯著但**贏不過 price-only benchmark**，且無單一實驗通過 multiple-testing 校正；驗證分數最好的那個實驗**在每一個頻率都虧錢**。

**X-Ray.** 這篇的價值不在 SOTA，在**外科手術式地把一個被忽略的污染源切開**：過去文本因子的回測把 look-ahead 藏在特徵裡，樣本外切分再乾淨也救不回來——因為 encoder 在 2021 年讀過 2018 年之後的世界。NoLBERT 把預訓練語料鎖死在 1976–1995，是唯一在「表示端」而非「標籤端」做 point-in-time 的作法，這在五軸上是把 `alpha=因子挖掘` 的前提條件往前推了一步。第二個貢獻更務實：它示範了**研究流程本身的 Pareto 前沿**——同樣六個 reader、八個模型、三種頻率、九種組合規則，最後發現「衝擊基準窗」的邊際效應大於「換 encoder」的邊際效應。這也是它最掃興的地方：NoLBERT 只做到與 LDA 打平，組合 Sharpe 對零顯著但對 price-only 不顯著，且搜尋規模（hundreds of experiments per frequency）本身就把顯著性吃掉了。對量化讀者的意義是負向的但重要：**你以為的文本 alpha，很可能只是量測窗與多重比較的產物**。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作（Prakash et al., 2026 / Foley et al., 2026）

| 維度 | 前作（作者自述的弱點） | 本文改動 | 打到的坑 |
|---|---|---|---|
| 表示端 look-ahead | 直接沿用 frozen sentence transformer；encoder 訓練語料與樣本重疊 | NoLBERT 僅以 1976–1995 文本預訓練，再以**同文段落對比微調**（無標籤、無報酬）改造成句編碼器 | 預訓練穿越（hindsight sits in the features） |
| 全樣本擬合 | topic model 與 vocabulary 在**全樣本**上擬合 | 六個 reader 全部只用**up to 2019** 的文本擬合 | 主題/詞表層面的 look-ahead |
| 評估協議 | universe 用了年初不可得的資訊；部分選擇在評估月上做；組合未計交易成本 | 模型擬合 2013–2019 / 選擇 2020–2021 / **2022–2024 held out**；九種組合規則、net of costs | 選擇偏差 + 成本未計 |
| 衝擊定義 | 5 日窗 | **63 交易日**均值為基準 | 注意力水準的慢變成分未被剝離 |

### 1.2 ⚡ Eureka

> **「衝擊的基準比衝擊的讀者重要」——把 attention shock 從『相對上週』改成『相對過去 63 個交易日』，六個 reader、三個頻率、每一個都提升 test rank IC。**

直覺：新聞注意力有很強的**主題層面慢變趨勢**（某題材連續數月升溫）。用 5 日窗做基準，等於把「這週比上週多講了一點」當訊號，其中大部分是趨勢的日內噪音；用 63 日窗，才把「相對這一季的常態，今天異常地被談論」隔離出來。這是**去均值而非去趨勢**的粗略版本，成本近乎零，卻是全文唯一對所有 reader 一致有效的改動。

### 1.3 信息流

```
1.26M articles (2012–2024)
        │
        ├─[NoLBERT 1976–1995 pretrain]─┐
        │                              ├─ 同文段落對比微調（無標籤/無報酬）
        ├─[MiniLM]──[FinBERT]──[LDA]───┘        │
        │                                        ▼
        └──────────────► 六個 reader 各自輸出 topic assignment（擬合僅用 up to 2019）
                                 │
                                 ▼
                    daily topic attention  a_{k,t}
                                 │
                    shock  s_{k,t} = a_{k,t} − mean(a_{k, t−1..t−63})
                                 │
                    stock exposure  β_{i,k,t} = cov(r_i, s_k)/var(s_k)
                                 │
        + firm-level news chars + price chars
                                 │
                    八個模型（OLS / IPCA / 樹集成 / EBM …）
                                 │
              九種組合規則 × {monthly, biweekly, weekly}，net of costs
                                 │
                    2022–2024 held-out 報告
```

## §2 · 數學層

📌 **Napkin Formula**

```
s_{k,t} = a_{k,t} − (1/63) · Σ_{j=1}^{63} a_{k,t−j}          # 注意力衝擊（63 取代 5）
β_{i,k,t} = Cov(r_{i,·}, s_{k,·}) / Var(s_{k,·})             # 股票對衝擊的曝險
r_{i,t+1} = f_θ( β_{i,·,t} , news_{i,t} , price_{i,t} )      # 八模型，監督回歸
L_contrastive = −log [ exp(sim(z_i, z_i')/τ) / Σ_j exp(sim(z_i, z_j)/τ) ]   # 同文段落為正對
```

**複雜度**：embedding 端每篇 O(D·d)（D 段、d 維）；主題與曝險端每期 O(K·N·63)（K 主題、N 股票、63 日窗）——**把 5 換成 63 的邊際成本幾乎可忽略**，這是它划算的原因。對比學習的 batch 內負對為 O(B²)。

**直覺**：三行分別對應「訊號怎麼定義」「曝險怎麼估」「報酬怎麼預測」。關鍵在第二行——β 是**對 shock 而非對 attention 水準**的回歸，所以 63 日窗的選擇會直接改變特徵的橫截面排序，而不只是 smoothen。

**Loss / 訓練細節**：文本端**完全 unsupervised**（同文不同段落為正對，batch 內其他文本為負對，Gao et al., 2021 式），刻意不讓報酬進入表示學習——因為比較對象是「表示」。報酬端才是監督回歸：八個模型在 2013–2019 擬合，2020–2021 只用來做**選擇**，2022–2024 只報告。超參、τ、聚類主題數 K：**未披露**。

## §2.5 · 帶數字走一遍（示意）

> ⚠️ 以下數字全部為**假設/示意**，用來展示機制流向，**不是論文實證結果**。論文真實數字見 §5。

1. **假設**某主題 k 在某日的 attention 份額 `a_{k,t} = 0.80`，過去 63 個交易日（不含當日）均值 `mean = 0.50` → shock `s = +0.30`。
2. 假設該主題 shock 的歷史標準差 `σ = 0.15` → 標準化 shock `z = 2.0`。
3. 假設股票 i 對該 shock 的曝險 `β_{i,k} = 0.4` → 該股在此因子上的訊號 `0.4 × 0.30 = 0.12`（示意單位）。
4. **對照 5 日窗**：若過去 5 日均值為 `0.75`（主題正在升溫），則 shock 僅 `+0.05`，`z ≈ 0.33`。**同一則新聞、同一個 reader，訊號強度差一個量級**——這就是「63 日勝過 5 日」的機制來源：5 日窗把趨勢誤當基準，自我抵消。
5. 八個模型在 2013–2019 上把 `{β, news chars, price chars}` 映射到相對報酬預測；等權組合跨 reader × 模型後才進九種組合規則，扣成本。

**這套流程的破口在第 3 步**：β 用「過去」估，但若該主題的 shock 序列本身有前視污染（例如新聞來源是事後才收集的），β 會被高估——原文 §5 自述的資料問題正是這一類。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 語料規模 | **1.26 million** 篇財經新聞 |
| 語料時段 | **2012 to 2024** |
| 市場 / 標的 | 美國，**1,000 largest US stocks** 的日報酬 |
| 頻率 | 日頻特徵 → **monthly / biweekly / weekly** 三個再平衡頻率 |
| 擬合 / 驗證 / 測試 | **2013–2019** / **2020–2021** / **2022–2024 held out** |
| Reader 擬合窗 | 六個 reader 均只用 **up to 2019** 的文本擬合 |
| 資料來源與抓取方式 | TBD（截斷） |
| Universe 的 point-in-time 構造 | TBD——原文僅稱採 stricter protocol，具體規則未在可見段落披露 |
| 樣本外與容量假設 | 容量／周轉／成本參數 **未披露**；僅知「net of costs」與九種組合規則 |

**原文自述的資料缺陷（可直接引用）**：article bodies attached to the wrong headline（正文錯掛標題）、a news source collected **after the fact**（事後回補的新聞源）、missing returns that depend on the future（缺失報酬與未來相關）。三者在方法上都會製造**假可預測性**，且都不是換 encoder 能修的。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD（未披露） | TBD（NoLBERT 為 Kakhbod and Li, 2025 之外部模型，權重釋出情況未核） | 原文 **CC BY-NC-SA 4.0**（論文授權，非代碼授權） | TBD | 新聞語料 TBD；美股報酬常見來源可替代，但**六 reader 的同文段落對比微調腳本未披露** |

> 判定：本頁可見範圍內**沒有 repo 與 checkpoint 資訊**。可複現的部分是「63 日 shock + 八模型 + 九規則」的**協議**，不是 NoLBERT 的微調本身。

## §5 · 評測 / Benchmark

| 數據集 / 市場 | Metric | 前SOTA（逐字基線） | 本方法 | Δ |
|---|---|---|---|---|
| 2012–2024，1.26M 篇，1,000 largest US stocks，test 2022–2024 | test rank IC | LDA **未披露** | NoLBERT **未披露**（原文：forecasts about as well as LDA） | 未披露 |
| 同上 | test rank IC | MiniLM **未披露** | NoLBERT **未披露** | 未披露 |
| 同上 | test rank IC | FinBERT **未披露** | NoLBERT **未披露** | 未披露 |
| 同上 | 主題穩定性 / 均勻度（量化值） | LDA / MiniLM / FinBERT **未披露** | NoLBERT「most stable and most even」，數值 **未披露** | 未披露 |
| 同上，monthly，decile 組合，net | Sharpe | price-only benchmark **未披露** | **1.12** | 未披露 |
| 同上，monthly，buffered 組合，net | Sharpe | price-only benchmark **未披露** | **1.19** | 未披露 |
| 同上 | 對零的顯著性 | 0（虛無） | **1.12 / 1.19**，significant against zero **before** adjusting for the search | 未披露 |
| 同上 | 校正後顯著性（deflated Sharpe / FDR） | — | **no single experiment survives a false-discovery correction**（數值未披露） | 未披露 |
| 全部頻率 | 淨報酬 | — | **驗證分數最佳之單一實驗：loses money at every frequency**（數值未披露） | 未披露 |

**哪部分是真 capability**：「63 日窗對**每一個 reader、每一個頻率**都提升 test rank IC」是本文最硬的結論形狀——單一 reader 提升可能是雜訊，**跨 reader 一致提升**更像機制（去均值窗改變的是特徵定義，不是模型容量）。但幅度未披露，因此**無法判斷是經濟顯著還是統計顯著**。

**哪部分可能是搜尋膨脹**：1.12 / 1.19 對零顯著、對 price-only 不顯著、且無單一實驗通過 FDR——三者合起來的判讀是：**這個 Sharpe 大體上買的是 price 因子，文本項的增量未被識別出來**。「驗證最佳者全頻率虧錢」則是對 `2020–2021 選模型` 這一整套協議的反例，比任何 Δ 都值得記。

**成本**：原文只說 net of costs，費率、周轉、借券、滑價**全部未披露**，因此 1.12 / 1.19 不可與其他論文的 gross Sharpe 直接比。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

原文第 6 節為 limitations，但**本頁所見正文截斷於第 3 節，作者自述限制未核**。已可引用的自述缺陷集中於資料層（見 §3）：錯掛標題、事後回補的新聞源、與未來相關的缺失報酬。另自述：test 無法排除 reader 之間的中等幅度差異（the test cannot rule out moderate differences between readers）。

### 6.2 推斷的隱含假設（**推斷，非作者自述**）

| 假設 | 內容 | 失效條件 |
|---|---|---|
| **Point-in-time 只做了一半**（推斷） | 只有 NoLBERT 的預訓練窗被鎖死；MiniLM / FinBERT 的預訓練語料截止日**未披露**，若晚於樣本段，則比較基準本身仍帶 hindsight，會**低估** NoLBERT 的相對優勢或使差距不可解釋 | reader 間比較在此條件下不對稱 |
| **Regime 依賴**（推斷） | 2012–2024 是單一宏觀體制；2020–2021 驗證窗恰為疫情後高波動期，用來「選實驗」會偏向在該體制有效的模型 | 換體制後選擇結果不遷移 |
| **容量**（推斷） | 1,000 largest US stocks、月/雙週/週頻；未披露周轉與 AUM 假設，容量曲線未知 | 高頻版本（weekly）對成本更敏感，net 後可能翻負 |
| **成本**（推斷） | 「net of costs」但無費率；九種組合規則中 buffered 版（1.19）高於 decile 版（1.12），暗示**換手抑制本身有貢獻**，而非純 alpha | 若真實成本高於隱含值，buffered 的優勢會被吃掉 |
| **資料泄漏**（作者已自述，非推斷） | 錯掛標題 + 事後回補新聞源 + 未來相關的缺失報酬 | 任何一項都足以製造假 IC |
| **Survivorship** | 本頁可見文本**未提及** universe 的存續處理；不可假定存在或不存在 | TBD |
| **表示 vs 監督的邊界**（推斷） | 刻意讓報酬不進入文本模型，是為了「比較表示」——這是**實驗設計的純度**，不是投資設計的最優解 | 若目標是最大化 Sharpe，這個約束是自縛 |

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| **LDA**（Bybee et al., 2023 血統） | 詞袋、無語序；本文中最強的非 point-in-time reader，NoLBERT 僅與之打平 | 學術實作常見（**未驗證**） | 基線 |
| **MiniLM**（Wang et al., 2020） | 小型蒸餾句編碼器，前作所用；預訓練截止日**未披露** | 權重公開（**未驗證**） | 基線 |
| **FinBERT**（Huang et al., 2023） | 金融域 BERT 主題分類器；域內適配 vs 時點純淨的交換 | 權重公開（**未驗證**） | 基線 |
| **NoLBERT**（Kakhbod and Li, 2025） | 唯一 pre-1996 預訓練；本文加同文對比微調 + 聚類，**不做生成式標註** | 未披露 | 本文方法 |
| **LLM 生成式標註**（Fedyk et al., 2026 所述風險） | 標籤依賴 prompt、帶系統性偏誤；本文明確以此為由改用「聚類固定 embedding」 | — | 被排除的替代路線 |

🎤 **Interview Tip**

- **問：NoLBERT 相對 LDA 的優勢是什麼？**
  - **正確答**：不是預測力——原文是「about as well as LDA」。優勢有二：(a) 它是**唯一把預訓練語料鎖在 1976–1995** 的 reader，表示端沒有 hindsight，這讓「樣本外」在特徵層面也成立；(b) 主題**更穩定、更均勻**（量化值未披露）。真正的績效槓桿不在換 reader，在**把 shock 基準從 5 日改成 63 日**，這對每一個 reader 都有效。
  - **錯答**：「NoLBERT 打敗 LDA 所以 Sharpe 更高」——原文沒有這個結論；或「Sharpe 1.12 是顯著 alpha」——原文同時說它贏不過 price-only benchmark、且無單一實驗通過 FDR。

### 7.1 可證偽預測

以 arXiv 2610.22745v1 發布日 **2026-10-15** 為起算點，觀察窗內（後續 follow-up 出現時，具體時點未披露）可檢驗：

1. **若**後續研究把 MiniLM / FinBERT 的預訓練語料也嚴格截到 2019 之前，**則** reader 之間的 IC 差距會進一步縮小（因為不公平的 hindsight 優勢被移除）——若差距反而擴大，本文的「point-in-time 不吃預測力」結論被推翻。
2. **若**把 63 日窗換成更長窗（如 126 日或 252 日），**則**依本文機制（去均值 vs 去趨勢），IC 應先升後降；若單調上升，說明「63」不是機制點而只是搜尋點。
3. **若**同樣協議搬到非美市場或小型股，**則** 1.12 / 1.19 這種「對零顯著、對 price 不顯著」的形態應重現——若重現後變成**對 price 顯著**，代表文本因子在低流動性/低覆蓋處才有增量，本文的美股大型股設定低估了它。

## §8 · For the Reader

- **因子研究員**：把這篇當**協議 checklist** 讀，不是當模型讀。(a) 表示端的 pretraining 截止日要與樣本外起點比對；(b) topic model / vocabulary 不可全樣本擬合；(c) 選擇集（2020–2021）與報告集（2022–2024）必須硬隔離；(d) 一定要報 price-only benchmark。這四條會砍掉你 pipeline 裡大部分假 alpha。
- **高頻執行**：weekly 是三個頻率中最吃成本的一端，而原文成本參數**未披露**。實務上先把 63 日 shock 特徵接到你自己的成本模型上跑，看 net 後是否仍正——這是本文最便宜、最可獨立驗證的部分。
- **組合配置**：1.12（decile）vs 1.19（buffered）的差距提示**換手抑制的貢獻大於 reader 選擇**。等權組合跨 reader × 模型（Rapach et al., 2010 的常識）在本文是唯一存活的東西，但它仍未勝過 price-only——配置上應把文本項視為 **price 因子的正交化候選**，而非獨立 alpha sleeve。
- **LLM-agent 研究者**：本文明確拒絕「叫生成模型標主題」，理由是 prompt 依賴與系統性偏誤（Fedyk et al., 2026），改用「固定 embedding + 聚類」。這是一條**可重現性優先**的路線選擇，值得在 agent pipeline 裡當默認保守選項。
- **研究學生**：把 §5 最後一列讀三遍——「驗證分數最佳的單一實驗，在每個頻率都虧錢」。這是 validation-based model selection 在數百個實驗規模下的實證反面教材，配合 Harvey et al. (2016)、Bailey & López de Prado (2014)、White (2000)、Hansen (2005) 一起讀，比任何方法章節都有用。
- **RL 策略**：不要從這篇拿 reward shaping。文本端是**無監督表示**、報酬端是**監督回歸**，全篇沒有序列決策；能借用的是「特徵定義窗長（5 → 63）」這種低維、可消融的設計，而不是模型架構。

## References

- **原論文**：Prakash et al. (2026). *Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns.* arXiv:2610.22745v1 [q-fin.ST], 2026-10-15. https://arxiv.org/abs/2610.22745v1 ｜授權 CC BY-NC-SA 4.0
- **NoLBERT**：Kakhbod and Li (2025) — 1976–1995 預訓練 encoder（本文表示端基礎）
- **敘事資產定價框架**：Bybee et al. (2023) — LDA 主題 → attention shock → 曝險
- **對比學習**：Gao et al. (2021) — 同文雙視角正對；Reimers and Guterych (2019) — sentence encoder
- **基線**：Blei et al. (2003) LDA；Wang et al. (2020) MiniLM；Huang et al. (2023) FinBERT；Chen et al. (2022) LLM embedding regression
- **預測與推論**：Gu et al. (2020)；Kelly et al. (2019) IPCA；Bell et al. (2026b) EBM；Rapach et al. (2010) forecast combination
- **多重比較**：Harvey et al. (2016)；Bailey and López de Prado (2014) deflated Sharpe；White (2000)；Hansen (2005)
- **Look-ahead 文獻**：Lopez-Lira et al. (2025)；Sarkar and Vafa (2024)；He et al. (2025)
- **前作**：Prakash et al. (2026)；Foley et al. (2026)
- **主題品質**：Röder et al. (2015) NPMI coherence
- 來源連結：https://arxiv.org/abs/2610.22745