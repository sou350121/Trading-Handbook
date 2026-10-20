<!-- ontology-5axis data=文本另类 horizon=中长周期 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# NoLBERT 解構（NoLBERT）

> **來源發布**：2026-10-15 · （無 venue） · arXiv [2610.22745](https://arxiv.org/abs/2610.22745)
> **arXiv 原文**：[Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns](https://arxiv.org/abs/2610.22745v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把「預訓練穿越（look-ahead）」從特徵端擋掉的一次方法論實驗——用 1976–1995 預訓練的 NoLBERT 加上同文段落對比學習，做成 point-in-time sentence encoder，再放進 Bybee et al. (2023) 的 narrative 曝險框架裡，與 LDA / MiniLM / FinBERT 在同一批 1.26M 篇新聞上做嚴格時序比較。它解的不是預測精度，而是「你以為的樣本外，是不是真的樣本外」。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `中长周期` | `监督回归` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 用 1976–1995 語料預訓練、tokenizer 也由該期語料建立的 NoLBERT，以「同一篇文章的兩段話互為正樣本」做無標籤對比適配，得到以歷史語料預訓練、再以截至 2019 年文章適配的 sentence encoder；② 核心 trick 有兩個：對比適配（無 label、無報酬，文章邊界是唯一監督）與 63 日注意力衝擊取代 5 日；③ 對「文本另类 × 中长周期 × 監督回歸」這格★ 的意義是：把 alpha 供應鏈裡最難查的前瞻泄漏源（encoder）換成可稽核的，作者報告適配用 150 分鐘單卡（不是完整研究成本）；④ 關鍵實證：全 reader × 全模型等權組合月度淨 Sharpe 1.12、buffered 1.19，但對價格基準不顯著，而驗證期選出的最佳實驗（overall pick）淨 Sharpe 為 -0.23、-0.27、-1.08，每個頻率都虧錢。

**X-Ray.** 這篇的價值不在 Sharpe，在「協定」。作者關注兩個可能污染歷史研究的環節：encoder 預訓練期與評估期重疊，以及選擇與評估沒有隔離。作者把兩者都拆開給你看代價——代價是：point-in-time 限制本身沒有證明能提升預測（他們自己說「no evidence that the point-in-time restriction costs forecasting power」，也說這不是「improves」的證據），而嚴格的樣本外選擇規則直接讓數百個實驗中最亮眼的那個變成虧錢機器。在本樣本較一致的結果來自量測層：注意力衝擊的視窗（63 日 vs 5 日）比換哪個 reader 更重要，在 46–48 組配對中 IC 提升 1.28 / 0.87 / 0.49，且橫跨測試期兩半與 2024 換新聞源都成立。它也親手示範了「為什麼不要挑單一實驗」：EBM 在驗證期最強，但 98%/58%/79% 的模型權重落在配對項上，而那些配對項多數配到 40%–50% 公司是插補噪聲的 firm-news 特徵；把配對項刪掉、什麼都不重估，Sharpe 就從 -0.23、-0.52、-1.08 回到 0.00、0.23、-0.41。對量化讀者的意義很直白：這是一份「怎麼把文本因子做到可以誠實面對 FDR」的施工紀錄，不是一個可直接上線的 alpha。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作（Prakash et al. 2026 / Foley et al. 2026）

| 面向 | 前作做法（作者自述弱點） | 本文改動 | 換來什麼／代價 |
|---|---|---|---|
| 文本表示 | frozen sentence transformer（MiniLM，2021 發布），讀過樣本期文本 | NoLBERT（1976–1995 預訓練、30,000 word pieces 詞表同期）+ 同文段落對比適配；另設 60 / 20 topics 兩個版本 | 原始預訓練與詞表不含後期文本；本文適配會更新權重，所用文章截至 2019 年；代價是子詞切得更碎（1.49 tokens per word vs MiniLM 1.43），64-token 視窗覆蓋更少；且適配與預訓練期兩個變因綁在一起，無法分離 |
| 樣本紀律 | topic model 與詞表在 full sample 上擬合；股票池用到年初不可得資訊；部分選擇在評估月上做 | topic models / vocabularies / centroids 只擬合到 2019；選擇只在 2020–2021；2022–2024 僅報告；股票池每年 6 月底重排、當年 6 月起用 | 選擇與評估解耦，驗證期很短（24 / 52 / 104 期）成為新瓶頸 |
| 評估與交易 | 單一 Sparse IPCA 定價模型，成本前評估 | 8 個預測器（OLS/Ridge/Lasso/IPCA/Sparse IPCA/XGBoost/RF/EBM）× 9 條組合規則 × 3 個頻率，10 bp 成本後 | 搜尋規模膨脹到每頻率約 600 個實驗，迫使作者引入 DSR / White / Hansen（代價是幾乎沒有單一結果能過關） |

補充的資料層修補（不是模型改動但影響量級）：93 個 ticker 的標題—正文錯配，涉及 2012–2023 文章的 7.3%；以及舊版日報酬檔在 2017 年後只保留當年 6 月名單股票，導致即將出榜的股票缺近月報酬、這批股票之後月均落後 3% 到 5%，樹模型學會找它們、讓「純價格特徵」跑出週頻淨 Sharpe 1.56 的假象。

### 1.2 ⚡ Eureka

> **一句話**：把「同一篇文章的兩段話」當成唯一監督訊號去微調一個 1995 年就停筆的 encoder——不用 label、不用報酬、預訓練與 tokenizer 止於 1976–1995；後續適配則讀截至 2019 年的文章，不能把兩個階段混成「權重從未讀過後期文本」；然後用「63 日相對注意力」而不是「本週新聞」去定義 shock。

直覺兩層：
1. **適配的必要性**：NoLBERT 只有 masked-word 目標，沒有句級目標。直接平均 token 向量時，測試期兩篇無關文章的平均餘弦是 0.53，而標題與它自己的導語是 0.59——差距小到無法聚成 60 個 topic。對比學習把「同文不同段」拉近、把 batch 內其餘 31 篇推遠，這個「推力」本身就是分散機制，所以作者明確說沒有任何 whitening 或後處理。
2. **視窗的必要性**：5 日衝擊記的是「這週的新聞」，63 日衝擊記的是「哪些主題相對過去一季在搶到注意力」。前者是事件噪聲，後者是敘事累積；作者在驗證期用事先寫好的規則挑了 63 日，而且這個優勢在測試期對**每一個** reader、**每一個**頻率都成立。

### 1.3 信息流

```
文章（headline + 前 500 字元正文），日期無時刻 → 第 d 日新聞自 d+1 收盤起可用
        │
        ▼
[Reader] LDA 60 / MiniLM 60 / FinBERT 20 / NoLBERT 60 / NoLBERT-20 / LDA(all articles)
        │  π_{a,k} ≥ 0, Σ_k π_{a,k} = 1
        ▼
日注意力 θ_{k,d} = mean over 當日抽樣文章（每日 ≤150 篇；<20 篇的日子無讀數）
        │
        ▼
衝擊 s^{(w)}_{k,d} = θ_{k,d} − (1/w)Σ_{j=1..w} θ_{k,d−j}，對市場報酬正交化
        │  w ∈ {5, 63}
        ▼
曝險 b_{i,k,t} = 過去 252 日 EW 共變異（半衰期 60，至少 120 個報酬）
        │
        ▼
特徵：8 個價格 + K 曝險 + (1+K) 公司新聞 + placebo 曝險 + FinBERT/LM 語氣 + 64 維標題嵌入 + 6 個 news×price
        │  橫截面排名縮放至 [-1,1]；缺失以同日隨機股票整塊代入
        ▼
8 個模型 → 期內排名等權集成 → 6 readers × 8 models 的 48 模型組合
        │
        ▼
9 條組合規則（decile / buffered / cap-weighted 各版本 / rank-weighted / mean-variance / long-only / tilt），10 bp per dollar traded
```

## §2 · 數學層

📌 **Napkin Formula**

注意力衝擊與曝險（主設計的核心量測）：

```
s^(w)_{k,d} = θ_{k,d} − (1/w) · Σ_{j=1..w} θ_{k,d−j}          （對市場報酬正交化）
b^(w)_{i,k,t} = Cov^EW_{d<t}( r_{i,d} , s^(w)_{k,d} )          （252 日窗、半衰期 60、≥120 個報酬）
```

對比適配（symmetric InfoNCE，B=32、τ=0.05）：

```
L = −(1/2B) Σ_i [ log( exp(a_iᵀb_i/τ) / Σ_j exp(a_iᵀb_j/τ) )
                + log( exp(a_iᵀb_i/τ) / Σ_j exp(a_jᵀb_i/τ) ) ]
```

**直覺**：第一式把「今天這個主題佔了多少版面」減掉「過去 w 天平均佔多少」——留下的是**注意力轉移**，不是注意力水準；正交化是為了不讓市場整體波動冒充敘事衝擊。第二式是雙向 softmax：a_i 要在 32 個 b 裡認出自己的 b_i，b_i 也要在 32 個 a 裡認出自己的 a_i，分母裡的其他 31 個就是「推力」。

**複雜度（本頁量級分析）**：讀者適配與逐篇編碼後，曝險端需對股票、topic 與滾動窗口運算；直接計算的量級可寫為 O(K × 252)。作者未提供完整分項耗時，不能據此判定成本中心。

**兩種溫度不可混用**：§4.4 對比適配的 InfoNCE τ=0.05；§4.2 由 embedding 對 centroid 相似度轉成 topic 權重的 softmax temperature=0.07。MiniLM 使用 60 clusters；NoLBERT 另有 60／20 topics 版本。

**Loss / 訓練細節**：AdamW，learning rate 2×10⁻⁵，weight decay 0.01，linear warm-up 後 cosine decay，gradient clipping at 1，bfloat16（DeBERTa 的相對位置項在半精度會溢位）；單張 RTX 3060、限時 150 分鐘、完成 6,205 steps，約 0.87 個 epoch。**無 head、無 label、無報酬**；pooling 取最後一層實體 token 平均並單位化（不用 first-position，因為 DeBERTa 沒在那個位置受過句級訓練）。

## §2.5 · 帶數字走一遍（Worked Example）

以下全為**假設／示意**，不是論文實證結果，只演示 attention shock 的定義。

1. 假設某主題當日注意力為 0.50，過去 5 日均值為 0.40、過去 63 日均值為 0.20。
2. 則短窗 shock 為 0.10、長窗 shock 為 0.30。同一日觀測在不同基準下，偏離量不同；這本身不證明哪個窗口預測更好。
3. 實作還需在各估計窗對市場報酬正交化，並以歷史報酬與 shock 的 EW **共變異**作曝險；不是除以 shock 方差的迴歸 beta。
4. 依原設計只在驗證集選窗口，測試集報告效果。示意數字不進 §5 benchmark，也不充當來源數值。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 新聞來源 | 文章清單 FinMultiTime（Nasdaq.com 各 ticker 頁面），正文 FNSPID；兩者皆 public on Hugging Face |
| 規模 | 1,255,289 篇文章，2012–2024；保留正文 ≥400 字元、合併重複、剔除 16 December 2023 的異常大量寫入 |
| 讀取方式 | headline + 正文前 500 字元；所有模型讀同一份日抽樣（每日 ≤150 篇 → 529,657 篇），擬合上限每日 60 篇、截至 2019（140,209 篇） |
| 頻率 / 時點 | 日頻新聞；日期無時刻，第 d 日新聞自 d+1 收盤啟用；每週工作日文章中位數由 2012 的 83 篇增至 2023 的 789 篇 |
| 已知文本缺陷 | 93 個 ticker 的標題—正文錯配，影響 2012–2023 的 7.3%；2024 無 FNSPID 覆蓋，改用 FinMultiTime 完整段落、否則僅標題 |
| 報酬與股票池 | CRSP daily/monthly + Fama–French；每年 6 月底取市值前 1,000 大普通股（share codes 10、11，NYSE/AMEX/Nasdaq），2010–2024 每年重排，持有全部 2,062 檔曾入榜股票；每期約 984 檔，面板 141,761 stock-months / 306,746 stock-fortnights / 614,525 stock-weeks |
| 切分 | 訓練 2013–2019（84 個月 / 182 雙週 / 365 週）、驗證 2020–2021（24 / 52 / 104）、測試 2022–2024（36 / 78 / 156）；持有期歸於其起始日所屬切分 |
| 覆蓋率假設 | 過去 126 交易日無文章視為未覆蓋：訓練 / 驗證 / 測試 stock-months 分別為 50%、39%、40%；缺失特徵以同日隨機股票整塊代入（價格 0.3%–1.2%、narrative 曝險 <1%、公司新聞與語氣 39%–50%） |
| 容量假設 | 僅 1,000 大股票；10 bp per dollar traded 固定成本，無借券成本、無價格衝擊；每實驗單一種子 |

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 原文 §8 稱「Code, derived topic outputs, forecasts and the scripts behind every table and figure are in the project repository [link]」——**URL 未披露**（連結在可讀文本中為佔位符） |
| Checkpoint | 預訓練 NoLBERT 引自 Kakhbod and Li (2025), arXiv:2509.01110；本文**適配後的 encoder 權重是否釋出：所讀文本未見** |
| License | 論文本身 CC BY-NC-SA 4.0；**代碼與 checkpoint 授權未披露** |
| 複現難度 | 中高：encoder 適配在作者的單次設定以消費級 GPU 執行 150 分鐘（不保證其他環境相同），但完整管線為 564 experiments per frequency × 3 頻率 × 9 組合規則，且需 CRSP |
| 數據可得性 | 新聞（FNSPID / FinMultiTime）公開；CRSP 經 WRDS 授權、不可轉散布，repo 提供重建 universe 與報酬檔的腳本；Fama–French 因子公開 |

## §5 · 評測 / Benchmark

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| US 1,000 大股票，月度，測試 2022–2024 | Test rank IC（×100），price+text | Price ensemble 3.3 | All readers, 63 d 3.9 | +0.6 |
| 同上 | 淨 Sharpe，decile long-short | Price ensemble 0.66 | All readers, 63 d 1.12 | +0.46 |
| 同上 | 淨 Sharpe，buffered long-short | Price ensemble 0.53 | All readers, 63 d 1.19 | +0.66 |
| 同上 | 淨 Sharpe，decile（對照驗證期選出的最佳實驗） | Overall pick -0.23 | All readers, 63 d 1.12 | +1.35 |
| 同上 | Test rank IC（×100），63 d（reader 對 reader） | LDA 2.85 | NoLBERT 2.59 | -0.26 |
| US 1,000 大股票，雙週，測試 2022–2024 | 淨 Sharpe，decile | Price ensemble 0.10 | All readers, 63 d 0.69 | +0.59 |

**Δ 的解讀——哪部分是真 capability、哪部分是別的東西**

- **Δ IC 與 Δ Sharpe 的來源不同**。3.9 vs 3.3 的 IC 差距，主要不是「NoLBERT 比較會讀財經新聞」；作者自己在 reader 層面給的證據是反方向的：單看 reader，LDA 的 63 d IC（2.85 / 2.01 / 0.94）在月度與雙週都高於 NoLBERT（2.59 / 1.34 / 1.01），而所有 reader 之間的成對差異在 text-only 集合上都不顯著。組合的增益來自「把 6 個 reader × 8 個模型用期內排名等權平均」這件事本身（Table 4：ensemble vs its members，月度 6 對配對中 Δ SR 由 0.22 到 0.55，配對方向一致（不新增換算百分比））。
- **成本是真實扣過的，但只扣了一種成本**。週頻最誠實：price-and-text 組合 gross Sharpe 1.02、net 0.30（buffered 0.39）——gross 與 net 的差距明顯，兩者為風險調整指標，不宜直接解讀為成本占收入比例。月度 buffered 的 12.4% 年化淨報酬、10.4% 波動、8% 最大回撤，是在 10 bp 固定、無借券成本、無價格衝擊的假設下算出來的；本文沒有可配置 AUM 或完整容量曲線，不能由股票池規模認定成本模型可實盤成立。
- **統計顯著性很薄**。月度 price+text 對零顯著（t=2.36, p=0.045；buffered t=2.42, p=0.040），但對價格基準不顯著（p=0.463 / 0.267），DSR 只有 0.04 與 0.06。作者自述每頻率約 600 個實驗，無技能下的最佳 Sharpe 預期會落在 2.2 到 2.7；沒有任何實驗通過 10% FDR 門檻。這意味著「1.12」應該被讀成「這個協定下能看到的量級」，不是「策略的 Sharpe」。
- **overall pick 那 -0.23 / -0.27 / -1.08 是全篇最有訊息量的數字**。它展示本實驗中的選擇失效，不能證明所有短驗證期選擇都必然過擬合：EBM 在驗證期 IC 是 2.4 / 2.0 / 1.7（各頻率最高的 base model），測試期卻接近零。這是模型選擇的失效，不是 EBM 的失效——刪掉配對項後 Sharpe 回到 0.00 / 0.23 / -0.41，證明問題出在 98% / 58% / 79% 權重落在配對項、而配對項多接到 40%–50% 為插補噪聲的 firm-news 特徵。
- **種子噪聲已達取樣誤差量級**。九個實驗各跑十個種子，樹模型在特徵多時 Sharpe 可從 -1.00 動到 1.47（幅度 2.5），與三年報酬的取樣誤差同級——這是本樣本需要評估種子不確定性的理由，未證明任意等權組合在其他設定都更好。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（僅列可定位於所讀文本者）

- **統計功效不足**（§6 Limitations）：測試期 36 個月 / 78 個雙週 / 156 週，年化 Sharpe 的標準誤約 0.58，只有真實 Sharpe 約 1.7 以上才有 80% 檢定力；reader 之間、組合與價格基準之間的差異都不顯著。
- **搜尋規模**（§6 Limitations）：每頻率約 600 個實驗 × 9 條組合規則；無實驗通過 FDR 門檻，DSR 很小；跨 reader 的組合是在主實驗跑完後才形成的，其 p 值未計入該項選擇。
- **文本模型的時間純度不一致**（§6 Limitations）：NoLBERT 只適配到 2019 年底，但 MiniLM 與 FinBERT topic classifier 是用樣本期內文本訓練的；2019 年後不重擬合，各 reader 的 coherence 在後續文本上下降 29% 到 37%。
- **新聞源非 point-in-time**（§3.1、§6 Limitations）：新聞是樣本期末才從 ticker 頁面蒐集，缺樣本期內被併購的公司、覆蓋多的公司只留最近文章；公司層新聞對 40%–50% 股票是插補；2024 文章來自第二個收集，ticker 更少且無標題。
- **成本與宇宙假設**（§6 Limitations）：10 bp 固定、無借券成本、無價格衝擊；宇宙限 1,000 大股票；每實驗單一種子。
- **適配與預訓練期無法分離**（§4.4）：NoLBERT 是「舊預訓練 + 本文適配」，MiniLM 是「新預訓練 + 原樣使用」，兩者比較混了兩個變因；作者指出需一個同配方適配的 MiniLM 才能分離。
- **驗證期優勢幅度小、視窗只試了四個**（§6）：63 日的驗證期優勢本身不大，且只搜了 w ∈ {1, 5, 21, 63}，測試期增益幅度應與多重檢定結果一併解讀。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **Regime 依賴（推斷）**：5 日衝擊的 IC 由 2022–2023 的「略正」轉為 2024 的 -2.7 / -1.6 / -1.2，而 63 日幾乎不變（1.7 與 1.8 月度）；價格特徵在 2024 更強。這暗示「新聞敘事 alpha」的可用性隨年份漂移，而作者承認兩半之間的差異都不顯著——也就是說，這個 regime 判斷本身也在低檢定力之下。適用條件：結論只在 2022–2024 的 1,000 大美股、日內新聞無時刻的設定下被檢驗過。
- **容量與成本（推斷）**：在本文固定成本模型下，頻率與組合規則對淨結果有明顯影響（週頻 gross 1.02 → net 0.30），作者配對比較報告 buffer 改善淨報酬 0.4 / 1.1 / 1.4 個百分點（依頻率）；換手欄位與單位應直接核對原表，不在此重算比例。真實成本、借券費與價格衝擊變動後，需要重算每條規則的淨結果；不能推定只有 buffered 能存活；本頁不對具體 breakeven 成本做推論，原文未給。
- **數據泄漏（部分為作者自述、部分推斷）**：作者已在 §4.6 明說「缺失值可以依賴未來」——無新聞的公司更可能是後來被併購的，所以他們用同日隨機股票整塊代入而非用均值或丟棄。這是作者採用的處理方式；但 firm-news 區塊的 40%–50% 插補值在統計上是「同日他股的噪聲」，偏重這些插補特徵可能增加不穩定性；作者的 EBM 分解提供了本樣本的例子，不能推成所有模型的必然結果。
- **Survivorship 的雙面性（推斷）**：股票池用每年 6 月底市值重排、回填曾入榜全部 2,062 檔的日報酬，以修正舊報酬檔缺口；這是資料集構造，不等於實際組合持有所有股票；但新聞源缺被併購公司，等於在**特徵**端引入了與退出事件相關的缺失，不能據此認定報酬端已排除所有倖存者問題；兩個資料環節需分別核查，需分別評估。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| LDA（60 topics，Gibbs 抽樣，10,000 unigrams+bigrams） | 詞袋、無語序；本文最強的傳統基線，63 d 下 IC 2.85/2.01/0.94，coherence 最高（NPMI 0.58）；但穩定性最低（58% 文章在換種子重擬合後保留 topic） | 方法公開 | 仍是最難打敗的基線；作者結論是 NoLBERT「forecasts about as well as LDA」 |
| MiniLM（paraphrase-MiniLM-L3-v2 + spherical k-means） | 前作的 reader；2021 發布＝讀過樣本期文本；NPMI 0.49、與 NoLBERT 的分割只有部分一致（AMI 0.45） | 模型公開 | 被本文當作「有穿越風險」的對照組 |
| FinBERT（20 類主題分類器，只用標題） | 監督式固定標籤集，無法擴充 topic；classifier 在 2022 文本上微調（落在測試期內） | 模型公開 | 標籤集由外部決定，是與「無 label 適配」相對的另一條路 |
| NoLBERT 原始版（未適配） | 只有 masked-word 目標，在本文未適配基線下區分力較弱（無關文章平均餘弦 0.53 vs 標題-導語 0.59） | 見 Kakhbod and Li (2025) | 本文的適配是把它變成可用 reader 的必要步驟 |
| 前作 Prakash et al. (2026) / Foley et al. (2026) | 同一 narrative 框架，但 full-sample 擬合、年初不可得資訊、評估月選參、成本前評估 | 見原文引用 | 本文即為其「協定修正版」 |
| 舊版報酬檔＋樹模型（負面對照） | 2017 年起缺即將出榜股票報酬，週頻淨 Sharpe 1.56 全來自此缺口 | 資料缺陷 | 修好後消失；本頁視為「假 alpha 的教科書案例」 |

🎤 **Interview Tip**

- **問：NoLBERT 這篇的主要貢獻是更強的文本因子嗎？**
  - **正確答**：不是。它證明的是「point-in-time 限制在此設定下是可負擔的」——適配 150 分鐘、無 label，topic 最穩定（69% / 80% 對 LDA 58%），預測力與 LDA 相當而非更好；作者明說「It is not evidence that it improves forecasts」。本樣本較一致的結果來自量測層：63 日衝擊優於 5 日，對每個 reader、每個頻率都成立。
  - **錯答**：說它用 1976–1995 預訓練所以「消除了前視偏差所以 Sharpe 更高」——本文沒有任何數字支持「更高」，且月度組合對價格基準不顯著。
- **問：驗證期最好的模型為什麼不能拿去交易？**
  - **正確答**：overall pick 是各頻率驗證 IC 最高的 base-model 實驗，測試期淨 Sharpe 是 -0.23 / -0.27 / -1.08；另對 price-and-LDA-text EBM 的拆解顯示 98% / 58% / 79% 的權重落在配對項，而那些配對項接到 40%–50% 為插補噪聲的 firm-news 特徵；刪掉配對項、不重估任何參數，Sharpe 回到 0.00 / 0.23 / -0.41。overall pick 與這組 EBM 只在月度／週頻相同，雙週不同；這是本樣本的選擇與特徵不穩定性證據，不能判定模型族必然失效。
  - **錯答**：說「EBM 不可解釋所以不穩」或給出任意的失效閾值／成本門檻——原文沒有給這種閾值。

### 7.1 待驗證研究假說（本頁提出；覆核日期 2027-06-30）

- 以相同文章、相同適配配方比較 NoLBERT 與 MiniLM，分離預訓練時段和適配的效果；不預設誰的 IC 較高，也不預設差異顯著性。
- 在新樣本凍結窗口選擇規則、成本與股票池後，重比短窗與長窗。若排序改變，應把「窗口效果較一致」限制於本文樣本。
- 以多個種子重跑選模與集成，同時報完整分布及選擇調整後結果。本文樹模型的 Sharpe 範圍不能直接當作另一模型或 RL 環境的噪聲下界。

## §8 · For the Reader

- **因子研究員**：分別記錄預訓練、tokenizer、適配與 topic 擬合的截止日。NoLBERT 的歷史預訓練不代表適配後權重仍只讀舊文本；曝險按 §4.5 用共變異，不能自行換成回歸 beta。
- **執行研究員**：先重算自己的成交成本、借券與衝擊模型。本文週頻 gross Sharpe 1.02、net 0.30 是固定 10 bp 設定下的結果，不是實盤成本上限。
- **組合配置**：把月度 Sharpe 1.12／1.19 和價格基準 0.66／0.53 一起報，保留相對基準不顯著與搜尋調整限制。不同組合規則的差異不是新增 alpha 的因果證明。
- **LLM-agent 研究者**：固定 embedding、聚類與同一套評分有助於使表示比較可檢查；生成式標註若加入流程，仍需重新驗證時間邊界與 prompt 敏感度。
- **RL／研究學生**：本文研究表示與監督式報酬預測，沒有驗證 RL 策略。可借鑑資料與選擇隔離，但不可把本文 Sharpe 或種子變異直接外推為另一環境的性能或噪聲。

## References

- 原論文： *Reading the News Without Hindsight: NoLBERT Topics and the Cross-Section of Stock Returns*, arXiv:2610.22745v1 [q-fin.ST], 15 Oct 2026. https://arxiv.org/abs/2610.22745v1
- 方法 lineage（依原文引用）：
  - NoLBERT：Kakhbod and Li (2025), arXiv:2509.01110 — point-in-time encoder 本體。
  - SimCSE 對比目標：Gao et al. (2021), EMNLP — 本文適配的 loss 來源。
  - Narrative asset pricing 框架：Bybee, Kelly, Su (2023), *Review of Financial Studies* 36(12) — 注意力→衝擊→曝險的原始設計。
  - 前作對照：Prakash et al. (2026), SSRN 7534559；Foley et al. (2026), arXiv:2609.34169。
  - 機器學習報酬預測：Gu, Kelly, Xiu (2020), *RFS* 33(5)；IPCA：Kelly, Pruitt, Su (2019), *JFE* 134(3)。
  - 可解釋模型：Bell et al. (2026b), *JFE* 181（EBM）；Lou et al. (2013), KDD。
  - 多重檢定：White (2000)；Hansen (2005)；Bailey and López de Prado (2014)；Harvey, Liu, Zhu (2016)。
  - 資料：Dong, Fan, Peng (2024), FNSPID, arXiv:2402.06698；Xu et al. (2025), FinMultiTime, arXiv:2506.05019；CRSP via WRDS；Kenneth R. French Data Library。
- 來源鏈接：https://arxiv.org/abs/2610.22745 （全文可讀文本；圖片／附件未核，圖 1–10 與表 A1–A8 的數值以正文敘述為準）