<!-- ontology-5axis data=文本另类 horizon=中长周期 paradigm=生成式大模型 alpha=端到端表征 autonomy=人机协同可解释 -->

# FinBERT + 两级LLM评分链（GPT-5.4-mini摘要→Claude Sonnet 4评级）+ 随机森林/ridge混合排序 解構（FinBERT + 两级LLM评分链（GPT-5.4-mini摘要→Claude Sonnet 4评级）+ 随机森林/ridge混合排序）

> **來源發布**：2026-10-27 · （無 venue，會議欄標 `Under review`） · arXiv [2610.38056](https://arxiv.org/abs/2610.38056)
> **arXiv 原文**：[What Language Models Add to Healthcare-Equity Rankings, and What They Could Have Known](https://arxiv.org/abs/2610.38056v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文本另類 × 中長周期 × 生成式大模型 × 端到端表征 × 人機協同可解釋」，但它解的 prior gap 不是「LLM 能不能賺錢」，而是**同一個排序系統裡，LLM 的增量到底是不是後見偏誤**——用固定文本、換 rater、配 clean-clean 安慰劑把兩者分開。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `中长周期` | `生成式大模型` | `端到端表征` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1；Appendix 各表以純文字串流呈現，欄位對齊有雜訊，數字仍以可讀到的逐字值為準。
**TL;DR:** ① 把 187 欄非市場特徵（balance sheet、attention、analyst、FinBERT/兩級 LLM 文本評分）逐通道放進同一 fold 篩選與 ridge head，做成一張可對帳的 component ledger；② 核心 trick 是「固定文本、換 rater」：同一批 62,492 份 GPT-5.4-mini 日摘要由 Claude Sonnet 4 / GPT-5.4-mini / FinBERT / GPT-2 四把尺評分，看 frontier−clean 的 IC gap 是否在訓練 cutoff 後縮小，再用 FinBERT-vs-GPT-2 的 clean-clean 安慰劑檢驗平行趨勢；③ 對「生成式大模型＝Alpha」這條軸是**負面證據**：9 欄被 |t|≥4 納入，沒有一欄是文本評分，full hybrid 的 12 個月 neutral IC 由 0.063 升到 0.093，來源記為 48%，但增量來自 balance sheet 與 attention；④ 唯一站得住的文本增量是科學文本在 3 個月的 +0.010（t 2.8，bootstrap [+0.006,+0.015]），作者自己標為 exploratory，且其 identity placebo 為 +0.035 對 standalone +0.040。

**X-Ray.** 這篇的價值不在 Alpha，在**鑑別設計**。同軸對手（Glasserman-Lin 的匿名化前後、Sarkar-Vafa 的歷史可得資訊檢定）都是「換輸入」或「換時點」，本文是「同一份文本、換一把尺、加一個乾淨對乾淨的安慰劑」——這才讓「frontier 模型知道結局」變成可證偽命題，而不是修辭。它同時暴露了 LLM-in-ranking 的工程坑：polarity 欄在 18 個 fold–horizon 全部被篩掉（最近的 miss |t| 3.96），而真正有預測力的是**不需要語言模型的 attention count**（記錄數與覆蓋分析師數）。打不開的 envelope 很清楚：單一產業（美國醫療／生命科學）、survivor-only 母體、只報 rank IC、明確聲明不做報酬／成本／可實施性主張；而且文本塔的 12 個月預測力被 within-company shuffle 幾乎完全複製，意思是「它測到的是公司身分不變量，不是本月新聞」。對量化讀者的意義：把這篇當**方法論模板與陰性對照**讀，不要當因子候選讀。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作常見做法 | 本文做法 | 對結果的影響 |
|---|---|---|---|
| ① Component ledger | 端到端報一個「LLM 加持後」的總增量 | 每次只把一個 channel 放進候選池，重跑同一 per-fold 篩選＋ridge head，每列都是該 channel 的 hybrid 對同一 market-only 基底 | 增益可歸因；但作者明說 rows 是 paired gains「do not add」，通道間資訊重疊，fundamental +0.022 與 attention +0.021 不能相加去對 full +0.030 |
| ② 固定文本、換 rater 的 cutoff 檢定 | 比 cutoff 前後的模型表現，或原文 vs 匿名化 | 同一批日摘要由四把尺評分；以 target-end-after-cutoff 定義 post 窗；配 FinBERT-vs-GPT-2 clean-clean 安慰劑 | 1m／3m 的 frontier−clean gap 在 cutoff **前**就最多 +0.004，cutoff 後不縮 → 沒有「後見紅利」可解釋；12m gap 下降但安慰劑同步下降 +0.070（t 2.8），不識別 |
| ③ 身分安慰劑 | 只做 cross-company shuffle | 再加 within-company shuffle（保留公司、打亂時序） | 文本 standalone IC 被安慰劑追平甚至在六格中五格更高 → 技能是持續性公司語言 |

### 1.2 ⚡ Eureka

**「不要問 LLM 準不準，要問同一篇文章給兩個世代打分，gap 會不會在 cutoff 後消失；再用一對都乾淨的尺，看這個 gap 本來會不會自己消失。」** 直覺：後見偏誤的指紋是「cutoff 前有優勢、cutoff 後優勢消失」，但很多東西（訓練集重疊、覆蓋率隨時間變化、產業 regime）也會產生同樣形狀，所以必須有 clean-clean 安慰劑做平行趨勢檢定。本文最狠的一刀是補上第二個安慰劑——**身分安慰劑**——直接說明文本的預測力來自「這家公司是誰」而非「這個月發生了什麼」。

### 1.3 信息流

```
market (10,130,326 records, 14 price-volume feats)
        │
        ├─► contest-selected random forest (walk-forward, train 2010..Y-2) ──► market score ──► market-only ranking
        │
sentiment / fundamental / scientific / macro (187 cols)
        │
        └─► per-fold screen: rank-residualize vs market score + 4 price/liquidity controls
                 │  admit if Newey-West |t| >= 4, 5% coverage, 15 names/month, 24 months
                 ▼
            ridge head (alpha=10, demeaned within month)  ──► hybrid (full model)
                 │
                 └─ regime blend: b_w = w·rank_fund + (1-w)·rank_att, w per state tercile (LOYO)

text tower (522,968 records → 74,575 company-months, frozen encoder → walk-forward ridge head)
        └─► text ranking  ──► additive test vs market-only / hybrid (LOYO weight 0.1)

LLM stages (dashed): GPT-5.4-mini summary -> Claude Sonnet 4 rating -> screened polarity column (never admitted)
                     narrative layer -> deterministic checker + GPT-5 judge
```

## §2 · 數學層

📌 **Napkin Formula**

```
neutral IC  :  e_i = r_i - a_hat - b_hat' c_i     (c_i = price, dollar volume, volatility, Amihud 的百分位)
               IC_neutral = rho_S(e, y)           (y 為未殘差化的實現超額報酬)

blend       :  b_w = w · rank_fund + (1-w) · rank_att ,  w 依 state tercile 在 {0,0.1,...,1} 上選
```

複雜度／成本：每個 fold 561 次檢定（184 個 firm-level 欄 + 3 個 macro conditioning，3 個 horizon），12m 宣稱建立在 65 個重疊月上，bootstrap 為 circular block（block 12、10,000 draws）。

直覺：作者自己承認 `rho_S(e,y)` 因為 Spearman 會對殘差重排名，**不是**標準 partial correlation，「neutral IC」只是這篇論文的操作標籤。這點很關鍵——1m／3m 的 full model raw IC 增量是 −0.000 與 −0.001，所有短天期「增益」都只存在於控制變數殘差化之後；換句話說，短天期的 Δ 有很大一塊是「相對價格／流動性座標的重新排序」，不是報酬預測力的淨增加。

Loss／訓練細節：market block 為 open contest 選出的 random forest（2015–2018 walk-forward，contest 結果本文未重現）；ridge 的 α=10，特徵在月內去均值；文本頭的 ridge penalty 由 {10,100,1000,10000} 中以 five-fold group CV（以月為 group）選取，最大化 out-of-fold 月均 Spearman IC；test year Y 只用 Y−2 以前的年份訓練（Y−1 整年排除，避免 12 個月目標重疊）。

## §2.5 · 帶數字走一遍（Worked Example）

以下**全部是假設／示意的玩具數字**，用來手算「neutral IC 為何可能與 raw IC 反向」，與論文實證無關。

1. **輸入（4 家公司，一個月）**：分數百分位 `r = [0.9, 0.6, 0.4, 0.1]`；唯一控制變數（示意用 dollar volume 百分位）`c = [0.8, 0.5, 0.2, 0.1]`。
2. **月內 OLS**：`mean(r)=0.5`、`mean(c)=0.4`；`Sxy = 0.31`、`Sxx = 0.30` → `β̂ = 1.033`、`α̂ = 0.087`。預測值 `[0.913, 0.603, 0.293, 0.190]`。
3. **殘差** `e = r − pred = [−0.013, −0.003, +0.107, −0.090]`。解讀：分數最高的那家被「高價＋高流動性」解釋掉了，真正超出控制的是第三家。
4. **實現超額報酬排序** `y = [0.05, 0.02, −0.01, −0.03]`。`rank(e) = [1, 3, 4, 2]`、`rank(y) = [4, 3, 2, 1]` → `Σd² = 14`、`rho_S = 1 − 84/60 = −0.4`；而 `rank(r) = [4,3,2,1]` 與 `rank(y)` 完全相同，raw IC `= +1.0`。
5. **結論（示意）**：同一組分數，raw IC 是 +1.0、neutral IC 是 −0.4。這就是為什麼本文所有短天期結論都掛在 neutral 上、而 raw 欄必須並列——也是為什麼不能把 neutral IC 的 Δ 直接讀成「可交易 Alpha」。真實系統用四個控制變數、以百分位排名殘差化，且對候選欄還額外對 market score 殘差化（篩選比指標更嚴）。
6. **身分安慰劑為何會追平（示意）**：若把第三、四家公司的文本向量互換「月份」而不換「公司」，兩家的分數幾乎不動（因為文本講的還是同一家公司的恆常詞彙），排序不變 → IC 不變。論文六格中五格出現這個現象，所以作者判定文本技能主要是 persistent company-level language。

## §3 · 數據層

| 面向 | 內容 |
|---|---|
| 母體 | 4,868 家公司；FactSet 四位 NAICS 代碼屬醫療與生命科學類；January 2010 至 June 2026 |
| 每月橫截面 | 價格 ≥ \$1 且 trailing dollar volume 前段，佔母體 28.5% |
| 預測目標 | 對 XBI 的 log 超額報酬，21 / 63 / 252 sessions（1m / 3m / 12m） |
| 五條 pipeline | market 10,130,326；sentiment 2,097,935（news／social／analyst 三層）；fundamental 33,309（XBRL）；scientific 24,680；macro 4,289 |
| 非市場欄位 | 187（184 個 firm-level + 3 個 macro conditioning），至少一個 fold–horizon 被納入者 9 欄，無一為文本評分 |
| 文本塔 | 522,968 records → 74,575 company-months / 1,136 companies（2015 起），覆蓋 41.5–43.7% 評估列；科學文本 33%；每筆截 6,000 characters、最近 30 天 |
| 樣本外機制 | walk-forward：test year Y 用 2010 至 Y−2 訓練，Y−1 整年排除；text head 同協定 |
| 對齊 | 所有非市場特徵按 public-availability date join 後 carry forward |
| 容量／成本 | 未披露；論文自述「no return, cost or implementation claim is made」 |
| 已知偏誤 | sentiment／fundamental／scientific 三條 pipeline 於 June 2026 針對當時上市名單蒐集 → 所有非市場增益都是 within-survivor 估計，bias 方向作者稱 unknown |

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 無公開 repo（原文：「No public repository accompanies this submission」） | GPT-2 small `gpt2`（124M）；Qwen3-Embedding-0.6B（`databricks-qwen3-embedding-0-6b` endpoint）；FinBERT `ProsusAI/finbert`；`claude-sonnet-4-20250514`；`gpt-5.4-mini-2026-03-17`；GPT-5 作 narrative judge | 未披露 | 高：vendor endpoint 與授權資料不可重發；prompt 版本數未記錄 | 分析碼、逐字 prompt（Appendix B）、文字模型設定（Appendix F）、derived per-company-day scores 可應要求提供（需機構合規審查）；licence 原始資料（價格、FactSet 分類、news／social／analyst feeds、filings、registry extracts）不可轉散佈 |

## §5 · 評測 / Benchmark

| 數據集／市場 | Metric | 前SOTA（基線，逐字） | 本方法 | Δ |
|---|---|---|---|---|
| 全體 ledger，2020–2025（72／72／65 個月） | 12m price/liquidity-neutral rank IC | 0.063（market-only） | 0.093（full hybrid） | +0.030 |
| 同上 | 12m raw rank IC | 0.289（market-only） | 0.298（full hybrid） | +0.009 |
| 同上 | 1m neutral rank IC | 0.020（market-only） | 0.028（full hybrid） | +0.008 |
| 同上 | 3m neutral rank IC | 0.027（market-only） | 0.044（full hybrid） | +0.017 |
| 文本覆蓋列（41.5–43.7%） | 12m neutral rank IC | 0.057（market-only） | 0.110（hybrid） | +0.053 |
| 科學文本列（33%） | 3m neutral rank IC | 0.046（hybrid） | 0.056（hybrid + scientific-text, current model） | +0.010 |
| 12m regime blend（65 個月） | neutral rank IC | 0.084（static LOYO blend） | 0.097（state = trailing 12m XBI return） | +0.013 |
| 同上，換比較對象 | neutral rank IC | 0.093（full hybrid） | 0.097（regime blend） | +0.004 |
| 提示式預測（300 家公司，15,627 rows） | 12m neutral rank IC | 0.096（market score） | 0.183（hybrid） | +0.087 |

**哪些 Δ 是真 capability、哪些不是**

- **是真 capability 的部分**：12m 的 +0.030 neutral 由 balance sheet（+0.022，t 2.8）與 attention counts（+0.021，t 3.3）承擔，且 full model 的 block-bootstrap 區間 [+0.014,+0.046] 不含零。但這是**rank association**，不是報酬；作者在 §7 明說 IC 只是排序關聯。
- **不能算 LLM capability**：text polarity 欄（frontier chain 與 GPT-2 兩代）在 18 個 fold–horizon 全部零，最近的 miss |t| 3.96；analyst rating 欄同樣全零（最近 miss |t| 3.67）。作者用 post hoc magnitude bound（如 frontier chain 0.042／0.059／0.065）說明「不能排除實務上有意義的效果」，但同時聲明 bound 是欄位量級上界、不是模型增益上界。
- **科學文本的 +0.010 要打折看**：standalone 0.040（t 5.4）對 identity placebo 0.035，margin 沒有區間；blend 進 hybrid 的 +0.010（t 2.8、[+0.006,+0.015]）在 unadjusted 下顯著，但作者自標 exploratory，且 event-month 子群（3,170 rows，16%）的 +0.064 對比子群 hybrid 基線 −0.023 是**事後**子群比較（L6c 非預先指定）。
- **可能是控制變數殘差化的產物**：1m／3m raw IC 增量為 −0.000 與 −0.001，短天期全部增益只在 neutral 上存在。
- **可能是後見／覆蓋偏誤**：12m gap 在 frontier rater 上確實下降，但 clean-clean 安慰劑下降更多，平行趨勢假設不成立；作者結論是「不識別污染」，不是「證明乾淨」。且 GPT-5.4-mini 摘要的 31 August 2025 cutoff 統治所有摘要列，所謂 clean rater 只在 rater 端乾淨。
- **樣本相對性**：0.063→0.093 是 ledger 全樣本；文本覆蓋列 hybrid 是 0.110、科學文本列 12m hybrid 是 0.105。作者用 Table 8 逐一列基線，明說任何增量不得跨樣本對讀——這是本頁把每列都標樣本的原因。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§7（survivorship）**：所有非市場欄位只存在於 June 2026 上市的公司，因此每個增益都是 within-survivor 估計，bias 方向未知；attention 的 recency count 尤其可能反映 retrieval-time 覆蓋。移除 recency 欄後，1m／3m／12m 增益變為 0、+0.009、+0.022。
- **§7（重疊窗與檢定力）**：12m 宣稱建立在 65 個重疊月；post-cutoff 窗只有 9 至 15 個月、落在薄弱的樣本末段，所以 leakage 估計區間很寬；12m 的對照在兩把乾淨尺本身分歧時不識別污染；要有檢定力需要目標窗**全落**在 cutoff 後，得等 2026 與 2027 資料。
- **§5（state 結果）**：state 分析在設定上 exploratory、在檢定上 semi-out-of-sample（tercile 用全部 65 個月切點，兩個 state 是看到年度結果後才選）。
- **§7（clean rater 強度）**：GPT-2 rater 是 124M 參數、用標題長度的 weak label 微調；更強的 pre-2020 rater 可能改變 GPT-2 列的結論。
- **§7（narrative）**：只用 deterministic rules 檢核，沒有人類讀者、沒有 usefulness 評估；prescience test 已指定但未執行。
- **Appendix H（多重檢定）**：每個校正只在自己family內（screen 3.92、lenses 2.99），**沒有任何校正橫跨整體搜尋清單**；headline ledger 的 t 值、state permutation 的 p 值、科學文本增量都未為此調整。
- **§3／Table 7（provenance）**：Qwen3-Embedding 的 cutoff「Not disclosed」；FinBERT 的 checkpoint 是否即 Araci (2019) 描述的權重「not independently verifiable」。
- **Appendix D（切分慣例）**：post 窗的報酬有部分發生在 cutoff 之前；12m 只有兩個 formation 完全落在 Claude cutoff 之後、GPT-5.4-mini 則為零。

### 6.2 推斷的隱含假設（本頁推斷，非作者陳述）

- **可交易性假設**：月度換手、以月末價格成交、可做空、無借券費與市場衝擊成本。論文只給 rank IC 並明確不做成本主張，所以任何 Δ 都不能換算成 PnL；「12m 比 1m 好」也可能只是長天期 IC 對控制變數更不敏感。**限制適用於本文全部 Δ**，不可外推成「LLM 文本在實盤無效」的普遍斷言。
- **線性 head 假設**：ridge 對凍結 embedding 是線性的；作者明說 per-fold fine-tuning 未做，所以「文本沒增量」的結論只在 frozen representation + ridge + 此篩選規則下成立。
- **身分安慰劑的鑑別力假設**：within-company shuffle 保留了「公司是誰」與所有持久文本特徵；作者自承沒有 identity-only baseline，因此無法把 identity 與 persistent textual feature 分離——這是設計層的殘餘混淆，不是可忽略的細節。
- **regime 假設**：state tercile 用全樣本切點、權重 LOYO 但屬回顧式；六個年度裡每種 state 只有少數離散 episode（Table 14 列出的連續區段），permutation 的 p 值解到約 10^-4 但重排的 episode 數很少，作者自述「addressed overlap, not the small count of episodes」。
- **摘要器污染假設**：所有以日摘要為輸入的比較共用 31 August 2025 cutoff；「rater-clean」不等於「pipeline-clean」。只有 raw-record chain 是端到端乾淨。
- **XBI 基準假設**：超額報酬全以 XBI 定義，產業 beta 與規模暴露已被吸收進「超額」，但 neutral IC 只控制四個價格／流動性變數，不控制基本面風格。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Glasserman & Lin (2023) | 原文 vs 匿名化標題、模型訓練窗內外 | 所讀文本未見 | 本文以「固定文本、換 rater + clean-clean 安慰劑」取代匿名化，額外要求平行趨勢 |
| Sarkar & Vafa (2024) | 檢定歷史預測日當下不可得的資訊 | 所讀文本未見 | 本文 1m／3m 的 gap 在 cutoff 前就存在且不縮；12m 安慰劑同步下滑 → 不識別 |
| Lopez-Lira & Tang (2023)／Chen, Kelly & Xiu (2023) | LLM 情緒或表徵 → 報酬預測力 | 所讀文本未見 | 本文：frontier polarity 與 GPT-2 評分欄在 18 個 fold–horizon 全不被 |t|≥4 納入 |
| Li et al. (2026) | LLM 擇時策略在更長更寬回測中優勢縮小 | 所讀文本未見 | 作者稱最接近的先例；差別在本文是橫截面排序，不是擇時 |
| Ke, Kelly & Xiu (2019) | 文本 supervised PCA 預測報酬 | 所讀文本未見 | 本文改用 frozen encoder + walk-forward ridge head，並加 identity placebo |
| Gu, Kelly & Xiu (2020)／Harvey et al. (2016) | 橫截面 ML 與多重檢定 | 所讀文本未見 | 本文的 screen（561 tests、Bonferroni 3.92）就是這條線的工程化，但作者承認校正只 local to family |

🎤 **Interview Tip**

- **正確答**：「這篇的主結果是審計性的陰性結論：通過 |t|≥4 篩選的 9 欄裡沒有一欄是文本評分；full hybrid 的 12m neutral IC 從 0.063 到 0.093 是 balance sheet 與 attention 給的，文本只留下科學文本在 3 個月的 +0.010，而且作者的 identity placebo（+0.035）幾乎追平 standalone（+0.040）。真正可搬走的是 identification design：同一文本、換 rater、clean-clean 安慰劑、target-end-after-cutoff 切分。」
- **錯答**：「FinBERT + GPT-5.4-mini 摘要 + Claude 評級的兩級鏈打敗市場模型，12 個月 IC 從 0.063 升到 0.093 就是 LLM 的功勞。」——0.093 是全 hybrid，文本評分一欄都沒進模型；而且這是 rank IC，不是報酬，作者明確不做成本與可實施性主張。

### 7.1 可證偽預測（帶日期）

1. **2027 年內**：當 2026–2027 的 12 個月目標窗全落在 GPT-5.4-mini 與 Claude 的 cutoff 之後，若 frontier−clean gap 仍與 clean-clean 安慰劑同向下跌，則「無後見」的讀法被削弱；若 gap 平穩而安慰劑也平穩，H2 的陰性結論被支持。
2. **以未使用資料重跑科學文本 3 個月**：預測 paired 區間含零、identity placebo 追平——作者已把 identity-only baseline 與 matched text-versus-structured 比較列為決定性測試。
3. **擴大到含下市的母體**：預測 attention recency 的 12m 增益不再顯著（作者稱 bias 方向未知，故只預測不顯著、不預測符號）。
4. **延長樣本**：預測 frontier polarity 欄仍不被 |t|≥4 納入；若被納入，則「篩選門檻太嚴」而非「LLM 無效」的解釋需被優先檢驗。

## §8 · For the Reader

- **因子研究員**：把 component ledger 的做法抄走——每個 channel 單獨進候選池、跑同一 fold 規則、報 paired gain 且聲明不可相加。文本因子務必加 within-company shuffle；若安慰劑追平，你的因子講的是公司身分，不是事件。
- **高頻執行**：這篇對你幾乎沒有可用訊號：全部結論在 21 至 252 sessions，且 1m 的 raw IC 增量是 −0.000。可借用的是 target-end-after-cutoff 的切窗慣例，用來檢查你自己新聞 alpha 的 cutoff 泄漏。
- **組合配置**：把 XBI 相對與 neutral IC 的關係看清楚——neutral 只在控制價格／流動性後有意義，而 short leg 與借券成本不在論文範圍。regime blend 對 full hybrid 的 +0.004（t 0.8）不足以支持配置層改動。
- **LLM-agent 開發者**：narrative layer 的機械檢核（6,252 個數字全 traced、1,599 個 null 全守住、0 次 forecast lexicon 命中、79 篇因 “regulatory” 一詞卡在同一個 category 規則）是可直接複用的 audit harness；但 GPT-5 judge 的 κ 0.24 說明「用 LLM 審 LLM」會系統性漏檢（79 漏 65）。
- **研究學生**：這是很好的 negative-result 寫作範本——把 H1 到 H8 與時間戳列出來，明說 H5 是看到模式後才形式化、H8 與六個 lens 是「weak sense 的預註冊」，並承認沒有任何校正橫跨整體搜尋清單。

## References

- Haotong Fu, Xiyue Zhang, Jack Rose, Ding Sun, Erich Scheller, Eric Rose. *What Language Models Add to Healthcare-Equity Rankings, and What They Could Have Known.* arXiv:2610.38056v1 [q-fin.CP], 27 Oct 2026. https://arxiv.org/abs/2610.38056v1
- Gu, Kelly & Xiu (2020), *Empirical Asset Pricing via Machine Learning*, RFS 33(5) — 橫截面 ML 與 walk-forward 協定血統。
- Harvey, Liu & Zhu (2016), *…and the Cross-Section of Expected Returns*, RFS 29(1) — 多重檢定門檻；Jensen, Kelly & Pedersen (2023), *Is There a Replication Crisis in Finance?*, JF 78(5)。
- Tetlock (2007), *Giving Content to Investor Sentiment*, JF 62(3)；Ke, Kelly & Xiu (2019), *Predicting Returns with Text Data*, NBER WP 26186 — 文本 → 報酬的 lineage。
- Glasserman & Lin (2023), arXiv:2309.17322；Sarkar & Vafa (2024), SSRN 4754678；Ludwig, Mullainathan & Rambachan (2025), NBER WP 33344 — look-ahead 辨識問題的三條前線。
- Lopez-Lira & Tang (2023), arXiv:2304.07619；Chen, Kelly & Xiu (2023), SSRN 4416687；Li, Kim, Cucuringu & Ma (2026), KDD '26, doi:10.1145/3770854.3785702 — LLM 訊號與其在長回測中縮水的先例。
- Araci (2019), *FinBERT*, arXiv:1908.10063；Malo et al. (2014), JASIST 65(4)；Maia et al. (2018), WWW'18 Companion；Radford et al. (2019), GPT-2；Zhang et al. (2025), *Qwen3 Embedding*, arXiv:2506.05176 — 模型與標籤血統。
- 來源連結：arXiv 摘要頁 https://arxiv.org/abs/2610.38056v1 （本次輸入為完整可讀全文文本；圖片、Appendix 圖表排版與附件未核）。