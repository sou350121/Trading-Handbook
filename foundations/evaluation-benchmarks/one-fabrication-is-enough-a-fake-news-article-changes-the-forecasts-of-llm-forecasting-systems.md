<!-- ontology-5axis data=文本另类 horizon=跨周期 paradigm=生成式大模型 alpha=端到端表征 autonomy=全自动黑盒 -->

# One Fabrication Is Enoug 解構（One Fabrication Is Enoug）

> **來源發布**：2026-11-11 · （無 venue） · arXiv [2611.15102](https://arxiv.org/abs/2611.15102)
> **arXiv 原文**：[One Fabrication Is Enough: A Fake News Article Changes the Forecasts of LLM Forecasting Systems](https://arxiv.org/abs/2611.15102v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「文本另類 × 跨周期 × 生成式大模型 × 端到端表征 × 全自動黑盒」——它不造新 alpha，而是替「檢索增強＝免費午餐」這個假設定價：把攻擊者降到「只發一篇文、不碰模型與 retriever」，仍打穿兩個已部署多代理預測系統的架構（作者以 frozen archive 重新實作）。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `跨周期` | `生成式大模型` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把最弱攻擊者（只知道問題與方向、只能發文）放到五個前沿模型與 AIA Forecaster／Grok ForecastBench 兩個部署架構上實測。②核心 trick 不是優化 passage，而是「進入 context」：一旦文章在 context 內，反向預測翻轉 74% to 82%，但真正的瓶頸是 exposure。③對「端到端表征＋全自動黑盒」軸★：表徵層把檢索文本整個吞下，模型對偽造文與近半數真新聞一視同仁，唯一記錄懷疑的 supervisor 卻沒有規則可執行。④關鍵實證：部署系統自检索命中 88.3／90.0；stock own retrieval 端到端反轉 33%（AIA＋flash）；curated 全體 direction accuracy clean 47.9、attacked 52.2，文章方向錯時 clean 50.4、attacked 14.9。

**X-Ray.** 這篇的價值不在「LLM 會被騙」這個常識，而在把攻擊拆成兩段可分別量測的乘積：exposure × follow。舊工程的坑是——retrieval poisoning 文獻預設攻擊者能碰 retriever／embedding、能對 query 優化、能灌大量文件，於是防禦（partitioned retrieval、passage filtering）都設計成對付「大量被優化過的 passage」；本文證明攻擊者只需一篇「問題導向」的普通新聞稿，因為 deployed 系統的多代理廣搜本來就是 exposure 放大器（10 個 agent 各自查詢＝把單 query 漏掉的都補上）。它也順手打掉一個量化圈常用的評核習慣：accuracy 對這種攻擊是零敏度的（攻擊者在事實發生前寫稿，對錯各半，兩半互相抵消）。打不開的 envelope 很清楚：全部是 2026 年的五個模型、九個窗口、frozen archive，沒有 live web、沒有交易成本、沒有 PnL，指標是方向與絕對誤差而非 Sharpe。對量化讀者：把「新聞」從 alpha 來源重估為特徵污染面；事件驅動／新聞情緒因子的 IC 在敵對文本下不可辨識，而你現有的回測監控（rolling IC、hit rate）正好是本文證明無效的那類聚合指標。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（RAG poisoning / guardrail 文獻） | 本文 |
|---|---|---|
| 攻擊者知識 | 需 retriever／embedding／目標 query 優化，或注入多文件（PoisonedRAG 一線） | 只知道問題與想要的方向；不碰模型、retriever、query，無回饋 |
| 攻擊動作 | 優化 passage、注入指令 | 發一篇可讀新聞稿（市場題允許引用虛構來源與數字，但不得陳述 resolution） |
| 評測標的 | 小開源模型 forecaster、QA benchmark | 五個前沿模型 + 兩個部署系統架構（AIA Forecaster、Grok ForecastBench）＋ ForecastBench／Polymarket／Kalshi |
| 機制歸因 | 多為 end-to-end 成功率 | genuine-article 對照 ＋ 逐 passage 驗證測試 ＋ rationale 審計（三件套） |
| 防禦評估 | 防禦是否擋住 | 擋住 ＋ **無攻擊日**的 clean decision 變動 ＋ 真新聞丟棄率 ＋ 呼叫成本 |
| 第二攻擊 | — | destabilising attack（兩篇對立），首測 |

### 1.2 ⚡ Eureka

**「不要攻擊模型，攻擊 context 的入場券。」** 直覺：模型對 knowledge cutoff 之後的事件沒有任何驗證器，所以「偽造」與「真新聞」在表徵上同構——成功率不是 belief 的函數，而是 exposure 的函數。推論也很反直覺：救援手段（多代理廣搜、集成、median）降低的是「每單位 exposure 的跟隨率」，但同時把 exposure 從 16.7 拉到 88.3／90.0，於是端到端幾乎沒救。

### 1.3 信息流

```
attacker（只發文）
   │  publish fabricated article(s) → 開放新聞檔案（付費 wire / syndicated feed）
   ▼
[frozen archive: 2,867 to 4,521 passages / window, 220-word chunks]
   │  query（系統自己寫，攻擊者看不到）
   ▼
retriever ──► exposure（瓶頸）──► top-5 passages ──► LLM forecaster ──► ŷ
   │                │                                      │
   │          Protocol A：強制塞 rank 1（量 follow）      └─► rationale（95% 以偽造事件為理由）
   │          Protocol B：自然檢索（量 exposure）
   ▼
[防禦層] passage filter（judge 每段是否 fabricated）｜partitioned retrieval（hash 切 8 份取 median）
   │
   └─► deployed 系統：10 agents / 8 runs → supervisor（唯一看來源者，記錄懷疑但不作為）
```

## §2 · 數學層

📌 **Napkin Formula**

```
Reversal   = Pr[ sign(y₁)=a | sign(y₀)=−a ]          a = 攻擊者方向
End-to-end ≈ e·R_cond + (1−e)·floor                  （本頁推斷的分解式，原文未給等式）
Harm       = Δ accuracy | attacker wrong              （聚合 accuracy 對它零敏度）
```

**複雜度**：single retrieval ＝ 1 次檢索 ＋ 1 次呼叫；合併防禦 ＝ 8 次檢索 ＋ 40 次 filter 呼叫 ＋ 8 次模型呼叫 ＝ 48 LLM calls／forecast。**直覺**：反轉率是條件量（已讀到才算），exposure 才是可被攻擊者操作的自變量；把兩者相乘才會得到使用者真正承受的量。**Loss／訓練**：無訓練、無梯度、不改權重；ensemble 與 deployed 系統 sampling temperature 0.7，filter 與 judge Temperature 0，judge 用 Gemini 3.7 Flash。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部是示意用的假設數字，不是本文實證結果。** 論文真實數字留在 §5。

1. 假設 100 題，其中 **60 題** clean 預測與攻擊方向相反（其餘 40 題本來就同意，不計）。
2. 假設 exposure `e` ＝ 20% → 被檢索到的反向題 ＝ 60 × 0.20 ＝ **12 題**。
3. 假設「讀到就跟随」`R_cond` ＝ 75% → 由文章造成的反轉 ＝ 12 × 0.75 ＝ **9 題**。
4. 未檢索到的 48 題仍有雜訊地板，假設 floor ＝ 10% → 48 × 0.10 ＝ **4.8 題**。
5. 端到端反轉 ＝ (9 ＋ 4.8) / 60 ≈ **23%**。
6. 現在只改一件事：用四種不同 framing 把 `e` 拉到 60% → 36 × 0.75 ＝ 27，24 × 0.10 ＝ 2.4，端到端 ≈ **49%**。

**讀者照算就懂的地方**：`R_cond` 完全沒動，端到端卻翻倍——這就是原文 framing 槓桿的形狀（真實對照：exposure 25.0 → 59.1，端到端反轉 26.2 → 41.5）。反過來說，任何只壓 `R_cond` 的防禦（更會推理、更多 agent 投票）在 `e` 高的系統上幾乎無效。

## §3 · 數據層

- **兩套題庫**：183 stock questions（1 日／5 日報酬、macro 事件後報酬、兩股價差）＋ 308 prediction-market questions（sports／politics／economic releases，平台原文措辭）。
- **窗口與 archive**：stock 用四個兩週窗口，as-of 27 February、1 May、26 June、31 July 2026；market 用五個窗口，as-of 24 April 至 7 August 2026。每窗口取 as-of 前 14 天真實新聞，切 220-word passages（35-word overlap），共 2,867 to 4,521 passages per window；stock 四窗分別 2,867／2,993／3,251／3,008，market 五窗 4,412／4,147／4,315／4,521／3,724。
- **來源**：ForecastBench 2026 年 4 月輪、Polymarket、Kalshi 公開 API，按事前固定規則篩選（開盤且 45 天內 resolution、成交量門檻 \$50k（sports）／\$30k（politics, economic）／\$5k（Kalshi）、as-of 價格 0.05 and 0.95、每事件一題）。所有題目與攻擊方向跑前凍結。
- **控制組**：fresh-news controls 57（stock）與 90（market）、24 題 off-target（文章不提的公司）。deployed 系統跑 100 題 stock（其中 60 題做自有檢索）與全部 308 題 market。
- **洩漏防線**：dated-phrase gate 移除 as-of 之後陳述結果的 passage；每窗口前用 20 個 2026 事件＋四個 pre-2026 控制題探測，全模型對 2026 項全答 “unknown”。
- **容量假設**：全部在 frozen archive 上跑，無 live web、無真實下單；成本約 113,000 次呼叫，stock 花 \$1,150（deployed 佔 \$700；AIA 每 cell \$0.26 對 single-retrieval \$0.003），market 約 \$600。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 匿名化倉庫；原文腳註：「Code, configurations and result tables will be released on publication.」→ URL TBD |
| Checkpoint | 無自訓權重；五個商用 API 模型（Gemini 3.7 Flash、GPT-5.6、GLM-5.3 Flash、Grok 4.20、Claude Sonnet 5）。檢索堆疊：BGE-M3 dense、BM25、bge-reranker-v2-m3、RRF（常數 60） |
| License | 未披露 |
| 複現難度 | 高：需自建 frozen archive（2,867 to 4,521 passages × nine windows）＋ 113,000 次呼叫的預算 ＋ 兩個 deployed 系統的重新實作（廠商 web search 被替換） |
| 數據可得性 | 偽造文章僅在研究用途協議下提供、不公開；真實 archive 未釋出；prompts、configs、seeds、result tables 釋出；文章由 Cohere Command A（＋Qwen 3.7 Plus 補 11 篇）與 Claude Sonnet 4.5 撰寫，均在評測集之外 |

## §5 · 評測 / Benchmark

> 本文沒有交易回測：Sharpe／IR／MDD／成本後 PnL 皆未披露。下表「前SOTA」欄依模板填的是**未攻擊的 clean 基線**，不是任何 SOTA——本文比較的是同一系統「有無被攻擊」，不是模型對模型的性能排名。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Curated stock，183 q／2,745 cells | direction accuracy %（全體） | 47.9 | 52.2 | +4.3 |
| Curated stock，文章方向錯（87 q） | direction accuracy % | 50.4 | 14.9 |  |
| Curated stock，文章方向對（96 q） | direction accuracy % | 45.6 | 86.0 |  |
| ForecastBench，74 ticker-horizons／1,110 cells | direction accuracy %（全體） | 54.8 | 56.3 | +1.5 |
| ForecastBench，文章方向錯（38） | direction accuracy % | 51.2 | 26.7 |  |
| ForecastBench，文章方向對（36） | direction accuracy % | 58.5 | 87.6 | +29.1 |

（註）原文另於 Table VIII 給出 change 列；本頁不重算這些 change，Δ 列只保留可由端點直接驗算且無混淆者，其餘留空。

**反轉率（無 clean 對照，故不入上表）**：stock 池化 82.2 [77.9, 86.1]，逐模型 GLM-5.3 Flash 75.0、Gemini 3.7 Flash 83.4、GPT-5.6 87.3、Grok 4.20 96.6、Claude Sonnet 5 74.8；market 池化 74.0 [69.1, 78.9]，分域 Sports 51.1／Politics 87.8／Economic 82.6／price-level 88.2。對照：clean 重跑 placebo 16.7、market 未檢索 floor 4.8、真新聞 decision flip 22.4（stock）與 17.4（market）。

**防禦對照（皆為攻擊後狀態，故不設 Δ）**

| 防禦 | 數據集 | Reversal 未防禦 % | Reversal 防禦後 % | Clean decisions changed % |
|---|---|---|---|---|
| Passage filtering | curated | 74.4 | 23.3 | 27.0 |
| Passage filtering | ForecastBench | 67.3 | 17.4 | 14.4 |
| Partitioned retrieval | curated | 82.0 | 14.2 | 21.0 |
| Partitioned retrieval | ForecastBench | 22.0 | 13.9 | 23.3 |

**解讀。** ①**74% to 82% 是條件量，不是攻擊的期望效果**：它假設文章已在 context。同一批 stock 題端到端只剩 33%（AIA＋flash），而單檢索的 33% 大部分來自 floor（exposure 16.7）——兩個 33% 數值相同、組成完全不同，這是全文最容易被誤讀的一格。②**真正可歸因為「攻擊者能力」的是 exposure 槓桿**：framing 從 25.0 拉到 59.1、端到端 26.2 → 41.5，而單篇加量幾乎無效（21.3 → 28.0）。③**能力 vs 選樣偏誤**：文章的 competence gate 要求「screen model 在放入文章後朝攻擊方向移動」，市場題更把此 gate 由攔截改為報告（作者自述因它會拒掉能推動受測系統的文章）——這是對 effect size 的**選擇機制**，會系統性放大 `R_cond`；端到端因受 exposure 上限約束，放大較小，但仍非無偏。④**無前瞻偏差的證據較強**：frozen archive ＋ dated-phrase gate ＋ knowledge probe，且 ForecastBench 用的是作者沒寫的題目（7 日 horizon 反轉 72.3 命中事前設定的複現準則）。⑤**成本未計**：全文無交易成本、無滑點、無可交易性檢驗；direction accuracy 不是 PnL，短 horizon 效果最大（7 日 72.3 vs 30 日 53.5）恰恰也是最貼近下單窗的那一段，但兩者之間沒有任何成本或容量分析可依。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（§VI Limitations，可定位）

- 全部系統跑在**過去 as-of 日的 frozen archive** 上，因此沒有任何模型能檢索到結果；兩個 deployed 系統是在 3,000 to 4,500 passages 的自有 archive 上的**重新實作**，不含廠商 web search 與 ranking，exposure 數字描述的是它們的公開搜尋行為在該 archive 上的表現。
- **未做 live-web 測試**（需要在開放網路植入偽造財經新聞）。
- 文章來自三個 author models 與兩套 article rules（一保守、一擬真）；copies 與 distortions **未對 filter 做優化**，因此測到的是攻擊者能力的一個實例。
- 防禦側：partitioned retrieval 用 passage id hash 切自己 archive，是**最乾淨的情形**；未測三種 deployer 可能加入的控制：source allowlist（作者指出攻擊者用的付費 wire 會被 allowlist 放行）、source-level isolation、自帶檢索的 filter。
- §V-D 的規模實驗**只靠 42 questions**，更多文章下的侵蝕估計區間很寬。
- 證據涵蓋 five models、nine windows、four question types，**全部在 2026 年**；跨窗口與未見過的題目可複現，「尚未跨年份」。
- market 題中 sports 最弱，AIA Forecaster 幾乎不動（Appendix C）。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **regime 依賴（推斷）**：follow rate 綁定在 2026 年的模型世代與新聞生態；換世代、換模型家族可能整體位移（本文同一系統內兩個模型家族差 33–42 points，這說明系統層的防禦設計遠不如「模型剛好不聽話」重要）。不得推論所有 LLM 系統在任何時期都如此。
- **容量與防禦經濟（推斷）**：攻擊邊際成本≈一次發文；防禦邊際成本 48 calls／forecast、丟棄 39% 真新聞、並改變 29% 的 clean decisions。這是一條**防禦方經濟上不可持續**的不對稱，而非「防禦無效」的普遍結論。
- **成本／可交易性缺口（推斷）**：全文沒有把方向準確率映射到持倉或 PnL，也沒有任何交易成本；curated 文章方向錯時 clean 50.4、attacked 14.9（Table VIII），把這段落差直接讀成策略虧損是越界。
- **樣本選擇（非典型 survivorship，但仍有外部效度代價）**：ForecastBench 49 個 ticker 中，一個 archive 太薄、一個 market snapshot 遺失，最後 37 個；屬明確排除並揭露，不是隱性倖存者偏差，但 37 檔的可外推性有限。
- **單一 prompt 的擾動被當成單因子（推斷）**：Protocol A 一次改動同時改變「文章存在」與「rank 位置」，文章本身與位置效應並未完全隔離（destabilising 實驗有做 rank 互換，directional 沒有）；因此 82% 不能整數歸因於「內容」而非「排第一」。
- **失敗≠不可能（推斷）**：未測 >16 篇文章、未測優化型攻擊者、未測 source-provenance 防禦；這些是空白，不是否證。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| PoisonedRAG [9] 及後續 [15–18] | 需對 retriever／embedding 優化或大量注入；本文移除這些前提 | 未披露 | published |
| Indirect prompt injection [10] | 注入的是指令；本文文章不含任何指令 | 未披露 | published |
| Lu & Zhang [14]（concurrent） | 同為 publish-only，但對象是小開源模型 forecaster；本文延伸至前沿模型與部署架構，並加機制歸因與防禦成本 | TBD | concurrent |
| RobustRAG／partitioned retrieval [13] | 防禦「有界數量的 corrupted passage」；本文用 4 framed／16 distorted 打穿 | 未披露 | published |
| Passage filtering（Prompt Shields [27]／NeMo Guardrails [28]） | 產品級 guardrail 篩注入與敏感內容；本文把它拿來篩「整個事件是編的」，並量到 34.2 的真新聞誤殺 | 專有／未披露 | deployed |

🎤 **Interview Tip**
- **正確答**：「74% to 82% 是 conditional on article in context 的上界；端到端只有 33%，而單檢索的 33% 大半是 floor（exposure 16.7、未檢索 floor 20），AIA 的 33% 大半是文章（exposure 88.3）。所以真正的攻擊面變量是 exposure，防禦要壓 exposure／做 provenance，不是偵測文字風格。」
- **錯答**：「82% 的預測都被翻轉，所以 LLM 預測不可信／乾脆別用檢索」——同時犯三個錯：混淆條件量與端到端、忽略 accuracy 幾乎不變代表傷害是「對齊攻擊者」而非「變差」、以及忽略防禦成本這一側（48 calls、29% clean decision 變動）。

### 7.1 可證偽預測（本頁推斷，帶日期）

1. **2027-06-30 前**：若無 provenance-based 防禦進入生產，任何公開的 web-search 預測 agent 在自有 archive 上的 exposure 仍會遠高於單檢索基線（可對照本文 88.3／90.0 vs 16.7）——若出現反例，本頁的「exposure 由架構而非模型決定」判斷即失效。
2. **2027-12-31 前**：沿用同一 protocol 的後續工作，30 日 horizon 的反轉率不會超過 7 日 horizon（本文方向：72.3 → 53.5）；若被反轉，代表「效果隨 horizon 衰減」的機制解釋需修訂。
3. **2027-12-31 前**：作者提議的 corroboration-across-sources 防禦若被實作，會在真新聞丟棄率上付出可測代價（作者明言 unmeasured）；若代價近零，則本文「驗證不可行」的核心論斷被削弱。

## §8 · For the Reader

- **因子研究員**：把新聞／事件特徵的 IC 檢定改成「敵對文本下的穩健性」——你現在用的 hit rate 與 rolling IC 正是本文證明零敏度的那類聚合指標。可先做的低成本實驗：在自有新聞庫注入同一個事件的四種 framing，量 IC 位移而不是量平均報酬。
- **高頻執行**：短 horizon 效果最大（7 日 72.3 vs 30 日 53.5）意味著衝擊落在「下了單就來不及改」的窗；執行層該加的是**來源單一性檢查**（單一 wire、無第二來源）與發布時間戳異常，而非模型端過濾。
- **組合配置**：本文沒有任何 PnL、容量、成本；把它當成「新聞 alpha 的尾部風險備忘」，而不是配置依據。真正的配置含義是：依賴外部檢索的訊號其有效容量不可估（未披露）。
- **LLM-agent builder**：AIA 的 supervisor 讀出懷疑（29.0 的 rationale 質疑事件）卻只在 10 of 300 次高信心時覆寫——這是**監控-控制缺口**的教科書案例。可立即落地的規則：單一來源事件折價、事件需 n 個獨立來源、把 forecast 歸因到具體 passage。
- **RL／策略研究者**：這裡的可學習訊號不在模型而在**揭露策略**——何時該輸出 no call。過濾防禦自己把 no-call share 從 16% 推到 29%，說明「棄權」是一個可訓練的動作，而本文的 destabilising 攻擊（兩篇對立）恰好證明模型不會自發學會它（614 條 rationale 無一對方向避險）。
- **研究學生**：這篇是「如何把一個聳動結論寫成可證偽實驗」的範本——genuine-article 對照、placebo 重跑、floor 定義、事前固定的複現準則、以及把防禦成本放進同一張表。照這個骨架重做任何 RAG 安全性題目，審稿人很難挑。

## References

- 原論文：Zeyu Zhang, Bradly C. Stadie, *One Fabrication Is Enough: A Fake News Article Changes the Forecasts of LLM Forecasting Systems*, arXiv:2611.15102v1 [cs.CR], 11 Nov 2026. https://arxiv.org/abs/2611.15102v1
- Lineage（攻擊）：Zou et al., PoisonedRAG, USENIX Security 2025, arXiv:2402.07867 [9]；Greshake et al., Indirect Prompt Injection, AISec 2023, arXiv:2302.12173 [10]；Lu & Zhang, *The Corroboration Illusion*, 2026 [14]（concurrent）。
- Lineage（預測系統與基準）：Halawi et al., NeurIPS 2024, arXiv:2402.18563 [2]；Alur et al., AIA Forecaster, arXiv:2511.07678 [5]；xAI Grok ForecastBench system [6]；Karger et al., ForecastBench [7]；Polymarket [11]、Kalshi [12]。
- Lineage（防禦）：Xiang et al., RobustRAG, arXiv:2405.15556 [13]；Microsoft Prompt Shields [27]；NVIDIA NeMo Guardrails [28]；Zheng et al., LLM-as-a-judge, arXiv:2306.05685 [31]。
- 來源鏈接：https://arxiv.org/abs/2611.15102