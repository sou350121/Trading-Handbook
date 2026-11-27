<!-- ontology-5axis data=图关系 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=人机协同可解释 -->

# LENS 解構（LENS）

> **來源發布**：2026-11-03 · CIKM26 · arXiv [2611.04217](https://arxiv.org/abs/2611.04217)
> **arXiv 原文**：[LENS: Reliability-Aware Stock Prediction via LLM-Extracted Priors and News-Driven Semantic Graphs](https://arxiv.org/abs/2611.04217v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「圖關係 × 日頻波段 × 監督回歸 × 端到端表征 × 人機協同可解釋」的交點；它解的不是「沒有關係證據」，而是原文點名的 prior gap——既有方法把每條抽取出來的邊當成 **equally trustworthy binary fact**，沒有機制表達不同關係來源在股票、時間、抽取方法上的異質可靠性。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `图关系` | `日频波段` | `监督回归` | `端到端表征` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文可讀全文（有原文則以原文為準）。圖片（Figure 1/2/3 之具體曲線）、附件與代碼未在本次輸入，細節待升 v1。
**TL;DR:** ①把 LLM 公司關係抽取當成隨機過程跑 M=5 次，用 ρ(e)=f(e)·c̄(e) 把「出現頻率 × 自報信心」壓成連續邊權，靜態圖固定不隨測試期更新；②動態側用新聞標題 embedding + 指數衰減窗口（τ=10, α=0.5）+ mutual top-k₁ 稀疏建每日語義圖，最後用 per-stock、per-day 的 gated fusion 在兩圖之間調配；③對「圖關係」這條軸★的意義是：可靠性不再在圖構建階段被二值化掉，而是以連續特徵流進 static attention 的 key/value 與融合門控；④S&P 500 上 AR@5 0.5232 / SR@5 1.8694，對最強基線 COGRASP 的 0.4493 / 1.5249（原文摘要表述為提升 16.4% 與 22.6%）。

**X-Ray.** 放回五軸 Pareto：LENS 不是在架構上開新戰場，而是在「圖關係」這條軸上補了一個被同行默認忽略的度量維度——邊的可信度。它解的舊工程坑很具體：LLM 抽取的 run-to-run 不穩定（前作 Zhou et al. 2025 直接取單次輸出當確定邊集）、新聞覆蓋長尾造成的孤立節點（共現圖與單日相似度圖的通病）、以及 additive／static gating 這類固定融合規則無法隨股票與日期調整兩個圖的相對可信度。對量化讀者的真正價值不在 52.32% 這個年化數字，而在方法論：把「資料品質」寫進可微參數（ρ 進 attention logit 與 value），而不是寫進前處理腳本。但它打不開的 envelope 同樣清楚：兩個宇宙都是美股大型股、測試窗自 Oct–Dec 2023 至 Oct–Dec 2024 逐季前滾、共五折、成本只計 5 bps 單邊換手、無實盤；人工驗證只有 120 條三元組、單一抽取器（o3-mini）；而「可靠性」這件事在論文中是用 LLM 自報 confidence 代理，該 confidence 的校準本身未被獨立檢驗。換句話說，它把 LLM 的不確定性做成了特徵，但沒有回答「這個不確定性估計是否可信」——這是它與真正 reliability-aware 系統之間仍差的一步。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作做法 | LENS 做法 | 解掉的坑 |
|---|---|---|---|
| 靜態邊語義 | 單次 zero-shot LLM 抽取，輸出即確定邊集（Zhou et al. 2025）；或 GICS／WikiData 這類穩定但粗糙的 taxonomy | M=5 次獨立抽取，ρ(e)=f(e)·c̄(e) 連續加權；另設 Strict／Highrecall／All 三種保留策略（θ_f=0.6、θ_c=0.8） | 抽取隨機性與幻覺不再靠硬閾值一刀切，而是按比例衰減其在 message passing 中的影響 |
| 動態邊來源 | 歷史價格相關（GERU 等，原文指其「largely redundant with the price-derived node features」）；或新聞共現（COGRASP） | 新聞標題 → Voyage-3-large embedding → 指數衰減窗口聚合 → cosine 相似度 → mutual top-k₁=20 稀疏，孤立節點回退 top-k₂=5 | 單日圖對噪聲敏感、長窗圖被過時資訊稀釋、長尾股票在新聞圖上變成孤立點 |
| 兩圖融合 | 原文 Related Work 明批的 fixed fusion rules（additive 或 static gating） | per-stock、per-day gated fusion：λ 為各分支 intrinsic logit（MLP_S/MLP_D），δ 為 concat 後的 context correction（MLP_ctx），softmax 後加權 | 兩圖的相對可信度隨股票與日期變動，單一純量權重無法表達 |
| 可靠性注入點 | 無（邊只有存在／不存在） | ρ 同時注入 static attention 的 **key**（決定誰被注意）與 **value**（決定聚合什麼），Eq. 7–10 | 可靠性若只留在圖構建階段，進入網絡後就被抹平；key-only／value-only 消融均較差 |

### 1.2 ⚡ Eureka

**把「這條邊有多可信」從圖構建的副產品，升格為貫穿 attention 與 gate 的連續特徵。**

直覺：傳統 pipeline 在「建圖」這一步就把不確定性丟掉（留／不留），之後的 GNN 只能對已經二值化的世界做推理。LENS 反過來——不確定性不丟，壓縮成 ρ∈[0,1]，讓下游用學到的權重去決定要信多少；於是「低頻但集體有用」的邊不必被硬砍（Strict 的 recall 代價），「高頻但語義錯」的邊也不會與真邊等權。

### 1.3 信息流

```
OHLCV + MA/RSI/MACD (T=20) ──► Transformer temporal encoder ──► Z_t (N×d, d=128)
                                                                    │
   LLM (o3-mini) × M=5 runs                                          │
   prompt 限制證據 ≤2023-09-30                                       │
   triples (i,r,j,c) ──► ρ(e)=f(e)·c̄(e) ──► G_S (14 類關係, 固定)    │
                                          │                          │
                                          ▼                          ▼
                              Static Graph Transformer      Dynamic Graph Transformer
                              e_ij = ρ_ij·W_rel[r_ij]       bias = log(1+w_D)
                              注入 key 與 value              (Graphormer 式結構偏置)
                                          │                          ▲
                                          │                          │
   TRNA headlines ──► Voyage-3-large ──► 指數衰減 τ=10,α=0.5 ──► cos sim
                                          │                  ──► mutual top-k₁=20
                                          │                  ──► 孤立點補 top-k₂=5
                                          ▼                          │
                                    H_S ──┐    ┌── H_D ──────────────┘
                                          ▼    ▼
                              Gated Fusion: softmax(λ_S+δ_S, λ_D+δ_D)
                                          │
                                          ▼
                              LayerNorm ──► 2-layer MLP ──► ŷ (1-day log return)
                                          │
                                          ▼
                              MSE + φ·RankNet(ε-band, τ scaling)
```

## §2 · 數學層

📌 **Napkin Formula**

- ρ(e) = f(e) · c̄(e)，其中 f(e)=n(e)/M，c̄(e) 為該邊在出現 run 中的平均自報信心。
- e_ij = ρ_ij · W_rel[r_ij]，同時進 static attention 的 key（Eq. 8）與 value（Eq. 10）。
- h_final = LayerNorm( Σ_{k∈{S,D}} Softmax(λ^k + δ^k) · h^k )。

**直覺**：第一式是「出現幾次 × 平均有多肯定」的乘積——頻率抓跨 run 穩定性，信心抓單次抽取的自我評估；相乘而非取 max，是為了讓「常出現但每次都不太確定」與「偶然出現一次但很肯定」都不至於獨大。第二式決定 ρ 不是裝飾：它以乘性尺度直接縮放關係 embedding 的模長，因此低可靠邊在 attention logit（key 側）與聚合內容（value 側）雙雙被壓低。第三式是 MoE 式殘差門控，λ 給各分支獨立的梯度通路，δ 只學「修正量」而非從零預測權重，原文稱此舉是為了 optimization stability。

**Loss / 訓練**：L = L_mse + φ·L_rank；L_rank 為 RankNet 的 binary cross-entropy，只在 |y_i−y_j|>ε 的有效對集合 Ω 上計算，τ 為 sigmoid 陡度（與窗口 τ 同名，原文重用符號）。φ、τ、ε 的具體數值**所讀文本未見（僅給符號）**。優化器 Adam，lr 1×10⁻³，weight decay 1×10⁻⁵，dropout 0.1，batch size 32，100 epochs，early stopping patience 20。

**複雜度**：靜態圖為一次性成本 O(M·N) 次 LLM 呼叫（N=450／87，M=5），測試期完全凍結；每日動態圖為 O(N²·d) 的 cosine 两兩相似度 + top-k 稀疏；GNN 兩支各 1 層、temporal encoder 2 層。實際 wall-clock 與 API 花費**未披露**。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ 以下所有數字為**假設／示意**，僅用於手算展示機制，**不是論文實證結果**；論文真實數字一律見 §5。門檻 θ_f=0.6、θ_c=0.8、τ=10、α=0.5、k₁=20、k₂=5 為原文 §4.2／§4.3／§5.5 給定的預設值。

1. **單邊聚合**：假設 A→B 的 `Supplier` 邊在 5 次抽取中出現 3 次，三次自報信心為 0.9、0.8、0.7。則 f(e)=3/5=0.6，c̄(e)=0.8，**ρ(e)=0.6×0.8=0.48**。
2. **策略分流**：Strict 判 f(e)≥0.6 → 保留；Highrecall 額外允許 c_max=0.9>0.8 → 也保留；All 無條件保留。若換成「No reliability weight」變體（同一邊集但 ρ→1），這條邊注入 attention 的尺度會從 0.48 變成 1——**可靠性訊號被抹平的同時，edge feature 的量級也被改變**（本頁推斷：這是該消融並非「純可靠性」隔離的原因）。
3. **衰減窗口**：τ=10、α=0.5 下，權重 π_k ∝ exp(−0.5k)，相鄰日的相對權重示意為 1、0.61、0.37（未對 τ=10 全部項做正規化）。直覺：即便某檔股票今天沒新聞，昨天與前天的 embedding 仍以衰減權重留在窗口裡，避免節點孤立。
4. **稀疏化**：A、B 的 cosine 相似度必須互為 top-k₁=20 才連邊；若 A 因新聞稀疏而無任何互惠鄰居，才用 top-k₂=5 補最小連通性。
5. **門控**：假設某日該股 λ_S+δ_S=1、λ_D+δ_D=0，softmax 後靜態權重示意約為 e/(1+e)≈0.73——這就是原文 Figure 3 描述的行為：低新聞覆蓋股（如 NVR）靜態權重偏高，高覆蓋股（如 NVDA）門控更靈敏。

## §3 · 數據層

| 項目 | 內容（逐字自原文） |
|---|---|
| 市場 / 樣本 | S&P 500 與 NASDAQ 100；成分股名單固定於 **September 30, 2023**，並以「全期完整日頻交易資料」過濾後為 **450** 與 **87** 檔 |
| 特徵 | 日頻 OHLCV + 標準技術指標 MA、RSI、MACD，正規化為 per-stock 日頻特徵向量；lookback T=20 |
| 新聞語料 | Thomson Reuters News Archive（TRNA），以 assetId metadata 映射至成分股（多公司稿件指派給每個被標記公司，不在標題做名稱匹配）；僅取 headlines；embedding 用 Voyage-3-large |
| 頻率 / 目標 | 預測 1 日 log return y=log(p^{t+1}/p^t)，學習的是 cross-sectional ranking（y_i>y_j 則 i 排前） |
| 切分 | rolling-window 五折：每折約 7 年訓練、3 個月驗證、3 個月測試；測試窗自 **Oct–Dec 2023 至 Oct–Dec 2024** 逐季前滾 |
| 洩漏控制 | (i) 抽取器知識截止 **2023-10-01**（o3-mini）；(ii) prompt 限制可採信證據窗為 **on or before 2023-09-30**；(iii) 五個測試折全部始於 October 2023 之後；動態圖只用 day t 當日或之前發布的 headlines |
| 容量假設 | 論文的組合為 top-K（K∈{5,10,20}）等權、每日再平衡的 long-only 組合，計 5 bps 單邊換手成本；**換手率、可交易容量、衝擊成本模型均未披露** |

三條時間邊界必須分開記：**LLM 預訓練截止 2023-10-01**（原文註腳稱此為 knowledge cutoff 而非 release date）／**抽取 prompt 的證據窗 ≤2023-09-30**／**測試期 Oct 2023–Oct 2024**。三者幾乎首尾相接——靜態圖在測試期起點時「剛好新鮮」，隨時間推進其老化程度本身就是原文所說「part of the test signal」。文本中**未見**任何對抽取器做微調或適配的描述；Voyage-3-large 的訓練資料截止日**未披露**。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露（所讀文本未見連結；附件待核） |
| Checkpoint | 未披露 |
| License | 論文文本為 CC BY 4.0；代碼授權未披露 |
| 複現難度 | 高：需 o3-mini API 對每檔股票 × 14 種關係跑 5 次（原文稱該模型不開放使用者控制 temperature，只能固定 prompt 與 API 設定）、TRNA 商業授權語料、Voyage-3-large embedding、單張 NVIDIA L40 |
| 數據可得性 | TRNA 需商業訂閱；成分股名單與技術指標可重建；新聞→成分股的 assetId 映射依賴 TRNA metadata，替代語料會改變 §5 全部數字 |
| 工程細節可得 | 超參完整（M=5、θ_f=0.6、θ_c=0.8、Highrecall、τ=10、α=0.5、k₁=20、k₂=5、T=20、d=128、4 heads、1 層 GNN、2 層 temporal encoder、10 seeds、seed 由 global seed 42 決定性衍生） |

## §5 · 評測 / Benchmark

**主表**：前SOTA 欄＝原文 Table 1 所稱 the strongest baseline **COGRASP**（原文措辭為 strongest baseline，本頁不稱其為作者宣稱之 SOTA）；Δ＝本方法 − COGRASP。數值逐字照抄 Table 1（原表以十進位呈現 AR/SR，未在數字後綴 %）。只列 COGRASP 在該 metric 確為最高基線的列；MRR 不列入，因原文未把 MRR 的單一最強基線指名為 COGRASP。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| S&P 500 | AR@5 | 0.4493 | 0.5232 | +0.0739 |
| S&P 500 | SR@5 | 1.5249 | 1.8694 | +0.3445 |
| S&P 500 | AR@10 | 0.3404 | 0.3826 | +0.0422 |
| S&P 500 | SR@10 | 1.4969 | 1.6238 | +0.1269 |
| S&P 500 | AR@20 | 0.2927 | 0.3018 | +0.0091 |
| S&P 500 | SR@20 | 1.4088 | 1.4575 | +0.0487 |
| NASDAQ 100 | AR@5 | 0.3411 | 0.3939 | +0.0528 |
| NASDAQ 100 | SR@5 | 1.1357 | 1.3725 | +0.2368 |
| NASDAQ 100 | AR@10 | 0.2218 | 0.2460 | +0.0242 |
| NASDAQ 100 | SR@10 | 0.9637 | 1.0710 | +0.1073 |

**其他基線（逐字，不合成區間）**：S&P 500 AR@5——FinGPT 0.3997／MLP 0.3564／Transformer 0.3287／StockMixer 0.2855／MASTER 0.2888／RSR-I 0.1901／RTGCN 0.1695／GERU 0.1424／o3-mini 0.2501／ALSTM 0.2401。原文摘要把功勞歸於與 COGRASP 的比較（16.4%／22.6%，NASDAQ 15.5%／20.9%），與上表 Δ 的相對量一致。

**MRR（不列 Δ）**：S&P 500 的較高基線是 StockMixer 0.0365（Transformer 0.0364），LENS 0.0371；NASDAQ 100 較高基線 Transformer 0.0915，LENS 0.0937。原文未把 MRR 的單一最強基線指名為 COGRASP，故不合成前SOTA。

**消融（S&P 500，逐字自 Table 3／5／6，不另計算差值）**

| 變體 | MRR | AR@5 | SR@5 |
|---|---|---|---|
| Highrecall（LENS 預設） | 0.0371 | 0.5232 | 1.8694 |
| No reliability weight（同邊集，ρ→1） | 0.0369 | 0.4647 | 1.7099 |
| Strict（3,162 邊，mean ρ 0.67） | 0.0370 | 0.4589 | 1.6955 |
| All（10,627 邊，mean ρ 0.32） | 0.0377 | 0.3740 | 1.4139 |
| Single-run（M=1） | 0.0371 | 0.3338 | 1.2761 |
| GICS | 0.0354 | 0.3724 | 1.3937 |
| WikiData | 0.0351 | 0.3649 | 1.4237 |
| News Co-occurrence（static） | 0.0370 | 0.3582 | 1.3670 |
| w/o Static Graph | 0.0361 | 0.3375 | 1.2143 |
| Price（動態圖換成價格） | 0.0342 | 0.3881 | 1.4627 |
| News co-occurrence（動態圖換共現） | 0.0342 | 0.4238 | 1.5430 |
| w/o Dynamic Graph | 0.0267 | 0.2617 | 1.0854 |
| Fusion: Equal (0.5/0.5) | 0.0354 | 0.4361 | 1.6278 |
| Fusion: Concat (MLP) | 0.0312 | 0.3608 | 1.3756 |
| Fusion: Learned（單一純量權重） | 0.0371 | 0.4246 | 1.5899 |
| Edge injection: Key only | 0.0367 | 0.4999 | 1.8136 |
| Edge injection: Value only | 0.0356 | 0.5060 | 1.8302 |

**人工驗證（Table 2）**：120 條三元組（90 條來自 Highrecall、30 條來自 single-run），三位標註者，Fleiss' κ=0.58（moderate），80 全一致／30 多數決／10 三方分歧。Highrecall relaxed 91.1% vs single-run 83.3%，unsupported 率由 16.7% 降至 8.9%；strict 54.4% vs 56.7%（n=30，two-proportion z-test p=0.83）；f(e)≥0.6 子集 relaxed 92.9%／strict 78.6%；非競爭關係 96.2%／63.5%；按可靠性三分位 strict 正確率 36.7% / 50.0% / 76.7%。**覆蓋分層 MAE（Table 7）**：High→Low 的落差由 Dynamic Only 的 5.39% 收窄到 LENS (Gated) 的 3.18%（LENS Equal 為 4.24%）。

**解讀——哪部分是真 capability**：最紮實的不是 AR@5 的絕對水平，而是三件事互相咬合：(a) **matched 消融**（No reliability weight 固定同一 Highrecall 邊集、只把 ρ 換成 1）仍掉到 0.4647/1.7099，說明連續權重本身有貢獻；(b) **MAE 覆蓋分層收斂**（5.39%→4.24%→3.18%）是唯一直接針對「可靠性機制是否按設計運作」的分層證據，且與 Figure 3 的門控行為方向一致；(c) **穩定性**：對 COGRASP 在五個 rolling quarterly test folds 全部勝出 AR@5/SR@5，10 個 seed 中 9 個勝 AR@5、10 個勝 SR@5。**哪些要打折**：(i) MRR 絕對量極小（LENS 0.0371，S&P 500 基線中較高者 StockMixer 0.0365），且 All 變體 MRR 0.0377 高於預設 Highrecall 0.0371——排序能力並非單調跟隨可靠性設計，MRR 不宜當成機制證據；(ii) 52.32% 年化來自 3 個月測試窗的年化放大與每日再平衡 top-5，換手率未披露，文本只描述 5 bps 單邊換手成本，未見其他衝擊成本模型；(iii) 成分股以「全期完整資料」過濾（S&P 500 → 450），該過濾條件跨越測試窗，屬本頁推斷的 universe-level 前視；(iv) 顯著性檢定是 **over 10 seeds 的 paired t-test**，量的是種子變異而非跨時間抽樣變異，只有五個獨立測試折，統計力度有限（本頁推斷）；(v) Table 1 是 end-to-end 系統比較，各基線保留原始資訊源，作者自己在 §5.3 承認這點。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位至原文 §6）

1. **單一抽取器**：靜態圖僅來自一個 LLM，其知識截止早於評估期，雖控制了資訊洩漏，但「cross-extractor robustness untested」——換一個抽取器是否仍成立，未測。
2. **市場與 regime 侷限**：兩個宇宙都是美國大型股；在公司關係更稀疏的市場、或在「not primarily bullish」的 regime 下行為如何，作者明言 remains open。
3. **非實盤**：評估是歷史回測而非實盤部署；比較對象是已發表的學術基線，不含業界專有 ensemble。
4. **方法內部自承的粗糙處（§4.3 末）**：headline-level 語義 embedding 比顯式事件／實體關係抽取更粗，會容納 topical co-mentions；作者把更細粒度的事件抽取列為 future work，並主張靜態 prior 與 gated fusion 對此類噪聲有緩衝。

### 6.2 本頁推斷的隱含假設（非作者原文，僅標為推斷）

- **宇宙構建含前視**（推斷）：成分股以「全期完整日頻交易資料」篩選，而「全期」涵蓋測試窗，等於用未來資訊決定誰進樣本；此為 universe-level survivorship，非作者所述之三處洩漏控制之一。
- **新聞時間戳粒度未披露**（推斷，需核）：原文只說動態圖用「headlines published on or before day t」，未區分盤中／收盤後。若含收盤後稿件，對 t→t+1 的 close-to-close 預測即存在同日盤後資訊優勢。
- **ρ 的校準假設**（推斷）：整套方法把 LLM 自報 confidence 當作可靠性代理，但該 confidence 是否跨股票、跨關係類型、跨 prompt 一致校準，文本只給出人工驗證中的可靠性三分位單調關聯（見 §5 人工驗證段）與小型樣本，不足以支撐跨域外推。
- **ρ→1 消融不是純隔離**（推斷）：把 ρ 換成 1 同時改變了 edge feature 在 key/value 中的尺度；Highrecall 與 No reliability weight 的 AR@5 數值差異見 §5 Table 3（原文數值），但該消融混合了「可靠性訊息」與「尺度校準」兩種效應，無法完全歸因於前者。
- **成本與容量**（推斷）：每日再平衡 top-5 等權的換手率未披露，交易成本僅計及原文 §5.4 預設的單邊換手成本假設，而非實證估計；在小市值或高波動 regime 下該假設失效的速度未知。
- **輔助數據缺失不是生存者偏差**：TRNA 對某些公司覆蓋稀疏是「長尾覆蓋」而非「樣本被刪除」，本文以 fallback 連接與 gated fusion 處理，這點與 survivorship 是不同問題，不應混談。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| COGRASP（IJCAI-25） | 同樣用新聞建圖，但走共現統計而非語義 embedding；無 LLM 先驗、無可靠性加權 | 代碼狀態未核 | 原文最強基線，S&P AR@5 0.4493 / SR@5 1.5249 |
| Zhou et al. 2025（同作者前作） | 單次 zero-shot LLM 抽圖，邊為 binary；LENS 的直接前作與被批評對象 | 未核 | 本文 M=5+ρ 的對照組即 Single-run（AR@5 0.3338） |
| GERU（TSMC 2023） | 純價格相關動態圖；本文指其與價格特徵冗餘 | 未核 | Table 1 S&P AR@5 0.1424 / SR@5 0.5765 |
| MASTER（AAAI-24） | 市場引導式 transformer，隱式建模相關，不建顯式圖 | 未核 | Table 1 S&P AR@5 0.2888 |
| FinGPT / o3-mini 直接預測 | 把 LLM 當端到端預測器或情緒標註器，而非圖先驗來源 | 未核 | S&P AR@5 分別 0.3997 / 0.2501，原文稱「treating LLMs as standalone predictors is less effective」 |

🎤 **Interview Tip**
問：「LENS 的增益主要來自 LLM 靜態先驗，還是新聞動態圖？」
- **錯答**：「來自 LLM 先驗，因為 w/o Static Graph 掉到 0.3375 比 w/o Dynamic 的 0.2617 好。」——把兩個不同殘缺程度的變體當成可比的歸因，且忽略兩者都遠低於 full model，這正是原文 §5.6 所說的 complementarity。
- **正答**：「不能單獨歸因。兩個移除實驗各自掉到 0.3375 與 0.2617，都遠低於 0.5232，說明必須同時在場；而且 Table 1 是 end-to-end 系統比較，基線保留各自資訊源，只有 Table 3/5/6 才是 matched 消融。連那個最像『純可靠性』的 No reliability weight 消融，也因為把 ρ 設為 1 而同時改變了 edge feature 尺度，不能算乾淨隔離——這是我讀完後的推斷，不是作者的話。」

### 7.1 可證偽預測（帶日期）

1. **2026-12-31 前**：若有人在 2025–2026 的樣本外重跑同一協定，LENS 對 COGRASP 的 AR@5 相對提升應顯著小於原文 S&P 500 的 16.4%——因為靜態圖凍結於 2023-09-30，其有效期限隨時間衰減；若提升反而擴大，則說明增益來自動態圖或 fusion 而非 LLM 先驗。
2. **代碼釋出後 30 日內**（假設釋出）：檢查新聞時間戳處理，若 day t 圖納入當日收盤後發布的 headlines，則 S&P 500 AR@5 0.5232 中應有一部分為時間戳前視，可在移除盤後稿件後重測驗證。
3. **2026-12-31 前**：若把成分股改為「以樣本內可得性」而非「全期完整資料」構建，LENS 相對 COGRASP 的差距應縮小；差距不縮小則本頁「推斷的隱含假設」的 universe 前視推斷被否證。

## §8 · For the Reader

- **因子研究員**：ρ(e)=f(e)·c̄(e) 這個「頻率×信心」的乘性聚合可直接移植到任何 LLM 生成的特徵（供應鏈、專利、招聘關係），但請先做自己的校準檢定——本文只在 120 條三元組上給了三分位單調關聯，遠不足以支撐跨域使用。真正可借用的是「別在預處理階段二值化」這一原則。
- **高頻執行**：本文每日再平衡 top-5 等權、只計 5 bps 單邊換手，換手率未披露；對日內／T+1 執行者，關鍵風險是新聞時間戳粒度（見「推斷的隱含假設」），務必先確認 headline 的可得時點再談 alpha。
- **組合配置**：把它當成「關係型 alpha 的一種來源」而非擇時工具——測試窗僅五折（Oct–Dec 2023 至 Oct–Dec 2024）、作者自承非牛市 regime 未驗；配置前應要求跨 regime 的分層結果。
- **LLM-agent 開發者**：LENS 示範了 agent 不確定性的正確用法——不要用多次投票取多數（硬閾值），而要把投票分布與自報信心一起送進下游可微模組。gated fusion 的 λ+δ 解耦（內在評估 + 上下文修正）也是穩定訓練的實用技巧。
- **RL 策略研究者**：門控模組本質上是一個 2-way soft routing，可以無痛推廣到 N 個資訊源；但本文的 δ 是 concat 後的一次性修正，不是序列決策，若要做 regime-aware routing，需自行擴展。
- **研究學生**：這篇是「同一個模型、只改資料可靠性表達」的教科書式消融設計——注意它如何用 matched 消融（固定邊集、只動 ρ）把一個模糊概念（可靠性）逼成可測量對象，也注意它在 §5.3 誠實承認 Table 1 是 end-to-end 比較。

## References

- 原文：Hao Zhou, Tiejun Ma, Luis Felipe Costa Sperb. *LENS: Reliability-Aware Stock Prediction via LLM-Extracted Priors and News-Driven Semantic Graphs.* CIKM '26. arXiv:2611.04217v1 [cs.LG], 03 Nov 2026. https://arxiv.org/abs/2611.04217
- 直接前作（靜態圖來源與被批評對象）：Zhou, Costa Sperb, Ma. *Zero-Shot Extraction of Stock Relationship Graphs with LLMs.* FinNLP 2025. doi:10.18653/v1/2025.finnlp-2.2
- 最強基線（新聞共現圖）：Li et al. *COGRASP: Co-Occurrence Graph Based Stock Price Forecasting.* IJCAI-25. doi:10.24963/ijcai.2025/837
- 排名任務與靜態圖基線：Feng et al. *Temporal Relational Ranking for Stock Prediction.* ACM TOIS 2019. doi:10.1145/3309547
- 價格相關動態圖：Tian et al. *Learning Dynamic Dependencies With Graph Evolution Recurrent Unit for Stock Predictions.* IEEE TSMC 2023. doi:10.1109/TSMC.2023.3284840
- 動態圖結構偏置：Ying et al. *Do Transformers Really Perform Bad for Graph Representation?* NeurIPS 2021
- 靜態圖 edge-feature 注入：Dwivedi & Bresson. *A Generalization of Transformer Networks to Graphs.* arXiv:2012.09699
- 門控靈感：Shazeer et al. *Outrageously Large Neural Networks: Sparsely-Gated Mixture-of-Experts.* ICLR 2017
- 排序損失：Burges et al. *Learning to Rank using Gradient Descent.* ICML 2005. doi:10.1145/1102351.1102363
- 滾動窗協定：Gu, Kelly, Xiu. *Empirical Asset Pricing via Machine Learning.* RFS 2020. doi:10.1093/rfs/hhaa009
- Embedding 基準：Tang & Yang. *FinMTEB.* EMNLP 2025. doi:10.18653/v1/2025.emnlp-main.179
- 來源鏈接：arXiv HTML 全文（本頁唯一事實來源）；Figure 1/2/3 與附件未提取，相關曲線細節待核。