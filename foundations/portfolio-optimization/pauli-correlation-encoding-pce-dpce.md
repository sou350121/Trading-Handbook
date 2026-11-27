<!-- ontology-5axis data=图关系 horizon=中长周期 paradigm=元学习搜索 alpha=组合执行优化 autonomy=全自动黑盒 -->

# Pauli Correlation Encoding (PCE / dPCE) 解構（Pauli Correlation Encoding (PCE / dPCE)）

> **來源發布**：2026-11-24 · （無 venue） · arXiv [2611.32769](https://arxiv.org/abs/2611.32769)
> **arXiv 原文**：[Large-Scale Streaming Portfolio Optimization with Pauli Correlation Encoding](https://arxiv.org/abs/2611.32769v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點＝圖關係／中長周期／變分參數搜索／組合執行優化／全自動黑盒。解的是 [9] 靜態單快照研究留下的 prior gap：圖割選股在**每次調倉都重跑**的滾動部署下，成本與收益是否還站得住。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `图关系` | `中长周期` | `元学习搜索` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把 PCE 圖割選股從單一相關快照推進到 `window_size=126` 日滾動窗、`step_size=21` 日換倉的流式調倉，並新增 κ 層聯合多割變體 dPCE。②Trick 有兩層：`O(log n)` qubit 編碼（N=5,000 只用 15 qubits）＋把 shot budget 從硬體約束降級為指標選擇問題——cut-value 在 1,000 shots 已達 exact 值的 94 – 107%。③對「組合執行優化」這條軸★，真正的貢獻是把 rebalance cadence 確立為較乾淨的控制項（turnover 隨 step_size 上升，而總報酬與 step_size 為負 rank correlation，兩者方向相反），而非把 risk-aversion 旋鈕當風險控制（對 Sharpe 非單調）。④關鍵數字：N=50 總報酬 338.2% vs 等權 Buy & Hold 83.9%；N=5,000、2021–2024 空頭窗 −11.0% vs −17.9%——但基線是**全市場等權**、無 matched-k 對照、回測 frictionless。

**X-Ray.** 這是一個純粹的組合執行工具，不是 alpha 模型：它不預測報酬，只決定相關圖怎麼切群、每群取一檔等權持有。被解掉的工程坑有二：(a) 靜態研究無法回答的「每次調倉重跑是否還划算」；(b) qubit 壓縮在滾動部署下的真實成本結構——PCE 用 k−1 次小電路呼叫、dPCE 用 1 次大電路，§4.9 量到 N=50→300 時 PCE 0.5s→6.5s（~13×）對 dPCE 0.65s→73s（~111×），這一條基本上把 dPCE 判出實務隊列。它打不開的 envelope 同樣清楚：持股數 = `n_clusters`（N∈{50,100,200,300} 時為 3/5/10/15），而基線是全市場等權 Buy & Hold，**沒有同持股數的隨機子集對照**（作者自己在 §5 第一點承認），所以「圖割選股有效」與「少量集中持股本來就打贏全市場等權」在這份證據裡無法分離。所有回測 frictionless，τ 只證明了 turnover 下降（10 組配置平均 11.3%±2.8%），未證明淨成本後報酬改善。對量化讀者的可攜價值不在「量子」，而在控制項排序的方法論：cadence 單調、ρ 非單調。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作 | 本文 | 為什麼這條改變了評測含義 |
|---|---|---|---|
| 部署形態 | [9] 單一歷史相關快照、一次圖割、一個組合、樣本外評一次 | 每個 rebalance date 用 126 日 trailing window 重估相關、重聚類、持有 21 日、窗戶前移重複 | 每個超參的成本／收益**複利**在整段回測上，不再一次性支付 |
| 演算法家族 | [8] 通用 graph-cutting 的單次示範；[9] 套到組合的靜態版 | 兩種流式變體：PCE 遞歸二分（k−1 次量子呼叫、子圖遞減）與 dPCE 聯合多割（1 次呼叫、作用在 κn 節點的分層圖） | 引入一條新取捨軸：qubit 數 vs 呼叫次數 |
| 選股目標 | 純聚類後取代表檔 | 疊上 stickiness 加成 `score(a)=Sharpe_window(a)+τ·1[a∈held_{t-1}]`，且 ρ 同時驅動相關加權、流動性下限（排除成交量最低的 25ρ%）、τ=τ_max·ρ | 把 stickiness 加成寫進**選股分數**（Eq. 6），而非事後扣費 |

### 1.2 ⚡ Eureka

**「部署成本由 cut-value 的飽和點決定，不是由 bitstring 的恢復精度決定。」** 量測 shot 增加時，Hamming distance 到 exact 解一路不飽和（到 10⁵ 仍下降），但 portfolio 相關的 cut-value 在 1,000 shots 就到達 exact 值的 94 – 107% 並持平——因為還在翻的 bit 集中在低邊權（分散化無關）節點上，`x̃` 本已接近 0。這把「shot budget」從一個看起來只能等硬體的硬約束，變成一個**換指標就能省**的量。第二層直覺是 qubit 壓縮本身：6 節點玩具例只需 m=2 個 qubit（Figure 1），N=5,000 只需 15 個——這才是「大規模」二字的來源。

### 1.3 信息流

```
daily OHLCV（窗內 T=window_size 日）
   │ Eq.(7)(8) recency×volume 權重 w_t（λ=2^(-1/τ(ρ)), τ(ρ)=21+231ρ）
   ▼
加權相關 ρ̂_uv ──Eq.(5)──► w_uv = 1-|ρ̂_uv|，窗內線性 rescale 到 [0,1]，低於門檻邊直接丟棄
   ▼
相關圖 G=(V,E,w)，|V|=N
   │ Eq.(2)：3Σ_{k≤order} C(m,k) ≥ n → m qubits
   ▼
U(θ)（LayeredHEA, 3 repetition layers）──量測 ⟨Π_i⟩──► x̃_i = tanh(α⟨Π_i⟩), α=m^(order/2)
   │ Eq.(3) PCE：遞歸 k-1 次二分  ／  Eq.(4) dPCE：單次 κ 層聯合
   ▼
COBYQA 外迴圈（gradient-free；warm-start 沿用上一窗 θ 為初值）
   ▼
b_i = 1[x̃_i>0] → k clusters → 每群 1 檔等權（cost-aware score + 流動性下限）
   ▼
持有 step_size=21 日 → 窗戶前移 → 下一輪
```

## §2 · 數學層

📌 **Napkin Formula**

- qubit 需求：`3·Σ_{k=1}^{order} C(m,k) ≥ n`，m 取最小整數（Eq. 2）
- PCE loss：`L_PCE = Σ_{(u,v)∈E} w_uv·x̃_u·x̃_v + β·v·(1/n·Σ_i x̃_i²)²`，`x̃_i=tanh(α·result_i)`，`v=|E|/2+(n-1)/4`，`β=0.5`（Eq. 3）
- 建圖：`w_uv = 1 - |ρ̂_uv|`（Eq. 5）

複雜度：qubit 數 `O(log n)`；PCE 每次調倉 k−1 次量子呼叫（後續呼叫作用在遞減子圖，故首次全圖呼叫的 qubit 數為上界）；dPCE 1 次呼叫但作用在 κn 節點、`κ=⌈log₂k⌉`。

直覺：cut term 對高權重邊（低相關對）逼 `x̃_u x̃_v → -1`，即推到不同側；balance regularizer 罰全體同號的不平衡分割；最後 threshold 成離散 label，**全文報告的是離散 cut C(b)，不是鬆弛值本身**。dPCE 的關鍵改動是 Eq.(4) 把係數換成 `2w_uv-1 = 1-2|ρ_uv|`——強相關對得負係數被拉進同群、弱相關對得正係數被分開；不做這個 shift，強相關對在原損失裡貢獻 ≈0，會在群間亂散。

訓練：COBYQA（gradient-free，適合 shot-based 噪聲目標與無人值守的每次調倉）、LayeredHEA 為預設 ansatz（3 層；BrickworkPCE 可用但非預設）、warm-start 預設開啟。所讀文本給出的 `(N, order, m)` 組合為：`order=2` at N∈{50,100}、`order=3` at N∈{200,300,1,601}、`order=4` at N=5,000；對應 6 qubits at N=50、8 at N∈{100,200}、9 at N=300、15 at N=1,601 與 N=5,000（§2.2）。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部為假設/示意數字，非論文實證結果。** 論文實證數字一律見 §5。

1. **假設** 只看 3 檔資產 A、B、C，窗內相關 ρ̂：`ρ_AB=0.1`、`ρ_AC=0.85`、`ρ_BC=0.1`。按 Eq.(5)（示意，略去窗內 rescale）：`w_AB=0.90`、`w_AC=0.15`、`w_BC=0.90`。語義檢查：A、C 同向 → 低權重邊；A、B 與 B、C 幾乎無關 → 高權重邊。
2. **Eq.(2) 檢查**：假設 order=1、m=2，則 `3·C(2,1)=6 ≥ n=3`——此處僅示意 m=2 可容納 n=3；按 Eq.(2) 取最小 m 時，n=3、order=1 只需 m=1（`3·C(1,1)=3 ≥ 3`）。m=2 對應的是原論文 Figure 1 的 6 節點簡化實例，並非 n=3 的最小 qubit 需求。
3. **假設** 最佳化收斂後量到 `⟨Π_A⟩=+0.5, ⟨Π_B⟩=-0.6, ⟨Π_C⟩=+0.4`；threshold `b_i=1[x̃_i>0]`（x̃ 取 tanh，示意）→ `b=(1,0,1)`，即 clusters = {A,C}、{B}。
4. **離散 cut 值**（Eq. 1）：跨群邊為 AB 與 BC → `C(b)=w_AB+w_BC=0.90+0.90=1.80`。對照 `{A,B}|{C}` 得 `0.90+0.15`、`{A}|{B,C}` 得 `0.90+0.15`，都更低——最優解確實把「一起動的 A、C」放同群、把異類 B 切出去。這就是「最大化 cut = 最大化分散化」的全部內容。
5. **選代表 + 成本感知**：假設 `ρ=0.5` → `τ=τ_max·ρ=0.05·0.5=0.025`；假設 `Sharpe_window(A)=1.2`、`Sharpe_window(C)=0.8`，且 C 為上期持有 → `score(C)=0.8+0.025=0.825 < 1.2`，仍換成 A。示意重點：τ 只在候選差距小於 hurdle 時才擋得住換倉，它是一個「Sharpe 單位的門檻」，不是 price-impact 模型。

## §3 · 數據層

三個日頻 OHLCV 資料集（Appendix A, Table 6），**同一 N 內的實驗固定同一回測窗**：

| 資料集 | 原始 tickers | 窗口 | 完整歷史 | clean (N) | 用途 |
|---|---|---|---|---|---|
| S&P 500, 5yr | 505 | 2013-02-08–2018-02-07 | 470 | 470（用 300） | N≤300、§3.3 市場圖、方法學參數研究 |
| `prices.csv` | 2,070 | 2013-02-08–2018-02-07 | 1,671 | 1,601 | N=1,601 大規模 |
| `all_stock_data.csv` | 9,315 | 2021-01-01–2024-01-01 | 6,869 | 5,049（用 5,000） | N=5,000 |

- **頻率/時段**：日頻；rolling window，每次持有樣本外 `step_size=21` 交易日，窗戶前移 21 日（相鄰窗重疊 `window_size-step_size` 日，相關估計**不獨立**）。
- **清洗**：先定回測窗，再只保留該窗內價格完整無缺口者；單日漲跌幅 >50%（含拆股/股利調整後）視為資料品質問題剔除。
- **市場圖實測（N=50，Table 1）**：節點 50、邊 483、density 0.394、mean degree 19.32、mean clustering coefficient 0.699、連通分量 1、孤立節點 0、mean edge weight 0.772——密集且高聚類、單一主連通塊，正是「少數幾刀就能切出 cohesive 社區」的結構。
- **成對相關分佈（Table 7，欄名標單位為 %）**：

| 宇宙 | Mean | Median | Std | % <0 | % >0.5 | Pairs |
|---|---|---|---|---|---|---|
| S&P 500, N=300 | 0.278 | 0.267 | 0.118 | 0.3 | 3.8 | 44,850 |
| `prices.csv`, N=1,601 | 0.230 | 0.222 | 0.111 | 1.3 | 1.7 | 1,280,800 |
| `all_stock_data.csv`, N=5,000 | 0.166 | 0.147 | 0.134 | 7.0 | 1.5 | 12,497,500 |

- **容量假設**：僅有「排除成交量最低 25ρ%」的流動性下限，**未披露任何 AUM／衝擊成本／容量上限**；未披露資料供應商與取得方式（檔名為 `prices.csv`、`all_stock_data.csv`）。樣本外＝滾動窗內外切分，無獨立 holdout 段。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 未披露 | 不適用（每個 rebalance 重新最佳化變分參數，非固定權重；warm-start 僅沿用上一窗 θ） | 未披露 | 高：需量子態向量模擬器（可訓練到收斂）＋ COBYQA ＋ 自建 rolling backtest ＋ `networkx` 圖統計 | 未披露；僅知三份資料集名稱與窗口，無下載連結 |

## §5 · 評測 / Benchmark

**註**：所讀原文的比較對象是 **equal-weight Buy & Hold over 同一候選全集**（§3.9），原文從未稱其為 SOTA。下表沿用模板欄名，前SOTA 欄即該基線的逐字數值。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| S&P 500, N=50（2013-02-08–2018-02-07） | 總報酬 | 83.9% | 338.2% | +254.3pp |
| S&P 500, N=50（同上） | Sharpe | 5.09 | 7.94 | +2.85 |
| S&P 500, N=300（同上） | 總報酬 | 66.4% | 125.1% | +58.7pp |
| S&P 500, N=300（同上） | Sharpe | 5.37 | 6.68 | +1.31 |
| `prices.csv`, N=1,601（同上） | 總報酬 | 79.2% | 87.3% | +8.1pp |
| `prices.csv`, N=1,601（同上） | Sharpe | 5.03 | 4.62 | −0.41 |
| `all_stock_data.csv`, N=5,000（2021-01-01–2024-01-01，†不可與上列直接比） | 總報酬 | −17.9% | −11.0% | +6.9pp |
| `all_stock_data.csv`, N=5,000（同上†） | Sharpe | −1.76 | −0.47 | +1.29 |

**Δ 是不是真 capability？拆三層看：**

1. **確定是真效果的部分**：同一窗口、同一全集下，PCE 在 N∈{50,100,200,300} × window∈{63,126,252} 的**每一個** (N, window) 組合都勝過 B&H（§4.1、§4.4，圖 6／圖 9 待核）；且 N=5,000 的空頭窗是「少虧」而非「多賺」——PCE 最差點 0.804/美元（−19.6%，2022-06-02）vs B&H 0.778（−22.2%，2023-10-04），且在 29 個調倉日中 4 次回到 1 元以上（峰值 +5.6%，2023-02-02），B&H 只有 2 次、且 2021-10-01 之後再也沒有。這個「回撤期少虧」的證據方向是乾淨的（來源未做 beta／因子分解，故不宜解讀為不靠 beta）。
2. **高度可疑的部分（overfitting／指標選擇）**：Sharpe 用「年化均值／年化波動、rf=0」（§3.5），作者自述在 2021–2024 高利率窗會**虛高**；且 Sharpe 隨 ρ 非單調（PCE, N=50：6.30→6.93→5.42→7.42→7.77），N=1,601 的 Sharpe 反而輸 B&H（PCE 4.62 vs B&H 5.03），所以「Sharpe 全面勝出」不成立。dPCE 在 N=50 亦非穩贏：ρ=0.5 的 Sharpe 為 4.36、ρ=1 為 5.04，均低於同列 B&H 的 5.09。
3. **完全沒計的部分（成本／容量）**：回測 frictionless，「交易成本感知」只改了選股分數；總報酬隨 τ 開啟的變動是 −53 到 +56 個百分點、無一致方向；唯一穩健的是 turnover 下降（10 組配置平均 11.3%±2.8%，逐列由 6.8% 到 15.6%）。**這裡的 Δ 是投資組合集中度 + 圖割選股的混合效果，不是可歸因於圖割的淨 alpha。**

**內部一致性紅旗**：§4.5 文字稱 dPCE 在小 N 峰值報酬較高，卻同時列出「dPCE ~3.2× vs. PCE ~4.4×」（N=50, W=126 d），敘述與數字方向相反——圖 9／圖 10 待核。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **無 matched-k 對照**（§5 第一點）：基線是全市場等權，缺「從同一全集隨機抽 k 檔」的控制組，因此無法分離「圖割選股」與「集中度／異質波動」效果。作者自稱這是最有價值的缺失實驗。
- **單一路徑、無區間估計**（§5 第二點）：每個配置只有一個歷史路徑的點估計，全文沒有任何 bootstrap CI 或顯著性檢定；§4.6 的 rank correlation 部分由**僅 4 個配置**算出。
- **規模與 regime 混淆**（§5 第三點／§4.9）：N=5,000 只有一個資料點，且必須換到 2021–2024 窗，不能當成 N=1,601 趨勢的延續。
- **dPCE 的 cadence 敏感性只刻畫到 N=100**（§5 第四點／§4.6），更大 N 是惡化、平台化或反轉未知。
- **126 日窗最優無法分解**（§4.1 末）：Marchenko–Pastur 訊號特徵值佔比隨 T 單調上升到 T≈150–250 日後平台化，無法解釋 252 日最差；作者歸因為 staleness-vs-noise，但明言「沒有乾淨分解，列為 open question」。且該診斷要求 T>N，對 N=1,601 與 N=5,000 根本不適用。
- **Sharpe 慣例**（§3.5）：rf=0，所有策略與基線一致套用，相對比較不受影響，但絕對水平虛高。
- **order 非每尺度最佳化**（§2.2）：是「維持 qubit 在模擬器可行預算內」的可行性選擇，不是 qubit 效率最佳化。

### 6.2 推斷的隱含假設（本頁推斷，非作者自述）

- **存活者偏差（推斷）**：清洗規則是「回測窗內價格完整無缺口」＋剔除單日 >50% 變動。這會系統性排除期內下市／停牌／被併購的標的，等於在**已知存活**的宇宙上做選股。作者只把它寫成 data-quality check，未討論對 Δ 的方向性影響。圖表／附件待核。
- **成本模型太薄（推斷）**：τ 是 Sharpe 單位的 hurdle，不是衝擊函數；作者自述這是刻意的單參數簡化（§3.5）。在 frictionless 回測下，τ 的價值只能以 turnover 代理，**不能**推論淨成本後的報酬改善。
- **圖割 ≠ 風險模型（推斷）**：聚類只用 `|ρ̂|` 相似度，目標函數裡沒有因子暴露、產業中性、beta 或波動目標；「分散化」在此僅指相關性分群。持股數 = n_clusters（N=5,000 時為 30），故組合風險主要由 k 決定，不由 cut 質量決定。
- **全自動黑盒的前提（推斷）**：COBYQA 無 per-instance tuning 是為了無人值守；但 N=1,601 的部署證據用的是 1,000 shots，更大 N 的 shot 預算與電路深度可行性未驗證（N=5,000 也是 1,000 shots，單點）。
- **regime 依賴（推斷）**：唯一的空頭證據是單一路徑的 2021–2024；「跌市少虧」目前是**一個**觀測，不是已建立的性質。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Equal-weight Buy & Hold（本文基線） | 持股數（k vs 全集）、換手 | 不適用 | 本文唯一實測基線 |
| 層次聚類 / HRP（[3] López de Prado） | 聚類方式；無 qubit 壓縮 | 未披露 | 僅在 §1 被引為動機，**本文未實測比較** |
| Spectral graph partitioning | 同一圖割目標的經典近似 | 未披露 | 僅在 §1 被引，無實測數字 |
| QAOA / 量子退火（[4][5][6]） | 一變量一 qubit → 規模受限 | 未披露 | 僅在 §1 對比編碼效率，無實測 |
| PCE 靜態版（[9]） | 單快照 vs 滾動重估 | 未披露 | 前作；本文未重跑其靜態基線做同表對比 |
| dPCE（本文變體） | qubit 數 ×κn vs 呼叫數 k−1 | 未披露 | 本文內部對比：return 近似、runtime 與 cadence 敏感性明顯劣 |

🎤 **Interview Tip**
- **正確答**：「這篇的 benchmark 是等權 Buy & Hold 全市場，不是 matched-k 隨機組合；所以它證明的是『PCE 選出的 k 檔打贏全持有』，不是『圖割比隨機抽 k 檔好』。而且 N=5,000 那組 −11.0% vs −17.9% 同時換了規模與市場 regime，作者自己標成不可直接比。」
- **錯答**：「PCE 在 5,000 檔上 Sharpe 也贏 SOTA」——N=5,000 的 Sharpe 是 −0.47 vs −1.76（兩者都負），N=1,601 的 Sharpe 其實是 B&H 5.03 高於 PCE 4.62；且原文沒有任何 SOTA 宣稱。另一個常見錯答是把「交易成本感知」說成「已計入成本」——回測是 frictionless，只量到 turnover 下降。

### 7.1 可證偽預測（帶日期）

- 若在 **2027-12-31** 前出現跟進工作，補做「同全集隨機抽 k 檔」對照，且 PCE 對該對照的優勢消失，則本文的選股 capability 主張應降級為集中度效應。
- 若在 **2027-06-30** 前有人把 N=5,000 跑在 2013-02-08–2018-02-07 這類同窗口上，而 PCE 總報酬不再高於 B&H（N=1,601 為 87.3% vs 79.2%），則「規模擴大仍增值」的敘事被削弱——這正是作者列為 missing experiment 的那一項。
- 若 dPCE 的 cadence 敏感性在 N>100 被測出不再變差（目前 PCE rank correlation 在 −0.2 到 −0.4、dPCE 在 −0.4 到 −0.8 並隨 N 與 n_clusters 更負），則「dPCE 僅適合小宇宙＋短 step」的實務判準需要修正。

## §8 · For the Reader

- **因子研究員**：把這篇當「圖割 clustering 的 streaming 骨架」讀，不要當 alpha 論文。可直接借的是 Eq.(7)(8) 的 recency×volume 加權相關估計，與「ρ 同時驅動三機制」這種耦合旋鈕的反面教材（對 Sharpe 非單調，PCE N=50 從 6.30→6.93→5.42→7.42→7.77）。要驗的是 matched-k 對照。
- **高頻執行**：與你無關的層在毫秒；有關的是 `step_size`——對 turnover 是 rank correlation 1.00 的正向、對總報酬在每個配置都負向，即**不頻繁調倉是 dominated strategy**。另外 τ 是 Sharpe 單位 hurdle 而非衝擊函數，別直接接進 execution 層。
- **組合配置**：持股數 = n_clusters，這是唯一真正的風險旋鈕，比 ρ 乾淨。若你要的是回撤控制，本文唯一空頭證據（−19.6% vs −22.2% 最差點、29 個調倉日中 4 次回到水上）只是一個資料點。
- **LLM-agent / 自動化 pipeline**：warm-start 的結果值得記住——cut 質量 0.926 vs 冷啟 0.889，但**收斂迭代數完全沒減少**（每窗同 Count 收斂或撞上限）。把「warm-start 加速收斂」寫進 agent 的 cost model 是錯的。
- **RL 策略**：dPCE 的 κ 層聯合損失（Eq. 4）是「同一份圖複製 κ 層 + 乘積型 soft-AND 指示器」的多標籤鬆弛技巧，可移植到任何多類別指派問題；代價是 qubit／節點數 ×κn。
- **研究學生**：這篇是很好的「如何把靜態 benchmark 論文擴成 streaming」範本，同時也是「單一路徑、無 CI、frictionless」的典型弱點清單；§5 的四條 limitations 本身就是一份可做的後續工作列表。

## References

- 原論文：V. P. Soloviev, J. Kirsopp, *Large-Scale Streaming Portfolio Optimization with Pauli Correlation Encoding*, arXiv:2611.32769v1 [quant-ph], 24 Nov 2026 — https://arxiv.org/abs/2611.32769v1
- Lineage（編碼）：[8] Sciorilli et al., *Towards large-scale quantum optimization solvers with few qubits*, arXiv:2401.09421（PCE 原始 O(log n) 編碼，本文 §4.8 shot 協定沿用其「訓練用 noiseless、只在 final readout 加 shot」）
- Lineage（組合應用）：[9] Soloviev & Krompiec, *Large-scale portfolio optimization using Pauli correlation encoding*, Sci. Rep. 16(1):25158；arXiv:2511.21305（本文直接接續的靜態前作）
- 擴展：[10] Padín-Martínez et al. arXiv:2602.17479；[11] Soloviev & Adhikari arXiv:2604.05637；[12] Alonso et al. arXiv:2606.18914；[13] Reeves et al. arXiv:2610.03949；[14] Sharma & Lau arXiv:2503.12121；[15] Yoshioka et al., Phys. Rev. Applied 26:044004 / arXiv:2607.24722；[16] do Carmo et al., Braz. J. Phys. 56(1):49 / arXiv:2509.14414
- 方法學：[1] Markowitz (1952)；[3] López de Prado (2016)；[17] Almgren & Chriss (2000)；[18] RiskMetrics (1996)；[19] Sim, Johnson & Aspuru-Guzik (2019)（entangling capability / expressibility 診斷）
- 來源連結：https://arxiv.org/abs/2611.32769v1（本次輸入為完整可讀全文文本；圖片、附件未提取，Figure 6–18 與 Appendix A 圖表待核）