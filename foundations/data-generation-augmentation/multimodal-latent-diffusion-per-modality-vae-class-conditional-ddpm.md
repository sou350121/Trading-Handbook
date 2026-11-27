<!-- ontology-5axis data=多模态 horizon=日频波段 paradigm=生成式大模型 alpha=风险择时 autonomy=全自动黑盒 -->

# Multimodal latent diffusion (per-modality VAE + class-conditional DDPM) 解構

> **來源發布**：2026-11-05 ·（無 venue 欄；`\workshoptitle` 標為 *Second Workshop on ML × OR: Mathematical Foundations and Operational Integration of Machine Learning for Uncertainty-Aware Decision-Making*）· arXiv [2611.07282](https://arxiv.org/abs/2611.07282)
> **arXiv 原文**：[Scenario Augmentation for Rare Financial Events via Multimodal Latent Diffusion](https://arxiv.org/abs/2611.07282v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片（Figures 1–3）／附件未核_
> **核心定位**：五軸落點＝多模態（tabular／DeepInsight 影像／等權收益序列）、日頻 5–8 日窗、生成式（per-modality VAE + class-conditional latent diffusion）、以 CVaR 標籤導向少數 regime 的分類輔助、全自動黑盒。它補的 prior gap 不是「更強分類器」，而是**少數類樣本的可生成性**：在只有 TabDDPM／TabSyn 這類純表格生成器的生態裡，補上「對齊多視角 + 類別條件」以維持少數類流形的局部真實度。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `多模态` | `日频波段` | `生成式大模型` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文可讀全文。細節待升 v1。
**TL;DR:** 把同一個市場窗口拆成 tabular／DeepInsight 影像／等權收益序列三模態，各訓一個 VAE、以對齊損失與潛碼標準化把三者拉到可比尺度後拼接，再用 CVaR 定義的 rare/normal 標籤做類別條件潛擴散生成少數類樣本。核心 trick 是**潛碼尺度與協方差對齊（standardize + tabular 部分 whitening）**，讓拼接後不至於被單一模態支配。對「風險擇時」軸的意義：少數類樣本不再靠重加權或 SMOTE 硬撐，而是可生成、可控標籤的壓力情境。關鍵實證：Real+Synth→Real 的 XGBoost AUC 在四個窗長為 0.9532 至 0.9596（原文逐字），皆高於對應 real-only 與兩個表格擴散基線。

**X-Ray.** 這篇的價值不在 AUC 絕對值，而在**它把「多模態」當成 latent 正則化而非特徵工程**：影像模態在 N=7、N=8 時 168 與 192 個特徵已超過 144 像素網格、作者自認是 lossy auxiliary view，卻仍帶來 Precision／Density／Coverage 的優勢——暗示增益來自對齊後的潛空間幾何，而非影像本身裝了多少資訊（此歸因未經消融，屬本頁推斷）。它解的舊坑是 TabSyn 在少數類上的塌縮：N=7 時 Recall 0.9887 配上 Precision 0.0065，是典型「broad coverage 但不在真實流形上」。它打不開的 envelope 很清楚：（a）全篇無交易層評估，作者自述不宣稱 end-to-end portfolio policy；（b）標籤是**同期**窗口的 CVaR 標記，不是未來事件預測，把 AUC 讀成「預測暴跌」即誤讀；（c）樣本期間未披露，無法判斷跨 regime 外推。對量化讀者：這是**資料層 augmentation 元件**，可插在 rare-event 分類器前，不是 alpha 訊號本身。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作（TabDDPM／TabSyn）做法 | 本文做法 | 解決什麼 |
|---|---|---|---|
| 表示層 | 只建模結構化表格視圖 | 三模態同步：tabular（Transformer VAE）＋DeepInsight 影像（conv VAE）＋等權收益序列（biLSTM VAE） | 表格生成器捕捉不到跨特徵／跨資產局部結構與時序動態 |
| 潛空間 | 表格特徵空間或單一 latent score-based diffusion | 各模態等維潛碼 → 影像／序列用訓練折 mean/std 標準化、tabular 額外部分 whitening → 拼接 `z=[z_tab,z_img,z_seq]` | 防止單一模態在拼接表示中支配；給擴散一個條件更好的空間 |
| 條件 | 一般為無條件或類別無關 | CVaR 導出的 rare/normal label embedding 條件化 MLP denoiser（EDM-style continuous schedule） | 生成「目標性」導向少數 regime，而非全域平均分佈 |

> 血緣：作者自述 extends MAGE-ID（intrusion detection 的多模態潛擴散），場景換成金融序列與 CVaR 稀有事態；影像轉換沿用 DeepInsight；rare-window 標籤沿用 Mondal et al. (2025) 的風險感知構造。**注意「diffusion」實作為 MLP denoiser，非 U-Net／DDPM 卷積骨幹**——框架名中的 DDPM 只是方法家族標籤。

### 1.2 ⚡ Eureka 一句話 trick

**先把三個模態各自的潛碼拉平到同一尺度與協方差，再拼接去做條件擴散。** 直覺：三個 VAE 編出來的潛碼量綱與相關結構都不同，直接 concat 等於讓方差大的模態單方面決定擴散軌跡；標準化＋部分白化後，對齊損失 `λ_align(‖z_tab−z_img‖²+‖z_tab−z_seq‖²)` 才有意義，類別條件也才能真正作用在「聯合」流形上，而不是某一模態的邊緣分佈上。

### 1.3 信息流 ASCII 圖

```
window (N 日 x 8 檔, open/close/volume/gap)
   ├─ tabular ──────────► Transformer VAE ──► z_tab ──┐ (標準化 + 部分 whitening)
   ├─ DeepInsight 12x12 影像 ► conv VAE ────► z_img ──┤ 重建 + KL + 對齊 loss
   └─ 等權收益序列 ─────► biLSTM VAE ─────► z_seq ──┘
                                                  │
                                    concat z = [z_tab, z_img, z_seq]
                                                  │
                       Stage 2: MLP denoiser + label embedding(y)
                                 EDM-style 前向加噪 / 反向去噪
                                                  │
                        分離 latent blocks → 反轉標準化 → 各模態解碼
                                                  │
                     取 tabular 合成樣本 → 併入訓練集 → XGBoost 分類器
```

## §2 · 數學層

📌 **Napkin Formula**

Stage 1（模態編碼 + 對齊）：
`L_VAE = Σ_m [ ‖x_m − x̂_m‖² + β·D_KL(q_φm(z_m|x_m) ‖ N(0,I)) ] + λ_align( ‖z_tab − z_img‖² + ‖z_tab − z_seq‖² )`

Stage 2（條件潛擴散，VAE 參數凍結）：
`L_diff = E_{z,ε,t}[ ‖ε_θ(z_t, t, y) − ε‖² ]`

**直覺**：第一式同時要求「各自能重建」與「同一窗口的三視角在潛空間靠近」；第二式只在潛空間學聯合分佈，使生成成本與原始特徵維度（N=8 時 192 維）解耦。**訓練細節（逐字）**：VAE 與 diffusion 兩階段各訓練 2000 epochs；VAE 參數在第二階段凍結；取樣自高斯噪聲起，反向後分離 latent blocks、反轉標準化、逐模態解碼。**未披露**：β、λ_align、潛碼維度、MLP 層寬／層數、EDM 噪聲排程參數、diffusion steps、生成樣本數量。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ 以下全部是**假設／示意**數字，只為讓讀者照算一遍流程；**不是**論文實證結果。論文真實數字只在 §5 出現。

1. **假設**一個 N=5 窗口、僅 2 個特徵（真實為 120 個），標準化後 `x_tab=[1.0, 2.0]`。
2. VAE 重建假設 `x̂_tab=[0.9, 2.1]` → 重建項 `0.1²+0.1²=0.02`。
3. 假設三模態潛碼（維度 2、示意）`z_tab=[0.5,−0.2]`、`z_img=[0.3,−0.1]`、`z_seq=[0.4,−0.15]`。
   對齊項：`0.2²+0.1²=0.05`；`0.1²+0.05²=0.0125`；合計 `0.0625`。取**假設** `λ_align=0.1`（原文未披露）→ 貢獻 `0.00625`。
4. 拼接 `z=[0.5,−0.2,0.3,−0.1,0.4,−0.15]`。假設噪聲刻度 σ=0.5、抽到 `ε=[0.2,−0.4,…]` → `z_t=[0.6,−0.4,…]`（EDM 真實前向式與離散化原文未披露，此處僅示意）。
5. denoiser 給 `ε̂=[0.15,−0.35]` → `‖ε̂−ε‖²=0.05²+0.05²=0.005`；反傳只更新 `ε_θ`，三個 VAE 不動。
6. 取樣：自純高斯跑反向，切回三段、反轉標準化、解碼 → 得到一條**標籤被指定為 rare（y=1）**的合成 tabular 樣本，丟進 XGBoost 訓練集。

照算一遍會看到：**類別條件不是事後貼標，而是寫進生成軌跡**——這正是 §5 中 Synth→Real AUC 偏高卻不能當成「學會預測」的原因。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 市場／標的 | 八檔美國大型科技股（"eight major U.S. technology equities"）；具體 ticker 所讀文本未見 |
| 頻率 | 日頻 |
| 窗口 | N ∈ {5, 6, 7, 8} 日，滾動窗口 |
| 變量 | 每檔 open price、close price、volume；隔夜 gap 定義為 `Gap_t = Open_t − Close_{t−1}` |
| 排序 | 股票按 CAPM beta 排序（beta 數值未披露） |
| 正規化 | min–max，僅用訓練折統計量；DeepInsight 佈局亦只用訓練折學習 |
| 樣本量（逐字） | N=5：normal 5272 / rare 527；N=6：5174 / 624；N=7：5085 / 712；N=8：5005 / 791 |
| 稀有比例 | 9.09%–13.65%（隨 N 上升） |
| 標籤 | `L_t=−R_t`（R_t 為等權收益），`CVaR_α = E[L_t | L_t > VaR_α]`；窗口 CVaR 超過「prescribed risk threshold」即 rare（閾值數值**未披露**） |
| 樣本期間 | **未披露**（所讀文本未見起訖日期） |
| 驗證協議 | 5-fold CV；每 fold 內 normalization、DeepInsight placement、latent transformations、VAE 訓練、diffusion 訓練、合成「只用訓練分割」 |

**容量假設**：作者挑 5–8 日窗的理由是稀有行情多為緊湊多日、更長窗會稀釋局部訊號並減少完整樣本。**本頁推斷**：滾動窗口彼此重疊，5-fold 切分下 fold 間樣本高度自相關，所報 fold std 很可能低估真實不確定性（原文未討論）。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 所讀文本未見（未給本法連結）；僅提及基線用 TabDDPM／TabSyn 的 "official implementations" |
| Checkpoint | 未披露 |
| License | 未披露 |
| 複現難度 | 中高：需自實作 Transformer VAE／conv VAE／biLSTM VAE、DeepInsight 佈局、EDM-style MLP denoiser；β、λ_align、潛碼維度、排程參數全未披露 |
| 數據可得性 | 美股日頻價量公開可得，但 8 檔 ticker 與樣本期間未披露 → 無法逐字復刻切分 |
| 算力實測（A.3，N=5，同一 NVIDIA A100，不含前處理與評估） | TabSyn 訓練 643.01 秒／取樣 5.47 秒；TabDDPM 463.38 秒／14.34 秒；本法 697.87 秒／3.25 秒（三者中訓練最長、取樣最短；成本落在三個編碼器） |

## §5 · 評測 / Benchmark

> 表內僅列 mean；±std 與完整 PRDC（Precision／Recall／Density／Coverage）見原文 Table 1。`Real→Real` 是無增強參考而非基線，故不入表、不計 Δ。**前SOTA 為模板欄名**：來源並未宣稱任何 SOTA，實際競品只有 TabDDPM、TabSyn 兩個表格擴散基線，故逐行拆列、不合併成區間。Δ 定義為「本方法 − 該基線」。AUC 為無量綱分數，Δ 不加 pp。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 美股科技×8, N=5, Synth→Real | AUC | TabDDPM 0.9057 | 0.9263 | +0.0206 |
| 美股科技×8, N=5, Synth→Real | AUC | TabSyn 0.9020 | 0.9263 | +0.0243 |
| 美股科技×8, N=5, Real+Synth→Real | AUC | TabDDPM 0.9349 | 0.9550 | +0.0201 |
| 美股科技×8, N=5, Real+Synth→Real | AUC | TabSyn 0.9383 | 0.9550 | +0.0167 |
| 美股科技×8, N=6, Synth→Real | AUC | TabDDPM 0.9129 | 0.9300 | +0.0171 |
| 美股科技×8, N=6, Synth→Real | AUC | TabSyn 0.8678 | 0.9300 | +0.0622 |
| 美股科技×8, N=6, Real+Synth→Real | AUC | TabDDPM 0.9451 | 0.9596 | +0.0145 |
| 美股科技×8, N=6, Real+Synth→Real | AUC | TabSyn 0.9430 | 0.9596 | +0.0166 |
| 美股科技×8, N=7, Synth→Real | AUC | TabDDPM 0.9101 | 0.9151 | +0.0050 |
| 美股科技×8, N=7, Synth→Real | AUC | TabSyn 0.8603 | 0.9151 | +0.0548 |
| 美股科技×8, N=7, Real+Synth→Real | AUC | TabDDPM 0.9445 | 0.9532 | +0.0087 |
| 美股科技×8, N=7, Real+Synth→Real | AUC | TabSyn 0.9425 | 0.9532 | +0.0107 |
| 美股科技×8, N=8, Synth→Real | AUC | TabDDPM 0.8913 | 0.9048 | +0.0135 |
| 美股科技×8, N=8, Synth→Real | AUC | TabSyn 0.8151 | 0.9048 | +0.0897 |
| 美股科技×8, N=8, Real+Synth→Real | AUC | TabDDPM 0.9372 | 0.9558 | +0.0186 |
| 美股科技×8, N=8, Real+Synth→Real | AUC | TabSyn 0.9374 | 0.9558 | +0.0184 |

**哪部分是真 capability**：只有 `Real+Synth→Real` 有資訊價值——它測「生成少數類樣本併進真實訓練集」是否帶來額外判別力，且它是唯一與 real-only 參考可對照的欄位（原文舉例 N=6：0.9342→0.9596；N=8：0.9236→0.9558，逐字）。對 TabSyn 的差距要打折看，因為 TabSyn 自身不穩：其 Synth→Real 在 N=6／7／8 分別 0.8678、0.8603、0.8151，std 達 0.0674、0.0737、0.1001（逐字）。**哪部分近同義反覆**：`Synth→Real` 高分很大程度上是「條件生成已把標籤 y 寫進樣本」的必然結果，AUC 0.9263 反映生成器學到類別流形，**不等於**學到可外推的風險訊號。

**哪一格證據最弱**：N=7 的 `Real+Synth→Real`，TabDDPM 0.9445±0.0085 vs 本法 0.9532±0.0034（逐字），差距與兩側 fold std 同量級；N=7 的 `Synth→Real` 更是 0.9101±0.0126 vs 0.9151±0.0057，幾乎重疊。相對於 N=5／6／8，這一窗的優勢應視為不確立。

**未計成本**：全篇無交易成本、無倉位、無停損，AUC 差異不可換算為 PnL；作者 §1 自述不宣稱評估 end-to-end portfolio policy。**洩漏風險**：每 fold 的 normalization、DeepInsight 佈局、latent 轉換、VAE、diffusion、合成都限定訓練分割（§3 明示），此點乾淨；但滾動窗口重疊導致的 fold 間相關未處理（本頁推斷），且**樣本期間未披露**，無法判斷是否集中於單一科技股 regime。

**保真度對照（逐字，未合成區間）**：N=7 時 TabSyn Recall 0.9887±0.0050 但 Precision 0.0065±0.0045、Density 0.0024±0.0017、Coverage 0.0046±0.0028；本法 Precision 0.3892±0.0366、Recall 0.5545±0.0233、Density 0.3452±0.0513、Coverage 0.3327±0.0209。N=8 同型態（TabSyn Precision 0.0034、本法 0.3301）。反之 N=5、N=6 的 Density 是 TabDDPM 較高（N=5：0.3154 vs 0.2373；N=6：0.1679 vs 0.1289），本法在 Recall／Coverage 勝出——「best overall balance」是作者定性判斷，原文未給聚合指標，**不可**當成每格全勝。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **不做投資組合評估**：§1 貢獻段 "The study focuses on predictive decision support; it does not claim to evaluate an end-to-end portfolio policy."
- **任務是同期分類，非未來預測**：§2 "the downstream task is contemporaneous rare-window classification, not future-event forecasting"；標籤描述與輸入模態**同一個 N 日窗**。
- **市場覆蓋窄、模態貢獻未拆**：§5 "The current evaluation focuses on U.S. technology equities; future work will consider broader markets and further analyze the contribution of each modality." → 模態消融未做。
- **影像模態有損**：§2 自承 N=7、N=8 的 168 與 192 個特徵超過 144 像素網格，碰撞與平均不可避免，"the image is treated as a lossy auxiliary view"。
- **CVaR 閾值未給值**：僅 "exceeds the prescribed risk threshold"，數值未披露。
- **Appendix A 範圍**：A.1 類別分佈（Table 2）、A.2 影像示例（Figures 2–3，未提取、待核）、A.3 執行時間（N=5）；正文所稱 "additional class counts" 對應 A.1。

### 6.2 推斷的隱含假設（本頁推斷，非作者陳述）

1. **潛空間平穩性**：三 VAE 與 denoiser 假設聯合潛分佈在訓練期內近似穩定；若 regime 漂移，罕見窗潛碼落在先驗之外，條件生成會放大錯誤流形。
2. **標籤可條件化 ⇒ 標籤在潛空間可分**：若 CVaR 標籤與特徵近獨立，條件擴散只生成「被貼上 rare 標籤的 normal 樣本」。此假設未被單獨檢驗。
3. **窗口重疊不影響 CV**：滾動窗口造成 fold 間相關，i.i.d. 假設偏強；跨 regime 外推未知。
4. **增益歸因於對齊而非輔助模態本身**：作者把優勢歸於 modality-specific encoders 與 latent normalization（§4），但無消融；移除影像或序列後增益是否仍在，**未驗證**。此為歸因未隔離，不是生存者偏差問題。
5. **容量／成本**：生成樣本數、合成比例、擴增倍數**未披露**（A.3 僅給秒數），故無法判斷增益來自樣本品質或樣本量。
6. **無交易摩擦建模**：即使分類有效，日頻 5–8 日窗的 rare 訊號在被交易後是否存續，無任何成本假設可依。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| TabDDPM | 單模態表格 diffusion；本法多模態潛空間 + 類別條件 | 基線採官方實作；本法 repo 未披露 | 四窗 `Real+Synth→Real` 均高於 TabDDPM（差距見 §5，N=7 最弱） |
| TabSyn | latent score-based；本法 per-modality VAE + EDM-style MLP denoiser | 官方實作 | TabSyn 在 N=7／8 出現 Precision／Density／Coverage 近零、Recall 近 1 的塌縮型態 |
| MAGE-ID（血緣） | 同框架、場景為 intrusion detection；本法換成金融序列 + CVaR 標籤 | 會議論文 | 方法論沿用，屬 lineage 非對手 |
| Real→Real（無增強參考） | 非生成器，是下界參照 | — | 四窗 `Real+Synth→Real` 皆高於 real-only（如 N=6：0.9342→0.9596；N=8：0.9236→0.9558） |

🎤 **Interview Tip**
- **正確答**：「這是資料層 rare-event augmentation。核心貢獻是把三模態潛碼做尺度／協方差對齊再拼接，讓類別條件擴散作用在聯合流形上。可轉移證據只有 `Real+Synth→Real`；`Synth→Real` 高分主要是條件生成把標籤寫進樣本的必然結果，不能當預測力。」
- **錯答**：「它用多模態 diffusion 預測科技股暴跌、AUC 最高 0.9596（Real+Synth→Real，N=6）。」——錯三處：任務是**同期**窗口分類不是預測；樣本是 8 檔科技股單一市場；0.9596 是 `Real+Synth→Real` 的 XGBoost AUC，不是策略績效，且全篇無交易成本。

### 7.1 可證偽預測（帶日期）

- 若 2027-06-30 前出現後續版本（或 workshop 相機版）並補上模態消融，本頁預測：**移除 seq 模態對 N=7／N=8 的 Recall／Coverage 傷害大於移除 img 模態**（影像在該兩窗已超格、作者自認 lossy）。若消融顯示 img 貢獻更大，則「增益來自對齊而非影像資訊量」的推論被證偽。
- 若 2027-06-30 前公開樣本期間與 ticker 清單，本頁預測：rare 樣本會集中於少數高波動區段，折外 AUC 的 fold 間變異將高於原文所報 std。若實測 fold std 與原文同量級，則「重疊窗口低估不確定性」被證偽。
- 若任一後續工作把本法接到含交易成本的日頻回測而訊號仍存續，本頁「AUC 增益不可換算 PnL」的保留態度須下調；在該證據出現前維持保留。

## §8 · For the Reader

- **因子研究員**：當 rare-event 樣本生成器用，不是 alpha 模型。先覆核一件事：換成自己的標籤（不限 CVaR）後，是否只在 `Synth→Real` 好看而 `Real+Synth→Real` 無增益——若是，生成器只是在複述標籤。
- **高頻執行**：5–8 日窗、日頻、無成本假設，與執行層幾無交集。真要接，先問合成樣本會不會誘導分類器在少數 regime 過度交易，以及未披露的合成比例。
- **組合配置**：CVaR 標籤＋稀有事態合成，適合作壓力情境庫（stress scenario library）的原料；但 CVaR 是**同期**窗統計量，不能當前瞻預警，需自行接上部位與損失映射。
- **LLM-agent**：可作 agent 的「合成情境供應器」——需要罕見市況樣本時用類別條件補齊。標籤由外部 CVaR 規則決定，agent 不可自行改寫 y。
- **RL 策略**：這是 supervised 分類層的 augmentation，不是 transition model。可設想把它生成的 rare 窗當環境初始狀態分佈的稀有分支，但原文完全未做，屬**未驗證**延伸。
- **研究學生**：三個可做的缺口——(1) 模態消融（作者明列 future work）；(2) 樣本期間／ticker 揭露後的 regime 穩健性；(3) β、λ_align、潛碼維度、EDM 排程的敏感度分析。也別把 Table 1 的 bold 當全格最優（N=5／6 的 Density、N=7／8 的 Recall 都不屬本法）。

## References

- Arab Loodaricheh, Raja, SenGupta. *Scenario Augmentation for Rare Financial Events via Multimodal Latent Diffusion.* arXiv:2611.07282v1 [cs.LG], 05 Nov 2026. https://arxiv.org/abs/2611.07282v1（`\workshoptitle`：Second Workshop on ML × OR）
- Kotelnikov et al. *TabDDPM: Modelling Tabular Data with Diffusion Models.* ICML 2023, PMLR 202:17564–17579.（基線）
- Zhang et al. *Mixed-Type Tabular Data Synthesis with Score-based Diffusion in Latent Space (TabSyn).* ICLR 2024.（基線）
- Loodaricheh, Manshaei, Raja. *MAGE-ID: A Multimodal Generative Framework for Intrusion Detection Systems.* ICNC 2026, 585–589.（方法血緣）
- Sharma et al. *DeepInsight.* Scientific Reports 9(1):11399, 2019.（影像模態轉換）
- Mondal, Chandak, Chakrabarty, SenGupta. *From Pixels to Profits: A Novel Approach to Identify Rare Events for a Group of US Equities.* Annals of Data Science, 2025.（rare-window 標籤構造）
- Rockafellar & Uryasev. *Optimization of Conditional Value-at-Risk.* Journal of Risk 2(3):21–41, 2000.（CVaR 定義）
- Kingma & Welling. *Auto-Encoding Variational Bayes.* ICLR 2014；Ho et al. *DDPM.* NeurIPS 2020；Naeem et al. *PRDC metrics.* ICML 2020, PMLR 119:7176–7185.
- 註：正文提到 "EDM-style schedule"，但所讀參考文獻列表未見對應 Karras et al. 條目，**待核**。