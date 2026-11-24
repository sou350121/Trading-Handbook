<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=人机协同可解释 -->

# GARCH-LSTM + forward bootstrap 解構（GARCH-LSTM + forward bootstrap）

> **來源發布**：2026-11-09 · （無 venue） · arXiv [2611.13212](https://arxiv.org/abs/2611.13212)
> **arXiv 原文**：[Prediction Inference for Hybrid Models on Realized Volatility Forecasting](https://arxiv.org/abs/2611.13212v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 日頻 × 監督回歸 × 風險擇時 × 人機協同可解釋」。解的 prior gap 不是預測精度，而是 **GARCH-LSTM 文獻幾乎只做點估計、不給不確定性**——本文補上 multi-step prediction interval 與 coverage 診斷，屬「預測推論（prediction inference）」而非 alpha 生成。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把 Kim & Won (2018) 的 GARCH-LSTM 混合模型均值部換成 ARMA 並與 LSTM 做聯合 maximum likelihood，同時用 forward bootstrap 為 realized volatility 建多步預測區間。②核心 trick 是 bootstrap 遞迴把偽序列餵回資訊集，再用 empirical residual 分佈 $\widehat{F}_\eta$ 模擬厚尾、用 conformal score 對區間寬度做一次性 post-hoc 膨脹（不需重訓）。③對「風險擇時」軸★的意義是：波動率預測終於從點估計變成帶 coverage 標籤的區間，可直接餵給部位 sizing／風控門檻。④實證 4 個美股指數 × 16 scenarios：$\mid \mathrm{Cov}-0.95 \mid$ 最好者為 M-H-GARCH + conformal + $\widehat{F}_\eta$ 的 0.025，基線 GARCH(1,1) 為 0.065（名義 0.95，Mean-coverage 0.925 vs 0.885）。

**X-Ray.** 這篇的價值不在「更準」，而在**把一個只會被 RMSE 排名的模型族，接上可被檢驗的推論層**。舊工程坑很具體：GARCH-LSTM 的輸出是 $\hat\sigma_t$，但下游風控要的是 $RV_{T,H}$ 的區間；過去只能假設常態或用解析式，遇到 LSTM 這種黑箱就斷鏈。本文用 forward bootstrap 以固定的 $\hat G(\hat\Theta,\cdot)$ 遞迴生成未來路徑，納入殘差厚尾（$\widehat F_\eta$），再用 conformal 補 finite-sample 欠覆蓋；參數估計變異並未被重抽，§3.3 與 §6 自述未被完全捕捉，PPI 才需重訓，這個組合在五軸 Pareto 上是**低成本、可解釋、不改變原模型結構**的一條路。但它打不開的 envelope 同樣清楚：（a）$\widehat{F}_\eta$ 的尾部來自 2020–2025 這段含 COVID crash 的樣本，對「同級衝擊再現」的樂觀偏誤無法自我診斷；（b）conformal extension 作者自承不滿足 exchangeability、**沒有邊際覆蓋保證**（§3.3），所以 0.925 不是保證值而是樣本內觀察；（c）全程無交易成本、無執行摩擦，16 scenarios 背後只有 4 個高度共動的美股指數，不能當跨市場證據。對量化讀者：把它當「不確定性量化樣板 + 一個便宜的事後校正技巧」拿來用，比當成新預測模型更有價值。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | Kim & Won (2018) / M-H-GARCH | 本文 ARMA-H-GARCH |
|---|---|---|
| 條件均值 $\mu_t$ | rolling mean $\hat\mu_t=\frac{1}{p}\sum_{i=1}^{p}Y_{t-i}$，$p$ 由 cross-validation 選 | ARMA(L,K)，每 replication 以 AIC 在 $L,K\le 5$ 選階 |
| 參數估計 | 均值與變異數分離（sample mean + LSTM） | 聯合 negative log-likelihood 式 (7)，LSTM 權重與 $\Pi$ 同時梯度下降 |
| 不確定性 | Kim & Won 原版僅點預測；本文 M-H-GARCH 則有 Algorithm 1 的 forward bootstrap PI | forward bootstrap $M=2000$ 抽樣 + $\widehat{F}_\eta$ + conformal extension |
| 訓練 loss | MSE 式 (3)，以 $\sqrt{\pi/2}\, \mid \hat\varepsilon_t \mid$ 作 $\sigma_t$ 的無偏代理 | 同 (3) 用於 M-H-GARCH；ARMA-H-GARCH 改用式 (7) |
| 評估 | RMSE、MAPE | 追加 Coverage、Length、$\mid \mathrm{Cov}-0.95 \mid$、Min-coverage |

第三項再拆兩層：**(i) 重抽樣分佈**（$\eta^*\sim N(0,1)$ 或 $\widehat{F}_\eta$）；**(ii) 區間寬度校正**（有/無 conformal）。原文合計比較 11 種 prediction interval。

### 1.2 ⚡ Eureka

> **把 LSTM 當成 $\sigma_t$ 的生成器而不是預測器：bootstrap 遞迴地把偽觀測餵回 $\mathcal{F}_{T+h}$，於是 $RV_{T,H}$ 的整條分佈被「模擬出來」而非「推導出來」；conformal 只負責在最後把寬度乘上一個由歷史失誤學來的膨脹因子，完全避開 PPI 那種「每個 bootstrap 樣本都要重訓」的成本。**

直覺：點預測只需要 $\mathbb{E}[\cdot]$，區間需要的是 predictive root 的分佈。LSTM 讓解析式不可得，但**模擬**不需要解析式——只要你能遞迴地生成未來路徑。欠覆蓋的來源是「估計變異沒被算進去」，conformal score 就是量測「這條區間在歷史上漏掉多少」的單一標量。

### 1.3 信息流

```
Y_{1:T}
  ├─► GARCH(1,1) / EGARCH(1,1) / EWMA 擬合 ──► Θ_t (8 維參數塊)
  │                                                  │
  │                                                  ▼
  │                                     LSTM(32 hidden, seq=5)
  │                                                  │
  │                                                  ▼
  │                                            σ̂_{T+h}
  │                                                  │
  └─► ARMA(L,K) 或 rolling mean ──► μ̂_{T+h} ─────────┤
                                                     ▼
                            Y*_{T+h} = μ̂_{T+h} + σ̂_{T+h}·η*_{T+h}
                              (η* ~ N(0,1) 或 F̂_η)
                                                     │
                                   遞迴 h = 1..H，重複 M = 2000 次
                                                     ▼
                              {RV*^{(m)}_{T,H}} ──► F̂_RV
                                                     │
                        ┌────────────────────────────┴──────────────┐
                        ▼                                           ▼
             點預測 = mean(F̂_RV)                    95% PI = [q̂_{α/2}, q̂_{1−α/2}]
                                                                    │
                                                      conformal score S_j (式 10)
                                                                    ▼
                                          [L_T − q̂_{S,1−α}W_T, U_T + q̂_{S,1−α}W_T]
```

## §2 · 數學層

📌 **Napkin Formula**

$$\text{MSE}=\frac{1}{T}\sum_{t=1}^{T}\Big(\hat\sigma_t-\sqrt{\pi/2}\cdot|\hat\varepsilon_t|\Big)^2 \tag{3}$$

$$\mathcal{L}(\Pi,\Theta)=\frac{1}{2N}\sum_{t=t_0}^{T}\left[\log\sigma_t^2(\Pi,\Theta)+\frac{\varepsilon_t(\Pi)^2}{\sigma_t^2(\Pi,\Theta)}\right] \tag{7}$$

$$\Big[L_T-\widehat{q}_{S,1-\alpha}W_T,\; U_T+\widehat{q}_{S,1-\alpha}W_T\Big],\quad W_T:=U_T-L_T \tag{11}$$

**直覺**：式 (3) 是「用 $\sqrt{\pi/2} \mid \hat\varepsilon_t \mid$ 這個對 $\sigma_t$ 無偏的代理，去監督一個從未被觀測過的變異數」。式 (7) 是標準 Gaussian quasi-likelihood——它讓 ARMA 的 $\Pi$ 與 LSTM 的 $\Theta$ 在同一個梯度圖裡更新，這是與 M-H-GARCH 最本質的分野。式 (11) 因為 conformal score 恆非負，區間只會變寬不會變窄。

**Loss / 訓練細節**（Appendix A）：單層 LSTM、32 hidden、sequence length 5、Adam、learning rate $10^{-3}$；full-batch；early stopping 以最後 20% 訓練樣本為驗證窗，連續 100 個 epoch 未改善即停，上限 1200 epochs（ARMA 聯合階段上限 900）；ARMA 兩階段（先 300 epochs 常數變異數 CMLE 初始化，再聯合）；AR 與 MA 多項式以 $\tanh$ 重參數化並經 Durbin–Levinson 限制在 $(-0.98,0.98)$；輸出 $\hat\sigma_t=s\cdot\mathrm{softplus}(\cdot)$ 保證正值；變異數序列截斷於 $[10^{-12},1]$。

**複雜度**：原文未明示每個 rolling window 是否重訓；每次 bootstrap 需 $M\times H$ 次前向；conformal extension 另需 $J=T-H-\varpi+1$ 次遞迴 PI，每次再 $M\times H$ 前向 → 總前向數 $O(J\cdot M\cdot H)$。wall-clock 未披露。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ **以下全部是假設/示意數字，用來手算機制，不是論文實證結果。論文真實數字見 §5。**

設 $T$ 為預測原點，$H=2$，用 M-H-GARCH，$p=1$，$\alpha=0.05$。

1. **第一步變異數**：假設 LSTM 給 $\hat\sigma_{T+1}=0.0100$；抽到 $\eta^*_{T+1}=+1.50$（示意）→ $\varepsilon^*_{T+1}=0.0100\times1.50=0.0150$。
2. **第一步均值**：$p=1$ 的 rolling mean 假設 $\hat\mu_{T+1}=0.0002$（示意）→ $Y^*_{T+1}=0.0002+0.0150=0.0152$。
3. **餵回資訊集**：把 $Y^*_{T+1}$ 當成真觀測，重算 $\mathcal{F}_{T+1}$；假設 LSTM 給 $\hat\sigma_{T+2}=0.0120$，抽到 $\eta^*_{T+2}=-0.80$ → $\varepsilon^*_{T+2}=-0.0096$；$\hat\mu_{T+2}=Y^*_{T+1}=0.0152$ → $Y^*_{T+2}=0.0152-0.0096=0.0056$。
4. **組 RV**：$\overline{Y}_{T,2}=(0.0152+0.0056)/2=0.0104$；$RV^*_{T,2}=\sqrt{\tfrac{1}{2}\big[(0.0152-0.0104)^2+(0.0056-0.0104)^2\big]}\approx0.0048$（示意）。
5. **重複 M 次**得 $\{RV^{*(m)}_{T,2}\}_{m=1}^{2000}$，取 $\alpha/2$ 與 $1-\alpha/2$ 分位當 $[L_T,U_T]$；點預測取 mean。
6. **Conformal**：歷史上某次遞迴區間 $[L^r,U^r]$ 漏掉真值 $RV$，則 $S=\max\{L^r-RV,\;RV-U^r,\;0\}/(U^r-L^r)$；取 $1-\alpha$ 分位 $\widehat{q}_{S,1-\alpha}$，最終區間為 $[L_T-\widehat{q}_{S,1-\alpha}W_T,\;U_T+\widehat{q}_{S,1-\alpha}W_T]$，再從下方截斷於 0（因 $RV_{T,H}\ge 0$）。

關鍵觀察：步驟 3 的「假觀測餵回」是 bootstrap 能產生**多步**分佈的唯一來源；步驟 6 的膨脹因子是**全域標量**，不隨 $h$ 改變——這是它便宜的原因，也是它粗糙的原因。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 模擬 DGP | Model-1：GARCH(1,1) + Student-$t$(5) + ARMA(1,1)；Model-2：時變係數 $\alpha_{1,t}=0.1-0.05V_t$、$\beta_{1,t}=0.55+0.25V_t$，$V_t=t/T$ |
| 模擬規模 | $T\in\{250,500\}$、$H\in\{5,20\}$、$M=2000$、$R=500$、$\alpha=0.05$、burn-in 500 |
| 實證市場 | S&P 500、NASDAQ Composite、NYSE Composite、Dow Jones Industrial Average |
| 時段 | 2020-01-02 至 2025-12-31（含 2020-03 的 COVID 崩跌、2022 下跌、2025-04 關稅波動） |
| 樣本量 | 1508 closing prices → 1507 log-returns；四指數同日曆 |
| 滾動窗 | 250（1-year）與 500（2-year）；$T=250$ 得 1238 次預測、$T=500$ 得 988 次 |
| 情境數 | 4 指數 × 2 窗長 × 2 horizon = 16 scenarios |
| 來源 | Yahoo Finance（公開） |
| 目標量 | $RV_{T,H}=\sqrt{\frac{1}{H}\sum_{h=1}^{H}(Y_{T+h}-\overline{Y}_{T,H})^2}$，式 (8) |

**樣本外與容量假設**：所有評估皆為 rolling-window 偽樣本外，原文未明示每個窗口重訓（僅稱 rolling-window predictions）；conformal score 則用**全樣本**訓練的模型做遞迴預測產生（§3.3 自述，這是覆蓋率估計偏樂觀的來源之一，見 §6）。無交易成本、無容量約束、無借券/滑價建模——這是純預測評估，不是策略回測。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD（所讀文本未見連結） |
| Checkpoint | TBD（所讀文本未見） |
| License | CC BY 4.0（arXiv 頁面標示） |
| 複現難度 | 中高：$M=2000$ bootstrap × $R=500$ replications × 11 種 PI；原文自述模擬在 OSG Consortium 上完成 |
| 隨機性控制 | `BASE_SEED=20261023`，`np.random.SeedSequence([20261023, model, T, rep])` |
| 數據可得性 | Yahoo Finance 公開；四指數代碼原文未逐字列出（TBD） |
| 關鍵超參 | LSTM 32 hidden / seq 5 / Adam $10^{-3}$ / early stopping 100 epochs / 上限 1200（聯合階段 900） |

## §5 · 評測 / Benchmark

> 下表「前SOTA」欄依原文所給的**個別基線**逐行列出（GARCH(1,1)、EGARCH(1,1)、naive LSTM），**未合成任何 SOTA 區間**；$\Delta$ 一律為「本方法 − 該列基線」。點預測列的本方法為 ARMA-H-GARCH；區間列的本方法另行標註。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 模擬 Model-1, $T=250$, $H=5$ | RMSE (×10^-3)；基線 GARCH(1,1) | 4.78 | 3.79 | −0.99 |
| 模擬 Model-1, $T=250$, $H=5$ | RMSE (×10^-3)；基線 EGARCH(1,1) | 4.44 | 3.79 | −0.65 |
| 模擬 Model-1, $T=250$, $H=5$ | RMSE (×10^-3)；基線 naive LSTM | 4.39 | 3.79 | −0.60 |
| 模擬 Model-1, $T=250$, $H=5$ | MAPE (%)；基線 GARCH(1,1) | 90.83 | 64.15 | −26.68 |
| 模擬 Model-2, $T=250$, $H=5$ | RMSE (×10^-3)；基線 GARCH(1,1) | 3.73 | 3.51 | −0.22 |
| 模擬 Model-1, $T=500$, $H=20$ | RMSE (×10^-3)；基線 GARCH(1,1) | 3.39 | 3.29 | −0.10 |
| 模擬 Model-2, $T=500$, $H=20$ | RMSE (×10^-3)；基線 GARCH(1,1) | 4.75 | 4.87 | +0.12 |
| 實證 4 美股指數 × 16 scenarios | Rel-RMSE；基線 GARCH(1,1) | 1.000 | 0.952 | −0.048 |
| 實證 4 美股指數 × 16 scenarios | Rel-MDAE；基線 EGARCH(1,1) | 0.942 | 0.942 | 0.000 |
| 實證 4 美股指數 × 16 scenarios | Mean-coverage (95% PI)；基線 GARCH(1,1)；本方法 M-H-GARCH + conformal + $\widehat{F}_\eta$ | 0.885 | 0.925 | +0.040 |
| 實證 4 美股指數 × 16 scenarios | $\lvert\mathrm{Cov}-0.95\rvert$；基線 GARCH(1,1)；本方法 M-H-GARCH + conformal + $\widehat{F}_\eta$ | 0.065 | 0.025 | −0.040 |
| 實證 4 美股指數 × 16 scenarios | Rel-length；基線 GARCH(1,1)；本方法 M-H-GARCH + conformal + $\widehat{F}_\eta$ | 1.000 | 1.437 | +0.437 |
| 實證 4 美股指數 × 16 scenarios | Mean-coverage (95% PI)；基線 GARCH(1,1)；本方法 ARMA-H-GARCH + conformal + $\widehat{F}_\eta$ | 0.885 | 0.886 | +0.001 |

**哪些是真 capability**：
- **均值部換 ARMA + 聯合 MLE 在 point 面有真實但邊際的增益**。$T=250$、$H=5$ 的 Model-1 是最好看的一格（3.79 對 4.78 / 4.44 / 4.39），但同設定的 MAPE 仍高達 64.15；在 Table 3 的 $T=500$、$H=20$ 設定，ARMA-H-GARCH 的 RMSE 為 3.29，GARCH(1,1) 為 3.39。
- **原文 §4.1 的「ARMA-H-GARCH dominates other methods」措辭與 Table 4 不一致**：Model-2、$T=500$、$H=20$ 下 ARMA-H-GARCH 為 4.87，劣於 GARCH(1,1) 的 4.75 與 EGARCH(1,1) 的 4.81；Model-2、$T=250$、$H=20$ 亦為 3.67 對 EGARCH(1,1) 的 3.59。**這是本頁核對表格後的判斷：優勢是 regime 與 horizon 依賴的，不是普遍的。**
- **point 與 interval 的最佳模型不是同一個**，這是最有價值的分離結果：實證 Rel-RMSE 最佳是 ARMA-H-GARCH（0.952），但其 Mean-coverage 僅 0.886、$\lvert\mathrm{Cov}-0.95\rvert$ 0.064；區間最佳是 M-H-GARCH + conformal + $\widehat{F}_\eta$（0.925 / 0.025），其 Rel-RMSE 為 1.111（比 GARCH(1,1) 差）。原文 §5 結尾也直接建議「點預測用 ARMA-H-GARCH、區間用 M-H-GARCH+conformal+$\widehat{F}_\eta$」——**這等於承認兩者不可兼得**。

**哪些可能是過擬合 / 樂觀偏誤 / 未計成本**：
- **$p$ 的 oracle 選擇（本頁推斷）**：§4 明言模擬中「report the optimal $p$ according to the root mean squared error」，這是用真值挑超參，實務不可得，會系統性高估 M-H-GARCH 的點預測。
- **conformal score 的可交換性被破壞（作者自述 §3.3）**：遞迴 RV 重疊、模型用全樣本訓練 → 0.925 的覆蓋率**不是**邊際保證，樂觀偏誤方向未量化。
- **$\widehat{F}_\eta$ 的尾部來自含 COVID 的 2020–2025 樣本（本頁推斷）**：對「更極端衝擊」的外推能力未測。
- **成本未計**：全文無交易成本、無部位、無執行假設，因此這些數字**不能**轉譯成 Sharpe/IR 或可交易 alpha；本頁拒絕做此換算。
- **16 scenarios 不是 16 個獨立市場**：4 個美股指數同漲同跌，有效樣本遠小於名義情境數。
- **Rel-length 的代價不可忽略**：最佳覆蓋的設定 Rel-length 為 1.437（GARCH(1,1) 的 Rel-length 為 1.000），代價是區間相對更寬；naive LSTM 的 Rel-length 1.784 最寬、Mean-coverage 卻只有 0.856——**寬不等於覆蓋好**。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

| 出處 | 作者自述限制 |
|---|---|
| §3.3 | conformal extension **不具**標準 conformal 的 marginal coverage guarantee：conformal scores $\{S_\varpi,\ldots,S_{T-H}\}$ 因 RV 窗重疊而相關、且模型以全樣本訓練，**exchangeability 不成立** |
| §3.3 | 較嚴格做法（每 $H$ 個時間索引取一個 score、或隨擴張窗重訓）會「unavoidably increases the computational burden」，本文未採用 |
| §3.3 | asymptotics 僅陳述為「possible to show」，**理論分析明確 postpone to future work** |
| §6 | 「our prediction interval still undercovers future volatility even after being widened by conformal prediction」——估計變異未被完全捕捉 |
| §6 | PPI（pertinent prediction interval）較保守但計算重，本文未實作；hybrid resampling 留作 future work |
| §6 | 未處理高頻數據、未接 realized GARCH / HEAVY 類模型 |

### 6.2 本頁推斷的隱含假設（明標推斷）

1. **[推斷] 條件分佈在滾動窗內近似平穩**：$M=2000$ 次 bootstrap 抽樣把 $\widehat{\Theta}$ 視為固定，模型參數估計變異只透過 conformal 標量近似吸收。若 regime 切換快於窗長（250/500 日），此假設失效。
2. **[推斷] $\widehat{F}_\eta$ 的經驗分佈可代表未來殘差尾**：2020–2025 含 COVID 與 2025-04 關稅波動，是「尾部豐富」的樣本；但同時也意味著樣本外若出現同級事件，覆蓋率表現可能是內插而非外推。**這是雙面刃，不是單純優點。**
3. **[推斷] 超參選擇與評估同源**：$p$、$(L,K)$ 在模擬中以 RMSE/AIC 選；實證 rolling 的選參細節（是否每窗重選、用哪段資料）所讀文本未見完整描述，**待 Appendix 全文與代碼核**。
4. **[推斷] 無交易成本與容量假設**：$RV$ 預測誤差與可實現的風險調整績效之間沒有給出映射；不可把 coverage 改善直接讀成投資績效改善。此為**適用條件限制**，不是對所有波動率策略的否定。
5. **[推斷] 市場代表性**：四個美股指數同期間，不能外推至新興市場、商品、加密或單一個股；跨市場有效性**未測試 ≠ 已證明無效**。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Kim & Won (2018) GARCH-LSTM | 均值為 rolling mean、僅點預測；本文換 ARMA + 聯合 MLE + PI | TBD | 本文主要 baseline |
| Hajizadeh et al. (2012) GARCH-NN | 兩步法（GARCH 輸出餵 NN），無不確定性量化 | TBD | 本文引用為 lineage |
| GARCH(1,1) / EGARCH(1,1) | 純計量、參數少；PI 用同一 forward bootstrap 建 | 公開（`arch` 套件） | 實證 benchmark（Rel 值 = 1.000 / 0.956 等） |
| naive LSTM | 只吃報酬序列、不吃 $\Theta$；本文證明其在部分情境被經典模型擊敗 | TBD | 消融對照 |
| Politis (2015) PPI | 以 bootstrap 逼近 predictive root，捕捉估計變異；但需對每個 bootstrap 樣本重訓 | 論文公開 | 本文因成本未採用 |
| Vovk et al. (2005) 標準 conformal | 需 exchangeability 才保證 marginal coverage；本文僅借用「分數 → 膨脹寬度」的想法 | 公開 | 本文為 heuristic extension，無保證 |

🎤 **Interview Tip**
- **正確答**：「這篇的貢獻分兩層且不可混談——point 層是 ARMA 均值 + 聯合 MLE，uncertainty 層是 forward bootstrap + $\widehat{F}_\eta$ + conformal。前者增益邊際且在 Model-2 的 $H=20$ 會輸給 EGARCH/GARCH；後者讓 $\mid \mathrm{Cov}-0.95 \mid$ 從 0.065 降到 0.025，但作者自承 exchangeability 不成立、無邊際覆蓋保證。而且點預測最佳與區間最佳不是同一個模型。」
- **錯答**：「它提出新波動率模型，Sharpe 提升 X%／勝過 SOTA。」——原文沒有交易策略、沒有 Sharpe、沒有合成 SOTA，這種答法會立刻暴露只讀了 abstract。

### 7.1 可證偽預測（帶日期）

1. **2027-06 前**：若作者釋出代碼（Repo 目前 TBD），以「每 $H$ 索引取一個 conformal score + 擴張窗重訓」重跑，實證 Mean-coverage 應**上升**且 Rel-length 進一步變寬——若反而下降，則「score 相關性」不是欠覆蓋主因。（本頁推斷）
2. **2027-06 前**：將同一 pipeline 套到非美股市場（如新興市場指數或商品），若 Mean-coverage 顯著低於 0.925，則本文區間的可攜性被否證。（本頁推斷）
3. **任何時候**：若把 $\widehat{F}_\eta$ 的擬合窗換成不含 2020-03 的樣本，coverage 應下降——用以檢驗「尾部資訊來自 COVID 樣本」的假設。（本頁推斷）
4. **檢核點**：Appendix A 的 oracle $p$ 選擇若改為真正的 rolling CV，M-H-GARCH 的模擬 RMSE 優勢應縮小或消失。（本頁推斷）

## §8 · For the Reader

- **因子研究員**：把 §2 的式 (3) 當「潛在變異數的代理監督」樣板抄走——$\sqrt{\pi/2} \mid \hat\varepsilon_t \mid$ 這個無偏代理可用於任何「標籤不可觀測」的迴歸（如 future vol、future beta）。但別把 ARMA-H-GARCH 當萬用贏家，先在自己的 DGP 上驗 $H=20$ 那兩格。
- **高頻執行**：這篇的 horizon 是 5 日與 20 日，資料是日頻收盤，**與 intraday 執行無關**；若你要的是盤中 risk gate，這套 bootstrap 遞迴的 $O(J\cdot M\cdot H)$ 成本需要重估（wall-clock 未披露）。
- **組合配置**：真正可用的是「區間 → 部位上限」這條鏈：用 $U_T$（而非點預測）做 vol targeting 的分母，是本文最直接的落地方式。但注意 Rel-length 1.437 意味著你會被要求更保守的槓桿。
- **LLM-agent / RL 策略**：conformal extension 的「post-hoc 寬度膨脹」是可移植到任何 agent 輸出區間的通用 pattern——前提是你接受它只是 heuristic。RL 的 value interval 同樣面臨 exchangeability 問題。
- **研究學生**：這是一篇很好的「如何把 prediction inference 接到 ML 模型」範本；重點讀 §3.3 的 Remark 3.2（為何 PI (9) 必然欠覆蓋）與 Algorithm 3 的第 6 步（score 的正規化）。**最大的寫作教訓**：§4.1 說 "dominates other methods"（摘要僅稱 "lowest point-forecast errors in most settings"），表格卻有反例——永遠自己核表。

## References

- Parrella, K. & Wu, K. (2026). *Prediction Inference for Hybrid Models on Realized Volatility Forecasting.* arXiv:2611.13212v1 [stat.AP], 09 Nov 2026. https://arxiv.org/abs/2611.13212v1
- Kim, H. Y. & Won, C. H. (2018). Forecasting the volatility of stock price index: a hybrid model integrating LSTM with multiple GARCH-type models. *Expert Systems with Applications* 103, 25–37.
- Hajizadeh, E. et al. (2012). A hybrid modeling approach for forecasting the volatility of S&P 500 index return. *Expert Systems with Applications* 39(1), 431–436.
- Politis, D. N. (2015). *Model-Free Prediction and Regression.* Springer.
- Pan, L. & Politis, D. N. (2016). Bootstrap prediction intervals for linear, nonlinear and nonparametric autoregressions. *JSPI* 177, 1–27.
- Wu, K. & Politis, D. N. (2024a). Bootstrap prediction inference of nonlinear autoregressive models. *Journal of Time Series Analysis* 45(5), 800–822.
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Hansen, P. R. & Lunde, A. (2005). A forecast comparison of volatility models: does anything beat a GARCH(1,1)? *Journal of Applied Econometrics* 20(7), 873–889.