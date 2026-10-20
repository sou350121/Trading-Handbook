<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=生成式大模型 alpha=端到端表征 autonomy=全自动黑盒 -->

# OrderFlowDiT 解構（OrderFlowDiT）

> **來源發布**：2026-10-15 · （無 venue） · arXiv [2610.22727](https://arxiv.org/abs/2610.22727)
> **arXiv 原文**：[OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting](https://arxiv.org/abs/2610.22727v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：微觀盤口 × 5 分鐘 × 生成式 × 端到端表徵的**條件式限價單供給分佈模型**；解的是「coordinate-wise MSE 對價格軸無距離感、balanced OT 又把總量正規化掉」這個 prior gap，並附帶一套 full-field + proper scoring rule 的評測框架。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `生成式大模型` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 用條件 Diffusion Transformer 生成未來 5 分鐘「未正規化」的 bid/ask volume-at-price field（B=10001 bins、1 tick 間距），保留總量與價位配置兩個維度；② 核心 trick 是在解碼回 shares 的場上、對 denoising MSE 之外加一個**去偏 unbalanced Sinkhorn divergence**（ground cost = tick 位移，ε=4、ρ=20、λ_UOT=0.1、α_n² 噪聲門控），讓 loss 同時感知「放錯價位」與「總量不匹配」；③ 對「生成式大模型 × 端到端表徵」這條軸★的意義是：把 generative world model 的評測從 summary statistics 推到 proper scoring rules，而不是又一個 alpha 模型；④ 30 檔美股、1,950 個驗證條件上，Base-UOT 相對 Base-MSE 讓 full-field transport cost 降 45.8%、transport energy score 降 26.1%。

**X-Ray.** 放回五軸 Pareto：它相對 message-level 自迴歸生成器（Nagy et al. 2023）、DiffVolume 的 resting-volume 快照、queue-reactive Hawkes 的參數化強度，換來的是一個**可被 proper scoring rule 打分**的聚合場——代價是丟掉 event order、cancellation、execution。這個取捨很誠實，也很致命：submission field 無法決定下一個 book state（§3.1、§8 自述），所以整篇沒有任何 PnL、Sharpe、成本層的數字，`alpha=端到端表徵` 這格是**待驗證的期票**，不是已交付的 alpha。真正的工程貢獻在 loss 幾何：10001 bins 上的 MSE 是「無距離」的，100 股放錯 1 檔與放錯 500 檔同罰；而 balanced OT 要求等質量，偏偏預測與實現的 5 分鐘送單量本來就不等。UOT 一次解掉兩個坑，且輔助項加在 shares 空間而非 velocity 空間、用 α_n² 在高噪聲步自動淡出——這是這篇最值得偷的設計。它打不開的 envelope：無 impact model、單一驗證日兼選模日、單 seed、固定 30 檔大型股面板。GRPO 那組結果則是反例教材：reward 一調，ES_T 與 volume CRPS 變好，ES_2、d_full、rank calibration 一起變差——分佈指標不是同一個方向的。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（代表） | OrderFlowDiT | 為什麼是改動 |
|---|---|---|---|
| 預測目標 | message-level 自迴歸序列（Nagy et al. 2023）；resting-volume 快照（DiffVolume, Wang & Ventre 2025）；agent-based 流（ABIDES） | 未正規化 bid/ask 5 分鐘 **submission field**，B=10001 bins、1 tick、mid-anchored | 同時保留「價位配置」與「絕對總量」；捨棄 within-window event order |
| 訓練目標 | coordinate-wise denoising MSE | MSE + debiased **unbalanced** Sinkhorn divergence（decoded shares 上、FFT log-domain、dual potentials detached） | MSE 對 tick 位移不敏感；balanced OT 的等質量假設與任務衝突 |
| 評估 | summary statistics（LOB-Bench 的 L₁/W₁ + learned discriminator） | d_full + ES_T + ES_2 + volume CRPS + rank calibration（RI / interior coverage） | 平均場距離會獎勵過度集中；proper scoring rule 才對「分佈」評分 |
| 微調（附加） | — | GRPO on reward = −ℓ_k + ½(G−1)⁻¹Σ inter-draw 分離項 | 檢驗「訓練 reward 改善」能否遷移到分佈指標 |

### 1.2 ⚡ Eureka

**「把 OT 的 ground cost 設成 tick 距離、把邊際約束鬆成 KL 懲罰，然後把它加在解碼回 shares 的場上，而不是 velocity 上。」**

直覺：MSE 只會說「這個 bin 的數字錯了」；UOT 會說「你把 100 股放在第 3 檔，實際在第 8 檔，位移 5 ticks，而且總量還少 40 股」——錯在哪，變成了有單位的幾何成本（share·ticks）。balanced OT 做不到這件事，因為它一開始就假設兩邊質量相等；而這裡「總量」恰恰是作者最不想丟掉的資訊（§1 明言 unit-mass normalization 會丟掉 absolute volume）。

### 1.3 信息流

```
X_k (2×B book depth) ─┐
best-quote ind (2×B) ─┤
A_k^- (2×B obs flow) ─┴──→ C_k (6×B) ──┐
                                        ├─ patch embed (P=16) + sinusoidal PE
Z_n (2×B noisy target) ─────────────────┘        → L=626 tokens
                                                  │
                        [n, h_k, sym_k, log Δw, mean cond token] ──AdaLN-Zero──┤
                                                  ↓
                                        12× DiT block (d=512, 8 heads)
                                                  ↓
                                        linear head → velocity v_φ (2×B)
                                          ↙                    ↘
                              L_MSE (masked bins)      Ŷ = α_n Z_n − σ_n v_φ
                                                                ↓ Â = T⁻¹(Ŷ) in shares
                                                       S̄(Â_s, A⁺_{k,s})  [UOT, Base-UOT only]
                                                                ↓
                                                    realised A⁺_k  ← 只在此處接觸
```

採樣：R=4 條獨立高斯噪聲、50 DDIM steps（η=0）、EMA 權重 → ensemble。

## §2 · 數學層

📌 **Napkin Formula**

$$\mathcal{L}=\mathbb{E}\Big[\mathcal{L}_{\text{MSE}}+\frac{\lambda_{\text{UOT}}\,\alpha_n^{2}}{2}\sum_{s}\bar{S}\big(\hat{A}_s,\,A^{+}_{k,s}\big)\Big],\qquad \bar{S}(u,v)=\frac{\mathcal{U}_{\epsilon,\rho}(u,v)-\tfrac12\mathcal{U}_{\epsilon,\rho}(u,u)-\tfrac12\mathcal{U}_{\epsilon,\rho}(v,v)+\tfrac{\epsilon}{2}[M(u)-M(v)]^{2}}{\max\{[M(u)+M(v)]/2,\,1\}}$$

其中 $\mathcal{U}_{\epsilon,\rho}(u,v)=\min_{\pi\ge0}\{\langle C,\pi\rangle+\epsilon\,\mathrm{KL}(\pi\|u\otimes v)+\rho\,\mathrm{KL}(\pi\mathbf{1}\|u)+\rho\,\mathrm{KL}(\pi^{\top}\mathbf{1}\|v)\}$，$C_{ij}= \mid i-j \mid$（ticks）。

**直覺**：三項各司其職——第一項是 transport 成本（位移），後兩項鬆弛邊際（允許質量不等），最後的 $+\tfrac{\epsilon}{2}[M(u)-M(v)]^2$ 是去偏項（把 Sinkhorn 的自偏誤扣掉，讓 $S(u,u)=0$）。分母除以平均質量，是為了讓不同成交量的窗口之間 loss scale 可比。$\alpha_n^2$ 因子讓高噪聲步（乾淨場估計不可靠）的輔助訊號自動變小。

**複雜度**：B=10001 的均勻網格上，Sinkhorn kernel product 用 FFT 卷積，單次迭代 O(B log B)；迭代次數與 log-domain dual 求解細節所讀文本未見。每個訓練樣本要對 bid/ask 各做一次，且 dual potentials **detached**（梯度只走 cost 對解碼場的依賴）。

**Loss / 訓練細節**：1000-level cosine schedule；$Z_n=\alpha_nY+\sigma_n\varepsilon$、$v=\alpha_n\varepsilon-\sigma_nY$；masked velocity loss 只用有效 bin；AdamW、constant LR、batch 64、weight decay 0.01、grad clip 1、mixed precision、EMA 0.9999、單一 seed 42。Base：58.7M 參數（d=512、12 blocks、8 heads）、30,000 steps、LR 2×10⁻⁴；Large：174.3M（d=768、16 blocks、12 heads）、100,000 steps、LR 10⁻⁴。

## §2.5 · 帶數字走一遍（Worked Example）

> ⚠️ 以下全部是**假設/示意的玩具數字**，只用來手算機制；不是論文實證結果。論文真實數字一律在 §5。真實系統 B=10001，這裡壓成 5 個 bin。

**Step 1 — 假想輸入。** 某一個 forecast boundary、單邊（bid）：實現 $A^+=[0,30,50,20,0]$ 股（$M=100$）；模型解碼後 $\hat{A}=[0,10,40,50,0]$ 股（$M=100$）。

**Step 2 — 形狀距離 $w_s$（式 13）。** 歸一化 CDF：\$V=[0,.3,.8,1,1]\$、\$U=[0,.1,.5,1,1]\$；$w_s=\sum_j \mid F_U-F_V \mid =0+.2+.3+0+0=0.5$ ticks。（$w_s$ 只罰形狀，不看總量。）

**Step 3 — 合成 full-field cost（式 14–15）。** $d_{\text{full}}=\min(m,n)\,w_s+\lambda \mid m-n \mid$，$\lambda=50$ ticks → $100\times0.5+50\times0=50$ share·ticks。

**Step 4 — 總量不對會怎樣。** 若 $\hat{A}$ 形狀不變但總量是 200：$d_{\text{full}}=100\times0.5+50\times \mid 100-200 \mid =5050$。**質量項直接壓過形狀項**——這是 $\lambda=50$ ticks 的語義：1 股總量誤差，等價於把 1 股搬 50 檔。所以 $d_{\text{full}}$ 名義上是 transport cost，實務上很大程度是 volume metric，必須與 Sided W₁ 一起讀。

**Step 5 — UOT 分支（式 8–9）。** 把 \$u=[0,10,40,50,0]\$、\$v=[0,30,50,20,0]\$ 丟進 $\mathcal{U}_{\epsilon,\rho}$，$C_{ij}= \mid i-j \mid$、$\epsilon=4$、$\rho=20$：質量差由 KL 邊際懲罰吸收（不是硬約束、也不像 §5 的 $d_{\text{full}}$ 用 L¹ 罰），得到純量 $S$；去偏後除以平均質量得 $\bar S$，乘 $\lambda_{\text{UOT}}\alpha_n^2/2$ 加回 MSE。**關鍵**：若這一步發生在高噪聲步（$\alpha_n\to0$），輔助項幾乎不貢獻；噪聲越低、越接近乾淨場，UOT 才真正發話。

## §3 · 數據層

- **來源**：LOBSTER full-depth order-message data，固定 30 檔美股面板（AAPL, AMGN, AMZN, AXP, BA, CAT, CRM, CSCO, CVX, DIS, GS, HD, HON, IBM, JNJ, JPM, KO, MCD, MMM, MRK, MSFT, NKE, NVDA, PG, SHW, TRV, UNH, V, VZ, WMT）。
- **時段／頻率**：每場 10:30–16:00 ET；共同 1 cent 價格格；$\Delta w=300$ 秒。
- **表示**：mid-anchored grid $K=5000$ → $B=10001$ bins；log(1+v) 後標準化（bid/ask 分別估 $\mu,\sigma$，含零值 cell，$\tilde\sigma=\max\{\sigma,10^{-6}\}$）；解碼 $\max\{\exp(z)-1,0\}$、$z$ 截斷在 32。
- **切分**：Base 用 2026-06-08 至 06-22 共 10 個交易日；Large 加上 05-11 至 06-05 共 19 天 → 29 天。window-grid offsets 0/60/120/180/240 秒 → Base 96,300、Large 279,270 個訓練對。
- **評估語料**：2026-06-23 固定語料 1,950 conditions、每檔 65 個、target window 10:35–15:55 ET；**同日兼作 checkpoint selection**，06-24 保留在 configured test split 但未用於本文比較。
- **樣本外與容量假設**：訓練與評估是同一批 30 檔；作者明說 per-stock 結果只刻畫「this observed panel」。跨 universe、跨 regime 的外推能力**所讀文本未見**。
- **覆蓋率**：Figure 1 的 ±300 ticks 視窗只持有 53% 的送單量與 1.2% 的 resting depth——**新送單量比掛單深度更集中在近盤口**，這也是為何全場（未截斷）評估是必要的。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 所讀文本未見（全文無 repo／URL 指向程式碼） |
| Checkpoint | 所讀文本未見（只描述 EMA 與 checkpoint-selection 規則，未給發佈位置） |
| License | 未披露 |
| 複現難度 | 高：需 LOBSTER full-depth 授權資料、B=10001 網格、FFT-Sinkhorn、58.7M/174.3M DiT、單 seed 42 無法區分種子變異 |
| 數據可得性 | 商業資料（LOBSTER）；股票名單與切分日期在 §6.1 逐字給出，但 lag0 語料需自行重建 |

## §5 · 評測 / Benchmark

> 註：此處的「前SOTA」欄實為**同任務基線**，論文未宣稱 SOTA；一律逐行列出單一基線數值，未合成區間。所有 Δ 欄只放**原文明述的相對變化**（↓/↑ 為原文動詞方向），本文不自行做減法。

| 數據集/市場 | Metric（越低越好） | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 30 檔美股 · 2026-06-23 · 1,950 conditions | d_full/10⁶（share·ticks） | Base-MSE 110.829 | Base-UOT 60.074 | ↓45.8%（原文自述） |
| 同上 | ES_T | Base-MSE 2.152 | Base-UOT 1.591 | ↓26.1%（原文自述） |
| 同上 | ES_2 | Base-MSE 0.414 | Base-UOT 0.387 | 未披露 |
| 同上 | Volume CRPS | Base-MSE 0.460 | Base-UOT 0.283 | 未披露 |
| 同上 | Sided W₁（ticks） | Base-MSE 149.220 | Base-UOT 123.552 | 未披露 |
| 同上 | NTVE | Base-MSE 1.058 | Base-UOT 0.447 | 未披露 |
| 同上 | MSE_log | Base-MSE 0.656 | Base-UOT 0.599 | 未披露 |
| 同上 | d_full/10⁶ | Empirical 95.698 | Base-UOT 60.074 | 未披露 |
| 同上 | ES_T | Empirical 2.018 | Base-UOT 1.591 | 未披露 |
| 同上 | ES_2 | Empirical 0.496 | Base-UOT 0.387 | 未披露 |
| 同上 | NTVE | Skew-normal mixture 0.328 | Base-UOT 0.447 | 未披露（基線較優） |
| 同上 | d_full/10⁶ | Base-UOT 60.074 | Base-GRPO 62.025 | ↑3.2%（原文自述） |
| 同上 | ES_T | Base-UOT 1.591 | Base-GRPO 1.494 | ↓6.1%（原文自述） |
| 同上 | ES_2 | Base-UOT 0.387 | Base-GRPO 0.409 | ↑5.7%（原文自述） |
| 同上 | Volume CRPS | Base-UOT 0.283 | Base-GRPO 0.243 | ↓14.0%（原文自述） |
| 同上 | RI | Base-UOT 0.256 | Base-GRPO 0.269 | 未披露 |
| 同上 | Interior-rank coverage（nominal 0.600） | Base-UOT 0.395 | Base-GRPO 0.385 | 未披露 |

**哪些 Δ 是真 capability，哪些不是：**

- **d_full 的 ↓45.8% 要打折。** 訓練用的 UOT（KL 鬆弛 + tick ground cost）與評測用的 $d_{\text{full}}$（CDF 形狀 + L¹ 質量罰）是**同族但不同式**——作者自己在 §5.1 明說「It also differs from the entropic, KL-relaxed training objective in (8)」。同族意味著這是「對齊評測」的增益，不能全記在泛化能力上。
- **ES_2 的 0.414 → 0.387 最接近真 capability**：它是 binwise log-field 幾何（式 18），與 transport 完全不同的度量，且是 strictly proper。跨幾何仍改善，才說明模型真的更接近條件分佈。
- **ES_T 2.152 → 1.591** 介於兩者之間：proper，但 mass 項用 log、與訓練的線性質量罰不同。
- **NTVE 是反證**：skew-normal 兩條基線的 0.328 都比 Base-UOT 的 0.447 好。作者也承認「room for improvement in marginal volume calibration」。
- **GRPO 是典型的 reward 遷移失敗樣本**：ES_T 與 volume CRPS 變好，但 ES_2、d_full、interior coverage（0.395 → 0.385，離 nominal 0.600 更遠）一起惡化。機制在 §7.2 講得很清楚：$ES=T-D/2$，GRPO 同時拉近 target distance 與 inter-draw distance，transport 幾何下前者降幅夠大、joint-field 幾何下不夠——**生成變得更準但更同質**。而且 GRPO 同時換了 sampler（20 步 η=1 vs 50 步 η=0），權重與採樣兩個因素綁死，無法分離。
- **可能的樂觀偏誤**：2026-06-23 同時是 checkpoint selection day 與 reported eval day；單一 seed；描述性比較。這些不是 data leakage（conditioning 嚴格用 $\tau_k$ 之前的 message，$\tau_k$ 當刻的新單只進 target），但**選模泄漏**讓「validation 數字」不能當 held-out 讀。
- **成本未計**：全表沒有任何交易成本、滑價、Sharpe/IR/MDD。這是分佈保真度 benchmark，不是策略 benchmark。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **§1 / §3.1 / §8**：submission field 只彙總新送單，**不含 cancellation、execution，也不保留 event order**；bid/ask 支撐可以重疊（同價位不同時間可能收到不同方向的單）。因此「generated submission field alone does not determine the next book state」。
- **§8（核心限制，作者原文）**：generating submission flow does not complete a financial world model，仍需一個 **impact model** 才能把 book、order flow、trading action 連到後續 book state 與價格；這是 next step，本文未做。
- **§3.3**：模型只在有限網格（$K=5000$、$B=10001$）內預測，網格外流量只記錄作 coverage diagnostics。
- **§6.1**：訓練與評估是同一批 30 檔股票，per-stock 結果只刻畫此面板；同日兼作 checkpoint selection，故結果是「validation-set comparisons following chronological training」。
- **§6.4 / §7**：比較是描述性的，每個 learned configuration 只有一個訓練 seed。
- **§5.2**：$\mathrm{ES}_T$ 因 shape/mass/side 三項可加、不識別相依性，**不是嚴格 proper**（只 proper）；propriety 不延伸到「同一 ensemble 內混有空／非空場」的情形。
- **§7.2**：GRPO 結果「describe the combined change in weights and sampling」。
- **§7.3**：Large 配置同時改容量、資料、優化步數、學習率，「does not isolate their individual effects」。
- **§6.2**：skew-normal 基線只用近端 flow、不用 book depth；Empirical 只用 instrument identity + time of day + anchor，不用 flow/depth。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **Regime 依賴（推斷）**：全部結論建立在單一評估日 2026-06-23 上，未按波動/流動性 regime 分層。跨 regime 的 UOT 增益是否維持**未驗證**；不應把「every stock 都改善」讀成「所有市場狀態都改善」。
- **容量假設（推斷）**：1 tick 網格、5 分鐘聚合、10:30–16:00 連續交易時段；對跳動更大的低價股或 24 小時市場，$B=10001$ 的覆蓋與 λ=50 ticks 的語義都要重校。
- **成本層缺口（推斷）**：本文沒有任何交易成本假設，因為根本沒有策略層。任何「用此模型做執行」的推論都需要外部 impact model，本文不提供。
- **洩漏邊界（推斷）**：特徵嚴格用 $\tau_k$ 之前的 message，時間切分是 chronological，這部分乾淨；但評估日 = 選模日是**選模層面的樂觀偏誤**，非資料洩漏。兩者不可混為一談。
- **面板選擇（推斷）**：固定 30 檔大型美股、訓練與評估同名單——這是 **universe 固定**，不是 survivorship bias 的證明；但也意味著結果不能外推到小型股、退市股或多 venue 情境。
- **單 seed（推斷）**：seed 42 一組到底，無種子變異估計；GRPO 的 dispersion 崩縮是否為穩定現象，所讀文本無法判定。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Nagy et al. 2023（token-level 自迴歸 message 生成） | 保留 event order、可接 simulator；但不給 full-field 分佈評分 | 未披露 | 概念對比 |
| DiffVolume（Wang & Ventre 2025） | 生成 resting-volume 快照，非「新送單量」 | 未披露 | 概念對比 |
| MarS（Li et al. 2025） | 互動式 simulator、order/order-batch 生成；本文只做 upstream 供給場 | 未披露 | 概念對比 |
| queue-reactive Hawkes（Huang et al. 2015；Wu et al. 2019） | 參數化強度 vs 端到端學分佈 | 未披露 | 概念對比 |
| LOB-Bench（Nagy et al. 2025） | summary statistics + discriminator；本文補 full-field transport + proper scoring rules | 未披露 | 互補評測框架 |
| COT-GAN / Sinkhorn GAN（Xu et al. 2020；Genevay et al. 2018） | transport 在「分佈之間」；本文在「單一預測–目標對」之內 | 未披露 | 方法 lineage |
| DDPO / DanceGRPO（Black et al. 2024；Xue et al. 2025） | reward fine-tuning 用於影像／影片；本文用於條件訂單流 | 未披露 | 方法 lineage |
| 統計基線（skew-normal / mixture / Empirical） | 逐 boundary 擬合、deterministic R=1；NTVE 反勝 | 未披露 | 已實測基線 |

🎤 **Interview Tip**

> **Q：UOT 輔助損失為什麼加在 decoded field（shares）而不是 velocity 空間？**
> **正確答**：transport cost 的幾何只有在 shares 空間才有語義——$C_{ij}= \mid i-j \mid$ 是 tick 位移，$M(u)-M(v)$ 是股數差；velocity 空間的座標差沒有價格距離的意義。而且要先去標準化（$T^{-1}$）、clip 到非負，質量才有物理意義。再加 $\alpha_n^2$ 門控，讓高噪聲步的輔助訊號淡出。
> **錯答**：「UOT 是 regularizer，加在哪一層都等價」或「UOT 取代了 MSE」。兩者都不是：論文明寫是 augment／alongside，λ_UOT=0 就退回 MSE-only。

> **Q：為什麼 GRPO 讓 ES_T 變好、ES_2 卻變差？模型不是退步了嗎？**
> **正確答**：$ES_j=T_j-D_j/2$。GRPO 同時降低 target distance 與 inter-draw distance；在 transport 幾何下前者降幅蓋過 dispersion 損失，在 joint-field 幾何下蓋不過。這是「更準但更同質」的取捨，而且它同時換了 sampler（20 步 η=1），無法分離權重與採樣貢獻。
> **錯答**：「ES_2 算錯了」或「ES_T 才是對的」。兩個都是 proper（ES_2 還是 strictly proper），方向不一致本身就是結論。

### 7.1 可證偽預測

1. **2027-06-30 前**：若沒有人補上 §8 要求的 impact model 並串接 cancellation/execution，這條研究線不會出現以 Sharpe / IR / MDD 為指標的評測。（依據：submission field 無法決定下一期 book state。）
2. **2026-12-31 前**：若在 30 檔以外（小型股、多 venue、非 1 cent tick）重跑，UOT 對 $d_{\text{full}}$ 的相對增益會低於本文的 45.8%。（依據：大型股高流動性使 5 分鐘供給場較平滑，λ=50 ticks 的質量／形狀權衡在此面板最有利。）
3. **2027-03-31 前**：GRPO 的 dispersion 崩縮（ES_2 上升、interior coverage 下降）在 ≥3 個 seeds 下仍會重現，且無法只靠調 KL 係數 0.02 消除——因為 reward 裡的 inter-draw 項係數是逐對平均、量級低於 target 項。

## §8 · For the Reader

- **因子研究員**：這篇不是 alpha 論文，別在裡面找 Sharpe。可用的是**評測觀**——把「生成場的均值誤差」換成 proper scoring rule（ES / CRPS）+ calibration（rank coverage）。任何做 order-flow 特徵生成、情境模擬的研究，都該用 §5 那八個量自查，而不只是看平均場 MSE。
- **高頻執行**：$\Delta w=300$ 秒、1 tick 網格、10:30–16:00 的設定，對執行排程偏粗；但「5 分鐘內新送單量落在哪裡」正好是 optimal execution 裡的流動性補充率輸入（Obizhaeva & Wang 2013 的動機）。可以把它當**上游供給預測**，但必須自己接 impact model，本文不含。
- **組合配置**：用處在壓力情境生成——$R=4$ 條獨立樣本代表 5 分鐘後的可能供給場。注意 calibration 並不完美（interior coverage 0.395 遠低於 nominal 0.600），別把 ensemble 當校準好的機率用。
- **LLM-agent / 世界模型研究者**：這是「生成式模型 × 金融」的一個乾淨 case study：作者誠實地把模型定位成 world model 的**元件**而非模擬器，並且用 proper scoring rule 防範「生成很真但分佈錯了」的陷阱。GRPO 那節是 reward hacking 的現成教材。
- **RL 策略研究者**：GRPO 的失敗模式值得抄進 checklist——reward 加 diversity 項不等於分佈正確；$ES=T-D/2$ 的分解能直接指出是「變準」還是「變同質」。同時注意此處 GRPO 與 sampler 改動綁死，不是乾淨的 A/B。
- **研究學生**：讀 §4.2 + §5.1 的對照——**訓練用的 UOT 與評測用的 $d_{\text{full}}$ 刻意不同式**（KL 鬆弛 vs L¹ 質量罰、entropic vs CDF 形狀）。這種「訓練目標不與評測指標完全對齊」的自我約束，是這篇方法論上最值得學的習慣。

## References

- 原論文：Jung Hun Phee, Tucker Balch. *OrderFlowDiT: Transport-Regularized Diffusion Transformer for Probabilistic Limit-Order Supply Forecasting.* arXiv:2610.22727v1 [cs.LG], 15 Oct 2026. https://arxiv.org/abs/2610.22727v1
- 擴散／DiT lineage：Ho et al. 2020 (DDPM)；Song et al. 2021 (DDIM)；Peebles & Xie 2023 (DiT, arXiv:2212.09748)
- 最優傳輸 lineage：Peyré & Cuturi 2019；Genevay et al. 2018 (Sinkhorn GAN)；Feydy et al. 2019；Séjourné et al. 2019 (UOT Sinkhorn, arXiv:1910.12958)；Piccoli & Rossi 2014；Xu et al. 2020 (COT-GAN)
- 評分規則：Gneiting & Raftery 2007；Gneiting et al. 2007；Ferro 2014；Matheson & Winkler 1976；Székely & Rizzo 2013；Lyons 2013
- 市場模擬／評測：Nagy et al. 2023；Nagy et al. 2025 (LOB-Bench)；Li et al. 2025 (MarS)；Li et al. 2020 (Stock-GAN)；Coletta et al. 2022 (World Agent)；Wang & Ventre 2025 (DiffVolume)；Byrd et al. 2020 (ABIDES)；Huang et al. 2015；Wu et al. 2019；Cont 2001；Vyetrenko et al. 2020
- RL 微調：Black et al. 2024 (DDPO)；Shao et al. 2024 (GRPO)；Xue et al. 2025 (DanceGRPO)
- 來源鏈接：https://arxiv.org/abs/2610.22727v1（本次輸入為完整可讀全文文本；圖片／附件未提取，Figure 1–4 內容僅以正文描述為據）