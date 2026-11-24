<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=强化学习 alpha=组合执行优化 autonomy=全自动黑盒 -->

# Optimal execution in a f 解構（Optimal execution in a f）

> **來源發布**：2026-11-08 · （無 venue） · arXiv [2611.11373](https://arxiv.org/abs/2611.11373)
> **arXiv 原文**：[Optimal execution in a fractional setting with reinforcement learning](https://arxiv.org/abs/2611.11373v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：在 fBm（H≠1/2）線性衝擊清算中，用 policy gradient + NN 直接最大化 CARA 期望效用，補上「AC 確定性解不再最優、但無閉式解」的缺口；五軸落在 RL／執行優化，不是 alpha 預測。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `高频日内` | `强化学习` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文。細節待升 v1。

**TL;DR:** ①把 Almgren–Chriss 的線性衝擊清算放進 fBm，H≠1/2 時價格增量有序列相關，確定性 AC 排程不再最優；②trick 是把過去 p 個收益的線性預測 \(\hat R_{t,t+\tau}\) 塞進策略特徵，用 REINFORCE 以 AC 為 baseline 學「相對 AC 的偏離」，H=1/2 時預測退化為 0、策略回到 AC；③對 RL／執行軸★的意義是：不是學 alpha，而是把非馬可夫記憶壓成一個可執行特徵，繞開無閉式解的困境；④關鍵實證數字：Table 1 在 H=0.2 給 AC −991.8、NN(p=5) −644.7±3.0（n=10^6），但全是模擬 fBm，不是真實市場樣本外。

**X-Ray.** 這篇的價值不在「RL 打敗 AC」的口號，而在把「fBm 的序列相關」轉成一個低維、可解釋的執行特徵：\(\hat R_{t,t+\tau}\)。它同時避開兩個舊坑：一是 mean-variance 在 H≠1/2 下的時間不一致，改用 CARA 期望效用；二是 fBm 下最優控制無閉式解，改用 policy gradient 數值逼近。五軸上，它標成「量價表格／高頻日內」，但原文的清算 horizon 是 T=5 days、步長 Δ=0.05 days，特徵只有價格路徑與庫存，沒有真實 LOB／成交量表格——這是五軸標記與原文證據的第一個張力。它打開的 envelope 很窄：H 與 σ 必須已知或可估，且每個 H 單獨訓練；一旦真實市場的 H 不穩定、或成本／放空限制進場，模擬裡的 CE 優勢不自動搬過去。對量化讀者，最可搬動的是「用預測標量當執行特徵、用 AC 當 control variate」這個結構，而不是 Table 1 的具體數字。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作 / AC 基線 | 本文 | 讀者該注意 |
|---|---|---|---|
| 價格過程 | 布朗 \(B^{1/2}\) | fBm \(B^H\)，H≠1/2 | H 需已知或估計；整條 pipeline 是 H-specific |
| 目標 | 均值方差 \(\mathrm{MV}_\lambda\) | CARA \(\mathbb E[-\exp(-\lambda\Pi_T)]\) | 避免時間不一致；H=1/2 時與 MV 等價（\(\tilde\lambda=\lambda/2\)） |
| 策略類 | 確定性、有限變差 | NN 參數化隨機策略，條件於 \(F_t\) | 非馬可夫信息進 \(\hat R_{t,t+\tau}\) |
| 求解 | HJB／閉式 | REINFORCE + AC baseline advantage | 無閉式解下的數值逼近 |

### 1.2 ⚡ Eureka

**一句話：** 把 fBm 的記憶壓成單一標量預測 \(\hat R_{t,t+\tau}\)，再用 AC 當 baseline 的 advantage，讓網路只學「何時該偏離 AC」。
**直覺：** H≠1/2 時，過去收益能預測未來方向；H=1/2 時 \(\hat R=0\)，網路沒有可偷的信息，最優退化成 AC。於是「超越 AC」不是靠更多參數，而是靠一個由 H 決定的預測特徵。

### 1.3 信息流 ASCII 圖

```
H, σ ──► fBm 模擬（Wood–Chan，路徑 [-L, T]）
              │
過去收益 Z_t ──► 線性預測 \hat R_{t,t+τ} = β^T Z_t
              │
F_t=(t, q_t, S_t, \hat R) ──► MLP μ_θ（兩層 20 ReLU）
              │
q_{t+1} ~ N(μ_θ, σ_explo²) ──► q_T=0（強制清倉）
              │
Π_T = X_T - Q S_0 ──► G^nn = -exp(-λΠ_T)
              │
同一路徑 AC 的 G^ac ──► A = G^nn - E[G^ac|G_t] ──► REINFORCE 更新 θ
```

## §2 · 數學層

**📌 Napkin Formula**

\[
\sup_{v,\;q_T=0}\mathbb E\{-\exp(-\lambda\Pi_T)\},\qquad
\Pi_T=X_T-QS_0,\qquad
X_t=\int_0^t\left(S_sv_s-\eta v_s^2\right)ds
\]

策略梯度（含 AC baseline 的 advantage）：

\[
\widetilde{\nabla_\theta J}(\theta)=
\frac1m\sum_{i=1}^m\sum_{k=0}^{N-2}
A_{t_k}^i\,
\frac{q_{t_{k+1}}^i-\mu_\theta(F_{t_k}^i)}{\sigma_{\mathrm{explo}}^2}\,
\nabla_\theta\mu_\theta(F_{t_k}^i)
\]

**直覺：** \(A_{t_k}^i\) 是「同一條 fBm 路徑上，NN 相對 AC 多賺或少賺」；正的 advantage 把 \(\mu_\theta\) 推向當次抽樣動作。AC 是 baseline，不是對手；它的作用是降低梯度方差，讓「偏離 AC」的信號更乾淨。
**複雜度：** fBm 用 Wood–Chan，總成本 \(O(nN\log N)\)；每一步預測要逆一個 \(p\times p\) 協方差矩陣；網路是兩層 20 ReLU。學習率 \(\alpha\)、batch \(m\)、探索噪聲 \(\sigma_{\mathrm{explo}}\) 的具體值所讀文本未見。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為流程示意，數字只引用原文設定與 Table 1，不是新實驗；不計算新的差值。

**步驟一：** 取原文 §4.1 的設定：S0=100、Q=1000、T=5 days、Δ=0.05 days、H=0.2、p=5。這些是原文的模擬參數，不是本文新假設。

**步驟二：** 由過去 5 個 lag 收益 \(Z_t\) 與 fBm 協方差算 \(\hat R_{t,t+\tau}=\beta^\top Z_t\)，其中 \(\beta=\Sigma_Z^{-1}\Sigma_{RZ}^\top\)。H=0.2 時增量負相關，預測項不為 0；若 H=1/2，預測項退化為 0。

**步驟三：** 把 \(F_t=(t,q_t,S_t,\hat R_{t,t+\tau})\) 餵進 \(\mu_\theta\)，得到高斯策略的均值；抽樣 \(q_{t+1}=\mu_\theta+\sigma_{\mathrm{explo}}\varepsilon\)，最後強制 \(q_T=0\)。

**步驟四：** 用同一條路徑算 \(\Pi_T^{nn}\) 與 AC 的 \(\Pi_T^{ac}\)，得 \(G^{nn}\)、\(G^{ac}\)，再算 advantage \(A\)。這一步是「同一路徑配對」，不是拿不同模擬路徑硬比。

**步驟五：** 用式 (9) 更新 \(\theta\)。在 H=0.2 的 Table 1，NN(p=5) 的 CE 是 −644.7±3.0，AC 是 −991.8（n=10^6，95% CI）；方向是 NN 較高。這裡只引用原值，不把差值寫成新數字。

## §3 · 數據層

- **資料來源：** 全部是模擬 fBm 路徑，不是真實市場量價表格。Wood–Chan circulant embedding，n=10^6、N=800；路徑模擬在擴展窗口 [-L, T]，L=35，T=5 days，Δ=0.05 days。
- **頻率／時段：** 步長 Δ=0.05 days，原文稱約 T 的 1%；清算 horizon T=5 days。五軸標「高頻日內」，但原文不是 tick 級，而是日內到多日的執行排程。
- **參數：** S0=100 \$/share、Q=1000 shares、σ=0.015 S0、η=10^{-3} \$×day/share²、λ=10^{-3} \$^{-1}、τ≃0.94 days（原文稱約 liquidation period 的 20%）。
- **特徵：** \(F_t=(t,q_t,S_t,\hat R_{t,t+\tau})\)；\(\hat R\) 由 p 個過去收益的線性預測而來，p=5 是原文的平衡點；p=7 在部分 H 下最深 lag 大於 100 days。
- **樣本外：** 測試用同參數新生成的 n=10^6 條軌跡；這是「模擬分佈內樣本外」，不是真實市場樣本外。容量假設、真實衝擊校準、借券／放空限制所讀文本未見。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 所讀文本未見 |
| Checkpoint | 所讀文本未見 |
| License | 所讀文本未見 |
| 複現難度 | 中高：需 fBm 模擬、H／σ 估計、REINFORCE、PyTorch 自動微分；原文給演算法但未給代碼 |
| 數據可得性 | 模擬可復現；真實市場數據未使用，故無真實數據可得性問題 |
| 關鍵依賴 | H-specific 訓練；每個 H 需單獨模型；\(\hat R\) 需 H 才能算 |

## §5 · 評測 / Benchmark

> 表內「前SOTA」欄實際是 **AC 基線**，不是 SOTA；不合成任何前SOTA 區間。Δ 欄留空，避免把模擬內差值誤讀成真實改善量。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| fBm 模擬，H=0.2（n=10^6） | CE（95% CI） | AC 基線 −991.8 | NN(p=5) −644.7±3.0 | — |
| fBm 模擬，H=0.2（n=10^6） | CE（95% CI） | AC 基線 −991.8 | NN(p=1) −699.6±3.4 | — |
| fBm 模擬，H=0.2（n=10^6） | CE（95% CI） | AC 基線 −991.8 | NN(p=7) −638.4±3.8 | — |
| fBm 交叉評估，H=0.35 | CE（95% CI） | AC 基線 −996.3 | NN(H=0.35) −900.4±3.0 | — |
| fBm 交叉評估，H=0.5 | CE（95% CI） | AC 基線 −1034.0 | NN(H=0.5) −1035.1±2.5 | — |
| fBm 交叉評估，H=0.65 | CE（95% CI） | AC 基線 −1106.1 | NN(H=0.65) −1035.0±2.3 | — |
| fBm 參數誤設，M=1 | CE（95% CI） | AC 基線 −990.9 | NN −866.8±2.9 | — |
| fBm 參數誤設，True parameters | CE（95% CI） | AC 基線 −990.9 | NN −823.0±3.1 | — |
| fBm 分佈，H=0.75 | \(\mathrm{MV}_\lambda\) | AC −1823.5 | NN −1427.4 | — |
| fBm 分佈，H=0.6 | \(\mathrm{MV}_\lambda\) | AC −1620.3 | NN −1554.5 | — |

**解讀：**
- **真 capability 的部分：** Table 1／Table 2 的改善是在獨立生成的模擬軌跡上量測，且 H=1/2 時 NN(H=0.5) −1035.1±2.5 與 AC −1034.0 幾乎重合，符合「無信息時退回 AC」的 sanity check。這是模型內的算法效果，不是真實市場 alpha。
- **可能是過擬合／分佈內樣本外：** Table 2 的 cross-evaluation 是同一 fBm 分佈下換 H 測試，對角線最好只說明「H-specific 訓練有必要」，不等於真實 regime 切換下的穩健。Table 3 的參數誤設只測 H=0.3、σ=1.5；原文說 20% 相對誤差只讓 CE 退化約 5%，但這是該設定下的結果，不能外推所有 H。
- **尾部與事後選取：** §4.4 的單路徑差值 +2483／−569（H=0.7）與 +766／−863（H=0.3）是事後按 \(\Pi_T^{nn}-\Pi_T^{ac}\) 選的 99／1 百分位路徑，不是平均處理效應。原文自己說 H>1/2 的平均優勢「not obtained by doing better on most paths」，而是來自極端路徑；H<1/2 才是多數路徑較好。
- **成本未計：** 模型只有線性臨時衝擊 \(\eta v^2\)，未見手續費、買賣價差、延遲、永久衝擊、最小交易單位、借券／放空限制。H=0.7 的策略會把庫存打到 -270 shares，這在真實市場是放空回補，不是無摩擦的數學負庫存。
- **指標語義：** CE 是 certainty equivalent，原文用它給 CARA 效用貨幣含義；\(\mathrm{MV}_\lambda\) 是均值方差準則。兩者不可混成同一欄，也不可把 CE 的差值直接叫 alpha。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§5 結論：** 價格過程需服從 fBm，而 fBm 對真實市場數據仍是討論中的模型。
- **§5 結論：** H→0 或 H→1 時，學習過程仍可能 chaotic，因為 reward 波動強。
- **§3 開頭：** 整條 pipeline 是 H-specific；H 必須已知，才能模擬訓練軌跡與計算預測，每個 H 需單獨訓練模型。
- **§3.2：** 計算 \(\hat R_{t,t+\tau}\) 需要知道 H，因此 H 估計精度很重要。
- **§4.7：** 實務上 H 與 σ 需從有限樣本估計；原文只在此節测試 H=0.3、σ=1.5 的誤設穩健性。
- **圖表待核：** Figure 1–11 的完整數值未在可讀文本中逐點列出；本頁只引用正文明確給出的數字，其餘標「圖表／附件待核」。

### 6.2 推斷的隱含假設（本頁推斷）

- **H 與 σ 固定：** 策略假設清算期內 H、σ 不變；若真實市場 H 隨 regime 漂移，H-specific 模型不自動成立。
- **線性臨時衝擊、無永久衝擊：** 只建模 \(\tilde S=S-\eta v\)；若真實執行有永久衝擊或非線性衝擊，最優偏離方向可能改變。
- **無摩擦：** 無手續費、價差、延遲、放空限制；H>1/2 的短倉策略在真實市場需借券與風控，未必可執行。
- **可連續交易：** Δ=0.05 days 的離散網格；真實 tick／lot size 未測。
- **單資產：** 原文以 fBm 的時間序列相關替代跨資產對沖；多資產的相關結構未測。
- **模擬分佈內樣本外：** 測試軌跡與訓練同參數生成；不是真實市場樣本外，也不是生存者偏差檢驗（本文未用輔助數據，此項不適用）。
- **CARA 參數固定：** λ=10^{-3} \$^{-1} 是原文校準值；不同 λ 下的最優偏離未見系統性掃描。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Almgren–Chriss (2000) | 布朗、確定性、均值方差 | 經典 | 本文 baseline；H=1/2 時本文退回它 |
| Dolinsky–Greenstein (2025) | fBm 下只最大化期望收益 | 論文 | 不同目標；無風險厭惡 |
| Han–Hu–Song (2013) | fBm 最大值原理，H∈(1/2,1) | 論文 | 限制在 H>1/2，且無終端清算約束；只給必要條件 |
| Guasoni–Mishura–Rásonyi (2021) | 高頻 fBm、時間加性 MV | 論文 | 不同目標；本文用 CARA |
| Bank–Soner–Voß (2017) | 對沖誤差 vs 交易成本 | 論文 | 不同框架；本文是清算 |
| 本文 | policy gradient + CARA + \(\hat R\) 特徵 | 未見 code | 模擬 fBm；H-specific |

**🎤 Interview Tip**
- **正確答：** 這篇的核心是「用 \(\hat R_{t,t+\tau}\) 把 fBm 的序列相關壓成執行特徵，用 AC 當 baseline 的 REINFORCE 學偏離；CARA 是為了避開 MV 的時間不一致；H=1/2 退化回 AC 是 sanity check；證據只在模擬。」
- **錯答：** 「它學到了 alpha」——不對，這是執行優化，不是預測收益因子；「它在真實市場打敗 AC」——原文沒有真實市場實驗；「H 不需要知道」——原文明說整條 pipeline 是 H-specific；「+2483 是平均改善」——那是事後選的 99 百分位單路徑。

### 7.1 可證偽預測（本頁預測，帶日期）

若在 **2027-06-30** 前，用真實日頻／日內數據估計 H，且 H 顯著偏離 1/2，按原文 pipeline 訓練並在真實樣本外評估，則應觀察到 NN 的 CE 高於 AC；若 H 估計不穩定、或 NN 的優勢主要集中於尾部路徑而非多數路徑，則本預測失效。若真實成本（價差、延遲、放空費）進場後優勢消失，則原文的模擬結論不延伸到實盤。

## §8 · For the Reader

- **因子研究員：** 這不是新因子，而是執行層。可借的是「把預測標量當執行特徵」的結構；但 \(\hat R\) 需要 H，若你的市場 H 不穩定，先做 H 穩定性檢驗，不要直接搬 Table 1 的數字。
- **高頻執行：** 注意原文 horizon 是 T=5 days、Δ=0.05 days，不是 tick 級；真正的短倉回補（-270 shares）在實盤有借券與風控成本。可先測「AC + 預測偏離」的簡化版，不必上 NN。
- **組合配置：** CARA 會壓尾部，原文在 H=0.6 觀察到「降方差但降均值」；若你的效用是 MV，這種 trade-off 要重新校準 λ，不能直接把 CE 當收益。
- **RL 策略：** 可抄的工程點是 same-path baseline、AC 當 control variate、H-specific 訓練、強制 \(q_T=0\)；要注意 exploration 噪聲與 batch 的具體值原文未披露，需自己調。
- **研究學生：** 想複現，先讀 §2.2 Wood–Chan 與 §2.3 二階差分估 H；§2.4 的兩期時間不一致例子给 H*≃0.83，是理解「為何不用 MV」的好入口。

## References

- 原論文：Matthieu Garcin, Jiang Pu, Thomas Verrier, *Optimal execution in a fractional setting with reinforcement learning*, arXiv:2611.11373v1, 2026. [連結](https://arxiv.org/abs/2611.11373v1)
- Almgren & Chriss (2000), *Optimal execution of portfolio transactions* — 本文 AC 基線與參數校準來源。
- Mandelbrot & Van Ness (1968), *Fractional Brownian motions, fractional noises and applications* — fBm 定義。
- Garcin (2022), *Forecasting with fractional Brownian motion: a financial perspective* — \(\hat R\) 預測與 lag 選擇。
- Dolinsky & Greenstein (2025), *A note on optimal liquidation with linear price impact* — fBm 下只最大化期望的對照。
- Guasoni, Mishura & Rásonyi (2021), *High-frequency trading with fractional Brownian motion* — 時間加性 MV 的對照。
- Han, Hu & Song (2013), *Maximum principle for general controlled systems driven by fractional Brownian motions* — H∈(1/2,1) 的最大值原理對照。
- Bank, Soner & Voß (2017), *Hedging with temporary price impact* — 對沖誤差框架對照。
- Sutton & Barto (2018), *Reinforcement learning: an introduction* — policy gradient／REINFORCE。
- Wood & Chan (1994), *Simulation of stationary Gaussian processes in [0,1]^d* — fBm 模擬。
- Coeurjolly (2001), *Estimating the parameters of a fractional Brownian motion by discrete variations of its sample paths* — H／σ 估計。