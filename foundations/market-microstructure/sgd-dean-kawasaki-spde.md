<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=组合执行优化 autonomy=全自动黑盒 -->

# SGD市场做市 / Dean-Kawasaki SPDE 模型 解構（SGD市场做市 / Dean-Kawasaki SPDE 模型）

> **來源發布**：2026-11-18 · （無 venue） · arXiv [2611.24028](https://arxiv.org/abs/2611.24028)
> **arXiv 原文**：[Algorithmic Market Making Can Lead to Rough Volatility Through the Order Book](https://arxiv.org/abs/2611.24028v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把做市商報價寫成異步一階梯度下降，聚合為 information-driven mini-batch SGD，連續極限得到守恆型 Dean–Kawasaki SPDE；多 horizon 的指數記憶混合成冪律，導出 rough volatility。五軸落點是微觀盤口＋高頻日內＋因果結構；Alpha 軸以組合執行優化定位，但原文沒有策略 PnL 或交易訊號。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 每個做市 desk 用一階梯度把掛單距離拉向未知 oracle 的 target，只在與自身 horizon 相關的新資訊到達時動作，聚合後是 information-driven mini-batch SGD。② 連續極限下 order book volume 服從守恆 Dean–Kawasaki SPDE；不同 lookback horizon 的指數記憶混合成 \(t^{-1-\alpha}\)，標準標度給 \(H=\alpha-1/2\)。③ 對「因果結構」軸★：rough vol 的 kernel 不再完全外生擬合，而由 book shape 與 horizon 分佈內生決定。④ Binance 六市場六個月、about 790 million orders 上，\(H(N_b)\) between 0.07 and 0.09，\(H(\mathrm{RV}_b)\) lies within 0.018 of \(H(N_b)\)；但 fitted \(\alpha\) between 0.39 and 0.50，極限 \(H=\alpha-1/2\) 在這些值下不為正，作者以有限 \(T\) 的 desk 模擬補位。

**X-Ray.** 這篇的價值不在「又一個 rough vol 模型」，而在把三條線焊在一起：Gess–Gvalani–Konarovskyi 的 conservative SPDE 作為 SGD 平均場極限、Jaisson–Rosenbaum 的近不穩定 Hawkes 作為冪律記憶、以及做市商梯度報價作為微觀主體。它解掉的舊工程坑是「rough vol 的 microfoundation 一直缺做市商」：過去要嘛假設交易者拆單（EEFR18）、要嘛假設無套利（JR20），做市商被當成背景流動性；這裡做市商是唯一主體，book volume 守恆、只重分配，mid price 由兩側 Laplace 形狀推動。它打不開的 envelope 也清楚：\(\alpha\le1/2\) 沒有極限定理（§8 自述）；模型沒有 taker、沒有庫存／逆選擇成本、沒有 fee/rebate；Binance 數據沒有 participant ID，activity 用 taker count 代理；\(H\) 的估計對日內 pattern、binning、noise correction 敏感（Table A.2–A.4）。對量化讀者，這是「流動性生成過程的結構先驗」，不是可直接放進組合的 alpha。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作典型做法 | 本文改動 | 代價／張力 |
|---|---|---|---|
| 微觀主體 | 交易者拆單、無套利論證（EEFR18, JR20）；做市商是背景 | 每個做市 desk 是一階梯度規則，朝 oracle target 拉；雙邊參與者拆成兩個單邊 desk | 原文 §2.1 明說 no takers；marketable order 不在模型內 |
| Book 演化 | queue-reactive／latent LOB 直接指定 arrival、cancel、departure（Huang–Lehalle–Rosenbaum 2015; Donier et al. 2015） | 只移動訂單、volume 守恆；連續極限是 Dean–Kawasaki 型守恆 SPDE | 需 Assumption 3.2/3.3 把 \(\gamma\to0\)、\(T_k=\tau_k/\gamma\)、\(\beta=\bar\beta/\gamma^2\) 綁在一起 |
| 記憶來源 | Hawkes kernel 外生擬合，或直接假設冪律 | 每個 cluster 的指數記憶 \(\omega_k(s)\) 依 volume \(w_k\propto T_k^{-\alpha}\) 混合，內生產生 \(t^{-1-\alpha}\) | 需要 many-desk／\(\Omega\to1^+\) 極限才收斂到乾淨冪律；有限 \(K\) 有 log-periodic 修正 |
| 粗糙度 | 假設 order flow 近不穩定，exponent 外生 | \(\|\phi\|=\beta\bar\kappa\) 由 book shape 決定；\(H=\alpha-1/2\) | \(\alpha\le1/2\) 未證明；Binance fitted \(\alpha\) between 0.39 and 0.50，極限不為正 |
| 驗證 | 多為價格／波動率統計 | 模擬 protocol 逐步檢驗＋Binance 六市場＋book 微觀統計 \(\varsigma,\tau,\rho\) | 所讀文本未見樣本外交易、交易成本、PnL |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

**Trick：** 把「哪個 desk 何時動」寫成 Bernoulli 抽樣，讓 step 的條件期望是梯度、條件方差是 mini-batch 噪聲；再把所有 desk 的 step 加總，volume 的隨機流動自然寫成守恆 SPDE——因為每個 desk 只重分配 volume，不憑空增減。

**直覺：** Book 像兩側的流體，做市商只用梯度把自家水滴拉回目標深度；資訊到達時間不同，所以水滴異步、成批地動。聚合後 book 的「形狀」在動，但「總量」不變；mid price 只被兩側靠近 touch 的體積差推動。多種 horizon 讓記憶不是單一指數衰減，而是一疊指數的混合；在對數尺度上近似冪律，於是波動率看起來粗糙。

### 1.3 信息流

- \(M_n\) 依 (2) 移動：\(\theta_n=-c(Z^+_n-Z^-_n)\)，\(M_{n+1}=M_n+h\theta_n\)。
- 所有訂單依 (3) 平移：\(x_{i,j}\leftarrow\max\{x_{i,j}-\sigma_i\theta_n,0\}\)。
- 每個 desk 依 (5) 更新 trend \(P_k\) 與 activity \(U_k\)。
- 以 \(p_k=\min\{p_0+\beta U_k,1\}\) 獨立動作 (8)。
- 動作的 desk 依 (9) 做梯度步，改變 \(\Delta Z_i\)，形成 \(I_n=\sum_{d_i\ \text{acts}}c\,m_i|\Delta Z_i|\) (7)。
- \(I_n\) 進入 \(U_k\)，提高後續動作機率；book 形狀改變下一期 \(Z^\pm\) 與 \(\theta_{n+1}\)。
- 這條回饋鏈給出 kernel \(\phi(s)\)、\(\|\phi\|=\beta\bar\kappa\)，再接到近不穩定 Hawkes 的 \(H=\alpha-1/2\)。

## §2 · 數學層

📌 **Napkin Formula**

單一 desk 的梯度步（(9) 的前一式）：

\[
x_i \leftarrow \max\{x_i-\gamma(x_i-x_i^*)+\sigma_i\Pi P_k,\;0\},\qquad
\lambda_k=\frac{\Pi P_k}{\gamma}.
\]

動作機率與自我激勵（(8)、(5)、(7)）：

\[
p_k=\min\{p_0+\beta U_k,1\},\qquad
U_k\leftarrow U_k+\frac{I_{n-1}-U_k}{T_k},\qquad
I_n=\sum_{d_i\ \text{acts}}c\,m_i|\Delta Z_i|.
\]

連續極限的 volume 方程（Proposition 3.10 後的 Dean–Kawasaki 變體）：

\[
d\mu_t=-\partial_y(\mathcal V\mu_t)\,dt+\frac{\gamma}{2}\partial_y^2(G^2\mu_t)\,dt
-\sqrt{\gamma}\sum_i\partial_y(G\,\mathbf 1_i\,\mu_t)\,dW_i.
\]

**直覺：** 每個 term 都是 \(y\) 的導數，所以總 volume 不變；book 只透過 volume 在深度間的流動改變形狀，mid price 只回應這些流動。原文未給 Big-O 或 wall-clock；這不是神經網絡訓練，沒有 backprop、沒有 optimizer state。

**Loss：**

\[
\mathcal L_i(x)=\frac12\sum_{j=1}^L(x_j-x_{i,j}^*)^2-\sigma_i\lambda_k\sum_{j=1}^Lx_j,
\qquad \lambda_k=\frac{\Pi P_k}{\gamma}.
\]

第一項把掛單拉回 oracle target，第二項讓 desk 順勢傾斜：上升趨勢中 ask 遠離、bid 靠近。\(\gamma\in(0,1)\) 是步長；\(\Pi\ge0\) 控制傾斜強度。

**核心推導鏈：**

1. Lemma 3.1：同一 desk 的訂單噪聲完全相關，不同 desk 獨立；協方差由 \(p_k(1-p_k)\gamma^2\partial_j\mathcal L_i\partial_{j'}\mathcal L_i\) 給出。
2. Theorem 3.5：protocol 與 SDE 在有限 horizon 上離 ODE 的偏差是 \(O(\sqrt{\gamma})\)。
3. Proposition 3.10：volume measure 滿足守恆 SPDE；每個 term 都是 \(y\) 的導數，所以總 volume \(\sum_iLm_i=2\) 常數。
4. Proposition 4.3–4.9：activity 被過去 updates 驅動，kernel 是 \(\phi(s)=\beta\sum_l\kappa_lw_l\omega_l(s)\)；時間尺度上 \(\bar\phi(t)=\|\phi\|\sum_l\varpi_l e^{-t/\tau_l}/\tau_l\)。
5. Proposition 4.10–4.11：horizon 在對數尺度上均勻、\(\Omega\to1^+\)、\(K\to\infty\) 時，\(\bar\phi(t)\sim t^{-1-\alpha}\)，tail exponent 只由 volume 的 \(\alpha\) 決定。
6. Theorem 5.3：\(\|\phi\|\to1^-\) 且 \(\alpha\in(1/2,1)\) 時，activity 的極限粗糙度是 \(H=\alpha-1/2\)；Corollary 5.12 把平方波動率接上同一個 \(H\)。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下是原文 §3.3 的示意算例，屬玩具假設，不是 Binance 校準，也不是本頁新造的實證結果。** 原文用它說明 book 形狀如何推動 mid price。

1. **假設兩側形狀。** Asks 與其 targets 服從 Gamma shape 2、scale 1.5；bids 與其 targets 在 \([1,11]\) 均勻。
2. **寫兩側靠近 touch 的份額。** 由 (1) 的 Laplace 形式，原文給
   \[
   Z^+=\Bigl(1+\frac{1.5}{\ell}\Bigr)^{-2},\qquad
   Z^-=e^{-1/\ell}\frac{\ell}{10}\bigl(1-e^{-10/\ell}\bigr).
   \]
3. **取 \(\ell=1.5\)。** 原文逐字給 \(Z^+=1/4\)、\(Z^-\approx0.077\)。Asks 靠近 touch 的份額較大，所以 asks 壓得較重。
4. **代入 mid price 反應 (2)。** \(\theta_n=-c(Z^+-Z^-)\)，原文給 \(\theta_n\approx-0.17c\)：mid price 往 bids 的方向移動。
5. **看 book 如何被 reshape。** 由 (3)，每個 ask 遠離 mid price \(|\theta_n|\)、每個 bid 靠近 \(|\theta_n|\)；若沒有 desk 回補，兩側壓力平衡的總位移是 \(\Delta^*=(\ell/2)\ln(Z^+/Z^-)\approx0.88\)。但 desk 的梯度步會把訂單拉回 target，所以這個平衡不會靜止。
6. **接上 §4。** 每次 desk 動作改變 \(Z^\sigma\)，進而改變下一次 \(\theta\)；動作大小 \(I_n\) 又進入 \(U_k\)，提高後續動作機率。這就是 kernel \(\phi\) 的來源。

## §3 · 數據層

**模擬數據。** Table B.1 的共用設定：\(c=0.3\)、\(\ell=1.5\)、\(\gamma=0.1\)、\(\Pi=1\)、\(K=24\)、\(T_k=2\cdot1.49^{k-1}\)、\(w_k\propto T_k^{-\alpha}\)、\(N=4\) desks per cluster and side、\(L=6\) orders per desk、target density \(x e^{-x/1.5}/1.5^2\)。不同實驗的步數：Dean–Kawasaki check \(10^5\) steps；response \(600\times3000\)；roughness \(2\cdot10^7\)；single cluster \(2\cdot10^7\)；mid price \(10^7\)；book shape \(3\cdot10^5+2\cdot10^6\) per \(s\)。這些是模擬，不是市場數據。

**Binance 數據。** 六個市場：BTC、ETH、SOL 的 spot 與 perpetual futures against USDT。期間 180 days，從 2026-04-10 到 2026-10-06（§7.1）。每秒聚合 trades；Binance 把同一側、同一價格、100 ms 內的成交合併為一個 aggregated trade。Activity \(N_b\) 用 5 分鐘 bins 的 taker orders 計數。Book snapshots 約每 30 秒，給 0.2% 與 1% 到 5% 的累積 volume。Perpetual futures 的 book 分析用 2026-09-07 到 2026-10-06（Table 9）。摘要稱約 790 million orders。

**tick 的語義。** 原文 §7 說這些市場的 tick 是「\(10^{-6}\) 到 \(10^{-4}\) of the price」，即 tick 相對價格的比值；Table B.4 逐市場給 tick/price：BTC \(1.2\cdot10^{-6}\)、ETH \(3.7\cdot10^{-6}\)、SOL \(8.3\cdot10^{-5}\)。這是無量綱比值，不是「\(10^{-6}\) 到 \(10^{-4}\) 個價格單位」。

**樣本外與容量。** 所讀文本未見 train/test split；H 在六個 30 天 block 上估 mean 與 standard error（Table 7），kernel 用整段 spectrum 擬合（Appendix A.4）。所讀文本未見交易成本、容量曲線、PnL。**時間邊界分列：** 論文發布 2026-11-18；trades 與 spot/perp 數據 2026-04-10 至 2026-10-06；perp book snapshots 2026-09-07 至 2026-10-06。本文不涉及預訓練、tokenizer、適配或 topic 擬合，沒有這些時間邊界。

**代理變數。** Binance 公開數據沒有做市商動作，作者用 taker orders \(N_b\) 當 activity 代理（§7）。這是全篇實證的關鍵測量限制：book 的微觀變化被歸因於做市商移動訂單，但數據沒有 participant ID。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | `https://github.com/rsn870/SGD_lob_rough_vol`（原文 Data and code availability） |
| Checkpoint | 未披露 |
| License | 論文 CC BY-SA 4.0；repo license 未披露 |
| 複現難度 | 高：需重寫 protocol 模擬、Dean–Kawasaki 檢定、Hawkes spectrum fit、Binance 清洗與 H 估計；原文 Appendix A/B 給出多數設定，但 full pipeline 未逐行核對 |
| 數據可得性 | 高：raw Binance data 公開在 `https://data.binance.vision`；processed data 在 repo |
| 硬體需求 | 未披露；模擬步數到 \(2\cdot10^7\)，但未給 wall-clock 或 CPU/GPU |

## §5 · 評測 / Benchmark

**先說結論：這篇沒有 method-vs-SOTA 的表格。** 原文的「baseline」是理論極限與錯誤設定（independent orders、single cluster），不是外部競爭模型。因此下面前SOTA欄一律「未披露」，Δ 留空；把理論極限硬塞進前SOTA欄會製造假的 SOTA 區間。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Binance BTC spot | \(H(N_b)\) | 未披露 | \(0.083\pm0.004\) | |
| Binance BTC perp | \(H(N_b)\) | 未披露 | \(0.085\pm0.004\) | |
| Binance ETH spot | \(H(N_b)\) | 未披露 | \(0.084\pm0.003\) | |
| Binance ETH perp | \(H(N_b)\) | 未披露 | \(0.085\pm0.004\) | |
| Binance SOL spot | \(H(N_b)\) | 未披露 | \(0.080\pm0.003\) | |
| Binance SOL perp | \(H(N_b)\) | 未披露 | \(0.071\pm0.006\) | |
| Binance BTC perp | \(\varsigma\) | 未披露 | 0.37 | |
| Binance ETH perp | \(\varsigma\) | 未披露 | 0.51 | |
| Binance SOL perp | \(\varsigma\) | 未披露 | 0.53 | |
| Binance BTC perp | \(\tau\) (s) | 未披露 | 12 | |
| Binance ETH perp | \(\tau\) (s) | 未披露 | 20 | |
| Binance SOL perp | \(\tau\) (s) | 未披露 | 21 | |
| Binance BTC perp | \(\rho\)（all pairs） | 未披露 | −0.30 | |
| Binance ETH perp | \(\rho\)（all pairs） | 未披露 | −0.21 | |
| Binance SOL perp | \(\rho\)（all pairs） | 未披露 | −0.28 | |
| Binance BTC perp | \(H(Q_b)\) | 未披露 | 0.050 | |
| Binance ETH perp | \(H(Q_b)\) | 未披露 | 0.037 | |
| Binance SOL perp | \(H(Q_b)\) | 未披露 | 0.054 | |

**活動與波動率的一致性（來源數值，非 Δ）：**

| bin | \(H(N_b)\) | \(H(\mathrm{RV}_b)\) | largest \(\lvert H(\mathrm{RV}_b)-H(N_b)\rvert\) | correlation of \(\ln N_b\), \(\ln\mathrm{RV}_b\) |
|---|---|---|---|---|
| 15 min | 0.078–0.088 | 0.080–0.091 | 0.013 | 0.83–0.90 |
| 30 min | 0.094–0.107 | 0.090–0.103 | 0.011 | 0.87–0.93 |
| 60 min | 0.123–0.136 | 0.113–0.130 | 0.018 | 0.88–0.95 |

原文說 At every bin length \(H(\mathrm{RV}_b)\) lies within 0.018 of \(H(N_b)\)；\(\ln N_b\) 與 \(\ln\mathrm{RV}_b\) 的 correlation 為 0.83 到 0.95。

**模擬 vs 理論（不是 SOTA 對比，只列來源數值，不計算 Δ）：**

| 實驗 | Metric | 理論極限（非基線） | 模擬估計 |
|---|---|---|---|
| 24 clusters, \(\alpha=0.6\) | \(H(A_b)\) | 0.10（\(\alpha-1/2\)） | 0.152 |
| 24 clusters, \(\alpha=0.6\) | \(H(V_b)\) | 0.10 | 0.147 |
| 24 clusters, \(\alpha=0.7\) | \(H(A_b)\) | 0.20 | 0.217 |
| 24 clusters, \(\alpha=0.7\) | \(H(V_b)\) | 0.20 | 0.201 |
| single cluster | \(H(A_b)\) | 0.5 | 0.473 |
| single cluster | \(H(V_b)\) | 0.5 | 0.470 |

**哪些是真 capability，哪些是可疑的一致性：**

- **真 capability：** Table 1 的 Dean–Kawasaki 檢定。六個設定下 slope of \(\Delta_n\) on \(\mu_n\) 為 1.021、0.993、1.067、1.007、1.016、1.025，原文說 within 7% of one；variance with \(s_n\) 為 0.999、0.999、0.986、0.995、1.001、0.991，within 1.5% of one；若誤把同 desk 訂單當獨立，variance 變成 1.18 到 4.23。這是模型內部機制的直接檢定，不是外部 SOTA 比較。
- **真 capability（但有限）：** Table 2 的 response ratio 逐項為 1.00、0.99、0.99、1.01、1.01、1.02、1.02；原文說 stays within 2% of one over three decades of lags。Table 3 的 log-periodic bound 在 \(\Omega=1.49\) 時 below \(3.1\cdot10^{-9}\)。這說明 kernel 的冪律近似在模擬中很乾淨。
- **可能是有限 \(T\) 偏差：** Table 4 中 \(\alpha=0.6\) 的 \(H(A_b)=0.152\)、\(H(V_b)=0.147\)，高於極限 0.10；原文說「both exceed it by 0.05 at our finite \(\mathcal{T}\)」。\(\alpha=0.7\) 則 \(H(A_b)=0.217\)、\(H(V_b)=0.201\)，與 0.20 接近。這是有限樣本收斂速度的證據，不是普遍偏差。
- **模擬消融不能外推：** Table 5 的 24 clusters vs single cluster 是特定模擬設定；single cluster 的 \(H(S_b(d))\) 掉到 0.23–0.29，原文說 book recovery 在 24 clusters 時 within memory、single cluster 時 not。不能寫成「所有市場都必須多 horizon 才 rough」。
- **Binance 一致性有裂縫：** Table 9 中 SOL 的 \(H(Q_b)\) 與 \(H(N_b)\) 差 within 0.008，但 BTC 與 ETH 分別低 0.03 與 0.04；\(\ln Q_b\) 與 \(\ln N_b\) 的相關性只有 0.45 到 0.56，遠低於模擬的 0.98 與 0.99。作者自己說 BTC/ETH 的 book 變化 follows the activity less closely than in our model。這是部分市場的一致性較弱，不是全稱成功。
- **最脆的環節：** fitted \(\alpha\) between 0.39 and 0.50，Theorem 5.3 要求 \(\alpha\in(1/2,1)\)，所以極限 \(H=\alpha-1/2\) 在這些 fitted 值下不為正。作者改用有限 \(T=10^5\) 或 \(10^6\) s 的 desk 模擬去比，說數據超過模擬 0.016 與 0.009，95% 區間分別為 \([-0.005,0.036]\) 與 \([-0.012,0.030]\)；\(T=10^4\) s 時 desks 低於數據 0.039，區間 \([0.023,0.055]\)。這是「資料與 memory 超過一天的 desks 一致」，不是「定理已被 Binance 驗證」。把這寫成「H=α−1/2 已實證」是錯的。

**過擬合／前瞻偏差／成本：** 所讀文本未見樣本外測試、未見交易成本、未見執行延遲、未見 PnL。H 的估計移除日內與週內 pattern（§7.1），Table A.2 顯示移除 pattern 使 \(H(N_b)\) 降 0.03 到 0.05；Table A.3 顯示 binning 會把 known \(H=0.08\) 抬到 0.159、0.186、0.233（15/30/60 min）。這些是估計偏差，不是策略過擬合，但足以讓「H 很小」的結論需要敏感度分析。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **沒有 taker。** §2.1 明說 "As our model has no takers"；模型只有做市商移動訂單，沒有其他 arrivals、cancellations、annihilations。這使 book 的 volume 守恆，但也排除 marketable order 與跨邊搬量。
- **理想化 book。** §2.1：每個 desk 固定 \(L\) 張等量訂單，只移動、不新增／取消；真實交易所的 cancel-and-requote 被壓成「同 volume 換距離」。§3.2 的 Assumption 3.3(b) 要求沒有訂單碰到 touch，這在真實 book 不必然成立。
- **Regime 假設。** Assumption 3.2 要求 \(c=\gamma\bar c\)、\(T_k=\tau_k/\gamma\)、\(\beta=\bar\beta/\gamma^2\)；Assumption 3.3 要求 \(p_0+\bar\beta u_k\le\bar p<1\) 與所有距離為正。這些是連續極限的技術條件，不是市場事實。
- **\(\alpha\le1/2\) 未覆蓋。** §8 明說 "We leave extensions such as a limit theorem for tails \(\alpha\le1/2\), tests on data of single orders and markets with lower liquidity to future work." 這是作者自己承認的理論邊界。
- **Binance 沒有做市商動作。** §7 明說 "As the data contain no actions of market makers, we count the taker orders \(N_b\) in each bin \(b\) as the activity." 這是代理變數，不是直接觀測。
- **BTC/ETH 的 book 一致性較弱。** §7.3：SOL 的 \(H(Q_b)=H(N_b)\) within 0.008，但 BTC 與 ETH 的 \(H(Q_b)\) 分別低 0.03 與 0.04；相關性 0.45 到 0.56，低於模擬的 0.98 與 0.99。作者說這些 book 的變化 follows the activity less closely than in our model。
- **tick 平滑的適用範圍。** Remark 3.18：平滑寬度約一個 tick 最接近真實 book；small-tick stocks 的 spread 常超過一個 tick，所以 \(\kappa_\varepsilon\) 接近 one；這段近似最適合 small-tick stocks。

### 6.2 推斷的隱含假設（本頁推斷，非作者自述）

- **Oracle target 外生。** 每個 desk 的 \(x_i^*\) 是未知 oracle 給的目標距離，不随 desk 自己的成交、庫存、逆選擇而變。推斷：這排除了 inventory risk 與 adverse selection 的反饋；真實做市商的 target 會隨庫存與 toxic flow 調整。若把 \(x_i^*\) 內生化，kernel 的 \(\kappa_l\) 與 \(\|\phi\|\) 可能不再是常數。
- **沒有成本與延遲。** 模型沒有 fee/rebate、沒有 cancel 延遲、沒有 queue position 價值。推斷：\(\tau=12\) 到 21 秒的 book 回補時間是「物理」估計，不是「最優」報價頻率；不能用來推論策略盈利能力。
- **Activity 代理的選擇偏差。** 用 taker count 當 activity，等於假設 taker 到達與做市商動作的資訊到達同比例。推斷：在 taker 主導的市場（低流動性、大 tick），這個代理可能失效；原文 §8 也把 lower-liquidity markets 留給未來。
- **H 估計的測量誤差。** Table A.2–A.4 顯示 pattern removal、binning、noise correction 都會改變 \(H\)。推斷：Binance 的 \(H(N_b)\) 0.07 到 0.09 不應被當成「真實粗糙度」的精確點估計；不同 bin 的 \(H(N_b)\) 在 15/30/60 min 為 0.078–0.088、0.094–0.107、0.123–0.136，本身就隨尺度上升。
- **Regime 依賴。** 六個月樣本只覆蓋 2026-04-10 至 2026-10-06 的加密市場；沒有跨越極端波動或流動性枯竭的 regime。推斷：\(\|\phi\|\) 接近 1 的結論可能依賴這段樣本的做市商競爭程度；不能外推到其他年份或傳統股票。
- **不是 alpha。** 全文沒有交易訊號、沒有持倉、沒有成本調整。推斷：任何把 \(H\) 或 \(\|\phi\|\) 直接當因子的人，都必須自己補上預測目標與交易成本；原文沒有提供這些。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Jaisson–Rosenbaum 2015/2016 近不穩定 Hawkes | 記憶 kernel 外生給定；本文由 horizon 分佈與 book shape 內生 | 否 | 理論；本文的 \(H=\alpha-1/2\) 直接借用其極限 |
| El Euch–Fukasawa–Rosenbaum 2018；Jusselin–Rosenbaum 2020 | 從無套利／拆單推 rough vol；本文從做市 SGD 推 | 否 | 理論；本文是替代 microfoundation，不是取代 |
| Cont–Das 2024 rough vol artefact | 質疑 \(H<1/2\) 是估計誤差；本文用 activity 與 RV 雙序列互相印證，但同一估計偏差仍在 | 部分 | 方法論對立；本文引用 [CD24] 並做 noise correction，但未完全反駁 |
| Huang–Lehalle–Rosenbaum 2015 queue-reactive；Donier et al. 2015 latent LOB | 指定 arrival/departure 或 latent supply-demand；本文用梯度規則＋守恆 SPDE | 否 | 不同建模路線；本文的 book 微觀統計 \(\varsigma,\rho\) 可用來區分 |
| Avellaneda–Stoikov 2008；Guéant–Lehalle–Fernandez-Tapia 2013 | 做市最優控制／庫存風險；本文描述性、無最優控制 | 部分 | 互補；本文可為最優控制提供 book 形狀的生成模型 |
| Gess–Gvalani–Konarovskyi 2025 conservative SPDE as SGD | 一般 SGD 粒子極限；本文加 book shape、self-excitation、horizon mixture | 是 | 數學骨架；本文的 Theorem 3.5/Proposition 3.10 建立在其上 |

🎤 **Interview Tip**

- **正確答：** 「這篇的貢獻是把 rough volatility 的微觀來源從交易者拆單／無套利，換成做市商的異步梯度更新；數學上把 SGD 的 conservative SPDE 極限接到 Dean–Kawasaki，再用多 horizon 混合出冪律 kernel，得到 \(H=\alpha-1/2\)。實證是描述性一致性：Binance 六市場的 \(H(N_b)\) 在 0.07 到 0.09，\(H(\mathrm{RV}_b)\) 與它 lies within 0.018，book 的 \(\varsigma<1\)、\(\rho<0\)、回補時間 12 到 21 秒。但 fitted \(\alpha\) between 0.39 and 0.50，定理要求的 \(\alpha>1/2\) 沒有被直接驗證；作者用有限 \(T\) 的 desk 模擬補位。這是生成模型，不是可交易 alpha。」
- **錯答：** 「它證明 Binance 的 Hurst 參數等於 \(\alpha-1/2\)，所以可以用來預測波動率。」——原文 fitted \(\alpha\) between 0.39 and 0.50，極限 \(H\) 不為正；且沒有樣本外、沒有成本、沒有 PnL。
- **錯答：** 「它把做市商當 agent 做 RL 訓練。」——沒有 reward、沒有 policy optimization；desk 只是朝外生 oracle target 做梯度步。

### 7.1 可證偽預測（本頁推斷的檢查點，非原文承諾）

原文 §7 自己列了六個可檢驗關係：\(H(N_b)<1/2\)、\(\|\phi\|\approx1\) 且 \(\phi(s)\propto s^{-1-\alpha}\)、\(H(\mathrm{RV}_b)=H(N_b)\)、\(\operatorname{Corr}(\Delta V_n(d),\Delta V_n(d+1))<0\)、\(\varsigma<1\)、\(H(Q_b)=H(N_b)\)。可以把它們當成證偽條件：

1. **在 2026-10-06 之後的樣本外期間**，若 perpetual futures 的 \(\rho\ge0\) 或 \(\varsigma\ge1\)，則「做市商移動訂單」機制在該市場失效；若 \(\rho\) 仍落在 −0.30 到 −0.16、\(\varsigma\) 仍落在 0.37 到 0.53，則 book 微觀統計的支持延續。
2. **若 single-order 數據（原文 §8 的 future work）顯示** \(H(Q_b)\) 與 \(H(N_b)\) 的差距顯著大於 Table 9 的 within 0.008 到 0.04，則 §6.5 的「book 在 desk memory 內回補」假設不成立。
3. **若某市場 fitted \(\alpha\le1/2\)**，Theorem 5.3 不覆蓋；此時若仍觀測到 \(H>0\)，只能靠有限 \(T\) 模擬解釋，不能說定理已被驗證。這是理論邊界，不是數據失效。
4. **若把 \(x_i^*\) 內生化（加入庫存／逆選擇）後**，\(\|\phi\|\) 不再接近 1 或 kernel 不再是冪律，則本文的「外生 oracle」假設是關鍵失效點。

## §8 · For the Reader

- **因子研究員：** 把 \(H(N_b)\)、\(H(\mathrm{RV}_b)\)、\(\|\phi\|\) 當流動性／波動率 regime 特徵之前，先復現 Table A.2–A.4 的敏感度：pattern removal 會讓 \(H\) 降 0.03 到 0.05，binning 會讓 known \(H=0.08\) 抬到 0.159/0.186/0.233。若你的因子對這些估計選擇不穩定，先解決測量誤差，再談預測力。
- **高頻執行：** \(\tau=12\) 到 21 秒、\(\rho\) 在 −0.30 到 −0.16、\(\varsigma\) 在 0.37 到 0.53 是可用的 book 回補診斷；但它們是描述性統計，不是最優報價頻率。若你要把它接進 execution scheduler，必須自己補 cancel 延遲、fee/rebate、queue position 與逆選擇成本。
- **組合配置：** 這是生成模型，不是違約或流動性風險模型。可用來壓力測試「做市商 horizon 分佈變長／變短」對 rough vol 的影響：在模型中，horizon 的分佈直接決定 \(t^{-1-\alpha}\) 的 tail；但沒有校準到 PnL，不能直接給配置權重。
- **RL／LLM-agent 策略：** 不要把 desk 當 RL agent；這裡沒有 reward，只有朝 oracle target 的梯度步。若要做 agent-based 模拟，這是很好的 book 生成器；若要做策略，需自己加 inventory、adverse selection、market impact 與執行延遲。
- **研究學生：** 先讀 Gess–Gvalani–Konarovskyi 2025 的 conservative SPDE 極限，再讀 Jaisson–Rosenbaum 2016 的近不穩定 Hawkes；本文是把兩者接起來的中間層。親手復現 Table 1 與 Table 2 的模擬（\(10^5\) 與 \(600\times3000\) steps）比直接處理 about 790 million orders 的 Binance 數據便宜得多。

## References

- 原論文：Sai Niranjan Ramachandran, *Algorithmic Market Making Can Lead to Rough Volatility Through the Order Book*, arXiv:2611.24028v1, 2026-11-18. https://arxiv.org/abs/2611.24028v1
- 代碼與 processed data：https://github.com/rsn870/SGD_lob_rough_vol
- Raw Binance data：https://data.binance.vision
- Gess, Gvalani, Konarovskyi, *Conservative SPDEs as fluctuating mean field limits of stochastic gradient descent*, Probability Theory and Related Fields, 2025.
- Jaisson, Rosenbaum, *Limit theorems for nearly unstable Hawkes processes*, Annals of Applied Probability, 2015.
- Jaisson, Rosenbaum, *Rough fractional diffusions as scaling limits of nearly unstable heavy tailed Hawkes processes*, Annals of Applied Probability, 2016.
- El Euch, Fukasawa, Rosenbaum, *The microstructural foundations of leverage effect and rough volatility*, Finance and Stochastics, 2018.
- Jusselin, Rosenbaum, *No-arbitrage implies power-law market impact and rough volatility*, Mathematical Finance, 2020.
- Cont, Das, *Rough volatility: fact or artefact?*, Sankhyā B, 2024.
- Avellaneda, Stoikov, *High-frequency trading in a limit order book*, Quantitative Finance, 2008.
- Guéant, Lehalle, Fernandez-Tapia, *Dealing with the inventory risk: a solution to the market making problem*, Mathematics and Financial Economics, 2013.
- Huang, Lehalle, Rosenbaum, *Simulating and analyzing order book data: the queue-reactive model*, JASA, 2015.
- Donier, Bonart, Mastromatteo, Bouchaud, *A fully consistent, minimal model for non-linear market impact*, Quantitative Finance, 2015.
- Dean, *Langevin equation for the density of a system of interacting Langevin processes*, Journal of Physics A, 1996.
- Kawasaki, *Stochastic model of slow dynamics in supercooled liquids and dense colloidal suspensions*, Physica A, 1994.