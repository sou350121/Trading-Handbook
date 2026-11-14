<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=强化学习 alpha=风险择时 autonomy=全自动黑盒 -->

# Distributional TD3 (quantile critics) + empirical replay environment 解構（Distributional TD3 (quantile critics) + empirical replay environment）

> **來源發布**：2026-11-07 · （無 venue） · arXiv [2611.10956](https://arxiv.org/abs/2611.10956)
> **arXiv 原文**：[Risk-Sensitive Reinforcement Learning for an Options Market Maker: An Empirical Study with Real S&P 500 Options](https://arxiv.org/abs/2611.10956v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在量價表格×日頻波段×強化學習×風險擇時×全自動黑盒；prior gap 是多數 RL hedging 在模型生成市場訓練與評估，本文用 empirical replay 把「訓練市場含什麼 regime」變成可控變量，並以 CVaR95 而非平均 PnL 作目標。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `强化学习` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把訓練市場當作被操縱的實驗變量：用 1996–2017 SPX 期權鏈回放取代 GBM/SABR 模擬器，訓練 quantile-critic TD3 直接優化 CVaR95。核心 trick 是 twin distributional critics 的 element-wise min 加上 empirical replay，讓策略在真實 crash 序列中學到 state-dependent short net delta。對「风险择时」軸★的意義是：尾部保險能力主要來自訓練路徑的 regime 內容，而非單純模型容量。樣本外 2018–2023，CVaR95 由 delta hedging 的 -80.2 改善到 -77.7（+2.5 點，3.1%），但 Q2 +8.9%、Q4 -7.4%，且 63 日 calendar-block bootstrap 區間 [-2.1,+8.1] 含零。

**X-Ray.** 這篇的價值不在「distributional TD3 又贏 delta」——作者自己把 learner 固定、只變訓練環境，這個設計比模型本身更值得讀。它解了兩個舊工程坑：一是 prior work 多在 GBM/SABR 內訓練與評估，regime 由模型決定；二是平均 PnL 目標看不到尾部，quantile critics 讓 CVaR95 直接進 actor 梯度。它打不開的 envelope 也清楚：單一 SPX realized path、日頻、3 bps 固定成本、無融資成本、無市場衝擊、observation 只有 2 維且丢掉 $h_{t-1}$，因此成本不可預期；gamma/vega、多資產、desk-level 容量都在作者自述的 future work 裡。對量化讀者，真正的訊號是「empirical path 誘發的 short net delta tilt」而非通用 alpha：Q2/Q3 買到保護，Q1/Q4 付保費，且 calendar-block 區間含零。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（代表） | 本文 |
|---|---|---|
| 訓練市場 | 多為 model-generated markets，如 GBM、Heston 或其他 calibrated stochastic processes；Cao et al. (2021) 在 GBM 與 SABR 環境研究 call option | empirical replay：SPX 1996–2017 歷史路徑 + SPX mid-quotes 與 OptionMetrics implied volatility/Greeks；另設 GBM、oracle-$\sigma$ GBM、SABR、hybrid 對照 |
| 組合與流量 | 多為 individual options；Mikkilä and Kanniainen (2023) 與 Zernikov (2026) 研究 individual options，無 client flow、無 distributional tail-risk objective | client option portfolio + Poisson arrivals；初始 two short calls + one long call；40 日 episode、每日 rebalance |
| 目標泛函 | Cao et al. (2021) 強調 accounting PnL reward、改善給定 hedging objective（原文未將其目標泛函標為期望 PnL）；Cao et al. (2023) 用 D4PG | quantile critics $N=99$，actor 直接優化 CVaR95/VaR95/VaR99；twin critic element-wise min |
| 比較設計 | 多在單一 simulator 內與 delta 比較 | 固定 portfolio mechanics、arrival、3 bps 成本、40 日 horizon、2018–2023 test；只變訓練環境；hybrid 分離 option valuation |

注意：empirical vs GBM 同時改變價格路徑與期權估值來源，原文明說這比較識別的是「整體訓練環境」而非歷史路徑的孤立效果；hybrid 才是分離 option valuation 的設計。

### 1.2 ⚡ Eureka

**一句話 trick：** 把「訓練市場」當 treatment、把 RL learner 當固定裝置；分位 critic 讓尾部風險直接進 actor 梯度，而不是先學平均再事後看尾部。

**直覺：** 在真實 crash 路徑上，空頭 option book 的尾部損失不是靠「更準的 delta」解決，而是靠「什麼時候少對沖」解決。agent 在負 delta book 上買少一點股票，留下 state-dependent short net delta；下跌時吃到保護，上升時付保費。這正是 Q2/Q3 與 Q1/Q4 的符號翻轉。

### 1.3 信息流 ASCII 圖

```
SPX historical path (t) ──► empirical replay episode (40 trading days)
        │                         │
        │                         ├─ client option arrivals (Poisson λ_high/λ_low)
        │                         ├─ book re-aggregation (mid-quotes, OptionMetrics Greeks)
        │                         ▼
        │                  observation o_t = (S~_t, δ~_t^book)
        │                         │
        │                         ▼
        │                  actor π_φ → a_t ∈ [-0.2, 2.0]
        │                         │
        │                         ▼
        │                  h_t = -a_t δ_t^book
        │                         │
        └────────────► next day S_{t+1}; reward = stock PnL + book PnL - 3 bps cost
                                  │
                                  ▼
                     twin quantile critics Zθ1, Zθ2 (N=99)
                                  │
                                  ▼
                     element-wise min → distributional Bellman target
                                  │
                                  ▼
                     actor update on CVaR95 of min critic
```

## §2 · 數學層

📌 **Napkin Formula**

$$
h_t=a_t\Delta_t^{\mathrm{target}}=-a_t\delta_t^{\mathrm{book}},\qquad a_t\in[-0.2,2.0]
$$

$$
y_n=r_t+\gamma\min\left\{\theta^-_{1,n}(o_{t+1},\tilde a_{t+1}),\ \theta^-_{2,n}(o_{t+1},\tilde a_{t+1})\right\},\qquad n=1,\ldots,N
$$

$$
\mathcal{L}_i(\theta_i)=\frac{1}{N^2}\sum_{m=1}^{N}\sum_{n=1}^{N}\rho^{\kappa}_{\tau_m}\left(y_n-\theta_{i,m}(o_t,a_t)\right)
$$

**直覺：** 每個 critic 輸出 $N=99$ 個固定分位數 $\tau_n=n/100$，用 quantile Huber loss 逼近報酬分布；actor 不看平均，而是看兩個 critic 逐元素取 min 後分布的下尾 CVaR95。Twin-critic min 壓抑稀疏極端分位的樂觀偏差，target critics 每 100 步才 soft update，避免分布目標跟著政策抖動。

**Loss／訓練細節：** $\gamma=1$、$\kappa=1$；target policy smoothing 噪聲 $\sigma_{\mathrm{tps}}=0.2$、裁剪 $c_{\mathrm{tps}}=0.5$；actor 每 $d=2$ 個 critic steps 更新，target actor 每 2 步以 $\tau_{\mathrm{actor}}=0.002$ Polyak 更新，target critics 每 100 步以 $\tau_{\mathrm{critic}}=0.002$ 更新。policy 是 layer-normalized MLP 256-256-256，critic 是 512-512-256 加 quantile head；Adam learning rate $10^{-4}$、batch 256、replay buffer $10^{6}$、constant Gaussian exploration noise $\sigma=0.3$、60,000 episodes、無 early stopping。CVaR95 的 actor 目標是下尾分位數的 unnormalized sum，與 tail mean 差一個正倍數，因此梯度方向一致；CVaR99 在 99 個分位下下尾只有單一分位，與 VaR99 同向，所以不另訓 CVaR99 agent。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下全為**假設／示意**，不是論文實證；論文真實數字留在 §5。

1. 假設某日 client book 的 net delta 為 $\delta_t^{\mathrm{book}}=-20$（示意單位），則 delta-neutral target 是 $\Delta_t^{\mathrm{target}}=+20$。
2. 假設 policy 輸出 $a_t=0.95$（示意），則 $h_t=-0.95\times(-20)=19$；淨 delta 為 $(1-0.95)\times(-20)=-1$，即 under-hedge 負 delta book，留下小幅 net short。
3. 假設次日 $S$ 由 10 跌到 9.8（示意，-2%），book PnL 近似 $\delta_t^{\mathrm{book}}\Delta S=(-20)(-0.2)=+4$；stock PnL 為 $19\times(-0.2)=-3.8$；合計約 $+0.2$。
4. 假設前一日 $h_{t-1}=20$，交易 1 股，成本為 $3\text{ bps}\times1\times9.8\approx0.00294$（示意；3 bps 來自原文，其餘為示意）。
5. 若改用 $a_t=1$：$h_t=20$，stock PnL $-4$，book $+4$，合計 0，成本 0。也就是說，這個 toy 例子裡 under-hedge 用約 0.003 的成本換取下跌時的 +0.2 緩衝；上漲時符號相反，就是 Q1/Q4 保費的 toy 版。

## §3 · 數據層

- **市場與頻率：** S&P 500 index 與 SPX options；daily rebalancing；每個 episode 40 trading days。訓練期 1996–2017；測試期 January 2018 到 August 2023。
- **價格與估值：** empirical replay 直接用歷史 S&P 500 index path；期權帳簿用 SPX mid-quotes，implied volatility 與 Greeks 來自 OptionMetrics，contract-level Greeks 不做重算。hybrid 用相同歷史路徑與 listed contracts，但以 Black–Scholes、constant volatility 0.19、zero rates 估值。GBM 用 $\sigma=0.19$、zero drift、$S_0=10$；SABR 用 volatility of volatility 0.6、$\rho=-0.7$、initial level 0.19。
- **合約與流量：** 初始 book 固定為 three near-the-money S&P 500 options：two short calls + one long call。新 client options 為 call/put，moneyness [0.99,1.01]、20–60 days to maturity，從當日歷史 option chain 以 volume-weighted 抽樣；calls 與 puts 各用 Poisson arrival，$\lambda_{\mathrm{high}}=0.5$、$\lambda_{\mathrm{low}}=0.25$。當 absolute one-day index return 超過 $\sigma/\sqrt{252}$ 時用 high intensity，否則 low；觸發用的是 $t$ 期已實現報酬，不用未來資訊。門檻換算約為 empirical/hybrid 1.01%、GBM/SABR 1.20%、oracle-$\sigma$ GBM 1.32%。
- **成本與會計：** 股票交易成本 3 bps，代表 bid-ask spreads 與 execution slippage；$r_f=0$；到期日或之後以 intrinsic value 結算並移除；40 日 horizon 仍存續的期權按市價 mark，無 terminal liquidation cost；新到期權按 mid-quote 入帳、無 inception PnL。
- **樣本外與容量：** 測試期 2018–2023；1,386 個 window start dates；40 日窗口重疊，作者用 start-date clustered bootstrap 與 63-trading-day calendar-block bootstrap 處理。容量、市場衝擊、資金規模假設：所讀文本未見，圖表／附件待核。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 未披露 | 未披露 | 未披露 | TBD（需 OptionMetrics SPX 期權鏈、Greeks、mid-quotes；實作 quantile critic 與 empirical replay） | SPX 與 OptionMetrics 為商業數據；公開可得性未披露 |

## §5 · 評測 / Benchmark

> 下表「前SOTA」是模板欄名；本文基線是 **delta hedging（$a_t=1$）**，原文未稱其為 SOTA。Δ 固定為「本方法 − 前SOTA」；CVaR95 為負值，越接近 0 越好，Δ 正值代表改善。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| SPX 2018–2023（all regimes，5 runs pooled） | CVaR95（原文未標單位） | -80.2 | -77.7 | +2.5 |
| SPX Q1（HV↑） | CVaR95 | -122.5 | -134.6 | -12.1 |
| SPX Q2（HV↓） | CVaR95 | -201.9 | -184.0 | +17.9 |
| SPX Q3（LV↓） | CVaR95 | -67.6 | -64.6 | +3.0 |
| SPX Q4（LV↑） | CVaR95 | -50.9 | -54.7 | -3.8 |

**解讀：**

- **真 capability 在 Q2/Q3，不在全樣本平均。** Q2 的 +17.9 與 Q3 的 +3.0 是尾部保護的主要來源；Q1 的 -12.1 與 Q4 的 -3.8 是保費。原文的 -9.9%／+8.9%／+4.4%／-7.4% 正是把這些 Δ 除以 delta-hedge tail loss 後的比率。
- **統計顯著性不穩。** start-date clustered 95% CI 為 [+1.1,+3.8]；63-trading-day calendar-block bootstrap 為 [-2.1,+8.1]，含零。五個 training runs 都改善，但這是在單一 realized path 上。
- **成本不是主因。** RL 交易量 1,468 shares vs delta 1,520 shares；平均累積成本 4.43 vs 4.59，差 0.16。作者把成本率從 0 掃到 20 bps，CVaR95 改善維持 1.9 到 4.6 points，所以低換手不能解釋尾部結果。
- **規則對照：** Table 3 中 constant ratio $a=0.97$ 的 CVaR95 變化 +1.1、Q2 +4.9、Q4 -0.0；short offset $m=5$ 為 +1.1、Q2 +8.7、Q4 -2.7；band around $0.95\delta$（$b=5$）為 +1.3；RL 為 +2.5、Q2 +17.9、Q4 -3.8。原文說 best rule 只恢復約一半的增益。RL 比「匹配其平均 net delta 的 fixed short offset」多 1.2 points（start-date-clustered interval [+0.6,+1.9]）；這個 1.2 的比較對象不是 Table 3 的 $m=5$（$m=5$ 列為 +1.1），兩者不可混用。
- **訓練環境對照（Table 4，原文數值，未計算 Δ）：** empirical 平均 net delta -6.1、CVaR95 改善 +3.3%（range [+2.9,+3.8]）；GBM +0.5、-1.0%（[-1.5,-0.1]）；oracle-$\sigma$ GBM +0.1、+0.0%（[-1.2,+0.8]）；SABR -4.3、+1.1%（[+0.3,+1.7]）；hybrid -2.8、+1.9%（[+1.5,+2.2]）。Table 4 的 empirical 是三次 run 子集，不能與 Table 2 五次 run 的 3.1% 互相代稱。hybrid 在五次 run 下 CVaR95 比 empirical 低 0.8 points，start-date interval [-1.6,-0.1]，但 calendar-block interval [-3.8,+1.5]，所以「用市場報價估值」的增量在較保守區間下不顯著。
- **風險目標穩健性（Table 5）：** CVaR95 的 net delta -5.7、improvement +2.5、+3.1%、std. 0.4、range [+2.0,+3.1]；VaR95 為 -2.5、+0.5、+1.0%、0.1、[+0.4,+0.6]；VaR99 為 -10.4、+0.4、+0.4%、1.5、[-0.6,+2.0]。三個目標都留下負 net delta，但 VaR99 只與同尺度 fixed short offset 相當。
- **過擬合／前瞻偏差檢查：** 訓練與測試切分清楚；arrival rule 用當期實現報酬，不用未來；oracle-$\sigma$ GBM 用測試期波動 0.21，作者明說這是 real time 不可得，且它仍只做到 +0.0%，所以「波動水平」本身不是答案。真正的未解是：單一 test path、重疊窗口、以及 empirical vs GBM 同時改價格路徑與估值來源，不能把全部差異歸給單一項。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **單一 out-of-sample 期間（§6）：** 只基於 2018–2023 一條歷史測試；calendar-block 區間含零；作者要求其他市場與更長期評估。
- **未做 TD3 組件消融（§1、§6）：** 作者明說沒有 pinpoint 哪個保守 TD3 設計造成尾部增益，controlled ablation 留給 future research；因此不能把功勞全歸給 quantile critic 或 twin-critic min。
- **環境只是 proof of concept（§6）：** 未含 higher-order risk exposures（gamma/vega）、multiple underlying assets、richer trading costs。
- **client arrival 對稱（§3.2）：** Poisson arrival 對 call/put 與方向對稱，未複刻更細的 volume asymmetry；實際 order flow 的 call/put、上漲/下跌不對稱不在模型內。
- **重疊窗口與 bootstrap（§5.1、§5.2、§5.4）：** 40 日測試窗口重疊；start-date clustered 與 63-day calendar-block 給出不同結論；hybrid 的增量在 calendar block 下不顯著。
- **圖表細節待核：** Figure 1/2 的具體數值未在可讀文本中逐點列出；本頁只使用正文與表列的逐字數值。

### 6.2 推斷的隱含假設（本頁推斷）

- **Regime 依賴是策略的一部分，不是雜訊。** Q2/Q3 賺、Q1/Q4 賠，意味著部署前需要 regime 判斷或明確的風險預算；不能把全樣本 +2.5 當成任何市場狀態下的穩定 alpha。
- **單一路徑與重疊窗口壓縮有效樣本。** 雖然有 1,386 個 window start dates，但 40 日窗口重疊、且都來自同一條 SPX realized path；calendar-block 區間含零就是這個限制的價格。
- **容量與市場衝擊未建模。** 3 bps 是固定百分比成本，沒有 market impact、沒有 liquidity limit、沒有 financing cost；策略容量未披露。把此結果直接推到大型 desk 或 intraday 執行，是未測試的跳躍。
- **POMDP 的 observation 被刻意縮小。** $o_t$ 只有 normalized price 與 book delta，且排除 $h_{t-1}$；原文明說 agent 因此不能完全預期自己的 rebalancing cost。這是正則化選擇，也是失效邊界：更複雜的 book 或成本結構下，2 維 observation 可能不足。
- **Option chain 抽樣有流動性選擇。** 合約以 volume-weighted 從歷史 chain 抽樣，且只取 moneyness [0.99,1.01]、20–60 days；這會偏向流動性較好的合約。所讀文本未見對 listing/liquidity survivorship 的處理，圖表／附件待核。
- **數據泄漏風險主要被控制，但不能完全排除。** 主 empirical replay 用 1996–2017 訓練、2018–2023 測試；arrival trigger 用當期實現報酬；oracle-$\sigma$ GBM 是明示的 look-ahead 控制。OptionMetrics IV/Greeks 的 vintage 與 revision 細節，所讀文本未見。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Delta hedging（$a_t=1$） | 規則基線、無學習 | 是 | CVaR95 -80.2；Q2 -201.9、Q4 -50.9；被 empirical TD3 在 all-regime 與 Q2/Q3 超過，但 Q1/Q4 反向 |
| 六條 hedge-ratio rules | 訓練期選參、可解釋 | 規則公開 | RL 全部超過；best rule 只恢復約一半增益；constant ratio $a=0.97$ +1.1、short offset $m=5$ +1.1、band $b=5$ +1.3，RL +2.5 |
| GBM-trained TD3 | 同一 learner、simulator price path | 演算法公開；代碼未披露 | 平均 net delta +0.5、CVaR95 -1.0%（[-1.5,-0.1]）；未打敗 delta |
| Oracle-$\sigma$ GBM TD3 | 用測試期波動 0.21 的 look-ahead 控制 | 控制組 | 平均 net delta +0.1、CVaR95 +0.0%（[-1.2,+0.8]）；波動水平本身不能解釋 |
| SABR-trained TD3 | stochastic-volatility simulator | 演算法公開；代碼未披露 | 平均 net delta -4.3、CVaR95 +1.1%（[+0.3,+1.7]）；只恢復部分效果，且原文說不優於同尺度 fixed short tilt |
| Hybrid TD3 | 歷史路徑 + Black–Scholes 0.19 估值 | 演算法公開；代碼未披露 | 平均 net delta -2.8、CVaR95 +1.9%（[+1.5,+2.2]）；低於 empirical，calendar-block 下增量不顯著 |
| D4PG／Cao et al. (2023) | distributional actor-critic，但無 twin-critic min 與 delayed updates | 論文公開 | 本文未直接實測 D4PG；只作為方法血緣，不能推論 D4PG 表現較差 |
| Mikkilä and Kanniainen (2023)／Zernikov (2026) | empirical deep hedging、individual options、無 client flow／distributional tail objective | 論文公開 | 未直接可比；本文補的是 portfolio-level client flow 與 CVaR95 |

🎤 **Interview Tip**

- **正確答：** 這篇把訓練環境當處理變量：在相同 portfolio mechanics、3 bps 成本、40 日 episode 下，empirical replay 的 CVaR95 從 delta 的 -80.2 到 RL 的 -77.7（+2.5 點，3.1%）；但 63 日 calendar-block 區間 [-2.1,+8.1] 含零，且 Q1/Q4 反向（-9.9%／-7.4%）。所以它是 regime-conditional 的尾部保險，不是通用 alpha；而且作者沒有做 TD3 組件消融，不能把增益全歸給 quantile critic。
- **錯答：** 「distributional TD3 比 delta 好 3.1%，所以 RL 在所有市場打敗 delta」；或把 delta hedging 稱作 SOTA；或把「RL 比匹配平均 net delta 的 fixed short offset 多 1.2 points」與 Table 3 的 $m=5$（+1.1）混為一談；或說 GBM 基線是 -80.2。

### 7.1 可證偽預測（帶日期）

若 2024–2026 的樣本外期間以 Q1/Q4 型上升或低波動行情為主，依本文的 regime 條件性，CVaR95 改善應縮小甚至轉負；若在 2026-12-31 前出現的公開後續測試（其他指數或延長樣本）仍只在 Q2/Q3 型下跌高波動 regime 見到增益，則支持「保險溢價」解讀而非通用尾部 alpha。反之，若在沒有高波動下跌的期間仍穩定出現 all-regime 正改善，則本文的 regime 依賴結論需要修正。

## §8 · For the Reader

- **因子研究員：** 不要直接把 CVaR95 改善當 alpha。先拆 regime：Q2/Q3 是保護、Q1/Q4 是保費；再把 net delta tilt 當風險因子曝露看，而不是當預測訊號。重疊窗口與 calendar-block 區間是必查項。
- **高頻執行／做市：** 這是日頻、40 日 horizon、3 bps 固定成本、單一 SPX 的結果；沒有 queue、latency、market impact、inventory limit。要推到 intraday 或大容量 desk，需另做成本與衝擊模型，不能直接套用。
- **組合配置：** 把它看成尾部保險策略：在下跌高波動 regime 買保護，在上升市場付 premium（平均 PnL -0.65 per episode）。配置上需要 regime overlay 或風險預算，而不是全天候開著。
- **RL 策略／LLM-agent 研究者：** 訓練環境是 first-class hyperparameter。這篇的實驗設計（固定 learner、只變環境、hybrid 分離估值）比模型細節更可複用；但 observation 只有 2 維、POMDP memoryless、成本不可預期，這些都是可改進的點。
- **研究學生：** 最小可復現實驗是 empirical vs GBM vs hybrid 三環境對照；需要 OptionMetrics SPX 期權鏈、Greeks、mid-quotes，以及 40 日重疊窗口的 bootstrap。先做 component ablation，再談更大的 desk-level 結論。

## References

- 原論文：Chen, Cheng, Hull, Li, Poulos, Shi, Veneris, Wang. *Risk-Sensitive Reinforcement Learning for an Options Market Maker: An Empirical Study with Real S&P 500 Options.* arXiv:2611.10956. [https://arxiv.org/abs/2611.10956](https://arxiv.org/abs/2611.10956)
- Lineage：Cao et al. (2021) deep hedging of derivatives using RL；Cao et al. (2023) gamma/vega hedging with D4PG；Buehler et al. (2019) deep hedging；Kolm and Ritter (2019) dynamic replication and hedging；Mikkilä and Kanniainen (2023) empirical deep hedging；Zernikov (2026) what deep hedging learns.
- 演算法：Dabney et al. (2018) quantile regression distributional RL；Fujimoto et al. (2018) TD3；Barth-Maron et al. (2018) D4PG；Bellemare et al. (2017) distributional RL.
- 市場微結構與期權：Avellaneda and Stoikov (2008)；Ho and Stoll (1981)；Gârleanu et al. (2009)；Hagan et al. (2002) SABR；Black and Scholes (1973)；Merton (1973)；Leland (1985).