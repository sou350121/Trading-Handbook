<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=强化学习 alpha=组合执行优化 autonomy=全自动黑盒 -->

# PPO + Heston 解構（PPO + Heston）

> **來源發布**：2026-11-24 · （無 venue） · arXiv [2611.32790](https://arxiv.org/abs/2611.32790)
> **arXiv 原文**：[Delta-Vega Hedging with Deep Reinforcement Learning: From Simulation to Reality](https://arxiv.org/abs/2611.32790v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格／日頻波段／強化學習／組合執行優化／全自動黑盒」。它把 RL hedging 從多數單工具 Delta 推向成本感知的雙工具 Delta-Vega-like 對沖，並用 observable-only state 做 Heston→SPX 的 sim-to-real。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `强化学习` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ①用 PPO 在 Heston 風險中性仿真中同時交易標的與長天期 ATM call，reward 是 mean-variance 的 one-step P&L。②核心 trick 是 state 只放 \(S,C,\tilde C\)、持倉與到期時間，完全不餵 instantaneous variance；兩個期權價格與期限隱含 ATM implied vol 與 term structure。③這使策略可在 SPX 2019–2022 直接零樣本部署，對照 practitioner BS Delta／Delta-Vega；成本上升時 PPO 以更平滑交易換取成本節省。④全樣本 SPX 上 PPO total P&L std 0.484 對 Delta-Vega 0.560（表頭單位：percent of \(S_0\)），但 mean P&L 差異不顯著，作者明說 15 個獨立窗口不足以支撐強統計主張。

**X-Ray.** 五軸上，它把 alpha 軸壓在組合執行優化，而不是預測；RL 只是求解器，Heston 是訓練環境。舊坑有三：單工具 Delta RL 留下 vega 暴露；成本感知 Delta-Vega 的 no-trade region 被視為 open challenge；Heston 訓練若把 instantaneous variance 放進 state，部署時還要 VIX proxy 或模型因子。本文用 observable-only state 把 proxy 需求拆掉。預測它打不開的 envelope：單一標的、ATM European call、long-dated hedge option、proportional half-spread、無 market impact、日頻或日內 5 步；多資產、路徑依賴、exotic、market impact 容量不在內。對量化讀者，應讀成「Heston 先驗＋成本平滑」的 policy distillation，不是新風險溢價；2021 低波動年 Delta-Vega objective 0.13 對 PPO 0.02，已提示 regime 依賴。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作 / 常見做法 | 本文改動 | 讀者要點 |
|---|---|---|---|
| 對沖工具 | RL 多為 single-instrument Delta；少數用第二工具者，要麼對 Gamma 基準，要麼在模擬中做 Vega proof of concept | 同時交易 underlying 與一個 longer-dated ATM call，兩者都計 proportional costs | 把「Vega 對沖」從模擬概念推到有成本、有實盤基準的雙工具政策 |
| 狀態 | [18] 在 Heston 訓練時把 instantaneous variance 放入 observables，部署時用 VIX proxy；[17] 把 implied-vol 模型因子放入 state | state 只含 \(S_i,C_i,\tilde C_i,\mathcal H_i^S,\mathcal H_i^{\tilde C},\tau_i,\tilde\tau_i\)，不含 \(\nu_t\) | 讓 sim-to-real 不依賴 volatility proxy；波動信息由兩個期權價格隱式攜帶 |
| 驗證 | 多數只在合成環境測試；[29] 比較 simulation-trained 與 empirically-trained | 先做 frictionless Heston 對 semi-analytical Delta-Vega，再做成本網格，最後直接轉 SPX 2019–2022 對 practitioner BS | 把「in-model 最優」與「out-of-model 可用」分開驗證 |

### 1.2 ⚡ Eureka

**不觀測 \(\nu\)，讓 \(C\) 與 \(\tilde C\) 的價格／期限自己編碼 volatility 信息。** 直覺：Heston 的 instantaneous variance 不可交易也不可直接觀測，但同一標的的兩個期權價格是 \(\nu\) 的可交易投影；agent 只要學「在這兩個價格坐標下的持倉映射」，sim-to-real 就不需要 VIX proxy 或模型校正。

### 1.3 信息流 ASCII 圖

Heston simulator: \((S_t,\nu_t)\rightarrow C_t,\tilde C_t\)  
　　　　　　　│ 只取市場可觀測量；\(\nu_t\) 不進 state  
　　　　　　　▼  
state \(s_i=(S_i,C_i,\tilde C_i,\mathcal H_i^S,\mathcal H_i^{\tilde C},\tau_i,\tilde\tau_i)\)  
　　　　　　　│  
　　　　　　　▼  
PPO actor \(\pi_\theta(a\mid s)\) → action \(a_i=(\mathcal H_{i+1}^S,\mathcal H_{i+1}^{\tilde C})\)  
　　　　　　　│  
　　　　　　　▼  
one-step P&L（含 asset／option costs）→ reward \(=\mathrm{P\&L}_{i+1}-\lambda \mathrm{P\&L}_{i+1}^2\)  
　　　　　　　│  
　　　　　　　▼  
critic／GAE → PPO clipped update；部署時同一 state 映射直接吃 SPX 報價

## §2 · 數學層

📌 **Napkin Formula**：

\[
\max_{\{\mathcal H_i^S,\mathcal H_i^{\tilde C}\}}\ \mathbb E[w_n]-\lambda\,\mathbb V[w_n],
\qquad
r_{i+1}=\mathrm{P\&L}_{i+1}-\lambda\,\mathrm{P\&L}_{i+1}^{2}.
\]

\[
\mathrm{P\&L}_i
=\underbrace{C_{i-1}-C_i}_{\text{option}}
+\underbrace{\mathcal H_i^S(S_i-S_{i-1})}_{\text{asset}}
+\underbrace{\mathcal H_i^{\tilde C}(\tilde C_i-\tilde C_{i-1})}_{\text{hedge option}}
-c\bigl|S_{i-1}(\mathcal H_i^S-\mathcal H_{i-1}^S)\bigr|
-\tilde c\bigl|\tilde C_{i-1}(\mathcal H_i^{\tilde C}-\mathcal H_{i-1}^{\tilde C})\bigr|.
\]

**直覺**：這不是直接學 Heston 的 \(\Delta,\mathcal V\)，而是把「成本—方差」寫進每步 reward，讓 PPO 在持倉空間 \(\mathcal A=[h_l,h_u]^2\) 裡學一個 cost-aware 的 Delta-Vega-like 映射。Frictionless 時 \(\gamma=0\)，問題退化成 \(n\) 個獨立 one-step static mean-variance；有成本時 \(\gamma=0.99\)。**複雜度**：推論 FLOPs 未披露；網路與訓練規模見 Appendix A Table 6：synthetic actor \(128\times128\)、critic \(128\times128\times128\times128\)，sim-to-real actor \(256\times256\)、critic \(256\times256\times256\times256\)；learning rate \(3\times10^{-4}\)、clip \(0.1\)、critic loss weight \(0.5\)、\(\lambda_{\text{GAE}}=0.99\)、epochs 25、batch 128／256、iterations 300／1,200。PPO 不用 entropy bonus。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下全部是**假設／示意**，不是論文實證；只沿用原文的訓練尺度數字，不重算論文結果。論文真實數字留在 §5。

1. **假設/示意**：取原文訓練尺度 \(S_0=100\)、\(K=100\)、主期權 \(T=60\) days、對沖期權 \(\tilde T=120\) days、\(c=1\) BPS、\(\tilde c=1\%\)、\(\lambda=1\)。  
2. 在 \(t_i\) 觀察 \(s_i=(S_i,C_i,\tilde C_i,\mathcal H_i^S,\mathcal H_i^{\tilde C},\tau_i,\tilde\tau_i)\)；這裡沒有 \(\nu_i\)。  
3. actor 輸出 \(a_i=(\mathcal H_{i+1}^S,\mathcal H_{i+1}^{\tilde C})\)，落在 \([0,1]^2\)（原文 \(h_l=0,h_u=1\)）。  
4. 若示意上「完全不調倉」，即 \(\Delta\mathcal H=0\)，則 asset cost 與 option cost 兩項都為 0；若調倉，option cost 以 \(\tilde c\,|\tilde C_i(\mathcal H_{i+1}^{\tilde C}-\mathcal H_i^{\tilde C})|\) 計。  
5. 把三個 P&L 腿與兩個成本項代入上式得到 \(\mathrm{P\&L}_{i+1}\)；reward 為 \(\mathrm{P\&L}_{i+1}-\lambda \mathrm{P\&L}_{i+1}^2\)。示意例不輸出數值，因為 \(\tilde C_i\) 與當步價格變動未在原文給定。  
6. PPO 用 GAE 與 clipped ratio 更新；frictionless 時每步只看下一期 P&L，有成本時持倉平滑與成本節省才進入跨步權衡。真實結果見 §5 表。

## §3 · 數據層

- **合成訓練／測試**：Heston 風險中性路徑；synthetic 主實驗用單一參數集 \(\kappa=1.03,\bar\nu=0.08,\eta=1.0,\nu_0=0.04,\rho=-0.65\)，\(S_0=100\)、\(K=100\)、\(T=60\) days、\(\tilde T=120\) days；每天 5 個價格、\(n=300\) steps、\(\delta t=1/(5\cdot252)\) years；\(N_{\text{train}}=40,000\)、\(N_{\text{test}}=10,000\)。  
- **成本網格**：baseline \(c=1\) BPS、\(\tilde c=1\%\)；另加 \(\tilde c=5\%\) 與 \(c=5\) BPS、\(\tilde c=5\%\)；\(\lambda=1\)。  
- **Sim-to-real 訓練集**：以 2018 年 251 個 SPX 交易日逐日校準 Heston，\(N_{\text{train}}=360,000\) episodes，平均分配到 251 個參數集；八個 maturity buckets \([60,100,145,195,250,305,360,420]\) trading days，主／對沖期權 maturity gap 固定 130 trading days；每日調倉、60 trading days episode；\(c=1\) BPS、\(\tilde c=1\%\)、\(\lambda=0.5\)。  
- **實證測試**：SPX 期權日資料 2018 年 1 月至 2022 年 7 月；2018 整年保留校準，2019–2022 為 out-of-sample；最終 1,193 episodes、15 個獨立時間窗；60 trading days 非重疊窗口；主合約 inception 時 \(|\log(K/S_0)|\le0.03\)，主／對沖期權 maturity 上限分別 600／800 calendar days；每日報價缺失最多橋接 2 天，最終樣本 0.2% 日觀測被橋接。  
- **2018 校準面流動性篩選**：open interest 至少 500 或 volume 至少 10、relative spread 不大於 15%、每 expiry 保留 25 個最流動 strikes，平均每日約 220 個可用 quotes。  
- **樣本外與容量假設**：只有 proportional half-spread，無 commission、無 market impact、無 time-dependent spread；capacity 未驗證。episodes 在同一時間窗內共享 SPX 路徑，作者明說不應視為獨立抽取；year-level 約四個窗口，統計力有限。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 所讀文本未見；圖表／附件待核 |
| Checkpoint | 所讀文本未見；圖表／附件待核 |
| License | 所讀文本未見；圖表／附件待核 |
| 複現難度 | TBD；PPO 超參在 Appendix A Table 6 可定位，但完整訓練腳本未見 |
| 數據可得性 | SPX 期權 quotes 來自 DeltaNeutral LLC「Historical Option Data – Level 2」；Heston 合成資料由作者自行生成；商業資料授權與 snapshot 未披露 |

## §5 · 評測 / Benchmark

> 表頭沿用模板的「前SOTA」，但來源沒有 SOTA 級比較；實際填的是 Delta-Vega／Delta-only 基線。Δ 固定為「本方法 − 前SOTA」；Std 越低越好，負 Δ 代表降低；Objective 越高越好，正 Δ 代表改善。表列數值若原表以表頭標單位，則不逐格補 `%`。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| Heston 低摩擦（基線=Delta-Vega；\(c=1\) BPS, \(\tilde c=1\%\)） | Total P&L mean（% of \(S_0\)） | -0.170 | -0.080 | +0.090 pp |
| Heston 低摩擦（基線=Delta-Vega） | Total P&L std（% of \(S_0\)） | 0.249 | 0.429 | +0.180 pp |
| Heston 低摩擦（基線=Delta-Vega） | Objective | -0.23 | -0.26 | -0.03 |
| Heston 低摩擦（基線=Delta-Vega） | Asset cost（% of \(S_0\)） | 0.036 | 0.025 | -0.011 pp |
| Heston 低摩擦（基線=Delta-Vega） | Option cost（% of \(S_0\)） | 0.132 | 0.055 | -0.077 pp |
| Heston 中摩擦（基線=Delta-Vega；\(c=1\) BPS, \(\tilde c=5\%\)） | Objective | -0.87 | -0.51 | +0.36 |
| Heston 中摩擦（基線=Delta-Vega） | Total P&L std（% of \(S_0\)） | 0.419 | 0.482 | +0.063 pp |
| Heston 中摩擦（基線=Delta-Vega） | Option cost（% of \(S_0\)） | 0.659 | 0.251 | -0.408 pp |
| Heston 高摩擦（基線=Delta-Vega；\(c=5\) BPS, \(\tilde c=5\%\)） | Objective | -1.07 | -0.56 | +0.51 |
| Heston 高摩擦（基線=Delta-Vega） | Total P&L std（% of \(S_0\)） | 0.482 | 0.460 | -0.022 pp |
| Heston 高摩擦（基線=Delta-Vega） | Asset cost（% of \(S_0\)） | 0.180 | 0.096 | -0.084 pp |
| Heston held-out sim-to-real（\(n=10,000\)；基線=BS Delta-Vega） | Total P&L std（% of \(S_0\)） | 0.582 | 0.481 | -0.101 pp |
| Heston held-out sim-to-real（基線=BS Delta-Vega） | Objective | -0.20 | -0.16 | +0.04 |
| Heston held-out sim-to-real（基線=BS Delta-Vega） | Option cost（% of \(S_0\)） | 0.020 | 0.036 | +0.016 pp |
| S&P 500 2019–2022（\(n=1193\)；基線=BS Delta-Vega） | Total P&L std（% of \(S_0\)） | 0.560 | 0.484 | -0.076 pp |
| S&P 500 2019–2022（基線=BS Delta-Vega） | Objective | -0.20 | -0.07 | +0.13 |
| S&P 500 2019–2022（基線=BS Delta-Vega） | Total P&L mean（% of \(S_0\)） | -0.047 | 0.051 | +0.098 pp |
| S&P 500 2019–2022（基線=BS Delta-Vega） | Asset cost（% of \(S_0\)） | 0.010 | 0.007 | -0.003 pp |
| S&P 500 2019–2022（基線=BS Delta-Vega） | Option cost（% of \(S_0\)） | 0.050 | 0.052 | +0.002 pp |
| S&P 500 2019–2022（\(n=1193\)；基線=BS Delta-only） | Total P&L std（% of \(S_0\)） | 1.174 | 0.484 | -0.690 pp |
| S&P 500 2019–2022（基線=BS Delta-only） | Objective | -1.03 | -0.07 | +0.96 |
| S&P 500 2019–2022（基線=BS Delta-only） | Total P&L mean（% of \(S_0\)） | -0.339 | 0.051 | +0.390 pp |

**解讀**：低摩擦下 PPO objective -0.26 對 Delta-Vega -0.23，std 0.429 對 0.249；這不是 capability 碾壓，而是 benchmark 仍近最優時的小幅折衷。中／高摩擦下 PPO objective -0.51 對 -0.87、-0.56 對 -1.07；option leg cost 0.251 對 0.659、0.252 對 0.659，P&L std 0.482 對 0.419、0.460 對 0.482，mean 也較高（-0.277 對 -0.697、-0.350 對 -0.841）。作者論述重點是成本與波動權衡，不能把優勢全歸因於單一成本項。SPX 全樣本（表頭單位 percent of \(S_0\)）PPO std 0.484 對 Delta-Vega 0.560、Delta 1.174；objective -0.07 對 -0.20、-1.03；mean 0.051 對 -0.047、-0.339，但作者明說 mean 不顯著。尾部：PPO skewness 0.91 對 Delta-Vega -2.31、Delta-only -3.05；\(\text{CVaR}_{95}\) 0.97% 對 1.84%、4.78%。真 capability 在 P&L dispersion 與成本平滑；可能被環境塑造的是 overexposure：PPO 平均持 0.77 單位 hedging option 對 Delta-Vega 0.70，持 0.12 單位 underlying 對 0.18；Heston 模型 hedge 平均 0.96 對 practitioner 0.70，方向一致。前瞻偏差：所讀文本未見明確 lookahead；2018 校準、2019–2022 測試，但 episodes 共享路徑、有效窗口 15 個，統計弱。成本：benchmark 是 cost-unaware，RL 是 cost-aware，這會放大相對優勢，不能讀成 pure alpha（本頁推斷）。2021 低波動年 Delta-Vega objective 0.13 對 PPO 0.02，2020 高波動年 Delta std 2.18% 對 PPO 0.72%、Delta-Vega 1.03%，來源明說優勢集中高波動，不是全市場。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

- **Section 6**：日頻與有限時間跨度限制獨立 episodes 數量，year-level 統計主張不強；更長、更高頻資料才能嚴謹評估跨 regime out-of-sample。  
- **Section 5.3.1／5.3.3**：同一時間窗內 episodes 不獨立，year-level 約四個窗口；15 個獨立時間窗下 mean P&L 與零不可區分，mean-variance objective 需謹慎讀。  
- **Section 6**：Heston 只是可處理的模擬器之一；rough 或 path-dependent volatility 可能提升 sim-to-real 保真度。  
- **Section 6**：只做 plain vanilla European options 與單一 mean-variance reward；exotic payoff、tail-sensitive objective 未測。  
- **Section 6**：proportional cost model 抽離 market impact 與 time-dependent spreads；更真實 microstructure 或直接使用 observable bid-ask 是下一步。  
- **Section 5.3.1**：state 不含 strike \(K\)，因此未探索 OTM／ITM contracts；作者引 [42] 指出這是可走但未走的路。  
- **Section 4.4／5.3.3**：持倉 bounds \(h_l=0,h_u=1\)；sim-to-real 的 Heston model hedge ratio 可能超過 1，需 clipping。  
- **Section 5.1**：到期前 ATM Gamma 高、最優持倉變化尖銳，且此類 state 每 episode 只拜訪一次，是 policy 最難學的區域。

### 6.2 推斷的隱含假設（本頁推斷）

- **Regime 依賴**：優勢集中 2020 高波動，2021 低波動 underperform；推斷它不是全市場穩定 alpha，而是高 vol／高 vega 離散度環境的保險式改善。  
- **Risk-neutral 訓練**：作者明說訓練在 \(\mathbb Q\) 下，學不到 drift／risk premia；本頁推斷 SPX 上 mean 0.051% 不應讀成 alpha，可能是 real-world drift／risk premia 或樣本偶然。  
- **成本／容量**：只建模 half-spread，沒有 market impact、commission、time-dependent spread；推斷容量與大單可執行性未驗證，不能把回測成本節省直接外推到實盤。  
- **Sim-to-real 分佈偏移**：2018 校準、2019–2022 測試；Heston 錯配會以 residual exposures 形式轉移到 SPX，overexposure 與訓練模型一致正是證據。  
- **樣本選擇**：要求每個交易日有效報價、流動性與 ATM 篩選、maturity caps；推斷這會偏向流動合約，**但不是把缺失輔助數據直接稱作生存者偏差**；需另有完整合約面板才能判斷。  
- **統計推論**：15 個獨立窗口、year-level 約四個；推斷任何「年度最優」都應視為描述性，不是顯著擊敗。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Deep Hedging [5] | 單工具 empirical Delta；Heston 雙工具只到 variance swap proof of concept | 未披露 | Quantitative Finance 2019 |
| Gamma/Vega DRL [7] | Gamma＋Vega，SABR 模擬；Delta 腿假設精確且零成本 | 未披露 | Frontiers in AI 2023 |
| [17] implied-vol surface | straddle＋short-dated option、Gamma 基準；state 用模型因子 | 未披露 | arXiv 2025 |
| Empirical Deep Hedging [29] | 比較 simulation-trained 與 empirically-trained；本文以此對照 sim-to-real 路線 | 未披露 | Quantitative Finance 2023 |
| Classical no-trade region [27,41] | Delta only、成本感知、無學習；Delta-Vega 擴展被稱 open challenge | N/A | Published |
| Generative model hedge [11] | 生成 IV surface 後算 hedge ratios；不是學 policy | 未披露 | Annals of OR 2025 |

🎤 **Interview Tip**  
**正確答**：這篇的貢獻是 observable-state sim-to-real＋cost-aware 雙工具對沖；PPO 在 SPX 的優勢主要是 P&L dispersion 與 tail，不是 mean alpha；低摩擦下它沒有碾壓 Heston Delta-Vega，benchmark 還是 cost-unaware。  
**錯答**：「PPO 發掘了 vega risk premium」、「zero-shot 所以沒有 distribution shift」、「SOTA 被大幅超越」、「成本感知 Delta-Vega 已對齊」——這些都把 policy distillation、成本平滑與統計弱的 mean 混成 capability。

### 7.1 可證偽預測（下一個公開版本，日期 TBD）

1. 若加入 cost-aware Delta-Vega（no-trade region）基線，低摩擦 objective 差距會縮小；若 PPO 仍明顯勝出，才支持「學到超越成本平滑的 Vega 信息」。  
2. 若擴到更高頻或跨標的 panel，2020 高波動集中優勢會衰減，mean P&L 仍不顯著；否則才算 regime-robust。  
3. 若 state 加入 \(\nu\) 或 implied-vol surface factors，overexposure 會改變；若不變，則 overexposure 主要來自成本／約束而非 Heston 訓練分佈。  

## §8 · For the Reader

- **因子研究員**：不要把 SPX mean 0.051% 當 alpha；把它當成本感知 Vega 對沖的 benchmark，檢查 residual Delta／Vega 與 2020／2021 regime 分解。  
- **高頻執行**：成本模型只有 half-spread，沒有 market impact；不可直接把表列 total P&L std 0.484（單位：percent of \(S_0\)）搬到日內大單，需補 queue／impact／spread dynamics。  
- **組合配置**：雙工具確實把 Delta-only 的 dispersion 壓低（來源說 52% 與 59%），但尾部集中 2020，2021 underperform；配置時按流動性與 regime 縮放。  
- **RL 策略**：observable-only state 是可借的工程手法；但訓練環境會塑造 policy，必須用 cost-aware 基線、多 seed、year-level 檢定，不能只看單一 objective。  
- **研究學生**：先讀 Table 1／4／5 與 §5.3.1 的 15 個窗口，再看 Appendix A Table 6；這篇的「成功 transfer」是描述性，不是強統計。  

## References

- Colombi, M., Di Francesco, M., Rossi, P. “Delta-Vega Hedging with Deep Reinforcement Learning: From Simulation to Reality.” arXiv:2611.32790, 2026. https://arxiv.org/abs/2611.32790v1  
- Heston, S. “A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options.” Review of Financial Studies, 1993.  
- Schulman, J. et al. “Proximal Policy Optimization Algorithms.” arXiv:1707.06347, 2017.  
- Kolm, P., Ritter, G. “Dynamic Replication and Hedging: A Reinforcement Learning Approach.” Journal of Financial Data Science, 2019.  
- Buehler, H. et al. “Deep Hedging.” Quantitative Finance, 2019.  
- Cao, J. et al. “Gamma and Vega Hedging Using Deep Distributional Reinforcement Learning.” Frontiers in AI, 2023.  
- Mikkilä, O., Kanniainen, J. “Empirical Deep Hedging.” Quantitative Finance, 2023.  
- François, P. et al. “Deep Hedging with Options Using the Implied Volatility Surface.” arXiv:2504.06208, 2025.  
- Cont, R., Vuletić, M. “Data-Driven Hedging with Generative Models.” Annals of Operations Research, 2025.  
- DeltaNeutral LLC. “Historical Option Data – Level 2.” End-of-day historical options data, 2026. https://historicaloptiondata.com/data-structure-greeks/