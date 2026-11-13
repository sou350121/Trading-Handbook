<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=强化学习 alpha=风险择时 autonomy=人机协同可解释 -->

# Gated Conformal Semantic Risk Control (Gated-CSRC) 解構（Gated Conformal Semantic Risk Control (Gated-CSRC)）

> **來源發布**：2026-11-10 · （無 venue） · arXiv [2611.14699](https://arxiv.org/abs/2611.14699)
> **arXiv 原文**：[Conformal Risk Certificates for Portfolio Optimization Under Uncertainty: A Gated Reinforcement-Learning Approach](https://arxiv.org/abs/2611.14699v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點是文本另类 × 日频波段 × 强化学习 × 风险择时 × 人机协同可解释；它解的是 RL 組合缺乏 pre-trade、可重放、有限樣本覆蓋的風險憑證，而不是再造一個 LLM alpha 模型。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `强化学习` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把公告／新聞映射成 schema-validated semantic risk state，經分桶共形校準成 factor-shock ellipsoid，再用非對稱 SOC gate 審查 PPO 提案。核心 trick 是 LLM 不做收益預測、只做風險態；gate 只在提案超出認證 worst-case semantic loss budget 時介入。對「文本另类 + 日频波段 + RL + 风险择时 + 人机协同可解释」的意義，是把可審計性放在執行前而非事後解釋。九檔美國 Select Sector SPDR ETFs、2019–2023 OOS：Raw PPO 12.7% / 4.05%，Gated CSRC 11.7% / 0.00%，McNemar p<10^{-15}；Sharpe 與 mean return 差異未檢出（p=0.493 / p=0.175）。

**X-Ray.** 這篇的價值不在 alpha，而在把「文本風險」從軟性特徵改成可重放的可行域。它避開兩個坑：LLM 校準不足，以及事後解釋無法證明事前合規；schema state 限制語言模型權限，分桶共形半徑給出有限樣本目標，SOC gate 只修剪越界提案。Envelope 很清楚：日頻、九檔 ETF、現金零報酬、無空頭／市場衝擊成本、local approximate exchangeability；對盤中新聞與更高頻 regime 切換，作者自己說不適合。對量化讀者，應把它讀成 execution-layer safety premium 的測量，而非收益模型；Equal Weight 的 12.06% 也提醒它不支配靜態配置。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作典型 | Gated-CSRC | 讀者含義 |
|---|---|---|---|
| LLM 的權限 | 文本→排名、方向、sentiment score、收益預測 | 文本→schema-validated semantic risk state，不選股、不預測報酬 | 把語言模型從 alpha 生成器降級為風險感測器，讓錯誤模式可審計 |
| 不確定集來源 | 歷史 moments／covariance 的 robust ellipsoid | semantic shape matrix + rolling bucketed conformal radius | 文本改變 shock center 與 cross-sector geometry；半徑由 realized nonconformity 決定 |
| 執行層 | risk-sensitive reward 或每日 unconditional projection | asymmetric SOC gate：合規提案原樣執行，越界才解 projection | 把「訓練時風險懲罰」與「執行時可行性否決」分開，降低無謂干預 |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

一句話：把 LLM 的輸出權限從「收益／排名」降到「schema 風險態」，用分桶共形半徑把文本風險放大成可認證的橢球，最後只讓越界提案被 SOCP 投影。

直覺：LLM 的軟性信心不是 coverage；共形校准把「過去同語義桶的實際衝擊誤差」變成半徑，讓文本只透過 feasible action set 影響交易，而不是透過一個不可審計的 score 影響期望報酬。

### 1.3 信息流 ASCII 圖

```text
timestamped docs D_t ──► Llama-3-Instruct (schema JSON) ──► z_t / x_t
                                                        │
                                      m_t^sem, Σ_t^sem ◄─┘
rolling 504-day nonconformity + semantic bucket ──► ρ_t
                                                        │
                                         certificate C_t=(m,Σ,ρ,l)
                                                        │
PPO actor ──► raw proposal w̃_t ──► cash floor / turnover ──► w̄_t
                                                        │
                              L_t(w̄_t^r) ≤ l_t ? ──yes──► execute w_t=w̄_t
                                                        │
                                                       no
                                                        ▼
                                      SOCP projection ──► w_t
                                                        │
                              audit log: docs, state, bucket, ρ, flag, residuals
```

## §2 · 數學層

📌 **Napkin Formula**：

```text
L_t(w_t^r) = −(w_t^r)^T m_t^sem + ρ_t · sqrt((w_t^r)^T Σ_t^sem w_t^r)
gate: if L_t(w̄_t^r) > l_t then SOCP projection
ρ_t = Q_{1−α}({e_τ | τ ∈ C_t, b_τ = b_t}),  α=0.10
```

**複雜度與直覺**：對每個候選配置，worst-case semantic loss 是 closed-form quadratic form，避免在執行迴圈內做 nested adversarial search；只有 gate 觸發時才解 SOCP。實驗固定為 nine-ETF universe，quadratic form 很小；真正的成本在 projection 與資料前處理，不在 adversarial search。半徑由 rolling two-year 校準窗口的 empirical order statistic 決定，稀疏桶有 fallback，目標是 90% empirical coverage within the bucketed design。

**Loss / 訓練細節**：PPO reward 同時懲罰 turnover 與 semantic-budget breach：

```text
r_{t+1}=κ_R ln(1+R_{p,t+1}) − λ_turn ||w̄_t−w_{t−1}||_1 − λ_sem max(0, L_t(w̄_t^r)−l_t)
```

來源給定 κ_R=100、λ_turn=0.02、λ_sem=20。actor 看得到 x_t，但看不到 m_t^sem、Σ_t^sem、ρ_t；訓練不在 gradient loop 內解 projection。這是刻意的：policy 只負責提案，certificate 與 gate 負責否決。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為機制示意，不是論文實證；權重、m、Σ、r 的數值所讀文本未見，故以符號代入，不補預設值。僅使用來源已出現的常數。

1. 假設某日 schema 回傳 x_t（severity、uncertainty、macro、liquidity、earnings、1−conf 與 affected-sector scores）。若 regime 宣告 risk_off，s_t 至少抬到 0.60；risk_on 則衰減 broad stress。
2. 由 x_t 算 s_t、g_{i,t}、m_{i,t}^sem=−λ_m δ_i(s_t+g_{i,t})，再由 ψ_{i,t} 與 φ_{i,t}^sem 構造 Σ_t^sem=D_t Σ̂_t D_t+εI，ε=10^{-6}。defensive sectors 的 δ_i 衰減，Energy／Financials 額外吃 macro-liquidity loading。
3. 用 rolling 504-day 的 nonconformity scores 取分桶 order statistic，α=0.10；若同桶少於 60 筆，fallback 到 unconditional history，且至少需 126 筆；半徑下限 ρ_min=0.005。
4. PPO 提出 raw proposal，先過 cash floor 與 turnover limiter 得 w̄_t；預算 l_t=max(l_min=0.020, κ_t L_t^ref, 1.03 L_t(w_{t−1}^r))。計算 L_t(w̄_t^r)；若 L_t(w̄_t^r)≤l_t 直接執行，否則解 SOCP。
5. 寫入 audit record：來源文件、semantic state、bucket、ρ_t、raw action、projected action、intervention flag、constraint residuals。若要逐點 numerical walk-through，需作者 release 的 certificate／weights；所讀文本未見，圖表／附件待核。

## §3 · 數據層

- **市場／頻率**：nine U.S. Select Sector SPDR ETFs：XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLY、XLV；daily adjusted closing prices，2015-01-01 至 2023-12-31。Real Estate 與 Communication Services 因不滿足 full-horizon availability 被排除。
- **文本**：Financial News and Stock Price Integration Dataset（FNSPID），來源覆蓋 1999–2023；實驗取 shared 2015–2023 window。每篇文章按 timestamp 指派交易日；官方收盤後發布的文章移到下一個 eligible trading date。每日 block 串接 headline 與每篇 body 前 200 characters；超過 context budget 時，每日上限 20 documents。
- **模型**：Llama-3-Instruct，8-billion-parameter；4-bit weight quantization、half-precision computation；temperature 0，作為 deterministic schema extractor。schema failure（缺欄、非法範圍、markdown wrapper、malformed JSON）走 logged neutral fallback，不終止回測。
- **樣本外**：rolling walk-forward，每個 fold 48-month training + 12-month test；fold 序列為 2015–2018 訓練對 2019、2016–2019 對 2020、2017–2020 對 2021、2018–2021 對 2022、2019–2022 對 2023。OOS 共 1,256 trading days；包含 2020 crash。
- **容量假設**：九檔流動性 ETF、日頻 end-of-day rebalancing；交易成本 10 basis points per unit traded weight；未見 market impact、slippage、借券／空頭成本。容量測試未披露。

**時間邊界分列（不可互相套用）**

| 項目 | 來源可定位事實 | 時間邊界 |
|---|---|---|
| 預訓練 cutoff | 所讀文本未見 | 未披露；不可用 2015–2023 樣本或 2026 發布日反推模型知識截止 |
| tokenizer | 所讀文本未見 | 未披露 |
| 適配／微調 | 使用 Llama-3-Instruct；未見再訓練或微調步驟；4-bit quantization 被作者稱為 engineering constraint，不是 extraction accuracy 不變的證據 | 未披露 |
| topic／semantic mapping | 無 topic model；schema 固定；語義轉換常數在 reported walk-forward 前固定，不在 test window 重估 | 不在 test window 擬合 |

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| `https://github.com/MoZarei-AI/Gated_CSRC`（作者稱 publication 後公開） | 未披露；Llama-3-Instruct 8B 的具體 checkpoint 連結未見 | 未披露 | TBD；repo 尚未公開，資料授權與環境細節待核 | 公開市場數據 + FNSPID；processed semantic certificates 與 derived outputs 稱在授權允許下釋出 |

## §5 · 評測 / Benchmark

> 前SOTA 欄是模板欄名；此處基線是 **Raw PPO**，不是論文自稱 SOTA。Δ 僅放在本固定五欄表，且僅對同一資料集／同一 fold 的 Raw PPO vs Gated CSRC 計算。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 九檔美國 Select Sector SPDR ETFs / 2019–2023 OOS（基線：Raw PPO） | Mean annual return | 12.7% | 11.7% | -1.0pp |
| 九檔美國 Select Sector SPDR ETFs / 2019–2023 OOS（基線：Raw PPO） | Mean Sharpe | 1.13 | 1.09 | -0.04 |
| 九檔美國 Select Sector SPDR ETFs / 2019–2023 OOS（基線：Raw PPO） | Mean semantic violation rate | 4.05% | 0.00% | -4.05pp |

**解讀**：違規率的 Δ 是這篇最硬的 capability 證據：Raw PPO 在 paired sample 的 1,256 天中有 51 天違規，fold 平均 4.05%（paired sample 表中為 4.06%），Gated CSRC 觀察到 0.00%，exact McNemar p<10^{-15}。但這只是「對本文 certificate」的合規，不是對所有風險模型的普遍保證；certificate 本身由 LLM schema state 與分桶共形半徑生成，ablation 的 Shuffled LLM 4.05%、Loughran–McDonald 4.45%、Neutral 4.45% 支持時間對齊有信息，但不證明經濟因果機制。

收益與風險調整後的 Δ 不是 dominance：return 點估計 12.7%→11.7%，來源稱此為 100-basis-point auditable safety premium；但 Newey–West p=0.175 未檢出 mean-return loss，Jobson–Korkie p=0.493 未檢出 Sharpe gap。Hard CSRC 10.9% / 1.04 / 0.00% 與 Gated 11.7% / 1.09 / 0.00% 的差異隔離了 gating 本身；兩者 binary safety 的 McNemar p=1.000，不能把 Hard 與 Gated 的回報差說成統計顯著。Equal Weight 12.06% / 0.670 / 0.00% 高於 Gated 的 return 點估計，作者也明說 CSRC 是 safety governor，不是 universal return engine。Momentum Top-3 -0.12% / -44.95% / 23.48%，加 CSRC 後 0.94% / -41.70% / 0.10%，說明 concentrated winner path 才是 gate 最需要介入的地方；Inverse Volatility 0.20%→0.00%，Sortino 0.735→0.710，則顯示可行性執法有小型成本。

**可能過擬合／未計成本**：語義轉換常數稱在 walk-forward 前固定、未做 retrospective grid search；但 universe 只有九檔、OOS 只到 2023、成本只有 10 basis points、無 market impact／slippage；這些限制使結果不能外推到大宇宙、盤中或高衝擊策略。Risk-profile 也顯示非單調：Conservative 11.7% / 1.09 / 21.1% intervention / 0.0% strict violation；Moderate 11.8% / 1.10 / 15.0% / 3.7%；Aggressive 12.9% / 1.14 / 0.0% / 3.3%。放寬後 violation 不是單調排序，作者自己也說這是 configurable return-safety exchange，不是 monotone violation law。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- §3.4／§6：校準依賴 rolling two-year semantic buckets 內的 local approximate exchangeability；作者明說不主張 exact stationarity，也不移除 non-stationarity。
- §6：使用 cached daily semantic certificates，較適合 institutional end-of-day rebalancing，不適合 ultra-low-latency trading on streaming news。
- §4.2：4-bit quantization 是 engineering constraint，作者未把它當成 extraction accuracy 不變的證據。
- §6：single-pass semantic extraction 是下一階段可改進點；future work 提 committee-based certificate validation。
- §5.6：financial performance tests 的 distributional support 較弱；日報酬非 normality、序列相關，Jobson–Korkie 與 Newey–West 只讀成 approximate diagnostic。
- §5.7／§6：證據支持 safety improvement，不建立 financial outperformance；不能寫成 return dominance。
- §4.1：九檔 ETF，Real Estate 與 Communication Services 因 full-horizon availability 被排除；這是 universe 設計限制。

### 6.2 推斷的隱含假設（本頁推斷，非作者原話）

- **Regime／local exchangeability（推斷）**：分桶只提供局部可比性，不是全局 coverage guarantee。若 semantic bucket 對未來 regime 錯配，ρ_t 的 90% 目標只對校準設計成立；此限制適用於本文九檔、2015–2023、日頻設定，不能推成所有模型／市場的必然結論。
- **Timestamp 對齊（推斷）**：作者用收盤後文章移到下一交易日來避免 lookahead；若 FNSPID timestamp 或文章抓取時間有誤，certificate 仍可能泄漏。此假設需資料審計，所讀文本未見逐篇 timestamp 驗證。
- **成本／容量（推斷）**：10 basis points 的 proportional cost 不含 market impact、slippage、流動性折價；九檔 ETF 的容量結論不能套到小市值或高周轉策略。容量與 breakeven 成本未披露。
- **LLM extraction 穩定性（推斷）**：temperature 0 只保證同輸入同環境的 deterministic decode，不保證 schema 語義正確；neutral fallback 可能把格式失敗轉成「風險為中性」，這在壓力期可能低估風險。作者只說 fallback 防止格式錯誤變成執行失敗。
- **Cash 為安全資產（推斷）**：cash 被建模為 liquid zero-return reserve，沒有通膨、信用或擠兌風險；gate 只能把資金移進現金，不能空頭或外部對沖。
- **Universe selection（推斷）**：排除 Real Estate／Communication Services 是 full-horizon availability 的設計選擇，不應直接稱 survivorship bias；但它會限制外推到更完整 sector universe。輔助數據缺失也不等於生存者偏差。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Raw PPO | 同一個 text-conditioned state，但無 certificate 與 gate | 未披露 | 基線；12.7% / 1.13 / 4.05% |
| Hard CSRC | 同一 certificate，但每日無條件投影 | 未披露 | 10.9% / 1.04 / 0.00%；與 Gated 的 binary safety p=1.000 |
| Momentum Top-3 + LegacyCap + CSRC | heuristic semantic ceiling vs calibrated ellipsoid | 未披露 | 4.84% / 0.286 / 0.00% |
| LLM sentiment／ranking trading agents | text→ranking／return vs text→risk state | 未披露 | 背景對照，不是本文主 counterfactual |
| Classical robust／conformal portfolio | 歷史 moments／一般共形 vs semantic shape + bucket conformal | 未披露 | lineage；本文差異在文字改變 geometry 與 feasible set |

🎤 **Interview Tip**

| 問法 | 正確答 | 錯答 |
|---|---|---|
| 這篇的 LLM 在做什麼？ | LLM 只產生 schema-validated semantic risk state，經 conformal calibration 變成 factor-shock ellipsoid；不選股、不預測報酬。 | 「LLM 預測收益／選 sector，所以 alpha 很強。」 |
| 主要實證是什麼？ | Raw PPO 12.7% / 4.05% 違規；Gated 11.7% / 0.00%，McNemar p<10^{-15}；return 與 Sharpe 差異未檢出（p=0.175／p=0.493），不是 return dominance。 | 「Gated 在所有 benchmark 上收益最高／Sharpe 顯著更好。」 |
| Hard 與 Gated 差在哪？ | 兩者同一 certificate；Hard 每日投影，Gated 只在 breach 時投影。binary safety p=1.000，不能說兩者回報等價。 | 「Hard 和 Gated 的 return 統計等價，因為 p=1.000。」 |
| Coverage 保證有多強？ | 只在 rolling two-year semantic buckets 的 local approximate exchangeability 下目標 90% empirical coverage；作者不主張 exact stationarity。 | 「共形校准保證所有非平穩市場的 90% coverage。」 |

### 7.1 可證偽預測帶日期

驗證日：來源 fold 序列的終點 2023 之後的下一個 12-month test window（具體年份未披露；不得用發布日反推）。在相同九檔 ETF、相同 10 basis points 成本與相同 walk-forward 協議下：

1. 若把 True LLM certificate 換成 Time-Shuffled LLM 控制，final true violation rate 應回到來源已給的 4.05%，而不是 0.00%；若 repo 重現後 Shuffled 仍為 0.00%，本文的「時間對齊有執行價值」主張就被削弱。
2. 若把 conformal miscoverage 放寬到 Aggressive profile 的 α=0.25，strict true violation rate 不應維持 0.0%；來源在同一 fold 已觀察 3.3%，若新 fold 仍為 0.0%，則「放寬安全邊界會重新暴露」的機制需重修。
3. 若作者公開 repo 後，Hard CSRC 與 Gated CSRC 在相同 certificate 下的 return 差仍存在，但 binary safety 仍 p=1.000，則應把差異歸因於 intervention frequency，而不是 certificate 本身；此預測待 repo 與新 fold 驗證。

## §8 · For the Reader

- **因子研究員**：把 Gated-CSRC 當 execution filter，不是 alpha 因子。可測的是「文本風險態是否改善 tail／drawdown 的條件覆蓋」，不是把 LLM score 塞進 expected return。先重現 Shuffled 與 LM lexicon 的 4.05%／4.45% 對照，再談增量。
- **高頻執行**：這篇不適合你。作者明說 cached daily certificates 更適合 end-of-day rebalancing，不是 streaming news 的 ultra-low-latency；盤中版本、latency、impact 成本皆未披露。
- **組合配置**：可把 risk-aversion profile 當治理旋鈕：Conservative 0.0% strict violation / 21.1% intervention，Moderate 3.7% / 15.0%，Aggressive 3.3% / 0.0%；但要知道這不是單調 law，且 return 11.7%／11.8%／12.9% 是在同一組九檔 ETF 與成本假設下。
- **LLM-agent 研究者**：值得抄的是「schema + conformal + gate」的權限分離，而不是 prompt。把 LLM 輸出限制在 typed risk state，讓 coverage 由 empirical nonconformity 決定；不要把 softmax confidence 當 coverage。
- **RL 策略研究者**：gate 在 gradient loop 外，reward 只加 semantic-budget penalty；這避免把 projection 微分進 PPO。下一步應測 actor 看不到 m／Σ／ρ 的資訊不對稱是否限制學習，以及 Hard vs Gated 的 intervention frequency 是否可學習。
- **研究學生**：先等 repo 公開，重跑 fold 與 ablation；核對 timestamp 規則、bucket fallback（<60 筆／≥126 筆）、α=0.10、ρ_min=0.005、l_min=0.020。不要把 2015–2023 的結果外推成所有市場；也不要把 2026 發布日當成 Llama 預訓練截止。

## References

- Zarei, Mohammad; MohammadpourAzari, Amirhossein; Habibi, Moslem. *Conformal Risk Certificates for Portfolio Optimization Under Uncertainty: A Gated Reinforcement-Learning Approach.* arXiv:2611.14699v1 [q-fin.PM], 10 Nov 2026. https://arxiv.org/abs/2611.14699v1
- Vovk, Gammerman, Shafer. *Algorithmic Learning in a Random World.* Springer, 2005.（conformal prediction lineage）
- Angelopoulos, Bates. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv:2107.07511, 2021.
- Loughran, McDonald. *When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks.* Journal of Finance, 2011.
- Schulman et al. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347, 2017.
- Ben-Tal, Nemirovski. *Robust Convex Optimization.* Mathematics of Operations Research, 1998；Goldfarb, Iyengar. *Robust Portfolio Selection Problems.* Mathematics of Operations Research, 2003.
- DeMiguel, Garlappi, Uppal. *Optimal versus naive diversification: how inefficient is the 1/N portfolio strategy?* Review of Financial Studies, 2009.
- Guo et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
- McNemar. *Note on the sampling error of the difference between correlated proportions or percentages.* Psychometrika, 1947；Jobson, Korkie. *Performance hypothesis testing with the Sharpe and Treynor measures.* Journal of Finance, 1981；Newey, West. *A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.* Econometrica, 1987.