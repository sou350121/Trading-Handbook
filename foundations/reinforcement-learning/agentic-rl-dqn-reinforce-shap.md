<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=强化学习 alpha=多智能体博弈 autonomy=Agent自主演进 -->

# Agentic RL 三原型框架（DQN+REINFORCE+SHAP） 解構（Agentic RL 三原型框架（DQN+REINFORCE+SHAP））

> **來源發布**：2026-10-15 · （無 venue） · arXiv [2610.22522](https://arxiv.org/abs/2610.22522)
> **arXiv 原文**：[Agentic AI in Automated Markets](https://arxiv.org/abs/2610.22522v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點在「量價表格 × 高頻日內 × 強化學習 × 多智能體博弈 × Agent 自主演進」；它真正補的 prior gap 不是又一個 RL 交易策略，而是把 reward function 當成行為標籤生成器，讓「套利者／操縱者／跟風者」三種原型從同一 exposure-control 環境中長出可量測的交易足跡。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `高频日内` | `强化学习` | `多智能体博弈` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 用 DQN 與 REINFORCE 在同一個 asset-agnostic exposure-control 環境中，以 reward shaping 定義理性套利者、操縱者、散戶跟風者三類原型；核心 trick 是「行為寫進 reward、策略從資料湧現」，再用 SHAP 驗證各原型依賴的信號；對「多智能體博弈」軸★的意義是政策可作為行為強度的量測器，而非直接 alpha；pilot 未給正式量化對比，僅報告 rolling 7-day Sharpe 大致在 -20 與 +20 annualized 之間擺動，且作者明言這是 motivating evidence。

**X-Ray.** 這篇的價值不在「RL 能不能賺錢」，而在把 ABM 的手寫規則換成可學習的 reward-defined archetype，並用 SHAP 做行為指紋核對。它解掉舊工程坑的一部分：行為假設顯式化、策略不必手刻、同一環境可比較 value-based 與 policy-gradient。但 envelope 很窄：agents 是 price-taker、分環境訓練、無價格衝擊與策略交互；action space 粗到 exposure 只動 0.5；pilot 每配置 one training run，且 rational REINFORCE 在 AAVE 與 SUI 出現 zero exposure 的退化解。對量化讀者，這是一篇「行為分類與監管度量」的方法論，不是可交易策略；把 Sharpe 擺動或 stress 足跡當成跨市場必然因果，會直接誤讀。

## §1 · 架構 / Core Mechanism

**三大改動 vs 前作**

| 維度 | 前作／常見做法 | 本文改動 | 讀者該問 |
|---|---|---|---|
| 行為生成 | ABM 手寫 fundamentalist／chartist／noise trader 規則 | 以 reward shaping 定義 archetype，policy 由 RL 學習 | reward 是否真的等價於行為？有無 proxy 化？ |
| 演算法比較 | 多數 RL trading 只選一種演算法 | 同一環境比 DQN（value-based）與 REINFORCE with baseline（policy-gradient） | 差異是演算法能力，還是探索／崩潰偽影？ |
| 可解釋性 | 黑箱策略事後解釋 | SHAP DeepExplainer 對 action probs／Q-values 做歸因，核對 archetype 信號依賴 | top feature 是相關還是因果？ |

**⚡ Eureka 一句話 trick**：不要手寫「操縱者會洗單」；把「turnover bonus」寫進 reward，讓操縱者自己學出快速翻倉，再用 SHAP 看它是否真的依賴 short-term volatility 與 position state。直覺是：reward 是行為的基因型，policy 是表現型，SHAP 是表現型的驗證。

**信息流**

state `s_t`（drawdown、cross-asset return、momentum、volatility、position、time of day）
→ policy（DQN 的 `Q(s,a)` 或 REINFORCE 的 softmax）
→ action `a_t ∈ {sell, hold, buy}` → `Δe_t ∈ {-0.5, 0, +0.5}`，`e_t ∈ [0,1]`
→ base reward（vol-normalized net return − stress penalty）
→ archetype reward shaping（rational penalties；manipulator turnover bonus；retail momentum bonus）
→ SHAP DeepExplainer（action probs／Q-values）→ top features

## §2 · 數學層

📌 **Napkin Formula**

\[
r_t^{\text{base}}=\frac{e_t R_{t+1}-c\,|e_t-e_{t-1}|}{\hat{\sigma}_t+\epsilon}-\lambda_{\text{stress}}\,\mathbf{1}[\text{stress}_t]\,e_t
\]

\[
\text{manipulator bonus}=\kappa_{\text{turn}}|e_t-e_{t-1}|,\qquad
\text{retail bonus}=\kappa_{\text{mom}}\operatorname{sign}(\text{mom}_t)(e_t-0.5)
\]

複雜度：所讀文本未披露。直覺：base reward 先做波動歸一化與交易成本扣減，再對 stress regime 中的持倉施加懲罰；archetype 只改 reward，不改環境，因此行為差異可歸因於目標函數。DQN 用 two-layer MLP 估 `Q(s,a)`、replay buffer、target network 每 `N` steps 同步、`ε`-greedy 線性衰減；REINFORCE 用 two-layer policy network 與 two-layer value baseline，advantage 為 Monte Carlo return 減 baseline，batch 內標準化 return。reward weights（`κ_turn`、`κ_mom`、`λ_stress`）跨資產固定、在 validation window 設定，每配置 five random seeds，結果取平均；具體數值未披露。

## §2.5 · 帶數字走一遍（Worked Example）

以下為**假設／示意**，只用所讀文本已有的 exposure 網格與 action 步長示範計算路徑，不是論文實證結果；缺失參數不補預設值。

- 設 `e_{t-1}=0`（flat，原文定義）。
- 選 `buy`，action 使 exposure 動 `+0.5`，得 `e_t=0.5`（原文網格）。
- 交易成本項為 `c × |0.5 − 0| = c × 0.5`；`c` 未披露，所以成本數值未披露。
- base reward 分子為 `0.5 × R_{t+1} − c × 0.5`；`R_{t+1}` 與 `\hat{\sigma}_t` 未披露，reward 數值未披露。
- 若 `stress_t=1`，stress penalty 為 `λ_stress × 0.5`；`λ_stress` 未披露。
- 對 retail，momentum bonus 為 `κ_mom × sign(mom_t) × (0.5 − 0.5) = 0`；這是 `e_t=0.5` 的算術結果，但 `κ_mom` 與 `mom_t` 的實際值未披露。操縱者 bonus 為 `κ_turn × 0.5`，`κ_turn` 未披露。

結論：可以照著走完 exposure 與 bonus 的代數路徑，但因為 `c`、`R_{t+1}`、`\hat{\sigma}_t`、`λ_stress`、`κ` 未披露，無法得到可驗證的數值輸出；任何具體 reward／Sharpe 數字都應視為未披露。

## §3 · 數據層

- 市場與頻率：pilot 用 hourly cryptocurrency data，資產為 AAVE、ETC、SUI；所讀文本未見更細的資料來源、交易所、清洗規則。
- 規模：作者在 §8 自述 pilot 約 `1,750 hours per asset`；§3「Data and Features」正文在所讀文本中未見，圖表／附件待核。
- 時段與 stress 定義：CRSP 的 stress indicator 為 `VWRETD below its 5th percentile` 或 `VIX jump above its 95th percentile`；crypto pilot 的 stress 定義是否沿用同一規則，所讀文本未披露。
- 樣本外與容量：agents 是 price-taker，無價格衝擊；shared limit-order-book 與三原型同場交易被列為 future work。樣本外切分、訓練／驗證／測試邊界、容量假設均未披露。
- 特徵：drawdown、cross-asset return、momentum、short-term volatility、position state、time of day 等；特徵構造細節與是否嚴格 point-in-time，所讀文本未見。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露 |
| Checkpoint | 未披露 |
| License | CC BY 4.0（原文標示） |
| 複現難度 | 高（本頁推斷）：reward weights 數值未披露、`N` 未披露、pilot 每配置 one training run、rational REINFORCE 在 AAVE 與 SUI 出現 policy collapse；作者自述需 multi-seed 與 reward-weight tuning |
| 數據可得性 | crypto hourly 資料的具體來源未披露；CRSP 相關 Stage 2 檢驗的授權與下載細節，所讀文本未見 |

## §5 · 評測 / Benchmark

所讀文本沒有給出「前 SOTA 基線 vs 本法」的配對數值；因此下表不合成前 SOTA、不內插空白、不把 Δ 塞進前 SOTA／本方法格。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| AAVE, ETC, SUI（pilot） | Crisis frequency：Hours in a 24-hour window with return below -2% | 未披露 | 未披露 | 未披露 |
| AAVE, ETC, SUI（pilot） | Crisis persistence：Visual run length | 未披露 | 未披露 | 未披露 |
| AAVE, ETC, SUI（pilot） | Efficiency：Rolling 7-day Sharpe | 未披露 | 未披露 | 未披露 |
| AAVE, ETC, SUI（pilot） | Return volatility：Rolling 24-hour σ | 未披露 | 未披露 | 未披露 |

解讀：原文只給視覺化與定性排序——retail 的 stress episodes 最 frequent 但最短，manipulator 最少但最 persistent，rational 的 rolling Sharpe 最穩定但 return volatility 較高；這些不是可與前 SOTA 對打的 benchmark 數字。`-20` 與 `+20` annualized 是所有 agent 的 rolling 7-day Sharpe 擺動範圍，不是本法相對基線的優勢；`0.02` on SUI 是 manipulator 在 stress 期的波動尖峰，不能當成平均績效。真正的 capability 是「同一環境下 reward shaping 產生不同 stress footprint」的定性分離；但 pilot 每配置 one training run，無正式統計檢定，且 rational REINFORCE 的 zero exposure 是退化解，不能算能力。成本面：`c` 存在但數值未披露；stress 定義在 CRSP 與 crypto pilot 是否一致未披露；SHAP background 用 randomly sampled historical states，是否嚴格樣本外未披露，因此過擬合與前瞻偏差不能排除。

## §6 · 失效與隱含假設

**6.1 論文自述 limitations（可定位至 §8）**

- No price impact or interaction：agents 是 price-taker、分開環境訓練，Stage 1 不能顯示一個原型如何推動價格或影響另一個原型；Stage 2 是 correlational（§8）。
- Policies as signal composites：archetype intensity 是公開信號的非線性函數；控制 size、value、momentum、volatility、turnover 只能降低、不能消除 intensity 代理已知特徵的風險（§8）。
- Coarse action space：exposure 以 `0.5` 步長在 `[0,1]` 移動，且無空頭；更細或連續 action space 與 short selling 會容許更豐富行為（§8）。
- Pilot scale：pilot 用 one run per configuration、約 `1,750 hours per asset`；rational REINFORCE 的 policy collapse 顯示需 multi-seed training 與 reward-weight tuning 才能下正式結論（§8）。

**6.2 推斷的隱含假設（本頁推斷，非作者明述）**

- Regime 依賴：stress 由 CRSP 的 `VWRETD`／`VIX` 規則定義，crypto pilot 的 stress 規則未披露；若 pilot 與 Stage 2 的 regime 定義不同，stress footprint 的排序未必可移植。
- 容量與價格衝擊：price-taker 假設使「操縱者 turnover bonus」不會遇到自身價格衝擊；這使操縱者的 persistent stress 可能被低估或高估，容量結論不可外推。
- 成本敏感度：base reward 有 proportional cost `c`，但 `c` 未披露；manipulator 的 turnover bonus 與交易成本之間的淨效果無法判定，策略排序可能隨成本改變。
- 數據泄漏：SHAP 用 randomly sampled historical states 當 background、test period 當 evaluation set，但訓練／驗證／測試的嚴格邊界未披露；特徵是否 point-in-time 未披露，不能宣稱無泄漏。
- Survivorship／資產選擇：AAVE、ETC、SUI 的篩選規則未披露；所讀文本不足以判定 survivorship bias，也不能排除選擇效應。輔助數據缺失不是 survivorship bias 的同義詞。
- 策略交互缺失：三個 archetype 分環境訓練，市場由 deterministic AMM 匹配、market makers 被排除；這是一個制度假設，不是所有市場的必然描述。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 手寫 ABM（LeBaron） | 行為由規則 imposed；本文由 reward 學習 | 未披露 | 前作／文獻 |
| Direct RL trading（Moody & Saffell；Deng et al.） | 單一策略優化；本文多原型 reward shaping 與演算法對比 | 未披露 | 前作／文獻 |
| DQN／REINFORCE canonical（Mnih et al.；Williams） | 演算法基線；本文同環境比 value-based 與 policy-gradient | 未披露 | 基線 |
| SHAP（Lundberg & Lee） | 事後歸因工具；本文用於驗證 archetype 信號依賴 | 未披露 | 工具 |

🎤 **Interview Tip**

- 正確答：這篇的核心是「reward-defined archetype + policy as measurement device + SHAP 驗證」；pilot 是 motivating evidence，不是正式 causal test；strategy-level Sharpe 只說明策略如何利用市場，不說明市場效率。
- 錯答：把三個原型當成已校准的市場參與者；把 DQN 與 REINFORCE 的結果互相代稱；把 pilot 的視覺排序說成所有模型／市場的必然因果；把 SHAP top feature 當成 causal effect。

**7.1 可證偽預測帶日期**

原文 Table 2 給出 H1–H4，並說 CRSP tests 可 confirm or reject；Stage 2 的具體時程未披露（TBD）。可證偽條件如下：

- H1：retail intensity 對 crisis frequency 的 `β_RET>0`；若符號相反或不可辨識，herding 監管理由減弱。
- H2：manipulator intensity 對 crisis persistence 的 `β_MAN>0`；若不可辨識，turnover／order-cycling 監管優先級減弱。
- H3：rational intensity 對 `|VR−1|` 的 `β_RA<0`；若不符，rational 改善效率的主張不成立。
- H4：rational intensity 對 volatility 的 `β_RA>0`、對 frequency 約為 `0`；若 frequency 也上升，「volatile but not stress」的監管含義失效。

## §8 · For the Reader

- **因子研究員**：把 archetype intensity 當成公開信號的非線性 composite，適合做 regime／conditioning 變數，不宜直接當 standalone alpha；必控 size、value、momentum、volatility、turnover，否則 intensity 可能只是已知特徵的代理。
- **高頻執行**：turnover、order-to-trade、cancellation ratio 是這篇最可操作的監管介面；但 pilot 無 LOB、無價格衝擊、無同場交互，不能從中推容量或做市成本。
- **組合配置**：把「volatility」與「stress persistence」分開看；原文的關鍵訊息是最高波動的策略不必然是最 persistent drawdown 的策略，監管若只盯 volatility 可能壓抑 adaptive repositioning。
- **LLM-agent／RL 策略**：reward shaping 就是行為規格；要優先檢查 policy collapse、zero exposure 退化解、reward weight 敏感度與 multi-seed 穩定性，而不是只看 best run 的 Sharpe。
- **研究學生**：這是一篇很好的「方法論骨架」：shared limit-order-book、更細 action space、short selling、正式統計檢定是直接可接的 extension；但不要把 pilot 的視覺結果寫成已證實的市場因果。

## References

- Aldridge, I. et al. *Agentic AI in Automated Markets*. arXiv:2610.22522v1. https://arxiv.org/abs/2610.22522
- LeBaron, B. Agent-based computational finance. *Handbook of Computational Economics*.
- Kyle, A. S. Continuous auctions and insider trading. *Econometrica*.
- Glosten, L. R. & Milgrom, P. R. Bid, ask and transaction prices in a specialist market. *Journal of Financial Economics*.
- Moody, J. & Saffell, M. Learning to trade via direct reinforcement. *IEEE Transactions on Neural Networks*.
- Deng, Y. et al. Deep direct reinforcement learning for financial signal representation and trading. *IEEE Transactions on Neural Networks and Learning Systems*.
- Mnih, V. et al. Human-level control through deep reinforcement learning. *Nature*.
- Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*.
- Lundberg, S. M. & Lee, S. A unified approach to interpreting model predictions. *NeurIPS*.
- Barber, B. M. & Odean, T. All that glitters: attention and news. *Review of Financial Studies*.
- Gandal, N. et al. Price manipulation in the Bitcoin ecosystem. *Journal of Monetary Economics*.
- Cong, L. W. et al. Crypto wash trading. *Management Science*.