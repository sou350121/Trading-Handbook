<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# FToken 解構（FToken）

> **來源發布**：2026-11-23 · （無 venue） · arXiv [2611.30112](https://arxiv.org/abs/2611.30112)
> **arXiv 原文**：[FToken: Factors as Tokens for Dual-Axis Cross-Sectional Return Prediction](https://arxiv.org/abs/2611.30112v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：FToken 把股票-因子 panel 的「因子列」當 token，補上 factor-axis attention 與 stock-aligned readout，再用 regime gate 融合 stock-axis 分支；五軸落在量價表格、日頻波段、監督回歸、端到端表征、全自動黑盒。它針對的 prior gap 是：多數 financial Transformer 只沿股票軸或時間軸做 attention，因子通常只是 stock token 的特徵，無法直接比較「因子截面」。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ① FToken 在既有 stock-axis Transformer 外，新增 factor-column token 分支：每個因子在當日股票橫截面上的取值編成一個 token，做 F×F self-attention。② 核心 trick 是用可學習的 stock-aligned decoder（W_back、w_read）把 factor states 還原成每檔股票一個 forward-return 預測，再與 stock-axis 分支按 trailing factor covariance 與 market volatility 門控融合。③ 這直接針對「factor 只能當 stock token 特徵、無法直接比較因子截面」的 prior gap，並用 same-axis stock ensemble 控制「只是多模型平均」。④ 在 NASDAQ 158-factor panel，equal-weight dual-axis IC 0.0421、full gate 0.0443，對比作者所稱 strongest external baseline DoubleEnsemble 0.0392；NASDAQ gate 的 Ann. Ret. 0.0842 最高，但 equal-weight 的 portfolio IR 1.34 高於 gate 1.12。

**X-Ray.** FToken 的 Pareto 位移不在「更大模型」，而在 token 軸：讓 attention 直接比較同日跨股票的因子截面，再用 stock-aligned dense decoder 把 F×d 狀態還原成 N 檔預測。這解了舊工程坑：因子只能當 stock token 特徵、無法直接比較因子截面；同時用 two independently trained stock models、stock-axis capacity control、factor-token MLP 把「只是多模型平均」與「只是參數變多」拆開。Envelope 很清楚：W_in/W_back 對每檔股票有專屬參數，要求固定 universe 與 ticker order；三個 benchmark 是 fixed filtered pools，作者自述 survivorship；回測是 daily close、top-50 dropout、5bp 單邊，liquidity/crowding 未計。對量化讀者，價值在可檢驗的軸多樣性診斷（residual correlation、principal-angle distance、CKA）與目標依賴的取捨：NASDAQ gate 的 Ann. Ret. 0.0842 高於 equal-weight 0.0754，但 equal-weight IR 1.34 高於 gate 1.12；不要把 IC Δ 直接當 IR Δ。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作／對照 | FToken |
|---|---|---|
| Token 軸 | 多數 financial Transformer 以 time 或 stocks 為 attention 軸；factor 是 stock token 的特徵 | 每個 factor column 一個 token，做 F×F self-attention |
| 輸出對齊 | factor 表徵留在 factor 空間，或只做 stock 預測 | learned `W_back` + `w_read` 將 factor states 解碼回 N 檔股票收益 |
| 融合 | 單一模型或同軸 ensemble | stock-axis 與 factor-axis 兩分支；gate 用 trailing factor covariance concentration/effective rank + trailing 20-day index volatility |
| 控制實驗 | 少見同數據同 target 的軸對照 | 與 two independent stock models、stock-axis capacity control、factor-token MLP 對照；另有 stock-query readout |

### 1.2 ⚡ Eureka

把 panel 轉置：不是只問「股票 i 像哪檔股票」，而是問「因子 j 的橫截面像哪個因子」；再用可學習 decoder 把因子狀態投影回股票，讓兩種預測可直接平均或門控。直覺是：若兩個分支的誤差相關低於兩個同軸 stock 模型，平均才有互補性；FToken 用 residual correlation 與 principal-angle distance 把這個直覺做成可量測的診斷。作者用 same-axis stock ensemble 對照：NASDAQ 158-factor 上 cross-axis 平均 IC 0.0421，two stock models 平均 0.0352；cross-axis residual correlation 0.541，two stock models 0.823。

### 1.3 信息流

Z_t (N×F) → row encode → stock tokens → stock attention → yhat^s (N)

Z_t (N×F) → column encode：H_t^(0)=Zbar_t^T W_in → factor tokens (F×d) → factor self-attention → H_t (F×d) → W_back、w_read → yhat^f (N)

trailing C_t（L=60 日因子協方差平均）、effective rank、20-day index volatility → regime vector r_t → softmax gate → (g_t^s, g_t^f)

yhat_t = g_t^s yhat_t^s + g_t^f yhat_t^f，且 g_t^s+g_t^f=1

## §2 · 數學層

📌 **Napkin Formula**

`H_t^(0)=Zbar_t^T W_in`；`H_t^(ℓ)=Concat_a[softmax(Q_a K_a^T/√d_a) V_a] W_O`；`R_t^f=M_t(H_t W_back)^T`；`yhat_t^f=R_t^f w_read`；`(g_t^s,g_t^f)=softmax(h_θg(r_t))`；`yhat_t=g_t^s yhat_t^s+g_t^f yhat_t^f`。

複雜度：stock-attention layer 為 `O(N^2 d + N d^2)`；factor branch 為 `O(F^2 d + N F d + F d^2)`。以 F=158、N 約 500 到 1,700，factor branch 的 attention matrix 較小，但 dense encoder/decoder 各需 Nd 參數；作者量到 factor branch 有 1.93× stock branch 參數、0.51× peak memory、0.64× per-epoch time（NASDAQ）。

直覺：factor token 的每個 entry 是「同一日不同股票」的觀察值，attention score 表示兩個因子截面的關係；decoder 再把一個 factor state 投影回股票順序，否則 factor branch 的輸出無法與 stock branch 逐股比較。Loss 是 cross-sectionally normalized returns 上的 MSE；兩階段訓練：先訓練兩分支，凍結後用較晚的 route-validation interval 以 ensemble MSE 訓練 router；超參數在 route-validation 以 ICIR 選擇。Prop. 0 給出條件最優權重 `g*=(v_f−c)/D`，`D=v_s+v_f−2c`，凸約束取 `clip(g*,0,1)`；`R(g)−R(g*)=D(g−g*)^2`。

## §2.5 · 帶數字走一遍（Worked Example）

以下數字全部是**假設/示意**，不是論文實證結果；論文真實數字留在 §5。

1. 假設某 regime 下兩分支的條件誤差動差為 `v_s=1.0`、`v_f=1.2`、`c=0.9`（示意值）。
2. 依 Proposition 0，`D=v_s+v_f−2c=1.0+1.2−2×0.9=0.4`。
3. 最優股票權重 `g*=(v_f−c)/D=(1.2−0.9)/0.4=0.75`；factor 權重 `1−g*=0.25`。
4. 在 `g=0.75` 時，ensemble risk `R=0.75^2×1.0+0.25^2×1.2+2×0.75×0.25×0.9=0.975`；stock-only risk 為 1.0，factor-only 為 1.2，所以此玩具例中平均降低了 risk。
5. 這只是示意：若 `v_s`、`v_f`、`c` 改變，`g*` 與 risk 會變；不能把單一玩具例的方向寫成普遍規律。

## §3 · 數據層

| 層級 | Universe / 時段 | 特徵與 label | Split | 評估 |
|---|---|---|---|---|
| Native five-feature | NASDAQ、NYSE：Feng et al. 2019，2013 to 2017；S&P 500：StockMixer 發佈，474 stocks，2016 to 2022 | 每 stock-day 五個 normalized price features；label 為 one-day raw return | NASDAQ/NYSE 756/252/237；S&P 500 1006/253/352（train/route/test，交易日） | released evaluator：IC、ICIR（原文說 published tables 稱 rank IC）、P@10、daily top-five portfolio 的 annualized Sharpe ratio，before costs |
| 158-factor | 同上三個 universe；從相同 public OHLCV 計算 Qlib Alpha158 | 158 個 factors；label 為 one-day forward close-to-close return with a one-day lag，across stocks on each date 正規化 | 同上 | IC、ICIR、RankIC、RankICIR；top-K dropout portfolio after costs，對各 universe index |

- 固定 universe 與 ticker order：N=1026（NASDAQ）、1737（NYSE）、474（S&P 500）。缺失值先以 daily cross-sectional median 填充，再用 availability mask；loss 與 metrics 只在 active entries 上評估。
- 樣本外：benchmark 已按 train/route/test 切分；gate 在 route-validation 上選架構與 covariance window，test 只評估一次。所有因子構造、截面正規化、缺失值處理、covariance/volatility window 只用預測時點前資訊。
- 容量／ADV 估算：所讀文本未見。回測只給 top-K=50、每日 5 replacements、close 成交、5bp 單邊成本、252 交易日年化、10 seeds。
- Survivorship：作者在 §5 明說三個 benchmark 是 fixed pools of stocks filtered for near-complete trading histories，return estimates conditional on membership and subject to survivorship bias。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| `https://anonymous.4open.science/r/FToken-FC1C`（來源 Appendix B） | 未披露 | 未披露（文本只說資料來源 open licenses） | 中高（本頁判斷）：需 Alpha158、index series、fixed universe/masks、10 seeds；native 與 158 兩套 evaluator | 公開 benchmark repos（Feng et al.、StockMixer）；158-factor backtest 另需各 universe index series |

## §5 · 評測 / Benchmark

固定五欄；Δ = 本方法 − 前SOTA。前SOTA 欄只放所讀原文逐字給的單一具名基線，沒有把多個基線 min/max 成假 SOTA。表中取 seed means；原文對 stochastic 方法另給標準差，這裡不重複。DoubleEnsemble 在 158-factor 被作者稱為 strongest external baseline；StockMixer、ESTIMATE、LightGBM 只在對應列明示為外部基線，不是把所有基線合成一個 SOTA 區間。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---:|---:|---:|
| NASDAQ 158-factor（基線：DoubleEnsemble，來源稱 strongest external baseline） | IC | 0.0392 | 0.0443 | +0.0051 |
| S&P 500 158-factor（基線：DoubleEnsemble） | IC | 0.0356 | 0.0405 | +0.0049 |
| NYSE 158-factor（基線：DoubleEnsemble） | IC | 0.0281 | 0.0316 | +0.0035 |
| NASDAQ native five-feature（基線：StockMixer，外部基線） | IC | 0.042 | 0.046 | +0.004 |
| NYSE native five-feature（基線：ESTIMATE，外部基線） | IC | 0.030 | 0.032 | +0.002 |
| S&P 500 native five-feature（基線：StockMixer，外部基線） | IC | 0.040 | 0.043 | +0.003 |
| NASDAQ 158-factor（基線：LightGBM，外部基線） | Ann. Ret. | 0.0685 | 0.0842 | +0.0157 |
| NASDAQ 158-factor（基線：LightGBM，外部基線） | IR | 1.09 | 1.12 | +0.03 |
| S&P 500 158-factor（基線：DoubleEnsemble，外部基線） | Ann. Ret. | 0.0524 | 0.0701 | +0.0177 |
| S&P 500 158-factor（基線：DoubleEnsemble，外部基線） | IR | 1.04 | 0.99 | -0.05 |
| NYSE 158-factor（基線：DoubleEnsemble，外部基線） | Ann. Ret. | 0.0342 | 0.0434 | +0.0092 |

解讀：158-factor 三市場的 IC Δ 均為正；NASDAQ 與 S&P 500 的 equal-weight vs DoubleEnsemble 另給 paired 95% CI（NASDAQ advantage 0.0029，95% CI [0.0006, 0.0052]；S&P 500 difference 0.0033，95% CI [0.0009, 0.0057]），NYSE 未見同樣 CI。gate over equal weighting：NASDAQ +0.0022（95% CI [0.0007, 0.0037]）、S&P 500 +0.0016（95% CI [0.0002, 0.0030]）。Native NYSE 的 0.032 vs 0.030 被作者明說在 seed 變異內，不能寫成穩定領先；作者也說 NYSE 最高 Sharpe 是 StockMixer、S&P 500 最高 P@10 是 ESTIMATE。

Portfolio：NASDAQ gate 的 Ann. Ret. 0.0842 高於 equal-weight 0.0754，但 equal-weight IR 1.34 高於 gate 1.12；S&P 500 gate 與 equal-weight Ann. Ret. 0.0701/0.0702 幾乎相同，IR 0.99/1.35；NYSE equal-weight Ann. Ret. 0.0439 略高於 gate 0.0434。成本已按 5bp 單邊計，liquidity/crowding 未計；fixed filtered pools 有 survivorship。這些 Δ 不全是可移植的 capability：跨軸互補性的診斷是本文較可信的部分（NASDAQ 158-factor residual correlation 0.541 vs 0.823；principal-angle distance 0.476 vs 0.208；CKA 0.41 vs 0.78），但這是單一市場/單一特徵層的相關排序，不能寫成所有模型／市場的必然因果。作者稱 preprocessing、架構、gate features、portfolio settings 在 test 前固定，paired CI 用 20-day moving-block bootstrap 且未做多重比較調整；固定 universe 與 survivorship 是結構性偏差，不是調參能消除。gate 提升 IC 與 NASDAQ Ann. Ret.，但 equal-weight 的 IR 較高；來源未將此差異歸因為訓練目標，亦未檢定其來源。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

- §5：三個 benchmark 是 fixed pools of stocks filtered for near-complete trading histories，return estimates conditional on membership and subject to survivorship bias。
- §5：dense factor branch 把 learned weights 與 named stocks 關聯，universe 與 mask 成為 model specification 的一部分；stock-query readout 與 mask-rate 分析只是在這些 pools 內做敏感度。
- §5：principal-angle distances 只指特定 layer 與 subspace rank。
- §5：portfolio 結果假設 daily close 成交、stated one-way costs；liquidity、crowding、structural breaks、live trading feedback 可能改變 tradeoff。
- §4.3：HIST 需要 concept graphs、MASTER 用 market-guided feature selection；public panels 缺這些 auxiliary inputs，因此不能把這兩類模型直接放進同一比較。
- §4.7：NYSE 五特徵的 gated model 對 ESTIMATE 的差異在 variation across seeds 內；作者沒有把它寫成穩定顯著。

### 6.2 推斷的隱含假設

- **固定 universe / ticker order（來源 §3.2 + 本頁推斷）**：W_in 的每一列與 W_back 的每一行對應特定股票；股票重排需同步置換權重。這使模型對成分股變動、新上市、下市不自然；來源有 mask-rate 與 constituent turnover 敏感度，但沒有給完整動態 universe 方案。
- **Gate 的 regime 特徵是低維 summary（本頁推斷）**：C_t 是 L=60 個每日截面因子協方差矩陣的平均，每個日協方差由當日 n_t 檔股票估計；來源未量化譜估計自由度。只能推斷 gate 的 concentration/effective rank 是低維 summary，其有效性依賴 regime 與誤差協方差關係穩定，不能直接外推為完整 158×158 譜的穩定估計。
- **回測執行假設（本頁推斷）**：top-K=50、每日 5 replacements、close 成交、5bp 單邊；沒有 market impact、借券、漲跌停、流動性分層。成本 2.5 到 10bp 內 routing 優勢由 1.0 收窄到 0.6 個百分點（來源），但超出此範圍的行為未測。
- **Survivorship 不是 auxiliary data missing（來源 §5 + 本頁推斷）**：fixed filtered pools 直接影響 return estimates 的母體；這與「某個 auxiliary graph 缺失」不同，不能靠補資料解決。HIST/MASTER 的 auxiliary inputs 缺失是另一類 benchmark 限制。
- **軸互補性不是普遍保證（本頁推斷）**：ensemble 增益取決於 residual correlation 與成員品質；NASDAQ 158-factor 的 cross-axis residual correlation 0.541 vs two stock models 0.823，但 NYSE 五特徵差異在 seed 變異內。不能把單一 universe 的相關排序寫成所有市場的必然因果。
- **特徵/標籤時序（來源 §4.2 + 本頁推斷）**：158-factor tier 用 one-day forward close-to-close、one-day lag、截面正規化；native tier 用 one-day raw return。兩層不可混用 label；Alpha158 由同一 OHLCV 重算，仍需自行核對 vendor 時間戳與 index series 對齊。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| StockMixer | MLP 混合 indicator/time/stock axes；沒有 factor-column attention | 所讀文本未見 | 外部基線；來源把它延伸到 158-feature 輸入 |
| DoubleEnsemble | sample reweighting + feature selection 的 ensemble；158-factor 最強外部基線 | 所讀文本未見 | 外部基線 |
| GAT | stock graph attention；仍是 stock-axis | 所讀文本未見 | 外部基線 |
| iTransformer / Feature-Axis Transformer | variable/feature-axis tokens 用於時間序列或 stock-movement；不是 stock-factor panel 的 stock-aligned decoder | 所讀文本未見 | lineage，非直接 baseline |
| MASTER / HIST | market-guided feature selection 或 concept graph；public panels 缺 auxiliary inputs | 所讀文本未見 | 來源 §4.3 明說不能直接比較 |

🎤 **Interview Tip**
- 正確答：FToken 的關鍵是「factor-column token + stock-aligned decoder」，讓 factor branch 與 stock branch 預測同一個 N 維目標；gate 只是二階，same-axis stock ensemble 才是控制「多模型平均」的關鍵對照。還要能說出：NASDAQ 158-factor 的 cross-axis residual correlation 0.541 低於 two stock models 的 0.823，平均 IC 0.0421 高於 two stock models 的 0.0352。
- 錯答：說成「把股票 Transformer 加大」或「gate 一定優於 equal-weight」。來源在 NASDAQ 158-factor 給 equal-weight IR 1.34、gate 1.12；S&P 500 equal-weight IR 1.35、gate 0.99。gate 的優勢是特定目標下的 IC/Ann. Ret.，不是全面支配。

### 7.1 可證偽預測

本頁可證偽預測（非來源事實，尚待驗證）：在 **2027-06-30** 前，若以相同 NASDAQ 158-factor split、fixed universe、top-50、5bp 單邊成本重跑，full regime gate 的 portfolio IR 仍低於 equal-weight dual-axis（來源目前 gate 1.12、equal-weight 1.34），則「gate 提升風險調整後組合效率」在該設定下不被支持；若 gate 的 IR 反超，則本頁對 gate/equal-weight tradeoff 的保守解讀被推翻。

## §8 · For the Reader

- **因子研究員**：把 factor branch 當成「因子截面互動」的端到端表徵，而不是新因子挖掘；先看 residual correlation 0.541 vs 0.823 與 same-axis control，再判斷增益是否來自軸多樣性。κ_t 是表徵子空間診斷；來源 §4.8 明說控制 residual correlation 後，distance 與 gain 的關聯不顯著，不要把 κ_t 當獨立 alpha 預測器。
- **高頻執行**：FToken 是日頻、close 成交、top-50 dropout、5bp 單邊；NASDAQ gate turnover 0.198、equal-weight 0.178。live liquidity、crowding、market impact 未測；成本 2.5 到 10bp 內優勢由 1.0 收窄到 0.6 個百分點，不能外推為更高成本仍成立。
- **組合配置**：若目標是 Ann. Ret.，NASDAQ gate 0.0842 高於 equal-weight 0.0754；若目標是 IR，NASDAQ equal-weight 1.34 高於 gate 1.12，S&P 500 equal-weight 1.35 高於 gate 0.99。先定義目標函數，再選 gate 或 equal-weight；不要把 IC 的 Δ 直接當 IR 的 Δ。
- **LLM-agent / RL 策略**：這篇是 supervised regression，不是 RL；可借的是「同一 panel 兩種 tokenization + 可學習路由」的模組化思路，但不能把 gate 當成 RL policy 的必然组件。
- **研究學生**：復現時先固定 universe/ticker order 與 mask，重跑 two-stage gate；檢查 W_in/W_back 的股票專屬參數、route-validation 只用一次、test 只看一次。若只做 joint training，來源 §4.10 說 two-stage gate 的 IC 比 jointly trained gate 高 0.0045。

## References

- 原論文：Juntang Wang, Yihan Wang, Sheng’en Li, Dongmian Zou, Shixin Xu. _FToken: Factors as Tokens for Dual-Axis Cross-Sectional Return Prediction_. arXiv:2611.30112v1, 2026. https://arxiv.org/abs/2611.30112v1
- 代碼：`https://anonymous.4open.science/r/FToken-FC1C`（來源 Appendix B）
- Lineage：Vaswani et al. 2017（Transformer）；Liu et al. 2024（iTransformer）；Ho et al. 2019（Axial Attention）；Seo and Yang 2024（Feature-Axis Transformer）；Fan and Shen 2024（StockMixer）；Zhang et al. 2020（DoubleEnsemble）；Li et al. 2024（MASTER）；Xu et al. 2021（HIST）；Yang et al. 2020（Qlib Alpha158）；Krogh and Vedelsby 1995；Brown et al. 2005（ensemble error covariance）。