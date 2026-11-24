<!-- ontology-5axis data=量价表格 horizon=跨周期 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# MC-JEPA 解構（MC-JEPA）

> **來源發布**：2026-11-10 · （無 venue） · arXiv [2611.13572](https://arxiv.org/abs/2611.13572)
> **arXiv 原文**：[Market-Conditioned JEPA: Learning from Unpredictable Market Shocks](https://arxiv.org/abs/2611.13572v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點＝量价表格／跨周期／監督回歸／端到端表征／全自動黑盒。它解的是 per-asset 自監督的 prior gap：市場共同衝擊被當成不可約噪聲，因此表徵學不到「資產如何回應市場」；MC-JEPA 把已實現市場衝擊降級為訓練期特權信息，推理端仍保留純 per-asset encoder。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `跨周期` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文可讀全文（與正文衝突時以原文為準）。細節待升 v1。

**TL;DR:** ①用 panel 內其他資產**未來 target embedding** 的 leave-one-out 均值構造市場條件，只在訓練期使用；②核心 trick 是通道限制——base predictor 看不到條件，市場項只能經 `b_ψ(z_i,h)·wᵀc̃` 進入，壓低 conditional loss 就等價於逼表徵編碼曝險；③對端到端表征軸★的意義是：截面資訊換成訓練期監督後可完全丟棄，推論接口與標準 per-asset encoder 相同；④測試期 Vol 120m RankIC `.517` vs per-asset JEPA `.444`、Volume 120m `.594` vs `.573`，但 Ret 20d `.1389` 低於 TS2Vec `.1454`，且 Ret 5d 對 per-asset JEPA `t=1.8` 未過 1.96。

**X-Ray.** 這篇的價值不是「又一個時序 foundation model」，而是把一個**架構約束**當成歸納偏置：市場因子若可自由進入預測器，它會直接解釋 target，且條件本身由未來 embedding 構成，資訊上近乎洩漏；作者把通路鎖成 single-index 的 `loading×factor`，讓「條件有沒有用」變成「表徵裡有沒有曝險」的可檢驗命題。放在五軸 Pareto 上，它精準解掉兩個舊工程坑：(i) 截面模型（HIST／MASTER 類、FactorVAE）推理時要同儕資訊，部署時受 universe 變動與 channel-mixing 的分布漂移懲罰；(ii) 純 per-asset 自監督（TS2Vec／CPC／patch 重建）在 risk、liquidity 這類共同因子佔比高的目標上監督不足。但它打不開的 envelope 也很清楚：Appendix E 自承條件只移除 `4.5±0.9%` 的無條件誤差，而 panel 均值雖解釋 `26%` 的變異；K=1 只在 Ret 1d／Ret 5d／Ret 20d 最好，Ret 120m 反而是 K=4 的 `.0702` 高於 K=1 的 `.0694`——**去噪是次要機制，return 增益另有來源，作者也承認 denoising does not account for all of it**。對量化讀者：真正可搬走的是「用未來截面均值當訓練期正則」這個 pattern，不是這個 repo 的具體 backbone。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 構件 | 前作／比較對象 | MC-JEPA | 為何改得動 |
|---|---|---|---|
| 監督來源 | per-asset JEPA、TS2Vec、CPC 僅用自身歷史 `x_i` 預測自身未來表徵 | 訓練期加入 leave-one-out 市場條件 `c̃_{−i,h}`（式 5），由同 panel 其他資產的**未來 target embedding** 平均而來 | 把不可預測市場衝擊由不可約噪聲轉為監督信號 |
| 條件進入方式 | 無市場條件；截面模型可直接讀同儕資訊 | 只能經 `B_ψ(z_i,h)·g(c̃_{−i,h})` 進入；base predictor `a_φ` 從不看條件（式 6） | 市場項無法單獨解釋 target，逼表徵編碼曝險 |
| 目標函數 | `ℓ_h(P_φ(z_i,h), u_{i,h})` | `(1−λ)·ℓ(û⁰) + λ·ℓ(û^c)`，`λ=0.75`（式 7） | 保留自身可預測部分，另加條件項；`λ=0` 即退回 per-asset JEPA |

推理接口不變：使用的是 `z_i=f_θ(x_i)`，不需要 peer assets、market index 或未來觀測。額外成本是 loading network 及 target encoder 對 condition pool 的前向；pool `|M_t|=32` 可大於計分集 `|S_t|=8`，因此市場估計可聚合更多資產而不增加 online encoder 評估。

### 1.2 ⚡ Eureka

**「讓市場因子只能透過你自己的載荷說話。」** 因為條件由其他資產的**未來** embedding 造出，若允許它自由進入預測器，模型會直接複製答案、學不到曝險；把通路鎖死成 `b_ψ(z_i)·(wᵀc̃)` 後，**市場條件對預測的額外影響**只能經由 `z_i` 算出的載荷進入。shared base `a_φ` 仍會同時收到 unconditional 與 conditional 兩項 loss 的梯度，因此不能說「條件項梯度只修 `b_ψ`」；可主張的是：市場條件若要改變預測，必須先被 `b_ψ(z_i,h)` 讀成資產特定載荷。Prop. 1 把它寫成：無條件預測的最優解只需 `ū(x_i)=α_i+B_i m`（曝險只透過期望因子出現），條件預測的最優解則需要 `α_i` 與 `B_i` 同時可從 `z` 還原。

### 1.3 信息流

**訓練期**：`x_i (L=960 bars) → f_θ → z_i`；`z_i → a_φ(z_i,h) → û⁰_{i,h}`；`z_i → b_ψ(z_i,h)`。其他資產 `y_j → f_θ̄(sg) → u_j`，取 `mean(M_t\\{i})` 並標準化得 `c̃`，再 `wᵀc̃`；`û^c_{i,h}=a_φ(z_i,h)+b_ψ(z_i,h)·wᵀc̃`。自身 `y_i → f_θ̄(sg) → u_{i,h}` 作 target（stop-grad）。兩條預測共同進入式 (7) 的加權 loss。

**推理期**：`x_i → f_θ → z_i`。結束；無 panel、無 market index、無未來觀測。

## §2 · 數學層

📌 **Napkin Formula**

- `û⁰_{i,h} = a_φ(z_i,h)`
- `û^c_{i,h} = a_φ(z_i,h) + b_ψ(z_i,h) · wᵀ c̃_{−i,h}`
- `c̃_{−i,h} = ( mean_{j∈M_t\\{i}} u_{j,h} − μ^c_h ) / s^c_h`
- `L = (1−λ)·E ℓ_h(û⁰,u) + λ·E ℓ_h(û^c,u)`，`ℓ_h(û,u) = (1/D)‖û − (u−μ_h)/s_h‖²`

**直覺**：`ℓ_h` 用 running mean/cov 標準化，預測「均值」時 loss 約為 1，所以跨 horizon 可比較。條件項的關鍵限制不是「梯度只走 `b_ψ`」，而是**條件對預測的邊際作用**必須經 `b_ψ(z_i,h)g(c̃)`；`a_φ` 是兩條路徑共享的 base，仍會被兩項 loss 共同訓練。這正是 Prop. 1(b) 的 `E tr(δ_B Σ_f δ_Bᵀ)` 懲罰項在實作中的對應物：若 `z` 沒保留曝險，conditional 分支無法有效利用已實現市場衝擊。

**複雜度／訓練**：encoder 12 層 Transformer、`D=256`、MLP `1024`、8 heads，encoder 參數 `9.52M`；context `960` bars／patch `10` bars＝96 patches；horizons `H={1,60,240}` bars、target block `30` bars；每步 `128` decision times、pool `32`、scored `8`；AdamW `β=(0.9,0.95)`、lr `3×10^{-4}`、weight decay `0.05`、grad clip `1.0`、`20,000` steps、EMA momentum `τ` 由 `0.996` 線性至 `1`、bfloat16、`two NVIDIA H800 GPUs`。FLOPs／推論延遲數值：所讀文本未見，圖表／附件待核。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部為假設／示意數字，只為手算機制，與論文實證結果無關；論文真實數字一律見 §5。**

1. **假設** `K=1`，為方便手算把 `D` 縮成 3（真實 `D=256`）。某資產 `i` 在決策時刻 `t` 給出 `x_i`，encoder 得到 3 維 `z_i`。
2. **假設** base predictor 輸出 `a_φ(z_i,h) = (1, 0, −1)`。
3. **假設** 其他資產未來 embedding 取均值後標準化得 `c̃`，且學到的 `wᵀc̃ = 2`。
4. **假設** loading head 讀 `z_i` 得到 `b_ψ(z_i,h) = (0.5, −0.5, 1)`。則 `b·(wᵀc̃) = (1, −1, 2)`，`û^c = a + b·2 = (2, −1, 1)`。
5. 若標準化後 target 是 `u = (2, −1.2, 1.1)`（假設），`û^c − u = (0, 0.2, −0.1)`，平方和 `0 + 0.04 + 0.01 = 0.05`，`ℓ_c = 0.05/3 ≈ 0.0167`。若表徵**沒編碼曝險**、`b_ψ` 只能輸出 `(0,0,0)`，則 `û^c = a = (1,0,−1)`，`û^c − u = (−1, 1.2, −2.1)`，平方和 `1 + 1.44 + 4.41 = 6.85`，`ℓ_c = 6.85/3 ≈ 2.283`。**關鍵**：`a_φ` 兩條路徑共用、且從不看 `c̃`；條件只能經 `b_ψ` 改變預測，因此 conditional loss 的額外壓力會集中到「`b_ψ` 能否從 `z_i` 讀出曝險」這條路徑上——但 `a_φ` 本身仍受兩項 loss 共同訓練。這就是「市場項不能單獨解釋 target」在數字上的樣子。

## §3 · 數據層

| 時間邊界 | 區間／設定（逐字） |
|---|---|
| 數據範圍 | `1-minute OHLCV bars of Chinese A-share stocks from 2015 to 2025` |
| **預訓練**截止 | `Pretraining uses decision times before July 2022` |
| **probe 適配** | `fit ridge probes on 2019-01 to 2022-06` |
| 超參／選模 | `select the ridge penalty and the hyperparameters of MC-JEPA on 2022-07 to 2022-12` |
| 樣本外測試 | `report results on 2023-01 to 2025-12` |
| Tokenizer／topic 擬合 | 所讀文本未見 tokenizer 或 topic 擬合階段 |

- **輸入**：`L=960` 根 1 分鐘 bar，特徵用「自身 context window」的均值與標準差標準化；未來 bar 用同一組統計量（作者主張 `without leaking future information`）。日頻 target 於 `15:00` 收盤決策，分鐘 target 於 `11:30` 收盤決策；評估時每股每日一個 window，結束於 target 的 decision time。
- **panel 抽樣**：每步 `128` 個 decision time，pool `|M_t|=32`（於 `t` 附近全市場均勻抽），scored `|S_t|=8`（pool 的均勻隨機子集）。
- **回測**：point-in-time CSI300 成分股，每日取 ridge 預測 Ret 20d 最高的 `7–8` 檔，持有 `20` 個交易日，`20` 個重疊 tranche、帳本約 `150` 檔；買 `5bp`、賣 `15bp`；價格為 BaoStock 日線後復權收盤；`Suspended days are removed from the data, so a held stock earns zero return while suspended`。
- **容量假設**：股票總數、可用資金規模、分鐘級訓練數據供應商——所讀文本未見，圖表／附件待核。
- **⚠ 時間邊界重疊（本頁推斷，尚待驗證）**：probe 適配區間 `2019-01 to 2022-06` 與「`before July 2022` 的預訓練 decision times」重疊；因 probe 是凍結表徵上的線性讀出、測試期在其後，泄漏量級有限，但來源未討論此重疊的影響。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | `https://github.com/OpenFinMind/MC-JEPA`（正文腳註逐字） |
| Checkpoint | 本方法 checkpoint：所讀文本未見；Table 6 只列**基線** checkpoint：`NeoQuasar/Kronos-mini`、`NeoQuasar/Kronos-small`、`NeoQuasar/Kronos-base`、`google/timesfm-2.5-200m-pytorch`、`amazon/chronos-2`、`AutonLab/MOMENT-1-small`、`AutonLab/MOMENT-1-base` |
| License | 所讀文本未見 |
| 複現難度 | 中高：需 1 分鐘級 A 股全市場 2015–2025、`two NVIDIA H800 GPUs`、`20,000` steps；但 Table 4／Table 5 超參齊備，encoder `9.52M` |
| 數據可得性 | 回測價格 BaoStock；分鐘級訓練數據來源所讀文本未見；universe 建構細節所讀文本未見 |

## §5 · 評測 / Benchmark

**表 A：RankIC（測試期 2023-01 至 2025-12，凍結表徵 + ridge probe）**

> 「前SOTA」欄在此**不是文獻 SOTA**：它放的是同架構、`λ=0` 的 per-asset JEPA 基線數值。RankIC 以小數表示，Δ 為本方法 − 基線的同單位小數差。

| 數據集/市場（基線：per-asset JEPA） | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 中國 A 股 | Ret 120m RankIC | .0677 | .0694 | .0017 |
| 中國 A 股 | Ret 1d RankIC | .0712 | .0746 | .0034 |
| 中國 A 股 | Ret 5d RankIC | .1038 | .1074 | .0036 |
| 中國 A 股 | Ret 20d RankIC | .1337 | .1389 | .0052 |
| 中國 A 股 | Beta 20d RankIC | .359 | .374 | .015 |
| 中國 A 股 | Vol 120m RankIC | .444 | .517 | .073 |
| 中國 A 股 | Volume 120m RankIC | .573 | .594 | .021 |
| 中國 A 股 | Volume 1d RankIC | .440 | .454 | .014 |
| 中國 A 股 | Open vol share RankIC | .391 | .397 | .006 |

**表 B：歷史回測（CSI300 多頭，2023–2025，淨 5/15bp）**

| 數據集/市場（基線：per-asset JEPA） | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| CSI300 多頭 | AER | 0.117 | 0.135 | 0.018 |
| CSI300 多頭 | Sharpe | 1.11 | 1.19 | 0.08 |
| CSI300 多頭 | IR | 1.20 | 1.32 | 0.12 |
| CSI300 多頭 | MDD（越低越好） | 0.132 | 0.117 | -0.015 |

**解讀。**

（i）逐欄最強基線不同，不能合成一個假 SOTA：Ret 20d 的 TS2Vec `.1454` 高於 MC-JEPA `.1389`；Vol 120m 的 CPC `.552` 高於 MC-JEPA `.517`；Beta 20d 的 CPC `.391` 高於 MC-JEPA `.374`；Table 8 的 MDD 5d 上 Alpha158 `.590` 高於 CPC `.545` 與 MC-JEPA `.525`。Chronos-2 只在 `both volume targets` 為次佳，不能擴及全部 liquidity。

（ii）真正像 capability 的是 **risk／liquidity** 一段：對 per-asset JEPA 的配對 Newey–West t 值在 exposure、risk、liquidity 上 `ranging from 6.9 to 49.4`，且 Table 12 顯示 `3/3` seeds 一致——與 Prop. 1 的曝險機制同向，機制與數字互相印證。

（iii）**return 一段要打折**：來源自陳 Ret 5d 對 per-asset JEPA `t=1.8`（未過 1.96）、Ret 20d `t=2.2`，分鐘 return 的四個 horizon `None of the four horizons is significant against per-asset JEPA`；且明言測試為 `unadjusted tests at the 5% level`，所讀文本未見多重比較校正。

（iv）**TSFM 的勝負要分 horizon 講**：來源稱 `Every TSFM is below MC-JEPA on all three return horizons`，指的是 Ret 120m／Ret 1d／Ret 5d；Ret 20d 不成立：Chronos-2 `.1390`、Kronos-mini `.1395`、Kronos-small `.1418`、Kronos-base `.1409` 均高於 MC-JEPA `.1389`。Tables A 與 Table 12 不可互相代稱：表 A 是期間平均 RankIC 的兩個來源值並列；Table 12 給的是「日度 RankIC 差的均值」（例如 Ret 1d 對 per-asset JEPA 為 `0.0034`、對 placebo 為 `0.0026`；Ret 20d 對 per-asset JEPA 為 `0.0052`），定義不同。

（v）**回測**：來源原文稱 `raises excess return by 1.8 points and lowers drawdown from 0.132 to 0.117`，MC-JEPA 自身三種子標準差為 `0.011`（AER）與 `0.001`（MDD）；本頁推斷 MDD 的改善比 AER 穩定。但組合僅 `7–8` 檔、`20` 日持有、帳本約 `150` 檔，且成本只有固定 `5/15bp`——換手率數值、資金容量曲線、滑價敏感度皆所讀文本未見。

（vi）**別把歸因搞混**：K=1 只在 Ret 1d／Ret 5d／Ret 20d 最好；Ret 120m 的 K=4 為 `.0702`，高於 K=1/MC-JEPA `.0694`。K=32 時 Beta 20d 達 `.382`，高於 K=1 的 `.374`——「去噪比例」與「return 增益」不單調對應，條件至少走了兩條路徑，作者在 §4.5 亦承認 denoising does not account for all of it。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（§6 逐項；另標 Appendix E 者為作者自評量級）

1. **單一市場、單一頻率**：全部實證為中國 A 股、單一 bar 頻率；作者點名 heavy retail participation、daily price limits、T+1 settlement 可能影響市場衝擊的傳導，其他市場／資產類別／頻率的增益 `remains untested`（§6）。
2. **命題與實作的落差**：Prop. 1 假設**線性因子模型且已知真實因子**，MC-JEPA 實際用的是**估計出來的 leave-one-out 市場條件**（§6；§3.3 亦明言 `c_{−i,h}` 只是 `a noisy proxy for the realized market factor`）。
3. **K=1 的表達力**：只取單因子，industry／style 等更豐富的截面結構 `left to be captured implicitly`（§6）。
4. **載荷不是 beta**：輸入與 target 皆逐窗標準化，學到的載荷抓的是「與市場的共動」而非 beta；乘上窗口價格波動率才回復截面 beta 排序；作者建議 `Targets that preserve each asset's scale could make the loadings a beta directly`（§6、§4.4、Appendix F）。
5. **去噪量級有限（Appendix E）**：條件只移除 `4.5±0.9%` 的無條件誤差，而 panel 均值解釋 `26%` 的變異；K 加到 8 以上亦 `levels off below 9%`，作者自評 `the denoising effect on real data is modest`。

### 6.2 本頁推斷的隱含假設（非作者自述，尚待驗證；圖表／附件待核）

- **regime 依賴**：訓練截止 `before July 2022`、測試 `2023-01 to 2025-12`，僅一次跨 regime 切分；來源描述其 leads from early 2024 並在 2025 drawdown 期間保持領先，但單一路徑無從判斷 A 股微觀結構變化是否為增益來源。若個股 beta 離散度壓縮，loading 通道的信息量會下降——機制推論，未測試。
- **容量／成本**：`5/15bp` 對 CSI300 這類高流動性池子或合理，但換到中小盤或放大資金後成本結構改變；換手率與容量曲線所讀文本未見 → TBD。
- **條件池的截面結構**：`c_{−i,h}` 是 `M_t` 的等權均值，`M_t` 於 `t` 附近均勻隨機抽樣；若改成市值加權，行為可能不同。注意 Beta 20d 的市場定義本身即等權均值（§D.1），兩者對齊，此對齊未被作者討論。
- **數據泄漏邊界**：條件使用**其他資產的未來 target embedding**，屬明確的特權信息設計；防抄答案靠 leave-one-out 排除自身。可反證證據來自 placebo：改用不同 decision time 構造條件，`recovers some of the gains on volatility and volume but not in returns or market exposure`，且對 placebo 在九個 target 全數顯著——這是本頁認為最扎實的一段。
- **survivorship**：回測用 point-in-time CSI300 成分股（已緩解）；但 `Suspended days are removed…a held stock earns zero return while suspended` 是**假設**而非事實：複牌跳空未建模，方向未定。輔助數據缺失與此無關。
- **選模自由度**：λ、K、pool size 皆在 `2022-07 to 2022-12` 選定，來源稱在測試期 `also peak there`；Table 3 的變體數 × 9 個 target 意味選模自由度不低，測試期恰好一致需保留懷疑。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Per-asset JEPA（λ=0，同架構同數據） | 有無對齊的市場條件；唯一乾淨的單因子隔離 | 同 repo（推斷） | 來源稱 `improves all nine targets`；但 Ret 5d 對其 `t=1.8` 未過 1.96 |
| TS2Vec / CPC | 對比式自監督；在部分欄勝出 | 是（各自官方實作） | TS2Vec Ret 20d `.1454` > MC-JEPA `.1389`；CPC Vol 120m `.552` > `.517`、Beta 20d `.391` > `.374`；CPC 是 Table 8 三個日頻 risk target 的最佳 encoder（Vol 1d `.601`、Vol 5d `.581`、MDD 5d `.545`），但 Alpha158 的 MDD 5d `.590` 更高 |
| Chronos-2 / Kronos / TimesFM-2.5 / MOMENT | 凍結 TSFM；per-series、無跨資產訓練信號 | checkpoint 皆公開（Table 6） | 在 Ret 120m／Ret 1d／Ret 5d 全數低於 MC-JEPA；Ret 20d 不成立（Chronos-2 `.1390`、Kronos-mini `.1395`、Kronos-small `.1418`、Kronos-base `.1409` 均高於 `.1389`）；Chronos-2 在 `both volume targets` 次佳 |
| FactorVAE（Duan et al., 2022） | 同樣用未來截面資訊當 privileged information，但**監督式、日頻工程特徵、推理仍需截面聚合** | 未核 | 概念最近的前作 |
| HIST / MASTER 等截面神經模型 | 推理期需 graph／inter-stock attention；MC-JEPA 推理純 per-asset | 未核 | 部署形態差異大於精度差異 |
| 傳統 trailing beta（20/60/120 日） | 無學習、推論需全市場報酬 | 是 | historical betas RankIC `0.525`，`z_i+trailing betas` 達 `0.539`；另 Table 14 的 trailing 4-day beta 為 `.434`，不可與 `0.525` 混為一談 |
| 無標籤載荷估計（raw loading／×CV） | 無需標籤的曝險讀出 | 是 | raw loading RankIC `-0.06`；×`CV_i` 為 `.368`；監督 probe 為 `.374`；`CV_i` 單獨為 `.287` |

**🎤 Interview Tip**

- **問**：「你把市場因子餵給 predictor，不就是訓練期洩漏未來？」
  **正確答**：是特權信息，不是洩漏——(a) 條件排除自身（leave-one-out），(b) 只在訓練期存在，推理端 `z_i=f_θ(x_i)` 只依賴自身歷史，(c) 條件對預測的額外作用必須經 `b_ψ(z_i)`，市場項無法單獨解釋 target。**錯答**：「JEPA 本來就不重建輸入所以無所謂」——那是講表徵空間，跟這裡的資訊邊界是兩件事。

- **問**：「為什麼 K=1 是對的？」
  **正確答**：K=1 是**任務相關的選模結果**，不是普遍最優：Ret 1d／Ret 5d／Ret 20d 在 K=1 最好，risk 與 exposure 則偏好 K>1（K=32 時 Beta 20d `.382`）；Ret 120m 反而是 K=4 的 `.0702` 高於 K=1 的 `.0694`。選模依 validation 的 Ret 1d／Ret 5d。**錯答**：「單因子就是 CAPM，所以 K=1 理論上完備。」

- **問**：「載荷可以直接當 beta 用嗎？」
  **正確答**：不行。raw loading 對 Beta 20d 的 RankIC 只有 `-0.06`；乘上窗口 close 的 `CV_i` 後升到 `.368`，接近監督 probe 的 `.374`；但 trailing 4-day beta 是 `.434`，20/60/120 日 historical betas 是 `0.525`，三者不可互代。**錯答**：「loading 就是 beta，因為它是單指數模型的載荷。」

### 7.1 可證偽預測（本頁提出，尚待驗證）

1. **2027-12-31 前**：若在美股或港股、或 5 分鐘頻率重跑，MC-JEPA 對 per-asset JEPA 的增益應仍集中在 exposure／risk／liquidity，而 return 增益在多數 horizon 上不顯著（依 Appendix E：條件僅移除 `4.5±0.9%` 的無條件誤差）。若 return 增益反而更大且跨市場穩定，則「return 增益來自 asset-specific 表徵變乾淨」的解釋需修正。
2. **2027-06-30 前**：若把 target 改為**保留資產自身尺度**（作者 §6 建議），則 raw loading 對 Beta 20d 的 RankIC 應由 `-0.06` 顯著轉正、且不需乘 `CV_i`；若仍為負，表示載荷抓的是相關係數而非標準化造成的問題。
3. **可反證的失效點**：若把 `M_t` 改成市值加權（而非等權均值），而 Beta 20d RankIC 反而下降，則「等權因子與等權 target 定義對齊」是現有增益的部分來源。

## §8 · For the Reader

- **因子研究員**：可直接搬的是「未來截面均值當訓練期正則」這個 pattern——不必換 backbone，只要在現有 per-asset encoder 上加一條 loading 通道與 LOO 條件，就能測「我的表徵到底有沒有編碼曝險」。注意 return 增益多半不顯著（Ret 5d `t=1.8`），把它當**表徵診斷工具**更划算：載荷×`CV_i` 在無標籤下對 Beta 20d RankIC `.368`，接近監督 probe 的 `.374`。
- **高頻執行**：推理端純 per-asset、`960` 根 1 分鐘 bar、`9.52M` 參數，延遲與一般時序 encoder 同級——這是它相對截面模型的真實賣點。但回測只給固定 `5/15bp`、`7–8` 檔集中持倉，換手與容量曲線所讀文本未見，上實盤前須自行補做。
- **組合配置**：`z_i + trailing betas` 對 Beta 20d 的 RankIC 達 `0.539`（vs 純表徵 `.374`），意味學到的曝險信息與傳統估計**互補而非替代**；若已有風險模型，邊際價值在 Vol／MDD 這類 risk target。但 MDD 5d 上 Alpha158 `.590` 高於 CPC `.545` 與 MC-JEPA `.525`，不要假設深度表徵全面壓過工程因子。
- **LLM-agent／RL 策略**：本方法沒有決策或執行模組，五軸 `autonomy=全自動黑盒` 指的是「訓練完即凍結、下游線性 probe」；要接 agent 的話，價值在提供一個**無截面依賴的狀態編碼器**。
- **研究學生**：這是「一個架構約束 → 一個可證命題 → 一個可反證的 placebo」的範式：Prop. 1 給必要性，placebo（錯位日期）給對齊性，synthetic panel 在已知參數下檢驗 recovery，回報的是 linear-probe R² 的提升（曝險 probe R² 由 `0.51–0.68` 升至 `0.69–0.80`；asset-specific 由 `0.62–0.78` 升至 `0.79–0.85`），**不是上界**。想抄方法論，抄這三段，不要只抄 loss。

## References

- 原文：Yuqi Wang, Fengyuan Liu, Mukai Li, Haochen Luo, Lei Li, Qi Liu. *Market-Conditioned JEPA: Learning from Unpredictable Market Shocks.* arXiv:2611.13572v1 [cs.LG], 10 Nov 2026. https://arxiv.org/abs/2611.13572v1　·　Code: https://github.com/OpenFinMind/MC-JEPA
- Lineage（JEPA）：Assran et al., 2023, *Self-supervised learning from images with a joint-embedding predictive architecture*, arXiv:2301.08243.
- Lineage（因子結構／特權信息）：Sharpe, 1964（CAPM／single-index）；Fama & French, 1993；Gu, Kelly & Xiu, 2021（conditional autoencoder）；Duan et al., 2022（FactorVAE，未來截面報酬作 privileged information）。
- Lineage（時序自監督與 TSFM 基線）：Nie et al., 2023（PatchTST）；Yue et al., 2022（TS2Vec）；van den Oord et al., 2019（CPC）；Goswami et al., 2024（MOMENT）；Das et al., 2024（TimesFM）；Ansari et al., 2025（Chronos-2）；Shi et al., 2025（Kronos）；Yang et al., 2020（Qlib／Alpha158；回測年化慣例）。
- 相關近期工作（原文引用，未核）：Luo et al., 2026（STOCK-JEPA）；Merchant et al., 2026（Towards financial world modeling）；Rahimikia et al., 2025；Alonso & Franklin, 2026（TSFM 在財務報酬上增益有限）。