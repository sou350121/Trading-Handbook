<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=生成式大模型 alpha=端到端表征 autonomy=全自动黑盒 -->

# WGAN + EGGROLL 解構（WGAN + EGGROLL）

> **來源發布**：2026-10-19 · ICAIF26 · arXiv [2610.25604](https://arxiv.org/abs/2610.25604)
> **arXiv 原文**：[Adversarial Post-Training of Foundation Models for High-Frequency Trading](https://arxiv.org/abs/2610.25604v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「微觀盤口 / 高頻日內 / 生成式大模型 / 端到端表征 / 全自動黑盒」。解決的 prior gap 很具體：AR 式 LOB 基礎模型用 cross-entropy 預訓練，既不懲罰自由滾動中的誤差累積（exposure bias），也瞄準不了想要的市場屬性；而撮合引擎不可微，梯度回不到生成器。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `生成式大模型` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 在預訓練 LOB 基礎模型之後加一個對抗後訓練階段，critic 在真實不可微撮合引擎吐出的 rollout 上**從零在線訓練**。② 生成器更新不靠梯度，靠 EGGROLL 在 rank-4 LoRA 子空間做 KL 信任域內的進化擾動。③ 對「生成式大模型」這條軸，關鍵在於換掉了目標函數而不是加大預訓練——IsoFLOP 續預訓練控制只到 +0.5%（[-2.3, +3.4]），換不來同樣增益。④ 十個 seed 的 sealed test：WS-21 −17.6%、Mean-L₁ −17.0%，代價不到 50 M training tokens 與 12 GPU-hours/seed。

**X-Ray.** 這篇的定位不是「更強的預測器」，而是「更省錢的模擬器精修器」。放回五軸 Pareto，它在「時間尺度＝高頻日內」與「Alpha＝端到端表征」上其實沒有推進 alpha，而是在「範式＝生成式大模型」這條軸上把一個工程障礙（不可微環境）轉成了可優化物件。它解掉的舊坑很典型：ES 在這個圈子被當成「只有 from-scratch 超大 population 才划算」的東西，作者則論證後訓練鄰域稠密、G=16 就夠；同時它沒有迴避 reward hacking，反而把 hacking 窗口當成可測量對象，用一份不含 critic 信號的外部選模協議去紀律化它。它打不開的 envelope 也很清楚：critic 只看訊息流、看不到 standing book，於是 spread 反而惡化 +34%；所有 rollout 只跨 500 條訊息；只測 GOOG。對量化讀者的意義是——這是一條「合成訂單流」的基礎設施，不是策略；把它當 alpha 源會失望，把它當壓力測試與 RL 環境的底座才有意義。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（token-level AR LOB 基礎模型） | 本文 |
|---|---|---|
| 訓練目標 | 下一 token 交叉熵（條件分佈） | Wasserstein-1 對抗目標，critic 在線訓練、從零開始，K 個成員取悲觀分 |
| 更新途徑 | 梯度回傳 | 梯度不穿撮合引擎 → EGGROLL 零階／進化策略 |
| 更新參數 | 全參微調 | 凍結主幹，僅 SSM 內部投影核以 rank-4 LoRA 因子擾動並全量更新 |
| 正則化 | 無對應機制 | KL 信任域直接塞進 fitness |
| 選模 | 驗證 loss | 外部 LOB-Bench 網格選模（不含 critic 信號）+ sealed test |
| 資料輸入 | 常聚焦 top-k 檔位 | 完整訊息流，不截檔 |

### 1.2 ⚡ Eureka

**把「不可微」從障礙改寫成「可評估的適應度函數」**：既然撮合引擎不能反傳，就不要反傳——直接在 rank-4 LoRA 子空間採樣成對擾動，用「悲觀集成分數 − KL 罰」當每條擾動的 fitness，按 context 做中心化排序再加權合成一次更新。直覺是：預訓練模型周圍的改進方向是稠密的，你不需要找到唯一正確的梯度，只需要在足夠多的方向裡按秩篩出「平均而言更好」的那些。

### 1.3 信息流

真實 LOB 訊息流 ──┐
　　　　　　　　　├─► Critic 集成（K 個成員） ─► 悲觀分 mean−std ─┐
生成 rollout ─────┘　　　　▲　　　　　　　　　　　　　　　　　　　 │
　　　　　　　　　　　　　 │ 在線訓練（R₁ 零中心梯度罰項）　　　　 ├─► 逐 context 中心化秩
　　　　　　　　　　　　　 └──── 真／假窗口對 ◄─ 真實窗口池 ───────┘　　　× KL 信任域罰
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 │
　 θ + Σ wᵢ (Aᵢ Bᵢᵀ) ◄── Adam ◄── EGGROLL 更新 ◄── 跨 context 加權平均 ◄┘
　　　　（僅 SSM 內部投影核，rank-4 LoRA 因子）

生成側走左路：模型在真實撮合引擎裡滾 500 條訊息；評分側走中路：critic 同時看真窗口與假 rollout，並被 R₁ 罰項壓平在資料流形附近；更新側走右路：分數→秩→加權→LoRA→Adam，全程不經過引擎的導數。

## §2 · 數學層

📌 **Napkin Formula**

- critic 目標：`L_D = E[D(gen)] − E[D(real)] + (γ/2)·E_{x~real}[‖∇ₓD(x)‖²]`（Wasserstein-1 對偶 + 零中心 R₁ 梯度罰）
- 適應度：`f_i = (mean_k D_k(rollout_i) − std_k D_k(rollout_i)) − λ·KL(π_θi ‖ π_anchor)`
- 更新：`θ ← Adam_η( Σ_q Σ_i ŵ_{i,q} · (A_i B_iᵀ) )`，`ŵ` 為逐 context 中心化秩分數的跨 context 平均

**複雜度**：每步 G×Q = 16×128 = 2,048 條 rollout；每條 500 條訊息；K 個 critic 成員各評一次。真正的成本不在參數量（生成器 78.5 M），在 rollout 數量——這也解釋了為什麼作者把預算花在 Q（每方向 128 個共享 context）而不是 G。

**直覺**：秩變換把分數壓成有界量，單一 context 的離群高分不會主宰更新；悲觀分（mean − std）把「成員之間意見不一致」當成懲罰；KL 罰則是把「離錨點太遠」直接定價進適應度，而不是事後裁剪。

**訓練細節**：critic 每次抽 Q=128 個 context（無放回）跨擾動共享，避免 context 抽樣噪聲污染 fitness；真實窗口池 512 個，每五步重抽一次，讓 critic 的真集在短期內近似平穩又不至於記住個別窗口。生成器端 30 步為頭條配置，checkpoint 每兩步存一個。

## §2.5 · 帶數字走一遍（Worked Example）

以下**全部為假設／示意值**，只為把「分數怎麼變成權重」手算清楚，不是論文實證結果。

**Step 0｜設定（假設）**：G = 4 個擾動方向（論文為 16），Q = 2 個共享 context（論文為 128），K = 3 個 critic 成員（論文為 5），rank r = 4，σ = 0.003（此值為原文真值）。

**Step 1｜採樣**：方向 i 抽兩個低秩因子 Aᵢ、Bᵢ，擾動 = AᵢBᵢᵀ（秩 4），係數 σ。四條擾動模型各在兩個 context 上滾 500 條訊息。

**Step 2｜打分（假設值）**：critic 集成給出悲觀分，再扣 KL 罰後得 fᵢq：

| 方向 i | fᵢ(ctx1) | fᵢ(ctx2) |
|---|---|---|
| 0 | 2.0 | 1.0 |
| 1 | 4.0 | 3.0 |
| 2 | 3.0 | 4.0 |
| 3 | 1.0 | 2.0 |

**Step 3｜逐 context 中心化秩**：G=4，升序排名 0..3，減去均值 1.5 → 分數落在 {−1.5, −0.5, +0.5, +1.5}。
- ctx1：方向 3 → −1.5；方向 0 → −0.5；方向 2 → +0.5；方向 1 → +1.5
- ctx2：方向 0 → −1.5；方向 3 → −0.5；方向 1 → +0.5；方向 2 → +1.5

**Step 4｜跨 context 平均**：方向 0 = (−0.5 + −1.5)/2 = −1.0；方向 1 = (+1.5 + +0.5)/2 = +1.0；方向 2 = (+0.5 + +1.5)/2 = +1.0；方向 3 = (−1.5 + −0.5)/2 = −1.0。

注意這裡的重點：方向 1 在 ctx1 的原始分 4.0 遠高於方向 2 的 3.0，但兩者最終權重同為 +1.0——**秩變換把「單一 context 的高分」壓成有界訊息**，方向 1 在 ctx2 掉到 +0.5 就立刻反映出來。這正是作者不用 raw advantage 而用 rank transform 的原因。

**Step 5｜合成更新**：Δθ = Σᵢ wᵢ(AᵢBᵢᵀ)，四個秩-4 矩陣按 ±1.0 加權疊加，交 Adam 更新僅 SSM 內部投影核。

**Step 6｜KL 罰的量級（假設值）**：若方向 1 的 KL = 0.01、λ = 0.1，罰項 = 0.001——相對分數尺度 4.0 是一階小量，單步幾乎不改變排序。它的作用在累積：原文量到 step 10 到 step 30 之間 KL 由 2.0 升至 6.2×10⁻³，漂移是逐步長出來的，不是單步被剪掉的。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 標的 | GOOG 單一股票 |
| 訓練期 | 2022 至 2025，約 1,000 個交易日 |
| 測試期 | 2026 年 1–2 月的 28 個交易日（sealed panel） |
| 頻率 | 訊息級（order submission / cancellation / deletion / execution），奈秒時間戳 |
| 格式 | LOBSTER；價格以 10⁻⁴ USD 計；成交量截斷在 9,999 股 |
| 清洗 | 丟棄盤外訊息、隱藏成交、交叉成交、停牌指示 |
| 切分 | 預訓練僅用約 50% 訓練語料；後訓練在**未見過的訓練資料**上做 |
| Token | 每則訊息 26 tokens，共享詞表 2,112 個 token（含四個保留特殊符號） |
| 上下文 | 500 條訊息 ≈ 13,000 tokens；top-k 取樣 k=50 |

**樣本外設計**是本篇相對乾淨的一環：選模用的窗口抽自訓練時段，sealed test 用最接近當下的 28 天未見資料，兩者不重疊；且後訓練階段用的是預訓練沒碰過的語料，避免「新資料特權」。作者對測試期有做體檢——Table 1 逐期列出 Msgs/day、Exec.%、Spread、Daily RV、Mid，結論是「test period shows no dramatic regime shift」，事件組成與波動率跨期穩定（**具體數值見原文 Table 1，本頁不重列**）。

**容量假設**：單股票、單一年份跨度。跨股票、跨交易所、跨資產類別的遷移性，所讀文本未見，圖表／附件待核。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | `github.com/alexandre-bismuth/WGAN-Post-Training`（原文腳註 1） |
| License | 論文為 CC BY 4.0；程式碼授權**所讀文本未見** |
| Checkpoint | 未披露——78.5 M 參數的預訓練 Mamba-3 生成器是否隨附，所讀文本未見 |
| 複現難度 | 高：需 GPU 原生撮合引擎 + 預訓練權重 + LOBSTER 全量 GOOG |
| 數據可得性 | LOBSTER 為商業資料；作者是否另行釋出未披露 |
| 算力 | 12 GPU-hours/seed；文中提及單張 GH200 上約 50k tokens/s 的生成吞吐 |

## §5 · 評測 / Benchmark

主表只列原文以**配對 Δ** 報告的主結果——注意原文沒有給出 WS-21 與 Mean-L₁ 的絕對水準，因此前SOTA／本方法兩欄只能標未披露，Δ 欄是原文表 2 的逐字值。

| 數據集／市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| GOOG · 28 交易日 sealed panel（4,096 windows） | ΔWS-21（相對凍結預訓練 anchor） | 未披露（原文僅給配對 Δ） | 未披露（原文僅給配對 Δ） | −17.6% |
| 同上 | ΔMean-L₁（相對凍結預訓練 anchor） | 未披露（原文僅給配對 Δ） | 未披露（原文僅給配對 Δ） | −17.0% |

配對 95% bootstrap 區間分別為 [−19.7, −15.3] 與 [−18.9, −15.8]，十個 seed 的均值與逐 seed 都排除了零；逐 seed WS-21 落在 −11.3% 到 −23.9% 之間。

**三個控制臂（逐字對照，不合成區間）**：
- **Selection-null**（N=8 個隨機 LoRA 擾動，走完全相同的選模與 sealed 流程）：ΔWS-21 **+5.3%**（[+2.5, +8.1]）、ΔMean-L₁ **+7.8%**（[+5.9, +9.6]）。這是「選模本身會製造虛假增益」這一假說的正面反證——管線不但沒製造增益，還把隨機擾動選成了更差的模型。
- **IsoFLOP 續預訓練**（同語料、同食譜、算力對齊單一後訓練 seed）：ΔWS-21 **+0.5%**（[−2.3, +3.4]）、ΔMean-L₁ **−0.4%**（[−2.2, +1.4]），區間全跨零。
- **錯誤累積**：anchor 的散度在 500 條訊息內成長 **3.1×**，後訓練後降到 **2.3×**，多餘散度少 **26%**（[−33%, −20%]）；selection-null 反而多累積 **27%**（[+15%, +37%]）；續預訓練 **+3.9%**（[−6%, +14%]），等於沒動。

**哪些 Δ 是真 capability**：兩條互相獨立的度量同向——分佈層面（LOB-Bench 的 21 個邊際統計）與逐位置層面（KL(true‖gen) 的累積曲線）都改善，且三個控制臂同時落空，這是很難靠單一 pipeline 偏差偽造出來的組合。

**哪些要打折**：
1. **同源風險**。critic 在訊息流上學判別，評估用的 LOB-Bench 也是同一族微結構邊際統計。雖然選模刻意不用 critic 信號、sealed test 用未見時段，但「用 A 族統計訓練、用 A 族統計驗收」的結構沒變。所讀文本未見任何獨立於 LOB-Bench 的第三方分佈檢定。
2. **spread 反例**。作者自承 spread **惡化 +34%**——這是「改進不是全分佈的」的直接證據：critic 能看到的邊際改善了，critic 看不到的 book-level 聯合量漂移了。
3. **下游是零結果而非正結果**。mid-price 三分類方向準確度在 100、250、500、1,000 條訊息四個 horizon 上分別為 **+0.011**（[−0.008, +0.031]）、**+0.012**（[−0.007, +0.028]）、**−0.004**（[−0.024, +0.016]）、**+0.009**（[−0.010, +0.027]），全部跨零。作者把它讀成「能力被保留」，但這同時意味著**沒有任何預測力增益**；而且作者自己指出 anchor 與真實走勢的相關性在約 400 條訊息內就衰減到零，所以這個檢定的鑑別力本來就有限。
4. **成本項只計訓練**。12 GPU-hours/seed 不含評估——選模要掃 checkpoint 網格、sealed test 要跑 4,096 個窗口 × 500 條訊息。評估成本所讀文本未給。

## §6 · 失效與隱含假設

### 6.1 論文自述的限制（僅列可定位者）

- **§7.1**：`all rollouts here span 500 messages`，更長 horizon 是否能進一步壓低誤差累積，作者明言 remains open。
- **§6.2.1**：`the critic scores the message stream but never observes the standing book`——spread 這類 book-level 聯合量只被間接看到，這是 +34% 惡化的機制解釋，作者把它列為需要後續用「可替換 reward 模組」重定向的對象。
- **§7.1**：market-impact fidelity 仍是 future work，尚未實作；作者把它點名為訓練交易 agent 的前置條件。
- **§4.1**：資料為單一股票 GOOG、訓練 2022–2025、測試 2026 年 1–2 月。
- **§5.1.1**：reward hacking 窗口是結構性的——三個 seed 在約 15–20 步後開始利用 critic，seed 1 到第 50 步甚至回到 +25.4%；因此最後一個 checkpoint 不可部署，必須外部選模。
- **§6.3**：下游只測了 mid-price 方向，且該訊號本身極弱（約 400 條訊息內衰減到零）。

### 6.2 推斷的隱含假設（本頁推斷，非作者主張）

- **撮合引擎即真值**。整條管線的目標函數由引擎語義定義；若引擎與真實交易所的微觀規則有偏差，critic 會忠實地把生成器拉向那個偏差。所讀文本未見引擎實作與交易所規則的逐條對照。
- **21 個邊際統計＝真實度的充分代理**。這是 Goodhart 的標準入口。作者用外部選模與 sealed test 削弱了它，但沒有消除——評估與優化仍共用同一個統計族。
- **單票單段的平穩性**。測試期「no dramatic regime shift」是作者的觀察，不構成對其他 regime（急跌、熔斷、流動性枯竭）的保證；跨標的遷移性完全未測。
- **容量與衝擊未觸及**。模擬器保真度提升不等於可承載策略容量；本文沒有注入訂單、沒有量測衝擊響應，這一整條軸是空的。
- **成本與延遲不在模型內**。所有指標都是統計距離，沒有任何交易成本、滑價或延遲建模。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| token-level AR LOB 基礎模型（Nagy 2023；MarS 2025） | 純 next-token CE，無對抗階段 | 所讀文本未列授權 | 本文的 anchor／基線 |
| Stock-GAN（Li 2020） | 用神經網路近似撮合以繞開不可微 | 所讀文本未見 | 前作，非基礎模型 |
| CGAN world agents（Coletta 2021/2022） | 條件 GAN、多 agent 環境內部署 | 所讀文本未見 | 前作 |
| TRADES（Berti 2025） | 擴散去噪生成 | 所讀文本未見 | 前作 |
| SeqGAN 式 REINFORCE / GRPO 臂 | token 空間策略梯度 | GRPO 開源 | 本文消融臂：ΔWS-21 −16.0%、ΔMean-L₁ −17.3% |
| 擴散模型的對抗後訓練（Lin 2025） | 精神最接近，但無撮合引擎 | 所讀文本未見 | 靈感來源 |
| ABIDES（Byrd 2020） | 手工 agent 池，可互動但難貼合真實訂單流 | 開源 | 互補而非競爭 |

**🎤 Interview Tip**

問：「WS-21 降了 17.6%，是不是代表這個模擬器可以拿去訓 RL 交易 agent 了？」

- **錯答**：「是，模擬器更真實了，agent 的訓練環境就對齊了現實。」
- **正確答**：「不行。論文證明的是兩件窄的事：在 GOOG 28 天 sealed panel 上，21 個 LOB-Bench 邊際統計的平均 Wasserstein 距離下降 17.6%；以及 500 條訊息 horizon 內逐位置 KL 累積變慢（3.1× → 2.3×）。它**沒有**測 market-impact fidelity——作者自己在 §7.1 列為 future work，而 RL agent 最需要的恰恰是注入訂單後的反事實價格響應。更關鍵的是 spread 反而惡化 +34%，說明 book-level 聯合量漂移了，因為 critic 從來沒看過 standing book。下游 mid-price 方向準確度在所有 horizon 上區間跨零，也說明增益不在預測力上。正確的定位是：這是**模擬器精修**，不是策略增益。」

### 7.1 可證偽預測（本頁提出，尚待驗證）

1. **2027 年前**若把同一 pipeline 換到第二隻流動性相近的股票，sealed ΔWS-21 應仍落在 **−11.3% 到 −23.9%** 這個逐 seed 區間附近；若明顯收窄到個位數百分比，則「單票擬合」的解釋成立。
2. 若把 spread 這類 book-level 聯合量加入 critic 輸入或 reward，**+34%** 的 spread 惡化應顯著收斂；若否，問題不在 critic 視野而在別處（例如引擎本身對 touch 的處理）。
3. 把 rollout horizon 從 500 拉長，**2.3×** 這個累積比率應回升——作者已明言此軸未測，這是直接可證偽的。
4. reward hacking 轉折點會隨 G 增大而提前：原文已觀察到 G=16 中位選到 20 步、G=32 與 G=64 選到 15 步；若在更大 G 上轉折點不提前，「更銳的梯度估計 → 更快撞上 hacking 窗口」這個機制解釋就有問題。

## §8 · For the Reader

- **因子研究員**：這篇不給你 alpha。可用的地方在於「受控合成訂單流」——但先問一個問題：spread 漂移 +34% 會不會污染你要測的價差／流動性類因子？如果要拿它做因子穩健性壓力測試，務必先在你的因子族上量一次 spread 敏感度。
- **高頻執行**：12 GPU-hours/seed 換 17.6% 的真實度提升，成本結構是這篇最實用的部分。但你的執行模型依賴 impact，而 impact 這條軸本文完全沒碰——目前不能用它校準執行成本。
- **組合配置**：現階段不可上生產。單票、單段、無成本建模、下游方向預測零增益。可觀察的是它作為「情境生成器」的潛力，但要等跨標的驗證。
- **LLM-agent / RL 策略**：這篇最有遷移價值的是**範式**而非數字——critic 是一個可替換的 reward 模組，你可以把它換成 impact、換成做市收益、換成任何能從 rollout 算出來的量。同時它給了一份誠實的 reward hacking 時間表（15–20 步後開始利用 critic），這在設計你自己的 reward 時可直接借用。
- **研究學生**：最值得抄的不是 WGAN 也不是 EGGROLL，是**控制設計**——selection-matched null（用隨機擾動走完整條選模管線，證明管線不製造增益）與 IsoFLOP 控制（證明增益不是「多算一點預訓練」的代理）。這兩個控制把一篇容易被打回的方法文，變成了一篇因果歸因站得住的方法文。

## References

- Bismuth, A., Li, K., Marti-Shahandeh, A., Mohl, V., Foerster, J., Calinescu, A. *Adversarial Post-Training of Foundation Models for High-Frequency Trading.* ICAIF '26. arXiv:2610.25604v1. https://arxiv.org/abs/2610.25604
- 代碼：https://github.com/alexandre-bismuth/WGAN-Post-Training
- **Lineage（GAN 側）**：Arjovsky et al. 2017（Wasserstein GAN）；Gulrajani et al. 2017（WGAN-GP）；Heusel et al. 2017（TTUR）；Mescheder et al. 2018（零中心 R₁ 罰項）；Miyato et al. 2018（spectral normalization）
- **Lineage（零階／後訓練側）**：Salimans et al. 2017（ES）；Sarkar et al. 2025（EGGROLL / Evolution Strategies at the Hyperscale）；Shao et al. 2024（GRPO）；Gao et al. 2023（reward model overoptimization）；Gan & Isola 2026（neural thickets）
- **Lineage（市場模擬側）**：Nagy et al. 2023（token-level AR LOB）；Nagy et al. 2025（LOB-Bench）；Li et al. 2025（MarS）；Wheeler & Varner 2024（MarketGPT）；Li et al. 2020（Stock-GAN）；Coletta et al. 2021/2022（CGAN world agents）；Berti et al. 2025（TRADES）；Byrd et al. 2020（ABIDES）
- **基礎設施**：Hu et al. 2022（LoRA）；Lahoti et al. 2026（Mamba-3）；Frey et al. 2023（JAX-LOB）；Huang & Polak 2011（LOBSTER）