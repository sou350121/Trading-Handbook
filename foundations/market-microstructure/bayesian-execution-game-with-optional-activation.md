<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=因果结构 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Bayesian execution game with optional activation 解構（Bayesian execution game with optional activation）

> **來源發布**：2026-11-19 · （無 venue） · arXiv [2611.25762](https://arxiv.org/abs/2611.25762)
> **arXiv 原文**：[Trading Before a Committed Rebalance](https://arxiv.org/abs/2611.25762v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 高頻日內 × 因果結構 × 組合執行優化 × 人機協同可解釋」。它解的不是預測精度，而是**把「公共預測 → 可選庫存供給 → 機構執行帳單」寫成可精確求解的貝葉斯數量博弈**，把信息價值從 RMSE 重構成「均衡成本函數 ψ 的曲率」。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `高频日内` | `因果结构` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①建構帶可選激活（optional activation）的貝葉斯執行博弈：機構必須完成外生 target，交易者觀察同一公共信號後自行決定是否付固定成本進場備貨；②核心 trick 是把均衡後的機構成本壓成單一後驗均值函數 ψ(μ)，信息排序等價於 ψ 的凹性，因而可用 Blackwell 比較逐點判定；③對「因果結構／執行優化」這條軸★，它給出反直覺判準——**預測更準不等於執行更便宜**，固定參與名單下信息必然有益，但可選激活在 entry 與滿參與兩處可反轉排序；④實證僅為描述性：所有擬合均值預測的 pooled RMSE ratio 都高於零剩餘收益基準（如 Returns + range + count/size 為 1.0168），而 range-scaled 區間帶在 2023–2025 的 interval score 為 342.38 bp、Raw 為 384.96 bp。

**X-Ray.** 這篇在五軸 Pareto 上把「高頻日內」與「組合執行優化」的交集推到一個很窄但很深的位置：不建模限價簿微結構、不估計實際 impact 係數，而是把衝擊衰減、排程整合、終端統一價格拍賣全部參數化成 (K, H, λ, η₀, η)，換取**閉式均衡 + 精確有限 N 修正**。它解掉的舊工程坑是：以往「信息有害」文獻（Crego 2020、Diamond–Verrecchia 1991）只能停在「參與改變」的定性層面，而執行博弈文獻（Campbell–Nutz 2026、Fu et al. 2024/2026）處理瞬態衝擊下的最優執行或均值場清算，但未把「公開預測誘導的可選激活」寫進同一個可逐點判定的現金帳比較。本文用 exact cash account 把「庫存有益」與「信息有益」拆成兩個不同命題。它打不開的 envelope 很清楚：排程 φ 外生（不能等更晚信號，作者自述為 extension）、風險中性（後驗方差只進常數項、不改部位）、線性終端供給（非策略交易商均衡）、參數常數。對量化讀者最實際的意義是：**把信息採購與執行成本放在同一張現金帳上核算**；但本頁要提醒，其價格證據章節只刻畫 target 不確定性，並未識別 activation 或機構成本（摘要逐字：without identifying activation or institutional costs）。

## §1 · 架構 / Core Mechanism

### 三大改動 vs 前作

| 維度 | 前作（原文 Table 1 所選最近文獻） | 本文改動 | 為什麼是斷點 |
|---|---|---|---|
| 信息價值分類 | Crego (2020)／Diamond–Verrecchia (1991)：公開信息披露改變參與與逆選擇，方向為定性 | 把信息價值歸約為 ψ(μ) 的凹性；凹 ⇔ 所有 Blackwell refinement 在該區間弱降成本 | 從「信息好/壞」敘事變成可逐點檢定的曲率條件 |
| 參與的內生性 | Do–Kettering (2025)：信息與內生 entry 在數量競爭中互動，但無終端拍賣現金帳 | 可選激活 + 統一價格拍賣 + 顯式 terminal cash payment，導出 e、k 兩係數 | 誰的成本改變被現金帳釘死，而非效用函數假設 |
| target 的內外生 | Zhao (2026)：LETF reset 的 target 依賴受影響價格（反饋通道） | target 對受影響執行價格外生，且機構非 Stackelberg 領導者 | 把「預測 → 參與 → 成本」隔離成單一因果鏈 |
| 組合維度 | Min–Maglaras–Moallemi (2022)：cross-impact 下組合執行，信息非重點 | 共享激活 + 矩陣判據 P_a ⪯ 0；等精度信號方向可相反 | 預測誤差下降量不足以排序組合信息 |

### ⚡ Eureka

**一句話**：當庫存供給可選時，機構的成本是後驗均值 μ 的一個函數 ψ，而 ψ 的凹性——不是預測精度——決定公共信息的好壞。

**直覺**：ψ 有兩股力互相拉扯。第一股是直接預測收益（-(d²/2A)μ²，永遠凹、永遠有益）；第二股是「信息把庫存招進來」的 incidence（eμm - km²）。固定名單時第二股被寫成 μ² 的負係數，兩股同向；可選激活時 m(μ) 是內生均衡量，在 entry 邊界（m 從 0 起跳）與滿參與邊界（m 斜率突變）會產生 ψ′ 的跳躍 J_τ，凹性可被擊穿。

### 信息流

t=0 ── 公共信號 Y ──► μ = E[D|Y], v = E[(D-μ)²|Y]
　　　　 │
　　　　 ├─► 機構：同時選 z ∈ ℝ（Nash，非先動承諾）
　　　　 └─► 交易者：觀察私有 f，決定是否付 f 激活 → 選 q_f
　　　　　　　　m = ∫ q_f · 1{f enters} dF(f)
預事件 [0,1]：排程 φ_t；İ_t = -ρI_t + κ(z+m)φ_t
　　　　 → K(z+m) 為預付均價衝擊、H(z+m) 為帶入拍賣的殘留位移
t=1 ── D 揭示 ──► 統一價格拍賣 p_A = H(z+m) + λ(D - z - m)
　　　　 └─► 機構付 Kz(z+m) + η₀z²/2 + (D-z)p_A；entrant 個體利潤 q·p_A − Kq(z+m) − ηq²/2，支付 f 進場

## §2 · 數學層

📌 **Napkin Formula**

均衡（Theorem 1，Eq. 11–12）：

a = F( η s² μ² / (2(η + c a)²) ),　q = sμ / (η + c a),　m = a q,　z = (dμ - b m)/A

成本泛函（Eq. 16）：ψ(μ) = -(d²/2A)μ² + e·μ·m(μ) - k·m(μ)²

係數（Eq. 8, 10, 14）：b = λ + K - H，A = η₀ + 2b，d = 2λ - H，s = λ - bd/A，c = b(1 - b/A)，e = db/A - (λ - H)，k = b²/(2A)

**複雜度**：單一標量根。h(a) = a - F(ηs²μ²/[2(η+ca)²]) 在 a₂ > a₁ 時滿足 h(a₂) - h(a₁) ≥ a₂ - a₁ > 0，故根唯一且單調，可用 bracketed bisection（原文逐字：globally reliable bisection）。無迭代訓練、無隨機優化。

**直覺**：s 是「信號載荷」——把機構最優反應代進 entrant 一階條件後剩下的有效斜率；c 是「擁擠係數」——同伴進場侵蝕個體利潤的速率。兩者一起決定 entry 門檻 μ₀ = √(2ηf₀)/s。

**Loss／訓練細節**：本文無 ML 訓練。η₀、η 是臨時執行費係數（temporary charge），不是學習率；F 是 activation cost 的 CDF，不是 fitness。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下為示意算術**，參數取自原文 §5.3 的 explicit example（λ=η₀=η=1、K=H=0、F 為 [0.02, 0.04] 上的 uniform）。目的是讓讀者照著算一遍「為什麼 e<0 也會有有害信息」。**這不是本頁新增的實證結果。**

**Step 1｜由原語算係數。** b = 1，A = 3，d = 2，s = 1/3，c = 2/3，e = -1/3，k = 1/6（原文逐字對應 A=3, b=1, d=2, s=1/3, c=2/3, e=-1/3, k=1/6）。

**Step 2｜確認這是「庫存有益」的環境。** e = -1/3 < 0，由 Proposition 2，任何 μ 下進場庫存都弱降低機構成本（H=0 時 e = -λη₀/A < 0 是 benchmark）。

**Step 3｜定位滿參與邊界。** μ₁ = ((η + c)/s)·√(2f₁/η) = √2。越過此點全體人口都已進場，entry margin 耗盡，但每人數量不受限，聚合庫存繼續以 m = t₁μ 成長，t₁ = s/(η + c) = 1/5。

**Step 4｜算邊界兩側的庫存斜率。** 由 endpoint expansion，左斜率 L₁ = sB₁/(η + cB₁) = 5/13；右斜率就是 t₁ = 1/5。**注意 5/13 > 1/5：左側更陡。** 這是全部機制的來源。

**Step 5｜算 ψ′ 的跳躍。** J₁ = μ₁(e - 2kt₁)(t₁ - L₁) = 24√2/325 > 0。雖然 e < 0（庫存有益），但 (e - 2kt₁) < 0 且 (t₁ - L₁) < 0，負負得正。

**Step 6｜翻成信息比較。** 在 μ₁ 放一個兩點對稱先驗，用噪聲信號做 refinement，成本變化 ΔC = (12√2/325)ε + O(ε²) > 0。同一實驗若把參與固定在 a=1，成本變化是 -0.74ε² < 0。**同樣的信號、同樣的先驗，只因參與可選，符號翻轉。**

## §3 · 數據層

分列時間邊界（不可互相套用）：

| 資料塊 | 頻率／範圍 | 時間邊界（逐字） | 用途 |
|---|---|---|---|
| 五分鐘 bar panel | 26 檔 geared ETF + 5 檔 benchmark proxy（SPY/QQQ/IWM/DIA/SOXX） | 2021-11-09 至 2026-11-13 | 日內特徵、均值／不確定性預測 |
| 共同預測樣本 | 四家族共同日期 | 2023–2025 共 717 normal sessions；2026 共 219 | Table 5 |
| 共同參考樣本起點 | — | 2023-02-10（activity normalization 與 ridge 各需完整前期窗） | 特徵暖機 |
| 日線 turnover panel | 31 檔 ETF | 2024-11-18 至 2026-11-13，499 dates | Appendix O |
| 因子資料 vintage | Fama–French 五因子＋動能＋電子設備產業 | 2024-11-19 至 2026-09-30，466 return dates；September 2026 CRSP database | Appendix P |

**怎麼來**：五分鐘資料以 completed bars、New York time 為準；主約定把 label 當 interval start（11:55 與 15:55）。樣本起點是 31 檔中**最晚的第一筆觀測**，早於此的觀測一律排除於 normalization 與 estimation（Table IA.12 逐字記錄此邊界，而非各產品 inception）。

**樣本外與容量假設**：2026 是 **retrospective temporal evaluation**，不是 untouched holdout——原文 §10.2 逐字承認早期探索性分析影響了設計、部分 2026 價格在看過。預測用 fixed 252-observation rolling ridge（penalty 0.1，training-only standardization）。**容量假設未建立**：原文未把 turnover 換算成 committed demand D，Appendix O.1 逐字說明這些是 secondary-market trading measures，不是 creations/redemptions/衍生品名目。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露。所讀文本僅稱 "replication code"、"replication archive"、"replication outputs"，未見 URL |
| Checkpoint | N/A（無 ML 權重）。等價物是 Proposition 7 的 directed-rounding 曲率證書：50 位精度、交替級數 13 項餘項界，區間 [4.40037496282478, 4.40039330270280] |
| License | CC BY 4.0（arXiv 頁面標示） |
| 複現難度 | 中—高。標量根與閉式成本易做；命題 7 的正曲率存在性需高精度區間算術與 lognormal 分位數夾擠（z_* = Φ⁻¹(1/4) 落在 [-0.674490, -0.674489]） |
| 數據可得性 | 因子（Kenneth French Data Library）與日線（Massive）為外部可得；五分鐘 panel 的**原始抽取設定與絕對成交量單位未載明**（§J.2 逐字），只能做相對活動度與窗內權重 |
| 參數可得性 | impact 與 activation 參數為 illustrative，非估計值（§2.1） |

## §5 · 評測 / Benchmark

> **表格語義聲明**：原文**沒有 SOTA 概念**，也沒有通用 benchmark 協定。下表「前SOTA」欄實為原文報告的**對照 band 數值**（Raw 或 RV scaled），「本方法」欄為本文比較的 Range scaled band 數值。各 band 均為 95% interval 的對稱零中心區間，分數越低越好。差值未在表內自行計算；原文 Table IA.16 的配對差值列於表下方解讀。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| SPY/QQQ/IWM/DIA 共同日期 2023–2025（717 日） | 95% interval score, bp ↓（Raw vs Range scaled） | 384.96 | 342.38 | 見解讀 |
| 同上 2023–2025 | 95% interval score, bp ↓（RV scaled vs Range scaled） | 352.51 | 342.38 | 見解讀 |
| SPY/QQQ/IWM/DIA 共同日期 2026（219 日） | 95% interval score, bp ↓（Raw vs Range scaled） | 283.63 | 256.93 | 見解讀 |
| 同上 2026 | 95% interval score, bp ↓（RV scaled vs Range scaled） | 257.59 | 256.93 | 見解讀 |

**配對 interval score 差值（原文 Table IA.16，單位 bp，負值有利於前者）**：

- 2023–2025：Range − Raw = **-42.59**，個體 95% 日期分塊區間 [-101.07, -3.21]；RV − Raw = **-32.45**，區間 [-88.34, 5.54]；Range − RV = **-10.13**，區間 [-15.22, -5.55]。
- 2026：Range − Raw = **-26.70**，區間 [-52.61, -3.28]；RV − Raw = **-26.04**，區間 [-53.95, -0.50]；Range − RV = **-0.66**，區間 [-7.08, 6.71]。

**均值預測（無優於基線者，故不列 Δ）**：Table 5 Panel A 中所有均值預測的 pooled RMSE ratio 都 > 1，即劣於零剩餘收益：Historical mean 1.0022／1.0011、Returns 1.0062／1.0116、Returns + range 1.0159／1.0123、Returns + range + count/size 1.0168／1.0122、Returns + range + volume 1.0141／1.0121（左為 2023–2025、右為 2026）。零預測的 RMSE 水平是 66.94 與 49.47 bp。加入 count/size 的配對增量是 8.22 squared bp，日期分塊區間 [-39.15, 43.39]；加 volume 的增量區間亦含零。原文逐字限定：「This finding applies to the declared models and does not establish that public information is uninformative.」

**哪部分是真 capability，哪部分是脆弱面**：

- **穩健的一塊是 Range − RV，不是 Range − Raw。** 2023–2025 的 Range − RV = -10.13，個體 95% 日期分塊區間 [-15.22, -5.55]；換到 HAC(10) 推論區間 [-14.94, -5.33]、Holm p < 0.001，族系調整後仍顯著。反之 Range − Raw 的 -42.59 在 bootstrap 下區間 [-101.07, -3.21]，但 HAC 下變 [-93.55, 8.38]、Holm p 0.304——**同一組預測，換推論方法就翻**。本頁判讀：這是分數改善高度集中所致，不是方法本身不穩。
- **集中度是真風險。** 2025 一年就貢獻參考期淨改善的 96.51%；五大絕對貢獻日期佔 65.80%（2026 為 61.51%）。2025-04-09 單日四個剩餘收益落在 803 到 1,123 bp，raw band 平均全寬 254.61、range band 849.60 bp，**四條 range band 全部 miss**，其較小損失來自 exceedance penalty 下降而非涵蓋。
- **前瞻／事後選擇**：2026 樣本非 untouched holdout（§10.2 自述）；Table IA.26 的替代窗長在共同日期上重算，2024–2025 的 252 窗 Range − Raw 的 HAC 區間 [-135.70, 13.29] 含零——點估計一致為負（-61.20），但推論不穩。
- **成本未計**：這些是**預測損失（bp）**，不是交易成本、不是 impact、不是機構執行帳單。§8 反覆強調數值為 constructed scenario 的 normalized cash units，不是 bps of assets。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位章節）

- **§3.2（Eq. 6 後）**：終端供給是彈性排程 p_A = H(z+m) + λn，**不是**策略交易商均衡；無拍賣容量約束、無拍賣後操縱機會。
- **§2.1／摘要**：impact 與 activation 參數為 illustrative；歷史基金資產負債表與機構訂單未觀察；價格證據 "without identifying activation or institutional costs"。
- **§5.1（Theorem 3 證明後）**：converse 允許 target law 在候選反例之間變動，**不允許**固定 prior 下任意後驗均值分布（受 Gentzkow–Kamenica 2016 刻畫約束）。
- **§7.1**：Theorem 11 只在**對稱**貝葉斯均衡類別內主張唯一，不主張非對稱均衡唯一；單調性論證用 N ≥ 2。
- **§6.1**：連續性不給出「當信息位移趨零時仍一致」的擾動鄰域。
- **§10.2**：2026 為回顧性時序評估，非未觸碰 holdout；早期探索性分析影響設計。
- **§11**：結論限於 fixed-schedule、risk-neutral quantity game、linear terminal supply、target 外生於受影響價格；關心機構現金支出，**不是社會福利**。
- **§J.2**：五分鐘 panel 的原始抽取設定與絕對成交量單位未載明。
- **§K**：close-related 路徑僅為描述性證據，不識別 auction-price reversal。

### 6.2 推斷的隱含假設（本頁推斷，非原文斷言）

- **排程外生**（推斷）：φ 固定意味著不能「等更好的信號再執行」。§11 自己把它列為 extension（allow traders to wait），因此所有信息價值結論都條件在「決策時點與執行機會固定」。跨到可延後執行的系統，ψ 需重寫。
- **風險中性 + 線性供給**（推斷）：後驗方差 v 只以 λv 進入成本、不改部位。任何有庫存風險溢價或非線性 impact 的場景，m(μ) 的形狀會變，entry／saturation 兩處跳躍的符號不能直接搬。
- **activation cost 與 target／信號族獨立**（Assumption 1）：這是 Theorem 3 可寫成 E[ψ(E[D|H])] 的關鍵。若 f 與 target 相關（例如行情好時營運成本更高），Eq. 17 的分解失效。
- **固定參與名單必有益**（原文結論，非假設）：ψ_a 的括號嚴格為負，因此這是**在本文模型內**的定理；不可外推成「所有系統中先鎖定流動性必然劃算」。
- **樣本邊界**（推斷，待核）：ETF 名單以現存發行文件為準（ProShares、Direxion 等 2026 年產品頁），已退市／改名產品的納入規則所讀文本未見；不能由此推論生存者偏差的方向或大小，僅能說這是輔助數據邊界。
- **容量未識別**（推斷）：turnover 是次級市場交易量，不等於 committed demand、creations 或衍生品名目（Appendix O.1 逐字支持）。任何把 Table IA.30 的金額當 D 的用法都會越界。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Zhao (2026), Preying on leveraged ETFs | target 是否外生於受影響價格；capacity 在 open 前承諾 vs fringe 在拍賣中進入 | 否 | working paper, October 15 version |
| Do & Kettering (2025) | 有無 terminal auction 的顯式現金支付；信息＋entry 在純數量競爭 vs 執行帳 | 否 | YERI WP 2025RWP-253 |
| Pegoraro–Sammon–Shim (2026) | 機構數量承諾是否固定；index reconstitution vs LETF reset | 否 | working paper |
| Min–Maglaras–Moallemi (2022) | 信息維度＋共享激活；cross-impact 下的組合信息排序 | 否 | Operations Research |
| Barbon et al. (2022) | 理論信息價值 vs 實證 end-of-day 流動性供給 | 否 | SFI TR 22-40 |
| Crego (2020)／Diamond–Verrecchia (1991) | 有害信息的來源：逆選擇／做市商退出 vs 可選庫存供給 | 否 | JFE／JF |
| Campbell & Nutz (2026) | 瞬態衝擊下 N 個交易者的最優執行 vs 可選激活＋終端拍賣 | 否 | Mathematical Finance (Early View) |

🎤 **Interview Tip**

- **錯答**：「這篇證明信息越精確越好／越差。」——把 Theorem 3 的雙向性壓成單向。
- **正確答**：分三層講。(i) 固定參與名單下，ψ_a 是 μ² 的嚴格負係數，任何 Blackwell refinement 都降成本；(ii) 可選激活下，entry 邊界（e > 0 時）與滿參與邊界（e < 2kt₁，含所有 e ≤ 0）各有一次 ψ′ 上跳，J_τ > 0 就能構造出使成本上升的兩點對稱 refinement；(iii) 平滑內部還有一條與邊界無關的路徑，條件是 G·m″ > Q(m′)，Proposition 7 給出 lognormal 全支撐、a = 1/4 的存在性例（μ_* ≈ 0.82506008，4.4003 < ψ″(μ_*) < 4.4005），且該效應可小到執行帳單的 0.000627%——**存在不等於量級大**。
- **加分答**：補一句「有限 N 會翻符號」——每人自身衝擊把 entry 門檻抬到 μ₀,N > μ₀，Corollary 13 給出 Nε 才是近 entry 的正確尺度，且兩個極限不可交換（Eq. 45）。

### 7.1 可證偽預測（帶時點，本頁提出）

1. **到 2027-12-31 前**（本頁提出的檢驗時點，非原文），若把 range-scaled band 套用到 2027 年新增樣本（同 SPY/QQQ/IWM/DIA 四家族、同 11:55→15:55 口徑），其相對 raw band 的 interval score 改善幅度**應不如** 2023–2025 的 -42.59 那麼大（即數值更接近零），且在 HAC(10) 下未經 Holm 調整的原始 p 應不為小值；2026 的 HAC 95% 區間為 [-50.44, -2.96]、原始 p=0.027，但 Holm 調整後 p=0.137 不顯著。若觀察到顯著且幅度更大，則集中度解釋（96.51% 來自 2025）被削弱。門檻數值 TBD。
2. **方向性信息的存在性檢驗**：Corollary 15 要求 ψ″(r) > 0 的平滑狀態。可檢驗版本是取一組共享 activation 的籃子（原文 Appendix N 曾用 QQQ/SOXX 做聯合預測，但未做此激活實驗），構造 n = (1,1)/√2 與 v = (1,-1)/√2 的等精度信號，檢驗 radial 與 tangential 的成本代理是否異號。**此檢驗尚未設計，成本代理的定義與顯著性門檻 TBD**；若在 ψ″ > 0 未被獨立證實的市場上得到同號，則是對模型的否證而非支持。
3. **均值預測的否證點**：若任何一個新特徵集在共同四家族日期上把 pooled RMSE ratio 壓到 1 以下並通過 HAC 與 Holm，則「宣告模型範圍內無可用的日內剩餘收益預測」失效。目前最近的候選是 Historical mean 在 2026 的 1.0011，仍 > 1。

## §8 · For the Reader

- **因子研究員**：把「預測改進」與「執行成本變化」拆成兩本帳。本文的接口是 ψ 的曲率，不是 IC 或 RMSE；Eq. 25 明確說真嵌套條件均值必然降 MSE，所以 Theorem 3 的反例**不是**「回歸擬合差」，而是「參與重組」。若你的 alpha 管線下游有可選流動性供給者，RMSE 改善不能當執行價值。
- **高頻執行**：entry 閾值附近的參與是非對稱的——上修吸引不到已耗盡的人口，下修卻會抽走有用的庫存。這是 saturation 端有害信息的機制（與前置衝擊無關，H = 0 也會發生，見 Eq. 30）。實務含義：公開預測的發布時點相對 entry 門檻的位置，比預測精度更關鍵。
- **組合配置**：等精度不等價。isotropic 執行下，magnitude 信號（沿 n）的成本變化記為 +0.000227846、composition 信號（沿 v 正交）記為 -0.0000209461，而兩者對 MSE 的削減在 Table 4 兩欄均記為 .000100。組合信息的排序要看 P_a ⪯ 0 這個矩陣判據；同時正交對角化只是充分條件，Appendix H.4 給了現金支付層面驗證的反例。
- **RL／agent 策略**：把 activation cost 建成 agent 的內生參與決策而非外生流動性參數，這正是本文相對 mean-field liquidation 文獻的位置。有限 N 的 ξ₀ ≈ 0.953145、γ₀ ≈ 0.0333783 提供了一個「信息解析度 vs 策略規模」的耦合尺度，可作為 multi-agent 模擬的 sanity check。
- **研究學生**：這是「精確均衡 + 反例 + 存在性證書」的範式，不是大樣本實證。讀法建議：先看 Theorem 1（唯一性靠 h(a) 的單調性，五行證明），再看 Proposition 2（把 m 的 effect 拆成 eμm - km²，這是全部經濟內容的來源），再把 Theorem 3 的 Jensen 論證與 Theorem 4 的跳躍公式對照——你會發現全篇只有一個技巧反覆使用：**先把問題壓成單變量，再對它做二階分析**。

## References

- 原論文：*Trading Before a Committed Rebalance*，arXiv:2611.25762v1 [q-fin.TR]，19 Nov 2026，CC BY 4.0。https://arxiv.org/abs/2611.25762v1
- 信息幾何 lineage：Blackwell (1953), *Equivalent comparisons of experiments*, Ann. Math. Stat.；Gentzkow & Kamenica (2016), *A Rothschild–Stiglitz approach to Bayesian persuasion*, AER；Kamenica & Gentzkow (2011), *Bayesian persuasion*, AER。
- 執行／跨影響 lineage：Min, Maglaras & Moallemi (2022), *Cross-sectional variation of intraday liquidity, cross-impact, and their effect on portfolio execution*, Operations Research；Campbell & Nutz (2026), *Optimal execution among N traders with transient price impact*, Mathematical Finance；Fu, Hager & Horst (2024, 2026)。
- 最接近的經濟對象：Zhao (2026), *Preying on leveraged ETFs*；Do & Kettering (2025), *The value of information in oligopoly with endogenous entry*；Pegoraro, Sammon & Shim (2026), *Optimal index-linked rebalancing with anticipatory trading*。
- 實證對照：Barbon, Beckmeyer, Buraschi & Moerke (2022), SFI TR 22-40；Brøgger (2021), *The market impact of predictable flows*, JBF；Ivanov & Lenkey (2018), *Do leveraged ETFs really amplify late-day returns and volatility?*, JFM。
- 方法／計量：Newey & West (1987)；Gneiting & Raftery (2007)；Giacomini & White (2006)；Martens & van Dijk (2007)。
- 資料源：Kenneth French Data Library（September 2026 CRSP vintage）；Massive daily unadjusted aggregates and corporate actions。
- 圖片／附件（Figure 1–3、Figure IA.1–IA.17、Table IA.1–IA.35 之完整內容）：本次輸入未提取，**圖表／附件待核**。