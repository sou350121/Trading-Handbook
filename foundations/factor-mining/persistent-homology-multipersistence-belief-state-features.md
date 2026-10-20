<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# Persistent Homology / Multipersistence Belief-State Features 解構（Persistent Homology / Multipersistence Belief-State Features）

> **來源發布**：2026-10-19 · （無 venue） · arXiv [2610.25756](https://arxiv.org/abs/2610.25756)
> **arXiv 原文**：[Topological Compression of Prediction Market Belief States for Event Forecasting](https://arxiv.org/abs/2610.25756v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點在「量價表格／中长周期／監督回歸／因子挖掘／人機協同可解釋」。它補的 prior gap 是：預測市場價格已被當作機率預測使用，但「一籃子相關市場的信念狀態整體形狀」是否含增量資訊，過去只有 PCA 線性壓縮與一般 TDA 文獻各自處理，本文把兩者接在同一條消融鏈上（market price → +PCA → +鄰域控制 → +拓撲）。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把 Polymarket 的 171 個相關合約聚成 6 個 family，用 24 個 active-set family-state 特徵經兩成分 PCA 壓成全域信念狀態；②核心 trick 是在 k=20 的相關鄰域上用 d=1−|ρ| 建 Vietoris–Rips 過濾，抽 H0/H1 的有限區間數、總持續性、最大持續性與持續熵，另用 α=1−|ρ|、δ=‖PCA 殘差差‖ 的雙過濾做一維切片當 multipersistence 特徵；③對「因子挖掘」軸★的意義是：拓撲量是可手算、可稽核的描述子，理論上有 stability 保證，比黑箱嵌入更適合當風控可解釋的中間層；④關鍵逐字數字：Logit + PCA + controls 的 log loss 0.1355 在加入 VR 拓撲後為 0.1166，Random Forest + PCA + controls 的 Brier 0.0359 在加入 MP 特徵後為 0.0126。

**X-Ray.** 放回五軸 Pareto：這篇不是 alpha 新來源，而是「同一組價格祖先」的表示學習分支——在本文樣本與比較設定下，**新增的背景狀態描述子有增量預測表現**；這不足以證明拓撲本身不可替代。它解掉的舊工程坑有兩個：（a）active-set 面板，把「市場還沒開／已結算」與「缺值」分開，避免 PCA/PH 學到市場可得性；（b）把拓撲放在「PCA+鄰域控制」之後才測，這是本文最乾淨的一步，因為控制變數已含鄰域均值、離散度、相關強度與相關變異。但它打不開的 envelope 很清楚：標籤 Y_i 是**市場級常數**，532,979 列只對應 171 個市場級結果，市場之間也可能相關，任何 row-level 增益都可能被 171 個簇吸收；而作者自己的市場簇 bootstrap 區間（如 Random Forest MP 的 log loss 增益 [0.0603, 0.1030]）仍為正，這是它最有力的防守，但也承認了不獨立性。對量化讀者的意義：把它當**特徵工程與校準的案例**讀，不要當成可交易的 alpha 讀——本頁所讀文本未建立可實盤的成本、成交與容量驗證。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 面向 | 傳統做法（本文引用之 baseline 脈絡） | 本文改動 | 為何是改動而非增量 |
|---|---|---|---|
| 信念狀態的定義 | 單一市場的 YES 價格即機率 | 用 active-set 建 family-level 向量 F_t = (c_f, μ_f, m_f, s_f)，共 24 個特徵／6 個 family | 把「哪些市場此刻存在」變成狀態的一部分，而非丟棄或補缺值 |
| 壓縮方式 | 只用 PCA 做線性降維 | PCA（2 成分）當線性基準，另加**局部** VR 過濾的 H0/H1 摘要 | 全域線性 + 局部非線性分層，是可分離的消融而非混合黑箱 |
| 多尺度處理 | 單一 ε 的 VR 過濾 | 以 (α=1−\|ρ\|, δ=‖r_i−r_j‖) 建雙過濾，再做**一維降維**取標準 PH 特徵 | 明確借用 multipersistence 的原則，但**不計算**完整多參數不變量 |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

> **把「PCA 殘差」當成第二條過濾軸：先在原始相關距離上問「誰跟誰像」，再在 PCA 解釋不掉的殘差距離上問「誰跟誰像得不一樣」——只有兩軸都夠近的市場群才會在該參數值進入單純複形。**

直覺：PCA 抓到共同主方向後，剩下的殘差就是「主題已定價、個別合約卻分歧」的部分。若這些分歧在局部鄰域裡形成環（H1），代表市場之間存在結構性張力；作者猜這正是增量資訊所在（§5.1 明示為 conjecture，未驗證）。注意這是**假設**，不是本文已證成的因果機制。

### 1.3 信息流 ASCII 圖

```
Polymarket 原始 contracts
   │  Gamma API / Data API / CLOB API
   ▼
171 個已結算 binary markets（152 NO / 19 YES），按 family 分 6 組
   │  逐小時聚合 → active set A_t（中位數 34，範圍 17–68）
   ▼
family-state F_t ∈ R^24 ──標準化(訓練期 μ, s)──► PCA (K = 2)
   │                                              │
   │                                              ├─► z_t^PCA（全域線性狀態）
   │                                              └─► 殘差 r_i
   │  訓練期相關矩陣 ρ
   ├──► N_k(i), k = 20 ；d_ij = 1 − |ρ_ij|
   │        └─► VR_ε(N_k(i), d) ─► H0/H1 摘要 T_i（有限區間數、總持續性、最大持續性、持續熵）
   │
   └──► 雙過濾 α_ij = 1 − |ρ_ij| , δ_ij = ‖r_i − r_j‖
            └─► 一維降維切片 ─► H0/H1 摘要 T_i^MP

特徵拼裝：(p_{i,t}, A_t, z_t^PCA, C_i, T_i 或 T_i^MP) ──► Logit / XGBoost / Random Forest
                                                              ▼
                                                       p̂_{i,t} vs Y_i（Brier / LogLoss / AUC / ECE）
```

## §2 · 數學層

📌 **Napkin Formula**

1. 線性壓縮：`z_t^PCA = V_K^T · F̄_t`，其中 `F̄_t` 為以訓練期均值／標準差標準化後的 24 維 family-state，`K = 2`（原文說明兩成分在樣本外表現最好，10 成分以上 log loss 明顯變差）。
2. 局部過濾：鄰域 `N_k(i)`（`k = 20`），距離 `d_ij = 1 − |ρ_ij|`，取 `VR_ε(N_k(i), d)` 的 `H0`／`H1`。
3. 雙過濾：`α_ij = 1 − |ρ_ij|`、`δ_ij = ‖r_i − r_j‖`；單純形 σ 在參數 `(a,b)` 進入複形 **若且唯若** 對所有 `i,j ∈ σ` 有 `α_ij ≤ a` 且 `δ_ij ≤ b`。

**複雜度**：單點雲為 `k = 20` 的 VR 過濾，複雜度對 `k` 是多項式級、在此規模可行；雙過濾僅取一維切片，作者明言是為了避開完整多參數持久同調的計算與摘要成本。具體執行時間、記憶體與特徵抽取常數**所讀文本未見，圖表／附件待核**。

**直覺**：`d_ij = 1 − |ρ_ij|` 取絕對值，是因為作者認為正相關與負相關都代表「共享潛在因子」；把負相關也壓成小距離，等於把「同向」與「反向」視為同一種結構連結。這是一個**建模選擇而非定理**，也是後面失效模式的來源之一。

**Loss／訓練細節**：Logit 走機率輸出加 log loss；XGBoost 目標為 `L = Σ ℓ(y_i, ŷ_i) + Σ Ω(f_m)`，樹複雜度 penalty `Ω(f) = γT + ½λ Σ w_j²`；Random Forest 為 `B` 棵樹平均。正則化超參（`γ`、`λ`、`η`、樹深、樹數、`B`）的具體取值**所讀文本未見，未披露**。類別不平衡以 proper scoring rules、校準檢查與時序評估處理，非重加權。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部是「假設／示意」的玩具數字，用來手算機制，不是論文實證結果。** 論文的真實數字一律留在 §5。

**步驟 1｜假設一個 4 市場的局部鄰域**（真文獻用 `k = 20`，這裡縮小到 4 個點以便手算）。設訓練期相關係數為：

| 對 | ρ | d = 1 − \|ρ\| |
|---|---|---|
| (A,B) | 0.9 | 0.1 |
| (B,C) | 0.9 | 0.1 |
| (C,D) | 0.9 | 0.1 |
| (D,A) | 0.9 | 0.1 |
| (A,C) | 0.1 | 0.9 |
| (B,D) | 0.1 | 0.9 |

**步驟 2｜依 ε 逐級加邊**：ε = 0.1 時，AB、BC、CD、DA 四條邊同時進入 → 四個頂點併成一個連通分量，並形成一個四邊環。ε = 0.9 時，對角邊 AC、BD 進入，三角形 ABC 與 ACD 被填滿，環被殺死。

**步驟 3｜讀 H0（連通分量）barcode**：4 個分量在 ε = 0 出生；ε = 0.1 時 3 個分量死亡併入。故 H0 有 3 條有限區間，皆為 [0, 0.1)。→ 有限區間數 = 3；總持續性 = 0.1 + 0.1 + 0.1 = 0.3；最大持續性 = 0.1。

**步驟 4｜讀 H1（環）barcode**：環在 ε = 0.1 出生、ε = 0.9 死亡 → 1 條區間 [0.1, 0.9)。→ 有限區間數 = 1；總持續性 = 0.8；最大持續性 = 0.8。**這就是「結構張力」的量化**：四個市場兩兩高度相關但對角弱，於是形成一個寬 0.8 的環。

**步驟 5｜加第二條過濾軸**：假設這四點在 PCA 殘差空間的距離為 δ_AB = δ_BC = δ_CD = δ_DA = 0.2，而 δ_AC = δ_BD = 0.7。取一維切片（例如沿 α = δ 的對角路徑），則在 `(a,b) = (0.1, 0.2)` 時四點已同時滿足 α ≤ a 與 δ ≤ b，環照樣成形；若 δ_AC 也被壓到 0.2，對角邊更早進入，環會提早死亡、最大持續性下降。**MP 特徵與 VR 特徵的差別，就來自這條「第二軸把哪些邊提前拉進來」**。

**步驟 6｜拼進模型**：把 `(有限區間數, 總持續性, 最大持續性, 持續熵)` 共 8 個量（H0/H1 各 4）接到 `(p_{i,t}, A_t, z_t^PCA, C_i)` 之後，送進 logit。若 θ 這組係數在樣本外把 Brier 壓低，才叫「拓撲有增量」；單看這裡的 0.8 沒有意義，**拓撲特徵的值本身不是 alpha，係數的樣本外穩定性才是**。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 平台 | Polymarket |
| 市場池 | Macro、crypto、crypto-adjacent 的已結算 binary markets |
| 市場數 | 171（152 NO / 19 YES） |
| Family 分佈 | Macro Fed/rates 96；BTC price 56；ETH price 14；crypto policy 2；MicroStrategy 2；crypto regulation 1 |
| 頻率／表示 | 逐小時 active-set 面板 |
| 樣本量 | 532,979 個 market-time 觀測，涵蓋 14,464 個逐小時時間戳 |
| 活躍市場數 | 中位數 34，範圍 17 到 68 |
| 特徵維度 | 24 個 family-state 特徵，跨 6 個 family |
| 日期區間 | 08/06/2024–04/01/2026 |
| 資料源 | Gamma API；Data API；CLOB API |
| 切分 | 按時間序列的 chronological folds，共 17 折（由 §4.5 的「16 / 17」「17 / 17」計數反推折數） |
| 目標 | Y_i = 1 若市場 i 最終 resolve YES，否則 0（**市場級常數，不隨 t 變**） |

**建構要點**：`A_t = {i ∈ M : 市場 i 在 t 已開且尚未結算}`；family 特徵為活躍數 `c_f,t`、活躍機率均值 `μ_f,t`、中位數 `m_f,t`、標準差 `s_f,t`。缺失的 family 特徵以**當前折訓練時間戳**學到的 imputer 填補，再套用到之後的測試時間戳，避免未來資訊回灌。

**樣本外與容量假設**：本文只做機率預測評估，**未披露**手續費、買賣價差、滑價、可成交名義量或資金容量；「sufficient price observations」「quality checks」「sufficient overlap with analysis period」的具體門檻**所讀文本未見，未披露**。CLOB-sampled prices 的取樣規則與時戳對齊方式**所讀文本未見，圖表／附件待核**。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | `https://github.com/ajerfy/Topology-of-Prediction-Market-Belief-Dynamics`（§6 結論段逐字給出） |
| Checkpoint | 未披露（所讀文本未見預訓練權重；本文方法為特徵工程＋樹模型／logit，理論上不需 checkpoint） |
| License | 所讀文本未見，未披露 |
| 複現難度 | TBD（資料需自 Polymarket API 重抓；文中未給 pin 住的 snapshot、random seed、XGB/RF 超參，故無法從文本端複現） |
| 數據可得性 | 需自 Gamma / Data / CLOB API 取得；作者未聲明是否釋出凍結資料集，所讀文本未見 |

## §5 · 評測 / Benchmark

> 前 SOTA 欄一律填「該列對照模型本身的逐字數值」，非合成 SOTA。Δ 為同行兩數之算術差，負值代表本方法較低（Brier／LogLoss／ECE 越低越好；AUC 越高越好）。百分點差未另行換算，僅保留原始小數精度。

**自然市場（171 市場、532,979 列、17 折）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 自然市場 | Brier（market-implied → Logit+PCA） | 0.0489 | 0.0432 | −0.0057 |
| 自然市場 | LogLoss（market-implied → Logit+PCA） | 0.1673 | 0.1565 | −0.0108 |
| 自然市場 | Brier（Logit+PCA → +controls） | 0.0432 | 0.0362 | −0.0070 |
| 自然市場 | LogLoss（Logit+PCA → +controls） | 0.1565 | 0.1355 | −0.0210 |
| 自然市場 | AUC（market-implied → Logit+PCA+controls） | 0.9419 | 0.9492 | +0.0073 |
| 自然市場 | Brier（Logit+PCA+controls → +VR） | 0.0362 | 0.0311 | −0.0051 |
| 自然市場 | LogLoss（Logit+PCA+controls → +VR） | 0.1355 | 0.1166 | −0.0189 |
| 自然市場 | AUC（Logit+PCA+controls → +VR） | 0.9492 | 0.9629 | +0.0137 |
| 自然市場 | ECE（Logit+PCA+controls → +VR） | 0.0161 | 0.0055 | −0.0106 |
| 自然市場 | Brier（XGBoost+controls → +MP） | 0.0402 | 0.0231 | −0.0171 |
| 自然市場 | LogLoss（XGBoost+controls → +MP） | 0.1426 | 0.0932 | −0.0494 |
| 自然市場 | Brier（RandomForest+controls → +MP） | 0.0359 | 0.0126 | −0.0233 |
| 自然市場 | LogLoss（RandomForest+controls → +MP） | 0.1544 | 0.0764 | −0.0780 |

**平衡市場（YES/NO 對半的穩健性宇宙，§5.2）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 平衡市場 | Brier（market-implied → Logit+PCA+controls） | 0.1660 | 0.1318 | −0.0342 |
| 平衡市場 | LogLoss（market-implied → Logit+PCA+controls） | 0.5204 | 0.4008 | −0.1196 |
| 平衡市場 | Brier（Logit+PCA+controls → +VR） | 0.1318 | 0.1185 | −0.0133 |
| 平衡市場 | LogLoss（Logit+PCA+controls → +VR） | 0.4008 | 0.3604 | −0.0404 |
| 平衡市場 | Brier（XGBoost+controls → +MP） | 0.1189 | 0.0399 | −0.0790 |
| 平衡市場 | LogLoss（XGBoost+controls → +MP） | 0.3789 | 0.1727 | −0.2062 |
| 平衡市場 | Brier（RandomForest+controls → +MP） | 0.0802 | 0.0143 | −0.0659 |
| 平衡市場 | LogLoss（RandomForest+controls → +MP） | 0.2651 | 0.0481 | −0.2170 |

**不確定性**：市場簇 bootstrap 的 log loss 增益區間，Logit VR vs controls 為 [0.0069, 0.0319]、XGB MP 為 [0.0143, 0.0821]、RF MP 為 [0.0603, 0.1030]；時戳簇區間分別為 [0.0175, 0.0183]、[0.0434, 0.0446]、[0.0787, 0.0805]，全為正。折層一致性：Logit VR 在 16 / 17 折改進 log loss、15 / 17 折改進 Brier；XGB MP 為 17 / 17 與 17 / 17；RF MP 為 16 / 17 與 16 / 17。

**哪部分是 capability、哪部分是虛胖**

- **最可信的一段是 Logit VR vs Logit+PCA+controls**：對照模型已含 `p_{i,t}`、`A_t`、`z_t^PCA` 與四項鄰域控制（均值、離散度、平均相關強度、相關變異），拓撲還要再加分，才叫「形狀」有別於「一階統計」。這段在所列比較的 ECE 為 0.0055；仍需結合評分方式、分箱及不確定性，不能只憑一個 ECE 判定經濟可用性。
- **XGB／RF + MP 的大幅 Δ 要打折**：對照模型是 `p, A_t, z_t^PCA, C_i`，**不含任何 PCA 殘差資訊**。MP 的第二條過濾軸 δ 恰恰是用 PCA 殘差算出來的，所以這場比較同時放進了「殘差」與「形狀」兩件事，無法分離。若把殘差向量直接餵給 XGB／RF 而不做拓撲，增益剩多少——**該消融所讀文本未見，未披露**。這是本文最關鍵的缺口。
- **校準與判別反向**：RF+MP、XGB+MP 的 Brier／LogLoss 最好，但 ECE 分別為 0.0562 與 0.0403，反而比 market-implied 的 0.0374 更差（Logit VR 的 0.0055 最佳）。用於交易前需額外核對校準、成本與經濟損益；較佳排序指標本身不保證交易可行。
- **標籤結構性風險**：532,979 列共享 171 個結果，且 152 個是 NO。RF+MP 的 Brier 0.0126 這種量級，在高度不平衡且列間不獨立的面板上，應先當成「過擬合到少數市場」的候選解釋，而非直接當 capability。市場簇 bootstrap 已部分回應，但 171 個簇的檢定力有限，**本頁判為尚待獨立樣本驗證的假說**。
- **前瞻／成本**：Y_i 是未來結算結果，作者的設計以形成時點可得資訊建構特徵；逐折實作與時點一致性仍需獨立核查；但全文**未披露**交易成本、成交可行性與從預測到結算的持有期，所以 Δ 不能直接換成可實現收益。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位至原文章節）

1. **類別不平衡**（§5.2 第一點）：宇宙內 NO 遠多於 YES；作者以 proper scoring rules、校準檢查、時序評估處理，並另跑平衡宇宙穩健性（Table 15）。作者引外部數字說明此為預測市場通性（該引用數字屬文獻 [25]，非本文樣本）。
2. **宇宙不可外推**（§5.2 第二點）：「our market universe is centered around macro, crypto, and crypto-adjacent markets… results should not be generalized automatically to all prediction markets.」缺乏共同潛在驅動的市場，未必同樣受益。
3. **未計算完整多參數不變量**（§5.2 第三點）：只做 VR 與 multipersistence-inspired 的一維降維，未算 rank invariant、multi-graded Betti numbers 或 fibered barcode，理由是計算可管理性；作者自陳完整理論物件是自然延伸。

（另：§3.5 亦自陳是「multipersistence-adjacent」而非真正多參數持久同調；§5.1 的機制解釋作者明示為 conjecture。）

### 6.2 推斷的隱含假設（**本頁推斷**，非作者陳述）

- **假設「取 |ρ| 合併正負相關」不損資訊**。若正相關與負相關在事件結構上意義相反（例如 BTC 上看與 BTC 下看合約），`d_ij = 1 − |ρ_ij|` 會把對沖關係與同向關係壓成同一種連結，H1 環的語義可能被污染。此為本頁推斷，需以「只保留正相關」的變體檢驗，**該變體所讀文本未見**。
- **假設 171 個市場足以支撐鄰域 k=20 的拓撲估計**。每個鄰域的相關矩陣僅由訓練期估計，市場數少、且 family 極不均衡（Macro 96 vs crypto regulation 1），小型 family 的拓撲摘要統計性質不穩。
- **regime 依賴**：Table 9／10／11 顯示增益在 crypto、macro、MicroStrategy 與 25–75% 機率帶較大。作者在 §4.4 只說「gains are not concentrated within a single market domain」，未做 regime 切換下的穩定性檢定；**跨 regime 是否必然成立，本頁不主張**。
- **容量與成本**：無周轉率、無持倉期、無沖銷假設。`p̂` 是結算機率而非 t+1 價格，任何把它當交易信號的推論都超出本文範圍。
- **數據泄漏**：chronological folds、訓練期 imputer 與相關矩陣旨在減少前視風險；本頁尚未逐折執行或核對程式與可讀文字的一致性，不能據此宣稱所有泄漏均已排除。
- **輔助數據缺失**：本文未使用成交量、未平倉或訂單簿深度作為特徵，這不是 survivorship 問題，而是資訊集限制；結論只在「純價格 + 可得性」的資訊集內成立。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 純 PCA 因子壓縮（本文 §3.3 基準） | 全域線性 vs 局部非線性形狀 | 是 | 本文已內建為對照，且 PCA 單獨即勝過 market-implied |
| 鄰域一階統計（均值／離散度／相關強度／相關變異） | 一階動差 vs 多尺度拓撲不變量 | 是 | 本文已內建為 controls，VR 在其上仍有增益 |
| 相關矩陣聚類／網路中心性特徵 | 單尺度社群結構 vs 過濾下的 birth-death | TBD | 所讀文本未見此對照，圖表／附件待核 |
| PCA 殘差**直接**入模（不做拓撲） | 殘差資訊本身 vs 殘差的形狀 | TBD | **本文缺此消融**，是 MP 增益歸因的核心爭議點 |
| 完整多參數持久同調不變量（rank invariant／fibered barcode） | 一維切片 vs 真二維不變量 | 是 | 作者自陳未做（§5.2 第三點） |
| 圖神經網路／市場圖嵌入 | 固定 VR 過濾 vs 可學習訊息傳遞 | TBD | 所讀文本未見比較 |

🎤 **Interview Tip**

- **問**：「這個拓撲特徵戰勝了什麼？憑什麼說它不是重複 PCA？」
  **正確答**：最強的一段是 Logit + VR 對比 Logit + PCA + 鄰域控制（log loss 0.1355 → 0.1166），因為控制變數已含鄰域均值、離散度、平均相關強度與相關變異，拓撲要再加分才算「形狀」有別於一階統計；同時該模型 ECE 0.0055 為全場最佳。**錯答**：「因為 RF+MP 把 Brier 打到 0.0126 所以拓撲最強」——那個對照不含 PCA 殘差，殘差與形狀兩件事被混在一起，且該模型 ECE 0.0562 比 market-implied 更差。
- **問**：「532,979 筆樣本，統計上很穩吧？」
  **正確答**：樣本單位不是列而是市場，171 個市場只有 171 個市場級結果，市場之間也可能相關、其中 152 個是 NO；作者用市場簇與時戳簇 bootstrap 回應（RF MP 的 log loss 增益市場簇區間 [0.0603, 0.1030] 仍為正），但簇數有限，我不會把它當成大樣本結論。**錯答**：「五十萬筆所以 p 值很小」。

### 7.1 可證偽預測（帶日期）

- **2026-12-31 前**：若有人補上「PCA 殘差向量直接入 XGB／RF、不做拓撲」的消融，我預測 XGB+MP 與 RF+MP 相對該殘差基準的 log loss 增益會顯著縮小（成為本文最大的歸因修正）。此為**本頁推斷假說**，非論文結論。
- **2026-12-31 前**：把 `d_ij = 1 − |ρ_ij|` 改為僅正相關（`d_ij = 1 − max(ρ_ij, 0)` 或對負相關另設距離）後，若 Logit VR 的 ECE 0.0055 優勢消失，則顯示結果對距離定義敏感；這項變動本身不足以證明「把對沖誤讀成同向」是唯一原因。此為**本頁推斷假說**。
- **2026-12-31 前**：若把評估改為「每個市場只取最後一個可交易時戳為一列」（171 列），Logit VR 相對 controls 的 log loss 增益應仍為正但幅度縮小；若轉負，需進一步區分市場加權方式、時點選擇與列間依賴，不能由單一改動直接歸因。此為**本頁推斷假說**。

## §8 · For the Reader

- **因子研究員**：把這篇當「**分層消融的範本**」讀，不是當 alpha 讀。可直接搬走的是 active-set 面板建構（避免把「尚未上市／已結算」誤編為缺值）與「先上控制變數再上複雜特徵」的順序紀律。要補的洞是 §5 那段缺的殘差消融。
- **高頻／執行**：本文與你無關的部分是頻率——它用逐小時面板、預測的是**結算**而不是下一跳價格。可直接借鏡的是：任何機率輸出都應同時報 ECE，ECE 0.0055 是此樣本的校準結果，仍不足以判定線性模型更適合下注。
- **組合配置**：把 family-state 的 `(c_f, μ_f, m_f, s_f)` 理解為**跨合約的主題擁擠度／分歧度**變數，可研究將這組變數移植至其他市場，但其有效性需重新驗證。但注意作者已聲明宇宙限於 macro／crypto／crypto-adjacent，不可外推。
- **LLM-agent／事件驅動**：文中的「市場分歧結構」與 LLM 抽取的新聞事件信念可以做對照實驗——若拓撲環在事件衝擊後才形成，那它是**事後描述**而非**事前信號**；本文未做事件時戳對齊，這是可接的延伸。
- **RL 策略**：不建議直接把 `p̂` 當 reward 或狀態。可行的是把 `T_i^MP` 當狀態的一部分，但在 171 個市場、17 折的資料量下，需特別防範過擬合與市場辨識捷徑，本文沒有驗證 RL 效果。
- **研究學生**：這篇的 §2 是相當完整的 PH 教科書式鋪陳（simplicial complex → filtration → VR → homology → persistence module → barcode → bottleneck → stability → multipersistence），可當入門讀物；但 §2 與 §4 之間的落差（理論多參數 vs 實作一維切片）正是最值得練批判閱讀的地方。

## References

- 原論文：Aadit Jerfy, *Topological Compression of Prediction Market Belief States for Event Forecasting*, arXiv:2610.25756v1 [q-fin.TR], 19 Oct 2026. https://arxiv.org/abs/2610.25756v1
- 代碼：https://github.com/ajerfy/Topology-of-Prediction-Market-Belief-Dynamics
- Lineage（本文引用之關鍵前作）：
  - Zomorodian & Carlsson, *Computing persistent homology*, Discrete & Computational Geometry, 2005. https://doi.org/10.1007/s00454-004-1146-y
  - Cohen-Steiner, Edelsbrunner & Harer, *Stability of persistence diagrams*, Discrete & Computational Geometry, 2007. https://doi.org/10.1007/s00454-006-1276-5
  - Botnan & Lesnick, *An introduction to multiparameter persistence*, arXiv:2203.14289. https://arxiv.org/abs/2203.14289
  - Pun, Xia & Lee, *Persistent-homology-based machine learning and its applications: A survey*, arXiv:1811.00252. https://arxiv.org/abs/1811.00252
  - Chen & Guestrin, *XGBoost: A scalable tree boosting system*, KDD 2016. https://doi.org/10.1145/2939672.2939785
  - Breiman, *Random forests*, Machine Learning, 2001. https://doi.org/10.1023/A:1010933404324
  - Brier, *Verification of forecasts expressed in terms of probability*, Monthly Weather Review, 1950.
- 來源連結：https://arxiv.org/abs/2610.25756