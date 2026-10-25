<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=元学习搜索 alpha=因子挖掘 autonomy=Agent自主演进 -->

# LineageEvo 解構（LineageEvo）

> **來源發布**：2026-10-21 · （無 venue） · arXiv [2610.29663](https://arxiv.org/abs/2610.29663)
> **arXiv 原文**：[Lineage-Aware Search Control for LLM-Based Evolutionary Factor Mining](https://arxiv.org/abs/2610.29663v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量价表格 × 日频波段 × 元学习搜索 × 因子挖掘 × Agent自主演进」。它補的 prior gap 不是「讓 LLM 更會寫公式」，而是把 **lineage（血緣）當成 individual 與 population 之間被忽略的中間控制單元**——同一份譜系狀態既當作 operator prior 的持久作用域，又當作有限評估預算的分配單位。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `元学习搜索` | `因子挖掘` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把演化下降建成 lineage DAG，血緣身分由 primary 繼承邊遞迴決定，crosssecondary 邊只保留供體祖先。②核心 trick 是「一個 lineage、兩種用途」：譜系內的 LSSE 蒸餾成 mutation/crossover 各自的 lineage prior，與 global operator prior 做確定性融合後渲染成軟引導；同時 LSPS 先按譜系分預算、再在譜系內採父代，抑制強分支接管。③對「元学习搜索」這條軸★的意義是：被元學習的不是因子權重，而是 **變異算子本身用什麼歷史證據**，且證據的作用域由演化拓撲決定。④關鍵實證：CSI 500 上 IC 由最強基線 EoH 的 0.0399 升到 0.0512，ICIR 由 0.2493 升到 0.2740；S&P 500 上 IC 由 AlphaPROBE 的 0.0119 升到 0.0134、ICIR 由 AlphaSAGE 的 0.0668 升到 0.0907。

**X-Ray.** 這篇的貢獻不在「LLM 生成因子」這一層——那已經是紅海——而在**承認 LLM 變異是弱可控的隨機算子**，於是改用外部狀態去偏置它，而不是改 prompt 話術。放回 Pareto：它換來的是「同樣 B=200 的有效評估預算下更高的樣本效率與更寬的結構覆蓋」，代價是**整個搜尋變成有狀態的 DAG 維護 + 每次觸發一次額外 LLM 重寫**，工程複雜度高於單層反射式演化。它真正解掉的舊坑是 fitness-based selection 的 takeover：把「誰被選中」從個體層挪到譜系層，並把「好解保存」與「預算分配」拆成 elite archive 與 active population 兩套機制。它**打不開的 envelope** 很清楚：lineage prior 是從過去的 train/val 證據蒸餾出的文字先驗，對於「驗證期沒有出現過的 regime」沒有外推機制；而且判斷「同一 lineage 的局部改動可遺傳」這件事，在 DSL 語義層是否成立，論文只用附錄案例支撐，未做跨市場統計檢定（所讀文本未見）。對量化讀者：這是「搜尋控制層」的可遷移設計，不是可直接下單的 alpha 源。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（EvoAlpha / TreEvo / QuantaAlpha / ReEvo 一類 LLM-EA） | LineageEvo | 換來什麼／代價 |
|---|---|---|---|
| 搜尋單元 | 個體（LLM 改哪個父代）＋種群（fitness 選誰） | 在兩者之間插入 **lineage DAG**，節點是所有有效因子，primary 邊記 backbone 繼承，secondary 邊記 crossover 供體祖先 | 狀態可跨世代累積；代價是每代需維護 DAG 與譜系根遞迴查詢 |
| 變異引導 | 每次生成臨時給 prompt context（含反思） | **LSSE → operator-specific prior**（mutation / crossover 分開），與 global prior 由 `Φ_o` 確定性融合；強調權重 `λ_{k,o}` 由該 lineage 的績效趨勢與停滯狀態調節 | 引導可持久；代價是需事件觸發的 LLM 重寫器與 schema 校驗 |
| 預算分配 | fitness 加權直接抽父代 | **LSPS**：先抽 lineage（品質 `b_k` 與平衡 `d_k` 混合），再在 lineage 內按 `q(f)` 抽父代；另設 elite archive 與分配解耦 | 抑制 takeover；代價是 `λ` 與每 lineage 上限 0.40 需調 |
| 交叉對稱性 | 一般視為兩親本對稱重組 | **非對稱**：第一選中 lineage 的父代固定為 primary、提供 backbone，子代繼承其 lineage | 譜系身分穩定；代價是無法表達「雙主幹」混合 |

### 1.2 ⚡ Eureka

**一句話：把「血緣」從一個描述性標籤，升格為同時承載「經驗往哪存」與「預算往哪發」的單一作用域鍵。**

直覺：LLM 變異的 prompt context 是無狀態的，同一族因子反覆踩同一個坑卻不會被記住；而純 fitness 選擇又會讓早期強分支吃掉全部預算。這兩件事看似無關，但都缺一個「介於個體與種群之間、有連續性」的桶。lineage 正好是這個桶——因為 mutation 只改局部子樹、crossover 保留 primary backbone，同一 lineage 的後代在結構與 DSL 語義上天然連續，於是把經驗寫進這個桶、把預算發給這個桶，兩件事共用同一個鍵。

### 1.3 信息流

```
[iteration t]  Ω^t = (G^t, P^t, A^t, {S_{k,o}^t}, {G_o^t})
      │
      ├─(Module 1-2) LSPS: 抽 lineage k ~ π_lin(k)  →  在 P_k^t 內抽父代 f ~ π_par(f|k)
      │                （crossover 時再由另一 lineage 抽 secondary donor）
      │
      ├─(Module 3-5) 取出 S_{k,mut}^t / S_{k,cross}^t 與 G_o^t
      │                → Φ_o 融合（λ_{k,o}^t 依績效趨勢/停滯調節）→ 渲染自然語言 guidance C_t
      │                → LLM: f_{t+1} ~ p_θ(·| o_t, P_t, C_t, Γ)
      │
      ├─ 確定性校驗（語法/DSL/重複/可執行/NaN…）—— 不合格者不進 DAG、不吃有效預算
      │
      ├─ train/val 評估 → 事件 e_{t+1}（績效、算子、父子關係、表達式差異、泛化證據）
      │
      └─ Ω^{t+1} = U(Ω^t, f_{t+1}, e_{t+1})
             ├─ 更新 lineage DAG / active population / elite archive
             └─ 觸發式先驗重寫：S_{k,o}, G_o ← Ψ_o(...)
      └──────────────── 回饋至下一輪 LSPS ────────────────┘
```

## §2 · 數學層

📌 **Napkin Formula**

$$
q(f)=\alpha T(f)+\beta V(f)-\gamma\left|\mathcal{M}_{\mathrm{tr}}(f)-\mathcal{M}_{\mathrm{val}}(f)\right|,\quad (\alpha,\beta,\gamma)=(0.2,0.8,0.1)
$$

$$
\pi_{\mathrm{lin}}(k)=\lambda\frac{b_k}{\sum_\ell b_\ell}+(1-\lambda)\frac{d_k}{\sum_\ell d_\ell},\quad b_k=\max_{f\in\mathcal{P}_k}q(f),\ d_k=\frac{1}{|\mathcal{P}_k|+1},\ \lambda=0.8
$$

$$
\pi_{\mathrm{par}}(f\mid k)=\frac{q(f)}{\sum_{g\in\mathcal{P}_k}q(g)},\qquad \rho^t(v)=\rho^t(u)\ \text{若}\ (u,v)\in\mathcal{E}_{\mathrm{pri}}^t
$$

**直覺 2-3 句**：`q(f)` 是「驗證期強度優先、訓練-驗證方向反轉重罰」的穩健分數（validation 權重 0.8，絕對差懲罰 0.1），它是父代選擇、lineage 品質、最終選因子三處共用的同一把尺。`π_lin` 是「品質（`b_k`）× 覆蓋（`d_k`）」的凸組合，`λ=0.8` 偏品質但保留 0.2 給稀疏譜系；`d_k` 用 `1/(|P_k|+1)` 而非 `1/|P_k|`，讓空/極小譜系仍有非零權重。譜系根遞迴定義保證每個非根節點只有一條 incoming primary 邊，因此 lineage 分割是良定義的。

**Loss / 訓練細節**：沒有梯度訓練——LLM 是 frozen 生成算子，`θ` 不更新；所有「學習」都發生在結構化 prior `S_{k,o}` 與 `G_o` 的**文字狀態更新**上，由事件觸發（改進/退化閾值見下）驅動 LLM 重寫器，重寫失敗時由確定性 manager 保留舊狀態。`prior` 在文中被明確定義為 structured pre-generation inductive bias，不是歸一化貝氏分布（§Method, Lineage-Conditioned LLM Variation）。

**複雜度**：每次迭代一次譜系分布（對 `m_t` 個活躍譜系取 `b_k`/`d_k`）＋一次譜系內父代採樣；LLM 調用為「每個有效子代一次生成」＋「觸發時一次完整 prior 重寫」，無效/重複嘗試只記日誌、不進有效預算。原文未給 Big-O 或實際耗時，所讀文本未見。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下數字**全部為假設/示意**，只為把公式算一遍，不是論文結果；論文真實數字留在 §5。

1. **假設**活躍池只有兩個 lineage：L1 有 2 個因子 `f_A, f_B`，`q(f_A)=0.06, q(f_B)=0.04`；L2 有 3 個因子，譜系內最大 `q=0.08`。
2. 算譜系量：`b_1=0.06, b_2=0.08`；`d_1=1/(2+1)=0.3333, d_2=1/(3+1)=0.25`。
3. 算分布（`λ=0.8`）：`π_lin(L1)=0.8×(0.06/0.14)+0.2×(0.3333/0.5833)≈0.4571`；`π_lin(L2)≈0.5429`。→ **即使 L1 的品質較差，因為它更稀疏，仍拿到接近一半的預算**，這就是 `d_k` 在做的事。
4. 抽中 L1 後抽父代：`π_par(f_A|L1)=0.06/0.10=0.60`，`f_B=0.40`。→ 品質在譜系內仍然主導。
5. 融合與觸發（示意）：假設該 lineage 近期持續改善，融合權重取 local 0.7 / global 0.3，渲染出以局部有效改法為主的 guidance；子代生成後 `|IC_child|=0.0400`、`|IC_par|=0.0250`，差 0.0150 ≥ `max(0.003, 0.30×0.0250=0.0075)` → **改進觸發成立**，進入 prior 重寫。
6. 對照真實案例（所讀原文 Appendix A，逐字）：CSI 500 一次 mutation 把 `TsSkew` 換成 `TsCov`，validation `|IC|` 由 0.0209 升到 0.0369，同時觸發 lineage 與 global prior 重寫；sibling 分支的下一次事件直接收到該 prior 並沿用同一替換模式。觸發邏輯與第 5 步同構。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 市場 | CSI 500（中國 A 股）、S&P 500（美股） |
| 頻率 | daily（Qlib-compatible pipeline） |
| 預測目標 | `Ref($close,-2)/Ref($close,-1)-1`（next-period return） |
| 訓練 | 2015-01-01 至 2020-12-31 |
| 驗證 | 2021-01-01 至 2022-04-30 |
| 測試 | 2022-05-01 至 2026-04-30 |
| 樣本外聲明 | 測試期僅在最終選擇規則與因子集固定後才訪問；因子方向由 validation 決定（`sign(M_val)`），不使用任何 test 統計量 |
| 回測 | Qlib TopK-Dropout：`topk=50`、`n_drop=5`、`risk_degree=0.95`、每日調倉 |
| 成本 | `open_cost=0.0015`、`close_cost=0.0025`、`min_cost=5` |
| 成分股數量／資料供應商版本 | 所讀文本未見 |

**容量假設**：TopK=50、每日調倉、`min_cost=5` 的設定隱含「中大型流動性個股、單筆可承受最小成本」的執行環境；原文未給資金規模、換手率或容量曲線（所讀文本未見，圖表待核）。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 所讀文本未披露（正文與附錄未出現程式碼連結） |
| Checkpoint | 不自訓權重；生成端為 `deepseek-v4-flash`（OpenAI-compatible client），temperature / maximum tokens = 0.7 / 8000 |
| License | 所讀文本未披露 |
| 複現難度 | 中高：Appendix E 給了四類 prompt 骨架（seed / mutation / crossover / prior rewriting）與四個 prior JSON schema，但 DSL 與 operator 定義在範本中被 `[Shared DSL and operator definitions omitted.]` 省略；Appendix D 給了完整超參（B=200、初始 lineage 根 10、每代 3 mutation + 2 crossover、每槽最多 5 次嘗試、population 50 / elite 20、單 lineage 上限 0.40、`(0.2,0.8,0.1)`、`λ_LSPS=0.8`、觸發閾值分市場） |
| 數據可得性 | Qlib 框架公開；具體行情/財務資料版本與清洗腳本所讀文本未披露 |

**可複現性缺口**：真正卡住複現的不是超參，而是 **DSL 算子集合 + prior 的實際 JSON schema 細節 + 每個市場的 prompt 實例**；前兩者在正文只給欄位名（`successful_mutation_patterns` / `failed_mutation_patterns` / `hint` / `bias_risk` 等），第三者在 Appendix E 為骨架。

## §5 · 評測 / Benchmark

> 前 SOTA 欄**逐字取該指標在所讀原文 Table 1 中最強的單一基線**，未對多個基線做 min/max 合成；Δ 一律為「本方法 − 前SOTA」。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| CSI 500 | IC ↑ | EoH 0.0399 | 0.0512 | +0.0113 |
| CSI 500 | ICIR ↑ | EoH 0.2493 | 0.2740 | +0.0247 |
| CSI 500 | IR ↑ | AlphaSAGE 0.7654 | 0.9856 | +0.2202 |
| CSI 500 | MDD ↓（表中為非負幅度） | AlphaPROBE 20.48% | 19.39% | —（由前兩欄比較） |
| S&P 500 | IC ↑ | AlphaPROBE 0.0119 | 0.0134 | +0.0015 |
| S&P 500 | ICIR ↑ | AlphaSAGE 0.0668 | 0.0907 | +0.0239 |
| S&P 500 | IR ↑ | GP 1.0958 | 1.1693 | +0.0735 |
| S&P 500 | MDD ↓（表中為非負幅度） | ReEvo 18.94% | 17.05% | —（由前兩欄比較） |

**AR（不列 Δ，只列逐字值）**：CSI 500 本方法 15.55%，同市場最強基線 Alpha158 為 15.50%；S&P 500 本方法 27.00%，同市場最強基線 EoH 為 23.13%。CSI 500 的 AR 差距在逐字精度下極小，**「最強投資組合表現」在 CSI 500 上主要來自 IR 與 MDD，而不是 AR**。

**消融（CSI 500，Table 2，逐字）**：Full 0.0512 / 0.2740；w/o all priors 0.0384 / 0.1994；w/o lineage priors 0.0390 / 0.2041；w/o global priors 0.0495 / 0.2579；w/o mutation priors 0.0438 / 0.2673；w/o crossover priors 0.0445 / 0.2362。**lineage prior 的移除（0.0390）比 global prior 的移除（0.0495）掉得更深**，這是「譜系內經驗才是主要養分」的直接證據；在這組 CSI 500 消融中，移除 crossover 先驗後 ICIR 為 0.2362，低於移除 mutation 先驗後的 0.2673（Full 為 0.2740）；僅就本表而言，crossover 先驗對 ICIR 的貢獻更大。這項比較不單獨證明 primary lineage／secondary 更新規則的因果作用。

**多樣性（CSI 500，Table 3，逐字）**：w/o LSPS 為 IC 0.0381 / ICIR 0.2415 / Lineage Entropy 0.405 / AST Dist. 0.509 / Eff. Struct. 37.89；Full 為 0.0512 / 0.2740 / 2.154 / 0.747 / 41.90。**同時改善品質與多樣性**是這篇最反直覺的結果，因為一般認為多樣性要靠犧牲 fitness 換。

**效率**：Fig.3 記載 LineageEvo 在約 75 次有效評估內到達較強平台期——此為圖說文字敘述，數值曲線本身未提取（圖表待核）。

**哪些 Δ 是真 capability、哪些存疑**：
- **較可信**：消融是同一 pipeline、同一 LLM、同一預算下的受控對照，lineage prior 與 LSPS 的增益方向一致且量級可觀，這部分讀作機制性效果。
- **存疑一（基線公平性）**：所有 LLM 方法共用 `deepseek-v4-flash`，但 prompt 工程、重試策略與預算控制是否對等，正文只說「same fusion and prompt infrastructure unless a prior source is explicitly removed」是針對自家變體，**對外部基線的 prompt 對等性未披露**。
- **存疑二（單次運行）**：所讀文本未見種子數、多次運行均值/方差或顯著性檢定；Table 1 的 Δ 是點估計，**不可讀成跨市場的必然因果**。
- **存疑三（成本與容量）**：成本設定存在且統一，但 `min_cost=5` 的具體含義與資金曲線未給；MDD 為算術累積下的 signed drawdown 幅度，與幾何回撤不同口徑。
- **存疑四（選擇效應）**：最終因子從 elite archive 按 `q(f)` 選、方向由 validation 定，test 不參與——這一條寫得明確，屬**可控**；但「top 20 factors 合併等權」的口徑使結果對因子相關性結構敏感，原文未給相關性矩陣。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位者）

- **§Conclusion**：作者明說未來工作才去比較 lineage 與其他中間組織單元（clusters、niches、retrieval neighborhoods）在經驗重用與搜尋控制上的各自優勢——即**「lineage 是最佳中間單元」在本文並未被證明，只是被採用**。
- **§Method, Lineage-Conditioned LLM Variation**：作者主動界定 `prior` 是 structured inductive bias，而非歸一化貝氏分布——即**不提供任何不確定性量化或校準保證**。
- **§Method, Problem Setup**：lineage 的語義連續性主張（nearby descendants retain greater structural and DSL-level financial-semantic continuity）標注為「as verified in Appendix」，而 Appendix A 給的是**單一 CSI 500 案例的兩次 mutation 事件**，非統計檢定（此為本頁觀察，非作者自述限制）。
- 所讀文本未見：失敗案例統計、跨市場的 lineage 深度/寬度分布、LLM 重寫失敗率的量化。圖表／附件待核。

### 6.2 推斷的隱含假設（本頁推斷，非作者主張）

| 假設 | 內容 | 失效條件 |
|---|---|---|
| **Regime 依賴** | prior 由 train/val 的改進/退化事件蒸餾，validation 窗口為 2021-01-01 至 2022-04-30 | 若測試期主導風格與 validation 期結構性不同，`S_{k,o}` 記住的是「上一段 regime 有效的局部改法」，會系統性偏置變異方向；且停滯時才提高 global prior 權重，屬於**事後補救而非事前偵測** |
| **譜系身分穩定** | 子代 lineage = primary parent 的 lineage，secondary donor 只記邊 | 當 crossover 的供體子樹貢獻大於 backbone 時，「這個子代屬於哪一族」的語義開始失真，prior 會被餵進與自身血緣不符的證據 |
| **局部改動可遺傳** | mutation 只改葉/小子樹，故同族後代語義連續 | 若 DSL 中存在強交互算子（原文 DSL 定義被省略，無法核對），局部替換可能導致輸出分布劇變，連續性假設斷裂 |
| **預算守恆** | 無效/重複候選不吃有效預算（B=200） | 若不成立，譜系間比較會被「哪個譜系更難生成合法子代」污染——原文稱無效嘗試不進預算，屬**已聲明且可控** |
| **容量與成本** | TopK=50、每日調倉、固定 `open_cost/close_cost/min_cost` | 換手放大或資金規模上升時，factor-level 的 IC 增益未必轉化為淨 AR；本文 AR 在 CSI 500 上與 Alpha158 幾乎持平，正是這個轉化鏈的薄弱環節 |
| **數據泄漏** | 方向由 validation 定、test 只在最終選擇後訪問、prompt 內只餵絕對強度不含符號 | 這一條設計上乾淨；但**LLM 預訓練語料可能已含 2022 之後的市場敘事**，這一條論文未討論（所讀文本未見），屬模型層而非流程層的泄漏風險 |
| **輔助數據缺失非倖存者偏差** | 股票池與停牌/退市處理未在正文說明 | 所讀文本未見，待 Appendix/代碼核對；不能假設已處理 |

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| EvoAlpha（LLM-EA 因子挖掘） | 有演化但無譜系狀態；經驗不跨世代持久 | 所讀文本未披露 | 基線，Table 1 未列 |
| TreEvo（樹狀思維演化） | 用樹結構表徵公式層級，但無 lineage 級預算分配 | 所讀文本未披露 | 基線，Table 1 未列 |
| QuantaAlpha（軌跡級變異/交叉） | 重用「軌跡」經驗，屬個體/路徑層；LineageEvo 是拓撲層 | 所讀文本未披露 | 基線，Table 1 未列 |
| ReEvo（反思式超啟發） | 自由形式反思 vs **事件觸發 + schema 校驗的結構化 prior 重寫** | 所讀文本未披露 | 基線，Table 1 有數字（CSI 500 IC 0.0303 / ICIR 0.1950） |
| AlphaPROBE（檢索 + 圖偏置演化） | 圖是「檢索偏置」，LineageEvo 的圖是「預算與先驗的作用域」 | 所讀文本未披露 | 基線，S&P 500 IC 最強者 |
| RD-Agent（多智能體因子+模型聯合優化） | 優化對象含模型，非純因子搜尋 | 所讀文本未披露 | 基線，早期收斂快但平台低 |
| AlphaSAGE（GFlowNet 結構感知） | 非 LLM 的結構化探索，多樣性機制內生 | 所讀文本未披露 | 基線，S&P 500 ICIR 最強者 |

🎤 **Interview Tip**

- **問**：為什麼把控制單元放在 lineage，而不是加強 prompt 反思或加大 population？
- **對的答法**：因為兩個失效模式不同源但共桶——(a) LLM 變異的 context 無狀態，導致同一族因子反覆踩同一坑；(b) fitness 選擇在有限預算下會 takeover。lineage 同時是「經驗的桶」與「預算的桶」，一次解決兩個；而且 ablation 顯示拆掉 lineage prior 的損失（IC 0.0390）大於拆掉 global prior（0.0495），拆掉 LSPS 則 IC 掉到 0.0381 且 lineage entropy 由 2.154 掉到 0.405——**品質與多樣性同向變差**，說明這不是單純的 exploration-exploitation 折衷。
- **錯的答法**：「因為 LLM 語義能力強，譜系資訊讓 prompt 更豐富」——這是把機制講成話術；本文的關鍵是**確定性融合 + 分層抽樣**這兩個非 LLM 元件，LLM 只負責語義變異。

### 7.1 可證偽預測（本頁預測，尚待驗證）

1. **2027-04-30 前**：若在其他 A 股指數（如 CSI 300）重跑，LSPS 帶來的 lineage entropy 提升幅度應仍為數量級跳升（0.405 → 2.154 這種級別），而 IC 增益應明顯縮小——因為大市值池的結構冗餘更低。
2. **2027-04-30 前**：把 `λ_LSPS` 由 0.8 單調下調，IC 應先平後降，而 entropy 應單調上升；若出現 IC 與 entropy 同時上升的區間，則「品質-覆蓋折衷」的敘事需修正。
3. **2027-04-30 前**：把 prior 重寫改為**無條件每代觸發**（去掉事件觸發門檻），預期 IC 下降——因為低資訊事件會稀釋結構化 prior 的信噪比。
4. **本頁斷言**：這套機制在**日頻以上、因子 DSL 局部可組合**的場景可遷移；遷到高頻 tick 級特徵時，因「局部子樹替換」的語義連續性假設不成立，預期增益消失（未驗證）。

## §8 · For the Reader

- **因子研究員**：把 LSPS 的兩階段抽樣（先譜系後個體）與 `q(f)=0.2T+0.8V−0.1|Δ|` 這把尺直接搬進你現有的 GP/LLM 搜尋器——**這兩個元件不依賴 LLM，是最低成本的移植點**。先量測你自己的 lineage entropy 與 AST 距離基線，再決定要不要動預算分配。
- **組合配置 / 多因子組合**：注意 Table 1 是「top 20 因子等權合併」的結果，多樣性指標改善意味著**候選間的相關結構更鬆**，這對 downstream 的優化器（非等權）價值可能大於對等權組合；但相關係數矩陣所讀文本未見，需自行重建。
- **LLM-agent 研究者**：這篇提供了「agent 記憶的作用域該由什麼決定」的一個非語義答案——**由演化拓撲決定**。可與 retrieval neighborhood、cluster 這兩種替代作用域做 A/B，作者自己在結論中把這件事列為 future work。
- **RL / 搜尋演算法研究者**：把 fitness-based selection 換成「分層預算 + 獨立精英檔案」是經典 diversity-control 手法在 LLM 算子上的再應用；值得測的是 **elite archive 與 active population 解耦**這一點——它讓「保存好解」不再消耗探索預算，這是可以獨立消融的設計。
- **研究學生**：讀法建議先讀 §Method 的 Eq.(3)(4)（lineage 根遞迴）與 Eq.(6)(8)(9)（LSPS 三段），再看 Appendix D 的觸發閾值；Appendix A 的 prior 文字片段是全文唯一能看見「先驗長什麼樣」的地方，別跳過。

## References

- 原論文：Xintong Liu, Chenkai Zhang, Yanzhao Ma, Peng Yang. *Lineage-Aware Search Control for LLM-Based Evolutionary Factor Mining.* arXiv:2610.29663v1 [q-fin.CP], 2026-10-21. https://arxiv.org/abs/2610.29663v1
- 譜系／多樣性理論線：Burke et al. (2003), *Is increased diversity in genetic programming beneficial?*；Gabor et al. (2018), *Inheritance-based diversity measures for explicit convergence control in evolutionary algorithms*；Burlacu et al. (2013), *Visualization of genetic lineages and inheritance information in genetic programming*；Sudholt (2018), *The benefits of population diversity in evolutionary algorithms*.
- LLM-EA 線：Liu et al. (2023), *Large language models as evolutionary optimizers*；Liu et al. (2024), *Evolution of heuristics (EoH)*；Ye et al. (2024), *ReEvo*；Luo et al. (2026b), *EvoAlpha*；Ren et al. (2025), *TreEvo*；Han et al. (2026), *QuantaAlpha*.
- 因子挖掘基線：Kakushadze (2016), *101 Formulaic Alphas*；Yang et al. (2020), *Qlib*；Yu et al. (2023), *AlphaGen*；Chen et al. (2025), *AlphaSAGE*；Li et al. (2025), *RD-Agent-Quant*；Guo et al. (2026), *AlphaPROBE*；Wang et al. (2023), *Alpha-GPT*；Tang et al. (2025), *AlphaAgent*.
