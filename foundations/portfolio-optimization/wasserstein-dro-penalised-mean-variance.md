<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Wasserstein DRO penalised mean–variance 解構（Wasserstein DRO penalised mean–variance）

> **來源發布**：2026-11-23 · （無 venue） · arXiv [2611.30046](https://arxiv.org/abs/2611.30046)
> **arXiv 原文**：[Wasserstein Penalisation as an Endogenous Breadth Constraint in Penalised Mean-Variance Portfolio Selection](https://arxiv.org/abs/2611.30046v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「日頻、量價共變異數 + 監督式參數估計、長倉組合最佳化、可解釋歸因」這一點上。它解的不是最佳化問題，而是**歸因問題**：Wasserstein value guarantee 文獻（Mohajerin Esfahani & Kuhn 2018；Gao & Kleywegt 2023；Aolaritei et al. 2026）保證的是決策值，本文指出在 long-only simplex 上 penalised scalarisation 的 radius 只等於一個**寬度上限**，因此「robust 勝過 baseline」在未做寬度匹配前不可識別。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①作者證明 penalised Wasserstein-robust MV 的解，恰等於「在自身 ||φ||₂ 上封頂」的普通 MV 解（Theorem 5.2，唯一性不需要 Σ≻0），並給出單一 ridge 乘數 ν=τ/(2h\*) 令它一次求解即可（Corollary 5.3）；②trick 的核心是：penalty 在 simplex 上只是權重向量的代數函數，τ=√δ 買的是 √HHI 的價格，不是抗不確定性；③對「組合執行優化／可解釋」這條軸★最致命的一點是 Theorem 3.1——正定、譜下界、均值離散度上界**都不保證**寬度，long-only 最優解可以逼近單一資產（Table 3 把 N_eff 從 7.07 壓到 1.00）；④可核查的實證是事前審計：以追蹤自身歷史寬度為上限，可回收總增益 +45.7 的約五分之六，殘餘 +7.7（δ=10⁻⁶，5/6 panel 顯著；單位 10⁻⁴ 日均報酬）。

**X-Ray.** 這篇的價值不在演算法，在**把一個被當成「不確定性價格」的旋鈕重新命名為「集中度價格」**。放在五軸 Pareto 上，它同時踩到兩個舊坑：一是 robust 文獻只有 value guarantee、沒有 composition guarantee，作者用一個對角 witness（κ₀ 固定、ᾱ=0）證明這個缺口是量詞層級的、不是技術性的；二是實證文獻拿 robust 去打 1/N 或 sample MV，而這兩個對手的寬度本來就不同，於是「robustness 溢價」與「寬度溢價」被混算。它打開的 envelope 很窄但很硬：penalised scalarisation、long-only simplex、單期、p=2；作者自己在 Proposition 5.4 用一個 N=2 的反例指出 return-floored form 不成立，並對 exact robust form 只說支撐超平面論證不可用、不主張不存在其他形式的等價，這種**主動標示失效邊界**的寫法在 q-fin 裡少見。對量化讀者的意義是負面的但可立即使用：如果你在做 DRO 調參，你的調參預算可能花在兩個量級較小的邊際上（來源用「one to two orders of magnitude」描述寬度邊際遠寬於 robustness 項），真正該設定與披露的是 N_eff，不是 δ。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（lineage） | 本文改動 | 後果 |
|---|---|---|---|
| 保證的對象 | value / goal-value guarantee（coverage、避維度詛咒、radius 特徵化） | 改問**組成**（N_eff）；Theorem 3.1 給出「PD + 譜下界 + 均值離散度上界仍可塌成單一資產」的反例，且 witness 取 ᾱ=0 | 任何不帶 curvature 條件的 value guarantee 買不到 composition 聲明 |
| 參數語義 | radius 由資料校準，是分佈不確定性的價格 | 在 simplex 上 Proposition 4.1／4.2 顯示 τ=√δ 只對 √HHI 定價；Theorem 5.2 升級為**等式**（解唯一、約束 active） | radius 沒有 in-sample 信息超出寬度選擇；δ 不可反推 |
| 比較設計 | robust vs 1/N、vs sample MV | 要求 breadth-matched benchmark；Prop 5.1 觀測等價 + Theorem 5.2 → ex post 比較「by construction」為零 | 未匹配寬度的比較，連「哪個策略更好」都不穩定（Table 7 跨子期間變號） |
| scalarisation | (P)/(K)/(2) 常被當成同一件事 | Prop 5.4 用 N=2、μ=(0,2)′、Σ=diag(1,3)、τ=1/2 的算例證明 (K) 不成立、(2) 論證不可用 | 三個問題在實證上選出 7.1／31.2／1.3 個有效持倉的書，不可互相比較 |

### 1.2 ⚡ Eureka

> 在 long-only simplex 上，‖φ‖₂ 就是 HHI 的單調變換；Wasserstein penalty 因此只是權重向量的代數函數——**半徑買到的是寬度，不是抗不確定性。**

直覺：你以為自己在為「分佈估錯」付保費，其實你在為「不要那麼集中」付費；而一旦你承認這件事，任何「robust 贏了」的表格都必須先回答：對手的 N_eff 跟你一樣嗎？

### 1.3 信息流

```
日報酬面板 (N=255)
   └─ 樣本均值 μ̂ ＋ Ledoit–Wolf Σ̂
        └─ (P) max_{φ∈Δ} φ'μ̂ − λφ'Σ̂φ − τ‖φ‖₂,  τ=√δ
             ├─ 解 φ*_δ，取 h* = ‖φ*_δ‖₂
             │    ├─ 輸出 weights（§5.4：不識別，殘差 10⁻⁸ ~ 5×10⁻³）
             │    └─ 輸出 value（Theorem 5.2：與 breadth-matched 相同）
             └─ 等價重寫：ridge MV，penalty φ'(λΣ̂+νI)φ，ν=τ/(2h*)
                  └─ 歸因審計：事前固定 cap → 殘餘 rent（Table 1）
                       └─ Reporting standard（§5.5 十項清單）
```

## §2 · 數學層

📌 **Napkin Formula**

$$\max_{\varphi\in\Delta}\ \varphi'\hat\mu-\lambda\varphi'\hat\Sigma\varphi-\tau\|\varphi\|_2,\qquad \tau=\sqrt{\delta}\ \ \text{(P)}$$
$$\varphi^{\star}_{\delta}=\arg\max\{U(\varphi):\varphi\in C,\ \|\varphi\|_2\leq h^{\star}\},\qquad h^{\star}=\|\varphi^{\star}_{\delta}\|_2\ \ \text{(Thm 5.2)}$$
$$\text{等價 ridge：}\ \nu=\tau/(2h^{\star})\ \ \text{(Cor. 5.3)}$$

**直覺**：Theorem 5.2 是支撐超平面性質的強化版——penalty 的最優解必然落在「自己那個球的球面上」的約束 MV 解，且因為 simplex 上兩個相異點不可能共線，嚴格凸性把唯一性也一併買下（作者明說 Σ≻0 不需要）。Corollary 5.3 把它變成工程可用的形式：不必對 h 做搜尋，一次帶 ridge 的 MV 求解即可。

**複雜度**：所讀文本未見（僅稱「one solve rather than searched for」）；solver 精度以 ‖φ^cap(h\*) − φ\*_δ‖₂ 中位數 1.3×10⁻⁵、norm-cap solver 達標到 8×10⁻⁹ 報告。

**「訓練」細節**：無訓練。估計端＝樣本均值 + Ledoit–Wolf 共變異數（未另註者），λ=10，δ∈{10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³}；推論端＝Newey–West（lag max(4·HOLD,10)）+ stationary bootstrap（B=2000、期望區塊長 20）+ Romano–Wolf step-down 校正，且**不**把 14 個重疊 panel 合併做顯著性檢定。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下僅用來源逐字給出的數字（Appendix B）：倉位上限 0.6、權重 (0.6,0.4,0)、(0.6,0.2,0.2)、squared norms 0.52 與 0.44。其餘以符號保留，**不是論文實證結果**。

1. **假設** N=3、倉位上限 0.6。候選 A：φ=(0.6,0.4,0)。來源給出 squared norm 0.52。
2. 候選 B：φ′=(0.6,0.2,0.2)，同樣滿足 0.6 上限，來源給出 squared norm 0.44。→ 兩者 HHI 不同，**倉位上限不釘住寬度**；這正是 Appendix B 把「cap 讓 norm penalty 變成常數」這條路線否掉的理由。
3. 對 penalised form，Corollary 5.3 只要求把 ν=τ/(2h\*) 放進 ridge。這裡 h\*、τ、ν 以符號保留；重點不是數值，而是**不必搜尋球半徑**，直接解一次 penalty 為 φ′(λΣ̂+νI)φ 的 MV。
4. **假設**基準也被壓到同一 h\*，Theorem 5.2 說兩者目標值相同且約束 active——你觀測到的任何報酬差都**不含** in-sample 信息，只剩 Σ̂ 近平坦方向上的權重差（來源實測殘差 10⁻⁸ ~ 5×10⁻³）。
5. 若換成更大的 τ：Proposition 4.2 只保證 g(φ₂)≤g(φ₁) 的單調性，不保證連續、不保證一對一，因此**不能**由觀測到的權重反推 τ。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 市場／頻率 | 14 個 A 股 panel，日報酬，N=255，每個 panel 覆蓋 648 trading days between March 2018 and January 2024 |
| 來源 | 公開 Stockformer dataset；僅轉發日報酬矩陣，**不含** prices、identifiers、volumes、factor values |
| 估計／持有 | 60-day rolling window；§6.1 稱 held for five days（14 panel 主實驗）；Table 1／Table 5 用 first six panels、ten-day holding period |
| 窗口數 | 140 non-overlapping windows（Table 4 結構量） |
| 樣本外 | 5 日持有使 out-of-sample return blocks 在 panel 內不重疊；14 panel 之間**時間重疊**，故不做跨 panel 池化檢定，改報一致性計數 + 兩個 disjoint subperiod（2018-03 至 2020-10、2021-06 至 2024-01） |
| 成本 | 30 basis points，按 ℓ₁ turnover 每次再平衡收取一次、攤到 5 日持有 |
| 容量假設 | 所讀文本未見（無 AUM／衝擊模型；僅 N=255 與倉位上限 0.10／0.05 的橫截面約束） |

⚠️ 兩個設計不可混用：§6.3 明說 Table 6 與多設定檢查用 14 panel／5 日持有，Table 1 與 Table 5 用 first six panels／10 日持有，「Levels are therefore not comparable across the two designs」。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | https://github.com/ChenJiahaoJNU/GU2026-11（含每張表／圖的生成碼、歸檔輸出、實作錯誤日誌） |
| 數據 | https://huggingface.co/datasets/Changahou/GU2026_1（14 panel + per-file SHA-256 manifest） |
| Checkpoint | 不適用（無學習式模型）；solver 版本未披露 |
| License | 未披露 |
| 複現難度 | 中：僅需 QP/QCQP（(P)、(K)、(2) 與 norm-cap 變體）；難點在 14 panel × 5 tracking rule × 2 radius 的 Romano–Wolf + block bootstrap 管線 |
| 數據可得性 | 高（公開、checksum 可驗），但僅日報酬，無法重建成交量／因子 |

## §5 · 評測 / Benchmark

> **欄位語義警告**：本文**沒有** SOTA 競賽式比較；「前SOTA」欄一律是**基線本身**（EW／sample MV／return-matched／shrinkage），不得讀成 prior state-of-the-art。δ 的單位是半徑，不是報酬。

**表 A｜固定五欄表：known-truth 模擬（§6.4，N=255、T=60、λ=10、300 replications、五因子、稀疏 alpha 打在 twenty percent of assets）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 模擬因子模型 | median regret（% of attainable utility，越低越好） | equally weighted 91% | 83% | −8pp（83% − 91%） |
| 模擬因子模型 | 同上 | sample covariance mean–variance 548% | 83% | −465pp（83% − 548%） |
| 模擬因子模型 | 同上 | mean shrinkage 270% | 83% | −187pp（83% − 270%） |
| 模擬因子模型 | 同上 | penalized robust with a wide radius 296% | 83% | −213pp（83% − 296%） |
| 模擬因子模型 | 同上 | penalized robust with a validation-selected radius 471% | 83% | −388pp（83% − 471%） |
| 模擬因子模型 | 同上 | robust plus equally weighted 197% | 83% | −114pp（83% − 197%） |

解讀：來源把 83% 同時指派給「breadth-matched minimum variance」與「exactly solved robust」兩者，因此**不可**把這個差額全歸給 robustness；它恰恰是本文的論點（兩者在值上不可分）。重尾 t₃ 下排序不變，breadth-matched 與 exact robust 在 sixteen of sixteen configurations 勝過 EW，平均 83%–86%，而 sample MV 的 regret 為 167%–1574%（隨 N、T、λ 變動）——這是模擬，不是 A 股實證。

**表 B｜事前審計（Table 1，first six panels、60 日窗、10 日持有；單位 10⁻⁴ 日均報酬，正＝robust 較好；括號為 6 panel 中 NW 統計量超 2 且 bootstrap 區間排除零的個數）**

| cap 規則 | δ=10⁻⁶ | δ=10⁻⁴ |
|---|---|---|
| N_eff=20 | +26.7 (4/6) | +14.9 (3/6) |
| N_eff=34 | +31.4 (5/6) | +19.8 (4/6) |
| N_eff=50 | +34.5 (5/6) | +23.0 (5/6) |
| 過去寬度均值 | +7.7 (5/6) | +8.9 (2/6) |
| 過去寬度中位數 | +9.4 (4/6) | +11.3 (5/6) |
| 前一期寬度 | +7.0 (3/6) | +7.4 (5/6) |
| 近五期均值 | +7.4 (2/6) | +6.9 (2/6) |
| 近十期均值 | +8.4 (4/6) | +8.1 (3/6) |
| *參考：robust vs equally weighted* | +45.7 | +34.6 |

**表 C｜事後 rent（Table 6，14 panel、10⁻⁴ 日均報酬；sig＝NW t 絕對值超 2 且 bootstrap 排除零的 panel 數 /14）**

| δ | breadth-matched | return-matched | ordinary MV | equally weighted |
|---|---|---|---|---|
| 10⁻⁶ | +0.01 (0/14) | −2.05 (0/14) | −0.60 (1/14) | +29.82 (7/14) |
| 10⁻⁵ | +0.00 (0/14) | −2.62 (1/14) | −1.15 (1/14) | +29.25 (7/14) |
| 10⁻⁴ | +0.00 (0/14) | −4.75 (1/14) | −5.27 (4/14) | +24.63 (7/14) |

**表 D｜三個 scalarisation 不可互代（Table 5，first six panels、10 日持有）**

| 形式 | median ‖φ‖₂ | implied N_eff | gain vs EW（10⁻⁴/日） |
|---|---|---|---|
| penalised (P) | 0.375 | 7.1 | +45.7 |
| exact (D) | 0.179 | 31.2 | +10.8 |
| return-floored (K) | 0.882 | 1.3 | +57.5 |

**哪部分是 capability、哪部分是假象**：(i) 表 C 的 breadth-matched 欄是**定理的精度檢查**，不是證據——其報酬 rent 為 +0.01／+0.00／+0.00（by construction），而 median |U(φ^B)−U(φ*_δ)| 隨 δ 增大降至 1.8×10⁻⁸／1.2×10⁻⁹／8.6×10⁻¹⁵ 只說明 solver 收斂；(ii) 表 C 的 +29.82 對 EW 是「風險優化 + 寬度」的合併效果，(iii) 與 return-matched 的負值（−2.05／−2.62／−4.75）才是 penalty 的**報酬成本**，兩者都不測 robustness；(iv) 真正帶信息的只有表 B 的殘餘（+7.7 ~ +11.3），且它對 cap 規則敏感（固定 N_eff 目標只回收「a quarter to two fifths」）；(v) 假陽性風險作者自曝：λ=30 時 breadth-matched rent 出現 twelve significant panels 但中位量級僅 −0.009×10⁻⁴，對應的數值目標差 5.9×10⁻⁸（vs baseline 的 10⁻¹⁴）——近零變異數把 solver 偏差放大成 t 值（§5.4 明示此機制）。**未計入**：做空、借券、A 股漲跌停與 T+1 流動性（所讀文本未見）。

## §6 · 失效與隱含假設

### 6.1 作者自述限制（可定位）

- **Theorem 3.1 的 witness 只用對角共變異數**：只證明「不存在一致（uniform）保證」，未證明在經濟上合理的結構下寬度保證不可得；刻畫可達成的結構是 open（§7 Limitations and outlook）。Remark 2 另註：加入共同因子 b**11**′ 不影響 witness，但異質因子載荷（heterogeneous factor loadings）可能恢復寬度，作者明說「we do not claim otherwise」。
- **識別定理是靜態的**：有交易成本或多期時可行集改變、論證需重推；僅支撐超平面性質對任意凸可行集成立（§7）。
- **單一市場、單一 255 名股票池**；14 panel 時間重疊，推論靠 panel 內連續序列與兩個 disjoint subperiod，不靠跨 panel 獨立（§7）。
- **無分佈無關（distribution-free）的寬度界**：Appendix A 說明樣本共變異數下相關曲率為零、界變成空洞，只能與可量化譜下界的 regularizer 合用。
- **推論層自述**：多處 bootstrap 校正 p 值小於解析 p 值，兩估計量不一致，故把多重性結論讀成「存活 panel 計數」而非精確 p 值（§6.1）。
- **scope 限制**：等價性只屬於 penalised scalarisation，return-floored 與 exact robust 明確不繼承（Prop 5.4、Abstract）。

### 6.2 本頁推斷的隱含假設（明標推斷，圖表／附件待核）

- **regime 依賴（推斷）**：Table 7 顯示同一 robust 組合對 EW 在 panel A 為 +50.33、panel B 為 −4.61（10⁻⁴/日），對 return-matched 由 −21.08 走到 +0.69。作者用它論證「未匹配寬度的比較不穩定」；本頁進一步推斷：任何以此資料集校準的 δ 都不具跨 regime 可移植性，但這**不是**已證明的普遍命題。
- **容量／成本（推斷）**：30 basis points、ℓ₁ turnover 攤 5 日，是假設而非估計；倉位上限收到 0.05 時 EW rent 由約 +29×10⁻⁴ 降到 +5.7，說明效果對可交易性約束高度敏感——真實 A 股沖擊成本未建模。
- **數據泄漏（已檢查、未發現，但僅限所讀文本）**：Table 1 的 cap 只用 earlier windows（「no current or future information enters the cap」），這是本文設計的核心；反面（用當前窗實現寬度去設 cap）作者稱為 ex post、by construction 為零。若讀者自行以全樣本平均寬度設 cap，即等於前視。
- **文本內部不一致（待核）**：Abstract／§1 稱 penalised 組合「median Euclidean norm is 0.42 at δ=10⁻⁶, about six effective holdings」，10–90 百分位「two to nine」；但 §5.3 與 Table 5 給 median ‖φ‖₂=0.375／0.38、implied N_eff=7.1／「about seven」。兩者可能是不同半徑或不同 panel 設計，所讀文本未說明，**不可**擇一當定論。
- **輔助數據缺失 ≠ 生存者偏差（推斷）**：股票池組成、退市處理、產業分類皆未披露；這使「A 股 255 名」的代表性無法評估，但不足以斷言存在 survivorship bias。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Blanchet, Chen & Zhou (2022) exact dual | 提供 tractable dual 與 (2) 式；本文把它列為三個不可互代的 scalarisation 之一 | 否（本文引用） | 方法已成熟；本文補 composition 缺口 |
| Nguyen, Kuhn & Esfahani (2022) nonlinear shrinkage | 機制相同（robust ⇔ shrinkage）；本文新增的是**識別失敗的後果** | 否 | 作者自認「equivalence itself is not new」 |
| Jagannathan & Ma (2003) wrong constraints | 最接近的前身：約束即便先驗錯也能縮減組合；本文證明 Wasserstein penalty **正是**一個寬度上限 | 否 | 概念先行，本文形式化 |
| DeMiguel, Garlappi & Uppal (2009) 1/N | 本文顯示 1/N 是 τ→∞ 的極限（full simplex 下），且 EW 對照的 rent 隨 regime 變號 | 否 | 對手仍是最強的 naive baseline |
| 本頁對照：norm-constrained MV（直接給 h 網格） | 與 (P) 的 supporting-hyperplane 表徵相同、並非最佳化問題等價（Remark 3），但參數可直接解讀、可直接披露 | 本文 repo 公開 | 作者建議的實務替代 |

🎤 **Interview Tip**
- **正確答**：「Wasserstein radius 在 long-only penalised MV 裡不是不確定性價格，是寬度價格；要檢驗 robustness 必須先把 benchmark 的 N_eff 匹配到 robust 組合自己的 N_eff，否則你量到的是 shrinkage/寬度。作者的 Theorem 5.2 說匹配後目標值相同到 solver 精度；有信息的是事前 cap 的殘餘。」
- **錯答**：「DRO 半徑越大越穩健，所以 Sharpe 更高。」——本文的 Table 6 顯示對 return-matched 的 rent 是負的，對 EW 的正值主要來自寬度與風險優化；且 δ 變大時 penalty 的報酬成本上升（−2.05 → −4.75）。也不要把三個 scalarisation 的 N_eff（7.1／31.2／1.3）當成同一物件的不同調參。

### 7.1 可證偽預測

1. **2027-06-30 前**：若有獨立團隊在非 A 股、N 相當的 universe 上重跑 breadth-matched 對照，且 breadth-matched rent 的中位絕對值超過該市場 return-matched 或 EW 效果的 0.1%（本文在八個設定下報「at most 0.1%」），則 Theorem 5.2 的實作（ν=τ/(2h\*) 或 constraint 求解）有誤——因為這是定理，不是經驗規律。
2. **同期間**：以固定 N_eff 目標（如 20／34／50）設 cap 的回收比例，應仍顯著低於追蹤自身歷史寬度的規則（本文：a quarter to two fifths vs 約 five sixths）；若在新樣本上反轉，則「robust 組合是寬度規則」這一描述不成立。
3. **可立即檢查**：任何聲稱 DRO 改善的表格，若補上 breadth-matched 欄後效果變號，即複現本文 §1 的「omitting only the breadth-matched benchmark is enough to reverse the sign」。

## §8 · For the Reader

- **因子研究員**：把 N_eff（或 √HHI）當成與 λ 同級的顯式超參寫進 experiment grid；報告任何 robust/shrinkage 方法的增量前，先報對手與本法的 N_eff。你現有的「robust 勝 1/N」結論很可能是寬度效應。
- **高頻執行／交易成本**：本文成本模型是 30bp 一次性 ℓ₁ 攤 5 日，且 breadth-matched 與 robust 的 turnover 幾乎相同（作者據此說成本不改變結論）。若你的成本函數非線性，這個「成本中性」結論不自動成立——需自行重跑淨值序列。
- **組合配置**：作者的操作建議很直接——**直接選寬度並披露**（norm constraint 取代 ambiguity set），因為在本資料上風險最優書落在 effective number 約 thirty to fifty，超過該區間只損失報酬而不降回撤。這是單一市場的實證陳述，不可外推。
- **LLM-agent／RL 策略**：這是很好的「歸因陷阱」教材：代理人若以對手的實現寬度去設定自己的正則化半徑，等於在解同一個凸問題；把它做成 pipeline 時，請把 breadth cap 設為**過去窗**的函數（本文 Table 1 的 lower block 即是可抄的設計）。
- **研究學生**：讀 §5.4 與 §6.3 的「near-zero-variance → 大 t 值」段落，比讀主定理更有用；它解釋了為什麼在近似等價的兩條策略上做顯著性檢定會系統性騙人。

## References

- 原文：Jiahao Chen, *Wasserstein penalisation as an endogenous breadth constraint in penalised mean–variance portfolio selection*, arXiv:2611.30046v1 [q-fin.CP], 23 Nov 2026. https://arxiv.org/abs/2611.30046v1
- 代碼：https://github.com/ChenJiahaoJNU/GU2026-11　·　數據：https://huggingface.co/datasets/Changahou/GU2026_1
- Lineage（皆為原文所引）：Blanchet, Chen & Zhou (2022) *Management Science* 68(9)（exact dual, eq. 2）；Nguyen, Kuhn & Mohajerin Esfahani (2022) *Operations Research* 70(1)（robust ⇔ nonlinear shrinkage）；Mohajerin Esfahani & Kuhn (2018) *Math. Programming* 171；Gao & Kleywegt (2023) *Math. of OR* 48(2)；Aolaritei, Shafiee & Dörfler (2026, forthcoming)；Bennouna & Van Parys (2026)；Carassus & Wiesel (2025) *Finance and Stochastics* 29；Jagannathan & Ma (2003) *JF* 58(4)；DeMiguel, Garlappi & Uppal (2009) *RFS* 22(5)；Chopra & Ziemba (1993)；Best & Grauer (1991)；Kan & Smith (2008)；Ledoit & Wolf (2004)；Newey & West (1987)；Politis & Romano (1994)；Romano & Wolf (2005).
- 未核項：圖片（Figure 1 半徑—N_eff 單調圖、Figure 2 的 a=√δ/s(μ̂) 校準盒鬚圖）與附件不在本次輸入，圖中數值待核；Table 2 的 p=1／p=2 對照之完整欄位以正文 §2.2 描述為準。