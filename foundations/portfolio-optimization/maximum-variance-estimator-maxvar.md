<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Maximum Variance estimator (MaxVar) 解構（Maximum Variance estimator (MaxVar)）

> **來源發布**：2026-11-23 · （無 venue） · arXiv [2611.30843](https://arxiv.org/abs/2611.30843)
> **arXiv 原文**：[Cleaning correlation matrices with external information: the Maximum Variance estimat](https://arxiv.org/abs/2611.30843v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：在「量價表格／中長周期／監督回歸」軸上，MaxVar 把外生分組先驗注入相關矩陣清洗，填補 RIE 只能洗特徵值、結構化模型靠資料推斷結構的 prior gap；Alpha 生成落在組合執行優化，而非收益預測。此處「監督回歸」應讀成「用外生標籤監督結構估計」，不是 alpha 回歸。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 用資產類別／地區等外部分組，對每組構造單位範數最大方差組合，把相關矩陣前 K 個特徵向量約束在其張成子空間，其餘譜壓成單一特徵值。② 核心 trick 是「K 個組內最大方差組合＋單一 δ 補齊」：參數約 n 而非 n^2/2，觀測數小於資產數仍正定。③ 對「組合執行優化」軸★，它直接服務需要逆矩陣的 risk parity；RIE 只洗特徵值，無法利用分組。④ 1993–2018 的 47 期貨上，risk parity Sharpe 為 MaxVar 1.28、RIE 1.24、empirical 1.17（before costs）；Monte Carlo 距離 2.09 對 RIE 2.99、empirical 3.24，但該模擬按構造對 MaxVar 有利。

**X-Ray.** 這篇不是 alpha 預測論文，而是 risk model 估計器論文：它用外生分組換取小樣本下的數值穩定。舊坑很具體：RIE 家族只洗特徵值、保留噪聲特徵向量；原文另列三類結構化方法：區塊模型、以稀疏殘差建模的因子模型，以及用聚類從資料推斷結構的階層模型。MaxVar 把「資產類別／地區」壓成 K 個組內最大方差組合，令前 K 個特徵向量只能在其張成子空間內旋轉，其餘譜壓成單一 δ；參數從 n^2/2 降到約 n，且觀測數小於 n 仍正定。代價是 §2.3 明說的結構偏差：若真實相關結構不服合分組，偏差不隨 T 消失。它打不開的 envelope 也清楚：實證只有 47 期貨、K=3、單一 risk parity 組合，成本未計，Sharpe 差異小於統計誤差，未檢定顯著性；重疊分組、其他組合用途、更細分組都留給未來。對量化讀者，正確定位是「短窗 risk model 的強先驗 baseline」或 shrinkage target，不是通用 alpha，也不是無條件優於 RIE 的結論。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 前作／家族 | 核心做法 | 留下的問題 | MaxVar 的改動 |
|---|---|---|---|
| RIE／非線性收縮（Bun et al.；Ledoit & Wolf） | 只改特徵值，保留經驗特徵向量；旋轉不變 | 無法利用外生分組修正特徵向量 | 直接把特徵向量約束到外生分組張成的子空間 |
| 結構化收縮／block／factor（Ledoit & Wolf；Engle & Kelly；Fan et al.） | 向單因子、常相關、block 或稀疏殘差收縮 | 結構常由資料推斷或假設；錯結構有偏差 | 用事前已知分組，不從樣本聚類；偏差來源明確 |
| Valeyre et al. 2019 Fundamental MaxVar | 用財務準則建最大方差組合，清洗單股票相關矩陣 | 針對特定財務準則，非通用任意分組 | 推廣到任意分組（可重疊），給出性質與期貨回測 |
| Empirical matrix | 無偏差，但變異數約 n^2/T | 小樣本下逆矩陣放大噪聲 | 以結構偏差換低變異數，短窗更穩 |

### 1.2 ⚡ Eureka

一句話：把每個外生分組的最大方差組合當成低維「原子軌道」，用它們張成的子空間承載大特徵向量；正交補空間沒有資訊，就用單一 δ 補齊。直覺是：不必估計 n(n−1)/2 個相關，只估計每組載荷與組間協方差；分組正確時，去噪；分組錯誤時，把真實結構抹平。

### 1.3 信息流

外部分組 G_k（資產類別／地區）＋經驗相關 C（n×n）  
→ 每組 leading eigenvector p_k（支撐集在 G_k 內）  
→ P=[p_1…p_K]；G=PᵀP；H=PᵀCP  
→ 廣義特徵問題 H a = μ G a，得 Z=P G^(−1/2) Ω、μ  
→ F=Z M Zᵀ + δ(I−ZZᵀ)，δ=(n−Σμ)/(n−K)  
→ F̂=D^(−1/2) F D^(−1/2)（對角歸一化）  
→ risk parity 權重 ∝ Σ^(−1) F̂^(−1) 1

## §2 · 數學層

> 📌 Napkin Formula：  
> p_k = argmax_w wᵀCw，s.t. wᵀw=1、w_i=0（i∉G_k）；H a = μ G a，G=PᵀP、H=PᵀCP。  
> F = Z M Zᵀ + δ(I_n − ZZᵀ)，δ=(n−Σμ_k)/(n−K)；F̂_ij = F_ij / √(F_ii F_jj)。  
> 參數約 n+K^2/2；47 期貨／K=3 時原文給 50 parameters instead of 1081。複雜度原文未給；主要成本在 K 個組內 leading eigenvector 與 K×K 廣義特徵分解（本頁推斷）。

直覺：先讓每組內最強的共同因子（最大方差組合）定義該組的「方向」，再在這些方向的張成空間裡重新對角化；剩下 n−K 維沒有外生資訊，全部壓成同一個 δ，避免小特徵值被逆矩陣放大。loss／訓練細節：沒有監督 loss、沒有梯度訓練；C 由 EMA 估計（日波動 η=1/100、相關 η′），MaxVar 是確定性譜重構。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下數字全為**假設／示意**，不是論文實證；論文真實數字留在 §5。

假設 n=3、K=2，分組 G_1={1,2}、G_2={3}，假想相關矩陣 C=[[1, 0.5, 0.2], [0.5, 1, −0.2], [0.2, −0.2, 1]]。

1. 組內最大方差：G_1 的限制矩陣 [[1,0.5],[0.5,1]]，leading eigenvector 為 (1/√2, 1/√2)，方差 1.5；p_1=(1/√2,1/√2,0)。G_2 只有資產 3，p_2=(0,0,1)，方差 1。
2. 組間：p_1·p_2=0，故 G=I。這組假設的兩個跨組項正負抵消，p_1ᵀCp_2=(0.2−0.2)/√2=0；因此 H=diag(1.5,1)，μ=(1.5,1)。
3. 補齊值：δ=(n−Σμ)/(n−K)=(3−2.5)/(3−2)=0.5。
4. 重構：Z M Zᵀ 給 [[0.75,0.75,0],[0.75,0.75,0],[0,0,1]]；δ(I−ZZᵀ) 給 [[0.25,−0.25,0],[−0.25,0.25,0],[0,0,0]]；相加得 F=[[1,0.5,0],[0.5,1,0],[0,0,1]]。
5. 歸一化：F 的對角全為 1，所以 F̂=F；組內相關 0.5 保留。這個例子裡，兩個跨組相關 0.2 與 −0.2 在組合中抵消，因此重構值都是 0。一般情況下，組合之間的共變異可以非零，不能從這個例子推論所有跨組相關都會歸零。

## §3 · 數據層

- 市場／規模：47 futures，其中 24 on stock indices、14 on government bonds、9 on currencies（Table 2）。
- 時段：資料起 8 May 1985、迄 31 December 2018；統計期為 1 January 1993 至 31 December 2018。原文另述 Sharpe 統計誤差以 26 years 量級描述。
- 頻率與清洗：日收益以 EMA 估波動（η=1/100）後標準化；相關矩陣用週收益 EMA，η′=1/750（約 150 個獨立週收益）與 η′=1/100（約 20 個週收益）兩種窗。
- 成本假設：Transaction costs and market impact are not taken into account（§4.1）。
- 樣本外與容量：EMA 滚動估計，但所讀文本未見正式 train/test 切分；容量、保證金、展期成本未披露。原始數據供應商與清洗細節未披露。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 所讀文本未見 | 不適用／未見 | 未披露 | 中高：需 47 期貨日／週資料、EMA 參數、RIE 實作；本頁推斷 | 合約清單見 Table 2；原始數據供應商與清洗未披露 |

## §5 · 評測 / Benchmark

> 原文未使用 SOTA 一詞；「前SOTA」欄僅放所讀原文逐字基線數值，多基線逐行列出，**不合成區間**；Δ 省略，避免把不同基線混成假 SOTA。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 47 futures；Monte Carlo（T=150，100 samples）；基線 Empirical | Frobenius distance（越低越好） | 3.24 | 2.09 | — |
| 47 futures；Monte Carlo（T=150，100 samples）；基線 RIE | Frobenius distance（越低越好） | 2.99 | 2.09 | — |
| 47 futures，1993–2018，η′=1/750；基線 Empirical | Sharpe ratio（before costs） | 1.17 | 1.28 | — |
| 47 futures，1993–2018，η′=1/750；基線 RIE | Sharpe ratio（before costs） | 1.24 | 1.28 | — |
| 47 futures，1993–2018，η′=1/100；基線 RIE | Sharpe ratio（before costs） | 0.93 | 1.05 | — |
| 47 futures，1993–2018，η′=1/100；基線 Empirical | Sharpe ratio（before costs） | 未披露（原文表中為 –） | 1.05 | — |

解讀：Monte Carlo 這欄不能當 capability 證明。原文 §3 自述參考矩陣由 MaxVar 清洗 28 December 2018 的 47 期貨而得，按構造符合估計器假設，是「favourable」設定；2.09 低於 3.24 與 2.99 只說明假設正確時的去噪潛力，不說明偏差。真正決定性的是 §4.3 的 risk parity：1.28 高於 1.24 與 1.17，且 η′=1/100 時 1.05 對 0.93，原文稱 advantage increases when the estimation window is shortened。但原文同時警告：這些差異小於 26 年 Sharpe 的統計誤差（of the order of 1/√26≈0.2），且 statistical significance has not been tested；結果只涉及單一 portfolio，不能外推到其他用途。成本與市場衝擊未計，所以「真 capability」與「過擬合／成本未計」在所讀文本中無法分離；只能說在該 universe、該組合、該窗口下，MaxVar 的排序較前。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

- §2.3：MaxVar 對真實 C_0 有 bias，除非 C_0 正好符合 (5) 的結構；此 bias 不隨 T 消失，只是 variance 小。
- §3：Monte Carlo 的 reference matrix 有估計器假設的結構，是 favourable specification；只測「假設正確時去噪多少」，不測 bias。
- §4.1：Transaction costs and market impact are not taken into account。
- §4.2：heat map 的平滑「部分去噪、部分可能移除同類合約的真實差異」，兩者無法從圖分辨。
- §4.3：Sharpe 差異小於統計誤差（of the order of 1/√26≈0.2）；未檢定顯著性；只涉及單一 portfolio，不建立其他用途。
- §5：需檢定 Sharpe 差異顯著性、評估其他使用相關矩陣的組合、用更細分組降低 bias、把 MaxVar 當 shrinkage target。

### 6.2 推斷的隱含假設（本頁推斷，非作者自述）

- Regime 依賴：分組（資產類別／地區）在 1993–2018 的相關結構需相對穩定；若 regime 改變使組內最大方差組合不再承載主要共同因子，bias 可能壓過 variance 優勢。原文未做 regime 分段。
- 容量：47 期貨是 most liquid；策略容量、保證金、展期未分析，不能外推到低流動性合約。
- 成本：§4.1 明示未計；加成本後 Sharpe 排序可能改變，但所讀文本未給 breakeven 成本或失效閾值。
- 數據泄漏：外部分組若用樣本內資訊或事後分類會泄漏；原文分組是事前資產類別，但未描述分類時點與重平衡。輔助數據缺失不是 survivorship 的替代解釋。
- Survivorship：合約清單是 most liquid，納入規則未披露；若只含存活合約，可能有 survivorship，原文未披露。
- 重疊分組：§2.1 說分組可重疊（資產類別＋地區），但 §4 實證只用 K=3 的資產類別；重疊時 G 非單位、數值穩定性未測。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| RIE／非線性收縮 | 只改特徵值、旋轉不變；不利用外生分組 | 文獻公開；本頁未核實作 | 基線（Table 1／Table 3） |
| Ledoit–Wolf 線性收縮／常相關目標 | 向目標收縮；特徵向量部分保留 | 文獻公開 | 文獻基線，本文未直接回測 |
| Block／factor／稀疏殘差 | 結構由資料推斷或假設；錯結構有 bias | 文獻公開 | 文獻對照 |
| Valeyre et al. 2019 Fundamental MaxVar | 用財務準則建最大方差組合，單股票 | 原文引用 arXiv:2001.08911 | 前作 |
| MaxVar | 外生任意分組、重疊、正定、約 n 參數 | 所讀文本未見 code／repo | 本文 |

🎤 Interview Tip  
- 正確答：MaxVar 的關鍵是「用外生分組把特徵向量限制在 K 維子空間，以 bias 換 variance」；RIE 是旋轉不變、只洗特徵值。它的實證是 47 期貨、K=3、risk parity，且差異未檢定顯著性、成本未計。  
- 錯答：說它用 clustering 從資料找分組；說它旋轉不變；說 1.28 顯著優於 1.24；說它保證所有市場／所有組合都更好。

### 7.1 可證偽預測

以 2026-11-23 的 arXiv v1 為凍結版本：若在該日之後的首個完整樣本外年度，用同一 47 期貨、同一 risk parity 定義、同一 K=3 資產類別分組重跑，並把 η′ 設為 1/100，則 MaxVar 的 Sharpe 應不低於 RIE；若反轉，則「短窗下低變異數優勢」在該 universe 不成立。另一條：若把分組換成與資產類別無關的隨機分組，Monte Carlo 的距離優勢應消失；若仍存在，則優勢不是來自外生結構。以上為本頁預測，尚待驗證。

## §8 · For the Reader

- **因子研究員**：把 MaxVar 當 risk model 清洗器，不是 alpha。先測分組穩定性與 bias；可用 MaxVar 作 shrinkage target，把它的低 variance 與 empirical 的無 bias 結合（原文 §5 明列此擴展）。
- **高頻執行**：本文不是執行論文；短窗相關估計可能受益，但成本／impact 未計，且期貨展期與容量未分析。直接套用前需自行加成本與流動性約束。
- **組合配置**：risk parity／maximum diversification 是直接 use case；但實證只有單一 portfolio、單一 universe，Sharpe 差異未檢定顯著性。不要外推成「所有配置都更好」。
- **LLM-agent／RL 策略**：可把外生分組當可解釋先驗，但 agent 若自動選分組，需避免樣本內泄漏；原文未測自適應分組或線上更新。
- **研究學生**：先讀 §2 命題與 §2.5 玩具例，再做 Monte Carlo，最後才做 real data；復現需 RIE 與 47 期貨資料，數據供應商未披露。

## References

- Valeyre, S. (2026). Cleaning correlation matrices with external information: the Maximum Variance estimator. arXiv:2611.30843. https://arxiv.org/abs/2611.30843
- Valeyre, S., Kuperstein, S., Grebenkov, D. S., & Aboura, S. (2019). The market neutral fundamental maximum variance portfolios. arXiv:2001.08911.
- Bun, J., Bouchaud, J.-P., & Potters, M. (2016). Cleaning correlation matrices. Risk Magazine.
- Bun, J., Bouchaud, J.-P., & Potters, M. (2017). Cleaning large correlation matrices: tools from random matrix theory. Physics Reports 666, 1–109.
- Ledoit, O., & Wolf, M. (2003; 2004a; 2004b; 2012). 線性收縮、常相關目標、非線性收縮系列。
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise dressing of financial correlation matrices. Physical Review Letters 83, 1467–1470.
- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Mathematics of the USSR-Sbornik 1, 457–483.
- Choueifaty, Y., & Coignard, Y. (2008). Toward maximum diversification. Journal of Portfolio Management 35(1), 40–51.
- Markowitz, H. (1952). Portfolio selection. Journal of Finance 7, 77–91.
- Michaud, R. O. (1989). The Markowitz optimization enigma. Financial Analysts Journal 45(1), 31–42.
- Pafka, S., & Kondor, I. (2003). Noisy covariance matrices and portfolio optimization II. Physica A 319, 487–494.