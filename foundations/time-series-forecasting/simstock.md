<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# SimStock 解構

> **發布**：2024-08-27 · （無 venue）
> **QuantML 導讀**：[股票相似性的时间表示学习及其在投资管理中的应用](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247485992&idx=1&sn=9c13d3832825cf98c8b1eb593acf6ce9&chksm=ce7e6d36f909e4203370742f24372136d5bec9d84ccf2044028a2cab8cd7ac3f3a8e7f6fe0aa#rd)
> **核心定位**：落點於「日频波段 × 端到端表征 × 黑盒自動化」，試圖以時間域泛化（Temporal Domain Generalization）彌補傳統量價相關性在跨市場/非平穩環境下的結構斷裂。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出 SimStock 框架，將自監督學習與 LSTM 驅動的參數動態更新結合，學習非平穩市場下的股票時序穩健表示。② 核心 trick 為 DRAIN 時間域泛化機制 + 維度損壞（Dimension Corruption）生成 SSL 視圖 + 三元組損失約束嵌入空間。③ 對「端到端表征」軸★：跳脫固定窗口歷史回報的線性假設，以動態參數流適配分佈漂移。④ 關鍵實證數字：各任務提升幅度均「未披露」，僅定性描述為「更優/最低回撤/更低跟蹤誤差」。

**X-Ray.** 在五軸 Pareto 中，SimStock 放棄了高頻微結構的即時性，押注日頻波段的「結構相似性遷移」。它解了傳統 Rolling Correlation 的兩個工程坑：一是靜態窗口無法捕捉 regime shift，二是行業分類標籤的滯後與模糊。然而，其自監督預訓練缺乏明確的下游 Alpha 對齊目標（如直接優化 Sharpe 或 IR），表征空間的金融可解釋性仍屬黑盒。對量化讀者而言，此框架不直接產出交易信號，而是提供一組「動態協方差先驗」與「配對候選池」；若無法解決訓練-推理參數漂移的實時同步問題，其在實盤中的 envelope 將受限於數據延遲與 LSTM 狀態維護的算力開銷。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 傳統方法 (Rolling Corr/Industry) | 通用 SSL (TS2VEC等) | SimStock |
|---|---|---|---|
| 分佈適配 | 靜態滾動窗口/固定標籤 | 數據增強視圖，參數凍結 | DRAIN 時間域泛化 + LSTM 動態參數流 |
| 視圖生成 | 原始價格/收益率序列 | 掩碼/裁剪/縮放 | 維度損壞 (Dimension Corruption) + 時間轉換模塊 |
| 優化目標 | 線性相關性/DTW距離 | 對比損失/重建損失 | 三元組損失 (Triplet Loss) 約束嵌入流形 |

**1.2 ⚡ Eureka** 「用 LSTM 預測模型權重的時間演化軌跡，而非直接預測價格。」直覺：將參數空間本身視為一個時間序列，讓模型學會「如何隨時間改變自己」，從而隱式編碼市場 regime 的漂移。

**1.3 信息流 ASCII**
[Price + Industry Meta] -> [Time Transform (MA/Window)] -> [Tokenization]
      |
      v
[Dimension Corruption] -> [Self-Attention Aggregator] -> [Embedding Space]
      |
      v
[LSTM State] -> [Dynamic Parameter Update (DRAIN)] -> [Triplet Loss]
      |
      v
[Target Domain Inference] -> [Similarity Matrix / Pairs / Covariance]

## §2 · 數學層
📌 **Napkin Formula**:
$\theta_{t+1} = \text{LSTM}(\theta_t, x_t)$  (參數動態更新)
$\mathcal{L}_{triplet} = \max(0, d(f_\theta(a), f_\theta(p)) - d(f_\theta(a), f_\theta(n)) + \alpha)$
複雜度: $O(T \cdot N \cdot d^2)$ 隨時間步 $T$、股票數 $N$、嵌入維度 $d$ 線性/二次增長。
直覺: 損失函數不直接優化收益率，而是強制相似股票（正樣本）在嵌入空間收斂，不相似股票（負樣本）推開。LSTM 負責將歷史訓練動態壓縮為當前時刻的權重初始化，實現跨時間域的參數遷移。
Loss/訓練: 自監督預訓練為主，無明確下游監督信號；採用三元組採樣策略，訓練時分時間域批次輸入，推理時依賴 LSTM 狀態載入目標域初始參數。

## §3 · 數據層
規模/頻率: 5大主要交易所日頻數據，2018-2023。模態為量價特徵 + 靜態行業元數據。
來源/處理: 未披露具體數據供應商與清洗流程。時間轉換模塊使用移動平均生成多週期變體。
樣本外/容量: 假設訓練域與目標域存在分佈漂移但底層結構相似。未驗證跨資產類別（如股轉債/商品）的泛化能力，容量假設依賴日頻流動性，未計入滑點與衝擊成本。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | TBD |
| Checkpoint | 未披露 |
| License | 未披露 |
| 複現難度 | 中高（需自構建時間域劃分邏輯與 LSTM 參數更新器） |
| 數據可得性 | 中（需標準日頻 OHLCV + 行業分類，清洗與對齊工作量 TBD） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 5大交易所 (2018-2023) | 相似股檢索 (Corr/DTW) | 未披露 | 未披露 | 未披露 |
| 配對交易 | 最終財富 / Max Drawdown | 未披露 | 未披露 | 未披露 |
| 主題ETF跟蹤 | 跟蹤誤差 / 波動率 | 未披露 | 未披露 | 未披露 |
| 組合優化 (MVO) | 風險調整回報 | 未披露 | 未披露 | 未披露 |
解讀: 所有 Δ 均為「未披露」。定性優勢可能來自 SSL 對非線性結構的捕捉，但缺乏交易成本調整與嚴格 OOS 劃分（如時間序列交叉驗證）的細節。跟蹤誤差下降可能源於行業元數據的隱式先驗，而非純量價表征的真實能力提升。需警惕樣本內過擬合與前瞻偏差（LSTM 狀態若未嚴格隔離訓練/推理時間戳）。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**: 缺乏嵌入空間時間穩定性的理論分析；未納入文本/新聞數據；參數估計的貝葉斯/收縮先驗未深度融合。
**6.2 推斷的隱含假設**:
- Regime 依賴: 假設歷史漂移模式可平滑外推至未來，未處理突發結構性斷裂（如政策急轉/流動性枯竭）。
- 容量/成本: 日頻波段策略容量未量化，未計入雙邊交易成本與滑點，配對交易 Z-score 信號可能在高波動期失效。
- 數據泄漏: 時間域劃分若未嚴格隔離，LSTM 狀態更新可能隱含未來信息；行業標籤更新滯後可能導致標籤泄漏。
- Survivorship: 未明確說明是否包含已退市股票，若僅用現存股票訓練，存在顯著倖存者偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| TS2VEC / TST | 參數凍結 vs 動態參數流 (DRAIN) | Open | 成熟基線 |
| 傳統 Rolling Cov / MCD | 線性/統計假設 vs 端到端非線性表征 | Open | 工業標準 |
| 圖神經網絡 (GNN) 股票網絡 | 靜態圖結構 vs 時間域動態參數演化 | 部分 | 研究熱點 |
🎤 **Interview Tip**
- 正確答: 「SimStock 的核心不在於預測收益率，而在於用 LSTM 動態更新模型權重來適配時間分佈漂移，其 SSL 表征主要用於構建動態相似性矩陣與協方差先驗。實盤落地需解決 LSTM 狀態的實時同步與交易成本摩擦。」
- 錯答: 「它用自監督學習直接預測未來股價漲跌，比 LSTM 預測價格更準確。」（混淆了表征學習與監督回歸目標）
**7.1 可證偽預測**: 若 2025-Q4 前未公開完整 OOS 回測代碼與成本調整後的 Sharpe/IR 數據，該框架在機構實盤中的採用率將低於 5%（僅限研究參考）。

## §8 · For the Reader
- **因子研究員**: 將 SimStock 嵌入視為「動態行業中性化因子」的替代方案，測試其與傳統動量/價值因子的正交性，避免與現有因子庫共線。
- **組合配置/風控**: 利用其輸出的動態相似性矩陣替換歷史協方差，進行風險預算分配與壓力測試，觀察極端行情下的矩陣條件數變化與逆矩陣穩定性。
- **高頻/執行**: 不適用。日頻表征無法捕捉盤中流動性微結構，強行降頻使用會引入信號延遲與滑點損耗，執行層應關注信號觸發頻率與訂單簿深度匹配。
- **研究學生**: 重點複現 DRAIN 參數更新邏輯與維度損壞視圖生成，驗證 LSTM 狀態初始化對下游配對交易勝率的邊際貢獻，並嘗試加入交易成本約束進行消融實驗。

## References
- SimStock: Temporal Representation Learning of Stock Similarity for Investment Management (2024)
- Lineage: DRAIN (Temporal Domain Generalization) · TS2VEC · Triplet Loss SSL
- QuantML 導讀: [股票相似性的时间表示学习及其在投资管理中的应用](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247485992&idx=1&sn=9c13d3832825cf98c8b1eb593acf6ce9&chksm=ce7e6d36f909e4203370742f24372136d5bec9d84ccf2044028a2cab8cd7ac3f3a8e7f6fe0aa#rd)