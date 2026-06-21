<!-- ontology-5axis data=量价表格 horizon=跨周期 paradigm=监督回归 alpha=组合执行优化 autonomy=全自动黑盒 -->

# DeepUnifiedMom 解構（DeepUnifiedMom）

> **發布**：2024-06-14 · （無 venue） · arXiv [2406.08742](https://arxiv.org/abs/2406.08742)
> **QuantML 導讀**：[DeepUnifiedMom: 多任务学习及MoE统一时序动量资产组合构建](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247484736&idx=1&sn=3cde94d9b3c0644d8114bcd5092d0cf8&chksm=ce7e625ef909eb489cca0f1906a260bd34232218da58d78900852cb5cf1fdcb1c49ca2c4e78e#rd)
> **核心定位**：落點於「監督回歸 × 組合執行優化」軸，以 Multi-Task MoE 解構傳統 TSMOM 單週期/等權分配的結構性缺陷，將動量訊號生成與資本配置統一於端到端訓練迴圈。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 將快/中/慢速動量訊號預測與組合權重分配統一為多任務學習架構；② 核心 trick 為 Multi-gate MoE 分離週期依賴特徵，並引入 soft-capped Sharpe 損失直接優化資本分配網絡（CAN）；③ 對「組合執行優化」軸★，跳過傳統兩階段（先預測後優化）的誤差累積，以風險調整回報為錨點進行端到端梯度更新；④ 關鍵實證數字：測試期 Sharpe 2.33 / Sortino 3.88 / MDD -1.02%（未披露交易成本與滑點調整）。

**X-Ray.** 本框架將動量策略的「訊號生成」與「權重配置」從解耦的兩階段工程，壓縮為單一的端到端監督回歸迴圈。在五軸 Pareto 上，它明確放棄了高頻/微結構建模，專注於日頻跨週期（1/3/6月）的趨勢延續性，並用 MoE 的門控機制替代傳統等權或靜態風險平價的硬編碼規則。其工程價值在於：透過 soft-capped Sharpe 損失直接對齊組合層目標，有效抑制了金融時間序列中高噪聲特徵導致的過擬合，這正是傳統兩階段動量模型常見的失效模式。然而，該架構的 envelope 仍受限於日頻期貨連續合約的流動性假設，且 CAN 的 tanh/softmax 輸出未明確處理做空限制與保證金追繳的離散跳躍。對量化讀者而言，此方法不適合直接搬磚至股票截面，但「以組合層風險指標為訓練損失」的范式，為因子組合優化提供了一條可證偽的端到端路徑。

## §1 · 架構 / Core Mechanism
### 1.1 三大改動 vs 前作
| 維度 | 傳統 TSMOM / 兩階段 ML | DeepUnifiedMom | 工程意涵 |
|---|---|---|---|
| 訊號生成 | 單週期獨立計算或靜態等權 | Multi-Task LSTM + MoE 分離快/中/慢速 | 參數共享降低樣本需求，門控動態分配週期依賴 |
| 資本配置 | 事後 MVO / Risk Parity / 等權 | 端到端 CAN 網絡直接輸出權重 | 消除兩階段誤差累積，權重與預測聯合優化 |
| 訓練目標 | 預測值 RMSE / 分類交叉熵 | Soft-capped Sharpe Loss + 任務 RMSE | 損失函數直接對齊組合層風險調整回報，抑制極端值過擬合 |

### 1.2 ⚡ Eureka 一句話 trick + 直覺
用 soft-capping 對數變換截斷 Sharpe 的極端梯度，讓 CAN 在訓練時「不敢」過度集中於單一高波動資產，直覺上等同於在損失函數內嵌了一個動態風險懲罰項。

### 1.3 信息流 ASCII 圖
```
[Input: Vol-Adj Log Returns (3d~252d)]
          ↓
[Shared LSTM Backbone] → 參數共享提取跨資產趨勢特徵
          ↓
[Multi-Gate MoE] → Fast(1M) / Medium(3M) / Slow(6M) 門控路由
          ↓
[Task-Specific FNNs] → 預測前瞻性 TSMOM 訊號 (RMSE Loss)
          ↓
[Capital Allocation Network (CAN)] → tanh/softmax 輸出組合權重
          ↓
[Soft-Capped Sharpe Loss] → 端到端梯度回傳更新 LSTM/MoE/CAN
```

## §2 · 數學層
📌 **Napkin Formula:**
`𝓛_total = 𝓛_RMSE + λ · 𝓛_Sharpe_soft`
`𝓛_Sharpe_soft = log(1 + Sharpe) if Sharpe > τ else Sharpe  （τ 為截斷閾值，TBD）`
複雜度：O(T · N · D) 前向傳播，MoE 門控引入 O(N_experts) 稀疏路由，整體訓練成本與標準 MTL 同量級。
直覺：總損失將「訊號預測準確度」與「組合風險調整回報」綁定。Soft-capping 本質是對 Sharpe 分佈的右尾進行對數壓縮，防止回測期內的極端行情產生爆炸梯度，迫使模型學習更平滑的權重遷移。
訓練細節：SGD + Adam 混合優化（未披露具體切換條件），Grid Search 超參，Early Stopping (patience=5 epochs)。

## §3 · 數據層
- **資料規模/頻率/市場/時段**：Pinnacle Data Corp CLC 資料庫，49 種期貨合約，日頻（Daily），1990-01 至 2023-12。
- **特徵工程**：過去 3, 5, 10, 21, 63, 126, 252 日對數收益率，經資產波動率標準化（Vol-Adj）。
- **樣本外與容量假設**：Expanding Window CV（首輪訓練 10 年，20% 驗證，每年滾動測試 1 年）。假設日頻期貨流動性充足，未披露最小交易單位/保證金比例/隔夜跳空風險處理。容量推測為中低頻機構級，但依賴連續合約拼接（Backward Ratio），需警惕換月滑點與展期收益（Roll Yield）的潛在洩漏。

## §4 · 代碼層
| 維度 | 狀態 |
|---|---|
| Repo / Checkpoint | TBD（QuantML 導讀提及「論文下載見星球/加入星球請掃我」，未開源） |
| License | TBD |
| 複現難度 | 中等（架構標準 MTL+MoE，但 Soft-Capped Sharpe 的 τ 閾值與 λ 權重需自行 Grid Search） |
| 數據可得性 | 高（Pinnacle CLC 為商業期貨數據，國內可替換為 CTA 指數或 Wind/聚寬期貨連續合約） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 49 Futures (1990-2023) | Sharpe Ratio | 未披露 | 2.33 | 未披露 |
| 49 Futures (1990-2023) | Sortino Ratio | 未披露 | 3.88 | 未披露 |
| 49 Futures (1990-2023) | Max Drawdown | 未披露 | -1.02% | 未披露 |
| 49 Futures (1990-2023) | Annual Return | 未披露 | 未披露 | 未披露 |

**解讀**：Δ 的顯著性主要來自 CAN 的端到端優化與 Soft-Capping 的穩健性。但 -1.02% MDD 在 34 年回測中極度異常，強烈暗示：① 未計入交易成本/滑點/保證金追繳；② 可能使用了未來函數（如換月點精確對齊或波動率標準化使用了全樣本統計量）；③ 期貨連續合約拼接方式可能平滑了跳空風險。真 capability 在於風險調整回報的穩定性，而非絕對收益。

## §6 · 失效與隱含假設
### 6.1 論文自述 limitations
- 門控機制尚未引入稀疏性約束（Sparsity），可能導致計算冗餘。
- 缺乏可解釋 AI（XAI）模塊，黑盒權重分配難以通過合規審計。
- 未來計劃引入 Transformer 架構，暗示當前 LSTM 對超長週期依賴捕捉有限。

### 6.2 推斷的隱含假設
- **Regime 依賴**：模型高度依賴趨勢延續環境（Trend-following friendly），在震盪市（Range-bound）或波動率 regime shift 時，CAN 可能無法快速降槓桿。
- **容量與成本**：日頻調倉 49 個期貨合約，假設滑點與手續費可忽略。實盤中展期成本（Roll Cost）與隔夜跳空將直接侵蝕 Sharpe。
- **數據泄漏風險**：波動率標準化若使用滾動視窗外數據或全樣本均值，將導致前瞻偏差。Early Stopping (patience=5) 在金融非平穩分佈下易觸發過早收斂。
- **Survivorship**：Pinnacle CLC 雖涵蓋已退市合約，但連續合約拼接若未嚴格處理交割規則，可能隱含幸存者偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 TSMOM (Moskowitz 2012) | 靜態等權/單週期 vs 動態 MoE 多週期 | 是 | 行業基準 |
| 兩階段 ML 動量 (LSTM/GRU + MVO) | 預測與優化解耦 vs 端到端 CAN 聯合優化 | 視項目 | 常見工程實踐 |
| RL 組合管理 (PPO/SAC) | 離散動作/獎勵延遲 vs 監督回歸/直接 Sharpe 損失 | 是 | 高算力/低穩定性 |

🎤 **Interview Tip**
- ✅ **正確答**：「DeepUnifiedMom 的核心創新不在於 MoE 本身，而在於將組合層的 Sharpe 比率透過 soft-capping 轉化為可微損失，直接回傳梯度更新訊號網絡。這解決了兩階段動量模型中『預測準確但組合風險失控』的錯配問題。實盤需重點驗證展期成本與波動率 regime shift 下的權重衰減。」
- ❌ **錯答**：「它用 Transformer 取代了 LSTM，所以預測更準。」（論文明確指出 LSTM 為 backbone，Transformer 僅為未來工作；且核心是損失函數設計而非架構替換。）

### 7.1 可證偽預測帶日期
- 若 2025-06-30 前，該架構在含真實滑點/保證金規則的日頻 CTA 實盤中，Sharpe 未能維持 >1.2，則證明 Soft-Capped Sharpe 損失在離散跳空與流動性枯竭環境下存在泛化邊界。

## §8 · For the Reader
- **因子研究員**：將 Vol-Adj Log Returns 替換為自研多頻率因子，測試 MoE 門控是否能自動學習因子權重衰減曲線，避免人工設定 decay 參數。
- **高頻執行**：此框架不適用。日頻調倉與期貨連續合約假設無法承載 tick-level 微結構噪聲，需轉向 RL 或訂單簿深度模型。
- **組合配置/CTA 經理**：關注 CAN 的權重輸出穩定性。可將其作為「動態風險預算模塊」嵌入現有系統，但必須在損失函數外疊加硬約束（如單資產權重上限、總槓桿限制）以滿足風控合規。
- **研究學生**：複現重點不在於調參，而在於實現 Soft-Capped Sharpe 的梯度計算與驗證 τ 閾值對過擬合的抑制邊界。建議先用合成數據（Synthetic Trend/Noise）驗證損失函數的凸性。

## References
- [1] DeepUnifiedMom: Unified Time-series Momentum Portfolio Construction via Multi-Task Learning with Multi-Gate Mixture of Experts. arXiv:2406.08742, 2024.
- [2] Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. Journal of Financial Economics.
- [3] QuantML 導讀：DeepUnifiedMom: 多任务学习及MoE统一时序动量资产组合构建. [鏈接](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247484736&idx=1&sn=3cde94d9b3c0644d8114bcd5092d0cf8&chksm=ce7e625ef909eb489cca0f1906a260bd34232218da58d78900852cb5cf1fdcb1c49ca2c4e78e#rd)