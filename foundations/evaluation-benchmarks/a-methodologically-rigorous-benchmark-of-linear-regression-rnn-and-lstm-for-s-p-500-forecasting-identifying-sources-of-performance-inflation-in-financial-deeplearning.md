<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# A Methodologically Rigor 解構（A Methodologically Rigor）

> **來源發布**：2026-11-20 ·（無 venue）· arXiv [2611.27978](https://arxiv.org/abs/2611.27978)
> **arXiv 原文**：[A Methodologically Rigorous Benchmark of Linear Regression, RNN, and LSTM for S&P 500 Forecasting: Identifying Sources of Performance Inflation in Financial Deep Learning](https://arxiv.org/abs/2611.27978v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點在「量價表格 → 日頻水平預測 → 監督回歸 → 端到端黑盒」，解決的不是模型能力，而是 deep-learning-for-stock-prediction 文獻的評估污染：以 naive persistence 為最低門檻、N=30 次重複分布報告、不做小波去噪，重新對線性、RNN、LSTM 進行 like-for-like 審計。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①在統一、防洩漏的時序管線下，把 OLS/Ridge/Lasso/Elastic Net/Bayesian Ridge、vanilla RNN、LSTM 對 S&P 500 次日收盤價做 like-for-like 重評；②核心 trick 是「先打 naive persistence 基線、再報 N=30 次重複的均值與標準差」而非 best-of-N，並指出小波去噪評估目標、缺少 naive 基線、best-of-N 這三種報告習慣是文獻 RMSE 虛高的主要來源；③這對「監督回歸 + 全自動黑盒」這條軸特別致命，因為黑盒模型的擬合能力可以輕易把水平序列的持續性成分擬合出漂亮的 R，卻打不贏 53.16 指數點的 naive persistence；④表 5 中最佳多層 RNN `[100,50,20]` 的 30 次平均 RMSE 為 **53.97（標準差 0.31）**，仍比 naive 的 **53.16** 差 **0.81**，且整篇 24 個架構 × 30 次重複（共 720 個訓練模型）無一配置的平均 RMSE 低於 naive 基線（§7.1、§10）。

**X-Ray.** 這篇論文的價值不在「新模型」而在「新協議」。在五軸 Pareto 上，它把「端到端黑盒 + 監督回歸」推到最誠實的位置：**回歸目標是水平（level）而非報酬，判別指標 R 在 near-random-walk 上必然趨近 0.99（§6.1），所以真正的能力鑑別只能落在 RMSE/MAPE 與 naive 基線的差距上**。它解決了三個舊工程坑：小波去噪雙重應用於 train/test 造成 RMSE 虛降（§5.1）、缺乏 naive persistence 造成無參照比較（§5.2）、best-of-N 忽略 run-to-run variance（§5.3）。但它明確打不開的 envelope 也清楚：（a）單一 80/20 時序切分，不是 walk-forward（§10 未來工作自述）；（b）只預測指數水平，不預測收益/波動/方向，因此即便 RMSE 小幅改善也不能直接轉成交易 alpha；（c）未把交易成本、容量、市場衝擊納入決策函數（§1 僅在引言提及這些概念，正文沒有成本模型）；（d）只涵蓋 S&P 500 指數、日頻，無法外推至個股、ETF、國際指數或日內頻率。對量化讀者的意義：**任何沒有 naive persistence 對照、沒有分布報告、或對評估目標做過平滑的「深度學習打敗基準」結果，都應先被視為評估偽陽性，直到通過本文的三重協議**。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 前作常見做法（本文審計對象） | 本文改動 | 目的 / 效果 |
|---|---|---|
| 小波去噪同時作用於訓練目標與測試目標（§5.1，引 Bao et al. 2017） | 完全不對 train/test 目標做小波去噪；所有 RMSE 對原始收盤價計算 | 避免「RMSE 量測的是對平滑參考序列的擬合」；近隨機漫步目標上此做法可讓 RMSE 虛降 10–20 指數點（§5.1 原文） |
| 只把候選模型互相比（如 LSTM vs 少特徵線性回歸），從不引入 naive persistence（§5.2） | 所有模型都對 naive persistence `ŷ_{t+1}=y_t`（式 1）比較，並作為最低門檻 | 隔離「模型學到超過水平持續性的結構」；避免把擬合水平成分誤認為預測力 |
| 只報 N 次重複中最佳（最低 RMSE）結果（§5.3，引 Bailey et al. 2017；Hewamalage et al. 2023） | 報 N=30 次重複的 mean 與 std（另附最小值僅供完整性） | 避免把 run-to-run variance 誤認為模型能力；本文自述 best-of-N 可低估期望表現 1–3 指數點（§5.3） |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

**Eureka：** 先讓模型證明自己能打贏 `ŷ_{t+1}=y_t`，再談架構深淺；而且要比的是 30 次重複的均值，不是 N-best。

直覺：日頻指數收盤價的水平序列近似隨機漫步，任何模型只要跟得上「昨天的收盤價」，R 都會自然逼近 0.99；所以真正的判別點是 **RMSE 能否低於 naive**。這件事在文獻裡很容易被三種報告習慣掩蓋：去噪的評估目標把 RMSE 壓低、缺少 naive 基線讓人誤以為模型比同類好、best-of-N 把長尾變異誤讀為能力。本文把三者一起校正後，得到「沒有模型能贏 naive」的結論。

### 1.3 信息流 ASCII 圖

```
Yahoo Finance OHLC ─┐
FRED 宏觀 5 序列   ─┼─► 時間順序合併、僅用 train 統計量做 StandardScaler
MACD/ATR/RSI 技術  ─┘        │
                             ▼
                   滑動窗口 L=5，目標 y_{t+1}
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
   線性五兄弟（OLS/Ridge/           RNN / LSTM 單層・多層
   Lasso/ElasticNet/BayesianRidge）   （30 次重複、網格搜參）
              │                              │
              └──────────────┬──────────────┘
                             ▼
               對原始收盤價計算 RMSE / MAPE / R
                             │
                             ▼
               與 Naive Persistence 53.16 指數點比
                             │
                             ▼
      均值仍高於 53.16 → 不採信為預測能力；DM 檢驗 p>0.05
```

## §2 · 數學層

📌 **Napkin Formula**

- naive persistence（式 1）：`ŷ_{t+1}^{naive} = y_t`
- 評價主指標（式 19）：`RMSE = sqrt( (1/n) Σ_i (y_i − ŷ_i)^2 )`
- DM 檢驗（式 22）：`d_t = L(y_t − ŷ_t^{(1)}) − L(y_t − ŷ_t^{(2)})`，H0：`E[d_t] = 0`，小樣本校正見 Harvey et al. (1997)，本文 `T=966`、`h=1`（§8.2）

**直覺：** 若 naive 已經能達到很低的 RMSE，任何模型都必須在這個最低門檻之上解釋額外結構；否則 RMSE 的「改善」只是評估方式造成的假象。DM 檢驗再補一層推論保障：即便模型 RMSE 稍低，若損失差分的均值不顯著異於 0，也不能宣稱有預測價值。

**Loss / 訓練細節：** 神經模型以 MSE 為目標（式 18，`L(θ) = (1/N) Σ_i (y_i − ŷ_i)^2`），BPTT 求梯度；超參數用網格搜尋在驗證集上選擇，§6.3 稱網格為 3 個優化器（Adam、Nadam、AdamW）× 3 個學習率（`10^{-1}`、`10^{-2}`、`10^{-3}`）× 3 個 batch size（4、8、16），並以驗證損失早停、最多 100 epochs。**待核（來源內部矛盾）：** 表 2／4／5 有多個配置實際選中 RMSprop（LSTM 150／200、單層 RNN 10、多層 RNN `[10,5]`／`[20,10]`／`[50,20]`），與 §6.3 所述三優化器網格不符；實際搜索空間是否含 RMSprop 需以原文程式碼確認，不宜視為可直接複現的網格。所讀文本未見模型參數量與訓練 FLOPs；圖表／附件待核。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為**明確標示的假設/示意**玩具數字，不是論文實證結果；論文真實數字一律見 §5。

**假設：** 測試集只有 3 天，真實收盤價 `y = [100, 102, 101]`；naive 預測 `ŷ_{t+1}=y_t` 給出 `[100, 102]`（用前一日預測次日）；假想模型預測 `[99, 104]`。

**步驟 1：** 計算每點誤差。
- naive：`y − ŷ = [2, −1]`
- 模型：`[3, −3]`

**步驟 2：** 平方誤差。
- naive：`[4, 1]`
- 模型：`[9, 9]`

**步驟 3：** RMSE。
- naive：`√((4+1)/2) = √2.5 ≈ 1.581`
- 模型：`√((9+9)/2) = √9 = 3.000`

**步驟 4：** 判讀。模型 RMSE 比 naive 大，所以在此示意中模型沒有預測優勢。若作者只報 best-of-N、並且只評估對去噪目標的擬合，這兩個有偏選擇**可能**翻轉結論（§5.1–§5.3 的機制），但任何翻轉都應被視為方法論偽陽性。

**步驟 5：** DM 檢驗的損失差分。`d_t = L(模型) − L(naive) = [9−4, 9−1] = [5, 8]`，均值 `6.5 > 0`，與「模型累積更多平方誤差」一致（對應 §8.2 的正向 `\bar{d}` 判讀方向）。

## §3 · 數據層

- **樣本與頻率：** 2006 年 1 月至 2025 年 7 月，日頻（§3、§3.5）；包含 Global Financial Crisis（2007–2009）與 COVID-19 市場崩盤（2020）（§3、Figure 1 待核）。
- **來源：** S&P 500 的 OHLC 來自 Yahoo Finance；5 個宏觀指標來自 FRED 與 Yahoo Finance：VIX、USDX、EFFR、UNRATE、UMCSENT（§3.1–3.2）；技術指標 MACD、ATR、RSI 自行計算（§3.3）。
- **特徵與目標：** 最終 9 個特徵為 `Close`（作為前一日值）、VIX、USDX、EFFR、UNRATE、UMCSENT、MACD、ATR、RSI；目標是 `t+1` 收盤價（§3.4）。Open/High/Low 因與 Close 相關係數超過 0.99 而被丟棄（§3.1）。
- **切分與樣本外：** 80/20 時序切分，訓練約 3,840 天（2006 年 1 月 – 約 2021 年 9 月），測試約 966 天（2021 年 9 月 – 2025 年 7 月）；驗證集取訓練尾端 10%（§3.5）。StandardScaler 的 mean/std 僅用訓練集估計（§3.5）。
- **窗口：** 神經模型 L=5 的滑動窗口（§4.3.3），與 Bhandari et al. (2022) 一致。
- **容量假設：** 指數級日頻資料規模本身不構成容量瓶頸；本文只做水平預測，未建模交易成本、市場衝擊或容量（§1 提及成本概念，實驗未見成本模型），因此容量／成本維度在此不可推論（本頁推斷）。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露（所讀文本未見代碼連結） |
| Checkpoint | 未披露（所讀文本未見權重發布） |
| License | 未披露 |
| 複現難度 | 中高：資料可從 Yahoo Finance + FRED 取得；特徵工程、滑動窗口、時序切分在 §3–§4 描述可復現；但程式碼未發布、超參數搜索與 N=30 的完整配置需自行重建 |
| 數據可得性 | 公開（Yahoo Finance OHLC + FRED 5 宏觀序列）；頻率對齊採前向填充（§3.2） |

## §5 · 評測 / Benchmark

> **欄名說明：** 本文語境的「前SOTA」欄實為 **Naive Persistence 基線**（`ŷ_{t+1}=y_t`，§4.1），不是任何已發表模型的 SOTA；因此不應解讀為「打敗前最佳」。
> **Δ 定義：** 依本頁規則，`Δ = 本方法 − 前SOTA`；與原文表 6 的 `Δ = Naive − Model` 符號方向相反，本文逐一換算並在正負號上標示。

| 數據集/市場 | Metric | 前SOTA（Naive Persistence 基線） | 本方法 | Δ（本方法−基線） |
|---|---|---|---|---|
| S&P 500 次日收盤（測試集約 966 天） | RMSE（指數點） | 53.16 | 54.63（Lasso，最佳線性） | +1.47 |
| S&P 500 次日收盤 | RMSE（指數點） | 53.16 | 54.06（單層 LSTM，30 neurons） | +0.90 |
| S&P 500 次日收盤 | RMSE（指數點） | 53.16 | 54.41（多層 LSTM，`[150,100]`） | +1.25 |
| S&P 500 次日收盤 | RMSE（指數點） | 53.16 | 56.27（單層 RNN，50 neurons） | +3.11 |
| S&P 500 次日收盤 | RMSE（指數點） | 53.16 | 53.97（多層 RNN，`[100,50,20]`） | +0.81 |

**表外關鍵數字（逐字）：** 最佳單層 LSTM（30 neurons）的 `R = 0.9972`，MAPE 欄均值 0.843、標準差 0.010（欄名單位 %）；最佳多層 RNN（`[100,50,20]`）的 `R = 0.9972`，MAPE 欄均值 0.837、標準差 0.004（欄名單位 %）；naive 的 `R = 0.9973`，MAPE 欄數值 0.817（欄名單位 %）（表 1、表 2、表 5）。這些 R 都在 0.99 以上、MAPE 欄數值見表；**說明 R 幾乎沒有判別力**（§6.1 自述 R 常 >0.99）；真正有鑑別力的是 RMSE 相對 naive 的差距。

**解讀：**

- **全部 Δ 為正，意即無模型在均值上勝過 naive。** 這不是「模型變壞」，而是「在日頻水平預測上，水平持續性本身非常難打敗」。本文據此主張：**deep-learning 在這一任務上的先前大幅改善很可能是評估協議造成的**（§5.4、§7.1）。
- **Linearity vs 非線性差異極小**：五個線性規格 RMSE 全落在 54.63–55.19，極差 0.56（§6.2）；單層 LSTM 最佳與最差分別為 54.06 與 56.81（表 2），說明主要限制不是估計變異，而是模型形式與近隨機漫步水平序列的結構不匹配。
- **深度效果依家族而異**：在 LSTM 家族內深度並非單調有益——多層 LSTM `[150,100]` 的 54.41 反而略差於單層 30 neurons 的 54.06（§6.4）；但在 RNN 家族內，原文 §6.6 稱深度帶來「small but consistent」的改善，多層 RNN `[100,50,20]` 的 53.97（標準差 0.31）是全部 neural 配置中最接近 naive 的，但差距仍有 0.81（§6.6、§7.2）。**這 0.81 是描述性差距，不是 capability 的證據**；本文自述「不足以建議投入生產」（§7.2）。
- **穩定性與能力必須分開看**：單層 RNN 150 neurons 的 `σ = 7.06`（§6.5）遠大於其他配置，說明在某些超參組合下訓練本身不穩定，因此「最佳 N 次」在這種配置下最容易被誤讀為能力。
- **集成沒有改變結論**：LSTM 30-neuron 集成 53.90、RNN `[100,50,20]` 集成 53.71（表 7），DM 檢定（HLN 小樣本校正）在 `T=966`、`α=0.05` 下皆不拒絕 H0（p 分別 `0.178`、`0.250`）；也就是說集成的 RMSE 仍高於 naive，且無法在統計上區分優劣（§8.2）。
- **過擬合／前視偏差風險已被作者主動審計**：本文明確不做小波去噪評估、超參數只在驗證切片選擇、30 次重複分布報告；因此剩下的前視風險主要是**宏觀序列前向填充可能引入 vintage 偏差**（§3.2 僅描述 forward-fill，未討論 FRED 數據修訂／vintage），以及**單一 80/20 切分無法檢驗跨 regime 穩定性**（§10 未來工作自述）。**成本未計**：本文全程以水平預測衡量，不涉及交易成本與市場衝擊（與 §1 引 Fleming et al. 2003 的論述不一致：引言提成本，實驗未建模成本，屬本文邊界）。
- **文本內部矛盾（待核）：** §6.6 第二段寫「three-layer `[100,50,20]` 配置在我們協議下產生平均 RMSE 為 `[54.62`（標準差 `0.64]`）」，但同節第一段與表 5 都寫 `53.97`（標準差 `0.31`）；此處所讀文本自相矛盾，應以表 5 與 §7.1 的 `53.97`（標準差 `0.31`）為準並標 `TBD`。另 §10 未來工作列「DM 檢驗」為未來工作，但 §8 已執行 DM 檢驗，屬章節自我指涉不一致，圖表／附件待核。

## §6 · 失效與隱含假設

### 6.1 論文自述的限制（可定位的作者限制）

- **§8.2：** 測試集 `T=966`，統計功效不足以偵測模型與 naive 的差異；未拒絕 H0 不等於兩者等效，而是「在此樣本下無法證明差異」。
- **§10：** 負面結論「specific to one-day-ahead S&P 500 closing levels with a particular feature set」；未來需在個股、sector ETF、國際指數、其他資產類別上複製。
- **§6.5：** 單層 RNN 150 neurons 的 `σ = 7.06` 屬訓練不穩定；作者自述「用更細的學習率網格重新調參可能更穩定，但不改變主要結論」。
- **§8.3：** 集成增益（LSTM 0.16 點、RNN 0.26 點）小於文獻常見的 1–3 點，作者歸因於「個體重複已收斂到相近局部最優」與「近隨機漫步目標上方差下界有限」。
- **§10 Future Work 列舉的邊界：** 更高頻目標（intraday/minute-level）、替代預測目標（log return/volatility/direction）、walk-forward 評估、attention/TFT/CNN-LSTM、更廣泛的資產類別複製。**這些是作者自述尚未打開的 envelope，不是本文已證偽的方向。**
- **圖表／附件待核：** Figure 1–13 的具體圖形對應（例如各配置的 replicate 分布、預測 vs 實際）尚未從圖檔核對；文本中 Figure 3–12 的描述以文字摘要為準。

### 6.2 推斷的隱含假設（本頁推斷，非作者原話）

- **Regime 依賴：** 樣本含 GFC 與 COVID，但只有單一 80/20 切分；測試期內（2021-09–2025-07）的制度組成與訓練期不同，模型在訓練期學到的係數未必能遷移到測試期的高通膨／升息環境。**這是本頁推斷，未由 walk-forward 證明**。
- **容量與成本：** 本文不做交易回測，因此 naive 之外的「改進」即使出現，也無法直接對應到可執行策略；容量、市場衝擊、手續費皆未計入。**不可把 RMSE 差異解讀為可實現 alpha**。
- **數據洩漏：** 作者明確採用 train-only scaling 與時序切分；但 FRED 宏觀序列（UNRATE、UMCSENT、EFFR）若使用最終修訂值而非當時 vintage，會有 revision bias；本文 §3.2 只說 forward-fill，未見 vintage 處理描述，**此為本頁推斷的潛在洩漏點，須圖表／附件核對**。
- **Survivorship：** 目標是指數而非個股，構造上不會有個股退市倖存者偏差；但若未來延伸到個股／ETF，須重新檢查。**不可把指數層的結論直接套到個股層**。
- **統計推論邊界：** 本文的 DM 檢定只覆蓋「最好兩族配置的集成」，不是所有 24 個配置的 multiple testing；即使未來出現單一配置 p<0.05，也需多重比較校正。**此為本頁推斷，作者未在 §8 討論多重比較**。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Bhandari et al. (2022) | 相同特徵集（VIX/USDX/EFFR/UNRATE/UMCSENT/MACD/ATR/RSI），聲稱單層 LSTM 在 RMSE/MAPE/R 勝過更深堆疊 | 所讀文本未見代碼 | 本文重評**重現**其家族內深度方向：單層 LSTM（30 neurons，54.06±0.56，MAPE 0.843、R 0.9972）優於最佳多層 LSTM（`[150,100]`，54.41±0.50，MAPE 0.847、R 0.9971）（表 2、表 3）；但此僅限 LSTM 家族內之深度比較，全部 LSTM 配置仍不及 naive 基線 53.16 |
| 經典計量（ARIMA/GARCH 家族） | 經典統計模型，未進入本文 benchmark | 工具箱可用 | 本文未納入；未來工作未列 |
| Temporal Fusion Transformer / CNN-LSTM | 注意力與混合架構 | 所讀文本未見 | 本文未來工作（§10 明列） |
| 隨機漫步/EMH 觀點 | 認為日頻水平預測≈持續性 | 文獻 | 本文結果與之一致，可視為對 EMH 的實證支持 |
| 本頁「前SOTA」語境 | 即 Naive Persistence，不是任何已發表模型 | 無參數、無特徵 | 本文全篇的主基線 |

🎤 **Interview Tip**

- **正確答：** 「這篇不是提出新模型，而是提出評估協議：先用 naive persistence `53.16` 當最低門檻，再用 `N=30` 重複分布報告，並且不對評估目標做小波去噪。結論是 24 個架構共 720 個訓練模型裡沒有任何一個平均 RMSE 低於 naive；最佳多層 RNN `53.97`（標準差 `0.31`）仍差 `0.81`，DM 檢定對集成也不顯著（p 分別 `0.178`、`0.250`）。這說明在這個任務上，深度模型學到的信號不足以超越水平持續性，且文獻中常見的改善很可能來自評估偽陽性。」
- **錯答：** 「LSTM 比線性回歸好，所以深度學習有用。」或「本文證明深度學習在金融預測上無效。」——兩種都是過度推論。**有限實驗的失敗不等於普遍不可能**；本文的負面結果僅限 S&P 500 次日收盤 + 特定 9 特徵集 + 單一時序切分，尚未測試的更高頻、報酬/波動目標、走步前推、注意力架構都仍是 open。

### 7.1 可證偽預測帶日期

1. **到 2027-12-31 為止，** 若後續研究在相同 2006–2025 S&P 500 日頻資料上改用 walk-forward（多個滾動測試窗），並宣稱某神經模型平均 RMSE < naive persistence，需同時報告 30 次重複的 mean 與 std 及 DM 顯著性；**若無這兩項，本頁預測該「改善」會被證明為評估偽陽性**（可證偽形式：獨立複製若重現平均 RMSE < 53.16，則本文結論被限縮至「單一 80/20 切分下不成立」）。
2. **到 2027-12-31 為止，** 本文的單層 RNN 150 neurons 在 `σ = 7.06` 的不穩定性，若改用更細的學習率網格（<0.1）重調後 σ 顯著下降且平均值不變，將支持「不穩定來自學習率選擇而非架構本身」；若 σ 仍高，則支持「架構容量過大時訓練不穩定」——兩者都可在同一資料上檢驗。
3. **到 2027-12-31 為止，** 若把目標改為 log return 或 direction（本文 §10 未來工作），且相同模型族在相同協議下仍打不贏 naive 的「return=0」或「direction=多數類」基線，則本文結論可外推到「日頻 S&P 500 的短期預測上限逼近 naive」；若出現顯著打贏，則本文結論限縮於「水平目標」。

## §8 · For the Reader

- **因子研究員：** 把 naive persistence 寫進每一個短週期回歸的 standard baseline，並且用 mean 與 std 取代 N-best；在相關係數 R 大於 0.99 的任務上，**不要只報 R**，因為水平序列會讓任何跟得上水平的模型都獲得高 R（§6.1）。
- **高頻執行：** 本文不覆蓋日內資料；若你的目標是 minute-level 或 order book，日頻結論不可直接搬用（§10 未來工作明列 higher-frequency 為 open）。但「評估協議先於模型」的紀律可直接套用。
- **組合配置：** 把本文當成 sanity check：在任何把 ML 信號放進配置前，先確認信號 RMSE（或 Sharpe、IR）打贏 naive/等權/買持等零參數基線；否則只是把訓練方差包裝成 alpha。
- **LLM-agent：** 本文的三種膨脹來源（去噪目標、缺少 naive 基線、best-of-N）同樣適用於 LLM 生成交易信號的評估；在 prompt/agent 端到端流程中，尤其容易用「最佳幾次 prompt」代替分布，應避免。
- **RL 策略：** 不要把「回測曲線漂亮」當能力；先跑 naive/隨機/買持基線，再談 RL 策略的增量。本文的 RMSE + DM 框架可直接移植到 RL 回測的統計檢定。
- **研究學生：** 這篇論文是「方法論審計」的範本：先定義基線，再審計自己的評估協議；小波去噪、缺失基線、N-best 三個坑是最常見的 reviewer 問題，值得先自審。

## References

- 原論文：Sitaula, H., & Opitz, J. (2026). *A Methodologically Rigorous Benchmark of Linear Regression, RNN, and LSTM for S&P 500 Forecasting: Identifying Sources of Performance Inflation in Financial Deep Learning.* arXiv:2611.27978v1 [cs.LG]. https://arxiv.org/abs/2611.27978v1
- 直接 lineage（本文重評對象）：Bhandari, H. N., Rimal, B., Pokhrel, N. R., Rimal, R., Dahal, K. R., & Khatri, R. K. C. (2022). Predicting stock market index using LSTM. *Machine Learning with Applications*, 9, 100320. https://doi.org/10.1016/j.mlwa.2022.100320
- 方法論 lineage：
  - Bao, W., Yue, J., & Rao, Y. (2017). A deep learning framework for financial time series using stacked autoencoders and long-short term memory. *PLOS ONE*, 12(7), e0180944.（小波去噪評估目標）
  - Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–69.（best-of-N / 回測過擬合）
  - Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation for data scientists: common pitfalls and best practices. *Data Mining and Knowledge Discovery*, 37(2), 788–832.
  - Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.（DM 檢定）
  - Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.（HLN 小樣本校正）
- 來源鏈接：arXiv 摘要頁 https://arxiv.org/abs/2611.27978；本頁依據 arXiv v1 可讀全文文本；Figure 1–13 未核。