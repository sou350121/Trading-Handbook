<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# Momentum Transformer (TFT) 解構（Momentum Transformer (TFT)）

> **發布**：2024-12-18 · （無 venue）
> **QuantML 導讀**：[增强动量策略：动量Transformer模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247488434&idx=1&sn=e8431bb689e27562a620cdb6cf3c6d31&chksm=ce7e74acf909fdba23e88cba2b6157174f0d50d61d3fab4ebf305a80c01d90bd43ff6e4d1a33#rd)
> **核心定位**：將僅解碼器 TFT 架構從低協方差期貨場域硬遷至股票日頻波段，以負 Sharpe Ratio 為損失端到端輸出權重。解了傳統 MACD 動量無法適應 regime shift 與 LSTM 對長週期非平穩性敏感的 prior gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** 將 TFT 遷移至股票日頻交易，以負 Sharpe Ratio 為損失端到端輸出頭寸。核心 trick 為融合 Changepoint Detection 與多頭注意力，並透過擴展時間窗口與增加注意力頭數捕捉長短週期依賴。此設計直接對齊日頻波段軸的動態再平衡需求，以模型內生風險調整取代傳統因子疊加。導讀給出四年期平均年回報率 4.14% 與平均 Sharpe Ratio 1.12，但明確指出未超越同期僅多頭策略的 4.04% 回報。

**X-Ray.** 本法將 TFT 從低協方差期貨/外匯場域硬遷至股票市場，觸及「端到端表徵」與「日頻波段」的 Pareto 邊界。傳統 MACD 動量坑在於靜態參數無法適應 regime shift，而 LSTM 對長週期非平穩性敏感；TFT 以 Changepoint Detection 標記狀態轉換，並用 Multi-Head Attention 分配歷史步長權重，直接以負 Sharpe Ratio 優化頭寸，跳過因子工程與線性組合的次優解。然而，股票高波動與部門內高協方差使 CPD 訊號頻繁觸發且置信度虛高，模型實質退化為波動率抑制器而非趨勢捕捉器。擴展窗口至 378 與增加 Attention Heads 至 6 反而引發過擬合，顯示日頻股票數據的資訊密度不足以支撐更大容量的端到端表徵。對量化讀者而言，此架構驗證了「損失函數設計 > 架構堆疊」的鐵律，但實戰需警惕股票市場的高協方差會稀釋注意力機制的長期依賴學習能力，適用 envelope 限於波動率中低且趨勢明確的個股池。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作 (Wood et al. / LSTM) | 本法 (TFT-Stocks) | 工程意圖 |
|---|---|---|---|
| 時間窗口 | 252 交易日 | 擴展至 378 交易日 | 讓 Attention 覆蓋一年半動態，對抗動量失效期 |
| 注意力機制 | 4 個 Attention Heads | 增加至 6 個 Attention Heads | 捕捉更多元模式，從噪聲中辨別 Changepoint |
| 狀態感知 | 無 / 僅 LSTM 隱狀態 | 引入 Changepoint Detection (CPD) 模塊 | 標記市場 regime shift，補償 LSTM 對長週期非平穩性的盲區 |

⚡ **Eureka:** 以負 Sharpe Ratio 為損失函數直接優化頭寸，讓模型內生學習「風險調整後的動量」而非單純預測收益率。
直覺：傳統動量策略是「預測價格 → 規則轉倉位」的兩階段解耦流程，本法將預測與倉位生成合併為單次前向傳播，損失函數直接懲罰高波動與低回報，迫使 Attention 權重自動聚焦於能穩定貢獻風險調整收益的歷史步長。

```
[Raw Prices/Returns] → [LSTM Encoder] → [Variable Selection Network] → [GLU] → [GRN]
  ↓
[Changepoint Detection (CPD)] ──────────────────────────────────────┘
  ↓
[Multi-Head Attention] → [FFN (tanh)] → [Position Output] → [Loss: -Sharpe Ratio]
```

## §2 · 數學層
📌 **Napkin Formula:**
$\mathcal{L} = -\frac{\mathbb{E}[r_p] - r_f}{\sigma_p}$, 其中 $r_p = \mathbf{w}^\top \mathbf{r}$, $\mathbf{w} = \text{FFN}(\text{Attn}(\mathbf{X}_{1:T}))$
直覺：損失函數直接對齊交易目標，梯度反向傳播時會自動壓制高波動資產的權重 $\mathbf{w}$，無需額外設計風險預算或波動率目標。訓練採用擴展窗口（Expanding Window）：初始 2017-2019 訓練，逐年滑動至 2023 測試，驗證集比例從 10% 調至 20% 以防過擬合。

## §3 · 數據層
頻率為日頻，市場為美國股票（CRSP & Compustat），時段為 2017-2023。資料來源為按 SIC 代碼劃分的 9 大行業部門，每部門取市值排名前 5 的公司，最終構建低協方差投資組合。輸入特徵僅限收盤價與歷史收盤價，經 CPD 腳本處理輸出 `cp_location`, `cp_score` 等狀態變數。樣本外採用逐年擴展窗口測試（2020-2023）。容量假設受限于計算資源：CPD 回溯窗口僅能維持 21 天（原論文為 21 與 126），樣本量從 10 年壓縮至 7 年，且無法將時間窗口進一步擴展至兩年（GPU 記憶體耗盡）。

## §4 · 代碼層
| 維度 | 狀態/細節 |
|---|---|
| Repo | TBD |
| Checkpoint | 未披露 |
| License | 未披露 |
| 複現難度 | 高（需處理 CRSP/Compustat 多版本股票數據洩漏問題，且 CPD 計算耗時長、記憶體需求 >12GB） |
| 數據可得性 | 低（依賴付費數據庫 CRSP/Compustat，且需自行編寫 SIC 篩選與 CPD 腳本） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 基線 (MACD 動量) | 基線 (僅多頭) | 本方法 (Base TFT) | Δ (vs MACD) | Δ (vs 僅多頭) |
|---|---|---|---|---|---|---|
| 美國股票 (2020-2023) | 平均年回報率 | -1.07% | 4.04% | 4.14% | +5.21% | +0.10% |
| 美國股票 (2020-2023) | 平均 Sharpe Ratio | -0.18 | 0.57 | 1.12 | +1.3 | +0.55 |
| 美國股票 (2020-2023) | 平均年波動率 | 未披露 | 未披露 | 9.05% (四年平均) | 未披露 | 未披露 |

*解讀:* 本方法相對 MACD 基線的 Δ 確實來自端到端風險調整優化，但相對「僅多頭」的 Δ 極小（回報僅高 0.10%），顯示模型實質功能為波動率壓縮而非額外 Alpha 生成。導讀明確指出擴展窗口與增加注意力頭數的變體對非 CPD 模型產生負面影響，且帶 CPD 模型受更大窗口拖累，此 Δ 部分可能源自訓練期（2017-2019）的過擬合或樣本內特徵匹配。此外，股票組合波動率（9.05%）遠高於原論文期貨組合（疫情期 1.51%），高波動環境稀釋了 Attention 的長期依賴學習，導致 Sharpe Ratio 提升主要來自波動率分母壓制，而非分子回報的結構性突破。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 股票市場波動性遠高於原論文適用的期貨/外匯市場，導致 CPD 幾乎總是產生高置信度變化點，訊號噪聲過大；缺乏長期 CPD（126 天窗口因算力限制被捨棄）使模型無法捕捉長期趨勢反轉；增加注意力頭數僅略微改善性能，未能超越無 CPD 基礎模型；部門內 5 檔股票仍存高協方差，分散化效果不及預期。
**6.2 推斷的隱含假設:**
*   **Regime 依賴:** 假設市場存在可被 CPD 標記的離散狀態轉換，但股票日頻數據的連續高波動可能使狀態邊界模糊，導致模型在震盪市中頻繁切換頭寸。
*   **容量與成本:** 未計入交易成本與滑點。日頻再平衡 20-30 檔股票，若考慮實盤摩擦，Sharpe Ratio 1.12 的淨值可能大幅衰減。
*   **數據泄漏防範:** 導讀明確提及初期因多版本股票導致未來值泄漏，修復後才正確。隱含假設為數據清洗管道能完全隔離同公司不同代碼的未來資訊。
*   **樣本外穩定性:** 擴展窗口測試僅覆蓋 2020-2023（含疫情與加息週期），未驗證長週期牛熊轉換下的穩健性。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| LSTM Deep Momentum | 序列建模能力 vs 長週期依賴捕捉 | 部分開源 | 成熟基線 |
| 傳統 MACD / 動量因子 | 靜態規則 vs 端到端損失優化 | 開源 | 標準對照 |
| 強化學習倉位管理 (RL) | 離線損失擬合 vs 線上環境交互 | 部分開源 | 實驗階段 |

🎤 **Interview Tip:**
*   ✅ 正確答：「本法核心不在於預測收益率，而在於以負 Sharpe Ratio 為損失函數進行端到端倉位優化。Attention 機制在此的作用是為歷史價格步長分配動態權重，配合 CPD 標記 regime shift，使模型在波動率放大時自動降倉。實戰瓶頸在於股票高協方差與 CPD 噪聲會稀釋注意力權重的有效性。」
*   ❌ 錯答：「這模型用 Transformer 預測明天股價漲跌，然後根據預測結果買入。因為 Attention 能看很長歷史，所以比 LSTM 準。」（混淆了預測目標與倉位優化目標，忽略損失函數設計與市場微結構限制。）

**7.1 可證偽預測:** 若將本法部署於 2024-2025 年低波動、趨勢明確的市場環境，且維持相同 CPD 窗口與 Attention Heads 配置，其平均 Sharpe Ratio 應顯著高於 1.12；若市場維持高波動與頻繁震盪，模型表現將退化為僅多頭策略的波動率濾鏡，Sharpe Ratio 不會產生結構性突破。（驗證截止：2025-12-31）

## §8 · For the Reader
*   **因子研究員:** 勿將 Attention 權重直接解讀為因子暴露。本法權重分配受負 Sharpe 損失驅動，本質是風險預算分配器。建議將 CPD 輸出作為 regime filter，疊加傳統動量因子進行線性組合，而非直接替換。
*   **組合配置/風控:** 關注模型在尾部回撤期的倉位切換頻率。導讀指出模型在僅多頭大幅回撤時表現穩定，但這是以犧牲上行爆發力為代價。實盤需設定最大倉位切換成本閾值，避免頻繁調倉侵蝕淨值。
*   **量化開發/工程:** 優先解決數據管道中的「同公司多代碼」未來值泄漏問題。CPD 計算可考慮離線預計算並緩存 `cp_score`，避免訓練時重複執行回溯，節省 GPU 記憶體以騰出空間擴展時間窗口。

## References
*   Wood et al., *Trading with the Momentum Transformer: An Intelligent and Interpretable Architecture* (原論文)
*   QuantML 導讀: [增强动量策略：动量Transformer模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247488434&idx=1&sn=e8431bb689e27562a620cdb6cf3c6d31&chksm=ce7e74acf909fdba23e88cba2b6157174f0d50d61d3fab4ebf305a80c01d90bd43ff6e4d1a33#rd)
*   數據源: CRSP / Compustat (SIC 分類篩選)