<!-- ontology-5axis data=量价表格 horizon=高频日内 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# Informer 解構（Informer）

> **發布**：2025-03-25 · （無 venue）
> **QuantML 導讀**：[基于 Informer 结构的高频比特币策略研究](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489777&idx=1&sn=fa21f1ac62aa7e2e2aefc89c4af56128&chksm=ce7e7feff909f6f999ebd2464ee617a84209323139b43d42ac2c36dac709816923569a720a84#rd)
> **核心定位**：將 Informer 架構引入 BTC 高頻量價預測，透過客製化損失函數（GMADL）與滾動窗口評估，解決傳統 Transformer 在金融時間序列中對極端回報與交易成本敏感的 prior gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** 本文將 Informer 架構應用於 BTC 5/15/30 分鐘量價數據，透過對比 RMSE、Quantile 與 GMADL 損失函數，驗證端到端深度學習信號在頻繁換倉環境下的有效性。核心 trick 在於以 GMADL 損失函數替代傳統 MSE，使模型對高頻回報的極值與趨勢轉折更魯棒。這對「端到端表征」軸具指標意義：證明客製化損失設計比單純堆疊注意力層更能決定高頻策略的實戰存活率。關鍵實證數字：5 分鐘 GMADL 策略年化回報率達到 115%，最大回撤僅為 32.7%。

**X-Ray.** 本方法在五軸 Pareto 上明確落於「高頻/黑盒/端到端」的極端區間，但刻意避開了純價格預測的陷阱，轉而將損失函數設計錨定在交易決策的邊際效益上。它解了舊工程坑：傳統 Transformer 在金融序列上易受極端波動干擾導致梯度不穩，GMADL 透過分位數調整平滑了訓練信號。然而，其 envelope 受限於滾動窗口的固定切分與未計入滑點/流動性衝擊的假設，在真實訂單簿深度不足時必然失效。對量化讀者而言，此文價值不在架構本身，而在於「損失函數選擇決定策略鋒利度」的實證結論，提示高頻 Alpha 生成應從網絡結構微調轉向目標函數的經濟意義對齊。

## §1 · 架構 / Core Mechanism
| 維度 | 原始 Informer (時序預測) | 本文適配 (高頻交易) |
|---|---|---|
| 預測目標 | 長序列未來數值回歸 | BTC 區間開盤至收盤相對回報 |
| 優化目標 | RMSE / MSE | GMADL / Quantile 客製化損失 |
| 決策框架 | 純數值輸出 | 預測值閾值觸發 Long/Short/Flat 換倉 |

⚡ **Eureka:** 用交易導向的損失函數（GMADL）直接優化模型對極端回報的敏感度，取代純統計誤差最小化，使網絡輸出天然貼合趨勢跟隨的決策邊界。
```
[5/15/30m K線] → [Informer Encoder/Decoder] → [GMADL Loss Optimized Prediction] → [Threshold Decision] → [Long/Short/Flat]
```

## §2 · 數學層
📌 **Napkin Formula:** `r_t = (P_{close} - P_{open}) / P_{open}` ； `Loss = GMADL(r_pred, r_true)` [公式細節未披露] ； 複雜度：未披露
**直覺:** 將預測目標錨定在區間相對回報，GMADL 透過對稱/分位調整抑制極值對梯度的支配，使模型輸出更貼合趨勢跟隨的決策邊界，而非單純拟合價格軌跡。
**Loss/訓練細節:** 隨機搜索超參，驗證集選最低損失組合，訓練時間約每運行 1 小時。

## §3 · 數據層
- **資料規模/頻率/市場/時段:** BTC/USDT，2019-08-21 至 2024-07-24，5/15/30 分鐘 K 線。
- **怎麼來:** Binance API 下載，8 個缺口（總計缺失超過 101 小時）用前值填充。
- **樣本外與容量假設:** 滾動窗口（24個月 IS / 6個月 OOS，步長 6個月）；假設可交易任意部分、全倉買賣、以區間收盤價成交、需計交易費用。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 中（需自行對接 API 並實作 GMADL 損失） | 高（Binance 公開歷史 K 線） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| BTC/USDT 5m | AR | 未披露 | 115% | 未披露 |
| BTC/USDT 5m | MDD | 77.3% (B&H) | 32.7% | 44.6% |

**解讀:** 5 分鐘頻率下 GMADL 策略的 AR 與 MDD 顯著優於 B&H，Δ 反映的是高頻趨勢跟隨對極端回撤的過濾能力。但導讀未披露交易費用與滑點計入細節，且策略在 15/30 分鐘頻率下優勢縮減甚至被 RSI 反超，顯示該 Δ 高度依賴特定頻率與驗證窗口切分（敏感性分析指出窗口數量改變會導致性能變差），存在過擬合與前瞻偏差風險。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** Quantile Informer 表現不佳；RMSE 在更高頻下性能下降；GMADL 性能可能受評估窗口數量選擇影響（孤立事件風險）；未進行更廣泛敏感性分析。
**6.2 推斷的隱含假設:** Regime 依賴（比特幣高波動與趨勢市表現好，震盪市可能失效）；容量假設（全倉買賣、可交易任意部分，未考慮大單衝擊）；成本假設（僅提交易費用，未量化具體 bp 與滑點）；數據泄漏風險（缺口填充用前值，可能引入微小前瞻）；Survivorship（BTC/USDT 為頭部流動性對，未涵蓋小幣）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| LSTM/GRU | 序列建模機制（RNN vs Attention） | Open | 成熟 |
| TFT | 靜態協變量與門控選擇 | Open | 成熟 |
| MACD/RSI | 特徵工程 vs 端到端學習 | Open | 成熟 |

🎤 **Interview Tip:** 
正確答：高頻策略的瓶頸不在網絡深度，而在損失函數是否對齊交易決策的經濟目標（如 GMADL 對極值魯棒性）；
錯答：Transformer 憑藉全局注意力就能自動過濾所有市場噪音，無需客製化損失。

**7.1 可證偽預測帶日期:** 若 2025-Q4 加密貨幣市場進入低波動震盪行情，GMADL 策略的換倉頻率將急劇下降，年化回報率預計跌破 50%，且最大回撤將向 B&H 的歷史區間收斂。

## §8 · For the Reader
- **因子研究員:** 關注 GMADL 損失函數的數學形式與梯度特性，嘗試將其移植至傳統多因子模型的回報預測層。
- **高頻執行:** 警惕「全倉買賣」與「前值填充缺口」的實盤落差，實盤需加入訂單簿深度過濾與動態倉位控制。
- **組合配置:** 將該策略視為高頻動能子策略，與低頻均值回歸策略正交組合，利用其 MDD 控制優勢平滑整體曲線。

## References
- 原論文: TBD (無 venue/arxiv)
- Lineage: Vaswani et al. (Transformer) → Zhou et al. (Informer) → 本文 (Informer + GMADL for Crypto)
- QuantML 導讀鏈接: [基于 Informer 结构的高频比特币策略研究](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489777&idx=1&sn=fa21f1ac62aa7e2e2aefc89c4af56128&chksm=ce7e7feff909f6f999ebd2464ee617a84209323139b43d42ac2c36dac709816923569a720a84#rd)