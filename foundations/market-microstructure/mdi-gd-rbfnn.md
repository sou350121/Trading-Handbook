<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=监督回归 alpha=因子挖掘 autonomy=全自动黑盒 -->

# MDI-GD竞争聚类RBFNN框架 解構（MDI-GD竞争聚类RBFNN框架）

> **發布**：2024-12-24 · （無 venue）
> **QuantML 導讀**：[在线高频交易股票预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247488504&idx=1&sn=18be32739d0029fdd5a743354ec59430&chksm=ce7e74e6f909fdf0b18b254c7e283c3e6c91be5da516c1b8f1f0f9d8c6a5bc644ce3337dcfb9#rd)
> **核心定位**：落點於「自動黑盒」與「因子挖掘」軸，解決 HFT 場景下人工特徵選擇與拓撲搜尋的延遲瓶頸。以線上競爭機制取代靜態超參數調優，將決策鏈路壓縮至事件級響應。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出全自動 HFT 預測協議，將 GD 轉譯為特徵重要性指標與 MDI 線上競爭。② 核心 trick 是引入輪廓係數動態優化 k-means 聚類數，驅動 RBFNN 進行納秒級 LOB 中間價回歸。③ 對「自動黑盒」軸而言，它移除了人工特徵工程與靜態拓撲搜尋的滯後性，實現事件級自適應。④ 導讀指出 MDI 在 60 個案例中 36 個取得最低 RRMSE，且特徵重要性方法平均每 10 個交易事件切換一次。

**X-Ray.** 放回五軸 Pareto，此框架將「因子挖掘」與「自動黑盒」推向極致，代價是犧牲了模型的可解釋性與廣譜泛化能力。它解決了傳統 HFT 機器學習流程中兩大工程坑：一是靜態特徵選擇導致的概念漂移滯後，二是 k-means 肘部法在線上環境的計算延遲。透過 MDI 與 GD 的線上競爭，系統能在微觀盤口結構快速切換時自動重構輸入空間。然而，其 envelope 受限於 RBFNN 的局部逼近特性與各向同性聚類假設，無法捕捉長週期宏觀 regime 切換或跨資產聯動。對量化讀者而言，這不是一個直接可交易的 Alpha 來源，而是一個高頻特徵工程自動化模組的參考實作；其價值在於證明線上拓撲自適應在數據流中的可行性，但實盤落地必須補齊交易成本模型與滑點模擬，否則低回歸誤差僅是數學上的過擬合幻覺。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 傳統 HFT ML 協議 | 本框架 (MDI-GD-RBFNN) | 工程意義 |
|---|---|---|---|
| 特徵重要性 | 人工篩選 / 靜態 RF-MDI | MDI 與 GD 線上競爭矩陣 | 消除人工滯後，適應盤口快速切換 |
| 聚類拓撲 | 肘部法 / 手動指定 k | 輪廓係數動態計算最優 k | 線上環境免於耗時網格搜尋 |
| 學習範式 | 離線訓練 / 批次更新 | 滑動窗口 100 事件 + 累積五折 | 實現事件級在線學習與模型遞歸 |

**1.2 ⚡ Eureka 一句話 trick + 直覺**
將 GD 的權重更新軌跡轉譯為特徵重要性向量，與樹模型的 MDI 在同一距離矩陣上「賽跑」，勝者直接決定 RBFNN 的隱藏層中心分佈。直覺上，這等於讓梯度下降與不純度降低在線上環境中互相校準，避免單一指標在特定 regime 下失效。

**1.3 信息流 ASCII 圖**
```
[LOB Tick Stream] -> [Sliding Window (100 evt, 99 overlap)]
              |
              v
    [MDI Vector] <---(Compete)---> [GD Vector]
              |                         |
              v                         v
[Correlation Matrix] -> [Distance Matrix] -> [K-Means (k via Silhouette)]
              |
              v
[RBFNN Hidden Layer (Centers/Widths)] -> [Linear Output] -> [Mid-Price Forecast]
              |
              v
    [Cumulative 5-Fold Update] -> [Next Window]
```

## §2 · 數學層
**📌 Napkin Formula**
$$ \Delta i_{j,f} = \frac{N_j}{N_{all}} \cdot MSE_j - \sum_{child} \frac{N_{child}}{N_{all}} \cdot MSE_{child} $$
$$ \text{Silhouette}(i) = \frac{b_i - a_i}{\max(a_i, b_i)} $$
$$ \hat{y} = \sum_{j=1}^{k} w_j \cdot \exp\left(-\frac{\|x - c_j\|^2}{2\sigma_j^2}\right) $$
**直覺**：MDI 量化節點分裂對 MSE 的貢獻；輪廓係數衡量樣點與同簇/異簇的相對緊密度；RBFNN 以高斯核將非線性盤口距離映射至線性輸出。
**Loss/訓練**：回歸任務以 MSE 為節點不純度度量；GD 透過迭代更新權重向量生成重要性矩陣；RBFNN 隱藏層中心由競爭後的最優 k-means 決定，輸出層權重以閉式解或最小二乘計算。整體複雜度受控於窗口大小與聚類數，適合線上遞歸。

## §3 · 數據層
- **資料規模/頻率/市場/時段**：Refinitiv 提供 20 只美國大型股票數據，時間解析度為納秒級。
- **來源與處理**：直接取自 LOB 最佳買入/賣出價及對應量，衍生簡單與擴展特徵集。
- **樣本外與容量假設**：採用滑動窗口（100 事件，重疊 99）與累積五折協議。導讀未披露具體總樣本數與回測區間長度，僅提及「每只股票三個月」共 60 個案例。假設盤口微結構在短窗口內保持平穩，未考慮跨日或節假日斷點。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（需自行實作 MDI/GD 線上競爭與輪廓係數動態 k 判定） |
| 數據可得性 | 低（依賴 Refinitiv 納秒級 LOB 數據，零售難以獲取） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 20 US Large-Cap (納秒級) | RRMSE (Simple Set) | 未披露 | MDI 在 36/60 案例最低 | 未披露 |
| 20 US Large-Cap (納秒級) | RRMSE (Extended Set) | 未披露 | GD 在 GOOGL 等案例最低 | 未披露 |
| 20 US Large-Cap (納秒級) | RMSE | 未披露 | 多場景顯示低 RMSE | 未披露 |

**解讀**：導讀僅提供相對表現（MDI 勝出 36 例）與定性描述（低 RMSE），未給出絕對數值或明確基線模型對比。此 Δ 反映的是「線上自適應機制」在特定股票上的相對優勢，而非絕對預測精度突破。低 RRMSE 可能源於納秒級數據的高自相關性與滑動窗口的數據泄漏風險（累積五折將測試集轉為訓練集），實盤中未計入交易成本與滑點前，此類回歸指標極易過擬合。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**
- 狹窄 AI 方法，僅針對 LOB 中間價預測。
- 特徵工程依賴現有最佳價量構建，缺乏更複雜的自主特徵。
- 缺乏廣泛的基準建模框架挑戰 RBFNN 拓撲。
- 假設 k-means 聚類為各向同性（恆定方差），現實盤口結構常呈異質性。
- 數據集長度不足，需延長以提升穩健性。

**6.2 推斷的隱含假設**
- **Regime 依賴**：假設微觀盤口結構在 100 事件窗口內平穩，未處理流動性驟降或閃崩 regime。
- **容量/成本**：未計入納秒級執行成本、訂單簿深度限制與市場衝擊，低 RMSE 不等於正期望收益。
- **數據泄漏**：累積五折協議將歷史測試數據納入後續訓練，在線上環境中構成嚴重的前瞻偏差。
- **Survivorship**：僅選取 20 只大型股，忽略已退市或流動性枯竭標的，存在生存者偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 RF/GBDT + 靜態特徵 | 線上競爭 vs 離線靜態 | 開源生態成熟 | 廣泛使用，但延遲高 |
| LSTM/Transformer HFT | 序列建模 vs 事件級聚類回歸 | 部分開源 | 計算成本高，實時性弱 |
| RL 自動化交易 | 策略優化 vs 特徵/拓撲優化 | 研究階段為主 | 訓練不穩，實盤難控 |

**🎤 Interview Tip**
- **正確答**：「此框架的核心價值在於將 GD 轉化為線上特徵重要性指標，與 MDI 競爭以動態決定 RBFNN 的輸入空間。它解決了靜態特徵選擇的滯後問題，但各向同性聚類假設與累積五折協議在實盤中會引入前瞻性偏差與結構失配。」
- **錯答**：「這是一個能直接產生高頻 Alpha 的預測模型，因為它的 RMSE 很低，所以可以直接掛單交易。」（忽略成本、滑點、數據泄漏與狹窄目標）

**7.1 可證偽預測帶日期**
若於 2025-06-30 前，該框架在包含交易成本與真實訂單簿深度的公開 HFT 基準測試中，未能證明其 RRMSE 優勢能轉化為正 Sharpe 或正 AR，則其「線上自適應」僅為數學層面的過擬合，不具備實盤價值。

## §8 · For the Reader
- **因子研究員**：可參考 MDI/GD 線上競爭機制，將其嵌入因子正交化或動態加權流程，替代靜態 IC 衰減模型。
- **高頻執行**：注意累積五折協議的前瞻性偏差；實盤需將此模組與訂單執行算法解耦，僅作盤口結構探針。
- **組合配置**：此框架輸出為納秒級價格預測，不直接適用日頻組合優化；若需整合，需先將預測信號降頻並過濾交易成本，否則會產生虛假容量假象。

## References
- Adamantios Ntakaris & Gbenga Ibikunle. *Online High-Frequency Trading Stock Forecasting with Automated Feature Clustering and Radial Basis Function Neural Networks*. （無 venue）, 2024.
- QuantML 導讀：[在线高频交易股票预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247488504&idx=1&sn=18be32739d0029fdd5a743354ec59430&chksm=ce7e74e6f909fdf0b18b254c7e283c3e6c91be5da516c1b8f1f0f9d8c6a5bc644ce3337dcfb9#rd)
- Lineage: Random Forest MDI (Breiman et al.) / K-Means & Silhouette (Rousseeuw) / RBFNN (Broomhead & Lowe) / Online Learning Protocols.