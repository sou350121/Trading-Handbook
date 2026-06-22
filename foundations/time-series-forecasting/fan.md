---
title: "FAN"
description: "落點於 **监督回归 × 端到端表征** 軸，針對 **量价表格/日频波段** 數據。解決了傳統實例歸一化僅能提取均值/趨勢、無法動態捕捉演變季節性與頻譜結構的 **prior gap**，將非平穩信號剝離為可預測的頻域殘差。"
---
<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

> **發布**：2024-10-13 · （無 venue）
> **QuantML 導讀**：[频率自适应归一化用于非平稳时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487027&idx=1&sn=5b926cdd42c1bb19e4b6a90fec37b1a2&chksm=ce7e692df909e03bcdb66f737db1b2cea6495bd67904c7c108f91fdb77f797b206a620a3a1bf#rd)
> **核心定位**：落點於 **监督回归 × 端到端表征** 軸，針對 **量价表格/日频波段** 數據。解決了傳統實例歸一化（如 RevIN/SAN）僅能提取均值/趨勢、無法動態捕捉演變季節性與頻譜結構的 **prior gap**，將非平穩信號剝離為可預測的頻域殘差。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出頻域實例歸一化 FAN，以 DFT 動態提取 TopK 主導頻率取代統計歸一化。② 核心 trick 是用 MLP 顯式預測這些頻率成分的未来演變，實現可逆去歸一化。③ 對「端到端表征」軸★，它解耦了非平穩趨勢/季節性與平穩動態，大幅降低背景模型拟合高頻噪聲的負擔。④ 關鍵實證：在 8 個基準數據集上 MSE 平均提升 7.76%–37.90%（具體拆分與交易成本未披露）。

**X-Ray.** FAN 將時間序列預測的 Pareto 前沿從「時域統計量壓縮」推向「頻譜動態剝離」。傳統 InstanceNorm 假設非平穩性僅是均值/方差的緩慢漂移，但在金融/宏觀數據中，季節性與週期性往往以非線性、時變頻率存在。FAN 的工程價值在於：它不修改 backbone，而是以 **可插拔的頻域預處理+後處理模塊** 形式存在，強制模型只學習平穩殘差，將非平穩信號的預測外包給輕量 MLP。這解決了長序列預測中梯度被趨勢項淹沒的舊坑。然而，它的 envelope 受限於 DFT 的全局視窗假設與 TopK 截斷的頻譜泄漏風險；在 regime shift 劇烈或頻率成分高度重疊的市場中，MLP 對頻率演變的淺層非線性擬合可能失效。對量化讀者而言，FAN 不是直接產 alpha 的因子，而是 **特徵工程與預測頭部的解耦框架**，適合嵌入多因子預測流水線，但需警惕頻域操作引入的潛在 look-ahead 與計算延遲。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作 (RevIN/SAN/Dish-TS) | FAN (本方法) | 工程意義 |
|---|---|---|---|
| 歸一化基準 | 時域統計量 (均值/方差/分位數) | 頻域 TopK 主導頻率成分 | 避開均值漂移對季節性模式的掩蓋 |
| 非平穩信號處理 | 靜態移除或簡單線性回歸 | DFT 提取 + MLP 動態預測未來演變 | 將去歸一化從「固定映射」升級為「可學習殘差」 |
| 模型耦合度 | 緊耦合於特定 backbone | 模型無關 (Model-Agnostic) 插件 | 可無縫替換 DLinear/Transformer/CNN 的 Norm 層 |

⚡ **Eureka:** 放棄時域統計假設，直接在頻域把「非平穩性」定義為振幅最大的 TopK 頻率，並用 MLP 預測它們如何隨時間漂移，最後用 IDFT 拼回預測值。

**信息流 ASCII:**
Input X (T) -> [DFT] -> Freq Spectrum -> [TopK Filter] -> X_freq (Non-stationary)
                                  |
                                  v
X_stable = X - IDFT(X_freq) -> [Backbone Model] -> Ŷ_stable
                                  |
                                  v
X_freq + Context -> [MLP Predictor] -> Ŷ_freq (Predicted Non-stationary)
                                  |
                                  v
Final Ŷ = Ŷ_stable + IDFT(Ŷ_freq)  <-- Reversible Denormalization

## §2 · 數學層
📌 **Napkin Formula:**
$$ \mathcal{F}(X) = \text{DFT}(X), \quad \mathcal{K} = \text{TopK}(|\mathcal{F}(X)|) $$
$$ X_{\text{stable}} = X - \text{IDFT}(\mathcal{F}(X) \odot \mathbb{I}_{\mathcal{K}}), \quad \hat{Y}_{\text{freq}} = \text{MLP}(X_{\text{freq}}, \text{context}) $$
$$ \hat{Y} = \hat{Y}_{\text{stable}} + \text{IDFT}(\hat{Y}_{\text{freq}}) $$

**直覺:** 頻域濾波相當於帶通/低通分離，TopK 保留了能量最高的週期/趨勢項。MLP 學習頻率振幅與相位的時變規律，避免背景模型強行拟合非平穩跳變。
**複雜度:** DFT/IDFT 為 $O(T \log T)$，TopK 選擇 $O(K)$，MLP 為 $O(K \cdot d_{\text{mlp}})$。整體僅增加常數級開銷，不改變 backbone 漸進複雜度。
**Loss/訓練:** 標準 MSE/MAE 監督回歸。FRL (Frequency Residual Learning) 與背景模型端到端聯合訓練，梯度可透過 IDFT 與 MLP 反向傳播。未披露是否使用頻域正則化或頻率一致性約束。

## §3 · 數據層
- **資料規模/頻率:** 8 個公開基準數據集 (ETTm2, Electricity, Exchange, Traffic, Weather 等)，覆蓋電力、匯率、交通、氣象。頻率未明確標註為金融日頻，但 Exchange/Traffic 具多變量時間序列特性。
- **來源與處理:** 標準公開 benchmark，統一進行 z-score 歸一化作為 baseline 對照。訓練/驗證/測試劃分 7:2:1。
- **預測長度:** $H \in \{96, 168, 336, 720\}$，覆蓋短至超長週期。
- **樣本外假設:** 嚴格時間序列劃分，未提及滾動窗口或 walk-forward 驗證。金融實盤需警惕靜態劃分帶來的 regime 過擬合。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | TBD（導讀提及「論文及代碼下載見星球」，未給出 GitHub 鏈接） |
| Checkpoint | 未披露 |
| License | 未披露 |
| 複現難度 | 低 (DFT/IDFT/MLP 為標準組件，可作為 PyTorch 插件替換 Norm 層) |
| 數據可得性 | 高 (均為公開 benchmark，金融實盤需自行構建對標數據) |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前 SOTA (RevIN/SAN等) | 本方法 (FAN) | Δ |
|---|---|---|---|---|
| 8 個基準集 (平均) | MSE | 未披露 | 未披露 | 7.76% – 37.90% (導讀原文區間) |
| ExchangeRate (H=720) | MSE/MAE | 未披露 | 未披露 | 顯著優於基線 (導讀定性描述) |
| 其他數據集 | IR/Sharpe/AR/MDD | 未披露 | 未披露 | 未披露 |

**解讀:** 提升區間 7.76%–37.90% 為 MSE 平均改善，屬純預測精度指標。**真 capability** 體現在長週期 ($H=720$) 與高非平穩性數據集 (Exchange) 上，證明頻域剝離有效降低了趨勢/季節性對長期預測的干擾。**潛在偏差:** ① 未披露交易成本、滑點與 turnover，MSE 下降不等於實盤 Sharpe 提升；② 靜態 7:2:1 劃分可能包含未來信息或忽略結構性斷點；③ TopK 選擇若依賴全局統計或驗證集調參，存在數據泄漏風險。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:**
- TopK 頻率成分數值 $K$ 需手動設定或依賴啟發式規則，缺乏自主動態確定機制。
- 僅驗證了 4 種 backbone (DLinear, Informer, FEDformer, SCINet)，未覆蓋 RNN/RL 架構。

**6.2 推斷的隱含假設:**
- **Regime 依賴:** DFT 假設信號在視窗內近似平穩或週期性重疊，若市場經歷流動性枯竭或政策突變，頻譜結構會劇變，TopK 濾波可能誤傷有效信號。
- **容量與成本:** MLP 預測頻率演變增加了推理延遲，在低延遲執行場景中需評估開銷。
- **數據泄漏/前瞻:** DFT 若使用全序列或未來視窗計算振幅，將引入嚴重 look-ahead。實盤必須嚴格使用滾動因果視窗 (Causal DFT)。
- **Survivorship:** 公開 benchmark 通常為靜態選股/選標的，未處理退市/合併偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| RevIN / SAN | 時域統計量 vs 頻域動態頻率 | Open | 成熟基線 |
| Dish-TS | 去季節性分解 vs 頻譜殘差學習 | Open | 成熟基線 |
| FEDformer | 內置 FFT 注意力 vs 外掛頻域歸一化 | Open | 架構級 vs 插件級 |

🎤 **Interview Tip:**
- **正確答:** 「FAN 的核心不是預測價格，而是預測『非平穩性的演變軌跡』。它將歸一化從靜態統計操作轉為可學習的頻域殘差映射，解耦了趨勢/季節性與平穩動態，適合嵌入任何回歸型預測頭，但需嚴格保證因果視窗以防 look-ahead。」
- **錯答:** 「FAN 是用傅里葉變換直接預測股價，比 Transformer 更準。」（混淆了歸一化模塊與預測模型，且忽略其模型無關屬性與實盤成本約束。）

**7.1 可證偽預測:** 若在 2025-Q3 前，將 FAN 應用於 A 股日頻多因子預測流水線，在嚴格 walk-forward 與含成本回測下，其 Sharpe 提升若未顯著優於 RevIN+LSTM 基線，則證明頻域剝離在橫截面異質性強的金融數據中收益遞減。

## §8 · For the Reader
- **因子研究員:** 將 FAN 視為特徵標準化層的升級版。在構建多因子預測模型時，用 FAN 替換標準 Z-Score/RevIN，觀察長週期因子衰減是否延緩。注意 TopK 需按因子頻譜特性獨立設定。
- **組合配置/風險:** FAN 提升的是點預測精度，不直接輸出分佈或風險矩陣。若用於組合優化，需將 FAN 輸出接入 Copula 或 GARCH 層以捕捉尾部風險，避免將 MSE 改善誤讀為波動率下降。
- **模型架構師/高頻執行:** FAN 的 DFT/IDFT 與 MLP 可並行化，但需評估推理延遲。在低延遲場景中，可預計算 TopK 頻率索引並緩存 MLP 權重；若延遲敏感，建議降頻使用或僅在日頻波段策略中部署。

## References
- 原論文: Frequency Adaptive Normalization for Non-stationary Time Series Forecasting (2024-10-13, 無 venue)
- Lineage: RevIN (Kim et al., 2021) → SAN → Dish-TS → FAN
- QuantML 導讀: [频率自适应归一化用于非平稳时间序列预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487027&idx=1&sn=5b926cdd42c1bb19e4b6a90fec37b1a2&chksm=ce7e692df909e03bcdb66f737db1b2cea6495bd67904c7c108f91fdb77f797b206a620a3a1bf#rd)