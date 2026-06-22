<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# 寻找“丑陋的优势”：在加密市场的混乱中收割超额A 解構（寻找“丑陋的优势”：在加密市场的混乱中收割超额A）

> **發布**：2025-11-29 · （無 venue）
> **QuantML 導讀**：[寻找“丑陋的优势”：在加密市场的混乱中收割超额Alpha](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492528&idx=1&sn=685b03c39275c8a64e4ae38ddfb52886&chksm=ce7d84aef90a0db8a3bc4a5bb8cd4927a101e711263987598e7438b399503da0adbe9531b314#rd)
> **核心定位**：落點於日频波段與因子挖掘軸，解決 TradFi 趨勢策略在低效市場中因過度平滑與特徵膨脹導致的 Alpha 衰減。以 Vol-normalization 替代複雜 ML 特徵，將市場微觀結構缺陷（期現背离、資金費率扭曲、價格不敏感流）轉化為可組合的截面信號。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 放棄學術級特徵工程，改用波動率標準化趨勢信號與 Dirty Carry 組合。② 核心 trick 在於利用期現背离與價格不敏感流識別操縱，並透過鏈上金庫實現透明執行。③ 對日频波段軸★：證明「簡單信號+嚴格風險預算」在低效市場能突破 TradFi 的 Sharpe 天花板。④ 導讀指出加密市場 Vol-targeted CTA 可實現约1.7的夏普比率，顯著高於传统金融。

**X-Ray.** 本文實質是在 Pareto 前沿上向左下方（低複雜度、高穩健性）移動。傳統因子挖掘常陷入特徵共線性與過擬合陷阱，而本文將 Vol-normalization 作為唯一截面對齊錨點，直接切斷了高頻噪音與市值規模的混淆效應。它解了兩個舊工程坑：一是 CTA 在加密市場的凸性與夏普不可能三角（透過 Vol Targeting 放棄極端長尾換取機構化融資曲線）；二是 Dirty Carry 在二三線交易所的定價偏差捕獲。然而，該框架打不開的 Envelope 極明確：當加密市場流動性收縮、資金費率轉負或鏈上 MEV 搶跑加劇時，趨勢信號的衰減速度將呈指數級上升，且 Dirty Carry 會瞬間轉為負 Carry。對量化讀者而言，這不是新模型，而是信號組合的權重重分配：趨勢提供方向性 Beta，Dirty Carry 提供低相關性 Alpha，操縱信號提供尾部對沖。實戰價值在於將「不可解釋的市場混亂」拆解為可定價的風險溢價來源。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作/傳統做法 | 本法架構 | 工程意圖 |
|---|---|---|---|
| 信號生成 | 複雜 ML 特徵/長週期趨勢 | Vol-normalized 突破/均線交叉 + 中短期窗口 | 截斷過擬合，對齊不同流動性資產的波動率尺度 |
| 風險預算 | Loose Pants（二元全倉/寬止損） | Vol Targeting（動態倉位縮放） | 犧牲正向凸性換取 Sharpe 穩定性與融資可行性 |
| 執行層 | CEX 黑箱/手動調倉 | 鏈上訂單簿 + 金庫模式（Vaults） | 消除對手方風險，實現策略透明與自動分潤 |

⚡ **Eureka Trick:** `Signal_{norm} = Signal_{raw} / \sigma_{t}` 配合期現背离閾值觸發，將「市場操縱」轉化為高勝率做空信號。直覺：操縱者拋壓不計滑點，價格不敏感流會率先撕裂 Basis，趨勢信號僅負責確認方向。

```
[Price/Funding Data] → Vol-Normalizer → [Trend Signal] ─┐
[Spot-Futures Basis] → Divergence Filter → [Manipulation Signal] ─┤ → [Vol-Targeted Portfolio] → [On-Chain Vault Execution]
[Funding Rate] → Dirty Carry Mapper → [Carry Signal] ────────┘
```

## §2 · 數學層
📌 **Napkin Formula:**
$$w_i = \frac{\sigma_{target}}{\sigma_i} \cdot \text{sign}(Trend_i) \cdot \mathbb{I}(|Basis_i| > \theta)$$
複雜度：$O(N \cdot T)$ 截面計算，無迭代訓練。直覺：權重由目標波動率與資產實時波動率反比決定，趨勢方向與 Basis 閾值作為非線性開關。Loss/訓練細節：無監督回歸或梯度下降；信號權重依賴歷史波動率滾動估計與靜態閾值 $\theta$，屬規則型組合優化。

## §3 · 數據層
- **規模/頻率/市場**：加密資產量價與鏈上數據，日频波段。選幣池限制於市值前50或前1/3。
- **來源**：CEX 行情（Binance/Bybit 等頭部） + 二三線交易所資金費率 + 鏈上金庫/ThorChain 等流動性路由。
- **樣本外與容量假設**：導讀未披露具體回測區間與樣本量。容量假設隱含於「頭部交易所執行趨勢、邊緣交易所執行套利」的分層邏輯中，小市值資產趨勢效應衰減或轉負。

## §4 · 代碼層
| 欄位 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | 未披露 |
| License | 未披露 |
| 複現難度 | 中（需跨所數據對齊與鏈上 RPC 節點） |
| 數據可得性 | 部分公開（CEX API 易得，鏈上操縱流需自定義清洗） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 传统金融Vol-targeted CTA | Sharpe | <1.0 | 约1.7 | 未披露 |

**解讀**：Δ 的來源並非模型容量提升，而是市場效率差異與風險預算重構。加密市場的高波動與羊群效應放大了趨勢信號的持續性，Vol Targeting 平滑了收益曲線。但此 Δ 極易受前瞻偏差侵蝕：導讀未計入資金費率翻轉成本、鏈上 Gas 波動與滑點。若將 Dirty Carry 納入，實際淨值 Sharpe 可能顯著低於毛值。此為真實 Capability 與成本未計的混合體。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**
- 長期趨勢信號在加密市場效果不佳，易受劇烈回撤侵蝕。
- Loose Pants 策略收益分布極度偏態，依賴運氣與方差，夏普較低。
- 加密市場速通傳統金融恐慌史，需警惕對手方風險、智能合約風險與系統性崩盤。

**6.2 推斷的隱含假設**
- **Regime 依賴**：假設市場維持高波動與正資金費率環境；若進入低波動熊市或費率長期為負，Dirty Carry 與趨勢信號將同時失效。
- **容量假設**：策略僅對市值前50或前1/3資產有效；小市值趨勢轉負，容量上限受二三線交易所流動性碎片化限制。
- **數據泄漏風險**：鏈上操縱信號（價格不敏感流）若依賴延遲 RPC 或公開瀏覽器數據，實盤執行時可能已被 MEV 搶跑。
- **成本假設**：未明確披露交易成本與融資成本閾值；Dirty Carry 在「垃圾交易所」的利潤可能被提現摩擦與清算風險抵消。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統 ML 因子挖掘 | 特徵複雜度 vs Vol-normalization 極簡主義 | 未披露 | 過擬合風險高 |
| Loose Pants CTA | 凸性捕獲 vs Vol Targeting 平滑曲線 | 未披露 | 回撤不可控 |
| 鏈上 Alpha 捕獲 | 智能合約自動化 vs 手動/CEX 執行 | 未披露 | 對手方風險轉移 |

🎤 **Interview Tip**
- **正確答**：「Vol-normalization 不是降維，而是截面風險對齊。趨勢提供方向，Dirty Carry 提供低相關 Beta，操縱信號提供尾部對沖。實盤需動態監控資金費率符號與 Basis 收斂速度，否則 Carry 會轉為負向拖累。」
- **錯答**：「用 LSTM 預測價格就能捕獲所有 Alpha，簡單均線交叉在加密市場已經失效。」（忽略市場微觀結構與風險預算的組合價值）

**7.1 可證偽預測帶日期**：若加密市場整體資金費率連續未披露個交易日轉負，且頭部交易所 Basis 收斂至未披露水平，本法 Dirty Carry 與趨勢信號的聯合 Sharpe 將跌破未披露閾值。（具體日期與數值導讀未披露，留待實盤驗證）

## §8 · For the Reader
- **因子研究員**：放棄高維特徵堆疊，優先實現 Vol-normalization 截面對齊。將 Basis 背离與資金費率作為獨立因子層，而非價格預測的輔助特徵。
- **高頻執行**：日频趨勢無需低延遲架構，但 Dirty Carry 與操縱信號需跨所數據對齊與鏈上 RPC 節點優化。滑點模型必須納入 Gas 波動與流動性碎片化。
- **組合配置**：將 Vol Targeting 視為風險預算錨點，而非收益增強器。組合權重應隨市場波動率動態縮放，避免在超級單邊行情中因強制減倉錯失長尾。

## References
- 原論文/框架：寻找“丑陋的优势”：在加密市场的混乱中收割超额A
- QuantML 導讀：[寻找“丑陋的优势”：在加密市场的混乱中收割超额Alpha](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492528&idx=1&sn=685b03c39275c8a64e4ae38ddfb52886&chksm=ce7d84aef90a0db8a3bc4a5bb8cd4927a101e711263987598e7438b399503da0adbe9531b314#rd)
- Lineage：Vol-targeted CTA / Dirty Carry / On-chain Vaults / Market Microstructure Alpha