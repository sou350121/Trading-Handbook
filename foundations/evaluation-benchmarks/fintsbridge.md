<!-- ontology-5axis data=量价表格 horizon=跨周期 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# FinTSBridge 解構（FinTSBridge）

> **發布**：2025-03-11 · （無 venue）
> **QuantML 導讀**：[融合前沿深度学习模型的金融时序预测框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489625&idx=1&sn=571070b27861b9058ffc9ee96a6ea4c6&chksm=ce7e7f47f909f6512ec3d1b249e59dcb2cf08bf3ef673c7544294e44ad46a2dfcf86b5d2a961#rd)
> **核心定位**：落點於「量价表格 × 跨周期 × 監督回歸 × 端到端表徵 × 全自動黑盒」軸心，解決傳統時間序列預測在金融場景中「誤差指標與實盤信號穩定性脫鉤」的 prior gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 構建跨頻率金融數據集（GSMI/OPTION/BTCF）並系統對齊十餘種 SOTA 模型。② 首創 msIC/msIR 替代純誤差評估，直接捕捉多步預測的時序相關性與穩定性。③ 對「端到端表徵」軸★：將模型輸出從點對點數值對齊至實盤所需的信號方向與一致性，彌合學術指標與交易需求的斷層。④ 導讀未給量化結果。

**X-Ray.** FinTSBridge 並非提出新架構，而是做了一層「評估層重映射」。學術界慣用的 MSE/MAE 在非平穩金融序列中會獎勵 Naive 模型（低誤差但無方向預測力），導致模型優化目標與實盤 Sharpe/IR 嚴重錯位。該框架將 Pareto 前沿從「誤差最小化」推向「相關性穩定性最大化」，直擊量化工程中长期被忽視的「信號衰減與 regime 切換」坑。但它的 envelope 打不開實盤成本結構：msIC/msIR 僅衡量離線序列排序一致性，未內建滑點、流動性衝擊與倉位規模約束。對量化讀者的意義在於：提供了一套可插拔的離線過濾儀表板，可用於早期淘汰「過擬合誤差但無實盤方向性」的模型，但不可直接作為策略上線的決策依據。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作（主流 TS 預測） | FinTSBridge |
|---|---|---|
| 評估指標 | MSE / MAE（點對點誤差） | msIC / msIR（序列排名相關性與穩定性比率） |
| 數據頻率 | 單一頻率（日/小時/分鐘） | 跨頻率對齊（日頻指數 / 分鐘期權 / 小時加密） |
| 驗證協議 | 純預測誤差回測 | 策略模擬對齊（擇時 / 多空 / 組合權重） |

**1.2 ⚡ Eureka**
用排名相關係數的均值與標準差比率（msIC/msIR）替代純誤差，直接對齊「預測方向一致性」與「實盤信號穩定性」。

**1.3 信息流 ASCII**
```
[多頻 OHLCV] → [Log-Diff + 錨定100] → [SOTA Backbone] → [msIC/msIR Calculator] → [策略模擬器] → [實盤信號過濾]
      ↑              ↑                      ↑                  ↑                    ↑
   市場數據        量綱對齊             監督回歸訓練        相關性穩定性評估      離線經濟價值 proxy
```

## §2 · 數學層
📌 **Napkin Formula**
$msIC = \frac{1}{B \cdot C} \sum_{b=1}^{B} \sum_{c=1}^{C} \text{RankCorr}(y_{b,c}, \hat{y}_{b,c})$
$msIR = \frac{\text{mean}(msIC_{sample})}{\text{std}(msIC_{sample})}$
**直覺**：放棄絕對數值誤差，轉向序列內排序一致性。msIC 抓方向預測力，msIR 抓跨樣本穩定性（有效信號 vs 時變分佈噪聲）。
**Loss/訓練**：導讀未披露自定義 Loss，模型仍依賴標準 MSE/MAE 監督回歸訓練；msIC/msIR 僅作為離線評估與 Early Stopping 指標。複雜度與 Backbone 一致（如 Transformer $O(T^2)$ 或 Linear $O(T)$），排名相關計算額外增加 $O(T \log T)$。

## §3 · 數據層
- **規模/頻率/市場/時段**：GSMI（20個全球指數，2005-2024，日頻）；OPTION（CSI 300 ETF期權，2024，分鐘級）；BTCF（比特幣現貨-永續合約，2020-2024，小時級）。
- **來源與預處理**：公開市場數據（導讀未披露具體供應商）。採用對數差分並減去初始值，再加常數項100錨定基線，解決量綱差異與負值問題。
- **樣本外與容量假設**：嚴格時間切分（導讀未披露具體劃分比例）。容量假設受限于日/小時/分鐘頻率的實盤滑點與合約流動性（未驗證），分鐘級期權數據僅適合極小資金或做市商級別。

## §4 · 代碼層
| 項目 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（依賴標準 TS 模型庫，需自寫 msIC/msIR 與策略回測邏輯） |
| 數據可得性 | 高（GSMI/BTCF 可從公開源獲取，OPTION 需國內期權數據源） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| GSMI / OPTION / BTCF | MSE/MAE | 未披露 | 未披露 | 未披露 |
| GSMI / OPTION / BTCF | msIC/msIR | 未披露 | 未披露 | 未披露 |

**解讀**：導讀明確指出「沒有一個模型在每個指標上表現出絕對優勢」，且僅給出定性結論（如 PSformer 在12個實例中最佳，Naive 誤差低但相關性差）。此處 Δ 欄全填「未披露」以遵守數字紀律。真 capability 在於 msIC/msIR 能過濾掉 Naive 模型那種「低誤差但無方向預測力」的偽信號；但導讀未提供實盤 Sharpe/IR/MDD 或交易成本調整後的淨值，當前 Δ 僅反映離線相關性對齊，未驗證策略層面的經濟價值。過擬合風險在於多步預測的誤差累積未受 msIC/msIR 直接懲罰，且分鐘級 OPTION 數據的實盤執行成本（滑點/手續費）未計入，前瞻性偏差風險存在於全局預處理錨定若未嚴格滾動計算。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：導讀未提供傳統 limitations 章節，但隱含指出：現有模型在金融非平穩低信噪比下無絕對優勢；msIC/msIR 未直接優化預測誤差；策略回測未計入交易成本與流動性約束。
**6.2 推斷的隱含假設**：
- **Regime 依賴**：msIR 依賴歷史樣本相關性分佈，若市場結構突變（如波動率 regime switch），std(msIC) 會飆升導致 msIR 失真。
- **容量/成本**：分鐘級 OPTION 與小時級 BTCF 策略容量極低，不適合大資金；離線評估未內建成本模型。
- **數據泄漏**：預處理使用全局對數錨定（減去初始值），若未嚴格按時間窗口滾動計算，會引入未來信息。
- **Survivorship**：GSMI 20個指數為當前活躍品種，未提及退市指數處理，存在生存者偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| DLinear / PatchTST | 純誤差優化 vs 相關性穩定性優化 | Open | 廣泛使用 |
| 傳統 IC/IR 因子 | 單步單變量 vs 多步多變量序列對齊 | Open | 標準實踐 |
| RL/Agent 框架 | 離線監督回歸 vs 在線序列決策 | Open | 探索期 |

🎤 **Interview Tip**：
- **正確答**：「msIC/msIR 不是替代 MSE/MAE，而是補齊金融預測中『方向一致性』與『信號穩定性』的評估維度；實盤需結合成本調整與滾動窗口防泄漏。」
- **錯答**：「用 msIC 替換 Loss 函數就能直接提升 Sharpe。」

**7.1 可證偽預測帶日期**：若 2025-Q3 前，主流 TS 模型在 GSMI 日頻數據上 msIC 提升但實盤多空組合淨值未顯著優於 Naive（扣除交易成本），則證明 msIC/msIR 與經濟價值存在脫鉤。

## §8 · For the Reader
- **因子研究員**：將 msIC/msIR 納入因子測試儀表板，替代單一 IC/IR，重點監控因子信號的跨期穩定性而非瞬時預測力。
- **高頻執行**：OPTION 分鐘級數據的 msIR 高值區間可作為流動性提供/撤單的過濾條件，但需對沖訂單簿衝擊成本。
- **組合配置**：BTCF 小時級多空策略的 msIC 序列可用於動態權重調整，但必須加入波動率目標與最大回撤閾值（導讀未給）。
- **研究學生**：復現時嚴格實現滾動時間窗口預處理，避免全局錨定導致的 look-ahead bias；將 msIR 作為 Early Stopping 的監控指標。

## References
- FinTSBridge 原論文（無 venue，2025-03-11）
- QuantML 導讀：[融合前沿深度学习模型的金融时序预测框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247489625&idx=1&sn=571070b27861b9058ffc9ee96a6ea4c6&chksm=ce7e7f47f909f6512ec3d1b249e59dcb2cf08bf3ef673c7544294e44ad46a2dfcf86b5d2a961#rd)
- Lineage: Autoformer, FEDformer, PatchTST, iTransformer, DLinear, PSformer, TiDE, Koopa, TimesNet, TSMixer, TimeMixer, Stationary, Informer, Crossformer, MICN, RevIN, Transformer.