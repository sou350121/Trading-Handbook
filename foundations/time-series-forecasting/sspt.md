<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# SSPT 解構

> **發布**：2025-08-16 · KDD 2025
> **QuantML 導讀**：[KDD 25 | 基于股票数据定制化预训练模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491386&idx=1&sn=cee770205d698016d6a5a7db359b1fc4&chksm=ce7e7824f909f132674506690f1e7fc07ed9d2b39217f955f3f82c738af477b20cfebb243b63#rd)
> **原始論文**：[SSPT: Secondary Structure Prediction Triangle](https://doi.org/10.1109/aiccsa.2009.5069410)（2009 IEEE/ACS International Conference on Computer Systems and Applications · 2009 · 被引 0 · Crossref）
> **核心定位**：落點於「端到端表征」軸，將預訓練重心從架構堆疊轉向目標函數設計。解了金融時間序列預訓練的 prior gap：通用掩碼預測與對比學習忽略股價非平穩性與截面身份異質性，導致表征停留在記憶隨機漫步而非提取底層統計特徵。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 針對日頻截面選股任務，設計三項定制化預訓練目標（代碼分類/板塊分類/移動均價預測）。② 核心 trick 是放棄預測單點價格，改用「分類識別+窗口均價平滑」繞過非平穩噪聲，配合分層微調策略。③ 這對端到端表征軸★的關鍵在於證明：預訓練的邊界由任務對齊度決定，而非 Transformer 深度。④ 導讀未給量化結果。

**X-Ray.** SSPT 將預訓練的 Pareto 前沿從「架構深度/對比樣本量」推向「目標函數對齊」。它解了兩個舊工程坑：掩碼值預測在隨機漫步中過擬合噪聲，以及對比學習忽略截面身份異質性。透過「分類識別+均價平滑」，模型被迫提取底層統計特徵而非記憶價格路徑。然而，它打不開的 envelope 很明確：日頻 Top-K 輪換的容量天花板與交易成本未建模，且 MA 特徵天然攜帶前瞻偏差風險。對量化讀者的意義在於，證明端到端表征的瓶頸不在層數，而在預訓練任務是否內嵌金融邏輯；實盤前必須將分類 logits 解耦為獨立因子，並嚴格驗證 regime 切換下的權重穩定性。

## §1 · 架構 / Core Mechanism
| 維度 | 前作/通用預訓練 | SSPT 改動 | 工程意圖 |
|---|---|---|---|
| 預訓練目標 | Masked Value Prediction / 對比學習 | Stock Code Classification (SCC) + Stock Sector Classification (SSC) | 強迫模型學習截面身份與板塊異質性，提取可區分的隱性模式 |
| 噪聲處理 | 直接預測掩碼點值 | Moving Average Prediction (MAP) | 用窗口均價平滑短期波動，規避非平穩序列的過擬合 |
| 參數遷移 | 全量微調或固定 backbone | 分層策略（Frozen / Fine-tuned / Scratch Head） | 保留通用表征的同時，為下游選股任務重建預測頭 |

⚡ **Eureka 一句話 trick**：不猜單點價格，改猜「這支股票是誰」與「這段區間均價」，用分類識別+趨勢平滑繞過隨機漫步噪聲。

```
[OHLCV + 5/10/20/30 MA] 
       ↓
[2-Layer Transformer Backbone]
       ↓
┌──────────────┬──────────────┬──────────────┐
│ SCC Head (CE)│ SSC Head (CE)│ MAP Head (MSE)│  ← Pre-training
└──────────────┴──────────────┴──────────────┘
       ↓ (Fine-tuning)
[Return Prediction Head] → 1-day Return → Rank Top-K
```

## §2 · 數學層
📌 **Napkin Formula**: `L_pre = w1*L_SCC + w2*L_SSC + w3*L_MAP` (CrossEntropy + CrossEntropy + MSE)。複雜度 `O(N*L^2*d)`（2層 Transformer，`N` 樣本，`L` 序列長，`d` 隱層維度）。
**直覺**: 分類任務迫使表征空間按統計參數分簇；MAP 將高頻噪聲積分為低頻趨勢，損失函數對隨機波動不敏感。
**Loss/訓練細節**: 預訓練多任務加權優化；微調階段 `L_fine = L_reg + λ*L_rank`。各任務最優凍結層數不同（SCC 不凍結、SSC 凍結嵌入層、MAP 極度敏感需驗證集調權）。

## §3 · 數據層
- **規模/頻率/市場**: 5 個數據集（NASDAQ / NYSE / FTSE-100 / TOPIX-100 / NASDAQ-recent），日頻。劃分為 3年訓練 / 1年驗證 / 1年測試。
- **特徵來源**: 標準 OHLCV + 5/10/20/30 日移動平均線。
- **樣本外與容量假設**: 假設日頻截面輪換 inefficiency 持續存在；未披露交易成本與滑價假設，容量上限未驗證。

## §4 · 代碼層
| 項目 | 狀態 |
|---|---|
| Repo | 知識星球（未公開 GitHub） |
| Checkpoint | 未披露 |
| License | 未披露 |
| 複現難度 | 中（需自行實現分類頭、MA 預測邏輯與分層凍結策略） |
| 數據可得性 | 高（標準日頻 OHLCV + MA 計算） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| NASDAQ / NYSE / TOPIX-100 | IRR / SR | 未披露 | 未披露 | 未披露 |
| FTSE-100 / NASDAQ-recent | IRR / SR | 未披露 | 未披露 | 未披露 |

**解讀**: 導讀僅給出定性描述（「顯著優於」「最佳或次佳」「持續跑贏」），未提供任何具體數值，故 Δ 欄嚴格留空。聲稱的 performance gain 可能來自目標函數對齊帶來的表征質量提升，但需警惕：① MA 特徵若計算不當易引入前瞻偏差；② 多任務權重高度依賴驗證集調優，存在過擬合特定市場 regime 的風險；③ 未計入換手成本與滑價，實盤 SR 可能大幅衰減。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**: 多任務同時訓練存在目標衝突；MAP 對損失權重極度敏感，魯棒性不足；微調策略需針對不同任務單獨搜索，缺乏統一自動化方案。
**6.2 推斷的隱含假設**: 
- Regime 依賴：假設股票/板塊的統計特徵分佈在訓練期與測試期穩定，未處理結構性斷點（如政策轉向、流動性枯竭）。
- 成本假設：隱含假設日頻 Top-K 輪換無摩擦，未建模容量限制與交易成本。
- 數據泄漏風險：MA 特徵若使用當日收盤價計算後再預測次日收益，需嚴格確認時間戳對齊，否則易產生 look-ahead bias。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| GATs / Graph-RNN | 架構驅動（圖結構/時空卷積） vs 目標驅動（定制化預訓練任務） | TBD | TBD |
| Informer / PatchTST | 長序列時效建模 vs 截面身份識別+趨勢平滑 | TBD | TBD |

🎤 **Interview Tip**: 
- ✅ 正確答：預訓練的價值不在於 Transformer 層數或對比樣本規模，而在於目標函數是否內嵌金融數據的非平穩性與截面異質性。SSPT 用分類與均價預測替代掩碼預測，是將「數學優化」轉向「金融邏輯對齊」的範式轉移。
- ❌ 錯答：只要加大對比學習 batch size 或加深 Transformer，模型就能自動學到穩健的股票表征。

**7.1 可證偽預測**: 若將 SSPT 直接遷移至 2023-2024 高波動/AI 驅動行情，其基於歷史 MA 與分類任務的表征將因 regime shift 失效，實盤 SR 將回落至未預訓練基線水平（預測驗證日期：2025-12-31）。

## §8 · For the Reader
- **因子研究員**: 將 SCC/SSC 的分類 logits 或中間層激活值解耦為截面因子，輸入線性模型或樹模型，可檢驗「身份異質性表征」是否攜帶獨立於動量/價量的 alpha。
- **高頻執行/組合配置**: 日頻輪換策略必須嚴格計入滑價與換手成本。導讀未披露成本敏感性，實盤前需做 breakeven turnover 分析，否則理論 SR 將被交易摩擦吞噬。
- **LLM-agent / RL 策略**: 此框架證明「任務設計 > 架構堆疊」。可將 MAP 的「平滑目標替代噪聲點預測」思想遷移至多模態預訓練或 RL 的 reward shaping，避免直接優化高頻隨機波動。

## References
- KDD 2025. *Stock Specialized Pre-trained Transformer (SSPT)*.
- QuantML 導讀: [KDD 25 | 基于股票数据定制化预训练模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491386&idx=1&sn=cee770205d698016d6a5a7db359b1fc4&chksm=ce7e7824f909f132674506690f1e7fc07ed9d2b39217f955f3f82c738af477b20cfebb243b63#rd)