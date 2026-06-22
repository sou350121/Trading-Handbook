---
title: "TISFM"
description: "落點於「中觀行業指數動態融合」的監督回歸黑盒。解決了傳統 SIP 模型在跨市場環境下，靜態拼接宏觀與微觀數據導致的異質性干擾與 regime 切換遲滯問題。"
---
<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

> **發布**：2026-01-14 · （無 venue）
> **QuantML 導讀**：[动态行业融合Transformer用于指数预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492996&idx=1&sn=d9bdee39404b728ea957726d4230c891&chksm=ce7d829af90a0b8cbc3d72ee62c51eba9ed06d652ab070221caff484b49f7e7cc3aa7366e637#rd)
> **核心定位**：落點於「中觀行業指數動態融合」的監督回歸黑盒。解決了傳統 SIP 模型在跨市場環境下，靜態拼接宏觀與微觀數據導致的異質性干擾與 regime 切換遲滯問題。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 將中觀行業指數作為動態橋樑，透過實時市場狀態預測生成權重，並行編碼不同體制下的行業空間特徵並自適應聚合。② 核心 trick 是「狀態感知並行編碼 + 雙向交叉注意力」，取代靜態拼接。③ 對「端到端表征」軸而言，它用離散 regime 權重軟化連續時間序列的結構斷點，降低高維異質特徵的過擬合風險。④ 導讀給出 SSEI MAE 1.4156 / DA 0.6335。

**X-Ray.** TISFM 的本質是將「行業輪動」從事後因子轉為實時狀態變量。傳統 Transformer 在 SIP 任務常陷入高維特徵的注意力稀釋，該架構用 K-means 預定義的 4 種 regime 權重做軟路由，強制模型在空間維度（行業間）與時間維度（市場趨勢）解耦編碼。這解了靜態融合帶來的共線性噪聲坑，但代價是 regime 邊界的離散化假設。其 envelope 打不開高頻微結構與外生政策衝擊的瞬時定價，對量化讀者的意義在於：提供了一套可插拔的「狀態門控」模塊，可與現有頻域/圖神經網絡基線組合，但需警惕離散聚類在尾部風險下的權重滯後。

## §1 · 架構 / Core Mechanism
1.1 三大改動 vs 前作
| 改動維度 | 前作/基線常見做法 | TISFM 改動 |
|---|---|---|
| 特徵融合路徑 | 靜態拼接或單向 Attention | 狀態感知並行編碼 + 雙向 Cross-Attention |
| 時間/空間處理 | 時空耦合編碼 | 空間（行業橫截面）與時間（市場序列）解耦並行 |
| 預測目標 | 單日收盤價/收益率 | 20日 SMA 對數差（平滑趨勢） |

1.2 ⚡ Eureka 一句話 trick + 直覺
用市場指數首時間步的編碼向量經 FFN+Softmax 生成 4 維 regime 權重，直接對 4 套並行行業 Transformer 的輸出做加權求和，讓模型「先看大勢，再挑板塊」。

1.3 信息流 ASCII 圖
```
[市場指數序列] -> 線性投影+PE -> [Transformer Encoder] -> H_market_ctx
     |
     v (取首步向量) -> FFN -> Softmax -> W_state (4維)
[行業指數序列] -> 線性投影+PE -> H_industry
     |
     v (複製4份) -> [4x Parallel Transformer Encoders] -> H_industry_regime_i
     |
     v (加權求和 W_state) -> H_industry_agg -> [Industry Temporal Encoder] -> H_industry_ctx
     |
     v (Bidirectional Cross-Attention: H_market_ctx <-> H_industry_ctx) -> H_fused
     |
     v (Residual + H_market_ctx) -> MLP -> SMA Log Diff Prediction
```

## §2 · 數學層
📌 Napkin Formula：
$W_{state} = \text{Softmax}(\text{FFN}(H_{market\_ctx}^{[0]}))$
$H_{industry\_agg} = \sum_{i=1}^{4} W_{state}^{(i)} \cdot \text{Enc}_i(H_{industry})$
複雜度：$O(L \cdot d^2)$ 自注意力為主，並行編碼不增加漸進複雜度，僅常數倍算力。
直覺：權重 $W$ 將連續市場動態投影到離散 regime 空間，加權聚合實作「軟切換」，避免硬閾值帶來的梯度斷裂。
Loss/訓練：導讀未披露具體 loss 函數與優化器細節，推測為 MSE/MAE 回歸損失，Adam 優化。

## §3 · 數據層
- 規模/頻率/市場：日頻。中國A股（SSEI）、美股（S&P 500）、港股（HSI）。
- 時段：2014-01-01 至 2023-12-31。
- 來源：Yahoo Finance。
- 特徵工程：54項技術指標 -> RF 重要性篩選 -> 12項代表特徵。行業指數：中國12個申萬一級，美國11個GICS，香港9個恒生分類。
- 樣本外與容量假設：導讀未明確劃分 train/val/test 區間與滑動窗口策略。容量假設受限於日頻與中長週期，適合中低頻組合配置，不支撐高頻交易。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | QuantML 知識星球（未公開 GitHub） |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中高（需自行爬取 Yahoo Finance 數據、實現 RF 特徵篩選與 K-means 預聚類） |
| 數據可得性 | 公開（Yahoo Finance），但行業指數代碼映射需手動對齊 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA (逐字) | 本方法 | Δ |
|---|---|---|---|---|
| SSEI (中國) | MAE | 未披露 | 1.4156 | 未披露 |
| SSEI (中國) | MSE | 未披露 | 3.1323 | 未披露 |
| SSEI (中國) | RMSE | 未披露 | 1.7698 | 未披露 |
| SSEI (中國) | DA | PatchTST 0.6477 | 0.6335 | -0.0142 |
| S&P 500 (美國) | MAE | 未披露 | 2.2245 | 未披露 |
| S&P 500 (美國) | MSE | 未披露 | 6.8160 | 未披露 |
| S&P 500 (美國) | DA | 未披露 | 0.6923 | 未披露 |
| HSI (香港) | MAE | 未披露 | 3.5096 | 未披露 |
| HSI (香港) | DA | 未披露 | 0.5385 | 未披露 |

**解讀：** Δ 僅在 SSEI DA 欄可計算，TISFM 落後 PatchTST 0.0142，但導讀指出 PatchTST 誤差指標較高，顯示 TISFM 在「精度-方向」權衡上更穩。其餘 Δ 為「未披露」，無法斷言絕對領先。部分增益可能源於預測目標重定義（SMA 對數差天然平滑噪聲）而非架構本身，且未計入交易成本與滑點，實盤 Sharpe/IR 可能收斂。

## §6 · 失效與隱含假設
6.1 論文自述 limitations：香港市場 DA 較低（0.5385），導讀歸因於國際資本流動與政策變化帶來的極高波動性，當前特徵集未能完全捕捉短期方向信號。
6.2 推斷的隱含假設：
- Regime 依賴：K-means 預聚類假設市場狀態具備穩定離散結構，在極端尾部事件（如閃崩、流動性枯竭）下權重生成可能滯後或錯配。
- 容量/成本：日頻中長週期，容量較大，但 SMA 對數差目標導致信號滯後，實盤需考慮換倉頻率與摩擦成本。
- 數據泄漏：RF 特徵篩選與 K-means 若未嚴格按時間序列劃分（TimeSeriesSplit），易引入前瞻偏差。
- Survivorship：Yahoo Finance 歷史數據通常含已退市成分，需確認是否經 survivorship bias 校正。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| PatchTST | 時空解耦 vs 狀態門控路由 | Yes | SOTA 基線 |
| FEDformer | 頻域增強 vs 離散 regime 加權 | Yes | SOTA 基線 |
| 傳統多因子模型 | 靜態線性/樹模型 vs 動態非線性融合 | N/A | 成熟 |

🎤 Interview Tip:
- 正確答：「TISFM 的核心不是單純堆疊 Attention，而是用市場狀態權重做軟路由，解決行業特徵高維異質性帶來的注意力稀釋。實盤需重點驗證 regime 聚類的時間穩定性與權重滯後對尾部風險的暴露。」
- 錯答：「它比 Transformer 好是因為用了交叉注意力，所以預測更準。」（忽略狀態感知與目標平滑的貢獻，混淆架構創新與數據工程）

7.1 可證偽預測帶日期：若 2026-06-30 前，在未經 survivorship bias 校正的 A 股日頻數據上復現，其 DA 將跌破未披露水平且 MAE 波動率顯著偏離導讀數值，則證明其增益高度依賴數據清洗與目標平滑。

## §8 · For the Reader
- **因子研究員**：將 `W_state` 視為動態因子暴露權重，可提取其時間序列與傳統動量/價值因子做正交性檢驗，避免特徵共線。
- **組合配置**：SMA 對數差目標適合中週期倉位調整，建議結合波動率目標（Vol-Targeting）控制換倉頻率，摩擦成本是實盤成敗關鍵。
- **LLM-agent / 研究學生**：此架構可作為「狀態門控」模塊的範本，嘗試將 K-means 替換為線上 HMM 或 RL 策略，實現 regime 的動態發現而非預定義。

## References
- 原論文：TISFM (無 venue, 2026)
- QuantML 導讀：[动态行业融合Transformer用于指数预测](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492996&idx=1&sn=d9bdee39404b728ea957726d4230c891&chksm=ce7d829af90a0b8cbc3d72ee62c51eba9ed06d652ab070221caff484b49f7e7cc3aa7366e637#rd)
- Lineage: PatchTST (ICLR 2023) / FEDformer (ICML 2022) / K-means Regime Switching in Finance