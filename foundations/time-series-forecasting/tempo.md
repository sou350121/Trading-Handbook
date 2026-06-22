<!-- ontology-5axis data=多模态 horizon=跨周期 paradigm=生成式大模型 alpha=端到端表征 autonomy=全自动黑盒 -->

# TEMPO 解構

> **發布**：2024-09-26 · ICLR 2024
> **QuantML 導讀**：[ICLR 24 | 基于提示词的生成预训练Transformer时间序列预测框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486820&idx=1&sn=de9fd547b916d5a9341c2baba4525864&chksm=ce7e6a7af909e36cf81eb809d72fd2aabddca22728245f4157443ba9ac983538065985920b4e#rd)
> **核心定位**：五軸落點於「生成式大模型 × 端到端表征 × 全自动黑盒」，解了傳統時序模型對趨勢/季節性/殘差交互建模的歸納偏置缺失，以及跨域分佈漂移時的提示適配痛點。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `多模态` | `跨周期` | `生成式大模型` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出 TEMPO 框架，將時序分解為趨勢/季節性/殘差，並為各分量設計半軟提示詞接入 GPT 解碼器。② 核心 trick 是分量級提示拼接 + LoRA 適配分佈，實現跨域零樣本泛化。③ 對「生成式大模型」軸★，證明 LLM 架構無需重訓即可透過結構化解耦與輕量適配接管時序表征。④ 在多個基準上 MAE 較前 SOTA 分別提升 6.5% 與 19.1%（原文數據）。

**X-Ray.** TEMPO 本質是將時序預測從「純序列擬合」推向「結構化解耦 + 提示適配」，在五軸 Pareto 上以推理成本換取跨域泛化。它解決了舊工程坑：傳統 Transformer/CNN 對季節性與趨勢的耦合建模容易在跨域時失效，TEMPO 透過分解+半軟提示，把分佈漂移的適配成本壓低到 LoRA 層級。預測它打不開的 envelope：高頻/低信噪比金融數據的實盤穩定性。時序分解在低頻/高信噪比（電力、氣象）有效，但金融價格序列的趨勢/季節性極弱且非平穩，分解可能引入偽信號或前瞻偏差。對量化讀者意義：不適合直接當 Alpha 生成器，但可作為多模態因子融合或 regime 識別的表征提取器；提示設計思路可移植到傳統時序模型做輕量適配。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作 (Time-Series Transformer / LLM-for-TS) | TEMPO |
|---|---|---|
| 輸入表征 | 原始序列直接 Embedding 或簡單 Patch | 趨勢/季節性/殘差三分量解耦 + 分量級半軟提示拼接 |
| 適配機制 | 全參數微調或硬提示 | 僅更新位置嵌入/層歸一化 + LoRA 適配分佈 |
| 學習設定 | 一對一/一對多監督學習 | 多對一零樣本（跨域預訓練 → 未見目標集） |

**1.2 ⚡ Eureka 一句話 trick + 直覺**
把時序「拆開餵」給 GPT，讓每個分量帶專屬提示詞，模型只需學「怎麼組合提示」而非「重記數據分佈」。

**1.3 信息流 ASCII 圖**
```
X (Raw TS) → Decompose → [Trend, Seasonality, Residual]
                ↓
Prompt Gen → [P_t, P_s, P_r] (Semi-soft)
                ↓
Concat → [T+P_t, S+P_s, R+P_r] → GPT Decoder Blocks
                ↓
Update: PosEmb, LayerNorm + LoRA adapters
                ↓
Output → Ŷ (H-step forecast)
```

## §2 · 數學層
**📌 Napkin Formula：**
$X \in \mathbb{R}^{n \times L} \xrightarrow{\text{Decomp}} X_T, X_S, X_R$
$\hat{Y} = \text{GPT}_\Phi(\text{Concat}([X_T, P_T], [X_S, P_S], [X_R, P_R]))$
複雜度：$O(L \cdot d^2)$ 自注意力 + LoRA 降秩適配 $O(r \cdot d)$，訓練僅更新 PosEmb/LN/LoRA 參數。

**直覺：** 分解將非平穩序列轉為相對平穩分量，提示詞提供任務/域先驗，GPT 專注於分量間的非線性交互。Loss 為標準回歸 MSE/MAE，訓練採零樣本多源預訓練 + 目標集推理（無目標集梯度更新）。訓練超參（optimizer/LR/epochs）未披露。

## §3 · 數據層
- **資料規模/頻率/市場/時段**：電力 (ETTh1/2, ETTm1/2, 15min/1h)、交通 (Traffic, 加州高速)、氣象 (Weather, 德國 21 指標)、新聞 (GDELT)、金融 (TETS)。頻率從分鐘到季度不等。
- **怎麼來**：公開基準數據集 + TETS（結合上下文與時序的新基準）。
- **樣本外與容量假設**：零樣本設定（多源訓練 → 單目標未見集推理）。未披露具體樣本數、訓練/驗證/測試劃分比例、數據清洗與缺失值處理細節。

## §4 · 代碼層
| 項目 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | 未披露 |
| 複現難度 | 中高（需自實現時序分解 + 半軟提示拼接 + LoRA 掛載至 GPT 解碼器） |
| 數據可得性 | 高（ETT/Traffic/Weather 為公開基準，TETS 需確認開源狀態） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| ETT/Traffic/Weather/Electricity | MAE | 未披露 | 未披露 | 提升 6.5% / 19.1%（原文表述） |
| ETT/Traffic/Weather/Electricity | MSE | 未披露 | 未披露 | 未披露 |
| TETS (多模態) | SMAPE | 未披露 | 未披露 | 超越所有基線（原文表述） |

**解讀：** Δ 主要來自零樣本跨域泛化能力與多模態融合，但基準均為低頻/高信噪比物理/社會數據。金融 TETS 雖標註「金融」，實為新聞摘要+時序，非傳統量價數據。MAE 提升可能部分來自分解對季節性數據的先天優勢，而非模型架構的絕對泛化。未計推理成本（GPT 解碼器 + 提示拼接）與過擬合風險（零樣本設定下目標集無梯度，但提示設計可能隱含數據窺探）。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**
未明確列出 limitations 章節（導讀未提及）。SHAP 分析指出季節性影響顯著，且預測長度增加時準確性潛在下降。

**6.2 推斷的隱含假設**
- **Regime 依賴**：強依賴趨勢/季節性可分離的平穩或週期性數據，對金融價格的隨機漫步/結構斷點極度敏感。
- **容量/成本**：GPT 解碼器 + 三分量提示拼接，推理延遲與顯存佔用高於專用時序模型（如 PatchTST/iTransformer），不適合低延遲實盤。
- **數據泄漏**：零樣本「多對一」預訓練若未嚴格隔離目標域，提示設計可能過擬合特定分佈特徵。
- **Survivorship/前瞻**：分解算法（如 STL/移動平均）若未嚴格使用滾動窗口，易引入前瞻偏差。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Time-LLM / LLM4TS | 全序列 Embedding vs 分量解耦+半軟提示 | 開源 | 活躍 |
| PatchTST / iTransformer | 專用時序架構 vs 生成式預訓練模型適配 | 開源 | 活躍 |
| TimesFM / Timer | 基礎模型零樣本 vs 提示驅動輕量適配 | 開源/閉源 | 活躍 |

**🎤 Interview Tip**
- **正確答**：TEMPO 的核心貢獻是「結構化解耦 + 提示適配」，將時序預測的歸納偏置從序列依賴轉為分量交互，適合低頻多模態表征提取，但推理成本高且對非平穩金融數據泛化存疑。
- **錯答**：「TEMPO 是金融 Alpha 生成器，直接替換 Transformer 就能跑實盤。」（忽略分解假設、推理延遲與零樣本設定限制）

**7.1 可證偽預測帶日期**
若 2025-Q2 前無開源代碼/Checkpoint 發布，或實盤回測顯示在 A 股/加密貨幣日頻數據上 Sharpe < 0.5，則其「端到端表征泛化」假設在低信噪比市場失效。

## §8 · For the Reader
- **因子研究員**：將「半軟提示」思路移植至傳統時序模型（如 LSTM/Transformer），做輕量域適配；避免直接將時序分解用於價格序列，改用殘差/波動率分解。
- **高頻執行**：不適用。GPT 解碼器 + 提示拼接的推理延遲無法滿足微秒/毫秒級需求，建議關注輕量化時序基礎模型。
- **組合配置**：可將 TEMPO 的輸出作為宏觀/行業趨勢的輔助信號，與風險模型結合；需嚴格驗證分解算法的滾動視窗以避免前瞻偏差。
- **LLM-agent / RL 策略**：提示設計可作為 Agent 的「環境狀態編碼器」，結合 RL 做動態權重分配；但需解決 GPT 輸出確定性與梯度回傳問題。
- **研究學生**：重點複現「分量級提示拼接 + LoRA 適配」管道，對比硬提示/軟提示/無提示的零樣本表現，量化分解帶來的信息增益。

## References
- 原論文：TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting (ICLR 2024)
- Lineage: Time-LLM / LLM4TS / PatchTST / TimesFM
- QuantML 導讀鏈接：[ICLR 24 | 基于提示词的生成预训练Transformer时间序列预测框架](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486820&idx=1&sn=de9fd547b916d5a9341c2baba4525864&chksm=ce7e6a7af909e36cf81eb809d72fd2aabddca22728245f4157443ba9ac983538065985920b4e#rd)