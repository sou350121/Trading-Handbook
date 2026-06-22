---
title: "MASS"
description: "落點於「生成式大模型 × 多智能体博弈」軸，將 LLM 從「預設流程預測器」重構為「端到端分佈優化器」，解決了傳統智能體模擬無法直接對接組合決策與動態適配市場風格切換的 Prior Gap。"
---
<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=生成式大模型 alpha=多智能体博弈 autonomy=Agent自主演进 -->

> **發布**：2025-07-05 · （無 venue）
> **QuantML 導讀**：[北京大学 & 正仁量化 | MASS：基于多智能体规模化模拟投资组合](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490927&idx=1&sn=3720c677c02ccda54e395d650f3c8200&chksm=ce7e7a71f909f3670a54f170ff36e80f5c0c1cb1ea617264ffafbd0772ab44a5ae5d4387326d#rd)
> **核心定位**：落點於「生成式大模型 × 多智能体博弈」軸，將 LLM 從「預設流程預測器」重構為「端到端分佈優化器」，解決了傳統智能體模擬無法直接對接組合決策與動態適配市場風格切換的 Prior Gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `生成式大模型` | `多智能体博弈` | `Agent自主演进` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 以 512 個異質 LLM Agent 模擬市場微結構，直接輸出組合權重而非單邊預測；② 核心 Trick 為「市場分歧假說」信號聚合 + 每日在線反向優化智能體分佈；③ 對「Agent自主演進」軸★，證明多智能體規模定律可驅動 RankIC 線性增長；④ 導讀未給量化結果。

**X-Ray.** 置於五軸 Pareto 前沿，MASS 用「分佈搜索」替代「提示詞工程」，直接擊穿 LLM 上下文窗口與固定工作流的工程瓶頸。其反向優化機制將市場風格切換轉化為分佈權重的梯度更新，具備強 regime 適應性。但該框架打不開的 envelope 在於：未內建交易成本與流動性約束，512 個 Agent 每日前向推理的延遲與 API 成本在實盤中將構成硬性閾值；且「分歧-共識」線性組合本質是低頻因子合成，無法捕捉高頻訂單流衝擊。對量化讀者而言，其價值不在於直接實戰，而在於提供了一套可微的 Agent 分佈調優範式，可與傳統風險模型解耦後作為信號生成器。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作/傳統做法 | MASS 改動 | 工程意義 |
|---|---|---|---|
| 決策導向 | 模擬研究（脫離實盤）或應用研究（固化流程） | 聚合信息直接驅動組合構建 | 打通 LLM 輸出到交易執行的最後一哩路 |
| 優化機制 | 依賴先驗假設與固定 Prompt Chain | 端到端反向優化動態調整分佈 | 消除人工特徵工程偏差，實現數據驅動自適應 |
| 規模擴展 | 智能體數量通常不超過 64 個 | 擴展至 512 個並驗證規模定律 | 突破單 Agent 上下文瓶頸，以數量換取市場理解深度 |

⚡ **Eureka Trick:** 用模擬退火每日微調 Agent 分佈權重，讓市場「自己教自己」如何聚合分歧，而非依賴靜態規則。

**信息流 ASCII:**
```
[Macro, Firm Features] 
       ↓
LLM Agent Pool (16 types × 32 agents) → 隨機分配觀察池
       ↓
Forward: Strategy Gen → Stock Pick → 分散決策信號
       ↓
Score Aggregation: Consensus/Disagreement 線性組合
       ↓
Backward Opt: 滑動窗口內最大化 Signal-Return 相似度 → 更新分佈 W
       ↓
Portfolio Output (日頻調倉)
```

## §2 · 數學層
📌 **Napkin Formula:**
$S_{t,i} = \alpha \cdot \text{Consensus}_{t,i} + (1-\alpha) \cdot \text{Disagreement}_{t,i}$
$\max_{\mathbf{w}} \text{Sim}(\text{Signal}(\mathbf{w}), \text{Return})$
**複雜度:** $O(N_{agents} \times T_{days} \times \text{LLM\_inference\_cost})$，每日優化循環依賴模擬退火，非梯度下降。
**直覺:** 將組合構建轉化為分佈參數的在線學習問題。高共識低分歧資產被賦予更高權重，分歧項用於捕捉異質信念導致的估值扭曲。
**Loss/訓練細節:** 每日提取回顧窗口內的實現收益與構建信號，計算相似度分數作為優化目標；優化器為模擬退火，無顯式損失函數，屬黑盒搜索優化。

## §3 · 數據層
**規模/頻率/市場/時段:** 2023全年A股（SSE 50 / CSI 300 / ChiNext 100）+ 2025第一季度（CSI A500）；日頻；開盤 15 分鐘內買入並持有 T 天。
**來源:** 手動收集構建公司層面特徵與宏觀指標。
**樣本外與容量假設:** 使用 2025 全新數據排除模型記憶；容量假設依賴日頻調倉節奏與 LLM API 吞吐量，未討論滑點與衝擊成本對規模的約束。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 高（需大規模 LLM API 調度、每日優化循環與特徵工程） |
| 數據可得性 | 低（手動構建特徵與宏觀數據，未開源） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| SSE 50 / CSI 300 / ChiNext 100 | IC / ICIR / RankIC / RICIR | LightGBM / DTML / SEP / FINCON / Trading Agents / 傳統代理指標：未披露 | 未披露 | 未披露 |
| CSI A500 (2025 Q1) | IC / ICIR / RankIC / RICIR | 未披露 | 未披露 | 未披露 |

**解讀:** 導讀未給出任何具體數值，Δ 欄無法計算。從機制推斷，RankIC 的「近似線性增長」反映的是信號排序能力的穩定性提升，屬真 capability；但「更低的回撤」與「累計超額收益」若未計入交易成本與滑點，極可能為前瞻偏差或過擬合。多智能體分解雖緩解了 LLM 處理海量信息的瓶頸，但每日 512 次推理的延遲在實盤中會直接侵蝕日頻策略的 Alpha。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 數據集多樣性待提升（缺多模態如研報/K線圖）；未整合擇時策略與倉位控制。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 隱含假設市場分歧與未來收益呈穩定線性關係，在流動性枯竭或極端恐慌 regime 下可能失效。
- **容量/成本:** 未建模交易成本，日頻調倉的 API 成本與執行滑點將構成實盤硬性閾值。
- **數據泄漏:** 特徵工程依賴手動構建，宏觀數據發布時間若未嚴格對齊交易日，存在潛在泄漏。
- **Survivorship:** 股票池為指數成分股，通常含退市股處理，但導讀未明確說明是否使用生存者偏差校正。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| SEP / FINCON | 固定工作流/自我反思 vs 端到端分佈優化 | TBD | v0.5 |
| 傳統因子模型 | 靜態權重/線性回歸 vs 動態智能體分佈搜索 | 部分開源 | 成熟 |

🎤 **Interview Tip:** 
- **正確答法:** 聚焦「分佈優化替代提示詞工程」與「規模定律驗證」，指出 MASS 本質是將組合構建轉化為分佈參數的在線學習問題，而非單邊預測。
- **錯答法:** 將 MASS 等同於傳統 LLM 預測股價，或混淆為 RL 訓練（實為模擬退火黑盒搜索，無 Reward Model 或 Policy Gradient）。

**7.1 可證偽預測帶日期:** 若 2025-12-31 前未開源代碼與完整特徵字典，該框架將停留在學術模擬階段，無法進入機構實盤驗證。

## §8 · For the Reader
- **因子研究員:** 提取「共識-分歧」線性組合邏輯，轉化為傳統量價因子，避開 LLM 推理成本。
- **高頻執行:** 該框架為日頻低頻策略，不適用；需關注調倉成本與執行延遲對 Alpha 的侵蝕。
- **組合配置:** 將 MASS 視為信號生成模塊，與風險模型（如 Barra）解耦後進行權重分配，利用其動態分佈特性做風格輪動。
- **LLM-agent/RL 策略:** 借鑒「反向優化分佈」機制，替代固定的 Prompt Chain，探索非梯度優化器在 Agent 調度中的應用。
- **研究學生:** 復現 512 Agent 規模定律實驗，驗證優化器（模擬退火 vs 梯度下降）的收斂性與超參數敏感性。

## References
- 原論文：MASS: Multi-Agent Scalable Simulation for Portfolio Construction（無 venue / arxiv=None）
- Lineage: LLM Multi-Agent Systems / Portfolio Optimization / Scaling Laws in Agents
- QuantML 導讀鏈接：[北京大学 & 正仁量化 | MASS：基于多智能体规模化模拟投资组合](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490927&idx=1&sn=3720c677c02ccda54e395d650f3c8200&chksm=ce7e7a71f909f3670a54f170ff36e80f5c0c1cb1ea617264ffafbd0772ab44a5ae5d4387326d#rd)