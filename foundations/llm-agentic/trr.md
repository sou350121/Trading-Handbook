<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=风险择时 autonomy=人机协同可解释 -->

# TRR 解構

> **發布**：2024-10-24 · （無 venue）
> **QuantML 導讀**：[LLMs在股票投资组合崩溃中的时间关系推理](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487194&idx=1&sn=5b871784d4777989432d607b36f6ec2e&chksm=ce7e69c4f909e0d258d6c6f18752d9af9d3694a3121f93ac9066748b694f02c662d98ac5dedd#rd)
> **核心定位**：將 LLM 零樣本推理與動態關係圖結合，解決傳統靜態知識圖譜無法捕捉「黑天鵝」事件下跨資產傳染路徑的 prior gap。落點於「文本另类 × 日频波段 × 生成式大模型」軸，以人機協作可解釋架構重構風險擇時信號。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 提出 TRR 框架，利用 LLM 零樣本推理動態構建新聞-股票影響圖，實現日频投資組合崩潰檢測。② 核心 trick 是將推理拆解為「腦暴建圖 → 記憶模塊存時序上下文 → PageRank 注意力過濾關鍵路徑 → 零樣本崩潰推理」，突破傳統靜態關係與純文本提示局限。③ 這對「風險擇時」軸★的意義在於：將 LLM 從「單篇新聞情緒打分器」升級為「跨資產傳染路徑追蹤器」，直接對沖尾部風險。④ 關鍵實證數字：AUROC 平均比最強基線（ToG）高出 10.6%。

**X-Ray.** 在「文本另类 × 生成式大模型」的 Pareto 前沿上，TRR 切中了舊工程坑：傳統關係型預測依賴中心數據庫的靜態三元組，或純 LLM prompt 的單點語義，兩者皆無法建模「事件發生後跨資產影響力的時序擴散與衰減」。TRR 用 Memory + PageRank 將圖結構壓縮為 LLM 可消化的上下文子圖，實質是將「非結構化新聞流」轉譯為「可追蹤的風險傳染拓撲」。對量化讀者而言，這不是一個直接可交易的 Alpha 生成器，而是一個高頻/日频風險監控的「可解釋前置過濾器」。但它的 envelope 打不開：LLM 推理延遲與 API 成本使其無法用於毫秒級執行；圖擴散深度受 prompt token 限制，長尾實體鏈接易斷裂；且崩潰標籤依賴固定 -2.0% 閾值，在波動率 regime 切換時會產生偽信號。它真正價值在於為組合配置器提供「壓力測試路徑可視化」，而非替代統計擇時。

## §1 · 架構 / Core Mechanism
1.1 三大改動 vs 前作
| 維度 | 前作 (ToG/GoT/靜態KG) | TRR 改動 | 工程意義 |
|---|---|---|---|
| 關係建模 | 靜態知識圖譜/單次搜索 | 動態腦暴建圖 (Brainstorming) | 捕捉前所未有的事件傳染鏈 |
| 時序處理 | 無或簡單拼接 | 記憶模塊 (Memory) + 指數衰減 | 解決新聞影響力隨時間淡忘問題 |
| 上下文壓縮 | 單一路徑/合併思想 | PageRank 注意力過濾 (Top-q) | 突破 LLM token 限制，保留關係拓撲 |

1.2 ⚡ Eureka 一句話 trick + 直覺
用 PageRank 在時序影響圖上跑分，只把得分最高的 q 個實體鏈路餵給 LLM 做最終零樣本判斷，本質是「圖結構的注意力機制替代了暴力 prompt 拼接」。

1.3 信息流 ASCII 圖
新聞文章 x_j
   ↓ (Brainstorming 迭代)
子實體 e_h → 影響鏈 C_en
   ↓ (Memory 檢索/存儲 + 指數衰減)
時序圖 G_temporal = ∪(G ∪ M_en)
   ↓ (PageRank 打分 → 過濾 Top-q)
關鍵子圖 G_TTR
   ↓ (Zero-shot Inference Prompt)
崩潰概率 P(crash) ≤ -2.0%

## §2 · 數學層
📌 Napkin Formula:
G_temporal = ∪_{v_en ∈ Z} (G ∪ M_en)
PR(e_n) ∝ Σ_{e_i ∈ In(e_n)} (PR(e_i) / |Out(e_i)|) × decay(M_en)
複雜度：圖構建 O(I·J·k)，PageRank 收斂 O(V·E)，推理 O(1) per day。
直覺：記憶模塊 M_en 累積歷史影響鏈，PageRank 迭代傳遞分數並按記憶權重加權，最終截取 Top-q 路徑形成壓縮圖 G_TTR。Loss/訓練細節：無傳統梯度下降，屬零樣本推理框架，依賴 prompt 工程與 API 調用，溫度設為 0.0 以確保確定性。

## §3 · 數據層
資料規模/頻率/市場/時段：路透社金融新聞數據集（擴展至 2020 年），日频（Daily）。具體文章數量與股票池規模未披露。
怎麼來：直接抓取未經手動過濾的通用金融新聞，由 LLM 自行判斷與目標投資組合的相關性。
樣本外與容量假設：標籤定義為投資組合日回報率 ≤ -2.0%（歷史第 95 百分位）。樣本外驗證細節未披露，假設為時間序列劃分。容量受 LLM API 限流與每日推理成本制約，適合機構級中低頻監控，不適合散戶實時交易。

## §4 · 代碼層
| 項目 | 狀態/細節 |
|---|---|
| Repo | TBD（導讀提及「論文及代碼下載見星球」，未公開 GitHub） |
| Checkpoint | GPT-3.5-turbo (OpenAI API) |
| License | 未披露 |
| 複現難度 | 中高（依賴 OpenAI API 成本與 Prompt 調優，圖構建邏輯需自編） |
| 數據可得性 | 路透社金融新聞需商業授權或特定數據源，Yahoo Finance 價格數據公開 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Reuters News + Yahoo Prices (Global/US TBD) | AUROC | ToG (未披露具體值) | TRR | +10.6% |
| 同數據集 | IR/Sharpe/AR/MDD | 未披露 | 未披露 | 未披露 |
解讀：Δ +10.6% 在 AUROC 上屬顯著提升，主要來自「動態圖路徑過濾」對噪声新聞的降噪能力。但需警惕：① AUROC 對極端不平衡數據（崩潰 rare event）較穩健，但未披露 Precision/Recall/F1，可能以高 False Positive 換取高 TPR；② 未計入 LLM API 調用成本與推理延遲；③ 標籤閾值 -2.0% 為靜態歷史分位，未考慮波動率 regime 切換，存在前瞻偏差風險。

## §6 · 失效與隱含假設
6.1 論文自述 limitations：組件可進一步專業化擴展；宏觀危機檢測需引入更多統計指標（政府債務、貿易流動等）構建集成系統。
6.2 推斷的隱含假設：
- Regime 依賴：假設新聞影響力衰減服從指數規律，但在流動性危機或閃崩中可能失效。
- 容量/成本：每日全量新聞腦暴 + PageRank + LLM 推理，API 成本與延遲不適合高頻。
- 數據泄漏：記憶模塊 M_en 每日更新，若訓練/測試劃分未嚴格按時間切分，易引入未來信息。
- Survivorship：未提及股票池是否處理退市/並購樣本，Yahoo Finance 歷史數據通常含 survivorship bias。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| ToG (Think-on-Graph) | 靜態搜索 vs 動態時序記憶+PageRank過濾 | 開源 | SOTA Baseline |
| BiGRU+Attention | 序列建模 vs 圖結構零樣本推理 | 開源 | 過時/低效 |
| 傳統風險模型 (Historical VaR/CVaR) | 純價格統計 vs 新聞事件驅動拓撲 | 開源 | 互補 |
🎤 Interview Tip:
- 正確答：「TRR 的本質是將 LLM 的零樣本能力與圖注意力結合，解決靜態知識圖譜無法捕捉『前所未有事件』的傳染路徑問題。其核心價值在於風險監控的可解釋性與壓力測試路徑可視化，而非直接生成交易信號。」
- 錯答：「TRR 是一個可以直接實盤交易的 Alpha 因子模型，因為它比 ToG 準確率高 10.6%。」（忽略 AUROC 性質、成本、延遲與不平衡數據陷阱）
7.1 可證偽預測帶日期：若 2025-12-31 前，機構將 TRR 架構直接接入實盤日频組合再平衡，且未對 API 成本與推理延遲進行工程優化，則其淨夏普比率將因交易成本與滑點轉為負值。

## §8 · For the Reader
- **因子研究員**：將 TRR 的圖過濾邏輯視為「事件驅動因子的拓撲增強版」，可嘗試用靜態圖神經網絡 (GNN) 替代 LLM 腦暴以降低成本，但需驗證動態關係捕獲能力。
- **組合配置/風險經理**：TRR 最適合做尾部風險的「可解釋壓力測試儀」。將生成的 G_TTR 圖可視化，用於合規報告與極端情景推演，而非直接觸發減倉。
- **LLM-Agent/RL 策略開發者**：注意 TRR 的 Memory 模塊設計可遷移至多智能體協作中的「共享工作記憶」。但需警惕 Prompt 注入風險與 API 限流，建議先用開源小模型 (如 Qwen2.5-7B) 做本地化腦暴驗證。

## References
- 原論文：Temporal Relational Reasoning in LLMs for Stock Portfolio Crashes (2024)
- Lineage: ToG (Think-on-Graph) → GoT (Graph of Thoughts) → CoT → Standard IO Prompt
- QuantML 導讀鏈接：[LLMs在股票投资组合崩溃中的时间关系推理](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247487194&idx=1&sn=5b871784d4777989432d607b36f6ec2e&chksm=ce7e69c4f909e0d258d6c6f18752d9af9d3694a3121f93ac9066748b694f02c662d98ac5dedd#rd)