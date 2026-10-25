<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=强化学习 alpha=组合执行优化 autonomy=全自动黑盒 -->

# When News Gets Old: Port 解構（When News Gets Old: Port）

> **來源發布**：2026-10-22 · （無 venue） · arXiv [2610.30564](https://arxiv.org/abs/2610.30564)
> **arXiv 原文**：[When News Gets Old: Portfolio Allocation with Reinforcement Learning and Large Language Models](https://arxiv.org/abs/2610.30564v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸分類桶為「文本另类 × 日频波段 × 强化学习 × 组合执行优化 × 全自动黑盒」；其中「日频波段」是本手冊的既有分類標籤，本研究實際採週頻配置／再平衡，不代表每日換倉。它解的是 LLM 抽出 SEC 營運信號後「舊新聞該保留、告知時點、還是硬過濾」這個介面選擇，而不是發明新的 alpha 因子。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `强化学习` | `组合执行优化` | `全自动黑盒` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①固定 GPT-4.1 mini 從 SEC 揭露抽 `z∈{-1,0,1}`，把「新聞老化」拆成 news age 與 time since arrival，並在 RL 與 direct LLM 兩種決策器下比較 News、News+Age、Filtered News。②核心 trick 不是讓 policy 自己學會忽略舊新聞，而是把「超過 8 週即歸零」當成一條可消融的 rule-based 正則化。③對文本＋RL 這條軸，關鍵發現是「RL 對時點有反應」與「報酬改善」是兩件事：12 週老化下 RL 選擇改變 52% of cases，但歷史淨報酬無一致增益。④LLM 直接決策下 8 週過濾在 4 週延遲給 18.43% 年化淨報酬，高於 News 的 16.02% 與 News+Age 的 17.17%；但 RL Market Data 的 19.07% 仍高於此值。

**X-Ray.** 這篇的價值不在「LLM 打敗 RL」，而在把「舊新聞」操作化成兩個可觀測時間量，並用同一個 action menu、同一套 execution/cost 規則做 controlled probe。工程上它解了「LLM 信號進 policy 後，老化資訊到底要當特徵還是當垃圾」的介面坑；但它沒有解「舊新聞是否已被價格吸收」的因果識別。預測 envelope：10 檔大型美股、週頻、long-only、固定 extractor、10bp 成本下的平均結果；一到日內執行、小型股、或 extractor 換版，外部效度待核（本頁推斷）。對量化讀者，最有價值的是它把 action-change probe 與 paired intervals 分開報告，而不是 18.43% 的單點。RL Market Data 19.07% 高於 LLM filtered 18.43%，提醒讀者：這不是全表 SOTA，而是「在 LLM 決策器內部，過濾比保留好」的條件性結論。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作／常見做法 | 本文做法 | 讀者要點 |
|---|---|---|---|
| 信號生成 | LLM 供給 RL [3]，或 LLM agent 用 memory 直接決策 [4] | 固定 LLM 只讀一次 SEC 報告，輸出 `z∈{-1,0,1}` 加 evidence；所有 policy 共用 stored signals | 抽取與決策解耦，避免 policy 各自重讀文本 |
| 老化處理 | 以重複新聞／stale information 觀察投資者反應 [7] | 顯式拆 `τ`＝news age、`g`＝time since arrival；News／News+Age／Filtered News 三設定 | 老化從「文字語義」變成「可消融的時間介面」 |
| 決策器 | RL 交易 [1,2] 或 LLM agent [4] | 同 action menu 下比 Double DQN 與 direct LLM；HOLD 與 target portfolios 共用 execution | 控制執行與成本後再比輸入設定 |

### 1.2 ⚡ Eureka

**一句話：** 把「新聞老化」拆成兩個時間量，並把「過濾」當成一條 rule-based 正則化，而不是叫 policy 自己學會忽略。

**直覺：** 若價格已吸收舊報告，保留舊 `z` 只會增加 policy 的 action 自由度與換手；硬過濾直接刪除低信息量狀態，等於用一個低成本 prior 換取較低 variance。但這不證明過濾普遍最優，只證明在本文 menu、成本、樣本下平均較高（本頁推斷）。

### 1.3 信息流 ASCII 圖

```text
SEC periodic disclosures
        │
        ▼
fixed LLM extractor (GPT-4.1 mini) ──► z_i,t ∈ {-1,0,1} + evidence
        │                                   │
        │                          stored signals (shared)
        ▼                                   ▼
market m_t + holdings w̄_t ──► o_t = (w̄_t, m_t, z_t, τ_t, g_t)
        │                                   │
        ├─ News: no τ,g                      ├─ Filtered: z̃ = z·1{τ≤8}
        ▼                                   ▼
RL Double DQN  or  direct LLM policy ──► target q or HOLD
        │
        ▼
next week first open execution, 10bp per dollar bought/sold
```

## §2 · 數學層

📌 **Napkin Formula**

```text
τ_i,t = b_t − u_i,t,     g_i,t = b_t − v_i,t,     v_i,t = u_i,t + δ ≤ b_t
z̃_i,t = z_i,t · 1{τ_i,t ≤ h},     h = 8 weeks
r_{t+1} = κ log(V_{t+1}/V_t),     max_π E_π[Σ γ^t r_{t+1}]
```

**直覺：** `τ` 問「報告出版多久了」，`g` 問「報告到手多久了」；同一份四週前出版、今天才到的報告，`τ` 與 `g` 不同。Filtered News 不看 `τ`、`g` 的數值，只在 `τ` 超過 cutoff 時把 `z` 歸零，資產仍可投資。**Loss／訓練：** Double DQN，Huber smooth L1，Adam `3×10^-4`，`γ=0.99`，reward multiplier `100`，replay `100,000`，minibatch `256`，target-network hard copy every `100` optimizer updates。推論時 greedily 選最高 predicted value，不探索、不再訓練。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下數字是**假設／示意**，不是論文實證結果；論文真實數字留在 §5。

1. 假設某報告在決策 cutoff 的 `τ` 設為 8 週（示意；原文 cutoff `h=8`），`g` 設為 4 週（示意；原文有 4-week delay）。
2. Filtered News 的條件是 raw news age **strictly greater than** eight weeks 才歸零；`τ=8` 不歸零，`z` 保留。
3. News+Age 會把 `τ=8` 與 `g=4` 餵給 policy；policy 從 57 actions 中選 target 或 HOLD。
4. 若做 controlled probe，把同一報告再老化 12 週（原文用 12），`τ` 超過 cutoff，Filtered 把 `z` 設 0；News+Age 的 `τ/g` 變大但 `z` 不變。
5. 最後 action 進 next week first open 執行，成本 10bp per dollar。示意結束；實際 action-change 52% 與 18.43% 是 §5 實證。

## §3 · 數據層

- **市場／頻率：** 10 檔美國股票（AAPL, AXP, BA, CAT, CSCO, CVX, HD, JNJ, KO, VZ），週頻組合決策，歷史測試 2023–2025；每日價格覆蓋 2019 through early January 2026。所讀文本未見容量／ADV 假設，圖表／附件待核。
- **文本：** 277 篇 SEC excerpts，每篇 800–1,000 words，出版 2019 through 2025；2023–2025 cohorts 含 119 reports。Extractor 為 `gpt-4.1-mini-2025-04-14`，temperature zero，output limit 192 tokens。
- **延遲：** report delivery delays 0、1、4 weeks；另做 controlled test，把已可見報告的 publication/arrival 一起往回移，市場資料、holdings、原始信號不變。
- **訓練折：** 2023 用 2020–2021（103 weeks）訓練、2022（51）驗證、52 測試週；2024 用 2020–2022（155）、2023（51）、53 測試週；2025 用 2020–2023（207）、2024（52）、52 測試週。所讀文本未見更細的資料清洗規則，TBD。
- **Extractor 品質：** held-out synthetic reports 200 篇中正確分類 191 篇，95.5% accuracy、macro-F1 0.956；真實 SEC 文本 228 of 277 outputs（82.3%）符合 format 與 verbatim-evidence 要求，48 rejected outputs 與 one request failure 變成 zero signals。真實文本 semantic accuracy 無人標，無法評估。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 未披露（原文僅稱 reproduction materials／reproduction package） | 未披露 | 未披露 | TBD；需重跑 120 trained policies、API 呼叫與 paired bootstrap | SEC public + Yahoo Finance adjusted close；exact executed prompts 在 reproduction materials，API access TBD |

## §5 · 評測 / Benchmark

Δ 固定定義為「本方法 − 前SOTA」；前SOTA 欄只放所讀原文逐字給的單一基線值，不合成 SOTA 區間。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：LLM News → LLM Filtered | 16.02% | 18.43% | —（由前兩欄比較） |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：LLM News+Age → LLM Filtered | 17.17% | 18.43% | —（由前兩欄比較） |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：RL News → RL News+Age | 17.81% | 18.02% | +0.21pp |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：RL News → RL Filtered | 17.81% | 18.03% | +0.22pp |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：RL Market Data → LLM Filtered | 19.07% | 18.43% | —（由前兩欄比較） |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：Equal Allocation → LLM Filtered | 17.63% | 18.43% | +0.80pp |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annualized return after costs：Buy and Hold → LLM Filtered | 17.19% | 18.43% | —（由前兩欄比較） |
| 美股 10 檔 / 2023–2025 / 4-week delay | Sharpe ratio：LLM News → LLM Filtered | 0.98 | 1.12 | +0.14 |
| 美股 10 檔 / 2023–2025 / 4-week delay | Maximum drawdown：LLM News → LLM Filtered | 12.01% | 10.83% | —（由前兩欄比較） |
| 美股 10 檔 / 2023–2025 / 4-week delay | Annual trading volume（×）：LLM News → LLM Filtered | 18.87 | 19.03 | +0.16 |
| 美股 10 檔 / 2023–2025 / 4-week delay | Total trading costs：LLM News → LLM Filtered | 2.02% | 2.05% | —（由前兩欄比較） |

**解讀：** `18.43%` 是 LLM Filtered 在 4-week delay 的平均年化淨報酬，不是全表 SOTA；RL Market Data 的 `19.07%` 更高。Table V 的年度配對區間顯示 timing/filter 效果在年份與模型間正負不一；原文說 **None of the 36 primary annual timing/filter comparisons has an eight-week-block 95% interval excluding zero**。因此這組平均報酬比較應讀作「在這個 10 檔樣本與 2023–2025 期間的平均」，不是穩定 capability。真實 SEC 文本的 semantic accuracy 無人標，`228 of 277` 只量 format/evidence compliance。主結果使用 10bp；Appendix B-E 的年度成本檢查在四週新聞延遲下，將全部十個主要方法以 0/10/30bp 重放，固定原決策並重算持倉與費用。正文只報 News+Age 在 30bp 下由 `18.02%` 降至 `12.32%`；所讀正文未列 Filtered 在 30bp 下的數值（圖表待核）。2023 曾作 validation、2024–2025 已檢視，屬 retrospective，不能排除研究者已看過結果的影響。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- 回溯性評估：2023 had served as validation in a prior analysis，2024–2025 had already been inspected（Appendix B-B）。
- 不確定性區間是 unadjusted exploratory intervals；annual segments resampled independently，neglects dependence across year boundaries（Appendix A-A）。
- 真實 SEC 文本的 semantic accuracy cannot be assessed without human labels；`228 of 277` 只量 format 與 verbatim-evidence compliance（Appendix B-D）。
- 控制測試只量 portfolio-choice change，不建立 RL 系統性降低舊信號影響（§IV-B：decision changes show sensitivity without establishing that RL systematically reduces the influence of older signals）。
- GPT-6 Astra 是 API alias，不是 dated snapshot（Appendix C-A）。

### 6.2 推斷的隱含假設（本頁推斷，待核）

- **樣本 envelope：** 10 檔大型美股、週頻、long-only fully invested；容量、market impact、借券／稅未測。推斷不適用日內高頻或小型股。
- **Zero signal 語義混合：** 原文明說 zero 可代表 neutral report、no available report、或 format/evidence check 失敗。推斷這會讓 `τ/g` 與過濾效果的解釋混雜。
- **Cutoff 是設計選擇：** `h=8 weeks` 未見 threshold sweep；推斷 8 週不是估計出的最優 cutoff，不能說「8 週普遍最佳」。
- **成本模型：** 只計 fees；30bp 重放已使 News+Age 由 `18.02%` 降至 `12.32%`，推斷更高成本或 market impact 下排序可能翻轉（未驗證）。
- **Regime 依賴：** 2023–2025 是特定市場期間；推斷跨年度外部效度待核。
- **Survivorship：** 作者用 2019 DJIA ETF holdings 與 25 issuers 固定 universe，later returns or coverage 不選股（Appendix B-D），所以不是典型事後選股；但仍是 10 檔大市值樣本，不是全市場。輔助數據缺失不是生存者偏差。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| FinRL-DeepSeek [3] | LLM 風險／情感注入 RL，vs 本文固定 extractor＋時間過濾消融 | 未披露 | 概念前作；本文未做數值對比 |
| FinMem [4] | layered memory 的保留／遺忘，vs 本文顯式 `τ/g` 與 8 週硬過濾 | 未披露 | 概念對照；未在本文明確 benchmark |
| Tetlock [7] | stale information 的重複新聞反應，vs 本文把 stale 操作化成可消融時間介面 | 未披露 | 文獻 lineage；非同一實證設計 |
| RL Market Data（本文 baseline） | 不用新聞，只市場資料＋holdings | N/A | 本文最強基線之一：`19.07%` |
| Equal Allocation / Buy and Hold | 被動配置 vs 主動 policy | N/A | 本文內建被動基準：`17.63%`／`17.19%` |

🎤 **Interview Tip**

- **正確答：** 這篇的賣點是把 news aging 拆成 `τ` 與 `g`，並發現「告知時點」與「硬過濾」是不同機制；LLM 過濾平均最高，但年度區間含零，且 RL Market Data `19.07%` 更高。RL 的 action change `52%` 只證明它對時點敏感，不證明它改善報酬。
- **錯答：** 說「RL 學會忽略舊新聞所以績效提升」或「`18.43%` 是 SOTA」。原文明確說 action change 不建立系統性降低影響，且 RL Market Data 更高。

### 7.1 可證偽預測帶日期

在 **2026-12-31** 前，若把 8 週 cutoff 換成其他 `h` 並用相同的 paired time-block intervals 重跑：若 LLM filtered − News 的年度 Δ 多數仍含零，則本文的平均優勢應讀作樣本內不穩定；若多數年份的區間排除零，則外部樣本強化「過濾有用」的說法。此預測不涉及本文已給的單點 `18.43%`，只檢驗 cutoff 與年度穩健性（本頁推斷，尚待驗證）。

## §8 · For the Reader

- **因子研究員：** 把 news age 與 time since arrival 當兩個獨立特徵，先做 event-time 分層；注意 zero signal 混合 neutral／no report／format failure，不能直接當「無信號」。
- **高頻執行：** 本文是週頻、next week first open、10bp；不能直接推日內。30bp 對 News+Age 的敏感度是成本警訊，應先做 impact 模型再談上線。
- **組合配置：** 8 週硬過濾是低成本 regularizer，但 `h` 未掃描；先做 threshold robustness 與 paired block bootstrap，再決定是否把過濾寫進投資政策。
- **LLM-agent：** 不要把「LLM 抽信號」與「LLM 直接選組合」混為一談；本文 extractor 固定，direct LLM 只看匿名數值欄位，company names、dates、raw reports 被省略。
- **RL 策略：** action change ≠ return improvement；用 controlled probe 與 paired intervals 分開評估，否則容易把「對時點敏感」誤讀成「alpha 改善」。
- **研究學生：** 這是「消融設計＋不確定性報告」的範本，但 universe 小、retrospective、單一 extractor；不宜當 SOTA 引用，應引它的介面拆解與失敗模式。

## References

- 原論文：Bolin Song, Surui Xiang, *When News Gets Old: Portfolio Allocation with Reinforcement Learning and Large Language Models*, arXiv:2610.30564v1, 22 Oct 2026. [https://arxiv.org/abs/2610.30564v1](https://arxiv.org/abs/2610.30564v1)
- [1] J. Moody and M. Saffell, “Learning to trade via direct reinforcement,” *IEEE Transactions on Neural Networks*, vol. 12, no. 4, pp. 875–889, 2001.
- [2] Z. Jiang, D. Xu, and J. Liang, “A deep reinforcement learning framework for the financial portfolio management problem,” arXiv:1706.10059, 2017.
- [3] M. Benhenda, “FinRL-DeepSeek: LLM-infused risk-sensitive reinforcement learning for trading agents,” arXiv:2502.07393, 2025.
- [4] Y. Yu et al., “FinMem: A performance-enhanced LLM trading agent with layered memory and character design,” arXiv:2311.13743, 2023.
- [5] N. Gârleanu and L. H. Pedersen, “Dynamic trading with predictable returns and transaction costs,” *The Journal of Finance*, vol. 68, no. 6, pp. 2309–2340, 2013.
- [6] N. Gârleanu and L. H. Pedersen, “Dynamic portfolio choice with frictions,” *Journal of Economic Theory*, vol. 165, pp. 487–516, 2016.
- [7] P. C. Tetlock, “All the news that’s fit to reprint: Do investors react to stale information?” *The Review of Financial Studies*, vol. 24, no. 5, pp. 1481–1512, 2011.
- [8] H. van Hasselt, A. Guez, and D. Silver, “Deep reinforcement learning with double q-learning,” *AAAI*, vol. 30, no. 1, pp. 2094–2100, 2016.
- [9] P. Henderson et al., “Deep reinforcement learning that matters,” *AAAI*, vol. 32, no. 1, pp. 3207–3214, 2018.
- [10] R. Agarwal et al., “Deep reinforcement learning at the edge of the statistical precipice,” *NeurIPS*, vol. 34, 2021. [https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)
- [11] SPDR Dow Jones Industrial Average ETF Trust, “Annual report, october 31, 2019,” SEC Form N-30D, 2019.
- [12] Yahoo Finance, “What is the adjusted close?” accessed: Oct. 20, 2026.
