# 審計報告 · 自動化因子研究系統的可驗證成效與公開失敗（2026-07-22）

> **這是一份證據紀錄，不是綜述。** 它保留了完整的審計過程、通過的 17 條主張、**被殺掉的 8 條主張**、以及所有已知邊界。結論的解讀見 [Crossing · 自動化搜索 vs 預註冊驗證](/crossing/autoresearch-vs-preregistration)。

**Status:** v1.0 — 對抗式文獻審計，2026-07-22 執行。所有引文皆逐字核對原始 PDF/HTML。

## 研究問題

已經有人做過的「自動化因子研究系統」（automated / AI-driven factor mining & autoresearch）實際成效與公開失敗案例。三條線：(1) 遺傳規劃 / 符號回歸（GPLearn、AlphaEvolve、AutoAlpha、AlphaGen、WorldQuant Alpha101/191 與 BRAIN/WebSim 眾包平台）；(2) LLM 驅動的因子挖掘與 agentic quant research；(3) 商業 / 機構實作（WorldQuant、Two Sigma、Kavout、Qraft）。

核心追問：**樣本外表現到底如何？有沒有獨立驗證或事後被推翻的紀錄？用什麼機制對抗多重檢定 / data snooping？失敗模式是什麼？** 明確排除行銷說法，只要可驗證證據與「別人踩過的坑」。

## 方法與審計統計

**流程**：問題分解為 6 個檢索角度 → 每角度獨立並行檢索 → URL 去重 → 抓取全文並抽取可證偽主張 → **每條主張三票對抗驗證（驗證者被要求「嘗試推翻」，不確定時預設判為推翻；需 2/3 推翻票才殺掉）** → 語義去重合成 → 依信心排序。

| 指標 | 數值 |
|---|--:|
| 檢索角度 | 6 |
| 抓取來源 | 25（URL 重複剔除 6） |
| 抽取的可證偽主張 | 119 |
| 進入對抗驗證 | 25（預算限制丟棄 5） |
| **確認** | **17** |
| **推翻** | **8** |
| 未能驗證 | 0 |
| 合成後最終條目 | 10 |
| Agent 呼叫總數 | 108 |
| 耗時 | 約 16 分鐘 |

**6 個角度**：① 自動因子挖掘的樣本外實證與翻車紀錄（broad）② RL/formulaic alpha 論文與獨立複現（academic）③ LLM 驅動 / agentic quant research（recent）④ 多重檢定與過擬合防護機制（statistical guardrails）⑤ 商業與眾包平台的可驗證績效（contrarian）⑥ 衰減、擁擠、成本吃掉 alpha（practitioner）。

## 總結論

**證據是一面倒的。** 橫跨 GP/符號回歸、RL formulaic mining、LLM agent 迴圈三條線，幾乎每一套已發表的「自動化因子研究」系統報的都是自家的、單一切分的、只有 IC 的結果，而**倖存的證據裡沒有任何一條包含多重檢定或 data snooping 校正**——沒有 deflated Sharpe、沒有 PBO/CSCV、沒有 White's Reality Check。而這些方法搜索的正是組合上極其龐大的空間，也就是名義 5% 偽陽性率會變成近乎必然偽發現的那個 regime。

存在不利檢定的地方，它是毀滅性的，而且**來自提倡者自己**：AlphaQCM 的附錄顯示，僅僅把 holdout 從 2021-2022 延長到 2021-2024，幾乎每個方法的樣本外 IC 就腰斬。種子與種子之間的離散度相對於效果量而言很大，所以這條文獻裡的單次運行結果不可重現，而任何頭條數字的獨立第三方複現實質上不存在。

## 確認的主張（17 條 → 合成為 10 條）

### 1 · 大規模回測必然製造偽陽性，而常規檢驗會為它蓋章 · 信心：高 · 票數 3-0

純隨機漫步生成 1,000 天日價（約 4 年），4 參數月度規則在 22×20×10×2 = **8,800 節點網格**上優化，勝出者樣本內年化 Sharpe **1.27**，PSR 統計量 **2.83**——「意味著真 Sharpe 低於 0 的機率不到 1%」。而 CSCV 給出 **PBO = 55%**，約 **53% 的樣本外 Sharpe 為負**。

同篇原文：「演算法研究與高效能運算的近期進展，讓在有限資料上測試數百萬、數十億個替代投資策略變得幾乎不費力……5% 的偽陽性機率只在我們**恰好套用一次**檢定時才成立。」

**來源**：Bailey, Borwein, López de Prado & Zhu, *The Probability of Backtest Overfitting*, J. Computational Finance 20(4)。[PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

**保留**：示範是單一模擬路徑，53%/55% 是一次實現值；且名義試驗數會高估多重性，因候選彼此相關（effective trials 文獻見 López de Prado & Lewis 2018）——但即使有效試驗數只有數百，名義 5% 門檻依然被打破。

### 2 · 兩條硬約束：反過擬合指標不能當目標；單一 holdout 無法回答過擬合問題 · 信心：高 · 票數 3-0

(a)「我們必須警告讀者不要用 CSCV 來指導最優策略的搜索。那將構成對本方法的嚴重誤用……當一個度量變成目標，它就不再是好的度量……PBO 不應該是選擇所依賴的目標函數。」（引 Strathern 1997）

(b)「只要研究者試過不只一組配置，過擬合就永遠存在……hold-out 方法不考慮嘗試次數……『這個回測過擬合了嗎』的答案不是真或假，而是一個取決於試驗次數的機率。」

**獨立佐證**：Arian, Norouzi & Seco (2024), *Backtest overfitting in the machine learning era*, Knowledge-Based Systems 305——單路徑與 walk-forward 方法「在偽發現防護上有顯著缺陷」，組合式淨化交叉驗證（CPCV）在 PBO 與 deflated Sharpe 上支配它們。

**保留**：古典統計仍然成立——一個鎖死且**恰好使用一次**的 holdout 能給出選定策略的無偏點估計；原文的反對意見是關於統計檢定力，以及實務上 holdout 會被重複使用（「讓投資人只能猜測過擬合的**程度**」）。

**對建造者的直接推論**：既有的預註冊 + holdout 閘門是必要但不充分的——一旦迴圈產生數千個候選，閘門必須加上一個明確的試驗計數器，餵給在迴圈永不優化的 holdout 上計算的 deflated Sharpe / PBO。

### 3 · 最便宜的防線是搜索前的 pretest · 信心：高 · 票數 3-0

Chen & Navet (2007), *Failure of Genetic-Programming Induced Trading Strategies: Distinguishing between Efficient Markets and Inefficient Algorithms*, Springer CIEF Vol. II。

「在第一階段先檢驗這根針是否存在，會比直接套用完整版搜索來得經濟。我們稱這個程序為 pretest。」四種 pretest 區分「零智能策略」與「樂透交易」；Pretest 1 在**唯一候選數**上精確匹配 GP（500 族群 × 100 世代 ≈ 5 萬個隨機候選），構成一個被精確刻畫的 null。

停止規則逐字：「如果 pretest 的結果顯示 GP 沒有優於隨機搜索或隨機交易行為的統計證據，那我們認為就沒有必要再投入資源於 GP。」

**重要限定**：它是**帶避險的兩分支閘門**，不是硬停——輸給樂透交易 = 沒東西可學（放棄）；在可學資料上輸給隨機搜索 = 機器配錯（作者明確授權繼續調參）。作者自述僅為「初步證據」，3 個交易所有 1 個不一致，且 pretest 框架**無任何獨立複現**。年代（2006/07）與現代 RL/LLM 挖掘文獻近乎零的採用率，意味著這是一個原理紮實但非標準的協定。

**來源**：[Springer](https://link.springer.com/chapter/10.1007/978-3-540-72821-4_11) · [ICONIP 2006 全文](https://www.aiecon.org/staff/shc/pdf/iconip2006.pdf) · [HAL](https://inria.hal.science/inria-00168269)

### 4 · RL formulaic alpha 文獻把問題框成搜索，從不框成統計推斷 · 信心：高 · 票數 3-0

對 AlphaGen (KDD'23)、AlphaQCM (ICML'25)、TLRS (2025)、RiskMiner (ICAIF'24) 全文檢索，`multiple testing`、`data snooping`、`deflated Sharpe`、`PBO`、`CSCV`、`White Reality Check`、`false discovery`、`Bonferroni` **出現次數皆為 0**。唯一防線是驗證集選模型，而該選擇步驟本身即誘發未校正的選擇偏誤。

逐項核對：
- **AlphaGen**（2306.12964）：唯一防護是年代切分 train 2009-2018 / valid 2019 / test 2020-2021，超參「依 Qlib 給的 benchmark 設定」。
- **TLRS**（2507.20263）：144k 字元 HTML 全文檢索——`overfit*`=0、`multiple test*`=0、`data snoop*`=0、`deflated`=0、`PBO`=0、`CSCV`=0、`false discover*`=0、`backtest*`=0。摘要純粹把問題框為「MDP 的稀疏獎勵……限制了對龐大符號搜索空間的探索」。
- **AlphaQCM**（PMLR v267）：17 頁 camera-ready 的 pypdf 抽取——`turnover`=0、`transaction cost`=0、`slippage`=0、`sharpe`=0、`deflated`=0、`multiple testing`=0、`p-value`=0、`data snooping`=0、`backtest`=0；`portfolio` 僅修辭性出現一次。
- **RiskMiner**：risk-**seeking** MCTS 明確優化最佳情況而非平均情況——多重性防護的反面。

**獨立佐證**：Springer FCS 2025（10.1007/s11704-025-41061-5）指出原版 AlphaGen「取得中等表現，但也觀察到在訓練集上的過擬合」（樣本內平均 IC > 0.1），且 RL formulaic 生成器「在測試集上表現不佳」。

**註**：AlphaGen 的 alpha pool 確實有互相關冗餘過濾，但那是**多樣性約束**，不是多重檢定校正。

### 5 · 樣本外證據底盤結構性薄弱 · 信心：高 · 票數 3-0 與 2-1

AlphaGen 的全部樣本外主張建立在**單一固定的 2 年中國 A 股窗**（2020-01-01 至 2021-12-31），沒有 walk-forward、沒有滾動起點、沒有第二個測試期；AlphaQCM 用單一 2021-2022 區塊。頭條優勢絕對值偏小：AlphaGen CSI300 IC 0.0725 / RankIC 0.0806 vs XGBoost 0.0404 / 0.0576；CSI500 0.0438 / 0.0727 vs 0.0353 / 0.0639。第三方複現報告 AlphaGen 顯著較低：**IC 0.045 / RankIC 0.058**（arXiv 2401.02710，把 AlphaGen 當 Baseline [8]）。

**兩項對強版本批評的修正**（驗證過程中被抓出）：
1. 「2020-2021 是異常有利的小市值/動量體制」這個說法**事實錯誤**——2020 年 A 股是大市值/「核心資產」年（CSI300 約 +27% vs CSI500 約 +21%），2021 年反轉（CSI300 約 −5%，CSI500 約 +16%）。誠實的說法是「一段代表性未知的單一體制片段」，且頭條回測跑在大市值宇宙上。
2. AlphaGen 減 XGBoost 的 0.0321 差距在約 486 個測試日上約為 5 個標準誤，所以它**不是**字面意義上的「在抽樣噪音內」。可辯護的反對是體制/選擇噪音與搜索多重性，不是天真的 t 檢定。

後續文獻（AlphaForge、RiskMiner、Alpha²、AlphaEval）是**重新 benchmark** AlphaGen，而非為它的樣本外設計辯護。

### 6 · 最強的衰減證據來自提倡者自己的附錄 · 信心：高 · 票數 3-0

AlphaQCM 附錄 G.3「Impact of Expanded Dataset」：「我們擴展測試集……訓練與驗證期保持不變……所有其他實驗設定與 Table 1 一致」，接著是「**幾乎所有方法都出現明顯的樣本外 IC 下降。這個問題或許可以透過在新資訊到來時重新擬合模型來解決。**」

| 方法 · 宇宙 | 2021/01–2022/12 | 2021/01–2024/12 |
|---|--:|--:|
| AlphaQCM · CSI 500 | 9.55% | 5.87% |
| AlphaGen · CSI 500 | 8.08% | 4.19% |
| AlphaGen · CSI 300 | 8.13% | 4.13% |
| AlphaQCM · CSI 300 | 8.49% | 5.48% |

**兩項限定**：(i)「每個方法都腰斬」是寬鬆說法——跌幅從約 12%（Alpha101 3.44→3.02）到約 50%（AlphaGen），論文自己的用詞是「幾乎所有」；(ii) IC 仍為正，且 AlphaQCM 在延長窗仍勝基線，所以這是重度衰減而非死亡。**關鍵**：延長窗**包含** 2021-2022，因此 2023-2024 單獨的隱含邊際 IC 顯著低於 5.87%——這強化衰減的讀法。論文從未拆解「衰減 vs 中國 A 股體制轉換」。提倡者自行揭露不利穩健性結果，是可能的最強證據來源。

**來源**：[OpenReview PDF](https://openreview.net/pdf?id=3sXMHlhBSs) · [PMLR camera-ready](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhu25ag/zhu25ag.pdf)

### 7 · 單次運行結果不可重現 · 信心：高 · 票數 3-0

跨 10 個隨機種子：GP + 互相關過濾在 Market 宇宙 **IC 0.84% ± 2.27%**；GP 無過濾 1.32% ± 2.01%；PPO + 過濾 2.15% ± 1.86%。而 AlphaQCM 在 CSI300 上勝 AlphaGen 的幅度（8.49% vs 8.13%，差 0.36%）**小於兩方法各自的種子離散度**（1.03% / 0.94%）。

**三項對強版本讀法的修正**：
1. 「兩者都與零無異」對 PPO + 過濾**是錯的**——10 種子的均值標準誤 = 1.86/√10 = 0.59%，t ≈ 3.7；真正與零無異的只有 GP + 過濾（t ≈ 1.17）。可辯護的說法是**單次運行不可靠，且有實質機率落在負 IC**。
2. CSI300 是對 AlphaQCM 最不利的一格；在 CSI500（9.55 vs 8.08）與 Market（9.16 vs 6.04）它的優勢有數個 sigma 寬，所以不能從 CSI300 一格推廣出「AlphaQCM ≈ AlphaGen」。
3. GP/PPO 基線由競爭方法的作者自行實作與調參——經典的欠調基線設定，所以「GP 一無是處」是編輯性修飾；論文自己只說 suboptimal。

**另一項第三方 benchmark 顯示 IC 排名與回測損益排名可以完全反轉**：AlphaQCM 年化 3.42% / Sharpe 0.1973 vs AlphaGen 6.19% / 0.3425（CSI300）。

### 8 · 評估協定本身是最深的坑 · 信心：高 · 票數 3-0 / 3-0 / 2-1

頭條結果只報截面 IC/RankIC，沒有淨成本報酬、沒有風險調整、沒有多重性校正。

- **TLRS**：摘要唯二的量化主張是 RankIC 相對「既有勢能塑形演算法」提升 9.29%，以及特徵維度上從線性降到常數的複雜度改進——零個報酬/Sharpe/回撤/換手/成本數字。（**範圍限定**：TLRS 第 V 節確有「Factors Evaluation」子節但無法取得，所以「全文無回測」**未被確立**，只確立了頭條主張純屬統計。）
- **AlphaQCM**：「我們選擇 IC 作為評估樣本外表現最重要的指標」，附錄 A 只定義 IC，四張表全是 IC，三張圖沒有淨值曲線。
- **AlphaEval**（2508.13174，v2 2026-06-02）：摘要逐字——「既有評估指標主要包括回測與相關性度量……相關性指標雖高效，只評估預測能力，忽略了時序穩定性、穩健性、多樣性與可解釋性等其他關鍵性質。」對 v1 與 v2 全文檢索 `multiple testing`、`deflated Sharpe`、`PBO`、`CSCV`、`White's reality check`、`data snooping`、`false discovery`、`Bonferroni` **皆為 0**；五個操作化指標（PPS、相對排名熵、擾動保真、LLM 邏輯分、多樣性熵）**不含任何試驗調整機制**。

**保留**：AlphaEval「這個領域只做兩件事」的框架是利害關係方的自利文獻回顧，不是獨立調查——deflated Sharpe / PBO 文獻確實存在，只是被它忽略；且 AlphaEval v2 明確聲明不取代回測。AlphaEval 本身無任何獨立評估或反駁。

### 9 · LLM agent 分支報出最像組合的數字，但成本模型不成立 · 信心：高 · 票數 3-0

AlphaAgent（arXiv 2502.16789v2）Table 2 逐字：CSI500 年化超額 **11.00%**、IR 1.488、MDD −9.36%、IC 0.0212；S&P500 **8.74%**、IR 1.0545、MDD −9.10%、IC 0.0056；測試窗 2021-01-01 至 2024-12-31。

**成本假設**：CSI500 買 0.05% / 賣 0.15%；**S&P500 只收賣出 0.05%——買入免費，無價差、無衝擊、無借券。**

唯一的統計控制是對 RD-Agent 的兩兩 t 檢定，p = 0.031 / 0.011 / 0.038——這種邊際在天真的多重性校正下就活不下來。搜尋批評或複現只找到論文本身、作者 GitHub、以及把它當 baseline 引用的後續工作，**確認零獨立驗證**。

**最能說明問題的內部張力**：S&P 500 的結果建立在 IC 0.0056（接近零的截面預測力）上，卻宣稱 IR 1.05——一個只收賣出費的成本模型不足以讓人相信這個組合能存活。

### 10 · LLM 分支自述的失敗機制是因子同質化 · 信心：中 · 票數 3-0

AlphaAgent 摘要逐字：「遺傳規劃等傳統方法容易快速 alpha 衰減，主因是其易於過擬合。同時，大型語言模型驅動的方法儘管有潛力，卻往往未能對因子同質化施加正則——導致訊號擁擠與加速衰減。」

其解方是基於 AST 的原創性篩選：$s(f_i,f_j) = \max\{|t_i| : t_i \cong t_j\}$（最大公共子樹相似度），對照 Alpha101 這類既有 alpha zoo 施加懲罰；另有複雜度項（符號長度 + 參數數 + 假設對齊）針對「易過擬合」的過度工程構造。**這是反擁擠的新穎性懲罰，明確不是統計多重檢定控制。**

**信心為中而非高**：這是關於「論文的論證」的主張，而論文未提供該同質化機制的獨立因果證據——沒有擁擠度量測、沒有第三方複現、只有邊際的自家消融 p 值。應讀成「**領先的 LLM 挖掘論文主張 X**」，而非「X 已被確立」。

## 被推翻的主張（8 條 · 請勿引用）

這些說法看起來合理、且部分在中文量化圈流傳極廣，但**未通過三票對抗驗證**：

| # | 被殺的主張 | 票數 | 來源 |
|---|---|:--:|---|
| 1 | 447 個已發表異象在統一複現下有 286 個（**64%**）在 5% 水準不顯著 | 1-2 | [NBER w23394](https://www.nber.org/system/files/working_papers/w23394/w23394.pdf) |
| 2 | 套用 Harvey-Liu-Zhu 的 t≥3 門檻後失敗數升到 380/447（**85%**），只剩約 15% 存活 | 0-3 | 同上 |
| 3 | 複現失敗的主要機械成因是微型股加權（微型股約 3% 市值但約 60% 檔數） | 0-3 | 同上 |
| 4 | Chen & Navet 把整整十年的 GP 交易研究刻畫為無定論、作者習慣性延遲判斷 | 0-3 | [Springer CIEF](https://link.springer.com/chapter/10.1007/978-3-540-72821-4_11) |
| 5 | TLRS 靠獎勵「與 Alpha101 專家公式的結構相似度」提升因子品質（= 刻意偏向已擁擠結構） | 1-2 | [TLRS](https://arxiv.org/html/2507.20263) |
| 6 | 2025-2026 多數已發表自動 alpha 挖掘模型是閉源，作者直言這阻礙可複現性 | 0-3 | [AlphaEval](https://arxiv.org/abs/2508.13174) |
| 7 | AlphaAgent 的五年衰減對照顯示 Alpha158 / GP / RSI 的 IC 從 0.022-0.036 掉到接近零 | 0-3 | [AlphaAgent](https://arxiv.org/html/2502.16789v2) |
| 8 | 競爭系統在 CSI500 上近乎無用：AlphaForge 3.45% / IC 0.0146、RD-Agent 0.78% / IC 0.0113 | 0-3 | 同上 |

> **第 1、2 條特別值得警告**：「因子動物園有六成到八成五複現失敗」是中文量化討論裡最常被引用的一組數字之一。它**沒有以上述形式通過驗證**。要用請直接回 NBER 工作論文原文逐字核對。
>
> **第 7、8 條的性質不同**：那是利害關係方報告的競爭對手成績。本手冊的 [AlphaAgent 解構頁](/foundations/factor-mining/alphaagent) 目前引用了第 7 條的衰減對照作為「真實抗衰減能力體現」——該引用應降級為「論文自述、未經獨立驗證」。

## 已知邊界

**範圍缺口——商業/機構那條線實質上沒有答案。** 關於 WorldQuant（Alpha101/191、WebSim/BRAIN 眾包）、Two Sigma、Kavout、Qraft 的主張**零條通過驗證**。這本身就是一個發現：這些公司未公開任何綁定其自動化研究流程的、可獨立驗證的樣本外實績，因此任何關於其成效的陳述都會建立在行銷材料上——正好是本次審計排除的東西。請讀成「**不存在可驗證的公開證據**」，而非「它們無效」。Alpha101 在本報告中僅作為學術論文裡的 benchmark/錨定函式庫出現，從不是一份經審計的實盤紀錄。

**來源品質不對稱。** 方法論那條線（Bailey / López de Prado、Chen & Navet）是同行評審的一手來源，且**不會衰減**——它是定理與模擬，不是實證。系統那條線幾乎全是 arXiv/會議論文的自我報告加自家基線。**本報告裡每一個量化績效數字都是提倡者自己的回測。** 唯一找到的第三方複現（arXiv 2401.02710 把 AlphaGen 報成 IC 0.045 vs 原文 0.0725）方向**向下**，而另一份 benchmark 在從 IC 換成回測損益時把 AlphaQCM vs AlphaGen 的排名**完全反轉**。基線由競爭方法的作者實作，是經典的欠調基線設定，因此「相對優越性」是每張表裡最弱的部分。

**地理與體制集中。** 幾乎全部證據是中國 A 股（CSI300/CSI500）、2020-2024、短的單一區塊測試窗。遷移到標的少得多、截面寬度低得多、且沒有漲跌停微結構的美股/全球 ETF 宇宙，**可遷移性未確立**。**截面 IC 這個指標在 10-30 檔 ETF 的輪動宇宙上幾乎不適用。**

**仍不確定的**：(i) 觀察到的 IC 從 2021-2022 到 2021-2024 腰斬，是 alpha 衰減、擁擠、還是中國體制轉換——沒有論文做過拆解。(ii) 持續 refit 是否真能救回衰減的 alpha——它被提出（「或許可以」）但從未被測試。(iii) TLRS 第 V 節可能含組合層指標而無法取得，故「只有 IC」對其頭條成立、對全文未確立。(iv) AlphaEval 的驗證部分依賴對已抓取全文的負向關鍵字檢索，故不能 100% 排除某處孤立提及（但列舉出的指標集獨立確認了未實作任何試驗調整）。

**時效性。** 系統那條線移動很快（引用論文橫跨 2025-02 至 2026-06 的修訂），排行榜會變；方法論那條線（2006-2016）不會。任何關於「當前 SOTA 排名」的斷言保質期以月計。

## 未解問題

1. **有沒有任何一套自動因子挖掘系統——學術或商業——公開過帶預註冊協定、明確 trial count、且附 deflated Sharpe 或 PBO 的樣本外紀錄？** 本次驗證的證據集裡沒有。這可能意味著從未有人做過，也可能只是本次檢索未觸及。**對想建這種迴圈的人，這是最有決策價值的單一空白。**
2. WorldQuant BRAIN/WebSim 眾包平台實際支付與部署了什麼？有沒有任何監管申報、訴訟紀錄、前員工說法或基金績效揭露，把自動 alpha 提交量與實現的基金報酬連起來？行銷側文件充分，可驗證側本次無法觸及。
3. **候選高度相關時，多重性懲罰該怎麼算？** effective-number-of-independent-trials 文獻（Harvey & Liu；López de Prado & Lewis 2018 的無監督聚類）存在但本次未評估——對候選近似克隆的小型 ETF 輪動宇宙，名義 N 與有效 N 的差距可能極大，並直接決定 deflated Sharpe 閘門是否可用。
4. **Chen & Navet 的 pretest 在現代數據與小宇宙上還有鑑別力嗎？** 原始證據自述初步、1/3 市場不一致，19 年無複現。若有效，它是研究迴圈上最便宜的前置閘門；若無效，迴圈需要另一種預先承諾的檢定。

## 來源清單（25 個，按角度）

**方法論（primary，不衰減）**
- [The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf) — Bailey, Borwein, López de Prado, Zhu
- [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) — Bailey & López de Prado
- [Chen & Navet 2007 · Springer](https://link.springer.com/chapter/10.1007/978-3-540-72821-4_11) · [ICONIP 全文](https://www.aiecon.org/staff/shc/pdf/iconip2006.pdf) · [HAL](https://inria.hal.science/inria-00168269)
- [Backtest overfitting in the ML era · KBS 305](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)
- [SSRN 2460551](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551) · [Portfolio Optimization Book §8.3](https://portfoliooptimizationbook.com/book/8.3-dangers-backtesting.html)
- [NBER w23394](https://www.nber.org/system/files/working_papers/w23394/w23394.pdf)（**注意：本報告推翻了 3 條基於此文的常見引用**）

**系統（arXiv/會議自我報告）**
- [AlphaGen · 2306.12964](https://arxiv.org/abs/2306.12964) · [ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2306.12964)
- [AlphaQCM · OpenReview](https://openreview.net/pdf?id=3sXMHlhBSs) · [PMLR](https://raw.githubusercontent.com/mlresearch/v267/main/assets/zhu25ag/zhu25ag.pdf)
- [TLRS · 2507.20263](https://arxiv.org/html/2507.20263) · [RiskMiner · 2402.07080](https://arxiv.org/abs/2402.07080)
- [AlphaAgent · 2502.16789v2](https://arxiv.org/html/2502.16789v2) · [AlphaEval · 2508.13174](https://arxiv.org/abs/2508.13174)
- [第三方複現 · 2401.02710](https://arxiv.org/abs/2401.02710)（AlphaGen 修正為 IC 0.045）
- [2502.21206](https://arxiv.org/abs/2502.21206) · [2412.00896](https://arxiv.org/html/2412.00896v1)

**商業/眾包（本線無主張通過驗證）**
- [Quantopian 關閉 · QuantRocket](https://www.quantrocket.com/blog/quantopian-shutting-down/) · [Quantopian 教訓](https://whatworksintrading.substack.com/p/the-rise-and-fall-of-quantopian-lessons)
- [Qraft AI ETF 清算公告](https://www.stocktitan.net/news/QRFT/exchange-traded-concepts-to-close-and-liquidate-qraft-ai-enhanced-u-6wawjnx0qv1k.html)

---

**解讀** → [Crossing · 自動化搜索 vs 預註冊驗證](/crossing/autoresearch-vs-preregistration) ｜ **落地** → [自動化研究迴圈建造指南](/guides/autoresearch-loop) ｜ **實戰對照組** → [投資哲學量化](/use-cases/philosophy-quantification)
