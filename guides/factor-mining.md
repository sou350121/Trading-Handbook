# 因子挖掘全景：從手工 alpha 到自動化因子生產線

> 這一族坐落在「監督回歸 / 元學習搜索 / 強化學習 / 生成式大模型」四種範式與「因子挖掘」機制的交叉帶，時間尺度多為日频波段到中長周期。它真正在賣的不是「更高 IC 的單因子」，而是**一套把因子從發現到過濾到組合的生產線**，以及在這條線上不斷後撤的護城河：可解釋性、搜索效率、抗衰減。

## 這一族在解什麼問題

因子挖掘的根本張力只有一句話：**你想榨乾非線性，但市場只為可歸因、可持有、還沒被擁擠掉的東西付錢。** 這條張力貫穿全部 61 頁。

第一層是**可解釋 vs 榨乾非線性**。傳統手工 alpha（價值、動量、反轉）可歸因、可審計、監管友好，但預測上限低且衰減快；深度因子能吃到高階交互，但權重估計在低信噪比截面裡被 SGD 雜訊淹沒。[PTK](../foundations/factor-mining/ptk.md) 把這條張力講得最透：它刻意放棄端到端聯合優化的理論最優，把 DNN 降級為純特徵提取器、用顯式核方法接管定價，換來的是估計效率的帕累托改進——因為「特徵有預測力但組合權重過擬合」才是實盤真正的坑。[alpha-vs](../foundations/factor-mining/alpha-vs.md) 則把它上升為一份架構選型決策樹：範式選擇不取決於誰更先進，而取決於市場微觀結構與組織算力稟賦——在機構主導、線性主導的市場，Alpha Mining 的可審計性與低算力門檻仍是 Pareto 最優。

第二層、也是這一族的暗線，是**「誰持續付錢」**。因子一旦被發表、被複製、被 ETF 化，人均利潤就被稀釋。[art-344 雙曲線衰減模型](../foundations/factor-mining/art-344.md) 用博弈論把這件事寫成閉式解：固定 Alpha 容量下，套利者以泊松速率進入，人均利潤自然呈 $1/(1+\lambda t)$ 的厚尾衰減。這意味著**因子挖掘的競爭對手不是別的模型，而是時間和擁擠**。因此這一族最新的一批工作（[AlphaAgent](../foundations/factor-mining/alphaagent.md)、[AlphaAgentEvo](../foundations/factor-mining/alphaagentevo.md)）已經不再比誰的 IC 高，而是比誰能「抗衰減」——把 AST 結構約束當成一種正則，逼模型脫離已被佔滿的語法擁擠帶。

把這兩層合起來看，這一族的產出從來不是「終端策略」，而是**因子生產線的某一節**：發現、篩選、正交化、動態組合、擁擠監控。這是讀這一族的正確心態——不要問「這個因子能不能直接上倉」，要問「它補的是生產線的哪一節、下游還缺什麼」。

## 方法譜系（演進主線）

把真實頁面串成一條演進脈絡，每一代解的是上一代的一個結構性缺陷。

**第 0 代：符號回歸 / 遺傳規劃（GP）。** 因子挖掘的起點是把公式當成可搜索的符號樹，用交叉變異在算子空間裡爬山。它的死穴是兩個：離散變異導致適應度評估必須跑實盤回測（高延遲），以及盲目交叉堆疊出大量語義無效的算子（量價相加）。這一代在本手冊裡沒有獨立頁面，但它是所有後續頁面 §7 對照表裡的公共基線（gplearn / DSO / AlphaGen）。

**第 1 代：把搜索變成可導或可規劃。** [AlphaForge](../foundations/factor-mining/alphaforge.md) 用 Predictor 作 Proxy Optimizer，把「跑回測算 IC」降維成「神經網絡前向推理」，於是離散符號搜索變成連續梯度優化——這是它比 GP 多解的一件事。[Alpha2](../foundations/factor-mining/alpha2.md) 走另一條路：把因子公式拆成四元組指令流、用 DRL 引導 MCTS 逐步「寫代碼」，並加維度一致性硬約束從源頭斬斷量價相加這類無意義運算。[AlphaEvolve](../foundations/factor-mining/alphaevolve.md) 則把「行業內排名/去均值」抽象成可進化的圖節點（RelationOp），把截面中性化寫進變異算子，配指紋緩存去重加速。這一代的共同進步：**從盲目枚舉走向結構化、可加速的搜索**。

**第 2 代：強化學習與探索設計。** RL 把因子生成建模為序列決策 MDP，痛點轉移到金融任務特有的「非平穩 + 稀疏獎勵」。[AlphaQCM](../foundations/factor-mining/alphaqcm.md) 的貢獻是探索獎勵：它從有偏分位數估計裡「將錯就錯」提取無偏方差當探索信號，直接對策稀疏獎勵。[Style Miner](../foundations/factor-mining/style-miner.md) 把因子構造寫成約束 MDP，用線性增長的自適應乘子把硬約束（自相關穩定）轉成可調正則，繞開不可微反饋。[AlphaAgentEvo](../foundations/factor-mining/alphaagentevo.md) 是這一代最新形態：用 GRPO 群組歸一化去掉 Critic 的顯存負擔、五維層次化獎勵拆解稀疏金融指標、AST 相似度硬閾值鎖住搜索半徑防 Reward Hacking。這一代多解的是：**在低信噪比環境裡讓搜索既高效又不崩潰**。

**第 3 代：生成式大模型（LLM）生成因子。** LLM 帶來的是「語義先驗」：把研究員的自然語言直覺直接翻譯成因子。[Alpha-GPT](../foundations/factor-mining/alpha-gpt.md) 讓 LLM 只負責生成種子因子與結構化 Prompt、把組合爆炸留給 GP，是「語義錨定 + 符號搜索」的混合架構。[AlphaAgent](../foundations/factor-mining/alphaagent.md) 用 Idea/Factor/Eval 三智能體閉環，配 AST 相似度懲罰逼因子脫離 Alpha101 這類擁擠庫。[Proj-SUE](../foundations/factor-mining/proj-sue.md) 是另一種用法：不用 LLM 寫公式，而是用 LLM 文本嵌入把靜態行業分類換成動態語義同業網絡，構建盈餘溢出因子。這一代多解的是：**把「想法到因子」的認知摩擦壓到近零，並用非結構化文本開出量價之外的新數據面**。

**第 4 代：工業化過濾與動態組合。** 當生成能力過剩，瓶頸就從「搜索」轉到「過濾」與「組合」。[LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) 把這一步講得最誠實：280 個 LLM 候選因子過完多重檢驗、因子吸收、異象賽馬，FF5 下只剩極少數存活，多數高 t 值公式是「舊酒裝新瓶」的已知風險溢價重包裝。組合端則有 [FactorMoE](../foundations/factor-mining/factormoe.md)（鏈式 MoE + Gumbel-TopK 動態門控，保留公式化因子的歸因通道）與 [AlphaForge](../foundations/factor-mining/alphaforge.md) 的滾動 IC/ICIR 動態線性加權，以及正交化流水線 [HF9/Post-Adaptive-LASSO](../foundations/factor-mining/hf9-post-adaptive-lasso.md)（用 Adaptive LASSO 頻率排序剝離偽 alpha）。這一代的共識：**因子挖掘的護城河已經從「能不能生成」搬到「能不能過濾與正交化」**。

一條旁支值得單列：**「簡單 + 歸納偏置」派**。它是對整條深度化譜系的反例。[BRT-NN 遞歸排序](../foundations/factor-mining/brt-nn.md) 在 18,000 個信號宇宙上證明，用歷史 t 值遞歸排序做降維、讓 ML 只學「如何加權有效信號」，夏普顯著高於把高維直接喂進 BRT——實盤 alpha 的瓶頸在輸入端維度災難，不在模型深度。[Unicorn Edge](../foundations/factor-mining/unicorn-edge.md) 更極端：不發明新因子，只用一個離散狀態過濾器決定「什麼時候用舊因子」，訓練成本近零。這條旁支提醒你：譜系的「進步」方向不等於實盤的「有效」方向。

## 三到五條核心張力

**張力一：搜索自由度 vs 過擬合 / 擁擠。** GP 與純 LLM 生成處在「自由度」一端——表達力強但極易生成擬合噪音、彼此高相關的因子。[AlphaAgent](../foundations/factor-mining/alphaagent.md) 與 [AlphaAgentEvo](../foundations/factor-mining/alphaagentevo.md) 站在「約束」一端，用 AST 結構相似度硬約束把全局搜索降維成局部流形微調。**我的裁決：約束派方向正確，但要小心它的隱藏代價**——AST 相似度只捕捉語法樹同構，識別不了金融意義上的「功能等價」；且 AlphaAgentEvo 自己承認熊市 Pass@3 掉到 0.581（牛市 0.963），說明硬閾值在 regime 反轉時會從護欄變成創新阻礙。約束是對的，但閾值必須隨波動率動態化。

**張力二：靜態因子 vs regime 自適應。** 一端是靜態線性加權（GP 產出 + 固定權重），另一端是條件化 / 動態組合。[FactorMoE](../foundations/factor-mining/factormoe.md)（雙路引導動態門控）、[AlphaForge](../foundations/factor-mining/alphaforge.md)（滾動 IC 權重）、[Unicorn Edge](../foundations/factor-mining/unicorn-edge.md)（狀態過濾器）都在自適應這端。**裁決：自適應是真需求，但它把過擬合從因子層搬到了門控層**。AlphaForge 的滾動 IC 權重本質是短週期動量路由，在低信噪比期會產生高 turnover（權重震盪）；FactorMoE 自承若風格切換頻率短於門控更新週期，Top-k 掩碼會滯後截斷。動態組合必須配 turnover 懲罰，否則交易成本吃掉你自適應賺來的那點 alpha。

**張力三：黑盒榨取 vs 可歸因定價。** [PTK](../foundations/factor-mining/ptk.md) 與 [alpha-vs](../foundations/factor-mining/alpha-vs.md) 是這條張力的兩個最深的思考。PTK 的解法是「解耦」——特徵交給 DNN、權重交給核方法/馬科維茨，於是黑盒只留在特徵層。alpha-vs 的解法是「端到端可微優化器」——把組合約束編碼進損失的梯度方向，讓預測網絡直接優化組合績效而非 MSE。**裁決：兩者不矛盾，是同一問題的兩種切法**。解耦（PTK）在估計效率與可審計性上贏，端到端（alpha-vs）在預測-分配對齊上贏；選哪個取決於你的市場是「估計雜訊主導」還是「兩階段錯配主導」。對多數實盤團隊，PTK 式解耦是更安全的默認。

**張力四：因子生成能力 vs 過濾管線嚴苛度。** 這是第 4 代揭示的最反直覺的張力。[LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) 與 [factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md) 站在同一立場：生成不是瓶頸，過濾才是。前者用 FF5/q-factor/異象賽馬三重過濾把 280 個候選砍到個位數；後者證明 2.9 萬個數據挖掘因子在嚴格多重測試校正下，樣本外預測力不遜於同行評審因子。**裁決：這條張力的贏家明確——把研發預算從「造更多因子」轉到「更狠的多重檢驗校正 + 已知因子吸收」**。多重測試校正才是護城河，不是模型參數量。

**張力五：擁擠是 alpha 信號還是風險信號。** [comprehensive-crowded-factor](../foundations/factor-mining/comprehensive-crowded-factor.md) 與 [art-344](../foundations/factor-mining/art-344.md) 給出對立又互補的答案。前者（UBS）認為擁擠度的「跳變」是可交易的順勢 alpha，因為資金流入的推升效應非對稱地壓倒撤離的壓迫效應；後者用博弈論證明擁擠度**不預測收益**、只預測尾部崩盤概率。**裁決：兩者其實不衝突**——UBS 賺的是「進入擁擠過程」的順勢動量（跳變事件），art-344 說的是「已擁擠水平」不能做均值擇時。實盤結論：擁擠度的變化率可接信號模塊，擁擠度的絕對水平只接風險預算模塊。混用兩者就是 art-344 的錯答。

## 什麼在持續有效 vs 什麼被擁擠掉

跨頁面歸納，有結構性理由**持續賺錢**的機制通常滿足「有人被迫付錢 / 有結構性摩擦」：

- **信息擴散的滯後**。[Proj-SUE](../foundations/factor-mining/proj-sue.md) 的溢出效應來自投資者注意力有限、同業超預期信息沿業務網絡緩慢擴散；[alpha 內部人周內擇時](../foundations/factor-mining/alpha.md) 的信號只在異常放量（高關注度）時激活。這類 alpha 的「付錢方」是注意力受限的散戶，只要人類注意力仍稀缺就難被完全套利。
- **正交的、可歸因的殘餘 alpha**。[art-351 協同智能](../foundations/factor-mining/art-351.md) 把分析師短期觀點做正交化後當純特質性 alpha 疊加；[HF9](../foundations/factor-mining/hf9-post-adaptive-lasso.md) 用正則化把偽 alpha 剝乾淨後留下的才是真技能。經過已知因子吸收後仍顯著的那一小塊，衰減得最慢。
- **結構化 / 理論收斂的因子**。[Zookeeper](../foundations/factor-mining/zookeeper.md) 用等權線性集成統一三大商品風險溢價理論、拒絕易過擬合的非線性——它賭的是「理論有支撐的溢價比數據挖掘的溢價命長」，這與 [factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md) 反直覺的發現形成張力（後者說風險型因子反而衰減更快），值得對照讀。

明顯**被擁擠掉 / 快速衰減**的：

- **已被 FF5/q-factor 吸收的重包裝因子**。[LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) 的鐵證：FMB 下 t=5.39 的因子，FF5 下 t=-0.37。這類「舊酒裝新瓶」上線即衰減。
- **公開、機械、易複製的機械因子**。[art-344](../foundations/factor-mining/art-344.md) 的雙曲線衰減對機械因子（動量/反轉）擬合極好、對判斷因子擬合極差——越機械、越公開、進入速率 $\lambda$ 越高，人均利潤衰減越快。2015 年後 ETF 資金涌入使 $\lambda$ 加速，模型系統性高估了殘餘 alpha。
- **依賴 regime 持續性的極端夏普**。[Unicorn Edge](../foundations/factor-mining/unicorn-edge.md) 的夏普高度依賴牛市觸發率（牛市 67%、崩盤 8%），且作者自承幸存者偏差高估業績 20-30%。這種 alpha 不是被擁擠掉，而是被 regime 切換一次性抹掉。
- **[經驗貝葉斯](../foundations/factor-mining/empirical-bayes.md) 的實證斷點**：會計因子與小盤股的可預測性在 2004 年信息技術普及後從高位（早期年化 8.17%）衰減到 2.03%。這是「技術普及消滅信息優勢」的教科書案例。

一句話總結：**賺的是別人的摩擦與被迫交易；一旦你的 alpha 機制變得公開、機械、無摩擦，它就是別人的成本、也很快是你的成本。**

## 失效模式合集（因子挖掘的坑）

把各頁 §6 的隱含假設歸納成一張清單。幾乎每一頁都在這張表上踩了至少三個坑，回測看起來越漂亮，越要逐項排查。

1. **前瞻偏差（Look-ahead）。** 會計數據的「報告日 vs 實際發佈日」不對齊是最常見的隱性泄漏（[factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md)、[LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) 都點名）；截面分位數標準化若用全樣本統計量計算會泄漏未來分佈（[PTK](../foundations/factor-mining/ptk.md)）；LLM 訓練語料可能已見過樣本期數據（[Proj-SUE](../foundations/factor-mining/proj-sue.md) 靠 2024 純樣本外測試排除記憶）；RAG 檢索的「專有 alpha 庫」若未按時間窗劃分會泄漏（[Alpha-GPT](../foundations/factor-mining/alpha-gpt.md)）。

2. **幸存者偏差（Survivorship）。** 幾乎所有頁面 §6 都承認未明確處理退市/ST 股。[Unicorn Edge](../foundations/factor-mining/unicorn-edge.md) 最誠實地量化了它——用當前成分股回測高估業績 20-30%。CRSP/Compustat 合併庫雖含退市股，但 delisting return 處理不當會低估尾部風險。

3. **容量與擁擠。** 公式化因子池容量普遍有限。[AlphaForge](../foundations/factor-mining/alphaforge.md) 因子池最佳大小僅 10；[Proj-SUE](../foundations/factor-mining/proj-sue.md) 的 20 日漂移意味大資金難在日頻重倉；[comprehensive-crowded-factor](../foundations/factor-mining/comprehensive-crowded-factor.md) 的 ≥3 分位跳變樣本極少、尾部流動性受限。回測夏普不含容量約束時是紙面數字。

4. **交易成本 / 滑點未計。** 這是全族最普遍的坑：幾乎每頁 §5 的 IC/Sharpe 都是毛值。[BRT-NN](../foundations/factor-mining/brt-nn.md) 給出最清晰的對照——同一套信號，基本面策略年調倉淨回報為正、歷史收益策略月調倉換手巨大導致淨回報持續為負。高換手因子（[Unicorn Edge](../foundations/factor-mining/unicorn-edge.md) 日均換手 42%）尤其危險。動態組合（[AlphaForge](../foundations/factor-mining/alphaforge.md)、[FactorMoE](../foundations/factor-mining/factormoe.md)）的權重 turnover 是隱藏成本源。

5. **樣本內過擬合 / 多重檢驗未校正。** 這是因子挖掘的原罪。[factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md) 與 [empirical-bayes](../foundations/factor-mining/empirical-bayes.md) 給出兩種解藥：前者是多重測試校正（Bonferroni/FDR），後者是經驗貝葉斯收縮（把 t 統計量向零收縮、剔除運氣項）。傳統 FDR 過保守（幾乎沒策略過閾），EB 是更透明的替代。凡是報單次回測極端夏普、不做多重校正的，默認過擬合。

6. **因子擁擠 / regime 依賴。** [art-344](../foundations/factor-mining/art-344.md) 證明擁擠信號只能做尾部風險預警不能做 alpha 擇時；幾乎每頁 §6 都承認 regime 切換會使歷史統計失效（AST 相似度、滾動 IC、宏觀-異象線性關係在 regime shift 時全部可能斷裂）。

7. **歸因模糊（組合模型吸收因子預測力）。** [Alpha2](../foundations/factor-mining/alpha2.md) 用 XGBoost 做組合、[BRT-NN](../foundations/factor-mining/brt-nn.md) 用 BRT 做加權，都存在「組合模型本身吸收了因子預測力」的歸因陷阱——你以為是因子好，其實是下游 ML 好。評估因子必須固定組合層。

## 讀法（reading paths）

**路徑 A：想立刻上生產的量化研究員（因子工廠 + 過濾）。** 目標是搭一條能跑的因子生產線，先學過濾、再學生成、最後學動態組合。
1. [factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md) — 建立「校正才是護城河」的心態
2. [empirical-bayes](../foundations/factor-mining/empirical-bayes.md) — 用 EB 收縮替代 t 值閾值篩選
3. [BRT-NN](../foundations/factor-mining/brt-nn.md) — 特徵工程 / 遞歸排序降維勝過黑盒
4. [AlphaForge](../foundations/factor-mining/alphaforge.md) — 生成器-預測器代理搜索 + 滾動 IC 組合
5. [HF9/Post-Adaptive-LASSO](../foundations/factor-mining/hf9-post-adaptive-lasso.md) — 正交化 / 剝離偽 alpha
6. [LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) — 工業化過濾流水線（多重檢驗 + 因子吸收 + 異象賽馬）
7. [PTK](../foundations/factor-mining/ptk.md) — 特徵與定價解耦，把深度學習安全落地

**路徑 B：做因子組合 / 風控的（正交化 + regime + 擁擠）。** 目標是把一堆因子拼成穩健組合、並監控它什麼時候會崩。
1. [PTK](../foundations/factor-mining/ptk.md) — 解耦框架：特徵交模型、權重交核方法
2. [alpha-vs](../foundations/factor-mining/alpha-vs.md) — 可微優化器 / 端到端預測-分配對齊 vs 兩階段
3. [FactorMoE](../foundations/factor-mining/factormoe.md) — 動態門控組合，保留可歸因通道
4. [Style Miner](../foundations/factor-mining/style-miner.md) — 約束 RL 構造高解釋力風格因子（風險模型用）
5. [art-344 雙曲線衰減](../foundations/factor-mining/art-344.md) — 擁擠度只接風險預算、不接信號
6. [comprehensive-crowded-factor](../foundations/factor-mining/comprehensive-crowded-factor.md) — 擁擠跳變的非對稱與 Long/Short 腿貢獻拆解
7. [HF9/Post-Adaptive-LASSO](../foundations/factor-mining/hf9-post-adaptive-lasso.md) — 稀疏因子結構做績效歸因

**路徑 C：想追前沿生成式 / RL 因子的（LLM Agent + RL 探索）。** 目標是理解 2025-2026 這一批 Agentic / 生成式框架到底多解了什麼。
1. [Alpha2](../foundations/factor-mining/alpha2.md) — DRL + MCTS 程序生成 + 維度一致性硬約束
2. [AlphaQCM](../foundations/factor-mining/alphaqcm.md) — 分布 RL 用無偏方差引導探索破解稀疏獎勵
3. [Alpha-GPT](../foundations/factor-mining/alpha-gpt.md) — LLM 語義種子 + GP 符號搜索混合
4. [AlphaAgent](../foundations/factor-mining/alphaagent.md) — 三智能體閉環 + AST 相似度懲罰抗擁擠
5. [AlphaAgentEvo](../foundations/factor-mining/alphaagentevo.md) — GRPO + 層次化獎勵 + AST 鄰域約束防 Reward Hacking
6. [Proj-SUE](../foundations/factor-mining/proj-sue.md) — LLM 嵌入構建動態同業網絡（生成式的另一種用法）
7. [LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) — 冷水澡：生成力過剩、存活率極低

## 開放問題 / 值得下注的方向

以下每條都寫成可證偽的判斷，帶理由。

1. **抗衰減會成為比 IC 更重要的一級指標，且「AST/結構約束」會被證明是必要但不充分的。** 理由：[AlphaAgent](../foundations/factor-mining/alphaagent.md)/[AlphaAgentEvo](../foundations/factor-mining/alphaagentevo.md) 已把抗衰減當賣點，但 AST 相似度只抓語法同構、抓不到功能等價，且熊市 Pass@3 明顯下滑。**可證偽點**：若某個純 AST 約束框架在 2026-2027 一個完整牛熊週期裡，樣本外 IR 衰減率不顯著低於無約束基線，則「結構約束足以抗衰減」被證偽——我賭它會被證偽，真正抗衰減的是「經濟功能去重 + 已知因子吸收」，不是語法去重。

2. **多重檢驗校正 / 貝葉斯收縮會取代單次 t 值成為因子上線的默認閘門。** 理由：[factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md) 與 [empirical-bayes](../foundations/factor-mining/empirical-bayes.md) 已證明校正後數據挖掘不遜於學術因子，而生成式框架讓候選因子數量爆炸、選擇偏差被放大。**可證偽點**：若未來三年主流開源因子平台仍以單次回測夏普為主要排序、且 EB/FDR 校正未成為標配，則此判斷被證偽。

3. **「解耦」（特徵 vs 定價分離）會比「端到端」在實盤佔上風。** 理由：[PTK](../foundations/factor-mining/ptk.md) 指出實盤真正的坑是定價權重過擬合，而 [alpha-vs](../foundations/factor-mining/alpha-vs.md) 自承端到端優化器（IPMO/多期）無現成庫、收斂穩定性未驗證。**可證偽點**：若端到端可微優化器在 N>500 的大股票池上被證明穩定收斂且扣成本後穩定超越解耦流水線，則此判斷被證偽。

4. **擁擠度的正確用法會收斂到「風險預算模塊」，把它當 alpha 信號的做法會系統性虧錢。** 理由：[art-344](../foundations/factor-mining/art-344.md) 的博弈論證明 + 樣本外夏普落後因子動量，直接證偽了擁擠擇時的 alpha 生成能力；[comprehensive-crowded-factor](../foundations/factor-mining/comprehensive-crowded-factor.md) 的順勢 alpha 其實賺的是「進入擁擠過程」的動量而非「擁擠水平」本身。**可證偽點**：若基於實時擁擠殘差的多空組合扣成本後夏普穩定顯著為正，則被證偽。

5. **文本 / 另類數據的生成式因子（LLM 嵌入路線）比 LLM 寫公式路線更有結構性空間。** 理由：[LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) 證明在量價 + 會計語法空間裡 LLM 幾乎無法憑空造新溢價（市場已被佔滿），而 [Proj-SUE](../foundations/factor-mining/proj-sue.md) 用嵌入開出的是「動態語義同業網絡」這一量價之外的新數據面。**可證偽點**：若在同一嚴格過濾管線下，LLM 寫公式的存活率追平或超過 LLM 嵌入/另類數據路線，則此判斷被證偽。

## 代表頁面索引

**先讀（⚡核心）** — 建立這一族的骨架與最深的張力：
- [PTK](../foundations/factor-mining/ptk.md) — 特徵/定價解耦，深度因子的安全落地範式
- [AlphaAgent](../foundations/factor-mining/alphaagent.md) — LLM + AST 約束的抗衰減閉環
- [AlphaAgentEvo](../foundations/factor-mining/alphaagentevo.md) — GRPO + 層次化獎勵 + AST 鄰域約束
- [AlphaForge](../foundations/factor-mining/alphaforge.md) — 生成器-預測器代理搜索 + 動態組合
- [AlphaQCM](../foundations/factor-mining/alphaqcm.md) — 分布 RL 無偏方差探索
- [Alpha-GPT](../foundations/factor-mining/alpha-gpt.md) — LLM 語義錨定 + GP 混合
- [empirical-bayes](../foundations/factor-mining/empirical-bayes.md) — 貝葉斯收縮校正數據挖掘偏差
- [Proj-SUE](../foundations/factor-mining/proj-sue.md) — LLM 嵌入構建動態同業溢出因子
- [alpha-vs](../foundations/factor-mining/alpha-vs.md) — 範式選型決策樹（Alpha Mining vs DL）
- [CogAlpha](../foundations/factor-mining/cogalpha.md) — LLM 認知推理 + 進化算法（代碼級變異）
- [AlphaSAGE](../foundations/factor-mining/alphasage.md) — RGCN 結構感知 + GFlowNets 多樣性生成
- [AlphaSharpe](../foundations/factor-mining/alphasharpe.md) — LLM + 進化策略生成風險調整指標
- [Alpha-R1](../foundations/factor-mining/alpha-r1.md) — GRPO 對齊推理模型做上下文條件因子門控

**再讀（重要🔧）** — 過濾、組合、擁擠、RL 探索的關鍵補充：
- [factor-zoo-vs-peer-review](../foundations/factor-mining/factor-zoo-vs-peer-review.md) — 校正後數據挖掘不遜於學術因子
- [LLM Factor Lab](../foundations/factor-mining/llm-factor-lab.md) — 工業化過濾管線與真實存活率
- [FactorMoE](../foundations/factor-mining/factormoe.md) — 鏈式 MoE 動態門控組合
- [art-344 雙曲線衰減](../foundations/factor-mining/art-344.md) — 擁擠度只預測尾部風險不預測收益
- [comprehensive-crowded-factor](../foundations/factor-mining/comprehensive-crowded-factor.md) — 擁擠跳變的非對稱 alpha
- [HF9/Post-Adaptive-LASSO](../foundations/factor-mining/hf9-post-adaptive-lasso.md) — 稀疏正交化剝離偽 alpha
- [Alpha2](../foundations/factor-mining/alpha2.md) — DRL + MCTS 程序生成 + 維度一致性
- [AlphaEvolve](../foundations/factor-mining/alphaevolve.md) — AutoML + 進化 + RelationOp 圖節點
- [Style Miner](../foundations/factor-mining/style-miner.md) — 約束 RL 構造風格因子
- [BRT-NN 遞歸排序](../foundations/factor-mining/brt-nn.md) — 特徵工程 / 歸納偏置勝過黑盒
- [Unicorn Edge](../foundations/factor-mining/unicorn-edge.md) — 狀態過濾器條件激活舊因子
- [QFR](../foundations/factor-mining/qfr.md) — 低方差 REINFORCE + IR 獎勵塑造
- [HPPO-TO](../foundations/factor-mining/hppo-to.md) — 分層 RL + 遷移學習挖日內因子
- [LLM+MCTS Alpha Mining](../foundations/factor-mining/llm-mcts-alpha-mining.md) — LLM + MCTS 回測反饋引導搜索

**參考** — 特定機制、評估工具與周邊：
- [NeuralFactors](../foundations/factor-mining/neuralfactors.md) — VAE/CIWAE 聯合學習因子暴露與收益
- [Fama-MacBeth](../foundations/factor-mining/fama-macbeth.md) — 散戶異質性與收益預測力
- [alpha 內部人周內擇時](../foundations/factor-mining/alpha.md) — 高關注度激活的非線性 alpha
- [art-351 協同智能](../foundations/factor-mining/art-351.md) — 分析師短期觀點正交疊加
- [Zookeeper](../foundations/factor-mining/zookeeper.md) — 線性集成統一商品風險溢價理論
- [Chatterjee ξ](../foundations/factor-mining/chatterjee.md) — 非參數相關系數，補 IC 在非線性上的失效
- [factor-asymmetry-bear-market](../foundations/factor-mining/factor-asymmetry-bear-market.md) — 因子非對稱性與熊市空頭端
- [art-21 RL 篩因子](../foundations/factor-mining/art-21.md) — 把因子選擇建模為 MDP
