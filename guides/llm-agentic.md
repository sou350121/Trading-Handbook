# LLM-Agentic 交易全景：從新聞情緒讀取到多智能體對話決策

> 這一族把 LLM 塞進交易流程的每個環節：讀非結構化敘事、生成因子、多角色辯論、乃至自主執行。真價值在「處理長尾非結構化資訊」的少數幾層，其餘多是把幻覺、延遲、與 look-ahead 洩漏包裝成新架構的 demo。本文替一個對 LLM 炒作免疫的量化研究員拆開這 50 篇，指出哪幾層 alpha 是真的、哪幾層一放到樣本外就露餡。

## 這一族在解什麼問題

傳統量化的信息通道是結構化的：價量表格進因子、因子進組合。LLM 進交易真正打開的通道只有一個 —— 讀那些「無法塞進 float 向量」的東西：財報電話會議的管理層語氣、社群媒體快訊、研報敘事、跨句因果鏈。這一點在多篇頁面裡是共識而非炒作。[HybridRAG](../foundations/llm-agentic/hybridrag.md) 把電話會議 Q&A 的管理層指引與供應鏈關聯轉成可回測特徵；[art-34](../foundations/llm-agentic/art-34.md) 用實證量化研報自動化邊界（78.7% 內容可自動提取/生成，21.3% 需人工判斷），這是把「LLM 讀文本」從口號變成可審計分工的少數硬結論之一。凡是這一類「非結構化 → 結構化因子」的預處理，是這一族的結構性正資產。

這裡要先破一個常見誤解：LLM 的優勢不是「預測」，而是「翻譯」。它不擅長把價格序列回歸成下一個數值 —— [AlphaBench](../foundations/llm-agentic/alphabench.md) 直接證明讓 LLM 對因子零樣本絕對評分是數學上的欠定問題，準確率等同擲硬幣。它擅長的是把人類語言裡的因果敘事翻譯成結構。[LLMFactor](../foundations/llm-agentic/llmfactor.md) 的直覺說得最透：LLM 不擅長直接回歸數值，但極擅長因果敘事，SKGP 就是把市場驅動邏輯「翻譯」成 LLM 的強項領域再推理。認清這一點，就能過濾掉全族一半的炒作 —— 凡是宣稱 LLM 能「預測漲跌」的，多半在濫用它；凡是讓 LLM 做「讀-抽-結構化」的，才用在刀口上。

主線張力在於：一旦離開「讀敘事」這條窄路，LLM 加的東西大多是包裝。多智能體辯論、CoT 推理鏈、角色扮演風控官 —— 這些讓決策鏈「看起來可解釋」，但幾乎沒有一篇能證明辯論本身產生了單模型基準之外的 alpha。更危險的是三個被關錯層的問題：**幻覺**（LLM 編造不存在的財報數字或記憶索引，[FINCON](../foundations/llm-agentic/fincon.md) 自述「記憶體事件不存在索引」）、**延遲與成本**（多輪對話流水線的 RTT 動輒數十秒，[TiMi](../foundations/llm-agentic/timi.md) 實測 TradingAgents 端到端延遲 25071ms）、以及最致命的 **look-ahead 洩漏**（LLM 的預訓練語料很可能已涵蓋回測期，模型「知道未來」）。這一族最有價值的工作，不是又發明一個 agent 拓撲，而是把這三個問題各自關進正確的層 —— 而多數 demo 恰恰是把它們藏起來。

## 方法譜系（演進主線）

**節點一：情緒/事件抽取（把文本壓成一個分數）。** 最早的用法是把 LLM 當更聰明的情緒分類器。[FinBERT & LLM Prompting](../foundations/llm-agentic/finbert-llm-prompting.md) 證明 9 條金融語境 few-shot 示例能讓通用 GPT 逼近領域微調的 FinBERT（0.88 準確率基準），把垂直領域從「重微調」拉到「輕提示」；[CopBERT/CopGPT](../foundations/llm-agentic/copbert-copgpt.md) 則在商品新聞上對比雙向編碼與單向生成，指出 GPT 預測力更強但幻覺更多。這一節多解了「標註成本」與「跨語境遷移」，但沒解「情緒分數→PnL」的轉化率，信號一到聚合層就衰減（[llm-financial-sentiment](../foundations/llm-agentic/llm-financial-sentiment.md) 單句 F1 89.5%，聚合成日度信號後 F1 僅 54.21%）。

**節點二：從情緒到可執行因子（讓 LLM 生成結構，而非分數）。** 進化方向是不再要 LLM 直接吐分數，而是吐「結構化因子」或「市場驅動標籤」。[LLMFactor](../foundations/llm-agentic/llmfactor.md) 用 SKGP 三階段提示（填空補背景→定向抽因子→時序轉文本聯合推理）把新聞轉成可審計因子；[llm-financial-sentiment](../foundations/llm-agentic/llm-financial-sentiment.md) 更激進，用價格實際觸及動態止盈/止損屏障替代人工情緒標籤，讓標籤天生內嵌風控邏輯。這一節多解了「情緒標籤與交易閾值脫節」，把 LLM 從預測器降維為因子生成器 —— 這是全族最穩健的方向之一，因為回測引擎接管了物理約束。

**節點三：單體 LLM 自主決策（推理鏈 + 自主檢索）。** [2-43-ai（北大光華）](../foundations/llm-agentic/2-43-ai.md) 是這一節的標竿：自主聯網 LLM 每日對 Russell 1000 生成 -5 到 +5 吸引力評分，Top-20 組合報出日度 Alpha 0.184%（t=2.46）、年化夏普 2.43。它的關鍵不在推理鏈，而在 Nowcasting 設計（T-1 收盤後輸入、T 開盤前完成）物理切斷前視偏差，以及「不可復現但可審計」的實時搜索工作流。多解了「人工預篩信息集帶來的歸因偏差」，但代價是 Bottom-20 組合 Alpha 與零無異 —— 它只會識別贏家，構不出多空。

**節點四：檢索增強與記憶（RAG / 工具索引 / 經驗池）。** 當工具與知識爆炸，提示塞不下，檢索登場。[HybridRAG](../foundations/llm-agentic/hybridrag.md) 融合向量（抓廣度）與知識圖譜（抓深度）雙路檢索；[RAG-MCP](../foundations/llm-agentic/rag-mcp.md) 把工具選擇正交解耦為向量最近鄰檢索，只注入 top-1 工具模式避免提示膨脹。記憶則往經驗累積走：[XALPHA](../foundations/llm-agentic/xalpha.md) 的 Cross Brain FAA 歸因與 GOOD/BAD 記憶池打破「每次生成視為獨立事件」的冷啟動。多解了「上下文窗口硬限制」與「經驗不可累積」，但引入了檢索器語義對齊誤差這個新失效點。

**節點五：多智能體對話決策（分析師/辯論/風控角色）。** 譜系主幹。[TradingAgents](../foundations/llm-agentic/tradingagents.md) 把投行分工映射為 LangGraph 五階段狀態機，引入看漲/看跌結構化辯論；[AlphaAgents](../foundations/llm-agentic/alphaagents.md)（BlackRock）用 AutoGen Round Robin 群聊強制跨模態交叉驗證；[FINCON](../foundations/llm-agentic/fincon.md) 加經理-分析師分層與雙層 CVaR/CVRF 風控；[HedgeAgents](../foundations/llm-agentic/hedgeagents.md) 把對沖基金三類會議機制搬進來；[MASS](../foundations/llm-agentic/mass.md) 與 [ContestTrade](../foundations/llm-agentic/contesttrade.md) 則把「協作」翻轉為「規模化模擬」與「內部競賽淘汰」。多解了「單體 LLM 認知過載與黑箱」，提供了可審計的決策日誌 —— 但辯論是否真加了 alpha，是全族最大的懸案（見下節張力）。

**節點五點五：把新聞閉環進時序預測（agent 迴圈當數據清洗器）。** 介於「因子生成」與「多智能體」之間，有一條容易被忽略的支線：不讓 LLM 直接預測，而讓它當非結構化數據的動態過濾器。[llm-news-to-forecast](../foundations/llm-agentic/llm-news-to-forecast.md) 設計推理 agent（過濾/分類新聞）+ 評估 agent（用預測誤差反饋優化過濾邏輯）的雙閉環，把新聞轉成條件 token 餵給微調 LLM。它的洞見是：真 alpha 不來自 LLM 的生成能力，而來自評估 agent 反饋構建的「新聞-波動」因果過濾鏈。這比單純的情緒分數進了一步 —— 但它自己也承認，若粗粒度新聞與本地化資產錯配，這條鏈只能當事件驅動的輔助特徵管道，而非端到端信號。這一節多解了「非結構化數據的動態對齊與降噪」，卻沒解「agent 迴圈的淨 alpha 是否高於靜態新聞因子基線」這個根本問題。

**節點六：工具使用與自主執行（把推理與執行解耦）。** 最成熟的收尾方向是承認 LLM 不該在線上做決策。[TiMi](../foundations/llm-agentic/timi.md) 把 LLM 徹底降為離線代碼生成器與約束求解器，在線只跑無狀態 CPU 腳本（延遲 137ms vs TradingAgents 25071ms）；[LLMOE](../foundations/llm-agentic/llmoe.md) 讓 LLM 只當路由器（輸出樂觀/悲觀標籤），數值擬合交給 FNN 專家（MSFT TR 65.44 vs MLP 33.92，SR 2.14 vs LSTM 1.39，但用的是「全入全出」倉位、未計滑點）。[RAG-MCP](../foundations/llm-agentic/rag-mcp.md) 也屬這一節的工具側 —— 把工具發現正交解耦為向量檢索，只注入 top-1 工具模式，避免 agent 擴展到數千工具時觸發 NIAH 式召回崩潰。多解了「在線推理延遲與滑點吞噬利潤」以及「工具集爆炸下的提示膨脹」—— 這是把延遲與記憶負載關進離線/外部層的正確工程姿勢。

這條譜系的內在邏輯值得一句總結：**從節點一到節點六，是一部「LLM 逐步被剝奪決策權」的歷史**。早期讓 LLM 直接吐分數、吐買賣、吐辯論結論；成熟的工作反而一步步把 LLM 逼回它真正擅長的環節 —— 讀敘事、生成結構、當路由器、離線編譯 —— 而把數值、約束、執行交還給確定性組件。凡是逆這個方向走（讓 LLM 管得越多）的框架，樣本外通常越脆。

## 三到五條核心張力

**張力一：單體推理 vs 多智能體辯論真的更好嗎。** 這是全族最貴的信仰。多智能體派（[TradingAgents](../foundations/llm-agentic/tradingagents.md)、[AlphaAgents](../foundations/llm-agentic/alphaagents.md)）主張辯論強制輸出權衡過程、降低單維過擬合方差。但翻開它們的 §5：TradingAgents 的 30.5% 年化只來自單一標的 AAPL 特定區間，非跨市場統計顯著性；AlphaAgents 在 15 只隨機科技股上 4 個月回測，導讀連具體夏普都沒給。**裁決**：辯論的價值是「方差縮減與可審計」，不是「alpha 生成」。[AlphaAgents](../foundations/llm-agentic/alphaagents.md) 自己的頁面最誠實 —— 它是「信號過濾器與合規審計層」，該嵌進 Black-Litterman 當 View Generator，而非下單引擎。對抗性辯論反而放大過擬合風險，因為 prompt 極易在歷史數據上找到事後合理的敘事。多智能體是組織形態的升級，不是 alpha 來源的升級。

**張力二：LLM 生成因子 vs LLM 直接決策。** 兩條路涇渭分明。生成因子派（[XALPHA](../foundations/llm-agentic/xalpha.md)、[AlphaBench](../foundations/llm-agentic/alphabench.md)、[RD-Agent(Q)](../foundations/llm-agentic/rd-agent-q.md)、[LLMFactor](../foundations/llm-agentic/llmfactor.md)）把 LLM 當「受約束的假說驗證器」，回測引擎接管物理約束；直接決策派（[FinRLlama](../foundations/llm-agentic/finrllama.md)、[FinAgent](../foundations/llm-agentic/finagent.md)）讓 LLM 直接輸出買賣。**裁決**：生成因子這條路結構上更穩。[AlphaBench](../foundations/llm-agentic/alphabench.md) 給出決定性證據 —— 純文本讓 LLM 零樣本絕對評分因子時，準確率徘徊 0.46-0.52，等同擲硬幣；一旦改成 SFT 成對比較 + AST 結構輸入，GPT-4.1-Mini 從 0.44 躍到 0.83，跨市場 SP500 仍守住 0.64。這證明 LLM 的價值在「相對排序算子結構」，不在「絕對數值預測」。直接決策派則反覆撞牆：[FinRLlama](../foundations/llm-agentic/finrllama.md) 只有 NVDA 累計回報過 1.5，XOM 直接跌成負，且自述訓練用了「三天前向收盤價」——這是教科書級的 look-ahead 洩漏。

**張力三：記憶/回測數據的 look-ahead 洩漏關在哪一層。** LLM 進交易最陰險的坑：模型知道未來。有三種來源 ——（a）預訓練語料涵蓋回測期；（b）新聞/財報時間戳與開盤未對齊；（c）因子代碼裡的隱性未來函數。**裁決**：只有物理隔離管用，事後聲明無效。正面教材是 [DeepFund](../foundations/llm-agentic/deepfund.md) —— 它索性放棄歷史回測，強制知識截止日期後的實時數據輸入，從源頭切斷「時間旅行」；結果殘酷：9 個模型只有 1 個（Grok 3 mini Beta）錄得 +1.1%，多數淨虧損。[2-43-ai](../foundations/llm-agentic/2-43-ai.md) 的 Nowcasting 是另一種物理隔離。技術層面，[XALPHA](../foundations/llm-agentic/xalpha.md) 的動態噪聲擾動測試（在未來收益序列注入噪聲，若因子值隨之變則判定泄露）是把洩漏審計從靜態代碼審查升級為運行時因果斷言 —— 但它自己也承認防不住「截面內未來信息洩漏」。凡是還在歷史回測上報漂亮夏普又不談語料污染的，一律降權。

**張力四：LLM 該在線上推理還是離線編譯。** [TiMi](../foundations/llm-agentic/timi.md) 給了這一族最反直覺也最務實的答案：把 LLM 從「交易員」降級為「離線代碼生成器 + 風險數學建模師」，實盤只跑確定性腳本。對照組是 TradingAgents 25071ms 的端到端延遲 —— 在分鐘級以上調倉這已是致命傷。**裁決**：延遲必須關進離線層。TiMi 把「調參」轉成「求解可行域」（線性/對數約束），在線 O(1) 無狀態執行，這是全族少見的機械理性。但它的代價很誠實：離線約束推導無法實時捕捉結構性斷裂，且回測到實盤 ARR/SR 普遍下滑（主流幣 SR 從 1.25 降到 0.79），暗示約束過擬合 2024 波動特徵。離線編譯解決延遲，不解決非平穩。

**張力五：多智能體協作 vs 內部競賽淘汰 vs 規模化模擬。** 就算接受多智能體有其位置，「多個 agent 該怎麼組織」本身也分三派，且互相打臉。協作派（[TradingAgents](../foundations/llm-agentic/tradingagents.md)、[FINCON](../foundations/llm-agentic/fincon.md)）讓 agent 分工並辯論收斂；競賽派 [ContestTrade](../foundations/llm-agentic/contesttrade.md) 明確批評靜態協作 —— 它用 ZI Trader 零智能交易者剝離 LLM 推理噪聲，純靠原子陳述的模擬 P&L 量化因子內在價值，再交 LightGBM 預測、0/1 背包分配，本質是把「評審會」自動化並用實時 P&L 淘汰劣質輸出；規模派 [MASS](../foundations/llm-agentic/mass.md) 則直接放棄協作與競賽，用 512 個異質 agent 端到端反向優化分佈，聲稱驗證了多智能體規模定律。**裁決**：競賽/量化剝離派邏輯上最站得住。理由是它承認了一個殘酷事實 —— 市場噪聲能欺騙 LLM 的語義理解，但無法欺騙嚴格受限的模擬交易結果。[ContestTrade](../foundations/llm-agentic/contesttrade.md) 的 Sigmoid 上下文衰減建模（決策價值 = 信息量 × 推理能力，且推理能力隨上下文長度衰減）是全族少見的對 LLM 能力邊界的誠實量化。但三派共同的軟肋一樣：52.80% CR/3.12 SR 都未計交易成本與 A 股 T+1/漲跌停約束，RankIC 絕對值（0.054/0.079）偏低暗示 alpha 衰減快。組織形態的優化蓋不住信號本身的脆弱。

**張力六：多模態視覺輸入是真增益還是新過擬合面。** [FinAgent](../foundations/llm-agentic/finagent.md) 把 K 線圖與新聞文本統一為多模態輸入，報出單資產 92.27% 回報、相對基線 +84.39%。表面看多模態是純加分 —— 多一個模態多一份信息。**裁決**：對這條保持最高警惕。單資產 90%+ 的回報幾乎必然是過擬合或牛市貝塔，而多模態恰恰擴大了過擬合面：K 線圖的視覺特徵與新聞文本可以在歷史數據上找到無數事後合理的組合。FinAgent 頁面自己也承認組合管理屬 Future Work、僅限單資產/選股，且「LLM 隨機性與工具過擬合」需硬約束閘門。多模態的真價值（若有）在於處理「圖表 + 文本」這類人類分析師才能整合的異構信息，但它同時把 look-ahead 面從一個模態擴到兩個模態，審計難度倍增。

## 什麼在持續有效 vs 什麼是 demo

**結構性有效（跨頁面歸納）：** 第一，**非結構化長尾資訊的結構化預處理**。這是 LLM 唯一無可替代的優勢 —— 傳統模型讀不了電話會議語氣、跨句因果、研報敘事。[HybridRAG](../foundations/llm-agentic/hybridrag.md)（Faithfulness/Relevance 0.96、Context Recall 1）、[art-34](../foundations/llm-agentic/art-34.md)（財務/公司類問題 70.6%/54.6% 可自動化）、[LLMFactor](../foundations/llm-agentic/llmfactor.md) 都落在這條線上。第二，**LLM 當受約束的因子生成器**，回測引擎當物理約束層（[AlphaBench](../foundations/llm-agentic/alphabench.md)、[XALPHA](../foundations/llm-agentic/xalpha.md)、[RD-Agent(Q)](../foundations/llm-agentic/rd-agent-q.md)）。第三，**推理與執行解耦**（[TiMi](../foundations/llm-agentic/timi.md)、[LLMOE](../foundations/llm-agentic/llmoe.md) 的 router-expert 分離）。這三類的共同點：LLM 只負責它擅長的語義/結構環節，數值與執行交給確定性組件。

**回測漂亮但實盤露餡（demo）：** 凡是「單標的/極小樣本 + 漂亮絕對收益 + 不談成本語料污染」的，基本是 demo。[TradingAgents](../foundations/llm-agentic/tradingagents.md) 的 AAPL 30.5%、[FinAgent](../foundations/llm-agentic/finagent.md) 的單資產 92.27% 回報、[HedgeAgents](../foundations/llm-agentic/hedgeagents.md) 的 ARR 71.60%/SR 2.41 —— 這些數字的共同問題不是造假，而是**樣本外邊界模糊**：回測期極可能落在 LLM 訓練語料覆蓋範圍內，模型不是在預測而是在回憶。[DeepFund](../foundations/llm-agentic/deepfund.md) 是這條的照妖鏡：一旦強制知識截止日期後的實時數據，多數 LLM 立刻淨虧損。這解釋了一個殘酷規律 —— **同一批 LLM 交易框架，在歷史回測上動輒夏普 2-3，在真正的樣本外正向測試裡多數不賺錢**。差距的來源就是語料污染。[StockAgent](../foundations/llm-agentic/stockagent.md) 更誠實地把自己定位為「行為機制沙盒」而非交易引擎，因為它的合成市場根本沒有真實流動性與滑點。

**兩者之間的灰帶：** [2-43-ai](../foundations/llm-agentic/2-43-ai.md) 的夏普 2.43 是全族少數扛得住審視的 —— 因為它用 Nowcasting 物理隔離、多因子控制後 Alpha 仍顯著（t=2.46）。但即便如此，它也只在純多頭有效（Bottom-20 無 alpha），且高換手（57.4%）依賴極低摩擦（1.6 bps），流動性一收縮就失效。有效的東西通常很窄。

**一條可操作的判別法則：** 把任一 LLM 交易頁面的 §5 拆成兩問 —— 第一，這個數字是「相對排序/分類準確率」還是「絕對收益/夏普」？前者（[AlphaBench](../foundations/llm-agentic/alphabench.md) 的 pairwise 0.83、[art-34](../foundations/llm-agentic/art-34.md) 的提取 84%）通常是真 capability，因為它測的是 LLM 擅長的結構任務；後者（[FinAgent](../foundations/llm-agentic/finagent.md) 的 92.27%、[HedgeAgents](../foundations/llm-agentic/hedgeagents.md) 的 71.60%）多半含貝塔與語料污染。第二，樣本外是「歷史回測」還是「知識截止日後正向測試」？只有後者（[DeepFund](../foundations/llm-agentic/deepfund.md)、[2-43-ai](../foundations/llm-agentic/2-43-ai.md)、[ContestTrade](../foundations/llm-agentic/contesttrade.md) 的 2025 H1 純樣本外）才可信。兩問都過的頁面，全族不超過三分之一。

## 失效模式合集

跨頁面 §6 歸納「LLM 上交易的坑」：

- **時間洩漏（模型知道未來）：最普遍也最致命。** 三種形態：預訓練語料涵蓋回測期（[DeepFund](../foundations/llm-agentic/deepfund.md) 全篇在治這個）、時間戳未對齊（[TradingAgents](../foundations/llm-agentic/tradingagents.md)/[AlphaAgents](../foundations/llm-agentic/alphaagents.md) 的新聞爬取時間 vs 開盤）、代碼隱性未來函數（[XALPHA](../foundations/llm-agentic/xalpha.md) 的動態噪聲擾動測試專治此）。最赤裸的是 [FinRLlama](../foundations/llm-agentic/finrllama.md) 自述用「三天前向收盤價」做訓練對齊。
- **幻覺數字：** LLM 編造不存在的財報數字或記憶索引。[FINCON](../foundations/llm-agentic/fincon.md) 自述「記憶體事件不存在索引」；[CopBERT/CopGPT](../foundations/llm-agentic/copbert-copgpt.md) 指 GPT「邏輯自洽但事實錯誤」。緩解手段是 AST 硬過濾（[AlphaBench](../foundations/llm-agentic/alphabench.md)、[XALPHA](../foundations/llm-agentic/xalpha.md)）與置信度評分。
- **延遲與成本：** 多輪對話流水線的 RTT 是硬牆（[TiMi](../foundations/llm-agentic/timi.md) 測 TradingAgents 25071ms）。日度 512 次前向推理（[MASS](../foundations/llm-agentic/mass.md)）的 API 成本會直接侵蝕日頻 alpha。幾乎每篇 §6 都把「未計入 token/API 成本」列為隱含假設。
- **prompt 過擬合與不可複現：** 提示模板易對特定新聞語境過度適配（[llm-financial-sentiment](../foundations/llm-agentic/llm-financial-sentiment.md) 自述「提示工程易過擬合」）；LLM 版本更新導致結果漂移（[finbert-llm-prompting](../foundations/llm-agentic/finbert-llm-prompting.md) 自述「LLM 版本更新會導致結果波動」）。[2-43-ai](../foundations/llm-agentic/2-43-ai.md) 的實時搜索工作流「時間設計不可復現」既是防洩漏的特性，也是複現的死穴。
- **評測用了污染的基準與絕對評分幻想：** [AlphaBench](../foundations/llm-agentic/alphabench.md) 給出最重要的一課 —— 零樣本讓 LLM 對因子絕對評分是數學上的欠定問題，準確率等同隨機（0.46-0.52）。凡是拿 CoT 讓模型直接打 RankIC 分的評測，都在腦補。
- **成本與流動性假設不誠實：** 幾乎所有頁面都未建模滑點與衝擊成本。[ContestTrade](../foundations/llm-agentic/contesttrade.md) 的 52.80% CR/3.12 SR 未計交易成本；A 股 T+1 與漲跌停會直接阻斷信號執行。純多頭牛市貝塔常被誤報為 alpha（[2-43-ai](../foundations/llm-agentic/2-43-ai.md) 自述需警惕 Russell 1000 漲 26% 的貝塔疊加）。連做得最細的 [XALPHA](../foundations/llm-agentic/xalpha.md)（明確設 open 0.05%/close 0.15% 成本、Top-50/Drop-5 換手）也承認尾部流動性枯竭時 0.15% 的 close-side 假設可能低估真實衝擊。
- **合成市場的流動性幻覺：** 模擬派（[StockAgent](../foundations/llm-agentic/stockagent.md)、[MASS](../foundations/llm-agentic/mass.md)）的價格由 LLM 性格 prompt 與 BBS 信息流疊加生成，根本沒有真實訂單簿深度。[StockAgent](../foundations/llm-agentic/stockagent.md) 的破產機制簡單到「現金為負即清倉」，無保證金追繳與強制平倉延遲。這類環境能做行為壓力測試，但把它的盈虧分佈當實盤 alpha 是範疇錯誤。
- **聚合層衰減被系統性隱藏：** 單句/單事件的高分往往在聚合成日度信號後暴跌。[llm-financial-sentiment](../foundations/llm-agentic/llm-financial-sentiment.md) 的單句 F1 89.5% 聚合後只剩 54.21%（僅比 LSTM 基線高 5pp），這個落差在多數只報單句準確率的頁面裡被隱藏了。真正決定 PnL 的是聚合後的信號質量，不是分類器的原始準確率。

## 讀法（reading paths）

**路徑 A：想用 LLM 做情緒/另類數據因子。** 先建立「文本→分數」的基準與其衰減認知，再往「文本→結構化因子」升級，最後看檢索如何餵基本面上下文。
1. [finbert-llm-prompting](../foundations/llm-agentic/finbert-llm-prompting.md)（few-shot 逼近微調的基準線）
2. [CopBERT/CopGPT](../foundations/llm-agentic/copbert-copgpt.md)（生成 vs 編碼的可解釋性權衡）
3. [llm-financial-sentiment](../foundations/llm-agentic/llm-financial-sentiment.md)（市場驅動標籤 + 情緒聚合衰減的殘酷真相）
4. [LLMFactor](../foundations/llm-agentic/llmfactor.md)（SKGP 把新聞轉可審計因子）
5. [HybridRAG](../foundations/llm-agentic/hybridrag.md)（電話會議→結構化特徵的預處理器）
6. [art-34](../foundations/llm-agentic/art-34.md)（研報自動化的邊界量化，界定人機分工）

**路徑 B：研究多智能體架構。** 從最簡辯論到分層風控，再到競賽/模擬，最後看「解耦執行」如何反殺高延遲。
1. [TradingAgents](../foundations/llm-agentic/tradingagents.md)（分析師-辯論-執行-風控五階段狀態機，開源可跑）
2. [AlphaAgents](../foundations/llm-agentic/alphaagents.md)（Round Robin 群聊，把它當 View Generator 而非下單引擎）
3. [FINCON](../foundations/llm-agentic/fincon.md)（經理-分析師分層 + 雙層 CVaR/CVRF 風控）
4. [HedgeAgents](../foundations/llm-agentic/hedgeagents.md)（對沖基金三會議機制映射）
5. [ContestTrade](../foundations/llm-agentic/contesttrade.md)（把協作翻成內部競賽淘汰 + ZI Trader 剝離 LLM 噪聲）
6. [MASS](../foundations/llm-agentic/mass.md)（規模化模擬 + 反向優化分佈，多智能體的規模定律）
7. [TiMi](../foundations/llm-agentic/timi.md)（結論篇：把 LLM 踢出在線迴路，離線編譯 + 無狀態執行）

**路徑 C：關心可複現性與洩漏審計的懷疑論者。** 先看正向測試如何揭穿回測陷阱，再看物理隔離與運行時洩漏審計，最後理解「絕對評分」為何是欠定問題。
1. [DeepFund](../foundations/llm-agentic/deepfund.md)（強制知識截止後數據，揭示多數 LLM 淨虧損）
2. [2-43-ai](../foundations/llm-agentic/2-43-ai.md)（Nowcasting 物理隔離的正面教材，夏普 2.43 扛得住審視）
3. [XALPHA](../foundations/llm-agentic/xalpha.md)（動態噪聲擾動測試：運行時因果斷言防未來函數）
4. [AlphaBench](../foundations/llm-agentic/alphabench.md)（零樣本絕對評分是欠定問題的數學證明）
5. [FinRLlama](../foundations/llm-agentic/finrllama.md)（反面教材：自述用前向收盤價的 look-ahead）
6. [StockAgent](../foundations/llm-agentic/stockagent.md)（誠實把自己定位為沙盒，非交易引擎）

## 開放問題 / 值得下注的方向

**我不信的 alpha（附理由）：**
- **多智能體辯論本身產生超額收益。** 我不信。[TradingAgents](../foundations/llm-agentic/tradingagents.md) 的 30.5% 是單標的敘事過擬合，[AlphaAgents](../foundations/llm-agentic/alphaagents.md) 15 股 4 個月連夏普都不敢報。辯論的真價值是方差縮減與審計，把它當 alpha 引擎是類別錯誤。**可證偽判斷**：若這類框架在 12 個月內仍拿不出跨市場（SPY/QQQ）嚴格時序交叉驗證的 Sharpe，其「博弈優勢」就只是單標的過擬合。
- **歷史回測上夏普 2-3 的 LLM 直接交易框架。** 我高度懷疑。[DeepFund](../foundations/llm-agentic/deepfund.md) 已用正向測試證明多數 LLM 樣本外淨虧，差距來源就是語料污染。**可證偽判斷**：任何報出高夏普卻不做「知識截止日後正向測試」的框架，把它放到截止日後區間，夏普會顯著塌陷。

**值得下注的方向：**
- **物理隔離的正向測試會成為 LLM 交易的唯一可信基準。** [DeepFund](../foundations/llm-agentic/deepfund.md) 與 [2-43-ai](../foundations/llm-agentic/2-43-ai.md) 的 Nowcasting 指向同一結論：離線回測對 LLM 系統性失效，未來嚴肅工作必須跑實時正向測試。這是方法論層面最確定的下注。
- **LLM 當「受約束的因子生成器 + 硬代碼過濾」是唯一規模化的正路。** [AlphaBench](../foundations/llm-agentic/alphabench.md) 的 0.44→0.83 與 [XALPHA](../foundations/llm-agentic/xalpha.md) 的 IR 1.5368 都來自「LLM 生成 + AST 約束 + 回測驗證」，而非讓 LLM 直接決策。賭注：中頻因子庫的 LLM 化會沿這條路收斂，純 prompt 即策略的路線會被淘汰。
- **推理與執行解耦（離線編譯）是延遲問題的終局。** [TiMi](../foundations/llm-agentic/timi.md) 的 137ms 對 25071ms 不是漸進改良而是範式切換。賭注：任何需要在分鐘級以下執行的 LLM 策略，最終都會把 LLM 逐出在線迴路。
- **開放懸案：非結構化 alpha 的容量上限。** [2-43-ai](../foundations/llm-agentic/2-43-ai.md) 的高換手（57.4%）+ 低摩擦（1.6 bps）暗示 alpha 來自短視窗信息消化效率，這類優勢天生容量受限。值得研究的是：LLM 讀敘事的 alpha 究竟能承載多少資金，一放大就被自身衝擊成本吃掉的臨界點在哪。

## 代表頁面索引

**先讀（建立正確心智模型，三層各一支柱）：**
- [DeepFund](../foundations/llm-agentic/deepfund.md) — 正向測試揭穿回測陷阱，全族最重要的方法論
- [AlphaBench](../foundations/llm-agentic/alphabench.md) — LLM 該當相對排序器不是絕對裁判，因子生成的地基
- [TiMi](../foundations/llm-agentic/timi.md) — 推理/執行解耦，延遲問題的終局
- [2-43-ai（北大光華）](../foundations/llm-agentic/2-43-ai.md) — 少數扛得住審視的正面案例（Nowcasting + 夏普 2.43）

**再讀（架構與因子生成的主幹）：**
- [TradingAgents](../foundations/llm-agentic/tradingagents.md) — 多智能體開源標竿
- [XALPHA](../foundations/llm-agentic/xalpha.md) — 記憶驅動因子挖掘 + 運行時洩漏審計
- [FINCON](../foundations/llm-agentic/fincon.md) — 分層架構 + 雙層風控
- [ContestTrade](../foundations/llm-agentic/contesttrade.md) — 內部競賽 + ZI Trader 剝離 LLM 噪聲
- [LLMFactor](../foundations/llm-agentic/llmfactor.md) — 新聞轉可審計結構化因子
- [LLMOE](../foundations/llm-agentic/llmoe.md) — LLM 當路由器，router-expert 解耦
- [RD-Agent(Q)](../foundations/llm-agentic/rd-agent-q.md) — 因子-模型協同自動化研發

**參考（子方向深掘）：**
- [HybridRAG](../foundations/llm-agentic/hybridrag.md) / [RAG-MCP](../foundations/llm-agentic/rag-mcp.md) — 檢索增強與工具索引
- [llm-financial-sentiment](../foundations/llm-agentic/llm-financial-sentiment.md) / [finbert-llm-prompting](../foundations/llm-agentic/finbert-llm-prompting.md) / [CopBERT/CopGPT](../foundations/llm-agentic/copbert-copgpt.md) — 情緒/因子抽取基準線
- [AlphaAgents](../foundations/llm-agentic/alphaagents.md) / [HedgeAgents](../foundations/llm-agentic/hedgeagents.md) / [MASS](../foundations/llm-agentic/mass.md) — 多智能體變體
- [FinAgent](../foundations/llm-agentic/finagent.md) — 多模態端到端交易
- [StockAgent](../foundations/llm-agentic/stockagent.md) — 行為機制模擬沙盒
- [FinRLlama](../foundations/llm-agentic/finrllama.md) — RL 市場反饋對齊（附 look-ahead 反面教材）
- [llm-news-to-forecast](../foundations/llm-agentic/llm-news-to-forecast.md) — 新聞條件時序預測
- [art-34](../foundations/llm-agentic/art-34.md) — 研報自動化邊界量化
