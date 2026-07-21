# 執行與組合 RL 全景：把交易寫成 MDP 到底值不值

> RL 在交易的真正戰場不是「選股」，而是「執行、對沖、路由」這類有即時反饋、有明確約束的子問題；把整條收益曲線硬塞進一個 MDP 獎勵，回測越漂亮，實盤越危險。

## 這一族在解什麼問題

這 39 頁全部圍繞同一個核心動作：把某個交易任務重寫成馬可夫決策過程（MDP 或 POMDP），然後用策略梯度或 Q-learning 去優化。但「任務」的選擇差異巨大，且這個差異決定了成敗。三大戰場清楚可分：第一是最優執行與做市，即在給定要買賣的量下，怎麼拆單、掛單、對沖，代表為 [貝萊德 DRL 訂單執行](../foundations/reinforcement-learning/art-359.md)、[JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md)、[IaC-MARL](../foundations/reinforcement-learning/iac-marl.md)、[DeepScalper](../foundations/reinforcement-learning/deepscalper.md)；第二是組合配置，即在多資產間分配權重，代表為 [MetaTrader](../foundations/reinforcement-learning/metatrader.md)、[ReCAP](../foundations/reinforcement-learning/recap.md)、[HRT](../foundations/reinforcement-learning/hrt.md)、[R²-SAC](../foundations/reinforcement-learning/r-sac.md)；第三是端到端策略，即直接從行情吐出買賣方向甚至選股，代表為 [AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md)、[iRDPG](../foundations/reinforcement-learning/irdpg.md)、[DQN 期貨](../foundations/reinforcement-learning/dqn.md)。

主線張力貫穿全族：金融數據非平穩、信噪比極低，而 RL 天生樣本效率差、對超參與隨機種子敏感。這意味著任何在單一歷史路徑上「學會賺錢」的策略，都難以區分它學到的是結構性 Alpha 還是對特定行情的記憶。[MetaTrader](../foundations/reinforcement-learning/metatrader.md) 直接自陳「多次隨機種子訓練的性能標準差通常大於預測模型」，這句話應該貼在每個做 RL 交易的人的螢幕上。更深一層的張力是建模假設本身的代價：把交易寫成 MDP，等於假設狀態轉移與獎勵可以被一個馬可夫過程刻畫，但真實市場的價格衝擊、對手方適應、跨訂單現金互斥，全都違反這個假設。這一族最有價值的頁面，往往不是新算法，而是老老實實承認並繞開這個代價的工程設計。

## 方法譜系（演進主線）

**value-based（DQN 類）** 是最早落地的一支，因為交易的離散動作（多/空/平）天然契合 Q-learning。[DQN 期貨](../foundations/reinforcement-learning/dqn.md) 用波動率標準化回報做狀態、GBM/VG 模擬數據擴充訓練集；[TF-Agents](../foundations/reinforcement-learning/tf-agents.md) 與 [Auto.gov](../foundations/reinforcement-learning/auto-gov.md)（DeFi 抵押因子治理，DQN+目標網絡+優先經驗回放）延續這條線。多解了什麼：離散決策的可控性與訓練穩定性，但代價是無法輸出連續倉位或對沖比率。[DeepScalper](../foundations/reinforcement-learning/deepscalper.md) 用分支對抗 Q 網絡（BDQ）把價格分支與數量分支解耦，正是為了緩解離散動作組合爆炸（|A_price|×|A_qty|）這個 value-based 的死穴。

**policy-gradient / actor-critic（PPO/DDPG/SAC/TD3 類）** 是當前主流，因為組合權重與對沖比率是連續量。[HRT](../foundations/reinforcement-learning/hrt.md) 用 PPO 選股、DDPG 調倉；[TD3 期權對沖](../foundations/reinforcement-learning/td3.md) 用截斷雙 Q 抑制價值高估直接輸出連續對沖比率；[SAC-CNN-MHA](../foundations/reinforcement-learning/sac-cnn-mha.md) 與 [R²-SAC](../foundations/reinforcement-learning/r-sac.md) 走 SAC 熵正則路線；[QTMRL](../foundations/reinforcement-learning/qtmrl.md) 用 A2C 聯合優化策略與價值網絡。多解了什麼：連續動作空間下的細粒度倉位控制。但這一支引入了新坑，即高維連續動作的信用分配崩潰，這正是 [R²-SAC](../foundations/reinforcement-learning/r-sac.md) 放棄端到端輸出權重、改為「鬆弛動作+決策邏輯」兩階段的動機。

**分佈式/離線/持續 RL** 是對「金融不能安全在線探索」的回應。[MetaTrader](../foundations/reinforcement-learning/metatrader.md) 用雙層元學習與數據變換集成保守 TD 目標抑制離線 RL 的價值高估（本質是 CQL/IQL 家族的金融特化）；[ReCAP](../foundations/reinforcement-learning/recap.md) 用 CUSUM 動態切分市場制度、只存策略參數差值 Δθ 對抗 Alpha 衰減與災難性遺忘（持續學習支）。多解了什麼：把 RL 從「一次性訓練-部署」推向「在線演進」，同時避免真金白銀去探索。

**與預測/生成結合的混合體** 承認純 RL 探索太貴，選擇注入先驗。[iRDPG](../foundations/reinforcement-learning/irdpg.md) 用 Q-Filter 門控的行為克隆把 Dual Thrust 專家軌跡安全注入 RL；[R²-SAC](../foundations/reinforcement-learning/r-sac.md) 融合 TCN 拐點預測與 Hawkes-GAT 排序；[FinRL-DeepSeek](../foundations/reinforcement-learning/finrl-deepseek.md) 用 LLM 評分以接近 1 的係數線性擾動 PPO/CVaR-PPO；[AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md) 與 [Stock-Evol-Instruct](../foundations/reinforcement-learning/stock-evol-instruct.md)、[LLM x RL 從新聞學習](../foundations/reinforcement-learning/llm-x.md) 則把 LLM 本身變成 RL 策略體。多解了什麼：用先驗換樣本效率，站在經典策略肩膀上定向進化，而非無頭蒼蠅式隨機探索。

**多智能體/市場模擬** 是把博弈與並行算力推到極致的一支。[MacroHFT](../foundations/reinforcement-learning/macrohft.md) 按 regime 分解訓練子代理再由超代理路由；[OPHR](../foundations/reinforcement-learning/ophr.md) 拆成擇時 Agent 與對沖路由 Agent；[JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md) 與 [JAX-LOB](../foundations/reinforcement-learning/jax-lob.md)、[集成+GPU 並行模擬](../foundations/reinforcement-learning/ensemble-rl-gpu-sim.md) 則靠 JAX/PyTorch vmap 把環境向量化到 GPU，把訓練從「週」壓到「小時」。多解了什麼：讓大規模超參掃描、異構對手博弈與方差收斂在工程上可行,但這條線最容易誤把「算力紅利」當成「Alpha 能力」。

## 三到五條核心張力

**張力一：端到端 RL vs 預測+優化兩步法。** 一端是把選股/擇時/執行全塞進一個策略網絡端到端優化，如 [AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md)、[QTMRL](../foundations/reinforcement-learning/qtmrl.md)；另一端是明確拆成預測模組加執行模組，如 [R²-SAC](../foundations/reinforcement-learning/r-sac.md)（SAC 出粗略動作，TCN/Hawkes-GAT 做精煉）、[HRT](../foundations/reinforcement-learning/hrt.md)（PPO 選、DDPG 執行）、[FNN+DRL](../foundations/reinforcement-learning/fnn-drl.md)（FNN 選股、TCN-PPO 調倉）。裁決：在高維資產池上，端到端幾乎必然遭遇信用分配崩潰與動作慣性，[R²-SAC](../foundations/reinforcement-learning/r-sac.md) 與 [HRT](../foundations/reinforcement-learning/hrt.md) 的解耦不是保守，而是對維度災難的正面回應。端到端只在單資產或工具調用可統一為行動空間時（如 [AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md) 的工具增強 MDP）才具說服力，且它明確承認未處理跨資產協方差與組合優化。

**張力二：線上 RL vs 離線 RL（金融沒法安全探索）。** 金融的殘酷之處在於，探索一個壞動作就是真實虧錢，這使標準線上 RL 的試錯範式失效。離線 RL 的核心病是價值高估，即對訓練集沒見過的狀態-動作對給出虛高 Q 值。[MetaTrader](../foundations/reinforcement-learning/metatrader.md) 的裁決很優雅：用「數據變換集合的最小值」替代多個獨立 Q 網絡的集成來壓制高估，並解耦市場狀態 H（被動）與賬戶狀態 Z（主動），使對市場狀態施加變換不違反貝爾曼方程。這比盲目在線微調更安全。[iRDPG](../foundations/reinforcement-learning/irdpg.md) 的 Q-Filter 是另一種回應：只有當 Critic 確認專家動作價值更高時才計算行為克隆損失，避免被「完美交易」的事後諸葛信號帶偏。

**張力三：獎勵設計——夏普 vs 效用 vs 風險約束。** 這是全族最致命的張力，因為代理獎勵與真實投資目標的錯位會直接摧毀策略。反面教材極其清晰：[混合量子-經典 PPO](../foundations/reinforcement-learning/quantum-ppo-sector-rotation.md) 用「預測明日市值前 10 板塊」做代理獎勵，結果量子變體訓練獎勵高達 2883.30（超過經典 Transformer 的 2795.30），但實盤累積回報僅 93.14%、全面落後經典的 124.29%，作者自陳「獎勵-性能錯位是根本難題」。正面設計包括：[iRDPG](../foundations/reinforcement-learning/irdpg.md) 用微分夏普比率把單步收益對齊長期風險調整目標；[FinRL-DeepSeek](../foundations/reinforcement-learning/finrl-deepseek.md) 用 CVaR-PPO 顯式懲罰尾部；[TD3 期權對沖](../foundations/reinforcement-learning/td3.md) 在獎勵中內生化 PnL 波動懲罰。裁決：獎勵函數與 Sharpe/MDD 的數學對齊，優先級高於任何架構新穎性；高表達力模型在錯位獎勵下只會放大過擬合。

**張力四：模擬環境的真實性（市場衝擊/成交假設）。** RL 的所有結論都建立在模擬器之上，而模擬器的成交假設往往是最大的謊言。[JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md) 是誠實的極端案例:因為環境每步自動取消未成交訂單且獎勵未按交易量歸一化，「永不交易（DoNothing）」竟成了最優解,這暴露了模擬器設計如何直接決定策略行為。[JAX-LOB](../foundations/reinforcement-learning/jax-lob.md) 提供 GPU 加速的 LOB 撮合，但仍警告零填充與固定時間窗可能引入樣本外偏差。裁決：模擬器越接近真實 LOB 撮合優先級（如 [JAX-LOB](../foundations/reinforcement-learning/jax-lob.md)、[TradeMaster](../foundations/reinforcement-learning/trademaster.md) 把滑點/成本/槓桿硬編入 step），結論越可信;純數學向量化環境（如 [集成+GPU 模擬](../foundations/reinforcement-learning/ensemble-rl-gpu-sim.md)）一旦接入真實訂單簿微結構，並行效率與績效都會坍縮。

**張力五：黑盒 vs 可審計/可解釋。** 合規與歸因需求推動了一條可解釋支線。[ProtoHedge](../foundations/reinforcement-learning/protohedge.md) 用固定原型庫與相似度加權替換黑盒網絡，讓每個對沖決策可追溯至具體歷史場景，代價僅為相對黑盒 Deep Hedging 效用差 -0.40%（Black-Scholes 環境）與 -0.33%（隨機波動率環境）；[Logic-Q](../foundations/reinforcement-learning/logic-q.md) 用程序草圖嵌入專家趨勢知識、後驗調整 DRL 策略；[TINs](../foundations/reinforcement-learning/tins.md) 把 MACD 等指標重構為可解釋神經網絡。裁決：當監管或內部審計要求歸因穩定性時，這條線提供了幾乎無損的透明度替代方案，但原型庫的靜態性使其無法動態適應未見 regime。

## 什麼在持續有效 vs 什麼是回測幻覺

跨頁面歸納，一個清晰的規律浮現：RL 的結構性優勢集中在有明確即時反饋、有真實約束、Alpha 來源不是方向預測的子問題上。最優執行是最好的例子。[貝萊德 DRL 訂單執行](../foundations/reinforcement-learning/art-359.md) 明確指出「別把執行算法當 Alpha，它的 Sharpe 來自風險預算分配而非方向預測」,它用二次衝擊懲罰讓 Agent 在極端分鐘自動降頻避開滑點，這是傳統 TWAP/VWAP 依賴 U 型成交量曲線假設所做不到的。期權對沖同理：[TD3 期權對沖](../foundations/reinforcement-learning/td3.md) 在 c=0.1% 成本下累積回報 -45.51%、優於 Delta 對沖的 -85.64%,[期權對沖 8 算法對比](../foundations/reinforcement-learning/art-187.md) 發現到期才結算的稀疏獎勵下 MCPG 的 RSQP 為 0.8111、優於 TD 類算法。這些場景 RL 有效，因為它優化的是執行成本與風險敞口這種有物理反饋的量，而非猜漲跌。

反過來，端到端選股與組合配置是回測幻覺的重災區。原因有三，全部在 §6 反覆出現。其一，環境不真實：[FinRL](../foundations/reinforcement-learning/finrl.md) 自陳日頻 env 假設容量無限、成本固定，大資金上線必遭滑點侵蝕，其可證偽預測直言「若無公開報告顯示 FinRL 預設 env 在實盤扣除真實滑點後 Sharpe > 1.0，則證明其成本模型過於理想化」。其二，獎勵過擬合：[量子 PPO](../foundations/reinforcement-learning/quantum-ppo-sector-rotation.md) 的「高分低能」是教科書級案例。其三，樣本外太短或選擇偏差：[AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md) 樣本外窗口僅 30 交易日、僅測 5 支現存科技大盤股，存在顯著 survivorship bias。值得注意的中間地帶是持續學習：[ReCAP](../foundations/reinforcement-learning/recap.md) 給出 NAS100 測試期累積回報 164.89%、夏普 1.14，但它同時披露成本升至 20 bps 時 CR 降至 130.12%、SR 降至 1.02，這種「敢公開成本敏感性」的框架，其結論比只報一個漂亮數字的可信得多。

## 失效模式合集

跨全族 §6 歸納，RL 上交易的坑高度同質，值得逐條內化為 checklist。

- **sim-to-real 差距。** 幾乎每頁都栽在這裡。模擬器不建模真實 LOB 撮合優先級、冰山單、部分成交，導致 [JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md) 出現「永不交易最優」、[貝萊德執行](../foundations/reinforcement-learning/art-359.md) 警告「流動性幻覺」。純向量化 GPU 環境（[集成+GPU](../foundations/reinforcement-learning/ensemble-rl-gpu-sim.md)）加速是真、Alpha 是假。
- **獎勵駭客與獎勵錯位。** [量子 PPO](../foundations/reinforcement-learning/quantum-ppo-sector-rotation.md) 學會「刷分」而非穩健配置；[DeepScalper](../foundations/reinforcement-learning/deepscalper.md) 承認 Hindsight 獎勵在視野 h>180 時總回報反而下降,存在過擬合特定日內趨勢的風險。獎勵函數與真實目標正交，模型容量越大越糟。
- **樣本效率與種子敏感。** [MetaTrader](../foundations/reinforcement-learning/metatrader.md) 自陳多種子性能標準差大於預測模型;[art-88](../foundations/reinforcement-learning/art-88.md) 整篇就是為了對付「超參敏感導致單一驗證集僥倖過擬合」。
- **非平穩導致策略漂移。** [ReCAP](../foundations/reinforcement-learning/recap.md) 的 Alpha 衰減、[MacroHFT](../foundations/reinforcement-learning/macrohft.md) 的實盤 regime 切換滯後、[R²-SAC](../foundations/reinforcement-learning/r-sac.md) 的 TCN 拐點在政策突變時失效,都是分佈偏移的具體形態。凍結的基座在超長週期（[ReCAP](../foundations/reinforcement-learning/recap.md) 17 年）下會與當前市場產生不可逆漂移。
- **前瞻偏差與成交假設。** Hindsight/事後極值信號是重災區:[iRDPG](../foundations/reinforcement-learning/irdpg.md) 的 BC 用「一天內最高/最低價」做專家信號，若未嚴格按時間戳切分即吸收未來信息;[MacroHFT](../foundations/reinforcement-learning/macrohft.md) 的低通濾波、[HRT](../foundations/reinforcement-learning/hrt.md) 的新聞情緒時間對齊，都可能引入 look-ahead bias。
- **不可複現的隨機種子與訓練捷徑。** [OPHR](../foundations/reinforcement-learning/ophr.md) 的離線神諭初始化依賴未來 RV、實盤不可用;[TD3](../foundations/reinforcement-learning/td3.md) 每滾動窗口隨機搜索 5 組超參可能過擬合。RL 論文若不報告多種子方差，其單一數字近乎無意義。

## 讀法（reading paths）

**路徑一：做最優執行/做市的工程師。** 從真實反饋最強的子問題入手。① [貝萊德 DRL 訂單執行](../foundations/reinforcement-learning/art-359.md)（獎勵塑形控制衝擊 vs 價格風險的範式）→ ② [DeepScalper](../foundations/reinforcement-learning/deepscalper.md)（BDQ 動作分支+Hindsight 獎勵）→ ③ [JAX-LOB](../foundations/reinforcement-learning/jax-lob.md)（GPU LOB 模擬器底座）→ ④ [JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md)（異構對手博弈沙盒與「永不交易」教訓）→ ⑤ [IaC-MARL](../foundations/reinforcement-learning/iac-marl.md)（多訂單現金共享的意圖通信）→ ⑥ [MacroHFT](../foundations/reinforcement-learning/macrohft.md)（regime 條件路由）→ ⑦ [TD3 期權對沖](../foundations/reinforcement-learning/td3.md)（成本感知連續對沖）。

**路徑二：做組合配置 RL 的研究員。** 從解耦與泛化控制入手。① [FinRL](../foundations/reinforcement-learning/finrl.md)（環境 scaffolding 與成本內生化的起點）→ ② [HRT](../foundations/reinforcement-learning/hrt.md)（選股/執行分層解耦）→ ③ [R²-SAC](../foundations/reinforcement-learning/r-sac.md)（鬆弛-精煉兩階段對抗信用分配崩潰）→ ④ [MetaTrader](../foundations/reinforcement-learning/metatrader.md)（雙層元學習+保守 TD 抗離線高估）→ ⑤ [ReCAP](../foundations/reinforcement-learning/recap.md)（CUSUM 持續學習抗 Alpha 衰減）→ ⑥ [FinRL-DeepSeek](../foundations/reinforcement-learning/finrl-deepseek.md)（LLM 評分擾動+CVaR 尾部約束）→ ⑦ [TradeMaster](../foundations/reinforcement-learning/trademaster.md)（標準化評測基準）。

**路徑三：懷疑 RL 選股的驗證派。** 從「怎麼證偽一個漂亮回測」入手。① [art-88 應對過擬合](../foundations/reinforcement-learning/art-88.md)（把過擬合形式化為假設檢驗+組合交叉驗證）→ ② [量子 PPO](../foundations/reinforcement-learning/quantum-ppo-sector-rotation.md)（獎勵錯位導致高分低能的反面教材）→ ③ [FinRL](../foundations/reinforcement-learning/finrl.md)（成本內生化與框架陷阱）→ ④ [MetaTrader](../foundations/reinforcement-learning/metatrader.md)（自陳種子方差大於預測模型）→ ⑤ [ReCAP](../foundations/reinforcement-learning/recap.md)（成本敏感性的誠實披露範本）→ ⑥ [AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md)（30 日樣本外+5 支存續股的偏差警示）。

## 開放問題 / 值得下注的方向

- **可下注（正向）：執行與對沖類 RL 會持續跑贏靜態基線，但僅在成本非零區間。** [TD3 期權對沖](../foundations/reinforcement-learning/td3.md) 已明示規律——c=0.1%/0.05% 時 RL 勝，c=0.01% 極低摩擦時 RL 反而輸給純 Delta 的理論最優（-64.82pp）。可證偽判斷：任何宣稱在零佣金極度充裕流動性下 RL 執行仍顯著優於基線的結果，大概率是過擬合。理由：RL 的價值來自成本感知的動態降頻，摩擦消失則優勢消失。

- **可下注（負向）：純端到端 RL 選股在扣真實滑點後難以維持 Sharpe > 1.0。** 這是 [FinRL](../foundations/reinforcement-learning/finrl.md) 可證偽預測的直接推論，也被 [量子 PPO](../foundations/reinforcement-learning/quantum-ppo-sector-rotation.md) 與 [AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md) 的成本/樣本外缺口佐證。理由：選股 Alpha 靠方向預測，而 RL 在低信噪比下的方向預測並不比監督學習強，反而多背了樣本效率與種子方差的負擔。

- **值得研究：持續學習/制度自適應是 RL 交易少數有結構性理由的方向。** [ReCAP](../foundations/reinforcement-learning/recap.md) 的 Δθ 差值存儲+CUSUM 制度檢測，把非平穩性從敵人變成可建模的信號。可證偽判斷：若在流動性差的中小盤上 CUSUM 閾值不自適應，策略庫會過度切分、SR 跌破 0.80（[ReCAP](../foundations/reinforcement-learning/recap.md) 自述）。理由：非平穩是金融 RL 的根本病，直接對它建模比假裝平穩更誠實。

- **值得研究：LLM 作為先驗注入器（而非決策體）的輕量路線。** [FinRL-DeepSeek](../foundations/reinforcement-learning/finrl-deepseek.md) 的「擾動係數接近 1」與 [iRDPG](../foundations/reinforcement-learning/irdpg.md) 的 Q-Filter 門控，共享一個哲學:先驗貼著策略梯度走、不覆蓋它。可證偽判斷:若把同樣的 LLM 評分+擾動係數搬到非美股日頻數據上績效顯著劣化，則證明文本先驗與資產尾部風險的映射不可遷移。

- **待驗證：MARL 市場模擬能否從「沙盒」升級為「壓力測試標準」。** [JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md) 的可證偽預測是——若無法支撐超過 10 種異構智能體穩定並發，其「異構並行」設計在真實微結構下的擴展性即被證偽。理由:對抗性對手方適應是檢驗執行算法魯棒性的正確方向,但算力紅利不等於博弈真實性。

## 代表頁面索引

**先讀（框架與範式定調）**
- [FinRL](../foundations/reinforcement-learning/finrl.md) — 環境 scaffolding 與金融摩擦內生化，理解「框架不等於策略」。
- [貝萊德 DRL 訂單執行](../foundations/reinforcement-learning/art-359.md) — RL 在執行類子問題的結構性優勢範本。
- [MetaTrader](../foundations/reinforcement-learning/metatrader.md) — 離線 RL 價值高估與雙層元學習的核心裁決。
- [量子 PPO 行業輪動](../foundations/reinforcement-learning/quantum-ppo-sector-rotation.md) — 獎勵錯位「高分低能」的反面教材。
- [art-88 應對過擬合](../foundations/reinforcement-learning/art-88.md) — 把回測過擬合形式化為假設檢驗的驗證方法論。

**再讀（子戰場深化）**
- [ReCAP](../foundations/reinforcement-learning/recap.md) — CUSUM 持續學習+Δθ 差值庫抗 Alpha 衰減。
- [R²-SAC](../foundations/reinforcement-learning/r-sac.md) — 鬆弛-精煉解耦對抗高維信用分配崩潰。
- [HRT](../foundations/reinforcement-learning/hrt.md) — PPO 選股+DDPG 執行的分層交替訓練。
- [TD3 期權對沖](../foundations/reinforcement-learning/td3.md) — 成本感知連續對沖與成本區間依賴規律。
- [DeepScalper](../foundations/reinforcement-learning/deepscalper.md) — BDQ 動作分支+Hindsight 獎勵的日內範式。
- [iRDPG](../foundations/reinforcement-learning/irdpg.md) — Q-Filter 門控行為克隆的先驗注入模板。
- [OPHR](../foundations/reinforcement-learning/ophr.md) — 擇時 Agent+對沖路由 Agent 的 MARL 拆解。
- [MacroHFT](../foundations/reinforcement-learning/macrohft.md) — regime 條件適配器+記憶路由（MoE 於 RL 執行層）。

**參考（模擬器、多智能體與延伸）**
- [JAX-LOB](../foundations/reinforcement-learning/jax-lob.md) / [JaxMARL-HFT](../foundations/reinforcement-learning/jaxmarl-hft.md) / [集成+GPU 並行模擬](../foundations/reinforcement-learning/ensemble-rl-gpu-sim.md) — GPU 加速模擬與算力紅利邊界。
- [IaC-MARL](../foundations/reinforcement-learning/iac-marl.md) — 多訂單執行的意圖感知通信。
- [AlphaQuanter](../foundations/reinforcement-learning/alphaquanter.md) / [FinRL-DeepSeek](../foundations/reinforcement-learning/finrl-deepseek.md) / [LLM x RL 從新聞學習](../foundations/reinforcement-learning/llm-x.md) / [Stock-Evol-Instruct](../foundations/reinforcement-learning/stock-evol-instruct.md) — LLM 與 RL 結合的譜系。
- [ProtoHedge](../foundations/reinforcement-learning/protohedge.md) / [Logic-Q](../foundations/reinforcement-learning/logic-q.md) / [TINs](../foundations/reinforcement-learning/tins.md) — 可解釋/可審計支線。
- [TradeMaster](../foundations/reinforcement-learning/trademaster.md) / [FinRL-Meta](../foundations/reinforcement-learning/finrl-meta.md) — 標準化評測與 DataOps 平台。
- [DQN 期貨](../foundations/reinforcement-learning/dqn.md) / [TF-Agents](../foundations/reinforcement-learning/tf-agents.md) / [Auto.gov](../foundations/reinforcement-learning/auto-gov.md) — value-based 起點。
- [QTMRL](../foundations/reinforcement-learning/qtmrl.md) / [DeepAries](../foundations/reinforcement-learning/deeparies.md) / [SAC-CNN-MHA](../foundations/reinforcement-learning/sac-cnn-mha.md) / [FNN+DRL](../foundations/reinforcement-learning/fnn-drl.md) / [Model A](../foundations/reinforcement-learning/model-a.md) — 組合配置與擇時的 actor-critic 變體。
- [art-187 期權對沖 8 算法對比](../foundations/reinforcement-learning/art-187.md) / [PPO 策略自適應選擇](../foundations/reinforcement-learning/ppo-hedge-fund-strategy-selection.md) / [TRIALS](../foundations/reinforcement-learning/trials.md) / [配對交易 RL](../foundations/reinforcement-learning/art-32.md) / [NEAT-Python](../foundations/reinforcement-learning/neat-python.md) — 對沖、擇時與配對交易的專項應用。
</content>
</invoke>
