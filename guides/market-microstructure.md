# 市場微結構全景：從 LOB 建模到訂單流 Alpha

> 這一族研究的是限價簿（LOB）與訂單流在毫秒到日內尺度上如何形成價格，以及在扣掉延遲與成本後還剩多少可交易的 edge。核心問題只有一句：這條 alpha 你來得及吃嗎。

## 這一族在解什麼問題

微結構與其他 alpha 族的根本差異，是它的信號在「被觀測到」與「可被交易」之間有一道物理鴻溝。日頻因子的半衰期以月計，微結構信號的半衰期以毫秒到分鐘計，於是延遲、排隊優先權、成交概率、逆向選擇這些在別處被當成二階摩擦的東西，在這裡是一階的決定變量。整族頁面反覆出現的主線張力是：**回測畫面漂亮的 alpha，實盤往往來不及吃**——不是因為信號是假的，而是因為吃它需要的速度、容量或成交假設不成立。

這道鴻溝有四個具體形態。第一是 LOB 的動態本身：盤口不是一張靜態快照，而是加/撤/成交事件的異步流，[HLOB](../foundations/market-microstructure/hlob.md) 與 [TLOB](../foundations/market-microstructure/tlob.md) 都在爭論如何把這個非歐、非平穩的結構餵進神經網絡而不被前瞻污染。第二是訂單流不平衡（OFI）：[COI](../foundations/market-microstructure/coi.md) 與 [order-flow 歸一化](../foundations/market-microstructure/order-flow-explanatory-power.md) 都指出，樸素 OFI 在拆單與 HFT 搶跑的環境下已經信號飽和，必須重構分母或重構共現拓撲才能挖回異質信息。第三是價格衝擊：[基於事件的衝擊剖面](../foundations/market-microstructure/art-311.md) 與 [影響中性測度變換](../foundations/market-microstructure/art-395.md) 分別從實證與理論兩端處理「我下的單怎麼推動價格、又怎麼反過來限制我」。第四是延遲本身作為稅：[高頻軍備競賽](../foundations/market-microstructure/art-118.md) 直接把速度劣勢量化成一個可計量的流動性成本項。

把這四點串起來，這一族的統一敘事是：微結構 alpha 的存續條件不是「信號預測力」，而是「在你的延遲與容量下，信號的正反饋循環是否還沒被別人切斷」。[波動率歸一化 Tick Size](../foundations/market-microstructure/tick-size.md) 把這句話推到極致——它證明 2008 年後短週期趨勢的失效不是擁擠、不是成本，而是 HFT 撤單在細 Tick 合約中物理性地切斷了「信號→衝擊→強化」的循環，信號本身消亡了。

## 方法譜系（演進主線）

**節點一：低頻價差與統計計量的地板。** 譜系起點不是深度學習，而是把微結構量測做對。[EDGE](../foundations/market-microstructure/edge.md) 用日頻 OHLC 加 GMM 把有效價差估計的偏差-方差前沿推到接近高頻基準（與分鐘級相關性從 56.17% 升至 88.79%），它多解的是「回測裡的交易成本代理變數系統性失真」這個坑。[PVAR](../foundations/market-microstructure/pvar.md) 則用分鐘級面板 VAR 做因果證偽，證明 ETF 收益率衝擊最多只解釋標的組合方差的 0.26%——它多解的是「別把一個不存在的傳導當 alpha 來挖」。這一層的價值在於劃定基準線：任何更花俏的模型都要先跨過這些乾淨的零假設。

**節點二：深度 LOB 方向預測（DeepLOB 類）。** 這是頁面數量最密的一支。[art-11](../foundations/market-microstructure/art-11.md) 用隨機森林加平滑標籤做短期方向分類，發現淺層盤口不平衡特徵預測力最強；[HLOB](../foundations/market-microstructure/hlob.md) 引入 TMFG 圖拓撲與同調卷積，多解「CNN 把 LOB 當規則網格、忽略價格層級非線性耦合」；[TLOB](../foundations/market-microstructure/tlob.md) 把 Transformer 拆成時間軸與特徵軸雙重注意力，多解「單維注意力混淆時間演變與價量交互」；[OPTM-LSTM](../foundations/market-microstructure/optm-lstm.md) 與 [Attn-LOB 的特徵提取器](../foundations/market-microstructure/attn-lob.md)、[基於注意力的 LOB 預測](../foundations/market-microstructure/lob.md)、[MDI-GD-RBFNN](../foundations/market-microstructure/mdi-gd-rbfnn.md) 各自在架構細節（在線特徵重要性、結構正則化、自動化特徵選擇）上做增量。這一整支多解的是「特徵工程的維度災難」，但它們共享同一個未打開的 envelope：全部只優化 F1/MCC/勝率等預測指標，沒有一個把真實成交成本閉環進來。

**節點三：訂單流不平衡與微觀因子。** 這一支把 OFI 從聚合指標往「異質性分解」推。[COI](../foundations/market-microstructure/coi.md) 用 1ms 共現窗口把訂單流拆成五類條件失衡；[ClusterLOB](../foundations/market-microstructure/clusterlob.md) 用 K-means++ 把訂單分成方向性/機會主義/做市商三類；[order-flow 歸一化](../foundations/market-microstructure/order-flow-explanatory-power.md) 改用市值而非交易量做分母；[小數點後的 Alpha](../foundations/market-microstructure/alpha.md) 則把整數關口的限價單堆積提煉成信號；[譜交易量模型](../foundations/market-microstructure/art-332.md) 從頻域抽算法定時切片的「市場心跳」；[art-333](../foundations/market-microstructure/art-333.md) 把動量溢出拆成日內與隔夜的非對稱結構。這一支多解「聚合訂單流已經被內卷到信號飽和」，共同手法是往拓撲/分類/歸一化維度找異質溢價。

**節點四：生成式與大模型 LOB。** [金融基礎模型](../foundations/market-microstructure/art-363.md) 驗證 Scaling Laws 在微觀數據上成立，用事件流 Tokenization 替代時間切片；[FlowHFT](../foundations/market-microstructure/flowhft.md) 用流匹配把噪聲映射成專家動作序列。這一支多解「淺層樹模型的表達力上限」與「RL 單步誤差累積」，但兩者都卡在同一個死穴：**沒有完美的 simulator**，端到端 RL 無法閉環，模型停在「預測器」而非「執行器」。

**節點五：做市與最優執行。** [牛津做市困境](../foundations/market-microstructure/art-330.md) 用邏輯回歸過濾「虛假失衡」以規避逆向選擇；[Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md) 在模擬盤口裡訓練 RL 做市商對抗 TWAP；[Attn-LOB](../foundations/market-microstructure/attn-lob.md) 用連續動作空間做 DRL 做市；[影響中性測度變換](../foundations/market-microstructure/art-395.md) 從 Kyle-Back 理論端推最優執行的解析解。這一支的共識——也是整族最反直覺的結論——是做市盈利的核心不在預測方向，而在管理成交概率與逆向選擇的權重。

**橫向的方法論支流。** [MDS](../foundations/market-microstructure/mds-metric-dependence-screening.md) 提出用 Fréchet 變差在非歐度量空間做橫截面初篩，反對「粗暴降頻毀掉微觀 alpha」；[消失的指數效應](../foundations/market-microstructure/vanishing-index-effect.md) 與 [高頻軍備競賽](../foundations/market-microstructure/art-118.md) 則是兩個「把過去的 alpha 除名」的頁面——前者證明指數納入效應乘數 M 從約 6.75 跌到約 0.37、異象已趨零，後者把延遲劣勢量化成 0.42 bps 的稅。

## 三到五條核心張力

**張力一：統計計量模型 vs 深度學習。** 一端是 [EDGE](../foundations/market-microstructure/edge.md)、[PVAR](../foundations/market-microstructure/pvar.md)、[基於事件的衝擊剖面](../foundations/market-microstructure/art-311.md) 這類非參數/解析框架——它們的賣點是可審計、無梯度、幾乎沒有前瞻風險，art-311 甚至用 CERN ROOT 的稀疏直方圖把計算複雜度從「VAR 理論需 78 年」降到 280 個計算日。另一端是 [HLOB](../foundations/market-microstructure/hlob.md)、[TLOB](../foundations/market-microstructure/tlob.md)、[金融基礎模型](../foundations/market-microstructure/art-363.md) 的端到端表征。**裁決**：深度學習在方向預測的指標上確實贏，但它們無一例外承認自己的勝率是「預測指標」而非「淨 PnL」，且 TLOB 自己指出隨市場效率提升準確率下降（符合 EMH）。對working quant，統計計量這一端提供的是可靠的地板與成本錨，深度學習這一端提供的是尚未閉環的預測潛力；把後者當因子提取器、前者當成本模型，比二選一更務實。

**張力二：預測方向 vs 直接優化執行。** 節點二整支在賭「預測中間價方向」，但 [牛津做市困境](../foundations/market-microstructure/art-330.md) 給出一個直接反例：它放棄順勢 Imbalance 信號，證明「預測易得性與獲利易得性呈反比」，把樸素做市的夏普從 -109.0 拉到 +11.97 靠的不是預測更準，而是用邏輯回歸過濾掉高逆向選擇的成交。[Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md) 同向印證：RL 做市商的利潤來自穩定價格與庫存管理（流動性供給），而非前置搶跑。**裁決**：在做市/執行層，「預測方向」是紅海且會被逆向選擇反噬；把目標函數從「猜對方向」換成「管理成交概率與逆向選擇」是這一族最可遷移的認知。

**張力三：真實 LOB vs 模擬環境的成交/衝擊假設。** [FlowHFT](../foundations/market-microstructure/flowhft.md)、[Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md)、[影響中性測度變換](../foundations/market-microstructure/art-395.md) 全部依賴模擬器或理論極限：FlowHFT 用 Jump-Diffusion + Hawkes 生成 324 萬狀態-動作對、零滑點假設；art-395 依賴「噪聲交易者無限大」把隨機博弈退化成確定性變分。對面是 [牛津做市困境](../foundations/market-microstructure/art-330.md) 的 23 萬筆實盤 Maker 訂單、[COI](../foundations/market-microstructure/coi.md) 的 457 只股票逐筆數據。**裁決**：[金融基礎模型](../foundations/market-microstructure/art-363.md) 已經把話說死——「無完美 Simulator」是端到端 RL 的死穴。模擬結果再漂亮，domain gap（尤其是訂單簿深度的非線性響應與撮合延遲）就是那條吃掉 alpha 的隱形成本。實盤驗證優先於模擬 SR。

**張力四：單簿 vs 跨場/跨資產微結構。** 多數頁面停在單一標的的盤口，但 [基於事件的衝擊剖面](../foundations/market-microstructure/art-311.md)（板塊間微秒級引領-跟隨）、[COI](../foundations/market-microstructure/coi.md) 的 nis-c/nis-b 跨股共現、[PVAR](../foundations/market-microstructure/pvar.md) 的 ETF-標的傳導、[art-333](../foundations/market-microstructure/art-333.md) 的跨公司動量溢出把視角拉到跨資產拓撲。**裁決**：跨資產微結構的統計信號常常真實（art-311 的溢出率遠大於 1），但它更受延遲鴻溝支配——art-311 明說 ~100 微秒延遲是執行算法的生死線，跨場信號往往只是「套利機器的機械傳遞」，沒有優先級通道就只能吃殘羹。跨資產是 alpha 的富礦，也是延遲門檻最高的區域。

**張力五：信號 vs 稅。** 這一族有兩個頁面根本不在生產 alpha，而在把摩擦成本量化為基準錨：[高頻軍備競賽](../foundations/market-microstructure/art-118.md)（0.42 bps 延遲套利稅、占有效價差 32.82%）與 [消失的指數效應](../foundations/market-microstructure/vanishing-index-effect.md)（乘數 M 崩塌、異象趨零）。**裁決**：對working quant，這兩頁的實盤價值可能高於任何預測模型——它們告訴你哪些「歷史 alpha」已經被市場結構內化成成本，把它們留在因子庫裡只會製造前瞻偏差。

## 什麼在持續有效 vs 什麼是回測幻覺

**扣掉延遲與成本後仍有 edge 的信號**，跨頁面歸納出三個共同特徵。其一是**不依賴速度優勢的異質性分解**：[COI](../foundations/market-microstructure/coi.md) 的雙重排序在 5bps 往返成本下仍給出 22.27% 年化與 1.11 夏普（作者自己扣了成本，這在本族極罕見），因為它捕獲的是傳統風險因子解釋不了的異質溢價（R² 3.16%-6.24% vs 基準 55%-99%），而非搶跑收益。其二是**有外生因果驗證的結構信號**：[小數點後的 Alpha](../foundations/market-microstructure/alpha.md) 用股票拆分/股利做外生衝擊驗證，並在 22 個海外市場中 18 個復現，這種可證偽的因果比純相關穩健得多。其三是**把摩擦當基準而非 alpha**：[EDGE](../foundations/market-microstructure/edge.md) 與 [高頻軍備競賽](../foundations/market-microstructure/art-118.md) 提供的成本錨，是唯一在任何延遲下都「有效」的東西，因為它們量的就是成本本身。

**回測漂亮但實盤來不及吃的**，也有清晰的識別特徵。第一類是**只報預測指標、不報成交後淨值**：[HLOB](../foundations/market-microstructure/hlob.md) 的 73.3%/60%/33% 勝率、[TLOB](../foundations/market-microstructure/tlob.md) 的長視野 F1 優勢，都是 F1/MCC/pT 這類分類指標，作者自己在 §6 承認未計 TCA 成本、Stable 標籤閾值未對齊真實 Tick Size——這些是「預測力」，不是「獲利力」，兩者被 art-330 證明成反比。第二類是**純模擬 SR**：[FlowHFT](../foundations/market-microstructure/flowhft.md) 的 PnL 26.79/25.82 沒有基線對照、零滑點假設，domain gap 未驗證。第三類是**微秒級跨場信號**：[art-311](../foundations/market-microstructure/art-311.md) 的溢出率大於 1 只證明統計領先，不保證套利空間；沒有 FPGA 直連就吃不到。第四類是**已被市場除名的異象**：[消失的指數效應](../foundations/market-microstructure/vanishing-index-effect.md) 的納入效應已從 1990 年代 7.4% 跌到 0.3%，剩下的在實盤已無套利空間。第五類是**細 Tick 稀疏簿上的短週期趨勢**：[波動率歸一化 Tick Size](../foundations/market-microstructure/tick-size.md) 證明這類信號在 ST 合約上已經物理性消亡，零延遲回測依然失效。

一個貫穿全族的判準：**凡是作者敢在 §5 扣成本、在 §6 承認容量地板、在 §7 給出帶日期的可證偽預測的，可信度顯著高於只在模擬器裡報絕對 PnL 的。** COI、小數點後 Alpha、牛津做市困境屬前者；FlowHFT、HLOB、TLOB 屬後者。

## 失效模式合集

跨頁面把各自 §6 的「隱含假設」歸納，微結構的坑高度集中在六類。

**前瞻偏差（Look-ahead）。** 這一族最隱蔽的前瞻不在標籤，而在標準化與拓撲構建。[HLOB](../foundations/market-microstructure/hlob.md) 的 TMFG 用訓練期平均互信息矩陣，若未嚴格隔離測試期就會滲入前瞻；[ClusterLOB](../foundations/market-microstructure/clusterlob.md) 專門用前向滾動標準化（100 訂單滑窗）來堵這個洞，並警告 30 分鐘桶聚合可能引入跨桶信息平滑；[TLOB](../foundations/market-microstructure/tlob.md) 的雙線性歸一化依賴 batch 統計量，推理時就是 OOD 偏差來源。

**延遲假設不現實。** [art-311](../foundations/market-microstructure/art-311.md) 的 ~100 微秒生死線、[COI](../foundations/market-microstructure/coi.md) 的 1ms 共現窗口對數據饋送延遲極度敏感（需交易所直連/FPGA 解碼）、[金融基礎模型](../foundations/market-microstructure/art-363.md) 明說推理延遲壓不到微秒級則 Smart 優勢被滑點吞噬——延遲不是可以事後校正的常數，它決定信號能否被吃到。

**成交假設過樂觀。** [FlowHFT](../foundations/market-microstructure/flowhft.md)、[art-395](../foundations/market-microstructure/art-395.md) 假設零滑點/固定點差/訂單簿深度無限；[Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md) 只計 1 bps 手續費、假設 fRL 直接獲知 TWAP 方向（實盤要靠信號提取）。這些假設在模擬器裡成立，在有限深度的真實簿口會系統性高估收益。

**容量極小。** [ClusterLOB](../foundations/market-microstructure/clusterlob.md) 的小盤股 φ3 給出 3.374 SR、7.03 Calmar，但這幾乎確定是低容量下的流動性補償，實盤滑點會吞掉大部分；[牛津做市困境](../foundations/market-microstructure/art-330.md) 自陳屬「低容量、高週轉執行層收益」，不宜放大槓桿或跨市場平移。高風險調整回報在微結構層往往是容量極小的信號。

**簿口快照對齊與撮合機制依賴。** [COI](../foundations/market-microstructure/coi.md) 依賴 NASDAQ 連續競價，在 A 股集合競價/漲跌停下拓撲會斷裂；[art-118](../foundations/market-microstructure/art-118.md) 的微秒同步高度依賴單一交易所（LSE）的硬體時鐘與 Price-Time 撮合邏輯；[小數點後的 Alpha](../foundations/market-microstructure/alpha.md) 在日本市場因無小數位報價直接失效。信號綁定特定撮合機制，跨市場遷移就是遷移失效。

**Tick 數據清洗與 regime 依賴。** [基於事件的衝擊剖面](../foundations/market-microstructure/art-311.md) 警告時間戳同步誤差/丟包會讓直方圖對齊產生系統性偏差；[COI](../foundations/market-microstructure/coi.md) 的孤立 COI 在 2020 疫情期間預測力失效；[牛津做市困境](../foundations/market-microstructure/art-330.md) 的特徵重要性一旦市場從散戶小單主導切換到機構大單主導就會塌陷。微結構信號的 regime 邊界比日頻因子窄得多，且極端行情（閃崩、流動性真空）恰恰是這些模型 §6 一致承認打不開的 envelope。

## 讀法（reading paths）

**路徑一：做 LOB 預測 / OFI 因子的因子研究員。** 先建立基準與坑位認知：[EDGE](../foundations/market-microstructure/edge.md)（成本估計的地板）→ [art-11](../foundations/market-microstructure/art-11.md)（淺層盤口不平衡特徵預測力最強的樸素結論）→ [HLOB](../foundations/market-microstructure/hlob.md)（圖拓撲先驗）→ [TLOB](../foundations/market-microstructure/tlob.md)（時空解耦與非平穩歸一化）→ [COI](../foundations/market-microstructure/coi.md)（OFI 的異質性分解，含扣成本結果）→ [order-flow 歸一化](../foundations/market-microstructure/order-flow-explanatory-power.md)（市值分母修復信號腐蝕）→ [ClusterLOB](../foundations/market-microstructure/clusterlob.md)（標籤一致性工程）→ [小數點後的 Alpha](../foundations/market-microstructure/alpha.md)（外生因果驗證的範本）。

**路徑二：做最優執行 / 做市的執行工程師。** [影響中性測度變換](../foundations/market-microstructure/art-395.md)（理論端的執行節奏推演）→ [牛津做市困境](../foundations/market-microstructure/art-330.md)（做市的核心是管理逆向選擇而非預測方向）→ [Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md)（RL 做市商作為流動性供給者）→ [Attn-LOB](../foundations/market-microstructure/attn-lob.md)（連續動作 DRL 做市與混合獎勵）→ [FlowHFT](../foundations/market-microstructure/flowhft.md)（流匹配序列生成規避單步誤差累積）→ [譜交易量模型](../foundations/market-microstructure/art-332.md)（把週期項嵌入 VWAP 執行降低成本）。

**路徑三：關心模擬真實性與延遲的驗證派。** [高頻軍備競賽](../foundations/market-microstructure/art-118.md)（延遲劣勢的稅有多少）→ [基於事件的衝擊剖面](../foundations/market-microstructure/art-311.md)（微秒延遲是執行生死線）→ [波動率歸一化 Tick Size](../foundations/market-microstructure/tick-size.md)（零延遲回測依然失效的物理性解釋）→ [消失的指數效應](../foundations/market-microstructure/vanishing-index-effect.md)（被市場除名的異象如何識別）→ [PVAR](../foundations/market-microstructure/pvar.md)（乾淨的因果證偽基準）→ [金融基礎模型](../foundations/market-microstructure/art-363.md)（無完美 simulator 的死穴）→ [MDS](../foundations/market-microstructure/mds-metric-dependence-screening.md)（粗暴降頻毀掉微觀 alpha 的警告）。

## 開放問題 / 值得下注的方向

**其一：深度 LOB 模型會被逼著把成交成本閉環進損失函數。** 目前 [HLOB](../foundations/market-microstructure/hlob.md)/[TLOB](../foundations/market-microstructure/tlob.md)/[OPTM-LSTM](../foundations/market-microstructure/optm-lstm.md) 全部優化分類指標，而 [art-330](../foundations/market-microstructure/art-330.md) 已證明預測力與獲利力反相關。可證偽判斷：若到 2026 年底仍無一個 DeepLOB 類架構在 §5 直接報扣費淨 SR 且顯著跑贏簡單 MLP 基線（TLOB 自己的 7.1 就是這個賭注），則這整支停在「離線特徵提取器」而非可上線 alpha。理由：分類指標與 PnL 的鴻溝是結構性的，不是調參能補的。

**其二：跨資產微結構是下一個富礦，但門檻是延遲不是模型。** [art-311](../foundations/market-microstructure/art-311.md)、[COI](../foundations/market-microstructure/coi.md) 的跨股共現、[art-333](../foundations/market-microstructure/art-333.md) 的跨公司溢出都指向真實的跨資產結構。可證偽判斷：這些信號的 Sharpe 對延遲的敏感度會遠高於對模型複雜度的敏感度——把 art-311 的溢出率信號放進一個 >1ms 延遲的執行環境，超額收益應趨零。理由：跨場信號本質是套利機器的機械傳遞，沒有優先級通道就是負和。

**其三：做市的認知範式已經翻轉，預測方向的做市會持續虧損。** [art-330](../foundations/market-microstructure/art-330.md)、[Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md) 一致指向「做市盈利=管理成交概率與逆向選擇」。可證偽判斷：任何仍以「順勢 Imbalance 預測方向」為核心的做市策略，在散戶小單主導向機構大單主導切換的市場（art-330 自己的 7.1 預測）中夏普會回落到無經濟顯著性。理由：逆向選擇是做市的一階成本，順勢信號恰好最大化這個成本。

**其四：生成式 LOB 的真正瓶頸是 simulator 保真度，不是模型容量。** [art-363](../foundations/market-microstructure/art-363.md)（Scaling Laws 成立但無完美 simulator）與 [FlowHFT](../foundations/market-microstructure/flowhft.md)（純模擬驗證）合起來說明，Scaling 能提升預測，但端到端執行 RL 卡在 domain gap。可證偽判斷：若無人先解決「訂單簿深度非線性響應 + 撮合延遲」的高保真模擬，生成式 LOB 就只能做離線表征基座。理由：回測無法複刻自己下單造成的市場反應，這是模型無關的物理限制。

**其五：微結構「除名」清單會繼續變長。** [消失的指數效應](../foundations/market-microstructure/vanishing-index-effect.md) 與 [波動率歸一化 Tick Size](../foundations/market-microstructure/tick-size.md) 是同一個機制的兩個實例——市場結構演化把過去的 alpha 內化成成本或直接殺死。可證偽判斷：更多依賴「特定微觀結構不變」的信號（如 [小數點後的 Alpha](../foundations/market-microstructure/alpha.md) 依賴散戶偏好整數報價）會隨算法搶跑而衰減，該頁自己的 7.1 就押注 2026-06-30 前整數關口深度失衡被抹平。理由：任何被公開的微結構 alpha 都在邀請套利者把它填平，半衰期只會越來越短。

## 代表頁面索引

**先讀（建立整族認知，⚡ 為主）**
- [牛津做市困境](../foundations/market-microstructure/art-330.md) — 預測力與獲利力反相關，做市的核心是管理逆向選擇。
- [COI](../foundations/market-microstructure/coi.md) — OFI 異質性分解，本族少數敢扣成本的頁面。
- [波動率歸一化 Tick Size](../foundations/market-microstructure/tick-size.md) — 信號物理性消亡，零延遲回測依然失效的因果解釋。
- [order-flow 歸一化](../foundations/market-microstructure/order-flow-explanatory-power.md) — 市值分母修復訂單流信號腐蝕。
- [小數點後的 Alpha](../foundations/market-microstructure/alpha.md) — 外生因果驗證的範本，跨市場復現。

**再讀（分支代表，深化各節點）**
- [HLOB](../foundations/market-microstructure/hlob.md)、[TLOB](../foundations/market-microstructure/tlob.md) — 深度 LOB 方向預測的兩種架構路線。
- [Hawkes LOB + PPO/SIL](../foundations/market-microstructure/hawkes-lob-ppo-sil.md)、[FlowHFT](../foundations/market-microstructure/flowhft.md) — RL/生成式做市的模擬環境賭注。
- [基於事件的衝擊剖面](../foundations/market-microstructure/art-311.md) — 跨資產微秒級因果拓撲。
- [EDGE](../foundations/market-microstructure/edge.md) — 低頻成本估計的地板。
- [影響中性測度變換](../foundations/market-microstructure/art-395.md) — 最優執行的理論端。
- [金融基礎模型](../foundations/market-microstructure/art-363.md) — Scaling Laws 與無完美 simulator 的死穴。

**參考（增量方法與方法論支流）**
- [ClusterLOB](../foundations/market-microstructure/clusterlob.md)、[art-11](../foundations/market-microstructure/art-11.md)、[OPTM-LSTM](../foundations/market-microstructure/optm-lstm.md)、[MDI-GD-RBFNN](../foundations/market-microstructure/mdi-gd-rbfnn.md)、[基於注意力的 LOB 預測](../foundations/market-microstructure/lob.md)、[Attn-LOB](../foundations/market-microstructure/attn-lob.md) — LOB 建模與訂單流因子的增量。
- [譜交易量模型](../foundations/market-microstructure/art-332.md)、[art-333](../foundations/market-microstructure/art-333.md) — 頻域與跨時段的微觀因子。
- [高頻軍備競賽](../foundations/market-microstructure/art-118.md)、[消失的指數效應](../foundations/market-microstructure/vanishing-index-effect.md)、[PVAR](../foundations/market-microstructure/pvar.md) — 摩擦稅、異象除名與因果證偽。
- [MDS](../foundations/market-microstructure/mds-metric-dependence-screening.md)、[art-171](../foundations/market-microstructure/art-171.md) — 非歐初篩與高頻標籤不平衡的方法論。
