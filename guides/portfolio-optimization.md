# 組合優化全景：從均值方差到端到端可微配置

> 這一族坐落在「監督回歸 / 生成式大模型 / 強化學習 / 元學習搜索」四種範式與「組合執行優化」機制的交叉帶，時間尺度以日频波段為主、向中长周期兩端延伸。它真正在賣的不是「更準的收益預測」，而是**把預測誤差關進籠子的那道工序**：協方差怎麼估、觀點怎麼加、換手怎麼壓、決策目標怎麼直接回傳梯度。

## 這一族在解什麼問題

配置的根本張力，一句話：**你把預測塞進優化器，優化器會把預測的誤差放大成權重的災難。** 這不是修辭，是均值方差（Markowitz）的結構性宿命——目標函數對輸入均值向量的一階敏感度極高，微小的預測偏差經過協方差矩陣求逆後被指數級放大成極端、發散、頻繁翻轉的權重。整整 34 頁裡，幾乎每一頁的 X-Ray 段都在用不同的手術刀切這同一條張力。

第一層是**兩步法 vs 端到端**。傳統流水線先預測（最小化 MSE/MAE）再優化（最大化效用），但兩個目標根本不對齊：預測像素級準確不等於決策成本最低。[RTS-PnO](../foundations/portfolio-optimization/rts-pno.md) 把這件事講到見骨——它明說 PnO 範式下 MSE/MAE 普遍下降，因為模型學會了「忽略對決策無害的預測誤差、專注極值點」，用 SPO+ 代理損失讓不可微的 LP 求解器把真實決策成本的梯度回傳到預測網絡。[SIT](../foundations/portfolio-optimization/signature-informed-transformer-sit.md) 的一句話最狠：它的優勢「不是預測精度紅利，而是架構紅利」，端到端 CVaR 優化消除的是誤差向配置端的非線性放大，而非提升 IC。

第二層是**協方差/風險模型的設定誤差**。優化器放大的不只是均值誤差，還有協方差誤差——而協方差矩陣在高維下必然病態。[2024 JFQA 夏普獎論文](../foundations/portfolio-optimization/2024-jfqa-1-n.md) 給出這條張力的理論地板：高維估計策略存在**系統性向下偏誤**（不是隨機噪聲，是偏誤），N>T 時協方差矩陣直接奇異，1/N 在單因子結構下反而漸近最優。[RIEnet](../foundations/portfolio-optimization/rienet.md) 則從工程端回應：傳統 Ledoit-Wolf/NLS 優化的是 Frobenius 范數（矩陣本身的 MSE），與實盤真正在乎的樣本外組合方差脫節，於是它直接把損失錨定到「調倉日的實現組合方差」，讓梯度流回特徵值清洗函數。

第三層、也是這一族的暗線，是**換手與成本——配置真正的敵人**。所有回測夏普都建立在「無摩擦」假設上，而這一族最誠實的頁面反覆在 §5 解讀裡拆穿這件事：[DeePM](../foundations/portfolio-optimization/deepm.md) 用 Top-K 集成的 Jensen 不等式隱式壓低換手率當作賣點；[RIEnet](../foundations/portfolio-optimization/rienet.md) 的換手率高達 57%（AO 基準僅 18%），只能靠「風險控制溢價覆蓋交易成本」來自辯。更根本的是，成本不只是配置層的稅，它本身就是一個可被端到端優化的目標——[TLN-VWAP](../foundations/portfolio-optimization/tln-vwap.md) 把這件事推到執行層：它繞開「預測成交量曲線→分配訂單」的兩步法，直接把 VWAP 滑點寫成損失函數用自動微分優化，並用一句話點破整族的共性——「Loss Engineering > Architecture Hunting」。這條主線提醒讀者：讀這一族不要只看夏普數字，要問**這個夏普是在多高的換手率、多低的滑點假設下算出來的**。把這三層合起來，這一族的產出從來不是「終端策略」，而是**配置流水線的某一節**：估風險、生觀點、壓換手、對齊決策目標。而貫穿三層的方法論母題只有一個——**把真正在乎的東西寫進損失函數，而不是寄望於中間代理目標的準確度**。

## 方法譜系（演進主線）

把真實頁面串成一條演進脈絡，每一代解的是上一代的一個結構性缺陷。

**第 0 代：經典均值方差與 1/N 錨點。** 起點是 Markowitz 均值方差與它的兩個退化解：全局最小方差（GMV）與等權 1/N。這一代在手冊裡沒有獨立頁面，但它是幾乎所有頁面 §7 對照表的公共基線。真正把它的價值重新定價的是 [2024 JFQA 夏普獎論文](../foundations/portfolio-optimization/2024-jfqa-1-n.md)：它把 1/N 從「懶惰的備份策略」升格為**高維估計風險下的漸近最優錨點**，並給出維度分治的實戰路徑——低維（N<T）把 1/N 與 GMV 線性組合、高維（N>T）引入異象/ML 條件信息構建正交 Alpha 再與 1/N 對沖。這一代確立的規矩：所有複雜模型的超額收益必須先穿透 1/N 底線。

**第 1 代：兩步法的內部升級（風險模型與觀點）。** 既然不動兩步法的框架，就分別把「預測觀點」和「風險矩陣」做扎實。觀點端，Black-Litterman（BL）用貝葉斯精度加權把主觀觀點與市場均衡先驗融合，天生自帶「不確定時退守先驗」的風控——[STABLE](../foundations/portfolio-optimization/stable.md) 把它推到極致：用條件擴散模型生成多路徑收益分佈，把多路徑樣本協方差當作 BL 的主觀精度 Ω，讓模型自己告訴優化器「我有多不確定」。風險模型端，[RIEnet](../foundations/portfolio-optimization/rienet.md) 用 BiLSTM 在特徵值排序空間做譜收縮，實現參數量與資產維度解耦、小組合訓練後直接外推千股。這一代多解的是：**在不碰兩步法框架的前提下，把兩個輸入分別做到樣本外穩健**。

**第 2 代：可微優化器與決策聚焦學習（DFL/端到端）。** 這是這一族的主戰場。核心命題是：既然兩步法目標錯配，就把優化層變成可導、讓決策損失直接回傳。技術上分兩支。一支是**把優化層本身可微化**：[BPQP](../foundations/portfolio-optimization/bpqp.md) 利用 KKT 矩陣結構把隱式微分的二階線性系統重寫為一階友好的分離式 QP，是端到端訓練的「算力槓桿」而非 Alpha 源。另一支是**把不可微金融指標平滑成損失**：[Sharpe/Omega/CVaR 端到端框架](../foundations/portfolio-optimization/sharpe-omega-cvar.md) 用 Softplus/變分公式把指標平滑化直接反傳；[SIT](../foundations/portfolio-optimization/signature-informed-transformer-sit.md) 把路徑特徵注入注意力偏置、端到端優化 CVaR；[GAT](../foundations/portfolio-optimization/gat.md) 用負對數夏普比率當可微損失、TMFG 圖濾波降維；[DeepUnifiedMom](../foundations/portfolio-optimization/deepunifiedmom.md) 用 soft-capped Sharpe 損失統一多週期動量的訊號生成與資本分配；[art-272 利潤損失函數](../foundations/portfolio-optimization/art-272.md) 用平滑 Sigmoid 近似 sign 直接把利潤當損失。這一代多解的是：**讓「配哪些、配多少」和「損失函數」變成同一件事**。

**第 2.5 代：繞開端到端非凸性的監督學習變體。** 端到端直接優化 Sharpe/Sortino 的致命傷是非凸梯度場崎嶇、訓練易崩。[DSL](../foundations/portfolio-optimization/dsl.md) 與 [SDFL](../foundations/portfolio-optimization/sdfl.md) 走了一條聰明的旁路：預計算「理論最優權重」當監督標籤，用凸的 Cross-Entropy 擬合，再用 Deep Ensemble 收縮配置方差。它把非凸組合優化降維成標準分類問題，換來訓練穩定性——代價是標籤在動能反轉時滯後。[DSPO](../foundations/portfolio-optimization/dspo.md) 是同一思路在排序組合上的變體，用單調邏輯回歸損失直接最大化排序組合收益。

**第 3 代：生成式與分佈式配置。** 命題轉為：與其點估計收益，不如生成整個收益分佈餵給風險約束優化器。這裡有三種形態值得分清。其一是**生成場景餵給尾部風險優化器**：[改進 CTGAN + CVaR](../foundations/portfolio-optimization/ctgan-cvar.md) 用 t-SNE+HDBSCAN 識別市場狀態當條件、CTGAN 生成多模態收益場景、再交 CVaR 線性規劃求權重，解的是傳統 CVaR 對歷史路徑依賴過強、無法外推極端行情的坑。其二是**生成分佈當 BL 觀點**：[STABLE](../foundations/portfolio-optimization/stable.md) 把擴散多路徑協方差當 BL 主觀精度。其三、也是最反直覺的一種——**不生成收益、而是克隆行為**：[cGAN 策略克隆](../foundations/portfolio-optimization/cgan.md) 從基金經理的持倉數據裡學出 8 維潛在策略向量，無需預設效用函數即可生成組合權重，它多解的是「多目標效用函數難以靜態參數化」與「經理人行為異質性無法規模化捕捉」——產出不是交易信號，而是一個可插拔的 Agent 生成器，用於壓力測試與反事實歸因。LLM 分支則把「觀點生成」外包給語言模型：[LLM-BL](../foundations/portfolio-optimization/llm-bl.md) 與 [LLM-BLM](../foundations/portfolio-optimization/llm-blm.md) 用 LLM 多次採樣的均值當 BL 觀點向量、方差當置信度矩陣 Ω，把生成模型的內在熵值當動態風險預算。這一代多解的是：**把「不確定性」從事後濾網變成生成分佈的一等公民**。

**第 4 代：RL 配置與求解器升級。** RL 把配置建模為序列決策 MDP，痛點是單一 reward 混編收益與風險導致約束越界。這一代的共同動作是「解耦」——把混在一起的東西拆開分別治。[MASA](../foundations/portfolio-optimization/masa.md) 解耦的是收益與風控：RL 產粗倉位、求解器打硬約束、觀察者調獎勵權重。[MoEDRLPM](../foundations/portfolio-optimization/moedrlpm.md) 解耦的是宏觀擇時與微觀選股：時空自適應嵌入矩陣分離時空先驗、MoE 路由器動態切換交易專家，避免單一 agent 在牛熊轉換時梯度衝突。[HARLF](../foundations/portfolio-optimization/harlf.md) 解耦的是模態：底層按量價 vs FinBERT 情感隔離、中層元智能體動態加權、頂層超級智能體用前瞻性獎勵監督融合，示範了「非結構化文本如何安全注入 RL 狀態空間」（把 LLM 輸出降維成月度標量分數）。[AlphaGAT](../foundations/portfolio-optimization/alphagat.md)（PPO+GAT 動態分配因子權重）、[SciPhyRL](../foundations/portfolio-optimization/sciphyrl.md)（PINN 離線解 HJB 得連續時間策略、用 PDE 殘差硬約束替代 reward 軟梯度以解離線 RL 的數據飢渴）是這一代的另兩種切法。求解器旁支則有 [Grid-FW](../foundations/portfolio-optimization/grid-fw.md)（布爾鬆弛 + Frank-Wolfe 解稀疏最小方差，繞開 MIP 指數超時）與 [YAND](../foundations/portfolio-optimization/yand.md)（微分幾何仿射正規方向解含高階矩的組合優化）。

一條旁支值得單列：**「純規則 / 可解釋」派**，它是對整條深度化譜系的反例。[DAR4020](../foundations/portfolio-optimization/dar4020.md) 只用 60 個月滾動相關性排序做非對稱多空敞口、零參數學習，卻拿到正期望的防禦性配置；[偏度組合管理](../foundations/portfolio-optimization/art-409.md) 證明高階矩管理無需黑盒、Fama-MacBeth 線性投影加排序即可系統性剝離尾部溢價；[異質性信念均衡模型](../foundations/portfolio-optimization/art-399.md) 用會計恒等式與折現方程把主觀預期轉成可求解的隱含收益率，做逆週期的宏觀配置濾波器。這條旁支提醒你：譜系的「進步」方向不等於實盤的「有效」方向。

## 三到五條核心張力

**張力一：兩步法 vs 端到端。** 一端是先預測再優化（PtO/PFL），另一端是決策損失直接回傳（PnO/DFL/E2E）。[RTS-PnO](../foundations/portfolio-optimization/rts-pno.md) 與 [SIT](../foundations/portfolio-optimization/signature-informed-transformer-sit.md) 站在端到端一端，[2024 JFQA 論文](../foundations/portfolio-optimization/2024-jfqa-1-n.md) 隱含站在兩步法一端（它的維度分治本質是把估計問題留在優化層外）。**裁決：端到端的方向正確，但它的紅利被高估了。** 端到端真正消除的是「目標錯配」與「誤差放大」這道結構性損失，而非提升預測力——SIT 自己承認優勢是「架構紅利非預測紅利」。更關鍵的是端到端付出了兩筆隱藏代價：其一，直接把未來收益錨進損失（如 [art-272](../foundations/portfolio-optimization/art-272.md)）必然引入強前瞻偏差，訓練集洩漏未來信息；其二，非凸 Sharpe 梯度場崎嶇到 [DSL](../foundations/portfolio-optimization/dsl.md) 寧可退回監督學習繞道。所以真正穩健的默認不是純端到端，而是 [DSL](../foundations/portfolio-optimization/dsl.md)/[SDFL](../foundations/portfolio-optimization/sdfl.md) 這種「凸代理標籤 + 集成方差收縮」的半決策聚焦路線。

**張力二：點估計 vs 分佈/魯棒。** 一端是把單點預測均值塞進 MVO，另一端是生成整個分佈或做分佈魯棒優化。[STABLE](../foundations/portfolio-optimization/stable.md)、[改進 CTGAN+CVaR](../foundations/portfolio-optimization/ctgan-cvar.md)、[LLM-BL](../foundations/portfolio-optimization/llm-bl.md) 都在分佈這端。**裁決：分佈化是對的，但它的價值來源常被搞錯。** LLM-BL 的真 capability 不是 LLM 預測多準，而是「用方差錨定置信度」平滑掉 MVO 的極端權重——它的 X-Ray 直言此法「並非預測模型，而是觀點過濾與風險預算分配器」。同理 STABLE 的夏普提升「主要來自 BL 對尾部權重的收斂，而非預測準確率堆疊」。這條張力的隱藏陷阱是**生成模型的尾部平滑效應**：CTGAN 頁面自承合成數據在尾部極值處的平滑可能掩蓋真實的尾部相依性，KS 檢驗只保證邊緣分佈相似。所以分佈化真正買到的是「不確定時自動退守」的內建風控，不是更準的收益，別把它當 Alpha 源賣。

**張力三：靜態協方差 vs 動態/收縮估計。** 一端是歷史樣本協方差直接求逆（高維必崩），另一端是收縮/清洗/降維。[RIEnet](../foundations/portfolio-optimization/rienet.md)（BiLSTM 譜收縮）、[GAT](../foundations/portfolio-optimization/gat.md)（TMFG 平面稀疏濾波）、[Grid-FW](../foundations/portfolio-optimization/grid-fw.md)（稀疏基數約束限制非零權重）都在動態/降維這端。**裁決：協方差工程是這一族最被低估、卻回報最確定的一節。** 理由是它繞開了預測——協方差比收益均值好估得多，且 [RIEnet](../foundations/portfolio-optimization/rienet.md) 證明「把損失從矩陣 Frobenius 范數換成組合實現方差」就能拿到穩健的夏普提升。但要小心兩個坑：RIEnet 的庫侖氣體/特徵值連續性假設在流動性枯竭時特徵值排序會劇烈跳變而失效；GAT 的 TMFG 平面性可能過度過濾掉危機狀態下的跨市場聯動邊。動態協方差在正常 regime 是確定紅利，在斷裂 regime 是首先失守的環節。

**張力四：無約束最優 vs 換手/成本約束。** 一端是數學上的最優權重（頻繁翻轉、高換手），另一端是把交易成本、稀疏性、換手率硬編進優化。[DeePM](../foundations/portfolio-optimization/deepm.md)（Top-K 集成隱式壓換手）、[Grid-FW](../foundations/portfolio-optimization/grid-fw.md)（稀疏基數約束降交易筆數）、[DAR4020](../foundations/portfolio-optimization/dar4020.md)（月度再平衡 + 規則型）在約束這端。**裁決：這是實盤與回測的分水嶺，也是這一族最系統性的謊言區。** 絕大多數頁面的 §5 解讀都不得不加一句「未計交易成本/滑點，實盤 Δ 將顯著縮水」——[SIT](../foundations/portfolio-optimization/signature-informed-transformer-sit.md) 明說 5-10 bps 摩擦可能吞噬 Alpha，[RIEnet](../foundations/portfolio-optimization/rienet.md) 的 57% 換手率是懸在頭上的刀。真正把成本當一等公民的少數頁面反而值得信：[RIEnet](../foundations/portfolio-optimization/rienet.md) 用 IBKR 級摩擦模擬器（含佣金/規費/融資利息）驗證高換手仍有淨值優勢，這種帶摩擦的宣稱才有實盤參考價值。裁決明確：**看到高夏普先問換手率與滑點假設，沒有帶摩擦驗證的配置法一律當回測產物**。

**張力五：RL 隱式學風控 vs 求解器硬約束。** 純 RL 派相信序列決策能自己學會風險控制，混合派則堅持把硬約束交給凸優化求解器。[MoEDRLPM](../foundations/portfolio-optimization/moedrlpm.md)/[HARLF](../foundations/portfolio-optimization/harlf.md) 偏純 RL（風險靠 reward 設計與資本管理模塊），[MASA](../foundations/portfolio-optimization/masa.md) 是混合派的旗幟。**裁決：混合派在實盤約束下勝面更大，但這不是 RL 的失敗、而是 RL 的正確用法。** MASA 頁面的面試 Tip 把道理講透：單一 reward 權重敏感、連續動作在劇烈波動下必然約束越界，把硬約束（槓桿、多空、行業中性）交給求解器投影、讓 RL 專注非線性收益探索，是分工而非替代。純 RL 的軟懲罰（Lagrangian penalty）在金融非平穩環境下調參極難，且 TD3/PPO 的價值過估計會被放大。所以這條張力的正解不是「RL vs 求解器」的二選一，而是 [Grid-FW](../foundations/portfolio-optimization/grid-fw.md) 頁面提示的另一種組合——把求解器當成 RL 的「動作空間投影層」，讓 RL 的連續輸出經稀疏/合規投影後才落地。RL 負責找 Alpha，求解器負責讓 Alpha 可執行。

## 什麼在持續有效 vs 什麼被擁擠掉

有結構性理由持續有效的，是**繞開收益預測、只做風險/約束工程的那些工序**。協方差清洗（[RIEnet](../foundations/portfolio-optimization/rienet.md)）、稀疏求解（[Grid-FW](../foundations/portfolio-optimization/grid-fw.md)）、可微優化層（[BPQP](../foundations/portfolio-optimization/bpqp.md)）之所以耐用，是因為它們不預測 Alpha、不與其他資金搶同一份收益，只是把「同樣的預測」轉換成「更穩健、更省成本的權重」。這類工序的競爭對手是數值穩定性和算力，不是市場擁擠，因此不會被套利掉。同理，[2024 JFQA](../foundations/portfolio-optimization/2024-jfqa-1-n.md) 揭示的 **1/N 分散化底線**是結構性的——它源於高維估計風險這條數學鐵律，只要維度災難存在，1/N 就永遠是那個難以擊敗的錨。

被擁擠掉的，是**在真實成本與容量下把數學最優當賣點的那些**。[RIEnet](../foundations/portfolio-optimization/rienet.md) 的 57% 換手在小資金下能靠風險溢價撐住，但容量一大，衝擊成本會把那 +0.21 夏普吃光。[異象/ML 高維路徑](../foundations/portfolio-optimization/2024-jfqa-1-n.md) 更直接——JFQA 頁面自己給出可證偽門檻：若公開異象因子庫的滾動樣本外夏普中位數跌破 0.5，高維路徑的超額收益將被交易成本完全吞噬。生成式配置（[CTGAN](../foundations/portfolio-optimization/ctgan-cvar.md)、[LLM-BL](../foundations/portfolio-optimization/llm-bl.md)）的擁擠風險則來自另一面：它們的回測窗口都極短（LLM-BL 僅 8 個月、STABLE 僅半年），且處於低波動/趨勢市，簡單分散化本就極優，優勢可能是窗口紅利而非結構紅利。裁決：**「不預測、只轉換」的工序有護城河；「預測 + 高換手 + 短窗口」的宣稱要打大折扣**。

## 失效模式合集

跨頁面 §6 歸納出的「配置的坑」高度同構，幾乎每一頁都踩在同一組地雷上。

**協方差估計誤差被優化器放大。** 這是這一族的頭號殺手。所有依賴協方差求逆的方法在 N>T 或病態矩陣下都會崩：[BPQP](../foundations/portfolio-optimization/bpqp.md) 明說一階 ADMM 求解器在金融協方差病態時可能陷入振盪、「數十倍加速」退化成穩定性妥協；[RIEnet](../foundations/portfolio-optimization/rienet.md) 的特徵值排序在流動性枯竭時跳變導致清洗失效。防禦手段是收縮估計器（Ledoit-Wolf）與稀疏/降維，但這些手段本身在斷裂 regime 也會失守。

**換手成本未計 / 被低估。** 幾乎是普遍性坑。[DeepUnifiedMom](../foundations/portfolio-optimization/deepunifiedmom.md) 的 -1.02% MDD 在 34 年回測中「極度異常」，其解讀直接列出三個嫌疑：未計交易成本、疑用未來函數、期貨連續合約拼接平滑了跳空。[art-272](../foundations/portfolio-optimization/art-272.md) 的 53.91% 熊市回報建立在無成本 + 損失函數直接錨定 t+1 收益（強前瞻）之上。凡是不帶摩擦驗證的高夏普，默認打折。

**Regime 依賴 / 樣本外斷裂。** 這一族的模型都在賭「訓練期的統計結構在樣本外穩定」，而配置恰恰在 regime 切換時最需要它、也最容易失效。[STABLE](../foundations/portfolio-optimization/stable.md) 的卡爾曼濾波在宏觀-個股相關性結構性斷裂時滯後；[DAR4020](../foundations/portfolio-optimization/dar4020.md) 賭相關性排序穩定，但極端危機中相關性趨於 1、對沖失效、淨多頭敞口反成回撤來源；[DeePM](../foundations/portfolio-optimization/deepm.md) 的固定宏觀圖譜先驗無法適應供應鏈重組或貨幣體系脫鉤。

**前瞻偏差 / 數據洩漏。** 配置的洩漏往往藏在細節裡。[RTS-PnO](../foundations/portfolio-optimization/rts-pno.md) 的保形預測若 Calibration Set 與訓練集時間重疊，分位數會被低估、風險約束形同虛設；[DeePM](../foundations/portfolio-optimization/deepm.md) 專門用 Directed Delay（Key/Value 嚴設 t-1）來防異步閉盤的 Look-ahead Bias；生成式方法（[LLM-BL](../foundations/portfolio-optimization/llm-bl.md)）還有隱性洩漏——LLM 訓練數據可能已內化歷史宏觀模式。

**容量與生存偏差。** 高頻/高換手方法的容量假設常是空頭支票：[art-78 排名空間統計套利](../foundations/portfolio-optimization/art-78.md) 的高夏普高度依賴 ≤2 bps 極低滑點，流動性枯竭時排名跳躍使再平衡成本指數級上升。[RIEnet](../foundations/portfolio-optimization/rienet.md) 的流動性過濾規則本身可能引入 Survivorship Bias。讀這一族，看到「S&P500 隨機抽樣」要先問退市股怎麼處理。

## 讀法（reading paths）

**① 機構多資產配置（allocator）。** 目標是拿到能上底倉的穩健配置法，先建立錨點與風控直覺，再挑帶摩擦驗證的實戰件。
1. [2024 JFQA 夏普獎論文](../foundations/portfolio-optimization/2024-jfqa-1-n.md) — 先立 1/N 底線與維度分治的世界觀。
2. [STABLE](../foundations/portfolio-optimization/stable.md) — BL 貝葉斯融合作為機構級「AI 管概率、經典管約束」的標準接口。
3. [RIEnet](../foundations/portfolio-optimization/rienet.md) — 協方差清洗這一節的最佳嵌入件，帶 IBKR 摩擦驗證。
4. [DAR4020](../foundations/portfolio-optimization/dar4020.md) — 純規則、可解釋的負 Beta 防禦錨，適合底倉。
5. [DeePM](../foundations/portfolio-optimization/deepm.md) — 宏觀多資產端到端配置的部署藍圖與換手控制技巧。
6. [異質性信念均衡模型](../foundations/portfolio-optimization/art-399.md) — 逆週期宏觀濾波器，做 SAA 的戰略閾值。

**② 端到端可微優化研究（DFL/E2E）。** 目標是搞懂「決策損失怎麼回傳」的技術棧，從優化層可微化到指標平滑到繞道方案。
1. [BPQP](../foundations/portfolio-optimization/bpqp.md) — 可微優化層的算力槓桿，KKT 重寫為分離 QP。
2. [RTS-PnO](../foundations/portfolio-optimization/rts-pno.md) — SPO+ 代理損失 + 保形動態約束，PnO 的完整範式。
3. [Sharpe/Omega/CVaR 端到端框架](../foundations/portfolio-optimization/sharpe-omega-cvar.md) — 不可微金融指標怎麼平滑成損失。
4. [SIT](../foundations/portfolio-optimization/signature-informed-transformer-sit.md) — 路徑特徵注意力偏置 + 端到端 CVaR 的架構紅利範例。
5. [DSL](../foundations/portfolio-optimization/dsl.md) — 繞開非凸的凸代理標籤 + Deep Ensemble 方差收縮。
6. [art-272 利潤損失函數](../foundations/portfolio-optimization/art-272.md) — 平滑 Sigmoid 把利潤直接當損失，看它的前瞻偏差代價。
7. [DeepUnifiedMom](../foundations/portfolio-optimization/deepunifiedmom.md) — soft-capped Sharpe 損失統一訊號與資本分配。

**③ 風控 / 魯棒優化取向（risk & robust）。** 目標是把「不確定性」與「尾部風險」當一等公民，掌握分佈生成、CVaR、稀疏約束這條線。
1. [改進 CTGAN + CVaR](../foundations/portfolio-optimization/ctgan-cvar.md) — 狀態條件生成場景 + CVaR 線性規劃，看尾部平滑陷阱。
2. [STABLE](../foundations/portfolio-optimization/stable.md) — 多路徑協方差當 BL 主觀精度的內建風控。
3. [LLM-BL](../foundations/portfolio-optimization/llm-bl.md) — 用 LLM 輸出方差動態標定 Ω 當風險預算。
4. [Grid-FW](../foundations/portfolio-optimization/grid-fw.md) — 稀疏基數約束，用選股數量硬控估計誤差放大。
5. [MASA](../foundations/portfolio-optimization/masa.md) — RL 產收益、求解器打硬約束的鬆耦合風控範本。
6. [RIEnet](../foundations/portfolio-optimization/rienet.md) — 協方差工程作為風險模型底座。
7. [TLN-VWAP](../foundations/portfolio-optimization/tln-vwap.md) — 把 VWAP 滑點寫進損失，執行層的成本內生化。

## 開放問題 / 值得下注的方向

**一、「不預測、只轉換」的工序會成為配置的護城河主戰場，可下注。** 理由：協方差清洗、可微求解器、稀疏約束不與市場搶收益，回報來自數值穩定性而非擁擠帶外的 Alpha。可證偽門檻：若未來兩年帶摩擦驗證的協方差工程件（如 [RIEnet](../foundations/portfolio-optimization/rienet.md) 一類）在千股容量下仍能維持樣本外夏普優勢，方向成立；若容量一擴夏普就塌到 1/N 附近，則證明它只是小資金玩具。

**二、生成式配置的價值會收斂到「不確定性量化」，而非「收益預測」。** 理由：[LLM-BL](../foundations/portfolio-optimization/llm-bl.md)/[STABLE](../foundations/portfolio-optimization/stable.md) 的實證優勢已被拆穿主要來自 BL 的尾部權重收斂與方差錨定，非預測準確率。可證偽判斷：若把這些方法的生成分佈換成任何合理的「寬先驗 + 樣本協方差」，配置績效不顯著下降，則證明生成模型只是個貴的不確定性估計器，該用更便宜的方法替代。

**三、換手率會取代夏普成為這一族真正的評價軸。** 理由：34 頁裡幾乎每一頁的失效模式都指向未計成本，帶摩擦驗證的頁面（[RIEnet](../foundations/portfolio-optimization/rienet.md)）反而最可信。可下注方向：把換手/成本從「事後扣減」變成「損失函數內生項」的工作會勝出——[DeePM](../foundations/portfolio-optimization/deepm.md) 的 Jensen 集成、[Grid-FW](../foundations/portfolio-optimization/grid-fw.md) 的稀疏約束是雛形，但沒有一頁把成本模型做進端到端梯度。誰先做出「換手感知的可微 CVaR」，誰吃這一波。

**四、RL 配置在真實成本 + 非平穩下大概率不如「RL 粗調 + 求解器硬約束」的混合式。** 理由：[MASA](../foundations/portfolio-optimization/masa.md) 已示範單一 reward 混編風險必然約束越界，純 RL 的樣本效率在金融小樣本下是硬傷。可證偽判斷：若未來出現一個純端到端 RL 配置件，在含真實滑點/保證金規則的日頻實盤中 Sharpe 穩定超過 MASA 式混合架構，則此判斷被推翻——目前沒有。

**五、端到端 DFL 的前瞻偏差是被系統性低估的定時炸彈。** 理由：[art-272](../foundations/portfolio-optimization/art-272.md) 直接把 t+1 收益錨進損失，這在訓練期是標準監督設定、在實盤是致命洩漏。值得下注：嚴格滾動、防洩漏的 DFL 復現件，其真實樣本外夏普大概率遠低於論文宣稱——誰能給出乾淨的、帶前瞻審計的 DFL 基準，誰就重新定義了這一族的可信地板。

## 代表頁面索引

**先讀（建立世界觀與核心手術刀，5 頁）**
- [2024 JFQA 夏普獎論文](../foundations/portfolio-optimization/2024-jfqa-1-n.md) — 1/N 底線與高維估計風險的理論地板。
- [RTS-PnO](../foundations/portfolio-optimization/rts-pno.md) — 兩步法目標錯配與端到端 PnO 的最清晰演示。
- [STABLE](../foundations/portfolio-optimization/stable.md) — 生成分佈 + BL 貝葉斯融合的混合架構標桿。
- [RIEnet](../foundations/portfolio-optimization/rienet.md) — 協方差工程 + 帶摩擦驗證的實戰典範。
- [BPQP](../foundations/portfolio-optimization/bpqp.md) — 可微優化層的算力基礎設施。

**再讀（技術譜系展開，8 頁）**
- [DeePM](../foundations/portfolio-optimization/deepm.md) — 宏觀端到端配置 + 因果過濾 + 換手控制。
- [SIT](../foundations/portfolio-optimization/signature-informed-transformer-sit.md) — 路徑特徵 + 端到端 CVaR。
- [DSL](../foundations/portfolio-optimization/dsl.md) — 凸代理標籤繞開非凸的半決策聚焦。
- [改進 CTGAN + CVaR](../foundations/portfolio-optimization/ctgan-cvar.md) — 狀態條件生成 + CVaR 線性規劃。
- [LLM-BL](../foundations/portfolio-optimization/llm-bl.md) — LLM 觀點方差錨定 BL 置信度。
- [MASA](../foundations/portfolio-optimization/masa.md) — RL 鬆耦合 + 求解器硬約束。
- [Grid-FW](../foundations/portfolio-optimization/grid-fw.md) — 稀疏最小方差的求解器升級。
- [DeepUnifiedMom](../foundations/portfolio-optimization/deepunifiedmom.md) — soft-capped Sharpe 統一多週期動量。

**參考（旁支與特定場景，8 頁）**
- [DAR4020](../foundations/portfolio-optimization/dar4020.md) — 純規則非對稱多空的防禦性配置。
- [異質性信念均衡模型](../foundations/portfolio-optimization/art-399.md) — 逆週期宏觀信念分歧濾波器。
- [偏度組合管理](../foundations/portfolio-optimization/art-409.md) — 高階矩管理的可解釋線性路徑。
- [art-272 利潤損失函數](../foundations/portfolio-optimization/art-272.md) — 利潤當損失的端到端 + 前瞻偏差警示。
- [GAT](../foundations/portfolio-optimization/gat.md) — 圖注意力 + 負對數夏普可微損失。
- [SciPhyRL](../foundations/portfolio-optimization/sciphyrl.md) — PINN 離線解 HJB 的連續時間配置。
- [art-78 排名空間統計套利](../foundations/portfolio-optimization/art-78.md) — 空間變換降維 + 高頻再平衡的容量邊界。
- [Sharpe/Omega/CVaR 端到端框架](../foundations/portfolio-optimization/sharpe-omega-cvar.md) — 不可微金融指標平滑化的工具箱。
- [cGAN 策略克隆](../foundations/portfolio-optimization/cgan.md) — 從持倉克隆經理人策略的 Agent 生成器，用於壓測與歸因。
- [TLN-VWAP](../foundations/portfolio-optimization/tln-vwap.md) — 執行層 Loss Engineering，成本直接可微優化。
- [MoEDRLPM](../foundations/portfolio-optimization/moedrlpm.md) — 時空嵌入 + MoE 路由解耦宏觀擇時與微觀選股。
- [HARLF](../foundations/portfolio-optimization/harlf.md) — 三層分層 RL 融合情感與量價的多模態配置。
- [AlphaGAT](../foundations/portfolio-optimization/alphagat.md) — 兩階段因子挖掘 + PPO/GAT 動態因子權重分配。
- [SDFL](../foundations/portfolio-optimization/sdfl.md) — 半決策聚焦學習 + 深度集成的穩健調倉。
- [DSPO](../foundations/portfolio-optimization/dspo.md) — 高頻日頻融合 + 單調邏輯回歸損失的排序組合。
- [YAND](../foundations/portfolio-optimization/yand.md) — 微分幾何求解含偏度峰度的千級資產組合優化。
- [DGT 動態網格](../foundations/portfolio-optimization/dgt.md) — 加密市場以「重置」替「終止」的資本管理。
- [DiT-LSTM-SVAR](../foundations/portfolio-optimization/dit-lstm-svar.md) — SVAR 噪聲過濾剔除隨機游走標的的高頻組合。
- [art-201 宏觀狀態檢測 TAA](../foundations/portfolio-optimization/art-201.md) — K-means 經濟狀態當條件先驗的戰術資產配置。
- [STRAPSim](../foundations/portfolio-optimization/strapsim.md) — 語義匹配度量組合相似性的成分感知工具。
