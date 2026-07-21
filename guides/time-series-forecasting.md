# 時序基礎模型全景：從 Transformer 變體到金融時序基座

> 這一族追問一個看似技術實則哲學的問題：一個在電力、氣象、交通序列上刷穿 benchmark 的通用時序模型，憑什麼能在低信噪比、非平穩、regime 頻繁切換的金融價格上生出 Alpha？本文串起 87 篇解構，給出一條「什麼移得動、什麼移不動」的判斷線。

## 這一族在解什麼問題

通用時序預測（TSF）過去五年是一條被公開數據集餵養出來的技術線。ETT、Weather、Electricity、Traffic 這些基準有一個共同的隱性前提：**信號能量遠高於噪聲能量，且非平穩性主要以「均值/趨勢的緩慢漂移」形式出現**。在這種前提下，只要把趨勢與季節性剝乾淨，剩下的殘差就相對可預測。整條演進主線 —— 從長程注意力、到 patch 分解、到頻域多尺度、再到零樣本基礎模型 —— 本質都在優化「如何更便宜、更泛化地擬合這類可預測結構」。

金融時序把這個前提整個掀翻。三件事讓通用 TSF 的歸納偏置系統性失配：

**其一，信噪比是負的量級差。** [LSTM/GBM 置換檢驗](../foundations/time-series-forecasting/lstm-gbm.md)給了最冷酷的實證：在約 72604 根 MNQ 5 分鐘 K 線上，LSTM 與 GBM 的 combined OOS accuracy 徘徊在 50.00% 到 50.89%，等同拋硬幣；即使某折（Fold 3）GBM 跑出 54.76%，200 次標籤置換的隨機上界卻高達 61.31%（p=0.135）。這不是模型不夠強，是**單品種分鐘線的樣本量根本填不滿時序模型的參數空間**。通用 TSF 從不需要面對「真實準確率必須顯著超越隨機打亂標籤的極值」這道閘門，因為物理序列的信號足夠強，隨便一個模型都能顯著。

**其二，非平穩性是結構性斷裂，不是緩慢漂移。** [FinCast](../foundations/time-series-forecasting/fincast.md) 把傳統 Transformer/LSTM 在金融數據上的典型病症命名為「預測坍塌」—— 輸出退化為平庸的均值。原因是異方差與肥尾讓 MSE 梯度被少數極端點主導，模型學會了「猜均值最安全」。這是通用 TSF 的損失函數（純 MSE/MAE）在金融上的第一個死穴。

**其三，橫截面相關性不可放棄，但通用 TSF 的架構默認放棄它。** 幾乎所有基礎模型（[Time-MoE](../foundations/time-series-forecasting/time-moe.md)、[FinCast](../foundations/time-series-forecasting/fincast.md)、[TimesFM](../foundations/time-series-forecasting/timesfm.md)）都建立在 channel-independence 假設上：每個變量獨立映射。這在電力負荷預測裡無傷大雅，在組合優化裡是結構性缺陷 —— 你放棄了 contemporaneous correlation，也就放棄了風險平價與統計套利的地基。

所以這一族在金融語境下真正解的，不是「更準的點預測」，而是**如何把通用 TSF 的三個隱性前提逐個替換成金融前提**：把 MSE 換成分位數/尾部感知損失、把靜態歸一化換成 regime 自適應、把 channel-independence 補回橫截面結構、把點預測升級成分佈預測。誰替換得徹底，誰才可能在真實滑點後留下淨值。這是全篇的主線。

## 方法譜系（演進主線）

以下把 87 篇裡的代表節點串成一條技術脈絡，每個節點標注它「相對前一代多解了什麼」。

**節點零：長程注意力變體（Informer/Autoformer/FEDformer 世系）。** 這一代的問題意識是 Transformer 的 $O(T^2)$ 在長序列上爆顯存。[Informer](../foundations/time-series-forecasting/informer.md) 是這條線在金融上的實測代表：把稀疏注意力架構搬到 BTC 5/15/30 分鐘量價。但它給出的最重要結論恰恰**不在架構**：5 分鐘 GMADL 損失策略年化 115%、最大回撤 32.7%，證明「客製化損失函數比堆注意力層更決定高頻策略的存活率」。這是整族的第一個反直覺信號 —— 架構的邊際收益在金融上遠低於損失函數。

**節點一：patch 化與線性打臉（PatchTST/DLinear 世系）。** 這一代發現把序列切成 patch、甚至直接用線性層，就能匹敵複雜 Transformer。本族沒有獨立的 DLinear 頁，但它的幽靈貫穿全篇：[xLSTM-Mixer](../foundations/time-series-forecasting/xlstm-mixer.md) 明確以 RevIN+NLinear 打底、再堆 sLSTM 補殘差，並在對比表裡點名「DLinear 極簡強健，但無法捕獲高頻非線性耦合」。[TabPFN-TS](../foundations/time-series-forecasting/tabpfn-ts.md) 更激進，直接拋棄自回歸與滯後特徵，只用時間戳的正弦/餘弦日曆特徵，就在 MASE 上超越 Chronos-Mini 7.7%、對參數量 65 倍的 Chronos-Large 領先 3.0%。這一代多解的是：**證明複雜序列架構在週期性數據上常是過擬合的溫床，正確的歸納偏置比容量更重要。**

**節點二：頻域與多尺度分解（FEDformer/TimesNet 之後的頻域細化）。** 這一代把「非平穩」重新定義為「頻譜結構的時變」。[FAN](../foundations/time-series-forecasting/fan.md) 是純粹的代表：放棄時域統計歸一化，用 DFT 提取 TopK 主導頻率，再用 MLP 顯式預測這些頻率成分的未來演變，最後 IDFT 拼回 —— 在 8 個基準上 MSE 平均改善 7.76%–37.90%。[ARMA Attention](../foundations/time-series-forecasting/arma-attention.md) 把 ARMA 的移動平均項隱式嵌入線性注意力，讓 AR 項管長期趨勢、MA 項吸短期衝擊，維持 $O(N)$。這一代多解的是：**把非平穩性的預測「外包」給輕量頻域模塊，讓 backbone 只學平穩殘差，避免梯度被趨勢項淹沒。**

**節點三：線性複雜度序列骨幹（Mamba/xLSTM/RWKV 世系）。** 這一代用狀態空間或線性 RNN 替換自注意力，換取 $O(N)$ 與長序列可行性。[SAMBA](../foundations/time-series-forecasting/samba.md) 用雙向 Mamba 加自適應圖卷積，在 NASDAQ/NYSE/DJIA 上宣稱 IC 較次優提升 33%–85%（見下文對這類「相對第二好模型」數字的警告）。這一代多解的是**長回看窗口下不觸發 OOM**，但也引入雙向掃描的前瞻洩漏隱患。

**節點四：基礎模型與零樣本遷移（TimesFM/Timer/Chronos/LLM 改造世系）。** 這是本族的重心。三條子路線：
- **原生時序基座**：[Time-MoE](../foundations/time-series-forecasting/time-moe.md) 用稀疏 MoE 在 3000 億時間點上把容量推到 24 億參數而推理延遲不變；[Timer](../foundations/time-series-forecasting/timer.md) 把異構序列壓成單序列 S3 格式做 decoder-only 自回歸生成，首次在 TS 領域驗證 Scaling Laws。
- **表格/LLM 借殼**：[TabPFN-TS](../foundations/time-series-forecasting/tabpfn-ts.md) 借表格基礎模型 TabPFN；[TEMPO](../foundations/time-series-forecasting/tempo.md) 把序列分解成趨勢/季節/殘差、各配半軟提示詞餵 GPT，MAE 較前 SOTA 提升 6.5% 與 19.1%。
- **金融領域遷移**：[TimesFM](../foundations/time-series-forecasting/timesfm.md) 用對數變換 MSE 損失把通用基座持續預訓練到價格數據，解決崩盤 NaN 與跨資產尺度差異。這一代多解的是**免特徵工程、免逐任務重訓的跨周期泛化**，代價是黑盒與 channel-independence。

**節點五：金融專用時序基座（Kronos/FinCast/Delphyne/FinPFN）。** 這是譜系的終點，也是與通用 TSF 的分水嶺。[Kronos](../foundations/time-series-forecasting/kronos.md) 為 K 線量身打造二元球面量化 tokenizer，把連續 OHLCVA 拆成粗/細子標記做分層自回歸，零樣本 RankIC 較領先 TSFM 提升 93%、波動率 MAE 降 9%。[FinCast](../foundations/time-series-forecasting/fincast.md) 用 PQ-loss（分位數+Huber+趨勢+專家正則）防坍塌、令牌級稀疏 MoE 隔離領域噪聲，在 3632 條未見序列上零樣本 MSE 降 20%、MAE 降 10%。[Delphyne](../foundations/time-series-forecasting/delphyne.md)（CMU & Bloomberg）則做了最誠實的降維：**把基礎模型的價值主張從「零樣本泛化」明確收縮到「微調初始化器」**，用學生 T 分佈混合直擊重尾。[FinPFN](../foundations/time-series-forecasting/finpfn.md) 走另一條路 —— 元學習，把橫截面收益預測重構為序列元任務，以上一期特徵-收益關係為條件，單次前向即適應 regime 切換，CSI 500 樣本外 IR 0.85、切換期 0.97。這一代多解的是**通用前提的全部三個死穴同時被替換**，但也把「數據泄漏、survivorship、合成數據與真實滑點落差」的風險推到最高。

## 三到五條核心張力

**張力一：複雜注意力 vs 線性/表格基線。** 一端是整條 Transformer 變體世系與 [SAMBA](../foundations/time-series-forecasting/samba.md) 這類重架構；另一端是 [TabPFN-TS](../foundations/time-series-forecasting/tabpfn-ts.md)（純日曆特徵）、[xLSTM-Mixer](../foundations/time-series-forecasting/xlstm-mixer.md)（NLinear 打底）點名的 DLinear 幽靈。**裁決**：在低信噪比金融數據上，複雜度只有在被強正則化「馴服」時才是資產。這裡最關鍵的反例是 [RFF+岭回歸](../foundations/time-series-forecasting/rff.md) —— 它用隨機矩陣理論證明，在適當正則化下高維複雜模型的樣本外夏普率嚴格單調遞增（高複雜度設定年化 Sharpe 0.47 vs 線性 0.46、alpha t 值 3.0）。注意這個 Δ 只有 0.01：它推翻的不是「線性夠用」，而是「複雜必過擬合」的懶惰教條；它同時要求 λ 必須動態匹配維度，否則永久上升退化為雙升。所以裁決不是「線性贏」也不是「複雜贏」，而是**複雜度是可調參數而非懲罰項，正則化強度應以組合層 Sharpe/IR 為目標搜索，而非優化預測精度**。

**張力二：通用基座 vs 金融專用。** 一端是 [Time-MoE](../foundations/time-series-forecasting/time-moe.md)、[TimesFM](../foundations/time-series-forecasting/timesfm.md)、[Timer](../foundations/time-series-forecasting/timer.md)（在電力/氣象上預訓練，宣稱可遷移）；另一端是 [Kronos](../foundations/time-series-forecasting/kronos.md)、[FinCast](../foundations/time-series-forecasting/fincast.md)（金融語料專訓）。[Delphyne](../foundations/time-series-forecasting/delphyne.md) 提供了最鋒利的中間結論：跨域預訓練會產生**負遷移（negative transfer）**，零樣本在金融數據上「必然失效」，基礎模型的真實價值退化為「快速遺忘偏差的初始化空間」。**裁決**：通用基座在金融上不能直接零樣本用，這幾乎是共識（[TimesFM](../foundations/time-series-forecasting/timesfm.md) 的整篇貢獻就是「不做損失重構會 NaN」）。金融專用基座（Kronos/FinCast）在自報指標上大勝，但它們的數字全部未計交易成本、未披露真實滑點後的 IR/AR，且 Kronos 明確警告合成數據的 TSTR 回測與真實流動性枯竭有偏差。務實立場：**把任何基座當特徵提取器/regime 探針，不當下單信號生成器。**

**張力三：點預測 vs 分佈/機率預測。** 一端是大量純 MSE 回歸模型（[Quantformer](../foundations/time-series-forecasting/quantformer.md)、[SAMBA](../foundations/time-series-forecasting/samba.md)、[Time-MoE](../foundations/time-series-forecasting/time-moe.md) 的點頭）；另一端是機率輸出派（[FinCast](../foundations/time-series-forecasting/fincast.md) 的 PQ-loss 分位數、[Delphyne](../foundations/time-series-forecasting/delphyne.md) 的學生 T 混合、[NeuralForecast/TFT](../foundations/time-series-forecasting/neuralforecast-tft.md) 的分位數損失+置信閾值）。[Timer](../foundations/time-series-forecasting/timer.md) 自報的一個 limitation 直接點題：它不支持機率預測，只給點估計。**裁決**：在金融上，點預測的精度幻覺是致命的 —— MSE 降 20% 不等於 Sharpe 提升，因為肥尾與異方差讓「均值準」與「賺錢」脫鉤。分佈預測不是錦上添花，是把尾部風險顯式化的必要條件；風險擇時/波動率目標策略尤其需要分位數而非點值。

**張力四：可微訓練損失 vs 不可微交易目標。** 一端是拿 MSE/MAE 當損失、事後再映射成信號的傳統做法；另一端是直接把交易目標塞進優化循環：[NeuralForecast/TFT](../foundations/time-series-forecasting/neuralforecast-tft.md) 把 Sharpe/Sortino 硬編碼為驗證回調，[Momentum Transformer (TFT)](../foundations/time-series-forecasting/momentum-transformer-tft.md) 與 [Yale 非線性 TSMOM](../foundations/time-series-forecasting/yale.md) 直接以負 Sharpe 為損失端到端輸出頭寸，[Informer](../foundations/time-series-forecasting/informer.md) 用 GMADL 損失。**裁決**：這是本族被反覆實證的一條鐵律 —— **損失函數的經濟意義對齊，回報遠高於架構堆疊**。但要警惕它不是萬靈丹：[Momentum Transformer (TFT)](../foundations/time-series-forecasting/momentum-transformer-tft.md) 以負 Sharpe 為損失，四年平均年回報 4.14%、Sharpe 1.12，卻**沒跑贏同期純多頭的 4.04%**，且擴大窗口到 378、注意力頭到 6 反而過擬合。目標對齊解決的是「優化對了東西」，解決不了「數據裡本來就沒有足夠 Alpha」。

**張力五：追求 Alpha 最大化 vs Sharpe 邊界內的最優風險暴露。** [Yale 非線性 TSMOM](../foundations/time-series-forecasting/yale.md) 是這條張力的教科書：ANN 以最大化 Sharpe 為目標，自動學出「中間線性、極端回撤」的 S 型加權，日度 12 個月回看 Sharpe 0.84（線性 0.70）、對比 XSMOM 0.65 vs 0.48。**裁決**：非線性映射在金融上最穩健的用途，往往不是把預測做得更準，而是**把預測強度非線性地翻譯成風險暴露** —— 在極端信號處主動縮倉換凸性。但它自報 2015–2024 所有動量策略表現下降，且在長期低波動 regime（VIX 長期 <15）下 S 型回撤機制會失效、優勢收斂至與線性無統計差異。

## 什麼在持續有效 vs 什麼是刷榜

跨頁面歸納，以下設計在真實金融數據上反覆站得住：

- **損失函數的經濟意義對齊。** 從 [Informer](../foundations/time-series-forecasting/informer.md) 的 GMADL、[FinCast](../foundations/time-series-forecasting/fincast.md) 的 PQ-loss、[Yale](../foundations/time-series-forecasting/yale.md) 的負 Sharpe、到 [NeuralForecast/TFT](../foundations/time-series-forecasting/neuralforecast-tft.md) 的驗證回調，這是全族最一致的正向信號。它有效是因為它直接修正了「訓練優化的東西」與「實盤賺的錢」之間的目標錯配 —— 而這個錯配在物理序列上根本不存在，所以通用 TSF 從不重視它。
- **相對形態而非絕對尺度。** [FinCast](../foundations/time-series-forecasting/fincast.md) 的 Instance Normalization、[TimesFM](../foundations/time-series-forecasting/timesfm.md) 的對數變換損失、[xLSTM-Mixer](../foundations/time-series-forecasting/xlstm-mixer.md) 的 RevIN，本質都是強迫模型學相對變化，這與量化裡的 Z-score/去均值同源，是跨資產泛化的地基。
- **估計策略紀律 > 模型複雜度。** [HAR-VIX](../foundations/time-series-forecasting/har-vix.md) 給出最反潮流的證據：配置得當的線性 HAR（校準滾動窗口步長與重估頻率、注入 VIX 狀態因子），在 MCS 與經濟效用框架下普遍優於或持平 RF/GBT/FFNN。有效的不是線性本身，是**重估頻率與窗口步長決定了樣本外壽命**這個工程紀律。
- **顯式非平穩性剝離。** [FAN](../foundations/time-series-forecasting/fan.md) 的頻域殘差學習有效，因為它讓 backbone 不必強行擬合非平穩跳變。
- **分佈預測捕尾部。** [Delphyne](../foundations/time-series-forecasting/delphyne.md) 的學生 T 混合、[FinCast](../foundations/time-series-forecasting/fincast.md) 的分位數輸出，在重尾金融數據上比高斯 NLL 更貼合真實。

以下大概率是刷榜，換到金融就崩：

- **只在 ETT/Weather/Electricity 上報 MSE 的架構優勢。** [Time-MoE](../foundations/time-series-forecasting/time-moe.md) 零樣本 MSE 降 >23%、[xLSTM-Mixer](../foundations/time-series-forecasting/xlstm-mixer.md) 56 情境勝 41 —— 這些數字全來自平穩物理序列。金融量價的自相關結構、噪聲比、斷裂頻率截然不同，MoE 路由在低信噪比下可能退化為隨機激活。
- **「相對第二好模型提升 X%」的 IC/RIC 巨幅數字。** [SAMBA](../foundations/time-series-forecasting/samba.md) 自報 NASDAQ IC 提升 85.38%，但頁面自己的解讀點破：靜態 80/5/15 劃分未處理非平穩性，如此巨幅提升更可能暗示**基線極差或訓練集分佈覆蓋了測試集**（數據泄漏），改成嚴格 walk-forward 加 5bp 成本後 IC 提升會收斂到 10%–20%。
- **未披露成本的單期收益宣稱。** [Quantformer](../foundations/time-series-forecasting/quantformer.md) 宣稱「優於另外 100 種因子策略」卻不給任何逐字數值、不給調倉頻率與成本 —— 這是無法驗證的定性宣稱，日頻/週頻高周轉未計成本必被滑點吞噬。
- **小樣本上的高準確率。** [LSTM/GBM 置換檢驗](../foundations/time-series-forecasting/lstm-gbm.md) 是這條的終極反例：單次 walk-forward 的 54.76% 在置換檢驗下毫無統計意義。**顯著性必須優先於準確率。**
- **零樣本金融預測的神話。** [Delphyne](../foundations/time-series-forecasting/delphyne.md) 直說零樣本在金融上必然失效；[TimesFM](../foundations/time-series-forecasting/timesfm.md) 的整篇工作就是證明「不做金融領域微調與損失重構，通用基座直接套會 NaN」。

## 失效模式合集

把各頁 §6 歸納成「時序模型上金融的坑」，按殺傷力排序：

**低信噪比下的假精度（頭號殺手）。** [LSTM/GBM](../foundations/time-series-forecasting/lstm-gbm.md) 證明單品種分鐘線樣本量填不滿參數空間，模型退化為預測基礎先驗（51.80% base rate）的盲盒；發生頻率 5% 的形態在訓練集僅出現不到 50 次。防線：擴張窗口分箱切斷泄漏 + 置換檢驗建零假設分佈 + Purged CV。任何不過這道閘的高準確率都應假定為噪聲擬合。

**分佈漂移 / regime 斷裂。** 幾乎每頁 §6 都列它。[FinCast](../foundations/time-series-forecasting/fincast.md)/[Time-MoE](../foundations/time-series-forecasting/time-moe.md) 的零樣本能力建立在「預訓練覆蓋目標分佈」假設上，遇政策突變/流動性枯竭且訓練集無類似樣本即失效；[Kronos](../foundations/time-series-forecasting/kronos.md) 的 BSQ 碼本在極端波動下離散誤差放大；[FinPFN](../foundations/time-series-forecasting/finpfn.md) 高度依賴近期狀態，Lookback lag 延到 t-2/t-3，IR 從 0.85 崩到 0.53/0.45。

**前瞻洩漏（隱蔽且致命）。** [SAMBA](../foundations/time-series-forecasting/samba.md) 的雙向 Mamba 在實盤推理必須嚴格因果截斷，否則雙向掃描直接洩漏未來；[FAN](../foundations/time-series-forecasting/fan.md) 若用全序列或未來視窗計算 DFT 振幅即引入 look-ahead，必須用 Causal DFT；[Timer](../foundations/time-series-forecasting/timer.md) 的 S3 池化與窗口採樣若不按時間軸切分極易混入未來信息；[FinPFN](../foundations/time-series-forecasting/finpfn.md) 的 Barra 風險調整若用全樣本估計會引入輕微前瞻。

**回看窗過長 / 容量過擬合。** [Momentum Transformer (TFT)](../foundations/time-series-forecasting/momentum-transformer-tft.md) 擴窗到 378、注意力頭到 6 反而過擬合，顯示日頻股票數據的信息密度撐不起更大容量；[TimesFM](../foundations/time-series-forecasting/timesfm.md) 自述超過 100 週期或更大 LR 即過擬合。

**channel-independence 丟橫截面。** [FinCast](../foundations/time-series-forecasting/fincast.md)/[Time-MoE](../foundations/time-series-forecasting/time-moe.md)/[TimesFM](../foundations/time-series-forecasting/timesfm.md) 都放棄 contemporaneous correlation，在組合優化/風險平價層是結構性缺陷，需手動補回協方差矩陣。

**容量與延遲（實盤工程坑）。** 自回歸生成模型（[Kronos](../foundations/time-series-forecasting/kronos.md)、[Timer](../foundations/time-series-forecasting/timer.md)、[Time-MoE](../foundations/time-series-forecasting/time-moe.md)）的串行 rollout 延遲高，全部自報「不適合高頻」；[DeepVol](../foundations/time-series-forecasting/deepvol.md) 的真實成本不在模型而在分鐘級數據的 I/O 與存儲。

**成本未計的名義收益。** 這是跨頁面最普遍的免責聲明 —— 幾乎所有 IC/RankIC/MSE 都未計滑點、換手率、賣空摩擦。[FinPFN](../foundations/time-series-forecasting/finpfn.md) 特別點出中國市場 bottom-decile 在 T+1 與融券限制下的摩擦損耗；[HAR-VIX](../foundations/time-series-forecasting/har-vix.md) 的經濟效用一旦實盤滑點超過其成本代理（過去 9 個月買賣價差中位數）即轉負。

## 讀法（reading paths）

**路徑 A：要一個能上線的日頻預測基線。** 目標是穩、可解釋、成本可控，不追 SOTA。
1. [HAR-VIX](../foundations/time-series-forecasting/har-vix.md) —— 先接受「線性 + 估計紀律可能就夠了」，把重估頻率/窗口步長納入校驗流水線。
2. [RFF+岭回歸](../foundations/time-series-forecasting/rff.md) —— 學會把複雜度當可調參數，用組合 Sharpe/IR 搜正則化強度。
3. [xLSTM-Mixer](../foundations/time-series-forecasting/xlstm-mixer.md) —— 若要上非線性，先用 RevIN+NLinear 打底再堆殘差，別一步到 Transformer。
4. [NeuralForecast/TFT](../foundations/time-series-forecasting/neuralforecast-tft.md) —— 把 Sharpe 硬編進驗證回調，避免因子過擬合 MSE。
5. [Yale 非線性 TSMOM](../foundations/time-series-forecasting/yale.md) —— 學會用負 Sharpe 損失把信號強度非線性翻譯成風險暴露。
6. [LSTM/GBM 置換檢驗](../foundations/time-series-forecasting/lstm-gbm.md) —— 上線前用置換檢驗驗證你的準確率是否真的顯著。

**路徑 B：研究基礎模型 / 零樣本遷移。** 目標是搞清楚「移得動什麼」。
1. [Time-MoE](../foundations/time-series-forecasting/time-moe.md) —— 理解稀疏 MoE 如何解耦訓練容量與推理延遲。
2. [Timer](../foundations/time-series-forecasting/timer.md) —— 理解 S3 格式與 decoder-only 自回歸如何驗證 TS 領域 Scaling Laws。
3. [TabPFN-TS](../foundations/time-series-forecasting/tabpfn-ts.md) —— 見識「正確歸納偏置 > 堆容量」的反直覺路徑。
4. [TimesFM](../foundations/time-series-forecasting/timesfm.md) —— 學金融領域遷移的損失重構（對數 MSE + 掩碼增強）。
5. [Delphyne](../foundations/time-series-forecasting/delphyne.md) —— 接受負遷移現實，把基座降維為微調初始化器。
6. [FinCast](../foundations/time-series-forecasting/fincast.md) —— 看金融專用基座如何用 PQ-loss + MoE + 頻率嵌入同時補三個死穴。
7. [Kronos](../foundations/time-series-forecasting/kronos.md) —— 看 K 線離散 tokenization（BSQ）如何把非平穩壓成可標記的語言。
8. [FinPFN](../foundations/time-series-forecasting/finpfn.md) —— 換一條元學習路線，把 regime 自適應做成序列元任務。

**路徑 C：做高頻 / 日內短程。** 目標是在最惡劣的信噪比下不自欺。
1. [LSTM/GBM 置換檢驗](../foundations/time-series-forecasting/lstm-gbm.md) —— 必讀第一課：小樣本分鐘線等同拋硬幣，先把預期打到地板。
2. [Informer](../foundations/time-series-forecasting/informer.md) —— 看 GMADL 損失如何決定高頻策略鋒利度（架構其次）。
3. [DeepVol](../foundations/time-series-forecasting/deepvol.md) —— 用擴張因果卷積直接吃原始高頻收益做次日波動率，注意排除隔夜。
4. [多時間框架 K 線相似度](../foundations/time-series-forecasting/art-370.md) —— 放棄參數訓練，用 Pearson 相似度 + 跨周期共識過濾，低資源可解釋。
5. [SAMBA](../foundations/time-series-forecasting/samba.md) —— 若要長回看窗口，用 Mamba 線性複雜度，但務必因果截斷雙向掃描。

## 開放問題 / 值得下注的方向

**下注一（高置信）：金融基礎模型的真實增量在「低數據微調」而非「零樣本」，且會被 [Delphyne](../foundations/time-series-forecasting/delphyne.md) 路線收斂。** 可證偽判斷：到 2026 年底，若無任何金融基座（[Kronos](../foundations/time-series-forecasting/kronos.md)/[FinCast](../foundations/time-series-forecasting/fincast.md)）公開一份含 5bp 滑點與換手率限制、淨 Sharpe 顯著超過簡單動量因子的實盤回測，則「零樣本金融基座」作為下單引擎的主張證偽，其定位坐實為特徵提取器。理由：三篇金融基座的所有勝出數字都停在 IC/MSE 層，無一穿透到成本後的組合層。

**下注二（中置信）：channel-independence 是金融基座的下一個被攻破的堡壘。** 可證偽判斷：現有基座（[Time-MoE](../foundations/time-series-forecasting/time-moe.md)/[TimesFM](../foundations/time-series-forecasting/timesfm.md)/[FinCast](../foundations/time-series-forecasting/fincast.md)）在需要 contemporaneous correlation 的統計套利/風險平價任務上，會被顯式建模橫截面的架構（如 [FinPFN](../foundations/time-series-forecasting/finpfn.md) 的橫截面組採樣、[SAMBA](../foundations/time-series-forecasting/samba.md) 的自適應圖）超越。理由：放棄橫截面是為了預訓練工程的便利，不是金融上正確的歸納偏置。

**下注三（中置信）：損失函數工程的邊際收益將超過架構工程，並成為金融 TSF 的主戰場。** 可證偽判斷：未來兩年金融 TSF 的 Sharpe 提升，來自損失設計（[FinCast](../foundations/time-series-forecasting/fincast.md) PQ-loss、[Yale](../foundations/time-series-forecasting/yale.md) 負 Sharpe、[Informer](../foundations/time-series-forecasting/informer.md) GMADL）的比例，將顯著高於來自新骨幹（Mamba/xLSTM/新注意力）的比例。理由：[Informer](../foundations/time-series-forecasting/informer.md) 與 [Momentum Transformer (TFT)](../foundations/time-series-forecasting/momentum-transformer-tft.md) 已用實測給出「損失 > 架構」的鐵律。

**下注四（低置信、高賠率）：非線性映射的價值會從「預測」全面轉向「風險暴露翻譯」。** 可證偽判斷：以 [Yale](../foundations/time-series-forecasting/yale.md) 的 S 型加權為代表的「信號強度→倉位」非線性映射，其獨立 Alpha 貢獻將被證明主要來自尾部凸性（風控），而非中區預測力提升；在長期低波動 regime 下這一貢獻消失。理由：[Yale](../foundations/time-series-forecasting/yale.md) 自己預測 VIX 長期 <15 時 S 型優勢收斂至與線性無統計差異。

**下注五（開放問題，無定論）：小樣本高頻是死胡同還是待解問題？** [LSTM/GBM](../foundations/time-series-forecasting/lstm-gbm.md) 論斷單品種 <10 萬樣本時序預測等同拋硬幣，並給了證偽條件：若有人在 <10 萬樣本上通過 p<0.05 置換檢驗且實盤滑點後 Sharpe>1.0，則死胡同論被推翻。可能的出路是 Tick 級把樣本拉到百萬級或跨標的池化 —— 但這兩條路都改變了「小樣本」前提本身。

## 代表頁面索引

**先讀（建立判斷框架，四篇）**
- [LSTM/GBM 置換檢驗](../foundations/time-series-forecasting/lstm-gbm.md) —— 顯著性優先於準確率，全族方法論地基。
- [Delphyne](../foundations/time-series-forecasting/delphyne.md) —— 負遷移與「基座即初始化器」，戳破零樣本神話。
- [RFF+岭回歸](../foundations/time-series-forecasting/rff.md) —— 複雜度是可調參數，以 Sharpe 搜正則化。
- [HAR-VIX](../foundations/time-series-forecasting/har-vix.md) —— 估計策略紀律 > 模型複雜度。

**再讀（各技術節點代表，八篇）**
- [FinCast](../foundations/time-series-forecasting/fincast.md) —— 金融基座的教科書式三死穴修補（PQ-loss/MoE/頻率嵌入）。
- [Kronos](../foundations/time-series-forecasting/kronos.md) —— K 線離散 tokenization 專用基座。
- [FinPFN](../foundations/time-series-forecasting/finpfn.md) —— 元學習做 regime 自適應。
- [TimesFM](../foundations/time-series-forecasting/timesfm.md) —— 通用基座的金融領域遷移範式。
- [Time-MoE](../foundations/time-series-forecasting/time-moe.md) —— 稀疏 MoE 解耦容量與延遲。
- [FAN](../foundations/time-series-forecasting/fan.md) —— 頻域非平穩性剝離。
- [Yale 非線性 TSMOM](../foundations/time-series-forecasting/yale.md) —— 負 Sharpe 損失 + S 型風險翻譯。
- [Informer](../foundations/time-series-forecasting/informer.md) —— 損失函數 > 架構的高頻實證。

**參考（延伸與對照）**
- [Timer](../foundations/time-series-forecasting/timer.md) —— S3 格式 + decoder-only 生成基座。
- [TabPFN-TS](../foundations/time-series-forecasting/tabpfn-ts.md) —— 純日曆特徵的表格遷移路線。
- [TEMPO](../foundations/time-series-forecasting/tempo.md) —— LLM 借殼 + 分量提示詞。
- [xLSTM-Mixer](../foundations/time-series-forecasting/xlstm-mixer.md) —— 線性打底 + 遞歸殘差。
- [SAMBA](../foundations/time-series-forecasting/samba.md) —— 雙向 Mamba + 自適應圖（注意泄漏警告）。
- [ARMA Attention](../foundations/time-series-forecasting/arma-attention.md) —— 線性注意力嵌入 MA 項。
- [ARMD](../foundations/time-series-forecasting/armd.md) —— 確定性滑動擴散生成。
- [DeepVol](../foundations/time-series-forecasting/deepvol.md) —— 擴張因果卷積直吃高頻做波動率。
- [Momentum Transformer (TFT)](../foundations/time-series-forecasting/momentum-transformer-tft.md) —— 負 Sharpe 損失遷移到股票（含未跑贏純多頭的誠實對照）。
- [Quantformer](../foundations/time-series-forecasting/quantformer.md) —— Transformer 編碼器改造做 A 股單期收益（定性宣稱的警示樣本）。
- [StockMixer](../foundations/time-series-forecasting/stockmixer.md) —— 純 MLP 個股↔市場超圖式混合。
- [NeuralForecast/TFT](../foundations/time-series-forecasting/neuralforecast-tft.md) —— 交易指標嵌入驗證循環。
- [CSM](../foundations/time-series-forecasting/csm.md) —— 條件概率分解替代 Copula 做擇時。
- [多時間框架 K 線相似度](../foundations/time-series-forecasting/art-370.md) —— 非參數 Pearson 相似度 + 跨周期共識。
