# 时序预测基础

> **收錄**：面向价格/收益率/波动率的序列建模，涵盖Transformer变体、RNN/LSTM、MLP/Mixer、状态空间模型及频率自适应架构
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：100 篇（91 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [FinPFN](finpfn) | ⚡ | Journal of Financial Markets | 提出FinPFN框架，利用真实金融数据训练Transformer元学习模型，动态适应市场风格切换 |
| [RFF+岭回归](rff) | ⚡ | RFF+岭回归 | 颠覆奥卡姆剃刀，证明正则化下高维复杂模型在收益预测与择时夏普率上严格优于简单模型。 |
| [FinCast](fincast) | ⚡ | CIKM 2025 | 提出首个十亿参数金融时序基础模型FinCast，结合稀疏MoE、频率嵌入与PQ-loss，实现跨 |
| [Kronos](kronos) | ⚡ | Kronos | 提出专为金融K线设计的基础模型Kronos，通过分层离散化标记器与自回归Transformer实 |
| [Enhancer](enhancer) | ⚡ | KDD 2025 | 提出Enhancer元学习框架，通过时序点过程注意力与关系因果干预机制，联合缓解时序与关系分布漂 |
| [OpDS](opds) | ⚡ | OpDS | 提出算子深度平滑(OpDS)，用图神经网络算子将离散期权报价直接映射为光滑无套利隐含波动率曲面， |
| [DHMOE](dhmoe) | ⚡ | AAAI25 | 提出DHMOE框架，用扩散模型生成参数指导层级化MoE，融合多模态数据实现股票端到端排序预测。 |
| [TimeDART](timedart) | ⚡ | ICML25 | TimeDART融合自回归与扩散去噪，提出自监督时间序列预训练框架，有效捕捉长程动态与局部模式， |
| [TabPFN-TS](tabpfn-ts) | ⚡ | TabPFN-TS | 将表格基础模型TabPFN通过日历特征工程适配至时间序列预测，以11M参数实现零样本SOTA性能 |
| [ARMD](armd) | ⚡ | AAAI 2025 | 提出自回归滑动扩散模型ARMD，将时序演化视为确定性滑动过程，结合线性逆网络实现高精度时间序列预 |
| [TPGN](tpgn) | ⚡ | NeurIPS 2024 | 提出并行门控网络PGN及时间框架TPGN，以O(1)信息路径和并行计算替代RNN，实现长程时序预 |
| [Timer](timer) | ⚡ | Timer | 提出Timer生成式预训练Transformer，通过统一数据集与单序列格式实现时间序列少样本预 |
| [Time-MoE](time-moe) | ⚡ | Time-MoE | 提出基于稀疏MoE的解码器Transformer基础模型，在3000亿时间点数据集上预训练，实现 |
| [TEMPO](tempo) | ⚡ | ICLR 2024 | 提出TEMPO框架，利用趋势-季节性分解与半软提示词，基于GPT实现零样本时间序列预测，在多模态 |
| [aLLM4TS](allm4ts) | ⚡ | ICML24 | 提出aLLM4TS框架，通过自监督多补丁预测与两阶段训练，将LLM适配至时间序列表示学习，显著提 |
| [CSM](csm) | 🔧 | JBF | 提出CSM框架，将收益符号条件化于波动率幅度，避开Copula建模，实现低成本高稳定的指数择时预 |
| [LSTM/GBM对比与置换检验](lstm-gbm) | 🔧 | LSTM/GBM对比与置换检验 | 对比LSTM与GBM在MNQ 5分钟数据上的预测能力，结合置换检验证明小样本时序模型实为噪音拟合 |
| [MiM-StocR](mim-stocr) | 🔧 | MiM-StocR | 提出MiM-StocR框架，通过动量线辅助任务、Adaptive-k排序损失与CQB优化策略，提 |
| [诺贝尔物理学，居然能用来预测市场？—— 一个多时间框架量化交](art-370) | 🔧 | — | 基于多时间框架K线模式相似度匹配，构建低资源消耗、高可解释的量化交易系统，通过跨周期共识与置信度 |
| [SSPT](sspt) | 🔧 | SSPT | 提出SSPT框架，通过股票代码/行业分类与移动平均预测三项定制预训练任务，提升Transform |
| [TwinFormer](twinformer) | 🔧 | TwinFormer | 提出双层Transformer+GRU架构，通过局部/全局稀疏注意力与均值池化实现线性复杂度的长 |
| [TISFM](tisfm) | 🔧 | TISFM | 提出TISFM模型，通过市场状态感知与双向交叉注意力动态融合行业指数，提升多市场指数预测精度。 |
| [Yale | 非线性时间序列动量](yale) | 🔧 | — | 本文揭示TSMOM信号与未来收益的非线性关系，用ANN学习S型加权函数，显著提升Sharpe与尾 |
| [ISEPT](isept) | 🔧 | ISEPT | 提出ISEPT框架，将K线图经CAE编码后由MLP预测配对夏普率，并通过实盘绩效反馈闭环实现选择 |
| [MWUM在线集成框架](mwum) | 🔧 | MWUM在线集成框架 | 提出无梯度在线集成算法，动态加权异构模型预测行业收益，实现稳健的行业轮动策略。 |
| [DMMV](dmmv) | 🔧 | NeurIPS25 [arXiv](https://arxiv.org/abs/2505.24003) | 提出DMMV框架，将时序分解为趋势与周期，分别用数值模型与大视觉模型建模并自适应融合，提升长时序 |
| [收益加权损失函数](art-319) | 🔧 | 收益加权损失函数 | 提出收益加权损失函数，结合CNN/Attention与行业嵌入，实现日频股票信号预测与高夏普交易 |
| [PortfolioMASTER](portfoliomaster) | 🔧 | PortfolioMASTER | 系统对比点态、对偶与列表损失函数对Transformer量化选股模型排序能力与组合业绩的影响。 |
| [FINN扩展框架](finn) | 🔧 | FINN扩展框架 | 提出基于ONNX Scan与FINN编译器的FPGA部署流程，实现混合精度量化LSTM的高效硬件 |
| [Bloomberg | 如何增强OHLC数据预测效果](bloomberg-ohlc) | 🔧 | — | 验证在OHLC数据中加入价格点精确时间戳，能显著提升MLP/RNN/Transformer预测下 |
| [自适应TFT建模](tft) | 🔧 | 自适应TFT建模 | 提出基于相对高点自适应分割与末端模式分类的TFT框架，为不同市场阶段训练专家模型，提升加密货币短 |
| [SSPT](sspt) | 🔧 | KDD 2025 | 提出三种股票定制化预训练任务，构建双层Transformer模型SSPT，在日频选股任务上显著提 |
| [MERA](mera) | 🔧 | MERA | 提出MERA模块，结合自监督检索增强与稀疏混合专家网络，实现股票多模式精准路由与日收益率预测。 |
| [TimePro](timepro) | 🔧 | ICML 25 | 提出基于Mamba的TimePro模型，通过变量与时间感知超状态机制解决多元时序多延迟问题，实现 |
| [MLF](mlf) | 🔧 | KDD 2025 | 提出多周期学习框架MLF，通过自适应分块、冗余过滤与可学习加权集成，提升金融时序预测精度与效率。 |
| [Transformer+MADL](transformer-madl) | 🔧 | Transformer+MADL | 提出MADL方向损失函数，结合Transformer预测股/币日收益率，通过滚动回测验证其生成交 |
| [MIGA](miga) | 🔧 | MIGA | 提出MIGA框架，将MoE与组内注意力结合，通过动态路由与IC损失优化，实现A股日频收益预测的S |
| [Co-CPC](co-cpc) | 🔧 | Co-CPC | 提出Co-CPC框架，融合Copula宏观依赖建模与CPC自监督学习，解耦训练以降低数据与模型不 |
| [TabPFN](tabpfn) | 🔧 | TabPFN | 利用FFT提取时间序列内在周期构建特征，结合TabPFN实现零样本时序预测，优于TabPFN-T |
| [StockAICloud](stockaicloud) | 🔧 | StockAICloud | 提出StockAICloud框架，结合LSTM/GRU/CNN预测日频股价，重点对比AWS EC |
| [FSatten / SOatten](fsatten-soatten) | 🔧 | AAAI 25 | 提出频域谱注意力FSatten与缩放正交注意力SOatten，用傅里叶变换与正交映射替代传统隐空 |
| [NGBoost](ngboost) | 🔧 | ICML19 [arXiv](https://arxiv.org/abs/1910.03225) | 介绍斯坦福NGBoost算法，结合自然梯度与多参数提升，实现灵活可扩展的概率回归与不确定性量化预 |
| [Delphyne](delphyne) | 🔧 | Delphyne | 提出Delphyne预训练模型，通过架构改进与微调缓解跨域负迁移，在通用与金融时序预测任务中表现 |
| [Quantformer](quantformer) | 🔧 | Quantformer | 提出Quantformer，以线性层替代词嵌入并移除解码器，直接处理量价数据预测A股未来收益。 |
| [FreqMoE](freqmoe) | 🔧 | FreqMoE | 提出FreqMoE，通过FFT动态分解频带并分配MoE专家，结合门控权重与残差迭代优化，高效提升 |
| [DeepVol](deepvol) | 🔧 | DeepVol | 提出DeepVol模型，利用扩张因果卷积直接处理高频原始数据，实现次日波动率的高精度预测。 |
| [MiTS-Transformer](mits-transformer) | 🔧 | MiTS-Transformer | 提出将原始Transformer适配连续时间序列的最小修改方案（MiTS），并通过位置编码扩展（ |
| [CryptoPulse](cryptopulse) | 🔧 | CryptoPulse | 提出CryptoPulse框架，融合宏观币圈波动、技术指标与LLM情绪分析，通过双路径预测与情绪 |
| [Informer](informer) | 🔧 | Informer | 本文利用Informer架构处理5/15/30分钟BTC量价数据，通过对比不同损失函数与基准策略 |
| [LSTD](lstd) | 🔧 | AAAI25 | 提出LSTD框架，基于因果表示学习解耦长短期潜在状态，通过平滑与中断约束提升在线非平稳时序预测性 |
| [基于新型收益加权损失函数的深度学习模型](art-177) | 🔧 | — | 提出收益加权交叉熵损失函数，结合CNN与行业Embedding构建日频选股模型，显著提升夏普比率 |
| [MSGNet](msgnet) | 🔧 | MSGNet | 提出MSGNet模型，通过FFT分解多尺度周期，结合自适应图卷积与多头注意力，捕捉多变量时序的跨 |
| [ReFocus](refocus) | 🔧 | ReFocus | 提出ReFocus模型，通过自适应中频能量优化与跨通道关键频选择，解决时序预测频谱间隙，显著超越 |
| [TimeKAN](timekan) | 🔧 | TimeKAN | 提出TimeKAN，通过级联频率分解与多阶KAN学习不同频带时序模式，实现高效长序列预测。 |
| [CryptoPulse](cryptopulse) | 🔧 | CryptoPulse | 提出CryptoPulse框架，融合LLM情绪分析、宏观代理指标与技术指标，通过双重预测机制实现 |
| [FININ](finin) | 🔧 | FININ | 提出FININ模型，通过多模态编码与双注意力机制量化新闻间互动及对市场影响，实现日频市场预测。 |
| [SAMBA](samba) | 🔧 | SAMBA | 结合双向Mamba与自适应图卷积，高效捕捉长序列依赖与特征交互，实现日频收益率预测。 |
| [从“人机对抗”到“人机协作”：股票分析的技艺与人工智能](art-146) | 🔧 | — | 构建AI分析师预测12个月股票收益，对比人类表现，揭示人机协作可显著降低极端预测错误并提升稳健性 |
| sktime | 🔧 | sktime | 本文系统介绍sktime库的核心功能、模块架构及与主流框架的对比，并提供预测、分类与特征提取的实 |
| [LightGBM](lightgbm) | 🔧 | LightGBM | 本文通过设计新型技术指标特征与对比7种目标/特征转换方法，优化LightGBM在AAPL日频股价 |
| [Sampled RNN](sampled-rnn) | 🔧 | Sampled RNN | 提出结合随机特征与Koopman算子理论的无梯度RNN训练方法，通过线性求解替代反向传播，提升时 |
| [Momentum Transformer (TFT)](momentum-transformer-tft) | 🔧 | Momentum Transformer (TFT) | 将TFT动量Transformer迁移至股票日频交易，以负夏普比率为损失端到端输出头寸，验证其在 |
| [TimesFM](timesfm) | 🔧 | TimesFM | 将TimesFM基础时序模型微调至金融价格预测，改进对数损失与掩码策略，显著提升多市场预测准确率 |
| [GTSbot](gtsbot) | 🔧 | GTSbot | 结合LSTM预测与导数趋势分类，动态优化网格进出场点与仓位管理，提升收益并显著降低回撤。 |
| [ES-RNN](es-rnn) | 🔧 | ES-RNN | 将M4冠军ES-RNN模型移植至PyTorch并GPU加速，训练提速322倍，保持高精度且提升代 |
| [MinT预测整合框架](mint) | 🔧 | MinT预测整合框架 | 结合聚类与MinT预测整合框架，对DJIA指数及成分股进行分层时序预测，显著提升样本外准确性与策 |
| [基于机器学习的公司基本面预测](art-105) | 🔧 | — | 系统评估22种模型预测公司基本面的性能，证明全局深度学习模型在不确定性估计与因子回测中显著优于传 |
| [StockMixer](stockmixer) | 🔧 | AAAI24 | 提出基于MLP的StockMixer架构，通过指标、时间（多尺度因果掩码）与股票（超图式市场交互 |
| [DF²M](df-m) | 🔧 | ICML 2024 | 提出DF²M贝叶斯非参数模型，结合IBP、高斯过程与深度核，实现高维函数时间序列的可解释预测。 |
| [Fin-GAN](fin-gan) | 🔧 | Fin-GAN | 提出Fin-GAN框架，结合LSTM与经济学驱动损失函数，实现金融时序的概率预测与方向分类，优化 |
| [NeuralForecast/TFT](neuralforecast-tft) | 🔧 | NeuralForecast/TFT | 基于TFT模型与自定义风险收益指标，实现日频股市预测的端到端训练、滚动预测与样本外评估。 |
| [NeuralForecast/TFT](neuralforecast-tft) | 🔧 | NeuralForecast/TFT | 提出用夏普比率等交易指标替代MSE作为深度学习时序模型的验证指标，使模型优化更贴合日间交易的风险 |
| [MCI-GRU](mci-gru) | 🔧 | MCI-GRU | 提出MCI-GRU模型，融合改进GRU、GAT与多头交叉注意力，从量价数据中联合提取时序、横截面 |
| [深度学习与NLP在加密货币预测中的应用：整合金融、区块链和社](nlp) | 🔧 | — | 融合社媒文本与价格数据，用BART零样本分类与深度学习预测BTC/ETH日频涨跌及局部极值，提升 |
| （当前）为何你不需要深度学习来进行时间序列预测，以及你需要什 | 🔧 | — | 基于M5/Kaggle竞赛经验，论证当前时序预测应优先使用GBM与特征工程，而非深度学习，强调交 |
| [xLSTM-Mixer](xlstm-mixer) | 🔧 | xLSTM-Mixer | 提出xLSTM-Mixer，结合线性预测与sLSTM块，通过时间/变量/多视图混合实现高精度长期 |
| [MultiRC](multirc) | 🔧 | MultiRC | 提出MultiRC模型，通过多尺度重构与对比学习捕捉前兆信号，实现无需异常标注的时间序列异常预测 |
| [FAN](fan) | 🔧 | FAN | 提出频率自适应归一化(FAN)，通过傅里叶变换提取主导频率成分并用MLP预测其变化，有效缓解非平 |
| [ARMA Attention](arma-attention) | 🔧 | ARMA Attention | 提出ARMA注意力机制，将移动平均项融入线性注意力，在保持O(N)复杂度下提升时间序列预测的长短 |
| [GMM-HMM](gmm-hmm) | 🔧 | GMM-HMM | 本文复现GMM-HMM用于日频股价预测，引入方向预测精度(DPA)指标，并验证了预测前用近期数据 |
| [MANA-Net](mana-net) | 🔧 | CIKM 2024 | 提出MANA-Net，通过动态市场-新闻注意力机制加权聚合情感，解决均质化问题以提升日频市场预测 |
| [ADD](add) | 🔧 | ADD | 提出ADD框架，通过解耦市场/超额收益特征、动态自蒸馏去噪及特征混合增强，提升A股趋势预测与回测 |
| [Fieldy](fieldy) | 🔧 | Fieldy | 提出Fieldy模型，通过两阶段层次Transformer并行编码行列字段，捕获表格时序数据的细 |
| [MASSER](masser) | 🔧 | NeurIPS22 Workshop | 结合自监督与元学习，通过两阶段表示学习缓解数据短缺与时变领域偏移，提升股票运动预测泛化性。 |
| [SimStock](simstock) | 🔧 | SimStock | 提出SimStock框架，结合自监督学习与时间域泛化，学习股票时序稳健表示，用于相似股挖掘、配对 |
| [RWKV-TS](rwkv-ts) | 🔧 | RWKV-TS | 提出线性复杂度RNN架构RWKV-TS，通过分块与WKV算子实现高效长序列建模，在预测/分类/检 |
| [端到端期权交易深度学习框架](art-36) | 🔧 | 端到端期权交易深度学习框架 | 牛津大学提出端到端深度学习框架，直接优化夏普比率生成期权交易信号，结合换手率正则化显著提升高成本 |
| [HAR-VIX](har-vix) | 🔧 | HAR-VIX | 对比HAR与ML模型预测实现波动率，发现配置得当的HAR模型在准确性、经济效用及可解释性上普遍优 |
| [CVAE](cvae) | 🔧 | CVAE | 提出基于CVAE的股票日交易量预测模型，引入指数再平衡等提前信息，通过潜在空间建模非线性动态，优 |
| [Stockformer](stockformer) | 🔧 | [arXiv](https://arxiv.org/abs/2401.06139) | 提出Stockformer模型，结合小波变换高低频分离与多任务自注意力网络，实现日频量价因子的收 |
| [StockFormer](stockformer) | 🔧 | StockFormer | 提出StockFormer模型，结合小波变换高低频分离与多任务自注意力网络，实现沪深300量价因 |
| [TIMEVIEW](timeview) | 🔧 | ICLR 2024 | 提出双层透明度框架与TIMEVIEW模型，用B样条与静态特征实现可解释的时间序列预测，兼顾性能与 |
| [DoubleAdapt](doubleadapt) | 🔧 | KDD [arXiv](https://arxiv.org/abs/2306.09862) | 提出DoubleAdapt元学习框架，通过数据与模型双适配器缓解分布偏移，实现股票趋势增量学习预 |
| BRNN | 📖 | BRNN | 详解双向RNN架构与训练算法，通过独立正反向状态链融合完整时序上下文，在序列预测任务中显著优于单 |
| 量化中神经网络选择何种优化器？ | 📖 | — | 对比Adam与NAG优化器在LSTM/GRU上的股价预测表现，证实GRU-Adam组合精度最高且 |
| 多模态时间序列分析：教程与综述 | 📖 | — | 系统综述多模态时间序列分析的挑战、方法（融合/对齐/迁移）、数据集与跨领域应用，重点探讨LLM与 |
| LLM在金融时间序列中的应用及对比 | 📖 | [arXiv](https://arxiv.org/abs/2410.19025) | 对比传统时序模型与LLM基础模型在金融数据预测中的表现，评估少样本与零样本学习能力。 |
| TrendMaster | 📖 | TrendMaster | 介绍TrendMaster框架，基于Transformer架构封装股票数据加载、训练与预测流程， |
| 综述：深度学习用于加密货币预测 | 📖 | — | 系统综述深度学习在加密货币预测中的应用，对比传统与主流DL模型，验证多变量Conv-LSTM在价 |
| 综述：深度学习模型在加密货币市场的应用与验证 | 📖 | — | 系统综述深度学习在加密货币价格预测中的应用，对比ARIMA、LSTM、CNN及Transform |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
