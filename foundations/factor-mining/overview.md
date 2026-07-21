# 因子挖掘与特征工程

> **收錄**：公式型Alpha生成、AutoML/遗传编程挖掘、特征选择、风格因子构造、信号衰减与拥挤度分析
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：68 篇（62 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [PTK](ptk) | ⚡ | PTK | 提出PTK解耦框架，DNN专注提取非线性因子，显式核方法优化组合，大幅提升样本外夏普与宏观对齐度 |
| [耶鲁 X 牛津 | 知情交易与预期收益](yale-oxford-informed-trading) | ⚡ | — | 利用上交所账户数据构建机构历史超额交易利润指标，验证信息不对称对大盘股横截面收益的定价作用。 |
| [AlphaAgentEvo](alphaagentevo) | ⚡ | AlphaAgentEvo | 提出基于GRPO与AST约束的自演化智能体框架，实现因子在多轮交互中自动迭代优化。 |
| [Alpha挖掘vs深度学习：量化中的争议性困境](alpha-vs) | ⚡ | — | 系统对比Alpha挖掘与深度学习优劣，提出基于市场微观结构的范式选择原则与混合架构。 |
| [深度可视化因子择时框架](art-354) | ⚡ | 深度可视化因子择时框架 | 将206个股票因子的历史收益轨迹转为图像，用CNN学习边界与波动率特征生成择时信号，显著提升Al |
| [Alpha-R1](alpha-r1) | ⚡ | Alpha-R1 | 提出Alpha-R1框架，利用GRPO强化学习对齐8B推理模型，基于价格与新闻语义动态筛选Alp |
| [CogAlpha](cogalpha) | ⚡ | CogAlpha | 结合LLM认知推理与进化算法，通过七层Agent架构与代码级变异交叉，自动挖掘高预测性、可解释的 |
| [Proj SUE](proj-sue) | ⚡ | Proj SUE | 利用LLM文本嵌入构建动态同业网络，量化盈余惊喜溢出效应，生成高夏普Alpha因子并解释主流横截 |
| [Attention Factor Model](attention-factor-model) | ⚡ | Attention Factor Model | 提出基于注意力机制的端到端统计套利模型，联合学习潜在因子与交易策略，直接优化净夏普比率，显著优于 |
| [AlphaSAGE](alphasage) | ⚡ | [arXiv](https://arxiv.org/abs/2509.25055) | 提出AlphaSAGE框架，融合RGCN结构感知与GFlowNets多样性生成，通过多维奖励自动 |
| [Alpha-GPT](alpha-gpt) | ⚡ | Alpha-GPT | 本文提出Alpha-GPT系统，结合LLM与遗传规划，通过人机交互工作流自动化生成并优化公式型A |
| [AlphaAgent](alphaagent) | ⚡ | AlphaAgent | 提出AlphaAgent框架，结合LLM与AST正则化机制，通过多智能体协同挖掘抗衰减的持久Al |
| [FRA特征筛选算法](fra) | ⚡ | FRA特征筛选算法 | 系统评估多源数据对加密市场预测的影响，提出FRA特征筛选算法，揭示长短周期核心驱动因子。 |
| [AlphaQCM](alphaqcm) | ⚡ | ICML25 | 提出AlphaQCM分布强化学习框架，利用无偏方差引导探索，高效挖掘协同公式化Alpha因子。 |
| [UMI](umi) | ⚡ | KDD 2025 | 提出UMI框架，通过协整注意力与自监督任务学习个股及市场层非理性因子，提升股票收益预测。 |
| [经验贝叶斯(EB)](empirical-bayes) | ⚡ | 经验贝叶斯(EB) | 美联储首席经济学家用经验贝叶斯法系统挖掘13.6万策略，校正数据挖掘偏差，发现会计因子与小盘股在 |
| [AlphaSharpe](alphasharpe) | ⚡ | AlphaSharpe | 利用LLM结合进化策略迭代生成与优化风险调整指标，显著提升资产排名相关性与组合夏普比率。 |
| [结构化深度排序因子模型](art-121) | ⚡ | JFQA | 提出结构化深度神经网络，将传统特征排序自动化为潜在因子生成框架，以最小化定价误差解释横截面收益。 |
| [Supervised Lexicon Learning](supervised-lexicon-learning) | 🔧 | [arXiv](https://arxiv.org/abs/2607.14174) | 基于监督词表学习提取10-K全文与风险章节情绪，验证聚合层级对收益与波动率预测信号有效性的影响。 |
| [LLM Factor Lab](llm-factor-lab) | 🔧 | LLM Factor Lab | 用LLM在预设会计公式空间内自主演化因子，经严苛学术检验仅极少数存活。 |
| [FactorMoE](factormoe) | 🔧 | Complex & Intelligent Systems | 提出FactorMoE框架，用链式MoE与注意力机制动态组合公式化Alpha因子，提升非平稳市场 |
| [周五潜伏？知情交易周内择时的非线性 Alpha](alpha) | 🔧 | — | 揭示内部人交易Alpha高度依赖市场关注度，周五买入/周一卖出仅在异常放量时产生显著短期超额收益 |
| [因子的非对称性：为什么Alpha在熊市反而更强？](factor-asymmetry-bear-market) | 🔧 | — | 本文检验全球56市场133个因子，发现Alpha在熊市显著更强且由空头端主导，提出基于市场状态动 |
| [行为因子统治A股？如何对收益系统性归因](behavioral-factors-a-share) | 🔧 | — | 结合TreeSHAP与消融实验，系统诊断XGBoost多因子模型中各因子的独立贡献与替代效应，揭 |
| [Zookeeper](zookeeper) | 🔧 | Zookeeper | 针对CTA截面窄痛点，用线性集成统一三大商品风险溢价理论，构建抗过拟合的Zookeeper因子。 |
| [Comprehensive Crowded Factor](comprehensive-crowded-factor) | 🔧 | UBS Research | 基于UBS专有拥挤因子，研究拥挤度拐点对股价的非对称影响，构建月度调仓多空策略，显著增强价值/动 |
| [QuantPedia CEO教你如何设计多时间框架趋势策略](quantpedia-ceo) | 🔧 | — | 本文演示比特币多时间框架趋势策略构建：日线定方向、小时线金叉入场、反向K线追踪离场，以逻辑结构替 |
| [协同智能](art-351) | 🔧 | 协同智能 | 提出协同智能框架，强制对齐1-3个月分析师短期观点，将其作为特质性Alpha与量化因子正交叠加， |
| [HF9模型/Post-Adaptive-LASSO](hf9-post-adaptive-lasso) | 🔧 | HF9模型/Post-Adaptive-LASSO | 基于Adaptive LASSO从44个因子中筛选出HF9模型，有效剥离异象风险溢价，精准识别对 |
| [双曲线衰减模型](art-344) | 🔧 | 双曲线衰减模型 | 基于博弈论推导因子Alpha双曲线衰减机制，实证表明拥挤度不预测收益而有效预警反转类因子尾部风险 |
| [寻找“丑陋的优势”：在加密市场的混乱中收割超额Alpha](crypto-ugly-edge-alpha) | 🔧 | — | 本文探讨在低效加密市场中，通过波动率标准化趋势跟踪、资金费率套利及识别操纵信号获取超额Alpha |
| [ORCA](orca) | 🔧 | ORCA | 提出ORCA框架，将对比学习与OU过程物理信息正则化结合，端到端聚类挖掘具有稳健均值回归属性的资 |
| [Unicorn Edge](unicorn-edge) | 🔧 | Unicorn Edge | 提出基于63日正收益占比>60%的漂移状态过滤器，条件化激活价值与反转因子，实现极高样本外夏普。 |
| [对比资产嵌入框架](art-320) | 🔧 | 对比资产嵌入框架 | 提出基于假设检验采样的对比学习框架，从日频收益序列学习资产嵌入，用于行业分类与对冲组合优化。 |
| [Mixture Decoupled](mixture-decoupled) | 🔧 | Mixture Decoupled | 提出因子与新闻的混合预测模型及解耦训练法，解决融合纠缠与训练不稳定问题，提升多空选股稳健性。 |
| [美联储 ｜ 有形之手：基金经理预期与A股非对称价格效应](fund-manager-expectation-asymmetry) | 🔧 | — | 利用文本分析量化基金经理宏观增长预期，揭示其如何通过调仓交易引发A股非对称价格冲击，并提升市场定 |
| [MIQUBO](miqubo) | 🔧 | MIQUBO | 将特征选择建模为MIQUBO问题，利用量子退火最大化互信息与条件互信息，有效降低因子冗余并提升下 |
| [BRT/NN+递归排序](brt-nn) | 🔧 | JFE | 基于1.8万个基本面信号构建ML策略，证明特征工程与归纳偏置比复杂模型更能提升实盘Alpha。 |
| [DeepSupp](deepsupp) | 🔧 | DeepSupp | 提出DeepSupp框架，利用多头注意力处理动态价量相关性矩阵，结合自编码器与DBSCAN聚类精 |
| [多源融合通用选股框架](art-225) | 🔧 | 多源融合通用选股框架 | 提出多源融合DIY选股框架，将DTW规整成本融入XGBoost梯度，结合个体风险筛选与群体MCD |
| [Fama-MacBeth回归](fama-macbeth) | 🔧 | JFQA | 利用5300万账户数据揭示中国散户异质性：小散交易呈反向预测，大散具择时能力，但高换手成本侵蚀超 |
| [LLM+MCTS Alpha Mining](llm-mcts-alpha-mining) | 🔧 | LLM+MCTS Alpha Mining | 结合大语言模型与蒙特卡洛树搜索，利用回测反馈引导搜索，自动化挖掘可解释的公式化Alpha因子。 |
| [FinBERT+XGBoost+SHAP](finbert-xgboost-shap) | 🔧 | FinBERT+XGBoost+SHAP | 结合FinBERT情绪评分与XGBoost+SHAP可解释框架，构建跨宏观资产的日频新闻情绪Al |
| [Chatterjee ξ](chatterjee) | 🔧 | Chatterjee ξ | 介绍Chatterjee相关系数(ξ)，对比皮尔逊系数，提供Python实现与量化因子评估中的适 |
| [HPPO-TO](hppo-to) | 🔧 | HPPO-TO | 提出HPPO-TO框架，结合分层强化学习与迁移学习，自动挖掘日内风险因子，在三大市场实现25%超 |
| AlphaPy | 🔧 | AlphaPy | 基于Python与scikit-learn构建的算法交易ML框架，集成技术指标特征工程、多模型训 |
| [Style Miner](style-miner) | 🔧 | Style Miner | 提出Style Miner，用约束强化学习(CMDP+PPO)自动挖掘兼具高解释力与高稳定性的股 |
| [RealMLP](realmlp) | 🔧 | NeurIPS24 | 提出RealMLP改进表格数据建模，结合PBLD嵌入、可训练激活与元学习调参，在精度与训练时间上 |
| [基于动量与移动平均线的趋势跟踪](art-107) | 🔧 | Quantitative Finance | 对比动量与移动平均线趋势跟踪规则，基于AR模型证明MA在不确定市场下对窗口变化更鲁棒且平均夏普更 |
| [RSAP-DFM](rsap-dfm) | 🔧 | RSAP-DFM | 提出RSAP-DFM模型，通过双态转换与对抗后验因子，自适应从量价数据提取宏观信息并动态映射至股 |
| [Street PE](street-pe) | 🔧 | Street PE | 提出剔除一次性项目的拟制性盈余(Street PE)替代GAAP收益，有效解释股价过度波动并显著 |
| [QFR](qfr) | 🔧 | QFR | 提出QFR算法，改进REINFORCE引入基线与IR奖励塑造，实现低方差、高稳定性的公式化Alp |
| [MLP](mlp) | 🔧 | MLP | 提出基于MLP的加密货币交易算法，通过新颖标记方案将预测转为分类，结合技术指标与蜡烛图特征实现多 |
| [AlphaForge](alphaforge) | 🔧 | AlphaForge | 用生成-预测神经网络挖掘公式化Alpha，结合动态权重组合成Mega-Alpha，在A股实盘中验 |
| [TWMA/ResNet Trader](twma-resnet-trader) | 🔧 | TWMA/ResNet Trader | 利用ResNet分析K线图提取重要性权重，通过三重I加权移动平均增强传统技术交易信号，并在A股实 |
| [AlphaEvolve](alphaevolve) | 🔧 | AlphaEvolve | 提出AlphaEvolve框架，结合AutoML与进化算法自动挖掘低相关性、高夏普的公式型Alp |
| [超越Fama-French因子：短期信号的阿尔法机会](fama-french) | 🔧 | — | 结合5类短期信号与低成本交易规则，构建出独立于Fama-French因子且稳健的净阿尔法策略。 |
| [NeuralBeta](neuralbeta) | 🔧 | NeuralBeta | 提出NeuralBeta模型动态估计单/多变量β系数，引入可解释输出层显式量化样本权重，提升透明 |
| [NeuralFactors](neuralfactors) | 🔧 | NeuralFactors | Bloomberg提出基于VAE/CIWAE的NeuralFactors模型，通过深度学习自动学 |
| [如何使用强化学习筛选因子](art-21) | 🔧 | [arXiv](https://arxiv.org/abs/2101.09460) | 将因子选择建模为MDP，利用TD强化学习动态探索特征子集，以最大化预测精度并提升模型可解释性。 |
| [XTab](xtab) | 🔧 | ICML23 | 提出基于联邦学习的跨表格预训练框架XTab，通过独立特征提取器与共享Transformer骨干， |
| [Alpha2](alpha2) | 🔧 | Alpha2 | 提出Alpha2框架，利用DRL与MCTS将因子挖掘转化为程序生成任务，自动搜索逻辑合理且低相关 |
| [3万个因子，数据挖掘能超越同行审议的因子吗？](factor-zoo-vs-peer-review) | 🔧 | — | 对比2.9万数据挖掘因子与同行评审因子，发现两者样本外预测能力相当，风险理论因子衰减更快，数据挖 |
| JFQA｜投资者关注度与内幕交易 | 📖 | JFQA | 本文发现内部人士利用散户关注度引发的错误定价进行机会主义交易，高关注时卖出、低关注时买入，该策略 |
| ORB+RelVol | 📖 | ORB+RelVol | 导读美股ORB策略论文，指出叠加开盘相对成交量筛选可大幅提升收益，但提示复现结果存疑需谨慎。 |
| 利用基本面分析预测股票趋势 | 📖 | — | 本文对比LR、CNN与LSTM在基本面数据上的股票趋势预测效果，发现LR在测试集上泛化能力最强， |
| A股投资者情绪与日内过度交易行为的关联性 | 📖 | — | 基于股吧文本构建情绪指数，实证检验其对A股日内过度换手率的影响，发现情绪正向驱动过度交易且机构更 |
| [像生成语言一样生成Alpha：基于语法引导学习的因子挖掘](alpha) | ❌ | — | 本文实为QuantML社群推广帖，未提供具体论文内容或方法细节，仅提及基于语法引导的因子挖掘概念 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
