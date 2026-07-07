# 强化学习与策略搜索

> **收錄**：DRL/MARL在选股、调仓、做市、执行与对冲中的应用，涵盖PPO/SAC/分布RL、奖励函数设计与环境模拟
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：41 篇（39 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [ReCAP](recap) | ⚡ | KDD26 | 提出ReCAP持续学习框架，通过CUSUM自适应切分市场制度、存储策略差异向量与动态网关融合，解 |
| [OPHR](ophr) | ⚡ | OPHR | 提出OPHR多智能体强化学习框架，通过OP-Agent择时与HR-Agent路由对冲，在加密货币 |
| [JaxMARL-HFT](jaxmarl-hft) | ⚡ | JaxMARL-HFT | 提出首个GPU加速的HFT多智能体强化学习框架JaxMARL-HFT，基于JAX实现异构智能体并 |
| [AlphaQuanter](alphaquanter) | ⚡ | AlphaQuanter | 提出AlphaQuanter单Agent框架，用RL端到端优化LLM工具调用与交易决策，实现高收 |
| [MetaTrader](metatrader) | ⚡ | MetaTrader | 提出MetaTrader双层强化学习框架，通过数据变换与保守TD目标解决金融非平稳性下的OOD泛 |
| [MacroHFT](macrohft) | ⚡ | KDD'24 | 提出MacroHFT框架，通过市场分解训练子代理，并用记忆增强超代理混合决策，提升加密货币高频交 |
| [PPO策略自适应选择框架](ppo-hedge-fund-strategy-selection) | 🔧 | PPO策略自适应选择框架 | 提出基于PPO的强化学习框架，动态切换动量与均值回归策略，以夏普比率为奖励优化对冲基金风险调整后 |
| [FNN+DRL组合框架](fnn-drl) | 🔧 | FNN+DRL组合框架 | 结合FNN基本面选股与PPO-CTCN周度调仓，引入相对收益奖励与湍流风控，实现稳健跑赢基准的组 |
| [贝莱德 ｜ 深度强化学习在最优订单执行中的应用](art-359) | 🔧 | — | 提出基于Actor-Critic的DRL模型解决日内最优订单执行，通过收益与冲击惩罚平衡，显著优 |
| [ProtoHedge](protohedge) | 🔧 | ProtoHedge | 提出ProtoHedge可解释对冲框架，通过原型相似度加权生成决策，在保持完全透明的同时实现与黑 |
| [中国科学院大学 ｜ LLM x 强化学习：从新闻中学习交易](llm-x) | 🔧 | — | 利用LLM提取新闻情绪，结合原始量价数据输入RL智能体，实现无手工特征的端到端加密货币交易。 |
| [DeepAries](deeparies) | 🔧 | DeepAries | 提出DeepAries框架，将调仓间隔作为离散动作与权重联合优化，结合Transformer与P |
| [TD3](td3) | 🔧 | TD3 | 基于TD3算法训练DRL智能体对冲标普500期权，在17年样本外数据上验证其优于传统Delta对 |
| [TINs](tins) | 🔧 | TINs | 提出TINs框架，将MACD等传统指标重构为可解释神经网络，并用强化学习动态优化参数以提升交易收 |
| [R²-SAC](r-sac) | 🔧 | R²-SAC | 提出R²-SAC框架，将SAC策略解耦为“松弛”生成个股粗略动作与“精炼”融合拐点择时及图谱选股 |
| [QTMRL](qtmrl) | 🔧 | QTMRL | 提出QTMRL框架，将多维技术指标与A2C强化学习结合，实现自适应投资组合管理与交易决策。 |
| [Stock-Evol-Instruct](stock-evol-instruct) | 🔧 | Stock-Evol-Instruct | 结合DQN/DDQN与LLM，提出Stock-Evol-Instruct指令演进算法微调大模型， |
| [混合量子-经典PPO框架](quantum-ppo-sector-rotation) | 🔧 | 混合量子-经典PPO框架 | 构建混合量子-经典RL框架用于台股市行业轮动，发现量子模型虽训练奖励高，但因奖励设计错位，实盘回 |
| [IaC-MARL](iac-marl) | 🔧 | IaC-MARL | 提出多智能体强化学习框架解决多订单执行问题，设计意图感知通信协议与价值归因方法实现智能体协同与现 |
| [iRDPG](irdpg) | 🔧 | iRDPG | 提出iRDPG模型，将交易建模为POMDP，结合RDPG与模仿学习（示范回放+行为克隆），在股指 |
| [Logic-Q](logic-q) | 🔧 | Logic-Q | 提出Logic-Q框架，用轻量级程序草图嵌入专家趋势知识，后验调整DRL策略动作分布，提升订单执 |
| [DQN](dqn) | 🔧 | DQN | 基于DQN构建多资产期货交易的MDP框架，利用波动率标准化回报作为状态，通过真实与模拟数据训练全 |
| [深度强化学习在期权对冲中的应用](art-187) | 🔧 | — | 本文系统对比8种DRL算法在动态期权对冲中的表现，发现MCPG与PPO在稀疏奖励环境下显著优于传 |
| [强化学习新突破：集成方法+GPU并行模拟用于交易](ensemble-rl-gpu-sim) | 🔧 | — | 结合集成学习与GPU大规模并行模拟，解决金融强化学习中的策略不稳定与采样瓶颈，提升交易收益与稳定 |
| [Auto.gov](auto-gov) | 🔧 | Auto.gov | 提出Auto.gov框架，用DQN强化学习动态调整DeFi借贷协议抵押因子，提升治理效率与抗攻击 |
| [FinRL-DeepSeek](finrl-deepseek) | 🔧 | FinRL-DeepSeek | 结合LLM新闻评分扰动PPO/CVaR-PPO策略，实现风险与推荐双维度的强化学习自动化交易。 |
| [NEAT-Python](neat-python) | 🔧 | NEAT-Python | 结合NEAT算法与多技术指标，设计多目标适应度函数与渐进训练，实现低回撤、高稳定性的日频波段交易 |
| [DeepScalper](deepscalper) | 🔧 | DeepScalper | 提出DeepScalper框架，融合分支对抗Q网络、Hindsight奖励与多模态市场嵌入，实现 |
| [TradeMaster](trademaster) | 🔧 | NeurIPS22 [arXiv](https://arxiv.org/abs/2201.01901) | 南洋理工大学开源的强化学习量化交易平台，集成数据预处理、Gym环境、主流RL算法与多维评估，助力 |
| [FinRL](finrl) | 🔧 | FinRL | 介绍FinRL开源DRL框架，提供数据-环境-代理三层架构与模块化设计，支持股票交易与组合管理的 |
| [TRIALS](trials) | 🔧 | TRIALS | 提出分层强化学习框架，将配对选择与交易统一优化，在美股与A股数据上实现更优的风险收益比。 |
| [HRT](hrt) | 🔧 | HRT | 提出分层强化学习框架HRT，用PPO选股、DDPG调仓，通过交替训练优化股票选择与执行，提升夏普 |
| [深度强化学习应对加密货币交易过拟合](art-88) | 🔧 | — | 提出基于假设检验与组合交叉验证的过拟合概率估计方法，有效筛选并拒绝DRL交易代理，提升加密货币实 |
| [TF-Agents](tf-agents) | 🔧 | TF-Agents | 基于TF-Agents构建DQN交易智能体，融合量价与技术宏观特征，通过自定义环境学习多空持有策 |
| [Model A](model-a) | 🔧 | Model A | 提出多智能体DNN+强化学习模型，仅用开盘价与成交量预测标普500期货日频方向，通过动态仓位控制 |
| [SAC-CNN-MHA](sac-cnn-mha) | 🔧 | SAC-CNN-MHA | 提出基于SAC与CNN-MHA的RL组合管理模型，引入借贷机制与PnL奖励函数，在加密市场实现高 |
| [强化学习在配对交易中的应用与优化](art-32) | 🔧 | — | 将强化学习引入加密货币配对交易，通过自定义观察/行动空间与多部分奖励塑形，优化开平仓时机与仓位比 |
| [JAX-LOB](jax-lob) | 🔧 | [arXiv](https://arxiv.org/abs/2308.13289) | 提出基于JAX的GPU加速LOB模拟器JAX-LOB，集成GYMNAX环境，实现RL交易训练5- |
| [FinRL-Meta](finrl-meta) | 🔧 | FinRL-Meta | 介绍FinRL-Meta开源框架，通过DataOps自动化数据管理与Gym环境构建，支持DRL代 |
| 从做市到资管：金融强化学习落地全景 | 📖 | — | 基于PRISMA综述167篇文献，构建合成数据集元分析金融RL绩效，揭示做市技术溢出与工程实现优 |
| gym-anytrading | 📖 | gym-anytrading | 本文探讨强化学习在量化交易中的应用，对比DQN/PPO/A2C算法，分析技术指标输入、归一化方法 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
