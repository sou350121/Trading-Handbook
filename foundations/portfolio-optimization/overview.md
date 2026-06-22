# 组合优化与资产配置

> **收錄**：均值方差/风险平价/Black-Litterman、稀疏优化、执行算法(VWAP/冲击成本)、多资产轮动与战术配置
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：34 篇（31 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [异质性信念均衡模型](art-399) | ⚡ | 异质性信念均衡模型 | 重构70年机构预期数据，验证异质性信念模型，证明机构逆周期预期对中长期市场回报与波动率具强预测力 |
| [STABLE](stable) | ⚡ | ICLR26 | 用条件扩散模型生成多路径收益率作为BL主观观点，结合卡尔曼动态敏感度与历史先验贝叶斯融合，实现稳 |
| [DeePM](deepm) | ⚡ | DeePM | 提出DeePM框架，通过定向延迟、宏观图谱先验与SoftMin-EVaR目标，实现端到端宏观资产 |
| [RTS-PnO](rts-pno) | ⚡ | KDD25 | 提出RTS-PnO框架，通过端到端PnO范式与自适应不确定性约束对齐预测与决策目标，实盘降低8. |
| [BPQP](bpqp) | ⚡ | NeurIPS24 | 提出BPQP框架，将可微分凸优化的反向传播重构为二次规划问题，实现前后向分离，大幅提升端到端投资 |
| [拒绝先预测再优化：将 Sharpe/Omega/CVaR塞入](sharpe-omega-cvar) | 🔧 | — | 提出端到端可微组合优化框架，将Sharpe/Omega/CVaR平滑为损失函数，直接输出权重并优 |
| [YAND](yand) | 🔧 | YAND | 提出YAND算法，利用微分几何仿射正规方向与精确线搜索，在千级资产规模上高效求解含偏度峰度的组合 |
| [DAR4020](dar4020) | 🔧 | DAR4020 | 基于220年数据提出DAR4020策略，通过因子相关性动态多空配置，结合趋势跟踪有效对冲60/4 |
| [cGAN策略克隆框架](cgan) | 🔧 | cGAN策略克隆框架 | 提出条件GAN框架，无需预设效用函数即可从持仓数据中学习基金经理策略的潜在表征并生成投资组合。 |
| [LLM-BLM框架](llm-blm) | 🔧 | LLM-BLM框架 | 将LLM预测均值与方差系统转化为BL模型的观点向量与置信矩阵，实现LLM驱动的组合优化，并揭示L |
| [AlphaGAT](alphagat) | 🔧 | IJCAI 25 | 提出两阶段AlphaGAT框架：先用CATimeMixer挖掘低相关Alpha因子，再用PPO+ |
| [Signature-Informed Transformer (SIT)](signature-informed-transformer-sit) | 🔧 | Signature-Informed Transformer (SIT) | 提出SIT框架，将路径特征融入Transformer注意力，端到端直接优化CVaR实现稳健资产配 |
| [STRAPSim](strapsim) | 🔧 | STRAPSim | 提出STRAPSim方法，通过语义匹配、动态权重转移与残差感知，精准度量投资组合相似性，优于传统 |
| [MoEDRLPM](moedrlpm) | 🔧 | MoEDRLPM | 提出MoEDRLPM模型，结合时空Transformer与混合专家机制，通过DRL实现股票权重分 |
| [将利润作为损失函数构造端到端交易模型](art-272) | 🔧 | — | 提出四种利润导向损失函数，使时序模型无需RL即可端到端输出组合仓位决策，回测显著跑赢RL基准。 |
| [HARLF](harlf) | 🔧 | HARLF | 提出HARLF分层框架，结合FinBERT情感分析与DRL，通过三层智能体融合多模态信号实现月度 |
| [DSL](dsl) | 🔧 | DSL | 提出DSL框架，将组合优化转化为监督学习任务，直接预测最优权重，结合深度集成显著降低方差并提升风 |
| [DiT-LSTM-SVAR](dit-lstm-svar) | 🔧 | DiT-LSTM-SVAR | 结合DiT数据增强、LSTM预测与SVAR噪声过滤，构建剔除随机游走股票的高频投资组合。 |
| [2024 JFQA 夏普奖出炉! 为何1/N分散化组合难以被](2024-jfqa-1-n) | 🔧 | JFQA | 解析2024 JFQA夏普奖论文，揭示1/N组合因估计风险与单因子分散优势难以被击败，并提出结合 |
| [DGT](dgt) | 🔧 | DGT | 提出动态网格策略(DGT)，以“重置”替代“终止”机制，在加密市场实现正期望与更低回撤。 |
| [Grid-FW](grid-fw) | 🔧 | Grid-FW | 提出基于布尔松弛与可调参数的连续优化框架，用改进Frank-Wolfe算法高效求解稀疏最小方差组 |
| [基于机器学习及宏观经济状态检测的战术资产配置](art-201) | 🔧 | — | 结合宏观数据与K-means聚类检测经济状态，将其作为概率条件输入预测模型，优化战术资产配置策略 |
| [LLM-BL](llm-bl) | 🔧 | LLM-BL | 利用LLM预测股票收益及方差构建观点，融入Black-Litterman模型进行双周调仓组合优化 |
| [TLN-VWAP](tln-vwap) | 🔧 | TLN-VWAP | 提出基于深度学习的VWAP执行框架，绕过成交量预测，直接优化滑点损失，在加密货币市场实现更低执行 |
| [改进CTGAN与CVaR组合优化](ctgan-cvar) | 🔧 | 改进CTGAN与CVaR组合优化 | 用改进CTGAN生成合成收益场景，结合CVaR线性规划优化资产配置，提升收益与风控。 |
| [SDFL](sdfl) | 🔧 | SDFL | 提出半决策导向学习(SDFL)结合深度集成，将组合优化转化为监督学习，提升深度学习调仓的稳定性与 |
| skfolio | 🔧 | skfolio | 介绍基于scikit-learn的Python库skfolio，演示均值方差优化、有效前沿绘制及 |
| [排名空间统计套利](art-78) | 🔧 | 排名空间统计套利 | 提出基于市值排名的统计套利框架，利用残差收益的强均值回归特性，结合神经网络与日内再平衡机制实现高 |
| [MASA](masa) | 🔧 | AAMAS 24 | 提出MASA多智能体框架，结合TD3强化学习与约束求解器及市场观察者，实现动态投资组合的风险与回 |
| [GAT](gat) | 🔧 | GAT | 本文利用GAT结合距离相关性与TMFG图滤波，以负对数夏普比率为损失函数，实现大规模时变投资组合 |
| [DeepUnifiedMom](deepunifiedmom) | 🔧 | [arXiv](https://arxiv.org/abs/2406.08742) | 提出DeepUnifiedMom框架，结合多任务学习与MoE统一捕捉多周期动量，通过软截断夏普损 |
| [DSPO](dspo) | 🔧 | DSPO | 提出DSPO端到端框架，融合高频与日频数据，通过单调逻辑回归损失直接优化股票排序组合构建。 |
| 宏观动量策略 | 📖 | 宏观动量策略 | 本文系统阐述基于宏观基本面趋势的动量策略，论证其跨资产低相关性、熊市对冲优势及与趋势跟踪的互补性 |
| 随机投资组合理论(SPT) | 📖 | 随机投资组合理论(SPT) | 系统梳理美股宏观特性（资本分布、多样性、内在波动与排名动态），为SPT与主动组合管理提供实证基础 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
