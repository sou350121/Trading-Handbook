# 预测驱动 vs 策略驱动（Supervised Forecasting vs RL Policy Search）

> **本質衝突**：监督学习把「预测收益率」当目标，信号准但不知道怎么交易；强化学习把「最大化回报」当目标，会交易但不知道为什么。两条路在**优化目标与交易成本的内化位置**上分叉。

**Status:** v0.6 — Opus 手寫綜合，非摘要。RL 极已随语料增长大幅补实（reinforcement-learning 入库 ~39 篇，含做市/执行/对冲/持续学习等正例）。

## 中心张力

监督预测与策略搜索的根本分歧，不在模型架构（两边都能用 Transformer/LSTM/MoE），而在**损失函数把交易成本放在哪里**。监督回归把目标钉死在「下一期收益率的点估计」或「截面排序」上，IC/RankIC 是它的母语；它对成本、滑点、换手、容量一无所知，这些全部被推到一个事后的、人手搭的「信号→仓位」转换层里。这个解耦让监督路线**可归因、可单测、可热插拔**——你能把任意一个 forecaster 接到同一套组合构建器后面（语料里 StockFormer、xLSTM-Mixer、ARMA-Attention 全是这种「换头不换身」的产物）。代价是：预测误差在转换层会被**非线性放大**，一个 RankIC 0.05 的因子在 5bp 双边成本、日频全换手下完全可能是负 Sharpe，而 forecaster 训练时根本看不到这件事。

强化学习把这个转换层**吞进损失函数**。仓位、调仓时点、甚至成本本身都进入 reward，于是策略「天生」就知道高换手要付费、知道流动性薄时要缩量（MacroHFT 的记忆增强、DeepAries 的「该不该现在调仓」决策都是这个逻辑）。但这份内化是要还的：RL 的 reward 极稀疏、信号噪比远低于监督的逐样本标签，于是它**疯狂依赖环境模拟器的保真度**。模拟器里没有的冲击成本、没建模的对手方反应、回测里穿越的未来流动性，全都会变成「在沙盒里赚钱、上实盘亏钱」的 sim-to-real gap。在哪里咬人最狠：**容量**（RL 学到的微观时机往往是小资金现象，放大就被自己的冲击吃掉）、**regime 脆性**（监督因子换 regime 是 IC 衰减，RL 换 regime 可能是策略直接发散）、**可审计性**（风控/合规要追问「为什么这笔」，监督有因子值可指，RL 只有一个 Q 值）。

## 五轴投影

| 轴 | 预测驱动（监督） | 策略驱动（RL） | 是否判别 |
|---|---|---|---|
| 数据模态 | 量价/表格为主，文本可拼 | 量价 + 微观盘口（reward 需要执行细节） | 弱判别——两端都能吃多模态 |
| 时间尺度 | 日频/波段为甜区 | 高频/日内甜区（成本内化收益最大） | **判别**——RL 越往高频价值越大 |
| 学习范式 | 监督/回归分类 | 强化学习/策略搜索 | **核心判别轴** |
| Alpha生成机制 | 端到端表征或显式因子，**信号**为输出 | 组合/执行优化，**仓位**为输出 | **核心判别轴** |
| 人机协作度 | 全自动黑盒或可解释皆可 | 偏全自动黑盒，可解释性最差 | **判别**——RL 落点更靠黑盒端 |

> 正交轴：**数据模态**。两端都在往多模态走（监督有 MANA-Net 加新闻权重，RL 有 AlphaQuanter 吃多模态），模态不是这条张力的分水岭。真正分水岭是**学习范式 × Alpha生成机制**这对组合。

## 判别维度对比表

| 维度 | 预测驱动（监督） | 策略驱动（RL） |
|---|---|---|
| 样本效率 | 高——逐样本标签，密集监督信号 | 低——reward 稀疏，常需远多于监督的交互量 |
| 成本/滑点感知 | 无，靠事后转换层硬塞 | 原生内化进 reward |
| 容量衰减 | 较缓，因子拥挤是渐进的 | 陡峭，学到的微观时机放大即失效 |
| 解释性 | 中-高，因子值/IC 可追溯 | 低，Q 值/策略网络难归因 |
| 回测过拟合风险 | 中——前瞻偏差、IC 挑选偏差 | 高——叠加**模拟器偏差**这一独立风险源 |
| 实盘落地成本 | 低——可热插拔、可单测每个 forecaster | 高——需高保真模拟器 + 在线安全护栏 |
| 失效场景 | 换 regime → IC 衰减；加成本 → 高换手因子翻负 | sim-to-real gap；reward hacking；薄流动性发散 |
| 调试手感 | 好——可冻结模型逐层查 IC | 差——策略不稳定、种子敏感、难复现 |

## 何时选哪边 / 何时崩

**选监督预测，当**：你的 alpha 主要来自截面排序、信号衰减慢（日频~波段）、需要向风控/资方解释每一笔仓位的归因、或团队还没有可信的高保真模拟器。先把 forecaster 做扎实，把成本-换手约束放进**后接的**组合层（见[两步法 vs 联合优化](/crossing/predict-then-optimize-vs-end-to-end/overview)），是大多数日频选股的正确起点。**崩点**：当成本占 alpha 比重高、当策略本质是「择时/执行」而非「选标的」时，监督的解耦反而把成本信息切断，转换层成了过拟合温床。

**选 RL 策略搜索，当**：成本/冲击是收益的一阶项（高频做市、执行、配对交易的时机），或决策序列耦合强（动态调仓周期、对冲再平衡）——这些场景里「预测」和「行动」无法干净拆开。**崩点**：模拟器不保真（最常见死法）；reward 设计诱导 reward hacking（如只奖励 PnL 不罚回撤 → 学出裸卖波动率）；策略对训练种子/超参极敏感导致实盘不可复现；以及容量幻觉——在历史 LOB 重放里赚的钱，实盘下单会移动它自己依赖的价格。

**组合路线**（语料里的主流缝合）：用监督预测的**置信度约束 RL 的探索边界**——forecaster 给方向先验，RL 只在置信区间内决定 sizing/timing；或反过来，把 RL 当**因子筛选器**而非交易员（语料里「用强化学习筛选因子」「Alpha2 用 DRL 挖逻辑公式因子」都是把 RL 收缩到搜索而非执行，规避了 sim-to-real）。

## 代表方法

**预测驱动（监督）一极**（损失钉在收益率点估计/截面排序、成本推到外部转换层、可热插拔）：
- [StockFormer](/foundations/time-series-forecasting/stockformer)（time-series-forecasting · 2247484801）— 多任务回归/分类 + 高低频分离，典型「信号为输出」
- [xLSTM-Mixer](/foundations/time-series-forecasting/xlstm-mixer)（time-series-forecasting · 2247487252）— 纯预测，混合多变量，「换头不换身」的预测器
- [ARMA-Attention](/foundations/time-series-forecasting/arma-attention)（time-series-forecasting · 2247487011）— 监督序列拟合
- [DoubleAdapt](/foundations/time-series-forecasting/doubleadapt)（time-series-forecasting · 2247484726）— 元学习提升预测，仍是预测范式
- [PortfolioMASTER 选股损失函数](/foundations/time-series-forecasting/portfoliomaster)（time-series-forecasting · 2247491994）— 直击「选股该用什么 loss」，监督路线试图把成本缝进 loss 的代表
- [成本感知执行过滤器](/foundations/evaluation-benchmarks/cost-aware-execution-filter)（evaluation-benchmarks · 2247493963）— **监督路线的痛点实证**：高预测精度在加成本后净收益翻负，正是「forecaster 看不到成本、转换层成过拟合温床」的活体反例

**策略驱动（RL）一极**（仓位/调仓时点/成本吞进 reward、原生内化成本、依赖模拟器保真）：
- [MacroHFT](/foundations/reinforcement-learning/macrohft)（reinforcement-learning · 2247484852）— ⚡ 记忆增强上下文感知 HFT，成本内化的标杆
- [MetaTrader 双层 RL](/foundations/reinforcement-learning/metatrader)（reinforcement-learning · 2247490889）— ⚡ 用时序变换构 OOD 子集做双层元学习、保守 TD 抑制价值高估，**正面治 RL 的非平稳/价值高估病**
- [OPHR 波动率交易 RL](/foundations/reinforcement-learning/ophr)（reinforcement-learning · 2247492259）— ⚡ 择时 + 对冲双 agent，离线神谕初始化解延迟奖励，成本/时机耦合强、监督拆不开的纯例
- [ReCAP 防 Alpha 衰减](/foundations/reinforcement-learning/recap)（reinforcement-learning · 2247493992）— ⚡ KDD 26，CUSUM 检测 regime 切换 + 轻量策略库动态融合，**直面「RL 换 regime 可能直接发散」**这条崩点
- [DeepAries 智能调仓时机](/foundations/reinforcement-learning/deeparies)（reinforcement-learning · 2247492044）— 学「何时调仓」而非固定周期，RL 内化时机的纯例
- [DeepScalper 日内短暂机会](/foundations/reinforcement-learning/deepscalper)（reinforcement-learning · 2247489018）— Hindsight 奖励 + 微观/宏观多模态，高频侧 RL 价值最大的甜区
- [HRT 分层强化交易员](/foundations/reinforcement-learning/hrt)（reinforcement-learning · 2247487175）— MIT，双层 RL 拆选股与执行，决策序列耦合的样本
- [R²-SAC 松弛-精炼策略](/foundations/reinforcement-learning/r-sac)（reinforcement-learning · 2247491524）— 同花顺，SAC 输出粗动作再精炼，对「RL 直接输出权重不稳」的工程修补
- [贝莱德 DRL 最优订单执行](/foundations/reinforcement-learning/art-359)（reinforcement-learning · 2247492961）— reward 含收益+冲击二次惩罚，成本内化在执行层
- [深度强化学习应对加密货币过拟合](/foundations/reinforcement-learning/art-88)（reinforcement-learning · 2247487160）— 直面 RL 在交易里的过拟合
- [AlphaQuanter 端到端 Agentic RL](/foundations/reinforcement-learning/alphaquanter)（reinforcement-learning · 2247492036）— ⚡ 端到端，跨向[Agent 自主演进](/crossing/human-in-loop-vs-autonomous-agent/overview)
- [TF-Agents 深度 Q 学习交易](/foundations/reinforcement-learning/tf-agents)（reinforcement-learning · 2247487035）/ [Model A 动态深度网络决策](/foundations/reinforcement-learning/model-a)（reinforcement-learning · 2247485953）— RL 交易基础样本

**缝合带（监督先验约束 RL / RL 收缩为搜索器，规避 sim-to-real）：**
- [Alpha2 用 DRL 挖逻辑公式因子](/foundations/factor-mining/alpha2)（factor-mining · 2247484882）— RL 收缩到因子搜索而非交易员，规避 sim-to-real gap
- [用强化学习筛选因子](/foundations/factor-mining/art-21)（factor-mining · 2247485027）— 同思路，RL 当筛选器
- [AlphaSAGE GFlowNets 结构感知挖掘](/foundations/factor-mining/alphasage)（factor-mining · 2247491844）— ⚡ 用 GFlowNets 替代传统 RL 采样多样高奖励因子，「把 RL 关进搜索」的进阶
- [Alpha-R1 因子筛选推理模型](/foundations/factor-mining/alpha-r1)（factor-mining · 2247492859）— ⚡ GRPO 以客观市场回报当奖励训 LLM 做动态因子门控，RL 当搜索/选择器
- [AlphaQCM 分布强化学习挖因子](/foundations/factor-mining/alphaqcm)（factor-mining · 2247490762）— ⚡ ICML 25，分布 RL 搜因子
- [QFR 方差有界 RL 挖因子](/foundations/factor-mining/qfr)（factor-mining · 2247486244）/ [Style-Miner 风格因子构造](/foundations/factor-mining/style-miner)（factor-mining · 2247488487）— RL 在搜索而非执行侧的稳健化
