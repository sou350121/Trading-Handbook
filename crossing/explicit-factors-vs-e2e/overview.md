# 可解释因子 vs 端到端黑盒（Explicit Factors vs End-to-End Representation）

> **本質衝突**：显式因子有公式、有金融先验、能向风控解释，但拟合上限被人类想象力封顶；端到端表征非线性强、能榨干量价里的弱信号，但归因失败、过拟合难防。落地常把因子当 RL 状态先验或损失正则项。

**Status:** v0.5 — Opus 手寫綜合，非摘要。代表方法隨 Pass A 語料增長補全。

## 中心张力

这条张力是量化最古老的一条，也是最容易被「benchmark 漂亮」骗到的一条。显式因子路线（公式型 alpha、符号回归、风格因子）输出的是**人能读的表达式**：`(close - vwap) / std(volume, 20)` 这种东西。它的价值不在拟合精度，而在**可证伪性与可归因**——你能解释它为什么 work（流动性溢价、动量、估值），能在它失效时知道是哪个经济直觉破了，能向合规说清楚每个因子的暴露。代价是拟合上限：人类先验封住了它能表达的函数空间，非线性交互、高阶时序结构基本抓不到，单因子 IC 天花板低，得靠几百个因子的截面组合堆出来（语料里「3万个因子」直接问的就是「暴力挖掘能不能超过同行评议的因子」——这本身就是这条张力的实证）。

端到端表征路线（深度时序模型、图网络、Transformer 选股）把「因子」溶进了网络的隐层，不再有可读的表达式，直接从量价/图结构端到端拟合收益。它能榨干因子路线够不着的弱非线性信号，benchmark IC 普遍更高（StockFormer、Fieldy、THGNN、FinGAT 这些都是这一极）。但代价是三重的：**归因失败**（信号坏了不知道是哪根隐层的锅）、**过拟合更隐蔽**（参数量大，回测 IC 高很可能是记住了历史 regime 而非学到结构）、**容量与拥挤度盲区**（看不到自己的暴露落在哪些已被套利掉的风格上）。在哪里咬人：风控/合规要求可解释（很多机构硬约束端到端模型只能做 overlay 不能做主仓）、换 regime 时端到端模型衰减比因子更陡且更难诊断、以及「benchmark Goodhart」——一旦某个 TS benchmark 成为竞赛目标，端到端模型的 IC 提升里有多少是真 capability、多少是对该数据集的过拟合，几乎无法在论文里分清（这正是 [TFB](/foundations/evaluation-benchmarks/tfb) 这类基准存在的理由）。

## 五轴投影

| 轴 | 可解释因子 | 端到端黑盒 | 是否判别 |
|---|---|---|---|
| 数据模态 | 量价/表格（公式好写） | 量价 + 图 + 文本（隐层吃得下） | 弱判别——因子也在往另类数据扩 |
| 时间尺度 | 跨周期/波段（因子衰减慢） | 日频/波段为主 | 弱判别 |
| 学习范式 | 监督回归 + 符号/元学习搜索 | 监督回归（深度） | 部分判别——搜索范式偏因子端 |
| Alpha生成机制 | **显式因子挖掘** | **端到端表征** | **核心判别轴** |
| 人机协作度 | 人机协同/可解释 | 全自动黑盒 | **核心判别轴** |

> 正交轴：**时间尺度**。两极都覆盖日频~波段，频率不分边。真正的分水岭是 **Alpha生成机制 × 人机协作度**——「alpha 长在哪、人能不能读」。注意有一类「**可解释的端到端**」缝合体（TIMEVIEW、FRI 框架、NeuralFactors）专门在这两轴的中间地带做文章。

## 判别维度对比表

| 维度 | 可解释因子 | 端到端黑盒 |
|---|---|---|
| 拟合上限 | 低——人类先验封顶 | 高——非线性 + 高阶交互 |
| 解释性/归因 | 高——公式可读、暴露可拆 | 低——隐层不可读 |
| 回测过拟合风险 | 中——但拥挤度可监控 | 高——参数多、易记 regime |
| 实盘稳定性 | 高——衰减渐进、可热替换单因子 | 中-低——换 regime 衰减陡且难诊断 |
| 合规/风控友好度 | 高——可作主仓 | 低——常被限为 overlay |
| 拥挤度可见性 | 高——能算因子拥挤、IC 衰减曲线 | 低——暴露藏在隐层 |
| 维护成本 | 中——因子库要持续淘汰 | 高——重训昂贵、调试难 |
| 失效场景 | 因子拥挤/被套利、风格切换 | regime 漂移 + benchmark Goodhart + 沉默退化 |

## 何时选哪边 / 何时崩

**选显式因子，当**：合规/风控要求可解释（受监管资金、对外发产品）、需要长期可维护的因子库、或 alpha 来自有清晰经济直觉的来源（动量/估值/流动性）。**崩点**：人类先验耗尽——市场里剩下的 alpha 越来越多是非线性/高阶的，纯公式因子的边际新增 IC 趋零；且公式因子最容易被同行复现 → 拥挤 → 被套利（[因子的非对称性](/foundations/factor-mining/alpha)讨论的「为什么熊市 alpha 反而强」本质就是拥挤度的反面）。

**选端到端，当**：你有足够样本外验证预算、目标市场流动性深（容量不是一阶约束）、且能接受「不可解释但有效」的 overlay 角色。**崩点**：样本外验证不充分时，端到端的高 IC 极可能是过拟合幻觉；换 regime 时它衰减又快又哑（不报警）；以及落不了主仓——很多实盘里端到端模型最终只能贡献几个 bp 的 overlay，因为风控不批它当核心。

**组合路线**（语料里的主流，也是最务实的）：**因子做端到端的先验/正则**——把公式因子当深度模型的输入特征或损失正则项，既保住可解释骨架又借非线性补强（[AlphaForge](/foundations/factor-mining/alphaforge) 的 Generator-Predictor、AlphaGAT（2247491933）把因子挖掘塞进图注意力都是这条路）；或反过来用端到端去**生成候选因子**，再用人类先验做事后筛选与归因（[Alpha2](/foundations/factor-mining/alpha2)、AlphaSAGE 用 GFlowNets/DRL 搜公式因子——产物仍是可读公式，鱼与熊掌兼得的甜点）。

## 代表方法

**可解释因子一极：**
- [3万个因子能否超越同行评议因子](/foundations/factor-mining/3)（factor-mining · 2247484872）— 直接拷问暴力挖掘 vs 先验因子
- [Street PE 用收益估值](/foundations/factor-mining/street-pe)（factor-mining · 2247486921）— 经典可解释估值因子
- [因子的非对称性](/foundations/factor-mining/alpha)（factor-mining · 2247494088）— 拥挤度/regime 视角
- [VarLiNGAM 时序因果发现](/foundations/causal-structural/varlingam)（causal-structural · 2247486034）— 因果可解释的极端
- AlphaEvolve（factor-mining · 2247485843）— AutoML 挖公式因子，产物可读 (待补)

**端到端黑盒一极：**
- [StockFormer](/foundations/time-series-forecasting/stockformer)（time-series-forecasting · 2247484801）— 多任务深度选股
- [Fieldy 细颗粒度层次 Transformer](/foundations/time-series-forecasting/fieldy)（time-series-forecasting · 2247486273）
- [THGNN 异构图神经网络](/foundations/graph-networks/thgnn)（graph-networks · 2247486310）
- [FinGAT 金融图注意力](/foundations/graph-networks/fingat)（graph-networks · 2247486368）
- [RSR 时序关系排序](/foundations/graph-networks/rsr)（graph-networks · 2247486355）

**缝合带（可解释的端到端 / 因子 × 深度）：**
- [TIMEVIEW 通向透明的时序预测](/foundations/time-series-forecasting/timeview)（time-series-forecasting · 2247484754）— 端到端但强行可解释
- [AlphaForge](/foundations/factor-mining/alphaforge)（factor-mining · 2247486006）— 公式因子 + 动态加权，附实盘验证
- [Alpha2 DRL 挖逻辑公式因子](/foundations/factor-mining/alpha2)（factor-mining · 2247484882）— 搜索产出可读公式
- [XTab 表格预训练](/foundations/factor-mining/xtab)（factor-mining · 2247484944）— 表征学习 + 表格先验
- NeuralFactors（factor-mining · 2247485681）— Bloomberg 生成模型，因子结构可读 (待补)
- AlphaSAGE GFlowNets 结构感知挖掘（factor-mining · 2247491844）— 搜索范式，产物可读 (待补)
- FRI 面向解释性的金融图谱量化框架（graph-networks · 2247491788）— 图端到端但强可解释 (待补)

<!-- TODO: enrich examples when full corpus distilled — 因子端 16 篇、端到端散落 TS/graph 较充实；「可解释的端到端」缝合带是手册最有价值区，待语料增长后单独成节 -->
