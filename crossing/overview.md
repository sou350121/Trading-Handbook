# Crossing — 按张力导航，而非按方法族

> Foundations 按**方法族**切（时序 / 图 / RL / LLM / 微观结构…）；Crossing 把同一批方法**重新切一刀**，按它们在真实交易里被迫做的**取舍**对齐。一个做过实盘的人不会问「这是 Transformer 还是 GNN」，他会问「这条路的容量、成本、可解释性、regime 脆性在哪崩」。六张张力图就是这六把刀。

**Status:** v0.5 — Opus 手寫綜合。每张图含五轴投影 + 判别维度对比 + 何时崩 + 代表方法（链接到 Foundations 解构页）。语料仍在增长，部分极的正例标 (待补)。

## 六张张力图

| # | 张力 | 一句话冲突 |
|---|---|---|
| 1 | [预测驱动 vs 策略驱动](/crossing/supervised-vs-rl/overview) | 损失函数把交易成本放在外面的转换层，还是吞进 reward 里——准信号 vs 会交易。 |
| 2 | [可解释因子 vs 端到端黑盒](/crossing/explicit-factors-vs-e2e/overview) | alpha 长成人能读的公式，还是溶进隐层——可归因/可合规 vs 拟合上限更高。 |
| 3 | [大模型逻辑 vs 专用时序基座](/crossing/llm-reasoning-vs-ts-baselines/overview) | 信号来自语义叙事还是数值结构——LLM 读得懂为什么但算不准，TS 基座算得准但看不到为什么。 |
| 4 | [静态关联 vs 动态微观流](/crossing/static-graph-vs-dynamic-flow/overview) | 两个差几个数量级的时间常数——关系图定权重（慢/容量大）vs 订单流定时机（快/容量墙）。 |
| 5 | [两步法 vs 联合优化](/crossing/predict-then-optimize-vs-end-to-end/overview) | 预测和组合是两个可单测模块，还是一个反传到底的损失——可审计的硬约束 vs 全局最优但梯度塌缩。 |
| 6 | [人机协同 vs 全自动智能体](/crossing/human-in-loop-vs-autonomous-agent/overview) | 最终决策权和兜底责任在谁手上——可问责/可拉闸 vs 速度广度碾压但同质化踩踏。 |

## 怎么读

- **先看「五轴投影」**：每张图会指认哪几根轴**判别**两极、哪几根**正交**（不分边）。这是把方法投回[五轴本体](/cheat-sheet/ontology)的入口。
- **再看「判别维度对比表」**：样本效率 / 容量衰减 / 解释性 / 回测过拟合风险 / 实盘落地成本 / 失效场景——desk quant 真正会问的列。
- **最关键是「何时崩」**：每一极的失效模式不是缺点罗列，是「在什么工况下这条路会把你害死」。
- **「代表方法」**链接到 Foundations 解构页（已写的），或给出 zone + msgid 注册行（标 (待补) 的待语料增长后补写）。

> 同一个方法常出现在多张图里（如 MacroHFT 同时在图 1 的 RL 极和图 4 的微观流极），这是设计特性——跨族信息由五轴坐标承载，在 Crossing 撈出。
