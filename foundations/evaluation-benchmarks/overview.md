# 评测基准与失效分析

> **收錄**：回测防穿越、样本外稳定性、过拟合检测、鲁棒性量化、公开基准(TFB/QuantBench)与模型衰减归因
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：18 篇（17 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [KTD-FIN](ktd-fin) | ⚡ | KTD-FIN | 提出KTD-FIN基准，通过数据掩码与Barra归因剥离记忆与风格，证明主流LLM交易收益实为风 |
| [FinStressTS](finstressts) | ⚡ | FinStressTS | 构建参数可控的金融时序合成基准FinStressTS，系统诊断深度模型在低信噪比与机制切换下的失 |
| [深度优于广度：线性模型样本外衰减](art-337) | ⚡ | — | 推导线性模型样本外夏普衰减解析公式，量化过拟合惩罚，证明强信号优于弱信号堆砌。 |
| [盈亏平衡夏普比率](art-279) | ⚡ | JFE | 揭示多因子模型高样本内夏普比率的陷阱，量化估计风险导致的样本外衰减，并提出考虑估计风险的盈亏平衡 |
| [大语言模型在时间序列预测中真的有效吗？](art-33) | ⚡ | — | 通过系统消融证明LLM在时序预测中无性能优势且计算昂贵，简单注意力模型即可替代，呼吁转向LLM+ |
| [Cost-Aware Execution Filter](cost-aware-execution-filter) | 🔧 | Cost-Aware Execution Filter | 揭示高频交易中预测精度与实盘收益的割裂，提出交易成本感知过滤器，证明执行端纪律比复杂模型更关键。 |
| [LightQuant](lightquant) | 🔧 | LightQuant | 构建中国股市多模态数据集CSMD与轻量级回测框架LightQuant，结合LLM因子提取提升股价 |
| [Walk-Forward Analysis / 分层验证框架](walk-forward-analysis) | 🔧 | Walk-Forward Analysis / 分层验证框架 | 本文系统剖析传统回测过拟合风险，详解Walk-Forward分析原理、WFE指标及分层验证框架， |
| [FinSearchComp](finsearchcomp) | 🔧 | FinSearchComp | 提出开源金融搜索推理基准FinSearchComp，含635题三档任务，系统评测21款LLM代理 |
| [AlphaEval](alphaeval) | 🔧 | AlphaEval | 提出免回测的AlphaEval框架，从预测、稳定、鲁棒、逻辑、多样性五维评估Alpha挖掘模型， |
| [RobustiPy](robustipy) | 🔧 | RobustiPy | 提出RobustiPy框架，通过多元宇宙样分析与设定曲线，系统化量化模型设定不确定性，提升实证研 |
| [QuantBench](quantbench) | 🔧 | QuantBench | 提出QuantBench工业级基准平台，覆盖因子挖掘至订单执行全流程，系统评估AI模型在量化投资 |
| [LOBCAST](lobcast) | 🔧 | LOBCAST | 全面对比15个基于LOB的深度学习趋势预测模型，发现其在FI-2010上过拟合，跨市场泛化与鲁棒 |
| [FinTSB](fintsb) | 🔧 | [arXiv](https://arxiv.org/abs/2502.18834) | 提出FinTSB金融时序评测基准，涵盖多元波动模式数据集、统一多维指标框架与真实交易约束，填补学 |
| [FinTSBridge](fintsbridge) | 🔧 | FinTSBridge | 提出FinTSBridge框架，构建三类金融时序数据集与msIC/msIR评估指标，系统评测十余 |
| [LOB-Bench](lob-bench) | 🔧 | LOB-Bench | 提出LOB-Bench基准框架，通过分布差异、市场影响响应与对抗判别器，定量评估生成式模型在限价 |
| [TFB](tfb) | 🔧 | TFB | 提出TFB基准测试，涵盖10领域8000+数据集，系统评估35种时序预测方法，提供统一流水线与公 |
| 机器学习在金融市场中的应用：何时有效，何时无效 | 📖 | — | 系统梳理ML在金融市场的适用场景与核心瓶颈（数据噪声、黑箱、过拟合、市场非平稳），并探讨相对价值 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
