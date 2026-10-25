# 评测基准与失效分析

> **收錄**：回测防穿越、样本外稳定性、过拟合检测、鲁棒性量化、公开基准(TFB/QuantBench)与模型衰减归因
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：30 篇（24 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| FactorBench | ⚡ | FactorBench | FactorBench 在五大市场统一评测九种自动因子挖掘方法共约五千个因子，检验风格剥离与交易 |
| [Bayes-Stein Shrinkage](bayes-stein-shrinkage.md) | ⚡ | Bayes-Stein Shrinkage | 基于流动性过滤与贝叶斯收缩，验证2006年后经典学术Alpha在大盘股中收益趋零，揭示回测与实盘 |
| [非对称可解释性原则](art-408.md) | ⚡ | 非对称可解释性原则 | 提出非对称可解释性原则，将透明度要求从特征层后移至决策与监控层，构建六层架构防过拟合与可审计。 |
| [KTD-FIN](ktd-fin.md) | ⚡ | KTD-FIN | 提出KTD-FIN基准，通过数据掩码与Barra归因剥离记忆与风格，证明主流LLM交易收益实为风 |
| [FinStressTS](finstressts.md) | ⚡ | FinStressTS | 构建参数可控的金融时序合成基准FinStressTS，系统诊断深度模型在低信噪比与机制切换下的失 |
| [深度优于广度：线性模型样本外衰减](art-337.md) | ⚡ | — | 推导线性模型样本外夏普衰减解析公式，量化过拟合惩罚，证明强信号优于弱信号堆砌。 |
| [盈亏平衡夏普比率](art-279.md) | ⚡ | JFE | 揭示多因子模型高样本内夏普比率的陷阱，量化估计风险导致的样本外衰减，并提出考虑估计风险的盈亏平衡 |
| [大语言模型在时间序列预测中真的有效吗？](art-33.md) | ⚡ | — | 通过系统消融证明LLM在时序预测中无性能优势且计算昂贵，简单注意力模型即可替代，呼吁转向LLM+ |
| [Base-Rate-Honest Benchmark](base-rate-honest-benchmark.md) | 🔧 | [arXiv](https://arxiv.org/abs/2607.12248) | 本文指出股票方向预测中原始准确率易受牛市基线率误导，提出基于滚动验证与诚实基线的可复现评测协议， |
| [NoLBERT](nolbert.md) | 🔧 | [arXiv](https://arxiv.org/abs/2610.22745) | 构建时点NoLBERT新闻主题，严格样本外比较文本因子；最佳验证实验亏钱，组合未胜价格基准。 |
| [比过拟合更隐蔽：一套日内系统逐单对账挖出的六类系统缺陷](art-436.md) | 🔧 | [arXiv](https://arxiv.org/abs/2610.21846) | 作者把实盘成交与离线重放逐笔核对一致，定位出数据源默认参数、复权、时区、配置链路等六类隐蔽工程缺 |
| FinAutoRubric | 🔧 | FinAutoRubric | 提出FinAutoRubric，用专家原则库＋异构双模型沙盒交叉验算，自动生成带数值锚定的金融研 |
| [Cost-Aware Execution Filter](cost-aware-execution-filter.md) | 🔧 | Cost-Aware Execution Filter | 揭示高频交易中预测精度与实盘收益的割裂，提出交易成本感知过滤器，证明执行端纪律比复杂模型更关键。 |
| [LightQuant](lightquant.md) | 🔧 | LightQuant | 构建中国股市多模态数据集CSMD与轻量级回测框架LightQuant，结合LLM因子提取提升股价 |
| [Walk-Forward Analysis / 分层验证框架](walk-forward-analysis.md) | 🔧 | Walk-Forward Analysis / 分层验证框架 | 本文系统剖析传统回测过拟合风险，详解Walk-Forward分析原理、WFE指标及分层验证框架， |
| [FinSearchComp](finsearchcomp.md) | 🔧 | FinSearchComp | 提出开源金融搜索推理基准FinSearchComp，含635题三档任务，系统评测21款LLM代理 |
| [AlphaEval](alphaeval.md) | 🔧 | AlphaEval | 提出免回测的AlphaEval框架，从预测、稳定、鲁棒、逻辑、多样性五维评估Alpha挖掘模型， |
| [RobustiPy](robustipy.md) | 🔧 | RobustiPy | 提出RobustiPy框架，通过多元宇宙样分析与设定曲线，系统化量化模型设定不确定性，提升实证研 |
| [QuantBench](quantbench.md) | 🔧 | QuantBench | 提出QuantBench工业级基准平台，覆盖因子挖掘至订单执行全流程，系统评估AI模型在量化投资 |
| [LOBCAST](lobcast.md) | 🔧 | LOBCAST | 全面对比15个基于LOB的深度学习趋势预测模型，发现其在FI-2010上过拟合，跨市场泛化与鲁棒 |
| [FinTSB](fintsb.md) | 🔧 | [arXiv](https://arxiv.org/abs/2502.18834) | 提出FinTSB金融时序评测基准，涵盖多元波动模式数据集、统一多维指标框架与真实交易约束，填补学 |
| [FinTSBridge](fintsbridge.md) | 🔧 | FinTSBridge | 提出FinTSBridge框架，构建三类金融时序数据集与msIC/msIR评估指标，系统评测十余 |
| [LOB-Bench](lob-bench.md) | 🔧 | LOB-Bench | 提出LOB-Bench基准框架，通过分布差异、市场影响响应与对抗判别器，定量评估生成式模型在限价 |
| [TFB](tfb.md) | 🔧 | TFB | 提出TFB基准测试，涵盖10领域8000+数据集，系统评估35种时序预测方法，提供统一流水线与公 |
| Academic Alpha 实现框架（AQR 白皮书） | 📖 | Academic Alpha 实现框架（AQR 白皮书） | 解读AQR白皮书：学术Alpha变实盘Alpha要过数据时点、中性化、成本容量、持有期四道损耗。 |
| AI时代的Alpha衰减 | 📖 | — | 解读纽大论文：AI普及使量化因子半衰期从五年压缩到18个月，算法同质化与反身性数据污染是主因。 |
| ICML 2026 量化论文合集 | 📖 | ICML 2026 | 系统梳理ICML 2026量化核心论文，涵盖组合优化、LLM金融评估、市场模拟、强化学习与高频时 |
| Prospectus-based AI identification | 📖 | HKMA Working Paper | 基于全球1.9万只AI基金披露文本与持仓数据，实证检验AI选股是否加剧策略拥挤与顺周期踩踏。 |
| KDD 2026 量化论文合集 | 📖 | KDD 2026 | 系统梳理KDD 2026量化金融论文，覆盖LLM策略、组合管理、宏观预测、风控反欺诈与交易自动化 |
| 机器学习在金融市场中的应用：何时有效，何时无效 | 📖 | — | 系统梳理ML在金融市场的适用场景与核心瓶颈（数据噪声、黑箱、过拟合、市场非平稳），并探讨相对价值 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
