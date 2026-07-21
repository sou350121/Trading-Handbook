# 大模型与智能体

> **收錄**：LLM在因子挖掘、情绪分析、研报生成、RAG增强及多智能体交易工作流中的适配、微调与推理架构
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：63 篇（52 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [XALPHA](xalpha) | ⚡ | XALPHA | 提出记忆驱动的多脑闭环AI量化研究员XALPHA，实现从研报阅读、假说生成、代码实现到验证反思的 |
| [Agentic SDLC](agentic-sdlc) | ⚡ | Agentic SDLC | 本文探讨Optiver与Jane Street如何将AI Agent引入量化开发全流程，构建闭环 |
| [PokerSkill](pokerskill) | ⚡ | PokerSkill | 提出PokerSkill框架，用确定性规则与预算系统约束LLM，无需训练或求解器即可在无限注德扑 |
| [AlphaBench](alphabench) | ⚡ | AlphaBench | 构建AlphaBench基准评测LLM因子挖掘能力，揭示零样本评估失效，提出SFT成对比较与AS |
| [TiMi](timi) | ⚡ | ICLR26 | 提出TiMi框架，离线用LLM生成策略代码与数学约束，在线轻量CPU执行，解决大模型实盘延迟与参 |
| [北大光华 | 年化夏普比率2.43！基于自主市场分析的实时A](2-43-ai) | ⚡ | — | 利用自主联网LLM每日评估美股，生成吸引力评分，Top-20组合实现日频高夏普Alpha，但仅擅 |
| [DeepAnalyze](deepanalyze) | ⚡ | DeepAnalyze | 提出8B参数Agentic LLM，通过5种动作指令与课程强化学习，实现全自动数据科学分析与报告 |
| [DeepFund](deepfund) | ⚡ | NeurIPS25 | 提出DeepFund实时基准，用多智能体LLM在真实市场中测试基金投资，揭示回测穿越陷阱与模型交 |
| [ContestTrade](contesttrade) | ⚡ | ContestTrade | 提出ContestTrade框架，通过数据与研究双团队内部竞赛机制，实现LLM多智能体交易信号的 |
| [ASI-ARCH](asi-arch) | ⚡ | ASI-ARCH | 提出ASI-ARCH全自主多智能体系统，通过LLM驱动进化搜索自动发现线性注意力架构，并实证科学 |
| [MASS](mass) | ⚡ | MASS | 提出MASS框架，利用LLM多智能体规模化模拟市场，通过反向优化动态调整分布以构建投资组合。 |
| [One-shot EM](one-shot-em) | ⚡ | One-shot EM | 提出单样本熵最小化(EM)方法，仅用1条无标签数据与10步优化即可重塑LLM输出分布，推理性能媲 |
| [RD-Agent(Q)](rd-agent-q) | ⚡ | RD-Agent(Q) | 提出基于LLM的多智能体框架RD-Agent(Q)，实现量化因子与模型的协同自动化研发与迭代优化 |
| [Agent Trading Arena](agent-trading-arena) | ⚡ | Agent Trading Arena | 提出“代理交易竞技场”零和博弈环境，验证LLM处理视觉图表比纯文本在交易决策上更优，结合反思模块 |
| [HedgeAgents](hedgeagents) | ⚡ | HedgeAgents | 提出HedgeAgents多智能体系统，模拟对冲基金架构，通过LLM专家与经理协同及三类会议机制 |
| [QuantAgent](quantagent) | ⚡ | QuantAgent | 提出QuantAgent双层循环框架，通过LLM自主迭代与实盘反馈自我完善，实现金融Alpha信 |
| [FINCON](fincon) | ⚡ | NeurIPS 2024 | 提出基于LLM的多智能体框架FINCON，通过经理-分析师层级架构与双层风控机制，实现单股交易与 |
| [FinAgent](finagent) | ⚡ | KDD 2024 | 提出FinAgent多模态交易智能体，融合文本与K线图，通过记忆、反思与工具增强模块实现端到端自 |
| [Grounded Event Extraction System](grounded-event-extraction-system) | 🔧 | [arXiv](https://arxiv.org/abs/2607.08346) | 提出两阶段LLM系统，从SEC 8-K文件中抽取细粒度事件标签，并通过引用验证与质量评分实现可审 |
| [SkillOpt](skillopt) | 🔧 | SkillOpt | 将Agent技能文档视为可训练参数，引入类深度学习优化器（学习率、验证集门控、慢更新）实现技能自 |
| [OU-LLM配对交易框架](ou-llm) | 🔧 | JP Morgan | 用OU均值回归模型替代Z-score构建配对交易基线，融合NLP情绪过滤与LLM冲击分类，显著提 |
| [TRR](trr) | 🔧 | TRR | 提出TRR框架，利用LLM模拟头脑风暴、记忆、注意力与推理，基于新闻动态建图预测投资组合次日崩溃 |
| [FinSight](finsight) | 🔧 | FinSight | 提出FinSight多智能体框架，通过代码驱动、迭代视觉增强与两阶段写作，自动化生成高质量多模态 |
| [AlphaAgents](alphaagents) | 🔧 | AlphaAgents | 构建基于LLM的多智能体辩论框架AlphaAgents，通过基本面、情绪与估值智能体协作生成股票 |
| 专业级AI股票分析提示词 | 🔧 | — | 提供结构化LLM提示词模板，自动化生成涵盖基本面、技术面与情绪面的专业股票研报。 |
| [LLMOE](llmoe) | 🔧 | LLMOE | 提出LLMOE框架，用LLM作动态路由器融合量价与新闻多模态数据，动态分配至乐观/悲观专家模型， |
| [TradingAgents](tradingagents) | 🔧 | [arXiv](https://arxiv.org/abs/2412.20138) | 开源TradingAgents多智能体LLM交易框架，模拟投行分工，通过分析师-研究员-交易员- |
| [RAG-MCP](rag-mcp) | 🔧 | RAG-MCP | 提出RAG-MCP框架，通过外部向量索引动态检索最相关工具模式，解决大模型工具选择中的提示膨胀与 |
| [LLMoE](llmoe) | 🔧 | LLMoE | 提出LLMoE框架，用LLM作路由器动态整合量价与新闻数据，路由至情绪专家预测次日走势并生成交易 |
| [Alpha Grail](alpha-grail) | 🔧 | Alpha Grail | 本文提出基于LLM与多智能体的Alpha挖掘框架，通过多模态数据生成、置信度筛选与DNN权重优化 |
| [KAG](kag) | 🔧 | KAG | 介绍KAG知识增强生成框架，结合多模态RAG与LLM Agents，构建支持逻辑推理与跨模态问答 |
| [FinBloom](finbloom) | 🔧 | FinBloom | 提出FinBloom框架，通过微调7B模型构建金融Agent，自动生成结构化请求并结合RAG检索 |
| [利用LLM重新审视金融情感分析](llm-financial-sentiment) | 🔧 | — | 用市场实际涨跌替代人工情绪标签，结合行情指标提示微调LLM，实现高胜率短期趋势预测与交易。 |
| [Logic-RL](logic-rl) | 🔧 | Logic-RL | 基于规则强化学习复现DeepSeek-R1推理能力，用逻辑谜题训练7B模型，验证RL比SFT泛化 |
| scikit-llm | 🔧 | scikit-llm | 介绍scikit-llm开源库，将LLM无缝接入scikit-learn生态，提供零/少样本分类 |
| [FinRLlama](finrllama) | 🔧 | ICAIF24 | 提出RLMF框架，用市场反馈强化学习微调LLaMA，将新闻情感与价格动态结合生成交易信号。 |
| [端到端基于LLM的增强型交易系统](llm-end-to-end-trading) | 🔧 | — | 本文构建端到端交易系统，利用FinGPT实时分析新闻与社媒情绪，结合技术指标生成交易信号，回测验 |
| [PPTAgent](pptagent) | 🔧 | PPTAgent | 中科院提出PPTAgent框架，采用基于编辑的两阶段方法，结合LLM与代码交互实现PPT自动生成 |
| [SusGen-GPT](susgen-gpt) | 🔧 | SusGen-GPT | 提出SusGen-30K数据集与TCFD-Bench基准，基于QLoRA微调7-8B模型并结合R |
| [LlamaLens](llamalens) | 🔧 | LlamaLens | 基于Llama-3.1-8B微调多语言专用LLM LlamaLens，用于新闻与社媒事实核查、情 |
| [FinVision](finvision) | 🔧 | FinVision | 提出FinVision多模态多智能体框架，结合LLM视觉与文本推理，实现可解释的日频股票交易决策 |
| [TRR](trr) | 🔧 | TRR | 提出TRR框架，利用LLM零样本推理构建新闻-股票影响图，结合记忆与注意力机制，实现投资组合崩溃 |
| 基于Conv-LSTM和LLM集成模型用于整体股​票预测 | 🔧 | — | 结合Conv-LSTM处理量价时序与BERT/T5分析新闻情感，构建层次化混合模型提升日频股价预 |
| [从新闻到预测：基于LLM的时间序列预测](llm-news-to-forecast) | 🔧 | — | 利用LLM智能体自动筛选推理新闻，将其作为条件输入微调大模型，实现融合文本事件的时间序列预测。 |
| [FinBERT & LLM Prompting](finbert-llm-prompting) | 🔧 | FinBERT & LLM Prompting | 对比GPT-4o/3.5与FinBERT在金融新闻情绪分析中的表现，验证少样本提示工程的有效性。 |
| [MoA](moa) | 🔧 | MoA | Vanguard提出MoA多智能体RAG框架，用多个小型LLM协同处理金融文档，在低成本下提升回 |
| [ECC Analyzer](ecc-analyzer) | 🔧 | ECC Analyzer | 提出ECC Analyzer框架，结合LLM摘要、RAG细粒度提取与多模态融合，预测股票短期波动 |
| [HybridRAG](hybridrag) | 🔧 | HybridRAG | 提出HybridRAG，融合知识图谱与向量检索增强LLM，显著提升金融财报问答的准确性与召回率。 |
| [如何利用微调LLMs预测股票收益率](llms) | 🔧 | — | 本文微调编码器/解码器LLM，将财务新闻文本直接映射为股票收益预测信号，验证聚合表征优于传统情感 |
| [StockAgent](stockagent) | 🔧 | StockAgent | 提出StockAgent多智能体框架，利用LLM驱动不同性格代理在模拟市场中交易，通过BBS交互 |
| [取代分析师？大模型撰写研究报告的可行性分析](art-34) | 🔧 | — | 评估大模型自动化生成股票研报的可行性，发现78.7%内容可自动提取/生成，仅21.3%需人工判断 |
| [GraphRAG](graphrag) | 🔧 | GraphRAG | 介绍微软GraphRAG原理，结合Ollama本地大模型构建上市公司招股书知识图谱，实现全局关系 |
| [LLMFactor](llmfactor) | 🔧 | LLMFactor | 提出LLMFactor框架，利用顺序知识引导提示(SKGP)从新闻中提取可解释因子，结合历史量价 |
| [FinRobot](finrobot) | 🔧 | FinRobot | FinRobot是开源金融AI代理平台，集成多源LLM与RAG，通过金融思维链与多模态处理实现研 |
| [CopBERT/CopGPT](copbert-copgpt) | 🔧 | CopBERT/CopGPT | 对比BERT与GPT在商品新闻情感分析中的表现，指出GPT预测更强但存幻觉，BERT可解释性更优 |
| 2025Q3最佳AI量化论文 | 📖 | — | 盘点2025Q3 SSRN前沿论文，聚焦高复杂度模型定价、图网络多资产预测、LLM提取分歧因子及 |
| Web3 × AI Agents: 生态、集成与挑战 | 📖 | — | 系统梳理133个Web3 AI代理项目，从生态分布、DeFi应用、治理、安全与信任机制五维度解析 |
| 清华大学&易方达基金 | 综述：量化与大模型的融合 | 📖 | — | 系统梳理金融基础模型(FFM)的三大分支、训练范式、数据集、应用场景及未来挑战。 |
| LLM在金融领域的应用综述 | 📖 | — | 本文系统综述LLM在金融领域的应用，划分为四大框架，梳理RAG/CoT/微调等关键技术，并展望自 |
| MCP | 📖 | MCP | 系统梳理MCP协议架构、生命周期与安全风险，评估其AI工具集成现状，并提出安全治理与生态发展建议 |
| 从深度学习到大模型：一篇文章搞懂量化投资中的AI应用 | 📖 | — | 本文系统梳理了AI在量化阿尔法策略中的演进，重点对比深度学习与LLM在因子挖掘、预测、组合优化及 |
| [Alpha Grail多智能体框架](alpha-grail) | 📖 | Alpha Grail多智能体框架 | 提出基于LLM与多智能体的量化框架，自动挖掘多模态Alpha因子，动态评估市场并优化权重，实现自 |
| 普林斯顿&牛津大学 | 大模型在金融领域的应用、前景和挑战 | 📖 | [arXiv](https://arxiv.org/abs/2406.11903) | 全面综述LLM在金融领域的应用、模型演进、数据集基准及面临的数据、建模与伦理挑战。 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
