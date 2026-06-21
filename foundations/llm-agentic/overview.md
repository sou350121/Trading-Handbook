# 大模型与智能体

> **收錄**：LLM在因子挖掘、情绪分析、研报生成、RAG增强及多智能体交易工作流中的适配、微调与推理架构
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：13 篇（11 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| FinAgent | ⚡ | KDD 2024 | 提出FinAgent多模态交易智能体，融合文本与K线图，通过记忆、反思与工具增强模块实现端到端自 |
| [TRR](trr) | 🔧 | TRR | 提出TRR框架，利用LLM零样本推理构建新闻-股票影响图，结合记忆与注意力机制，实现投资组合崩溃 |
| 基于Conv-LSTM和LLM集成模型用于整体股​票预测 | 🔧 | — | 结合Conv-LSTM处理量价时序与BERT/T5分析新闻情感，构建层次化混合模型提升日频股价预 |
| [从新闻到预测：基于LLM的时间序列预测](llm) | 🔧 | — | 利用LLM智能体自动筛选推理新闻，将其作为条件输入微调大模型，实现融合文本事件的时间序列预测。 |
| [FinBERT & LLM Prompting](finbert-llm-prompting) | 🔧 | FinBERT & LLM Prompting | 对比GPT-4o/3.5与FinBERT在金融新闻情绪分析中的表现，验证少样本提示工程的有效性。 |
| [ECC Analyzer](ecc-analyzer) | 🔧 | ECC Analyzer | 提出ECC Analyzer框架，结合LLM摘要、RAG细粒度提取与多模态融合，预测股票短期波动 |
| 如何利用微调LLMs预测股票收益率 | 🔧 | — | 本文微调编码器/解码器LLM，将财务新闻文本直接映射为股票收益预测信号，验证聚合表征优于传统情感 |
| StockAgent | 🔧 | StockAgent | 提出StockAgent多智能体框架，利用LLM驱动不同性格代理在模拟市场中交易，通过BBS交互 |
| 取代分析师？大模型撰写研究报告的可行性分析 | 🔧 | — | 评估大模型自动化生成股票研报的可行性，发现78.7%内容可自动提取/生成，仅21.3%需人工判断 |
| GraphRAG | 🔧 | GraphRAG | 介绍微软GraphRAG原理，结合Ollama本地大模型构建上市公司招股书知识图谱，实现全局关系 |
| LLMFactor | 🔧 | LLMFactor | 提出LLMFactor框架，利用顺序知识引导提示(SKGP)从新闻中提取可解释因子，结合历史量价 |
| [FinRobot](finrobot) | 🔧 | FinRobot | FinRobot是开源金融AI代理平台，集成多源LLM与RAG，通过金融思维链与多模态处理实现研 |
| 普林斯顿&牛津大学 | 大模型在金融领域的应用、前景和挑战 | 📖 | [arXiv](https://arxiv.org/abs/2406.11903) | 全面综述LLM在金融领域的应用、模型演进、数据集基准及面临的数据、建模与伦理挑战。 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
