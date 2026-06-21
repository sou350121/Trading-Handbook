# 时序预测基础

> **收錄**：面向价格/收益率/波动率的序列建模，涵盖Transformer变体、RNN/LSTM、MLP/Mixer、状态空间模型及频率自适应架构
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：18 篇（16 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| Timer | ⚡ | Timer | 提出Timer生成式预训练Transformer，通过统一数据集与单序列格式实现时间序列少样本预 |
| Time-MoE | ⚡ | Time-MoE | 提出基于稀疏MoE的解码器Transformer基础模型，在3000亿时间点数据集上预训练，实现 |
| TEMPO | ⚡ | ICLR 2024 | 提出TEMPO框架，利用趋势-季节性分解与半软提示词，基于GPT实现零样本时间序列预测，在多模态 |
| aLLM4TS | ⚡ | ICML24 | 提出aLLM4TS框架，通过自监督多补丁预测与两阶段训练，将LLM适配至时间序列表示学习，显著提 |
| （当前）为何你不需要深度学习来进行时间序列预测，以及你需要什 | 🔧 | — | 基于M5/Kaggle竞赛经验，论证当前时序预测应优先使用GBM与特征工程，而非深度学习，强调交 |
| xLSTM-Mixer | 🔧 | xLSTM-Mixer | 提出xLSTM-Mixer，结合线性预测与sLSTM块，通过时间/变量/多视图混合实现高精度长期 |
| MultiRC | 🔧 | MultiRC | 提出MultiRC模型，通过多尺度重构与对比学习捕捉前兆信号，实现无需异常标注的时间序列异常预测 |
| FAN | 🔧 | FAN | 提出频率自适应归一化(FAN)，通过傅里叶变换提取主导频率成分并用MLP预测其变化，有效缓解非平 |
| ARMA Attention | 🔧 | ARMA Attention | 提出ARMA注意力机制，将移动平均项融入线性注意力，在保持O(N)复杂度下提升时间序列预测的长短 |
| MANA-Net | 🔧 | CIKM 2024 | 提出MANA-Net，通过动态市场-新闻注意力机制加权聚合情感，解决均质化问题以提升日频市场预测 |
| ADD | 🔧 | ADD | 提出ADD框架，通过解耦市场/超额收益特征、动态自蒸馏去噪及特征混合增强，提升A股趋势预测与回测 |
| Fieldy | 🔧 | Fieldy | 提出Fieldy模型，通过两阶段层次Transformer并行编码行列字段，捕获表格时序数据的细 |
| SimStock | 🔧 | SimStock | 提出SimStock框架，结合自监督学习与时间域泛化，学习股票时序稳健表示，用于相似股挖掘、配对 |
| Stockformer | 🔧 | [arXiv](https://arxiv.org/abs/2401.06139) | 提出Stockformer模型，结合小波变换高低频分离与多任务自注意力网络，实现日频量价因子的收 |
| StockFormer | 🔧 | StockFormer | 提出StockFormer模型，结合小波变换高低频分离与多任务自注意力网络，实现沪深300量价因 |
| TIMEVIEW | 🔧 | ICLR 2024 | 提出双层透明度框架与TIMEVIEW模型，用B样条与静态特征实现可解释的时间序列预测，兼顾性能与 |
| DoubleAdapt | 🔧 | KDD [arXiv](https://arxiv.org/abs/2306.09862) | 提出DoubleAdapt元学习框架，通过数据与模型双适配器缓解分布偏移，实现股票趋势增量学习预 |
| 综述：深度学习模型在加密货币市场的应用与验证 | 📖 | — | 系统综述深度学习在加密货币价格预测中的应用，对比ARIMA、LSTM、CNN及Transform |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
