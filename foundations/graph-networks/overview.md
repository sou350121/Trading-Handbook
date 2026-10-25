# 图网络与关系建模

> **收錄**：利用股票间相关性、产业链、分析师覆盖或动态共现关系构建异构图/动态图，进行趋势预测、风格挖掘与组合构建
> **五軸**：数据模态 · 时间尺度 · 学习范式 · Alpha生成机制 · 人机协作度
> **本 zone 現有**：27 篇（26 篇成解構頁，其餘為 registry）

## 解構清單

| 方法 | 評級 | 來源 | TL;DR |
|---|---|---|---|
| [FNIRVAR](fnirvar) | ⚡ | FNIRVAR | 提出FNIRVAR模型，结合静态因子与SBM网络VAR，通过谱聚类挖掘高维收益中的隐含群组结构， |
| [NAVIS](navis) | 🔧 | [arXiv](https://arxiv.org/abs/2607.12067) | 基于SEC 13F构建动态二分图，提出NAVIS模型预测机构持仓权重，建立时序图学习基准。 |
| [DGNSDE](dgnsde) | 🔧 | WWW26 | 融合DTW时滞图与几何SDE，在连续时间轴上显式建模股票领先-滞后效应与随机波动，提升截面选股收 |
| [FinD³](find) | 🔧 | CIKM 2025 | 提出FinD³框架，结合双三次状态空间与演化超图，高效建模3D量价时序与动态股票关联以挖掘Alp |
| [GraFiN-Gen](grafin-gen) | 🔧 | GraFiN-Gen | 提出基于图的概率集成框架，用LASSO学习稀疏跨资产权重，混合生成器分布以提升多资产预测夏普比率 |
| [COGRASP](cograsp) | 🔧 | COGRASP | 提出COGRASP模型，利用社交媒体共现构建动态股票图，结合多尺度ALSTM预测日频股价涨跌。 |
| [DINS](dins) | 🔧 | DINS | 针对Meme股社交网络提出领域感知负采样策略DINS，结合多维采样与正样本增强，显著提升动态图嵌 |
| [DeltaLag](deltalag) | 🔧 | DeltaLag | 提出DeltaLag框架，用稀疏交叉注意力动态识别领涨股及最优滞后期，结合原始价量特征与排序损失 |
| [FRI框架](fri) | 🔧 | FRI框架 | 提出FRI框架与SPNews数据集，实现独立于下游任务的动态金融图谱质量评估与解释性验证。 |
| [NGAT](ngat) | 🔧 | NGAT | 提出节点级图注意力网络NGAT，联合预测股票长期收益率趋势与波动率，解决图模型异构性与短视预测问 |
| [DGRCL](dgrcl) | 🔧 | DGRCL | 提出DGRCL框架，融合DTW动态建图与静态关系约束的对比学习，显著提升股票日频涨跌预测精度。 |
| [PDU](pdu) | 🔧 | KDD 2025 | 提出PDU模型，通过渐进式学习时序、关系与多期依赖，并结合不确定性对比机制，实现稳健的股票排序与 |
| [FRI](fri) | 🔧 | FRI | 提出独立于下游任务的FRI评估框架与SPNews数据集，通过新闻共现构建动态图谱，用可解释指标验 |
| [Multi-GCGRU](multi-gcgru) | 🔧 | Multi-GCGRU | 提出Multi-GCGRU框架，融合持股/行业/主题等多图结构与GRU，捕捉股票交叉影响与时间依 |
| [ML-GAT](ml-gat) | 🔧 | ML-GAT | 提出ML-GAT模型，融合LSTM价格、BERT新闻与Wikidata关系，通过双层图注意力机制 |
| [GAT-CNNpred](gat-cnnpred) | 🔧 | GAT-CNNpred | 提出结合图神经网络与CNN的混合架构，通过特征相关性建图提取每日与时间维度特征，实现股票指数短期 |
| [如何利用加密货币价格相关性网络优化投资组合？](art-214) | 🔧 | — | 结合价格预测与共识聚类网络，从相关簇中选股并最大化夏普比率，优化加密货币投资组合。 |
| [GALSTM](galstm) | 🔧 | GALSTM | 提出GALSTM模型，结合霍克斯过程与图注意力LSTM挖掘股票相关性，实现A股高频日内交易与动态 |
| [DishFT-GNN](dishft-gnn) | 🔧 | DishFT-GNN | 提出基于蒸馏的未来感知GNN框架，通过教师模型挖掘历史-未来分布关联，指导学生模型提升股票趋势预 |
| [GATanalysts](gatanalysts) | 🔧 | GATanalysts | 利用图注意力网络挖掘分析师共同覆盖网络中的动量溢出效应，构建端到端选股策略，实现29.44%年化 |
| [KGTransformer](kgtransformer) | 🔧 | KGTransformer | 结合微调LLM与动态知识图谱，构建FinDKG数据集与KGTransformer模型，实现金融趋 |
| [FinGAT](fingat) | 🔧 | FinGAT | 提出FinGAT模型，结合GRU与图注意力网络自动学习股票及行业间潜在关系，通过多任务学习实现日 |
| [RSR](rsr) | 🔧 | RSR | 提出RSR框架，结合LSTM与时间图卷积(TGC)建模股票时序与关系，将预测转化为排序任务以提升 |
| [THGNN](thgnn) | 🔧 | CIKM22 | 提出THGNN模型，基于历史价格构建时态异构图，结合Transformer与图注意力网络预测股票 |
| [HATS](hats) | 🔧 | HATS | 提出HATS分层图注意力网络，通过状态与关系双层注意力聚合多源公司关联信息，提升个股与指数涨跌预 |
| [MGDPR](mgdpr) | 🔧 | MGDPR | 提出MGDPR模型，利用信息熵与信号能量动态构建多关系股票图，结合图扩散与并行保留机制捕捉动态关 |
| Graph Transformer | 📖 | Graph Transformer | 本文系统综述图Transformer的设计视角、分类体系、多场景应用及未来挑战，为图结构数据建模 |

> ⚡/🔧 page-worthy 寫成整頁解構（點方法名進入）；📖 邊際/綜述只留 registry 一行。
