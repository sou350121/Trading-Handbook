<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=元学习搜索 alpha=因子挖掘 autonomy=Agent自主演进 -->

# CosmosAlpha 解構（CosmosAlpha）

> **來源發布**：2026-11-03 · （無 venue） · arXiv [2611.04555](https://arxiv.org/abs/2611.04555)
> **arXiv 原文**：[CosmosAlpha: Multi-Relational Graph Search with Pareto-Guided Exploration for LLM-Driven Alpha Mining](https://arxiv.org/abs/2611.04555v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點為量價表格 × 日頻波段 × 元學習搜索 × 因子挖掘 × Agent 自主演進。它補的 prior gap 是：LLM 因子搜索的持久狀態多為生成血緣樹／DAG，跨分支的結構相似與截面暴露相關性沒有被當成可傳播的關係；CosmosAlpha 把已評估因子池做成多關係圖，並用 Pareto 選父代。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `元学习搜索` | `因子挖掘` | `Agent自主演进` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ① 把 LLM 因子挖掘的搜索狀態從生成樹／DAG 升级為 pool-level 多關係圖（generation／structure／exposure correlation）。② Graph Flow 在圖上做固定點傳播，Pareto 以 Quality、Fertility、Diversity 選父代。③ 對「元學習搜索」軸★的意義是：搜索狀態本身成為可複用的關係記憶，跨分支共享結構與暴露證據。④ CSI 300 mean rank 1.43、CSI 500 mean rank 1.29（越低越好；同 stack 的 CosmosAlpha-MCTS 分別為 7.14／4.14）。

**X-Ray.** CosmosAlpha 的貢獻不在發明新因子算子，也不在訓練更強的序列模型，而是把「已評估因子池」提升為一階搜索對象。這在五軸上同時壓中元學習搜索 × 因子挖掘 × Agent 自主演進：LLM 仍是生成器，但搜尋的控制面被外置到圖與 Pareto archive。它解掉的舊工程坑是 generation-centric 搜索的資訊局部性——AlphaPROBE 的 lineage DAG 只有血緣，AlphaSAGE 的 AST encoder 只編碼單一表達式，Alpha Jungle 的 MCTS 只沿樹回傳；CosmosAlpha 讓「不同分支但結構／暴露相近」的因子互為鄰居。預測它打不開的 envelope：第一，結構邊與相關邊都是每輪在當前節點上重建的快照，regime 切換後暴露相關性會漂移，若只靠歷史評估日曆，Graph Flow 會把舊鄰居的效用錯誤擴散；第二，Pearson 相關只捕捉線性同期暴露，對非線性／尾部互補的冗餘不敏感；第三，Top-30 稀疏化與固定閾值 0.20／0.10 是工程先驗，對不同 universe 的因子密度未必可移植。對量化讀者：這篇值得借的是「把搜索預算花在父代選擇」的框架，而不是某個因子公式；復現時必須把 LLM 成本、資料快照與回測成本分開審計。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（代表） | CosmosAlpha | 差異的實質 |
|---|---|---|---|
| 搜索狀態 | AlphaPROBE：lineage DAG；Alpha Jungle：MCTS 樹；QuantaAlpha：trajectory | pool-level 多關係圖 \(G=(V,E^{gen},E^{str},E^{corr})\) | 把「血緣」擴充為三種並存關係，跨分支可共享 |
| 父代評估 | 單節點分數，或沿樹／血緣回傳 | Graph Flow 固定點傳播 \(F^{(k+1)}=(1-\gamma)R+\gamma PF^{(k)}\) | 鄰居證據進入每個節點，\(\gamma=0\) 成自然消融 |
| 父代選擇 | 單一標量，或品質優先 | Pareto fronts on \((Q,F,D)\) + crowding distance | 保留 quality／fertility／diversity 的 trade-off |
| 生成 | 通用 prompt 或樹上擴展 | parent-grounded mutation，單親、至多 2 個子代 | 限制漂移；作者明說不做 crossover |

### 1.2 ⚡ Eureka

**一句話 trick**：把「已評估因子池」當成一張三關係圖來做 personalized-PageRank 式傳播，再把傳播後的三維效用丢進 NSGA-II 的 non-dominated sorting 選父代——搜索狀態本身變成可複用的關係記憶。

**直覺**：血緣只說「誰生誰」；結構 Jaccard 與暴露 \(|Pearson|\) 說「不同分支其實是同一家族」。圖傳播讓一個分支的品質／繁殖力／多樣性證據流到鄰居，Pareto 則避免把 fertility 與 diversity 壓成一個分數。

### 1.3 信息流

    LLM hypothesis / factor design
              │
              ▼
    parent-grounded mutation（單親，至多 2 個子代）
              │
              ▼
    validate → dedup → factor calc → evaluate（僅訓練段回饋）
              │
              ▼
    admit valid unique nodes ──► 更新三關係圖：
                                  E_gen：直接 parent–child（對稱化、單位權重）
                                  E_str：canonical subexpression Jaccard ≥ 0.20，Top-30
                                  E_corr：|mean cross-sectional Pearson| ≥ 0.10，Top-30
              │
              ▼
    Graph Flow：F ← (1-γ)R + γPF，P = (P_gen+P_str+P_corr)/3，γ=0.75
              │
              ▼
    Pareto fronts on F=(Q,F,D) + crowding → 每輪至多 4 個父代
              │
              └──────────────► 下一輪（預算 = expansion attempts）

## §2 · 數學層

📌 **Napkin Formula**

\[
P=\frac{1}{3}\left(P^{\mathrm{gen}}+P^{\mathrm{str}}+P^{\mathrm{corr}}\right),\qquad
F^{(k+1)}=(1-\gamma)R+\gamma PF^{(k)},\quad F^{(0)}=R
\]

\[
\mathbf{R}_i=(Q_i^{\mathrm{self}},Q_i^{\mathrm{fert}},Q_i^{\mathrm{div}}),\qquad
Q_i^{\mathrm{fert}}=\operatorname{Top3Mean}(y_{i,1},y_{i,2},\ldots)
\]

**直覺**：\(R\) 是節點自己的 Quality／Fertility／Diversity；\(P\) 把三種關係各自列歸一化後等權混合，避免較密的相似度圖壓過較稀疏的血緣圖。Graph Flow 是固定點迭代：第一項保留節點自身證據，第二項把鄰居證據擴散進來。\(\gamma=0\) 就退回無傳播的 ablation。

**複雜度／訓練細節**：Graph Flow 是推理階段的稀疏固定點迭代，不訓練參數；實作上每節點結構／相關鄰居至多 Top-30（union 對稱化後度可超過 30），\(\gamma=0.75\)，tol \(10^{-8}\)，最多 100 iterations。整體複雜度所讀文本未披露。LLM 是黑箱生成器，沒有可微 loss；搜索回饋來自 Quality archive 的 \(\Delta HV\)、RankIC 統計與圖上的多目標排序。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下所有數字均為**假設／示意**，不是論文實證結果；論文真實數字留在 §5。

1. **假設**三個節點 A／B／C，原始效用 \(R_A=(0.9,0.2,0.3)\)、\(R_B=(0.6,0.8,0.1)\)、\(R_C=(0.3,0.1,0.9)\)，三維分別代表 Quality／Fertility／Diversity。
2. **假設**行隨機轉移矩陣 \(P\) 為：A→[0.5,0.5,0]；B→[0.5,0,0.5]；C→[0,0.5,0.5]。
3. 取 \(\gamma=0.75\)，先算 \(PR\)：A 列 \(=0.5R_A+0.5R_B=(0.75,0.5,0.2)\)；B 列 \(=0.5R_A+0.5R_C=(0.6,0.15,0.6)\)；C 列 \(=0.5R_B+0.5R_C=(0.45,0.45,0.5)\)。
4. 再算 \(F^{(1)}=0.25R+0.75PR\)：A \(=(0.7875,0.425,0.225)\)；B \(=(0.6,0.3125,0.475)\)；C \(=(0.4125,0.3625,0.6)\)。
5. **Pareto 比較**：A 的 Quality 最高但 Diversity 最低；B 的 Fertility 最低，Quality 與 Diversity 居中；C 的 Diversity 最高但 Quality 最低。三者互不支配，同在前沿；此時用 crowding distance 在 front 內選父代（示意：選 A 與 C，保留「高品質」與「高多樣」兩條路）。
6. **讀法**：傳播把節點拉向鄰居平均，會稀釋單點極值；Pareto 的作用是不要把稀釋後的三維壓成一個 scalar，在本例中，B 的 Fertility 雖最低，仍因 Quality 與 Diversity 的取捨留在非支配前沿。若改用單一分數，B 是否保留取決於權重。

## §3 · 數據層

| 項目 | 設定 |
|---|---|
| Universes | CSI 300 與 CSI 500，日頻 A 股，post-adjusted prices |
| Mining／training | 2010-04-01–2019-05-31；搜索回饋與 LightGBM 擬合 |
| Validation | 2019-06-01–2020-12-31；僅用於 LightGBM early stopping |
| Test | 2021-01-01–2025-12-31；held-out，僅算最終預測與組合結果 |
| Label | one-trading-day forward close-to-close return |
| 股票資格 | at least one year since listing；blacklist 與 tradability masks；financial stocks retained |
| 截面處理 | MAD winsorization multiplier 5、neutralization、Z-score；neutralization exposures 含 industry、log market capitalization、20-day return、turnover、volatility |
| 因子評估門檻 | 每日至少 100 檔 finite eligible stocks 且 80% coverage；節點至少 2 個 valid dates、覆蓋 80% candidate evaluation dates |
| 搜索空間 | 日頻；rolling windows \(\{5,10,20\}\)；expression depth 上限 15；最多 512 characters、32 call nodes、6 distinct fields |
| 因子池目標 | approximately 150 evaluable factors（各自動化方法按輸出因子數對齊，非等算力） |
| 回測 | Qlib TopK-Drop：hold 50 stocks、replace up to 5 daily、open price、price-limit 0.095、trade unit 100 shares；initial account \(10^8\)、invested fraction 0.95 |
| 成本／容量 | 所讀文本未見交易成本、滑點、容量假設；不可把 ARR／IR 當淨收益 |

樣本外設計是時間切分，不是隨機切分；作者稱 mining 與 search feedback 只用 training，validation 只做 early stopping，test 直到因子池與模型配置固定後才使用。這個設計降低了直接的 test leakage，但不排除 LLM 预训练语料见过因子文献或市場模式（本頁推斷，待核）。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | https://github.com/wzhmtldswz/CosmosAlpha |
| Checkpoint | TBD（所讀文本未見模型權重連結） |
| License | 所讀文本未見 |
| 複現難度 | 中高：需 Python ≥3.12、LightGBM ≥4.6,<5、Qlib ≥0.9.7,<0.10；LLM 後端與 prompt manifest 需對齊；point-in-time 成分與資料快照需自備 |
| 數據可得性 | A 股日頻；公開下載與授權條件所讀文本未披露 |
| LLM 成本 | C.4 披露 100 Graph expansion attempts 的平均累計調用時間與 token 用量（DeepSeek-V4-Flash／Kimi K3／GPT-5.6-Sol 各有記錄）；這是 LLM 調用成本，不是交易成本 |

## §5 · 評測 / Benchmark

> **前SOTA 欄的語義**：逐行放「該 Metric 上數值最好的單一非本法基線」之名稱與逐字數值；不是把多個基線 min/max 成一個 SOTA 區間。Δ 固定為「本方法 − 前SOTA」，只放在末欄；Mean rank 越低越好，MDD 為 signed、越接近 0 越好。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| CSI 300 | IC | Alpha158 0.0320 | 0.0324 | 0.0004 |
| CSI 300 | ICIR | AlphaPROBE 0.3536 | 0.3441 | -0.0095 |
| CSI 300 | RankIC | Alpha158 0.0479 | 0.0494 | 0.0015 |
| CSI 300 | RankICIR | Alpha158 0.5668 | 0.5670 | 0.0002 |
| CSI 300 | ARR (%) | QuantaAlpha 7.22 | 8.44 | +1.22pp |
| CSI 300 | IR | AlphaAgent 0.4393 | 0.5332 | 0.0939 |
| CSI 300 | MDD (%) | R&D-Agent -24.87 | -25.07 | -0.20pp |
| CSI 300 | Mean rank ↓ | Alpha158／AlphaForge 3.86 | 1.43 | -2.43 |
| CSI 500 | IC | CosmosAlpha-MCTS 0.0356 | 0.0360 | 0.0004 |
| CSI 500 | ICIR | Alpha158 0.3927 | 0.3997 | 0.0070 |
| CSI 500 | RankIC | Alpha158 0.0618 | 0.0627 | 0.0009 |
| CSI 500 | RankICIR | Alpha158 0.7898 | 0.7562 | -0.0336 |
| CSI 500 | ARR (%) | AlphaAgent 12.76 | 13.01 | +0.25pp |
| CSI 500 | IR | AlphaAgent 0.6798 | 0.6689 | -0.0109 |
| CSI 500 | MDD (%) | AlphaAgent -26.51 | -26.42 | +0.09pp |
| CSI 500 | Mean rank ↓ | CosmosAlpha-MCTS 4.14 | 1.29 | -2.85 |

**哪部分 Δ 是真 capability**：最可歸因的是 CosmosAlpha-Graph 對 CosmosAlpha-MCTS 的控制比較——兩者共享 factor language、evaluator、LLM backbone 與下游 LightGBM／portfolio protocol，因此圖搜索組織的差異被隔離得較好。主表 CSI 300 mean rank 1.43 對 7.14、CSI 500 1.29 對 4.14，方向一致；Appendix J 的三運行均值（Graph IC 0.0326、ARR 欄 8.45、MDD 欄 -25.10；MCTS IC 0.0285、ARR 欄 5.71、MDD 欄 -31.23）也同向。但主表是單次主實驗，Appendix J 是三運行聚合，兩者不可互相代稱。

**哪部分不是碾壓**：預測層的增量很窄。CSI 300 的 IC 是 Alpha158 0.0320 對本法 0.0324，RankICIR 是 0.5668 對 0.5670；ICIR 輸給 AlphaPROBE 0.3536 對 0.3441；MDD 輸給 R&D-Agent -24.87 對 -25.07。CSI 500 的 RankICIR 是 Alpha158 0.7898 對本法 0.7562，IR 是 AlphaAgent 0.6798 對 0.6689。真正的差距集中在 ARR／IR 與 mean rank，這提示組合層的因子池多樣性與下游 LightGBM 互動可能是放大器，也可能是特定回測配置的產物。

**過擬合／前視／成本**：作者用 point-in-time 成分與交易資格、held-out test，且 mining 只用 training；但所讀文本未見交易成本、滑點與容量假設，ARR／IR 是零基準絕對收益。MDD 是 signed arithmetic cumulative return 的深度，不是回撤持續期；持續期未披露。跨基線比較時，各基線保留自己的搜索過程，預算按輸出因子數 approximately 150 對齊，不是等 LLM token／算力，因此不能把全部差異歸因於圖結構。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **市場範圍**：只評估 CSI 300 與 CSI 500；Conclusion 明說 future work 才擴展到 additional markets、high-frequency 與 fundamental factors。
- **AlphaJungle 未直接重現**：§4.1／C.7 明說提交時未找到 AlphaJungle 官方公開實現，因此以 CosmosAlpha-MCTS 作 transparent MCTS-based proxy；不能把 CosmosAlpha-MCTS 等同 AlphaJungle。
- **離線回測警語**：Ethics statement 明說這是 offline research evaluation，backtest 不應解讀為 financial advice 或保證 live-trading performance；automated factor discovery 若無獨立驗證與風險控制，可能鼓勵 overfitting 或 unsafe deployment。
- **單親 mutation**：A.6 明說目前是 single-parent mutation，不做兩父代 crossover。
- **預算對齊方式**：C.2 明說各自動化方法按「approximately 150 evaluable factors」的輸出目標對齊，而不是把 MCTS iteration、Chain step、Graph expansion attempt 視為等價算力。
- **LLM 成本已披露但非交易成本**：C.4 給出 100 Graph expansion attempts 的平均累計調用時間與 token 用量；這是運算成本，不是市場衝擊成本。

### 6.2 推斷的隱含假設（本頁推斷，待核）

- **Regime 依賴**：structure／correlation 邊在當前 admitted nodes 上重建，correlation 用 evaluation calendar 的每日截面相關平均；若市場風格切換，\(|C_{ij}|\) 的鄰居結構會漂移，Graph Flow 可能把舊鄰居的效用錯誤擴散。這是推斷，原文未做 regime 分層檢驗。
- **容量與成本**：TopK-Drop 持有 50 檔、每日最多換 5 檔、100 股整手、開盤價成交；容量與滑點未披露，高換手策略的淨 ARR／IR 可能低於表列。不能說「成本超過某閾值就失效」，因為所讀文本沒有該閾值。
- **相關性度量的邊界**：Pearson 只捕捉線性同期暴露；絕對值把正負相關都當冗餘，對「方向相反但可對沖」的因子會過度懲罰。此為方法語義推斷。
- **圖閾值的可移植性**：structure ≥ 0.20、correlation ≥ 0.10、Top-30 是固定先驗；對因子密度更高或更低的 universe，可能需要重標定。這是推斷，原文只給 CSI 300／500 結果。
- **數據泄漏與 survivorship**：作者用 historical constituent membership 與 trading-eligibility masks point-in-time，緩解了成分與資格的前視；Table 14 列明使用時點對齊的產業分類（point-in-time industry classification）。退市股票處理與停牌填充仍待核。輔助數據缺失本身不是 survivorship 偏差，兩者不可混為一談。
- **LLM 预训练知識泄漏**：LLM 可能見過因子文獻或市場模式，這不是傳統數據泄漏，但可能讓「生成」看起來比實際更有效；本頁推斷，原文未做去知識檢驗。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| AlphaPROBE | lineage DAG；retrieval + graph-biased evolution；無 structure／exposure 多關係同時傳播 | Yes（C.7 有 repo） | arXiv 2026 |
| Alpha Jungle | MCTS 樹搜索；無 pool-level 多關係圖；作者稱無官方公開實現 | No（C.7） | AAAI 2026 |
| AlphaSAGE | AST structure-aware encoder + GFlowNet；編碼單一表達式，非完整因子池的多關係 | Yes（C.7 有 repo） | ICLR 2026 |
| AlphaAgent | AST-level originality 正則；multi-agent workflow | Yes（C.7 有 repo） | KDD 2025 |
| QuantaAlpha | trajectory-level evolution；非圖狀態 | Yes（C.7 有 repo） | arXiv 2026 |
| CosmosAlpha-MCTS | 同 stack 的 lineage MCTS 控制組；用來隔離圖搜索效果 | 隨論文 repo | 本文 baseline |
| Alpha158 | 手工因子庫，非搜索 | Yes（Qlib） | 2020 |
| GP | 遺傳規劃，非 LLM | Yes（gplearn） | 1992 |

🎤 **Interview Tip**

- **正確答**：CosmosAlpha 的 novelty 是 search-state representation——把 evaluated factor pool 做成三關係圖；Graph Flow 是傳播機制，Pareto 是父代選擇機制；最強證據是 Graph 對同 stack MCTS 的控制比較，且有三個 LLM backbone 與三個獨立 run 的穩定性分析。預測層增量窄，主要優勢在組合層與 mean rank。
- **錯答**：說它發明了新 alpha 公式；說 Graph Flow 是訓練一個 GNN；說下游 LightGBM 不同所以不公平（其實共享）；說 ARR 提升全部來自圖結構（跨基線預算非等算力，且成本未計）；把 CosmosAlpha-MCTS 說成 AlphaJungle 本人。

### 7.1 可證偽預測（帶日期）

以 **2026-11-03** 這個來源版本為觀察起點：

1. 若作者後續把同一協議搬到第三個 A 股 universe，且 factor language／evaluator／LLM／下游 protocol 不變，則 Graph 對同 stack MCTS 的 mean rank 優勢應重現；若反轉或消失，則「圖結構普遍有效」的 claim 受限，較可能是 CSI 300／500 特定。此預測待驗證。
2. 若把 correlation 邊改為 rolling／regime-aware，且其他項固定，則在風格切換段的 MDD 應改善；若無改善，則靜態相關邊不是主要瓶頸。此預測待驗證。

## §8 · For the Reader

- **因子研究員**：借它的 search-state 設計，不要先借因子公式。復現時先做 Graph vs MCTS 的同 stack 對照，固定 LLM、evaluator、prompt 與預算；再檢查 Alpha158 在 IC／RankICIR 已接近本法，別把組合層差距誤讀成預測層突破。
- **高頻執行**：這是日頻、開盤價成交、TopK-Drop 持有 50 檔、每日最多換 5 檔的框架；成本、滑點、容量未披露。上線前先做成本敏感度（成本數字 TBD），不要把表列 ARR／IR 當淨收益。
- **組合配置**：ARR／IR／MDD 是零基準絕對收益；MDD 是深度不是持續期。把「因子池多樣性」與「換手／交易成本」分開評估，否則 diversity 帶來的好處可能被換手吃掉。
- **LLM-agent 工程**：parent-grounded mutation、closed JSON contract、duplicate 與 validation pipeline 是真正的工程重點；Graph Flow 是非學習式的圖上傳播。原文未比較它和 GNN 的推理成本。C.4 另有列出 LLM token 用量。若要做 production，先做 prompt／model 的版本鎖定。
- **RL 策略**：這不是 RL。Fertility 的 \(\Delta HV\) 是 archive-based reward，不是 policy gradient；Graph Flow 也不是 value propagation。別把「繁殖力」誤讀成 exploration bonus。
- **研究學生**：Ablation 矩陣清楚（\(\gamma=0\)、quality scalarization、flow scalarization、去 F／D、去 G／S／C），但 quality scalarization 同時改 archive、reference set 與 Fertility，不能把該列的差異全部歸因於單一機制；這是原文自己明說的複合改動。

## References

- Wang, Z.-H., Lang, Z.-Y., Li, Y.-Y., Lan, Q.-Y., Piao, Y.-S. (2026). *CosmosAlpha: Multi-Relational Graph Search with Pareto-Guided Exploration for LLM-Driven Alpha Mining.* arXiv:2611.04555. https://arxiv.org/abs/2611.04555
- Repo: https://github.com/wzhmtldswz/CosmosAlpha
- Guo et al. (2026). *AlphaPROBE: Alpha Mining via Principled Retrieval and On-Graph Biased Evolution.* arXiv:2602.11917. https://arxiv.org/abs/2602.11917
- Shi et al. (2026). *Navigating the Alpha Jungle: An LLM-powered MCTS Framework for Formulaic Alpha Factor Mining.* AAAI 2026.
- Chen et al. (2026). *AlphaSAGE: Structure-aware Alpha Mining via GFlowNets for Robust Exploration.* ICLR 2026. https://openreview.net/forum?id=zRKF4ln2VE
- Tang et al. (2025). *AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay.* KDD 2025.
- Han et al. (2026). *QuantaAlpha: An Evolutionary Framework for LLM-Driven Alpha Mining.* arXiv:2602.07085. https://arxiv.org/abs/2602.07085
- Yu et al. (2023). *Generating Synergistic Formulaic Alpha Collections via Reinforcement Learning.* KDD 2023.
- Shi et al. (2025). *AlphaForge: A Framework to Mine and Dynamically Combine Formulaic Alpha Factors.* AAAI 2025.
- Yang et al. (2020). *Qlib: An AI-oriented Quantitative Investment Platform.* arXiv:2009.11189. https://arxiv.org/abs/2009.11189
- Koza (1992). *Genetic Programming: On the Programming of Computers by Means of Natural Selection.* MIT Press.
- Deb et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation.
- Klicpera et al. (2019). *Predict then Propagate: Graph Neural Networks Meet Personalized PageRank.* ICLR 2019.
- Zhou et al. (2003). *Learning with Local and Global Consistency.* NeurIPS 2003.