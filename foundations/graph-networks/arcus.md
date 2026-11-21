<!-- ontology-5axis data=图关系 horizon=中长周期 paradigm=因果结构 alpha=风险择时 autonomy=人机协同可解释 -->

# ARCUS 解構（ARCUS）

> **來源發布**：2026-11-19 · （無 venue） · arXiv [2611.25860](https://arxiv.org/abs/2611.25860)
> **arXiv 原文**：[Neckpinches in a Positively Curved Market: Ricci Flow with Surgery on the Nikkei-225](https://arxiv.org/abs/2611.25860v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 Nikkei-225 相關圖放在「圖關係 × 中長周期 × 幾何結構分解 × 風險分組 × 人機可解釋」的位置；補上正曲率相關圖上 Ricci flow 手術的遞歸化與自適應停流，免剪枝、免預設社區數。五軸標籤中的「因果結構」在此應讀作資料驅動的層級結構分解，而非 causal identification。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `图关系` | `中长周期` | `因果结构` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ①把 226-node Nikkei-225 相關圖視為 ORC-spherical，用離散 Ricci flow 讓曲率分布極化；②以曲率缺口比 G、3-iteration 穩定窗與最小迭代數自適應停流，再做絕對曲率門檻手術並遞歸；③對「圖關係／中長周期」軸的意義是：不必先剪 MST/PMFG，也不必指定社區數；④關鍵實證：20 iterations 後 7,728 條低曲率邊被切，得 184/42 兩部；ARCUS 以 G_min=10、N_min=15 執行 28 surgeries、13 levels、29 terminal clusters，但來源未給任何 IR/Sharpe/AR/MDD 績效。

**X-Ray.** ARCUS 的價值不在預測，而在把「相關矩陣太密、社群偵測需先剪枝」這個舊工程坑，改成在全連接圖上讓曲率自己長出對比：ORC-spherical 的 Nikkei-225 初始每邊正曲率，flow 20 iterations 後才出現 -0.146 到 0.014 的低曲率族群與 0.497 到 0.512 的高曲率 bulk，7,728 條邊被切。這使它對「預設社區數」不敏感，但代價是結果高度依賴 weight scaling、G_min 與固定 universe；低層 8 個節點 G<15，21 of 28 splits 只剝離小於 N_min 的小部，樹深且不平衡。對量化讀者，它是風險分組／regime 監測的可解釋前處理，不是 alpha 或擇時模型；沒有樣本外、成本、容量與 official sector 對照，envelope 限於日頻橫截面結構描述。

## §1 · 架構 / Core Mechanism

### 三大改動 vs 前作

| 改動 | 前作／固定排程 | 本文 ARCUS | 讀者影響 |
|---|---|---|---|
| 停止規則 | 固定 20-iteration Ricci flow，之後用 Eq. (5) 手選閾值 | 每節點逐迭代檢查：至少 5 iterations、G>G_min、lower population 連續 3 iterations 不變；hard cap 100 | 停止點由資料決定，不再由使用者排程 |
| 切割準則 | κ_ij^(20) < 0.2 w_ij^(20)，factor 0.2 由 inspection 選定 | 移除 κ≤κ_lo，即最大曲率缺口的下界；是絕對曲率門檻 | 切割不依賴 flowed weight 的相對比例 |
| 遞歸 | 單次手術，得 184/42 | 對 N≥N_min 的 part 遞歸；part B 定義為非最大連通分量的聯集 | 產生 13 levels、29 terminal clusters；binary split 是實作而非測得社區數 |
| 參數 | Ref. [1] NASDAQ-100 用 G_min=15 | 本文用 G_min=10、N_min=15、α=1/2 | 低層結果 conditional on G_min=10；作者未測 G_min=15 重跑 |

### Eureka

一句話：Ricci flow 在正曲率均勻圖上先把「潛在」社群放大成曲率雙峰；ARCUS 用 gap ratio G 與穩定窗決定何時停、在哪切，然後遞歸，不需要先剪枝或指定社區數。直覺是：flow 收縮正曲率高曲率邊、拉伸近平坦邊，等到低曲率族群穩定分離，缺口本身就是手術線。

### 信息流

```text
daily returns r_i(t)
   -> Pearson correlation C_ij
   -> weighted graph w_ij^(0)=sqrt(2(1-C_ij))  [Section 5 scaling]
   -> Ollivier-Ricci curvature κ_ij  [α=1/2, OTD]
   -> Ricci flow iterations: w <- (1-κ)w, rescale mean to 1
   -> sort κ, find largest gap [κ_lo, κ_hi]
   -> G = (μ(κ+)-μ(κ-)) / σ_pooled
   -> stop if iter≥5, G>G_min, lower-population size stable 3 iters
   -> surgery: remove κ≤κ_lo
   -> part A = largest connected component; part B = remaining nodes
   -> recurse if N≥N_min=15; else terminal
```

## §2 · 數學層

📌 **Napkin Formula**

\[
\tilde w_{ij}^{(l+1)}=(1-\kappa_{ij}^{(l)})w_{ij}^{(l)},\qquad
w_{ij}^{(l+1)}=\tilde w_{ij}^{(l+1)}\frac{|E|}{\sum_{(k,m)\in E}\tilde w_{km}^{(l+1)}}
\]

\[
G=\frac{\mu(\kappa^{+})-\mu(\kappa^{-})}{\sigma_{\text{pooled}}},\qquad
\sigma_{\text{pooled}}=\left[\tfrac12\left(\sigma(\kappa^{+})^2+\sigma(\kappa^{-})^2\right)\right]^{1/2}
\]

直覺：第一式讓正曲率邊收縮、近平坦／負曲率邊伸長；第二式把兩群曲率的分離程度標準化，G 越大代表缺口越空、切在哪都一樣。複雜度：所讀文本未見；loss／訓練：無監督，無 loss，停止規則是 gap-and-stability，不是梯度訓練。主要參數：α=1/2、G_min=10、N_min=15、stability window 3 iterations、minimum 5 iterations、hard cap 100 iterations。

## §2.5 · 帶數字走一遍（Worked Example）

以下不是論文新實驗，而是用所讀原文節點 1 已給的逐字數值走一遍流程；μ(κ+)、μ(κ−)、σ_pooled 未披露，因此無法從頭重算 G。

1. 取 root graph：N=226，Section-5 scaling 下初始 κ_min=0.4338，且所有邊曲率為正，故為 ORC-spherical。
2. 逐迭代跑 Ricci flow；第 20 iterations 時，最大曲率缺口落在 κ_lo=0.0116 與 κ_hi=0.4993 之間。
3. 由 Eq. (6) 得到 G=50.93；因 G>G_min=10，且 lower population 已連續 3 iterations 穩定，停止規則觸發。
4. 手術移除 κ≤κ_lo 的邊，共 7,728 條；最大連通分量為 184 nodes，其餘為 42 nodes。
5. 對 184 與 42 遞歸（42≥N_min=15）；整棵樹最終 28 surgeries、13 levels、29 terminal clusters。

## §3 · 數據層

- **市場／頻率**：Nikkei-225 日頻 close-to-close simple returns，Pearson correlation；分析只用 1,572 Tokyo Stock Exchange trading sessions，從 6 January 2020 到 12 June 2026 inclusive。Yahoo! Finance 下載範圍為 1 January 2010 – 12 June 2026，但 earlier prices are not used。
- **圖規模**：226 nodes、25,425 edges；T/N≈7，作者明說 correlation matrix 有 substantial estimation noise，且不濾波，讓 flow 作用在 raw matrix。
- **universe**：固定 Nikkei-225 ticker 清單，不隨 index changes 更新；no survivorship correction is made。清單含 1 April 2026 review 前後不一致的成分，附錄 Table 4 列 Tokyo Stock Exchange codes；anc/S1_constituents.csv 給 ticker、company name、terminal cluster、ARCUS node 與 data/index-membership note。
- **缺失／晚上市**：兩個 delisted tickers 無價格資料，以 C_ij=0 placeholder 進圖；三者為 window 內上市，correlations 用 pairwise-complete observations。因不同 pair 用不同樣本，C 不是 positive semidefinite，有一個 eigenvalue -0.097。兩個 placeholder 合計增加 449 of 25,425 edges，作者稱最多承載任何真實節點 non-lazy mass 的 0.8%。
- **樣本外／容量／成本**：所讀文本未見樣本外預測、交易成本、容量或 turnover 假設；本文是描述性橫截面圖分析，不是交易回測。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 本文 ARCUS 專用 repo：所讀文本未見；依賴 GraphRicciCurvature，https://github.com/saibalmars/GraphRicciCurvature |
| Checkpoint | 未披露 |
| License | 所讀文本未見 |
| 複現難度 | TBD（未驗證）；需 Yahoo! Finance 日頻資料、GraphRicciCurvature version 0.6.1、Section-5 scaling 與 G_min=10／N_min=15 設定 |
| 數據可得性 | 原文稱 Yahoo! Finance；anc/S1_constituents.csv 提供節點與 terminal cluster 對照；完整價格資料未附 |

## §5 · 評測 / Benchmark

表註：本文沒有 IR/Sharpe/AR/MDD 或任何交易／預測基線；「前SOTA」欄一律未披露，Δ 不計算。以下數值均為所讀原文逐字給出的幾何／聚類診斷，不是績效改善量。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---:|---:|---|
| Nikkei-225 | 節點數 N | 未披露 | 226 |  |
| Nikkei-225 | 邊數 | 未披露 | 25,425 |  |
| Nikkei-225 | T/N | 未披露 | ≈7 |  |
| Nikkei-225（Section 4 scaling） | 初始最小曲率 | 未披露 | 0.468 |  |
| Nikkei-225（Section 4 scaling） | 初始最大曲率 | 未披露 | 0.502 |  |
| Nikkei-225（Section 5 scaling） | 初始最小曲率 κ_min | 未披露 | 0.4338 |  |
| Nikkei-225 20-iteration flow | 高曲率邊數（0.497–0.512） | 未披露 | 17,697 |  |
| Nikkei-225 20-iteration flow | 低曲率邊數（-0.146–0.014） | 未披露 | 7,728 |  |
| Nikkei-225 fixed surgery | 最大連通分量節點數 | 未披露 | 184 |  |
| Nikkei-225 fixed surgery | 其餘節點數 | 未披露 | 42 |  |
| Nikkei-225 fixed surgery | modularity | 未披露 | 0.183 |  |
| ARCUS root | G | 未披露 | 50.93 |  |
| ARCUS root | 停流迭代 | 未披露 | 20 |  |
| ARCUS root | surgery edges | 未披露 | 7,728 |  |
| ARCUS root | 最大連通分量節點數 | 未披露 | 184 |  |
| ARCUS root | 其餘節點數 | 未披露 | 42 |  |
| ARCUS whole | surgeries | 未披露 | 28 |  |
| ARCUS whole | levels | 未披露 | 13 |  |
| ARCUS whole | terminal clusters | 未披露 | 29 |  |
| ARCUS whole | terminal cluster 最小 size | 未披露 | 2 |  |
| ARCUS whole | terminal cluster 最大 size | 未披露 | 14 |  |
| ARCUS whole | median stopping iterations | 未披露 | 17 |  |
| ARCUS whole | G 最小 | 未披露 | 10.02 |  |
| ARCUS whole | G 最大 | 未披露 | 92.44 |  |
| ARCUS whole | G<15 的節點數 | 未披露 | 8 |  |
| ARCUS whole | 手術集等於兩部間邊的節點數 | 未披露 | 26 |  |
| ARCUS whole | 小部 < N_min 的 split 數 | 未披露 | 21 |  |
| 終端簇 C8 | 內部平均相關 ρ_in | 未披露 | 0.85 |  |
| 終端簇 C21 | 內部平均相關 ρ_in | 未披露 | 0.82 |  |
| 終端簇 C2 | 內部平均相關 ρ_in | 未披露 | 0.79 |  |
| 終端簇 C4 | 內部平均相關 ρ_in | 未披露 | 0.77 |  |
| 終端簇 C11 | 內部平均相關 ρ_in | 未披露 | 0.77 |  |

解讀：最強的內部穩健性證據是 top-level split 的 members 與 fixed-schedule surgery 完全相同，且 fixed schedule 下手術集對 gap 內任何 threshold 不敏感——原文說 7,728 條邊正是 184×42，且任何落在 [0.014,0.497] 的 threshold 給同樣手術。但這不是獨立驗證：Section 4 與 Section 5 使用不同 weight scaling，作者明說 not strictly comparable；ARCUS 低層有 8 of the 28 nodes 在 G<15 停流，且全部 N≤27，所以 13 levels 的深度 conditional on G_min=10。21 of 28 splits 只剝離小於 N_min 的小部，這是「deep, unbalanced hierarchy」的來源。終端簇的 ρ_in−ρ_out 對 2,000 random groups 的 p 值雖多數 <0.001，但作者自己說 clusters 與 cohesion 用同一組 correlations，是 descriptive check, not independent validation；C12 的 p=0.123 是例外。兩個 zero-filled placeholders 落在 C28，作者明說其位置反映 missing data 而非 market structure。所有數字都不能解讀為 alpha、IR 或成本後收益；本文沒有交易回測。

## §6 · 失效與隱含假設

### 論文自述 limitations（§6 Limitations）

- **Descriptive industry reading**：§5.3 的產業描述是作者 by inspection；未與 official sector classification（例如 Nikkei 或 TOPIX sector schemes）對比，cohesion statistics 使用產生 clusters 的同一組 correlations（§6 Limitations）。
- **G_min 與其他參數**：結果只對單一設定（α=0.5、G_min=10、N_min=15）；對參數與 estimation window 的 sensitivity 尚未檢查（§6 Limitations）。
- **Estimation noise**：T/N≈7，correlation matrix 未 cleaning；hierarchy 是否能在 random-matrix filtering 或 correlation-matrix null model 下存活，尚未測試（§6 Limitations）。
- **Two weight scalings**：Sections 4 與 5 使用 Eq. (1) 的不同 scaling；因為 curvature 在此實作依賴 weight scale，兩者 not strictly comparable（§6 Limitations）。
- **Fixed universe and missing data**：node set 是固定清單，不隨 index changes 更新；兩個 delisted tickers 以 zero correlation 進圖並落在 C28；三個 window 內上市股票只覆蓋較短樣本（§6 Limitations、§2）。
- **Binary splits by construction**：part B 是所有非最大連通分量的聯集，所以「two parts」反映實作，不是測得的 community 數（§6 Limitations）。

### 推斷的隱含假設（本頁推斷）

- **ORC-spherical 不是普遍市場常態**：本文的無預存對比前提，依賴 Nikkei-225 共通市場因子使 pairwise correlations 幾乎均勻偏高；若市場在 flow 前已有 negative-curvature inter-community edges，ARCUS 的「分離必須由 flow 自己生成」前提不成立。這是本頁推斷，不能推廣到所有市場。
- **低層分裂對 G_min 敏感**：8 個小節點 G<15，作者說 lower levels conditional on G_min=10；在 G_min=15 下是否仍在 recorded iteration 停止，作者明說未測。不能把 29 clusters 當成參數無關的穩定結構。
- **固定 universe 與 zero-fill 的影響未隔離**：作者說 placeholders 最多承載 0.8% non-lazy mass，但這不是 survivorship 修正；C28 與其鄰接是否因 placeholder 改變，需 rerun without them，所讀文本未見結果。
- **無樣本外／成本／容量**：本文沒有交易策略、沒有 turnover、沒有交易成本、沒有容量假設；任何「風險擇時 alpha」解讀都超出 envelope。這是本頁推斷。
- **相關矩陣非 PSD 的幾何後果未量化**：C 有一個 eigenvalue -0.097，作者只直接檢查 triangle inequality；這對 OTD 曲率與 flow 的影響，所讀文本未見量化分析（未驗證）。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| MST／PMFG + community detection [3,4,5,6] | 先剪枝再偵測；ARCUS 保留全連接、用曲率缺口入手術 | 文獻方法 | 非本文基線 |
| Recursive modularity on Japanese stocks [20] | 以 modularity 優化遞歸；ARCUS 以 G 與 stability 停流 | 文獻方法 | 非本文基線 |
| ORC fragility／systemic risk [12,13] | 曲率作風險指標；ARCUS 作遞歸分割與 hierarchy | 文獻方法 | 非本文基線 |
| Ref. [1] ARCUS on NASDAQ-100 | G_min=15、101 nodes、five levels；本文 G_min=10、226 nodes、13 levels | 前作 | 不同市場與設定，不可直接算 Δ |
| Fixed-schedule Ricci flow + Eq. (5) | 20 iterations + 手選 0.2 threshold；top-level members 與 ARCUS 相同 | 本文內部基線 | 不同 weight scaling，作者說 not strictly comparable |

🎤 **Interview Tip**

- **正確答**：ARCUS 是無監督的圖幾何遞歸手術；它解決的是「相關圖全連接、社群結構潛伏、不能先剪枝」的前處理問題。top-level 184/42 與 fixed schedule 一致是內部穩健性，不是樣本外 alpha；低層受 G_min=10 與 N_min=15 制約，且沒有成本／容量／official sector 驗證。
- **錯答**：「29 clusters 可直接做交易策略並打敗 SOTA」或「G=50.93 證明模型有效」。本文沒有 IR/Sharpe/AR/MDD 基線，前SOTA 欄未披露；G 是聚類分離度，不是收益或風險調整績效。也不能說 13 levels 是市場結構的必然，作者明說不能把樹形差異歸因於市場結構。

### 可證偽檢驗

以所讀文本的 next steps 做可證偽檢驗，日期以原文窗口 2026-06-12 為截止：在相同 2020-01-06 至 2026-06-12 窗口重跑 G_min=15；若 8 個小節點的分裂完全不變，則「低層 conditional on G_min=10」被削弱。另一檢驗：把兩個 placeholder 移除後重跑，若 C28 成員與鄰接不變，則 placeholder 對 C28 的影響可忽略。實際重跑結果 TBD（未驗證）。

## §8 · For the Reader

- **因子研究員**：把 terminal clusters 當風險分組／中性化籃子前，先做 official sector 對照與 sample-out；本文的 ρ_in 高（shipping 0.85、banks 0.82）只說明同一組 correlations 下的 cohesion，不是獨立因子有效性。T/N≈7 且未 cleaning，當心 noisy correlation 被 flow 放大。
- **高頻執行**：不適用。日頻收盤相關圖、無 intraday 資料、無成本／滑價／容量；ARCUS 的 hierarchy 不是執行信號。
- **組合配置**：可把 hierarchy 當風險預算的候選分層，但 21 of 28 splits 只剝離小於 N_min 的小部，樹深且不平衡；配置前需檢查小簇的穩定性與 turnover，本文未給。
- **LLM-agent／RL 策略**：可把 clusters 當 state abstraction 或 universe 分組特徵，但沒有 reward 驗證、沒有樣本外；zero-filled placeholders 與固定 universe 會把 missing data 編進 state，需先清理。
- **研究學生**：復現重點是 weight scaling（Section 4 vs Section 5）、GraphRicciCurvature 0.6.1、G 的 largest-gap 定義與 3-iteration stability；先確認 triangle inequality 與 non-PSD 的處理，再談 13 levels 的意義。

## References

- Siddharta Laloux, Bhargavi Srinivasan, “Neckpinches in a Positively Curved Market: Ricci Flow with Surgery on the Nikkei-225,” arXiv:2611.25860v1 [q-fin.ST], 19 Nov 2026. https://arxiv.org/abs/2611.25860v1
- Ref. [1] B. Srinivasan, “Intrinsic Geometry of the Stock Market from Graph Ricci Flow,” arXiv:2510.15942 [q-fin.ST] (2025).
- Ref. [14] C.-C. Ni, Y.-Y. Lin, F. Luo, J. Gao, “Community detection on networks with Ricci flow,” Sci. Rep. 9, 9984 (2019).
- Ref. [25] C.-C. Ni, Y.-Y. Lin, F. Luo, J. Gao, GraphRicciCurvature, https://github.com/saibalmars/GraphRicciCurvature.
- Ref. [26] A. Gosztolai, A. Arnaudon, “Unfolding the multiscale structure of networks with dynamical Ollivier-Ricci curvature,” Nature Communications 12, 4561 (2021).
- Ref. [19] W.-S. Jung et al., “Group dynamics of the Japanese market,” Physica A 387, 537 (2008).
- Ref. [20] T. Isogai, “Clustering of Japanese stock returns by recursive modularity optimization for efficient portfolio diversification,” J. Complex Netw. 2, 557 (2014).