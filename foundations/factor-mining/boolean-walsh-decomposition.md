<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# Boolean/Walsh decomposition 解構（Boolean/Walsh decomposition）

> **來源發布**：2026-11-06 · （無 venue） · arXiv [2611.09834](https://arxiv.org/abs/2611.09834)
> **arXiv 原文**：[The Logic of Factor Portfolios: Boolean Rules, Interactions, and Trading Costs](https://arxiv.org/abs/2611.09834v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在量價表格／中長周期／監督回歸／因子挖掘／人機協同可解釋；它把 decision rule 寫成 threshold literals 的 Boolean function，補上「規則曝險—交互項—換手—交易成本」這條過去常被當成 construction detail 的缺口，而不是提出新的報酬預測器。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `监督回归` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把選股規則寫成 threshold literals 的 Boolean function，用 Walsh 展開精確拆成單訊號與交互組合。②同一組係數加上 literal persistence 推換手；AND 與 OR 的 first-order 曝險相同、interaction 符號相反，門檻實作會丟資訊。③對因子挖掘的意義是 rule audit 與 pre-trade cost model，不是「交互項越多越好」的 alpha 生成器。④美國 1973–2024：預測與實現 persistence 相關係數至少 0.999，286 條 AND/OR/majority 規則的 predicted turnover 對 realized turnover 斜率 1.02；1995–2024 平均 quoted-spread cost 由 first order 的 4.8 bp/月升至 second 的 7.5、third 的 9.5，absolute premium 高於成本的比例為 58%、24%、6%。

**X-Ray.** 這篇的 Pareto 位置很清楚：以可解釋性與可審計性換取預測力，不與 ML 黑箱比 best-rule search，而是把任一規則的曝險與換手變成可計算物件。它解掉舊坑——AND/OR 的交互曝險常被混成同一種「組合」，此處用 Walsh 係數分開；換手也不是回測附帶統計，而由 literal persistence 與 interaction order 事前決定。Envelope 同樣明顯：結果依賴 U.S. 1973–2024、12 個訊號、quoted half-spread 成本代理；會計訊號同月翻轉會讓獨立性假設失效，高階 interaction 在個別組合多數吃不過成本。對量化讀者，這是 rule audit 與 cost model，不是新的 return forecasting model；第二階 interaction 的 OOS Sharpe 增益應與成本、短借、容量一起看。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 面向 | 傳統／前作 | 本文改動 | 讀者意義 |
|---|---|---|---|
| 規則表徵 | 把 sort、screen、tree leaf 視為 portfolio construction 的細節 | 規則寫成 literals 的 Boolean function f；AND、OR、threshold、tree leaf 都是同一語言 | 規則可被精確審計，不再只是回測設定 |
| 報酬分解 | 用 factor sort 或 2×3 因子取 first-order 曝險 | Walsh expansion 精確拆成 first-order 與 interaction portfolios；AND 與 OR 只差 interaction 符號 | 交互項不是殘差，而是有方向的規則結構 |
| 換手與成本 | 換手由回測事後統計，成本另做 | 同一組 Walsh 係數加 literal persistence 推換手；threshold 與 smooth score 的成本差異可事前比較 | 交易前即可估算 interaction 的成本代價 |
| 測試資產 | conventional sorts 常不在同一 span | character portfolios of order ≤ k span 所有建構在 k 個 literals 上的規則 | 一組 test assets 可同時檢定大量規則 |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

同一組 Walsh 係數既描述規則的曝險，也透過 literal persistence 的乘積決定換手；因此 interaction 不是統計產物，而是有交易成本代價的規則結構。

直覺：AND 是「兩個訊號都要好」，OR 是「其中一個好就夠」；兩者對單訊號的曝險一樣，但對「兩個訊號同向」的押注方向相反。當訊號本身有持久性，乘積項的持久性是各項持久性的乘積，階數越高、換手越快。

### 1.3 信息流 ASCII 圖

```text
signals s_{j,i,t}
   ↓ rank within universe
u_{j,i,t} ∈ (0,1]
   ↓ threshold q
literals x_{j,q,i,t}=1{u_{j,i,t}>q}
   ↓ Boolean rule f(x)
Walsh coefficients \hat f(S)  ──→  character portfolios Π^S
   ↓                                   ↓
interaction profile w_S(f)          exposures / alphas
   ↓
literal persistence ρ_j
   ↓
ρ_S = Π_{j∈S} ρ_j
   ↓
turnover τ(f) = (1−β) Σ_S w_S (1−ρ_S)
   ↓
quoted-spread cost / net premium
```

## §2 · 數學層

📌 **Napkin Formula**

f(x) = Σ_S \hat f(S) χ_S(x)，其中 χ_S(x)=Π_{j∈S}(2x_j−1)

F^f_{t+1} = Σ_{S≠∅} \hat f(S) Π^S_{t+1}

τ(f) = (1−β) Σ_{S≠∅} w_S(f)(1−ρ_S)，ρ_S=Π_{j∈S}ρ_j

**複雜度**：展開有 2^n 個係數；對 fixed literals 的規則空間可有限窮舉。推論與換手公式是閉式或低維加總，不需要訓練 loss；真正的估計發生在 out-of-sample tangency weights 的 ridge，penalty 由 training window 內 five-fold cross-validation 選出。

**直覺**：Walsh 基底是 Boolean function 的正交基底；係數只依賴規則，不依賴資料，所以分解逐月精確成立。訊號相關會讓 character portfolios 彼此相關，但不改變分解本身。換手則由「literal 會不會翻」與「規則載入哪個階數」共同決定；獨立性與 persistence 是預測換手的關鍵假設。

## §2.5 · 帶數字走一遍（Worked Example）

以下不是市場實證，而是用原文 Table 3 的參數例（k=2、p=1/2、ρ=0.9）走一遍；數字均為原文表格／說明例，不新增估算。

1. **示意輸入**：兩個 median literals x1、x2，p=1/2、ρ=0.9；原文 Table 3 給 a=0.95、b=0.05。
2. **Walsh 展開**：x1∧x2 = (1/4)(1+χ1+χ2+χ12)；x1∨x2 = (1/4)(3+χ1+χ2−χ12)。兩者 first-order 係數同為 1/4，interaction 係數分別為 +1/4 與 −1/4。
3. **交互持久性**：ρ12 = 0.9×0.9 = 0.81（原文說明例）。
4. **AND 腿**：coverage 0.250，τ=0.098（Table 3）；理由是交集要所有 literal 同時留在 favorable side，任何一個離開就可能出場。
5. **OR 腿**：coverage 0.750，τ=0.033（Table 3）；聯集只在所有 literal 同時失效時才丟股，所以腿較寬、換手較低。
6. **讀法**：同一個 second-order 曝險，AND 與 OR 的 first-order 曝險相同，差別在 interaction 符號；若只看報酬不看 interaction profile，會把兩者誤認為同一種組合。

## §3 · 數據層

- **樣本**：CRSP monthly/daily 與 Compustat annual fundamentals，經 CRSP–Compustat Merged（link types LU、LC；primary links P、C）連結。美國 common shares（CRSP share codes 10、11），NYSE/AMEX/NASDAQ；每家公司保留市值最大的一檔；剔除 financial firms（SIC 6000–6999）與低於 NYSE 市值第 20 百分位的股票。
- **規模與頻率**：623 個月度報酬，February 1973 到 December 2024；平均每月 1,257 檔。股票必須在 t 時 12 個訊號齊全才進 universe。
- **時點**：會計變數在 fiscal year end 後六個月才使用，且不往前帶超過 548 天；市場變數在月底 t 量測；positions 持有 t+1 月。universe 在 t 不要求 t+1 有報酬。
- **缺失報酬**：0.4% 的 stock-months 沒有 t+1 月報酬，主結果給零 excess return；其中 89% 是 mergers，1% 是 performance-related delistings；以 CRSP delisting returns 替代的 robustness 在 TableOA.6。
- **成本層**：以 quoted half-spread 代理 one-way cost；1995 年起有所有交易所的 closing quotes。universe 的 median half-spread 由 1995 年的 76 bp 降到 2020 年後的 2.4 bp。成本按月底 target weight 變化收取，包含進出 universe；組合合併時在 stock level 淨額化；不計 rebalancing 日之間的 price drift。
- **容量假設**：所讀文本未見明確 AUM 或 market-impact 校準；quoted spread 不是 effective spread 或大單 impact，容量屬 TBD。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 所讀文本未見 | 不適用／所讀文本未見 | 所讀文本未見 | 高：需 CRSP/Compustat 授權、訊號與成本管線、delisting 處理 | CRSP/Compustat 商業授權；作者未給公開程式 |

## §5 · 評測 / Benchmark

> 下表**不是**全論文方法 vs SOTA；「前SOTA」欄僅為模板欄名，實際是同一 exposure 的 **median literal 基線**，本方法欄是 **rank-product 實作**。Δ 只放同列兩個來源數值之差。

| 數據集/市場 | Metric | 前SOTA（基線：median literal） | 本方法（rank-product 實作） | Δ |
|---|---:|---:|---:|---:|
| U.S. stocks, 1973–2024, order 1 | Persistence (correlation) | 0.796 | 0.853 | +0.057 |
| U.S. stocks, 1973–2024, order 2 | Persistence (correlation) | 0.625 | 0.725 | +0.100 |
| U.S. stocks, 1973–2024, order 3 | Persistence (correlation) | 0.493 | 0.624 | +0.131 |

**解讀**：真正屬於 capability 的部分，是 Walsh 分解的精確性與 span 結果——character portfolios of order ≤ k 覆蓋所有建構在 k 個 literals 上的規則，因此不是單一規則的曲線擬合。原文的實證支持也很直接：各階內 predicted 與 realized persistence 的相關係數至少 0.999；286 條 AND/OR/majority 腿的 predicted turnover 對 realized turnover 斜率 1.02。這些是「規則邏輯→換手」的映射證據，不是 interaction 必然帶來正 alpha 的證據。

成本面則是限制條件的核心。1995–2024，平均 quoted-spread cost per unit gross exposure 由 first order 的 4.8 bp/月升到 second 的 7.5、third 的 9.5；absolute premium 高於成本的比例由 first order 的 58% 降到 second 的 24%、third 的 6%。OOS tangency 在 2005–2024 由 first-order 的 0.18 gross / 0.11 net，加到 second-order 的 0.48 gross / 0.32 net；但 third-order 在 median literals 由 0.48 升到 0.61 gross，在 70th-percentile literals 卻由 0.64 降到 0.41 gross。這是 mixed evidence，不能寫成「階數越高越好」。

**可能是過擬合／偏差的環節**：個別 interaction portfolio 多數不過成本；OOS 增益主要來自 2005–2019，2020–2024 first-order 與 order ≤2 都約 0.5；成本只用 quoted half-spread，未計 short-sale borrowing，且 price drift 被忽略。這些都讓「淨 Sharpe 增益」不能直接外推為可交易容量。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§5.3 成本代理**：quoted spreads overstate the cost of small trades in liquid stocks, understate the cost of large trades, and omit short-sale borrowing costs, which we do not model；price drift between rebalancing dates is ignored。
- **§6.1 獨立性**：Independence across signals is the weakest part of Assumption1，且對 accounting signals 最明顯失效。flip correlation 為 asset growth 與 sales growth 的 0.38、return on equity 與 asset growth 的 0.25；flip correlation ≥0.15 的 second-order characters，realized persistence 平均比預測高 0.035，低於 0.05 時只高 0.001；含此類 literal pair 的腿交易量比預測少 7%。
- **§5.1 完整訊號要求**：要求 12 個訊號齊全，會把 universe 偏向 seasoned firms；beta 需要 24 個月報酬、long-term reversal 需要 36 個月。
- **§5.2 缺失報酬**：0.4% 的 stock-months 沒有 t+1 月報酬，主結果給零；89% 是 mergers、1% 是 performance-related delistings；TableOA.6 以 delisting returns 替代後主要統計不變。
- **§7.1／§8 樣本外不穩定**：second-order 的 OOS 增益主要來自 2005–2019；2020–2024 first-order 與 order ≤2 都約 0.5；third-order 在 median 與 70th-percentile literals 方向不一致，作者建議停在 second order 或 truncate higher orders。

### 6.2 推斷的隱含假設（本頁推斷）

- ** regime 依賴（推斷）**：全部實證是 U.S. 1973–2024，成本段是 1995–2024；不能直接外推到新興市場、微盤或不同漲跌停／放空制度。若 quoted spread 與實際成交成本的關係改變，order 1/2/3 的成本排序未必成立。
- **容量與成本（推斷）**：模型把成本掛在 target weight 變化上，但沒有 AUM、impact curve、borrow fee 的校準；quoted spread 對小單高估、對大單低估。高階 interaction 的淨值結果對成本假設敏感。
- **數據泄漏／時點（推斷）**：作者用 fiscal year end 後六個月、月底 ranks、t+1 持有，時點設計清楚；但 accounting signals 同月翻轉會讓獨立性失效，等於用錯 persistence 乘積。這是模型偏差，不是 lookahead。
- **生存者偏差（推斷）**：universe 不要求 t+1 有報酬，缺失報酬給零並用 delisting returns 做 robustness；但「12 訊號齊全」造成 seasoned-firm tilt。這是樣本選擇限制，不能直接等同 survivorship bias，也不能說已完全排除。
- **組合估計（推斷）**：OOS tangency 的 ridge 權重只用 1973–2004 估計，成本是事後扣；權重優化是否內生化成本，所讀文本未見，不能把 net Sharpe 當成可實行的最優配置。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Conventional sorts／FF6 test assets | breakpoints 與 weights 不在 character span 內 | 資料部分公開；程式未見 | 原文：1995–2024 median portfolios order ≤2 拒絕 FF6 p=0.007，conventional sorts 不拒絕 |
| ML rule search（Gu et al. 2020；Bryzgalova et al. 2025；Cong et al. 2025） | 搜尋最佳規則 vs 分析給定規則 | 程式未見 | OA.4 窮舉規則；最佳規則 OOS 衰減，DSR 低於 0.95 |
| Tree leaves／decision trees | leaves 是 conjunctions；本文 span 所有 k-literal 規則 | 部分概念重疊 | 不是同一個學習器；可審計性與成本連結不同 |
| Rank-product／smooth interactions（Kozak et al. 2020） | threshold vs smooth resolution | 資料／程式未見 | 同 exposure 更細解析度；成本比較取決於 quadratic 或 proportional |

🎤 **Interview Tip**

- **正確答**：Walsh 是對 Boolean rule 的精確正交分解；AND 與 OR 的 first-order 曝險相同、interaction 符號相反；換手由 interaction profile 與 literal persistence 乘積決定，所以高階 interaction 成本更高。這是 rule audit 與 cost model。
- **錯答**：把 0.999 的 persistence 相關當成樣本外預測力；說 interaction 一定提高 Sharpe；說 rank-product 在所有階數都低換手——原文 first order 在 proportional costs 下 rank 的 turnover per unit risk 是 0.241、median 是 0.205，方向相反。

### 7.1 可證偽預測（尚待驗證）

- 在 **2027-12-31 前**，用相同 12 訊號與 CRSP/Compustat 管線延伸 2025–2026：若 order 1/2/3 的平均 quoted-spread cost 不再隨 order 上升，則本文的成本預測在該樣本失效。
- 在 **2027-12-31 前**，對 order 2/3 的 rank-product 與 median 同風險比較：若 rank-product 的 proportional-cost turnover 不再低於 median 在多數 pairs/triples，則 Proposition 4 的 implementation implication 失效。
- 在 **2027-12-31 前**，把 short-sale borrowing 與 effective spread／impact 加入後：若 second-order interaction 的 net Sharpe 增益不再為正，則 2005–2024 的 0.11 到 0.32 只能視為 quoted-spread 下的結果，不能外推為可交易 alpha。

## §8 · For the Reader

- **因子研究員**：先把現有規則寫成 Boolean circuit，檢查 AND/OR 的 interaction 符號與 interaction share；用 character portfolios 的 span 做聯合檢定，不要把「交互項顯著」直接寫成經濟機制。若要復現，先從 Table 3 的 AND/OR/majority 換手公式對帳。
- **高頻執行／交易成本**：把 order 1/2/3 的 turnover 與 quoted-spread cost 分開看；first-order 在 proportional costs 下未必比 smooth score 差，second/third order 才是 smooth 的優勢區。buffer zone 與 truncation 是兩個可測的降成本槓桿；short borrow 與 impact 仍需另補。
- **組合配置**：第二階 interaction 的 OOS 淨 Sharpe 在 2005–2024 有增益，但個別 interaction portfolio 只有 24% 的 second-order、6% 的 third-order 超過成本；配置上應把 interaction 當成昂貴的衛星曝險，並檢查 2020–2024 的衰減。
- **LLM-agent／規則審計**：把投資政策寫成 Boolean rule 後，可用 BDD/SMT 做靜態驗證。原文 OA.6 的示例將規則化簡為 166 個 conjunctions、908 個 literals、251 個 BDD nodes，並檢查 long/short legs 不重疊；這是「可解釋」落到可驗證的具體路徑。
- **研究學生**：最小可復現練習是兩個 median literals 的 Walsh 表：AND/OR 的係數、interaction share 1/3、ρ12=0.81，以及 Table 3 的 coverage 與 τ。先把這張表算對，再進 CRSP/Compustat 的 12 訊號管線。

## References

- Ziyao Wang, Svetlozar T. Rachev, Frank J. Fabozzi. *The Logic of Factor Portfolios: Boolean Rules, Interactions, and Trading Costs.* arXiv:2611.09834, 2026. https://arxiv.org/abs/2611.09834v1
- O’Donnell, R. (2014). *Analysis of Boolean Functions.* Cambridge University Press.
- Benjamini, I., Kalai, G., Schramm, O. (1999). Noise sensitivity of Boolean functions and applications to percolation.
- Hoeffding, W. (1948). A class of statistics with asymptotically normal distribution.
- Efron, B., Stein, C. (1981). The jackknife estimate of variance.
- Gârleanu, N., Pedersen, L. H. (2013). Dynamic trading with predictable returns and transaction costs.
- Fama, E. F., French, K. R. (2015). A five-factor asset pricing model.
- Kozak, S., Nagel, S., Santosh, S. (2020). Shrinking the cross-section.
- Bryzgalova, S., Pelger, M., Zhu, J. (2025). Forest through the trees: building cross-sections of stock returns.
- Cong, L. W., Feng, G., He, J., He, X. (2025). Growing the efficient frontier on panel trees.