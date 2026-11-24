<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=风险择时 autonomy=全自动黑盒 -->

# 扩展对数指数族（path-independent ADL rules） 解構（扩展对数指数族（path-independent ADL rules））

> **來源發布**：2026-11-19 · （無 venue） · arXiv [2611.26623](https://arxiv.org/abs/2611.26623)
> **arXiv 原文**：[Path-Independent Deleveraging Rules](https://arxiv.org/abs/2611.26623v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點是 micro／高頻日內／因果結構／風險擇時／全自動黑盒。它解的不是 alpha 預測，而是壓力情境下 ADL 損失分配規則的公理化表徵：把「路徑無關、不可重開套利、可校準」收斂成 extended log-exponential family，再在風險—公平張力下選參數。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把 ADL 規則以 equal treatment、continuity、consistency 與 path independence（DPI+UPI）刻畫成 extended log-exponential family；reopen-proofness 推出 entry-price independence，split／merge-proofness 則對 residual map 加 superadditivity／subadditivity。Homogeneity 把 finite homogeneous 規則收窄為 \(b(x)=\theta p_\tau/x\)（另有 boundary \(w(x)=x^\gamma\)），本文再聚焦 \(\theta\le0\) 子族，讓 \(\theta\) 從 0 往 \(-\infty\) 由 proportional division 走向 water-filling。關鍵張力是 \(\theta\le0\) 上 fairness 與 platform risk 反向：越靠 0 越公平、越負越降風險；但只有 \(\theta=0\)／proportional 同時 split- and merge-proof，\(\theta<0\) 只 split-proof。Hyperliquid 10 October 2025 的 18 市場反事實重放中，\(\theta=-1\) 的 24-hour breach share 為 0.563%（observed allocation 1.176%），burden Gini 0.386（observed allocation 0.834）。

**X-Ray.** 這篇的 Pareto 位移在「清算機制設計」而非交易信號：它把 ADL 從交易所自訂排序，提升為可公理化、可排序、可事後反事實審計的規則族。對量化讀者，價值不在多一個因子，而在於把「壓力時誰被砍、砍多少、砍後平台還剩多少尾部風險」變成可檢驗的約束優化。它解決的舊坑是：proportional 簡單但不降風險；water-filling 降風險但把負擔集中；split／merge／reopen 讓交易者可用帳戶結構或 entry price 套利。本文用 path independence 把規則形狀鎖住，再用 homogeneity 與 manipulation immunity 切出可操作的子族。它打不開的 envelope 很清楚：單資產 isolated margin、被動持有殘餘倉位、只重放首批 ADL、未重建 cross-margin liquidation 與後續 cascade；因此不能直接外推為「所有市場、所有保證金制度都最優」。對實盤／回測研究員，這是 mechanism audit 工具，不是可交易 alpha。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作／參照 | 本文做法 | 讀者意義 |
|---|---|---|---|
| 公理化表徵 | Campbell et al. 2026 以 water-filling 做 state-by-state 損失最小；Chitra et al. 2026 研究 sequential implementation；Schlegel 2026 給 scalar bi-compositional division | Theorem 2.2：equal treatment of equal accounts、continuity、consistency、path independence（DPI+UPI）四者 iff extended log-exponential family；允許 residual schedule 隨 position 與 equity 連續依賴 | 把「規則長什麼樣」變成可證唯一性問題，而不是交易所拍腦袋；但四條公理缺一不可 |
| 操縱免疫 | 既有 ADL 有的用 unrealised P&L／leverage 排序；split／merge 操縱在 claims 文獻有前例 | Prop 3.1：在 consistent、DPI、equal treatment 下，reopening-proofness iff entry-price independence；Lemma 3.2：split／merge-proofness iff residual map 的 superadditivity／subadditivity；Prop 3.3 再限制 \(b\) 或 \(w\) | 封住「關倉再開倉改 entry price」與「拆／併帳戶改善待遇」；但 proportional 是唯一同時 split- and merge-proof 的成員，\(\theta<0\) 只 split-proof |
| 風險—公平權衡 | water-filling 在 pointwise risk 最優；proportional 在 percentage-reduction Lorenz 最公平 | Homogeneity 給 finite \(b(x)=\theta p_\tau/x\) 與 boundary \(w(x)=x^\gamma\)；Cor 4.4 給 fairness 排序、Prop 5.4 給 risk 排序；Cor 6.1 證明 \(\theta\le0\) 上兩者反向；Prop 6.3 用歷史風險上限校準 \(\theta^*\) | 給平台一個單參數菜單：不是二選一，而是可審計的連續折衷；但 calibration 選出的 \(\theta^*\le0\) 一般只保 split-proof，不保 merge-proof |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

**一句話：** 先用 equal treatment、continuity、consistency 與 path independence 把 ADL 規則逼成 extended log-exponential 排程；homogeneity 再把 finite homogeneous 規則壓成 \(b(x)=\theta p_\tau/x\)（boundary 另有 \(w(x)=x^\gamma\)）；在 \(\theta\le0\) 子族中，\(\theta\) 是「從 proportional 往 water-filling 走多遠」的旋鈕，平台用風險上限選最靠 0 的 \(\theta^*\)。

直覺：每個帳戶有一條 residual schedule，清算水準 \(\lambda\) 是共同的「預算線」；\(\theta\) 越負，越接近 water-filling，越優先砍高槓桿，平台尾部損失越低，但百分比減倉越不均。反過來，\(\theta\) 越靠近 0，越接近 proportional，越公平，但風險越高。注意這不是「免疫所有操縱」的萬能族：\(\theta<0\) 只保證 split-proof，merge-proof 要 \(\theta\ge0\)，兩者兼得只有 \(\theta=0\)。

### 1.3 信息流 ASCII 圖

帳戶型別 \((q_i,e_i)\)
        │
        ▼
參數剖面 \(b(e_i)=\theta p_\tau/e_i\)  ──►  個別 residual schedule
        │                         \(F_\lambda(q_i,e_i)=\varphi_b^{-1}(\lambda\varphi_b(q_i))\)
        ▼
共同清算水準 \(\lambda\)：\(\sum_i F_\lambda(q_i,e_i)=\widehat E\)
        │
        ▼
殘餘倉位 \(r_i\)；強制減倉 \(q_i-r_i\)
        │
        ├─► fairness：percentage reduction \(h_i=1-r_i/q_i\) 的 position-weighted Lorenz 分散
        └─► risk：殘餘槓桿 \(p_\tau r_i/e_i\) 的尾部損失／breach
        │
        ▼
歷史 calibration：在 risk bound \(\overline R\) 下選 max \(\theta\)（最公平）

## §2 · 數學層

📌 **Napkin Formula**

\[
F_{\lambda}(q_i,e_i)=\varphi_{b(e_i)}^{-1}\!\left(\lambda\,\varphi_{b(e_i)}(q_i)\right),\qquad
\varphi_b(x)=\frac{e^{bx}-1}{b}\ (b\neq0),\quad \varphi_0(x)=x
\]

Homogeneous finite rule 取 \(b(e_i)=\theta p_\tau/e_i\)，清算水準 \(\lambda\) 由 \(\sum_i F_\lambda(q_i,e_i)=\widehat E\) 決定。所讀文本未見數值計算複雜度；計算上是求單一 common clearing level，再逐帳戶套 schedule。直覺：先把 position 經 \(\varphi_b\) 變換到「分數空間」，乘上共同 \(\lambda\)，再反變換回 residual position；\(b<0\) 讓小倉位保留比例較高、大倉位被砍較多，\(b=0\) 退化成 proportional。本文不是 ML 訓練；calibration 是最小化分散度 subject to 歷史風險上限 \(R_{\mathcal C}(\theta)\le\overline R\)，風險準則由平台選定（所讀文本以 replay 的 peak loss 類指標示範）。

## §2.5 · 帶數字走一遍（Worked Example）

以下用原文 Figure 2 的數值作示意算例（原文自帶，非 10 October 2025 重放實證，也不是本頁新造參數）。

1. 設定 \(\theta=-1\)、\(p_\tau=1\)、\(q=(2,2)\)、\(e=(2,1)\)、target \(\widehat E=2\)。
2. 原文給個別排程 \(F_\lambda(2,e_i)=-e_i\log\!\left(1-\lambda(1-\exp(-2/e_i))\right)\)；對 \(e_1=2\) 與 \(e_2=1\) 分別是 \(-2\log(1-\lambda(1-e^{-1}))\) 與 \(-\log(1-\lambda(1-e^{-2}))\)。
3. 解兩條排程相加等於 target，原文得 \(\lambda^*\approx0.678\)。
4. 代回排程得 \(r_1\approx1.118\)、\(r_2\approx0.882\)；相加回到 \(\widehat E=2\)。高 equity 的 account 1 保留較多 residual，符合原文「increasing the first account’s equity increases its residual position」。

這個例子只示範「共同清算水準 + equity-dependent schedule」怎麼運作；真正的 fairness／risk 比較在 §5 的 18 市場重放。

## §3 · 數據層

- **事件與市場**：Hyperliquid 10 October 2025 ADL episode；取 18 個有價格覆蓋的市場，每個市場取第一個占該市場 recorded ADL volume 至少 1% 的 ADL batch。
- **時段**：這些 batch 發生在 21:16 到 21:19 UTC，approximately 3 minute long time interval；mark price 約每 three seconds 更新（原文說 validators 發布新 oracle prices 時）。
- **樣本規模**：重建後 39,915 account–market observations；initial short notional 為 \$5.219 billion；observed reductions 移除 \$534.588 million，或 10.24% of that notional；覆蓋 selected batches 的 99.79% ADL notional。
- **排除**：排除 snapshot 後創建的帳戶、23 個 non-positive marked equity 的 position、32 個 leverage 超過市場最大值兩倍的 replay artifacts；排除對每個規則、observed benchmark、分母與 market targets 一致套用。
- **價格與估值**：ADL 以 previous mark price 成交；mark lag 使 execution price 在 16 of 18 markets 高於 batch 前最後一筆交易，最高 up to 11%。只發布 one-minute marks，作者用 order books、trades、Binance prices 重建 three-second mark，對 batch prices 的 median error 為 0.9%。
- **樣本外／容量假設**：replay 只重分配 first batches，不重放 entire subsequent ADL cascade；被動持有 residual positions，未重建 cross-margin liquidation、後續交易、collateral transfers、funding、liquidations、price impact。這些限制在較長 horizon 更顯著（§7.4）。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| ConejoCapital `HyperMultiAssetedADL` | 不適用（非 ML checkpoint） | 未披露 | 中高：需重建 population 與 mark | 公開 GitHub repository；提供 ADL fills、market data、clearinghouse snapshots |
| ConejoCapital `HyperReplay` | 不適用（非 ML checkpoint） | 未披露 | 中高：需處理 mark 重建與排除規則 | 公開 GitHub repository；Hyperliquid public archive |
| 本文 reproduction files | 未披露 | 未披露 | 需核對：完整 horizon 與參數結果在 reproduction code | 所讀文本未見附件清單，圖表／附件待核 |

## §5 · 評測 / Benchmark

**注意：** 來源沒有把 Hyperliquid observed allocation 稱為 SOTA；下表沿用模板欄名，實際基線是 observed allocation。water-filling 是本文族內端點，也不是前SOTA。\(\theta=-1\) 是文中示例參數，不是平台最終選定 \(\theta^*\)；所讀文本未見實際選定 \(\theta^*\) 的數值。

| 數據集/市場 | Metric | 前SOTA（基線：Hyperliquid observed；非SOTA） | 本方法（\(r^\theta\), \(\theta=-1\)） | Δ（本方法−基線） |
|---|---:|---:|---:|---:|
| Hyperliquid 18 markets, 10 October 2025 | Burden Gini | 0.834 | 0.386 | -0.448 |
| Hyperliquid 18 markets, 10 October 2025 | MAD (pp) | 15.42 | 3.99 | -11.43 |
| Hyperliquid 18 markets, 10 October 2025 | Top 10% share (%) | 65.5 | 32.8 | -32.7 |
| Hyperliquid 18 markets, 10 October 2025 | 24-hour notional breaching margin (%) | 1.176 | 0.563 | -0.613 |
| Hyperliquid 18 markets, 10 October 2025 | 24-hour excess peak loss (\$1,000) | 1,099.3 | 159.7 | -939.6 |

原文說 every evaluated \(\theta\le-0.5\)，including water-filling，在 Table 1 的 every statistic 都優於 observed allocation。Table 1 中 water-filling 的 24-hour breach 欄為 0.540；正文給 \(\theta=-1.25\) 的 24-hour breach share 0.558%，slightly below \(\theta=-1\) 的 0.563%，observed 為 1.176%。\(\theta=-1.25\) 的 burden Gini 0.442、\(\theta=-1\) 0.386、water-filling 0.735；\(\theta=-1.25\) 的 excess peak loss 為 \$101,729，\(\theta=-1\) 為 \$159,684；這正是「中間參數可換公平、保留多數風險改善」的張力。observed allocation 本身 fully closes 12,333 of the 12,378 positions it reduces，這是其 burden Gini 偏高的來源之一。

Table 2 以 dollar reductions per unit of equity 重做分散度：Hyperliquid Gini 0.8941／MAD 0.2956；water-filling 0.8575／0.2557；\(\theta=-1\) 0.5982／0.1453；proportional 0.2908／0.0745；equity-weighted CEL 0.0087／0.0045；原文說相應 mean 在 every rule 下為 0.1831。這組結果支持：即使 fairness 改以絕對減倉衡量，中間規則仍比 water-filling 與 observed 更均勻。

**解讀：** 真 capability 在於「同一市場目標下重新分配首批 ADL」能同時降低 burden dispersion 與 24-hour breach，這是 rule-level allocation effect。不能全歸因於規則本身：replay 只重分配 first batches、被動持有 residual、mark 用重建而非全觀測、單資產 isolated margin、未重建 cross-margin 與後續 cascade。估值也敏感：用 last trade price 時 \(\theta=-1\) 的 24-hour breach 為 0.564%、observed 為 1.232%；只 mark deleveraged coin 時為 0.805% 與 1.334%。Bootstrap 300 條 price paths 下，24-hour expected breach \(\theta=-1\) 0.174% vs observed 0.270%（preceding 30 days），0.253% vs 0.451%（subsequent week）；方向保留，但水平隨估值與路徑變。所讀文本未見成本後或 live PnL 的 Δ，因此下表 Δ 只能讀作 counterfactual allocation 差異，不是可實現淨收益。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

- **單一壓力事件、被動部位、單資產假設**：Conclusion 明說 replay evaluates one stress episode under passive-position and single-asset assumptions。
- **只重放首批 ADL，不重放後續 cascade**：§7.1 說 exercise reallocates these first batches; it does not replay the entire subsequent ADL cascade。
- **未重現 cross-margin liquidation**：§7.4 說 account equity 按 marked notionals 分配到各 position，再逐 position 評估，does not reproduce cross-margin liquidation。
- **mark 重建而非全觀測**：§7.4 說 marks without an ADL fill in the block are reconstructed rather than observed；one-minute marks 需重建 three-second mark。
- **保證金階梯固定**：§7.4 說 maintenance-margin rates 固定用 replay recorded metadata values，未重建 historical margin tiers。
- **被動評估省略後續行為**：§7.4 說 passive evaluation omits subsequent trading, collateral transfers, funding, liquidations and price impact；這些限制在 longer horizons 更嚴重。
- **breach 不是實際清算**：§7.2 說 breach shares 是 threshold breaches of passive positions, not observed subsequent liquidations；\(L^{\mathrm{peak}}\) 是 stress criterion，not a realised loss of the platform or its insurance fund。
- **成本與資金流未入模型**：§2.1 明說 ignore transaction costs and funding, and retain realised P&L。
- **homogeneity 的適用邊界**：§2.3 說 homogeneity 排除 fixed absolute thresholds；若市場深度使絕對倉位大小重要，這個假設可能不合適（原文未給數值門檻）。

所讀文本未見完整圖表與附件明細；上述僅列正文可定位的作者限制，其餘圖表／附件待核。

### 6.2 推斷的隱含假設（本頁推斷）

- **Regime 依賴**：校準風險上限來自特定壓力 episode；在低波動或流動性分層不同的 regime，\(\theta\) 的風險排序未必同向。原文只給單一 replay，未給其他 episode 的數值。
- **單資產 isolated margin**：模型把每個 market 的 position 與 equity 分開評估；若真實帳戶是 cross-margin，equity 分配與 liquidation 連鎖會改變 breach 與 loss。原文 §7.4 已列此限制。
- **被動持有 residual**：replay 假設 ADL 後不再交易、不再補保證金；真實交易者會對 ADL 反應，這會改變後續價格與風險。
- **成本與價格衝擊**：§2.1 忽略 transaction costs and funding；§7.4 省略 price impact。本頁推斷：重分配本身可能改變市場深度，尤其在高頻日內尺度。
- **Calibration 的風險上限是外生**：Prop 6.3 給定 \(\overline R\) 與 calibration set \(\mathcal C\)；所讀文本未見 \(\overline R\) 的實際數值或平台如何選 \(\overline R\)。這是治理選擇，不是模型內生。
- **Split／merge 免疫不是全族**：Cor 3.5 的直接含義是 \(\theta<0\) 只 split-proof、\(\theta>0\) 只 merge-proof，只有 \(\theta=0\) 兩者兼得。若平台按 Prop 6.3 選 \(\theta^*<0\)，合併帳戶仍可能是操縱面；這是規則層的必然邊界，不是 replay 實驗結果。
- **Entry-price independence 只覆蓋 ADL allocation**：規則刻意不用 entry price；若平台其他風控層仍用 entry price，reopen 操縱可能轉移到別處。本文只處理 ADL allocation。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Campbell et al. 2026, risk-based ADL | 以 water-filling 做 state-by-state 損失最小；本文給 rule-level uniqueness 與 fairness–risk 菜單 | 未披露 | 被本文引用為 arXiv:2603.15963v2 |
| Chitra et al. 2026, ADL as online learning | 研究 sequential implementation；本文是靜態公理化 allocation rule | 未披露 | 被本文引用為 arXiv:2602.15182 |
| Chitra 2025, ADL impossibilities and optimization | 研究 loss socialisation、solvency、fairness 的不可能性與優化；本文用 path independence 收斂規則族 | 未披露 | 被本文引用為 arXiv:2512.01112 |
| Schlegel 2026, bi-compositional division rules | scalar claimant types；本文擴到 position 與 equity 二維帳戶 | 未披露 | 被本文引用為 arXiv:2609.01489v1 |
| Hyperliquid observed ADL | 文件描述以 unrealised P&L 與 leverage 排序；本文刻意 entry-price independent | 未披露 | deployed benchmark；本文用 recorded reductions，未重建其排序演算法 |

🎤 **Interview Tip**

- **正確答**：先說「Theorem 2.2 需要 equal treatment、continuity、consistency、path independence 四條，才等價於 extended log-exponential family」；再說「homogeneity 只給 finite \(b=\theta p_\tau/x\)，boundary 另有 \(w=x^\gamma\)，不是只剩 \(\theta\) 族」；最後補「\(\theta\le0\) 上 fairness 與 risk 反向，calibration 選 \(\theta^*\le0\)，所以 split-proof 成立、一般非 merge-proof；只有 \(\theta=0\) 兼得。10 October 2025 replay 是 first-batch counterfactual，不是 live PnL」。
- **錯答**：把它說成 alpha 因子或高頻預測模型；說 path independence 單獨就能刻畫；說 split＋merge＋homogeneity 收窄成整個 \(\theta\) 族；說 water-filling 在所有 fairness 指標都最優；把 observed allocation 當 SOTA 並自行合成領先幅度；忽略 cross-margin、cascade、mark reconstruction 與估值敏感性。

### 7.1 可證偽預測（以 2026-11-19 原文為基準，2027-11-19 前可檢驗）

1. 若後續公開回放把 cross-margin liquidation 與完整 ADL cascade 納入，且在同一 risk bound 下 \(\theta=-1\) 仍同時改善 breach 與 burden dispersion，則本文的外部有效性增強；若改善消失，則 first-batch／single-asset 假設是主因。
2. 若平台按 Prop 6.3 上線 \(\theta^*<0\)，交易者仍能透過 merge 改善 ADL 待遇，則 Cor 3.5 的「\(\theta<0\) 不 merge-proof」在實務接口上被證實為漏洞；若平台改選 \(\theta=0\)，split／merge 都免疫，但失去 water-filling 方向的風險改善。
3. 若在非 10 October 2025 的壓力 episode 中，\(\theta\le0\) 的 fairness–risk 反向排序不再成立，則 calibration 的單調性需要重新檢驗；所讀文本未給其他 episode 的數值。

## §8 · For the Reader

- **因子研究員**：不要把 \(\theta\) 當 alpha 參數；把它當「壓力情境下的損失分配算子」。可研究的是：不同 market 的 \(\theta\) 校準是否穩定、burden dispersion 是否預測後續流動性與波動。
- **高頻執行**：關注 mark lag、last trade vs mark、three-second mark 重建；本文顯示同一規則在 last-trade valuation 與 mark-only valuation 下 breach share 會變，執行層不能只看規則名。
- **組合配置**：把 ADL 規則視為交易所風險參數；若平台採用較負 \(\theta\)，尾部風險較低但高槓桿帳戶被砍更集中，對沖成本與基差風險會轉移。也要注意 \(\theta<0\) 不保 merge-proof。
- **LLM-agent／RL 策略**：這是 mechanism design 環境，不是單智能體 alpha；若要做 agent，需把 split／merge／reopen 當 action space，並用 path independence 檢查 reward hacking。
- **研究學生**：讀 Theorem 2.2、Prop 3.1、Prop 3.3、Cor 3.5、Theorem 4.3、Prop 5.4、Prop 6.3 這條主線；Appendix 的獨立性例子與 Example 6.4 是理解公理必要性與非齊次規則優勢的捷徑。

## References

- 原論文：Christoph Schlegel and Akaki Mamageishvili, _Path-Independent Deleveraging Rules_, arXiv:2611.26623v1, 2026-11-19. https://arxiv.org/abs/2611.26623v1
- Lineage：Campbell et al. 2026, _Risk-based auto-deleveraging_, arXiv:2603.15963v2；Chitra et al. 2026, _Autodeleveraging as online learning_, arXiv:2602.15182；Chitra 2025, _Autodeleveraging: impossibilities and optimization_, arXiv:2512.01112；Schlegel 2026, _Bi-compositional division rules_, arXiv:2609.01489v1。
- 數據／重放：ConejoCapital, `HyperMultiAssetedADL`；ConejoCapital, `HyperReplay`；Hyperliquid public archive（連結見原文 §7）。