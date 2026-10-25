<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=风险择时 autonomy=人机协同可解释 -->

# State-Dependent Hawkes Process (SDHP) 解構（State-Dependent Hawkes Process (SDHP)）

> **來源發布**：2026-10-16 · （無 venue） · arXiv [2610.23619](https://arxiv.org/abs/2610.23619)
> **arXiv 原文**：[Cross-product Modelling in Continuous Intraday Electricity Markets via Hawkes Processes](https://arxiv.org/abs/2610.23619v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「微觀盤口 × 高頻日內 × 因果結構」——它不預測價格，而是估計 Hourly 與 Quarter-Hourly 兩個 XBID 合約之間的**訂單流傳導結構**，並讓這個結構隨可觀測的流動性狀態切換。填的 prior gap 是：跨產品溢出「在壓力下是否改變方向與時標」在 IDC 電力市場完全沒有實證。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把一個 Hourly XBID 合約與其四個 Quarter-Hourly 子合約建成狀態依賴多變量 Hawkes，用 LSI（spread / 波動 / queue imbalance）把交易切成 Normal 與 Stressed 兩種 regime；②核心 trick 是用溢出乘數 $\Psi^{(x)}=(I-G^{(x)})^{-1}$ 而非首輪分支係數 $G$ 讀跨產品傳導，在近臨界對角結構下把「看起來可忽略的 0.021」放大成 2.390 量級；③對「因果結構」軸重要，因為跨產品領先關係被證明不是常數——壓力下 $\Psi_{Q\to H}/\Psi_{H\to Q}$ 由 2.8 壓到 1.5；④關鍵實證：$G^{(x)}$ 的光譜半徑 Normal 0.9518 / Stressed 0.9555，即約 95% 訂單流是內生的。

**X-Ray.** 這篇的價值不在「又一個 Hawkes 變體」，而在把**結構本身的非穩定性**變成估計對象。常數參數 Hawkes 會把 Normal 與 Stressed 平均成一個誰都不描述的結構；而用首輪分支係數 $G$ 判讀跨產品溢出，會因為自激近臨界（$\rho\approx0.95$）而被對角回聲放大——兩個坑它都補了。它明確打不開的 envelope 也很乾淨：沒有價格、沒有 PnL、沒有成本、沒有執行層；事件只含 order submissions 不含 cancellations；時間戳只到毫秒；只覆蓋兩個產品、四個月、單一市場（荷蘭）。對量化讀者的意義不是「找到 alpha」，而是拿到一個可操作的警示：**平靜期校準的跨合約對沖比率，在壓力期會系統性失效，而失效方向可預測（更對稱、更快）**。它給的是風險擇時的結構條件，不是擇時信號本身。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 構面 | 前作 / 對照 | 本文 SDHP | 定位 |
|---|---|---|---|
| 事件空間 | [40] 單一 hourly 產品；[33] 德國 IDC market order arrivals 單產品 | $E=2$：Hourly（H）與其四個嵌套 Quarter-Hourly（Q）同步窗口 | §III-B1 |
| 基線強度 | §III-A 稱相對 [24] 新增 state-dependent baseline；§I 又稱 regime-dependent baseline **承自 [40]** | $\mu_{x,e}$，$x\in\{0,1\}$ | §III-A 是相對 [24] 的擴展，§I 是對 [40] 的承襲；兩者可並存。[40] 是否已含此設定，本文未展開，待核其原文 |
| Kernel | [40] 單一 exponential | 1-/2-/3-Exp mixture 與 power-law 四族並選 | §III-D、§IV-A |
| 梯度 | [40] 的解析梯度 | 擴展到 exponential 與 power-law 兩族（含 penalty 梯度） | §III-H2、App. B |
| 溢出度量 | 無跨產品度量；VAR 文獻只做聚合頻率 | $G^{(x)}$ 與 $\Psi^{(x)}=(I-G^{(x)})^{-1}$ | §III-G、§IV-C |

代價是參數量：3-Exp 52 個、2-Exp 36 個、1-Exp 20 個、power-law 28 個（Table II）。收斂率隨之單調下降（99.5% / 98.1% / 95.6% / 82.4%）。

### 1.2 ⚡ Eureka

**近臨界對角結構會把微弱的跨產品耦合放大兩個數量級；所以判讀跨產品傳導必須看 $\Psi=(I-G)^{-1}$，不能看 $G$。** 直覺：一個 H 訂單平均直接觸發 0.021 個 Q 訂單，但它進入的 Q 盤本身有 0.90 的自激放大，且放大後的每個 Q 訂單還能回傳——幾何級數收斂在 $\rho\approx0.95$ 附近，所以「首輪係數可忽略」和「累積效應不可忽略」可以同時成立。

### 1.3 信息流

```
EPEX SPOT XBID L3 LOB (NL, 2024 四季首月)
        │  H + 其 4 個 Q 子合約
        ▼
時間對齊：只保留五個合約同時開放的窗口
        （以 12:00–13:00 交付小時為例，GCT 交錯：H 與 Q1 @11:00, Q2 @11:15, Q3 @11:30, Q4 @11:45；截掉非重疊段避免假零活動）
        │
        ▼
S(t) = 1{ LSI(t) ≥ q0.75 }   LSI = ⅓ z_spread + ⅓ z_vol + ⅓ z_QIabs
        q_glob = 0.264 ；slow-z 用 λ=0.01 的 EWMA；波動窗 10 分鐘
        │
        ▼
λ_e(t) = Σ_x 1{S(t)=x} ( μ_{x,e} + Σ_{e'∈{H,Q}} Σ_{t_i<t:e_i=e'} κ^{(x)}_{e'→e}(t−t_i) )
        κ = 3 個 exponential 分量（R=3）
        │
        ▼
懲罰 MLE：J(θ)=L(θ) − Σ_x ω_x (max{0, ρ_x − 0.95})²,  ω=10³,  L-BFGS-B, log 座標
        │
        ▼
G^{(x)}_{e'e} = Σ_r α/β   →   ρ_x（內生比例）   →   Ψ^{(x)}=(I−G^{(x)})^{-1}
        │
        ▼
方向 A=(Ψ_H→Q − Ψ_Q→H)/(Ψ_H→Q + Ψ_Q→H)　·　量級　·　時標 t50/t90/mean lag
```

## §2 · 數學層

📌 **Napkin Formula**

$$\lambda_e(t)=\sum_{x=0}^{1}\mathbf 1\{S(t)=x\}\Big(\mu_{x,e}+\sum_{e'\in\{H,Q\}}\sum_{t_i<t:\,e_i=e'}\kappa^{(x)}_{e'\to e}(t-t_i)\Big),\quad \kappa^{(x)}_{e'\to e}(\tau)=\sum_{r=1}^{3}\alpha^{(x,r)}_{e'\to e}e^{-\beta^{(x,r)}_{e'\to e}\tau}$$

$$G^{(x)}_{e'e}=\sum_{r}\frac{\alpha^{(x,r)}_{e'\to e}}{\beta^{(x,r)}_{e'\to e}},\qquad \Psi^{(x)}=(I-G^{(x)})^{-1},\qquad \text{stability: }\rho_x<1$$

**直覺**：$\rho_x$ 就是「這一 regime 下有多少比例的訂單流是市場對自己的反應」；$G$ 是首輪，$\Psi$ 是整個擴散簇的總和。$\rho_x\to1$ 時 $\Psi$ 爆炸，市場變脆。

**Loss / 訓練**：多會話對數似然（式 12，逐合約獨立擬合）減去 per-state 光譜半徑懲罰（$\rho^\star=0.95$，$\omega=10^3$，軟約束，超標才啟動）。log 座標參數化（$\log\mu,\log\alpha,\log\beta$），解析梯度見 App. B。指數核可遞迴更新（式 14），單次評估 $\mathcal O(NRE^2X)$；power-law 無遞迴表示，須截斷於 $\Delta_{\max}$，成本 $\mathcal O(N\bar WE)$。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部為「假設／示意」的玩具數字，用來手算機制，不是論文實證結果。** 論文的真實數值一律見 §5。輸入刻意取原文 §IV-C1 描述級的四捨五入值 0.87 / 0.02 / 0.05 / 0.90。

**Step 1 — 假設一組估計出來的 kernel 參數**（3 個分量，$\beta$ 分別取 0.05 / 0.5 / 5）：

| channel | 慢 α / (α/β) | 中 α / (α/β) | 快 α / (α/β) | 合計 $G$ |
|---|---|---|---|---|
| H→H | 0.005 / 0.10 | 0.20 / 0.40 | 1.85 / 0.37 | **0.87** |
| H→Q | 0 / 0 | 0.005 / 0.01 | 0.05 / 0.01 | **0.02** |
| Q→H | 0.0005 / 0.01 | 0.01 / 0.02 | 0.10 / 0.02 | **0.05** |
| Q→Q | 0.015 / 0.30 | 0.20 / 0.40 | 1.00 / 0.20 | **0.90** |

**Step 2 — 穩定性檢查**：$\operatorname{tr}G=1.77$，$\det G=0.782$，$\lambda_{\max}\approx(1.77+0.07)/2\approx0.92<1$ → 次臨界，懲罰項關閉。

**Step 3 — 算溢出乘數**：$I-G=\begin{pmatrix}0.13&-0.02\\-0.05&0.10\end{pmatrix}$，$\det=0.012$，故
$$\Psi=(I-G)^{-1}=\tfrac{1}{0.012}\begin{pmatrix}0.10&0.02\\0.05&0.13\end{pmatrix}\approx\begin{pmatrix}8.33&1.67\\4.17&10.83\end{pmatrix}$$

**Step 4 — 讀方向**：首輪 H→Q 只有 0.02，但全簇 $\Psi_{H\to Q}\approx1.67$——**放大了約 83 倍**；反方向 $\Psi_{Q\to H}\approx4.17$。

**Step 5 — 壓成一個數**：$A=(1.67-4.17)/(1.67+4.17)\approx-0.43$，負號代表總傳導由 Q 流向 H。（示意算術，與論文報的 $-0.442$ 同號同量級，但數值不同，不可互相替代。）

## §3 · 數據層

- **市場**：荷蘭 EPEX SPOT 連續日內（XBID / SIDC），level 3 limit order book。研究對象為 Hourly XBID 與其四個 Quarter-Hourly 子合約，兩者合計佔荷蘭 2024 年總交易量 86%（§II）。
- **時間**：2024 年各季第一個月（January / April / July / October），涵蓋季節變異。合約於交付前一日 14:00 CET 開盤；XBID 於交付前 1 小時關閉，intrazonal 於交付前 5 分鐘關閉。
- **樣本**：2,915 個 delivery contract，逐合約獨立擬合（§IV-A）。
- **單一 session 定義**：一個 Hourly 合約 + 其四個 Q 子合約，取最後五小時交易，再截斷到五者**同時開放**的重疊窗口；理由是納入非重疊段會製造假的零活動，嚴重偏誤基線與交叉核估計（§III-A）。
- **事件**：僅 order submissions，**不含 cancellations**（§V-A 說明：撤單與改單常落在同一毫秒）。
- **解析度**：毫秒級（EPEX SPOT 上限），小於 1 毫秒的訂單無法分離（§V-A）。
- **狀態構造**：LSI 的三個分量各佔 ⅓，皆用 slow-baseline z-score（EWMA，λ=0.01，約當 100 筆觀測的有效記憶）；波動窗 10 分鐘；queue imbalance 取絕對值。閾值取全樣本 75 百分位，$q^{\text{glob}}_{0.75}=0.264$。
- **容量假設**：未披露。所讀文本未見任何關於可交易規模、沖擊成本或委託簿深度的容量測試。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露（所讀文本未見代碼連結） |
| Checkpoint（擬合參數） | 未披露 |
| License | 論文文本標示 CC BY 4.0；**代碼授權未披露** |
| 複現難度 | 高：需 EPEX SPOT L3 LOB 授權數據、自建兩族解析梯度、per-contract L-BFGS-B |
| 數據可得性 | 商業數據（EPEX SPOT / XBID）；所讀文本未見公開數據集或衍生特徵 |
| 超參（部分可複現） | $\rho^\star=0.95$、$\omega=10^3$、$\lambda=0.01$、$q_{0.75}=0.264$、$\beta\in[10^{-4},1000.0]$、$\mu$ 下界 $10^{-4}$、初始 branching $b\sim\mathcal U(0.01,0.30)$；其餘 box 約束見 App. C |

## §5 · 評測 / Benchmark

**重要限定**：本文**沒有外部 SOTA 基線**，也沒有任何 IR / Sharpe / AR / MDD 之類的交易績效指標。下表「前 SOTA」欄取論文內部的最簡基線（1-Exp kernel），「本方法」欄取論文最終選用的 3-Exp SDHP。這是**核設定的消融**，不是與他人方法的排名。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Dutch XBID IDC, 四個月, 2,915 contracts | NLL/event（1-Exp→3-Exp；3-Exp 最低） | 1.2708 | 1.1950 | −0.0758 |
| 同上 | Median $W_1$ to Exp(1)（1-Exp→3-Exp；越小越好） | 0.0834 | 0.0651 | −0.0183 |
| 同上 | Kolmogorov–Smirnov 不拒絕率（pooled；1-Exp→3-Exp） | 0.856 | 0.926 | +0.070 |
| 同上 | Ljung–Box 不拒絕率（pooled） | 0.221（1-Exp） | 0.268（3-Exp） | +0.047 |
| 同上 | 收斂率（1-Exp→3-Exp） | 99.5% | 95.6% | —（由前兩欄比較） |

**逐條解讀（嚴格限於原文數值）**

1. **KS 通過率的提升是真 capability 還是過擬合？** 0.856 → 0.926 是所擬合合約上的殘差診斷改善，不能單獨區分泛化能力與過擬合。理想時間重置零分佈不含模型參數，但 Appendix A-C 明言，代入估計參數後的 KS／Anderson–Darling 零分佈不同於完全指定假說的表列分佈；跨規格比較統計量大小仍有意義，直接對照臨界值則有保留。正文通過率與附錄檢定保留意見之間的張力，本文未充分解釋。Power-law 似然另為截斷近似（§III-I），也不可與 exponential 族直接比 NLL。
2. **但沒有任何 kernel 是正確規格。** Ljung–Box 不拒絕率只有 0.221 → 0.268，原文自述「between 9% and 35%」且「no kernel considered here can be regarded as correctly specified」。殘差序列相關對多數合約仍存在——這是本頁認為**最容易被讀者忽略的一條**。
3. **3-Exp vs 2-Exp 的邊際增益很可能在雜訊內。** 原文：Median $W_1$ 由 0.0661 降到 0.0651，約 1.5%；殘差通過率差不到半個百分點；2-Exp 在 July 的三個殘差準則上還略優。選 3-Exp 的理由是「pooled 全贏 + 在 Stressed 與 Q 流上差距最大」，屬**判斷而非顯著性**。
4. **power-law 的 NLL/event 1.2480 不可與 exponential 族相比**（截斷似然），但它的殘差表現（$W_1$ 0.0963，Ljung–Box 0.144）與收斂率（82.4%）都最差，這部分是可比結論。
5. **成本／前瞻偏差**：原文完全沒有交易成本、滑價或 PnL 欄位。$\Psi$ 的放大（Table XIII 的 Amplif. 113 / 97）是**統計量而非可實現收益**，任何「用 $\Psi$ 做跨合約對沖」的推論都是本頁外推。
6. **估計量的中位數不可串接**：原文明確警告 $\Psi$ 是 $G$ 的非線性函數，故 $\Psi$ 的中位數不是中位數 $G$ 反推的 $\Psi$；Table XIII 兩欄分開算、不可互推。同理，中位數 $G$ 反算的 $\rho$ 也與報出的 $\rho_x$ 不同。
7. **異質性警告**：Table XIV 顯示 Normal regime 下 $\Psi_{Q\to H}$ 從 April 的 12.31 到 October 的 4.64，而 A 的月際 range 是 0.361。單看 pooled 點估計會嚴重誤判。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位到章節）

| 限制 | 出處 | 後果 |
|---|---|---|
| EPEX SPOT 時間戳僅毫秒；小於 1 毫秒的訂單無法分離 | §V-A | 最快 kernel 分量的 decay rate 在 12 to 15% 的合約上落在上界，該分量的**時標不可解析**（但其乘載的 branching mass 仍可識別） |
| 只建模 order submissions，排除 cancellations | §V-A | 提交與撤單常同毫秒；分兩個模型可避開同時事件，但會丟掉「加/撤流動性」之間的交叉效應 |
| power-law 在 $\Delta_{\max}$ 截斷，exponential 為精確 | §III-I、§V-A | power-law 似然是近似、與 exponential 不可比；$\rho_x$ 繼承了 $\Delta_{\max}$ 之外的尾部外推，$\eta\downarrow1$ 時最嚴重 |
| 只覆蓋 Hourly 與 Quarter-Hourly，排除 Half-Hourly | §V-A | Half-Hourly 流動性不足、每 regime 事件數不夠 |
| 跨產品 channel 的慢分量常落在 $\alpha$ 下界 | §IV-D2 | H→Q 為 48.3%、Q→H 為 49.4% 的合約（Normal 下）；快分量在 12 to 15% 的合約觸上界；慢分量半衰期 IQR 在 H→Q 跨 2.4 decades、Q→H 跨 3.8 decades → **個別分量時標「indicative rather than precise」** |

### 6.2 本頁推斷的隱含假設（明標推斷，非作者陳述）

1. **狀態外生性（推斷）**：$S(t)$ 被當成可觀測的外生路徑，但 LSI 由 spread / 波動 / queue imbalance 構成，這三者本身部分是訂單流的函數。若 LSI 與強度共同由同一批事件驅動，「壓力下結構改變」就可能部分是機械同義反覆而非獨立因果。原文用 slow-z（λ=0.01）與 75 百分位來緩解，但未做狀態內生性檢定。
2. **只建模提交 ⇒ 流動性抽取路徑可能被低估（推斷）**：若壓力期的微觀結構改變主要透過撤單實現，本文的事件空間看不到。
3. **逐合約獨立擬合、無參數共享（推斷）**：活躍度低的合約事件數少，估計不穩；原文承認 $\mu$ 在部分合約落在下界 $10^{-4}$，並說此時基線應讀作「外生份額的上界」——即**內生性結論若偏誤，方向是保守的**。
4. **近臨界估計對誤設高度敏感（推斷）**：$\rho\approx0.95$ 使 $\Psi$ 對 $G$ 的擾動極度敏感（$\Psi$ 的 Amplif. 達 113），任何殘餘序列相關都可能位移 $\rho$ 與 $\Psi$ 的量級。
5. **樣本外邊界（推斷）**：四個月、單一市場、單一年度。跨市場（德國、北歐）與跨年度（再生能源滲透率變化）外推未驗證。供給側或管制變動可能改變 regime 定義的意義。
6. **閾值穩健性（部分已驗證）**：$q^{\text{glob}}_{0.75}=0.264$ 是在全樣本上算的；原文以 leave-one-session-out 檢驗，stressed fraction 落在 $[0.18,0.32]$（mean 0.256, min/max 0.181/0.316），這部分屬**已披露的穩健性檢查**，不是本頁推斷。
7. **不是普遍不可能**：以上全部是「在本文的資料與事件空間下」。未測試（Half-Hourly、cancellations、微秒級數據、其他市場）不等於已證明無效。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Diebold–Yilmaz VAR 變異數分解 [41] | 聚合、低頻；無法處理事件級數據 | 是 | §I 明指不適用 |
| 多變量 Hawkes 溢出文獻 [44][45] | [44] 限門檻觸發的極端不流動事件；[45] 與 DCC-GARCH 聚合框架整合 | 部分 | §I 指其未覆蓋連續微觀結構 |
| 前作 SDHP [40] | 單一產品、單一 exponential kernel | 未披露 | 本文擴展到跨產品 + kernel 選擇 |
| Graf von Luckner & Kiesel [33] | 德國 IDC market order arrivals、單產品、參數隨 delivery hour 變 | 未披露 | 本文的對照組，非直接比較 |
| Chatziandreou & Karbach [34] | Hawkes + transient impact 推最優執行 | 未披露 | 本文完全不做執行策略 |

🎤 **Interview Tip**

問：「跨產品首輪係數只有 0.021，是不是可以說跨產品傳導可忽略？」

- **錯答**：「是，0.021 太小，兩本書基本獨立。」—— 這是把 $G$ 當成 $\Psi$，忽略了近臨界對角結構的幾何級數放大。原文說單看 $G$ 會低估約兩個數量級。
- **正確答**：「不能。$\rho\approx0.95$ 的近臨界結構下，$(I-G)^{-1}$ 會把首輪 0.021 放大到整簇量級（Normal 的 $\Psi_{H\to Q}$ 為 2.390）。而且這個放大倍數本身隨 regime 變（Amplif. 113 → 97），所以跨產品效應不是常數，平靜期校準的避險比率在壓力期會失真。另外注意 $\Psi$ 的中位數不能由中位數 $G$ 反推，兩者不可互相代稱。」

### 7.1 可證偽預測

- **2027 年前，若在 2025 年各季首月重估**：Stressed regime 的 $A$ 月際 range 應仍顯著小於 Normal 的 0.361（本文 Stressed 為 0.047）。若 range 不收斂，則「壓力下結構趨同」不成立。
- **若把 Half-Hourly 加入為第三個事件類型（$E=3$）**：Q→H 慢分量仍應高頻落在 $\alpha$ 下界（本文 49.4%）。若加入第三產品後該比例大幅下降，則慢分量是 kernel 過參數化而非真實時標。
- **若取得微秒級時間戳**：最快分量 $\beta$ 觸上界的合約比例應由 12 to 15% 顯著下降。若幾乎不變，則該上界反映的是資料生成機制而非錄製解析度。
- **若納入 cancellations**：預期 Stressed 的 $\Psi_{Q\to H}$ 下降幅度會與本文不同方向或量級——本文的「壓力下更對稱」若是靠提交端主導，加入撤單後應被削弱。此為本頁假設，尚待驗證。

## §8 · For the Reader

**因子研究員**：把「跨產品領先滯後」當成狀態依賴量而非常數。$A$ 與 $\Psi_{Q\to H}$ 有明確的 regime 條件依賴；用平靜期估計的領先關係做因子暴露，會在壓力期系統性失真。可直接拿的是 LSI 的構造方式（三個慢 z-score 各 ⅓）作為 regime 標籤特徵。

**高頻執行**：Q→H 的平均延遲由 6.05 秒塌縮到 0.93 秒（6.5×），$t_{90}$ 由 17.8 秒降到 2.38 秒。如果你的對沖假設是用數秒尺度的跨合約反應時間設計的，它在壓力期會過慢。H→Q 是唯一在壓力下**變慢**的通道（0.24 → 0.50 秒）而量級同時上升。

**組合配置**：不要把平靜期的跨合約對沖比率當常數。原文的核心經濟結論是領導結構在壓力下弱化（2.8 → 1.5），且**平靜期越極端的月份（April、January），壓力下被侵蝕得越多**（月際相關 −0.99）。這是一個「越依賴越不可靠」的結構。

**LLM-agent / 研究助理**：SDHP 在這裡是**結構感知模組**，不是 alpha 模型。如果你的 pipeline 需要一個「市場現在處於什麼傳導結構」的狀態輸入，$\rho_x$、$A$、$t_{50}$ 是三個低維、可解釋的介面量；但別把 $\Psi$ 的放大倍數當成收益預期。

**RL 策略**：$S(t)$ 可作為 observation 的一個通道；但 $\rho\approx0.95$ 意味著環境接近臨界，episode 邊界與 discount 的選擇會直接影響學到的行為（此為本頁推斷，原文無 RL 實驗）。用 $\Psi$ 而不是 $G$ 做跨合約特徵。

**研究學生**：兩個可直接搬走的方法論。(1) kernel 選擇用**參數無關**的時間重置殘差做跨族比較，別用似然——截斷會讓似然不可比。(2) 多會話 Hawkes 要按 $\rho_x$ 做 per-state 軟懲罰（$\rho^\star=0.95$），並在報告時附上 penalty 的實際量級。

## References

- Jhabli, AlSkaif, Bennin, Tekinerdogan, Pennings. *Cross-product Modelling in Continuous Intraday Electricity Markets via Hawkes Processes.* arXiv:2610.23619v1 [q-fin.TR], 2026-10-16. https://arxiv.org/abs/2610.23619v1
- 前作（本文的方法基礎）：Jhabli et al. *Modelling regime shifts in continuous intraday electricity markets with state-dependent Hawkes processes.* arXiv:2610.08169, 2026.（[40]）
- Morariu-Patrichi & Pakkanen. *State-dependent Hawkes processes and their application to limit order book modelling.* Quantitative Finance 22(3), 2022.（[24]）
- Hawkes. *Spectra of some self-exciting and mutually exciting point processes.* Biometrika 58(1), 1971.（[19]）
- Diebold & Yilmaz. *Better to give than to receive: predictive directional measurement of volatility spillovers.* IJF 28(1), 2012.（[41]）
- Bernstein. *Sur les fonctions absolument monotones.* Acta Mathematica 52, 1929.（[46]，冪律核作為指數核連續混合的理論依據）
- 來源鏈接：https://arxiv.org/abs/2610.23619
