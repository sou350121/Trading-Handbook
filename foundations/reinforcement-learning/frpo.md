<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=强化学习 alpha=端到端表征 autonomy=全自动黑盒 -->

# FRPO 解構（FRPO）

> **來源發布**：2026-11-05 · （無 venue） · arXiv [2611.07750](https://arxiv.org/abs/2611.07750)
> **arXiv 原文**：[Foresee to Trade: Forecast-Augmented Policy Optimization for Portfolio Management](https://arxiv.org/abs/2611.07750v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：量价表格 × 日频波段 × 强化学习 × 端到端表征 × 全自动黑盒。補的 prior gap 不是「預測更準」，而是**把 GRPO 的 group-relative baseline 從「同一 prompt 多樣本」搬到「同一市場路徑 + 同一初始帳戶多軌跡」**，以共同隨機數比較軌跡回報；它在期望條件下可降低共同因子創新的相對變異，但並非無條件消掉共同市場因子，也非保證 total policy-gradient variance 下降。同路徑比較讓 finRL 可拔掉最脆弱的 critic。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `强化学习` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①同一 encoder 上掛兩個頭：policy 頭做連續調倉，FSH 頭用未來 5 個交易日、橫截面中心化、歷史波動率歸一化的相對報酬當輔助監督；②核心 trick 是在同一歷史行情、同一初始帳戶下採多條軌跡做組內標準化優勢（critic-free），並用預測變異數 + 梯度對齊的 stop-gradient 門控調輔助權重；③對「強化學習 × 端到端表征」這條軸的意義是：**監督訊號可以來自未來訓練窗口，但推理路徑完全不碰未來**——FSH 只在訓練時存在；④DJIA30 CR 75.32，對比 PPO 的 54.70（單位 %，Table 1）；但 TOPIX Core30 的 Sharpe 0.624 低於 MASA 的 0.717，報酬優勢不等於風險調整全面勝出。

**X-Ray.** 這篇的真正貢獻被標題誤導了：預測頭（FSH）在推理時被整個丟掉，它不是 alpha 生成器，而是**表徵正則化器**——用未來相對報酬的梯度去塑形 encoder，讓 policy 看到更具方向感的 64 維特徵。真正動刀的是 §3.2：用 same-path group 比較替代 value network。這個選擇有原文自己的實證背書——Appendix C.2 的 PPO 診斷顯示最終 checkpoint 的 value 預測 explained variance 為 −0.2204（DJIA30）與 −0.0210（TOPIX Core30），即 critic 在高噪聲日頻報酬下比常數預測還差；同時高波動 regime 的 MSE 約為低波動 regime 的 2.1 / 2.2 倍。把 baseline 換成「同路徑同帳戶的橫截面均值」在工程上是對的。但它的 envelope 也很清楚：五個 setting 全部只在小型股票池（如 30 檔與 6 檔）、long-only 無槓桿、price-taker、1-bp per-side、單一 2019–2021 樣本外窗；High-Volatility Growth 的 CR 305.29 與 Defensive 的 70.23 差距，更像風格 beta 的暴露差異而非通用能力。對量化讀者：這是「critic-free + 輔助表徵學習」的乾淨模板，不是可直接上線的組合配置器。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（GRPO / Financial DAPO / FinRL-PPO） | FRPO 的改動 | 為何關鍵 |
|---|---|---|---|
| baseline 估計 | GRPO：同 prompt 多樣本組內標準化；PPO：learned critic + GAE | 同**市場路徑 + 同初始帳戶**的多條軌跡組內標準化，無 value network | 共同隨機數比較；避開 critic 擬合噪聲報酬 |
| 分組條件 | 離散 token 序列，prompt 相同 | 連續調倉，共享外生價格路徑與帳戶狀態 | 把 sequence-level 優勢廣播到每個 rebalancing step |
| 輔助監督 | DeepClair 把 forecast 接進 allocation；IMM 把 predictive signal 當 policy 輸入 | FSH **只在訓練**，推理只用 encoder + policy | 避免預測誤差污染執行路徑 |
| 輔助權重 | 固定權重或無 | Adaptive Gate：預測變異數 × 梯度對齊，stop-gradient | 輔助任務與 policy 梯度衝突時自動降權 |

### 1.2 ⚡ Eureka

**把 GRPO 的 group 當成「共同隨機數配對實驗」，而不是當成「便宜的 baseline」。** 直覺：同一段行情、同一初始帳戶下跑多條不同動作的虛擬帳戶，任何完全相同的加法回報項會在組內中心化後精確抵消；因此不靠 critic 判斷狀態價值，而是直接比較「同路徑下哪個動作相對好」。但原文明確限定：動態暴露下 residual factor-innovation variance 取決於 exposure dispersion，只有 \(D_{\rm dyn}<(G-1)C_{\rm dyn}\) 在期望層級成立時，same-path 的該變異數才低於 independent windows；它不保證共同市場 beta 全部消除，也「does not guarantee lower total policy-gradient variance」。

### 1.3 信息流

    行情特徵(16-step) + 帳戶狀態(cash/holdings)
            └────────────┬─────────────┘
                         ▼
            shared Transformer encoder  e_t ∈ R^64
              ├──────────────────────────────┐
              ▼                              ▼
       π_θ 高斯 → tanh → 帳戶約束       FSH g_ψ → (μ, logσ²)
       (賣不出持倉/買不超現金,            │  逐資產 N
        long-only, 無槓桿)               │
              ▼                          ▼
       執行 → 淨值 → 軌跡回報 R      ŷ = clip((u−ū)/ν, ±y_max)
              │                          │
              ▼                          ▼
       組內標準化 A^(g)            heteroscedastic Gaussian NLL
              │                          │
              └──── L_GR ────┐   ┌── α(σ², grad-align) · ℓ_FS ──┘
                             ▼   ▼
                     L_FRPO = L_GR − λ_H·H + E[α·ℓ_FS]
       推理：僅 encoder + π_θ（FSH 丟棄）

## §2 · 數學層

📌 **Napkin Formula**

    A_ω^(g) = (R_ω^(g) − R̄_ω) / (σ_R,ω + ε)              ← 軌跡級優勢，廣播到每一步
    α_k,j   = sg[ α_0 · exp(−u_k,j/τ) · clip((1+m_k)/2, 0.1, 1) ]
    L_FRPO  = E[−min(ρA, ρ̂A)] − λ_H·E[H(π)] + E[α·ℓ_FS]

複雜度：每次更新需多條完整 rollout（group size 5），無 critic 前向／反向；輔助支路只多 \(N\) 個輸出的 NLL，主力成本仍是 rollout 收集（每 seed 51,200 transitions）。

直覺三句：(1) 軌跡回報 \(R\) 是同路徑下不同動作的結果，組內中心化使相同的加法回報項抵消（附錄 D 將回報拆為 \(R_g=U_g+M_g\)，相同加性 \(M\) 在中心化後完全抵消）；(2) 組內標準化讓「相對排序 + 離散度」成為優勢，而 \(10^{-4}\) 的 reward scaling 主要影響與 FSH loss 的相對權重；(3) 門控降低「不確定性高」且「與 policy 梯度打架」的輔助樣本權重，stop-gradient 使其不反傳。

loss/訓練細節（逐字可查者）：\(\gamma=1\)；reward 為 \(10^{-4}\) 倍扣費後 NAV 增量；\(H=5\) 交易日；目標經橫截面中心化、歷史波動率歸一化、clip 至 \(\pm y_{\max}\)（**\(y_{\max}\) 數值未披露**）；FSH 用 heteroscedastic Gaussian NLL；\(\alpha_0=4\)；\(\tau\) 取第一個 gated minibatch 的 \(u\) 中位數、floored at \(10^{-8}\) 後固定；\(m_k=0.9m_{k-1}+0.1c_k\)。

## §2.5 · 帶數字走一遍（Worked Example，**全為示意假設，非論文實證**）

1. 假設某歷史片段 \(\omega\) 上，\(G=5\)（原文設定）條軌跡從同一現金／持倉出發，扣費後期末淨值增量為 \(V_T^{(g)}-V_0\)。以符號表示，不給具體金額。
2. 每步獎勵 \(r=10^{-4}\cdot\Delta V\)，\(\gamma=1\)，故 \(R^{(g)}=10^{-4}\cdot(V_T^{(g)}-V_0)\)。回報被縮到極小數量級。
3. 組內中心化：\(\bar R=(1/5)\sum R^{(g)}\)，得 \(R^{(g)}-\bar R=10^{-4}\cdot(V_T^{(g)}-\bar V_T)\)。
4. 標準化：\(A^{(g)}=(R^{(g)}-\bar R)/(\sigma_R+\epsilon)\)。分子與 \(\sigma_R\) 都會隨 reward scaling 同步縮放，但 \(\epsilon\) 不縮放；只有 \(\epsilon\) 相對於 \(\sigma_R\) 可忽略時，優勢才近似不變。原文未給 \(\epsilon\) 數值。
5. 同一 \(A^{(g)}\) 廣播到該軌跡的每個決策步，代入 clipped surrogate 得 \(\ell_{\rm GR}\)；同時 FSH 對每個 \((t,i)\) 輸出 \(\mu,\log\sigma^2\)，與目標算 NLL，乘上 \(\alpha_{k,j}\) 後與 \(\ell_{\rm GR}\) 相加回傳 encoder。
6. 輸出：一次梯度更新，encoder 同時被「哪條軌跡賺更多」與「未來 5 日相對報酬像不像」兩個訊號拉動。**reward scaling 不是讓優勢不變的充分條件；\(\alpha_0=4\) 才是輔助支路的顯式尺度開關。**

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 市場宇宙 | DJIA30、TOPIX Core30；另三組 6 檔組合：Mainstream（QQQ, SPY, AAPL, AMZN, Alphabet, MSFT）、Defensive（BAC, JNJ, JPM, PG, WMT, XOM）、High-Volatility Growth（AVGO, BRK.B, META, NFLX, NVDA, TSLA） |
| 頻率／決策 | 日頻，每日調倉；16-step 觀察序列 |
| 主切分 | 訓練 2013-01-01 ～ 2018-12-31；測試 2019-01-01 ～ 2021-08-31 |
| 平移切分 | w01：訓練 2013-03-01 ～ 2019-02-28／測試 2019-03-01 ～ 2021-11-30；w02：訓練 2013-06-01 ～ 2019-05-31／測試 2019-06-01 ～ 2022-02-28 |
| 特徵 | 公開 OHLCV 導出的技術指標 + 帳戶狀態（現金、持倉）；作者稱不需專有金融訊號 |
| 成本 | 統一 1-bp per-side；敏感性掃至 50 bp（Figure 5） |
| 技術／市場基線配置 | DIA 與 1311.T 作為 DJIA30 與 TOPIX Core30 市場代理；Mainstream 用 SPY，另兩組合 B&H 持有每檔 100 股（Baselines and Metrics，主文） |
| 初始資本 | **未披露**（文本僅列為共享設定） |
| 成分股 point-in-time | **所讀文本未見，圖表／附件待核**（生存者偏差風險） |

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD — Appendix E 僅稱 “We will release” code / configs / reproduction scripts / analysis scripts |
| Checkpoint | 未披露（僅述每 25,000 步存檔；無公開連結） |
| License | TBD |
| 複現難度 | 中。FinRL 框架 + 公開 OHLCV；但需 5 個 setting × 10 seeds × 51,200 transitions，且 TOPIX 使用不同 epoch（4）與 clip（0.2） |
| 數據可得性 | 價格資料公開；成分股快照與初始資本未見公開細節 |

## §5 · 評測 / Benchmark

前SOTA 欄填**所讀原文逐字給定的單一基線值**（不合成區間，不稱作者宣稱 SOTA）。CR/AR 單位為 %，比率為無量綱。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| DJIA30 | CR (%)（基線 PPO） | 54.70 | 75.32 | +20.62pp |
| DJIA30 | AR (%)（基線 PPO） | 17.35 | 22.78 | +5.43pp |
| DJIA30 | Sharpe（基線 A2C） | 0.695 | 0.796 | +0.101 |
| TOPIX Core30 | CR (%)（基線 MASA） | 39.25 | 48.50 | +9.25pp |
| TOPIX Core30 | AR (%)（基線 MASA） | 13.83 | 15.67 | +1.84pp |
| TOPIX Core30 | Sharpe（基線 MASA） | 0.717 | 0.624 | −0.093 |
| TOPIX Core30 | Sortino（基線 MASA） | 1.039 | 0.991 | −0.048 |
| Mainstream | CR (%)（基線 PPO） | 159.54 | 173.63 | +14.09pp |
| Mainstream | Sharpe（基線 MASA） | 1.673 | 1.607 | −0.066 |
| Defensive | CR (%)（基線 MASA） | 58.18 | 70.23 | +12.05pp |
| Defensive | Sharpe（基線 MASA） | 0.837 | 0.872 | +0.035 |
| High-Volatility Growth | CR (%)（基線 DeepAries-k3） | 232.80 | 305.29 | +72.49pp |
| High-Volatility Growth | Sharpe（基線 DeepAries-k3） | 1.721 | 1.604 | −0.117 |

**解讀。** 真 capability 的部分：**報酬維度在五個 setting 全勝**（原文亦以 25 個 dataset–metric 比較中 19 次第一、平均排名 1.28 佐證），且是十 seeds 均值，不是單一 seed。但**風險調整維度只在 DJIA30 與 Defensive 全面領先**；TOPIX Core30 的 Sharpe/Sortino 輸給 MASA，Mainstream 的 Sharpe 輸給 MASA，High-Volatility Growth 的 Sharpe/Sortino/Calmar 三項都輸給 DeepAries-k3。因此「全面最優」僅成立於 CR/AR，不可外推為風險調整能力。

機制歸因（不是全部歸給預測頭）：結構消融 Gap 為 DJIA30 +10.09、TOPIX +8.02、Mainstream +9.40、Defensive +12.84、High-Vol +80.78（單位 pp，原文逐字），且是對「較強的那個 ablation」取差；High-Vol 的 +80.78pp 遠大於其他 setting，合理懷疑混入風格 beta 而非決策品質（本頁推斷，待核）。same-path 的隔離實驗（DJIA30、關閉 FSH 與 Gate、十個 paired seeds）給出 Same-path CR 56.06 對 Independent windows 46.45，原文自報 9.61 個百分點的 paired 差——這是**單一市場、關閉輔助支路**下的結果，不能推成所有市場的必然因果。時序信用分配的對照（Trajectory 75.32 vs RTG 63.74 / Immediate 46.89 / GAE 41.43 on DJIA30）顯示「整條軌跡廣播」優於 reward-to-go 與 GAE，但這兩組對照同時改了優勢構造、折扣、歸一化與 critic 梯度，不能只歸因於「無 critic」。matched PPO-GAE（同 FSH + Gate）為 53.35，Full FRPO 75.32，仍保留差距，但也同樣是多項同時變動。

成本面向：原文稱 50 bp 下五個 setting 中四個保留「more than 95%」的 1-bp 報酬，Defensive 保留 84.9%；逐 setting 的 retention 曲線在 Figure 5（**圖未提取，數值待核**）。未見市場衝擊／容量估計。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **Limitations 段（正文 Conclusion 後）**：理論分析只覆蓋 non-anticipating 交易策略下「中心化因子創新」的變異數，且變異數排序只在對市場與動作隨機性的**期望層級**成立；實證基於 2019–2021 的歷史 OHLCV、技術指標與帳戶狀態，平移測試延伸至 2022-02；框架聚焦結構化金融訊號與資產級目標，FSH 變異數只刻畫金融目標的不確定性；未來可納入 multimodal Web 資訊、end-to-end language agents、使用者偏好建模。
- **§3.2 自述**：same-path 比較「does not guarantee lower total policy-gradient variance」；共享路徑的效應取決於共變結構。
- **Appendix D 自述**：\(V_{\rm same}<V_{\rm ind}\) 僅在 \(D_{\rm dyn}<(G-1)C_{\rm dyn}\) 時成立（期望層級，非每個實現的 group）；且「neither model guarantees lower total-return or standardized policy-gradient variance」；費用與其他回報項可與 \(M_g\) 共變。
- **§4.5/§4.6 自述的邊界**：Defensive 是成本最敏感的 setting；平移切分下風險調整指標的領先者會換人（w02 的 DJIA30 Sharpe 由 TD3 領先、Calmar 由 PPO 領先）。

### 6.2 推斷的隱含假設（本頁推斷，非作者陳述）

- **Regime 依賴**：測試窗 2019–2021（延伸至 2022-02）涵蓋特定多頭與高波動成長行情；High-Volatility Growth 的 CR 305.29 與 Defensive 的 70.23 落差，強烈提示結果對風格 beta 敏感。若換到橫盤或利率上行 regime，報酬優勢是否保留**未測**。
- **容量／衝擊**：全文採 price-taker，僅以 per-side fee 建模摩擦，未見 market impact 或 AUM 規模假設；小型股票池結論不可外推至高維選股池。
- **成本**：50 bp 的 retention 僅以圖呈現（待核），且 Defensive 的 84.9% 是逐字可查的最低點；更高換手策略在更高費率下的降解形狀未知。
- **數據泄漏**：FSH 標籤用未來 5 日價（\(h=\min(5,T-t)\)，僅在 rollout 邊界縮短），訓練窗內合法；但 shared encoder 會把未來資訊蒸餾進表徵，若成分股清單或特徵計算含事後修訂，即構成泄漏。point-in-time 成分股與特徵版本**所讀文本未見**。
- **Survivorship**：DJIA30 / TOPIX Core30 是否為測試期當時成分未說明；三組 6 檔組合（尤其含 NVDA、TSLA、META）事後選樣特徵明顯，這不是「輔助數據缺失」問題，而是**選樣本身即為處置變數**。
- **統計推論**：只報十 seeds 均值與排名，未見標準差、信賴區間或顯著性檢定；25 個比較取 19 次第一，未做多重比較校正。
- **輔助支路的歸因**：w/o Gate 與 w/o Gate & FSH 兩臂是巢狀移除，Gap 同時含「有無輔助監督」與「有無門控」兩種效應，不能拆成 FSH 的單獨貢獻。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| GRPO / DeepSeekMath（Shao et al. 2024） | 分組條件由 prompt 換成市場路徑 + 帳戶；離散 token → 連續調倉 | 是（方法公開） | 方法血緣 |
| Financial DAPO（Zha and Liu 2025） | 同樣 group-relative，但監督訊號來自新聞；FRPO 用未來相對報酬 + 無 critic | TBD | 最接近的同軸競品 |
| DeepClair（Choi et al. 2024） | forecast 直接進 allocation；FRPO 的預測頭推理時移除 | TBD | 監督方式分歧 |
| IMM（Niu et al. 2024） | predictive signal 當 policy 輸入 + expert guidance；FRPO 當表徵正則化 | TBD | 輸入 vs 訓練訊號 |
| DeepTrader / MASA / DeepAries / EarnMore | 皆有 critic 或特定策略結構（市場嵌入／多智能體風控／調倉間隔／可遮罩表徵） | 各自官方 repo（原文稱已復現並調參） | 實證對照 |
| FRPO 自身 | same-path + FSH + Gate 三件套 | 宣稱將開源 | 待釋出 |

🎤 **Interview Tip**
- **正確答**：「FRPO 的相對優勢主要來自 same-path 的共同隨機數分組與 critic-free 更新；FSH 只是訓練期的表徵塑形，推理時被移除——這從『FSH predictions are not inputs to the policy head』與 ablation 的 Gap 可以驗證。但同路徑分組只在特定期望條件下降低因子創新變異，且風險調整指標只在部分 setting 領先，TOPIX 的 Sharpe 輸 MASA。」
- **錯答**：「它把未來收益預測餵給 policy 所以更準」——直接與 §3.1 的推理路徑矛盾；或「它在五個市場全面 SOTA」——把 CR/AR 的勝利誤推廣到 Sharpe/Sortino/Calmar。

### 7.1 可證偽預測帶日期

1. **2027-06-30 前**：若第三方在比本文更大的股票池且加入 market impact 的設定下複現，FRPO 相對 PPO 的 CR 領先幅度不應外推至 DJIA30 上 Full FRPO 對 PPO 的 Table 1 水平；若仍維持同量級領先，則「小宇宙／低摩擦」的 envelope 假說被否證。
2. **2026-12-31 前**：若釋出的 code 可跑，關閉 FSH 但保留 same-path（即 w/o Gate & FSH 臂）在 High-Volatility Growth 應重現 224.51 附近的 CR；若復現值遠低於此，代表原結果對 seed 或資料快照敏感。
3. **可立即檢驗**：把 w02 的風險調整排名變動（DJIA30 Sharpe 由 TD3 領先）當成 regime 敏感性的前哨——若後續切分持續出現「報酬贏、Sharpe 輸」的分裂，則「輔助監督提升決策品質」的敘事應降級為「提升尾部暴露」。

## §8 · For the Reader

- **因子研究員**：FSH 的標籤工程可直接借——橫截面中心化去掉市場方向、除以歷史波動率做風險調整、再 clip 控制厚尾，這是把「相對強弱」變成穩定監督訊號的標準三步。但注意它只是輔助任務；若要當 alpha，需自己驗證 IC 與換手，原文未給 FSH 的樣本外預測指標。
- **高頻執行**：不適用。日頻、1-bp per-side、price-taker、無 impact 模型；Defensive 在 50 bp 下只剩 84.9% 是全文最刺眼的成本訊號。要上線得先補周轉率與沖擊成本曲線。
- **組合配置**：把 FRPO 當「風格配置顧問」而非「選股器」——它的強項是在固定 6 檔籃子裡決定曝險時序（High-Volatility Growth 的 CR 305.29 vs B&H 62.47 是風格擇時能力的體現，也是 beta 暴露的體現，兩者無法從所讀文本分離）。若要放進多資產，先做 regime 分層回測。
- **RL 策略研究者**：same-path grouping 是可搬遷的設計模式——任何「環境外生、動作影響狀態」的連續控制任務都可考慮用它替代 critic；附錄 D 的 \(C_{\rm dyn}/D_{\rm dyn}\) 分解提供了「何時有效」的可檢驗條件 \(D_{\rm dyn}<(G-1)C_{\rm dyn}\)，這比多數實證論文的口號式辯護誠實。
- **LLM-agent 開發者**：Adaptive Gate 的「梯度對齊 + 預測不確定性」雙因子加權，可直接移植到多任務微調；stop-gradient 確保門控本身不被最佳化，是可複用的工程細節。
- **研究學生**：這篇的寫法值得學——每個機制都配了一個「關掉它」的對照臂（w/o Gate、w/o Gate & FSH、Independent windows、RTG/Immediate/GAE、PPO-GAE matched），但每個對照都同時改動多個因子，是可複製的實驗設計，也是可批評的歸因弱點。

## References

- 原論文：Liu, Feng, Xu, Lai, Li, Zhang, Chen, Zhang, Pang, Guo. *Foresee to Trade: Forecast-Augmented Policy Optimization for Portfolio Management.* arXiv:2611.07750v1 [cs.LG], 2026-11-05. https://arxiv.org/abs/2611.07750v1
- 方法血緣：Shao et al., *DeepSeekMath* (GRPO), arXiv:2402.03300, 2024；Schulman et al., *PPO*, arXiv:1707.06347, 2017；Zha & Liu, *A New DAPO Algorithm for Stock Trading*, IEEE IDS 2025。
- 共同隨機數：Glasserman & Yao, *Some Guidelines and Guarantees for Common Random Numbers*, Management Science 38(6), 1992。
- 環境與基線：Liu et al., *FinRL*, ICAIF 2021；Liu et al., *FinRL-Meta*, NeurIPS 2022；Wang et al., *DeepTrader*, AAAI 2021；Li et al., *MASA*, AAMAS 2024；Kim et al., *DeepAries*, CIKM 2025；Zhang et al., *EarnMore*, WWW 2024。
- 預測式監督對照：Choi et al., *DeepClair*, CIKM 2024；Niu et al., *IMM*, IJCAI 2024。