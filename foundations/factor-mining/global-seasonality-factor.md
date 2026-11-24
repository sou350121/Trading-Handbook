<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# Global Seasonality Factor 解構（Global Seasonality Factor）

> **來源發布**：2026-11-11 · arXiv · arXiv [2611.15657](https://arxiv.org/abs/2611.15657)
> **arXiv 原文**：[Global Seasonality Factor Investing](https://arxiv.org/abs/2611.15657v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 日頻波段 × 監督回歸 × 因子挖掘 × 人機協同可解釋」。它補的是季節性文獻長期停留在個別日曆異象、缺統一因子框架與淨成本 OOS 的 gap：把五種日曆週期做成逐市場、非線性、可回測的全球期貨因子。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文；細節待升 v1。
**TL;DR:** ①把 day-of-week／day-of-month／day-of-quarter／day-of-half-year／day-of-year 五種日曆特徵，逐市場擬合成 50 個全球期貨市場的季節性因子。②Trick 是每個市場×特徵×horizon 用 Gaussian kernel（bandwidth 0.50）做非參數條件期望，再跨有效 horizon 等權集成。③對 **alpha=因子挖掘** 軸★的意義：日曆位置是確定性、非價格衍生的資訊，因子與 carry/momentum/skew 低相關，可作分散化組件。④加入 seasonality 後，EW carry+momentum+skew 的淨 Sharpe 由 0.80 升至 1.04（原文稱 approximately 30%）。

**X-Ray.** 它不在更高頻或更大量價特徵上競爭，而是在**特徵維度**開新供給：日曆位置不從價格衍生，天然與 carry/momentum/skew 低相關（Table 7：0.0415／0.0092／0.0221）。它解掉的舊工程坑是季節性長期停在描述性異象、難以逐市場樣本外交易；kernel 平滑讓相鄰日曆日共享樣本，把每個市場的季節形狀變成可回測倉位。打不開的 envelope 也清楚：快週期 gross 強但淨成本後為負（day-of-week 淨 approximately -4.0、day-of-month approximately -0.50），慢週期才留下淨值，且集中於少數部門。對量化讀者的意義是：這是低相關、容量假設有限的分散化 alpha，不是獨立高 Sharpe 策略；真正 capability 在增量 Sharpe 與每期改善，而非絕對報酬。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作 / 傳統 | 本文改動 |
|---|---|---|
| 模型範圍 | traditional factor modelling 以單一模型跨所有市場擬合 | 逐市場（individual market level）擬合，讓每個 feature-market pair 表達獨特季節形狀 |
| 函數形式 | 傳統線性時序模型；Moskowitz et al. [33] 已引入非線性因子模型 | 非參數 Nadaraya–Watson kernel regression（Gaussian，bandwidth 0.50） |
| 季節性表徵 | 文獻多為個別日曆異象（January effect、sell in May 等） | 五種日曆特徵 × 有效 horizon 配對，排除 day-of-week 後做等權 ensemble |
| 評估 | 本文未提供前作同口徑的淨成本 OOS 對照 | walk-forward OOS（1995-01-01–2026-06-30）、淨成本、縮放至 20% vol |

### 1.2 ⚡ Eureka

**一句話 trick**：把「某市場在某個日曆位置的平均未來報酬」當成一個獨立的非參數回歸問題——用 Gaussian kernel 在日曆整數日上做局部加權平均，得到該市場該日曆位置的條件期望，再縮放成 vol-adjusted 倉位。

**直覺**：季節性不是全局常數，而是每個市場自己的形狀；kernel 平滑讓相鄰日曆日共享樣本，換來估計穩健性，代價是可能抹掉尖銳但真實的局部訊號。bandwidth 0.50 是這個 trade-off 的固定旋鈕。

### 1.3 信息流

price_t → r_t → 40-day EMA vol → y_t = (1/√m)Σ r_{t+i+l}/σ̂_{t+i}，l=2  
　↓  
減 drift μ → s_t（標準化至約零均值、單位方差）  
　↓  
feature x_t ∈ {0–4, 0–30, 0–91, 0–182, 0–365}  
　↓  
Nadaraya–Watson：ŝ_t = Σ_j ω_b(x_j−x_t)·s_j，Gaussian b=0.50，逐市場  
　↓  
per feature × horizon m∈{1,5,21,42,63,84,105,126}，forecast length < feature periodicity  
　↓  
跨 horizon 平均 → m-day MA 平滑 → ÷40-day vol → 倉位  
　↓  
daily rebalance，次日收盤執行，20% vol 目標，淨成本

## §2 · 數學層

📌 **Napkin Formula**

$$\hat s_k^{(t)}=\sum_{j=1}^{T}\omega_b(x_k^{(j)}-x_k^{(t)})\,s_k^{(j)},\quad \omega_b=\frac{\kappa_b(\cdot)}{\sum_i\kappa_b(\cdot)},\quad \kappa_b(x)=\frac{1}{\sqrt{2\pi}}e^{-0.5\|x/b\|^2},\ b=0.50$$

**直覺**：Nadaraya–Watson 是局部常數回歸，輸出是鄰近日曆日報酬的凸組合；無顯式 loss、無參數訓練，屬 lazy learner。**複雜度**：原文未給複雜度；推斷預測時需對訓練集所有點加權，O(T) per query（原文未提加速）。**訓練細節**：bandwidth 0.50 固定；所讀文本未見 bandwidth 的樣本內選擇程序或正則化設定。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下為**機制示意步驟，非論文實證數值**；單點權重與輸出取決於實際樣本，原文未給單點示例，數值一律「未披露」。

1. 取某市場 k、特徵 day-of-quarter（x∈0–91）、horizon m=21（在 {1,5,21,42,63,84,105,126} 內，且嚴格小於該特徵週期）。
2. 訓練集所有 (x_j, s_j)，s_j 已去 drift μ 並標準化至約零均值、單位方差（實際樣本值未披露）。
3. 對查詢日 x_t，算 d_j=x_j−x_t，權重 ω_j ∝ exp(−0.5‖d_j/0.50‖²)，再歸一化使 Σω_j=1。
4. 輸出 ŝ_t=Σω_j s_j；這是該市場在該日曆位置的條件期望，量級取決於 s 的單位方差與平滑程度（單點輸出未披露）。
5. 對所有有效 horizon 取平均，經 m-day MA 平滑，除以 40-day vol estimate，縮放至 20% vol 目標；每日 rebalance、次日收盤執行。

## §3 · 數據層

- **樣本**：50 個全球期貨市場（Table A1），涵蓋 bond／currency／energy／equity／grain／metal／soft。
- **頻率/時段**：日頻，CSI 提供，最早 January 1950 至 June 2026；日曆 basis roll 至最近最流動合約。
- **訓練/OOS**：所有可用數據 prior to January 1, 1995 起訓；市場需累積至少 10 年訓練數據才入組合；OOS 為 January 1, 1995–June 30, 2026；每年 refit，embargo m+1 days。
- **容量假設**：Table A1 成本（tick size、slippage、commission）估計自 AUM approximately \$100 million USD、年化波動 approximately 20% 的實盤策略；更大容量的滑點未披露。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| 所讀文本未見 | 所讀文本未見 | 所讀文本未見 | TBD（kernel + walk-forward 邏輯簡單，但 50 市場拼接與成本表是門檻） | 需 CSI 訂閱；公開數據包所讀文本未見 |

## §5 · 評測 / Benchmark

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 50 global futures, OOS 1995-01-01–2026-06-30 | Net Sharpe（同一 global factor portfolio：EW carry+momentum+skew，加入 vs 未加入 seasonality） | 基線：EW carry+momentum+skew = 0.80 | 加入 seasonality 後 = 1.04 | +0.24 |
| 同上 | Net Sharpe（seasonality ensemble） | 未披露（carry/momentum/skew 各自淨 Sharpe 所讀文本未見） | 0.62 |  |
| 同上 | Gross Sharpe（seasonality ensemble） | 未披露 | 0.85 |  |
| 同上 | Net Sharpe（day-of-year） | 未披露 | 0.39 |  |
| 同上 | Net Sharpe（day-of-week） | 未披露 | approximately -4.0 |  |
| 同上 | Net Sharpe（day-of-month） | 未披露 | approximately -0.50 |  |

**解讀**：來源未把 carry/momentum/skew 稱作 SOTA；此表以基線處理。唯一可算 Δ 的是同一 global factor portfolio 加入 seasonality 前後（0.80→1.04），其餘列只列來源水平，不跨組合硬減。真正的 capability 是低相關與每期改善：Table 7 的 Season 與 Carry/Momentum/Skew 相關係數為 0.0415／0.0092／0.0221，且原文稱加入後每個 evaluation period 淨 Sharpe 都改善。成本已計且懲罰快週期：day-of-week 淨 approximately -4.0、day-of-month 淨 approximately -0.50；day-of-quarter 淨 approximately zero（2005 後 approximately 0.30），day-of-half-year 0.33、day-of-year 0.39。過擬合/多重比較風險：有效 feature–horizon 配對受 forecast length < feature periodicity 限制（例如 day-of-week 只配 daily、day-of-month 只配 daily/weekly），但仍有多個部門與 horizon 的回歸；所讀文本未見顯著性校正。Alpha 回歸（Table 1–6）的 9.32%／15.38%／10.87%／11.65%／11.35%／15.77% 是 seasonality 對三因子回歸的截距，不是打敗 SOTA 的績效差，不可與 Sharpe 混用。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

- **§4 Conclusion**：連續調整時間序列 roll 至最近最流動合約，商品市場可能拼接不同庫存／生產／作物生命週期的價格；建議改用相同到期月份或 calendar spreads 隔離更強季節性訊號。
- **§4 Conclusion**：季節性可作為 conditional feature 增強既有時序因子關係（未在本文實作）。
- **§4 Conclusion**：可能存在其他季節性異象（衍生品到期、事件驅動）值得研究（本文未測）。

### 6.2 推斷的隱含假設（本頁推斷，非作者原文）

- **Regime 依賴**：原文稱 seasonality 在 2005–2014 表現突出，most recent decade 相對其他時序因子 underperform；因此 ensemble 的增量不是跨 regime 的必然。
- **成本敏感**：快週期 net 為負，隱含「無執行優勢者不應碰 day-of-week／day-of-month」；成本結構改變會翻轉結論。Table A1 成本對應 AUM approximately \$100 million USD、年化波動 approximately 20%，更大容量未披露。
- **數據泄漏**：walk-forward + embargo m+1 days 已處理重疊；但 bandwidth 0.50 若在 full sample 選定，存在 subtle leakage（所讀文本未見選參程序）。
- **Survivorship／市場納入**：50 市場清單與至少 10 年訓練數據的納入規則來源有述，但退市或流動性不足市場如何處理所讀文本未見；不能把 Table A1 當作無生存者偏差的證明。
- **多重比較**：有效配對受 forecast length < feature periodicity 限制，但仍有多個特徵、horizon 與部門回歸；所讀文本未見顯著性校正。單一顯著 alpha 不宜寫成所有市場／所有期間的必然因果。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Time series momentum（Moskowitz et al. 2012；2025 nonlinear） | 價格動能 vs 日曆位置；本文借其非線性框架但特徵不同 | No（所讀文本未見 repo） | 文獻存在 |
| Carry（Koijen et al. 2018；Butler & Butler 2023） | 期限結構 vs 日曆 | No | 已發表 |
| Skew（Baltas & Salinas 2022） | 三階矩 vs 日曆 | No | 已發表 |
| 傳統季節性異象（Thaler 1987；Bouman & Jacobsen 2002；Heston & Sadka 2008） | 個別市場／個別異象 vs 統一因子框架＋walk-forward 淨成本 | No | 已發表 |

🎤 **Interview Tip**
- **正確答**：核心不是「發現季節性」，而是「逐市場非線性 kernel + 有效 feature-horizon 配對 + 等權 ensemble + 淨成本 OOS」。關鍵實證：day-of-week 淨 approximately -4.0、day-of-month 淨 approximately -0.50；day-of-quarter 淨 approximately zero（2005 後 approximately 0.30），day-of-half-year 0.33、day-of-year 0.39；ensemble 淨 0.62，加入 global factor portfolio 後 0.80→1.04，且與三因子相關 <0.05。
- **錯答**：說「seasonality 打敗 momentum／carry／skew」——原文只稱 comparable，未給三者各自 Sharpe；或說「慢週期都淨正」——day-of-quarter 是例外；或說「day-of-week 可交易」——其淨 Sharpe approximately -4.0。

### 7.1 可證偽預測（本頁推斷，檢驗區間用來源 OOS）

- 以相同 50 市場、b=0.50、Table A1 成本，在 1995-01-01–2026-06-30 複現，day-of-year 淨 Sharpe 應接近 0.39；若顯著偏離，優先檢查 roll 規則、40-day vol 估計與成本映射。
- 若 turnover 未顯著下降，day-of-week 淨 Sharpe 仍應為負（approximately -4.0）；若為正，表示執行或成本假設不同。
- 若 bandwidth 改以 nested CV 選擇，ensemble 淨 Sharpe 可能偏離 0.62；方向未披露，待驗證。

## §8 · For the Reader

- **因子研究員**：可借鑑「逐市場 kernel + 有效 horizon 等權」的結構，把日曆或事件特徵接進來；注意 bandwidth 選參與多重比較校正。
- **高頻執行**：day-of-week／day-of-month 的淨值負值說明成本是殺手；除非有執行優勢，否則優先慢週期（day-of-year、day-of-half-year），並避開 day-of-quarter 的淨值例外。
- **組合配置**：seasonality ensemble 與 carry／momentum／skew 相關 <0.05，是分散化候選；但 recent decade underperform，且成本表僅對應 approximately \$100 million AUM，不宜重倉。
- **LLM-agent／RL 策略**：可把日曆位置當 conditional feature 餵給現有模型（作者亦建議此方向），而非當獨立 alpha。
- **研究學生**：這是「描述性異象 → 可交易因子」的完整 pipeline 範例，含成本、OOS、embargo，適合拆解學習；但勿把單一 alpha 顯著性外推為普遍因果。

## References

- Butler, A., Butler, A., & Zawadzki, M. (2026). *Global Seasonality Factor Investing*. arXiv:2611.15657. [連結](https://arxiv.org/abs/2611.15657v1)
- Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics* 104(2).
- Moskowitz, T. J., Sabbatucci, R., Tamoni, A., & Uhl, B. (2025). Nonlinear time series momentum. SSRN.
- Butler, A., Zawadzki, M., & Butler, A. (2026). Idiosyncratic market effects in global time series momentum. ReSolve Asset Management.
- Koijen, R. S. J., Moskowitz, T. J., Pedersen, L. H., & Vrugt, E. B. (2018). Carry. *Journal of Financial Economics* 127(2).
- Butler, A., & Butler, A. (2023). Managed futures carry: a practitioner’s guide. ReSolve Asset Management.
- Baltas, N., & Salinas, G. (2022). Cross-asset skew. *The Journal of Portfolio Management*.
- Nadaraya, E. A. (1964); Watson, G. S. (1964). Kernel regression 原始文獻.
- Thaler, R. (1987); Bouman, S., & Jacobsen, B. (2002); Heston, S. L., & Sadka, R. (2008) — 季節性異象 lineage.