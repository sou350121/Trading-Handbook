<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=风险择时 autonomy=人机协同可解释 -->

# NRM 解構（NRM）

> **來源發布**：2026-11-05 · ICAIF 2026 · arXiv [2611.07287](https://arxiv.org/abs/2611.07287)
> **arXiv 原文**：[Neural Realized Measures: Forecast-Supervised Intraday Aggregation for Volatility Forecasting](https://arxiv.org/abs/2611.07287v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：NRM 把「已實現測度的日內聚合」從預測不可知的固定預處理，改成由下游預測目標監督的可學習加權；它保留 RV／BV／GK 原語與 bar 級歸因，只放開聚合權重。五軸落點是「量價表格 × 日頻波段 × 監督回歸 × 風險擇時 × 人機協同可解釋」，解的是經典 RRM「固定 `w=1`」與端到端網路「丟失統計原語」之間的 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文可讀全文文本；圖 1–3 未提取，圖表／附件待核。細節待升 v1。
**TL;DR:** ① 提出 NRM，在 RV／BV／GK 原語序列上學習日內聚合權重；`w=1` 精確退化回 RRM。② 核心 trick 是正、均值歸一、上下文相關的 softmax 權重，加上 QLIKE 與 KL-to-uniform 正則聯合訓練；Proposition 3.1 把偏離 RRM 的幅度綁在日內原語離散度與 KL 上。③ 對「風險擇時 × 人機協同可解釋」軸的意義是：學習的是權重而非黑箱表徵，`w_{t,k}x_{t,k}` 可直接審計。④ 在六個加密貨幣、五分鐘資料、SF／HAR／MIDAS 三頭下，54 個配對比較的 QLIKE 全部改善，均值增益為 7.23%、8.15%、9.22%。

**X-Ray.** 放回五軸 Pareto，NRM 不是與 HARNet／DeepVol 爭表徵上限，而是在測度構造階段做「最小可學習干預」：不動原語、不動下游頭，只放開日內聚合。它解的舊坑有兩個：經典 RRM 對所有 bar 一視同仁，忽略日內位置、局部對比與當日分布；端到端網路雖靈活，卻丟掉已實現測度的統計結構與嵌套基準。KL-to-uniform 正則與 Proposition 3.1 的界讓「偏離 RRM」有可檢查的上界，但這只限制偏離幅度，不保證方向正確或樣本外必然勝出。本文證據是 QLIKE 54 個配對全勝、RMSE 17/18、MAE 15/18–18/18（分頭逐列），不能外推成所有指標、所有市場的結構性保證。它打不開的 envelope 也清楚：STLG 只用 OHLC 與原語衍生量，作者自陳排除成交量與訂單簿；驗證只在加密 24/7 UTC 日結構上完成，對帶交易時段與隔夜缺口的傳統市場沒有證據。對量化讀者的意義：這是一個「有界偏離 + 可證偽機制」的乾淨範式，適合遷移到時段結構明確的市場，但遷移前必須先處理交易日曆與 gap 的一致性。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 面向 | 經典 RRM（Eq. 1） | NRM（Eq. 2, 9–10） | 機制 |
|---|---|---|---|
| 日內聚合 | 固定 `w_{t,k}=1`，預測不可知 | 正、均值歸一、上下文相關的 `w^m_{t,k}`，由 STLG 網路輸出 | 均勻權重精確退化為 RRM |
| 原語 | RV `r^2`、BV `(π/2)|r_t||r_{t-1}|`、GK `½ρ²−(2ln2−1)r²` | 完全不變 | 保留統計結構與 bar 級歸因 `w·x` |
| 學習目標 | 先驗式測度構造，無預測動機 | 與下游頭（SF／HAR／MIDAS）聯合訓練 | QLIKE + KL-to-uniform 監督聚合 |

### 1.2 ⚡ Eureka 一句話 trick + 直覺

**Eureka 句**：「把聚合權重放在 softmax 上、再加均值歸一約束，讓均勻權重恰是特例；用 KL-to-uniform 正則懲罰偏離，再用 Proposition 3.1 的界把偏離綁在原語離散度上。」

直覺：原語決定「每根 bar 貢獻什麼」，聚合規則決定「這些貢獻怎麼壓成日統計量」。傳統管線把第二件事凍結在預測目標之前；NRM 只把第二件事交給預測目標，第一件事原封不動。因為輸出仍是原語的顯式加權和，`w_{t,k}x_{t,k}` 可以直接看——既不丟可解釋性，也不丟 RRM 作為嵌套基準。

### 1.3 信息流 ASCII 圖

OHLC bar (t, k)
   ├─ r_{t,k}=C−O, ρ_{t,k}=H−L ──► x^m_{t,k}  (RV / BV / GK primitive)
   └─ Shape (BR,UR,LR) + Time (sin,cos) + Local d^m_{t,k} + Global (med,sd)
                │                              │
                └────────► u^{m,STLG}_{t,k} ◄─┘
                              │
                        score net g_θ（GELU, dropout 0.1）
                              │  s^m_{t,k}
                        softmax τ=1，均值歸一
                              │  w^m_{t,k},  (1/K)Σw=1
                              ▼
                    X^m_t = Σ_k w^m_{t,k} x^m_{t,k}
                              │
                downstream head（SF / HAR / MIDAS，各自獨立一套 NRM）
                              │
                         ŷ_{t+1}=log V̂_{t+1}

## §2 · 數學層

📌 **Napkin Formula**

X^m_t = Σ_{k=1}^{K} w^m_{t,k} x^m_{t,k},  w^m_{t,k} = K · exp(s^m_{t,k}/τ) / Σ_j exp(s^m_{t,j}/τ)
s.t. (1/K)Σ_k w^m_{t,k} = 1, τ=1, s^m_{t,k}=g_θ(u^{m,STLG}_{t,k})
Loss = QLIKE(V_t, V̂_t) + λ · (1/|B|)Σ_t D^m_t,  D^m_t = (1/K)Σ_k w^m_{t,k} log w^m_{t,k}

**複雜度**：每日 K 根 bar（本文 K=288），每根 8 維 STLG 輸入經小 MLP（隱維 [8,4] 或 [16,8]）→ 每次 softmax O(K)；日總計 O(K·d + K)。訓練用 AdamW、batch 16、最多 200 epochs、patience 20、lr 3×10⁻³。

**直覺**：softmax 把分數變成機率，再乘 K 得均值為 1 的「相對權重」——均勻分數即 `w=1`。`w>1` 上調、`w<1` 下調；偏離由 KL 懲罰控制，KL 正則係數 λ 從 {3×10⁻⁴, 10⁻³, 3×10⁻³, 10⁻², 3×10⁻²} 驗證集選。QLIKE 用於預測目標（`ŷ=log V̂`），KL-to-uniform 用於聚合層正則。Proposition 3.1：`|X^m_t − R^m_t| ≤ K·Δ^m_t·√(D^m_t/2)`，`Δ^m_t = max_k x − min_k x`，故偏離受原語離散度與 KL 聯合上界；`D=0` 若且唯若 `w=1`。

## §2.5 · 帶數字走一遍（Worked Example，示意）

以下數字皆為**假設／示意**，不是論文實證；論文真實數字在 §5。為便於手算，取 `K=4`（論文本實驗 `K=288`）。

1. **假設** 4 根 bar 的原語貢獻 `x = [1.0, 4.0, 0.5, 1.5]`（RV 示意），STLG 網路輸出分數 `s = [0.1, 0.3, 0.0, 0.1]`。
2. **均值歸一 softmax**（`τ=1`）：
   - `exp(s) = [1.105, 1.350, 1.000, 1.105]`，總和約 4.560
   - `w = K·exp(s)/Σexp(s) = 4·exp(s)/4.560`，約 `[0.969, 1.184, 0.877, 0.969]`
   - 檢查：`(1/4)·Σw` 約 1.000
3. **聚合**：`X = 0.969·1.0 + 1.184·4.0 + 0.877·0.5 + 0.969·1.5`，約 7.598；對照 RRM `R = Σx = 7.000`。
4. **偏離 vs 界**：`Δ = 4.0 − 0.5 = 3.5`；`D = (1/4)Σ w log w`，約 0.006；上界 `K·Δ·√(D/2)` 約 0.778；實際 `|X−R|` 約 0.598，小於上界。
5. **解讀**：分數高的那根 bar（原語最大）被上調、其它被下調；均勻分數（`s` 全相等）會讓 `w=1`、`X=R`——這是 RRM 被嵌套的機制來源。

## §3 · 數據層

**市場／頻率**：Binance 現貨價格指數 5 分鐘 OHLC，6 幣：BNB、BTC、DOGE、ETH、SOL、XRP。現貨價格指數聚合主要中心化交易所，降低單一場地流動性／買賣價差敏感度，但不消除全部微觀結構噪聲。

**樣本與切分**（依 §4.2，以 target date 定義）：
- 各資產從 2020-01-01 後首個完整日起，至 2026-05-31
- 驗證：2023-06 至 2024-05；測試：2024-06 至 2026-05
- 完整日 K=288 bars；不完整日剔除
- 目標：`y_{t+1} = log R^{RV}_{t+1}`（次日五分鐘報酬 RV），預測 `V̂ = exp(ŷ)`，不做 smearing／clipping
- 所讀文本未逐資產列出訓練期端點；不可由驗證期反推訓練期

**模擬**（§4.1）：Heston + Li 等 (2025) 事件壓力，事件位置均勻抽取防止 NRM 學到確定性時序；三機制——Continuous、Jump（compound-Poisson）、Gradual jump（短暫持續位移）。
- 參數：X₀=log(1200)、μ=0.05、κ=5、θ=0.0225、η=0.4、ρ=−1/√2、jump intensity λ=1/5、jump-size standard deviation 0.9%
- Euler Δt = 5/(252×24×60)，每日 288 根 5 分鐘 bar
- 每機制 2,373 模擬日，按時序 60% training、20% validation、20% test
- 10 個 Monte Carlo 種子（同時變路徑與神經初始化）
- 目標：同期 `IV^{oracle}_d = ∫_d^{d+1} σ_s² ds`；次日 `IV^{oracle}_{d+1}`

**樣本外與容量假設**：No test observation is used for fitting, specification selection, or seed selection。每日至多 288 根 bar；加密 24/7 共同 UTC 日邊界，無隔夜缺口——作者自陳此便利性同時限制外部效度（§7）。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD（所讀文本未見 GitHub 連結） |
| Checkpoint | TBD（所讀文本未見發佈） |
| License | arXiv 頁標示 CC BY 4.0（非明確代碼授權；代碼授權 TBD） |
| 複現難度 | 中—高：需重寫 softmax 聚合層 + QLIKE+KL 聯合訓練；換算 8 維 STLG 特徵（含 ε=10⁻¹² 穩定項、BV 首根 `x^BV_{t,1}=0`、日邊界不環繞） |
| 數據可得性 | Binance 現貨 5 分鐘 OHLC 公開；需自行對齊 UTC 日與不完整日剔除規則 |

## §5 · 評測 / Benchmark

> 表名中的「前SOTA」為模板欄；本文對照的是**經典 RRM 基線**（均勻聚合），不是外部 SOTA。Δ 固定定義為「本方法 − 前SOTA」；QLIKE／RMSE 越低越好，故負值表示 NRM 較低。所有數字均從原文逐字抄錄；Δ 僅在兩值單位相同時計算。

### 5.1 Heston 模擬（絕對 RMSE，單位 10⁻⁵）

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| IV_t · Continuous · RV | RMSE（10⁻⁵） | 0.981 | 0.607 | −0.374 |
| IV_t · Jump · RV | RMSE（10⁻⁵） | 4.820 | 0.665 | −4.155 |
| IV_t · Gradual jump · RV | RMSE（10⁻⁵） | 7.658 | 1.433 | −6.225 |
| IV_{t+1} · Continuous · RV | RMSE（10⁻⁵） | 1.606 | 1.476 | −0.130 |
| IV_{t+1} · Jump · RV | RMSE（10⁻⁵） | 4.784 | 1.386 | −3.398 |
| IV_{t+1} · Gradual jump · RV | RMSE（10⁻⁵） | 7.245 | 1.687 | −5.558 |

原文另報告 BV／GK 的 rRMSE 全部小於一（§5.1 表 1），並稱「NRM lowers RMSE in every primitive–regime combination」。Continuous 增益小、事件壓力下大；作者明言小 RV／GK ratio 反映 RRM 在 estimand mismatch 下誤差放大，不是「近完美恢復 IV」。模擬診斷的是有限樣本測度與預測內容，不是無偏性或一致性。

### 5.2 加密實證（按資產，QLIKE；驗證 MSE 選模後）

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| BTC | QLIKE | 0.301 | 0.278 | −0.023 |
| ETH | QLIKE | 0.272 | 0.260 | −0.012 |
| SOL | QLIKE | 0.253 | 0.246 | −0.007 |
| BNB | QLIKE | 0.348 | 0.342 | −0.006 |
| XRP | QLIKE | 0.560 | 0.541 | −0.019 |
| DOGE | QLIKE | 0.648 | 0.579 | −0.069 |
| 6 幣 Mean | QLIKE | 0.397 | 0.374 | −0.023 |

**配對比較（Table 2）**：18 個規格 × 3 個頭，共 54 個 QLIKE 配對全部 NRM 勝出；SF／HAR／MIDAS 均值增益 7.23%／8.15%／9.22%，全部 pooled QLIKE 檢定顯著。RMSE 三頭各 17/18 勝；MAE 勝出數為 SF 18/18、HAR 15/18、MIDAS 16/18，其中 MIDAS MAE pooled p=0.062（不顯著）。

**靈活基準（Table 3）**：Selected NRM 在 6 幣平均 QLIKE 最低（0.374）；Selected RRM 0.397、HARNet 0.442、DeepVol 0.439、XGB 0.426。各市場次低不總是靈活基準：SOL 的 Selected RRM 0.253 低於 XGB 0.256；XRP 的 Selected RRM 0.560 低於 DeepVol 0.632 與 HARNet 0.648。作者明言最強競爭基準因市場而異，且未識別跨資產差異的市場特徵。

**是真 capability 還是可能的失真？**
- **真 capability 的線索**：paired 設計固定原語、下游預測規格與資料設定，只換聚合層；`w=1` 為嵌套特例；Proposition 3.1 給偏離上界；模擬用 oracle IV 而非 RV 代理，避開代理噪聲對比。
- **額外容量風險**：NRM 多出聚合層，作者在 §4.4 明言「parameter counts need not be equal because the NRM aggregation layer is omitted for RRM」。因此不能把全部增益都歸因於「學習聚合」而非額外參數容量；需搭配消融、KL 正則與權重診斷。
- **前瞻偏差風險**：Global 變量用「已完成當日」訊息（§3.2），對預測 t+1 不引入 look-ahead；同期模擬也允許當日完成後比較。但「同日完成」在部署時序上仍需檢查；此為本頁推斷。
- **成本未計**：論文為波動率預測，不涉及交易成本；QLIKE 改善不直接等於策略淨收益。

### 5.3 消融（§5.3 / Table 5，HAR–NRM leave-one-group-out）

Table 5 Overall：Drop Time 的 ΔQLIKE 為 +4.54%、Drop Local +4.01%、Drop Global +2.54%、Drop Shape +1.11%；ΔRMSE 分別為 +1.06%、+0.56%、+0.20%、+0.29%。增益主要來自 Time 與 Local（日內時序 + 局部對比），Global 次之；Shape 最不穩定（BV 行 Drop S 為 −0.48%）。這些是單一加密樣本內的排序，不可寫成跨所有模型／市場的必然因果結論。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（定位：§7 Conclusion 末段；STLG 另見 §3.2）

1. **Task-specific representation**：NRM 是 forecasting-specification-specific，不能當可重用通用測度；每個 asset／primitive／head 需各自估計。
2. **市場外推有限**：加密 24/7 UTC 日結構一致，但未建立對帶交易時段、隔夜缺口的傳統市場之泛化。
3. **Empirical target 為 noisy realized proxy**：市場端 `R^{RV}_{t+1}` 是代理而非真 latent IV。
4. **模擬非完全校準**：Heston + 事件壓力為受控壓力，不是完整微觀結構模型。
5. **STLG 資訊界**：只用 OHLC 與原語衍生量，刻意排除成交量與訂單簿（§3.2 末），故 RRM–NRM 增益可歸因於學習聚合而非擴充資訊集。

### 6.2 推斷的隱含假設（本頁推斷）

- **regime 穩定**：訓練樣本與測試樣本的日內結構需大致同分布；若出現結構斷點（交易所事件、市場微結構改變），KL 正則只能限制偏離量，不能保證方向正確。所讀文本未逐資產列出訓練期端點，圖表／附件待核。
- **參數容量**：NRM 多一層聚合網路，且作者明言 parameter counts need not be equal；paired 設計隔離的是下游預測規格與資料設定，不能把全部差異都歸因於單一「學習聚合」項。
- **容量／成本**：未計交易成本、滑價與衝擊；QLIKE 改善不等於可部署 alpha。
- **日內時序**：Global 用當日完整分布，若部署時 t+1 預測需在 t 日收盤前完成，則特徵可得性需另加檢查；此為本頁推斷。
- **生存者偏差**：6 幣的選幣標準所讀文本未見；若以事後存活／知名度選幣，可能引入 survivorship，此為本頁推斷，原文未做此檢定。
- **單一資料源**：只用 Binance 現貨指數；對其他交易所、監管變動或衍生品市場的推論屬外推。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 經典 RRM（Barndorff-Nielsen & Shephard 2004 等） | 固定 `w=1` 的日內聚合；NRM 把它作為 `w=1` 嵌套特例 | 概念公開 | 本文 paired 基線 |
| HARNet（Reisenhofer et al. 2022） | 在 HAR 結構上做神經卷積；從頭學壓縮 | 未聲明 | 靈活基準 |
| DeepVol（Moreno-Pino & Zohren 2024） | 膨脹因果卷積直接吃高頻輸入，端到端 | 未聲明 | 靈活基準 |
| XGBoost（Chen & Guestrin 2016） | 日／週／月 HAR 因子 + 滯後絕對／帶符號日收益 | 未聲明（本頁未核） | 非神經非線性基準 |
| 端到端表徵學習 | 取代原語與測度；NRM 保留原語、只學權重 | — | 概念對立面 |
| Sufficient forecasting／attention-based MIL | 任務相關聚合，但池化不受限學習表徵；NRM 聚合預定義原語 | 未聲明 | 概念鄰居 |

🎤 **Interview Tip**
- **正確答**：NRM 不是新測度原語，而是把日內聚合權重做成「正、均值歸一、隨 STLG 上下文變」的可學習層；均勻權重精確回復 RRM；KL-to-uniform 與 Proposition 3.1 把偏離綁在離散度上；權重和原語相乘保留 bar 級歸因；論文宣稱的是 forecast-supervised statistical aggregation，不是通用 attention。
- **錯答**：「NRM 是一個新的注意力機制／通用測度框架，可套用到任何 label 上」——原文 §2 明確與 attention-based MIL 區分；「NRM 保證在所有市場贏」——只在 6 幣 5 分鐘樣本與 Heston 模擬中驗證；「QLIKE 改善即策略 alpha」——論文未計交易成本；「NRM 是可重用通用測度」——作者自陳是 forecasting-specification-specific。

### 7.1 可證偽預測（帶日期）

- **預測 A（2027–2028）**：作者 §7 明列 future work 為 transferable aggregation、alternative day boundaries、traditional asset classes、economic-value applications。若出現把 NRM 遷移到有交易時段／隔夜缺口的股票市場且未重寫 STLG 的 Time（`sin(2πk/K)`）與日邊界處理，其相對 RRM 的 QLIKE 增益應小於本文報告的 SF／HAR／MIDAS 均值增益 7.23%、8.15%、9.22%。可證偽點：觀察該假想論文的 paired QLIKE 改善幅度。**尚待驗證**。
- **預測 B（2027 前）**：若出現第三方複現，且以 RRM 為基線在同一 6 幣、同切分（驗證 2023-06 至 2024-05；測試 2024-06 至 2026-05）下重跑，54 個 QLIKE 配對全勝且均值增益落在本文報告的 7.23%／8.15%／9.22% 附近，則本頁「真 capability」判斷成立；若出現任一 QLIKE 配對反向，則需重新評估 STLG 特徵與訓練穩定性。**尚待驗證**。
- **預測 C**：若把 STLG 擴充加入成交量／訂單簿（作者刻意排除於 STLG），依 §3.2 的「資訊集等價」邏輯，RRM–NRM 的增益不應被此擴充完全吞掉——若吞掉，則原論文的「學習聚合」貢獻需重新界定。**尚待驗證**。

## §8 · For the Reader

- **因子研究員**：把「聚合權重可學習」視為對經典測度的最小侵入式改造。可直接借用 STLG（Shape／Time／Local／Global 8 維）作為 regime 條件；但需注意 KL-to-uniform 的 λ 對「偏離均勻」的調控——λ 大時接近 RRM，λ 小時才發揮重分配效果。跨資產排序（作者稱 BNB 增益最小、BTC 與 DOGE 較大）提示：增益可能與日內結構非平穩性相關，值得做為研究假說。
- **高頻／執行**：NRM 只用完成日資訊，無盤中即時通道；`w_{t,k}` 是日終回顧性權重，不能直接用於 intraday 執行。若要把 `w_{t,k}` 用於盤中倉位，需另建逐 bar 預測版本——本文未提供，屬外推。
- **組合配置**：把 NRM 的 `X^m_t` 當作 regime 過濾而非交易訊號；驗證期／測試期切分可作為 walk-forward 起點。注意 XRP／DOGE 的絕對 QLIKE 較高（0.541／0.579），配置時需按標的做 variance targeting。
- **LLM-agent 研究者**：NRM 的「有界偏離」思路可作為 agent 決策中「可解釋重加權」的模板——權重是機率單純形上的正則化偏離，不是黑箱。可借鏡 KL-to-uniform 的「保留 baseline 作為特例」設計，作為 agent 輸出可審計性的結構性保證。
- **RL 策略研究者**：NRM 不是 RL，但其「監督信號驅動聚合」可作為 reward-shaping 的靈感——用下游目標反推上層聚合。注意：本文沒有 joint exploration／policy 更新，是純監督回歸；把它說成 RL 是誤讀。
- **研究學生**：建議閱讀順序：§1 intro → §3.2 STLG → §3.3 Proposition 3.1 的證明 → §4.3 三個下游頭 → §5.3 ablation。核心可學的工程直覺：**「什麼都不改，只改一層」是有效的 inductive bias**；以及在論文中尋找「命題 + 界」這類可證偽的結構性保證，比只看 benchmark 數字更能判斷方法是否 robust。

## References

- **原論文**：Brian Sing Fan Chan, Jennifer So Kuen Chan. *Neural Realized Measures: Forecast-Supervised Intraday Aggregation for Volatility Forecasting.* ICAIF '26 (Milan, Italy), arXiv:2611.07287v1 [q-fin.ST], 2026-11-05. https://arxiv.org/abs/2611.07287v1
- **Lineage — 測度原語**：Andersen, Bollerslev, Diebold & Labys (2003) *Econometrica* 71(2):579–625；Barndorff-Nielsen & Shephard (2004) *Journal of Financial Econometrics* 2(1):1–37；Garman & Klass (1980) *Journal of Business* 53(1):67–78；Martens & van Dijk (2007) *Journal of Econometrics* 138(1):181–207；Rogers & Satchell (1991) *Annals of Applied Probability* 1(4):504–512。
- **Lineage — 下游頭與神經基準**：Corsi (2009) *Journal of Financial Econometrics* 7(2):174–196（HAR）；Ghysels, Santa-Clara & Valkanov (2006) *Journal of Econometrics* 131(1–2):59–95（MIDAS）；Reisenhofer, Bayer & Hautsch (2022) arXiv:2205.07719（HARNet）；Moreno-Pino & Zohren (2024) *Quantitative Finance* 24(8):1105–1127（DeepVol）；Chen & Guestrin (2016) KDD, pp. 785–794（XGBoost）。
- **Lineage — 方法論鄰居**：Fan, Xue & Yao (2017) *Journal of Econometrics* 201(2):292–306（sufficient forecasting）；Ilse, Tomczak & Welling (2018) ICML, PMLR 80:2127–2136（attention-based MIL）；Li, Nolte, Nolte & Yu (2025) *Journal of Econometrics* 250:106014（realized candlestick wicks）；Chan, Cheung, Chen & Chan (2026) SSRN working paper（time-occupation information）。
- **統計檢定**：Diebold & Mariano (1995) *JBES* 13(3):253–263；Harvey, Leybourne & Newbold (1997) *International Journal of Forecasting* 13(2):281–291；Patton (2011) *Journal of Econometrics* 160(1):246–256（imperfect proxy 下 QLIKE 比較）。
- **來源連結**：arXiv HTML 全文（https://arxiv.org/abs/2611.07287v1）；圖 1–3 原圖未在本次可讀文本中提取，圖表待核。