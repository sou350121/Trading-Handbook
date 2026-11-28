<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=监督回归 alpha=端到端表征 autonomy=人机协同可解释 -->

# Historical Visual Retrieval (DINOv2-S + fine-tuned encoder) 解構（Historical Visual Retrieval (DINOv2-S + fine-tuned encoder)）

> **來源發布**：2026-11-18 · （無 venue） · arXiv [2611.23723](https://arxiv.org/abs/2611.23723)
> **arXiv 原文**：[When Price Charts Rhyme: Historical Visual Retrieval for Stock Return Prediction](https://arxiv.org/abs/2611.23723v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點＝量價表格／日頻波段／監督回歸／端到端表徵／人機協同可解釋。它解掉的 prior gap 是 Jiang et al. (2023) 那類「圖→報酬」黑箱：把 forecasting head 換成非參數 kNN，讓每筆預測可回溯到帶日期的歷史圖。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `监督回归` | `端到端表征` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把 20 日 OHLC＋MA20＋成交量的彩色 K 線圖用 DINOv2-S 編成 384 維表徵，在「後續五日報酬已實現」的遞迴擴張歷史圖庫裡取 cosine top-K 近鄰，用近鄰已實現報酬的等權平均當預測；再用 return-rank KL 把度量對齊到報酬（1993–1999 微調、2000 選模、2001 前凍結）。Eureka 是**只學度量、不學 head**，因此 before/after 可比、alpha 可歸因到近鄰集合。關鍵實證：預訓練 DINO 的 EW H−L Sharpe 為 2.81，對齊後 FT-DINO 為 6.37。（線索中的「微調後 2.81」有誤：2.81 屬凍結的預訓練基準，正文為準。）

**X-Ray.** 這篇的貢獻不在「視覺模型很強」，而在把 return prediction 重新分解成「度量學習 ＋ 非參數鄰居平均」，讓 alpha 的位置從權重搬到近鄰集合，因此失敗面也換了：過擬合風險從參數搬到 library 與度量。Pareto 上它沒有碾壓全場——EW 85% 與 CVW 28% 之間的落差、以及 CVW 淨成本後 Sharpe 0.01，說明這是一個小型、低價、難套利股票的現象，成本與容量 envelope 很窄。真正站得住的對比是同資訊集下的 CNN（EW Sharpe 4.46），以及 spanning 後 CNN 的 EW alpha 塌到 3.7%（t 0.74）而 FT-DINO 仍留 40.95%——這是不對稱吸收，不只是「兩者都有效」。但 ridge 對照（EW 5.99 vs 6.37）提醒：大部分增益來自表徵本身，kNN 規則只加邊際。可解釋性分三層，成本不同：鄰居回溯幾乎免費、SAE 狀態是事後敘事、steering（High-Volatility Range 至 −10 使 Sharpe 掉到 2.32）才是真正有因果意味的介入。對量化讀者的意義：把它當「可審計的技術分析特徵抽取器」比當策略更划算。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（Jiang et al. 2023 CNN／Murray et al. 2024／Xue & Zhou 2026） | 本文 |
|---|---|---|
| 輸入渲染 | 黑白 20 日價量圖；或數值 return path | 彩色 20 日 OHLC bar＋MA20＋volume，原圖 64×60，送網路前 resize 至 518×518 |
| 預測頭 | 參數化 head，圖直接映到報酬 | 非參數 kNN：top-K 近鄰已實現五日報酬等權平均 |
| 表徵來源 | 從零訓練或直接監督於報酬 | 自然圖像自監督（LVD-142M，約 142 million 張），再以 return-rank KL 對齊 |
| 時間閘門 | 一般時序切分 | 逐觀測限制 τ+5 ≤ t 的遞迴擴張 library |
| 可解釋單位 | 係數、saliency map | 帶日期的歷史圖本身＋SAE 稀疏視覺狀態＋視覺 atlas |

### 1.2 ⚡ Eureka

> **把 forecasting head 整個刪掉，改成「查表取平均」；學習預算全押在度量上。**

直覺：若「長得像的圖後面接的報酬」有共同成分，最有效的估計量不是一個學出來的映射函數，而是近鄰結果的平均——前提是度量把「像」定義在與報酬有關的方向上。Return-rank alignment 只改度量、不改規則，所以 DINO 與 FT-DINO 的差異可乾淨歸因於「哪些歷史圖被視為可比」。

### 1.3 信息流

```
CRSP daily 1993–2025 ──▶ 20 日窗 ──▶ render(OHLC+MA20+vol, 紅綠, 64×60)
                                            │ resize 518×518
                                            ▼
                              DINOv2-S (frozen｜FT-DINO)
                                            │ final-layer CLS
                                            ▼
                                 h ∈ R^384 ──▶ z = h/‖h‖₂
                                            │
每週最後交易日 query t ──────────────────────┤
                                            ▼
              cosine top-K over L_t = {(j,τ) : τ+5 ≤ t}   ← 時間閘門（無前視）
                                            ▼
                       R̂ = (1/K) Σ R_{j,τ+1:τ+5}   ← 等權，K=1,000 主結果
                                            ▼
                    週度橫截面十分位 ──▶ EW / VW / CVW 的 H−L
                                            
微調支線：1993–1999 圖 ──▶ u = PctRank_t(R_{t+1:t+5})
   p_ab ∝ exp(−|u_a−u_b|/κ_r)  vs  q_ab ∝ exp(z_aᵀz_b/κ_z)  ──▶ KL ──▶ FT-DINO
   2000 選模 ──▶ 2001 前凍結 ──▶ 重編碼 query 與 library
```

## §2 · 數學層

📌 **Napkin Formula**

```
R̂^(K)_{i,t} = (1/K) · Σ_{(j,τ) ∈ N_K(i,t)} R_{j,τ+1:τ+5},   N_K = TopK cosine(z_{i,t}, z_{j,τ}),  τ+5 ≤ t

L_rank = (1/B) Σ_a KL(p_a ‖ q_a),  p_ab ∝ exp(−|u_a−u_b|/κ_r),  q_ab ∝ exp(z_aᵀz_b/κ_z)
```

複雜度：每個查詢週要對整個 eligible library 做一次 384 維 cosine 排序，O(N_q · |L_t| · 384)；`|L_t|` 未披露（1993 起逐日遞迴擴張，僅知 out-of-sample 有 4,569,085 個 stock-week 觀測與 24.2 million 張 2001–2025 圖）。K 主結果 1,000。

直覺：第一式是把「歷史相似情境的後果」當成條件期望的非參數估計；第二式只是把「報酬排名近」偷換成「表徵近」，用 KL 把兩個 soft neighborhood 分佈拉齊——它保留了排名距離的序資訊，而非二元正負分類。

訓練細節：全參數微調 DINOv2-S，AdamW ＋ linear warmup ＋ cosine decay，訓練集跑一個 epoch，以 2000 完整驗證集的 KL loss 選 checkpoint；`κ_r`、`κ_z`、學習率、batch size 皆 **未披露**（附錄 A.2 只描述流程，所讀文本未見數值）。SAE 端為 JumpReLU，16,384 個 latent，loss 為重建誤差＋λ₀·L0，λ₀ 未披露。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部為「假設／示意」玩具數字，不是論文實證結果。論文真實數字只在 §5。**

1. **編碼**：假設 query 圖得 z_q（384 維、單位長度）；library 中三個候選相似度假設 0.91 / 0.88 / 0.86，取 K=3。
2. **近鄰取平均（等權）**：三者後續五日已實現報酬假設 +2.4% / +1.8% / −0.6%，則 R̂ = (2.4 + 1.8 − 0.6)/3 = **1.2%**。這就是「預測」——沒有任何參數 head。
3. **只換度量、規則不動**：同一 query、同一 library，改用 FT-DINO 度量後近鄰假設換成另一組，報酬 +1.5% / +0.9% / +0.3%，R̂ = (1.5 + 0.9 + 0.3)/3 = **0.9%**。訊號變了，但公式一字未改。
4. **對齊一步**：假設 batch 內 a、b 的 return rank u_a = 0.95、u_b = 0.90，κ_r = 0.1，則 r_ab = −|0.05|/0.1 = **−0.5**；若目前 z_aᵀz_b = 0.30、κ_z = 0.1，相似度項為 3.0。KL 的梯度方向就是「把 a、b 在表徵空間拉近」，或壓低 a 與排名差很大的成員的 q 機率。
5. **進組合**：假設該週橫截面十分位切點使該股分別落在 decile 8 與 decile 9，兩個 R̂ 造成不同的 H−L 曝險——這就是「同一規則、不同度量」在投組層的傳導路徑。

## §3 · 數據層

- **樣本**：CRSP 日頻 common stocks（NYSE／AMEX／Nasdaq），剔除 ADR、非美國註冊公司、abnormal trading status；1993–2025（日開高低價 1992 年 6 月才可得）。1993–2000 供初始 library，2001–2025 為 out-of-sample。
- **頻率設計**：library 是**日頻**（讓每週查詢有更豐富的帶日期歷史路徑），query 是**週頻**（每週最後交易日）。每個觀測要求完整 20 日價格歷史＋完整後續五日報酬。
- **規模**：out-of-sample 4,569,085 個 stock-week 觀測；用於 atlas 的 2001–2025 圖共 24.2 million 張，聚成 4,096 個 prototype。
- **渲染正規化**：窗內首收盤價歸一為 1，其後收盤價由 CRSP 報酬遞迴建構（splits／dividends 不造成假跳空），開高低按當日收盤價比例縮放，價格在窗內再縮放，成交量相對窗內最大值縮放——所以圖只保留形狀，不帶名目價格與股數水準。
- **輔助資料**：WRDS 日頻買賣方發起量（2001–2025，Lee–Ready 簽方向）；securities lending fees（2008 起，故 limits-to-arbitrage 雙重排序自 2008 起）；RSVOL 零售占比（2006–2024）；Google search attention。Robinhood demand shock 僅 2018–2020。
- **樣本外與容量假設**：作者的限制寫在時間閘門（τ+5 ≤ t）與凍結點（2001），但**未披露** library 觀測總數、AUM／容量估計、以及是否納入 delisting return（所讀文本未見）。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD（所讀文本未見代碼連結） |
| Checkpoint（FT-DINO 權重） | TBD（僅說明用 2000 驗證集選模後凍結，未見釋出） |
| SAE 權重 | TBD（16,384 latent JumpReLU，未見釋出） |
| License | 論文 CC BY 4.0；代碼授權 TBD |
| 複現難度 | 高：需 CRSP 日頻 OHLC＋WRDS 簽單流＋securities lending fees＋Google Trends；DINOv2-S 本身公開 |
| 數據可得性 | 低—中：核心資料為商業資料庫；DINOv2-S／MAE／CLIP 為公開模型 |
| 算力足跡 | 未披露（訓練為單一 epoch 全參數微調，但 batch、解析度以外的硬體細節未見） |

## §5 · 評測 / Benchmark

> **表內「前SOTA」欄一律填基線本身的逐字數值，不是 SOTA：本頁基線為凍結的預訓練 DINOv2-S（benchmark DINO）與 Jiang et al. (2023) 的 20 日黑白圖 CNN。論文未使用 SOTA 一詞，也不存在可合成的 SOTA 區間。**所有 Δ 皆為「本方法 − 該列基線」，只出現在末欄。

| 數據集/市場 | Metric | 前SOTA（基線） | 本方法 | Δ |
|---|---|---|---|---|
| 美股 CRSP 2001–2025，週度十分位 H−L | EW 年化報酬（DINO 基線） | 26% | 85% | +59pp |
| 同上 | EW Alpha（FF5+MOM+STR） | 25% | 81% | +56pp |
| 同上 | EW Sharpe | 2.81 | 6.37 | +3.56 |
| 同上 | CVW 年化報酬 | 7% | 28% | +21pp |
| 同上 | CVW Sharpe | 0.87 | 2.38 | +1.51 |
| 同上 | EW Sharpe（CNN 基線） | 4.46 | 6.37 | +1.91 |
| 同上，淨交易成本 R5 | EW 年化報酬（DINO 基線） | -6% | 55% | +61pp |
| 同上，淨交易成本 R5 | EW Sharpe（DINO 基線） | -0.63 | 4.12 | +4.75 |
| 同上，淨交易成本 R5 | CVW Sharpe（DINO 基線） | -2.39 | 0.01 | +2.40 |

**解讀——哪部分是真 capability：**
- 「FT-DINO 勝 CNN」在同資訊集下成立（EW Sharpe 6.37 vs 4.46，CVW 2.38 vs 1.82），且 spanning 後不對稱：CNN 的 EW alpha 塌到 3.7%（t 0.74），FT-DINO 仍留 40.95%（t 16.53）／CVW 14.66%（t 8.50）。這是「檢索表徵吸收直接預測模型」的證據，不是兩者互補。
- 但 ridge 對照（FT-DINO-Ridge EW Sharpe 5.99 vs retrieval 6.37；CVW 2.43 vs 2.38）顯示**大部分增益來自表徵本身**，kNN 只加邊際。把功勞全記在「retrieval」上是誤讀。
- 消融的機制指紋很清楚：Same-Stock 掉到 1.38（78% 跌幅）而 Cross-Ind. 仍有 6.24 → 訊號主要是**跨股票類比**；No Volume 掉到 4.98（22% 跌幅）而 No MA 6.36 → 量能是第二支柱；翻轉／旋轉 query 使 EW Sharpe 變 −0.45 與 −0.07 → 模型學到的是**路徑方向與時序**，不是紋理。

**哪部分可能是過擬合／偏差／成本未計：**
- 容量是最大裂縫：EW 85% 對 CVW 28%，且 CVW 淨成本後 Sharpe 只有 0.01（DINO 為 −2.39）。加價格篩選（≥ \$5）後 EW 掉到 36%（Sharpe 3.94），剔除最小 20% 也是 36%（3.44）——大量 spread 來自低價與小型股。
- 成本只採 10 bp／20 bp 假設，未計借券費、衝擊、short availability；lending fee 只進 limits-to-arbitrage 分組，不進成本。
- 多重比較：K grid × 3 種加權 × 3 種窗口 × 多個 encoder 的掃描未見 formal multiple-testing 校正（本頁推斷）。
- 子樣本衰減：EW Sharpe 8.49（2001–2009）→ 7.91（2010–2017）→ 4.41（2018–2025），作者自述「48% decline」；以 1993–2007 重訓的 FT-DINO-NEW 在 2010–2025 為 73%／6.79，vs 原模型 62%／5.51——凍結模型的衰減部分可用重訓回收。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **§4.3 salience 兩點 caveat**：salience 合成含「近鄰後續報酬絕對值」，而訊號平均的是帶符號值，故部分排序可能機械；Q5 選到近期贏家，單靠對該股自身價格的注意力也能產生同樣的買入—反轉，不需歷史類比。作者明言「We interpret this evidence as suggestive.」
- **§4.3 末＋Appendix D.4**：全樣本對「最相似／最 salient／最近」加權反而變差（composite salience 加權 EW Sharpe 5.50；Google search 歷史權重 2.68／當前 3.72；abnormal volume 1.30），與 memory-based 模型預期相反。
- **Appendix D.2**：RSVOL 有明確梯度：FT-DINO 的 annualized conditional 5–1 spread 由最低 RSVOL 五分位的 10% 升到最高的 88%（DINO 為 1% 到 34%）。相對地，Robinhood demand shock 在五個 shock 五分位間沒有清晰梯度（最低 41%、最高 29%），且此 proxy 僅覆蓋約兩年（2018–2020），作者因此稱零售證據「suggestive」；不得把 RSVOL 寫成無梯度。
- **§4.1／D.1**：days-to-cover 是唯一反向的 proxy，作者歸因於 Asquith et al. (2005) 指出的衡量問題。
- **§3.4**：消融時 encoder 未重訓，「these tests measure only how much the fixed representation relies on each chart element」——即消融量的是固定表徵的依賴度，不是可重新學習後的損失。
- **§3.4**：Same-Stock 下跌「Part of the decline reflects the much smaller library available from a single stock's past」。
- 所讀文本未見獨立的 Limitations 章節，亦未見對容量／AUM 的作者量化說明（圖表／附件待核）。

### 6.2 推斷的隱含假設（明標推斷）

- **Regime 依賴（推斷）**：2001–2025 全期與三個子期間皆正，但 2018–2025 明顯轉弱；若訊號本質是「難套利股的暫時賣壓」，則在流動性改善、散戶結構改變或做空成本下降的 regime 下會衰減。此為推斷，論文未做 regime 條件檢定。
- **容量／成本（推斷）**：EW 週轉 600%，四週持有降到 173%（淨 Sharpe 1.78）；CVW 淨 Sharpe 0.01。推論是「可交易的只有 EW 小型股那一小塊」，但作者未給 breakeven 成本或容量曲線——本頁不補數字。
- **數據泄漏（推斷，風險低）**：時間閘門以觀測為單位（τ+5 ≤ t），比多數文嚴格；微調／SAE 皆在 2001 前完成。Atlas 使用 2001–2025 out-of-sample 20-day charts，prototype 著色為其成員後續五日報酬的橫截面平均排名；作者刻意只用 2001 後圖，以避免 1993–2000 in-sample return outcomes 進入 prototype coloring。但著色本身仍是事後已實現報酬的描述性標籤，不是可交易訊號——讀者不應把它當 alpha 圖。
- **Survivorship（推斷／未見）**：樣本建構只說剔除 ADR、非美註冊、abnormal trading status 與缺價歷史；是否含 delisting return、退市股如何處理，所讀文本未見。
- **單一干預不可外推**：§3.4 的消融每次只改一個圖元素但**不重訓**，§5.2 的 steering 只動 query 不動 library——這些是不同層級的介入，不能把「steering 掉 64%」讀成「該視覺狀態貢獻 64% 的 alpha」。
- **檢索 vs 線性 head（推斷）**：兩者在同一表徵上接近（5.99 vs 6.37），意味「非線性 kNN 的必要性」未被證明；若換市場或更長 horizon，先試線性 head 是更省的 baseline。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Jiang et al. (2023) 20 日黑白圖 CNN | 直接 image→return，參數化 head | 方法公開；資料 CRSP 專有 | 本文主基線，同資訊集下 EW Sharpe 4.46 vs FT-DINO 6.37 |
| Lee et al. (2026) future-aligned soft contrastive / cross-sectional retrieval | 同樣是「對齊＋檢索」 | 所讀文本未載代碼 | 最接近的同軸對手；本文只引用、未做數值對比 |
| Murray et al. (2024) charting by machines | 報酬路徑＋NN 直接預測 | 未見 | 同軸但輸入非圖像 |
| Xue & Zhou (2026) | 預訓練 ViT 於加密貨幣圖像 | 未見 | 不同市場、不同資產 |
| Bøjstrup et al. (2026) | hierarchical ViT；本文借用其 5 個 pattern 定義 | 未見 | 作為 top-down 驗證的 pattern 來源，非績效對手 |
| Wang et al. (2026) 視覺外推 | 補圖右半、模擬投資人外推 | 未見 | 方向相反：他們隔日為正、之後為負；本文後續週月為正且六個月不還原 |
| 短期反轉（Jegadeesh 1990；Lehmann 1990） | 純量過去報酬排序 | — | 控制後 FT-DINO 仍留 40.95%（t 16.53） |

🎤 **Interview Tip**
- **正確答**：「這篇把 return prediction 拆成『度量學習 ＋ 非參數鄰居平均』；alignment 只動度量不動規則，所以 before/after 乾淨可比。它的機制指紋是 Same-Stock 掉到 1.38、Cross-Ind. 仍 6.24，說明 alpha 來自跨股票類比。但 ridge 對照 5.99 說明增益主要來自表徵，且 CVW 淨成本後 Sharpe 0.01，容量很窄。」
- **錯答**：「DINOv2 直接輸出報酬」（不是，是 kNN 取平均）／「這是黑箱，所以不可信」（鄰居層可回溯，SAE 層才是事後敘事）／「Sharpe 6.37 可以實盤」（EW 小型股現象，CVW 淨為 0.01）／「微調後 Sharpe 2.81」（2.81 是凍結的預訓練基準）。

### 7.1 可證偽預測（帶日期）

1. **2027-12-31 前**：若他人以 2026 年後資料、凍結同一 FT-DINO 重跑 EW 十分位，Sharpe 應落在 4.41 附近或更低（2018–2025 已見水準）。若仍穩定 > 6，則「凍結模型隨時間衰減、需重訓」的推斷被否證。
2. **2027-06-30 前**：把成本提高到 20 bp 以上並計入借券費重估 CVW，淨 Sharpe 應 ≤ 1（本文 CVW 淨為 0.01）。若獨立複現能穩定 > 1，則本頁「容量 envelope 極窄」的判斷錯。
3. **2028 年前**：若在獨立樣本上 Same-Ind.（6.06）與 Cross-Ind.（6.24）的差距顯著擴大，則「跨股票類比為主要來源」的機制描述需修正為「同產業內類比為主」。

## §8 · For the Reader

- **因子研究員**：先做兩件事再決定要不要上 GPU——(a) 用 FT-DINO 表徵跑 ridge（作者已給 EW 5.99／CVW 2.43），(b) 對 STR／MOM／FF5 做 spanning。若線性 head 就吃掉大部分，kNN 只是可解釋性的溢價，不是 alpha。
- **高頻執行**：600% 週轉 ＋ 10/20 bp 假設太樂觀；H20（173% 週轉、淨 Sharpe 1.78）才是現實 operating point。先量自己的衝擊成本曲線，再談 EW 85%。
- **組合配置**：這是 CVW 淨 Sharpe 0.01 的訊號——只能當「EW 小型股＋難套利」的衛星，不能當可配置核心。把它當作 limits-to-arbitrage 主題下的訊號源，而不是獨立策略。
- **LLM-agent 研究者**：本框架是可搬移的模板——「檢索庫 ＋ 時間閘門 ＋ 近鄰結果平均 ＋ 稀疏特徵解釋」。把 K 線圖換成事件文件、把 DINOv2 換成文字編碼器，機制不變；可解釋性的三層（鄰居／稀疏特徵／atlas）值得照抄。
- **RL／序列策略研究者**：注意它完全沒有 policy 或 reward learning；「alignment」只是重新定義狀態空間的度量。若你的環境有明確 reward，這篇的價值在 state representation 的對齊目標（排名距離 KL），不在決策層。
- **研究學生**：return-rank alignment（以排名距離為 soft target、以 cosine 相似度為 model 分佈、最小化 KL）可直接移植到任何回歸表徵；但 κ_r、κ_z 與訓練超參在所讀文本中未披露，複現前要自己掃。

## References

- 原論文：Chen, H., Liu, H., Wu, K., Zhou, D. (2026). *When Price Charts Rhyme: Historical Visual Retrieval for Stock Return Prediction.* arXiv:2611.23723v1 [q-fin.ST], 18 Nov 2026. https://arxiv.org/abs/2611.23723v1 （License: CC BY 4.0）
- 直接前作：Jiang, Kelly, Xiu (2023), *(Re-)Imag(in)ing Price Trends*, JF 78, 3193–3249（本文 CNN 基線與圖像設計來源）。
- 表徵對齊 lineage：Zha et al. (2023) Rank-N-Contrast；Lee et al. (2026) Cross-sectional asset retrieval via future-aligned soft contrastive learning（arXiv:2602.10711）。
- 視覺骨幹：Oquab et al. (2024) DINOv2（LVD-142M）；He et al. (2022) MAE；Radford et al. (2021) CLIP。
- 解釋工具：Olson et al. (2025) SAE on vision embeddings；Rajamanoharan et al. (2024) JumpReLU SAE。
- 經濟機制 lineage：Shleifer & Vishny (1997) limits to arbitrage；Coval & Stafford (2007) fire sales；Bordalo et al. (2012) salience；Da et al. (2011) attention；Chan (2003) news vs no-news drift/reversal。
- 同軸對照：Murray et al. (2024) *Charting by Machines*, JFE 153；Xue & Zhou (2026) JBF 190；Bøjstrup et al. (2026) SSRN 6274640；Wang et al. (2026) SSRN 7087542。