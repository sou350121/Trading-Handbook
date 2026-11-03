<!-- ontology-5axis data=文本另类 horizon=跨周期 paradigm=监督回归 alpha=风险择时 autonomy=人机协同可解释 -->

# ModernFinBERT + HAR-X 解構（ModernFinBERT + HAR-X）

> **來源發布**：2026-10-29 · （無 venue） · [arXiv 頁面](https://arxiv.org/abs/2610.39636)
> **arXiv 原文**：[Information Flow and Market Risk: Transformer-Based Evidence on Sentiment and Volatility Dynamics](https://arxiv.org/abs/2610.39636v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點為文本另类 × 跨周期 × 监督回归 × 风险择时 × 人机协同可解释；它解的不是「再造一個更強波動率模型」，而是用 matched HAR-X、purged walk-forward、BH/HAC 多層敏感性，對「market-wide transformer 文本是否對個股 RV 有增量」做嚴格樣本外否證。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `跨周期` | `监督回归` | `风险择时` | `人机协同可解释` |

**Status:** 初稿 — 基於 arXiv 原文；細節待升。
**TL;DR:** ① 把 ModernFinBERT 的五類新聞變數接入 matched HAR-X，對 11 檔美股做 h=1/5/22 的嚴格 walk-forward。② 核心 trick 不是新網路，而是固定模型宇宙、initial-only 選參、purged folds、BH/HAC 多層敏感性，把 text、VIX、AAII 放在同一線性家族比較。③ 對「文本另类」軸★：結果多數不敵 VIX 與 naive；HAR-Text 僅在 2/33 個 expanding cells 勝過 HAR，長 HAC 下有利拒絕數為 0。④ 關鍵實證：Naive 在三個 horizon 的平均 QLIKE 最低，相對 HAR 的降幅為 2.12%、4.86%、33.23%（原文逐字）。

**X-Ray.** 這篇的價值在「拆掉文本溢價的腳手架」，不在模型新奇。它把 text 放進與 VIX、AAII 同估計器、同 BASE、同評估日期的 matched HAR-X，讓「文本有沒有用」變成可被否證的損失比較；再用 252-session 平滑目標、initial-only 選參、稀疏 refit、BH/HAC 把常見的過度顯著壓掉。結果是負面但乾淨：text 的兩個短帶寬有利 cell 在長 HAC 下消失，且同 cell 的 Naive 改善更大，因此不能歸因於文本。它打不開的 envelope 也清楚：11 檔事後倖存大型股、market-wide 而非 firm-specific 文本、ModernFinBERT 無 point-in-time 訓練截止、無交易成本與容量分析。對量化讀者，這是一篇「先證明你的文本特徵在 matched baseline 下沒有增量」的校準器，而不是可直接上線的 alpha 論文。

## §1 · 架構 / Core Mechanism

### 1.1 三個設計支點 vs 前作

| 構面 | 前作（所讀文本所述 Article 2 / Article 3） | 本文 | 對讀者的意義 |
|---|---|---|---|
| 預測對象 | Article 3 發現 text-derived emotions 對短horizon方向預測不穩；Article 2 指 VIX 與後續交易動態較一致 | 個股 annualized Yang–Zhang RV level，h=1/5/22 | 把「文本能否預測報酬」換成「文本能否預測風險」 |
| 資訊通道 | 前作多為單一或少數情緒指標 | text、VIX、AAII 三個概念上不同的 market-wide 通道，放入 matched HAR-X | 防止 VIX/問卷/組合的增益被誤記為文本功勞 |
| 評估設計 | 前作未見同等級的多重檢定與 HAC 敏感性 | 固定 11 模型宇宙、initial-only 選參、五個 chronological blocks、792-test BH family、252+h−1 長帶寬 | 把「最小損失排名」與「可拒絕的增量」分開 |

### 1.2 ⚡ Eureka

**一句話：** 把「文本有沒有用」降維成「同估計器、同 BASE、同評估日期的 matched HAR-X 對照」；若 text 不能在這個對它最有利的線性比較中穩定勝出，就不必先談 nonlinear 或更大模型。

直覺：HAR 的持久性很強，任何新增 block 都必須在 purge 後的 OOS 打敗它，而不是在 in-sample 的 SHAP 排名裡看起來重要。

### 1.3 信息流 ASCII 圖

```text
NYT articles
  → ModernFinBERT posteriors (bullish / bearish / neutral)
  → session assignment (16:00 ET rule)
  → daily aggregates: sentiment, directionality, entropy, intensity, volume
  → lag-1 + mean5 + mean22
  → TEXT block (15 predictors)

OHLC daily
  → Yang–Zhang trailing 252-session variance
  → forward average over h ∈ {1,5,22}
  → annualized target

VIX daily ── lag-1 + mean5 + mean22 ──→ VIX block (3)
AAII survey ─ as-of merge + lag ──────→ AAII block (12)

BASE (10) + one/more blocks
  → HAR / HAR-X / Elastic Net / RF / XGB / Naive / Stacking
  → purged expanding & rolling walk-forward
  → QLIKE primary; DM / CW; BH; long-HAC
  → ex-post regime strata + SHAP
```

## §2 · 數學層

📌 **Napkin Formula**

```text
YZ_252: rv_{i,t} = σ_o² + k_N σ_c² + (1−k_N) σ_RS²,
        k_N = 0.34 / (1.34 + (N+1)/(N−1)),  N = 252

target: y_{i,t}^{(h)} = sqrt(252 · (1/h) Σ_{j=1}^{h} rv_{i,t+j}),
        h ∈ {1, 5, 22}

QLIKE:  L_t = r_t − log r_t − 1,  r_t = y_t² / (ŷ_t⁺)²
```

**直覺：** 目標不是「未來 h 期報酬的變異數」，而是「未來 h 個 trailing 252-session YZ 估計的平均再開根號」；這是一個高度平滑、帶共享歷史觀測的預測物件，會誘發比 h 更長的損失差分依賴，所以作者把 252+h−1 當依賴穩健推論規則。**複雜度：** 每個 stock-date 重算 trailing 252 視窗；計算量隨 N 線性（本頁推斷）。**Loss/訓練：** 全部模型 fit annualized volatility level，不做 log variance target；QLIKE 為主，raw squared error 進 DM/CW；hyperparameters 只在最初 252 個完整觀測選一次，之後凍結，five blocks 只 refit 係數/樹，不再 tune。

## §2.5 · 帶數字走一遍（Worked Example）

> 本節只使用原文逐字常數與符號；原文未給可復算的單點輸入，故不引入玩具數值，避免臆造。

1. 取 $N=252$ 的 OHLC 窗口，依原文式算 $o_{i,s}$、$c_{i,s}$、$u_{i,s}$、$d_{i,s}$。
2. 用原文 $k_N = 0.34 / (1.34 + (N+1)/(N−1))$，代入 $N=252$；單點數值原文未披露，標 `TBD`。
3. 把 overnight、open-to-close、Rogers–Satchell 三個 variance 成分加權成 $rv_{i,t}$。
4. 對 $h \in \{1,5,22\}$，取未來 $h$ 個 $rv$ 的平均，乘 252，開根號，得 $y_{i,t}^{(h)}$。
5. 預測器只用到 \$t−1\$；text 先按 trading session 聚合，再 lag-1 / mean5 / mean22。
6. 輸出 $\hat{y}$，與 $y$ 算 QLIKE；數值結果留在 §5，本節不補算。

## §3 · 數據層

- **股票池：** AAPL、ABBV、ABT、AMZN、JNJ、JPM、LLY、MA、UNH、WMT、XOM，共 11 檔；每檔 h=1/5/22，合計 33 個 ticker–horizon cells。原文明言這是 **ex-post selected panel of 11 currently large U.S.-listed firms**，不是 point-in-time 大型股宇宙重構；`No ticker–horizon cell is imputed.`
- **時段：** source-data 為 2000-01-01 至 2022-12-31；effective modeling sample 因 252-session YZ 初始化與 lag 建構，長歷史股票一般自 2001 起，ABBV 與 MA 較晚。**時間邊界分列：** ① ModernFinBERT 上游預訓練/微調截止日 **未披露**，模型卡未給 verified training-data cutoff；最早可驗證權重日期為 July 4, 2025，但這是 release date，不是 cutoff。② 下游適配：**沒有**對 ModernFinBERT 微調；HAR-X 係數在每個外層 fold refit，hyperparameters 只在最初 252 個完整觀測選一次。③ 主題/語料對齊：NYT 文章按 16:00 ET 規則映射到 U.S. equity trading session，文本特徵再 lag；這是 forecasting pipeline 的時間對齊，不是 classifier 的 point-in-time 保證。
- **文本語料：** Kaggle `NYT Articles 2.1M 2000–Present`；analyzed corpus 363,030 documents、18,180,668 tokens、vocabulary 185,573；average document length 50.08、median 27、p99 272、max 88,822；empty 7,291（2.01%）；outliers above p99 3,622。
- **標籤分布：** neutral 234,173（64.51%）、bearish 86,012（23.69%）、bullish 42,845（11.80%）；class imbalance ratio 5.47；corpus-level label entropy 1.264 bits。
- **去重與重疊：** exact duplicate texts 15,333（4.22%）；duplicate URLs 0；near-duplicate pairs 2（threshold 0.90）；Jaccard overlap 0.310 / 0.297 / 0.444；OOV tokens 11,869 / 8,168。
- **Timestamp audit：** 359,908 篇 study-period articles；233,808 篇在 trading day at or before 16:00 ET 歸當日；57,057 篇 after 16:00 ET；63,455 篇 weekend；5,588 篇 exchange holiday / non-trading day；359,897 篇成功映射；11 篇 after final 2022 session 未分配。
- **ModernFinBERT 一致性審計：** N=2,700；overall accuracy 0.577；balanced accuracy 0.484；macro F1 0.481；weighted F1 0.577；Cohen’s κ=0.214；multiclass Brier 0.656；multiclass log loss 1.331。Class-specific：bearish F1 0.438、neutral F1 0.689、bullish F1 0.316。Confidence ≥0.90 的子組 accuracy 0.655，但仍有 34.5% 與 blinded validation labels 不一致。原文明言這不是 expert ground truth，參考標籤由 ChatGPT 5.6 Sol 產生且無 seed。
- **容量/樣本外假設：** 五個 chronological test blocks、expanding 為主、rolling 252 為 robustness；initial 252 觀測選參、63 觀測 stacking calibration、purge h。無 post-test embargo，因未來觀測不進 earlier fit。容量與交易成本未建模。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 未披露；所讀文本未見公開 code link |
| Checkpoint | `tabularisai/ModernFinBERT`（Hugging Face） |
| License | 未披露 |
| 複現難度 | 高：需 Yahoo Finance OHLC、CBOE VIX、AAII 供給序列、Kaggle NYT corpus；作者稱 archive 保留 fitted models / SHAP arrays / split manifest，但本次未見 public repo |
| 數據可得性 | 股價與 VIX 公開；AAII 為 supplied chronological survey series；NYT 需 Kaggle 下載；ModernFinBERT 權重公開 |
| 關鍵缺口 | ModernFinBERT training-data cutoff 未披露；2,700-text audit 未重新生成；完整 165-row fold export 與 full cell-by-regime loss table 原文稱保留在 replication archive，本次未見 |

## §5 · 評測 / Benchmark

**固定五欄表（Δ = 本方法 − 前SOTA；QLIKE 越低越好，故正值表示本方法較差）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---:|---:|---:|---:|
| 11 檔美股 pooled，h=1，expanding | mean QLIKE | 0.000292（HAR） | 0.000446（HAR-Text） | +0.000154 |
| 11 檔美股 pooled，h=5，expanding | mean QLIKE | 0.000711（HAR） | 0.001307（HAR-Text） | +0.000596 |
| 11 檔美股 pooled，h=22，expanding | mean QLIKE | 0.005700（HAR） | 0.007835（HAR-Text） | +0.002135 |

**Mean normalized QLIKE（原文 Table 4.14 逐字）**

| 模型 | h=1 | h=5 | h=22 | QLIKE cell wins（33 tasks） | matched QLIKE better |
|---|---:|---:|---:|---:|---:|
| HAR | 0.000292 | 0.000711 | 0.005700 | 4 | — |
| HAR-VIX | 0.000420 | 0.001083 | 0.007870 | 17 | 18/33 |
| HAR-AAII | 0.000385 | 0.001054 | 0.008681 | 0 | 2/33 |
| HAR-Text | 0.000446 | 0.001307 | 0.007835 | 0 | 2/33 |
| HAR-Full | 0.001236 | 0.003461 | 0.012130 | 0 | 2/33 |
| Naive | 0.000286 | 0.000676 | 0.003806 | 12 | — |

**解讀：**

- **Naive 的「最低平均損失」不是全面支配。** 原文逐字：Naive 在三個 horizon 的平均 QLIKE 最低，相對 HAR 降幅 2.12%、4.86%、33.23%；但這是 equal-weight cross-stock averages。h=22 的 33.23% 集中於 JNJ、JPM、AAPL，貢獻 20.41、10.67、5.91 pp；MA 與 LLY 只加 0.50、0.11 pp，其餘六檔合計抵消 4.37 pp。HAR 在 11 檔 h=22 中有 6 檔 QLIKE 低於 Naive；Naive 在 33 個 stock–horizon cells 中 13 個低於 HAR。這是 concentrated descriptive pattern，不是 uniform dominance。
- **Text 的增量在 matched 比較中不成立。** HAR-Text 在 33 個 expanding cells 中只有 JNJ、JPM（h=22）兩個勝出；原文附錄給的 absolute QLIKE differences 為 0.008653 與 0.003374。但同 cells 的 Naive 對 HAR 改善為 0.012795 與 0.006692，更大；因此這兩個短帶寬有利結果不能識別 text-specific value。短帶寬 h 下，primary QLIKE 有 2 個 favorable、30 個 adverse adjusted rejections；在 252+h−1 長帶寬下，favorable 為 0、adverse 為 3。**這是全文最關鍵的失效模式：短 HAC 的顯著被長 HAC 殺掉。**
- **VIX 是「最頻繁 cell winner」，不是「最低 mean loss」。** HAR-VIX 在 17 個 QLIKE cells 最低、18/33 matched QLIKE better，但 mean QLIKE 在每個 horizon 都高於 HAR；primary QLIKE 有 8 個 favorable、11 個 adverse rejections。原文逐字：HAR-VIX 相對 HAR 的 mean QLIKE 在 h=1/5/22 分別 worse by 43.77%、52.44%、38.07%。HAR-Text 對應 worse by 52.62%、83.95%、37.45%。**不要把这些逐字基線數字合成一個假 SOTA 區間。**
- **AAII 與 Full block 沒有穩定增益。** HAR-AAII 沒有 favorable primary QLIKE rejection；HAR-Full 有 1 個。AAII 的 shares 與 spread 代數相依，individual coefficients 與 SHAP 不應作結構解釋。
- **Tree/Stacking 的大損失不是「非線性無效」的證據。** 原文明言這關乎 initial-only selection、稀疏 refits、以及固定 Ridge λ=1 的 calibration rule；stacking 的五個係數和在 165 次 fit 的中位數，expanding 為 0.00477、rolling 為 0.00434，預測近似常數，故在許多 adverse comparisons 產生極小 p 值。Stacking 貢獻 66 個 tests，保留是為了維持 ex-ante model universe，不是實質證據。
- **CW 的有利結果不能替代 QLIKE。** HAR-Text 的 CW 在 14 個 cells 為正、13 個 globally significant，但 CW 是 adjusted raw level squared error 的 one-sided test，不是 QLIKE 增益。
- **Rolling 更不利。** Rolling 下 Naive 赢 31 個 QLIKE comparisons；HAR-Text 只有 0 個 favorable、29 個 adverse primary QLIKE rejections；長 HAC 下 0 個 favorable、24 個 adverse；BY 下 0 個 favorable、6 個 adverse。JNJ h=22 是唯一在兩個 windows 都為正點估計的 cell。
- **Regime 只描述異質性。** HAR-Text 在高波動 state 的 wins 為 10/33（expanding）、5/33（rolling），低波動 state 為 2/33、2/33；相對改善在高 state 較大的 cells 為 22/33、21/33。但 median relative improvement 在兩個 state 都為負：expanding 為 −8.73% / −56.28%，rolling 為 −100.49% / −194.89%。這是「較不壞」，不是「普遍正向增益」。
- **過擬合/前瞻偏差/成本：** 作者已把 text 設為 fixed ex-post measurement instrument，承認無法排除 ModernFinBERT 權重編碼 evaluation corpus 之後的語言/經濟資訊；也沒有交易成本與容量分析。QLIKE 的 proxy robustness 依賴 Patton (2011) 類假設，不能證明此平滑 OHLC target 是 next-session integrated variance 的不偏 proxy。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§2.1：** 11 檔是 ex-post selected panel of currently large U.S.-listed firms，不是 point-in-time reconstruction；結論限制在此 panel，不應推廣到 2000–2022 所有曾符合大型股資格的企業。
- **§3.3：** ModernFinBERT 是 fixed ex-post textual measurement instrument；training-data cutoff 未披露；July 4, 2025 只是 confirmed model release date，不是 cutoff；不能建立「模型只訓練於 evaluation news sample 結束前」的主張。
- **§2.2：** 2,700-text agreement audit 不是 expert ground truth；參考標籤由 ChatGPT 5.6 Sol 在 August 2026 產生，無 reproducibility seed 或 decoding parameters；不能宣稱 absolute classification validity。
- **§3.2：** target 是 forward average of trailing 252-session YZ estimates，不是 only next h returns 的 variance；重疊造成 loss differential 的長依賴，因此需要 252+h−1 HAC。
- **§3.7：** hyperparameters 只在最初 252 個完整觀測選一次；initial sample 小，h=22 的第一個 inner training set 只有 41 個觀測；參數在早期市場環境選定，未必在後期最優；結果評估的是 frozen-selection procedure，不是 continuously re-tuned system。
- **§3.6：** Stacking 用 fixed Ridge λ=1，calibration sample 短，係數和強烈收縮至近零，行為近似 intercept-only near-constant reference combination；其差績效是此 prespecified calibration rule 的性質，不是 forecast combination 一般的反證。
- **§3.11：** 兩狀態 Markov-switching 用 full-sample smoothed probabilities，strictly ex-post descriptive strata；不是 real-time regime-timing signal，模型也不把它們當 predictors。
- **§3.12：** SHAP 解釋 fitted forecasts 的 within-model attribution，不是 causal effects，也不建立 incremental OOS forecast value。
- **§4.3：** 完整 165-row fold comparison export 與 full cell-by-regime loss table 未在本節提供；該小節的量是 descriptive summaries，不是 independently auditable confirmatory tests。
- **§4.3 / §D.1.1：** rolling HAR-Full 對 MA 在 h=5 有一個 negative forecast；positivity floor 10−6 產生 extreme QLIKE observation，保留不刪；post-hoc floor sensitivity 不改主結果。
- **§5：** 結論條件於 long-window target、sparse refits、fixed early-sample settings、strongly shrunk stacking；additional information 在這個特定高度持久 target 下不一定可靠改善。

### 6.2 推斷的隱含假設（本頁推斷，待驗證）

- **Stationarity within window：** expanding/rolling 252 內假設 HAR 係數近似穩定；若 2008、2020 類結構斷裂主導，fold-specific 係數不穩會放大 Naive 優勢。這是本文結果的合理機制之一，但不是唯一解釋。
- **Market-wide text 對 11 檔同質：** 同一個 daily text signal 合併進每檔股票，隱含「broad news 對不同 sector 的增量相近」；若 sector-specific exposure 重要，pooled 結論會稀釋。
- **Session mapping 足夠：** 16:00 ET 規則把 after-close/weekend/holiday 映射到 next session，但沒有驗證 intraday release time；AAII 也只靠 supplied date field 做 as-of merge，historical release time 未獨立驗證。
- **VIX 時序：** VIX close 先 shift 一個 trading day，避免 same-day equity forecast；這假設 day-t VIX 的 16:15 ET 資訊不會透過其他管道進入 day-t 決策。
- **Survivorship：** 11 檔是事後選出的現存大型股；輔助數據缺失不是 survivorship 的替代控制，面板本身就有 survivorship bias。
- **成本/容量：** 這是 volatility forecast evaluation，不是可交易策略；沒有 transaction cost、borrow、impact、capacity 假設，不能直接換成 alpha。
- **多重檢定依賴：** BH 在 792 tests 上做一次；rolling 另有 792 與 joint 1,584；BY 只作敏感性。若 test statistics 的依賴結構不滿足 BH 條件，adjusted rejection 的詮釋仍有限。
- **QLIKE proxy：** QLIKE 的 proxy robustness 需要假設；平滑 YZ-252 target 不是 next-session integrated variance 的無偏 proxy，損失排序不能直接解讀為「對真實變異數的預測能力」。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| HAR-RV（Corsi 2009） | canonical 三個 regressors；本文用 augmented HAR，BASE 含 10 個 predictors（5 volatility + 5 log-volume） | 概念公開 | 本文 baseline |
| HAR-VIX | 同線性家族加 VIX block（3 predictors） | 公開 | 最頻繁 cell winner；mean QLIKE 每個 horizon 高於 HAR |
| HAR-AAII | 同線性家族加 AAII block（12 predictors） | 公開 | 無 favorable primary QLIKE rejection |
| FinBERT / ModernFinBERT | 文本測量層；ModernBERT-based 三分類 | 權重公開；training cutoff 未披露 | fixed ex-post measurement，非 point-in-time |
| Bucci (2020) neural-network RV | 非線性/網路預測 | 文獻 | 本文未實作，不能直接比較 |
| HARK（Buccheri & Corsi 2021） | nonlinear HAR、measurement error | 文獻 | 本文引用但未納入模型宇宙 |
| Stacking | forecast combination | 本文 | fixed Ridge λ=1，強縮至近常數 |
| GARCH / Realized GARCH | conditional volatility 家族 | 公開 | 不在本文 11 模型宇宙 |

**🎤 Interview Tip**

- **正確答：** 這篇是 negative result + measurement audit。Text 在 matched HAR-X 下沒有穩定增量：HAR-Text 只贏 2/33 expanding cells，短帶寬的兩個 favorable 在 252+h−1 下歸零；Naive 的平均 QLIKE 最低但只集中在少數股票；VIX 是最頻繁 cell winner 但 mean loss 更差；ModernFinBERT 的 semantic audit accuracy 0.577、κ=0.214，且 training cutoff 未知，所以不能說它是 point-in-time 信號。SHAP 只解釋 fitted forecast，不是因果或增量。
- **錯答：** 「FinBERT 表徵更強，所以 volatility 預測更好」；「VIX 平均損失最低」；「2/33 勝出代表 text 有效」；「SHAP 上 text 排名高就是增量信息」；「長 HAC 只是保守，短 HAC 顯著就夠」。

### 7.1 可證偽預測（帶期限）

- 若把 target 從 trailing 252-session 平均改成 only next h returns 的 variance，且維持相同 11 檔與 expanding folds，則 HAR-Text 目前的 2/33 勝出數不會自動保留；**預測期限：下一次 arXiv 修訂版（若發布）或後續公開重現**。此為本頁推斷，待驗證。
- 若改用有 verified point-in-time training cutoff 的 sentiment 模型，text 的增量更可能下降而非上升；**預測期限：下一次公開重現**。此為本頁推斷，待驗證。
- 若把 evaluation 改成 trading P&L 而非 QLIKE，Naive 的平均損失優勢不會自動轉成可交易 alpha；**預測期限：後續工作**。此為本頁推斷，待驗證。

## §8 · For the Reader

- **因子研究員：** 先把你的文本特徵放進 matched HAR-X，與 VIX、AAII 同估計器、同 BASE、同評估日期比；不要只看 SHAP 排名。若 text 只在少數 cells 勝出，先檢查同 cells 的 Naive 是否改善更大，這通常是 baseline 不穩而不是 text 有用。
- **高頻執行：** 這是 daily session-level 文本，不是 intraday alpha；16:00 ET mapping 與 VIX lag 已把 same-day 資訊切開。若你要做 intraday，需另建 timestamp 與 release-time 驗證，不能直接搬。
- **組合配置：** 把此文當 risk-model 校準器：它告訴你 market-wide text 在這個 long-window target 下不穩定；若要上線，應先做 cost/capacity 與 regime-conditional 的獨立 OOS，而不是用 in-sample regime labels 擇時。
- **LLM-agent：** ModernFinBERT 是 measurement instrument，不是 generator；它的 training cutoff 未披露，audit 也非 expert ground truth。若你的 agent 要用它做 signal，先做 point-in-time 檢查與 confidence subset 穩健性。
- **RL 策略：** 本文的 target 高度平滑，reward 若是 next-h variance 會完全不同；不要把這裡的 negative result 直接解讀成「text 對 RL 無用」，也不要把 Naive 的 QLIKE 優勢當成 policy 優勢。
- **研究學生：** 這篇是 multiple-testing 與 HAC 敏感性的良好教材：short bandwidth 的 2 個 favorable 在 long bandwidth 下變 0 個；BH family 因 stacking 的 near-constant forecasts 而改變 rank threshold；這些都要在 replication 中保留，不能只挑顯著結果。

## References

- Lis, S., Ślepaczuk, R., & Sakowski, P. *Information Flow and Market Risk: Transformer-Based Evidence on Sentiment and Volatility Dynamics.* arXiv. [原文連結](https://arxiv.org/abs/2610.39636v1)
- Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics.*
- Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics.*
- Clark, T. E., & West, K. D. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics.*
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics.*
- Buccheri, G., & Corsi, F. (2021). HARK the SHARK: Realized volatility modeling with measurement errors and nonlinear dependencies. *Journal of Financial Econometrics.*
- Bucci, A. (2020). Realized volatility forecasting with neural networks. *Journal of Financial Econometrics.*
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS.*
- ModernFinBERT checkpoint: [tabularisai/ModernFinBERT](https://huggingface.co/tabularisai/ModernFinBERT)