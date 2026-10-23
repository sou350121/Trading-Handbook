<!-- ontology-5axis data=量价表格 horizon=跨周期 paradigm=监督回归 alpha=风险择时 autonomy=全自动黑盒 -->

# AddMart 解構（AddMart）

> **來源發布**：2026-10-19 ·（無 venue）· arXiv [2610.25671](https://arxiv.org/abs/2610.25671)
> **arXiv 原文**：[Arbitrage-Free Implied Volatility Surface Reconstruction via Additive Martingales](https://arxiv.org/abs/2610.25671v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在「量價表格 × 跨周期 × 監督回歸 × 全自動黑盒」；解的是 few-shot IV 曲面重建裡「用有限網格罰項換無套利」的 prior gap——把 static no-arbitrage 從 loss 搬到輸出參數化。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `跨周期` | `监督回归` | `风险择时` | `全自动黑盒` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ①把 few-shot IV 曲面重建改成「回歸指數可加鞅的增量密度」，再由卷積與 Black 反解還原曲面。②核心 trick 是輸出非負增量密度、全支撐線性卷積出合法 risk-neutral marginal：calendar 由 convex order、butterfly 由 marginal 非負自動成立，訓練 loss 裡沒有任何 arbitrage penalty。③這對「監督回歸 × 全自動黑盒」軸★的意義是拿掉 accuracy–regularization trade-off，代價是表達力被 additive 結構鎖住。④日線資產等權 overall IV-MAE 1.65 vs HyperIV 1.72，但 NVDA 上 HyperIV 2.46 優於 2.54、reliable 區 HyperIV 1.27 優於 1.29。

**X-Ray.** AddMart 的真正位置不是「更準的 IV 回歸器」，而是把 IV 曲面重建改成**密度生成問題**。五軸的 Alpha 欄標「風險擇時」其實是粗略映射——它輸出 risk-neutral marginal，不是方向訊號；價值在報價、風控與相對定價。它解掉的舊坑很具體：HyperIV／Deep Smoothing 在有限網格上用 penalty 逼近 butterfly／calendar，形成 accuracy–regularization trade-off 與 grid-dependent 的軟約束；AddMart 讓無套利成為座標變換的性質，penalty 為零，RND 直接合法，VaR／CVaR／crash prob 免數值微分與事後修補。代價是表達力被 additive、independent-increment 結構鎖住：作者自述 long-horizon skew flattening（LEAPS）與 VIX 這類強均值回歸／跳躍標的失配。Envelope 因此很清楚：短中期、流動性足以提供 3×3 參考報價的股票／ETF 期權；不是 LEAPS 尾部對沖、不是 VIX 短端、也不是 HFT（單面 forward 6.98 ms、end-to-end 7.60 ms）。對量化讀者的意義：這是一條「把硬約束寫進生成過程」的工程路線，可移植到任何需要合法機率測度的曲面／期限結構問題。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（HyperIV／Deep Smoothing） | AddMart | 後果 |
|---|---|---|---|
| 輸出物件 | 直接回歸 IV(k,τ) | 回歸 G 個 log-increment 的密度 η_g | 誤差經定價映射傳導，不是直接作用在 IV |
| 無套利 | 有限網格上的 soft penalty（calendar／butterfly／density normalization） | 非負增量 + 全支撐卷積硬保證（Theorem 1） | 訓練 loss 無 penalty；無 accuracy–regularization trade-off |
| RND | 對預測 IV 二次微分，可能負密度 | marginal P_j 直接是機率測度 | VaR／CVaR／crash prob 免數值微分與事後修補 |
| 到期結構 | 各到期日獨立預測，再靠 penalty 串 | 同一串增量卷積，calendar 由 convex order | 跨到期一致性是結構性的 |
| 推理 | 單次前向（HyperIV）／每面優化（DS、SSVI） | 單次前向 + 無參數 decoder | amortized；但卷積 + Black 反解較慢 |

### 1.2 ⚡ Eureka

**不要預測曲面，預測曲面的生成元。** 把 forward-normalized 價格寫成 unit-mean 非負鞅 M_{τ_j}=∏_{g≤j} e^{L_g}/E[e^{L_g}]：鞅性等價於 marginals 在 convex order 下單調（calendar），每個 marginal 是合法機率測度（butterfly）。網路只負責吐每個 L_g 的密度，其餘是確定性卷積與定價。直覺：你不需要在 loss 裡「處罰」負密度或日曆違規，因為你的參數化根本表達不出它們。

### 1.3 信息流

```
9 ref quotes (k_i,τ_i,σ_i)
  └─ linear → R^128
      └─ 2×Transformer encoder (2 heads, no positional encoding)
          └─ mean pooling over quotes → h ∈ R^64
              └─ for each g=1..G: [h, τ_g, √τ_g] ∈ R^66
                  └─ MLP 128×2 GELU → R^{3R}, R=8
                      └─ (π softmax, μ=0.3√Δτ·μ̂, s=√Δτ[0.02+softplus(ℓ̂−1)])
                          └─ GMM 密度 → 離散化 X=2048 on [−5,5] → η̂_g
                              └─ zero-padded FFT 線性卷積 → ν_j, ν_j*
                                  └─ suffix sums → c(z,τ_j) → 到期插值
                                      └─ Black bisection 反解 → IV(k,τ)
                                          └─ 副產物 P_j（RND）→ VaR/CVaR/crash prob
```

## §2 · 數學層

📌 **Napkin Formula**

```
M_{τ_j} = e^{A_{τ_j}} / Z_j = ∏_{g≤j} Y_g ,   Y_g = e^{L_g} / E_Q[e^{L_g}]
c(z,τ_j) = E_Q[(M_{τ_j} − z)^+] = ∫_R (e^x/Z_j − z)^+ ν_j(dx) ,  ν_j = η_1 * ⋯ * η_j
```

**複雜度**：G=30 個 maturity buckets、每個 η_g 在 X=2048 點離散化；累積支撐 L=G(X−1)+1 atoms，實作 zero-pad 每個 FFT 到 next power of two 做全支撐線性卷積（無 circular wraparound）。單面 forward 6.98 ms、end-to-end 7.60 ms（Table 10）。訓練 decoder 用 fixed-width cropped convolution，推理換回 full-support；Lemma 2 給 cropped 版本的 calendar residual bound（2(δ_j+δ_{j+1})）。

**Loss／訓練**：Huber β=0.05 作用在 IV 殘差；forward 用 60 次 bisection 反解 Black，backward 走 implicit differentiation，Tikhonov λ_V=0.01（最大放大 1/(2λ_V)=50）。**無 penalty、無 regularization、無 constraint term**。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下全部為示意假設，不是論文實證數值；論文真實數字只在 §5。**

1. 假設只有 G=2 個 maturity bucket，每個 L_g 是兩點分佈：P(L=0)=1/2、P(L=x)=1/2。
2. 算 normalizer：E[e^L]=(1+e^x)/2。
3. 算 multiplicative increment：Y 取值 1/E[e^L] 與 e^x/E[e^L]；M_1=Y_1、M_2=Y_1Y_2。
4. 取 z=1，c(1,τ_1)=E[(M_1−1)^+]=(1/2)·max(e^x/E[e^L]−1, 0)。
5. 檢查 calendar：M_2=M_1·Y_2，Y_2 獨立且 E[Y_2]=1；因 (m−z)^+ 對 m 凸，Jensen 給 c(1,τ_2)≥c(1,τ_1)。整個過程沒有訓練、沒有 penalty。
6. 極限檢查：x→0 時 e^x/E[e^L]→1，call 值→0；x 變大時 smile 張開。**增量分佈的離散度直接控制 smile 寬度**，這也是為什麼 ablation 裡單一 Gaussian component 在 SPY 明顯不夠（reliable IV-MAE 2.99 vs 預設 0.56）。

## §3 · 數據層

- **來源／頻率**：OPRA top-of-book quotes，商業數據商 massive.com；mid-quote 定價；put–call parity 推 forward；Black–76 反解 IV。日線每日 15:59 ET 取一面；盤中每 15 分鐘（09:45–15:45 ET），每日最多 25 個 snapshot。
- **規模**：7 個日線標的（SPY、QQQ、IWM、AAPL、GOOGL、META、NVDA）+ 3 個盤中標的（SPY、QQQ、IWM），共 10 個 dataset。正文 §5.1 寫 approximately 40.1 million option quotes across 70,027 surfaces；Appendix E／Table 6 逐字合計 40,129,821 contracts。
- **參考集**：固定 9 個 quotes＝3 個最接近 7／30／90 天的到期 × 3 個 delta（0.50、0.25、−0.25）。所有其餘合約為重建目標。
- **樣本外**：共同 cutoff 2025-02-01；之前做 train/val（pre-cutoff 最後 10% 為 validation），之後全部 test。5 個 seeds。§1 稱 16-month test period、2,500 audited model-date surfaces。
- **過濾**：正 bid/ask 與 size、ask≥bid、相對 spread ≤0.35、價格 ≥0.1、τ∈[0.5/365,2]；quote 須在 T 前 500 ms 內更新。每個 timestamp 至少 4 個到期、40 個合約才保留。
- **關鍵預處理**：每個 asset／scale 都做 static-arbitrage cleaning——逐到期移除最大 convexity 違規的 interior strike，再用 16 個 log-moneyness bins 的 longest non-decreasing subsequence 強制 calendar consistency，然後第二次 butterfly pass。**評估是在已清理的曲面上做的**，這點對 §5 的解讀很重要。
- **容量假設**：訓練 batch 16 surfaces；推論 B32 達 960 surfaces/s，峰值記憶體 127.64／2939.44 MiB。

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | TBD（所讀文本未見 URL） |
| Checkpoint | TBD（未見釋出說明） |
| License | 論文 CC BY 4.0；代碼授權未披露 |
| 複現難度 | 中高：需 OPRA 商業數據、FFT 全支撐卷積、implicit IV backward、Lemma 2 的 cropped 訓練／full-support 推理雙軌 |
| 數據可得性 | 商業授權（massive.com），非公開；Appendix E 給到過濾與 forward 推導細節，但無法免費重建同一 test set |

## §5 · 評測 / Benchmark

**表 A：重建精度（僅取 mean；std 見原文 Table 1）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Daily SPY（vs HyperIV） | Overall IV-MAE ×10⁻² | 0.99 | 0.89 | −0.10 |
| Daily QQQ（vs HyperIV） | Overall IV-MAE ×10⁻² | 1.31 | 1.25 | −0.06 |
| Daily IWM（vs HyperIV） | Overall IV-MAE ×10⁻² | 1.27 | 1.17 | −0.10 |
| Daily AAPL（vs HyperIV） | Overall IV-MAE ×10⁻² | 1.55 | 1.40 | −0.15 |
| Daily GOOGL（vs HyperIV） | Overall IV-MAE ×10⁻² | 2.43 | 2.37 | −0.06 |
| Daily META（vs HyperIV） | Overall IV-MAE ×10⁻² | 2.03 | 1.95 | −0.08 |
| Daily NVDA（vs HyperIV） | Overall IV-MAE ×10⁻² | 2.46 | 2.54 | +0.08 |
| Daily SPY（vs HyperIV） | Reliable IV-MAE ×10⁻² | 0.57 | 0.56 | −0.01 |
| Daily GOOGL（vs HyperIV） | Reliable IV-MAE ×10⁻² | 2.07 | 2.22 | +0.15 |
| Minute SPY（vs HyperIV） | Overall IV-MAE ×10⁻² | 0.96 | 0.92 | −0.04 |
| Minute QQQ（vs GNO） | Overall IV-MAE ×10⁻² | 1.01 | 1.23 | +0.22 |
| Minute IWM（vs HyperIV） | Overall IV-MAE ×10⁻² | 1.11 | 1.09 | −0.02 |
| Daily 7-asset equal avg（vs HyperIV） | Overall IV-MAE ×10⁻² | 1.72 | 1.65 | −0.07 |
| Daily 7-asset equal avg（vs HyperIV） | Reliable IV-MAE ×10⁻² | 1.27 | 1.29 | +0.02 |

**表 B：static-arbitrage 違規率（%，801×30 price grid；前SOTA 欄在此為對比基線的逐字值）**

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Daily SPY（vs HyperIV） | Calendar violation (%) | 0.7310 | 0 | −0.7310 |
| Daily SPY（vs HyperIV） | Butterfly violation (%) | 1.6295 | 0 | −1.6295 |
| Daily META（vs HyperIV） | Calendar violation (%) | 2.6205 | 0 | −2.6205 |
| Daily META（vs HyperIV） | Butterfly violation (%) | 8.1659 | 0 | −8.1659 |
| Minute QQQ（vs GNO） | Calendar violation (%) | 25.4820 | 0 | −25.4820 |
| Minute SPY（vs VAE） | Calendar violation (%) | 21.0067 | 0 | −21.0067 |

**解讀。**

- **Overall 是真 capability，但幅度小且不普遍**：日線 7 個資產中 6 個 overall 最低，NVDA 是例外（2.54 vs 2.46）。reliable 區剛好相反，asset-equal 1.29 vs 1.27，HyperIV 在四個單名「marginally better」。注意原文文字說 HyperIV 在四個單名較好，但 Table 1 的 bold 顯示 AAPL（1.18 vs 1.18）與 NVDA（1.80 vs 1.80）是並列；minute SPY reliable 也出現文字說 AddMart 最佳、表上 GNO 0.55 較低的張力。這些是**值得核原文表格與文字一致性**的點，不是本頁能裁定的。
- **Arbitrage 才是結構性差異**：AddMart 在 801×30 全支撐 grid 上，calendar 與 butterfly 都是 0（numerical precision 內）；SSVI 也 0，但每個曲面要獨立優化；Deep Smoothing 仍有小 butterfly 違規；amortized 對手中 HyperIV 日線 calendar 0.60 – 2.62%、butterfly 1.63 – 8.17%，GNO calendar 高達 22.28%。**但要注意測試曲面本身已過 static-arbitrage cleaning**，所以「零違規」的絕對難度被降低；真正有意義的是：對手在同樣清理過的數據上仍然違規。
- **RND 與尾部**：HyperIV 在 NVDA 一年期有 21.8% integrated negative mass；CVaR 實驗中 AddMart 在 6 個 asset–horizon 裡 5 個最低（QQQ 180d 是 HyperIV 2.88 優於 3.05）。更關鍵的是未修補前的 invalid-density rate：HyperIV 10%–100%、GNO 5.2%–100%、VAE 100%——baseline 的 CVaR 依賴事後 clip/renormalize，AddMart 不需要。
- **可能被高估的部分**：這是重建／定價任務，不是可交易 alpha；mid-quote 進、mid-quote 出，所讀文本未見以 bid/ask 可成交價或交易成本計算的 PnL。cross-asset zero-shot 的整體優勢也有限：AddMart overall 5.51 最低、贏 HyperIV 76.2% 的 directed pairs，但 reliable 區 HyperIV 4.81 優於 5.33，transfer penalty 3.34×。壓力測試（2025-04-01–10、2026-03-23–31）顯示所有方法都退化，AddMart 的優勢是「可控退化 + 約束不失效」，不是「不退化」。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **§6 Conclusion**：additive increment structure 造成 long-horizon skew flattening，LEAPS 精度下降；對高度不對稱／厚尾標的（如 VIX）效果較差；serial convolution 與 IV inversion 使其不適合 ultra-high-frequency inference，但 minute scale 仍實用。
- **Appendix G.5**：SPY 上 ATM-skew term structure 約一年內可跟，之後預測 skew 比市場衰減更快；AAPL 554-day slice 預測 smile 明顯偏平。VIX 的誤差集中在最短到期，隨期限快速下降；作者歸因於 shared additive increments 對跨到期施加了比股票期權更強的 restriction。**即使 VIX 配適差，calendar／butterfly 仍為零**——失效是 market fit，不是 measure validity。
- **Appendix G.5 實務影響**：LEAPS 報價稀疏、spread 較寬；VIX 是不同標的過程；兩者不驅動主要 benchmark 結果，但不應作為 long-horizon tail hedging 或短端 VIX trading 的 standalone pricing model。

### 6.2 推斷的隱含假設（本頁推斷，非作者自述）

- **條件獨立增量 → 跨到期依賴受限**：模型用同一串 independent L_g 串起所有 maturity；這對 forward skew 的持續性是天生的限制。Appendix C.4 的 Proposition 3 已給 standardized log-skew 以 O(j^{-1/2}) 衰減、標準化後趨向 Gaussian，這是結構性質，不是訓練不足。
- **Static no-arbitrage ≠ dynamic no-arbitrage**：Theorem 1 只保證每個時點的 butterfly／calendar；不對 smile dynamics、對沖損益或動態交易策略的無套利做保證。把它當「動態無套利模型」是誤讀。
- **測試集已被清理**：Appendix E.1 對每個 asset／scale 做 static-arbitrage cleaning 後才保留曲面；因此「對原始未清理報價的違規率」所讀文本未見，圖表／附件待核。推斷：在極端日或深翼原始報價上，零違規的邊際可能縮小（未驗證）。
- **參考集與樣本選擇**：固定 3 maturity × 3 delta，且「A surface that cannot provide all nine contracts is discarded」。推斷：結果覆蓋的是流動性足以湊出 9 個合約的曲面；不流動標的、極端短端或無法配對 forward 的日子被排除，這是一種樣本選擇，不是全市場無條件覆蓋。
- **European-equivalent 假設**：§3 自述 listed equity／ETF options 是 American-style，但用 liquid OTM quotes（immediate exercise 無價值）轉成 European-equivalent Black–76 IV；「limits early-exercise and dividend effects」而非消除。推斷：深度 ITM、高股息或 hard-to-borrow 名稱不在保證範圍。
- **Risk-neutral ≠ physical**：§5.3 與 G.1 都明說 CVaR／crash prob 是 option-implied risk-neutral tail exposure，不是 realized loss forecast。把它當避險績效預測會錯。
- **容量／延遲**：full-support FFT 在 B32 峰值 2939.44 MiB；單面 end-to-end 7.60 ms。推斷：多資產、多到期、大量客製 strike 的即時查詢成本會線性上升；HFT 場景不適用。
- **跨資產非 invariant**：transfer penalty 3.34×，SPY→NVDA reliable 15.61、NVDA→SPY 19.66；推斷：不能把單一 source checkpoint 當通用曲面先驗，資產相似度主導 transfer 品質。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| HyperIV（Yang et al., 2025） | 直接 IV 回歸 + 三項 soft regularizer；amortized few-shot | 原文稱沿用 released set-embedding／hypernetwork | 主要對手：overall 略輸、reliable 略勝；違規率非零 |
| SSVI（Gatheral & Jacquier, 2014） | per-surface 4 參數校準；零違規但每面優化 | 經典公開 | 非 amortized，延遲不可比 |
| Deep Smoothing（Ackerer et al., 2020） | per-surface NN + penalty | 公開（NeurIPS 2020） | 小 butterfly 違規殘留 |
| GNO／Operator Deep Smoothing | operator 架構、無 arbitrage penalty | 公開 | 不穩定、違規率高（calendar 高達 22.28%） |
| VolNP（Zhuang & Wu, 2025） | SABR-derived pretraining + likelihood | 未在本文 benchmark | 只列 related work |
| Ning et al. VAE／VolGAN | 生成式曲面，非 quote-conditioned few-shot | 公開 | 不同 setting |
| Lin & Liu（2024） | additive logistic martingale、parametric family | — | 共享 additive backbone，非 mixture／set encoder |

🎤 **Interview Tip**

- **正確答**：「AddMart 的重點不是網路更大或 penalty 更重，而是換了輸出物件——輸出增量密度，讓 no-arbitrage 變成座標變換的性質。Calendar 來自 convex order（Kellerer），butterfly 來自每個 marginal 是非負機率測度；loss 裡沒有 penalty，RND 直接合法。代價是 independent additive increments 表達不了 long-horizon skew persistence，所以 LEAPS 與 VIX 是已知失效區。」
- **錯答**：「它在 loss 裡加了更嚴格的 butterfly／calendar penalty，所以違規率低。」——這正好把 hard constraint 說成 soft penalty，忽略 Theorem 1 的 by-construction 論證。
- **錯答**：「它保證動態無套利，所以可以直接做對沖策略。」——只保證 static；dynamic hedge PnL 不在 Theorem 1 範圍。

### 7.1 可證偽預測（帶日期）

- **2026-12-31 前**：若用未經 Appendix E.1 static-arbitrage cleaning 的原始 OPRA 報價重跑，AddMart 的 calendar／butterfly pair violation 在 10⁻¹⁰ 容差下仍應為 0（架構性質）；但 IV-MAE 應高於 Table 1（未驗證）。
- **2026-12-31 前**：在 >365 天 LEAPS 或 VIX 上套同一 9-quote 協議，AddMart 相對 HyperIV 的 overall 優勢應消失或反轉；作者已在 G.5 給方向，但未給完整 cross-asset 表（未驗證）。
- **2026-12-31 前**：若仍無官方 repo／checkpoint，獨立複現門檻維持「需商業 OPRA 數據 + 自行實作 full-support FFT decoder」，第三方無法只靠論文數字重現 Table 2 的零違規稽核（TBD）。

## §8 · For the Reader

- **因子研究員**：把 P_j 的 CDF、廣義分位數、crash probability 當特徵，而不是把 IV 曲面當特徵；但緊記這是 risk-neutral，不是 physical。LEAPS 與 VIX 的 skew 會被抹平，別把它們的尾部特徵餵進 factor。
- **高頻執行**：單面 end-to-end 7.60 ms、B32 1.040 ms／960 surfaces/s，是 minute-scale 報價刷新與一般 market-making 的量級，不是 HFT。要壓延遲就得放棄 full-support FFT，但 Lemma 2 說 cropped 版本會引入 calendar residual（ablation 中 circular 版 SPY 3.652%、NVDA 0.070%）。
- **組合配置**：可直接用輸出的 marginal 算 CVaR／crash prob 做情景，但這是 option-implied 風險中性測度；若要轉成 physical，需自己加風險溢價調整。論文未給交易成本或執行延遲下的 PnL。
- **LLM-agent／tool 設計**：適合包成 pricing／risk 的 deterministic tool（輸入 9 quotes、輸出曲面＋RND），不適合當 alpha 生成器；它的價值是「輸出永遠是合法機率測度」，減少下游程式的防禦性 clip。
- **RL 策略**：不直接相關；但可當環境中的觀測生成器或 reward 的風險度量來源，前提是策略 horizon 不落在 LEAPS／VIX 失效區。
- **研究學生**：讀三個地方就夠——Theorem 1（by-construction 的證明骨架）、Lemma 2（cropped decoder 的誤差界，告訴你「訓練近似、推理精確」怎麼寫）、Proposition 3（additive 類的 long-horizon 結構限制）。這是「把硬約束寫進生成過程」的標準模板，可搬到任何曲面／期限結構問題。

## References

- Luo, Lin, Luo, Minh, Zhang, Liu. *Arbitrage-Free Implied Volatility Surface Reconstruction via Additive Martingales.* arXiv:2610.25671v1 [q-fin.CP], 19 Oct 2026. CC BY 4.0. https://arxiv.org/abs/2610.25671v1
- Kellerer (1972). *Markov-Komposition und eine Anwendung auf Martingale.* Mathematische Annalen.（convex order／peacock 的理論基礎）
- Breeden & Litzenberger (1978). *Prices of state-contingent claims implicit in option prices.* Journal of Business.（RND 與 call 二階導）
- Carr & Torricelli (2021). *Additive logistic processes in option pricing.* Finance and Stochastics.（additive martingale lineage）
- Hirsch, Profeta, Roynette, Yor (2011). *Peacocks and associated martingales, with explicit constructions.* Springer.
- Yang, Chen, Shu, Hospedales (2025). *HyperIV: real-time implied volatility smoothing.* ICML.（主要 amortized 對手）
- Ackerer, Tagasovska, Vatter (2020). *Deep smoothing of the implied volatility surface.* NeurIPS.（per-surface penalty 路線）
- Gatheral & Jacquier (2014). *Arbitrage-free SVI volatility surfaces.* Quantitative Finance.（SSVI 基線）
- Zaheer et al. (2017). *Deep sets.* NeurIPS.（permutation-invariant set encoder）
- Rockafellar & Uryasev (2000). *Optimization of conditional value-at-risk.* Journal of Risk.（CVaR 表示）
- 數據鏈：OPRA top-of-book via massive.com；Nasdaq BBO via Databento XNAS；forward 由 put–call parity 推導，IV 用 Black–76 反解。