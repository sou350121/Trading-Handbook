<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=元学习搜索 alpha=端到端表征 autonomy=全自动黑盒 -->

# TabFM (Google Research) 解構（TabFM (Google Research)）

> **來源發布**：2026-11-10 · （無 venue） · arXiv [2611.13550](https://arxiv.org/abs/2611.13550)
> **arXiv 原文**：[Tabular Foundation Models in Financial Predictions](https://arxiv.org/abs/2611.13550v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落點為表格 ICL／中长周期／元学习搜索／端到端表征／全自动黑盒；它把表格 foundation model 的 in-context learning 用於 next-year earnings-change 方向預測，補上「免任務訓練、免調參，且避免 LLM 式記住公司結果」的 prior gap。此處的「量价表格」在原文實際是財務報表 firm-year panel，不是高頻量价資料，外推時須保留這個差異。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `中长周期` | `元学习搜索` | `端到端表征` | `全自动黑盒` |

**Status:** v0.5 — 基於arXiv 原文（有原文則以原文為準）。細節待升 v1。

**TL;DR:** ① 把 Google Research 的 TabFM 引入會計盈餘方向預測：合成表格預訓練＋ICL，推理不微調、不調參。② 核心 trick 是把「如何從 context 抽取預測關係」放進預訓練，使用時只餵歷史樣本與待預測樣本。③ 對「元学习搜索」軸★的意義是：配置搜索被前置，推論端只剩 context 長度與 ensemble 成本。④ 關鍵實證：DF 樣本 2023–2024 pooled AUC 69.53%，Llama 67.88%，樹模型 62.75%–63.78%；但這是單一 benchmark、單一樣本、130 predictors 的結果，不能外推為 universal SOTA。

**X-Ray.** 五軸上它是「表格 ICL＋合成預訓練」，把選模／調參從每個任務搬到預訓練，推理只付 context 前向成本；這解了傳統樹模型逐年 grid search 與 LLM 微調的工程坑，也避開「記住公司結果」這類最粗暴的泄漏。但 envelope 同樣清楚：無會計知識、忽略欄名與列序，panel 的同一公司重複觀測與時變關係不是預訓練分佈；130 predictors 以上、低延遲、多批次推論與高 GPU 記憶體都是未解區。對量化讀者，最有價值的不是 69.53% 這個 pooled AUC，而是「免訓練」換來「重複推論成本」的成本結構，以及單一 benchmark 不能外推成 universal SOTA 的警告。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 改動 | 前作／對照 | TabFM 做法 | 代價／邊界 |
|---|---|---|---|
| 預訓練資料 | 樹模型：任務內訓練；Llama：文字預訓練後 fine-tune | 數億合成表格（SCM 生成），參數學「如何從 context 學」 | 無會計知識、忽略欄名；預訓練上限 16,384 rows／100 columns |
| 適配方式 | 樹模型：grid search＋validation；Llama：fine-tune 大量參數 | ICL：固定參數，餵 in-context training sample＋test rows | 每次預測重跑 context；時間與 GPU memory 隨 context 上升 |
| 表格表徵 | 樹：threshold splits；LLM：token sequence | row／column attention＋row compression＋in-context transformer | 列序被忽略；panel 重複觀測與時變關係未內建 |

### 1.2 ⚡ Eureka

一句話：把「學習怎麼學」前置到合成表格預訓練，推理時只做 context-conditioned prediction；因此沒有 task-specific gradient step，也沒有 grid search。直覺是：樹模型把成本花在每次任務的訓練／調參，Llama 把成本花在 fine-tune，TabFM 把成本花在每次把歷史樣本讀進 context。

### 1.3 信息流

數值表格（rows=firm-years, cols=financial items）→ 每格轉 vector → row／column attention 交替 → 每列壓成 row vector → in-context transformer 讀 training rows＋test rows → logits／probability → ensemble 平均（主規格 8 members）→ 預測 next-year earnings increase 機率

## §2 · 數學層

📌 **Napkin Formula**：  
\[
\hat{p}(y_i^{+}\mid x_i,\mathcal{D}_{\text{ctx}})=f_{\theta}(x_i;\mathcal{D}_{\text{ctx}})
\]
其中 \(\theta\) 在預訓練後固定，\(\mathcal{D}_{\text{ctx}}\) 由使用者提供；預訓練以合成 SCM 表格反覆預測 withheld outcomes，用 prediction error 更新 \(\theta\)。

**複雜度**：所讀文本未給解析式；只給定性——推理時間與 GPU memory 隨 in-context training sample 大小上升，且每個 ensemble member 需單獨前向，時間隨 member 數上升。精確 FLOP／複雜度公式：TBD。

**Loss／訓練細節**：TabFM 支援 regression 與 classification；本頁讀到的是 classification checkpoint，輸出 earnings-increase 機率。無 fine-tuning、無 configuration search；preprocessing 參數由 in-context sample 估計，欄名不進入模型。預訓練資料由 computer programs 生成，非實際公司；預訓練截止日所讀文本未披露。

## §2.5 · 帶數字走一遍（Worked Example）

以下為**假設／示意**，不是論文實證；不使用論文數字，也不把示意值帶入 §5。

- **Step A**：假設 context 為 \(\mathcal{D}_{\text{ctx}}=\{(x_j,y_j)\}_{j\in\mathcal{J}}\)，其中 \(y_j\in\{y^{+},y^{-}\}\)，待預測列為 \(x_{*}\)。
- **Step B**：TabFM 在 context 內對每個 predictor 做縮放；欄名被忽略，模型只看到數值。
- **Step C**：row attention 讓 \(x_{*}\) 的每個會計科目與同列其他科目交互；column attention 讓它與 context 中其他 firm-years 的同科目比較。
- **Step D**：每列壓成 row vector；in-context transformer 以 training rows 的 \(y_j\) 為條件，輸出 \(\hat{p}(y_{*}^{+}\mid x_{*},\mathcal{D}_{\text{ctx}})\)。
- **Step E**：多個 ensemble members 對同一資訊做不同 normalization、column order、outcome coding，平均後給最終機率；若用 TabFM+，則以 validation predictions 加權並 calibrate。

## §3 · 數據層

- **規模／頻率／市場／時段**：DF 提供的會計 panel，62,796 firm-years，7,446 firms，financial statement years 2005–2024，全部 December 31 fiscal year-end；130 predictors。主測試用 6,569 firm-years（2023–2024 報表），預測 2024／2025 earnings changes。
- **怎麼來**：DF 樣本；TabFM 不做 task training。每個 test year 用前 18 年作 in-context training（2023 用 2005–2022；2024 用 2006–2023），沒有 hold-out validation year。
- **樣本外與容量假設**：主測試受 Llama 限制只用 2023／2024；TabFM 可回測更早年份。預訓練上限 16,384 rows／100 columns，實際 application 約 56,000 in-context observations／130 predictors，屬外推；來源明說不能由此推論更大 predictor set 的 scaling。
- **資料形狀**：130 predictors 由 26 個損益表變數（當期與前兩年）＋26 個資產負債表變數（當期與前一年）組成；金額多為 unscaled millions，EPS 等六個 predictor 為 dollars。

**時間邊界分列**

| 邊界 | 所讀文本 |
|---|---|
| 預訓練 | 合成 SCM 表格，數億 datasets；預訓練資料截止日未披露；TabFM 公開發布 2026-06-30 |
| Tokenizer／欄名 | TabFM 只吃數值、忽略欄名；沒有 Llama 式文字 tokenizer 邊界可套（本頁推斷） |
| 適配 | ICL，參數固定；無 fine-tuning；每次預測重讀 context |
| Topic 擬合 | 不適用；所讀文本未見 topic model |

Llama 3.2 的 knowledge cutoff 是 December 2023（Meta 2024），不可套到 TabFM；DF 用 2023／2024 測試年避開 Llama 記憶，來源 footnote 10 仍討論 2024 Q1／Q2 已在 Llama 3.2 發布時公布。TabFM 則在 2026-06-30 發布，晚於 2026 Q1 earnings；來源說完整測試要等 fiscal 2027 earnings 公布（early 2028），這是 technological look-ahead bias 的來源之一。

## §4 · 代碼層

| Repo／Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|
| google-research/tabfm；Hugging Face `google/tabfm-1.0.0-pytorch`，revision `77cb9cc1b4fd3a9c77fbb9552c218200bb4dab83`；`tabfm 1.0.1`；PyTorch 2.8.0／CUDA 12.8 | 所讀文本未見（TBD） | 中高：模型與設定可查，需 GPU；`max_num_rows=None`、`max_num_features=500`，seeds 固定，test batches 最多 9,999 | DF 提供樣本；公開下載連結所讀文本未見 |
| Meta Llama-3.2-3B-Instruct／DF 的 SFT-Llama predictions | 所讀文本未見（TBD） | 高：DF fine-tune 約 108 hours on one NVIDIA H200 GPU；本頁用 DF 的 observation-level predictions | DF 提供 predictions |
| RF／SGB／GBDT-X：scikit-learn 1.9.1／XGBoost 3.4.1 | 所讀文本未見（TBD） | 中：Appendix A.2 給 grids 與選定配置；RF 在 CPU，SGB／GBDT-X 在 GPU | 特徵來自 DF 樣本 |

## §5 · 評測 / Benchmark

| 數據集／市場 | Metric | 前SOTA（此處為基線，非文獻 SOTA） | 本方法 | Δ |
|---|---|---|---|---|
| DF 會計 panel，2023–2024 pooled | Pooled AUC (%) | Llama：67.88 | TabFM：69.53 | — |
| 同上 | Pooled AUC (%) | GBDT-DF：62.75 | TabFM：69.53 | +6.78pp |
| 同上 | Pooled AUC (%) | RF：63.51 | TabFM：69.53 | +6.02pp |
| 同上 | Pooled AUC (%) | SGB：63.78 | TabFM：69.53 | +5.75pp |
| 同上 | Pooled AUC (%) | GBDT-X：63.57 | TabFM：69.53 | +5.96pp |

Δ 定義為「TabFM − 基線」，只列可由同一列兩個逐字數值直接算出的差。Llama 列來源 Table 2B 逐字報告 pairwise 差異 1.66（p=0.001），但顯示 AUC 四捨五入後與算術差不等，故 Δ 留空；來源也說 AUC differences 由 unrounded values 計算。前SOTA 欄不是文獻 SOTA，而是原文的比較基線。

**解讀**：對樹模型的 Δ 發生在同一年、同 130 predictors、同一批 test observations，且來源的年度、omit、small-sample 分析大致維持 TabFM 排序；這比較像 capability，而不是單純調參運氣。但 TabFM 主規格是作者選定的 8-member 版本，其他 TFM／版本差距小，來源自己說 TFM 仍 early stage，所以 69.53% 不宜當成經嚴格 nested 選模後的無偏估計（本頁推斷）。Llama 比較受 2023／2024 窗口與其 pretraining 限制；TabFM 的合成預訓練降低「記住公司結果」，但不排除設計層面的 look-ahead（作者 footnote 2）。成本只計時間、記憶體、FLOP，未計金錢、延遲 SLO、交易成本；AUC 不是淨收益，來源 footnote 3 也明說 AUC 可能不捕捉實務價值。

**穩健性與反證**：年度 2011–2024 的 pooled AUC，TabFM 69.52%、SGB 64.82%（Table 3A）；TabFM 在 14 年中 13 年高於 SGB、14 年高於 RF 與 GBDT-X，唯一例外是 2018 年略低於 SGB。5% 子樣本中 TabFM AUC 64.47%，仍高於樹模型的全樣本 AUC；drop DF top 20 predictors 時，TabFM 掉 1.31 個百分點，樹模型掉 1.95 到 2.50 個百分點。perturbation 測試中，randomizing 最後一位數只讓 TabFM 掉 0.38 個百分點，落在樹模型範圍內；Levy 的 GPT-4 測試則失去全部預測力。這些是同一 DF 樣本內的穩健性，不是跨市場因果；來源 footnote 5 也提醒，樹模型或 Llama 若更 extensively trained／fine-tuned，或在其他 context（bankruptcy、clinical）中，可能追上或超過 TFM。

**成本對照**：TabFM 兩個 test years 的預測約 11.83 minutes、peak GPU memory 24.27 GiB；Llama fine-tuning 約 108 hours on one NVIDIA H200 GPU、peak 134 GiB，forecasts 約六分鐘；GBDT-DF 的 configuration search 約 120 hours on a 128-core CPU server。SGB 約 7.5 minutes，RF 與 GBDT-X 約 130 與 140 minutes，但樹模型幾乎全部時間在 up-front training，訓練後 incremental forecasts 近乎 instant；TabFM 沒有 training step，時間每次預測重複發生。來源明說：同一 trained model 要產很多預測時樹模型更有效率；training sample 常更新或 accuracy 優先時 TabFM 更有優勢。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations

- **單一 benchmark、單一樣本、單一 predictor set**：不能建立某一 forecasting technology 的普遍優勢；沒有 class of models 可能在所有 settings 勝出（Abstract、§1、§6）。
- ** predictor 上限**：比較用最多 130 個 financial statement predictors，不能說明更大 predictor set 的 performance 與 computational cost（§1、§2.2）。
- **無會計知識、忽略欄名**：TabFM 不知道欄位是 net income 或 total assets；學到的關係只來自 context 數值（§1、§2.3）。
- **panel 假設不匹配**：預訓練每列獨立，會計 panel 有同一公司重複觀測；預訓練單一 generating process，會計關係隨時間變；TabFM 忽略 row order，不能區分較舊與較近觀測（§2.3）。
- **look-ahead 非零**：合成預訓練不讓模型記住公司結果，但不排除所有 look-ahead；若 synthetic data 近似公司 earnings-generating process 且過程隨時間改變，in-sample tests 可能高估 real-time performance；設計選擇也可能反映當前知識；TabFM 在 forecast dates 尚不存在，屬 technological look-ahead bias（footnote 2、§6）。
- **依賴開發者說法**：使用者必須相信 Google 對預訓練資料的描述；作者說沒有理由懷疑，測試也未見 memorization 跡象（footnote 2、§6）。
- **TabFM-Auto 未評估**：因其用 LLM 建構 predictors，作者擔心 LLM 以未來資訊選 predictor 的 look-ahead（footnote 6）。
- **其他 TFM 早期**：TabICLv2、TabPFN-3.5、TabFM 1.1.0 比較只宜保留，不宜強推論（§5）。
- **基線可能未盡力**：樹模型或 Llama 若更 extensively trained／fine-tuned 可能更好；樹模型在其他 context 也能 match／outperform TFM（footnote 5）。
- **成本數字跨硬體**：Llama 與 GBDT-DF 的時間來自 DF 的不同硬體，只能看 orders of magnitude（§3.4）。
- **AUC 的實務限制**：one-year-ahead earnings changes 未必是最 relevant task；AUC 未必捕捉預測差異的實務價值（footnote 3）。
- **組合未測**：來源說結合 technologies 可能優於單用，例如 LLM 建 features 給 TFM／樹模型，但本文未評估（§1、§6）。

### 6.2 推斷的隱含假設（本頁推斷，非作者實證）

- **Point-in-time context 必須乾淨**：TabFM 的 leakage 主要取決於使用者給的 in-context sample；若 context 含重編、倖存者篩選或未來才知道的樣本，ICL 仍會學到泄漏。來源說 context 由使用者控制，但未測 restatement／survivorship 情境。
- **Panel 獨立性近似**：模型忽略 row order、預訓練每列獨立；推斷同一公司重複觀測的 within-firm correlation 不會被顯式建模。來源說可加 time predictors 或限近期樣本，但作者未探索；這不是已證明無效，而是尚未測試。
- **Regime 穩定**：年度 AUC 從 62.82% 到 78.86%，四模型 annual AUC pairwise correlations 0.89 到 0.99；推斷單一年或兩年結果對 regime 很敏感，來源自己也說 few test years 要 cautious。
- **容量與部署**：peak GPU memory 24.27 GiB、peak RAM 13.69 GiB；推斷需要高階 GPU，且每次預測重跑 context，不適合低延遲或超高頻。
- **成本邊界**：來源只量時間、記憶體、FLOP；推斷未計金錢、能源、排隊、除錯與 monitoring 成本，也未計交易成本與 capacity。
- **Survivorship**：所讀文本未見明確 survivorship 處理說明；不能把輔助資料缺失直接當存活者偏差，但若 DF 樣本要求連續可得資料，仍需核其附錄與樣本建構（TBD）。
- **模型選擇風險**：TabFM 不調參，但 ensemble size、版本、TabFM+ 仍是選擇；來源說主規格 AUC 最高，卻也說 TFM 早期、不宜強推論。推斷 69.53% 含某種 model-selection optionality。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| RF／SGB／GBDT-X／GBDT-DF | 任務內訓練＋grid search；incremental predict 近即時；AUC 較低 | scikit-learn／XGBoost open-source；license TBD | mature baseline |
| Fine-tuned Llama-3.2-3B-Instruct | 文字預訓練＋fine-tune；AUC 67.88；約 108 hours on H200 | open-weight；license TBD | DF comparison |
| TabICLv2 | TFM；1.17 minutes；AUC 67.57；GPU 30.46 GiB | 套件／checkpoint 來源提及；license TBD | early |
| TabPFN-3.5 | TFM；2.57 minutes；AUC 68.55；GPU 3.79 GiB | 套件／checkpoint 來源提及；license TBD | early |
| TabFM 1.1.0 | 約四分之一參數；時間約半；AUC 68.36（8 members） | checkpoint 來源提及；license TBD | early |
| TabFM+ | 32 members＋engineered predictors＋validation weighting／calibration；82.72 minutes；AUC 69.13 | preset 來源提及；license TBD | early |

🎤 **Interview Tip**  
**正確答**：TabFM 的賣點是 synthetic pretraining＋ICL，沒有 task-specific training／tuning；推論成本隨 context 與 ensemble 重複發生。比較要分開看 AUC 與成本：對 Llama 兩者都 favorably；對樹模型是 accuracy vs speed／training-update 的 trade-off。Llama 的 67.88 與 TabFM 的 69.53 是同一 DF sample、130 predictors、2023–2024 window；來源 pairwise 1.66 是 unrounded 計算，顯示值不可直接重算。  
**錯答**：說 TabFM 是 universal SOTA；說 synthetic pretraining 完全消除 look-ahead bias；說 TabFM 有會計知識或讀懂欄名；把樹模型數字合成一個假 SOTA 區間再算 delta；把 AUC 當交易淨收益。

### 7.1 可證偽預測（本頁研究假說，尚待驗證）

- **2028 年初**：fiscal 2027 earnings 公布後，可用 fiscal 2026 statements 做 TabFM 完全事後未知的測試（來源說 firms report fiscal 2027 earnings in early 2028）。若 TabFM 仍維持對樹模型的排序，支持合成預訓練無記憶泄漏；若排序翻轉，look-ahead／regime 解釋上升。
- **下一個 open-weight LLM 發布後（來源未給日期；以 2028 年初前為觀察窗）**：來源作者明說 newer open-weight model 按 DF 方式 fine-tune 後「quite possible」可超過 TabFM；本頁預測至少一個新 fine-tuned LLM 會與 TabFM 1.0.0 8-member 統計打平或更高。
- **在 >130 predictors 的同一樣本重跑（無來源日期）**：來源未測；本頁預測 TabFM 相對樹模型的優勢可能縮小，因預訓練上限 100 columns、release 上限 500 predictors，且來源說不能推論更大 predictor set 的 scaling。

## §8 · For the Reader

- **因子研究員**：把 TabFM 當 nonlinear ensemble baseline，而不是「懂會計的模型」。用 point-in-time context，報告 AUC 與經濟價值；測 130+ predictors、industry subsets、不同年份。若只報 pooled AUC，容易把 2017 這類高預測度年份的表現誤當 stable alpha。
- **高頻執行**：不要用於低延遲路徑。TabFM 兩個 test years 的預測約 11.83 minutes，且每次預測重讀 context；樹模型訓練後 incremental forecasts 近乎 instant。適合 offline batch、隔夜刷新或研究平台，不適合 event-driven／intraday。
- **組合配置**：把 AUC 當 ranking signal 的中間指標，不是淨收益。來源的 annual variability、small-sample 與 omit 分析支持「不是單純過擬合」，但沒有交易成本、capacity、滑價；配置前要自己做 economic value 與 capacity 測試。
- **LLM-agent**：把 TabFM 當數值 predictor，不要讓它假裝讀懂 line-item labels；它忽略欄名。Agent 應提供 raw table＋historical labels，並檢查 context 是否 point-in-time。要用 LLM 建 features 時，注意 TabFM-Auto 被作者排除的理由：LLM 選 predictor 可能引入 look-ahead。
- **RL 策略／研究學生**：這是很好的「免調參 baseline」與「成本結構」教材。可復現 checkpoint 與 DF 樣本，跑 2011–2024 年度、omit、5% 子樣本、firm-count、perturbation；再測 row order／time features。不要把單一樣本的相关排序写成所有模型／市場的必然因果；來源自己也說 no class likely dominate in all settings。

## References

- deHaan, E., & Huang, A. H. 2026. *Tabular Foundation Models in Financial Predictions.* Technical note. arXiv:2611.13550. https://arxiv.org/abs/2611.13550
- Kong, W., & Das, A. 2026. *Introducing TabFM: A zero-shot foundation model for tabular data.* Google Research Blog, June 30.
- Kong, W., Louidor Ilan, E., Nie, S., Narayan, T., Sen, R., Zhou, Y., Fu, D., Oymak, S., & Das, A. 2026. *TabFM: A zero-shot foundation model for tabular data.* arXiv:2609.37959.
- Google Research. 2026a. *TabFM: Tabular foundation models.* GitHub `google-research/tabfm`; Hugging Face `google/tabfm-1.0.0-pytorch`.
- Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. 2023. *TabPFN: A transformer that solves small tabular classification problems in a second.* ICLR.
- Hollmann, N., et al. 2025. *Accurate predictions on small data with a tabular foundation model.* Nature 637: 319–326.
- Qu, J., Holzmüller, D., Varoquaux, G., & Le Morvan, M. 2025. *TabICL: A tabular foundation model for in-context learning on large data.* ICML.
- Qu, J., et al. 2026. *TabICLv2: A better, faster, scalable, and open tabular foundation model.* ICML.
- Jäger, B., Erickson, N., Grinsztajn, L., et al. 2026. *TabPFN-3.5: Technical report.* arXiv:2609.17895.
- Dou, Y., & Fu, Q. 2026. *Supervised fine-tuning of large language models on financial statements to predict future earnings changes.* Working paper, NYU. SSRN 6484841.
- Levy, B. 2026. *Caution ahead: Numerical reasoning and look-ahead bias in AI models.* Journal of Accounting Research 64 (3): 1139–1188.
- Chen, X., Cho, Y. H., Dou, Y., & Lev, B. 2022. *Predicting future earnings changes using machine learning and detailed financial data.* Journal of Accounting Research 60 (2): 467–515.
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. 2022. *Why do tree-based models still outperform deep learning on typical tabular data?* NeurIPS 35: 507–520.