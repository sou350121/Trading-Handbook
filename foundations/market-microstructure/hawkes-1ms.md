<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=风险择时 autonomy=人机协同可解释 -->

# Hawkes过程 + 1ms事件定义 解構（Hawkes过程 + 1ms事件定义）

> **來源發布**：2026-10-26 · （無 venue） · arXiv [2610.36317](https://arxiv.org/abs/2610.36317)
> **arXiv 原文**：[Event definition and the branching ratio: Hawkes estimates of price-change clustering in the consolidated quote feed](https://arxiv.org/abs/2610.36317v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 consolidated NBBO 的「事件」從 feed 記錄的每一次 mid 變化，重定義為「持續 1ms 的 mid 水平」，證明 large-tick stocks 的亞毫秒 Hawkes 核與高分叉比主要是鎖盤／耗竭報價的機械配對，而非交易者反應。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `风险择时` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** ① 以「mid level 持續 1ms」重定義事件，合併 locks 與 sub-ms depleted states；② 用合成 consolidated feed 把量到的 brief states 加回擬合模型，復現 raw Hawkes 的亞毫秒核與分支比偏差；③ 這打在「因果結構」軸上：raw Hawkes 把 feed 的 mechanical pairing 讀成 excitation，branching ratio 不是 endogeneity 的乾淨測量；④ 112 stock-months 中 raw/clean n(1s) 中位數 3.1 [2.0–4.2]，raw 一秒內核質量 0.63 對 clean 0.20。

**X-Ray.** 這篇不是 alpha 論文，是 measurement / construct-validity 論文：它把「事件定義」本身當成識別問題的來源，而不是事後校正估計器。對做過 TAQ/NBBO 回測的人，最痛的是跨股票、跨月、跨論文的 Hawkes 分支比不可比——本文給出機制：large-tick stocks 約一半 NBBO 變化是 lock 或 depletion，1ms 內成對出現，fastest kernel 的權重幾乎等於「1ms 內有後續事件的比例」。它解了舊工程坑：不會再把 feed 的延遲偽影當成市場臨界性。但它打不開的 envelope 也很清楚：沒有交易信號、沒有樣本外收益、沒有成本模型；E1m 是 retrospective 定義，實時系統不能直接用；樣本只有 2017 年 8 檔股票，SIP 延遲此後已下降。對量化讀者的意義是負面的但重要：若你的 microstructural feature 用 raw event count 或 branching ratio，跨樣本比較前先聲明 threshold。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（raw NBBO Hawkes） | 本文（1ms persistence event） |
|---|---|---|
| 事件定義 | 每次 NBBO mid 變化即事件 | mid level 持續 ≥1ms；合併 locks 與 sub-ms states |
| 亞毫秒核 | 真實 excitation，約 40% 事件歸因於此 | feed artifact：lock/depletion 的 mechanical pairing |
| 分支比 | 0.6–0.7，讀作 endogeneity／near criticality | raw 混入 artifact 與 dilution；改報 n(1s)、n(10s) |
| 識別策略 | 估計器校正或直接解讀 | 合成 feed simulation，把 observation mechanism 加回 |

### 1.2 ⚡ Eureka

**一句話 trick：** 把事件從「feed 記錄了一次 mid 變化」改成「mid 水平持續了 1ms」，亞毫秒核就消失——因為它本來就是同一個價格移動被 consolidated feed 拆成兩個記錄事件。

**直覺：** 在 one-tick spread 的股票，價格上移通常需要兩個 venue 先後更新。SIP 先收到 bid 上移、後收到 ask 上移，中間 publish 一個 locked NBBO；mid 先走半個 tick、再走半個 tick，0.6ms 內出現兩次「事件」。Hawkes 只有一種方式解釋 1ms 內兩次事件：excitation。所以它把 feed 的處理順序擬合成交易者反應。

### 1.3 信息流 ASCII 圖

```
Exchange X bid 20.01 ──┐
                       ├─→ SIP consolidation（delays differ by venue）
Exchange Y ask 20.01 ──┘
        │
        ▼
Raw NBBO: 20.005 → 20.010 (locked) → 20.015
          E0: 2 up events, 0.6 ms apart
        │
        ▼ merge every state < h = 1 ms
Clean NBBO: 20.005 → 20.015
          E1m: 1 up event, stamped when new level established
        │
        ▼ bivariate Hawkes fit
Raw:   sub-ms spike + branching ratio inflated
Clean: kernel mass between ~10 ms and 30 s; n(1s) stable
```

## §2 · 數學層

📌 **Napkin Formula**

強度（gated bivariate Hawkes）：

```
λ_i(t) = 1{t ∉ D_t} [ μ(t) + Σ_j Σ_k α_ij,k β_k e^{−β_k(t−t_m^j)} ]
```

固定視窗核質量：

```
n(H) = Σ_k (a_k + c_k)(e^{−β_k δ} − e^{−β_k H}),   H > δ
```

raw 分支比的計數關係：

```
n_0 ≈ s + n_m · N_m / N_0
```

**複雜度：** 指數核的 decayed count 遞歸讓每個 likelihood evaluation 為 O(KN)；固定 β 時 likelihood 對 (μ,a,c) 凹，用 L-BFGS-B；外層對 log β 用 Nelder–Mead 多起點網格搜尋。**直覺：** refractory gate 把每個事件後 1ms 的強度設為 0，fastest component 不會被擬合到「不可能有事件」的空白區間；n(H) 是累積核質量，不依賴你把響應時間拆成幾個 exponential。**Loss／訓練細節：** 一個日曆月共用一組 kernel，baseline 為每日每 30 分鐘一格；K 由 BIC 選，112 個 clean stock-months 中 103 個選 K=4；不確定性用 parametric bootstrap 與 day-block bootstrap。

## §2.5 · 帶數字走一遍（Worked Example）

> 以下價格與時間取自原文 Table 1 的 illustrative 範例；Hawkes 參數與事件計數為**假設／示意**，不是論文實證結果。

1. **假想輸入：** 某 large-tick 股票 mid 從 20.005 上移到 20.015。交易所 X 先 post bid 20.01，SIP 先收到，publish locked NBBO 20.01/20.01；0.6ms 後 Y 的 ask 更新到 20.02，spread 回到 1 tick。
2. **Raw E0：** mid 記錄為 20.005 → 20.010 → 20.015，兩個 up events，間隔 0.6ms。假設這樣的真實移動有 100 次，raw 總事件數 N_0 = 200，其中 100 個是 1ms 內的「第二半」。
3. **量 s：** 1ms 內有後續事件的 raw 事件比例 s ≈ 100/200 = 0.50（示意值；原文樣本為 41–50%）。
4. **Clean E1m：** 合併後只剩 100 個事件，N_m = 100，N_m/N_0 = 0.50（示意值；原文中位數 0.46）。
5. **假設真實 excitation n_m = 0.40**（示意值），代入計數關係：n_0 ≈ 0.50 + 0.40 × 0.50 = 0.70。raw 分支比被推到 0.70，但其中 0.50 是 feed 配對，只有 0.20 是真實 excitation 被事件數翻倍稀釋後的結果。
6. **讀數：** 若只看 total branching ratio，raw 0.70 與 clean 0.54 的差看起來不大；但 n(1s) 的 raw/clean 是 0.63/0.20，這才是 artifact 真正藏身之處。

## §3 · 數據層

- **資料源：** NYSE Trade and Quote（Millisecond TAQ）透過 WRDS；`complete_nbbo` 補上標準 NBBO 檔省略的兩邊同一交易所報價；標準檔用於 venue 與延遲診斷。
- **期間：** 2016-10-31 至 2017-12-29，294 個交易日；剔除 1 個 10 月交易日與 3 個提早收盤日，14 個月共 290 天。每日觀測 10:00–15:30。
- **股票：** 8 檔：INTC、MU、MDLZ、BAC、WFC、T、TGT、ATVI；另在 2017 年 1 月加入 IBM、AAPL 與 Lee (2024b) 對照。112 個 stock-months 中 80 個 large-tick、18 個 intermediate、14 個 small-tick。
- **事件規模：** raw NBBO 變化每月 27,000–126,000；core stock 的 clean 事件每月 10,000–40,000。
- **樣本外與容量：** 無交易策略回測；simulation 是模型內 recovery 檢查，不是樣本外收益。容量假設未披露。

## §4 · 代碼層

| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| [github.com/hyoeun-lee/hawkes-nbbo-events](https://github.com/hyoeun-lee/hawkes-nbbo-events) | 無模型 checkpoint；repo 含 fitted parameters、bootstrap、simulation outputs；Zenodo [10.5281/zenodo.23675006](https://doi.org/10.5281/zenodo.23675006) | 未披露（TBD） | 中高：Python + 編譯遞歸；需 WRDS 訪問 | TAQ 需 license，不能公開分享；derived data 公開 |

## §5 · 評測 / Benchmark

> 本文非交易策略論文，無 IR/Sharpe/AR/MDD 可比。Metric 為 Hawkes 模型估計量；前 SOTA 欄為既有 raw-event 做法（E0），本方法欄為 clean E1m。

| 數據集/市場 | Metric | 前SOTA（raw E0） | 本方法（clean E1m） | Δ |
|---|---|---|---|---|
| US equities, TAQ NBBO, 112 stock-months | n(1s) kernel mass（中位數） | 0.63 | 0.20 | −0.43 |
| 同上 | n(10s) kernel mass（中位數） | 0.70 | 0.34 | −0.36 |
| 同上 | total branching ratio（中位數） | 0.72 | 0.54 | −0.18 |

**解讀：** 負 Δ 不是「本方法更差」，而是 clean 定義移除了 feed artifact，估計量下降。真正 capability 證據在 simulation：Table 4 中 E1m 對 true n(1s) 的恢復在 INTC 0.101 對 0.106、MDLZ 0.228 對 0.238、BAC 0.080 對 0.085、TGT 0.413 對 0.443，偏誤 −5% 到 −7%（原文：within 3–7%）。raw 的 3.1 倍 n(1s) 不是預測力，是把 lock/depletion 的機械配對計入 excitation；total 只高 33%，因為 artifact 與事件數翻倍的 dilution 部分抵消，horizon mass 才把 3 倍差異顯露出來。**可能是過擬合／前瞻偏差的部分：** E1m 是 retrospective，需等 1ms 才知道狀態是否持續，不能用於實時事件流；本文沒有交易回測，所以沒有成本未計的問題，但也沒有樣本外 alpha 證據。**不能歸因的部分：** clean 的 n(10s) 受 <5 分鐘 burst 影響，n(1s) 受 reversal 影響，兩者都是 linear symmetric kernel 的已知 misfit（§6.5），不是「clean 定義已完美」。

## §6 · 失效與隱含假設

### 6.1 論文自述 limitations（可定位）

- **樣本與時代（§7）：** 8 檔 large-tick stocks、2017 年 14 個月；feed lag 此後已下降，2024 年 Rule 612 tick amendments 尚未實施；定義可套用到新 consolidated data 與 direct feeds，但第一問題是「clean kernel 是否 whichever feed records it 都一樣」。
- **Retrospective 定義（§2.2）：** 狀態是否持續 h 只有 h 之後才知道，E1m 是價格路徑的重建，不是 real-time event stream。
- **模型形式（§7）：** linear symmetric Hawkes + piecewise-constant baseline；三個 misfit 各指向模型外的東西：依時段變化的 kernel、stochastic/regime-switching baseline、direction-dependent kernel 或 inhibition。
- **亞毫秒真實反應不可識別（§5.3）：** 一旦 brief states 存在（consolidated data 永遠有），E1m 會抹掉 genuine sub-ms component，raw fit 也無法隔離；作者只主張「consolidated data 無法識別」，不主張「不存在」。
- **BAC 的 clock caveat（§6.3.2）：** BAC 的 SIP lag 中位數 0.75ms，n(1s) 在 exchange time 低 18–28%，2ms threshold 是較安全選擇。
- **Tick regime 與 activity 混淆（§6.4.6）：** TGT、ATVI 的 small-tick 月份也是 busiest 月份；clean excitation 是否隨 regime 變，beyond activity，這些數據不能回答，作者不做此 claim。
- **Clean magnitudes 條件在線性對稱模型（§6.5）：** n(10s) 透過 bursts、n(1s) 透過 reversals，都受模型形式限制。

### 6.2 推斷的隱含假設（本頁推斷）

- **Scale separation（§4.3 明說）：** 外生強度變化慢於 baseline bin、內生響應快於 tens of seconds，兩者才能用時間尺度分開。n(10s) 在 5-minute bins 下 fall by 3–22%，就是這個假設變弱的地方。
- **Brief states 是 artifact 而非真實價格水平（作者用多項證據支持，但仍是 interpretation）：** 若某些 lock 其實是流動性提供者的真實報價意圖，E1m 會把它們當噪聲移除。
- **一個月內 kernel 共同（§3.3.3）：** 用 day-block bootstrap 檢驗，但 block resampling 仍假設 blocks 內可交換。
- **實時使用有 look-ahead（本頁推斷）：** E1m 需要未來 1ms 資訊；若把 E1m 事件當 online 特徵，會引入前視偏差。論文未討論交易用途。
- **選股 survivorship（本頁推斷，論文未討論）：** 8 檔股票從 2017 年交易量與價格區間選出，並非隨機樣本；結論對其他股票、其他年份的外推需保留。
- **成本／容量未涉及（本頁推斷）：** 本文沒有交易策略，因此沒有 breakeven cost 或容量分析；不應把 kernel mass 當可交易 alpha。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Lee (2024b) | raw NBBO multi-kernel，亞毫秒成分約 40% 事件 | 論文引用 | 本文重新解釋為 feed artifact |
| Lee and Seo (2023) | spread-dependent Hawkes，把 locked/crossed 當網路延遲移除（樣本中 1–2% 事件） | 論文引用 | large-tick stocks 中 locks 約一半，效果未被量測 |
| Lee (2024a) | 0.1s grid filter 移除超高頻活動 | 論文引用 | grid 連 10–100ms clustering 一起移除；n(1s) 低 37–60% |
| Filimonov and Sornette (2015) | 校準本身造成 apparent criticality | 論文引用 | 本文補充：total branching ratio 隨 baseline resolution 移動，n(1s) 不移動 |
| Barndorff-Nielsen et al. (2009) | realized-volatility cleaning rules | 論文引用 | 標準規則保留 locks 與 sub-ms states，artifact 存活 |
| Lee and Lee (2026) | single-exchange spread-gated Hawkes | 論文引用 | 互補：single exchange 無 locks；本文處理 consolidated feed 的 lock 機制 |

🎤 **Interview Tip**

- **正確答：** raw Hawkes 在 large-tick stocks 的亞毫秒核，權重幾乎等於「1ms 內有後續 NBBO 變化的比例」；這是 consolidated feed 把一個真實價格移動拆成兩個記錄事件，不是交易者反應。Branching ratio 混入 artifact 與 dilution，跨論文不可比；要報 fixed-horizon kernel mass 並聲明 event threshold。
- **錯答：** 把 0.6–0.7 的分支比直接讀成「市場接近 criticality」；或以為 clean 後 branching ratio 會大幅下降（total 只低 33%，horizon mass 才顯示 3 倍差異）。

### 7.1 可證偽預測帶日期

- **2026 年底前：** 若用 exchange-time NBBO 重建，E1m 的 n(1s) 與 SIP-time 差異應小（除 BAC 這類 SIP lag 接近 threshold 的股票）；原文已給 97–99.7% 事件一對一匹配、exchange-time 事件領先中位數 0.2–1.0ms。
- **2024 tick amendments 全面實施後：** large-tick 定義改變，lock share 應下降，raw/clean n(1s) ratio 應從 large-tick 的 3.8 往 small-tick 的 1.5 方向移動；clean kernel 的慢成分（~10ms–30s）應保持。
- **direct feed 取代 SIP 後：** brief states 更短，threshold 可降到 <1ms；若 clean kernel 與本文相同，則 n(1s) 應落在本文核心股票 0.10–0.26 的範圍附近（此為方向性預測，非精确数值）。

## §8 · For the Reader

- **因子研究員：** 不要用 raw NBBO event count 或 raw branching ratio 做跨股票、跨月比較；large-tick 與 small-tick 的 raw/clean ratio 從 3.8 到 1.5，raw 會壓縮差異。用 threshold-defined events，報 n(1s) 與 n(10s)，並附 day-block SE。
- **高頻執行：** lock 與 depletion 是真實交易活動，不是噪聲；但把它們當 Hawkes 事件會污染 excitation 估計。實時系統不能用 retrospective E1m；若要做 online 強度預測，需改用 causal 的 state definition 或接受 raw 的偏差。
- **組合配置：** 這不是 alpha 論文，沒有收益、Sharpe、MDD。它的價值是 measurement error：若你的 microstructure 特徵來自 TAQ NBBO，event definition 會改變你對「endogeneity」的讀數。
- **LLM-agent：** 若 agent 用 TAQ event count 做檢索或特徵，必須知道 raw count 包含 lock/depletion 配對；否則 agent 會把 feed 偽影當市場訊號。
- **RL 策略：** E1m 有 1ms look-ahead；不能直接當 online reward 或 state。若用於離線訓練，需把 event timestamp 對齊到「新 level 首次被報價」的時刻，並在推論時延遲 1ms。
- **研究學生：** 這是 measurement/construct validity 的乾淨案例：用 simulation 把 observation mechanism 加回，隔離「估計器偏差」與「事件定義偏差」。注意作者沒有說 raw Hawkes 錯，只說它測的是 feed，不是價格。

## References

- 原論文：Hyoeun Lee, “Event definition and the branching ratio: Hawkes estimates of price-change clustering in the consolidated quote feed,” arXiv:2610.36317v1 [q-fin.ST], 2026. https://arxiv.org/abs/2610.36317v1
- Lineage：Hawkes (1971); Hawkes and Oakes (1974); Ozaki (1979); Brémaud and Massoulié (1996); Bacry et al. (2013, 2015, 2016); Filimonov and Sornette (2012, 2015); Hardiman et al. (2013); Lee and Seo (2017, 2023); Lee (2024a, 2024b); Lee and Lee (2026); Holden et al. (2023); Shkilko et al. (2008); Barndorff-Nielsen et al. (2009).
- 代碼與 derived data：https://github.com/hyoeun-lee/hawkes-nbbo-events；Zenodo https://doi.org/10.5281/zenodo.23675006