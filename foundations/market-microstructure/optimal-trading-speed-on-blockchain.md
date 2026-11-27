<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=因果结构 alpha=组合执行优化 autonomy=人机协同可解释 -->

# Optimal Trading Speed on 解構（Optimal Trading Speed on）

> **來源發布**：2026-11-10 · （無 venue） · arXiv [2611.13616](https://arxiv.org/abs/2611.13616)
> **arXiv 原文**：[Optimal Trading Speed on Blockchain](https://arxiv.org/abs/2611.13616v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：五軸落在微觀盤口 × 高頻日內 × 因果結構 × 組合執行優化 × 人機協同可解釋。它解的是傳統「交易速度」文獻的識別死結：過去只能比較不同交易所的延遲，或把「速度」當個體投資選擇；本文把區塊間隔（block interval）當成鏈的設計變數，用同一條鏈的 Flashblocks 十倍提速做準實驗，證明租金分配、主動 LP 行為與 AMM 深度都由執行時鐘內生決定。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `微观盘口` | `高频日内` | `因果结构` | `组合执行优化` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（含 Internet Appendix 可讀文本）。細節待升 v1。
**TL;DR:** ①把區塊率 B 當唯一變動的市場設計參數，建構「兩套利者優先費競拍 + 單一主動 LP 跨塊撤補」的連續時間均衡；②核心 trick 是 *clash vs race*——B 變快使套利者從同塊競價轉為跨塊搶先，sequencer 份額被壓成 s(B)=λ/(B+λ)，LP 側對偶地出現「撤資只要早一塊就值錢」的內生門檻；③對「因果結構」軸★：固定所有人的訊息延遲 λ，只動鏈的時鐘就能改變租金分配與流動性深度；④Base Flashblocks（2 秒 → 200 毫秒）準實驗顯示：頂十分位套利交易的 mean gas bid 由 89 gwei 降至 17 gwei、區塊鏈費 DiD 下降 3.26 bps、淨 markout DiD 上升 2.73 bps，同時池深約下降 19%、~1 ETH 有效價差 DiD 上升 0.07 bps。

**X-Ray.** 這篇的理論零件並不新——arbitrage 競價是 Budish-Cramton-Shim 的延遲競賽，LP 集體行動困境是 Capponi-Jia 的 tragedy of the commons，interior optimum 是 Du-Zhu 的拍賣頻率。真正稀缺的是**識別**：同一條鏈、同一批參與者、同樣的訊息延遲，只把 block interval 從 2 秒壓到 200 毫秒，且區塊內 ordering rule（PGA）不變。這讓「速度」第一次成為接近乾淨的 treatment，而不是跨交易所的混雜比較。它解掉的舊坑是：LP 側文獻長期假設部位在交易中固定（Hasbrouck 2026；Milionis 2023），因為慢塊下單一 LP 只保住自己那份損失、競價必輸套利者（10% 池份額只值 1/10 的優先費）；快塊繞開了這場競價，改用「時序」而非「出價」保護資本。它打不開的 envelope 也很清楚：只測一池（WETH–USDC 5 bp）、主動 LP 只有一個 Base 錢包、未定位 depth peak 的 B_d、明言不做 welfare 結論。對量化讀者的意義：block speed 應該被當成策略層級的 regime 變數——同一套做市／套利邏輯在 2 秒與 200 毫秒下的最優行為不同，而且這個差異可交易（主動 LP per-dollar payoff DiD 提升 0.73 bps、獲利日占比由 51% 升至 89%）。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作 | 本文改動 | 為什麼會改變結論 |
|---|---|---|---|
| 速度變數的層級 | 個體訊息速度／交易所延遲是參與者的投資選擇（Biais et al. 2011；Budish et al. 2015） | 固定所有人的訊息延遲 λ，只變協議設定的區塊率 B（"Block speed is the only thing we vary"） | 租金分配不再是「誰花錢買速度」的結果，而是「鏈的時鐘」直接決定，可作為外生 treatment |
| 競爭形態 | 同塊優先費競拍被視為給定製度 | 推導出 clash 機率 λ/(B+λ)、race 機率 B/(B+λ)，競爭由拍賣漂移為 FCFS | sequencer 份額 s(B)=λ/(B+λ) 隨 B 單調下降；出價在快塊下買不到已成交的交易 |
| LP 的行為集合 | 交易中部位固定，或 JIT 觀測 pending swap 插隊 | 主動 LP 看到 gap 就撤資、靠「跨塊」而非「同塊插隊」避險 | 不需贏得優先費競拍即可保護資本 → 快塊使 active management 內生成為最優，並可分離 fee 收入與 toxic flow |

### 1.2 ⚡ Eureka

**同一群人、同樣的訊息延遲，只把區塊變快，就把套利競爭從「同塊比出價」推向「跨塊比到達」。** 直覺：慢塊時兩個套利交易都擠進同一塊，價高者成交，出價被推到幾乎等於獎金；快塊時你的交易在你的對手到達前就被下一塊執行掉，對手的出價再高也買不到一筆已經完成的交易。於是出價失去作用，租金留在套利者手上。LP 側是同一枚硬幣的反面：撤資只有落在「比套利更早的一塊」才值錢，而慢塊迫使她跟套利者競價——她只保住自己那份損失，永遠出不起價。

### 1.3 信息流

```
news (rate J) ──► |p* − pAMM| ∈ {0, M, H},  0 < M < f < H
                     │
        ┌────────────┴──────────────┐
        │                           │
  arbitrageur i (i=1,2)        active LP (capital q)
  latency ~ Exp(λ)             latency ~ Exp(λ)（同速，無優勢）
  bid b ∈ [0,1] of prize       withdraw / redeposit
        │                           │
        └────────► sequencer ◄──────┘
              blocks ~ Poisson(B)，塊內 PGA：出價高者先
              同塊衝突時 arbitrage 先於 withdrawal 執行
                     │
        ┌────────────┴──────────────┐
  prize ℓQ 分配：                 fee flow r(B)=r̄·B/(B+δ)
  sequencer  s(B)=λ/(B+λ)        按存入資本 pro rata 分享
  套利者      1−s(B)             passive P 每筆套利交易都承擔 ℓ
```

## §2 · 數學層

📌 **Napkin Formula**

```
s(B) = λ/(B+λ)        ⇒ odds s/(1−s) = λ/B          （sequencer 份額）
P(clash) = λ/(B+λ),  P(race) = B/(B+λ)              （競爭形態）
e(B) = B/(1+TB),     T = 4/J + 1/(2λ)               （套利成交率，e′>0）
D = P + qπ,          B_d = [Tδk + √(T²δ²k² + A_d·δk)] / A_d,  A_d = ℓ − δT(kT+ℓ)
```

**直覺**：s 的 odds 形式是最可攜的結果——塊速 m 倍提高，sequencer 份額 odds 會由 λ/B 等比縮小；論文另給映射 s₊ = s₋/[m−(m−1)s₋]。e(B) 上升代表套利更常執行，但這同時提高被動資本每單位時間的損失 ℓe；深度在最優點上是「救回逾期流動性交易」與「更頻繁的逆向選擇」兩力的權衡。

**複雜度**：無訓練。均衡地圖有三個邊界 0 < B₀ < B₁ < B₂：B₀ 是二次式正根（封閉解 IA.51），B₂ 是三次式正根（IA.52），B₁ 落在該六次分子在 (B₀,B₂) 區間內的唯一根上；作者明言不宣稱該六次分子只有單一正根，亦不宣稱可用算術與根式寫出（IA.2.4）。因此實務上每個 B 都要數值求根，成本 O(1)。

**Loss / 訓練細節**：不適用。取而代之的是四條參數限制 Assumption 3.1（2λ>J、k 的上界、0<δ<k/(20ℓ)、fee 相對 q 的下界）；作者自述常數 "twenty" 是「sufficient for our proofs, not an estimated market threshold」——不要把它當校準值。

## §2.5 · 帶數字走一遍（Worked Example）

> **以下數字全部取自原文 Internet Appendix IA.3.6 自帶的示意校準（J=λ=1、M=1、H=3），是作者用來說明「同一 gap 過程、兩種量測會反向」的玩具例子，不是實證結果。** 論文真實數字一律留在 §5。

1. **設定**：gap 只在 {0, M, H} 三態間跳，M=1 落在 no-arbitrage band 內，H=3 才可套利。區塊率 B 是唯一變動量。
2. **成交率**：e = B/(1+TB)，且 e′(B)=1/(1+TB)² > 0 —— 塊變快，套利交易更常執行。
3. **連續時間量測**：E_cont = H − e(4H−2M)/J，隨 e 上升而下降。
4. **區塊末端量測**：E_end = e(2M/J + H/(2λ))，隨 e 上升而上升（因為收掉 gap 的那一塊總是在 swap 後留下零 gap）。
5. **照算對照**：B 由 1 升到 2，E_cont 由 13/11 降到 1；同一組參數下 E_end 由 7/11 升到 7/10。
6. **結論**：兩個統計量對「變快」給出相反方向。若你在區塊結束後採樣價差，會得到「快 → 價差變差」的偽結論；本文 §5.6 最終用的是時間加權版本（DiD −0.11 bps）。

## §3 · 數據層

- **市場 / 標的**：Uniswap v3 WETH–USDC 5 bp 池，Base（treatment）與 Arbitrum（control）各一池；只測單一 fee tier。
- **鏈上事件**：Allium 提供 event-level 資料（Swap / Mint / Burn 加上 NFPM 的 IncreaseLiquidity / DecreaseLiquidity），以 transaction hash + event type + liquidity amount 為複合鍵回接 tokenId；未匹配到 NFPM 的直接互動大戶改用 (address, tick_lower, tick_upper) 重建部位。
- **參考價**：Tardis 的 Binance ETHUSDC 現貨 L2 五檔快照，約每 100 毫秒取樣、微秒級 matching-engine 時間戳。
- **時段**：樣本 2025-05-16 至 2025-09-15；pre = 05-16 至 07-06，post = 07-17 至 09-15，過渡期 07-07 至 07-16（Flashblocks 內部上線但未公開）剔除。
- **規模**：Base 日成交由 69.3 百萬美元升至 85.6、日 swap 由 30,679 升至 92,149、獨立交易地址由 4,344 升至 13,913；Arbitrum 同期為 168.9 → 257.9、33,387 → 43,310、1,318 → 1,755。Table A1 各期 swap 數：1,583,918 / 5,597,991 / 1,733,370 / 2,639,332。
- **樣本外與容量假設**：無真正的 holdout；識別靠 DiD + 附錄 B.2 的 ±4 週窗、placebo 日（2025-06-11）與剔除 outage 日。套利交易以「執行時 signed CEX–DEX gap > τ」分類，baseline τ=10 bp，並以 6、8 bp 做穩健性。LP 結果只按日期聚類（entity clusters 太少）；swap 級結果在固定大小分箱內估計。**容量未測試**：模型的主動 LP 是單一資本 q，實證 cohort 只有 1 個 Base 錢包與 4 個 Arbitrum 錢包。

## §4 · 代碼層

| 項目 | 內容 |
|---|---|
| Repo | 所讀文本未見（TBD） |
| Checkpoint | 不適用（解析均衡 + 計量，無 ML 模型） |
| License | TBD（所讀文本未見） |
| 複現難度 | 高：需 Allium 事件級鏈上資料 + Tardis L2 快照（商用授權）；LP 部位需自 2025-05-01 起回推重建；每張表的 panel 結構、固定效應、聚類層級都写在 caption 內，未統一 |
| 數據可得性 | 非公開；鏈上部分理論上可由節點重抓，但 NFPM 與 pool 事件的複合鍵對接、fee-checkpoint burn 剔除等清洗規則需自行復刻 |

## §5 · 評測 / Benchmark

> **基線說明**：這裡沒有「前 SOTA」。前SOTA 欄放的是**同一場地升級前的自身基線（Base 2 秒區塊）**，本方法欄是**升級後（200 毫秒）**；Δ 為鏈內原值差（本方法 − 前SOTA）。論文真正的識別量是 DiD 係數（Post×Base），另列於解讀，不與鏈內差混用。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Base WETH–USDC 5bp（2s → 200ms） | Net markout（bps，10s horizon） | 2.31 | 4.52 | +2.21 |
| Base WETH–USDC 5bp（2s → 200ms） | Blockchain fee（bps of trade size） | 3.92 | 0.66 | −3.26 |
| Base WETH–USDC 5bp（2s → 200ms） | Raw markout（bps，10s horizon） | 6.23 | 5.18 | −1.05 |
| Base 主動 LP | Net payoff（Volume-weighted bps，10s horizon） | 0.26 | 1.28 | +1.02 |
| Base 池，~1 ETH 交易 | Effective spread（bps，vs pre-swap marginal pool price） | 0.17 | 0.25 | +0.08 |

**解讀：哪部分是 capability，哪部分是租金移轉。** 三條中最乾淨的是費用端：區塊鏈費 DiD 下降 3.26 bps，而 raw markout DiD 只變動 0.53 bps（統計上不顯著），因此淨 markout 的上升幾乎全部來自「少付給 sequencer」，不是套利者變強。這與機制預測的簽名吻合：gas bid 的下降**只集中在有優先價值的大單**——P50–P60 為 0.37、P60–P70 為 0.04（皆不顯著），P70–P80 為 −0.75、P80–P90 為 −0.86、P90–P100 為 −1.31（top decile 的 73% 降幅），且尾端再切更深（P90–P95 −1.03、P95–P100 −1.47）。若只是網路壅塞下降，小單也該動；它沒動，這是機制證據而非雜訊。

**其餘市場品質變數原文只給 DiD 係數，未提供 Base 升級前後原值，不放入上表。** 價格效率改善：TWA deviation 為 −0.11 bps。流動性反向：log depth 為 −0.21（約 19% 降幅）；top-1 time-weighted liquidity share 為 +4.20 percentage points；~1 ETH effective spread 的 DiD 為 0.07；log ETH volume 為 −0.20（約 18%）。價差擴大的幅度隨交易規模上升（附錄 B.3：D2 的 0.002 到 D9 的 0.155），最頂十分位為 −0.028 且不顯著。

**DiD 與升級前後原值的落差要說清楚。** Base 的 net markout 由 2.31 升至 4.52，但對照鏈同期也由 8.86 降至 8.34，故 DiD 係數是 2.73。同理，Base 有效價差由 0.17 升至 0.25，DiD 係數是 0.07，論文另以「a 47% increase」描述相對幅度——絕對值很小，但相對衝擊大。

**哪些數字要打折。** LP 側的 cream-skimming（toxic-exposure share 由 62.9% 降至 47.5%，池整體只由 83.0% 降至 76.5%，DiD 為 −8.93 pp）是本文最漂亮的直接測度，但 cohort 是 1 vs 4 個錢包、只按日期聚類，且 placebo 日仍有 −4.81 pp（10% 顯著）；±4 週窗下 LP 的 rebalancing time 與 net payoff 都失去顯著性。深度端方向一致，但**成本未完整計入**：論文明說 blockchain fee 含轉嫁的 L1 data posting，因此 56% → 12% 的 sequencer 份額是上界。Arbitrum 同期沒有可比的 sequencer 份額（Timeboost 收益不落在 per-tx gas 內）。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **§6 Conclusion**：證據「does not locate the liquidity-maximizing block rate」，且「we draw no welfare conclusion」——只給方向，不給最優 B。
- **§5.5.1 / Table 5 caption**：主動 LP cohort 為 1 個 Base 地址對 4 個 Arbitrum 地址；美元絕對水平跨鏈不可比，只有 profitable-days share（51% → 89% vs 12% → 3%）是 scale-free 可比的。
- **§5.2**：LP 結果只按日期聚類，因 entity clusters 太少無法做雙向推論。
- **§5.4.3 footnote 30**：blockchain fee 含 L1 data posting 轉嫁，份額為 sequencer 自身租金的上界；Arbitrum 因 Timeboost 不可觀測，無法計算可比份額。
- **§5.4.1 / Appendix B.1**：套利分類是 proxy；5 bp 門檻下兩鏈比例差異大（52.8% vs 27.1%），previous-block 參考價未必等於當下池價。
- **Appendix A**：平行趨勢圖「describe the observed time patterns; they do not establish the counterfactual parallel-trends assumption」。
- **Appendix B.2**：placebo 對 cream-skimming 為 −4.81 pp（10% 顯著）；窄窗下多個係數失去顯著性。
- **§3.6 Assumption 3.1(iii)**：常數 20 是證明充分條件，非市場門檻。
- **IA.2.4**：中間邊界 B₁ 無封閉形式，且定理不主張其六次分子只有單一正根。

### 6.2 推斷的隱含假設（本頁推斷，非作者自述）

- **外推**：只測一池、一 fee tier、一資產對。長尾資產或高 fee tier 的毒性流比例不同，cream-skimming 的量級不可直接搬用（未測）。
- **ordering rule 交互**：Base 是 PGA；Arbitrum 對照組是 FCFS + Timeboost，且樣本後於 2026 年 9 月改採 PGA。Flashblocks 本身只改頻率，within-Base 的 before/after 較乾淨；但跨鏈 DiD 把 ordering rule 差異一併放進控制組。若 ordering rule 與 block speed 有交互作用，β₃ 不是純 block-speed 效應。
- **gas 前提**：主動 LP 需頻繁進出，原文 footnote 4 指 L2 的低 gas 是此策略可行的前提；在高 gas 環境（L1 或壅塞期）結論可能反轉。
- **資本規模**：模型要求 q 相對 fee 支撐的資本極小（η_q < 1/9），大資本進入會改變 fee 分配與深度結論。
- **選樣**：cohort 以升級前行為（≥10 活躍日、25 分位 rebalancing < 60 秒等）預先定義，這避免了用結果選樣；但被排除的兩個錢包在 5 月底即停止交易，屬樣本結構，仍限制了對「主動 LP 普遍性」的推論。
- **regime 依賴**：所有結論建立在「區塊率是 Poisson、區塊內規則不變」的架構上。本文能清楚分離的是 Base 內部 block-speed 改變；跨鏈與跨規則外推則需重估。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Budish, Cramton & Shim (2015) | 連續 vs 批次拍賣、速度競賽的社會浪費 | 否（理論） | 本文把「批次頻率」變成鏈上可調且可觀測的 B |
| Du & Zhu (2017) | 拍賣頻率的 interior optimum | 否 | 本文的 interior peak 落在 pool depth，且來源是 LP 側而非需求曲線 |
| Menkveld & Zoican (2017) | 交易所延遲對流動性的雙向效應 | 否 | 本文提供同鏈外生變動的實證對應 |
| Capponi & Jia (2025) | AMM LP 的 tragedy of the commons、部位固定 | 否 | 本文打開「LP 可撤資」的行為集合，給出 active/passive 的內生分界 |
| Wan & Adams (2022) / Capponi et al. (2024) | JIT 流動性（觀測 pending swap 插隊） | 否 | 本文的主動 LP 不靠 ordering、靠跨塊時序，因此在 FCFS 鏈上也可運作 |
| Wu & Öz (2026) | 同一個 Flashblocks 升級、cyclic arbitrage 搜尋 | 否 | 同期同資料，但看搜尋策略組成，不看租金分配與 LP |

🎤 **Interview Tip**

**問：區塊變快，為什麼有效價差反而變寬？**
**正確答**：快塊把套利競爭從同塊 priority auction 推向跨塊 race，sequencer 份額 s=λ/(B+λ) 下降、套利者保留更多，且套利交易更常執行（e 隨 B 上升）；同時主動 LP 可在更早的塊撤資，降低她對毒性流的曝險——但說降到零是錯的：post 她的 toxic-exposure share 仍為 47.5%。被動資本因不撤資、持續曝險於每一筆執行套利的損失，且該損失頻率隨 B 上升而增加，因此退出；池深下降、價差走寬。關鍵是「租金移轉 + 逆向選擇重新分配」，代價主要落在被動 LP 與流動性交易者；而且價格效率（TWA 偏離 −0.11 bps）是同向改善的，兩者並不矛盾。
**錯答**：「變快讓 LP 更快更新報價，價差應該縮窄」——這是把 AMM 當 CLOB。AMM 的報價靠交易推動，鏈的時鐘只決定誰先成交、誰先撤資。

### 7.1 可證偽預測（本頁預測，尚待驗證）

- 若 OP Mainnet（250 毫秒，2025 年 9 月起）在可比池上出現同向的 top-decile gas bid 下降與主動 LP rebalancing 加速 → 支持機制；若無 → 需追加 ordering rule 條件。
- 若 Arbitrum 於 2026 年 9 月改採 PGA 後，其 gas bid 的規模梯度向 Base 收斂 → 支持「PGA 是 gas bid 存在的前提」；這也是把 block speed 與 ordering rule 分離的天然檢驗。
- 若在 Base 上換成長尾資產或不同 fee tier 的池，cream-skimming 的 −8.93 pp 量級不成立 → 表示該測度依賴池本身的毒性流比例，而非普適常數。

## §8 · For the Reader

- **因子／統計套利研究員**：把 arbitrage markout 拆成 raw（機會）與 blockchain fee（租金）兩段再回測。本文 fee 端 DiD 降 3.26 bps、raw 端只動 0.53 bps 且不顯著——若你的回測把 gas 當常數，會在快鏈環境系統性誤判套利淨收益。
- **高頻執行／做市**：主動 LP 的 edge 不是搶 priority（gas price 的 DiD 是 0.03，不顯著），而是時序（rebalancing time −1.07 log，約 66% 加速）。把撤資成功率 B/[3(B+2λ)] 當成可在鏈上調校的參數，而不是固定的基礎設施常數。
- **組合配置／選池**：把鏈的 block interval 當成池層面的因子，與 fee tier、波動率一起回歸。interior depth peak 意味著同一資產在更快的鏈上不一定有更深的池——「快鏈＝好流動性」是錯的直覺。
- **LLM-agent／機制設計**：s/(1−s)=λ/B 這個 odds 形式極簡且可嵌入激勵分析；若你在設計 agent 的交易環境，block rate 是比 gas 參數更乾淨的槓桿。
- **RL 策略**：把 latency 與 block rate 分成兩個獨立參數，本文證明「固定 latency、只動 block rate」就足以改變均衡——這是環境 ablation 的現成軸線。
- **研究學生**：注意識別策略是 DiD 而非 RCT。Appendix A 自承平行趨勢未被證明，B.2 的 placebo（cream-skimming 仍有 −4.81 pp）是必讀，因為它劃定了這篇文章結論的信心邊界。

## References

- 原論文：*Optimal Trading Speed on Blockchain*，arXiv:2611.13616v1（2026-11-10）。https://arxiv.org/abs/2611.13616v1
- 速度與市場設計：Budish, Cramton & Shim (2015), QJE；Du & Zhu (2017), ReStud；Menkveld & Zoican (2017), RFS；Biais, Foucault & Moinas (2011)。
- AMM 流動性與逆向選擇：Capponi & Jia (2025), RFS；Lehar & Parlour (2023), JF；Milionis et al. (2023)；Hasbrouck, Rivera & Saleh (2026), Management Science；Park (2023)。
- 主動／JIT 流動性：Fan et al. (2023)；Wan & Adams (2022)；Capponi, Jia & Zhu (2024)；Malinova & Park (2023)。
- MEV 與 L2 結構：Daian et al. (2020)；Schwarz-Schilling et al. (2023)；Bahrani, Garimidi & Roughgarden (2024)；Adams (2024)；Caparros et al. (2023)；Gogol et al. (2024)；Wu & Öz (2026)。
- 制度事實來源：Base Flashblocks 工程部落格（2025）；Optimism（2025）；Offchain Labs（2026a, 2026b）；SEC Release No. 34-106402（2026-09-17）。