<!-- ontology-5axis data=文本另类 horizon=日频波段 paradigm=生成式大模型 alpha=端到端表征 autonomy=人机协同可解释 -->

# LLM零样本理由抽取+样本外R²评估 解構（LLM零样本理由抽取+样本外R²评估）

> **來源發布**：2026-11-17 · （無 venue） · arXiv [2611.22277](https://arxiv.org/abs/2611.22277)
> **arXiv 原文**：[Explaining Realized Returns: Roll's $R^2$ 40 Years Later](https://arxiv.org/abs/2611.22277v1)　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_
> **核心定位**：把 Roll (1988) 的 $R^2$ 之問重寫成一個**可測度的 concurrent 回歸問題**——用 LLM 從 WSJ 每日行情稿抽出「陳述的理由」、剝掉價格語言，再以 5-fold 樣本外 $R^2$ 對比宏觀／現金流／預期／資金流基線。五軸落點：文本另類 × 日頻 × 生成式大模型 × 端到端表征 × 人機協同可解釋。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `文本另类` | `日频波段` | `生成式大模型` | `端到端表征` | `人机协同可解释` |

**Status:** v0.5 — 基於 arXiv 原文（有原文則以原文為準）。細節待升 v1。
**TL;DR:** 把「事後能否解釋已實現報酬」變成回歸題：每日一篇 WSJ「Today's Markets」，LLM 先抽 reason–price 配對、再由第二道 prompt 篩掉任何含價格語言的片段（leakage 控制），理由文本用 embedding regression 或零樣本推論映射到 S&P 500 報酬。核心 trick 不是模型而是**遮罩**——沒有這道遮蔽，$R^2$ 就是同義反覆。以 5-fold 樣本外 $R^2$ 計，embedding regression 日頻 **29.82%**；零樣本（gpt-4o-mini-2024-07-18）日頻 **25.58%**，月度原始加總為 **-115.03%**，經 HAR 過濾後回升至 **49.84%**；embedding 日值加總月度由 **33.06%** 升至 **44.42%**。對 alpha 軸的意義：這是一條**解釋**管線，不是預測管線，任何把它當信號的讀法都越過了作者自設的邊界。

**X-Ray.** 這篇的價值不在「LLM 很強」，而在**它把「解釋」與「預測」在協方差結構上切開了**：作者明說理由是事後觀察、可用 hindsight 估計映射，因此用隨機 5-fold 而非 expanding window——這個設計選擇同時是它的方法論貢獻與它最大的可攻擊面（隨機摺不是可交易的外樣本）。它解掉的舊坑是 Roll 以降「新聞事件只當 0/1 虛擬變量」的信息損失：把事件發生與否升級為**理由內容**，並用 3,072 維 embedding 保留事件、方向與上下文；這也解釋了為何 ridge 全面勝過 lasso（信息分散在大量維度而非稀疏子集）。它打不開的 envelope 很清楚：單一媒體、單一欄位、單一資產（總體指數）、單一語言；zero-shot 那條臂的 cutoff（October 2023）與樣本（2002–2024）重疊，「零樣本」指的是無任務訓練樣本，不是無預訓練接觸。對量化讀者的實際用途：不要在它身上找 Sharpe，要拿它的**leakage 遮蔽協議 + 分群 active/overall $R^2$ 分解**去審自家新聞因子的同義反覆問題。

## §1 · 架構 / Core Mechanism

### 1.1 三大改動 vs 前作

| 維度 | 前作（Roll 1988 / Cutler-Poterba-Summers 1989 / Baker et al. 2021） | 本篇 | 為什麼是能力增量 |
|---|---|---|---|
| 解釋變量形態 | 新聞**發生與否**或人工分類標籤 | 文章**自陳理由的原文片段**（直接引語） | 從 incidence 升級到 content，保留方向與上下文 |
| 映射方式 | 分組比較／人工歸類 | ridge/lasso on 3,072 維 LLM embedding；或零樣本 prompt | 高維語義可直接進回歸 |
| 評估準則 | in-sample 調整 $R^2$、事件窗波動 | **5-fold 樣本外 $R^2$** + 兩條 criterion（非價格來源、映射穩定） | 把「事後解釋」也納入過擬合紀律 |
| 頻率 | 月度或事件日 | 日頻（非新聞基線僅月度可得） | 日頻本來無對照基線 |

註：原文對 Roll (1988) 明言「This is a different empirical exercise from ours」，不可把兩者的 $R^2$ 直接比大小。

### 1.2 ⚡ Eureka

**一句話**：解釋力之所以長期低落，不是因為沒有信息，而是因為沒有把「理由」從「價格敘述」裡物理切開——一旦用第二道 LLM prompt 把含價格語言的片段整段刪除，剩下的文本就是一個**與報酬無機械關聯、卻仍與報酬同步**的回歸量。

直覺：記者在同一段裡寫「股市因 X 而上漲」，X 與「上漲」在語言上黏在一起；若整段餵進模型，$R^2$ 反映的是分詞運氣。遮罩後 $R^2$ 才是對「世界事件→價格」這個映射的檢定。代價是 recall 下降（有些日子理由是空集合，作者把它設為 implied return = 0），這是保守偏誤而非樂觀偏誤。

### 1.3 信息流

```
WSJ "Today's Markets"（每交易日 1 篇，2002–2024）
        │
        ├─[Prompt 1] 抽 (reason, price_movement) 配對（僅直接引語）
        │
        ├─[Prompt 2] 洩漏篩檢：flag 任何指數/個股/債/匯/商品的價格語彙
        │              → 命中者整段刪除
        ▼
   reason snippets ──(同日拼接)──► reason composite
        │                              │
        │                     ┌────────┴────────┐
        │              text-embedding-3-large   [Prompt 3] zero-shot
        │                   3,072-d                    │
        │                     │                        │
        │              ridge/lasso（5-fold CV）   直接輸出百分比小數
        │                     │                        │
        └──────────────► 樣本外 R² ◄───────── HAR 過濾（去均值+去持續性）
                                    │
                          分群 active/overall R² 分解（不重估模型）
```

## §2 · 數學層

📌 **Napkin Formula**

$$R^2_{\text{OOS}}=1-\frac{\sum_{t\in\text{test}}(r_t-\hat r_t)^2}{\sum_{t\in\text{test}}(r_t-\bar r)^2},\qquad \min_\beta \sum_{t\in\text{train}}(r_t-\beta^\top x_t)^2+\lambda\|\beta\|_2^2,\; x_t\in\mathbb{R}^{3072}$$

$$\text{HAR: } \hat r_t=\beta_0+\beta_1\hat r_{t-1}+\beta_5\text{ma}_5\hat r_{t-1}+\beta_{22}\text{ma}_{22}\hat r_{t-1}+\beta_{63}\text{ma}_{63}\hat r_{t-1}+\hat\epsilon_t$$

**直覺**：第一式是判準——分母用**評估樣本均值**去平減，所以月度比不減均值更嚴；第二式是映射估計，$\lambda$ 與一切超參只在訓練摺上調；第三式是「除鏽」，把 implied return 序列裡可預測的持續性抽掉，只留衝擊成分再與報酬對齊。**關鍵限制**：HAR 只用 implied return 自身、不用 $r_t$，因此不破壞 criterion 1；但原文寫的是 full-sample HAR 迴歸（§B.1），這意味著過濾器本身看過整段樣本。

**Loss / 訓練細節**：無梯度訓練、無微調；embedding 模型（text-embedding-3-large、BERT 768 維）與生成模型（gpt-4o-mini-2024-07-18，temperature 0，每交易日保留一條回應、不跨生成平均）皆為凍結推理。零樣本臂無交叉驗證（無任務訓練樣本，故 CV 不適用亦不需要）。計算成本、$\lambda$ 網格、樣本天數：未披露。

## §2.5 · 帶數字走一遍（Worked Example）

**以下全部為假設/示意數字，非論文實證結果。**

1. **假設**某日文章抽到 3 組 reason–price 配對；其中 1 條理由句含「stocks rallied」，第二道 prompt 判為洩漏 → 刪除，剩 2 條拼接為 composite。
2. **假設**該 composite 經 text-embedding-3-large 得 3,072 維向量 $x_t$（維度是架構常數，非實證結果）。
3. **假設**訓練摺估出的 $\hat\beta$ 給出該日 implied return $\hat r_t=+0.60\%$（示意）；當日實際 $r_t=+0.40\%$（示意）。
4. **假設**該日殘差平方為 $(0.40-0.60)^2=0.04$（%²），而整個評估摺的 $\sum(r_t-\bar r)^2=4.00$（%²）→ 單日對 $R^2$ 的扣除為 $0.04/4.00=1\%$。這一步只示範方向，**單日不能推廣成論文結論**。
5. **持續性偏誤示意**：假設連續三個交易日的零樣本輸出為 $+0.9\%,+0.8\%,+1.0\%$，月合計 $+2.7\%$（示意），而實際月報酬 $+0.6\%$（示意）。日頻看似無傷，加總後 overshoot——這正是原文解釋月度原始加總 $R^2$ 為 **-115.03%** 的機制。
6. **HAR 過濾示意**：若迴歸把其中 $+0.85\%$ 判為可預測成分，殘差 $\hat\epsilon_t=+0.05\%$（示意）；用殘差序列重算並加總，才會回到正值端（原文實測為 **49.84%**）。

## §3 · 數據層

| 項目 | 內容 |
|---|---|
| 理由語料 | WSJ「Today's Markets」（亦標為 Monday's Markets…／U.S. Markets），**每交易日一篇**，2002–2024；原文稱「Nearly all trading days in the sample have a corresponding article」 |
| 時序對齊 | 以**報導所涵蓋的交易日**為索引（印刷版次日上午出版、收盤後常已上線），用 publication metadata 建索引並以價格資料核對 |
| 觀測單位 | 日頻 reason snippet → 當日 composite；月度透過**日度 embedding 相加**聚合 |
| 非新聞基線 | Macro：FRED-MD 126 條中留 99（剔除 Group 6 的 22 條與 Group 8 的 5 條）；Cash flow：4；Subjective expectations：4（2003Q1–2021Q3 重疊）；Trading flows：12（資料止於 2019Q1） |
| 頻率轉換 | 季頻以線性內插／重複轉月頻（GDP、盈餘、預期、流量皆然） |
| 樣本外協議 | **5-fold 隨機交叉驗證**，超參只在訓練摺調；作者明言不用 expanding/rolling，因為此題事後解釋、估計樣本不必早於評估樣本 |
| 泄漏控制 | 兩段式 prompt：抽配對 → 價格語言篩檢（含 pulled back / rebounded / buoyed 等隱性語彙） |
| 容量 / 生存者 | 總體指數無成分股生存者問題；**容量、交易成本、可投資性皆不在本文範圍**（非策略） |

## §4 · 代碼層

| 項目 | 狀態 |
|---|---|
| Repo | 所讀文本未見；TBD |
| Checkpoint | 無自訓權重；依賴 text-embedding-3-large、BERT（型號未披露）、gpt-4o-mini-2024-07-18 |
| License | 原文頁標 CC BY-NC-SA 4.0（指論文，非代碼） |
| 複現難度 | 中高：prompt 全文在 §A.2 可複現，但資料需 WSJ 訂閱檔案、embedding API 版本漂移、模型 cutoff 不可回滾 |
| 數據可得性 | 理由資料集未見釋出說明；FRED-MD／Shiller／Delao-Myers／Gabaix-Koijen 流量皆為公開或可重建 |

## §5 · 評測 / Benchmark

> **欄位說明**：模板欄名「前SOTA」在本表一律指**原文自己設的對照基線**；若為同方法未過濾版本（消融），會在 Metric 明示「raw → HAR」等，不把它冒充第三方 SOTA。原文未使用 SOTA 一詞，也無第三方排行榜。Metric 一律為 OOS $R^2$ (%)，精度照抄原文；基線僅以文字表述者不計算 Δ。

| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| S&P 500 月度 2002–2024 | OOS $R^2$ (%) · Trading flows（12 變量） vs 本法月度 ridge | roughly 10% | around 40% | —（基線為 roughly 表述，不計算） |
| 同上 | OOS $R^2$ (%) · Macro（FRED-MD，99 變量） | 未披露（原文僅稱多數組為負，無逐組數值） | around 40% | — |
| 同上 | OOS $R^2$ (%) · Cash flow（4 變量） | 未披露（同上） | around 40% | — |
| 同上 | OOS $R^2$ (%) · Subjective expectations（4 變量） | 未披露（同上） | around 40% | — |
| S&P 500 日頻 | OOS $R^2$ (%) · 非新聞基線 | 未披露（原文：daily data not available） | 29.82% | — |
| S&P 500 日頻 | OOS $R^2$ (%) · zero-shot：raw → HAR | 25.58% | 31.88% | +6.30pp |
| S&P 500 日頻 | OOS $R^2$ (%) · embedding regression：raw → HAR | 29.82% | 33.88% | +4.06pp |
| S&P 500 月度 | OOS $R^2$ (%) · zero-shot：raw 日值加總 → HAR 後加總 | -115.03% | 49.84% | +164.87pp |
| S&P 500 月度 | OOS $R^2$ (%) · embedding regression：日值加總 → HAR 後加總 | 33.06% | 44.42% | +11.36pp |

**三個不同的月度數字不可互代**（原文 §B.1 明確區分）：**around 40%** = 正文對 Table 3 月度 ridge 的文字表述（本頁不另補精確值）；**33.06%** = 日頻 embedding regression 的 fitted value 加總到月度（Table 4）；**44.42%** = 日頻 implied return 先 HAR 過濾再加總（Table 4）。任何人拿其中一個去和另一個比「提升」都是張冠李戴。同理，零樣本月度 raw 加總 **-115.03%** 與 embedding 月度 **33.06%** 是不同映射臂，不可互相代稱。

**哪部分是 capability**：日頻 29.82% 對比「日頻非新聞基線不存在」是乾淨的增量；月度 around 40% 對比 trading flows 的 roughly 10%，方向明確但**無法給精確 Δ**（基線是文字表述而非數值）。ridge 全面優於 lasso，說明信息分散在多維而非稀疏子集——這是對「embedding 承載了事件特異性」的間接支持。

**哪部分可能是假象**：(i) **隨機 5-fold 不是時序外樣本**，作者以此換取「事後解釋」的正當性；這使 $R^2$ 不可解讀為可實現的交易績效，且折間信息互通（同一事件在不同摺反覆出現）會溫和膨脹擬合。(ii) **zero-shot 臂的預訓練 cutoff October 2023 與樣本 2002–2024 大幅重疊**——本頁推斷：這不是 leakage 的實證認定，但它使「零樣本」只能理解為「無任務訓練樣本」，不能理解為「模型沒見過那段市場史」。(iii) **HAR 過濾用 full-sample 迴歸**（原文 §B.1 自述），雖只吃 implied return 不吃 $r_t$，仍帶有全樣本擬合成分；-115.03% → 49.84% 這個變化主要修的是**持續性偏誤**，屬方法修正而非世界知識的增加。(iv) 無交易成本可扣——因為根本沒有策略；若有人以這些 implied return 造倉，成本、容量、沖銷頻率全部 TBD。

**分群拆解**（日頻、pooled 模型不重估）：Corporate news / Macroeconomic activity / Investor sentiment 各自約 6%–9%；Monetary policy 約 4%；Covid 分群 active $R^2$ 約 15%、僅約 5% 樣本日出現、overall 約 2%。原文的 $A\approx B\times C$ 分解指出：**跨群 overall 差異主要來自出現頻率，而非條件解釋力**（多數群 active $R^2$ 落在 10%–15%，Commodity prices 與 Unassigned 低於 10%）。因此「哪類新聞更能解釋市場」這類排序，很大程度是 active-day 佔比的產物，不可讀成題材的因果效力。

**與結構化事件資料的對齊檢定**（Table 6，非 SOTA 對照）：Monetary Policy 分群的 implied return 對高頻 FOMC 衝擊迴歸，斜率 0.9393（t = 4.24，N = 193，$R^2$ = 0.0860）；Minutes releases 斜率 0.3046（t = 2.87，N = 182，$R^2$ = 0.0438）；Combined 0.6719（t = 5.16，N = 375，$R^2$ = 0.0667）。方向與顯著性支持「理由文本對齊真實政策衝擊」，但 $R^2$ 低，作者自述兩邊都是雜訊代理。另：USMPD 高頻回應對全日報酬的 $R^2$ 為 FOMC announcements 19.31%、press conferences 48.87%、statements 1.87%、Minutes 5.36%、Combined 14.40%——但該變量本身由價格構成，**違反 criterion 1**，只能當對齊證據，不能當解釋力上限。

## §6 · 失效與隱含假設

### 6.1 論文自述限制（可定位）

- **單篇保守性**（§1、§7）：每交易日僅一篇 WSJ 稿，作者自稱「intentionally conservative…likely understates」，擴到投行研究稿或全量新聞料應更高——此為**作者假設，非本文實證**。
- **非因果**（§1）：明言「Nor is the paper an attempt to identify the causal effect of one particular shock」，並引 Cochrane「causation does not imply variation」。
- **資金流內生性**（§2.3）：trading flows 的 roughly 10% 可能因流量反應價格而違反 criterion 1；作者明言「Addressing flow endogeneity…beyond our objective」。
- **預期匹配檢定偏弱**（§6.2）：作者自述流程「likely underpowered」，1,250 段（4.1%）理由、710 個交易日（12.5%）為**下界**；bootstrap 顯示 anticipated 群 active $R^2$ 11.47% 落在同尺寸隨機子集分布的 upper 1.2% 尾（p = 1.2%，1,000 次重抽）。
- **悲觀偏誤無法歸因**（§B.1）：zero-shot implied return 均值 -0.16%、日頻 AR(1) 為 16.19%，作者明言「We do not provide a definitive decomposition」新聞業偏誤 vs LLM 解讀偏誤，並自承 prompt 中 "impact" 一詞可能帶負面暗示。
- **群集 $R^2$ 分解為近似**（§A.3）：$A\approx B\times C$ 僅在 active-day 條件變異數接近全樣本時成立；作者指出 Investor sentiment、Covid 偏離較大。
- **非線性延伸未做**（§5.1）：明言更彈性的方法留待未來。
- **原文內部不一致（本頁標注）**：§4.2 稱最終得到「11 meta-cluster names」，§4.3 稱結構為「12 meta-clusters」；圖 4 為 12 群。以正文與圖為準仍需核圖，**圖表待核**。

### 6.2 本頁推斷的隱含假設（明標推斷）

- **媒體同質性（推斷）**：全部解釋力建立在單一媒體、單一固定專欄體例上；換報紙、換語言、換成券商 commentary，映射是否穩定未測。作者的「保守下界」說法若成立，擴料應提升；若不成立，則此管線吃的是 WSJ 的寫作慣例而非世界知識——這是可檢驗的分叉。
- **「無理由 = 零報酬」的編碼（推斷）**：無 snippet 日把 $\hat r_t$ 設為 0，邏輯是「無新聞則價格不動」；但真實市場在小事件日仍有波動，此約定系統性壓低對小波動日的擬合，屬於**保守**方向的偏誤，而非樂觀方向。
- **Regime 依賴（推斷，未給分段數值）**：分群活躍度隨時間劇烈變化（geopolitics 成波、monetary policy 疫後放大、Covid 一次性），意味整體 $R^2$ 可能是特定年份驅動；原文未見分段樣本 $R^2$，**數值未披露，圖表待核**。
- **供應商綁定（推斷）**：text-embedding-3-large 與 gpt-4o-mini-2024-07-18 皆為託管模型，版本下架即不可複現；BERT 對照的具體型號未披露。
- **輔助資料缺失不等於生存者偏差**：非新聞基線的樣本重疊期不同（預期資料 2003Q1–2021Q3、流量止於 2019Q1），各自的 $R^2$ 只在重疊月份計算，**不可跨基線比大小**；這是重疊窗差異，不是 survivorship。

## §7 · 對比 & 面試 Tip

| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Roll (1988) | 作者明言 empirical exercise 不同（因子模型 $R^2$、有無公司特定新聞日比較） | 否 | 被重新詮釋，非被複現 |
| Cutler, Poterba, Summers (1989) | in-sample 調整 $R^2$（結論為難以解釋超過三分之一），非樣本外 | 否 | 不可與本篇 $R^2$ 直接比 |
| Baker et al. (2021) | 19 國、人工分類、只解釋**極端跳動**；本篇解釋全部變異並可拆題材 | 未披露 | 問題不同 |
| Bybee, Kelly, Su (2023) narrative asset pricing | 用新聞造**風險因子**；本篇解釋已實現報酬 | 未披露 | 目標不同 |
| Chen, Kelly, Xiu (2022) / Lopez-Lira & Tang (2026) | 報酬**預測**；本篇 concurrent 解釋 | 未披露 | 目標不同 |
| Koijen & Levy (2026) | agentic AI + earnings call + 個股 + 事件窗；本篇總體 + 新聞 | 未披露 | 問題不同 |
| Campbell-Shiller 分解 / Gabaix-Koijen 校準 | 給 $R^2$=100%（近似恆等式）或約 90% 的變異歸因，但解釋變量由價格導出 | 否 | 違反 criterion 1，不可比 |

**🎤 Interview Tip**
- **錯答**：「這篇用 LLM 預測 S&P 500，日頻 $R^2$ 30%，做成策略 Sharpe 應該不錯。」——錯在三處：這是 concurrent 解釋不是預測；5-fold 是隨機摺不是時序外；原文從頭到尾沒有策略、成本或容量。
- **正確答**：「這篇把 Roll 的 $R^2$ 從修辭變成可操作判準：criterion 1 要求解釋變量非價格導出，criterion 2 要求映射穩定。真正的技術核心是**遮蔽**——若不做價格語言篩檢，$R^2$ 是同義反覆；作者用 5-fold 而非 expanding window，是因為事後解釋允許 hindsight。零樣本臂的 -115.03% → 49.84% 修的是 implied return 的持續性偏誤（AR(1) 16.19%），不是世界知識。分群的 overall $R^2$ 差異主要來自 active-day 佔比，不能讀成題材因果強度。」

### 7.1 可證偽預測

1. **至 2027-12-31**：若作者「單篇保守、擴料更高」之說成立，任何把同一遮蔽協議擴到多篇／多來源語料的 replication，日頻 OOS $R^2$ 應高於本篇的 **29.82%**；若擴料後反而下降或持平，則此管線吃的是單一專欄的寫作慣例。此為研究假說，尚待驗證。
2. **至 2027-12-31**：關掉第二道洩漏篩檢（只用 Prompt 1）後，$R^2$ 應顯著上升——這是對「遮罩是必要而非裝飾」的直接檢定；若無變化，則前段推論失效。此為研究假說，尚待驗證。
3. **至 2027-12-31**：以時序 expanding window 取代隨機 5-fold，日頻 $R^2$ 應下降；下降幅度即為隨機摺樂觀偏差的量級（**目前未披露**）。此為研究假說，尚待驗證。

## §8 · For the Reader

- **因子研究員**：直接偷 §3 的兩段式 prompt 與「reason/price_movement 分欄輸出」結構，去審自家新聞因子有沒有把報酬描述混進特徵；再用 §A.3 的 $R^2_{\text{overall}}=R^2_{\text{active}}\times$active share 分解，判斷你的因子是「常出現但解釋力弱」還是「罕見但一擊命中」。
- **高頻執行**：本篇無助於執行；唯一可用的是 §6.3 的對齊檢定思路——把 FOMC 窗（30–100 分鐘）的高頻衝擊當外生錨，檢驗你的文本特徵是否真的抓到了政策意外，而不是抓到同日其他事件。
- **組合配置**：把這條管線當**歸因工具**而非信號。分群活躍度時間序列（geopolitics 成波、monetary policy 疫後放大）可以當 regime 標籤的候選輸入，但需自行做時序外檢定——原文的分群結果全部基於同一 pooled 模型，未做 regime 條件評估。
- **LLM-agent 開發者**：這裡的 agent 設計值得抄的是**角色分工**：抽取器只做閱讀理解（不准推理）、篩檢器只做洩漏標記、推論器才做金融判斷。三者分離使錯誤可定位；合併成一個 prompt 就無法區分「抽錯」與「推錯」。
- **RL / 策略研究者**：zero-shot 那條臂的失敗模式（持續性偏誤在日頻無害、加總到月度即爆炸）是典型的**時間尺度和信號半衰期錯配**；HAR 過濾本質上是一個 1/5/22/63 日的線性去卷積。任何把日頻模型輸出聚合到低頻的設計都該先做這個診斷。
- **研究學生**：把 §2.1 的兩條 criterion 背下來，它是全文的方法論骨架——criterion 1 排除 Campbell-Shiller 與 Gabaix-Koijen 那類「$R^2$ 很高但變量由價格導出」的練習，criterion 2 排除一次性擬合。這兩條也是審任何「解釋型」論文的通用尺。

## References

- Kelly, B., Y. Su, and C. Zhang (2026). *Explaining Realized Returns: Roll's $R^2$ 40 Years Later.* arXiv:2611.22277v1 [q-fin.GN], 17 Nov 2026. https://arxiv.org/abs/2611.22277v1
- Roll, R. (1988). *R².* Journal of Finance 43:541–66.（被重新詮釋的問題來源）
- Cutler, D. M., J. M. Poterba, and L. H. Summers (1989). *What Moves Stock Prices?* Journal of Portfolio Management 15:4–12.
- Baker, S. R., N. Bloom, S. J. Davis, and M. C. Sammon (2021). *What Triggers Stock Market Jumps?* NBER WP 28687.
- Bybee, L., B. Kelly, and Y. Su (2023). *Narrative Asset Pricing.* RFS 36:4759–87.
- Chen, Y., B. T. Kelly, and D. Xiu (2022). *Expected Returns and Large Language Models.* SSRN 4416687.
- Koijen, R. S. J., and B. Levy (2026). *Assessing the Benefits of Optimized Agentic AI Systems for Asset Pricing.* NBER WP 35431.
- Gabaix, X., and R. S. Koijen (2021). *In Search of the Origins of Financial Fluctuations: The Inelastic Markets Hypothesis.* NBER WP.
- Corsi, F. (2009). *A Simple Approximate Long-Memory Model of Realized Volatility.* Journal of Financial Econometrics 7:174–96.（HAR 過濾來源）
- Acosta, M., A. Ajello, M. D. Bauer, F. Loria, and S. Miranda-Agrippino (2025). *Financial Market Effects of FOMC Communication.* IMFS WP.（USMPD 高頻衝擊來源）
- McCracken, M. W., and S. Ng (2016). *FRED-MD.* JBES 34:574–89.
- 來源鏈接：https://arxiv.org/abs/2611.22277v1（本頁依可讀全文文本；圖片／附件未核）