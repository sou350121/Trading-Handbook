<!-- ontology-5axis data=量价表格 horizon=中长周期 paradigm=监督回归 alpha=因子挖掘 autonomy=人机协同可解释 -->

# Street PE 解構（Street PE）

> **發布**：2024-10-03 · （無 venue）
> **QuantML 導讀**：[用收益对股票估值](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486921&idx=1&sn=0a1e5fcf7210691c8bcd65dbba037275&chksm=ce7e6ad7f909e3c1ba4192b227f6fac30ab714ff4e84bb10058f307fac2fdb14a361f286859#rd)
> **核心定位**：落點於「量價表格 × 中長週期 × 因子挖掘」。解了傳統價值因子因 GAAP 盈餘過渡性波動（transitory volatility）導致估值比率方差分解錯配、樣本外預測力崩潰的 prior gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 用剔除一次性項目的擬制性盈餘（Street earnings）替代 GAAP 收益構建估值比率。② 核心 trick 是透過分析師口徑過濾高過渡性波動，使 Street PE 的變動真正反映未來回報預期而非盈餘雜訊。③ 對「因子挖掘 × 中長週期」軸★，提供了一個具備強經濟直覺且抗週期雜訊的定價錨。④ 樣本外預測 R² 顯著為正，且 Street PE 上升 1 單位預測未來五年回報下降近 5%（未驗證）。

**X-Ray.** 本文本質是「會計雜訊過濾 × 資產定價」的因子工程突破，而非傳統 ML 黑盒。它精準擊中價值因子長期失效的工程坑：GAAP 盈餘的過渡性波動掩蓋了長期基本面，導致傳統估值比率（GAAP PE/CAPE）的方差分解錯配，預測力在樣本外崩潰。Street PE 透過分析師口徑剔除一次性項目，使比率變動真正錨定未來回報預期而非盈餘雜訊。對量化讀者而言，它提供了一個具備強經濟直覺的中長週期定價錨，可直接嵌入 Fama-MacBeth 橫截面回歸或作為組合層級的 regime filter。然而，其 envelope 明確受限於分析師覆蓋偏差與會計準則演進，無法處理微結構流動性、短週期動能或分析師集體樂觀/悲觀的 regime shift。實戰中應將其視為「基本面信號去噪層」，而非獨立交易信號。

## §1 · 架構 / Core Mechanism
| 維度 | GAAP PE / CAPE / PD（前作） | Street PE（本方法） | 工程/定價意義 |
|---|---|---|---|
| 盈餘定義 | 會計準則口徑（含商譽減值/重組等一次性項目） | 分析師共識口徑（剔除非經常性/過渡性項目） | 過濾會計雜訊，使 $E$ 的波動率結構貼近長期基本面趨勢 |
| 方差分解錨點 | 價格變動主要被解釋為「未來盈餘增長」 | 價格變動主要被解釋為「未來回報變化」 | 修復 Campbell-Shiller 分解的錯配，對齊過度波動之謎的實證共識 |
| 預測框架 | 傳統 OLS / 簡單均值回歸 | Stambaugh 偏差修正 + Goyal-Welch OOS + Fama-MacBeth | 解決預測回歸的序列相關與有限樣本偏誤，提升樣本外稳健性 |

⚡ **Eureka:** 用分析師口徑的擬制性盈餘替代 GAAP 收益，使估值比率的分子（價格）與分母（盈餘）的波動來源解耦，比率變動真正追蹤未來回報預期而非會計雜訊。
**信息流:**
```
Raw Financials (Compustat) ──┐
                             ├──> Analyst Adjustments (I/B/E/S) ──> Street Earnings ──> Street PE ──> Campbell-Shiller Decomp ──> Return Predictability (Fama-MacBeth)
GAAP Earnings ───────────────┘
```

## §2 · 數學層
📌 **Napkin Formula:** 
$$ \text{Street PE}_t = \frac{P_t}{E^{street}_t}, \quad \Delta \ln(\text{Street PE}_t) \approx \underbrace{\sum \rho^j \Delta e_{t+j}}_{\text{未來盈餘增長}} - \underbrace{\sum \rho^j r_{t+j}}_{\text{未來回報}} $$
**複雜度:** $O(N)$ 會計調整 + 標準計量回歸（無迭代優化）。
**直覺:** 當分母雜訊被剔除後，Campbell-Shiller 分解的權重從「盈餘增長項」大幅轉移至「未來回報項」，使估值比率成為回報預期的有效代理變數。
**Loss/訓練細節:** 非端到端 ML。採用 OLS 預測回歸，核心在統計修正：Stambaugh (1999) 偏差校正、重疊觀察值誤差結構處理、Goyal-Welch (2008) 樣本外 $R^2$ 與 Clark-West t 統計量驗證。橫截面採用 Fama-MacBeth 兩階段回歸控制 FF3 因子。

## §3 · 數據層
- **市場/時段:** 美國股市（S&P 500 為核心樣本），中長週期（1/3/5 年預測視角）。
- **頻率/規模:** 年度/季報頻率（依賴財報與分析師更新）。樣本容量取決於 I/B/E/S 覆蓋度與 Compustat 歷史深度。
- **來源:** I/B/E/S（擬制性盈餘共識）、Compustat（特殊項目/特殊線目複製）、SEC 財報。
- **樣本外與容量假設:** 採用 Goyal-Welch 滾動分割驗證 OOS 稳健性；容量假設偏向大中盤流動性股票，小盤股受分析師覆蓋偏差（coverage bias）限制，實戰需加權或過濾。

## §4 · 代碼層
| 欄位 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | N/A（計量模型，無權重檔案） |
| License | TBD |
| 複現難度 | Low（標準計量經濟學流程 + 公開會計數據） |
| 數據可得性 | I/B/E/S（機構付費/付費終端）、Compustat（機構付費）；特殊項目可透過 Compustat `SPI` 線目近似，但與分析師口徑存在殘差 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| US Large Cap | 樣本外 $R^2$ (1yr) | 負值（GAAP PE/CAPE/PD） | 顯著為正 | 未披露 |
| US Large Cap | Clark-West t-stat | 未披露 | 顯著 | 未披露 |
| US Large Cap | 5yr 回報預測係數 | 未披露 | 上升 1 單位 → 回報下降近 5% | 未披露 |
| US Cross-Section | Fama-MacBeth 因子溢價 | 未披露 | 顯著正（控制 FF3 後） | 未披露 |
| US Large Cap | IR / Sharpe / AR / MDD | 未披露 | 未披露 | 未披露 |

**解讀:** 真正的 capability 來自「方差分解權重轉移」與「樣本外 $R^2$ 由負轉正」，證明 Street PE 修復了傳統價值因子的預測失效。Δ 中的具體係數（如 -5%）可能受樣本期（含低利率/寬貨幣 regime）與 Stambaugh 修正幅度影響，實戰需計入交易成本、再平衡頻率與分析師修訂滯後，當前 Δ 未計入執行摩擦與覆蓋偏差。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 未處理交易成本/周轉率；依賴分析師共識的理性與一致性；未探討會計準則變革（如 IFRS/GAAP 差異）對跨市場適用的影響。
**6.2 推斷的隱含假設:** 
- **Regime 依賴:** 假設分析師調整行為與市場定價邏輯在樣本內外結構穩定；若進入高波動/流動性緊縮 regime，分析師可能滯後或集體下修，導致 Street PE 信號遲滯。
- **容量/成本:** 隱含大中盤流動性假設；小盤股因覆蓋不足會產生 selection bias。
- **數據泄漏:** I/B/E/S 共識為動態修訂序列，實證若未嚴格使用 snapshot 或 release-date 對齊，易引入前瞻偏差（look-ahead bias）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| GAAP PE / CAPE | 盈餘雜訊過濾 vs 原始會計/平滑 | N/A | 學術基線 |
| Book-to-Market | 資產負債表錨點 vs 盈餘流量錨點 | N/A | 經典價值因子 |
| Analyst Revision Factors | 預期變化率 vs 靜態估值比率 | N/A | 動能/基本面混合 |

🎤 **Interview Tip:** 
- ✅ 正確答：「Street PE 的核心不是預測盈餘，而是透過剔除過渡性項目使估值比率的方差分解對齊未來回報。實戰需嚴格處理 I/B/E/S 的 snapshot 時戳以避免 look-ahead，並用 Stambaugh 修正解決預測回歸的序列相關偏誤。」
- ❌ 錯答：「把它當 ML 特徵直接丟進 XGBoost 跑橫截面排序，忽略會計調整的經濟直覺與樣本外 $R^2$ 的統計檢驗。」

**7.1 可證偽預測:** 若至 `2025-12-31` 分析師修訂行為因 AI 輔助或監管收緊發生結構性偏移（如一次性項目定義泛化），Street PE 的樣本外 $R^2$ 將在高波動子樣本中衰減至 GAAP PE 水準以下。

## §8 · For the Reader
- **因子研究員:** 不要直接複製比率。實作時必須嚴格對齊 I/B/E/S 的 `release_date` 與 `as_of_date`，構建 snapshot 面板；將 Street PE 作為 Fama-MacBeth 的核心解釋變數，並對照 `SPI` (特殊項目) 做 robustness check。
- **組合配置/宏觀:** 將其視為「中長週期 regime filter」而非擇時信號。當 Street PE 處於歷史分位極值時，調整權益倉位權重或對沖比率，配合波動率曲面或信用利差確認 regime 轉換。
- **量化開發/數據工程:** 構建自動化 reconciliation pipeline：Compustat GAAP → 剔除 `SPI`/重組線目 → 對齊 I/B/E/S 共識。監控分析師覆蓋度變化（coverage churn）作為信號衰減的先行指標，實戰需加入 turnover 約束與成本模型。

## References
- 原論文: *Street PE* (2024) · 無 venue / arXiv: None
- Lineage: Shiller (1981) 過度波動之謎 → Campbell & Shiller (1988) 恒等式分解 → Stambaugh (1999) 預測回歸偏差修正 → Goyal & Welch (2008) 樣本外預測檢驗 → Fama & MacBeth (1973) 橫截面定價
- QuantML 導讀: [用收益对股票估值](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247486921&idx=1&sn=0a1e5fcf7210691c8bcd65dbba037275&chksm=ce7e6ad7f909e3c1ba4192b227f6fac30ab714ff4e84bb10058f307fac2fdb14a361f286859#rd)