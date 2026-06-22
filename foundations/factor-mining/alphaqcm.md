<!-- ontology-5axis data=量价表格 horizon=日频波段 paradigm=强化学习 alpha=因子挖掘 autonomy=人机协同可解释 -->

# AlphaQCM 解構

> **發布**：2025-06-18 · ICML25
> **QuantML 導讀**：[ICML 25 | 基于分布强化学习的Alpha因子挖掘](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490762&idx=1&sn=e6bd19c7f6fede113ab6f677629b5449&chksm=ce7e7bd4f909f2c275654c6c84008b687585ddddddc51047f4802217a8191326f340f8ea0509#rd)
> **核心定位**：落點於「強化學習 × 因子挖掘」軸，將公式化 Alpha 搜尋重構為序列決策 MDP，解了傳統 GP 指數爆炸與 AlphaGen 在非平穩/稀疏獎勵環境下的探索失效 gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `量价表格` | `日频波段` | `强化学习` | `因子挖掘` | `人机协同可解释` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 將協同公式化 Alpha 挖掘建模為 MDP 序列決策問題。② 核心 trick 為 QCM（Quantiled Conditional Moment），從有偏分位數估計中提取無偏方差作為探索獎勵。③ 對「因子挖掘」軸★，以無偏不確定性驅動探索，破解非平穩與獎勵稀疏雙重瓶頸。④ 導讀給出全市場測試集 IC 達 9.16%，較 AlphaGen 的 6.04% 領先 3.12pp。

**X-Ray.** 放回五軸 Pareto，該方法在「可解釋性」與「自動化」之間放棄端到端黑箱，選擇 RPN 令牌序列生成以維持公式化透明。解了舊工程坑：GP 的組合爆炸與 RL 的獎勵稀疏/非平穩。預測打不開的 envelope：線性組合元 Alpha 的假設限制了高階非線性交互；日頻波段在微結構噪音與未計交易成本下的實盤滑點可能侵蝕 IC 優勢。對量化讀者意義：提供了一套可審計的因子生成流水線，但需警惕 MDP 狀態定義（令牌序列）與真實市場狀態（宏觀/資金流）的錯配，實盤需加入成本約束與 regime 過濾。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | AlphaGen (PPO) | AlphaQCM | 工程意義 |
|---|---|---|---|
| 探索機制 | 標準策略熵/ε-greedy | QCM 無偏方差引導探索 | 直接對策非平穩與稀疏獎勵 |
| 狀態表示 | 傳統特徵/池狀態 | RPN 令牌序列 (Max 20) | 將公式構建轉為確定性序列決策 |
| 獎勵設計 | 增量 IC / 絕對收益 | 增量 IC + 無效公式負獎勵 | 聚焦池內邊際貢獻，強化協同性 |

**1.2 ⚡ Eureka**
將分位數與矩的關係構建為線性回歸，讓偏差項被截距吸收，從而從有偏分位數中提煉無偏方差作為探索獎勵。

**1.3 信息流 ASCII**
```
[State: Token Seq] → [Quantile Net (IQN)] → [QCM Regressor] → [Unbiased Variance]
       ↓
[Action Policy: Q + β*Var] → [Env: Linear Pool Fit & IC Δ] → [Reward]
```

## §2 · 數學層
**📌 Napkin Formula**
$Q_{\tau} \approx \mu + \sigma \cdot \Phi^{-1}(\tau) + \dots$ (Cornish-Fisher 展開)
OLS 回歸有偏分位數 $\hat{Q}_{\tau}$，偏差項 $\epsilon$ 被隔離至截距，解耦出無偏方差 $\hat{\sigma}^2$。
動作選擇：$a^* = \arg\max_a (Q(s,a) + \beta \cdot \hat{\sigma}^2(s,a))$
**直覺**：不糾結於修正有偏的均值估計，而是「將錯就錯」提取其波動結構，用不確定性驅動探索。
**Loss/訓練**：DQN MSE + IQN Quantile Huber Loss + QCM OLS 最小化殘差。雙網絡並行，回合末執行一次 OLS，訓練開銷略高於標準 DQN，但探索樣本效率提升。

## §3 · 數據層
- **資料規模/頻率/市場/時段**：中國 A 股（CSI 300 / CSI 500 / 全市場）+ S&P 500（泛化測試）。日頻。訓練 2010-2019 / 驗證 2020 / 測試 2021-2022。
- **怎麼來**：量價歷史數據，經逆波蘭表示法（RPN）轉為令牌序列。
- **樣本外與容量假設**：嚴格時間劃分樣本外。容量假設未披露，線性組合元 Alpha 暗示低頻/中頻容量，未計交易成本與衝擊。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 中（需自構建 RPN 環境與 IQN/DQN 雙網絡） | 未公開（導讀僅提「見星球」） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| CSI 300 | Out-of-sample IC | AlphaGen 8.13% | AlphaQCM 8.49% | 0.36pp |
| 全市場 | Out-of-sample IC | AlphaGen 6.04% | AlphaQCM 9.16% | 3.12pp |
| S&P 500 | Out-of-sample IC | 未披露 | 未披露 | 未披露 |

**解讀**：Δ 在複雜系統（全市場）中放大，證實 QCM 探索對高維稀疏獎勵有效。但 IC 為樣本外純預測力，未扣除交易成本/滑點/換手率，實盤 Sharpe 可能顯著低於 IC 暗示水平。部分 Δ 可能來自 AlphaGen 基線未針對非平穩性優化，屬算法設計紅利而非純粹因子質量躍升。其他基線（Alpha101, GP, MLP 等）導讀未給逐字 IC 值，故標未披露。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**
線性組合元 Alpha 限制高階交互；純數據驅動初期慢於專家預訓練（10%-20% 階段）；美國市場 IC 普遍低於 A 股。

**6.2 推斷的隱含假設**
- **Regime 依賴**：訓練期 2010-2019 結構可能與測試期 2021-2022 不同，MDP 非平穩性實為常態，模型可能過度適應特定波動率 regime。
- **容量/成本**：公式化因子池通常容量有限，未披露最大可承載資金與換手率假設。
- **數據泄漏/Survivorship**：RPN 生成過程若未嚴格隔離未來數據，易產生前瞻偏差；未明確說明是否處理退市股票。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| AlphaGen | 探索機制（PPO vs QCM無偏方差） | TBD | 已發表 |
| GP w/ filter | 搜尋範式（遺傳規劃 vs 序列決策RL） | TBD | 已發表 |

🎤 **Interview Tip**
- **正確答**：QCM 的核心是透過 Cornish-Fisher 展開的線性回歸，將分位數估計的偏差項隔離至截距，從而解耦出無偏方差作為探索獎勵，直接對策 MDP 的非平穩與稀疏獎勵。
- **錯答**：用強化學習直接預測股價/收益，或者認為方差是用來做風險控制的。

**7.1 可證偽預測帶日期**：若 2025-Q3 前無開源代碼與獨立機構在含成本環境下復現全市場 IC > 8.5%，則該方法實戰價值存疑。

## §8 · For the Reader
- **因子研究員**：關注 RPN 令牌生成邏輯與線性池組合機制，可嘗試將 QCM 探索獎勵嵌入現有因子生成流水線，替代隨機變異。
- **高頻執行**：日頻波段因子換手率未披露，實盤前務必用歷史成交數據模擬衝擊成本，IC 優勢可能被滑點吞噬。
- **組合配置**：線性元 Alpha 權重隨時間動態調整，需監控池內因子共線性與權重衰減，建議加入正交化或風險預算約束。
- **RL 策略**：學習如何將「有偏估計」轉化為「無偏矩」的統計技巧，可遷移至其他非平穩環境（如訂單流預測）。
- **研究學生**：理解分位數回歸與矩估計的對偶性，避免在實作中直接對有偏分位數求樣本方差。

## References
- AlphaQCM. ICML 2025.
- Lineage: AlphaGen (RL for Alpha Mining) · QRDQN/IQN (Distributional RL) · GP (Genetic Programming for Factors)
- QuantML 導讀鏈接：見上方