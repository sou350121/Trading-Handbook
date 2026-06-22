<!-- ontology-5axis data=微观盘口 horizon=高频日内 paradigm=强化学习 alpha=组合执行优化 autonomy=全自动黑盒 -->

# Hawkes LOB + PPO/SIL 解構（Hawkes LOB + PPO/SIL）

> **發布**：2025-11-03 · （無 venue）
> **QuantML 導讀**：[UCL x JP Morgan ｜ AI做市商悖论：利润飙升，对手滑点为何反降？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492212&idx=1&sn=7e6d45682844c8b65bd6c8659315d24d&chksm=ce7d856af90a0c7c488dba97b70cc4696ac4626fe1fb86d84a46aca31a40bb7245e6c0965398#rd)
> **核心定位**：落點於「高頻日內 × 組合執行優化」軸，解決傳統做市商依賴逆向選擇（Adverse Selection）與RL代理利潤/對手滑點零和博弈的Prior Gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 以PPO+SIL脈衝控制框架在Hawkes LOB中訓練HFT做市商對抗MFT TWAP執行。② 核心Trick是將TWAP狀態（無/買/賣）注入fRL狀態空間，並用SIL壓制PPO高方差。③ 對「組合執行優化」軸★：打破「RL利潤必伴隨對手滑點惡化」的假設，揭示穩定價格/高效庫存管理可獨立於逆向選擇獲利。④ fRL在TWAP買單期間Sharpe飆至150.43，但TWAP買方滑點從7.05 bps降至0.24 bps。

**X-Ray.** 本方法在五軸Pareto前沿上，刻意避開了「預測價格方向」的紅海，轉而將RL的優化目標錨定於「流動性供應效率與庫存動態平衡」。傳統HFT工程坑在於微觀結構噪声與事件驅動的非均勻時間步，Hawkes過程自激特性與脈衝控制（Pulse Control）恰好對齊了LOB的異步觸發機制。然而，該架構打不開的Envelope在於Regime依賴性極強：fRL的獲利完全綁定於TWAP的機械式切片行為，一旦對手切換至VWAP/POV或引入隨機化延遲，其狀態先驗將迅速失效。對量化讀者而言，這並非一個可直接上線的Alpha生成器，而是一個「執行成本對沖」的對策模型；它證明在微觀盤口層面，RL做市商的價值不在於掠奪（Front-running），而在於吸收確定性訂單流並平滑價格衝擊，這為機構MFT的執行演算法設計提供了全新的防禦視角。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作/基線 (uRL) | 本法 (fRL) | 工程意圖 |
|---|---|---|---|
| 狀態空間 | 僅LOB狀態與自身庫存 | 注入TWAP狀態標記（無/買/賣） | 將黑盒對抗轉為條件式流動性供應 |
| 優化目標 | 純PPO策略梯度 | PPO + SIL (Self-Imitation Learning) | 壓制脈衝控制下的高方差，錨定高回報軌跡 |
| 控制頻率 | 連續/事件驅動 | 脈衝控制 (Pulse Control) | 對齊HJB-QVI離散決策點，避免過度交易 |

**⚡ Eureka:** 將TWAP執行階段作為離散狀態先驗注入，使RL從「被動響應LOB事件」轉為「主動對齊確定性訂單流」。
**信息流:**
```
LOB Events → Hawkes Process → Pulse Trigger → [Decision Net] → [Action Net]
                                      ↑
                              (fRL State: TWAP Mode)
                                      ↓
                          Order Placement / Cancel / Skip
```

## §2 · 數學層
📌 **Napkin Formula:** HJB-QVI 近似解。PPO Loss: $L^{CLIP}(\theta) + \beta L^{SIL}(\theta)$。SIL項鼓勵模仿歷史高回報軌跡以壓制方差。
**直覺:** 脈衝控制將連續時間HJB轉為離散決策點，SIL提供行為克隆的穩定錨點，避免PPO在稀疏獎勵環境中發散。
**Loss/訓練細節:** 雙網絡架構（決策網絡決定觸發時機，行動網絡決定具體動作），狀態空間擴展1維（TWAP標記），計算複雜度隨LOB深度與訂單量級呈線性增長。

## §3 · 數據層
Hawkes LOB模擬環境。頻率：代理以固定頻率或LOB事件觸發調用。市場/時段：模擬盤口，無真實交易所時段。樣本外：140 episodes訓練 vs 246 episodes驗證（後者庫存偏差惡化，棄用）。容量假設：單代理對抗單TWAP，未測試多代理競爭或容量上限。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 中（需自建Hawkes LOB環境+PPO/SIL框架） | 導讀未提供模擬參數與代碼 |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| Hawkes LOB模擬 | TWAP買方滑點(bps) | hPOV基線 18.68 | fRL對抗 0.24 | -18.44 |
| Hawkes LOB模擬 | TWAP買方滑點(bps) | rPOV1基線 7.05 | fRL對抗 0.24 | -6.81 |
| Hawkes LOB模擬 | fRL Sharpe (買單期間) | uRL對抗hPOV -95.89 | fRL 150.43 | 246.32 |
| Hawkes LOB模擬 | fRL Sharpe (賣單期間) | uRL對抗hPOV -92.33 | fRL -4.96 | 87.37 |

**解讀:** Δ in slippage 是真 capability，反映fRL透過穩定價格/高效庫存管理吸收確定性訂單流，而非傳統逆向選擇。Δ in Sharpe 顯示極端非對稱學習：買單期間策略有效，賣單期間因庫存偏差（中位庫存未如預期減少）導致表現惡化。無過擬合跡象，但Regime依賴性高；成本僅計1 bps手續費，實盤延遲與衝擊成本未計。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** fRL賣單表現差（Sharpe -4.96），庫存未如預期減少；uRL測試中頻率差異不真實；246 episodes模型庫存偏差惡化。
**6.2 推斷的隱含假設:** Regime依賴（TWAP機械切片可預測）；容量假設（單對單模擬，未驗證多代理競爭或訂單簿深度飽和）；成本假設（僅計1 bps手續費，未計滑點以外的隱性成本或延遲）；數據泄漏風險（fRL直接獲知TWAP方向，實盤需依賴信號提取）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Traditional HFT MM | 依賴統計套利/逆向選擇 | 閉源 | 成熟 |
| RL MM (Standard) | 連續時間控制/高方差 | 部分開源 | 實驗 |
| 本法 (Hawkes+PPO/SIL) | 脈衝控制+TWAP狀態先驗/SIL降方差 | 未披露 | v0.5 |

🎤 **Interview Tip:** 
正確答法：強調脈衝控制對齊LOB異步事件，SIL解決PPO高方差，且RL做市商本質是流動性提供者而非掠奪者。
錯答法：認為RL做市商通過前置交易（Front-running）賺取TWAP滑點，或將滑點下降歸因於過擬合。

**7.1 可證偽預測:** 若2025-Q4實盤測試中，將TWAP替換為隨機化延遲的VWAP執行器，fRL的Sharpe比率將回落至<50且滑點優勢消失。

## §8 · For the Reader
* **因子研究員:** 將TWAP狀態標記轉化為微觀流動性因子，用於預測短期價格衝擊衰減。
* **高頻執行:** 參考脈衝控制架構重構執行演算法，將「緊急狀態」觸發閾值動態化。
* **組合配置:** 將RL做市商視為波動率吸收器，在組合層面對沖機構大單帶來的臨時衝擊成本。
* **RL策略:** SIL作為策略梯度穩定器，在稀疏獎勵環境中優先於純PPO；狀態空間注入先驗比盲目增加網絡深度更有效。

## References
* QuantML 導讀: [UCL x JP Morgan ｜ AI做市商悖论：利润飙升，对手滑点为何反降？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492212&idx=1&sn=7e6d45682844c8b65bd6c8659315d24d&chksm=ce7d856af90a0c7c488dba97b70cc4696ac4626fe1fb86d84a46aca31a40bb7245e6c0965398#rd)
* Lineage: Jain et al. 2025 (PPO+SIL框架) / Maitrier, Loeper, and Bouchaud 2025 (平方根定律與價格松弛建模)