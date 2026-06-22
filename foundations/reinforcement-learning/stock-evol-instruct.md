<!-- ontology-5axis data=多模态 horizon=日频波段 paradigm=强化学习 alpha=端到端表征 autonomy=Agent自主演进 -->

# Stock-Evol-Instruct 解構（Stock-Evol-Instruct）

> **發布**：2025-07-20 · （無 venue）
> **QuantML 導讀**：[结合强化学习与LLM的量化模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491063&idx=1&sn=de2d415fa30550d0c606827830db0cdd&chksm=ce7e7ae9f909f3ff02a63977c4f3f6972f4cf4837fb3b858914890f3a4ffc5cfe28bf201991a#rd)
> **核心定位**：落點於日頻波段與Agent自主演進，解決純DRL對非結構化新聞語義的鈍感，以及純LLM缺乏動態市場反饋閉環的prior gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** 結合DQN/DDQN與LLM，提出Stock-Evol-Instruct指令演進演算法微調大模型，實現新聞輔助的日頻股票交易。核心trick是用LLM分析新聞動態調整RL獎勵，並設計自動指令生成演算法產生高質量交易指令微調LLM，實現策略自主演進。這對「端到端表徵」與「Agent自主演進」軸具指標意義，因它將新聞語義直接映射為獎勵函數與監督訊號，繞過傳統因子工程。導讀給出Mistral-7B在JPM上ROI達53.15%，顯著超越純RL基線。

**X-Ray.** 放回五軸Pareto，本法在「日頻波段」與「Agent自主演進」交會，刻意放棄高頻低延遲，換取新聞語義的跨期定價權。它解了舊工程坑：傳統DRL特徵工程依賴人工選股/因子，本法用LLM-as-Judge與規則真值自動演進指令，將非結構化文本轉為可微獎勵與SFT數據。預測其打不開的envelope：僅依賴兩支個股（SLV/JPM）與有限提示集（20個），未覆蓋流動性衝擊、滑點與跨市場regime切換；LLM推理延遲與API成本在實盤將侵蝕日頻邊際利潤。對量化讀者意義：提供一條「語義獎勵塑形+指令數據飛輪」的RL-LLM協同路徑，但需嚴防前瞻偏差與過擬合單一標的。

## §1 · 架構 / Core Mechanism
**1.1 三大改動 vs 前作**
| 維度 | 前作 (純DRL/純LLM) | Stock-Evol-Instruct | 工程意圖 |
|---|---|---|---|
| 獎勵機制 | 固定價格變動/風險指標 | LLM新聞共識動態乘數（確認×2，衝突裁剪至[-1,1]） | 將語義置信度內化為RL獎勵塑形 |
| 數據生成 | 人工標註或靜態回測 | Stock-Evol-Instruct（深度/廣度演進+規則真值+淘汰） | 自動化高質量指令數據飛輪，替代人工SFT |
| 決策閉環 | 單向預測或單向執行 | LLM輔助DQN/DDQN + 微調LLM獨立代理 | 語義先驗與價格動能雙軌驗證，降低Q值高估 |

**1.2 ⚡ Eureka 一句話 trick + 直覺**
Trick：用規則系統（收盤vs開盤/2日MA）生成「真值」標籤，驅動LLM演進指令並進行SFT，同時讓LLM新聞解讀直接乘數化RL獎勵。
直覺：把LLM從「黑盒預測器」降維成「獎勵塑形器」與「數據工廠」，RL負責動態執行與風險控制，LLM負責跨模態語義對齊，兩者透過指令集與獎勵函數解耦耦合。

**1.3 信息流 ASCII 圖**
```
[新聞標題] -> LLM情感/方向預測 -> [獎勵塑形器] -> DQN/DDQN Agent -> [交易執行]
      |                                              |
      v                                              v
[歷史行情+規則真值] -> Stock-Evol-Instruct -> [高質量指令集] -> SFT微調LLM -> [獨立交易代理]
```

## §2 · 數學層
📌 **Napkin Formula**：
$Q_{target} = r + \gamma \cdot Q_{target\_net}(s', \arg\max_{a} Q_{main}(s', a))$ （DDQN解耦選擇與評估）
獎勵裁剪：$r_{clipped} = \text{clip}(r_{base} \cdot \mathbb{I}_{LLM\_agree}, -1, 1)$
複雜度：O(N·H) 前向推理 + O(T) 指令演進迭代，SFT微調依賴標準Cross-Entropy。
直覺：DDQN抑制Q值高估是RL基線，關鍵在於獎勵項引入LLM共識乘數，使策略梯度在新聞強烈看多/看空時放大更新步長；SFT loss 僅對齊指令-動作映射，不直接優化Sharpe/ROI，故財務指標屬下游涌现。
Loss/訓練細節：導讀未披露具體learning rate、batch size或SFT epoch數，僅提及使用pytorch復現與監督微調。

## §3 · 數據層
資料規模/頻率/市場/時段：日頻波段，標的為Silver (SLV) 與 JPMorgan (JPM)。數據劃分為訓練集與測試集，具體區間與樣本量未披露。
怎麼來：歷史價格時間序列 + 對應財經新聞標題。初始提示僅20個，經LLM評分保留>80分的9個。
樣本外與容量假設：僅兩支美股，未驗證跨市場/跨資產類別泛化；假設新聞標題與日頻價格存在穩定因果/領先關係，未計入流動性與滑點容量限制。

## §4 · 代碼層
| 欄位 | 內容 |
|---|---|
| Repo | TBD |
| Checkpoint | Mistral-7B / LLaMA-3-8B 微調權重（未提供下載鏈） |
| License | 依基座模型（Apache 2.0 / Llama 3 Community） |
| 複現難度 | 中（需LLM API/本地部署 + DRL環境搭建 + 指令演進邏輯實現） |
| 數據可得性 | 低（新聞標題與對應標籤需自行爬取或購買，規則真值邏輯需自訂） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| JPM | ROI | FinRL 0.04% | Mistral-7B 53.15% | 53.11% |
| JPM | ROI | FinGPT -8.28% | Mistral-7B 53.15% | 61.43% |
| SLV | ROI | FinRL 7.33% | Mistral-7B 48.36% | 41.03% |
| SLV | ROI | FinGPT -20.58% | Mistral-7B 48.36% | 68.94% |

**解讀**：Δ 計算僅基於導讀逐字ROI。Mistral-7B在兩標的的ROI大幅超越純RL與純LLM基線，顯示指令演進SFT有效將語義轉為盈利能力。但導讀明確指出SR與ROI存在不一致性（如Mistral-7B+DDQN在JPM上SR達2.29但ROI為-10.39%），此處Δ僅反映單一微調代理的財務結果，未涵蓋RL耦合狀態。高Δ可能來自過擬合兩支個股的特定新聞週期，且未扣除API推理成本與滑點，實盤邊際利潤需重新評估。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations**：初始提示僅20個，範圍受限；SR與ROI不一致性可能源於高風險導致巨大收益/損失，SR無法完全反映整體盈利能力；LLM在交易場景性能仍依賴指令微調改進。
**6.2 推斷的隱含假設**：
- Regime依賴：假設新聞標題對日頻價格的領先/同步關係在測試期穩定，未驗證黑天鵝或流動性枯竭regime。
- 容量/成本：未計入LLM推理延遲與API成本，日頻波段若頻繁調倉，成本可能侵蝕ROI。
- 數據泄漏：規則真值使用「收盤價vs開盤價/2日MA」，若新聞發布時間與收盤價計算存在時間錯位，可能引入前瞻偏差。
- Survivorship：僅選取SLV與JPM兩支活躍個股，未涵蓋退市或低流動性標的。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| FinRL (純DRL) | 僅依賴價格/技術指標，無語義輸入 | Open | 成熟基線 |
| FinGPT (純LLM) | 靜態微調，無動態RL反饋閉環 | Open | 金融微調基線 |
| Stock-Evol-Instruct | RL獎勵塑形 + 指令數據飛輪 + SFT代理 | 未開源 | v0.5 |

🎤 **Interview Tip**：
正確答：「本法核心不在於LLM直接預測漲跌，而在於將新聞語義轉化為RL獎勵乘數與SFT指令集，實現『語義先驗-動態執行』解耦。需警惕SR與ROI的脫鉤，以及單標的過擬合風險。」
錯答：「LLM預測準確率越高，交易回報就越高，所以直接微調LLM做預測就行。」（忽略RL反饋閉環與獎勵塑形機制，混淆預測指標與財務指標）
**7.1 可證偽預測帶日期**：若於TBD前公開跨市場（如A股/港股）日頻回測，且扣除API成本後ROI仍顯著高於FinRL，則驗證語義飛輪泛化性；否則僅為單市場過擬合。

## §8 · For the Reader
- **因子研究員**：關注LLM-as-Judge評分機制與指令演進邏輯，可嘗試將新聞情緒因子替換為LLM生成的動態權重，但需嚴格時間戳對齊防泄漏。
- **高頻執行**：本法屬日頻波段，不適用HFT；但獎勵裁剪與LLM共識機制可借鑒至中頻動能策略的風險過濾模塊。
- **組合配置**：SR與ROI不一致性提示需雙指標監控；可將微調LLM代理作為衛星策略，與傳統均值回歸主策略正交配置。
- **LLM-agent/RL策略**：指令數據飛輪（Evol-Instruct + 規則真值）是低成本構建垂直領域SFT數據的有效路徑，可遷移至期貨/外匯日頻交易。
- **研究學生**：重點複現DDQN與獎勵塑形器的耦合邏輯，驗證不同Prompt設計對SR/ROI的影響，並嘗試擴展至多標的組合優化。

## References
- 原論文：Stock-Evol-Instruct (無 venue, 2025-07-20)
- Lineage: WizardLM (Evol-Instruct) / DQN & DDQN / FinRL / FinGPT
- QuantML 導讀鏈接：[结合强化学习与LLM的量化模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247491063&idx=1&sn=de2d415fa30550d0c606827830db0cdd&chksm=ce7e7ae9f909f3ff02a63977c4f3f6972f4cf4837fb3b858914890f3a4ffc5cfe28bf201991a#rd)