<!-- ontology-5axis data=文本另类 horizon=跨周期 paradigm=生成式大模型 alpha=端到端表征 autonomy=全自动黑盒 -->

# One-shot EM 解構（One-shot EM）

> **發布**：2025-05-29 · （無 venue）
> **QuantML 導讀**：[九坤团队新作！一条数据训练AI超越上万条数据](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490558&idx=1&sn=71770df7b241030f2bef557a8b891b87&chksm=ce7e7ce0f909f5f68e8deb9e8a4788bc38419281ebeaee579d34670410632c13ac60e329937b#rd)
> **核心定位**：落點於 `生成式大模型 × 端到端表征` 軸，解構了 RL 後訓練對「海量標註數據 + 複雜獎勵函數 + 數千步迭代」的工程依賴。以單樣本無監督條件熵最小化，直接重塑 logits 分佈，釋放預訓練隱式推理能力。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 用單條無標籤數據與 10 步優化，直接最小化 token 級條件熵，取代傳統 RL 獎勵信號。② 核心 trick 是 `pass@k` 方差篩選高不確定性提示，觸發 logits 右偏（right-skewed shift），無監督釋放推理。③ 對 `端到端表征` 軸而言，證明「分佈塑形」優於「知識注入」，打破數據規模定律。④ 關鍵實證數字：MATH500 從 53.0 提升至 78.8（Δ 25.8）。

**X-Ray.** 在 `數據模態 × 學習範式` 的 Pareto 前沿上，One-shot EM 切斷了 RL 的數據-算力耦合鏈。它不學新知識，而是透過熵梯度強制模型對「正確路徑」過度自信，本質是分佈壓縮而非表征學習。這解了舊工程的三大坑：獎勵黑客（reward hacking）、標註瓶頸、以及 RL 收斂慢的計算開銷。但它的 envelope 極度脆弱：導讀明確指出第 10 步後損失持續下降但推理性能反降，證明 EM 是「分佈整形器」而非「優化器」，過度自信會導致 token 路徑坍縮。對量化讀者的意義在於：無監督目標函數在特定 regime（數學推理/低熵答案）可超越獎勵驅動，但必須嚴格配合 early-stopping 與溫度校準；若直接套用於高頻或多模態 Alpha 生成，需警惕 logit 右偏帶來的尾部風險與 seed 隨機性（導讀提及同設定下均分可相差多達兩倍）。

## §1 · 架構 / Core Mechanism
| 改動維度 | 傳統 RL / SFT 後訓練 | One-shot EM |
|---|---|---|
| 數據依賴 | 數千至數萬條高質量帶標籤數據 | 1 條無標籤數據 |
| 優化目標 | 獎勵最大化 + KL 約束 / 交叉熵 | Token 級條件熵最小化 |
| 收斂步數 | 數千步 | 10 步內收斂 |

**⚡ Eureka:** 放棄外部獎勵，直接最小化模型對自身預測的不確定性；用 `pass@k` 方差鎖定「能力臨界點」提示，10 步內強制 logits 右偏，無監督釋放推理。
**信息流 ASCII:**
```
Prompt (1 sample) → Pass@k Variance Filter → High-Uncertainty Selection
       ↓
Token-level Conditional Entropy Calculation → Gradient Update (10 steps)
       ↓
Logit Right-Shift → Greedy Decoding @ Temp 0.5 → Reasoning Output
```

## §2 · 數學層
📌 **Napkin Formula:**
$$L_{EM} = \sum_{t=t_{prompt}}^{T} H\left(p_\theta(\cdot \mid x_{<t})\right)$$
複雜度：單步 $O(T \cdot |\mathcal{V}|)$，總訓練僅 10 步。
**直覺:** 正確答案的分佈熵天然低於錯誤答案。EM 不依賴外部評分，而是讓模型在生成過程中自我壓縮不確定性，等價於對預訓練權重施加熵正則化梯度。
**Loss/訓練細節:** 學習率未披露，溫度 0.5，批量大小 64，最大響應長度 2048。損失在第 10 步降至較低水平後繼續下降，但推理性能開始衰退，證明目標函數僅在極短窗口內與能力正相關。

## §3 · 數據層
- **規模/頻率:** 單次訓練僅 1 條數據；實驗總計訓練 13,440 個模型以消除隨機性。
- **市場/時段:** 數學推理基準（MATH500, Minerva Math, 奧林匹克基準, AMC23）；NuminaMath 數據集用於 logit 分析。
- **來源與假設:** 數據池無標籤；假設「模型行為方差」可代理「熵敏感度」。樣本外假設：單樣本優於多樣本，因多樣本損失在第 3 步後持續波動，單樣本從第 3 步起平穩下降。

## §4 · 代碼層
| 欄位 | 狀態 |
|---|---|
| Repo | TBD |
| Checkpoint | TBD |
| License | TBD |
| 複現難度 | 中（需 Accelerate 框架，嚴格控制 seed 與溫度，早停機制關鍵） |
| 數據可得性 | 公開數學推理基準與 NuminaMath |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric (Accuracy) | 前SOTA (RL基線) | 本方法 (Qwen2.5-Math-7B) | Δ |
|---|---|---|---|---|
| MATH500 | Accuracy | 未披露 | 78.8 | 25.8 |
| Minerva Math | Accuracy | 未披露 | 35.3 | 24.3 |
| 奧林匹克基準 | Accuracy | 未披露 | 39.7 | 22.5 |
| AMC23 | Accuracy | 未披露 | 70.3 | 26.2 |
| 平均 | Accuracy | 未披露 | 未披露 | 24.7 |

**解讀:** Δ 屬真實能力躍升，但高度依賴 logit 右偏與溫度 0.5 的貪婪解碼。導讀未提供 RL 基線（Prime-Zero7B, RLVR-GRPO）的具體數值，故標未披露。需注意：Δ 在 10 步後會因過度自信而反轉；若未計入 seed 隨機性（均分可相差多達兩倍），實盤複現的期望收益將被高估。無成本/breakeven 數據，無法評估算力 ROI。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 
- 訓練隨機性極高，同設定下不同 seed 結果可相差多達兩倍。
- 第 10 步後損失持續下降但推理性能衰退，缺乏自適應早停或調度機制。
- 若應用於 RL 之後，會加劇分佈失真，導致四個基準性能穩步下降。

**6.2 推斷的隱含假設:**
- **Regime 依賴:** 假設「正確答案熵低」在數學/邏輯任務成立，但對開放域生成或高頻 Alpha 可能失效（正確路徑本身具有高熵或多模態）。
- **容量/成本:** 假設 10 步內分佈塑形足夠，忽略長尾提示的梯度飽和風險。
- **數據泄漏:** Pass@k 方差篩選若與評估集分佈重疊，可能引入前瞻偏差。
- **Survivorship:** 僅報告峰值性能與 16 次重複的平均，未披露失敗 seed 的分佈特徵。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| RLHF / GRPO | 獎勵驅動 vs 熵驅動；數據依賴 vs 單樣本 | 開源基線多 | 工業標準 |
| SFT | 監督交叉熵 vs 無監督熵最小化 | 開源 | 主流後訓練 |
| One-shot EM | 分佈右偏塑形 vs 知識注入 | 未開源 | v0.5 驗證期 |

🎤 **Interview Tip:** 
- ✅ 正確答：「EM 不是學習算法而是分佈整形器，它透過最小化條件熵觸發 logits 右偏，在 10 步內釋放預訓練隱式能力；但過度優化會導致路徑坍縮，必須配合早停與溫度校準。」
- ❌ 錯答：「EM 用一條數據就學會了數學推理，可以完全取代 RL，且訓練越久效果越好。」

**7.1 可證偽預測:** 若 2025-09-30 前無開源實作證明 EM 在代碼生成或對話任務上維持 >15pp 的 Δ 且 seed 方差 <30%，則該方法僅限於低熵推理 regime。

## §8 · For the Reader
- **因子研究員:** 將 `pass@k` 方差視為「不確定性代理因子」，可嘗試在 Alpha 生成階段引入熵正則化，替代複雜的獎勵塑形；但需嚴格監控第 10 步後的性能反轉。
- **高頻執行:** 不建議直接套用。EM 的 logit 右偏會壓縮輸出多樣性，在市場 regime 切換時易引發尾部風險；若用於信號過濾，需加入動態溫度衰減。
- **LLM-agent / RL 策略:** 將 EM 置於 SFT 或 RL 之前作為分佈預校準，可提升下游對監督信號的接受度；避免在 RL 之後疊加，否則會鎖定過於自信的模式並損害多樣性。

## References
- 九坤团队. (2025). *One-shot Entropy Minimization (EM)*. （無 venue）
- QuantML 導讀. (2025-05-29). [九坤团队新作！一条数据训练AI超越上万条数据](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247490558&idx=1&sn=71770df7b241030f2bef557a8b891b87&chksm=ce7e7ce0f909f5f68e8deb9e8a4788bc38419281ebeaee579d34670410632c13ac60e329937b#rd)
- Lineage: RLHF / GRPO / SFT 後訓練範式 → 無監督熵最小化 → 分佈塑形機制