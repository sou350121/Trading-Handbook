<!-- ontology-5axis data=量价表格 horizon=跨周期 paradigm=监督回归 alpha=端到端表征 autonomy=全自动黑盒 -->

# RWKV-TS 解構（RWKV-TS）

> **發布**：2024-08-11 · （無 venue）
> **QuantML 導讀**：[RWKV-TS: 超越传统RNN的时序模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247485732&idx=1&sn=5e362c713b852da5d597e0acb9e8c745&chksm=ce7e6e3af909e72c557fc9eb6aaf0f222329b727f1d3c28c2ebaf8f75b7fd17719dce26c2bae#rd)
> **核心定位**：將線性RNN（RWKV）遷移至時間序列，以O(L)複雜度與分塊技術切掉Transformer的二次方算力瓶頸，解決長序列依賴捕捉與推理延遲的Prior Gap。

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 將RWKV線性RNN架構引入時間序列預測/分類/檢測任務。② 核心Trick為實例歸一化+分塊（Patching）+可訓練WKV向量替代自注意力。③ 對「跨周期」與「自動黑盒」軸★，以O(L)複雜度實現並行計算，大幅壓降延遲與內存。④ 導讀未給量化結果。

**X-Ray.** RWKV-TS在「算力效率-序列長度」Pareto前沿上，用WKV算子與分塊技術硬切了Transformer的O(L^2)注意力矩陣。對量化實戰而言，它解了長窗口特徵提取時的顯存爆炸與推理延遲坑，使跨周期量價因子能在低延遲節點並行推演。但其單向編碼器架構與實例歸一化，隱含了對分佈漂移的強假設：實例歸一化會抹除跨樣本的絕對量級資訊（對價格/成交量絕對值敏感的因子極不友好），且單向結構在插補/回溯任務中天然劣於雙向模型。它打不開的Envelope是「非平穩宏觀 regime 切換」與「極端流動性枯竭」下的穩健性，因為線性RNN缺乏全局動態路由，WKV的循環更新在劇烈跳空時易產生狀態積累偏差。對量化讀者，此架構適合做高頻/中頻量價特徵的並行壓縮器，而非直接輸出交易信號的端到端黑盒。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作（LSTM/Transformer） | RWKV-TS 設計 |
|---|---|---|
| 序列處理 | 原始逐點輸入 / 全局注意力 | 實例歸一化 + 分塊（Patching） |
| 狀態更新 | 遞歸門控 / O(L^2) Self-Attention | 可訓練WKV向量循環更新 |
| 推理模式 | 逐步遞推（誤差累積） / 雙向編碼 | 單向Encoder-only並行計算 |

⚡ **Eureka:** 用可訓練的Time/Channel WKV向量做循環狀態更新，把二次方注意力降維成線性卷積式遞推。
📥 **信息流:** `Input → Instance Norm → Patching → [Token Shift → WKV Op → Output Gating] × N → Channel Mix → Flatten → Linear Head → Pred`

## §2 · 數學層
📌 **Napkin Formula:** `Complexity: O(L) time & space | Loss: MSE`
直覺：WKV向量將時間依賴編碼為可訓練的循環權重，避免全局矩陣乘法；分塊將序列長度L壓縮為N個Patch，進一步壓降狀態維度。訓練僅依賴均方誤差回傳，無複雜正則或對比損失。

## §3 · 數據層
資料規模/頻率/市場/時段：8個真實世界基準數據集（氣象/交通/電力/ILI/ETT）用於長期預測；M4數據集用於短期；10個UEA數據集用於分類；5個數據集用於異常檢測；6個數據集用於插補。頻率/市場/時段未披露。怎麼來：公開學術基準。樣本外與容量假設：依賴實例歸一化對齊分佈，容量假設未披露。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| TBD | TBD | TBD | 中（需實現WKV算子與分塊邏輯） | 公開學術數據集 |

## §5 · 評測 / Benchmark
| 數據集/任務 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 長期預測 (8 datasets) | MSE (相對) | TimesNet | RWKV-TS | ↓12.58% |
| 長期預測 (8 datasets) | MAE (相對) | TimesNet | RWKV-TS | ↓4.38% |
| 少樣本預測 (10% data) | MSE (相對 vs CNN) | CNN基線 | RWKV-TS | ↓28.95% |
| 少樣本預測 (10% data) | MSE (相對 vs MLP) | MLP基線 | RWKV-TS | ↓7.92% |
| 時間序列分類 (10 UEA) | Accuracy | 73.60% | 73.10% | -0.50% |
| 異常檢測 (5 datasets) | F1 | 未披露 | 83.89% | 未披露 |
| 插補 (6 datasets) | Metric | TimesNet/PatchTST | RWKV-TS | 劣於前SOTA |

**解讀:** Δ 來自學術基準的相對誤差降低，屬模型表徵與計算效率能力；但無交易成本/滑點/實盤流動性計入，非交易α。實例歸一化可能引入分佈對齊偏差，少樣本優勢可能來自分塊帶來的隱式正則，而非真實外推力。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 單向RNN結構導致插補任務無法利用缺失值後方資訊，性能劣於雙向Transformer；實例歸一化可能抹除跨樣本絕對量級差異。
**6.2 推斷的隱含假設:** 數據分佈在實例層面可通過零均值單位方差對齊（假設非平穩性僅體現在局部尺度）；容量依賴於分塊長度與WKV維度，未驗證極端高頻下的狀態更新延遲；無數據泄漏/生存者偏差說明，學術基準通常含未來資訊或平滑處理。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| Transformer (Autoformer等) | O(L^2)注意力 vs O(L) WKV循環 | Open | 學術主流 |
| PatchTST | 分塊+Transformer vs 分塊+RWKV | Open | 學術主流 |
| DLinear | 線性通道不可知 vs 端到端非線性RNN | Open | 學術主流 |

🎤 **Interview Tip:** 
正確答：WKV算子將自注意力的全局矩陣乘法替換為線性循環狀態更新，實現O(L)複雜度與並行訓練，但犧牲了雙向全局路由能力。
錯答：RWKV-TS是Transformer的變體，只是把Attention換成了RNN。
**7.1 可證偽預測:** 2025-Q3前，若將RWKV-TS直接套用於A股日頻量價預測，其實例歸一化將導致跨板塊絕對價格資訊丟失，實盤表現劣於未歸一化的線性基線。

## §8 · For the Reader
- **因子研究員:** 用分塊+WKV做高維量價特徵的並行壓縮器，替代PCA/自編碼器，降低特徵工程維度災難。
- **高頻執行:** O(L)推理延遲適合邊緣節點實時特徵計算，但需驗證狀態緩存一致性與並行批處理的吞吐瓶頸。
- **組合配置:** 避免直接輸出權重，WKV的循環積累在震盪市易產生滯後偏差，建議僅作風險因子預警或波動率濾鏡。

## References
- 原論文：RWKV-TS（無 venue，2024-08-11）
- Lineage: RWKV (Peng et al.) → PatchTST → DLinear
- QuantML 導讀鏈接：[RWKV-TS: 超越传统RNN的时序模型](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247485732&idx=1&sn=5e362c713b852da5d597e0acb9e8c745&chksm=ce7e6e3af909e72c557fc9eb6aaf0f222329b727f1d3c28c2ebaf8f75b7fd17719dce26c2bae#rd)