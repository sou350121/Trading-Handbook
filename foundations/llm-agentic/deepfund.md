<!-- ontology-5axis data=多模态 horizon=日频波段 paradigm=生成式大模型 alpha=组合执行优化 autonomy=Agent自主演进 -->

# DeepFund 解構

> **發布**：2025-10-28 · NeurIPS25
> **QuantML 導讀**：[NeurIPS 25 ｜ AI投资：是财富密码，还是回测陷阱？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492144&idx=1&sn=5f8cd500aef477194ef9912676a6400b&chksm=ce7d852ef90a0c383fa4798ef6b5ed71880d090bac15469c2ac08cb8477412b8e40ffc7dc138#rd)
> **核心定位**：五軸落點於「生成式大模型 × 組合執行優化」的評估基準層，解了歷史回測中訓練語料覆蓋測試期導致的「時間旅行」與信息泄露 prior gap。

**五軸座標**

| 數據模態 | 時間尺度 | 學習範式 | Alpha機制 | 人機協作 |
|:-:|:-:|:-:|:-:|:-:|
| `多模态` | `日频波段` | `生成式大模型` | `组合执行优化` | `Agent自主演进` |

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:** ① 構建無信息洩漏的實時正向測試基準 ② 多智能體工作流強制知識截止期後數據輸入 ③ 對「生成式大模型」軸★，揭示LLM在真實交易中的個性分化與淨虧損常態 ④ 僅1/9模型錄得+1.1%累積回報，多數淨虧損。

**X-Ray.** 該框架位於「評估基準」而非「Alpha生產」Pareto象限。它解了靜態回測的語料穿越坑，將驗證環境從離線歷史推演切換至動態API流。打不開的envelope在於未建模真實滑點、市場衝擊成本與高頻延遲，無法直接映射至實盤PnL。對量化讀者意義：提供LLM交易行為的壓力測試環境，但非部署藍圖；需將其實時信號流與傳統風險模型解耦後再組合，避免LLM「個性」與因子暴露疊加產生偽α。

## §1 · 架構 / Core Mechanism
| 改動維度 | 前作/傳統做法 | DeepFund 改動 |
|---|---|---|
| 驗證環境 | 歷史數據回測 | 實時正向測試 (Live Forward Testing) |
| 決策結構 | 單模型輸出 | 多智能體決策框架 (Multi-Agent Decision Framework) |
| 模型接入 | 固定/微調模型 | LLM工廠 (LLM Factory) 統一接口 |

⚡ **Eureka:** 強制知識截止日期後數據輸入，物理隔離訓練語料與實盤流，從源頭切斷「時間旅行」作弊路徑。
```
[Live Environment] --(股價/持倉/歷史)--> [API Gateway]
       |
       v
[Multi-Agent Workflow] --> [Financial Planner] --> [Analyst Team] --> [Portfolio Manager]
       |                      (策略大腦)          (技術/基本面/宏觀等)      (決策/風控/記憶)
       |
       v
[LLM Factory] --> [GPT-4.1 / DeepSeek-V3 / Grok 3 mini Beta ...]
```

## §2 · 數學層
📌 **Napkin Formula:**
信號有效性 = 4144 / 4320 ≈ 96%
決策有效性 = 1059 / 1080 ≈ 98%
複雜度: O(N_agents × T_days × LLM_inference_cost)

直覺: 非梯度優化問題，而是Prompt約束下的多智能體狀態機。系統不學習參數，僅在固定temperature下執行規則路由與信號聚合。
Loss/訓練: 無微調，依賴標準化Prompt與相同的溫度參數確保公平性。

## §3 · 數據層
資料規模/頻率/市場/時段: 美股Berkshire前五大(AAPL, AXP, BAC, KO, CVX)，日頻，2025-03-17至2025-04-17（24交易日）。
怎麼來: Alpha Vantage/Yahoo Finance API 接入股價、交易量、財務報表、新聞、宏觀指標。
樣本外與容量假設: 強制知識截止期後數據輸入；初始現金10萬美元，未計滑點與衝擊成本。

## §4 · 代碼層
| Repo | Checkpoint | License | 複現難度 | 數據可得性 |
|---|---|---|---|---|
| github.com/HKUSTDial/DeepFund | TBD | TBD | 中（依賴付費API與LLM調用） | 高（公開API） |

## §5 · 評測 / Benchmark
| 數據集/市場 | Metric | 前SOTA | 本方法 | Δ |
|---|---|---|---|---|
| 美股(Berkshire前五大) | CR | 買入並持有(未披露) | Grok 3 mini Beta: +1.1% | TBD |
| 美股(Berkshire前五大) | MDD | Grok 3 mini Beta: 5.5% | DeepSeek: 14.5% | 9.0% |

解讀: Δ 反映的是實盤環境下的行為分化，非模型絕對能力。多數淨虧損暗示當前LLM缺乏真正的風險定價能力；信號有效性(96%)與決策有效性(98%)的高比例不等於PnL轉化。部分Δ可能源於市場β（FOMC/關稅事件驅動）而非α，且未計交易成本，實盤淨值將進一步收斂。

## §6 · 失效與隱含假設
**6.1 論文自述 limitations:** 多數LLM錄得淨交易虧損，凸顯主動型基金管理局限性；模型在識別強烈市場反轉信號時存在共同短板。
**6.2 推斷的隱含假設:** Regime依賴（FOMC/關稅事件驅動波動）、容量限制（10萬美元模擬盤）、成本未計（僅100美元API費，忽略交易成本與滑點）、數據泄漏風險低但Prompt偏差存在、Survivorship（選Berkshire現持倉，非全市場）。

## §7 · 對比 & 面試 Tip
| 同軸對手 | 關鍵差異軸 | Open? | Status |
|---|---|---|---|
| 傳統回測框架/靜態金融QA | 實時正向測試 vs 歷史數據 | 是 | v0.5 |

🎤 **Interview Tip:** 
正確答: 「該框架是評估基準而非生產工具，核心價值在於隔離訓練語料泄露；實盤部署需將多智能體信號流與傳統風險模型解耦，並補償執行層延遲與滑點。」
錯答: 「DeepFund可以直接用於實盤交易，因為它生成了98%的有效決策且成本僅100美元。」

**7.1 可證偽預測帶日期:** 若2025-Q3實盤擴展至全市場股票池，LLM信號有效性將降至90%以下，且MDD分化將由政策事件驅動轉為流動性枯竭驅動。

## §8 · For the Reader
- **因子研究員:** 將其實時信號流與傳統風險模型解耦，避免LLM「個性」與因子暴露疊加產生偽α。
- **高頻執行:** 框架未建模滑點與衝擊成本，不可直接對接實盤路由；需補償執行層延遲與訂單簿深度假設。
- **組合配置:** 參考Grok的現金管理策略（初始40%投入，維持60%現金），在波動率 regime 切換時動態調整權重。
- **LLM-agent/RL策略:** 將多智能體工作流視為環境，用RL微調Prompt策略而非直接輸出交易指令。
- **研究學生:** 聚焦「信號有效性(96%)」與「決策有效性(98%)」的gap，分析LLM在整合多源數據時的權重分配偏差。

## References
- 原論文: NeurIPS 2025 (DeepFund)
- Lineage: LiveCodeBench (實時基準測試範式) / TAT-QA / FinanceBench (靜態金融QA)
- QuantML 導讀鏈接: [NeurIPS 25 ｜ AI投资：是财富密码，还是回测陷阱？](https://mp.weixin.qq.com/s?__biz=Mzg2MzAwNzM0NQ==&mid=2247492144&idx=1&sn=5f8cd500aef477194ef9912676a6400b&chksm=ce7d852ef90a0c383fa4798ef6b5ed71880d090bac15469c2ac08cb8477412b8e40ffc7dc138#rd)