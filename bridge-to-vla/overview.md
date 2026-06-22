# 跨冊接口 · 照見四冊

Trading-Handbook 是照見（Pulsar）系列第四冊，視角是 **Decision / 決策端**。它和另外三冊不是並列的選題，而是**同一套機器學習方法論在不同感知-決策回路上的投影**。這頁給跨冊讀者一張接線圖：量化里的某條技術線，在隔壁冊是什麼樣子。

| 冊 | 視角 | 與本冊共享的技術線 |
|---|---|---|
| [VLA-Handbook](https://github.com/sou350121/VLA-Handbook) | Action 端 | **強化學習 / Agent**：交易執行 RL 與機器人動作策略同屬「部分可觀測下的序貫決策」，共享 sim-to-real 落差、獎勵設計、容量/安全護欄 |
| [Spatial-Intelligence-Handbook](https://github.com/sou350121/Spatial-Intelligence-Handbook) | Perception 端 | **圖 / 關係結構**：本冊 `graph-networks` 的公司關係圖、動態另類數據圖，與空間冊的關係推理同根；都在問「結構從哪來、怎麼隨時間演化」 |
| [Physics-Controllable-Generation-Handbook](https://github.com/sou350121/Physics-Controllable-Generation-Handbook) | Generation 端 | **生成式 / 世界模型**：本冊 `data-generation-augmentation` 的市場模擬器、合成行情，與可控物理生成同屬「可控生成 + 分佈搬移」，共享過擬合與可信度評估難題 |

## 四條真正跨冊的暗線

- **序貫決策（RL）**：`reinforcement-learning` + `market-microstructure` 的執行/做市策略 ↔ VLA 的動作策略。共同病灶：sim-to-real、非平穩、延遲反饋。見本冊 [预测驱动 vs 策略驱动](/crossing/supervised-vs-rl/overview) 與 [人机协同 vs 全自动智能体](/crossing/human-in-loop-vs-autonomous-agent/overview)。
- **Agentic 工程**：`llm-agentic` 的多智能體交易框架 ↔ Agent-Playbook 的工具調用、規劃、護欄。本冊語料證實「智能體數量 ≠ 自主度」，這條教訓對所有 Agent 系統通用。
- **世界模型 / 可控生成**：市場模擬與合成數據 ↔ 物理可控生成。共享「生成得像 ≠ 對下游有用」的評估陷阱（見本冊 `evaluation-benchmarks` 的打假線）。
- **因果 / 結構**：`causal-structural` 的機制轉換定價、regime 識別，是四冊共同的底層語言——把關聯換成可干預的結構。

## 怎麼用這頁

如果你從 VLA / Spatial / Physics 冊過來：先讀本冊 [五軸本體論](/cheat-sheet/ontology)，再按上表的共享技術線跳到對應 zone。五軸里的「學習範式」「人機協作度」兩軸，在四冊之間語義一致，是跨冊對照的錨點。

> 接線圖會隨四冊各自成熟持續加密；本頁標 `Status: v0.5`。
