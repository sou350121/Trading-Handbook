# Maintainer

維護者：Pulsar / 照見。語料抓取見 `scripts/pulsar/`（enumerate→fetch→distill）。

## 現役管線契約

模型：現有 DeepSeek 帳戶的 `deepseek-flash`；不切換到 qwen、GPT 或 OpenRouter。歷史寫作規範與當前執行契約見 [管線合約](scripts/pulsar/QWEN_USAGE.md)。

每日與週度入口分別是 `daily_update.sh`、`radar_weekly.sh`。兩者經 `run_pipeline.py`，共用檔案鎖。原每日 21:00、週六 15:00 排程保留；修復發布需維護者批准。

1. 完整驗證來源；錯誤／截斷／缺日期不能當作沒有新文。
2. DeepSeek 分類與寫作，逐篇續跑；新頁必須通過來源、結構與數字支持檢查。
3. 執行完整 Handbook 結構稽核。既有文章正文不能混入自動改寫提交。
4. 維護者啟用發布後，提交允許的生成物、推送並核對遠端 commit。
5. 離線索引已提交內容，再檢查完整性與 commit 一致。

無 `--publish` 僅準備草稿。`--allow-verified-partial` 另需維護者批准；它允許已過各項檢查的獨立產物先發布，整體收據仍標 `partial`、退出非零並告警。來源失敗交易會把該來源的頁面和 seen/state 一起回復。不得把它改成完整成功。

## 伺服器檢查與補跑

`watch_pipeline.py` 預設僅檢查。`--apply` 可在 09:00、21:30 兩個預留時段補跑每日管線，每個應跑時段最多兩次。其他檢查時段只做檢查與離線索引修復；週度來源故障保留告警，避免任意啟動耗時內容工作並撞上別的排程。

本修復發布後新增以下 cron（保留舊 09:00 watchdog 入口）：

```cron
15,30,45 * * * * /usr/bin/python3.11 /home/claudeuser/trading-handbook-work/scripts/pulsar/watch_pipeline.py --apply --notify >>/tmp/trading-watchdog.log 2>&1
```

所有收據只在主機 `data/health/`。只有狀態、組件名稱與去重標記可進入 GitHub Issues；不送主機資訊、原始日誌、來源正文或憑證。未通過的草稿在私有 `data/rejected/`，不能上傳。

## 本地搜尋

```bash
python3.11 scripts/pulsar/local_index.py
python3.11 scripts/pulsar/local_index.py --check
python3.11 scripts/pulsar/local_index.py --query '交易成本 前瞻偏差'
```

沿用主機既有 `_local_memory_search.py` 的 BM25/CJK 引擎，無網路呼叫、模型下載或 API 付款。查詢的日期是倉庫快照日期，不是論文出版日期。索引與 manifest 不入 Git。

## 最小驗證

```bash
python3.11 -m unittest discover -s scripts/pulsar -p test_recovery.py
python3.11 scripts/handbook_audit.py
python3.11 scripts/pulsar/local_index.py --check
```

測試只模擬網路回應。實際來源與 DeepSeek 仍需獨立試跑；測試通過不得替代來源可用性或文章事實核查。

## 網站入口的邊界

GitHub `main` 是手冊正文。現有 GitHub Pages 從 `panel` 分支發布，內容為獨立 philab 哲學量化面板；該面板的儀器健康／裁決資料不是本手冊日更收據。本次手冊修復保留其分支與研究裁決，面板來源心跳缺失需在所屬研究系統追查。
