# GitHub 工程經驗筆記（quant 開源棧）

> 論文不會告訴你的事：**11 個核心 quant 開源庫的高互動 issues**，自動掃描、qwen 蒸餾成
> 「坑 → 解法」，逐條帶原 issue 連結可追溯。每週增量追加（append-only），發現日期見各條。
> 涵蓋：數據坑 / 回測失真 / 執行 / 性能 / 部署 / API 坑。

回 [手冊首頁](../README.md) · 部署總覽 [deployment](overview.md)


## microsoft/qlib
_微軟量化平台（因子/模型/回測全棧）_
- **[#1166](https://github.com/microsoft/qlib/issues/1166)**（closed · 16💬 · API坑 · 2026-W30） — 坑：`np.loadtxt` 的 `delimiter` 參數升級後禁止使用換行符（`\n` 或 `\r`）導致 `TypeError`。解法：升級 qlib 至修復版本（PR 1141）。
- **[#535](https://github.com/microsoft/qlib/issues/535)**（closed · 16💬 · 數據 · 2026-W30） — 坑：`dump_bin.py` 僅支援 `numeric fields`（int/float），非數值欄位會導致 dump 失敗。解法：使用 `--exlcude_fields` 排除非數值欄位，或改用 `--include_fields` 僅指定數值欄位。

- **[#1547](https://github.com/microsoft/qlib/issues/1547)**（closed · 23💬 · 執行 · 2026-W30） — 坑：get_data 腳本硬編碼舊版 URL 導致 409 Public access error。解法：修改 qlib/tests/data.py 第 163 行 file_name = _get_file_name("latest")。
- **[#96](https://github.com/microsoft/qlib/issues/96)**（closed · 22💬 · 部署 · 2026-W30） — 坑：pip install 或 setup.py install 後報 ModuleNotFoundError: No module named 'qlib.config'。解法：在 clean env 重新安裝可解決，疑為 repo 或環境污染問題。
- **[#446](https://github.com/microsoft/qlib/issues/446)**（closed · 20💬 · 數據 · 2026-W30） — 坑：自訂 mini csv 數據跑 LightGBM 報 TypeError: Can only swap levels on a hierarchical axis。解法：df 為空因 instruments 未包含自訂股票，將 market 改為 all 或建立自訂 stock pool。
- **[#651](https://github.com/microsoft/qlib/issues/651)**（closed · 20💬 · 數據 · 2026-W30） — 坑：自匯數據缺 csi300.txt 報錯，collector 腳本因 CSI 官網頁面改版抓不到資料。解法：先用 get_data.py 下載 v2 內建 csi300/csi100，勿硬跑 collector。
- **[#91](https://github.com/microsoft/qlib/issues/91)**（closed · 19💬 · 其他 · 2026-W30） — 坑：用戶質疑 qlib 接口與 BigQuant 雷同恐侵權。結論：作者聲明全為自寫並採 MIT license，可商業使用，命名雷同屬量化領域常見慣例。
- **[#720](https://github.com/microsoft/qlib/issues/720)**（closed · 18💬 · 回測失真 · 2026-W30） — 坑：US 回測填 NASDAQ-100 報 ValueError does not exist。解法：benchmark 須填 ^gspc/^ndx/^dji，並對應 market=SP500/NASDAQ100。
- **[#526](https://github.com/microsoft/qlib/issues/526)**（closed · 17💬 · 數據 · 2026-W30） — 坑：改用 month 頻跑 Alpha158 報 TypeError: Can only swap levels on a hierarchical axis。解法：疑為 pandas 版本或 swaplevel 邏輯不兼容，需印出 df.index 除錯確認。
- **[#185](https://github.com/microsoft/qlib/issues/185)**（closed · 17💬 · 執行 · 2026-W30） — 坑：mlflow 太重導致 benchmark 跑不起。結論：qlib 已將 mlflow 解耦為 storage backend，用戶可自實作 expm.py 接口取代。

## AI4Finance-Foundation/FinRL
_RL 交易框架（學術系主流基線）_
- **[#1011](https://github.com/AI4Finance-Foundation/FinRL/issues/1011)**（open · 16💬 · API坑 · 2026-W30） — 坑：Gym/Gymnasium 升級後 env.reset() 與 env.step() 回傳 tuple 長度變為 5，導致 unpack 報錯。解法：將 env.reset()[np.newaxis, :] 改為 env.reset()[0][np.newaxis, :] 並同步適配 step() 回傳值。
- **[#962](https://github.com/AI4Finance-Foundation/FinRL/issues/962)**（open · 16💬 · 數據 · 2026-W30） — 坑：Yahoo downloader 對 PG、TRV、UNH、CRM、VZ、V、WBA、WMT、DIS、DOW 等符號全部回傳 No data found for this date range, symbol may be delisted。結論：資料源失效或符號被除牌，需確認日期範圍與替代資料源。

- **[#573](https://github.com/AI4Finance-Foundation/FinRL/issues/573)**（open · 28💬 · 數據 · 2026-W30） — 坑：Paper Trading Demo 僅支援 OHLCV 週期數據，無法處理 tick level 數據。結論：秒級/ tick 級數據伴隨高延遲與大量 I/O 運算開銷，且交易成本可能吞噬利潤，目前分鐘級數據被視為實時部署的可行粒度。
- **[#440](https://github.com/AI4Finance-Foundation/FinRL/issues/440)**（closed · 21💬 · API坑 · 2026-W30） — 坑：使用 Stable Baselines3 時，超參數優化結果每次運行皆不同，無法固定。解法：在 DRL_prediction 中設定 deterministic=True 參數，即可在測試或驗證階段獲得確定性結果。
- **[#339](https://github.com/AI4Finance-Foundation/FinRL/issues/339)**（closed · 19💬 · 數據 · 2026-W30） — 坑：自訂 CSV 數據的列數不一致且無 'day' 欄位。解法：數據不需等長，系統會依日期與字母順序排列並省略缺失日；若需合併可基於日期欄位進行 left join；'day' 非必須，可作為 categorical column 傳入。
- **[#915](https://github.com/AI4Finance-Foundation/FinRL/issues/915)**（open · 19💬 · 部署 · 2026-W30） — 坑：在 Python 3.10 環境執行 pip install finrl 失敗，因套件限制 Python 版本需 <3.9。解法：修改 pyproject.toml 將 python 版本限制改為 ^3.9，或使用 Anaconda 環境安裝。
- **[#1022](https://github.com/AI4Finance-Foundation/FinRL/issues/1022)**（closed · 18💬 · API坑 · 2026-W30） — 坑：升級至 stable-baselines3 2.0.0a10 後，StockTradingEnv.reset() 報錯 unexpected keyword argument 'seed'。解法：此為新版 API 變更導致，需降級至 v1.5.0 或等待套件更新以相容新 API。
- **[#514](https://github.com/AI4Finance-Foundation/FinRL/issues/514)**（closed · 17💬 · API坑 · 2026-W30） — 坑：運行 Demo_MultiCrypto_Trading 時出現 TypeError: 'module' object is not callable。結論：此為底層依賴庫更新導致，目前尚無明確解法，需等待官方更新修復。
- **[#336](https://github.com/AI4Finance-Foundation/FinRL/issues/336)**（closed · 17💬 · 回測失真 · 2026-W30） — 坑：提高 hmax 參數會導致回測錯誤。結論：hmax 代表最大可買賣股數，數值過高會導致現金不足（cash shortage），若設定 patient flag 為 true 則會持續等待現金，最終導致程序終止。
- **[#518](https://github.com/AI4Finance-Foundation/FinRL/issues/518)**（closed · 16💬 · API坑 · 2026-W30） — 坑：初始化 StockTradingEnv 時報錯 unexpected keyword argument 'initial_amount'。解法：API 已變更，'initial_amount' 改為 'initial_list'，且 'buy_cost_pct' 與 'sell_cost_pct' 需改為列表格式傳入。

## nautechsystems/nautilus_trader
_事件驅動生產級交易引擎_
- **[#190](https://github.com/nautechsystems/nautilus_trader/issues/190)**（closed · 25💬 · API坑 · 2026-W30） — 坑：ccxt.pro 部分交易所 data checksum 邏輯仍為 TODO 且 live trading 穩定性未驗證。結論：目前仍有些東西要 work out，尚未有明確的 live trading 成功報告。
- **[#333](https://github.com/nautechsystems/nautilus_trader/issues/333)**（closed · 25💬 · 性能 · 2026-W30） — 坑：用 trade history 回測時 DataFrame size 變 unmanageable。解法：改用 Arctic 或 parquet(file system) with s3 來處理 scalability and maintenance，避免直接用 Dask。

- **[#257](https://github.com/nautechsystems/nautilus_trader/issues/257)**（closed · 42💬 · 部署 · 2026-W30） — 坑→Windows 編譯 Cython 需 MSVC 14.0 且 `long` 型別跨平台長度不一致；解法→移除 `uint128`、將內部時間戳標準化為 `int64_t` nanoseconds、改用 `libc.stdlib` 的 `rand` 函數以避開 `long` 依賴。
- **[#532](https://github.com/nautechsystems/nautilus_trader/issues/532)**（closed · 36💬 · 數據 · 2026-W30） — 坑→回測文件過時且 API 頻繁變更（如 `import_from_data_loader` 移除、`process_files` 需 `block_parser`），導致 CSV 導入失敗；解法→改用 `develop` 分支並參考更新後的 Loading External Data 文件，自行實作 `parser` 函數。
- **[#39](https://github.com/nautechsystems/nautilus_trader/issues/39)**（closed · 34💬 · 執行 · 2026-W30） — 坑→單一 `DataClient` 耦合 `DataManager` 限制多交易所接入；解法→將 `DataClient` 與 `DataEngine` 解耦，支援多 `DataClient` 與 `ExecutionClient` 註冊，並引入 `asyncio` 處理網路 IO 以避免 GIL 瓶頸。
- **[#199](https://github.com/nautechsystems/nautilus_trader/issues/199)**（closed · 28💬 · 數據 · 2026-W30） — 坑→OrderBook 使用 `(Price, Quantity)` 元組有 ~0.5μs 建構開銷；解法→重構為 `(Decimal, Decimal)` 降低開銷，並計劃將 OrderBook 快照加入 tick stream 作為 `Data` 基類以支援回測排序。
- **[#179](https://github.com/nautechsystems/nautilus_trader/issues/179)**（closed · 28💬 · 性能 · 2026-W30） — 坑→重複回測需重新生成 QuoteTick/TradeTick，2M 筆資料耗時 7.48 sec；解法→使用 `SortedDict` 快取可降至 0.67 sec，但長週期（如 2 年）資料達 67GB 無法全放記憶體，需改用 `yield(chunk)` 分塊生產者。
- **[#2507](https://github.com/nautechsystems/nautilus_trader/issues/2507)**（closed · 27💬 · 回測失真 · 2026-W30） — 坑→Rust 實作指標缺乏 Python/Cython 的參數驗證（如 `Condition.positive_int`）且註釋邏輯矛盾（如 AMA 的 slow/fast 大小關係）；解法→以 Python 為基準逐項比對，補齊 Rust 的參數檢查與修正文件註釋。
- **[#3287](https://github.com/nautechsystems/nautilus_trader/issues/3287)**（closed · 27💬 · API坑 · 2026-W30） — 坑→Binance 於 2025-12-09 將 USDⓈ-M Futures 條件單遷移至 Algo Service，舊端點回傳 `-4120` 錯誤；解法→升級 SDK 改用新的 Algo Order API 端點提交訂單。
- **[#3561](https://github.com/nautechsystems/nautilus_trader/issues/3561)**（closed · 27💬 · API坑 · 2026-W30） — 坑→`nautilus-ibapi==10.37.2` 的 protobuf `Contract` 欄位名已改為 `primaryExch`，但 `client_utils.py` 仍寫 `primaryExchange` 導致 `AttributeError`，且新 API 要求訂單 ID 為 `int`；解法→升級至 TWS API 10.43 並修正欄位名稱與型別轉換。

## QuantConnect/Lean
_QuantConnect 開源引擎（實盤+回測）_
- **[#669](https://github.com/QuantConnect/Lean/issues/669)**（closed · 17💬 · API坑 · 2026-W30） — 坑：用戶強烈需求用 TradeBar 繪製 K 線圖但官方長期未支援。結論：該功能需求明確但開發優先級低，長期擱置未解決。
- **[#30](https://github.com/QuantConnect/Lean/issues/30)**（closed · 17💬 · 數據 · 2026-W30） — 坑：統計模組耦合且缺乏自訂基準。解法：重構為實例驅動並分離 Benchmark 介面；針對分倉/加倉場景，需明確定義 FIFO 或均價法則以確保損益計算準確。

- **[#166](https://github.com/QuantConnect/Lean/issues/166)**（closed · 70💬 · API坑 · 2026-W30） — 坑：Options 支援需跨層級改動（解析/儲存/物件/定價）。解法：Phase 1 先實作 Vanilla Black-Scholes 預設 implied vol 與 Greeks，並開放使用者替換模型。
- **[#452](https://github.com/QuantConnect/Lean/issues/452)**（closed · 43💬 · 部署 · 2026-W30） — 坑：直接遷移 .Net Core 風險高且函式庫不相容。解法：改對齊 .Net Standard 2.0 surface，利用 Portability Analyzer 驗證相容性，維持 Mono 跨平台現狀。
- **[#146](https://github.com/QuantConnect/Lean/issues/146)**（closed · 29💬 · 回測失真 · 2026-W30） — 坑：US 股票 T+3 結算期間資金不可用，直接扣 Cash 會導致 Equity 曲線跳動。解法：將 Pending Margin 獨立計算，僅從 MarginRemaining 扣除，維持 Cash 總額不變。
- **[#371](https://github.com/QuantConnect/Lean/issues/371)**（closed · 28💬 · 執行 · 2026-W30） — 坑：TDAmeritrade API 缺乏 make order 實作且文件封閉，需 $150k+ 門檻。結論：因存取限制與實作難度，暫緩整合，轉向評估 E-Trade。
- **[#525](https://github.com/QuantConnect/Lean/issues/525)**（closed · 27💬 · API坑 · 2026-W30） — 坑：Python 社群偏好 snake_case 與 C# PascalCase 衝突。解法：維持 API 命名一致以利學習，先完成文件再整合 pandas/numpy/scipy。
- **[#96](https://github.com/QuantConnect/Lean/issues/96)**（closed · 21💬 · 其他 · 2026-W30） — 坑：參數優化邏輯不應塞入 Engine 核心。解法：使用 C# Attributes 標記參數，由外部基礎設施（Cloud/Local）負責注入與搜尋最佳化。
- **[#2886](https://github.com/QuantConnect/Lean/issues/2886)**（closed · 18💬 · 數據 · 2026-W30） — 坑：IQFeed 下載器效能不佳。解法：已有基礎實作，需重構以大幅提升效能並提交至 Toolbox。

## kernc/backtesting.py
_輕量回測庫_
- **[#251](https://github.com/kernc/backtesting.py/issues/251)**（closed · 13💬 · 執行 · 2026-W30） — 預設市價單在下一根K線開盤價執行，無法精確觸及自訂線；解法：改用 limit 單指定開倉價，或設 tp/sl 指定平倉價，亦可開 trade_on_close=True 改在收盤價執行。
- **[#82](https://github.com/kernc/backtesting.py/issues/82)**（closed · 13💬 · 執行 · 2026-W30） — 設 tp/sl 後實際成交價可能偏離設定值；坑：若下根K線開盤價已穿過 tp/sl 閾值，系統會直接以該開盤價執行出場，導致滑點。

- **[#81](https://github.com/kernc/backtesting.py/issues/81)**（open · 24💬 · API坑 · 2026-W30） — 坑：框架原生不支援實盤交易。解法/結論：需另建獨立類別實作，共用 Strategy 並改寫 _Broker 與資料餵入邏輯，週期需與回測 bar 匹配。
- **[#649](https://github.com/kernc/backtesting.py/issues/649)**（closed · 22💬 · 回測失真 · 2026-W30） — 坑：15m 資料點過多觸發自動降頻時，resample 邏輯會引發長度不匹配錯誤。解法/結論：強制傳入字串降頻或設 resample=False 可避開，但會失去自動降頻功能。
- **[#20](https://github.com/kernc/backtesting.py/issues/20)**（open · 19💬 · API坑 · 2026-W30） — 坑：框架不支援多標的/多部位同時回測。解法/結論：無原生支援，需自行用 for 迴圈逐標的跑 Backtest.optimize，預設亦無最大持倉時間限制。
- **[#5](https://github.com/kernc/backtesting.py/issues/5)**（closed · 17💬 · 執行 · 2026-W30） — 坑：bt.optimize() 拋出 BrokenProcessPool。解法/結論：多為策略內未捕獲異常或記憶體耗盡導致子程序崩潰；需檢查策略邏輯與記憶體，並確保主程式有 __main__ guard。
- **[#987](https://github.com/kernc/backtesting.py/issues/987)**（closed · 16💬 · API坑 · 2026-W30） — 坑：bt.plot(resample='2H') 拋 TypeError。解法/結論：pandas 2.0+ 的 Index.get_loc 移除 method 參數導致相容性問題；降版至 1.4.4 或改用 get_indexer 可解。
- **[#51](https://github.com/kernc/backtesting.py/issues/51)**（closed · 15💬 · 性能 · 2026-W30） — 坑：optimize 子程序記憶體持續增長最終崩潰。解法/結論：ProcessPoolExecutor 與 numpy 已知記憶體洩漏；嘗試 bounded_pool_executor 或縮減參數空間，但無法完全根除。
- **[#90](https://github.com/kernc/backtesting.py/issues/90)**（closed · 14💬 · API坑 · 2026-W30） — 坑：將 Indicator 存入 dict 會導致無交易。解法/結論：Backtest 僅掃描 strategy 實例屬性；需用 setattr(self, name, self.I(...)) 掛載，再同步存入 dict 供後續使用。
- **[#28](https://github.com/kernc/backtesting.py/issues/28)**（closed · 13💬 · 回測失真 · 2026-W30） — 坑：同一步驟觸發新單與停損會導致雙重部位與資料錯位。解法/結論：框架設計允許同 K 線內成交價與停損價同時觸發，需接受此行為或自行調整訂單觸發順序。

## pst-group/pysystemtrade
_Rob Carver 系統化期貨（實盤跑真錢）_
- **[#495](https://github.com/pst-group/pysystemtrade/issues/495)**（closed · 17💬 · 回測失真 · 2026-W30） — 坑：報告生成時 Config.fill_with_defaults 被多次呼叫，導致 diagnostic/reporting 的 config 與 backtest 的 config 混淆，notional_trading_capital 錯誤回退至 defaults.yaml。解法：在 run_system_classic.py 加入 is not None 條件判斷，強制用傳入參數覆蓋 config 的 capital 與 base_currency。
- **[#241](https://github.com/pst-group/pysystemtrade/issues/241)**（closed · 16💬 · API坑 · 2026-W30） — 坑：執行 update_fx_prices 時 ib_insync 層報錯 'Couldn't evaluate ibFxPricesData'，常因 IB gateway 未啟動或連線參數錯誤。解法：確認 gateway 狀態、指定 account 參數連線，並升級 ib_insync 與 IB gateway 版本以解決間歇性連線延遲。

- **[#274](https://github.com/pst-group/pysystemtrade/issues/274)**（closed · 39💬 · 數據 · 2026-W30） — 坑：數據存在大段缺失或手動更新遺漏導致 assertion 失敗與 sharding 警告。解法：手動修正缺失數據與 roll，並確保 capital 與 config 同步。
- **[#267](https://github.com/pst-group/pysystemtrade/issues/267)**（closed · 29💬 · 數據 · 2026-W30） — 坑：缺少 multiple price 數據時 diagnostics 嘗試讀取標籤導致 crash。解法：在無 multiple price 時移除標籤機制，並手動建立 roll calendar。
- **[#284](https://github.com/pst-group/pysystemtrade/issues/284)**（closed · 25💬 · 執行 · 2026-W30） — 坑：冷啟動手動執行時，手動填入訂單後未正確向上傳遞，導致 contract level 無部位。解法：嚴格依序執行 fill、pass fills upwards 與 end of day 流程。
- **[#452](https://github.com/pst-group/pysystemtrade/issues/452)**（closed · 25💬 · API坑 · 2026-W30） — 坑：IB API 偶發回傳非預期佣金物件格式導致 stack_handler 崩潰。解法：重啟系統，並用 interactive order stack 檢查 raw orders 以確認 API 回傳結構。
- **[#313](https://github.com/pst-group/pysystemtrade/issues/313)**（closed · 22💬 · 部署 · 2026-W30） — 坑：volatility_calculation 模組路徑異動導致 Exception。解法：刪除 yaml 中的 volatility_calculation 設定以回退至 defaults.yaml，或更新 func 路徑。
- **[#400](https://github.com/pst-group/pysystemtrade/issues/400)**（closed · 21💬 · 部署 · 2026-W30） — 坑：路徑字串拼接缺少分隔符號導致 os.rename 找不到路徑。解法：改用標準檔案管理函數拼接路徑，並支援 dot format 與跨 OS 路徑。
- **[#1085](https://github.com/pst-group/pysystemtrade/issues/1085)**（closed · 20💬 · API坑 · 2026-W30） — 坑：IB 合約找不到時，錯誤處理邏輯會再次請求合約詳情引發 event loop 衝突。解法：非技術性錯誤，直接從現有合約物件或 config 取得資訊以避免重複請求。
- **[#1578](https://github.com/pst-group/pysystemtrade/issues/1578)**（closed · 19💬 · 回測失真 · 2026-W30） — 坑：動態優化會讀取已移除合約的舊權重，導致理想系統計算失真。解法：系統 runner 應更新所有 raw optimal positions，將已移除合約的權重歸零。

## freqtrade/freqtrade
_加密貨幣自動交易（最大社區）_
- **[#6710](https://github.com/freqtrade/freqtrade/issues/6710)**（closed · 44💬 · 部署 · 2026-W30） — 坑：Windows 使用 Docker 部署時 Web UI 無法顯示。解法：需補充更多環境資訊與截圖才能定位問題。
- **[#8998](https://github.com/freqtrade/freqtrade/issues/8998)**（closed · 43💬 · 執行 · 2026-W30） — 坑：`adjust_trade_position()` 計算 `max_stake` 時，若無交易觸發，錢包餘額更新延遲（最長 1 小時）或受 `available_capital` 限制，導致誤報餘額不足。解法：建議在計算前呼叫 `self.wallets.update()` 強制刷新。

- **[#1371](https://github.com/freqtrade/freqtrade/issues/1371)**（closed · 64💬 · 執行 · 2026-W30） — 坑：Bot 因 amount 精度計算與交易所不一致導致 Insufficient funds 無法賣出。→ 結論：需嚴格對齊交易所精度，並確認 CCXT 與 Bot 的 rounding/truncating 邏輯差異。
- **[#11008](https://github.com/freqtrade/freqtrade/issues/11008)**（closed · 57💬 · 數據 · 2026-W30） — 坑：Live 模式缺乏 orderflow 數據且下載耗時長。→ 結論：Bot 會自動下載 max_candles 數據，測試時可縮小 max_candles 與 cache_size 以加速啟動。
- **[#892](https://github.com/freqtrade/freqtrade/issues/892)**（closed · 47💬 · 其他 · 2026-W30） — 坑：缺乏做空功能限制策略在熊市表現。→ 結論：該功能依賴 CCXT 底層支援，需等待上游 issue 解決。
- **[#4653](https://github.com/freqtrade/freqtrade/issues/4653)**（closed · 47💬 · 部署 · 2026-W30） — 坑：Web UI 與主程式同時存取 SQLite 導致 database is locked 崩潰。→ 結論：SQLite 存在檔案層級鎖定問題，需確保交易與查詢的 transactional 隔離。
- **[#1853](https://github.com/freqtrade/freqtrade/issues/1853)**（closed · 45💬 · 部署 · 2026-W30） — 坑：Hyperopt 因 config JSON 語法錯誤或舊 pickle 檔報錯。→ 結論：修正 JSON 逗號並刪除 user_data/hyperopt_results.pickle 即可。
- **[#2509](https://github.com/freqtrade/freqtrade/issues/2509)**（closed · 45💬 · 其他 · 2026-W30） — 坑：ta-lib 底層 C 庫過舊且難以替換。→ 結論：嘗試用 finta 替換時發現計算邏輯不同導致測試失敗，Python 生態缺乏成熟替代方案。
- **[#9163](https://github.com/freqtrade/freqtrade/issues/9163)**（closed · 45💬 · 其他 · 2026-W30） — 坑：Telegram 顯示的損益數據與實際不符。→ 結論：升級至最新 dev 版本，開發者已修正 partial exit 後的 avg profit 顯示邏輯。

## hummingbot/hummingbot
_做市機器人（execution/microstructure）_
- **[#3909](https://github.com/hummingbot/hummingbot/issues/3909)**（closed · 27💬 · API坑 · 2026-W30） — 坑：交易所對特定交易對有最小交易量精度限制（如 CRO-USDT 不支援小數），Bot 發送帶小數的訂單會被 API 強制取整為整數，導致所需資金超出帳戶餘額而觸發 insufficient balance 錯誤。解法：開發 Connector 時必須嚴格對接交易所的 lot size 規則，在發送訂單前將數量精度截斷或取整，避免精度不符引發的執行失敗。
- **[#6897](https://github.com/hummingbot/hummingbot/issues/6897)**（closed · 27💬 · 其他 · 2026-W30） — 坑：交易所協議大版本升級（如 dYdX v4）會導致舊版 Connector 失效，需投入大量工程資源重構以符合新架構與官方 Checklist。解法：透過 Bounty 機制公開招募開發者，並強制要求遵循 Contribution Guidelines 與 Checklist 進行開發與審查，確保升級後的 Connector 品質與合規性。

- **[#6585](https://github.com/hummingbot/hummingbot/issues/6585)**（closed · 55💬 · API坑 · 2026-W30） — 交易所強制停用舊版 API 會導致連接器直接失效，需依賴社區 Bounty 機制集資推動升級至新版 API。
- **[#7207](https://github.com/hummingbot/hummingbot/issues/7207)**（closed · 52💬 · API坑 · 2026-W30） — WebSocket 消息解析時發生 TypeError 會導致訂單狀態無法追蹤，需修復 JSON 解析邏輯以正確處理 WebSocketError。
- **[#8046](https://github.com/hummingbot/hummingbot/issues/8046)**（closed · 44💬 · API坑 · 2026-W30） — 新增交易所連接器需嚴格遵循官方架構標準並補齊單元與整合測試，以確保開源代碼的穩定性與可維護性。
- **[#6395](https://github.com/hummingbot/hummingbot/issues/6395)**（closed · 42💬 · API坑 · 2026-W30） — CEX 與 DEX 錢包/鏈組合的餘額數據結構不同，混用同一指令會造成混淆，需拆分為獨立指令並支援顯示代幣額度。
- **[#5076](https://github.com/hummingbot/hummingbot/issues/5076)**（closed · 37💬 · 執行 · 2026-W30） — 長時間運行時若遺漏訂單取消完成事件，機器人會陷入無限重試循環，需增強狀態恢復機制以處理缺失事件。
- **[#7125](https://github.com/hummingbot/hummingbot/issues/7125)**（closed · 34💬 · API坑 · 2026-W30） — 社區開發新連接器需先將全額 Bounty 資金打入基金會錢包完成資助流程，方可正式啟動開發任務。
- **[#7810](https://github.com/hummingbot/hummingbot/issues/7810)**（closed · 33💬 · API坑 · 2026-W30） — 部分交易所採用 EVM 風格的 EIP-712 簽名而非傳統 HMAC，連接器開發需特別實作對應的結構化數據簽名邏輯。
- **[#6024](https://github.com/hummingbot/hummingbot/issues/6024)**（closed · 29💬 · API坑 · 2026-W30） — 交易所 API 大改版會導致舊連接器失效，開發者通常會選擇等待新 API 穩定後再進行重構，以避免初期頻繁變更。

## stefan-jansen/machine-learning-for-trading
_ML4T 書配套代碼（學習者踩坑集中地）_
- **[#130](https://github.com/stefan-jansen/machine-learning-for-trading/issues/130)**（closed · 8💬 · 部署 · 2026-W30） — 坑：Python 3.8 建立的 .h5 檔無法在 Python 3.5.6 環境開啟→解法：升級 Zipline 至 3.8 以統一環境，或重裝環境。
- **[#40](https://github.com/stefan-jansen/machine-learning-for-trading/issues/40)**（closed · 8💬 · 部署 · 2026-W30） — 坑：Windows 下 docker run -v $(pwd):... 報錯找不到檔案→解法：將 $(pwd) 替換為 repo 的絕對路徑，並確認 Docker 已授權存取該磁碟。

- **[#113](https://github.com/stefan-jansen/machine-learning-for-trading/issues/113)**（closed · 17💬 · 部署 · 2026-W30） — Pickle protocol 5 不相容與 KeyError 坑→解法：確保讀寫 notebook 使用相同 Docker 環境與 Python 版本以維持一致性。
- **[#103](https://github.com/stefan-jansen/machine-learning-for-trading/issues/103)**（closed · 15💬 · 數據 · 2026-W30） — Pandas 無法解析自訂日期字串進行 split 坑→解法：確認 index 類型非 string，需轉換為 datetime 物件。
- **[#55](https://github.com/stefan-jansen/machine-learning-for-trading/issues/55)**（closed · 11💬 · 部署 · 2026-W30） — GCP/Mac 執行 zipline ingest 出現 Permission denied 坑→解法：需手動設定目錄寫入權限或將 user 加入 docker group。
- **[#21](https://github.com/stefan-jansen/machine-learning-for-trading/issues/21)**（closed · 10💬 · 回測失真 · 2026-W30） — Zipline 回測找不到 US 資產導致 ValueError 坑→解法：用 sqlite client 開啟 assets db，將 exchanges 表中的 country_code 欄位 Null 值手動改為 'US'。
- **[#64](https://github.com/stefan-jansen/machine-learning-for-trading/issues/64)**（closed · 10💬 · 性能 · 2026-W30） — Storage benchmark 讀取 parquet 導致 kernel crash 坑→解法：減少 generate_test_data 的 nrows 與 numerical_cols 參數以降低記憶體佔用，避免 Docker VM 資源上限。
- **[#8](https://github.com/stefan-jansen/machine-learning-for-trading/issues/8)**（closed · 9💬 · 其他 · 2026-W30） — 書籍翻譯版發現公式遺漏、圖片張冠李戴與章節編號錯誤坑→現狀：編輯失誤，需等待線上版更新修正。
- **[#100](https://github.com/stefan-jansen/machine-learning-for-trading/issues/100)**（closed · 9💬 · API坑 · 2026-W30） — Pandas 讀取 HDFStore 報錯無法設定 WRITEABLE flag 坑→解法：此為 NumPy 1.16.0 版本 bug，需更新環境至 1.18.5 或執行 conda update --all。
- **[#49](https://github.com/stefan-jansen/machine-learning-for-trading/issues/49)**（closed · 8💬 · 執行 · 2026-W30） — Notebook 執行 Variable not defined 與路徑錯誤坑→解法：需將讀取 predictions.h5 的 cell 上移，並將路徑中的 'train' 改為 'test'。

## OpenBB-finance/OpenBB
_開源投研終端（數據工程的坑）_
- **[#6706](https://github.com/OpenBB-finance/OpenBB/issues/6706)**（closed · 32💬 · 其他 · 2026-W30） — 坑：社群貢獻者易重複搶佔同一 issue。解法：引入 oss-gg bot 強制 48h 內提交 draft PR，否則自動 unassign 釋放任務。
- **[#6713](https://github.com/OpenBB-finance/OpenBB/issues/6713)**（closed · 26💬 · 其他 · 2026-W30） — 坑：貢獻者逾期未交 PR 阻塞進度。解法：oss-gg bot 在 48h 基礎上追加 12h 倒數提醒，未提交即自動 unassign 以維持交付節奏。

- **[#4872](https://github.com/OpenBB-finance/OpenBB/issues/4872)**（closed · 44💬 · 部署 · 2026-W30） — 坑→Ubuntu Docker 跑 X11 顯示圖表變白視窗；解法→確認 compose 檔名是 `docker-compose.x11.yaml` 而非 `docker-compose-X11.yaml`，並掛載 `/tmp/.X11-unix` 與設定 `DISPLAY` 環境變數。

## polakowo/vectorbt
_向量化回測（速度取向）_
- **[#181](https://github.com/polakowo/vectorbt/issues/181)**（closed · 13💬 · 回測失真 · 2026-W31） — 坑：`generate_ohlc_stop_exits` 觸發停損/停利後，若未傳入 `price` 參數，回測會用 `Close` 價格結算，導致實際虧損大於設定的 `sl_stop`。解法：將 `generate_ohlc_stop_exits` 輸出的 `out_dict` 中的停損/停利價格作為 `price` 參數傳入 `Portfolio.from_signals`，確保觸發時以正確價格執行。
- **[#11](https://github.com/polakowo/vectorbt/issues/11)**（closed · 12💬 · 其他 · 2026-W31） — 坑：目前版本不支援做空（shorting）功能。結論：開發者確認此為未來功能，但當前優先級較低，正專注於撰寫專案文件與 API 參考。

- **[#113](https://github.com/polakowo/vectorbt/issues/113)**（closed · 25💬 · 回測失真 · 2026-W30） — 坑：Stop loss 需要乾淨的訊號，若訊號相鄰會導致 Portfolio.stats 與 Dashboard 數據不一致。解法：生成 Stop loss 前，先對 entries 與 exits 執行 `.vbt.signals.first(reset_by=..., allow_gaps=True)` 清理訊號。
- **[#17](https://github.com/polakowo/vectorbt/issues/17)**（closed · 23💬 · API坑 · 2026-W30） — 坑：`vbt.MA.from_combinations` 傳入整數陣列會引發 Numba TypeError。解法：將 `windows` 陣列轉型為 `np.float64`，或升級至 0.6 版本以支援任意整數類型。
- **[#31](https://github.com/polakowo/vectorbt/issues/31)**（closed · 17💬 · 回測失真 · 2026-W30） — 坑：`SizeType='TargetPercent'` 時，若 size 為 0 會導致部位無法平倉、進出場日期顯示錯誤。解法：將 size 的 0 替換為 `-np.inf`，或移除原始碼特定行並發 PR。
- **[#111](https://github.com/polakowo/vectorbt/issues/111)**（closed · 16💬 · 其他 · 2026-W30） — 坑：使用者想將隨機策略結果持久化。解法：利用 `Portfolio` 物件的 `config` 屬性存取初始化參數，透過 `Portfolio(**config)` 重建並儲存策略。
- **[#131](https://github.com/polakowo/vectorbt/issues/131)**（closed · 15💬 · API坑 · 2026-W30） — 坑：`direction='shortonly'` 搭配 `size_type='percent'` 會導致 share_flow 全為 0。解法：改用 `Portfolio.from_orders` 搭配 `size_type='targetpercent'`，將 size 設為目標百分比（如 0.01）與 0 來控制開平倉。
- **[#77](https://github.com/polakowo/vectorbt/issues/77)**（closed · 15💬 · 性能 · 2026-W30） — 坑：多資產回測時呼叫 `portfolio.total_return()` 觸發 Numba Segfault。解法：此為記憶體不足所致，需減少超參數數量或將模擬分塊（chunks）執行。
- **[#156](https://github.com/polakowo/vectorbt/issues/156)**（open · 15💬 · 執行 · 2026-W30） — 坑：需限制特定時段進場但允許任意時段出場。解法：建立布林值 `activity_mask` 序列，在允許時段設為 True，不允許時段設為 False 傳入 `from_order_func` 過濾訊號。
- **[#342](https://github.com/polakowo/vectorbt/issues/342)**（closed · 13💬 · 部署 · 2026-W30） — 坑：股票回測出現指標斷層或需 dropna。解法：設定 `vbt.settings['array_wrapper']['freq']` 與 `year_freq` 處理頻率，並在 `Portfolio.from_signals` 傳入 `size_granularity=1` 確保整數部位。
