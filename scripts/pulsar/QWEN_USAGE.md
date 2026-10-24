# Trading 自動內容管線合約（2026-10 修復）

本次依維護者指示改用現有 DeepSeek 帳戶；不使用 GPT、Opus、qwen 或 OpenRouter 作為自動管線供應商。下方舊 qwen/Opus 名稱描述歷史分工；現役呼叫統一經 `_qwen.py` 相容入口轉到 `_deepseek.py`。

- 分類／寫作均用 `deepseek-flash`；可由明確設定改為 `deepseek-v4-pro`，不自動切供應商。
- API key 僅在主機讀取現有 `DEEPSEEK_API_KEY`；金鑰、原文快取不入 Git。
- 保留既有五軸、評級、分類、寫作模板；分類欄位不合法就拒收，不補猜測預設。
- 空、截斷或非 JSON 回應均算失敗；401 不重試；網路／429／5xx 只在單次期限內重試。
- 每次只跑一個管線，LLM workers=1；每日與週度使用共同互斥鎖。
- 日更保留原 cron 的1200秒期限：列舉60秒，取文以180秒硬牆、210秒子程序上限，分類120秒、去重30秒、寫作600秒，預留稽核／發布／索引時間。取文以單工輪轉：未試項優先、同條件新文優先；已失敗或中斷項移到後面。未完成項列deferred／failed，不能當無新文或成功；只發布原閘已通過的獨立產物。
- 各來源需完整抓取與成功分類；錯誤、未分類積壓、日期缺失不算「沒有新文」。來源失敗的頁面與 seen/state 一起回復，另留失敗收據。
- NBER 缺日期的 RSS 以其官方 `ref.tsv` 與 `date.tsv` 交叉核對；逐篇編號、題名、作者與現行 RSS 一致才採用官方 issue_date。兩週回補依完整 w-series 元資料，不以抓取時間、DOI 註冊時間或論文編號推算日期；只取得元資料不得稱讀過全文。
- 實踐者雷達的 partial 模式以「完整來源」為交易單位：某來源取得／分類失敗或達上限，就回復該來源全部新增 seen／項目；其餘已完整核驗來源可保留，頁首明示缺口，整體仍 partial／非零。未帶明確 partial 旗標或發生非預期例外時，仍回復整批。
- Validea 原 Blog 不可讀時保留失敗監控；由其主站明確連結的官方 Excess Returns RSS 可作獨立「訪談節目」補充來源，不宣稱取回 Blog。節目只收題名、發布日期、連結與題名層的一句話主題導覽；未讀逐字稿／未聽錄音不得宣稱已驗證訪談結論。
- Pass B 必須收到完整的已快取可讀文本，不再只取開頭8,000／11,000字；來源超過600,000 UTF-8 bytes即告警交人工處理，不默默截斷。原文來源需fulltext provenance；沒有讀到的圖表／附件不得說「論文未披露」。導讀與原文用不同來源標籤，metadata線索不能覆蓋正文。
- SIGTERM 取消須終止整個runner，落failed收據並退出143；partial只處理一般來源失敗，不得吞掉取消並繼續提交／推送。
- 原文HTML擷取第4版保留表格儲存格分隔，數學式取其alttext一次，並將TeX的字面百分號轉義還原為可見百分號，避免MathML與TeX註記重複黏合或百分比單位誤拒；數值與精度不改，舊HTML快取需先備份、依固定arXiv版本重取，不得靠放寬數字比對來接受黏合欄位。來源collector被取消也須回復seen與未發布頁，落failed／143。
- 新頁若未過品質閘，最多依原規則重寫一次；退稿與原因只留私有 `data/rejected`，兩次仍失敗就退出告警。
- 導讀日期與論文日期分開標示；未讀原文不得稱作者自述限制；條件式推斷不能改成跨所有券商／系統的必然結論；輔助輸入缺失不能等同生存者偏差。
- 新解構需來源、五軸與 §1–8；摘要不能標成取得全文。數字檢查拒絕沒有來源支持的百分比／小數；這項機械檢查不能證明指標語義完全正確。
- 既有人工與歷史文章不做自動內容重寫。待發草稿不進已發布本地索引。
- 既有結構稽核擴至 radar、guides 與根文件。既有警告不冒稱全部硬閘。
- `--allow-verified-partial` 需維護者明確批准啟用：只允許已過來源／內容／結構閘的獨立輸出發布；失敗來源交易回復，整體收據仍為 partial 並告警，不更新完整成功時間。
- 預設只準備草稿；`--publish` 才提交／推送。修補發布仍需維護者批准。
- 成功需品質稽核通過、推送回讀一致、本地 BM25 索引完整且對應已發布 commit。

---

## 歷史合約（供理解舊語料，不是現役路由設定）

# QWEN_USAGE.md — Opus 主導下「如何使用 qwen」的操作合約

> 原則由 Peter 於 2026-06-21 設定：**智能要求不高的大量任務交給 qwen；放 qwen 上規模之前，先由 Opus 4.8 ultrathink 把怎麼用 qwen 定死。** 本文件就是那份契約。改 prompt / schema / 路由前先改本文件。

## 0. 分工總則

| 角色 | 定位 | 負責 |
|---|---|---|
| **Opus（我）** | 架構師 / 大腦 | 本體論、prompt 模板、JSON schema、品質閘、跨篇綜合（Crossing）、誤分類裁決、docs.json/audit/腳手架 |
| **qwen**（qwen3.5-plus, DashScope CodingPlan, thinking on） | 主力勞動 | 399× 的逐篇蒸餾、分類打標、來源論文抽取、評級、套模板成 markdown、zone overview 初稿 |

鐵律：**Opus 先想清楚並凍結合約 → qwen 在合約內無人值守批量跑 → Opus 抽樣校準、裁決長尾。** qwen 永遠不准自創 zone slug；超出注入清單的一律歸 `unsorted` 交 Opus。

## 1. 任務路由表

| 任務 | Owner | 理由 |
|---|---|---|
| 本體論 / 五軸 / zone 劃分 | Opus（qwen 出草稿） | 手冊靈魂，高判斷 |
| prompt 模板 + JSON schema | Opus | 定義契約本身 |
| 逐篇蒸餾（399×） | **qwen** | 對乾淨輸入做摘要/抽取 |
| 逐篇分類 → zone + 五軸座標 | **qwen** | 在固定本體內打標 |
| 來源論文抽取（venue/arXiv/框架名） | **qwen** | 抽取 |
| 評級 ⚡/🔧/📖/❌ | **qwen** | rubric 驅動 |
| dissection markdown（被選中的篇） | **qwen** | 模板化寫作 |
| zone overview 綜述 | qwen 初稿 → Opus 改 | 輕綜合 |
| Crossing 張力圖（跨篇本質衝突） | **Opus** | 真綜合，非摘要 |
| 去重裁決 / 誤分類修正 | **Opus** | 判斷 |
| audit / docs.json / 腳手架 | **Opus** | 架構 |

## 2. 兩段式 qwen 管線

### Pass A — Router + Metadata（全 399，便宜，temp 0.1，JSON-out）
- **輸入**：`title` + `body`(截 ≤7000 字) + 注入的 taxonomy（zone slug 清單 + 五軸 poles）
- **輸出 schema**（嚴格 JSON）：
```json
{
  "zone": "<taxonomy 裡的 slug，否則 unsorted>",
  "axes": {"data":"...","horizon":"...","paradigm":"...","alpha":"...","autonomy":"..."},
  "rating": "⚡|🔧|📖|❌",
  "tldr": "≤60 字一句話",
  "key_trick": "≤80 字核心 trick",
  "source": {"venue":"ICLR26|KDD|JFE|null","arxiv":"NNNN.NNNNN|null","framework":"名字|null","confidence":"high|mid|low"},
  "tags": ["≤5 個"],
  "dup_of": "重複的 pos 或 null",
  "page_worthy": true
}
```
- **Opus 閘**：跑完先抽 20 篇人工校準 prompt → 凍結 → 再跑全 399。

### Pass B — Dissection 全頁（只給 `page_worthy && rating∈{⚡,🔧} && dup_of==null`，估 ~120-180 篇；其餘變成 zone overview 裡的 registry 一行）
- **輸入**：`body` + Pass A metadata + dissection 模板（量化版：§架構 / §數學 napkin / §數據 / §代碼 / §評測 / §失效與隱含假設 / §對比 / §For Reader）
- **輸出**：依模板的完整 markdown。temp 0.4。
- **不寫**：paper abstract 摘要、主觀讚美。要的是 **對比 + 失效模式 + 如何組合**（沿用 sister handbook AGENTS 反 anti-pattern 規則）。

## 3. 品質閘（機械，零 LLM）
- JSON schema 合法；`zone` ∈ taxonomy slugs；`axes` 各值 ∈ 允許 poles。
- body floor：<300 字標 `thin`，不進 Pass B。
- 去重：同 zone 內 title Jaccard ≥0.6 → flag 給 Opus 裁決（不自動刪）。
- commit 前過 `scripts/handbook_audit.py`（斷鏈/孤兒頁/nav 一致性）。

## 4. qwen 調用紀律
- 走 `_qwen.py`（DashScope coding endpoint，BAILIAN key，thinking on）。
- temp：分類 0.1 / 寫作 0.4。retry 4×（429/5xx，backoff 5*(i+1)s）。
- 並發 ≤4（host 3.5GB RAM 無 swap，序列或小並發）。
- 全程**可續傳**：每篇輸出落 `data/distill/<msgid>.json`，重跑跳過已完成。

## 5. Opus 留在環裡的「縫」
1. 定稿本體論（已：審 qwen 草稿 → `data/taxonomy.json`）。
2. Pass A 跑前抽 20 樣校準。
3. 定 Pass B 入選政策（page_worthy 標準）。
4. 手寫 Crossing atlases + cheat-sheet 五軸總圖。
5. 跑 audit、定 docs.json、修長尾。

## 6. 多源情報層的 qwen 偵察角色（2026-07-22 增）
- **頂刊雷達分類**（`journal_radar.py`）：12 本頂刊新文 → {relevance, zone, rating, 一句話}；temp 0.1、thinking off、json_mode、signal.alarm(85) 硬牆。分類失敗不記 seen → 下週自動重試。
- **GitHub 工程蒸餾**（`gh_quant_issues.py`）：每庫一批高互動 issues → 「坑→解法」JSON；**只依據 issue 原文，數字須逐字在文本中**；thinking on、alarm(170)。
- **正典一句話**（`build_seminal_canon.py`）：開山論文各 ≤40 字「開創了什麼」；引用計數一律 Crossref 機械統計，qwen 不碰數字。
- **OA 一手解構**：復用 Pass A/B（同 §2-4 紀律），origin=oa 由管線機械蓋章，qwen 無權決定來源標籤。

## Trading 專輯日期與有限 Fetch 備援契約（2026-10）

- 專輯 `create_time` 統一轉成上海（UTC+8）日曆日，與原頁 `ct` 相同；不採本機日期、搜尋日期或模型提供日期。既有正式索引日期的遷移須另列逐筆身分、原始時間、舊日期及新日期，交維護者批准後才套用；本候選不修改索引或歷史快取。
- 原公開 HTML 與既有代理均無可讀正文時，才可用現有 DeepSeek Fetch `POST /v1/tools/fetch`，只提交同一個微信官方 HTTPS 公開文章 URL。沿用 `_deepseek._key()`，不新增憑證、供應商或充值；Fetch 成功按次計費，不呼叫 Search、不拿摘要補正文、不生成 HTML 偽裝原頁。原 HTML 成功時保留既有正文、HTML SHA256、題名及日期核對；日期衝突直接 failed，不改用 Fetch。
- Fetch 前核對 `data/health/album.json`：本次列舉完整、狀態與 current/retained 數量一致、`index_sha256` 與完整索引一致、`missing_ids` 與 retained 區段一致。只有索引前 `current_count` 筆可授權；請求條目的 msgid、itemidx、pos、URL、官方題名、原始 create_time、上海日期須逐一相符。retained history、缺收據、雜湊不符或日期不一致均不可解鎖。
- Fetch 每篇每個 process 最多一次、process 總計最多20次，失敗也占嘗試額度；HTTP401/402/403 不重試。HTTP timeout 最多20秒，回應最多2,000,000 bytes；串流按剩餘時間讀取，日更 POSIX 單工路徑另有20秒整次硬牆，並保留較早到期的180秒批次硬牆及輪轉／deferred 保護，不延長既有日更期限。
- 只接受 JSON 物件中的字串 `title`、`content`；若返回 `url`，必須與請求一致。實際返回題名須與官方題名一致，正文須達既有300字下限且含足夠可見文字；空、過大、非 JSON、驗證頁、HTML 頁殼、thin 正文或身分不符均維持 failed，不寫快取。存檔前重核專輯收據與雜湊。操作日誌只留受控錯誤碼，不記金鑰、完整請求 URL 或原始回應。
- Markdown 快取保存來源 `url`、實際 `title`、官方 `wechat_title`、msgid/itemidx、原始 `create_time`、`official_album_index_sha256`、`body_sha256`，並明標 `date_source=official_album_create_time`、`source_format=markdown`、`fetch_provider=deepseek_fetch`。不填 `source_html_sha256`，不宣稱 Markdown 提供發布日期。Markdown 只代表公開文章正文抽取，不代表取得圖像、附件或原論文全文；下游品質、數字及語意標準保持不變，只準備未發布候選。
- 原頁或代理已識別的題名／日期須在正文長度及驗證頁判定前核對；衝突直接 failed，不得由後續代理或 Fetch 蓋過。Markdown 的300字門檻只計可見文字，扣除圖片（含替代文字）、連結目的 URL、引用定義及 HTML 標記／非顯示內容；此計數不改寫儲存正文或正文 SHA256。
