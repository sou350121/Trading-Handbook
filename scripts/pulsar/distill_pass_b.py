#!/usr/bin/env python3
"""Pass B — qwen writes a full dissection page per AGENTS.md template (quant version).
Only for page_worthy ⚡/🔧 non-dup articles. Resumable. Writes foundations/<zone>/<slug>.md.

Usage: python3 distill_pass_b.py [--only-pos N] [--limit N] [--workers W]
"""
import json, pathlib, sys, argparse, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _qwen import ask
from _content_gate import validate_page, HTML_TEXT_VERSION

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"data"/"raw"; DIS = ROOT/"data"/"distill"
AXES_NAME = {"data":"數據模態","horizon":"時間尺度","paradigm":"學習範式","alpha":"Alpha生成機制","autonomy":"人機協作度"}

SYSTEM = ("你是『照見 Trading-Handbook』的資深解構作者。讀者是做過實盤/回測的量化研究員。"
          "你寫的不是論文摘要，而是【對比 + 失效模式 + 如何組合】的解構。中文輸出，繁體可，術語/框架名/指標保留英文。")

MAX_SOURCE_BYTES = 600_000

TEMPLATE = """請依下面骨架，把這篇方法寫成一頁解構 markdown。**不要寫成摘要**，要分析性論斷。
未知的數字/細節標 `TBD` 或 `（未驗證）`，不要編造。直接輸出 markdown，不要代碼圍欄。**只輸出最終頁面內容;絕不輸出你的思考過程、對骨架/規則的旁白或「此處依骨架要求…」之類的元敘述。**
【來源覆蓋】{source_scope}
只能根據本次提供的可讀文本下結論；不可把未提取的圖片／附件、未核對的原論文，或模型沒看見的內容說成「論文未披露」。缺項寫「所讀文本未見，圖表／附件待核」。metadata的TL;DR、框架名是待核線索，正文優先。
預訓練、tokenizer、適配、topic擬合各自的時間邊界須分列，不得把預訓練截止日套到適配後權重。不同模型、選模結果與集成結果不可互相代稱。單一樣本的相關排序或消融不得寫成所有模型／市場的必然因果結論；本頁研究假說須標示尚待驗證。
【硬規則 · 數字紀律】（違反任何一條都算嚴重錯誤）
1. 全頁任何數字（表格**與**解讀散文**都算**）必須是導讀正文的**逐字子字串**，或同一列 Δ 欄中兩個逐字數字的算術差。導讀沒有的一律寫「未披露」，**禁止臆造任何數值、百分比或區間**（絕不寫「~1.2-1.8」「+40%」這類猜測值）。寧可整列空白也不要編數字。
2. **禁止合成「前SOTA」**：若導讀只給了幾個不同基線模型各自的數字，就逐一列出（「基線A 54.61% / 基線B 50.23% / 本法 56.87%，領先約 2-6pp」），**絕不**把它們 min/max 成一個假的 SOTA 區間再算假 Δ。
3. **禁止杜撰樣本量、breakeven 成本、失效閾值**（如「樣本僅 4.8k」「成本超 5bp 失效」），除非該數字逐字出現在導讀。
4. **指標標籤不可混淆**：若導讀對「回撤深度」與「回撤持續期占比」等兩個不同量用了相近詞，拆成兩列、各自標清楚；**不可**塞進一列再補一個缺失值。
5. **保留導讀的原始精度**：正文寫「7%」就寫「7%」，不要補成「7.00%」假裝有更高精度。
6. 第一行的 `<!-- ontology-5axis ... -->` 註解**必填**，依五軸逐字輸出，置於 H1 之前。
7. **空白/缺失值不得補預設**：若導讀提到某指標/超參/解析度/基線「名稱」但**數值缺失**（空括號「（）」、空白、或被剝掉的圖片/公式留白），一律寫「未披露」。**絕不**用慣例預設值（影像 224×224、RL 折扣 0.99、動量 0.9 之類）填補，也**不內插**空白的基線格。
8. **禁止用日期反算時長**：別把起訖日期算成「X 年/月」當來源數字（且常算錯/重複），直接照抄日期區間。
9. **§5 欄位語義固定**：前SOTA 欄填**基線本身的數值**（逐字），本方法欄填本法數值；**絕不**只放模型名而不放其數值，**絕不**把 Δ 塞進前SOTA/本方法格（Δ 只進 Δ 欄）；百分點差用 `pp`（如 30.5% − (−5.09%) = +35.59pp）。
10. 數字以完整 token 逐字核對，包含正負號與小數精度，不是任意子字串。計算的 Δ 只能放在固定五欄表「數據集/市場 | Metric | 前SOTA | 本方法 | Δ」的末欄，且前SOTA、本方法各只含一個來源數值、單位相同；其他位置不要新增計算值。不符合此格式就省略 Δ，保留來源數值及語義說明。

【計算與表述澄清】
- Δ 固定定義為「本方法 − 前SOTA」。誤差／loss 越小越好時仍使用此方向，不把反向的「改善量」當 Δ。計算欄不能通過原有來源／單位規則時，省略該計算值，保留兩個來源原值與正確語義；不得在散文另增來源沒有的計算差。
- 1.1／1.2／1.3 等小節標記使用獨立 Markdown ### 標題，只寫小節名稱。實證數值放正文或表格，不得藏在標題以逃避核驗。
- 有限實驗中的失敗不等於普遍不可能；尚未測試不等於已證明無效。比較同時改變多個矩或參數時，不得把全部差異歸因於其中一項；若要隔離單一項，必須固定其他項，並明示玩具例子的假設。

骨架（嚴格按此順序與標題）：

<!-- ontology-5axis data={data} horizon={horizon} paradigm={paradigm} alpha={alpha} autonomy={autonomy} -->

# {name} 解構（{fw_en}）

> **{date_label}**：{date} · {venue}{arxiv_line}
> **{src_label}**：[{wechat_title}]({url}){prov}
> **核心定位**：（1-2 句：五軸落點 + 解了什麼 prior gap）

**Status:** v0.5 — 基於{src_label}（有原文則以原文為準）。細節待升 v1。
**TL;DR:**（≤4 句：①做什麼新事 ②核心 trick ③為什麼這對某條軸★ ④一個關鍵實證數字**——若來源沒有量化結果，寫「來源未給量化結果」，不要編數字**）

**X-Ray.**（150-300 字分析論斷：放回五軸 Pareto、解了哪些舊工程坑、預測它打不開的 envelope、對量化讀者意義。不是 summary）

## §1 · 架構 / Core Mechanism
（1.1 三大改動 vs 前作，用表；1.2 ⚡ Eureka 一句話 trick + 直覺；1.3 信息流 ASCII 圖）

## §2 · 數學層
（📌 Napkin Formula：1-3 行最關鍵 equation + 複雜度；直覺 2-3 句；loss/訓練細節）

## §2.5 · 帶數字走一遍（Worked Example）
（用**明確標「假設/示意」的玩具數字**把核心機制手算一遍：給一組假想輸入 → 中間量怎麼算 → 輸出什麼，讓讀者「照著算一遍就懂」。3-6 步。**這裡的數字是示意用途、不是論文實證結果**；論文真實數字一律留在 §5，缺的寫「未披露」。）

## §3 · 數據層
（資料規模/頻率/市場/時段、怎麼來、樣本外與容量假設）

## §4 · 代碼層
（表：Repo / Checkpoint / License / 複現難度 / 數據可得性。未知填 TBD）

## §5 · 評測 / Benchmark
（表：數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前SOTA | 本方法 | Δ；**前SOTA 欄只填導讀逐字給的單一基線值，若有多個基線就改成逐行列出、別合成區間，沒有就寫「未披露」**；解讀哪部分 Δ 是真 capability，哪部分可能是過擬合/前瞻偏差/成本未計。**解讀散文裡也不得出現導讀沒有的數字**）

## §6 · 失效與隱含假設
（6.1 論文自述 limitations：{limitations_rule}；6.2 推斷的隱含假設（明標推斷，限制適用條件，禁止跨所有系統的必然斷言，輔助數據缺失不是生存者偏差）：regime 依賴/容量/成本/數據泄漏/survivorship）

## §7 · 對比 & 面試 Tip
（表：同軸對手 | 關鍵差異軸 | Open? | Status；🎤 Interview Tip 正確答 vs 錯答；7.1 可證偽預測帶日期）

## §8 · For the Reader
（按 persona 分流，至少 3 條：因子研究員/高頻執行/組合配置/LLM-agent/RL 策略/研究學生）

## References
（列原論文 + lineage + 來源鏈接）

---

可用素材：
- 方法/框架名：{fw}
- 五軸座標：{axes_human}
- TL;DR 線索：{tldr}
- 核心 trick 線索：{trick}
- 來源：venue={venue} arxiv={arxiv} framework={fw}

{body_label}（你的主要事實來源）：
\"\"\"
{body}
\"\"\"
"""

def slugify(d):
    if d.get("slug_override"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,239}", d["slug_override"]):
            raise ValueError("slug_override_invalid")
        return d["slug_override"]
    fw = (d.get("source",{}) or {}).get("framework")
    base = fw if fw and fw.lower() not in ("null","none","") else d["title"]
    s = re.sub(r'[^a-zA-Z0-9]+','-', (base or "")).strip('-').lower()
    return s if (s and len(s) > 2 and not s.isdigit()) else f"art-{d['pos']}"

def build(d, raw):
    src = d.get("source",{}) or {}
    ax = d.get("axes",{})
    origin = src.get("origin", "quantml")
    arxiv = src.get("arxiv"); venue = src.get("venue") or "（無 venue）"
    fw = src.get("framework") or d["title"][:24]
    arxiv_line = f" · arXiv [{arxiv}](https://arxiv.org/abs/{arxiv})" if arxiv and arxiv not in ("null","None") else ""
    axes_human = " · ".join(f"{AXES_NAME[k]}={ax.get(k,'?')}" for k in AXES_NAME)
    body = raw.get('body', '')
    if not isinstance(body, str) or len(body) < 300:
        raise ValueError('source_too_thin')
    if len(body.encode('utf-8')) > MAX_SOURCE_BYTES:
        raise ValueError('source_context_too_large_requires_review')
    if origin in ('arxiv', 'oa') and not raw.get('source_kind', '').startswith('fulltext'):
        raise ValueError('fulltext_provenance_missing')
    if origin in ('arxiv', 'oa') and raw.get('source_kind') == 'fulltext-html' and raw.get('extraction_version') != HTML_TEXT_VERSION:
        raise ValueError('source_extraction_requires_refresh')
    if origin == "arxiv":
        src_label = "arXiv 原文"; body_label = "arXiv 完整快取可讀文本（含文末，未提取圖片／附件另列待核）"
        prov = "　·　_本頁依據已取得的 arXiv 可讀全文文本；圖片／附件未核_"
    elif origin == "oa":
        src_label = "期刊原文（OA）"; body_label = "OA 完整快取可讀文本（含文末，未提取圖片／附件另列待核）"
        prov = "　·　_本頁依據已取得的 OA 可讀全文文本；圖片／附件未核_"
    else:
        src_label = "QuantML 導讀"; body_label = "QuantML 導讀完整快取可讀文本（不代表已讀原論文）"
        prov = ""
    source_scope = (src_label + '，本次輸入完整快取文本，共' + str(len(body)) +
                    '字元；未提取圖片／附件不在本次輸入。' +
                    ('已提供原文可讀文本；§6.1僅列可定位的作者限制，不得套用「僅讀導讀」描述。' if origin in ('arxiv', 'oa')
                     else '僅讀導讀；§6.1必須說原論文作者限制未核，不得把導讀未見當作論文沒有。'))
    primary = origin in ('arxiv', 'oa')
    template = TEMPLATE.replace('導讀', '所讀原文') if primary else TEMPLATE
    prompt = template.format(
        name=fw, fw_en=fw, fw=fw, date=d.get("date",""), venue=venue,
        date_label="導讀發布" if origin == "quantml" else "來源發布",
        arxiv=arxiv, arxiv_line=arxiv_line, wechat_title=d.get("title",""),
        url=d.get("url",""), tldr=d.get("tldr",""), trick=d.get("key_trick",""),
        axes_human=axes_human, body=body,
        src_label=src_label, prov=prov, body_label=body_label, source_scope=source_scope,
        limitations_rule=('僅列所提供完整可讀文本可定位的作者限制，標原文章節；其餘明標本頁推斷，圖表／附件待核'
                          if primary else '僅讀導讀，必須明示「原論文作者限制未核」；不得把導讀未見當作論文沒有'),
        data=ax.get("data","?"),horizon=ax.get("horizon","?"),paradigm=ax.get("paradigm","?"),
        alpha=ax.get("alpha","?"),autonomy=ax.get("autonomy","?"))
    return prompt

_lock=threading.Lock(); _n={"ok":0,"skip":0,"fail":0}

def do_one(d):
    cand = sorted(RAW.glob(f"{d['msgid']}_*.json"))
    if not cand:
        with _lock:_n["skip"]+=1; return
    raw = json.loads(cand[0].read_text())
    slug = slugify(d)
    out = ROOT/"foundations"/d["zone"]/f"{slug}.md"
    if out.exists():
        with _lock:_n["skip"]+=1; return
    try:
        prompt = build(d, raw)
        for attempt in range(2):
            md = ask(prompt, system=SYSTEM, thinking=True, temperature=0.4,
                     max_tokens=32768, timeout=600).strip()
            if md.startswith("```"):
                md = md.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            try:
                validate_page(md, raw, d)
                break
            except ValueError as error:
                # Private draft evidence only; published pages and the search index stay intact.
                rejected = ROOT / "data/rejected"
                rejected.mkdir(parents=True, exist_ok=True)
                (rejected / (str(d["msgid"]) + f"-attempt{attempt+1}.md")).write_text(md)
                (rejected / (str(d["msgid"]) + ".reason")).write_text(str(error))
                if attempt == 1:
                    raise
                prompt = build(d, raw) + "\n上次草稿未通過機械品質檢查：" + str(error) + "。請依原始規則重寫，不可猜測或補造缺失值；無來源支持的實證數字寫未披露。"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        with _lock:
            _n["ok"]+=1
            print(f"  PAGE {d['zone']}/{slug}.md  ({len(md)}c)  <- {d['title'][:28]}", flush=True)
    except Exception as e:
        with _lock:
            _n["fail"]+=1
            print(f"  FAIL pos={d['pos']} :: {e}", file=sys.stderr, flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--only-pos",type=int,default=0)
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--shard",type=str,default="",help="i/m disjoint slice for parallel runs")
    ap.add_argument("--workers",type=int,default=2)
    args=ap.parse_args()
    ds=[json.loads(f.read_text()) for f in DIS.glob("*.json")]
    sel=[d for d in ds if d.get("page_worthy") and d.get("rating") in ("⚡","🔧") and not d.get("dup_of")]
    if args.only_pos: sel=[d for d in ds if d["pos"]==args.only_pos]
    sel.sort(key=lambda d:-d["pos"])
    if args.shard:
        i,m=(int(x) for x in args.shard.split("/")); sel=sel[i::m]
    if args.limit: sel=sel[:args.limit]
    print(f"Pass B on {len(sel)} pages, workers={args.workers}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(do_one,d) for d in sel]):
            fut.result()  # surface exceptions
    print(f"\nDONE ok={_n['ok']} skip={_n['skip']} fail={_n['fail']}",flush=True)

if __name__=="__main__":
    main()
    sys.exit(1 if _n["fail"] else 0)
