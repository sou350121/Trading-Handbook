#!/usr/bin/env python3
"""Pass B — qwen writes a full dissection page per AGENTS.md template (quant version).
Only for page_worthy ⚡/🔧 non-dup articles. Resumable. Writes foundations/<zone>/<slug>.md.

Usage: python3 distill_pass_b.py [--only-pos N] [--limit N] [--workers W]
"""
import json, pathlib, sys, argparse, re, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _qwen import ask

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"data"/"raw"; DIS = ROOT/"data"/"distill"
AXES_NAME = {"data":"數據模態","horizon":"時間尺度","paradigm":"學習範式","alpha":"Alpha生成機制","autonomy":"人機協作度"}

SYSTEM = ("你是『照見 Trading-Handbook』的資深解構作者。讀者是做過實盤/回測的量化研究員。"
          "你寫的不是論文摘要，而是【對比 + 失效模式 + 如何組合】的解構。中文輸出，繁體可，術語/框架名/指標保留英文。")

TEMPLATE = """請依下面骨架，把這篇方法寫成一頁解構 markdown。**不要寫成摘要**，要分析性論斷。
未知的數字/細節標 `TBD` 或 `（未驗證）`，不要編造。直接輸出 markdown，不要代碼圍欄。
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

骨架（嚴格按此順序與標題）：

<!-- ontology-5axis data={data} horizon={horizon} paradigm={paradigm} alpha={alpha} autonomy={autonomy} -->

# {name} 解構（{fw_en}）

> **發布**：{date} · {venue}{arxiv_line}
> **QuantML 導讀**：[{wechat_title}]({url})
> **核心定位**：（1-2 句：五軸落點 + 解了什麼 prior gap）

**Status:** v0.5 — 基於 QuantML 導讀 + 原論文（如有）。benchmark 細節待升 v1。
**TL;DR:**（≤4 句：①做什麼新事 ②核心 trick ③為什麼這對某條軸★ ④一個關鍵實證數字**——若導讀沒有量化結果，寫「導讀未給量化結果」，不要編數字**）

**X-Ray.**（150-300 字分析論斷：放回五軸 Pareto、解了哪些舊工程坑、預測它打不開的 envelope、對量化讀者意義。不是 summary）

## §1 · 架構 / Core Mechanism
（1.1 三大改動 vs 前作，用表；1.2 ⚡ Eureka 一句話 trick + 直覺；1.3 信息流 ASCII 圖）

## §2 · 數學層
（📌 Napkin Formula：1-3 行最關鍵 equation + 複雜度；直覺 2-3 句；loss/訓練細節）

## §3 · 數據層
（資料規模/頻率/市場/時段、怎麼來、樣本外與容量假設）

## §4 · 代碼層
（表：Repo / Checkpoint / License / 複現難度 / 數據可得性。未知填 TBD）

## §5 · 評測 / Benchmark
（表：數據集/市場 | Metric(IR/Sharpe/AR/MDD) | 前SOTA | 本方法 | Δ；**前SOTA 欄只填導讀逐字給的單一基線值，若有多個基線就改成逐行列出、別合成區間，沒有就寫「未披露」**；解讀哪部分 Δ 是真 capability，哪部分可能是過擬合/前瞻偏差/成本未計。**解讀散文裡也不得出現導讀沒有的數字**）

## §6 · 失效與隱含假設
（6.1 論文自述 limitations；6.2 推斷的隱含假設：regime 依賴/容量/成本/數據泄漏/survivorship）

## §7 · 對比 & 面試 Tip
（表：同軸對手 | 關鍵差異軸 | Open? | Status；🎤 Interview Tip 正確答 vs 錯答；7.1 可證偽預測帶日期）

## §8 · For the Reader
（按 persona 分流，至少 3 條：因子研究員/高頻執行/組合配置/LLM-agent/RL 策略/研究學生）

## References
（列原論文 + lineage + QuantML 導讀鏈接）

---

可用素材：
- 方法/框架名：{fw}
- 五軸座標：{axes_human}
- TL;DR 線索：{tldr}
- 核心 trick 線索：{trick}
- 來源：venue={venue} arxiv={arxiv} framework={fw}

QuantML 導讀正文（你的主要事實來源，截斷 8000 字）：
\"\"\"
{body}
\"\"\"
"""

def slugify(d):
    if d.get("slug_override"): return d["slug_override"]
    fw = (d.get("source",{}) or {}).get("framework")
    base = fw if fw and fw.lower() not in ("null","none","") else d["title"]
    s = re.sub(r'[^a-zA-Z0-9]+','-', (base or "")).strip('-').lower()
    return s or f"art-{d['pos']}"

def build(d, raw):
    src = d.get("source",{}) or {}
    ax = d.get("axes",{})
    arxiv = src.get("arxiv"); venue = src.get("venue") or "（無 venue）"
    fw = src.get("framework") or d["title"][:24]
    arxiv_line = f" · arXiv [{arxiv}](https://arxiv.org/abs/{arxiv})" if arxiv and arxiv not in ("null","None") else ""
    axes_human = " · ".join(f"{AXES_NAME[k]}={ax.get(k,'?')}" for k in AXES_NAME)
    return TEMPLATE.format(
        name=fw, fw_en=fw, fw=fw, date=d.get("date",""), venue=venue,
        arxiv=arxiv, arxiv_line=arxiv_line, wechat_title=d.get("title",""),
        url=d.get("url",""), tldr=d.get("tldr",""), trick=d.get("key_trick",""),
        axes_human=axes_human, body=raw["body"][:8000],
        data=ax.get("data","?"),horizon=ax.get("horizon","?"),paradigm=ax.get("paradigm","?"),
        alpha=ax.get("alpha","?"),autonomy=ax.get("autonomy","?"))

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
        md = ask(build(d, raw), system=SYSTEM, thinking=True, temperature=0.4,
                 max_tokens=7000, timeout=400)
        md = md.strip()
        if md.startswith("```"): md = md.strip("`"); md = md[md.find("\n")+1:]
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
