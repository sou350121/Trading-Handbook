#!/usr/bin/env python3
"""Pass A — qwen router+metadata over fetched articles (per QWEN_USAGE.md).
Opus-frozen prompt: classify into taxonomy zone + 5-axis coords + rating + source.
Resumable: writes data/distill/<msgid>.json, skips done.

Usage: python3 distill_pass_a.py [--sample N] [--limit N] [--workers W]
"""
import json, pathlib, sys, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _qwen import ask

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT/"data"/"raw"
OUT = ROOT/"data"/"distill"; OUT.mkdir(parents=True, exist_ok=True)
TAX = json.loads((ROOT/"data"/"taxonomy.json").read_text())

ZONES = [(z["slug"], z["name"], z.get("scope","")) for z in TAX["foundations_zones"]]
ZONE_SLUGS = {z[0] for z in ZONES} | {"unsorted"}

# Opus-frozen discrete pole sets (analyzable; qwen picks exactly one per axis)
AXES = {
 "data":     ["量价表格","微观盘口","文本另类","图关系","多模态"],
 "horizon":  ["高频日内","日频波段","中长周期","跨周期"],
 "paradigm": ["监督回归","强化学习","生成式大模型","因果结构","元学习搜索"],
 "alpha":    ["因子挖掘","端到端表征","组合执行优化","多智能体博弈","风险择时"],
 "autonomy": ["全自动黑盒","人机协同可解释","Agent自主演进"],
}
RATING_RUBRIC = ("⚡=对实盘或方法论有强冲击/新范式; 🔧=扎实可复用的工程或方法改进; "
                 "📖=综述/科普/边际贡献; ❌=低质/纯广告/与已有重复")

SYSTEM = "你是量化投资+机器学习资深研究员，负责把论文导读文章精准分类并抽取结构化元数据。只输出严格 JSON。"

def build_prompt(rec):
    zlist = "\n".join(f"  - {s} ({n})：{sc[:40]}" for s,n,sc in ZONES)
    alist = "\n".join(f"  - {k}: 从 [{' | '.join(v)}] 选一个" for k,v in AXES.items())
    body = rec["body"][:7000]
    return f"""文章标题：{rec['title']}
发布日期：{rec.get('date','')}

正文（截断到 7000 字）：
\"\"\"
{body}
\"\"\"

请把这篇文章归类并抽取，【只输出一个 JSON】：
{{
  "zone": "从下列 slug 选一个，都不符合才填 unsorted",
  "axes": {{"data":"","horizon":"","paradigm":"","alpha":"","autonomy":""}},
  "rating": "⚡ 或 🔧 或 📖 或 ❌",
  "tldr": "≤60字，一句话说清这篇做了什么",
  "key_trick": "≤80字，核心 trick / 与前作最关键的不同",
  "source": {{"venue":"如 ICLR26/KDD/JFE，没有填 null","arxiv":"NNNN.NNNNN 或 null","framework":"方法/框架名或 null","confidence":"high|mid|low"}},
  "tags": ["≤5个中文标签"],
  "page_worthy": true/false（是否值得写成一整页解构：有明确方法+可对比+非纯综述才 true）
}}

可选 zone：
{zlist}

五轴（每轴恰好选一个给定值）：
{alist}

评级标准：{RATING_RUBRIC}
只输出 JSON，不要解释、不要代码围栏。"""

def parse(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```",2)[-1] if t.count("```")>=2 else t.strip("`")
        if t.lstrip().startswith("json"): t = t.lstrip()[4:]
    i,j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j+1])

_lock = threading.Lock(); _stats={"ok":0,"fail":0,"skip":0}

def do_one(f):
    rec = json.loads(f.read_text())
    out = OUT/f"{rec['msgid']}.json"
    if out.exists():
        with _lock: _stats["skip"]+=1
        return
    try:
        raw = ask(build_prompt(rec), system=SYSTEM, thinking=True,
                  temperature=0.1, max_tokens=2000, json_mode=True, timeout=180)
        d = parse(raw)
        if d.get("zone") not in ZONE_SLUGS: d["zone"]="unsorted"
        for k,opts in AXES.items():
            v=d.get("axes",{}).get(k)
            if v not in opts: d.setdefault("axes",{})[k]= (v or "?")
        d.update({"pos":rec["pos"],"msgid":rec["msgid"],"title":rec["title"],
                  "date":rec.get("date",""),"url":rec["url"],"chars":rec["chars"]})
        # First-hand raws (arXiv / OA-journal pdf) are papers, not QuantML digests: stamp
        # source.origin so Pass B renders the primary-source header no matter which cron
        # (weekly dissect OR the daily raw sweep) happens to classify them.
        if rec["msgid"].startswith("arxiv_"):
            d.setdefault("source", {})["origin"] = "arxiv"
            d["source"]["arxiv"] = rec["msgid"][len("arxiv_"):]
        elif rec["msgid"].startswith("oa_"):
            d.setdefault("source", {})["origin"] = "oa"
        out.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        with _lock:
            _stats["ok"]+=1
            print(f"  {d['rating']} [{d['zone']:>22}] {d['axes'].get('paradigm','?'):>6} pw={str(d.get('page_worthy'))[:1]} | {rec['title'][:30]}", flush=True)
    except Exception as e:
        with _lock:
            _stats["fail"]+=1
            print(f"  FAIL pos={rec['pos']} {rec['title'][:24]} :: {e}", file=sys.stderr, flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sample",type=int,default=0,help="only N newest fetched")
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--workers",type=int,default=3)
    args=ap.parse_args()
    files=sorted(RAW.glob("*.json"), key=lambda p:-json.loads(p.read_text())["pos"])
    if args.sample: files=files[:args.sample]
    if args.limit: files=files[:args.limit]
    print(f"Pass A on {len(files)} articles, workers={args.workers}",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed(ex.submit(do_one,f) for f in files))
    print(f"\nDONE ok={_stats['ok']} skip={_stats['skip']} fail={_stats['fail']}",flush=True)

if __name__=="__main__":
    main()
