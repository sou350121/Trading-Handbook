#!/usr/bin/env python3
"""祖先論文正典 — walk the Crossref citation graph OUT of our resolved papers to find the
seminal works the whole corpus keeps citing, and render cheat-sheet/seminal-canon.md.

Mechanics (number-discipline: counts are mechanical, qwen only writes one-liners):
  1. collect every resolver-injected DOI + arXiv id from foundations pages
  2. Crossref /works/{doi} -> open reference lists (publisher-deposited); tally referenced DOIs
  3. candidates referenced by >= MIN_PAGES of our papers -> fetch their metadata + cited-by count
  4. qwen: one ≤40字 mechanism one-liner each (no numbers invented)
  5. render canon page grouped by era, each entry: title/venue/year/cited-by/幾頁在引/one-liner

Usage: python3.11 build_seminal_canon.py [--min 3] [--top 40] [--dry]
"""
import json, re, sys, time, signal, pathlib, argparse, urllib.request, urllib.parse, datetime as dt
from collections import defaultdict

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _qwen import ask

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "cheat-sheet" / "seminal-canon.md"
CACHE = ROOT / "data" / "canon_cache.json"      # crossref responses cache (gitignored)
MAILTO = "sou350121@gmail.com"
UA = {"User-Agent": f"Trading-Handbook canon builder (mailto:{MAILTO})"}


def http_json(url, timeout=25, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise last


def our_papers():
    """(doi_set, per-page zone map) from resolver-injected 原始論文 links + arXiv headers."""
    dois = {}
    for p in ROOT.glob("foundations/*/*.md"):
        if p.name == "overview.md":
            continue
        txt = p.read_text(encoding="utf-8")
        zone = p.parent.name
        for m in re.finditer(r'doi\.org/(10\.[^\s)\]"]+)', txt):
            dois.setdefault(m.group(1).rstrip(".,;"), set()).add(zone)
    return dois


def crossref_work(doi, cache):
    if doi in cache:
        return cache[doi]
    try:
        d = http_json(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}")["message"]
    except Exception:
        cache[doi] = None
        return None
    keep = {"title": (d.get("title") or [""])[0][:200],
            "container": (d.get("container-title") or [""])[0][:80],
            "year": (d.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "cited_by": d.get("is-referenced-by-count", 0),
            "refs": [r.get("DOI", "").lower() for r in d.get("reference", []) if r.get("DOI")]}
    cache[doi] = keep
    return keep


SYS = "你是量化交易手冊的正典編輯。為每篇開山論文寫一句 ≤40 字的中文機制/貢獻描述，術語留英文，不編數字。只輸出 JSON。"


def one_liners(entries):
    blocks = "\n".join(f"{i}. {e['title']} ({e['container']}, {e['year']})" for i, e in enumerate(entries))
    prompt = f"""為下列開山論文各寫一句 ≤40 字中文「它開創了什麼」：\n{blocks}\n
輸出 JSON: {{"lines": {{"0": "…", "1": "…", …}}}}"""
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("alarm")))
    signal.alarm(170)
    try:
        raw = ask(prompt, system=SYS, thinking=True, temperature=0.2, max_tokens=3000,
                  json_mode=True, timeout=150, retries=2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1]).get("lines", {})


def era(y):
    if not y: return "年代不明"
    if y < 1990: return "1950–1989 · 奠基"
    if y < 2005: return "1990–2004 · 因子與微結構成型"
    if y < 2015: return "2005–2014 · 統計學習進場"
    if y < 2020: return "2015–2019 · 深度學習遷移"
    return "2020– · 大模型與新範式"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=3, help="min # of our papers citing it")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    ours = our_papers()
    print(f"resolved papers in corpus: {len(ours)}", flush=True)

    # 1. tally what OUR papers reference
    tally = defaultdict(set)          # ref_doi -> set(our_doi)
    ok = 0
    for i, doi in enumerate(sorted(ours)):
        w = crossref_work(doi, cache)
        if not w:
            continue
        ok += 1
        for r in w["refs"]:
            tally[r].add(doi)
        if i % 20 == 0:
            CACHE.write_text(json.dumps(cache))
            print(f"  walked {i}/{len(ours)} (open refs ok so far: {ok})", flush=True)
    CACHE.write_text(json.dumps(cache))

    cands = [(r, len(s)) for r, s in tally.items() if len(s) >= a.min and r not in ours]
    cands.sort(key=lambda x: -x[1])
    cands = cands[:a.top * 2]          # fetch metadata for a buffer, filter junk after
    print(f"candidates >= {a.min} in-corpus citations: {len(cands)}", flush=True)

    entries = []
    for r, n in cands:
        w = crossref_work(r, cache)
        if not w or not w["title"] or len(w["title"]) < 8:
            continue
        zones = set()
        for od in tally[r]:
            zones |= ours.get(od, set())
        entries.append({"doi": r, "incorpus": n, "zones": sorted(zones)[:4], **w})
        if len(entries) >= a.top:
            break
    CACHE.write_text(json.dumps(cache))
    entries.sort(key=lambda e: (-e["incorpus"], -(e["cited_by"] or 0)))

    if a.dry:
        for e in entries[:15]:
            print(f"  {e['incorpus']:>2}頁引用 | {e['cited_by']:>6}被引 | {e['year']} | {e['title'][:70]}")
        return

    lines_map = {}
    try:
        for k in range(0, len(entries), 20):
            lines_map.update({str(int(kk) + k): v for kk, v in one_liners(entries[k:k + 20]).items()})
    except Exception as e:
        print(f"one-liner batch failed ({repr(e)[:60]}) — rendering without", file=sys.stderr)

    by_era = defaultdict(list)
    for i, e in enumerate(entries):
        e["line"] = (lines_map.get(str(i)) or "").strip()[:60]
        by_era[era(e["year"])].append(e)

    L = ["# 祖先論文正典（Seminal Canon）", "",
         f"> 本手冊 {len(ours)} 篇已解析論文的引文圖中，被 **≥{a.min} 頁**共同引用的開山之作——"
         "整個量化 ML 領域站在誰的肩膀上。引用計數來自 Crossref（機械統計，非人工挑選）。", "",
         "| 欄位 | 含義 |", "|---|---|",
         "| 冊內引用 | 本手冊多少篇解構頁的原始論文引了它 |",
         "| 全網被引 | Crossref is-referenced-by-count |", ""]
    order = ["1950–1989 · 奠基", "1990–2004 · 因子與微結構成型", "2005–2014 · 統計學習進場",
             "2015–2019 · 深度學習遷移", "2020– · 大模型與新範式", "年代不明"]
    for er in order:
        if er not in by_era:
            continue
        L.append(f"## {er}")
        L.append("")
        L.append("| 論文 | 年 | 冊內引用 | 全網被引 | 主要引用區 | 一句話 |")
        L.append("|---|---|--:|--:|---|---|")
        for e in sorted(by_era[er], key=lambda x: (-x["incorpus"], x["year"] or 9999)):
            zones = " ".join(f"`{z}`" for z in e["zones"])
            t = e["title"].replace("|", "\\|")
            L.append(f"| [{t}](https://doi.org/{e['doi']}) | {e['year'] or '?'} | {e['incorpus']} | "
                     f"{e['cited_by']} | {zones} | {e['line'] or '—'} |")
        L.append("")
    L.append(f"_由 `scripts/pulsar/build_seminal_canon.py` 生成 · {dt.date.today()} · "
             "候選僅含 Crossref 開放引文（部分出版社不開放參考文獻,會低估）。_")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT} with {len(entries)} canon entries", flush=True)


if __name__ == "__main__":
    main()
