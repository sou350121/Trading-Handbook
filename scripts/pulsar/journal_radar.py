#!/usr/bin/env python3
"""頂刊雷達 — poll top finance journals via Crossref (no fragile publisher RSS), qwen-classify
into the 10 zones, and resolve OA full text via Unpaywall so ⚡ papers can be first-hand dissected.

Why Crossref polling instead of RSS: publisher RSS (Wiley/Elsevier/OUP/T&F) is fragile and often
walled from this datacenter IP, while api.crossref.org's polite pool is proven-green here (same
backend as resolve_sources_v2). Titles/DOIs/dates come straight from the registration authority.

Flow (weekly, driven by radar_weekly.sh):
  1. per journal ISSN: /journals/{issn}/works filtered to recently *created* journal-articles
  2. dedup vs data/journal_seen.json -> qwen classify {relevance, zone, rating, one_line}
  3. core items -> Unpaywall (email param) -> if an OA pdf exists, queue into data/oa_queue.json
     (consumed by arxiv_dissect.py --oa for first-hand dissection)
  4. render radar/journals/<ISO-week>.md grouped by journal; skipped counts shown (no silent caps)

Usage: python3.11 journal_radar.py [--days 9] [--limit N] [--dry]
"""
import json, re, sys, time, signal, pathlib, argparse, datetime as dt, urllib.request, urllib.parse

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _qwen import ask

ROOT = pathlib.Path(__file__).resolve().parents[2]
FEEDS = json.loads((HERE / "journal_feeds.json").read_text())
TAX = json.loads((ROOT / "data" / "taxonomy.json").read_text())
ZONE_SLUGS = [z["slug"] for z in TAX["foundations_zones"]]
SEEN_F = ROOT / "data" / "journal_seen.json"
OA_QUEUE_F = ROOT / "data" / "oa_queue.json"
OUT_DIR = ROOT / "radar" / "journals"
MAILTO = "sou350121@gmail.com"
UA = {"User-Agent": f"Trading-Handbook journal radar (mailto:{MAILTO})"}


def http_json(url, timeout=25, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(2.0 * (i + 1))
    raise last


def load_json(p, default):
    try:
        return json.loads(pathlib.Path(p).read_text())
    except Exception:
        return default


def strip_jats(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    return re.sub(r"\s+", " ", s).strip()


def fetch_journal(j, days):
    since = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    url = (f"https://api.crossref.org/journals/{j['issn']}/works"
           f"?filter=from-created-date:{since},type:journal-article"
           f"&rows=40&sort=created&order=desc&select=DOI,title,abstract,created,issued,container-title"
           f"&mailto={MAILTO}")
    try:
        d = http_json(url)
    except Exception as e:
        print(f"  JOURNAL-FAIL {j['name']}: {repr(e)[:60]}", file=sys.stderr)
        return []
    out = []
    for it in d.get("message", {}).get("items", []):
        title = (it.get("title") or [""])[0].strip()
        doi = it.get("DOI", "")
        if not title or not doi:
            continue
        # guard against backfile floods: created is recent, but require issued year >= 2025 too
        parts = (it.get("issued", {}).get("date-parts") or [[None]])[0]
        year = parts[0] if parts and parts[0] else None
        if year and year < 2025:
            continue
        created = (it.get("created", {}).get("date-parts") or [[None]])[0]
        cdate = "-".join(f"{x:02d}" if i else str(x) for i, x in enumerate(created)) if created[0] else ""
        out.append({"doi": doi, "title": re.sub(r"\s+", " ", title), "journal": j["name"],
                    "abstract": strip_jats(it.get("abstract", ""))[:1200], "date": cdate})
    return out


CLASSIFY_SYS = ("你是量化交易手冊的頂刊掃描員。判斷一篇金融學術論文是否與『量化 ML / 系統化交易方法』相關，"
                "並分區。只輸出 JSON。")


def classify(item):
    zlist = " / ".join(ZONE_SLUGS)
    prompt = f"""論文（來自 {item['journal']}）：
標題：{item['title']}
摘要：{item['abstract'] or '（無摘要）'}

輸出 JSON：
{{"relevance": "core|adjacent|skip",
  "zone": "從 {zlist} 選一,不符合填 skip",
  "rating": "⚡ 或 🔧 或 📖",
  "one_line": "≤40字中文,說清它的機制/貢獻,不編數字"}}

判準：core=量化ML/系統化交易方法或其評測(值得入冊);adjacent=資產定價/微結構等重要實證,對量化讀者有背景價值;skip=公司金融/銀行/治理等與交易無關。
rating:⚡=方法性強、值得全文解構;🔧=重要可用;📖=背景。"""
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("alarm")))
    signal.alarm(85)
    try:
        raw = ask(prompt, system=CLASSIFY_SYS, thinking=False, temperature=0.1,
                  max_tokens=220, json_mode=True, timeout=55, retries=2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    i, j = raw.find("{"), raw.rfind("}")
    d = json.loads(raw[i:j + 1])
    if d.get("zone") not in ZONE_SLUGS:
        d["zone"] = "skip"
    if d.get("rating") not in ("⚡", "🔧", "📖"):
        d["rating"] = "📖"
    if d.get("relevance") not in ("core", "adjacent", "skip"):
        d["relevance"] = "skip"
    d["one_line"] = (d.get("one_line") or "").strip()[:60]
    return d


def unpaywall_pdf(doi):
    """Return OA pdf candidate urls, repository copies FIRST: publisher-hosted pdfs (OUP/Wiley...)
    403 this datacenter IP (verified — direct, allorigins and jina all walled), while university/
    SSRN repository copies usually fetch fine. Empty list = no OA."""
    try:
        d = http_json(f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={MAILTO}", retries=2)
    except Exception:
        return []
    repo, pub = [], []
    for loc in d.get("oa_locations") or []:
        pdf = loc.get("url_for_pdf")
        if not pdf:
            continue
        (repo if loc.get("host_type") == "repository" else pub).append(pdf)
    return repo + pub


def iso_week():
    y, w, _ = dt.date.today().isocalendar()
    return f"{y}-W{w:02d}"


def render(week, kept, skipped_n, counts):
    mon = dt.date.fromisocalendar(*[int(x) for x in week.split("-W")], 1)
    sun = mon + dt.timedelta(days=6)
    L = [f"# 頂刊雷達 · {week}", "",
         f"> 覆蓋 {mon} ~ {sun} · 12 本金融/量化頂刊新文（Crossref 註冊流）· 只列標題+DOI+我們的一句話機制解讀",
         f"> `⚡` = 方法性強、已排進一手全文解構隊列（有 OA 全文時） · `📄` = 找到開放取用全文", ""]
    if counts:
        L.append("**本週各刊計數**: " + " · ".join(f"{k} {v}" for k, v in counts.items()))
        L.append("")
    by_j = {}
    for it in kept:
        by_j.setdefault(it["journal"], []).append(it)
    hot = [it for it in kept if it["rating"] == "⚡"]
    if hot:
        L.append("## ⚡ 本週必看")
        L.append("")
        for it in sorted(hot, key=lambda x: x["title"]):
            oa = " 📄" if it.get("oa_pdf") else ""
            L.append(f"- **[{it['title']}](https://doi.org/{it['doi']})**{oa} · {it['journal']} · `{it['zone']}`")
            L.append(f"  {it['one_line']}")
        L.append("")
    for jname in sorted(by_j):
        L.append(f"## {jname}")
        L.append("")
        for it in sorted(by_j[jname], key=lambda x: (x["rating"] != "⚡", x["rating"] != "🔧", x["title"])):
            oa = " 📄" if it.get("oa_pdf") else ""
            zone = f" `{it['zone']}`" if it["zone"] in ZONE_SLUGS else ""   # adjacent items may have no zone
            L.append(f"- {it['rating']}{oa} [{it['title']}](https://doi.org/{it['doi']}){zone} — {it['one_line']}")
        L.append("")
    L.append(f"---\n\n_另有 {skipped_n} 篇與量化交易無關（公司金融/銀行/治理等），已過濾不列。_")
    L.append("\n回 [頂刊雷達總覽](overview.md) · [手冊首頁](../../README.md)")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0, help="cap classified items (test)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    seen = load_json(SEEN_F, {})
    week = iso_week()
    fresh = []
    for j in FEEDS:
        items = fetch_journal(j, a.days)
        new = [it for it in items if it["doi"] not in seen]
        print(f"  {j['name']:<38} {len(items):>3} recent, {len(new):>3} new", flush=True)
        fresh.extend(new)
    if a.limit:
        fresh = fresh[:a.limit]
    if not fresh:
        print("journal_radar: no new items")
        return

    kept, skipped = [], 0
    for it in fresh:
        try:
            c = classify(it)
        except Exception as e:
            print(f"  CLASSIFY-FAIL {it['doi']}: {repr(e)[:60]}", file=sys.stderr)
            continue  # not marked seen -> retried next run
        seen[it["doi"]] = week
        if c["relevance"] == "skip":
            skipped += 1
            continue
        it.update(c)
        if c["relevance"] == "core":
            cands = unpaywall_pdf(it["doi"])
            it["oa_pdf"] = cands[0] if cands else ""
            it["oa_cands"] = cands
        print(f"  {c['rating']} [{c['zone']:>24}] {'OA' if it.get('oa_pdf') else '  '} | {it['title'][:56]}", flush=True)
        kept.append(it)

    # queue ⚡+OA papers for first-hand dissection (consumed by arxiv_dissect --oa)
    q = load_json(OA_QUEUE_F, [])
    qdois = {x["doi"] for x in q}
    for it in kept:
        if it["rating"] == "⚡" and it.get("oa_cands") and it["doi"] not in qdois:
            q.append({"doi": it["doi"], "pdf_candidates": it["oa_cands"], "title": it["title"],
                      "journal": it["journal"], "zone": it["zone"], "date": it["date"], "week": week})
    counts = {}
    for it in kept:
        counts[it["journal"]] = counts.get(it["journal"], 0) + 1

    if a.dry:
        print(f"DRY: kept={len(kept)} skipped={skipped} oa_queue+={len(q) - len(qdois)}")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{week}.md").write_text(render(week, kept, skipped, counts))
    pathlib.Path(SEEN_F).write_text(json.dumps(seen, ensure_ascii=False, indent=1))
    pathlib.Path(OA_QUEUE_F).write_text(json.dumps(q, ensure_ascii=False, indent=1))
    write_overview()
    print(f"journal_radar: wrote {OUT_DIR}/{week}.md  kept={len(kept)} skipped={skipped} "
          f"oa_queue={len(q)}", flush=True)


def write_overview():
    weeks = sorted((p.stem for p in OUT_DIR.glob("*-W*.md")), reverse=True)
    names = " · ".join(f["name"] for f in FEEDS)
    L = ["# 頂刊雷達 Journal Radar", "",
         f"金融/量化 12 本頂級期刊（{names}）的新文，每週經 **Crossref 註冊流**輪詢一次（比出版社 RSS 穩），"
         "qwen 依機制歸入 10 個 zone、濾掉與量化交易無關的公司金融/銀行文獻。這補上了 arXiv 之外**真正的金融文獻主戰場**。", "",
         "⚡ = 方法性強；若 Unpaywall 找到開放取用全文（📄），自動進**一手全文解構**佇列，產出 foundations/ 正式解構頁。", "",
         "## 每週雷達 Weekly pages", ""]
    for w in weeks:
        L.append(f"- [{w}](./{w}.md)")
    L.append("")
    L.append(f"_索引由 `scripts/pulsar/journal_radar.py` 維護 · 更新於 {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC._")
    (OUT_DIR / "overview.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
