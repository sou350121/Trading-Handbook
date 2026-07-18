#!/usr/bin/env python3.11
"""ARXIV Q-FIN RADAR — weekly sweep of new arXiv quant papers, qwen-classified
into the Trading-Handbook's 10 zones, published as a weekly radar page.

Stdlib only (urllib/json/re/time/xml.etree). Resumable + idempotent:
- DEDUPE via data/radar_seen.json (append-only arXiv id set) so old ids are
  never reprocessed.
- Per-week item lists in data/radar_week_state.json so a week's page can be
  deterministically rebuilt (re-running in the same ISO week regenerates the
  same file from accumulated state).

Pipeline:
  1. FETCH  — arXiv API, categories q-fin.* + (cs.LG OR stat.ML) keyword-filtered.
              >=3s spacing between requests, exponential backoff on 429.
  2. DEDUPE — skip ids already in radar_seen.json.
  3. CLASSIFY — one qwen call per new paper (cap 80/run): relevant / zone /
              one_liner_zh / hot.
  4. PUBLISH — radar/arxiv/<year>-W<week>.md (one table per zone),
              radar/arxiv/overview.md (index), data/radar_queue.json (hot items).

Usage:
  python3.11 scripts/pulsar/arxiv_radar.py [--days 8] [--max-classify 80] [--dry-run]

Cron (weekly, Monday 06:20):
  20 6 * * 1 cd /home/claudeuser/trading-handbook-work && \
    python3.11 scripts/pulsar/arxiv_radar.py --days 8 >> data/radar.log 2>&1
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # trading-handbook-work/
DATA = os.path.join(ROOT, "data")
RADAR_DIR = os.path.join(ROOT, "radar", "arxiv")
SEEN_PATH = os.path.join(DATA, "radar_seen.json")
QUEUE_PATH = os.path.join(DATA, "radar_queue.json")
WEEK_STATE_PATH = os.path.join(DATA, "radar_week_state.json")

sys.path.insert(0, HERE)
import _qwen  # noqa: E402

# ---------------------------------------------------------------------------
# arXiv API config
# ---------------------------------------------------------------------------
ARXIV_URL = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
USER_AGENT = "trading-handbook-radar/1.0 (+https://github.com/; Pulsar q-fin radar)"

QFIN_CATS = ["q-fin.PM", "q-fin.TR", "q-fin.ST", "q-fin.CP", "q-fin.RM", "q-fin.GN"]
ML_CATS = ["cs.LG", "stat.ML"]

# Keyword filter applied to (cs.LG OR stat.ML) papers on title+abstract.
# Whole-ish token / phrase matching, case-insensitive.
ML_KEYWORDS = [
    "portfolio", "trading", "stock", "asset pricing", "alpha", "factor",
    "hedge", "market making", "limit order", "lob", "cryptocurrency",
    "futures", "volatility forecast",
]
# Precompile keyword regexes. Multiword phrases -> phrase match; single tokens
# use word boundaries to avoid substring false positives (e.g. "factory").
_KW_RES = []
for kw in ML_KEYWORDS:
    if " " in kw:
        _KW_RES.append(re.compile(re.escape(kw), re.I))
    else:
        _KW_RES.append(re.compile(r"\b" + re.escape(kw) + r"\b", re.I))

# The handbook's 10 canonical zone slugs (must match foundations/ dirs).
ZONES = [
    "time-series-forecasting",
    "graph-networks",
    "reinforcement-learning",
    "llm-agentic",
    "market-microstructure",
    "factor-mining",
    "portfolio-optimization",
    "causal-structural",
    "evaluation-benchmarks",
    "data-generation-augmentation",
]
ZONE_SET = set(ZONES)
# Human-facing zone headings (bilingual) for the published tables.
ZONE_TITLES = {
    "time-series-forecasting": "Time-Series Forecasting 時序預測",
    "graph-networks": "Graph & Networks 圖網絡",
    "reinforcement-learning": "Reinforcement Learning 強化學習",
    "llm-agentic": "LLM & Agentic 大模型與智能體",
    "market-microstructure": "Market Microstructure 市場微結構",
    "factor-mining": "Factor Mining 因子挖掘",
    "portfolio-optimization": "Portfolio Optimization 組合優化",
    "causal-structural": "Causal & Structural 因果與結構",
    "evaluation-benchmarks": "Evaluation & Benchmarks 評測基準",
    "data-generation-augmentation": "Data Generation & Augmentation 數據生成與增強",
}

MAX_RESULTS = 100  # arXiv page size
REQ_SPACING = 3.5  # seconds between requests (arXiv asks >=3s)
BACKOFF_429 = [10, 30, 60, 60, 60]  # exponential-ish backoff, 5 tries
_last_req_ts = [0.0]


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------
def _load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _atomic_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _atomic_write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# arXiv fetch with spacing + backoff
# ---------------------------------------------------------------------------
def _http_get(url):
    """GET with >=REQ_SPACING spacing and 429 exponential backoff (5 tries)."""
    for attempt in range(len(BACKOFF_429) + 1):
        # enforce spacing since last request
        gap = time.time() - _last_req_ts[0]
        if gap < REQ_SPACING:
            time.sleep(REQ_SPACING - gap)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", "replace")
            _last_req_ts[0] = time.time()
            return body
        except urllib.error.HTTPError as e:
            _last_req_ts[0] = time.time()
            if e.code == 429 and attempt < len(BACKOFF_429):
                wait = BACKOFF_429[attempt]
                print(f"  arXiv 429 (attempt {attempt + 1}) -> backoff {wait}s", flush=True)
                time.sleep(wait)
                continue
            # other HTTP error, or 429 out of retries
            print(f"  arXiv HTTP {e.code}: {e}", flush=True)
            if attempt < len(BACKOFF_429):
                wait = BACKOFF_429[attempt]
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            _last_req_ts[0] = time.time()
            print(f"  arXiv URLError (attempt {attempt + 1}): {e}", flush=True)
            if attempt < len(BACKOFF_429):
                time.sleep(BACKOFF_429[attempt])
                continue
            raise
    raise RuntimeError("arXiv fetch exhausted retries")


def _parse_arxiv_id(raw_id):
    """From 'http://arxiv.org/abs/2401.01234v2' -> '2401.01234' (version stripped)."""
    m = re.search(r"arxiv\.org/abs/([^v\s]+?)(v\d+)?$", raw_id.strip())
    if m:
        return m.group(1)
    # fallback: last path segment sans version
    tail = raw_id.rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", tail)


def _parse_entries(xml_text):
    """Parse an Atom feed into a list of paper dicts. Returns (entries, total)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        print(f"  XML parse error: {e}", flush=True)
        return [], 0
    total = 0
    tot_el = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    if tot_el is not None and tot_el.text and tot_el.text.isdigit():
        total = int(tot_el.text)
    out = []
    for entry in root.findall(f"{ATOM}entry"):
        rid = entry.findtext(f"{ATOM}id", "").strip()
        if not rid:
            continue
        aid = _parse_arxiv_id(rid)
        title = " ".join((entry.findtext(f"{ATOM}title", "") or "").split())
        summary = " ".join((entry.findtext(f"{ATOM}summary", "") or "").split())
        published = (entry.findtext(f"{ATOM}published", "") or "").strip()
        updated = (entry.findtext(f"{ATOM}updated", "") or "").strip()
        authors = []
        for a in entry.findall(f"{ATOM}author"):
            nm = a.findtext(f"{ATOM}name", "")
            if nm:
                authors.append(nm.strip())
        # canonical abs URL
        abs_url = f"https://arxiv.org/abs/{aid}"
        for link in entry.findall(f"{ATOM}link"):
            if link.get("rel") == "alternate" and link.get("href"):
                abs_url = link.get("href")
                break
        # primary category
        prim = entry.find(f"{ARXIV_NS}primary_category")
        primary_cat = prim.get("term") if prim is not None else ""
        cats = [c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")]
        out.append({
            "id": aid,
            "title": title,
            "abstract": summary,
            "published": published,
            "updated": updated,
            "authors": authors,
            "abs_url": abs_url,
            "primary_category": primary_cat,
            "categories": cats,
        })
    return out, total


def _fetch_search(search_query, since_dt):
    """Paginate a search_query, newest-first, stopping once entries are older
    than since_dt (arXiv has no reliable server-side date range for the free
    text API, so we sort by submittedDate desc and cut client-side)."""
    collected = []
    start = 0
    while True:
        params = {
            "search_query": search_query,
            "start": str(start),
            "max_results": str(MAX_RESULTS),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = ARXIV_URL + "?" + urllib.parse.urlencode(params)
        body = _http_get(url)
        entries, total = _parse_entries(body)
        if not entries:
            break
        stop = False
        for e in entries:
            # use published (original submission) for the date cut
            ts = _parse_ts(e.get("published") or e.get("updated"))
            if ts is None:
                collected.append(e)  # keep undated, let classify decide
                continue
            if ts < since_dt:
                stop = True
                break
            collected.append(e)
        print(f"    page start={start}: {len(entries)} entries "
              f"(kept running total {len(collected)}, feed total~{total})", flush=True)
        if stop or len(entries) < MAX_RESULTS:
            break
        start += MAX_RESULTS
        if start >= 2000:  # hard safety cap on pagination depth
            print("    reached pagination safety cap (2000)", flush=True)
            break
    return collected


def _parse_ts(s):
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _matches_keywords(paper):
    hay = (paper.get("title", "") + "  " + paper.get("abstract", ""))
    return any(rx.search(hay) for rx in _KW_RES)


def fetch_papers(days):
    """Fetch q-fin.* (all) + (cs.LG OR stat.ML) keyword-filtered over last `days`."""
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    print(f"[fetch] window: since {since_dt.isoformat()} ({days}d)", flush=True)

    by_id = {}

    # 1) q-fin categories (take all — these are already the target domain)
    qfin_query = " OR ".join(f"cat:{c}" for c in QFIN_CATS)
    print(f"[fetch] q-fin categories: {qfin_query}", flush=True)
    for p in _fetch_search(qfin_query, since_dt):
        by_id.setdefault(p["id"], p)
    n_qfin = len(by_id)
    print(f"[fetch] q-fin kept: {n_qfin}", flush=True)

    # 2) cs.LG OR stat.ML, keyword-filtered on title/abstract
    ml_query = " OR ".join(f"cat:{c}" for c in ML_CATS)
    print(f"[fetch] ML categories (keyword-filtered): {ml_query}", flush=True)
    ml_raw = _fetch_search(ml_query, since_dt)
    n_ml_kept = 0
    for p in ml_raw:
        if p["id"] in by_id:
            continue
        if _matches_keywords(p):
            by_id[p["id"]] = p
            n_ml_kept += 1
    print(f"[fetch] ML scanned {len(ml_raw)}, kept {n_ml_kept} after keyword filter", flush=True)

    papers = list(by_id.values())
    # newest first
    papers.sort(key=lambda p: p.get("published") or "", reverse=True)
    print(f"[fetch] total unique papers in window: {len(papers)}", flush=True)
    return papers


# ---------------------------------------------------------------------------
# CLASSIFY with qwen
# ---------------------------------------------------------------------------
CLASSIFY_SYSTEM = (
    "你是量化投資研究的分類助手，服務於一本量化交易手冊。"
    "你只判斷論文是否真正關於量化交易/投資（用機器學習/AI/統計方法做選股、擇時、"
    "組合、對沖、做市、因子、風險），並歸入手冊的固定 zone。"
    "純經濟學理論、宏觀計量、沒有市場/資產/交易落點的方法論，一律 relevant=false。"
    "只輸出 JSON，不要多餘文字。"
)

ZONE_GUIDE = (
    "zone 只能是以下之一（嚴格用 slug）：\n"
    "- time-series-forecasting：時序/序列模型做收益率或價格預測（RNN/Transformer/TCN/狀態空間等）\n"
    "- graph-networks：把資產/公司/事件建成圖，用 GNN 傳遞關係信號\n"
    "- reinforcement-learning：用 RL/策略梯度/Q學習做交易或組合決策\n"
    "- llm-agentic：用大語言模型/多智能體/檢索做研究、情緒、決策或代理執行\n"
    "- market-microstructure：限價單簿/做市/高頻/訂單流/微觀價格形成\n"
    "- factor-mining：自動挖因子/alpha 表達式/符號回歸/特徵構造\n"
    "- portfolio-optimization：組合權重/風險預算/端到端優化/約束下配置\n"
    "- causal-structural：因果推斷/結構因果/處理效應在金融的應用\n"
    "- evaluation-benchmarks：評測方法/基準集/回測嚴謹性/數據洩漏診斷/可復現\n"
    "- data-generation-augmentation：合成金融數據/生成模型/數據增強/去噪\n"
    "若跨多個，選最主導的機制軸。"
)


def _extract_json(text):
    """Pull the first JSON object out of a qwen reply (tolerant of fences/prose)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def classify_paper(paper):
    """One qwen call. Returns dict {relevant, zone, one_liner_zh, hot} or None on failure."""
    abstract = paper["abstract"][:2200]
    prompt = (
        f"{ZONE_GUIDE}\n\n"
        f"論文標題：{paper['title']}\n"
        f"摘要：{abstract}\n\n"
        "請輸出嚴格 JSON，字段：\n"
        '{\n'
        '  "relevant": true/false,   // 是否真的關於量化交易/投資（否則 false）\n'
        '  "zone": "上面 10 個 slug 之一",  // relevant=false 時可填最接近的\n'
        '  "one_liner_zh": "≤40字中文一句話，講這篇的核心機制/做法，不要複述摘要、不要英文",\n'
        '  "hot": true/false        // 是否值得日後做深度解構（方法新穎且對讀者有可操作價值）\n'
        "}\n"
        "只輸出這個 JSON。"
    )
    try:
        raw = _qwen.ask(
            prompt, system=CLASSIFY_SYSTEM, thinking=True,
            temperature=0.1, max_tokens=800, json_mode=True,
            retries=4, timeout=180,
        )
    except Exception as e:
        print(f"    qwen error for {paper['id']}: {e}", flush=True)
        return None
    obj = _extract_json(raw)
    if not isinstance(obj, dict):
        print(f"    unparseable qwen reply for {paper['id']}: {str(raw)[:120]!r}", flush=True)
        return None
    zone = str(obj.get("zone", "")).strip()
    if zone not in ZONE_SET:
        # qwen must not invent slugs; drop to evaluation bucket only if relevant
        zone = "evaluation-benchmarks" if obj.get("relevant") else ""
    one = str(obj.get("one_liner_zh", "")).strip()
    one = re.sub(r"\s+", " ", one)[:60]  # hard cap
    return {
        "relevant": bool(obj.get("relevant")),
        "zone": zone,
        "one_liner_zh": one,
        "hot": bool(obj.get("hot")),
    }


# ---------------------------------------------------------------------------
# Week helpers + rendering
# ---------------------------------------------------------------------------
def iso_week_id(dt=None):
    dt = dt or datetime.now(timezone.utc)
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _week_range_str(week_id):
    """From '2026-W29' -> 'YYYY-MM-DD to YYYY-MM-DD' (Mon..Sun of that ISO week)."""
    m = re.match(r"(\d{4})-W(\d{2})", week_id)
    if not m:
        return week_id
    year, week = int(m.group(1)), int(m.group(2))
    try:
        monday = datetime.fromisocalendar(year, week, 1)
    except ValueError:
        return week_id
    sunday = monday + timedelta(days=6)
    return f"{monday:%Y-%m-%d} to {sunday:%Y-%m-%d}"


def _md_escape(s):
    """Escape pipe/brackets so table cells and link text don't break markdown."""
    return (s or "").replace("|", "\\|").replace("[", "\\[").replace("]", "\\]").strip()


def _first_author_et_al(authors):
    if not authors:
        return "—"
    first = _md_escape(authors[0])
    return f"{first} et al." if len(authors) > 1 else first


def _pub_date(item):
    ts = item.get("published") or item.get("updated") or ""
    return ts[:10] if len(ts) >= 10 else (ts or "—")


def render_week_page(week_id, items):
    """items: list of dicts (paper meta + classification). Deterministic output."""
    rng = _week_range_str(week_id)
    relevant = [it for it in items if it.get("relevant") and it.get("zone")]
    n_hot = sum(1 for it in relevant if it.get("hot"))
    lines = []
    lines.append(f"# arXiv Q-FIN Radar — {week_id}")
    lines.append("")
    lines.append(f"**週期 Week:** {rng}  ")
    lines.append(
        f"**本週相關新論文 Relevant papers this week:** {len(relevant)} "
        f"（其中 ⚡ 候選深度解構 hot: {n_hot}）"
    )
    lines.append("")
    lines.append(
        "> arXiv q-fin 每週掃描，qwen 依機制歸入手冊 10 個 zone。"
        "⚡ = 候選深度解構（方法新穎且對讀者有可操作價值）。"
        "一句話為本手冊的分析，非摘要複述。"
    )
    lines.append("")

    if not relevant:
        lines.append("_本週窗口內無相關論文。_")
        lines.append("")
        return "\n".join(lines)

    # group by zone in canonical order
    by_zone = {z: [] for z in ZONES}
    for it in relevant:
        by_zone.setdefault(it["zone"], []).append(it)

    for zone in ZONES:
        rows = by_zone.get(zone, [])
        if not rows:
            continue
        # deterministic order inside a zone: date desc, then id
        rows.sort(key=lambda r: (_pub_date(r), r["id"]), reverse=True)
        hot_n = sum(1 for r in rows if r.get("hot"))
        heading = ZONE_TITLES.get(zone, zone)
        suffix = f" · ⚡{hot_n}" if hot_n else ""
        lines.append(f"## {heading} ({len(rows)}{suffix})")
        lines.append("")
        lines.append("| 日期 Date | 標題 Title | 作者 Authors | 一句話 What it does (機制) |")
        lines.append("|---|---|---|---|")
        for r in rows:
            date = _pub_date(r)
            title = _md_escape(r["title"])
            url = r["abs_url"]
            authors = _first_author_et_al(r.get("authors", []))
            one = _md_escape(r.get("one_liner_zh", "")) or "—"
            if r.get("hot"):
                title_cell = f"**⚡ [{title}]({url})**"
                one = f"**{one}**"
            else:
                title_cell = f"[{title}]({url})"
            lines.append(f"| {date} | {title_cell} | {authors} | {one} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        f"_由 `scripts/pulsar/arxiv_radar.py` 生成 · 更新於 "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}._"
    )
    lines.append("")
    return "\n".join(lines)


def render_overview(week_state):
    """Reverse-chronological index of weekly pages."""
    lines = []
    lines.append("# arXiv Q-FIN Radar")
    lines.append("")
    lines.append(
        "arXiv 的 q-fin（PM/TR/ST/CP/RM/GN）加上經關鍵詞過濾的 cs.LG / stat.ML "
        "量化論文，每週掃描一次，由 qwen 依核心機制歸入本手冊的 10 個 zone。"
        "這讓手冊在 QuantML 語料之外多了一條前瞻性的 arXiv 進料。"
    )
    lines.append("")
    lines.append("⚡ = 候選深度解構（method 新穎且對讀者有可操作價值，進 dissection 佇列）。")
    lines.append("")
    lines.append("## 每週雷達 Weekly pages")
    lines.append("")
    week_ids = sorted(week_state.keys(), reverse=True)
    if not week_ids:
        lines.append("_尚無週報。_")
    else:
        for wid in week_ids:
            st = week_state[wid]
            n_rel = st.get("n_relevant", 0)
            n_hot = st.get("n_hot", 0)
            rng = _week_range_str(wid)
            hot_txt = f", ⚡{n_hot}" if n_hot else ""
            lines.append(f"- [{wid}](./{wid}.md) — {rng} · {n_rel} papers{hot_txt}")
    lines.append("")
    lines.append(
        f"_索引由 `scripts/pulsar/arxiv_radar.py` 維護 · 更新於 "
        f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="arXiv Q-FIN weekly radar")
    ap.add_argument("--days", type=int, default=8, help="lookback window in days (default 8)")
    ap.add_argument("--max-classify", type=int, default=80,
                    help="cap new papers classified per run (default 80)")
    ap.add_argument("--dry-run", action="store_true",
                    help="fetch+classify but do not write seen/queue/pages")
    args = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs(RADAR_DIR, exist_ok=True)

    seen = set(_load_json(SEEN_PATH, []))
    week_state = _load_json(WEEK_STATE_PATH, {})
    queue = _load_json(QUEUE_PATH, [])
    queue_ids = {q.get("id") for q in queue}

    week_id = iso_week_id()
    print(f"=== arXiv Q-FIN Radar :: week {week_id} :: days={args.days} ===", flush=True)

    # 1+2. FETCH then DEDUPE
    papers = fetch_papers(args.days)
    n_fetched = len(papers)
    new_papers = [p for p in papers if p["id"] not in seen]
    print(f"[dedupe] fetched={n_fetched} already-seen={n_fetched - len(new_papers)} "
          f"new={len(new_papers)}", flush=True)

    # cap classification budget
    to_classify = new_papers[:args.max_classify]
    if len(new_papers) > args.max_classify:
        print(f"[classify] capping to {args.max_classify} (of {len(new_papers)} new)", flush=True)

    # 3. CLASSIFY
    classified = []  # full item dicts (paper + classification)
    n_relevant = 0
    zone_counts = {z: 0 for z in ZONES}
    newly_hot = []
    processed_ids = []
    for i, p in enumerate(to_classify, 1):
        print(f"[classify] {i}/{len(to_classify)} {p['id']} :: {p['title'][:70]}", flush=True)
        cls = classify_paper(p)
        if cls is None:
            # do NOT mark seen -> will retry next run
            continue
        processed_ids.append(p["id"])
        item = {
            "id": p["id"],
            "title": p["title"],
            "abs_url": p["abs_url"],
            "authors": p.get("authors", []),
            "published": p.get("published", ""),
            "updated": p.get("updated", ""),
            "primary_category": p.get("primary_category", ""),
            "relevant": cls["relevant"],
            "zone": cls["zone"],
            "one_liner_zh": cls["one_liner_zh"],
            "hot": cls["hot"],
        }
        classified.append(item)
        if item["relevant"] and item["zone"]:
            n_relevant += 1
            zone_counts[item["zone"]] = zone_counts.get(item["zone"], 0) + 1
            if item["hot"]:
                newly_hot.append(item)
        rel_txt = f"zone={item['zone']}" if item["relevant"] else "IRRELEVANT"
        print(f"    -> relevant={item['relevant']} {rel_txt} "
              f"hot={item['hot']} :: {item['one_liner_zh']}", flush=True)

    print(f"[classify] done: processed={len(processed_ids)} relevant={n_relevant} "
          f"newly-hot={len(newly_hot)}", flush=True)

    if args.dry_run:
        print("[dry-run] skipping writes.", flush=True)
        _print_summary(n_fetched, len(new_papers), n_relevant, zone_counts, len(newly_hot))
        return

    # 5. Merge into per-week state (idempotent: keyed by paper id).
    wk = week_state.get(week_id, {"items": {}, "n_relevant": 0, "n_hot": 0})
    if "items" not in wk:
        wk["items"] = {}
    for item in classified:
        wk["items"][item["id"]] = item  # overwrite is deterministic
    # recompute week rollups from stored items
    rel_items = [it for it in wk["items"].values() if it.get("relevant") and it.get("zone")]
    wk["n_relevant"] = len(rel_items)
    wk["n_hot"] = sum(1 for it in rel_items if it.get("hot"))
    wk["updated"] = datetime.now(timezone.utc).isoformat()
    week_state[week_id] = wk

    # 4. PUBLISH weekly page (rebuild from full stored week state -> deterministic)
    page_items = list(wk["items"].values())
    page_md = render_week_page(week_id, page_items)
    page_path = os.path.join(RADAR_DIR, f"{week_id}.md")
    _atomic_write_text(page_path, page_md)
    print(f"[publish] wrote {page_path} ({wk['n_relevant']} relevant / {wk['n_hot']} hot)", flush=True)

    # overview index
    overview_md = render_overview(week_state)
    overview_path = os.path.join(RADAR_DIR, "overview.md")
    _atomic_write_text(overview_path, overview_md)
    print(f"[publish] wrote {overview_path}", flush=True)

    # append hot items to queue (metadata only, dedup by id)
    added_q = 0
    for it in newly_hot:
        if it["id"] in queue_ids:
            continue
        queue.append({
            "id": it["id"],
            "title": it["title"],
            "zone": it["zone"],
            "date": _pub_date(it),
            "abs_url": it["abs_url"],
            "week": week_id,
        })
        queue_ids.add(it["id"])
        added_q += 1

    # persist state (mark processed ids as seen only after successful classify)
    for pid in processed_ids:
        seen.add(pid)
    _atomic_write_json(SEEN_PATH, sorted(seen))
    _atomic_write_json(WEEK_STATE_PATH, week_state)
    _atomic_write_json(QUEUE_PATH, queue)
    print(f"[state] seen={len(seen)} week-items={len(wk['items'])} "
          f"queue={len(queue)} (+{added_q} hot)", flush=True)

    _print_summary(n_fetched, len(new_papers), wk["n_relevant"], zone_counts, wk["n_hot"])
    print(f"[done] page: {page_path}", flush=True)


def _print_summary(n_fetched, n_new, n_relevant, zone_counts, n_hot):
    print("\n========== SUMMARY ==========", flush=True)
    print(f"fetched (window)   : {n_fetched}", flush=True)
    print(f"new (not seen)     : {n_new}", flush=True)
    print(f"relevant (this run): {n_relevant}", flush=True)
    print(f"hot (this run)     : {n_hot}", flush=True)
    print("per-zone (this run):", flush=True)
    for z in ZONES:
        c = zone_counts.get(z, 0)
        if c:
            print(f"  {z:32s} {c}", flush=True)
    print("=============================", flush=True)


if __name__ == "__main__":
    main()
