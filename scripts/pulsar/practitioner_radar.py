#!/usr/bin/env python3
"""Practitioner Radar — weekly quant-practitioner blog/institute digest.

Aggregates curated RSS/Atom feeds (scripts/pulsar/radar_feeds.json), dedupes,
qwen-classifies each fresh item into a handbook zone + one-line Chinese take,
and publishes radar/practitioner/<ISO-year>-W<week>.md (links + OUR takes only —
never article body text; copyright-safe).

Design:
  * stdlib only, python3.11, resumable.
  * Per-feed try/except — one dead feed never kills the run.
  * Dedupe set persisted at data/radar_prac_seen.json (link-hash).
  * Classified items persisted per ISO week at data/radar_prac_state.json so a
    same-week rerun regenerates the page deterministically (idempotent) without
    re-spending qwen calls.
  * qwen: one call per NEW item, json_mode, temperature 0.1, hard-capped 60/run.

Usage:
  python3.11 practitioner_radar.py [--days 8] [--cap 60] [--week 2026-W29]
                                   [--dry-run] [--no-llm]
"""
import argparse
import datetime as dt
import hashlib
import html
import json
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _qwen import ask  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
FEEDS_FILE = HERE / "radar_feeds.json"
DATA = ROOT / "data"
SEEN_FILE = DATA / "radar_prac_seen.json"          # gitignored (data/*)
STATE_FILE = DATA / "radar_prac_state.json"        # gitignored (data/*)
OUT_DIR = ROOT / "radar" / "practitioner"          # published pages

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# 10 canonical handbook zones (foundations/<slug>) + Chinese labels + one-line scope.
ZONES = [
    ("time-series-forecasting",     "时序预测",     "价格/收益率/波动率序列建模，Transformer/RNN/状态空间/频域架构"),
    ("graph-networks",              "图网络",       "股票关系图/产业链/共现图，异构图与动态图做预测与组合"),
    ("reinforcement-learning",      "强化学习",     "DRL/MARL 做选股调仓做市执行对冲，奖励设计与环境模拟"),
    ("llm-agentic",                 "大模型智能体", "LLM 做因子挖掘/情绪/研报/RAG 与多智能体交易工作流"),
    ("market-microstructure",       "微观高频",     "限价订单簿/订单流/做市/高频套利/流动性与价格形成"),
    ("factor-mining",               "因子挖掘",     "公式型 Alpha/AutoML/风格因子/信号衰减与拥挤度"),
    ("portfolio-optimization",      "组合优化",     "均值方差/风险平价/BL/执行算法/多资产轮动配置"),
    ("causal-structural",           "因果结构",     "因果发现/反事实/结构 VAR/事件对价格的因果路径"),
    ("evaluation-benchmarks",       "评测基准",     "回测防穿越/样本外/过拟合检测/鲁棒性/基准与衰减归因"),
    ("data-generation-augmentation","数据生成",     "GAN/扩散合成金融时序/插补/增强/分布外泛化"),
]
ZONE_SLUGS = {z[0] for z in ZONES}
ZONE_NAME = {z[0]: z[1] for z in ZONES}
ZONE_NAME["meta"] = "行业/职业/工具"  # non-zone bucket
ZONE_ORDER = [z[0] for z in ZONES] + ["meta"]

CLASSIFY_SYSTEM = (
    "你是量化交易手册的策展编辑。只根据给定的标题与来源，判断这条博客/机构文章是否是"
    "真正的量化交易/投资研究内容（不是产品营销/招聘/公司公告/纯新闻）。"
    "然后把它归入下面某一个方法族 zone，或 meta（行业动态/职业/工具，不属于具体方法族）。"
    "只输出一个 JSON 对象，不要解释。"
)


# ---------- persistence ----------
def load_json(p, default):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(p)


def link_hash(url):
    return hashlib.sha1((url or "").strip().lower().encode("utf-8")).hexdigest()[:16]


# ---------- feed fetch + parse ----------
def strip_ns(tag):
    return tag.split("}")[-1].lower() if tag else ""


def _text(el):
    return (el.text or "").strip() if el is not None else ""


def _find_child(entry, names):
    for c in entry:
        if strip_ns(c.tag) in names:
            return c
    return None


def _find_all(entry, names):
    return [c for c in entry if strip_ns(c.tag) in names]


def parse_date(s):
    """Best-effort date parse -> date object (or None)."""
    if not s:
        return None
    s = s.strip()
    fmts = [
        "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M %z", "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
    ]
    for f in fmts:
        try:
            d = dt.datetime.strptime(s, f)
            return d.date()
        except ValueError:
            continue
    # ISO 8601 fallback (handles fractional seconds / offsets like +00:00)
    try:
        s2 = s.replace("Z", "+00:00")
        return dt.datetime.fromisoformat(s2).date()
    except Exception:
        return None


def extract_link(entry):
    """Pick the best outbound/canonical link for an entry.

    Aggregators (Quantocracy) link OUT to the original blog. WordPress feeds
    expose that outbound target in <feedburner:origLink> / <link> is the
    aggregator permalink. We prefer, in order:
      1) Atom <link rel="alternate"> / first Atom link with href
      2) feedburner:origLink (outbound)
      3) plain <link> text (RSS2)
    """
    # Atom-style <link href=...>
    atom_alt = None
    atom_any = None
    for c in entry:
        if strip_ns(c.tag) == "link":
            href = c.attrib.get("href")
            rel = c.attrib.get("rel", "alternate")
            if href:
                if rel == "alternate" and atom_alt is None:
                    atom_alt = href
                if atom_any is None:
                    atom_any = href
    orig = _find_child(entry, {"origlink"})  # feedburner:origLink
    plain = None
    for c in entry:
        if strip_ns(c.tag) == "link" and (c.text or "").strip():
            plain = c.text.strip()
            break
    for cand in (plain, orig.text.strip() if orig is not None and orig.text else None,
                 atom_alt, atom_any):
        if cand and cand.startswith("http"):
            return cand
    return atom_alt or atom_any or plain or ""


def clean_title(t):
    t = html.unescape(t or "").strip()
    t = re.sub(r"\s+", " ", t)
    # feeds sometimes append " [PREMIUM]" / trailing markers — keep as-is (informative)
    return t


def fetch_feed(feed, days):
    """Return list of {feed_id, feed_name, title, link, date} within `days`.

    Never raises: per-feed failures are caught and logged.
    """
    url = feed["url"]
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": UA,
                          "Accept": "application/rss+xml,application/atom+xml,application/xml,text/xml,*/*"})
        with urllib.request.urlopen(req, timeout=20) as r:
            raw = r.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  FEED-FAIL {feed['id']:<18} {type(e).__name__}: {e}", file=sys.stderr)
        return [], "fetch-fail"
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"  PARSE-FAIL {feed['id']:<18} {e}", file=sys.stderr)
        return [], "parse-fail"

    cutoff = dt.date.today() - dt.timedelta(days=days)
    items = []
    # iterate any <item> (RSS2 / RSS1-RDF) and <entry> (Atom) anywhere in tree
    for entry in root.iter():
        tag = strip_ns(entry.tag)
        if tag not in ("item", "entry"):
            continue
        title_el = _find_child(entry, {"title"})
        title = clean_title(_text(title_el))
        if not title:
            continue
        link = extract_link(entry)
        if not link:
            continue
        # date fields: pubDate / dc:date / published / updated / date
        date_el = _find_child(entry, {"pubdate", "date", "published", "updated"})
        d = parse_date(_text(date_el))
        # keep if within window OR (no date -> pass through, conservative)
        if d is not None and d < cutoff:
            continue
        items.append({
            "feed_id": feed["id"],
            "feed_name": feed["name"],
            "title": title,
            "link": link.strip(),
            "date": d.isoformat() if d else "",
        })
    return items, "ok"


import signal


def _alarm_raise(_s, _f):
    raise TimeoutError("qwen hard wall-clock timeout")


# ---------- qwen classify ----------
def build_classify_prompt(item):
    zlines = "\n".join(f"  - {slug} ({name}): {scope}" for slug, name, scope in ZONES)
    return (
        f"文章标题: {item['title']}\n"
        f"来源博客/机构: {item['feed_name']}\n\n"
        f"可选 zone（方法族）:\n{zlines}\n"
        f"  - meta (行业/职业/工具): 行业动态、职业发展、工具/数据平台、访谈，不属于具体方法族\n\n"
        "输出 JSON:\n"
        "{\n"
        '  "relevant": true/false,   // 是否为真正的量化交易/投资研究内容（排除产品营销/招聘/公告/纯新闻）\n'
        '  "zone": "上面某个 slug 或 meta",\n'
        '  "one_liner_zh": "≤40字中文，讲清这条内容的机制/要点，不要复述标题",\n'
        '  "must_read": true/false   // 是否对实盘或方法论有较强启发、值得本周优先读\n'
        "}"
    )


def parse_json(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[-1] if t.count("```") >= 2 else t.strip("`")
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    i, j = t.find("{"), t.rfind("}")
    return json.loads(t[i:j + 1])


def classify(item):
    old = signal.signal(signal.SIGALRM, _alarm_raise)
    signal.alarm(85)  # hard bound: abort a black-holed DashScope connection that timeout=60 misses
    try:
        raw = ask(build_classify_prompt(item), system=CLASSIFY_SYSTEM, thinking=False,
                  temperature=0.1, max_tokens=220, json_mode=True, timeout=55, retries=1)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    d = parse_json(raw)
    zone = d.get("zone", "meta")
    if zone not in ZONE_SLUGS and zone != "meta":
        zone = "meta"
    ol = (d.get("one_liner_zh") or "").strip()
    if len(ol) > 60:
        ol = ol[:58] + "…"
    return {
        "relevant": bool(d.get("relevant", False)),
        "zone": zone,
        "one_liner_zh": ol,
        "must_read": bool(d.get("must_read", False)),
    }


# ---------- ISO week helpers ----------
def iso_week_key(d=None):
    d = d or dt.date.today()
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def week_range(week_key):
    """Return (monday_date, sunday_date) for an ISO week key like 2026-W29."""
    y, w = week_key.split("-W")
    y, w = int(y), int(w)
    monday = dt.date.fromisocalendar(y, w, 1)
    sunday = dt.date.fromisocalendar(y, w, 7)
    return monday, sunday


# ---------- markdown rendering ----------
def md_escape_cell(s):
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def render_weekly(week_key, items, feeds):
    """items: list of enriched dicts (relevant only). Deterministic ordering."""
    mon, sun = week_range(week_key)
    lines = []
    lines.append(f"# 实践者雷达 · {week_key}")
    lines.append("")
    lines.append(f"> 覆盖 {mon.isoformat()} ~ {sun.isoformat()} · "
                 "量化从业者博客/机构精选 · 仅标题+链接+我们的一句话解读（不转载正文）")
    lines.append("")

    # per-source counts (deterministic: by feed order in config, then id)
    fid_order = {f["id"]: i for i, f in enumerate(feeds)}
    counts = {}
    for it in items:
        counts[it["feed_id"]] = counts.get(it["feed_id"], 0) + 1
    if counts:
        parts = []
        for fid in sorted(counts, key=lambda k: (fid_order.get(k, 999), k)):
            name = next((f["name"] for f in feeds if f["id"] == fid), fid)
            parts.append(f"{name} {counts[fid]}")
        lines.append("**本週來源計數**: " + " · ".join(parts))
        lines.append("")

    # deterministic sort inside any group: date desc (empty last), then title
    def sort_key(it):
        return (it.get("date", "") == "",
                _neg_date(it.get("date", "")), it["title"].lower())

    # ⚡ 本週必讀
    must = [it for it in items if it.get("must_read")]
    must.sort(key=sort_key)
    if must:
        lines.append("## ⚡ 本週必讀")
        lines.append("")
        lines.append("| 来源 | 标题 | 一句话解读 |")
        lines.append("|---|---|---|")
        for it in must:
            zlabel = ZONE_NAME.get(it["zone"], it["zone"])
            lines.append(
                f"| {md_escape_cell(it['feed_name'])} · {zlabel} "
                f"| [{md_escape_cell(it['title'])}]({it['link']}) "
                f"| {md_escape_cell(it['one_liner_zh'])} |")
        lines.append("")

    # by-zone sections
    by_zone = {}
    for it in items:
        by_zone.setdefault(it["zone"], []).append(it)
    for zone in ZONE_ORDER:
        group = by_zone.get(zone)
        if not group:
            continue
        group.sort(key=sort_key)
        zlabel = ZONE_NAME.get(zone, zone)
        anchor = "🧭 " if zone == "meta" else ""
        lines.append(f"## {anchor}{zlabel}")
        lines.append("")
        lines.append("| 来源 | 标题 | 一句话解读 |")
        lines.append("|---|---|---|")
        for it in group:
            lines.append(
                f"| {md_escape_cell(it['feed_name'])} "
                f"| [{md_escape_cell(it['title'])}]({it['link']}) "
                f"| {md_escape_cell(it['one_liner_zh'])} |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*版权说明：本页仅收录公开 RSS 的标题与链接，加上本手册的一句话解读；"
                 "正文版权归原作者。索引见 [overview](/radar/practitioner/overview)。*")
    lines.append("")
    return "\n".join(lines)


def _neg_date(s):
    """Return a sortable value so newer dates sort first."""
    return tuple(-int(x) for x in s.replace("-", " ").split()) if s else (0,)


def render_overview(feeds, weeks_index):
    lines = []
    lines.append("# 实践者雷达")
    lines.append("")
    lines.append("每週聚合一批量化從業者博客與機構研究的 RSS，qwen 按十大方法族 zone 分類，"
                 "產出一頁分類週報。**只放標題、鏈接與我們的一句話解讀**，從不轉載正文（版權歸原作者）。"
                 "數據源以 `scripts/pulsar/radar_feeds.json` 配置，可增減。")
    lines.append("")
    lines.append("## 訂閱源")
    lines.append("")
    lines.append("| 来源 | 权重 | 链接 |")
    lines.append("|---|---|---|")
    for f in feeds:
        w = "核心" if f.get("weight") == "core" else "扩展"
        lines.append(f"| {md_escape_cell(f['name'])} | {w} | [{md_escape_cell(f['url'])}]({f['url']}) |")
    lines.append("")
    lines.append("## 週報索引")
    lines.append("")
    if weeks_index:
        for wk in weeks_index:  # already reverse-chron
            mon, sun = week_range(wk)
            lines.append(f"- [{wk}](/radar/practitioner/{wk}) — {mon.isoformat()} ~ {sun.isoformat()}")
    else:
        lines.append("*（暂无周报）*")
    lines.append("")
    return "\n".join(lines)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8, help="lookback window for feed entries")
    ap.add_argument("--cap", type=int, default=60, help="max qwen classify calls this run")
    ap.add_argument("--week", default=None, help="ISO week key override, e.g. 2026-W29")
    ap.add_argument("--dry-run", action="store_true", help="fetch+classify but don't write pages")
    ap.add_argument("--no-llm", action="store_true", help="skip qwen (only reuse cached state)")
    args = ap.parse_args()

    feeds = json.loads(FEEDS_FILE.read_text())
    week_key = args.week or iso_week_key()
    print(f"Practitioner Radar · week={week_key} · days={args.days} · cap={args.cap} · feeds={len(feeds)}")

    # 1) fetch all feeds
    all_items = []
    feed_status = {}
    for f in feeds:
        got, status = fetch_feed(f, args.days)
        feed_status[f["id"]] = (status, len(got))
        print(f"  {status:<10} {f['id']:<18} raw={len(got)}")
        all_items.extend(got)
    print(f"Fetched {len(all_items)} raw items across {len(feeds)} feeds")

    # 2) dedupe against global seen-set (link-hash)
    seen = set(load_json(SEEN_FILE, []))
    state = load_json(STATE_FILE, {})  # {week_key: [enriched items]}
    week_state = state.get(week_key, [])
    already_classified = {it["link"] for it in week_state}

    fresh = []
    for it in all_items:
        h = link_hash(it["link"])
        if it["link"] in already_classified:
            continue  # already in this week's state (idempotent rerun)
        if h in seen:
            continue  # seen in a prior run/week
        fresh.append(it)
    # de-dupe within this batch by link (feeds can echo each other)
    uniq = {}
    for it in fresh:
        uniq.setdefault(it["link"], it)
    fresh = list(uniq.values())
    print(f"Fresh (unseen, unclassified): {len(fresh)}")

    # 3) classify up to cap
    classified = []
    used = 0
    if not args.no_llm:
        for it in fresh:
            if used >= args.cap:
                print(f"  cap {args.cap} reached — deferring {len(fresh) - used} items to next run")
                break
            try:
                c = classify(it)
                used += 1
                enriched = {**it, **c}
                # mark seen only after successful classification (so failures retry next run)
                seen.add(link_hash(it["link"]))
                if c["relevant"]:
                    classified.append(enriched)
                    flag = "⚡" if c["must_read"] else "  "
                    print(f"  {flag} [{c['zone']:>24}] {it['feed_name'][:14]:<14} | {it['title'][:40]}")
                else:
                    print(f"     [ irrelevant, dropped   ] {it['feed_name'][:14]:<14} | {it['title'][:40]}")
            except Exception as e:
                print(f"  CLASSIFY-FAIL {it['feed_name'][:14]} :: {e}", file=sys.stderr)
        print(f"qwen calls used: {used}")

    # 4) merge into week state (relevant items only), dedupe by link, deterministic
    merged = {it["link"]: it for it in week_state}
    for it in classified:
        merged[it["link"]] = it
    week_items = list(merged.values())
    state[week_key] = week_items

    relevant_n = len(week_items)
    must_n = sum(1 for it in week_items if it.get("must_read"))
    print(f"Week {week_key}: relevant={relevant_n} must_read={must_n} (this run classified {len(classified)} new relevant)")

    if args.dry_run:
        print("--dry-run: not writing pages / state")
        return

    # 5) persist seen + state
    save_json(SEEN_FILE, sorted(seen))
    save_json(STATE_FILE, state)

    # 6) render weekly page (idempotent — full regenerate from week_items)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page = render_weekly(week_key, week_items, feeds)
    (OUT_DIR / f"{week_key}.md").write_text(page)
    print(f"Wrote {OUT_DIR / (week_key + '.md')} ({len(page)} bytes)")

    # 7) rebuild overview with reverse-chron index of all weekly files present
    week_files = sorted(
        (p.stem for p in OUT_DIR.glob("*-W*.md")),
        reverse=True)
    overview = render_overview(feeds, week_files)
    (OUT_DIR / "overview.md").write_text(overview)
    print(f"Wrote {OUT_DIR / 'overview.md'} (index: {len(week_files)} weeks)")


if __name__ == "__main__":
    main()
