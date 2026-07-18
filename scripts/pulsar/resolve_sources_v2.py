#!/usr/bin/env python3
"""Source resolver v2 — CROSSREF backend (verified un-throttled from this datacenter IP,
unlike Semantic Scholar / OpenAlex which 429 hard). Links every dissection page to its real
paper + citation count. Crossref polite pool returns title / DOI / is-referenced-by-count /
container-title(venue) / published(year).

Per page-worthy record (corpus_meta.json + the page's ## References English title):
  - build query = English title from References (primary) else framework name
  - Crossref query.bibliographic search (rows=3), accept if title Jaccard>=0.55 OR framework
    token appears in the Crossref title (a wrong link is worse than none)
  - link = arXiv abs (if corpus_meta.arxiv present) else https://doi.org/<DOI>
Resumable (skips already crossref-resolved). Persists every 10. ~0.6s spacing + 429 backoff.

Run:  python3.11 scripts/pulsar/resolve_sources_v2.py            # resolve
      python3.11 scripts/pulsar/resolve_sources_v2.py --inject   # inject header line (idempotent)
      (flags combine)
"""
import json, re, time, sys, urllib.request, urllib.parse, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
META = ROOT / "data" / "corpus_meta.json"
LINKS = ROOT / "data" / "source_links.json"
MAILTO = "pulsar-research@example.com"
CR = "https://api.crossref.org/works"
UA = "PulsarResearch/1.0 (mailto:%s)" % MAILTO
STOP = set("the a an of for to and or with via using on in from into over under is are model models "
           "framework approach method learning based deep neural network networks new novel toward "
           "towards study analysis prediction predicting forecasting trading stock financial market data".split())
ARXIV_RE = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})", re.I)
QUANTML_LINE = re.compile(r"^> \*\*QuantML 導讀\*\*.*$", re.M)
# credible CS / finance / ML venues — the real discriminator against bio/chem/junk false hits
VENUE_RE = re.compile(r"SIGKDD|Knowledge Discovery|NeurIPS|Neural Information|\bICML\b|\bICLR\b|\bAAAI\b|IJCAI|\bVLDB\b|\bCIKM\b|\bWSDM\b|Web Conference|\bWWW\b|SSRN|Electronic Journal|Findings of the Association|Empirical Methods|Computational Linguistics|\bIEEE\b|\bACM\b|Journal of Financ|Financial|Quantitative Financ|Portfolio|Journal of Bank|Econom|Forecast|Expert Systems|Knowledge.Based|Information Sciences|Machine Learning|Data Mining|Neurocomputing|Pattern Recognition|Applied Soft Comput|Decision Support|Review of Financ|Asset Pricing|arXiv|Preprint|Proceedings", re.I)


def _get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=25) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except Exception as e:
            code = getattr(e, "code", 0)
            if code in (429, 500, 502, 503) or code == 0:
                time.sleep(4 * (i + 1)); continue
            return None
    return None


def _norm(s):
    return set(w for w in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if w not in STOP and len(w) > 1)


def _jaccard(a, b):
    A, B = _norm(a), _norm(b)
    return len(A & B) / len(A | B) if (A and B) else 0.0


def _english_titles(text):
    i = text.rfind("## References")
    tail = text[i:] if i >= 0 else text[-1400:]
    cand = []
    m = re.search(r"原論文[:：]\s*([^\n（(]+)", tail)
    if m:
        cand.append(m.group(1))
    for line in tail.splitlines():
        line = re.sub(r"^[\-\*\d\.\s>]+", "", line).strip()
        if any(x in line for x in ("QuantML", "導讀", "mp.weixin", "Lineage")):
            continue
        letters = sum(c.isascii() and c.isalpha() for c in line)
        if letters >= 15 and letters >= 0.5 * max(1, len(line)) and 15 < len(line) < 170:
            cand.append(line)
    out = []
    for c in cand:
        c = re.sub(r"\(arXiv[^)]*\)|arXiv:\s*\S+|\((?:KDD|ICML|ICLR|NeurIPS|AAAI|WWW|CIKM|VLDB|JF|JFE|RFS)[^)]*\)|\(\d{4}\)", "", c, flags=re.I)
        c = re.sub(r"\s+", " ", c).strip(" .,:-—、")
        if len(c) > 12 and c not in out:
            out.append(c)
    return out[:2]


def cr_search(query):
    u = ("%s?query.bibliographic=%s&rows=3&select=title,DOI,is-referenced-by-count,published,container-title&mailto=%s"
         % (CR, urllib.parse.quote(query[:250]), MAILTO))
    r = _get(u)
    return (r or {}).get("message", {}).get("items", []) or []


def resolve():
    meta = json.loads(META.read_text())
    links = json.loads(LINKS.read_text()) if LINKS.exists() else {}
    recs = [r for r in meta if r.get("page_worthy") and r.get("rating") in ("⚡", "🔧") and not r.get("dup_of")]
    todo = [r for r in recs if not str(links.get(r["msgid"], {}).get("matched_by", "")).startswith("crossref")]
    print("records=%d crossref-done=%d todo=%d" % (len(recs), len(recs) - len(todo), len(todo)), flush=True)
    done = 0
    for n, r in enumerate(todo):
        p = ROOT / "foundations" / r["zone"] / ("%s.md" % r["slug"])
        text = p.read_text(encoding="utf-8") if p.exists() else ""
        fw = (r.get("framework") or "").strip()
        axid = r.get("arxiv") if (r.get("arxiv") and re.match(r"^\d{4}\.\d{4,5}$", str(r.get("arxiv")).strip())) else None
        if not axid:
            mm = ARXIV_RE.search(text)
            axid = mm.group(1) if mm else None
        fw_l = fw.lower()
        # (query, is_real_title): References English titles are trustworthy for Jaccard; a bare
        # framework acronym is NOT (framework==title gives jac=1.0 for coincidental bio/chem hits).
        queries = [(q, True) for q in _english_titles(text)]
        if fw and fw_l not in ("null", "none"):
            queries.append((fw, False))
        hit = None
        for q, is_title in queries:
            for it in cr_search(q):
                t = (it.get("title") or [""])[0]
                if not t:
                    continue
                jac = _jaccard(q, t)
                ven = (it.get("container-title") or [""])[0]
                fw_in_t = fw and len(fw) > 2 and fw_l in t.lower()
                ven_ok = bool(VENUE_RE.search(ven))
                # accept: (a) a strong multi-word title match — trust it regardless of venue; OR
                #         (b) framework name in title BUT only if the venue is a credible CS/finance
                #             venue (kills TimeView->Applied Bioinformatics, TFB->Catalysis, etc.)
                strong = is_title and len(q.split()) >= 4 and jac >= 0.55
                framework_credible = fw_in_t and ven_ok and (jac >= 0.30 or not is_title)
                if strong or framework_credible:
                    doi = it.get("DOI")
                    yr = (((it.get("published") or {}).get("date-parts") or [[None]])[0] or [None])[0]
                    hit = {"status": "resolved", "matched_by": "crossref" + ("+fw" if not strong else ""),
                           "title": t, "arxiv": axid,
                           "url": ("https://arxiv.org/abs/%s" % axid) if axid else ("https://doi.org/%s" % doi if doi else None),
                           "doi": doi, "venue": ven or None, "year": yr, "citations": it.get("is-referenced-by-count")}
                    break
                time.sleep(0.05)
            if hit:
                break
            time.sleep(0.5)
        if hit and hit["url"]:
            hit.update(slug=r["slug"], zone=r["zone"], framework=fw or None)
            links[r["msgid"]] = hit; done += 1
            print("  [%d/%d] OK %-26s cite=%s %s" % (n + 1, len(todo), r["slug"][:26], hit.get("citations"), (hit["venue"] or "")[:30]), flush=True)
        else:
            links[r["msgid"]] = {"status": "unresolved", "matched_by": "crossref", "slug": r["slug"], "zone": r["zone"]}
        if (n + 1) % 10 == 0:
            LINKS.write_text(json.dumps(links, ensure_ascii=False, indent=2))
            print("  ...persisted at %d (resolved so far %d)" % (n + 1, sum(1 for v in links.values() if v.get("status") == "resolved")), flush=True)
    LINKS.write_text(json.dumps(links, ensure_ascii=False, indent=2))
    tot = sum(1 for v in links.values() if v.get("status") == "resolved")
    print("DONE newly=%d total_resolved=%d/%d" % (done, tot, len(recs)), flush=True)


def inject():
    links = json.loads(LINKS.read_text()) if LINKS.exists() else {}
    n = 0
    for v in links.values():
        if v.get("status") != "resolved" or not v.get("slug") or not v.get("url"):
            continue
        p = ROOT / "foundations" / v["zone"] / ("%s.md" % v["slug"])
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if "**原始論文**" in text:
            continue
        m = QUANTML_LINE.search(text)
        if not m:
            continue
        bits = [v.get("venue"), str(v["year"]) if v.get("year") else None,
                ("被引 %s" % v["citations"]) if v.get("citations") is not None else None]
        meta_str = " · ".join(b for b in bits if b)
        line = "> **原始論文**：[%s](%s)%s" % (v.get("title") or "原文", v["url"],
                                             ("（%s · Crossref）" % meta_str) if meta_str else "")
        p.write_text(text[:m.end()] + "\n" + line + text[m.end():], encoding="utf-8")
        n += 1
    print("injected=%d pages" % n, flush=True)


if __name__ == "__main__":
    if "--inject" not in sys.argv or "--resolve" in sys.argv:
        resolve()
    if "--inject" in sys.argv:
        inject()
