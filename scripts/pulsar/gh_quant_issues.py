#!/usr/bin/env python3
"""GitHub 工程經驗層 — 掃 11 個核心 quant 開源庫的高互動 issues，qwen 蒸餾成「坑→解法」
工程筆記，逐條帶 issue 連結（可追溯）。這是論文層之外的第三種信息源：真實工程經驗。

Modeled on the proven VLA-Handbook gh-issues sensor. Weekly append-only: only NEW qualifying
issues are distilled and appended to each repo's section; the page keeps history.

Auth: token is read at runtime from this repo's own git remote URL (embedded PAT) — never
stored in the script or committed anywhere.

Usage: python3.11 gh_quant_issues.py [--repo owner/name] [--dry] [--init]
"""
import json, re, sys, time, signal, pathlib, argparse, subprocess, urllib.request, urllib.parse, datetime as dt

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _qwen import ask

ROOT = pathlib.Path(__file__).resolve().parents[2]
PAGE = ROOT / "deployment" / "github_engineering_notes.md"
SEEN_F = ROOT / "data" / "gh_issues_seen.json"

# min = comment threshold (per-repo noise floor: huge communities need a higher bar)
REPOS = [
    {"repo": "microsoft/qlib",                "min": 12, "desc": "微軟量化平台（因子/模型/回測全棧）"},
    {"repo": "AI4Finance-Foundation/FinRL",   "min": 10, "desc": "RL 交易框架（學術系主流基線）"},
    {"repo": "polakowo/vectorbt",             "min": 10, "desc": "向量化回測（速度取向）"},
    {"repo": "nautechsystems/nautilus_trader","min": 10, "desc": "事件驅動生產級交易引擎"},
    {"repo": "QuantConnect/Lean",             "min": 10, "desc": "QuantConnect 開源引擎（實盤+回測）"},
    {"repo": "kernc/backtesting.py",          "min": 10, "desc": "輕量回測庫"},
    {"repo": "pst-group/pysystemtrade",       "min": 6,  "desc": "Rob Carver 系統化期貨（實盤跑真錢）"},
    {"repo": "freqtrade/freqtrade",           "min": 30, "desc": "加密貨幣自動交易（最大社區）"},
    {"repo": "hummingbot/hummingbot",         "min": 15, "desc": "做市機器人（execution/microstructure）"},
    {"repo": "stefan-jansen/machine-learning-for-trading", "min": 6, "desc": "ML4T 書配套代碼（學習者踩坑集中地）"},
    {"repo": "OpenBB-finance/OpenBB",         "min": 25, "desc": "開源投研終端（數據工程的坑）"},
]


def gh_token():
    url = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], cwd=ROOT).decode()
    m = re.search(r"(ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+)", url)
    if not m:
        raise RuntimeError("no embedded token in remote url")
    return m.group(1)


def gh_json(url, token=None, timeout=25, retries=3):
    """token=None -> unauthenticated. Search MUST be unauth here: the classic PAT oddly fails
    /search with a validation error, while anonymous search works (10 req/min is plenty for
    11 repos). Issue/comment reads use the token (5000/hr)."""
    last = None
    for i in range(retries):
        try:
            headers = {"Accept": "application/vnd.github+json",
                       "User-Agent": "Trading-Handbook engineering sensor"}
            if token:
                headers["Authorization"] = f"token {token}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:
            last = e
            time.sleep(4 * (i + 1))
    raise last


def top_issues(repo, min_comments, n=10):
    q = urllib.parse.quote(f"repo:{repo} is:issue comments:>={min_comments}")
    d = gh_json(f"https://api.github.com/search/issues?q={q}&sort=comments&order=desc&per_page={n}")
    time.sleep(7)   # anonymous search quota: 10/min
    return d.get("items", [])


def issue_bundle(repo, num, token, body, max_comments=4):
    """issue body + first comments, trimmed — the raw material qwen distills from."""
    parts = [f"BODY:\n{(body or '')[:1500]}"]
    try:
        cs = gh_json(f"https://api.github.com/repos/{repo}/issues/{num}/comments?per_page={max_comments}", token)
        for c in cs[:max_comments]:
            parts.append(f"COMMENT({c.get('user', {}).get('login', '?')}):\n{(c.get('body') or '')[:1200]}")
    except Exception:
        pass
    return "\n\n".join(parts)


SYS = ("你是量化交易手冊的工程經驗蒸餾員。從 GitHub issue 原文中提取『真實工程教訓』：坑是什麼、"
       "怎麼解/結論是什麼。只依據給定文本，禁止编造；數字必須逐字出現在文本中才可引用。只輸出 JSON。")


def distill_repo(repo, new_items, token):
    blocks = []
    for it in new_items:
        bundle = issue_bundle(repo, it["number"], token, it.get("body", ""))
        blocks.append(f"=== ISSUE #{it['number']} [{it['state']}] {it['title']}\n{bundle[:3800]}")
    prompt = f"""下面是 {repo} 的 {len(blocks)} 個高互動 issue（含正文與部分評論）。逐個提取工程教訓。

{chr(10).join(blocks)}

輸出 JSON：{{"lessons": [{{"issue": 123, "category": "數據|回測失真|執行|性能|部署|API坑|其他",
  "lesson": "≤70字中文,格式『坑→解法/結論』,術語留英文,不知道解法就寫坑+現狀"}}]}}
規則：每個 issue 至多 1 條;內容太水/純提問無結論的 issue 可跳過;禁止编造數字。"""
    old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("alarm")))
    signal.alarm(170)
    try:
        raw = ask(prompt, system=SYS, thinking=True, temperature=0.2,
                  max_tokens=2500, json_mode=True, timeout=150, retries=2)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1]).get("lessons", [])


HEADER = """# GitHub 工程經驗筆記（quant 開源棧）

> 論文不會告訴你的事：**11 個核心 quant 開源庫的高互動 issues**，自動掃描、qwen 蒸餾成
> 「坑 → 解法」，逐條帶原 issue 連結可追溯。每週增量追加（append-only），發現日期見各條。
> 涵蓋：數據坑 / 回測失真 / 執行 / 性能 / 部署 / API 坑。

回 [手冊首頁](../README.md) · 部署總覽 [deployment](overview.md)

"""


def render_entry(repo, it, lesson, week):
    url = f"https://github.com/{repo}/issues/{it['number']}"
    state = "closed" if it["state"] == "closed" else "open"
    return (f"- **[#{it['number']}]({url})**（{state} · {it['comments']}💬 · {lesson.get('category','其他')} · {week}）"
            f" — {lesson.get('lesson','').strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="", help="only this repo (test)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    token = gh_token()
    seen = json.loads(SEEN_F.read_text()) if SEEN_F.exists() else {}
    y, w, _ = dt.date.today().isocalendar()
    week = f"{y}-W{w:02d}"

    page = PAGE.read_text() if PAGE.exists() else ""
    sections = {}          # repo -> list of new rendered lines
    total_new = 0
    for cfg in REPOS:
        repo = cfg["repo"]
        if a.repo and repo != a.repo:
            continue
        try:
            items = top_issues(repo, cfg["min"])
        except Exception as e:
            print(f"  REPO-FAIL {repo}: {repr(e)[:60]}", file=sys.stderr)
            continue
        old = set(seen.get(repo, []))
        new = [it for it in items if it["number"] not in old][:8]
        print(f"  {repo:<46} top={len(items)} new={len(new)}", flush=True)
        if not new:
            continue
        try:
            lessons = distill_repo(repo, new, token)
        except Exception as e:
            print(f"  DISTILL-FAIL {repo}: {repr(e)[:70]}", file=sys.stderr)
            continue   # not marked seen -> retried next week
        by_num = {l.get("issue"): l for l in lessons if l.get("lesson")}
        lines = []
        for it in new:
            l = by_num.get(it["number"])
            if l:
                lines.append(render_entry(repo, it, l, week))
        if lines:
            sections[repo] = (cfg["desc"], lines)
            total_new += len(lines)
        seen.setdefault(repo, [])
        seen[repo] = sorted(set(seen[repo]) | {it["number"] for it in new})

    if not sections:
        print("gh_quant_issues: nothing new")
        return
    if a.dry:
        for r, (d, ls) in sections.items():
            print(f"\n## {r} — {d}")
            print("\n".join(ls))
        return

    if not page:
        page = HEADER
    for repo, (desc, lines) in sections.items():
        h = f"## {repo}"
        if h in page:
            # append under the existing section header (after its desc line)
            idx = page.find(h)
            nl = page.find("\n", idx)
            # insert after the header + desc line block: find end of that line's paragraph
            insert_at = page.find("\n", nl + 1) + 1
            page = page[:insert_at] + "\n".join(lines) + "\n" + page[insert_at:]
        else:
            page += f"\n## {repo}\n_{desc}_\n\n" + "\n".join(lines) + "\n"
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page)
    SEEN_F.write_text(json.dumps(seen, indent=1))
    print(f"gh_quant_issues: +{total_new} lessons across {len(sections)} repos -> {PAGE}", flush=True)


if __name__ == "__main__":
    main()
