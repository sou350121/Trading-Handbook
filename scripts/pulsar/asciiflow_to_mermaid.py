#!/usr/bin/env python3
"""Convert each page's `**信息流 ASCII:**` code fence into a rendered Mermaid flowchart.

qwen reformats the EXISTING ascii diagram into mermaid (no new info). Safe degradation:
the page is only modified if the produced mermaid passes a structural check; otherwise
the original ascii is left untouched. Resumable (skips pages already converted / without
an ascii flow block). Sharded for parallel runs.

Run: python3 scripts/pulsar/asciiflow_to_mermaid.py [--shard i/m] [--workers W] [--limit N] [--dry-run]
"""
import glob, re, sys, pathlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _qwen
import fix_github_render as _github_render

ROOT = pathlib.Path(__file__).resolve().parents[2]
# the ascii flow block: a "信息流 ..." label line (bold may open before a §-number),
# then a fenced code block. We match from 信息流 onward; the leading "**1.3 " stays in
# the pre-match text so the bold label is reconstructed intact on replace.
FLOW_RE = re.compile(r'(\*{0,2}信息流[^\n]*?\n)```[a-zA-Z]*\n(.*?)\n```', re.S)

SYS = ("你是技術文檔圖表轉換器。把給定的 ASCII 流程圖**忠實**轉成 Mintlify 可渲染的 Mermaid flowchart。"
       "只重排為 Mermaid，不增刪節點、不添加任何新信息。只輸出一個 ```mermaid 代碼塊，不要任何解釋。")

PROMPT = """把下面這個 ASCII 流程圖轉成 Mermaid `flowchart TD`。

嚴格規則：
- 節點與箭頭與原圖**一一對應**，不要新增或刪除節點，不要編造內容。
- 節點 ID 用純 ascii（A,B,C,N1...）；節點**標籤**照抄原文（可含中英文），凡標籤含中文/空格/括號/斜線/標點，一律用雙引號包起來：`A["原標籤"]`。
- 分支（├─ └─）轉成多條邊。順序流（→ 或 ↓）轉成 `A --> B`。
- 只輸出一個 ```mermaid 代碼塊，第一行是 `flowchart TD`，不要輸出別的任何文字。

ASCII：
{ascii}
"""

_lock = threading.Lock(); _n = {"ok": 0, "skip": 0, "fail": 0}


def extract_mermaid(resp):
    m = re.search(r'```mermaid\s*\n(.*?)```', resp, re.S)
    body = m.group(1).strip() if m else resp.strip()
    if body.startswith("```"):
        body = re.sub(r'^```[a-zA-Z]*\n', '', body).rstrip("`").strip()
    return body


def valid(mer):
    if not mer:
        return False
    first = mer.splitlines()[0].strip()
    if not re.match(r'(flowchart|graph)\s', first):
        return False
    if mer.count('"') % 2:           # unbalanced label quotes
        return False
    if mer.count('[') != mer.count(']'):
        return False
    if '-->' not in mer and '---' not in mer:
        return False
    return len([l for l in mer.splitlines() if l.strip()]) >= 3


def process(path, dry=False):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if '```mermaid' in text:                       # already converted
        with _lock: _n["skip"] += 1
        return
    m = FLOW_RE.search(text)
    if not m:
        with _lock: _n["skip"] += 1
        return
    ascii_art = m.group(2)
    try:
        resp = _qwen.ask(PROMPT.format(ascii=ascii_art), system=SYS,
                         thinking=False, temperature=0.1, max_tokens=1500, retries=4)
        mer = extract_mermaid(resp)
        if not valid(mer):
            with _lock: _n["fail"] += 1
            print("  KEEP-ASCII (invalid mermaid) %s" % path, flush=True)
            return
        block = m.group(1) + "```mermaid\n" + mer + "\n```"
        block = _github_render.fix_mermaid(block)   # GitHub-safe labels (strip ["[]"], escape <>&)
        new = text[:m.start()] + block + text[m.end():]
        if dry:
            print("=" * 60, "\n", path, "\n", block[:500], flush=True)
        else:
            pathlib.Path(path).write_text(new, encoding="utf-8")
        with _lock:
            _n["ok"] += 1
            if not dry:
                print("  MERMAID %s" % path, flush=True)
    except Exception as e:
        with _lock: _n["fail"] += 1
        print("  FAIL %s :: %s" % (path, e), flush=True)


def main():
    dry = "--dry-run" in sys.argv
    workers = 2; limit = 0; shard = ""
    if "--workers" in sys.argv: workers = int(sys.argv[sys.argv.index("--workers") + 1])
    if "--limit" in sys.argv: limit = int(sys.argv[sys.argv.index("--limit") + 1])
    if "--shard" in sys.argv: shard = sys.argv[sys.argv.index("--shard") + 1]
    files = sorted(f for f in glob.glob(str(ROOT / "foundations/*/*.md"))
                   if not f.endswith("overview.md"))
    if shard:
        i, mm = (int(x) for x in shard.split("/")); files = files[i::mm]
    if limit: files = files[:limit]
    print("asciiflow->mermaid on %d pages, workers=%d%s" % (len(files), workers, " DRY" if dry else ""), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(process, f, dry) for f in files]):
            fut.result()
    print("\nDONE ok=%d skip=%d keep-ascii/fail=%d" % (_n["ok"], _n["skip"], _n["fail"]), flush=True)


if __name__ == "__main__":
    main()
