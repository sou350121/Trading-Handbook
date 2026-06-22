#!/usr/bin/env python3
"""Reading-experience pass over dissection pages (mechanical, idempotent, py3.6-safe).

For each foundations/<zone>/<page>.md (not overview):
  - add Mintlify YAML frontmatter (title + description) at the very top
  - drop the now-redundant `# H1` (Mintlify renders the frontmatter title)
  - keep the ontology-5axis comment (moved just under frontmatter)
  - insert a 5-axis "fingerprint" table right under the header blockquote

Run: python3 scripts/enhance_pages.py [--dry-run] [--only foundations/zone/x.md]
Safe to re-run: skips pages that already have frontmatter.
"""
import glob, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
AXES = [("data", "數據模態"), ("horizon", "時間尺度"), ("paradigm", "學習範式"),
        ("alpha", "Alpha機制"), ("autonomy", "人機協作")]


def parse_ontology(text):
    m = re.search(r'<!--\s*ontology-5axis\s+(.*?)-->', text)
    if not m:
        return None
    kv = dict(re.findall(r'(\w+)=([^\s]+)', m.group(1)))
    return kv


def yaml_escape(s):
    s = re.sub(r'\s+', ' ', s).strip().replace('"', "'")
    return s


def fingerprint_table(kv):
    head = "| " + " | ".join(label for _, label in AXES) + " |"
    sep = "|" + "|".join([":-:"] * len(AXES)) + "|"
    row = "| " + " | ".join("`%s`" % kv.get(key, "—") for key, _ in AXES) + " |"
    return "\n".join([head, sep, row])


def enhance(text):
    if text.lstrip().startswith("---"):
        return None  # already has frontmatter
    kv = parse_ontology(text)
    onto_m = re.search(r'(<!--\s*ontology-5axis.*?-->)', text)
    onto = onto_m.group(1) if onto_m else ""
    # H1: "# <name> 解構（<fw>）"
    h1_m = re.search(r'^#\s+(.+?)\s*$', text, re.M)
    if not h1_m:
        return None
    h1 = h1_m.group(1).strip()
    title = re.sub(r'\s*解[構构].*$', '', h1).strip() or h1
    # description: prefer 核心定位, else first TL;DR clause
    desc = ""
    d_m = re.search(r'\*\*核心定位\*\*[：:](.+)', text)
    if d_m:
        desc = d_m.group(1)
    else:
        t_m = re.search(r'\*\*TL;DR:\*\*(.+)', text)
        desc = t_m.group(1) if t_m else h1
    desc = yaml_escape(re.sub(r'[（(].*?[）)]', '', desc))[:140]
    # body = original minus the ontology comment line and the H1 line
    body = text
    if onto:
        body = body.replace(onto, "", 1)
    body = re.sub(r'^#\s+.+?\s*$\n?', '', body, count=1, flags=re.M)
    body = body.lstrip("\n")
    # insert fingerprint table after the header blockquote block (the `> ...` lines)
    fp = ("\n**五軸座標**\n\n" + fingerprint_table(kv) + "\n") if kv else ""
    if fp:
        # place after the first blockquote group (發布/導讀/核心定位)
        bq = re.search(r'((?:^>.*\n)+)', body, re.M)
        if bq:
            idx = bq.end()
            body = body[:idx] + fp + body[idx:]
        else:
            body = fp + "\n" + body
    fm = '---\ntitle: "%s"\ndescription: "%s"\n---\n' % (yaml_escape(title), desc)
    out = fm + (onto + "\n\n" if onto else "") + body
    return re.sub(r'\n{3,}', '\n\n', out)


def main():
    dry = "--dry-run" in sys.argv
    only = None
    if "--only" in sys.argv:
        only = sys.argv[sys.argv.index("--only") + 1]
    files = [only] if only else [f for f in glob.glob(str(ROOT / "foundations/*/*.md"))
                                 if not f.endswith("overview.md")]
    done = skip = 0
    for f in sorted(files):
        text = pathlib.Path(f).read_text(encoding="utf-8")
        new = enhance(text)
        if new is None:
            skip += 1
            continue
        if dry:
            print("=" * 70)
            print("FILE:", f)
            print(new[:900])
            done += 1
            if done >= 2:
                break
        else:
            pathlib.Path(f).write_text(new, encoding="utf-8")
            done += 1
    print("\nenhanced=%d skipped(already/none)=%d" % (done, skip))


if __name__ == "__main__":
    main()
