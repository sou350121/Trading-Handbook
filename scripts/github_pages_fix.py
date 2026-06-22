#!/usr/bin/env python3
"""GitHub-primary view: strip Mintlify YAML frontmatter and restore an `# H1` heading.

(We added frontmatter for Mintlify, which renders as an ugly table on GitHub and left
pages without a title heading.) Idempotent: only touches pages that still have frontmatter.
Keeps the ontology comment, the 五軸座標 fingerprint, and all content.

Run: python3 scripts/github_pages_fix.py [--dry-run]
"""
import glob, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FM_RE = re.compile(r'^---\n(.*?)\n---\n', re.S)


def fix(text):
    m = FM_RE.match(text)
    if not m:
        return None  # no frontmatter -> already github-shaped
    fm = m.group(1)
    tm = re.search(r'title:\s*"?(.*?)"?\s*$', fm, re.M)
    title = (tm.group(1).strip() if tm else "解構")
    body = text[m.end():].lstrip("\n")
    # body starts with the ontology comment; insert H1 right after it
    onto = re.match(r'(<!--\s*ontology-5axis.*?-->\n)', body, re.S)
    h1 = "# %s 解構\n" % title
    if "\n# " in body[:300] or body.startswith("# "):
        new = body  # already has an H1
    elif onto:
        new = onto.group(1) + "\n" + h1 + body[onto.end():]
    else:
        new = h1 + "\n" + body
    return new


def main():
    dry = "--dry-run" in sys.argv
    files = sorted(f for f in glob.glob(str(ROOT / "foundations/*/*.md"))
                   if not f.endswith("overview.md"))
    done = skip = 0
    for f in files:
        text = pathlib.Path(f).read_text(encoding="utf-8")
        new = fix(text)
        if new is None:
            skip += 1
            continue
        if dry:
            print("=" * 60, "\n", f, "\n", new[:380])
            done += 1
            if done >= 3:
                break
        else:
            pathlib.Path(f).write_text(new, encoding="utf-8")
            done += 1
    print("\nfixed=%d skipped(no-frontmatter)=%d" % (done, skip))


if __name__ == "__main__":
    main()
