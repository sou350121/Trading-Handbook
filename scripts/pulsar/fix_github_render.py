#!/usr/bin/env python3
"""Make math + mermaid render correctly on GitHub (GFM + MathJax + mermaid v10).

Fixes, all idempotent and scoped so content/numbers are never altered:

MATH (GitHub inline math $...$ is parsed AFTER table pipes, and unpaired/currency $ mis-pair):
  1. `|` inside a real inline `$...$` -> `\mid` (a raw pipe in a table cell splits the row).
  2. currency / literal `$` (e.g. `$181.68`) -> `\$` so it stops pairing into fake math spans.
  Real math spans and `$$...$$` blocks and code fences are protected first, so LaTeX is untouched.

MERMAID (node labels are rendered as HTML by GitHub; raw <>& and stray brackets break them):
  3. strip ascii-box leftover brackets:  ["[Granger Test]"] -> ["Granger Test"]
  4. inside quoted labels: & -> &amp;, < -> &lt;, > -> &gt;  (else `X_<T` is eaten as an HTML tag)

Shared with asciiflow_to_mermaid.py (source fix) so newly generated pages stay clean.

Usage: python3 fix_github_render.py [--check] [--dry-run N] [paths...]
  --check     report how many files/occurrences would change, write nothing
  --dry-run N  print unified-ish diffs for the first N changed files
  default      rewrite all pages under foundations/ guides/ crossing/ cheat-sheet/
"""
import glob, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

CJK = re.compile(r'[一-鿿　-〿＀-￯]')
MATHY = re.compile(r'[\\^_{}]')
SHORT_MATH = re.compile(r'^[A-Za-z0-9α-ωΑ-Ω=+\-*/().,;:~<>\s|]{1,20}$')


def _is_math(c):
    """Heuristic: is the text between two $ a real math span (vs a currency/prose false pair)?"""
    s = c.strip()
    if not s:
        return False
    if CJK.search(c):            # CJK inside -> prose/currency false pair, never math
        return False
    if MATHY.search(c):          # \ ^ _ { } -> definitely LaTeX
        return True
    # currency/prose false-pair guards (e.g. "112.79 | CSM: " between two currency $)
    if re.match(r'\d', s) and '|' in s:
        return False
    if re.search(r'[A-Za-z]{3,}:', s):        # "Baseline:", "CSM: " -> prose, not math
        return False
    if len(s) <= 20 and SHORT_MATH.match(s):   # short symbolic token e.g. R^2, k=3, \theta
        return True
    return False


def _fix_inline_math_line(line):
    if '$' not in line:
        return line
    keep = []

    def repl(m):
        c = m.group(1)
        if _is_math(c):
            keep.append('$' + re.sub(r'\s*(?<!\\)\|\s*', r' \\mid ', c).strip() + '$')
            return '\x01%d\x01' % (len(keep) - 1)   # protect real math span
        return '\\$' + c + '\\$'                     # false pair -> literal $
    # shortest real pairs first; (?<!\\) so we never re-touch an already-escaped \$
    line = re.sub(r'(?<!\\)\$([^$\n]*?)(?<!\\)\$', repl, line)
    line = re.sub(r'(?<!\\)\$', r'\\$', line)        # any surviving lone $ is currency/literal
    for i, v in enumerate(keep):
        line = line.replace('\x01%d\x01' % i, v)
    return line


def fix_math(text):
    """Protect code fences + $$blocks$$ + `inline code`, fix inline $...$, restore."""
    store = {}

    def protect(pat, s, tag, flags=0):
        def r(m):
            k = "\x00%s%d\x00" % (tag, len(store))
            store[k] = m.group(0)
            return k
        return re.sub(pat, r, s, flags=flags)

    text = protect(r'```.*?```', text, 'F', re.S)      # any fenced block (incl mermaid)
    text = protect(r'\$\$.*?\$\$', text, 'B', re.S)     # block math
    text = protect(r'`[^`\n]*`', text, 'C')             # inline code
    text = "\n".join(_fix_inline_math_line(l) for l in text.split("\n"))
    for k, v in store.items():
        text = text.replace(k, v)
    return text


_HTMLTAG = re.compile(r'</?[a-zA-Z][^<>]*>')   # <br/>, <sub>, </b>, ... valid mermaid html labels


def sanitize_mermaid_label(s):
    """One quoted mermaid label's inner text -> GitHub-safe (called on capture inside "...").
    Keeps inner brackets like E[R] (real notation) and valid html tags like <br/>; only STRAY
    <>& (e.g. X_<T eaten as a tag, S&P) are entity-escaped. Idempotent."""
    tags = []

    def prot(m):
        tags.append(m.group(0))
        return '\x02%d\x02' % (len(tags) - 1)
    s = _HTMLTAG.sub(prot, s)                                     # protect <br/> etc.
    s = re.sub(r'&(?!(amp|lt|gt|quot|nbsp|#\d+);)', '&amp;', s)   # bare & -> entity (idempotent)
    s = s.replace('<', '&lt;').replace('>', '&gt;')              # stray < > -> entities
    for i, t in enumerate(tags):
        s = s.replace('\x02%d\x02' % i, t)
    return s


def fix_mermaid(text):
    def repl(m):
        block = m.group(0)
        block = re.sub(r'\["\[([^"\[\]]*)\]"\]', r'["\1"]', block)  # ["[label]"] -> ["label"]
        out = []
        for ln in block.split('\n'):
            # only sanitize labels on lines with balanced quotes; an odd count means a malformed
            # label (stray ") whose mis-pairing would swallow the --> arrow. Leave those for manual fix.
            if ln.count('"') % 2 == 0:
                ln = re.sub(r'"([^"\n]*)"', lambda mm: '"' + sanitize_mermaid_label(mm.group(1)) + '"', ln)
            out.append(ln)
        return '\n'.join(out)
    return re.sub(r'```mermaid\n.*?```', repl, text, flags=re.S)


def fix_text(text):
    # collapse double-escaped ampersands (a resolver-injected citation bug: "&amp;amp;" renders the
    # literal entity on GitHub instead of "&"). Idempotent: "&amp;amp;"->"&amp;", "&amp;"->unchanged.
    text = re.sub(r'&amp;(amp;)+', '&amp;', text)
    return fix_math(fix_mermaid(text))


def main():
    args = sys.argv[1:]
    check = "--check" in args
    dry_n = int(args[args.index("--dry-run") + 1]) if "--dry-run" in args else 0
    paths = [a for a in args if not a.startswith("--") and not a.isdigit()]
    if paths:
        files = paths
    else:
        files = [f for d in ("foundations", "guides", "crossing", "cheat-sheet")
                 for f in glob.glob(str(ROOT / d / "**" / "*.md"), recursive=True)]
    changed = 0
    shown = 0
    for f in sorted(files):
        old = pathlib.Path(f).read_text(encoding="utf-8")
        new = fix_text(old)
        if new == old:
            continue
        changed += 1
        if dry_n and shown < dry_n:
            shown += 1
            print("=" * 70, "\n", f)
            for lo, ln in zip(old.split("\n"), new.split("\n")):
                if lo != ln:
                    print("  - " + lo[:160])
                    print("  + " + ln[:160])
        if not check and not dry_n:
            pathlib.Path(f).write_text(new, encoding="utf-8")
    verb = "would change" if (check or dry_n) else "fixed"
    print("\nfix_github_render: %s %d / %d files" % (verb, changed, len(files)))
    return changed


if __name__ == "__main__":
    main()
