#!/usr/bin/env python3
"""Trading-Handbook quality gate / audit.

Pre-push (and future-cron) gate for the Mintlify repo. Runs 8 checks and exits
nonzero if any HARD failure is found. HARD = broken links, docs.json nav
integrity, data-leak guard. Everything else is a warning.

Usage:
    python3 scripts/handbook_audit.py            # verbose (default)
    python3 scripts/handbook_audit.py --quiet     # only failures + summary

Exit codes: 0 = pass, 1 = at least one hard failure.
"""
import json
import os
import re
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Content dirs that hold doc pages. Anything else (data/, scripts/, .git/) is ignored.
CONTENT_DIRS = [
    "foundations", "crossing", "cheat-sheet", "benchmarks", "bridge-to-vla",
    "companies", "deployment", "frontier", "reports", "use-cases",
]
# Top-level standalone docs referenced by nav but living at repo root.
ROOT_DOCS = ["ONBOARDING.md", "AGENTS.md", "CONTRIBUTING.md", "MAINTAINER.md", "README.md"]
# Files never treated as orphans even when absent from nav.
ORPHAN_EXEMPT = {"README.md", "AGENTS.md"}
DOC_EXTS = (".md", ".mdx")

# Markdown link: ](target) — captures the target up to ) or whitespace.
LINK_RE = re.compile(r"\]\(\s*([^)\s]+)\s*\)")
ONTOLOGY_RE = re.compile(r"<!--\s*ontology-5axis")
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$")

QUIET = "--quiet" in sys.argv[1:]


def out(msg=""):
    if not QUIET:
        print(msg)


def fail_line(msg):
    print(msg)


def rel(p):
    """Repo-relative POSIX path for display."""
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def all_content_files():
    """Every .md/.mdx under the content dirs (sorted, repo-relative paths)."""
    files = []
    for d in CONTENT_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for dirpath, _dirs, names in os.walk(base):
            for n in names:
                if n.lower().endswith(DOC_EXTS):
                    files.append(pathlib.Path(dirpath) / n)
    return sorted(files, key=lambda p: rel(p))


def resolve_target(src_file, target):
    """Resolve a markdown link target to an on-disk file path, or None.

    Handles: root-relative (/foo/bar), relative (../x, ./x, x), the Mintlify
    no-extension convention (try .md / .mdx and index files), and trailing
    #anchor / ?query fragments which are stripped before resolution.
    """
    # Strip anchor and query.
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target:
        return "anchor-only"  # sentinel: pure #anchor, nothing to resolve
    if target.startswith("/"):
        base = ROOT / target.lstrip("/")
    else:
        base = (src_file.parent / target)
    base = pathlib.Path(os.path.normpath(str(base)))
    # Direct hit (link already carries an extension or is a real file).
    if base.is_file():
        return base
    # No-extension convention: foo -> foo.md / foo.mdx.
    for ext in DOC_EXTS:
        cand = pathlib.Path(str(base) + ext)
        if cand.is_file():
            return cand
    # Directory link -> overview/index page inside it.
    if base.is_dir():
        for stem in ("overview", "index"):
            for ext in DOC_EXTS:
                cand = base / (stem + ext)
                if cand.is_file():
                    return cand
    return None


def check_broken_links(files):
    """CHECK 1 (hard): markdown relative links that resolve to no file."""
    broken = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for m in LINK_RE.finditer(text):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:", "tel:")):
                continue
            if target.startswith("#"):
                continue
            resolved = resolve_target(f, target)
            if resolved is None:
                broken.append((rel(f), target))
    return broken


def load_nav_pages(docs):
    """Extract every page path string from docs.json navigation (recursive)."""
    pages = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "pages" and isinstance(v, list):
                    for item in v:
                        if isinstance(item, str):
                            pages.append(item)
                        else:
                            walk(item)
                else:
                    walk(v)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(docs.get("navigation", {}))
    return pages


def nav_page_to_file(page):
    """Map a docs.json nav page entry to an on-disk file path, or None."""
    base = ROOT / page
    if base.is_file():
        return base
    for ext in DOC_EXTS:
        cand = pathlib.Path(str(base) + ext)
        if cand.is_file():
            return cand
    return None


def check_nav_integrity(nav_pages):
    """CHECK 2 (hard): nav entries pointing to missing files."""
    missing = []
    for page in nav_pages:
        if nav_page_to_file(page) is None:
            missing.append(page)
    return missing


def check_orphans(files, nav_pages):
    """CHECK 3 (warning): content pages not referenced anywhere in nav."""
    # Normalize nav entries to a set of repo-relative stems (no extension).
    nav_norm = set()
    for page in nav_pages:
        stem = re.sub(r"\.(md|mdx)$", "", page)
        nav_norm.add(stem)
    orphans = []
    for f in files:
        if f.name in ORPHAN_EXEMPT:
            continue
        relp = rel(f)
        stem = re.sub(r"\.(md|mdx)$", "", relp)
        if stem not in nav_norm:
            orphans.append(relp)
    return orphans


def check_zone_overviews():
    """CHECK 4 (warning): foundations/<zone>/ dirs missing overview.md."""
    missing = []
    fbase = ROOT / "foundations"
    if not fbase.is_dir():
        return missing
    for child in sorted(fbase.iterdir()):
        if not child.is_dir():
            continue
        has = any((child / ("overview" + ext)).is_file() for ext in DOC_EXTS)
        if not has:
            missing.append(rel(child))
    return missing


def check_ontology_headers():
    """CHECK 5 (warning): non-overview foundations pages missing ontology header."""
    missing = []
    fbase = ROOT / "foundations"
    if not fbase.is_dir():
        return missing
    for dirpath, _dirs, names in os.walk(fbase):
        for n in sorted(names):
            if not n.lower().endswith(DOC_EXTS):
                continue
            if n.lower().startswith("overview."):
                continue
            f = pathlib.Path(dirpath) / n
            try:
                # ontology comment sits in the head (after optional YAML frontmatter)
                head = "\n".join(f.read_text(encoding="utf-8").splitlines()[:8])
            except (OSError, UnicodeDecodeError):
                head = ""
            if not ONTOLOGY_RE.search(head):
                missing.append(rel(f))
    return missing


def check_slug_collisions(files):
    """CHECK 6 (warning): same slug twice in a zone, or cross-zone title dup."""
    by_zone = {}            # (zone) -> {slug: [paths]}
    titles = {}             # title -> [paths]
    collisions = []
    for f in files:
        relp = rel(f)
        parts = relp.split("/")
        zone = "/".join(parts[:-1]) if len(parts) > 1 else "."
        slug = f.stem
        by_zone.setdefault(zone, {}).setdefault(slug, []).append(relp)
        # First-line markdown title for cross-zone exact dup detection.
        try:
            for raw in f.read_text(encoding="utf-8").splitlines():
                m = TITLE_RE.match(raw)
                if m:
                    titles.setdefault(m.group(1).strip(), []).append(relp)
                    break
        except (OSError, UnicodeDecodeError):
            pass
    for zone, slugs in by_zone.items():
        for slug, paths in slugs.items():
            if len(paths) > 1:
                collisions.append("slug '{0}' x{1} in {2}: {3}".format(
                    slug, len(paths), zone, ", ".join(paths)))
    for title, paths in titles.items():
        zones = set("/".join(p.split("/")[:-1]) for p in paths)
        if len(paths) > 1 and len(zones) > 1:
            collisions.append("title '{0}' across zones: {1}".format(
                title, ", ".join(paths)))
    return collisions


def check_data_leak():
    """CHECK 7 (hard): no data/raw or data/distill file tracked by git."""
    leaked = []
    try:
        p = subprocess.Popen(
            ["git", "ls-files", "data/"],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, _stderr = p.communicate()
    except OSError as e:
        return ["git ls-files failed: {0}".format(e)]
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", "replace")
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("data/raw/") or line.startswith("data/distill/"):
            leaked.append(line)
    return leaked


def main():
    docs_path = ROOT / "docs.json"
    try:
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        fail_line("FATAL: cannot read docs.json: {0}".format(e))
        return 1

    files = all_content_files()
    nav_pages = load_nav_pages(docs)
    zones = [c for c in sorted((ROOT / "foundations").iterdir())
             if c.is_dir()] if (ROOT / "foundations").is_dir() else []
    atlases = [f for f in files if rel(f).startswith("crossing/")
               and f.name.lower().startswith("overview.")
               and rel(f) != "crossing/overview.md"]

    broken = check_broken_links(files)
    nav_missing = check_nav_integrity(nav_pages)
    orphans = check_orphans(files, nav_pages)
    zone_no_overview = check_zone_overviews()
    no_ontology = check_ontology_headers()
    collisions = check_slug_collisions(files)
    leaked = check_data_leak()

    # ---- per-check report ----
    out("=" * 60)
    out("Trading-Handbook Audit  (root: {0})".format(ROOT))
    out("=" * 60)

    def report(label, items, hard, fmt=lambda x: x):
        tag = "HARD" if hard else "warn"
        if items:
            print("[{0}] {1}: {2}".format(
                "FAIL" if hard else "WARN", label, len(items)))
            for it in items:
                print("    - {0}".format(fmt(it)))
        else:
            out("[ ok ] {0} ({1})".format(label, tag))

    report("1. broken internal links", broken, True,
           fmt=lambda t: "{0}  ->  {1}".format(t[0], t[1]))
    report("2. docs.json nav integrity", nav_missing, True)
    report("3. orphan pages", orphans, False)
    report("4. zones missing overview.md", zone_no_overview, False)
    report("5. pages missing ontology header", no_ontology, False)
    report("6. slug / title collisions", collisions, False)
    report("7. data-leak (data/raw|distill tracked by git)", leaked, True)

    # ---- summary ----
    hard_fail = bool(broken) or bool(nav_missing) or bool(leaked)
    warn_count = len(orphans) + len(zone_no_overview) + len(no_ontology) + len(collisions)

    print("-" * 60)
    print("SUMMARY: pages={0} zones={1} atlases={2} nav_entries={3}".format(
        len(files), len(zones), len(atlases), len(nav_pages)))
    print("         broken_links={0} nav_missing={1} orphans={2} "
          "data_leaks={3} warnings={4}".format(
              len(broken), len(nav_missing), len(orphans), len(leaked), warn_count))
    print("RESULT: {0}".format("FAIL" if hard_fail else "PASS"))
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
