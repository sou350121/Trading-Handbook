#!/usr/bin/env python3
"""Wait for Pass B to finish, then regenerate overviews+nav, link-audit, commit & push.
3.6-safe. Logs to data/assemble.log."""
import subprocess, time, os, pathlib, re, datetime
ROOT = pathlib.Path(__file__).resolve().parents[1]
DEVNULL = open(os.devnull, "w")
def log(m): print(f"[{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {m}", flush=True)
def running(name): return subprocess.call(["pgrep","-f",name], stdout=DEVNULL, stderr=DEVNULL)==0
def sh(cmd): return subprocess.call(cmd, cwd=str(ROOT))

# 1) wait for Pass B
w=0
while running("distill_pass_b.py"):
    time.sleep(20); w+=20
    if w%180==0:
        n=len([p for p in (ROOT/"foundations").rglob("*.md") if p.name!="overview.md"])
        log(f"waiting on Pass B... pages={n}")
log("Pass B finished")

# 2) regenerate overviews + nav
sh(["python3","scripts/gen_overviews.py"]); sh(["python3","scripts/gen_nav.py"])

# 3) lightweight link audit: every ](slug) link in overviews must resolve
broken=[]
for ov in (ROOT/"foundations").glob("*/overview.md"):
    zone=ov.parent.name
    for m in re.finditer(r'\]\(([a-z0-9\-]+)\)', ov.read_text()):
        slug=m.group(1)
        if not (ov.parent/f"{slug}.md").exists():
            broken.append(f"{zone}/{slug}")
log(f"link audit: {len(broken)} broken" + (": "+", ".join(broken[:8]) if broken else " ✓"))

# 4) commit + push
npages=len([p for p in (ROOT/"foundations").rglob("*.md") if p.name!="overview.md"])
sh(["git","add","-A"])
msg=(f"content: {npages} dissection pages + linked overviews (current batch)\n\n"
     "Pass B (qwen) dissections; overviews now link page-worthy methods.\n\n"
     "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>")
rc=sh(["git","commit","-q","-m",msg])
if rc==0:
    sh(["git","push","-q","origin","main"])
    log(f"PUSHED {npages} pages")
else:
    log("nothing to commit")
log("ASSEMBLE DONE")
