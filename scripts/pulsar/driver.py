#!/usr/bin/env python3
"""Unattended driver: wait for fetch to finish -> gap-fill fetch -> full Pass A.
Pass B is left for Opus-supervised step. Logs to data/driver.log."""
import subprocess, time, pathlib, datetime
ROOT = pathlib.Path(__file__).resolve().parents[2]
def log(m): 
    line=f"[{datetime.datetime.utcnow().strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
_DEVNULL = open("/dev/null","w")
def running():
    return subprocess.call(["pgrep","-f","fetch_article.py --workers"],
                           stdout=_DEVNULL, stderr=_DEVNULL)==0
def run(cmd):
    log("RUN "+" ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT))
# 1) wait for the already-running fetch
waited=0
while running():
    time.sleep(30); waited+=30
    if waited % 300 == 0:
        n=len(list((ROOT/"data"/"raw").glob("*.json")))
        log(f"waiting on fetch... raw={n}/399")
log("initial fetch finished")
# 2) gap-fill pass (picks up FAILs / missed), then a second time for stubborn ones
run(["python3","scripts/pulsar/fetch_article.py","--workers","4"])
run(["python3","scripts/pulsar/fetch_article.py","--workers","3"])
n=len(list((ROOT/"data"/"raw").glob("*.json")))
log(f"fetch complete: raw={n}/399")
# 3) full Pass A over everything fetched
run(["python3","scripts/pulsar/distill_pass_a.py","--workers","3"])
d=len(list((ROOT/"data"/"distill").glob("*.json")))
log(f"Pass A complete: distilled={d}")
log("DRIVER DONE — ready for Opus-supervised Pass B + Crossing synthesis")
