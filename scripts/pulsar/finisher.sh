#!/bin/bash
# Wait for the 4 fetch shards to exit, then gap-fill residual missing positions
# (allorigins 520/522 are transient), run final Pass A, drop a sentinel.
cd /home/claudeuser/trading-handbook-work
log() { echo "[$(date -u +%H:%M:%S)] $*" >> data/finisher.log; }

log "finisher up; waiting for fetch shards to exit"
while pgrep -f 'fetch_article.py --poslist' >/dev/null; do sleep 30; done
log "all shards exited"

miss() {
python3 - <<'PY'
import json,glob
idx=json.load(open("data/index.json")); poss={a["pos"] for a in idx}
f=set()
for p in glob.glob("data/raw/*.json"):
    try:
        r=json.load(open(p))
        if len(r.get("body",""))>200: f.add(r["pos"])
    except: pass
m=sorted(poss-f)
print(",".join(map(str,m)))
PY
}

for round in 1 2 3 4; do
  M=$(miss)
  if [ -z "$M" ]; then log "no missing -- corpus complete"; break; fi
  cnt=$(echo "$M" | tr ',' '\n' | grep -c .)
  log "gap-fill round $round: $cnt missing"
  python3 scripts/pulsar/fetch_article.py --poslist "$M" --workers 3 >> data/gapfill.log 2>&1
  sleep 20
done

M=$(miss); cnt=$(echo "$M" | tr ',' '\n' | grep -c .); [ -z "$M" ] && cnt=0
log "final missing after gap-fill: $cnt  (positions: $M)"

log "final Pass A drain"
python3 scripts/pulsar/distill_pass_a.py --workers 3 >> data/passa_drain.log 2>&1
DIST=$(ls data/distill | wc -l); RAW=$(ls data/raw | wc -l)
log "SENTINEL FETCH_COMPLETE raw=$RAW distill=$DIST missing=$cnt"
echo "raw=$RAW distill=$DIST missing=$cnt positions=$M" > data/FETCH_COMPLETE
