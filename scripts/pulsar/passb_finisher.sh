#!/bin/bash
cd /home/claudeuser/trading-handbook-work
log(){ echo "[$(date -u +%H:%M:%S)] $*" >> data/passb_finisher.log; }
log "waiting for 3 Pass B shards"
while pgrep -f 'distill_pass_b.py --shard' >/dev/null; do sleep 30; done
log "shards exited; final single-process sweep (catches race-skips)"
python3 scripts/pulsar/distill_pass_b.py --workers 3 >> data/passb_sweep.log 2>&1
PG=$(find foundations -name '*.md' | grep -v overview | wc -l)
log "SENTINEL PASSB_COMPLETE pages=$PG"
echo "pages=$PG" > data/PASSB_COMPLETE
