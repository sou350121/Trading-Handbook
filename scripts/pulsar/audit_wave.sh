#!/bin/bash
# Block until >=BATCH unaudited stable pages exist OR Pass B is complete (or cap).
# Emits the unaudited batch to data/audit_batch.txt.
cd /home/claudeuser/trading-handbook-work
BATCH=${1:-45}
touch data/audited.txt
unaudited() {
  python3 - "$@" <<'PY'
import os, time, glob
audited=set(l.strip() for l in open("data/audited.txt") if l.strip())
now=time.time(); out=[]
for f in glob.glob("foundations/*/*.md"):
    if f.endswith("overview.md"): continue
    if f in audited: continue
    try:
        if now-os.path.getmtime(f) < 10: continue  # still settling / mid-write
    except OSError: continue
    out.append(f)
print("\n".join(sorted(out)))
PY
}
for _ in $(seq 1 120); do      # cap ~60 min
  mapfile -t U < <(unaudited)
  n=${#U[@]}
  if [ -f data/PASSB_COMPLETE ]; then
    printf "%s\n" "${U[@]}" > data/audit_batch.txt
    echo "WAVE final passb_done unaudited=$n"; exit 0
  fi
  if [ "$n" -ge "$BATCH" ]; then
    printf "%s\n" "${U[@]}" > data/audit_batch.txt
    echo "WAVE batch unaudited=$n"; exit 0
  fi
  sleep 30
done
mapfile -t U < <(unaudited); printf "%s\n" "${U[@]}" > data/audit_batch.txt
echo "WAVE cap_timeout unaudited=${#U[@]}"
