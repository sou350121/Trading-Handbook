#!/bin/bash
# Trading-Handbook daily-cron watchdog. The 21:00 daily run writes a heartbeat (data/.daily_ok)
# on every HEALTHY exit (noop / nothing-staged / pushed); an aborting or crashing run does NOT.
# If the heartbeat is stale (no healthy run in STALE_HOURS), the daily cron is stuck -> this
# watchdog SELF-KICKS a recovery run (the cron is now self-recovering) and, if that still fails,
# escalates a loud, dated alert. Rationale: `mail` is dead on this host, so silent failure must
# self-heal, not merely notify. (Without this, the git-quoting bug went unnoticed for 12 days.)
set -uo pipefail
cd /home/claudeuser/trading-handbook-work || exit 1
STALE_HOURS=${1:-72}
HB=data/.daily_ok
ALERT=/tmp/trading-ALERT.log
ts(){ date -u +"%Y-%m-%d %H:%M:%S"; }

age_hours(){ [ -f "$HB" ] && echo $(( ( $(date +%s) - $(stat -c %Y "$HB") ) / 3600 )) || echo 99999; }

AGE=$(age_hours)
if [ "$AGE" -lt "$STALE_HOURS" ]; then
  exit 0   # healthy — heartbeat fresh
fi

echo "[$(ts)] WATCHDOG: daily heartbeat stale (${AGE}h >= ${STALE_HOURS}h) -> self-kicking recovery" | tee -a "$ALERT" >> data/daily.log
flock -n /tmp/trading-daily.lock timeout 2400 /bin/bash scripts/pulsar/daily_update.sh >> data/daily.log 2>&1

AGE2=$(age_hours)
if [ "$AGE2" -lt "$STALE_HOURS" ]; then
  echo "[$(ts)] WATCHDOG: recovery succeeded (heartbeat refreshed, age=${AGE2}h)" | tee -a "$ALERT" >> data/daily.log
else
  # still stuck after a kick -> a real fault the recovery cannot self-heal; make it loud + durable.
  BEHIND=$(python3 - 2>/dev/null <<'PY'
import json, subprocess
try: c=len(json.loads(subprocess.check_output(["git","show","HEAD:data/corpus_meta.json"]).decode()))
except Exception: c=-1
try: a=len(json.load(open("data/index.json")))
except Exception: a=-1
print(a-c if a>=0 and c>=0 else "?")
PY
)
  echo "[$(ts)] WATCHDOG ESCALATION: daily cron STILL stale after recovery kick (age=${AGE2}h, handbook behind album by ${BEHIND}). Needs a human — inspect /home/claudeuser/trading-handbook-work/data/daily.log" | tee -a "$ALERT" >> data/daily.log
fi
