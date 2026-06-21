#!/bin/bash
cd /home/claudeuser/trading-handbook-work
for round in $(seq 1 40); do
  python3 scripts/pulsar/distill_pass_a.py --workers 3 >> data/passa_drain.log 2>&1
  # stop when no fetch shards remain AND distilled has caught up to fetched
  if ! pgrep -f 'fetch_article.py --poslist' >/dev/null; then
    python3 scripts/pulsar/distill_pass_a.py --workers 3 >> data/passa_drain.log 2>&1
    echo "[drain] fetch done, final Pass A complete (round $round)" >> data/passa_drain.log
    break
  fi
  sleep 60
done
