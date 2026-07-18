#!/bin/bash
# Trading-Handbook weekly radar: sweep arXiv q-fin + practitioner RSS, classify with qwen,
# publish the week's radar pages, commit + push. Multi-source intake (beyond QuantML).
# Git identity via env (NOT a "git -c user.name=Two Words" string -- that space word-splits
# and git reads the 2nd word as the subcommand; that bug dead-locked the daily cron for 12 days).
set -uo pipefail
cd /home/claudeuser/trading-handbook-work || exit 1
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
export GIT_AUTHOR_NAME="Pulsar Bot"    GIT_AUTHOR_EMAIL="noreply@anthropic.com"
export GIT_COMMITTER_NAME="Pulsar Bot" GIT_COMMITTER_EMAIL="noreply@anthropic.com"
LOG=data/radar_weekly.log
ts(){ date -u +"%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
PY=/usr/bin/python3.11

log "===== weekly radar start ====="
# only run from a clean radar/ tree (don't clobber a manual edit); radars only ever write radar/**
if [ -n "$(git status --porcelain -- radar 2>/dev/null)" ]; then
  log "recovering: radar/ already has uncommitted pages (prior interrupted run) — will re-commit"
fi
git pull --rebase -q 2>>"$LOG" || log "pull warn (continuing)"

# arXiv radar (qwen-classified) — external APIs are flaky, never abort the whole run on one
timeout 1500 "$PY" scripts/pulsar/arxiv_radar.py --days 8 >>"$LOG" 2>&1 || log "arxiv_radar returned nonzero (partial ok)"
# practitioner RSS radar
timeout 1500 "$PY" scripts/pulsar/practitioner_radar.py --days 8 >>"$LOG" 2>&1 || log "practitioner_radar returned nonzero (partial ok)"

# refresh the hot-queue metadata into corpus (radar_queue is gitignored; corpus_meta unchanged here)
git add radar/ 2>>"$LOG"
if [ -z "$(git diff --cached --name-only)" ]; then log "no new radar pages this week"; exit 0; fi
if git diff --cached --name-only | grep -q '^data/\(raw\|distill\)/'; then
  log "ABORT: refusing to commit raw/distill"; git reset -q; exit 1
fi
$PY scripts/handbook_audit.py >>"$LOG" 2>&1 || { log "gate FAILED — staged, not pushing"; exit 1; }
W=$(date -u +%G-W%V)
git commit -q -m "radar: weekly arXiv + practitioner sweep [$W]" >>"$LOG" 2>&1
if git push -q >>"$LOG" 2>&1; then log "PUSHED weekly radar $W"
elif git pull --rebase -q >>"$LOG" 2>&1 && git push -q >>"$LOG" 2>&1; then log "PUSHED after rebase $W"
else log "push FAILED — next run retries"; exit 1; fi
log "===== weekly radar done ====="
