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
# top-journal radar (Crossref polling + Unpaywall OA queue)
timeout 1800 "$PY" scripts/pulsar/journal_radar.py --days 9 >>"$LOG" 2>&1 || log "journal_radar returned nonzero (partial ok)"
# practitioner + institutional (NBER/Fed/BIS) RSS radar
timeout 1500 "$PY" scripts/pulsar/practitioner_radar.py --days 8 >>"$LOG" 2>&1 || log "practitioner_radar returned nonzero (partial ok)"
# GitHub quant-stack engineering notes (append-only weekly)
timeout 1500 "$PY" scripts/pulsar/gh_quant_issues.py >>"$LOG" 2>&1 || log "gh_quant_issues returned nonzero (partial ok)"

# arXiv ⚡ -> first-hand full-text dissection. Ingests queued hot papers as data/raw (QuantML
# schema) and reuses the origin-aware Pass A/B to write foundations/ pages -- the leverage move:
# primary-source dissections, not second-hand digests. A qwen outage just leaves the raws cached;
# the daily cron's own Pass A/B (also origin-aware) self-heals them on its next run.
timeout 2400 "$PY" scripts/pulsar/arxiv_dissect.py --limit 8 --workers 2 >>"$LOG" 2>&1 || log "arxiv_dissect returned nonzero (partial ok)"
# journal ⚡ with an Unpaywall OA pdf -> same first-hand dissection (origin=oa)
timeout 2400 "$PY" scripts/pulsar/arxiv_dissect.py --oa --limit 6 --workers 2 >>"$LOG" 2>&1 || log "oa_dissect returned nonzero (partial ok)"

# Downstream page processing (mirrors daily_update.sh); each is a no-op when nothing new landed.
"$PY" scripts/add_fingerprint.py >>"$LOG" 2>&1 || true
"$PY" scripts/pulsar/asciiflow_to_mermaid.py --workers 2 >>"$LOG" 2>&1 || true
"$PY" scripts/gen_overviews.py >>"$LOG" 2>&1 || true
"$PY" scripts/gen_nav.py >>"$LOG" 2>&1 || true
"$PY" scripts/pulsar/resolve_sources_v2.py --resolve --inject >>"$LOG" 2>&1 || true
"$PY" scripts/pulsar/fix_github_render.py >>"$LOG" 2>&1 || true   # GitHub math+mermaid render fixes (idempotent)
"$PY" scripts/export_corpus_meta.py >>"$LOG" 2>&1 || true

git add radar/ foundations/ crossing/ cheat-sheet/ deployment/ docs.json data/corpus_meta.json 2>>"$LOG"
if [ -z "$(git diff --cached --name-only)" ]; then log "no new radar/dissection pages this week"; exit 0; fi
if git diff --cached --name-only | grep -q '^data/\(raw\|distill\)/'; then
  log "ABORT: refusing to commit raw/distill"; git reset -q; exit 1
fi
$PY scripts/handbook_audit.py >>"$LOG" 2>&1 || { log "gate FAILED — staged, not pushing"; exit 1; }
W=$(date -u +%G-W%V)
git commit -q -m "radar: weekly arXiv + practitioner sweep + first-hand dissections [$W]" >>"$LOG" 2>&1
if git push -q >>"$LOG" 2>&1; then log "PUSHED weekly radar $W"
elif git pull --rebase -q >>"$LOG" 2>&1 && git push -q >>"$LOG" 2>&1; then log "PUSHED after rebase $W"
else log "push FAILED — next run retries"; exit 1; fi
log "===== weekly radar done ====="
