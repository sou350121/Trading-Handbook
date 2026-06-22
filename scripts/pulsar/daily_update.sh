#!/bin/bash
# Trading-Handbook daily incremental update (GitHub-primary, runs on this host's crontab).
# Detect new QuantML album articles -> fetch+classify+generate pages -> fingerprint + mermaid
# -> overviews/nav -> mechanical gate -> commit + push. No-op cleanly when nothing is new.
# Bodies (data/raw, data/distill) stay LOCAL (gitignored); only pages + index.json are pushed.
set -uo pipefail
cd /home/claudeuser/trading-handbook-work || exit 1
export PATH="/usr/local/bin:/usr/bin:/bin:$HOME/.local/bin:$PATH"
LOG=data/daily.log
ts(){ date -u +"%Y-%m-%d %H:%M:%S"; }
log(){ echo "[$(ts)] $*" | tee -a "$LOG"; }
GIT="git -c user.name=Pulsar Bot -c user.email=noreply@anthropic.com"

log "===== daily start ====="

# only run from a clean tree (avoid clobbering an in-progress manual edit)
if [ -n "$(git status --porcelain -- foundations crossing cheat-sheet docs.json 2>/dev/null)" ]; then
  log "ABORT: tracked content has uncommitted changes; skipping run"; exit 0
fi
git pull --rebase -q 2>>"$LOG" || { log "ABORT: git pull failed"; exit 1; }

# prev = last COMMITTED index (not working tree) so a crashed mid-run can't orphan new articles
git show HEAD:data/index.json > data/index.prev.json 2>/dev/null || echo "[]" > data/index.prev.json
python3 scripts/pulsar/enumerate_album.py >>"$LOG" 2>&1 || { log "ABORT: enumerate failed"; exit 1; }

NEW=$(python3 - <<'PY'
import json
old={(a["msgid"],a["itemidx"]) for a in json.load(open("data/index.prev.json"))}
new=[a for a in json.load(open("data/index.json")) if (a["msgid"],a["itemidx"]) not in old]
print(",".join(str(a["pos"]) for a in new))
PY
)
if [ -z "$NEW" ]; then log "no new articles; done"; exit 0; fi
CNT=$(echo "$NEW" | tr ',' '\n' | grep -c .)
log "NEW: $CNT article(s) at pos $NEW"

python3 scripts/pulsar/fetch_article.py --poslist "$NEW" --workers 3 >>"$LOG" 2>&1
python3 scripts/pulsar/distill_pass_a.py --workers 3 >>"$LOG" 2>&1

# conservative dedup: new record whose framework OR normalized title exactly matches an
# existing one -> mark dup_of (suppresses a redundant page). Collision scan catches same-slug.
python3 - "$NEW" >>"$LOG" 2>&1 <<'PY'
import json, glob, re, sys
newpos={int(x) for x in sys.argv[1].split(",") if x.strip()}
def norm(t): return re.sub(r'[^一-鿿a-zA-Z0-9]','', re.sub(r'^[A-Z]+\s*\d*\s*\|','',t or ''))
recs=[json.load(open(f)) for f in glob.glob("data/distill/*.json")]
old=[d for d in recs if d.get("pos") not in newpos]
for d in recs:
    if d.get("pos") not in newpos or d.get("dup_of"): continue
    fw=(d.get("source",{}) or {}).get("framework")
    for o in old:
        ofw=(o.get("source",{}) or {}).get("framework")
        if (fw and ofw and fw.strip().lower()==ofw.strip().lower()) or norm(d["title"])==norm(o["title"]):
            d["dup_of"]=o["msgid"]
            json.dump(d, open("data/distill/%s.json"%d["msgid"],"w"), ensure_ascii=False, indent=2)
            print("daily-dedup: pos%s dup_of pos%s"%(d["pos"],o["pos"])); break
PY

python3 scripts/pulsar/fix_collisions.py >>"$LOG" 2>&1 || true
python3 scripts/pulsar/distill_pass_b.py --workers 3 >>"$LOG" 2>&1
python3 scripts/add_fingerprint.py >>"$LOG" 2>&1
python3 scripts/pulsar/asciiflow_to_mermaid.py --workers 2 >>"$LOG" 2>&1 || true
python3 scripts/gen_overviews.py >>"$LOG" 2>&1
python3 scripts/gen_nav.py >>"$LOG" 2>&1
python3 scripts/pulsar/number_audit.py >>"$LOG" 2>&1 || true   # advisory

if ! python3 scripts/handbook_audit.py >>"$LOG" 2>&1; then
  log "GATE FAILED — not pushing (inspect $LOG)"; exit 1
fi

$GIT add foundations/ crossing/ cheat-sheet/ bridge-to-vla/ docs.json data/index.json 2>>"$LOG"
if git diff --cached --name-only | grep -q '^data/\(raw\|distill\)/'; then
  log "ABORT: refusing to commit raw/distill bodies"; git reset -q; exit 1
fi
if [ -z "$(git diff --cached --name-only)" ]; then log "nothing to commit"; exit 0; fi
$GIT commit -q -m "daily: +$CNT new QuantML article(s) [$(date -u +%Y-%m-%d)]" >>"$LOG" 2>&1
$GIT push -q >>"$LOG" 2>&1 && log "PUSHED +$CNT" || { log "push FAILED"; exit 1; }
log "===== daily done ====="
