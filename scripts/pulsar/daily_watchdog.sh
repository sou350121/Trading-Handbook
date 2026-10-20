#!/bin/bash
set -euo pipefail
cd /home/claudeuser/trading-handbook-work
exec /usr/bin/python3.11 scripts/pulsar/watch_pipeline.py --apply --notify
