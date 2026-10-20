#!/bin/bash
set -euo pipefail
cd /home/claudeuser/trading-handbook-work
exec /usr/bin/python3.11 scripts/pulsar/run_pipeline.py --job weekly --publish --allow-verified-partial
