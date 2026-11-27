#!/bin/bash
set -euo pipefail
cd /home/claudeuser/trading-handbook-work
export TRADING_DEEPSEEK_REPAIR_MODEL=deepseek-v4-pro
exec /usr/bin/python3.11 scripts/pulsar/run_pipeline.py --job daily --publish --allow-verified-partial
