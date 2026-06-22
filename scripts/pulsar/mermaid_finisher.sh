#!/bin/bash
cd /home/claudeuser/trading-handbook-work
while pgrep -f 'asciiflow_to_mermaid.py --shard' >/dev/null; do sleep 20; done
OK=$(grep -h '  MERMAID ' data/mermaid_*.log 2>/dev/null | wc -l)
KEEP=$(grep -h 'KEEP-ASCII' data/mermaid_*.log 2>/dev/null | wc -l)
TOT=$(grep -rl '```mermaid' foundations/*/*.md 2>/dev/null | grep -v overview | wc -l)
echo "converted=$OK keep-ascii=$KEEP pages-with-mermaid=$TOT" > data/MERMAID_COMPLETE
