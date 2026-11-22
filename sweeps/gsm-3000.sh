#!/usr/bin/env bash
# gsm-3000.sh -- GSM8K confirmation at 3,000 MiB for the amended pick (delta 0.001; results/AMENDMENT-1.md): delta 0, then 0.001.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== $(date -u +%T) gsm-3000-d0"
scripts/run_task.sh --name gsm-3000-d0 --cache-mib 3000 < /dev/null
sleep 15
echo "=== $(date -u +%T) gsm-3000-add0.001"
scripts/run_task.sh --name gsm-3000-add0.001 --cache-mib 3000 --bias 0.001 --bias-mode add < /dev/null
echo "=== $(date -u +%T) chain done"
