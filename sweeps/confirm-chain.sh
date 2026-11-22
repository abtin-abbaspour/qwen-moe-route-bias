#!/usr/bin/env bash
# confirm-chain.sh -- after the final sweeps: the amended speed runs (results/AMENDMENT-1.md), then a 5-question GSM8K smoke test,
# then (only if the smoke test answered 5 questions with at least 3 right) the GSM8K confirmation at 8,000 MiB (its pick, delta
# 0.0025, does not depend on the speed re-run): delta 0, then delta 0.0025.  On the host.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
scripts/sweep.sh sweeps/speed-confirm.txt
echo "=== $(date -u +%T) GSM8K smoke test"
scripts/run_task.sh --name gsm-smoke-8000-d0 --limit 5 --cache-mib 8000 < /dev/null
SMOKE=$(ls -d results/runs/*-gsm-smoke-8000-d0 | tail -1)
n=$(wc -l < "$SMOKE/gsm8k.jsonl" 2>/dev/null || echo 0)
ok=$(grep -c '"correct": true' "$SMOKE/gsm8k.jsonl" 2>/dev/null || echo 0)
echo "smoke: $n answered, $ok right"
if [ "$n" -ne 5 ] || [ "$ok" -lt 3 ]; then echo "=== smoke test failed; stopping"; exit 1; fi
sleep 15
echo "=== $(date -u +%T) gsm-8000-d0"
scripts/run_task.sh --name gsm-8000-d0 --cache-mib 8000 < /dev/null
sleep 15
echo "=== $(date -u +%T) gsm-8000-add0.0025"
scripts/run_task.sh --name gsm-8000-add0.0025 --cache-mib 8000 --bias 0.0025 --bias-mode add < /dev/null
echo "=== $(date -u +%T) chain done"
