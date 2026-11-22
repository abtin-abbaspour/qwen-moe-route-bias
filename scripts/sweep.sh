#!/usr/bin/env bash
# sweep.sh -- run the arms of a list file one after another through run_arm.sh (a fresh server each), with a pause between them.
#   scripts/sweep.sh sweeps/<list>.txt        each line: <name> [run_arm.sh / serve.sh options]; # comments and blank lines skipped
# Progress goes to stdout; each arm writes its own results/runs/<stamp>-<name>/ directory.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LIST=$1
PAUSE=${PAUSE:-15}
while IFS= read -r line || [ -n "$line" ]; do
  case $line in ''|'#'*) continue;; esac
  read -r -a a <<< "$line"
  name=${a[0]}; opts=("${a[@]:1}")
  echo "=== $(date -u +%T) $name: ${opts[*]}"
  "$ROOT/scripts/run_arm.sh" --name "$name" "${opts[@]}" < /dev/null 2>&1 | grep -E "^SUMMARY|run_arm.sh|^[a-z0-9_.-]+ +engine" || echo "=== $name failed"
  sleep "$PAUSE"
done < "$LIST"
echo "=== $(date -u +%T) sweep done"
