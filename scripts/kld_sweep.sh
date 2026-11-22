#!/usr/bin/env bash
# kld_sweep.sh -- run the rows of a list file through kld.sh one after another (each in the cuda13 distrobox).
#   scripts/kld_sweep.sh sweeps/<list>.txt     each line: <name> [kld.sh options]; # comments and blank lines skipped
# Rows go to results/kld/<name>/; a row whose summary.json exists is skipped (re-running a list resumes it).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
while IFS= read -r line || [ -n "$line" ]; do
  case $line in ''|'#'*) continue;; esac
  read -r -a a <<< "$line"
  name=${a[0]}; opts=("${a[@]:1}")
  if [ -f "$ROOT/results/kld/$name/summary.json" ]; then echo "=== $name: done before, skipped"; continue; fi
  echo "=== $(date -u +%T) $name: ${opts[*]}"
  distrobox enter cuda13 -- "$ROOT/scripts/kld.sh" --out "$ROOT/results/kld/$name" "${opts[@]}" < /dev/null > /dev/null 2>&1 \
    || echo "=== $name failed"
  python3 -I "$ROOT/scripts/summarize_kld.py" "$ROOT/results/kld/$name"
  sleep 5
done < "$1"
echo "=== $(date -u +%T) kld sweep done"
