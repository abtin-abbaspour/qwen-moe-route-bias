#!/usr/bin/env bash
# the quality harness check: a reference, then delta-0 rows at the cache sizes in question, one-token and three-token steps,
# one biased row (does the bias engage in llama-perplexity?) and the KV-type floor; run on the host, each row in the cuda13 box
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
K=$ROOT/results/kld
REF=$K/ref-f16-ub512/base.kld
run() { local name=$1; shift; echo "=== $(date -u +%T) $name: $*"
        distrobox enter cuda13 -- "$ROOT/scripts/kld.sh" --out "$K/$name" "$@" < /dev/null 2>&1 | tail -12
        python3 -I "$ROOT/scripts/summarize_kld.py" "$K/$name"; sleep 10; }
run ref-f16-ub512 --base --kv f16 --ub 512 --cache-mib 8000 --route-stats 0
run c8000-q8-ub512-d0 --ref "$REF" --ub 512 --cache-mib 8000 --route-stats 0
run c4500-ub1-d0      --ref "$REF" --ub 1 --cache-mib 4500
run c4500-ub3-d0      --ref "$REF" --ub 3 --cache-mib 4500
run c4500-ub1-add0.005 --ref "$REF" --ub 1 --cache-mib 4500 --bias 0.005 --bias-mode add
run c3000-ub1-d0      --ref "$REF" --ub 1 --cache-mib 3000
run c2000-ub1-d0      --ref "$REF" --ub 1 --cache-mib 2000
run c4500-ub1-d0-rep  --ref "$REF" --ub 1 --cache-mib 4500
echo "=== $(date -u +%T) kld check done"
