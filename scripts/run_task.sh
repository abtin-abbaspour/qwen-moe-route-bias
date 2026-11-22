#!/usr/bin/env bash
# run_task.sh -- one task arm: a fresh server (scripts/serve.sh in the cuda13 distrobox), GPU telemetry at 10 Hz, the GSM8K test
# through scripts/gsm8k.py (results/PREREGISTRATION.md: thinking off, greedy, 768 tokens), then the server is stopped. On the host.
#   scripts/run_task.sh --name gsm-4500-d0 [--limit N] [serve.sh options, e.g. --cache-mib 4500 --bias 0.001]
# Writes results/<UTC stamp>-<name>/: manifest.txt, server.log, gpu.csv, gsm8k.jsonl, task.txt
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
NAME=task; PORT=8080; LIMIT=0; ROUTE_STATS=0
ARGS=()
while [ $# -gt 0 ]; do case $1 in
  --name) NAME=$2; shift 2;;
  --port) PORT=$2; ARGS+=(--port "$2"); shift 2;;
  --limit) LIMIT=$2; shift 2;;
  --route-stats) ROUTE_STATS=$2; shift 2;;  # off by default: the counting costs 2-5 % decode speed
  *) ARGS+=("$1"); shift;;
esac; done
ARGS+=(--route-stats "$ROUTE_STATS" -- -lv 4)  # -lv 4: the MoE cache's INFO lines (final counters) reach the log
OUT=$ROOT/results/runs/$(date -u +%Y%m%dT%H%M%SZ)-$NAME
mkdir -p "$OUT"
URL=http://127.0.0.1:$PORT

if curl -fs "$URL/health" >/dev/null 2>&1; then echo "run_task.sh: something already serves $URL" >&2; exit 2; fi

{
  echo "name: $NAME"; echo "started_utc: $(date -u +%FT%TZ)"; echo "args: ${ARGS[*]}"
  echo "llama.cpp: $(git -C "$ROOT/third_party/llama.cpp" log -1 --format='%h %s')"
  echo "llama.cpp_uncommitted_diff_sha256: $(git -C "$ROOT/third_party/llama.cpp" diff HEAD | sha256sum | cut -c1-16)"
  echo "server_binary_sha256: $(sha256sum "$ROOT/third_party/llama.cpp/build-cuda13/bin/llama-server" | cut -c1-16)"
  echo "model: Qwen3.6-35B-A3B-UD-IQ4_XS.gguf $(stat -c %s "$ROOT/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" 2>/dev/null) bytes"
  echo "driver: $(nvidia-smi --query-gpu=name,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader)"
  echo "mem_available_kib: $(grep MemAvailable /proc/meminfo | awk '{print $2}')"
  echo "gpu_mem_used_mib_before: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
  echo "--- serve.sh --dry-run"
  distrobox enter cuda13 -- "$ROOT/scripts/serve.sh" --dry-run "${ARGS[@]}" 2>&1 < /dev/null  # --dry-run first: ARGS ends in -- <server args>
} > "$OUT/manifest.txt"

stop_server() {
  pkill -f '[b]uild-cuda13/bin/llama-server' 2>/dev/null || true
  [ -n "${TELEM:-}" ] && kill "$TELEM" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap stop_server EXIT

distrobox enter cuda13 -- "$ROOT/scripts/serve.sh" "${ARGS[@]}" > "$OUT/server.log" 2>&1 < /dev/null &
for i in $(seq 1 600); do
  if curl -fs "$URL/health" 2>/dev/null | grep -q ok; then break; fi
  if ! pgrep -f '[b]uild-cuda13/bin/llama-server' >/dev/null && [ "$i" -gt 30 ]; then
    echo "run_task.sh: the server exited; see $OUT/server.log" >&2; tail -20 "$OUT/server.log" >&2; exit 1
  fi
  sleep 1
done
curl -fs "$URL/health" | grep -q ok || { echo "run_task.sh: no healthy server after 600 s" >&2; exit 1; }
echo "gpu_mem_used_mib_loaded: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" >> "$OUT/manifest.txt"

nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,memory.used,utilization.gpu,pcie.link.gen.current \
           --format=csv -lms 100 > "$OUT/gpu.csv" &
TELEM=$!

python3 -I "$ROOT/scripts/gsm8k.py" --url "$URL" --data "$ROOT/data/gsm8k/test.jsonl" --out "$OUT/gsm8k.jsonl" \
        --limit "$LIMIT" | tee "$OUT/task.txt"
echo "finished_utc: $(date -u +%FT%TZ)" >> "$OUT/manifest.txt"
stop_server; sleep 2  # the MoE cache prints its final counters at shutdown
echo "run_task.sh: results in $OUT"
