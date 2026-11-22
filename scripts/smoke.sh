#!/usr/bin/env bash
# smoke.sh -- quick check of one server setting: start it (library logs on), one warmup request, then the same 512-token thinking
# request twice; prints decode speed, draft acceptance and the MoE cache lines of the log, then stops the server. Run it on the host.
#   scripts/smoke.sh <label> [serve.sh options]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LABEL=$1; shift
LOG=$ROOT/results/smoke/server-$LABEL.log
mkdir -p "$ROOT/results/smoke"
URL=http://127.0.0.1:8080
REQ='{"model":"q","messages":[{"role":"user","content":"Write a complete Python module implementing a red-black tree with insert, delete and search. Include docstrings."}],"max_tokens":512,"temperature":0,"chat_template_kwargs":{"enable_thinking":true}}'
WARM='{"model":"q","messages":[{"role":"user","content":"Say hello in five languages."}],"max_tokens":64,"temperature":0}'

stop_all() { # stop any llama-server and wait until it is gone (a server shutting down still holds the port and 15 GB pinned)
  pkill -f "[b]uild-cuda13/bin/llama-server" 2>/dev/null || true
  for _i in $(seq 1 120); do pgrep -f "[b]uild-cuda13/bin/llama-server" >/dev/null || return 0; sleep 0.5; done
  echo "smoke.sh: a llama-server would not stop" >&2; return 1
}
trap stop_all EXIT
stop_all
distrobox enter cuda13 -- "$ROOT/scripts/serve.sh" --log-every 500 "$@" -- -lv 4 > "$LOG" 2>&1 < /dev/null &
for _ in $(seq 1 300); do curl -fs "$URL/health" 2>/dev/null | grep -q ok && break; sleep 1; done
curl -fs "$URL/health" | grep -q ok || { echo "smoke.sh: no server"; tail -20 "$LOG"; exit 1; }
grep -q "couldn't bind" "$LOG" && { echo "smoke.sh: our server could not bind the port"; exit 1; }

curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$WARM" >/dev/null
for i in 1 2; do
  curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" | python3 -I -c "
import json, sys, hashlib
r = json.load(sys.stdin); t = r['timings']; m = r['choices'][0]['message']
text = (m.get('reasoning_content') or '') + (m.get('content') or '')
print(f\"$LABEL run $i: {t['predicted_n']} tok at {t['predicted_per_second']:.1f} tok/s, draft accepted {t.get('draft_n_accepted')}/{t.get('draft_n')}, \"
      f\"text md5 {hashlib.md5(text.encode()).hexdigest()[:10]}, starts: {text[:70]!r}\")"
done
stop_all; sleep 1
grep -E 'cache-aware|ubatch <= 8|ubatch  > 8' "$LOG" | cut -c1-200
