#!/usr/bin/env bash
# kld.sh -- the quality of one setting, by Killy's method: llama-perplexity in its KL-divergence mode on the wikitext-2 test text,
# 4,096-token chunks with the second half of each scored.  Run it inside the cuda13 distrobox:
#   distrobox enter cuda13 -- scripts/kld.sh --out results/kld/<name> --base [options]        writes the reference logits
#   distrobox enter cuda13 -- scripts/kld.sh --out results/kld/<name> --ref <file> [options]  scores a setting against them
#
# The engine and its switches are serve.sh's (same -ot split, cache, CPU share, bias), so a row measures the served setting.
# Options as serve.sh: --cache-mib N (8000) --bias D (0) --bias-mode add|mul --protect R --cpu-share F (0.9) --threads N (16)
#   --engine full|min --route-stats V (1: count the bias's swaps; costs nothing here but time) --drop --bias-layers L; and
#   --kv T        KV cache type (default q8_0, as served; the reference is f16)
#   --ub N        tokens per step (default 1: every token goes through the expert cache alone, as decode does without drafting;
#                 the cache and the bias serve steps of up to 32 tokens; 512 = the prompt path, every expert computed on the GPU)
#   --chunks N    4,096-token chunks (default 4: 8,188 scored positions, as RESULTS.md)
# Writes <out>/kld.log (llama-perplexity's output), <out>/cmd.txt, and with --base <out>/base.kld (about 0.5 MB a scored position).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-$ROOT/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
BIN=${BIN:-$ROOT/third_party/llama.cpp/build-cuda13/bin}
TEXT=${TEXT:-$ROOT/data/wikitext-2/wikitext-2-raw/wiki.test.raw}

OUT=; BASE=0; REF=
CACHE_MIB=8000; BIAS=0; MODE=add; PROTECT=0; CPU_SHARE=0.9; THREADS=16; ENGINE=full; ROUTE_STATS=1
KV=q8_0; UB=1; CHUNKS=4; CTX=4096
while [ $# -gt 0 ]; do case $1 in
  --out) OUT=$2; shift 2;;
  --base) BASE=1; shift;;
  --ref) REF=$2; shift 2;;
  --cache-mib) CACHE_MIB=$2; shift 2;;
  --bias) BIAS=$2; shift 2;;
  --bias-mode) MODE=$2; shift 2;;
  --protect) PROTECT=$2; shift 2;;
  --drop) DROP=1; shift;;
  --bias-layers) BIAS_LAYERS=$2; shift 2;;
  --cpu-share) CPU_SHARE=$2; shift 2;;
  --threads) THREADS=$2; shift 2;;
  --engine) ENGINE=$2; shift 2;;
  --route-stats) ROUTE_STATS=$2; shift 2;;
  --kv) KV=$2; shift 2;;
  --ub) UB=$2; shift 2;;
  --chunks) CHUNKS=$2; shift 2;;
  -h|--help) sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0;;
  *) echo "kld.sh: unknown option $1 (see --help)" >&2; exit 2;;
esac; done
[ -n "$OUT" ] || { echo "kld.sh: --out is required" >&2; exit 2; }
[ "$BASE" = 1 ] || [ -n "$REF" ] || { echo "kld.sh: give --base or --ref <file>" >&2; exit 2; }
case $REF in ""|/*) ;; *) REF=$ROOT/$REF;; esac  # a relative --ref is taken from the repo root
[ -f "$TEXT" ] || { echo "kld.sh: no $TEXT" >&2; exit 2; }
mkdir -p "$OUT"

OT='blk[.]([0-9]|[1-3][0-9])[.]ffn_(up|down|gate|gate_up)_(ch|)exps=CPU'
ARGS=(-m "$MODEL" -f "$TEXT" -c "$CTX" -b "$CTX" -ub "$UB" --chunks "$CHUNKS" -ngl 999 -fa on -t "$THREADS" --no-warmup -lv 4
      -ot "$OT" --moe-cache-mib "$CACHE_MIB" --moe-cache-cpu-share "$CPU_SHARE" --load-mode none -ctk "$KV" -ctv "$KV")
if [ "$BASE" = 1 ]; then
  ARGS+=(--kl-divergence-base "$OUT/base.kld")
else
  ARGS+=(--kl-divergence-base "$REF" --kl-divergence)
  export LLAMA_KLD_DUMP="$OUT/positions.bin"  # per-position values for paired comparisons
fi

export LLAMA_MOE_ROUTE_BIAS="$BIAS" LLAMA_MOE_ROUTE_BIAS_MODE="$MODE" LLAMA_MOE_ROUTE_STATS="$ROUTE_STATS"
if [ "$PROTECT" != 0 ]; then export LLAMA_MOE_ROUTE_PROTECT="$PROTECT"; fi
if [ "${DROP:-0}" = 1 ]; then export LLAMA_MOE_ROUTE_DROP=1; fi
if [ -n "${BIAS_LAYERS:-}" ]; then export LLAMA_MOE_ROUTE_BIAS_LAYERS="$BIAS_LAYERS"; fi
export OMP_PROC_BIND=close OMP_PLACES=cores
if [ "$ENGINE" = full ]; then
  # the same switches as serve.sh (glm53-offload recipes/common.sh, model-agnostic part)
  export LLAMA_TWIN_BACKENDS=0 CUDA_SCALE_LAUNCH_QUEUES=4x
  export LLAMA_MOE_ASYNC_FILL=1 LLAMA_MOE_ASYNC_FILL_TOKENS=8 LLAMA_MOE_CACHE_LFU=64 LLAMA_MOE_DEFERRED_PLAN=1
  export LLAMA_MOE_WEIGHT_INPLACE=1 LLAMA_MOE_UPLOAD_RING=1 LLAMA_MOE_UPLOAD_RING_MEMCPY=1 GGML_CUDA_RING_MIXED=1
  export LLAMA_MOE_ZERO_SKIP=1 LLAMA_MOE_CACHE_EQUAL_COVER=1 LLAMA_MOE_CACHE_BORROW_RING=1 LLAMA_MTP_MOE_CACHE_MIB=0
  export GGML_SCHED_GALLOCS=1 GGML_SCHED_ASYNC_HOST_INPUT=2 GGML_SCHED_COPIES=2 GGML_SCHED_PAIR_GATE_UP=4097
  export GGML_CUDA_Q8_1_DEDUPE=1 GGML_CUDA_MMVQ_SMALLK_MT=2
  export LLAMA_KV_LAZY=1 LLAMA_GRAPH_DUAL_SCHED=1 LLAMA_KV_SEQ_RM_BOUNDED=1 LLAMA_SYNC_SKIP_IDLE=1 LLAMA_SHORT_BATCH_MAX=128
  export LLAMA_PP_ONE_DEVICE=1
fi

{
  echo "started_utc: $(date -u +%FT%TZ)"
  echo "settings: cache ${CACHE_MIB} MiB, bias ${BIAS} (${MODE}), protect ${PROTECT}, drop ${DROP:-0}, layers ${BIAS_LAYERS:-all}, cpu share ${CPU_SHARE}, kv ${KV}, ub ${UB}," \
       "chunks ${CHUNKS}, engine ${ENGINE}, base ${BASE}, ref ${REF:-none}"
  env | grep -E '^(LLAMA_|GGML_|OMP_|CUDA_)' | sort
  printf '%q ' "$BIN/llama-perplexity" "${ARGS[@]}"; echo
} > "$OUT/cmd.txt"
t0=$(date +%s)
"$BIN/llama-perplexity" "${ARGS[@]}" > "$OUT/kld.log" 2>&1
echo "elapsed_s: $(( $(date +%s) - t0 ))" >> "$OUT/cmd.txt"
grep -E "Mean PPL|Mean    KLD|Same top|RMS Δp|Final estimate|cache-aware|ubatch <= 8|route total" "$OUT/kld.log" | cut -c1-220 || true
