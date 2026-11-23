#!/usr/bin/env bash
# serve.sh -- start the patched llama-server for this project's runs. Run it inside the cuda13 distrobox:
#   distrobox enter cuda13 -- scripts/serve.sh [options] [-- extra llama-server args]
#
# Adapted from killy-netsphere/glm53-offload (recipes/common.sh + 1gpu.sh): the routed experts stay in host RAM, the GPU keeps
# everything else plus an expert cache, most cache misses are computed on the CPU, and the MTP head drafts 2 tokens a step.
# The GLM-5.3-specific kernels and switches (sparse attention, lightning indexer, MLA tiles, IQ3_XXS fusion, two-card paths)
# are left out.
#
# Options:
#   --cache-mib N     GPU expert cache size (default 8000)
#   --bias D          route bias for the cached experts (default 0 = the plain router)
#   --bias-mode M     add: probs + D (the patch's rule); mul: probs * exp(D), i.e. logits + D (default add)
#   --drop            the bias's swapped-in experts get weight 0, the rest renormalized (LLAMA_MOE_ROUTE_DROP)
#   --bias-layers L   the bias only in layers L, e.g. 0-7,32-39 (LLAMA_MOE_ROUTE_BIAS_LAYERS; default all)
#   --protect R       keep the R best experts by the unbiased score selected (LLAMA_MOE_ROUTE_PROTECT, default 0 = off)
#   --cpu-share F     share of misses computed on the CPU once the cache is full (default 0.9, as shipped)
#   --ctx N           context (default 65536, as in the post's 64K configurations)
#   --threads N       CPU threads (default 16 = the physical cores, both CCDs)
#   --port P          (default 8080)
#   --log-every N     log the cache hit rate every N lookups (LLAMA_MOE_CACHE_LOG_EVERY, default off)
#   --route-stats V   LLAMA_MOE_ROUTE_STATS: 1 = count the bias's swaps (default), 0 = off, a path = also per-token records
#   --no-mtp          no MTP drafting
#   --engine E        full: the model-agnostic engine switches of common.sh; min: the cache and the bias only (default full)
#   --host-thp        pinned host buffers on transparent huge pages (GGML_CUDA_HOST_THP=1, glm53-offload patch 0037; off as shipped)
#   --no-aslr         start the server with address-space randomization off (setarch -R): the same memory layout every start
#   --no-work-share   the stock CPU split: each small expert matrix of a decode step divided into one fixed share per thread.
#                     By default (--work-share) those matrices go to one queue the CPU threads share (GGML_CPU_MMID_DYN=1,
#                     patches/0003), which keeps decode speed steady from one server start to the next; same output either way
#   --work-share-rows N  rows per piece of that queue (GGML_CPU_MMID_DYN_ROWS, default 32)
#   --dry-run         print the command and environment, start nothing
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=${MODEL:-$ROOT/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf}
BIN=${BIN:-$ROOT/third_party/llama.cpp/build-cuda13/bin}

CACHE_MIB=8000; BIAS=0; ROUTE_STATS=1; MODE=add; PROTECT=0; CPU_SHARE=0.9; CTX=65536; THREADS=16; PORT=8080; LOG_EVERY=0; MTP=1; ENGINE=full
DRY=0; WORK_SHARE=1
while [ $# -gt 0 ]; do case $1 in
  --cache-mib) CACHE_MIB=$2; shift 2;;
  --bias) BIAS=$2; shift 2;;
  --bias-mode) MODE=$2; shift 2;;
  --protect) PROTECT=$2; shift 2;;
  --drop) DROP=1; shift;;
  --bias-layers) BIAS_LAYERS=$2; shift 2;;
  --cpu-share) CPU_SHARE=$2; shift 2;;
  --ctx) CTX=$2; shift 2;;
  --threads) THREADS=$2; shift 2;;
  --port) PORT=$2; shift 2;;
  --log-every) LOG_EVERY=$2; shift 2;;
  --route-stats) ROUTE_STATS=$2; shift 2;;
  --no-mtp) MTP=0; shift;;
  --engine) ENGINE=$2; shift 2;;
  --host-thp) HOST_THP=1; shift;;
  --no-aslr) NO_ASLR=1; shift;;
  --work-share) WORK_SHARE=1; shift;;
  --no-work-share) WORK_SHARE=0; shift;;
  --work-share-rows) WORK_SHARE_ROWS=$2; shift 2;;
  --dry-run) DRY=1; shift;;
  --) shift; break;;
  -h|--help) sed -n '2,/^set -/p' "$0" | sed '$d'; exit 0;;
  *) echo "serve.sh: unknown option $1 (see --help)" >&2; exit 2;;
esac; done
case $MODE in add|mul) ;; *) echo "serve.sh: --bias-mode is add or mul" >&2; exit 2;; esac
case $ENGINE in full|min) ;; *) echo "serve.sh: --engine is full or min" >&2; exit 2;; esac

# routed experts of the 40 MoE layers in host memory; blk.40 (the MTP block) stays whole on the GPU
OT='blk[.]([0-9]|[1-3][0-9])[.]ffn_(up|down|gate|gate_up)_(ch|)exps=CPU'
ARGS=(-m "$MODEL" -ngl 999 -fa on --jinja --parallel 1 --metrics --no-warmup
      -c "$CTX" -t "$THREADS" --host 127.0.0.1 --port "$PORT"
      -ot "$OT" --moe-cache-mib "$CACHE_MIB" --moe-cache-cpu-share "$CPU_SHARE" --load-mode none
      -ctk q8_0 -ctv q8_0)
if [ "$MTP" = 1 ]; then
  ARGS+=(--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-backend-sampling)
fi

export LLAMA_MOE_ROUTE_BIAS="$BIAS" LLAMA_MOE_ROUTE_BIAS_MODE="$MODE" LLAMA_MOE_ROUTE_STATS="$ROUTE_STATS"
if [ "$PROTECT" != 0 ]; then export LLAMA_MOE_ROUTE_PROTECT="$PROTECT"; fi
if [ "${DROP:-0}" = 1 ]; then export LLAMA_MOE_ROUTE_DROP=1; fi
if [ -n "${BIAS_LAYERS:-}" ]; then export LLAMA_MOE_ROUTE_BIAS_LAYERS="$BIAS_LAYERS"; fi
if [ "$LOG_EVERY" != 0 ]; then export LLAMA_MOE_CACHE_LOG_EVERY="$LOG_EVERY"; fi
export OMP_PROC_BIND=close OMP_PLACES=cores
if [ "${HOST_THP:-0}" = 1 ]; then export GGML_CUDA_HOST_THP=1; fi
if [ "$WORK_SHARE" = 1 ]; then
  export GGML_CPU_MMID_DYN=1
  if [ -n "${WORK_SHARE_ROWS:-}" ]; then export GGML_CPU_MMID_DYN_ROWS="$WORK_SHARE_ROWS"; fi
fi
PRE=()
if [ "${NO_ASLR:-0}" = 1 ]; then PRE=(setarch "$(uname -m)" -R); fi

if [ "$ENGINE" = full ]; then
  # the model-agnostic switches of glm53-offload recipes/common.sh (MoE cache, upload ring, scheduler, KV) and 1gpu.sh
  export LLAMA_TWIN_BACKENDS=0 CUDA_SCALE_LAUNCH_QUEUES=4x
  export LLAMA_MOE_ASYNC_FILL=1 LLAMA_MOE_ASYNC_FILL_TOKENS=8 LLAMA_MOE_CACHE_LFU=64 LLAMA_MOE_DEFERRED_PLAN=1
  export LLAMA_MOE_WEIGHT_INPLACE=1 LLAMA_MOE_UPLOAD_RING=1 LLAMA_MOE_UPLOAD_RING_MEMCPY=1 GGML_CUDA_RING_MIXED=1
  export LLAMA_MOE_ZERO_SKIP=1 LLAMA_MOE_CACHE_EQUAL_COVER=1 LLAMA_MOE_CACHE_BORROW_RING=1 LLAMA_MTP_MOE_CACHE_MIB=0
  export GGML_SCHED_GALLOCS=1 GGML_SCHED_ASYNC_HOST_INPUT=2 GGML_SCHED_COPIES=2 GGML_SCHED_PAIR_GATE_UP=4097
  export GGML_CUDA_Q8_1_DEDUPE=1 GGML_CUDA_MMVQ_SMALLK_MT=2
  export LLAMA_KV_LAZY=1 LLAMA_GRAPH_DUAL_SCHED=1 LLAMA_KV_SEQ_RM_BOUNDED=1 LLAMA_SYNC_SKIP_IDLE=1 LLAMA_SHORT_BATCH_MAX=128
  export LLAMA_PP_ONE_DEVICE=1
fi

echo "serve.sh: cache ${CACHE_MIB} MiB, bias ${BIAS} (${MODE}), protect ${PROTECT}, cpu share ${CPU_SHARE}, ctx ${CTX}," \
     "threads ${THREADS}, mtp ${MTP}, engine ${ENGINE}, work share $([ "$WORK_SHARE" = 1 ] && echo on || echo off), aslr $([ "${NO_ASLR:-0}" = 1 ] && echo off || echo on)" >&2
if [ "$DRY" = 1 ]; then
  env | grep -E '^(LLAMA_|GGML_|OMP_|CUDA_)' | sort
  printf '%q ' "${PRE[@]}" "$BIN/llama-server" "${ARGS[@]}" "$@"; echo
  exit 0
fi
exec "${PRE[@]}" "$BIN/llama-server" "${ARGS[@]}" "$@"
