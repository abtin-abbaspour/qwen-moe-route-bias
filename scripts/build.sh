#!/usr/bin/env bash
# build.sh -- build the patched llama.cpp in the cuda13 distrobox (CUDA 13.0, as glm53-offload pins it: 13.2 miscompiles IQ kernels
# on sm_120). Flags follow glm53-offload docker/Dockerfile, plus test-backend-ops and llama-bench.
#   distrobox enter cuda13 -- scripts/build.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC=$ROOT/third_party/llama.cpp
command -v nvcc >/dev/null || { echo "build.sh: nvcc not found; run it inside the cuda13 distrobox" >&2; exit 2; }
nvcc --version | grep -q 'release 13.0' || { echo "build.sh: expected CUDA 13.0, got: $(nvcc --version | tail -1)" >&2; exit 2; }

CPU_FLAGS=(-DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON -DGGML_AVX512_VBMI=ON
           -DGGML_FMA=ON -DGGML_F16C=ON)
cmake -S "$SRC" -B "$SRC/build-cuda13" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
      -DGGML_NATIVE=OFF "${CPU_FLAGS[@]}" -DLLAMA_CURL=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=ON \
      -DLLAMA_USE_PREBUILT_UI=OFF  # the pinned commit's UI bundle (b3) is gone from the HF bucket; the server runs without its web UI
cmake --build "$SRC/build-cuda13" --config Release -j "$(nproc)" \
      --target llama-server llama-perplexity llama-quantize llama-bench test-backend-ops
ls -la "$SRC/build-cuda13/bin"
