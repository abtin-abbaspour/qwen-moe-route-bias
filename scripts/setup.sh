#!/usr/bin/env bash
# setup.sh -- fetch the pinned sources into third_party/ and apply the patches, in order:
#   1. llama.cpp at d02b4c0d (github.com/ggml-org/llama.cpp)
#   2. glm53-offload at b5e86ee (github.com/killy-netsphere/glm53-offload): its patches/glm53-offload.patch onto llama.cpp
#      (the expert cache, the route bias, the engine switches), and its bench/ prompts and probe used by scripts/run_arm.sh
#   3. this repo's patches/0001-*.patch and patches/0002-*.patch
# Then build inside the cuda13 distrobox: distrobox enter cuda13 -- scripts/build.sh   (see README.md)
#   scripts/setup.sh                 THIRD_PARTY=<dir> puts the checkouts elsewhere
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
TP=${THIRD_PARTY:-$ROOT/third_party}
LLAMA_URL=https://github.com/ggml-org/llama.cpp
LLAMA_COMMIT=d02b4c0d66796e0a44380611d038123fefb9fe83
GLM_URL=https://github.com/killy-netsphere/glm53-offload
GLM_COMMIT=b5e86ee9d63df629cda41944cbf85811f89726a6
GLM_PATCH_SHA256=9110942d0132c75aafadf600851007c75d427c7c4da4b5ccc420381314f347c8

fetch() {  # fetch <dir> <url> <commit>: a shallow checkout of one commit
  local dir=$1 url=$2 commit=$3
  git init -q "$dir"
  git -C "$dir" remote add origin "$url"
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
}

if [ -e "$TP/llama.cpp" ] || [ -e "$TP/glm53-offload" ]; then
  echo "setup.sh: $TP already holds llama.cpp or glm53-offload; remove them to set up again" >&2; exit 2
fi
mkdir -p "$TP"
echo "setup.sh: glm53-offload ${GLM_COMMIT:0:7}"
fetch "$TP/glm53-offload" "$GLM_URL" "$GLM_COMMIT"
echo "$GLM_PATCH_SHA256  $TP/glm53-offload/patches/glm53-offload.patch" | sha256sum -c --quiet
echo "setup.sh: llama.cpp ${LLAMA_COMMIT:0:7}"
fetch "$TP/llama.cpp" "$LLAMA_URL" "$LLAMA_COMMIT"
echo "setup.sh: applying glm53-offload.patch, then patches/0001 and 0002"
git -C "$TP/llama.cpp" apply "$TP/glm53-offload/patches/glm53-offload.patch"
git -C "$TP/llama.cpp" apply "$ROOT"/patches/0001-*.patch
git -C "$TP/llama.cpp" apply "$ROOT"/patches/0002-*.patch
echo "setup.sh: done. Next: distrobox enter cuda13 -- scripts/build.sh"
