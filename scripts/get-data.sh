#!/usr/bin/env bash
# get-data.sh -- download the test data and the model, and check their sha256:
#   data/wikitext-2/wikitext-2-raw/wiki.test.raw   KL-divergence text (wikitext-2 raw test split, the ggml-org CI copy; CC BY-SA)
#   data/gsm8k/test.jsonl                          GSM8K test set, 1,319 questions (the original OpenAI release; MIT)
#   models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf   the model (unsloth/Qwen3.6-35B-A3B-MTP-GGUF, 18.2 GB; with --model)
#   scripts/get-data.sh [--model]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL=0; [ "${1:-}" = --model ] && MODEL=1

get() {  # get <url> <file> <sha256>: download unless present, then verify
  local url=$1 file=$2 sum=$3
  mkdir -p "$(dirname "$file")"
  if [ ! -f "$file" ]; then
    echo "get-data.sh: $url"
    curl -fL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  echo "$sum  $file" | sha256sum -c --quiet || { echo "get-data.sh: checksum mismatch for $file" >&2; exit 1; }
}

W=$ROOT/data/wikitext-2
get https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip "$W/wikitext-2-raw-v1.zip" \
    ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11
[ -f "$W/wikitext-2-raw/wiki.test.raw" ] || unzip -q -o "$W/wikitext-2-raw-v1.zip" -d "$W"
echo "173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08  $W/wikitext-2-raw/wiki.test.raw" | sha256sum -c --quiet
get https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl "$ROOT/data/gsm8k/test.jsonl" \
    3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14
if [ "$MODEL" = 1 ]; then
  get https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
      "$ROOT/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf" \
      df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
fi
echo "get-data.sh: data ready$([ "$MODEL" = 1 ] && echo ", model ready")"
