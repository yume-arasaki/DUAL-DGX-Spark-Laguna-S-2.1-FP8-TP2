#!/bin/bash
# Download the FP8 checkpoint (113 GB) to spark-1.
# Requires HuggingFace token. Free tier works but is rate-limited.
set -euo pipefail

source ~/vllm-laguna-env/bin/activate

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first:"
    echo "  export HF_TOKEN=hf_YOUR_TOKEN_HERE"
    echo "  Get one at: https://huggingface.co/settings/tokens"
    exit 1
fi

export HF_XET_HIGH_PERFORMANCE=1

echo "[download] Downloading Laguna-S-2.1-FP8 (113 GB, 24 shards)..."
hf download poolside/Laguna-S-2.1-FP8 --local-dir ~/models/hf/Laguna-S-2.1-FP8

echo "[download] Verifying..."
SHARDS=$(ls ~/models/hf/Laguna-S-2.1-FP8/*.safetensors 2>/dev/null | wc -l)
if [ "$SHARDS" -ne 24 ]; then
    echo "ERROR: Expected 24 safetensors shards, found $SHARDS"
    exit 1
fi

SIZE=$(du -sh ~/models/hf/Laguna-S-2.1-FP8/ | cut -f1)
echo "[download] Done. $SHARDS shards, $SIZE total."

echo ""
echo "[download] Next: bash scripts/mirror-to-spark2.sh"
