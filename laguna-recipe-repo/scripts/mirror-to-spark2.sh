#!/bin/bash
# Mirror model weights from spark-1 to spark-2 over the direct RoCE link.
# Run on spark-1 after download-model.sh completes.
set -euo pipefail

SPARK2_IP="${SPARK2_IP:-10.0.0.2}"
SPARK2_USER="${SPARK2_USER:-yum-spark-2}"

echo "[mirror] Syncing FP8 checkpoint to spark-2 ($SPARK2_IP)..."
ssh "$SPARK2_USER@$SPARK2_IP" "mkdir -p ~/models/hf/Laguna-S-2.1-FP8"

rsync -a --partial --inplace \
    -e 'ssh -o ConnectTimeout=10' \
    ~/models/hf/Laguna-S-2.1-FP8/ \
    "$SPARK2_USER@$SPARK2_IP:~/models/hf/Laguna-S-2.1-FP8/"

echo "[mirror] Verifying on spark-2..."
SHARDS=$(ssh "$SPARK2_USER@$SPARK2_IP" 'ls ~/models/hf/Laguna-S-2.1-FP8/*.safetensors | wc -l')
if [ "$SHARDS" -ne 24 ]; then
    echo "ERROR: spark-2 has $SHARDS shards, expected 24"
    exit 1
fi

echo "[mirror] Done. $SHARDS shards on spark-2."
echo ""
echo "[mirror] Next: bash scripts/create-symlink.sh (on spark-2)"
