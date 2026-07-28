#!/bin/bash
# Start Ray cluster: head node on spark-1, worker on spark-2.
# Run the head command on spark-1, then the worker command on spark-2.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$SCRIPT_DIR/config/nccl-env.sh"
source ~/vllm-laguna-env/bin/activate

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
PORT="${PORT:-6379}"

# Clean slate
echo "[ray] Stopping any existing Ray..."
ray stop --force 2>/dev/null || true
rm -rf /tmp/ray

# Determine if we're the head or worker
MY_IP=$(hostname -I | awk '{print $1}')

if [ "$MY_IP" = "$HEAD_IP" ] || [ "$(hostname)" = "edgexpert-3d2e" ]; then
    # Head node
    echo "[ray] Starting HEAD on $HEAD_IP..."
    ray start --head \
        --node-ip-address="$HEAD_IP" \
        --port="$PORT"

    echo ""
    echo "[ray] Head started. Now run on spark-2 ($WORKER_IP):"
    echo "  source ~/laguna-recipe/config/nccl-env.sh"
    echo "  source ~/vllm-laguna-env/bin/activate"
    echo "  ray start --address=$HEAD_IP:$PORT --node-ip-address=$WORKER_IP"

    echo ""
    echo "[ray] After spark-2 joins, verify:"
    echo "  ray status  # should show 2 nodes, 2.0 GPU"

else
    # Worker node
    echo "[ray] Starting WORKER, connecting to head at $HEAD_IP:$PORT..."
    ray start --address="$HEAD_IP:$PORT" \
        --node-ip-address="$WORKER_IP"

    echo ""
    echo "[ray] Worker started. Verify on spark-1: ray status"
fi
