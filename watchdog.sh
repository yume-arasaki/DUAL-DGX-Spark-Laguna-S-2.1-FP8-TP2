#!/bin/bash
# Watchdog: auto-relaunch vLLM if the engine dies.
# Designed to be run via cron every 5 minutes.
#
# Install: crontab -e
#   */5 * * * * bash ~/laguna-recipe/scripts/watchdog.sh >> ~/laguna-watchdog.log 2>&1
#
# IMPORTANT: Pause the watchdog before manual restarts (else two servers fight port 8000).
set -uo pipefail

ENDPOINT="http://localhost:8000"
LOG="$HOME/laguna-fp8-tp2-k0.log"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Check if vLLM is responding
if curl -s --max-time 10 "$ENDPOINT/v1/models" >/dev/null 2>&1; then
    # Server healthy, check for EngineDeadError in recent logs
    if tail -50 "$LOG" 2>/dev/null | grep -q "EngineDeadError\|RPC call timed out"; then
        echo "[$(date)] WARNING: EngineDeadError detected in logs but server responding. Monitoring."
    fi
    exit 0
fi

echo "[$(date)] Server not responding. Checking for processes..."

# Check if vllm processes exist but are stuck
if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[$(date)] vLLM processes found but not responding. Killing gracefully..."
    # Graceful SIGTERM (NOT kill -9, which leaks Ray placement groups)
    pkill -SIGTERM -f "vllm serve"
    sleep 5

    # Force kill if still alive
    if pgrep -f "vllm serve" >/dev/null 2>&1; then
        echo "[$(date)] Graceful kill failed. Using SIGKILL..."
        pkill -9 -f "vllm serve"
        sleep 3
    fi
fi

# Verify Ray cluster still has 2 GPUs free
source ~/vllm-laguna-env/bin/activate 2>/dev/null
GPUS=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | head -1)
echo "[$(date)] Ray GPUs: ${GPUS:-unavailable}"

if [ -z "$GPUS" ]; then
    echo "[$(date)] Ray not reachable. Cannot relaunch."
    exit 1
fi

# Check if placement groups are leaked (2.0/2.0 used with no vllm running)
if echo "$GPUS" | grep -q "2.0/2.0"; then
    echo "[$(date)] Placement group leaked. Restarting Ray..."
    ray stop --force
    rm -rf /tmp/ray
    sleep 3
    source "$SCRIPT_DIR/config/nccl-env.sh"
    ray start --head --node-ip-address=10.0.0.1 --port=6379
    sleep 5
fi

# Relaunch
echo "[$(date)] Relaunching vLLM..."
bash "$SCRIPT_DIR/scripts/load.sh"
echo "[$(date)] Relaunch initiated. Check ~/laguna-fp8-tp2-k0.log"
