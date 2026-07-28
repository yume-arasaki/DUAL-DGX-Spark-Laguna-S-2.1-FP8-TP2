#!/bin/bash
# Load the Laguna TP2 recipe. Restores all 4 parts and launches.
# Usage: bash load.sh
# Prerequisites: Ray cluster running (start-ray.sh), model downloaded (download-model.sh)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RECIPE_DIR="$SCRIPT_DIR/config"
VENV="$HOME/vllm-laguna-env"
REASONING_DST="$VENV/lib/python3.12/site-packages/vllm/reasoning/poolside_v1_reasoning_parser.py"

echo "[load] Recipe dir: $SCRIPT_DIR"
source "$VENV/bin/activate" 2>/dev/null

# --- Part 1: Ensure parser patch is in the venv ---
echo "[load] Checking parser patch..."
if ! grep -q "PATCH 2026-07-28" "$REASONING_DST" 2>/dev/null; then
    echo "[load] Parser patch MISSING from venv -> applying..."
    bash "$SCRIPT_DIR/scripts/patch-parser.sh"
else
    echo "[load] Parser patch present."
fi

# --- Part 2: Verify Ray cluster ---
echo "[load] Checking Ray..."
GPUS=$(ray status 2>/dev/null | grep -oE "[0-9.]+/[0-9.]+ GPU" | head -1)
if [ -z "$GPUS" ]; then
    echo "[load] WARNING: Ray not reachable or no GPUs visible."
    echo "[load] Start Ray first:"
    echo "  spark-1: bash scripts/start-ray.sh"
    echo "  spark-2: bash scripts/start-ray.sh"
    exit 1
fi
echo "[load] Ray GPUs: $GPUS"

# --- Part 3: Source NCCL environment ---
source "$RECIPE_DIR/nccl-env.sh"

# --- Part 4: Launch vLLM ---
echo "[load] Launching vLLM with config from config/config.yaml..."
echo "[load] Model load ~8-10 min (112 GiB across 24 shards)..."
echo "[load] Monitor: tail -f ~/laguna-fp8-tp2-k0.log"

# Use the config.yaml directly
nohup vllm serve --config "$RECIPE_DIR/config.yaml" \
    > "$HOME/laguna-fp8-tp2-k0.log" 2>&1 &

echo "[load] Launched. PID=$!"
echo "[load] Log: ~/laguna-fp8-tp2-k0.log"
echo ""
echo "[load] Wait for 'Application startup complete' in the log."
echo "[load] Then verify: bash scripts/test.sh"
