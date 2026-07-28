#!/bin/bash
# Patch vLLM's poolside_v1 reasoning parser to default thinking=True.
# Without this, Laguna's <think> blocks leak into the content field.
#
# A vLLM upgrade SILENTLY REVERTS this patch. Re-run after any upgrade.
#
# Run on BOTH nodes.
set -euo pipefail

VENV="${VENV:-$HOME/vllm-laguna-env}"
DST="$VENV/lib/python3.12/site-packages/vllm/reasoning/poolside_v1_reasoning_parser.py"

# Find the repo root (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$SCRIPT_DIR/config/poolside_v1_reasoning_parser.py"

if [ ! -f "$SRC" ]; then
    echo "ERROR: Patch source not found at $SRC"
    exit 1
fi

if [ ! -f "$DST" ]; then
    echo "[patch] Target doesn't exist yet: $DST"
    echo "[patch] Is vLLM installed in $VENV?"
    exit 1
fi

# Check if already patched
if grep -q "PATCH 2026-07-28" "$DST" 2>/dev/null; then
    echo "[patch] Already applied. Skipping."
    exit 0
fi

# Backup and patch
cp "$DST" "${DST}.bak"
cp "$SRC" "$DST"

# Verify
if grep -q "PATCH 2026-07-28" "$DST"; then
    echo "[patch] Applied successfully."
    echo "[patch] Backup saved at ${DST}.bak"
else
    echo "ERROR: Patch did not apply correctly."
    exit 1
fi
