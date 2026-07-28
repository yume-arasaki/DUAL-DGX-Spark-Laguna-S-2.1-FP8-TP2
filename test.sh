#!/bin/bash
# Health check + decode speed test.
# Run on spark-1 after load.sh completes.
set -euo pipefail

ENDPOINT="${ENDPOINT:-http://localhost:8000}"

echo "[test] Checking API health..."
if ! curl -s --max-time 5 "$ENDPOINT/v1/models" | python3 -m json.tool >/dev/null 2>&1; then
    echo "ERROR: Server not responding at $ENDPOINT"
    echo "Check: tail -20 ~/laguna-fp8-tp2-k0.log"
    exit 1
fi

echo "[test] Models:"
curl -s "$ENDPOINT/v1/models" | python3 -m json.tool

echo ""
echo "[test] Sending test completion..."
START=$(python3 -c 'import time; print(time.time())')

RESULT=$(curl -s "$ENDPOINT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "laguna-fp8-tp2",
        "prompt": "Explain tensor parallelism in one sentence.",
        "max_tokens": 50,
        "temperature": 0.0
    }')

END=$(python3 -c 'import time; print(time.time())')

echo "$RESULT" | python3 -c "
import json, sys, time
data = json.load(sys.stdin)
usage = data.get('usage', {})
pt = usage.get('prompt_tokens', 0)
ct = usage.get('completion_tokens', 0)
text = data['choices'][0]['text'][:200]
print(f'Prompt tokens: {pt}')
print(f'Completion tokens: {ct}')
print(f'Output: {text}')
"

echo ""
echo "[test] Checking vLLM metrics..."
curl -s "$ENDPOINT/metrics" | grep -E 'vllm:num_requests_running|vllm:gpu_cache_usage_perc|vllm:prompt_tokens_total|vllm:generation_tokens_total' | head -8

echo ""
echo "[test] Checking GPU status..."
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "(nvidia-smi unavailable)"

echo ""
echo "[test] Done. If decode ran, the recipe is working."
