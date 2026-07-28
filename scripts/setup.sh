#!/bin/bash
# One-time setup: install python3.12-dev, create venv, install vLLM + dependencies.
# Run on BOTH nodes (spark-1 and spark-2).
set -euo pipefail

echo "[setup] Installing python3.12-dev (required for Triton JIT compilation)..."
sudo apt install -y python3.12-dev

echo "[setup] Enabling user lingering (keeps Ray alive after SSH disconnect)..."
loginctl enable-linger "$USER"

echo "[setup] Creating vLLM venv..."
python3 -m venv ~/vllm-laguna-env
source ~/vllm-laguna-env/bin/activate
pip install --upgrade pip

echo "[setup] Installing anthropic (fails to auto-resolve on aarch64)..."
pip install anthropic

echo "[setup] Installing vLLM + Ray + hf_transfer..."
pip install 'vllm>=0.25.1' hf_transfer ray

echo "[setup] Verifying..."
python3 -c 'import vllm; print(f"vLLM {vllm.__version__}")'
python3 -c 'import ray; print(f"Ray {ray.__version__}")'

echo "[setup] Verifying python3.12-dev..."
ls /usr/include/python3.12/Python.h || { echo "ERROR: Python.h not found. Re-run: sudo apt install python3.12-dev"; exit 1; }

echo ""
echo "[setup] Done on $(hostname)."
echo "[setup] Next steps:"
echo "  spark-1: bash scripts/download-model.sh"
echo "  spark-2: wait for mirror-to-spark2.sh from spark-1"
