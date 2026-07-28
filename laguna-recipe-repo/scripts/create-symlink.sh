#!/bin/bash
# Create path symlink on spark-2 so Ray workers can find the model.
# Ray workers inherit the model path from spark-1 (/home/yum-spark-1/models/...).
# But spark-2's home is /home/yum-spark-2/. Without this symlink, the worker
# gets HFValidationError because it can't find the local path.
#
# Run on spark-2. Requires sudo.
set -euo pipefail

SPARK1_USER="${SPARK1_USER:-yum-spark-1}"
SPARK2_USER="${SPARK2_USER:-yum-spark-2}"

echo "[symlink] Creating /home/$SPARK1_USER/models -> /home/$SPARK2_USER/models..."

# Check if symlink already exists
if [ -L "/home/$SPARK1_USER/models" ]; then
    echo "[symlink] Symlink already exists. Skipping."
    exit 0
fi

# Create the directory structure and symlink
sudo mkdir -p "/home/$SPARK1_USER"
sudo ln -s "/home/$SPARK2_USER/models" "/home/$SPARK1_USER/models"

# Verify
if ls "/home/$SPARK1_USER/models/hf/Laguna-S-2.1-FP8/config.json" >/dev/null 2>&1; then
    echo "[symlink] Verified. Model accessible at /home/$SPARK1_USER/models/hf/Laguna-S-2.1-FP8/"
else
    echo "ERROR: Symlink created but model not found. Check that mirror-to-spark2.sh ran successfully."
    exit 1
fi

echo "[symlink] Done."
