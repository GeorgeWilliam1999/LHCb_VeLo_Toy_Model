#!/usr/bin/env bash
# =============================================================================
# Aggregate results from all batches in a run
# =============================================================================
set -euo pipefail

CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Q_env"
BASE_DIR="/data/bfys/gscriven/Velo_toy"

RUNS_DIR="${1:?Missing runs directory}"

echo "========================================"
echo "Aggregating results from: $RUNS_DIR"
echo "========================================"

# Environment setup
eval "$("$CONDA" shell.bash hook)"
conda activate "$ENV_NAME"

export PYTHONPATH="$BASE_DIR/src:${PYTHONPATH:-}"

python "$BASE_DIR/bin/aggregate_results.py" --runs-dir "$RUNS_DIR"

echo "✅ Aggregation complete!"
