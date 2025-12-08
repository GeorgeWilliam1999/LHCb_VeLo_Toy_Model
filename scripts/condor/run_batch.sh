#!/usr/bin/env bash
# =============================================================================
# Condor job wrapper script for Velo Toy experiments
# 
# This script is executed on the worker node. It:
#   1. Activates the conda environment
#   2. Runs the experiment for the specified batch
#   3. Aggregates results
#
# Arguments:
#   $1 = batch ID
#   $2 = job tag (Cluster.Process)
#   $3 = runs directory (e.g., /data/bfys/gscriven/Velo_toy/results/runs_5)
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
CONDA_PREFIX="/data/bfys/gscriven/conda"
CONDA="$CONDA_PREFIX/bin/conda"
ENV_NAME="Q_env"

BASE_DIR="/data/bfys/gscriven/Velo_toy"
PYTHON_SCRIPT="$BASE_DIR/bin/run_batch.py"

# ----------------------------------------------------------------------
# Parse arguments
# ----------------------------------------------------------------------
BATCH="${1:?Missing batch ID}"
JOBTAG="${2:?Missing job tag}"
RUNS_DIR="${3:?Missing runs directory}"

echo "========================================"
echo "Velo Toy Experiment Runner"
echo "========================================"
echo "Batch ID:   $BATCH"
echo "Job Tag:    $JOBTAG"
echo "Runs Dir:   $RUNS_DIR"
echo "Hostname:   $(hostname)"
echo "Date:       $(date)"
echo "========================================"

# ----------------------------------------------------------------------
# Environment setup
# ----------------------------------------------------------------------
if [ ! -x "$CONDA" ]; then
    echo "[ERROR] Conda not found at $CONDA" >&2
    exit 1
fi

# Check if environment exists
if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[WARN] Environment $ENV_NAME not found, creating..."
    "$CONDA" create -y -n "$ENV_NAME" python=3.12
fi

echo "Using conda run -n $ENV_NAME"

# ----------------------------------------------------------------------
# Add source to path
# ----------------------------------------------------------------------
export PYTHONPATH="$BASE_DIR/src:$BASE_DIR:${PYTHONPATH:-}"
export MPLBACKEND=Agg

# ----------------------------------------------------------------------
# Run experiment batch
# ----------------------------------------------------------------------
echo ""
echo "[INFO] Running batch $BATCH..."

"$CONDA" run -n "$ENV_NAME" python "$PYTHON_SCRIPT" \
    --batch "$BATCH" \
    --runs-dir "$RUNS_DIR" \
    --params "$RUNS_DIR/params.csv"

echo ""
echo "[INFO] Batch $BATCH complete!"
echo "========================================"
