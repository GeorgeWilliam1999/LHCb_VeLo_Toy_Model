#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------
# Conda / environment setup
# ----------------------------------------------------------------------
PREFIX="/data/bfys/gscriven/conda"
CONDA="$PREFIX/bin/conda"
ENV="Q_env"

if [ ! -x "$CONDA" ]; then
  echo "Conda not found at $CONDA" >&2
  exit 1
fi

if ! "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV"; then
  "$CONDA" create -y -n "$ENV" python=3.12
fi

# ----------------------------------------------------------------------
# Arguments from submit file
#   $1 = batch ID
#   $2 = job tag (Cluster.Process)
#   $3 = runs root (e.g. /data/.../Velo_toy/runs_5)
# ----------------------------------------------------------------------
BATCH="$1"
JOBTAG="$2"
RUNS_ROOT="$3"

BASE_DIR="/data/bfys/gscriven/Velo_toy"
PARAMS="${RUNS_ROOT}/params.csv"

BATCH_DIR="${RUNS_ROOT}/batch_${BATCH}"
JOB_DIR="${BATCH_DIR}/${JOBTAG}"
LOGDIR="${BASE_DIR}/logs"

mkdir -p "${JOB_DIR}" "${LOGDIR}"

export MPLBACKEND=Agg

echo "============================================================"
echo " Velo toy full workflow for batch ${BATCH} (job ${JOBTAG})"
echo "============================================================"
date
echo "Experiment dir: ${RUNS_ROOT}"
echo "Params file:    ${PARAMS}"
echo "Batch dir:      ${BATCH_DIR}"
echo "Job run dir:    ${JOB_DIR}"
echo "------------------------------------------------------------"

# ----------------------------------------------------------------------
# 1) Generation step
# ----------------------------------------------------------------------
echo "[STEP] Generation"
"$CONDA" run -n "$ENV" python "${BASE_DIR}/velo_workflow.py" \
  generate \
  --params "${PARAMS}" \
  --batch "${BATCH}" \
  --outdir "${JOB_DIR}"

echo "[DONE] Generation"
echo "------------------------------------------------------------"

# ----------------------------------------------------------------------
# 2) Aggregation step (batch-level outputs)
# ----------------------------------------------------------------------
echo "[STEP] Aggregation"
"$CONDA" run -n "$ENV" python "${BASE_DIR}/velo_workflow.py" \
  aggregate \
  --runs-dir "${RUNS_ROOT}" \
  --batch "${BATCH}" \
  --out-dir "${BATCH_DIR}" \
  --store-full

echo "[DONE] Aggregation"
date
echo "Workflow complete for batch ${BATCH} (job ${JOBTAG})"
