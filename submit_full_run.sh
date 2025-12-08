#!/usr/bin/env bash
set -euo pipefail

BASE="/data/bfys/gscriven/Velo_toy"
cd "${BASE}"

SUB_FILE="${BASE}/velo_full_workflow.sub"
PARAM_GEN="${BASE}/gen_params.py"

if [ ! -f "${PARAM_GEN}" ]; then
  echo "[ERROR] Missing parameter generator script: ${PARAM_GEN}" >&2
  exit 1
fi

# Ensure logs directory exists for Condor stdout/stderr
mkdir -p "${BASE}/logs"

# ----------------------------------------------------------------------
# Find next runs_N directory
# ----------------------------------------------------------------------
LAST_NUM=$(find . -maxdepth 1 -type d -name 'runs_*' -printf '%f\n' \
  | sed -E 's/^runs_([0-9]+)$/\1/' \
  | sort -n \
  | tail -1 || echo "")

if [ -z "${LAST_NUM}" ]; then
  LAST_NUM=0
fi

NEXT_NUM=$((LAST_NUM + 1))
RUNS_ROOT="${BASE}/runs_${NEXT_NUM}"

echo "[INFO] Creating new experiment directory: ${RUNS_ROOT}"
mkdir -p "${RUNS_ROOT}"

# ----------------------------------------------------------------------
# Run parameter generator inside the new run directory
# ----------------------------------------------------------------------
echo "[INFO] Running gen_params.py to create ${RUNS_ROOT}/params.csv and batches.txt"
(
  cd "${RUNS_ROOT}"
  python "${PARAM_GEN}"
)

if [ ! -f "${RUNS_ROOT}/params.csv" ]; then
  echo "[ERROR] gen_params.py did not produce params.csv in ${RUNS_ROOT}" >&2
  exit 1
fi

if [ ! -f "${RUNS_ROOT}/batches.txt" ]; then
  echo "[ERROR] gen_params.py did not produce batches.txt in ${RUNS_ROOT}" >&2
  exit 1
fi

echo "[INFO] params.csv  : ${RUNS_ROOT}/params.csv"
echo "[INFO] batches.txt : ${RUNS_ROOT}/batches.txt"

echo "[INFO] Submitting Condor jobs for new experiment: runs_${NEXT_NUM}"

condor_submit \
  -append "runsroot=${RUNS_ROOT}" \
  -append "batches_file=${RUNS_ROOT}/batches.txt" \
  "${SUB_FILE}"
