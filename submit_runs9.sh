#!/bin/bash
# Submit script for runs_9 - Comprehensive parameter scan

RUNS_DIR="/data/bfys/gscriven/Velo_toy/runs_9"
BATCHES_FILE="${RUNS_DIR}/batches.txt"

if [ ! -f "$BATCHES_FILE" ]; then
    echo "Error: batches.txt not found at $BATCHES_FILE"
    echo "Run: python gen_params_runs9.py first"
    exit 1
fi

N_BATCHES=$(wc -l < "$BATCHES_FILE")
echo "Submitting $N_BATCHES jobs for runs_9..."

condor_submit \
    runsroot="${RUNS_DIR}" \
    batches_file="${BATCHES_FILE}" \
    /data/bfys/gscriven/Velo_toy/velo_full_workflow.sub

echo "Jobs submitted. Monitor with: condor_q"
