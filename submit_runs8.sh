#!/bin/bash
#
# Submit runs_8 jobs to Condor
# Corrected epsilon study with optimized SimpleHamiltonianFast
#

set -e

RUNS_DIR="/data/bfys/gscriven/Velo_toy/runs_8"
PARAMS_FILE="${RUNS_DIR}/params.csv"
BATCHES_FILE="${RUNS_DIR}/batches.txt"

if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "Error: $PARAMS_FILE not found. Run gen_params_runs8.py first."
    exit 1
fi

if [[ ! -f "$BATCHES_FILE" ]]; then
    echo "Error: $BATCHES_FILE not found. Run gen_params_runs8.py first."
    exit 1
fi

N_BATCHES=$(wc -l < "$BATCHES_FILE")
echo "Submitting $N_BATCHES jobs for runs_8..."

# Submit to Condor
condor_submit velo_full_workflow.sub \
    runsroot="$RUNS_DIR" \
    batches_file="$BATCHES_FILE"

echo "Jobs submitted. Monitor with: condor_q"
