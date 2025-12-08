#!/bin/bash
#
# submit_runs10.sh - Submit runs_10 with instruction-specified parameters
#
# Usage:
#   ./submit_runs10.sh          # Full factorial (43200 jobs) - WARNING: large!
#   ./submit_runs10.sh reduced  # Reduced scan (1000 jobs) - recommended first
#   ./submit_runs10.sh best     # Best config validation (400 jobs)
#

set -e

cd /data/bfys/gscriven/Velo_toy

MODE="${1:-reduced}"  # Default to reduced scan

echo "=== RUNS_10: Instruction-Specified Parameter Scan ==="
echo "Mode: $MODE"
echo ""

# Generate parameters
case $MODE in
    full)
        echo "WARNING: Full factorial = 43,200 jobs!"
        echo "Consider using 'reduced' mode first."
        read -p "Continue? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
        python gen_params_runs10.py full > runs_10/batches.txt
        ;;
    reduced)
        echo "Reduced scan: 1000 jobs (one-at-a-time parameter variation)"
        python gen_params_runs10.py reduced > runs_10/batches.txt
        ;;
    best)
        echo "Best config validation: 400 jobs (high statistics on best configs)"
        python gen_params_runs10.py best > runs_10/batches.txt
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Use: full, reduced, or best"
        exit 1
        ;;
esac

# Count jobs (skip comment lines)
N_JOBS=$(grep -c "^{" runs_10/batches.txt || echo 0)
echo ""
echo "Generated $N_JOBS jobs in runs_10/batches.txt"
echo ""

# Create output directories
mkdir -p runs_10/events
mkdir -p runs_10/logs

# Submit to condor
echo "Submitting to HTCondor..."
condor_submit velo_full_workflow.sub \
    batches=runs_10/batches.txt \
    output_dir=runs_10/events \
    log_dir=runs_10/logs

echo ""
echo "=== Submission Complete ==="
echo "Monitor with: condor_q"
echo "Check logs in: runs_10/logs/"
echo ""
echo "After completion, run analysis:"
echo "  python velo_workflow.py aggregate --events-dir runs_10/events --out runs_10/metrics.csv"
