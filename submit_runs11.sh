#!/bin/bash
#
# submit_runs11.sh - Submit runs_11 high-statistics multiple scattering scan
#
# Focus: σ_res = 10 µm (instruction default) with fine MS scan
# Total: 6,600 events across 66 batches
#

set -e

cd /data/bfys/gscriven/Velo_toy

echo "=== RUNS_11: High Statistics Multiple Scattering Scan ==="
echo ""
echo "Parameters:"
echo "  σ_res = 10 µm (FIXED - instruction default)"
echo "  σ_scatt = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0] mrad"
echo "  Scale = [3, 4, 5]"
echo "  Density = [sparse, dense]"
echo "  Repeats = 100 per config"
echo ""

# Generate parameters if not already done
if [ ! -f "runs_11/params.csv" ]; then
    echo "Generating parameter batches..."
    python3 gen_params_runs11.py
fi

# Count batches
N_BATCHES=$(wc -l < runs_11/batches.txt)
echo "Total batches: $N_BATCHES"
echo "Events per batch: 100"
echo "Total events: $((N_BATCHES * 100))"
echo ""

# Create log directory
mkdir -p runs_11/logs logs

# Submit using the standard workflow
echo "Submitting to HTCondor..."
condor_submit velo_full_workflow.sub \
    batches_file=/data/bfys/gscriven/Velo_toy/runs_11/batches.txt \
    runsroot=/data/bfys/gscriven/Velo_toy/runs_11

echo ""
echo "✓ Jobs submitted!"
echo ""
echo "Monitor with:"
echo "  condor_q"
echo "  watch -n 30 'condor_q | tail -5'"
echo ""
echo "After completion, aggregate with:"
echo "  python3 re_aggregate.py runs_11"
