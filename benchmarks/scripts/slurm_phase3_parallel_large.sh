#!/bin/bash
# Phase 3 — Core benchmarks on large datasets (D5-D7), parallel submission.
#
# Submits one SLURM job per (benchmark, dataset) pair. Memory and partition
# are scaled per dataset:
#   census_500k:  80G on cpu_preemptible
#   census_1m:   160G on cpu_preemptible
#   census_5m:   500G on cpu_high_mem (commented out by default)
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_phase3_parallel_large.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3: Core Benchmarks — Large Datasets (D5-D7, parallel) ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

BENCHMARKS=(compression write read_full read_selective parallel_scaling memory)

# Per-dataset resource configuration: "dataset:partition:memory:time"
DATASET_CONFIGS=(
    "census_500k:cpu_preemptible:80G:4:00:00"
    "census_1m:cpu_preemptible:160G:8:00:00"
    # "census_5m:cpu_high_mem:500G:12:00:00"
)

JOB_IDS=()
for bench in "${BENCHMARKS[@]}"; do
    for cfg in "${DATASET_CONFIGS[@]}"; do
        IFS=: read -r ds partition mem timelimit <<< "${cfg}"
        JOB_ID=$(sbatch --parsable \
            --job-name="p3_${bench:0:4}_${ds}" \
            --partition="${partition}" \
            --qos=normal \
            --cpus-per-task=32 \
            --mem="${mem}" \
            --time="${timelimit}" \
            --output="benchmarks/logs/phase3_${bench}_${ds}_%j.log" \
            --error="benchmarks/logs/phase3_${bench}_${ds}_%j.err" \
            --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/comprehensive/scripts/run_all.py --benchmarks ${bench} --datasets ${ds}")
        JOB_IDS+=("${JOB_ID}")
        echo "  Submitted: ${bench} × ${ds} → job ${JOB_ID}"
    done
done

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs."
echo "Monitor with: squeue -u \$USER --name=p3_"
echo "Results in: benchmarks/comprehensive/results/raw/"
