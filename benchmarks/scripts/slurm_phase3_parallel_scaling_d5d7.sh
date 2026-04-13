#!/bin/bash
# Phase 3 — parallel_scaling benchmarks on D5-D7 only.
#
# Submits one SLURM job per dataset with appropriate resources.
# Must use --cpus-per-task=32 since parallel_scaling tests up to 32 threads.
#
# Usage: bash benchmarks/scripts/slurm_phase3_parallel_scaling_d5d7.sh

set -euo pipefail

SCX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3: parallel_scaling — D5-D7 ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# Per-dataset resource configuration: "dataset:partition:memory:time"
DATASET_CONFIGS=(
    "census_500k:cpu_preemptible:80G:6:00:00"
    "census_1m:cpu_preemptible:160G:10:00:00"
    "census_5m:cpu_high_mem:500G:16:00:00"
)

JOB_IDS=()
for cfg in "${DATASET_CONFIGS[@]}"; do
    IFS=: read -r ds partition mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --job-name="p3_para_${ds}" \
        --partition="${partition}" \
        --qos=normal \
        --cpus-per-task=32 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase3_parallel_scaling_${ds}_%j.log" \
        --error="benchmarks/logs/phase3_parallel_scaling_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/comprehensive/scripts/run_all.py --benchmarks parallel_scaling --datasets ${ds}")
    JOB_IDS+=("${JOB_ID}")
    echo "  Submitted: parallel_scaling x ${ds} -> job ${JOB_ID}"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs."
echo "Monitor with: squeue -u \$USER --name=p3_para_"
echo "Results in: benchmarks/comprehensive/results/raw/"
