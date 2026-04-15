#!/bin/bash
# Parallel write scaling benchmarks (§3.5.2) on D1-D7.
#
# Submits one SLURM job per dataset with appropriate resources.
# Must use --cpus-per-task=32 since parallel_write_scaling tests up to 32 threads.
# Write scaling runs 2 modes (full + write_only) × 6 thread counts × n_runs,
# so time limits are generous.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_parallel_write_scaling.sh

set -euo pipefail

SCX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Parallel Write Scaling Benchmarks (§3.5.2) ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# Per-dataset resource configuration: "dataset:partition:memory:time"
# Small datasets (D1-D4): fast, minimal resources
# Large datasets (D5-D7): need more memory and time
DATASET_CONFIGS=(
    "pbmc3k:cpu_preemptible:80G:2:00:00"
    "pbmc10k:cpu_preemptible:80G:2:00:00"
    "smartseq2:cpu_preemptible:80G:3:00:00"
    "tabula_sapiens_100k:cpu_preemptible:80G:4:00:00"
    "census_500k:cpu_preemptible:160G:8:00:00"
    "census_1m:cpu_preemptible:160G:12:00:00"
    "census_5m:cpu_high_mem:500G:24:00:00"
)

JOB_IDS=()
for cfg in "${DATASET_CONFIGS[@]}"; do
    IFS=: read -r ds partition mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --job-name="pwscale_${ds}" \
        --partition="${partition}" \
        --qos=normal \
        --cpus-per-task=32 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/parallel_write_scaling_${ds}_%j.log" \
        --error="benchmarks/logs/parallel_write_scaling_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/comprehensive/scripts/run_all.py --benchmarks parallel_write_scaling --datasets ${ds}")
    JOB_IDS+=("${JOB_ID}")
    echo "  Submitted: parallel_write_scaling × ${ds} → job ${JOB_ID} (${partition}, ${mem}, ${timelimit})"
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs."
echo "Monitor with: squeue -u \$USER --name=pwscale_"
echo "Results in: benchmarks/comprehensive/results/raw/"
