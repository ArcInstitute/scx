#!/bin/bash
# Phase 0 — Pre-Sprint 1 regression baseline (D5-D7, parallel submission).
#
# Submits one SLURM job per (benchmark, dataset) pair = 18 parallel jobs,
# plus a final archival job that copies results to baseline_pre_sprint1/.
# Memory and partition are scaled per dataset:
#   census_500k:  80G on cpu_preemptible
#   census_1m:   160G on cpu_preemptible
#   census_5m:   500G on cpu_high_mem
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_phase0_baseline_large.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"
BASELINE_DIR="${SCX_DIR}/benchmarks/comprehensive/results/raw/baseline_pre_sprint1"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 0: Pre-Sprint 1 Baseline — Large Datasets (D5-D7) ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

BENCHMARKS=(compression write read_full read_selective parallel_scaling memory)

# Per-dataset resource configuration: "dataset:partition:memory:time"
# census_5m is excluded — requires too many resources for routine baseline runs.
# To include it, uncomment the census_5m line below.
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
            --job-name="p0_${bench:0:4}_${ds}" \
            --partition="${partition}" \
            --qos=normal \
            --cpus-per-task=32 \
            --mem="${mem}" \
            --time="${timelimit}" \
            --output="benchmarks/logs/phase0_${bench}_${ds}_%j.log" \
            --error="benchmarks/logs/phase0_${bench}_${ds}_%j.err" \
            --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/comprehensive/scripts/run_all.py --benchmarks ${bench} --datasets ${ds}")
        JOB_IDS+=("${JOB_ID}")
        echo "  Submitted: ${bench} × ${ds} (${partition}, ${mem}) → job ${JOB_ID}"
    done
done

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs."

# Archival job: runs after all benchmark jobs complete
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
ARCHIVE_ID=$(sbatch --parsable \
    --dependency=afterok:"${DEPS}" \
    --job-name=p0_archive_lg \
    --partition=cpu_preemptible \
    --qos=normal \
    --mem=4G \
    --time=00:10:00 \
    --output="benchmarks/logs/phase0_archive_large_%j.log" \
    --error="benchmarks/logs/phase0_archive_large_%j.err" \
    --wrap="mkdir -p ${BASELINE_DIR} && cp ${SCX_DIR}/benchmarks/comprehensive/results/raw/*.json ${BASELINE_DIR}/ && echo 'Archived to ${BASELINE_DIR}/'")

echo "Submitted archival job: ${ARCHIVE_ID} (depends on all ${#JOB_IDS[@]} benchmark jobs)"
echo ""
echo "Monitor with: squeue -u \$USER --name=p0_"
echo "Results will be archived to: ${BASELINE_DIR}/"
