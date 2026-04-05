#!/bin/bash
# Phase 3 — Core benchmarks on small/medium datasets (D1-D4), parallel submission.
#
# Submits one SLURM job per (benchmark, dataset) pair = 24 parallel jobs.
# Each job gets 32 CPUs, 80 GB, and a 2-hour time limit.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_phase3_parallel_small.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3: Core Benchmarks — Small/Medium Datasets (D1-D4, parallel) ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

BENCHMARKS=(compression write read_full read_selective parallel_scaling memory)
DATASETS=(pbmc3k pbmc10k smartseq2 tabula_sapiens_100k)

JOB_IDS=()
for bench in "${BENCHMARKS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        JOB_ID=$(sbatch --parsable \
            --job-name="p3_${bench:0:4}_${ds}" \
            --partition=cpu \
            --qos=normal \
            --cpus-per-task=32 \
            --mem=80G \
            --time=2:00:00 \
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
