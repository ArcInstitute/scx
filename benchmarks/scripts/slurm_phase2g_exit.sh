#!/bin/bash
# Phase 2G — Sprint 2 exit benchmark (D1-D4 + D5-D7, parallel submission).
#
# Submits one SLURM job per (benchmark, dataset) pair for all datasets,
# plus a final archival job that copies results to post_sprint2/.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_phase2g_exit.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"
RESULTS_DIR="${SCX_DIR}/benchmarks/comprehensive/results/raw/post_sprint2"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 2G: Sprint 2 Exit Benchmark — All Datasets ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

BENCHMARKS=(compression write read_full read_selective parallel_scaling memory)

# D1-D4 (small/medium) — cpu partition, 80 GB
SMALL_DATASETS=(pbmc3k pbmc10k smartseq2 tabula_sapiens_100k)
# D5-D7 (large) — cpu_high_mem partition, 500 GB
LARGE_DATASETS=(census_500k census_1m census_5m)

JOB_IDS=()

echo "--- Small/Medium Datasets (D1-D4) ---"
for bench in "${BENCHMARKS[@]}"; do
    for ds in "${SMALL_DATASETS[@]}"; do
        JOB_ID=$(sbatch --parsable \
            --job-name="2g_${bench:0:4}_${ds}" \
            --partition=cpu \
            --qos=normal \
            --cpus-per-task=32 \
            --mem=80G \
            --time=2:00:00 \
            --output="benchmarks/logs/phase2g_${bench}_${ds}_%j.log" \
            --error="benchmarks/logs/phase2g_${bench}_${ds}_%j.err" \
            --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/comprehensive/scripts/run_all.py --benchmarks ${bench} --datasets ${ds}")
        JOB_IDS+=("${JOB_ID}")
        echo "  Submitted: ${bench} × ${ds} → job ${JOB_ID}"
    done
done

echo ""
echo "--- Large Datasets (D5-D7) ---"
for bench in "${BENCHMARKS[@]}"; do
    for ds in "${LARGE_DATASETS[@]}"; do
        JOB_ID=$(sbatch --parsable \
            --job-name="2g_${bench:0:4}_${ds}" \
            --partition=cpu_high_mem \
            --qos=normal \
            --cpus-per-task=32 \
            --mem=500G \
            --time=4:00:00 \
            --output="benchmarks/logs/phase2g_${bench}_${ds}_%j.log" \
            --error="benchmarks/logs/phase2g_${bench}_${ds}_%j.err" \
            --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/comprehensive/scripts/run_all.py --benchmarks ${bench} --datasets ${ds}")
        JOB_IDS+=("${JOB_ID}")
        echo "  Submitted: ${bench} × ${ds} → job ${JOB_ID}"
    done
done

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs total."

# Archival job: runs after all benchmark jobs complete
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
ARCHIVE_ID=$(sbatch --parsable \
    --dependency=afterok:"${DEPS}" \
    --job-name=2g_archive \
    --partition=cpu \
    --qos=normal \
    --mem=4G \
    --time=00:10:00 \
    --output="benchmarks/logs/phase2g_archive_%j.log" \
    --error="benchmarks/logs/phase2g_archive_%j.err" \
    --wrap="mkdir -p ${RESULTS_DIR} && cp ${SCX_DIR}/benchmarks/comprehensive/results/raw/*.json ${RESULTS_DIR}/ && echo 'Archived to ${RESULTS_DIR}/'")

echo "Submitted archival job: ${ARCHIVE_ID} (depends on all ${#JOB_IDS[@]} benchmark jobs)"
echo ""
echo "Monitor with: squeue -u \$USER --name=2g_"
echo "Results will be archived to: ${RESULTS_DIR}/"
