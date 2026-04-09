#!/bin/bash
# Phase 5: Accelerator Benchmarks (Phase 4b) — Parallel SLURM submission
#
# Submits one SLURM job per (benchmark, dataset) pair. A build job runs first
# as a dependency for all benchmark jobs.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && bash benchmarks/scripts/slurm_phase5_accel_bench.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
VENV="${SCX_DIR}/.venv"
PYTHON="${VENV}/bin/python"
MATURIN="${VENV}/bin/maturin"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs benchmarks/results

echo "=== Phase 5: Accelerator Benchmarks (Phase 4b) — Parallel Submission ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# ── Step 0: Build pyscx in release mode ──────────────────────────────────────

BUILD_JOB=$(sbatch --parsable \
    --job-name="p5_build" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --output="benchmarks/logs/phase5_build_%j.log" \
    --error="benchmarks/logs/phase5_build_%j.err" \
    --wrap="cd ${SCX_DIR}/pyscx && ${MATURIN} develop --release && echo 'Build complete'")
echo "Build job: ${BUILD_JOB}"

# ── Step 1: Validation (pbmc3k, quick) ───────────────────────────────────────

VAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5_validate" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --output="benchmarks/logs/phase5_validate_%j.log" \
    --error="benchmarks/logs/phase5_validate_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode validate --skip-build")
echo "Validation job: ${VAL_JOB}"

# ── Step 2: Individual accelerator benchmarks ────────────────────────────────

JOB_IDS=()

# PCA benchmarks
for cfg in "tabula_sapiens_100k:80G:04:00:00" "census_500k:160G:08:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5_pca_${ds}" \
        --partition=cpu \
        --qos=normal \
        --cpus-per-task=16 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5_pca_${ds}_%j.log" \
        --error="benchmarks/logs/phase5_pca_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode pca --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  PCA x ${ds} -> job ${JOB_ID}"
done

# kNN benchmarks
for cfg in "tabula_sapiens_100k:80G:04:00:00" "census_500k:160G:08:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5_knn_${ds}" \
        --partition=cpu \
        --qos=normal \
        --cpus-per-task=16 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5_knn_${ds}_%j.log" \
        --error="benchmarks/logs/phase5_knn_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode knn --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  kNN x ${ds} -> job ${JOB_ID}"
done

# UMAP benchmarks
for cfg in "tabula_sapiens_100k:80G:04:00:00" "census_500k:160G:08:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5_umap_${ds}" \
        --partition=cpu \
        --qos=normal \
        --cpus-per-task=16 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5_umap_${ds}_%j.log" \
        --error="benchmarks/logs/phase5_umap_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode umap --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  UMAP x ${ds} -> job ${JOB_ID}"
done

# DE in-memory benchmarks
for cfg in "tabula_sapiens_100k:80G:04:00:00" "census_500k:160G:08:00:00" "census_1m:256G:16:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5_de_${ds}" \
        --partition=cpu \
        --qos=normal \
        --cpus-per-task=16 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5_de_${ds}_%j.log" \
        --error="benchmarks/logs/phase5_de_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode de --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  DE x ${ds} -> job ${JOB_ID}"
done

# DE streaming benchmark
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5_de_stream" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=64G \
    --time=08:00:00 \
    --output="benchmarks/logs/phase5_de_streaming_%j.log" \
    --error="benchmarks/logs/phase5_de_streaming_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode de_streaming --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  DE streaming -> job ${JOB_ID}"

# Pseudobulk benchmark
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5_pseudobulk" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=128G \
    --time=08:00:00 \
    --output="benchmarks/logs/phase5_pseudobulk_%j.log" \
    --error="benchmarks/logs/phase5_pseudobulk_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode pseudobulk --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Pseudobulk -> job ${JOB_ID}"

# Stratified benchmark
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5_stratified" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=80G \
    --time=04:00:00 \
    --output="benchmarks/logs/phase5_stratified_%j.log" \
    --error="benchmarks/logs/phase5_stratified_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accelerators.py --mode stratified --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Stratified -> job ${JOB_ID}"

# ── Step 3: Pipeline benchmark ───────────────────────────────────────────────

JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5_pipeline" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=256G \
    --time=16:00:00 \
    --output="benchmarks/logs/phase5_pipeline_%j.log" \
    --error="benchmarks/logs/phase5_pipeline_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accel_pipeline.py --mode all --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Pipeline -> job ${JOB_ID}"

# ── Step 4: Preprocessing benchmark ──────────────────────────────────────────

JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5_preprocess" \
    --partition=cpu \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=160G \
    --time=08:00:00 \
    --output="benchmarks/logs/phase5_preprocess_%j.log" \
    --error="benchmarks/logs/phase5_preprocess_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accel_preprocessing.py --mode all --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Preprocessing -> job ${JOB_ID}"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs + 1 build job + 1 validation job."
echo "Build job:     ${BUILD_JOB} (all benchmarks depend on this)"
echo "Validation:    ${VAL_JOB}"
echo ""
echo "Monitor with:  squeue -u \$USER --name=p5_"
echo "Results in:    benchmarks/results/accel_*.json"
echo "Logs in:       benchmarks/logs/phase5_*.log"
