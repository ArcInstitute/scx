#!/bin/bash
# Phase 9: Integration Pipeline Benchmark — All Phase II Optimizations
#
# Runs the full pipeline benchmark (SCX OOC, SCX Preprocess, Scanpy)
# with all Phase 6-8 optimizations: sparse covariance PCA, pre-ranked DE,
# and Rust-native Leiden.
#
# Usage: bash benchmarks/scripts/slurm_phase9_pipeline_bench.sh

set -euo pipefail

SCX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SCX_DIR}/.venv"
PYTHON="${VENV}/bin/python"
MATURIN="${VENV}/bin/maturin"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs benchmarks/results

echo "=== Phase 9: Integration Pipeline Benchmark ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# ── Step 0: Build pyscx in release mode ──────────────────────────────────────

BUILD_JOB=$(sbatch --parsable \
    --job-name="p9_build" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --output="benchmarks/logs/phase9_build_%j.log" \
    --error="benchmarks/logs/phase9_build_%j.err" \
    --wrap="cd ${SCX_DIR}/pyscx && ${MATURIN} develop --release && echo 'Build complete'")
echo "Build job: ${BUILD_JOB}"

# ── Step 1: Pipeline benchmark per dataset ───────────────────────────────────
# RAYON_NUM_THREADS and MALLOC_ARENA_MAX set to prevent OOM from GLIBC arenas
# on shared SLURM nodes that expose all CPUs (e.g. 192) to the process.

JOB_IDS=()

# tabula_sapiens_100k — 3 pipeline variants × 3 runs each
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p9_pipeline_tabula" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=80G \
    --time=08:00:00 \
    --output="benchmarks/logs/phase9_pipeline_tabula_%j.log" \
    --error="benchmarks/logs/phase9_pipeline_tabula_%j.err" \
    --export=ALL,RAYON_NUM_THREADS=16,MALLOC_ARENA_MAX=4 \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accel_pipeline.py --mode all --datasets tabula_sapiens_100k --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Pipeline x tabula_sapiens_100k -> job ${JOB_ID}"

# census_1m — 3 pipeline variants × 3 runs each (long-running)
JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p9_pipeline_census" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=16 \
    --mem=256G \
    --time=24:00:00 \
    --output="benchmarks/logs/phase9_pipeline_census_%j.log" \
    --error="benchmarks/logs/phase9_pipeline_census_%j.err" \
    --export=ALL,RAYON_NUM_THREADS=16,MALLOC_ARENA_MAX=4 \
    --wrap="cd ${SCX_DIR} && ${PYTHON} benchmarks/scripts/benchmark_accel_pipeline.py --mode all --datasets census_1m --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  Pipeline x census_1m -> job ${JOB_ID}"

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "Submitted ${#JOB_IDS[@]} benchmark jobs + 1 build job."
echo "Build job:   ${BUILD_JOB}"
echo ""
echo "Monitor with:  squeue -u \$USER --name=p9_"
echo "Results in:    benchmarks/results/accel_pipeline_benchmark.json"
echo "Logs in:       benchmarks/logs/phase9_*.log"
