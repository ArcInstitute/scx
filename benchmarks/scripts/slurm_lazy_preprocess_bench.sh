#!/bin/bash
# Phase 5c: Lazy Preprocessing & Column-Projected Aggregation Benchmarks
#
# Submits parallel SLURM jobs for each (benchmark, dataset) pair.
# A build job runs first as a dependency for all benchmark jobs.
# A gate evaluation job runs after all benchmarks complete.
#
# Benchmarks (§3.13.1–3.13.6):
#   transform_overhead      — Lazy transform slice overhead vs raw
#   col_projected_full      — Column-projected aggregation (6 scenarios)
#   agg_through_transforms  — Aggregation through lazy transforms
#   fused_opt               — Fused NormalizeTotal+Log1p vs sequential
#   memory_efficiency       — RSS comparison (lazy vs scanpy, 4 scenarios)
#   e2e_pipeline            — Full 11-stage OOC pipeline vs scanpy
#
# Usage: bash benchmarks/scripts/slurm_lazy_preprocess_bench.sh

set -euo pipefail

SCX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${SCX_DIR}/.venv"
PYTHON="${VENV}/bin/python"
MATURIN="${VENV}/bin/maturin"
SCRIPT="benchmarks/scripts/benchmark_lazy_preprocess.py"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs benchmarks/results

echo "=== Phase 5c: Lazy Preprocessing Benchmarks — Parallel Submission ==="
echo "Date: $(date)"
echo "Git commit: $(git rev-parse --short HEAD)"
echo ""

# ── Step 0: Build pyscx in release mode ──────────────────────────────────────

BUILD_JOB=$(sbatch --parsable \
    --job-name="p5c_build" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --output="benchmarks/logs/phase5c_build_%j.log" \
    --error="benchmarks/logs/phase5c_build_%j.err" \
    --wrap="cd ${SCX_DIR}/pyscx && ${MATURIN} develop --release && echo 'Build complete'")
echo "Build job: ${BUILD_JOB}"

# ── Step 1: Validation (pbmc3k, quick) ───────────────────────────────────────

VAL_JOB=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5c_validate" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=8 \
    --mem=16G \
    --time=01:00:00 \
    --output="benchmarks/logs/phase5c_validate_%j.log" \
    --error="benchmarks/logs/phase5c_validate_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode validate --datasets pbmc3k --skip-build")
echo "Validation job: ${VAL_JOB}"

# ── Step 2: Transform overhead (§3.13.1) ─────────────────────────────────────

JOB_IDS=()

for cfg in "tabula_sapiens_100k:80G:06:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5c_overhead_${ds}" \
        --partition=cpu_preemptible \
        --qos=normal \
        --cpus-per-task=32 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5c_overhead_${ds}_%j.log" \
        --error="benchmarks/logs/phase5c_overhead_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode transform_overhead --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  transform_overhead x ${ds} -> job ${JOB_ID}"
done

# ── Step 3: Column-projected aggregation (§3.13.3) ───────────────────────────

for cfg in "tabula_sapiens_100k:80G:06:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5c_colproj_${ds}" \
        --partition=cpu_preemptible \
        --qos=normal \
        --cpus-per-task=32 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5c_colproj_${ds}_%j.log" \
        --error="benchmarks/logs/phase5c_colproj_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode col_projected_full --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  col_projected_full x ${ds} -> job ${JOB_ID}"
done

# ── Step 4: Aggregation through transforms (§3.13.4) ─────────────────────────

for cfg in "tabula_sapiens_100k:80G:06:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5c_aggtrans_${ds}" \
        --partition=cpu_preemptible \
        --qos=normal \
        --cpus-per-task=32 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5c_aggtrans_${ds}_%j.log" \
        --error="benchmarks/logs/phase5c_aggtrans_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode agg_through_transforms --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  agg_through_transforms x ${ds} -> job ${JOB_ID}"
done

# ── Step 5: Fused optimization (§3.13.5) — census_1m only ────────────────────

JOB_ID=$(sbatch --parsable \
    --dependency=afterok:${BUILD_JOB} \
    --job-name="p5c_fused_census_1m" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=32 \
    --mem=256G \
    --time=12:00:00 \
    --output="benchmarks/logs/phase5c_fused_census_1m_%j.log" \
    --error="benchmarks/logs/phase5c_fused_census_1m_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode fused_opt --datasets census_1m --skip-build")
JOB_IDS+=("${JOB_ID}")
echo "  fused_opt x census_1m -> job ${JOB_ID}"

# ── Step 6: Memory efficiency (§3.13.2) ──────────────────────────────────────

for cfg in "census_500k:128G:06:00:00" "census_1m:256G:12:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5c_memeff_${ds}" \
        --partition=cpu_preemptible \
        --qos=normal \
        --cpus-per-task=16 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5c_memeff_${ds}_%j.log" \
        --error="benchmarks/logs/phase5c_memeff_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode memory_efficiency --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  memory_efficiency x ${ds} -> job ${JOB_ID}"
done

# ── Step 7: E2E out-of-core pipeline (§3.13.6) ──────────────────────────────

for cfg in "tabula_sapiens_100k:80G:06:00:00" "census_1m:256G:16:00:00"; do
    IFS=: read -r ds mem timelimit <<< "${cfg}"
    JOB_ID=$(sbatch --parsable \
        --dependency=afterok:${BUILD_JOB} \
        --job-name="p5c_e2e_${ds}" \
        --partition=cpu_preemptible \
        --qos=normal \
        --cpus-per-task=32 \
        --mem="${mem}" \
        --time="${timelimit}" \
        --output="benchmarks/logs/phase5c_e2e_${ds}_%j.log" \
        --error="benchmarks/logs/phase5c_e2e_${ds}_%j.err" \
        --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode e2e_pipeline --datasets ${ds} --skip-build")
    JOB_IDS+=("${JOB_ID}")
    echo "  e2e_pipeline x ${ds} -> job ${JOB_ID}"
done

# ── Step 8: Gate evaluation + report (depends on all benchmark jobs) ─────────

DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
GATE_JOB=$(sbatch --parsable \
    --dependency=afterok:${DEPS} \
    --job-name="p5c_gate" \
    --partition=cpu_preemptible \
    --qos=normal \
    --cpus-per-task=4 \
    --mem=8G \
    --time=00:30:00 \
    --output="benchmarks/logs/phase5c_gate_%j.log" \
    --error="benchmarks/logs/phase5c_gate_%j.err" \
    --wrap="cd ${SCX_DIR} && ${PYTHON} ${SCRIPT} --mode gate --skip-build")
echo ""
echo "Gate job: ${GATE_JOB} (runs after all benchmarks)"

# ── Summary ──────────────────────────────────────────────────────────────────

N_JOBS=${#JOB_IDS[@]}
echo ""
echo "=== Submitted ${N_JOBS} benchmark jobs + 1 build + 1 validation + 1 gate ==="
echo "Build: ${BUILD_JOB}"
echo "Validation: ${VAL_JOB}"
echo "Benchmarks: ${JOB_IDS[*]}"
echo "Gate: ${GATE_JOB}"
echo ""
echo "Monitor: squeue -u \$USER --name='p5c_*'"
echo "Results: benchmarks/results/lazy_preprocess_*.json"
echo "Report: benchmarks/results/lazy_preprocess_benchmark.md"
