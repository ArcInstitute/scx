#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/nickyoungblut/dev/rust/scx"
RESULTS_DIR="/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/results/raw"

# Activate conda environment (unset VIRTUAL_ENV to avoid maturin conflict)
unset VIRTUAL_ENV
export CONDA_PREFIX="/home/nickyoungblut/miniforge3/envs/scx-bench"
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Ensure SCX_DATA_DIR is set (works around bench_env.py Path("") bug)
source "${REPO_ROOT}/.env" 2>/dev/null || true
export SCX_DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"

cd "${REPO_ROOT}"

echo "=============================================="
echo "SCX Correctness Validation Suite"
echo "=============================================="
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "Node:      $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-local}"
echo "CPUs:      ${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory:    500G"
echo "Datasets:  pbmc3k tabula_sapiens_100k"
echo "Conda env: scx-bench (${CONDA_PREFIX})"
echo "Python:    $(python --version)"
echo "Started:   $(date -Is)"
echo ""

# Step 1: Build pyscx in release mode
echo "--- Building pyscx (release mode) ---"
cd "${REPO_ROOT}/pyscx"
maturin develop --release 2>&1 | tail -5
cd "${REPO_ROOT}"
echo ""

# Step 2: Run validation scripts
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

for DATASET in pbmc3k tabula_sapiens_100k; do
    echo ""
    echo "=============================================="
    echo "  Dataset: ${DATASET}"
    echo "=============================================="

    echo ""
    echo "--- §3.14.1: Scanpy Equivalence ---"
    python benchmarks/comprehensive/scripts/validate_scanpy_equivalence.py \
        --dataset "${DATASET}" \
        --output "${RESULTS_DIR}/correctness__scanpy_equiv__${DATASET}.json" \
        || echo "WARNING: scanpy equivalence validation failed for ${DATASET}"

    echo ""
    echo "--- §3.14.2: Backed-Mode Equivalence ---"
    python benchmarks/comprehensive/scripts/validate_backed_equivalence.py \
        --dataset "${DATASET}" \
        --output "${RESULTS_DIR}/correctness__backed_equiv__${DATASET}.json" \
        || echo "WARNING: backed equivalence validation failed for ${DATASET}"

    echo ""
    echo "--- §3.14.3: Preprocessing Path Cross-Validation ---"
    python benchmarks/comprehensive/scripts/validate_preprocessing_paths.py \
        --dataset "${DATASET}" \
        --output "${RESULTS_DIR}/correctness__preproc_paths__${DATASET}.json" \
        || echo "WARNING: preprocessing path validation failed for ${DATASET}"
done

echo ""
echo "=============================================="
echo "Validation Suite Complete"
echo "=============================================="
echo "Results directory: ${RESULTS_DIR}"
echo "Completed: $(date -Is)"

# Print summary from JSON files
echo ""
echo "--- Results Summary ---"
for f in ${RESULTS_DIR}/correctness__*; do
    if [[ -f "$f" ]]; then
        BASENAME=$(basename "$f")
        PASSED=$(python -c "import json; d=json.load(open('$f')); print(d.get('n_passed', 0))" 2>/dev/null || echo "?")
        FAILED=$(python -c "import json; d=json.load(open('$f')); print(d.get('n_failed', 0))" 2>/dev/null || echo "?")
        SKIPPED=$(python -c "import json; d=json.load(open('$f')); print(d.get('n_skipped', 0))" 2>/dev/null || echo "?")
        echo "  ${BASENAME}: passed=${PASSED} failed=${FAILED} skipped=${SKIPPED}"
    fi
done
echo ""
