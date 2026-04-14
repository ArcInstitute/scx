#!/usr/bin/env bash
# =============================================================================
# SCX Correctness Validation Suite — SLURM Submission
# =============================================================================
#
# Runs the three validation scripts (§3.14.1–3.14.3) on SLURM.
# Default: fast gate (pbmc3k only). Use --scale for tabula_sapiens_100k.
#
# Usage:
#     # Fast gate (pbmc3k only, ~10 min)
#     bash benchmarks/comprehensive/scripts/slurm_validation_suite.sh
#
#     # Scale gate (+ tabula_sapiens_100k, ~1 hr)
#     bash benchmarks/comprehensive/scripts/slurm_validation_suite.sh --scale
#
#     # Custom resources
#     bash benchmarks/comprehensive/scripts/slurm_validation_suite.sh \
#         --partition cpu_high_mem --mem 200G --time 04:00:00
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
RESULTS_DIR="${REPO_ROOT}/benchmarks/comprehensive/results/raw"

# Detect conda
CONDA_BASE=""
if [[ -d "$HOME/miniforge3" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Default SLURM parameters
# ---------------------------------------------------------------------------
PARTITION="cpu_preemptible"
CPUS=16
MEM="80G"
TIME="02:00:00"
JOB_NAME="scx-validation-suite"
CONDA_ENV=""  # Auto-detect

# Validation-specific flags
SCALE=false
DATASETS="pbmc3k"

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)    PARTITION="$2"; shift 2 ;;
        --cpus)         CPUS="$2"; shift 2 ;;
        --mem)          MEM="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --job-name)     JOB_NAME="$2"; shift 2 ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
        --scale)        SCALE=true; DATASETS="pbmc3k tabula_sapiens_100k"; TIME="04:00:00"; shift ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Build SLURM flags
# ---------------------------------------------------------------------------
SBATCH_FLAGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
    --output="${LOGS_DIR}/%x_%j.out"
    --error="${LOGS_DIR}/%x_%j.err"
)

# Auto-detect conda env
if [[ -z "$CONDA_ENV" ]]; then
    CONDA_ENV="scx-bench"
fi

# Resolve conda prefix
if [[ -z "$CONDA_BASE" ]]; then
    echo "ERROR: Cannot find conda installation. Set CONDA_BASE or install miniforge3."
    exit 1
fi
CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
if [[ ! -d "$CONDA_PREFIX" ]]; then
    echo "ERROR: Conda environment '${CONDA_ENV}' not found at ${CONDA_PREFIX}"
    echo "Create it with: bash benchmarks/comprehensive/scripts/install_dependencies.sh"
    exit 1
fi

# ---------------------------------------------------------------------------
# Build the job script
# ---------------------------------------------------------------------------
JOB_SCRIPT="${LOGS_DIR}/run_validation_suite.sh"
cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
RESULTS_DIR="${RESULTS_DIR}"

# Activate conda environment (unset VIRTUAL_ENV to avoid maturin conflict)
unset VIRTUAL_ENV
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

# Ensure SCX_DATA_DIR is set (works around bench_env.py Path("") bug)
source "\${REPO_ROOT}/.env" 2>/dev/null || true
export SCX_DATA_DIR="\${SCX_DATA_DIR:-\${SCX_WORK_DIR}/benchmarks/datasets}"

cd "\${REPO_ROOT}"

echo "=============================================="
echo "SCX Correctness Validation Suite"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Partition: \${SLURM_JOB_PARTITION:-local}"
echo "CPUs:      \${SLURM_CPUS_PER_TASK:-\$(nproc)}"
echo "Memory:    ${MEM}"
echo "Datasets:  ${DATASETS}"
echo "Conda env: ${CONDA_ENV} (\${CONDA_PREFIX})"
echo "Python:    \$(python --version)"
echo "Started:   \$(date -Is)"
echo ""

# Step 1: Build pyscx in release mode
echo "--- Building pyscx (release mode) ---"
cd "\${REPO_ROOT}/pyscx"
maturin develop --release 2>&1 | tail -5
cd "\${REPO_ROOT}"
echo ""

# Step 2: Run validation scripts
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_SKIPPED=0

for DATASET in ${DATASETS}; do
    echo ""
    echo "=============================================="
    echo "  Dataset: \${DATASET}"
    echo "=============================================="

    echo ""
    echo "--- §3.14.1: Scanpy Equivalence ---"
    python benchmarks/comprehensive/scripts/validate_scanpy_equivalence.py \\
        --dataset "\${DATASET}" \\
        --output "\${RESULTS_DIR}/correctness__scanpy_equiv__\${DATASET}.json" \\
        || echo "WARNING: scanpy equivalence validation failed for \${DATASET}"

    echo ""
    echo "--- §3.14.2: Backed-Mode Equivalence ---"
    python benchmarks/comprehensive/scripts/validate_backed_equivalence.py \\
        --dataset "\${DATASET}" \\
        --output "\${RESULTS_DIR}/correctness__backed_equiv__\${DATASET}.json" \\
        || echo "WARNING: backed equivalence validation failed for \${DATASET}"

    echo ""
    echo "--- §3.14.3: Preprocessing Path Cross-Validation ---"
    python benchmarks/comprehensive/scripts/validate_preprocessing_paths.py \\
        --dataset "\${DATASET}" \\
        --output "\${RESULTS_DIR}/correctness__preproc_paths__\${DATASET}.json" \\
        || echo "WARNING: preprocessing path validation failed for \${DATASET}"
done

echo ""
echo "=============================================="
echo "Validation Suite Complete"
echo "=============================================="
echo "Results directory: \${RESULTS_DIR}"
echo "Completed: \$(date -Is)"

# Print summary from JSON files
echo ""
echo "--- Results Summary ---"
for f in \${RESULTS_DIR}/correctness__*; do
    if [[ -f "\$f" ]]; then
        BASENAME=\$(basename "\$f")
        PASSED=\$(python -c "import json; d=json.load(open('\$f')); print(d.get('n_passed', 0))" 2>/dev/null || echo "?")
        FAILED=\$(python -c "import json; d=json.load(open('\$f')); print(d.get('n_failed', 0))" 2>/dev/null || echo "?")
        SKIPPED=\$(python -c "import json; d=json.load(open('\$f')); print(d.get('n_skipped', 0))" 2>/dev/null || echo "?")
        echo "  \${BASENAME}: passed=\${PASSED} failed=\${FAILED} skipped=\${SKIPPED}"
    fi
done
echo ""
EOFJOB
chmod +x "${JOB_SCRIPT}"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "=============================================="
echo "SCX Correctness Validation — SLURM Submission"
echo "=============================================="
echo "  Partition:  ${PARTITION}"
echo "  CPUs:       ${CPUS}"
echo "  Memory:     ${MEM}"
echo "  Time:       ${TIME}"
echo "  Datasets:   ${DATASETS}"
echo "  Conda env:  ${CONDA_ENV} (${CONDA_PREFIX})"
echo "  Scale mode: ${SCALE}"
echo "  Job script: ${JOB_SCRIPT}"
echo ""

JOB_ID=$(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted SLURM job: ${JOB_ID}"
echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Err: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.err"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
