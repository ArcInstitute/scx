#!/usr/bin/env bash
# =============================================================================
# SCX cell-eval / arc-bench Parity Validation — SLURM Submission
# =============================================================================
#
# Runs the parity validation tests that compare SCX-accelerated metrics
# against cell-eval and arc-bench reference implementations on synthetic data.
#
# Usage:
#     # Submit with defaults (16 CPUs, 32 GB, cpu_preemptible)
#     bash benchmarks/comprehensive/scripts/slurm_eval_parity.sh
#
#     # Custom resources
#     bash benchmarks/comprehensive/scripts/slurm_eval_parity.sh \
#         --partition cpu_high_mem --mem 64G --time 02:00:00
#
#     # Include slow performance tests
#     bash benchmarks/comprehensive/scripts/slurm_eval_parity.sh --slow
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"

# Detect conda
CONDA_BASE=""
if [[ -d "$HOME/miniforge3" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi

mkdir -p "$LOGS_DIR"

# ---------------------------------------------------------------------------
# Default SLURM parameters
# ---------------------------------------------------------------------------
PARTITION="cpu_preemptible"
CPUS=16
MEM="32G"
TIME="01:00:00"
JOB_NAME="scx-eval-parity"
CONDA_ENV="scx-bench-eval"
INCLUDE_SLOW=false

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
        --slow)         INCLUDE_SLOW=true; TIME="02:00:00"; MEM="64G"; shift ;;
        -h|--help)      head -20 "$0" | tail -17; exit 0 ;;
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

# Resolve conda prefix
if [[ -z "$CONDA_BASE" ]]; then
    echo "ERROR: Cannot find conda installation. Set CONDA_BASE or install miniforge3."
    exit 1
fi
CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
if [[ ! -d "$CONDA_PREFIX" ]]; then
    echo "ERROR: Conda environment '${CONDA_ENV}' not found at ${CONDA_PREFIX}"
    echo "Create it with: bash benchmarks/comprehensive/scripts/install_dependencies.sh --eval"
    exit 1
fi

# Build pytest flags
PYTEST_FLAGS="-v"
if $INCLUDE_SLOW; then
    PYTEST_FLAGS="-v -m ''"  # Include @pytest.mark.slow tests
else
    PYTEST_FLAGS="-v -m 'not slow'"
fi

# ---------------------------------------------------------------------------
# Build the job script
# ---------------------------------------------------------------------------
JOB_SCRIPT="${LOGS_DIR}/run_eval_parity.sh"
cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"

# Activate conda environment (unset VIRTUAL_ENV to avoid maturin conflict)
unset VIRTUAL_ENV
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

cd "\${REPO_ROOT}"

echo "=============================================="
echo "SCX cell-eval / arc-bench Parity Validation"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Partition: \${SLURM_JOB_PARTITION:-local}"
echo "CPUs:      \${SLURM_CPUS_PER_TASK:-\$(nproc)}"
echo "Memory:    ${MEM}"
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

# Step 2: Verify key dependencies
echo "--- Verifying dependencies ---"
python -c "
import cell_eval; print(f'cell-eval: OK')
import arc_bench; print(f'arc-bench: OK')
import pyscx; print(f'pyscx: OK')
import polars; print(f'polars: {polars.__version__}')
import pdex; print(f'pdex: OK')
" || { echo "ERROR: Missing dependencies. Did you create the scx-bench-eval env?"; exit 1; }
echo ""

# Step 3: Run parity tests
echo "--- Running parity validation tests ---"
echo ""
pytest pyscx/tests/test_cell_eval_parity.py ${PYTEST_FLAGS} \\
    --tb=short \\
    --no-header \\
    -x \\
    2>&1 | tee "\${REPO_ROOT}/benchmarks/comprehensive/logs/eval_parity_results.txt"
EXIT_CODE=\${PIPESTATUS[0]}

echo ""
echo "=============================================="
echo "Parity Validation Complete"
echo "=============================================="
echo "Exit code: \${EXIT_CODE}"
echo "Completed: \$(date -Is)"

exit \${EXIT_CODE}
EOFJOB
chmod +x "${JOB_SCRIPT}"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "=============================================="
echo "SCX Parity Validation — SLURM Submission"
echo "=============================================="
echo "  Partition:    ${PARTITION}"
echo "  CPUs:         ${CPUS}"
echo "  Memory:       ${MEM}"
echo "  Time:         ${TIME}"
echo "  Conda env:    ${CONDA_ENV} (${CONDA_PREFIX})"
echo "  Include slow: ${INCLUDE_SLOW}"
echo "  Job script:   ${JOB_SCRIPT}"
echo ""

JOB_ID=$(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted SLURM job: ${JOB_ID}"
echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Err: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.err"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
