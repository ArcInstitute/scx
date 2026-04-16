#!/usr/bin/env bash
# =============================================================================
# SCX cell-eval / arc-bench Parity Performance Benchmark — SLURM Submission
# =============================================================================
#
# Runs the `cell_eval_parity_perf` benchmark from the comprehensive suite.
# Measures SCX-accelerated perturbation metrics vs cell-eval / arc-bench
# reference implementations at scale (100K / 500K / 1M / 5M cells).
#
# Usage:
#     # Defaults: 32 CPUs, 128 GB, cpu_preemptible, pert_synth_100k
#     bash benchmarks/comprehensive/scripts/slurm_cell_eval_parity_perf.sh
#
#     # Larger scale on high-mem partition
#     bash benchmarks/comprehensive/scripts/slurm_cell_eval_parity_perf.sh \
#         --datasets pert_synth_1m --partition cpu_high_mem --mem 500G --time 12:00:00
#
#     # Multiple dataset sizes in one job (submitted serially)
#     bash benchmarks/comprehensive/scripts/slurm_cell_eval_parity_perf.sh \
#         --datasets "pert_synth_100k pert_synth_500k"
#
# Notes:
#   - For truly concurrent execution across dataset sizes use run_parallel.py
#     with `--benchmarks cell_eval_parity_perf` (each size submitted as an
#     independent job).
#   - The conda env must have cell-eval + arc-bench + polars + pdex installed.
#     Both `scx-bench` (updated in this plan) and `scx-bench-eval` work.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
mkdir -p "${LOGS_DIR}"

# Detect conda
CONDA_BASE=""
if [[ -d "$HOME/miniforge3" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
PARTITION="cpu_preemptible"
CPUS=32
MEM="128G"
TIME="04:00:00"
JOB_NAME="scx-ce-parity-perf"
CONDA_ENV="scx-bench"
VENV_PATH=""
DATASETS="pert_synth_100k"

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
        --venv)         VENV_PATH="$2"; shift 2 ;;
        --datasets)     DATASETS="$2"; shift 2 ;;
        -h|--help)      head -32 "$0" | tail -29; exit 0 ;;
        *)              echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -n "$VENV_PATH" ]]; then
    if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
        echo "ERROR: --venv '${VENV_PATH}' does not contain bin/python"
        exit 1
    fi
    ENV_SOURCE_LABEL="venv=${VENV_PATH}"
else
    if [[ -z "$CONDA_BASE" ]]; then
        echo "ERROR: Cannot find conda. Set CONDA_BASE, install miniforge3, or pass --venv."
        exit 1
    fi
    CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
    if [[ ! -d "$CONDA_PREFIX" ]]; then
        echo "ERROR: Conda env '${CONDA_ENV}' not found at ${CONDA_PREFIX}"
        echo "Create with: bash benchmarks/comprehensive/scripts/install_dependencies.sh"
        exit 1
    fi
    ENV_SOURCE_LABEL="conda=${CONDA_ENV}"
fi

SBATCH_FLAGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
    --output="${LOGS_DIR}/%x_%j.out"
    --error="${LOGS_DIR}/%x_%j.err"
)

# ---------------------------------------------------------------------------
# Build job script
# ---------------------------------------------------------------------------
JOB_SCRIPT="${LOGS_DIR}/run_cell_eval_parity_perf.sh"
cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
VENV_PATH="${VENV_PATH}"
CONDA_PREFIX_LOCAL="${CONDA_PREFIX:-}"

cd "\${REPO_ROOT}"

if [[ -n "\${VENV_PATH}" ]]; then
    unset CONDA_PREFIX
    export VIRTUAL_ENV="\${VENV_PATH}"
    export PATH="\${VENV_PATH}/bin:\$PATH"
else
    unset VIRTUAL_ENV
    export CONDA_PREFIX="\${CONDA_PREFIX_LOCAL}"
    export PATH="\${CONDA_PREFIX_LOCAL}/bin:\$PATH"
    export LD_LIBRARY_PATH="\${CONDA_PREFIX_LOCAL}/lib:\${LD_LIBRARY_PATH:-}"
fi

echo "=============================================="
echo "cell_eval_parity_perf — comprehensive bench"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Partition: \${SLURM_JOB_PARTITION:-local}"
echo "CPUs:      \${SLURM_CPUS_PER_TASK:-\$(nproc)}"
echo "Memory:    ${MEM}"
echo "Env source: ${ENV_SOURCE_LABEL}"
echo "Datasets:   ${DATASETS}"
echo "Started:   \$(date -Is)"
echo ""

# Build pyscx in release mode
echo "--- Building pyscx ---"
cd "\${REPO_ROOT}/pyscx"
maturin develop --release 2>&1 | tail -5
cd "\${REPO_ROOT}"
echo ""

# Dependency check
python -c "
import cell_eval; print('cell-eval: OK')
import arc_bench; print('arc-bench: OK')
import pyscx; print('pyscx: OK')
import polars; print(f'polars: {polars.__version__}')
import pdex; print('pdex: OK')
" || { echo "ERROR: Missing deps in env '${CONDA_ENV}'"; exit 1; }
echo ""

# Run the benchmark via the standard orchestrator
python benchmarks/comprehensive/scripts/run_all.py \\
    --benchmarks cell_eval_parity_perf \\
    --datasets ${DATASETS} \\
    --formats scx_auto

echo ""
echo "Completed: \$(date -Is)"
EOFJOB
chmod +x "${JOB_SCRIPT}"

# Submit
echo "=============================================="
echo "cell_eval_parity_perf — SLURM Submission"
echo "=============================================="
echo "  Partition: ${PARTITION}"
echo "  CPUs:      ${CPUS}"
echo "  Memory:    ${MEM}"
echo "  Time:      ${TIME}"
echo "  Conda env: ${CONDA_ENV}"
echo "  Datasets:  ${DATASETS}"
echo "  Script:    ${JOB_SCRIPT}"
echo ""

JOB_ID=$(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted SLURM job: ${JOB_ID}"
echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo ""
echo "Monitor with: squeue -u \$USER"
