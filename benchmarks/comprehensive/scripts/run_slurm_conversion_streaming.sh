#!/usr/bin/env bash
# =============================================================================
# Streaming h5ad → SCX conversion benchmark. Submits one SLURM job per
# dataset; the memory rule pinned in
# ~/.claude/projects/-home-nickyoungblut-dev-rust-scx/memory/feedback_lambda_hpc_jobs.md
# forbids inline benchmark runs.
#
# Usage:
#     bash benchmarks/comprehensive/scripts/run_slurm_conversion_streaming.sh
#     bash benchmarks/comprehensive/scripts/run_slurm_conversion_streaming.sh \
#         --datasets "census_1m census_5m census_10m"
#     bash benchmarks/comprehensive/scripts/run_slurm_conversion_streaming.sh \
#         --datasets pbmc3k --runs 3 --partition standard --mem 32G
#
# Resource defaults are sized so the *materialize* path can fit the
# whole CSR in memory at census-1m scale (~12 GB peak observed in
# docs/performance.md). For census_5m / census_10m, push --mem
# higher or accept that the materialize path will OOM and the
# streaming row reports cleanly.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
mkdir -p "$LOGS_DIR"

# Conda detection (matches run_slurm.sh).
CONDA_BASE=""
if [[ -d "$HOME/miniforge3" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null)"
fi

# Defaults — overridable via flags.
PARTITION="standard"
CPUS=16
MEM="64G"
TIME="04:00:00"
DATASETS="census_1m census_5m census_10m"
N_RUNS=3
CONDA_ENV="scx-bench"
JOB_NAME_PREFIX="scx-conv-stream"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition) PARTITION="$2"; shift 2 ;;
        --cpus)      CPUS="$2"; shift 2 ;;
        --mem)       MEM="$2"; shift 2 ;;
        --time)      TIME="$2"; shift 2 ;;
        --datasets)  DATASETS="$2"; shift 2 ;;
        --runs)      N_RUNS="$2"; shift 2 ;;
        --conda-env) CONDA_ENV="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

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

# One sbatch per dataset — matches the parallel-submission convention used
# by gate_candidate.py upstream so different datasets don't share a
# single long-lived job that's stuck behind one slow dataset.
for DATASET in $DATASETS; do
    JOB_NAME="${JOB_NAME_PREFIX}-${DATASET}"
    JOB_SCRIPT="${LOGS_DIR}/${JOB_NAME}.sh"

    cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

cd "\${REPO_ROOT}"

echo "=============================================="
echo "Streaming conversion benchmark — SLURM Job"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Dataset:   ${DATASET}"
echo "Runs:      ${N_RUNS}"
echo "Memory:    ${MEM}"
echo "Time:      ${TIME}"
echo "Conda env: ${CONDA_ENV} (\${CONDA_PREFIX})"
echo "Python:    \$(python --version)"
echo "Started:   \$(date -Is)"
echo ""

echo "--- Building pyscx (release mode) ---"
cd "\${REPO_ROOT}/pyscx"
maturin develop --release 2>&1 | tail -5
cd "\${REPO_ROOT}"
echo ""

echo "--- Running streaming conversion benchmark ---"
python benchmarks/comprehensive/scripts/run_all.py \\
    --benchmarks conversion_streaming \\
    --datasets ${DATASET} \\
    --runs ${N_RUNS}

echo ""
echo "=============================================="
echo "Completed: \$(date -Is)"
echo "=============================================="
EOFJOB
    chmod +x "${JOB_SCRIPT}"

    SBATCH_FLAGS=(
        --job-name="${JOB_NAME}"
        --partition="${PARTITION}"
        --cpus-per-task="${CPUS}"
        --mem="${MEM}"
        --time="${TIME}"
        --output="${LOGS_DIR}/%x_%j.out"
        --error="${LOGS_DIR}/%x_%j.err"
    )

    JOB_ID=$(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" | awk '{print $NF}')
    echo "Submitted ${DATASET}: job ${JOB_ID}"
    echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
done

echo ""
echo "Monitor with: squeue -u \$USER"
