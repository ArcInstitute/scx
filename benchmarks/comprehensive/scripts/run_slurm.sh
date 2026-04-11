#!/usr/bin/env bash
# =============================================================================
# SCX Comprehensive Benchmark Suite — SLURM Submission Template
# =============================================================================
#
# Submits the comprehensive benchmark suite to SLURM with configurable
# resources (memory, time, CPUs, exclusive mode).
#
# Usage:
#     # Submit with defaults (cpu_preemptible partition, 80 GB, 4 hours)
#     bash benchmarks/comprehensive/scripts/run_slurm.sh
#
#     # Submit with custom resources
#     bash benchmarks/comprehensive/scripts/run_slurm.sh \
#         --partition cpu_high_mem --mem 500G --time 08:00:00 --exclusive
#
#     # Submit a smoke test
#     bash benchmarks/comprehensive/scripts/run_slurm.sh --smoke
#
#     # Submit specific benchmarks
#     bash benchmarks/comprehensive/scripts/run_slurm.sh \
#         --benchmarks "compression read_full write"
#
#     # Submit specific datasets
#     bash benchmarks/comprehensive/scripts/run_slurm.sh \
#         --datasets "pbmc3k census_1m"
#
#     # Submit GPU benchmarks
#     bash benchmarks/comprehensive/scripts/run_slurm.sh \
#         --partition preemptible --gpus 1 --benchmarks "gpu_accelerators"
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
# Default SLURM parameters (overridable via CLI flags)
# ---------------------------------------------------------------------------
PARTITION="cpu_preemptible"
CPUS=16
MEM="80G"
TIME="04:00:00"
GPUS=""
EXCLUSIVE=""
JOB_NAME="scx-comprehensive-bench"
CONDA_ENV=""  # Auto-detect: scx-bench for CPU, scx-bench-gpu for GPU

# Benchmark-specific flags (passed through to run_all.py)
BENCH_FLAGS=""

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --partition)    PARTITION="$2"; shift 2 ;;
        --cpus)         CPUS="$2"; shift 2 ;;
        --mem)          MEM="$2"; shift 2 ;;
        --time)         TIME="$2"; shift 2 ;;
        --gpus)         GPUS="$2"; shift 2 ;;
        --exclusive)    EXCLUSIVE="--exclusive"; shift ;;
        --job-name)     JOB_NAME="$2"; shift 2 ;;
        --smoke)        BENCH_FLAGS="${BENCH_FLAGS} --smoke"; shift ;;
        --cold-cache)   BENCH_FLAGS="${BENCH_FLAGS} --cold-cache"; shift ;;
        --benchmarks)   BENCH_FLAGS="${BENCH_FLAGS} --benchmarks $2"; shift 2 ;;
        --datasets)     BENCH_FLAGS="${BENCH_FLAGS} --datasets $2"; shift 2 ;;
        --formats)      BENCH_FLAGS="${BENCH_FLAGS} --formats $2"; shift 2 ;;
        --include-additional) BENCH_FLAGS="${BENCH_FLAGS} --include-additional"; shift ;;
        --conda-env)    CONDA_ENV="$2"; shift 2 ;;
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

if [[ -n "$GPUS" ]]; then
    SBATCH_FLAGS+=(--gpus-per-node="${GPUS}")
fi
if [[ -n "$EXCLUSIVE" ]]; then
    SBATCH_FLAGS+=(--exclusive)
fi

# Auto-detect conda env if not specified
if [[ -z "$CONDA_ENV" ]]; then
    if [[ -n "$GPUS" ]]; then
        CONDA_ENV="scx-bench-gpu"
    else
        CONDA_ENV="scx-bench"
    fi
fi

# Resolve the conda env prefix
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

# Determine pyscx build features
MATURIN_FEATURES=""
if [[ -n "$GPUS" ]]; then
    MATURIN_FEATURES="--features gpu"
fi

# ---------------------------------------------------------------------------
# Build the job script
# ---------------------------------------------------------------------------
JOB_SCRIPT="${LOGS_DIR}/run_comprehensive.sh"
cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"

# Activate conda environment (non-interactive shell compatible)
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\$PATH"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

cd "\${REPO_ROOT}"

echo "=============================================="
echo "SCX Comprehensive Benchmark Suite — SLURM Job"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Partition: \${SLURM_JOB_PARTITION:-local}"
echo "CPUs:      \${SLURM_CPUS_PER_TASK:-$(nproc)}"
echo "Memory:    ${MEM}"
echo "Time:      ${TIME}"
echo "Conda env: ${CONDA_ENV} (\${CONDA_PREFIX})"
echo "Python:    \$(python --version)"
echo "Started:   \$(date -Is)"
echo ""

# Step 1: Build pyscx in release mode
echo "--- Building pyscx (release mode) ---"
cd "\${REPO_ROOT}/pyscx"
maturin develop --release ${MATURIN_FEATURES} 2>&1 | tail -5
cd "\${REPO_ROOT}"
echo ""

# Step 2: Run benchmarks
echo "--- Running Comprehensive Benchmarks ---"
python benchmarks/comprehensive/scripts/run_all.py ${BENCH_FLAGS}

echo ""
echo "=============================================="
echo "Completed: \$(date -Is)"
echo "=============================================="
EOFJOB
chmod +x "${JOB_SCRIPT}"

# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------
echo "=============================================="
echo "SCX Comprehensive Benchmark — SLURM Submission"
echo "=============================================="
echo "  Partition:  ${PARTITION}"
echo "  CPUs:       ${CPUS}"
echo "  Memory:     ${MEM}"
echo "  Time:       ${TIME}"
echo "  GPUs:       ${GPUS:-none}"
echo "  Conda env:  ${CONDA_ENV} (${CONDA_PREFIX})"
echo "  Exclusive:  ${EXCLUSIVE:-no}"
echo "  Bench args: ${BENCH_FLAGS:-'(defaults)'}"
echo "  Job script: ${JOB_SCRIPT}"
echo ""

JOB_ID=$(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted SLURM job: ${JOB_ID}"
echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Err: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.err"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs:    tail -f ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
