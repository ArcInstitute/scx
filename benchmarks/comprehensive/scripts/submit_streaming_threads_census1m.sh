#!/usr/bin/env bash
# =============================================================================
# Streaming conversion benchmark — thread-scaling submission for `census_1m`.
#
# Sets `SCX_CONV_STREAM_THREAD_COUNTS` so the benchmark iterates over
# `[1, 2, 4, 8, 16, 32]` thread counts, each in its own subprocess
# (rayon's global pool can't be reconfigured in-process). For each
# thread count, paired (streaming, materialize) conversions run and
# wall/peak-RSS land in the result JSON tagged with `extra.thread_count`.
#
# Expected outcome:
#   * streaming wall: flat across thread counts (Phase 4 MVP is a
#     sequential single-threaded loop).
#   * materialize wall: drops with threads until I/O- or
#     bandwidth-bound (rayon over pre-computed shard boundaries).
#
# Memory ceiling sized for materialise on census_1m (~14 GB) plus
# subprocess overhead and the parent process — 128 GB leaves clear
# headroom. Time ceiling sized for 6 thread counts × n_runs paired
# runs × ~80 s per materialise run + worst-case streaming wall
# multiplied by all thread counts.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
mkdir -p "$LOGS_DIR"

CONDA_ENV="${CONDA_ENV:-arcbench-scx-bench}"
CONDA_BASE="${HOME}/miniforge3"
CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
    echo "ERROR: conda not at ${CONDA_BASE}/bin/conda" >&2
    exit 1
fi
if [[ ! -d "$CONDA_PREFIX" ]]; then
    echo "ERROR: Conda env '${CONDA_ENV}' not at ${CONDA_PREFIX}" >&2
    exit 1
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi
DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"
FIXTURE="${DATA_DIR}/census_1m.h5ad"
if [[ ! -f "${FIXTURE}" ]]; then
    echo "ERROR: census_1m fixture not at ${FIXTURE}" >&2
    exit 1
fi

PARTITION="${PARTITION:-standard}"
CPUS="${CPUS:-32}"             # enough headroom for thread_count up to 32
MEM="${MEM:-128G}"
TIME="${TIME:-04:00:00}"
JOB_NAME="${JOB_NAME:-scx-conv-stream-threads-census1m}"
# Comma-separated list passed through to SCX_CONV_STREAM_THREAD_COUNTS.
THREAD_COUNTS="${THREAD_COUNTS:-1,2,4,8,16,32}"

JOB_SCRIPT="${LOGS_DIR}/${JOB_NAME}.sh"

cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\${HOME}/.cargo/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"
export HDF5_DIR="\${CONDA_PREFIX}"
export RUSTFLAGS="\${RUSTFLAGS:-} -C link-args=-Wl,-rpath,\${CONDA_PREFIX}/lib"

cd "\${REPO_ROOT}"

echo "=============================================="
echo "Streaming conversion thread scaling — census_1m"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Conda env: ${CONDA_ENV} (\${CONDA_PREFIX})"
echo "Fixture:   ${FIXTURE} (\$(du -h ${FIXTURE} | awk '{print \$1}'))"
echo "Threads:   ${THREAD_COUNTS}"
echo "Started:   \$(date -Is)"
echo ""

echo "--- Bootstrapping conda env (hdf5 + maturin) ---"
"${CONDA_BASE}/bin/conda" install -y --prefix "\${CONDA_PREFIX}" \\
    -c conda-forge "hdf5=1.12.*=nompi*" maturin 2>&1 | tail -10
echo ""

"\${HOME}/.cargo/bin/cargo" clean -p pyscx --release 2>&1 | tail -3 || true
rm -rf "\${REPO_ROOT}/target/maturin"
echo "--- Building pyscx (--features hdf5, release) ---"
cd "\${REPO_ROOT}/pyscx"
"\${CONDA_PREFIX}/bin/maturin" develop --release 2>&1 | tail -10
cd "\${REPO_ROOT}"
echo ""

echo "--- pyscx smoke ---"
"\${CONDA_PREFIX}/bin/python" -c "import pyscx; assert hasattr(pyscx, 'from_h5ad'); print('from_h5ad OK')"
echo ""

export SCX_WORK_DIR="${SCX_WORK_DIR}"
export SCX_DATA_DIR="${DATA_DIR}"
export SCX_CONV_STREAM_THREAD_COUNTS="${THREAD_COUNTS}"

echo "--- Running conversion_streaming with thread scaling ---"
"\${CONDA_PREFIX}/bin/python" -m benchmarks.comprehensive.scripts.run_all \\
    --benchmarks conversion_streaming \\
    --datasets census_1m \\
    --formats scx_auto

RESULTS_RAW="\${REPO_ROOT}/benchmarks/comprehensive/results/raw"
echo ""
echo "--- Results summary ---"
LATEST_JSON="\$(find "\${RESULTS_RAW}" -maxdepth 1 -name 'conversion_streaming__*census_1m*.json' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | awk '{print \$2}')"
if [[ -n "\${LATEST_JSON}" ]]; then
    echo "Latest result: \${LATEST_JSON}"
    "\${CONDA_PREFIX}/bin/python" - <<'PYEOF'
import json, os
p = os.environ.get("LATEST_JSON") or ""
PYEOF
    cat "\${LATEST_JSON}"
else
    echo "No conversion_streaming census_1m result JSON found under \${RESULTS_RAW}"
fi

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
echo "Submitted thread-scaling census_1m: job ${JOB_ID}"
echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Err: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.err"
echo ""
echo "Monitor: squeue -u \$USER -j ${JOB_ID}"
echo "Tail:    tail -f ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
