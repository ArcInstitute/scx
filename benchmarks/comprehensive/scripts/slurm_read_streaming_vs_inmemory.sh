#!/usr/bin/env bash
# =============================================================================
# Submit the Phase 6b streaming-vs-in-memory benchmarks to Lambda SLURM.
#
# Two parallel sweeps:
#
#   1. ``read_streaming_vs_inmemory``        on D5–D7 (census_500k /
#      census_1m / census_5m) × ``scx_auto``. Demonstrates the
#      streaming RSS-vs-wall trade-off at scale.
#
#   2. ``multimodal_read_streaming_vs_inmemory`` on K1 + K2
#      (cite_seq_pbmc / multiome_pbmc) × the SCX multimodal variants.
#      Exercises the Phase 6b ``to_mudata(backed=True)`` path.
#
# Both sweeps use ``--cold-cache`` so the per-job page cache is dropped
# between iterations — without it the in-memory mode would lean on warm
# pages from the streaming run that preceded it, washing out the
# meaningful comparison.
#
# Internally each sweep delegates to ``run_parallel.py``, which uses
# submitit to submit one SLURM job per (benchmark, dataset, format)
# triple. This wrapper exists so a single ``sbatch`` (or a direct
# invocation from any node) launches the full set with the right
# defaults for Lambda HPC.
#
# Usage:
#   sbatch benchmarks/comprehensive/scripts/slurm_read_streaming_vs_inmemory.sh
#
# Or directly from a login / worker node (the script itself is a thin
# wrapper that calls ``run_parallel.py``; submitit creates the actual
# sub-jobs):
#   bash benchmarks/comprehensive/scripts/slurm_read_streaming_vs_inmemory.sh
#
# Pass-through CLI overrides go after ``--``:
#   sbatch slurm_read_streaming_vs_inmemory.sh -- --partition standard
#   sbatch slurm_read_streaming_vs_inmemory.sh -- --datasets census_1m
#
# =============================================================================

#SBATCH --job-name=scx-stream-vs-inmem
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/tmp/scx-stream-vs-inmem-%j.out
#SBATCH --error=/tmp/scx-stream-vs-inmem-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
fi
cd "$REPO_ROOT"

LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
TMP_OUT="/tmp/scx-stream-vs-inmem-${SLURM_JOB_ID:-$$}.out"
TMP_ERR="/tmp/scx-stream-vs-inmem-${SLURM_JOB_ID:-$$}.err"
trap '
    mkdir -p "${LOGS_DIR}" 2>/dev/null || true
    [[ -f "${TMP_OUT}" ]] && cp "${TMP_OUT}" "${LOGS_DIR}/stream_vs_inmem_${SLURM_JOB_ID:-$$}.out" 2>/dev/null || true
    [[ -f "${TMP_ERR}" ]] && cp "${TMP_ERR}" "${LOGS_DIR}/stream_vs_inmem_${SLURM_JOB_ID:-$$}.err" 2>/dev/null || true
' EXIT

# Forwarded args.
EXTRA_ARGS=("$@")

# ---------------------------------------------------------------------------
# Conda / venv activation. Prefer scx-bench; fall back to .venv.
# ---------------------------------------------------------------------------
CONDA_BASE=""
if [[ -d "$HOME/miniforge3" ]]; then
    CONDA_BASE="$HOME/miniforge3"
elif command -v conda &>/dev/null; then
    CONDA_BASE="$(conda info --base 2>/dev/null || echo "")"
fi

if [[ -n "${CONDA_BASE}" && -d "${CONDA_BASE}/envs/scx-bench" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate scx-bench
    echo "[stream-vs-inmem] activated conda env scx-bench"
    PYTHON="python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    echo "[stream-vs-inmem] using project .venv"
else
    echo "[stream-vs-inmem] ERROR: no scx-bench conda env and no .venv/" >&2
    exit 1
fi

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

# Determinism knobs — same as slurm_capture_baseline.sh.
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

# Lambda has no ``cpu_high_mem`` partition; ``large_batch`` fills the
# same role (16 nodes, 14-day timeout, full 1.8 TB per node). Redirect
# the auto-promotion target so ``partition_for_memory()`` in
# ``run_parallel.py`` sends >MEM_HIGH_MEM_THRESHOLD_GB jobs there
# instead of failing with an unknown-partition error.
export SCX_BENCH_HIGH_MEM_PARTITION="${SCX_BENCH_HIGH_MEM_PARTITION:-large_batch}"

mkdir -p "${LOGS_DIR}"

echo "[stream-vs-inmem] git:    $(git rev-parse HEAD)"
echo "[stream-vs-inmem] branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[stream-vs-inmem] python: $(${PYTHON} --version)"

RUN_PARALLEL="${REPO_ROOT}/benchmarks/comprehensive/scripts/run_parallel.py"

# ---------------------------------------------------------------------------
# Sweep 1 — single-modality streaming vs in-memory on D5/D6/D7.
#
# Lambda partition choice: ``preemptible`` for the smaller two (faster
# scheduling against the full 22-node pool, 60s SIGTERM grace is fine
# at this read-only workload) and ``large_batch`` for census_5m
# (longer timeout headroom; the eager mode has to materialise ~75 GB
# into one CSR before iteration begins). ``run_parallel.py`` sizes
# memory per-job via ``estimate_memory_gb``; the wrapper just sets a
# sensible floor.
# ---------------------------------------------------------------------------
echo
echo "[stream-vs-inmem] Sweep 1/2 — single-modality on D5/D6/D7"
${PYTHON} "${RUN_PARALLEL}" \
    --benchmarks read_streaming_vs_inmemory \
    --datasets census_500k census_1m \
    --formats scx_auto \
    --partition preemptible \
    --cpus 8 \
    --mem-gb 64 \
    --timeout 240 \
    --cold-cache \
    "${EXTRA_ARGS[@]}"

${PYTHON} "${RUN_PARALLEL}" \
    --benchmarks read_streaming_vs_inmemory \
    --datasets census_5m \
    --formats scx_auto \
    --partition large_batch \
    --cpus 16 \
    --mem-gb 256 \
    --timeout 480 \
    --cold-cache \
    "${EXTRA_ARGS[@]}"

# ---------------------------------------------------------------------------
# Sweep 2 — multimodal streaming vs in-memory on K1 + K2.
#
# These datasets are small in absolute terms (CITE-seq ~85 MB,
# Multiome ~1 GB on disk) so peak RSS for the eager path stays in the
# low GB — but the comparison is *qualitative*: it locks the Phase 6b
# read path against regressions even when one of the modalities (e.g.
# ATAC's 144k peaks) inflates the column count.
# ---------------------------------------------------------------------------
echo
echo "[stream-vs-inmem] Sweep 2/2 — multimodal on K1 + K2"
${PYTHON} "${RUN_PARALLEL}" \
    --benchmarks multimodal_read_streaming_vs_inmemory \
    --datasets cite_seq_pbmc multiome_pbmc \
    --formats scx_multimodal_per_modality_auto \
    --partition preemptible \
    --cpus 4 \
    --mem-gb 32 \
    --timeout 120 \
    --cold-cache \
    "${EXTRA_ARGS[@]}"

echo
echo "[stream-vs-inmem] All sweeps submitted. Inspect:"
echo "  squeue -u \$USER"
echo "  ls -lt benchmarks/comprehensive/results/raw/ | head"
