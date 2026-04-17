#!/usr/bin/env bash
# =============================================================================
# Capture the pre-code-review regression baseline as a single SLURM job.
#
# This is a thin wrapper around capture_baseline.py; the Python script itself
# submits the per-(benchmark, dataset, format) jobs via submitit and blocks
# until they all finish.  The outer sbatch job exists mainly so the submission
# and the subsequent archival+fingerprint step run from inside the compute
# queue rather than from a head node.
#
# Usage:
#     # Default: "full" tier on cpu_preemptible.
#     sbatch benchmarks/comprehensive/scripts/slurm_capture_baseline.sh
#
#     # Small tier (D1-D4 only) — fast sanity check.
#     sbatch benchmarks/comprehensive/scripts/slurm_capture_baseline.sh --tier small
#
#     # XL tier (includes census_5m) — routes to cpu_high_mem.
#     sbatch benchmarks/comprehensive/scripts/slurm_capture_baseline.sh --tier xl
#
# =============================================================================

#SBATCH --job-name=scx-baseline
#SBATCH --partition=cpu_preemptible
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/tmp/scx-baseline-%j.out
#SBATCH --error=/tmp/scx-baseline-%j.err

set -euo pipefail

# Under sbatch, $0 points at SLURM's spool copy — not the original script —
# so deriving the repo from dirname($0) produces a path like /var/.  Prefer
# SLURM_SUBMIT_DIR (set by sbatch) and fall back to the original derivation
# only when this script is sourced / run outside SLURM.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
fi
cd "$REPO_ROOT"

# Mirror the tmp log files back into the repo at job end so everything
# lives alongside the other benchmark logs.  /tmp is the canonical writable
# path while the job is running; the copy is best-effort.
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
TMP_OUT="/tmp/scx-baseline-${SLURM_JOB_ID:-$$}.out"
TMP_ERR="/tmp/scx-baseline-${SLURM_JOB_ID:-$$}.err"
trap '
    mkdir -p "${LOGS_DIR}" 2>/dev/null || true
    [[ -f "${TMP_OUT}" ]] && cp "${TMP_OUT}" "${LOGS_DIR}/baseline_${SLURM_JOB_ID:-$$}.out" 2>/dev/null || true
    [[ -f "${TMP_ERR}" ]] && cp "${TMP_ERR}" "${LOGS_DIR}/baseline_${SLURM_JOB_ID:-$$}.err" 2>/dev/null || true
' EXIT

# ---------------------------------------------------------------------------
# CLI passthrough: everything after `--` is forwarded to capture_baseline.py.
# ---------------------------------------------------------------------------
PY_ARGS=("$@")

# ---------------------------------------------------------------------------
# Conda / venv activation.  Prefer scx-bench; fall back to the project .venv.
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
    echo "[baseline] activated conda env scx-bench"
    PYTHON="python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    echo "[baseline] using project .venv"
else
    echo "[baseline] ERROR: no scx-bench conda env and no .venv/" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Determinism knobs.  Match fingerprint_accelerators.py's defaults so that
# any fingerprint recomputed later on the same node produces identical bytes.
# ---------------------------------------------------------------------------
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# ---------------------------------------------------------------------------
# Load .env if present.  Populates SCX_WORK_DIR / SCX_DATA_DIR.
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

mkdir -p "${REPO_ROOT}/benchmarks/comprehensive/logs"

# ---------------------------------------------------------------------------
# Run the capture.
# ---------------------------------------------------------------------------
echo "[baseline] git: $(git rev-parse HEAD)  branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[baseline] python: $(${PYTHON} --version)"
echo "[baseline] invoking capture_baseline.py ${PY_ARGS[*]}"

${PYTHON} "${REPO_ROOT}/benchmarks/comprehensive/scripts/capture_baseline.py" "${PY_ARGS[@]}"
