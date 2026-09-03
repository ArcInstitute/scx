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
#     # Default: "full" tier.
#     sbatch benchmarks/comprehensive/scripts/slurm_capture_baseline.sh
#
#     # Small tier (D1-D4 only) — fast sanity check.
#     sbatch benchmarks/comprehensive/scripts/slurm_capture_baseline.sh --tier small
#
#     # XL tier (includes census_5m) — routes to cpu_high_mem.
#     sbatch benchmarks/comprehensive/scripts/slurm_capture_baseline.sh --tier xl
#
# The orchestrator's own sizing is the two #SBATCH lines below, and SLURM lets a
# command-line flag override an in-file directive — `#SBATCH` itself cannot read
# a shell variable, so that is the only way to change them per run:
#
#     sbatch --partition=cpu_preemptible --time=08:00:00 <this script> --tier small
#
# GPU *cells* are sized separately and ignore both this partition and
# `--partition`: set `SCX_BENCH_GPU_PARTITION` (see `config.GPU_PARTITION`).
#
# WHY 48 h AND NOT PREEMPTIBLE. The last two tier-full captures took 12.94 h and
# 6.40 h wall (2026-06-09 / 06-11, ~1,533 scheduled cells), and thirteen
# benchmarks have been registered since — worth ~24,000 estimated job-minutes on
# their own. The old `--time=12:00:00` was already under the observed wall, and
# the orchestrator holding submitit's wait loop is exactly the process a
# preemptible partition should not host: killing it strands the whole capture
# with results on disk and no snapshot. `cpu_batch` allows 14 days.
# =============================================================================

#SBATCH --job-name=scx-baseline
#SBATCH --partition=cpu_batch
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=48:00:00
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
# Thread pinning: DELIBERATELY NOT SET HERE.
#
# This block used to `export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"` (and
# OMP / MKL), justified as "match fingerprint_accelerators.py's defaults so
# that any fingerprint recomputed later on the same node produces identical
# bytes". That reasoning does not hold: `fingerprint_accelerators.run_fingerprints`
# opens with
#     os.environ.setdefault("RAYON_NUM_THREADS", str(PINNED_THREADS))
# so it pins ITSELF whatever the parent environment says. The export therefore
# changed nothing about fingerprint determinism and everything about the ~1500
# benchmark cells, which ran single-threaded.
#
# The damage is that it makes a capture incomparable with every promoted
# baseline in the tree. `environment.json` records `determinism_env`, and every
# promoted baseline reads `unset` for all three — so none was captured through
# this wrapper. A pinned capture gated against an unpinned baseline reports
# `accel_knn` +888%, `read_full` +469-616% on every SCX codec at census, and
# `bench_csc_dispatch` +585%, purely because rayon had one thread. Overall
# median ratio was 1.03x, so the distortion hides in the tail and reads as a
# handful of catastrophic regressions rather than a methodology error.
#
# Floors authored from single-threaded medians are worse than none: a normal
# multi-threaded run clears them by 5-9x, and the floor reads as coverage while
# providing none.
#
# Set them yourself if you specifically want a single-threaded capture; they
# are inherited, not overridden.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SCX_BENCH_N_RANKS — opt-in, deliberately not set.
#
# `cellset_gather`'s multi-rank arm emits
# `rank_scaling_efficiency__gather_random_r4`, which three floors name, and
# `multirank.resolve_n_ranks` gates it on this variable (default 1). So those
# floors have never had a capture to measure them.
#
# It is not on by default because the arm runs the gather at 1 rank AND at N,
# multiplying the cost of the benchmark that is already the most expensive at
# census scale — `cellset_gather/census_500k` does not fit a 205-minute budget
# as it is. Enable it for a targeted capture instead:
#
#     sbatch --export=ALL,SCX_BENCH_N_RANKS=4 <this script> \
#         --datasets tabula_sapiens_100k --benchmarks cellset_gather
#
# tabula is cheap (0.5 min of measured wall for the whole cell) and is the only
# one of the three floored datasets that can currently be captured.
# ---------------------------------------------------------------------------

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
