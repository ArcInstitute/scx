#!/usr/bin/env bash
# =============================================================================
# `SparseCellSetDataset` scattered-read route A/B.
#
# Reframes a copy of one registered fixture at a pinned row-group geometry and
# prices the two routes against it, one BenchmarkResult per arm.
#
# (The reframe originally existed because the registered fixtures were v3, where
# both routes collapse to full-shard. They are v4-framed now; see the driver's
# docstring for why the reframe is kept regardless.)
#
# Usage:
#     for f in tabula_sapiens_100k pbmc10k; do
#       SOURCE=$SCX_DATA_DIR/${f}_auto.scx \
#       WORKDIR=$SCRATCH/cellset-routes/$f OUT_DIR=$SCRATCH/cellset-routes \
#       RAW_SUBDIR=phase0_scatter_routes PLAN=random \
#         sbatch --job-name=cellset-routes-$f \
#         benchmarks/scripts/bench_cellset_scatter_routes.sbatch
#     done
#
# Submit these ONE AT A TIME (chain with `--dependency=afterany:$prev`): two
# arms of a timing A/B co-scheduled on one node bias each other unevenly, which
# is worse than noise.
# =============================================================================

# CPU-only: the subject is the cell-set gather's shard-decode route. The
# original directives were Lambda's (`--partition=standard --gres=gpu:1`),
# which reserve a GPU this benchmark never touches and name a partition that
# does not exist on every cluster. Override the partition at submit time
# (`sbatch --partition=...`) where the default does not apply.
#SBATCH --partition=cpu_preemptible
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/cellset-routes-%x-%j.out
#SBATCH --error=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/cellset-routes-%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/nickyoungblut/dev/rust/scx}"
cd "$REPO_ROOT"
mkdir -p benchmarks/comprehensive/logs

SOURCE="${SOURCE:?set SOURCE=<path to ONE .scx>}"
WORKDIR="${WORKDIR:?set WORKDIR=<scratch dir>}"
OUT_DIR="${OUT_DIR:?set OUT_DIR=<dir for result .json files>}"
mkdir -p "$OUT_DIR"
N_RUNS="${N_RUNS:-3}"
N_BATCHES="${N_BATCHES:-50}"   # the `on` route re-decodes per batch, so this is
                               # the knob that decides whether the job fits its wall

eval "$(/home/nickyoungblut/miniforge3/bin/conda shell.bash hook)"
conda activate scx-bench
# 5c's lesson: whether a job survives otherwise depends on whether the
# submitting shell had a venv active.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

# Resolve the data root from the repo's own `.env` unless the submitting
# environment already pinned it. Hardcoding one cluster's path here made the
# script unrunnable anywhere else, and silently so — `bench_env` would fall
# back to `$SCX_WORK_DIR/benchmarks/datasets` under a directory that does not
# exist, and the failure surfaces as a missing fixture rather than as a
# misconfiguration.
if [[ -z "${SCX_WORK_DIR:-}" && -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi
export SCX_WORK_DIR="${SCX_WORK_DIR:?set SCX_WORK_DIR or provide a repo-root .env}"
export SCX_DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"
# `$REPO_ROOT/pyscx/python` FIRST, and this is load-bearing, not tidiness.
# `scx-bench` has a pyscx *wheel* installed in site-packages (from
# /tmp/pr08-wheel-branch-*), so a bare `import pyscx` resolves to that build —
# and it reports the same `__version__` as the checkout while being a different
# binary. The wheel installed 2026-09-09 predates OPT-FORMATIO-1 (merged
# 2026-09-11) and has no `row_group_*` counters at all, which is precisely the
# change this A/B exists to re-measure across. `$REPO_ROOT` alone does NOT
# fix it: the repo root holds the `pyscx/` *crate* directory, which has no
# `__init__.py`, so the import falls through to site-packages.
export PYTHONPATH="${REPO_ROOT}/pyscx/python:${REPO_ROOT}"
# RUST_LOG governs the `scx` CLI SUBPROCESSES this driver spawns (`scx optimize`
# for the reframe, `scx info` for the premise checks): scx-cli initialises
# `env_logger::Builder::from_env(...default_filter_or("info"))`, so RUST_LOG is
# the gate there — and `warn` is BELOW that `info` default, i.e. this line makes
# the CLI quieter, it does not make it noisier.
#
# It does NOT govern pyscx's in-process Rust logging. pyscx calls
# `pyo3_log::try_init()`, which forwards `log` records into Python `logging`;
# with no handler configured, `logging.lastResort` writes WARNING+ to
# `sys.stderr`. Measured: a `scx-loader` warn fires with RUST_LOG unset, and
# `contextlib.redirect_stderr` captures it while raw fd 2 stays empty. The
# filter that decides there is the *Python* root logger level.
export RUST_LOG=warn
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # flush progress so a timeout-kill leaves a trail

python -c "import pyscx; print('pyscx', pyscx.__version__, pyscx.__file__)"
# Preflight the build, not just the feature. Each of these has silently
# produced a wrong measurement before:
#   1. the resolved pyscx is THIS checkout's, not an env wheel with a matching
#      version string;
#   2. the `scatter_block_index` kwarg exists at all.
# The third check — that the build post-dates OPT-FORMATIO-1 — lives in the
# driver itself (`_require_row_group_counters`) so it protects every
# invocation, not only this wrapper.
REPO_ROOT="$REPO_ROOT" python -c "
import os, sys, pyscx
want = os.path.join(os.environ['REPO_ROOT'], 'pyscx', 'python') + os.sep
if not pyscx.__file__.startswith(want):
    sys.exit(f'preflight: pyscx resolved to {pyscx.__file__}, not {want} — '
             'the env wheel would be measured instead of this checkout')
sig = pyscx.SparseCellSetDataset.__text_signature__ or ''
assert 'scatter_block_index' in sig, 'this pyscx predates the kwarg; rebuild with maturin develop --release'
print('pyscx build: this checkout; scatter_block_index kwarg: present')
"

# ONE source per job, deliberately. `benchmarks/README.md` requires independent
# benchmark x dataset pairs to be submitted as separate SLURM jobs so they run
# concurrently; looping several datasets inside one allocation serialises them
# and bills the largest dataset's wall clock to all of them.
stem=$(basename "$SOURCE" .scx)
python benchmarks/scripts/bench_cellset_scatter_routes.py \
    --source "$SOURCE" --workdir "$WORKDIR" --out "$OUT_DIR/$stem.json" \
    --n-runs "$N_RUNS" --n-batches "$N_BATCHES" \
    --plan "${PLAN:-random}" \
    ${RAW_SUBDIR:+--raw-subdir "$RAW_SUBDIR"}
