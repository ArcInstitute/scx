#!/usr/bin/env bash
# =============================================================================
# `SparseCellSetDataset` scattered-read route A/B.
#
# Reframes a copy of one registered fixture to v4 (the registered ones are v3,
# where both routes collapse to the same path) and prices the two routes.
#
# Usage:
#     SOURCE=/data/scx-dev/benchmarks/datasets/tabula_sapiens_100k_auto.scx \
#     WORKDIR=$HOME/tmp/cellset-routes OUT=$HOME/tmp/cellset-routes/tabula.json \
#         sbatch --job-name=cellset-routes-tabula \
#         benchmarks/scripts/bench_cellset_scatter_routes.sbatch
# =============================================================================

#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/cellset-routes-%x-%j.out
#SBATCH --error=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/cellset-routes-%x-%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/nickyoungblut/dev/rust/scx}"
cd "$REPO_ROOT"
mkdir -p benchmarks/comprehensive/logs

SOURCE="${SOURCE:?set SOURCE=<path to .scx>}"
WORKDIR="${WORKDIR:?set WORKDIR=<scratch dir>}"
OUT="${OUT:?set OUT=<result .json>}"
N_RUNS="${N_RUNS:-3}"

eval "$(/home/nickyoungblut/miniforge3/bin/conda shell.bash hook)"
conda activate scx-bench
# 5c's lesson: whether a job survives otherwise depends on whether the
# submitting shell had a venv active.
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

export SCX_WORK_DIR=/data/scx-dev
export SCX_DATA_DIR=/data/scx-dev/benchmarks/datasets
export PYTHONPATH="${REPO_ROOT}"
export RUST_LOG=warn
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1   # flush progress so a timeout-kill leaves a trail

python -c "import pyscx; print('pyscx', pyscx.__version__, pyscx.__file__)"
python -c "
import pyscx, sys
sig = pyscx.SparseCellSetDataset.__text_signature__ or ''
assert 'scatter_block_index' in sig, 'this pyscx predates the kwarg; rebuild with maturin develop --release --features gpu'
print('scatter_block_index kwarg: present')
"

python benchmarks/scripts/bench_cellset_scatter_routes.py \
    --source "$SOURCE" --workdir "$WORKDIR" --out "$OUT" --n-runs "$N_RUNS"
