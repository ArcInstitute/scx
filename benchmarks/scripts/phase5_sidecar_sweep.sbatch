#!/usr/bin/env bash
# =============================================================================
# Phase 5 / T5.1+T5.2 — IndexPlanDataset sidecar on/off × batch-size sweep.
#
# Measures the sidecar (L1+L2) throughput win vs. legacy full-shard decode and
# confirms peak RSS does not regress, on the regenerated sidecar-carrying
# scx_auto fixtures. One job per dataset.
#
# Usage:
#   DATASET=census_1m N_BATCHES=60 sbatch --job-name=p5-sweep-census_1m \
#       --mem=200G --time=6:00:00 benchmarks/scripts/phase5_sidecar_sweep.sbatch
# =============================================================================

#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/p5-sweep-%x-%j.out
#SBATCH --error=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/p5-sweep-%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/nickyoungblut/dev/rust/scx}"
cd "$REPO_ROOT"
mkdir -p benchmarks/comprehensive/logs

DATASET="${DATASET:?set DATASET=<name>}"
N_BATCHES="${N_BATCHES:-60}"

echo "=== Phase 5 sidecar sweep on $(hostname) (job ${SLURM_JOB_ID}) ==="
echo "    dataset : ${DATASET}  n_batches=${N_BATCHES}"

eval "$(/home/nickyoungblut/miniforge3/bin/conda shell.bash hook)"
conda activate scx-bench
python -c "import pyscx; assert pyscx.__version__ >= '0.9', pyscx.__version__; print('pyscx', pyscx.__version__)"

export SCX_WORK_DIR=/data/scx-dev
export SCX_DATA_DIR=/data/scx-dev/benchmarks/datasets
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export RUST_LOG=warn
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Report fixture sidecar coverage before benchmarking (non-fatal).
python benchmarks/scripts/check_sidecar_coverage.py --report \
    --scx-bin "${REPO_ROOT}/target/release/scx" \
    "${SCX_DATA_DIR}/${DATASET}_auto.scx"

python benchmarks/scripts/phase5_sidecar_sweep.py --dataset "${DATASET}" --n-batches "${N_BATCHES}"

echo "=== Done (job ${SLURM_JOB_ID}) ==="
