#!/usr/bin/env bash
# =============================================================================
# GPU benchmark on Lambda HPC.
#
# Runs the `pyscx_index_plan_dataset_workers2` scenario against a
# synthetic 100k-cell fixture, alongside a `num_workers=0` reference for
# direct comparison. Captures throughput / cells-per-sec / peak RSS into
# a JSON file under `benchmarks/comprehensive/results/`.
#
# Usage:
#     sbatch benchmarks/scripts/lambda_phase7_index_plan_workers2.sbatch
# =============================================================================

#SBATCH --job-name=scx-phase7-index-plan-w2
#SBATCH --partition=preemptible
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/phase7-index-plan-w2-%j.out
#SBATCH --error=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/phase7-index-plan-w2-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/nickyoungblut/dev/rust/scx}"
cd "$REPO_ROOT"

mkdir -p benchmarks/comprehensive/logs

echo "=== Lambda HPC node: $(hostname) ==="
echo "    SLURM job id    : ${SLURM_JOB_ID}"
echo "    cpus available  : ${SLURM_CPUS_ON_NODE:-?}"
echo "    GPUs            : ${SLURM_GPUS:-?} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-})"
echo "    repo            : $REPO_ROOT"
nvidia-smi -L 2>&1 || echo "(no nvidia-smi)"

OUT="benchmarks/comprehensive/results/phase7-lambda-index-plan-workers2-${SLURM_JOB_ID}.json"

export RUST_LOG=warn
export OMP_NUM_THREADS=1
export SCX_WORK_DIR=/scratch/${UID:-10011}/scx_phase7
mkdir -p "$SCX_WORK_DIR"

.venv/bin/python benchmarks/scripts/lambda_phase7_index_plan_workers2.py --output "$OUT"

echo "=== Done — results: $OUT ==="
