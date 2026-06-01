#!/bin/bash
#SBATCH --job-name=route_gate_verify
#SBATCH --partition=gpu,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
# Relative log paths: SLURM resolves these against the submission directory,
# so submit from the repo root (`sbatch benchmarks/comprehensive/scripts/...`).
#SBATCH --output=benchmarks/comprehensive/logs/route_gate_verify.%j.out
#SBATCH --error=benchmarks/comprehensive/logs/route_gate_verify.%j.err

# Self-contained GPU verification of the section-4 route-gate signals.
# Rebuilds pyscx (release, gpu) in the bench-gpu env, runs the touched
# accelerator triples in-process, and checks the new route floors.
#
# Paths are derived dynamically so any maintainer can run this unedited:
#   REPO_ROOT  — inferred from this script's location (scripts/ is 3 levels deep)
#   CONDA_BASE — override env var; defaults to $HOME/miniforge3
#   SCX_BENCH_GPU_ENV — conda env name; defaults to scx-bench-gpu
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p benchmarks/comprehensive/logs
source .env
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true

CONDA_BASE="${CONDA_BASE:-$HOME/miniforge3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "${SCX_BENCH_GPU_ENV:-scx-bench-gpu}"
# .env activates the dev .venv (exports VIRTUAL_ENV); maturin refuses when
# both VIRTUAL_ENV and CONDA_PREFIX are set. Drop VIRTUAL_ENV so maturin
# targets the active conda env (scx-bench-gpu).
unset VIRTUAL_ENV

echo "=== GPU visible ==="
nvidia-smi -L || true

# Ensure the bench-gpu env's pyscx carries the section-4 stamping changes.
echo "=== rebuilding pyscx (release, gpu) ==="
( cd pyscx && maturin develop --release --features gpu )

export SCX_GPU_DE_V3=1
OUT="${REPO_ROOT}/benchmarks/comprehensive/results/route_gate_verify_${SLURM_JOB_ID:-local}"
echo "=== running route-gate driver → ${OUT} ==="
python benchmarks/comprehensive/scripts/_route_gate_verify_driver.py "${OUT}"
echo "=== driver exit: $? ==="
