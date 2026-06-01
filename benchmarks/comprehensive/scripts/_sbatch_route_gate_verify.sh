#!/bin/bash
#SBATCH --job-name=route_gate_verify
#SBATCH --partition=gpu,gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=56G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/route_gate_verify.%j.out
#SBATCH --error=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/route_gate_verify.%j.err

# Self-contained GPU verification of the section-4 route-gate signals.
# Rebuilds pyscx (release, gpu) in the bench-gpu env, runs the touched
# accelerator triples in-process, and checks the new route floors.
set -euo pipefail

cd /home/nickyoungblut/dev/rust/scx
source .env
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh
conda activate scx-bench-gpu
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
OUT="/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/results/route_gate_verify_${SLURM_JOB_ID:-local}"
echo "=== running route-gate driver → ${OUT} ==="
python benchmarks/comprehensive/scripts/_route_gate_verify_driver.py "${OUT}"
echo "=== driver exit: $? ==="
