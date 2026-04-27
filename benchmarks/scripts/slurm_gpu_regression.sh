#!/bin/bash
# DEPRECATED (Phase 9 2026-04-23). Use the comprehensive framework:
#     sbatch benchmarks/comprehensive/scripts/run_slurm.sh --include-accel
# Kept in-tree one release for rollback.
#
# SLURM wrapper for the Phase-8 GPU accelerator regression driver
# (parallel-grid mode — the documented model per AGENTS.md:21).
#
# This outer job does NOT take a GPU allocation. It runs on a small
# CPU slot and has exactly two responsibilities:
#
#   1. Rebuild pyscx with --release --features gpu into the shared
#      conda env (one rebuild, visible to every downstream cell).
#   2. Invoke `gpu_regression_driver.sh` (default mode = parallel
#      grid), which then sbatches per-(benchmark) cells on GPU nodes.
#
# Step 2 itself takes no GPU — it just sbatches + polls squeue until
# all cells finish, then runs the diff against the frozen baseline.
#
# Submit with:
#     sbatch benchmarks/scripts/slurm_gpu_regression.sh
#
# For one-off interactive debugging on a GPU node you already hold:
#     bash benchmarks/scripts/gpu_regression_driver.sh --inline
# which bypasses sbatch and runs every benchmark sequentially in the
# current allocation. Not recommended for regression runs — per
# AGENTS.md:21, "Always use parallel SLURM job submission (one job per
# benchmark x dataset pair) rather than sequential single-job scripts."
#
# Exit: 0 pass, 1 regression / hard-floor flag, 2 infra.

#SBATCH --job-name=scx_gpu_regression
#SBATCH --partition=cpu_preemptible
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=benchmarks/logs/gpu_regression_%j.log

set -euo pipefail
REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"

echo "=== SLURM job $SLURM_JOB_ID on $(hostname) — coordinator (CPU-only) ==="
date -Iseconds
echo ""

# Source .env so the driver inherits SCX_WORK_DIR, etc.
if [[ -f "$REPO/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

# Bind conda env for the rebuild step (driver re-binds internally).
CONDA_PREFIX="${CONDA_PREFIX_OVERRIDE:-/home/nickyoungblut/miniforge3/envs/scx-gpu}"
export CONDA_PREFIX PATH LD_LIBRARY_PATH
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Invoke the driver in default (parallel-grid) mode. "$@" forwards any
# additional flags (--skip-tests, --pre, --dataset, etc) passed on the
# sbatch CLI.
exec bash benchmarks/scripts/gpu_regression_driver.sh "$@"
