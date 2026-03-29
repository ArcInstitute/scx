#!/bin/bash
#SBATCH --job-name=scx_gpu_pca_opt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/logs/gpu_pca_optimization_%j.log

set -euo pipefail

REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"

# Use the scx-gpu conda environment for RAPIDS compatibility.
export CONDA_PREFIX="/home/nickyoungblut/miniforge3/envs/scx-gpu"
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p benchmarks/logs benchmarks/results

echo "============================================================"
echo "SCX GPU PCA Optimization Benchmark"
echo "============================================================"
echo ""

echo "=== System info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Building pyscx in release mode (with GPU features) ==="
cargo clean -p scx-gpu -p pyscx --release 2>/dev/null || true
cd pyscx && maturin develop --release --features gpu && cd ..
echo ""

# ── 1. GPU PCA Validation (cosine similarity) ──
echo "============================================================"
echo "1/2  GPU PCA Validation (pbmc3k + census_1m)"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_pca.py --mode validate 2>&1
echo ""

# ── 2. GPU PCA Timing Benchmark ──
echo "============================================================"
echo "2/2  GPU PCA Timing Benchmark"
echo "============================================================"
python benchmarks/scripts/benchmark_gpu_pca.py --mode bench --n-runs 3 2>&1
echo ""

# ── Summary ──
echo "============================================================"
echo "Results"
echo "============================================================"
echo ""
echo "--- JSON outputs ---"
ls -la benchmarks/results/gpu_pca_*.json 2>/dev/null || echo "No JSON files found"
echo ""
echo "--- Report ---"
cat benchmarks/results/gpu_pca_benchmark.md 2>/dev/null || echo "No report generated"
echo ""

echo "=== DONE ==="
