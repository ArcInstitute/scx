#!/bin/bash
#SBATCH --job-name=lazy_preprocess_bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/logs/lazy_preprocess_bench_%j.log

set -euo pipefail

REPO="/home/nickyoungblut/dev/rust/scx"
VENV="${REPO}/.venv"
PYTHON="${VENV}/bin/python"

cd "$REPO"
mkdir -p benchmarks/logs benchmarks/results

echo "=== System info ==="
hostname
date
lscpu | head -20 || true
free -h
echo ""

echo "=== Building pyscx in release mode ==="
cd pyscx && "${VENV}/bin/maturin" develop --release && cd ..
echo ""

echo "=== Phase 4d Lazy Preprocessing Benchmark ==="
echo ""

# Validate on pbmc3k (quick correctness check)
echo "--- Step 1: Correctness validation (pbmc3k) ---"
"$PYTHON" benchmarks/scripts/benchmark_lazy_preprocess.py \
    --mode validate --datasets pbmc3k 2>&1
echo ""

# Benchmark on census_1m (memory + timing)
echo "--- Step 2: Performance benchmarks (census_1m) ---"
"$PYTHON" benchmarks/scripts/benchmark_lazy_preprocess.py \
    --mode bench --datasets census_1m 2>&1
echo ""

# Evaluate Go/No-Go gates
echo "--- Step 3: Go/No-Go gate ---"
"$PYTHON" benchmarks/scripts/benchmark_lazy_preprocess.py \
    --mode gate 2>&1
echo ""

echo "=== DONE ==="
echo "Results: benchmarks/results/lazy_preprocess_benchmark.md"
