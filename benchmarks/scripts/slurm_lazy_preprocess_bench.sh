#!/bin/bash
#SBATCH --job-name=lazy_preprocess_bench
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=benchmarks/logs/lazy_preprocess_bench_%j.log

set -euo pipefail

REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"
mkdir -p benchmarks/logs benchmarks/results

echo "=== System info ==="
hostname
date
lscpu | head -20
free -h
echo ""

echo "=== Phase 4d Lazy Preprocessing Benchmark ==="
echo ""

# Validate on pbmc3k (quick correctness check)
echo "--- Step 1: Correctness validation (pbmc3k) ---"
.venv/bin/python benchmarks/scripts/benchmark_lazy_preprocess.py \
    --mode validate --datasets pbmc3k
echo ""

# Benchmark on census_1m (memory + timing)
echo "--- Step 2: Performance benchmarks (census_1m) ---"
.venv/bin/python benchmarks/scripts/benchmark_lazy_preprocess.py \
    --mode bench --datasets census_1m
echo ""

# Evaluate Go/No-Go gates
echo "--- Step 3: Go/No-Go gate ---"
.venv/bin/python benchmarks/scripts/benchmark_lazy_preprocess.py \
    --mode gate
echo ""

echo "=== DONE ==="
echo "Results: benchmarks/results/lazy_preprocess_benchmark.md"
