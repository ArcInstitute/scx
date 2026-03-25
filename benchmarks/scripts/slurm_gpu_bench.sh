#!/bin/bash
#SBATCH --job-name=scx_gpu_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/gpu_bench_%j.log

set -euo pipefail

REPO="/home/nickyoungblut/dev/rust/scx"
cd "$REPO"

mkdir -p benchmarks/logs benchmarks/results

echo "=== System info ==="
hostname
date
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

echo "=== Building scx-gpu in release mode ==="
cargo build --release -p scx-gpu --features bench --bin gpu_bench 2>&1
echo ""

echo "=== Running GPU microbenchmarks ==="
# stdout → JSON file, stderr → SLURM log (progress messages)
cargo run --release -p scx-gpu --features bench --bin gpu_bench \
    > benchmarks/results/gpu_bench_json.txt
echo ""

echo "=== JSON output ==="
cat benchmarks/results/gpu_bench_json.txt
echo ""

echo "=== Generating markdown report ==="
.venv/bin/python benchmarks/scripts/benchmark_gpu_decode.py \
    --from-json benchmarks/results/gpu_bench_json.txt \
    --output benchmarks/results/gpu_benchmark.md

echo ""
echo "=== Report ==="
cat benchmarks/results/gpu_benchmark.md

echo ""
echo "=== DONE ==="
