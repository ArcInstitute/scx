#!/bin/bash
#SBATCH --job-name=fused_bench
#SBATCH --partition=cpu_preemptible
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/logs/fused_bench_%j.log

set -euo pipefail

echo "=== System info ==="
hostname
lscpu | head -20
echo ""

echo "=== Building release ==="
cargo build -p scx-engine --tests --release 2>&1
echo ""

echo "=== Running fused ops benchmarks ==="
cargo test -p scx-engine bench_query_fused --release -- --nocapture --ignored 2>&1
echo ""

echo "=== DONE ==="
