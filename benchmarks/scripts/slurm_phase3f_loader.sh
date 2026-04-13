#!/bin/bash
#SBATCH --job-name=phase3f_loader
#SBATCH --partition=cpu_preemptible
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=benchmarks/logs/phase3f_loader_%j.log
#SBATCH --error=benchmarks/logs/phase3f_loader_%j.err

# Phase 3F — Training loader prefetch scheduling benchmark.
# Measures training throughput (batches/sec) with offset-sorted shard access,
# MADV_WILLNEED prefetch, and coalesced madvise hints.
#
# Runs on D1 (pbmc3k), D3 (smartseq2), D4 (tabula_sapiens_100k), D6 (census_1m).
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase3f_loader.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs benchmarks/results

echo "=== Phase 3F: Training Loader Prefetch Scheduling Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Rebuild pyscx in release mode to pick up prefetch scheduling changes
echo "--- Building pyscx (release) ---"
cd pyscx && ../.venv/bin/maturin develop --release && cd ..
echo ""

# Run training loader benchmark on each dataset
DATASETS=("pbmc3k" "smartseq2" "tabula_sapiens_100k" "census_1m")

for DS in "${DATASETS[@]}"; do
    echo "=== Dataset: ${DS} ==="
    echo "--- Throughput benchmark ---"
    ${PYTHON} benchmarks/scripts/benchmark_loader.py \
        --dataset "${DS}" \
        --output "benchmarks/results/phase3f_loader_${DS}.json" \
        || echo "WARN: ${DS} benchmark failed"
    echo ""
done

echo ""
echo "=== Phase 3F loader benchmark complete ==="
echo "Date: $(date)"
echo ""
echo "Results in: benchmarks/results/phase3f_loader_*.json"
