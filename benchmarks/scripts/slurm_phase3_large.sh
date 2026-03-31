#!/bin/bash
#SBATCH --job-name=phase3_large
#SBATCH --partition=cpu_high_mem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/logs/phase3_large_%j.log
#SBATCH --error=benchmarks/logs/phase3_large_%j.err

# Phase 3 — Core benchmarks on large datasets (D5-D8).
# Requires high-memory partition for census_5m and census_10m.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase3_large.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3: Core Benchmarks — Large Datasets (D5-D8) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

DATASETS="census_500k,census_1m,census_5m,census_10m"
BENCHMARKS="compression write read_full read_selective parallel_scaling memory"

echo "--- Running benchmarks: ${BENCHMARKS} ---"
echo "--- Datasets: ${DATASETS} ---"
echo ""

${PYTHON} benchmarks/comprehensive/scripts/run_all.py \
    --benchmarks ${BENCHMARKS} \
    --datasets ${DATASETS//,/ }

echo ""
echo "=== Phase 3 (large) complete ==="
echo "Date: $(date)"
echo ""
echo "Results in: benchmarks/comprehensive/results/raw/"
