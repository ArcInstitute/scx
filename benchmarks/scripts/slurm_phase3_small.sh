#!/bin/bash
#SBATCH --job-name=phase3_small
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=benchmarks/logs/phase3_small_%j.log
#SBATCH --error=benchmarks/logs/phase3_small_%j.err

# Phase 3 — Core benchmarks (compression, write, read_full, read_selective,
# parallel_scaling, memory) on small/medium datasets (D1-D4).
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase3_small.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3: Core Benchmarks — Small/Medium Datasets (D1-D4) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

DATASETS="pbmc3k,pbmc10k,smartseq2,tabula_sapiens_100k"
BENCHMARKS="compression write read_full read_selective parallel_scaling memory"

echo "--- Running benchmarks: ${BENCHMARKS} ---"
echo "--- Datasets: ${DATASETS} ---"
echo ""

${PYTHON} benchmarks/comprehensive/scripts/run_all.py \
    --benchmarks ${BENCHMARKS} \
    --datasets ${DATASETS//,/ }

echo ""
echo "=== Phase 3 (small) complete ==="
echo "Date: $(date)"
echo ""
echo "Results in: benchmarks/comprehensive/results/raw/"
echo ""
echo "For larger datasets (D5-D8), submit:"
echo "  sbatch benchmarks/scripts/slurm_phase3_large.sh"
