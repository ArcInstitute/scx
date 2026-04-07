#!/bin/bash
#SBATCH --job-name=phase3c_pcodec
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=benchmarks/logs/phase3c_pcodec_%j.log
#SBATCH --error=benchmarks/logs/phase3c_pcodec_%j.err

# Phase 3C — Pcodec benchmark: compression ratio and decode speed comparison
# against Zstd on D1 (pbmc3k), D3 (smartseq2), D4 (tabula_sapiens_100k).
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase3c_pcodec.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3C: Pcodec Benchmark ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Rebuild pyscx release with Pcodec support
echo "--- Rebuilding pyscx (release) ---"
cd pyscx && ../.venv/bin/maturin develop --release 2>&1 | tail -3 && cd ..
echo ""

DATASETS="pbmc3k smartseq2 tabula_sapiens_100k"
FORMATS="scx_pcodec scx_zstd scx_auto scx_scx1 scx_lz4 scx_none"
BENCHMARKS="compression write read_full"

echo "--- Formats: ${FORMATS} ---"
echo "--- Datasets: ${DATASETS} ---"
echo "--- Benchmarks: ${BENCHMARKS} ---"
echo ""

${PYTHON} benchmarks/comprehensive/scripts/run_all.py \
    --benchmarks ${BENCHMARKS} \
    --datasets ${DATASETS} \
    --formats ${FORMATS}

echo ""
echo "=== Phase 3C Pcodec benchmark complete ==="
echo "Date: $(date)"
echo ""
echo "Results in: benchmarks/comprehensive/results/raw/"
