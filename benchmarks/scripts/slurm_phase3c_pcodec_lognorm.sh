#!/bin/bash
#SBATCH --job-name=phase3c_lognorm
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=benchmarks/logs/phase3c_lognorm_%j.log
#SBATCH --error=benchmarks/logs/phase3c_lognorm_%j.err

# Phase 3C — Pcodec benchmark on log-normalized float data.
# Step 1: Create log-normalized h5ad files (normalize_total + log1p).
# Step 2: Run compression/write/read benchmarks with Pcodec vs other codecs.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase3c_pcodec_lognorm.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 3C: Pcodec Benchmark on Log-Normalized Data ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Step 1: Rebuild pyscx release
echo "--- Rebuilding pyscx (release) ---"
cd pyscx && ../.venv/bin/maturin develop --release 2>&1 | tail -3 && cd ..
echo ""

# Step 2: Create log-normalized datasets
echo "--- Creating log-normalized datasets ---"
${PYTHON} benchmarks/scripts/prep_lognorm_datasets.py
echo ""

# Step 3: Run benchmarks on log-normalized datasets
DATASETS="pbmc3k_lognorm smartseq2_lognorm tabula_sapiens_100k_lognorm"
FORMATS="scx_pcodec scx_zstd scx_auto scx_lz4 scx_none"
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
echo "=== Phase 3C Pcodec log-normalized benchmark complete ==="
echo "Date: $(date)"
echo "Results in: benchmarks/comprehensive/results/raw/"
