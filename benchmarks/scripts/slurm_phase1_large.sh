#!/bin/bash
#SBATCH --job-name=phase1_large
#SBATCH --partition=cpu_high_mem
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=benchmarks/logs/phase1_large_%j.log
#SBATCH --error=benchmarks/logs/phase1_large_%j.err

# Phase 1 — Compressed h5ad variants for large datasets (D7: 86 GB, D8: 176 GB).
# Requires high-memory partition since h5py copy still needs substantial memory for
# HDF5 metadata and buffering.
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase1_large.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

# Load environment from .env if present
if [ -f "${SCX_DIR}/.env" ]; then
    set -a; source "${SCX_DIR}/.env"; set +a
fi
if [ -z "${SCX_WORK_DIR:-}" ]; then
    echo "ERROR: SCX_WORK_DIR is not set. Define it in ${SCX_DIR}/.env or export it."
    exit 1
fi
DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 1: Large Datasets (D7-D8) Compressed Variants ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

LARGE_DATASETS="census_5m,census_10m"

# Step 1: Verify D7 and D8 datasets and record metadata
echo "--- Step 1: Verify datasets & record metadata (D7-D8) ---"
${PYTHON} benchmarks/scripts/verify_datasets.py --datasets "${LARGE_DATASETS}"
echo ""

# Step 2: Generate compressed h5ad variants for D7 and D8
echo "--- Step 2: Generate compressed h5ad variants (D7-D8) ---"
${PYTHON} benchmarks/scripts/generate_compressed_h5ad.py \
    --datasets "${LARGE_DATASETS}" \
    --compressions gzip,lzf

echo ""
echo "=== Phase 1 (large) complete ==="
echo "Date: $(date)"

# List all files in the datasets directory for verification
echo ""
echo "--- All dataset files ---"
ls -lhS "${DATA_DIR}/"
