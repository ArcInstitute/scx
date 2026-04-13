#!/bin/bash
#SBATCH --job-name=phase1_small
#SBATCH --partition=cpu_preemptible
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=benchmarks/logs/phase1_small_%j.log
#SBATCH --error=benchmarks/logs/phase1_small_%j.err

# Phase 1 — Dataset verification, metadata, and compressed h5ad variants
# for small/medium datasets (D1-D6, up to ~11 GB).
#
# Usage: cd /home/nickyoungblut/dev/rust/scx && sbatch benchmarks/scripts/slurm_phase1_small.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"
mkdir -p benchmarks/logs

echo "=== Phase 1: Small/Medium Datasets (D1-D6) ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo ""

SMALL_DATASETS="pbmc3k,pbmc10k,smartseq2,tabula_sapiens_100k,census_500k,census_1m"

# Step 1: Verify D1-D6 and record metadata
echo "--- Step 1: Verify datasets & record metadata (D1-D6) ---"
${PYTHON} benchmarks/scripts/verify_datasets.py --datasets "${SMALL_DATASETS}"
echo ""

# Step 2: Generate compressed h5ad variants (D1-D6 only)
echo "--- Step 2: Generate compressed h5ad variants (D1-D6) ---"
${PYTHON} benchmarks/scripts/generate_compressed_h5ad.py \
    --datasets "${SMALL_DATASETS}" \
    --compressions gzip,lzf
echo ""

echo "=== Phase 1 (small) complete ==="
echo "Date: $(date)"
echo ""
echo "For large datasets (D7, D8), submit:"
echo "  sbatch benchmarks/scripts/slurm_phase1_large.sh"
