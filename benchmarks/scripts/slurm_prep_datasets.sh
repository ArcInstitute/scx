#!/bin/bash
#SBATCH --job-name=prep_datasets
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=benchmarks/logs/prep_datasets_%j.log
#SBATCH --error=benchmarks/logs/prep_datasets_%j.err

# Prepare benchmark datasets D1-D6 (D7/D8 need separate high-memory jobs)
# Usage: sbatch benchmarks/scripts/slurm_prep_datasets.sh

set -euo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
SCRATCH="/scratch/ctc/nickyoungblut/scx"
DATASETS="${SCRATCH}/benchmarks/datasets"
PYTHON="${SCX_DIR}/.venv/bin/python"

cd "${SCX_DIR}"

echo "=== Dataset Preparation ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "Target: ${DATASETS}"
echo ""

mkdir -p "${DATASETS}"

# ------------------------------------------------------------------
# D1: pbmc3k (symlink — already available)
# ------------------------------------------------------------------
echo "--- D1: pbmc3k ---"
if [ -L "${DATASETS}/pbmc3k.h5ad" ] || [ -f "${DATASETS}/pbmc3k.h5ad" ]; then
    echo "  Already exists"
else
    ln -sf "${SCRATCH}/pbmc3k.h5ad" "${DATASETS}/pbmc3k.h5ad"
    echo "  Symlinked"
fi
${PYTHON} -c "
import anndata as ad
a = ad.read_h5ad('${DATASETS}/pbmc3k.h5ad', backed='r')
print(f'  Shape: {a.n_obs} x {a.n_vars}')
"

# ------------------------------------------------------------------
# D2: pbmc10k (download from 10x Genomics)
# ------------------------------------------------------------------
echo ""
echo "--- D2: pbmc10k ---"
${PYTHON} benchmarks/scripts/download_pbmc10k.py

# ------------------------------------------------------------------
# D3: smartseq2 (symlink — already available)
# ------------------------------------------------------------------
echo ""
echo "--- D3: smartseq2 ---"
if [ -L "${DATASETS}/smartseq2.h5ad" ] || [ -f "${DATASETS}/smartseq2.h5ad" ]; then
    echo "  Already exists"
else
    ln -sf "${SCRATCH}/smartseq2.h5ad" "${DATASETS}/smartseq2.h5ad"
    echo "  Symlinked"
fi
${PYTHON} -c "
import anndata as ad
a = ad.read_h5ad('${DATASETS}/smartseq2.h5ad', backed='r')
print(f'  Shape: {a.n_obs} x {a.n_vars}')
"

# ------------------------------------------------------------------
# D4: tabula_sapiens_100k (symlink — already available)
# ------------------------------------------------------------------
echo ""
echo "--- D4: tabula_sapiens_100k ---"
if [ -L "${DATASETS}/tabula_sapiens_100k.h5ad" ] || [ -f "${DATASETS}/tabula_sapiens_100k.h5ad" ]; then
    echo "  Already exists"
else
    ln -sf "${SCRATCH}/tabula_sapiens_100k.h5ad" "${DATASETS}/tabula_sapiens_100k.h5ad"
    echo "  Symlinked"
fi
${PYTHON} -c "
import anndata as ad
a = ad.read_h5ad('${DATASETS}/tabula_sapiens_100k.h5ad', backed='r')
print(f'  Shape: {a.n_obs} x {a.n_vars}')
"

# ------------------------------------------------------------------
# D5: census_500k (download from CELLxGENE Census)
# ------------------------------------------------------------------
echo ""
echo "--- D5: census_500k ---"
${PYTHON} benchmarks/scripts/download_census_500k.py

# ------------------------------------------------------------------
# D6: census_1m (symlink — already available)
# ------------------------------------------------------------------
echo ""
echo "--- D6: census_1m ---"
if [ -L "${DATASETS}/census_1m.h5ad" ] || [ -f "${DATASETS}/census_1m.h5ad" ]; then
    echo "  Already exists"
else
    ln -sf "${SCRATCH}/census_1m.h5ad" "${DATASETS}/census_1m.h5ad"
    echo "  Symlinked"
fi
${PYTHON} -c "
import anndata as ad
a = ad.read_h5ad('${DATASETS}/census_1m.h5ad', backed='r')
print(f'  Shape: {a.n_obs} x {a.n_vars}')
"

echo ""
echo "=== D1-D6 Preparation Complete ==="
echo ""
echo "Remaining (require high-memory SLURM jobs):"
echo "  D7: census_5m  -> sbatch benchmarks/scripts/slurm_build_census_5m.sh"
echo "  D8: census_10m -> sbatch benchmarks/scripts/slurm_build_census_10m.sh"
echo ""
ls -lh "${DATASETS}/"
