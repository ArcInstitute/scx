#!/usr/bin/env bash
# Download benchmark datasets for SCX benchmarks.
# Usage: bash download_datasets.sh [DATA_DIR]
#
# Data directory defaults to /scratch/ctc/nickyoungblut/scx/
# Each dataset is downloaded only if not already present (idempotent).

set -euo pipefail

DATA_DIR="${1:-/scratch/ctc/nickyoungblut/scx}"
VENV="$(cd "$(dirname "$0")/../.." && pwd)/.venv"
PYTHON="${VENV}/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python venv not found at $VENV"
    echo "Run: uv venv .venv && uv pip install scanpy cellxgene-census anndata"
    exit 1
fi

mkdir -p "$DATA_DIR"
echo "=== SCX Benchmark Dataset Downloader ==="
echo "Data directory: $DATA_DIR"
echo ""

# --------------------------------------------------------------------------
# 1. PBMC 3K (~30 MB h5ad)
# --------------------------------------------------------------------------
PBMC_H5AD="$DATA_DIR/pbmc3k.h5ad"
if [ -f "$PBMC_H5AD" ]; then
    echo "[SKIP] PBMC 3K already exists: $PBMC_H5AD"
else
    echo "[DOWNLOAD] PBMC 3K filtered feature-barcode matrix..."
    PBMC_H5="$DATA_DIR/pbmc3k_filtered_gene_bc_matrices.h5"
    curl -sL -o "$PBMC_H5" \
        "https://cf.10xgenomics.com/samples/cell-exp/1.1.0/pbmc3k/pbmc3k_filtered_gene_bc_matrices_h5.h5"
    echo "[CONVERT] Converting PBMC 3K to h5ad via scanpy..."
    "$PYTHON" -c "
import scanpy as sc
adata = sc.read_10x_h5('$PBMC_H5')
adata.var_names_make_unique()
adata.write_h5ad('$PBMC_H5AD')
print(f'  Written: {adata.n_obs} cells x {adata.n_vars} genes')
"
    rm -f "$PBMC_H5"
    echo "[DONE] PBMC 3K: $PBMC_H5AD"
fi

# --------------------------------------------------------------------------
# 2. Tabula Sapiens subset (~2 GB h5ad, 100K cells)
# --------------------------------------------------------------------------
TABULA_H5AD="$DATA_DIR/tabula_sapiens_100k.h5ad"
if [ -f "$TABULA_H5AD" ]; then
    echo "[SKIP] Tabula Sapiens subset already exists: $TABULA_H5AD"
else
    echo "[DOWNLOAD] Tabula Sapiens subset (100K cells) from CELLxGENE..."
    "$PYTHON" -c "
import cellxgene_census
import anndata
import numpy as np

print('  Opening CELLxGENE Census...')
with cellxgene_census.open_soma() as census:
    # Query Tabula Sapiens data (organism: Homo sapiens, collection: Tabula Sapiens)
    adata = cellxgene_census.get_anndata(
        census,
        organism='Homo sapiens',
        obs_value_filter=\"dataset_id == '53d208b0-2cfd-4366-9866-c3c6114081bc'\",
    )
    # Subsample to 100K cells if larger
    if adata.n_obs > 100_000:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, 100_000, replace=False)
        idx.sort()
        adata = adata[idx].copy()
    adata.write_h5ad('$TABULA_H5AD')
    print(f'  Written: {adata.n_obs} cells x {adata.n_vars} genes')
" || echo "[WARN] Tabula Sapiens download failed (requires cellxgene-census). Skipping."
    if [ -f "$TABULA_H5AD" ]; then
        echo "[DONE] Tabula Sapiens: $TABULA_H5AD"
    fi
fi

# --------------------------------------------------------------------------
# 3. CELLxGENE Census subset (~20 GB h5ad, 1M cells)
# --------------------------------------------------------------------------
CENSUS_H5AD="$DATA_DIR/census_1m.h5ad"
if [ -f "$CENSUS_H5AD" ]; then
    echo "[SKIP] Census 1M subset already exists: $CENSUS_H5AD"
else
    echo "[DOWNLOAD] CELLxGENE Census 1M cell subset..."
    "$PYTHON" -c "
import cellxgene_census
import anndata
import numpy as np

print('  Opening CELLxGENE Census...')
with cellxgene_census.open_soma() as census:
    # Get a broad human dataset, then subsample to 1M cells
    adata = cellxgene_census.get_anndata(
        census,
        organism='Homo sapiens',
        obs_value_filter=\"tissue_general == 'blood'\",
    )
    if adata.n_obs > 1_000_000:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, 1_000_000, replace=False)
        idx.sort()
        adata = adata[idx].copy()
    adata.write_h5ad('$CENSUS_H5AD')
    print(f'  Written: {adata.n_obs} cells x {adata.n_vars} genes')
" || echo "[WARN] Census 1M download failed (requires cellxgene-census). Skipping."
    if [ -f "$CENSUS_H5AD" ]; then
        echo "[DONE] Census 1M: $CENSUS_H5AD"
    fi
fi

# --------------------------------------------------------------------------
# 4. Smart-seq2 dataset (non-UMI protocol)
# --------------------------------------------------------------------------
SMARTSEQ_H5AD="$DATA_DIR/smartseq2.h5ad"
if [ -f "$SMARTSEQ_H5AD" ]; then
    echo "[SKIP] Smart-seq2 already exists: $SMARTSEQ_H5AD"
else
    echo "[DOWNLOAD] Smart-seq2 dataset from CELLxGENE..."
    "$PYTHON" -c "
import cellxgene_census
import anndata
import numpy as np

print('  Opening CELLxGENE Census...')
with cellxgene_census.open_soma() as census:
    # Smart-seq2 data from CELLxGENE
    adata = cellxgene_census.get_anndata(
        census,
        organism='Homo sapiens',
        obs_value_filter=\"assay == 'Smart-seq2'\",
    )
    # Subsample if very large
    if adata.n_obs > 50_000:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, 50_000, replace=False)
        idx.sort()
        adata = adata[idx].copy()
    adata.write_h5ad('$SMARTSEQ_H5AD')
    print(f'  Written: {adata.n_obs} cells x {adata.n_vars} genes')
" || echo "[WARN] Smart-seq2 download failed (requires cellxgene-census). Skipping."
    if [ -f "$SMARTSEQ_H5AD" ]; then
        echo "[DONE] Smart-seq2: $SMARTSEQ_H5AD"
    fi
fi

# --------------------------------------------------------------------------
# 5. CELLxGENE Census 10M blood subset (~15 GB h5ad)
# --------------------------------------------------------------------------
CENSUS_10M_H5AD="$DATA_DIR/census_10m_blood.h5ad"
if [ -f "$CENSUS_10M_H5AD" ]; then
    echo "[SKIP] Census 10M blood already exists: $CENSUS_10M_H5AD"
else
    echo "[DOWNLOAD] CELLxGENE Census 10M blood subset..."
    "$PYTHON" -c "
import cellxgene_census
import anndata
import numpy as np

print('  Opening CELLxGENE Census...')
with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census,
        organism='Homo sapiens',
        obs_value_filter=\"tissue_general == 'blood'\",
    )
    print(f'  Raw query: {adata.n_obs} cells x {adata.n_vars} genes')
    if adata.n_obs > 10_000_000:
        np.random.seed(42)
        idx = np.random.choice(adata.n_obs, 10_000_000, replace=False)
        idx.sort()
        adata = adata[idx].copy()
    adata.write_h5ad('$CENSUS_10M_H5AD')
    print(f'  Written: {adata.n_obs} cells x {adata.n_vars} genes')
" || echo "[WARN] Census 10M download failed (requires cellxgene-census). Skipping."
    if [ -f "$CENSUS_10M_H5AD" ]; then
        echo "[DONE] Census 10M blood: $CENSUS_10M_H5AD"
    fi
fi

# --------------------------------------------------------------------------
# 6. Convert all h5ad datasets to .scx format
# --------------------------------------------------------------------------
echo ""
echo "=== Converting h5ad → SCX ==="
for h5ad in "$DATA_DIR"/*.h5ad; do
    if [ ! -f "$h5ad" ]; then
        continue
    fi
    scx="${h5ad%.h5ad}.scx"
    if [ -f "$scx" ]; then
        echo "[SKIP] SCX already exists: $(basename "$scx")"
    else
        echo "[CONVERT] $(basename "$h5ad") → $(basename "$scx")..."
        "$PYTHON" -c "
import sys
sys.path.insert(0, '$(cd "$(dirname "$0")/../.." && pwd)/pyscx')
import pyscx
import anndata

adata = anndata.read_h5ad('$h5ad')
pyscx.from_anndata(adata, '$scx')
import os
size_mb = os.path.getsize('$scx') / 1e6
print(f'  Written: {adata.n_obs} cells x {adata.n_vars} genes ({size_mb:.1f} MB)')
" || echo "[WARN] Conversion failed for $(basename "$h5ad"). Skipping."
    fi
done

echo ""
echo "=== Download & Conversion Summary ==="
echo "Data directory: $DATA_DIR"
echo ""
echo "h5ad files:"
for f in "$DATA_DIR"/*.h5ad; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  $size  $(basename "$f")"
    fi
done
echo ""
echo "SCX files:"
for f in "$DATA_DIR"/*.scx; do
    if [ -f "$f" ]; then
        size=$(du -h "$f" | cut -f1)
        echo "  $size  $(basename "$f")"
    fi
done
