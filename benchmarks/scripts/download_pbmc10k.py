#!/usr/bin/env python3
"""Download the 10x Genomics PBMC 10k dataset and convert to h5ad.

Source: 10x Genomics public datasets (10k PBMCs, 10x v3 chemistry).
Downloads the filtered feature-barcode matrix in HDF5 format,
reads with scanpy, and saves as h5ad.
"""
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import scanpy as sc
import anndata as ad

# Make the comprehensive/bench_env shim resolvable regardless of CWD.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "comprehensive"))
from bench_env import DATA_DIR  # noqa: E402

OUTPUT_DIR = str(DATA_DIR)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pbmc10k.h5ad")

# 10x Genomics public dataset: 10k PBMCs from a Healthy Donor (v3 chemistry)
# This is the filtered feature-barcode matrix in HDF5 format
URL = "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.h5"

def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists: {OUTPUT_PATH}")
        adata = ad.read_h5ad(OUTPUT_PATH)
        print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download to temp file. cf.10xgenomics.com returns HTTP 403 for the
    # default `Python-urllib/x.y` User-Agent; supply a UA so the fetch
    # succeeds across all 10x dataset CDNs.
    h5_path = os.path.join(OUTPUT_DIR, "pbmc_10k_v3_filtered_feature_bc_matrix.h5")
    if not os.path.exists(h5_path):
        print(f"Downloading PBMC 10k dataset from 10x Genomics...")
        print(f"  URL: {URL}")
        req = urllib.request.Request(
            URL,
            headers={"User-Agent": "scx-benchmarks/0.4.0 (research use)"},
        )
        with urllib.request.urlopen(req) as resp, open(h5_path, "wb") as f:
            shutil.copyfileobj(resp, f)
        print(f"  Downloaded to: {h5_path}")
        print(f"  Size: {os.path.getsize(h5_path) / 1e6:.1f} MB")
    else:
        print(f"Raw h5 already exists: {h5_path}")

    # Read with scanpy
    print("Reading with scanpy.read_10x_h5()...")
    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()

    print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
    print(f"  X dtype: {adata.X.dtype}, format: {adata.X.format}")
    print(f"  NNZ: {adata.X.nnz:,}")
    sparsity = 1 - adata.X.nnz / (adata.n_obs * adata.n_vars)
    print(f"  Sparsity: {sparsity:.4f}")

    # Ensure CSR format
    import scipy.sparse as sp
    if not sp.issparse(adata.X) or adata.X.format != 'csr':
        adata.X = sp.csr_matrix(adata.X)
        print("  Converted to CSR")

    # Write h5ad
    print(f"Writing h5ad to: {OUTPUT_PATH}")
    adata.write_h5ad(OUTPUT_PATH)
    print(f"  h5ad size: {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")

    # Clean up raw h5
    os.remove(h5_path)
    print("Done!")


if __name__ == "__main__":
    main()
