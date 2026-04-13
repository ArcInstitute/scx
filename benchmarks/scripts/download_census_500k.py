#!/usr/bin/env python3
"""Download census_500k: 500K blood cells from CELLxGENE Census.

Uses the CELLxGENE Census API to query human blood cells
and subsample to exactly 500,000 cells.
"""
import os
import sys
import numpy as np

import cellxgene_census
import anndata as ad
import scipy.sparse as sp

from bench_env import DATA_DIR

OUTPUT_DIR = str(DATA_DIR)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "census_500k.h5ad")
TARGET_CELLS = 500_000
SEED = 42


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists: {OUTPUT_PATH}")
        adata = ad.read_h5ad(OUTPUT_PATH, backed='r')
        print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Opening CELLxGENE Census (stable)...")
    census = cellxgene_census.open_soma(census_version="stable")

    # Query blood tissue cells
    print("Querying blood cells from Census...")
    print("  Filter: tissue_general == 'blood', organism = Homo sapiens")

    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_value_filter="tissue_general == 'blood'",
    )

    print(f"  Retrieved: {adata.n_obs} cells x {adata.n_vars} genes")

    if adata.n_obs < TARGET_CELLS:
        print(f"  WARNING: Only {adata.n_obs} blood cells available (need {TARGET_CELLS})")
        print(f"  Using all available cells")
    else:
        # Subsample to target
        print(f"  Subsampling to {TARGET_CELLS} cells (seed={SEED})...")
        rng = np.random.default_rng(SEED)
        indices = rng.choice(adata.n_obs, size=TARGET_CELLS, replace=False)
        indices.sort()  # Keep sorted for efficient slicing
        adata = adata[indices].copy()
        print(f"  After subsampling: {adata.n_obs} cells")

    # Ensure CSR format
    if not sp.issparse(adata.X) or adata.X.format != 'csr':
        print("  Converting to CSR...")
        adata.X = sp.csr_matrix(adata.X)

    print(f"  NNZ: {adata.X.nnz:,}")
    sparsity = 1 - adata.X.nnz / (adata.n_obs * adata.n_vars)
    print(f"  Sparsity: {sparsity:.4f}")

    # Write
    print(f"Writing h5ad to: {OUTPUT_PATH}")
    adata.write_h5ad(OUTPUT_PATH)
    print(f"  h5ad size: {os.path.getsize(OUTPUT_PATH) / 1e9:.2f} GB")

    census.close()
    print("Done!")


if __name__ == "__main__":
    main()
