#!/usr/bin/env python3
"""Build census_500k.h5ad from existing census chunk files.

Takes the first 5 chunks (5 × 100K = 500K cells).
All chunks are already filtered to blood tissue from CELLxGENE Census.
"""
import os
import sys
import gc
import time

import anndata as ad
import scipy.sparse as sp
import numpy as np

CHUNKS_DIR = "/scratch/ctc/nickyoungblut/scx"
OUTPUT_DIR = "/scratch/ctc/nickyoungblut/scx/benchmarks/datasets"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "census_500k.h5ad")
N_CHUNKS = 5  # 5 × 100K = 500K cells


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists: {OUTPUT_PATH}")
        adata = ad.read_h5ad(OUTPUT_PATH, backed='r')
        print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    chunk_files = [
        os.path.join(CHUNKS_DIR, f"_census_chunk_{i:03d}.h5ad")
        for i in range(1, N_CHUNKS + 1)
    ]

    for f in chunk_files:
        if not os.path.exists(f):
            print(f"ERROR: Missing chunk file: {f}")
            sys.exit(1)

    print(f"Building census_500k from {N_CHUNKS} chunks...")
    t0 = time.time()

    adatas = []
    total_cells = 0
    for i, chunk_file in enumerate(chunk_files):
        print(f"  Reading chunk {i+1}/{N_CHUNKS}: {os.path.basename(chunk_file)}...")
        adata = ad.read_h5ad(chunk_file)
        if sp.issparse(adata.X) and adata.X.format != 'csr':
            adata.X = sp.csr_matrix(adata.X)
        total_cells += adata.n_obs
        adatas.append(adata)
        print(f"    {adata.n_obs} cells (cumulative: {total_cells:,})")

    print(f"\nConcatenating {len(adatas)} chunks ({total_cells:,} cells)...")
    combined = ad.concat(adatas, join='outer', merge='first')
    combined.obs_names_make_unique()
    del adatas
    gc.collect()

    print(f"  Shape: {combined.n_obs} x {combined.n_vars}")

    if sp.issparse(combined.X) and combined.X.format != 'csr':
        combined.X = sp.csr_matrix(combined.X)

    print(f"  NNZ: {combined.X.nnz:,}")
    sparsity = 1 - combined.X.nnz / (combined.n_obs * combined.n_vars)
    print(f"  Sparsity: {sparsity:.4f}")

    print(f"Writing to: {OUTPUT_PATH}")
    combined.write_h5ad(OUTPUT_PATH)
    elapsed = time.time() - t0
    print(f"  h5ad size: {os.path.getsize(OUTPUT_PATH) / 1e9:.2f} GB")
    print(f"  Time: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
