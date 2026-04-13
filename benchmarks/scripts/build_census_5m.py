#!/usr/bin/env python3
"""Build census_5m.h5ad from existing census chunk h5ad files.

Concatenates the first 50 chunks (50 × 100K = 5M cells) from
WORK_DIR/_census_chunk_*.h5ad (see bench_env.py)

This script requires significant memory (~200+ GB) and should be
run on a high-memory SLURM node.
"""
import os
import sys
import gc
import time

import anndata as ad
import scipy.sparse as sp
import numpy as np

from bench_env import WORK_DIR, DATA_DIR

CHUNKS_DIR = str(WORK_DIR)
OUTPUT_DIR = str(DATA_DIR)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "census_5m.h5ad")
N_CHUNKS = 50  # 50 chunks × 100K cells = 5M cells


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists: {OUTPUT_PATH}")
        adata = ad.read_h5ad(OUTPUT_PATH, backed='r')
        print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find chunk files
    chunk_files = sorted(
        [os.path.join(CHUNKS_DIR, f"_census_chunk_{i:03d}.h5ad") for i in range(1, N_CHUNKS + 1)]
    )

    # Verify all chunks exist
    for f in chunk_files:
        if not os.path.exists(f):
            print(f"ERROR: Missing chunk file: {f}")
            sys.exit(1)

    print(f"Building census_5m from {N_CHUNKS} chunks...")
    print(f"  Chunk dir: {CHUNKS_DIR}")
    print(f"  Output: {OUTPUT_PATH}")

    # Use on_disk concatenation to avoid loading everything into memory
    # Read chunks one at a time and concatenate
    t0 = time.time()

    adatas = []
    total_cells = 0
    for i, chunk_file in enumerate(chunk_files):
        print(f"  Reading chunk {i+1}/{N_CHUNKS}: {os.path.basename(chunk_file)}...", flush=True)
        adata = ad.read_h5ad(chunk_file)
        # Ensure CSR
        if sp.issparse(adata.X) and adata.X.format != 'csr':
            adata.X = sp.csr_matrix(adata.X)
        total_cells += adata.n_obs
        adatas.append(adata)
        print(f"    {adata.n_obs} cells (cumulative: {total_cells:,})", flush=True)

    print(f"\nConcatenating {len(adatas)} chunks ({total_cells:,} total cells)...")
    combined = ad.concat(adatas, join='outer', merge='first')
    combined.obs_names_make_unique()
    del adatas
    gc.collect()

    print(f"  Combined shape: {combined.n_obs} x {combined.n_vars}")
    print(f"  X format: {combined.X.format if sp.issparse(combined.X) else type(combined.X)}")

    # Ensure CSR
    if sp.issparse(combined.X) and combined.X.format != 'csr':
        combined.X = sp.csr_matrix(combined.X)

    print(f"  NNZ: {combined.X.nnz:,}")
    sparsity = 1 - combined.X.nnz / (combined.n_obs * combined.n_vars)
    print(f"  Sparsity: {sparsity:.4f}")

    print(f"Writing h5ad to: {OUTPUT_PATH}")
    combined.write_h5ad(OUTPUT_PATH)
    elapsed = time.time() - t0
    print(f"  h5ad size: {os.path.getsize(OUTPUT_PATH) / 1e9:.2f} GB")
    print(f"  Total time: {elapsed:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
