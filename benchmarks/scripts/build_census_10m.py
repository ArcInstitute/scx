#!/usr/bin/env python3
"""Build census_10m.h5ad from existing census chunk h5ad files.

Concatenates all 99 chunks (99 × 100K ≈ 9.9M cells) from
/scratch/ctc/nickyoungblut/scx/_census_chunk_*.h5ad, then pads
with resampled cells to reach exactly 10M cells.

ALTERNATIVE: If 9.9M is close enough, we just use the 99 chunks as-is,
since the benchmark spec says "10M+" and labels it "~10M".

This script requires massive memory (~500+ GB) and should be
run on a high-memory SLURM node.
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
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "census_10m.h5ad")
N_CHUNKS = 99  # All available chunks


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
    missing = [f for f in chunk_files if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing {len(missing)} chunk files:")
        for f in missing[:5]:
            print(f"  {f}")
        sys.exit(1)

    print(f"Building census_10m from {N_CHUNKS} chunks...")
    print(f"  Chunk dir: {CHUNKS_DIR}")
    print(f"  Output: {OUTPUT_PATH}")

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
        if (i + 1) % 10 == 0:
            print(f"    Cumulative: {total_cells:,} cells", flush=True)

    print(f"\nConcatenating {len(adatas)} chunks ({total_cells:,} total cells)...")
    combined = ad.concat(adatas, join='outer', merge='first')
    combined.obs_names_make_unique()
    del adatas
    gc.collect()

    print(f"  Combined shape: {combined.n_obs} x {combined.n_vars}")

    # If we need to pad to exactly 10M, resample a subset of cells
    target = 10_000_000
    if combined.n_obs < target:
        deficit = target - combined.n_obs
        print(f"  Padding {deficit:,} cells by resampling to reach {target:,}...")
        rng = np.random.default_rng(42)
        resample_idx = rng.choice(combined.n_obs, size=deficit, replace=True)
        pad = combined[resample_idx].copy()
        combined = ad.concat([combined, pad], join='outer', merge='first')
        combined.obs_names_make_unique()
        del pad
        gc.collect()
        print(f"  After padding: {combined.n_obs} x {combined.n_vars}")
    elif combined.n_obs > target:
        print(f"  Truncating to {target:,} cells...")
        combined = combined[:target].copy()
        print(f"  After truncation: {combined.n_obs} x {combined.n_vars}")

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
