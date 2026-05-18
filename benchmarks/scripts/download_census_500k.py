#!/usr/bin/env python3
"""Download census_500k: 500K blood cells from CELLxGENE Census.

Uses the CELLxGENE Census API to query human blood cells
and subsample to exactly 500,000 cells.

Memory note: ``cellxgene_census.get_anndata(..., obs_value_filter=...)``
materialises the full filtered slab before any subsampling. "Blood"
tissue spans several million cells and the dense materialise spikes
peak RSS past 500 GB — overwhelming a `standard` partition node.

Instead, this script pulls the obs metadata first (cheap — just
soma_joinids), randomly subsamples to ``TARGET_CELLS``, and passes
those joinids through ``obs_coords=`` so ``get_anndata`` only
materialises the chosen rows.
"""
import os
import sys
from pathlib import Path

import numpy as np

import cellxgene_census
import anndata as ad
import scipy.sparse as sp

# Make the comprehensive/bench_env shim resolvable regardless of CWD.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "comprehensive"))
from bench_env import DATA_DIR  # noqa: E402

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

    print("Opening CELLxGENE Census (stable)...")
    census = cellxgene_census.open_soma(census_version="stable")

    # Step 1 — pull soma_joinids only (cheap; obs metadata fits in
    # tens of MB). This avoids materialising the full multi-million
    # cell blood slice that the prior implementation went through
    # before subsampling — Lambda nodes OOM'd at 192 GB on that path.
    print("Querying soma_joinids for blood cells (obs-only)...")
    print("  Filter: tissue_general == 'blood', organism = Homo sapiens")
    obs_df = cellxgene_census.get_obs(
        census,
        organism="Homo sapiens",
        value_filter="tissue_general == 'blood'",
        column_names=["soma_joinid"],
    )
    joinids = obs_df["soma_joinid"].to_numpy()
    print(f"  Candidate cells: {len(joinids):,}")

    # Step 2 — pre-sample at the obs layer, before materialising X.
    if len(joinids) < TARGET_CELLS:
        print(f"  WARNING: Only {len(joinids)} blood cells available "
              f"(need {TARGET_CELLS})")
        sampled = np.sort(joinids)
    else:
        rng = np.random.default_rng(SEED)
        sampled = np.sort(rng.choice(joinids, size=TARGET_CELLS, replace=False))
    print(f"  Materialising X for {len(sampled):,} sampled cells...")

    # Step 3 — materialise only the sampled rows. obs_coords accepts
    # an integer numpy array of soma_joinids; get_anndata's underlying
    # SOMA query restricts to that exact set.
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_coords=sampled.astype(np.int64),
    )
    print(f"  Materialised: {adata.n_obs} cells x {adata.n_vars} genes")

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
