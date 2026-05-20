#!/usr/bin/env python3
"""Download census_1m: 1M blood cells from CELLxGENE Census.

ID-first subsample to keep peak memory bounded by the *target* set,
not the entire blood-cell universe. The naive `download_census_500k.py`
pattern (`get_anndata(filter=...)` then numpy subsample) materialises
every matching cell into RAM first — that grew past 512 GB on this
Census version and OOM'd.

The flow here:
  1. Open the `obs` SOMA dataframe and read only `soma_joinid` for
     cells matching the filter. This is lightweight (~tens of MB
     for ~50 M joinids).
  2. Subsample joinids deterministically (seed=42).
  3. Call `get_anndata(obs_coords=joinids)` — Census materialises
     only the selected 1 M cells, not the full universe.
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
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "census_1m.h5ad")
TARGET_CELLS = 1_000_000
SEED = 42
OBS_FILTER = "tissue_general == 'blood'"


def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"Output already exists: {OUTPUT_PATH}")
        adata = ad.read_h5ad(OUTPUT_PATH, backed='r')
        print(f"  Shape: {adata.n_obs} x {adata.n_vars}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Opening CELLxGENE Census (stable)...")
    census = cellxgene_census.open_soma(census_version="stable")

    # Step 1: lightweight joinid pull. Walking `obs` for matching
    # `soma_joinid` reads scalar 64-bit IDs only — no expression
    # data, no metadata columns beyond what the filter needs.
    print(f"Pulling joinids for filter: {OBS_FILTER}")
    obs_iter = (
        census["census_data"]["homo_sapiens"]
        .obs.read(value_filter=OBS_FILTER, column_names=["soma_joinid"])
    )
    joinids_chunks = [tbl["soma_joinid"].to_numpy() for tbl in obs_iter]
    joinids = np.concatenate(joinids_chunks) if joinids_chunks else np.array([], dtype=np.int64)
    print(f"  Universe: {joinids.size:,} cells")

    if joinids.size < TARGET_CELLS:
        print(f"  WARNING: Only {joinids.size} cells available (need {TARGET_CELLS})")
        sel = joinids
    else:
        rng = np.random.default_rng(SEED)
        sel_idx = rng.choice(joinids.size, size=TARGET_CELLS, replace=False)
        sel_idx.sort()
        sel = joinids[sel_idx]
        print(f"  Subsampled to {sel.size:,} joinids (seed={SEED})")

    # Step 2: materialise only the selected cells.
    print("Fetching AnnData for selected cells...")
    adata = cellxgene_census.get_anndata(
        census,
        organism="Homo sapiens",
        obs_coords=sel,
    )
    print(f"  Got: {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    if not sp.issparse(adata.X) or adata.X.format != "csr":
        print("  Converting to CSR...")
        adata.X = sp.csr_matrix(adata.X)

    print(f"  NNZ: {adata.X.nnz:,}")
    sparsity = 1 - adata.X.nnz / (adata.n_obs * adata.n_vars)
    print(f"  Sparsity: {sparsity:.4f}")

    print(f"Writing h5ad to: {OUTPUT_PATH}")
    adata.write_h5ad(OUTPUT_PATH)
    print(f"  h5ad size: {os.path.getsize(OUTPUT_PATH) / 1e9:.2f} GB")

    census.close()
    print("Done!")


if __name__ == "__main__":
    main()
