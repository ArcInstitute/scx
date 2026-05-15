#!/usr/bin/env python
"""Emit a small synthetic h5ad fixture for the streaming-conversion
benchmark smoke test (Phase 10 of STREAMING-CONVERSION.md). The real
canonical fixtures (`census_1m` / `census_5m` / `census_10m`) live
elsewhere; this exists so the SLURM smoke job can prove the harness
end-to-end on a tiny in-memory matrix without bringing those onto
the local node.

Output is a CSR h5ad with uint8 integer counts, no obsm / varm / uns /
layers — that's the simplest shape that exercises both the
streaming pipeline and the materialising path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--n-obs", type=int, default=50_000)
    p.add_argument("--n-vars", type=int, default=5_000)
    p.add_argument("--density", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import numpy as np
    import scipy.sparse as sp
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(args.seed)
    n_obs = int(args.n_obs)
    n_vars = int(args.n_vars)
    density = float(args.density)
    nnz = int(n_obs * n_vars * density)

    # Sparse uint8 counts. Build COO then convert to CSR.
    rows = rng.integers(0, n_obs, size=nnz, dtype=np.int64)
    cols = rng.integers(0, n_vars, size=nnz, dtype=np.int64)
    vals = rng.integers(1, 256, size=nnz, dtype=np.int32).astype(np.float32)
    x = sp.coo_matrix((vals, (rows, cols)), shape=(n_obs, n_vars)).tocsr()
    # Dedup-merge accumulated duplicates from the random row/col draws.
    x.sum_duplicates()

    obs = pd.DataFrame({"cell_id": [f"cell_{i}" for i in range(n_obs)]})
    obs.index = obs["cell_id"]
    var = pd.DataFrame({"gene_id": [f"gene_{i}" for i in range(n_vars)]})
    var.index = var["gene_id"]

    adata = ad.AnnData(X=x, obs=obs, var=var)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out} ({size_mb:.1f} MB) "
          f"n_obs={n_obs} n_vars={n_vars} nnz={x.nnz}")


if __name__ == "__main__":
    main()
