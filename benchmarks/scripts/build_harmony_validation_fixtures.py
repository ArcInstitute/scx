#!/usr/bin/env python3
"""Generate the three validation fixtures consumed by
`pyscx/tests/test_harmony_validation.py`.

For each validation set (name, source dataset, batch column, n_cap, n_pcs):
  1. Load h5ad, subsample if needed, run scanpy normalise/log/scale/PCA.
  2. Save `pca` (f32) + `batch` (i32) + `params` dict to
     `benchmarks/results/harmony/reference/<name>/inputs.npz`.
  3. Call `generate_harmony_reference.R` to produce the R-harmony
     reference; load the resulting `.npz` and re-save alongside inputs
     as `r_reference.npz`.

Run once before `pytest pyscx/tests/test_harmony_validation.py`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "scripts"))
from bench_env import DATA_DIR  # noqa: E402

RSCRIPT = os.environ.get(
    "RSCRIPT", "/home/nickyoungblut/miniforge3/envs/rscx/bin/Rscript"
)
REF_ROOT = REPO_ROOT / "benchmarks" / "results" / "harmony" / "reference"

# (name, dataset, batch_col or None, n_cap, n_pcs)
SETS = [
    ("pbmc_small", "pbmc3k", None, 2700, 30),
    ("cell_lines", "smartseq2", "dataset_id", 9478, 20),
    ("hlca_subset", "tabula_sapiens_100k", "donor_id", 50_000, 30),
]


def build_pca(dataset: str, n_cap: int, n_pcs: int, batch_col: str | None):
    import anndata as ad
    import scanpy as sc

    p = DATA_DIR / f"{dataset}.h5ad"
    print(f"  loading {p}", flush=True)
    adata = ad.read_h5ad(p)
    adata.obs_names_make_unique()
    if adata.n_obs > n_cap:
        rng = np.random.default_rng(0)
        idx = rng.choice(adata.n_obs, size=n_cap, replace=False)
        idx.sort()
        adata = adata[idx].copy()

    x_max = float(adata.X.max())
    if x_max > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    # Deliberately skip sc.pp.scale — on 100k x 61k inputs it densifies to
    # ~24 GB and OOMs. sc.tl.pca(zero_center=True) handles mean-centering
    # internally without requiring the dense scaled matrix.
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    sc.tl.pca(adata, n_comps=n_pcs, random_state=0, zero_center=True)

    if batch_col and batch_col in adata.obs.columns:
        col = adata.obs[batch_col].astype(str).to_numpy()
        _, codes = np.unique(col, return_inverse=True)
        batch = codes.astype(np.int32)
    else:
        rng = np.random.default_rng(0)
        batch = rng.integers(0, 3, size=adata.n_obs, dtype=np.int32)

    return adata.obsm["X_pca"].astype(np.float32), batch


def main() -> int:
    REF_ROOT.mkdir(parents=True, exist_ok=True)
    for name, dataset, batch_col, n_cap, n_pcs in SETS:
        out_dir = REF_ROOT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        inputs_path = out_dir / "inputs.npz"
        ref_path = out_dir / "r_reference.npz"

        if inputs_path.exists() and ref_path.exists():
            print(f"[{name}] already exists; skip")
            continue

        print(f"[{name}] building PCA + batch…", flush=True)
        pca, batch = build_pca(dataset, n_cap, n_pcs, batch_col)

        params = {
            "dataset": dataset,
            "batch_col": batch_col or "synthetic",
            "n_cap": n_cap,
            "n_pcs": n_pcs,
            "nclust": 100,
            "theta": 2.0,
            "max_iter": 10,
            "seed": 0,
        }
        np.savez(
            inputs_path,
            pca=pca,
            batch=batch,
            params=np.array(params, dtype=object),
        )
        print(f"  wrote {inputs_path}")

        # Fire R harmony.
        print(f"[{name}] running R harmony reference…", flush=True)
        pca_npy = out_dir / "_pca.npy"
        batch_npy = out_dir / "_batch.npy"
        out_prefix = out_dir / "_rref"
        np.save(pca_npy, pca)
        np.save(batch_npy, batch)
        subprocess.check_call(
            [
                RSCRIPT,
                str(REPO_ROOT / "benchmarks/scripts/generate_harmony_reference.R"),
                "--pca", str(pca_npy),
                "--batch", str(batch_npy),
                "--out-prefix", str(out_prefix),
                "--theta", str(params["theta"]),
                "--nclust", str(params["nclust"]),
                "--max-iter", str(params["max_iter"]),
                "--seed", str(params["seed"]),
            ]
        )

        # The R script produces `<prefix>.npz` — move to r_reference.npz
        # and drop the RDS / scratch npys.
        shutil.copy(f"{out_prefix}.npz", ref_path)
        for p in [
            pca_npy,
            batch_npy,
            Path(f"{out_prefix}.npz"),
            Path(f"{out_prefix}.rds"),
        ]:
            if p.exists():
                p.unlink()
        print(f"  wrote {ref_path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
