"""Create log-normalized versions of benchmark datasets for Pcodec evaluation.

Applies scanpy normalize_total(target_sum=1e4) + log1p to produce genuinely
float-valued h5ad files where Pcodec's advantage over Zstd should be visible.
"""

import os
import sys
from pathlib import Path

import anndata
import scanpy as sc

DATA_DIR = Path(
    os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx/benchmarks/datasets")
)

DATASETS = ["pbmc3k", "smartseq2", "tabula_sapiens_100k"]


def main():
    for name in DATASETS:
        raw_path = DATA_DIR / f"{name}.h5ad"
        out_path = DATA_DIR / f"{name}_lognorm.h5ad"

        if out_path.exists():
            print(f"[skip] {out_path} already exists")
            continue

        if not raw_path.exists():
            print(f"[skip] {raw_path} not found")
            continue

        print(f"[load] {raw_path} ...", flush=True)
        adata = anndata.read_h5ad(raw_path)
        print(f"  shape={adata.shape}, X dtype={adata.X.dtype}")

        print("  normalize_total + log1p ...", flush=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"  X dtype after={adata.X.dtype}")

        print(f"  writing {out_path} ...", flush=True)
        adata.write_h5ad(out_path)
        print(f"  done ({out_path.stat().st_size / 1e6:.1f} MB)")

    print("\nAll done.")


if __name__ == "__main__":
    main()
