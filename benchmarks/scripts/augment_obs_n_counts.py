#!/usr/bin/env python3
"""Ensure an h5ad carries ``obs['n_counts'] = X.sum(axis=1)``.

Idempotent: no-ops if the column already exists and is finite. Used by
the dataset prep pipeline so selective-predicate benchmarks
(``cost_model``, ``cloud_reader_vs_pull``) can synthesize a quantile
filter on any staged dataset. Safe to re-run on already-staged h5ad.

Invoked by ``benchmarks/scripts/download_datasets.sh`` after each
successful ``write_h5ad(...)`` and can also be run standalone on
datasets that were staged before this script existed.

Usage:
    python augment_obs_n_counts.py <h5ad_path> [--force]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import anndata
import numpy as np


def _row_sum(x) -> np.ndarray:
    """Return X.sum(axis=1) as a 1-D float32 ndarray (sparse or dense)."""
    s = x.sum(axis=1)
    return np.asarray(s).ravel().astype(np.float32, copy=False)


def augment(path: Path, force: bool) -> int:
    adata = anndata.read_h5ad(path)
    has_col = "n_counts" in adata.obs.columns
    if has_col and not force:
        col = np.asarray(adata.obs["n_counts"].values)
        if np.isfinite(col).all():
            print(
                f"skip: n_counts already present (n={adata.n_obs}, "
                f"min={col.min():.3g}, median={np.median(col):.3g}, "
                f"max={col.max():.3g})"
            )
            return 0
        print("warn: n_counts present but not all finite — recomputing")

    n_counts = _row_sum(adata.X)
    adata.obs["n_counts"] = n_counts

    tmp = Path(tempfile.mkstemp(suffix=".h5ad", dir=str(path.parent))[1])
    try:
        adata.write_h5ad(tmp)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(
        f"wrote n_counts: n={adata.n_obs}, "
        f"min={n_counts.min():.3g}, median={np.median(n_counts):.3g}, "
        f"max={n_counts.max():.3g} -> {path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="h5ad file to augment in place")
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute n_counts even if already present",
    )
    args = parser.parse_args(argv)
    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2
    return augment(args.path, args.force)


if __name__ == "__main__":
    sys.exit(main())
