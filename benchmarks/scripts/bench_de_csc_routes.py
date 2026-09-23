#!/usr/bin/env python
"""Phase-2 §5.2/§5.3 DE route benchmark: CSR vs CSC-direct, and CSC densify vs
exact sparse-nnz Wilcoxon.

Runs **one** (mode, dataset) configuration per process — the caller launches a
fresh process per mode so process RSS starts clean. The script sets the
`SCX_ACCEL_WILCOXON_NNZ` gate itself from `--mode`, before pyscx is imported,
because pyscx reads it once per process through a `OnceLock`. Prints one JSON
line to stdout with wall-clock, peak RSS, and the route the op actually took.

Modes:
  csr        — prefer_format="csr"                            (CSR streaming, §5.2 baseline)
  csc_dense  — prefer_format="csc" + SCX_ACCEL_WILCOXON_NNZ=0 (CSC-direct densify, §5.2)
  csc_nnz    — prefer_format="csc"                            (§5.3 exact sparse-nnz, the default)

The input file MUST carry a CSC sidecar (`scx build-csc`) and the full cell axis
(no obs subset — a deletion vector disables the CSC route).

Usage:
  python bench_de_csc_routes.py \
      --scx <sidecar.scx> --mode csc_nnz --groupby cell_type
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import time


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# Same candidate order as benchmarks/comprehensive/benchmarks/accel_de.py.
_GROUPBY_CANDIDATES = (
    "cell_type",
    "leiden",
    "louvain",
    "cluster",
    "perturbation",
    "target",
)


def _select_groupby(adata) -> str:
    for c in _GROUPBY_CANDIDATES:
        if c in adata.obs.columns and 2 <= adata.obs[c].astype("str").nunique() <= 200:
            return c
    # Fallback: any categorical-ish column with a sane group count.
    for c in adata.obs.columns:
        try:
            n = adata.obs[c].astype("str").nunique()
        except Exception:
            continue
        if 2 <= n <= 200:
            return c
    raise SystemExit("no usable groupby column found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scx", required=True)
    ap.add_argument("--mode", required=True, choices=["csr", "csc_dense", "csc_nnz"])
    ap.add_argument("--groupby", default=None)
    ap.add_argument("--gene-chunk-size", type=int, default=None)
    args = ap.parse_args()

    # Before the import, and overriding whatever the caller exported, so the
    # mode names the kernel that runs; the recorded `route` confirms it.
    os.environ["SCX_ACCEL_WILCOXON_NNZ"] = "0" if args.mode == "csc_dense" else "1"
    import pyscx

    prefer_format = "csr" if args.mode == "csr" else "csc"

    adata = pyscx.open(args.scx).to_anndata(backed=True)
    groupby = args.groupby or _select_groupby(adata)
    # A categorical carries every level it was declared with, used or not, and
    # `rank_genes_groups` refuses a participating group with fewer than two
    # cells (since 0.17) — so census_1m's `disease`, 17 levels present out of
    # the census-wide vocabulary, raised before the first gene was tested.
    if hasattr(adata.obs[groupby], "cat"):
        adata.obs[groupby] = adata.obs[groupby].cat.remove_unused_categories()
    n_groups = int(adata.obs[groupby].astype("str").nunique())
    n_obs, n_vars = adata.shape

    gc.collect()
    t0 = time.perf_counter()
    pyscx.accel.rank_genes_groups(
        adata,
        groupby,
        reference="rest",  # 1-vs-rest → the arm the nnz kernel accelerates
        prefer_format=prefer_format,
        device="cpu",
        gene_chunk_size=args.gene_chunk_size,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0

    route = adata.uns.get("scx_accel", {}).get("rank_genes_groups", {}).get("route", "?")
    print(
        json.dumps(
            {
                "mode": args.mode,
                "prefer_format": prefer_format,
                "nnz_flag": args.mode == "csc_nnz",
                "route": route,
                "groupby": groupby,
                "n_obs": int(n_obs),
                "n_vars": int(n_vars),
                "n_groups": n_groups,
                "wall_ms": round(wall_ms, 1),
                "peak_rss_mb": round(_peak_rss_mb(), 1),
            }
        )
    )


if __name__ == "__main__":
    main()
