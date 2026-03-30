#!/usr/bin/env python3
"""Subprocess worker for isolated memory measurement of lazy preprocessing.

Each task runs in a fresh process so ru_maxrss (high-water mark) reflects
only that task's peak RSS.  Outputs a single JSON line to stdout.

Usage (called by benchmark_lazy_preprocess.py, not directly):
    python benchmark_lazy_preprocess_worker.py --task lazy_preprocess --dataset census_1m
    python benchmark_lazy_preprocess_worker.py --task full_pipeline --dataset census_1m
    python benchmark_lazy_preprocess_worker.py --task materialized_preprocess --dataset census_1m
"""

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))

DATA_DIR = Path(os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx"))


def _rss_mb():
    """Current peak RSS in MB (Linux: ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def measure_lazy_preprocess(dataset_name: str) -> dict:
    """Open SCX backed, normalize+log1p, exercise streaming agg.  Report peak RSS."""
    gc.collect()
    import pyscx

    scx_path = str(DATA_DIR / f"{dataset_name}.scx")
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    n_obs, n_vars = adata.n_obs, adata.n_vars

    t0 = time.perf_counter()
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    pyscx.accel.log1p(adata)
    # Exercise the lazy path with a streaming aggregation
    _ = adata.X.sum(axis=0)
    elapsed = time.perf_counter() - t0

    return {
        "task": "lazy_preprocess",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "peak_rss_mb": round(_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
    }


def measure_full_pipeline(dataset_name: str) -> dict:
    """Full pipeline: QC -> normalize -> log1p -> PCA -> kNN -> UMAP -> Leiden.

    Uses pyscx.accel.filter_cells/filter_genes (streaming, no materialization)
    instead of scanpy's versions which materialize the backed dataset.
    """
    gc.collect()
    import pyscx
    import scanpy as sc

    scx_path = str(DATA_DIR / f"{dataset_name}.scx")
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    n_obs, n_vars = adata.n_obs, adata.n_vars
    print(f"  X type after open: {type(adata.X).__name__}", flush=True)

    t0 = time.perf_counter()

    # QC filtering (streaming, stays backed)
    pyscx.accel.filter_cells(adata, min_genes=200)
    print(f"  X type after filter_cells: {type(adata.X).__name__} shape={adata.shape}", flush=True)
    pyscx.accel.filter_genes(adata, min_cells=3)
    print(f"  X type after filter_genes: {type(adata.X).__name__} shape={adata.shape}", flush=True)

    # Lazy preprocessing (stays backed)
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    print(f"  X type after normalize_total: {type(adata.X).__name__}", flush=True)
    pyscx.accel.log1p(adata)
    print(f"  X type after log1p: {type(adata.X).__name__}", flush=True)

    # PCA streams through lazy transforms (no materialization)
    pyscx.accel.pca(adata, n_comps=50)
    print(f"  X type after pca: {type(adata.X).__name__}", flush=True)
    pyscx.accel.neighbors(adata, n_neighbors=15)
    pyscx.accel.umap(adata)
    pyscx.accel.leiden(adata)

    elapsed = time.perf_counter() - t0

    return {
        "task": "full_pipeline",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "n_obs_after_filter": adata.n_obs,
        "peak_rss_mb": round(_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
        "n_leiden_clusters": int(adata.obs["leiden"].nunique()),
    }


def measure_materialized_preprocess(dataset_name: str) -> dict:
    """Baseline: scanpy normalize+log1p on fully materialized data."""
    gc.collect()
    import anndata
    import scanpy as sc

    h5ad_path = str(DATA_DIR / f"{dataset_name}.h5ad")

    t0 = time.perf_counter()
    adata = anndata.read_h5ad(h5ad_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    elapsed = time.perf_counter() - t0

    return {
        "task": "materialized_preprocess",
        "dataset": dataset_name,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "peak_rss_mb": round(_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
    }


TASKS = {
    "lazy_preprocess": measure_lazy_preprocess,
    "full_pipeline": measure_full_pipeline,
    "materialized_preprocess": measure_materialized_preprocess,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    result = TASKS[args.task](args.dataset)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
