#!/usr/bin/env python3
"""
Subprocess worker for RSS time-series measurement of lazy preprocessing benchmarks.

Runs in a fresh process so ru_maxrss reflects only this workload.
Samples RSS every 100ms via a background thread polling /proc/self/statm.
Outputs a single JSON object to stdout.

Tasks:
  lazy_preprocess_rss     — Backed normalize+log1p + streaming agg
  scanpy_preprocess_rss   — In-memory scanpy normalize+log1p
  lazy_pca_rss            — Backed normalize+log1p + streaming PCA
  scanpy_pca_rss          — In-memory scanpy preprocess + PCA
  e2e_ooc_pipeline        — Full 11-stage OOC pipeline (backed throughout)
  e2e_scanpy_pipeline     — Full pipeline via scanpy in-memory

Usage (called by benchmark_lazy_preprocess.py, not directly):
    python benchmark_lazy_preprocess_rss_worker.py --task lazy_preprocess_rss --dataset census_1m
"""

import argparse
import gc
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))

from bench_env import WORK_DIR


# ──────────────────────────────────────────────────────────────────────────────
# RSS sampling
# ──────────────────────────────────────────────────────────────────────────────

def _rss_mb():
    """Current RSS in MB from /proc/self/statm (Linux)."""
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * resource.getpagesize() / (1024 * 1024)
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _peak_rss_mb():
    """Peak RSS (high-water mark) in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class RssSampler:
    """Background thread that samples RSS at regular intervals."""

    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = []
        self._stop = threading.Event()
        self._t0 = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self):
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._t0
            self.samples.append({"time_s": round(elapsed, 3), "rss_mb": round(_rss_mb(), 1)})
            time.sleep(self.interval_ms / 1000)

    def to_list(self):
        return self.samples


# ──────────────────────────────────────────────────────────────────────────────
# Task: lazy_preprocess_rss
# ──────────────────────────────────────────────────────────────────────────────

def task_lazy_preprocess_rss(dataset_name: str) -> dict:
    """Open SCX backed, normalize+log1p, streaming agg. Record RSS time-series."""
    gc.collect()
    import pyscx

    sampler = RssSampler(interval_ms=100)
    sampler.start()

    scx_path = str(WORK_DIR / f"{dataset_name}.scx")
    t0 = time.perf_counter()

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    n_obs, n_vars = adata.n_obs, adata.n_vars

    pyscx.accel.normalize_total(adata, target_sum=1e4)
    pyscx.accel.log1p(adata)
    # Exercise streaming path
    _ = adata.X.sum(axis=0)

    elapsed = time.perf_counter() - t0
    sampler.stop()

    return {
        "task": "lazy_preprocess_rss",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
        "rss_time_series": sampler.to_list(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task: scanpy_preprocess_rss
# ──────────────────────────────────────────────────────────────────────────────

def task_scanpy_preprocess_rss(dataset_name: str) -> dict:
    """Load h5ad, scanpy normalize+log1p. Record RSS time-series."""
    gc.collect()
    import anndata
    import scanpy as sc

    sampler = RssSampler(interval_ms=100)
    sampler.start()

    h5ad_path = str(WORK_DIR / f"{dataset_name}.h5ad")
    t0 = time.perf_counter()

    adata = anndata.read_h5ad(h5ad_path)
    n_obs, n_vars = adata.n_obs, adata.n_vars

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    elapsed = time.perf_counter() - t0
    sampler.stop()

    return {
        "task": "scanpy_preprocess_rss",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
        "rss_time_series": sampler.to_list(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task: lazy_pca_rss
# ──────────────────────────────────────────────────────────────────────────────

def task_lazy_pca_rss(dataset_name: str) -> dict:
    """Backed + lazy normalize+log1p + streaming PCA. Record RSS time-series."""
    gc.collect()
    import pyscx

    sampler = RssSampler(interval_ms=100)
    sampler.start()

    scx_path = str(WORK_DIR / f"{dataset_name}.scx")
    t0 = time.perf_counter()

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    n_obs, n_vars = adata.n_obs, adata.n_vars

    pyscx.accel.normalize_total(adata, target_sum=1e4)
    pyscx.accel.log1p(adata)

    # PCA streams through lazy transforms
    try:
        pyscx.accel.pca(adata, n_comps=50)
    except (ValueError, TypeError):
        adata.X = adata.X.to_memory()
        pyscx.accel.pca(adata, n_comps=50)

    elapsed = time.perf_counter() - t0
    sampler.stop()

    return {
        "task": "lazy_pca_rss",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
        "rss_time_series": sampler.to_list(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task: scanpy_pca_rss
# ──────────────────────────────────────────────────────────────────────────────

def task_scanpy_pca_rss(dataset_name: str) -> dict:
    """Load h5ad + scanpy preprocess + PCA. Record RSS time-series."""
    gc.collect()
    import anndata
    import scanpy as sc

    sampler = RssSampler(interval_ms=100)
    sampler.start()

    h5ad_path = str(WORK_DIR / f"{dataset_name}.h5ad")
    t0 = time.perf_counter()

    adata = anndata.read_h5ad(h5ad_path)
    n_obs, n_vars = adata.n_obs, adata.n_vars

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=50)

    elapsed = time.perf_counter() - t0
    sampler.stop()

    return {
        "task": "scanpy_pca_rss",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "wall_clock_s": round(elapsed, 3),
        "rss_time_series": sampler.to_list(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task: e2e_ooc_pipeline
# ──────────────────────────────────────────────────────────────────────────────

def task_e2e_ooc_pipeline(dataset_name: str) -> dict:
    """Full 11-stage OOC pipeline on backed data. Per-stage timing + RSS."""
    gc.collect()
    import numpy as np
    import pyscx
    import scanpy as sc

    sampler = RssSampler(interval_ms=100)
    sampler.start()

    stage_timings = {}
    t_total = time.perf_counter()

    # 1. Open backed
    t0 = time.perf_counter()
    scx_path = str(WORK_DIR / f"{dataset_name}.scx")
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    n_obs, n_vars = adata.n_obs, adata.n_vars
    stage_timings["open"] = round(time.perf_counter() - t0, 3)

    # 2. QC metrics
    t0 = time.perf_counter()
    try:
        pyscx.accel.calculate_qc_metrics(adata, qc_vars=["mt"])
    except Exception:
        # Fallback: compute mito pct manually if qc_metrics doesn't work on backed
        var_names = adata.var_names
        mt_mask = var_names.str.startswith("MT-") | var_names.str.startswith("mt-")
        if mt_mask.any():
            adata.obs["n_genes_by_counts"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()
            total = np.asarray(adata.X.sum(axis=1)).ravel()
            mt_sum = np.asarray(adata.X[:, mt_mask].sum(axis=1)).ravel()
            adata.obs["total_counts"] = total
            adata.obs["pct_counts_mt"] = mt_sum / np.maximum(total, 1) * 100
    stage_timings["qc"] = round(time.perf_counter() - t0, 3)

    # 3. Filter cells
    t0 = time.perf_counter()
    pyscx.accel.filter_cells(adata, min_genes=200)
    stage_timings["filter_cells"] = round(time.perf_counter() - t0, 3)

    # 4. Filter genes
    t0 = time.perf_counter()
    pyscx.accel.filter_genes(adata, min_cells=3)
    stage_timings["filter_genes"] = round(time.perf_counter() - t0, 3)

    # 5. Normalize (before HVG — standard scanpy workflow order)
    t0 = time.perf_counter()
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    stage_timings["normalize"] = round(time.perf_counter() - t0, 3)

    # 6. Log1p
    t0 = time.perf_counter()
    pyscx.accel.log1p(adata)
    stage_timings["log1p"] = round(time.perf_counter() - t0, 3)

    # 7. HVG (streaming via pyscx — no materialization)
    t0 = time.perf_counter()
    n_top = min(2000, adata.n_vars)
    pyscx.accel.highly_variable_genes(
        adata, n_top_genes=n_top, flavor="seurat_v3",
        subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0,
    )
    stage_timings["hvg"] = round(time.perf_counter() - t0, 3)

    # 8. PCA
    t0 = time.perf_counter()
    try:
        pyscx.accel.pca(adata, n_comps=50)
    except (ValueError, TypeError):
        adata.X = adata.X.to_memory()
        pyscx.accel.pca(adata, n_comps=50)
    stage_timings["pca"] = round(time.perf_counter() - t0, 3)

    # 9. kNN
    t0 = time.perf_counter()
    pyscx.accel.neighbors(adata, n_neighbors=15)
    stage_timings["knn"] = round(time.perf_counter() - t0, 3)

    # 10. UMAP
    t0 = time.perf_counter()
    pyscx.accel.umap(adata)
    stage_timings["umap"] = round(time.perf_counter() - t0, 3)

    # 11. Leiden
    t0 = time.perf_counter()
    pyscx.accel.leiden(adata, resolution=1.0, random_state=42)
    stage_timings["leiden"] = round(time.perf_counter() - t0, 3)

    total_s = round(time.perf_counter() - t_total, 3)
    sampler.stop()

    leiden_labels = adata.obs["leiden"].values.tolist()

    return {
        "task": "e2e_ooc_pipeline",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "n_obs_after_filter": adata.n_obs,
        "n_vars_after_hvg": adata.n_vars,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "wall_clock_s": total_s,
        "stage_timings": stage_timings,
        "rss_time_series": sampler.to_list(),
        "leiden_labels": leiden_labels,
        "n_leiden_clusters": len(set(leiden_labels)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Task: e2e_scanpy_pipeline
# ──────────────────────────────────────────────────────────────────────────────

def task_e2e_scanpy_pipeline(dataset_name: str) -> dict:
    """Full pipeline via scanpy in-memory. Per-stage timing + RSS."""
    gc.collect()
    import anndata
    import scanpy as sc

    sampler = RssSampler(interval_ms=100)
    sampler.start()

    stage_timings = {}
    t_total = time.perf_counter()

    # 1. Load
    t0 = time.perf_counter()
    h5ad_path = str(WORK_DIR / f"{dataset_name}.h5ad")
    adata = anndata.read_h5ad(h5ad_path)
    n_obs, n_vars = adata.n_obs, adata.n_vars
    stage_timings["load"] = round(time.perf_counter() - t0, 3)

    # 2. QC metrics
    t0 = time.perf_counter()
    adata.var["mt"] = adata.var_names.str.startswith("MT-") | adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
    stage_timings["qc"] = round(time.perf_counter() - t0, 3)

    # 3. Filter cells
    t0 = time.perf_counter()
    sc.pp.filter_cells(adata, min_genes=200)
    stage_timings["filter_cells"] = round(time.perf_counter() - t0, 3)

    # 4. Filter genes
    t0 = time.perf_counter()
    sc.pp.filter_genes(adata, min_cells=3)
    stage_timings["filter_genes"] = round(time.perf_counter() - t0, 3)

    # 5. Normalize (before HVG — standard scanpy workflow order)
    t0 = time.perf_counter()
    sc.pp.normalize_total(adata, target_sum=1e4)
    stage_timings["normalize"] = round(time.perf_counter() - t0, 3)

    # 6. Log1p
    t0 = time.perf_counter()
    sc.pp.log1p(adata)
    stage_timings["log1p"] = round(time.perf_counter() - t0, 3)

    # 7. HVG (on normalized+log1p data, using default seurat flavor)
    t0 = time.perf_counter()
    n_top = min(2000, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3",
            subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0,
        )
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
    stage_timings["hvg"] = round(time.perf_counter() - t0, 3)

    # 8. PCA
    t0 = time.perf_counter()
    sc.pp.pca(adata, n_comps=50)
    stage_timings["pca"] = round(time.perf_counter() - t0, 3)

    # 9. kNN
    t0 = time.perf_counter()
    sc.pp.neighbors(adata, n_neighbors=15)
    stage_timings["knn"] = round(time.perf_counter() - t0, 3)

    # 10. UMAP
    t0 = time.perf_counter()
    sc.tl.umap(adata)
    stage_timings["umap"] = round(time.perf_counter() - t0, 3)

    # 11. Leiden
    t0 = time.perf_counter()
    sc.tl.leiden(adata, resolution=1.0, random_state=42)
    stage_timings["leiden"] = round(time.perf_counter() - t0, 3)

    total_s = round(time.perf_counter() - t_total, 3)
    sampler.stop()

    leiden_labels = adata.obs["leiden"].values.tolist()

    return {
        "task": "e2e_scanpy_pipeline",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "n_obs_after_filter": adata.n_obs,
        "n_vars_after_hvg": adata.n_vars,
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "wall_clock_s": total_s,
        "stage_timings": stage_timings,
        "rss_time_series": sampler.to_list(),
        "leiden_labels": leiden_labels,
        "n_leiden_clusters": len(set(leiden_labels)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

TASKS = {
    "lazy_preprocess_rss": task_lazy_preprocess_rss,
    "scanpy_preprocess_rss": task_scanpy_preprocess_rss,
    "lazy_pca_rss": task_lazy_pca_rss,
    "scanpy_pca_rss": task_scanpy_pca_rss,
    "e2e_ooc_pipeline": task_e2e_ooc_pipeline,
    "e2e_scanpy_pipeline": task_e2e_scanpy_pipeline,
}


def main():
    parser = argparse.ArgumentParser(
        description="RSS time-series worker for lazy preprocessing benchmarks"
    )
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    result = TASKS[args.task](args.dataset)
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    main()
