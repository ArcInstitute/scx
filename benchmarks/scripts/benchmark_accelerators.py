#!/usr/bin/env python3
"""
SCX Phase 4b Accelerator Benchmarks — CPU Rust-native vs Scanpy

Compares pyscx.accel.* (Rust-native CPU accelerators) against their scanpy
equivalents across dataset sizes. Covers COMPREHENSIVE-BENCHMARKING.md §3.10.

Benchmarks:
  - PCA: pyscx.accel.pca() vs sc.pp.pca()
  - kNN: pyscx.accel.neighbors() vs sc.pp.neighbors()
  - UMAP: pyscx.accel.umap() vs sc.tl.umap()
  - DE (in-memory): pyscx.accel.rank_genes_groups() vs sc.tl.rank_genes_groups()
  - DE (streaming): gene-chunked DE on backed data
  - Pseudobulk DE: pyscx.accel.pseudobulk_dex()
  - Stratified DE: stratify_by parameter for single-cell + pseudobulk

Usage:
    # Quick correctness validation on pbmc3k
    python benchmarks/scripts/benchmark_accelerators.py --mode validate

    # Individual benchmarks
    python benchmarks/scripts/benchmark_accelerators.py --mode pca
    python benchmarks/scripts/benchmark_accelerators.py --mode knn --datasets census_500k census_1m
    python benchmarks/scripts/benchmark_accelerators.py --mode de_streaming

    # Everything
    python benchmarks/scripts/benchmark_accelerators.py --mode all

    # Generate markdown report from existing JSON
    python benchmarks/scripts/benchmark_accelerators.py --mode report
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DATA_DIR = Path(os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx"))

DATASETS = {
    "pbmc3k": {"h5ad": DATA_DIR / "pbmc3k.h5ad", "cells": 2_700},
    "tabula_sapiens_100k": {"h5ad": DATA_DIR / "tabula_sapiens_100k.h5ad", "cells": 100_000},
    "census_500k": {"h5ad": DATA_DIR / "census_500k.h5ad", "cells": 500_000},
    "census_1m": {"h5ad": DATA_DIR / "census_1m.h5ad", "cells": 1_000_000},
}

BENCH_DATASETS = ["tabula_sapiens_100k", "census_500k", "census_1m"]
DE_STREAMING_DATASETS = ["tabula_sapiens_100k", "census_1m"]
PSEUDOBULK_DATASETS = ["tabula_sapiens_100k", "census_1m"]
STRATIFIED_DATASETS = ["tabula_sapiens_100k"]

N_RUNS = 3


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def get_rss_mb() -> float:
    """Current RSS in MB from /proc/self/statm."""
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def ensure_scx_file(dataset_name: str) -> Path | None:
    """Ensure SCX file exists, converting from h5ad if needed."""
    import pyscx

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    scx_path = h5ad_path.with_suffix(".scx")

    if scx_path.exists():
        return scx_path
    if not h5ad_path.exists():
        return None

    print(f"  Converting {dataset_name} h5ad -> SCX...")
    import anndata
    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, str(scx_path))
    del adata
    gc.collect()
    print(f"    Saved: {scx_path}")
    return scx_path


def load_preprocessed_adata(dataset_name: str, n_hvgs: int = 2000):
    """Load dataset in memory, preprocess (normalize, log1p, HVG).

    Returns preprocessed AnnData with scipy CSR X.
    """
    import anndata
    import scanpy as sc

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        print(f"  SKIP: {h5ad_path} not found")
        return None

    print(f"  Loading {dataset_name} ({info['cells']:,} cells)...")
    adata = anndata.read_h5ad(str(h5ad_path))

    print(f"  Preprocessing (normalize, log1p, HVG)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = min(n_hvgs, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3",
            subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
        )
    except Exception:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)

    print(f"  Ready: {adata.n_obs:,} x {adata.n_vars:,}")
    return adata


def load_backed_adata(dataset_name: str):
    """Load dataset as SCX-backed AnnData."""
    import pyscx

    scx_path = ensure_scx_file(dataset_name)
    if scx_path is None:
        print(f"  SKIP: {dataset_name} not available")
        return None

    print(f"  Loading {dataset_name} in backed mode...")
    adata = pyscx.open(str(scx_path)).to_anndata()
    print(f"  Backed: {adata.n_obs:,} x {adata.n_vars:,}")
    return adata


def ensure_groupby_column(adata, n_groups: int = 10, column: str = "leiden"):
    """Add a groupby column with approximately n_groups clusters.

    Uses deterministic assignment based on obs indices.
    """
    if column in adata.obs.columns and adata.obs[column].nunique() >= 3:
        return column

    import pandas as pd
    rng = np.random.RandomState(42)
    labels = rng.choice([f"group_{i}" for i in range(n_groups)], size=adata.n_obs)
    adata.obs[column] = pd.Categorical(labels)
    print(f"  Added synthetic '{column}' column with {n_groups} groups")
    return column


def ensure_perturbation_labels(adata):
    """Add synthetic perturbation and donor columns for pseudobulk DE."""
    import pandas as pd
    if "perturbation" not in adata.obs.columns:
        rng = np.random.RandomState(42)
        n = adata.n_obs
        pert_labels = ["treatment_A", "treatment_B", "treatment_C", "control"]
        donor_labels = ["donor_1", "donor_2", "donor_3"]
        adata.obs["perturbation"] = pd.Categorical(
            rng.choice(pert_labels, size=n)
        )
        adata.obs["donor"] = pd.Categorical(
            rng.choice(donor_labels, size=n)
        )
        print(f"  Added synthetic perturbation (4 levels) and donor (3 levels)")


def ensure_cell_type_column(adata, n_types: int = 10):
    """Ensure cell_type column exists (use real if available, synthetic otherwise)."""
    import pandas as pd
    if "cell_type" in adata.obs.columns and adata.obs["cell_type"].nunique() >= 3:
        return
    rng = np.random.RandomState(42)
    types = [f"type_{i}" for i in range(n_types)]
    adata.obs["cell_type"] = pd.Categorical(rng.choice(types, size=adata.n_obs))
    print(f"  Added synthetic cell_type column with {n_types} types")


def cosine_similarity_per_pc(pcs_a, pcs_b):
    """Sign-invariant cosine similarity per principal component."""
    n_comps = min(pcs_a.shape[1], pcs_b.shape[1])
    sims = np.zeros(n_comps)
    for i in range(n_comps):
        a = pcs_a[:, i].astype(np.float64)
        b = pcs_b[:, i].astype(np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            sims[i] = 0.0
        else:
            sims[i] = abs(np.dot(a, b) / (norm_a * norm_b))
    return sims


def compute_recall_at_k(indices_approx, indices_exact, k):
    """Fraction of exact kNN captured by approximate kNN."""
    n_obs = len(indices_approx)
    total_recall = 0.0
    for i in range(n_obs):
        approx_set = set(indices_approx[i][:k])
        exact_set = set(indices_exact[i][:k])
        total_recall += len(approx_set & exact_set) / k
    return total_recall / n_obs


def get_knn_indices_from_adata(adata, n_neighbors):
    """Extract kNN indices from adata.obsp['distances'] CSR matrix."""
    import scipy.sparse as sp
    dist_csr = adata.obsp["distances"]
    if not sp.issparse(dist_csr):
        raise ValueError("Expected sparse distances matrix")
    n_obs = dist_csr.shape[0]
    indices = np.zeros((n_obs, n_neighbors), dtype=np.int64)
    for i in range(n_obs):
        row_start = dist_csr.indptr[i]
        row_end = dist_csr.indptr[i + 1]
        row_indices = dist_csr.indices[row_start:row_end]
        row_dists = dist_csr.data[row_start:row_end]
        sorted_idx = np.argsort(row_dists)[:n_neighbors]
        n_fill = min(len(sorted_idx), n_neighbors)
        indices[i, :n_fill] = row_indices[sorted_idx[:n_fill]]
    return indices


def get_de_gene_names(adata, n_top=100):
    """Extract top DE gene names per group from adata.uns['rank_genes_groups']."""
    rgg = adata.uns["rank_genes_groups"]
    groups = list(rgg["names"].dtype.names)
    result = {}
    for g in groups:
        result[g] = list(rgg["names"][g][:n_top])
    return result


def compute_de_gene_overlap(genes_a, genes_b, n_top=100):
    """Compute fraction of overlapping top DE genes per group."""
    groups = set(genes_a.keys()) & set(genes_b.keys())
    if not groups:
        return 0.0
    overlaps = []
    for g in groups:
        set_a = set(genes_a[g][:n_top])
        set_b = set(genes_b[g][:n_top])
        overlaps.append(len(set_a & set_b) / n_top)
    return float(np.mean(overlaps))


def save_json(data, filename):
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  JSON saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# PCA Benchmark (§3.10.1)
# ──────────────────────────────────────────────────────────────────────────────

def run_pca_benchmark(datasets=None, n_runs=N_RUNS):
    """PCA: pyscx.accel.pca() vs sc.pp.pca() at 100K, 500K, 1M cells."""
    import pyscx
    import scanpy as sc

    if datasets is None:
        datasets = BENCH_DATASETS

    print(f"\n{'='*60}")
    print(f"PCA Benchmark — SCX vs Scanpy (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []
    for ds_name in datasets:
        adata = load_preprocessed_adata(ds_name)
        if adata is None:
            continue

        n_obs, n_vars = adata.n_obs, adata.n_vars
        n_comps = min(50, n_vars - 1)

        # SCX PCA
        print(f"\n  SCX PCA on {ds_name} ({n_runs} runs)...")
        scx_times = []
        scx_pcs = None
        scx_var_ratio = None
        for run in range(n_runs):
            adata_copy = adata.copy()
            gc.collect()
            rss_before = get_rss_mb()
            t0 = time.perf_counter()
            pyscx.accel.pca(adata_copy, n_comps=n_comps, device="cpu")
            wall = time.perf_counter() - t0
            rss_after = get_rss_mb()
            scx_times.append(wall)
            if run == 0:
                scx_pcs = adata_copy.obsm["X_pca"].copy()
                scx_var_ratio = adata_copy.uns["pca"]["variance_ratio"].copy()
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_copy
            gc.collect()

        # Scanpy PCA
        print(f"  Scanpy PCA on {ds_name} ({n_runs} runs)...")
        scanpy_times = []
        scanpy_pcs = None
        scanpy_var_ratio = None
        for run in range(n_runs):
            adata_copy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            sc.pp.pca(adata_copy, n_comps=n_comps)
            wall = time.perf_counter() - t0
            scanpy_times.append(wall)
            if run == 0:
                scanpy_pcs = adata_copy.obsm["X_pca"].copy()
                scanpy_var_ratio = adata_copy.uns["pca"]["variance_ratio"].copy()
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_copy
            gc.collect()

        # Quality metrics
        sims = cosine_similarity_per_pc(scx_pcs, scanpy_pcs)
        var_corr = float(np.corrcoef(scx_var_ratio[:n_comps],
                                      scanpy_var_ratio[:n_comps])[0, 1])

        scx_median = _median(scx_times)
        scanpy_median = _median(scanpy_times)
        speedup = scanpy_median / scx_median if scx_median > 0 else 0

        result = {
            "benchmark": "accel_pca",
            "dataset": ds_name,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_comps": n_comps,
            "scx_times_s": [round(t, 3) for t in scx_times],
            "scanpy_times_s": [round(t, 3) for t in scanpy_times],
            "scx_median_s": round(scx_median, 3),
            "scanpy_median_s": round(scanpy_median, 3),
            "speedup": round(speedup, 2),
            "cosine_sim_mean": round(float(np.mean(sims)), 6),
            "cosine_sim_min": round(float(np.min(sims)), 6),
            "variance_ratio_pearson_r": round(var_corr, 6),
            "pass_cosine": float(np.min(sims)) > 0.99,
            "pass_variance": var_corr > 0.99,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(result)

        print(f"  SCX median: {scx_median:.2f}s, Scanpy median: {scanpy_median:.2f}s, "
              f"Speedup: {speedup:.1f}x")
        print(f"  Cosine sim min: {float(np.min(sims)):.4f}, "
              f"Var ratio r: {var_corr:.4f}")

        del adata
        gc.collect()

    save_json(results, "accel_pca_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# kNN Benchmark (§3.10.2)
# ──────────────────────────────────────────────────────────────────────────────

def run_knn_benchmark(datasets=None, n_runs=N_RUNS):
    """kNN: pyscx.accel.neighbors() vs sc.pp.neighbors() at 100K, 500K, 1M."""
    import pyscx
    import scanpy as sc
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import adjusted_rand_score

    if datasets is None:
        datasets = BENCH_DATASETS

    print(f"\n{'='*60}")
    print(f"kNN Benchmark — SCX HNSW vs Scanpy (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []
    n_neighbors = 15

    for ds_name in datasets:
        adata = load_preprocessed_adata(ds_name)
        if adata is None:
            continue

        # Run PCA first (scanpy, shared baseline)
        sc.pp.pca(adata, n_comps=50)
        X_pca = adata.obsm["X_pca"].astype(np.float32)
        n_obs = adata.n_obs

        # Exact brute-force kNN (ground truth) — only for smaller datasets
        indices_exact = None
        if n_obs <= 200_000:
            print(f"\n  Computing exact brute-force kNN for {ds_name}...")
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1,
                                  metric="euclidean", algorithm="brute")
            nn.fit(X_pca)
            _, indices_exact_raw = nn.kneighbors(X_pca)
            indices_exact = indices_exact_raw[:, 1:]  # remove self-hits

        # SCX kNN
        print(f"  SCX kNN on {ds_name} ({n_runs} runs)...")
        scx_times = []
        scx_indices = None
        for run in range(n_runs):
            adata_scx = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            pyscx.accel.neighbors(adata_scx, n_neighbors=n_neighbors, device="cpu")
            wall = time.perf_counter() - t0
            scx_times.append(wall)
            if run == 0:
                scx_indices = get_knn_indices_from_adata(adata_scx, n_neighbors)
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_scx
            gc.collect()

        # Scanpy kNN
        print(f"  Scanpy kNN on {ds_name} ({n_runs} runs)...")
        scanpy_times = []
        scanpy_indices = None
        for run in range(n_runs):
            adata_scanpy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            sc.pp.neighbors(adata_scanpy, n_neighbors=n_neighbors)
            wall = time.perf_counter() - t0
            scanpy_times.append(wall)
            if run == 0:
                scanpy_indices = get_knn_indices_from_adata(adata_scanpy, n_neighbors)
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_scanpy
            gc.collect()

        scx_median = _median(scx_times)
        scanpy_median = _median(scanpy_times)
        speedup = scanpy_median / scx_median if scx_median > 0 else 0

        # Quality metrics
        result = {
            "benchmark": "accel_knn",
            "dataset": ds_name,
            "n_obs": n_obs,
            "n_neighbors": n_neighbors,
            "scx_times_s": [round(t, 3) for t in scx_times],
            "scanpy_times_s": [round(t, 3) for t in scanpy_times],
            "scx_median_s": round(scx_median, 3),
            "scanpy_median_s": round(scanpy_median, 3),
            "speedup": round(speedup, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if indices_exact is not None:
            recall_scx = compute_recall_at_k(scx_indices, indices_exact, n_neighbors)
            recall_scanpy = compute_recall_at_k(scanpy_indices, indices_exact, n_neighbors)
            result["recall_scx_vs_exact"] = round(recall_scx, 4)
            result["recall_scanpy_vs_exact"] = round(recall_scanpy, 4)
            result["pass_recall"] = recall_scx > 0.90
            print(f"  Recall@{n_neighbors} SCX: {recall_scx:.4f}, "
                  f"Scanpy: {recall_scanpy:.4f}")

        # Leiden ARI
        try:
            adata_scx_l = adata.copy()
            pyscx.accel.neighbors(adata_scx_l, n_neighbors=n_neighbors, device="cpu")
            sc.tl.leiden(adata_scx_l, resolution=1.0)

            adata_scanpy_l = adata.copy()
            sc.pp.neighbors(adata_scanpy_l, n_neighbors=n_neighbors)
            sc.tl.leiden(adata_scanpy_l, resolution=1.0)

            ari = adjusted_rand_score(
                adata_scx_l.obs["leiden"], adata_scanpy_l.obs["leiden"]
            )
            result["leiden_ari"] = round(ari, 4)
            result["pass_leiden_ari"] = ari > 0.90
            print(f"  Leiden ARI (SCX vs Scanpy graphs): {ari:.4f}")
            del adata_scx_l, adata_scanpy_l
        except Exception as e:
            print(f"  Leiden ARI failed: {e}")

        print(f"  SCX median: {scx_median:.2f}s, Scanpy median: {scanpy_median:.2f}s, "
              f"Speedup: {speedup:.1f}x")

        results.append(result)
        del adata
        gc.collect()

    save_json(results, "accel_knn_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# UMAP Benchmark (§3.10.3)
# ──────────────────────────────────────────────────────────────────────────────

def run_umap_benchmark(datasets=None, n_runs=N_RUNS):
    """UMAP: pyscx.accel.umap() vs sc.tl.umap() at 100K, 500K, 1M."""
    import pyscx
    import scanpy as sc
    from sklearn.manifold import trustworthiness

    if datasets is None:
        datasets = BENCH_DATASETS

    print(f"\n{'='*60}")
    print(f"UMAP Benchmark — SCX vs Scanpy (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        adata = load_preprocessed_adata(ds_name)
        if adata is None:
            continue

        # PCA + kNN (shared baseline via scanpy)
        sc.pp.pca(adata, n_comps=50)
        sc.pp.neighbors(adata, n_neighbors=15)
        X_pca = adata.obsm["X_pca"].astype(np.float32)
        n_obs = adata.n_obs

        # SCX UMAP
        print(f"\n  SCX UMAP on {ds_name} ({n_runs} runs)...")
        scx_times = []
        scx_umap = None
        for run in range(n_runs):
            adata_scx = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            pyscx.accel.umap(adata_scx, device="cpu")
            wall = time.perf_counter() - t0
            scx_times.append(wall)
            if run == 0:
                scx_umap = adata_scx.obsm["X_umap"].copy()
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_scx
            gc.collect()

        # Scanpy UMAP
        print(f"  Scanpy UMAP on {ds_name} ({n_runs} runs)...")
        scanpy_times = []
        scanpy_umap = None
        for run in range(n_runs):
            adata_scanpy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            sc.tl.umap(adata_scanpy)
            wall = time.perf_counter() - t0
            scanpy_times.append(wall)
            if run == 0:
                scanpy_umap = adata_scanpy.obsm["X_umap"].copy()
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_scanpy
            gc.collect()

        scx_median = _median(scx_times)
        scanpy_median = _median(scanpy_times)
        speedup = scanpy_median / scx_median if scx_median > 0 else 0

        # Trustworthiness
        trust_scx = float(trustworthiness(X_pca, scx_umap, n_neighbors=15))
        trust_scanpy = float(trustworthiness(X_pca, scanpy_umap, n_neighbors=15))

        result = {
            "benchmark": "accel_umap",
            "dataset": ds_name,
            "n_obs": n_obs,
            "scx_times_s": [round(t, 3) for t in scx_times],
            "scanpy_times_s": [round(t, 3) for t in scanpy_times],
            "scx_median_s": round(scx_median, 3),
            "scanpy_median_s": round(scanpy_median, 3),
            "speedup": round(speedup, 2),
            "trustworthiness_scx": round(trust_scx, 4),
            "trustworthiness_scanpy": round(trust_scanpy, 4),
            "pass_trustworthiness": trust_scx > 0.90,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(result)

        print(f"  SCX median: {scx_median:.2f}s, Scanpy median: {scanpy_median:.2f}s, "
              f"Speedup: {speedup:.1f}x")
        print(f"  Trustworthiness SCX: {trust_scx:.4f}, Scanpy: {trust_scanpy:.4f}")

        del adata
        gc.collect()

    save_json(results, "accel_umap_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# DE In-Memory Benchmark (§3.10.4)
# ──────────────────────────────────────────────────────────────────────────────

def run_de_inmemory_benchmark(datasets=None, n_runs=N_RUNS):
    """In-memory Wilcoxon DE: pyscx.accel.rank_genes_groups() vs scanpy."""
    import pyscx
    import scanpy as sc
    from scipy.stats import spearmanr

    if datasets is None:
        datasets = BENCH_DATASETS

    print(f"\n{'='*60}")
    print(f"DE In-Memory Benchmark — SCX vs Scanpy (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        adata = load_preprocessed_adata(ds_name)
        if adata is None:
            continue

        n_obs = adata.n_obs
        groupby = ensure_groupby_column(adata, n_groups=10)
        n_groups = adata.obs[groupby].nunique()

        # SCX DE
        print(f"\n  SCX DE on {ds_name} ({n_runs} runs)...")
        scx_times = []
        scx_genes = None
        for run in range(n_runs):
            adata_scx = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            pyscx.accel.rank_genes_groups(adata_scx, groupby=groupby,
                                           reference="rest")
            wall = time.perf_counter() - t0
            scx_times.append(wall)
            if run == 0:
                scx_genes = get_de_gene_names(adata_scx, n_top=100)
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_scx
            gc.collect()

        # Scanpy DE
        print(f"  Scanpy DE on {ds_name} ({n_runs} runs)...")
        scanpy_times = []
        scanpy_genes = None
        for run in range(n_runs):
            adata_scanpy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            sc.tl.rank_genes_groups(adata_scanpy, groupby=groupby,
                                    method="wilcoxon")
            wall = time.perf_counter() - t0
            scanpy_times.append(wall)
            if run == 0:
                scanpy_genes = get_de_gene_names(adata_scanpy, n_top=100)
            print(f"    Run {run+1}: {wall:.2f}s")
            del adata_scanpy
            gc.collect()

        scx_median = _median(scx_times)
        scanpy_median = _median(scanpy_times)
        speedup = scanpy_median / scx_median if scx_median > 0 else 0

        # Quality: gene overlap
        gene_overlap = compute_de_gene_overlap(scx_genes, scanpy_genes, n_top=100)

        # Quality: p-value Spearman correlation (from one representative run)
        pval_corr = None
        try:
            adata_scx_pv = adata.copy()
            pyscx.accel.rank_genes_groups(adata_scx_pv, groupby=groupby,
                                           reference="rest")
            adata_scanpy_pv = adata.copy()
            sc.tl.rank_genes_groups(adata_scanpy_pv, groupby=groupby,
                                    method="wilcoxon")
            first_group = list(adata_scx_pv.uns["rank_genes_groups"]["pvals"].dtype.names)[0]
            pvals_scx = adata_scx_pv.uns["rank_genes_groups"]["pvals"][first_group]
            pvals_scanpy = adata_scanpy_pv.uns["rank_genes_groups"]["pvals"][first_group]
            # Use top genes only (non-trivial p-values)
            n_top_pv = min(200, len(pvals_scx))
            mask = (pvals_scx[:n_top_pv] > 0) & (pvals_scanpy[:n_top_pv] > 0)
            if mask.sum() > 10:
                corr, _ = spearmanr(pvals_scx[:n_top_pv][mask],
                                     pvals_scanpy[:n_top_pv][mask])
                pval_corr = float(corr)
            del adata_scx_pv, adata_scanpy_pv
        except Exception as e:
            print(f"    P-value correlation failed: {e}")

        result = {
            "benchmark": "accel_de_inmemory",
            "dataset": ds_name,
            "n_obs": n_obs,
            "n_groups": n_groups,
            "scx_times_s": [round(t, 3) for t in scx_times],
            "scanpy_times_s": [round(t, 3) for t in scanpy_times],
            "scx_median_s": round(scx_median, 3),
            "scanpy_median_s": round(scanpy_median, 3),
            "speedup": round(speedup, 2),
            "gene_overlap_top100": round(gene_overlap, 4),
            "pass_gene_overlap": gene_overlap > 0.80,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if pval_corr is not None:
            result["pval_spearman_r"] = round(pval_corr, 4)
            result["pass_pval_corr"] = pval_corr > 0.95

        results.append(result)

        print(f"  SCX median: {scx_median:.2f}s, Scanpy median: {scanpy_median:.2f}s, "
              f"Speedup: {speedup:.1f}x")
        print(f"  Gene overlap (top-100): {gene_overlap:.4f}")
        if pval_corr is not None:
            print(f"  P-value Spearman r: {pval_corr:.4f}")

        del adata
        gc.collect()

    save_json(results, "accel_de_inmemory_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# DE Streaming Benchmark (§3.10.4b)
# ──────────────────────────────────────────────────────────────────────────────

def run_de_streaming_benchmark(datasets=None, n_runs=N_RUNS):
    """Streaming gene-chunked DE on backed data at 100K, 1M."""
    import pyscx

    if datasets is None:
        datasets = DE_STREAMING_DATASETS

    print(f"\n{'='*60}")
    print(f"DE Streaming Benchmark — gene-chunked (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        # In-memory baseline
        adata_mem = load_preprocessed_adata(ds_name)
        if adata_mem is None:
            continue

        n_obs = adata_mem.n_obs
        groupby = ensure_groupby_column(adata_mem, n_groups=10)

        print(f"\n  In-memory DE baseline for {ds_name}...")
        pyscx.accel.rank_genes_groups(adata_mem, groupby=groupby, reference="rest")
        inmem_genes = get_de_gene_names(adata_mem, n_top=50)

        # Backed streaming DE with different chunk sizes
        for chunk_size in [1000, 5000, 10000]:
            print(f"  Streaming DE (chunk_size={chunk_size}) on {ds_name} ({n_runs} runs)...")
            backed = load_backed_adata(ds_name)
            if backed is None:
                continue

            # Copy obs metadata for groupby
            backed.obs[groupby] = adata_mem.obs[groupby].values

            stream_times = []
            stream_genes = None
            for run in range(n_runs):
                gc.collect()
                rss_before = get_rss_mb()
                t0 = time.perf_counter()
                pyscx.accel.rank_genes_groups(backed, groupby=groupby,
                                               reference="rest",
                                               gene_chunk_size=chunk_size)
                wall = time.perf_counter() - t0
                rss_after = get_rss_mb()
                stream_times.append(wall)
                if run == 0:
                    stream_genes = get_de_gene_names(backed, n_top=50)
                print(f"    Run {run+1}: {wall:.2f}s, RSS delta: "
                      f"{rss_after - rss_before:.0f} MB")

            stream_median = _median(stream_times)

            # Quality: should be identical to in-memory
            gene_overlap = compute_de_gene_overlap(stream_genes, inmem_genes, n_top=50)

            result = {
                "benchmark": "accel_de_streaming",
                "dataset": ds_name,
                "n_obs": n_obs,
                "gene_chunk_size": chunk_size,
                "stream_times_s": [round(t, 3) for t in stream_times],
                "stream_median_s": round(stream_median, 3),
                "gene_overlap_top50_vs_inmem": round(gene_overlap, 4),
                "pass_identical": gene_overlap == 1.0,
                "peak_rss_mb": round(get_rss_mb(), 0),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results.append(result)

            print(f"    Median: {stream_median:.2f}s, "
                  f"Gene overlap vs in-memory: {gene_overlap:.4f}")

            del backed
            gc.collect()

        del adata_mem
        gc.collect()

    save_json(results, "accel_de_streaming_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Pseudobulk DE Benchmark (§3.10.5b)
# ──────────────────────────────────────────────────────────────────────────────

def run_pseudobulk_benchmark(datasets=None, n_runs=N_RUNS):
    """Pseudobulk DE: pyscx.accel.pseudobulk_dex() at 100K, 1M."""
    import pyscx

    if datasets is None:
        datasets = PSEUDOBULK_DATASETS

    print(f"\n{'='*60}")
    print(f"Pseudobulk DE Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        adata = load_preprocessed_adata(ds_name)
        if adata is None:
            continue

        n_obs = adata.n_obs
        ensure_perturbation_labels(adata)

        # In-memory pseudobulk
        print(f"\n  In-memory pseudobulk DE on {ds_name} ({n_runs} runs)...")
        inmem_times = []
        inmem_result_df = None
        for run in range(n_runs):
            adata_copy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            try:
                df = pyscx.accel.pseudobulk_dex(
                    adata_copy,
                    groupby=["perturbation", "donor"],
                    test_col="perturbation",
                    reference="control",
                )
                wall = time.perf_counter() - t0
                inmem_times.append(wall)
                if run == 0:
                    inmem_result_df = df
                print(f"    Run {run+1}: {wall:.2f}s")
            except ImportError as e:
                print(f"    pydeseq2 not installed: {e}")
                break
            except Exception as e:
                print(f"    Run {run+1} failed: {e}")
                break
            del adata_copy
            gc.collect()

        # Backed pseudobulk
        print(f"  Backed pseudobulk DE on {ds_name} ({n_runs} runs)...")
        backed = load_backed_adata(ds_name)
        backed_times = []
        backed_result_df = None
        if backed is not None:
            ensure_perturbation_labels(backed)
            for run in range(n_runs):
                gc.collect()
                rss_before = get_rss_mb()
                t0 = time.perf_counter()
                try:
                    df = pyscx.accel.pseudobulk_dex(
                        backed,
                        groupby=["perturbation", "donor"],
                        test_col="perturbation",
                        reference="control",
                    )
                    wall = time.perf_counter() - t0
                    rss_after = get_rss_mb()
                    backed_times.append(wall)
                    if run == 0:
                        backed_result_df = df
                    print(f"    Run {run+1}: {wall:.2f}s, RSS delta: "
                          f"{rss_after - rss_before:.0f} MB")
                except Exception as e:
                    print(f"    Run {run+1} failed: {e}")
                    break
            del backed
            gc.collect()

        result = {
            "benchmark": "accel_pseudobulk",
            "dataset": ds_name,
            "n_obs": n_obs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if inmem_times:
            result["inmem_times_s"] = [round(t, 3) for t in inmem_times]
            result["inmem_median_s"] = round(_median(inmem_times), 3)

        if backed_times:
            result["backed_times_s"] = [round(t, 3) for t in backed_times]
            result["backed_median_s"] = round(_median(backed_times), 3)

        # Quality: log2FC correlation between backed and in-memory
        if inmem_result_df is not None and backed_result_df is not None:
            try:
                # Merge on gene name and compare log2FC
                merged = inmem_result_df.merge(
                    backed_result_df, on="gene", suffixes=("_inmem", "_backed")
                )
                if len(merged) > 10:
                    l2fc_corr = float(np.corrcoef(
                        merged["log2FoldChange_inmem"].values,
                        merged["log2FoldChange_backed"].values
                    )[0, 1])
                    result["l2fc_pearson_r"] = round(l2fc_corr, 4)
                    result["pass_l2fc_corr"] = l2fc_corr > 0.99

                    # Significant gene overlap
                    sig_inmem = set(
                        merged.loc[merged["padj_inmem"] < 0.05, "gene"]
                    )
                    sig_backed = set(
                        merged.loc[merged["padj_backed"] < 0.05, "gene"]
                    )
                    if sig_inmem:
                        sig_overlap = len(sig_inmem & sig_backed) / len(sig_inmem)
                        result["sig_gene_overlap"] = round(sig_overlap, 4)
                        result["pass_sig_overlap"] = sig_overlap > 0.95
                    print(f"  log2FC r: {l2fc_corr:.4f}, "
                          f"Sig overlap: {result.get('sig_gene_overlap', 'N/A')}")
            except Exception as e:
                print(f"  Quality comparison failed: {e}")

        results.append(result)
        del adata
        gc.collect()

    save_json(results, "accel_pseudobulk_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Stratified DE Benchmark (§3.10.5c)
# ──────────────────────────────────────────────────────────────────────────────

def run_stratified_benchmark(datasets=None, n_runs=N_RUNS):
    """Stratified DE: stratify_by for single-cell + pseudobulk at 100K."""
    import pyscx

    if datasets is None:
        datasets = STRATIFIED_DATASETS

    print(f"\n{'='*60}")
    print(f"Stratified DE Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        adata = load_preprocessed_adata(ds_name)
        if adata is None:
            continue

        n_obs = adata.n_obs
        ensure_perturbation_labels(adata)
        ensure_cell_type_column(adata, n_types=10)

        # Single-cell stratified DE
        print(f"\n  Single-cell stratified DE on {ds_name} ({n_runs} runs)...")
        sc_strat_times = []
        for run in range(n_runs):
            adata_copy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            try:
                df = pyscx.accel.rank_genes_groups(
                    adata_copy,
                    groupby="perturbation",
                    stratify_by=["cell_type"],
                    min_cells_per_stratum=50,
                )
                wall = time.perf_counter() - t0
                sc_strat_times.append(wall)
                print(f"    Run {run+1}: {wall:.2f}s")
            except Exception as e:
                print(f"    Run {run+1} failed: {e}")
                break
            del adata_copy
            gc.collect()

        # Pseudobulk stratified DE
        print(f"  Pseudobulk stratified DE on {ds_name} ({n_runs} runs)...")
        pb_strat_times = []
        for run in range(n_runs):
            adata_copy = adata.copy()
            gc.collect()
            t0 = time.perf_counter()
            try:
                df = pyscx.accel.pseudobulk_dex(
                    adata_copy,
                    groupby=["perturbation", "donor"],
                    test_col="perturbation",
                    reference="control",
                    stratify_by=["cell_type"],
                    min_cells_per_stratum=50,
                )
                wall = time.perf_counter() - t0
                pb_strat_times.append(wall)
                print(f"    Run {run+1}: {wall:.2f}s")
            except ImportError as e:
                print(f"    pydeseq2 not installed: {e}")
                break
            except Exception as e:
                print(f"    Run {run+1} failed: {e}")
                break
            del adata_copy
            gc.collect()

        result = {
            "benchmark": "accel_stratified",
            "dataset": ds_name,
            "n_obs": n_obs,
            "n_strata": int(adata.obs["cell_type"].nunique()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        if sc_strat_times:
            result["sc_stratified_times_s"] = [round(t, 3) for t in sc_strat_times]
            result["sc_stratified_median_s"] = round(_median(sc_strat_times), 3)

        if pb_strat_times:
            result["pb_stratified_times_s"] = [round(t, 3) for t in pb_strat_times]
            result["pb_stratified_median_s"] = round(_median(pb_strat_times), 3)

        results.append(result)
        del adata
        gc.collect()

    save_json(results, "accel_stratified_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick Validation (pbmc3k)
# ──────────────────────────────────────────────────────────────────────────────

def run_validation():
    """Quick correctness validation on pbmc3k."""
    import pyscx
    import scanpy as sc
    from sklearn.manifold import trustworthiness

    ds_name = "pbmc3k"

    print(f"\n{'='*60}")
    print(f"Quick Validation — {ds_name}")
    print(f"{'='*60}")

    adata = load_preprocessed_adata(ds_name)
    if adata is None:
        print("  SKIP: pbmc3k not available")
        return []

    results = []

    # PCA validation
    print(f"\n  PCA validation...")
    adata_scx = adata.copy()
    pyscx.accel.pca(adata_scx, n_comps=50, device="cpu")
    adata_scanpy = adata.copy()
    sc.pp.pca(adata_scanpy, n_comps=50)
    sims = cosine_similarity_per_pc(adata_scx.obsm["X_pca"],
                                     adata_scanpy.obsm["X_pca"])
    pca_pass = float(np.min(sims)) > 0.99
    print(f"    Cosine sim min: {float(np.min(sims)):.4f} "
          f"({'PASS' if pca_pass else 'FAIL'})")
    results.append({"test": "pca_cosine_sim", "min": round(float(np.min(sims)), 4),
                     "pass": pca_pass})

    # kNN validation
    print(f"  kNN validation...")
    adata_knn = adata.copy()
    sc.pp.pca(adata_knn, n_comps=50)
    from sklearn.neighbors import NearestNeighbors
    X_pca = adata_knn.obsm["X_pca"].astype(np.float32)
    nn = NearestNeighbors(n_neighbors=16, metric="euclidean", algorithm="brute")
    nn.fit(X_pca)
    _, indices_exact = nn.kneighbors(X_pca)
    indices_exact = indices_exact[:, 1:]

    adata_knn_scx = adata_knn.copy()
    pyscx.accel.neighbors(adata_knn_scx, n_neighbors=15, device="cpu")
    scx_indices = get_knn_indices_from_adata(adata_knn_scx, 15)
    recall = compute_recall_at_k(scx_indices, indices_exact, 15)
    knn_pass = recall > 0.90
    print(f"    Recall@15: {recall:.4f} ({'PASS' if knn_pass else 'FAIL'})")
    results.append({"test": "knn_recall", "recall": round(recall, 4),
                     "pass": knn_pass})

    # UMAP validation
    print(f"  UMAP validation...")
    adata_umap = adata_knn_scx.copy()
    pyscx.accel.umap(adata_umap, device="cpu")
    trust = float(trustworthiness(X_pca, adata_umap.obsm["X_umap"], n_neighbors=15))
    umap_pass = trust > 0.90
    print(f"    Trustworthiness: {trust:.4f} ({'PASS' if umap_pass else 'FAIL'})")
    results.append({"test": "umap_trustworthiness", "value": round(trust, 4),
                     "pass": umap_pass})

    # DE validation
    print(f"  DE validation...")
    adata_de = adata.copy()
    groupby = ensure_groupby_column(adata_de, n_groups=5)
    pyscx.accel.rank_genes_groups(adata_de, groupby=groupby, reference="rest")
    adata_de_sc = adata.copy()
    adata_de_sc.obs[groupby] = adata_de.obs[groupby].values
    sc.tl.rank_genes_groups(adata_de_sc, groupby=groupby, method="wilcoxon")
    overlap = compute_de_gene_overlap(
        get_de_gene_names(adata_de, 100),
        get_de_gene_names(adata_de_sc, 100),
    )
    de_pass = overlap > 0.80
    print(f"    Gene overlap (top-100): {overlap:.4f} "
          f"({'PASS' if de_pass else 'FAIL'})")
    results.append({"test": "de_gene_overlap", "overlap": round(overlap, 4),
                     "pass": de_pass})

    all_pass = all(r["pass"] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    save_json(results, "accel_validation.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report():
    """Generate markdown report from existing JSON results."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Phase 4b Accelerator Benchmark Report",
        "",
        f"**Generated**: {ts}",
        f"**Benchmark**: SCX Rust-native CPU accelerators vs Scanpy",
        "",
    ]

    # PCA
    pca_path = RESULTS_DIR / "accel_pca_benchmark.json"
    if pca_path.exists():
        pca_data = json.loads(pca_path.read_text())
        lines += [
            "## 1. PCA: pyscx.accel.pca() vs sc.pp.pca()",
            "",
            "| Dataset | Cells | SCX (s) | Scanpy (s) | Speedup | Cosine Sim Min | Var Ratio r | Pass |",
            "|---------|-------|---------|------------|---------|----------------|-------------|------|",
        ]
        for r in pca_data:
            p = "Y" if r.get("pass_cosine") and r.get("pass_variance") else "N"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['scx_median_s']:.2f} | "
                f"{r['scanpy_median_s']:.2f} | {r['speedup']:.1f}x | "
                f"{r['cosine_sim_min']:.4f} | {r['variance_ratio_pearson_r']:.4f} | {p} |"
            )
        lines.append("")

    # kNN
    knn_path = RESULTS_DIR / "accel_knn_benchmark.json"
    if knn_path.exists():
        knn_data = json.loads(knn_path.read_text())
        lines += [
            "## 2. kNN: pyscx.accel.neighbors() vs sc.pp.neighbors()",
            "",
            "| Dataset | Cells | SCX (s) | Scanpy (s) | Speedup | Recall@15 SCX | Leiden ARI |",
            "|---------|-------|---------|------------|---------|---------------|------------|",
        ]
        for r in knn_data:
            recall = r.get("recall_scx_vs_exact", "N/A")
            if isinstance(recall, float):
                recall = f"{recall:.4f}"
            ari = r.get("leiden_ari", "N/A")
            if isinstance(ari, float):
                ari = f"{ari:.4f}"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['scx_median_s']:.2f} | "
                f"{r['scanpy_median_s']:.2f} | {r['speedup']:.1f}x | {recall} | {ari} |"
            )
        lines.append("")

    # UMAP
    umap_path = RESULTS_DIR / "accel_umap_benchmark.json"
    if umap_path.exists():
        umap_data = json.loads(umap_path.read_text())
        lines += [
            "## 3. UMAP: pyscx.accel.umap() vs sc.tl.umap()",
            "",
            "| Dataset | Cells | SCX (s) | Scanpy (s) | Speedup | Trust SCX | Trust Scanpy |",
            "|---------|-------|---------|------------|---------|-----------|--------------|",
        ]
        for r in umap_data:
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['scx_median_s']:.2f} | "
                f"{r['scanpy_median_s']:.2f} | {r['speedup']:.1f}x | "
                f"{r['trustworthiness_scx']:.4f} | {r['trustworthiness_scanpy']:.4f} |"
            )
        lines.append("")

    # DE In-Memory
    de_path = RESULTS_DIR / "accel_de_inmemory_benchmark.json"
    if de_path.exists():
        de_data = json.loads(de_path.read_text())
        lines += [
            "## 4. DE (In-Memory): pyscx.accel.rank_genes_groups() vs sc.tl.rank_genes_groups()",
            "",
            "| Dataset | Cells | SCX (s) | Scanpy (s) | Speedup | Gene Overlap | P-val r |",
            "|---------|-------|---------|------------|---------|--------------|---------|",
        ]
        for r in de_data:
            pval = r.get("pval_spearman_r", "N/A")
            if isinstance(pval, float):
                pval = f"{pval:.4f}"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['scx_median_s']:.2f} | "
                f"{r['scanpy_median_s']:.2f} | {r['speedup']:.1f}x | "
                f"{r['gene_overlap_top100']:.4f} | {pval} |"
            )
        lines.append("")

    # DE Streaming
    stream_path = RESULTS_DIR / "accel_de_streaming_benchmark.json"
    if stream_path.exists():
        stream_data = json.loads(stream_path.read_text())
        lines += [
            "## 5. DE (Streaming): gene-chunked on backed data",
            "",
            "| Dataset | Chunk Size | Time (s) | Gene Overlap vs In-Mem | Identical | Peak RSS (MB) |",
            "|---------|-----------|----------|------------------------|-----------|---------------|",
        ]
        for r in stream_data:
            lines.append(
                f"| {r['dataset']} | {r['gene_chunk_size']:,} | "
                f"{r['stream_median_s']:.2f} | "
                f"{r['gene_overlap_top50_vs_inmem']:.4f} | "
                f"{'Y' if r.get('pass_identical') else 'N'} | "
                f"{r.get('peak_rss_mb', 'N/A')} |"
            )
        lines.append("")

    # Pseudobulk
    pb_path = RESULTS_DIR / "accel_pseudobulk_benchmark.json"
    if pb_path.exists():
        pb_data = json.loads(pb_path.read_text())
        lines += [
            "## 6. Pseudobulk DE: pyscx.accel.pseudobulk_dex()",
            "",
            "| Dataset | Cells | In-Mem (s) | Backed (s) | log2FC r | Sig Overlap |",
            "|---------|-------|------------|------------|----------|-------------|",
        ]
        for r in pb_data:
            inmem = r.get("inmem_median_s", "N/A")
            backed = r.get("backed_median_s", "N/A")
            l2fc = r.get("l2fc_pearson_r", "N/A")
            sig = r.get("sig_gene_overlap", "N/A")
            if isinstance(inmem, float):
                inmem = f"{inmem:.2f}"
            if isinstance(backed, float):
                backed = f"{backed:.2f}"
            if isinstance(l2fc, float):
                l2fc = f"{l2fc:.4f}"
            if isinstance(sig, float):
                sig = f"{sig:.4f}"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {inmem} | {backed} | {l2fc} | {sig} |"
            )
        lines.append("")

    # Stratified
    strat_path = RESULTS_DIR / "accel_stratified_benchmark.json"
    if strat_path.exists():
        strat_data = json.loads(strat_path.read_text())
        lines += [
            "## 7. Stratified DE",
            "",
            "| Dataset | Cells | Strata | SC Stratified (s) | PB Stratified (s) |",
            "|---------|-------|--------|--------------------|--------------------|",
        ]
        for r in strat_data:
            sc_t = r.get("sc_stratified_median_s", "N/A")
            pb_t = r.get("pb_stratified_median_s", "N/A")
            if isinstance(sc_t, float):
                sc_t = f"{sc_t:.2f}"
            if isinstance(pb_t, float):
                pb_t = f"{pb_t:.2f}"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['n_strata']} | {sc_t} | {pb_t} |"
            )
        lines.append("")

    report = "\n".join(lines)
    md_path = RESULTS_DIR / "accel_benchmark.md"
    md_path.write_text(report)
    print(f"\nReport saved: {md_path}")
    print(report)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX Phase 4b Accelerator Benchmarks (CPU)"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["pca", "knn", "umap", "de", "de_streaming",
                 "pseudobulk", "stratified", "validate", "all", "report"],
        help="Benchmark mode (default: all)",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Override dataset list",
    )
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip pyscx release build check")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        ensure_release_build()

    mode = args.mode
    datasets = args.datasets
    n_runs = args.n_runs

    if mode == "report":
        generate_report()
        return

    if mode in ("validate", "all"):
        run_validation()

    if mode in ("pca", "all"):
        run_pca_benchmark(datasets=datasets, n_runs=n_runs)

    if mode in ("knn", "all"):
        run_knn_benchmark(datasets=datasets, n_runs=n_runs)

    if mode in ("umap", "all"):
        run_umap_benchmark(datasets=datasets, n_runs=n_runs)

    if mode in ("de", "all"):
        run_de_inmemory_benchmark(datasets=datasets, n_runs=n_runs)

    if mode in ("de_streaming", "all"):
        run_de_streaming_benchmark(
            datasets=datasets or DE_STREAMING_DATASETS, n_runs=n_runs
        )

    if mode in ("pseudobulk", "all"):
        run_pseudobulk_benchmark(
            datasets=datasets or PSEUDOBULK_DATASETS, n_runs=n_runs
        )

    if mode in ("stratified", "all"):
        run_stratified_benchmark(
            datasets=datasets or STRATIFIED_DATASETS, n_runs=n_runs
        )

    # Always generate report at the end
    generate_report()


if __name__ == "__main__":
    main()
