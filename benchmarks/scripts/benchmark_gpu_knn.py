#!/usr/bin/env python3
"""
SCX GPU kNN (CAGRA) Validation & Benchmark

Validates GPU-accelerated kNN via cuVS CAGRA against CPU HNSW and exact
brute-force kNN. Measures recall@k, Leiden clustering agreement, and
wall-clock timing across dataset sizes.

Usage:
    # Quick recall validation on pbmc3k
    python benchmarks/scripts/benchmark_gpu_knn.py --mode recall --dataset pbmc3k

    # Leiden agreement on tabula_sapiens_100k
    python benchmarks/scripts/benchmark_gpu_knn.py --mode leiden --dataset tabula_sapiens_100k

    # Timing benchmark across all available datasets
    python benchmarks/scripts/benchmark_gpu_knn.py --mode bench

    # Run all validations + benchmarks
    python benchmarks/scripts/benchmark_gpu_knn.py --mode all

    # Submit via SLURM:
    sbatch benchmarks/scripts/slurm_gpu_knn_bench.sh
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
    "census_1m": {"h5ad": DATA_DIR / "census_1m.h5ad", "cells": 1_000_000},
    "census_5m": {"h5ad": DATA_DIR / "census_5m.h5ad", "cells": 5_000_000},
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_adata_with_pca(dataset_name: str, n_comps: int = 50):
    """Load h5ad dataset and compute PCA if not already present."""
    import anndata
    import scanpy as sc

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        print(f"  SKIP: {h5ad_path} not found")
        return None

    print(f"  Loading {dataset_name} ({info['cells']:,} cells)...")
    adata = anndata.read_h5ad(str(h5ad_path))

    # Run standard preprocessing if PCA not present
    if "X_pca" not in adata.obsm:
        print(f"  Running preprocessing + PCA (n_comps={n_comps})...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_top = min(2000, adata.n_vars)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top, flavor="seurat_v3",
                subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
            )
        except (ImportError, Exception):
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
        sc.pp.pca(adata, n_comps=n_comps)

    print(f"  Loaded: {adata.n_obs:,} cells, PCA shape {adata.obsm['X_pca'].shape}")
    return adata


def compute_recall_at_k(indices_approx, indices_exact, k: int) -> float:
    """Fraction of exact kNN captured by approximate kNN, averaged over all points."""
    n_obs = len(indices_approx)
    total_recall = 0.0
    for i in range(n_obs):
        approx_set = set(indices_approx[i][:k])
        exact_set = set(indices_exact[i][:k])
        total_recall += len(approx_set & exact_set) / k
    return total_recall / n_obs


def exact_knn(X: np.ndarray, k: int):
    """Compute exact brute-force kNN using sklearn."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="brute")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # Remove self-hits (first column)
    return distances[:, 1:], indices[:, 1:]


def get_knn_indices_from_adata(adata, n_neighbors: int):
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
        # Sort by distance, take top-k
        sorted_idx = np.argsort(row_dists)[:n_neighbors]
        n_fill = min(len(sorted_idx), n_neighbors)
        indices[i, :n_fill] = row_indices[sorted_idx[:n_fill]]
    return indices


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Recall@k validation
# ──────────────────────────────────────────────────────────────────────────────

def run_recall_validation(dataset_name: str = "pbmc3k",
                          n_neighbors: int = 15) -> dict:
    """Validate GPU kNN recall vs exact brute-force and CPU HNSW."""
    import pyscx

    print(f"\n{'='*60}")
    print(f"Recall@{n_neighbors} Validation — {dataset_name}")
    print(f"{'='*60}")

    adata = load_adata_with_pca(dataset_name)
    if adata is None:
        return {"error": f"Dataset {dataset_name} not available"}

    X_pca = adata.obsm["X_pca"].astype(np.float32)
    n_obs = X_pca.shape[0]

    # 1. Exact brute-force kNN (ground truth)
    print(f"\n  Computing exact brute-force kNN (n={n_obs:,}, k={n_neighbors})...")
    t0 = time.perf_counter()
    _, indices_exact = exact_knn(X_pca, n_neighbors)
    t_exact = time.perf_counter() - t0
    print(f"    Done in {t_exact:.2f}s")

    # 2. CPU HNSW kNN
    print(f"  Computing CPU HNSW kNN...")
    adata_cpu = adata.copy()
    t0 = time.perf_counter()
    pyscx.accel.neighbors(adata_cpu, n_neighbors=n_neighbors, device="cpu")
    t_cpu = time.perf_counter() - t0
    indices_cpu = get_knn_indices_from_adata(adata_cpu, n_neighbors)
    print(f"    Done in {t_cpu:.2f}s")

    # 3. GPU CAGRA kNN
    print(f"  Computing GPU CAGRA kNN...")
    adata_gpu = adata.copy()
    try:
        t0 = time.perf_counter()
        pyscx.accel.neighbors(adata_gpu, n_neighbors=n_neighbors, device="gpu")
        t_gpu = time.perf_counter() - t0
        indices_gpu = get_knn_indices_from_adata(adata_gpu, n_neighbors)
        gpu_available = True
        print(f"    Done in {t_gpu:.2f}s")
    except Exception as e:
        print(f"    GPU kNN failed: {e}")
        gpu_available = False
        t_gpu = None
        indices_gpu = None

    # 4. Compute recall metrics
    recall_cpu_vs_exact = compute_recall_at_k(indices_cpu, indices_exact, n_neighbors)
    print(f"\n  Recall@{n_neighbors} (CPU HNSW vs exact): {recall_cpu_vs_exact:.4f}")

    result = {
        "benchmark": "gpu_knn_recall",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_neighbors": n_neighbors,
        "t_exact_s": round(t_exact, 3),
        "t_cpu_s": round(t_cpu, 3),
        "recall_cpu_vs_exact": round(recall_cpu_vs_exact, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if gpu_available:
        recall_gpu_vs_exact = compute_recall_at_k(indices_gpu, indices_exact, n_neighbors)
        recall_gpu_vs_cpu = compute_recall_at_k(indices_gpu, indices_cpu, n_neighbors)
        print(f"  Recall@{n_neighbors} (GPU CAGRA vs exact): {recall_gpu_vs_exact:.4f}")
        print(f"  Recall@{n_neighbors} (GPU CAGRA vs CPU HNSW): {recall_gpu_vs_cpu:.4f}")

        result["t_gpu_s"] = round(t_gpu, 3)
        result["recall_gpu_vs_exact"] = round(recall_gpu_vs_exact, 4)
        result["recall_gpu_vs_cpu"] = round(recall_gpu_vs_cpu, 4)
        result["pass_recall_gpu_vs_exact"] = recall_gpu_vs_exact > 0.95
        result["pass_recall_gpu_vs_cpu"] = recall_gpu_vs_cpu > 0.90
    else:
        result["gpu_available"] = False
        result["pass_recall_gpu_vs_exact"] = False

    # Verdict
    pass_all = result.get("pass_recall_gpu_vs_exact", False)
    result["pass"] = pass_all

    print(f"\n  Verdict: {'PASS ✅' if pass_all else 'FAIL ❌'}")
    print(f"    recall@{n_neighbors} (GPU vs exact): "
          f"{result.get('recall_gpu_vs_exact', 'N/A')} (target > 0.95)")

    del adata, adata_cpu, adata_gpu
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Leiden agreement validation
# ──────────────────────────────────────────────────────────────────────────────

def run_leiden_validation(dataset_name: str = "tabula_sapiens_100k",
                          n_neighbors: int = 15) -> dict:
    """Validate that GPU and CPU kNN graphs produce comparable Leiden clusters."""
    import pyscx
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    print(f"\n{'='*60}")
    print(f"Leiden Agreement Validation — {dataset_name}")
    print(f"{'='*60}")

    adata = load_adata_with_pca(dataset_name)
    if adata is None:
        return {"error": f"Dataset {dataset_name} not available"}

    n_obs = adata.n_obs

    # CPU path
    print(f"\n  Building CPU kNN graph + Leiden...")
    adata_cpu = adata.copy()
    t0 = time.perf_counter()
    pyscx.accel.neighbors(adata_cpu, n_neighbors=n_neighbors, device="cpu")
    t_cpu_knn = time.perf_counter() - t0
    sc.tl.leiden(adata_cpu, resolution=1.0)
    t_cpu_total = time.perf_counter() - t0
    print(f"    kNN: {t_cpu_knn:.2f}s, total (kNN+Leiden): {t_cpu_total:.2f}s")
    print(f"    Clusters: {adata_cpu.obs['leiden'].nunique()}")

    # GPU path
    print(f"  Building GPU kNN graph + Leiden...")
    adata_gpu = adata.copy()
    try:
        t0 = time.perf_counter()
        pyscx.accel.neighbors(adata_gpu, n_neighbors=n_neighbors, device="gpu")
        t_gpu_knn = time.perf_counter() - t0
        sc.tl.leiden(adata_gpu, resolution=1.0)
        t_gpu_total = time.perf_counter() - t0
        gpu_available = True
        print(f"    kNN: {t_gpu_knn:.2f}s, total (kNN+Leiden): {t_gpu_total:.2f}s")
        print(f"    Clusters: {adata_gpu.obs['leiden'].nunique()}")
    except Exception as e:
        print(f"    GPU kNN failed: {e}")
        gpu_available = False
        t_gpu_knn = t_gpu_total = None

    result = {
        "benchmark": "gpu_knn_leiden",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_neighbors": n_neighbors,
        "t_cpu_knn_s": round(t_cpu_knn, 3),
        "t_cpu_total_s": round(t_cpu_total, 3),
        "n_clusters_cpu": int(adata_cpu.obs["leiden"].nunique()),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if gpu_available:
        ari = adjusted_rand_score(adata_cpu.obs["leiden"], adata_gpu.obs["leiden"])
        nmi = normalized_mutual_info_score(adata_cpu.obs["leiden"], adata_gpu.obs["leiden"])

        print(f"\n  Agreement:")
        print(f"    ARI: {ari:.4f} (target > 0.90)")
        print(f"    NMI: {nmi:.4f} (target > 0.85)")

        result["t_gpu_knn_s"] = round(t_gpu_knn, 3)
        result["t_gpu_total_s"] = round(t_gpu_total, 3)
        result["n_clusters_gpu"] = int(adata_gpu.obs["leiden"].nunique())
        result["ari"] = round(ari, 4)
        result["nmi"] = round(nmi, 4)
        result["pass_ari"] = ari > 0.90
        result["pass_nmi"] = nmi > 0.85
        result["pass"] = ari > 0.90 and nmi > 0.85
    else:
        result["gpu_available"] = False
        result["pass"] = False

    verdict = result.get("pass", False)
    print(f"\n  Verdict: {'PASS ✅' if verdict else 'FAIL ❌'}")

    del adata, adata_cpu
    if gpu_available:
        del adata_gpu
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: Timing benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_timing_benchmark(datasets: list[str] | None = None,
                         n_neighbors: int = 15,
                         n_runs: int = 3) -> list[dict]:
    """Benchmark GPU vs CPU kNN timing across dataset sizes."""
    import pyscx

    if datasets is None:
        datasets = ["tabula_sapiens_100k", "census_1m", "census_5m"]

    print(f"\n{'='*60}")
    print(f"kNN Timing Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        adata = load_adata_with_pca(ds_name)
        if adata is None:
            continue

        n_obs = adata.n_obs
        print(f"\n  --- {ds_name} ({n_obs:,} cells) ---")

        # CPU timing
        cpu_times = []
        for run in range(n_runs):
            adata_run = adata.copy()
            t0 = time.perf_counter()
            pyscx.accel.neighbors(adata_run, n_neighbors=n_neighbors, device="cpu")
            cpu_times.append(time.perf_counter() - t0)
            del adata_run
            gc.collect()
        cpu_median = sorted(cpu_times)[len(cpu_times) // 2]
        print(f"    CPU HNSW: {cpu_median:.3f}s (median of {n_runs})")

        result = {
            "benchmark": "gpu_knn_timing",
            "dataset": ds_name,
            "n_obs": n_obs,
            "n_neighbors": n_neighbors,
            "cpu_times_s": [round(t, 3) for t in cpu_times],
            "cpu_median_s": round(cpu_median, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # GPU timing
        gpu_times = []
        try:
            for run in range(n_runs):
                adata_run = adata.copy()
                t0 = time.perf_counter()
                pyscx.accel.neighbors(adata_run, n_neighbors=n_neighbors, device="gpu")
                gpu_times.append(time.perf_counter() - t0)
                del adata_run
                gc.collect()
            gpu_median = sorted(gpu_times)[len(gpu_times) // 2]
            speedup = cpu_median / gpu_median if gpu_median > 0 else 0
            print(f"    GPU CAGRA: {gpu_median:.3f}s (median of {n_runs}), speedup: {speedup:.1f}x")

            result["gpu_times_s"] = [round(t, 3) for t in gpu_times]
            result["gpu_median_s"] = round(gpu_median, 3)
            result["speedup"] = round(speedup, 1)
        except Exception as e:
            print(f"    GPU CAGRA: FAILED ({e})")
            result["gpu_available"] = False

        results.append(result)
        del adata
        gc.collect()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(recall_result: dict | None,
                    leiden_result: dict | None,
                    timing_results: list[dict] | None) -> str:
    """Generate markdown report from all validation + benchmark results."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU kNN (CAGRA) Validation & Benchmark Report",
        "",
        f"**Generated**: {ts}",
        "",
    ]

    # ── Recall ──
    if recall_result and "error" not in recall_result:
        lines += [
            "## 1. Recall@k Validation",
            "",
            f"**Dataset**: {recall_result['dataset']} ({recall_result['n_obs']:,} cells)",
            f"**k**: {recall_result['n_neighbors']}",
            "",
            "| Method | Recall vs Exact | Time (s) | Target | Pass? |",
            "|--------|----------------|----------|--------|-------|",
        ]
        lines.append(
            f"| CPU HNSW | {recall_result['recall_cpu_vs_exact']:.4f} "
            f"| {recall_result['t_cpu_s']:.3f} | — | — |"
        )
        if recall_result.get("recall_gpu_vs_exact") is not None:
            gpu_pass = "PASS ✅" if recall_result["pass_recall_gpu_vs_exact"] else "FAIL ❌"
            lines.append(
                f"| GPU CAGRA | {recall_result['recall_gpu_vs_exact']:.4f} "
                f"| {recall_result['t_gpu_s']:.3f} | > 0.95 | {gpu_pass} |"
            )
        else:
            lines.append("| GPU CAGRA | N/A (cuVS unavailable) | — | > 0.95 | SKIP |")
        lines.append(
            f"| Exact brute-force | 1.0000 "
            f"| {recall_result['t_exact_s']:.3f} | — | — |"
        )
        if recall_result.get("recall_gpu_vs_cpu") is not None:
            lines += [
                "",
                f"GPU CAGRA vs CPU HNSW recall@{recall_result['n_neighbors']}: "
                f"**{recall_result['recall_gpu_vs_cpu']:.4f}** (target > 0.90)",
            ]
        lines.append("")

    # ── Leiden ──
    if leiden_result and "error" not in leiden_result:
        lines += [
            "## 2. Leiden Clustering Agreement",
            "",
            f"**Dataset**: {leiden_result['dataset']} ({leiden_result['n_obs']:,} cells)",
            "",
            "| Metric | Value | Target | Pass? |",
            "|--------|-------|--------|-------|",
        ]
        if leiden_result.get("ari") is not None:
            ari_pass = "PASS ✅" if leiden_result["pass_ari"] else "FAIL ❌"
            nmi_pass = "PASS ✅" if leiden_result["pass_nmi"] else "FAIL ❌"
            lines.append(f"| ARI | {leiden_result['ari']:.4f} | > 0.90 | {ari_pass} |")
            lines.append(f"| NMI | {leiden_result['nmi']:.4f} | > 0.85 | {nmi_pass} |")
            lines.append(
                f"| Clusters (CPU/GPU) | "
                f"{leiden_result['n_clusters_cpu']}/{leiden_result['n_clusters_gpu']} | — | — |"
            )
        else:
            lines.append("| ARI | N/A (GPU unavailable) | > 0.90 | SKIP |")
        lines.append("")

    # ── Timing ──
    if timing_results:
        lines += [
            "## 3. Timing Benchmark",
            "",
            "| Dataset | Cells | CPU HNSW (s) | GPU CAGRA (s) | Speedup |",
            "|---------|-------|-------------|--------------|---------|",
        ]
        for r in timing_results:
            gpu_time = f"{r['gpu_median_s']:.3f}" if "gpu_median_s" in r else "N/A"
            speedup = f"{r['speedup']:.1f}x" if "speedup" in r else "N/A"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | "
                f"{r['cpu_median_s']:.3f} | {gpu_time} | {speedup} |"
            )
        lines.append("")

    # ── Summary ──
    lines += ["## Summary", ""]
    all_pass = True
    if recall_result and "error" not in recall_result:
        rp = recall_result.get("pass", False)
        lines.append(f"- Recall@k: **{'PASS ✅' if rp else 'FAIL ❌'}**")
        all_pass = all_pass and rp
    if leiden_result and "error" not in leiden_result:
        lp = leiden_result.get("pass", False)
        lines.append(f"- Leiden agreement: **{'PASS ✅' if lp else 'FAIL ❌'}**")
        all_pass = all_pass and lp
    if timing_results:
        lines.append(f"- Timing: {len(timing_results)} datasets benchmarked")

    lines += [
        "",
        f"**Overall: {'PASS ✅' if all_pass else 'FAIL ❌'}**",
        "",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX GPU kNN (CAGRA) Validation & Benchmark"
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=["recall", "leiden", "bench", "all"],
        help="Validation mode (default: all)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASETS.keys()),
        help="Dataset for recall/leiden (default: pbmc3k for recall, tabula_sapiens_100k for leiden)",
    )
    parser.add_argument(
        "--n-neighbors", type=int, default=15, help="Number of neighbors (default: 15)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=3, help="Number of timing runs (default: 3)"
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Recall ──
    if args.mode in ("recall", "all"):
        ds = args.dataset or "pbmc3k"
        recall_result = run_recall_validation(ds, args.n_neighbors)
        all_results["recall"] = recall_result

        json_path = RESULTS_DIR / "gpu_knn_recall.json"
        json_path.write_text(json.dumps(recall_result, indent=2))
        print(f"\n  JSON saved: {json_path}")

    # ── Leiden ──
    if args.mode in ("leiden", "all"):
        ds = args.dataset or "tabula_sapiens_100k"
        leiden_result = run_leiden_validation(ds, args.n_neighbors)
        all_results["leiden"] = leiden_result

        json_path = RESULTS_DIR / "gpu_knn_leiden.json"
        json_path.write_text(json.dumps(leiden_result, indent=2))
        print(f"\n  JSON saved: {json_path}")

    # ── Timing ──
    if args.mode in ("bench", "all"):
        bench_datasets = (
            [args.dataset] if args.dataset
            else ["tabula_sapiens_100k", "census_1m", "census_5m"]
        )
        timing_results = run_timing_benchmark(bench_datasets, args.n_neighbors, args.n_runs)
        all_results["timing"] = timing_results

        json_path = RESULTS_DIR / "gpu_knn_timing.json"
        json_path.write_text(json.dumps(timing_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    # ── Report ──
    report = generate_report(
        all_results.get("recall"),
        all_results.get("leiden"),
        all_results.get("timing"),
    )
    md_path = RESULTS_DIR / "gpu_knn_benchmark.md"
    md_path.write_text(report)
    print(f"\n  Report saved: {md_path}")

    # Print report to stdout
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    # Only auto-build if not running under SLURM (SLURM script builds with --features gpu)
    if "SLURM_JOB_ID" not in os.environ:
        ensure_release_build()
    main()
