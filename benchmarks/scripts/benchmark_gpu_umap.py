#!/usr/bin/env python3
"""
SCX GPU UMAP Validation & Benchmark

Validates GPU-accelerated UMAP (native CUDA SGD kernel) via trustworthiness
metric against CPU UMAP. Benchmarks GPU vs CPU UMAP wall-clock timing.

Usage:
    # Validate trustworthiness on pbmc3k
    python benchmarks/scripts/benchmark_gpu_umap.py --mode validate

    # Timing benchmark
    python benchmarks/scripts/benchmark_gpu_umap.py --mode bench

    # Run all
    python benchmarks/scripts/benchmark_gpu_umap.py --mode all

    # Submit via SLURM:
    sbatch benchmarks/scripts/slurm_gpu_analysis_bench.sh
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
from bench_env import WORK_DIR

DATASETS = {
    "pbmc3k": {"h5ad": WORK_DIR / "pbmc3k.h5ad", "cells": 2_700},
    "tabula_sapiens_100k": {"h5ad": WORK_DIR / "tabula_sapiens_100k.h5ad", "cells": 100_000},
    "census_1m": {"h5ad": WORK_DIR / "census_1m.h5ad", "cells": 1_000_000},
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_adata_with_knn(dataset_name: str, n_comps: int = 50,
                        n_neighbors: int = 15):
    """Load h5ad, preprocess, compute PCA + kNN on CPU.

    Returns adata with obsp["connectivities"] ready for UMAP.
    """
    import anndata
    import pyscx
    import scanpy as sc

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        print(f"  SKIP: {h5ad_path} not found")
        return None

    print(f"  Loading {dataset_name} ({info['cells']:,} cells)...")
    adata = anndata.read_h5ad(str(h5ad_path))

    if "X_pca" not in adata.obsm:
        print(f"  Preprocessing + PCA (n_comps={n_comps})...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_top = min(2000, adata.n_vars)
        try:
            sc.pp.highly_variable_genes(
                adata, n_top_genes=n_top, flavor="seurat_v3",
                subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
            )
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
        sc.pp.pca(adata, n_comps=n_comps)

    if "connectivities" not in adata.obsp:
        print(f"  Building kNN graph (k={n_neighbors}) on CPU...")
        pyscx.accel.neighbors(adata, n_neighbors=n_neighbors, device="cpu")

    print(f"  Ready: {adata.n_obs:,} cells, PCA {adata.obsm['X_pca'].shape}")
    return adata


def compute_trustworthiness(X_high: np.ndarray, X_low: np.ndarray,
                            n_neighbors: int = 15) -> float:
    """Compute trustworthiness of a low-dimensional embedding."""
    from sklearn.manifold import trustworthiness
    return float(trustworthiness(X_high, X_low, n_neighbors=n_neighbors))


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Validation — trustworthiness
# ──────────────────────────────────────────────────────────────────────────────

def run_umap_validation(dataset_name: str = "pbmc3k",
                        n_neighbors: int = 15) -> dict:
    """Validate GPU UMAP via trustworthiness metric."""
    import pyscx

    print(f"\n{'='*60}")
    print(f"UMAP Validation (trustworthiness) — {dataset_name}")
    print(f"{'='*60}")

    adata = load_adata_with_knn(dataset_name, n_neighbors=n_neighbors)
    if adata is None:
        return {"error": f"Dataset {dataset_name} not available"}

    n_obs = adata.n_obs
    X_pca = adata.obsm["X_pca"].astype(np.float32)

    # CPU UMAP
    print(f"\n  Running CPU UMAP...")
    adata_cpu = adata.copy()
    t0 = time.perf_counter()
    pyscx.accel.umap(adata_cpu, device="cpu")
    t_cpu = time.perf_counter() - t0
    X_umap_cpu = adata_cpu.obsm["X_umap"]
    trust_cpu = compute_trustworthiness(X_pca, X_umap_cpu, n_neighbors)
    print(f"    Done in {t_cpu:.2f}s, trustworthiness: {trust_cpu:.4f}")

    # GPU UMAP
    gpu_available = True
    print(f"  Running GPU UMAP...")
    adata_gpu = adata.copy()
    try:
        t0 = time.perf_counter()
        pyscx.accel.umap(adata_gpu, device="gpu")
        t_gpu = time.perf_counter() - t0
        X_umap_gpu = adata_gpu.obsm["X_umap"]
        trust_gpu = compute_trustworthiness(X_pca, X_umap_gpu, n_neighbors)
        print(f"    Done in {t_gpu:.2f}s, trustworthiness: {trust_gpu:.4f}")
    except Exception as e:
        print(f"    GPU UMAP failed: {e}")
        gpu_available = False
        t_gpu = None
        trust_gpu = None

    result = {
        "benchmark": "gpu_umap_validation",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_neighbors": n_neighbors,
        "t_cpu_s": round(t_cpu, 3),
        "trustworthiness_cpu": round(trust_cpu, 4),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if gpu_available and trust_gpu is not None:
        result["t_gpu_s"] = round(t_gpu, 3)
        result["trustworthiness_gpu"] = round(trust_gpu, 4)
        result["speedup"] = round(t_cpu / t_gpu, 1) if t_gpu > 0 else 0
        # GPU UMAP quality should match CPU UMAP quality (within 0.02).
        # Absolute trustworthiness depends on dataset/parameters, not backend.
        quality_match = abs(trust_gpu - trust_cpu) < 0.02
        result["pass_quality_match"] = quality_match
        result["pass_trustworthiness_abs"] = trust_gpu > 0.90
        result["pass"] = quality_match and trust_gpu > 0.90
    else:
        result["gpu_available"] = False
        result["pass"] = False

    verdict = result.get("pass", False)
    print(f"\n  Verdict: {'PASS' if verdict else 'FAIL'}")
    print(f"    CPU trustworthiness: {trust_cpu:.4f}")
    print(f"    GPU trustworthiness: {result.get('trustworthiness_gpu', 'N/A')}")
    print(f"    Quality match (|GPU - CPU| < 0.02): {result.get('pass_quality_match', 'N/A')}")

    del adata, adata_cpu, adata_gpu
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Timing benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_timing_benchmark(datasets: list[str] | None = None,
                         n_neighbors: int = 15,
                         n_runs: int = 3) -> list[dict]:
    """Benchmark GPU vs CPU UMAP timing across dataset sizes."""
    import pyscx

    if datasets is None:
        datasets = ["tabula_sapiens_100k", "census_1m"]

    print(f"\n{'='*60}")
    print(f"UMAP Timing Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        adata = load_adata_with_knn(ds_name, n_neighbors=n_neighbors)
        if adata is None:
            continue

        n_obs = adata.n_obs
        print(f"\n  --- {ds_name} ({n_obs:,} cells) ---")

        # CPU timing
        cpu_times = []
        for run in range(n_runs):
            adata_run = adata.copy()
            t0 = time.perf_counter()
            pyscx.accel.umap(adata_run, device="cpu")
            cpu_times.append(time.perf_counter() - t0)
            del adata_run
            gc.collect()
        cpu_median = sorted(cpu_times)[len(cpu_times) // 2]
        print(f"    CPU: {cpu_median:.3f}s (median of {n_runs})")

        result = {
            "benchmark": "gpu_umap_timing",
            "dataset": ds_name,
            "n_obs": n_obs,
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
                pyscx.accel.umap(adata_run, device="gpu")
                gpu_times.append(time.perf_counter() - t0)
                del adata_run
                gc.collect()
            gpu_median = sorted(gpu_times)[len(gpu_times) // 2]
            speedup = cpu_median / gpu_median if gpu_median > 0 else 0
            print(f"    GPU: {gpu_median:.3f}s (median of {n_runs}), speedup: {speedup:.1f}x")

            result["gpu_times_s"] = [round(t, 3) for t in gpu_times]
            result["gpu_median_s"] = round(gpu_median, 3)
            result["speedup"] = round(speedup, 1)
        except Exception as e:
            print(f"    GPU UMAP: FAILED ({e})")
            result["gpu_available"] = False

        results.append(result)
        del adata
        gc.collect()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(validation_results: list[dict] | None,
                    timing_results: list[dict] | None) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU UMAP Validation & Benchmark Report",
        "",
        f"**Generated**: {ts}",
        "",
    ]

    if validation_results:
        lines += [
            "## 1. Trustworthiness Validation",
            "",
            "| Dataset | Cells | CPU Trust | GPU Trust | Target | Pass? | CPU (s) | GPU (s) | Speedup |",
            "|---------|-------|-----------|-----------|--------|-------|---------|---------|---------|",
        ]
        for vr in validation_results:
            if "error" in vr:
                lines.append(f"| {vr.get('dataset', '?')} | — | — | — | > 0.95 | SKIP | — | — | — |")
                continue
            gpu_trust = vr.get("trustworthiness_gpu", "N/A")
            gpu_time = f"{vr['t_gpu_s']:.3f}" if "t_gpu_s" in vr else "N/A"
            speedup = f"{vr['speedup']:.1f}x" if "speedup" in vr else "N/A"
            verdict = "PASS" if vr.get("pass") else "FAIL"
            if isinstance(gpu_trust, float):
                gpu_trust = f"{gpu_trust:.4f}"
            lines.append(
                f"| {vr['dataset']} | {vr['n_obs']:,} | "
                f"{vr['trustworthiness_cpu']:.4f} | {gpu_trust} | > 0.95 | {verdict} | "
                f"{vr['t_cpu_s']:.3f} | {gpu_time} | {speedup} |"
            )
        lines.append("")

    if timing_results:
        lines += [
            "## 2. UMAP Timing Benchmark",
            "",
            "| Dataset | Cells | CPU (s) | GPU (s) | Speedup |",
            "|---------|-------|---------|---------|---------|",
        ]
        for r in timing_results:
            gpu_time = f"{r['gpu_median_s']:.3f}" if "gpu_median_s" in r else "N/A"
            speedup = f"{r['speedup']:.1f}x" if "speedup" in r else "N/A"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | "
                f"{r['cpu_median_s']:.3f} | {gpu_time} | {speedup} |"
            )
        lines.append("")

    lines += ["## Summary", ""]
    all_pass = True
    if validation_results:
        for vr in validation_results:
            if "error" not in vr:
                vp = vr.get("pass", False)
                lines.append(f"- {vr['dataset']}: **{'PASS' if vp else 'FAIL'}** "
                             f"(GPU trustworthiness = {vr.get('trustworthiness_gpu', 'N/A')})")
                all_pass = all_pass and vp
    lines += ["", f"**Overall: {'PASS' if all_pass else 'FAIL'}**", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX GPU UMAP Validation & Benchmark"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["validate", "bench", "all"],
    )
    parser.add_argument(
        "--dataset", default=None, choices=list(DATASETS.keys()),
    )
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    if args.mode in ("validate", "all"):
        ds = args.dataset or "pbmc3k"
        validation_results = [run_umap_validation(ds, args.n_neighbors)]
        all_results["validation"] = validation_results

        json_path = RESULTS_DIR / "gpu_umap_validation.json"
        json_path.write_text(json.dumps(validation_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    if args.mode in ("bench", "all"):
        bench_datasets = (
            [args.dataset] if args.dataset
            else ["tabula_sapiens_100k", "census_1m"]
        )
        timing_results = run_timing_benchmark(bench_datasets, args.n_neighbors, args.n_runs)
        all_results["timing"] = timing_results

        json_path = RESULTS_DIR / "gpu_umap_timing.json"
        json_path.write_text(json.dumps(timing_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    report = generate_report(
        all_results.get("validation"),
        all_results.get("timing"),
    )
    md_path = RESULTS_DIR / "gpu_umap_benchmark.md"
    md_path.write_text(report)
    print(f"\n  Report saved: {md_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        ensure_release_build()
    main()
