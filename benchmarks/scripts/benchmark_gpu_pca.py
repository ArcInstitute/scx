#!/usr/bin/env python3
"""
SCX GPU PCA Validation & Benchmark

Validates GPU-accelerated PCA (cuSPARSE SpMM + cuSOLVER QR) against CPU PCA
via per-PC cosine similarity. Benchmarks GPU vs CPU PCA wall-clock timing
across dataset sizes.

Covers Phase4-GPU.md Steps 1 (SpMM benchmark) and 2 (PCA validation + benchmark).

Usage:
    # Validate cosine similarity on pbmc3k
    python benchmarks/scripts/benchmark_gpu_pca.py --mode validate --dataset pbmc3k

    # Timing benchmark across datasets
    python benchmarks/scripts/benchmark_gpu_pca.py --mode bench

    # Run all validations + benchmarks
    python benchmarks/scripts/benchmark_gpu_pca.py --mode all

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
    "census_5m": {"h5ad": WORK_DIR / "census_5m.h5ad", "cells": 5_000_000},
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_scx_file(dataset_name: str) -> Path:
    """Ensure an SCX file exists for the dataset, converting from h5ad if needed."""
    import pyscx

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    scx_path = h5ad_path.with_suffix(".scx")

    if scx_path.exists():
        return scx_path

    if not h5ad_path.exists():
        return None

    print(f"  Converting {dataset_name} h5ad → SCX...")
    import anndata
    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, str(scx_path))
    del adata
    gc.collect()
    print(f"    Saved: {scx_path}")
    return scx_path


def load_backed_adata(dataset_name: str):
    """Load dataset as SCX-backed AnnData (required for GPU PCA path)."""
    import pyscx

    scx_path = ensure_scx_file(dataset_name)
    if scx_path is None:
        print(f"  SKIP: {dataset_name} not available")
        return None

    print(f"  Loading {dataset_name} in backed mode...")
    exp = pyscx.open(str(scx_path))
    adata = exp.to_anndata()
    print(f"  Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata


def preprocess_for_pca(adata):
    """Run standard preprocessing (normalize, log1p, HVG) before PCA.

    For backed SCX data, we need to materialize X first for preprocessing,
    then write back to SCX. Instead, use scanpy to preprocess a copy and
    write the preprocessed version.
    """
    import scanpy as sc

    # If X is already float and small enough, preprocess in-place
    # For backed data, materialize first
    if hasattr(adata.X, 'backed'):
        # Read all data into memory for preprocessing
        import scipy.sparse as sp
        adata.X = adata.X[:]  # materialize
        if sp.issparse(adata.X):
            adata.X = adata.X.tocsr()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    n_top = min(2000, adata.n_vars)
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3",
            subset=True, span=0.3 if adata.n_obs < 10_000 else 1.0
        )
    except Exception:
        # Fallback: use default flavor if seurat_v3 fails
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)

    return adata


def load_preprocessed_backed(dataset_name: str, n_hvgs: int = 2000):
    """Load dataset, preprocess, convert to SCX, and open in backed mode.

    Returns a backed AnnData suitable for both GPU and CPU PCA paths.
    """
    import anndata
    import pyscx
    import tempfile

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        print(f"  SKIP: {h5ad_path} not found")
        return None, None

    print(f"  Loading {dataset_name} ({info['cells']:,} cells)...")
    adata = anndata.read_h5ad(str(h5ad_path))

    print(f"  Preprocessing (normalize, log1p, HVG)...")
    adata = preprocess_for_pca(adata)

    # Write preprocessed data to a temporary SCX file for backed access
    tmp_scx = WORK_DIR / f"{dataset_name}_preprocessed.scx"
    if not tmp_scx.exists():
        print(f"  Writing preprocessed SCX to {tmp_scx}...")
        pyscx.from_anndata(adata, str(tmp_scx))

    # Open in backed mode
    print(f"  Opening preprocessed SCX in backed mode...")
    backed_adata = pyscx.open(str(tmp_scx)).to_anndata()
    print(f"  Backed: {backed_adata.n_obs:,} cells × {backed_adata.n_vars:,} genes")

    return backed_adata, adata


def cosine_similarity_per_pc(pcs_a: np.ndarray, pcs_b: np.ndarray) -> np.ndarray:
    """Compute sign-invariant cosine similarity per principal component.

    PCA components are defined up to sign, so we use abs(cos_sim).
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Validation — cosine similarity GPU vs CPU PCA
# ──────────────────────────────────────────────────────────────────────────────

def run_pca_validation(dataset_name: str = "pbmc3k",
                       n_comps: int = 50) -> dict:
    """Validate GPU PCA vs CPU PCA via per-PC cosine similarity."""
    import pyscx

    print(f"\n{'='*60}")
    print(f"PCA Validation (cosine similarity) — {dataset_name}")
    print(f"{'='*60}")

    backed_adata, mem_adata = load_preprocessed_backed(dataset_name)
    if backed_adata is None:
        return {"error": f"Dataset {dataset_name} not available"}

    n_obs = backed_adata.n_obs
    n_vars = backed_adata.n_vars
    n_comps = min(n_comps, n_vars - 1)

    # CPU PCA (backed mode)
    print(f"\n  Running CPU PCA (n_comps={n_comps})...")
    adata_cpu = backed_adata  # backed mode for CPU
    t0 = time.perf_counter()
    pyscx.accel.pca(adata_cpu, n_comps=n_comps, device="cpu")
    t_cpu = time.perf_counter() - t0
    pcs_cpu = adata_cpu.obsm["X_pca"].copy()
    print(f"    Done in {t_cpu:.2f}s, shape: {pcs_cpu.shape}")

    # GPU PCA (backed mode)
    # Re-open backed adata for fresh GPU run
    backed_adata_gpu, _ = load_preprocessed_backed(dataset_name)
    gpu_available = True
    try:
        print(f"  Running GPU PCA (n_comps={n_comps})...")
        t0 = time.perf_counter()
        pyscx.accel.pca(backed_adata_gpu, n_comps=n_comps, device="gpu")
        t_gpu = time.perf_counter() - t0
        pcs_gpu = backed_adata_gpu.obsm["X_pca"].copy()
        print(f"    Done in {t_gpu:.2f}s, shape: {pcs_gpu.shape}")
    except Exception as e:
        print(f"    GPU PCA failed: {e}")
        gpu_available = False
        t_gpu = None
        pcs_gpu = None

    result = {
        "benchmark": "gpu_pca_validation",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "n_comps": n_comps,
        "t_cpu_s": round(t_cpu, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if gpu_available and pcs_gpu is not None:
        sims = cosine_similarity_per_pc(pcs_cpu, pcs_gpu)
        mean_sim = float(np.mean(sims))
        min_sim = float(np.min(sims))
        n_pass = int(np.sum(sims > 0.99))

        print(f"\n  Per-PC cosine similarity (GPU vs CPU):")
        print(f"    Mean: {mean_sim:.6f}")
        print(f"    Min:  {min_sim:.6f}")
        print(f"    PCs with cos_sim > 0.99: {n_pass}/{n_comps}")

        result["t_gpu_s"] = round(t_gpu, 3)
        result["speedup"] = round(t_cpu / t_gpu, 1) if t_gpu > 0 else 0
        result["cosine_sim_mean"] = round(mean_sim, 6)
        result["cosine_sim_min"] = round(min_sim, 6)
        result["cosine_sim_per_pc"] = [round(s, 6) for s in sims.tolist()]
        result["n_pcs_pass"] = n_pass
        result["pass"] = min_sim > 0.99
    else:
        result["gpu_available"] = False
        result["pass"] = False

    verdict = result.get("pass", False)
    print(f"\n  Verdict: {'PASS' if verdict else 'FAIL'}")
    print(f"    Min cosine sim: {result.get('cosine_sim_min', 'N/A')} (target > 0.99)")

    del backed_adata, backed_adata_gpu, mem_adata
    gc.collect()
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Timing benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_timing_benchmark(datasets: list[str] | None = None,
                         n_comps: int = 50,
                         n_runs: int = 3) -> list[dict]:
    """Benchmark GPU vs CPU PCA timing across dataset sizes."""
    import pyscx

    if datasets is None:
        datasets = ["tabula_sapiens_100k", "census_1m", "census_5m"]

    print(f"\n{'='*60}")
    print(f"PCA Timing Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        backed_adata, _ = load_preprocessed_backed(ds_name)
        if backed_adata is None:
            continue

        n_obs = backed_adata.n_obs
        n_vars = backed_adata.n_vars
        n_c = min(n_comps, n_vars - 1)
        print(f"\n  --- {ds_name} ({n_obs:,} cells × {n_vars:,} genes, {n_c} PCs) ---")

        # CPU timing
        cpu_times = []
        for run in range(n_runs):
            adata_run, _ = load_preprocessed_backed(ds_name)
            t0 = time.perf_counter()
            pyscx.accel.pca(adata_run, n_comps=n_c, device="cpu")
            cpu_times.append(time.perf_counter() - t0)
            del adata_run
            gc.collect()
        cpu_median = sorted(cpu_times)[len(cpu_times) // 2]
        print(f"    CPU: {cpu_median:.3f}s (median of {n_runs})")

        result = {
            "benchmark": "gpu_pca_timing",
            "dataset": ds_name,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_comps": n_c,
            "cpu_times_s": [round(t, 3) for t in cpu_times],
            "cpu_median_s": round(cpu_median, 3),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # GPU timing
        gpu_times = []
        try:
            for run in range(n_runs):
                adata_run, _ = load_preprocessed_backed(ds_name)
                t0 = time.perf_counter()
                pyscx.accel.pca(adata_run, n_comps=n_c, device="gpu")
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
            print(f"    GPU PCA: FAILED ({e})")
            result["gpu_available"] = False

        results.append(result)
        del backed_adata
        gc.collect()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(validation_results: list[dict] | None,
                    timing_results: list[dict] | None) -> str:
    """Generate markdown report."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU PCA Validation & Benchmark Report",
        "",
        f"**Generated**: {ts}",
        "",
    ]

    # ── Validation ──
    if validation_results:
        lines += [
            "## 1. Cosine Similarity Validation (GPU vs CPU PCA)",
            "",
        ]
        for vr in validation_results:
            if "error" in vr:
                lines.append(f"**{vr.get('dataset', '?')}**: SKIPPED ({vr['error']})")
                lines.append("")
                continue

            ds = vr["dataset"]
            lines += [
                f"### {ds} ({vr['n_obs']:,} cells x {vr['n_vars']:,} genes, {vr['n_comps']} PCs)",
                "",
            ]

            if vr.get("cosine_sim_mean") is not None:
                verdict = "PASS" if vr["pass"] else "FAIL"
                lines += [
                    f"| Metric | Value | Target | Pass? |",
                    f"|--------|-------|--------|-------|",
                    f"| Mean cosine sim | {vr['cosine_sim_mean']:.6f} | > 0.99 | — |",
                    f"| Min cosine sim | {vr['cosine_sim_min']:.6f} | > 0.99 | {verdict} |",
                    f"| PCs passing | {vr['n_pcs_pass']}/{vr['n_comps']} | all | — |",
                    f"| CPU time | {vr['t_cpu_s']:.3f}s | — | — |",
                    f"| GPU time | {vr['t_gpu_s']:.3f}s | — | — |",
                    f"| Speedup | {vr.get('speedup', 'N/A')}x | — | — |",
                    "",
                ]
            else:
                lines += ["GPU unavailable", ""]

    # ── Timing ──
    if timing_results:
        lines += [
            "## 2. PCA Timing Benchmark",
            "",
            "| Dataset | Cells | Genes | PCs | CPU (s) | GPU (s) | Speedup |",
            "|---------|-------|-------|-----|---------|---------|---------|",
        ]
        for r in timing_results:
            gpu_time = f"{r['gpu_median_s']:.3f}" if "gpu_median_s" in r else "N/A"
            speedup = f"{r['speedup']:.1f}x" if "speedup" in r else "N/A"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['n_vars']:,} | "
                f"{r['n_comps']} | {r['cpu_median_s']:.3f} | {gpu_time} | {speedup} |"
            )
        lines.append("")

    # ── Summary ──
    lines += ["## Summary", ""]
    all_pass = True
    if validation_results:
        for vr in validation_results:
            if "error" not in vr:
                vp = vr.get("pass", False)
                lines.append(f"- {vr['dataset']}: **{'PASS' if vp else 'FAIL'}** "
                             f"(min cosine sim = {vr.get('cosine_sim_min', 'N/A')})")
                all_pass = all_pass and vp
    if timing_results:
        lines.append(f"- Timing: {len(timing_results)} datasets benchmarked")

    lines += ["", f"**Overall: {'PASS' if all_pass else 'FAIL'}**", ""]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX GPU PCA Validation & Benchmark"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["validate", "bench", "all"],
        help="Mode (default: all)",
    )
    parser.add_argument(
        "--dataset", default=None, choices=list(DATASETS.keys()),
        help="Dataset for validation (default: pbmc3k + census_1m)",
    )
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ── Validation ──
    if args.mode in ("validate", "all"):
        val_datasets = [args.dataset] if args.dataset else ["pbmc3k", "census_1m"]
        validation_results = []
        for ds in val_datasets:
            result = run_pca_validation(ds, args.n_comps)
            validation_results.append(result)
        all_results["validation"] = validation_results

        json_path = RESULTS_DIR / "gpu_pca_validation.json"
        json_path.write_text(json.dumps(validation_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    # ── Timing ──
    if args.mode in ("bench", "all"):
        bench_datasets = (
            [args.dataset] if args.dataset
            else ["tabula_sapiens_100k", "census_1m", "census_5m"]
        )
        timing_results = run_timing_benchmark(bench_datasets, args.n_comps, args.n_runs)
        all_results["timing"] = timing_results

        json_path = RESULTS_DIR / "gpu_pca_timing.json"
        json_path.write_text(json.dumps(timing_results, indent=2))
        print(f"\n  JSON saved: {json_path}")

    # ── Report ──
    report = generate_report(
        all_results.get("validation"),
        all_results.get("timing"),
    )
    md_path = RESULTS_DIR / "gpu_pca_benchmark.md"
    md_path.write_text(report)
    print(f"\n  Report saved: {md_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        ensure_release_build()
    main()
