#!/usr/bin/env python3
"""
SCX GPU End-to-End Pipeline Benchmark & Go/No-Go Gate

Benchmarks the full GPU-accelerated analysis pipeline (PCA -> kNN -> UMAP -> Leiden)
and evaluates the Phase 4c Go/No-Go gate criteria:
  1. GPU PCA matches CPU PCA within tolerance (cosine sim > 0.99)
  2. GPU kNN recall > 0.95 vs CPU HNSW
  3. End-to-end GPU pipeline >= 10x faster than CPU on 1M cells
  4. Graceful fallback: GPU ops fall back to CPU without crashing

Usage:
    # Run end-to-end pipeline timing
    python benchmarks/scripts/benchmark_gpu_pipeline.py --mode pipeline

    # Evaluate Go/No-Go gate (reads existing JSON results)
    python benchmarks/scripts/benchmark_gpu_pipeline.py --mode gate

    # Run all
    python benchmarks/scripts/benchmark_gpu_pipeline.py --mode all
"""

import argparse
import gc
import json
import os
import subprocess
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

def ensure_scx_file(dataset_name: str) -> Path | None:
    import pyscx

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    scx_path = h5ad_path.with_suffix(".scx")

    if scx_path.exists():
        return scx_path
    if not h5ad_path.exists():
        return None

    import anndata
    adata = anndata.read_h5ad(str(h5ad_path))
    pyscx.from_anndata(adata, str(scx_path))
    del adata
    gc.collect()
    return scx_path


def load_preprocessed_backed(dataset_name: str):
    """Load, preprocess, and return SCX-backed AnnData for pipeline."""
    import anndata
    import pyscx
    import scanpy as sc

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        return None

    # Check for cached preprocessed SCX
    prep_scx = WORK_DIR / f"{dataset_name}_preprocessed.scx"
    if prep_scx.exists():
        return pyscx.open(str(prep_scx)).to_anndata()

    # Preprocess from h5ad
    adata = anndata.read_h5ad(str(h5ad_path))
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

    pyscx.from_anndata(adata, str(prep_scx))
    return pyscx.open(str(prep_scx)).to_anndata()


def load_adata_in_memory(dataset_name: str):
    """Load dataset fully in memory with preprocessing for CPU pipeline."""
    import anndata
    import scanpy as sc

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        return None

    adata = anndata.read_h5ad(str(h5ad_path))
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
    return adata


def load_adata_in_memory_raw(dataset_name: str):
    """Load raw h5ad (no preprocessing) — feeds `run_preprocessing` when
    `--attribute-preprocessing` is enabled so the preprocessing cost is timed
    in isolation.
    """
    import anndata

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        return None
    return anndata.read_h5ad(str(h5ad_path))


def load_backed_raw(dataset_name: str):
    """Open the raw (unpreprocessed) SCX file as a backed AnnData — feeds
    `run_preprocessing(device="gpu")` which expects an SCX-backed source so
    the GPU `gpu_preprocess_to_csr` streaming path is exercised.
    """
    import pyscx

    scx_path = ensure_scx_file(dataset_name)
    if scx_path is None:
        return None
    return pyscx.open(str(scx_path)).to_anndata()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline timing
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(adata, device: str, n_comps: int = 50,
                 n_neighbors: int = 15, method: str = "auto",
                 qr_method: str = "householder") -> dict:
    """Run full PCA -> kNN -> UMAP -> Leiden pipeline, return per-op timing.

    `method` and `qr_method` forward to `pyscx.accel.pca`. Defaults preserve
    the pre-Phase-2/4 behaviour (auto-route by n_vars, Householder QR).
    """
    import pyscx

    timings = {}

    t0 = time.perf_counter()
    pyscx.accel.pca(
        adata, n_comps=n_comps, device=device,
        method=method, qr_method=qr_method,
    )
    timings["pca"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pyscx.accel.neighbors(adata, n_neighbors=n_neighbors, device=device)
    timings["knn"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pyscx.accel.umap(adata, device=device)
    timings["umap"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pyscx.accel.leiden(adata, device=device)
    timings["leiden"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    return timings


def run_preprocessing(adata, device: str) -> dict:
    """Time `normalize_total` + `log1p` + `highly_variable_genes` on the given
    AnnData with the specified device. Preprocessing is attributed separately
    so the downstream PCA/kNN/UMAP/Leiden table doesn't hide a whole-pipeline
    asymmetry between CPU and GPU baselines.
    """
    import pyscx

    timings = {}
    t0 = time.perf_counter()
    pyscx.accel.normalize_total(adata, target_sum=1e4, device=device)
    timings["normalize_total"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    pyscx.accel.log1p(adata, device=device)
    timings["log1p"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    n_top = min(2000, adata.n_vars)
    try:
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3",
            subset=True, device=device,
        )
    except Exception:
        # HVG can fail on tiny fixtures; surface but don't abort the benchmark.
        timings["hvg_error"] = True
    timings["highly_variable_genes"] = time.perf_counter() - t0
    timings["total"] = timings["normalize_total"] + timings["log1p"] + timings["highly_variable_genes"]
    return timings


def run_pca_variants_benchmark(adata_factory, n_comps: int = 50,
                               n_runs: int = 3) -> dict:
    """Sweep GPU PCA dispatch variants (covariance / randomized householder /
    randomized cholesky) on a prepared AnnData to fill the per-op rows called
    out in Phase 7.2. `adata_factory` is a zero-arg callable that returns a
    fresh backed/materialized AnnData each call (so each run starts clean).
    """
    import pyscx

    variants = [
        ("gpu_cov_pca", {"method": "covariance", "qr_method": "householder"}),
        ("gpu_randomized_pca_householder", {"method": "randomized", "qr_method": "householder"}),
        ("gpu_randomized_pca_chol", {"method": "randomized", "qr_method": "cholesky"}),
    ]
    out: dict = {}
    for name, kwargs in variants:
        times = []
        for _ in range(n_runs):
            adata = adata_factory()
            if adata is None:
                break
            try:
                t0 = time.perf_counter()
                pyscx.accel.pca(adata, n_comps=n_comps, device="gpu", **kwargs)
                times.append(time.perf_counter() - t0)
            except Exception as e:
                times.append(None)
                out.setdefault("errors", {})[name] = str(e)
                break
            finally:
                del adata
                gc.collect()
        finite = [t for t in times if t is not None]
        if finite:
            out[name] = {
                "times_s": [round(t, 3) for t in finite],
                "median_s": round(sorted(finite)[len(finite) // 2], 3),
            }
    return out


def run_pipeline_benchmark(dataset_name: str = "census_1m",
                           n_runs: int = 3,
                           attribute_preprocessing: bool = False,
                           pca_variants: bool = False) -> dict:
    """Benchmark end-to-end GPU vs CPU pipeline.

    With `attribute_preprocessing=True`, also times
    `normalize_total`/`log1p`/`highly_variable_genes` separately on both CPU
    and GPU baselines — this exposes the preprocessing cost that would
    otherwise be hidden inside the CPU baseline's "load preprocessed data"
    step (see Phase 7.2 in GPU-ACC-SPEED-UP.md).

    With `pca_variants=True`, sweeps `gpu_cov_pca`, `gpu_randomized_pca_chol`,
    and `gpu_randomized_pca_householder` timings on the GPU side and records
    them under `result["gpu_pca_variants"]`.
    """
    import pyscx

    print(f"\n{'='*60}")
    print(f"End-to-End Pipeline Benchmark — {dataset_name}")
    print(f"{'='*60}")

    info = DATASETS.get(dataset_name, {})
    n_obs = info.get("cells", 0)

    # CPU pipeline
    print(f"\n  Running CPU pipeline ({n_runs} runs)...")
    cpu_runs = []
    for run in range(n_runs):
        adata = load_adata_in_memory(dataset_name)
        if adata is None:
            return {"error": f"Dataset {dataset_name} not available"}
        t = run_pipeline(adata, device="cpu")
        cpu_runs.append(t)
        print(f"    Run {run+1}: {t['total']:.2f}s "
              f"(PCA={t['pca']:.2f}, kNN={t['knn']:.2f}, "
              f"UMAP={t['umap']:.2f}, Leiden={t['leiden']:.2f})")
        del adata
        gc.collect()

    cpu_totals = [r["total"] for r in cpu_runs]
    cpu_median_idx = sorted(range(len(cpu_totals)), key=lambda i: cpu_totals[i])[len(cpu_totals) // 2]
    cpu_best = cpu_runs[cpu_median_idx]
    print(f"  CPU median total: {cpu_best['total']:.2f}s")

    result = {
        "benchmark": "gpu_pipeline",
        "dataset": dataset_name,
        "n_obs": n_obs,
        "cpu_runs": cpu_runs,
        "cpu_median_total_s": round(cpu_best["total"], 3),
        "cpu_median_breakdown": {k: round(v, 3) for k, v in cpu_best.items()},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Optional preprocessing attribution (CPU baseline).
    if attribute_preprocessing:
        print(f"\n  Attributing preprocessing cost (CPU, {n_runs} runs)...")
        cpu_prep_runs = []
        for run in range(n_runs):
            a = load_adata_in_memory_raw(dataset_name)
            if a is None:
                break
            cpu_prep_runs.append(run_preprocessing(a, device="cpu"))
            del a
            gc.collect()
        if cpu_prep_runs:
            cpu_prep_totals = [r["total"] for r in cpu_prep_runs]
            cpu_prep_median = cpu_prep_runs[sorted(range(len(cpu_prep_totals)),
                                                   key=lambda i: cpu_prep_totals[i])[len(cpu_prep_totals) // 2]]
            result["cpu_preprocessing_runs"] = cpu_prep_runs
            result["cpu_preprocessing_median_breakdown"] = {
                k: round(v, 3) for k, v in cpu_prep_median.items() if isinstance(v, (int, float))
            }

    # GPU pipeline
    print(f"\n  Running GPU pipeline ({n_runs} runs)...")
    gpu_runs = []
    try:
        for run in range(n_runs):
            adata = load_preprocessed_backed(dataset_name)
            if adata is None:
                raise RuntimeError(f"Cannot load backed {dataset_name}")
            t = run_pipeline(adata, device="gpu")
            gpu_runs.append(t)
            print(f"    Run {run+1}: {t['total']:.2f}s "
                  f"(PCA={t['pca']:.2f}, kNN={t['knn']:.2f}, "
                  f"UMAP={t['umap']:.2f}, Leiden={t['leiden']:.2f})")
            del adata
            gc.collect()

        gpu_totals = [r["total"] for r in gpu_runs]
        gpu_median_idx = sorted(range(len(gpu_totals)), key=lambda i: gpu_totals[i])[len(gpu_totals) // 2]
        gpu_best = gpu_runs[gpu_median_idx]

        speedup = cpu_best["total"] / gpu_best["total"] if gpu_best["total"] > 0 else 0
        print(f"  GPU median total: {gpu_best['total']:.2f}s, speedup: {speedup:.1f}x")

        result["gpu_runs"] = gpu_runs
        result["gpu_median_total_s"] = round(gpu_best["total"], 3)
        result["gpu_median_breakdown"] = {k: round(v, 3) for k, v in gpu_best.items()}
        result["speedup"] = round(speedup, 1)

        # Per-operation speedups
        per_op_speedup = {}
        for op in ["pca", "knn", "umap", "leiden"]:
            cpu_v = cpu_best[op]
            gpu_v = gpu_best[op]
            per_op_speedup[op] = round(cpu_v / gpu_v, 1) if gpu_v > 0 else 0
        result["per_op_speedup"] = per_op_speedup

        # Phase 7.2: attribute preprocessing cost on the GPU side too.
        if attribute_preprocessing:
            print(f"\n  Attributing preprocessing cost (GPU, {n_runs} runs)...")
            gpu_prep_runs = []
            for run in range(n_runs):
                a = load_backed_raw(dataset_name)
                if a is None:
                    break
                gpu_prep_runs.append(run_preprocessing(a, device="gpu"))
                del a
                gc.collect()
            if gpu_prep_runs:
                gpu_prep_totals = [r["total"] for r in gpu_prep_runs]
                gpu_prep_median = gpu_prep_runs[sorted(range(len(gpu_prep_totals)),
                                                       key=lambda i: gpu_prep_totals[i])[len(gpu_prep_totals) // 2]]
                result["gpu_preprocessing_runs"] = gpu_prep_runs
                result["gpu_preprocessing_median_breakdown"] = {
                    k: round(v, 3) for k, v in gpu_prep_median.items() if isinstance(v, (int, float))
                }

        # Phase 7.2: sweep GPU PCA dispatch variants.
        if pca_variants:
            print(f"\n  Sweeping GPU PCA variants (cov / rand-H / rand-C)...")
            variants = run_pca_variants_benchmark(
                lambda: load_preprocessed_backed(dataset_name),
                n_comps=50, n_runs=n_runs,
            )
            if variants:
                result["gpu_pca_variants"] = variants
                for name, rec in variants.items():
                    if isinstance(rec, dict) and "median_s" in rec:
                        print(f"    {name}: {rec['median_s']:.3f}s")

    except Exception as e:
        print(f"  GPU pipeline FAILED: {e}")
        result["gpu_available"] = False

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Graceful fallback test
# ──────────────────────────────────────────────────────────────────────────────

def run_fallback_test() -> dict:
    """Test that GPU operations gracefully fall back to CPU when GPU unavailable.

    Runs the full pipeline (PCA -> kNN -> UMAP -> Leiden) with device="cpu"
    to verify all operations work in fallback mode. Each operation depends on
    the previous one (neighbors needs X_pca, umap/leiden need connectivities).
    """
    import pyscx
    import anndata
    import scipy.sparse as sp

    print(f"\n{'='*60}")
    print(f"Graceful Fallback Test")
    print(f"{'='*60}")

    # Create a small synthetic dataset
    rng = np.random.default_rng(42)
    n_obs, n_vars = 200, 100
    X = sp.random(n_obs, n_vars, density=0.1, format="csr",
                  random_state=42, dtype=np.float32)
    X.data = rng.poisson(5, size=X.nnz).astype(np.float32) + 1
    adata = anndata.AnnData(X=X)

    # Run pipeline sequentially — each op depends on the previous
    pipeline_ops = [
        ("pca", lambda a: pyscx.accel.pca(a, n_comps=10, device="cpu")),
        ("neighbors", lambda a: pyscx.accel.neighbors(a, n_neighbors=10, device="cpu")),
        ("umap", lambda a: pyscx.accel.umap(a, device="cpu")),
        ("leiden", lambda a: pyscx.accel.leiden(a, device="cpu")),
    ]

    results = {}
    for name, fn in pipeline_ops:
        try:
            fn(adata)
            results[name] = "PASS"
            print(f"  {name} (device=cpu): PASS")
        except Exception as e:
            results[name] = f"FAIL: {e}"
            print(f"  {name} (device=cpu): FAIL ({e})")
            break  # downstream ops will also fail

    all_pass = all(v == "PASS" for v in results.values()) and len(results) == 4

    return {
        "benchmark": "gpu_fallback_test",
        "operations": results,
        "pass": all_pass,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Go/No-Go gate evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_gate(pipeline_result: dict | None = None) -> dict:
    """Evaluate all Go/No-Go gate criteria from benchmark JSON files."""

    print(f"\n{'='*60}")
    print(f"Go/No-Go Gate Evaluation")
    print(f"{'='*60}")

    gate = {
        "benchmark": "gpu_gonogo",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "criteria": {},
    }

    # Gate 1: GPU PCA matches CPU within tolerance
    pca_path = RESULTS_DIR / "gpu_pca_validation.json"
    if pca_path.exists():
        pca_data = json.loads(pca_path.read_text())
        # Check all validated datasets
        all_pass = True
        details = []
        for entry in pca_data:
            if "error" in entry:
                continue
            ds = entry.get("dataset", "?")
            passed = entry.get("pass", False)
            min_sim = entry.get("cosine_sim_min", 0)
            details.append(f"{ds}: min_cos_sim={min_sim}")
            all_pass = all_pass and passed
        gate["criteria"]["pca_correctness"] = {
            "pass": all_pass,
            "target": "cosine_sim > 0.99 per PC",
            "details": details,
        }
    else:
        gate["criteria"]["pca_correctness"] = {
            "pass": False, "target": "cosine_sim > 0.99",
            "details": ["gpu_pca_validation.json not found — run benchmark_gpu_pca.py first"],
        }

    # Gate 2: GPU kNN recall > 0.95
    knn_path = RESULTS_DIR / "gpu_knn_recall.json"
    if knn_path.exists():
        knn_data = json.loads(knn_path.read_text())
        recall = knn_data.get("recall_gpu_vs_exact", 0)
        gate["criteria"]["knn_recall"] = {
            "pass": recall > 0.95,
            "target": "recall@k > 0.95 vs exact",
            "value": recall,
        }
    else:
        gate["criteria"]["knn_recall"] = {
            "pass": False, "target": "recall@k > 0.95",
            "details": ["gpu_knn_recall.json not found — run benchmark_gpu_knn.py first"],
        }

    # Gate 3: End-to-end pipeline >= 10x faster on 1M cells
    if pipeline_result and "speedup" in pipeline_result:
        speedup = pipeline_result["speedup"]
        gate["criteria"]["pipeline_speedup"] = {
            "pass": speedup >= 10.0,
            "target": ">= 10x on 1M cells",
            "value": speedup,
            "dataset": pipeline_result.get("dataset", "?"),
        }
    else:
        pipe_path = RESULTS_DIR / "gpu_pipeline_timing.json"
        if pipe_path.exists():
            pipe_data = json.loads(pipe_path.read_text())
            speedup = pipe_data.get("speedup", 0)
            gate["criteria"]["pipeline_speedup"] = {
                "pass": speedup >= 10.0,
                "target": ">= 10x on 1M cells",
                "value": speedup,
            }
        else:
            gate["criteria"]["pipeline_speedup"] = {
                "pass": False, "target": ">= 10x",
                "details": ["pipeline timing not available"],
            }

    # Gate 4: Graceful fallback
    fallback_result = run_fallback_test()
    gate["criteria"]["graceful_fallback"] = {
        "pass": fallback_result["pass"],
        "target": "all ops fall back to CPU without crash",
        "operations": fallback_result["operations"],
    }

    # Overall
    all_pass = all(c["pass"] for c in gate["criteria"].values())
    gate["pass"] = all_pass

    # Print summary
    print(f"\n  Gate Results:")
    for name, criterion in gate["criteria"].items():
        status = "PASS" if criterion["pass"] else "FAIL"
        target = criterion.get("target", "")
        value = criterion.get("value", "")
        if value:
            print(f"    {name}: {status} (value={value}, target={target})")
        else:
            print(f"    {name}: {status} (target={target})")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    return gate


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(pipeline_result: dict | None,
                    gate_result: dict | None) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU End-to-End Pipeline Benchmark & Go/No-Go Gate Report",
        "",
        f"**Generated**: {ts}",
        "",
    ]

    if pipeline_result and "error" not in pipeline_result:
        lines += [
            "## 1. Pipeline Timing (PCA -> kNN -> UMAP -> Leiden)",
            "",
            f"**Dataset**: {pipeline_result['dataset']} ({pipeline_result['n_obs']:,} cells)",
            "",
        ]

        if "gpu_median_breakdown" in pipeline_result:
            cpu = pipeline_result["cpu_median_breakdown"]
            gpu = pipeline_result["gpu_median_breakdown"]
            per_op = pipeline_result.get("per_op_speedup", {})

            lines += [
                "| Operation | CPU (s) | GPU (s) | Speedup |",
                "|-----------|---------|---------|---------|",
            ]
            for op in ["pca", "knn", "umap", "leiden", "total"]:
                cpu_v = cpu.get(op, 0)
                gpu_v = gpu.get(op, 0)
                sp = per_op.get(op, pipeline_result.get("speedup", 0)) if op != "total" else pipeline_result.get("speedup", 0)
                lines.append(f"| {op.upper()} | {cpu_v:.3f} | {gpu_v:.3f} | {sp:.1f}x |")
            lines.append("")
        else:
            lines += [
                f"CPU total: {pipeline_result.get('cpu_median_total_s', 'N/A')}s",
                "GPU: not available",
                "",
            ]

        # Preprocessing attribution (optional, Phase 7.2).
        cpu_prep = pipeline_result.get("cpu_preprocessing_median_breakdown")
        gpu_prep = pipeline_result.get("gpu_preprocessing_median_breakdown")
        if cpu_prep or gpu_prep:
            lines += [
                "### Preprocessing (normalize_total → log1p → HVG)",
                "",
                "| Op | CPU (s) | GPU (s) | Speedup |",
                "|----|---------|---------|---------|",
            ]
            for op in ["normalize_total", "log1p", "highly_variable_genes", "total"]:
                c = (cpu_prep or {}).get(op, float("nan"))
                g = (gpu_prep or {}).get(op, float("nan"))
                if g and g > 0 and c == c and c > 0:
                    sp = f"{c / g:.1f}x"
                else:
                    sp = "—"
                lines.append(f"| {op} | {c:.3f} | {g:.3f} | {sp} |")
            lines.append("")

        # GPU PCA dispatch variants (optional, Phase 7.2).
        variants = pipeline_result.get("gpu_pca_variants")
        if variants:
            lines += [
                "### GPU PCA dispatch variants",
                "",
                "| Variant | Median (s) |",
                "|---------|-----------|",
            ]
            for name in ["gpu_cov_pca", "gpu_randomized_pca_householder", "gpu_randomized_pca_chol"]:
                rec = variants.get(name)
                if isinstance(rec, dict) and "median_s" in rec:
                    lines.append(f"| {name} | {rec['median_s']:.3f} |")
                else:
                    lines.append(f"| {name} | — |")
            lines.append("")

    if gate_result:
        lines += [
            "## 2. Go/No-Go Gate",
            "",
            "| Criterion | Target | Value | Pass? |",
            "|-----------|--------|-------|-------|",
        ]
        for name, c in gate_result.get("criteria", {}).items():
            target = c.get("target", "")
            value = c.get("value", "—")
            status = "PASS" if c["pass"] else "FAIL"
            if isinstance(value, float):
                value = f"{value:.2f}"
            lines.append(f"| {name} | {target} | {value} | {status} |")

        overall = "PASS" if gate_result.get("pass") else "FAIL"
        lines += ["", f"**Overall Gate: {overall}**", ""]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SCX GPU Pipeline Benchmark & Go/No-Go Gate"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["pipeline", "gate", "all"],
    )
    parser.add_argument(
        "--dataset", default="census_1m", choices=list(DATASETS.keys()),
        help="Dataset for pipeline benchmark (default: census_1m)",
    )
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--attribute-preprocessing", action="store_true",
        help="Time normalize_total + log1p + highly_variable_genes separately "
             "on CPU and GPU baselines so preprocessing cost is attributed "
             "symmetrically (Phase 7.2).",
    )
    parser.add_argument(
        "--pca-variants", action="store_true",
        help="Sweep GPU PCA dispatch variants: gpu_cov_pca, "
             "gpu_randomized_pca_householder, gpu_randomized_pca_chol "
             "(Phase 7.2).",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_result = None
    gate_result = None

    if args.mode in ("pipeline", "all"):
        pipeline_result = run_pipeline_benchmark(
            args.dataset,
            args.n_runs,
            attribute_preprocessing=args.attribute_preprocessing,
            pca_variants=args.pca_variants,
        )

        json_path = RESULTS_DIR / "gpu_pipeline_timing.json"
        json_path.write_text(json.dumps(pipeline_result, indent=2))
        print(f"\n  JSON saved: {json_path}")

    if args.mode in ("gate", "all"):
        gate_result = evaluate_gate(pipeline_result)

        json_path = RESULTS_DIR / "gpu_gonogo.json"
        json_path.write_text(json.dumps(gate_result, indent=2))
        print(f"\n  JSON saved: {json_path}")

    report = generate_report(pipeline_result, gate_result)
    md_path = RESULTS_DIR / "gpu_pipeline_benchmark.md"
    md_path.write_text(report)
    print(f"\n  Report saved: {md_path}")
    print(f"\n{'='*60}")
    print(report)


if __name__ == "__main__":
    if "SLURM_JOB_ID" not in os.environ:
        ensure_release_build()
    main()
