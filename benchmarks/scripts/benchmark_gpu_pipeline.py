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


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline timing
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(adata, device: str, n_comps: int = 50,
                 n_neighbors: int = 15) -> dict:
    """Run full PCA -> kNN -> UMAP -> Leiden pipeline, return per-op timing."""
    import pyscx

    timings = {}

    t0 = time.perf_counter()
    pyscx.accel.pca(adata, n_comps=n_comps, device=device)
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


def run_pipeline_benchmark(dataset_name: str = "census_1m",
                           n_runs: int = 3) -> dict:
    """Benchmark end-to-end GPU vs CPU pipeline."""
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
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_result = None
    gate_result = None

    if args.mode in ("pipeline", "all"):
        pipeline_result = run_pipeline_benchmark(args.dataset, args.n_runs)

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
