#!/usr/bin/env python3
"""
SCX Leiden Benchmark — Rust-native vs Python leidenalg

Phase 8b validation and benchmarking:
  1. Timing: Rust Leiden (pyscx.accel.leiden) vs Python (sc.tl.leiden / leidenalg)
  2. Quality: ARI between the two implementations
  3. Modularity check: positive quality
  4. Resolution parameter: higher γ → more communities

Usage:
    python benchmarks/scripts/benchmark_leiden.py --mode all
    python benchmarks/scripts/benchmark_leiden.py --mode all --datasets tabula_sapiens_100k
    python benchmarks/scripts/benchmark_leiden.py --mode report
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
    "tabula_sapiens_100k": {"h5ad": DATA_DIR / "tabula_sapiens_100k.h5ad", "cells": 100_000},
    "census_1m": {"h5ad": DATA_DIR / "census_1m.h5ad", "cells": 1_000_000},
}

N_RUNS = 3
N_COMPS = 50
N_NEIGHBORS = 15
N_HVGS = 2000


def get_rss_mb():
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def save_json(data, filename):
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  JSON saved: {path}")


def preprocess_for_leiden(dataset_name):
    """Load dataset and run preprocessing up to neighbors (shared for both backends)."""
    import anndata
    import scanpy as sc
    import pyscx

    info = DATASETS[dataset_name]
    h5ad_path = info["h5ad"]
    if not h5ad_path.exists():
        print(f"    SKIP: {h5ad_path} not found")
        return None

    print(f"  Loading {dataset_name}... RSS={get_rss_mb():.0f}MB")
    adata = anndata.read_h5ad(str(h5ad_path))
    print(f"  Loaded. RSS={get_rss_mb():.0f}MB")

    # Drop .raw to save memory (can be huge for large datasets)
    if adata.raw is not None:
        adata.raw = None
        gc.collect()
        print(f"  Dropped .raw. RSS={get_rss_mb():.0f}MB")

    print(f"  Preprocessing: normalize -> log1p -> HVG -> PCA -> neighbors...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_top = min(N_HVGS, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)

    gc.collect()
    print(f"  After HVG subset. RSS={get_rss_mb():.0f}MB")

    # Use Rust accelerators for PCA + neighbors (fast)
    pyscx.accel.pca(adata, n_comps=N_COMPS, device="cpu")
    pyscx.accel.neighbors(adata, n_neighbors=N_NEIGHBORS, device="cpu")
    gc.collect()

    print(f"  Preprocessed: {adata.n_obs:,} cells, {adata.n_vars:,} genes, "
          f"connectivities shape {adata.obsp['connectivities'].shape}, "
          f"RSS={get_rss_mb():.0f}MB")
    return adata


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark: Rust Leiden vs Python leidenalg
# ──────────────────────────────────────────────────────────────────────────────

def bench_leiden_rust(adata, resolution=1.0, seed=42, n_runs=N_RUNS):
    """Benchmark Rust-native Leiden (pyscx.accel.leiden)."""
    import pyscx

    times = []
    for run in range(-1, n_runs):
        t0 = time.perf_counter()
        pyscx.accel.leiden(
            adata, resolution=resolution, key_added="leiden_rust",
            random_state=seed, device="cpu"
        )
        elapsed = time.perf_counter() - t0
        gc.collect()

        if run < 0:
            print(f"    Rust warmup: {elapsed:.2f}s")
        else:
            times.append(elapsed)
            print(f"    Rust run {run+1}/{n_runs}: {elapsed:.2f}s")

    membership = adata.obs["leiden_rust"].values.copy()
    modularity = adata.uns.get("leiden_rust", {}).get("modularity", None)
    n_comms = adata.uns.get("leiden_rust", {}).get("n_communities", None)
    backend = adata.uns.get("leiden_rust", {}).get("backend", "unknown")

    return {
        "times": times,
        "median_s": float(np.median(times)),
        "membership": membership,
        "modularity": modularity,
        "n_communities": n_comms,
        "backend": backend,
    }


def bench_leiden_python(adata, resolution=1.0, seed=42, n_runs=N_RUNS):
    """Benchmark Python leidenalg (sc.tl.leiden).

    Uses a single run (no warmup) when n_obs > 50k to conserve memory,
    since igraph graph construction is extremely memory-hungry.
    """
    import scanpy as sc

    effective_runs = 1 if adata.n_obs > 50_000 else n_runs

    times = []
    for run in range(-1, effective_runs):
        t0 = time.perf_counter()
        sc.tl.leiden(adata, resolution=resolution, random_state=seed,
                     key_added="leiden_python")
        elapsed = time.perf_counter() - t0
        gc.collect()  # Free igraph C objects between runs

        if run < 0:
            print(f"    Python warmup: {elapsed:.2f}s")
        else:
            times.append(elapsed)
            print(f"    Python run {run+1}/{effective_runs}: {elapsed:.2f}s")

    membership = adata.obs["leiden_python"].values.copy()

    modularity = None
    n_comms = len(set(membership))

    return {
        "times": times,
        "median_s": float(np.median(times)),
        "membership": membership,
        "modularity": modularity,
        "n_communities": n_comms,
        "backend": "leidenalg",
    }


def bench_resolution_effect(adata, seed=42):
    """Verify that higher resolution yields more communities."""
    import pyscx

    resolutions = [0.5, 1.0, 2.0]
    results = []
    for res in resolutions:
        key = f"leiden_res_{res}"
        print(f"    γ={res:.1f}: starting... RSS={get_rss_mb():.0f}MB", flush=True)
        pyscx.accel.leiden(
            adata, resolution=res, key_added=key,
            random_state=seed, device="cpu"
        )
        n_comms = adata.uns[key]["n_communities"]
        modularity = adata.uns[key]["modularity"]
        results.append({"resolution": res, "n_communities": n_comms, "modularity": modularity})
        print(f"    γ={res:.1f}: {n_comms} communities, Q={modularity:.4f}, RSS={get_rss_mb():.0f}MB")
        # Clean up to save memory
        if key in adata.obs.columns:
            del adata.obs[key]
        if key in adata.uns:
            del adata.uns[key]
        gc.collect()

    # Check: community count should be non-decreasing with resolution
    comms = [r["n_communities"] for r in results]
    monotonic = all(comms[i] <= comms[i+1] for i in range(len(comms)-1))
    return results, monotonic


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def run_leiden_benchmark(datasets=None, n_runs=N_RUNS):
    from sklearn.metrics import adjusted_rand_score

    if datasets is None:
        datasets = list(DATASETS.keys())

    print(f"\n{'='*60}")
    print(f"Leiden Benchmark — Rust-native vs Python leidenalg")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        info = DATASETS.get(ds_name, {})
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({info.get('cells', '?'):,} cells)")
        print(f"{'='*60}")

        adata = preprocess_for_leiden(ds_name)
        if adata is None:
            continue

        result = {
            "benchmark": "leiden",
            "dataset": ds_name,
            "n_obs": adata.n_obs,
            "n_vars": adata.n_vars,
            "n_runs": n_runs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # --- Resolution parameter (run first, before Python to save memory) ---
        print(f"\n  Resolution parameter test:")
        res_results, monotonic = bench_resolution_effect(adata)
        result["resolution_test"] = res_results
        result["resolution_monotonic"] = monotonic
        print(f"  Community count monotonic with resolution: {monotonic}")

        # --- Rust Leiden ---
        print(f"\n  Rust-native Leiden (pyscx.accel.leiden):")
        rust_result = bench_leiden_rust(adata, n_runs=n_runs)
        result["rust_median_s"] = rust_result["median_s"]
        result["rust_times_s"] = rust_result["times"]
        result["rust_modularity"] = rust_result["modularity"]
        result["rust_n_communities"] = rust_result["n_communities"]
        result["rust_backend"] = rust_result["backend"]

        # --- Modularity check ---
        print(f"\n  Modularity (Rust): {rust_result['modularity']}")
        result["modularity_positive"] = (
            rust_result["modularity"] is not None and rust_result["modularity"] > 0
        )

        # Clean up Rust results from adata before Python run
        if "leiden_rust" in adata.obs.columns:
            del adata.obs["leiden_rust"]
        if "leiden_rust" in adata.uns:
            del adata.uns["leiden_rust"]
        gc.collect()

        # --- Python Leiden ---
        print(f"\n  Python Leiden (sc.tl.leiden / leidenalg):")
        python_result = bench_leiden_python(adata, n_runs=n_runs)
        result["python_median_s"] = python_result["median_s"]
        result["python_times_s"] = python_result["times"]
        result["python_n_communities"] = python_result["n_communities"]

        # --- ARI comparison ---
        ari = adjusted_rand_score(rust_result["membership"], python_result["membership"])
        result["ari_rust_vs_python"] = round(ari, 4)
        print(f"\n  ARI (Rust vs Python): {ari:.4f}")

        # --- Speedup ---
        if python_result["median_s"] > 0:
            speedup = python_result["median_s"] / rust_result["median_s"]
            result["speedup_vs_python"] = round(speedup, 2)
            print(f"  Speedup: {speedup:.2f}x")

        # --- Summary ---
        print(f"\n  ── Summary for {ds_name} ──")
        print(f"  Rust Leiden:   {rust_result['median_s']:.2f}s "
              f"({rust_result['n_communities']} communities)")
        print(f"  Python Leiden: {python_result['median_s']:.2f}s "
              f"({python_result['n_communities']} communities)")
        print(f"  ARI: {ari:.4f}")
        print(f"  Modularity > 0: {result['modularity_positive']}")
        print(f"  Resolution monotonic: {monotonic}")

        results.append(result)
        del adata
        gc.collect()

    save_json(results, "leiden_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Leiden Benchmark Report — Rust-native vs Python leidenalg",
        "",
        f"**Generated**: {ts}",
        f"**Phase**: 8b — Parallel Rust Leiden",
        "",
    ]

    path = RESULTS_DIR / "leiden_benchmark.json"
    if not path.exists():
        lines.append("No results found.")
        report = "\n".join(lines)
        (RESULTS_DIR / "leiden_benchmark.md").write_text(report)
        print(report)
        return report

    data = json.loads(path.read_text())

    # Overview table
    lines += [
        "## Performance",
        "",
        "| Dataset | Cells | Rust (s) | Python (s) | Speedup | ARI | Rust Communities | Python Communities |",
        "|---------|------:|--------:|-----------:|--------:|----:|-----------------:|-------------------:|",
    ]
    for r in data:
        lines.append(
            f"| {r['dataset']} | {r['n_obs']:,} | "
            f"{r['rust_median_s']:.2f} | {r['python_median_s']:.2f} | "
            f"{r.get('speedup_vs_python', 'N/A')}x | "
            f"{r['ari_rust_vs_python']:.4f} | "
            f"{r['rust_n_communities']} | {r['python_n_communities']} |"
        )
    lines.append("")

    # Validation checks
    lines += ["## Validation", ""]
    for r in data:
        lines.append(f"### {r['dataset']}")
        lines.append(f"- **ARI ≥ 0.80**: {'PASS' if r['ari_rust_vs_python'] >= 0.80 else 'FAIL'} "
                      f"(ARI = {r['ari_rust_vs_python']:.4f})")
        lines.append(f"- **Modularity > 0**: {'PASS' if r.get('modularity_positive') else 'FAIL'} "
                      f"(Q = {r.get('rust_modularity', 'N/A')})")
        lines.append(f"- **Resolution monotonic**: "
                      f"{'PASS' if r.get('resolution_monotonic') else 'FAIL'}")
        lines.append(f"- **Backend**: {r.get('rust_backend', 'unknown')}")
        lines.append("")

        # Resolution details
        if "resolution_test" in r:
            lines.append("**Resolution parameter test:**")
            lines.append("")
            lines.append("| γ | Communities | Modularity |")
            lines.append("|--:|----------:|-----------:|")
            for rt in r["resolution_test"]:
                lines.append(
                    f"| {rt['resolution']} | {rt['n_communities']} | "
                    f"{rt['modularity']:.4f} |"
                )
            lines.append("")

    # Target check
    lines += ["## Target: Leiden ≤ 1,000s on census_1m", ""]
    census = [r for r in data if r["dataset"] == "census_1m"]
    if census:
        r = census[0]
        rust_s = r["rust_median_s"]
        ok = rust_s <= 1000
        lines.append(f"- Rust Leiden on census_1m: **{rust_s:.1f}s** "
                      f"({'PASS' if ok else 'FAIL'}, target ≤ 1,000s)")
        python_s = r["python_median_s"]
        lines.append(f"- Python leidenalg on census_1m: {python_s:.1f}s")
    else:
        lines.append("- census_1m not benchmarked yet")
    lines.append("")

    report = "\n".join(lines)
    md_path = RESULTS_DIR / "leiden_benchmark.md"
    md_path.write_text(report)
    print(f"\nReport saved: {md_path}")
    print(report)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="SCX Leiden Benchmark (Phase 8b)")
    parser.add_argument("--mode", default="all", choices=["all", "report"],
                        help="Run benchmarks or just generate report")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--n-runs", type=int, default=N_RUNS)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        ensure_release_build()

    if args.mode == "report":
        generate_report()
        return

    run_leiden_benchmark(datasets=args.datasets, n_runs=args.n_runs)
    generate_report()


if __name__ == "__main__":
    main()
