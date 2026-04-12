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
    """Load dataset and run preprocessing up to neighbors (shared for both backends).

    Uses scanpy (not pyscx) for PCA and neighbors to avoid initializing rayon's
    global thread pool with all available CPUs, which causes OOM on SLURM nodes
    with many CPUs (e.g. 192) due to GLIBC per-thread memory arenas.
    """
    import anndata
    import scanpy as sc

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

    # Use scanpy for PCA + neighbors (avoids rayon global pool init on SLURM)
    sc.tl.pca(adata, n_comps=N_COMPS)
    sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS)
    gc.collect()

    print(f"  Preprocessed: {adata.n_obs:,} cells, {adata.n_vars:,} genes, "
          f"connectivities shape {adata.obsp['connectivities'].shape}, "
          f"RSS={get_rss_mb():.0f}MB")
    return adata


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark: Rust Leiden vs Python leidenalg
# ──────────────────────────────────────────────────────────────────────────────

def bench_leiden_rust(adata, resolution=1.0, seed=42, n_runs=N_RUNS, parallel=False):
    """Benchmark Rust-native Leiden (pyscx.accel.leiden)."""
    import pyscx
    import ctypes

    mode_label = "parallel" if parallel else "sequential"
    key = "leiden_rust"

    # Force glibc to return freed pages between runs to prevent OOM
    # from heap fragmentation on SLURM nodes with many CPUs.
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _malloc_trim = _libc.malloc_trim
    except (OSError, AttributeError):
        _malloc_trim = None

    times = []
    for run in range(-1, n_runs):
        t0 = time.perf_counter()
        pyscx.accel.leiden(
            adata, resolution=resolution, key_added=key,
            random_state=seed, device="cpu", parallel=parallel,
        )
        elapsed = time.perf_counter() - t0
        gc.collect()
        if _malloc_trim:
            _malloc_trim(0)  # Return freed pages to OS

        if run < 0:
            print(f"    Rust ({mode_label}) warmup: {elapsed:.2f}s")
        else:
            times.append(elapsed)
            print(f"    Rust ({mode_label}) run {run+1}/{n_runs}: {elapsed:.2f}s")

    membership = adata.obs[key].values.copy()
    modularity = adata.uns.get(key, {}).get("modularity", None)
    n_comms = adata.uns.get(key, {}).get("n_communities", None)
    backend = adata.uns.get(key, {}).get("backend", "unknown")

    return {
        "times": times,
        "median_s": float(np.median(times)),
        "membership": membership,
        "modularity": modularity,
        "n_communities": n_comms,
        "backend": backend,
        "mode": mode_label,
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
    import ctypes
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _malloc_trim = _libc.malloc_trim
    except (OSError, AttributeError):
        _malloc_trim = None

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
        if _malloc_trim:
            _malloc_trim(0)  # Return freed pages to OS
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
    print(f"Leiden Benchmark — Rust sequential/parallel vs Python leidenalg")
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

        # --- Rust Leiden (sequential — default, matches C++ leidenalg) ---
        print(f"\n  Rust Leiden (sequential):")
        rust_seq = bench_leiden_rust(adata, n_runs=n_runs, parallel=False)
        result["rust_seq_median_s"] = rust_seq["median_s"]
        result["rust_seq_times_s"] = rust_seq["times"]
        result["rust_seq_modularity"] = rust_seq["modularity"]
        result["rust_seq_n_communities"] = rust_seq["n_communities"]
        result["rust_seq_backend"] = rust_seq["backend"]
        # Keep for backwards compat
        result["rust_median_s"] = rust_seq["median_s"]
        result["rust_times_s"] = rust_seq["times"]
        result["rust_modularity"] = rust_seq["modularity"]
        result["rust_n_communities"] = rust_seq["n_communities"]
        result["rust_backend"] = rust_seq["backend"]

        # --- Modularity check ---
        print(f"\n  Modularity (Rust seq): {rust_seq['modularity']}")
        result["modularity_positive"] = (
            rust_seq["modularity"] is not None and rust_seq["modularity"] > 0
        )

        # Clean up before parallel run
        if "leiden_rust" in adata.obs.columns:
            del adata.obs["leiden_rust"]
        if "leiden_rust" in adata.uns:
            del adata.uns["leiden_rust"]
        gc.collect()

        # --- Rust Leiden (parallel — conflict-free batched) ---
        print(f"\n  Rust Leiden (parallel):")
        rust_par = bench_leiden_rust(adata, n_runs=n_runs, parallel=True)
        result["rust_par_median_s"] = rust_par["median_s"]
        result["rust_par_times_s"] = rust_par["times"]
        result["rust_par_modularity"] = rust_par["modularity"]
        result["rust_par_n_communities"] = rust_par["n_communities"]

        # Clean up before Python run
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

        # --- ARI comparisons ---
        ari_seq = adjusted_rand_score(rust_seq["membership"], python_result["membership"])
        ari_par = adjusted_rand_score(rust_par["membership"], python_result["membership"])
        result["ari_rust_seq_vs_python"] = round(ari_seq, 4)
        result["ari_rust_par_vs_python"] = round(ari_par, 4)
        # Keep for backwards compat
        result["ari_rust_vs_python"] = round(ari_seq, 4)
        print(f"\n  ARI (Rust seq vs Python): {ari_seq:.4f}")
        print(f"  ARI (Rust par vs Python): {ari_par:.4f}")

        # --- Speedup ---
        if python_result["median_s"] > 0:
            speedup_seq = python_result["median_s"] / rust_seq["median_s"]
            speedup_par = python_result["median_s"] / rust_par["median_s"]
            result["speedup_seq_vs_python"] = round(speedup_seq, 2)
            result["speedup_par_vs_python"] = round(speedup_par, 2)
            result["speedup_vs_python"] = round(speedup_seq, 2)
            print(f"  Speedup (seq): {speedup_seq:.2f}x")
            print(f"  Speedup (par): {speedup_par:.2f}x")

        # --- Summary ---
        print(f"\n  ── Summary for {ds_name} ──")
        print(f"  Rust seq:      {rust_seq['median_s']:.2f}s "
              f"({rust_seq['n_communities']} comms, ARI={ari_seq:.4f})")
        print(f"  Rust par:      {rust_par['median_s']:.2f}s "
              f"({rust_par['n_communities']} comms, ARI={ari_par:.4f})")
        print(f"  Python:        {python_result['median_s']:.2f}s "
              f"({python_result['n_communities']} comms)")
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
        "# Leiden Benchmark Report — Rust seq/par vs Python leidenalg",
        "",
        f"**Generated**: {ts}",
        f"**Phase**: 8c — Sequential + Parallel Rust Leiden",
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
        "| Dataset | Cells | Rust seq (s) | Rust par (s) | Python (s) | Seq speedup | ARI seq | ARI par | Seq comms | Par comms | Py comms |",
        "|---------|------:|-----------:|-----------:|-----------:|------------:|--------:|--------:|----------:|----------:|---------:|",
    ]
    for r in data:
        seq_s = r.get('rust_seq_median_s', r.get('rust_median_s', 0))
        par_s = r.get('rust_par_median_s', 'N/A')
        py_s = r['python_median_s']
        ari_seq = r.get('ari_rust_seq_vs_python', r.get('ari_rust_vs_python', 0))
        ari_par = r.get('ari_rust_par_vs_python', 'N/A')
        seq_comms = r.get('rust_seq_n_communities', r.get('rust_n_communities', '?'))
        par_comms = r.get('rust_par_n_communities', 'N/A')
        py_comms = r['python_n_communities']
        speedup_seq = r.get('speedup_seq_vs_python', r.get('speedup_vs_python', 'N/A'))
        lines.append(
            f"| {r['dataset']} | {r['n_obs']:,} | "
            f"{seq_s:.2f} | {par_s if isinstance(par_s, str) else f'{par_s:.2f}'} | {py_s:.2f} | "
            f"{speedup_seq}x | "
            f"{ari_seq:.4f} | {ari_par if isinstance(ari_par, str) else f'{ari_par:.4f}'} | "
            f"{seq_comms} | {par_comms} | {py_comms} |"
        )
    lines.append("")

    # Validation checks
    lines += ["## Validation", ""]
    for r in data:
        ari_seq = r.get('ari_rust_seq_vs_python', r.get('ari_rust_vs_python', 0))
        ari_par = r.get('ari_rust_par_vs_python', None)
        lines.append(f"### {r['dataset']}")
        lines.append(f"- **ARI seq ≥ 0.80**: {'PASS' if ari_seq >= 0.80 else 'FAIL'} "
                      f"(ARI = {ari_seq:.4f})")
        if ari_par is not None:
            lines.append(f"- **ARI par**: {ari_par:.4f}")
        lines.append(f"- **Modularity > 0**: {'PASS' if r.get('modularity_positive') else 'FAIL'} "
                      f"(Q = {r.get('rust_seq_modularity', r.get('rust_modularity', 'N/A'))})")
        lines.append(f"- **Resolution monotonic**: "
                      f"{'PASS' if r.get('resolution_monotonic') else 'FAIL'}")
        lines.append(f"- **Backend**: {r.get('rust_seq_backend', r.get('rust_backend', 'unknown'))}")
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
