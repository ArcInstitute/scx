#!/usr/bin/env python3
"""
SCX Phase 4a Preprocessing Benchmark

Compares pyscx.preprocess() (streaming Rust-native) against scanpy
normalize_total + log1p at 100K, 500K, 1M cells.

Metrics: wall-clock time, peak RSS, output file size, numerical correctness.

Usage:
    python benchmarks/scripts/benchmark_accel_preprocessing.py --mode validate
    python benchmarks/scripts/benchmark_accel_preprocessing.py --mode bench
    python benchmarks/scripts/benchmark_accel_preprocessing.py --mode all
    python benchmarks/scripts/benchmark_accel_preprocessing.py --mode report
"""

import argparse
import gc
import json
import os
import sys
import tempfile
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
N_RUNS = 3


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def get_rss_mb():
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def ensure_scx_file(dataset_name):
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
    return scx_path


def save_json(data, filename):
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"  JSON saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Correctness Validation
# ──────────────────────────────────────────────────────────────────────────────

def run_validation(datasets=None):
    """Verify pyscx.preprocess() output matches scanpy normalize+log1p."""
    import anndata
    import pyscx
    import scanpy as sc
    import scipy.sparse as sp

    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k"]

    print(f"\n{'='*60}")
    print(f"Preprocessing Validation — pyscx.preprocess() vs scanpy")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        info = DATASETS[ds_name]
        h5ad_path = info["h5ad"]
        if not h5ad_path.exists():
            print(f"  SKIP: {ds_name} not available")
            continue

        print(f"\n  Validating {ds_name}...")

        # Scanpy reference
        adata_ref = anndata.read_h5ad(str(h5ad_path))
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        if sp.issparse(adata_ref.X):
            X_ref = adata_ref.X.toarray()
        else:
            X_ref = np.asarray(adata_ref.X)

        # SCX preprocess
        scx_path = ensure_scx_file(ds_name)
        if scx_path is None:
            continue

        with tempfile.NamedTemporaryFile(suffix=".scx", delete=False,
                                          dir=str(DATA_DIR)) as tmp:
            prep_path = tmp.name

        try:
            pyscx.preprocess(str(scx_path), prep_path,
                              operations=["normalize_total", "log1p"],
                              target_sum=1e4)

            adata_scx = pyscx.open(prep_path).to_anndata()
            if sp.issparse(adata_scx.X):
                X_scx = adata_scx.X.toarray()
            else:
                X_scx = np.asarray(adata_scx.X)

            # Compare
            max_abs_diff = float(np.max(np.abs(X_ref - X_scx)))
            mean_abs_diff = float(np.mean(np.abs(X_ref - X_scx)))

            # Relative error (where reference is non-zero)
            nonzero_mask = np.abs(X_ref) > 1e-10
            if nonzero_mask.any():
                rel_err = np.abs(X_ref[nonzero_mask] - X_scx[nonzero_mask]) / np.abs(X_ref[nonzero_mask])
                max_rel_err = float(np.max(rel_err))
            else:
                max_rel_err = 0.0

            pass_correctness = max_abs_diff < 1e-5
            print(f"    Max abs diff: {max_abs_diff:.2e} "
                  f"({'PASS' if pass_correctness else 'FAIL'})")
            print(f"    Mean abs diff: {mean_abs_diff:.2e}")
            print(f"    Max rel err: {max_rel_err:.2e}")

            results.append({
                "test": "preprocess_correctness",
                "dataset": ds_name,
                "n_obs": adata_ref.n_obs,
                "n_vars": adata_ref.n_vars,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "max_rel_err": max_rel_err,
                "pass": pass_correctness,
            })

            del adata_scx, X_scx
        finally:
            try:
                os.unlink(prep_path)
            except OSError:
                pass

        del adata_ref, X_ref
        gc.collect()

    save_json(results, "accel_preprocessing_validation.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Performance Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(datasets=None, n_runs=N_RUNS):
    """Benchmark pyscx.preprocess() vs scanpy at 100K, 500K, 1M."""
    import anndata
    import pyscx
    import scanpy as sc

    if datasets is None:
        datasets = BENCH_DATASETS

    print(f"\n{'='*60}")
    print(f"Preprocessing Benchmark (median of {n_runs} runs)")
    print(f"{'='*60}")

    results = []

    for ds_name in datasets:
        info = DATASETS[ds_name]
        h5ad_path = info["h5ad"]
        if not h5ad_path.exists():
            print(f"  SKIP: {ds_name} not available")
            continue

        scx_path = ensure_scx_file(ds_name)
        if scx_path is None:
            continue

        n_obs = info["cells"]

        # SCX preprocess
        print(f"\n  SCX preprocess on {ds_name} ({n_runs} runs)...")
        scx_times = []
        scx_file_sizes = []
        for run in range(n_runs):
            gc.collect()
            rss_before = get_rss_mb()

            with tempfile.NamedTemporaryFile(suffix=".scx", delete=False,
                                              dir=str(DATA_DIR)) as tmp:
                prep_path = tmp.name

            try:
                t0 = time.perf_counter()
                pyscx.preprocess(str(scx_path), prep_path,
                                  operations=["normalize_total", "log1p"],
                                  target_sum=1e4)
                wall = time.perf_counter() - t0
                rss_after = get_rss_mb()

                scx_times.append(wall)
                file_size_mb = os.path.getsize(prep_path) / (1024 * 1024)
                scx_file_sizes.append(file_size_mb)
                print(f"    Run {run+1}: {wall:.2f}s, file: {file_size_mb:.1f} MB, "
                      f"RSS delta: {rss_after - rss_before:.0f} MB")
            finally:
                try:
                    os.unlink(prep_path)
                except OSError:
                    pass

        # Scanpy preprocess
        print(f"  Scanpy preprocess on {ds_name} ({n_runs} runs)...")
        scanpy_times = []
        for run in range(n_runs):
            gc.collect()
            rss_before = get_rss_mb()

            adata = anndata.read_h5ad(str(h5ad_path))
            t0 = time.perf_counter()
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            wall = time.perf_counter() - t0
            rss_after = get_rss_mb()

            scanpy_times.append(wall)
            print(f"    Run {run+1}: {wall:.2f}s, "
                  f"RSS delta: {rss_after - rss_before:.0f} MB")
            del adata
            gc.collect()

        scx_median = _median(scx_times)
        scanpy_median = _median(scanpy_times)
        speedup = scanpy_median / scx_median if scx_median > 0 else 0

        result = {
            "benchmark": "accel_preprocessing",
            "dataset": ds_name,
            "n_obs": n_obs,
            "scx_times_s": [round(t, 3) for t in scx_times],
            "scanpy_times_s": [round(t, 3) for t in scanpy_times],
            "scx_median_s": round(scx_median, 3),
            "scanpy_median_s": round(scanpy_median, 3),
            "speedup": round(speedup, 2),
            "scx_output_file_size_mb": round(_median(scx_file_sizes), 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(result)

        print(f"  SCX median: {scx_median:.2f}s, Scanpy median: {scanpy_median:.2f}s, "
              f"Speedup: {speedup:.1f}x")
        print(f"  SCX output file size: {_median(scx_file_sizes):.1f} MB")

    save_json(results, "accel_preprocessing_benchmark.json")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Report Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_report():
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Preprocessing Benchmark Report",
        "",
        f"**Generated**: {ts}",
        f"**Benchmark**: pyscx.preprocess() vs scanpy normalize_total + log1p",
        "",
    ]

    # Validation
    val_path = RESULTS_DIR / "accel_preprocessing_validation.json"
    if val_path.exists():
        val_data = json.loads(val_path.read_text())
        lines += [
            "## Correctness Validation",
            "",
            "| Dataset | Cells | Max Abs Diff | Mean Abs Diff | Pass |",
            "|---------|-------|--------------|---------------|------|",
        ]
        for r in val_data:
            p = "Y" if r["pass"] else "N"
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['max_abs_diff']:.2e} | "
                f"{r['mean_abs_diff']:.2e} | {p} |"
            )
        lines.append("")

    # Performance
    bench_path = RESULTS_DIR / "accel_preprocessing_benchmark.json"
    if bench_path.exists():
        bench_data = json.loads(bench_path.read_text())
        lines += [
            "## Performance",
            "",
            "| Dataset | Cells | SCX (s) | Scanpy (s) | Speedup | Output Size (MB) |",
            "|---------|-------|---------|------------|---------|------------------|",
        ]
        for r in bench_data:
            lines.append(
                f"| {r['dataset']} | {r['n_obs']:,} | {r['scx_median_s']:.2f} | "
                f"{r['scanpy_median_s']:.2f} | {r['speedup']:.1f}x | "
                f"{r['scx_output_file_size_mb']:.1f} |"
            )
        lines.append("")

    report = "\n".join(lines)
    md_path = RESULTS_DIR / "accel_preprocessing_benchmark.md"
    md_path.write_text(report)
    print(f"\nReport saved: {md_path}")
    print(report)
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="SCX Preprocessing Benchmark (Phase 4a)"
    )
    parser.add_argument(
        "--mode", default="all",
        choices=["validate", "bench", "all", "report"],
        help="Mode (default: all)",
    )
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

    if args.mode in ("validate", "all"):
        run_validation(datasets=args.datasets)

    if args.mode in ("bench", "all"):
        run_benchmark(datasets=args.datasets, n_runs=args.n_runs)

    generate_report()


if __name__ == "__main__":
    main()
