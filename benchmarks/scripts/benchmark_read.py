#!/usr/bin/env python3
"""Benchmark read performance: SCX vs h5ad (Task 17.3)."""

import os
import resource
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))

DATA_DIR = os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx")
N_WARMUP = 1
N_REPEATS = 3


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def benchmark_read(h5ad_path):
    """Benchmark read times for a single dataset."""
    import anndata
    import pyscx

    result = {"dataset": h5ad_path.stem}

    # Prepare SCX file
    with tempfile.TemporaryDirectory() as tmpdir:
        scx_path = os.path.join(tmpdir, "test.scx")
        adata = anndata.read_h5ad(str(h5ad_path))
        pyscx.from_anndata(adata, scx_path)
        del adata

        # --- Warm cache: h5ad ---
        # Warmup
        for _ in range(N_WARMUP):
            anndata.read_h5ad(str(h5ad_path))

        h5ad_times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            anndata.read_h5ad(str(h5ad_path))
            h5ad_times.append(time.perf_counter() - t0)
        result["h5ad_warm_s"] = _median(h5ad_times)

        # --- Warm cache: SCX ---
        for _ in range(N_WARMUP):
            pyscx.open(scx_path).to_anndata()

        scx_times = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            pyscx.open(scx_path).to_anndata()
            scx_times.append(time.perf_counter() - t0)
        result["scx_warm_s"] = _median(scx_times)

        result["speedup"] = result["h5ad_warm_s"] / result["scx_warm_s"] if result["scx_warm_s"] > 0 else float("inf")

        # --- Peak RSS ---
        import tracemalloc

        tracemalloc.start()
        pyscx.open(scx_path).to_anndata()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["scx_peak_mb"] = peak / 1e6

        tracemalloc.start()
        anndata.read_h5ad(str(h5ad_path))
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result["h5ad_peak_mb"] = peak / 1e6

    return result


def run_all():
    """Run read benchmarks on all datasets."""
    data_path = Path(DATA_DIR)
    h5ad_files = sorted(data_path.glob("*.h5ad")) if data_path.exists() else []
    if not h5ad_files:
        print("No h5ad files found. Run download_datasets.sh first.")
        return []

    results = []
    for h5ad_path in h5ad_files:
        print(f"Benchmarking read: {h5ad_path.name}...")
        try:
            result = benchmark_read(h5ad_path)
            results.append(result)
            print(
                f"  h5ad: {result['h5ad_warm_s']:.3f}s, "
                f"SCX: {result['scx_warm_s']:.3f}s, "
                f"speedup: {result['speedup']:.1f}x"
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    run_all()
