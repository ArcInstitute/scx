#!/usr/bin/env python3
"""Benchmark write performance: h5ad → SCX conversion (Task 17.4)."""

import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))

DATA_DIR = os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx")
N_REPEATS = 3


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def benchmark_write(h5ad_path):
    """Benchmark h5ad → SCX conversion time."""
    import anndata
    import pyscx

    h5ad_size = h5ad_path.stat().st_size
    adata = anndata.read_h5ad(str(h5ad_path))

    times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(N_REPEATS):
            scx_path = os.path.join(tmpdir, f"test_{i}.scx")
            t0 = time.perf_counter()
            pyscx.from_anndata(adata, scx_path)
            times.append(time.perf_counter() - t0)

    median_time = _median(times)
    mb_per_s = (h5ad_size / 1e6) / median_time if median_time > 0 else float("inf")

    return {
        "dataset": h5ad_path.stem,
        "h5ad_size_mb": h5ad_size / 1e6,
        "write_time_s": median_time,
        "mb_per_s": mb_per_s,
    }


def run_all():
    """Run write benchmarks on all datasets."""
    data_path = Path(DATA_DIR)
    h5ad_files = sorted(data_path.glob("*.h5ad")) if data_path.exists() else []
    if not h5ad_files:
        print("No h5ad files found. Run download_datasets.sh first.")
        return []

    results = []
    for h5ad_path in h5ad_files:
        print(f"Benchmarking write: {h5ad_path.name}...")
        try:
            result = benchmark_write(h5ad_path)
            results.append(result)
            print(
                f"  {result['h5ad_size_mb']:.1f} MB in {result['write_time_s']:.3f}s "
                f"({result['mb_per_s']:.1f} MB/s)"
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    run_all()
