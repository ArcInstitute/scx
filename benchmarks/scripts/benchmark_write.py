#!/usr/bin/env python3
"""Benchmark write performance: h5ad → SCX conversion (Task 17.4)."""

import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

from bench_env import WORK_DIR
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


BENCHMARK_DATASETS = [
    "pbmc3k", "smartseq2", "tabula_sapiens_100k", "census_1m",
]


def benchmark_write_per_codec(h5ad_path):
    """Benchmark write with each codec."""
    import anndata
    import pyscx

    h5ad_size = h5ad_path.stat().st_size
    adata = anndata.read_h5ad(str(h5ad_path))

    result = {
        "dataset": h5ad_path.stem,
        "h5ad_size_mb": h5ad_size / 1e6,
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
    }

    for codec in ["auto", "none", "scx1", "zstd"]:
        times = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(N_REPEATS):
                scx_path = os.path.join(tmpdir, f"test_{codec}_{i}.scx")
                t0 = time.perf_counter()
                try:
                    pyscx.from_anndata(adata, scx_path, codec=codec)
                    times.append(time.perf_counter() - t0)
                except Exception as e:
                    result[f"write_{codec}_s"] = None
                    result[f"write_{codec}_mb_s"] = None
                    break
        if times:
            median_time = _median(times)
            result[f"write_{codec}_s"] = round(median_time, 3)
            result[f"write_{codec}_mb_s"] = round((h5ad_size / 1e6) / median_time, 1) if median_time > 0 else 0

    # Default "write_time_s" and "mb_per_s" for auto codec
    result["write_time_s"] = result.get("write_auto_s", 0)
    result["mb_per_s"] = result.get("write_auto_mb_s", 0)

    del adata
    import gc; gc.collect()
    return result


def run_all():
    """Run write benchmarks on benchmark datasets."""
    data_path = Path(WORK_DIR)

    results = []
    for name in BENCHMARK_DATASETS:
        h5ad_path = data_path / f"{name}.h5ad"
        if not h5ad_path.exists():
            print(f"SKIP: {name}.h5ad not found")
            continue
        print(f"Benchmarking write: {h5ad_path.name}...")
        try:
            result = benchmark_write_per_codec(h5ad_path)
            results.append(result)
            for codec in ["auto", "none", "scx1", "zstd"]:
                t = result.get(f"write_{codec}_s")
                m = result.get(f"write_{codec}_mb_s")
                if t:
                    print(f"  {codec:5s}: {t:.3f}s ({m:.1f} MB/s)")
                else:
                    print(f"  {codec:5s}: FAILED")
        except Exception as e:
            print(f"  ERROR: {e}")

    return results


if __name__ == "__main__":
    ensure_release_build()
    run_all()
