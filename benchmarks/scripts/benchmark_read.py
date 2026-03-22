#!/usr/bin/env python3
"""Benchmark read performance: SCX vs h5ad vs scanpy backed mode.

Measures full read time, peak memory, open time, and parallel scaling.
"""

import gc
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))
sys.path.insert(0, str(Path(__file__).parent))
from build_release import ensure_release_build

DATA_DIR = os.environ.get("SCX_DATA_DIR", "/scratch/ctc/nickyoungblut/scx")
N_WARMUP = 1
N_REPEATS = 3

# Benchmark datasets (skip chunk files)
BENCHMARK_DATASETS = [
    "pbmc3k", "smartseq2", "tabula_sapiens_100k", "census_1m",
]


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _rss_mb():
    """Current peak RSS in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def benchmark_read(dataset_name):
    """Benchmark read times for a single dataset."""
    import anndata
    import pyscx

    h5ad_path = Path(DATA_DIR) / f"{dataset_name}.h5ad"
    scx_path = Path(DATA_DIR) / f"{dataset_name}.scx"

    if not h5ad_path.exists():
        return None
    if not scx_path.exists():
        # Convert on-the-fly
        adata = anndata.read_h5ad(str(h5ad_path))
        pyscx.from_anndata(adata, str(scx_path))
        del adata; gc.collect()

    result = {
        "dataset": dataset_name,
        "h5ad_size_mb": h5ad_path.stat().st_size / 1e6,
        "scx_size_mb": scx_path.stat().st_size / 1e6,
    }

    # --- h5ad full read ---
    for _ in range(N_WARMUP):
        adata = anndata.read_h5ad(str(h5ad_path)); del adata; gc.collect()

    h5ad_times = []
    for _ in range(N_REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        adata = anndata.read_h5ad(str(h5ad_path))
        h5ad_times.append(time.perf_counter() - t0)
        result.setdefault("n_obs", adata.n_obs)
        result.setdefault("n_vars", adata.n_vars)
        del adata; gc.collect()
    result["h5ad_warm_s"] = round(_median(h5ad_times), 4)

    # --- SCX full read ---
    for _ in range(N_WARMUP):
        adata = pyscx.open(str(scx_path)).to_anndata(); del adata; gc.collect()

    scx_times = []
    for _ in range(N_REPEATS):
        gc.collect()
        t0 = time.perf_counter()
        adata = pyscx.open(str(scx_path)).to_anndata()
        scx_times.append(time.perf_counter() - t0)
        del adata; gc.collect()
    result["scx_warm_s"] = round(_median(scx_times), 4)
    result["speedup"] = round(result["h5ad_warm_s"] / result["scx_warm_s"], 2) if result["scx_warm_s"] > 0 else 0

    # --- SCX open time (header + catalog only, no data) ---
    open_times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        exp = pyscx.open(str(scx_path))
        open_times.append(time.perf_counter() - t0)
        del exp
    result["scx_open_s"] = round(_median(open_times), 6)

    # --- Peak memory (tracemalloc) ---
    import tracemalloc

    gc.collect()
    tracemalloc.start()
    adata = pyscx.open(str(scx_path)).to_anndata()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["scx_peak_mb"] = round(peak / 1e6, 1)
    del adata; gc.collect()

    tracemalloc.start()
    adata = anndata.read_h5ad(str(h5ad_path))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["h5ad_peak_mb"] = round(peak / 1e6, 1)
    del adata; gc.collect()

    # --- Parallel read scaling (1, 2, 4, 8 threads) ---
    thread_counts = [1, 2, 4, 8]
    parallel_results = {}
    for threads in thread_counts:
        env = os.environ.copy()
        env["RAYON_NUM_THREADS"] = str(threads)
        # Run in subprocess to control thread count
        cmd = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"), "-c",
            f"""
import time, sys; sys.path.insert(0, '{PROJECT_ROOT}/pyscx')
import pyscx
# Warmup
pyscx.open('{scx_path}').to_anndata()
# Timed
times = []
for _ in range(3):
    t0 = time.perf_counter()
    pyscx.open('{scx_path}').to_anndata()
    times.append(time.perf_counter() - t0)
times.sort()
print(times[1])
"""
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
            t = float(out.stdout.strip())
            parallel_results[threads] = round(t, 4)
        except Exception:
            parallel_results[threads] = None

    result["parallel_scaling"] = parallel_results
    base = parallel_results.get(1)
    if base and base > 0:
        result["parallel_speedups"] = {
            k: round(base / v, 2) if v and v > 0 else None
            for k, v in parallel_results.items()
        }

    return result


def run_all():
    """Run read benchmarks on all datasets."""
    results = []
    for name in BENCHMARK_DATASETS:
        print(f"Benchmarking read: {name}...")
        try:
            result = benchmark_read(name)
            if result is None:
                print(f"  SKIP: {name}.h5ad not found")
                continue
            results.append(result)
            print(
                f"  h5ad: {result['h5ad_warm_s']:.3f}s, "
                f"SCX: {result['scx_warm_s']:.3f}s, "
                f"speedup: {result['speedup']:.1f}x, "
                f"open: {result['scx_open_s']*1000:.1f}ms"
            )
            if "parallel_speedups" in result:
                scaling = ", ".join(
                    f"{k}t={v:.1f}x" for k, v in result["parallel_speedups"].items() if v
                )
                print(f"  parallel: {scaling}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    return results


if __name__ == "__main__":
    ensure_release_build()
    run_all()
