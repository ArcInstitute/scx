#!/usr/bin/env python3
"""
Worker subprocess for MADV_DONTNEED RSS benchmark.

Runs in a fresh process so ru_maxrss reflects only this workload.
Measures peak RSS and RSS time series during streaming aggregation.

Usage:
    python benchmark_madvise_rss_worker.py <scx_path> <operation>

    operation: "row_sums" or "col_sums"
"""

import json
import resource
import sys
import threading
import time


def _rss_mb():
    """Current RSS in MB from /proc/self/statm (Linux)."""
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * resource.getpagesize() / (1024 * 1024)
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _sample_rss(interval_ms, samples, stop_event):
    """Background thread: sample RSS at regular intervals."""
    while not stop_event.is_set():
        samples.append(_rss_mb())
        time.sleep(interval_ms / 1000)


def main():
    scx_path = sys.argv[1]
    operation = sys.argv[2]

    import numpy as np
    import pyscx

    exp = pyscx.open(scx_path, verify=False)
    adata = exp.to_anndata(backed=True, cache_shards=0)

    # Start RSS sampling (100ms interval)
    samples = []
    stop = threading.Event()
    sampler = threading.Thread(
        target=_sample_rss, args=(100, samples, stop), daemon=True
    )
    sampler.start()

    t0 = time.perf_counter()
    if operation == "row_sums":
        result = np.asarray(adata.X.sum(axis=1)).ravel()
    elif operation == "col_sums":
        result = np.asarray(adata.X.sum(axis=0)).ravel()
    else:
        raise ValueError(f"Unknown operation: {operation}")
    elapsed = time.perf_counter() - t0

    stop.set()
    sampler.join(timeout=1)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB

    json.dump(
        {
            "peak_rss_mb": round(peak_rss, 1),
            "wall_clock_s": round(elapsed, 3),
            "rss_samples_mb": [round(s, 1) for s in samples],
            "n_samples": len(samples),
            "result_sum": float(result.sum()),
            "shape": list(adata.X.shape),
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main()
