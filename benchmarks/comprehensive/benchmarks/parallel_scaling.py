"""
Parallel Read Scaling benchmark — COMPREHENSIVE-BENCHMARKING.md §3.5.

Measures how read throughput scales with thread count for formats that
support parallel I/O (primarily SCX via rayon). For each thread count in
THREAD_COUNTS, performs n_runs full-file reads and records wall time,
then computes speedup and parallel efficiency relative to single-threaded
performance.
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
import tempfile
from pathlib import Path

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    THREAD_COUNTS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

_SINGLE_THREADED_RUNNERS = {"h5ad_runner"}

_THREAD_ENV_VARS = [
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]


def _set_thread_count(thread_count: int) -> dict[str, str | None]:
    """Set thread-pool environment variables and return the previous values.

    Returns a dict mapping env var name to its previous value (or None if
    it was unset), so the caller can restore them later.
    """
    prev: dict[str, str | None] = {}
    for var in _THREAD_ENV_VARS:
        prev[var] = os.environ.get(var)
        os.environ[var] = str(thread_count)
    return prev


def _restore_thread_env(prev: dict[str, str | None]) -> None:
    """Restore environment variables to their previous values."""
    for var, val in prev.items():
        if val is None:
            os.environ.pop(var, None)
        else:
            os.environ[var] = val


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Execute the parallel read scaling benchmark.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to benchmark against.
    format_variant : FormatVariant
        Target format (e.g. SCX auto, Zarr zstd).
    n_runs : int
        Number of timed iterations per thread count (after warm-up).
    cold_cache : bool
        If True, drop OS page caches before each timed run.

    Returns
    -------
    BenchmarkResult
        Structured result with per-run timings tagged by thread count,
        plus metadata containing scaling summary, speedup, and efficiency.
    """
    runner = make_runner(format_variant)
    h5ad_path = dataset.h5ad_path

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. "
            f"Run dataset preparation first."
        )

    is_single_threaded = format_variant.runner in _SINGLE_THREADED_RUNNERS
    thread_counts = [1] if is_single_threaded else THREAD_COUNTS

    result = BenchmarkResult(
        benchmark="parallel_scaling",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "thread_counts": thread_counts,
            "single_threaded_only": is_single_threaded,
        },
    )

    # Use pre-converted file if available, otherwise convert to temp dir.
    _cleanup = None
    if converted_path is not None and Path(converted_path).exists():
        _converted = Path(converted_path)
    else:
        _cleanup = tempfile.TemporaryDirectory(prefix=f"scx_bench_{dataset.name}_")
        _converted = Path(_cleanup.name) / f"{dataset.name}.{format_variant.key}"
        logger.info(
            "Converting %s -> %s (%s)",
            h5ad_path.name,
            format_variant.name,
            _converted,
        )
        runner.convert_from_h5ad(h5ad_path, _converted)

    result.file_size_bytes = runner.file_size(_converted)
    try:
        scaling_summary: dict[str, float] = {}

        for thread_count in thread_counts:
            logger.info(
                "--- Thread count: %d (format: %s, dataset: %s) ---",
                thread_count, format_variant.key, dataset.name,
            )

            prev_env = _set_thread_count(thread_count)
            try:
                for i in range(N_WARMUP_RUNS):
                    runner.read_full(_converted)
                    gc.collect()

                wall_times: list[float] = []
                for i in range(n_runs):
                    if cold_cache:
                        runner._drop_caches()
                    gc.collect()

                    timing = runner.read_full(_converted)
                    result.add_run(
                        wall_s=timing.wall_s, user_s=timing.user_s,
                        sys_s=timing.sys_s, peak_rss_mb=timing.peak_rss_mb,
                        threads=thread_count,
                    )
                    wall_times.append(timing.wall_s)

                median_wall = statistics.median(wall_times)
                scaling_summary[str(thread_count)] = round(median_wall, 6)
            finally:
                _restore_thread_env(prev_env)

        baseline = scaling_summary.get("1")
        speedup: dict[str, float] = {}
        efficiency: dict[str, float] = {}

        if baseline is not None and baseline > 0:
            for tc_str, median_s in scaling_summary.items():
                tc = int(tc_str)
                sp = baseline / median_s if median_s > 0 else 0.0
                speedup[tc_str] = round(sp, 3)
                efficiency[tc_str] = round(sp / tc, 3) if tc > 0 else 0.0

        result.metadata["scaling_wall_s"] = scaling_summary
        result.metadata["speedup"] = speedup
        result.metadata["efficiency"] = efficiency
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    return result
