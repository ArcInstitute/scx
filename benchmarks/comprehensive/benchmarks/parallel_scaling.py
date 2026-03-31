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
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)

# Formats that are inherently single-threaded — skip thread counts > 1
_SINGLE_THREADED_RUNNERS = {"h5ad_runner"}

# Environment variables controlling thread pools for various libraries
_THREAD_ENV_VARS = [
    "RAYON_NUM_THREADS",       # Rust rayon (SCX)
    "OMP_NUM_THREADS",         # OpenMP (used by some HDF5/NumPy backends)
    "OPENBLAS_NUM_THREADS",    # OpenBLAS
    "MKL_NUM_THREADS",         # Intel MKL
    "NUMEXPR_MAX_THREADS",     # numexpr
]


def _make_runner(fmt: FormatVariant) -> FormatRunner:
    """Instantiate the appropriate runner for a format variant."""
    from benchmarks.comprehensive.runners.h5ad_runner import H5adRunner
    from benchmarks.comprehensive.runners.zarr_runner import ZarrRunner
    from benchmarks.comprehensive.runners.tiledb_runner import TileDBRunner
    from benchmarks.comprehensive.runners.scx_runner import ScxRunner
    from benchmarks.comprehensive.runners.bpcells_runner import BPCellsRunner
    from benchmarks.comprehensive.runners.parquet_runner import ParquetRunner

    runners: dict[str, type[FormatRunner]] = {
        "h5ad_runner": H5adRunner,
        "zarr_runner": ZarrRunner,
        "tiledb_runner": TileDBRunner,
        "scx_runner": ScxRunner,
        "bpcells_runner": BPCellsRunner,
        "parquet_runner": ParquetRunner,
    }
    cls = runners[fmt.runner]
    return cls(**fmt.params)


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
    runner = _make_runner(format_variant)
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

    with tempfile.TemporaryDirectory(prefix=f"scx_bench_{dataset.name}_") as tmp_dir:
        converted_path = Path(tmp_dir) / f"{dataset.name}.{format_variant.key}"

        # -- Convert h5ad to target format --
        logger.info(
            "Converting %s -> %s (%s)",
            h5ad_path.name,
            format_variant.name,
            converted_path,
        )
        runner.convert_from_h5ad(h5ad_path, converted_path)

        # -- Record file size --
        result.file_size_bytes = runner.file_size(converted_path)
        logger.info(
            "Converted file size: %.2f MB",
            result.file_size_bytes / (1024 * 1024),
        )

        # Per-thread-count median timings for the scaling summary
        scaling_summary: dict[str, float] = {}

        for thread_count in thread_counts:
            logger.info(
                "--- Thread count: %d (format: %s, dataset: %s) ---",
                thread_count,
                format_variant.key,
                dataset.name,
            )

            prev_env = _set_thread_count(thread_count)
            try:
                # -- Warm-up run(s) --
                for i in range(N_WARMUP_RUNS):
                    logger.info(
                        "  Warm-up run %d/%d (threads=%d)",
                        i + 1, N_WARMUP_RUNS, thread_count,
                    )
                    runner.read_full(converted_path)
                    gc.collect()

                # -- Timed runs --
                wall_times: list[float] = []
                for i in range(n_runs):
                    if cold_cache:
                        runner._drop_caches()

                    gc.collect()

                    logger.info(
                        "  Timed run %d/%d (threads=%d)",
                        i + 1, n_runs, thread_count,
                    )
                    timing = runner.read_full(converted_path)

                    result.add_run(
                        wall_s=timing.wall_s,
                        user_s=timing.user_s,
                        sys_s=timing.sys_s,
                        peak_rss_mb=timing.peak_rss_mb,
                        threads=thread_count,
                    )
                    wall_times.append(timing.wall_s)
                    logger.info(
                        "    wall=%.3fs  rss=%.1fMB",
                        timing.wall_s,
                        timing.peak_rss_mb,
                    )

                median_wall = statistics.median(wall_times)
                scaling_summary[str(thread_count)] = round(median_wall, 6)
                logger.info(
                    "  Median wall time (threads=%d): %.3fs",
                    thread_count,
                    median_wall,
                )
            finally:
                _restore_thread_env(prev_env)

        # -- Compute speedup and parallel efficiency --
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

        logger.info(
            "Parallel scaling complete: %s / %s",
            format_variant.key,
            dataset.name,
        )
        if speedup:
            max_tc = max(scaling_summary.keys(), key=int)
            logger.info(
                "  Speedup at %s threads: %.2fx  (efficiency: %.1f%%)",
                max_tc,
                speedup.get(max_tc, 0.0),
                efficiency.get(max_tc, 0.0) * 100,
            )

    return result
