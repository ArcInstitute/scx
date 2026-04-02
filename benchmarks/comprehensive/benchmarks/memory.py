"""
Memory Efficiency benchmark — COMPREHENSIVE-BENCHMARKING.md SS3.7.

Measures current RSS (not cumulative peak) before and after key operations
to quantify per-operation memory overhead across all format variants.

Operations measured:
  1. read_full  — full expression matrix load
  2. read_subset_1k — selective read of QUERY_N_CELLS random cells
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    QUERY_N_CELLS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


def _current_rss_mb() -> float:
    """Read *current* RSS from /proc/self/statm (Linux only).

    Unlike ``resource.getrusage().ru_maxrss`` which is a monotonically
    increasing high-water mark, this returns the actual resident set size
    at the moment of the call.
    """
    try:
        with open("/proc/self/statm") as f:
            parts = f.read().split()
            # Field 1 (index 1) is resident set size in pages.
            return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return 0.0


def _measure_operation(runner, op_name: str, op_fn, *args, **kwargs):
    """Run *op_fn* and return (timing, baseline_rss, peak_rss, delta_rss).

    Performs a full GC and records RSS before and after the operation so the
    delta reflects only the memory consumed by that operation.
    """
    gc.collect()
    baseline_rss = _current_rss_mb()

    timing = op_fn(*args, **kwargs)

    peak_rss = _current_rss_mb()
    delta_rss = max(peak_rss - baseline_rss, 0.0)

    # Ensure any intermediate objects held by the runner are released.
    gc.collect()

    return timing, baseline_rss, peak_rss, delta_rss


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Execute the memory efficiency benchmark.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to benchmark against.
    format_variant : FormatVariant
        Target format (e.g. SCX auto, h5ad gzip, Zarr zstd).
    n_runs : int
        Number of timed iterations (after warm-up).
    cold_cache : bool
        If True, drop OS page caches before each timed run.

    Returns
    -------
    BenchmarkResult
        Structured result with per-run memory measurements and metadata.
    """
    runner = make_runner(format_variant)
    h5ad_path = dataset.h5ad_path

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. "
            f"Run dataset preparation first."
        )

    result = BenchmarkResult(
        benchmark="memory",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "query_n_cells": QUERY_N_CELLS,
        },
    )

    rng = np.random.default_rng(RANDOM_SEED)
    n_cells = min(QUERY_N_CELLS, dataset.n_obs)
    cell_indices = np.sort(rng.choice(dataset.n_obs, size=n_cells, replace=False))

    # Use pre-converted file if available, otherwise convert to temp dir.
    _cleanup = None
    if converted_path is not None and Path(converted_path).exists():
        _converted = Path(converted_path)
    else:
        _cleanup = tempfile.TemporaryDirectory(prefix=f"scx_bench_mem_{dataset.name}_")
        _converted = Path(_cleanup.name) / f"{dataset.name}.{format_variant.key}"
        logger.info(
            "Converting %s -> %s (%s)",
            h5ad_path.name, format_variant.name, _converted,
        )
        runner.convert_from_h5ad(h5ad_path, _converted)

    result.file_size_bytes = runner.file_size(_converted)

    try:
        for i in range(N_WARMUP_RUNS):
            runner.read_full(_converted)
            gc.collect()

        read_full_deltas: list[float] = []
        read_subset_deltas: list[float] = []

        for i in range(n_runs):
            if cold_cache:
                runner._drop_caches()

            timing, baseline, peak, delta = _measure_operation(
                runner, "read_full", runner.read_full, _converted,
            )
            read_full_deltas.append(delta)
            result.add_run(
                wall_s=timing.wall_s, user_s=timing.user_s,
                sys_s=timing.sys_s, peak_rss_mb=timing.peak_rss_mb,
                operation="read_full",
                baseline_rss_mb=round(baseline, 2),
                current_rss_mb=round(peak, 2),
                delta_rss_mb=round(delta, 2),
            )

            if cold_cache:
                runner._drop_caches()

            timing_sub, baseline_sub, peak_sub, delta_sub = _measure_operation(
                runner, "read_subset_1k", runner.read_subset,
                _converted, cell_indices=cell_indices,
            )
            read_subset_deltas.append(delta_sub)
            result.add_run(
                wall_s=timing_sub.wall_s, user_s=timing_sub.user_s,
                sys_s=timing_sub.sys_s, peak_rss_mb=timing_sub.peak_rss_mb,
                operation="read_subset_1k",
                baseline_rss_mb=round(baseline_sub, 2),
                current_rss_mb=round(peak_sub, 2),
                delta_rss_mb=round(delta_sub, 2),
            )
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    # -- Summarize per-operation median deltas in metadata --
    if read_full_deltas:
        result.metadata["median_delta_rss_mb_read_full"] = round(
            statistics.median(read_full_deltas), 2
        )
    if read_subset_deltas:
        result.metadata["median_delta_rss_mb_read_subset_1k"] = round(
            statistics.median(read_subset_deltas), 2
        )

    logger.info(
        "Memory benchmark complete: %s / %s — median delta read_full=%.1fMB  read_subset_1k=%.1fMB",
        format_variant.key,
        dataset.name,
        result.metadata.get("median_delta_rss_mb_read_full", 0.0),
        result.metadata.get("median_delta_rss_mb_read_subset_1k", 0.0),
    )
    return result
