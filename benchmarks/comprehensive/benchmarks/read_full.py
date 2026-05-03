"""
Read Performance (Full File Load) benchmark.

Measures wall-clock time and peak RSS for reading an entire expression matrix
into an in-memory CSR, across all format variants.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant, N_WARMUP_RUNS
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Execute the full-file read benchmark.

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
        Structured result with per-run timings, file size, and metadata.
    """
    runner = make_runner(format_variant)
    h5ad_path = dataset.h5ad_path

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. "
            f"Run dataset preparation first."
        )

    result = BenchmarkResult(
        benchmark="read_full",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
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

    try:
        result.file_size_bytes = runner.file_size(_converted)
        logger.info(
            "File size: %.2f MB",
            result.file_size_bytes / (1024 * 1024),
        )

        # -- Warm-up run(s) --
        for i in range(N_WARMUP_RUNS):
            logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
            runner.read_full(_converted)
            gc.collect()

        # -- Timed runs --
        for i in range(n_runs):
            if cold_cache:
                runner._drop_caches()

            gc.collect()

            logger.info("Timed run %d/%d", i + 1, n_runs)
            timing = runner.read_full(_converted)

            result.add_run(
                wall_s=timing.wall_s,
                user_s=timing.user_s,
                sys_s=timing.sys_s,
                peak_rss_mb=timing.peak_rss_mb,
            )
            logger.info(
                "  wall=%.3fs  rss=%.1fMB",
                timing.wall_s,
                timing.peak_rss_mb,
            )
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    logger.info(
        "Benchmark complete: %s / %s — median %.3fs",
        format_variant.key,
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result
