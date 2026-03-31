"""
Read Performance (Full File Load) benchmark — COMPREHENSIVE-BENCHMARKING.md §3.3.

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
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)


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


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
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
    runner = _make_runner(format_variant)
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

        # -- Warm-up run(s) --
        for i in range(N_WARMUP_RUNS):
            logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
            runner.read_full(converted_path)
            gc.collect()

        # -- Timed runs --
        for i in range(n_runs):
            if cold_cache:
                runner._drop_caches()

            gc.collect()

            logger.info("Timed run %d/%d", i + 1, n_runs)
            timing = runner.read_full(converted_path)

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

    logger.info(
        "Benchmark complete: %s / %s — median %.3fs",
        format_variant.key,
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result
