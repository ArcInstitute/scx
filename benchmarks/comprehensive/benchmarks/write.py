"""
Write Performance (Conversion) benchmark -- COMPREHENSIVE-BENCHMARKING.md S3.2.

Measures h5ad -> format conversion: wall time, peak RSS, output size,
and write throughput for each format variant.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)


def _make_runner(fmt: FormatVariant) -> FormatRunner:
    """Instantiate the runner for *fmt*."""
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
    """Run the write (conversion) benchmark.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to convert.
    format_variant : FormatVariant
        Target format and codec configuration.
    n_runs : int
        Number of timed iterations.
    cold_cache : bool
        If True, drop OS page caches before each run.

    Returns
    -------
    BenchmarkResult with benchmark="write".
    """
    runner = _make_runner(format_variant)
    h5ad_path = dataset.h5ad_path
    source_h5ad_bytes = os.path.getsize(h5ad_path)

    result = BenchmarkResult(
        benchmark="write",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "source_h5ad_bytes": source_h5ad_bytes,
            "cold_cache": cold_cache,
        },
    )

    logger.info(
        "write benchmark: dataset=%s format=%s n_runs=%d cold_cache=%s",
        dataset.name,
        format_variant.key,
        n_runs,
        cold_cache,
    )

    for i in range(n_runs):
        with tempfile.TemporaryDirectory(prefix="scx_write_bench_") as tmpdir:
            output_path = Path(tmpdir) / f"output.{format_variant.key}"

            if cold_cache:
                FormatRunner._drop_caches()

            logger.info("  run %d/%d -> %s", i + 1, n_runs, output_path)
            cr = runner.convert_from_h5ad(h5ad_path, output_path)

            result.add_run(
                wall_s=cr.wall_s,
                peak_rss_mb=cr.peak_rss_mb,
                write_throughput_mb_s=cr.write_throughput_mb_s,
                output_size_bytes=cr.output_size_bytes,
            )
        # TemporaryDirectory cleaned up here -- avoids disk space issues

    return result
