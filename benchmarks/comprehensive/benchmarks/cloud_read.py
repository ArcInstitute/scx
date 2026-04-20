"""
Cloud full-read benchmark.

Cross-format: runs ``runner.read_cloud(url)`` for every format that
declares the ``"cloud_read"`` capability. Returns ``None`` for runners
without that capability (silent skip).

Runners that *declare* the capability but fail to implement it propagate
``NotImplementedError`` loudly, per the base-class contract.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import (
    ensure_cloud_fixture,
    require_gcp_credentials,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_TEST_BUCKET,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    provider: str = "gcs",
) -> BenchmarkResult | None:
    if provider != "gcs":
        raise ValueError(
            f"Only 'gcs' provider is supported in Phase 5 (got {provider!r})"
        )

    runner = make_runner(format_variant)
    if "cloud_read" not in runner.capabilities:
        logger.info(
            "Skipping cloud_read for %s — runner does not declare cloud_read",
            format_variant.key,
        )
        return None

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted {format_variant.key} file for {dataset.name}. "
            f"Run conversion first (--formats {format_variant.key})."
        )

    require_gcp_credentials()
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    result = BenchmarkResult(
        benchmark="cloud_read",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "n_warmup": N_WARMUP_RUNS,
            "cold_cache": cold_cache,
        },
    )
    result.file_size_bytes = runner.file_size(Path(converted_path))

    for i in range(N_WARMUP_RUNS):
        logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
        runner.read_cloud(cloud_url)
        gc.collect()

    for i in range(n_runs):
        if cold_cache:
            runner._drop_caches()
        gc.collect()

        logger.info("cloud_read run %d/%d ← %s", i + 1, n_runs, cloud_url)
        timing = runner.read_cloud(cloud_url)
        result.add_run(
            wall_s=timing.wall_s,
            user_s=timing.user_s,
            sys_s=timing.sys_s,
            peak_rss_mb=timing.peak_rss_mb,
            **(timing.extra or {}),
        )
        logger.info("  wall=%.3fs  rss=%.1fMB", timing.wall_s, timing.peak_rss_mb)

    logger.info(
        "cloud_read complete: %s / %s — median %.3fs",
        format_variant.key,
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result
