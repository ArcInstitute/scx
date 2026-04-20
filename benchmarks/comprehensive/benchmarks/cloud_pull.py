"""
Cloud pull throughput benchmark.

SCX-only. Downloads the shared cloud fixture for ``dataset`` via
``pyscx.pull`` into a fresh temp local ``.scx`` on each iteration, recording
wall-clock time, bytes downloaded, and throughput. Selective-pull is covered
by a separate ``cloud_reader_vs_pull`` benchmark (not yet landed).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import (
    ensure_cloud_fixture,
    require_gcp_credentials,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_TEST_BUCKET,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

_SCX_TRIGGER_KEY = "scx_auto"


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    provider: str = "gcs",
) -> BenchmarkResult | None:
    if format_variant.key != _SCX_TRIGGER_KEY:
        return None
    if provider != "gcs":
        raise ValueError(
            f"Only 'gcs' provider is supported in Phase 5 (got {provider!r})"
        )
    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted SCX file for {dataset.name}. "
            f"Run conversion first (--formats scx_auto)."
        )

    require_gcp_credentials()
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    runner = make_runner(format_variant)
    result = BenchmarkResult(
        benchmark="cloud_pull",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "cold_cache": cold_cache,
        },
    )
    result.file_size_bytes = Path(converted_path).stat().st_size

    for i in range(n_runs):
        with tempfile.TemporaryDirectory(prefix=f"scx_cloud_pull_{dataset.name}_") as tmp:
            dest = Path(tmp) / "pulled.scx"
            logger.info("Pull run %d/%d ← %s", i + 1, n_runs, cloud_url)
            timing = runner.pull(cloud_url, dest)
            result.add_run(
                wall_s=timing.wall_s,
                user_s=timing.user_s,
                sys_s=timing.sys_s,
                peak_rss_mb=timing.peak_rss_mb,
                **(timing.extra or {}),
            )
            mbps = (timing.extra or {}).get("throughput_mbps", 0.0)
            logger.info("  wall=%.3fs  throughput=%.1f MB/s", timing.wall_s, mbps)

    logger.info(
        "cloud_pull complete: %s — median %.3fs",
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result
