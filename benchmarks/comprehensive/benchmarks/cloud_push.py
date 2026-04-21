"""
Cloud push throughput benchmark.

SCX-only. Uploads a local ``.scx`` file to a unique per-run path under
``GCS_TEST_BUCKET`` using ``pyscx.push`` (exploded ``.scxd/`` layout) and
records wall-clock time, bytes uploaded, and throughput.

Uses a unique destination per run so every iteration measures a real
upload — deduplication at the bucket side would otherwise skew repeat runs.
Destinations are cleaned up in a ``finally`` block via ``gsutil rm -r``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import require_gcp_credentials
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_TEST_BUCKET,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

_SCX_TRIGGER_KEY = "scx_auto"


def _gsutil_rm_trees(urls: list[str]) -> None:
    """Best-effort cleanup of one or more cloud directories.

    Batched so the per-call ``gsutil`` startup cost amortizes — with
    ``n_runs=3`` the sequential pattern paid the ~1s Python-launch +
    auth-handshake 3 times per benchmark invocation.
    """
    if not urls or shutil.which("gsutil") is None:
        return
    subprocess.run(
        ["gsutil", "-m", "rm", "-r", *urls],
        capture_output=True, text=True, timeout=300,
    )


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    provider: str = "gcs",
) -> BenchmarkResult | None:
    """Measure SCX push throughput to GCS.

    Returns ``None`` for non-SCX formats (mirrors ``fragment_ops.py``) so
    the orchestrator silently skips those triples.
    """
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

    runner = make_runner(format_variant)
    local_path = Path(converted_path)
    file_size = local_path.stat().st_size

    result = BenchmarkResult(
        benchmark="cloud_push",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "n_runs": n_runs,
            "cold_cache": cold_cache,
        },
    )
    result.file_size_bytes = file_size

    upload_urls: list[str] = []
    try:
        run_tag = f"bench_push_{int(time.time())}"
        for i in range(n_runs):
            dest = f"{GCS_TEST_BUCKET}/{run_tag}_{dataset.name}_{i}.scxd/"
            upload_urls.append(dest)
            logger.info("Push run %d/%d → %s", i + 1, n_runs, dest)
            timing = runner.push(str(local_path), dest)
            result.add_run(
                wall_s=timing.wall_s,
                user_s=timing.user_s,
                sys_s=timing.sys_s,
                peak_rss_mb=timing.peak_rss_mb,
                **(timing.extra or {}),
            )
            mbps = (timing.extra or {}).get("throughput_mbps", 0.0)
            logger.info("  wall=%.3fs  throughput=%.1f MB/s", timing.wall_s, mbps)
    finally:
        _gsutil_rm_trees(upload_urls)

    logger.info(
        "cloud_push complete: %s — median %.3fs",
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result
