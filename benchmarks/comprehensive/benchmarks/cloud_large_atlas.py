"""
Large-atlas streaming pull (Phase F.5).

SCX-only. Streams a 50 GB+ atlas from GCS via ``pyscx.pull`` and asserts
peak RSS stays within the performance-model bound published in
``docs/cloud.md`` (≈240 MB). The assertion is the correctness check —
a streaming pull that balloons RSS as the file grows indicates a
regression in the reorder-buffer / streaming-write pipeline (see
``scx-cloud/src/pull.rs``).

Fixture staging is out-of-band — see
``benchmarks/comprehensive/scripts/setup_cloud_test_data.sh`` for the
one-time upload ritual. This module fails fast if the fixture is absent
rather than auto-uploading (a 50GB upload is not a thing we want to
trigger implicitly from a benchmark run).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import (
    CloudIOCounters,
    cloud_path_exists,
    cloud_url_for,
    require_gcp_credentials,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_TEST_BUCKET,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)

_SCX_TRIGGER_KEY = "scx_auto"

# Performance-model bound from docs/cloud.md. Exceeding this is a
# correctness failure — the pull pipeline's memory footprint is supposed
# to be bounded by the reorder buffer + parallelism, not by file size.
LARGE_ATLAS_PEAK_RSS_MB_LIMIT = 240.0

# RSS sampling cadence during the pull. 100 ms is short enough to catch
# transient spikes (e.g. during the catalog rewrite at EOF) without
# dominating CPU overhead on the benchmark host.
_RSS_POLL_INTERVAL_S = 0.1


class _RssSampler(threading.Thread):
    """Polls /proc/self/status RSS at a fixed cadence into a peak tracker."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.peak_mb: float = 0.0
        self.samples: int = 0

    def run(self) -> None:
        while not self._stop.is_set():
            rss = FormatRunner._get_rss_mb()
            if rss > self.peak_mb:
                self.peak_mb = rss
            self.samples += 1
            self._stop.wait(_RSS_POLL_INTERVAL_S)

    def stop(self) -> float:
        self._stop.set()
        self.join(timeout=_RSS_POLL_INTERVAL_S * 5)
        return self.peak_mb


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
    if format_variant.key != _SCX_TRIGGER_KEY:
        return None

    import pyscx

    require_gcp_credentials()
    cloud_url = cloud_url_for(dataset, format_variant, provider=provider)
    if not cloud_path_exists(cloud_url):
        raise FileNotFoundError(
            f"Large-atlas fixture not found at {cloud_url!r}. Stage it via "
            f"benchmarks/comprehensive/scripts/setup_cloud_test_data.sh "
            f"before running this benchmark."
        )

    result = BenchmarkResult(
        benchmark="cloud_large_atlas",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "peak_rss_mb_limit": LARGE_ATLAS_PEAK_RSS_MB_LIMIT,
            "rss_sample_interval_s": _RSS_POLL_INTERVAL_S,
        },
    )

    failures: list[str] = []
    for i in range(n_runs):
        logger.info(
            "cloud_large_atlas run %d/%d ← %s", i + 1, n_runs, cloud_url,
        )
        with tempfile.TemporaryDirectory(prefix="scx_large_atlas_") as tmp:
            local = os.path.join(tmp, "atlas.scx")
            sampler = _RssSampler()
            sampler.start()
            # Pre-initialize so an exception inside pyscx.pull doesn't
            # shadow the real error with an UnboundLocalError when the
            # finally/record_pull_stats branch runs below.
            stats: dict = {}
            t0 = time.perf_counter()
            try:
                stats = pyscx.pull(cloud_url, local)
            finally:
                peak_mb = sampler.stop()
            wall = time.perf_counter() - t0

            counters = CloudIOCounters()
            counters.record_pull_stats(stats)
            passed = peak_mb <= LARGE_ATLAS_PEAK_RSS_MB_LIMIT
            if not passed:
                failures.append(
                    f"run {i + 1}: peak_rss={peak_mb:.1f}MB exceeds bound "
                    f"{LARGE_ATLAS_PEAK_RSS_MB_LIMIT:.1f}MB"
                )

            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak_mb,
                bytes_downloaded=counters.bytes_downloaded,
                sections_downloaded=counters.sections_downloaded,
                throughput_mbps=float(stats.get("throughput_mbps", 0.0)),
                pyscx_elapsed_secs=float(stats.get("elapsed_secs", wall)),
                rss_samples=sampler.samples,
                rss_bound_passed=passed,
                rss_bound_mb=LARGE_ATLAS_PEAK_RSS_MB_LIMIT,
            )
            logger.info(
                "  wall=%.1fs peak_rss=%.1fMB bytes=%d pass=%s",
                wall, peak_mb, counters.bytes_downloaded, passed,
            )

    result.metadata["rss_bound_failures"] = failures
    if failures:
        # Surface a loud error — the benchmark is a correctness check,
        # not a throughput sweep. Reporting will still land the JSON
        # (useful for diagnostics) but the benchmark outcome is failure.
        raise AssertionError(
            f"cloud_large_atlas peak-RSS bound violated on {dataset.name}: "
            + "; ".join(failures)
        )

    logger.info(
        "cloud_large_atlas complete: %s — %d runs within %.1fMB bound",
        dataset.name, n_runs, LARGE_ATLAS_PEAK_RSS_MB_LIMIT,
    )
    return result
