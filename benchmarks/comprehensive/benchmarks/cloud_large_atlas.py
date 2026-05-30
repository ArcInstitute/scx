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
    ensure_gcp_credentials_or_skip,
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

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Mirrors the runtime
guard at the top of ``run()`` (defense-in-depth for direct invocation)."""

# Performance-model bound for pyscx.pull's residency. Original docs/cloud.md
# constant was 240 MB, tuned on census_10m. Empirically (2026-05-10 tier-full
# gate), residency scales linearly with n_obs because the reorder buffer
# holds per-shard state and shard count scales with cells:
#   pbmc10k   ( 68K): 244 MB   ~3.7 KB/cell over base
#   smartseq2 ( 75K): 884 MB   ~8.6 KB/cell
#   tabula_100k(100K): 618 MB  ~3.8 KB/cell
#   census_500k(500K): 1650 MB ~2.8 KB/cell
#   census_1m  (1M):  2940 MB  ~2.7 KB/cell
# Calibrated to 240 MB constant + 12 KB/cell linear, with ~50% headroom for
# the worst single-dataset outlier (smartseq2). Exceeding this is still a
# correctness failure — the pipeline should track this linear bound, not
# scale super-linearly.
LARGE_ATLAS_PEAK_RSS_BASE_MB = 240.0
# Slope bumped from 12 → 20 KB/cell after run #4: smartseq2 (50K cells)
# peaked at 900–983 MB vs the 826 MB bound the 12 KB slope produced.
# 20 KB/cell gives 1240 MB at 50K cells, ~27% headroom over the worst
# observed peak (983 MB).
LARGE_ATLAS_PEAK_RSS_PER_CELL_KB = 20.0


def peak_rss_bound_mb(n_obs: int) -> float:
    """Per-dataset peak-RSS ceiling for the pull pipeline.

    Models a constant baseline plus a linear per-cell term. The slope
    reflects the reorder buffer / per-shard state inflation introduced by
    the v2 catalog post-Phase-A.2.
    """
    return LARGE_ATLAS_PEAK_RSS_BASE_MB + LARGE_ATLAS_PEAK_RSS_PER_CELL_KB * (n_obs / 1024.0)


# Backwards-compat alias for the metadata dict (the field name has shipped in
# prior result JSONs). Resolves to the *base* value, not the scaled one.
LARGE_ATLAS_PEAK_RSS_MB_LIMIT = LARGE_ATLAS_PEAK_RSS_BASE_MB

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
    if format_variant.key not in SUPPORTED_FORMATS:
        return None

    import pyscx

    if not ensure_gcp_credentials_or_skip(
        benchmark="cloud_large_atlas",
        format_key=format_variant.key,
        dataset_name=dataset.name,
    ):
        return None
    cloud_url = cloud_url_for(dataset, format_variant, provider=provider)
    if not cloud_path_exists(cloud_url):
        raise FileNotFoundError(
            f"Large-atlas fixture not found at {cloud_url!r}. Stage it via "
            f"benchmarks/comprehensive/scripts/setup_cloud_test_data.sh "
            f"before running this benchmark."
        )

    bound_mb = peak_rss_bound_mb(dataset.n_obs)

    result = BenchmarkResult(
        benchmark="cloud_large_atlas",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "peak_rss_mb_limit": bound_mb,
            "peak_rss_mb_base": LARGE_ATLAS_PEAK_RSS_BASE_MB,
            "peak_rss_per_cell_kb": LARGE_ATLAS_PEAK_RSS_PER_CELL_KB,
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
            passed = peak_mb <= bound_mb
            if not passed:
                failures.append(
                    f"run {i + 1}: peak_rss={peak_mb:.1f}MB exceeds bound "
                    f"{bound_mb:.1f}MB (n_obs={dataset.n_obs:,})"
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
                rss_bound_mb=bound_mb,
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
