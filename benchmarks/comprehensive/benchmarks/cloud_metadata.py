"""
Cloud metadata-open latency benchmark.

Cross-format metadata-only open latency:
  * SCX: ``pyscx.open_cloud(url)`` + touch ``n_obs / n_vars / nnz``
  * Zarr: ``zarr.open(url)`` + touch ``attrs['shape']``
  * TileDB-SOMA: ``Experiment.open(url)`` + touch obs/var counts
  * SLAF: ``SLAFArray(url)`` + touch ``shape``

Each runner's ``read_cloud_metadata(url)`` helper does the minimal open
and touches only schema-level properties; array reads are excluded so the
benchmark measures first-GET / catalog-parse latency, not bandwidth.
Returns ``None`` for formats whose runner does not provide the helper
(silent skip).
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import (
    ensure_cloud_fixture,
    ensure_gcp_credentials_or_skip,
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

REQUIRED_CAPABILITIES: frozenset[str] = frozenset({"cloud_metadata"})
"""Runner-capability requirement — read by ``run_parallel.py``'s cohort
builder so incompatible (bench, format) cells never get submitted. Mirrors
the runtime guard at the top of ``run()`` (defense-in-depth for direct
invocation)."""

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — the only SCX layout with a cloud fixture suffix
(``config._FORMAT_KEY_TO_CLOUD_SUFFIX``). The Scx1 ``compact_trial_g*``
variants declare the ``cloud_metadata`` capability (via ``scx_runner``) but
have no cloud layout, so ``DatasetConfig.cloud_url`` raises ``ValueError``
for them — pinning the allow-list keeps them out of the cohort (a failed
cloud job otherwise breaks the next cohort's ``afterok:`` dependency). Read
by ``run_parallel.py``'s cohort builder; mirrored by the runtime guard in
``run()``."""


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    provider: str = "gcs",
) -> BenchmarkResult | None:
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    if provider != "gcs":
        raise ValueError(
            f"Only 'gcs' provider is supported in Phase 5 (got {provider!r})"
        )

    runner = make_runner(format_variant)
    if "cloud_metadata" not in runner.capabilities:
        logger.info(
            "Skipping cloud_metadata for %s — runner does not declare cloud_metadata",
            format_variant.key,
        )
        return None
    open_metadata = runner.read_cloud_metadata

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted {format_variant.key} file for {dataset.name}. "
            f"Run conversion first (--formats {format_variant.key})."
        )

    if not ensure_gcp_credentials_or_skip(
        benchmark="cloud_metadata",
        format_key=format_variant.key,
        dataset_name=dataset.name,
    ):
        return None
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    result = BenchmarkResult(
        benchmark="cloud_metadata",
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
        open_metadata(cloud_url)
        gc.collect()

    for i in range(n_runs):
        if cold_cache:
            runner._drop_caches()
        gc.collect()

        logger.info("cloud_metadata run %d/%d ← %s", i + 1, n_runs, cloud_url)
        timing = open_metadata(cloud_url)
        result.add_run(
            wall_s=timing.wall_s,
            user_s=timing.user_s,
            sys_s=timing.sys_s,
            peak_rss_mb=timing.peak_rss_mb,
            **(timing.extra or {}),
        )
        logger.info("  wall=%.6fs", timing.wall_s)

    logger.info(
        "cloud_metadata complete: %s / %s — median %.6fs",
        format_variant.key,
        dataset.name,
        result.median_wall_s or 0.0,
    )
    return result
