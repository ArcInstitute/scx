"""
Cloud filtered-query benchmark.

Cross-format: runs ``runner.read_cloud_filtered_query(url, predicate)`` for
every predicate in ``queries.default_predicates()`` against a runner that
declares the ``"cloud_filtered"`` capability. Mirrors ``read_selective.py``'s
capability-gated loop: silent skip (``None``) for runners without the
capability, contract-violation propagation for runners that advertise it but
fail to implement it.

The result file collapses all per-predicate runs under a single
``BenchmarkResult(benchmark="cloud_filtered", format=..., dataset=...)``
with per-run ``extra`` tagging ``predicate``, ``predicate_literal``, and
``native_mechanism`` so reports can group by ``(dataset, predicate, format)``.

Bytes-transferred and GET-count telemetry are scoped to Phase F.2; each run
carries the placeholder ``telemetry="phase_f_deferred"`` so downstream
reporting won't render empty columns until that infrastructure lands.
"""

from __future__ import annotations

import gc
import logging
import statistics
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
from benchmarks.comprehensive.queries import (
    EqPredicate,
    GtPredicate,
    RandomSamplePredicate,
    default_predicates,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


def _obs_columns(dataset: DatasetConfig) -> set[str]:
    """Return the obs column set for a dataset without loading X.

    A missing source h5ad is a soft-failure path (returns empty set and a
    ``WARNING``) so the benchmark can run on cloud-only deployments.
    """
    import anndata

    try:
        adata = anndata.read_h5ad(dataset.h5ad_path, backed="r")
    except FileNotFoundError:
        logger.warning(
            "Source h5ad not found for %s (%s); skipping all filtered_query "
            "predicates for this dataset.",
            dataset.name, dataset.h5ad_path,
        )
        return set()
    try:
        return set(adata.obs.columns)
    finally:
        adata.file.close()


def _applicable_predicates(dataset: DatasetConfig):
    """Filter ``default_predicates`` to those whose columns exist on *dataset*."""
    cols = _obs_columns(dataset)
    out = []
    for pred in default_predicates():
        if isinstance(pred, RandomSamplePredicate):
            out.append(pred)
        elif isinstance(pred, (EqPredicate, GtPredicate)):
            if pred.column in cols:
                out.append(pred)
            else:
                logger.info(
                    "Skipping predicate %s — obs column %r absent in %s",
                    pred.name, pred.column, dataset.name,
                )
    return out


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
    if "cloud_filtered" not in runner.capabilities:
        logger.info(
            "Skipping cloud_filtered for %s — runner does not declare cloud_filtered",
            format_variant.key,
        )
        return None

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted {format_variant.key} file for {dataset.name}. "
            f"Run conversion first (--formats {format_variant.key})."
        )

    predicates = _applicable_predicates(dataset)
    if not predicates:
        logger.info(
            "Skipping cloud_filtered for %s / %s — no predicates apply to this dataset",
            format_variant.key, dataset.name,
        )
        return None

    require_gcp_credentials()
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    result = BenchmarkResult(
        benchmark="cloud_filtered",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "n_warmup": N_WARMUP_RUNS,
            "cold_cache": cold_cache,
            "predicates": [p.name for p in predicates],
        },
    )
    result.file_size_bytes = runner.file_size(Path(converted_path))

    # One warm-up across the first predicate — JIT / credential caches.
    for i in range(N_WARMUP_RUNS):
        logger.info(
            "Warm-up run %d/%d (%s)", i + 1, N_WARMUP_RUNS, predicates[0].describe(),
        )
        runner.read_cloud_filtered_query(cloud_url, predicates[0])
        gc.collect()

    per_predicate_walls: dict[str, list[float]] = {}
    for predicate in predicates:
        walls = per_predicate_walls.setdefault(predicate.name, [])
        for i in range(n_runs):
            if cold_cache:
                runner._drop_caches()
            gc.collect()

            logger.info(
                "cloud_filtered %s run %d/%d ← %s",
                predicate.name, i + 1, n_runs, cloud_url,
            )
            # Capability is declared — let contract violations propagate.
            timing = runner.read_cloud_filtered_query(cloud_url, predicate)
            extra = dict(timing.extra or {})
            extra.setdefault("provider", provider)
            extra.setdefault("telemetry", "phase_f_deferred")
            extra["predicate_name"] = predicate.name
            extra["predicate_literal"] = predicate.describe()
            result.add_run(
                wall_s=timing.wall_s,
                user_s=timing.user_s,
                sys_s=timing.sys_s,
                peak_rss_mb=timing.peak_rss_mb,
                **extra,
            )
            walls.append(timing.wall_s)
            logger.info(
                "  wall=%.3fs  rss=%.1fMB  mechanism=%s",
                timing.wall_s, timing.peak_rss_mb,
                extra.get("native_mechanism", "unknown"),
            )

    # Per-predicate summary so reporting doesn't need to re-aggregate.
    summary: dict[str, dict[str, float]] = {}
    for pname, walls in per_predicate_walls.items():
        if not walls:
            continue
        summary[pname] = {
            "median_s": round(statistics.median(walls), 6),
            "min_s": round(min(walls), 6),
            "max_s": round(max(walls), 6),
            "p95_s": round(
                sorted(walls)[max(0, int(0.95 * len(walls)) - 1)], 6,
            ),
            "n_runs": len(walls),
        }
    result.metadata["per_predicate_summary"] = summary

    logger.info(
        "cloud_filtered complete: %s / %s — %d predicates × %d runs",
        format_variant.key, dataset.name, len(predicates), n_runs,
    )
    return result
