"""
CloudReader vs full-pull benchmark (Phase F.1).

SCX-only. Two scenarios, both reporting wall-clock + bytes-downloaded +
GET-count proxy:

  * ``metadata_only`` — ``pyscx.open_cloud(url)`` (single GET against the
    front catalog) compared against a full ``pyscx.pull(url, dest)``.
    Captures the magnitude of "don't-download-the-whole-dataset" for
    metadata-only workloads (``scx info``).

  * ``selective_*`` at ``~5%``, ``~20%``, ``~80%`` cell selectivity.
    Three methods at each selectivity:
      - ``pull_full`` — full ``pyscx.pull`` followed by a local filter
        (the baseline cost of downloading everything).
      - ``pull_filtered`` — ``pyscx.pull`` with ``filter=`` (downloads
        only matching shards, writes them to local disk).
      - ``open_cloud_query`` — Phase 7's selective query path:
        ``pyscx.open_cloud(url).query().filter_obs(filter).collect()``
        — no local materialisation, results land directly in memory.
    Scores the break-even selectivity where a full pull starts winning
    and isolates query-vs-pull overhead at the same selectivity.

The predicate for the selectivity sweep is synthesized from the dataset's
obs table: we sort by ``n_counts`` (when present) and pick a threshold
that yields the target fraction. When ``n_counts`` is absent we fall back
to a ``cell_integer_id`` modulo predicate on the SCX-synthesized
``row_id`` column — exact selectivity, no obs dependency. Selectivity
targets that round to < 1 cell are skipped.

Emits one ``BenchmarkResult`` per ``(dataset, scx_auto)`` with per-run
``extra`` tagging ``scenario`` (``metadata_only`` / ``selective_5pct`` /
``selective_20pct`` / ``selective_80pct``) and ``method``
(``open_cloud`` / ``pull_full`` / ``pull_filtered``).
"""

from __future__ import annotations

import gc
import logging
import os
import statistics
import tempfile
from pathlib import Path

from benchmarks.comprehensive.cloud_fixtures import (
    CloudIOCounters,
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
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)

_SCX_TRIGGER_KEY = "scx_auto"

_SELECTIVITY_TARGETS = (0.05, 0.20, 0.80)


def _threshold_predicate_for_fraction(
    dataset: DatasetConfig, fraction: float,
) -> tuple[str | None, str]:
    """Return a ``(filter_expr, describe)`` pair targeting ~fraction of cells.

    Uses an ``n_counts`` threshold chosen via numpy quantile (present on
    most census datasets). Returns ``(None, reason)`` when the column is
    absent or the h5ad isn't accessible — SCX's ``filter_obs`` parser
    doesn't support arithmetic expressions like ``row_id % N == 0``, so
    there's no stride-hash fallback for this benchmark.
    """
    import anndata
    import numpy as np

    try:
        adata = anndata.read_h5ad(dataset.h5ad_path, backed="r")
    except FileNotFoundError:
        return (
            None,
            f"h5ad not found for {dataset.name} — cannot synthesize predicate",
        )

    try:
        if "n_counts" in adata.obs.columns:
            col = np.asarray(adata.obs["n_counts"])
            cutoff = float(np.quantile(col, 1.0 - fraction))
            return (
                f"n_counts > {cutoff}",
                f"n_counts > {cutoff:.2f} (~{fraction * 100:.0f}%)",
            )
    finally:
        adata.file.close()

    return (
        None,
        f"n_counts absent on {dataset.name} — cannot synthesize predicate",
    )


def _time_call(fn, *args, **kwargs) -> tuple[object, float, float]:
    """Run ``fn`` under ``timed_run``; return (result, wall_s, peak_rss_mb)."""
    import time
    gc.collect()
    rss_before = FormatRunner._get_rss_mb()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    rss_after = FormatRunner._get_rss_mb()
    return result, wall, max(rss_before, rss_after)


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

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted SCX file for {dataset.name}. "
            f"Run Phase A conversion first (--formats scx_auto)."
        )

    import pyscx

    if not ensure_gcp_credentials_or_skip(
        benchmark="cloud_reader_vs_pull",
        format_key=format_variant.key,
        dataset_name=dataset.name,
    ):
        return None
    runner = make_runner(format_variant)
    cloud_url = ensure_cloud_fixture(
        dataset, format_variant, Path(converted_path), provider=provider,
    )

    result = BenchmarkResult(
        benchmark="cloud_reader_vs_pull",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "cloud_url": cloud_url,
            "n_runs": n_runs,
            "n_warmup": N_WARMUP_RUNS,
            "cold_cache": cold_cache,
            "selectivity_targets": list(_SELECTIVITY_TARGETS),
        },
    )
    result.file_size_bytes = runner.file_size(Path(converted_path))

    # -----------------------------------------------------------------
    # Scenario 1: metadata-only — open_cloud vs full pull
    # -----------------------------------------------------------------
    logger.info("metadata_only: %s", dataset.name)
    for i in range(n_runs):
        gc.collect()
        logger.info("  open_cloud run %d/%d", i + 1, n_runs)
        handle, wall, rss = _time_call(pyscx.open_cloud, cloud_url)
        # Touch common metadata fields so the timing includes catalog parse.
        n_obs = handle.n_obs
        n_vars = handle.n_vars
        nnz = handle.nnz
        # open_cloud doesn't expose bytes_downloaded today; record 0 and
        # tag the telemetry gap for Phase F.2 follow-up.
        result.add_run(
            wall_s=wall, peak_rss_mb=rss,
            scenario="metadata_only",
            method="open_cloud",
            n_obs=int(n_obs),
            n_vars=int(n_vars),
            nnz=int(nnz),
            bytes_downloaded=0,
            get_count_proxy=1,
            telemetry="phase_f_deferred",
        )
        logger.info("    open_cloud wall=%.3fs", wall)

        # Full pull — bytes + sections come from the stats dict.
        with tempfile.TemporaryDirectory(prefix="scx_f1_pull_") as tmp:
            local = os.path.join(tmp, "pulled.scx")
            logger.info("  pull_full run %d/%d", i + 1, n_runs)
            stats, wall, rss = _time_call(pyscx.pull, cloud_url, local)
            counters = CloudIOCounters()
            counters.record_pull_stats(stats)
            result.add_run(
                wall_s=wall, peak_rss_mb=rss,
                scenario="metadata_only",
                method="pull_full",
                bytes_downloaded=counters.bytes_downloaded,
                get_count_proxy=counters.sections_downloaded,
                pyscx_elapsed_secs=float(stats.get("elapsed_secs", wall)),
                throughput_mbps=float(stats.get("throughput_mbps", 0.0)),
            )
            logger.info("    pull_full wall=%.3fs bytes=%d",
                        wall, counters.bytes_downloaded)

    # -----------------------------------------------------------------
    # Scenario 2: predicate-selective sweep — pull_filtered vs pull_full
    # -----------------------------------------------------------------
    for target in _SELECTIVITY_TARGETS:
        filter_expr, describe = _threshold_predicate_for_fraction(dataset, target)
        scenario_name = f"selective_{int(target * 100)}pct"
        if filter_expr is None:
            logger.info(
                "%s skipped on %s: %s",
                scenario_name, dataset.name, describe,
            )
            continue
        logger.info("%s: %s — predicate=%s", scenario_name, dataset.name, describe)

        for i in range(n_runs):
            gc.collect()

            # pull_filtered — downloads only matching shards.
            with tempfile.TemporaryDirectory(prefix="scx_f1_pf_") as tmp:
                local = os.path.join(tmp, "filtered.scx")
                logger.info(
                    "  pull_filtered run %d/%d (target=%.2f)",
                    i + 1, n_runs, target,
                )
                try:
                    stats, wall, rss = _time_call(
                        pyscx.pull, cloud_url, local, filter_expr,
                    )
                except Exception as exc:  # noqa: BLE001 — report & skip
                    logger.warning(
                        "pull_filtered failed for predicate %r: %s",
                        filter_expr, exc,
                    )
                    continue
                counters = CloudIOCounters()
                counters.record_pull_stats(stats)
                result.add_run(
                    wall_s=wall, peak_rss_mb=rss,
                    scenario=scenario_name,
                    method="pull_filtered",
                    predicate=describe,
                    target_fraction=target,
                    matching_cells=int(stats.get("matching_cells", 0)),
                    downloaded_shards=int(stats.get("downloaded_shards", 0)),
                    skipped_shards=int(stats.get("skipped_shards", 0)),
                    bytes_downloaded=counters.bytes_downloaded,
                    bytes_saved=int(stats.get("bytes_saved", 0)),
                    get_count_proxy=counters.sections_downloaded,
                )
                logger.info(
                    "    pull_filtered wall=%.3fs bytes=%d matching_cells=%d",
                    wall, counters.bytes_downloaded,
                    int(stats.get("matching_cells", 0)),
                )

            # open_cloud_query — Phase 7 selective query path. No local
            # materialisation; matching cells land in memory as a
            # QueryResult / AnnData. Bytes downloaded telemetry is not
            # surfaced on this path yet (deferred with the rest of
            # CloudQueryOptions); we record the cell / shard accounting
            # the QueryResult exposes (matching_cells, skipped_shards).
            logger.info(
                "  open_cloud_query run %d/%d (target=%.2f)",
                i + 1, n_runs, target,
            )
            try:
                def _run_query():
                    exp = pyscx.open_cloud(cloud_url)
                    return exp.query().filter_obs(filter_expr).collect()

                qresult, wall, rss = _time_call(_run_query)
            except Exception as exc:  # noqa: BLE001 — report & skip
                logger.warning(
                    "open_cloud_query failed for predicate %r: %s",
                    filter_expr, exc,
                )
                continue
            matching_cells = int(qresult.n_obs)
            skipped_shards = int(qresult.skipped_shards)
            total_shards = int(qresult.total_shards)
            result.add_run(
                wall_s=wall, peak_rss_mb=rss,
                scenario=scenario_name,
                method="open_cloud_query",
                predicate=describe,
                target_fraction=target,
                matching_cells=matching_cells,
                downloaded_shards=total_shards - skipped_shards,
                skipped_shards=skipped_shards,
                bytes_downloaded=0,
                bytes_saved=0,
                get_count_proxy=0,
                telemetry="phase_f_deferred",
            )
            logger.info(
                "    open_cloud_query wall=%.3fs matching_cells=%d "
                "skipped_shards=%d/%d",
                wall, matching_cells, skipped_shards, total_shards,
            )
            # Drop the in-memory result before the next iteration so
            # peak RSS comparisons stay clean.
            del qresult
            gc.collect()

    # -----------------------------------------------------------------
    # Per-scenario / per-method medians so reports can pivot without
    # re-aggregating all per-run records.
    # -----------------------------------------------------------------
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for run_rec in result.runs:
        scen = run_rec.extra.get("scenario", "unknown")
        method = run_rec.extra.get("method", "unknown")
        bucket = summary.setdefault(scen, {}).setdefault(method, {
            "_walls": [], "_bytes": [],
        })
        bucket["_walls"].append(run_rec.wall_s)
        bucket["_bytes"].append(run_rec.extra.get("bytes_downloaded", 0))
    out: dict[str, dict[str, dict[str, float]]] = {}
    for scen, methods in summary.items():
        out[scen] = {}
        for method, buckets in methods.items():
            walls = buckets["_walls"]
            byts = buckets["_bytes"]
            if not walls:
                continue
            out[scen][method] = {
                "median_wall_s": round(statistics.median(walls), 6),
                "median_bytes_downloaded": int(statistics.median(byts)) if byts else 0,
                "n_runs": len(walls),
            }
    result.metadata["per_scenario_summary"] = out

    logger.info(
        "cloud_reader_vs_pull complete: %s — %d runs across %d scenarios",
        dataset.name, len(result.runs), len(summary),
    )
    return result
