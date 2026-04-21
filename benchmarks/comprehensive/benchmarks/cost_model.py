"""
Cost model benchmark (Phase F.2).

SCX-only cost-per-query model. For a fixed workload (metadata, selective
5% / 20%, full read) and the set of SCX layouts available on GCS, computes
cents per 1 million cells queried using the pinned ``GCS_PRICING`` table.

Today SCX has one committed cloud layout — the exploded ``.scxd/`` produced
by ``pyscx.push`` + consumed by ``pyscx.pull`` / ``pyscx.open_cloud``. The
benchmark surface is forward-compatible with alternate layouts (packed
``.scx`` behind the front catalog, cloud-optimized) — a layout is added by
wiring a new cloud-URL resolution + label into ``_layout_to_cloud_url``.

Each run records::

    extra = {
        "scenario":        metadata | selective_5pct | selective_20pct | full_read,
        "layout":          scxd_exploded | scx_packed | ...,
        "bytes_downloaded": int,
        "get_count_proxy": int (sections_downloaded or derived),
        "matching_cells":  int (0 for metadata/full),
        "egress_usd":      float,
        "request_usd":     float,
        "total_usd":       float,
        "usd_per_million_cells_queried": float,
    }

The ``usd_per_million_cells_queried`` field is the headline metric — it
normalizes cost across dataset sizes and selectivities so the reporting
layer can directly pivot ``(layout × scenario)``.
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
    require_gcp_credentials,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    GCS_PRICING,
    GCS_TEST_BUCKET,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

_SCX_TRIGGER_KEY = "scx_auto"

# Currently only the exploded layout is wired for cloud (``.scxd/`` on GCS).
# Future layouts (packed `.scx` with front catalog, region-replicated) plug
# into this list once ``ensure_cloud_fixture`` knows how to materialize them.
_LAYOUTS: list[tuple[str, str]] = [
    ("scxd_exploded", "exploded .scxd (cloud-ready)"),
]


def _n_counts_threshold_predicates(dataset: DatasetConfig) -> dict[int, str | None]:
    """Return ``{5: "n_counts > X", 20: "n_counts > Y"}`` or Nones.

    Reads the dataset's obs `n_counts` column backed-mode and picks
    thresholds via numpy quantile so each predicate yields approximately
    the target fraction of rows. When the column is absent (or the h5ad
    isn't readable), returns Nones — callers skip the scenario rather
    than synthesize a predicate the SCX parser can't evaluate.
    """
    import anndata
    import numpy as np

    result: dict[int, str | None] = {5: None, 20: None}
    try:
        adata = anndata.read_h5ad(dataset.h5ad_path, backed="r")
    except (FileNotFoundError, OSError):
        return result
    try:
        if "n_counts" in adata.obs.columns:
            col = np.asarray(adata.obs["n_counts"])
            for pct in (5, 20):
                fraction = pct / 100.0
                cutoff = float(np.quantile(col, 1.0 - fraction))
                result[pct] = f"n_counts > {cutoff}"
    finally:
        adata.file.close()
    return result


def _cost_per_million_cells(bytes_downloaded: int, gets: int, cells: int) -> dict:
    """Compute egress + request cost for one run, normalized per 1M cells.

    Uses the pinned ``GCS_PRICING`` table. Egress is priced at the
    same-region rate (0 USD today for intra-region GCE ↔ GCS) by default;
    the cross-region rate is emitted alongside as a "what if" sanity check.
    """
    gb = bytes_downloaded / (1024 ** 3)
    egress_same = gb * GCS_PRICING["egress_same_region_usd_per_gb"]
    egress_cross = gb * GCS_PRICING["egress_cross_region_usd_per_gb"]
    request = (gets / 10_000) * GCS_PRICING["class_b_per_10k_usd"]
    total_same = egress_same + request
    per_m = (total_same / max(cells, 1)) * 1_000_000 if cells > 0 else 0.0
    return {
        "egress_usd": round(egress_same, 8),
        "egress_cross_region_usd": round(egress_cross, 8),
        "request_usd": round(request, 8),
        "total_usd": round(total_same, 8),
        "usd_per_million_cells_queried": round(per_m, 8),
    }


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

    require_gcp_credentials()
    runner = make_runner(format_variant)

    result = BenchmarkResult(
        benchmark="cost_model",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "provider": provider,
            "bucket": GCS_TEST_BUCKET,
            "pricing": dict(GCS_PRICING),
            "layouts": [lid for lid, _ in _LAYOUTS],
            "scenarios": [
                "metadata", "selective_5pct", "selective_20pct", "full_read",
            ],
            "n_runs": n_runs,
            "n_warmup": N_WARMUP_RUNS,
        },
    )
    result.file_size_bytes = runner.file_size(Path(converted_path))

    # Only the exploded layout is wired today — emit one run per scenario.
    for layout_id, layout_label in _LAYOUTS:
        cloud_url = ensure_cloud_fixture(
            dataset, format_variant, Path(converted_path), provider=provider,
        )
        logger.info("layout=%s (%s) cloud_url=%s", layout_id, layout_label, cloud_url)

        for i in range(n_runs):
            gc.collect()

            # ---- metadata ----
            logger.info("  metadata run %d/%d", i + 1, n_runs)
            import time
            t0 = time.perf_counter()
            handle = pyscx.open_cloud(cloud_url)
            _ = handle.n_obs; _ = handle.n_vars; _ = handle.nnz
            wall = time.perf_counter() - t0
            # open_cloud has no bytes telemetry today — upper-bound as the
            # catalog+header size from a companion pull would be cheating
            # because the measurement doesn't actually issue those GETs.
            cost = _cost_per_million_cells(0, 1, 0)
            result.add_run(
                wall_s=wall,
                scenario="metadata",
                layout=layout_id,
                layout_label=layout_label,
                matching_cells=0,
                bytes_downloaded=0,
                get_count_proxy=1,
                telemetry="phase_f_deferred",
                **cost,
            )

            # ---- selective 5% / 20% ----
            # Use an obs-column threshold predicate (`n_counts > X`) chosen
            # via quantile to hit the target fraction. SCX's filter_obs
            # parser doesn't support arithmetic expressions like
            # `row_id % N == 0`, so the stride-hash fallback used in other
            # Python-SQL formats won't work here — if the dataset has no
            # `n_counts` column, skip the selective scenario rather than
            # emit a failed attempt.
            selective_predicate = _n_counts_threshold_predicates(dataset)
            for pct in (5, 20):
                filter_expr = selective_predicate.get(pct)
                if filter_expr is None:
                    logger.info(
                        "  selective_%dpct skipped: no n_counts obs column "
                        "on %s (SCX filter_obs can't evaluate row_id modulo)",
                        pct, dataset.name,
                    )
                    continue
                with tempfile.TemporaryDirectory(prefix=f"scx_cm_s{pct}_") as tmp:
                    local = os.path.join(tmp, f"filtered_{pct}.scx")
                    logger.info(
                        "  selective_%dpct run %d/%d predicate=%s",
                        pct, i + 1, n_runs, filter_expr,
                    )
                    t0 = time.perf_counter()
                    try:
                        stats = pyscx.pull(cloud_url, local, filter_expr)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "pull_filtered failed (predicate=%r): %s",
                            filter_expr, exc,
                        )
                        continue
                    wall = time.perf_counter() - t0
                    counters = CloudIOCounters()
                    counters.record_pull_stats(stats)
                    matching = int(stats.get("matching_cells", 0))
                    cost = _cost_per_million_cells(
                        counters.bytes_downloaded,
                        counters.sections_downloaded,
                        matching,
                    )
                    result.add_run(
                        wall_s=wall,
                        scenario=f"selective_{pct}pct",
                        layout=layout_id,
                        layout_label=layout_label,
                        predicate=filter_expr,
                        matching_cells=matching,
                        downloaded_shards=int(stats.get("downloaded_shards", 0)),
                        skipped_shards=int(stats.get("skipped_shards", 0)),
                        bytes_downloaded=counters.bytes_downloaded,
                        bytes_saved=int(stats.get("bytes_saved", 0)),
                        get_count_proxy=counters.sections_downloaded,
                        **cost,
                    )

            # ---- full_read ----
            with tempfile.TemporaryDirectory(prefix="scx_cm_full_") as tmp:
                local = os.path.join(tmp, "pulled.scx")
                logger.info("  full_read run %d/%d", i + 1, n_runs)
                t0 = time.perf_counter()
                stats = pyscx.pull(cloud_url, local)
                wall = time.perf_counter() - t0
                counters = CloudIOCounters()
                counters.record_pull_stats(stats)
                matching = dataset.n_obs  # full read queries every cell
                cost = _cost_per_million_cells(
                    counters.bytes_downloaded,
                    counters.sections_downloaded,
                    matching,
                )
                result.add_run(
                    wall_s=wall,
                    scenario="full_read",
                    layout=layout_id,
                    layout_label=layout_label,
                    matching_cells=matching,
                    bytes_downloaded=counters.bytes_downloaded,
                    get_count_proxy=counters.sections_downloaded,
                    **cost,
                )

    # Per-(layout, scenario) median of the headline USD/1M metric.
    summary: dict[tuple[str, str], list[float]] = {}
    for rec in result.runs:
        key = (rec.extra.get("layout", ""), rec.extra.get("scenario", ""))
        summary.setdefault(key, []).append(
            rec.extra.get("usd_per_million_cells_queried", 0.0)
        )
    result.metadata["per_layout_scenario_median_usd_per_million"] = {
        f"{layout}::{scen}": round(statistics.median(vals), 8)
        for (layout, scen), vals in summary.items() if vals
    }

    logger.info(
        "cost_model complete: %s — %d runs across %d layouts × %d scenarios",
        dataset.name, len(result.runs), len(_LAYOUTS),
        len({s for (_, s) in summary}),
    )
    return result
