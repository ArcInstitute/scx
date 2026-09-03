"""
Selective Read (Query / Subsetting) benchmark.

Measures sub-matrix extraction performance across five scenarios:
  - row_slice: read a random subset of cells (all genes)
  - row_mask_gather: read every eighth cell (an interleaved mask that touches
    every shard — the ``handle[mask]`` gather arc-reactor's guide calling does)
  - col_projection: read a random subset of genes (all cells)
  - combined: read a random subset of both cells and genes
  - filtered_query: format-specific filtered query (skipped if unsupported)

Each scenario is run n_runs times.  Results are returned in a single
BenchmarkResult with per-run ``extra`` dicts tagging the scenario name.

Each run also carries a **sparse per-scenario** ``wall_s__<scenario>`` key.
``median_wall_s`` on this benchmark is an average over three fixed scenarios
plus up to three predicates, so on census_1m the recorded IQR (11.54 s against
a 13.85 s median) is scenario *mixing*, not noise — a number that cannot carry
a threshold. The per-scenario medians already existed in
``metadata["scenario_summary"]``, but thresholds read ``runs[].extra`` and
never ``metadata``, so nothing could gate them.

Every run also carries ``peak_rss_mb__<scenario>``: the **true in-region peak**
sampled by ``PeakRssSampler`` around the read. The reserved ``peak_rss_mb`` on
the run record is the runner's ``max(before, after)`` of two instantaneous
readings, which cannot see a transient the read allocates and frees before it
returns — exactly the shape of the 2× copy the bounded row gather removed
(REC-1). Floor the sparse key, never the reserved one.

Scenario names, for anyone writing one of those thresholds: the four base
scenarios are ``row_slice`` / ``row_mask_gather`` / ``col_projection`` /
``combined``; the predicate arms are ``filtered_query__<predicate>``, i.e.
``filtered_query__cell_type_eq_t_cell``, ``filtered_query__n_counts_gt_1000``,
``filtered_query__random_1pct``. ``n_counts_gt_1000`` does not run on the
census fixtures — ``obs['n_counts']`` is injected only for pbmc3k and
tabula_sapiens_100k by ``scripts/augment_obs_n_counts.py`` — and a threshold on
it there would be a permanent missing-metric violation.
"""

from __future__ import annotations

import gc
import logging
import statistics
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    QUERY_N_CELLS,
    QUERY_N_HVGS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.queries import default_predicates
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler
from benchmarks.comprehensive.runners import make_runner
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)


def _obs_columns(dataset: DatasetConfig) -> set[str]:
    """Return the obs column set for a dataset without loading X.

    A missing source h5ad is the one legitimate soft-failure path (benchmarks
    may run in environments without every dataset materialized) and returns
    an empty set with a ``WARNING``. Every other error propagates — a corrupt
    h5ad is a harness bug, not a predicate-skip signal.
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
    """Filter ``default_predicates`` to those whose columns exist on *dataset*.

    ``RandomSamplePredicate`` has no column requirement and always applies.
    """
    from benchmarks.comprehensive.queries import (
        EqPredicate,
        GtPredicate,
        RandomSamplePredicate,
    )

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


def _generate_indices(
    dataset: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-generate deterministic random cell and gene indices.

    Returns
    -------
    cell_idx : ndarray of shape (min(QUERY_N_CELLS, n_obs),)
    gene_idx : ndarray of shape (min(QUERY_N_HVGS, n_vars),)
    mask_idx : ndarray of shape (ceil(n_obs / 8),) — every eighth cell, the
        sorted index form of an interleaved boolean mask. It touches every
        shard of every format, so it measures the per-shard gather cost, not
        which shards a random draw happened to land in.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    cell_idx = rng.choice(
        dataset.n_obs,
        size=min(QUERY_N_CELLS, dataset.n_obs),
        replace=False,
    )
    gene_idx = rng.choice(
        dataset.n_vars,
        size=min(QUERY_N_HVGS, dataset.n_vars),
        replace=False,
    )
    # Sort for locality-friendly access patterns
    cell_idx.sort()
    gene_idx.sort()
    mask_idx = np.arange(0, dataset.n_obs, 8, dtype=np.int64)
    return cell_idx, gene_idx, mask_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Run the S3.4 selective-read benchmark.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to query.
    format_variant : FormatVariant
        Target format to convert and then query.
    n_runs : int
        Number of timed repetitions per scenario.
    cold_cache : bool
        If True, drop OS page caches before each run.

    Returns
    -------
    BenchmarkResult with benchmark="read_selective".
    """
    runner = make_runner(format_variant)
    h5ad_path = dataset.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Source h5ad not found: {h5ad_path}")

    cell_idx, gene_idx, mask_idx = _generate_indices(dataset)

    logger.info(
        "read_selective benchmark: dataset=%s format=%s n_runs=%d "
        "cold_cache=%s n_cells=%d n_genes=%d",
        dataset.name,
        format_variant.key,
        n_runs,
        cold_cache,
        len(cell_idx),
        len(gene_idx),
    )

    result = BenchmarkResult(
        benchmark="read_selective",
        format=format_variant.key,
        dataset=dataset.name,
        scenario={
            "name": "selective_read",
            "mode": "query",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
            "storage_backend": "local",
        },
        metadata={
            "cold_cache": cold_cache,
            "query_n_cells": int(len(cell_idx)),
            "query_n_hvgs": int(len(gene_idx)),
            "query_n_mask_cells": int(len(mask_idx)),
            "random_seed": RANDOM_SEED,
        },
    )

    scenarios: list[tuple[str, np.ndarray | None, np.ndarray | None]] = [
        ("row_slice", cell_idx, None),
        ("row_mask_gather", mask_idx, None),
        ("col_projection", None, gene_idx),
        ("combined", cell_idx, gene_idx),
    ]

    # Use pre-converted file if available, otherwise convert to temp dir.
    _cleanup = None
    if converted_path is not None and Path(converted_path).exists():
        output_path = Path(converted_path)
        result.file_size_bytes = runner.file_size(output_path)
    else:
        _cleanup = tempfile.TemporaryDirectory(prefix="scx_selread_bench_")
        output_path = Path(_cleanup.name) / f"converted.{format_variant.key}"
        logger.info("  converting %s -> %s ...", h5ad_path, output_path)
        convert_result = runner.convert_from_h5ad(h5ad_path, output_path)
        result.file_size_bytes = convert_result.output_size_bytes

    try:
        # -- Warm-up run(s) --
        for i in range(N_WARMUP_RUNS):
            logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
            runner.read_subset(output_path, cell_indices=cell_idx, gene_indices=gene_idx)
            gc.collect()

        scenario_times: dict[str, list[float]] = {}

        for scenario_name, c_idx, g_idx in scenarios:
            scenario_times[scenario_name] = []
            for i in range(n_runs):
                if cold_cache:
                    FormatRunner._drop_caches()

                logger.info("  %s run %d/%d", scenario_name, i + 1, n_runs)
                # True in-region peak: the runner's own `peak_rss_mb` is
                # max(before, after) of two instantaneous readings and misses
                # a transient the read frees before returning.
                with PeakRssSampler() as sampler:
                    tr = runner.read_subset(
                        output_path, cell_indices=c_idx, gene_indices=g_idx,
                    )

                scenario_times[scenario_name].append(tr.wall_s)
                result.add_run(
                    wall_s=tr.wall_s, user_s=tr.user_s, sys_s=tr.sys_s,
                    peak_rss_mb=tr.peak_rss_mb, scenario=scenario_name,
                    # Sparse per-scenario keys: only runs of THIS scenario carry
                    # them, so a threshold on one medians one scenario rather
                    # than the mix. See the module docstring.
                    **{
                        f"wall_s__{scenario_name}": round(tr.wall_s, 6),
                        f"peak_rss_mb__{scenario_name}": round(sampler.peak_mb, 3),
                    },
                )

        # -- Filtered-query scenarios (capability-gated) --
        declares_fq = "filtered_query" in runner.capabilities
        if declares_fq:
            predicates = _applicable_predicates(dataset)
            for predicate in predicates:
                scen_key = f"filtered_query__{predicate.name}"
                scenario_times[scen_key] = []
                for i in range(n_runs):
                    if cold_cache:
                        FormatRunner._drop_caches()
                    logger.info(
                        "  %s run %d/%d (%s)",
                        scen_key, i + 1, n_runs, predicate.describe(),
                    )
                    # Capability is declared, so a failure here is a contract
                    # violation — let it propagate.
                    with PeakRssSampler() as sampler:
                        tr = runner.read_filtered_query(output_path, predicate)
                    scenario_times[scen_key].append(tr.wall_s)
                    extra = tr.extra or {}
                    result.add_run(
                        wall_s=tr.wall_s, user_s=tr.user_s, sys_s=tr.sys_s,
                        peak_rss_mb=tr.peak_rss_mb,
                        scenario=scen_key,
                        predicate=predicate.name,
                        native_mechanism=extra.get("native_mechanism", "unknown"),
                        **{
                            f"wall_s__{scen_key}": round(tr.wall_s, 6),
                            f"peak_rss_mb__{scen_key}": round(sampler.peak_mb, 3),
                        },
                    )
        else:
            logger.info(
                "  filtered_query not advertised by %s (capabilities=%s), skipping",
                format_variant.key, sorted(runner.capabilities),
            )
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    # Add per-scenario summary statistics to metadata.
    scenario_summary: dict[str, dict[str, float]] = {}
    for name, times in scenario_times.items():
        if times:
            scenario_summary[name] = {
                "median_s": round(statistics.median(times), 6),
                "min_s": round(min(times), 6),
                "max_s": round(max(times), 6),
                "n_runs": len(times),
            }
    # Kept as the human-readable view. It is NOT the gateable one: thresholds
    # read `runs[].extra` only, never `metadata` — which is why the
    # `wall_s__<scenario>` keys above exist. Both are derived from the same
    # `scenario_times`, so they cannot disagree.
    result.metadata["scenario_summary"] = scenario_summary

    return result
