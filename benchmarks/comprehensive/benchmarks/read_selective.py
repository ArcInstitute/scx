"""
Selective Read (Query / Subsetting) benchmark -- COMPREHENSIVE-BENCHMARKING.md S3.4.

Measures sub-matrix extraction performance across four scenarios:
  - row_slice: read a random subset of cells (all genes)
  - col_projection: read a random subset of genes (all cells)
  - combined: read a random subset of both cells and genes
  - filtered_query: format-specific filtered query (skipped if unsupported)

Each scenario is run n_runs times.  Results are returned in a single
BenchmarkResult with per-run ``extra`` dicts tagging the scenario name.
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
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)


def _generate_indices(
    dataset: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-generate deterministic random cell and gene indices.

    Returns
    -------
    cell_idx : ndarray of shape (min(QUERY_N_CELLS, n_obs),)
    gene_idx : ndarray of shape (min(QUERY_N_HVGS, n_vars),)
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
    return cell_idx, gene_idx


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

    cell_idx, gene_idx = _generate_indices(dataset)

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
        metadata={
            "cold_cache": cold_cache,
            "query_n_cells": int(len(cell_idx)),
            "query_n_hvgs": int(len(gene_idx)),
            "random_seed": RANDOM_SEED,
        },
    )

    scenarios: list[tuple[str, np.ndarray | None, np.ndarray | None]] = [
        ("row_slice", cell_idx, None),
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
                tr = runner.read_subset(
                    output_path, cell_indices=c_idx, gene_indices=g_idx,
                )

                scenario_times[scenario_name].append(tr.wall_s)
                result.add_run(
                    wall_s=tr.wall_s, user_s=tr.user_s, sys_s=tr.sys_s,
                    peak_rss_mb=tr.peak_rss_mb, scenario=scenario_name,
                )

        try:
            read_filtered = getattr(runner, "read_filtered_query", None)
            if read_filtered is not None:
                scenario_times["filtered_query"] = []
                for i in range(n_runs):
                    if cold_cache:
                        FormatRunner._drop_caches()
                    logger.info("  filtered_query run %d/%d", i + 1, n_runs)
                    tr = read_filtered(
                        output_path, cell_indices=cell_idx, gene_indices=gene_idx,
                    )
                    scenario_times["filtered_query"].append(tr.wall_s)
                    result.add_run(
                        wall_s=tr.wall_s, user_s=tr.user_s, sys_s=tr.sys_s,
                        peak_rss_mb=tr.peak_rss_mb, scenario="filtered_query",
                    )
        except (NotImplementedError, TypeError):
            logger.info("  filtered_query not supported for %s, skipping", format_variant.key)
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
    result.metadata["scenario_summary"] = scenario_summary

    return result
