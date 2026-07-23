"""
Cell-set (S=64) gather-throughput benchmark (data-load Phase 0).

The STATE/STATE3 hot path is *not* i.i.d. sequential batching — it is random
gather of small covariate-grouped **cell sets** (``2026-07-22_DATA-LOAD-OPT.md``
§0/§2). This benchmark measures exactly that, distinct from the
``ml_loader``/``ooc_loader`` batch rate, by driving
``pyscx.SparseCellSetDataset.iter_with_plans`` with fixed-size sets of ``S=64``
cells.

SCX-only (``format_variant.key`` in the SCX codec set) — returns ``None`` for
every other format (gating pattern from ``index_plan`` / ``fragment_ops``).

Scenarios
---------
``gather_random``
    Each set is ``S`` uniformly-random rows across ``[0, n_obs)`` — the
    pessimistic scatter that stresses the shared shard cache.
``gather_grouped``
    Each set is ``S`` rows drawn from a single covariate group (a categorical
    ``obs`` column read once via ``read_obs``; falls back to an
    ``index // group_size`` bucketing when no suitable column exists). Models
    the ``(cell_type, perturbation)``-grouped access the models actually issue,
    and rewards grouped-sharding locality.

Per-run ``extra`` keys (sparse per-scenario):
    ``cellsets_per_sec__<sc>``  — sets/s (the STATE3-relevant throughput)
    ``cells_per_sec__gather__<sc>`` — cells/s
    ``ttfb_first_set_s__<sc>``  — time to first batch (loader spin-up + first gather)
    ``peak_rss_mb__<sc>``       — true in-epoch peak RSS
    ``cache_policy``            — cold_fadvise / warm
    ``shard_cache_hit_rate__<sc>`` — from ``SparseCellSetDataset.cache_metrics()``
        when the counter is present, else ``None``.
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from benchmarks.comprehensive.cache_control import drop_file_cache
from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto", "scx_fast"})
"""SCX-only — the cell-set gather path is codec-agnostic at the API level, so we
measure only the two default codecs (adaptive `auto`, decode-max `fast`)."""

# S=64 is the STATE3 perturbation set size (report §2.3).
_SET_SIZE = 64
# Cells per batch ≈ ML_BATCH_SIZE → 16 sets/batch.
_SETS_PER_BATCH = 16
# The current raw-local gather path is unoptimized (the report's Phase-1 target):
# a scattered 1024-cell batch costs ~O(seconds) even on small files. Throughput
# (sets/s) is a rate, so a modest batch count measures it faithfully without a
# runaway epoch — 500 batches × 5 runs × 2 scenarios timed out an 11k-cell fixture
# at 95 min. Keep the count small and uniform; scale down further for big files.
_DEFAULT_N_BATCHES = 50
_LARGE_N_OBS = 30_000
_MIN_N_BATCHES = 30
_WARMUP_BATCHES = 3
# Cap timed runs regardless of the harness default — the throughput median is
# stable across a few runs and each run is expensive on the pre-optimization path.
_MAX_N_RUNS = 3
# Locality bucket when no categorical obs column is usable (≈ one shard).
_LOCALITY_GROUP_SIZE = 4096
# Skip obs columns whose cardinality exceeds this — a near-unique column (e.g.
# a per-cell soma_joinid) yields singleton "groups" (grouped == random) AND made
# the old per-value `np.where` grouping O(n_distinct × n_obs), which blew up /
# FAST_FAILed on census_5m. Prefer a real covariate (cell_type-scale).
_MAX_GROUPS = 5000


def _have_pyscx() -> bool:
    try:
        import pyscx  # noqa: F401

        return True
    except ImportError:
        return False


def _n_batches_for(n_obs: int) -> int:
    if n_obs < _LARGE_N_OBS:
        return _DEFAULT_N_BATCHES
    scaled = 400_000 // max(1, n_obs)
    return int(min(_DEFAULT_N_BATCHES, max(_MIN_N_BATCHES, scaled)))


# ---------------------------------------------------------------------------
# Covariate grouping
# ---------------------------------------------------------------------------


def _resolve_groups(scx_path: str, n_obs: int) -> list[np.ndarray]:
    """Return a list of row-index arrays, one per covariate group.

    Reads a single categorical/object ``obs`` column via ``read_obs`` (matrix-
    free; no X touched). Picks the first column with ``1 < n_distinct <=
    _MAX_GROUPS`` (a real covariate; high-cardinality columns are skipped).
    Groups are built with an O(n log n) argsort split — NOT a per-value
    ``np.where`` scan, which is O(n_distinct × n_obs) and blows up at census
    scale. Falls back to ``index // _LOCALITY_GROUP_SIZE`` buckets when no
    column qualifies (barcode-index-only fixtures / atlases)."""
    try:
        import pyscx

        exp = pyscx.open(scx_path)
        keys = list(exp.obs_keys())
        for col in keys:
            try:
                df = exp.read_obs([col])
            except Exception:  # noqa: BLE001
                continue
            if col not in df.columns:
                continue
            codes = df[col].astype("category").cat.codes.to_numpy()
            if codes.size == 0:
                continue
            n_distinct = int(codes.max()) + 1
            # Require a modest, real covariate cardinality.
            if not (1 < n_distinct <= _MAX_GROUPS):
                continue
            # O(n log n) group split: sort row indices by code, then cut at
            # the unique-value boundaries. No per-value scan.
            order = np.argsort(codes, kind="stable").astype(np.uint64)
            sorted_codes = codes[order.astype(np.int64)]
            _, starts = np.unique(sorted_codes, return_index=True)
            groups = [g for g in np.split(order, starts[1:]) if g.size > 0]
            if len(groups) > 1:
                logger.info("cellset grouping on obs[%r]: %d groups", col, len(groups))
                return groups
    except Exception as e:  # noqa: BLE001
        logger.info("obs grouping unavailable (%s); using index buckets", e)
    # Fallback: contiguous index buckets.
    n_groups = max(2, (n_obs + _LOCALITY_GROUP_SIZE - 1) // _LOCALITY_GROUP_SIZE)
    return [
        np.arange(g * _LOCALITY_GROUP_SIZE, min((g + 1) * _LOCALITY_GROUP_SIZE, n_obs), dtype=np.uint64)
        for g in range(n_groups)
    ]


# ---------------------------------------------------------------------------
# Plan generators — yield SparseCellSetDataset 4-tuples
#   (file_ids u32[], rows u64[], role_tags i32[], set_offsets i64[])
# ---------------------------------------------------------------------------


def _pack_plan(sets: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = np.concatenate(sets).astype(np.uint64)
    file_ids = np.zeros(rows.size, dtype=np.uint32)
    role_tags = np.zeros(rows.size, dtype=np.int32)
    offsets = np.zeros(len(sets) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([s.size for s in sets], dtype=np.int64)
    return file_ids, rows, role_tags, offsets


def _random_plans(n_obs: int, n_batches: int, seed: int = 0) -> Iterator[tuple]:
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        sets = [
            rng.integers(0, n_obs, size=_SET_SIZE).astype(np.uint64)
            for _ in range(_SETS_PER_BATCH)
        ]
        yield _pack_plan(sets)


def _grouped_plans(
    groups: list[np.ndarray], n_batches: int, seed: int = 0
) -> Iterator[tuple]:
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    for _ in range(n_batches):
        sets = []
        for _ in range(_SETS_PER_BATCH):
            g = groups[int(rng.integers(0, n_groups))]
            # Sample S with replacement if the group is smaller than S.
            replace = g.size < _SET_SIZE
            sel = rng.choice(g, size=_SET_SIZE, replace=replace)
            sets.append(sel.astype(np.uint64))
        yield _pack_plan(sets)


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


@dataclass
class _Outcome:
    n_sets: int
    n_cells: int
    wall_s: float
    ttfb_s: float
    peak_rss_mb: float
    shard_cache_hit_rate: float | None


def _run_gather(
    scx_path: str, plans_factory: Callable[[], Iterator[tuple]]
) -> _Outcome:
    import pyscx

    gc.collect()
    ds = pyscx.SparseCellSetDataset([scx_path])
    n_sets = 0
    n_cells = 0
    ttfb_s = 0.0
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        first = True
        for batch in ds.iter_with_plans(plans_factory()):
            if first:
                ttfb_s = time.perf_counter() - t0
                first = False
            set_offsets = batch["set_offsets"]
            n_sets += len(set_offsets) - 1
            n_cells += int(batch["shape"][0])
        wall_s = time.perf_counter() - t0
    hit_rate: float | None = None
    try:
        cm = ds.cache_metrics()
        hits = float(cm.get("hits", 0))
        misses = float(cm.get("misses", 0))
        if hits + misses > 0:
            hit_rate = round(hits / (hits + misses), 4)
    except Exception:  # noqa: BLE001
        pass
    return _Outcome(
        n_sets=n_sets,
        n_cells=n_cells,
        wall_s=wall_s,
        ttfb_s=ttfb_s,
        peak_rss_mb=sampler.peak_mb,
        shard_cache_hit_rate=hit_rate,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = True,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """SCX-only S=64 cell-set gather throughput. Drops the page cache before
    each timed run. Returns ``None`` for non-SCX formats / missing fixture."""
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    if not _have_pyscx():
        logger.warning("Skipping cellset_gather: pyscx not importable")
        return None
    if converted_path is not None and Path(converted_path).exists():
        scx_path = str(converted_path)
    else:
        try:
            p = dataset.path_for_format(format_variant.key)
        except (ValueError, FileNotFoundError):
            p = None
        if p is None or not p.exists():
            raise FileNotFoundError(
                f"Missing converted SCX file for {dataset.name}. "
                f"Run Phase A conversion first (--formats {format_variant.key})."
            )
        scx_path = str(p)

    n_obs = dataset.n_obs
    n_batches = _n_batches_for(n_obs)
    n_runs = max(1, min(n_runs, _MAX_N_RUNS))
    groups = _resolve_groups(scx_path, n_obs)

    result = BenchmarkResult(
        benchmark="cellset_gather",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "set_size": _SET_SIZE,
            "sets_per_batch": _SETS_PER_BATCH,
            "n_batches": n_batches,
            "n_groups": len(groups),
            "cold_cache": cold_cache,
        },
    )
    result.file_size_bytes = Path(scx_path).stat().st_size

    scenarios: list[tuple[str, Callable[[int], Iterator[tuple]]]] = [
        ("gather_random", lambda nb: _random_plans(n_obs, nb)),
        ("gather_grouped", lambda nb: _grouped_plans(groups, nb)),
    ]

    warmup = min(n_batches, _WARMUP_BATCHES)
    for scenario_name, plans in scenarios:
        # Warm code (tokio/rayon) with a tiny pass; timed reads are cold.
        try:
            _run_gather(scx_path, lambda: plans(warmup))
        except Exception as e:  # noqa: BLE001
            logger.error("  warmup failed for %s: %s", scenario_name, e)
            continue

        run_sps: list[float] = []
        run_cps: list[float] = []
        run_rss: list[float] = []
        for i in range(n_runs):
            cache_policy = drop_file_cache(scx_path) if cold_cache else "warm"
            try:
                out = _run_gather(scx_path, lambda: plans(n_batches))
            except Exception as e:  # noqa: BLE001
                logger.error("  run %d/%d failed for %s: %s", i + 1, n_runs, scenario_name, e)
                continue
            sps = out.n_sets / out.wall_s if out.wall_s > 0 else 0.0
            cps = out.n_cells / out.wall_s if out.wall_s > 0 else 0.0
            result.add_run(
                wall_s=out.wall_s,
                peak_rss_mb=out.peak_rss_mb,
                scenario=scenario_name,
                n_sets=out.n_sets,
                n_cells=out.n_cells,
                cache_policy=cache_policy,
                **{
                    f"cellsets_per_sec__{scenario_name}": round(sps, 1),
                    f"cells_per_sec__gather__{scenario_name}": round(cps, 0),
                    f"ttfb_first_set_s__{scenario_name}": round(out.ttfb_s, 4),
                    f"peak_rss_mb__{scenario_name}": round(out.peak_rss_mb, 1),
                    f"shard_cache_hit_rate__{scenario_name}": out.shard_cache_hit_rate,
                },
            )
            run_sps.append(sps)
            run_cps.append(cps)
            run_rss.append(out.peak_rss_mb)
            logger.info(
                "    %s: wall=%.3fs sets/s=%.1f cells/s=%.0f rss=%.1fMB cache=%s",
                scenario_name,
                out.wall_s,
                sps,
                cps,
                out.peak_rss_mb,
                cache_policy,
            )

        if run_sps:
            result.metadata.setdefault("scenario_summary", {})[scenario_name] = {
                "n_runs": len(run_sps),
                "median_cellsets_per_sec": round(statistics.median(run_sps), 1),
                "median_cells_per_sec": round(statistics.median(run_cps), 0),
                "median_peak_rss_mb": round(statistics.median(run_rss), 1),
            }
        gc.collect()

    return result
