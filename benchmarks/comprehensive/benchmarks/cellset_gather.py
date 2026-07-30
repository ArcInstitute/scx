"""
Cell-set gather-throughput benchmark (data-load Phase 0).

The STATE/STATE3 hot path is *not* i.i.d. sequential batching — it is random
gather of small covariate-grouped **cell sets**: 64–512 cells grouped by
``(cell_type, perturbation)``, frequently spanning files, with paired controls and
a per-set gene-panel subsample. This benchmark measures exactly that, distinct
from the ``ml_loader``/``ooc_loader`` batch rate, by driving
``pyscx.SparseCellSetDataset.iter_with_plans`` with fixed-size cell sets. Results
and the premise-gate verdict live in ``docs/performance.md`` § "Out-of-core loader
— cold-cache measurements and the P-1 premise gate".

SCX-only (``format_variant.key`` in the SCX codec set) — returns ``None`` for
every other format (gating pattern from ``index_plan`` / ``fragment_ops``).

Scenarios
---------
``gather_random`` / ``gather_grouped`` (**S=64**)
    The STATE3 perturbation set size. ``gather_random`` draws each set
    uniformly across ``[0, n_obs)`` — the pessimistic scatter that stresses the
    shared shard cache. ``gather_grouped`` draws each set from a single
    covariate group (a categorical ``obs`` column read once via ``read_obs``;
    falls back to ``index // group_size`` bucketing when no suitable column
    exists), modelling the ``(cell_type, perturbation)``-grouped access the
    models actually issue and rewarding grouped-sharding locality.

``gather_random_s512`` / ``gather_grouped_s512`` (**S=512**)
    The observational / pretraining / ICL regime (report §2.3: S=128–512), which
    STATE3's own measurement never covered — its 0.6%-of-step-time number is
    S=64, B=4, GPU-memory-ceilinged. This is the P-1(b) gate arm. Cells per
    batch are held ≈ ``ML_BATCH_SIZE`` across set sizes so the two regimes move
    the same number of cells per batch and only the set granularity differs.

``gather_random_r<N>`` (**N concurrent ranks**, opt-in via ``SCX_BENCH_N_RANKS``)
    The P-1(c) gate arm: the same per-rank workload run in ``N`` spawned
    processes against one file, reporting
    ``rank_scaling_efficiency = median(per-rank rate at N) / (rate at 1)``.
    ≈1 means the loader is not a shared bottleneck; ≈1/N means the ranks
    serialise on the page cache / filesystem. Two deliberate choices:

    * **``random``, not ``grouped``** — random scatter is the worst case for
      cache contention (the thing being measured), and its plan generator needs
      only ``n_obs``, so nothing large crosses the process boundary.
    * **a distinct seed per rank** — identical seeds would have every rank touch
      the same shards, so the shared page cache would *help* and efficiency
      would read ≈1 for the wrong reason.

    Its 1-rank reference goes through the same spawned-child path as the N-rank
    arm, so the two differ only in concurrency. Do **not** compare the rank
    arm's absolute rate against ``gather_random``: the reference is the
    ``_r<N>`` arm's own 1-rank run, recorded alongside it.

Per-run ``extra`` keys (sparse per-scenario):
    ``cellsets_per_sec__<sc>``  — sets/s (the STATE3-relevant throughput)
    ``cells_per_sec__gather__<sc>`` — cells/s
    ``ttfb_first_set_s__<sc>``  — time to first batch (loader spin-up + first gather)
    ``peak_rss_mb__<sc>``       — true in-epoch peak RSS
    ``cache_policy``            — cold_fadvise / warm
    ``shard_cache_hit_rate__<sc>`` — from ``SparseCellSetDataset.cache_metrics()``
        when the counter is present, else ``None``.
Rank arm additionally:
    ``cellsets_per_sec__<sc>`` — **aggregate** across ranks (what the node gets)
    ``cellsets_per_sec_per_rank__<sc>`` — median single-rank rate at N
    ``cellsets_per_sec_1rank__<sc>`` — the 1-rank reference
    ``rank_scaling_efficiency__<sc>`` — the P-1(c) answer
    ``total_peak_rss_mb__<sc>`` — summed across ranks
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
from benchmarks.comprehensive.config import DatasetConfig, FormatVariant, ML_BATCH_SIZE
from benchmarks.comprehensive.multirank import (
    rank_efficiency,
    resolve_n_ranks,
    run_ranks,
    summarize_ranks,
)
from benchmarks.comprehensive.results import BenchmarkResult, require_runs
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto", "scx_fast"})
"""SCX-only — the cell-set gather path is codec-agnostic at the API level, so we
measure only the two default codecs (adaptive `auto`, decode-max `fast`)."""

# S=64 is the STATE3 perturbation set size; S=512 the observational/ICL regime
# (report §2.3). Sets per batch are chosen to hold cells/batch ≈ ML_BATCH_SIZE
# (1024) so the two regimes differ in set granularity, not in batch volume.
_SET_SIZE_S64 = 64
_SET_SIZE_S512 = 512
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
# The rank arm pays for (1 + N) workloads per run, so cap it harder than the
# single-process scenarios while still giving a spread to sanity-check against.
_RANK_N_RUNS = 2
# Locality bucket when no categorical obs column is usable (≈ one shard).
_LOCALITY_GROUP_SIZE = 4096
# Skip obs columns whose cardinality exceeds this — a near-unique column (e.g.
# a per-cell soma_joinid) yields singleton "groups" (grouped == random) AND made
# the old per-value `np.where` grouping O(n_distinct × n_obs), which blew up /
# FAST_FAILed on census_5m. Prefer a real covariate (cell_type-scale).
_MAX_GROUPS = 5000


def _sets_per_batch(set_size: int) -> int:
    """Sets per batch holding cells/batch ≈ ``ML_BATCH_SIZE`` (min 1)."""
    return max(1, ML_BATCH_SIZE // set_size)


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
            # pandas encodes NaN/missing as code -1. Left in, those rows collapse
            # into a single spurious "group" of unrelated cells, which would be
            # measured as covariate locality that doesn't exist. Drop them and keep
            # the surviving rows' original positions.
            valid = codes >= 0
            if not valid.any():
                continue
            row_ids = np.flatnonzero(valid).astype(np.uint64)
            codes = codes[valid]
            n_distinct = int(codes.max()) + 1
            # Require a modest, real covariate cardinality.
            if not (1 < n_distinct <= _MAX_GROUPS):
                continue
            # O(n log n) group split: sort row indices by code, then cut at
            # the unique-value boundaries. No per-value scan.
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            _, starts = np.unique(sorted_codes, return_index=True)
            groups = [g for g in np.split(row_ids[order], starts[1:]) if g.size > 0]
            if len(groups) > 1:
                logger.info("cellset grouping on obs[%r]: %d groups", col, len(groups))
                return groups
    except Exception as e:  # noqa: BLE001
        logger.info("obs grouping unavailable (%s); using index buckets", e)
    # Fallback: contiguous index buckets. `max(2, ...)` forces at least two
    # buckets, which for `n_obs < _LOCALITY_GROUP_SIZE` makes the second one
    # empty — and `rng.choice` on an empty group raises "a cannot be empty
    # unless no samples are taken", killing the whole grouped scenario. Filter
    # empties, then split a single bucket in half so "grouped" still means more
    # than one group on small fixtures.
    if n_obs <= 0:
        # Nothing to group. Returning [] would make `_grouped_plans` call
        # `rng.integers(0, 0)`, which raises — an empty dataset should skip the
        # scenario, not crash it.
        return []
    n_groups = max(2, (n_obs + _LOCALITY_GROUP_SIZE - 1) // _LOCALITY_GROUP_SIZE)
    buckets = [
        np.arange(
            g * _LOCALITY_GROUP_SIZE,
            min((g + 1) * _LOCALITY_GROUP_SIZE, n_obs),
            dtype=np.uint64,
        )
        for g in range(n_groups)
    ]
    buckets = [b for b in buckets if b.size > 0]
    if len(buckets) == 1 and buckets[0].size >= 2:
        half = buckets[0].size // 2
        buckets = [buckets[0][:half], buckets[0][half:]]
    return buckets


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


def _random_plans(
    n_obs: int,
    n_batches: int,
    set_size: int,
    sets_per_batch: int,
    seed: int = 0,
) -> Iterator[tuple]:
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        sets = [
            rng.integers(0, n_obs, size=set_size).astype(np.uint64)
            for _ in range(sets_per_batch)
        ]
        yield _pack_plan(sets)


def _grouped_plans(
    groups: list[np.ndarray],
    n_batches: int,
    set_size: int,
    sets_per_batch: int,
    seed: int = 0,
) -> Iterator[tuple]:
    rng = np.random.default_rng(seed)
    n_groups = len(groups)
    for _ in range(n_batches):
        sets = []
        for _ in range(sets_per_batch):
            g = groups[int(rng.integers(0, n_groups))]
            # Sample S with replacement if the group is smaller than S. At
            # S=512 most real covariate groups are smaller than the set, so
            # replacement is the norm rather than the exception here — which is
            # exactly what STATE3's own under-full-sentence padding does.
            replace = g.size < set_size
            sel = rng.choice(g, size=set_size, replace=replace)
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


def _cache_hit_rate(ds: Any) -> float | None:
    try:
        cm = ds.cache_metrics()
        hits = float(cm.get("hits", 0))
        misses = float(cm.get("misses", 0))
        if hits + misses > 0:
            return round(hits / (hits + misses), 4)
    except Exception:  # noqa: BLE001
        pass
    return None


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
    return _Outcome(
        n_sets=n_sets,
        n_cells=n_cells,
        wall_s=wall_s,
        ttfb_s=ttfb_s,
        peak_rss_mb=sampler.peak_mb,
        shard_cache_hit_rate=_cache_hit_rate(ds),
    )


# ---------------------------------------------------------------------------
# Rank-arm worker — MUST stay module-level so `spawn` can pickle it by reference
# ---------------------------------------------------------------------------


def _rank_gather_worker(
    rank: int,
    n_ranks: int,
    scx_path: str,
    n_obs: int,
    set_size: int,
    sets_per_batch: int,
    n_batches: int,
) -> dict[str, Any]:
    """One rank's share of the concurrent gather. Runs in a spawned child.

    Constructs its **own** ``SparseCellSetDataset`` — a parent-constructed one
    used post-fork trips pyscx's PID guard, and under ``spawn`` there is nothing
    inherited to reuse anyway. Seeded on ``rank`` so ranks touch different rows
    (see the module docstring on why identical seeds would flatter the result).
    """
    import pyscx

    ds = pyscx.SparseCellSetDataset([scx_path])
    n_sets = 0
    n_cells = 0
    t0 = time.perf_counter()
    for batch in ds.iter_with_plans(
        _random_plans(n_obs, n_batches, set_size, sets_per_batch, seed=1000 + rank)
    ):
        n_sets += len(batch["set_offsets"]) - 1
        n_cells += int(batch["shape"][0])
    gather_wall_s = time.perf_counter() - t0
    return {
        "n_sets": n_sets,
        "n_cells": n_cells,
        "gather_wall_s": round(gather_wall_s, 4),
        "cellsets_per_sec": round(n_sets / gather_wall_s, 4) if gather_wall_s > 0 else 0.0,
        "cells_per_sec": round(n_cells / gather_wall_s, 1) if gather_wall_s > 0 else 0.0,
        "shard_cache_hit_rate": _cache_hit_rate(ds),
    }


def _run_rank_arm(
    scx_path: str, n_obs: int, set_size: int, sets_per_batch: int, n_batches: int, n_ranks: int
) -> dict[str, Any] | None:
    """1-rank reference + N-rank arm, both through the spawned-child path."""
    worker_args = (scx_path, n_obs, set_size, sets_per_batch, n_batches)
    single = run_ranks(1, _rank_gather_worker, worker_args)
    many = run_ranks(n_ranks, _rank_gather_worker, worker_args)
    one_s = summarize_ranks(single, "cellsets_per_sec")
    many_s = summarize_ranks(many, "cellsets_per_sec")
    if one_s is None or many_s is None:
        logger.error("rank arm produced no usable rate (1-rank=%s, N-rank=%s)", one_s, many_s)
        return None
    return {
        "n_ranks": n_ranks,
        "n_ranks_reported": many_s["n_ranks_reported"],
        "aggregate_cellsets_per_sec": many_s["aggregate"],
        "per_rank_median_cellsets_per_sec": many_s["per_rank_median"],
        "one_rank_cellsets_per_sec": one_s["per_rank_median"],
        "rank_scaling_efficiency": rank_efficiency(single, many, "cellsets_per_sec"),
        "max_wall_s": many_s["max_wall_s"],
        "total_peak_rss_mb": many_s["total_peak_rss_mb"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Scenario:
    name: str
    set_size: int
    plan_kind: str  # "random" | "grouped"

    @property
    def sets_per_batch(self) -> int:
        return _sets_per_batch(self.set_size)


# Ordering matters only for log readability. The two S=64 names are FROZEN — six
# `thresholds.yaml` absolute floors key off `cellsets_per_sec__gather_random` /
# `__gather_grouped`, and renaming either turns those into "missing metric"
# violations rather than a regression signal.
_SCENARIOS: tuple[_Scenario, ...] = (
    _Scenario("gather_random", _SET_SIZE_S64, "random"),
    _Scenario("gather_grouped", _SET_SIZE_S64, "grouped"),
    _Scenario("gather_random_s512", _SET_SIZE_S512, "random"),
    _Scenario("gather_grouped_s512", _SET_SIZE_S512, "grouped"),
)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = True,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """SCX-only cell-set gather throughput at S=64 and S=512, plus an opt-in
    N-rank concurrency arm. Drops the page cache before each timed run. Returns
    ``None`` for non-SCX formats / missing fixture."""
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
            # Clean skip, matching this function's docstring and the sibling
            # `ooc_loader` / `obs_open` modules — a raise here fails the whole
            # cohort job (several benchmark × dataset tasks share one SLURM job)
            # over a fixture that was simply never converted. The absence is not
            # silent either way: the triple's `thresholds.yaml` floors then report
            # a missing metric at gate time.
            logger.warning(
                "Skipping cellset_gather for %s/%s — no converted SCX fixture "
                "(run Phase A conversion first: --formats %s)",
                format_variant.key,
                dataset.name,
                format_variant.key,
            )
            return None
        scx_path = str(p)

    n_obs = dataset.n_obs
    n_batches = _n_batches_for(n_obs)
    n_runs = max(1, min(n_runs, _MAX_N_RUNS))
    groups = _resolve_groups(scx_path, n_obs)
    n_ranks = resolve_n_ranks()

    result = BenchmarkResult(
        benchmark="cellset_gather",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            # Retained for backward comparability with the pre-S=512 baseline,
            # where a single module-level set size applied to every scenario.
            "set_size": _SET_SIZE_S64,
            "sets_per_batch": _sets_per_batch(_SET_SIZE_S64),
            "set_sizes": {s.name: s.set_size for s in _SCENARIOS},
            "n_batches": n_batches,
            "n_groups": len(groups),
            "cold_cache": cold_cache,
            "n_ranks": n_ranks,
        },
    )
    result.file_size_bytes = Path(scx_path).stat().st_size

    def _plans_for(sc: _Scenario) -> Callable[[int], Iterator[tuple]]:
        if sc.plan_kind == "random":
            return lambda nb: _random_plans(n_obs, nb, sc.set_size, sc.sets_per_batch)
        return lambda nb: _grouped_plans(groups, nb, sc.set_size, sc.sets_per_batch)

    warmup = min(n_batches, _WARMUP_BATCHES)
    for sc in _SCENARIOS:
        if sc.plan_kind == "grouped" and not groups:
            logger.warning(
                "  skipping %s — no usable covariate groups resolved", sc.name
            )
            continue
        plans = _plans_for(sc)
        # Warm code (tokio/rayon) with a tiny pass; timed reads are cold.
        try:
            _run_gather(scx_path, lambda: plans(warmup))
        except Exception as e:  # noqa: BLE001
            logger.error("  warmup failed for %s: %s", sc.name, e)
            continue

        run_sps: list[float] = []
        run_cps: list[float] = []
        run_rss: list[float] = []
        for i in range(n_runs):
            cache_policy = drop_file_cache(scx_path) if cold_cache else "warm"
            try:
                out = _run_gather(scx_path, lambda: plans(n_batches))
            except Exception as e:  # noqa: BLE001
                logger.error("  run %d/%d failed for %s: %s", i + 1, n_runs, sc.name, e)
                continue
            sps = out.n_sets / out.wall_s if out.wall_s > 0 else 0.0
            cps = out.n_cells / out.wall_s if out.wall_s > 0 else 0.0
            result.add_run(
                wall_s=out.wall_s,
                peak_rss_mb=out.peak_rss_mb,
                scenario=sc.name,
                set_size=sc.set_size,
                n_sets=out.n_sets,
                n_cells=out.n_cells,
                cache_policy=cache_policy,
                **{
                    f"cellsets_per_sec__{sc.name}": round(sps, 1),
                    f"cells_per_sec__gather__{sc.name}": round(cps, 0),
                    f"ttfb_first_set_s__{sc.name}": round(out.ttfb_s, 4),
                    f"peak_rss_mb__{sc.name}": round(out.peak_rss_mb, 1),
                    f"shard_cache_hit_rate__{sc.name}": out.shard_cache_hit_rate,
                },
            )
            run_sps.append(sps)
            run_cps.append(cps)
            run_rss.append(out.peak_rss_mb)
            logger.info(
                "    %s (S=%d): wall=%.3fs sets/s=%.1f cells/s=%.0f rss=%.1fMB cache=%s",
                sc.name,
                sc.set_size,
                out.wall_s,
                sps,
                cps,
                out.peak_rss_mb,
                cache_policy,
            )

        if run_sps:
            result.metadata.setdefault("scenario_summary", {})[sc.name] = {
                "n_runs": len(run_sps),
                "set_size": sc.set_size,
                "median_cellsets_per_sec": round(statistics.median(run_sps), 1),
                "median_cells_per_sec": round(statistics.median(run_cps), 0),
                "median_peak_rss_mb": round(statistics.median(run_rss), 1),
            }
        gc.collect()

    # --- P-1(c): N concurrent ranks ---------------------------------------
    # Skipped at N=1: a "4 ranks vs 1 rank" ratio is undefined there, and the
    # arm costs (1 + N) workloads.
    if n_ranks > 1:
        rank_sc = f"gather_random_r{n_ranks}"
        rank_effs: list[float] = []
        for i in range(_RANK_N_RUNS):
            cache_policy = drop_file_cache(scx_path) if cold_cache else "warm"
            try:
                arm = _run_rank_arm(
                    scx_path,
                    n_obs,
                    _SET_SIZE_S64,
                    _sets_per_batch(_SET_SIZE_S64),
                    n_batches,
                    n_ranks,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("  rank arm run %d/%d failed: %s", i + 1, _RANK_N_RUNS, e)
                continue
            if arm is None:
                continue
            result.add_run(
                wall_s=arm["max_wall_s"] or 0.0,
                peak_rss_mb=arm["total_peak_rss_mb"] or 0.0,
                scenario=rank_sc,
                set_size=_SET_SIZE_S64,
                n_ranks=n_ranks,
                n_ranks_reported=arm["n_ranks_reported"],
                cache_policy=cache_policy,
                **{
                    f"cellsets_per_sec__{rank_sc}": arm["aggregate_cellsets_per_sec"],
                    f"cellsets_per_sec_per_rank__{rank_sc}": arm[
                        "per_rank_median_cellsets_per_sec"
                    ],
                    f"cellsets_per_sec_1rank__{rank_sc}": arm["one_rank_cellsets_per_sec"],
                    f"rank_scaling_efficiency__{rank_sc}": arm["rank_scaling_efficiency"],
                    f"total_peak_rss_mb__{rank_sc}": arm["total_peak_rss_mb"],
                },
            )
            if arm["rank_scaling_efficiency"] is not None:
                rank_effs.append(arm["rank_scaling_efficiency"])
            logger.info(
                "    %s: aggregate=%.1f sets/s per_rank=%.1f 1rank=%.1f eff=%s rss_total=%sMB cache=%s",
                rank_sc,
                arm["aggregate_cellsets_per_sec"],
                arm["per_rank_median_cellsets_per_sec"],
                arm["one_rank_cellsets_per_sec"],
                arm["rank_scaling_efficiency"],
                arm["total_peak_rss_mb"],
                cache_policy,
            )
        if rank_effs:
            result.metadata.setdefault("scenario_summary", {})[rank_sc] = {
                "n_runs": len(rank_effs),
                "set_size": _SET_SIZE_S64,
                "n_ranks": n_ranks,
                "median_rank_scaling_efficiency": round(statistics.median(rank_effs), 4),
            }
        gc.collect()

    require_runs(result, scx_path)
    return result
