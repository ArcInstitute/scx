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

``downsample_python`` / ``downsample_rust`` (data-load 1B)
    The same scattered S=64 plan with count-depth augmentation applied two ways:
    gather-then-numpy-per-cell (the status quo, since the rust collate path never
    ran ``finalize_csr_row`` and therefore refused downsampling), versus the draw
    applied inside the gather. Both arms move the same cells, so the ratio
    isolates the draw. Records ``applicable: False`` with a reason when the
    installed pyscx predates the primitive or the fixture is shallower than the
    target — in which case every row short-circuits and a bare ~1.0× would read
    as "the primitive doesn't help".

``collate_rust`` (data-load 1C)
    ``pyscx.collate_cellset_gathered`` on batches gathered **beforehand** —
    the STATE3 "3A hybrid" collation: per-cell preprocessing, the top-K encoder
    crop, drop-to-PAD masking of withheld query genes, the target gather at
    query positions, and ``library_size``. Timed alone because the kernel is
    pure compute and releases the GIL, so folding it into a gather loop would
    report scattered-read time, which every other arm here already measures.

    Two premises the arm asserts rather than assumes. ``enc_mask_positions``
    is **non-empty** and withholds a non-zero number of positions: the kernel
    tests a gene against the withheld set (a lookup per surviving top-K gene)
    only when the mask array is non-empty, so the empty "perturbation path" —
    what the one pre-existing Python call site passes — would skip the branch
    the arm exists to price. And ``hide_readout`` is all-zero,
    because a set bit short-circuits the whole encoder path to a single
    GENE_MASK token. ``k_enc`` / ``k_dec`` / ``mode`` / the mask fraction are
    pinned module constants, since the metric is a rate per cell and every one
    of them scales it.

    The mask premise is measured, not argued. On pbmc3k at the pinned shape
    (``k_enc=2048``, ``k_dec=1024``, 25% withheld, ten S=64 batches), collating
    the *same* batches with the mask array replaced by an empty one runs at
    **78.7 µs/cell against 120.1 µs/cell — 1.53×**. So ~34% of this arm's wall
    was the withheld-gene set and its per-gene probe, and an empty-mask arm
    would have been blind to it.

    That 34% is what OPT-LOADER-4 went after, and the mechanism has since
    changed: the kernel no longer rebuilds a per-row ``HashSet`` from the
    ``k_dec`` query ids and SipHash-probes it, but sorts each *set's* panel once
    and binary-searches it per surviving gene. The branch the premise selects is
    the same one, so the premise and the pinned shape still hold; the figures
    above describe the pre-change kernel and are the "before" side of that
    change, not a current measurement.

Per-run ``extra`` keys (sparse per-scenario):
    ``cellsets_per_sec__<sc>``  — sets/s (the STATE3-relevant throughput)
    ``cells_per_sec__gather__<sc>`` — cells/s
    ``ttfb_first_set_s__<sc>``  — time to first batch (loader spin-up + first gather)
    ``peak_rss_mb__<sc>``       — true in-epoch peak RSS
    ``cache_policy``            — cold_fadvise / warm
    ``shard_cache_hit_rate__<sc>`` — from ``SparseCellSetDataset.cache_metrics()``
        when the counter is present, else ``None``.
Collate arm additionally:
    ``us_per_cell__collate``    — microseconds of collate per cell, the
        allocation-sensitive quantity OPT-LOADER-4 moves
    ``k_enc`` / ``k_dec`` / ``collate_mode`` / ``enc_mask_fraction`` — the
        pinned shape, recorded so a capture is self-describing
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
import os
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
    # Which scattered-read route the gather actually took. `full_shard_groups`
    # = decoded a whole shard into the LRU and served from cache;
    # `block_index_groups` = decoded only the touched row groups and bypassed
    # the LRU. Exactly one of the two is non-zero for a given setting, so the
    # pair is a route label, not two independent counters.
    full_shard_groups: int = 0
    block_index_groups: int = 0


def _supports_scatter_block_index() -> bool:
    """Whether this pyscx build accepts the `scatter_block_index` kwarg.

    Read from the pyo3 text signature rather than inferred from the `TypeError`
    a missing kwarg raises: that catch also swallows a genuine type bug anywhere
    inside the gather or the plan iterator and reports it as "your build is
    old", which is the same silent-miscoverage failure the route counters exist
    to eliminate. Nothing is caught here either — by the time this is called the
    main scenarios have already imported pyscx and constructed the class, so an
    ImportError or AttributeError is a real failure, not an old build.
    """
    import pyscx

    # `getattr` with a default for the attribute ONLY: a test double or a
    # non-pyo3 stand-in is a plain function with no signature metadata, and that
    # genuinely means "cannot offer the kwarg". An ImportError, or
    # `SparseCellSetDataset` missing outright, still raises — those are broken
    # environments, not old builds, and were what the previous blanket
    # `except Exception` wrongly reported as "your pyscx predates the kwarg".
    sig = getattr(pyscx.SparseCellSetDataset, "__text_signature__", None)
    return "scatter_block_index" in (sig or "")


def _cache_hit_rate(ds: Any) -> float | None:
    """Hit rate alone, for the three arms that do not need the route counters."""
    try:
        cm = ds.cache_metrics()
        hits = float(cm.get("hits", 0))
        misses = float(cm.get("misses", 0))
        if hits + misses > 0:
            return round(hits / (hits + misses), 4)
    except Exception:  # noqa: BLE001
        pass
    return None


def _cache_snapshot(ds: Any) -> tuple[int, int, float | None]:
    """`(full_shard_groups, block_index_groups, hit_rate)` from ONE metrics read.

    One read, not three: `cache_metrics()` crosses the Python boundary and the
    counters must describe the same instant to be interpretable together.
    """
    cm = ds.cache_metrics()
    hits = float(cm.get("hits", 0))
    misses = float(cm.get("misses", 0))
    hit_rate = round(hits / (hits + misses), 4) if hits + misses > 0 else None
    return (
        int(cm.get("full_shard_groups", 0)),
        int(cm.get("block_index_groups", 0)),
        hit_rate,
    )


def _run_gather(
    scx_path: str,
    plans_factory: Callable[[], Iterator[tuple]],
    cache_shards: int | None = None,
    max_memory_mb: int | None = None,
    scatter_block_index: bool | None = None,
) -> _Outcome:
    import pyscx

    gc.collect()
    kwargs: dict[str, Any] = {}
    if cache_shards is not None:
        kwargs["cache_shards"] = cache_shards
    if max_memory_mb is not None:
        kwargs["max_memory_mb"] = max_memory_mb
    if scatter_block_index is not None:
        # Left unset by every other caller *on purpose*: the shipped default is
        # the thing under measurement, and passing it explicitly everywhere
        # would make the benchmark blind to a change in it.
        kwargs["scatter_block_index"] = scatter_block_index
    ds = pyscx.SparseCellSetDataset([scx_path], **kwargs)
    n_sets = 0
    n_cells = 0
    ttfb_s = 0.0
    # `try/finally`, not a trailing close: this helper is called several times
    # per scenario per run and its callers catch and continue, so a gather that
    # raises would otherwise leak that reader's mmap and descriptors for the
    # rest of the job.
    try:
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
        # Read the counters BEFORE closing: `close()` is terminal on this class
        # and the accessors raise afterwards.
        full_shard, block_index, hit_rate = _cache_snapshot(ds)
    finally:
        ds.close()
    return _Outcome(
        n_sets=n_sets,
        n_cells=n_cells,
        wall_s=wall_s,
        ttfb_s=ttfb_s,
        peak_rss_mb=sampler.peak_mb,
        shard_cache_hit_rate=hit_rate,
        full_shard_groups=full_shard,
        block_index_groups=block_index,
    )


# ---------------------------------------------------------------------------
# data-load 1A: cache sizing
# ---------------------------------------------------------------------------

# What STATE3 was actually passing when it hit 143 s/batch. SCX's own default is
# 128 and would have been fine; the pathology came from a *consumer-side*
# default, which is why the fix is a diagnostic rather than a default change.
_UNDERSIZED_CACHE_SHARDS = 16


# ---------------------------------------------------------------------------
# data-load 1B: native count downsample vs the Python per-cell equivalent
# ---------------------------------------------------------------------------

# STATE3's configured depth augmentation. Chosen well below any real cell's
# library size so the sampler actually runs on every row; a target above the
# fixture's depth takes the no-op branch and the arm would measure nothing.
_DOWNSAMPLE_TARGET = 2_000
_DOWNSAMPLE_SEED = 7
_DOWNSAMPLE_METHOD = "multinomial"


def _numpy_downsample_batch(batch: dict, rng_seed: int) -> int:
    """The per-cell numpy draw the Rust primitive replaces.

    Deliberately shaped like ``RawCountDownsampler`` + ``finalize_csr_row``'s
    post-processing rather than a vectorised idealisation: a fresh
    ``default_rng`` per cell (it seeds from ``(seed, path, cell_idx)``, so it
    cannot be hoisted), ``np.rint`` to integer trials, the
    ``library_size <= target`` short-circuit, and the positive-only prune. That
    is the cost a consumer pays today, so it is the honest comparison arm.

    Returns the surviving nnz so the caller cannot be optimised away.
    """
    indptr = batch["indptr"]
    data = batch["data"]
    kept = 0
    for r in range(len(indptr) - 1):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        if hi <= lo:
            continue
        counts = data[lo:hi]
        trials = np.rint(counts).astype(np.int64, copy=False)
        library = int(trials.sum(dtype=np.int64))
        if library <= 0 or library <= _DOWNSAMPLE_TARGET:
            kept += int((trials > 0).sum())
            continue
        rng = np.random.default_rng(rng_seed + r)
        if _DOWNSAMPLE_METHOD == "binomial":
            sampled = rng.binomial(trials, _DOWNSAMPLE_TARGET / library)
        else:
            sampled = rng.multinomial(_DOWNSAMPLE_TARGET, trials / library)
        kept += int((sampled > 0).sum())
    return kept


def _run_gather_python_downsample(
    scx_path: str,
    plans_factory: Callable[[], Iterator[tuple]],
) -> _Outcome:
    """Gather without downsampling, then downsample in Python — the status quo."""
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
            _numpy_downsample_batch(batch, _DOWNSAMPLE_SEED)
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


def _run_gather_rust_downsample(
    scx_path: str,
    plans_factory: Callable[[], Iterator[tuple]],
) -> _Outcome:
    """Gather with the downsample applied inside Rust, before the batch returns."""
    import pyscx

    gc.collect()
    ds = pyscx.SparseCellSetDataset(
        [scx_path],
        downsample_target_library_size=_DOWNSAMPLE_TARGET,
        downsample_method=_DOWNSAMPLE_METHOD,
        downsample_seed=_DOWNSAMPLE_SEED,
    )
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


def _downsample_supported(scx_path: str, plan: tuple) -> tuple[bool, str | None]:
    """Probe whether the installed pyscx downsamples natively, and whether the
    fixture is deep enough for the arm to mean anything.

    Returns ``(ok, reason_if_not)``. Never raises: a raise out of ``run`` fails
    the whole cohort SLURM job, and this is one optional arm.
    """
    try:
        import pyscx

        if not hasattr(pyscx, "downsample_counts_csr"):
            return False, "pyscx build predates the native downsampler (scx Phase 1B)"
        plain = pyscx.SparseCellSetDataset([scx_path])
        batch = next(iter(plain.iter_with_plans(iter([plan]))))
        indptr = batch["indptr"]
        data = batch["data"]
        libs = [
            float(data[int(indptr[r]) : int(indptr[r + 1])].sum())
            for r in range(len(indptr) - 1)
        ]
        if not libs:
            return False, "probe batch was empty"
        median_lib = statistics.median(libs)
        if median_lib <= _DOWNSAMPLE_TARGET:
            # Every row would take the no-op branch, so the A/B would be timing a
            # short-circuit. Recording the reason matters: a bare ~1.0× would read
            # as "the Rust primitive doesn't help", which is the wrong conclusion
            # drawn from the wrong fixture.
            return False, (
                f"fixture too shallow: median library size {median_lib:.0f} <= target "
                f"{_DOWNSAMPLE_TARGET}, so every row short-circuits"
            )
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"probe failed: {e}"


# ---------------------------------------------------------------------------
# data-load 1C: collate (OPT-LOADER-4)
# ---------------------------------------------------------------------------


def _collate_supported() -> tuple[bool, str | None]:
    """Whether the installed pyscx exposes the collate kernel at contract v2+.

    Returns ``(ok, reason_if_not)`` and never raises — a raise out of ``run``
    fails the whole cohort SLURM job, and this is one optional arm. Deliberately
    cheaper than ``_downsample_supported``: that probe has to gather a real
    batch to learn whether the fixture is deep enough, whereas this one only
    needs the symbol and the contract version, so it opens no dataset (and
    therefore cannot leak one, which the downsample probe does).
    """
    try:
        import pyscx

        if not hasattr(pyscx, "collate_cellset_gathered"):
            return False, "pyscx build predates the collate kernel"
        version = getattr(pyscx, "COLLATE_CELLSET_CONTRACT_VERSION", None)
        if version is None:
            return False, "pyscx exposes no COLLATE_CELLSET_CONTRACT_VERSION"
        if int(version) < 2:
            return False, (
                f"collate contract v{version} predates the crop/mask semantics "
                f"this arm measures (needs >= 2)"
            )
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"probe failed: {e}"


def _gather_batches(
    scx_path: str, plans_factory: Callable[[], Iterator[tuple]],
) -> list[dict]:
    """Gather batches and hold them, so the collate arm can be timed alone.

    Collate is pure compute on an already-gathered batch. Timing it inside the
    gather loop would report a number dominated by scattered reads — the very
    thing every other arm in this module already measures — and a 0.85x floor on
    that would fire on page-cache weather rather than on the kernel.
    """
    import pyscx

    ds = pyscx.SparseCellSetDataset([scx_path])
    try:
        return list(ds.iter_with_plans(plans_factory()))
    finally:
        ds.close()


def _collate_inputs(batch: dict, rng: np.random.Generator) -> dict[str, Any]:
    """Build the four caller-supplied arrays for one gathered batch.

    The kernel is RNG-free by design — the decoder query panel and the per-cell
    withholding mask are the caller's job — so they are built here, once per
    batch and outside the timed region.

    The query panel is drawn from **genes the set actually contains**, not
    uniformly over the vocabulary. With ~61k genes and ~2k non-zeros per cell a
    uniform panel would essentially never hit, so `masked.contains(gid)` would
    always miss and no gene would ever be withheld: the arm would still pay the
    `HashSet` build (which is most of what OPT-LOADER-4 removes) but would never
    reach the compaction or the all-masked fallback. Drawing from the set's own
    genes also matches what a gene panel is.
    """
    indptr = batch["indptr"]
    indices = batch["indices"]
    set_offsets = batch["set_offsets"]
    n_rows = int(batch["shape"][0])
    n_genes_total = int(batch["shape"][1])
    n_sets = len(set_offsets) - 1
    k_dec = _COLLATE_K_DEC

    query = np.empty(n_sets * k_dec, dtype=np.int32)
    for s in range(n_sets):
        lo, hi = int(set_offsets[s]), int(set_offsets[s + 1])
        present = indices[int(indptr[lo]) : int(indptr[hi])]
        if present.size:
            pool = np.unique(present)
            picked = rng.choice(pool, size=k_dec, replace=pool.size < k_dec)
        else:
            # An empty set still needs a well-formed panel; the kernel's target
            # gather then misses on every position, which is the correct answer.
            picked = rng.integers(0, max(1, n_genes_total), size=k_dec)
        query[s * k_dec : (s + 1) * k_dec] = picked.astype(np.int32, copy=False)

    # Non-empty by construction — see _COLLATE_MASK_FRACTION.
    enc_mask = (
        rng.random(n_rows * k_dec) < _COLLATE_MASK_FRACTION
    ).astype(np.uint8)

    return {
        "query_gene_ids": query,
        "enc_mask_positions": enc_mask,
        # All zero: a non-zero `hide_readout` short-circuits the whole encoder
        # path to a single GENE_MASK token, skipping the sort, the mask and the
        # crop. One set bit per cell would silently delete the arm's subject.
        "hide_readout": np.zeros(n_rows, dtype=np.uint8),
        "n_measured": np.full(n_sets, n_genes_total, dtype=np.uint32),
        "n_genes_total": n_genes_total,
        "n_sets": n_sets,
        "n_rows": n_rows,
    }


def _run_collate(prepared: list[tuple[dict, dict[str, Any]]]) -> _Outcome:
    """Collate every prepared batch, timing only the kernel calls."""
    import pyscx

    gc.collect()
    n_sets = 0
    n_cells = 0
    ttfb_s = 0.0
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        for i, (batch, aux) in enumerate(prepared):
            pyscx.collate_cellset_gathered(
                batch["indptr"],
                batch["indices"],
                batch["data"],
                batch["set_offsets"],
                batch["cell_indices"],
                batch["file_ids"],
                batch["role_tags"],
                _COLLATE_K_DEC,
                aux["query_gene_ids"],
                aux["enc_mask_positions"],
                aux["hide_readout"],
                aux["n_measured"],
                _COLLATE_K_ENC,
                _COLLATE_MODE,
                aux["n_genes_total"],
            )
            if i == 0:
                ttfb_s = time.perf_counter() - t0
            n_sets += aux["n_sets"]
            n_cells += aux["n_rows"]
        wall_s = time.perf_counter() - t0
    return _Outcome(
        n_sets=n_sets,
        n_cells=n_cells,
        wall_s=wall_s,
        ttfb_s=ttfb_s,
        peak_rss_mb=sampler.peak_mb,
        # No dataset is open during a collate, so there is no cache to report.
        shard_cache_hit_rate=None,
    )


# Every one of these is PINNED rather than derived, because the metric is a
# rate per cell and the kernel's cost is a function of all of them: `k_enc`
# sets the PAD-init loop and the number of mask probes, `k_dec` sets the target
# gather and the size of the withheld-gene set, the mask fraction decides
# whether the `HashSet` is built at all. A host-dependent value would make
# `us_per_cell__collate` incomparable between captures, which is the one thing
# a 0.85x floor cannot tolerate.
#
# The magnitudes follow STATE3's 3A hybrid: a top-K encoder crop of ~2k genes
# and a decoder query panel of the same order.
_COLLATE_K_ENC = 2048
_COLLATE_K_DEC = 1024

# `pass_through` on purpose. It is the only mode that needs no `pflog_alpha`,
# and it does the least per-element work (`enc_vals[i] = tgt_vals[i] = raw`),
# which *maximises* the share of wall attributable to the crop / mask / target
# gather — the part OPT-LOADER-4 changes. A log-taking mode would dilute the
# signal with transcendentals the fix does not touch.
_COLLATE_MODE = "pass_through"

# Fraction of each cell's query positions withheld from the encoder crop.
#
# This is NOT a tuning knob, it is the arm's premise. `collate_cell` consults
# the withheld set — a lookup per surviving top-K gene — only when
# `enc_mask_positions` is non-empty; an empty array is the perturbation path and
# takes `None`, skipping the branch the arm exists to measure (and, since
# OPT-LOADER-4, also skipping the per-set panel sort that replaced the per-row
# `HashSet` build). The one pre-existing Python call site
# (`pyscx/tests/test_fork_safety.py`) passes empty, so copying it would have
# produced a green arm over the wrong branch.
_COLLATE_MASK_FRACTION = 0.25

# Batches held resident and collated. At S=64 a batch is ~1024 cells, so ten
# batches is ~10k cells; at tabula's ~1950 nnz/cell that is ~160 MB of CSR —
# enough for a stable rate, small enough not to move the pooled peak much.
_COLLATE_N_BATCHES = 10

# Fixed so two captures collate the same panels and masks.
_COLLATE_SEED = 20260902


# ---------------------------------------------------------------------------
# W6: per-kernel tokenisation arms
# ---------------------------------------------------------------------------
#
# The collate arm above times the whole kernel chain on one shape. These four
# time each `pyscx.tokenize` kernel on its own, so a change to one is not
# averaged away by the other three. They reuse the batches the collate arm
# already gathered — a second gather would dominate the wall and would measure
# scattered reads, which every other arm in this module already covers.
#
# Every shape below is PINNED for the same reason the collate arm's are: the
# metric is a rate per cell and each kernel's cost is a function of its
# parameters. A host-dependent value would make the numbers incomparable
# between captures.

# Encoder crop width — the same as the collate arm's, so the two are comparable.
_TOKENIZE_K = _COLLATE_K_ENC
# Geneformer v2's context length.
_TOKENIZE_L_MAX = 2048
# scGPT's bin count.
_TOKENIZE_N_BINS = 51
# UCE's sample_size.
_TOKENIZE_SAMPLE_N = 1024
# Fixed so two captures draw the same statistics vector and the same samples.
_TOKENIZE_SEED = 20260914

_TOKENIZE_ARMS = ("crop", "rank", "bin", "sample")


def _tokenize_supported() -> tuple[bool, str | None]:
    """Whether the installed pyscx exposes the W6 kernels.

    Never raises, for the same reason `_collate_supported` does not: a raise out
    of `run` fails the whole cohort SLURM job and this is one optional arm.
    """
    try:
        import pyscx

        tok = getattr(pyscx, "tokenize", None)
        if tok is None:
            return False, "pyscx build predates the tokenize namespace"
        version = getattr(tok, "CONTRACT_VERSION", None)
        if version is None:
            return False, "pyscx.tokenize exposes no CONTRACT_VERSION"
        missing = [k for k in ("top_k", "rank_tokens", "bin_values", "sample_genes")
                   if not hasattr(tok, k)]
        if missing:
            return False, f"pyscx.tokenize is missing {missing}"
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, f"probe failed: {e}"


def _tokenize_inputs(
    batch: dict, rng: np.random.Generator, scx_path: str
) -> dict[str, Any]:
    """Build the per-kernel inputs, assert what is load-bearing, record the rest.

    Two kinds of premise, treated differently on purpose:

    **Under this function's control ⇒ raise.** The rank kernel's whole
    difference from a bare sort is the per-gene divide, and an all-ones (or
    near-constant) statistics vector makes it a no-op that reorders nothing. The
    vector is drawn here, so a violation is a bug in this file and must not
    produce a number. Same rule as the collate arm's all-zero-mask guard.

    **A property of the fixture ⇒ report, and skip that arm.** Whether a file's
    cells carry more non-zeros than `n`, or more than one distinct value per
    row, is not something this arm can arrange. `_downsample_supported` already
    set the precedent of gathering a real batch to learn whether the fixture is
    deep enough; the difference is that these are per-kernel, so one thin
    fixture silences one arm rather than all four.

    One premise is deliberately NOT asserted, because stating it would be
    wrong: the crop does not get *more* expensive when it truncates. Its cost is
    the sort over the row's positive entries, which runs whatever `k` is; `k`
    only decides how much of the output is PAD-init. At the shapes here
    (`k = 2048` against ~1950 nnz/cell on tabula) it does not truncate at all —
    Phase 1 measured exactly that and concluded a partial select would save
    nothing. So the ratio is **recorded** rather than gated, and a fixture whose
    rows are far thinner than `k` gets a warning saying the arm is mostly timing
    the PAD-init loop.
    """
    import pyscx

    indptr = batch["indptr"]
    data = batch["data"]
    n_rows = int(batch["shape"][0])
    n_genes_total = int(batch["shape"][1])
    row_nnz = np.diff(indptr.astype(np.int64))
    median_nnz = int(np.median(row_nnz)) if n_rows else 0

    # Strictly positive, and deliberately NOT constant — see the docstring.
    gene_stats = rng.random(n_genes_total).astype(np.float32) * 0.9 + 0.05
    if float(gene_stats.min()) <= 0.0:
        raise RuntimeError("tokenize rank arm: a non-positive gene statistic")
    if float(gene_stats.std()) < 0.05:
        raise RuntimeError(
            "tokenize rank arm: the statistics vector is nearly constant, so the "
            "per-gene divide would not reorder anything and the arm would time a "
            "bare sort"
        )

    # Distinct non-zero values per row, probed rather than computed over every
    # row: the bin kernel's quantile edges all collapse onto one value when a
    # row carries fewer than two, and it then takes the degenerate branch.
    probe = min(n_rows, 64)
    distinct = [
        int(len(np.unique(data[int(indptr[r]) : int(indptr[r + 1])])))
        for r in range(probe)
    ]
    distinct_min = min(distinct) if distinct else 0

    skip: dict[str, str] = {}
    if median_nnz == 0:
        for arm in _TOKENIZE_ARMS:
            skip[arm] = "every probed row is empty"
    if distinct_min < 2:
        skip["bin"] = (
            f"a probed row has {distinct_min} distinct value(s); every quantile "
            "edge would collapse and the arm would time the degenerate branch"
        )
    if median_nnz and median_nnz * 4 < _TOKENIZE_K:
        logger.warning(
            "  tokenize crop arm: median nnz/cell %d is under a quarter of k=%d, "
            "so much of this arm's wall is the PAD-init loop rather than the sort",
            median_nnz,
            _TOKENIZE_K,
        )

    return {
        "gene_stats": gene_stats,
        "n_genes_total": n_genes_total,
        "n_rows": n_rows,
        "n_sets": len(batch["set_offsets"]) - 1,
        "file_identity": int(pyscx.downsample_file_identity(scx_path)),
        "median_nnz": median_nnz,
        "crop_fill_ratio": round(median_nnz / _TOKENIZE_K, 4) if median_nnz else 0.0,
        "distinct_values_min": distinct_min,
        "skip": skip,
    }


def _run_tokenize(arm: str, prepared: list[tuple[dict, dict[str, Any]]]) -> _Outcome:
    """Time one kernel over every prepared batch."""
    import pyscx.tokenize as tok

    gc.collect()
    n_sets = n_cells = 0
    ttfb_s = 0.0
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        for i, (batch, aux) in enumerate(prepared):
            ip, ix, dt = batch["indptr"], batch["indices"], batch["data"]
            if arm == "crop":
                tok.top_k(ip, ix, dt, _TOKENIZE_K, aux["n_genes_total"])
            elif arm == "rank":
                tok.rank_tokens(
                    ip, ix, dt, aux["gene_stats"], _TOKENIZE_L_MAX, "bench",
                )
            elif arm == "bin":
                tok.bin_values(ip, ix, dt, _TOKENIZE_N_BINS)
            elif arm == "sample":
                tok.sample_genes(
                    ip, ix, dt, _TOKENIZE_SAMPLE_N,
                    _TOKENIZE_SEED, aux["file_identity"],
                )
            else:  # pragma: no cover - guarded by _TOKENIZE_ARMS
                raise ValueError(f"unknown tokenize arm {arm!r}")
            if i == 0:
                ttfb_s = time.perf_counter() - t0
            n_sets += aux["n_sets"]
            n_cells += aux["n_rows"]
        wall_s = time.perf_counter() - t0
    return _Outcome(
        n_sets=n_sets,
        n_cells=n_cells,
        wall_s=wall_s,
        ttfb_s=ttfb_s,
        peak_rss_mb=sampler.peak_mb,
        shard_cache_hit_rate=None,
    )


def _budget_mb_for(scx_path: str, cache_shards: int) -> int | None:
    """``max_memory_mb`` that holds ``cache_shards`` average shards, +12% headroom.

    Derived from the loader's own model (``memory_budget()``) rather than
    guessed, so the "sized correctly" arm is configured the way the warning
    tells a user to configure it. ``None`` ⇒ leave the budget adaptive.

    The **non-cache** terms are part of the need, not slack. Since ORG-9.10-5
    the tuner charges the interpreter constant it reports, so a budget sized for
    the shards alone is short by that constant and the auto-tune shrinks the
    cache back below what this arm is trying to test. Both terms are read off
    the same ``memory_budget()`` call so this cannot drift from the model again.
    (It sizes the loader's *cache*; the gathered batch is not charged, so this
    is not a bound on the process.)
    """
    try:
        import pyscx

        budget = pyscx.SparseCellSetDataset([scx_path]).memory_budget()
        per_shard = int(budget["shard_decoded_bytes"])
        if per_shard <= 0:
            return None
        breakdown = budget["breakdown"]
        non_cache = int(breakdown["total_bytes"]) - int(breakdown["cache_bytes"])
        need_mb = (cache_shards * per_shard + non_cache) // (1024 * 1024)
        return max(64, need_mb + need_mb // 8)
    except Exception as e:  # noqa: BLE001
        logger.warning("  could not derive a budget for %d shards: %s", cache_shards, e)
        return None


def _shard_count(scx_path: str) -> int | None:
    """CSR shard count, for context on whether the sizing arm can say anything."""
    try:
        import pyscx

        return int(pyscx.open(scx_path).shard_count)
    except Exception:  # noqa: BLE001
        return None


def _read_path_probe(scx_path: str, plans_factory: Callable[[], Iterator[tuple]]) -> dict:
    """Which scattered-read path this process actually takes, from counters.

    `SparseCellSetDataset` defaults to the full-shard warm+cache route, so on a
    default-constructed dataset this normally reports `full_shard_groups > 0`
    and `hits + misses > 0` — the regime the cache-sizing arm below needs to be
    meaningful. It can still report the other route: `scatter_block_index=True`
    on a **framed** (v4) file decodes only the touched row groups and **never
    populates the whole-shard LRU** (`hits + misses == 0`,
    `block_index_groups > 0`), and there `cache_shards` is irrelevant by
    construction — a ~1.0× cache-sizing ratio then means "the cache was
    bypassed", not "sizing doesn't help".

    Measured here rather than inferred from the file's framing, because the
    route is decided by three things at once: framing, the per-dataset
    `scatter_block_index` kwarg, and the process-global
    `SCX_SCATTER_BLOCK_INDEX` switch (read once per process via `OnceLock`).
    """
    import pyscx

    ds = pyscx.SparseCellSetDataset([scx_path], cache_shards=16)
    for _ in ds.iter_with_plans(plans_factory()):
        pass
    m = ds.cache_metrics()
    touched = int(m.get("hits", 0)) + int(m.get("misses", 0))
    return {
        "lru_consulted": touched > 0,
        "hits_plus_misses": touched,
        "full_shard_groups": int(m.get("full_shard_groups", 0)),
        "block_index_groups": int(m.get("block_index_groups", 0)),
    }


def _suggested_cache_shards(scx_path: str, plans: list[tuple]) -> int | None:
    """Max ``suggested_cache_shards`` over ``plans`` — the cache that would hold
    the widest batch's working set, or ``None`` to skip the arm.

    Returns ``None`` rather than raising on *any* failure: an older ``.so``
    without the method, an unreadable fixture, a broken pyscx. This arm is
    diagnostic, and a raise here would fail the whole cohort SLURM job (several
    benchmark × dataset tasks share one) — and would also mask the
    ``require_runs()`` guard, which is what must report a non-functional pyscx.
    """
    try:
        import pyscx

        probe = pyscx.SparseCellSetDataset([scx_path])
        if not hasattr(probe, "suggested_cache_shards"):
            return None
        # One plan argument, in the shape `iter_with_plans` consumes
        # (ORG-9.10-4 folded the former `(file_ids, rows)` arity into it).
        best = 0
        for plan in plans:
            best = max(best, int(probe.suggested_cache_shards(plan)))
        return best or None
    except Exception as e:  # noqa: BLE001
        logger.warning("  cache-sizing arm unavailable: %s", e)
        return None


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
    if n_obs <= 0:
        # Guarded here rather than per-plan-generator: `_random_plans` would raise
        # on `rng.integers(0, 0)` and `_grouped_plans` on an empty group list, so a
        # single early skip covers both instead of half-covering one.
        logger.warning(
            "Skipping cellset_gather for %s/%s — dataset reports n_obs=%d",
            format_variant.key,
            dataset.name,
            n_obs,
        )
        return None
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
                    # Route label for THIS run, at the shipped default — the
                    # same keys `index_plan.py` already emits per scenario.
                    # These are what make a silent route flip gateable: the
                    # counters were previously computed once, by
                    # `_read_path_probe`, into
                    # `metadata["cache_sizing"]["read_path"]`, and
                    # `compare_against_baseline.py` only ever reads
                    # `runs[].extra` — so no threshold could see them.
                    f"full_shard_groups__{sc.name}": out.full_shard_groups,
                    f"block_index_groups__{sc.name}": out.block_index_groups,
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

    # --- data-load 1A: cache sizing --------------------------------------
    # Reproduces STATE3's pathology and measures what `suggested_cache_shards`
    # buys: the same scattered S=64 plan run with the consumer-side
    # `cache_shards=16` it was passing, then with the value the helper derives
    # from the plan. Not a regression floor — it is an A/B whose *ratio* is the
    # result, and its arms deliberately misconfigure one side.
    sizing_sc = _SCENARIOS[0]  # gather_random @ S=64
    sizing_plans = _plans_for(sizing_sc)
    sample = list(sizing_plans(min(n_batches, 32)))
    suggested = _suggested_cache_shards(scx_path, sample)
    n_shards = _shard_count(scx_path)
    try:
        read_path = _read_path_probe(scx_path, lambda: sizing_plans(warmup))
    except Exception as e:  # noqa: BLE001
        logger.warning("  read-path probe failed: %s", e)
        read_path = {"lru_consulted": None}
    if read_path.get("lru_consulted") is False:
        logger.info(
            "  cache-sizing arm: the whole-shard LRU is BYPASSED on this path "
            "(block_index_groups=%d, hits+misses=0) — `cache_shards` cannot "
            "affect throughput here. That is not the shipped default for this "
            "class. `SCX_SCATTER_BLOCK_INDEX` cannot explain it either — the "
            "env var is an off-switch (`block_index_eligible` ANDs it with the "
            "per-reader flag), so it can only force the full-shard path, never "
            "this one. The remaining explanations are a pyscx predating the "
            "`scatter_block_index` kwarg (reader default `true`) or the "
            "per-reader setter being dropped.",
            read_path.get("block_index_groups", 0),
        )
    if suggested is None:
        logger.warning(
            "  skipping cache-sizing arm — suggested_cache_shards unavailable "
            "(stale pyscx .so?)"
        )
    elif suggested <= _UNDERSIZED_CACHE_SHARDS:
        # A file with few shards cannot demonstrate anything here: a scattered
        # plan touches ≤ the undersized cache, so both arms hold the whole
        # working set and the ratio is 1.00× *by construction*. Recording the
        # reason matters — a bare 1.00× reads as "the helper doesn't help",
        # which would be the wrong conclusion drawn from the wrong fixture.
        logger.info(
            "  cache-sizing arm not applicable: plan touches %d shard(s) "
            "(file has %s), at or below the undersized cache of %d — needs a "
            "many-shard dataset (census_*) to be informative",
            suggested,
            n_shards if n_shards is not None else "?",
            _UNDERSIZED_CACHE_SHARDS,
        )
        result.metadata["cache_sizing"] = {
            "applicable": False,
            "read_path": read_path,
            "reason": "plan working set fits the undersized cache; fixture has too few shards",
            "suggested_cache_shards": suggested,
            "undersized_cache_shards": _UNDERSIZED_CACHE_SHARDS,
            "file_shard_count": n_shards,
        }
    else:
        # Following the advice means raising BOTH knobs. `suggested_cache_shards`
        # is a count, but the byte budget also has to hold that many shards —
        # census_500k wants 31 while the adaptive 4 GB cap affords only 22, so
        # `cache_shards=31` alone would under-deliver and understate the win.
        # This is exactly what the sizing warning tells a user to do.
        suggested_mb = _budget_mb_for(scx_path, suggested)
        logger.info(
            "  cache sizing: suggested=%d shards (max_memory_mb=%s) vs undersized=%d",
            suggested,
            suggested_mb,
            _UNDERSIZED_CACHE_SHARDS,
        )
        sizing_rates: dict[str, float] = {}
        for arm_name, shards, budget_mb in (
            ("cache_undersized", _UNDERSIZED_CACHE_SHARDS, None),
            ("cache_suggested", suggested, suggested_mb),
        ):
            try:
                _run_gather(scx_path, lambda: sizing_plans(warmup), shards, budget_mb)
            except Exception as e:  # noqa: BLE001
                logger.error("  warmup failed for %s: %s", arm_name, e)
                continue
            arm_sps: list[float] = []
            arm_rss: list[float] = []
            for i in range(n_runs):
                cache_policy = drop_file_cache(scx_path) if cold_cache else "warm"
                try:
                    out = _run_gather(
                        scx_path, lambda: sizing_plans(n_batches), shards, budget_mb
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error("  run %d/%d failed for %s: %s", i + 1, n_runs, arm_name, e)
                    continue
                sps = out.n_sets / out.wall_s if out.wall_s > 0 else 0.0
                result.add_run(
                    wall_s=out.wall_s,
                    peak_rss_mb=out.peak_rss_mb,
                    scenario=arm_name,
                    set_size=sizing_sc.set_size,
                    n_sets=out.n_sets,
                    n_cells=out.n_cells,
                    cache_shards=shards,
                    max_memory_mb=budget_mb,
                    cache_policy=cache_policy,
                    **{
                        f"cellsets_per_sec__{arm_name}": round(sps, 1),
                        f"peak_rss_mb__{arm_name}": round(out.peak_rss_mb, 1),
                        f"shard_cache_hit_rate__{arm_name}": out.shard_cache_hit_rate,
                    },
                )
                arm_sps.append(sps)
                arm_rss.append(out.peak_rss_mb)
                logger.info(
                    "    %s (cache_shards=%d, budget=%s): sets/s=%.1f rss=%.1fMB hit_rate=%s",
                    arm_name,
                    shards,
                    budget_mb,
                    sps,
                    out.peak_rss_mb,
                    out.shard_cache_hit_rate,
                )
            if arm_sps:
                sizing_rates[arm_name] = statistics.median(arm_sps)
                result.metadata.setdefault("scenario_summary", {})[arm_name] = {
                    "n_runs": len(arm_sps),
                    "cache_shards": shards,
                    "max_memory_mb": budget_mb,
                    "median_cellsets_per_sec": round(statistics.median(arm_sps), 1),
                    "median_peak_rss_mb": round(statistics.median(arm_rss), 1),
                }
            gc.collect()

        if len(sizing_rates) == 2 and sizing_rates["cache_undersized"] > 0:
            speedup = sizing_rates["cache_suggested"] / sizing_rates["cache_undersized"]
            result.metadata["cache_sizing"] = {
                # `applicable` says the arms RAN; `read_path.lru_consulted` says
                # whether the knob under test was even on the critical path. A
                # ~1.0× with lru_consulted=False is the cache being bypassed.
                "applicable": True,
                "read_path": read_path,
                "file_shard_count": n_shards,
                "suggested_cache_shards": suggested,
                "suggested_max_memory_mb": suggested_mb,
                "undersized_cache_shards": _UNDERSIZED_CACHE_SHARDS,
                "undersized_cellsets_per_sec": round(sizing_rates["cache_undersized"], 1),
                "suggested_cellsets_per_sec": round(sizing_rates["cache_suggested"], 1),
                "speedup": round(speedup, 3),
            }
            logger.info(
                "  cache sizing: %d→%d shards = %.2f×",
                _UNDERSIZED_CACHE_SHARDS,
                suggested,
                speedup,
            )

    # --- the scattered-read route: a capability probe, NOT a timed A/B ----
    # `SparseCellSetDataset` picks between two scattered-read routes, and until
    # the kwarg was restored the choice was neither settable per dataset nor
    # visible to the gate: the shipped default silently flipped in the sidecar
    # removal and no benchmark, floor or test noticed for two months.
    #
    #   default / False   full-shard decode into the shared LRU, served from
    #                     cache on reuse. The shipped default.
    #   True              decode only the touched row groups, bypassing the LRU.
    #                     Bounded peak RAM; for a working set that exceeds the
    #                     cache.
    #
    # The gateable signal is the per-scenario `full_shard_groups__*` /
    # `block_index_groups__*` extras already emitted above, on the runs that
    # execute anyway. This block adds only a capability bit.
    #
    # ⚠️ It deliberately does NOT time both routes and `add_run` them. Those
    # samples would land in the same `runs[]` that `BenchmarkResult.median_wall_s`
    # and `peak_rss_mb_median` are computed over, and that
    # `compare_against_baseline.py` gates — so a diagnostic arm would move this
    # triple's timing and RSS rows. That is not hypothetical: the committed
    # `cellset_gather_s512_raises_pooled_rss.md` justification exists precisely
    # because adding the S=512 arms moved pooled `peak_rss_mb_median` +19.47%.
    # It is also a wall-clock trap to time: on framed tabula the block-index
    # route runs at 3.19 vs 587.5 cellsets/s, so a timed 50-batch arm costs ~4
    # minutes per run. The priced comparison lives in
    # `benchmarks/scripts/bench_cellset_scatter_routes.py`, which reframes a
    # copy at a pinned row-group geometry.
    #
    # ⚠️ That trap is no longer hypothetical. This comment used to read "the
    # moment the fixtures are reframed to v4" — the registered `*_auto.scx`
    # fixtures ARE v4-framed today (pyscx 0.11.0, `shufdelta`), against the
    # v3 / `scx1` ones this block was written for. So the probe below reaches
    # the block-index route on the registered fixture, and the reader default
    # `scatter_block_index=false` is the only thing keeping the *timed*
    # scenarios on the full-shard path.
    probe_sc = _SCENARIOS[0]  # gather_random @ S=64
    if not _supports_scatter_block_index():
        logger.warning(
            "  route probe unavailable: this pyscx has no `scatter_block_index` "
            "kwarg on SparseCellSetDataset"
        )
    else:
        try:
            probe_out = _run_gather(
                scx_path, lambda: _plans_for(probe_sc)(1), scatter_block_index=True
            )
        except Exception as e:  # noqa: BLE001
            logger.error("  route probe failed: %s", e)
        else:
            capable = probe_out.block_index_groups > 0
            env_switch = os.environ.get("SCX_SCATTER_BLOCK_INDEX")
            killed = env_switch is not None and (
                env_switch == "0" or env_switch.lower() == "false"
            )
            result.metadata["scatter_block_index_route"] = {
                # Whether the block-index route was reached with the kwarg on.
                # `False` does NOT mean "unframed" on its own: the predicate
                # `block_index_eligible` ANDs three more things — the
                # process-global `SCX_SCATTER_BLOCK_INDEX` switch, the row-range
                # cost window, and `shard_is_framed`. The env value is recorded
                # beside the verdict so a `false` can be attributed rather than
                # guessed at; if it is off, that alone explains this and nothing
                # about the fixture follows.
                "block_index_reachable": capable,
                "env_scatter_block_index": env_switch,
                "env_kill_switch_engaged": killed,
                "probe_full_shard_groups": probe_out.full_shard_groups,
                "probe_block_index_groups": probe_out.block_index_groups,
                "n_batches": 1,
                "timed": False,
            }
            logger.info(
                "  route probe: block_index_reachable=%s "
                "(full_shard=%d block_index=%d, 1 batch, untimed%s)",
                capable,
                probe_out.full_shard_groups,
                probe_out.block_index_groups,
                "; SCX_SCATTER_BLOCK_INDEX kill-switch is ENGAGED, which alone "
                "explains an unreachable route" if killed else "",
            )


    # --- data-load 1B: native downsample vs the Python per-cell draw -------
    # The capability 1B ships is that `scx_rust_collate` and count-depth
    # augmentation can finally coexist; this arm prices what moving the draw into
    # the gather costs (or saves) against the status quo, which is gather-then-
    # numpy-per-cell. Both arms move the same cells over the same plans, so the
    # ratio isolates the draw. Not a regression floor.
    ds_sc = _SCENARIOS[0]  # gather_random @ S=64
    ds_plans = _plans_for(ds_sc)
    ds_ok, ds_reason = _downsample_supported(scx_path, next(iter(ds_plans(1))))
    if not ds_ok:
        logger.info("  downsample arm not applicable: %s", ds_reason)
        result.metadata["downsample"] = {"applicable": False, "reason": ds_reason}
    else:
        ds_rates: dict[str, float] = {}
        for arm_name, runner in (
            ("downsample_python", _run_gather_python_downsample),
            ("downsample_rust", _run_gather_rust_downsample),
        ):
            try:
                runner(scx_path, lambda: ds_plans(warmup))
            except Exception as e:  # noqa: BLE001
                logger.error("  warmup failed for %s: %s", arm_name, e)
                continue
            arm_sps: list[float] = []
            arm_rss: list[float] = []
            for i in range(n_runs):
                cache_policy = drop_file_cache(scx_path) if cold_cache else "warm"
                try:
                    out = runner(scx_path, lambda: ds_plans(n_batches))
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "  run %d/%d failed for %s: %s", i + 1, n_runs, arm_name, e
                    )
                    continue
                sps = out.n_sets / out.wall_s if out.wall_s > 0 else 0.0
                result.add_run(
                    wall_s=out.wall_s,
                    peak_rss_mb=out.peak_rss_mb,
                    scenario=arm_name,
                    set_size=ds_sc.set_size,
                    n_sets=out.n_sets,
                    n_cells=out.n_cells,
                    downsample_target_library_size=_DOWNSAMPLE_TARGET,
                    downsample_method=_DOWNSAMPLE_METHOD,
                    cache_policy=cache_policy,
                    **{
                        f"cellsets_per_sec__{arm_name}": round(sps, 1),
                        f"peak_rss_mb__{arm_name}": round(out.peak_rss_mb, 1),
                    },
                )
                arm_sps.append(sps)
                arm_rss.append(out.peak_rss_mb)
                logger.info(
                    "    %s: sets/s=%.1f rss=%.1fMB cache=%s",
                    arm_name,
                    sps,
                    out.peak_rss_mb,
                    cache_policy,
                )
            if arm_sps:
                ds_rates[arm_name] = statistics.median(arm_sps)
                result.metadata.setdefault("scenario_summary", {})[arm_name] = {
                    "n_runs": len(arm_sps),
                    "set_size": ds_sc.set_size,
                    "median_cellsets_per_sec": round(statistics.median(arm_sps), 1),
                    "median_peak_rss_mb": round(statistics.median(arm_rss), 1),
                }
            gc.collect()

        if len(ds_rates) == 2 and ds_rates["downsample_python"] > 0:
            speedup = ds_rates["downsample_rust"] / ds_rates["downsample_python"]
            result.metadata["downsample"] = {
                "applicable": True,
                "target_library_size": _DOWNSAMPLE_TARGET,
                "method": _DOWNSAMPLE_METHOD,
                "python_cellsets_per_sec": round(ds_rates["downsample_python"], 1),
                "rust_cellsets_per_sec": round(ds_rates["downsample_rust"], 1),
                "speedup": round(speedup, 3),
            }
            logger.info("  downsample: rust vs python-per-cell = %.2f×", speedup)

    # --- data-load 1C: collate throughput (OPT-LOADER-4) -----------------
    # `pyscx.collate_cellset_gathered` ran in no benchmark and had no floor —
    # `thresholds.yaml` matched zero lines for "collate" — which is what P0-6
    # names. The `downsample_rust` arm above is gather+downsample, a different
    # call.
    #
    # Timed on batches gathered *beforehand*: collate is pure compute and
    # releases the GIL, so folding it into a gather loop would report I/O.
    co_ok, co_reason = _collate_supported()
    if not co_ok:
        logger.info("  collate arm not applicable: %s", co_reason)
        result.metadata["collate"] = {"applicable": False, "reason": co_reason}
    else:
        co_sc = _SCENARIOS[0]  # gather_random @ S=64 — the STATE3 set size
        co_plans = _plans_for(co_sc)
        co_arm = "collate_rust"
        try:
            co_batches = _gather_batches(
                scx_path, lambda: co_plans(_COLLATE_N_BATCHES)
            )
        except Exception as e:  # noqa: BLE001
            logger.error("  collate arm gather failed: %s", e)
            result.metadata["collate"] = {
                "applicable": False, "reason": f"gather failed: {e}",
            }
            co_batches = []
        if co_batches:
            rng = np.random.default_rng(_COLLATE_SEED)
            prepared = [(b, _collate_inputs(b, rng)) for b in co_batches]
            masked_positions = int(
                sum(int(aux["enc_mask_positions"].sum()) for _b, aux in prepared)
            )
            # Premise, asserted rather than assumed: an all-zero mask array is
            # *non-empty* and so still takes the kernel's masking branch, but a
            # mask that withholds nothing would leave the compaction loop with
            # nothing to skip. Zero here means the RNG or the fraction changed.
            if masked_positions == 0:
                raise RuntimeError(
                    "collate arm built an all-zero enc_mask_positions "
                    f"({_COLLATE_MASK_FRACTION=}); the withheld-gene set would "
                    "be empty and the arm would measure the unmasked path"
                )
            try:
                _run_collate(prepared)  # warm the kernel + allocator
            except Exception as e:  # noqa: BLE001
                logger.error("  collate warmup failed: %s", e)
                result.metadata["collate"] = {
                    "applicable": False, "reason": f"warmup failed: {e}",
                }
                prepared = []
            co_rates: list[float] = []
            co_us: list[float] = []
            for i in range(n_runs if prepared else 0):
                try:
                    out = _run_collate(prepared)
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        "  collate run %d/%d failed: %s", i + 1, n_runs, e
                    )
                    continue
                sps = out.n_sets / out.wall_s if out.wall_s > 0 else 0.0
                us_per_cell = (
                    out.wall_s * 1e6 / out.n_cells if out.n_cells else 0.0
                )
                result.add_run(
                    wall_s=out.wall_s,
                    peak_rss_mb=out.peak_rss_mb,
                    scenario=co_arm,
                    set_size=co_sc.set_size,
                    n_sets=out.n_sets,
                    n_cells=out.n_cells,
                    # The kernel's cost is a function of all four, so they are
                    # recorded beside the rate rather than only in the source.
                    k_enc=_COLLATE_K_ENC,
                    k_dec=_COLLATE_K_DEC,
                    collate_mode=_COLLATE_MODE,
                    enc_mask_fraction=_COLLATE_MASK_FRACTION,
                    # Page cache is irrelevant here — nothing is read — so the
                    # arm records what it is rather than borrowing the gather
                    # arms' cold/warm label.
                    cache_policy="in_memory",
                    **{
                        f"cellsets_per_sec__{co_arm}": round(sps, 1),
                        "us_per_cell__collate": round(us_per_cell, 3),
                        f"peak_rss_mb__{co_arm}": round(out.peak_rss_mb, 1),
                        f"ttfb_first_set_s__{co_arm}": round(out.ttfb_s, 4),
                    },
                )
                co_rates.append(sps)
                co_us.append(us_per_cell)
                logger.info(
                    "    collate: sets/s=%.1f us/cell=%.2f rss=%.1fMB",
                    sps, us_per_cell, out.peak_rss_mb,
                )
            if co_rates:
                result.metadata["collate"] = {
                    "applicable": True,
                    "k_enc": _COLLATE_K_ENC,
                    "k_dec": _COLLATE_K_DEC,
                    "mode": _COLLATE_MODE,
                    "enc_mask_fraction": _COLLATE_MASK_FRACTION,
                    "n_batches": len(prepared),
                    "masked_positions": masked_positions,
                    "median_cellsets_per_sec": round(
                        statistics.median(co_rates), 1
                    ),
                    "median_us_per_cell": round(statistics.median(co_us), 3),
                }
                result.metadata.setdefault("scenario_summary", {})[co_arm] = {
                    "n_runs": len(co_rates),
                    "set_size": co_sc.set_size,
                    "median_cellsets_per_sec": round(
                        statistics.median(co_rates), 1
                    ),
                    "median_us_per_cell": round(statistics.median(co_us), 3),
                }
        # --- W6: per-kernel tokenisation arms ----------------------------
        # Reuses the batches the collate arm gathered, so this costs no I/O.
        tk_ok, tk_reason = _tokenize_supported()
        if not tk_ok:
            logger.info("  tokenize arms not applicable: %s", tk_reason)
            result.metadata["tokenize"] = {"applicable": False, "reason": tk_reason}
        elif not co_batches:
            result.metadata["tokenize"] = {
                "applicable": False,
                "reason": "no gathered batches (the collate arm's gather failed)",
            }
        else:
            tk_rng = np.random.default_rng(_TOKENIZE_SEED)
            # Deliberately NOT wrapped in a try: a premise failure inside
            # `_tokenize_inputs` must propagate. The arm would still produce a
            # number without its premise, and a number over the wrong branch is
            # worse than no number — the same rule as the collate arm's
            # all-zero-mask guard, which also raises.
            tk_prepared = [
                (b, _tokenize_inputs(b, tk_rng, scx_path)) for b in co_batches
            ]
            tk_summary: dict[str, Any] = {
                "applicable": True,
                "contract_version": int(
                    __import__("pyscx").tokenize.CONTRACT_VERSION
                ),
                "k": _TOKENIZE_K,
                "l_max": _TOKENIZE_L_MAX,
                "n_bins": _TOKENIZE_N_BINS,
                "sample_n": _TOKENIZE_SAMPLE_N,
                "n_batches": len(tk_prepared),
                "median_nnz_per_cell": tk_prepared[0][1]["median_nnz"],
                "crop_fill_ratio": tk_prepared[0][1]["crop_fill_ratio"],
                "min_distinct_values_per_probed_row": tk_prepared[0][1][
                    "distinct_values_min"
                ],
                "arms": {},
            }
            # A fixture-driven skip is recorded per arm, never silently dropped:
            # an absent metric reads as "not captured", and the reason is what
            # tells a later reader whether to go find a deeper fixture.
            tk_skips = tk_prepared[0][1]["skip"]
            for arm in _TOKENIZE_ARMS:
                tk_arm = f"tokenize_{arm}"
                if arm in tk_skips:
                    logger.info("  tokenize %s arm skipped: %s", arm, tk_skips[arm])
                    tk_summary["arms"][arm] = {
                        "applicable": False,
                        "reason": tk_skips[arm],
                    }
                    continue
                try:
                    _run_tokenize(arm, tk_prepared)  # warm kernel + allocator
                except Exception as e:  # noqa: BLE001
                    logger.error("  tokenize %s warmup failed: %s", arm, e)
                    tk_summary["arms"][arm] = {"error": str(e)}
                    continue
                tk_us: list[float] = []
                for i in range(n_runs):
                    try:
                        out = _run_tokenize(arm, tk_prepared)
                    except Exception as e:  # noqa: BLE001
                        logger.error(
                            "  tokenize %s run %d/%d failed: %s",
                            arm, i + 1, n_runs, e,
                        )
                        continue
                    sps = out.n_sets / out.wall_s if out.wall_s > 0 else 0.0
                    us_per_cell = (
                        out.wall_s * 1e6 / out.n_cells if out.n_cells else 0.0
                    )
                    result.add_run(
                        wall_s=out.wall_s,
                        peak_rss_mb=out.peak_rss_mb,
                        scenario=tk_arm,
                        set_size=co_sc.set_size,
                        n_sets=out.n_sets,
                        n_cells=out.n_cells,
                        # Recorded beside the rate: each kernel's cost is a
                        # function of its own parameter, so a rate without it is
                        # not comparable between captures.
                        k_enc=_TOKENIZE_K,
                        l_max=_TOKENIZE_L_MAX,
                        n_bins=_TOKENIZE_N_BINS,
                        sample_n=_TOKENIZE_SAMPLE_N,
                        cache_policy="in_memory",
                        **{
                            f"cellsets_per_sec__{tk_arm}": round(sps, 1),
                            f"us_per_cell__{arm}": round(us_per_cell, 3),
                            f"peak_rss_mb__{tk_arm}": round(out.peak_rss_mb, 1),
                            f"ttfb_first_set_s__{tk_arm}": round(out.ttfb_s, 4),
                        },
                    )
                    tk_us.append(us_per_cell)
                    logger.info(
                        "    tokenize %s: us/cell=%.2f rss=%.1fMB",
                        arm, us_per_cell, out.peak_rss_mb,
                    )
                if tk_us:
                    tk_summary["arms"][arm] = {
                        "n_runs": len(tk_us),
                        "median_us_per_cell": round(statistics.median(tk_us), 3),
                    }
                    result.metadata.setdefault("scenario_summary", {})[tk_arm] = {
                        "n_runs": len(tk_us),
                        "set_size": co_sc.set_size,
                        "median_us_per_cell": round(statistics.median(tk_us), 3),
                    }
            result.metadata["tokenize"] = tk_summary
            tk_prepared = None

        # Both names have to go: `prepared` holds references to the same batch
        # dicts, so dropping only `co_batches` would leave ~160 MB of CSR
        # resident through the rank arm and into this triple's pooled peak.
        co_batches = None
        prepared = None
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
