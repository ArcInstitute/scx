"""
IndexPlanDataset Throughput benchmark.

Plan-driven paired-batch reads (the perturbation-training workload). SCX-only
(`format_variant.key == "scx_auto"`) — for every other variant `run` returns
``None`` so the orchestrator silently skips it. Mirrors the ``fragment_ops``
and ``cell_eval_parity_perf`` gating pattern.

Scenarios
---------
``pyscx_index_plan_random``
    `pyscx.IndexPlanDataset` driven by a uniformly-random plan generator.
    Pessimistic locality assumption — stresses shard cache misses.

``pyscx_index_plan_locality``
    `pyscx.IndexPlanDataset` driven by a controlled-locality plan generator
    (`(cell_type, batch)`-keyed pairing simulated by chunking the row index
    space when no obs columns are available). Realistic ST training pattern.

``pyscx_backed_python_loop``
    `pyscx.open(path).to_anndata(backed=True).X[plan]` row gather + Python-
    side projection / normalize. The cell-load-scx `ScxBackedSparseDataset`
    path that this benchmark exists to characterise.

``pyscx_training_dataset``
    `pyscx.TrainingDataset` sequential. Different access pattern; included
    as a throughput ceiling, not a like-for-like comparison.

Per-scenario metric keys emitted into ``RunRecord.extra``:
    ``batches_per_sec__<scenario>``  — float
    ``cells_per_sec__<scenario>``    — float (= 2 × pairs × batches / s)
    ``peak_rss_mb__<scenario>``      — float (best-effort delta)
    ``shard_cache_hit_rate__<scenario>`` — None (placeholder; the
        ``BackedCsrReader`` does not expose a hit-rate counter today; will
        appear once that lands).
    ``memory_budget_total_mb__<scenario>`` — float;
        ``IndexPlanDataset.memory_budget()["breakdown"]["total_bytes"]`` in MB; ``None``
        for non-IndexPlan scenarios. Lets the harness validate the loader's
        per-component memory model against the actual RSS at scenario end.
    ``estimate_overshoot_mb__<scenario>`` — float; scenario-local
        ``ru_maxrss`` growth (``post − pre``, floored at 0) minus
        ``memory_budget_total_mb``. The growth term — not the absolute peak
        — is used because ``ru_maxrss`` is process-wide and monotonic, so
        subtracting the baseline isolates this scenario's contribution from
        prior scenarios' high-water marks. Positive ⇒ estimator
        under-counted relative to growth; ≤ 0 ⇒ scenario did not push past
        any prior peak (or under-ran the model).
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    ML_BATCH_SIZE,
    QUERY_N_HVGS,
)
from benchmarks.comprehensive.data_wait import (
    data_wait_fraction,
    steady_state_wait,
)
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)


# Only triggered for scx_auto. We don't re-measure across codecs since the
# IndexPlanDataset path is codec-agnostic at the API level.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Mirrors the runtime
guard at the top of ``run()`` (defense-in-depth for direct invocation)."""

# Defaults: 1024 pairs/batch, 1000 batches. Capped against dataset.n_obs
# at runtime so a 100-cell test fixture doesn't blow up.
_DEFAULT_PAIRS_PER_BATCH = 1024
_DEFAULT_N_BATCHES = 1000

# Locality grouping fallback when no categorical obs column is suitable.
# Cells with the same `index // _LOCALITY_GROUP_SIZE` form a group; pert and
# ctrl are sampled from the same group. A group of 4096 corresponds to roughly
# one shard of the typical 16k-shard fixture, giving high cache reuse.
_LOCALITY_GROUP_SIZE = 4096

# Cost-knob scaling (mirrors read_scattered.py). The fixed 1000-batch × up-to-5
# timed-run × up-to-5-scenario worst case (plus a full 1000-batch untimed
# warm-up per scenario) times out ~30 min/run on framed (v4) 50k/100k-cell
# `scx_auto` fixtures — the same class the pre-tune read_scattered hit. Scale
# batch + run counts down for large datasets so the per-job wall fits the SLURM
# budget. Small (<30k-cell) datasets keep the historical 1000-batch / uncapped-
# run behaviour so their existing baseline rows stay valid; only the untimed
# warm-up is capped universally (it feeds cache/tokio init, never a recorded
# metric, so shrinking it is measurement-neutral).
#
# Intentional asymmetry vs read_scattered.py: read_scattered is a *new*
# benchmark with no historical baseline, so it always scales and floors at 12
# batches. index_plan already has baseline rows (incl. absolute bps floors in
# thresholds.yaml), so it keeps the <30k full-1000 branch and uses a higher
# _MIN_N_BATCHES (50) to bound how far the >=30k throughput measurement can move
# from the value the existing floors were calibrated against. NOTE: the >=30k
# absolute bps floors (tabula/census_1m) were calibrated under the old
# 1000-batch regime; re-validate/re-tune them when the deferred index_plan
# recapture folds the smartseq2/tabula rows into LATEST.
_LARGE_N_OBS = 30_000
_MIN_N_BATCHES = 50
_WARMUP_BATCHES = 3


def _n_batches_for(n_obs: int) -> int:
    """Batch count: full ``_DEFAULT_N_BATCHES`` under ``_LARGE_N_OBS``, else
    scaled inversely with dataset size and floored at ``_MIN_N_BATCHES``."""
    if n_obs < _LARGE_N_OBS:
        return _DEFAULT_N_BATCHES
    scaled = 800_000 // max(1, n_obs)
    return int(min(_DEFAULT_N_BATCHES, max(_MIN_N_BATCHES, scaled)))


def _n_runs_for(n_obs: int, harness_n_runs: int) -> int:
    """Timed-run count: unchanged under ``_LARGE_N_OBS``; capped to 2 on large
    (multi-shard, slow-per-batch) datasets. Throughput medians are stable, so a
    large dataset doesn't need the harness's default 5 passes."""
    if n_obs < _LARGE_N_OBS:
        return harness_n_runs
    return max(1, min(harness_n_runs, 2))


def _have_pyscx() -> bool:
    try:
        import pyscx  # noqa: F401

        return True
    except ImportError:
        return False


def _peak_rss_mb() -> float:
    """High-water-mark RSS via ``ru_maxrss``."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _null_model_ms() -> float:
    """Per-batch fixed cost for the R3 data-wait arm, in ms. ``0`` = off.

    ``pyscx_index_plan_dataset_workers2`` drives a consumer loop with **no
    model step at all** — it counts batches — so a data-wait fraction measured
    against it is ~1.0 by construction and says nothing about whether a
    training run would be data-bound. The fraction needs a step to be a
    fraction *of*, so this knob buys one: ``time.sleep`` (not a spin, because
    releasing the GIL while the two DataLoader workers prefetch is what a real
    step does) of a declared duration per batch.

    Read per call rather than pinned at import so a test can set the env and
    re-read it without reloading the module, and **default 0**: a fixed cost
    added unconditionally would inflate this benchmark's pooled
    ``median_wall_s``, which is gated against ``LATEST``, and read as a timing
    regression on a benchmark whose subject did not change. Phase 0 turns it on
    for one separate, non-promoted capture.

    A malformed value resolves to ``0`` rather than raising: this is read
    inside a capture that may already be hours in.
    """
    raw = os.environ.get("SCX_BENCH_R3_NULL_MODEL_MS", "")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "SCX_BENCH_R3_NULL_MODEL_MS=%r is not a number — R3 null model off",
            raw,
        )
        return 0.0


# ---------------------------------------------------------------------------
# Plan generators
# ---------------------------------------------------------------------------


def _random_plans(
    n_obs: int, pairs_per_batch: int, n_batches: int, seed: int = 0
) -> Iterator[list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        pert = rng.integers(0, n_obs, size=pairs_per_batch).astype(np.int64)
        ctrl = rng.integers(0, n_obs, size=pairs_per_batch).astype(np.int64)
        yield list(zip(map(int, pert), map(int, ctrl)))


def _locality_plans(
    n_obs: int,
    group_size: int,
    pairs_per_batch: int,
    n_batches: int,
    seed: int = 0,
) -> Iterator[list[tuple[int, int]]]:
    """Both pert and ctrl drawn from the same `index // group_size` group.
    Mimics `(cell_type, batch)`-keyed pairing without depending on obs
    columns (some test fixtures have only `cell_id`)."""
    rng = np.random.default_rng(seed)
    n_groups = max(1, (n_obs + group_size - 1) // group_size)
    for _ in range(n_batches):
        plan = []
        groups = rng.integers(0, n_groups, size=pairs_per_batch)
        for g in groups:
            lo = int(g) * group_size
            hi = min(lo + group_size, n_obs)
            pert = int(rng.integers(lo, hi))
            ctrl = int(rng.integers(lo, hi))
            plan.append((pert, ctrl))
        yield plan


# ---------------------------------------------------------------------------
# Per-scenario runners
# ---------------------------------------------------------------------------


@dataclass
class _ScenarioOutcome:
    n_batches: int
    n_cells: int
    wall_s: float
    peak_rss_mb: float
    # Estimator validation: `memory_budget_total_mb` mirrors the loader's
    # `IndexPlanDataset.memory_budget()["breakdown"]["total_bytes"]` in MB;
    # `estimate_overshoot_mb` is `(scenario-local ru_maxrss growth) -
    # memory_budget_total_mb`. The growth term is `max(0, ru_maxrss_after -
    # ru_maxrss_before)`, NOT `peak_rss_mb` — `ru_maxrss` is process-wide
    # and monotonic, so subtracting the baseline is what isolates *this*
    # scenario's contribution from the cumulative high-water set by earlier
    # scenarios in the same process. A positive overshoot means the
    # estimator under-counted relative to actual growth; a negative or
    # zero value means the scenario did not push past any prior peak (or
    # under-ran the model). `None` for non-IndexPlan scenarios that don't
    # expose `memory_budget()`.
    memory_budget_total_mb: float | None = None
    estimate_overshoot_mb: float | None = None
    # Sidecar adoption: count of `read_rows_with` shard request-groups served
    # by the O(rows) scx1 decode sidecar vs. full-shard decode (cumulative,
    # loader-level, sampled from `IndexPlanDataset.cache_metrics()` after the
    # run). `sidecar_adoption_rate = sidecar_groups / (sidecar_groups +
    # full_shard_groups)` is the primary success signal for the sidecar work —
    # wall-clock cannot distinguish "sidecar reached" from "prefetch warmed the
    # shard first". `None` for non-IndexPlan scenarios that have no cache
    # metrics. NOTE: until the gather is routed through `read_rows_with`
    # (Phase 1), IndexPlan scenarios report 0 sidecar groups — the intended
    # pre-fix baseline.
    sidecar_groups: int | None = None
    full_shard_groups: int | None = None
    # F5 (row-group framing): count of `read_rows_with` shard request-groups
    # served by the codec-agnostic block-index path (a framed v2 shard decoded
    # only in its touched groups). The framed-adoption signal, symmetric with
    # `sidecar_groups` / `full_shard_groups`; `> 0` on a scattered read over a
    # framed file proves the row-group path is exercised (the "ship the no-op"
    # guard). `None` for older pyscx that lacks the key.
    block_index_groups: int | None = None
    # Consumer-observed per-batch latency (ms): the wall time between successive
    # batches yielded by `iter_with_plans` (prefetch-overlapped). The sidecar
    # (L1+L2) trades an async full-shard warm for a synchronous O(rows) sidecar
    # decode on the gather thread (§6.4), so per-batch latency — not just
    # aggregate throughput — is the metric that catches a borderline-dense
    # regression. `None` for non-IndexPlan scenarios.
    gather_latency_ms_mean: float | None = None
    gather_latency_ms_p50: float | None = None
    gather_latency_ms_p99: float | None = None
    # Phase-0 gate (R3): the fraction of the timed region the consumer spent
    # blocked in `next(loader)`, against the fixed-cost step `_null_model_ms()`
    # buys. **Only the two fractions are gated on the knob** — a fraction needs
    # a step to be a fraction *of*, so it is `None` when the knob is off (and
    # `None` rather than `0.0`: the gate skips a null metric, while a zero would
    # read as "the loader never stalled", the claim the metric exists to test).
    # `ttfb_s`, the percentiles and `n_steady_steps` are properties of the
    # loader alone, meaningful with or without a consumer step, so they are
    # always emitted.
    data_wait_fraction: float | None = None
    batch_wait_ms_p50: float | None = None
    batch_wait_ms_p95: float | None = None
    batch_wait_ms_p99: float | None = None
    batch_wait_ms_max: float | None = None
    null_model_ms: float | None = None
    # Startup split out — see `steady_state_wait`. The first `next(loader)`
    # also pays DataLoader worker spawn here, which is larger than the tokio
    # spin-up the single-process path pays.
    ttfb_s: float | None = None
    data_wait_fraction_steady: float | None = None
    n_steady_steps: int = 0


def _run_index_plan(
    scx_path: str,
    plans_factory: Callable[[], Iterator[list[tuple[int, int]]]],
    pairs_per_batch: int,
    *,
    hvg_indices: np.ndarray | None,
    normalize: bool,
    sort_by_shard: bool,
    lookahead: int,
    cache_shards: int,
    max_plan_size: int,
) -> _ScenarioOutcome:
    import pyscx

    gc.collect()
    rss0 = _peak_rss_mb()

    ds = pyscx.IndexPlanDataset(
        scx_path,
        hvg_indices=hvg_indices,
        normalize=normalize,
        cache_shards=cache_shards,
        sort_by_shard=sort_by_shard,
        lookahead=lookahead,
        max_plan_size=max_plan_size,
        max_memory_mb=8192,
    )
    # Snapshot the estimator's per-component breakdown right after ctor so
    # the auto-tune output (post-reduction) drives the overshoot delta.
    budget = ds.memory_budget()
    # The six per-component byte keys live under `breakdown` (ORG-9.10-4 gave
    # every class that reports a budget the same envelope). This read is inside
    # the warm-up that `run()` wraps in `except Exception: continue`, so getting
    # it wrong drops both IndexPlan scenarios — including the floored `locality`
    # one — while the job stays green. Confirmed: the old key raises
    # `KeyError: 'total_bytes'` here.
    budget_total_mb = budget["breakdown"]["total_bytes"] / (1024 * 1024)

    t0 = time.perf_counter()
    seen = 0
    cells = 0
    per_batch_s: list[float] = []
    prev = t0
    for batch in ds.iter_with_plans(plans_factory(), lookahead=lookahead):
        now = time.perf_counter()
        per_batch_s.append(now - prev)
        prev = now
        seen += 1
        cells += 2 * batch["X"].shape[0]
    wall = time.perf_counter() - t0
    # Sample loader-level cache metrics (cumulative across the run). The
    # `sidecar_groups` / `full_shard_groups` keys were added alongside the
    # sidecar gather work; tolerate older pyscx that lacks them.
    try:
        cm = ds.cache_metrics()
        sidecar_groups = int(cm.get("sidecar_groups", 0))
        full_shard_groups = int(cm.get("full_shard_groups", 0))
        block_index_groups = int(cm.get("block_index_groups", 0))
    except Exception:
        sidecar_groups = None
        full_shard_groups = None
        block_index_groups = None
    peak_rss_after = _peak_rss_mb()
    peak_rss = max(rss0, peak_rss_after)
    # Scenario-local ru_maxrss growth — eliminates cross-scenario
    # contamination when `run()` executes multiple scenarios in the same
    # process (`ru_maxrss` is monotonic for the process lifetime). Reads as
    # 0 for scenarios that don't push past a prior peak; that's a weaker
    # but honest signal vs. inheriting an unrelated scenario's high-water.
    peak_rss_growth = max(0.0, peak_rss_after - rss0)
    # Per-batch gather latency (ms). Drop the first interval — it includes the
    # initial prefetch fill / lazy tokio-runtime spin-up, not steady-state gather.
    steady = per_batch_s[1:] if len(per_batch_s) > 1 else per_batch_s
    if steady:
        ms = sorted(v * 1000.0 for v in steady)
        mean_ms = sum(ms) / len(ms)
        p50_ms = ms[len(ms) // 2]
        p99_ms = ms[min(len(ms) - 1, int(len(ms) * 0.99))]
    else:
        mean_ms = p50_ms = p99_ms = None
    return _ScenarioOutcome(
        n_batches=seen,
        n_cells=cells,
        wall_s=wall,
        peak_rss_mb=peak_rss,
        memory_budget_total_mb=round(budget_total_mb, 1),
        estimate_overshoot_mb=round(peak_rss_growth - budget_total_mb, 1),
        sidecar_groups=sidecar_groups,
        full_shard_groups=full_shard_groups,
        block_index_groups=block_index_groups,
        gather_latency_ms_mean=round(mean_ms, 3) if mean_ms is not None else None,
        gather_latency_ms_p50=round(p50_ms, 3) if p50_ms is not None else None,
        gather_latency_ms_p99=round(p99_ms, 3) if p99_ms is not None else None,
    )


def _run_backed_python(
    scx_path: str,
    plans_factory: Callable[[], Iterator[list[tuple[int, int]]]],
    pairs_per_batch: int,
    *,
    hvg_indices: np.ndarray | None,
    normalize: bool,
    target_sum: float = 1e4,
) -> _ScenarioOutcome:
    """Manual baseline mirroring the cell-load-scx `ScxBackedSparseDataset`
    path: gather rows via ``adata.X[idx]``, materialize, project, normalize."""
    import pyscx

    gc.collect()
    rss0 = _peak_rss_mb()

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    backed_x = adata.X

    t0 = time.perf_counter()
    seen = 0
    cells = 0
    for plan in plans_factory():
        n = len(plan)
        all_idx = np.empty(2 * n, dtype=np.int64)
        for i, (p, c) in enumerate(plan):
            all_idx[i] = p
            all_idx[n + i] = c
        sub = backed_x[all_idx]
        dense = sub.toarray().astype(np.float32)
        if hvg_indices is not None:
            dense = dense[:, hvg_indices]
        if normalize:
            sums = dense.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            dense = np.log1p(dense * (target_sum / sums))
        seen += 1
        cells += 2 * n
    wall = time.perf_counter() - t0
    return _ScenarioOutcome(
        n_batches=seen,
        n_cells=cells,
        wall_s=wall,
        peak_rss_mb=max(rss0, _peak_rss_mb()),
    )


_HAS_TORCH = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# `num_workers > 0` IndexPlanDataset scenario
# ---------------------------------------------------------------------------
#
# Mirrors the `pyscx_training_dataset_workers2` scenario from
# `ml_loader.py`: wraps `pyscx.IndexPlanDataset` in a lazy-construct
# IterableDataset shim and drives it via `DataLoader(num_workers=2)`.
# The shim shards by plan index (`i % num_workers == worker_id`) so each
# pair is yielded exactly once across workers. Pre-Phase-2 this path
# was unsafe under fork-mode workers (same family of hazards as
# TrainingDataset, sans rayon); post-Phase-2 it runs cleanly — see
# `pyscx/tests/test_fork_safety.py::test_fork_index_plan_dataset`.

if _HAS_TORCH:
    import torch.utils.data as _td_for_ipds_shim

    class _LazyIndexPlanShardedShim(_td_for_ipds_shim.IterableDataset):  # type: ignore[misc]
        """Top-level (picklable under spawn) IterableDataset shim around
        `pyscx.IndexPlanDataset`. Constructs the inner dataset lazily
        inside `__iter__` so the eager fork-detection check at
        `python/index_plan.rs` does not fire."""

        def __init__(
            self,
            scx_path: str,
            n_obs: int,
            pairs_per_batch: int,
            n_batches: int,
            hvg_indices: np.ndarray | None,
            normalize: bool,
            sort_by_shard: bool,
            lookahead: int,
            cache_shards: int,
            max_plan_size: int,
        ) -> None:
            super().__init__()
            self.scx_path = scx_path
            self.n_obs = n_obs
            self.pairs_per_batch = pairs_per_batch
            self.n_batches = n_batches
            self.hvg_indices = hvg_indices
            self.normalize = normalize
            self.sort_by_shard = sort_by_shard
            self.lookahead = lookahead
            self.cache_shards = cache_shards
            self.max_plan_size = max_plan_size

        def __iter__(self):
            import pyscx
            import torch.utils.data as _td

            info = _td.get_worker_info()
            worker_id = info.id if info is not None else 0
            num_workers = info.num_workers if info is not None else 1

            # Each worker generates the same plans (deterministic seed)
            # and emits only the ones whose index matches its worker_id.
            # Direct comparability with `pyscx_index_plan_random` under
            # `num_workers=0`.
            plans = _random_plans(
                self.n_obs, self.pairs_per_batch, self.n_batches, seed=0
            )

            ds = pyscx.IndexPlanDataset(
                self.scx_path,
                hvg_indices=self.hvg_indices,
                normalize=self.normalize,
                cache_shards=self.cache_shards,
                sort_by_shard=self.sort_by_shard,
                lookahead=self.lookahead,
                max_plan_size=self.max_plan_size,
                max_memory_mb=8192,
            )
            for i, batch in enumerate(
                ds.iter_with_plans(plans, lookahead=self.lookahead)
            ):
                if i % num_workers == worker_id:
                    yield batch


def _passthrough_collate(batch):
    """Top-level passthrough collate (lambdas are not picklable under spawn)."""
    return batch


def _run_index_plan_workers2(
    scx_path: str,
    n_obs: int,
    pairs_per_batch: int,
    n_batches: int,
    *,
    hvg_indices: np.ndarray | None,
    normalize: bool,
    sort_by_shard: bool,
    lookahead: int,
    cache_shards: int,
    max_plan_size: int,
    persistent_workers: bool = False,
) -> _ScenarioOutcome:
    """Drive `pyscx.IndexPlanDataset` through `DataLoader(num_workers=2)`
    via the lazy-construct shim. The cell count reported is `2 × pairs ×
    batches_yielded` (matches `_run_index_plan`)."""
    if not _HAS_TORCH:
        return _ScenarioOutcome(
            n_batches=0, n_cells=0, wall_s=0.0, peak_rss_mb=0.0
        )
    import torch.utils.data as _td

    gc.collect()
    rss0 = _peak_rss_mb()

    shim = _LazyIndexPlanShardedShim(
        scx_path=scx_path,
        n_obs=n_obs,
        pairs_per_batch=pairs_per_batch,
        n_batches=n_batches,
        hvg_indices=hvg_indices,
        normalize=normalize,
        sort_by_shard=sort_by_shard,
        lookahead=lookahead,
        cache_shards=cache_shards,
        max_plan_size=max_plan_size,
    )
    loader = _td.DataLoader(
        shim,
        batch_size=None,
        num_workers=2,
        persistent_workers=persistent_workers,
        collate_fn=_passthrough_collate,
    )

    # Explicit iterator, not `for batch in loader:`, so each `next()` can be
    # timed: the R3 half of the phase-0 data-wait gate. `null_ms` is 0 by
    # default, in which case this is the same loop plus two `perf_counter()`
    # calls per batch and the fraction is reported as `None`.
    null_ms = _null_model_ms()
    null_s = null_ms / 1000.0
    waits_s: list[float] = []
    # `t0` BEFORE `iter(loader)`: PyTorch spawns the worker processes while
    # building the iterator, so timing from after it excluded worker spawn from
    # both `wall` and `ttfb_s` while the docs credited it to the first `next()`.
    t0 = time.perf_counter()
    it = iter(loader)
    first_wait_from = t0
    seen = 0
    cells = 0
    wall = 0.0
    while True:
        w0 = first_wait_from if not waits_s else time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        waits_s.append(time.perf_counter() - w0)
        seen += 1
        cells += 2 * batch["X"].shape[0]
        if null_s:
            # The stand-in training step. Sleeping (rather than spinning)
            # releases the GIL so the two worker processes prefetch behind it,
            # which is the overlap a real step gives the loader.
            time.sleep(null_s)
        # Close the timed region after the last SUCCESSFUL step. The `next()`
        # that raises StopIteration also runs `_shutdown_workers()` under
        # `persistent_workers=False`, and that join lands in `wall` but not in
        # `waits_s` — biasing `data_wait_fraction` down by a teardown cost that
        # is a real fraction of a 50-batch region.
        wall = time.perf_counter() - t0
    # Percentiles from `steady` (post-startup slice) — see `steady_state_wait`.
    steady = steady_state_wait(waits_s, wall)
    return _ScenarioOutcome(
        n_batches=seen,
        n_cells=cells,
        wall_s=wall,
        peak_rss_mb=max(rss0, _peak_rss_mb()),
        # Only meaningful against a step: with no null model the consumer is a
        # counter, so the fraction would be ~1.0 by construction and would say
        # nothing about a training run.
        data_wait_fraction=(
            data_wait_fraction(waits_s, wall) if null_ms else None
        ),
        batch_wait_ms_p50=steady["p50_ms"],
        batch_wait_ms_p95=steady["p95_ms"],
        batch_wait_ms_p99=steady["p99_ms"],
        batch_wait_ms_max=steady["max_ms"],
        null_model_ms=null_ms,
        ttfb_s=steady["ttfb_s"],
        data_wait_fraction_steady=(
            steady["data_wait_fraction_steady"] if null_ms else None
        ),
        n_steady_steps=steady["n_steady_steps"],
    )


def _run_training_dataset(
    scx_path: str,
    pairs_per_batch: int,
    n_batches_target: int,
    *,
    hvg_indices: np.ndarray | None,
    normalize: bool,
) -> _ScenarioOutcome:
    """Sequential ceiling — different access pattern, included as the
    throughput upper bound for SCX reads."""
    import pyscx

    gc.collect()
    rss0 = _peak_rss_mb()

    ds = pyscx.TrainingDataset(
        scx_path,
        batch_size=2 * pairs_per_batch,
        hvg_indices=list(hvg_indices) if hvg_indices is not None else None,
        normalize=normalize,
        log1p=normalize,
    )

    t0 = time.perf_counter()
    seen = 0
    cells = 0
    for batch in ds:
        seen += 1
        cells += batch["X"].shape[0]
        if seen >= n_batches_target:
            break
    wall = time.perf_counter() - t0
    return _ScenarioOutcome(
        n_batches=seen,
        n_cells=cells,
        wall_s=wall,
        peak_rss_mb=max(rss0, _peak_rss_mb()),
    )


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def _resolve_hvg(n_vars: int) -> np.ndarray | None:
    """HVG indices for the projected scenarios. Use the canonical
    ``QUERY_N_HVGS`` from the bench config (default 2000); fall back to all
    genes for tiny fixtures."""
    if n_vars <= QUERY_N_HVGS:
        return None
    rng = np.random.default_rng(0)
    hvg = rng.choice(n_vars, size=QUERY_N_HVGS, replace=False).astype(np.uint32)
    hvg.sort()
    return hvg


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """SCX-only IndexPlanDataset throughput benchmark.

    Returns ``None`` for non-``scx_auto`` variants (gating pattern from
    ``fragment_ops`` / ``cell_eval_parity_perf``).

    For every scenario we run ``n_runs`` timed iterations after a single
    untimed warm-up; per-scenario metrics are sparse-emitted into
    ``RunRecord.extra`` so the gate aggregates across each scenario's own
    runs without aliasing.
    """
    if format_variant.key not in SUPPORTED_FORMATS:
        return None
    if not _have_pyscx():
        logger.warning("Skipping index_plan: pyscx not importable")
        return None

    if converted_path is None or not Path(converted_path).exists():
        # Match fragment_ops: fail loudly so the orchestrator surfaces the
        # missing-conversion as a clear error.
        raise FileNotFoundError(
            f"Missing converted SCX file for {dataset.name}. "
            f"Run Phase A conversion first (--formats scx_auto)."
        )
    scx_path = str(converted_path)

    n_obs = dataset.n_obs
    n_vars = dataset.n_vars
    pairs_per_batch = min(_DEFAULT_PAIRS_PER_BATCH, max(1, n_obs // 4))
    # Preserve the harness-requested counts alongside the effective (scaled)
    # ones so the result JSON records whether the >=30k down-scaling applied —
    # otherwise a reader can't tell a natively-small run from a capped one.
    harness_n_runs = n_runs
    n_batches = _n_batches_for(n_obs)
    n_runs = _n_runs_for(n_obs, n_runs)
    hvg = _resolve_hvg(n_vars)

    result = BenchmarkResult(
        benchmark="index_plan",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "pairs_per_batch": pairs_per_batch,
            "n_batches_target": n_batches,
            "n_batches_default": _DEFAULT_N_BATCHES,
            "hvg_size": int(hvg.size) if hvg is not None else None,
            "normalize": True,
            "sort_by_shard": True,
            "lookahead": 4,
            "cache_shards": 128,
            "locality_group_size": _LOCALITY_GROUP_SIZE,
            "n_runs": n_runs,
            "n_runs_harness": harness_n_runs,
            "cold_cache": cold_cache,
            "batch_size_seq_ceiling": 2 * pairs_per_batch,
        },
    )
    result.file_size_bytes = Path(scx_path).stat().st_size

    common = dict(
        hvg_indices=hvg,
        normalize=True,
    )
    common_index_plan = dict(
        sort_by_shard=True,
        lookahead=4,
        cache_shards=128,
        max_plan_size=max(pairs_per_batch, 16384),
        **common,
    )

    # Each scenario takes a batch count `nb` so the untimed warm-up can run a
    # small `_WARMUP_BATCHES` pass instead of a full `n_batches` one (the full
    # warm-up per scenario was a large slice of the timeout on big datasets).
    scenarios: list[tuple[str, Callable[[int], _ScenarioOutcome]]] = [
        (
            "pyscx_index_plan_random",
            lambda nb: _run_index_plan(
                scx_path,
                lambda: _random_plans(n_obs, pairs_per_batch, nb),
                pairs_per_batch,
                **common_index_plan,
            ),
        ),
        (
            "pyscx_index_plan_locality",
            lambda nb: _run_index_plan(
                scx_path,
                lambda: _locality_plans(
                    n_obs, _LOCALITY_GROUP_SIZE, pairs_per_batch, nb
                ),
                pairs_per_batch,
                **common_index_plan,
            ),
        ),
        (
            "pyscx_backed_python_loop",
            lambda nb: _run_backed_python(
                scx_path,
                lambda: _locality_plans(
                    n_obs, _LOCALITY_GROUP_SIZE, pairs_per_batch, nb
                ),
                pairs_per_batch,
                **common,
            ),
        ),
        (
            "pyscx_training_dataset",
            lambda nb: _run_training_dataset(
                scx_path,
                pairs_per_batch,
                n_batches_target=nb,
                **common,
            ),
        ),
    ]

    # `num_workers > 0` IndexPlanDataset scenario. Gated on `_HAS_TORCH`
    # because pyscx ships without torch as a direct dep; only
    # ml-flavoured callers have it installed.
    if _HAS_TORCH:
        scenarios.append(
            (
                "pyscx_index_plan_dataset_workers2",
                lambda nb: _run_index_plan_workers2(
                    scx_path,
                    n_obs=n_obs,
                    pairs_per_batch=pairs_per_batch,
                    n_batches=nb,
                    **common_index_plan,
                ),
            )
        )

    warmup_batches = min(n_batches, _WARMUP_BATCHES)
    for scenario_name, runner in scenarios:
        # Single untimed warm-up per scenario to drive page caches + lazy
        # tokio init — same pattern as ml_loader. Capped to `_WARMUP_BATCHES`
        # (a full warm-up feeds no recorded metric, only caches).
        try:
            logger.info("warmup %s on %s", scenario_name, dataset.name)
            runner(warmup_batches)
        except Exception as e:
            logger.error("warmup failed for %s: %s", scenario_name, e)
            continue

        for i in range(n_runs):
            try:
                if cold_cache:
                    # Drop OS page caches between timed runs. Same approach
                    # ml_loader uses (best-effort; needs root or sysctl).
                    try:
                        os.system("sync && sysctl -q vm.drop_caches=3")
                    except Exception:
                        pass

                outcome = runner(n_batches)
            except Exception as e:
                logger.error("run %d/%d failed for %s: %s", i + 1, n_runs, scenario_name, e)
                continue

            wall = outcome.wall_s
            bps = outcome.n_batches / wall if wall > 0 else 0.0
            cps = outcome.n_cells / wall if wall > 0 else 0.0

            # Sidecar adoption rate = fraction of `read_rows_with` shard
            # request-groups served by the O(rows) decode sidecar vs.
            # full-shard decode. `None` for scenarios with no cache metrics
            # (manual baselines) and when no groups were observed. This is the
            # primary success signal for the sidecar gather work — the L1 gather
            # alone shows no wall-clock change (prefetch warms shards first), so
            # this adoption counter is what proves the sidecar path was reached.
            sc = outcome.sidecar_groups
            fs = outcome.full_shard_groups
            if sc is None or fs is None or (sc + fs) == 0:
                adoption_rate = None
            else:
                adoption_rate = round(sc / (sc + fs), 4)

            run_extra: dict[str, Any] = {
                "scenario": scenario_name,
                "n_batches": outcome.n_batches,
                "n_cells": outcome.n_cells,
                "batches_per_sec": round(bps, 2),
                "cells_per_sec": round(cps, 0),
                f"batches_per_sec__{scenario_name}": round(bps, 2),
                f"cells_per_sec__{scenario_name}": round(cps, 0),
                f"peak_rss_mb__{scenario_name}": round(outcome.peak_rss_mb, 1),
                # Placeholder — BackedCsrReader does not yet expose a hit
                # counter. Slot is preserved so the gate's threshold key
                # remains stable when the counter lands.
                f"shard_cache_hit_rate__{scenario_name}": None,
                # Sidecar adoption — primary success signal for the sidecar
                # gather work. `None` for manual baselines / no observed
                # groups; 0.0 until the gather is routed through
                # `read_rows_with` (Phase 1). Raw counts kept for debugging.
                f"sidecar_adoption_rate__{scenario_name}": adoption_rate,
                f"sidecar_groups__{scenario_name}": outcome.sidecar_groups,
                f"full_shard_groups__{scenario_name}": outcome.full_shard_groups,
                # F5 framed-adoption counter — `> 0` proves a scattered read
                # over a framed (v4) file exercised the row-group block-index
                # path. The Rust hard gate is
                # `scx-format-io backed_tests::read_rows_with_block_index_all_codecs`
                # (asserts the exact count); this surfaces it in the harness too.
                f"block_index_groups__{scenario_name}": outcome.block_index_groups,
                f"gather_latency_ms_p50__{scenario_name}": outcome.gather_latency_ms_p50,
                f"gather_latency_ms_p99__{scenario_name}": outcome.gather_latency_ms_p99,
                # Estimator validation: `None` for scenarios that don't
                # go through `IndexPlanDataset` (no `memory_budget()`
                # accessor on the manual baselines).
                f"memory_budget_total_mb__{scenario_name}": (
                    outcome.memory_budget_total_mb
                ),
                f"estimate_overshoot_mb__{scenario_name}": (
                    outcome.estimate_overshoot_mb
                ),
                # Phase-0 gate (R3). `None` on every scenario that does not
                # drive a consumer against a step — which is all of them
                # unless `SCX_BENCH_R3_NULL_MODEL_MS` is set. The slot is
                # emitted uniformly so the key stays stable, exactly as
                # `shard_cache_hit_rate__*` above.
                f"data_wait_fraction__{scenario_name}": outcome.data_wait_fraction,
                f"batch_wait_ms_p50__{scenario_name}": outcome.batch_wait_ms_p50,
                f"batch_wait_ms_p95__{scenario_name}": outcome.batch_wait_ms_p95,
                f"batch_wait_ms_p99__{scenario_name}": outcome.batch_wait_ms_p99,
                f"batch_wait_ms_max__{scenario_name}": outcome.batch_wait_ms_max,
                f"null_model_ms__{scenario_name}": outcome.null_model_ms,
                # The headline `p` for R3 — startup excluded.
                f"data_wait_fraction_steady__{scenario_name}": (
                    outcome.data_wait_fraction_steady
                ),
                f"ttfb_s__{scenario_name}": outcome.ttfb_s,
                f"n_steady_steps__{scenario_name}": outcome.n_steady_steps,
            }
            result.add_run(
                wall_s=wall,
                peak_rss_mb=outcome.peak_rss_mb,
                **run_extra,
            )

    return result
