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
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)


# Only triggered for scx_auto. We don't re-measure across codecs since the
# IndexPlanDataset path is codec-agnostic at the API level.
_SCX_TRIGGER_KEY = "scx_auto"

# Defaults: 1024 pairs/batch, 1000 batches. Capped against dataset.n_obs
# at runtime so a 100-cell test fixture doesn't blow up.
_DEFAULT_PAIRS_PER_BATCH = 1024
_DEFAULT_N_BATCHES = 1000

# Locality grouping fallback when no categorical obs column is suitable.
# Cells with the same `index // _LOCALITY_GROUP_SIZE` form a group; pert and
# ctrl are sampled from the same group. A group of 4096 corresponds to roughly
# one shard of the typical 16k-shard fixture, giving high cache reuse.
_LOCALITY_GROUP_SIZE = 4096


def _have_pyscx() -> bool:
    try:
        import pyscx  # noqa: F401

        return True
    except ImportError:
        return False


def _peak_rss_mb() -> float:
    """High-water-mark RSS via ``ru_maxrss``."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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

    t0 = time.perf_counter()
    seen = 0
    cells = 0
    for batch in ds.iter_with_plans(plans_factory(), lookahead=lookahead):
        seen += 1
        cells += 2 * batch["X"].shape[0]
    wall = time.perf_counter() - t0
    return _ScenarioOutcome(
        n_batches=seen,
        n_cells=cells,
        wall_s=wall,
        peak_rss_mb=max(rss0, _peak_rss_mb()),
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
        `python.rs:416-421` does not fire."""

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

    t0 = time.perf_counter()
    seen = 0
    cells = 0
    for batch in loader:
        seen += 1
        cells += 2 * batch["X"].shape[0]
    wall = time.perf_counter() - t0
    return _ScenarioOutcome(
        n_batches=seen,
        n_cells=cells,
        wall_s=wall,
        peak_rss_mb=max(rss0, _peak_rss_mb()),
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
    if format_variant.key != _SCX_TRIGGER_KEY:
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
    n_batches = _DEFAULT_N_BATCHES
    hvg = _resolve_hvg(n_vars)

    result = BenchmarkResult(
        benchmark="index_plan",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "pairs_per_batch": pairs_per_batch,
            "n_batches_target": n_batches,
            "hvg_size": int(hvg.size) if hvg is not None else None,
            "normalize": True,
            "sort_by_shard": True,
            "lookahead": 4,
            "cache_shards": 128,
            "locality_group_size": _LOCALITY_GROUP_SIZE,
            "n_runs": n_runs,
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

    scenarios: list[tuple[str, Callable[[], _ScenarioOutcome]]] = [
        (
            "pyscx_index_plan_random",
            lambda: _run_index_plan(
                scx_path,
                lambda: _random_plans(n_obs, pairs_per_batch, n_batches),
                pairs_per_batch,
                **common_index_plan,
            ),
        ),
        (
            "pyscx_index_plan_locality",
            lambda: _run_index_plan(
                scx_path,
                lambda: _locality_plans(
                    n_obs, _LOCALITY_GROUP_SIZE, pairs_per_batch, n_batches
                ),
                pairs_per_batch,
                **common_index_plan,
            ),
        ),
        (
            "pyscx_backed_python_loop",
            lambda: _run_backed_python(
                scx_path,
                lambda: _locality_plans(
                    n_obs, _LOCALITY_GROUP_SIZE, pairs_per_batch, n_batches
                ),
                pairs_per_batch,
                **common,
            ),
        ),
        (
            "pyscx_training_dataset",
            lambda: _run_training_dataset(
                scx_path,
                pairs_per_batch,
                n_batches_target=n_batches,
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
                lambda: _run_index_plan_workers2(
                    scx_path,
                    n_obs=n_obs,
                    pairs_per_batch=pairs_per_batch,
                    n_batches=n_batches,
                    **common_index_plan,
                ),
            )
        )

    for scenario_name, runner in scenarios:
        # Single untimed warm-up per scenario to drive page caches + lazy
        # tokio init — same pattern as ml_loader.
        try:
            logger.info("warmup %s on %s", scenario_name, dataset.name)
            runner()
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

                outcome = runner()
            except Exception as e:
                logger.error("run %d/%d failed for %s: %s", i + 1, n_runs, scenario_name, e)
                continue

            wall = outcome.wall_s
            bps = outcome.n_batches / wall if wall > 0 else 0.0
            cps = outcome.n_cells / wall if wall > 0 else 0.0

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
            }
            result.add_run(
                wall_s=wall,
                peak_rss_mb=outcome.peak_rss_mb,
                **run_extra,
            )

    return result
