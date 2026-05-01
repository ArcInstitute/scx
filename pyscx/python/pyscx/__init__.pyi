"""Type stubs for the parts of `pyscx` that benefit most from static typing.

This file is intentionally narrow — currently it stubs `IndexPlanDataset`
and its iterator, since those are the most error-prone callers will run
into (kwarg-heavy constructor, plan iterator protocol, batch dict schema).

Other pyscx symbols re-exported via `from .pyscx import *` are typed as
`Any` to type-checkers until / unless someone needs more coverage.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any, Tuple, TypedDict

import numpy as np


# ---------------------------------------------------------------------------
# IndexPlanDataset (PER-CELL-CONTROL-PAIRING.md)
# ---------------------------------------------------------------------------


class _CategoricalObs(TypedDict):
    codes: np.ndarray  # int32
    categories: list[str]


class _IndexPlanBatchDict(TypedDict):
    X: np.ndarray  # [B, n_output_genes] float32 — perturbed rows
    X_paired: np.ndarray  # [B, n_output_genes] float32 — control rows
    pairs: list[tuple[int, int]]
    obs: dict[str, np.ndarray | _CategoricalObs]
    obs_paired: dict[str, np.ndarray | _CategoricalObs]


class IndexPlanBatchIter:
    """Iterator returned by `IndexPlanDataset.iter_with_plans`.

    Yields paired batch dicts (`_IndexPlanBatchDict`-shaped) and raises
    `StopIteration` when the plan stream ends.
    """

    def __iter__(self) -> "IndexPlanBatchIter": ...
    def __next__(self) -> _IndexPlanBatchDict: ...
    def __repr__(self) -> str: ...


class IndexPlanDataset:
    """Plan-driven paired-batch reader for ML training.

    See `docs/api.md` § `IndexPlanDataset` for a full description; the
    short version: each yielded batch is a list of
    `(perturbed_cell, control_cell)` pairs gathered via the cached
    `BackedCsrReader`, with optional HVG projection + normalize+log1p.

    Sibling to `TrainingDataset`. Use `TrainingDataset` for sequential
    catalog-order streaming; use `IndexPlanDataset` when the consumer
    needs to drive batch composition.
    """

    def __init__(
        self,
        path: str,
        *,
        hvg_indices: np.ndarray | None = None,
        obs_columns: list[str] | None = None,
        normalize: bool | None = None,
        log1p: bool | None = None,
        target_sum: float | None = None,
        cache_shards: int | None = None,
        sort_by_shard: bool | None = None,
        lookahead: int | None = None,
        max_plan_size: int | None = None,
        max_memory_mb: int | None = None,
    ) -> None: ...

    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def n_output_genes(self) -> int: ...

    def effective_cache_shards(self) -> int:
        """Resolved `cache_shards` after memory-budget auto-tuning. May be
        less than the user-requested value."""
        ...

    def effective_lookahead(self) -> int:
        """Resolved default lookahead after memory-budget auto-tuning. Used
        by `iter_with_plans` when the caller passes `lookahead=None`."""
        ...

    def iter_with_plans(
        self,
        plans: Iterable[list[tuple[int, int]]] | Iterator[list[tuple[int, int]]],
        lookahead: int | None = None,
    ) -> IndexPlanBatchIter:
        """Drive the loader from a Python iterable of plans (each plan a
        list of `(perturbed_idx, control_idx)` tuples). Returns an
        iterator of paired batch dicts.

        ``lookahead=None`` (default) uses ``effective_lookahead()``;
        ``lookahead=0`` disables shard prefetching; larger values trade
        RAM for I/O hiding.
        """
        ...

    def _next_batch_for_test(
        self, plan: list[tuple[int, int]]
    ) -> _IndexPlanBatchDict:
        """Unstable / debug-only single-plan entry point. Use
        `iter_with_plans` in production."""
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Other pyscx symbols re-exported via `from .pyscx import *` are typed as
# `Any` here. Add explicit stubs above if/when type-checking on those
# symbols becomes load-bearing.
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any: ...
