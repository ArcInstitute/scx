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
# IndexPlanDataset
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
# Experiment — the core open-file handle (T3.4 metadata accessors + repr).
# ---------------------------------------------------------------------------


class Experiment:
    """Handle to an open SCX file. Returned by `pyscx.open(path)`."""

    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def shard_count(self) -> int: ...
    @property
    def format_version(self) -> int: ...
    @property
    def codec_id(self) -> int: ...
    @property
    def path(self) -> str: ...
    @property
    def has_csc(self) -> bool: ...
    @property
    def has_deletions(self) -> bool: ...
    @property
    def layer_names(self) -> list[str]: ...
    @property
    def obs_keys(self) -> list[str]: ...
    @property
    def var_keys(self) -> list[str]: ...
    @property
    def obsm_keys(self) -> list[str]: ...
    @property
    def varm_keys(self) -> list[str]: ...
    @property
    def uns_keys(self) -> list[str]: ...

    def info(self) -> str:
        """One-line codec / shard / format-version internals."""
        ...

    def to_anndata(
        self,
        backed: bool = ...,
        cache_shards: int = ...,
        var_names: list[str] | None = ...,
        obs_filter: str | None = ...,
        layers: list[str] | None = ...,
        preserve_slots: bool = ...,
        modality: str | None = ...,
        eager: bool = ...,
        memory_budget: Any = ...,
        obsm: list[str] | None = ...,
    ) -> Any: ...

    def query(self) -> Any: ...
    def provenance(self) -> list[dict[str, Any]]: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Module-level entry points.
# ---------------------------------------------------------------------------


def open(path: Any, verify: bool = ...) -> Experiment:  # noqa: A001
    """Open an SCX file as an `Experiment`."""
    ...


def read(path: Any, *, verify: bool = ..., **kwargs: Any) -> Any:
    """Read an SCX file into an AnnData (= `open(path).to_anndata(**kwargs)`)."""
    ...


def write(adata: Any, path: Any, **kwargs: Any) -> None:
    """Write an AnnData to an SCX file (= `from_anndata(adata, path, **kwargs)`)."""
    ...


def validate(path: Any) -> list[tuple[str, bool]]: ...


# ---------------------------------------------------------------------------
# Cloud surface (present only when built with `--features cloud`).
# ---------------------------------------------------------------------------


class CloudExperiment:
    """Cloud-hosted SCX handle. Returned by `pyscx.open_cloud(url)`."""

    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def nnz(self) -> int: ...
    @property
    def shard_count(self) -> int: ...
    @property
    def format_version(self) -> int: ...
    @property
    def codec_id(self) -> int: ...

    def query(self) -> Any: ...
    def __repr__(self) -> str: ...


def open_cloud(url: str) -> CloudExperiment: ...


def read_cloud(
    url: str,
    *,
    obs_filter: str | None = ...,
    var_names: list[str] | None = ...,
) -> Any:
    """One-call cloud read into an AnnData (= `open_cloud(url).query()…
    collect().to_anndata()`)."""
    ...


# ---------------------------------------------------------------------------
# Other pyscx symbols re-exported via `from .pyscx import *` are typed as
# `Any` here. Add explicit stubs above if/when type-checking on those
# symbols becomes load-bearing.
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any: ...
