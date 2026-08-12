"""Type stubs for the parts of `pyscx` that benefit most from static typing.

This file is intentionally narrow — currently it stubs `TrainingDataset`,
`IndexPlanDataset`, and its iterator, since those are the most error-prone
callers will run into (kwarg-heavy constructor, plan iterator protocol,
batch dict schema).

Other pyscx symbols re-exported via `from .pyscx import *` are typed as
`Any` to type-checkers until / unless someone needs more coverage.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, Tuple, TypedDict

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

    While draining, the shard-cache counters are sampled periodically; a
    `UserWarning` is emitted **once** if they indicate the working set exceeds
    `cache_shards` (see `IndexPlanDataset.suggested_cache_shards`).
    """

    def __iter__(self) -> "IndexPlanBatchIter": ...
    def __next__(self) -> _IndexPlanBatchDict: ...
    def __repr__(self) -> str: ...

    def metrics(self) -> dict[str, dict[str, int]]:
        """Cache- and prefetch-side counters as
        ``{"cache": {...}, "prefetch": {...}}``. ``cache`` is
        loader-cumulative (shared with `IndexPlanDataset.cache_metrics`);
        ``prefetch`` is per-iter. Safe after exhaustion."""
        ...


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
        hvg_indices: Sequence[int] | np.ndarray | None = None,
        obs_columns: list[str] | None = None,
        normalize: bool | None = None,
        log1p: bool | None = None,
        target_sum: float | None = None,
        pflog: bool | None = None,
        pflog_alpha: float | None = None,
        cache_shards: int | None = None,
        sort_by_shard: bool | None = None,
        lookahead: int | None = None,
        max_plan_size: int | None = None,
        max_memory_mb: int | None = None,
        scatter_block_index: bool | None = None,
    ) -> None: ...

    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def n_output_genes(self) -> int: ...
    @property
    def closed(self) -> bool:
        """True once `close()` has run. Never raises."""
        ...

    def close(self) -> None:
        """Release the tokio runtime, GIL detached and bounded by 5 s.

        Idempotent, and **terminal** — unlike `TrainingDataset.close()`, which
        rebuilds on the next `__iter__`. This class's runtime is built exactly
        once so a forked child can never inherit it, so it cannot be rebuilt;
        every other method raises `RuntimeError` afterwards. `closed` and
        `__repr__` keep working.
        """
        ...

    def effective_cache_shards(self) -> int:
        """Resolved `cache_shards` after memory-budget auto-tuning. May be
        less than the user-requested value — a reduction emits a
        `UserWarning` at construction naming the budget that would hold the
        request."""
        ...

    def effective_lookahead(self) -> int:
        """Resolved default lookahead after memory-budget auto-tuning. Used
        by `iter_with_plans` when the caller passes `lookahead=None`."""
        ...

    def cache_metrics(self) -> dict[str, int]:
        """Cumulative shard-cache counters since construction: `hits`,
        `misses`, `evictions`, `bytes_inserted`, `duplicate_waiters`,
        `peak_bytes_in_cache`, `full_shard_groups`, `block_index_groups`."""
        ...

    def memory_budget(self) -> dict[str, int]:
        """Per-component budget breakdown from the construction auto-tune,
        plus `max_memory_mb` (the resolved value in force — adaptive when the
        constructor was passed none), `effective_cache_shards`, and
        `effective_lookahead`."""
        ...

    def suggested_cache_shards(self, plan: list[tuple[int, int]]) -> int:
        """Distinct CSR shards `plan` touches — the `cache_shards` that would
        let the whole plan stay resident for one gather.

        Pure index arithmetic from the catalog's shard row ranges: no I/O, no
        decode. Size the cache from the plans you will issue instead of
        guessing::

            probe = pyscx.IndexPlanDataset(path)
            need = max(probe.suggested_cache_shards(p) for p in plans[:64])
            ds = pyscx.IndexPlanDataset(path, cache_shards=need)
        """
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


class _SparseCellSetBatchDict(TypedDict):
    indptr: np.ndarray  # [total_rows + 1] int64 — CSR row offsets
    indices: np.ndarray  # [nnz] int32 — gene ids (raw-local, or global if remapped)
    data: np.ndarray  # [nnz] float32 — values (raw counts by default)
    shape: tuple[int, int]  # (total_rows, n_cols)
    cell_indices: np.ndarray  # [total_rows] uint64 — source row id, request order
    file_ids: np.ndarray  # [total_rows] uint32 — source file_id per row
    set_offsets: np.ndarray  # [n_sets + 1] int64 — row ranges delimiting each set
    role_tags: np.ndarray  # [total_rows] int32 — per-row role tag


class SparseCellSetBatchIter:
    """Iterator returned by `SparseCellSetDataset.iter_with_plans`.

    Yields sparse batch dicts (`_SparseCellSetBatchDict`-shaped) and raises
    `StopIteration` when the plan stream ends.

    While draining, the shard-cache counters are sampled periodically; a
    `UserWarning` is emitted **once** if they indicate the working set exceeds
    `cache_shards` (see `SparseCellSetDataset.suggested_cache_shards`).
    """

    def __iter__(self) -> "SparseCellSetBatchIter": ...
    def __next__(self) -> _SparseCellSetBatchDict: ...
    def __repr__(self) -> str: ...

    def cache_metrics(self) -> dict[str, int]:
        """Shared shard-cache counters — same keys as
        `SparseCellSetDataset.cache_metrics`. Safe after exhaustion."""
        ...


class SparseCellSetDataset:
    """Native sparse cell-set reader (SCX-DATA-LOADER §4).

    Multi-file: gathers role-tagged, variable-size cell-set *batches* as
    sparse CSR through the shared prefetch engine, emitting the §4.4 batch
    contract. Each plan item is one batch of cell sets, passed as a tuple
    ``(file_ids, rows, role_tags, set_offsets)`` of arrays.

    Output is raw-local CSR by default (the caller remaps to its global gene
    vocab); pass per-file ``remap_tables`` to emit global-vocab CSR. Sibling
    to `IndexPlanDataset`, but sparse (not dense pairs) and multi-file.
    """

    def __init__(
        self,
        paths: list[str],
        *,
        cache_shards: int | None = None,
        max_memory_mb: int | None = None,
        lookahead: int | None = None,
        remap_tables: list[list[int]] | None = None,
        n_global_genes: int | None = None,
        normalize: bool | None = None,
        log1p: bool | None = None,
        target_sum: float | None = None,
        downsample_target_library_size: int | None = None,
        downsample_method: Literal["binomial", "multinomial"] | None = None,
        downsample_seed: int | None = None,
    ) -> None:
        """``downsample_target_library_size`` enables a seeded per-row count
        downsample applied **inside the gather**, before the batch is returned —
        which is what keeps a caller's query sampling (drawn from ``counts > 0``)
        consistent with the counts the collate kernel then sees.

        ``downsample_method`` is ``"multinomial"`` (default) or ``"binomial"``;
        ``downsample_seed`` defaults to ``0``. The draw is keyed on
        ``(seed, method, resolved path, row)``, so it is invariant to manifest
        order and to scheduling, and reproducible across processes. Passing a
        method or seed without a target is an error, not a silent no-op.

        Enabling this rounds every cell's counts to integers (ties-to-even), even
        cells already below target."""
        ...

    @property
    def n_files(self) -> int: ...
    @property
    def n_cols(self) -> int: ...
    @property
    def closed(self) -> bool:
        """True once `close()` has run. Never raises."""
        ...

    def close(self) -> None:
        """Release the prefetch engine's tokio runtime, GIL detached and
        bounded by 5 s. Idempotent and **terminal** — see
        `IndexPlanDataset.close`."""
        ...

    def iter_with_plans(
        self,
        plans: Iterable[tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]],
        lookahead: int | None = None,
    ) -> SparseCellSetBatchIter:
        """Drive the loader from a Python iterable of batch plans, each a
        tuple ``(file_ids: u32[], rows: u64[], role_tags: i32[],
        set_offsets: i64[])``. Returns an iterator of sparse batch dicts.

        Per batch: ``file_ids``, ``rows``, and ``role_tags`` are parallel arrays
        of length ``total_rows`` (one entry per cell). ``set_offsets`` has length
        ``n_sets + 1`` and delimits each cell set as ``rows[set_offsets[s] :
        set_offsets[s + 1]]``; on the common path every row of a set shares a
        ``file_id``. Each ``file_id`` indexes into the constructor ``paths``.

        Malformed plans raise rather than crash the worker: a ``file_id`` ≥
        ``n_files`` or a non-monotonic / out-of-bounds ``set_offsets`` raises
        ``RuntimeError``; an out-of-range ``row`` raises ``IndexError``.

        ``lookahead=None`` uses the constructor default; ``0`` disables shard
        prefetching; larger values trade RAM for I/O hiding.
        """
        ...

    def cache_metrics(self) -> dict[str, Any]:
        """Cumulative shard-cache counters since construction (hits, misses,
        evictions, bytes, peak) — the multi-file sibling of
        `IndexPlanDataset.cache_metrics`. For runtime hit/miss/eviction
        observability."""
        ...

    def memory_budget(self) -> dict[str, int]:
        """Resolved shard-cache budget: `max_memory_mb` (the value in force —
        adaptive when the constructor was passed none), `cache_shards`,
        `affordable_cache_shards`, `shard_decoded_bytes`.

        Unlike `IndexPlanDataset.memory_budget` there is no batch-buffer term:
        on the sparse path the shard cache *is* the budget."""
        ...

    def suggested_cache_shards(
        self, file_ids: Sequence[int], rows: Sequence[int]
    ) -> int:
        """Distinct ``(file_id, shard)`` pairs a plan touches — the
        `cache_shards` that would let the whole batch stay resident.

        `file_ids` and `rows` are the first two elements of the plan tuple
        `iter_with_plans` consumes. Pure index arithmetic: no I/O, no decode::

            probe = pyscx.SparseCellSetDataset(paths)
            need = max(probe.suggested_cache_shards(f, r) for f, r, _, _ in plans[:64])
            ds = pyscx.SparseCellSetDataset(paths, cache_shards=need)
        """
        ...

    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# TrainingDataset
# ---------------------------------------------------------------------------


class _TrainingBatchDict(TypedDict):
    X: np.ndarray  # [B, n_output_genes] float32 — dense expression
    obs: dict[str, np.ndarray | _CategoricalObs]
    cell_indices: np.ndarray  # [B] int64 — global row indices


class TrainingDataset:
    """High-throughput sequential streaming dataset for ML training.

    Each ``for batch in dataset:`` loop is one epoch; shards are reshuffled
    between epochs. Sibling to ``IndexPlanDataset`` (which drives batch
    composition from a plan iterator).

    IMPORTANT — transforms are ON by default: ``normalize`` and ``log1p``
    both default to ``True``, so batches are total-count normalized
    (``target_sum=1e4``) and ``log1p``-transformed even though the file
    stores raw counts. The yielded ``X`` is log-normalized, **not** raw
    counts. Count-likelihood models (scVI, scANVI, count autoencoders) need
    raw counts — pass ``normalize=False, log1p=False``.
    """

    def __init__(
        self,
        path: str,
        batch_size: int | None = None,
        hvg_indices: Sequence[int] | np.ndarray | None = None,
        obs_columns: list[str] | None = None,
        normalize: bool | None = None,  # default True — see class docstring
        log1p: bool | None = None,  # default True — see class docstring
        target_sum: float | None = None,
        pflog: bool | None = None,
        pflog_alpha: float | None = None,
        shard_group_size: int | None = None,
        prefetch_batches: int | None = None,
        seed: int | None = None,
        max_memory_mb: int | None = None,
        modality: str | None = None,
    ) -> None: ...

    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def n_output_genes(self) -> int: ...
    @property
    def effective_batch_size(self) -> int: ...

    def __iter__(self) -> "TrainingDataset": ...
    def __next__(self) -> _TrainingBatchDict: ...
    def memory_budget(self) -> dict[str, Any]: ...
    def close(self) -> None: ...
    def __repr__(self) -> str: ...


# ---------------------------------------------------------------------------
# Experiment — the core open-file handle (T3.4 metadata accessors + repr).
# ---------------------------------------------------------------------------


class GroupShard:
    """One shard of a grouped (sorted) SCX layout. Yielded by
    `Experiment.iter_group_shards`."""

    @property
    def shard_index(self) -> int: ...
    @property
    def global_start(self) -> int: ...
    @property
    def global_stop(self) -> int: ...
    @property
    def labels(self) -> list[str]: ...
    @property
    def groups(self) -> dict[str, tuple[int, int]]: ...
    def to_anndata(self) -> Any: ...
    def read_group(self, label: str) -> Any: ...
    def __repr__(self) -> str: ...


class Experiment:
    """Handle to an open SCX file. Returned by `pyscx.open(path)`."""

    @property
    def n_obs(self) -> int: ...
    @property
    def n_obs_physical(self) -> int: ...
    @property
    def n_vars(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
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
    def obs_metadata_shard_count(self) -> int: ...
    @property
    def var_metadata_shard_count(self) -> int: ...
    @property
    def index_dtype(self) -> int: ...
    @property
    def is_multimodal(self) -> bool: ...
    @property
    def n_modalities(self) -> int: ...
    @property
    def modality_names(self) -> list[str]: ...
    # List-returning accessors — callable methods (not properties): the
    # AnnData-style `*_keys()` family plus `layer_names()` (F1/F7).
    def layer_names(self) -> list[str]: ...
    def obs_keys(self) -> list[str]: ...
    def var_keys(self) -> list[str]: ...
    def obsm_keys(self) -> list[str]: ...
    def varm_keys(self) -> list[str]: ...
    def uns_keys(self, modality: str | None = ...) -> list[str]: ...
    def read_uns(self, modality: str | None = ...) -> dict | None:
        """Full `uns` dict with tagged envelopes reconstructed.

        Reads only the small `uns` JSON section — does not materialize
        obs, var, obsm, or X. Returns ``None`` when the file has no
        `uns` section. Safe on multi-hundred-GB atlases where
        ``to_anndata()`` would OOM.

        On multimodal files, pass ``modality=<name>`` to read that
        modality's ``uns/<name>`` section instead of the global one.
        Unknown modality names raise ``KeyError``.
        """
        ...

    def read_obs(self, columns: list[str] | None = ...) -> Any:
        """Read obs (optionally a column projection) as a pandas DataFrame,
        without materializing X. `columns=None` reads the full obs table."""
        ...

    def read_var(
        self, columns: list[str] | None = ..., *, modality: str | None = ...
    ) -> Any:
        """Read var as a pandas DataFrame, without materializing X.

        The var-axis mirror of `read_obs`. Unlike `read_obs`, `columns` is a
        convenience projection applied after the decode rather than pushed
        into the reader: `var` is sized by `n_vars` (a few MB even on an
        atlas), so there is nothing to save at the I/O layer. The pandas index
        column (gene names) is always retained; an unknown column raises
        `KeyError` naming what is available.

        On a multimodal file pass `modality=<name>` for that modality's var;
        omitting it reads the global/single-modality var. Unknown name →
        `KeyError`."""
        ...

    def distinct_values(
        self, col: str, *, limit: int | None = ..., sort: bool = ...
    ) -> tuple[list[str], bool]:
        """Distinct values of a string/categorical obs column as
        `(values, has_more)`. Never decodes X."""
        ...

    def obs_categorical(self, col: str) -> tuple[np.ndarray, list[str]]:
        """`(codes, categories)` for a string/categorical obs column.

        `codes` is `int32`, one entry per obs row, `-1` for missing
        (`pandas.Categorical.codes` convention); `categories[code]` is the
        string value. Computed shard-by-shard and returned as numpy directly —
        no Arrow-IPC round-trip, no pandas frame, X never touched.

        Category order is **first-seen**, not lexicographic. Unreferenced
        dictionary levels are retained (as pandas retains unused levels).
        Raises `ValueError` for non-string columns.

        Physical row space: `len(codes) == n_obs_physical`, not `n_obs`. On a
        file with deletion vectors those differ and indexing by a logical row id
        addresses the wrong cell (`read_obs` has the same contract).
        """
        ...

    def obs_categorical_many(
        self, cols: list[str]
    ) -> list[tuple[np.ndarray, list[str]]]:
        """`obs_categorical` for several columns in one shard pass.

        Returns `(codes, categories)` per column in `cols` order. N columns
        cost one projected read per obs shard instead of N."""
        ...

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
        preserve_var_order: bool = ...,
        strict_var_names: bool = ...,
        container: str = ...,
        data_dtype: str | None = ...,
        index_dtype: str | None = ...,
        allow_lossy: bool = ...,
    ) -> Any: ...

    def to_mudata(
        self,
        backed: bool = ...,
        cache_shards: int = ...,
        container: str | dict[str, str] | None = ...,
        data_dtype: str | dict[str, str] | None = ...,
        index_dtype: str | dict[str, str] | None = ...,
        allow_lossy: bool = ...,
    ) -> Any:
        """Materialise a multimodal SCX file as a ``mudata.MuData``.

        ``container`` / ``data_dtype`` / ``index_dtype`` each accept either a
        scalar (applied to every modality) or a dict keyed by modality name
        (e.g. ``data_dtype={"rna": "uint16", "atac": "uint8"}``); a modality
        with no override keeps the byte-identical zero-copy ``f32`` CSR. A dict
        key naming no modality raises ``ValueError``. ``container="dense"`` is
        not yet supported (CSR only). The narrow kwargs require ``backed=False``.
        """
        ...

    def query(self, modality: str | None = None) -> Any:
        """Start a lazy query pipeline (``.filter_obs()`` / ``.select_genes()`` /
        ``.collect()``).

        ``modality`` scopes the query to one modality of a multimodal file
        (X / ``select_genes`` / ``filter_var`` resolve against that modality's
        var; ``filter_obs`` stays on the shared global obs axis). On a
        multimodal file ``modality`` is required (omitting raises ``ValueError``);
        an unknown name raises ``KeyError``. Omit it on single-modality files.
        """
        ...

    def provenance(self) -> list[dict[str, Any]]: ...

    def detection_counts(self, axis: str = "var", modality: str | None = None) -> Any:
        """Per-gene detection counts for ALL genes (numpy int64, length n_vars).

        The first argument is ``axis`` (only ``"var"`` is supported), NOT a
        gene list — this returns the whole per-gene array. For the cells
        expressing one specific gene, use ``cells_expressing(gene)``.
        """
        ...

    def cells_expressing(self, gene: int | str, modality: str | None = None) -> Any:
        """Global row indices of cells expressing ``gene`` (numpy uint32).

        ``gene`` is an integer index or a var name. Companion to
        ``detection_counts`` (which returns the per-gene array for all genes).
        """
        ...

    def gather_rows_sparse(
        self,
        rows: Any,
        modality: str | None = None,
        cache_shards: int = 4,
    ) -> Any:
        """Gather ``rows`` as a ``scipy.sparse.csr_matrix`` in request order.

        Synchronous sparse gather over the backed reader: each touched shard is
        decoded once; no intermediate ``ScxCsr`` is allocated (zero-copy only at
        the numpy handoff). ``rows`` (numpy ``uint64``) may contain duplicates
        and need not be sorted. Returns raw-local gene indices (no global-vocab
        remap). ``cache_shards`` bounds peak decoded-shard memory for the gather
        (not a speedup knob — each shard is decoded once per call). A fresh
        reader is opened per call (fork-safe; this is the eval / random-access
        utility, not the training hot path). Out-of-range ids raise
        ``IndexError``. Drop-in for the backed ``adata.X[rows]`` analysis path.
        """
        ...

    def to_gpu_anndata(
        self,
        var_names: list[str] | None = ...,
        obs_filter: str | None = ...,
        layers: list[str] | None = ...,
        obsm: list[str] | None = ...,
        device: str = ...,
        memory_budget: Any = ...,
        preserve_var_order: bool = ...,
        strict_var_names: bool = ...,
        container: str = ...,
        data_dtype: str | None = ...,
        index_dtype: str | None = ...,
        allow_lossy: bool = ...,
    ) -> Any:
        """Materialise on-device as an AnnData backed by a
        ``cupyx.scipy.sparse.csr_matrix`` (f32-native). Rejects the
        dtype/container narrowing kwargs at runtime."""
        ...

    def mark_deleted(self, mask: np.ndarray) -> int:
        """Mark cells for deletion via a boolean mask (length ``n_obs``);
        returns the number of rows newly marked. Writes a deletion vector."""
        ...

    def validate(self, deep: bool = ...) -> list[tuple[str, bool]]:
        """Run structural checks, returning ``(check_name, passed)`` pairs.
        ``deep=True`` additionally re-verifies per-shard checksums."""
        ...

    # --- Grouped (sorted) layout accessors ---
    def read_group(self, label: str) -> Any:
        """Read all cells of ``label`` (grouped layout) as an AnnData."""
        ...

    def read_reference(self) -> Any | None:
        """Read the reference group as an AnnData, or ``None`` if absent."""
        ...

    def group_labels(self) -> list[str]: ...
    def iter_group_shards(self) -> list[GroupShard]: ...

    # --- Multimodal accessors ---
    def modality_id(self, name: str) -> int | None:
        """Resolve a modality name to its id, or ``None`` if unknown."""
        ...

    def modality_info(self, modality_id: int) -> dict[str, Any] | None:
        """Per-modality metadata dict (name, n_vars, nnz, shard counts, …),
        or ``None`` if the id is unknown."""
        ...

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


def validate(path: Any, deep: bool = ...) -> list[tuple[str, bool]]: ...


def build_csc(
    input: Any,
    output: Any | None = ...,
    memory_limit: str = ...,
    force: bool = ...,
    csc_cols_per_shard: int = ...,
) -> None:
    """Add a CSC (column-major) sidecar built from `input`'s CSR shards.

    ``output=None`` (the default) rebuilds `input` **in place** via temp file +
    atomic rename; pass a path to write a copy instead. ``force`` applies only
    to the copy-out form and is rejected with ``output=None``.
    """
    ...


def mark_deleted(path: str, cell_indices: Sequence[int]) -> int:
    """Mark specific global cell indices as logically deleted, returning the
    total number of deleted cells (including any previously deleted). Deletions
    apply to the whole cell across every modality; reclaim the rows with
    `compact`."""
    ...


def optimize(
    input: str, output: str, codec: str = ..., shard_obs: str = ...
) -> None: ...


def sort(
    input: str,
    output: str,
    by: Sequence[str],
    reverse: bool = ...,
    shard_size: int | None = ...,
    codec: str = ...,
    index_obs: Sequence[str] | None = ...,
    index_var: Sequence[str] | None = ...,
    index_preset: str | None = ...,
    index_auto_threshold: int | None = ...,
    memory_budget: str | None = ...,
    temp_dir: str | None = ...,
    bitmap: str = ...,
    rebuild_csc: bool = ...,
    csc_cols_per_shard: int = ...,
    csc_memory_limit: str = ...,
    group_by: str | None = ...,
    reference: str | Sequence[str] | dict[str, str] | None = ...,
    group_target_bytes: int | str | None = ...,
    group_max_bytes: int | str | None = ...,
    group_write_block_bytes: int | str | None = ...,
) -> None:
    """Globally reorder cells (the obs axis) by an obs key, for X-read locality
    and contiguous predicate-index shard ranges on the sort key. See
    `shuffle` for the random-order counterpart."""
    ...


def shuffle(
    input: str,
    output: str,
    seed: int = ...,
    shard_size: int | None = ...,
    codec: str = ...,
    index_obs: Sequence[str] | None = ...,
    index_var: Sequence[str] | None = ...,
    index_preset: str | None = ...,
    index_auto_threshold: int | None = ...,
    memory_budget: str | None = ...,
    temp_dir: str | None = ...,
    bitmap: str = ...,
    rebuild_csc: bool = ...,
    csc_cols_per_shard: int = ...,
    csc_memory_limit: str = ...,
) -> None:
    """Globally reorder cells (the obs axis) by a seeded random permutation, so
    a training loader gets i.i.d. batches at any `shard_group_size`. `seed` is
    recorded in provenance and is the only record of the permutation. X usually
    grows on zstd/shufdelta-coded files; the row order is the inverse of what a
    predicate index wants."""
    ...


def collate_cellset_gathered(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    set_offsets: np.ndarray,
    cell_indices: np.ndarray,
    file_ids: np.ndarray,
    role_tags: np.ndarray,
    k_dec: int,
    query_gene_ids: np.ndarray,
    enc_mask_positions: np.ndarray,
    hide_readout: np.ndarray,
    n_measured: np.ndarray,
    k_enc: int,
    mode: str,
    n_genes_total: int,
    target_sum: float | None = None,
    lib_size_redef: bool | None = None,
    pflog_alpha: float | None = None,
) -> dict[str, Any]:
    """Collate an already-gathered, **global-vocab** CSR batch (from
    `SparseCellSetDataset.iter_with_plans`) into stacked encoder/target tensors
    (state3 "3A hybrid"). Pure compute; releases the GIL. Returns a dict of flat
    stacked arrays plus the shape scalars ``n_rows``/``k_enc``/``k_dec``.

    ``pflog_alpha`` is required when ``mode="pflog_raw"`` (PFlog v4) and rejected
    otherwise; estimate it once with ``pyscx.accel.pflog`` over the training
    manifest. Note this kernel does **not** downsample — count-depth augmentation
    belongs in the gather stage, ahead of query sampling; see
    `SparseCellSetDataset`'s ``downsample_*`` arguments and
    `downsample_counts_csr`."""
    ...


def downsample_file_identity(path: str) -> int:
    """Stable 64-bit RNG-key identity for an ``.scx`` path (blake3 of the
    canonical path).

    Keyed on the *resolved* path rather than on manifest position, so a reordered
    manifest — or a subset of it — draws the same counts for the same cells. Pass
    these values as ``file_identities`` to `downsample_counts_csr` when you gather
    your own CSR; `SparseCellSetDataset` derives them from ``paths`` for you."""
    ...


def downsample_counts_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    rows: np.ndarray,
    file_identities: np.ndarray,
    target_library_size: int,
    method: Literal["binomial", "multinomial"] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Seeded per-row count downsample over an already-gathered CSR batch.

    Returns a dict of ``indptr`` / ``indices`` / ``data``. **Rows shrink** —
    counts that sample to zero are pruned — so the returned ``indptr`` differs
    from the input's.

    ``method`` is ``"multinomial"`` (default; hits ``target_library_size``
    exactly) or ``"binomial"`` (hits it in expectation). Negatives are clipped and
    counts are rounded ties-to-even before sampling, and a row already at or below
    target is left alone apart from that rounding.

    ``rows`` and ``file_identities`` are parallel to the batch's rows and supply
    the RNG key; get the identities from `downsample_file_identity`. An empty
    ``file_identities`` keys on ``(seed, method, row)`` alone — correct for a
    single-file batch, ambiguous across files (two files' row 5 would share a
    draw).

    Prefer `SparseCellSetDataset`'s ``downsample_*`` arguments when the loader is
    doing the gather: applying the draw before the batch leaves Rust is what keeps
    query sampling and the collated numerics consistent. Pure compute; releases
    the GIL."""
    ...


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
    def shape(self) -> tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    @property
    def shard_count(self) -> int: ...
    @property
    def format_version(self) -> int: ...
    @property
    def codec_id(self) -> int: ...

    # Schema discovery — list obs/var column names (the `filter_obs`
    # vocabulary). Unlike the local `Experiment.*_keys()` footer read, the
    # cloud path fetches and assembles the full obs/var section to derive
    # the schema; the result is cached on this handle.
    def obs_keys(self) -> list[str]: ...
    def var_keys(self) -> list[str]: ...

    def read_obs(self, columns: list[str] | None = ...) -> Any:
        """Read obs as a pandas DataFrame over the cloud path.

        `columns` is a genuine pushdown: each obs shard is fetched as a
        projected range read, so the network cost is the requested columns'
        bytes rather than the whole obs body. The pandas index column is always
        retained. `columns=None` fetches and caches the full assembled table."""
        ...

    def read_var(
        self, columns: list[str] | None = ..., *, modality: str | None = ...
    ) -> Any:
        """Read var as a pandas DataFrame over the cloud path.

        Mirror of the local `Experiment.read_var`. Unlike this class's
        `read_obs`, `columns` is **not** a network pushdown — it is a
        projection applied after the fetch, because `var` is one section sized
        by `n_vars` and there is no per-column range read to save. The pandas
        index column (gene names) is always retained.

        `modality=<name>` selects one modality's gene axis; unknown name →
        `KeyError`."""
        ...

    def distinct_values(
        self, col: str, *, limit: int | None = ..., sort: bool = ...
    ) -> tuple[list[str], bool]:
        """Distinct values of a string/categorical obs column as
        `(values, has_more)`. Mirrors `Experiment.distinct_values`."""
        ...

    def obs_categorical(self, col: str) -> tuple[np.ndarray, list[str]]:
        """`(codes, categories)` over the cloud path. Mirrors
        `Experiment.obs_categorical` — same `-1`-for-null, first-seen-order and
        unreferenced-level semantics. Never fetches the full obs body."""
        ...

    def obs_categorical_many(
        self, cols: list[str]
    ) -> list[tuple[np.ndarray, list[str]]]:
        """`obs_categorical` for several columns in one pass over the obs
        shards. Mirrors `Experiment.obs_categorical_many`."""
        ...

    def query(self, modality: str | None = None) -> Any:
        """Start a lazy cloud query pipeline.

        ``modality`` scopes the query to one modality of a multimodal file
        (X / ``select_genes`` / ``filter_var`` resolve against that modality's
        var; ``filter_obs`` stays on the shared global obs axis). On a
        multimodal file ``modality`` is required (omitting → ``ValueError``);
        an unknown name → ``KeyError``. Omit it on single-modality files.
        """
        ...

    def __repr__(self) -> str: ...


def open_cloud(url: str) -> CloudExperiment: ...


def read_cloud(
    url: str,
    *,
    obs_filter: str | None = ...,
    var_names: list[str] | None = ...,
    modality: str | None = ...,
) -> Any:
    """One-call cloud read into an AnnData (= `open_cloud(url).query()…
    collect().to_anndata()`).

    ``modality`` scopes the read to one modality of a multimodal file (see
    ``CloudExperiment.query``).
    """
    ...


# Contract version for the native cell-set path: the kernel's
# encoder-crop/mask/target semantics, the accepted preprocess-mode strings, and
# the gather stage's value contract (clip, downsample). Consumers (e.g. state3)
# assert this at rust_collate setup to fail loudly on version skew. Currently 2.
# See scx-loader/src/sparse_cellset_collate.rs for what it does and does not cover.
COLLATE_CELLSET_CONTRACT_VERSION: int


# ---------------------------------------------------------------------------
# Other pyscx symbols re-exported via `from .pyscx import *` are typed as
# `Any` here. Add explicit stubs above if/when type-checking on those
# symbols becomes load-bearing.
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any: ...
