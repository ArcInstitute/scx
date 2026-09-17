"""Type stubs for the parts of `pyscx` that benefit most from static typing.

This file is intentionally narrow — currently it stubs `TrainingDataset`,
`IndexPlanDataset`, and its iterator, since those are the most error-prone
callers will run into (kwarg-heavy constructor, plan iterator protocol,
batch dict schema), plus the backed handle classes' materialization surface
(`to_memory` / `toarray` / `tocsr` / `tocsc`), which is undiscoverable
otherwise because `scipy.sparse.issparse` is False for a handle.

Other pyscx symbols re-exported via `from .pyscx import *` are typed as
`Any` to type-checkers until / unless someone needs more coverage.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Literal, NoReturn, Tuple, TypedDict

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
        `peak_bytes_in_cache`, `full_shard_groups`, `block_index_groups`,
        `row_group_hits`, `row_group_misses`, `row_group_evictions`,
        `row_group_bytes_inserted`, `row_group_duplicate_waiters`,
        `admitted_group_bytes`, `rejected_group_bytes`, `reuse_admissions`,
        `parallel_group_decodes`.

        `hits` … `duplicate_waiters` describe **whole-shard** entries; the
        `row_group_*` set describes the decoded **row groups** a framed
        scattered gather retains in the same LRU, under the same byte budget
        (`max_memory_mb`). On a framed file with `scatter_block_index=True`
        the whole-shard counters stay at 0 and the row-group ones carry the
        signal: `row_group_hits` is what the second batch over a hot region
        gets for free. `peak_bytes_in_cache` gauges both kinds together, so
        `peak <= bytes_inserted + row_group_bytes_inserted`. Retention is
        decided once per plan: the plan's footprint in the shared LRU — the
        row groups of every framed shard it touches, over every file, plus the
        shards it takes whole — must fit `max_memory_mb`'s cache share
        `budget / (lookahead + 1)`, and the prefetcher pre-decodes the eligible
        groups of exactly the admitted plans. So `row_group_misses` growing while
        `row_group_bytes_inserted` stays flat means the working set is over
        budget (raise `max_memory_mb`), not that the cache is off.
        `SCX_ROW_GROUP_CACHE=0` disables row-group retention process-wide.

        A plan **over** that share no longer forfeits retention outright: it
        keeps the groups another plan of the lookahead window also touches — a
        control pool, shared neighbours, a repeated pair member — and still
        drops its cold tail. `reuse_admissions` counts the plans that got such a
        partial verdict, and `admitted_group_bytes` / `rejected_group_bytes` is
        the split it decided (charged once per GROUP lookup, the unit
        `row_group_hits` counts, not once per requested row). The room such a
        verdict may spend is the share **minus the shards the plan takes
        whole** — those are inserted regardless of any admission decision. A workload whose plans share nothing inside a
        window reads `reuse_admissions == 0`, which is the pre-existing
        behaviour and not a fault."""
        ...

    def memory_budget(self) -> dict[str, Any]:
        """`breakdown` — the six-key per-component estimate from the
        construction auto-tune, in the shape every class that reports a budget
        uses — plus
        `max_memory_mb` (the resolved value in force, adaptive when the
        constructor was passed none), `effective_cache_shards` and
        `effective_lookahead`.

        Also `max_blocking_threads`: the cap on simultaneously-running shard
        decodes. It lives on the prefetch engine this class shares with
        `SparseCellSetDataset`, and is sized from the CONSTRUCTOR `lookahead` —
        which bounds in-flight plans, not the blocking task a plan spawns per
        distinct `(file, shard)` it touches.
        """
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
        """Shared shard-cache and reader-registry counters — same keys as
        `SparseCellSetDataset.cache_metrics`, including the `reader_*` set. The
        flat half of `metrics()`, kept because it predates it. Safe after
        exhaustion."""
        ...

    def metrics(self) -> dict[str, dict[str, int]]:
        """Cache- and prefetch-side counters as
        ``{"cache": {...}, "prefetch": {...}}`` — the same shape
        `IndexPlanBatchIter.metrics()` returns. ``cache`` is
        loader-cumulative (shared with `SparseCellSetDataset.cache_metrics`);
        ``prefetch`` is per-iter and resets on each `iter_with_plans` call.
        Safe after exhaustion.

        ``prefetch["prefetch_skipped_reader_limit"]`` counts plans that were not
        prefetched at all because they touch more distinct files than
        ``reader_limit`` can keep resident; such a plan still gets a real
        admission verdict, and the key is 0 on any dataset without a limit.

        ``prefetch["prefetch_skipped_block_index"]`` counts the L2
        *prefetch-time* decision: shards not warmed *whole* so the gather could
        take the block-index path. Such a shard is not left cold: when the
        plan's row groups fit ``max_memory_mb / (lookahead + 1)`` the
        prefetcher pre-decodes them into the row-group LRU instead, and the
        gather reports them as ``cache_metrics()["row_group_hits"]``. The
        counter stays 0 against an unframed file — and
        constructing one *with* ``scatter_block_index=True`` also emits a
        preflight ``UserWarning``, though this class defaults that kwarg off, so
        the default path is silent. It is **not** interchangeable with
        ``cache_metrics()["block_index_groups"]``, which is the route the
        gather took — at ``lookahead=0`` no prefetch runs, so every counter
        here is 0 while ``block_index_groups`` is positive."""
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
        scatter_block_index: bool | None = None,
        max_plan_rows: int | None = None,
        reader_limit: int | None = None,
    ) -> None:
        """``scatter_block_index`` (default ``False`` — the opposite of
        ``IndexPlanDataset``) gates the block-index (row-group) scattered gather
        and its prefetch warm-skip on row-group-framed files. It is off by
        default because the typical cell-set workload is cache-friendly (sorted
        data + a reused control pool → a small working set that fits the shard
        cache): decoding each hot shard once into the LRU and reusing it across
        batches was measured to beat the row-group path — 2.80 → 4.55 steps/s
        and 337 ms → ~5 ms per gather on a 50-file Tahoe atlas, at ≈ ``.h5ad``
        parity.

        That measurement predated the row-group LRU, and the re-measure (cache
        sized to the file, sets/s off vs on) found no fixed winner: tabula
        100k/7 shards 899 vs 5.4 random and 812 vs 6.9 grouped, but census_1m/62
        shards 11.4 vs 6.9 random and **300 vs 459 grouped**. Full-shard
        degrades with corpus size; the row-group route is roughly flat in it and
        tracks plan locality, and the two cross near 1M cells. The default stays
        ``False`` because flipping it costs 117–167× on 100k-cell corpora.

        Pass ``True`` for a large corpus with local plans, or for cache-hostile
        runs (working set ≫ cache, low shard reuse), where the row-group
        decode's bounded peak RAM is the memory-safe choice.
        The process-wide ``SCX_SCATTER_BLOCK_INDEX=0`` env var remains a hard
        kill-switch that forces the full-shard path regardless of this argument.

        ``downsample_target_library_size`` enables a seeded per-row count
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

    def gather(
        self,
        file_ids: Sequence[int],
        rows: Sequence[int],
        role_tags: Sequence[int],
        set_offsets: Sequence[int],
    ) -> _SparseCellSetBatchDict:
        """Gather **one** plan synchronously and return its batch dict.

        The same plan shape ``iter_with_plans`` consumes and the same batch dict
        it yields, for a caller holding a single plan rather than a stream::

            batch = ds.gather(*plan)

        **Admission is decided per call.** ``iter_with_plans`` takes one
        row-group verdict per plan and compares it against its divided share
        of the budget (``budget / (lookahead + 1)``), since a lookahead window's
        worth of plans has to coexist; a standalone ``gather`` has no window, so
        it takes one verdict over the plan against the whole byte budget. Output is
        identical either way — the verdict changes what the cache *retains*, not
        what is read — but the batch is gathered on the calling thread rather
        than a prefetched one, so this is not the way to drive an epoch.

        Raises exactly as the iterator does: ``RuntimeError`` for a malformed
        plan or a cross-file set with no ``remap_tables``, ``IndexError`` for a
        row past a file's ``n_obs``.
        """
        ...

    def cache_metrics(self) -> dict[str, Any]:
        """Cumulative shard-cache counters since construction: `hits`,
        `misses`, `evictions`, `bytes_inserted`, `duplicate_waiters`,
        `peak_bytes_in_cache`, `full_shard_groups`, `block_index_groups`, and
        the `row_group_*` set (`hits`, `misses`, `evictions`,
        `bytes_inserted`, `duplicate_waiters`) for the decoded row groups a
        framed `scatter_block_index=True` gather retains — the multi-file
        sibling of `IndexPlanDataset.cache_metrics`, same keys and meanings.

        Four further keys describe the **reader registry** rather than the
        shard cache: `reader_opens`, `reader_evictions`, `reader_resident` and
        `reader_hwm`. They are absent from `IndexPlanDataset.cache_metrics`,
        which is single-file and would report them as permanently zero. At the
        default `reader_limit=None`, `reader_opens` equals the manifest size
        and the other three never move. `reader_opens` counts the *registry's*
        opens — the handles it was given plus every reopen since — not the
        constructor's manifest scan, which opens every file once whatever the
        limit and would report a constant. Compare `reader_hwm` with
        `reader_limit`: the limit bounds handles the registry may drop, so a
        plan leasing more files at once than the limit exceeds it rather than
        blocking, and `reader_hwm` is where that shows.

        The last two report which scattered-read route the gathers took.
        `block_index_groups > 0` proves the row-group path ran and remains the
        authority on which route was taken. Opening an all-unframed set with
        `scatter_block_index=True` warns at construction; the warning's
        *absence* establishes only that at least one file is framed, never that
        a gather took the route. The
        prefetcher's own decisions are visible separately, via
        `SparseCellSetBatchIter.metrics()["prefetch"]`."""
        ...

    def memory_budget(self) -> dict[str, Any]:
        """Resolved shard-cache budget: `breakdown` (the six-key
        `BudgetBreakdown` every class that reports a budget uses), plus `max_memory_mb`
        (the value in force — adaptive when the constructor was passed none),
        `cache_shards`, `effective_cache_shards`, `shard_decoded_bytes`,
        `max_plan_rows`, `mean_nnz_per_row`, `max_blocking_threads`,
        `reader_limit` and `budget_exceeded`. `max_plan_rows`, when set, also REFUSES a plan wider
        than it — `gather` / `iter_with_plans` raise rather than allocate for a
        batch the cache was not sized for.

        `cache_bytes` and `python_overhead_bytes` are the non-zero terms by
        default: on the sparse path the shard cache *is* the budget, and there
        is no plan-tuple term. Pass `max_plan_rows` and `batch_buffer_bytes`
        joins them — plus `transient_bytes`, the second buffer a gather holds
        while it assembles the batch from a deduplicated read, on the
        configurations that cannot avoid one (a remap, a downsample, or more
        than one file). `total_bytes` fits
        `max_memory_mb` unless `budget_exceeded` is True, which means even a
        one-shard cache does not fit.

        That is a statement about the cache this loader sizes, not a ceiling on
        process RSS: the batch's per-row transients are never charged and the
        batch itself only when `max_plan_rows` declares how wide plans get (this path
        has no `max_plan_size`, so plan width is otherwise the caller's), and
        the LRU keeps a single oversize shard rather than refusing to cache it,
        so one above-average shard can sit above the byte cap.

        `max_blocking_threads` is the cap on simultaneously-running shard
        decodes in the prefetch engine — the other thing between a wide plan and
        unbounded transient memory. `lookahead` bounds in-flight *plans*, not
        the tasks a plan spawns.

        `reader_limit` is reported but deliberately **not** in `breakdown`: the
        breakdown is the byte model the shard cache is sized against, and an
        open reader's ~104 kB is not one of its terms. Charging readers there
        would shrink the cache by something the tuner has never accounted for;
        reporting the cap here says what the knob is without pretending it is
        priced. At `None` every file in the manifest stays open, which on a
        26k-file manifest is ~2.8-3.2 GB that no budget here describes."""
        ...

    def suggested_cache_shards(
        self,
        plan: tuple[
            Sequence[int], Sequence[int], Sequence[int], Sequence[int]
        ],
    ) -> int:
        """Distinct ``(file_id, shard)`` pairs a plan touches — the
        `cache_shards` that would let the whole batch stay resident.

        Takes one plan in the same `(file_ids, rows, role_tags, set_offsets)`
        shape `iter_with_plans` consumes; the last two are ignored (and not
        even converted). A tuple of the wrong arity raises `ValueError`. Pure
        index arithmetic: no I/O, no decode::

            probe = pyscx.SparseCellSetDataset(paths)
            need = max(probe.suggested_cache_shards(p) for p in plans[:64])
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

    @property
    def closed(self) -> bool:
        """True from `close()` until the next `__iter__` rebuilds.

        Deliberately not the terminal flag `IndexPlanDataset.closed` is — see
        `close()`. Never raises.
        """
        ...

    def __iter__(self) -> "TrainingDataset": ...
    def __next__(self) -> _TrainingBatchDict: ...

    def memory_budget(self) -> dict[str, Any]:
        """Memory budget diagnostics.

        `breakdown` is the six-key `BudgetBreakdown` every class that reports a
        budget uses (`cache_bytes`, `batch_buffer_bytes`,
        `lookahead_overhead_bytes`, `transient_bytes`, `python_overhead_bytes`,
        `total_bytes`); alongside it are this class's own
        `shard_group_size`, `prefetch_batches`, `batch_size`, `estimated_mb`,
        `mmap_mb` and `budget_exceeded`.

        `mmap_mb` is reported here and nowhere else, but it is **not** budgeted
        on any class: every loader treats the mmap'd file as evictable kernel
        page cache and excludes it, so `estimated_mb` and
        `breakdown["total_bytes"]` are the same number. `budget_exceeded` means
        the auto-tune could not fit even at its minimums, and it also raises a
        `UserWarning` at construction.
        """
        ...

    def close(self) -> None:
        """Shut the pipeline down: join the I/O + decode threads and release the
        rayon pool, with the GIL detached.

        Idempotent, and **not** terminal — unlike `IndexPlanDataset.close()`,
        the next `__iter__` rebuilds the pool and starts a fresh epoch, because
        this class's pool and runtime are per-epoch anyway.
        """
        ...

    def __repr__(self) -> str: ...


class MultimodalTrainingDataset:
    """Sequential streaming dataset over several modalities of one SCX file.

    One `TrainingPipeline` per requested modality, iterated in lockstep: each
    batch carries the same `cell_indices` row ordering across modalities, and
    the wrapper raises `RuntimeError` if the per-modality shufflers diverge
    (usually a file-construction bug — differing shard layouts).

    Transforms are ON by default, exactly as on `TrainingDataset`: pass
    `normalize=False, log1p=False` for raw counts.
    """

    def __init__(
        self,
        path: str,
        modalities: Sequence[str],
        batch_size: int | None = None,
        hvg_indices: Sequence[int] | np.ndarray | None = None,
        obs_columns: list[str] | None = None,
        return_dict: bool | None = None,
        normalize: bool | None = None,  # default True — see class docstring
        log1p: bool | None = None,  # default True — see class docstring
        target_sum: float | None = None,
        pflog: bool | None = None,
        pflog_alpha: float | None = None,
        shard_group_size: int | None = None,
        prefetch_batches: int | None = None,
        seed: int | None = None,
        max_memory_mb: int | None = None,
    ) -> None: ...

    @property
    def n_obs(self) -> int: ...
    @property
    def n_vars(self) -> dict[str, int]:
        """Per-modality gene counts, keyed by modality name."""
        ...

    @property
    def modality_names(self) -> list[str]: ...
    @property
    def closed(self) -> bool:
        """True from `close()` until the next `__iter__` rebuilds. Never raises."""
        ...

    def __iter__(self) -> "MultimodalTrainingDataset": ...
    def __next__(self) -> Any:
        """A dict `{"X": {modality: ndarray}, "obs": {...}, "cell_indices": ...}`
        when built with `return_dict=True` (the default), else a tuple of the
        per-modality X arrays."""
        ...

    def memory_budget(self) -> dict[str, Any]:
        """Memory budget diagnostics.

        `batch_size` and `shard_group_size` are the values pinned uniformly
        across modalities (the loader forces every modality onto the
        cross-modality minimum so their batches stay row-aligned).
        `max_memory_mb` is what you asked for; `effective_total_mb` is what the
        per-modality split actually budgets, and **it can be larger** — the
        split is proportional to per-modality nnz with a floor, and a share
        below that floor cannot be constructed at all. `modalities` maps each
        modality name to the same envelope `TrainingDataset.memory_budget()`
        returns, `breakdown` included."""
        ...

    def close(self) -> None:
        """Shut every modality's pipeline down, GIL detached. Idempotent, and
        **not** terminal — see `TrainingDataset.close`."""
        ...

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


# ---------------------------------------------------------------------------
# Backed handle classes — the lazy matrices that appear as `X`, a layer, or an
# aligned value after `pyscx.open(path).to_anndata(backed=True)`.
#
# Stubbed for the materialization surface only, because that is what callers
# reach for and what is otherwise undiscoverable: `scipy.sparse.issparse` is
# False for these, so the usual "is this sparse?" reflex misroutes and nothing
# points at `to_memory()`. See `pyscx.is_backed_handle`.
#
# WARNING: each class keeps `def __getattr__(self, name: str) -> Any`, and it is
# load-bearing while these stubs stay partial -- do not delete it. A class stub
# is NOT partial to a type checker: the module-level `__getattr__` at the bottom
# of this file covers only module attributes, so once a class is declared here
# every member omitted from it becomes an error instead of falling through to
# `Any`. Without the per-class escape, declaring these four would have broken
# `.dtype`, `.copy()`, `.sum()` and ~20 other real members for py.typed
# consumers -- a regression caused by *adding* stubs.
#
# `__getattr__` does NOT reach special methods, though: Python looks those up on
# the type, so `adata.X[0:10]` and `x > 0` stayed broken until `__getitem__` and
# friends were declared outright. Each class therefore lists exactly the dunders
# its Rust class implements -- no more (a declared-but-absent dunder would let a
# type checker green-light a call that fails at runtime) and no fewer.
# `tests/test_backed.py::test_handle_stubs_declare_every_runtime_special`
# enforces both directions.
# ---------------------------------------------------------------------------


class ScxBackedSparseDataset:
    """Lazy CSR handle over an SCX file's `X`."""

    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    @property
    def stored_dtype(self) -> np.dtype:
        """On-disk value encoding (`uint8`/`uint16`/`uint32`/`float32`/`float16`;
        the widest when shards mix). `dtype` stays `float32` — the decode type."""
        ...
    @property
    def cache_shards(self) -> int:
        """Decoded-shard LRU size this handle was opened with; 0 = no cache."""
        ...
    def to_memory(self) -> Any: ...
    def toarray(self) -> np.ndarray: ...
    def tocsr(self) -> Any: ...
    def tocsc(self) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...
    # numpy's array protocol is implemented only to *refuse*: `np.asarray` on a
    # handle raises TypeError instead of decoding the whole matrix (it used to
    # return a 0-d object array). Use `to_memory()` / `toarray()` / a slice.
    def __array__(self, dtype: Any = ..., copy: Any = ...) -> NoReturn: ...
    # Comparisons return a mask, not a bool -- hence the deliberate override of
    # `object.__eq__` / `__ne__`, the same shape numpy's stubs use.
    def __eq__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __ne__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __lt__(self, other: Any) -> Any: ...
    def __le__(self, other: Any) -> Any: ...
    def __gt__(self, other: Any) -> Any: ...
    def __ge__(self, other: Any) -> Any: ...
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __matmul__(self, other: Any) -> Any: ...
    def __rmatmul__(self, other: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...


class ScxBackedLayerDataset:
    """Lazy CSR handle over one of an SCX file's layers."""

    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    @property
    def stored_dtype(self) -> np.dtype:
        """On-disk value encoding (`uint8`/`uint16`/`uint32`/`float32`/`float16`;
        the widest when shards mix). `dtype` stays `float32` — the decode type."""
        ...
    @property
    def cache_shards(self) -> int:
        """Decoded-shard LRU size this handle was opened with; 0 = no cache."""
        ...
    def to_memory(self) -> Any: ...
    def toarray(self) -> np.ndarray: ...
    def tocsr(self) -> Any: ...
    def tocsc(self) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...
    # numpy's array protocol is implemented only to *refuse*: `np.asarray` on a
    # handle raises TypeError instead of decoding the whole matrix (it used to
    # return a 0-d object array). Use `to_memory()` / `toarray()` / a slice.
    def __array__(self, dtype: Any = ..., copy: Any = ...) -> NoReturn: ...
    # Comparisons return a mask, not a bool -- hence the deliberate override of
    # `object.__eq__` / `__ne__`, the same shape numpy's stubs use.
    def __eq__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __ne__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __lt__(self, other: Any) -> Any: ...
    def __le__(self, other: Any) -> Any: ...
    def __gt__(self, other: Any) -> Any: ...
    def __ge__(self, other: Any) -> Any: ...
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __matmul__(self, other: Any) -> Any: ...
    def __rmatmul__(self, other: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...


class ScxBackedObsmDataset:
    """Lazy handle over a dense aligned store (`obsm` / `varm`).

    Dense, so it exposes no `tocsr` / `tocsc` / `nnz` — deliberately absent
    rather than missing.
    """

    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def cache_shards(self) -> int:
        """Decoded-shard LRU size this handle was opened with; 0 = no cache."""
        ...
    def to_memory(self) -> Any: ...
    def toarray(self) -> np.ndarray: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...
    def __array__(self, dtype: Any = ..., copy: Any = ...) -> np.ndarray: ...
    # No comparison / arithmetic dunders: this class implements none. What
    # `dir()` shows are `object`'s own slots, so declaring them here would let a
    # type checker accept `obsm > 0`, which raises at runtime.
    def __getattr__(self, name: str) -> Any: ...


class ScxLazyTransformedDataset:
    """Lazy handle carrying stacked transforms (e.g. normalize_total, log1p)."""

    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    @property
    def stored_dtype(self) -> np.dtype:
        """On-disk value encoding (`uint8`/`uint16`/`uint32`/`float32`/`float16`;
        the widest when shards mix). `dtype` stays `float32` — the decode type."""
        ...
    @property
    def cache_shards(self) -> int:
        """Decoded-shard LRU size this handle was opened with; 0 = no cache."""
        ...
    def to_memory(self) -> Any: ...
    def toarray(self) -> np.ndarray: ...
    def tocsr(self) -> Any: ...
    def tocsc(self) -> Any: ...
    def __getitem__(self, key: Any) -> Any: ...
    def __len__(self) -> int: ...
    # numpy's array protocol is implemented only to *refuse*: `np.asarray` on a
    # handle raises TypeError instead of decoding the whole matrix (it used to
    # return a 0-d object array). Use `to_memory()` / `toarray()` / a slice.
    def __array__(self, dtype: Any = ..., copy: Any = ...) -> NoReturn: ...
    # Comparisons return a mask, not a bool -- hence the deliberate override of
    # `object.__eq__` / `__ne__`, the same shape numpy's stubs use.
    def __eq__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __ne__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __lt__(self, other: Any) -> Any: ...
    def __le__(self, other: Any) -> Any: ...
    def __gt__(self, other: Any) -> Any: ...
    def __ge__(self, other: Any) -> Any: ...
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __matmul__(self, other: Any) -> Any: ...
    def __rmatmul__(self, other: Any) -> Any: ...
    def __getattr__(self, name: str) -> Any: ...


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
    def value_encoding(self) -> str:
        """On-disk value encoding of the CSR shards as `scx info` prints it:
        `"uint16"` when uniform, `"mixed (uint8, uint16)"` when shards differ,
        `"n/a"` with no shards. One 76-byte header read per shard, no decode."""
        ...
    @property
    def is_integer(self) -> bool:
        """True when every CSR shard is integer-encoded (the stored values are
        counts); False for any float shard or a file with no shards."""
        ...
    @property
    def max_value(self) -> int:
        """Largest stored value, from the per-shard catalog stats. 0 for a
        float-encoded file (float shards record no range) — read with
        `is_integer`. Physical: deleted rows still count."""
        ...
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
    def obsp_keys(self) -> list[str]:
        """Keys of the ``obsp`` pairwise graphs. Pure catalog scan.

        Lists only the COO forms every conversion path writes; the CSR-backed
        ``ObspCsrShard`` has no read API and is not named here."""
        ...

    def read_obsp_rows(
        self,
        key: str,
        start: int,
        stop: int,
        *,
        logical: bool = True,
    ) -> Any:
        """Rows ``[start, stop)`` of ``obsp/<key>`` as a scipy CSR matrix.

        The bounded counterpart to ``to_anndata().obsp[key]``: only the shards
        covering the range are decoded. A graph stored as one unsharded section
        — what ``scx sort`` emits — is decoded whole and then sliced.

        **Row space.** ``logical=True`` (the default, matching `read_obs`) takes
        ``start`` / ``stop`` as live row indices, drops any edge whose either
        endpoint is deleted and renumbers both axes into live space, so the
        column extent is ``n_obs``. ``logical=False`` is the physical graph with
        no filtering and column extent ``n_obs_physical``.

        Each call resolves the shard layout afresh (footer schemas only, no
        payload), and a ``logical`` range is bounded by the **physical** span it
        covers — two live rows at opposite ends of a heavily deleted file read
        the whole graph."""
        ...

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

    def read_obs(
        self, columns: list[str] | None = ..., *, logical: bool = ...
    ) -> Any:
        """Read obs (optionally a column projection) as a pandas DataFrame,
        without materializing X. `columns=None` reads the full obs table.

        `logical=True` (default) returns the **live** rows — deletion vectors
        applied, so `len(read_obs()) == n_obs == len(to_anndata(backed=True).obs)`
        row for row. `logical=False` returns the physical table:
        `n_obs_physical` rows, deleted rows in place. (`to_h5ad(obs_mask=)` and
        `mark_deleted(mask)` accept a mask in either row space.) Identical on a
        file without deletions. Changed in 0.17: the default used to be the
        physical table on every file. A frame computed from `read_obs()` can
        still be landed with `attach_obs_columns(positional=True)` or
        `modify_metadata(obs=)`, which accept either row space."""
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

    def obs_categorical(
        self, col: str, *, logical: bool = ...
    ) -> tuple[np.ndarray, list[str]]:
        """`(codes, categories)` for a string/categorical obs column.

        `codes` is `int32`, one entry per obs row, `-1` for missing
        (`pandas.Categorical.codes` convention); `categories[code]` is the
        string value. Computed shard-by-shard and returned as numpy directly —
        no Arrow-IPC round-trip, no pandas frame, X never touched.

        Category order is **first-seen**, not lexicographic. Unreferenced
        dictionary levels are retained (as pandas retains unused levels).
        Raises `ValueError` for non-string columns.

        Row space: `logical=True` (default) gives one code per live row
        (`len(codes) == n_obs`, aligned with `read_obs()`); `logical=False`
        one per physical row (`n_obs_physical`, deleted rows in place).
        Indexing one space by the other's row ids addresses the wrong cell.
        `categories` is the same either way. Changed in 0.17: the default
        used to be physical, like `read_obs()`.
        """
        ...

    def obs_categorical_many(
        self, cols: list[str], *, logical: bool = ...
    ) -> list[tuple[np.ndarray, list[str]]]:
        """`obs_categorical` for several columns in one shard pass.

        Returns `(codes, categories)` per column in `cols` order. N columns
        cost one projected read per obs shard instead of N. `logical=` as on
        `obs_categorical`."""
        ...

    def info(self) -> str:
        """One-line codec / shard / format-version internals, including the
        `value_encoding=`, `is_integer=` and `max_value=` tokens (also
        available as the getters of those names). Reads one shard header per
        CSR shard, so O(shards) rather than O(1)."""
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
        obsp: list[str] | None = ...,
        varp: list[str] | None = ...,
        varm: list[str] | None = ...,
        raw: bool = ...,
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
        layer: str | None = None,
        logical: bool = True,
    ) -> Any:
        """Gather ``rows`` as a ``scipy.sparse.csr_matrix`` in request order.

        The bounded shard-wise gather: each touched shard is decoded once (a
        sparse request on a row-group-framed shard decodes only the touched row
        groups) and the result is assembled once into exact-size buffers, so
        peak memory is the result plus the shard cache, plus up to
        ``cache_shards`` shards decoding in flight while that cache fills (at
        most ``2 × cache_shards`` decoded shards beside the result) — never a
        second copy of the result.
        ``rows`` is a boolean mask or any 1-D
        integer array-like (list, ``range``, ndarray of any integer dtype); it
        may contain duplicates and need not be sorted; negative indices wrap
        once. Returns raw-local gene indices (no global-vocab remap).

        ``logical=True`` (default) indexes the rows ``Experiment.n_obs`` /
        ``read_obs()`` describe — deletion vectors applied, exactly as
        ``to_anndata(backed=True).X[rows]`` does. ``logical=False`` indexes the
        physical file rows (``n_obs_physical``), deleted cells included.
        Changed in 0.17: the method used to address physical rows only, so on
        a file with deletion vectors the same ids now select different cells.
        ``layer=`` gathers from that layer instead of ``X`` (``ValueError`` if
        the layer does not exist; not supported on a multimodal file — index
        the modality's layer handle from ``to_mudata()`` instead).

        ``cache_shards`` bounds peak decoded-shard memory for the gather (not a
        speedup knob — each shard is decoded once per call). A fresh reader is
        opened per call (fork-safe; this is the eval / random-access utility,
        not the training hot path). Out-of-range ids, and a boolean mask whose
        length is not the row count, raise ``IndexError``. Equivalent to the
        backed ``adata.X[rows]`` / ``adata.layers[name][rows]`` analysis path,
        without building the AnnData.
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
        obsp: list[str] | None = ...,
        varp: list[str] | None = ...,
        varm: list[str] | None = ...,
        raw: bool = ...,
    ) -> Any:
        """Materialise on-device as an AnnData backed by a
        ``cupyx.scipy.sparse.csr_matrix`` (f32-native). Rejects the
        dtype/container narrowing kwargs at runtime."""
        ...

    def mark_deleted(self, mask: Any) -> int:
        """Mark cells for deletion via a boolean mask (numpy array or pandas
        Series) in either row space: ``n_obs`` entries (the live rows
        ``read_obs()`` describes) or ``n_obs_physical`` entries (every physical
        row, as ``read_obs(logical=False)`` describes); any other length raises
        naming both counts. A Series with a labelled index is checked for order
        (the file's barcodes in a different order raise). Returns the total
        number of deleted rows. Writes a deletion vector."""
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
    """Open an SCX file as an `Experiment` handle."""
    ...


def is_backed_handle(obj: Any) -> bool:
    """Is `obj` a lazy SCX handle rather than an in-memory matrix?

    True for `ScxBackedSparseDataset`, `ScxBackedLayerDataset`,
    `ScxBackedObsmDataset` and `ScxLazyTransformedDataset`. Note that
    `scipy.sparse.issparse` is False for all of them, and `np.asarray` on a
    sparse handle raises `TypeError` (call `to_memory()` / `toarray()`; the
    dense obsm handle does materialise).
    """
    ...


def read(path: Any, *, verify: bool = ..., **kwargs: Any) -> Any:
    """Read an SCX file into an `anndata.AnnData` in one call
    (= `open(path, verify=verify).to_anndata(**kwargs)`)."""
    ...


def write(adata: Any, path: Any, **kwargs: Any) -> None:
    """Write an `anndata.AnnData` to an SCX file in one call
    (= `from_anndata(adata, path, **kwargs)`)."""
    ...


def validate(path: Any, deep: bool = ...) -> list[tuple[str, bool]]:
    """Validate an SCX file's catalog + per-section BLAKE3 checksums
    (`deep=` additionally re-decodes every sparse shard)."""
    ...


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
    input: str,
    output: str,
    codec: str = ...,
    shard_obs: str = ...,
    memory_budget: int | str | None = ...,
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
    query_offsets: np.ndarray | None = None,
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
    `downsample_counts_csr`.

    ``query_offsets`` (contract v3) switches the decoder query from per-set to
    per-row. Omitted, the query is the per-set ``[n_sets, k_dec]`` panel it has
    always been and every **pre-existing field** is byte-identical to v2's — the
    returned dict is not identical, because ``target_pad_mask`` is a new key on
    every path. Supplied, it is an
    ``[n_rows + 1]`` prefix array over a ragged ``query_gene_ids``,
    ``n_measured`` becomes per-row, ``enc_mask_positions`` becomes parallel to
    the ragged query rather than ``k_dec``-strided, and ``k_dec`` is the padded
    output width — ``target_pad_mask`` in the returned dict says which target
    slots are padding rather than a real zero."""
    ...


_NeighborhoodPlan = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""One cell-set plan: ``(file_ids, rows, role_tags, set_offsets)``, the shape
`SparseCellSetDataset.gather` and `.iter_with_plans` take."""


def neighborhood_plans_from_graph(
    path: Any,
    key: str = "connectivities",
    *,
    file_id: int,
    k: int | None = None,
    weight_order: Literal["desc", "asc"] | None = None,
    include_center: bool = True,
    drop_deleted: bool = True,
    chunk_rows: int = 65536,
) -> tuple[list[_NeighborhoodPlan], np.ndarray]:
    """One cell-set plan per cell, from a stored ``obsp`` neighbourhood graph.

    Each set is the centre first (``role_tag`` 0) then its neighbours
    (``role_tag`` 1). ``k`` keeps only ``k`` edges, taken at the
    ``weight_order`` end with ties by column ascending; ``None`` keeps every
    stored edge in column order.

    ⚠️ ``weight_order`` is **required whenever ``k`` is given** and has no
    default: ``"desc"`` suits an affinity graph (``connectivities``: larger =
    closer), ``"asc"`` a distance graph (``distances``: larger = farther). A
    default would be right for the default key and silently wrong the moment a
    caller changed only the key, returning each cell's ``k`` **farthest**
    neighbours.

    ``file_id`` is **required** — this file's position in the
    `SparseCellSetDataset` manifest the plans will be gathered with, not a file
    identity, and nothing checks it against the file.

    Rows are **physical**: a deleted centre yields no set (so ``centers`` is
    shorter than ``n_obs``) and a deleted neighbour is never emitted. Without
    ``k`` a set is short by whatever is gone; with ``k``, deleted rows are not
    candidates, so the ``k`` best *live* neighbours are taken and the set is
    still ``k`` wide.

    The graph is read one ``chunk_rows`` range at a time; a graph stored as a
    single unsharded section — what ``scx sort`` emits — is decoded whole."""
    ...


def neighborhood_plans_from_coords(
    path: Any,
    obsm_key: str = "spatial",
    *,
    file_id: int,
    k: int | None = None,
    radius: float | None = None,
    include_center: bool = True,
    drop_deleted: bool = True,
) -> tuple[list[_NeighborhoodPlan], np.ndarray]:
    """One cell-set plan per cell, from ``obsm`` coordinates, with no index.

    Exactly one of ``k`` or ``radius`` is required. A per-file uniform grid is
    built at call time (O(n)) and searched ring by ring, so the answer is exact;
    ties are ordered by squared distance ascending, then row ascending. Integer
    and float64 coordinate columns are narrowed to float32. A NaN or infinite
    coordinate on a kept cell raises rather than being bucketed somewhere.

    ⚠️ **1-D, 2-D or 3-D only** — the grid search is exponential in the
    dimensionality, so a wide embedding such as a 50-component ``X_pca`` would
    not return rather than merely being slow. Use
    `neighborhood_plans_from_graph` over a kNN written into ``obsp`` instead.

    ``file_id`` is **required**, for the reason
    `neighborhood_plans_from_graph` gives."""
    ...


def batch_plans(
    plans: Iterable[_NeighborhoodPlan],
    sets_per_batch: int,
    *,
    shuffle_seed: int | None = None,
) -> list[_NeighborhoodPlan]:
    """Concatenate single-set plans into batch plans of ``sets_per_batch`` sets.

    Overlapping neighbourhoods are the reuse signal the shard cache acts on, and
    batching is what puts overlapping sets in one plan where it can see them.
    ``shuffle_seed`` reorders the **sets**, never a set's members, reproducibly.
    The last batch is short rather than dropped."""
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
    def n_obs(self) -> int:
        """Live row count (header `n_obs` minus deleted rows), like the local
        `Experiment.n_obs`; one small section read on a file with deletions
        (raises if that read fails, so it cannot disagree with `read_obs()`).
        Changed in 0.17: was the physical header count."""
        ...
    @property
    def n_obs_physical(self) -> int:
        """Physical row count straight from the header — every row, deleted
        ones included. Equals `n_obs` on a file with no deletions."""
        ...
    @property
    def has_deletions(self) -> bool:
        """`True` when the file carries a deletion vector. Header-only."""
        ...
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
    def info(self) -> str:
        """One-line codec / shard / format-version internals, token for token the
        same as `Experiment.info()` (the `value_encoding=` / `is_integer=` tokens
        cost one 76-byte range read per CSR shard)."""
        ...

    # Schema discovery — list obs/var column names (the `filter_obs`
    # vocabulary). Unlike the local `Experiment.*_keys()` footer read, the
    # cloud path fetches and assembles the full obs/var section to derive
    # the schema; the result is cached on this handle.
    def obs_keys(self) -> list[str]: ...
    def var_keys(self) -> list[str]: ...

    def read_obs(
        self, columns: list[str] | None = ..., *, logical: bool = ...
    ) -> Any:
        """Read obs as a pandas DataFrame over the cloud path.

        `logical=True` (default) returns the live rows (deletion vectors
        applied, `len == n_obs`); `logical=False` the physical table — the
        same contract as `Experiment.read_obs`, changed in 0.17 alongside it.
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

    def obs_categorical(
        self, col: str, *, logical: bool = ...
    ) -> tuple[np.ndarray, list[str]]:
        """`(codes, categories)` over the cloud path. Mirrors
        `Experiment.obs_categorical` — same `-1`-for-null, first-seen-order and
        unreferenced-level semantics, same `logical=` row space. Never fetches
        the full obs body."""
        ...

    def obs_categorical_many(
        self, cols: list[str], *, logical: bool = ...
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
# encoder-crop/mask/target semantics, the accepted preprocess-mode strings, the
# gather stage's value contract (clip, downsample), and since v3 the per-row
# query addressing (`query_offsets`) plus the `target_pad_mask` output.
# Consumers (e.g. state3) assert this at rust_collate setup to fail loudly on
# version skew. Currently 3; a call that omits `query_offsets` reproduces every
# pre-existing field byte for byte, but the returned dict gains a
# `target_pad_mask` key on every path, so the payload is not identical.
# See scx-loader/src/sparse_cellset_collate.rs for what it does and does not cover.
COLLATE_CELLSET_CONTRACT_VERSION: int


# ---------------------------------------------------------------------------
# Other pyscx symbols re-exported via `from .pyscx import *` are typed as
# `Any` here. Add explicit stubs above if/when type-checking on those
# symbols becomes load-bearing.
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any: ...
