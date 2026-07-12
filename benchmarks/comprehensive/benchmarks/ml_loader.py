"""
ML Data Loader Throughput benchmark.

Measures batched iteration throughput for ML training workloads across
four loader implementations:

  - SCX: ``pyscx.TrainingDataset`` (triple-buffered pipeline)
  - AnnData: Load full h5ad into RAM, manual numpy batching
  - TileDB-SOMA-ML: ``tiledbsoma_ml.ExperimentDataset`` + DataLoader
  - scDataLoader: ``scdataloader.SimpleAnnDataset`` + DataLoader

Scenarios per loader:

  - raw: no HVG projection, no normalization
  - hvg: 2,000-gene HVG projection, no normalization
  - norm: no HVG projection, normalize_total + log1p
  - hvg_norm: HVG projection + normalize_total + log1p

Additionally, for SCX formats on GPU nodes, a ``gpu_train`` scenario
runs a scVI-equivalent VAE training loop and records GPU utilization.

Metrics: batches/sec, cells/sec, time-to-first-batch (TTFB), peak RSS,
GPU utilization (gpu_train only).
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import statistics
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    ML_BATCH_SIZE,
    N_WARMUP_RUNS,
    QUERY_N_HVGS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

_HAS_PYSCX = False
try:
    import pyscx  # noqa: F401

    _HAS_PYSCX = True
except ImportError:
    pass

_HAS_TORCH = False
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    pass

_HAS_SOMA_ML = False
try:
    import tiledbsoma  # noqa: F401
    from tiledbsoma_ml import ExperimentDataset  # noqa: F401

    _HAS_SOMA_ML = True
except ImportError:
    pass

_HAS_SCDATALOADER = False
try:
    from scdataloader import SimpleAnnDataset  # noqa: F401

    _HAS_SCDATALOADER = True
except ImportError:
    pass

_HAS_SLAF = False
try:
    import slaf  # noqa: F401

    _HAS_SLAF = True
except ImportError:
    pass

_HAS_SHARDAD = False
try:
    import shardad  # noqa: F401

    _HAS_SHARDAD = True
except ImportError:
    pass

_HAS_CELLSTREAM = False
try:
    import cellstream  # noqa: F401

    _HAS_CELLSTREAM = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCX_KEYS = {
    "scx_auto",
    "scx_fast",
    "scx_scx1",
    "scx_zstd",
    "scx_lz4",
    "scx_none",
    "scx_pcodec",
}

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    _SCX_KEYS | {"h5ad_none", "h5ad_gzip", "tiledb_soma", "slaf", "shardad", "cellstream"}
)
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Derived from
``_resolve_loader`` (the canonical static dict) — any change here must
match the keys that ``_resolve_loader`` recognises. The runtime guard
in ``run()`` still catches direct invocation."""


# Above this cell count the `num_workers>0` DataLoader scenarios
# (`pyscx_training_dataset_workers2[_persistent]`) are skipped: at census scale
# (≥500k cells) they reliably exceed the SLURM time limit (num_workers=2 spawn
# each rebuild the dataset over the full epoch), and because the benchmark only
# writes its result JSON after *all* scenarios complete, a workers2 time-out
# discards the whole (dataset, format) result — including the raw/hvg_norm/gpu_train
# scenarios that finished. Skipping keeps census scx rows capturable. tabula
# (100k) completes workers2 comfortably; census_500k (500k) does not, so the
# default threshold sits between them. Env-tunable via SCX_BENCH_WORKERS2_MAX_OBS.
def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` on unset/malformed
    input (e.g. a float string like ``250000.0``) instead of raising at
    import time and taking the whole benchmark module down."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


_WORKERS2_MAX_OBS: int = _env_int("SCX_BENCH_WORKERS2_MAX_OBS", 250000)


def _scx_memory_budget_mb() -> int | None:
    """Loader `max_memory_mb` scaled to the SLURM allocation.

    The default (4096 MB) is too small for atlas-scale full-width datasets
    (1M × 61,497): the loader's memory-budget auto-tune then collapses
    ``batch_size`` to its 64 minimum, which both makes batches/sec
    incomparable across codecs (a larger-on-disk codec tips over the
    threshold first) and inflates the epoch to ~16× more tiny batches
    (a cause of the census `workers2` time-outs). Use most of whatever
    ``--mem`` the SLURM job got so ``batch_size`` stays at 1024. Returns
    ``None`` off-SLURM so local/CI runs keep the built-in default.
    """
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            return max(4096, int(int(per_node) * 0.6))
        except ValueError:
            pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    n_cpu = os.environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu and n_cpu:
        try:
            return max(4096, int(int(per_cpu) * int(n_cpu) * 0.6))
        except ValueError:
            pass
    return None

_SCENARIOS: list[tuple[str, bool, bool]] = [
    # (name, hvg, normalize)
    ("raw", False, False),
    ("hvg", True, False),
    ("norm", False, True),
    ("hvg_norm", True, True),
]

_N_TTFB_RUNS = 3

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class _EpochResult:
    """Outcome of a single epoch iteration."""

    n_batches: int = 0
    n_cells: int = 0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


from benchmarks.comprehensive.rss import current_rss_mb as _current_rss_mb


def _timed_epoch(fn, *args, **kwargs) -> tuple[_EpochResult, float, float, float, float]:
    """Run *fn* and return (EpochResult, wall_s, user_s, sys_s, peak_rss_mb)."""
    gc.collect()
    rss_before = _current_rss_mb()
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()

    result = fn(*args, **kwargs)

    wall_s = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    rss_after = _current_rss_mb()

    return (
        result,
        wall_s,
        r1.ru_utime - r0.ru_utime,
        r1.ru_stime - r0.ru_stime,
        max(rss_before, rss_after),
    )


def _measure_ttfb(first_batch_fn, n_reps: int = _N_TTFB_RUNS) -> list[float]:
    """Measure time-to-first-batch over *n_reps* repetitions."""
    times: list[float] = []
    for _ in range(n_reps):
        gc.collect()
        t0 = time.perf_counter()
        first_batch_fn()
        times.append(time.perf_counter() - t0)
    return times


# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------


def _resolve_loader(format_key: str) -> str | None:
    """Map format key to loader type. Returns None for unsupported formats."""
    if format_key in _SCX_KEYS:
        return "scx"
    if format_key == "h5ad_none":
        return "anndata"
    if format_key == "h5ad_gzip":
        return "scdataloader"
    if format_key == "tiledb_soma":
        return "soma"
    if format_key == "slaf":
        return "slaf"
    if format_key == "shardad":
        return "shardad"
    if format_key == "cellstream":
        return "cellstream"
    return None


def _loader_available(loader_type: str) -> bool:
    """Check whether required dependencies are installed."""
    if loader_type == "scx":
        return _HAS_PYSCX
    if loader_type == "anndata":
        try:
            import anndata  # noqa: F401

            return True
        except ImportError:
            return False
    if loader_type == "scdataloader":
        return _HAS_SCDATALOADER and _HAS_TORCH
    if loader_type == "soma":
        return _HAS_SOMA_ML and _HAS_TORCH
    if loader_type == "slaf":
        return _HAS_SLAF and _HAS_TORCH
    if loader_type == "shardad":
        return _HAS_SHARDAD
    if loader_type == "cellstream":
        return _HAS_CELLSTREAM
    return False


# ---------------------------------------------------------------------------
# Per-loader epoch functions
# ---------------------------------------------------------------------------


def _run_scx_epoch(
    path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    import pyscx

    hvg_indices = list(range(QUERY_N_HVGS)) if hvg else None
    ds = pyscx.TrainingDataset(
        path,
        batch_size=batch_size,
        hvg_indices=hvg_indices,
        normalize=normalize,
        log1p=normalize,
        seed=seed,
        max_memory_mb=_scx_memory_budget_mb(),
    )
    n_batches = 0
    n_cells = 0
    for batch in ds:
        n_batches += 1
        n_cells += batch["X"].shape[0]
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


# ---------------------------------------------------------------------------
# `num_workers > 0` scenarios
#
# `pyscx_training_dataset_workers2` and `pyscx_training_dataset_workers2_persistent`
# wrap `pyscx.TrainingDataset` in a tiny `IterableDataset` shim that
# constructs the inner dataset lazily inside `__iter__` (the pattern used by
# `cell-load-scx/ScxTrainingDataset`). Each worker shards by batch index
# (`i % info.num_workers == info.id`) so every cell is yielded exactly once
# across workers — same scope as `num_workers=0` for direct throughput
# comparison.
#
# Pre-Phase-2: this code path deadlocked at the first batch. Post-Phase-2:
# emits `batches_per_sec__*`, `peak_rss_mb__*`, `cells_per_sec__*` for the
# regression gate (`thresholds.yaml`).
# ---------------------------------------------------------------------------


if _HAS_TORCH:
    import torch.utils.data as _td_for_shim

    class _LazyShardedShim(_td_for_shim.IterableDataset):  # type: ignore[misc]
        """Top-level (picklable under spawn) IterableDataset shim around
        `pyscx.TrainingDataset`. Mirrors the `cell-load-scx` lazy-construct
        pattern exactly so the benchmark exercises the production code
        path. Defined at module level (inside the `_HAS_TORCH` guard) so
        `multiprocessing.spawn` workers can pickle it by qualified name."""

        def __init__(
            self,
            path: str,
            batch_size: int,
            hvg: bool,
            normalize: bool,
            seed: int,
        ) -> None:
            super().__init__()
            self.path = path
            self.batch_size = batch_size
            self.hvg = hvg
            self.normalize = normalize
            self.seed = seed

        def __iter__(self):
            import pyscx
            import torch.utils.data as _td

            info = _td.get_worker_info()
            worker_id = info.id if info is not None else 0
            num_workers = info.num_workers if info is not None else 1

            hvg_indices = list(range(QUERY_N_HVGS)) if self.hvg else None
            ds = pyscx.TrainingDataset(
                self.path,
                batch_size=self.batch_size,
                hvg_indices=hvg_indices,
                normalize=self.normalize,
                log1p=self.normalize,
                seed=self.seed,
                max_memory_mb=_scx_memory_budget_mb(),
            )
            try:
                for i, batch in enumerate(ds):
                    if i % num_workers == worker_id:
                        yield batch
            finally:
                # Phase 2.5 close() — release the per-pipeline rayon pool +
                # tokio runtime promptly between epochs / workers.
                ds.close()


def _make_workers2_iterable_dataset(
    path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
):
    """Construct the lazy-sharded shim. Requires torch to be installed —
    callers gate on `_HAS_TORCH` before invoking."""
    if not _HAS_TORCH:
        raise RuntimeError(
            "_make_workers2_iterable_dataset requires torch.utils.data; "
            "callers must gate on _HAS_TORCH"
        )
    return _LazyShardedShim(path, batch_size, hvg, normalize, seed)


def _run_scx_dataloader_workers_epoch(
    path: str,
    batch_size: int,
    hvg: bool,
    normalize: bool,
    seed: int,
    num_workers: int,
    persistent_workers: bool,
    n_epochs: int,
) -> _EpochResult:
    """Drive a full epoch (or `n_epochs`) through
    `torch.utils.data.DataLoader(num_workers>0)` against the lazy-construct
    shim. Returns aggregate counts across all epochs."""
    import torch.utils.data as _td

    ds = _make_workers2_iterable_dataset(path, batch_size, hvg, normalize, seed)
    loader = _td.DataLoader(
        ds,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=_passthrough_collate,
    )
    n_batches = 0
    n_cells = 0
    for _ in range(n_epochs):
        for batch in loader:
            n_batches += 1
            n_cells += batch["X"].shape[0]
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _passthrough_collate(batch):
    """Top-level passthrough collate (lambdas are not picklable under spawn).
    The DataLoader calls this with a single yielded batch; we return it
    unchanged so the consumer sees the same dict shape as `num_workers=0`."""
    return batch


def _run_anndata_epoch(
    h5ad_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    import anndata

    adata = anndata.read_h5ad(h5ad_path)
    n_obs = adata.n_obs
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_obs)

    n_batches = 0
    n_cells = 0
    for start in range(0, n_obs, batch_size):
        batch_idx = indices[start : start + batch_size]
        X = adata.X[batch_idx]
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = X.astype(np.float32, copy=False)
        if hvg:
            X = X[:, :QUERY_N_HVGS]
        if normalize:
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums * 1e4
            X = np.log1p(X)
        n_batches += 1
        n_cells += X.shape[0]

    del adata
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _run_soma_epoch(
    soma_path: str, batch_size: int, hvg: bool, normalize: bool
) -> _EpochResult:
    import tiledbsoma
    import torch
    from tiledbsoma_ml import ExperimentDataset

    exp = tiledbsoma.Experiment.open(soma_path)
    query = None
    try:
        query = exp.axis_query("RNA")
        dataset = ExperimentDataset(
            query=query,
            layer_name="data",
            batch_size=batch_size,
            shuffle=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=0
        )

        n_batches = 0
        n_cells = 0
        for batch in loader:
            # ExperimentDataset returns (X_tensor, obs_tensor) tuple
            # or a single tensor depending on version
            if isinstance(batch, (tuple, list)):
                X = batch[0]
            elif isinstance(batch, dict):
                X = batch.get("X", batch.get("soma_data", next(iter(batch.values()))))
            else:
                X = batch
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            if hasattr(X, "toarray"):
                X = X.toarray()
            if X.ndim < 2:
                # Skip malformed batches
                n_batches += 1
                n_cells += len(X) if hasattr(X, "__len__") else batch_size
                continue
            X = np.asarray(X, dtype=np.float32)
            if hvg and X.shape[1] > QUERY_N_HVGS:
                X = X[:, :QUERY_N_HVGS]
            if normalize:
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                X = X / row_sums * 1e4
                X = np.log1p(X)
            n_batches += 1
            n_cells += X.shape[0]
    finally:
        if query is not None:
            query.close()
        exp.close()

    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _run_slaf_epoch(
    slaf_path: str, batch_size: int, hvg: bool, normalize: bool
) -> _EpochResult:
    """Iterate ``SLAFDataLoader`` (geneformer tokenizer) for one epoch.

    SLAF's ``SLAFDataLoader`` returns tokenized batches (``input_ids``,
    ``attention_mask``, ``cell_ids``) rather than raw dense ``X``. Counting
    batches and cells is still meaningful for throughput, but HVG / normalize
    are inherent to the tokenizer's gene-ranking output; we log the scenario
    flags in the result for provenance.
    """
    from slaf import SLAFArray
    from slaf.ml import SLAFDataLoader

    slaf_array = SLAFArray(slaf_path)
    max_genes = QUERY_N_HVGS if hvg else 2048
    loader = SLAFDataLoader(
        slaf_array=slaf_array,
        tokenizer_type="geneformer",
        batch_size=batch_size,
        max_genes=max_genes,
    )

    n_batches = 0
    n_cells = 0
    for batch in loader:
        input_ids = batch.get("input_ids") if isinstance(batch, dict) else None
        if input_ids is None:
            # Defensive: unexpected batch shape
            n_batches += 1
            n_cells += batch_size
            continue
        n_batches += 1
        n_cells += int(input_ids.shape[0])

    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _ttfb_slaf(slaf_path: str, batch_size: int, hvg: bool) -> None:
    from slaf import SLAFArray
    from slaf.ml import SLAFDataLoader

    slaf_array = SLAFArray(slaf_path)
    loader = SLAFDataLoader(
        slaf_array=slaf_array,
        tokenizer_type="geneformer",
        batch_size=batch_size,
        max_genes=QUERY_N_HVGS if hvg else 2048,
    )
    for _ in loader:
        break


def _run_shardad_epoch(
    shad_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    """Shuffled row-slice loader on a shardad archive.

    shardad has no native batched DataLoader, so the honest "loader you'd build
    on shardad" is a shuffled random-access reader: permute cell ids, slice into
    batches, materialize each batch via ``arch[idx].to_anndata`` and apply the
    same HVG / normalize as the other loaders. ``n_workers=1`` on the per-batch
    read is deliberate — shardad's parallel subset read spawns a fresh process
    pool per call (~1-2 s), which would dominate a per-batch loop; serial is the
    representative random-access cost. (shardad's `iter_group_shards` streaming
    path needs a grouped archive, which the ML datasets are not.)
    """
    from shardad import ShardedArchive

    arch = ShardedArchive(shad_path)
    n_obs = int(arch.n_obs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_obs)

    n_batches = 0
    n_cells = 0
    for start in range(0, n_obs, batch_size):
        # sort within-batch ids for read locality (order within a batch is
        # irrelevant to throughput; the batch membership is unchanged).
        idx = np.sort(perm[start : start + batch_size])
        adata = arch[idx].to_anndata(container="csr", n_workers=1)
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if hvg and X.shape[1] > QUERY_N_HVGS:
            X = X[:, :QUERY_N_HVGS]
        if normalize:
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = np.log1p(X / row_sums * 1e4)
        n_batches += 1
        n_cells += X.shape[0]
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _ttfb_shardad(shad_path: str, batch_size: int, seed: int = RANDOM_SEED) -> None:
    from shardad import ShardedArchive

    arch = ShardedArchive(shad_path)
    n_obs = int(arch.n_obs)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.permutation(n_obs)[:batch_size])
    _ = arch[idx].to_anndata(container="csr", n_workers=1).X


def _run_cellstream_epoch(
    cellstream_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    """Shuffled row-gather loader on a cellstream store.

    cellstream is a random-access gather store with no built-in DataLoader, so the
    honest "loader you'd build on it" mirrors the shardad path: permute cell ids,
    slice into batches, gather each batch via ``store.gather_rows`` and apply the
    same HVG / normalize as the other loaders.
    """
    import cellstream

    store = cellstream.open(cellstream_path)
    try:
        n_obs = int(store.n_obs)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_obs)

        n_batches = 0
        n_cells = 0
        for start in range(0, n_obs, batch_size):
            # sort within-batch ids for read locality (batch membership unchanged).
            idx = np.sort(perm[start : start + batch_size])
            X = store.gather_rows(idx)
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float32)
            if hvg and X.shape[1] > QUERY_N_HVGS:
                X = X[:, :QUERY_N_HVGS]
            if normalize:
                row_sums = X.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                X = np.log1p(X / row_sums * 1e4)
            n_batches += 1
            n_cells += X.shape[0]
        return _EpochResult(n_batches=n_batches, n_cells=n_cells)
    finally:
        store.close()


def _ttfb_cellstream(cellstream_path: str, batch_size: int, seed: int = RANDOM_SEED) -> None:
    import cellstream

    store = cellstream.open(cellstream_path)
    try:
        n_obs = int(store.n_obs)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.permutation(n_obs)[:batch_size])
        _ = store.gather_rows(idx)
    finally:
        store.close()


def _run_scdataloader_epoch(
    h5ad_path: str, batch_size: int, hvg: bool, normalize: bool
) -> _EpochResult:
    import anndata
    import torch
    from scdataloader import SimpleAnnDataset
    from torch.utils.data import DataLoader

    adata = anndata.read_h5ad(h5ad_path)
    dataset = SimpleAnnDataset(adata)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    n_batches = 0
    n_cells = 0
    for batch in loader:
        if isinstance(batch, torch.Tensor):
            X = batch.numpy()
        elif isinstance(batch, dict):
            X = batch.get("X", next(iter(batch.values())))
            if isinstance(X, torch.Tensor):
                X = X.numpy()
        else:
            X = np.asarray(batch)
        X = np.asarray(X, dtype=np.float32)
        if hvg and X.ndim == 2 and X.shape[1] > QUERY_N_HVGS:
            X = X[:, :QUERY_N_HVGS]
        if normalize and X.ndim == 2:
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums * 1e4
            X = np.log1p(X)
        n_batches += 1
        n_cells += X.shape[0] if X.ndim >= 1 else batch_size

    del adata
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


# ---------------------------------------------------------------------------
# TTFB functions (construct loader → get first batch → return)
# ---------------------------------------------------------------------------


def _ttfb_scx(path: str, batch_size: int, hvg: bool, normalize: bool, seed: int) -> None:
    import pyscx

    hvg_indices = list(range(QUERY_N_HVGS)) if hvg else None
    ds = pyscx.TrainingDataset(
        path,
        batch_size=batch_size,
        hvg_indices=hvg_indices,
        normalize=normalize,
        log1p=normalize,
        seed=seed,
        max_memory_mb=_scx_memory_budget_mb(),
    )
    for _ in ds:
        break


def _ttfb_anndata(
    h5ad_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> None:
    import anndata

    adata = anndata.read_h5ad(h5ad_path)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(adata.n_obs)
    batch_idx = indices[:batch_size]
    X = adata.X[batch_idx]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32, copy=False)
    if hvg:
        X = X[:, :QUERY_N_HVGS]
    if normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = X / row_sums * 1e4
        np.log1p(X, out=X)
    del adata


def _ttfb_soma(soma_path: str, batch_size: int) -> None:
    import tiledbsoma
    import torch
    from tiledbsoma_ml import ExperimentDataset

    exp = tiledbsoma.Experiment.open(soma_path)
    query = None
    try:
        query = exp.axis_query("RNA")
        dataset = ExperimentDataset(
            query=query, layer_name="data", batch_size=batch_size, shuffle=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
        for _ in loader:
            break
    finally:
        if query is not None:
            query.close()
        exp.close()


def _ttfb_scdataloader(h5ad_path: str, batch_size: int) -> None:
    import anndata
    from scdataloader import SimpleAnnDataset
    from torch.utils.data import DataLoader

    adata = anndata.read_h5ad(h5ad_path)
    dataset = SimpleAnnDataset(adata)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in loader:
        break
    del adata


# ---------------------------------------------------------------------------
# GPU training scenario (SCX-only)
# ---------------------------------------------------------------------------


@dataclass
class _GpuEpochResult:
    """Result of a GPU training epoch."""

    n_batches: int = 0
    n_cells: int = 0
    wall_s: float = 0.0
    batches_per_sec: float = 0.0
    cells_per_sec: float = 0.0
    avg_gpu_util_pct: float = 0.0
    gpu_util_samples: int = 0


def _build_scvi_vae(n_input: int):
    """Build a scVI-equivalent VAE on cuda:0."""
    import torch
    import torch.nn as nn

    class ScviVAE(nn.Module):
        def __init__(self, n_in, n_latent=128, n_hidden=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_in, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.z_mean = nn.Linear(n_hidden, n_latent)
            self.z_var = nn.Linear(n_hidden, n_latent)
            self.decoder = nn.Sequential(
                nn.Linear(n_latent, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
            )
            self.px_rate = nn.Linear(n_hidden, n_in)

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.z_mean(h), self.z_var(h)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
            rate = torch.exp(self.px_rate(self.decoder(z)))
            recon = torch.mean(rate - x * torch.log(rate + 1e-8))
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return recon + kl

    return ScviVAE(n_input)


def _parse_nvidia_smi_log(log_path: Path) -> list[int]:
    """Parse nvidia-smi dmon log and return GPU utilization samples."""
    gpu_utils: list[int] = []
    if not log_path.exists():
        return gpu_utils
    for line in log_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                gpu_utils.append(int(parts[1]))
            except ValueError:
                pass
    return gpu_utils


def _run_gpu_train_epoch(
    scx_path: str, batch_size: int, seed: int
) -> _GpuEpochResult:
    """Run a full training epoch on GPU and measure utilization."""
    import pyscx
    import torch

    device = torch.device("cuda:0")
    n_hvg = QUERY_N_HVGS

    model = _build_scvi_vae(n_hvg).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = pyscx.TrainingDataset(
        scx_path,
        batch_size=batch_size,
        hvg_indices=list(range(n_hvg)),
        normalize=True,
        log1p=True,
        seed=seed,
        max_memory_mb=_scx_memory_budget_mb(),
    )

    # Warmup epoch
    for batch in ds:
        X = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
        loss = model(X)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Start nvidia-smi monitoring
    smi_log = Path(tempfile.gettempdir()) / f"nvidia_smi_log_{os.getpid()}.csv"
    smi_proc = subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-f", str(smi_log)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)  # let nvidia-smi start

    # Timed training epoch
    n_batches = 0
    n_cells = 0
    try:
        # Re-create dataset for fresh epoch
        ds = pyscx.TrainingDataset(
            scx_path,
            batch_size=batch_size,
            hvg_indices=list(range(n_hvg)),
            normalize=True,
            log1p=True,
            seed=seed + 1,
            max_memory_mb=_scx_memory_budget_mb(),
        )
        t0 = time.perf_counter()
        for batch in ds:
            X = torch.from_numpy(batch["X"]).to(device, non_blocking=True)
            loss = model(X)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_batches += 1
            n_cells += batch["X"].shape[0]
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0

        time.sleep(2)  # final GPU samples
    finally:
        smi_proc.terminate()
        smi_proc.wait()

    del model, optimizer
    torch.cuda.empty_cache()

    # Parse GPU utilization
    gpu_utils = _parse_nvidia_smi_log(smi_log)
    smi_log.unlink(missing_ok=True)

    # Trim startup/shutdown noise
    trimmed = gpu_utils[1:-1] if len(gpu_utils) > 2 else gpu_utils
    avg_util = sum(trimmed) / len(trimmed) if trimmed else 0.0

    bps = n_batches / wall_s if wall_s > 0 else 0.0
    cps = n_cells / wall_s if wall_s > 0 else 0.0

    return _GpuEpochResult(
        n_batches=n_batches,
        n_cells=n_cells,
        wall_s=wall_s,
        batches_per_sec=bps,
        cells_per_sec=cps,
        avg_gpu_util_pct=avg_util,
        gpu_util_samples=len(gpu_utils),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the ML data loader throughput benchmark.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to benchmark against.
    format_variant : FormatVariant
        Determines which loader to use (SCX, AnnData, SOMA, scDataLoader).
    n_runs : int
        Number of timed epoch iterations per scenario (after warm-up).
    cold_cache : bool
        If True, drop OS page caches before each timed run.
    converted_path : Path | None
        Pre-converted file path (optional).

    Returns
    -------
    BenchmarkResult | None
        Structured result with per-run timings across scenarios, or None
        if the format is not applicable to ML loader benchmarks.
    """
    loader_type = _resolve_loader(format_variant.key)
    if loader_type is None:
        return None

    if not _loader_available(loader_type):
        logger.warning(
            "Skipping ml_loader for %s — required dependencies not installed",
            format_variant.key,
        )
        return None

    h5ad_path = dataset.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. Run dataset preparation first."
        )

    gpu_available = _HAS_TORCH and torch.cuda.is_available()

    result = BenchmarkResult(
        benchmark="ml_loader",
        format=format_variant.key,
        dataset=dataset.name,
        scenario={
            "name": "default",
            "batch_size": ML_BATCH_SIZE,
            "n_hvgs": QUERY_N_HVGS,
            "device": "gpu" if gpu_available else "cpu",
        },
        metadata={
            "batch_size": ML_BATCH_SIZE,
            "n_hvgs": QUERY_N_HVGS,
            "n_warmup_epochs": N_WARMUP_RUNS,
            "n_ttfb_runs": _N_TTFB_RUNS,
            "cold_cache": cold_cache,
            "loader_type": loader_type,
            "gpu_available": gpu_available,
        },
    )

    # -- Resolve data path --
    _cleanup = None
    if loader_type == "scx":
        if converted_path is not None and Path(converted_path).exists():
            data_path = str(converted_path)
        else:
            try:
                persistent = dataset.path_for_format(format_variant.key)
                if persistent.exists():
                    data_path = str(persistent)
                else:
                    raise FileNotFoundError
            except (ValueError, FileNotFoundError):
                _cleanup = tempfile.TemporaryDirectory(
                    prefix=f"scx_bench_{dataset.name}_"
                )
                out = Path(_cleanup.name) / f"{dataset.name}.{format_variant.key}"
                logger.info("Converting %s -> %s", h5ad_path.name, out)
                runner = make_runner(format_variant)
                runner.convert_from_h5ad(h5ad_path, out)
                data_path = str(out)
    elif loader_type == "anndata":
        data_path = str(h5ad_path)
    elif loader_type == "scdataloader":
        gzip_path = dataset.h5ad_gzip_path
        data_path = str(gzip_path if gzip_path.exists() else h5ad_path)
    elif loader_type == "soma":
        soma_path = dataset.soma_path
        if not soma_path.exists():
            logger.warning(
                "SOMA path not found: %s — skipping ml_loader for tiledb_soma",
                soma_path,
            )
            return None
        data_path = str(soma_path)
    elif loader_type == "slaf":
        if converted_path is not None and Path(converted_path).exists():
            data_path = str(converted_path)
        else:
            persistent = dataset.slaf_path
            if persistent.exists():
                data_path = str(persistent)
            else:
                _cleanup = tempfile.TemporaryDirectory(
                    prefix=f"scx_bench_slaf_{dataset.name}_"
                )
                out = Path(_cleanup.name) / f"{dataset.name}.slaf"
                logger.info("Converting %s -> %s", h5ad_path.name, out)
                runner = make_runner(format_variant)
                runner.convert_from_h5ad(h5ad_path, out)
                data_path = str(out)
    elif loader_type == "shardad":
        if converted_path is not None and Path(converted_path).exists():
            data_path = str(converted_path)
        else:
            persistent = dataset.shardad_path
            if persistent.exists():
                data_path = str(persistent)
            else:
                _cleanup = tempfile.TemporaryDirectory(
                    prefix=f"scx_bench_shardad_{dataset.name}_"
                )
                out = Path(_cleanup.name) / f"{dataset.name}.shad"
                logger.info("Converting %s -> %s", h5ad_path.name, out)
                runner = make_runner(format_variant)
                runner.convert_from_h5ad(h5ad_path, out)
                data_path = str(out)
    elif loader_type == "cellstream":
        if converted_path is not None and Path(converted_path).exists():
            data_path = str(converted_path)
        else:
            persistent = dataset.cellstream_path
            if persistent.exists():
                data_path = str(persistent)
            else:
                _cleanup = tempfile.TemporaryDirectory(
                    prefix=f"scx_bench_cellstream_{dataset.name}_"
                )
                out = Path(_cleanup.name) / f"{dataset.name}.cellstream"
                logger.info("Converting %s -> %s", h5ad_path.name, out)
                runner = make_runner(format_variant)
                runner.convert_from_h5ad(h5ad_path, out)
                data_path = str(out)
    else:
        return None

    # Record file size
    dp = Path(data_path)
    if dp.is_file():
        result.file_size_bytes = dp.stat().st_size
    elif dp.is_dir():
        total = 0
        for dirpath, _, filenames in os.walk(dp):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                if os.path.isfile(fp):
                    total += os.path.getsize(fp)
        result.file_size_bytes = total

    scenario_summary: dict[str, dict[str, Any]] = {}

    try:
        # ---------------------------------------------------------------
        # Data scenarios (CPU)
        # ---------------------------------------------------------------
        for scenario_name, hvg, normalize in _SCENARIOS:
            logger.info(
                "--- Scenario: %s (format=%s, dataset=%s) ---",
                scenario_name,
                format_variant.key,
                dataset.name,
            )

            # Build the epoch function and TTFB function for this loader
            if loader_type == "scx":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_scx_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize, RANDOM_SEED
                )
                ttfb_fn = lambda hvg=hvg, normalize=normalize: _ttfb_scx(
                    data_path, ML_BATCH_SIZE, hvg, normalize, RANDOM_SEED
                )
            elif loader_type == "anndata":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_anndata_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize, RANDOM_SEED
                )
                ttfb_fn = lambda hvg=hvg, normalize=normalize: _ttfb_anndata(
                    data_path, ML_BATCH_SIZE, hvg, normalize, RANDOM_SEED
                )
            elif loader_type == "soma":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_soma_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize
                )
                ttfb_fn = lambda: _ttfb_soma(data_path, ML_BATCH_SIZE)
            elif loader_type == "scdataloader":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_scdataloader_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize
                )
                ttfb_fn = lambda: _ttfb_scdataloader(data_path, ML_BATCH_SIZE)
            elif loader_type == "slaf":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_slaf_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize
                )
                ttfb_fn = lambda hvg=hvg: _ttfb_slaf(data_path, ML_BATCH_SIZE, hvg)
            elif loader_type == "shardad":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_shardad_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize, RANDOM_SEED
                )
                ttfb_fn = lambda: _ttfb_shardad(data_path, ML_BATCH_SIZE)
            elif loader_type == "cellstream":
                epoch_fn = lambda hvg=hvg, normalize=normalize: _run_cellstream_epoch(
                    data_path, ML_BATCH_SIZE, hvg, normalize, RANDOM_SEED
                )
                ttfb_fn = lambda: _ttfb_cellstream(data_path, ML_BATCH_SIZE)
            else:
                continue

            # Warmup
            try:
                for _ in range(N_WARMUP_RUNS):
                    logger.info("  Warmup epoch")
                    epoch_fn()
                    gc.collect()
            except Exception as e:
                logger.error("  Warmup failed for scenario %s: %s", scenario_name, e)
                scenario_summary[scenario_name] = {"error": str(e)}
                continue

            # TTFB
            try:
                ttfb_times = _measure_ttfb(ttfb_fn, _N_TTFB_RUNS)
                ttfb_median = statistics.median(ttfb_times)
            except Exception as e:
                logger.warning("  TTFB measurement failed: %s", e)
                ttfb_times = []
                ttfb_median = None

            # Timed runs
            run_bps: list[float] = []
            run_cps: list[float] = []
            run_rss: list[float] = []

            for i in range(n_runs):
                if cold_cache:
                    try:
                        subprocess.run(["sync"], check=False)
                        with open("/proc/sys/vm/drop_caches", "w") as f:
                            f.write("3\n")
                    except (PermissionError, OSError):
                        logger.warning(
                            "Could not drop caches (permission denied). "
                            "Cold-cache benchmark will run with warm cache."
                        )

                logger.info("  Timed run %d/%d", i + 1, n_runs)
                try:
                    epoch_result, wall_s, user_s, sys_s, peak_rss = _timed_epoch(
                        epoch_fn
                    )
                except Exception as e:
                    logger.error("  Run %d failed: %s", i + 1, e)
                    continue

                bps = (
                    epoch_result.n_batches / wall_s if wall_s > 0 else 0.0
                )
                cps = (
                    epoch_result.n_cells / wall_s if wall_s > 0 else 0.0
                )

                # Per-scenario keys are emitted only on this scenario's runs
                # (sparse), so the gate's median over runs[].extra[metric]
                # naturally focuses on one scenario per threshold entry.
                # TTFB is added to the first run only — sparse but present.
                run_extra: dict[str, Any] = {
                    "scenario": scenario_name,
                    "n_batches": epoch_result.n_batches,
                    "n_cells": epoch_result.n_cells,
                    "batches_per_sec": round(bps, 1),
                    "cells_per_sec": round(cps, 0),
                    f"batches_per_sec__{scenario_name}": round(bps, 1),
                    f"cells_per_sec__{scenario_name}": round(cps, 0),
                }
                if i == 0 and ttfb_median is not None:
                    run_extra[f"ttfb_median_s__{scenario_name}"] = round(
                        ttfb_median, 4
                    )

                result.add_run(
                    wall_s=wall_s,
                    user_s=user_s,
                    sys_s=sys_s,
                    peak_rss_mb=peak_rss,
                    **run_extra,
                )

                run_bps.append(bps)
                run_cps.append(cps)
                run_rss.append(peak_rss)

                logger.info(
                    "    wall=%.3fs  bps=%.1f  rss=%.1fMB",
                    wall_s,
                    bps,
                    peak_rss,
                )

            # Scenario summary
            summary: dict[str, Any] = {"n_runs": len(run_bps)}
            if run_bps:
                summary["median_batches_per_sec"] = round(statistics.median(run_bps), 1)
                summary["median_cells_per_sec"] = round(statistics.median(run_cps), 0)
                summary["median_peak_rss_mb"] = round(statistics.median(run_rss), 1)
            if ttfb_median is not None:
                summary["ttfb_median_s"] = round(ttfb_median, 4)
                summary["ttfb_all_s"] = [round(t, 4) for t in ttfb_times]

            scenario_summary[scenario_name] = summary
            gc.collect()

        # ---------------------------------------------------------------
        # GPU training scenario (SCX-only)
        # ---------------------------------------------------------------
        if loader_type == "scx" and gpu_available:
            logger.info(
                "--- Scenario: gpu_train (format=%s, dataset=%s) ---",
                format_variant.key,
                dataset.name,
            )
            try:
                gpu_results: list[_GpuEpochResult] = []
                for i in range(n_runs):
                    logger.info("  GPU training run %d/%d", i + 1, n_runs)
                    gpu_res = _run_gpu_train_epoch(
                        data_path, ML_BATCH_SIZE, RANDOM_SEED + i
                    )
                    gpu_results.append(gpu_res)

                    rss = _current_rss_mb()
                    result.add_run(
                        wall_s=gpu_res.wall_s,
                        peak_rss_mb=rss,
                        scenario="gpu_train",
                        n_batches=gpu_res.n_batches,
                        n_cells=gpu_res.n_cells,
                        batches_per_sec=round(gpu_res.batches_per_sec, 1),
                        cells_per_sec=round(gpu_res.cells_per_sec, 0),
                        avg_gpu_util_pct=round(gpu_res.avg_gpu_util_pct, 1),
                        gpu_util_samples=gpu_res.gpu_util_samples,
                        model="scVI-equivalent VAE (2L, 128h, 128z)",
                        batches_per_sec__gpu_train=round(gpu_res.batches_per_sec, 1),
                        cells_per_sec__gpu_train=round(gpu_res.cells_per_sec, 0),
                        avg_gpu_util_pct__gpu_train=round(gpu_res.avg_gpu_util_pct, 1),
                    )

                    logger.info(
                        "    wall=%.3fs  bps=%.1f  gpu_util=%.1f%%",
                        gpu_res.wall_s,
                        gpu_res.batches_per_sec,
                        gpu_res.avg_gpu_util_pct,
                    )

                if gpu_results:
                    med_bps = statistics.median(r.batches_per_sec for r in gpu_results)
                    med_cps = statistics.median(r.cells_per_sec for r in gpu_results)
                    avg_util = statistics.median(
                        r.avg_gpu_util_pct for r in gpu_results
                    )
                    scenario_summary["gpu_train"] = {
                        "n_runs": len(gpu_results),
                        "median_batches_per_sec": round(med_bps, 1),
                        "median_cells_per_sec": round(med_cps, 0),
                        "avg_gpu_util_pct": round(avg_util, 1),
                        "target_gpu_util_pct": 85,
                        "pass": avg_util >= 85,
                    }
            except Exception as e:
                logger.error("  GPU training failed: %s", e)
                scenario_summary["gpu_train"] = {"error": str(e)}
            gc.collect()

        # ---------------------------------------------------------------
        # `num_workers > 0` DataLoader scenarios (SCX-only)
        # ---------------------------------------------------------------
        if loader_type == "scx" and _HAS_TORCH and dataset.n_obs > _WORKERS2_MAX_OBS:
            logger.info(
                "Skipping num_workers>0 scenarios for %s (n_obs=%d > %d): they time "
                "out at this scale and would discard the whole result. Set "
                "SCX_BENCH_WORKERS2_MAX_OBS to override.",
                dataset.name,
                dataset.n_obs,
                _WORKERS2_MAX_OBS,
            )
            for scenario_name in (
                "pyscx_training_dataset_workers2",
                "pyscx_training_dataset_workers2_persistent",
            ):
                scenario_summary[scenario_name] = {
                    "skipped": f"n_obs {dataset.n_obs} > {_WORKERS2_MAX_OBS}"
                }
        elif loader_type == "scx" and _HAS_TORCH:
            for scenario_name, persistent_workers, n_epochs_per_run in (
                ("pyscx_training_dataset_workers2", False, 1),
                ("pyscx_training_dataset_workers2_persistent", True, 2),
            ):
                logger.info(
                    "--- Scenario: %s (format=%s, dataset=%s) ---",
                    scenario_name,
                    format_variant.key,
                    dataset.name,
                )
                # Use the `hvg_norm` configuration so this scenario's
                # throughput is directly comparable to the existing
                # `batches_per_sec__hvg_norm` floor under num_workers=0.
                hvg, normalize = True, True
                epoch_fn = (
                    lambda hvg=hvg,
                    normalize=normalize,
                    persistent_workers=persistent_workers,
                    n_epochs_per_run=n_epochs_per_run: _run_scx_dataloader_workers_epoch(
                        data_path,
                        ML_BATCH_SIZE,
                        hvg,
                        normalize,
                        RANDOM_SEED,
                        num_workers=2,
                        persistent_workers=persistent_workers,
                        n_epochs=n_epochs_per_run,
                    )
                )

                # Warmup
                try:
                    for _ in range(N_WARMUP_RUNS):
                        logger.info("  Warmup epoch")
                        epoch_fn()
                        gc.collect()
                except Exception as e:
                    logger.error(
                        "  Warmup failed for scenario %s: %s",
                        scenario_name,
                        e,
                    )
                    scenario_summary[scenario_name] = {"error": str(e)}
                    continue

                # Timed runs (no separate TTFB measurement — the
                # DataLoader-spawn overhead is part of "first batch latency"
                # in the workers2 scenario by definition; tracking it
                # separately would double-count the same delay).
                run_bps_w2: list[float] = []
                run_cps_w2: list[float] = []
                run_rss_w2: list[float] = []

                for i in range(n_runs):
                    logger.info("  Timed run %d/%d", i + 1, n_runs)
                    try:
                        epoch_result, wall_s, user_s, sys_s, peak_rss = (
                            _timed_epoch(epoch_fn)
                        )
                    except Exception as e:
                        logger.error("  Run %d failed: %s", i + 1, e)
                        continue

                    # Normalise by `n_epochs_per_run` so the metric is
                    # per-epoch throughput regardless of how many epochs
                    # the persistent variant ran (apples-to-apples vs
                    # the non-persistent variant).
                    bps_per_epoch = (
                        (epoch_result.n_batches / n_epochs_per_run) / wall_s
                        if wall_s > 0
                        else 0.0
                    )
                    cps_per_epoch = (
                        (epoch_result.n_cells / n_epochs_per_run) / wall_s
                        if wall_s > 0
                        else 0.0
                    )
                    # The wall-clock for the persistent variant covers
                    # multiple epochs; report wall_s_per_epoch for sanity.
                    wall_s_per_epoch = wall_s / n_epochs_per_run

                    run_extra = {
                        "scenario": scenario_name,
                        "n_batches": epoch_result.n_batches,
                        "n_cells": epoch_result.n_cells,
                        "n_epochs_per_run": n_epochs_per_run,
                        "wall_s_per_epoch": round(wall_s_per_epoch, 4),
                        "batches_per_sec": round(bps_per_epoch, 1),
                        "cells_per_sec": round(cps_per_epoch, 0),
                        f"batches_per_sec__{scenario_name}": round(
                            bps_per_epoch, 1
                        ),
                        f"cells_per_sec__{scenario_name}": round(cps_per_epoch, 0),
                        f"peak_rss_mb__{scenario_name}": round(peak_rss, 1),
                    }

                    result.add_run(
                        wall_s=wall_s,
                        user_s=user_s,
                        sys_s=sys_s,
                        peak_rss_mb=peak_rss,
                        **run_extra,
                    )

                    run_bps_w2.append(bps_per_epoch)
                    run_cps_w2.append(cps_per_epoch)
                    run_rss_w2.append(peak_rss)

                    logger.info(
                        "    wall=%.3fs/epoch  bps=%.1f  rss=%.1fMB",
                        wall_s_per_epoch,
                        bps_per_epoch,
                        peak_rss,
                    )

                summary_w2: dict[str, Any] = {"n_runs": len(run_bps_w2)}
                if run_bps_w2:
                    summary_w2["median_batches_per_sec"] = round(
                        statistics.median(run_bps_w2), 1
                    )
                    summary_w2["median_cells_per_sec"] = round(
                        statistics.median(run_cps_w2), 0
                    )
                    summary_w2["median_peak_rss_mb"] = round(
                        statistics.median(run_rss_w2), 1
                    )
                summary_w2["num_workers"] = 2
                summary_w2["persistent_workers"] = persistent_workers
                summary_w2["n_epochs_per_run"] = n_epochs_per_run
                scenario_summary[scenario_name] = summary_w2
                gc.collect()

    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    result.metadata["scenario_summary"] = scenario_summary

    logger.info(
        "Benchmark complete: ml_loader / %s / %s — %d scenarios",
        format_variant.key,
        dataset.name,
        len(scenario_summary),
    )
    return result
