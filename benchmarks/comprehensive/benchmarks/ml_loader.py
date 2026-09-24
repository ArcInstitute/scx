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
import functools
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

from benchmarks.comprehensive.cache_control import drop_file_cache
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    ML_BATCH_SIZE,
    N_WARMUP_RUNS,
    QUERY_N_HVGS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.data_wait import (
    data_wait_fraction,
    steady_state_wait,
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
    _SCX_KEYS | {"h5ad_none", "h5ad_gzip", "tiledb_soma", "slaf"}
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
    (1M × 61,497): the loader's memory-budget auto-tune shrinks
    ``shard_group_size`` and, historically, ``batch_size`` to its 64 minimum,
    which both makes batches/sec incomparable across codecs (a larger-on-disk
    codec tips over the threshold first) and inflates the epoch to ~16× more
    tiny batches (a cause of the census `workers2` time-outs). Use most of
    whatever ``--mem`` the SLURM job got. Returns ``None`` off-SLURM so
    local/CI runs keep the built-in default.

    The ``batch_size`` half of that rationale is **historical**: it was caused
    by the sequential model counting the whole mmap'd file against the budget,
    so any file above the 4096 MB adaptive cap exceeded it on that term alone.
    ORG-9.10-5 removed the mmap term. This scaling is kept because it still
    protects ``shard_group_size`` (hence shuffle entropy), and because dropping
    it in the same change would confound the A/B that validates the change.
    Retiring it is a follow-up, once a capture at the default budget exists.
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

# Dataset -> {arm name: (obs column, its cardinality)} for the OPT-LOADER-6
# cardinality arm. Only datasets listed here run it.
#
# The column choice is the whole arm. `obs_to_pydict`
# (`scx-loader/src/python/convert.rs`) rebuilds a categorical column's category
# list as fresh `PyUnicode` objects **per batch**, and hands an Int64 column
# straight to `PyArray1::from_vec` — so the cost tracks the number of
# *categories*, not the number of rows. census_1m's `soma_joinid` has a million
# distinct values and is `int64`: it takes the cheap branch and would have
# measured nothing. `observation_joinid` is the categorical, with 957,955
# categories (verified on `census_1m_auto.scx` via
# `Experiment.obs_categorical`); `sex` has 3 and is the control.
#
# Cardinalities are recorded here, and into `runs[].extra`, so a capture says
# what it measured rather than requiring a reader to re-probe the fixture.
# The arm's premise, as a checkable bound rather than a comment. The point is a
# *separation* in cardinality, and only the high side needs to be large: at
# 957,955 vs 3 the declared pair clears both of these by three orders of
# magnitude, so a fixture that fails them has lost the premise rather than
# merely drifted.
#
# Observing the counts is not enough on its own. A reconvert that keeps
# `observation_joinid` categorical but collapses it to, say, 1,000 categories
# would leave the arm green while destroying what it measures — the per-batch
# `PyUnicode` rebuild is proportional to the category count — and a *faster*
# high-cardinality arm would read as an improvement. So a collapse fails the
# preflight instead of warning.
_OBS_HIGHCARD_MIN_CATEGORIES = 100_000
_OBS_CARDINALITY_MIN_RATIO = 1_000

_OBS_CARDINALITY_SPEC: dict[str, dict[str, tuple[str, int]]] = {
    "census_1m": {
        "raw_obs_lowcard": ("sex", 3),
        "raw_obs_highcard": ("observation_joinid", 957_955),
    },
}


def _attach_to_scenario_runs(
    result: BenchmarkResult, scenario: str, extras: dict[str, Any],
) -> int:
    """Merge *extras* into every run of *scenario*; return how many got them.

    For a value derived *after* both arms have run and which still has to be
    gateable. `_load_current_raw_metric` medians over the runs that carry a
    key, so writing the same value onto each run of one scenario reads
    identically to a dedicated record — without being one.

    That distinction is the point. An earlier version appended a
    `wall_s=0.0, peak_rss_mb=0.0` bookkeeping run, whose comment claimed it
    could not perturb the pooled medians. `BenchmarkResult.median_wall_s` /
    `median_rss_mb` and `capture_baseline._median_rss` take a median over
    *every* run, so a 0.0/0.0 sample drags both down, widens `wall_s_iqr` and
    bumps `n_runs`; a reviewer measured a four-run loader median 11.5 -> 11.0
    with triple the IQR.
    """
    n = 0
    for run_rec in result.runs:
        if run_rec.extra.get("scenario") == scenario:
            run_rec.extra.update(extras)
            n += 1
    return n


def _probe_obs_cardinality(
    path: str, columns: list[str],
) -> tuple[dict[str, int], list[str]]:
    """Observed category count per column, plus the ones that cannot be used.

    Returns ``({column: n_categories}, problems)``. A column is a *problem*
    when it is absent from obs or is not stored as a dictionary — and the
    second half is the point. The arm's whole subject is
    `obs_to_pydict`'s **Categorical** branch; a column that survives a fixture
    reconvert under the same name but as an `Int64` sends the loader down the
    `PyArray1::from_vec` branch instead, so the arm would run, emit its
    metrics, and measure nothing.

    Declared counts used to be hard-coded into `runs[].extra` (`3` and
    `957_955`), which described the fixture as it was rather than as it is.
    The observed count is recorded now, so a reconvert changes the number
    instead of silently invalidating it.

    A preflight rather than a try/except around the arm: `TrainingDataset`
    validates `obs_columns` in its constructor and raises, and catching that
    the way the `gpu_train` arm catches its own failures would turn a broken
    fixture into a scenario with zero runs — a silent missing metric for any
    threshold on it.

    Failing to open the file is itself reported as a problem. The predecessor
    returned "no problems" there, on the reasoning that the arm's own error
    would surface instead — true while the caller needed only the column
    *names*, and a `KeyError: 'sex'` the moment it started indexing the
    observed counts. Widening a function's return domain without auditing its
    caller is the shape a fix round most reliably regresses in; a reviewer
    caught this one.
    """
    counts: dict[str, int] = {}
    problems: list[str] = []
    try:
        import pyscx

        exp = pyscx.open(path)
        try:
            available = set(exp.obs_keys())
            for col in columns:
                if col not in available:
                    problems.append(f"{col}: absent from obs")
                    continue
                try:
                    _codes, categories = exp.obs_categorical(col)
                except Exception as e:  # noqa: BLE001
                    # `obs_categorical` raises on a non-dictionary column,
                    # which is exactly the case that must not pass silently.
                    problems.append(f"{col}: not a categorical column ({e})")
                    continue
                counts[col] = len(categories)
        finally:
            close = getattr(exp, "close", None)
            if close is not None:
                close()
    except Exception as e:  # noqa: BLE001
        logger.warning("  obs-column preflight could not open %s: %s", path, e)
        return {}, [f"could not open {path}: {e}"]
    return counts, problems


def _cardinality_premise_problems(
    spec: dict[str, tuple[str, int]], observed: dict[str, int],
) -> list[str]:
    """Whether the observed counts still support a cardinality comparison.

    Two bounds, both on the *observed* numbers:

    * the high-cardinality column must actually be high-cardinality
      (``>= _OBS_HIGHCARD_MIN_CATEGORIES``), since the cost this arm prices is
      proportional to the category count;
    * the two columns must be separated by ``>= _OBS_CARDINALITY_MIN_RATIO``,
      since the whole design is a ratio between them.

    A reconvert that keeps the column categorical but collapses its dictionary
    passes the type check and fails these — and it is the dangerous case,
    because the collapsed arm is *faster* and would read as an improvement
    against a floor. Recording the observed count is what makes the drift
    visible; refusing here is what stops it being scored.
    """
    high_col, _ = spec["raw_obs_highcard"]
    low_col, _ = spec["raw_obs_lowcard"]
    high = observed.get(high_col, 0)
    low = max(observed.get(low_col, 0), 1)
    problems: list[str] = []
    if high < _OBS_HIGHCARD_MIN_CATEGORIES:
        problems.append(
            f"{high_col}: {high} categories is below the "
            f"{_OBS_HIGHCARD_MIN_CATEGORIES} this arm needs — the per-batch "
            f"category rebuild it prices scales with that count"
        )
    if high < low * _OBS_CARDINALITY_MIN_RATIO:
        problems.append(
            f"{high_col}/{low_col} = {high}/{low} is below the "
            f"{_OBS_CARDINALITY_MIN_RATIO}x separation the comparison needs"
        )
    return problems


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
    return False


# ---------------------------------------------------------------------------
# Per-loader epoch functions
# ---------------------------------------------------------------------------


def _hvg_indices(path: str, hvg: bool) -> list[int] | None:
    """The HVG projection to hand `TrainingDataset`, or `None` for no projection.

    `None` rather than `range(QUERY_N_HVGS)` when the file is narrower than the
    projection, for two reasons that point the same way. All three call sites
    used an unconditional `range(2000)`, which on a narrower file asks the
    loader for column 2000 of 200 and raises `HVG index 200 is out of range` —
    the failure `test_frozen_floor_keys_still_emitted` has been reporting, where
    the whole `hvg_norm` scenario is lost along with the
    `samples_per_sec__hvg_norm` floors keyed to it. And it made this path
    asymmetric with the competitors': `ooc_loader._apply_hvg_norm` guards with
    `X.shape[1] > QUERY_N_HVGS` and simply does not project, so on such a file
    every competitor would have measured an unprojected epoch while SCX errored.
    Skipping the projection is what mirrors them; projecting to the full width
    would still route through the HVG machinery and time something they are not.

    The boundary is `>`, matching the competitor guard exactly, and that is
    where the only behaviour change on a registered dataset lives: the four
    `pert_synth_*` fixtures are exactly 2000 vars, so the old code projected
    all 2000 columns and the new code projects none. That is the intended
    direction — `_apply_hvg_norm` also does nothing at exactly 2000, so the
    two paths now agree there, where before SCX did the projection work and
    the competitors did not. No `ml_loader` / `ooc_loader` floor sits on a
    `pert_synth_*` dataset, so it moves no gated number; the boundary is
    pinned by a test rather than left to this comment.

    Nothing registered is below 2000 (the narrowest real fixture is 5000
    vars), so the crash only ever reached the narrow synthetic fixtures the
    tests build. One function because three call sites drifting apart on which
    of them clamps is how this would come back.
    """
    if not hvg:
        return None
    return list(range(QUERY_N_HVGS)) if _n_vars(path) > QUERY_N_HVGS else None


@functools.lru_cache(maxsize=None)
def _n_vars(path: str) -> int:
    """`n_vars` off the catalog, once per path per process.

    Cached because `_hvg_indices` is called from inside the timed epoch
    functions, and every scenario runs a warmup epoch before the timed ones —
    so the open lands in the warmup and the timed region adds nothing. A
    catalog open is sub-millisecond against epochs measured in seconds, but a
    benchmark should not put even that in the region it is reporting.
    """
    import pyscx

    return int(pyscx.open(path).n_vars)


def _run_scx_epoch(
    path: str, batch_size: int, hvg: bool, normalize: bool, seed: int,
    obs_columns: list[str] | None = None,
) -> _EpochResult:
    """One SCX epoch. ``obs_columns`` projects obs metadata into each batch.

    ``obs_columns=None`` (every scenario but the two ``raw_obs_*`` arms) yields
    ``batch["obs"] == {}``, so the headline ``batches_per_sec__raw`` is a
    pure-X number and stays comparable with the competitor loaders, none of
    which is asked for obs either.
    """
    import pyscx

    hvg_indices = _hvg_indices(path, hvg)
    ds = pyscx.TrainingDataset(
        path,
        batch_size=batch_size,
        hvg_indices=hvg_indices,
        obs_columns=obs_columns,
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

            hvg_indices = _hvg_indices(self.path, self.hvg)
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

    hvg_indices = _hvg_indices(path, hvg)
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
    # Phase-0 gate: how much of the timed epoch the consumer spent blocked in
    # `next(iterator)`. `None` (not 0.0) when no step ran — see
    # `data_wait_fraction`.
    data_wait_s: float = 0.0
    data_wait_fraction: float | None = None
    batch_wait_ms_p50: float | None = None
    batch_wait_ms_p95: float | None = None
    batch_wait_ms_p99: float | None = None
    batch_wait_ms_max: float | None = None
    # The headline `p`: the first `next()` pays tokio spin-up and the first
    # shard decode once per epoch, and folding that into the fraction makes it
    # a function of epoch length rather than of the loader.
    ttfb_s: float | None = None
    data_wait_fraction_steady: float | None = None
    n_steady_steps: int = 0


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
    # One `next()` duration per step. Driven through an explicit iterator
    # rather than `for batch in ds:` purely to get that seam: the fraction of
    # step time a model spends waiting on data is what gates the tier-3 loader
    # work, and a bare `for` leaves nowhere to measure it.
    waits_s: list[float] = []
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
        # `t0` BEFORE `iter(ds)`: `TrainingDataset.__iter__` calls
        # `TrainingPipeline::start_epoch`, which builds the epoch stages. Timing
        # from after it put that cost outside both `wall_s` and `ttfb_s`, while
        # the docs said the first `next()` pays pipeline spin-up. Fold iterator
        # construction into the first wait so `ttfb_s` measures what it claims.
        t0 = time.perf_counter()
        it = iter(ds)
        first_wait_from = t0
        while True:
            w0 = first_wait_from if not waits_s else time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            waits_s.append(time.perf_counter() - w0)
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

    # Percentiles come from `steady`, which computes them on the post-startup
    # slice. Taking them from `wait_percentiles(waits_s)` made `max_ms` equal to
    # `ttfb_s` restated in ms.
    steady = steady_state_wait(waits_s, wall_s)
    return _GpuEpochResult(
        n_batches=n_batches,
        n_cells=n_cells,
        wall_s=wall_s,
        batches_per_sec=bps,
        cells_per_sec=cps,
        avg_gpu_util_pct=avg_util,
        gpu_util_samples=len(gpu_utils),
        data_wait_s=sum(waits_s),
        data_wait_fraction=data_wait_fraction(waits_s, wall_s),
        batch_wait_ms_p50=steady["p50_ms"],
        batch_wait_ms_p95=steady["p95_ms"],
        batch_wait_ms_p99=steady["p99_ms"],
        batch_wait_ms_max=steady["max_ms"],
        ttfb_s=steady["ttfb_s"],
        data_wait_fraction_steady=steady["data_wait_fraction_steady"],
        n_steady_steps=steady["n_steady_steps"],
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
        # High-cardinality obs projection (SCX-only) — OPT-LOADER-6
        #
        # `obs_columns=` was passed nowhere in the suite, so the cost of
        # materialising obs metadata per batch had no instrument at all. It is
        # not a flat cost: `obs_to_pydict` (scx-loader/src/python/convert.rs)
        # hands an Int64 column straight to numpy, but rebuilds a *categorical*
        # column's entire category list as fresh `PyUnicode` objects on every
        # batch. So the quantity of interest is cardinality, and the arm runs
        # twice — a 3-category column and a ~958k-category one — because a
        # single absolute rate cannot separate "obs projection costs
        # something" from "cardinality costs something", which is the whole
        # claim. The ratio is the signal a fix has to move.
        # ---------------------------------------------------------------
        obs_spec = _OBS_CARDINALITY_SPEC.get(dataset.name)
        # `scx_auto` only, not every `_SCX_KEYS` variant. The cost this arm
        # prices is Python-side and codec-independent, so running it on all
        # seven SCX formats would pay 7x for identical information — the same
        # reasoning `build_csc` and `fragment_ops` give for pinning a single
        # trigger. It also keeps the composition justification to one triple:
        # LATEST carries census_1m rows for all seven formats, and the other
        # five would otherwise acquire two deliberately slower scenarios with
        # nothing explaining the shift.
        if format_variant.key == "scx_auto" and obs_spec is not None:
            observed, problems = _probe_obs_cardinality(
                data_path, [c for c, _n in obs_spec.values()]
            )
            if observed and not problems:
                problems = _cardinality_premise_problems(obs_spec, observed)
            if problems:
                # Recorded, not swallowed into a per-scenario "error" key: an
                # unusable column yields zero runs, and a threshold on a metric
                # no run carries is a missing-metric violation rather than a
                # skip — which is the loud outcome, but only if the reason is
                # findable. See thresholds.yaml's Deferred entry.
                reason = (
                    f"obs cardinality arm cannot run on {data_path}: "
                    f"{problems}. It needs two *categorical* columns of very "
                    f"different cardinality; check the fixture was converted "
                    f"from the census h5ad with obs intact."
                )
                logger.warning("  obs-cardinality arm skipped: %s", reason)
                scenario_summary["raw_obs_cardinality"] = {
                    "applicable": False, "reason": reason,
                }
            else:
                obs_rates: dict[str, float] = {}
                for arm_name, (column, declared) in obs_spec.items():
                    n_categories = observed[column]
                    if n_categories != declared:
                        # Not fatal by itself: the premise that matters is the
                        # observed separation, enforced by
                        # `_cardinality_premise_problems` above, and a drift
                        # that still clears it is real. What is stale is the
                        # spec's number, which should be updated.
                        logger.warning(
                            "  %s: %r has %d categories, spec says %d; "
                            "recording the observed count",
                            arm_name, column, n_categories, declared,
                        )
                    logger.info(
                        "--- Scenario: %s (obs_columns=[%r], %d categories) ---",
                        arm_name, column, n_categories,
                    )
                    # Warm once, untimed. A construction failure here is NOT
                    # caught: `TrainingDataset` validates `obs_columns` up
                    # front and raises `obs column '…' not found in file
                    # (available: …)`, and that is a broken fixture, not an
                    # inapplicable arm.
                    _run_scx_epoch(
                        data_path, ML_BATCH_SIZE, False, False, RANDOM_SEED,
                        obs_columns=[column],
                    )
                    arm_bps: list[float] = []
                    cache_policy = "warm"
                    for _ in range(n_runs):
                        # The main `_SCENARIOS` loop evicts before every timed
                        # epoch, and this arm did not — so a `cold_cache=True`
                        # campaign would have measured a warmed file here and
                        # labelled it cold, with the second arm additionally
                        # riding the first arm's reads.
                        #
                        # `drop_file_cache` (posix_fadvise, per file, returns
                        # its own policy label) rather than the privileged
                        # system-wide `/proc/sys/vm/drop_caches` write the main
                        # loop still uses: unprivileged, targeted, and the
                        # house helper the other data-load benchmarks call. The
                        # main loop's version is pre-existing and untouched.
                        if cold_cache:
                            cache_policy = drop_file_cache(data_path)
                        gc.collect()
                        t0 = time.perf_counter()
                        epoch = _run_scx_epoch(
                            # Fixed `RANDOM_SEED`, matching the `raw` scenario
                            # rather than `gpu_train`'s `RANDOM_SEED + i`: both
                            # obs arms and `raw` must walk the shards in the
                            # same order, or the delta between them is partly
                            # shard-order luck rather than the obs projection —
                            # which is the whole claim.
                            data_path, ML_BATCH_SIZE, False, False,
                            RANDOM_SEED, obs_columns=[column],
                        )
                        wall_s = time.perf_counter() - t0
                        bps = epoch.n_batches / wall_s if wall_s > 0 else 0.0
                        cps = epoch.n_cells / wall_s if wall_s > 0 else 0.0
                        rss = _current_rss_mb()
                        result.add_run(
                            wall_s=wall_s,
                            peak_rss_mb=rss,
                            scenario=arm_name,
                            n_batches=epoch.n_batches,
                            n_cells=epoch.n_cells,
                            obs_column=column,
                            obs_n_categories=n_categories,
                            obs_n_categories_declared=declared,
                            cache_policy=cache_policy,
                            **{
                                f"batches_per_sec__{arm_name}": round(bps, 1),
                                f"cells_per_sec__{arm_name}": round(cps, 0),
                                f"peak_rss_mb__{arm_name}": round(rss, 1),
                            },
                        )
                        arm_bps.append(bps)
                        logger.info(
                            "    wall=%.3fs  bps=%.1f  (%d categories)",
                            wall_s, bps, n_categories,
                        )
                    if arm_bps:
                        obs_rates[arm_name] = statistics.median(arm_bps)
                        scenario_summary[arm_name] = {
                            "n_runs": len(arm_bps),
                            "obs_column": column,
                            "obs_n_categories": n_categories,
                            "cache_policy": cache_policy,
                            "median_batches_per_sec": round(
                                statistics.median(arm_bps), 1
                            ),
                        }
                    gc.collect()
                lo = obs_rates.get("raw_obs_lowcard")
                hi = obs_rates.get("raw_obs_highcard")
                if lo and hi:
                    # >1 means the high-cardinality column is slower, which is
                    # the expected direction; OPT-LOADER-6 drives it toward 1.
                    slowdown = lo / hi
                    # The *ratio* is host-dependent in a way the per-batch
                    # delta is not, so milliseconds per batch is the quantity
                    # to record and to floor.
                    #
                    # Measured through this arm at census_1m, 2 runs of a full
                    # 977-batch epoch: 4.2 -> 4.1 batches/s, **7.86 ms/batch**,
                    # 1.033x. A standalone 40-batch cold probe had reported
                    # +109 ms/batch, which agreed with the review doc's
                    # 60-120 ms/batch estimate and is ~14x too high — 40
                    # batches cannot separate an 8 s construction cost from a
                    # per-batch one. The mechanism is confirmed live (the
                    # 957,955-entry `categories` list is a fresh object every
                    # batch), so ~8 ns per category is simply what it costs:
                    # ~3% of a 238 ms batch, not the several-fold predicted.
                    overhead_ms = (1.0 / hi - 1.0 / lo) * 1000.0
                    scenario_summary["raw_obs_cardinality"] = {
                        "applicable": True,
                        "lowcard_batches_per_sec": round(lo, 1),
                        "highcard_batches_per_sec": round(hi, 1),
                        "highcard_slowdown_vs_lowcard": round(slowdown, 3),
                        "highcard_overhead_ms_per_batch": round(overhead_ms, 2),
                    }
                    # Emitted into `extra` as well as metadata, because only
                    # `extra` is gateable — but attached to the runs that
                    # already exist, NOT to a new zero-wall bookkeeping record.
                    #
                    # An earlier version added such a record and claimed it
                    # "cannot perturb the pooled medians any further than the
                    # two timed arms already do". That is arithmetically false:
                    # `BenchmarkResult.median_wall_s` / `median_rss_mb` and
                    # `capture_baseline._median_rss` take a median over *every*
                    # run, so injecting a 0.0/0.0 sample drags both down, wid-
                    # ens `wall_s_iqr` and bumps `n_runs`. A reviewer measured
                    # a four-run loader median 11.5 -> 11.0 with triple the IQR.
                    #
                    # `_load_current_raw_metric` medians over the runs that
                    # *carry* the key, so writing the same value onto each
                    # high-cardinality run yields the identical gate reading
                    # with no synthetic measurement.
                    attached = _attach_to_scenario_runs(
                        result,
                        "raw_obs_highcard",
                        {
                            "obs_highcard_slowdown_vs_lowcard": round(slowdown, 3),
                            "obs_highcard_overhead_ms_per_batch": round(
                                overhead_ms, 2
                            ),
                            "obs_lowcard_column": obs_spec["raw_obs_lowcard"][0],
                            "obs_highcard_column": obs_spec["raw_obs_highcard"][0],
                        },
                    )
                    logger.info(
                        "  obs cardinality: %.1f -> %.1f batches/s "
                        "(%.2fx slower, +%.1f ms/batch; on %d run(s))",
                        lo, hi, slowdown, overhead_ms, attached,
                    )

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
                        # Phase-0 gate (R1). `n_steps` is `n_batches` above;
                        # the model and batch size are already recorded on
                        # this run and in `result.scenario`.
                        data_wait_fraction__gpu_train=(
                            None
                            if gpu_res.data_wait_fraction is None
                            else round(gpu_res.data_wait_fraction, 5)
                        ),
                        data_wait_s__gpu_train=round(gpu_res.data_wait_s, 4),
                        batch_wait_ms_p50__gpu_train=gpu_res.batch_wait_ms_p50,
                        batch_wait_ms_p95__gpu_train=gpu_res.batch_wait_ms_p95,
                        # p99 and max, not p50/p95 alone: at census scale the
                        # wait is a tail and p95 does not show it.
                        batch_wait_ms_p99__gpu_train=gpu_res.batch_wait_ms_p99,
                        batch_wait_ms_max__gpu_train=gpu_res.batch_wait_ms_max,
                        # The headline `p` for R1 — startup excluded. See
                        # `steady_state_wait` for why the all-steps fraction
                        # above is not the number to quote.
                        data_wait_fraction_steady__gpu_train=(
                            None
                            if gpu_res.data_wait_fraction_steady is None
                            else round(gpu_res.data_wait_fraction_steady, 6)
                        ),
                        ttfb_s__gpu_train=(
                            None if gpu_res.ttfb_s is None else round(gpu_res.ttfb_s, 4)
                        ),
                        n_steady_steps__gpu_train=gpu_res.n_steady_steps,
                    )

                    logger.info(
                        "    wall=%.3fs  bps=%.1f  gpu_util=%.1f%%  "
                        "data_wait=%s steady (%s all-steps; ttfb=%ss; "
                        "p50=%s p95=%s p99=%s max=%s ms over %d steps)",
                        gpu_res.wall_s,
                        gpu_res.batches_per_sec,
                        gpu_res.avg_gpu_util_pct,
                        "n/a"
                        if gpu_res.data_wait_fraction_steady is None
                        else f"{gpu_res.data_wait_fraction_steady:.3%}",
                        "n/a"
                        if gpu_res.data_wait_fraction is None
                        else f"{gpu_res.data_wait_fraction:.1%}",
                        gpu_res.ttfb_s,
                        gpu_res.batch_wait_ms_p50,
                        gpu_res.batch_wait_ms_p95,
                        gpu_res.batch_wait_ms_p99,
                        gpu_res.batch_wait_ms_max,
                        gpu_res.n_batches,
                    )

                if gpu_results:
                    med_bps = statistics.median(r.batches_per_sec for r in gpu_results)
                    med_cps = statistics.median(r.cells_per_sec for r in gpu_results)
                    avg_util = statistics.median(
                        r.avg_gpu_util_pct for r in gpu_results
                    )
                    fractions = [
                        r.data_wait_fraction
                        for r in gpu_results
                        if r.data_wait_fraction is not None
                    ]
                    steadies = [
                        r.data_wait_fraction_steady
                        for r in gpu_results
                        if r.data_wait_fraction_steady is not None
                    ]
                    ttfbs = [r.ttfb_s for r in gpu_results if r.ttfb_s is not None]
                    p50s = [
                        r.batch_wait_ms_p50
                        for r in gpu_results
                        if r.batch_wait_ms_p50 is not None
                    ]
                    p95s = [
                        r.batch_wait_ms_p95
                        for r in gpu_results
                        if r.batch_wait_ms_p95 is not None
                    ]
                    p99s = [
                        r.batch_wait_ms_p99
                        for r in gpu_results
                        if r.batch_wait_ms_p99 is not None
                    ]
                    # The max ACROSS runs, not a median of maxima: the tail is
                    # the subject and a median over three would hide the worst.
                    maxes = [
                        r.batch_wait_ms_max
                        for r in gpu_results
                        if r.batch_wait_ms_max is not None
                    ]
                    scenario_summary["gpu_train"] = {
                        "n_runs": len(gpu_results),
                        "median_batches_per_sec": round(med_bps, 1),
                        "median_cells_per_sec": round(med_cps, 0),
                        "avg_gpu_util_pct": round(avg_util, 1),
                        "target_gpu_util_pct": 85,
                        "pass": avg_util >= 85,
                        # Phase-0 gate (R1): the `p` that decides whether the
                        # tier-3 loader work is worth funding for this regime.
                        # The startup-excluded figure is the headline; the
                        # all-steps one rides beside it because on a short
                        # epoch it is dominated by a single time-to-first-batch
                        # and reads ~50x higher. `None` in either means no
                        # epoch produced a step — an unmeasured cell, not a
                        # loader that never stalled.
                        "median_data_wait_fraction_steady": (
                            round(statistics.median(steadies), 6)
                            if steadies
                            else None
                        ),
                        "median_ttfb_s": (
                            round(statistics.median(ttfbs), 4) if ttfbs else None
                        ),
                        "median_data_wait_fraction_all_steps": (
                            round(statistics.median(fractions), 5)
                            if fractions
                            else None
                        ),
                        "median_batch_wait_ms_p50": (
                            round(statistics.median(p50s), 3) if p50s else None
                        ),
                        "median_batch_wait_ms_p95": (
                            round(statistics.median(p95s), 3) if p95s else None
                        ),
                        "median_batch_wait_ms_p99": (
                            round(statistics.median(p99s), 3) if p99s else None
                        ),
                        "max_batch_wait_ms": (max(maxes) if maxes else None),
                        "n_steps": [r.n_batches for r in gpu_results],
                        "model": "scVI-equivalent VAE (2L, 128h, 128z)",
                        "batch_size": ML_BATCH_SIZE,
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
