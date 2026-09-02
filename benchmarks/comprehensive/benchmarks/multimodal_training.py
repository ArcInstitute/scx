"""Multimodal training-throughput benchmark (Phase K.4).

Compares two loader paths against a multimodal dataset:

* ``scx_multimodal_per_modality_auto`` /
  ``scx_multimodal_uniform_auto`` —
  ``pyscx.MultimodalTrainingDataset`` (triple-buffered Rust pipeline,
  per-modality CSR shards).
* ``h5mu_uncompressed`` / ``h5mu_gzip`` /
  ``zarr_mudata_zstd`` — scvi-tools' ``AnnTorchDataset`` applied
  per-modality and concatenated batch-wise. Mirrors what a researcher
  would do today when running TOTALVI / MultiVI without an SCX-native
  loader.

Reports per-scenario:

* ``batches_per_sec`` — total training batches per wall-clock second.
* ``cells_per_sec``  — sum across modalities.
* ``time_to_first_batch_s`` — median over ``_N_TTFB_RUNS`` reps.
* ``peak_rss_mb`` — the high-water RSS *during* the measured epoch,
  sampled by ``PeakRssSampler``. It was ``getrusage(RUSAGE_SELF).ru_maxrss``
  until the peak-sampler change: a real high-water mark, but scoped to the
  whole process lifetime, so a peak from the time-to-first-batch reps or from a
  previous
  epoch's eager ``read_h5mu`` was re-reported as this epoch's. (The
  ``max(before, after)`` idiom that guarded it was a no-op — ``ru_maxrss``
  is monotone, so ``before <= after`` always.)

This is *not* a model-training benchmark; we exercise only the
dataloader. Time-to-first-batch and steady-state throughput are the
load-bearing comparison metrics.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler
from benchmarks.comprehensive.runners import make_runner
from benchmarks.comprehensive.runners.base import FormatRunner

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# How many batches to iterate per measured epoch. Multimodal batches
# are heavier than single-modality (one dense slab per modality), so we
# keep the count modest. The bench is throughput-dominated, not
# count-dominated.
_BATCHES_PER_EPOCH = 100

# Time-to-first-batch is high-variance on large h5mu sources (cold
# scipy CSR materialisation), so we take the median of three reps.
_N_TTFB_RUNS = 3

# Default training batch size.
_BATCH_SIZE = 4096


# --------------------------------------------------------------------------
# Result aggregator
# --------------------------------------------------------------------------


@dataclass
class _EpochResult:
    n_batches: int
    n_cells: int
    wall_s: float
    peak_rss_mb: float


def _disable_hdf5_locking() -> None:
    """Multiple parallel SLURM jobs may read the same source `.h5mu`
    concurrently; the default h5py file lock raises ``BlockingIOError:
    errno 11`` on contended reads. Disable locking for read-only opens
    (the source is never mutated during a benchmark).
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# --------------------------------------------------------------------------
# Scenario runners
# --------------------------------------------------------------------------


def _run_scx_multimodal_epoch(
    path: Path,
    modality_names: tuple[str, ...],
    batch_size: int,
    seed: int,
    n_batches_target: int,
) -> _EpochResult:
    """Iterate ``pyscx.MultimodalTrainingDataset`` and report
    throughput. ``n_batches_target`` caps the iteration so large
    datasets don't dominate wall-clock; the per-modality cells per
    second remain comparable."""
    import pyscx

    ds = pyscx.MultimodalTrainingDataset(
        str(path),
        modalities=list(modality_names),
        batch_size=batch_size,
        normalize=False,
        log1p=False,
        seed=seed,
        return_dict=True,
    )
    gc.collect()
    t0 = time.perf_counter()
    n_batches = 0
    n_cells = 0
    with PeakRssSampler() as sampler:
        for batch in ds:
            # `batch["X"]` is a dict {modality_name: ndarray}.
            x_dict = batch["X"]
            # Use the first modality's row count (rows are aligned across
            # modalities — same global cell index).
            any_modality = next(iter(x_dict.values()))
            n_cells += any_modality.shape[0]
            n_batches += 1
            if n_batches >= n_batches_target:
                break
    wall = time.perf_counter() - t0
    return _EpochResult(
        n_batches=n_batches,
        n_cells=n_cells,
        wall_s=wall,
        peak_rss_mb=sampler.peak_mb,
    )


def _run_h5mu_epoch(
    h5mu_path: Path,
    modality_names: tuple[str, ...],
    batch_size: int,
    seed: int,
    n_batches_target: int,
) -> _EpochResult:
    """Baseline path: load h5mu eagerly, build a numpy index iterator,
    yield per-batch dense slabs per modality. This is what scvi-tools'
    ``AnnTorchDataset`` does internally (just specialised to
    multimodal); we do it directly so the bench doesn't depend on the
    full scvi-tools registration ceremony."""
    import mudata
    import scipy.sparse as sp

    _disable_hdf5_locking()
    mu = mudata.read_h5mu(str(h5mu_path))
    # Materialise per-modality X as scipy CSR and capture the global
    # n_obs from the outer obs.
    n_obs = mu.n_obs
    rng = np.random.default_rng(seed)
    indices = np.arange(n_obs)
    rng.shuffle(indices)

    # Pre-fetch each modality's matrix to avoid per-batch dict lookup.
    mats = {name: mu.mod[name].X for name in modality_names}
    for name, X in mats.items():
        if not sp.issparse(X):
            mats[name] = sp.csr_matrix(X)

    gc.collect()
    t0 = time.perf_counter()
    n_batches = 0
    n_cells = 0
    with PeakRssSampler() as sampler:
        for start in range(0, n_obs, batch_size):
            rows = indices[start : start + batch_size]
            for name, X in mats.items():
                slab = X[rows].toarray()
                # Force materialization
                _ = slab.shape
            n_cells += rows.size
            n_batches += 1
            if n_batches >= n_batches_target:
                break
    wall = time.perf_counter() - t0
    return _EpochResult(
        n_batches=n_batches,
        n_cells=n_cells,
        wall_s=wall,
        peak_rss_mb=sampler.peak_mb,
    )


def _run_zarr_mudata_epoch(
    zarr_path: Path,
    modality_names: tuple[str, ...],
    batch_size: int,
    seed: int,
    n_batches_target: int,
) -> _EpochResult:
    import mudata
    import scipy.sparse as sp

    mu = mudata.read_zarr(str(zarr_path))
    n_obs = mu.n_obs
    rng = np.random.default_rng(seed)
    indices = np.arange(n_obs)
    rng.shuffle(indices)

    mats = {name: mu.mod[name].X for name in modality_names}
    for name, X in mats.items():
        if not sp.issparse(X):
            mats[name] = sp.csr_matrix(X)

    gc.collect()
    t0 = time.perf_counter()
    n_batches = 0
    n_cells = 0
    with PeakRssSampler() as sampler:
        for start in range(0, n_obs, batch_size):
            rows = indices[start : start + batch_size]
            for name, X in mats.items():
                slab = X[rows].toarray()
                _ = slab.shape
            n_cells += rows.size
            n_batches += 1
            if n_batches >= n_batches_target:
                break
    wall = time.perf_counter() - t0
    return _EpochResult(
        n_batches=n_batches,
        n_cells=n_cells,
        wall_s=wall,
        peak_rss_mb=sampler.peak_mb,
    )


# --------------------------------------------------------------------------
# Time-to-first-batch
# --------------------------------------------------------------------------


def _measure_ttfb_scx(path: Path, modality_names: tuple[str, ...]) -> list[float]:
    import pyscx

    times: list[float] = []
    for _ in range(_N_TTFB_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        ds = pyscx.MultimodalTrainingDataset(
            str(path),
            modalities=list(modality_names),
            batch_size=_BATCH_SIZE,
            seed=0,
            return_dict=True,
        )
        next(iter(ds))
        times.append(time.perf_counter() - t0)
        del ds
    return times


def _measure_ttfb_h5mu(path: Path, modality_names: tuple[str, ...]) -> list[float]:
    import mudata
    import scipy.sparse as sp

    _disable_hdf5_locking()
    times: list[float] = []
    for _ in range(_N_TTFB_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        mu = mudata.read_h5mu(str(path))
        # Take the first batch_size rows of every modality.
        for name in modality_names:
            X = mu.mod[name].X[:_BATCH_SIZE]
            if sp.issparse(X):
                X.toarray()
        times.append(time.perf_counter() - t0)
    return times


def _measure_ttfb_zarr(path: Path, modality_names: tuple[str, ...]) -> list[float]:
    import mudata
    import scipy.sparse as sp

    times: list[float] = []
    for _ in range(_N_TTFB_RUNS):
        gc.collect()
        t0 = time.perf_counter()
        mu = mudata.read_zarr(str(path))
        for name in modality_names:
            X = mu.mod[name].X[:_BATCH_SIZE]
            if sp.issparse(X):
                X.toarray()
        times.append(time.perf_counter() - t0)
    return times


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    if not dataset.multimodal:
        raise ValueError(
            f"multimodal_training: dataset {dataset.name!r} is "
            f"single-modality; use ml_loader instead"
        )

    # Resolve the data path for the given format. multimodal_compression
    # already converts the dataset to each format variant; we read the
    # output here. Falls back to the format's persistent path on
    # disk when ``converted_path`` is None.
    src: Path
    if converted_path is not None and Path(converted_path).exists():
        src = Path(converted_path)
    else:
        src = dataset.path_for_format(format_variant.key)
        if not src.exists():
            raise FileNotFoundError(
                f"Pre-converted file for {format_variant.key} not found at "
                f"{src}; run multimodal_compression first to materialize it."
            )

    log.info(
        "Multimodal training: dataset=%s format=%s src=%s",
        dataset.name,
        format_variant.key,
        src,
    )

    # Dispatch on the format's loader path. SCX uses the
    # `MultimodalTrainingDataset` Rust pipeline; h5mu and zarr_mudata
    # use the eager-mudata baseline.
    if format_variant.key.startswith("scx_multimodal"):
        ttfbs = _measure_ttfb_scx(src, dataset.modality_names)
        epochs = [
            _run_scx_multimodal_epoch(
                src, dataset.modality_names, _BATCH_SIZE, seed=i,
                n_batches_target=_BATCHES_PER_EPOCH,
            )
            for i in range(n_runs)
        ]
    elif format_variant.key in ("h5mu_uncompressed", "h5mu_gzip"):
        ttfbs = _measure_ttfb_h5mu(src, dataset.modality_names)
        epochs = [
            _run_h5mu_epoch(
                src, dataset.modality_names, _BATCH_SIZE, seed=i,
                n_batches_target=_BATCHES_PER_EPOCH,
            )
            for i in range(n_runs)
        ]
    elif format_variant.key == "zarr_mudata_zstd":
        ttfbs = _measure_ttfb_zarr(src, dataset.modality_names)
        epochs = [
            _run_zarr_mudata_epoch(
                src, dataset.modality_names, _BATCH_SIZE, seed=i,
                n_batches_target=_BATCHES_PER_EPOCH,
            )
            for i in range(n_runs)
        ]
    else:
        raise ValueError(
            f"multimodal_training: unsupported format {format_variant.key!r}"
        )

    # Aggregate.
    median_ttfb = float(np.median(ttfbs))
    walls = np.asarray([e.wall_s for e in epochs], dtype=float)
    bps = np.asarray(
        [e.n_batches / e.wall_s if e.wall_s > 0 else 0.0 for e in epochs],
        dtype=float,
    )
    cps = np.asarray(
        [e.n_cells / e.wall_s if e.wall_s > 0 else 0.0 for e in epochs],
        dtype=float,
    )
    # Each epoch now carries its own in-region peak (they are independent
    # samples, not a monotone ru_maxrss series), so this is a real max.
    peak_rss = max(e.peak_rss_mb for e in epochs) if epochs else 0.0

    log.info(
        "  -> ttfb=%.3fs  batches/s=%.1f  cells/s=%.0f  rss=%.0f MB",
        median_ttfb,
        float(np.median(bps)),
        float(np.median(cps)),
        peak_rss,
    )

    # Actual batches produced per epoch. ``_BATCHES_PER_EPOCH`` is only a
    # *cap*; the SCX loader does not cycle, so a small file exhausts in a
    # single pass (e.g. cite_seq 5,247 cells / batch 4,096 ≈ 2 batches).
    # Recording the observed count keeps the raw JSON honest — the derived
    # ``batches_per_sec`` is over this many batches, not the 100 target.
    n_batches_observed = int(np.median([e.n_batches for e in epochs])) if epochs else 0

    metadata = {
        "n_modalities": len(dataset.modality_names),
        "modality_names": list(dataset.modality_names),
        "batch_size": _BATCH_SIZE,
        "n_batches_per_epoch_target": _BATCHES_PER_EPOCH,
        "n_batches_observed_median": n_batches_observed,
        "time_to_first_batch_s": round(median_ttfb, 6),
        "ttfb_runs_s": [round(t, 6) for t in ttfbs],
        # ``cells_per_sec`` is the floored, batch-size-invariant throughput
        # metric (see thresholds.yaml). ``batches_per_sec`` is retained for
        # continuity but is batch-size- and single-pass-sensitive.
        "batches_per_sec_median": round(float(np.median(bps)), 4),
        "cells_per_sec_median": round(float(np.median(cps)), 2),
        "peak_rss_mb": round(peak_rss, 2),
    }

    result = BenchmarkResult(
        benchmark="multimodal_training",
        format=format_variant.key,
        dataset=dataset.name,
        metadata=metadata,
    )
    for e in epochs:
        # Headline metrics flow into ``runs[].extra`` so the gate's
        # ``check_absolute_floors`` can read them
        # (``compare_against_baseline.py:_load_current_raw_metric``).
        # ``time_to_first_batch_s`` is measured once outside the
        # epoch loop and broadcast into every run so the
        # median-across-runs reduction collapses to the right
        # scalar without a special-case path. ``metadata`` keeps
        # the human-readable summary.
        result.add_run(
            wall_s=e.wall_s,
            peak_rss_mb=e.peak_rss_mb,
            n_batches=e.n_batches,
            n_cells=e.n_cells,
            batches_per_sec=(e.n_batches / e.wall_s) if e.wall_s > 0 else 0.0,
            cells_per_sec=(e.n_cells / e.wall_s) if e.wall_s > 0 else 0.0,
            time_to_first_batch_s=median_ttfb,
        )
    return result
