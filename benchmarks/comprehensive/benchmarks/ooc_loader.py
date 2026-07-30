"""
Out-of-core sequential-loader throughput benchmark (data-load Phase 0).

The honest, cache-cold counterpart to ``ml_loader``. ``ml_loader`` reports a
warm ``batches_per_sec`` that, on Census-1M, is largely a page-cache artifact —
the same trap annbatch's paper documents inflating BioNeMo-SCDL from 2.5k to 110k
samples/s purely through cache residency. This benchmark instead:

  * **drops the page cache before every timed epoch** via
    ``cache_control.drop_file_cache`` (unprivileged ``posix_fadvise`` — works on
    shared SLURM nodes, unlike ``ml_loader``'s root-only ``/proc`` path), and
  * reports **samples/s + epoch wall-time + true-peak RSS** so an epoch on a
    dataset ≫ node RAM (run under a memory-constrained SLURM allocation; see
    ``SCX_BENCH_OOC_MEM_CAP_GB`` in ``config.py``) is measured with real cache
    misses.

Loaders compared (one per format key):

  ============  ==================================================
  format key    loader
  ============  ==================================================
  ``scx_*``     ``pyscx.TrainingDataset`` (triple-buffered pipeline)
  ``tiledb_soma`` ``tiledbsoma_ml.ExperimentDataset``
  ``h5ad_none`` full-RAM AnnData numpy batching (the anti-pattern baseline)
  ``annloader`` ``anndata.experimental.AnnLoader`` over a backed h5ad
  ``annbatch``  ``annbatch.Loader`` over a pre-shuffled sharded zarr
  ``scdataset`` ``scdataset.scDataset`` (block-sample + batched-fetch) over backed h5ad
  ============  ==================================================

Two scenarios (endpoints of the ``ml_loader`` matrix, to bound census-scale
wall time): ``raw`` (no HVG, no normalize) and ``hvg_norm`` (2k-HVG projection +
normalize_total + log1p).

Per-run ``extra`` keys (sparse per-scenario, so one ``thresholds.yaml`` entry
targets one scenario):
    ``samples_per_sec__<sc>``   — cells/s (the headline out-of-core metric)
    ``epoch_wall_s__<sc>``      — seconds for one full epoch
    ``batches_per_sec__<sc>``   — batches/s
    ``peak_rss_mb__<sc>``       — true in-epoch peak RSS (sampler thread)
    ``cache_policy``            — ``cold_fadvise`` / ``warm`` (was the drop real?)

BioNeMo-SCDL is intentionally **not** wired here — its NeMo/CUDA deps need an
isolated env; it's a deferred fast-follow (see thresholds.yaml deferred floors).
"""

from __future__ import annotations

import gc
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

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
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler
from benchmarks.comprehensive.runners import make_runner

# Reuse the battle-tested SCX / SOMA / full-RAM-AnnData epoch functions rather
# than re-implementing them; importing ml_loader also brings its `_HAS_*`
# probes (harmless).
from benchmarks.comprehensive.benchmarks.ml_loader import (
    _EpochResult,
    _run_anndata_epoch,
    _run_scx_epoch,
    _run_soma_epoch,
    _scx_memory_budget_mb,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional competitor imports (guarded — absent lib → clean skip)
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

_HAS_ANNLOADER = False
try:
    from anndata.experimental import AnnLoader  # noqa: F401

    _HAS_ANNLOADER = True
except ImportError:
    pass

_HAS_ANNBATCH = False
try:
    from annbatch import DatasetCollection, Loader  # noqa: F401

    _HAS_ANNBATCH = True
except ImportError:
    pass

_HAS_SCDATASET = False
try:
    from scdataset import BlockShuffling, scDataset  # noqa: F401

    _HAS_SCDATASET = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCX_KEYS = {"scx_auto", "scx_fast", "scx_scx1", "scx_zstd", "scx_lz4", "scx_none", "scx_pcodec"}

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    _SCX_KEYS | {"tiledb_soma", "h5ad_none", "annloader", "annbatch", "scdataset"}
)
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Mirrors
``_resolve_loader`` + the runtime guard in ``run()``."""

# Endpoints of the ml_loader scenario matrix. (name, hvg, normalize).
_OOC_SCENARIOS: list[tuple[str, bool, bool]] = [
    ("raw", False, False),
    ("hvg_norm", True, True),
]

# scDataset block/fetch knobs (b <= f/2, per arXiv:2506.01883).
_SCDATASET_BLOCK_SIZE = 32
_SCDATASET_FETCH_FACTOR = 8

# annbatch Loader fetch knobs (contiguous chunk of `chunk_size` rows,
# `preload_nchunks` fetched concurrently).
_ANNBATCH_CHUNK_SIZE = 32
_ANNBATCH_PRELOAD_NCHUNKS = 32

# ---------------------------------------------------------------------------
# Loader dispatch
# ---------------------------------------------------------------------------


def _resolve_loader(format_key: str) -> str | None:
    if format_key in _SCX_KEYS:
        return "scx"
    return {
        "tiledb_soma": "soma",
        "h5ad_none": "anndata_full",
        "annloader": "annloader",
        "annbatch": "annbatch",
        "scdataset": "scdataset",
    }.get(format_key)


def _loader_available(loader_type: str) -> bool:
    if loader_type == "scx":
        return _HAS_PYSCX
    if loader_type == "anndata_full":
        try:
            import anndata  # noqa: F401

            return True
        except ImportError:
            return False
    if loader_type == "soma":
        return _HAS_SOMA_ML and _HAS_TORCH
    if loader_type == "annloader":
        return _HAS_ANNLOADER
    if loader_type == "annbatch":
        return _HAS_ANNBATCH
    if loader_type == "scdataset":
        return _HAS_SCDATASET and _HAS_TORCH
    return False


# ---------------------------------------------------------------------------
# Competitor epoch functions (SCX / SOMA / full-RAM reuse ml_loader's)
# ---------------------------------------------------------------------------


def _apply_hvg_norm(X: np.ndarray, hvg: bool, normalize: bool) -> np.ndarray:
    """Shared HVG-project + normalize_total + log1p, matching ml_loader."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    if X.ndim < 2:
        return X
    if hvg and X.shape[1] > QUERY_N_HVGS:
        X = X[:, :QUERY_N_HVGS]
    if normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        X = np.log1p(X / row_sums * 1e4)
    return X


def _run_annloader_epoch(
    h5ad_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    """`anndata.experimental.AnnLoader` over a backed h5ad — the per-cell random
    access baseline (AnnLoader issues backed row reads through a sampler)."""
    import anndata
    from anndata.experimental import AnnLoader

    adata = anndata.read_h5ad(h5ad_path, backed="r")
    loader = AnnLoader(adata, batch_size=batch_size, shuffle=True, use_default_converter=False)
    n_batches = 0
    n_cells = 0
    try:
        for batch in loader:
            X = _apply_hvg_norm(batch.X, hvg, normalize)
            n_batches += 1
            n_cells += X.shape[0] if X.ndim >= 1 else batch_size
    finally:
        if getattr(adata, "isbacked", False) and adata.file is not None:
            adata.file.close()
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _run_annbatch_epoch(
    zarr_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    """`annbatch.Loader` over a pre-shuffled sharded zarr `DatasetCollection`
    (written by `annbatch_runner.convert_from_h5ad`). CPU path: `to=None`,
    `preload_to_gpu=False` → scipy CSR batches."""
    import anndata as ad
    from annbatch import DatasetCollection, Loader

    # Use the zarrs Rust codec pipeline when available (annbatch's local-FS
    # perf recommendation); harmless no-op if the extra isn't installed.
    try:
        import zarr

        import zarrs  # noqa: F401

        zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    except Exception:  # noqa: BLE001
        pass

    collection = DatasetCollection(zarr_path, mode="r")
    loader = Loader(
        batch_size=batch_size,
        chunk_size=_ANNBATCH_CHUNK_SIZE,
        preload_nchunks=_ANNBATCH_PRELOAD_NCHUNKS,
        shuffle=True,
        preload_to_gpu=False,
        to=None,
        rng=np.random.default_rng(seed),
    )
    n_batches = 0
    n_cells = 0
    with ad.settings.override(remove_unused_categories=False):
        loader = loader.use_collection(collection)
        for batch in loader:
            X = batch["X"] if isinstance(batch, dict) else batch
            X = _apply_hvg_norm(X, hvg, normalize)
            n_batches += 1
            n_cells += X.shape[0] if X.ndim >= 1 else batch_size
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _run_scdataset_epoch(
    h5ad_path: str, batch_size: int, hvg: bool, normalize: bool, seed: int
) -> _EpochResult:
    """`scdataset.scDataset` (block sampling + batched fetch, arXiv:2506.01883)
    over a backed h5ad, driven through a `DataLoader(batch_size=None)`."""
    import anndata
    import torch.utils.data as td
    from scdataset import BlockShuffling, scDataset

    data = anndata.read_h5ad(h5ad_path, backed="r")
    strategy = BlockShuffling(block_size=_SCDATASET_BLOCK_SIZE)

    # A backed AnnData cannot be indexed view-of-view (scDataset's default
    # fetch→batch double-indexes), so materialize each fetched block into
    # memory here; the batch step then slices the in-memory block.
    def _fetch_to_memory(collection, indices):
        return collection[np.asarray(indices)].to_memory()

    ds = scDataset(
        data,
        strategy,
        batch_size=batch_size,
        fetch_factor=_SCDATASET_FETCH_FACTOR,
        fetch_callback=_fetch_to_memory,
    )
    loader = td.DataLoader(ds, batch_size=None, num_workers=0)
    n_batches = 0
    n_cells = 0
    try:
        for batch in loader:
            # Default fetch returns an AnnData subset; be tolerant of ndarray.
            X = batch.X if hasattr(batch, "X") else batch
            X = _apply_hvg_norm(X, hvg, normalize)
            n_batches += 1
            n_cells += X.shape[0] if X.ndim >= 1 else batch_size
    finally:
        if getattr(data, "isbacked", False) and data.file is not None:
            data.file.close()
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


def _epoch_fn_for(
    loader_type: str, data_path: str, hvg: bool, normalize: bool
) -> Callable[[], _EpochResult]:
    """Bind the correct epoch closure for the loader + scenario."""
    bs = ML_BATCH_SIZE
    seed = RANDOM_SEED
    if loader_type == "scx":
        return lambda: _run_scx_epoch(data_path, bs, hvg, normalize, seed)
    if loader_type == "soma":
        return lambda: _run_soma_epoch(data_path, bs, hvg, normalize)
    if loader_type == "anndata_full":
        return lambda: _run_anndata_epoch(data_path, bs, hvg, normalize, seed)
    if loader_type == "annloader":
        return lambda: _run_annloader_epoch(data_path, bs, hvg, normalize, seed)
    if loader_type == "annbatch":
        return lambda: _run_annbatch_epoch(data_path, bs, hvg, normalize, seed)
    if loader_type == "scdataset":
        return lambda: _run_scdataset_epoch(data_path, bs, hvg, normalize, seed)
    raise ValueError(f"no epoch fn for loader_type={loader_type!r}")


# ---------------------------------------------------------------------------
# Timed epoch with true-peak RSS + cold drop
# ---------------------------------------------------------------------------


@dataclass
class _TimedEpoch:
    n_batches: int
    n_cells: int
    wall_s: float
    peak_rss_mb: float


def _timed_cold_epoch(epoch_fn: Callable[[], _EpochResult]) -> _TimedEpoch:
    """Run one epoch under a background peak-RSS sampler."""
    gc.collect()
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        res = epoch_fn()
        wall_s = time.perf_counter() - t0
    return _TimedEpoch(
        n_batches=res.n_batches,
        n_cells=res.n_cells,
        wall_s=wall_s,
        peak_rss_mb=sampler.peak_mb,
    )


# ---------------------------------------------------------------------------
# Data-path resolution
# ---------------------------------------------------------------------------


def _resolve_data_path(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    loader_type: str,
    converted_path: Path | None,
) -> str | None:
    """Return the on-disk path for the loader, converting on the fly only for
    formats with a persistent-path property. Returns ``None`` to signal a clean
    skip (missing SOMA / competitor fixture)."""
    key = format_variant.key
    if loader_type == "scx":
        if converted_path is not None and Path(converted_path).exists():
            return str(converted_path)
        try:
            p = dataset.path_for_format(key)
            if p.exists():
                return str(p)
        except (ValueError, FileNotFoundError):
            pass
        return None  # orchestrator runs Phase-A conversion; skip if absent
    if loader_type in ("anndata_full", "annloader", "scdataset"):
        # These read the source h5ad (backed for annloader/scdataset).
        return str(dataset.h5ad_path) if dataset.h5ad_path.exists() else None
    if loader_type == "soma":
        return str(dataset.soma_path) if dataset.soma_path.exists() else None
    if loader_type == "annbatch":
        p = dataset.annbatch_path
        if p.exists():
            return str(p)
        # Pre-shuffle on the fly via the runner (annbatch's DatasetCollection).
        if converted_path is not None and Path(converted_path).exists():
            return str(converted_path)
        return None
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = True,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Out-of-core sequential-loader throughput. Drops the page cache before
    every timed epoch (``cold_cache`` defaults ``True`` — the whole point).

    Returns ``None`` for unsupported formats or missing deps/fixtures (clean
    skip; the orchestrator records it as not-applicable)."""
    loader_type = _resolve_loader(format_variant.key)
    if loader_type is None:
        return None
    if not _loader_available(loader_type):
        logger.warning(
            "Skipping ooc_loader for %s — required deps not installed",
            format_variant.key,
        )
        return None

    data_path = _resolve_data_path(dataset, format_variant, loader_type, converted_path)
    if data_path is None:
        logger.warning(
            "Skipping ooc_loader for %s/%s — no data fixture available",
            format_variant.key,
            dataset.name,
        )
        return None

    result = BenchmarkResult(
        benchmark="ooc_loader",
        format=format_variant.key,
        dataset=dataset.name,
        scenario={"name": "out_of_core", "batch_size": ML_BATCH_SIZE, "n_hvgs": QUERY_N_HVGS},
        metadata={
            "batch_size": ML_BATCH_SIZE,
            "n_hvgs": QUERY_N_HVGS,
            "loader_type": loader_type,
            "cold_cache": cold_cache,
            "scdataset_block_size": _SCDATASET_BLOCK_SIZE,
            "scdataset_fetch_factor": _SCDATASET_FETCH_FACTOR,
            "annbatch_chunk_size": _ANNBATCH_CHUNK_SIZE,
            "annbatch_preload_nchunks": _ANNBATCH_PRELOAD_NCHUNKS,
        },
    )
    dp = Path(data_path)
    if dp.is_file():
        result.file_size_bytes = dp.stat().st_size

    scenario_summary: dict[str, dict[str, Any]] = {}
    for scenario_name, hvg, normalize in _OOC_SCENARIOS:
        logger.info(
            "--- ooc_loader scenario=%s format=%s dataset=%s ---",
            scenario_name,
            format_variant.key,
            dataset.name,
        )
        epoch_fn = _epoch_fn_for(loader_type, data_path, hvg, normalize)

        # One untimed warm-up: warms code (tokio/rayon/numba JIT) but NOT the
        # data cache — we drop the cache before each timed epoch below, so the
        # timed reads are genuinely cold.
        try:
            for _ in range(N_WARMUP_RUNS):
                epoch_fn()
                gc.collect()
        except Exception as e:  # noqa: BLE001
            logger.error("  warmup failed for %s: %s", scenario_name, e)
            scenario_summary[scenario_name] = {"error": str(e)}
            continue

        run_sps: list[float] = []
        run_wall: list[float] = []
        run_rss: list[float] = []
        for i in range(n_runs):
            cache_policy = "warm"
            if cold_cache:
                cache_policy = drop_file_cache(data_path)
            try:
                te = _timed_cold_epoch(epoch_fn)
            except Exception as e:  # noqa: BLE001
                logger.error("  run %d/%d failed for %s: %s", i + 1, n_runs, scenario_name, e)
                continue

            sps = te.n_cells / te.wall_s if te.wall_s > 0 else 0.0
            bps = te.n_batches / te.wall_s if te.wall_s > 0 else 0.0
            result.add_run(
                wall_s=te.wall_s,
                peak_rss_mb=te.peak_rss_mb,
                scenario=scenario_name,
                n_batches=te.n_batches,
                n_cells=te.n_cells,
                cache_policy=cache_policy,
                **{
                    f"samples_per_sec__{scenario_name}": round(sps, 0),
                    f"cells_per_sec__{scenario_name}": round(sps, 0),
                    f"batches_per_sec__{scenario_name}": round(bps, 1),
                    f"epoch_wall_s__{scenario_name}": round(te.wall_s, 3),
                    f"peak_rss_mb__{scenario_name}": round(te.peak_rss_mb, 1),
                },
            )
            run_sps.append(sps)
            run_wall.append(te.wall_s)
            run_rss.append(te.peak_rss_mb)
            logger.info(
                "    wall=%.3fs samples/s=%.0f rss=%.1fMB cache=%s",
                te.wall_s,
                sps,
                te.peak_rss_mb,
                cache_policy,
            )

        summary: dict[str, Any] = {"n_runs": len(run_sps)}
        if run_sps:
            summary["median_samples_per_sec"] = round(statistics.median(run_sps), 0)
            summary["median_epoch_wall_s"] = round(statistics.median(run_wall), 3)
            summary["median_peak_rss_mb"] = round(statistics.median(run_rss), 1)
        scenario_summary[scenario_name] = summary
        gc.collect()

    result.metadata["scenario_summary"] = scenario_summary
    logger.info(
        "ooc_loader complete: %s / %s — %d scenarios",
        format_variant.key,
        dataset.name,
        len(scenario_summary),
    )
    return result
