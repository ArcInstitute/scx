"""
ML Data Loader Throughput benchmark — COMPREHENSIVE-BENCHMARKING.md §3.6.

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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCX_KEYS = {"scx_auto", "scx_scx1", "scx_zstd", "scx_lz4", "scx_none", "scx_pcodec"}

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


def _current_rss_mb() -> float:
    """Current RSS in MB via /proc/self/statm (not high-water mark)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, IndexError, ValueError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


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
    )
    n_batches = 0
    n_cells = 0
    for batch in ds:
        n_batches += 1
        n_cells += batch["X"].shape[0]
    return _EpochResult(n_batches=n_batches, n_cells=n_cells)


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
                        pass

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

                result.add_run(
                    wall_s=wall_s,
                    user_s=user_s,
                    sys_s=sys_s,
                    peak_rss_mb=peak_rss,
                    scenario=scenario_name,
                    n_batches=epoch_result.n_batches,
                    n_cells=epoch_result.n_cells,
                    batches_per_sec=round(bps, 1),
                    cells_per_sec=round(cps, 0),
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
