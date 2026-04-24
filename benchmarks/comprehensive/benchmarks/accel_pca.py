"""
GPU / CPU PCA accelerator benchmark — Phase 9.1 of GPU-ACC-SPEED-UP.md.

Reference implementation: first `accel_*` dimension migrated into the
comprehensive framework. Replaces the standalone
`benchmarks/scripts/benchmark_gpu_pca.py` for gate-integrated use.

## Variants

Each `format_variant.key` selects one PCA implementation that the module
benchmarks. Cells are fully independent — `run_parallel.py` can submit
each (variant × dataset) as its own GPU-allocated SLURM job, matching
the "one job per benchmark × dataset pair" model (`AGENTS.md:21`).

| `format_variant.key` | Backend | Device | Method | QR Method |
|---|---|---|---|---|
| `accel_pca__scanpy_cpu`            | scanpy    | CPU | scanpy default (arpack)       | n/a |
| `accel_pca__pyscx_cpu_auto`        | pyscx     | CPU | auto (covariance if n_vars≤5000) | n/a |
| `accel_pca__pyscx_gpu_cov`         | pyscx     | GPU | covariance                    | n/a |
| `accel_pca__pyscx_gpu_rand_hh`     | pyscx     | GPU | randomized                    | householder |
| `accel_pca__pyscx_gpu_rand_chol`   | pyscx     | GPU | randomized                    | cholesky |

## Emitted metrics

| Field | Where stored |
|---|---|
| Wall time per run | standard `BenchmarkResult.runs[i].wall_s` |
| Peak RSS per run  | standard `peak_rss_mb` |
| Cosine similarity (min / mean over top-k PCs vs scanpy CPU) | `metadata["cosine_sim_min"]`, `metadata["cosine_sim_mean"]` |
| n_comps, n_obs, n_vars, density | `metadata` |
| backend identifier (`adata.uns["pca"]["backend"]`) | `metadata["backend"]` |

The `scanpy_cpu` variant always runs first per cell (small wall-time
cost) to produce the reference embedding; subsequent pyscx variants
compute cosine sim against that reference.

## Dataset fixture

Accelerator benchmarks load the dataset from `dataset.h5ad_path`
directly (no format conversion step). The h5ad files are preprocessed
on-the-fly: `normalize_total` + `log1p` + `highly_variable_genes`
(2000 HVGs). This matches the scanpy-reference workflow used by
`benchmark_gpu_pca.py` for apples-to-apples comparison.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners.accel_runner import (
    AcceleratorRunner,
    PreprocessedFixture,
)

logger = logging.getLogger(__name__)


# Public variant registry — included in `config.get_formats(include_accel=True)`
# so run_parallel / run_all discover them. Keys use the `accel_pca__<impl>`
# convention so result files sort together.
def accel_pca_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy PCA (CPU)", key="accel_pca__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (CPU auto)", key="accel_pca__pyscx_cpu_auto",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (GPU covariance)", key="accel_pca__pyscx_gpu_cov",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (GPU randomized, Householder)",
            key="accel_pca__pyscx_gpu_rand_hh",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (GPU randomized, Cholesky)",
            key="accel_pca__pyscx_gpu_rand_chol",
            category="accel", runner="accel_runner",
        ),
    ]


# Backwards-compat re-exports — earlier accel_* modules import these names
# directly from accel_pca. Point them at AcceleratorRunner instead of the
# removed module-level helpers. The aliases can be removed once every
# accel_*.py has migrated to the runner-instance API (Phase 9.2 follow-up).
PcaFixture = PreprocessedFixture


def _load_preprocessed(dataset: DatasetConfig, n_comps: int) -> PreprocessedFixture:
    """Back-compat shim — sibling `accel_*.py` modules import this by name."""
    return AcceleratorRunner.instance().load_preprocessed(dataset, n_comps)


# Fixture cache and timing helpers used to live here; now owned by
# `AcceleratorRunner.instance()`. Back-compat aliases so existing sibling
# modules keep working — they pass the runner's dict by reference.
_fixture_cache = AcceleratorRunner.instance()._cache


def _get_rss_mb() -> float:
    return AcceleratorRunner.get_rss_mb()


def _get_cpu_times() -> tuple[float, float]:
    return AcceleratorRunner.get_cpu_times()


# Availability flags — exposed as module-level names for back-compat with
# sibling modules that cache them at import time. These read through to the
# runner's cached probes.
_HAS_PYSCX = AcceleratorRunner.instance().has_pyscx()
_HAS_PYSCX_GPU = AcceleratorRunner.instance().has_gpu()


# ---------------------------------------------------------------------------
# Per-variant implementations
# ---------------------------------------------------------------------------

def _run_scanpy_cpu(adata: Any, n_comps: int, _seed: int) -> str:
    import scanpy as sc

    sc.pp.pca(adata, n_comps=n_comps, random_state=_seed)
    return "scanpy-cpu"


def _run_pyscx_cpu_auto(adata: Any, n_comps: int, seed: int) -> str:
    import pyscx

    pyscx.accel.pca(adata, n_comps=n_comps, device="cpu", random_state=seed)
    return adata.uns["pca"].get("backend", "scx-accel-cpu")


def _run_pyscx_gpu_cov(adata: Any, n_comps: int, seed: int) -> str:
    import pyscx

    pyscx.accel.pca(
        adata, n_comps=n_comps, device="gpu",
        method="covariance", random_state=seed,
    )
    return adata.uns["pca"].get("backend", "scx-gpu-cusparse")


def _run_pyscx_gpu_rand_hh(adata: Any, n_comps: int, seed: int) -> str:
    import pyscx

    pyscx.accel.pca(
        adata, n_comps=n_comps, device="gpu",
        method="randomized", qr_method="householder", random_state=seed,
    )
    return adata.uns["pca"].get("backend", "scx-gpu-cusparse")


def _run_pyscx_gpu_rand_chol(adata: Any, n_comps: int, seed: int) -> str:
    import pyscx

    pyscx.accel.pca(
        adata, n_comps=n_comps, device="gpu",
        method="randomized", qr_method="cholesky", random_state=seed,
    )
    return adata.uns["pca"].get("backend", "scx-gpu-cusparse")


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    # key: (implementation, requires_gpu)
    "accel_pca__scanpy_cpu": (_run_scanpy_cpu, False),
    "accel_pca__pyscx_cpu_auto": (_run_pyscx_cpu_auto, False),
    "accel_pca__pyscx_gpu_cov": (_run_pyscx_gpu_cov, True),
    "accel_pca__pyscx_gpu_rand_hh": (_run_pyscx_gpu_rand_hh, True),
    "accel_pca__pyscx_gpu_rand_chol": (_run_pyscx_gpu_rand_chol, True),
}


# ---------------------------------------------------------------------------
# Correctness: cosine similarity vs scanpy reference
# ---------------------------------------------------------------------------

def _sign_agnostic_cosine_per_pc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise sign-agnostic cosine similarity between two (n_obs × k)
    PCA embeddings. PC signs are arbitrary, so we take abs(dot / norms).
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    k = a.shape[1]
    cos = np.empty(k, dtype=np.float64)
    for j in range(k):
        x = a[:, j].astype(np.float64)
        y = b[:, j].astype(np.float64)
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        denom = max(nx * ny, 1e-12)
        cos[j] = float(abs(float(x @ y) / denom))
    return cos


# ---------------------------------------------------------------------------
# Main entry point — framework contract
# ---------------------------------------------------------------------------

def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,  # unused — accelerators don't convert
    n_comps: int = 50,
) -> BenchmarkResult | None:
    """Execute the PCA accelerator benchmark for a single variant.

    Returns `None` when the variant is unavailable (e.g. GPU variant on a
    CPU-only host); `run_all.py` / `run_parallel.py` interpret `None` as
    a non-regression "skipped" cell.
    """
    variant_key = format_variant.key
    if variant_key not in _VARIANT_IMPLS:
        logger.warning("Unknown PCA variant %s — skipping", variant_key)
        return None

    impl, requires_gpu = _VARIANT_IMPLS[variant_key]

    if variant_key.startswith("accel_pca__pyscx") and not _HAS_PYSCX:
        logger.warning("pyscx not installed — skipping %s", variant_key)
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        logger.warning("GPU not available — skipping %s", variant_key)
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    result = BenchmarkResult(
        benchmark="accel_pca",
        format=variant_key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_comps": n_comps,
            "n_obs": fixture.n_obs,
            "n_vars": fixture.n_vars,
            "random_seed": RANDOM_SEED,
        },
    )

    # Produce / cache the scanpy reference embedding for cosine-sim comparison.
    # (Scanpy cell runs this work anyway; other variants reuse it.)
    ref_key = ("ref_embedding", dataset.name, n_comps)
    if ref_key not in _fixture_cache:
        ref_adata = fixture.adata.copy()
        _run_scanpy_cpu(ref_adata, n_comps, RANDOM_SEED)
        _fixture_cache[ref_key] = np.asarray(ref_adata.obsm["X_pca"], dtype=np.float32)

    ref_embedding: np.ndarray = _fixture_cache[ref_key]  # type: ignore[assignment]

    # Warm-up
    for i in range(N_WARMUP_RUNS):
        logger.info("Warm-up run %d/%d for %s", i + 1, N_WARMUP_RUNS, variant_key)
        warm_adata = fixture.adata.copy()
        impl(warm_adata, n_comps, RANDOM_SEED)
        del warm_adata
        gc.collect()

    backend = ""
    cos_mean_runs: list[float] = []
    cos_min_runs: list[float] = []

    for i in range(n_runs):
        gc.collect()
        t_adata = fixture.adata.copy()

        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()

        backend = impl(t_adata, n_comps, RANDOM_SEED)

        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        # Cosine similarity vs scanpy reference (top-k PCs).
        try:
            emb = np.asarray(t_adata.obsm["X_pca"], dtype=np.float32)
            cos = _sign_agnostic_cosine_per_pc(ref_embedding, emb)
            cos_mean_runs.append(float(np.mean(cos)))
            cos_min_runs.append(float(np.min(cos)))
        except Exception as e:
            logger.warning("Cosine-sim check failed for %s run %d: %s",
                           variant_key, i + 1, e)

        result.add_run(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
        )
        logger.info(
            "  %s run %d: wall=%.3fs  rss=%.1fMB  cos_min=%.4f",
            variant_key, i + 1, wall, max(rss_before, rss_after),
            cos_min_runs[-1] if cos_min_runs else float("nan"),
        )
        del t_adata
        gc.collect()

    # Aggregate correctness across runs — PCA is deterministic modulo
    # randomized-seed dependence, so all runs should report near-identical
    # numbers; the median is defensive.
    if cos_mean_runs:
        result.metadata["cosine_sim_mean"] = round(
            float(np.median(cos_mean_runs)), 6
        )
        result.metadata["cosine_sim_min"] = round(
            float(np.median(cos_min_runs)), 6
        )
    result.metadata["backend"] = backend

    logger.info(
        "Benchmark complete: %s / %s — median %.3fs, cos_min=%s",
        variant_key, dataset.name,
        result.median_wall_s or 0.0,
        result.metadata.get("cosine_sim_min", "n/a"),
    )
    return result


# Timing helpers live on `AcceleratorRunner`; the back-compat shims
# (`_get_rss_mb`, `_get_cpu_times`) near the top of this module delegate
# to them.
