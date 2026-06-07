"""
GPU / CPU UMAP accelerator benchmark.

Follows the `accel_pca.py` / `accel_knn.py` pattern. UMAP requires a
precomputed kNN graph; this module ensures one is built per fixture
before dispatching to the UMAP variant.

## Variants

| `format_variant.key` | Backend | Device |
|---|---|---|
| `accel_umap__scanpy_cpu`       | scanpy.tl.umap (umap-learn) | CPU |
| `accel_umap__pyscx_cpu`        | pyscx.accel.umap (faer SGD) | CPU |
| `accel_umap__pyscx_gpu`        | pyscx.accel.umap (native CUDA SGD) | GPU |

## Correctness metric

Embeddings are stochastic — direct coordinate comparison fails.
Instead we report UMAP **trustworthiness** (neighbor preservation
score vs PCA space, sklearn's `sklearn.manifold.trustworthiness`),
stored as `metadata["trustworthiness"]`. Threshold: > 0.9.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.benchmarks.accel_pca import (
    _extract_route,
    _fixture_cache,
    _get_cpu_times,
    _get_rss_mb,
    _load_preprocessed,
)
from benchmarks.comprehensive.benchmarks.accel_knn import (
    _ensure_pca,
    _run_scanpy_neighbors,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result
from benchmarks.comprehensive.runners.accel_runner import AcceleratorRunner

logger = logging.getLogger(__name__)

_HAS_PYSCX = False
_HAS_PYSCX_GPU = False
try:
    import pyscx  # noqa: F401
    _HAS_PYSCX = True
    try:
        _HAS_PYSCX_GPU = bool(pyscx.accel.gpu_info())
    except Exception:
        _HAS_PYSCX_GPU = False
except ImportError:
    pass


def accel_umap_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy UMAP (CPU)", key="accel_umap__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx UMAP (CPU)", key="accel_umap__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx UMAP (GPU)", key="accel_umap__pyscx_gpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="rapids-singlecell UMAP (GPU)",
            key="accel_umap__rapids_singlecell_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


def _run_scanpy_umap(adata: Any, _seed: int) -> str:
    import scanpy as sc
    sc.tl.umap(adata, random_state=_seed)
    return "scanpy-cpu"


def _run_pyscx_cpu(adata: Any, seed: int) -> str:
    import pyscx
    pyscx.accel.umap(adata, device="cpu", random_state=seed)
    return adata.uns.get("umap", {}).get("backend", "scx-accel-cpu")


def _run_pyscx_gpu(adata: Any, seed: int) -> str:
    import pyscx
    pyscx.accel.umap(adata, device="gpu", random_state=seed)
    return adata.uns.get("umap", {}).get("backend", "scx-gpu-cuda")


def _run_rapids_singlecell(adata: Any, seed: int) -> str:
    """rapids-singlecell GPU competitor (V3 task 2.9). Runs UMAP on the
    precomputed scanpy neighbors graph (kept GPU-resident via
    `convert_all=True`), isolating the UMAP stage; `X_umap` returns to host for
    the trustworthiness check.
    """
    import rapids_singlecell as rsc

    rsc.get.anndata_to_GPU(adata, convert_all=True)
    rsc.tl.umap(adata, random_state=seed)
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    return "rapids-singlecell-gpu"


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    "accel_umap__scanpy_cpu": (_run_scanpy_umap, False),
    "accel_umap__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_umap__pyscx_gpu": (_run_pyscx_gpu, True),
    "accel_umap__rapids_singlecell_gpu": (_run_rapids_singlecell, True),
}


def _trustworthiness(pca: np.ndarray, umap: np.ndarray, k: int = 15) -> float:
    """sklearn trustworthiness — higher is better (max 1.0).

    Subsamples to 2000 cells on large datasets to keep the O(n² log n)
    distance matrix tractable. Stochastic embeddings typically score
    0.90–0.95 on typical single-cell data.
    """
    try:
        from sklearn.manifold import trustworthiness
    except ImportError:
        logger.warning("sklearn not installed — skipping trustworthiness")
        return float("nan")
    n = pca.shape[0]
    if n > 2000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=2000, replace=False)
        pca = pca[idx]
        umap = umap[idx]
    try:
        return round(float(trustworthiness(pca, umap, n_neighbors=k)), 4)
    except Exception as e:
        logger.warning("trustworthiness failed: %s", e)
        return float("nan")


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    n_comps: int = 50,
    n_neighbors: int = 15,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    impl, requires_gpu = _VARIANT_IMPLS[key]
    if key.startswith("accel_umap__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None
    if key == "accel_umap__rapids_singlecell_gpu" and not AcceleratorRunner.instance().has_rapids_singlecell():
        write_missing_result(
            benchmark="accel_umap", format_key=key, dataset=dataset.name,
            missing_reason="no_rapids_singlecell",
            notes="rapids-singlecell import failed; install into scx-bench-gpu",
        )
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    result = BenchmarkResult(
        benchmark="accel_umap",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_comps": n_comps,
            "n_neighbors": n_neighbors,
            "n_obs": fixture.n_obs,
            "n_vars": fixture.n_vars,
            "random_seed": RANDOM_SEED,
        },
    )

    # Ensure PCA + neighbors are precomputed on a shared fixture copy
    # that every run will clone (UMAP requires `obsp['connectivities']`).
    base_key = ("umap_base", dataset.name, n_neighbors)
    if base_key not in _fixture_cache:
        base = fixture.adata.copy()
        _ensure_pca(base, n_comps)
        _run_scanpy_neighbors(base, n_neighbors, RANDOM_SEED)
        _fixture_cache[base_key] = base
    base: Any = _fixture_cache[base_key]

    # Warm-up
    for _ in range(N_WARMUP_RUNS):
        warm = base.copy()
        impl(warm, RANDOM_SEED)
        del warm
        gc.collect()

    backend = ""
    trust: list[float] = []

    for i in range(n_runs):
        gc.collect()
        a = base.copy()
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        backend = impl(a, RANDOM_SEED)
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        extras: dict[str, Any] = {}
        try:
            pca = np.asarray(a.obsm["X_pca"], dtype=np.float32)
            um = np.asarray(a.obsm["X_umap"], dtype=np.float32)
            trust.append(_trustworthiness(pca, um, k=n_neighbors))
            if not np.isnan(trust[-1]):
                extras["trustworthiness"] = trust[-1]
        except Exception as e:
            logger.warning("trustworthiness failed for %s run %d: %s", key, i + 1, e)

        # Route + GPU gate signal. GPU UMAP runs the native CUDA SGD kernel (or
        # cuML), recorded as gpu_dense; the variant is skipped on non-GPU
        # hosts, so a cpu_* route is a silent fallback → 0.0 fails the gate.
        route = _extract_route(a, "umap")
        if route is not None:
            extras["gpu_dispatch_route"] = route
            if requires_gpu:
                extras["umap_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )

        result.add_run(
            wall_s=wall, user_s=u1 - u0, sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs  trust=%s",
            key, i + 1, wall, trust[-1] if trust else float("nan"),
        )
        del a
        gc.collect()

    if trust:
        trust_clean = [t for t in trust if not np.isnan(t)]
        if trust_clean:
            result.metadata["trustworthiness"] = round(
                float(np.median(trust_clean)), 4
            )
    result.metadata["backend"] = backend
    return result
