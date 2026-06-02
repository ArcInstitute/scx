"""
GPU / CPU kNN accelerator benchmark.

Follows the `accel_pca.py` pattern. Each variant is one implementation
of the kNN connectivity graph, scheduled as an independent cell by
`run_parallel.py`.

## Variants

| `format_variant.key` | Backend | Device |
|---|---|---|
| `accel_knn__scanpy_cpu`       | scanpy.pp.neighbors (HNSW via `pynndescent`) | CPU |
| `accel_knn__pyscx_cpu`        | pyscx.accel.neighbors (instant-distance HNSW) | CPU |
| `accel_knn__pyscx_gpu_cagra`  | pyscx.accel.neighbors (cuVS CAGRA)            | GPU |

## Correctness metric

Recall@k vs the scanpy CPU reference (neighbor-set overlap averaged
across cells). Reported as `metadata["recall_vs_scanpy"]`. A PR that
drops recall below ~0.9 on pbmc3k is flagged by Phase-9 thresholds.

## Fixture

Shares `_load_preprocessed` + reuses the scanpy reference embedding
produced by `accel_pca.py`'s fixture cache. Running `accel_pca` before
`accel_knn` in the same worker saves one preprocessing pass; when
submitit schedules each cell in its own worker the caches don't share,
so each cell pays the preprocessing cost once.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.benchmarks.accel_pca import (
    PcaFixture,
    _extract_route,
    _fixture_cache,
    _get_cpu_times,
    _get_rss_mb,
    _load_preprocessed,
    _run_scanpy_cpu,
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult

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


def accel_knn_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy neighbors (CPU)", key="accel_knn__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx neighbors (CPU HNSW)", key="accel_knn__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx neighbors (GPU CAGRA)", key="accel_knn__pyscx_gpu_cagra",
            category="accel", runner="accel_runner",
        ),
    ]


# ---------------------------------------------------------------------------
# Per-variant implementations
# ---------------------------------------------------------------------------

def _ensure_pca(adata: Any, n_comps: int) -> None:
    """Run scanpy PCA in-place if missing — kNN needs `X_pca`."""
    if "X_pca" in adata.obsm:
        return
    import scanpy as sc
    sc.pp.pca(adata, n_comps=n_comps, random_state=RANDOM_SEED)


def _run_scanpy_neighbors(adata: Any, n_neighbors: int, _seed: int) -> str:
    import scanpy as sc
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, random_state=_seed)
    return "scanpy-cpu"


def _run_pyscx_cpu(adata: Any, n_neighbors: int, seed: int) -> str:
    import pyscx
    pyscx.accel.neighbors(
        adata, n_neighbors=n_neighbors, device="cpu", random_state=seed,
    )
    return adata.uns["neighbors"].get("backend", "scx-accel-cpu")


def _run_pyscx_gpu_cagra(adata: Any, n_neighbors: int, seed: int) -> str:
    import pyscx
    pyscx.accel.neighbors(
        adata, n_neighbors=n_neighbors, device="gpu", random_state=seed,
    )
    return adata.uns["neighbors"].get("backend", "scx-gpu-cagra")


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    "accel_knn__scanpy_cpu": (_run_scanpy_neighbors, False),
    "accel_knn__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_knn__pyscx_gpu_cagra": (_run_pyscx_gpu_cagra, True),
}


def _recall_at_k(ref_conn: Any, test_conn: Any) -> float:
    """Neighbor-set Jaccard overlap averaged across cells.

    Both `ref_conn` and `test_conn` are `scipy.sparse` connectivity
    matrices (n_obs × n_obs). For each cell, compare the neighbor sets
    (indices of non-zero entries). Returns the mean overlap size / k.
    """
    import scipy.sparse as sp

    if not sp.issparse(ref_conn) or not sp.issparse(test_conn):
        return float("nan")
    r = ref_conn.tocsr()
    t = test_conn.tocsr()
    n = r.shape[0]
    if n == 0:
        return 1.0

    # Sample up to 500 cells to keep the recall computation fast on large
    # datasets; full enumeration at 1M cells would be expensive.
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(500, n), replace=False)

    total = 0.0
    for i in idx:
        ri = set(r.indices[r.indptr[i]:r.indptr[i + 1]].tolist())
        ti = set(t.indices[t.indptr[i]:t.indptr[i + 1]].tolist())
        if not ri:
            continue
        total += len(ri & ti) / max(len(ri), 1)
    return round(total / len(idx), 4)


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
        logger.warning("Unknown kNN variant %s — skipping", key)
        return None
    impl, requires_gpu = _VARIANT_IMPLS[key]
    if key.startswith("accel_knn__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    result = BenchmarkResult(
        benchmark="accel_knn",
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

    # Ensure a reference connectivity graph via scanpy is cached so other
    # variants can compute recall.
    ref_key = ("ref_connectivities", dataset.name, n_neighbors)
    if ref_key not in _fixture_cache:
        ref_adata = fixture.adata.copy()
        _ensure_pca(ref_adata, n_comps)
        _run_scanpy_neighbors(ref_adata, n_neighbors, RANDOM_SEED)
        _fixture_cache[ref_key] = ref_adata.obsp["connectivities"]
    ref_conn = _fixture_cache[ref_key]

    # Warm-up
    for i in range(N_WARMUP_RUNS):
        warm = fixture.adata.copy()
        _ensure_pca(warm, n_comps)
        impl(warm, n_neighbors, RANDOM_SEED)
        del warm
        gc.collect()

    backend = ""
    recalls: list[float] = []

    for i in range(n_runs):
        gc.collect()
        a = fixture.adata.copy()
        _ensure_pca(a, n_comps)

        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        backend = impl(a, n_neighbors, RANDOM_SEED)
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        extras: dict[str, Any] = {}
        try:
            recalls.append(_recall_at_k(ref_conn, a.obsp["connectivities"]))
            extras["recall_vs_scanpy"] = recalls[-1]
        except Exception as e:
            logger.warning("recall-check failed for %s run %d: %s", key, i + 1, e)

        # Route + GPU gate signal. GPU kNN runs cuVS CAGRA (gpu_csr); the
        # variant is skipped on non-GPU hosts, so a cpu_* route is a silent
        # fallback (cuVS unavailable) → 0.0 fails the gate.
        route = _extract_route(a, "neighbors")
        if route is not None:
            extras["gpu_dispatch_route"] = route
            if requires_gpu:
                extras["knn_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )

        result.add_run(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs rss=%.1fMB recall=%.3f",
            key, i + 1, wall, max(rss_before, rss_after),
            recalls[-1] if recalls else float("nan"),
        )
        del a
        gc.collect()

    if recalls:
        result.metadata["recall_vs_scanpy"] = round(float(np.median(recalls)), 4)
    result.metadata["backend"] = backend
    logger.info(
        "Benchmark complete: %s / %s — median %.3fs, recall=%s",
        key, dataset.name, result.median_wall_s or 0.0,
        result.metadata.get("recall_vs_scanpy", "n/a"),
    )
    return result
