"""
Highly Variable Genes (HVG) accelerator benchmark — Phase 9.

Same pattern as the other accel_* modules. The GPU HVG path (Phase 5)
is narrowed to single-batch seurat_v3; other configurations fall back
to CPU. This benchmark exercises the single-batch seurat_v3 flavour
on all variants to compare like-for-like.

## Variants

| `format_variant.key` | Backend | Device |
|---|---|---|
| `accel_hvg__scanpy_cpu`   | scanpy.pp.highly_variable_genes (seurat_v3) | CPU |
| `accel_hvg__pyscx_cpu`    | pyscx.accel.highly_variable_genes(device="cpu") | CPU |
| `accel_hvg__pyscx_gpu`    | pyscx.accel.highly_variable_genes(device="gpu") | GPU |

Correctness: top-N gene overlap vs scanpy reference (Jaccard). Stored
as `metadata["hvg_overlap_vs_scanpy"]`.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.benchmarks.accel_pca import (
    _fixture_cache,
    _get_cpu_times,
    _get_rss_mb,
)
from benchmarks.comprehensive.benchmarks.accel_preprocess import _load_raw
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    QUERY_N_HVGS,
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


def accel_hvg_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy HVG seurat_v3 (CPU)",
            key="accel_hvg__scanpy_cpu", category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx HVG (CPU)",
            key="accel_hvg__pyscx_cpu", category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx HVG (GPU seurat_v3)",
            key="accel_hvg__pyscx_gpu", category="accel", runner="accel_runner",
        ),
    ]


def _run_scanpy(adata: Any, n_top: int) -> str:
    import scanpy as sc
    sc.pp.highly_variable_genes(
        adata, n_top_genes=n_top, flavor="seurat_v3",
        span=0.3 if adata.n_obs < 10_000 else 1.0,
    )
    return "scanpy-cpu"


def _run_pyscx_cpu(adata: Any, n_top: int) -> str:
    import pyscx
    pyscx.accel.highly_variable_genes(
        adata, n_top_genes=n_top, flavor="seurat_v3", device="cpu",
    )
    return "pyscx-cpu"


def _run_pyscx_gpu(adata: Any, n_top: int) -> str:
    import pyscx
    pyscx.accel.highly_variable_genes(
        adata, n_top_genes=n_top, flavor="seurat_v3", device="gpu",
    )
    return "pyscx-gpu"


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    "accel_hvg__scanpy_cpu": (_run_scanpy, False),
    "accel_hvg__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_hvg__pyscx_gpu": (_run_pyscx_gpu, True),
}


def _extract_route(adata: Any, op: str) -> str | None:
    """Read ``adata.uns["scx_accel"][op]["route"]``, or None if absent."""
    try:
        return adata.uns["scx_accel"][op]["route"]
    except Exception:
        return None


def _hvg_jaccard(ref_var: Any, test_var: Any) -> float:
    """Jaccard of the two HVG gene sets."""
    try:
        ref_set = set(ref_var.index[ref_var["highly_variable"]].tolist())
        test_set = set(test_var.index[test_var["highly_variable"]].tolist())
    except Exception:
        return float("nan")
    if not ref_set or not test_set:
        return 0.0
    return round(len(ref_set & test_set) / len(ref_set | test_set), 4)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    n_top_genes: int = QUERY_N_HVGS,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    impl, requires_gpu = _VARIANT_IMPLS[key]
    if key.startswith("accel_hvg__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None

    raw = _load_raw(dataset)

    result = BenchmarkResult(
        benchmark="accel_hvg",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_top_genes": n_top_genes,
            "n_obs": raw.n_obs,
            "n_vars": raw.n_vars,
            "random_seed": RANDOM_SEED,
        },
    )

    # scanpy reference var DataFrame for Jaccard overlap.
    ref_key = ("hvg_ref_var", dataset.name, n_top_genes)
    if ref_key not in _fixture_cache:
        ref = raw.copy()
        _run_scanpy(ref, n_top_genes)
        _fixture_cache[ref_key] = ref.var.copy()
    ref_var = _fixture_cache[ref_key]

    for _ in range(N_WARMUP_RUNS):
        warm = raw.copy()
        impl(warm, n_top_genes)
        del warm
        gc.collect()

    backend = ""
    jaccards: list[float] = []
    for i in range(n_runs):
        gc.collect()
        a = raw.copy()
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        backend = impl(a, n_top_genes)
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()
        extras: dict[str, Any] = {}
        try:
            jaccards.append(_hvg_jaccard(ref_var, a.var))
            if not np.isnan(jaccards[-1]):
                extras["hvg_overlap_vs_scanpy"] = jaccards[-1]
        except Exception as e:
            logger.warning("HVG overlap failed for %s run %d: %s", key, i + 1, e)

        # Record the accelerator route pyscx stamped, plus a numeric gate
        # signal for the GPU variant. HVG GPU runs the seurat_v3 atomic-CSR
        # kernel (route gpu_csr); the variant is skipped on non-GPU hosts,
        # so a recorded cpu_* route means dispatch silently fell back → 0.0.
        route = _extract_route(a, "highly_variable_genes")
        if route is not None:
            extras["gpu_dispatch_route"] = route
            if requires_gpu:
                extras["hvg_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )
        result.add_run(
            wall_s=wall, user_s=u1 - u0, sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs overlap=%s",
            key, i + 1, wall, jaccards[-1] if jaccards else float("nan"),
        )
        del a
        gc.collect()

    if jaccards:
        jc = [j for j in jaccards if not np.isnan(j)]
        if jc:
            result.metadata["hvg_overlap_vs_scanpy"] = round(
                float(np.median(jc)), 4
            )
    result.metadata["backend"] = backend
    return result
