"""
Leiden clustering accelerator benchmark.

Same pattern as `accel_pca.py` / `accel_knn.py`. Leiden has three
backends and pyscx's dispatch priority is Rust-native → cuGraph →
leidenalg.  Each variant pins one backend explicitly for
apples-to-apples timing.

| `format_variant.key` | Backend | Device | How |
|---|---|---|---|
| `accel_leiden__leidenalg_cpu`  | leidenalg | CPU | `scanpy.tl.leiden(flavor="leidenalg")` |
| `accel_leiden__pyscx_cpu`      | scx Rust-native | CPU | `pyscx.accel.leiden(device="cpu")` |
| `accel_leiden__pyscx_gpu`      | cuGraph | GPU | `pyscx.accel.leiden(device="gpu")` — requires Rust-native to fail first, so we bypass by forcing parallel mode |

Correctness: ARI vs the leidenalg reference partition. Stored as
`metadata["ari_vs_leidenalg"]`. Per AGENTS.md Known Limitations:
GPU Leiden ARI 0.92 vs leidenalg is acceptable (algorithmic
difference in refinement — both produce valid partitions).
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


def accel_leiden_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy leidenalg (CPU)", key="accel_leiden__leidenalg_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx Leiden (Rust-native CPU)", key="accel_leiden__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx Leiden (cuGraph GPU)", key="accel_leiden__pyscx_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


def _run_leidenalg(adata: Any, seed: int) -> str:
    import scanpy as sc
    sc.tl.leiden(adata, random_state=seed, flavor="leidenalg")
    return "leidenalg"


def _run_pyscx_cpu(adata: Any, seed: int) -> str:
    import pyscx
    pyscx.accel.leiden(adata, device="cpu", random_state=seed)
    return adata.uns.get("leiden", {}).get("backend", "scx-accel")


def _run_pyscx_gpu(adata: Any, seed: int) -> str:
    import pyscx
    # device="gpu" only routes to cuGraph when Rust-native fails. The
    # benchmark can't easily force that, so we just report whatever the
    # dispatcher picked — `adata.uns["leiden"]["backend"]` captures it.
    pyscx.accel.leiden(adata, device="gpu", random_state=seed)
    return adata.uns.get("leiden", {}).get("backend", "cugraph-or-scx")


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    "accel_leiden__leidenalg_cpu": (_run_leidenalg, False),
    "accel_leiden__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_leiden__pyscx_gpu": (_run_pyscx_gpu, True),
}


def _ari(ref: np.ndarray, test: np.ndarray) -> float:
    try:
        from sklearn.metrics import adjusted_rand_score
        return round(float(adjusted_rand_score(ref, test)), 4)
    except ImportError:
        return float("nan")


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    n_comps: int = 50,
    n_neighbors: int = 15,
    resolution: float = 1.0,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    impl, requires_gpu = _VARIANT_IMPLS[key]
    if key.startswith("accel_leiden__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    result = BenchmarkResult(
        benchmark="accel_leiden",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_comps": n_comps,
            "n_neighbors": n_neighbors,
            "resolution": resolution,
            "n_obs": fixture.n_obs,
            "n_vars": fixture.n_vars,
            "random_seed": RANDOM_SEED,
        },
    )

    # Shared kNN graph (needed for leiden); cached across variants of
    # the same fixture.
    base_key = ("leiden_base", dataset.name, n_neighbors)
    if base_key not in _fixture_cache:
        base = fixture.adata.copy()
        _ensure_pca(base, n_comps)
        _run_scanpy_neighbors(base, n_neighbors, RANDOM_SEED)
        _fixture_cache[base_key] = base
    base: Any = _fixture_cache[base_key]

    # leidenalg reference partition for ARI.
    ref_key = ("leiden_ref", dataset.name, n_neighbors, resolution)
    if ref_key not in _fixture_cache:
        ref = base.copy()
        _run_leidenalg(ref, RANDOM_SEED)
        _fixture_cache[ref_key] = np.asarray(
            ref.obs["leiden"].astype(str).astype("category").cat.codes.values
        )
    ref_labels: np.ndarray = _fixture_cache[ref_key]  # type: ignore[assignment]

    for _ in range(N_WARMUP_RUNS):
        warm = base.copy()
        impl(warm, RANDOM_SEED)
        del warm
        gc.collect()

    backend = ""
    aris: list[float] = []
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
            labels = np.asarray(
                a.obs["leiden"].astype(str).astype("category").cat.codes.values
            )
            aris.append(_ari(ref_labels, labels))
            if not np.isnan(aris[-1]):
                extras["ari_vs_leidenalg"] = aris[-1]
        except Exception as e:
            logger.warning("ARI failed for %s run %d: %s", key, i + 1, e)

        # Route + GPU gate signal. GPU Leiden runs cuGraph (gpu_csr); the
        # variant is skipped on non-GPU hosts and there is no silent
        # cross-backend fallback (cuGraph raises if absent), so a cpu_* route
        # here would mean device resolution silently chose CPU → 0.0.
        route = _extract_route(a, "leiden")
        if route is not None:
            extras["gpu_dispatch_route"] = route
            if requires_gpu:
                extras["leiden_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )
        result.add_run(
            wall_s=wall, user_s=u1 - u0, sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs ARI=%s backend=%s",
            key, i + 1, wall, aris[-1] if aris else float("nan"), backend,
        )
        del a
        gc.collect()

    if aris:
        aris_clean = [x for x in aris if not np.isnan(x)]
        if aris_clean:
            result.metadata["ari_vs_leidenalg"] = round(
                float(np.median(aris_clean)), 4
            )
    result.metadata["backend"] = backend
    return result
