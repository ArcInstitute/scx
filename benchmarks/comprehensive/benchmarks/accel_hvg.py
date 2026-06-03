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

# Reuse DE's SCX-with-CSC fixture machinery so the GPU HVG variant exercises
# the column-major CSC reduce route (gpu_csc_v3) on a backed dataset rather
# than the in-memory CSR-atomic path. Shared verbatim to avoid divergence.
from benchmarks.comprehensive.benchmarks.accel_de import (
    _as_scx_backed_if_available,
    _ensure_scx_csc_fixture,
    _propagate_route,
)
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


# HVG result columns copied from the backed adata (where the kernel wrote
# them) back to the runner's in-memory adata so the Jaccard overlap below
# still sees them. Gene order is identical (same dataset, preserved through
# from_anndata → to_anndata(backed=True)).
_HVG_VAR_COLS = (
    "highly_variable",
    "means",
    "variances",
    "variances_norm",
    "highly_variable_rank",
)


def _run_pyscx_gpu(adata: Any, n_top: int) -> str:
    import pyscx

    # When the bench built an SCX fixture with a CSC sidecar for this dataset,
    # run on the backed dataset so single-batch seurat_v3 GPU HVG auto-routes
    # to the column-major reduce (gpu_csc_v3). Otherwise run in-memory (the
    # CSR-atomic gpu_csr path). Mirrors accel_de's pdex_ref GPU dispatch.
    scx_adata = _as_scx_backed_if_available(adata)
    target = scx_adata if scx_adata is not None else adata
    pyscx.accel.highly_variable_genes(
        target, n_top_genes=n_top, flavor="seurat_v3", device="gpu",
    )
    if scx_adata is not None:
        _propagate_route(scx_adata, adata)
        # Copy the HVG var columns back so run()'s Jaccard (on adata.var) and
        # any downstream readers see them.
        for col in _HVG_VAR_COLS:
            if col in scx_adata.var.columns:
                adata.var[col] = scx_adata.var[col].to_numpy()
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

    # For the GPU variant, build an SCX-with-CSC fixture so single-batch
    # seurat_v3 GPU HVG auto-routes to the column-major reduce (gpu_csc_v3).
    # `None` (pyscx missing or conversion failed) falls back to the in-memory
    # CSR-atomic gpu_csr path. The path is stashed on each per-run adata's uns
    # so `_run_pyscx_gpu` → `_as_scx_backed_if_available` can pick it up.
    scx_csc_path = (
        _ensure_scx_csc_fixture(raw, dataset.name) if requires_gpu and _HAS_PYSCX else None
    )

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
            "scx_csc_fixture": bool(scx_csc_path),
        },
    )

    # scanpy reference var DataFrame for Jaccard overlap.
    ref_key = ("hvg_ref_var", dataset.name, n_top_genes)
    if ref_key not in _fixture_cache:
        ref = raw.copy()
        _run_scanpy(ref, n_top_genes)
        _fixture_cache[ref_key] = ref.var.copy()
    ref_var = _fixture_cache[ref_key]

    # Resolve the GPU timing target once, outside the timed loop, so the
    # one-time `pyscx.open` + `to_anndata(backed=True)` construction is excluded
    # from the measured HVG wall time (apples-to-apples with the in-memory CSR
    # variant). Safe to reuse across warmup + all timed runs: HVG with
    # `subset=False` does not mutate X, and re-writing the var HVG columns is
    # idempotent. `None` → in-memory path via `impl()` — this also covers a
    # failed open, which then correctly trips `hvg_route_csc_direct` (the CSC
    # reduce never ran).
    gpu_backed = None
    if requires_gpu and scx_csc_path is not None:
        try:
            import pyscx
            gpu_backed = pyscx.open(str(scx_csc_path)).to_anndata(backed=True)
        except Exception as e:
            logger.warning(
                "accel_hvg: failed to open SCX-with-CSC fixture %s (%s); "
                "GPU variant falls back to in-memory CSR path",
                scx_csc_path, e,
            )
            gpu_backed = None

    for _ in range(N_WARMUP_RUNS):
        if gpu_backed is not None:
            import pyscx
            pyscx.accel.highly_variable_genes(
                gpu_backed, n_top_genes=n_top_genes, flavor="seurat_v3", device="gpu",
            )
        else:
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
        if gpu_backed is not None:
            # Time only the HVG call on the pre-opened backed dataset; the
            # one-time open already happened above.
            import pyscx
            pyscx.accel.highly_variable_genes(
                gpu_backed, n_top_genes=n_top_genes, flavor="seurat_v3", device="gpu",
            )
            wall = time.perf_counter() - t0
            backend = "pyscx-gpu"
            # Teardown outside the timed region: surface the route + HVG var
            # columns on `a` for the route gate and the Jaccard overlap below.
            # Gene order is identical (the CSC fixture was built from `raw`).
            _propagate_route(gpu_backed, a)
            for col in _HVG_VAR_COLS:
                if col in gpu_backed.var.columns:
                    a.var[col] = gpu_backed.var[col].to_numpy()
        else:
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

        # Record the accelerator route pyscx stamped, plus numeric gate
        # signals for the GPU variant. The GPU variant is skipped on non-GPU
        # hosts, so a recorded cpu_* route means dispatch silently fell back.
        #   - hvg_route_gpu_correct: 1.0 iff a GPU route ran (gpu_csr or
        #     gpu_csc_v3), 0.0 on a silent CPU fallback.
        #   - hvg_route_csc_direct: when a CSC fixture was built, asserts the
        #     column-major reduce actually dispatched (route == gpu_csc_v3);
        #     1.0 when no CSC fixture was built (vacuous, mirrors DE's
        #     de_route_csc_direct), so it never false-fails on a non-CSC host.
        route = _extract_route(a, "highly_variable_genes")
        if route is not None:
            extras["gpu_dispatch_route"] = route
            if requires_gpu:
                extras["hvg_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )
                csc_fixture = scx_csc_path is not None
                extras["hvg_route_csc_direct"] = (
                    1.0 if (not csc_fixture or route == "gpu_csc_v3") else 0.0
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
