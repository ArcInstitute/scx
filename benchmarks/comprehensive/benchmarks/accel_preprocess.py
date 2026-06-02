"""
Preprocessing (normalize_total + log1p) accelerator benchmark — Phase 9.

Same pattern as the other accel_* modules. Each variant runs ONE
normalize+log1p implementation on a fresh AnnData loaded from the
h5ad fixture.

## Variants

| `format_variant.key` | Backend | Device | Materialization |
|---|---|---|---|
| `accel_preprocess__scanpy_cpu`   | scanpy.pp.{normalize_total,log1p} | CPU | eager |
| `accel_preprocess__pyscx_cpu`    | pyscx.accel.{normalize_total,log1p}(device="cpu") | CPU | lazy (ScxLazyTransformedDataset) |
| `accel_preprocess__pyscx_gpu`    | pyscx.accel.{normalize_total,log1p}(device="gpu") | GPU | eager (fusion marker triggers single GPU pass) |

Correctness: max absolute difference vs scanpy reference. For the GPU
path, ~1e-5 tolerance is expected (f32 vs f64 intermediate drift).
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
)
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)

# Subsample ceiling for the elementwise correctness diff. Full-matrix
# compare on census-scale data densifies both operands into 100+ GB
# arrays via scipy's sparse subtract fallbacks on certain lazy paths;
# sampling 10K rows keeps the estimate statistically meaningful without
# the memory cliff. Seeded for reproducibility.
_MAX_DIFF_SUBSAMPLE_ROWS = 10_000

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


def accel_preproc_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="scanpy normalize+log1p (CPU)",
            key="accel_preprocess__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx normalize+log1p (CPU lazy)",
            key="accel_preprocess__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx normalize+log1p (GPU eager)",
            key="accel_preprocess__pyscx_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


def _load_raw(dataset: DatasetConfig) -> Any:
    """Load raw h5ad (no preprocessing) — the `accel_preproc` variants are
    precisely what add the preprocessing, so the fixture starts fresh.
    """
    key = ("raw_adata", dataset.name)
    if key in _fixture_cache:
        return _fixture_cache[key]
    import anndata
    adata = anndata.read_h5ad(str(dataset.h5ad_path))
    _fixture_cache[key] = adata
    return adata


def _run_scanpy(adata: Any, target_sum: float = 1e4) -> str:
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    return "scanpy-cpu-eager"


def _run_pyscx_cpu(adata: Any, target_sum: float = 1e4) -> str:
    import pyscx
    pyscx.accel.normalize_total(adata, target_sum=target_sum, device="cpu")
    pyscx.accel.log1p(adata, device="cpu")
    return "pyscx-cpu-lazy"


def _run_pyscx_gpu(adata: Any, target_sum: float = 1e4) -> str:
    import pyscx
    pyscx.accel.normalize_total(adata, target_sum=target_sum, device="gpu")
    pyscx.accel.log1p(adata, device="gpu")
    return "pyscx-gpu-eager-fused"


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    "accel_preprocess__scanpy_cpu": (_run_scanpy, False),
    "accel_preprocess__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_preprocess__pyscx_gpu": (_run_pyscx_gpu, True),
}


def _max_abs_diff(a: Any, b: Any) -> float:
    """Max absolute difference between two X matrices (sparse-safe).

    Both operands are typically scipy CSR after normalize+log1p; densifying
    them at census scale (1M × 28K × f32 = 112 GB) OOMs. Subtract in-place
    sparse when possible — log1p and normalize_total preserve the sparsity
    pattern so `(a - b).nnz ≤ max(a.nnz, b.nnz)`.
    """
    import scipy.sparse as sp
    if a.shape != b.shape:
        return float("nan")
    if sp.issparse(a) and sp.issparse(b):
        diff = (a - b).tocsr()
        if diff.nnz == 0:
            return 0.0
        return round(float(np.max(np.abs(diff.data))), 6)
    xa = a.toarray() if sp.issparse(a) else np.asarray(a)
    xb = b.toarray() if sp.issparse(b) else np.asarray(b)
    return round(float(np.max(np.abs(xa - xb))), 6)


def _ensure_scx_fixture(raw: Any, dataset_name: str) -> Path | None:
    """Materialise `raw` as a plain SCX file (no CSC sidecar — normalize/log1p
    stream CSR shards) so the GPU variant can open it backed and actually hit
    the GPU shard-streaming kernel. Without a backed input,
    `normalize_total(device="gpu")` falls back to CPU on in-memory scipy X
    (no GPU kernel for materialized dense/sparse), so the "GPU" variant would
    never run on GPU.

    Cached under ``$SCX_BENCH_TMPDIR/preproc_fixtures/`` (default
    ``/tmp/preproc_fixtures``); ``SCX_BENCH_REBUILD_FIXTURES=1`` forces a rebuild
    (``SCX_BENCH_REBUILD_CSC=1`` is honoured as a backward-compat alias).
    Returns None if pyscx is missing or the conversion failed.
    """
    import os
    if not _HAS_PYSCX:
        return None
    # NOTE: Path("") is PosixPath(".") and .is_dir() is True, so an unset env
    # must be detected on the raw string before constructing the Path — else
    # the fixture lands in the cwd instead of the /tmp fallback.
    base_str = os.environ.get("SCX_BENCH_TMPDIR") or os.environ.get("SCX_WORK_DIR", "")
    out_dir = Path(base_str) / "preproc_fixtures" if base_str else Path("/tmp/preproc_fixtures")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("accel_preprocess: cannot create %s (%s)", out_dir, e)
        return None
    scx_path = out_dir / f"{dataset_name}.preproc.scx"
    rebuild = any(
        os.environ.get(k, "") in ("1", "true", "TRUE")
        for k in ("SCX_BENCH_REBUILD_FIXTURES", "SCX_BENCH_REBUILD_CSC")
    )
    if scx_path.exists() and not rebuild:
        return scx_path
    try:
        if scx_path.exists():
            scx_path.unlink()
        import pyscx
        pyscx.from_anndata(raw, str(scx_path))
    except Exception as e:
        logger.warning("accel_preprocess: pyscx.from_anndata(%s) failed: %s", dataset_name, e)
        return None
    return scx_path


def _open_backed(path: Path) -> Any:
    """Open an SCX file as a backed AnnData (X is ScxBackedSparseDataset), so
    GPU normalize/log1p take the eager shard-streaming kernel."""
    import pyscx
    return pyscx.open(str(path)).to_anndata(backed=True)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    target_sum: float = 1e4,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    impl, requires_gpu = _VARIANT_IMPLS[key]
    if key.startswith("accel_preprocess__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None

    raw = _load_raw(dataset)

    result = BenchmarkResult(
        benchmark="accel_preprocess",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "target_sum": target_sum,
            "n_obs": raw.n_obs,
            "n_vars": raw.n_vars,
        },
    )

    # Cache scanpy reference for max-abs-diff correctness check.
    ref_key = ("preproc_ref_X", dataset.name, target_sum)
    if ref_key not in _fixture_cache:
        ref = raw.copy()
        _run_scanpy(ref, target_sum)
        _fixture_cache[ref_key] = ref.X.copy() if hasattr(ref.X, "copy") else ref.X
    ref_X = _fixture_cache[ref_key]

    # The GPU variant must run on a backed SCX input — `normalize_total`/
    # `log1p(device="gpu")` only engage the GPU shard-streaming kernel for
    # ScxBackedSparseDataset / ScxLazyTransformedDataset X; on in-memory scipy
    # X they fall back to CPU. Build a plain SCX fixture once and open it backed
    # per iteration so the variant genuinely exercises (and can gate) the GPU
    # route. CPU/scanpy variants stay on the in-memory `raw.copy()`.
    gpu_scx_path = _ensure_scx_fixture(raw, dataset.name) if requires_gpu else None

    def _fresh() -> Any:
        if gpu_scx_path is not None:
            return _open_backed(gpu_scx_path)
        return raw.copy()

    # Warm-up
    for _ in range(N_WARMUP_RUNS):
        warm = _fresh()
        impl(warm, target_sum)
        # Materialize the lazy chain for pyscx_cpu so timing reflects
        # "preprocessing complete" rather than "transform enqueued".
        _ = warm.X[:100]
        del warm
        gc.collect()

    backend = ""
    diffs: list[float] = []
    for i in range(n_runs):
        gc.collect()
        a = _fresh()
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        backend = impl(a, target_sum)
        # Force materialization so the lazy-pyscx-CPU variant's wall time
        # reflects end-to-end (not just transform enqueue).
        if "pyscx_cpu" in key:
            _ = a.X[:, :]  # materializes the lazy chain
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        extras: dict[str, Any] = {}

        # Record the accelerator route + a numeric gate signal for the GPU
        # variant. The GPU variant runs on a backed SCX input (see `_fresh`),
        # so both normalize_total and log1p (device="gpu") take the GPU
        # shard-streaming kernel and stamp route gpu_csr. The gate signal
        # covers the *whole* normalize→log1p chain: a cpu_* route on either op
        # (a silent partial fallback, e.g. log1p drops to CPU), or log1p never
        # stamping a route at all, scores 0.0 and fails the gate. (CPU/scanpy
        # variants record the route for visibility but don't emit the signal.)
        route_norm = _extract_route(a, "normalize_total")
        route_log1p = _extract_route(a, "log1p")
        if route_norm is not None:
            extras["gpu_dispatch_route"] = route_norm
        if route_log1p is not None:
            extras["gpu_dispatch_route_log1p"] = route_log1p
        if requires_gpu and route_norm is not None:
            both_gpu = route_norm.startswith("gpu_") and (route_log1p or "").startswith("gpu_")
            extras["preprocess_route_gpu_correct"] = 1.0 if both_gpu else 0.0

        try:
            if raw.n_obs > _MAX_DIFF_SUBSAMPLE_ROWS:
                rng = np.random.default_rng(RANDOM_SEED)
                idx = np.sort(
                    rng.choice(raw.n_obs, _MAX_DIFF_SUBSAMPLE_ROWS, replace=False)
                )
                diffs.append(_max_abs_diff(ref_X[idx, :], a.X[idx, :]))
            else:
                diffs.append(_max_abs_diff(ref_X, a.X))
            if not np.isnan(diffs[-1]):
                extras["max_abs_diff_vs_scanpy"] = diffs[-1]
        except Exception as e:
            logger.warning("max-abs-diff failed for %s run %d: %s", key, i + 1, e)

        result.add_run(
            wall_s=wall, user_s=u1 - u0, sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs max_abs_diff=%s",
            key, i + 1, wall, diffs[-1] if diffs else float("nan"),
        )
        del a
        gc.collect()

    if diffs:
        diffs_clean = [d for d in diffs if not np.isnan(d)]
        if diffs_clean:
            result.metadata["max_abs_diff_vs_scanpy"] = round(
                float(np.median(diffs_clean)), 6
            )
    result.metadata["backend"] = backend
    return result
