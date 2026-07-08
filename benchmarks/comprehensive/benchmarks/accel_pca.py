"""
GPU / CPU PCA accelerator benchmark.

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
| `accel_pca__pyscx_gpu_rand_hh`     | pyscx     | GPU | randomized                    | householder |
| `accel_pca__pyscx_gpu_rand_chol`   | pyscx     | GPU | randomized                    | cholesky |

## Emitted metrics

| Field | Where stored |
|---|---|
| Wall time per run | standard `BenchmarkResult.runs[i].wall_s` |
| Peak RSS per run  | standard `peak_rss_mb` |
| Cosine similarity (min / mean over top-k PCs vs scanpy CPU) | `metadata["cosine_sim_min"]`, `metadata["cosine_sim_mean"]` |
| Subspace principal-angle cosines (rotation/permutation/sign invariant) | `metadata["subspace_cos_min"]`, `metadata["subspace_cos_mean"]`, plus per-run keys in `runs[].extra` for gating |
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

import contextlib
import gc
import logging
import os
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
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result
from benchmarks.comprehensive.rss import current_rss_mb as _get_rss_mb
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
            name="pyscx PCA (GPU randomized, Householder)",
            key="accel_pca__pyscx_gpu_rand_hh",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (GPU randomized, Cholesky)",
            key="accel_pca__pyscx_gpu_rand_chol",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="rapids-singlecell PCA (GPU)",
            key="accel_pca__rapids_singlecell_gpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (GPU, rapids disabled → CPU fallback)",
            key="accel_pca__pyscx_gpu_no_rapids",
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


def _run_rapids_singlecell(adata: Any, n_comps: int, seed: int) -> str:
    """rapids-singlecell PCA route: drives the pyscx
    in-VRAM `device="gpu"` path, which after Phase 1 hands off to
    rapids-singlecell (`rsc.pp.pca`). This exercises + gates the real pyscx→rapids
    handoff (not raw rsc). `anndata_to_CPU(convert_all=True)` round-trips obsm so
    X_pca returns to host for the subspace-cosine correctness check.
    """
    import pyscx
    import rapids_singlecell as rsc

    pyscx.accel.pca(adata, n_comps=n_comps, device="gpu", random_state=seed)
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    return _extract_route(adata, "pca") or "rapids_singlecell_gpu"


def _run_pyscx_gpu_no_rapids(adata: Any, n_comps: int, seed: int) -> str:
    """Phase 2.2 no-rapids fallback: `device="gpu"` under SCX_DISABLE_RAPIDS=1
    (set by the run loop) must fall back to CPU with fallback_reason="no_rapids".
    """
    import pyscx

    pyscx.accel.pca(adata, n_comps=n_comps, device="gpu", random_state=seed)
    return adata.uns["pca"].get("backend", "scx-accel-cpu")


def _extract_route(adata: Any, op: str) -> str | None:
    """Read ``adata.uns["scx_accel"][op]["route"]``, or None if absent.

    Shared by the other accel benchmark modules (kNN / UMAP / Leiden /
    preprocess), which import it from here, to keep the route-gate signal
    convention identical across ops.
    """
    try:
        return adata.uns["scx_accel"][op]["route"]
    except Exception:
        return None


def _extract_fallback_reason(adata: Any, op: str) -> str | None:
    """Read ``adata.uns["scx_accel"][op]["fallback_reason"]``, or None if absent."""
    try:
        return adata.uns["scx_accel"][op]["fallback_reason"]
    except Exception:
        return None


# --- route-gate helpers (shared across accel_*) ---------------------------------
#
# Phase 1 flipped the in-VRAM `device="gpu"` default to rapids-singlecell. So on
# the scx-bench-gpu env (rapids installed) the native `pyscx_gpu_*` variants would
# route to rapids and break the native `*_route_gpu_correct` floors. We therefore
# (a) force the native kernels for the native variants via SCX_FORCE_NATIVE_GPU,
# and (b) carry the rapids route gate on the `*__rapids_singlecell_gpu` variant,
# which now drives pyscx (default → rapids) and asserts `rapids_singlecell_gpu`.

def is_rapids_variant(variant_key: str) -> bool:
    """True for the `accel_*__rapids_singlecell_gpu` variant."""
    return variant_key.endswith("__rapids_singlecell_gpu")


def is_no_rapids_variant(variant_key: str) -> bool:
    """True for the Phase 2.2 `accel_*__pyscx_gpu_no_rapids` fallback variant."""
    return variant_key.endswith("__pyscx_gpu_no_rapids")


@contextlib.contextmanager
def env_var(name: str, value: str):
    """Set ``os.environ[name]=value`` for the duration, restoring the prior value."""
    prev = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = prev


def force_native_gpu_env():
    """Pin the native SCX GPU kernels (not rapids) for the duration — used to
    wrap the native `pyscx_gpu_*` variant impls so their `*_route_gpu_correct`
    floors keep gating the native path after the Phase 1 default-flip."""
    return env_var("SCX_FORCE_NATIVE_GPU", "1")


def disable_rapids_env():
    """Force the rapids-absent fallback path (Phase 2.2 no-rapids gate)."""
    return env_var("SCX_DISABLE_RAPIDS", "1")


def dispatch_env(variant_key: str, requires_gpu: bool):
    """The env context a variant's impl must run under:
    - no-rapids variant  → SCX_DISABLE_RAPIDS=1 (exercise the CPU fallback)
    - rapids variant     → no override (default in-VRAM route → rapids)
    - native gpu variant → SCX_FORCE_NATIVE_GPU=1 (pin native kernels)
    - cpu variant        → nothing
    """
    if is_no_rapids_variant(variant_key):
        return disable_rapids_env()
    if requires_gpu and not is_rapids_variant(variant_key):
        return force_native_gpu_env()
    return contextlib.nullcontext()


def emit_route_signal(
    extras: dict[str, Any],
    variant_key: str,
    route: str | None,
    *,
    requires_gpu: bool,
    gpu_metric: str,
    rapids_metric: str,
    fallback_reason: str | None = None,
    no_rapids_metric: str | None = None,
) -> None:
    """Record `gpu_dispatch_route` + the numeric route-gate signal for the variant.

    - rapids variant     → `rapids_metric` = 1.0 iff route == "rapids_singlecell_gpu"
    - no-rapids variant  → `no_rapids_metric` = 1.0 iff route is cpu_* and
                           fallback_reason == "no_rapids"
    - native gpu variant → `gpu_metric` = 1.0 iff route starts with "gpu_"
    - cpu variant        → only the human-readable `gpu_dispatch_route` is recorded
    """
    if route is None:
        return
    extras["gpu_dispatch_route"] = route
    if is_rapids_variant(variant_key):
        extras[rapids_metric] = 1.0 if route == "rapids_singlecell_gpu" else 0.0
    elif is_no_rapids_variant(variant_key):
        if no_rapids_metric is not None:
            extras[no_rapids_metric] = (
                1.0 if route.startswith("cpu") and fallback_reason == "no_rapids" else 0.0
            )
    elif requires_gpu:
        extras[gpu_metric] = 1.0 if route.startswith("gpu_") else 0.0


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    # key: (implementation, requires_gpu)
    "accel_pca__scanpy_cpu": (_run_scanpy_cpu, False),
    "accel_pca__pyscx_cpu_auto": (_run_pyscx_cpu_auto, False),
    "accel_pca__pyscx_gpu_rand_hh": (_run_pyscx_gpu_rand_hh, True),
    "accel_pca__pyscx_gpu_rand_chol": (_run_pyscx_gpu_rand_chol, True),
    "accel_pca__rapids_singlecell_gpu": (_run_rapids_singlecell, True),
    "accel_pca__pyscx_gpu_no_rapids": (_run_pyscx_gpu_no_rapids, True),
}


# ---------------------------------------------------------------------------
# Correctness: cosine similarity vs scanpy reference
# ---------------------------------------------------------------------------

def _sign_agnostic_cosine_per_pc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise sign-agnostic cosine similarity between two (n_obs × k)
    PCA embeddings. PC signs are arbitrary, so we take abs(dot / norms).

    Note: this metric is sensitive to the basis (rotation, permutation)
    chosen by each implementation. Randomized PCA returns a rotated
    subspace by construction, so per-PC cosines drop well below 1.0
    even when the *subspace* itself is exact. Use
    ``_subspace_principal_cosines`` for the rotation-invariant
    correctness metric.
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


def _subspace_principal_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between the column-spans of ``a``
    and ``b`` — the rotation-invariant subspace correctness metric.

    Given two embeddings of shape (n × k), returns a length-k vector of
    cosines (descending). All k entries equal 1.0 iff the two
    column-spans coincide as subspaces — invariant to basis rotation,
    column permutation, and column-sign flips. The smallest entry is
    the strict measure: a single component direction missing from
    ``b`` drops it below 1.0 in proportion to the missing energy.

    Algorithm: orthonormalize each input via QR, form the k×k overlap
    matrix ``Q_a^T Q_b``, and return its singular values (the cosines
    of the principal angles between the subspaces).

    Numerical notes
    ---------------
    * Computed in float64 regardless of input dtype — float32 PCA
      embeddings on GPU would otherwise produce a noisy SVD on poorly
      conditioned columns.
    * Singular values are clipped into [0.0, 1.0]: SVD of a near-
      orthonormal product can return ``1.0 + ε`` (or tiny negatives
      via cancellation). The cosine interpretation is bounded.
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    a64 = np.ascontiguousarray(a, dtype=np.float64)
    b64 = np.ascontiguousarray(b, dtype=np.float64)
    q_a, _ = np.linalg.qr(a64)
    q_b, _ = np.linalg.qr(b64)
    s = np.linalg.svd(q_a.T @ q_b, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


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
    if variant_key == "accel_pca__rapids_singlecell_gpu" and not AcceleratorRunner.instance().has_rapids_singlecell():
        logger.warning("rapids-singlecell not installed — recording stub for %s", variant_key)
        write_missing_result(
            benchmark="accel_pca", format_key=variant_key, dataset=dataset.name,
            missing_reason="no_rapids_singlecell",
            notes="rapids-singlecell import failed; install into scx-bench-gpu",
        )
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    result = BenchmarkResult(
        benchmark="accel_pca",
        format=variant_key,
        dataset=dataset.name,
        scenario={
            "name": "pca",
            "device": "gpu" if requires_gpu else "cpu",
            "n_comps": n_comps,
        },
        comparison={
            "subject": {"impl": variant_key},
            "baseline": {"impl": "accel_pca__scanpy_cpu"},
            "metric": "subspace_cos_min",
            "status": "pending",
        },
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
        with dispatch_env(variant_key, requires_gpu):
            impl(warm_adata, n_comps, RANDOM_SEED)
        del warm_adata
        gc.collect()

    backend = ""
    cos_mean_runs: list[float] = []
    cos_min_runs: list[float] = []
    subspace_min_runs: list[float] = []
    subspace_mean_runs: list[float] = []

    for i in range(n_runs):
        gc.collect()
        t_adata = fixture.adata.copy()

        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()

        with dispatch_env(variant_key, requires_gpu):
            backend = impl(t_adata, n_comps, RANDOM_SEED)

        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        # Cosine similarity vs scanpy reference (top-k PCs).
        extras: dict[str, Any] = {}
        try:
            emb = np.asarray(t_adata.obsm["X_pca"], dtype=np.float32)
            cos = _sign_agnostic_cosine_per_pc(ref_embedding, emb)
            cos_mean_runs.append(float(np.mean(cos)))
            cos_min_runs.append(float(np.min(cos)))
            extras["cosine_sim_mean"] = cos_mean_runs[-1]
            extras["cosine_sim_min"] = cos_min_runs[-1]
            # Rotation/permutation/sign-invariant subspace metric. Catches
            # real correctness regressions on randomized variants that the
            # per-PC cosine misses by construction.
            sub = _subspace_principal_cosines(ref_embedding, emb)
            subspace_min_runs.append(float(np.min(sub)))
            subspace_mean_runs.append(float(np.mean(sub)))
            extras["subspace_cos_min"] = subspace_min_runs[-1]
            extras["subspace_cos_mean"] = subspace_mean_runs[-1]
        except Exception as e:
            logger.warning("Cosine-sim check failed for %s run %d: %s",
                           variant_key, i + 1, e)

        # Record the accelerator route + a numeric gate signal for the GPU
        # variants. GPU PCA runs the cuSPARSE+cuBLAS route (gpu_csr) for any
        # X type, including in-memory scipy; the variant is skipped on non-GPU
        # hosts, so a cpu_* route here is a silent fallback (e.g. the cuSPARSE
        # modern-ABI probe failed) → 0.0 fails the gate.
        emit_route_signal(
            extras,
            variant_key,
            _extract_route(t_adata, "pca"),
            requires_gpu=requires_gpu,
            gpu_metric="pca_route_gpu_correct",
            rapids_metric="pca_route_rapids_correct",
            fallback_reason=_extract_fallback_reason(t_adata, "pca"),
            no_rapids_metric="pca_fallback_no_rapids_correct",
        )

        result.add_run(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs  rss=%.1fMB  cos_min=%.4f  subspace_cos_min=%.4f",
            variant_key, i + 1, wall, max(rss_before, rss_after),
            cos_min_runs[-1] if cos_min_runs else float("nan"),
            subspace_min_runs[-1] if subspace_min_runs else float("nan"),
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
    if subspace_min_runs:
        result.metadata["subspace_cos_min"] = round(
            float(np.median(subspace_min_runs)), 6
        )
        result.metadata["subspace_cos_mean"] = round(
            float(np.median(subspace_mean_runs)), 6
        )
    result.metadata["backend"] = backend

    logger.info(
        "Benchmark complete: %s / %s — median %.3fs, cos_min=%s, subspace_cos_min=%s",
        variant_key, dataset.name,
        result.median_wall_s or 0.0,
        result.metadata.get("cosine_sim_min", "n/a"),
        result.metadata.get("subspace_cos_min", "n/a"),
    )
    return result


# Timing helpers live on `AcceleratorRunner`; the back-compat shims
# (`_get_rss_mb`, `_get_cpu_times`) near the top of this module delegate
# to them.
