"""
End-to-end PCA→kNN→UMAP residency benchmark.

This is the Phase-2 *residency* benchmark (V3 task 2.7): it measures the
device-resident fused pipeline (`pyscx.accel.pca_neighbors_umap`, which keeps the
PCA embedding → kNN graph → fuzzy graph on the GPU between stages — the
`DeviceEmbedding`/`DeviceKnnGraph`/`DeviceFuzzyGraph` handoffs from tasks
2.3/2.4) **against** the current host-boundary path that round-trips every
intermediate through host memory, a CPU reference, and the leading GPU-scanpy
stack (rapids-singlecell).

The handoff that 2.3/2.4 made device-resident is PCA→kNN→UMAP; preprocessing
(`normalize_total`/`log1p`/HVG) is supplied by the shared `AcceleratorRunner`
fixture (identical to `accel_pca`/`accel_knn`/`accel_umap`) so the four variants
are apples-to-apples on the same preprocessed matrix and only the residency of
the PCA→kNN→UMAP chain differs.

## Variants

| `format_variant.key` | Path | Device |
|---|---|---|
| `accel_pipeline__pyscx_cpu`            | pyscx pca → neighbors → umap (separate CPU calls) | CPU |
| `accel_pipeline__pyscx_gpu_resident`   | fused `pyscx.accel.pca_neighbors_umap` (device-resident) | GPU |
| `accel_pipeline__pyscx_gpu_hostboundary` | pyscx pca → neighbors → umap (separate GPU calls; host round-trip between stages) | GPU |
| `accel_pipeline__rapids_singlecell_gpu`  | `rsc.pp.pca` → `rsc.pp.neighbors` → `rsc.tl.umap` | GPU |

## Emitted metrics

| Field (`runs[].extra`) | Meaning |
|---|---|
| `pipeline_wall_s`        | end-to-end PCA→kNN→UMAP wall (== `wall_s`; the residency speedup signal) |
| `pca_subspace_cos_min`   | rotation-invariant PCA subspace cosine vs scanpy CPU |
| `knn_recall_vs_scanpy`   | neighbor-set Jaccard recall vs scanpy CPU |
| `umap_trustworthiness`   | sklearn trustworthiness of the UMAP embedding |
| `pipeline_route_gpu_correct` | resident variant only: 1.0 iff pca/neighbors/umap/pca_neighbors_umap all stamped `gpu_device_resident`, else 0.0 (a silent host-boundary fallback) |
| `graph_replay`/`math_mode`/`spmm_policy` | passthrough from `uns["scx_accel"]["pca"]` (visibility) |

The rapids-singlecell variant is informational this PR (no route gate): it is the
GPU competitor for the residency comparison. The broader per-op rapids-singlecell
matrix is V3 task 2.9.
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
    _run_scanpy_cpu,
    _subspace_principal_cosines,
    dispatch_env,
    is_rapids_variant,
)
from benchmarks.comprehensive.benchmarks.accel_knn import (
    _recall_at_k,
    _run_scanpy_neighbors,
)
from benchmarks.comprehensive.benchmarks.accel_umap import _trustworthiness
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


def accel_pipeline_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx pipeline (CPU)", key="accel_pipeline__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx pipeline (GPU device-resident)",
            key="accel_pipeline__pyscx_gpu_resident",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx pipeline (GPU host-boundary)",
            key="accel_pipeline__pyscx_gpu_hostboundary",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="rapids-singlecell pipeline (GPU)",
            key="accel_pipeline__rapids_singlecell_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


# ---------------------------------------------------------------------------
# Per-variant implementations. Each runs PCA→kNN→UMAP in-place on `adata`,
# leaving obsm["X_pca"], obsp["connectivities"], obsm["X_umap"] populated, and
# returns a backend identifier string.
# ---------------------------------------------------------------------------

def _run_pyscx_cpu(adata: Any, n_comps: int, n_neighbors: int, seed: int) -> str:
    import pyscx
    pyscx.accel.pca(adata, n_comps=n_comps, device="cpu", random_state=seed)
    pyscx.accel.neighbors(
        adata, n_neighbors=n_neighbors, device="cpu", random_state=seed
    )
    pyscx.accel.umap(adata, device="cpu", random_state=seed)
    return "scx-accel-cpu"


def _run_pyscx_gpu_resident(
    adata: Any, n_comps: int, n_neighbors: int, seed: int
) -> str:
    import pyscx
    # Fused device-resident path: PCA embedding → kNN graph → fuzzy graph stay
    # on the GPU between stages; only the final coordinates + connectivities
    # come back to host.
    pyscx.accel.pca_neighbors_umap(
        adata, n_comps=n_comps, n_neighbors=n_neighbors,
        device="gpu", random_state=seed,
    )
    return "scx-gpu-device-resident"


def _run_pyscx_gpu_hostboundary(
    adata: Any, n_comps: int, n_neighbors: int, seed: int
) -> str:
    import pyscx
    # Three separate GPU calls — each round-trips its result through host
    # memory before the next stage re-uploads it. This is the path the fused
    # resident variant replaces.
    pyscx.accel.pca(adata, n_comps=n_comps, device="gpu", random_state=seed)
    pyscx.accel.neighbors(
        adata, n_neighbors=n_neighbors, device="gpu", random_state=seed
    )
    pyscx.accel.umap(adata, device="gpu", random_state=seed)
    return "scx-gpu-host-boundary"


def _run_rapids_singlecell(
    adata: Any, n_comps: int, n_neighbors: int, seed: int
) -> str:
    """rapids-singlecell fused route (Phase 2): drives the pyscx fused
    `pca_neighbors_umap(device="gpu")`, which after Phase 1 runs the full rapids
    pipeline (rsc.pp.pca → rsc.pp.neighbors → rsc.tl.umap) and stamps
    `rapids_singlecell_gpu` on every stage. Results return to host for the
    correctness metrics."""
    import pyscx
    import rapids_singlecell as rsc

    pyscx.accel.pca_neighbors_umap(
        adata, n_comps=n_comps, n_neighbors=n_neighbors, device="gpu", random_state=seed,
    )
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    return _extract_route(adata, "pca_neighbors_umap") or "rapids_singlecell_gpu"


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    # key: (implementation, requires_gpu)
    "accel_pipeline__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_pipeline__pyscx_gpu_resident": (_run_pyscx_gpu_resident, True),
    "accel_pipeline__pyscx_gpu_hostboundary": (_run_pyscx_gpu_hostboundary, True),
    "accel_pipeline__rapids_singlecell_gpu": (_run_rapids_singlecell, True),
}


def _pipeline_route_gpu_correct(adata: Any) -> float:
    """1.0 iff the fused device-resident path stamped `gpu_device_resident` on
    every handoff op (pca / neighbors / umap) **and** the summary key; 0.0 on
    any silent host-boundary fallback. This realizes the device-residency
    route gate deferred from task 2.4.
    """
    expected = "gpu_device_resident"
    for op in ("pca", "neighbors", "umap", "pca_neighbors_umap"):
        if _extract_route(adata, op) != expected:
            return 0.0
    return 1.0


def _pipeline_route_rapids_correct(adata: Any) -> float:
    """1.0 iff the fused rapids pipeline stamped `rapids_singlecell_gpu` on every
    stage (pca / neighbors / umap) and the summary key; 0.0 otherwise (Phase 2)."""
    for op in ("pca", "neighbors", "umap", "pca_neighbors_umap"):
        if _extract_route(adata, op) != "rapids_singlecell_gpu":
            return 0.0
    return 1.0


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,  # unused — accelerators don't convert
    n_comps: int = 50,
    n_neighbors: int = 15,
) -> BenchmarkResult | None:
    """Execute the end-to-end residency benchmark for a single variant.

    Returns `None` for an unavailable variant (CPU-only host, missing pyscx);
    `run_parallel.py` treats `None` as a non-regression skipped cell. The
    rapids-singlecell variant, when GPU is present but the package is missing,
    writes a typed `no_rapids_singlecell` stub so the coverage gap is visible.
    """
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        logger.warning("Unknown pipeline variant %s — skipping", key)
        return None

    impl, requires_gpu = _VARIANT_IMPLS[key]

    if key.startswith("accel_pipeline__pyscx") and not _HAS_PYSCX:
        logger.warning("pyscx not installed — skipping %s", key)
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        logger.warning("GPU not available — skipping %s", key)
        return None
    if key == "accel_pipeline__rapids_singlecell_gpu" and not AcceleratorRunner.instance().has_rapids_singlecell():
        logger.warning("rapids-singlecell not installed — recording stub for %s", key)
        write_missing_result(
            benchmark="accel_pipeline",
            format_key=key,
            dataset=dataset.name,
            missing_reason="no_rapids_singlecell",
            notes="rapids-singlecell import failed; install into scx-bench-gpu",
        )
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    result = BenchmarkResult(
        benchmark="accel_pipeline",
        format=key,
        dataset=dataset.name,
        scenario={
            "name": "pca_neighbors_umap",
            "device": "gpu" if requires_gpu else "cpu",
            "n_comps": n_comps,
            "n_neighbors": n_neighbors,
        },
        comparison={
            "subject": {"impl": key},
            "baseline": {"impl": "accel_pipeline__pyscx_gpu_hostboundary"},
            "metric": "pipeline_wall_s",
            "status": "pending",
        },
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

    # Cache scanpy-CPU references (embedding + connectivities) for the per-stage
    # correctness metrics. Computed once per (dataset, n_comps, n_neighbors).
    ref_pca_key = ("ref_embedding", dataset.name, n_comps)
    ref_conn_key = ("ref_connectivities", dataset.name, n_neighbors)
    if ref_pca_key not in _fixture_cache or ref_conn_key not in _fixture_cache:
        ref = fixture.adata.copy()
        _run_scanpy_cpu(ref, n_comps, RANDOM_SEED)
        _fixture_cache[ref_pca_key] = np.asarray(
            ref.obsm["X_pca"], dtype=np.float32
        )
        _run_scanpy_neighbors(ref, n_neighbors, RANDOM_SEED)
        _fixture_cache[ref_conn_key] = ref.obsp["connectivities"]
        del ref
        gc.collect()
    ref_embedding: np.ndarray = _fixture_cache[ref_pca_key]
    ref_conn = _fixture_cache[ref_conn_key]

    # Warm-up
    for i in range(N_WARMUP_RUNS):
        logger.info("Warm-up run %d/%d for %s", i + 1, N_WARMUP_RUNS, key)
        warm = fixture.adata.copy()
        with dispatch_env(key, requires_gpu):
            impl(warm, n_comps, n_neighbors, RANDOM_SEED)
        del warm
        gc.collect()

    backend = ""
    sub_min_runs: list[float] = []
    recall_runs: list[float] = []
    trust_runs: list[float] = []

    for i in range(n_runs):
        gc.collect()
        a = fixture.adata.copy()

        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()

        with dispatch_env(key, requires_gpu):
            backend = impl(a, n_comps, n_neighbors, RANDOM_SEED)

        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        extras: dict[str, Any] = {"pipeline_wall_s": wall}

        # Per-stage correctness vs scanpy CPU.
        try:
            emb = np.asarray(a.obsm["X_pca"], dtype=np.float32)
            sub = _subspace_principal_cosines(ref_embedding, emb)
            sub_min_runs.append(float(np.min(sub)))
            extras["pca_subspace_cos_min"] = sub_min_runs[-1]
        except Exception as e:
            logger.warning("PCA subspace check failed for %s run %d: %s", key, i + 1, e)
        try:
            recall_runs.append(_recall_at_k(ref_conn, a.obsp["connectivities"]))
            extras["knn_recall_vs_scanpy"] = recall_runs[-1]
        except Exception as e:
            logger.warning("kNN recall check failed for %s run %d: %s", key, i + 1, e)
        try:
            pca = np.asarray(a.obsm["X_pca"], dtype=np.float32)
            um = np.asarray(a.obsm["X_umap"], dtype=np.float32)
            tw = _trustworthiness(pca, um, k=n_neighbors)
            if not np.isnan(tw):
                trust_runs.append(tw)
                extras["umap_trustworthiness"] = tw
        except Exception as e:
            logger.warning("UMAP trustworthiness failed for %s run %d: %s", key, i + 1, e)

        # Route gate: only the device-resident variant asserts the fused route.
        # Passthrough the PCA tuning metadata for visibility on the GPU variants.
        if key == "accel_pipeline__pyscx_gpu_resident":
            extras["pipeline_route_gpu_correct"] = _pipeline_route_gpu_correct(a)
            extras["gpu_dispatch_route"] = _extract_route(a, "pca_neighbors_umap")
        elif is_rapids_variant(key):
            extras["pipeline_route_rapids_correct"] = _pipeline_route_rapids_correct(a)
            extras["gpu_dispatch_route"] = _extract_route(a, "pca_neighbors_umap")
        try:
            pca_route_info = a.uns.get("scx_accel", {}).get("pca", {})
            for mk in ("graph_replay", "math_mode", "spmm_policy"):
                if pca_route_info.get(mk) is not None:
                    extras[mk] = pca_route_info[mk]
        except Exception:
            pass

        result.add_run(
            wall_s=wall,
            user_s=u1 - u0,
            sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        # Read this run's metrics from `extras` (nan if the check failed this
        # run) — not from the cross-run accumulators, whose `[-1]` would
        # mislabel a prior run's value as the current run's on a failure.
        logger.info(
            "  %s run %d: wall=%.3fs  rss=%.1fMB  subspace=%.4f recall=%.3f trust=%.3f",
            key, i + 1, wall, max(rss_before, rss_after),
            extras.get("pca_subspace_cos_min", float("nan")),
            extras.get("knn_recall_vs_scanpy", float("nan")),
            extras.get("umap_trustworthiness", float("nan")),
        )
        del a
        gc.collect()

    if sub_min_runs:
        result.metadata["pca_subspace_cos_min"] = round(
            float(np.median(sub_min_runs)), 6
        )
    if recall_runs:
        result.metadata["knn_recall_vs_scanpy"] = round(
            float(np.median(recall_runs)), 4
        )
    if trust_runs:
        result.metadata["umap_trustworthiness"] = round(
            float(np.median(trust_runs)), 4
        )
    result.metadata["backend"] = backend

    logger.info(
        "Benchmark complete: %s / %s — median %.3fs, subspace=%s recall=%s trust=%s",
        key, dataset.name, result.median_wall_s or 0.0,
        result.metadata.get("pca_subspace_cos_min", "n/a"),
        result.metadata.get("knn_recall_vs_scanpy", "n/a"),
        result.metadata.get("umap_trustworthiness", "n/a"),
    )
    return result
