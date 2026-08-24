"""
Harmony batch-integration accelerator benchmark.

Same pattern as `accel_pca.py` / `accel_knn.py` / `accel_leiden.py`.

| `format_variant.key` | Backend | Device |
|---|---|---|
| `accel_harmony__harmonypy_cpu` | harmonypy (torch, CPU) | CPU |
| `accel_harmony__pyscx_cpu`     | scx-accel Rust-native  | CPU |
| `accel_harmony__pyscx_gpu`     | scx-accel CUDA         | GPU |

**Why this module exists (Phase 7e, ORG-7.21-4).** Harmony was the last
documented parity claim in `scx-accel` with no gated benchmark and no
`thresholds.yaml` entry: this directory held 52 modules and none was Harmony.
The only harness was five ad-hoc SLURM scripts in `benchmarks/scripts/`, and
that directory's own README says new benchmarks must not go there.

**Correctness metric.** `mean_per_pc_r_vs_harmonypy` — the mean per-PC Pearson
correlation of the corrected embedding against harmonypy's, which is the same
quantity `docs/performance.md` reports against R harmony. It is *not* a
bit-parity claim and cannot be: SCX seeds k-means++ from `rand_chacha` where
harmonypy uses `sklearn.KMeans`, and the two objective cross-entropies differ by
a `log(2)` term (pinned as a divergence in
`scx-accel/src/harmony/harmony_reference_values.rs`). The exact numerics are
gated Rust-side by that module under plain `cargo test`; this floor is the
end-to-end sanity net, at the resolution a stochastic init allows.

**`harmony_route_gpu_correct`** is the dispatch gate: `device="gpu"` reporting a
`cpu_*` route means device resolution silently fell back, which is a hard gate
failure rather than a slow run.
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
from benchmarks.comprehensive.benchmarks.accel_knn import _ensure_pca
from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result
from benchmarks.comprehensive.runners.accel_runner import AcceleratorRunner

logger = logging.getLogger(__name__)

_HAS_PYSCX = AcceleratorRunner.instance().has_pyscx()
_HAS_PYSCX_GPU = AcceleratorRunner.instance().has_gpu()

# Obs columns that carry a real batch structure, most specific first. A dataset
# with none of them gets `_SYNTHETIC_BATCH` — deterministic thirds by row index.
# That still exercises the full algorithm at the right scale, which is what the
# timing floors measure; the correctness floor compares SCX against harmonypy on
# *the same* labels either way. `metadata["batch_key"]` records which was used,
# because a timing comparison across datasets is not meaningful if one of them
# had 3 synthetic batches and another had 40 real donors.
_BATCH_CANDIDATES = ("batch", "donor_id", "sample", "sample_id", "dataset_id", "assay")
_SYNTHETIC_BATCH = "_harmony_bench_batch"
_SYNTHETIC_N_BATCHES = 3

# Harmony's own defaults, pinned here so all three arms are timed on the same
# work. `max_iter_kmeans` is the sub-loop the Phase 7e M-step runs in, so it is
# the parameter this benchmark is most sensitive to.
_MAX_ITER = 10
_MAX_ITER_KMEANS = 6


def accel_harmony_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="harmonypy (CPU)", key="accel_harmony__harmonypy_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx Harmony (Rust-native CPU)", key="accel_harmony__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx Harmony (CUDA GPU)", key="accel_harmony__pyscx_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


def _resolve_batch_key(adata: Any) -> str:
    for col in _BATCH_CANDIDATES:
        if col not in adata.obs.columns:
            continue
        n_levels = adata.obs[col].astype(str).nunique()
        # One level gives Harmony nothing to correct; a per-cell column is not a
        # batch. Both make the run meaningless rather than merely uninteresting.
        if 2 <= n_levels <= max(2, len(adata) // 10):
            return col
    adata.obs[_SYNTHETIC_BATCH] = (
        np.arange(len(adata)) % _SYNTHETIC_N_BATCHES
    ).astype(str)
    return _SYNTHETIC_BATCH


def _run_harmonypy(adata: Any, batch_key: str, _seed: int) -> str:
    import harmonypy
    import pandas as pd

    ho = harmonypy.run_harmony(
        adata.obsm["X_pca"],
        pd.DataFrame({batch_key: adata.obs[batch_key].astype(str).values}),
        [batch_key],
        max_iter_harmony=_MAX_ITER,
        max_iter_kmeans=_MAX_ITER_KMEANS,
        verbose=False,
        # Pinned, not left to `device=None`. harmonypy 0.2.0 is torch-backed and
        # picks a device itself, so on a GPU node this variant would silently
        # become a GPU run under a name that says CPU — and the reference
        # embedding every other arm is scored against would change with the node
        # the job landed on.
        device="cpu",
    )
    adata.obsm["X_pca_harmony"] = np.asarray(ho.Z_corr, dtype=np.float32)
    return "harmonypy"


def _run_pyscx_cpu(adata: Any, batch_key: str, seed: int) -> str:
    import pyscx

    pyscx.accel.harmony_integrate(
        adata, batch_key, device="cpu", random_state=seed,
        max_iter=_MAX_ITER, max_iter_kmeans=_MAX_ITER_KMEANS,
    )
    return adata.uns.get("harmony", {}).get("backend", "scx-accel-cpu")


def _run_pyscx_gpu(adata: Any, batch_key: str, seed: int) -> str:
    import pyscx

    pyscx.accel.harmony_integrate(
        adata, batch_key, device="gpu", random_state=seed,
        max_iter=_MAX_ITER, max_iter_kmeans=_MAX_ITER_KMEANS,
    )
    return adata.uns.get("harmony", {}).get("backend", "scx-accel-gpu")


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    "accel_harmony__harmonypy_cpu": (_run_harmonypy, False),
    "accel_harmony__pyscx_cpu": (_run_pyscx_cpu, False),
    "accel_harmony__pyscx_gpu": (_run_pyscx_gpu, True),
}


def _mean_per_pc_r(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-PC Pearson r between two corrected embeddings.

    A constant PC has zero variance and `np.corrcoef` returns NaN for it; those
    are dropped rather than counted as 0.0, which would make the floor a
    function of how many degenerate PCs a dataset happens to carry.
    """
    if a.shape != b.shape:
        return float("nan")
    rs = []
    for pc in range(a.shape[1]):
        x, y = a[:, pc], b[:, pc]
        if x.std() == 0.0 or y.std() == 0.0:
            continue
        rs.append(abs(float(np.corrcoef(x, y)[0, 1])))
    return round(float(np.mean(rs)), 4) if rs else float("nan")


def _has_harmonypy() -> bool:
    try:
        import harmonypy  # noqa: F401

        return True
    except ImportError:
        return False


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
    n_comps: int = 50,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_IMPLS:
        return None
    impl, requires_gpu = _VARIANT_IMPLS[key]
    if key.startswith("accel_harmony__pyscx") and not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        return None
    if not _has_harmonypy():
        # Not a soft skip: harmonypy is both a variant and the correctness
        # reference for the other two, so its absence silently turns the
        # `mean_per_pc_r_vs_harmonypy` floor into a metric nothing reports —
        # which the gate reads as "no violation". Say so in the results.
        write_missing_result(
            benchmark="accel_harmony", format_key=key, dataset=dataset.name,
            missing_reason="no_harmonypy",
            notes="harmonypy import failed; it is the reference for "
                  "mean_per_pc_r_vs_harmonypy. `pip install harmonypy` into "
                  "this job's env (present in scx-bench and scx-gpu, absent "
                  "from scx-bench-gpu).",
        )
        return None

    fixture = _load_preprocessed(dataset, n_comps=n_comps)

    # Shared PCA + batch labels, cached across the three variants.
    base_key = ("harmony_base", dataset.name, n_comps)
    if base_key not in _fixture_cache:
        base = fixture.adata.copy()
        _ensure_pca(base, n_comps)
        bk = _resolve_batch_key(base)
        _fixture_cache[base_key] = (base, bk)
    base, batch_key = _fixture_cache[base_key]

    n_batches = int(base.obs[batch_key].astype(str).nunique())

    result = BenchmarkResult(
        benchmark="accel_harmony",
        format=key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "n_comps": n_comps,
            "max_iter": _MAX_ITER,
            "max_iter_kmeans": _MAX_ITER_KMEANS,
            "batch_key": batch_key,
            "batch_key_synthetic": batch_key == _SYNTHETIC_BATCH,
            "n_batches": n_batches,
            "n_obs": fixture.n_obs,
            "n_vars": fixture.n_vars,
            "random_seed": RANDOM_SEED,
        },
    )

    # harmonypy reference embedding, for the correctness floor. Computed once
    # per dataset and reused by all three variants — including the harmonypy
    # variant itself, where the comparison is against a *different run* of the
    # same implementation and therefore measures its own init stochasticity.
    # That is the useful control: it is the floor below which no arm can be
    # expected to land.
    ref_key = ("harmony_ref", dataset.name, n_comps)
    if ref_key not in _fixture_cache:
        ref = base.copy()
        _run_harmonypy(ref, batch_key, RANDOM_SEED)
        _fixture_cache[ref_key] = np.asarray(
            ref.obsm["X_pca_harmony"], dtype=np.float64
        )
        del ref
        gc.collect()
    ref_z: np.ndarray = _fixture_cache[ref_key]

    for _ in range(N_WARMUP_RUNS):
        warm = base.copy()
        impl(warm, batch_key, RANDOM_SEED)
        del warm
        gc.collect()

    backend = ""
    rs: list[float] = []
    for i in range(n_runs):
        gc.collect()
        a = base.copy()
        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()
        backend = impl(a, batch_key, RANDOM_SEED)
        wall = time.perf_counter() - t0
        u1, s1 = _get_cpu_times()
        rss_after = _get_rss_mb()

        extras: dict[str, Any] = {}
        try:
            got = np.asarray(a.obsm["X_pca_harmony"], dtype=np.float64)
            r = _mean_per_pc_r(ref_z, got)
            if not np.isnan(r):
                rs.append(r)
                extras["mean_per_pc_r_vs_harmonypy"] = r
        except Exception as e:
            logger.warning("per-PC r failed for %s run %d: %s", key, i + 1, e)

        route = _extract_route(a, "harmony_integrate")
        if route is not None:
            extras["gpu_dispatch_route"] = route
            if requires_gpu:
                extras["harmony_route_gpu_correct"] = (
                    1.0 if route.startswith("gpu_") else 0.0
                )

        result.add_run(
            wall_s=wall, user_s=u1 - u0, sys_s=s1 - s0,
            peak_rss_mb=max(rss_before, rss_after),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs r=%s backend=%s",
            key, i + 1, wall, rs[-1] if rs else float("nan"), backend,
        )
        del a
        gc.collect()

    if rs:
        result.metadata["mean_per_pc_r_vs_harmonypy"] = round(
            float(np.median(rs)), 4
        )
    result.metadata["backend"] = backend
    return result
