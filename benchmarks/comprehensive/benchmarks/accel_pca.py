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
    cpu_profile_capture,
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
        FormatVariant(
            name="pyscx PCA (GPU randomized, streaming power loop)",
            key="accel_pca__pyscx_gpu_streaming",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx PCA (GPU, backed multi-shard X)",
            key="accel_pca__pyscx_gpu_backed",
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


def _extract_resident_csr(adata: Any, op: str) -> bool | None:
    """Read ``adata.uns["scx_accel"][op]["resident_csr"]``, or None if absent.

    Absent (or `None`) means the route had no residency decision to make — a
    CPU or rapids route — and the caller records no arm label rather than
    guessing one.
    """
    try:
        value = adata.uns["scx_accel"][op]["resident_csr"]
    except (KeyError, TypeError, AttributeError):
        return None
    return None if value is None else bool(value)


def _run_pyscx_gpu_streaming(adata: Any, n_comps: int, seed: int) -> str:
    """The streaming GPU power loop — identical call to `_run_pyscx_gpu_rand_hh`,
    run under `SCX_GPU_PCA_RESIDENT=0`.

    Residency is decided dynamically against *free* VRAM, and on an 80 GB H100
    every gate dataset fits, so a default `accel_pca` run measures the resident
    arm on all three tiers. The streaming operator — the one Phase 8c's
    `PcaOperator` seam rewrote — has no coverage at all without this variant.

    `SCX_GPU_PCA_RESIDENT` is read **once per process** (`gpu_pca_resident.rs`
    documents this deliberately, and `pyscx/tests/test_gpu_pca_resident.py`
    uses subprocesses because of it), so the in-process context manager below is
    not sufficient on its own — `run_parallel.py` exports it in the worker's
    shell, and `pca_resident_csr` records which arm actually ran so a latched
    OnceLock fails the floor instead of silently mislabelling the number.
    """
    import pyscx

    pyscx.accel.pca(
        adata, n_comps=n_comps, device="gpu",
        method="randomized", qr_method="householder", random_state=seed,
    )
    return adata.uns["pca"].get("backend", "scx-gpu-cusparse")


# The backed-X arm's fixture. Keyed in `uns` rather than passed, because the
# variant impls take `(adata, n_comps, seed)` and nothing else — the same seam
# `accel_de` uses for its CSC fixture (`_bench_scx_with_csc_path`).
_BACKED_UNS_KEY = "_bench_scx_backed_path"


def _ensure_scx_backed_fixture(adata: Any, dataset_name: str) -> Path | None:
    """Materialise the preprocessed adata as an SCX file so PCA can run on a
    **backed, multi-shard** ``X``.

    Deliberately not a CSC fixture: PCA rejects `prefer_format="csc"` outright,
    so the sidecar would be built and never read. `csc="off"` skips it.

    `shard_size` is left at the writer's default on purpose — the point of this
    arm is the layout a user actually gets from `pyscx.from_anndata`, and
    pinning a bespoke value here would gate a layout nothing produces. The shard
    count is asserted at run time instead (see `_run_pyscx_gpu_backed`), so a
    default that drifts to one shard fails the floor rather than silently
    turning this arm into a copy of the in-memory ones.

    Cached on disk under ``$SCX_BENCH_TMPDIR/scx_pca_backed_fixtures/``;
    ``SCX_BENCH_REBUILD_BACKED=1`` forces a rebuild.
    """
    import os

    # NOTE: `Path("")` is `PosixPath(".")` and `.is_dir()` is True, so an unset
    # env must be detected on the raw STRING before constructing the Path — else
    # the fixture lands in the cwd instead of the /tmp fallback. Same trap, same
    # note, as `accel_preprocess._ensure_scx_fixture`.
    base_str = os.environ.get("SCX_BENCH_TMPDIR") or os.environ.get("SCX_WORK_DIR", "")
    out_dir = (
        Path(base_str) / "scx_pca_backed_fixtures"
        if base_str
        else Path("/tmp/scx_pca_backed_fixtures")
    )
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning("accel_pca: cannot create %s (%s); no backed fixture", out_dir, e)
        return None
    # The filename carries the matrix's shape and nnz. A bare
    # `{dataset}.bench_pca_backed.scx` in a shared `/tmp` is reused by any later
    # job for the same dataset name, so a fixture built from a *different*
    # preprocessed matrix (a changed HVG count, another branch's writer) would be
    # measured against this run's in-memory reference — incomparable cosines, and
    # on census no wall ceiling to notice the smaller problem.
    nnz = int(getattr(adata.X, "nnz", 0) or 0)
    ident = f"{int(adata.n_obs)}x{int(adata.n_vars)}x{nnz}"
    scx_path = out_dir / f"{dataset_name}.{ident}.bench_pca_backed.scx"
    rebuild = os.environ.get("SCX_BENCH_REBUILD_BACKED", "") in ("1", "true", "TRUE")
    if scx_path.exists() and not rebuild:
        logger.info("accel_pca: reusing cached backed fixture %s", scx_path)
        return scx_path
    if scx_path.exists():
        scx_path.unlink()
    try:
        import pyscx

        logger.info(
            "accel_pca: building backed fixture for %s -> %s (n_obs=%d n_vars=%d)",
            dataset_name, scx_path, int(adata.n_obs), int(adata.n_vars),
        )
        pyscx.from_anndata(adata, str(scx_path), csc="off")
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "accel_pca: pyscx.from_anndata(%s) failed: %s; the backed arm will "
            "report no measurement rather than silently measuring the in-memory path",
            dataset_name, e,
        )
        return None
    return scx_path


def _run_pyscx_gpu_backed(adata: Any, n_comps: int, seed: int) -> str:
    """GPU PCA over a **backed, multi-shard** ``X`` — the out-of-core path.

    This is the only arm that reaches `scx_gpu::gpu_pca::randomized_pca_core`
    with more than one shard, and therefore the only one that can observe
    anything about how the column-means pass iterates them. Every other arm
    here — `__pyscx_gpu_rand_hh`, `__pyscx_gpu_rand_chol` and
    `__pyscx_gpu_streaming` alike — runs on the runner's **in-memory** adata,
    which reaches GPU PCA through pyscx's `BorrowedCsrSource`, whose
    `n_shards()` is hard-coded 1. (`__pyscx_gpu_streaming` differs only by
    `SCX_GPU_PCA_RESIDENT=0`; it is the same single-shard source.) Measured:
    an in-memory `device="gpu"` PCA on pbmc3k reports
    `route: rapids_singlecell_gpu` and never enters the native path at all.

    Writes `X_pca` and the route stamp back onto the caller's adata so the
    runner's cosine / subspace correctness checks see this arm's embedding —
    it is the same matrix, only backed, so those numbers stay comparable to the
    sibling arms'.
    """
    import numpy as np
    import pyscx

    scx_path = adata.uns.get(_BACKED_UNS_KEY)
    if not scx_path:
        raise RuntimeError(
            "accel_pca__pyscx_gpu_backed: no backed fixture on this adata. "
            "Without one this arm would run on the in-memory X and report a "
            "single-shard number under a multi-shard name."
        )
    exp = pyscx.open(str(scx_path))
    n_shards = int(exp.shard_count)
    backed = exp.to_anndata(backed=True)
    pyscx.accel.pca(
        backed, n_comps=n_comps, device="gpu",
        method="randomized", qr_method="householder", random_state=seed,
    )
    adata.obsm["X_pca"] = np.asarray(backed.obsm["X_pca"])
    # The premise, carried as a number so it is gated rather than assumed.
    adata.uns["_bench_pca_backed_n_shards"] = n_shards
    # Assigned directly, and deliberately allowed to raise. A helper that
    # swallowed a failure here would turn a missing route stamp into a missing
    # `pca_route_gpu_correct` extra, which the floor gate reports as a violation
    # with no indication of the cause.
    adata.uns["scx_accel"] = dict(backed.uns["scx_accel"])
    adata.uns["pca"] = dict(backed.uns.get("pca") or {})
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


def is_streaming_variant(variant_key: str) -> bool:
    """True for the `accel_*__pyscx_gpu_streaming` arm (SCX_GPU_PCA_RESIDENT=0)."""
    return variant_key.endswith("__pyscx_gpu_streaming")


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


def streaming_pca_env():
    """Force the streaming GPU PCA power loop instead of the device-resident one.

    Composed with `force_native_gpu_env` rather than replacing it: the streaming
    arm is still a *native* GPU arm, so it needs the same rapids pin as its
    resident sibling.
    """
    return env_var("SCX_GPU_PCA_RESIDENT", "0")


@contextlib.contextmanager
def dispatch_env(variant_key: str, requires_gpu: bool):
    """The env context a variant's impl must run under:
    - no-rapids variant  → SCX_DISABLE_RAPIDS=1 (exercise the CPU fallback)
    - rapids variant     → no override (default in-VRAM route → rapids)
    - native gpu variant → SCX_FORCE_NATIVE_GPU=1 (pin native kernels)
    - streaming variant  → the above, plus SCX_GPU_PCA_RESIDENT=0
    - cpu variant        → nothing

    A real generator context manager, not a pre-entered `ExitStack`. The stack
    form called `enter_context()` during `dispatch_env(...)` evaluation rather
    than at `with` entry, so a bare call — or an exception raised between the
    two `enter_context()`s — leaked `SCX_FORCE_NATIVE_GPU` into the rest of the
    process. Every native GPU accel module (kNN / UMAP / Leiden / HVG /
    preprocess / pipeline) reaches this, not only PCA. Flagged by **Antigravity**
    and **Cursor Agent** on PR #474.
    """
    if is_no_rapids_variant(variant_key):
        with disable_rapids_env():
            yield
    elif requires_gpu and not is_rapids_variant(variant_key):
        with force_native_gpu_env():
            if is_streaming_variant(variant_key):
                with streaming_pca_env():
                    yield
            else:
                yield
    else:
        yield


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


# The streaming arm forces the slow power loop (SCX_GPU_PCA_RESIDENT=0 —
# re-decode and re-upload on every multiply, 3.2x the resident arm on tabula)
# and is floored only on these two tiers. Without a scope the orchestrator
# schedules it on census_1m as well: a different, ungated, much slower
# experiment than the one this PR measured (Cursor Agent, #474). Read by
# `run_parallel._bench_format_dataset_scope`.
FORMAT_DATASET_SCOPE: dict[str, frozenset[str]] = {
    "accel_pca__pyscx_gpu_streaming": frozenset({"pbmc3k", "tabula_sapiens_100k"}),
    # The backed arm needs a **multi-shard** source to mean anything, and shard
    # count follows n_obs: at the writer's default `shard_target_rows` (16384)
    # pbmc3k's 2,700 cells are one shard, where the decode-prefetch pipeline
    # takes its sequential fallback and this arm measures exactly what the
    # in-memory ones do. tabula_sapiens_100k gives 7 shards and census_500k 31.
    # `pca_backed_n_shards` is floored at >= 2 so a drifting default fails here
    # rather than quietly making the arm a duplicate.
    "accel_pca__pyscx_gpu_backed": frozenset({"tabula_sapiens_100k", "census_500k"}),
}


_VARIANT_IMPLS: dict[str, tuple[Callable[..., str], bool]] = {
    # key: (implementation, requires_gpu)
    "accel_pca__scanpy_cpu": (_run_scanpy_cpu, False),
    "accel_pca__pyscx_cpu_auto": (_run_pyscx_cpu_auto, False),
    "accel_pca__pyscx_gpu_rand_hh": (_run_pyscx_gpu_rand_hh, True),
    "accel_pca__pyscx_gpu_rand_chol": (_run_pyscx_gpu_rand_chol, True),
    "accel_pca__rapids_singlecell_gpu": (_run_rapids_singlecell, True),
    "accel_pca__pyscx_gpu_no_rapids": (_run_pyscx_gpu_no_rapids, True),
    "accel_pca__pyscx_gpu_streaming": (_run_pyscx_gpu_streaming, True),
    "accel_pca__pyscx_gpu_backed": (_run_pyscx_gpu_backed, True),
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

# Which PCA residency arm has already run in THIS process.
#
# `SCX_GPU_PCA_RESIDENT` is latched by a Rust `OnceLock` on the first native GPU
# PCA call, and no amount of `os.environ` juggling can reset it. So in a serial
# `run_all.py` pass the second arm silently produces the first arm's numbers
# under its own name. The opposing floors diagnose that *after* the capture is
# recorded; refusing the combination is what stops it being recorded at all.
# Found by **codex** on PR #474 (rounds 2 and 3). `run_parallel` is unaffected:
# one cohort per format_key, one process each, env exported before exec.
_RESIDENCY_ARM_RUN: str | None = None


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

    # Refuse a second residency arm in a process that already latched one.
    global _RESIDENCY_ARM_RUN
    if requires_gpu and not is_rapids_variant(variant_key):
        arm = "streaming" if is_streaming_variant(variant_key) else "resident"
        if _RESIDENCY_ARM_RUN is not None and _RESIDENCY_ARM_RUN != arm:
            logger.warning(
                "%s cannot run after the %s arm in the same process — "
                "SCX_GPU_PCA_RESIDENT is latched by a OnceLock on first use, so "
                "this cell would report the %s arm's numbers under the %s name",
                variant_key, _RESIDENCY_ARM_RUN, _RESIDENCY_ARM_RUN, arm,
            )
            write_missing_result(
                benchmark="accel_pca", format_key=variant_key, dataset=dataset.name,
                missing_reason="pca_residency_arm_latched",
                notes=(
                    f"the {_RESIDENCY_ARM_RUN} arm already ran in this process; "
                    "SCX_GPU_PCA_RESIDENT is read once per process. Use "
                    "run_parallel.py (one process per format_key) to measure both."
                ),
            )
            return None
        _RESIDENCY_ARM_RUN = arm

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

    # The backed arm needs an SCX file to open. Built once here, from the same
    # preprocessed matrix the in-memory arms use, and stashed in `uns` so it
    # survives the per-run `fixture.adata.copy()`. Built only for the arm that
    # wants it — every other variant would pay a whole conversion for a file it
    # never opens.
    if variant_key == "accel_pca__pyscx_gpu_backed":
        backed_path = _ensure_scx_backed_fixture(fixture.adata, dataset.name)
        if backed_path is None:
            logger.warning(
                "accel_pca__pyscx_gpu_backed: no backed fixture for %s; recording "
                "a missing result rather than measuring the in-memory path under "
                "this arm's name",
                dataset.name,
            )
            write_missing_result(
                benchmark="accel_pca", format_key=variant_key, dataset=dataset.name,
                missing_reason="backed_fixture_build_failed",
                notes="backed SCX fixture could not be built",
            )
            return None
        fixture.adata.uns[_BACKED_UNS_KEY] = str(backed_path)

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

        # 2.0 ranking oracle: capture the CPU decode/io/reduction/marshalling
        # breakdown for this op (no-op unless SCX_CPU_PROFILE=1). Populated into
        # `extras` → `runs[].extra`.
        extras: dict[str, Any] = {}

        rss_before = _get_rss_mb()
        u0, s0 = _get_cpu_times()
        t0 = time.perf_counter()

        with dispatch_env(variant_key, requires_gpu), cpu_profile_capture(extras):
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

        # Which power loop actually ran, as a number.
        #
        # `resident_csr` is True when the whole matrix was held device-resident
        # and False when the operator streamed (re-decoding and re-uploading on
        # every multiply). It is `None` where there is no residency decision —
        # CPU and rapids routes — so those variants record nothing.
        #
        # This is the arm label. Residency is decided against *free* VRAM at
        # call time, so the same input can go either way run to run and the
        # variant name alone cannot say which arm produced a wall. It is also
        # the only thing that catches `SCX_GPU_PCA_RESIDENT`'s OnceLock having
        # latched before the streaming variant's env was applied: the floors
        # pin `min 1.0` on the resident arm and `max 0.0` on the streaming one,
        # so a mislabelled run fails rather than reporting the wrong arm's number.
        resident = _extract_resident_csr(t_adata, "pca")
        if resident is not None:
            extras["pca_resident_csr"] = 1.0 if resident else 0.0

        # Backed arm: the shard count and this run's wall, as floorable extras.
        #
        # `wall_s` is a named `add_run` parameter, so it lands at the top level
        # of the record and the absolute-floor gate — which reads only
        # `runs[].extra` — cannot see it. A wall ceiling on this arm therefore
        # needs its own extra, or the floor resolves to "metric missing", which
        # counts as a violation on every run.
        #
        # `pca_backed_n_shards` is the premise. One shard means the decode
        # prefetch took its sequential fallback and this arm measured the same
        # thing the in-memory arms do; floored at >= 2 it fails loudly instead.
        if variant_key == "accel_pca__pyscx_gpu_backed":
            extras["pca_backed_wall_s"] = wall
            n_shards = t_adata.uns.get("_bench_pca_backed_n_shards")
            if n_shards is not None:
                extras["pca_backed_n_shards"] = float(n_shards)

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
