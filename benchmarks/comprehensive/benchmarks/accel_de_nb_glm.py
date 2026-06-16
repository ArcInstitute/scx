"""GPU pseudobulk NB-GLM differential-expression accelerator benchmark.

Exercises ``pyscx.accel.pdex_nb_glm`` (DESeq2-style negative-binomial GLM
pseudobulk DE, route ``gpu_nb_glm_csr``) CPU-vs-GPU on a synthetic, stratified
Perturb-seq fixture. Complements the per-cell DE bench (``accel_de.py``) — NB-GLM
is structurally different (it needs a replicate **stratifier** with ≥2 replicates
per condition, raw counts, and a many-perturbation regime), so it cannot ride
``accel_de``'s probe-the-real-dataset fixture and instead follows the
``cell_eval_parity_perf`` synthetic-in-process model.

## Variants

| ``format_variant.key``          | Path                       | Device |
|---------------------------------|----------------------------|--------|
| ``accel_de_nb_glm__pyscx_cpu``  | ``pyscx.accel.pdex_nb_glm``| CPU    |
| ``accel_de_nb_glm__pyscx_gpu``  | ``pyscx.accel.pdex_nb_glm``| GPU    |

The GPU variant additionally runs the CPU path (for CPU↔GPU concordance) and
``pdex_ref`` (a *different* algorithm — the only non-self-referential correctness
anchor, since pyDESeq2 OOMs in the comprehensive correctness reference).

## Signals (emitted into ``runs[].extra``)

Hard-gated (deterministic; see ``thresholds.yaml``):
  * ``nb_glm_route_gpu_correct`` — 1.0 iff a ``gpu_nb_glm_*`` route ran (0.0 on a
    silent GPU→CPU drop). The variant is skipped on non-GPU hosts, so a recorded
    route always means GPU was attempted.
  * ``nb_glm_cpu_gpu_concordant`` — 1.0 iff CPU↔GPU log2FC Spearman ≥ 0.999 AND
    max per-gene relative log2FC ≤ 2e-3. The 2e-3 bar is the precise-``f64``
    agreement floor: ``nb_glm.cu`` is built WITHOUT ``--use_fast_math`` — a
    fast-math/precision regression blows past it.
  * ``nb_glm_pdex_ref_concordant`` — 1.0 iff min(Spearman log2FC, Spearman FDR)
    vs ``pdex_ref`` ≥ 0.95.

Surfaced (node-dependent — tracked vs baseline, not absolute-floored):
  * ``nb_glm_gpu_speedup`` — CPU/GPU median wall ratio (same machine; scales with
    the GPU node's CPU core count, so deliberately not floored).
  * ``nb_glm_gpu_fit_genes_per_s`` — node-independent kernel throughput, read from
    the ``SCX_NBGLM_PROFILE`` profiler (``gpu_mle_fit + gpu_shrink_fit``).

Plus diagnostics ``gpu_dispatch_route`` / ``gpu_dispatch_fallback`` matching the
other accel modules.

## Skip semantics

On a non-GPU host the GPU variant returns ``None`` (no result JSON written) —
``compare_against_baseline.py`` keys absolute floors on whether the raw JSON
exists, so this is a "scoped-out triple" (silently skipped), NOT a missing-metric
violation. Writing a typed stub would create the file with no metric and
false-fail the floor. Mirrors ``accel_de``'s ``requires_gpu`` guard.

The benchmark only runs on synthetic datasets (``dataset.synthetic``); on any
other dataset it returns ``None``. Exercise it via
``run_parallel.py --benchmarks accel_de_nb_glm --datasets nb_glm_synth``.
"""

from __future__ import annotations

import logging
import os
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)

REFERENCE = "control"

# Shared pdex_nb_glm kwargs. min_cells_per_group=1: the synthetic fixture has
# small per-(pert,donor) blocks; we don't want the replicate filter to drop them.
_KW: dict[str, Any] = dict(
    groupby="perturbation",
    reference=REFERENCE,
    stratify_by=["donor"],
    min_cells_per_group=1,
)

_HAS_PYSCX = False
_HAS_PYSCX_GPU = False
try:
    import pyscx  # noqa: F401
    _HAS_PYSCX = True
    try:
        _HAS_PYSCX_GPU = bool(pyscx.accel.gpu_available())
    except Exception:
        _HAS_PYSCX_GPU = False
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------


def accel_de_nb_glm_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx pdex_nb_glm (CPU)",
            key="accel_de_nb_glm__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx pdex_nb_glm (GPU)",
            key="accel_de_nb_glm__pyscx_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


# requires_gpu per variant key.
_VARIANT_REQUIRES_GPU: dict[str, bool] = {
    "accel_de_nb_glm__pyscx_cpu": False,
    "accel_de_nb_glm__pyscx_gpu": True,
}


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """Peak RSS since process start, in MB (monotonic per-process)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _median_wall(fn: Callable[[], Any], reps: int) -> tuple[float, Any]:
    """Median wall-clock over ``reps`` runs; returns (median_s, last_result)."""
    times: list[float] = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rank correlation; ``None`` if degenerate. Uses scipy when
    available (index ``[0]`` for SciPy <1.10), else a numpy rank fallback."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    a, b = a[mask], b[mask]
    try:
        from scipy.stats import spearmanr
        rho = float(spearmanr(a, b)[0])
        return None if np.isnan(rho) else rho
    except ImportError:
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        if ra.std() == 0.0 or rb.std() == 0.0:
            return None
        return float(np.corrcoef(ra, rb)[0, 1])


def _to_pandas(df: Any) -> Any:
    """Coerce a polars (or pandas) DataFrame to pandas."""
    to_pd = getattr(df, "to_pandas", None)
    return to_pd() if callable(to_pd) else df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    key = format_variant.key
    if key not in _VARIANT_REQUIRES_GPU:
        return None
    requires_gpu = _VARIANT_REQUIRES_GPU[key]

    if not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        # No JSON written → the absolute-floor gate treats this as a
        # scoped-out triple, not a missing-metric violation (see module
        # docstring). Mirrors accel_de's requires_gpu guard.
        return None
    if not dataset.synthetic:
        logger.warning(
            "accel_de_nb_glm: dataset '%s' is not synthetic; skipping "
            "(needs the stratified replicate fixture from _pert_synth).",
            dataset.name,
        )
        return None

    import pyscx
    from benchmarks.comprehensive.benchmarks import _pert_synth

    # Cap repeats: pdex_nb_glm's CPU path at the many-perturbation scale costs
    # seconds-to-minutes per call, and the GPU variant runs gpu×reps + cpu×reps
    # + a pdex_ref anchor. 3 is enough for a stable median.
    reps = max(1, min(n_runs, 3))

    params = dict(dataset.synth_params)
    t_gen = time.perf_counter()
    adata = _pert_synth.make_raw_counts_stratified(**params)
    gen_s = time.perf_counter() - t_gen
    logger.info(
        "accel_de_nb_glm: fixture %s ready in %.1fs (n_obs=%d, n_vars=%d)",
        dataset.name, gen_s, int(adata.n_obs), int(adata.n_vars),
    )

    result = BenchmarkResult(
        benchmark="accel_de_nb_glm",
        format=key,
        dataset=dataset.name,
        metadata={
            "n_obs": int(adata.n_obs),
            "n_vars": int(adata.n_vars),
            "synth_params": params,
            "reps": reps,
            "device": "gpu" if requires_gpu else "cpu",
            "data_gen_s": round(gen_s, 3),
        },
    )

    baseline_rss = _peak_rss_mb()

    if not requires_gpu:
        # ── CPU baseline + pdex_ref anchor ───────────────────────────────
        # Warmup (discarded): first call pays one-time setup (rayon pool spin-up,
        # allocator warm) that would otherwise inflate the first timed rep.
        pyscx.accel.pdex_nb_glm(adata, device="cpu", **_KW)
        cpu_s, cpu_df = _median_wall(
            lambda: pyscx.accel.pdex_nb_glm(adata, device="cpu", **_KW), reps,
        )
        extra: dict[str, Any] = {}
        try:
            ref_df = pyscx.accel.pdex_ref(
                adata, "perturbation", reference=REFERENCE, is_log1p=False,
            )
            extra["nb_glm_pdex_ref_concordant"] = _pdex_ref_concordant(
                cpu_df, ref_df,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("accel_de_nb_glm: CPU pdex_ref anchor failed: %s", e)
        route = _route(adata)
        if route is not None:
            extra["gpu_dispatch_route"] = route
            result.metadata["route"] = route
        result.add_run(
            wall_s=cpu_s,
            peak_rss_mb=max(baseline_rss, _peak_rss_mb()),
            **extra,
        )
        return result

    # ── GPU variant: route + CPU↔GPU concordance + pdex_ref + perf ───────
    os.environ["SCX_NBGLM_PROFILE"] = "1"
    # Warmup (discarded, BEFORE the profiler reset): the first GPU call pays
    # one-time kernel load / JIT / context setup that would otherwise inflate
    # both the first timed rep's wall and the accumulated profiler fit_ms.
    pyscx.accel.pdex_nb_glm(adata, device="gpu", **_KW)
    pyscx.accel.nb_glm_profile_reset()
    gpu_s, gpu_df = _median_wall(
        lambda: pyscx.accel.pdex_nb_glm(adata, device="gpu", **_KW), reps,
    )
    prof = pyscx.accel.nb_glm_profile_snapshot()
    # Read the route immediately after the GPU runs — the CPU runs below
    # overwrite adata.uns["scx_accel"]["pdex_nb_glm"]["route"].
    route = _route(adata) or "<unknown>"
    fallback = _fallback(adata)

    cpu_s, cpu_df = _median_wall(
        lambda: pyscx.accel.pdex_nb_glm(adata, device="cpu", **_KW), reps,
    )

    extra = {
        "nb_glm_route_gpu_correct": 1.0 if route.startswith("gpu_nb_glm") else 0.0,
        "gpu_dispatch_route": route,
    }
    if fallback is not None:
        extra["gpu_dispatch_fallback"] = fallback

    # CPU↔GPU numerical concordance on log2FC.
    rho_cpu_gpu, max_rel = _cpu_gpu_agreement(cpu_df, gpu_df)
    if rho_cpu_gpu is not None:
        extra["nb_glm_cpu_gpu_log2fc_spearman"] = rho_cpu_gpu
    if max_rel is not None:
        extra["nb_glm_cpu_gpu_max_rel_log2fc"] = max_rel
    extra["nb_glm_cpu_gpu_concordant"] = (
        1.0
        if (rho_cpu_gpu is not None and max_rel is not None
            and rho_cpu_gpu >= 0.999 and max_rel <= 2e-3)
        else 0.0
    )

    # Independent anchor: NB-GLM (GPU) vs pdex_ref (a different algorithm).
    try:
        ref_df = pyscx.accel.pdex_ref(
            adata, "perturbation", reference=REFERENCE, is_log1p=False,
        )
        extra["nb_glm_pdex_ref_concordant"] = _pdex_ref_concordant(gpu_df, ref_df)
    except Exception as e:  # noqa: BLE001
        logger.warning("accel_de_nb_glm: GPU pdex_ref anchor failed: %s", e)
        extra["nb_glm_pdex_ref_concordant"] = 0.0

    # Surfaced perf (node-dependent — not floored).
    extra["nb_glm_gpu_wall_s"] = gpu_s
    extra["nb_glm_cpu_wall_s"] = cpu_s
    extra["nb_glm_gpu_speedup"] = (cpu_s / gpu_s) if gpu_s > 0 else None
    fit_ms = (
        prof.get("gpu_mle_fit", {}).get("ms", 0.0)
        + prof.get("gpu_shrink_fit", {}).get("ms", 0.0)
    )
    # Count the genes actually FITTED (post any internal filtering), not
    # adata.n_vars — the throughput must reflect the real kernel work, else it
    # over-reports when pdex_nb_glm drops genes. n_genes = rows / n_targets.
    try:
        g = _to_pandas(gpu_df)
        n_targets = int(g["target"].nunique())
        n_genes = int(g["feature"].nunique())
    except Exception:
        n_targets, n_genes = 0, 0
    # fit_ms accumulates across all `reps` GPU runs (profiler reset once), so
    # the work is n_targets·n_genes·reps fits over fit_ms milliseconds.
    extra["nb_glm_gpu_fit_genes_per_s"] = (
        (n_targets * n_genes * reps) / (fit_ms / 1e3) if fit_ms > 0 else None
    )

    result.metadata["route"] = route
    result.metadata["gpu_dispatch_route"] = route
    result.add_run(
        wall_s=gpu_s,
        peak_rss_mb=max(baseline_rss, _peak_rss_mb()),
        **extra,
    )
    return result


# ---------------------------------------------------------------------------
# Route + concordance helpers
# ---------------------------------------------------------------------------


def _route(adata: Any) -> str | None:
    try:
        return adata.uns["scx_accel"]["pdex_nb_glm"]["route"]
    except Exception:
        return None


def _fallback(adata: Any) -> str | None:
    try:
        return adata.uns["scx_accel"]["pdex_nb_glm"]["fallback_reason"]
    except Exception:
        return None


# Genes with |log2FC| below this carry essentially no effect; their relative
# log2FC error is dominated by a near-zero denominator (the kernels still agree
# in *absolute* terms — CPU↔GPU rank Spearman stays ~1.0). The precise-f64
# agreement floor is meaningful only where the LFC is well-determined, so the
# relative-tolerance check is restricted to this subset — mirroring the proven
# pytest's intent (its high-count fixture has no near-zero-LFC genes). The
# ranking-Spearman check below covers every gene.
_LFC_SIGNAL_FLOOR = 0.1


def _cpu_gpu_agreement(
    cpu_df: Any, gpu_df: Any,
) -> tuple[float | None, float | None]:
    """(Spearman log2FC over all genes, max relative |Δlog2FC| over signal genes)."""
    try:
        c = _to_pandas(cpu_df)
        g = _to_pandas(gpu_df)
        m = c.merge(g, on=["target", "feature"], suffixes=("_c", "_g"))
        if not len(m):
            return None, None
        lc = m["log2_fold_change_c"].to_numpy(dtype=np.float64)
        lg = m["log2_fold_change_g"].to_numpy(dtype=np.float64)
        rho = _spearman(lc, lg)
        fin = np.isfinite(lc) & np.isfinite(lg)
        sig = fin & (np.abs(lc) >= _LFC_SIGNAL_FLOOR)
        if not sig.any():
            return rho, None
        max_rel = float(
            (np.abs(lc[sig] - lg[sig]) / np.abs(lc[sig])).max()
        )
        return rho, max_rel
    except Exception as e:  # noqa: BLE001
        logger.warning("accel_de_nb_glm: CPU↔GPU agreement failed: %s", e)
        return None, None


def _pdex_ref_concordant(nb_df: Any, ref_df: Any) -> float:
    """1.0 iff min(Spearman log2FC, Spearman FDR) of NB-GLM vs pdex_ref ≥ 0.95."""
    try:
        nb = _to_pandas(nb_df)
        ref = _to_pandas(ref_df)
        m = nb.merge(ref, on=["target", "feature"], suffixes=("_nb", "_ref"))
        if not len(m):
            return 0.0
        rho_lfc = _spearman(
            m["log2_fold_change_nb"].to_numpy(),
            m["log2_fold_change_ref"].to_numpy(),
        )
        rho_fdr = _spearman(
            m["fdr_nb"].to_numpy(), m["fdr_ref"].to_numpy(),
        )
        if rho_lfc is None or rho_fdr is None:
            return 0.0
        return 1.0 if min(rho_lfc, rho_fdr) >= 0.95 else 0.0
    except Exception as e:  # noqa: BLE001
        logger.warning("accel_de_nb_glm: pdex_ref concordance failed: %s", e)
        return 0.0
