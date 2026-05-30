"""
Cell-eval / arc-bench parity performance benchmark.

Head-to-head wall-clock + peak-RSS comparison of the SCX-accelerated
perturbation metrics (``pyscx.accel.*``) against their Python reference
implementations in ``cell-eval`` and ``arc-bench``. Complements the
correctness parity suite at ``pyscx/tests/test_cell_eval_parity.py`` by
running at benchmark-relevant scales (100K / 500K / 1M cells).

Results are emitted as a single ``BenchmarkResult`` per dataset, with
per-operation wall times, median, speedup, and peak RSS stashed in
``metadata["operations"]``. Reporting picks these up via
``load_all_results(benchmark="cell_eval_parity_perf")``.

Operations benchmarked (skipped automatically at sizes where they become
infeasible):

  * ``pseudobulk`` — SCX ``pseudobulk_means`` vs
    ``PerturbationAnndataPair._bulk_anndata``
  * ``bulk_metrics`` — SCX ``perturbation_metrics`` (5 metrics bundled)
    vs cell-eval ``pearson_delta`` + ``mse`` + ``mae`` + ``mse_delta``
    + ``mae_delta``
  * ``discrimination_l1`` — SCX ``discrimination_score(metric="l1")``
    vs ``cell_eval.metrics._anndata.discrimination_score``
  * ``energy_distance_blas_f32`` — SCX ``energy_distance(backend="gemm",
    dtype="f32")`` (the default, headline combo) vs cell-eval ``edistance``
  * ``energy_distance_blas_f64`` — SCX ``energy_distance(backend="gemm",
    dtype="f64")`` vs cell-eval ``edistance``
  * ``energy_distance_scalar_f32`` — SCX ``energy_distance(backend="scalar",
    dtype="f32")`` vs cell-eval ``edistance``
  * ``energy_distance`` — alias for the slowest combo
    (``backend="scalar", dtype="f64"``); preserved so historical numbers
    in summaries stay comparable
    (all four skipped at >= 500K cells — the cell-eval reference's O(N^2)
    pairwise distances are infeasible there; SCX's gemm-fused path is
    tracked by the standalone Rust microbench at ``scx-accel/benches/distances.rs``)
  * ``knockdown_efficiency`` — SCX ``knockdown_efficiency`` vs
    arc-bench ``compute_control_baseline`` + ``compute_knockdown_efficiency``
    + ``compute_log_deviation``
  * ``clustering_agreement`` — SCX ``clustering_agreement`` vs
    ``cell_eval.metrics._anndata.ClusteringAgreement`` (skipped at >= 5M)

Only runs for the ``scx_auto`` format variant — the on-disk codec is
irrelevant because this benchmark operates on in-memory AnnData. Matches
the ``correctness`` benchmark's pattern.
"""

from __future__ import annotations

import logging
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult  # noqa: E402

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. The on-disk codec
does not affect this benchmark (operates on in-memory AnnData from the
synthetic generator), so we trigger once per dataset on ``scx_auto`` and
the runtime guard below catches direct invocation."""

# ---------------------------------------------------------------------------
# Skip rules — which operations to run at each n_obs tier
# ---------------------------------------------------------------------------

# Operations skipped when n_obs exceeds the given threshold. Overrides stack:
# a single operation can appear in multiple entries; it is skipped if any
# match. The string after ``:`` is the reason recorded in the result JSON.
# Non-marquee energy_distance variants — diagnostic only. At n_obs >= 100K
# cell-eval's O(N^2) reference is ~15 min/run; gating 3 diagnostic variants
# ~3x that isn't justified. Marquee `energy_distance_blas_f32` keeps running
# at 100K — it is the floored metric in `thresholds.yaml`.
_NON_MARQUEE_EDIST = (
    "energy_distance",
    "energy_distance_blas_f64",
    "energy_distance_scalar_f32",
)

_SKIP_RULES: list[tuple[int, str, str]] = [
    *(
        (100_000, op,
         "non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K")
        for op in _NON_MARQUEE_EDIST
    ),
    # 500K tier: every energy_distance variant (including the marquee
    # blas_f32) is infeasible — cell-eval's reference loop is O(N^2).
    (500_000, "energy_distance",
     "O(N^2) pairwise distance at n_obs >= 500K is infeasible"),
    *(
        (500_000, op,
         "O(N^2) pairwise distance at n_obs >= 500K is infeasible (cell-eval ref)")
        for op in ("energy_distance_blas_f32", *_NON_MARQUEE_EDIST[1:])
    ),
    (5_000_000, "clustering_agreement",
     "stochastic Leiden + kNN on centroid matrix is too slow at n_obs >= 5M"),
]


def _should_skip(op_name: str, n_obs: int) -> str | None:
    for threshold, op, reason in _SKIP_RULES:
        if op == op_name and n_obs >= threshold:
            return reason
    return None


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    """Peak RSS since process start, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _time_once(fn: Callable[[], Any]) -> tuple[float, float]:
    """Run ``fn`` once, return (wall_s, peak_rss_mb).

    ``peak_rss_mb`` is the getrusage peak at end of call — this is monotonic
    per-process, so the delta-from-baseline is computed by the caller.
    """
    t0 = time.perf_counter()
    fn()
    wall = time.perf_counter() - t0
    return wall, _peak_rss_mb()


def _run_timed(fn: Callable[[], Any], n_runs: int) -> dict[str, Any]:
    """Warmup + n_runs timed iterations. Returns ``{wall_s:[...], median, rss}``.

    RSS is reported as the maximum peak-RSS observed across all iterations
    minus the baseline taken immediately before the first warmup run.
    """
    baseline_rss = _peak_rss_mb()
    # Warmup (discarded)
    try:
        fn()
    except Exception:
        # Let the caller capture the error from a real run.
        pass

    walls: list[float] = []
    peak_rsses: list[float] = []
    for _ in range(n_runs):
        wall, peak = _time_once(fn)
        walls.append(wall)
        peak_rsses.append(peak)

    return {
        "wall_s": walls,
        "median_s": statistics.median(walls),
        "peak_rss_mb": max(peak_rsses),
        "rss_delta_mb": max(peak_rsses) - baseline_rss,
    }


# ---------------------------------------------------------------------------
# Per-operation drivers (SCX vs reference)
# ---------------------------------------------------------------------------


def _run_operation(
    name: str,
    scx_fn: Callable[[], Any],
    ref_fn: Callable[[], Any],
    n_runs: int,
) -> dict[str, Any]:
    """Time one (scx, reference) pair. Returns the per-op result dict."""
    rec: dict[str, Any] = {"name": name, "skipped": False}
    try:
        scx = _run_timed(scx_fn, n_runs)
        rec["scx_wall_s"] = scx["wall_s"]
        rec["scx_median_s"] = scx["median_s"]
        rec["scx_peak_rss_mb"] = scx["peak_rss_mb"]
        rec["scx_rss_delta_mb"] = scx["rss_delta_mb"]
    except Exception as e:  # noqa: BLE001 — capture any upstream failure
        rec["scx_error"] = f"{type(e).__name__}: {e}"
        return rec

    try:
        ref = _run_timed(ref_fn, n_runs)
        rec["ref_wall_s"] = ref["wall_s"]
        rec["ref_median_s"] = ref["median_s"]
        rec["ref_peak_rss_mb"] = ref["peak_rss_mb"]
        rec["ref_rss_delta_mb"] = ref["rss_delta_mb"]
    except Exception as e:  # noqa: BLE001
        rec["ref_error"] = f"{type(e).__name__}: {e}"
        return rec

    if rec["scx_median_s"] > 0:
        rec["speedup"] = rec["ref_median_s"] / rec["scx_median_s"]
    return rec


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
    """Run the cell-eval / arc-bench parity performance benchmark.

    Only executes for ``scx_auto`` — the on-disk codec does not affect this
    benchmark (operates on in-memory AnnData from the synthetic generator).

    ``converted_path`` is accepted to match the canonical benchmark contract
    but unused — synthetic datasets are generated in-process via
    ``_pert_synth.make_paired_adata()``.
    """
    if format_variant.key not in SUPPORTED_FORMATS:
        return None

    # pyscx is a hard project dependency — import unconditionally so a
    # missing build fails loudly (developer-environment bug) rather than
    # being swallowed as a soft skip.
    #
    # pyscx.accel is exposed as an attribute by the Rust binding, not a real
    # submodule — ``import pyscx.accel`` fails, but ``import pyscx; pyscx.accel``
    # works. Match the pattern used in pyscx/tests/test_cell_eval_parity.py.
    import pyscx
    acc = pyscx.accel

    # Lazy-import the bench-eval optional deps so ``--list`` and non-parity
    # benchmarks don't need cell-eval / arc-bench / pdex installed. Wrap in
    # try/except so dev machines without the scx-bench-eval conda env skip
    # gracefully (rather than crash) — the gate then reports a missing-
    # metric floor violation, which surfaces the env mistake clearly on
    # Chimera / CI. Mirrors the scoped optional-import pattern in
    # correctness.py for SLAF round-trip checks.
    try:
        from cell_eval import PerturbationAnndataPair
        from cell_eval.metrics._anndata import (
            ClusteringAgreement,
            discrimination_score as ce_discrimination_score,
            edistance as ce_edistance,
            mae as ce_mae,
            mae_delta as ce_mae_delta,
            mse as ce_mse,
            mse_delta as ce_mse_delta,
            pearson_delta as ce_pearson_delta,
        )
        from arc_bench.tools.normalize_transform.core import (
            compute_control_baseline,
            compute_knockdown_efficiency,
            compute_log_deviation,
        )
    except ImportError as e:
        logger.warning(
            "Skipping cell_eval_parity_perf: %s. Required packages "
            "(cell_eval, arc_bench, pdex) live in the scx-bench-eval "
            "conda env — activate it before running this benchmark.",
            e,
        )
        return None

    from benchmarks.comprehensive.benchmarks import _pert_synth

    if not dataset.synthetic:
        logger.warning(
            "cell_eval_parity_perf: dataset '%s' is not marked synthetic; "
            "skipping (the benchmark requires the gene-name-matching "
            "perturbation labels produced by _pert_synth).",
            dataset.name,
        )
        return None

    params = dataset.synth_params
    n_obs = params["n_obs"]
    n_vars = params["n_vars"]
    n_perts = params["n_perts"]
    seed = params["seed"]
    logger.info(
        "cell_eval_parity_perf: generating %s (n_obs=%d, n_vars=%d, n_perts=%d)",
        dataset.name, n_obs, n_vars, n_perts,
    )

    t_gen = time.perf_counter()
    adata_real, adata_pred = _pert_synth.make_paired_adata(
        n_obs=n_obs, n_vars=n_vars, n_perts=n_perts, seed=seed,
    )
    raw_adata = _pert_synth.make_raw_counts(
        n_obs=n_obs, n_vars=n_vars, n_perts=n_perts, seed=seed,
    )
    gen_s = time.perf_counter() - t_gen
    logger.info("  data ready in %.1fs", gen_s)

    # Cell-eval's PerturbationAnndataPair caches pseudobulk matrices on
    # pair.bulk_real / pair.bulk_pred — the first metric call populates them,
    # later calls reuse them. Our SCX counterparts always start cold
    # (perturbation_metrics rebuilds pseudobulk internally on every call;
    # discrimination_score and clustering_agreement do the same).
    #
    # For a fair per-operation comparison we must reset the cache before each
    # ref timing so both sides pay the cold pseudobulk cost. ``_cold_pair`` is
    # a factory that returns a fresh Pair; wrap each ref-side call with it.
    def _cold_pair() -> Any:
        return PerturbationAnndataPair(
            real=adata_real, pred=adata_pred,
            pert_col="perturbation", control_pert="control",
        )

    operations: list[dict[str, Any]] = []

    def _op(name: str, scx: Callable[[], Any], ref: Callable[[], Any]) -> None:
        skip = _should_skip(name, n_obs)
        if skip is not None:
            operations.append(
                {"name": name, "skipped": True, "skipped_reason": skip}
            )
            logger.info("  %s: SKIP (%s)", name, skip)
            return
        logger.info("  %s: running %d repeats", name, n_runs)
        rec = _run_operation(name, scx, ref, n_runs)
        operations.append(rec)
        sp = rec.get("speedup")
        if sp is not None:
            logger.info(
                "    SCX %.3fs  ref %.3fs  speedup %.1fx",
                rec["scx_median_s"], rec["ref_median_s"], sp,
            )

    _op(
        "pseudobulk",
        lambda: acc.pseudobulk_means(adata_real, "perturbation"),
        lambda: PerturbationAnndataPair._bulk_anndata(adata_real, "perturbation"),
    )

    def _ref_bulk() -> None:
        p = _cold_pair()
        ce_pearson_delta(p)
        ce_mse(p)
        ce_mae(p)
        ce_mse_delta(p)
        ce_mae_delta(p)

    _op(
        "bulk_metrics",
        lambda: acc.perturbation_metrics(adata_real, adata_pred),
        _ref_bulk,
    )

    _op(
        "discrimination_l1",
        lambda: acc.discrimination_score(adata_real, adata_pred, metric="l1"),
        lambda: ce_discrimination_score(_cold_pair(), metric="l1"),
    )

    # ── energy_distance: 4 (backend × dtype) combinations ──────────────
    #
    # SCX's energy_distance kernel is generic over `backend ∈ {"scalar",
    # "gemm"}` and `dtype ∈ {"f32", "f64"}`. The cell-eval reference is
    # the same cold ce_edistance call regardless of the SCX combo. Keep the
    # legacy ``energy_distance`` op as an alias for the slowest combo
    # (scalar+f64) so historical numbers in summaries stay comparable;
    # ``energy_distance_blas_f32`` is the new default headline.
    #
    # All four share the cell-eval reference, so a single ref-side cold
    # pair lookup per repeat is enough — but _run_operation re-times the
    # ref alongside each SCX call (matches existing pattern; warmup makes
    # this cheap on the second+ visit because cell-eval caches some
    # internals on the pair).
    def _ref_edistance() -> Any:
        return ce_edistance(_cold_pair())

    _op(
        "energy_distance_blas_f32",
        lambda: acc.energy_distance(
            adata_real, adata_pred, backend="gemm", dtype="f32",
        ),
        _ref_edistance,
    )
    _op(
        "energy_distance_blas_f64",
        lambda: acc.energy_distance(
            adata_real, adata_pred, backend="gemm", dtype="f64",
        ),
        _ref_edistance,
    )
    _op(
        "energy_distance_scalar_f32",
        lambda: acc.energy_distance(
            adata_real, adata_pred, backend="scalar", dtype="f32",
        ),
        _ref_edistance,
    )
    _op(
        "energy_distance",  # back-compat alias for scalar+f64 (slowest)
        lambda: acc.energy_distance(
            adata_real, adata_pred, backend="scalar", dtype="f64",
        ),
        _ref_edistance,
    )

    def _scx_knockdown() -> None:
        # knockdown_efficiency writes columns to adata.obs; use a fresh copy
        # each call so repeats aren't amortized across a populated obs.
        a = raw_adata.copy()
        import scanpy as sc
        sc.pp.normalize_total(a)
        acc.knockdown_efficiency(a, pert_col="perturbation", control="control")

    def _ref_knockdown() -> None:
        import scanpy as sc
        a = raw_adata.copy()
        sc.pp.normalize_total(a)
        baseline = compute_control_baseline(a, "perturbation", "control")
        compute_knockdown_efficiency(a, baseline, "perturbation", "control")
        baseline_log = __import__("numpy").log1p(baseline)
        sc.pp.log1p(a)
        compute_log_deviation(a, baseline_log, "perturbation", "control")

    _op("knockdown_efficiency", _scx_knockdown, _ref_knockdown)

    _op(
        "clustering_agreement",
        lambda: acc.clustering_agreement(
            adata_real, adata_pred,
            pert_col="perturbation", control="control", metric="ami",
        ),
        lambda: ClusteringAgreement(metric="ami")(_cold_pair()),
    )

    # Roll up summary timings so the BenchmarkResult.runs view has a single
    # "wall_s" per run (sum of SCX medians across non-skipped operations).
    # This mirrors how other benchmarks expose a top-level median_wall_s
    # while preserving the per-op detail in metadata.
    scx_medians = [
        op["scx_median_s"] for op in operations
        if not op.get("skipped") and "scx_median_s" in op
    ]
    total_scx_s = sum(scx_medians) if scx_medians else 0.0

    peak_rss = max(
        (op.get("scx_peak_rss_mb", 0.0) for op in operations if not op.get("skipped")),
        default=0.0,
    )

    result = BenchmarkResult(
        benchmark="cell_eval_parity_perf",
        format="scx_auto",
        dataset=dataset.name,
        metadata={
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_perts": n_perts,
            "seed": seed,
            "data_gen_s": round(gen_s, 3),
            "operations": operations,
        },
    )
    # Lift per-operation metrics from metadata.operations[] into runs[].extra
    # so the gate's _load_current_raw_metric (which reads only runs[].extra)
    # can floor them. Sparse keys: present only for non-skipped operations
    # that produced both SCX and reference timings.
    extra: dict[str, float] = {}
    for op in operations:
        if op.get("skipped"):
            continue
        name = op["name"]
        if "scx_median_s" in op:
            extra[f"scx_median_s__{name}"] = op["scx_median_s"]
            extra[f"scx_peak_rss_mb__{name}"] = op["scx_peak_rss_mb"]
        if "ref_median_s" in op:
            extra[f"ref_median_s__{name}"] = op["ref_median_s"]
        if "speedup" in op:
            extra[f"speedup__{name}"] = op["speedup"]
    result.add_run(wall_s=total_scx_s, peak_rss_mb=peak_rss, **extra)
    return result
