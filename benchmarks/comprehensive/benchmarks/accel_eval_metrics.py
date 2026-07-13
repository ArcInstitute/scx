"""GPU perturbation-evaluation metric accelerator benchmark (route gate).

Exercises the two GPU-accelerated cell-eval effect metrics CPU-vs-GPU on the
synthetic paired real/pred fixture from ``_pert_synth.make_paired_adata``:

  * ``perturbation_metrics`` — GPU runs the per-group pseudobulk means on the
    device (f64), the five bulk metrics on the host. Route ``gpu_dense``.
  * ``energy_distance`` (euclidean) — GPU runs a gemm-based pairwise-distance
    mean (``‖x−y‖²=‖x‖²+‖y‖²−2·xyᵀ``) at f32 with f64 reductions. Route
    ``gpu_dense``.

This is the route-correctness gate companion to ``cell_eval_parity_perf`` (a
CPU-only wall-clock parity bench that calls these same metrics without
``device=`` and reads no route). ``accel_de_nb_glm.py`` is the structural
template.

## Variants

| ``format_variant.key``                               | Op                     | Device |
|------------------------------------------------------|------------------------|--------|
| ``accel_eval_metrics__pyscx_perturbation_metrics_cpu``| ``perturbation_metrics``| CPU   |
| ``accel_eval_metrics__pyscx_perturbation_metrics_gpu``| ``perturbation_metrics``| GPU   |
| ``accel_eval_metrics__pyscx_energy_distance_cpu``     | ``energy_distance``    | CPU    |
| ``accel_eval_metrics__pyscx_energy_distance_gpu``     | ``energy_distance``    | GPU    |

The GPU variant additionally runs the CPU path (for CPU↔GPU concordance +
speedup).

## Signals (emitted into ``runs[].extra``)

Hard-gated (deterministic; see ``thresholds.yaml``):
  * ``perturbation_metrics_route_gpu_correct`` / ``energy_distance_route_gpu_correct``
    — 1.0 iff a ``gpu_*`` route ran (0.0 on a silent GPU→CPU drop). The GPU
    variant is skipped on non-GPU hosts, so a recorded route always means GPU
    was attempted.

Surfaced (node-dependent — tracked vs baseline, not absolute-floored):
  * ``<op>_cpu_gpu_concordant`` — 1.0 iff GPU matches CPU within the documented
    tolerance (perturbation_metrics ``atol=1e-5``; energy_distance ``atol=1e-4``,
    it is a correlation).
  * ``<op>_gpu_speedup`` — CPU/GPU median wall ratio (same machine).

Plus diagnostics ``gpu_dispatch_route`` / ``gpu_dispatch_fallback``.

## Skip semantics

On a non-GPU host the GPU variant returns ``None`` (no result JSON written) —
``check_absolute_floors`` keys floors on whether the raw JSON exists, so this is
a "scoped-out triple" (silently skipped), NOT a missing-metric violation.

Only runs on synthetic paired datasets (``dataset.synthetic``); exercise it via
``run_parallel.py --benchmarks accel_eval_metrics --datasets pert_synth_10k``.
"""

from __future__ import annotations

import logging
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult

logger = logging.getLogger(__name__)

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


def accel_eval_metrics_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx perturbation_metrics (CPU)",
            key="accel_eval_metrics__pyscx_perturbation_metrics_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx perturbation_metrics (GPU)",
            key="accel_eval_metrics__pyscx_perturbation_metrics_gpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx energy_distance (CPU)",
            key="accel_eval_metrics__pyscx_energy_distance_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx energy_distance (GPU)",
            key="accel_eval_metrics__pyscx_energy_distance_gpu",
            category="accel", runner="accel_runner",
        ),
    ]


# variant key -> (op, requires_gpu).
_VARIANTS: dict[str, tuple[str, bool]] = {
    "accel_eval_metrics__pyscx_perturbation_metrics_cpu": ("perturbation_metrics", False),
    "accel_eval_metrics__pyscx_perturbation_metrics_gpu": ("perturbation_metrics", True),
    "accel_eval_metrics__pyscx_energy_distance_cpu": ("energy_distance", False),
    "accel_eval_metrics__pyscx_energy_distance_gpu": ("energy_distance", True),
}

# CPU↔GPU concordance tolerance per op. energy_distance is a Pearson
# correlation of f32-gemm distances → 1e-4; perturbation_metrics accumulates
# pseudobulk in f64 → 1e-5.
_CONCORDANCE_ATOL: dict[str, float] = {
    "perturbation_metrics": 1e-5,
    "energy_distance": 1e-4,
}


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _median_wall(fn: Callable[[], Any], reps: int) -> tuple[float, Any]:
    times: list[float] = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), out


def _route(adata: Any, op: str) -> str | None:
    try:
        return adata.uns["scx_accel"][op]["route"]
    except Exception:
        return None


def _fallback(adata: Any, op: str) -> str | None:
    try:
        return adata.uns["scx_accel"][op]["fallback_reason"]
    except Exception:
        return None


def _call(op: str, real: Any, pred: Any, device: str) -> Any:
    import pyscx
    if op == "perturbation_metrics":
        return pyscx.accel.perturbation_metrics(real, pred, device=device)
    # energy_distance: default euclidean (the gemm GPU path).
    return pyscx.accel.energy_distance(real, pred, metric="euclidean", device=device)


def _max_abs_diff(op: str, cpu_out: Any, gpu_out: Any) -> float | None:
    """Max absolute CPU↔GPU difference, NaN-aware; None if degenerate."""
    try:
        if op == "energy_distance":
            c, g = float(cpu_out), float(gpu_out)
            if np.isnan(c) and np.isnan(g):
                return 0.0
            return abs(c - g)
        # perturbation_metrics: dict[str, dict[str, float]].
        worst = 0.0
        for metric, per_pert in cpu_out.items():
            for pert, cval in per_pert.items():
                gval = gpu_out[metric][pert]
                if np.isnan(cval) and np.isnan(gval):
                    continue
                worst = max(worst, abs(float(cval) - float(gval)))
        return worst
    except Exception as e:  # noqa: BLE001
        logger.warning("accel_eval_metrics: CPU↔GPU diff failed: %s", e)
        return None


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
    if key not in _VARIANTS:
        return None
    op, requires_gpu = _VARIANTS[key]

    if not _HAS_PYSCX:
        return None
    if requires_gpu and not _HAS_PYSCX_GPU:
        # No JSON written → scoped-out triple, not a missing-metric violation.
        return None
    if not dataset.synthetic:
        logger.warning(
            "accel_eval_metrics: dataset '%s' is not synthetic; skipping "
            "(needs the paired real/pred fixture from _pert_synth).",
            dataset.name,
        )
        return None

    import pyscx
    from benchmarks.comprehensive.benchmarks import _pert_synth

    reps = max(1, min(n_runs, 3))
    params = dict(dataset.synth_params)
    params.pop("density", None)  # make_paired_adata doesn't take density
    t_gen = time.perf_counter()
    real, pred = _pert_synth.make_paired_adata(**params)
    gen_s = time.perf_counter() - t_gen
    logger.info(
        "accel_eval_metrics: fixture %s ready in %.1fs (n_obs=%d, n_vars=%d) op=%s",
        dataset.name, gen_s, int(real.n_obs), int(real.n_vars), op,
    )

    result = BenchmarkResult(
        benchmark="accel_eval_metrics",
        format=key,
        dataset=dataset.name,
        metadata={
            "n_obs": int(real.n_obs),
            "n_vars": int(real.n_vars),
            "op": op,
            "synth_params": params,
            "reps": reps,
            "device": "gpu" if requires_gpu else "cpu",
            "data_gen_s": round(gen_s, 3),
        },
    )

    baseline_rss = _peak_rss_mb()

    if not requires_gpu:
        # ── CPU baseline ────────────────────────────────────────────────
        _call(op, real, pred, "cpu")  # warmup (discarded)
        cpu_s, _ = _median_wall(lambda: _call(op, real, pred, "cpu"), reps)
        route = _route(pred, op)
        extra: dict[str, Any] = {}
        if route is not None:
            extra["gpu_dispatch_route"] = route
            result.metadata["route"] = route
        result.add_run(
            wall_s=cpu_s,
            peak_rss_mb=max(baseline_rss, _peak_rss_mb()),
            **extra,
        )
        return result

    # ── GPU variant: route gate + CPU↔GPU concordance + speedup ─────────
    _call(op, real, pred, "gpu")  # warmup (discarded)
    gpu_s, gpu_out = _median_wall(lambda: _call(op, real, pred, "gpu"), reps)
    # Read the route immediately after the GPU runs — the CPU runs below
    # overwrite pred.uns["scx_accel"][op]["route"].
    route = _route(pred, op) or "<unknown>"
    fallback = _fallback(pred, op)

    cpu_s, cpu_out = _median_wall(lambda: _call(op, real, pred, "cpu"), reps)

    extra = {
        f"{op}_route_gpu_correct": 1.0 if route.startswith("gpu") else 0.0,
        "gpu_dispatch_route": route,
    }
    if fallback is not None:
        extra["gpu_dispatch_fallback"] = fallback

    max_diff = _max_abs_diff(op, cpu_out, gpu_out)
    atol = _CONCORDANCE_ATOL[op]
    if max_diff is not None:
        extra[f"{op}_cpu_gpu_max_abs_diff"] = max_diff
        extra[f"{op}_cpu_gpu_concordant"] = 1.0 if max_diff <= atol else 0.0

    extra[f"{op}_gpu_wall_s"] = gpu_s
    extra[f"{op}_cpu_wall_s"] = cpu_s
    extra[f"{op}_gpu_speedup"] = (cpu_s / gpu_s) if gpu_s > 0 else None

    result.metadata["route"] = route
    result.metadata["gpu_dispatch_route"] = route
    result.add_run(
        wall_s=gpu_s,
        peak_rss_mb=max(baseline_rss, _peak_rss_mb()),
        **extra,
    )
    return result
