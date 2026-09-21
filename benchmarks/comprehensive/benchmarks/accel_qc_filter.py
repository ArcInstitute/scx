"""Fused QC metrics + atomic axis filtering, pyscx vs scanpy.

Scanpy evaluates `calculate_qc_metrics`' cell totals, cell nonzero counts, gene
totals, gene nonzero counts and each `qc_vars` subset separately, so an analyst
asking for two QC subsets pays several full passes over X. Since 0.18
`pyscx.accel.calculate_qc_metrics` runs **one** native kernel on every kind of
`X` — backed, lazy, a backed layer handle, or an in-memory scipy/dense matrix —
collapsing that into one row-axis pass (a per-visible-column u64 bitmask
carrying up to 64 subsets at once) plus one column-axis pass, followed by atomic
`filter_cells` / `filter_genes`. That was profiled in Phase 4.1 but has never
been a first-class comprehensive benchmark, so nothing gates it against scanpy.

The three arms hold the *operation sequence* constant and vary only where `X`
lives, which is the comparison an analyst actually makes: backed SCX, eager SCX
in memory, and scanpy on an eager h5ad.

Two asymmetries have to be closed by hand or the arms are not measuring the same
work, and both default in a direction that would flatter SCX:

* `pyscx.accel.calculate_qc_metrics` defaults `percent_top=None`, while
  `sc.pp.calculate_qc_metrics` defaults to `(50, 100, 200, 500)`. Left alone the
  scanpy arm computes four extra order statistics the SCX arm skips. Both sides
  are passed the same explicit tuple here, clamped to `n_vars` — scanpy's own
  default raises on any file with fewer than 500 genes.
* pyscx defaults `inplace=True`, scanpy defaults `inplace=False`. The scanpy arm
  passes `inplace=True` explicitly, or it would return frames, write nothing,
  and skip the obs/var assignment the SCX arm pays for.

What the `pyscx_inmem` arm is **not**: fully native. `accel.filter_cells` /
`filter_genes` delegate to `sc.pp.*` for an in-memory scipy/dense `X` (the
native path is the backed/lazy one, where they edit a deletion vector and a
column projection instead of rebuilding the matrix). So that arm is a native QC
pass followed by a scanpy filter, and its gap to `scanpy_cpu` measures the fused
QC kernel alone. The backed arm is the one that is native end to end.

Process isolation: every arm, and every run of every arm, executes in a fresh
child that samples its own RSS. `PeakRssSampler` reads `/proc/self/statm`, so a
parent-side sampler around a subprocess measures nothing (0.0 MB observed across
a child that allocated 500 MB), and its `__enter__` seeds from the entry RSS, so
two arms sharing an interpreter make the second inherit the first's retained
heap. See `benchmarks/comprehensive/subproc_arm.py`.

Parity is computed against a scanpy reference produced by a separate **untimed**
child, cached per (dataset, qc_vars, percent_top) so the cost is paid once
rather than per arm. Each triple is its own SLURM job, so the reference cannot
be left to "whichever arm runs first".
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
)
from benchmarks.comprehensive.results import (
    BenchmarkResult,
    require_runs,
    write_missing_result,
)
from benchmarks.comprehensive.rss import PeakRssSampler, current_rss_mb
from benchmarks.comprehensive.subproc_arm import run_arm

logger = logging.getLogger(__name__)

# One variant per input mode. Accel-shaped keys: `run_parallel` pairs an
# `accel_*` benchmark only with formats prefixed `f"{bench_name}__"`, and
# `DatasetConfig.path_for_format` short-circuits them to the source h5ad.
SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "accel_qc_filter__pyscx_cpu",
    "accel_qc_filter__pyscx_inmem",
    "accel_qc_filter__scanpy_cpu",
})

#: Which engine and residency each key selects. `needs_scx` decides whether the
#: arm reads `<name>_auto.scx` or the source h5ad — `dataset.scx_path`
#: (`<name>.scx`) is a path property with no file behind it for any census
#: fixture, which is a trap `export_streaming` is still standing in.
_ARMS: dict[str, dict[str, Any]] = {
    "accel_qc_filter__pyscx_cpu": {"engine": "pyscx", "source": "scx"},
    "accel_qc_filter__pyscx_inmem": {"engine": "pyscx", "source": "h5ad"},
    "accel_qc_filter__scanpy_cpu": {"engine": "scanpy", "source": "h5ad"},
}

#: Passed explicitly to *both* engines; see the module docstring. Clamped per
#: dataset by `percent_top_for`.
PERCENT_TOP: tuple[int, ...] = (50, 100, 200, 500)

MIN_GENES = 200
MIN_CELLS = 3

STAGES: tuple[str, ...] = ("qc_metrics", "filter_cells", "filter_genes")

#: A generous per-arm ceiling. The slowest cell is scanpy on census_1m, whose
#: own `estimate_time_minutes` budget is 45 min + slope; this only has to catch
#: a genuine hang, and a timeout still returns the records already flushed.
_ARM_TIMEOUT_S = 4 * 3600


def accel_qc_filter_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx fused QC + filter (backed SCX)",
            key="accel_qc_filter__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx fused QC + filter (in-memory CSR)",
            key="accel_qc_filter__pyscx_inmem",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="scanpy QC + filter (in-memory h5ad)",
            key="accel_qc_filter__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
    ]


# ---------------------------------------------------------------------------
# QC subset definitions
# ---------------------------------------------------------------------------

#: `qc_vars` names **boolean columns of `adata.var`**, not gene name patterns,
#: on both engines. The masks are derived from `var_names` here so the two
#: engines are handed byte-identical subsets rather than each re-deriving them.
QC_VAR_PATTERNS: dict[str, tuple[str, ...]] = {
    "mt": ("MT-", "mt-", "MT.", "mt."),
    "ribo": ("RPS", "RPL", "Rps", "Rpl"),
}


def qc_var_masks(var_names: Any) -> dict[str, np.ndarray]:
    """Boolean mask per QC subset, keyed as it will appear in `adata.var`.

    A subset that matches nothing is still returned, and its match count is
    recorded in `metadata`: a silently empty `mt` subset makes
    `pct_counts_mt` identically zero on both engines, so the parity check would
    pass while comparing two columns of zeros. The count is what makes that
    visible.
    """
    names = np.asarray([str(v) for v in var_names])
    out: dict[str, np.ndarray] = {}
    for key, prefixes in QC_VAR_PATTERNS.items():
        mask = np.zeros(names.shape[0], dtype=bool)
        for p in prefixes:
            mask |= np.char.startswith(names, p)
        out[key] = mask
    return out


def percent_top_for(n_vars: int) -> tuple[int, ...]:
    """`PERCENT_TOP` clamped to the gene axis.

    Positions are 1-indexed and must lie in `1..=n_vars` on the SCX side;
    scanpy's equivalent raises `IndexError` past the end. pbmc3k-shaped fixtures
    with fewer than 500 genes would otherwise fail the scanpy arm outright,
    which is the exact crash `pyscx`'s `percent_top=None` default exists to
    avoid and which a benchmark must not reintroduce.
    """
    return tuple(p for p in PERCENT_TOP if p <= n_vars) or (min(50, max(1, n_vars)),)


# ---------------------------------------------------------------------------
# The measured work. Imported by the worker script below, so it stays
# in-process testable.
# ---------------------------------------------------------------------------

#: Columns the parity check compares. Integers must agree exactly; the float
#: fraction is compared at 1e-5.
PARITY_INT_COLS = ("total_counts", "n_genes_by_counts")
PARITY_FLOAT_COLS = ("pct_counts_mt", "pct_counts_ribo")


def _open_arm_adata(engine: str, source: str, path: Path) -> Any:
    import anndata

    if source == "scx":
        import pyscx

        return pyscx.open(str(path)).to_anndata(backed=True)
    return anndata.read_h5ad(str(path))


def _attach_qc_vars(adata: Any) -> dict[str, int]:
    masks = qc_var_masks(adata.var_names)
    counts = {}
    for key, mask in masks.items():
        adata.var[key] = mask
        counts[key] = int(mask.sum())
    return counts


def _qc_snapshot(adata: Any) -> dict[str, np.ndarray]:
    """The obs columns the parity check reads, as plain arrays."""
    out: dict[str, np.ndarray] = {}
    for col in (*PARITY_INT_COLS, *PARITY_FLOAT_COLS):
        if col in adata.obs.columns:
            out[col] = np.asarray(adata.obs[col].to_numpy(), dtype=np.float64)
    return out


def run_arm_once(
    engine: str,
    source: str,
    path: Path,
    *,
    dump_npz: Path | None = None,
    timed: bool = True,
) -> dict[str, Any]:
    """One full QC+filter sequence, timed per stage, in *this* process.

    Returns a record dict; `dump_npz` additionally persists the pre-filter QC
    columns and the post-filter shape for the parity comparison, written
    **after** the timed region so serialization never lands in a wall time.
    """
    import time

    adata = _open_arm_adata(engine, source, path)
    qc_var_counts = _attach_qc_vars(adata)
    qc_vars = list(QC_VAR_PATTERNS)
    ptop = percent_top_for(int(adata.n_vars))

    rec: dict[str, Any] = {
        "engine": engine,
        "source": source,
        "qc_var_counts": qc_var_counts,
        "percent_top": list(ptop),
        "n_obs_before": int(adata.n_obs),
        "n_vars_before": int(adata.n_vars),
        "x_type": type(adata.X).__name__,
    }

    def _stage(name: str, fn) -> None:
        with PeakRssSampler() as s:
            t0 = time.perf_counter()
            fn()
            rec[f"qc_wall_s__{name}"] = time.perf_counter() - t0
        rec[f"qc_peak_rss_mb__{name}"] = s.peak_mb

    with PeakRssSampler() as outer:
        t_all = time.perf_counter()
        if engine == "pyscx":
            import pyscx

            _stage("qc_metrics", lambda: pyscx.accel.calculate_qc_metrics(
                adata, qc_vars=qc_vars, percent_top=ptop, inplace=True,
            ))
            snapshot = _qc_snapshot(adata)
            _stage("filter_cells", lambda: pyscx.accel.filter_cells(
                adata, min_genes=MIN_GENES,
            ))
            _stage("filter_genes", lambda: pyscx.accel.filter_genes(
                adata, min_cells=MIN_CELLS,
            ))
        else:
            import scanpy as sc

            _stage("qc_metrics", lambda: sc.pp.calculate_qc_metrics(
                adata, qc_vars=qc_vars, percent_top=ptop, inplace=True,
            ))
            snapshot = _qc_snapshot(adata)
            _stage("filter_cells", lambda: sc.pp.filter_cells(
                adata, min_genes=MIN_GENES,
            ))
            _stage("filter_genes", lambda: sc.pp.filter_genes(
                adata, min_cells=MIN_CELLS,
            ))
        rec["qc_wall_s"] = time.perf_counter() - t_all
    rec["qc_peak_rss_mb"] = outer.peak_mb
    rec["n_obs_after"] = int(adata.n_obs)
    rec["n_vars_after"] = int(adata.n_vars)
    rec["rss_at_exit_mb"] = current_rss_mb()

    if dump_npz is not None:
        dump_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dump_npz,
            shape_after=np.asarray([adata.n_obs, adata.n_vars], dtype=np.int64),
            **snapshot,
        )
    del adata
    return rec


_WORKER_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    engine, source, path, dump = sys.argv[1:5]
    cold = sys.argv[5] == "1"

    from benchmarks.comprehensive.benchmarks.accel_qc_filter import run_arm_once

    if cold:
        from benchmarks.comprehensive.cache_control import drop_file_cache
        policy = drop_file_cache(path)
    else:
        policy = "warm"

    rec = run_arm_once(
        engine, source, Path(path),
        dump_npz=Path(dump) if dump else None,
    )
    rec["cache_policy"] = policy
    print(json.dumps(rec), flush=True)
""")


_REFERENCE_SCRIPT = textwrap.dedent("""\
    import json, sys
    from pathlib import Path

    h5ad_path, dump = sys.argv[1:3]

    from benchmarks.comprehensive.benchmarks.accel_qc_filter import run_arm_once

    rec = run_arm_once("scanpy", "h5ad", Path(h5ad_path), dump_npz=Path(dump))
    print(json.dumps({"reference_wall_s": rec["qc_wall_s"],
                      "qc_var_counts": rec["qc_var_counts"],
                      "percent_top": rec["percent_top"]}), flush=True)
""")


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def _reference_dir() -> Path:
    base = Path(os.environ.get("SCX_BENCH_TMPDIR") or os.environ.get("SCX_WORK_DIR", ""))
    root = base if base.is_dir() else Path("/tmp")
    return root / "scx_bench_qc_reference"


def _reference_tag(dataset: DatasetConfig) -> str:
    """Identity of the reference: dataset plus everything that changes its value.

    A cached reference computed under a different `percent_top` or a different
    subset definition is not a reference for this run, and reusing one silently
    would make the parity number describe a comparison nobody asked for.
    """
    payload = json.dumps(
        {
            "dataset": dataset.name,
            "percent_top": list(PERCENT_TOP),
            "patterns": {k: list(v) for k, v in sorted(QC_VAR_PATTERNS.items())},
            "min_genes": MIN_GENES,
            "min_cells": MIN_CELLS,
        },
        sort_keys=True,
    )
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def ensure_reference(dataset: DatasetConfig) -> Path | None:
    """Compute (or reuse) the scanpy reference for *dataset*, untimed.

    Returns the `.npz` path, or `None` when the source h5ad is absent or the
    reference child failed — parity is then simply not emitted, which the gate
    reads as a missing metric on a result that exists, i.e. a violation rather
    than a silent pass.
    """
    h5ad = dataset.h5ad_path
    if not h5ad.exists():
        logger.warning("accel_qc_filter: no source h5ad at %s; skipping parity", h5ad)
        return None
    out = _reference_dir() / f"{dataset.name}.{_reference_tag(dataset)}.npz"
    if out.exists():
        logger.info("accel_qc_filter: reusing scanpy reference %s", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("accel_qc_filter: building scanpy reference for %s (untimed)", dataset.name)
    outcome = run_arm(
        _REFERENCE_SCRIPT, [str(h5ad), str(out)],
        timeout_s=_ARM_TIMEOUT_S, label="accel_qc_filter reference",
    )
    if not outcome.ok or not out.exists():
        logger.warning(
            "accel_qc_filter: reference build failed, parity will be omitted.\n%s",
            outcome.failure_text("accel_qc_filter reference"),
        )
        return None
    return out


def parity_metrics(arm_npz: Path, ref_npz: Path) -> dict[str, float]:
    """Compare one arm's QC columns and final shape against the scanpy reference.

    `pct_counts_*` is where the two engines legitimately disagree on empty
    cells: pyscx reports `0.0`, scanpy reports `NaN`. Those positions are masked
    out rather than folded in.

    The fold is `np.max` over a collected list, deliberately **not** a running
    `max(worst, x)`. Python's `max` returns its first argument when the
    comparison is False, so `max(0.0, nan)` is `0.0` — a NaN that slipped past
    the mask would be silently swallowed and reported as perfect agreement,
    which is the failure this function exists to make impossible. numpy's `max`
    propagates, and a non-finite fold is converted to `inf` so it reads as a
    loud violation rather than a missing metric.
    """
    ours = np.load(arm_npz)
    ref = np.load(ref_npz)
    out: dict[str, float] = {}

    diffs: list[float] = [0.0]
    for col in PARITY_INT_COLS:
        if col not in ours or col not in ref:
            continue
        a, b = ours[col], ref[col]
        if a.shape != b.shape:
            return {"qc_metrics_max_abs_diff": float("inf"),
                    "filtered_shape_match_int": 0.0}
        if a.size:
            diffs.append(float(np.max(np.abs(a - b))))
    for col in PARITY_FLOAT_COLS:
        if col not in ours or col not in ref:
            continue
        a, b = ours[col], ref[col]
        if a.shape != b.shape:
            return {"qc_metrics_max_abs_diff": float("inf"),
                    "filtered_shape_match_int": 0.0}
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            diffs.append(float(np.max(np.abs(a[both] - b[both]))))
    worst = float(np.max(diffs))
    if not np.isfinite(worst):
        logger.warning(
            "accel_qc_filter: non-finite parity fold for %s — reporting inf so "
            "the gate sees a violation rather than a missing metric", arm_npz,
        )
        worst = float("inf")
    out["qc_metrics_max_abs_diff"] = worst

    shape_ok = bool(np.array_equal(ours["shape_after"], ref["shape_after"]))
    out["filtered_shape_match_int"] = 1.0 if shape_ok else 0.0
    out["n_obs_after"] = float(ours["shape_after"][0])
    out["n_vars_after"] = float(ours["shape_after"][1])
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    key = format_variant.key
    arm = _ARMS.get(key)
    if arm is None:
        logger.warning("accel_qc_filter: unknown variant %s — skipping", key)
        return None

    if arm["source"] == "scx":
        source_path = dataset.scx_auto_path
        missing_hint = "run benchmarks/scripts/reconvert_fixtures.py for scx_auto"
    else:
        # `converted_path` for an arm-shaped key is the source h5ad itself
        # (path_for_format short-circuits), never an .scx — but fall back to
        # the property so a direct invocation works.
        source_path = Path(converted_path) if converted_path else dataset.h5ad_path
        if source_path.suffix != ".h5ad":
            source_path = dataset.h5ad_path
        missing_hint = "stage the source h5ad first"

    if not source_path.exists():
        logger.warning("accel_qc_filter: %s missing for %s", source_path, key)
        write_missing_result(
            benchmark="accel_qc_filter", format_key=key, dataset=dataset.name,
            missing_reason="fixture_missing",
            notes=f"{source_path} does not exist; {missing_hint}",
        )
        return None

    ref_npz = ensure_reference(dataset) if arm["engine"] == "pyscx" else None

    result = BenchmarkResult(
        benchmark="accel_qc_filter",
        format=key,
        dataset=dataset.name,
        scenario={
            "name": "qc_metrics_then_filter",
            "engine": arm["engine"],
            "residency": "backed" if arm["source"] == "scx" else "in_memory",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
        },
        comparison={
            "subject": {"impl": key},
            "baseline": {"impl": "accel_qc_filter__scanpy_cpu"},
            "metric": "qc_wall_s",
            "status": "pending",
        },
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "source_path": str(source_path),
            "min_genes": MIN_GENES,
            "min_cells": MIN_CELLS,
            "percent_top": list(percent_top_for(dataset.n_vars)),
            "stages": list(STAGES),
            # Recorded so a reader can tell "parity omitted because the
            # reference failed" from "parity omitted because this is the
            # reference engine".
            "parity_reference": str(ref_npz) if ref_npz else None,
        },
    )

    dump_dir = _reference_dir() / "arms"
    dump_path = dump_dir / f"{key}__{dataset.name}.npz"

    # One warm-up child purely to populate the page cache; discarded. Skipped
    # under --cold-cache, where a warm page cache is exactly what the run is
    # trying not to have.
    if not cold_cache:
        for i in range(N_WARMUP_RUNS):
            logger.info("accel_qc_filter warm-up %d/%d for %s", i + 1, N_WARMUP_RUNS, key)
            run_arm(
                _WORKER_SCRIPT,
                [arm["engine"], arm["source"], str(source_path), "", "0"],
                timeout_s=_ARM_TIMEOUT_S, label=f"{key} warmup",
            )

    for i in range(n_runs):
        # A fresh child per *run*, not just per arm. Runs 2..n in one
        # interpreter would each seed their sampler with the previous run's
        # retained heap — glibc does not return large freed arenas — so the
        # median peak would drift upward with n_runs for reasons that have
        # nothing to do with the code under test.
        outcome = run_arm(
            _WORKER_SCRIPT,
            [
                arm["engine"], arm["source"], str(source_path),
                str(dump_path) if ref_npz is not None else "",
                "1" if cold_cache else "0",
            ],
            timeout_s=_ARM_TIMEOUT_S, label=f"{key} run {i + 1}",
        )
        if not outcome.ok or not outcome.records:
            raise RuntimeError(outcome.failure_text(f"accel_qc_filter {key} run {i + 1}"))
        rec = outcome.records[-1]

        extras: dict[str, Any] = {
            k: v for k, v in rec.items()
            if k.startswith("qc_wall_s") or k.startswith("qc_peak_rss_mb")
        }
        extras["cache_policy"] = rec.get("cache_policy", "warm")
        extras["x_type"] = rec.get("x_type")

        if ref_npz is not None and dump_path.exists():
            try:
                extras.update(parity_metrics(dump_path, ref_npz))
            except Exception as exc:  # noqa: BLE001
                logger.warning("accel_qc_filter: parity failed for %s: %s", key, exc)

        result.add_run(
            wall_s=float(rec["qc_wall_s"]),
            peak_rss_mb=float(rec["qc_peak_rss_mb"]),
            **extras,
        )
        logger.info(
            "  %s run %d: wall=%.3fs peak=%.1fMB qc=%.3fs cells=%.3fs genes=%.3fs "
            "shape %sx%s -> %sx%s",
            key, i + 1, rec["qc_wall_s"], rec["qc_peak_rss_mb"],
            rec.get("qc_wall_s__qc_metrics", float("nan")),
            rec.get("qc_wall_s__filter_cells", float("nan")),
            rec.get("qc_wall_s__filter_genes", float("nan")),
            rec.get("n_obs_before"), rec.get("n_vars_before"),
            rec.get("n_obs_after"), rec.get("n_vars_after"),
        )
        if i == 0:
            result.metadata["qc_var_counts"] = rec.get("qc_var_counts")
            result.metadata["percent_top"] = rec.get("percent_top")

    require_runs(result, str(source_path))

    parity = [r.extra.get("qc_metrics_max_abs_diff") for r in result.runs]
    parity = [p for p in parity if p is not None]
    if parity:
        result.metadata["qc_metrics_max_abs_diff"] = float(np.median(parity))
        result.overall_passed = bool(
            result.metadata["qc_metrics_max_abs_diff"] <= 1e-5
            and all(r.extra.get("filtered_shape_match_int") == 1.0 for r in result.runs)
        )

    logger.info(
        "accel_qc_filter complete: %s / %s — median %.3fs, parity=%s",
        key, dataset.name, result.median_wall_s or 0.0,
        result.metadata.get("qc_metrics_max_abs_diff", "n/a"),
    )
    return result
