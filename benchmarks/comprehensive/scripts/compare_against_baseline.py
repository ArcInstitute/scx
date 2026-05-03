#!/usr/bin/env python3
"""
Compare a post-change comprehensive benchmark run against the pre-change baseline.

This is the regression gate. Each PR's acceptance criteria include:
all previously-stable benchmarks still within 3% of baseline, and all
parity tests pass with unchanged tolerance.

Given two directories produced by capture_baseline.py, this script reports:

  - Timing deltas per (benchmark, format, dataset).  Flags anything outside
    the per-row *effective* tolerance as a regression. The effective tolerance
    is ``max(--timing-tolerance, --iqr-k * baseline_iqr / baseline_median)``
    so a noisy baseline (high run-to-run dispersion) gets a relaxed bound
    proportional to its own measured noise. With the default
    ``--timing-tolerance=0.03`` and ``--iqr-k=1.5``, a row whose baseline
    has IQR/median ≥ 0.02 (i.e. RSD-equivalent) automatically widens; a
    rock-stable row stays at the 3% floor. Falls back to the fixed
    --timing-tolerance against older baselines that lack ``wall_s_iqr``.
  - Peak RSS deltas.  Flagged at --rss-tolerance (default 10%).
  - File size deltas.  Flagged at --size-tolerance (default 1%) — size should
    be byte-stable across non-format-changing PRs.
  - Accelerator fingerprint diffs (exact hash comparison).  Any mismatch is
    a hard fail by default; pass --allow-fingerprint-drift to downgrade to
    a warning (e.g. for H10 or M13 fixes that intentionally change RNG output).

Exit codes:
  0 — no regressions at the current tolerances.
  1 — at least one regression or fingerprint mismatch.
  2 — baseline or current directory missing / malformed.

Usage
-----
    python benchmarks/comprehensive/scripts/compare_against_baseline.py \
        --baseline benchmarks/comprehensive/results/baseline_2026_04_17 \
        --current  benchmarks/comprehensive/results/candidate_2026_05_01

    # Weaken the timing gate on noisy shared nodes:
    compare_against_baseline.py ... --timing-tolerance 0.05

    # After an H10 PR, we expect fingerprints to change; just warn:
    compare_against_baseline.py ... --allow-fingerprint-drift

Report lands on stdout as markdown so PR bots can paste it directly, and as
JSON at --report-json for programmatic consumption.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# _justifications and _flakiness live beside this script — add parent
# to sys.path so `python scripts/compare_against_baseline.py` works
# without a package install. When invoked as `-m`, relative import works;
# otherwise fall back to path-based import.
try:
    from . import _justifications
    from . import _flakiness
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _justifications  # type: ignore[no-redef]
    import _flakiness  # type: ignore[no-redef]


# Default locations used by the on-demand gate workflow. All overridable
# via CLI flags; surfaced as module-level constants so the one-shot
# ``gate_candidate.py`` wrapper and the tests can reference them.
_COMPREHENSIVE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_BASELINES_DIR = _COMPREHENSIVE_DIR / "results" / "baselines"
_DEFAULT_LATEST_LINK = _DEFAULT_BASELINES_DIR / "LATEST"
_DEFAULT_JUSTIFICATIONS_DIR = _COMPREHENSIVE_DIR / "results" / "justifications"
_DEFAULT_FLAKINESS_DIR = _COMPREHENSIVE_DIR / "results" / "flakiness"
_DEFAULT_THRESHOLDS_YAML = _COMPREHENSIVE_DIR / "thresholds.yaml"

# Wall-time CV threshold above which a row is surfaced in the gate's
# "High-variance rows" informational section. Picked so a row whose
# observed RSD is comparable to the default 3% timing tolerance is
# flagged as a likely flakiness-ledger candidate.
_HIGH_CV_THRESHOLD = 0.05


def _resolve_default_baseline() -> Path | None:
    """Return the path ``--baseline`` defaults to, or None if absent.

    Prefers the ``baselines/LATEST`` symlink (or pointer file) written by
    ``promote_baseline.py``. Falls back to the single child directory of
    ``baselines/`` when exactly one exists (common early-adoption case).
    Returns None when the baseline tree is empty — the caller then errors
    with a clear bootstrap hint.
    """
    link = _DEFAULT_LATEST_LINK
    if link.is_symlink():
        return link.resolve()
    if link.is_file():
        # Pointer-file fallback (filesystems that reject symlinks).
        label = link.read_text().strip()
        candidate = _DEFAULT_BASELINES_DIR / label
        if candidate.is_dir():
            return candidate
    # No LATEST — if exactly one baseline exists, use it.
    if _DEFAULT_BASELINES_DIR.is_dir():
        children = sorted(
            p for p in _DEFAULT_BASELINES_DIR.iterdir()
            if p.is_dir() and p.name != "LATEST"
        )
        if len(children) == 1:
            return children[0]
    return None


@dataclass
class Delta:
    benchmark: str
    fmt: str
    dataset: str
    metric: str
    baseline: float | None
    current: float | None
    relative_change: float | None  # (current - baseline) / baseline
    is_regression: bool
    # Tolerance actually applied to this row — global default, possibly
    # widened by IQR-based noise compensation, possibly further raised by
    # a flakiness override.
    effective_tolerance: float | None = None
    # True iff a flakiness override raised this row's tolerance above the
    # IQR-widened bound; informational, used for status labelling.
    tolerance_relaxed: bool = False
    # True iff the IQR-based noise compensation widened the per-row
    # tolerance above the global ``--timing-tolerance`` default. Set only
    # for ``median_wall_s`` rows when the baseline carries ``wall_s_iqr``.
    noise_widened: bool = False
    # Baseline IQR / baseline median, the dimensionless dispersion the
    # gate uses to scale the noise band. ``None`` for non-timing metrics
    # and for baselines missing ``wall_s_iqr``.
    iqr_ratio: float | None = None
    # Coefficient of variation (stdev/mean) of ``runs[].wall_s`` from
    # the candidate raw JSON. Populated only for ``median_wall_s`` rows;
    # ``None`` everywhere else (including when n<2 or mean=0).
    wall_cv: float | None = None


def _load_summary(path: Path) -> dict[str, Any]:
    f = path / "summary.json"
    if not f.exists():
        raise FileNotFoundError(f"summary.json not found at {f}")
    data = json.loads(f.read_text())
    # capture_baseline.py stamps summary.json with schema_version;
    # legacy promoted baselines that predate the stamping omit it and
    # the check is silently skipped (back-compat). When the field is
    # present, refuse to diff against an unknown future version.
    sv = data.get("schema_version")
    if sv is not None:
        try:
            from benchmarks.comprehensive.results import SCHEMA_VERSION
        except ImportError:
            SCHEMA_VERSION = None  # pragma: no cover — impossible in practice
        if SCHEMA_VERSION is not None and sv > SCHEMA_VERSION:
            raise ValueError(
                f"{f} declares schema_version={sv} but this gate only "
                f"understands up to {SCHEMA_VERSION}. Upgrade the harness "
                f"or downgrade the snapshot."
            )
    return data


def _load_fingerprints(path: Path) -> dict[str, Any]:
    f = path / "fingerprints" / "fingerprints.json"
    if not f.exists():
        return {}
    return json.loads(f.read_text())


def _relative(base: float | None, cur: float | None) -> float | None:
    if base is None or cur is None:
        return None
    if base == 0:
        return None
    return (cur - base) / base


def diff_summaries(
    baseline_summary: dict[str, Any],
    current_summary: dict[str, Any],
    timing_tol: float,
    rss_tol: float,
    size_tol: float,
    *,
    gate: bool = False,
    tolerance_overrides: dict[_flakiness.Quad, float] | None = None,
    wall_cvs: dict[tuple[str, str, str], float | None] | None = None,
    iqr_k: float = 1.5,
) -> list[Delta]:
    """One Delta row per (benchmark, format, dataset, metric) tuple.

    When ``gate`` is False (default), behavior is unchanged: a row with
    baseline=None or current=None produces ``relative_change=None`` and
    ``is_regression=False``. When ``gate`` is True, a row present in
    baseline but missing from current (``current=None``) is flagged as a
    regression — a disappeared benchmark is a silent gap the gate must
    surface. Appearing benchmarks (baseline=None, current!=None) stay
    non-regressions; ``render_markdown`` logs them informationally.

    ``tolerance_overrides`` maps ``(benchmark, format, dataset, metric)``
    to a relaxed bound. When a row matches an override the per-row
    tolerance is the override's value instead of the global default;
    the row's ``effective_tolerance`` and ``tolerance_relaxed`` fields
    record what was actually applied. Overrides are loaded by the
    flakiness-ledger module from ``results/flakiness/`` markdown files.

    ``wall_cvs`` maps ``(benchmark, format, dataset)`` to the
    coefficient of variation of ``runs[].wall_s`` in the candidate
    snapshot. Stamped onto the matching ``median_wall_s`` deltas for
    reporting; never affects regression status.

    ``iqr_k`` scales the baseline-IQR contribution to the noise-widened
    timing tolerance: a ``median_wall_s`` row is gated at
    ``max(timing_tol, iqr_k * baseline_iqr / baseline_median)``. Falls
    back to ``timing_tol`` when the baseline lacks ``wall_s_iqr``
    (older captures); a single WARN is logged per gate run noting the
    missing field. Only affects ``median_wall_s``.
    """
    overrides = tolerance_overrides or {}
    cvs = wall_cvs or {}
    deltas: list[Delta] = []
    base_rows = baseline_summary.get("rows", {})
    cur_rows = current_summary.get("rows", {})
    all_keys = sorted(set(base_rows) | set(cur_rows))

    # Surface "baseline lacks variance metadata" exactly once, even if
    # hundreds of rows are missing the field. Spamming per row just makes
    # the gate output unreadable.
    iqr_missing_warned = False

    for key in all_keys:
        try:
            bench, fmt, dataset = key.split("__", 2)
        except ValueError:
            # Key doesn't follow the expected schema.  Skip silently; summary
            # regeneration will have already warned about malformed entries.
            continue

        base = base_rows.get(key, {})
        cur = cur_rows.get(key, {})

        for metric, default_tol in [
            ("median_wall_s",      timing_tol),
            ("peak_rss_mb_median", rss_tol),
            ("file_size_bytes",    size_tol),
        ]:
            b, c = base.get(metric), cur.get(metric)
            rel = _relative(b, c)
            quad = (bench, fmt, dataset, metric)

            # IQR-based noise widening for wall-time only. Other metrics
            # have their own variance treatment (RSS is a single point
            # sample today; file size is deterministic).
            iqr_ratio: float | None = None
            noise_widened = False
            row_default_tol = default_tol
            if metric == "median_wall_s" and b is not None and b > 0:
                b_iqr = base.get("wall_s_iqr")
                if b_iqr is None:
                    if not iqr_missing_warned:
                        logger.warning(
                            "baseline summary.json has no wall_s_iqr — "
                            "falling back to fixed --timing-tolerance for "
                            "all median_wall_s rows. Re-capture the baseline "
                            "with a capture_baseline.py that records "
                            "wall_s_iqr to enable variance-aware gating."
                        )
                        iqr_missing_warned = True
                else:
                    iqr_ratio = float(b_iqr) / float(b)
                    iqr_widened_tol = iqr_k * iqr_ratio
                    if iqr_widened_tol > row_default_tol:
                        row_default_tol = iqr_widened_tol
                        noise_widened = True

            override_tol = overrides.get(quad)
            # Overrides only relax (the loader rejects negative values);
            # if an entry is somehow at or below the row's default
            # (post-IQR-widening) we let the default win so a stale
            # ledger entry can't tighten the gate.
            relaxed = override_tol is not None and override_tol > row_default_tol
            tol = override_tol if relaxed else row_default_tol
            if override_tol is not None and not relaxed:
                # Surface silent no-ops: the operator committed an
                # override but it's at or below the row's effective
                # default (which itself may have been widened by IQR),
                # so the override has no effect. Without this warning
                # the entry is dead weight that quietly accumulates
                # in the ledger.
                logger.warning(
                    "Flakiness override for %s (tolerance=%.4f) is at or "
                    "below the row's effective tolerance (%.4f); override "
                    "has no effect — overrides only relax. Either raise the "
                    "tolerance or delete the entry.",
                    quad, override_tol, row_default_tol,
                )
            regressed = rel is not None and rel > tol
            if gate and b is not None and c is None:
                # Disappeared benchmark — only surface this once per triple
                # so the report doesn't double-count each metric. Use the
                # timing row as the canonical signal.
                regressed = regressed or (metric == "median_wall_s")
            wall_cv = cvs.get((bench, fmt, dataset)) if metric == "median_wall_s" else None
            deltas.append(Delta(
                benchmark=bench, fmt=fmt, dataset=dataset, metric=metric,
                baseline=b, current=c, relative_change=rel,
                is_regression=regressed,
                effective_tolerance=tol,
                tolerance_relaxed=relaxed,
                noise_widened=noise_widened,
                iqr_ratio=iqr_ratio,
                wall_cv=wall_cv,
            ))

    return deltas


def diff_fingerprints(
    baseline_fp: dict[str, Any],
    current_fp: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Returns (mismatches, missing).

    mismatches = entries whose hash differs.
    missing    = entries present in baseline but absent / failed in current.
    """
    mismatches: list[str] = []
    missing: list[str] = []
    base_entries = baseline_fp.get("fingerprints", {})
    cur_entries = current_fp.get("fingerprints", {})

    for name, ref in base_entries.items():
        if name not in cur_entries:
            missing.append(name)
            continue
        cur = cur_entries[name]
        if "hash" not in ref or "hash" not in cur:
            missing.append(name)
            continue
        if ref["hash"] != cur["hash"]:
            mismatches.append(
                f"{name}: baseline={ref['hash'][:16]}..., "
                f"current={cur['hash'][:16]}... "
                f"(dtype={ref.get('dtype')}, shape={ref.get('shape')})"
            )

    return mismatches, missing


# ---------------------------------------------------------------------------
# Absolute-floor thresholds (Phase G.2)
# ---------------------------------------------------------------------------


@dataclass
class FloorViolation:
    benchmark: str
    fmt: str
    dataset: str
    metric: str
    threshold: float
    observed: float | None
    direction: str = "min"  # "min" => observed must be >= threshold; "max" => <=


_VALID_DIRECTIONS = ("min", "max")
_FLOOR_REQUIRED_FIELDS = ("benchmark", "format", "dataset", "metric")


def _load_thresholds_yaml(path: Path) -> list[dict[str, Any]]:
    """Load ``thresholds.yaml`` and return a list of validated floor dicts.

    Expected shape:

        absolute_floors:
          - benchmark: cloud_push
            format: scx_auto
            dataset: tabula_sapiens_100k
            metric: throughput_mbps
            min: 50.0
            # direction: min  # default; use "max" for lower-is-better metrics

    Each entry must declare ``benchmark``, ``format``, ``dataset``, ``metric``,
    and EITHER ``min`` (lower floor) OR ``max`` (upper floor). ``direction``
    is optional and inferred from ``min``/``max`` presence; when both are
    present ``direction`` resolves ambiguity. Malformed entries raise
    ``ValueError`` at load time.
    """
    if not path.exists():
        raise FileNotFoundError(f"thresholds file not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level must be a mapping")
    floors_raw = raw.get("absolute_floors", [])
    if floors_raw is None:
        return []
    if not isinstance(floors_raw, list):
        raise ValueError(f"{path}: absolute_floors must be a list")
    floors: list[dict[str, Any]] = []
    for idx, spec in enumerate(floors_raw):
        if not isinstance(spec, dict):
            raise ValueError(
                f"{path}: absolute_floors[{idx}] must be a mapping, got {type(spec).__name__}"
            )
        missing = [f for f in _FLOOR_REQUIRED_FIELDS if spec.get(f) is None]
        if missing:
            raise ValueError(
                f"{path}: absolute_floors[{idx}] missing fields: {missing}"
            )
        has_min = "min" in spec and spec["min"] is not None
        has_max = "max" in spec and spec["max"] is not None
        if not (has_min or has_max):
            raise ValueError(
                f"{path}: absolute_floors[{idx}] must declare either 'min' or 'max'"
            )
        direction = spec.get("direction")
        if direction is None:
            direction = "min" if has_min else "max"
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"{path}: absolute_floors[{idx}] direction={direction!r} "
                f"must be one of {_VALID_DIRECTIONS}"
            )
        threshold_key = "min" if direction == "min" else "max"
        if spec.get(threshold_key) is None:
            raise ValueError(
                f"{path}: absolute_floors[{idx}] direction={direction!r} "
                f"requires a {threshold_key!r} value"
            )
        try:
            threshold = float(spec[threshold_key])
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{path}: absolute_floors[{idx}] {threshold_key}={spec[threshold_key]!r} "
                f"must be numeric"
            ) from e
        floors.append({
            "benchmark": str(spec["benchmark"]),
            "format": str(spec["format"]),
            "dataset": str(spec["dataset"]),
            "metric": str(spec["metric"]),
            "threshold": threshold,
            "direction": direction,
        })
    return floors


def _load_current_wall_cv(
    current_dir: Path,
    benchmark: str,
    fmt: str,
    dataset: str,
) -> float | None:
    """Coefficient of variation (stdev/mean) of ``runs[].wall_s`` in the
    candidate raw JSON. Returns ``None`` when the file is absent, n<2, or
    mean is non-positive (median-vs-zero protection mirroring
    ``_relative``). Surfaces run-to-run noise to the gate report so
    operators can identify ledger candidates without spelunking through
    raw JSON.
    """
    raw = current_dir / "raw" / f"{benchmark}__{fmt}__{dataset}.json"
    if not raw.exists():
        return None
    try:
        data = json.loads(raw.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    walls: list[float] = []
    for run in data.get("runs", []) or []:
        val = run.get("wall_s")
        if val is None:
            continue
        try:
            walls.append(float(val))
        except (TypeError, ValueError):
            continue
    if len(walls) < 2:
        return None
    mean = statistics.fmean(walls)
    if mean <= 0:
        return None
    return statistics.stdev(walls) / mean


def _collect_wall_cvs(
    current_dir: Path,
    keys: list[str],
) -> dict[tuple[str, str, str], float | None]:
    """Compute wall-time CV for each ``benchmark__format__dataset`` key.

    Skips keys whose triple is malformed or whose raw JSON is absent.
    Returns a dict that ``diff_summaries`` consumes for stamping CVs
    onto matching deltas.
    """
    out: dict[tuple[str, str, str], float | None] = {}
    for key in keys:
        try:
            bench, fmt, dataset = key.split("__", 2)
        except ValueError:
            continue
        out[(bench, fmt, dataset)] = _load_current_wall_cv(
            current_dir, bench, fmt, dataset,
        )
    return out


def _load_current_raw_metric(
    current_dir: Path,
    benchmark: str,
    fmt: str,
    dataset: str,
    metric: str,
) -> float | None:
    """Extract ``metric`` from a current-run raw JSON's runs[].extra field.

    Returns the median across runs when the metric is present; ``None`` when
    the file is absent or no run carries the metric. Used by the absolute-
    floor gate to pull telemetry fields (e.g. throughput_mbps) that don't
    appear in the summary.json shape.
    """
    raw = current_dir / "raw" / f"{benchmark}__{fmt}__{dataset}.json"
    if not raw.exists():
        return None
    data = json.loads(raw.read_text())
    values: list[float] = []
    for run in data.get("runs", []) or []:
        extra = run.get("extra", {}) or {}
        val = extra.get(metric)
        if val is None:
            continue
        try:
            values.append(float(val))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return statistics.median(values)


def check_absolute_floors(
    current_dir: Path,
    floors: list[dict[str, Any]],
) -> list[FloorViolation]:
    """Return one FloorViolation per (benchmark, format, dataset) whose
    named metric is missing, NaN, or violates the configured threshold.

    Each spec carries a ``direction`` ("min" or "max") — for ``direction="min"``
    the observed value must be >= threshold (e.g. throughput floors); for
    ``direction="max"`` it must be <= threshold (e.g. wall-time ceilings).
    """
    violations: list[FloorViolation] = []
    for spec in floors:
        benchmark = spec["benchmark"]
        fmt = spec["format"]
        dataset = spec["dataset"]
        metric = spec["metric"]
        threshold = spec["threshold"]
        direction = spec["direction"]
        observed = _load_current_raw_metric(
            current_dir, benchmark, fmt, dataset, metric,
        )
        if observed is None:
            violated = True
        elif direction == "min":
            violated = observed < threshold
        else:  # direction == "max"
            violated = observed > threshold
        if violated:
            violations.append(FloorViolation(
                benchmark=benchmark, fmt=fmt, dataset=dataset,
                metric=metric, threshold=threshold, observed=observed,
                direction=direction,
            ))
    return violations


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_pct(rel: float | None) -> str:
    if rel is None:
        return "—"
    return f"{rel * 100:+.2f}%"


def _fmt_num(x: float | None) -> str:
    if x is None:
        return "—"
    if abs(x) >= 1000:
        return f"{x:,.1f}"
    return f"{x:.3f}"


def _fmt_cv(cv: float | None) -> str:
    if cv is None:
        return "—"
    return f"{cv * 100:.1f}%"


def _fmt_tol(tol: float | None, relaxed: bool, noise_widened: bool = False) -> str:
    if tol is None:
        return "—"
    # ⚠ = flakiness override raised the bound past everything else.
    # ~  = automatic IQR-based widening raised the bound past the
    #      global default. Mutually exclusive in practice (relaxed
    #      implies override > widened bound), but ⚠ wins on display.
    if relaxed:
        suffix = " ⚠"
    elif noise_widened:
        suffix = " ~"
    else:
        suffix = ""
    return f"{tol * 100:.1f}%{suffix}"


def render_markdown(
    deltas: list[Delta],
    fp_mismatches: list[str],
    fp_missing: list[str],
    timing_tol: float,
    allow_fp_drift: bool,
    *,
    suppressed_triples: set[_justifications.Triple] | None = None,
    new_benchmarks: list[tuple[str, str, str]] | None = None,
    floor_violations: list[FloorViolation] | None = None,
    flakiness_overrides_loaded: int = 0,
    high_cv_threshold: float = _HIGH_CV_THRESHOLD,
) -> str:
    timing_regs = [d for d in deltas if d.metric == "median_wall_s" and d.is_regression]
    rss_regs    = [d for d in deltas if d.metric == "peak_rss_mb_median" and d.is_regression]
    size_regs   = [d for d in deltas if d.metric == "file_size_bytes" and d.is_regression]
    suppressed_triples = suppressed_triples or set()
    new_benchmarks = new_benchmarks or []
    floor_violations = floor_violations or []

    noise_widened_count = sum(
        1 for d in deltas
        if d.metric == "median_wall_s" and d.noise_widened
    )

    lines: list[str] = []
    lines.append("# Baseline Regression Report")
    lines.append("")
    lines.append(f"- Timing tolerance: {timing_tol * 100:.1f}%")
    if noise_widened_count:
        lines.append(
            f"- Timing rows widened by IQR (~): {noise_widened_count} "
            f"(baseline IQR/median exceeded the {timing_tol * 100:.1f}% floor)"
        )
    lines.append(f"- Timing regressions:     {len(timing_regs)}")
    lines.append(f"- Peak-RSS regressions:   {len(rss_regs)}")
    lines.append(f"- File-size regressions:  {len(size_regs)}")
    lines.append(f"- Fingerprint mismatches: {len(fp_mismatches)}"
                 f" (allowed: {'yes' if allow_fp_drift else 'no'})")
    lines.append(f"- Fingerprint missing:    {len(fp_missing)}")
    if suppressed_triples:
        lines.append(f"- Justification-suppressed triples: {len(suppressed_triples)}")
    if flakiness_overrides_loaded:
        lines.append(f"- Flakiness overrides loaded: {flakiness_overrides_loaded}")
    if floor_violations:
        lines.append(f"- Absolute-floor violations: {len(floor_violations)}")
    if new_benchmarks:
        lines.append(f"- New benchmarks (informational): {len(new_benchmarks)}")
    lines.append("")

    if fp_mismatches:
        lines.append("## Fingerprint mismatches")
        lines.append("")
        for m in fp_mismatches:
            lines.append(f"- `{m}`")
        lines.append("")

    if fp_missing:
        lines.append("## Fingerprint missing / failed")
        lines.append("")
        for m in fp_missing:
            lines.append(f"- `{m}`")
        lines.append("")

    if timing_regs or rss_regs or size_regs:
        lines.append("## Metric regressions")
        lines.append("")
        lines.append(
            "| Benchmark | Format | Dataset | Metric | Baseline | Current "
            "| Δ | CV | Tol | Status |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in sorted(timing_regs + rss_regs + size_regs,
                        key=lambda x: (x.metric, -(x.relative_change or 0))):
            suppressed = (d.benchmark, d.fmt, d.dataset) in suppressed_triples
            if suppressed:
                status = "suppressed"
            elif d.current is None and d.baseline is not None:
                status = "DISAPPEARED"
            elif d.tolerance_relaxed:
                status = "regressed (over relaxed)"
            else:
                status = "regressed"
            lines.append(
                f"| {d.benchmark} | {d.fmt} | {d.dataset} | {d.metric} | "
                f"{_fmt_num(d.baseline)} | {_fmt_num(d.current)} | "
                f"**{_fmt_pct(d.relative_change)}** | "
                f"{_fmt_cv(d.wall_cv)} | "
                f"{_fmt_tol(d.effective_tolerance, d.tolerance_relaxed, d.noise_widened)} | "
                f"{status} |"
            )
        lines.append("")

    # Informational: timing rows with high run-to-run RSD that aren't
    # already in the regressions table. Helps operators spot ledger
    # candidates without trawling raw JSON.
    high_cv_rows = sorted(
        [d for d in deltas
         if d.metric == "median_wall_s"
         and d.wall_cv is not None
         and d.wall_cv > high_cv_threshold
         and not d.is_regression],
        key=lambda x: -(x.wall_cv or 0),
    )
    if high_cv_rows:
        lines.append("## High-variance rows (informational)")
        lines.append("")
        lines.append(
            f"Wall-time CV > {high_cv_threshold * 100:.0f}% in the candidate "
            f"snapshot. These rows passed the gate but may be flakiness-"
            f"ledger candidates if they keep tripping it on subsequent runs."
        )
        lines.append("")
        lines.append("| Benchmark | Format | Dataset | CV | Tol |")
        lines.append("|---|---|---|---:|---:|")
        for d in high_cv_rows:
            lines.append(
                f"| {d.benchmark} | {d.fmt} | {d.dataset} | "
                f"{_fmt_cv(d.wall_cv)} | "
                f"{_fmt_tol(d.effective_tolerance, d.tolerance_relaxed, d.noise_widened)} |"
            )
        lines.append("")

    if floor_violations:
        lines.append("## Absolute-floor violations")
        lines.append("")
        lines.append("| Benchmark | Format | Dataset | Metric | Threshold | Observed |")
        lines.append("|---|---|---|---|---:|---:|")
        for v in floor_violations:
            obs = "missing" if v.observed is None else f"{v.observed:.3f}"
            op = ">=" if v.direction == "min" else "<="
            lines.append(
                f"| {v.benchmark} | {v.fmt} | {v.dataset} | {v.metric} | "
                f"{op} {v.threshold:.3f} | {obs} |"
            )
        lines.append("")

    if new_benchmarks:
        lines.append("## New benchmarks (informational)")
        lines.append("")
        for bench, fmt, dataset in sorted(new_benchmarks):
            lines.append(f"- `{bench}` / `{fmt}` / `{dataset}`")
        lines.append("")

    return "\n".join(lines)


def to_json_payload(
    deltas: list[Delta],
    fp_mismatches: list[str],
    fp_missing: list[str],
    *,
    suppressed_triples: set[_justifications.Triple] | None = None,
    new_benchmarks: list[tuple[str, str, str]] | None = None,
    floor_violations: list[FloorViolation] | None = None,
    flakiness_overrides_loaded: int = 0,
    flakiness_overrides_applied: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "deltas": [
            {
                "benchmark": d.benchmark, "format": d.fmt, "dataset": d.dataset,
                "metric": d.metric, "baseline": d.baseline, "current": d.current,
                "relative_change": d.relative_change, "is_regression": d.is_regression,
                "effective_tolerance": d.effective_tolerance,
                "tolerance_relaxed": d.tolerance_relaxed,
                "noise_widened": d.noise_widened,
                "iqr_ratio": d.iqr_ratio,
                "wall_cv": d.wall_cv,
            }
            for d in deltas
        ],
        "fingerprint_mismatches": fp_mismatches,
        "fingerprint_missing": fp_missing,
    }
    if suppressed_triples is not None:
        payload["suppressed_triples"] = [
            {"benchmark": b, "format": f, "dataset": d}
            for (b, f, d) in sorted(suppressed_triples)
        ]
    if new_benchmarks is not None:
        payload["new_benchmarks"] = [
            {"benchmark": b, "format": f, "dataset": d}
            for (b, f, d) in sorted(new_benchmarks)
        ]
    if floor_violations is not None:
        payload["floor_violations"] = [
            {
                "benchmark": v.benchmark, "format": v.fmt, "dataset": v.dataset,
                "metric": v.metric, "threshold": v.threshold,
                "direction": v.direction, "observed": v.observed,
            }
            for v in floor_violations
        ]
    payload["flakiness_overrides_loaded"] = flakiness_overrides_loaded
    if flakiness_overrides_applied is not None:
        payload["flakiness_overrides_applied"] = flakiness_overrides_applied
    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--baseline", type=Path, default=None,
        help="Canonical baseline snapshot directory. Defaults to "
             "`results/baselines/LATEST` (maintained by `promote_baseline.py`) "
             "for the on-demand gate workflow; pass an explicit path to pin "
             "a historical baseline.",
    )
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--timing-tolerance", type=float, default=0.03)
    parser.add_argument("--rss-tolerance",    type=float, default=0.10)
    parser.add_argument("--size-tolerance",   type=float, default=0.01)
    parser.add_argument(
        "--iqr-k", type=float, default=1.5,
        help="Multiplier on the baseline's wall-time IQR/median when "
             "widening per-row timing tolerance. Effective tol per "
             "median_wall_s row is max(--timing-tolerance, --iqr-k * "
             "baseline_iqr / baseline_median). Set 0 to disable noise "
             "widening (gate falls back to the fixed --timing-tolerance "
             "everywhere). Default 1.5 matches the standard IQR-based "
             "outlier multiplier; bump to 2.0 for more permissive "
             "preemptible-node gating.",
    )
    parser.add_argument(
        "--allow-fingerprint-drift", action="store_true",
        help="Don't fail on accelerator fingerprint mismatches (use for intentional "
             "numerical shifts like H10/M13).  Mismatches are still reported.",
    )
    parser.add_argument("--report-json", type=Path, default=None,
                        help="Also write a JSON payload to this path.")
    parser.add_argument(
        "--gate", action="store_true",
        help="Treat disappearing benchmarks as regressions, apply "
             "justifications to suppress known-bad rows, and run absolute "
             "floor checks from --thresholds. Without --gate, the diff "
             "surface is unchanged (backwards-compatible).",
    )
    parser.add_argument(
        "--justifications", type=Path, default=None,
        help="Directory of justification markdown files. Under --gate, "
             "defaults to `benchmarks/comprehensive/results/justifications/` "
             "when that exists; triples listed in any active justification "
             "are suppressed from the regression tally. See "
             "benchmarks/README.md for the markdown front-matter format.",
    )
    parser.add_argument(
        "--thresholds", type=Path, default=None,
        help="YAML file declaring absolute-floor thresholds "
             "(e.g. cloud_push throughput_mbps >= 50). Under --gate, "
             "defaults to `benchmarks/comprehensive/thresholds.yaml` "
             "when that file exists.",
    )
    parser.add_argument(
        "--flakiness", type=Path, default=None,
        help="Directory of flakiness-ledger markdown files declaring "
             "per-row relaxed tolerances. Under --gate, defaults to "
             "`benchmarks/comprehensive/results/flakiness/` when that "
             "exists. Each (benchmark, format, dataset, metric) row "
             "matched by an active override is gated at the override's "
             "tolerance instead of the global --timing-tolerance / "
             "--rss-tolerance / --size-tolerance. See "
             "benchmarks/README.md for the markdown front-matter format.",
    )
    parser.add_argument(
        "--only-benchmarks", nargs="+", default=None, metavar="BENCH",
        help="Restrict the diff to baseline rows whose benchmark is in "
             "this list (e.g. `--only-benchmarks accel_pca`). Without this "
             "flag, every benchmark in the baseline is compared and any "
             "missing in the candidate is flagged as DISAPPEARED under "
             "--gate. Use for narrow spot-check gates over a subset of the "
             "canonical surface. Floor specs whose benchmark isn't in the "
             "list are also dropped.",
    )
    args = parser.parse_args()

    # On-demand workflow: auto-resolve --baseline from LATEST if omitted.
    if args.baseline is None:
        resolved = _resolve_default_baseline()
        if resolved is None:
            print(
                "ERROR: --baseline not provided and no canonical baseline "
                "exists. Promote a snapshot first:\n"
                "  python benchmarks/comprehensive/scripts/promote_baseline.py "
                "--snapshot <candidate-dir> --version <label>",
                file=sys.stderr,
            )
            return 2
        args.baseline = resolved

    # Under --gate, auto-default the committed justifications + thresholds
    # + flakiness paths when the operator didn't pass them. Keeps on-demand
    # invocations to a single flag.
    if args.gate:
        if args.justifications is None and _DEFAULT_JUSTIFICATIONS_DIR.is_dir():
            args.justifications = _DEFAULT_JUSTIFICATIONS_DIR
        if args.thresholds is None and _DEFAULT_THRESHOLDS_YAML.is_file():
            args.thresholds = _DEFAULT_THRESHOLDS_YAML
        if args.flakiness is None and _DEFAULT_FLAKINESS_DIR.is_dir():
            args.flakiness = _DEFAULT_FLAKINESS_DIR

    try:
        baseline_summary = _load_summary(args.baseline)
        current_summary  = _load_summary(args.current)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # --only-benchmarks: filter both baseline and current row dicts so the
    # diff (and the gate's DISAPPEARED check) only operates on the requested
    # benchmark slice. Keeps narrow spot-check gates honest — the canonical
    # baseline covers the full surface, but a candidate captured with
    # `--benchmarks accel_pca` shouldn't be flagged for everything else
    # going missing.
    if args.only_benchmarks:
        allow = set(args.only_benchmarks)
        baseline_summary = dict(baseline_summary)
        baseline_summary["rows"] = {
            k: v for k, v in baseline_summary.get("rows", {}).items()
            if k.split("__", 2)[0] in allow
        }
        current_summary = dict(current_summary)
        current_summary["rows"] = {
            k: v for k, v in current_summary.get("rows", {}).items()
            if k.split("__", 2)[0] in allow
        }

    baseline_fp = _load_fingerprints(args.baseline)
    current_fp  = _load_fingerprints(args.current)

    # Under --gate: load justifications, flakiness overrides, and
    # absolute-floor specs; surface new benchmarks informationally.
    suppressed: set[_justifications.Triple] = set()
    floor_violations: list[FloorViolation] = []
    new_benchmarks: list[tuple[str, str, str]] = []
    tolerance_overrides: dict[_flakiness.Quad, float] = {}
    if args.gate:
        if args.justifications is not None:
            suppressed, _loaded = _justifications.load_active_triples(
                args.justifications,
            )
        if args.flakiness is not None:
            tolerance_overrides, _loaded_flaky = _flakiness.load_active_overrides(
                args.flakiness,
            )
            if args.only_benchmarks:
                allow = set(args.only_benchmarks)
                tolerance_overrides = {
                    quad: tol for quad, tol in tolerance_overrides.items()
                    if quad[0] in allow
                }
        if args.thresholds is not None:
            try:
                floors = _load_thresholds_yaml(args.thresholds)
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
            if args.only_benchmarks:
                allow = set(args.only_benchmarks)
                floors = [f for f in floors if f.get("benchmark") in allow]
            floor_violations = check_absolute_floors(args.current, floors)
        # New benchmarks: current has the key, baseline doesn't.
        base_keys = set(baseline_summary.get("rows", {}).keys())
        cur_keys = set(current_summary.get("rows", {}).keys())
        for key in sorted(cur_keys - base_keys):
            try:
                bench, fmt, ds = key.split("__", 2)
            except ValueError:
                continue
            new_benchmarks.append((bench, fmt, ds))

    # Compute per-row wall-time CV from the candidate raw JSONs. Cheap
    # (one extra read per (benchmark, format, dataset)) and only used
    # for reporting — never affects regression status.
    wall_cvs = _collect_wall_cvs(
        args.current, list(current_summary.get("rows", {}).keys()),
    )

    deltas = diff_summaries(
        baseline_summary, current_summary,
        args.timing_tolerance, args.rss_tolerance, args.size_tolerance,
        gate=args.gate,
        tolerance_overrides=tolerance_overrides,
        wall_cvs=wall_cvs,
        iqr_k=args.iqr_k,
    )
    fp_mismatches, fp_missing = diff_fingerprints(baseline_fp, current_fp)

    # Build the "applied" list — overrides that matched at least one
    # delta — so the JSON payload distinguishes "loaded but inert" from
    # "loaded and used".
    applied_quads = {
        (d.benchmark, d.fmt, d.dataset, d.metric)
        for d in deltas if d.tolerance_relaxed
    }
    flakiness_applied = [
        {
            "benchmark": q[0], "format": q[1], "dataset": q[2],
            "metric": q[3], "tolerance": tolerance_overrides[q],
        }
        for q in sorted(applied_quads)
    ]

    report = render_markdown(
        deltas, fp_mismatches, fp_missing,
        args.timing_tolerance, args.allow_fingerprint_drift,
        suppressed_triples=suppressed,
        new_benchmarks=new_benchmarks,
        floor_violations=floor_violations,
        flakiness_overrides_loaded=len(tolerance_overrides),
    )
    print(report)

    if args.report_json:
        args.report_json.write_text(json.dumps(
            to_json_payload(
                deltas, fp_mismatches, fp_missing,
                suppressed_triples=suppressed,
                new_benchmarks=new_benchmarks,
                floor_violations=floor_violations,
                flakiness_overrides_loaded=len(tolerance_overrides),
                flakiness_overrides_applied=flakiness_applied,
            ),
            indent=2, default=str,
        ))

    # Drop suppressed rows from the regression tally (only matters under
    # --gate, since suppression set is empty otherwise).
    active_regressions = [
        d for d in deltas
        if d.is_regression
        and (d.benchmark, d.fmt, d.dataset) not in suppressed
    ]
    if active_regressions:
        return 1
    if floor_violations:
        return 1
    if fp_mismatches and not args.allow_fingerprint_drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
