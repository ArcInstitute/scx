#!/usr/bin/env python3
"""
Compare a post-change comprehensive benchmark run against the pre-change baseline.

This is the regression gate named in 2026-04-17_CODE-REVIEW.md §12.4:

  "Each PR's acceptance criteria include: all previously-stable benchmarks
   still within 3% of baseline, and all parity tests pass with unchanged
   tolerance."

Given two directories produced by capture_baseline.py, this script reports:

  - Timing deltas per (benchmark, format, dataset).  Flags anything outside
    --timing-tolerance (default 3%) as a regression.
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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# _justifications lives beside this script — add parent to sys.path so
# `python scripts/compare_against_baseline.py` works without a package
# install. When invoked as `-m`, relative import works; otherwise fall back
# to path-based import.
try:
    from . import _justifications
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _justifications  # type: ignore[no-redef]


# Default locations used by the on-demand gate workflow. All overridable
# via CLI flags; surfaced as module-level constants so the one-shot
# ``gate_candidate.sh`` wrapper and the tests can reference them.
_COMPREHENSIVE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_BASELINES_DIR = _COMPREHENSIVE_DIR / "results" / "baselines"
_DEFAULT_LATEST_LINK = _DEFAULT_BASELINES_DIR / "LATEST"
_DEFAULT_JUSTIFICATIONS_DIR = _COMPREHENSIVE_DIR / "results" / "justifications"
_DEFAULT_THRESHOLDS_YAML = _COMPREHENSIVE_DIR / "thresholds.yaml"


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


def _load_summary(path: Path) -> dict[str, Any]:
    f = path / "summary.json"
    if not f.exists():
        raise FileNotFoundError(f"summary.json not found at {f}")
    data = json.loads(f.read_text())
    # summary.json itself doesn't carry schema_version today — the shape
    # stabilized pre-Phase-5. Raw per-run JSONs DO carry schema_version now;
    # refuse to diff when we detect an unknown future version.
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
) -> list[Delta]:
    """One Delta row per (benchmark, format, dataset, metric) tuple.

    When ``gate`` is False (default), behavior is unchanged: a row with
    baseline=None or current=None produces ``relative_change=None`` and
    ``is_regression=False``. When ``gate`` is True, a row present in
    baseline but missing from current (``current=None``) is flagged as a
    regression — a disappeared benchmark is a silent gap the gate must
    surface. Appearing benchmarks (baseline=None, current!=None) stay
    non-regressions; ``render_markdown`` logs them informationally.
    """
    deltas: list[Delta] = []
    base_rows = baseline_summary.get("rows", {})
    cur_rows = current_summary.get("rows", {})
    all_keys = sorted(set(base_rows) | set(cur_rows))

    for key in all_keys:
        try:
            bench, fmt, dataset = key.split("__", 2)
        except ValueError:
            # Key doesn't follow the expected schema.  Skip silently; summary
            # regeneration will have already warned about malformed entries.
            continue

        base = base_rows.get(key, {})
        cur = cur_rows.get(key, {})

        for metric, tol in [
            ("median_wall_s",      timing_tol),
            ("peak_rss_mb_median", rss_tol),
            ("file_size_bytes",    size_tol),
        ]:
            b, c = base.get(metric), cur.get(metric)
            rel = _relative(b, c)
            regressed = rel is not None and rel > tol
            if gate and b is not None and c is None:
                # Disappeared benchmark — only surface this once per triple
                # so the report doesn't double-count each metric. Use the
                # timing row as the canonical signal.
                regressed = regressed or (metric == "median_wall_s")
            deltas.append(
                Delta(bench, fmt, dataset, metric, b, c, rel, regressed)
            )

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
    minimum: float
    observed: float | None


def _load_thresholds_yaml(path: Path) -> list[dict[str, Any]]:
    """Load ``thresholds.yaml`` as a list of floor dicts.

    Keeps the parser tiny (no PyYAML dep in minimal envs). Supports only the
    narrow shape used by ``thresholds.yaml``:

        absolute_floors:
          - benchmark: cloud_push
            format: scx_auto
            metric: throughput_mbps
            min: 50.0

    Every top-level line must be ``absolute_floors:`` or whitespace/comment;
    entries are consumed as a list of dicts with string scalar values.
    """
    if not path.exists():
        raise FileNotFoundError(f"thresholds file not found: {path}")
    text = path.read_text()
    lines = text.splitlines()
    floors: list[dict[str, Any]] = []
    in_list = False
    current: dict[str, Any] | None = None
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and not line.startswith("\t"):
            in_list = stripped == "absolute_floors:"
            continue
        if not in_list:
            continue
        inner = stripped
        if inner.startswith("- "):
            if current is not None:
                floors.append(current)
            current = {}
            inner = inner[2:]
        if ":" not in inner:
            continue
        k, _, v = inner.partition(":")
        k = k.strip()
        v = v.strip()
        if v.startswith("'") and v.endswith("'"):
            v = v[1:-1]
        elif v.startswith('"') and v.endswith('"'):
            v = v[1:-1]
        if current is None:
            continue
        try:
            current[k] = float(v)
        except ValueError:
            current[k] = v
    if current is not None:
        floors.append(current)
    return floors


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
    values.sort()
    return values[len(values) // 2]


def check_absolute_floors(
    current_dir: Path,
    floors: list[dict[str, Any]],
) -> list[FloorViolation]:
    """Return one FloorViolation per (benchmark, format, dataset) whose
    named metric is missing, NaN, or below the configured minimum.
    """
    violations: list[FloorViolation] = []
    for spec in floors:
        benchmark = spec.get("benchmark")
        fmt = spec.get("format")
        dataset = spec.get("dataset")
        metric = spec.get("metric")
        minimum = spec.get("min")
        if not all([benchmark, fmt, dataset, metric]) or minimum is None:
            continue
        observed = _load_current_raw_metric(
            current_dir, benchmark, fmt, dataset, metric,
        )
        if observed is None or observed < float(minimum):
            violations.append(FloorViolation(
                benchmark=benchmark, fmt=fmt, dataset=dataset,
                metric=metric, minimum=float(minimum), observed=observed,
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
) -> str:
    timing_regs = [d for d in deltas if d.metric == "median_wall_s" and d.is_regression]
    rss_regs    = [d for d in deltas if d.metric == "peak_rss_mb_median" and d.is_regression]
    size_regs   = [d for d in deltas if d.metric == "file_size_bytes" and d.is_regression]
    suppressed_triples = suppressed_triples or set()
    new_benchmarks = new_benchmarks or []
    floor_violations = floor_violations or []

    lines: list[str] = []
    lines.append("# Baseline Regression Report")
    lines.append("")
    lines.append(f"- Timing tolerance: {timing_tol * 100:.1f}%")
    lines.append(f"- Timing regressions:     {len(timing_regs)}")
    lines.append(f"- Peak-RSS regressions:   {len(rss_regs)}")
    lines.append(f"- File-size regressions:  {len(size_regs)}")
    lines.append(f"- Fingerprint mismatches: {len(fp_mismatches)}"
                 f" (allowed: {'yes' if allow_fp_drift else 'no'})")
    lines.append(f"- Fingerprint missing:    {len(fp_missing)}")
    if suppressed_triples:
        lines.append(f"- Justification-suppressed triples: {len(suppressed_triples)}")
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
        lines.append("| Benchmark | Format | Dataset | Metric | Baseline | Current | Δ | Status |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in sorted(timing_regs + rss_regs + size_regs,
                        key=lambda x: (x.metric, -(x.relative_change or 0))):
            suppressed = (d.benchmark, d.fmt, d.dataset) in suppressed_triples
            status = "suppressed" if suppressed else (
                "DISAPPEARED" if d.current is None and d.baseline is not None
                else "regressed"
            )
            lines.append(
                f"| {d.benchmark} | {d.fmt} | {d.dataset} | {d.metric} | "
                f"{_fmt_num(d.baseline)} | {_fmt_num(d.current)} | "
                f"**{_fmt_pct(d.relative_change)}** | {status} |"
            )
        lines.append("")

    if floor_violations:
        lines.append("## Absolute-floor violations")
        lines.append("")
        lines.append("| Benchmark | Format | Dataset | Metric | Min | Observed |")
        lines.append("|---|---|---|---|---:|---:|")
        for v in floor_violations:
            obs = "missing" if v.observed is None else f"{v.observed:.3f}"
            lines.append(
                f"| {v.benchmark} | {v.fmt} | {v.dataset} | {v.metric} | "
                f"{v.minimum:.3f} | {obs} |"
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
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "deltas": [
            {
                "benchmark": d.benchmark, "format": d.fmt, "dataset": d.dataset,
                "metric": d.metric, "baseline": d.baseline, "current": d.current,
                "relative_change": d.relative_change, "is_regression": d.is_regression,
            }
            for d in deltas
        ],
        "fingerprint_mismatches": fp_mismatches,
        "fingerprint_missing": fp_missing,
    }
    if suppressed_triples is not None:
        payload["suppressed_triples"] = sorted(
            {"benchmark": b, "format": f, "dataset": d}
            for (b, f, d) in suppressed_triples
        ) if False else [
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
                "metric": v.metric, "min": v.minimum, "observed": v.observed,
            }
            for v in floor_violations
        ]
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
    # paths when the operator didn't pass them. Keeps on-demand invocations
    # to a single flag.
    if args.gate:
        if args.justifications is None and _DEFAULT_JUSTIFICATIONS_DIR.is_dir():
            args.justifications = _DEFAULT_JUSTIFICATIONS_DIR
        if args.thresholds is None and _DEFAULT_THRESHOLDS_YAML.is_file():
            args.thresholds = _DEFAULT_THRESHOLDS_YAML

    try:
        baseline_summary = _load_summary(args.baseline)
        current_summary  = _load_summary(args.current)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    baseline_fp = _load_fingerprints(args.baseline)
    current_fp  = _load_fingerprints(args.current)

    deltas = diff_summaries(
        baseline_summary, current_summary,
        args.timing_tolerance, args.rss_tolerance, args.size_tolerance,
        gate=args.gate,
    )
    fp_mismatches, fp_missing = diff_fingerprints(baseline_fp, current_fp)

    # Under --gate: load justifications + absolute floors, and surface new
    # benchmarks informationally.
    suppressed: set[_justifications.Triple] = set()
    floor_violations: list[FloorViolation] = []
    new_benchmarks: list[tuple[str, str, str]] = []
    if args.gate:
        if args.justifications is not None:
            suppressed, _loaded = _justifications.load_active_triples(
                args.justifications,
            )
        if args.thresholds is not None:
            try:
                floors = _load_thresholds_yaml(args.thresholds)
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
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

    report = render_markdown(
        deltas, fp_mismatches, fp_missing,
        args.timing_tolerance, args.allow_fingerprint_drift,
        suppressed_triples=suppressed,
        new_benchmarks=new_benchmarks,
        floor_violations=floor_violations,
    )
    print(report)

    if args.report_json:
        args.report_json.write_text(json.dumps(
            to_json_payload(
                deltas, fp_mismatches, fp_missing,
                suppressed_triples=suppressed,
                new_benchmarks=new_benchmarks,
                floor_violations=floor_violations,
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
