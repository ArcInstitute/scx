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
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    return json.loads(f.read_text())


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
) -> list[Delta]:
    """One Delta row per (benchmark, format, dataset, metric) tuple."""
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
) -> str:
    timing_regs = [d for d in deltas if d.metric == "median_wall_s" and d.is_regression]
    rss_regs    = [d for d in deltas if d.metric == "peak_rss_mb_median" and d.is_regression]
    size_regs   = [d for d in deltas if d.metric == "file_size_bytes" and d.is_regression]

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
        lines.append("| Benchmark | Format | Dataset | Metric | Baseline | Current | Δ |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in sorted(timing_regs + rss_regs + size_regs,
                        key=lambda x: (x.metric, -(x.relative_change or 0))):
            lines.append(
                f"| {d.benchmark} | {d.fmt} | {d.dataset} | {d.metric} | "
                f"{_fmt_num(d.baseline)} | {_fmt_num(d.current)} | "
                f"**{_fmt_pct(d.relative_change)}** |"
            )
        lines.append("")

    return "\n".join(lines)


def to_json_payload(
    deltas: list[Delta],
    fp_mismatches: list[str],
    fp_missing: list[str],
) -> dict[str, Any]:
    return {
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--baseline", required=True, type=Path)
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
    args = parser.parse_args()

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
    )
    fp_mismatches, fp_missing = diff_fingerprints(baseline_fp, current_fp)

    report = render_markdown(
        deltas, fp_mismatches, fp_missing,
        args.timing_tolerance, args.allow_fingerprint_drift,
    )
    print(report)

    if args.report_json:
        args.report_json.write_text(json.dumps(
            to_json_payload(deltas, fp_mismatches, fp_missing),
            indent=2, default=str,
        ))

    regressions = any(d.is_regression for d in deltas)
    if regressions:
        return 1
    if fp_mismatches and not args.allow_fingerprint_drift:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
