#!/usr/bin/env python3
"""
DEPRECATED as of Phase 9 (2026-04-23). Use the comprehensive framework:

    python benchmarks/comprehensive/scripts/compare_against_baseline.py \\
        --current <captured-snapshot> --gate

This gates against `comprehensive/results/baselines/LATEST →
v0.6.0-gpu-phase1-7`, which now includes accel_* dimensions alongside
format benchmarks. This file remains in-tree for one release for
rollback convenience; future diffs should use `compare_against_baseline.py`.

----------------------------------------------------------------------------

GPU accelerator regression diff — Phase 8.6.

Reads a PRE and a POST directory of `gpu_*.json` files (both produced by the
standalone `benchmark_gpu_*.py` scripts), computes per-op deltas, applies the
Phase 8.6 threshold table, and emits:

1. `--report` Markdown with per-benchmark pre/post/delta/verdict tables.
2. `--json-out` machine-readable verdict + every diff row.
3. Exit code: 0 = all within tolerance, 1 = regression(s) flagged,
   2 = missing inputs / unreadable JSON / empty PRE+POST.

Usage:

    python benchmarks/scripts/gpu_regression_diff.py \\
        --pre  benchmarks/results/pre_phases_1_7_baseline_2026_03 \\
        --post benchmarks/results \\
        --report benchmarks/results/phases_1_7_gpu_regression_report.md

Schemas handled (all are direct outputs of the `benchmark_gpu_*.py` scripts):

| File | Shape | Key stats |
|---|---|---|
| `gpu_pipeline_timing.json` | dict | `speedup`, `per_op_speedup.{pca,knn,umap,leiden}`, optional Phase-7 keys `gpu_preprocessing_median_breakdown`, `gpu_pca_variants.{gpu_cov_pca,gpu_randomized_pca_householder,gpu_randomized_pca_chol}.median_s` |
| `gpu_pca_timing.json` | list[dict] | per dataset: `gpu_median_s`, `speedup` |
| `gpu_knn_timing.json` | list[dict] | per dataset: `speedup` |
| `gpu_umap_timing.json` | list[dict] | per dataset: `speedup` |
| `gpu_preprocess_timing.json` | list[dict] | per dataset: `scx_cpu_median_s`, `speedup_vs_scanpy` (Phase-7.3 addition: `gpu_preprocess_device.json` keyed by op) |

The Phase-7.2 `gpu_pca_variants` and Phase-7.3 `gpu_preprocess_device.json`
keys are **optional** in PRE (they didn't exist pre-Phase-7); the diff marks
them as "new benchmark, POST only" rather than a regression.

Thresholds are indicative (matching the Phase 8.6 table in
`GPU-ACC-SPEED-UP.md`) and can be tuned via `--timing-tolerance` (fraction
of pre median allowed to slip before flagging, default 0.10) and
`--strict` (halves all tolerances).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------

@dataclass
class Row:
    """One pre/post comparison entry."""
    source: str              # source JSON filename
    benchmark: str           # e.g. "gpu_pipeline_timing:pca"
    dataset: str             # e.g. "census_1m" (or "-" when n/a)
    metric: str              # e.g. "speedup_vs_cpu" / "median_s"
    pre: float | None
    post: float | None
    unit: str = ""
    higher_is_better: bool = True
    verdict: str = "ok"      # "ok" | "regression" | "improvement" | "new" | "disappeared"
    delta_pct: float | None = None
    note: str = ""


@dataclass
class DiffReport:
    rows: list[Row] = field(default_factory=list)
    pre_path: str = ""
    post_path: str = ""
    tolerance: float = 0.10
    strict: bool = False

    @property
    def regressions(self) -> list[Row]:
        return [r for r in self.rows if r.verdict == "regression"]

    @property
    def improvements(self) -> list[Row]:
        return [r for r in self.rows if r.verdict == "improvement"]

    @property
    def new_entries(self) -> list[Row]:
        return [r for r in self.rows if r.verdict == "new"]

    @property
    def disappeared(self) -> list[Row]:
        return [r for r in self.rows if r.verdict == "disappeared"]


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------

def _load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        print(f"warning: {path}: invalid JSON ({e})", file=sys.stderr)
        return None


def _get(d: dict | None, *keys: str, default: Any = None) -> Any:
    """Nested-dict getter: `_get(d, 'a', 'b')` == d.get('a', {}).get('b')."""
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


# ----------------------------------------------------------------------
# Per-file diff rules
# ----------------------------------------------------------------------

def _pct_change(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None or pre == 0:
        return None
    return (post - pre) / pre


def _classify(row: Row, tolerance: float) -> str:
    """Set row.verdict + row.delta_pct based on pre/post and tolerance."""
    if row.pre is None and row.post is None:
        return "ok"
    if row.pre is None:
        return "new"
    if row.post is None:
        return "disappeared"

    row.delta_pct = _pct_change(row.pre, row.post)
    if row.delta_pct is None:
        return "ok"

    # For "higher is better" metrics (speedup, throughput), regressing means
    # the post value dropped *below* pre by more than tolerance.
    # For "lower is better" metrics (wall time), regressing means post rose
    # *above* pre by more than tolerance.
    if row.higher_is_better:
        # post < pre * (1 - tol) => regression
        if row.delta_pct < -tolerance:
            return "regression"
        if row.delta_pct > tolerance:
            return "improvement"
    else:
        if row.delta_pct > tolerance:
            return "regression"
        if row.delta_pct < -tolerance:
            return "improvement"
    return "ok"


def diff_pipeline_timing(pre_dir: Path, post_dir: Path) -> list[Row]:
    """Handles `gpu_pipeline_timing.json` — a single dict."""
    rows: list[Row] = []
    fn = "gpu_pipeline_timing.json"
    pre = _load_json(pre_dir / fn)
    post = _load_json(post_dir / fn)
    if pre is None and post is None:
        return rows

    dataset = _get(post or pre, "dataset") or "-"

    # End-to-end speedup.
    rows.append(Row(
        source=fn, benchmark="pipeline:e2e_speedup", dataset=dataset,
        metric="speedup_vs_cpu",
        pre=_get(pre, "speedup"),
        post=_get(post, "speedup"),
        unit="x", higher_is_better=True,
    ))

    # Per-op speedups.
    for op in ["pca", "knn", "umap", "leiden"]:
        rows.append(Row(
            source=fn, benchmark=f"pipeline:{op}", dataset=dataset,
            metric="speedup_vs_cpu",
            pre=_get(pre, "per_op_speedup", op),
            post=_get(post, "per_op_speedup", op),
            unit="x", higher_is_better=True,
        ))

    # GPU wall times per op.
    for op in ["pca", "knn", "umap", "leiden", "total"]:
        rows.append(Row(
            source=fn, benchmark=f"pipeline:{op}", dataset=dataset,
            metric="gpu_median_s",
            pre=_get(pre, "gpu_median_breakdown", op),
            post=_get(post, "gpu_median_breakdown", op),
            unit="s", higher_is_better=False,
        ))

    # Phase 7.2 additions — only present post.
    for sub in ["gpu_cov_pca", "gpu_randomized_pca_householder", "gpu_randomized_pca_chol"]:
        rows.append(Row(
            source=fn, benchmark=f"pca_variants:{sub}", dataset=dataset,
            metric="median_s",
            pre=_get(pre, "gpu_pca_variants", sub, "median_s"),
            post=_get(post, "gpu_pca_variants", sub, "median_s"),
            unit="s", higher_is_better=False,
            note="Phase 7.2 addition — absent in pre" if _get(pre, "gpu_pca_variants", sub) is None else "",
        ))

    for op in ["normalize_total", "log1p", "highly_variable_genes", "total"]:
        rows.append(Row(
            source=fn, benchmark=f"preprocessing:{op}", dataset=dataset,
            metric="gpu_median_s",
            pre=_get(pre, "gpu_preprocessing_median_breakdown", op),
            post=_get(post, "gpu_preprocessing_median_breakdown", op),
            unit="s", higher_is_better=False,
            note="Phase 7.2 --attribute-preprocessing addition" if _get(pre, "gpu_preprocessing_median_breakdown") is None else "",
        ))
    return rows


def diff_list_of_benchmarks(
    pre_dir: Path, post_dir: Path, fn: str,
    metric_keys: list[tuple[str, str, bool]],
) -> list[Row]:
    """Handles list-of-dict JSONs keyed by (benchmark, dataset).

    `metric_keys` is a list of (json_key, display_metric, higher_is_better).
    """
    rows: list[Row] = []
    pre_list = _load_json(pre_dir / fn) or []
    post_list = _load_json(post_dir / fn) or []
    if not isinstance(pre_list, list):
        pre_list = []
    if not isinstance(post_list, list):
        post_list = []

    def index_by_dataset(lst: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in lst:
            if isinstance(e, dict):
                out[e.get("dataset", "-")] = e
        return out

    pre_by = index_by_dataset(pre_list)
    post_by = index_by_dataset(post_list)
    datasets = sorted(set(pre_by) | set(post_by))
    for ds in datasets:
        pre_e = pre_by.get(ds, {})
        post_e = post_by.get(ds, {})
        bench = pre_e.get("benchmark") or post_e.get("benchmark") or fn.replace(".json", "")
        for key, label, higher_is_better in metric_keys:
            rows.append(Row(
                source=fn, benchmark=f"{bench}", dataset=ds, metric=label,
                pre=pre_e.get(key) if isinstance(pre_e.get(key), (int, float)) else None,
                post=post_e.get(key) if isinstance(post_e.get(key), (int, float)) else None,
                unit="s" if "median_s" in label else ("x" if "speedup" in label else ""),
                higher_is_better=higher_is_better,
            ))
    return rows


def diff_preprocess_device(pre_dir: Path, post_dir: Path) -> list[Row]:
    """Handles Phase 7.3 `gpu_preprocess_device.json`.

    Shape: list of {"dataset", "n_obs", "ops": {op: {"cpu": {..}, "gpu": {..},
    "speedup": float}}}. Pre is expected to be absent; if present, still diff.
    """
    rows: list[Row] = []
    fn = "gpu_preprocess_device.json"
    pre_list = _load_json(pre_dir / fn) or []
    post_list = _load_json(post_dir / fn) or []

    def index_by_dataset(lst: list[dict]) -> dict[str, dict]:
        return {e["dataset"]: e for e in lst if isinstance(e, dict) and "dataset" in e}

    pre_by = index_by_dataset(pre_list if isinstance(pre_list, list) else [])
    post_by = index_by_dataset(post_list if isinstance(post_list, list) else [])
    datasets = sorted(set(pre_by) | set(post_by))
    for ds in datasets:
        pre_ops = _get(pre_by.get(ds, {}), "ops") or {}
        post_ops = _get(post_by.get(ds, {}), "ops") or {}
        for op in sorted(set(pre_ops) | set(post_ops)):
            rows.append(Row(
                source=fn, benchmark=f"preprocess_device:{op}", dataset=ds,
                metric="cpu_vs_gpu_speedup",
                pre=_get(pre_ops, op, "speedup"),
                post=_get(post_ops, op, "speedup"),
                unit="x", higher_is_better=True,
                note="Phase 7.3 addition" if not pre_ops else "",
            ))
            rows.append(Row(
                source=fn, benchmark=f"preprocess_device:{op}", dataset=ds,
                metric="gpu_median_s",
                pre=_get(pre_ops, op, "gpu", "median_s"),
                post=_get(post_ops, op, "gpu", "median_s"),
                unit="s", higher_is_better=False,
            ))
    return rows


# ----------------------------------------------------------------------
# Threshold table (mirrors GPU-ACC-SPEED-UP.md §8.6)
# ----------------------------------------------------------------------

# Hard floors for specific signals: (predicate, reason).
# If the predicate returns True (regression detected), emit a loud flag.
def apply_hard_floors(report: DiffReport) -> list[str]:
    flags: list[str] = []

    def row(bench: str, metric: str) -> Row | None:
        for r in report.rows:
            if r.benchmark == bench and r.metric == metric:
                return r
        return None

    # Covariance PCA on 2K HVGs at 1M cells — Phase 2 target.
    # The variant row is post-only (Phase 7.2 addition), so we compare post
    # against the pre randomized-PCA speedup at 1M instead.
    pre_pca_speedup = row("pipeline:pca", "speedup_vs_cpu")
    if pre_pca_speedup and pre_pca_speedup.pre is not None and pre_pca_speedup.post is not None:
        if pre_pca_speedup.post < pre_pca_speedup.pre - 0.01:
            flags.append(
                f"pipeline:pca speedup regressed {pre_pca_speedup.pre:.2f}x → {pre_pca_speedup.post:.2f}x — "
                f"investigate Phase 2/3 refactor (expected ≥ pre)"
            )
        elif pre_pca_speedup.post < 1.5:
            flags.append(
                f"pipeline:pca post speedup {pre_pca_speedup.post:.2f}x < 1.5x target — Phase 2 cov-PCA "
                f"either didn't land or didn't route (check adata.uns['pca']['backend'])"
            )

    # Cholesky vs Householder — Cholesky should be ≤ Householder.
    chol = row("pca_variants:gpu_randomized_pca_chol", "median_s")
    hh = row("pca_variants:gpu_randomized_pca_householder", "median_s")
    if chol and hh and chol.post is not None and hh.post is not None:
        if chol.post > hh.post * 1.1:
            flags.append(
                f"pca_variants: Cholesky median {chol.post:.2f}s > Householder {hh.post:.2f}s — "
                f"CholeskyQR2 not faster; check that gpu_cholesky_qr2 is reached"
            )

    # End-to-end pipeline: post should be ≥ pre speedup.
    e2e = row("pipeline:e2e_speedup", "speedup_vs_cpu")
    if e2e and e2e.pre is not None and e2e.post is not None:
        if e2e.post < e2e.pre - 0.1:
            flags.append(
                f"pipeline:e2e_speedup regressed {e2e.pre:.2f}x → {e2e.post:.2f}x"
            )

    # Unmodified-path drift — kNN / UMAP / standalone Leiden should be
    # within ±10% of pre. `_classify` already flags this as regression if
    # it crosses tolerance; we just escalate here when the regression is
    # on a path Phases 1–7 didn't modify.
    for bench in ["pipeline:knn", "pipeline:umap", "pipeline:leiden"]:
        r = row(bench, "speedup_vs_cpu")
        if r and r.verdict == "regression":
            flags.append(
                f"{bench}: {r.pre:.2f}x → {r.post:.2f}x "
                f"(Δ {r.delta_pct*100:+.1f}%) — this path was NOT modified in "
                f"Phases 1–7; investigate shared plumbing (cudarc, GpuDevice, driver)"
            )

    return flags


# ----------------------------------------------------------------------
# Report emission
# ----------------------------------------------------------------------

def format_row(r: Row) -> list[str]:
    def fmt(v: float | None, unit: str) -> str:
        if v is None:
            return "—"
        if unit == "x":
            return f"{v:.2f}x"
        if unit == "s":
            return f"{v:.3f}s"
        return f"{v:.3f}"

    delta = (
        f"{r.delta_pct * 100:+.1f}%"
        if r.delta_pct is not None
        else "—"
    )
    verdict_glyph = {
        "ok": "✓",
        "regression": "✗",
        "improvement": "▲",
        "new": "+",
        "disappeared": "−",
    }.get(r.verdict, "?")
    return [
        r.source.replace(".json", ""),
        r.benchmark,
        r.dataset,
        r.metric,
        fmt(r.pre, r.unit),
        fmt(r.post, r.unit),
        delta,
        f"{verdict_glyph} {r.verdict}",
        r.note,
    ]


def write_report(report: DiffReport, flags: list[str], out_path: Path) -> None:
    lines: list[str] = [
        "# GPU accelerator regression report — Phase 8.6",
        "",
        f"- **PRE**:  `{report.pre_path}`",
        f"- **POST**: `{report.post_path}`",
        f"- **Tolerance**: ±{report.tolerance * 100:.1f}%"
        + (" (strict)" if report.strict else ""),
        f"- **Rows**: {len(report.rows)} total — "
        f"{len(report.regressions)} regression, "
        f"{len(report.improvements)} improvement, "
        f"{len(report.new_entries)} new, "
        f"{len(report.disappeared)} disappeared",
        "",
    ]

    if flags:
        lines += ["## Hard-floor flags (from §8.6 threshold table)", ""]
        lines += [f"- ⚠ {msg}" for msg in flags]
        lines.append("")

    if report.regressions:
        lines += [
            "## Regressions", "",
            "| file | benchmark | dataset | metric | pre | post | Δ | verdict | note |",
            "|------|-----------|---------|--------|-----|------|---|---------|------|",
        ]
        for r in report.regressions:
            lines.append("| " + " | ".join(format_row(r)) + " |")
        lines.append("")

    if report.improvements:
        lines += [
            "## Improvements", "",
            "| file | benchmark | dataset | metric | pre | post | Δ | verdict | note |",
            "|------|-----------|---------|--------|-----|------|---|---------|------|",
        ]
        for r in report.improvements:
            lines.append("| " + " | ".join(format_row(r)) + " |")
        lines.append("")

    if report.new_entries:
        lines += [
            "## New benchmarks (post only)", "",
            "| file | benchmark | dataset | metric | post | note |",
            "|------|-----------|---------|--------|------|------|",
        ]
        for r in report.new_entries:
            cells = format_row(r)
            # Drop pre / delta / verdict columns.
            lines.append(
                "| " + " | ".join([cells[0], cells[1], cells[2], cells[3], cells[5], cells[8]]) + " |"
            )
        lines.append("")

    if report.disappeared:
        lines += [
            "## Disappeared (pre only — possibly renamed or removed)", "",
            "| file | benchmark | dataset | metric | pre | note |",
            "|------|-----------|---------|--------|-----|------|",
        ]
        for r in report.disappeared:
            cells = format_row(r)
            lines.append(
                "| " + " | ".join([cells[0], cells[1], cells[2], cells[3], cells[4], cells[8]]) + " |"
            )
        lines.append("")

    # Always emit the full table too, for operators who want everything.
    lines += [
        "## All rows", "",
        "| file | benchmark | dataset | metric | pre | post | Δ | verdict | note |",
        "|------|-----------|---------|--------|-----|------|---|---------|------|",
    ]
    for r in sorted(report.rows, key=lambda r: (r.source, r.benchmark, r.dataset, r.metric)):
        lines.append("| " + " | ".join(format_row(r)) + " |")

    out_path.write_text("\n".join(lines))
    print(f"Wrote report: {out_path}")


def write_json(report: DiffReport, flags: list[str], out_path: Path) -> None:
    data = {
        "pre": report.pre_path,
        "post": report.post_path,
        "tolerance": report.tolerance,
        "strict": report.strict,
        "counts": {
            "total": len(report.rows),
            "regressions": len(report.regressions),
            "improvements": len(report.improvements),
            "new": len(report.new_entries),
            "disappeared": len(report.disappeared),
        },
        "hard_floor_flags": flags,
        "rows": [
            {
                "source": r.source,
                "benchmark": r.benchmark,
                "dataset": r.dataset,
                "metric": r.metric,
                "pre": r.pre,
                "post": r.post,
                "unit": r.unit,
                "higher_is_better": r.higher_is_better,
                "delta_pct": r.delta_pct,
                "verdict": r.verdict,
                "note": r.note,
            }
            for r in report.rows
        ],
    }
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote JSON verdict: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre", required=True, type=Path,
                    help="directory of pre-change gpu_*.json files")
    ap.add_argument("--post", required=True, type=Path,
                    help="directory of post-change gpu_*.json files")
    ap.add_argument("--report", type=Path, default=None,
                    help="Markdown report output path (default: "
                         "<post>/phases_1_7_gpu_regression_report.md)")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="machine-readable JSON verdict (default: "
                         "<post>/phases_1_7_gpu_regression_report.json)")
    ap.add_argument("--timing-tolerance", type=float, default=0.10,
                    help="fraction of pre median allowed to slip before "
                         "flagging (default: 0.10 = ±10%%)")
    ap.add_argument("--strict", action="store_true",
                    help="halve all tolerances")
    args = ap.parse_args()

    if not args.pre.is_dir():
        print(f"error: --pre {args.pre} is not a directory", file=sys.stderr)
        return 2
    if not args.post.is_dir():
        print(f"error: --post {args.post} is not a directory", file=sys.stderr)
        return 2

    tolerance = args.timing_tolerance / (2.0 if args.strict else 1.0)

    rows: list[Row] = []
    rows += diff_pipeline_timing(args.pre, args.post)
    rows += diff_list_of_benchmarks(
        args.pre, args.post, "gpu_pca_timing.json",
        [("gpu_median_s", "gpu_median_s", False), ("speedup", "speedup_vs_cpu", True)],
    )
    rows += diff_list_of_benchmarks(
        args.pre, args.post, "gpu_knn_timing.json",
        [("gpu_median_s", "gpu_median_s", False), ("speedup", "speedup_vs_cpu", True)],
    )
    rows += diff_list_of_benchmarks(
        args.pre, args.post, "gpu_umap_timing.json",
        [("gpu_median_s", "gpu_median_s", False), ("speedup", "speedup_vs_cpu", True)],
    )
    rows += diff_list_of_benchmarks(
        args.pre, args.post, "gpu_preprocess_timing.json",
        [("scx_cpu_median_s", "scx_cpu_median_s", False),
         ("speedup_vs_scanpy", "speedup_vs_scanpy", True)],
    )
    rows += diff_preprocess_device(args.pre, args.post)

    if not rows:
        print("error: no benchmark files found in either directory", file=sys.stderr)
        return 2

    for r in rows:
        r.verdict = _classify(r, tolerance)

    report = DiffReport(
        rows=rows, pre_path=str(args.pre), post_path=str(args.post),
        tolerance=tolerance, strict=args.strict,
    )
    flags = apply_hard_floors(report)

    default_report = args.post / "phases_1_7_gpu_regression_report.md"
    default_json = args.post / "phases_1_7_gpu_regression_report.json"
    write_report(report, flags, args.report or default_report)
    write_json(report, flags, args.json_out or default_json)

    # Console summary.
    print()
    print(f"pre:  {args.pre}")
    print(f"post: {args.post}")
    print(f"rows: {len(rows)}, regressions: {len(report.regressions)}, "
          f"improvements: {len(report.improvements)}, "
          f"new: {len(report.new_entries)}, disappeared: {len(report.disappeared)}")
    if flags:
        print("hard-floor flags:")
        for f in flags:
            print(f"  ⚠ {f}")

    if report.regressions or flags:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
