#!/usr/bin/env python3
"""Generate Phase 3G final benchmark comparison report.

Compares post-Sprint-3 results against the pre-Sprint-1 baseline.
Outputs Markdown tables showing cumulative improvement across all three sprints.

Usage:
    .venv/bin/python benchmarks/scripts/generate_phase3g_report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

# Directories
RESULTS_DIR = Path("benchmarks/comprehensive/results/raw")
BASELINE_DIR = RESULTS_DIR / "baseline_pre_sprint1"
OUTPUT_PATH = Path("benchmarks/results/phase3g_final_report.md")

# Benchmarks and formats to compare
BENCHMARKS = ["compression", "write", "read_full", "read_selective", "memory"]
SCX_FORMATS = ["scx_auto", "scx_scx1", "scx_zstd", "scx_lz4", "scx_pcodec", "scx_none"]
COMPETING_FORMATS = ["h5ad_none", "h5ad_gzip", "h5ad_lzf", "zarr_zstd", "zarr_lz4", "tiledb_soma"]
ALL_FORMATS = SCX_FORMATS + COMPETING_FORMATS
DATASETS = ["pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k", "census_500k", "census_1m"]


def load_results(directory: Path) -> dict[tuple[str, str, str], dict]:
    """Load all JSON results into a dict keyed by (benchmark, format, dataset)."""
    results = {}
    if not directory.exists():
        return results
    for path in directory.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            key = (data.get("benchmark", ""), data.get("format", ""), data.get("dataset", ""))
            if all(key):
                results[key] = data
        except (json.JSONDecodeError, OSError):
            continue
    return results


def get_metric(result: dict, benchmark: str) -> float | None:
    """Extract the primary metric for a given benchmark type."""
    if benchmark == "compression":
        size = result.get("file_size_bytes")
        return size / 1e6 if size else None  # MB
    elif benchmark in ("read_full", "read_selective", "write"):
        return result.get("median_wall_s")
    elif benchmark == "memory":
        runs = result.get("runs", [])
        if runs:
            rss_vals = [r.get("peak_rss_mb", 0) for r in runs if r.get("peak_rss_mb", 0) > 0]
            return statistics.median(rss_vals) if rss_vals else None
        return None
    return None


def metric_unit(benchmark: str) -> str:
    if benchmark == "compression":
        return "MB"
    elif benchmark in ("read_full", "read_selective", "write"):
        return "sec"
    elif benchmark == "memory":
        return "MB RSS"
    return ""


def pct_change(baseline: float, current: float) -> str:
    """Format percentage change (negative = improvement for time/size)."""
    if baseline == 0:
        return "N/A"
    change = (current - baseline) / baseline * 100
    if change < 0:
        return f"**{change:+.1f}%**"  # Bold for improvements
    return f"{change:+.1f}%"


def generate_report(baseline: dict, current: dict) -> str:
    lines = []
    lines.append("# Phase 3G: Final Sprint 3 Benchmark Comparison Report\n")
    lines.append(f"Generated from {len(current)} current results vs {len(baseline)} baseline results.\n")

    # --- SCX codec comparison (current results only) ---
    lines.append("## SCX Codec Comparison (Post-Sprint-3)\n")
    for bench in BENCHMARKS:
        unit = metric_unit(bench)
        lines.append(f"### {bench.replace('_', ' ').title()} ({unit})\n")
        header = "| Dataset |"
        separator = "|---------|"
        for fmt in SCX_FORMATS:
            short = fmt.replace("scx_", "")
            header += f" {short} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        for ds in DATASETS:
            row = f"| {ds} |"
            for fmt in SCX_FORMATS:
                val = get_metric(current.get((bench, fmt, ds), {}), bench)
                row += f" {val:.3f} |" if val is not None else " — |"
            lines.append(row)
        lines.append("")

    # --- Sprint 3 vs Pre-Sprint-1 Baseline (SCX formats) ---
    lines.append("## Cumulative Improvement: Post-Sprint-3 vs Pre-Sprint-1 Baseline\n")
    for bench in BENCHMARKS:
        unit = metric_unit(bench)
        lines.append(f"### {bench.replace('_', ' ').title()} ({unit})\n")
        header = "| Dataset | Format | Baseline | Current | Change |"
        separator = "|---------|--------|----------|---------|--------|"
        lines.append(header)
        lines.append(separator)

        for ds in DATASETS:
            for fmt in ["scx_auto", "scx_scx1", "scx_zstd"]:
                bkey = (bench, fmt, ds)
                ckey = (bench, fmt, ds)
                bval = get_metric(baseline.get(bkey, {}), bench)
                cval = get_metric(current.get(ckey, {}), bench)
                if bval is not None and cval is not None:
                    change = pct_change(bval, cval)
                    lines.append(f"| {ds} | {fmt} | {bval:.3f} | {cval:.3f} | {change} |")
        lines.append("")

    # --- SCX vs Competing Formats ---
    lines.append("## SCX (auto) vs Competing Formats\n")
    for bench in ["compression", "read_full", "write"]:
        unit = metric_unit(bench)
        lines.append(f"### {bench.replace('_', ' ').title()} ({unit})\n")
        header = "| Dataset |"
        separator = "|---------|"
        formats_to_show = ["scx_auto", "h5ad_gzip", "h5ad_lzf", "zarr_zstd", "tiledb_soma"]
        for fmt in formats_to_show:
            header += f" {fmt} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)

        for ds in DATASETS:
            row = f"| {ds} |"
            for fmt in formats_to_show:
                val = get_metric(current.get((bench, fmt, ds), {}), bench)
                row += f" {val:.3f} |" if val is not None else " — |"
            lines.append(row)
        lines.append("")

    # --- New Sprint 3 codecs (LZ4, Pcodec) ---
    lines.append("## Sprint 3 New Codecs: LZ4Shuffle and Pcodec\n")
    for bench in ["compression", "read_full", "write"]:
        unit = metric_unit(bench)
        lines.append(f"### {bench.replace('_', ' ').title()} ({unit})\n")
        header = "| Dataset | scx_auto | scx_lz4 | scx_pcodec | scx_zstd |"
        separator = "|---------|----------|---------|------------|----------|"
        lines.append(header)
        lines.append(separator)

        for ds in DATASETS:
            row = f"| {ds} |"
            for fmt in ["scx_auto", "scx_lz4", "scx_pcodec", "scx_zstd"]:
                val = get_metric(current.get((bench, fmt, ds), {}), bench)
                row += f" {val:.3f} |" if val is not None else " — |"
            lines.append(row)
        lines.append("")

    # --- Summary statistics ---
    lines.append("## Summary\n")
    improvements = []
    for bench in ["read_full", "write"]:
        for ds in DATASETS:
            for fmt in ["scx_auto", "scx_scx1", "scx_zstd"]:
                bval = get_metric(baseline.get((bench, fmt, ds), {}), bench)
                cval = get_metric(current.get((bench, fmt, ds), {}), bench)
                if bval and cval and bval > 0:
                    improvements.append((cval - bval) / bval * 100)

    if improvements:
        median_imp = statistics.median(improvements)
        lines.append(f"- **Median read/write improvement** across SCX formats and datasets: **{median_imp:.1f}%**")
        best = min(improvements)
        lines.append(f"- **Best improvement**: {best:.1f}%")
        worst = max(improvements)
        lines.append(f"- **Worst**: {worst:+.1f}%")
    else:
        lines.append("- No comparable baseline results found for read/write benchmarks.")

    lines.append("")
    return "\n".join(lines)


def main():
    print(f"Loading baseline results from {BASELINE_DIR}...")
    baseline = load_results(BASELINE_DIR)
    print(f"  Loaded {len(baseline)} baseline results")

    print(f"Loading current results from {RESULTS_DIR}...")
    current = load_results(RESULTS_DIR)
    print(f"  Loaded {len(current)} current results")

    report = generate_report(baseline, current)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report)
    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
