#!/usr/bin/env python3
"""Run all benchmarks and output markdown results table (Task 17.5)."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
sys.path.insert(0, str(PROJECT_ROOT / "pyscx"))

# Import individual benchmark modules
sys.path.insert(0, str(Path(__file__).parent))
import benchmark_compression
import benchmark_read
import benchmark_write


def format_size(bytes_val):
    """Format byte count as human-readable string."""
    if bytes_val is None:
        return "N/A"
    if bytes_val < 1e6:
        return f"{bytes_val / 1e3:.1f} KB"
    if bytes_val < 1e9:
        return f"{bytes_val / 1e6:.1f} MB"
    return f"{bytes_val / 1e9:.2f} GB"


def generate_markdown(compression, read, write):
    """Generate a markdown report from benchmark results."""
    lines = []
    lines.append("# SCX Benchmark Results\n")

    # --- Compression table ---
    lines.append("## Compression Ratios\n")
    lines.append("| Dataset | Cells | Genes | h5ad | SCX | Ratio | Zarr+Zstd | Zarr Ratio |")
    lines.append("|---------|-------|-------|------|-----|-------|-----------|------------|")
    for r in compression:
        zarr_size = format_size(r.get("zarr_zstd_size"))
        zarr_ratio = f"{r['zarr_ratio']:.3f}" if r.get("zarr_ratio") else "N/A"
        lines.append(
            f"| {r['dataset']} | {r.get('n_obs', '?'):,} | {r.get('n_vars', '?'):,} "
            f"| {format_size(r['h5ad_size'])} | {format_size(r['scx_size'])} "
            f"| {r['ratio']:.3f} | {zarr_size} | {zarr_ratio} |"
        )
    lines.append("")

    # --- Read table ---
    lines.append("## Read Performance (warm cache, median of 3)\n")
    lines.append("| Dataset | h5ad (s) | SCX (s) | Speedup | SCX Peak MB | h5ad Peak MB |")
    lines.append("|---------|----------|---------|---------|-------------|--------------|")
    for r in read:
        lines.append(
            f"| {r['dataset']} | {r['h5ad_warm_s']:.3f} | {r['scx_warm_s']:.3f} "
            f"| {r['speedup']:.1f}x | {r.get('scx_peak_mb', 0):.1f} "
            f"| {r.get('h5ad_peak_mb', 0):.1f} |"
        )
    lines.append("")

    # --- Write table ---
    lines.append("## Write Performance (h5ad → SCX, median of 3)\n")
    lines.append("| Dataset | h5ad Size | Time (s) | Throughput (MB/s) |")
    lines.append("|---------|-----------|----------|-------------------|")
    for r in write:
        lines.append(
            f"| {r['dataset']} | {r['h5ad_size_mb']:.1f} MB "
            f"| {r['write_time_s']:.3f} | {r['mb_per_s']:.1f} |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("SCX Benchmark Suite")
    print("=" * 60)
    print()

    print("--- Compression Benchmarks ---")
    compression_results = benchmark_compression.run_all()
    print()

    print("--- Read Benchmarks ---")
    read_results = benchmark_read.run_all()
    print()

    print("--- Write Benchmarks ---")
    write_results = benchmark_write.run_all()
    print()

    if not any([compression_results, read_results, write_results]):
        print("No results to report. Ensure datasets are downloaded.")
        return

    # Generate markdown
    md = generate_markdown(compression_results, read_results, write_results)

    # Print to stdout
    print("=" * 60)
    print(md)

    # Write to file
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_results.md"
    out_path.write_text(md)
    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
