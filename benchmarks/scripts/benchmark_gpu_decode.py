#!/usr/bin/env python3
"""
SCX GPU Decode Benchmarks

Runs GPU kernel microbenchmarks (Rust binary) and generates a markdown report.

Usage:
    # Run all microbenchmarks, generate default report
    python benchmarks/scripts/benchmark_gpu_decode.py

    # Custom output path
    python benchmarks/scripts/benchmark_gpu_decode.py --output benchmarks/results/gpu_benchmark.md

    # With dataset for future end-to-end benchmark (Phase H)
    python benchmarks/scripts/benchmark_gpu_decode.py --dataset census_1m
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
from bench_env import WORK_DIR

DATASETS = {
    "pbmc3k": {"scx": WORK_DIR / "pbmc3k.scx", "cells": 2_700, "genes": 32_738},
    "tabula_sapiens_100k": {"scx": WORK_DIR / "tabula_sapiens_100k.scx", "cells": 100_000, "genes": 60_000},
    "census_1m": {"scx": WORK_DIR / "census_1m.scx", "cells": 1_000_000, "genes": 61_497},
    "census_10m_blood": {"scx": WORK_DIR / "census_10m_blood.scx", "cells": 10_000_000, "genes": 60_000},
}


def run_rust_microbenchmarks() -> list[dict]:
    """Build and run the gpu_bench binary, parse JSON lines output."""
    print("Building scx-gpu in release mode...")
    build = subprocess.run(
        ["cargo", "build", "--release", "-p", "scx-gpu", "--bin", "gpu_bench"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        print(f"ERROR: cargo build failed:\n{build.stderr}", file=sys.stderr)
        sys.exit(1)
    print("  Build complete.")

    print("Running GPU microbenchmarks...")
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "scx-gpu", "--bin", "gpu_bench"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Print stderr (progress) to console
    if result.stderr:
        print(result.stderr, end="")

    if result.returncode != 0:
        print(f"ERROR: gpu_bench failed (exit code {result.returncode})", file=sys.stderr)
        sys.exit(1)

    # Parse JSON lines from stdout
    results = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"WARNING: skipping non-JSON line: {line}", file=sys.stderr)
    return results


def generate_report(results: list[dict], dataset: str | None = None) -> str:
    """Generate markdown report from benchmark results."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# SCX GPU Decode Benchmarks",
        "",
        f"**Generated**: {ts}",
        "",
    ]

    # --- Benchmark #1a: Rice ---
    rice = [r for r in results if r["benchmark"] == "rice_decode"]
    if rice:
        lines += [
            "## Benchmark #1: GPU vs CPU Decode Throughput",
            "",
            "### Rice Decode",
            "",
            "| N Values | CPU (μs) | GPU (μs) | Speedup |",
            "|----------|----------|----------|---------|",
        ]
        for r in rice:
            nv = f'{r["n_values"]:,}'
            lines.append(
                f'| {nv} | {r["cpu_median_us"]:.1f} | {r["gpu_median_us"]:.1f} | {r["speedup"]:.1f}x |'
            )
        lines.append("")

    # --- Benchmark #1b: FOR-BP ---
    forbp = [r for r in results if r["benchmark"] == "forbp_decode"]
    if forbp:
        lines += [
            "### FOR-BP Decode",
            "",
            "| Config | N Rows | NNZ | CPU (μs) | GPU (μs) | Speedup |",
            "|--------|--------|-----|----------|----------|---------|",
        ]
        for r in forbp:
            lines.append(
                f'| {r["label"]} | {r["n_rows"]:,} | {r["nnz"]:,} '
                f'| {r["cpu_median_us"]:.1f} | {r["gpu_median_us"]:.1f} | {r["speedup"]:.1f}x |'
            )
        lines.append("")

    # --- Benchmark #1c: Full shard ---
    shard = [r for r in results if r["benchmark"] == "shard_decode"]
    if shard:
        lines += [
            "### Full Shard Decode (Scx1)",
            "",
            "| Config | N Rows | NNZ | CPU (μs) | GPU (μs) | Speedup |",
            "|--------|--------|-----|----------|----------|---------|",
        ]
        for r in shard:
            lines.append(
                f'| {r["label"]} | {r["n_rows"]:,} | {r["nnz"]:,} '
                f'| {r["cpu_median_us"]:.1f} | {r["gpu_median_us"]:.1f} | {r["speedup"]:.1f}x |'
            )
        lines.append("")

    # --- Benchmark #3: End-to-end (DEFERRED) ---
    lines += [
        "## Benchmark #3: End-to-End Training Pipeline",
        "",
        "**Status**: DEFERRED — Phase H (loader integration) not yet complete.",
        "**CPU baseline**: 397.3 batches/sec (Phase 2, census_1m on H100).",
        "",
        "When Phase H is implemented, this benchmark will measure GPU-enabled",
        "`TrainingDataset` throughput against the CPU baseline.",
        "",
    ]

    # --- Benchmark #4: Sparse-to-dense ---
    s2d = [r for r in results if r["benchmark"] == "sparse_to_dense"]
    s2d_hvg = [r for r in results if r["benchmark"] == "sparse_to_dense_hvg"]
    if s2d:
        lines += [
            "## Benchmark #4: GPU vs CPU Sparse-to-Dense",
            "",
            "| Config | Rows | Cols | CPU (μs) | GPU (μs) | Speedup |",
            "|--------|------|------|----------|----------|---------|",
        ]
        for r in s2d:
            lines.append(
                f'| {r["label"]} | {r["n_rows"]:,} | {r["n_cols"]:,} '
                f'| {r["cpu_median_us"]:.1f} | {r["gpu_median_us"]:.1f} | {r["speedup"]:.1f}x |'
            )
        lines.append("")

    if s2d_hvg:
        lines += [
            "### With HVG Projection",
            "",
            "| Config | Rows | Output Cols | CPU (μs) | GPU (μs) | Speedup |",
            "|--------|------|-------------|----------|----------|---------|",
        ]
        for r in s2d_hvg:
            lines.append(
                f'| {r["label"]} | {r["n_rows"]:,} | {r["n_cols"]:,} '
                f'| {r["cpu_median_us"]:.1f} | {r["gpu_median_us"]:.1f} | {r["speedup"]:.1f}x |'
            )
        lines.append("")

    # --- Benchmark #5: Multi-shard ---
    multi = [r for r in results if r["benchmark"] == "multi_shard_decode"]
    if multi:
        lines += [
            "## Benchmark #5: Multi-Shard GPU Decode Scaling",
            "",
            "Single shard = 2048 rows, ~500 nnz/row, 30K cols, Scx1 codec.",
            "All shards decoded sequentially on the default CUDA stream.",
            "",
            "| Shards | GPU Total (μs) | Per-Shard (μs) | Throughput (shards/s) | Speedup vs CPU |",
            "|--------|----------------|----------------|----------------------|----------------|",
        ]
        for r in multi:
            per_shard = r["gpu_median_us"] / r["n_shards"]
            throughput = 1e6 / per_shard if per_shard > 0 else 0
            lines.append(
                f'| {r["n_shards"]} | {r["gpu_median_us"]:.0f} '
                f"| {per_shard:.0f} | {throughput:.0f} "
                f'| {r["speedup"]:.1f}x |'
            )
        lines.append("")

    # --- Summary ---
    lines += [
        "## Summary",
        "",
        "| Benchmark | Target | Result | Pass? |",
        "|-----------|--------|--------|-------|",
    ]

    # Check rice/forbp for >10K rows having >=10x speedup
    large_decode = [r for r in results if r["benchmark"] in ("rice_decode", "forbp_decode", "shard_decode")]
    large_speedups = []
    for r in large_decode:
        n = r.get("n_values") or r.get("n_rows") or 0
        if n >= 10_000:
            large_speedups.append(r["speedup"])

    if large_speedups:
        min_speedup = min(large_speedups)
        passed = min_speedup >= 10.0
        lines.append(
            f"| GPU decode (>10K elements) | >= 10x | {min_speedup:.1f}x min "
            f'| {"PASS" if passed else "FAIL"} |'
        )
    else:
        lines.append("| GPU decode (>10K elements) | >= 10x | No data | -- |")

    lines.append("| End-to-end training | > 397 b/s | DEFERRED | -- |")

    if s2d:
        max_s2d = max(r["speedup"] for r in s2d)
        lines.append(f"| Sparse-to-dense | Report | {max_s2d:.1f}x max | -- |")

    if multi:
        max_multi = max(r["speedup"] for r in multi)
        lines.append(f"| Multi-shard scaling | Report | {max_multi:.1f}x max | -- |")

    lines.append("| GDS Go/No-Go | >2x CPU | DEFERRED (nvidia-fs) | -- |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="SCX GPU Decode Benchmarks")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(DATASETS.keys()),
        help="Dataset for end-to-end benchmark (Phase H, currently deferred)",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=Path,
        help="Output markdown path (default: benchmarks/results/gpu_benchmark.md)",
    )
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (no effect yet)")
    parser.add_argument(
        "--from-json",
        default=None,
        type=Path,
        help="Generate report from pre-captured JSON lines file (skip build+run)",
    )
    args = parser.parse_args()

    output_path = args.output or (RESULTS_DIR / "gpu_benchmark.md")

    # 1. Get benchmark results
    if args.from_json:
        results = []
        for line in args.from_json.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    else:
        results = run_rust_microbenchmarks()
    if not results:
        print("ERROR: No benchmark results returned.", file=sys.stderr)
        sys.exit(1)

    # 2. Generate markdown report
    report = generate_report(results, args.dataset)

    # 3. Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    # 4. Save raw JSON
    json_path = output_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "micro_benchmarks": results,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
        )
    )

    print(f"\nReport: {output_path}")
    print(f"JSON:   {json_path}")


if __name__ == "__main__":
    main()
