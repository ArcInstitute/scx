#!/usr/bin/env python3
"""
MADV_DONTNEED RSS impact benchmark.

Measures peak RSS during streaming aggregation (row_sums) on large SCX files
by spawning worker subprocesses with clean address spaces.

Runs N_RUNS iterations and reports median peak RSS and RSS time series summary.

Usage:
    python benchmarks/scripts/benchmark_madvise_rss.py [--dataset census_1m] [--runs 3] [--operation row_sums]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKER = Path(__file__).parent / "benchmark_madvise_rss_worker.py"
from bench_env import WORK_DIR
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python")


def _median(values):
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_worker(scx_path, operation):
    """Run worker subprocess and parse JSON result."""
    result = subprocess.run(
        [PYTHON, str(WORKER), str(scx_path), operation],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  Worker failed: {result.stderr}", file=sys.stderr)
        return None
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="MADV_DONTNEED RSS benchmark")
    parser.add_argument("--dataset", default="census_1m")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--operation", default="row_sums", choices=["row_sums", "col_sums"])
    parser.add_argument("--label", default="", help="Label for this run (e.g., 'with_dontneed')")
    args = parser.parse_args()

    scx_path = WORK_DIR / f"{args.dataset}.scx"
    if not scx_path.exists():
        print(f"ERROR: {scx_path} not found", file=sys.stderr)
        sys.exit(1)

    scx_size_mb = scx_path.stat().st_size / (1024 * 1024)
    label = args.label or "default"

    print(f"=== MADV_DONTNEED RSS Benchmark ===")
    print(f"  Dataset:   {args.dataset} ({scx_size_mb:.0f} MB)")
    print(f"  Operation: {args.operation}")
    print(f"  Runs:      {args.runs}")
    print(f"  Label:     {label}")
    print()

    results = []
    for i in range(args.runs):
        print(f"  Run {i+1}/{args.runs}...", end=" ", flush=True)
        r = run_worker(scx_path, args.operation)
        if r is None:
            print("FAILED")
            continue
        results.append(r)
        rss_samples = r["rss_samples_mb"]
        rss_min = min(rss_samples) if rss_samples else 0
        rss_max = max(rss_samples) if rss_samples else 0
        print(
            f"peak={r['peak_rss_mb']:.0f} MB, "
            f"sampled_range=[{rss_min:.0f}-{rss_max:.0f}] MB, "
            f"time={r['wall_clock_s']:.1f}s, "
            f"samples={r['n_samples']}"
        )

    if not results:
        print("All runs failed.")
        sys.exit(1)

    peak_rss_values = [r["peak_rss_mb"] for r in results]
    wall_clock_values = [r["wall_clock_s"] for r in results]

    # RSS growth: difference between last and first RSS sample
    growths = []
    for r in results:
        s = r["rss_samples_mb"]
        if len(s) >= 2:
            growths.append(s[-1] - s[0])

    print()
    print(f"  Summary ({args.runs} runs):")
    print(f"    Median peak RSS:  {_median(peak_rss_values):.0f} MB")
    print(f"    Median wall clock: {_median(wall_clock_values):.1f}s")
    if growths:
        print(f"    Median RSS growth: {_median(growths):+.0f} MB (last - first sample)")

    # Save results
    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"madvise_rss_{args.dataset}_{label}.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "scx_size_mb": round(scx_size_mb, 1),
                "operation": args.operation,
                "label": label,
                "runs": [
                    {
                        "peak_rss_mb": r["peak_rss_mb"],
                        "wall_clock_s": r["wall_clock_s"],
                        "rss_samples_mb": r["rss_samples_mb"],
                    }
                    for r in results
                ],
                "median_peak_rss_mb": round(_median(peak_rss_values), 1),
                "median_wall_clock_s": round(_median(wall_clock_values), 3),
            },
            f,
            indent=2,
        )
    print(f"    Results saved: {out_path}")


if __name__ == "__main__":
    main()
