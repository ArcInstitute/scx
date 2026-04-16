#!/usr/bin/env python3
"""
SCX Comprehensive Benchmark Suite — Main Orchestrator.

Runs selected benchmarks on selected datasets and formats, collecting
structured JSON results. Designed to be run directly or submitted to
SLURM via run_slurm.sh.

Usage:
    # Run all benchmarks on all available datasets (primary formats only)
    python benchmarks/comprehensive/scripts/run_all.py

    # Smoke test on pbmc3k only
    python benchmarks/comprehensive/scripts/run_all.py --smoke

    # Select specific benchmarks
    python benchmarks/comprehensive/scripts/run_all.py --benchmarks compression read_full

    # Select specific datasets
    python benchmarks/comprehensive/scripts/run_all.py --datasets pbmc3k census_1m

    # Select specific formats
    python benchmarks/comprehensive/scripts/run_all.py --formats scx_auto h5ad_gzip

    # Include additional formats (BPCells, Parquet, AnnData-on-Zarr)
    python benchmarks/comprehensive/scripts/run_all.py --include-additional

    # Cold-cache reads (requires root for drop_caches)
    python benchmarks/comprehensive/scripts/run_all.py --cold-cache

    # Dry run — show what would be executed
    python benchmarks/comprehensive/scripts/run_all.py --dry-run

    # List available benchmarks, datasets, and formats
    python benchmarks/comprehensive/scripts/run_all.py --list
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    ALL_FORMATS,
    DATASETS,
    PRIMARY_FORMATS,
    RAW_RESULTS_DIR,
    FormatVariant,
    n_runs_for_dataset,
)
from benchmarks.comprehensive.results import BenchmarkResult, write_result
from benchmarks.comprehensive.sysinfo import collect_system_info


# ---------------------------------------------------------------------------
# Available benchmarks
# ---------------------------------------------------------------------------

AVAILABLE_BENCHMARKS = [
    "compression",         # §3.1 — Storage efficiency
    "write",               # §3.2 — Write / conversion performance
    "read_full",           # §3.3 — Full file load
    "read_selective",      # §3.4 — Selective read (query / subsetting)
    "parallel_scaling",    # §3.5.1 — Parallel read scaling
    "parallel_write_scaling",  # §3.5.2 — Parallel write scaling
    "ml_loader",           # §3.6 — ML data loader throughput
    "memory",              # §3.7 — Memory efficiency
    "append_update",       # §3.8 — Append / update (SCX-specific)
    "backed_mode",         # §3.9 — Backed mode performance
    "accelerators",        # §3.10 — Analysis accelerators (PCA, kNN, UMAP, DE)
    "gpu_accelerators",    # §3.11 — GPU accelerators
    "streaming_preproc",   # §3.12 — Streaming preprocessing pipeline
    "lazy_preproc",        # §3.13 — Lazy preprocessing & column-projected agg
    "correctness",         # §3.14 — Correctness validation suite
    "cell_eval_parity_perf",  # §3.15 — cell-eval / arc-bench parity perf
]

# Benchmarks appropriate for smoke testing
SMOKE_BENCHMARKS = ["compression", "read_full"]
SMOKE_DATASETS = ["pbmc3k"]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def discover_available_datasets() -> list[str]:
    """Return names of datasets that are ready to benchmark.

    Non-synthetic datasets must have their h5ad file present on disk.
    Synthetic datasets are always considered available — they're generated
    on demand by the benchmark module that uses them.
    """
    available = []
    for name, cfg in DATASETS.items():
        if cfg.synthetic or cfg.h5ad_path.exists():
            available.append(name)
    return available


def get_formats_by_keys(keys: list[str] | None) -> list[FormatVariant]:
    """Filter format variants by their keys. None = all primary formats."""
    if keys is None:
        return list(PRIMARY_FORMATS)
    all_by_key = {f.key: f for f in ALL_FORMATS}
    result = []
    for k in keys:
        if k in all_by_key:
            result.append(all_by_key[k])
        else:
            print(f"WARNING: Unknown format key '{k}', skipping.")
    return result


def run_benchmark(
    benchmark_name: str,
    dataset_name: str,
    formats: list[FormatVariant],
    cold_cache: bool = False,
    dry_run: bool = False,
) -> list[BenchmarkResult]:
    """Run a single benchmark type on a single dataset across all formats.

    Returns a list of BenchmarkResult objects.
    """
    cfg = DATASETS.get(dataset_name)
    if cfg is None:
        print(f"  SKIP: Unknown dataset '{dataset_name}'")
        return []

    # Synthetic datasets are materialized on demand by the benchmark module
    # (see benchmarks/comprehensive/benchmarks/_pert_synth.py) — skip the
    # existence gate so the benchmark gets a chance to generate the data.
    if not cfg.synthetic and not cfg.h5ad_path.exists():
        print(f"  SKIP: {dataset_name} h5ad not found at {cfg.h5ad_path}")
        return []

    n_runs = n_runs_for_dataset(dataset_name)

    if dry_run:
        print(f"  [DRY RUN] Would run '{benchmark_name}' on '{dataset_name}' "
              f"with {len(formats)} formats, {n_runs} runs each")
        return []

    results: list[BenchmarkResult] = []

    # Import the benchmark module dynamically
    try:
        benchmark_mod = _import_benchmark(benchmark_name)
    except ImportError as e:
        print(f"  SKIP: Benchmark module '{benchmark_name}' not yet implemented: {e}")
        return []

    if not hasattr(benchmark_mod, "run"):
        print(f"  SKIP: Benchmark module '{benchmark_name}' has no run() function")
        return []

    for fmt in formats:
        print(f"    Format: {fmt.name} ({fmt.key})")
        try:
            result = benchmark_mod.run(
                dataset=cfg,
                format_variant=fmt,
                n_runs=n_runs,
                cold_cache=cold_cache,
            )
            if result is not None:
                write_result(result)
                results.append(result)
                median = result.median_wall_s
                if median is not None:
                    print(f"      → {median:.3f}s (median of {n_runs} runs)")
                else:
                    print(f"      → completed ({len(result.runs)} runs)")
        except Exception as e:
            print(f"      ERROR: {e}")

    return results


def _import_benchmark(name: str):
    """Dynamically import a benchmark module."""
    import importlib
    return importlib.import_module(f"benchmarks.comprehensive.benchmarks.{name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCX Comprehensive Benchmark Suite — Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--benchmarks", nargs="+", metavar="NAME",
        help=f"Benchmark(s) to run. Available: {', '.join(AVAILABLE_BENCHMARKS)}",
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME",
        help="Dataset(s) to benchmark. Default: all available on disk.",
    )
    parser.add_argument(
        "--formats", nargs="+", metavar="KEY",
        help="Format variant key(s) to benchmark. Default: all primary formats.",
    )
    parser.add_argument(
        "--include-additional", action="store_true",
        help="Include additional formats (BPCells, Parquet, AnnData-on-Zarr).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=f"Quick smoke test: {SMOKE_BENCHMARKS} on {SMOKE_DATASETS}",
    )
    parser.add_argument(
        "--cold-cache", action="store_true",
        help="Drop OS caches between runs (requires root).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be run without executing.",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_available",
        help="List available benchmarks, datasets, and formats.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Override output directory (default: {RAW_RESULTS_DIR})",
    )

    return parser.parse_args()


def list_available() -> None:
    """Print available benchmarks, datasets, and formats."""
    print("=== Available Benchmarks ===")
    for b in AVAILABLE_BENCHMARKS:
        print(f"  - {b}")

    print("\n=== Datasets ===")
    available = discover_available_datasets()
    for name, cfg in DATASETS.items():
        status = "✅" if name in available else "🔲"
        print(f"  {status} {cfg.id}: {name:25s} ({cfg.n_obs:>12,} cells × {cfg.n_vars:>6,} genes)")

    print("\n=== Primary Formats ===")
    for f in PRIMARY_FORMATS:
        print(f"  - {f.key:20s} → {f.name}")

    print("\n=== Additional Formats ===")
    for f in ALL_FORMATS:
        if f.category == "additional":
            print(f"  - {f.key:20s} → {f.name}")


def main() -> None:
    args = parse_args()

    if args.list_available:
        list_available()
        return

    # Resolve benchmarks
    if args.smoke:
        benchmarks = SMOKE_BENCHMARKS
        datasets = SMOKE_DATASETS
    else:
        benchmarks = args.benchmarks or AVAILABLE_BENCHMARKS
        datasets = args.datasets or discover_available_datasets()

    # Resolve formats
    if args.formats:
        formats = get_formats_by_keys(args.formats)
    elif args.include_additional:
        formats = list(ALL_FORMATS)
    else:
        formats = list(PRIMARY_FORMATS)

    # Header
    sysinfo = collect_system_info()
    now = datetime.datetime.now().isoformat(timespec="seconds")

    print("=" * 70)
    print("SCX Comprehensive Benchmark Suite")
    print("=" * 70)
    print(f"  Timestamp:  {now}")
    print(f"  Host:       {sysinfo.get('hostname', 'unknown')}")
    print(f"  CPU:        {sysinfo.get('cpu', 'unknown')}")
    print(f"  RAM:        {sysinfo.get('ram_gb', '?')} GB")
    print(f"  Benchmarks: {benchmarks}")
    print(f"  Datasets:   {datasets}")
    print(f"  Formats:    {[f.key for f in formats]}")
    print(f"  Cold cache: {args.cold_cache}")
    print(f"  Dry run:    {args.dry_run}")
    print("=" * 70)
    print()

    # Save system info
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sysinfo_path = RAW_RESULTS_DIR / "system_info.json"
    with open(sysinfo_path, "w") as f:
        json.dump(sysinfo, f, indent=2, default=str)

    # Run benchmarks
    all_results: list[BenchmarkResult] = []
    total_start = time.perf_counter()

    for bench_name in benchmarks:
        print(f"\n{'─' * 50}")
        print(f"  Benchmark: {bench_name}")
        print(f"{'─' * 50}")

        for ds_name in datasets:
            print(f"\n  Dataset: {ds_name}")
            results = run_benchmark(
                benchmark_name=bench_name,
                dataset_name=ds_name,
                formats=formats,
                cold_cache=args.cold_cache,
                dry_run=args.dry_run,
            )
            all_results.extend(results)

    total_elapsed = time.perf_counter() - total_start

    # Summary
    print()
    print("=" * 70)
    print(f"  Completed: {len(all_results)} result(s) in {total_elapsed:.1f}s")
    print(f"  Results:   {RAW_RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
