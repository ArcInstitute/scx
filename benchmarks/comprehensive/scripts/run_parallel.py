#!/usr/bin/env python3
"""
SCX Comprehensive Benchmark Suite — Parallel SLURM Launcher via submitit.

Two-phase execution:
  Phase A: Convert each (dataset, format) pair to a persistent path.
  Phase B: Run benchmarks in parallel, reading from pre-converted files.

Each (benchmark, dataset, format) triple is submitted as an independent SLURM
job, enabling massive parallelism across the cluster.

Usage:
    # Full run on small datasets
    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k

    # Large datasets on high-mem partition
    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --datasets census_500k census_1m census_5m \\
        --partition cpu_preemptible --mem-gb 500 --timeout 480

    # Specific benchmarks and formats
    python benchmarks/comprehensive/scripts/run_parallel.py \\
        --benchmarks read_full read_selective \\
        --formats scx_auto zarr_zstd h5ad_gzip \\
        --datasets census_1m

    # Dry run
    python benchmarks/comprehensive/scripts/run_parallel.py --dry-run

    # Skip conversion (use existing pre-converted files)
    python benchmarks/comprehensive/scripts/run_parallel.py --skip-convert

    # Re-convert everything
    python benchmarks/comprehensive/scripts/run_parallel.py --overwrite
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import (  # noqa: E402
    ALL_FORMATS,
    DATASETS,
    PRIMARY_FORMATS,
    FormatVariant,
    n_runs_for_dataset,
)
from benchmarks.comprehensive.convert import convert_dataset_format  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult, write_result  # noqa: E402

logger = logging.getLogger(__name__)

BENCHMARK_NAMES = [
    "compression",
    "write",
    "read_full",
    "read_selective",
    "parallel_scaling",
    "parallel_write_scaling",
    "memory",
]

# Benchmarks that work from h5ad source and don't need pre-converted files.
_NO_CONVERSION = {"write", "parallel_write_scaling"}

LOGS_DIR = PROJECT_ROOT / "benchmarks" / "comprehensive" / "logs" / "submitit"


# ---------------------------------------------------------------------------
# Submitit job callables (must be picklable — defined at module level)
# ---------------------------------------------------------------------------


def _run_conversion(
    dataset_name: str,
    format_key: str,
    format_runner: str,
    format_params: dict,
    overwrite: bool,
) -> str:
    """Convert a single (dataset, format) pair. Returns the output path."""
    from benchmarks.comprehensive.config import DATASETS, FormatVariant
    from benchmarks.comprehensive.convert import convert_dataset_format

    dataset = DATASETS[dataset_name]
    fmt = FormatVariant(
        name=format_key, key=format_key,
        category="primary", runner=format_runner, params=format_params,
    )
    path = convert_dataset_format(dataset, fmt, overwrite=overwrite)
    return str(path)


def _run_benchmark(
    bench_name: str,
    dataset_name: str,
    format_key: str,
    format_runner: str,
    format_params: dict,
    n_runs: int,
    cold_cache: bool,
    converted_path_str: str | None,
) -> dict:
    """Run a single (benchmark, dataset, format) triple. Returns result dict."""
    import importlib
    from pathlib import Path

    from benchmarks.comprehensive.config import DATASETS, FormatVariant
    from benchmarks.comprehensive.results import write_result

    dataset = DATASETS[dataset_name]
    fmt = FormatVariant(
        name=format_key, key=format_key,
        category="primary", runner=format_runner, params=format_params,
    )
    converted_path = Path(converted_path_str) if converted_path_str else None

    mod = importlib.import_module(f"benchmarks.comprehensive.benchmarks.{bench_name}")

    # write.py and parallel_write_scaling.py don't accept converted_path
    if bench_name in _NO_CONVERSION:
        result = mod.run(dataset=dataset, format_variant=fmt,
                         n_runs=n_runs, cold_cache=cold_cache)
    else:
        result = mod.run(dataset=dataset, format_variant=fmt,
                         n_runs=n_runs, cold_cache=cold_cache,
                         converted_path=converted_path)

    write_result(result)
    return result.to_dict()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _slurm_params(args, is_conversion: bool = False) -> dict:
    """Build submitit executor parameters from CLI args."""
    timeout_min = args.timeout
    if is_conversion and timeout_min < 240:
        timeout_min = 240  # conversions need more time for large datasets

    return {
        "slurm_partition": args.partition,
        "cpus_per_task": args.cpus,
        "mem_gb": args.mem_gb,
        "timeout_min": timeout_min,
        "slurm_setup": [
            f"export PATH={PROJECT_ROOT}/.venv/bin:$PATH",
        ],
    }


def main() -> None:
    import submitit

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Resolve datasets
    datasets = args.datasets
    if not datasets:
        datasets = [name for name, cfg in DATASETS.items() if cfg.h5ad_path.exists()]

    # Resolve formats
    if args.formats:
        all_by_key = {f.key: f for f in ALL_FORMATS}
        formats = [all_by_key[k] for k in args.formats if k in all_by_key]
    elif args.include_additional:
        formats = list(ALL_FORMATS)
    else:
        formats = list(PRIMARY_FORMATS)

    # Resolve benchmarks
    benchmarks = args.benchmarks or BENCHMARK_NAMES

    logger.info("Datasets:   %s", datasets)
    logger.info("Formats:    %s", [f.key for f in formats])
    logger.info("Benchmarks: %s", benchmarks)

    # Determine which benchmarks need pre-converted files
    needs_conversion = [b for b in benchmarks if b not in _NO_CONVERSION]
    needs_write = "write" in benchmarks

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase A: Conversion ---
    conversion_paths: dict[tuple[str, str], str] = {}

    if needs_conversion and not args.skip_convert:
        logger.info("=" * 60)
        logger.info("Phase A: Converting datasets to target formats")
        logger.info("=" * 60)

        executor = submitit.AutoExecutor(folder=str(LOGS_DIR / "convert"))
        executor.update_parameters(**_slurm_params(args, is_conversion=True))

        conv_jobs: dict[tuple[str, str], submitit.Job] = {}

        for ds_name in datasets:
            cfg = DATASETS[ds_name]
            for fmt in formats:
                key = (ds_name, fmt.key)
                output_path = cfg.path_for_format(fmt.key)

                if output_path.exists() and not args.overwrite:
                    conversion_paths[key] = str(output_path)
                    logger.info("  Exists: %s / %s -> %s", ds_name, fmt.key, output_path)
                    continue

                if args.dry_run:
                    logger.info("  [DRY RUN] Would convert: %s / %s", ds_name, fmt.key)
                    continue

                job = executor.submit(
                    _run_conversion,
                    ds_name, fmt.key, fmt.runner, fmt.params, args.overwrite,
                )
                conv_jobs[key] = job
                logger.info("  Submitted: %s / %s -> job %s", ds_name, fmt.key, job.job_id)

        # Wait for conversion jobs to complete
        if conv_jobs and not args.dry_run:
            logger.info("Waiting for %d conversion jobs...", len(conv_jobs))
            for key, job in conv_jobs.items():
                try:
                    path = job.result()
                    conversion_paths[key] = path
                    logger.info("  Done: %s / %s -> %s", key[0], key[1], path)
                except Exception as e:
                    logger.error("  FAILED: %s / %s -> %s", key[0], key[1], e)
    elif needs_conversion:
        # --skip-convert: assume files exist at persistent paths
        for ds_name in datasets:
            cfg = DATASETS[ds_name]
            for fmt in formats:
                path = cfg.path_for_format(fmt.key)
                if path.exists():
                    conversion_paths[(ds_name, fmt.key)] = str(path)

    # --- Phase B: Benchmark jobs ---
    logger.info("=" * 60)
    logger.info("Phase B: Submitting benchmark jobs")
    logger.info("=" * 60)

    executor = submitit.AutoExecutor(folder=str(LOGS_DIR / "bench"))
    executor.update_parameters(**_slurm_params(args))

    bench_jobs: list[tuple[str, submitit.Job]] = []

    for bench_name in benchmarks:
        for ds_name in datasets:
            n_runs = n_runs_for_dataset(ds_name)
            for fmt in formats:
                key = (ds_name, fmt.key)
                label = f"{bench_name}/{ds_name}/{fmt.key}"

                # write and parallel_write_scaling don't need pre-converted files
                if bench_name in _NO_CONVERSION:
                    conv_path = None
                else:
                    conv_path = conversion_paths.get(key)
                    if conv_path is None:
                        logger.warning("  SKIP %s: no converted file", label)
                        continue

                if args.dry_run:
                    logger.info("  [DRY RUN] Would run: %s", label)
                    continue

                job = executor.submit(
                    _run_benchmark,
                    bench_name, ds_name, fmt.key, fmt.runner, fmt.params,
                    n_runs, args.cold_cache, conv_path,
                )
                bench_jobs.append((label, job))
                logger.info("  Submitted: %s -> job %s", label, job.job_id)

    if args.dry_run:
        n_conv = sum(1 for ds in datasets for fmt in formats
                     if not DATASETS[ds].path_for_format(fmt.key).exists() or args.overwrite)
        n_bench = len(benchmarks) * len(datasets) * len(formats)
        logger.info(
            "DRY RUN: would submit %d conversion + %d benchmark = %d total SLURM jobs",
            n_conv, n_bench, n_conv + n_bench,
        )
        return

    # Wait for benchmark jobs
    if bench_jobs:
        logger.info("Waiting for %d benchmark jobs...", len(bench_jobs))
        t0 = time.perf_counter()
        done, failed = 0, 0
        for label, job in bench_jobs:
            try:
                job.result()
                done += 1
            except Exception as e:
                logger.error("  FAILED: %s -> %s", label, e)
                failed += 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "All done: %d succeeded, %d failed in %.1fs (wall)",
            done, failed, elapsed,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCX Benchmark Suite — Parallel SLURM Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--benchmarks", nargs="+", metavar="NAME",
                        help=f"Benchmarks to run. Default: all. Available: {BENCHMARK_NAMES}")
    parser.add_argument("--datasets", nargs="+", metavar="NAME",
                        help="Datasets to benchmark. Default: all available on disk.")
    parser.add_argument("--formats", nargs="+", metavar="KEY",
                        help="Format keys. Default: all primary formats.")
    parser.add_argument("--include-additional", action="store_true",
                        help="Include additional formats (BPCells, Parquet).")
    parser.add_argument("--cold-cache", action="store_true",
                        help="Drop OS caches between runs.")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip conversion phase (assume files exist).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert even if output files exist.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be submitted without running.")

    # SLURM parameters
    parser.add_argument("--partition", default="cpu_preemptible",
                        help="SLURM partition (default: cpu_preemptible)")
    parser.add_argument("--cpus", type=int, default=16,
                        help="CPUs per task (default: 16)")
    parser.add_argument("--mem-gb", type=int, default=80,
                        help="Memory in GB per job (default: 80)")
    parser.add_argument("--timeout", type=int, default=240,
                        help="Timeout in minutes per job (default: 240)")

    return parser.parse_args()


if __name__ == "__main__":
    main()
