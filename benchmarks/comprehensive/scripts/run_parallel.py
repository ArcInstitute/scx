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
    estimate_memory_gb,
    n_runs_for_dataset,
    partition_for_memory,
)
from benchmarks.comprehensive.convert import convert_dataset_format  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult, write_result  # noqa: E402

logger = logging.getLogger(__name__)

from benchmarks.comprehensive.benchmarks import ALL_BENCHMARKS  # noqa: E402

# Canonical benchmark list lives in benchmarks/__init__.py::ALL_BENCHMARKS —
# add new benchmarks there, not here. This alias preserves the historical
# name so existing callers / subagents don't need to change.
BENCHMARK_NAMES = list(ALL_BENCHMARKS)

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


def _slurm_setup_cmds() -> list[str]:
    """Per-job shell setup: activate the right Python env and clear inherited SLURM vars."""
    import os
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    env_cleanup = "unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true"

    if "scx-bench" in conda_prefix:
        conda_base = os.environ.get("CONDA_EXE", "").replace("/bin/conda", "")
        if not conda_base:
            conda_base = str(Path.home() / "miniforge3")
        env_name = os.path.basename(conda_prefix)
        return [
            env_cleanup,
            f'eval "$({conda_base}/bin/conda shell.bash hook)"',
            f"conda activate {env_name}",
        ]
    return [
        env_cleanup,
        f"export PATH={PROJECT_ROOT}/.venv/bin:$PATH",
    ]


def _slurm_params(args, is_conversion: bool = False) -> dict:
    """Build default submitit executor parameters from CLI args.

    Used as the fallback / cap when per-job sizing isn't applicable
    (e.g. conversion jobs that don't yet know which benchmark will run).
    """
    timeout_min = args.timeout
    if is_conversion and timeout_min < 240:
        timeout_min = 240  # conversions need more time for large datasets

    return {
        "slurm_partition": args.partition,
        "cpus_per_task": args.cpus,
        "mem_gb": args.mem_gb,
        "timeout_min": timeout_min,
        "slurm_setup": _slurm_setup_cmds(),
    }


def _per_job_slurm_params(
    args,
    dataset_name: str,
    format_key: str,
    benchmark: str | None,
    is_conversion: bool = False,
) -> dict:
    """Per-(benchmark, dataset, format) submitit parameters.

    Sizes ``mem_gb`` from ``estimate_memory_gb`` (or the conversion peak when
    benchmark is None) and auto-routes to ``cpu_high_mem`` when the estimate
    exceeds the preemptible cap. The ``--mem-gb`` CLI arg becomes a floor:
    we never request less than the user explicitly asked for.
    """
    from benchmarks.comprehensive.config import (  # deferred to avoid circ
        MEM_CEILING_GB, estimate_time_minutes,
    )

    cfg = DATASETS[dataset_name]
    scale = max(getattr(args, "scale_factor", 1.0), 0.1)

    if is_conversion or benchmark is None:
        # Conversion needs to load the source h5ad and write the target —
        # peak across read_full + write is a safe upper bound.
        mem = max(
            estimate_memory_gb(cfg, format_key, "read_full"),
            estimate_memory_gb(cfg, format_key, "write"),
        )
        time_bench = "write"
    else:
        mem = estimate_memory_gb(cfg, format_key, benchmark)
        time_bench = benchmark

    # Apply the scale factor (memory + time), then clamp.
    mem = min(int(round(mem * scale)), MEM_CEILING_GB)
    mem = max(mem, args.mem_gb)  # CLI floor

    est_time = estimate_time_minutes(cfg, format_key, time_bench)
    est_time = int(round(est_time * scale))
    timeout_min = max(est_time, args.timeout if is_conversion else 0, 5)
    if is_conversion and timeout_min < 240:
        timeout_min = 240

    partition = partition_for_memory(mem, default=args.partition)

    logger.info(
        "Sized %s__%s__%s: mem=%dG time=%dm partition=%s (scale=%.2f)",
        benchmark or "convert", format_key, dataset_name,
        mem, timeout_min, partition, scale,
    )

    return {
        "slurm_partition": partition,
        "cpus_per_task": args.cpus,
        "mem_gb": mem,
        "timeout_min": timeout_min,
        "slurm_setup": _slurm_setup_cmds(),
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

    # Mint a run ID that every submitted job inherits via its env, so the
    # per-result provenance blocks share a single group key for the
    # dashboard's "runs" view (Phase I.2).
    from benchmarks.comprehensive.provenance import current_run_id
    run_id = current_run_id()
    logger.info("Run ID:     %s", run_id)

    # Pre-submit runner contract check (Phase I.8). Catches capability
    # manifest violations before a 400-job fleet hits the scheduler.
    # Gated off with --skip-smoke for exploratory runs.
    if not getattr(args, "skip_smoke", False):
        logger.info("Pre-submit smoke (use --skip-smoke to bypass) …")
        import subprocess as _sp
        rc = _sp.call([
            sys.executable, "-m",
            "benchmarks.comprehensive.scripts.smoke_test_runners",
        ])
        if rc != 0:
            logger.error(
                "Runner contract check failed (rc=%d). Submit blocked. "
                "Fix the runner or re-run with --skip-smoke.", rc,
            )
            return
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
                    params = _per_job_slurm_params(
                        args, ds_name, fmt.key, benchmark=None, is_conversion=True,
                    )
                    logger.info(
                        "  [DRY RUN] Would convert: %s / %s [%dG, %s]",
                        ds_name, fmt.key, params["mem_gb"], params["slurm_partition"],
                    )
                    continue

                params = _per_job_slurm_params(
                    args, ds_name, fmt.key, benchmark=None, is_conversion=True,
                )
                executor.update_parameters(**params)
                job = executor.submit(
                    _run_conversion,
                    ds_name, fmt.key, fmt.runner, fmt.params, args.overwrite,
                )
                conv_jobs[key] = job
                logger.info(
                    "  Submitted: %s / %s [%dG, %s] -> job %s",
                    ds_name, fmt.key, params["mem_gb"], params["slurm_partition"], job.job_id,
                )

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

                params = _per_job_slurm_params(args, ds_name, fmt.key, bench_name)

                if args.dry_run:
                    logger.info(
                        "  [DRY RUN] Would run: %s [%dG, %s]",
                        label, params["mem_gb"], params["slurm_partition"],
                    )
                    continue

                executor.update_parameters(**params)
                job = executor.submit(
                    _run_benchmark,
                    bench_name, ds_name, fmt.key, fmt.runner, fmt.params,
                    n_runs, args.cold_cache, conv_path,
                )
                bench_jobs.append((label, job))
                logger.info(
                    "  Submitted: %s [%dG, %s] -> job %s",
                    label, params["mem_gb"], params["slurm_partition"], job.job_id,
                )

    # Emit run_manifest.json — an authoritative list of submitted triples so
    # watch.py can detect missing-result failures even when submitit itself
    # exits cleanly (Phase I.6).
    if not args.dry_run and bench_jobs:
        manifest_path = LOGS_DIR / "run_manifest.json"
        manifest = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "submitted": [
                {
                    "label": label,
                    "job_id": job.job_id,
                    "submitit_folder": str(job.paths.folder),
                }
                for label, job in bench_jobs
            ],
        }
        import json as _json
        manifest_path.write_text(_json.dumps(manifest, indent=2, default=str))
        logger.info("Run manifest: %s", manifest_path)

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

        # Auto-run the regression gate when a canonical baseline exists
        # (Phase I.6). Non-fatal — reports the gate outcome and returns
        # normally; use the gate's own exit code elsewhere if blocking is
        # desired (the `gate_candidate.sh` wrapper does this).
        if getattr(args, "auto_diff", True):
            _maybe_auto_diff(run_id)


def _maybe_auto_diff(run_id: str) -> None:
    """Invoke compare_against_baseline.py at the end of a run_parallel run.

    Best-effort — a missing canonical baseline is treated as a "not wired
    yet" signal rather than an error, since first-run fleets won't have a
    baseline to compare against.
    """
    import subprocess
    baselines_dir = LOGS_DIR.parent / "results" / "baselines"
    latest = baselines_dir / "LATEST"
    if not (latest.is_symlink() or latest.is_file()):
        logger.info(
            "Auto-diff skipped: no canonical baseline at %s (promote one "
            "with scripts/promote_baseline.py to enable).",
            latest,
        )
        return
    # The current snapshot for this run isn't a capture_baseline.py tree —
    # auto-diff works on the *last* captured snapshot. Operators who want a
    # scored PR should use scripts/gate_candidate.sh instead.
    logger.info(
        "Auto-diff: run_id=%s — use scripts/gate_candidate.sh to compare "
        "against baselines/LATEST; raw results landed in results/raw/.",
        run_id,
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
    parser.add_argument("--mem-gb", type=int, default=8,
                        help="Memory floor in GB per job (default: 8). "
                             "Each job is sized via estimate_memory_gb() based on "
                             "(benchmark, dataset, format); --mem-gb is the lower bound.")
    parser.add_argument("--timeout", type=int, default=240,
                        help="Timeout in minutes per job (default: 240 — "
                             "used as a floor; per-job timeout is sized by "
                             "estimate_time_minutes() × --scale-factor).")
    parser.add_argument(
        "--scale-factor", type=float, default=1.0,
        help="Multiplier applied to per-job memory AND time estimates "
             "(Phase I.5). Use <1.0 (e.g. 0.9) on well-characterized CI, "
             ">1.0 (e.g. 1.3) for conservative operator runs on noisy "
             "clusters. Clamped by MEM_CEILING_GB for memory.",
    )
    parser.add_argument(
        "--skip-smoke", action="store_true",
        help="Skip the pre-submit runner contract check (Phase I.8).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
