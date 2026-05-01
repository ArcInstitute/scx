#!/usr/bin/env python3
"""
SCX Comprehensive Benchmark Suite — Parallel SLURM Launcher via submitit.

Two-phase execution:
  Phase A: Convert each (dataset, format) pair to a persistent path.
  Phase B: Run benchmarks in parallel, reading from pre-converted files.

Each (benchmark, dataset, format) triple is submitted as an independent SLURM
job, enabling massive parallelism across the cluster.

Phase A and Phase B overlap: as soon as a conversion job is submitted,
its dependent benchmark jobs are submitted with a SLURM
``--dependency=afterok:<jid>`` directive. SLURM holds them in PENDING until
the conversion lands, then releases them — so a fast conversion (pbmc3k)
unblocks its benchmarks long before a slow conversion (census_5m, h5ad_lzf)
finishes. Benchmarks whose target file already exists, and ones in
``_NO_CONVERSION``, submit with no dependency at all.

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
from typing import Any

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
# cell_eval_parity_perf generates synthetic data in-process (no source
# .h5ad on disk for pert_synth_*), so Phase A would fail — skip it.
_NO_CONVERSION = {"write", "parallel_write_scaling", "cell_eval_parity_perf"}

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

    # Benchmarks may return None when the (bench, format) combo is a
    # deliberate skip (e.g. fragment_ops / cloud_push / cloud_pull only
    # apply to SCX; capability-gated cross-format modules return None for
    # runners that don't declare the capability). That's a successful
    # no-op, not a failure — don't try to persist it.
    if result is None:
        return {"skipped": True, "benchmark": bench_name,
                "format": fmt.key, "dataset": dataset.name}

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
    # Disable srun's MPI bootstrap. Chimera's slurm.conf has MpiDefault=pmix,
    # but the pmix plugin isn't runtime-loadable — submitit's `srun` would
    # otherwise fail with "Cannot create context for mpi/pmix". We don't
    # use MPI; tell srun so.
    mpi_none = "export SLURM_MPI_TYPE=none"

    if "scx-bench" in conda_prefix:
        conda_base = os.environ.get("CONDA_EXE", "").replace("/bin/conda", "")
        if not conda_base:
            conda_base = str(Path.home() / "miniforge3")
        env_name = os.path.basename(conda_prefix)
        return [
            env_cleanup,
            mpi_none,
            f'eval "$({conda_base}/bin/conda shell.bash hook)"',
            f"conda activate {env_name}",
        ]
    return [
        env_cleanup,
        mpi_none,
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

    # Route accelerator benchmarks
    # to the GPU partition **only for variants that actually use the GPU**.
    # Variant naming convention: format_key ending in `_gpu` or containing
    # `_gpu_` indicates GPU dispatch (e.g. `accel_pca__pyscx_gpu_cov`,
    # `accel_knn__pyscx_gpu_cagra`). CPU-only variants (`accel_*__scanpy_cpu`,
    # `accel_*__pyscx_cpu*`, `accel_leiden__leidenalg_cpu`) flow through
    # partition_for_memory which auto-promotes to cpu_high_mem when sizing
    # exceeds the preemptible-GPU node's RAM ceiling — unblocking
    # accel_preprocess on census_1m (needs >64 GB).
    #
    # ml_loader on SCX formats also routes to GPU so the SCX-only
    # `gpu_train` scenario in benchmarks/ml_loader.py:949 can fire (it
    # silently no-ops on hosts without CUDA). Non-SCX ml_loader variants
    # have no GPU scenario and stay on CPU.
    extra_slurm: dict = {}
    needs_gpu = (
        benchmark is not None
        and (
            (
                benchmark.startswith("accel_")
                and ("_gpu_" in format_key or format_key.endswith("_gpu"))
            )
            or (benchmark == "ml_loader" and format_key.startswith("scx_"))
        )
    )
    if needs_gpu:
        partition = "preemptible"  # GPU partition on chimera
        extra_slurm["slurm_gres"] = "gpu:1"
        # Chimera's preemptible GPU QOS caps per-job memory at ~128 GB
        # (matching SLURM_DEFAULTS.gpu.mem_gb). Requests above that fail
        # with `QOSMaxGRESPerJob`. Clamp so census-scale preprocess cells
        # that actually use GPU compute don't get rejected at submit time.
        # (CPU variants have already been routed away via the `needs_gpu`
        # check and will pick up cpu_high_mem via partition_for_memory.)
        _GPU_MEM_CEILING_GB = 128
        if mem > _GPU_MEM_CEILING_GB:
            logger.warning(
                "Clamping %s__%s__%s mem from %dG to %dG (GPU QOS cap)",
                benchmark, format_key, dataset_name,
                mem, _GPU_MEM_CEILING_GB,
            )
            mem = _GPU_MEM_CEILING_GB

    logger.info(
        "Sized %s__%s__%s: mem=%dG time=%dm partition=%s%s (scale=%.2f)",
        benchmark or "convert", format_key, dataset_name,
        mem, timeout_min, partition,
        " gres=gpu:1" if extra_slurm.get("slurm_gres") else "",
        scale,
    )

    params: dict = {
        "slurm_partition": partition,
        "cpus_per_task": args.cpus,
        "mem_gb": mem,
        "timeout_min": timeout_min,
        "slurm_setup": _slurm_setup_cmds(),
    }
    params.update(extra_slurm)
    return params


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
    # `accel_formats()` is lazy-loaded — pulls in the `FormatVariant` entries
    # for each `accel_*.py` module's implementation variants (PCA: scanpy_cpu,
    # pyscx_cpu_auto, pyscx_gpu_cov, pyscx_gpu_rand_hh, pyscx_gpu_rand_chol;
    # kNN / UMAP / Leiden / preprocess / HVG similarly). Included whenever
    # --include-accel is set, when --formats explicitly names an accel key,
    # or when --benchmarks names any accel_* benchmark.
    from benchmarks.comprehensive.config import accel_formats
    need_accel = (
        args.include_accel
        or any((fk or "").startswith("accel_") for fk in (args.formats or []))
        or any((b or "").startswith("accel_") for b in (args.benchmarks or []))
    )
    pool = list(ALL_FORMATS)
    if need_accel:
        pool.extend(accel_formats())
    if args.formats:
        all_by_key = {f.key: f for f in pool}
        formats = [all_by_key[k] for k in args.formats if k in all_by_key]
    elif args.include_additional and need_accel:
        formats = pool
    elif args.include_additional:
        formats = list(ALL_FORMATS)
    elif need_accel:
        formats = list(PRIMARY_FORMATS) + accel_formats()
    else:
        formats = list(PRIMARY_FORMATS)

    # --no-gpu: drop accel GPU variants. By convention accel format keys are
    # `<bench>__<impl>_<device>[_<extra>]`, e.g. `accel_pca__pyscx_gpu_cov`.
    # The substring "_gpu" is unique to GPU variants — CPU keys use "_cpu" and
    # format-benchmark keys (zarr_zstd, scx_auto, …) don't include "_gpu".
    if args.no_gpu:
        before = len(formats)
        formats = [f for f in formats if "_gpu" not in f.key]
        dropped = before - len(formats)
        if dropped:
            logger.info("--no-gpu: dropped %d GPU accel format variants", dropped)

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
    # Gated off with --skip-smoke for exploratory runs. Skipped automatically
    # when no non-accel benchmarks are scheduled (accel benchmarks never
    # touch the `runners/*_runner.py` contract surface — they go through
    # `accel_*.py` modules in `comprehensive/benchmarks/`, which the smoke
    # gate doesn't validate).
    non_accel_benchmarks = [b for b in benchmarks if not b.startswith("accel_")]
    needs_smoke = (
        not getattr(args, "skip_smoke", False)
        and len(non_accel_benchmarks) > 0
    )
    if needs_smoke:
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
            sys.exit(1)
    elif not getattr(args, "skip_smoke", False):
        logger.info(
            "Pre-submit smoke skipped: no non-accel benchmarks scheduled. "
            "Pass --skip-smoke to suppress this notice."
        )
    logger.info("Datasets:   %s", datasets)
    logger.info("Formats:    %s", [f.key for f in formats])
    logger.info("Benchmarks: %s", benchmarks)

    # Determine which benchmarks need pre-converted files
    needs_conversion = [b for b in benchmarks if b not in _NO_CONVERSION]

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Phase A: Conversion ---
    # `conversion_paths` records files already on disk; `conv_jobs` records
    # conversions submitted this run (whose Phase-B dependents will gate on
    # `afterok:<jid>`). A `(ds, fmt)` key appears in at most one of the two.
    # `dry_run_conv_keys` mirrors `conv_jobs` for --dry-run so Phase B can
    # report would-be afterok edges without actually submitting anything.
    conversion_paths: dict[tuple[str, str], str] = {}
    conv_jobs: dict[tuple[str, str], submitit.Job] = {}
    dry_run_conv_keys: set[tuple[str, str]] = set()

    if needs_conversion and not args.skip_convert:
        logger.info("=" * 60)
        logger.info("Phase A: Converting datasets to target formats")
        logger.info("=" * 60)

        executor = submitit.AutoExecutor(folder=str(LOGS_DIR / "convert"))

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
                    dry_run_conv_keys.add(key)
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

        if conv_jobs and not args.dry_run:
            logger.info(
                "Submitted %d conversion job(s); benchmarks will release as each lands.",
                len(conv_jobs),
            )
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

    # Each entry: (label, bench_job, conv_key | None). conv_key points back
    # at conv_jobs so the post-hoc wait loop can attribute a benchmark
    # failure to a cancelled-by-dependency state vs a real benchmark crash.
    bench_jobs: list[tuple[str, submitit.Job, tuple[str, str] | None]] = []
    dep_jobid_for_label: dict[str, str | None] = {}

    for bench_name in benchmarks:
        for ds_name in datasets:
            n_runs = n_runs_for_dataset(ds_name)
            for fmt in formats:
                # Accelerator benchmarks are self-contained: each
                # `accel_X` module owns its own set of variants
                # (`accel_X__<impl>`). Pairing `accel_pca` with an
                # `accel_knn__*` format — or with any non-accel format —
                # schedules a cell whose `run()` returns None (waste).
                # Skip those pairings at the launcher level.
                if bench_name.startswith("accel_"):
                    if not fmt.key.startswith(f"{bench_name}__"):
                        continue
                elif fmt.key.startswith("accel_"):
                    # Non-accel benchmarks (read_full, etc.) don't pair
                    # with accel variants either.
                    continue

                key = (ds_name, fmt.key)
                label = f"{bench_name}/{ds_name}/{fmt.key}"

                # Decide the conversion source and the SLURM dependency.
                # Three states: (a) bench needs no conversion → no dep,
                # path is None; (b) conversion was submitted this run →
                # afterok dep on its job_id, path is the deterministic
                # on-disk target (visible after the conv job lands);
                # (c) converted file already on disk → no dep.
                dep_jobid: str | None = None
                if bench_name in _NO_CONVERSION:
                    conv_path = None
                elif key in conv_jobs:
                    dep_jobid = conv_jobs[key].job_id
                    conv_path = str(DATASETS[ds_name].path_for_format(fmt.key))
                elif args.dry_run and key in dry_run_conv_keys:
                    # In --dry-run we never actually submit Phase A,
                    # so use a placeholder jobid to surface the would-be
                    # afterok edge in the per-cell log.
                    dep_jobid = "<conv-pending>"
                    conv_path = str(DATASETS[ds_name].path_for_format(fmt.key))
                else:
                    conv_path = conversion_paths.get(key)
                    if conv_path is None:
                        logger.warning("  SKIP %s: no converted file", label)
                        continue

                params = _per_job_slurm_params(args, ds_name, fmt.key, bench_name)

                if args.dry_run:
                    dep_str = f" deps=afterok:{dep_jobid}" if dep_jobid else ""
                    logger.info(
                        "  [DRY RUN] Would run: %s [%dG, %s]%s",
                        label, params["mem_gb"], params["slurm_partition"], dep_str,
                    )
                    continue

                update_kwargs: dict[str, Any] = dict(params)
                if dep_jobid is not None:
                    extra = update_kwargs.get("slurm_additional_parameters", {}) or {}
                    extra = {**extra, "dependency": f"afterok:{dep_jobid}"}
                    update_kwargs["slurm_additional_parameters"] = extra
                executor.update_parameters(**update_kwargs)
                job = executor.submit(
                    _run_benchmark,
                    bench_name, ds_name, fmt.key, fmt.runner, fmt.params,
                    n_runs, args.cold_cache, conv_path,
                )
                bench_jobs.append((label, job, key if dep_jobid is not None else None))
                dep_jobid_for_label[label] = dep_jobid
                dep_str = f" deps=afterok:{dep_jobid}" if dep_jobid else ""
                logger.info(
                    "  Submitted: %s [%dG, %s]%s -> job %s",
                    label, params["mem_gb"], params["slurm_partition"], dep_str, job.job_id,
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
                    "dependency": dep_jobid_for_label.get(label),
                }
                for label, job, _conv_key in bench_jobs
            ],
        }
        import json as _json
        manifest_path.write_text(_json.dumps(manifest, indent=2, default=str))
        logger.info("Run manifest: %s", manifest_path)

    if args.dry_run:
        # Mirror the pairing filter from the Phase-B loop so dry-run
        # counts match what an actual submission would produce.
        def _pairs_ok(b: str, fk: str) -> bool:
            if b.startswith("accel_"):
                return fk.startswith(f"{b}__")
            return not fk.startswith("accel_")

        n_conv = sum(
            1 for ds in datasets for fmt in formats
            if not DATASETS[ds].path_for_format(fmt.key).exists() or args.overwrite
        )
        n_bench = sum(
            1 for b in benchmarks for ds in datasets for fmt in formats
            if _pairs_ok(b, fmt.key)
        )
        n_dep = sum(
            1 for b in benchmarks for ds in datasets for fmt in formats
            if _pairs_ok(b, fmt.key)
            and b not in _NO_CONVERSION
            and (
                not DATASETS[ds].path_for_format(fmt.key).exists() or args.overwrite
            )
        )
        logger.info(
            "DRY RUN: would submit %d conversion + %d benchmark "
            "(%d benchmark→conversion afterok edges) = %d total SLURM jobs",
            n_conv, n_bench, n_dep, n_conv + n_bench,
        )
        return

    # Wait for benchmark jobs. With afterok dependencies in play, the wait
    # loop sees three outcomes per job: success, failure (the benchmark
    # itself crashed), and dep_failed (SLURM cancelled the job because its
    # upstream conversion never succeeded — DependencyNeverSatisfied).
    # Distinguish the latter so dependency-cascade noise doesn't masquerade
    # as benchmark regressions in the operator's eye.
    if bench_jobs:
        logger.info("Waiting for %d benchmark jobs...", len(bench_jobs))
        t0 = time.perf_counter()
        done, failed, dep_failed = 0, 0, 0
        for label, job, conv_key in bench_jobs:
            try:
                job.result()
                done += 1
            except Exception as e:
                upstream_failed = False
                if conv_key is not None and conv_key in conv_jobs:
                    try:
                        conv_jobs[conv_key].result()
                    except Exception:
                        upstream_failed = True
                if upstream_failed:
                    logger.warning(
                        "  DEP_FAILED: %s -> upstream conversion %s (job %s) failed",
                        label, conv_key, conv_jobs[conv_key].job_id,
                    )
                    dep_failed += 1
                else:
                    logger.error("  FAILED: %s -> %s", label, e)
                    failed += 1

        # Surface conversion failures top-level. Without this, a conv crash
        # is only visible indirectly through `dep_failed` counts on its
        # dependents — operators have no top-level signal of *which*
        # conversion failed. By the time we get here every bench job is
        # terminal, so its upstream conv is terminal too; .result() is
        # cached and won't block.
        conv_done, conv_failed = 0, 0
        for conv_key, conv_job in conv_jobs.items():
            try:
                conv_job.result()
                conv_done += 1
            except Exception as e:
                logger.error(
                    "  CONV_FAILED: %s / %s (job %s) -> %s",
                    conv_key[0], conv_key[1], conv_job.job_id, e,
                )
                conv_failed += 1

        elapsed = time.perf_counter() - t0
        summary = f"{done} succeeded, {failed} failed"
        if dep_failed:
            summary += f", {dep_failed} dep_failed"
        if conv_failed:
            summary += f" (+{conv_failed} conversion failures)"
        elif conv_jobs:
            summary += f" (+{conv_done} conversions OK)"
        logger.info("All done: %s in %.1fs (wall)", summary, elapsed)

        # Auto-run the regression gate when a canonical baseline exists
        # (Phase I.6). Non-fatal — reports the gate outcome and returns
        # normally; use the gate's own exit code elsewhere if blocking is
        # desired (the `gate_candidate.py` wrapper does this).
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
    # scored PR should use scripts/gate_candidate.py instead.
    logger.info(
        "Auto-diff: run_id=%s — use scripts/gate_candidate.py to compare "
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
    parser.add_argument("--include-accel", action="store_true",
                        help="Include accelerator-variant formats (accel_pca__*, "
                             "accel_knn__*, etc.) registered via "
                             "config.accel_formats(). Auto-enabled when "
                             "--benchmarks names any accel_* benchmark or "
                             "--formats names any accel_* key.")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Drop accel formats whose key matches *_gpu* "
                             "(e.g. accel_pca__pyscx_gpu_cov, "
                             "accel_knn__pyscx_gpu). Use on CPU-only hosts or "
                             "when validating CPU-only changes on a GPU box. "
                             "CPU accel variants and format benchmarks are "
                             "unaffected.")
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
