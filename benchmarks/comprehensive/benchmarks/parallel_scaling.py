"""
Parallel Read Scaling benchmark — COMPREHENSIVE-BENCHMARKING.md §3.5.

Measures how read throughput scales with thread count for formats that
support parallel I/O (primarily SCX via rayon). For each thread count in
THREAD_COUNTS, performs n_runs full-file reads and records wall time,
then computes speedup and parallel efficiency relative to single-threaded
performance.

Each thread count is run in a **separate subprocess** so that thread pools
(rayon, OpenBLAS, etc.) are created fresh with the correct thread count.
Setting env vars in-process has no effect once these pools are initialized.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    THREAD_COUNTS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

_SINGLE_THREADED_RUNNERS = {"h5ad_runner"}

_THREAD_ENV_VARS = [
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]

# Inline worker script executed in a subprocess for each thread count.
# Reads from the converted file, performs warm-up + timed runs, and
# prints JSON timing results to stdout.
_WORKER_SCRIPT = textwrap.dedent("""\
    import gc
    import json
    import sys

    n_warmup = int(sys.argv[1])
    n_runs = int(sys.argv[2])
    converted_path = sys.argv[3]
    runner_name = sys.argv[4]
    runner_params_json = sys.argv[5]
    cold_cache = sys.argv[6] == "true"

    runner_params = json.loads(runner_params_json)

    from benchmarks.comprehensive.runners import make_runner
    from benchmarks.comprehensive.config import FormatVariant

    # Build a minimal FormatVariant just to instantiate the runner.
    fmt = FormatVariant(
        name="worker", key="worker", runner=runner_name, params=runner_params,
    )
    runner = make_runner(fmt)

    # Warm-up
    for _ in range(n_warmup):
        runner.read_full(converted_path)
        gc.collect()

    # Timed runs
    results = []
    for _ in range(n_runs):
        if cold_cache:
            runner._drop_caches()
        gc.collect()
        timing = runner.read_full(converted_path)
        results.append(timing.to_dict())

    print(json.dumps(results))
""")


def _run_with_thread_count(
    thread_count: int,
    n_warmup: int,
    n_runs: int,
    converted_path: Path,
    format_variant: FormatVariant,
    cold_cache: bool,
) -> list[dict]:
    """Run read_full in a subprocess with the given thread count.

    Returns a list of TimingResult dicts (one per timed run).
    """
    env = os.environ.copy()
    for var in _THREAD_ENV_VARS:
        env[var] = str(thread_count)

    proc = subprocess.run(
        [
            sys.executable, "-c", _WORKER_SCRIPT,
            str(n_warmup),
            str(n_runs),
            str(converted_path),
            format_variant.runner,
            json.dumps(format_variant.params),
            "true" if cold_cache else "false",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Parallel scaling worker failed (threads={thread_count}, "
            f"exit={proc.returncode}).\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}"
        )

    try:
        return json.loads(proc.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse worker JSON output: {exc}\n"
            f"--- stdout ---\n{proc.stdout}"
        ) from exc


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Execute the parallel read scaling benchmark.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to benchmark against.
    format_variant : FormatVariant
        Target format (e.g. SCX auto, Zarr zstd).
    n_runs : int
        Number of timed iterations per thread count (after warm-up).
    cold_cache : bool
        If True, drop OS page caches before each timed run.

    Returns
    -------
    BenchmarkResult
        Structured result with per-run timings tagged by thread count,
        plus metadata containing scaling summary, speedup, and efficiency.
    """
    runner = make_runner(format_variant)
    h5ad_path = dataset.h5ad_path

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. "
            f"Run dataset preparation first."
        )

    is_single_threaded = format_variant.runner in _SINGLE_THREADED_RUNNERS
    thread_counts = [1] if is_single_threaded else THREAD_COUNTS

    result = BenchmarkResult(
        benchmark="parallel_scaling",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "thread_counts": thread_counts,
            "single_threaded_only": is_single_threaded,
        },
    )

    # Use pre-converted file if available, otherwise convert to temp dir.
    _cleanup = None
    if converted_path is not None and Path(converted_path).exists():
        _converted = Path(converted_path)
    else:
        _cleanup = tempfile.TemporaryDirectory(prefix=f"scx_bench_{dataset.name}_")
        _converted = Path(_cleanup.name) / f"{dataset.name}.{format_variant.key}"
        logger.info(
            "Converting %s -> %s (%s)",
            h5ad_path.name,
            format_variant.name,
            _converted,
        )
        runner.convert_from_h5ad(h5ad_path, _converted)

    result.file_size_bytes = runner.file_size(_converted)
    try:
        scaling_summary: dict[str, float] = {}

        for thread_count in thread_counts:
            logger.info(
                "--- Thread count: %d (format: %s, dataset: %s) ---",
                thread_count, format_variant.key, dataset.name,
            )

            timing_dicts = _run_with_thread_count(
                thread_count=thread_count,
                n_warmup=N_WARMUP_RUNS,
                n_runs=n_runs,
                converted_path=_converted,
                format_variant=format_variant,
                cold_cache=cold_cache,
            )

            wall_times: list[float] = []
            for td in timing_dicts:
                result.add_run(
                    wall_s=td["wall_s"], user_s=td.get("user_s", 0.0),
                    sys_s=td.get("sys_s", 0.0),
                    peak_rss_mb=td.get("peak_rss_mb", 0.0),
                    threads=thread_count,
                )
                wall_times.append(td["wall_s"])

            median_wall = statistics.median(wall_times)
            scaling_summary[str(thread_count)] = round(median_wall, 6)

        baseline = scaling_summary.get("1")
        speedup: dict[str, float] = {}
        efficiency: dict[str, float] = {}

        if baseline is not None and baseline > 0:
            for tc_str, median_s in scaling_summary.items():
                tc = int(tc_str)
                sp = baseline / median_s if median_s > 0 else 0.0
                speedup[tc_str] = round(sp, 3)
                efficiency[tc_str] = round(sp / tc, 3) if tc > 0 else 0.0

        result.metadata["scaling_wall_s"] = scaling_summary
        result.metadata["speedup"] = speedup
        result.metadata["efficiency"] = efficiency
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    return result
