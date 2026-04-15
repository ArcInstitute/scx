"""
Parallel Write Scaling benchmark — COMPREHENSIVE-BENCHMARKING.md §3.5.2.

Measures how write (h5ad → format conversion) throughput scales with thread
count. SCX parallelizes shard encoding via rayon, so write speed should
improve with additional cores. Two modes are measured:

  - "full": End-to-end h5ad read + format write (real-world conversion).
  - "write_only": Pre-load AnnData in memory, time only the format write
    (isolates parallel encoding). Only applicable to SCX.

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

logger = logging.getLogger(__name__)

_SINGLE_THREADED_RUNNERS = {"h5ad_runner"}

# Runners that support the "write_only" mode (pre-loaded AnnData → format).
_SUPPORTS_WRITE_ONLY = {"scx_runner"}

_THREAD_ENV_VARS = [
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
]

# Inline worker script executed in a subprocess for each thread count.
# Converts from h5ad (or pre-loaded AnnData) to target format, performs
# warm-up + timed runs, and prints JSON timing results to stdout.
_WORKER_SCRIPT = textwrap.dedent("""\
    import gc
    import json
    import os
    import sys
    import tempfile
    import time
    from pathlib import Path

    n_warmup = int(sys.argv[1])
    n_runs = int(sys.argv[2])
    h5ad_path = sys.argv[3]
    runner_name = sys.argv[4]
    runner_params_json = sys.argv[5]
    cold_cache = sys.argv[6] == "true"
    mode = sys.argv[7]  # "full" or "write_only"

    runner_params = json.loads(runner_params_json)

    from benchmarks.comprehensive.runners import make_runner
    from benchmarks.comprehensive.config import FormatVariant

    fmt = FormatVariant(
        name="worker", key="worker", category="primary",
        runner=runner_name, params=runner_params,
    )
    runner = make_runner(fmt)

    # For write_only mode, pre-load AnnData once outside the timing loop.
    adata = None
    if mode == "write_only":
        import anndata
        adata = anndata.read_h5ad(h5ad_path)
        codec = runner_params.get("codec", "auto")

    def _get_rss_mb():
        try:
            with open("/proc/self/statm") as f:
                pages = int(f.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        except Exception:
            return 0.0

    def do_one_write(out_path):
        gc.collect()
        if cold_cache:
            runner._drop_caches()
        rss_before = _get_rss_mb()
        t0 = time.perf_counter()
        if mode == "write_only":
            import pyscx
            pyscx.from_anndata(adata, str(out_path), codec=codec)
            wall = time.perf_counter() - t0
            rss_after = _get_rss_mb()
            output_size = os.path.getsize(out_path)
            throughput = (output_size / (1024 * 1024)) / wall if wall > 0 else 0.0
            return {
                "wall_s": wall,
                "peak_rss_mb": max(rss_before, rss_after),
                "output_size_bytes": output_size,
                "write_throughput_mb_s": throughput,
            }
        else:
            cr = runner.convert_from_h5ad(h5ad_path, str(out_path))
            return {
                "wall_s": cr.wall_s,
                "peak_rss_mb": cr.peak_rss_mb,
                "output_size_bytes": cr.output_size_bytes,
                "write_throughput_mb_s": cr.write_throughput_mb_s,
            }

    # Warm-up
    for _ in range(n_warmup):
        with tempfile.TemporaryDirectory(prefix="scx_wscale_warmup_") as tmpdir:
            do_one_write(Path(tmpdir) / "output")

    # Timed runs
    results = []
    for _ in range(n_runs):
        with tempfile.TemporaryDirectory(prefix="scx_wscale_timed_") as tmpdir:
            result = do_one_write(Path(tmpdir) / "output")
            results.append(result)

    print(json.dumps(results))
""")


def _run_with_thread_count(
    thread_count: int,
    n_warmup: int,
    n_runs: int,
    h5ad_path: Path,
    format_variant: FormatVariant,
    cold_cache: bool,
    mode: str,
) -> list[dict]:
    """Run write conversion in a subprocess with the given thread count.

    Returns a list of result dicts (one per timed run).
    """
    env = os.environ.copy()
    for var in _THREAD_ENV_VARS:
        env[var] = str(thread_count)

    proc = subprocess.run(
        [
            sys.executable, "-c", _WORKER_SCRIPT,
            str(n_warmup),
            str(n_runs),
            str(h5ad_path),
            format_variant.runner,
            json.dumps(format_variant.params),
            "true" if cold_cache else "false",
            mode,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=7200,  # writes can be slower than reads; generous timeout
    )

    if proc.returncode != 0:
        raise RuntimeError(
            f"Parallel write scaling worker failed (threads={thread_count}, "
            f"mode={mode}, exit={proc.returncode}).\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}"
        )

    try:
        # Parse JSON from the last line — earlier lines may contain library
        # warnings or deprecation notices printed to stdout.
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"Failed to parse worker JSON output: {exc}\n"
            f"--- stdout ---\n{proc.stdout}"
        ) from exc


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
) -> BenchmarkResult:
    """Execute the parallel write scaling benchmark.

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
        Structured result with per-run timings tagged by thread count
        and mode, plus metadata containing scaling summary, speedup,
        and efficiency for each mode.
    """
    h5ad_path = dataset.h5ad_path

    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. "
            f"Run dataset preparation first."
        )

    source_h5ad_bytes = os.path.getsize(h5ad_path)

    is_single_threaded = format_variant.runner in _SINGLE_THREADED_RUNNERS
    thread_counts = [1] if is_single_threaded else THREAD_COUNTS

    supports_write_only = format_variant.runner in _SUPPORTS_WRITE_ONLY
    modes = ["full", "write_only"] if supports_write_only else ["full"]

    result = BenchmarkResult(
        benchmark="parallel_write_scaling",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "thread_counts": thread_counts,
            "modes": modes,
            "single_threaded_only": is_single_threaded,
            "source_h5ad_bytes": source_h5ad_bytes,
        },
    )

    # Per-mode scaling summaries: {mode: {thread_count_str: median_wall_s}}
    all_scaling: dict[str, dict[str, float]] = {}
    all_speedup: dict[str, dict[str, float]] = {}
    all_efficiency: dict[str, dict[str, float]] = {}

    for mode in modes:
        scaling_summary: dict[str, float] = {}

        for thread_count in thread_counts:
            logger.info(
                "--- threads=%d, mode=%s (format=%s, dataset=%s) ---",
                thread_count, mode, format_variant.key, dataset.name,
            )

            timing_dicts = _run_with_thread_count(
                thread_count=thread_count,
                n_warmup=N_WARMUP_RUNS,
                n_runs=n_runs,
                h5ad_path=h5ad_path,
                format_variant=format_variant,
                cold_cache=cold_cache,
                mode=mode,
            )

            wall_times: list[float] = []
            for td in timing_dicts:
                result.add_run(
                    wall_s=td["wall_s"],
                    peak_rss_mb=td.get("peak_rss_mb", 0.0),
                    threads=thread_count,
                    mode=mode,
                    write_throughput_mb_s=td.get("write_throughput_mb_s", 0.0),
                    output_size_bytes=td.get("output_size_bytes", 0),
                )
                wall_times.append(td["wall_s"])

            median_wall = statistics.median(wall_times)
            scaling_summary[str(thread_count)] = round(median_wall, 6)

        # Compute speedup and efficiency relative to single-threaded.
        baseline = scaling_summary.get("1")
        speedup: dict[str, float] = {}
        efficiency: dict[str, float] = {}

        if baseline is not None and baseline > 0:
            for tc_str, median_s in scaling_summary.items():
                tc = int(tc_str)
                sp = baseline / median_s if median_s > 0 else 0.0
                speedup[tc_str] = round(sp, 3)
                efficiency[tc_str] = round(sp / tc, 3) if tc > 0 else 0.0

        all_scaling[mode] = scaling_summary
        all_speedup[mode] = speedup
        all_efficiency[mode] = efficiency

    result.metadata["scaling_wall_s"] = all_scaling
    result.metadata["speedup"] = all_speedup
    result.metadata["efficiency"] = all_efficiency

    return result
