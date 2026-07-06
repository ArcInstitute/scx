"""Out-of-core peak-RSS boundary — scx (streaming) vs shardad (materialize).

The deepest structural difference between the two formats: scx can iterate the
full expression matrix in **bounded** memory (backed / streaming — ~one chunk
resident), while shardad **always materializes** the whole matrix into an
in-memory AnnData on read. This benchmark measures true peak RSS for a full-data
pass at increasing scale, so the report can show scx's peak staying ~flat while
shardad's grows with `n_obs` (and, at census scale, approaches or exceeds node
RAM — the capability boundary).

Scenarios (per format):
  * ``scx_stream``    — scx `iterate_streaming` (backed row-chunk pass, bounded).
  * ``scx_materialize`` — scx `read_full` (to_anndata, full CSR resident).
  * ``shardad_materialize`` — shardad `read_full` (whole matrix in RAM). shardad
    has no streaming path, so this is its only full-read mode.

Peak RSS is a **true high-water mark** via ``rss.PeakRssSampler`` (a background
sampler), not the 2-sample max the other read benches use — the materialize peak
is a transient that instantaneous sampling misses.

Cross-format (``SUPPORTED_FORMATS = {"scx_auto", "shardad"}``); consumes the
Phase-A converted file (not in ``run_parallel._NO_CONVERSION``). If the shardad
materialize OOMs at the largest scale, the killed job *is* the boundary result.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from benchmarks.comprehensive.config import STREAMING_CHUNK_ROWS, DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto", "shardad"})

# Full-matrix reads are heavy at census scale; cap reps well below N_RUNS_SMALL.
_MAX_RUNS = 2


def _timed_peak(fn) -> tuple[float, float]:
    """Run *fn*, returning ``(wall_s, true_peak_rss_mb)``."""
    import gc

    gc.collect()
    with PeakRssSampler() as sampler:
        t0 = time.perf_counter()
        fn()
        wall = time.perf_counter() - t0
    return wall, sampler.peak_mb


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Peak-RSS boundary for one (dataset, format)."""
    if format_variant is None or format_variant.key not in SUPPORTED_FORMATS:
        return None
    if converted_path is None or not Path(converted_path).exists():
        logger.info(
            "ooc_rss_boundary skip (%s, %s): no converted file at %s",
            format_variant.key, dataset.name, converted_path,
        )
        return None

    runner = make_runner(format_variant)
    path = str(converted_path)
    fmt = format_variant.key
    n = min(n_runs, _MAX_RUNS)

    result = BenchmarkResult(
        benchmark="ooc_rss_boundary",
        format=fmt,
        dataset=dataset.name,
        metadata={"n_obs": dataset.n_obs, "n_vars": dataset.n_vars, "cold_cache": cold_cache},
    )

    # (scenario, callable) per format. scx exposes both a bounded stream and a
    # full materialize; shardad only materializes.
    if fmt == "scx_auto":
        scenarios = [
            ("scx_stream", lambda: runner.iterate_streaming(path, STREAMING_CHUNK_ROWS)),
            ("scx_materialize", lambda: runner.read_full(path)),
        ]
    else:  # shardad
        scenarios = [("shardad_materialize", lambda: runner.read_full(path))]

    for scenario, op in scenarios:
        for i in range(n):
            if cold_cache:
                runner._drop_caches()
            wall, peak = _timed_peak(op)
            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak,
                scenario=scenario,
                n_obs=dataset.n_obs,
            )
            logger.info(
                "  %s[%s] %d/%d: wall=%.2fs peak_rss=%.0fMB",
                scenario, dataset.name, i + 1, n, wall, peak,
            )

    _emit_summary(result)
    return result


def _emit_summary(result: BenchmarkResult) -> None:
    import statistics

    buckets: dict[str, dict[str, list[float]]] = {}
    for rec in result.runs:
        sc = rec.extra.get("scenario")
        if not sc:
            continue
        b = buckets.setdefault(sc, {})
        b.setdefault("wall_s", []).append(rec.wall_s)
        b.setdefault("peak_rss_mb", []).append(rec.peak_rss_mb)

    summary = {
        sc: {
            "wall_s_median": round(statistics.median(v["wall_s"]), 4),
            "peak_rss_mb_median": round(statistics.median(v["peak_rss_mb"]), 1),
        }
        for sc, v in buckets.items()
    }
    result.metadata["per_scenario_medians"] = summary
