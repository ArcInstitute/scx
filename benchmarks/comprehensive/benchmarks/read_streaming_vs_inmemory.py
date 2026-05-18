"""Streaming vs in-memory row-iteration benchmark.

For formats that support backed mode, walks the full expression matrix
in fixed-size row chunks via two paths and reports both as separate
``operation`` runs on the same ``BenchmarkResult``:

  1. ``streaming`` — backed open + chunked read (X is never fully
     materialised). Peak RSS should stay close to one chunk's footprint.
  2. ``in_memory`` — eager open + chunked read on the materialised CSR.
     Peak RSS includes the entire matrix.

Per-chunk work is a trivial ``chunk.sum()`` reduction so the decoder
actually runs without adding meaningful compute. The same chunk_size
is used for both modes so wall-time deltas reflect format overhead
rather than chunking strategy.

Formats that don't declare the ``"backed_mode"`` capability — or that
declare it but route through an eager materialisation under the hood
(e.g. the AnnData-on-Zarr ``backed=True`` variant today) — surface a
``NotImplementedError`` from ``iterate_streaming`` / ``iterate_in_memory``
which this module catches and converts into a clean skip (``None``).

The orchestrator stores the result as ``read_streaming_vs_inmemory``.
The ``operation`` column on each run disambiguates the two modes;
``extra.matrix_sum`` is recorded so a follow-on correctness check can
assert ``sum_streaming == sum_in_memory`` within fp32 tolerance.
"""

from __future__ import annotations

import gc
import logging
import statistics
import tempfile
from pathlib import Path

from benchmarks.comprehensive.config import (
    DatasetConfig,
    FormatVariant,
    N_WARMUP_RUNS,
    STREAMING_CHUNK_ROWS,
)
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the streaming-vs-in-memory benchmark.

    Returns ``None`` (silent skip) when the format runner doesn't
    implement the iterate_* methods — i.e. lacks a true streaming path.

    Parameters
    ----------
    dataset : DatasetConfig
        Dataset to benchmark against.
    format_variant : FormatVariant
        Target format (e.g. SCX auto). Format must declare
        ``"backed_mode"`` in its runner ``capabilities`` set.
    n_runs : int
        Number of timed iterations per mode (after warm-up).
    cold_cache : bool
        If True, drop OS page caches before each timed run.
    converted_path : Path | None
        Optional pre-converted file path.
    """
    runner = make_runner(format_variant)

    if "backed_mode" not in runner.capabilities:
        logger.info(
            "Skipping %s: %s does not advertise backed_mode capability",
            format_variant.key,
            runner.name,
        )
        return None

    h5ad_path = dataset.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Source h5ad not found: {h5ad_path}. "
            f"Run dataset preparation first."
        )

    chunk_size = STREAMING_CHUNK_ROWS
    result = BenchmarkResult(
        benchmark="read_streaming_vs_inmemory",
        format=format_variant.key,
        dataset=dataset.name,
        scenario={
            "name": "streaming_vs_inmemory",
            "mode": "row_iteration",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
            "storage_backend": "local",
        },
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "chunk_size": chunk_size,
        },
    )

    # Use pre-converted file if available, otherwise convert to temp dir.
    _cleanup = None
    if converted_path is not None and Path(converted_path).exists():
        _converted = Path(converted_path)
    else:
        _cleanup = tempfile.TemporaryDirectory(
            prefix=f"scx_bench_stream_{dataset.name}_"
        )
        _converted = Path(_cleanup.name) / f"{dataset.name}.{format_variant.key}"
        logger.info(
            "Converting %s -> %s (%s)",
            h5ad_path.name,
            format_variant.name,
            _converted,
        )
        runner.convert_from_h5ad(h5ad_path, _converted)

    result.file_size_bytes = runner.file_size(_converted)

    streaming_walls: list[float] = []
    streaming_rss: list[float] = []
    in_memory_walls: list[float] = []
    in_memory_rss: list[float] = []

    try:
        # Warm-up — only on the in-memory path; the streaming path warms
        # itself naturally on the first timed iteration via the page
        # cache populated here.
        for i in range(N_WARMUP_RUNS):
            logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
            try:
                runner.iterate_in_memory(_converted, chunk_size)
            except NotImplementedError:
                logger.info(
                    "Skipping %s: runner advertises backed_mode but does "
                    "not implement iterate_in_memory",
                    format_variant.key,
                )
                return None
            gc.collect()

        for i in range(n_runs):
            if cold_cache:
                runner._drop_caches()
            gc.collect()

            try:
                logger.info("Streaming run %d/%d", i + 1, n_runs)
                t_stream = runner.iterate_streaming(_converted, chunk_size)
            except NotImplementedError:
                logger.info(
                    "Skipping %s: runner advertises backed_mode but does "
                    "not implement iterate_streaming",
                    format_variant.key,
                )
                return None
            streaming_walls.append(t_stream.wall_s)
            streaming_rss.append(t_stream.peak_rss_mb)
            extra = t_stream.extra or {}
            result.add_run(
                wall_s=t_stream.wall_s,
                user_s=t_stream.user_s,
                sys_s=t_stream.sys_s,
                peak_rss_mb=t_stream.peak_rss_mb,
                operation="streaming",
                chunk_size=extra.get("chunk_size"),
                n_chunks=extra.get("n_chunks"),
                matrix_sum=extra.get("matrix_sum"),
            )
            logger.info(
                "  streaming: wall=%.3fs  peak_rss=%.1fMB  n_chunks=%s",
                t_stream.wall_s,
                t_stream.peak_rss_mb,
                (t_stream.extra or {}).get("n_chunks"),
            )

            if cold_cache:
                runner._drop_caches()
            gc.collect()

            t_eager = runner.iterate_in_memory(_converted, chunk_size)
            in_memory_walls.append(t_eager.wall_s)
            in_memory_rss.append(t_eager.peak_rss_mb)
            extra_e = t_eager.extra or {}
            result.add_run(
                wall_s=t_eager.wall_s,
                user_s=t_eager.user_s,
                sys_s=t_eager.sys_s,
                peak_rss_mb=t_eager.peak_rss_mb,
                operation="in_memory",
                chunk_size=extra_e.get("chunk_size"),
                n_chunks=extra_e.get("n_chunks"),
                matrix_sum=extra_e.get("matrix_sum"),
            )
            logger.info(
                "  in_memory: wall=%.3fs  peak_rss=%.1fMB  n_chunks=%s",
                t_eager.wall_s,
                t_eager.peak_rss_mb,
                (t_eager.extra or {}).get("n_chunks"),
            )
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

    # Summarise per-mode medians and the streaming RSS advantage. Wall
    # ratio reports streaming/in_memory (>1 means streaming is slower);
    # rss ratio reports in_memory/streaming (>1 means streaming saves
    # memory).
    if streaming_walls and in_memory_walls:
        med_stream_wall = statistics.median(streaming_walls)
        med_inmem_wall = statistics.median(in_memory_walls)
        result.metadata["median_wall_s_streaming"] = round(med_stream_wall, 4)
        result.metadata["median_wall_s_in_memory"] = round(med_inmem_wall, 4)
        if med_inmem_wall > 0:
            result.metadata["wall_ratio_streaming_over_in_memory"] = round(
                med_stream_wall / med_inmem_wall, 3
            )
    if streaming_rss and in_memory_rss:
        med_stream_rss = statistics.median(streaming_rss)
        med_inmem_rss = statistics.median(in_memory_rss)
        result.metadata["median_peak_rss_mb_streaming"] = round(med_stream_rss, 1)
        result.metadata["median_peak_rss_mb_in_memory"] = round(med_inmem_rss, 1)
        if med_stream_rss > 0:
            result.metadata["rss_ratio_in_memory_over_streaming"] = round(
                med_inmem_rss / med_stream_rss, 3
            )

    logger.info(
        "Streaming-vs-in-memory complete: %s / %s — "
        "streaming wall=%.3fs rss=%.1fMB  vs  in_memory wall=%.3fs rss=%.1fMB",
        format_variant.key,
        dataset.name,
        result.metadata.get("median_wall_s_streaming", 0.0),
        result.metadata.get("median_peak_rss_mb_streaming", 0.0),
        result.metadata.get("median_wall_s_in_memory", 0.0),
        result.metadata.get("median_peak_rss_mb_in_memory", 0.0),
    )
    return result
