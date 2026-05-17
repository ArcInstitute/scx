"""Multimodal streaming vs in-memory row-iteration benchmark (Phase 6b).

Exercises the user-facing Phase 6b read path:

  - streaming  → ``reader.to_mudata(backed=True)`` + per-modality
                  chunked X iteration. Peak RSS is bounded by one
                  chunk per modality.
  - in_memory  → ``reader.to_mudata()`` + the same per-modality
                  chunked iteration on the materialised scipy CSR
                  per modality.

Gated on ``dataset.multimodal == True``; single-modality datasets
return ``None`` so the orchestrator skips silently.

Routed only through the multimodal SCX format variants
(``scx_multimodal_*``) — they're the only runners that go through
``to_mudata`` today. Other multimodal formats (h5mu, zarr-mudata) lack
a true backed-mode iterator and would conflate "streaming" with eager
materialise; we surface a directing skip for them.

The benchmark records two ``add_run`` entries per timed iteration with
the ``operation`` field tagging ``"streaming"`` vs ``"in_memory"``.
``metadata`` carries the medians and the streaming RSS advantage ratio
that's the headline of Phase 6b.
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

# Multimodal SCX variants — the only runners that pair
# ``convert_from_h5mu`` with the Phase 6b backed ``to_mudata`` path.
_SCX_MULTIMODAL_KEYS = (
    "scx_multimodal_per_modality_auto",
    "scx_multimodal_uniform_auto",
)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the multimodal streaming-vs-in-memory benchmark.

    Returns ``None`` for single-modality datasets, for formats without
    backed mudata, or when the runner doesn't implement the
    multimodal iterate methods.
    """
    if not dataset.multimodal:
        logger.info(
            "Skipping %s: dataset %s is single-modality — use "
            "read_streaming_vs_inmemory instead",
            format_variant.key,
            dataset.name,
        )
        return None

    if format_variant.key not in _SCX_MULTIMODAL_KEYS:
        logger.info(
            "Skipping %s: only SCX multimodal variants have a true "
            "backed to_mudata() path today",
            format_variant.key,
        )
        return None

    runner = make_runner(format_variant)
    if "backed_mode" not in runner.capabilities:
        logger.info(
            "Skipping %s: runner does not advertise backed_mode capability",
            format_variant.key,
        )
        return None

    h5mu_path = dataset.h5mu_path
    if not h5mu_path.exists():
        raise FileNotFoundError(
            f"Source .h5mu not found: {h5mu_path}. "
            f"Run benchmarks/scripts/download_{dataset.id.lower()}.py first."
        )

    chunk_size = STREAMING_CHUNK_ROWS
    result = BenchmarkResult(
        benchmark="multimodal_read_streaming_vs_inmemory",
        format=format_variant.key,
        dataset=dataset.name,
        scenario={
            "name": "multimodal_streaming_vs_inmemory",
            "mode": "row_iteration",
            "cache_state": "cold" if cold_cache else "warm",
            "device": "cpu",
            "storage_backend": "local",
        },
        metadata={
            "cold_cache": cold_cache,
            "n_warmup": N_WARMUP_RUNS,
            "chunk_size": chunk_size,
            "modality_names": list(dataset.modality_names),
        },
    )

    _cleanup = None
    if converted_path is not None and Path(converted_path).exists():
        _converted = Path(converted_path)
    else:
        _cleanup = tempfile.TemporaryDirectory(
            prefix=f"scx_bench_mm_stream_{dataset.name}_"
        )
        _converted = Path(_cleanup.name) / f"{dataset.name}.{format_variant.key}"
        logger.info(
            "Converting %s -> %s (%s)",
            h5mu_path.name,
            format_variant.name,
            _converted,
        )
        runner.convert_from_h5mu(h5mu_path, _converted)

    result.file_size_bytes = runner.file_size(_converted)

    streaming_walls: list[float] = []
    streaming_rss: list[float] = []
    in_memory_walls: list[float] = []
    in_memory_rss: list[float] = []

    try:
        # Warm-up on the in-memory path to populate the OS page cache.
        for i in range(N_WARMUP_RUNS):
            logger.info("Warm-up run %d/%d", i + 1, N_WARMUP_RUNS)
            try:
                runner.iterate_in_memory_multimodal(
                    _converted, chunk_size, dataset.modality_names
                )
            except NotImplementedError:
                logger.info(
                    "Skipping %s: runner advertises backed_mode but "
                    "does not implement iterate_in_memory_multimodal",
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
                t_stream = runner.iterate_streaming_multimodal(
                    _converted, chunk_size, dataset.modality_names
                )
            except NotImplementedError:
                logger.info(
                    "Skipping %s: runner advertises backed_mode but "
                    "does not implement iterate_streaming_multimodal",
                    format_variant.key,
                )
                return None
            streaming_walls.append(t_stream.wall_s)
            streaming_rss.append(t_stream.peak_rss_mb)
            extra_s = t_stream.extra or {}
            result.add_run(
                wall_s=t_stream.wall_s,
                user_s=t_stream.user_s,
                sys_s=t_stream.sys_s,
                peak_rss_mb=t_stream.peak_rss_mb,
                operation="streaming",
                chunk_size=extra_s.get("chunk_size"),
                total_chunks=extra_s.get("total_chunks"),
                matrix_sum=extra_s.get("matrix_sum"),
                per_modality=extra_s.get("per_modality"),
            )
            logger.info(
                "  streaming: wall=%.3fs  peak_rss=%.1fMB  chunks=%s  modalities=%s",
                t_stream.wall_s,
                t_stream.peak_rss_mb,
                extra_s.get("total_chunks"),
                extra_s.get("n_modalities"),
            )

            if cold_cache:
                runner._drop_caches()
            gc.collect()

            t_eager = runner.iterate_in_memory_multimodal(
                _converted, chunk_size, dataset.modality_names
            )
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
                total_chunks=extra_e.get("total_chunks"),
                matrix_sum=extra_e.get("matrix_sum"),
                per_modality=extra_e.get("per_modality"),
            )
            logger.info(
                "  in_memory: wall=%.3fs  peak_rss=%.1fMB  chunks=%s  modalities=%s",
                t_eager.wall_s,
                t_eager.peak_rss_mb,
                extra_e.get("total_chunks"),
                extra_e.get("n_modalities"),
            )
    finally:
        if _cleanup is not None:
            _cleanup.cleanup()

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
        "Multimodal streaming-vs-in-memory complete: %s / %s — "
        "streaming wall=%.3fs rss=%.1fMB  vs  in_memory wall=%.3fs rss=%.1fMB",
        format_variant.key,
        dataset.name,
        result.metadata.get("median_wall_s_streaming", 0.0),
        result.metadata.get("median_peak_rss_mb_streaming", 0.0),
        result.metadata.get("median_wall_s_in_memory", 0.0),
        result.metadata.get("median_peak_rss_mb_in_memory", 0.0),
    )
    return result
