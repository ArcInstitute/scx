"""
Shared conversion utility for the benchmark suite.

Converts datasets to target formats and caches results at persistent paths
(via DatasetConfig.path_for_format). Used by run_parallel.py to avoid
redundant conversions across benchmark modules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.runners import make_runner

logger = logging.getLogger(__name__)


def convert_dataset_format(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    overwrite: bool = False,
) -> Path:
    """Convert a dataset to the given format, writing to its persistent path.

    Skips conversion if the output already exists and *overwrite* is False.

    Returns the output path.
    """
    output_path = dataset.path_for_format(format_variant.key)

    if output_path.exists() and not overwrite:
        logger.info(
            "Skipping conversion (exists): %s -> %s at %s",
            dataset.name, format_variant.key, output_path,
        )
        return output_path

    logger.info(
        "Converting %s -> %s  (%s)",
        dataset.name, format_variant.key, output_path,
    )

    runner = make_runner(format_variant)
    # Per-dataset shard target, where the dataset declares one. Only the SCX
    # runner has the knob; on every other runner the attribute simply is not
    # there and the dataset's request is a no-op, which is correct — the
    # constraint it exists for (a bounded per-shard SCX read) has no analogue.
    if getattr(dataset, "shard_size", None) is not None and hasattr(runner, "shard_size"):
        runner.shard_size = dataset.shard_size
    if dataset.multimodal:
        # Phase K: multimodal datasets ship as `.h5mu`; route through
        # the multimodal-aware converter on the runner. Single-modality
        # runners raise NotImplementedError on this path, so an
        # accidental pass against a non-multimodal runner surfaces a
        # clear error instead of silently emitting a malformed file.
        runner.convert_from_h5mu(dataset.h5mu_path, output_path)
    else:
        runner.convert_from_h5ad(dataset.h5ad_path, output_path)

    logger.info("  Done: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6
                if output_path.is_file() else 0)
    return output_path
