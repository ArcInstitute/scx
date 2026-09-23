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


#: The CSC policy the shared `scx_auto` fixtures are converted with. See the comment
#: in `convert_dataset_format`.
FIXTURE_CSC_POLICY = "auto"


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
    # The shared `scx_auto` fixtures carry what a default conversion writes: a
    # CSC sidecar under the ingest `auto` rule (n_obs >= 50,000 and n_vars >=
    # 5,000). Stated explicitly rather than inherited from the library default,
    # so the fixture contents are pinned here and a reconvert cannot silently
    # drop the sidecar the route floors in `thresholds.yaml` depend on (the
    # `pipeline_ooc_constrained` `de_route_is_csc` floors fail on a fixture
    # without one). Only `scx_auto`: it is the fixture every accelerator and
    # pipeline benchmark opens, while the other SCX codec variants exist to
    # compare CSR layouts and keep the runner's CSR-only pin. Only the SCX
    # runner has the knob, and only its single-modality path reads it —
    # multimodal fixtures go through `from_mudata`, which takes its own `csc`
    # and is not changed here.
    if format_variant.key == "scx_auto" and hasattr(runner, "csc"):
        runner.csc = FIXTURE_CSC_POLICY
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
