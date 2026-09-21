"""Streaming gene-set scoring, pyscx vs `scanpy.tl.score_genes`.

`sc.tl.score_genes` bins genes by expression, samples control genes per bin
and subtracts the control mean. It **raises on a backed AnnData**, because its
slicing and binning need dense or eager CSR indexing — so the operation an
analyst reaches for on every marker panel is exactly the one that forces a
full materialisation. `pyscx.accel.score_genes` runs the same arithmetic over
the `for_each_shard_ordered` decode-prefetch engine and works on a backed `X`.

Three scoring flavours are benchmarked because they are three different asks:
`method="scanpy"` (parity), `method="mean"` (light marker score) and
`method="zscore"` (decoupleR-equivalent standardised score).

Parity note for Phase 2: SCX's `control` sampler is not numpy's, so it picks
different control genes from scanpy and the scores differ by ~10 % of their
range at the defaults. The benchmark must therefore lift scanpy's own
`adata.uns['score_genes']['ctrl_genes']` and pass them back through
`ctrl_genes=`; that is what isolates the streaming numerical kernel from
control-sampling variance, and it is the only way the planned
`score_spearman_vs_scanpy` floor means anything.

STUB — until Phase 2 implements the arms, `run()` writes a typed missing
result (`missing_reason="phase2_stub"`) and returns `None`, so the gap is
visible in the report instead of looking like a benchmark that ran and found
nothing to do. See `accel_qc_filter.py` for why the module exists
ahead of its implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "accel_score_genes__pyscx_cpu_scanpy",
    "accel_score_genes__pyscx_cpu_mean",
    "accel_score_genes__pyscx_cpu_zscore",
    "accel_score_genes__scanpy_cpu",
})


def accel_score_genes_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx score_genes (backed, scanpy parity)",
            key="accel_score_genes__pyscx_cpu_scanpy",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx score_genes (backed, mean)",
            key="accel_score_genes__pyscx_cpu_mean",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx score_genes (backed, zscore)",
            key="accel_score_genes__pyscx_cpu_zscore",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="scanpy score_genes (in-memory h5ad)",
            key="accel_score_genes__scanpy_cpu",
            category="accel", runner="accel_runner",
        ),
    ]


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Phase 2 stub — records nothing yet."""
    logger.info(
        "accel_score_genes is a Phase-2 stub (%s / %s): recording a typed gap",
        dataset.name, format_variant.key,
    )
    # A typed missing result rather than a bare `None`. Both make the
    # orchestrator move on, but `None` is written nowhere and is
    # indistinguishable in the report from a triple that was legitimately not
    # applicable — the suite's recurring "green over nothing measured" shape.
    # This way the coverage gap is visible, and a floor added against a
    # still-stubbed arm fails loudly instead of being skipped in silence
    # (`check_absolute_floors` treats a result that exists with the metric
    # absent as a violation, and a result that does not exist as a skip).
    write_missing_result(
        benchmark="accel_score_genes", format_key=format_variant.key,
        dataset=dataset.name, missing_reason="phase2_stub",
        notes="the gene-set-scoring arms are not implemented yet; only "
              "the orchestrator wiring is registered.",
    )
    return None
