"""Fused QC metrics + atomic axis filtering, pyscx vs scanpy.

Scanpy evaluates `calculate_qc_metrics`' cell totals, cell nonzero counts,
gene totals, gene nonzero counts and each `qc_vars` subset separately, so an
analyst asking for three QC subsets pays seven full passes over X. Since 0.18
`pyscx.accel.calculate_qc_metrics` runs **one** native kernel on every kind of
`X` — backed, lazy, a backed layer handle, or an in-memory scipy/dense matrix
— collapsing that into one row-axis pass (a per-visible-column u64 bitmask
carrying up to 64 subsets at once) plus one column-axis pass, followed by
atomic `filter_cells` / `filter_genes`. That was profiled in Phase 4.1 but has
never been a first-class comprehensive benchmark, so nothing gates it against
scanpy.

The three arms hold the *operation sequence* constant and vary only where `X`
lives, which is the comparison an analyst actually makes: backed SCX, eager
SCX in memory, and scanpy on an eager h5ad.

STUB — until Phase 1 implements the arms, `run()` writes a typed missing result
(`missing_reason="phase1_stub"`) and returns `None`, so the gap is visible in
the report instead of looking like a benchmark that ran and found nothing to
do. The module exists ahead of its implementation because `SUPPORTED_FORMATS`
and `accel_qc_filter_variants()` have to be registered for the benchmark to
resolve to any cells at all: without them
`run_parallel._bench_format_compatible` pairs it with zero formats and it is
scheduled silently never.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result

logger = logging.getLogger(__name__)

# One variant per input mode. Accel-shaped keys: `run_parallel` pairs an
# `accel_*` benchmark only with formats prefixed `f"{bench_name}__"`, and
# `DatasetConfig.path_for_format` short-circuits them to the source h5ad.
SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "accel_qc_filter__pyscx_cpu",
    "accel_qc_filter__pyscx_inmem",
    "accel_qc_filter__scanpy_cpu",
})


def accel_qc_filter_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx fused QC + filter (backed SCX)",
            key="accel_qc_filter__pyscx_cpu",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="pyscx fused QC + filter (in-memory CSR)",
            key="accel_qc_filter__pyscx_inmem",
            category="accel", runner="accel_runner",
        ),
        FormatVariant(
            name="scanpy QC + filter (in-memory h5ad)",
            key="accel_qc_filter__scanpy_cpu",
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
    """Phase 1 stub — measures nothing yet, but says so on the record.

    Returning `None` is the orchestrator's "not applicable, skip" contract (see
    `benchmarks/__init__.py`); the `write_missing_result` below is what keeps
    the skip *typed*. Note what is deliberately not done: returning an empty
    `BenchmarkResult` would serialize as a success with zero runs, which is the
    shape `results.require_runs` exists to reject.
    """
    logger.info(
        "accel_qc_filter is a Phase-1 stub (%s / %s): recording a typed gap",
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
        benchmark="accel_qc_filter", format_key=format_variant.key,
        dataset=dataset.name, missing_reason="phase1_stub",
        notes="the fused-QC arms are not implemented yet; only the "
              "orchestrator wiring is registered.",
    )
    return None
