"""The "laptop test": a full 9-stage analysis under a fixed memory ceiling.

Every component of the standard scverse pipeline (PCA, kNN, Leiden, DE) is
already benchmarked here in isolation. What is not: whether the *whole*
pipeline — open → QC → filter → normalize → log1p → HVG → PCA(50) → kNN(15) →
UMAP + Leiden → Wilcoxon DE — runs to completion on 1M cells inside the 16 or
32 GB an analyst's workstation actually has, while the scanpy in-memory path
is OOM-killed. That end-to-end claim is the one the community asks about and
the suite cannot currently make.

Naming note: this benchmark carries no `accel_` prefix but is structurally an
accelerator benchmark — self-contained, never touching the `runners/*_runner.py`
contract surface. `bench_csc_dispatch` is in the same position, and both are
named in `run_parallel._ARM_SHAPED_BENCHMARKS`, which drives the format
pairing, the format-pool trigger and the smoke-gate exclusion from one place.
Registering the benchmark side without the format side is the half-wired case
that matters: every benchmark with no `SUPPORTED_FORMATS` would then pick up
these private keys.

Memory clamping: the ceiling is the SLURM cgroup, not an in-process
`prlimit`. The harness already works this way — the `SCX_BENCH_OOC_MEM_CAP_GB`
block in `config.estimate_memory_gb` caps the request *below* a benchmark's
footprint precisely to force the out-of-core regime — and `MEMORY_BUDGET_GB`
below is read by an early return in that same function, ahead of its `+50%
safety` / round-to-8-GB tail.

Two further adjustments would otherwise raise the ceiling behind the arm's
back, so `_per_job_slurm_params` exempts this benchmark from both: the
`--scale-factor` multiplier (1.3, which `run_parallel.py --help` recommends for
noisy clusters, turns 16 GB into 21) and the `--mem-gb` floor (which
`capture_baseline.py` always passes). `pipeline_completed_bool` is a claim
about a specific ceiling; if the allocation is not the one on the label, the
number means nothing. `test_community_benchmark_wiring.py` pins both.

STUB — until Phase 3 implements the stages, `run()` writes a typed missing
result (`missing_reason="phase3_stub"`) and returns `None`, so the gap is
visible in the report instead of looking like a benchmark that ran and found
nothing to do.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult, write_missing_result

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({
    "pipeline_ooc_constrained__pyscx_16g",
    "pipeline_ooc_constrained__pyscx_32g",
    "pipeline_ooc_constrained__scanpy_16g",
    "pipeline_ooc_constrained__scanpy_32g",
})

# The memory ceiling each arm runs under, in GB, read by
# `config.estimate_memory_gb` so the SLURM allocation *is* the experiment.
# Declared here rather than parsed from the key suffix at the call site: the
# budget is this benchmark's property, and config.py should not have to know
# the spelling of an arm.
MEMORY_BUDGET_GB: dict[str, int] = {
    "pipeline_ooc_constrained__pyscx_16g": 16,
    "pipeline_ooc_constrained__pyscx_32g": 32,
    "pipeline_ooc_constrained__scanpy_16g": 16,
    "pipeline_ooc_constrained__scanpy_32g": 32,
}


def pipeline_ooc_constrained_variants() -> list[FormatVariant]:
    return [
        FormatVariant(
            name="pyscx backed streaming pipeline (16 GB budget)",
            key="pipeline_ooc_constrained__pyscx_16g",
            category="accel", runner="accel_runner",
            params={"engine": "pyscx", "budget_gb": 16},
        ),
        FormatVariant(
            name="pyscx backed streaming pipeline (32 GB budget)",
            key="pipeline_ooc_constrained__pyscx_32g",
            category="accel", runner="accel_runner",
            params={"engine": "pyscx", "budget_gb": 32},
        ),
        FormatVariant(
            name="scanpy in-memory pipeline (16 GB budget)",
            key="pipeline_ooc_constrained__scanpy_16g",
            category="accel", runner="accel_runner",
            params={"engine": "scanpy", "budget_gb": 16},
        ),
        FormatVariant(
            name="scanpy in-memory pipeline (32 GB budget)",
            key="pipeline_ooc_constrained__scanpy_32g",
            category="accel", runner="accel_runner",
            params={"engine": "scanpy", "budget_gb": 32},
        ),
    ]


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Phase 3 stub — records nothing yet."""
    logger.info(
        "pipeline_ooc_constrained is a Phase-3 stub (%s / %s): recording a typed gap",
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
        benchmark="pipeline_ooc_constrained", format_key=format_variant.key,
        dataset=dataset.name, missing_reason="phase3_stub",
        notes="the nine pipeline stages are not implemented yet; only "
              "the orchestrator wiring is registered.",
    )
    return None
