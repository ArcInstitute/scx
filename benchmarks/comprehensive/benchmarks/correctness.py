"""
Correctness Validation Suite — COMPREHENSIVE-BENCHMARKING.md §3.14.

Integrates the three validation scripts into the benchmark module system
used by run_all.py. Runs:
  1. Scanpy equivalence (§3.14.1)
  2. Backed-mode equivalence (§3.14.2)
  3. Preprocessing path cross-validation (§3.14.3)

Results are aggregated into a single BenchmarkResult with per-check details
stored in the metadata field.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant  # noqa: E402
from benchmarks.comprehensive.results import BenchmarkResult  # noqa: E402

logger = logging.getLogger(__name__)


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
) -> BenchmarkResult | None:
    """Run the correctness validation suite.

    Only executes for the ``scx_auto`` format variant to avoid re-running
    for every format in the benchmark matrix. Returns ``None`` for all
    other format variants.
    """
    # Only run once per dataset — use scx_auto as the trigger
    if format_variant.key != "scx_auto":
        return None

    from benchmarks.comprehensive.scripts import (
        validate_backed_equivalence,
        validate_scanpy_equivalence,
        validate_preprocessing_paths,
    )

    logger.info("Running correctness validation suite on %s", dataset.name)

    # Run each validation suite
    scanpy_checks = validate_scanpy_equivalence.run_all_checks(dataset.name)
    backed_checks = validate_backed_equivalence.run_all_checks(dataset.name)
    preproc_checks = validate_preprocessing_paths.run_all_checks(dataset.name)

    all_checks = scanpy_checks + backed_checks + preproc_checks

    n_passed = sum(1 for c in all_checks if c.passed and c.error is None)
    n_failed = sum(1 for c in all_checks if not c.passed and c.error is None)
    n_skipped = sum(1 for c in all_checks if c.error is not None)
    overall_passed = n_failed == 0

    result = BenchmarkResult(
        benchmark="correctness",
        format="scx_auto",
        dataset=dataset.name,
        metadata={
            "overall_passed": overall_passed,
            "n_passed": n_passed,
            "n_failed": n_failed,
            "n_skipped": n_skipped,
            "n_total": len(all_checks),
            "suites": {
                "scanpy_equivalence": [c.to_dict() for c in scanpy_checks],
                "backed_equivalence": [c.to_dict() for c in backed_checks],
                "preprocessing_paths": [c.to_dict() for c in preproc_checks],
            },
        },
    )

    # Record total wall time as a single run
    total_wall = sum(c.duration_s for c in all_checks)
    result.add_run(wall_s=total_wall)

    logger.info(
        "Correctness validation: %d passed, %d failed, %d skipped (%.1fs)",
        n_passed,
        n_failed,
        n_skipped,
        total_wall,
    )

    return result
