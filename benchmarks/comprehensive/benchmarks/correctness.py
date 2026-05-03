"""
Correctness Validation Suite.

Integrates the three validation scripts into the benchmark module system
used by run_all.py. Runs:
  1. Scanpy equivalence
  2. Backed-mode equivalence
  3. Preprocessing path cross-validation

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
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Run the correctness validation suite.

    Only executes for the ``scx_auto`` format variant to avoid re-running
    for every format in the benchmark matrix. Returns ``None`` for all
    other format variants.

    ``converted_path`` is accepted to match the canonical benchmark
    contract but unused — each validate_* script materializes its own
    SCX file from the source h5ad.
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

    # SLAF round-trip parity — skipped if slafdb is not installed in this env.
    slaf_checks = []
    try:
        from benchmarks.comprehensive.scripts import validate_slaf_equivalence

        slaf_checks = validate_slaf_equivalence.run_all_checks(dataset.name)
    except ImportError as e:
        logger.info("Skipping SLAF round-trip checks: %s", e)

    suites: dict[str, list] = {
        "scanpy_equivalence": scanpy_checks,
        "backed_equivalence": backed_checks,
        "preprocessing_paths": preproc_checks,
        "slaf_equivalence": slaf_checks,
    }

    all_checks = [c for checks in suites.values() for c in checks]

    n_passed = sum(1 for c in all_checks if c.passed and c.error is None)
    n_failed = sum(1 for c in all_checks if not c.passed and c.error is None)
    n_skipped = sum(1 for c in all_checks if c.error is not None)
    overall_passed = n_failed == 0

    # Per-suite breakdown — emitted into runs[].extra below so the gate
    # can floor each suite independently (a single-suite regression
    # surfaces with the suite name in the floor violation report).
    suite_counts: dict[str, dict[str, int]] = {
        name: {
            "n_passed": sum(1 for c in checks if c.passed and c.error is None),
            "n_failed": sum(1 for c in checks if not c.passed and c.error is None),
            "n_skipped": sum(1 for c in checks if c.error is not None),
            "n_total": len(checks),
        }
        for name, checks in suites.items()
    }

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
            "suites": {name: [c.to_dict() for c in checks]
                       for name, checks in suites.items()},
            "suite_counts": suite_counts,
        },
    )

    # Record total wall time as a single run. Surface gateable counts in
    # runs[].extra — the gate's _load_current_raw_metric reads only
    # runs[].extra[metric] (compare_against_baseline.py:341-371). The
    # metadata-only fields above are for human reporting; floors must
    # land on these per-run keys.
    total_wall = sum(c.duration_s for c in all_checks)
    extra: dict[str, int] = {
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_skipped": n_skipped,
        "n_total": len(all_checks),
        "overall_passed_int": 1 if overall_passed else 0,
    }
    for name, counts in suite_counts.items():
        # Always emit per-suite n_failed (0 when the suite was empty),
        # so a missing key is unambiguous "the validator wasn't run"
        # rather than "the validator returned 0 failures".
        extra[f"{name}_n_failed"] = counts["n_failed"]
        extra[f"{name}_n_passed"] = counts["n_passed"]
        extra[f"{name}_n_skipped"] = counts["n_skipped"]
    result.add_run(wall_s=total_wall, **extra)

    logger.info(
        "Correctness validation: %d passed, %d failed, %d skipped (%.1fs)",
        n_passed,
        n_failed,
        n_skipped,
        total_wall,
    )

    return result
