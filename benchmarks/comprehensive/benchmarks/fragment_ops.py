"""
Fragment / Manifest Operation Throughput benchmark.

Measures wall-clock and throughput for the four SCX fragment/manifest
mutations exposed by ``scx-ops`` via pyscx:

  * ``append``    — ingest new shards into a base ``.scx``
  * ``delete``    — deletion-vector construction for cell-index predicates
  * ``compact``   — full rewrite that reclaims deleted/orphaned bytes
  * ``rollback``  — revert the active manifest to the prior sequence

This is an SCX-only benchmark. For every non-SCX format variant the module
returns ``None`` (mirrors ``correctness.py``), so the orchestrator silently
skips those combinations.
"""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant, RANDOM_SEED
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.runners.base import FormatRunner

logger = logging.getLogger(__name__)


# Fragment-ops is only implemented for SCX. ``scx_auto`` is chosen as the
# single trigger so we don't re-measure for every codec variant.
_SCX_TRIGGER_KEY = "scx_auto"

# Number of random cell indices to mark deleted. Capped by n_obs at runtime.
_DELETE_N = 10_000


def _current_rss_mb() -> float:
    """Current resident-set size in MB (Linux /proc/self/statm)."""
    return FormatRunner._get_rss_mb()


def _gc() -> None:
    gc.collect()


def _copy_scx(src: Path, dst: Path) -> None:
    """Copy an SCX file atomically for a fresh per-run workspace."""
    shutil.copy2(src, dst)


def _time_op(fn, *args, **kwargs) -> tuple[float, float]:
    """Run *fn* and return ``(wall_s, rss_after_mb)``.

    The RSS value is the *current* resident-set size sampled immediately
    after ``fn`` returns — not a true peak. Sufficient for detecting gross
    regressions; if/when we need true peak, bracket with ``ru_maxrss``.
    """
    _gc()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    return wall, _current_rss_mb()


def _run_append(
    result: BenchmarkResult,
    base_scx: Path,
    workdir: Path,
    n_runs: int,
    n_rows: int,
) -> None:
    """Measure ``pyscx.append`` throughput by appending a copy of the base
    file into a fresh working copy on each iteration.
    """
    import pyscx

    # Pre-stage the "input" file once — this is the right-hand side of the
    # append and is not mutated.
    input_path = workdir / "append_input.scx"
    _copy_scx(base_scx, input_path)
    input_bytes = input_path.stat().st_size

    for i in range(n_runs):
        target_path = workdir / f"append_target_{i}.scx"
        _copy_scx(base_scx, target_path)
        size_before = target_path.stat().st_size

        wall, rss = _time_op(pyscx.append, str(target_path), str(input_path))

        size_after = target_path.stat().st_size
        throughput_mb_s = (input_bytes / (1024 * 1024)) / wall if wall > 0 else 0.0
        rows_per_sec = n_rows / wall if wall > 0 else 0.0

        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            operation="append",
            rows_inserted=n_rows,
            bytes_appended=input_bytes,
            size_before_bytes=size_before,
            size_after_bytes=size_after,
            throughput_mb_s=round(throughput_mb_s, 3),
            rows_per_sec=round(rows_per_sec, 1),
        )
        target_path.unlink(missing_ok=True)
        logger.info(
            "  append run %d/%d: wall=%.3fs throughput=%.1f MB/s rows/s=%.0f",
            i + 1, n_runs, wall, throughput_mb_s, rows_per_sec,
        )

    input_path.unlink(missing_ok=True)


def _run_delete(
    result: BenchmarkResult,
    base_scx: Path,
    workdir: Path,
    n_runs: int,
    n_rows: int,
) -> None:
    """Measure ``pyscx.mark_deleted`` throughput with a random-index
    predicate of size ``min(_DELETE_N, n_rows)``.
    """
    import pyscx

    rng = np.random.default_rng(RANDOM_SEED)
    n_delete = min(_DELETE_N, max(1, n_rows // 2))
    indices = rng.choice(n_rows, size=n_delete, replace=False).tolist()

    for i in range(n_runs):
        target_path = workdir / f"delete_target_{i}.scx"
        _copy_scx(base_scx, target_path)
        size_before = target_path.stat().st_size

        wall, rss = _time_op(pyscx.mark_deleted, str(target_path), indices)

        size_after = target_path.stat().st_size
        rows_per_sec = n_delete / wall if wall > 0 else 0.0

        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            operation="delete",
            rows_deleted=n_delete,
            size_before_bytes=size_before,
            size_after_bytes=size_after,
            rows_per_sec=round(rows_per_sec, 1),
        )
        target_path.unlink(missing_ok=True)
        logger.info(
            "  delete run %d/%d: wall=%.3fs rows/s=%.0f (deleted %d)",
            i + 1, n_runs, wall, rows_per_sec, n_delete,
        )


def _run_compact(
    result: BenchmarkResult,
    base_scx: Path,
    workdir: Path,
    n_runs: int,
    n_rows: int,
) -> None:
    """Measure ``pyscx.compact`` throughput on a file that has had an
    append + delete applied, so compact has real work to do (orphaned
    sections + logical deletions).
    """
    import pyscx

    rng = np.random.default_rng(RANDOM_SEED)
    n_delete = min(_DELETE_N, max(1, n_rows // 2))

    # One-time "dirty" input: base + append + delete. Each iteration copies
    # it to a fresh source path so compact starts from identical state.
    dirty_path = workdir / "compact_dirty.scx"
    _copy_scx(base_scx, dirty_path)
    pyscx.append(str(dirty_path), str(base_scx))
    dirty_indices = rng.choice(n_rows, size=n_delete, replace=False).tolist()
    pyscx.mark_deleted(str(dirty_path), dirty_indices)
    dirty_size = dirty_path.stat().st_size

    for i in range(n_runs):
        input_path = workdir / f"compact_in_{i}.scx"
        output_path = workdir / f"compact_out_{i}.scx"
        _copy_scx(dirty_path, input_path)
        size_before = input_path.stat().st_size

        wall, rss = _time_op(pyscx.compact, str(input_path), str(output_path))

        size_after = output_path.stat().st_size
        reclaimed = size_before - size_after
        throughput_mb_s = (size_before / (1024 * 1024)) / wall if wall > 0 else 0.0

        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            operation="compact",
            size_before_bytes=size_before,
            size_after_bytes=size_after,
            reclaimed_bytes=reclaimed,
            throughput_mb_s=round(throughput_mb_s, 3),
        )
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        logger.info(
            "  compact run %d/%d: wall=%.3fs throughput=%.1f MB/s reclaimed=%d B",
            i + 1, n_runs, wall, throughput_mb_s, reclaimed,
        )

    dirty_path.unlink(missing_ok=True)


def _run_rollback(
    result: BenchmarkResult,
    base_scx: Path,
    workdir: Path,
    n_runs: int,
    n_rows: int,
) -> None:
    """Measure ``pyscx.rollback`` wall-clock. Expected to be near-instant
    (single ``pwrite`` on the root catalog); the measurement catches
    regressions in manifest-revert cost.
    """
    import pyscx

    for i in range(n_runs):
        target_path = workdir / f"rollback_target_{i}.scx"
        _copy_scx(base_scx, target_path)
        # Stage an append so rollback has something to revert.
        pyscx.append(str(target_path), str(base_scx))
        size_before = target_path.stat().st_size

        wall, rss = _time_op(pyscx.rollback, str(target_path))

        size_after = target_path.stat().st_size
        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            operation="rollback",
            size_before_bytes=size_before,
            size_after_bytes=size_after,
        )
        target_path.unlink(missing_ok=True)
        logger.info(
            "  rollback run %d/%d: wall=%.3fs", i + 1, n_runs, wall,
        )


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the fragment-ops throughput benchmark.

    Only runs for ``scx_auto`` — fragment/manifest ops are not defined for
    the competitor formats. Returns ``None`` for every other variant so
    those SLURM jobs exit cleanly without writing spurious results.
    """
    if format_variant.key != _SCX_TRIGGER_KEY:
        return None

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted SCX file for {dataset.name}. "
            f"Run Phase A conversion first (--formats scx_auto)."
        )

    converted_path = Path(converted_path)
    n_rows = dataset.n_obs

    result = BenchmarkResult(
        benchmark="fragment_ops",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "n_runs_per_op": n_runs,
            "n_delete_indices": min(_DELETE_N, max(1, n_rows // 2)),
            "operations": ["append", "delete", "compact", "rollback"],
        },
    )
    result.file_size_bytes = converted_path.stat().st_size

    workroot = tempfile.TemporaryDirectory(
        prefix=f"scx_fragment_ops_{dataset.name}_",
    )
    workdir = Path(workroot.name)

    try:
        logger.info("append: %s (n_obs=%d)", dataset.name, n_rows)
        _run_append(result, converted_path, workdir, n_runs, n_rows)

        logger.info("delete: %s", dataset.name)
        _run_delete(result, converted_path, workdir, n_runs, n_rows)

        logger.info("compact: %s", dataset.name)
        _run_compact(result, converted_path, workdir, n_runs, n_rows)

        logger.info("rollback: %s", dataset.name)
        _run_rollback(result, converted_path, workdir, n_runs, n_rows)
    finally:
        workroot.cleanup()

    # Emit per-operation medians into metadata so the reporting layer does
    # not have to aggregate from per-run records.
    import statistics

    per_op: dict[str, dict[str, list[float]]] = {}
    for run_rec in result.runs:
        op = run_rec.extra.get("operation")
        if not op:
            continue
        per_op.setdefault(op, {}).setdefault("_walls", []).append(run_rec.wall_s)
        for k in ("throughput_mb_s", "rows_per_sec", "reclaimed_bytes"):
            v = run_rec.extra.get(k)
            if v is not None:
                per_op[op].setdefault(f"_{k}", []).append(v)

    summaries: dict[str, dict[str, float]] = {}
    for op, buckets in per_op.items():
        s: dict[str, float] = {}
        for k, vals in buckets.items():
            if not vals:
                continue
            s[k.lstrip("_") + "_median"] = round(statistics.median(vals), 6)
        summaries[op] = s
    result.metadata["per_op_medians"] = summaries

    logger.info(
        "Fragment-ops done: %s — %d runs across 4 operations",
        dataset.name,
        len(result.runs),
    )
    return result
