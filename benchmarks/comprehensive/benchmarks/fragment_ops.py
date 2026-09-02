"""
Fragment / Manifest Operation Throughput benchmark.

Measures wall-clock and throughput for the five SCX fragment/manifest
mutations exposed by ``scx-ops`` via pyscx:

  * ``append``     — ingest new shards into a base ``.scx``
  * ``delete``     — deletion-vector construction for cell-index predicates
  * ``compact``    — full rewrite that reclaims deleted/orphaned bytes
  * ``obs_import`` — key-joined in-place add of one obs column from a CSV
  * ``rollback``   — revert the active manifest to the prior sequence

plus, when ``<dataset>_full.scx`` has been built by
``benchmarks/scripts/prep_full_fixtures.py``:

  * ``compact_full`` — ``compact`` on a file that carries two ``obsm`` keys and
    a layer, neither of which any ordinary fixture in the suite has. Carries
    ``peak_rss_mb__compact_full``. The fixture also has a ``.raw``, but
    ``compact`` drops raw (with a warning) rather than carrying it, so this arm
    does **not** measure a raw copy — see ``_run_compact_full``.

This is an SCX-only benchmark. For every non-SCX format variant the module
returns ``None`` (mirrors ``correctness.py``), so the orchestrator silently
skips those combinations.

Every op's wall is also emitted as a sparse ``wall_s__<operation>`` key
---------------------------------------------------------------------

``wall_s`` is a **reserved** ``add_run`` parameter: it lands on the
``RunRecord`` and never reaches ``runs[].extra``, which is the only place
``compare_against_baseline`` can read a threshold from. So although this module
has always *timed* four in-place mutations, none of those timings was gateable —
only ``median_wall_s``, pooled across every arm, and that number is dominated
by whichever arm is cheapest.

That matters here more than elsewhere, because four of these five ops
(``append``, ``delete``, ``obs_import``, ``rollback``) commit through
``commit_in_place`` → ``finalize_header_with_checksum``, which streams offset
256 → EOF to recompute ``file_checksum`` regardless of how few bytes changed.
**``wall_s__rollback`` is the cleanest instrument for that in the suite**: a
rollback is one 4 KB pwrite plus two fsyncs plus the whole-file BLAKE3 rehash,
so essentially all of its wall is the checksum extent. ``wall_s__obs_import``
is the next cleanest — it adds one column and rehashes the file. Redefining the
extent to header + catalogs (OPT-FORMAT-1) should collapse both and leave
``wall_s__compact``, a genuine full rewrite, roughly where it is.
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
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)


# Fragment-ops is only implemented for SCX. ``scx_auto`` is chosen as the
# single trigger so we don't re-measure for every codec variant.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})
"""Format-key allow-list — read by ``run_parallel.py``'s cohort builder so
incompatible (bench, format) cells never get submitted. Mirrors the runtime
guard at the top of ``run()`` (defense-in-depth for direct invocation)."""

# Number of random cell indices to mark deleted. Capped by n_obs at runtime.
_DELETE_N = 10_000


def _gc() -> None:
    gc.collect()


def _copy_scx(src: Path, dst: Path) -> None:
    """Copy an SCX file atomically for a fresh per-run workspace."""
    shutil.copy2(src, dst)


def _time_op(fn, *args, **kwargs) -> tuple[float, float]:
    """Run *fn* and return ``(wall_s, peak_rss_mb)``.

    The RSS value is the high-water mark *while fn ran*, sampled by
    ``PeakRssSampler`` on a background thread. It was an end-of-op
    ``current_rss_mb()`` reading until the peak-sampler change, which could not
    see any of the ops measured here: ``append`` reads the whole input CSR
    before re-encoding
    and ``compact`` rewrites every section, both allocating and freeing a large
    transient that is gone by the time ``fn`` returns.

    ``PeakRssSampler`` seeds with the entry RSS, so anything this process is
    still holding from an earlier op is attributed to this one — hence the
    ``_gc()`` first.
    """
    _gc()
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    return wall, sampler.peak_mb


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
            **{"wall_s__append": round(wall, 6)},
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
            **{"wall_s__delete": round(wall, 6)},
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
            **{"wall_s__compact": round(wall, 6)},
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


def _run_compact_full(
    result: BenchmarkResult,
    full_scx: Path,
    workdir: Path,
    n_runs: int,
) -> None:
    """``pyscx.compact`` on a fixture that carries obsm keys and a layer.

    The `compact` arm above runs on the Phase-A `.scx`, which has neither: no
    source h5ad in the suite carries an `obsm` key or a layer, so
    `<name>_auto.scx` reports `obsm_keys == []` and `layer_names == []`, and a
    rewrite of one never reaches the code that copies them. This arm does.

    **It does not measure a `.raw` copy, even though the fixture has one.**
    `scx-ops`' carry table is explicit: under `compact`, `X | Layer | Obsm`
    are `RowFiltered` — carried — while `Raw` is `Dropped` with
    `warns: true` ("raw's obs axis is not filtered in lockstep with X (planned
    follow-up)"). So every run of this arm logs a raw-dropped warning, and the
    raw matrix contributes a read, not a resident copy. The export side is
    where raw's whole-matrix copy is measured
    (`export_streaming`'s `streaming_full` arm).

    Deliberately **not** dirtied with an append + delete first, the way the
    plain `compact` arm is. `append` refuses a file with a `.raw` outright —
    `scx-ops/src/append.rs:591` returns `OpsError::RawUnsupported` because it
    extends X's obs axis and not raw's — so that recipe is unavailable here.
    This arm times the rewrite itself, which is where the memory is.

    Emits `peak_rss_mb__compact_full` — sparse, so a threshold on it medians
    this arm alone and not the four ordinary ops. (Named for the house
    convention already used by `ooc_loader`, `ml_loader`, `cellset_gather` and
    `shuffle_layout` — `<metric>__<scenario>`. A `streaming_`-prefixed spelling
    would be actively misleading here: `compact` is a rewrite, not a streaming
    export, and that prefix belongs to `export_streaming`'s arms.)
    """
    import pyscx

    for i in range(n_runs):
        input_path = workdir / f"compact_full_in_{i}.scx"
        output_path = workdir / f"compact_full_out_{i}.scx"
        _copy_scx(full_scx, input_path)
        size_before = input_path.stat().st_size

        wall, rss = _time_op(pyscx.compact, str(input_path), str(output_path))

        size_after = output_path.stat().st_size
        throughput_mb_s = (size_before / (1024 * 1024)) / wall if wall > 0 else 0.0

        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            operation="compact_full",
            size_before_bytes=size_before,
            size_after_bytes=size_after,
            throughput_mb_s=round(throughput_mb_s, 3),
            **{
                "peak_rss_mb__compact_full": round(rss, 1),
                "wall_s__compact_full": round(wall, 6),
            },
        )
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        logger.info(
            "  compact_full run %d/%d: wall=%.3fs peak_rss=%.1f MB",
            i + 1, n_runs, wall, rss,
        )


def _obs_import_csv(base_scx: Path, csv_path: Path) -> tuple[int, str]:
    """Write a one-column annotation CSV keyed on *base_scx*'s obs index.

    Returns ``(n_rows, key_field)``. Setup only — outside every timed region.

    ``read_obs([])`` is the matrix-free projection ``obs_open.py`` uses; the
    projected frame keeps its barcode index by design, so an empty column list
    still yields the join keys. Measured 5.0 s at census_1m (1M rows).

    The CSV is written with an *unnamed* index column, which
    ``normalize_field_names`` renames to ``_index`` — the same physical field
    ``key="obs_names"`` resolves to on the target side. So one ``key=`` covers
    both sides and the two cannot desync, which is the point of joining by key
    string rather than row position.
    """
    import numpy as np
    import pandas as pd
    import pyscx

    exp = pyscx.open(str(base_scx))
    try:
        index = exp.read_obs([]).index.to_numpy()
    finally:
        close = getattr(exp, "close", None)
        if close is not None:
            close()

    rng = np.random.default_rng(RANDOM_SEED)
    frame = pd.DataFrame(
        {"synth_score": rng.random(len(index)).astype("float32")},
        index=pd.Index(index),
    )
    frame.to_csv(csv_path)
    return len(index), "obs_names"


def _run_obs_import(
    result: BenchmarkResult,
    base_scx: Path,
    workdir: Path,
    n_runs: int,
) -> None:
    """Measure ``pyscx.obs_import`` — a key-joined, in-place obs column add.

    This is the second in-place op with a *gateable* per-op wall in this
    module, and the one whose wall is almost entirely the file checksum:
    `attach_external_obs` never reads or rewrites X, layers, `var`, the CSC
    sidecar, `.raw`, deletion vectors or bitmaps, but `commit_in_place` still
    streams offset 256 -> EOF to recompute `file_checksum`. Adding one 8 MB
    column to census_1m therefore pays a 2.8 GB rehash.

    A fresh copy per run is required, not hygiene: the import is in place and
    `overwrite=False` (the default) refuses a column that already exists, so a
    second run against the same file would raise rather than re-measure.
    """
    import pyscx

    csv_path = workdir / "obs_import_source.csv"
    n_source_rows, key = _obs_import_csv(base_scx, csv_path)
    csv_bytes = csv_path.stat().st_size

    for i in range(n_runs):
        target_path = workdir / f"obs_import_target_{i}.scx"
        _copy_scx(base_scx, target_path)
        size_before = target_path.stat().st_size

        # Premise, untimed and checked before the measurement is recorded: a
        # join that matched nothing still writes an all-null column and still
        # pays the same whole-file rehash, so the wall would look exactly right
        # while measuring an import of nothing. `dry_run=True` reports the join
        # without writing.
        preview = pyscx.obs_import(
            str(target_path), str(csv_path), key=key, dry_run=True,
        )
        n_matched = int(preview.get("n_matched", -1))
        if n_matched != n_source_rows:
            raise RuntimeError(
                f"obs_import dry-run matched {n_matched} of {n_source_rows} "
                f"source rows on {target_path.name} (key={key!r}). The timed "
                f"import would rehash the whole file either way, so refusing "
                f"to record a wall for a join that did not land."
            )

        wall, rss = _time_op(
            pyscx.obs_import, str(target_path), str(csv_path), key=key,
        )

        size_after = target_path.stat().st_size
        rows_per_sec = n_source_rows / wall if wall > 0 else 0.0

        result.add_run(
            wall_s=wall,
            peak_rss_mb=rss,
            operation="obs_import",
            **{
                "wall_s__obs_import": round(wall, 6),
                "peak_rss_mb__obs_import": round(rss, 1),
            },
            rows_imported=n_source_rows,
            n_matched=n_matched,
            source_csv_bytes=csv_bytes,
            size_before_bytes=size_before,
            size_after_bytes=size_after,
            rows_per_sec=round(rows_per_sec, 1),
        )
        target_path.unlink(missing_ok=True)
        logger.info(
            "  obs_import run %d/%d: wall=%.3fs rows/s=%.0f (+%d bytes)",
            i + 1, n_runs, wall, rows_per_sec, size_after - size_before,
        )

    csv_path.unlink(missing_ok=True)


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
            # The sharpest OPT-FORMAT-1 signal in the suite: `rollback` is one
            # 4 KB pwrite plus `finalize_header_with_checksum`, which streams
            # offset 256 -> EOF. Almost all of this wall is the whole-file
            # BLAKE3 rehash, so an O(catalog) checksum extent would collapse it.
            **{"wall_s__rollback": round(wall, 6)},
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
    if format_variant.key not in SUPPORTED_FORMATS:
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
            "operations": [
                "append", "delete", "compact", "obs_import", "rollback",
            ],
        },
    )

    # The `.raw` + obsm + layer arm, when the fixture has been built. The skip
    # is recorded rather than silent — and it is loud at the gate too: a
    # threshold whose metric no run carries counts as a violation, not a skip,
    # so a missing fixture fails rather than quietly narrowing coverage.
    full_scx: Path | None = None
    if dataset.scx_full_path.exists():
        full_scx = dataset.scx_full_path
        result.metadata["operations"].append("compact_full")
        result.metadata["full_fixture_path"] = str(full_scx)
    else:
        result.metadata["compact_full_skipped_reason"] = (
            f"{dataset.scx_full_path} does not exist — build it with "
            f"`python benchmarks/scripts/prep_full_fixtures.py --datasets "
            f"{dataset.name}`"
        )
        logger.warning("%s", result.metadata["compact_full_skipped_reason"])
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

        if full_scx is not None:
            logger.info("compact_full: %s", dataset.name)
            _run_compact_full(result, full_scx, workdir, n_runs)

        logger.info("obs_import: %s", dataset.name)
        _run_obs_import(result, converted_path, workdir, n_runs)

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
        "Fragment-ops done: %s — %d runs across %d operations",
        dataset.name,
        len(result.runs),
        len(result.metadata["operations"]),
    )
    return result
