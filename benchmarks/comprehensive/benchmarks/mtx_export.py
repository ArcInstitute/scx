"""MatrixMarket export / ingest — `pyscx.to_mtx` and `pyscx.from_mtx`.

## Why this exists

Neither direction had a benchmark, a threshold or an RSS metric. MTX is the
interchange format people leave SCX *through*, and the export currently
materialises: it is the shape most likely to be made streaming, and there was
nothing to measure the change against.

## Arms

* ``to_mtx`` — `pyscx.to_mtx(scx, out_dir)`, writing `matrix.mtx.gz` +
  `barcodes.tsv.gz` + `features.tsv.gz`. Emits ``wall_s__to_mtx`` and
  ``peak_rss_mb__to_mtx``.
* ``to_mtx_with_deletions`` — the same export after marking 1% of cells deleted,
  so the keep-mask path is measured rather than assumed equivalent. Emits
  ``peak_rss_mb__to_mtx_with_deletions`` and ``wall_s__to_mtx_with_deletions``,
  and **raises** if the row count is not ``n_obs - n_deleted`` (pbmc3k:
  2700 -> 2673). Exported once regardless of ``n_runs``: the arm checks a row
  count, not a timing, and at tabula scale each export is ~11 minutes.
* ``from_mtx`` — ingest the arm-1 output back to SCX. Emits
  ``wall_s__from_mtx`` and ``peak_rss_mb__from_mtx``, and **raises** if the
  re-imported shape or nnz differs from the source.

## The header gate

``mtx_header_integer`` is 1.0 when `matrix.mtx.gz` declares
``%%MatrixMarket matrix coordinate integer general`` and 0.0 for ``real``.

This is a **contract**, not a statistic. `scx-mtx/src/write.rs` decides the
declared type from the materialised `f32` values — `all_integer = data.iter()
.all(|&v| v == v.floor() && v.is_finite() && v >= 0.0)` — and formats the body
with the same flag, so a file of integral counts writes `integer` and `1`
rather than `real` and `1.0`. Deriving that flag from the shard's
*value encoding* instead would flip an h5ad-sourced file, whose counts are
integral but stored `Float32`, to `real`. Downstream tools read the header.

Two things this arm cannot do, stated rather than papered over:

* **It cannot verify the fixture is float-encoded.** No Python surface reports a
  shard's value encoding — `Experiment` exposes `codec_id` and `index_dtype` but
  not the value width, and `scx info --json` does not carry it either. So this
  arm cannot distinguish "integer-encoded" from "float-encoded but integral",
  and if the fixture happened to be integer-encoded the gate would be weaker
  than intended. It still fails if the header flips, which is the regression
  that matters.
* **A header check is vacuous on a matrix that is not integral.**
  ``mtx_values_all_integral`` is emitted beside it, computed from the exported
  body, so a fixture that stops being integral shows up as a premise change
  rather than as a quietly passing gate.

The pair was checked against a fixture that *should* fail it. `pbmc3k` (raw
counts, float32, min 1.0 max 419.0) gives
``%%MatrixMarket matrix coordinate integer general`` and 1.0/1.0;
``pbmc3k_lognorm`` — same shape, same 2,286,884 non-zeros, log-normalised —
gives ``… coordinate real general`` and 0.0/0.0. So the header is re-derived per
file rather than fixed, and a threshold on it is not measuring a constant.
"""

from __future__ import annotations

import gc
import gzip
import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.config import (
    DATASETS,
    DatasetConfig,
    FormatVariant,
    RANDOM_SEED,
)
from benchmarks.comprehensive.results import BenchmarkResult, require_runs
from benchmarks.comprehensive.rss import PeakRssSampler, current_rss_mb

logger = logging.getLogger(__name__)

# MTX is an SCX-side conversion; `scx_auto` is the single trigger so the op is
# not re-measured per codec variant.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

# Per-format dataset allow-list, read by `run_parallel._bench_format_dataset_scope`
# at **cohort-build time**.
#
# Declaring `SUPPORTED_FORMATS` alone was not enough: the planner accepted every
# unimodal `scx_auto` dataset and treated this benchmark as conversion-dependent,
# so a `--tier full` run submitted `mtx_export x {census_500k, census_1m}` — 88 GB
# and 176 GB of requested memory plus their Phase-A conversions — and `run()`
# returned `None` on its first line. `--tier xl` added census_5m at 864 GB. Those
# are real queue slots for work that is discarded.
#
# This is the trap `_bench_format_dataset_scope`'s own docstring records ("they
# stubbed out-of-scope datasets *inside* `run()` ... eight wasted GPU jobs per
# full accel capture"). Populated from `MAX_N_OBS` below, so the scope and the
# runtime skip cannot drift apart.

# Fraction of cells marked deleted for the keep-mask arm.
_DELETE_FRACTION = 0.01

# Above this many cells the MTX arms are skipped. MTX is a gzipped *text*
# triplet and neither direction streams, so both are wall-clock bound by nnz
# with nothing to amortise it. Measured on pbmc3k (2,286,884 nnz): `to_mtx`
# 7.97 s = 287k nnz/s writing 7,750,747 B (3.4 B/nnz gzipped), `from_mtx`
# 5.08 s = 450k nnz/s. Extrapolated:
#
#   tabula_sapiens_100k   195M nnz   ~11 min per export, ~7 min per ingest
#   census_500k           ~1B nnz    ~an hour per export
#
# 100,000 is therefore the line: tabula runs (and is the smallest fixture where
# the peak means anything — pbmc3k's 466 MB is mostly interpreter baseline),
# census does not. `FORMAT_DATASET_SCOPE` above stops the orchestrator
# submitting an out-of-scope cell at all; `_skip_reason` remains as the
# defence-in-depth path for a direct `run()` call, and logs its reason.
MAX_N_OBS: int = 100_000

# Derived, never hand-listed: every unimodal dataset at or below the cap. A
# literal list would silently stop matching `MAX_N_OBS` the first time either
# changed.
FORMAT_DATASET_SCOPE: dict[str, frozenset[str]] = {
    "scx_auto": frozenset(
        name
        for name, ds in DATASETS.items()
        if not ds.multimodal and ds.n_obs <= MAX_N_OBS
    ),
}


def _skip_reason(n_obs: int) -> str | None:
    if n_obs > MAX_N_OBS:
        return (
            f"MTX is a gzipped text triplet and neither direction streams, so "
            f"both are O(nnz) with nothing to amortise it — measured 287k "
            f"non-zeros per second on export, which is about an hour per export "
            f"at census_500k's 747M. Skipped above n_obs={MAX_N_OBS:,}."
        )
    return None


def _read_header(mtx_dir: Path) -> str:
    """The `%%MatrixMarket` banner line of the exported matrix."""
    with gzip.open(mtx_dir / "matrix.mtx.gz", "rt") as fh:
        return fh.readline().strip()


def _body_is_integral(mtx_dir: Path) -> bool:
    """True when **every** value in the body parses as an integer.

    Guards the header gate from going vacuous: `integer` is the correct header
    only for a matrix whose values are integral, so if the fixture ever stops
    being one, that shows up here rather than as a threshold that passes for the
    wrong reason.

    Reads the whole body rather than the first N lines. An earlier version
    stopped after 200,000 entries, which is worse than it sounds: this MTX is
    column-major (`write_matrix_mtx` emits `col+1 row+1 val`), so a prefix is the
    *first genes*, and a matrix that is integral up front and not later would
    keep the header gate looking live. The scan is cheap beside the export it
    follows — 2.3M text lines against an 8 s gzip write on pbmc3k.
    """
    with gzip.open(mtx_dir / "matrix.mtx.gz", "rt") as fh:
        for line in fh:  # banner + any %-comments + the dimensions line
            if not line.startswith("%"):
                break
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                return False
            try:
                int(parts[2])
            except ValueError:
                return False
    return True


def _row_count(mtx_dir: Path) -> int:
    """Rows declared by the dimensions line (cells, after any deletion)."""
    with gzip.open(mtx_dir / "matrix.mtx.gz", "rt") as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            # MTX is column-major here: `n_vars n_obs nnz`, matching what
            # `write_matrix_mtx` emits (it writes `col+1 row+1 val`).
            return int(line.split()[1])
    raise RuntimeError(f"{mtx_dir}/matrix.mtx.gz has no dimensions line")


def _timed_to_mtx(scx_path: Path, out_dir: Path) -> tuple[float, float, float]:
    """`(wall_s, peak_rss_mb, entry_rss_mb)`.

    The entry reading is taken **here**, after `gc.collect()` and immediately
    before `PeakRssSampler` seeds itself, so `peak - entry` is the op's own
    allocation against the same baseline the sampler used. Taking it in the
    caller instead — before this function's `gc.collect()` — compares a pre-GC
    number against a post-GC seed and can go negative. `build_csc` gets this
    ordering right; this module did not until review caught the asymmetry.
    """
    import pyscx

    gc.collect()
    entry = current_rss_mb()
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.to_mtx(str(scx_path), str(out_dir))
    return time.perf_counter() - t0, sampler.peak_mb, entry


def _timed_from_mtx(mtx_dir: Path, scx_out: Path) -> tuple[float, float, float]:
    """`(wall_s, peak_rss_mb, entry_rss_mb)` — see :func:`_timed_to_mtx`."""
    import pyscx

    gc.collect()
    entry = current_rss_mb()
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.from_mtx(str(mtx_dir), str(scx_out))
    return time.perf_counter() - t0, sampler.peak_mb, entry


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Time both MTX directions for one dataset."""
    if format_variant is None or format_variant.key not in SUPPORTED_FORMATS:
        return None
    if dataset.multimodal:
        return None

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted SCX file for {dataset.name}. "
            f"Run Phase A conversion first (--formats scx_auto)."
        )

    # Checked before building the result: `run()` returning None is the
    # harness's skip protocol and discards any result object, so metadata
    # attached to one here would go nowhere. The reason goes to the job log.
    skip = _skip_reason(dataset.n_obs)
    if skip:
        logger.info("mtx_export skipped for %s: %s", dataset.name, skip)
        return None

    import pyscx

    converted_path = Path(converted_path)
    result = BenchmarkResult(
        benchmark="mtx_export",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "n_obs": dataset.n_obs,
            "n_vars": dataset.n_vars,
            "delete_fraction": _DELETE_FRACTION,
        },
    )
    result.file_size_bytes = converted_path.stat().st_size

    # Read once, from the file rather than from `DatasetConfig`: the round-trip
    # assertion below has to compare against what was exported, and a config
    # figure that drifted from the fixture would make it compare two guesses.
    src = pyscx.open(str(converted_path))
    try:
        source_shape = (src.n_obs, src.n_vars)
        source_nnz = src.nnz
    finally:
        _close = getattr(src, "close", None)
        if _close is not None:
            _close()

    workroot = tempfile.TemporaryDirectory(prefix=f"scx_mtx_{dataset.name}_")
    workdir = Path(workroot.name)
    try:
        # --- arm 1: export ---
        exported: Path | None = None
        for i in range(n_runs):
            out_dir = workdir / f"mtx_{i}"
            out_dir.mkdir()
            wall, peak, entry_rss = _timed_to_mtx(converted_path, out_dir)
            mtx_bytes = sum(p.stat().st_size for p in out_dir.iterdir())
            header = _read_header(out_dir)
            header_integer = 1.0 if " integer " in f" {header} " else 0.0
            integral = 1.0 if _body_is_integral(out_dir) else 0.0
            rows = _row_count(out_dir)

            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak,
                **{
                    "scenario": "to_mtx",
                    "run_idx": i,
                    "wall_s__to_mtx": round(wall, 6),
                    "peak_rss_mb__to_mtx": round(peak, 1),
                    "entry_rss_mb": round(entry_rss, 1),
                    "delta_rss_mb__to_mtx": round(peak - entry_rss, 1),
                    "mtx_bytes": mtx_bytes,
                    "mtx_rows_written": rows,
                    "mtx_header": header,
                    # The contract, and the premise that keeps it non-vacuous.
                    "mtx_header_integer": header_integer,
                    "mtx_values_all_integral": integral,
                },
            )
            logger.info(
                "  to_mtx run %d/%d: wall=%.2fs peak=%.1f MB bytes=%d rows=%d "
                "header=%r integral=%s",
                i + 1, n_runs, wall, peak, mtx_bytes, rows, header, bool(integral),
            )
            if exported is None:
                exported = out_dir  # keep the first for the from_mtx arm
            else:
                shutil.rmtree(out_dir)

        # --- arm 2: export with a deletion vector applied ---
        deleted_scx = workdir / "deleted.scx"
        shutil.copy2(converted_path, deleted_scx)
        rng = np.random.default_rng(RANDOM_SEED)
        # `source_shape[0]`, not `dataset.n_obs`: the round-trip check above
        # already decided the config figure is not what to compare against, and
        # the same reasoning applies harder here — `mark_deleted` indexes the
        # file's obs space, so a config that drifted from the fixture would
        # delete the wrong rows and then compare the result to the drifted
        # number. Found by review (Cursor Agent, Antigravity).
        source_n_obs = source_shape[0]
        n_delete = max(1, int(source_n_obs * _DELETE_FRACTION))
        idx = rng.choice(source_n_obs, size=n_delete, replace=False)
        pyscx.mark_deleted(str(deleted_scx), [int(v) for v in idx])
        # Exported once, not `n_runs` times: this arm checks the row count the
        # keep-mask path writes, not a stable timing, and at tabula scale each
        # export costs ~11 minutes.
        out_dir = workdir / "mtx_del"
        out_dir.mkdir()
        wall, peak, entry_rss = _timed_to_mtx(deleted_scx, out_dir)
        rows = _row_count(out_dir)
        # Compared, not merely recorded. An export that ignored the deletion
        # vector would otherwise write a normal successful result — the exact
        # assumption this arm was added to stop making.
        expected_rows = source_n_obs - n_delete
        if rows != expected_rows:
            raise RuntimeError(
                f"to_mtx wrote {rows} rows after marking {n_delete} of "
                f"{source_n_obs} cells deleted; expected {expected_rows}. "
                f"The keep mask was not applied."
            )
        result.add_run(
            wall_s=wall,
            peak_rss_mb=peak,
            **{
                "scenario": "to_mtx_with_deletions",
                "run_idx": 0,
                "wall_s__to_mtx_with_deletions": round(wall, 6),
                "peak_rss_mb__to_mtx_with_deletions": round(peak, 1),
                "entry_rss_mb": round(entry_rss, 1),
                "delta_rss_mb__to_mtx_with_deletions": round(peak - entry_rss, 1),
                "mtx_rows_written": rows,
                "n_deleted": n_delete,
            },
        )
        logger.info(
            "  to_mtx_with_deletions: wall=%.2fs peak=%.1f MB (delta %.1f) rows=%d",
            wall, peak, peak - entry_rss, rows,
        )
        shutil.rmtree(out_dir)
        deleted_scx.unlink(missing_ok=True)

        # --- arm 3: ingest ---
        assert exported is not None
        for i in range(n_runs):
            scx_out = workdir / f"from_mtx_{i}.scx"
            wall, peak, entry_rss = _timed_from_mtx(exported, scx_out)
            exp = pyscx.open(str(scx_out))
            try:
                shape = (exp.n_obs, exp.n_vars)
                nnz = exp.nnz
            finally:
                close = getattr(exp, "close", None)
                if close is not None:
                    close()
            # Compared, not merely recorded. `to_mtx` -> `from_mtx` is lossless
            # by contract; an ingest that dropped entries or transposed the
            # matrix would otherwise write a normal-looking successful result.
            if (shape, nnz) != (source_shape, source_nnz):
                raise RuntimeError(
                    f"MTX round-trip changed the matrix: got shape {shape} "
                    f"nnz={nnz}, source is {source_shape} nnz={source_nnz}."
                )
            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak,
                **{
                    "scenario": "from_mtx",
                    "run_idx": i,
                    "wall_s__from_mtx": round(wall, 6),
                    "peak_rss_mb__from_mtx": round(peak, 1),
                    "entry_rss_mb": round(entry_rss, 1),
                    "delta_rss_mb__from_mtx": round(peak - entry_rss, 1),
                    "roundtrip_n_obs": shape[0],
                    "roundtrip_n_vars": shape[1],
                    "roundtrip_nnz": nnz,
                },
            )
            logger.info(
                "  from_mtx run %d/%d: wall=%.2fs peak=%.1f MB -> %s nnz=%d",
                i + 1, n_runs, wall, peak, shape, nnz,
            )
            scx_out.unlink(missing_ok=True)
    finally:
        workroot.cleanup()

    require_runs(result, str(converted_path))
    return result
