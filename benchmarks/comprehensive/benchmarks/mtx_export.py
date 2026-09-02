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
  plus ``mtx_rows_written`` on both arms so the row count is checkable
  (pbmc3k: 2700 -> 2673). Run once regardless of ``n_runs`` — see
  :data:`_DELETION_ARM_RUNS`.
* ``from_mtx`` — ingest the arm-1 output back to SCX. Emits
  ``wall_s__from_mtx`` and ``peak_rss_mb__from_mtx``, and asserts the round-trip
  shape.

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

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant, RANDOM_SEED
from benchmarks.comprehensive.results import BenchmarkResult, require_runs
from benchmarks.comprehensive.rss import PeakRssSampler

logger = logging.getLogger(__name__)

# MTX is an SCX-side conversion; `scx_auto` is the single trigger so the op is
# not re-measured per codec variant.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

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
# census does not. The skip is recorded in metadata, never silent.
MAX_N_OBS: int = 100_000

# The deletion arm exists to exercise the keep-mask path and check the row count
# it writes, not to produce a stable timing, and at tabula scale each run costs
# ~11 minutes. One run is the whole signal.
_DELETION_ARM_RUNS: int = 1


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


def _body_is_integral(mtx_dir: Path, max_lines: int = 200_000) -> bool:
    """True when every value in the (sampled) body parses as an integer.

    Guards the header gate from going vacuous: `integer` is the correct header
    only for a matrix whose values are integral, so if the fixture ever stops
    being one, that shows up here rather than as a threshold that passes for the
    wrong reason. Sampled — the point is to notice a fixture change, not to
    re-verify every non-zero.
    """
    with gzip.open(mtx_dir / "matrix.mtx.gz", "rt") as fh:
        for line in fh:  # banner + any %-comments + the dimensions line
            if not line.startswith("%"):
                break
        for i, line in enumerate(fh):
            if i >= max_lines:
                break
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


def _timed_to_mtx(scx_path: Path, out_dir: Path) -> tuple[float, float]:
    import pyscx

    gc.collect()
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.to_mtx(str(scx_path), str(out_dir))
    return time.perf_counter() - t0, sampler.peak_mb


def _timed_from_mtx(mtx_dir: Path, scx_out: Path) -> tuple[float, float]:
    import pyscx

    gc.collect()
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.from_mtx(str(mtx_dir), str(scx_out))
    return time.perf_counter() - t0, sampler.peak_mb


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

    workroot = tempfile.TemporaryDirectory(prefix=f"scx_mtx_{dataset.name}_")
    workdir = Path(workroot.name)
    try:
        # --- arm 1: export ---
        exported: Path | None = None
        for i in range(n_runs):
            out_dir = workdir / f"mtx_{i}"
            out_dir.mkdir()
            wall, peak = _timed_to_mtx(converted_path, out_dir)
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
        n_delete = max(1, int(dataset.n_obs * _DELETE_FRACTION))
        idx = rng.choice(dataset.n_obs, size=n_delete, replace=False)
        pyscx.mark_deleted(str(deleted_scx), [int(v) for v in idx])
        for i in range(_DELETION_ARM_RUNS):
            out_dir = workdir / f"mtx_del_{i}"
            out_dir.mkdir()
            wall, peak = _timed_to_mtx(deleted_scx, out_dir)
            rows = _row_count(out_dir)
            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak,
                **{
                    "scenario": "to_mtx_with_deletions",
                    "run_idx": i,
                    "wall_s__to_mtx_with_deletions": round(wall, 6),
                    "peak_rss_mb__to_mtx_with_deletions": round(peak, 1),
                    "mtx_rows_written": rows,
                    "n_deleted": n_delete,
                },
            )
            logger.info(
                "  to_mtx_with_deletions run %d/%d: wall=%.2fs peak=%.1f MB rows=%d",
                i + 1, _DELETION_ARM_RUNS, wall, peak, rows,
            )
            shutil.rmtree(out_dir)
        deleted_scx.unlink(missing_ok=True)

        # --- arm 3: ingest ---
        assert exported is not None
        for i in range(n_runs):
            scx_out = workdir / f"from_mtx_{i}.scx"
            wall, peak = _timed_from_mtx(exported, scx_out)
            exp = pyscx.open(str(scx_out))
            try:
                shape = (exp.n_obs, exp.n_vars)
                nnz = exp.nnz
            finally:
                close = getattr(exp, "close", None)
                if close is not None:
                    close()
            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak,
                **{
                    "scenario": "from_mtx",
                    "run_idx": i,
                    "wall_s__from_mtx": round(wall, 6),
                    "peak_rss_mb__from_mtx": round(peak, 1),
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
