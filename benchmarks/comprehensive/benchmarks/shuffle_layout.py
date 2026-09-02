"""Global pre-shuffle benchmark — ``scx sort --shuffle`` / ``pyscx.shuffle`` (data-load Phase 1D).

``TrainingDataset`` randomizes in two levels — shard order, then a Fisher-Yates
shuffle *within* a shard group — so on a file whose rows arrived clustered (by
donor, plate, or cell type) batch composition is capped by ``shard_group_size``,
and widening it costs memory linearly. A seeded global permutation moves that
cost off the training loop, once, onto disk.

Four arms:

  * ``shuffle_write``    — cost of the rewrite itself (wall, peak RSS, cells/s).
  * ``size_delta``       — output size against the input, **swept across the
    on-disk codec variants**. This is the arm that answers the open question:
    ``scx1`` codes each row's gene indices independently of row order and should
    be size-neutral under a permutation, while ``zstd`` / ``shufdelta`` compress
    the shard byte stream as a whole and should grow. Sorting a 149M-cell file
    by ``cell_type`` already grew X ~8%; a random permutation is the worst case,
    not the best. *Measure it, don't assume it.*
  * ``train_throughput`` — cache-cold ``TrainingDataset`` epoch, shuffled vs not.
  * ``shuffle_quality``  — how mixed the row order actually is, analytically.

**Read ``train_throughput`` as a neutrality check, not a speedup.** A shuffled
file is still read sequentially, so the expected result is ~1.00×; that is the
arm passing, not the arm finding nothing. What would be a *finding* is a
regression — evidence that the permutation cost the sequential read something.
The throughput win a pre-shuffle buys is downstream (a consumer can drop
``shard_group_size`` and its memory) and is not measured here.

``shuffle_quality`` needs no X decode. Label codes come from ``obs_categorical``
(the 1C accessor), and batch composition is determined by the row order plus the
shard geometry: at ``shard_group_size=1`` a group *is* a shard, so per-block
total-variation distance from the global label distribution is exact with no RNG
at all. The ``sgs8`` figure Monte-Carlos over random 8-block pools, which is
faithful because the loader's level-1 shard permutation is uniform.

Per-run ``extra`` keys (sparse per-scenario, so one ``thresholds.yaml`` entry
targets one arm):
    ``cells_per_sec__shuffle_write``     — rewrite throughput
    ``output_size_bytes``                — shuffled file size
    ``file_size_ratio__<codec>``         — whole-file after/before
    ``x_size_ratio__<codec>``            — X-section after/before (needs the CLI)
    ``samples_per_sec__train_{unshuffled,shuffled}``
    ``epoch_wall_s__train_{unshuffled,shuffled}``
    ``peak_rss_mb__train_{unshuffled,shuffled}``
    ``label_tv_block__{before,after}``   — lower = better mixed
    ``label_tv_sgs8__{before,after}``
    ``cache_policy``                     — ``cold_fadvise`` / ``warm``

SCX-only (``SUPPORTED_FORMATS = {"scx_auto"}``); every other format variant
returns ``None`` so the orchestrator skips it. Listed in
``run_parallel.py::_NO_CONVERSION`` because it builds its own outputs.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from benchmarks.comprehensive.cache_control import drop_file_cache
from benchmarks.comprehensive.config import (
    PROJECT_ROOT,
    DatasetConfig,
    FormatVariant,
)
from benchmarks.comprehensive.scx_cli import INFO_JSON_PROBE, resolve_scx_bin
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler, current_rss_mb
from benchmarks.comprehensive.scx_cli import INFO_JSON_PROBE, resolve_scx_bin

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

#: Seed for every shuffle here. Matches `TrainingDataset`'s default and
#: `pyscx.shuffle`'s, so the capture measures what a user gets by default.
SHUFFLE_SEED = 42

#: Batch size for the throughput arm — the suite-wide ML batch size, so the
#: numbers sit next to `ooc_loader`'s.
_BATCH_SIZE = 1024

#: obs column used for the quality metric, per dataset. Absent → the quality arm
#: records `applicable: False` with a reason rather than inventing a column.
LABEL_SPEC: dict[str, str] = {
    "tabula_sapiens_100k": "cell_type",
    "census_500k": "cell_type",
    "census_1m": "cell_type",
    "census_5m": "cell_type",
    "smartseq2": "cell_type",
}

#: Codec variants swept by the size arm. `scx_auto` first so it is measured even
#: if a later variant is missing on disk.
_SIZE_CODEC_KEYS = (
    "scx_auto",
    "scx_scx1",
    "scx_zstd",
    "scx_shufdelta",
    "scx_lz4",
    "scx_pcodec",
)

#: A full rewrite of a census-scale file is minutes; three reps buy no signal
#: the median of two does not, and the arm is deterministic anyway.
_MAX_WRITE_RUNS = 2
_MAX_EPOCH_RUNS = 3

#: Monte-Carlo trials for the `sgs8` pooled-block figure.
_SGS8_TRIALS = 200


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_scx_bin() -> str | None:
    """An `scx` binary whose `info --json` works, or `None`.

    Only the *X-section* size metric needs it; whole-file sizes come from
    `stat()`, so an absent CLI narrows this arm rather than skipping it.

    The `$SCX_CLI_BIN` -> `target/release/scx` -> PATH walk lives in
    `benchmarks.comprehensive.scx_cli`, shared with
    `accel_to_gpu_anndata` (which needs `optimize --codec`) and
    `cloud_metadata` (which needs a `--features cloud` build). This wrapper
    stays because the *probe* is what differs between the three, and naming it
    here is what keeps that visible at the call site.
    """
    return resolve_scx_bin(INFO_JSON_PROBE)


def _geometry(scx_bin: str | None, path: Path) -> dict[str, int | str | None]:
    """`{shard_target_rows, n_csr_shards, x_bytes, x_bytes_source}` for a file.

    Two things here are load-bearing rather than cosmetic:

    **`shard_target_rows` must be carried into the shuffle.** Left at the
    writer's default, shuffling a 6-shard 600-row fixture emits *one* shard, and
    then every arm is measuring a shard-geometry change instead of a row-order
    change: the size ratio compares 6 shards against 1, and the throughput arm
    reported a spurious 5.97x purely from the reduced per-shard overhead. The
    only variable that may differ between the two files is row order.

    **X-section bytes, not whole-file bytes.** obs, var and the predicate index
    all change under a reorder too; `docs/performance.md` had to rule each out by
    hand when the 149M-cell sort grew. Isolating X is what makes the claim about
    *compression* rather than about bookkeeping.

    Falls back to `ceil(n_obs / shard_count)` for the geometry when no CLI is
    available — same shard count, boundaries off by up to one shard's rounding —
    and to `None` for `x_bytes`.
    """
    import pyscx

    exp = pyscx.open(str(path))
    n_shards = max(1, int(exp.shard_count))
    out: dict[str, object] = {
        "shard_target_rows": max(1, math.ceil(int(exp.n_obs) / n_shards)),
        "n_csr_shards": n_shards,
        "x_bytes": None,
        "x_bytes_source": "unavailable",
        "codec_breakdown": None,
    }
    if scx_bin is None:
        out["geometry_source"] = "derived (ceil(n_obs / shard_count))"
        return out
    try:
        proc = subprocess.run(
            [scx_bin, "info", "--json", str(path)],
            capture_output=True,
            timeout=600,
            check=True,
        )
        info = json.loads(proc.stdout)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("  geometry: `scx info --json` failed on %s: %s", path, exc)
        out["geometry_source"] = "derived (ceil(n_obs / shard_count))"
        return out

    out["x_bytes"] = sum(
        int(s["length"]) for s in info.get("sections", []) if str(s.get("type", "")) == "X/csr"
    )
    out["x_bytes_source"] = "scx info --json"
    # Per-shard codec histogram. This is what separates "the permutation cost
    # this codec compression" from "the adaptive `auto` heuristic picked a
    # different codec for the reordered shard" — two very different findings
    # that a size ratio alone cannot tell apart. `docs/performance.md`'s sort
    # section had to reason about exactly this flip without the data.
    out["codec_breakdown"] = info.get("codec_breakdown")
    if info.get("shard_target_rows"):
        out["shard_target_rows"] = int(info["shard_target_rows"])
    out["geometry_source"] = "scx info --json"
    return out


def _shuffle(
    src: Path,
    dst: Path,
    shard_size: int | None,
    codec: str = "auto",
    seed: int = SHUFFLE_SEED,
) -> None:
    import pyscx

    pyscx.shuffle(str(src), str(dst), seed=seed, shard_size=shard_size, codec=codec)


def _label_codes(path: Path, column: str) -> np.ndarray | None:
    """Integer label codes in physical obs row order, via the 1C accessor.

    Returns `None` when the column is absent. Note `obs_categorical` is indexed
    in *physical* row space; both files here are deletion-free (a shuffle
    materializes deletions away), so physical == logical.
    """
    import pyscx

    exp = pyscx.open(str(path))
    try:
        codes, _cats = exp.obs_categorical(column)
    except Exception as exc:
        logger.info("  obs_categorical(%s) unavailable on %s: %s", column, path.name, exc)
        return None
    return np.asarray(codes)


def _tv_from_counts(counts: np.ndarray, global_p: np.ndarray) -> float:
    """Total-variation distance between an empirical block and the global mix."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    return float(0.5 * np.abs(counts / total - global_p).sum())


def _label_mixing(codes: np.ndarray, block: int, n_levels: int) -> tuple[float, float | None]:
    """`(mean per-block TV, mean pooled-8-block TV)`.

    Lower is better mixed. A perfectly clustered file approaches 1.0 (each block
    holds one label while the corpus holds many); a uniformly shuffled file sits
    at the sampling-noise floor for the block size.

    The `sgs8` figure is `None` when the file has 8 or fewer shards: pooling
    every block *is* the corpus, so the answer is 0.0 by construction and would
    read as "perfectly mixed" on a maximally clustered file. Observed on a
    6-shard fixture, where it reported 0.0 both before and after.
    """
    # `-1` is the pandas null code; drop it rather than treat it as a level.
    valid = codes[codes >= 0]
    if valid.size == 0 or n_levels <= 1:
        return 0.0, None
    global_counts = np.bincount(valid, minlength=n_levels).astype(np.float64)
    global_p = global_counts / global_counts.sum()

    n_blocks = math.ceil(codes.size / block)
    per_block = np.zeros((n_blocks, n_levels), dtype=np.float64)
    for b in range(n_blocks):
        chunk = codes[b * block : (b + 1) * block]
        chunk = chunk[chunk >= 0]
        if chunk.size:
            per_block[b] = np.bincount(chunk, minlength=n_levels)

    tv_block = float(np.mean([_tv_from_counts(per_block[b], global_p) for b in range(n_blocks)]))

    if n_blocks <= 8:
        return tv_block, None

    # sgs=8: the loader pools 8 shards per group and the level-1 permutation is
    # uniform, so averaging over random 8-subsets is faithful. Seeded so the
    # figure is reproducible across captures.
    rng = np.random.default_rng(0xC0FFEE)
    tv_sgs8 = float(
        np.mean(
            [
                _tv_from_counts(per_block[rng.choice(n_blocks, 8, replace=False)].sum(0), global_p)
                for _ in range(_SGS8_TRIALS)
            ]
        )
    )
    return tv_block, tv_sgs8


def _scx_epoch(path: Path, batch_size: int) -> tuple[int, int]:
    """One full `TrainingDataset` epoch; returns `(n_batches, n_cells)`."""
    import pyscx

    ds = pyscx.TrainingDataset(str(path), batch_size=batch_size, seed=SHUFFLE_SEED)
    n_batches = n_cells = 0
    for batch in ds:
        n_batches += 1
        n_cells += batch["X"].shape[0]
    return n_batches, n_cells


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def _arm_shuffle_write(
    result: BenchmarkResult, src: Path, workdir: Path, n_runs: int, shard_size: int | None
) -> Path | None:
    """Time the rewrite. Returns the surviving shuffled file for later arms."""
    import pyscx

    n_obs = int(pyscx.open(str(src)).n_obs)
    keep: Path | None = None
    for i in range(n_runs):
        out = workdir / f"shuffled_{i}.scx"
        gc.collect()
        with PeakRssSampler() as sampler:
            t0 = time.perf_counter()
            _shuffle(src, out, shard_size)
            wall = time.perf_counter() - t0
        result.add_run(
            wall_s=wall,
            peak_rss_mb=sampler.peak_mb,
            scenario="shuffle_write",
            cache_policy="warm",
            n_obs=n_obs,
            output_size_bytes=out.stat().st_size,
            **{
                "cells_per_sec__shuffle_write": round(n_obs / wall, 1) if wall > 0 else 0.0,
                "peak_rss_mb__shuffle_write": round(sampler.peak_mb, 1),
            },
        )
        logger.info(
            "  shuffle_write %d/%d: wall=%.2fs rss=%.0fMB out=%.1fMB",
            i + 1,
            n_runs,
            wall,
            sampler.peak_mb,
            out.stat().st_size / 1e6,
        )
        if keep is not None:
            keep.unlink(missing_ok=True)
        keep = out
    return keep


def _arm_size_delta(
    result: BenchmarkResult, dataset: DatasetConfig, workdir: Path, scx_bin: str | None
) -> None:
    """Shuffle each on-disk codec variant and record the size delta.

    Single-shot per variant: the output is byte-deterministic at a fixed seed,
    so repeats measure nothing. Each output is deleted immediately — the sweep
    would otherwise hold six copies of a census-scale file.
    """
    measured: dict[str, dict[str, float | int | None]] = {}
    skipped: dict[str, str] = {}

    for key in _SIZE_CODEC_KEYS:
        codec = key.removeprefix("scx_")
        try:
            path = dataset.path_for_format(key)
        except ValueError:
            skipped[codec] = "no path mapping for this format key"
            continue
        if not path.exists():
            skipped[codec] = "fixture not built on this host"
            continue

        out = workdir / f"size_{codec}.scx"
        # Shuffle at the *input's* shard geometry. Without this the writer's
        # default applies and the ratio compares a different shard count, not a
        # different row order.
        geo_in = _geometry(scx_bin, path)
        try:
            # Pin the variant's own codec. Left at `auto` the writer re-selects
            # per shard, so every variant produced a byte-identical output and
            # the "sweep" measured auto-reselection instead of whether a
            # permutation grows *that* codec. Observed on pbmc10k: all six
            # variants landed on exactly 37,703,532 X bytes.
            _shuffle(path, out, geo_in["shard_target_rows"], codec=codec)
        except Exception as exc:
            skipped[codec] = f"shuffle failed: {exc}"
            out.unlink(missing_ok=True)
            continue

        geo_out = _geometry(scx_bin, out)
        before_file, after_file = path.stat().st_size, out.stat().st_size
        before_x, after_x = geo_in["x_bytes"], geo_out["x_bytes"]
        shards_match = geo_in["n_csr_shards"] == geo_out["n_csr_shards"]
        out.unlink(missing_ok=True)

        entry: dict[str, float | int | None] = {
            "file_bytes_before": before_file,
            "file_bytes_after": after_file,
            "file_size_ratio": round(after_file / before_file, 4) if before_file else None,
            "x_bytes_before": before_x,
            "x_bytes_after": after_x,
            "x_size_ratio": (
                round(after_x / before_x, 4) if before_x and after_x is not None else None
            ),
            "n_csr_shards_before": geo_in["n_csr_shards"],
            "n_csr_shards_after": geo_out["n_csr_shards"],
            "codec_breakdown_before": geo_in["codec_breakdown"],
            "codec_breakdown_after": geo_out["codec_breakdown"],
            # `auto` is the only variant that may re-select; a flip here is the
            # explanation for its ratio, and its absence rules the flip out.
            "codec_flipped": (
                geo_in["codec_breakdown"] != geo_out["codec_breakdown"]
                if geo_in["codec_breakdown"] is not None
                and geo_out["codec_breakdown"] is not None
                else None
            ),
            # Surfaced rather than asserted: a mismatch means the ratio is
            # confounded by shard count and the number should not be quoted.
            "shard_geometry_matched": shards_match,
        }
        if not shards_match:
            logger.warning(
                "  size_delta %s: shard count changed %s -> %s; the ratio is "
                "confounded by geometry, not just row order",
                codec,
                geo_in["n_csr_shards"],
                geo_out["n_csr_shards"],
            )
        measured[codec] = entry
        logger.info(
            "  size_delta %-9s file %.1f -> %.1f MB (%.3fx)%s",
            codec,
            before_file / 1e6,
            after_file / 1e6,
            entry["file_size_ratio"] or 0.0,
            f"  X {entry['x_size_ratio']:.3f}x" if entry["x_size_ratio"] else "",
        )

        extra: dict[str, float] = {f"file_size_ratio__{codec}": entry["file_size_ratio"]}
        if entry["x_size_ratio"] is not None:
            extra[f"x_size_ratio__{codec}"] = entry["x_size_ratio"]
        # Zero-wall record: this arm has no timing, but the gate only reads
        # `runs[].extra`, so a metric with no run is a metric the gate cannot
        # see (`grouped_sort` uses the same trick for its correctness flag).
        result.add_run(wall_s=0.0, peak_rss_mb=0.0, scenario=f"size_{codec}", **extra)

    result.metadata["size_delta"] = {
        "applicable": bool(measured),
        "reason": None if measured else "no codec fixtures present for this dataset",
        "x_bytes_source": "scx info --json" if scx_bin else "unavailable (whole-file only)",
        "measured": measured,
        "skipped": skipped,
    }
    if not scx_bin:
        logger.warning(
            "  size_delta: no `scx` binary with `info --json` — recording whole-file "
            "ratios only. Whole-file confounds obs/var/predicate-index changes with "
            "X re-compression; set $SCX_CLI_BIN for the X-only figure."
        )


def _arm_train_throughput(
    result: BenchmarkResult,
    src: Path,
    shuffled: Path,
    n_runs: int,
    cold_cache: bool,
) -> None:
    """Cache-cold `TrainingDataset` epoch on both files.

    Expected ~1.00×: a shuffled file streams the same way. Recorded so a
    *regression* would show up, not because a win is anticipated here.
    """
    for scenario, path in (("train_unshuffled", src), ("train_shuffled", shuffled)):
        # Warm code (tokio/rayon) with a tiny pass, then drop caches, so the
        # timed epochs are cold but the runtime is not paying first-call costs.
        try:
            import pyscx

            it = iter(pyscx.TrainingDataset(str(path), batch_size=_BATCH_SIZE, seed=SHUFFLE_SEED))
            next(it, None)
            del it
        except Exception as exc:
            logger.warning("  %s warmup failed: %s", scenario, exc)

        for i in range(n_runs):
            cache_policy = drop_file_cache(path) if cold_cache else "warm"
            gc.collect()
            try:
                with PeakRssSampler() as sampler:
                    t0 = time.perf_counter()
                    n_batches, n_cells = _scx_epoch(path, _BATCH_SIZE)
                    wall = time.perf_counter() - t0
            except Exception as exc:
                logger.warning("  %s run %d failed: %s", scenario, i + 1, exc)
                continue
            if wall <= 0 or n_cells == 0:
                continue
            result.add_run(
                wall_s=wall,
                peak_rss_mb=sampler.peak_mb,
                scenario=scenario,
                cache_policy=cache_policy,
                n_cells=n_cells,
                n_batches=n_batches,
                **{
                    f"samples_per_sec__{scenario}": round(n_cells / wall, 1),
                    f"epoch_wall_s__{scenario}": round(wall, 4),
                    f"peak_rss_mb__{scenario}": round(sampler.peak_mb, 1),
                },
            )
            logger.info(
                "  %s %d/%d: %.1f samples/s (%.2fs, %s)",
                scenario,
                i + 1,
                n_runs,
                n_cells / wall,
                wall,
                cache_policy,
            )

    _record_throughput_ratio(result)


def _record_throughput_ratio(result: BenchmarkResult) -> None:
    def median_of(key: str) -> float | None:
        vals = [r.extra[key] for r in result.runs if key in r.extra]
        return statistics.median(vals) if vals else None

    before = median_of("samples_per_sec__train_unshuffled")
    after = median_of("samples_per_sec__train_shuffled")
    result.metadata["train_throughput"] = {
        "applicable": before is not None and after is not None,
        "reason": None
        if before is not None and after is not None
        else "one or both epochs failed to produce a timed run",
        "unshuffled_samples_per_sec": before,
        "shuffled_samples_per_sec": after,
        "ratio": round(after / before, 4) if before and after else None,
        "expectation": (
            "~1.00x. A shuffled file is still read sequentially, so neutrality is "
            "the pass condition; a ratio well below 1 would mean the permutation "
            "cost the sequential read something."
        ),
    }


def _arm_shuffle_quality(
    result: BenchmarkResult, dataset: DatasetConfig, src: Path, shuffled: Path
) -> None:
    """How mixed the row order is, before and after. No X decode."""
    column = LABEL_SPEC.get(dataset.name)
    if column is None:
        result.metadata["shuffle_quality"] = {
            "applicable": False,
            "reason": f"no label column configured for {dataset.name} in LABEL_SPEC",
        }
        logger.info("  shuffle_quality not applicable: no LABEL_SPEC entry")
        return

    before_codes = _label_codes(src, column)
    after_codes = _label_codes(shuffled, column)
    if before_codes is None or after_codes is None:
        result.metadata["shuffle_quality"] = {
            "applicable": False,
            "reason": f"obs column {column!r} not readable as a categorical",
        }
        return

    n_levels = int(max(before_codes.max(), after_codes.max())) + 1
    if n_levels <= 1:
        result.metadata["shuffle_quality"] = {
            "applicable": False,
            "reason": f"obs column {column!r} has a single level — TV is 0 by construction",
        }
        return

    block_before = int(_geometry(None, src)["shard_target_rows"])
    block_after = int(_geometry(None, shuffled)["shard_target_rows"])
    tv_b, tv8_b = _label_mixing(before_codes, block_before, n_levels)
    tv_a, tv8_a = _label_mixing(after_codes, block_after, n_levels)

    extra: dict[str, float] = {
        "label_tv_block__before": round(tv_b, 5),
        "label_tv_block__after": round(tv_a, 5),
    }
    if tv8_b is not None and tv8_a is not None:
        extra["label_tv_sgs8__before"] = round(tv8_b, 5)
        extra["label_tv_sgs8__after"] = round(tv8_a, 5)
    result.add_run(wall_s=0.0, peak_rss_mb=0.0, scenario="shuffle_quality", **extra)

    result.metadata["shuffle_quality"] = {
        "applicable": True,
        "label_column": column,
        "n_levels": n_levels,
        "block_rows_before": block_before,
        "block_rows_after": block_after,
        # A block-size mismatch would make the before/after pair incomparable
        # (a single block spanning the whole file has TV 0 by construction).
        "block_geometry_matched": block_before == block_after,
        "label_tv_block": {"before": round(tv_b, 5), "after": round(tv_a, 5)},
        "label_tv_sgs8": (
            {"before": round(tv8_b, 5), "after": round(tv8_a, 5)}
            if tv8_b is not None and tv8_a is not None
            else {
                "applicable": False,
                "reason": "file has <= 8 shards; pooling 8 blocks is the whole "
                "corpus, so TV is 0 by construction on any row order",
            }
        ),
        "reading": (
            "Total-variation distance between a contiguous block's label mix and "
            "the corpus mix; lower is better mixed. `block` is one CSR shard "
            "(shard_group_size=1, exact, no RNG); `sgs8` pools 8 random shards, "
            "the loader's default group size."
        ),
    }
    if block_before != block_after:
        logger.warning(
            "  shuffle_quality: block size changed %d -> %d rows; before/after "
            "TV are not comparable",
            block_before,
            block_after,
        )
    logger.info(
        "  shuffle_quality (%s, %d levels, %d-row blocks): block TV %.4f -> %.4f, "
        "sgs8 TV %s -> %s",
        column,
        n_levels,
        block_before,
        tv_b,
        tv_a,
        f"{tv8_b:.4f}" if tv8_b is not None else "n/a",
        f"{tv8_a:.4f}" if tv8_a is not None else "n/a",
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = True,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Execute the pre-shuffle benchmark for one (dataset, scx_auto) pair."""
    if format_variant is not None and format_variant.key not in SUPPORTED_FORMATS:
        return None

    try:
        import pyscx  # noqa: F401
    except ImportError:
        logger.info("shuffle_layout: pyscx unavailable — skipping")
        return None
    if not hasattr(__import__("pyscx"), "shuffle"):
        logger.info("shuffle_layout: pyscx build predates `shuffle` (scx Phase 1D) — skipping")
        return None

    src = Path(converted_path) if converted_path else dataset.path_for_format(format_variant.key)
    if not src.exists():
        logger.info("shuffle_layout: no scx_auto fixture at %s — skipping", src)
        return None

    result = BenchmarkResult(
        benchmark="shuffle_layout",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "seed": SHUFFLE_SEED,
            "batch_size": _BATCH_SIZE,
            "scenarios": ["shuffle_write", "size_delta", "train_throughput", "shuffle_quality"],
        },
    )
    result.file_size_bytes = src.stat().st_size

    scx_bin = _resolve_scx_bin()
    src_geometry = _geometry(scx_bin, src)
    result.metadata["source_geometry"] = dict(src_geometry)
    workroot = tempfile.TemporaryDirectory(prefix=f"shuffle_layout_{dataset.name}_")
    workdir = Path(workroot.name)
    logger.info(
        "shuffle_layout: %s (src=%.1f MB, %s shards @ %s rows, workdir=%s, rss=%.0fMB)",
        dataset.name,
        src.stat().st_size / 1e6,
        src_geometry["n_csr_shards"],
        src_geometry["shard_target_rows"],
        workdir,
        current_rss_mb(),
    )
    try:
        shuffled = _arm_shuffle_write(
            result,
            src,
            workdir,
            min(n_runs, _MAX_WRITE_RUNS),
            src_geometry["shard_target_rows"],
        )

        try:
            _arm_size_delta(result, dataset, workdir, scx_bin)
        except Exception as exc:  # never let one arm sink the cohort job
            logger.warning("  size_delta arm failed: %s", exc)
            result.metadata["size_delta"] = {"applicable": False, "reason": f"arm raised: {exc}"}

        if shuffled is not None and shuffled.exists():
            try:
                _arm_shuffle_quality(result, dataset, src, shuffled)
            except Exception as exc:
                logger.warning("  shuffle_quality arm failed: %s", exc)
                result.metadata["shuffle_quality"] = {
                    "applicable": False,
                    "reason": f"arm raised: {exc}",
                }
            try:
                _arm_train_throughput(
                    result, src, shuffled, min(n_runs, _MAX_EPOCH_RUNS), cold_cache
                )
            except Exception as exc:
                logger.warning("  train_throughput arm failed: %s", exc)
                result.metadata["train_throughput"] = {
                    "applicable": False,
                    "reason": f"arm raised: {exc}",
                }
        else:
            reason = "shuffle_write produced no output"
            result.metadata["train_throughput"] = {"applicable": False, "reason": reason}
            result.metadata["shuffle_quality"] = {"applicable": False, "reason": reason}
    finally:
        workroot.cleanup()

    logger.info("shuffle_layout done: %s — %d runs", dataset.name, len(result.runs))
    return result
