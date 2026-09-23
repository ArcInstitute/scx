"""CSC sidecar build — `pyscx.build_csc` wall clock and peak memory.

## Why this exists

`scx build-csc` / `pyscx.build_csc` had **no benchmark, no threshold and no RSS
metric** before this module. That matters more than an ordinary coverage gap:
this is the only operation in the suite with a *declared* memory budget —
`build_csc(memory_limit=…)`, default `"4G"` — and nothing measured the gap
between what a caller asks for and what the op takes.

**The budget is now partly honoured, and this module is what said so.** The
streaming `CscBuilder` replaced a transpose that rescanned every nonzero of
every shard twice per column chunk while `run_build_csc` held every decoded
shard in a `Vec<ScxCsr>`. Before/after on one node, median of three:

    dataset               before MB   after MB   before/bud  after/bud   wall x
    pbmc10k                     967        990         0.24       0.24     1.40
    tabula_sapiens_100k        3580       3775         0.87       0.92     2.05
    census_500k                8998       5512         2.20       1.35     5.66
    census_1m                 15411       7138         3.76       1.74     9.36

Read that carefully before quoting it. The wall is 9.4x at census scale and the
peak fell by half, but **1.74x is not 1.0** — the bucket staging is bounded by
an asserted counter, while the source shard's re-encode, the writer and the
interpreter baseline are not. And the two small datasets got slightly *worse*:
below the spill threshold every bucket stays resident.

The re-encode has since gone: build-csc is now an in-place append that writes
no CSR shard. Measured the same way (f1f9604c vs the append, one job):

    dataset               before MB   after MB   before/bud  after/bud   wall x
    pbmc10k                     988        910         0.24       0.22     1.11
    tabula_sapiens_100k        3819       3713         0.93       0.91     1.24
    census_500k                5547       5122         1.35       1.25     1.29
    census_1m                  7216       5879         1.76       1.44     1.22

Still not 1.0: the decode, the writer and the interpreter baseline remain
outside the bound. `thresholds.yaml`'s floors are regression floors on an
improvement, not contract floors.

This module also earned its keep twice over during that change. The first
capture came back at 1.92x with tabula 10 % worse than the code being replaced,
which is how a capacity-vs-length accounting bug in the builder's own bound was
found — the Rust suite was green throughout, because nothing asserted the
number the change was about.

The pre-fix budget sweep below is kept because it is the evidence for what
"declared but ignored" meant, and `thresholds.yaml` item 14 cites it.

**The budget could not be honoured at all, before.** Sweeping it over a 128x range on
tabula_sapiens_100k (194.9M nnz) moves the op's own allocation by 26% and halves
the wall — so the knob buys throughput for memory rather than doing nothing — but
even the smallest setting overshoots by 4.4x:

    memory_limit   peak (MB)   op delta (MB)   wall (s)
    512MiB            2683.6          2232.8      536.0
    1G                2676.9          2189.8      374.6
    4G                2973.0          2454.8      262.0
    64G               3272.7          2759.1      250.1

**The breach is scale-dependent.** At a 4 GiB budget: pbmc3k 526 MB (0.13x,
almost all interpreter baseline), tabula 2970 MB (0.73x), census_500k 8190 MB
(2.00x raw, 1.89x with the baseline removed) — crossing the budget around 315M
non-zeros. It is not "always over" and not "always fine"; a threshold has to
pick its dataset deliberately.

The rewrite ops (`compact`, `merge`, `optimize`, `sort`, `subset`) and
streaming convert build their sidecars through the same `CscBuilder`, fed in the
same pass as X, so whatever this measures about the builder is not local to one
entry point.

## What it emits

Per run, into `runs[].extra`:

* ``wall_s__build_csc`` — the `build_csc` call alone, excluding the input copy.
* ``peak_rss_mb__build_csc`` — true in-region peak (`PeakRssSampler`).
* ``peak_over_memory_limit__build_csc`` — peak ÷ the requested budget. **This is
  the contract number**: a value above 1.0 means the op used more than the
  caller allowed. Emitted on every run, so a threshold on it can never be a
  missing-metric violation.
* ``delta_rss_mb__build_csc`` / ``delta_over_memory_limit__build_csc`` — the same
  thing with the interpreter baseline subtracted. The Python process is
  450-520 MB before this op allocates anything (measured across a budget
  sweep; it creeps as the process retains freed heap), and `PeakRssSampler`
  seeds itself with whatever that is at entry,
  so on a small fixture the raw ratio is mostly interpreter: pbmc3k's 526 MB
  peak is 0.13x the budget for a matrix of 2.3M non-zeros. Both are recorded
  rather than one being chosen, because which is right depends on whether the
  budget is read as "what this op may allocate" or "what the process may reach".
* ``memory_limit_mb`` / ``entry_rss_mb`` / ``input_bytes`` / ``output_bytes``
  — context. The premise (a sidecar was actually built) is enforced by raising,
  not recorded as a metric.

The ratio **is** floored in `thresholds.yaml` now, on both census tiers, between
the append's measured value and the rewrite's (1.30 / 1.60) rather than at the
contract's 1.0 — see item 14 of
that file's "Deferred floors" block for the capture and for what is still
outside the bound. `n_csc_shards__build_csc` is pinned there too, from both
directions: a memory ratio can be met by narrowing shards, which would fix
nothing.

## In-place, one copy per run

`build_csc(input, output=None)` mutates `input` (appends the sidecar), and
`force=True` with `output=None` raises. So each run copies the Phase-A `.scx`
to a fresh path first; the copy is outside the timed region and outside the
sampler.
"""

from __future__ import annotations

import gc
import logging
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult, require_runs
from benchmarks.comprehensive.rss import PeakRssSampler, current_rss_mb

logger = logging.getLogger(__name__)

# CSC is an SCX-only concept, and `scx_auto` is the single trigger so the op is
# not re-measured once per codec variant. Read by `run_parallel`'s cohort
# builder so incompatible cells are never scheduled.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

# The budget passed to `build_csc`, and the denominator of
# `peak_over_memory_limit__build_csc`. Pinned rather than left to the default so
# the ratio means the same thing on every host — the whole point of the metric is
# the gap between what a caller asks for and what the op takes.
#
# `MemoryBudget::parse` accepts binary prefixes only (`4G`, `512MiB`); decimal
# `4GB` is rejected.
_MEMORY_LIMIT = "4G"
_MEMORY_LIMIT_MB = 4 * 1024

# `build_csc`'s own default. One shard per 5000 columns.
_CSC_COLS_PER_SHARD = 5000


def run(
    dataset: DatasetConfig,
    format_variant: FormatVariant,
    n_runs: int,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Time `pyscx.build_csc` on the Phase-A `.scx` for one dataset."""
    if format_variant is None or format_variant.key not in SUPPORTED_FORMATS:
        return None
    if dataset.multimodal:
        # `build_csc` rejects a multimodal input outright (ValueError). Skipped
        # rather than allowed to raise, so a multimodal cell is a clean no-op.
        return None

    if converted_path is None or not Path(converted_path).exists():
        raise FileNotFoundError(
            f"Missing converted SCX file for {dataset.name}. "
            f"Run Phase A conversion first (--formats scx_auto)."
        )

    import pyscx

    converted_path = Path(converted_path)
    result = BenchmarkResult(
        benchmark="build_csc",
        format=format_variant.key,
        dataset=dataset.name,
        metadata={
            "memory_limit": _MEMORY_LIMIT,
            "memory_limit_mb": _MEMORY_LIMIT_MB,
            "csc_cols_per_shard": _CSC_COLS_PER_SHARD,
            "n_obs": dataset.n_obs,
            "n_vars": dataset.n_vars,
        },
    )
    result.file_size_bytes = converted_path.stat().st_size

    workroot = tempfile.TemporaryDirectory(prefix=f"scx_build_csc_{dataset.name}_")
    workdir = Path(workroot.name)
    try:
        for i in range(n_runs):
            target = workdir / f"build_csc_{i}.scx"
            shutil.copy2(converted_path, target)
            input_bytes = target.stat().st_size

            gc.collect()
            # The interpreter + pyscx import is 450-520 MB before this op does
            # anything, and `PeakRssSampler` seeds itself with exactly that. So
            # the raw peak overstates the op's own allocation, badly at small
            # scale — measured 526 MB on pbmc3k, whose whole matrix is 2.3M
            # non-zeros. Recording the entry RSS lets the delta be reported
            # beside the raw figure instead of forcing a choice between a
            # number that flatters the op and one that indicts it.
            entry_rss = current_rss_mb()
            t0 = time.perf_counter()
            with PeakRssSampler() as sampler:
                pyscx.build_csc(
                    str(target),
                    output=None,  # in place
                    memory_limit=_MEMORY_LIMIT,
                    csc_cols_per_shard=_CSC_COLS_PER_SHARD,
                )
            wall = time.perf_counter() - t0
            peak = sampler.peak_mb
            output_bytes = target.stat().st_size

            # Premise check, inline: without a sidecar the wall and peak above
            # timed something other than a sidecar build, and a run that
            # silently measured a no-op is worse than a missing one.
            #
            # Reads the header flag. An earlier version counted
            # `X_csc_shard_*` out of `Experiment.validate()` and called that a
            # catalog read — `ScxReader::validate`
            # (`scx-format-io/src/reader/integrity.rs`) verifies the whole-file
            # checksum and then BLAKE3-hashes every section payload, so it was a
            # full-file hash after every timed run.
            built = pyscx.open(str(target))
            try:
                has_csc = bool(built.has_csc)
                # Same already-open reader, same header, no extra I/O. The
                # shard count is the layout number `has_csc` cannot express,
                # and the one a memory bound can be met by regressing: narrow
                # shards are individually cheap, so a build that fell back to
                # chunk-width sharding would pass a peak floor while fixing
                # nothing.
                #
                # Read directly. A `getattr(..., 0)` fallback carried the
                # before/after capture against an older pyscx, but that capture
                # is done and the fallback is now worse than useless: a rename
                # would record `0` and fail the `min: 173` floor for the wrong
                # reason, looking like a layout regression instead of a missing
                # attribute.
                n_csc_shards = int(built.n_csc_shards)
            finally:
                _close = getattr(built, "close", None)
                if _close is not None:
                    _close()
            if not has_csc:
                raise RuntimeError(
                    f"build_csc({target}) produced no CSC sidecar, so the "
                    f"{wall:.2f}s / {peak:.1f} MB above measured something other "
                    f"than a sidecar build. Refusing to record it."
                )

            result.add_run(
                wall_s=wall,
                peak_rss_mb=peak,
                **{
                    "scenario": "build_csc",
                    "run_idx": i,
                    "wall_s__build_csc": round(wall, 6),
                    "peak_rss_mb__build_csc": round(peak, 1),
                    # The contract as a *caller* sees it: total process
                    # footprint against the budget they asked for. Emitted
                    # unconditionally so a threshold on it is never a
                    # missing-metric violation.
                    "peak_over_memory_limit__build_csc": round(
                        peak / _MEMORY_LIMIT_MB, 4
                    ),
                    # The same contract with the interpreter baseline removed —
                    # what this op alone allocated. The two diverge by the
                    # 450-520 MB baseline,
                    # which is most of the number on a small fixture and noise on
                    # a large one; whichever a future threshold uses, both are on
                    # record so the choice is visible.
                    "entry_rss_mb": round(entry_rss, 1),
                    "delta_rss_mb__build_csc": round(peak - entry_rss, 1),
                    "delta_over_memory_limit__build_csc": round(
                        (peak - entry_rss) / _MEMORY_LIMIT_MB, 4
                    ),
                    # Emitted unconditionally, like the ratios above, so a
                    # threshold on it can never be a missing-metric violation.
                    "n_csc_shards__build_csc": n_csc_shards,
                    "memory_limit_mb": _MEMORY_LIMIT_MB,
                    "input_bytes": input_bytes,
                    "output_bytes": output_bytes,
                },
            )
            logger.info(
                "  build_csc run %d/%d: wall=%.2fs peak=%.1f MB "
                "(%.2fx the %s budget; %.1f MB over a %.1f MB baseline = %.2fx)",
                i + 1, n_runs, wall, peak, peak / _MEMORY_LIMIT_MB, _MEMORY_LIMIT,
                peak - entry_rss, entry_rss,
                (peak - entry_rss) / _MEMORY_LIMIT_MB,
            )
            target.unlink(missing_ok=True)
    finally:
        workroot.cleanup()

    require_runs(result, str(converted_path))

    peaks = [r.peak_rss_mb for r in result.runs]
    deltas = [r.extra["delta_rss_mb__build_csc"] for r in result.runs]
    result.metadata["median_peak_rss_mb"] = round(statistics.median(peaks), 1)
    result.metadata["median_peak_over_memory_limit"] = round(
        statistics.median(peaks) / _MEMORY_LIMIT_MB, 4
    )
    result.metadata["median_delta_rss_mb"] = round(statistics.median(deltas), 1)
    result.metadata["median_delta_over_memory_limit"] = round(
        statistics.median(deltas) / _MEMORY_LIMIT_MB, 4
    )
    shard_counts = [r.extra["n_csc_shards__build_csc"] for r in result.runs]
    result.metadata["median_n_csc_shards"] = int(statistics.median(shard_counts))
    logger.info(
        "build_csc done: %s — median peak %.1f MB = %.2fx the %s budget, "
        "%d CSC shards",
        dataset.name,
        result.metadata["median_peak_rss_mb"],
        result.metadata["median_peak_over_memory_limit"],
        _MEMORY_LIMIT,
        result.metadata["median_n_csc_shards"],
    )
    return result
