"""CSC sidecar build — `pyscx.build_csc` wall clock and peak memory.

## Why this exists

`scx build-csc` / `pyscx.build_csc` had **no benchmark, no threshold and no RSS
metric** before this module. That matters more than an ordinary coverage gap:
this is the only operation in the suite with a *declared* memory budget —
`build_csc(memory_limit=…)`, default `"4G"` — and nothing measured the gap
between what a caller asks for and what the op takes. `write_csc_sidecar` takes
`&[ScxCsr]` by value and has no streaming path, so the cost is dominated by
holding the CSR and its column-major transpose at once.

Two things are measured, both written up in `thresholds.yaml`'s Deferred
floors, item 14.

**The budget cannot be honoured.** Sweeping it over a 128x range on
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

Five other operations accept `--rebuild-csc` and route through the same writer,
so whatever this measures is not local to one entry point.

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
* ``memory_limit_mb`` / ``entry_rss_mb`` / ``input_bytes`` / ``output_bytes`` /
  ``n_csc_shards`` — context, and the premise check.

The ratio is deliberately **not** floored in `thresholds.yaml` yet. A
`<= 1.25` row would pass on tabula (0.73x) and fail on census_500k (1.99x), so
until the sidecar writer streams a live floor would fail every gate run that
reaches census scale. The measurements, the crossover, and the activation recipe
are in that file's "Deferred floors" block, item 14.

## In-place, one copy per run

`build_csc(input, output=None)` mutates `input` (temp file + atomic rename), and
`force=True` with `output=None` raises. So each run copies the Phase-A `.scx`
to a fresh path first; the copy is outside the timed region and outside the
sampler.
"""

from __future__ import annotations

import gc
import logging
import shutil
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


def _csc_shard_count(path: Path) -> int:
    """Number of `X_csc_shard_*` sections in the output, or 0 if none.

    A **premise check**, not a measurement: if `build_csc` produced no sidecar
    the wall and peak above are timing the wrong thing entirely, and a run that
    silently measured a no-op is worse than a missing one. `Experiment.has_csc`
    answers the yes/no; the count is what distinguishes a single-shard sidecar
    from the multi-shard layout `csc_cols_per_shard` should produce at these
    widths.

    `validate()` is the only Python surface that enumerates sections — it
    returns `[(name, checksum_ok)]` straight off the catalog. It does not carry
    section lengths, which is why this counts shards rather than reporting
    sidecar bytes; the file-size delta is not a substitute, because a rewrite
    can re-encode X itself.
    """
    import pyscx

    exp = pyscx.open(str(path))
    try:
        if not exp.has_csc:
            return 0
        return sum(
            1 for name, _ in exp.validate() if name.startswith("X_csc_shard_")
        )
    finally:
        close = getattr(exp, "close", None)
        if close is not None:
            close()


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

            n_csc_shards = _csc_shard_count(target)
            if n_csc_shards == 0:
                raise RuntimeError(
                    f"build_csc({target}) produced no X_csc_shard_* sections, so "
                    f"the {wall:.2f}s / {peak:.1f} MB above measured something "
                    f"other than a sidecar build. Refusing to record it."
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
                    "memory_limit_mb": _MEMORY_LIMIT_MB,
                    "input_bytes": input_bytes,
                    "output_bytes": output_bytes,
                    "n_csc_shards": n_csc_shards,
                },
            )
            logger.info(
                "  build_csc run %d/%d: wall=%.2fs peak=%.1f MB "
                "(%.2fx the %s budget; %.1f MB over a %.1f MB baseline = %.2fx) "
                "csc_shards=%d",
                i + 1, n_runs, wall, peak, peak / _MEMORY_LIMIT_MB, _MEMORY_LIMIT,
                peak - entry_rss, entry_rss,
                (peak - entry_rss) / _MEMORY_LIMIT_MB, n_csc_shards,
            )
            target.unlink(missing_ok=True)
    finally:
        workroot.cleanup()

    require_runs(result, str(converted_path))

    import statistics

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
    logger.info(
        "build_csc done: %s — median peak %.1f MB = %.2fx the %s budget",
        dataset.name,
        result.metadata["median_peak_rss_mb"],
        result.metadata["median_peak_over_memory_limit"],
        _MEMORY_LIMIT,
    )
    return result
