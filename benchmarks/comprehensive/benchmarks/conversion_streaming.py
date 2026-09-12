"""
Streaming → SCX conversion benchmark: h5ad, plus 10x since OPT-CONVERT-9.

For a given dataset, this benchmark runs the two h5ad → SCX
conversion entry points back-to-back and records peak RSS, wall
clock, and basic output-equality metadata for each path:

- **streaming** — `pyscx.from_h5ad(path, out)` →
  `scx_convert::h5ad_to_scx_streaming` (Phase 4 MVP, sequential
  single-threaded loop). Peak memory bounded by one shard's worth of
  CSR plus the resident `indptr` (`(n_obs + 1) × 8` bytes).
- **materialize** — `pyscx.from_anndata(sc.read_h5ad(path), out)`,
  the legacy in-memory path. Peak memory scales with the full CSR
  triplet.

Output equality between the two paths is measured at the "structural"
level (n_obs / n_vars / nnz / catalog shard_count / has_csc). The finer claims
live in Rust — `convert_tests_streaming.rs::streaming_round_trip_matches_non_streaming`
for h5ad values, `convert_tests_tenx_stream.rs` for the 10x pair (a `scx-testkit`
digest A/B on a single shard, and an explicit pin on the multi-shard codec-seed
divergence). The benchmark only needs to flag drift, not characterise it. Note
the paths are *not* byte-identical under the default `--codec auto` on either
format: eager forces one whole-matrix codec seed, streaming seeds per shard.

**Three extra streaming arms** run on the datasets named in `_EXTRA_ARMS`. All
three exist because the arms above pass *no* conversion options at all — their
only kwarg is `reader_threads` — so this benchmark's memory numbers have only
ever been measured in the default configuration:

- **csc_always** — `from_h5ad(..., csc="always")`. `CscPolicy::default()` is
  `off`, so nothing in this suite has exercised `write_csc_sidecar`, which takes
  the whole CSR by value. Expected to breach the streaming ceiling.
- **index_preset_cellxgene** — `from_h5ad(..., index_preset="cellxgene")`,
  materialising predicate indexes at write time.
- **budget_bound** — `from_h5ad(..., memory_budget=2GiB, shard_size=2048,
  reader_threads=12)`, the **only** arm anywhere in this suite that passes a
  memory budget to anything other than `build_csc`. It carries
  `peak_over_memory_budget__budget_bound`: the process's peak RSS over the
  budget it was handed.
  Read that ratio as an **observed ceiling on a measured shape, not as proof of
  a bound** — every row of `scx-convert`'s allocation table is `enforced:
  false`, each naming what it does not price (frame expansion, intra-codec
  planes, encoded indptr, the bitmap, the density-guess readers), so one
  compressible fixture landing at ~0.55 cannot promote an estimate into a
  contract. Its value is regression detection: the ratio moving up means
  something started allocating that the model does not charge.
  A premise check refuses the arm unless the Rust log shows
  `granted < requested` reader threads — warning *presence* is not enough,
  since the derate also fires when it shrinks only the queue depth, and a
  dropped budget produces a *passing* number.

`csc_always` and `index_preset_cellxgene` are pinned to the same
`GATED_READER_THREADS` as the gated arm so their numbers are comparable;
`budget_bound` pins **12** in its own kwargs, because at 4 the derate leaves
threads alone and shrinks only depth. Each gets `<label>_peak_rss_mb` /
`<label>_wall_s` from the same f-string the base arms use — no separate
emission path, and no chance of an extra arm polluting the floored
`streaming_peak_rss_mb` key.

**Two 10x arms** (`tenx_streaming` / `tenx_materialize`) run on the datasets in
`_TENX_ARM_DATASETS`, gating `scx convert --from 10x`'s bounded-memory claim.
They are not `_EXTRA_ARMS` entries because this direction is not reachable from
`from_h5ad` at all — `pyscx.from_10x` routes through `scanpy.read_10x_h5` into
`from_anndata`, a third code path with no `stream` kwarg — so they drive the
**CLI**, each `scx convert` in its own process (the peak comes from
`getrusage(RUSAGE_CHILDREN)`, which is cumulative across reaped children).
Their source is a synthesised fixture, since no dataset in the suite is a 10x
file; `prep_tenx_fixture.py` builds it, and without it both arms skip.

Under `SCX_CONV_STREAM_THREAD_COUNTS` the extra arms and the 10x pair are
**not** swept — none varies along the thread axis, since each changes a
conversion option and the 10x materialising arm has no threads knob at all —
but they still run **once**, each at its own pinned thread count, before the
sweep, so a sweep capture does not silently lose the non-default coverage.

Thread scaling is opt-in via the `SCX_CONV_STREAM_THREAD_COUNTS` env
var (comma-separated, e.g. `1,2,4,8,16,32`). When set, each thread
count is exercised in a fresh subprocess with
`RAYON_NUM_THREADS=N`. Subprocess isolation is mandatory: rayon's
global pool is initialised once on first use and cannot be reconfigured
in-process. Unset → in-process at the inherited thread count
(existing behaviour; smoke runs don't pay subprocess overhead).
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
import warnings
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler


log = logging.getLogger(__name__)

# Env vars set on subprocess workers so any thread pool (rayon, BLAS,
# OpenMP) honours the requested thread count. Mirrors
# `parallel_write_scaling.py`'s `_THREAD_ENV_VARS`.
_THREAD_ENV_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)

_THREAD_COUNTS_ENV = "SCX_CONV_STREAM_THREAD_COUNTS"

# Only the canonical SCX variant runs: the metric is path-shape, not
# format-shape, and both paths emit the same format under the same auto-codec.
# Declared as a module constant (not only as the inline `run()` guard below) so
# `run_parallel._bench_format_compatible` filters at cohort-build time. Without
# it the orchestrator schedules one job per format in the tier, each of which
# returns `None` — the phantom `missing_result` entries that function exists to
# prevent. Same pattern as `obs_open.py` / `cellset_gather.py`.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

# Above this many cells the `materialize` arm is skipped and only `streaming`
# runs. The materialize arm loads the whole h5ad: measured 13.6 GB peak RSS on
# census_1m (98.2 s / 851 MB for streaming — see docs/performance.md
# "Streaming conversion"), which extrapolates to ~68 GB at census_5m and ~136 GB
# at census_10m. It is a comparison baseline, not the thing under contract — the
# `streaming_peak_rss_mb` floor gates the streaming arm — so it is bounded here
# rather than being sized for by `estimate_memory_gb` on every future full-tier
# capture. The skip is recorded in `metadata`, never silent.
MATERIALIZE_MAX_N_OBS: int = 1_000_000

# Reader threads for the **gated** streaming arm.
#
# The `streaming_peak_rss_mb` floor has to be machine-independent, and the
# parallel reader's bound is `shards_in_flight x per_shard_working_set` — so with
# `reader_threads` left to `available_parallelism()` the floor's value is a
# property of the runner, not of the code. Measured on census_1m
# (1402 nnz/cell => 351 MB per 16384-row shard): 5.67-5.84 GB true peak at 16
# threads, matching 16 x 351 MB to within 6%. The same build would pass on a
# 4-core runner and fail on a 64-core one.
#
# Pinning the gated arm makes the floor test the actual contract — "bounded by N
# shards, independent of file size" — deterministically. The default-parallelism
# number is still measured and recorded under
# `streaming_default_threads_peak_rss_mb`, just not gated; that is how this suite
# already treats wall clock.
GATED_READER_THREADS: int = 4


def _skip_materialize_reason(n_obs: int) -> str | None:
    """Why the materialize arm is not run at this scale, or `None` to run it."""
    if n_obs > MATERIALIZE_MAX_N_OBS:
        return (
            f"materialize loads the whole h5ad (~14 GB at 1M cells, scaling "
            f"linearly); skipped above n_obs={MATERIALIZE_MAX_N_OBS:,} so a "
            f"full-tier capture does not pay it. The streaming arm — the one "
            f"the thresholds.yaml floor gates — still runs."
        )
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Extra streaming arms: the same `from_h5ad` call with one conversion option
# changed. Each entry is `label -> (from_h5ad kwargs, dataset allow-list)`.
#
# Both exist because the default arm passes **no** conversion options at all
# (its only kwarg is `reader_threads`), so the bound this benchmark enforces is
# measured in a configuration real callers do not always use.
#
# * `csc_always` — `CscPolicy::default()` is `off`, so nothing here has ever
#   exercised `write_csc_sidecar`, which takes the whole CSR by value. Scoped to
#   `tabula_sapiens_100k` deliberately: this arm is expected to breach its
#   ceiling and therefore needs a justification, and justification suppression is
#   **whole-triple** — attaching one to census_1m would also suppress the live
#   `streaming_peak_rss_mb <= 4096` floor on that triple.
# * `index_preset_cellxgene` — materialises predicate indexes at write time.
#   Scoped to the CELLxGENE-Census-derived datasets, which are the ones carrying
#   the columns the preset names. Checked, not assumed: the preset names ten obs
#   columns (`cell_type`, `cell_type_ontology_term_id`, `tissue`,
#   `tissue_ontology_term_id`, `disease`, `assay`, `donor_id`,
#   `development_stage`, `sex`, `suspension_type` —
#   `scx-engine/src/index/diagnostics.rs`) and `tabula_sapiens_100k.h5ad` has all
#   ten. pbmc3k has none, and measurably runs it as a no-op: the arm's output is
#   4,379,713 B against the default arm's 4,379,851 B, a 138-byte provenance
#   difference and no index sections at all.
# The budget arm's numbers, derived rather than guessed (measured from
# smartseq2's `indptr`, 2026-09-09): at `shard_size=2048` its widest shard
# holds 7,951,265 nnz, which at `budget::WORKER_PHASE_BYTES_PER_NNZ` = 48 is
# 382 MB for one in-flight shard. A 2 GiB budget therefore admits
# `2048/382 = 5` outstanding and grants 4 reader threads, against the ~16 a
# runner would otherwise use — a real derate, which is what the premise check
# below requires. Under the *old* 16 B/nnz model the same shard cost 127 MB and
# 2 GiB granted 15, so this arm's granted-thread count is exactly what the
# charge changed.
#
# smartseq2 because it is deep-sequenced (2,628 nnz/cell mean) **and** in every
# `capture_baseline.TIERS` list, so a default gate run reaches it. The genuinely
# worst case for an encode transient is `chemogenetic_rgfp` (~6,700 nnz/cell of
# raw integer UMIs, where `encoded ~= payload` is tight), but it is off-tier —
# a floor there is only evaluated under an explicit `--datasets`, which is why
# it is recorded as a deferred floor rather than gated here.
_BUDGET_ARM_BYTES = 2 * 1024 * 1024 * 1024
# **Binary** MiB, matching `PeakRssSampler`, which divides by `1024 * 1024`
# (`rss.py`). Dividing the peak by `_BUDGET_ARM_BYTES / 1e6` instead mixes
# binary MiB into a decimal-MB denominator and skews the ratio ~4.8% low —
# which for a floor set at exactly 1.0 is 4.8% of borrowed headroom.
# `build_csc.py` uses the same convention (`_MEMORY_LIMIT_MB = 4 * 1024`).
_BUDGET_ARM_MB = _BUDGET_ARM_BYTES / (1024 * 1024)

_EXTRA_ARMS: dict[str, tuple[dict[str, object], frozenset[str]]] = {
    "csc_always": ({"csc": "always"}, frozenset({"tabula_sapiens_100k"})),
    "budget_bound": (
        # `reader_threads` is inside the arm's kwargs deliberately, so
        # `kwargs.update` overrides the `GATED_READER_THREADS` pin every other
        # extra arm inherits. At the pinned 4 with the default depth 4 the
        # derate leaves threads at 4 and only shrinks depth 4 -> 1: the arm
        # would have recorded a peak under a *depth* derate while claiming to
        # gate a *thread* derate, and the PR's measured "12 -> 4" would never
        # have happened on it. 12 forces `granted < requested` here, and keeps
        # doing so if `ENCODE_TRANSIENT_MULTIPLE` is later re-derived downward.
        {
            "memory_budget": str(_BUDGET_ARM_BYTES),
            "shard_size": 2048,
            "reader_threads": 12,
        },
        frozenset({"smartseq2"}),
    ),
    # census_500k / census_1m are EXCLUDED, and not because they lack the
    # preset's columns — checked with h5py, both carry all ten, as does
    # tabula. `index_preset="cellxgene"` emits no predicate-index sections at
    # either census scale while it works at 100k: the arm's output came out
    # 1,146,311,684 B against the default arm's 1,148,236,779 at census_500k
    # and 2,729,215,285 vs 2,736,680,651 at census_1m — *smaller*, where an
    # index can only make a file larger, so something beyond the missing index
    # differs between the two arms at scale. Observed in the tier-full capture
    # (jobs 2894727_2 / 2894798_2, 2026-09-02).
    #
    # The premise check is right to refuse, but it raises, which took out the
    # whole `conversion_streaming` cell at both census scales — including
    # `streaming_peak_rss_mb`, floored at <= 4096 on census_1m. Scoping the arm
    # away from the datasets where the feature is broken keeps the other arms'
    # rows; it does not make the defect less real, and the defect is not this
    # benchmark's to fix. Restore the two datasets here in the change that
    # fixes it — the premise check is what will confirm the fix.
    "index_preset_cellxgene": (
        {"index_preset": "cellxgene"},
        frozenset({"tabula_sapiens_100k"}),
    ),
}


def _structural_summary(scx_path: Path) -> dict[str, int]:
    """Structural fingerprint of an output, for cross-path equality.

    Returns `n_obs` / `n_vars` / `nnz` / `shard_count` / `has_csc`. Bit-level
    equality is asserted in the Rust tests — see Phase 8; this only needs to
    flag drift.

    `has_csc` is here for the `csc_always` arm and is the premise `_run_isolated`
    enforces: if that arm reports 0 it converted without building a sidecar, and
    its wall and peak measured the default path under the wrong label. It is 0 on
    every other arm, so adding it does not change the streaming-vs-materialize
    equality above.

    It reads the header flag. An earlier version counted `X_csc_shard_*` sections
    out of `reader.validate()`, which verifies the whole-file checksum and then
    BLAKE3-hashes **every section payload** — a full-file hash on every extra-arm
    run, to answer a question consumed as a boolean. `build_csc` had the same
    call and it was replaced there first; this was the other half of that finding.

    (The docstring claimed `n_csr_shards` / `n_csc_shards` before an earlier
    change while the body returned neither name — `shard_count` and nothing.)
    """
    import pyscx

    reader = pyscx.open(str(scx_path))
    try:
        return {
            "n_obs": int(reader.n_obs),
            "n_vars": int(reader.n_vars),
            "nnz": int(reader.nnz),
            "shard_count": int(reader.shard_count),
            "has_csc": int(bool(reader.has_csc)),
        }
    finally:
        close = getattr(reader, "close", None)
        if close is not None:
            close()


def _timed_streaming(
    h5ad_path: Path,
    out_path: Path,
    reader_threads: int | None = None,
    extra_kwargs: dict[str, object] | None = None,
) -> dict[str, float]:
    """Run `pyscx.from_h5ad` and capture wall + the true in-region peak RSS.

    Uses :class:`PeakRssSampler`, like the six sibling modules that already do
    (`ooc_rss_boundary`, `ooc_loader`, `cellset_gather`, `obs_open`,
    `shuffle_layout`, `grouped_read`).

    It previously took `max(before, after)` of the *instantaneous* RSS and
    argued here that the under-report was "harmless for regression detection"
    because "the streaming path's memory profile is bounded structurally" —
    which assumes the very property the `streaming_peak_rss_mb` floor exists to
    check. Two things followed, both measured on 2026-08-22:

    * it is not a peak, so a transient spike inside the call is invisible; and
    * `rss_before` picks up whatever the *previous* run left resident. The
      materialize arm allocates ~14 GB and glibc does not return it, so with
      the arms interleaved in one process the streaming figure climbed
      1722 -> 2983 -> 3472 MB across three identical runs and tripped the
      2048 MB floor. Job 2834649 measured the same build in a fresh process per
      thread count: 736 MB at 1 reader thread rising to 1771 MB at 16 — bounded
      by shards-in-flight, sub-linear, and inside the floor. The floor was
      correct; the measurement was not.

    Hence also the per-arm process isolation in `_run_isolated` below: no
    in-process sampler can subtract another arm's retained heap.
    """
    import pyscx

    kwargs: dict[str, object] = (
        {} if reader_threads is None else {"reader_threads": reader_threads}
    )
    # `extra_kwargs` is the arm's conversion option (e.g. `csc="always"`). It is
    # merged rather than replacing, so an extra arm still runs at the pinned
    # thread count and stays comparable with the default one.
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    # Capture what the derate actually granted, so a `memory_budget` arm can
    # prove the budget bound rather than inferring it.
    #
    # Two observables, and the second is the load-bearing one:
    #
    # * the `UserWarning` category (`"scx conversion: N warning(s) of type
    #   'reader_threads_derated'"`) says *a* derate happened — but
    #   `derate_threads_and_depth` emits it whenever
    #   `requested_threads + requested_depth > outstanding_max`, **including
    #   when the granted thread count is unchanged and only the queue depth
    #   shrank**. Presence alone is therefore not evidence that the encode
    #   charge moved anything.
    # * the Rust log record carries `requested: N, granted: M` verbatim
    #   (pyo3-log forwards it to Python `logging`), which is the actual
    #   quantity. `granted < requested` is what "the budget bound the thread
    #   count" means.
    #
    # Without either, a budget that never reaches the coordinator produces a
    # perfectly good number under the budget label — the same shape as
    # `csc_always` recording a default conversion.
    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with PeakRssSampler() as sampler:
                pyscx.from_h5ad(str(h5ad_path), str(out_path), **kwargs)
        wall = time.perf_counter() - t0
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)
    categories = sorted(
        {
            m.group(1)
            for w in caught
            if (m := re.search(r"warning\(s\) of type '([^']+)'", str(w.message)))
        }
    )
    derate = re.search(r"requested:\s*(\d+),\s*granted:\s*(\d+)", log_buf.getvalue())
    return {
        "wall_s": wall,
        "peak_rss_mb": sampler.peak_mb,
        "warning_categories": categories,
        "derate_requested_threads": int(derate.group(1)) if derate else None,
        "derate_granted_threads": int(derate.group(2)) if derate else None,
    }


def _timed_materialize(h5ad_path: Path, out_path: Path) -> dict[str, float]:
    """Run `pyscx.from_anndata(sc.read_h5ad(path), out)` and capture
    wall + RSS. Includes the h5ad read into Python (anndata) — the
    streaming path bypasses that entirely, so the comparison is
    apples-to-apples at the user-visible "convert this h5ad" level."""
    import anndata
    import pyscx

    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        adata = anndata.read_h5ad(h5ad_path)
        pyscx.from_anndata(adata, str(out_path))
        del adata
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "peak_rss_mb": sampler.peak_mb}


# ---------------------------------------------------------------------------
# Thread-scaling worker (subprocess-per-thread-count)
# ---------------------------------------------------------------------------

# Worker executed in a subprocess for one thread count. Runs `n_runs`
# paired (streaming, materialize) conversions on the supplied h5ad,
# emits one JSON list to stdout where each element is
# `{"scenario", "run_idx", "wall_s", "peak_rss_mb",
#   "warning_categories", "derate_requested_threads", "derate_granted_threads",
#   "structural"?, "output_bytes"?}`. The first run of each scenario
# also reports the structural fingerprint + output size so the parent
# can detect path drift without re-opening files.
# ---------------------------------------------------------------------------
# The 10x arms (OPT-CONVERT-9)
# ---------------------------------------------------------------------------

# `scx convert --from 10x` streams since OPT-CONVERT-9 and does so **by
# default**; `--stream=false` reaches the eager whole-matrix reader. These two
# arms measure that pair.
#
# Why they are not `_EXTRA_ARMS` entries: every entry there is a `from_h5ad`
# kwarg dict, and this direction is not reachable from `from_h5ad` at all.
# `pyscx.from_10x` exists but routes through `scanpy.read_10x_h5` into
# `from_anndata` — a third code path with no `stream` kwarg and no bounded
# reader — so it cannot express either arm. The CLI can, and is what a real
# CellBender-workflow caller uses.
#
# Scoped to `census_500k` deliberately, on two counts. It is in the `full`
# capture tier, so a default `gate_candidate.py` run reaches it — unlike an
# off-tier dataset, where `check_absolute_floors` skips the triple *silently*
# and a floor reads as coverage while providing none. And it is the largest
# tier dataset carrying **no** existing `conversion_streaming` floor and no
# expected-to-breach arm: `census_1m` holds `streaming_peak_rss_mb`,
# `smartseq2` holds `peak_over_memory_budget__budget_bound`, and
# `tabula_sapiens_100k` hosts both arms that need a justification. Justification
# suppression is whole-triple, so sharing a dataset with any of those would put
# this floor one justification away from silently disarming.
_TENX_ARM_DATASETS: frozenset[str] = frozenset({"census_500k"})


_TENX_WORKER_SCRIPT = textwrap.dedent("""\
    import json
    import resource
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    tenx_path = sys.argv[1]
    run_idx = int(sys.argv[2])
    stream = sys.argv[3]            # "true" | "false"
    reader_threads = sys.argv[4]    # or "" to inherit
    scx_bin = sys.argv[5]

    from benchmarks.comprehensive.benchmarks.conversion_streaming import (
        _structural_summary,
    )

    # One convert per worker process, no loop: `getrusage(RUSAGE_CHILDREN)` is a
    # cumulative high-water mark over every reaped child, so a second convert
    # here would inherit the first's peak and every later run would report a
    # monotone non-decreasing number. The parent spawns one of these per run.
    with tempfile.TemporaryDirectory(prefix="scx_bench_tenx_") as tmp:
        out = Path(tmp) / "out.scx"
        argv = [scx_bin, "convert", "--from", "10x", "--stream=" + stream]
        if reader_threads:
            argv += ["--reader-threads", reader_threads]
        argv += [tenx_path, str(out)]
        t0 = time.perf_counter()
        proc = subprocess.run(argv, capture_output=True, text=True)
        wall_s = time.perf_counter() - t0
        if proc.returncode != 0:
            # Terse, and deliberately NOT naming any string the parent
            # classifies on. An earlier version explained the stale-binary case
            # here; because `python -c` prints SystemExit's argument to stderr,
            # every failure -- an OOM, a corrupt fixture, a real convert bug --
            # then carried the parent's needle and was recorded as "stale
            # binary, arms skipped". Reproduced on a corrupt fixture with a
            # current binary. `scx`'s own stderr is the only source of truth.
            raise SystemExit(
                "scx convert --from 10x --stream=" + stream + " exited "
                + str(proc.returncode) + "\\n" + proc.stderr
            )
        # RUSAGE_CHILDREN, not RUSAGE_SELF: the thing being measured is the
        # `scx` process, and this worker exists so that `scx` is its only child.
        ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        # `ru_maxrss` is KiB on Linux and bytes on Darwin -- the same split
        # `rss.py` already handles. Without the branch a macOS run reports
        # 1024x high and trips every ceiling.
        peak_rss_mb = (
            ru / (1024.0 * 1024.0) if sys.platform == "darwin" else ru / 1024.0
        )
        rec = {
            "run_idx": run_idx,
            "wall_s": wall_s,
            "peak_rss_mb": peak_rss_mb,
            "structural": _structural_summary(out),
            "output_bytes": out.stat().st_size,
        }

    print(json.dumps([rec]))
""")


class TenxArmUnavailable(RuntimeError):
    """The 10x arms cannot run here — a stale/mis-featured `scx`, not a result.

    Distinct from a plain `RuntimeError` because the two need opposite handling:
    a genuine worker crash should fail the run, while a binary that predates
    OPT-CONVERT-9 (or was built without `--features hdf5`) must leave the h5ad
    arms — which carry this benchmark's primary floors — untouched. There is no
    probe that separates them up front: every `scx` since the flag shipped
    answers `convert --help` with `--stream`, including the builds that reject
    it on this direction, so the convert itself is the capability test.
    """


def _run_tenx_arm_subprocess(
    tenx_path: Path,
    n_runs: int,
    stream: bool,
    reader_threads: int | None,
    scx_bin: str,
) -> list[dict]:
    """Run ``n_runs`` of one 10x arm, each `scx convert` in its own process.

    The worker is a Python shim rather than a direct `subprocess.run` from here
    because the peak comes from `getrusage(RUSAGE_CHILDREN)`, which is scoped to
    *all* of the calling process's reaped children. One worker per **run** keeps
    `scx` the only child it ever reaps, so neither the other arm's peak nor the
    previous run's can leak into a record — the same process-boundary
    requirement :func:`_run_arm_subprocess` documents for the h5ad arms, for the
    same reason with a different mechanism.
    """
    records: list[dict] = []
    for run_idx in range(n_runs):
        proc = subprocess.run(
            [
                sys.executable, "-c", _TENX_WORKER_SCRIPT,
                str(tenx_path), str(run_idx), "true" if stream else "false",
                "" if reader_threads is None else str(reader_threads),
                scx_bin,
            ],
            capture_output=True,
            text=True,
            timeout=14400,
        )
        if proc.returncode != 0:
            combined = f"{proc.stdout}\n{proc.stderr}"
            if (
                "--stream is not supported for direction" in combined
                or "requires the 'hdf5' feature" in combined
            ):
                raise TenxArmUnavailable(
                    f"`{scx_bin}` cannot run the 10x arms (stream={stream}): it "
                    f"predates OPT-CONVERT-9 or lacks --features hdf5. Set "
                    f"$SCX_CLI_BIN or rebuild target/release/scx.\n{combined.strip()}"
                )
            raise RuntimeError(
                f"conversion_streaming 10x worker failed (stream={stream}, "
                f"run={run_idx}, exit={proc.returncode}).\n"
                f"--- stderr ---\n{proc.stderr}\n--- stdout ---\n{proc.stdout}"
            )
        try:
            records.extend(json.loads(proc.stdout.strip().splitlines()[-1]))
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(
                f"Failed to parse 10x worker JSON output: {exc}\n"
                f"--- stdout ---\n{proc.stdout}"
            ) from exc
    return records


def _assert_tenx_streaming_beat_materialize(result: BenchmarkResult) -> None:
    """Refuse a 10x pair where the streaming arm did not bound anything.

    The arms exist to measure a bound, and both of the ways they could stop
    doing so are silent. If `--stream=false` stopped reaching the eager reader
    the two arms would record the same path under two labels and both pass; if
    the streaming path regressed to a whole-matrix read the peaks would converge
    the other way. Either way `tenx_streaming_peak_rss_mb` would still be
    emitted and still clear its ceiling, having measured the wrong thing.

    A bare inequality is the check, not a ratio: the margin is a property of the
    dataset's nnz-per-cell and the shard size, so pinning a multiple here would
    be a second, unmeasured threshold. `thresholds.yaml` carries the ceiling.
    """
    peaks: dict[str, float] = {}
    for run in result.runs:
        for label in ("tenx_streaming", "tenx_materialize"):
            value = run.extra.get(f"{label}_peak_rss_mb")
            if value is not None:
                peaks[label] = max(peaks.get(label, 0.0), float(value))
    if len(peaks) < 2:
        return  # one arm did not run; nothing to compare, and not a claim
    if peaks["tenx_streaming"] >= peaks["tenx_materialize"]:
        raise RuntimeError(
            f"the 10x streaming arm peaked at "
            f"{peaks['tenx_streaming']:.1f} MB against the materialising arm's "
            f"{peaks['tenx_materialize']:.1f} MB. Streaming is supposed to be "
            f"the bounded path, so either `--stream=false` no longer reaches "
            f"the eager reader (both arms timed the same path) or the streaming "
            f"path regressed to a whole-matrix read. Refusing to record a "
            f"bound that was not measured."
        )


def _run_tenx_arms(
    dataset: DatasetConfig,
    n_runs: int,
    result: BenchmarkResult,
) -> None:
    """Add the `tenx_streaming` / `tenx_materialize` arms when in scope.

    Skips — recorded in `metadata`, never silently — when the dataset is out of
    scope, when `prep_tenx_fixture.py` has not been run for it, or when no
    resolvable `scx` binary has a `convert --stream`. A missing fixture is a
    skip rather than an error because the h5ad arms carry this benchmark's
    primary floors and must still run on a box where the 10x fixture was never
    built.
    """
    from benchmarks.comprehensive import scx_cli

    if dataset.name not in _TENX_ARM_DATASETS:
        result.metadata["tenx_arms_skipped"] = (
            f"{dataset.name} is not in the 10x arms' dataset scope "
            f"({sorted(_TENX_ARM_DATASETS)})"
        )
        return

    tenx_path = dataset.tenx_path
    if not tenx_path.exists():
        result.metadata["tenx_arms_skipped"] = (
            f"10x fixture not built: {tenx_path} is absent. Build it with "
            f"`python benchmarks/scripts/prep_tenx_fixture.py --datasets "
            f"{dataset.name}`."
        )
        log.warning("%s", result.metadata["tenx_arms_skipped"])
        return

    # Inlined rather than a named constant: there is one caller, and this probe
    # does not settle the question on its own — every `scx` since the flag
    # shipped answers with `--stream`, including builds that reject it on 10x.
    # It only weeds out a binary with no `convert --stream` at all; the convert
    # itself is the real capability test (`TenxArmUnavailable`).
    scx_bin = scx_cli.resolve_scx_bin(("convert", "--help"), requires=b"--stream")
    if scx_bin is None:
        result.metadata["tenx_arms_skipped"] = (
            "no resolvable `scx` binary has `convert --stream`; set "
            "$SCX_CLI_BIN or build target/release/scx"
        )
        log.warning("%s", result.metadata["tenx_arms_skipped"])
        return

    result.metadata["tenx_source_bytes"] = tenx_path.stat().st_size
    result.metadata["tenx_gated_reader_threads"] = GATED_READER_THREADS
    # Both arms run to completion **before** anything is recorded. A skip that
    # fired mid-loop used to leave a half-pair on the result: `tenx_streaming`
    # had already `add_run`'d, so the floored `tenx_streaming_peak_rss_mb`
    # survived while `_assert_tenx_streaming_beat_materialize` returned early on
    # `len(peaks) < 2` — the floor passing with the comparison it exists for
    # silently gone. The realistic trigger is the materialising arm, which is
    # the one that can OOM (8599 MB against streaming's 2593 MB on census_500k).
    arms = (
        ("tenx_streaming", True, GATED_READER_THREADS),
        ("tenx_materialize", False, None),
    )
    try:
        collected = [
            (label, threads, _run_tenx_arm_subprocess(
                tenx_path, n_runs, stream, threads, scx_bin
            ))
            for label, stream, threads in arms
        ]
    except TenxArmUnavailable as exc:
        # A binary that cannot run this direction must not take the h5ad arms
        # down with it: they carry this benchmark's primary floors, and the
        # docstring above promises a recorded skip rather than a crash. Only
        # this one shape is caught — a genuine worker failure still raises, or
        # the arms would go quiet for the wrong reason.
        result.metadata["tenx_arms_skipped"] = str(exc)
        log.warning("10x arms skipped: %s", exc)
        return

    structural: dict[str, dict | None] = {}
    ran: list[str] = []
    for label, threads, records in collected:
        ran.append(label)
        for rec in records:
            if rec.get("structural") is not None and label not in structural:
                structural[label] = rec["structural"]
                if rec.get("output_bytes") is not None:
                    result.metadata[f"{label}_output_bytes"] = rec["output_bytes"]
            result.add_run(
                wall_s=rec["wall_s"],
                peak_rss_mb=rec["peak_rss_mb"],
                scenario=label,
                run_idx=rec["run_idx"],
                reader_threads=threads,
                **{
                    f"{label}_peak_rss_mb": rec["peak_rss_mb"],
                    f"{label}_wall_s": rec["wall_s"],
                },
            )
            result.metadata["scenarios"].append(label)
            log.info(
                "  %s (reader_threads=%s) run %d: wall=%.2fs peak_rss=%.1f MB",
                label, threads, rec["run_idx"], rec["wall_s"], rec["peak_rss_mb"],
            )

    result.metadata["tenx_arms"] = ran
    # The two paths must produce the same matrix. They are *not* byte-identical
    # — the eager path forces one file-wide codec seed while the coordinator
    # seeds per shard — so this is the structural comparison, which is the same
    # claim the h5ad pair makes here. Bit-level agreement on the parts that do
    # agree is pinned in Rust (`convert_tests_tenx_stream`).
    result.metadata["tenx_structural"] = {
        **structural,
        "equal": structural.get("tenx_streaming") == structural.get("tenx_materialize"),
    }
    if structural.get("tenx_streaming") != structural.get("tenx_materialize"):
        log.warning(
            "10x structural mismatch streaming=%s materialize=%s",
            structural.get("tenx_streaming"), structural.get("tenx_materialize"),
        )
    _assert_tenx_streaming_beat_materialize(result)


_WORKER_SCRIPT = textwrap.dedent("""\
    import json
    import sys
    import tempfile
    from pathlib import Path

    h5ad_path = sys.argv[1]
    n_runs = int(sys.argv[2])
    scenario = sys.argv[3]          # "streaming" | "materialize"
    rt = sys.argv[4]                # reader_threads, or "" to inherit
    reader_threads = int(rt) if rt else None
    # from_h5ad kwargs for an extra arm, JSON, or "" for the default arm.
    extra_kwargs = json.loads(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else {}

    from benchmarks.comprehensive.benchmarks.conversion_streaming import (
        _timed_streaming,
        _timed_materialize,
        _structural_summary,
    )

    prefix = "scx_bench_stream_" if scenario == "streaming" else "scx_bench_bulk_"
    fname = "stream.scx" if scenario == "streaming" else "bulk.scx"

    records = []
    structural = None
    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
            out = Path(tmp) / fname
            if scenario == "streaming":
                t = _timed_streaming(
                    Path(h5ad_path), out, reader_threads, extra_kwargs
                )
            else:
                # `from_anndata` has no reader-threads knob; the arm is a
                # whole-file materialization either way.
                t = _timed_materialize(Path(h5ad_path), out)
            rec = {
                "scenario": scenario,
                "run_idx": run_idx,
                "reader_threads": reader_threads,
                "wall_s": t["wall_s"],
                "peak_rss_mb": t["peak_rss_mb"],
                # Only the streaming worker collects these; the materialize
                # arm's helper does not take a budget.
                "warning_categories": t.get("warning_categories", []),
                "derate_requested_threads": t.get("derate_requested_threads"),
                "derate_granted_threads": t.get("derate_granted_threads"),
            }
            if structural is None:
                structural = _structural_summary(out)
                rec["structural"] = structural
                rec["output_bytes"] = out.stat().st_size
            records.append(rec)

    print(json.dumps(records))
""")


def _run_arm_subprocess(
    h5ad_path: Path,
    n_runs: int,
    scenario: str,
    thread_count: int | None = None,
    reader_threads: int | None = None,
    extra_kwargs: dict[str, object] | None = None,
) -> list[dict]:
    """Run ``n_runs`` of **one** arm in a fresh subprocess.

    One arm per process is the point, not an implementation detail. The
    materialize arm allocates ~14 GB on census_1m and glibc does not return it,
    so with both arms interleaved in one process the *streaming* figure inherits
    the materialize arm's retained heap — measured at 1722 -> 2983 -> 3472 MB
    across three identical runs, tripping a 2048 MB floor that a fresh process
    puts at 1771 MB for the same build (job 2834649). No in-process sampler can
    subtract another arm's garbage, so the arms have to be separated by a process
    boundary.

    ``thread_count`` pins ``RAYON_NUM_THREADS`` and friends when set; ``None``
    inherits the caller's environment. Failure surfaces the worker's stderr
    verbatim.
    """
    env = os.environ.copy()
    if thread_count is not None:
        for var in _THREAD_ENV_VARS:
            env[var] = str(thread_count)

    proc = subprocess.run(
        [
            sys.executable, "-c", _WORKER_SCRIPT,
            str(h5ad_path), str(n_runs), scenario,
            "" if reader_threads is None else str(reader_threads),
            json.dumps(extra_kwargs) if extra_kwargs else "",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=14400,  # census_10m comfortably under 4 h for a single arm
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"conversion_streaming worker failed (scenario={scenario}, "
            f"threads={thread_count}, exit={proc.returncode}).\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}"
        )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            f"Failed to parse worker JSON output: {exc}\n"
            f"--- stdout ---\n{proc.stdout}"
        ) from exc


def _run_thread_count_subprocess(
    h5ad_path: Path,
    n_runs: int,
    thread_count: int,
) -> list[dict]:
    """Both arms at one thread count, **each in its own subprocess**.

    Was one process running the pair; see :func:`_run_arm_subprocess` for why
    that made the streaming figure carry the materialize arm's retained heap.
    """
    return (
        _run_arm_subprocess(h5ad_path, n_runs, "streaming", thread_count)
        + _run_arm_subprocess(h5ad_path, n_runs, "materialize", thread_count)
    )


def _parse_thread_counts(raw: str | None) -> list[int] | None:
    """Parse `SCX_CONV_STREAM_THREAD_COUNTS`. Returns None when unset
    or empty — caller falls back to the in-process single-count path.
    """
    if not raw:
        return None
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            n = int(tok)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value in {_THREAD_COUNTS_ENV}='{raw}': '{tok}' is not int"
            ) from exc
        if n < 1:
            raise ValueError(
                f"Invalid value in {_THREAD_COUNTS_ENV}='{raw}': '{tok}' must be >= 1"
            )
        out.append(n)
    return out or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    dataset: DatasetConfig,
    format_variant=None,  # unused — kept for harness signature parity
    n_runs: int = 1,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult:
    """Run the streaming-vs-materialising conversion benchmark for
    one dataset.

    Returns a `BenchmarkResult` whose `runs` list pairs `wall_s` and
    `peak_rss_mb` for each scenario, with `metadata["scenarios"]`
    naming which run is streaming vs materialize. Output structural
    summaries are stored under `metadata["structural"]` so any drift
    between paths is visible without re-running.

    n_runs > 1 is honoured for each path: median wall / max peak RSS
    are computed downstream from the per-run records.

    When `SCX_CONV_STREAM_THREAD_COUNTS` is set (e.g. `1,2,4,8,16,32`),
    each thread count is exercised in its own subprocess and runs are
    tagged with `extra.thread_count`. Per-(thread_count, scenario)
    scaling summaries land in `metadata.scaling_wall_s` /
    `metadata.speedup` / `metadata.efficiency`.
    """
    # The harness drives this benchmark once per (dataset, format_variant)
    # pair. Conversion outcome doesn't depend on the format variant —
    # both paths emit the same SCX format under the same auto-codec —
    # so we run only when the canonical `scx_auto` variant comes
    # through and silently skip every other format. Operators don't
    # need to remember `--formats scx_auto`; the rest no-op.
    if format_variant is not None and format_variant.key != "scx_auto":
        return None  # type: ignore[return-value]

    thread_counts = _parse_thread_counts(os.environ.get(_THREAD_COUNTS_ENV))

    # Refuse a sweep above the materialize cap BEFORE resolving the input,
    # mirroring `export_streaming.run`. The refusal does not depend on a fixture
    # being on disk, and putting it after the existence check made the shared
    # `test_thread_sweep_above_the_cap_is_refused_not_silently_uncapped`
    # parametrisation fail on this half with `FileNotFoundError` instead of the
    # `ValueError` it asserts — a pre-existing asymmetry between the two
    # modules, since `export_streaming` hoisted its guard and this one did not.
    # Found by review (Antigravity - Gemini 3.7 Flash).
    if thread_counts is not None and _skip_materialize_reason(dataset.n_obs):
        raise ValueError(
            f"{_THREAD_COUNTS_ENV} is set on a dataset with n_obs="
            f"{dataset.n_obs:,}, where the materialize arm is skipped "
            f"({_skip_materialize_reason(dataset.n_obs)}). A thread-scaling run "
            f"compares the two arms, so it needs both: either use a dataset at "
            f"or below n_obs={MATERIALIZE_MAX_N_OBS:,}, or unset "
            f"{_THREAD_COUNTS_ENV}."
        )

    h5ad_path = dataset.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Source h5ad not found: {h5ad_path}")

    source_bytes = h5ad_path.stat().st_size

    log.info(
        "Streaming conversion benchmark: dataset=%s source=%.1f MB n_runs=%d "
        "thread_counts=%s",
        dataset.name,
        source_bytes / 1e6,
        n_runs,
        thread_counts if thread_counts else "in-process default",
    )

    # `format` is fixed to the synthetic key the absolute-floor gate
    # references (`thresholds.yaml`), not the iterating `format_variant`,
    # because the metric is path-shape, not format-shape.
    result = BenchmarkResult(
        benchmark="conversion_streaming",
        format="scx_streaming_vs_materialize",
        dataset=dataset.name,
        metadata={
            "source_h5ad_bytes": source_bytes,
            "n_obs": dataset.n_obs,
            "n_vars": dataset.n_vars,
            "scenarios": [],
            "thread_counts": thread_counts if thread_counts else None,
        },
    )

    skip_materialize = _skip_materialize_reason(dataset.n_obs)
    result.metadata["materialize_skipped_reason"] = skip_materialize
    if skip_materialize:
        log.info("materialize arm skipped: %s", skip_materialize)

    if thread_counts is None:
        result = _run_isolated(
            h5ad_path, n_runs, result, skip_materialize, dataset.name,
        )
        # After the h5ad arms, so a 10x fixture that is absent or a stale `scx`
        # binary cannot cost this benchmark its primary floors.
        _run_tenx_arms(dataset, n_runs, result)
        return result

    # The sweep-above-the-cap refusal is hoisted above input resolution near the
    # top of this function, so there is deliberately no second copy here.
    result.metadata["extra_arms"] = []
    _run_extra_arms_once(h5ad_path, result, dataset.name)
    swept = _run_with_thread_scaling(h5ad_path, n_runs, thread_counts, result)
    # Deferred to here on purpose: the extra arms ran *before* the sweep, and
    # this check compares against `streaming_output_bytes`, which only the sweep
    # writes. Running it inside `_run_extra_arms_once` would always see
    # `plain is None`.
    _assert_index_arm_changed_the_output(
        list(swept.metadata.get("extra_arms", [])), swept
    )
    # Run once, not swept: neither 10x arm varies along the thread axis (the
    # materialising one has no reader-threads knob at all), and a sweep capture
    # that silently dropped them would emit no value for the 10x floor.
    _run_tenx_arms(dataset, n_runs, swept)
    return swept


def _assert_csc_arm_built_a_sidecar(
    labels: list[str] | tuple[str, ...],
    structural: dict[str, dict | None],
) -> None:
    """Refuse a `csc_always` result whose output has no sidecar.

    Without this the arm emits `csc_always_peak_rss_mb` after a worker that
    dropped `csc="always"` on either subprocess hop — the default conversion
    recorded under the CSC label. Self-contained: it needs only this arm's own
    structural record, so it can run the moment that arm returns.
    """
    if "csc_always" not in labels:
        return
    has_csc = (structural.get("csc_always") or {}).get("has_csc")
    if not has_csc:
        raise RuntimeError(
            f"the csc_always arm produced has_csc={has_csc!r}: "
            f"`csc=\"always\"` did not reach `from_h5ad`, so the arm timed "
            f"the default conversion under the CSC label. Refusing to record it."
        )


def _assert_budget_arm_actually_derated(
    labels: list[str] | tuple[str, ...],
    result: BenchmarkResult,
) -> None:
    """Refuse a `budget_bound` result whose budget did not bind.

    The failure this exists for is silent and *passing*: a budget that never
    reaches `derate_threads_and_depth` — dropped on either subprocess hop, or
    resolved to `None` by a coercion change — produces a perfectly good peak
    under the budget label, and the `peak_over_memory_budget__budget_bound`
    ratio then reports that an unconstrained conversion honoured a constraint
    it never saw. Same shape as `csc_always` timing the default conversion.

    The observable is `granted < requested`, read from the derate's own log
    record. **Not** the presence of the `reader_threads_derated` warning: that
    is emitted whenever `requested_threads + requested_depth > outstanding_max`,
    which includes the case where the granted thread count is unchanged and
    only the queue depth shrank. An earlier version of this check used warning
    presence and passed for exactly that reason — the arm was pinned to
    `GATED_READER_THREADS = 4` with depth 4, so the budget shrank depth 4 -> 1
    and left threads at 4, and the check reported that the encode charge had
    bound the thread count when it had not.

    Asserting the quantity also survives the follow-up this PR proposes: if
    `ENCODE_TRANSIENT_MULTIPLE` is later re-derived downward, a presence-based
    check would stop firing and refuse the arm as "budget did not bind",
    making a cheaper-but-correct cost model indistinguishable from a dropped
    budget.
    """
    if "budget_bound" not in labels:
        return
    pairs: list[tuple[int | None, int | None]] = []
    for run in result.runs:
        if run.extra.get("scenario") != "budget_bound":
            continue
        pairs.append(
            (
                run.extra.get("derate_requested_threads"),
                run.extra.get("derate_granted_threads"),
            )
        )
    if not [1 for r, g in pairs if r is not None and g is not None and g < r]:
        raise RuntimeError(
            "the budget_bound arm never recorded `granted < requested` reader "
            f"threads (observed {pairs!r}): `memory_budget` did not bind the "
            "thread count, so the arm timed a conversion the budget did not "
            "constrain and its peak-over-budget ratio would be meaningless. "
            "Refusing to record it."
        )


def _assert_index_arm_changed_the_output(
    labels: list[str] | tuple[str, ...],
    result: BenchmarkResult,
) -> None:
    """Refuse an `index_preset_cellxgene` result that shows no index effect.

    `Experiment` exposes no `has_obs_index`, so the check is the observable one:
    writing the preset's index sections makes the output strictly larger than the
    default arm's, on the same input with the same codec.

    **Fail-closed.** A missing size on either side means the comparison could not
    be made, which is not evidence that it passed — an earlier version returned
    quietly on `None` and could be satisfied by a worker that simply omitted
    `output_bytes`. Found by review (Antigravity - Gemini 3.7 Flash, Cursor Agent
    - Grok 4.6 High).

    ⚠️ Ordering: this needs the *default* arm's size, so it must run after that
    arm has recorded. On the thread-sweep path the extra arms run first, so the
    caller defers this until the sweep has written `streaming_output_bytes`.
    """
    if "index_preset_cellxgene" not in labels:
        return
    indexed = result.metadata.get("index_preset_cellxgene_output_bytes")
    plain = result.metadata.get("streaming_output_bytes")
    if indexed is None or plain is None:
        raise RuntimeError(
            f"cannot check the index_preset_cellxgene premise: output sizes were "
            f"indexed={indexed!r} default={plain!r}. Refusing to record an arm "
            f"whose effect could not be verified."
        )
    if indexed <= plain:
        raise RuntimeError(
            f"the index_preset_cellxgene arm wrote {indexed} bytes against the "
            f"default arm's {plain}: no predicate-index sections were emitted, "
            f"so `index_preset=\"cellxgene\"` did not take effect and the arm "
            f"timed the default conversion. Refusing to record it."
        )


def _run_isolated(
    h5ad_path: Path,
    n_runs: int,
    result: BenchmarkResult,
    skip_materialize: str | None = None,
    dataset_name: str | None = None,
) -> BenchmarkResult:
    """Default path: three arms plus any applicable extra arms, one subprocess each.

    * ``streaming`` at :data:`GATED_READER_THREADS` — carries
      ``streaming_peak_rss_mb``, the key ``thresholds.yaml`` floors. Pinned so
      the floor is a property of the code and not of the runner's core count.
    * ``streaming_default_threads`` at the inherited parallelism — carries
      ``streaming_default_threads_peak_rss_mb``, **recorded, not gated**, so the
      real-world number stays visible without making the gate machine-dependent.
    * ``materialize`` — the comparison baseline. No reader-threads knob applies.
    * any entry of :data:`_EXTRA_ARMS` whose dataset allow-list contains
      ``dataset_name`` — the same pinned streaming conversion with one option
      changed (``csc="always"``, ``index_preset="cellxgene"``). Each carries
      ``<label>_peak_rss_mb`` / ``<label>_wall_s`` for free from the f-string
      below, so no new emission path is needed.

    One arm per process is load-bearing, not tidiness: the materialize arm
    allocates tens of GB, glibc does not return it, and
    ``PeakRssSampler.__enter__`` seeds itself with the entry RSS — so a shared
    process makes one arm's peak include another's garbage. Measured: interleaved
    in one process, the streaming figure climbed 1722 -> 2983 -> 3472 MB across
    three identical runs.
    """
    # (label, reader_threads, from_h5ad kwargs)
    arms: list[tuple[str, int | None, dict[str, object]]] = [
        ("streaming", GATED_READER_THREADS, {}),
        ("streaming_default_threads", None, {}),
    ]
    if not skip_materialize:
        arms.append(("materialize", None, {}))

    applicable_extras = []
    for label, (kwargs, datasets) in _EXTRA_ARMS.items():
        if dataset_name in datasets:
            # Pinned to GATED_READER_THREADS so an extra arm's number is
            # comparable with the default arm's rather than with the runner —
            # unless the arm pins its own (budget_bound does, so the budget
            # derates the thread count and not only the queue depth). Resolved
            # **here**, not left to `kwargs.update` inside the worker: the
            # worker ran at 12 while every run's `extra["reader_threads"]` and
            # the log line said 4, and the sweep companion (which resolves it
            # with the same `.get`) then disagreed with this path under the
            # same label.
            arm_threads = kwargs.get("reader_threads", GATED_READER_THREADS)
            arms.append((label, arm_threads, kwargs))
            applicable_extras.append(label)
        else:
            result.metadata.setdefault("extra_arms_skipped", {})[label] = (
                f"{dataset_name} is not in this arm's dataset scope "
                f"({sorted(datasets)})"
            )
    result.metadata["extra_arms"] = applicable_extras

    structural: dict[str, dict | None] = {}
    for label, reader_threads, extra_kwargs in arms:
        scenario = "materialize" if label == "materialize" else "streaming"
        records = _run_arm_subprocess(
            h5ad_path, n_runs, scenario, reader_threads=reader_threads,
            extra_kwargs=extra_kwargs or None,
        )
        for rec in records:
            if rec.get("structural") is not None and label not in structural:
                structural[label] = rec["structural"]
                if rec.get("output_bytes") is not None:
                    result.metadata[f"{label}_output_bytes"] = rec["output_bytes"]
            result.add_run(
                wall_s=rec["wall_s"],
                peak_rss_mb=rec["peak_rss_mb"],
                **_extra_arm_run_extras(
                    label, rec, reader_threads, extra_kwargs or {}
                ),
            )
            result.metadata["scenarios"].append(label)
            log.info(
                "  %s (reader_threads=%s) run %d: wall=%.2fs peak_rss=%.1f MB",
                label, reader_threads, rec["run_idx"],
                rec["wall_s"], rec["peak_rss_mb"],
            )

    # Enforce the `csc_always` premise rather than recording it. The arm exists
    # to measure `write_csc_sidecar`; if `csc="always"` were dropped on either
    # subprocess hop the arm would still emit `csc_always_peak_rss_mb` and pass,
    # having timed the default conversion under the CSC label. Nothing reads
    # `metadata`, so storing `n_csc_shards` there was not a check.
    _assert_csc_arm_built_a_sidecar(applicable_extras, structural)
    _assert_index_arm_changed_the_output(applicable_extras, result)
    _assert_budget_arm_actually_derated(applicable_extras, result)

    result.metadata["gated_reader_threads"] = GATED_READER_THREADS
    result.metadata["structural"] = {
        **structural,
        # `None`, not `False`, when an arm did not run: nothing to compare is not
        # the same claim as "they differ".
        "equal": (
            None if skip_materialize
            else structural.get("streaming") == structural.get("materialize")
        ),
    }
    if not skip_materialize and structural.get("streaming") != structural.get("materialize"):
        log.warning(
            "Structural mismatch streaming=%s materialize=%s",
            structural.get("streaming"), structural.get("materialize"),
        )
    return result


def _extra_arm_run_extras(
    label: str,
    rec: dict,
    reader_threads: int | None,
    kwargs: dict[str, object],
) -> dict[str, object]:
    """The `add_run` extras for one extra-arm record.

    Shared by the default path and the thread-sweep companion because they had
    already diverged once: the sweep path recorded the timings and dropped
    `warning_categories`, the derate counts and
    `peak_over_memory_budget__budget_bound`, so a sweep capture emitted no
    value for the absolute floor that keys off it and skipped the premise
    check entirely. Reviewers reproduced that with a mocked record. One
    function, called twice, is what stops the next key from going missing on
    one path only.
    """
    extras: dict[str, object] = {
        "scenario": label,
        "run_idx": rec["run_idx"],
        "reader_threads": reader_threads,
        f"{label}_peak_rss_mb": rec["peak_rss_mb"],
        f"{label}_wall_s": rec["wall_s"],
        "warning_categories": rec.get("warning_categories") or [],
        "derate_requested_threads": rec.get("derate_requested_threads"),
        "derate_granted_threads": rec.get("derate_granted_threads"),
        "from_h5ad_kwargs": (
            ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) or "(none)"
        ),
    }
    if label == "budget_bound":
        # Dimensionless, and the only key that states the property under test:
        # did the process stay inside the budget it was given? An absolute MB
        # ceiling would also move with the runner's interpreter baseline
        # (~450-520 MB here), which a ratio against the budget absorbs.
        # Mirrors `build_csc`'s `peak_over_memory_limit__*`.
        extras[f"peak_over_memory_budget__{label}"] = (
            rec["peak_rss_mb"] / _BUDGET_ARM_MB
        )
    return extras


def _run_extra_arms_once(
    h5ad_path: Path,
    result: BenchmarkResult,
    dataset_name: str | None,
) -> None:
    """Run each in-scope `_EXTRA_ARMS` entry once, at its own pinned threads.

    `GATED_READER_THREADS` for `csc_always` / `index_preset_cellxgene`;
    whatever the arm put in its own kwargs otherwise (`budget_bound` pins 12).

    Used by the thread-scaling path, which sweeps `streaming` vs `materialize`
    and has no reason to sweep these — neither arm varies along the thread axis;
    each changes a *conversion option*. Documenting the omission was not enough:
    a sweep run would silently carry none of the non-default coverage the arms
    exist to provide. Found by review (codex - gpt-5.6-terra).

    One run each, not `n_runs`: the sweep is already the expensive path, and
    these are here for coverage rather than for a median.
    """
    structural: dict[str, dict | None] = {}
    for label, (kwargs, datasets) in _EXTRA_ARMS.items():
        if dataset_name not in datasets:
            continue
        # An arm may pin its own `reader_threads` in `kwargs` (budget_bound
        # does, so the budget derates the thread count and not only the queue
        # depth). Honour it here too, or the sweep companion runs a different
        # configuration from the default path under the same label.
        arm_threads = kwargs.get("reader_threads", GATED_READER_THREADS)
        records = _run_arm_subprocess(
            h5ad_path, 1, "streaming",
            reader_threads=arm_threads, extra_kwargs=kwargs,
        )
        for rec in records:
            # Kept, not discarded: the premise checks below and after the sweep
            # read exactly these. An earlier version of this function recorded
            # the timings and dropped both, which reopened on this path the
            # "wrong path, right label" hole the default path had just closed.
            if rec.get("structural") is not None and label not in structural:
                structural[label] = rec["structural"]
                if rec.get("output_bytes") is not None:
                    result.metadata[f"{label}_output_bytes"] = rec["output_bytes"]
            result.add_run(
                wall_s=rec["wall_s"],
                peak_rss_mb=rec["peak_rss_mb"],
                **_extra_arm_run_extras(
                    label, rec, arm_threads, kwargs
                ),
            )
            result.metadata["scenarios"].append(label)
        result.metadata.setdefault("extra_arms", []).append(label)
        log.info("  %s (thread-sweep companion run) recorded", label)

    # The CSC premise needs only this arm's own record, so it is enforced here.
    # The index premise compares against the *default* arm's output size, which
    # the sweep has not written yet — the caller runs that one afterwards.
    _assert_csc_arm_built_a_sidecar(
        list(result.metadata.get("extra_arms", [])), structural
    )
    # Self-contained too: it reads only this arm's own runs.
    _assert_budget_arm_actually_derated(
        list(result.metadata.get("extra_arms", [])), result
    )


def _run_with_thread_scaling(
    h5ad_path: Path,
    n_runs: int,
    thread_counts: list[int],
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Thread-scaling path: for each thread count in `thread_counts`,
    spawn a subprocess with `RAYON_NUM_THREADS={count}` and run paired
    (streaming, materialize) conversions inside. Aggregate per-(count,
    scenario) median wall / max peak RSS into the result metadata."""
    # {scenario: {thread_count_str: [wall_s, ...]}}
    walls: dict[str, dict[str, list[float]]] = {"streaming": {}, "materialize": {}}
    rss: dict[str, dict[str, list[float]]] = {"streaming": {}, "materialize": {}}
    structural_stream: dict[str, int] | None = None
    structural_bulk: dict[str, int] | None = None

    for tc in thread_counts:
        log.info("--- threads=%d (paired streaming + materialize) ---", tc)
        records = _run_thread_count_subprocess(h5ad_path, n_runs, tc)
        tc_str = str(tc)
        for rec in records:
            scenario = rec["scenario"]
            walls[scenario].setdefault(tc_str, []).append(rec["wall_s"])
            rss[scenario].setdefault(tc_str, []).append(rec["peak_rss_mb"])
            structural = rec.get("structural")
            output_bytes = rec.get("output_bytes")
            if scenario == "streaming" and structural is not None:
                if structural_stream is None:
                    structural_stream = structural
                if output_bytes is not None:
                    result.metadata["streaming_output_bytes"] = output_bytes
            elif scenario == "materialize" and structural is not None:
                if structural_bulk is None:
                    structural_bulk = structural
                if output_bytes is not None:
                    result.metadata["materialize_output_bytes"] = output_bytes

            result.add_run(
                wall_s=rec["wall_s"],
                peak_rss_mb=rec["peak_rss_mb"],
                **{
                    "scenario": scenario,
                    "run_idx": rec["run_idx"],
                    "thread_count": tc,
                    # Thread-count-qualified, NOT the bare `{scenario}_peak_rss_mb`.
                    # That bare key is what `thresholds.yaml` floors, and
                    # `_load_current_raw_metric` takes the MEDIAN of every run
                    # carrying it — so emitting it here would make the pinned
                    # rt=4 ceiling a median over whatever thread counts the
                    # operator swept, silently undoing the pinning. Found by
                    # review (codex - gpt-5.6-sol) on PR #451, reproduced with
                    # mocked workers: counts 1 and 16 both emitted the floored
                    # key. `submit_streaming_threads_census1m.sh` exercises this
                    # branch, so it is not hypothetical.
                    f"{scenario}_t{tc}_peak_rss_mb": rec["peak_rss_mb"],
                    f"{scenario}_t{tc}_wall_s": rec["wall_s"],
                },
            )
            result.metadata["scenarios"].append(scenario)
            log.info(
                "  threads=%d %s run %d: wall=%.2fs peak_rss=%.1f MB",
                tc, scenario, rec["run_idx"], rec["wall_s"], rec["peak_rss_mb"],
            )

    # Scaling summary: median wall + max peak RSS per (scenario, threads),
    # plus speedup/efficiency relative to single-thread for that scenario.
    scaling_wall: dict[str, dict[str, float]] = {}
    peak_rss_max: dict[str, dict[str, float]] = {}
    speedup: dict[str, dict[str, float]] = {}
    efficiency: dict[str, dict[str, float]] = {}

    for scenario in ("streaming", "materialize"):
        wall_summary = {
            tc_str: round(statistics.median(samples), 6)
            for tc_str, samples in walls[scenario].items()
        }
        rss_summary = {
            tc_str: round(max(samples), 3)
            for tc_str, samples in rss[scenario].items()
        }
        baseline = wall_summary.get("1")
        sp: dict[str, float] = {}
        eff: dict[str, float] = {}
        if baseline is not None and baseline > 0:
            for tc_str, w in wall_summary.items():
                tc = int(tc_str)
                s = baseline / w if w > 0 else 0.0
                sp[tc_str] = round(s, 3)
                eff[tc_str] = round(s / tc, 3) if tc > 0 else 0.0
        scaling_wall[scenario] = wall_summary
        peak_rss_max[scenario] = rss_summary
        speedup[scenario] = sp
        efficiency[scenario] = eff

    result.metadata["scaling_wall_s"] = scaling_wall
    result.metadata["peak_rss_max_mb"] = peak_rss_max
    result.metadata["speedup"] = speedup
    result.metadata["efficiency"] = efficiency

    result.metadata["structural"] = {
        "streaming": structural_stream,
        "materialize": structural_bulk,
        "equal": structural_stream == structural_bulk,
    }
    if structural_stream != structural_bulk:
        log.warning(
            "Structural mismatch streaming=%s materialize=%s",
            structural_stream,
            structural_bulk,
        )
    return result
