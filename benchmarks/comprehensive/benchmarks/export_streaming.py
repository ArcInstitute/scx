"""
Streaming SCX → h5ad / h5mu export benchmark (Phase 8).

The inverse of ``conversion_streaming.py``: takes a pre-built SCX file
and runs both export entry points back-to-back, capturing peak RSS,
wall clock, and basic output-equality metadata for each path:

- **streaming** — ``pyscx.to_h5ad(scx, out)`` (or ``to_h5mu`` for
  multimodal inputs), Phase 8's
  ``scx_convert::scx_to_h5ad_streaming`` /
  ``scx_to_h5mu_streaming`` / ``scx_modality_to_h5ad_streaming``.
  Peak memory bounded by one shard's worth of CSR per matrix
  written, plus the always-resident kept-row ``indptr``.
- **materialize** — ``pyscx.to_h5ad(scx, out, stream=False)`` (or
  ``to_h5mu(..., stream=False)``), the legacy in-memory path.
  Peak memory scales with the full CSR triplet.

Output equality between the two paths is measured at the
"structural" level (n_obs / n_vars / nnz / shard count read back via
``anndata`` / ``mudata``); bit-level equality is asserted in the
Rust round-trip tests at
``scx-convert/src/tests.rs::test_h5ad_csr_to_scx_to_h5ad_streaming_round_trip``
and friends. The benchmark only needs to flag drift, not
characterise it.

Thread scaling is opt-in via the
``SCX_EXPORT_STREAM_THREAD_COUNTS`` env var (comma-separated, e.g.
``1,2,4,8,16,32``). Mirrors the ingestion benchmark's
``SCX_CONV_STREAM_THREAD_COUNTS`` knob. Unset → in-process at the
inherited thread count (existing behaviour; smoke runs don't pay
subprocess overhead).
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig, FormatVariant
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import PeakRssSampler


log = logging.getLogger(__name__)

# Env vars set on subprocess workers so any thread pool (rayon, BLAS,
# OpenMP) honours the requested thread count. Mirrors
# ``conversion_streaming.py``'s ``_THREAD_ENV_VARS``.
_THREAD_ENV_VARS = (
    "RAYON_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
)

_THREAD_COUNTS_ENV = "SCX_EXPORT_STREAM_THREAD_COUNTS"

# Format keys that consume a multimodal SCX input. Multimodal datasets
# only run with these — single-modality SCX writers can't represent the
# multimodal layout.
_MULTIMODAL_SCX_KEYS = (
    "scx_multimodal_per_modality_auto",
    "scx_multimodal_uniform_auto",
)

# Declared as a module constant (not only as the inline `run()` guard below) so
# `run_parallel._bench_format_compatible` filters at cohort-build time; without
# it the orchestrator schedules one job per format in the tier, each returning
# `None`, which is the phantom `missing_result` case that function exists to
# prevent.
#
# **Only the single-modality key is listed, and the multimodal arm below is
# therefore not reachable through `run_parallel` today.** That is an orchestrator
# constraint, not an oversight: `_bench_format_compatible` requires
# `bench_is_multimodal == fmt_is_multimodal`, where `bench_is_multimodal` is
# membership in `_MULTIMODAL_BENCHMARKS` — so adding this benchmark to that set
# to reach `_MULTIMODAL_SCX_KEYS` would make its single-modality arm unreachable
# instead. The h5mu arm stays exercised out-of-band via
# `scripts/run_slurm_export_streaming.sh` and by the Rust round-trip tests named
# in this module's docstring. Named here rather than left silently
# unschedulable.
SUPPORTED_FORMATS: frozenset[str] = frozenset({"scx_auto"})

# Reader threads for the **gated** streaming arm. Mirrors
# `conversion_streaming.GATED_READER_THREADS` and exists for the same reason: the
# parallel export reader's bound is `shards_in_flight x per_shard`, so with
# `reader_threads` left to `available_parallelism()` the `streaming_peak_rss_mb`
# floor's value is a property of the runner. Measured on census_1m at the
# inherited 16 threads: 5.42-5.87 GB true peak. Pinned here so the floor tests
# the contract, not the core count; the default-parallelism number is recorded
# under `streaming_default_threads_peak_rss_mb` and not gated.
GATED_READER_THREADS: int = 4

# Above this many cells the `materialize` arm is skipped and only the streaming
# arms run. Mirrors `conversion_streaming.MATERIALIZE_MAX_N_OBS`, and it is
# needed for the same reason: registering this module in `ALL_BENCHMARKS` is what
# makes a tiered capture schedule it, and `to_h5ad(..., stream=False)` calls
# `read_all_csr_shards_filtered()` — the whole CSR triplet resident, measured at
# **13.98 GB** on census_1m and scaling linearly, so ~70 GB at census_5m and
# ~140 GB at census_10m. `estimate_memory_gb` has no arm for either streaming
# benchmark, so those jobs fall through to `peak_mb = base_mb` (2x the h5ad size)
# and would be under-provisioned. The ingest twin had this cap and this module
# did not — the asymmetry was found by review (Cursor Agent - Grok 4.6 High) on
# PR #451.
#
# The streaming arms — the ones `thresholds.yaml` floors — still run at every
# tier. The skip is recorded in `metadata`, never silent.
MATERIALIZE_MAX_N_OBS: int = 1_000_000


def _skip_materialize_reason(n_obs: int) -> str | None:
    """Why the materialize arm is not run at this scale, or `None` to run it."""
    if n_obs > MATERIALIZE_MAX_N_OBS:
        return (
            f"materialize holds the whole CSR triplet (~14 GB at 1M cells, "
            f"scaling linearly); skipped above n_obs={MATERIALIZE_MAX_N_OBS:,} so "
            f"a full-tier capture does not pay it. The streaming arms — the ones "
            f"the thresholds.yaml floor gates — still run."
        )
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _structural_summary_h5ad(h5ad_path: Path) -> dict[str, int]:
    """Read back an h5ad output and return the structural fingerprint
    used for cross-path equality detection.

    Reads at the h5py level rather than through ``anndata.read_h5ad``
    so the summary is robust to anndata's strict per-column IOSpec
    validation. The benchmark's purpose is to measure peak RSS of
    the export writer, not to certify anndata compatibility of every
    obs column; the structural fingerprint only needs `/X`'s shape
    and nnz, which are stable HDF5-level facts.
    """
    import h5py

    with h5py.File(h5ad_path, "r") as f:
        x = f["X"]
        shape = list(x.attrs.get("shape", [0, 0]))
        nnz = int(x["data"].shape[0]) if "data" in x else 0
    return {
        "n_obs": int(shape[0]) if len(shape) >= 1 else 0,
        "n_vars": int(shape[1]) if len(shape) >= 2 else 0,
        "nnz": nnz,
    }


def _structural_summary_h5mu(h5mu_path: Path) -> dict[str, int]:
    """Per-modality structural fingerprint sum: n_obs / n_vars / nnz
    aggregated across modalities. Avoids per-modality dict explosion
    in the result JSON while still flagging drift. h5py-only so it
    isn't gated on `mudata` (which isn't installed in every benchmark
    env)."""
    import h5py

    with h5py.File(h5mu_path, "r") as f:
        mod = f["mod"]
        n_obs = 0
        n_vars_total = 0
        nnz_total = 0
        n_modalities = 0
        for mname in mod:
            mod_root = mod[mname]
            x = mod_root["X"]
            shape = list(x.attrs.get("shape", [0, 0]))
            if n_obs == 0 and len(shape) >= 1:
                n_obs = int(shape[0])
            if len(shape) >= 2:
                n_vars_total += int(shape[1])
            if "data" in x:
                nnz_total += int(x["data"].shape[0])
            n_modalities += 1
    return {
        "n_obs": n_obs,
        "n_vars": n_vars_total,
        "nnz": nnz_total,
        "n_modalities": n_modalities,
    }


def _convert_to_scx(
    dataset: DatasetConfig,
    scx_out: Path,
) -> None:
    """Ingest the dataset's source h5ad / h5mu into ``scx_out``.

    Single-modality datasets go through ``pyscx.from_h5ad`` (streaming
    ingest, default codec). Multimodal datasets go through
    ``pyscx.from_h5mu`` so the multimodal layout is preserved.

    This is only used when the harness didn't pass ``converted_path``
    — ``run_parallel.py`` provides a pre-converted SCX file from the
    shared convert phase, so the in-benchmark conversion is the
    ``run_all.py`` fallback only.
    """
    import pyscx

    if dataset.multimodal:
        pyscx.from_h5mu(str(dataset.h5mu_path), str(scx_out))
    else:
        pyscx.from_h5ad(str(dataset.h5ad_path), str(scx_out))


def _timed_export_h5ad(
    scx_path: Path,
    out_path: Path,
    stream: bool,
    reader_threads: int | None = None,
) -> dict[str, float]:
    """Run ``pyscx.to_h5ad`` and capture wall + the true in-region peak RSS.

    Uses :class:`PeakRssSampler`, as ``ooc_rss_boundary`` / ``ooc_loader`` /
    ``cellset_gather`` / ``obs_open`` / ``shuffle_layout`` / ``grouped_read``
    already do.

    It previously took ``max(before, after)`` of the *instantaneous* RSS, and
    argued in this docstring that under-reporting was "harmless for regression
    detection" because "the streaming path's memory profile is bounded
    structurally" — which assumes exactly the property the
    ``streaming_peak_rss_mb`` floor exists to check. Measured consequence: the
    materialize arm, which calls ``read_all_csr_shards_filtered()`` and holds
    the whole ~11 GB CSR triplet for census_1m, reported **913 MB** — within 1%
    of the streaming arm. The comparison the benchmark is named for was
    indistinguishable in both arms, and the floor gated a number that was not a
    peak.
    """
    import pyscx

    kwargs = {} if reader_threads is None else {"reader_threads": reader_threads}
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.to_h5ad(str(scx_path), str(out_path), stream=stream, **kwargs)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "peak_rss_mb": sampler.peak_mb}


def _timed_export_h5mu(
    scx_path: Path,
    out_path: Path,
    stream: bool,
    reader_threads: int | None = None,
) -> dict[str, float]:
    """Multimodal sibling of ``_timed_export_h5ad``."""
    import pyscx

    kwargs = {} if reader_threads is None else {"reader_threads": reader_threads}
    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.to_h5mu(str(scx_path), str(out_path), stream=stream, **kwargs)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "peak_rss_mb": sampler.peak_mb}


# ---------------------------------------------------------------------------
# Thread-scaling worker (subprocess-per-thread-count)
# ---------------------------------------------------------------------------

# Worker executed in a subprocess for one thread count. Runs ``n_runs``
# paired (streaming, materialize) exports against the pre-built SCX and
# emits one JSON list to stdout, one element per run. Mirrors the
# ingestion benchmark's worker pattern.
_WORKER_SCRIPT = textwrap.dedent("""\
    import json
    import sys
    import tempfile
    from pathlib import Path

    scx_path = sys.argv[1]
    out_ext = sys.argv[2]      # "h5ad" or "h5mu"
    n_runs = int(sys.argv[3])
    scenario = sys.argv[4]     # "streaming" | "materialize"
    rt = sys.argv[5]           # reader_threads, or "" to inherit
    reader_threads = int(rt) if rt else None

    from benchmarks.comprehensive.benchmarks.export_streaming import (
        _timed_export_h5ad,
        _timed_export_h5mu,
        _structural_summary_h5ad,
        _structural_summary_h5mu,
    )

    timed = _timed_export_h5ad if out_ext == "h5ad" else _timed_export_h5mu
    structural_of = (
        _structural_summary_h5ad if out_ext == "h5ad" else _structural_summary_h5mu
    )
    stream = scenario == "streaming"
    prefix = "scx_bench_export_stream_" if stream else "scx_bench_export_bulk_"
    stem = "stream" if stream else "bulk"

    records = []
    structural = None
    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
            out = Path(tmp) / f"{stem}.{out_ext}"
            t = timed(Path(scx_path), out, stream, reader_threads)
            rec = {
                "scenario": scenario,
                "run_idx": run_idx,
                "reader_threads": reader_threads,
                "wall_s": t["wall_s"],
                "peak_rss_mb": t["peak_rss_mb"],
            }
            if structural is None:
                structural = structural_of(out)
                rec["structural"] = structural
                rec["output_bytes"] = out.stat().st_size
            records.append(rec)

    print(json.dumps(records))
""")


def _run_arm_subprocess(
    scx_path: Path,
    out_ext: str,
    n_runs: int,
    scenario: str,
    thread_count: int | None = None,
    reader_threads: int | None = None,
) -> list[dict]:
    """Run ``n_runs`` of **one** arm in a fresh subprocess.

    One arm per process, for the reason spelled out in
    ``conversion_streaming._run_arm_subprocess``: the materialize arm holds the
    whole CSR triplet (14.1-14.2 GB measured on census_1m), glibc does not return
    it, and ``PeakRssSampler.__enter__`` seeds itself with the entry RSS — so in a
    shared process one arm's peak includes the other's garbage. Visible even here,
    where the streaming figure crept 5418 -> 5872 MB across three runs.
    """
    env = os.environ.copy()
    if thread_count is not None:
        for var in _THREAD_ENV_VARS:
            env[var] = str(thread_count)

    proc = subprocess.run(
        [
            sys.executable, "-c", _WORKER_SCRIPT,
            str(scx_path), out_ext, str(n_runs), scenario,
            "" if reader_threads is None else str(reader_threads),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=14400,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"export_streaming worker failed (scenario={scenario}, "
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
    scx_path: Path,
    out_ext: str,
    n_runs: int,
    thread_count: int,
) -> list[dict]:
    """Both arms at one thread count, **each in its own subprocess**.

    Was one process running the pair; see :func:`_run_arm_subprocess`.
    """
    return (
        _run_arm_subprocess(scx_path, out_ext, n_runs, "streaming", thread_count)
        + _run_arm_subprocess(scx_path, out_ext, n_runs, "materialize", thread_count)
    )


def _parse_thread_counts(raw: str | None) -> list[int] | None:
    """Parse ``SCX_EXPORT_STREAM_THREAD_COUNTS``. Returns None when
    unset or empty — caller falls back to the in-process single-count
    path."""
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
    format_variant: FormatVariant | None = None,
    n_runs: int = 1,
    cold_cache: bool = False,
    converted_path: Path | None = None,
) -> BenchmarkResult | None:
    """Run the streaming-vs-materialising **export** benchmark for one
    dataset.

    Returns ``None`` for (dataset, format_variant) pairs that don't
    match the SCX-only export direction (the harness treats ``None``
    as a successful no-op skip).

    For single-modality datasets the benchmark exercises
    ``pyscx.to_h5ad``; for multimodal datasets (``dataset.multimodal
    == True``) it switches to ``pyscx.to_h5mu`` and the
    ``scx_multimodal_*`` format key.
    """
    # Gating — only run on SCX inputs, in the direction matching the
    # dataset's modality count.
    if format_variant is None:
        return None
    if dataset.multimodal:
        if format_variant.key not in _MULTIMODAL_SCX_KEYS:
            return None
        out_ext = "h5mu"
    else:
        if format_variant.key != "scx_auto":
            return None
        out_ext = "h5ad"

    thread_counts = _parse_thread_counts(os.environ.get(_THREAD_COUNTS_ENV))

    # Resolve the SCX input. Preference order:
    #   1. Caller-supplied `converted_path` (run_parallel.py's
    #      shared convert phase).
    #   2. Pre-existing scx_multimodal_path (multimodal) or
    #      scx_path (single-modality) under DATA_DIR. These are
    #      produced once by `reconvert_fixtures.py` and shared
    #      across the comprehensive suite — avoids re-running
    #      the costly h5mu/h5ad → SCX ingest per benchmark.
    #   3. Fallback: ingest the source h5mu/h5ad inline. Slower
    #      and also more fragile (the h5mu → SCX path doesn't
    #      handle every HDF5 type-conversion case yet).
    _cleanup_input = None
    pre_converted: Path | None = None
    if converted_path is not None and Path(converted_path).exists():
        pre_converted = Path(converted_path)
    elif dataset.multimodal and dataset.scx_multimodal_path.exists():
        pre_converted = dataset.scx_multimodal_path
    elif (not dataset.multimodal) and dataset.scx_path.exists():
        pre_converted = dataset.scx_path

    if pre_converted is not None:
        scx_in = pre_converted
        log.info("Using pre-converted SCX: %s", scx_in)
    else:
        _cleanup_input = tempfile.TemporaryDirectory(
            prefix=f"scx_bench_export_in_{dataset.name}_"
        )
        scx_in = Path(_cleanup_input.name) / f"{dataset.name}.scx"
        log.info(
            "Pre-converting %s → %s (%s)",
            (
                dataset.h5mu_path.name
                if dataset.multimodal
                else dataset.h5ad_path.name
            ),
            scx_in,
            format_variant.key,
        )
        _convert_to_scx(dataset, scx_in)

    scx_size = scx_in.stat().st_size
    log.info(
        "Streaming export benchmark: dataset=%s scx=%.1f MB direction=%s "
        "n_runs=%d thread_counts=%s",
        dataset.name,
        scx_size / 1e6,
        out_ext,
        n_runs,
        thread_counts if thread_counts else "in-process default",
    )

    # ``format`` is fixed to the synthetic key the absolute-floor gate
    # references (``thresholds.yaml``), not the iterating
    # ``format_variant``, because the metric is path-shape, not
    # format-shape — identical to the ingestion benchmark's choice.
    result = BenchmarkResult(
        benchmark="export_streaming",
        format="scx_streaming_vs_materialize",
        dataset=dataset.name,
        metadata={
            "direction": out_ext,
            "source_scx_bytes": scx_size,
            "n_obs": dataset.n_obs,
            "n_vars": dataset.n_vars,
            "multimodal": dataset.multimodal,
            "scenarios": [],
            "thread_counts": thread_counts if thread_counts else None,
        },
    )

    try:
        if thread_counts is None:
            return _run_isolated(scx_in, out_ext, n_runs, result, dataset.n_obs)
        return _run_with_thread_scaling(
            scx_in, out_ext, n_runs, thread_counts, result
        )
    finally:
        if _cleanup_input is not None:
            _cleanup_input.cleanup()


def _run_isolated(
    scx_in: Path,
    out_ext: str,
    n_runs: int,
    result: BenchmarkResult,
    n_obs: int = 0,
) -> BenchmarkResult:
    """Default path: three arms, each in its own subprocess.

    * ``streaming`` at :data:`GATED_READER_THREADS` — carries
      ``streaming_peak_rss_mb``, the key ``thresholds.yaml`` floors, pinned so
      the floor does not depend on the runner's core count.
    * ``streaming_default_threads`` at the inherited parallelism — recorded, not
      gated.
    * ``materialize`` — ``stream=False``, the comparison baseline.

    Replaces the previous in-process paired loop. That loop's docstring noted
    that it passed flat kwargs to ``add_run`` "so the floor at thresholds.yaml
    works end-to-end" and that the ingestion benchmark "nests these under
    ``extra.extra`` due to a historical kwarg quirk" — i.e. the broken ingest
    floor was known here and documented rather than fixed. Both are fixed now,
    and the AST guard in
    ``benchmarks/comprehensive/tests/test_floor_reachability.py`` keeps either
    module from regressing to the nested form.
    """
    skip_materialize = _skip_materialize_reason(n_obs)
    result.metadata["materialize_skipped_reason"] = skip_materialize
    if skip_materialize:
        log.info("materialize arm skipped: %s", skip_materialize)

    arms: list[tuple[str, int | None]] = [
        ("streaming", GATED_READER_THREADS),
        ("streaming_default_threads", None),
    ]
    if not skip_materialize:
        arms.append(("materialize", None))

    structural: dict[str, dict | None] = {}
    for label, reader_threads in arms:
        scenario = "materialize" if label == "materialize" else "streaming"
        records = _run_arm_subprocess(
            scx_in, out_ext, n_runs, scenario, reader_threads=reader_threads,
        )
        for rec in records:
            if rec.get("structural") is not None and label not in structural:
                structural[label] = rec["structural"]
                if rec.get("output_bytes") is not None:
                    result.metadata[f"{label}_output_bytes"] = rec["output_bytes"]
            result.add_run(
                wall_s=rec["wall_s"],
                peak_rss_mb=rec["peak_rss_mb"],
                **{
                    "scenario": label,
                    "run_idx": rec["run_idx"],
                    "reader_threads": reader_threads,
                    f"{label}_peak_rss_mb": rec["peak_rss_mb"],
                    f"{label}_wall_s": rec["wall_s"],
                },
            )
            result.metadata["scenarios"].append(label)
            log.info(
                "  %s (reader_threads=%s) run %d: wall=%.2fs peak_rss=%.1f MB",
                label, reader_threads, rec["run_idx"],
                rec["wall_s"], rec["peak_rss_mb"],
            )

    result.metadata["gated_reader_threads"] = GATED_READER_THREADS
    result.metadata["structural"] = {
        **structural,
        # `None`, not `False`, when the materialize arm did not run: nothing to
        # compare is not the same claim as "they differ".
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


def _run_with_thread_scaling(
    scx_in: Path,
    out_ext: str,
    n_runs: int,
    thread_counts: list[int],
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Thread-scaling path: for each thread count, spawn a subprocess
    with ``RAYON_NUM_THREADS={count}`` (and friends) set, run paired
    (streaming, materialize) exports inside, then aggregate per-
    (count, scenario) median wall / max peak RSS into metadata."""
    walls: dict[str, dict[str, list[float]]] = {"streaming": {}, "materialize": {}}
    rss: dict[str, dict[str, list[float]]] = {"streaming": {}, "materialize": {}}
    structural_stream: dict[str, int] | None = None
    structural_bulk: dict[str, int] | None = None

    for tc in thread_counts:
        log.info("--- threads=%d (paired streaming + materialize) ---", tc)
        records = _run_thread_count_subprocess(scx_in, out_ext, n_runs, tc)
        tc_str = str(tc)
        for rec in records:
            scenario = rec["scenario"]
            walls[scenario].setdefault(tc_str, []).append(rec["wall_s"])
            rss[scenario].setdefault(tc_str, []).append(rec["peak_rss_mb"])
            structural_rec = rec.get("structural")
            output_bytes = rec.get("output_bytes")
            if scenario == "streaming" and structural_rec is not None:
                if structural_stream is None:
                    structural_stream = structural_rec
                if output_bytes is not None:
                    result.metadata["streaming_output_bytes"] = output_bytes
            elif scenario == "materialize" and structural_rec is not None:
                if structural_bulk is None:
                    structural_bulk = structural_rec
                if output_bytes is not None:
                    result.metadata["materialize_output_bytes"] = output_bytes

            kwargs = {
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
            }
            result.add_run(
                wall_s=rec["wall_s"],
                peak_rss_mb=rec["peak_rss_mb"],
                **kwargs,
            )
            result.metadata["scenarios"].append(scenario)
            log.info(
                "  threads=%d %s run %d: wall=%.2fs peak_rss=%.1f MB",
                tc,
                scenario,
                rec["run_idx"],
                rec["wall_s"],
                rec["peak_rss_mb"],
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
