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

    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.to_h5ad(str(scx_path), str(out_path), stream=stream)
    wall = time.perf_counter() - t0
    return {"wall_s": wall, "peak_rss_mb": sampler.peak_mb}


def _timed_export_h5mu(
    scx_path: Path,
    out_path: Path,
    stream: bool,
) -> dict[str, float]:
    """Multimodal sibling of ``_timed_export_h5ad``."""
    import pyscx

    t0 = time.perf_counter()
    with PeakRssSampler() as sampler:
        pyscx.to_h5mu(str(scx_path), str(out_path), stream=stream)
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
    import time
    from pathlib import Path

    scx_path = sys.argv[1]
    out_ext = sys.argv[2]      # "h5ad" or "h5mu"
    n_runs = int(sys.argv[3])

    from benchmarks.comprehensive.benchmarks.export_streaming import (
        _timed_export_h5ad,
        _timed_export_h5mu,
        _structural_summary_h5ad,
        _structural_summary_h5mu,
    )

    timed = _timed_export_h5ad if out_ext == "h5ad" else _timed_export_h5mu
    structural = _structural_summary_h5ad if out_ext == "h5ad" else _structural_summary_h5mu

    records = []
    structural_stream = None
    structural_bulk = None
    streaming_output_bytes = None
    materialize_output_bytes = None

    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory(prefix="scx_bench_export_stream_") as tmp:
            stream_out = Path(tmp) / f"stream.{out_ext}"
            t = timed(Path(scx_path), stream_out, True)
            rec = {
                "scenario": "streaming",
                "run_idx": run_idx,
                "wall_s": t["wall_s"],
                "peak_rss_mb": t["peak_rss_mb"],
            }
            if structural_stream is None:
                structural_stream = structural(stream_out)
                streaming_output_bytes = stream_out.stat().st_size
                rec["structural"] = structural_stream
                rec["output_bytes"] = streaming_output_bytes
            records.append(rec)

        with tempfile.TemporaryDirectory(prefix="scx_bench_export_bulk_") as tmp:
            bulk_out = Path(tmp) / f"bulk.{out_ext}"
            t = timed(Path(scx_path), bulk_out, False)
            rec = {
                "scenario": "materialize",
                "run_idx": run_idx,
                "wall_s": t["wall_s"],
                "peak_rss_mb": t["peak_rss_mb"],
            }
            if structural_bulk is None:
                structural_bulk = structural(bulk_out)
                materialize_output_bytes = bulk_out.stat().st_size
                rec["structural"] = structural_bulk
                rec["output_bytes"] = materialize_output_bytes
            records.append(rec)

    print(json.dumps(records))
""")


def _run_thread_count_subprocess(
    scx_path: Path,
    out_ext: str,
    n_runs: int,
    thread_count: int,
) -> list[dict]:
    """Run paired streaming + materialize exports in a subprocess with
    ``RAYON_NUM_THREADS={thread_count}`` (and friends) set."""
    env = os.environ.copy()
    for var in _THREAD_ENV_VARS:
        env[var] = str(thread_count)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            _WORKER_SCRIPT,
            str(scx_path),
            out_ext,
            str(n_runs),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=14400,  # census_10m export under 4 h with comfortable headroom
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"export_streaming worker failed (threads={thread_count}, "
            f"exit={proc.returncode}).\n"
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
            return _run_in_process(scx_in, out_ext, n_runs, result)
        return _run_with_thread_scaling(
            scx_in, out_ext, n_runs, thread_counts, result
        )
    finally:
        if _cleanup_input is not None:
            _cleanup_input.cleanup()


def _run_in_process(
    scx_in: Path,
    out_ext: str,
    n_runs: int,
    result: BenchmarkResult,
) -> BenchmarkResult:
    """In-process variant: paired streaming + materialise per run at
    the inherited thread count. Smoke runs and the existing
    ``census_1m`` gate keep their wall budget low by skipping the
    subprocess hop.

    NB: We pass scenario / wall / RSS as flat kwargs to ``add_run``
    (not via ``extra={...}``) so the keys land at
    ``runs[i].extra.<metric>`` and the threshold gate's
    ``run.get("extra", {}).get(metric)`` lookup finds them. The
    ingestion benchmark nests these under ``extra.extra`` due to a
    historical kwarg quirk; this module deliberately keeps the flat
    shape so the floor at ``thresholds.yaml`` works end-to-end.
    """
    timed = _timed_export_h5ad if out_ext == "h5ad" else _timed_export_h5mu
    structural = (
        _structural_summary_h5ad if out_ext == "h5ad" else _structural_summary_h5mu
    )

    structural_stream: dict[str, int] | None = None
    structural_bulk: dict[str, int] | None = None

    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory(prefix="scx_bench_export_stream_") as tmp:
            stream_out = Path(tmp) / f"stream.{out_ext}"
            timings = timed(scx_in, stream_out, True)
            result.add_run(
                wall_s=timings["wall_s"],
                peak_rss_mb=timings["peak_rss_mb"],
                scenario="streaming",
                run_idx=run_idx,
                streaming_peak_rss_mb=timings["peak_rss_mb"],
                streaming_wall_s=timings["wall_s"],
            )
            result.metadata["scenarios"].append("streaming")
            if structural_stream is None:
                structural_stream = structural(stream_out)
                result.metadata["streaming_output_bytes"] = stream_out.stat().st_size
            log.info(
                "  streaming run %d: wall=%.2fs peak_rss=%.1f MB",
                run_idx,
                timings["wall_s"],
                timings["peak_rss_mb"],
            )

        with tempfile.TemporaryDirectory(prefix="scx_bench_export_bulk_") as tmp:
            bulk_out = Path(tmp) / f"bulk.{out_ext}"
            timings = timed(scx_in, bulk_out, False)
            result.add_run(
                wall_s=timings["wall_s"],
                peak_rss_mb=timings["peak_rss_mb"],
                scenario="materialize",
                run_idx=run_idx,
                materialize_peak_rss_mb=timings["peak_rss_mb"],
                materialize_wall_s=timings["wall_s"],
            )
            result.metadata["scenarios"].append("materialize")
            if structural_bulk is None:
                structural_bulk = structural(bulk_out)
                result.metadata["materialize_output_bytes"] = bulk_out.stat().st_size
            log.info(
                "  materialize run %d: wall=%.2fs peak_rss=%.1f MB",
                run_idx,
                timings["wall_s"],
                timings["peak_rss_mb"],
            )

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
                f"{scenario}_peak_rss_mb": rec["peak_rss_mb"],
                f"{scenario}_wall_s": rec["wall_s"],
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
