"""
Streaming h5ad → SCX conversion benchmark.

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
level (n_obs / n_vars / nnz / catalog n_csr_shards). Bit-level
equality is asserted in the Rust round-trip test
`scx-convert/src/tests.rs::streaming_round_trip_matches_non_streaming`;
the benchmark only needs to flag drift, not characterise it.

Thread scaling is opt-in via the `SCX_CONV_STREAM_THREAD_COUNTS` env
var (comma-separated, e.g. `1,2,4,8,16,32`). When set, each thread
count is exercised in a fresh subprocess with
`RAYON_NUM_THREADS=N`. Subprocess isolation is mandatory: rayon's
global pool is initialised once on first use and cannot be reconfigured
in-process. Unset → in-process at the inherited thread count
(existing behaviour; smoke runs don't pay subprocess overhead).
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

from benchmarks.comprehensive.config import DatasetConfig
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import current_rss_mb


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _structural_summary(scx_path: Path) -> dict[str, int]:
    """Read an SCX file and return the structural fingerprint
    (n_obs / n_vars / nnz / n_csr_shards / n_csc_shards) used for
    cross-path equality. Bit-level equality is asserted in the Rust
    tests — see Phase 8."""
    import pyscx

    reader = pyscx.open(str(scx_path))
    return {
        "n_obs": int(reader.n_obs),
        "n_vars": int(reader.n_vars),
        "nnz": int(reader.nnz),
        "shard_count": int(reader.shard_count),
    }


def _timed_streaming(h5ad_path: Path, out_path: Path) -> dict[str, float]:
    """Run `pyscx.from_h5ad` and capture wall + RSS deltas.

    Peak RSS uses the same max(before, after) sampling pattern as
    `FormatRunner.timed_run` (no sampler-thread upgrade yet). For
    short conversions this under-reports the true peak; the
    streaming path's memory profile is bounded structurally so the
    under-report is harmless for regression detection. Wall clock is
    monotonic.
    """
    import pyscx

    rss_before = current_rss_mb()
    t0 = time.perf_counter()
    pyscx.from_h5ad(str(h5ad_path), str(out_path))
    wall = time.perf_counter() - t0
    rss_after = current_rss_mb()
    return {"wall_s": wall, "peak_rss_mb": max(rss_before, rss_after)}


def _timed_materialize(h5ad_path: Path, out_path: Path) -> dict[str, float]:
    """Run `pyscx.from_anndata(sc.read_h5ad(path), out)` and capture
    wall + RSS. Includes the h5ad read into Python (anndata) — the
    streaming path bypasses that entirely, so the comparison is
    apples-to-apples at the user-visible "convert this h5ad" level."""
    import anndata
    import pyscx

    rss_before = current_rss_mb()
    t0 = time.perf_counter()
    adata = anndata.read_h5ad(h5ad_path)
    pyscx.from_anndata(adata, str(out_path))
    wall = time.perf_counter() - t0
    rss_after = current_rss_mb()
    return {"wall_s": wall, "peak_rss_mb": max(rss_before, rss_after)}


# ---------------------------------------------------------------------------
# Thread-scaling worker (subprocess-per-thread-count)
# ---------------------------------------------------------------------------

# Worker executed in a subprocess for one thread count. Runs `n_runs`
# paired (streaming, materialize) conversions on the supplied h5ad,
# emits one JSON list to stdout where each element is
# `{"scenario", "run_idx", "wall_s", "peak_rss_mb",
#   "structural"?, "output_bytes"?}`. The first run of each scenario
# also reports the structural fingerprint + output size so the parent
# can detect path drift without re-opening files.
_WORKER_SCRIPT = textwrap.dedent("""\
    import json
    import sys
    import tempfile
    import time
    from pathlib import Path

    h5ad_path = sys.argv[1]
    n_runs = int(sys.argv[2])

    from benchmarks.comprehensive.benchmarks.conversion_streaming import (
        _timed_streaming,
        _timed_materialize,
        _structural_summary,
    )

    records = []
    structural_stream = None
    structural_bulk = None
    streaming_output_bytes = None
    materialize_output_bytes = None

    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory(prefix="scx_bench_stream_") as tmp:
            stream_out = Path(tmp) / "stream.scx"
            t = _timed_streaming(Path(h5ad_path), stream_out)
            rec = {
                "scenario": "streaming",
                "run_idx": run_idx,
                "wall_s": t["wall_s"],
                "peak_rss_mb": t["peak_rss_mb"],
            }
            if structural_stream is None:
                structural_stream = _structural_summary(stream_out)
                streaming_output_bytes = stream_out.stat().st_size
                rec["structural"] = structural_stream
                rec["output_bytes"] = streaming_output_bytes
            records.append(rec)

        with tempfile.TemporaryDirectory(prefix="scx_bench_bulk_") as tmp:
            bulk_out = Path(tmp) / "bulk.scx"
            t = _timed_materialize(Path(h5ad_path), bulk_out)
            rec = {
                "scenario": "materialize",
                "run_idx": run_idx,
                "wall_s": t["wall_s"],
                "peak_rss_mb": t["peak_rss_mb"],
            }
            if structural_bulk is None:
                structural_bulk = _structural_summary(bulk_out)
                materialize_output_bytes = bulk_out.stat().st_size
                rec["structural"] = structural_bulk
                rec["output_bytes"] = materialize_output_bytes
            records.append(rec)

    print(json.dumps(records))
""")


def _run_thread_count_subprocess(
    h5ad_path: Path,
    n_runs: int,
    thread_count: int,
) -> list[dict]:
    """Run paired streaming + materialize conversions in a subprocess
    with `RAYON_NUM_THREADS={thread_count}` (and friends) set.

    Returns the worker's parsed JSON records. Failure surfaces the
    worker's stderr verbatim — debugging a `parallel_write_scaling`
    timing regression is far easier with the inner traceback visible.
    """
    env = os.environ.copy()
    for var in _THREAD_ENV_VARS:
        env[var] = str(thread_count)

    proc = subprocess.run(
        [sys.executable, "-c", _WORKER_SCRIPT, str(h5ad_path), str(n_runs)],
        capture_output=True,
        text=True,
        env=env,
        timeout=14400,  # streaming + materialize on census_10m comfortably under 4 h
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"conversion_streaming worker failed (threads={thread_count}, "
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

    h5ad_path = dataset.h5ad_path
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Source h5ad not found: {h5ad_path}")

    source_bytes = h5ad_path.stat().st_size
    thread_counts = _parse_thread_counts(os.environ.get(_THREAD_COUNTS_ENV))

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

    if thread_counts is None:
        return _run_in_process(h5ad_path, n_runs, result)
    return _run_with_thread_scaling(h5ad_path, n_runs, thread_counts, result)


def _run_in_process(
    h5ad_path: Path,
    n_runs: int,
    result: BenchmarkResult,
) -> BenchmarkResult:
    """Legacy path: run paired streaming + materialize in-process at
    the inherited thread count. Used when
    `SCX_CONV_STREAM_THREAD_COUNTS` is unset — smoke runs and the
    existing census_1m gate keep their wall budget low by skipping
    subprocess overhead."""
    structural_stream: dict[str, int] | None = None
    structural_bulk: dict[str, int] | None = None

    for run_idx in range(n_runs):
        with tempfile.TemporaryDirectory(prefix="scx_bench_stream_") as tmp:
            stream_out = Path(tmp) / "stream.scx"
            timings = _timed_streaming(h5ad_path, stream_out)
            # `streaming_peak_rss_mb` / `streaming_wall_s` are mirrored
            # into `extra` so `compare_against_baseline.py --gate` (which
            # only reads metrics from `runs[].extra`, see
            # `_load_current_raw_metric`) can floor the streaming
            # scenario without dragging the materialise scenario into
            # the median.
            result.add_run(
                wall_s=timings["wall_s"],
                peak_rss_mb=timings["peak_rss_mb"],
                extra={
                    "scenario": "streaming",
                    "run_idx": run_idx,
                    "streaming_peak_rss_mb": timings["peak_rss_mb"],
                    "streaming_wall_s": timings["wall_s"],
                },
            )
            result.metadata["scenarios"].append("streaming")
            if structural_stream is None:
                structural_stream = _structural_summary(stream_out)
                result.metadata["streaming_output_bytes"] = stream_out.stat().st_size
            log.info(
                "  streaming run %d: wall=%.2fs peak_rss=%.1f MB",
                run_idx,
                timings["wall_s"],
                timings["peak_rss_mb"],
            )

        with tempfile.TemporaryDirectory(prefix="scx_bench_bulk_") as tmp:
            bulk_out = Path(tmp) / "bulk.scx"
            timings = _timed_materialize(h5ad_path, bulk_out)
            result.add_run(
                wall_s=timings["wall_s"],
                peak_rss_mb=timings["peak_rss_mb"],
                extra={
                    "scenario": "materialize",
                    "run_idx": run_idx,
                    "materialize_peak_rss_mb": timings["peak_rss_mb"],
                    "materialize_wall_s": timings["wall_s"],
                },
            )
            result.metadata["scenarios"].append("materialize")
            if structural_bulk is None:
                structural_bulk = _structural_summary(bulk_out)
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
                extra={
                    "scenario": scenario,
                    "run_idx": rec["run_idx"],
                    "thread_count": tc,
                    f"{scenario}_peak_rss_mb": rec["peak_rss_mb"],
                    f"{scenario}_wall_s": rec["wall_s"],
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
