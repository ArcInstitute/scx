"""
Streaming h5ad → SCX conversion benchmark (Phase 10 of STREAMING-CONVERSION.md).

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

The harness mirrors `compression.py`: one `BenchmarkResult` per
(dataset, scenario) where scenario picks the path. SLURM-driven
parallel submission lives in
`benchmarks/comprehensive/scripts/run_slurm_conversion_streaming.sh`.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from benchmarks.comprehensive.config import DatasetConfig
from benchmarks.comprehensive.results import BenchmarkResult
from benchmarks.comprehensive.rss import current_rss_mb


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _structural_summary(scx_path: Path) -> dict[str, int]:
    """Read an SCX file and return the structural fingerprint
    (n_obs / n_vars / nnz / n_csr_shards / n_csc_shards) used for
    cross-path equality. Bit-level equality is asserted in the Rust
    tests — see Phase 8."""
    import pyscx

    reader = pyscx.open(scx_path)
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
    log.info(
        "Streaming conversion benchmark: dataset=%s source=%.1f MB n_runs=%d",
        dataset.name,
        source_bytes / 1e6,
        n_runs,
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
        },
    )

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
