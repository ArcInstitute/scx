"""
Benchmark result schema and JSON writer utilities.

All benchmark results are structured as JSON matching the schema defined in
COMPREHENSIVE-BENCHMARKING.md §5.3. This module provides:

  - ``BenchmarkResult``: dataclass for a single benchmark result
  - ``write_result()``: write a result to the raw results directory
  - ``load_result()``: load a result JSON file
  - ``load_all_results()``: load all results for a given benchmark type
"""

from __future__ import annotations

import datetime
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchmarks.comprehensive.config import RAW_RESULTS_DIR
from benchmarks.comprehensive.provenance import capture_run_provenance
from benchmarks.comprehensive.sysinfo import collect_system_info

# Bumped on any additive field to ``BenchmarkResult`` / ``RunRecord``. Readers
# that hard-code the schema (the gate, reports) verify this and refuse to
# diff across incompatible versions. Migration helper lives at
# `scripts/migrate_results.py`.
#
# v2: adds ``wall_s_iqr`` / ``n_runs`` to BenchmarkResult.to_dict() and to the
# rows of capture_baseline.py's summary.json so the gate can widen
# ``median_wall_s`` tolerance by the baseline's own measured noise (review
# §1.1). Older v1 raw JSONs and pre-bump promoted baselines just lack the
# field and the gate falls back to the fixed --timing-tolerance.
SCHEMA_VERSION = 2


@dataclass
class RunRecord:
    """A single benchmark run measurement."""
    wall_s: float
    user_s: float = 0.0
    sys_s: float = 0.0
    peak_rss_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Structured benchmark result matching JSON schema from §5.3.

    Example JSON output::

        {
          "benchmark": "read_full",
          "format": "scx_auto",
          "dataset": "census_1m",
          "timestamp": "2026-03-25T10:00:00",
          "system": { "hostname": "...", "cpu": "...", "ram_gb": 2113 },
          "runs": [
            { "wall_s": 12.491, "user_s": 11.2, "sys_s": 1.1, "peak_rss_mb": 264.2 },
            ...
          ],
          "median_wall_s": 12.491,
          "file_size_bytes": 2470000000
        }
    """
    benchmark: str                         # Which benchmark (e.g. "read_full")
    format: str                            # Format key (e.g. "scx_auto")
    dataset: str                           # Dataset name (e.g. "census_1m")
    runs: list[RunRecord] = field(default_factory=list)
    file_size_bytes: int | None = None
    timestamp: str = ""
    system: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat(timespec="seconds")
        if not self.system:
            self.system = collect_system_info()
        # Embed per-run provenance (git SHA, thread pinning, run_id) under
        # system.provenance. Always overwrites even for pre-populated system
        # dicts — provenance should reflect THIS run, not whatever was
        # pickled / inherited from a prior invocation.
        self.system["provenance"] = capture_run_provenance()

    @property
    def median_wall_s(self) -> float | None:
        """Median wall-clock time across runs."""
        if not self.runs:
            return None
        return statistics.median(r.wall_s for r in self.runs)

    @property
    def median_rss_mb(self) -> float | None:
        """Median peak RSS across runs."""
        if not self.runs:
            return None
        return statistics.median(r.peak_rss_mb for r in self.runs)

    @property
    def wall_s_iqr(self) -> float | None:
        """Inter-quartile range (p75 - p25) of wall_s across runs.

        Used by the gate to widen the per-row timing tolerance: a noisy
        baseline gets a relaxed bound proportional to its own measured
        dispersion (review §1.1). Returns ``None`` for fewer than 3 runs
        — IQR is not reliably estimable and the gate falls back to the
        fixed ``--timing-tolerance``. The n=2 case is specifically
        excluded: ``statistics.quantiles(..., method="exclusive")``
        extrapolates to give IQR = 1.5·|a−b|, and even the range |a−b|
        easily produces 100%+ widened tolerances that would mask real
        regressions (e.g. baseline runs of 1.0 s and 2.0 s ⇒ 100%
        widened bound at the default ``--iqr-k=1.5``). Falling back to
        the fixed tolerance is safer when the typical big-dataset path
        (``N_RUNS_LARGE=3``) drops a run upstream.
        """
        if len(self.runs) < 3:
            return None
        walls = [r.wall_s for r in self.runs]
        q = statistics.quantiles(walls, n=4, method="exclusive")
        return float(q[2] - q[0])

    @property
    def n_runs(self) -> int:
        """Number of recorded timed runs (warmup excluded)."""
        return len(self.runs)

    def add_run(
        self,
        wall_s: float,
        user_s: float = 0.0,
        sys_s: float = 0.0,
        peak_rss_mb: float = 0.0,
        **extra: Any,
    ) -> None:
        """Add a run measurement."""
        self.runs.append(RunRecord(
            wall_s=wall_s, user_s=user_s, sys_s=sys_s,
            peak_rss_mb=peak_rss_mb, extra=extra,
        ))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": self.benchmark,
            "format": self.format,
            "dataset": self.dataset,
            "timestamp": self.timestamp,
            "system": self.system,
            "runs": [asdict(r) for r in self.runs],
            "median_wall_s": self.median_wall_s,
            "wall_s_iqr": self.wall_s_iqr,
            "n_runs": self.n_runs,
        }
        if self.file_size_bytes is not None:
            d["file_size_bytes"] = self.file_size_bytes
        if self.metadata:
            d["metadata"] = self.metadata
        return d


def _result_filename(benchmark: str, format_key: str, dataset: str) -> str:
    """Generate a consistent filename for a result JSON."""
    return f"{benchmark}__{format_key}__{dataset}.json"


def write_result(result: BenchmarkResult) -> Path:
    """Write a BenchmarkResult to the raw results directory.

    Returns the path to the written JSON file.
    """
    RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = _result_filename(result.benchmark, result.format, result.dataset)
    path = RAW_RESULTS_DIR / filename
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
    return path


def load_result(path: str | Path) -> dict[str, Any]:
    """Load a single result JSON file."""
    with open(path) as f:
        return json.load(f)


def load_all_results(
    benchmark: str | None = None,
    format_key: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Load all matching result JSON files from the raw results directory.

    Parameters
    ----------
    benchmark : filter to a specific benchmark type (e.g. "read_full")
    format_key : filter to a specific format (e.g. "scx_auto")
    dataset : filter to a specific dataset (e.g. "census_1m")

    Returns
    -------
    List of result dicts, sorted by timestamp.
    """
    results = []
    if not RAW_RESULTS_DIR.exists():
        return results

    for path in sorted(RAW_RESULTS_DIR.glob("*.json")):
        try:
            data = load_result(path)
        except (json.JSONDecodeError, OSError):
            continue

        if benchmark and data.get("benchmark") != benchmark:
            continue
        if format_key and data.get("format") != format_key:
            continue
        if dataset and data.get("dataset") != dataset:
            continue

        results.append(data)

    return sorted(results, key=lambda d: d.get("timestamp", ""))
