"""
Result store: load, normalize, query, and validate raw benchmark JSON results.

Loads raw JSON files *once* and exposes stable query helpers so table builders
and report sections don't re-scan the filesystem independently.  Every cell
in the normalized view carries a ``SourceRef`` (raw_json, external_report,
manual, or unavailable) and a typed ``MissingReason`` for absent values.

Usage::

    from benchmarks.comprehensive.reporting.result_store import ResultStore
    store = ResultStore.from_defaults()
    rows = store.by_benchmark("compression")
    pivot = store.matrix("compression", formats, datasets, "file_size_bytes")
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from benchmarks.comprehensive.config import (
    PROJECT_ROOT,
    RAW_RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source references
# ---------------------------------------------------------------------------


class SourceKind(str, Enum):
    """Origin of a data value in the report."""

    raw_json = "raw_json"
    """Derived from a raw comprehensive-harness JSON file."""

    summary_json = "summary_json"
    """Derived from a captured snapshot / summary JSON."""

    external_report = "external_report"
    """Imported from a standalone report outside ``results/raw/``."""

    manual = "manual"
    """Manually curated value — requires a reason string."""

    unavailable = "unavailable"
    """Deliberately absent, with a machine-readable reason."""


@dataclass(frozen=True)
class SourceRef:
    """Provenance marker for a single data value or table.

    Every table, figure, and headline claim in the report should carry a
    ``SourceRef`` so that report-lint can verify freshness and
    traceability.
    """

    kind: SourceKind
    path: str | None = None
    """Filesystem path (relative to PROJECT_ROOT) or URI of the source."""

    reason: str | None = None
    """Short explanation (required for ``manual`` and ``unavailable``)."""

    def __repr__(self) -> str:
        parts = [f"kind={self.kind.value!r}"]
        if self.path:
            parts.append(f"path={self.path!r}")
        if self.reason:
            parts.append(f"reason={self.reason!r}")
        return f"SourceRef({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Typed missing reasons
# ---------------------------------------------------------------------------


class MissingReason(str, Enum):
    """Machine-readable reason for an absent benchmark value.

    Replaces display-only em-dashes (``—``) in table cells with typed
    reasons that report-lint can validate.
    """

    not_applicable = "not_applicable"
    not_supported = "not_supported"
    not_preserved_by_converter = "not_preserved_by_converter"
    skipped_too_expensive = "skipped_too_expensive"
    missing_fixture = "missing_fixture"
    missing_dependency = "missing_dependency"
    benchmark_failed = "benchmark_failed"
    not_run = "not_run"


# ---------------------------------------------------------------------------
# Scenario metadata
# ---------------------------------------------------------------------------


@dataclass
class ScenarioMeta:
    """Optional per-result scenario metadata.

    Fields are populated when the raw JSON carries them; otherwise they
    remain ``None`` and callers can filter or display accordingly.
    """

    name: str | None = None
    mode: str | None = None
    """E.g. ``full_pipeline``, ``write_only``, ``backed``, ``lazy``."""
    thread_count: int | None = None
    cache_state: str | None = None
    """E.g. ``warm``, ``cold``."""
    device: str | None = None
    """E.g. ``cpu``, ``gpu``, ``mixed``."""
    batch_size: int | None = None
    n_hvgs: int | None = None
    selectivity: float | None = None
    storage_backend: str | None = None
    """E.g. ``local``, ``gcs``."""

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ScenarioMeta:
        """Extract scenario metadata from a raw JSON result dict."""
        # Check top-level and metadata for scenario fields.
        scenario = raw.get("scenario", {}) or {}
        meta = raw.get("metadata", {}) or {}
        return cls(
            name=scenario.get("name") or meta.get("scenario_name"),
            mode=scenario.get("mode") or meta.get("mode"),
            thread_count=_int_or_none(
                scenario.get("thread_count") or meta.get("thread_count")
            ),
            cache_state=scenario.get("cache_state") or meta.get("cache_state"),
            device=scenario.get("device") or meta.get("device"),
            batch_size=_int_or_none(
                scenario.get("batch_size") or meta.get("batch_size")
            ),
            n_hvgs=_int_or_none(
                scenario.get("n_hvgs") or meta.get("n_hvgs")
            ),
            selectivity=_float_or_none(
                scenario.get("selectivity") or meta.get("selectivity")
            ),
            storage_backend=(
                scenario.get("storage_backend")
                or meta.get("storage_backend")
            ),
        )


# ---------------------------------------------------------------------------
# Comparison metadata (for equivalency/parity benchmarks)
# ---------------------------------------------------------------------------


@dataclass
class ComparisonMeta:
    """Metadata for an equivalency or parity comparison.

    For benchmarks like ``correctness``, ``cell_eval_parity_perf``, and
    Harmony validation, this records both sides of the comparison and
    the observed result.
    """

    subject_impl: str | None = None
    baseline_impl: str | None = None
    baseline_version: str | None = None
    metric: str | None = None
    value: float | None = None
    threshold: float | None = None
    status: str | None = None
    """E.g. ``pass``, ``fail``, ``skipped``."""
    notes: str | None = None

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> ComparisonMeta | None:
        """Extract comparison metadata, returning ``None`` if absent."""
        comp = raw.get("comparison", {}) or {}
        if not comp:
            # Try harness-level fields for correctness results.
            if raw.get("harness"):
                return cls(
                    subject_impl=raw.get("format"),
                    baseline_impl=raw.get("harness"),
                    status="pass" if raw.get("overall_passed") else "fail",
                )
            return None
        return cls(
            subject_impl=comp.get("subject", {}).get("impl")
            or raw.get("format"),
            baseline_impl=comp.get("baseline", {}).get("impl"),
            baseline_version=comp.get("baseline", {}).get("version"),
            metric=comp.get("metric"),
            value=_float_or_none(comp.get("value")),
            threshold=_float_or_none(comp.get("threshold")),
            status=comp.get("status"),
            notes=comp.get("notes"),
        )


# ---------------------------------------------------------------------------
# Normalized result row
# ---------------------------------------------------------------------------


@dataclass
class ResultRow:
    """A single normalized benchmark result row.

    Every raw JSON file produces one ``ResultRow`` in the store. Complex
    benchmarks with sub-results (e.g. per-op medians in ``fragment_ops``)
    keep the structured ``metadata`` dict for table builders to unpack
    domain-specifically.
    """

    benchmark: str
    format: str
    dataset: str
    timestamp: str

    # Primary scalar metrics (populated when available).
    median_wall_s: float | None = None
    file_size_bytes: int | None = None

    # The full raw dict, for table builders that need deep access.
    raw: dict[str, Any] = field(default_factory=dict, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    runs: list[dict[str, Any]] = field(default_factory=list, repr=False)

    # Typed provenance and status.
    source: SourceRef = field(
        default_factory=lambda: SourceRef(kind=SourceKind.raw_json)
    )
    scenario: ScenarioMeta = field(default_factory=ScenarioMeta)
    comparison: ComparisonMeta | None = None

    # Missing reason — set when this row represents a known gap.
    missing_reason: MissingReason | None = None

    def get_nested(self, dot_path: str) -> Any:
        """Navigate a dot-separated key path into ``raw``.

        Example: ``row.get_nested("metadata.per_op_medians.append")``
        """
        val: Any = self.raw
        for part in dot_path.split("."):
            if isinstance(val, dict):
                val = val.get(part)
            else:
                return None
        return val


# ---------------------------------------------------------------------------
# Result store
# ---------------------------------------------------------------------------


class ResultStore:
    """Central result store — load once, query many.

    Attributes
    ----------
    rows : list[ResultRow]
        All loaded results, normalized.
    external_rows : list[ResultRow]
        Results loaded from external (non-comprehensive) sources.
    """

    def __init__(self, rows: list[ResultRow] | None = None) -> None:
        self._rows: list[ResultRow] = rows or []

    # -- construction -------------------------------------------------------

    @classmethod
    def from_defaults(cls) -> ResultStore:
        """Load results from the standard raw directory + external sources."""
        store = cls()
        store.load_raw_dir(RAW_RESULTS_DIR)
        store.load_harmony_dir()
        return store

    def load_raw_dir(self, raw_dir: Path) -> int:
        """Load all ``*.json`` from *raw_dir*. Returns count loaded."""
        if not raw_dir.is_dir():
            logger.warning("Raw results directory does not exist: %s", raw_dir)
            return 0
        count = 0
        for path in sorted(raw_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", path.name, exc)
                continue
            rel = _relpath(path)
            row = _normalize_raw(data, source_path=rel)
            self._rows.append(row)
            count += 1
        logger.info("Loaded %d raw JSON results from %s", count, raw_dir)
        return count

    def load_harmony_dir(self) -> int:
        """Load Harmony / LISI results from the external runs directory."""
        harmony_dir = PROJECT_ROOT / "benchmarks" / "results" / "harmony" / "runs"
        if not harmony_dir.is_dir():
            return 0
        count = 0
        for path in sorted(harmony_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            rel = _relpath(path)
            row = _normalize_external(data, source_path=rel)
            self._rows.append(row)
            count += 1
        logger.info("Loaded %d harmony/LISI results", count)
        return count

    def add_manual(
        self,
        benchmark: str,
        format_key: str,
        dataset: str,
        *,
        reason: str,
        median_wall_s: float | None = None,
        file_size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResultRow:
        """Register a manually curated result with a required reason."""
        row = ResultRow(
            benchmark=benchmark,
            format=format_key,
            dataset=dataset,
            timestamp="",
            median_wall_s=median_wall_s,
            file_size_bytes=file_size_bytes,
            metadata=metadata or {},
            source=SourceRef(kind=SourceKind.manual, reason=reason),
        )
        self._rows.append(row)
        return row

    # -- query helpers ------------------------------------------------------

    @property
    def rows(self) -> list[ResultRow]:
        return self._rows

    def by_benchmark(self, name: str) -> list[ResultRow]:
        """All rows for a benchmark type (e.g. ``"compression"``)."""
        return [r for r in self._rows if r.benchmark == name]

    def by_domain(self, domain: str) -> list[ResultRow]:
        """All rows whose benchmark belongs to a domain grouping."""
        mapping = _BENCHMARK_DOMAIN_MAP
        return [r for r in self._rows if mapping.get(r.benchmark) == domain]

    def by_format(self, format_key: str) -> list[ResultRow]:
        """All rows for a specific format."""
        return [r for r in self._rows if r.format == format_key]

    def by_dataset(self, dataset: str) -> list[ResultRow]:
        """All rows for a specific dataset."""
        return [r for r in self._rows if r.dataset == dataset]

    def latest_for(
        self, benchmark: str, format_key: str, dataset: str
    ) -> ResultRow | None:
        """Most recent result for a (benchmark, format, dataset) triple."""
        matches = [
            r
            for r in self._rows
            if r.benchmark == benchmark
            and r.format == format_key
            and r.dataset == dataset
        ]
        if not matches:
            return None
        return max(matches, key=lambda r: r.timestamp)

    def matrix(
        self,
        benchmark: str,
        formats: list[str],
        datasets: list[str],
        metric: str,
    ) -> dict[str, dict[str, float | None]]:
        """Pivot into ``format -> dataset -> value`` for a scalar metric.

        *metric* is a dot-path navigated via ``ResultRow.get_nested`` for
        structured metadata, or a direct attribute name for top-level
        fields (``median_wall_s``, ``file_size_bytes``).
        """
        bench_rows = self.by_benchmark(benchmark)
        pivot: dict[str, dict[str, float | None]] = {}
        for row in bench_rows:
            if row.format not in formats or row.dataset not in datasets:
                continue
            if metric in ("median_wall_s", "file_size_bytes"):
                val = getattr(row, metric, None)
            else:
                val = row.get_nested(metric)
            pivot.setdefault(row.format, {})[row.dataset] = val
        return pivot

    def coverage(
        self,
        benchmarks: list[str],
        formats: list[str],
        datasets: list[str],
    ) -> dict[str, dict[str, dict[str, bool]]]:
        """Coverage matrix: benchmark -> format -> dataset -> present."""
        cov: dict[str, dict[str, dict[str, bool]]] = {}
        for bench in benchmarks:
            cov[bench] = {}
            for fmt in formats:
                cov[bench][fmt] = {}
                for ds in datasets:
                    cov[bench][fmt][ds] = any(
                        r.benchmark == bench
                        and r.format == fmt
                        and r.dataset == ds
                        for r in self._rows
                    )
        return cov

    def external_sources(self) -> list[ResultRow]:
        """All rows loaded from external (non-raw) sources."""
        return [
            r
            for r in self._rows
            if r.source.kind
            in (SourceKind.external_report, SourceKind.manual)
        ]

    # -- raw-dict compatibility layer ---------------------------------------

    def load_all_results_compat(
        self,
        benchmark: str | None = None,
        format_key: str | None = None,
        dataset: str | None = None,
    ) -> list[dict[str, Any]]:
        """Drop-in replacement for ``results.load_all_results()``.

        Returns the raw dicts so existing table builders work unchanged
        during migration.
        """
        out: list[dict[str, Any]] = []
        for row in self._rows:
            if benchmark and row.benchmark != benchmark:
                continue
            if format_key and row.format != format_key:
                continue
            if dataset and row.dataset != dataset:
                continue
            out.append(row.raw)
        return sorted(out, key=lambda d: d.get("timestamp", ""))


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_default_store: ResultStore | None = None


def get_store() -> ResultStore:
    """Return (and cache) the default ``ResultStore``."""
    global _default_store
    if _default_store is None:
        _default_store = ResultStore.from_defaults()
    return _default_store


def set_store(store: ResultStore) -> None:
    """Install *store* as the module-level default singleton.

    Prefer this over reaching into ``_default_store`` directly.
    """
    global _default_store
    _default_store = store


def reset_store() -> None:
    """Reset the cached store (useful in tests)."""
    global _default_store
    _default_store = None


# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------

_BENCHMARK_DOMAIN_MAP: dict[str, str] = {
    "compression": "storage",
    "read_full": "local_io",
    "read_selective": "local_io",
    "write": "local_io",
    "parallel_scaling": "local_io",
    "parallel_write_scaling": "local_io",
    "memory": "local_io",
    "fragment_ops": "operations",
    "cloud_filtered": "cloud",
    "cloud_read": "cloud",
    "cloud_reader_vs_pull": "cloud",
    "cost_model": "cloud",
    "cloud_metadata": "cloud",
    "cloud_push": "cloud",
    "cloud_pull": "cloud",
    "ml_loader": "loader",
    "multimodal_compression": "specialized",
    "multimodal_training": "specialized",
    "bench_csc_dispatch": "accelerator",
    "cell_eval_parity_perf": "specialized",
    "correctness": "correctness",
    "roundtrip": "correctness",
    "harmony_integrate": "specialized",
    "lisi": "specialized",
    # Accelerator benchmarks (accel_*) produce raw JSONs.
    "accel_pca": "accelerator",
    "accel_knn": "accelerator",
    "accel_umap": "accelerator",
    "accel_leiden": "accelerator",
    "accel_preprocess": "accelerator",
    "accel_hvg": "accelerator",
    "accel_pipeline": "accelerator",
    "accel_to_gpu_anndata": "accelerator",
    "accel_format_pipeline": "accelerator",
    # Differential expression accelerators — per-cell (Wilcoxon / pdex_ref) and
    # pseudobulk NB-GLM. Without these the DE benchmarks never bind to the
    # Accelerators chapter and never appear in the report.
    "accel_de": "accelerator",
    "accel_de_nb_glm": "accelerator",
    # Community analytical workflows. Unmapped is
    # not an error — `by_domain` does `mapping.get(...)`, so the rows simply
    # never reach a domain-grouped report section. Several existing benchmarks
    # are already in that state; don't copy the omission.
    "accel_qc_filter": "accelerator",
    "accel_score_genes": "accelerator",
    "pipeline_ooc_constrained": "specialized",
    "multimodal_atlas_streaming": "specialized",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _relpath(path: Path) -> str:
    """Return a PROJECT_ROOT-relative string for source tracking."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _int_or_none(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _try_missing_reason(value: Any) -> MissingReason | None:
    """Convert *value* to a ``MissingReason`` if it matches, else ``None``.

    Uses value-based lookup (try/except) instead of the fragile
    ``__members__`` name-based check.
    """
    if not value:
        return None
    try:
        return MissingReason(value)
    except ValueError:
        return None


def _normalize_raw(data: dict[str, Any], source_path: str) -> ResultRow:
    """Normalize a raw comprehensive-harness JSON into a ``ResultRow``."""
    missing = data.get("missing_reason")
    return ResultRow(
        benchmark=data.get("benchmark", ""),
        format=data.get("format", ""),
        dataset=data.get("dataset", ""),
        timestamp=data.get("timestamp", ""),
        median_wall_s=_float_or_none(data.get("median_wall_s")),
        file_size_bytes=_int_or_none(data.get("file_size_bytes")),
        raw=data,
        metadata=data.get("metadata", {}) or {},
        runs=data.get("runs", []) or [],
        source=SourceRef(kind=SourceKind.raw_json, path=source_path),
        scenario=ScenarioMeta.from_raw(data),
        comparison=ComparisonMeta.from_raw(data),
        missing_reason=_try_missing_reason(missing),
    )


def _normalize_external(data: dict[str, Any], source_path: str) -> ResultRow:
    """Normalize an external-source JSON (harmony/LISI)."""
    # External results have a different shape — derive what we can.
    run = data.get("run", {}) or {}
    return ResultRow(
        benchmark=data.get("benchmark", ""),
        format=data.get("impl", data.get("format", "")),
        dataset=data.get("dataset", ""),
        timestamp=data.get("timestamp", ""),
        median_wall_s=_float_or_none(run.get("wall_s")),
        raw=data,
        metadata=data.get("metadata", {}) or {},
        runs=[run] if run else [],
        source=SourceRef(
            kind=SourceKind.external_report, path=source_path
        ),
        scenario=ScenarioMeta.from_raw(data),
        comparison=ComparisonMeta.from_raw(data),
    )
