"""
Generate markdown summary tables from raw benchmark JSON results.

Reads from benchmarks/comprehensive/results/raw/ and produces formatted
markdown tables for each benchmark dimension.

All table functions consume the shared ``ResultStore`` singleton
so that raw JSON is loaded once per report-generation run. The legacy
``load_all_results`` name is preserved as a thin wrapper around the
store's compatibility layer.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any

from benchmarks.comprehensive.config import DATASETS, DatasetConfig
from benchmarks.comprehensive.config import DATASETS, DatasetConfig
from benchmarks.comprehensive.reporting.result_store import get_store, SourceRef, SourceKind
from benchmarks.comprehensive.reporting.report_model import TableBlock, TextBlock, Block

logger = logging.getLogger(__name__)


def load_all_results(
    benchmark: str | None = None,
    format_key: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Legacy wrapper — delegates to the ``ResultStore`` singleton.

    Preserves the exact return type (``list[dict]``) so that every
    existing table function works unchanged during the
    migration window.
    """
    return get_store().load_all_results_compat(
        benchmark=benchmark, format_key=format_key, dataset=dataset,
    )

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Datasets to include in main tables (exclude lognorm variants and census_10m
# which has limited results)
MAIN_DATASETS = [
    "pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k",
    "census_500k", "census_1m", "census_5m",
]

# Short display names for table headers
SHORT_NAMES = {
    "pbmc3k": "pbmc3k",
    "pbmc10k": "pbmc10k",
    "smartseq2": "smartseq2",
    "tabula_sapiens_100k": "tabula_100k",
    "census_500k": "census_500k",
    "census_1m": "census_1m",
    "census_5m": "census_5m",
    "census_10m": "census_10m",
}

# Preferred format display order
FORMAT_ORDER = [
    "scx_auto", "scx_scx1", "scx_zstd", "scx_lz4", "scx_pcodec", "scx_none",
    "zarr_zstd", "zarr_lz4", "tiledb_soma",
    "h5ad_gzip", "h5ad_lzf", "h5ad_none",
]

FORMAT_DISPLAY = {
    "scx_auto": "SCX (auto)",
    "scx_scx1": "SCX (scx1)",
    "scx_zstd": "SCX (zstd)",
    "scx_lz4": "SCX (lz4)",
    "scx_pcodec": "SCX (pcodec)",
    "scx_none": "SCX (none)",
    "zarr_zstd": "Zarr (zstd)",
    "zarr_lz4": "Zarr (blosc-lz4)",
    "tiledb_soma": "TileDB-SOMA",
    "h5ad_gzip": "h5ad (gzip)",
    "h5ad_lzf": "h5ad (lzf)",
    "h5ad_none": "h5ad (none)",
    "bpcells": "BPCells",
    "parquet_zstd": "Parquet (zstd)",
}


def _fmt_size(nbytes: float | None) -> Block:
    """Format byte count as human-readable size."""
    if nbytes is None:
        return "—"
    if nbytes < 1024:
        return f"{nbytes:.0f} B"
    elif nbytes < 1024 ** 2:
        return f"{nbytes / 1024:.1f} KB"
    elif nbytes < 1024 ** 3:
        return f"{nbytes / 1024 ** 2:.1f} MB"
    else:
        return f"{nbytes / 1024 ** 3:.2f} GB"


def _fmt_time(seconds: float | None) -> Block:
    """Format seconds as human-readable time."""
    if seconds is None:
        return "—"
    if seconds < 0.01:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 1:
        return f"{seconds:.3f}s"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def _fmt_mem(mb: float | None) -> Block:
    """Format memory in MB as human-readable."""
    if mb is None:
        return "—"
    if mb < 1024:
        return f"{mb:,.0f} MB"
    else:
        return f"{mb / 1024:.1f} GB"


def _fmt_num(val: float | None, decimals: int = 1) -> Block:
    """Format a number."""
    if val is None:
        return "—"
    if val >= 1000:
        return f"{val:,.0f}"
    return f"{val:.{decimals}f}"


def _build_pivot(
    results: list[dict[str, Any]],
    value_key: str,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Pivot results into format -> dataset -> value.

    Parameters
    ----------
    results : list of result dicts
    value_key : dot-separated key path (e.g. "median_wall_s", "file_size_bytes",
                "metadata.median_delta_rss_mb_read_full")
    datasets : restrict to these dataset names

    Returns
    -------
    dict mapping format_key -> {dataset_name -> value}
    """
    if datasets is None:
        datasets = MAIN_DATASETS

    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue

        # Navigate dot-separated key
        val: Any = r
        for part in value_key.split("."):
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break

        if fmt not in pivot:
            pivot[fmt] = {}
        pivot[fmt][ds] = val

    return pivot


def _pivot_to_tableblock(
    pivot: dict[str, dict[str, float | None]],
    datasets: list[str],
    formatter: callable,
    bold_best: str = "min",
    title: str = "Format",
    caption: str | None = None,
    wide: bool = False,
) -> TableBlock:
    """Construct a ``TableBlock`` directly from a pivot dict.

    This replaces the legacy ``_pivot_to_table`` → ``_lines_to_block``
    round-trip with direct ``TableBlock`` construction, preserving the
    bold-best logic as markdown ``**…**`` in cell strings.

    Parameters
    ----------
    pivot : format -> dataset -> value
    datasets : column order
    formatter : function to format values
    bold_best : ``"min"`` to bold the minimum per column, ``"max"`` for
        maximum, ``None`` for no bolding
    title : first column header (default ``"Format"``)
    caption : optional table caption
    wide : if ``True``, hint renderers to use wide-table layout
    """
    headers = [title] + [SHORT_NAMES.get(d, d) for d in datasets]

    # Determine best per column
    best_per_col: dict[str, float | None] = {}
    if bold_best:
        for ds in datasets:
            vals = []
            for fmt in pivot:
                v = pivot[fmt].get(ds)
                if v is not None:
                    vals.append(v)
            if vals:
                best_per_col[ds] = min(vals) if bold_best == "min" else max(vals)

    # Sort formats by FORMAT_ORDER
    sorted_fmts = sorted(
        pivot.keys(),
        key=lambda f: FORMAT_ORDER.index(f) if f in FORMAT_ORDER else 999,
    )

    rows: list[list[str]] = []
    for fmt in sorted_fmts:
        row = pivot[fmt]
        display = FORMAT_DISPLAY.get(fmt, fmt)
        cells = [display]
        for ds in datasets:
            v = row.get(ds)
            cell = formatter(v)
            if (
                bold_best
                and v is not None
                and best_per_col.get(ds) is not None
                and abs(v - best_per_col[ds]) < 1e-10 * max(abs(v), 1)
            ):
                cell = f"**{cell}**"
            cells.append(cell)
        rows.append(cells)

    return TableBlock(headers=headers, rows=rows, caption=caption, wide=wide)


# ---------------------------------------------------------------------------
# Headline derivation helpers
# ---------------------------------------------------------------------------
#
# These functions compute concrete headline metrics from the result store so
# that executive summaries, chapter summaries, and commentary blocks use
# data-derived values instead of hardcoded prose.  If data is unavailable,
# they return ``None`` and callers fall back to a qualified placeholder.


def derive_compression_headlines(
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Derive headline compression metrics from raw data.

    Returns a dict with keys:
      - ``best_format``: format with smallest size on largest dataset
      - ``best_size``: formatted size string
      - ``best_ratio_dataset``: dataset where best ratio occurs
      - ``best_ratio``: float ratio value
      - ``scx_vs_zarr_census_1m``: (scx_size, zarr_size) pair if available
      - ``scx_vs_zarr_census_5m``: (scx_ratio, zarr_ratio) pair if available
    """
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="compression")
    pivot = _build_pivot(results, "file_size_bytes", datasets)

    # Find h5ad_none baseline for ratios
    baseline: dict[str, float] = {}
    for r in results:
        if r.get("format") == "h5ad_none" and r.get("dataset") in datasets:
            baseline[r["dataset"]] = r.get("file_size_bytes", 0)

    out: dict[str, Any] = {}

    # Best format on largest available dataset
    for target_ds in reversed(datasets):
        if any(target_ds in pivot.get(f, {}) for f in pivot):
            col_vals = [
                (f, pivot[f][target_ds])
                for f in pivot if pivot[f].get(target_ds) is not None
            ]
            if col_vals:
                best_f, best_v = min(col_vals, key=lambda x: x[1])
                out["best_format"] = FORMAT_DISPLAY.get(best_f, best_f)
                out["best_size"] = _fmt_size(best_v)
                out["best_dataset"] = SHORT_NAMES.get(target_ds, target_ds)
            break

    # Best compression ratio per dataset
    best_ratio = 0.0
    best_ratio_ds = ""
    for ds in datasets:
        base = baseline.get(ds)
        if not base:
            continue
        for f in pivot:
            v = pivot[f].get(ds)
            if v and v > 0:
                ratio = base / v
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_ratio_ds = ds
    if best_ratio > 0:
        out["best_ratio"] = best_ratio
        out["best_ratio_dataset"] = SHORT_NAMES.get(best_ratio_ds, best_ratio_ds)

    # SCX auto vs Zarr zstd on specific datasets
    for ds_key in ["census_1m", "census_5m"]:
        scx_v = pivot.get("scx_auto", {}).get(ds_key) or pivot.get("scx_zstd", {}).get(ds_key)
        zarr_v = pivot.get("zarr_zstd", {}).get(ds_key)
        if scx_v is not None and zarr_v is not None:
            out[f"scx_vs_zarr_{ds_key}"] = (_fmt_size(scx_v), _fmt_size(zarr_v))
        # Also ratios
        base = baseline.get(ds_key)
        if base and scx_v and zarr_v and scx_v > 0 and zarr_v > 0:
            out[f"scx_ratio_{ds_key}"] = base / scx_v
            out[f"zarr_ratio_{ds_key}"] = base / zarr_v

    return out


def derive_read_headlines(
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Derive headline read performance metrics from raw data.

    Returns a dict with keys:
      - ``fastest_format``: best format on largest dataset
      - ``scx_vs_zarr_census_1m``: speedup of SCX over Zarr lz4
      - ``scx_vs_zarr_census_5m``: speedup of SCX over Zarr lz4
    """
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="read_full")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    out: dict[str, Any] = {}

    # Find fastest on largest dataset
    for target_ds in reversed(datasets):
        col_vals = [
            (f, pivot[f][target_ds])
            for f in pivot if pivot[f].get(target_ds) is not None
        ]
        if col_vals:
            best_f, _ = min(col_vals, key=lambda x: x[1])
            out["fastest_format"] = FORMAT_DISPLAY.get(best_f, best_f)
            break

    # SCX vs Zarr lz4 speedups
    for ds_key in ["census_1m", "census_5m"]:
        scx_fmts = ["scx_auto", "scx_lz4", "scx_zstd"]
        scx_v = None
        for sf in scx_fmts:
            v = pivot.get(sf, {}).get(ds_key)
            if v is not None:
                if scx_v is None or v < scx_v:
                    scx_v = v
        zarr_v = pivot.get("zarr_lz4", {}).get(ds_key)
        if scx_v is not None and zarr_v is not None and scx_v > 0:
            out[f"scx_vs_zarr_{ds_key}"] = zarr_v / scx_v

    return out


def derive_selective_read_headlines(
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Derive headline selective read metrics from raw data.

    Returns a dict with keys:
      - ``scx_vs_zarr_census_1m``: speedup of SCX over Zarr on census_1m
      - ``scx_vs_zarr_census_5m``: speedup of SCX over Zarr on census_5m
    """
    if datasets is None:
        datasets = [d for d in MAIN_DATASETS if d not in ("pbmc3k", "pbmc10k")]
    results = load_all_results(benchmark="read_selective")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    out: dict[str, Any] = {}

    for ds_key in ["census_1m", "census_5m"]:
        scx_fmts = ["scx_auto", "scx_lz4", "scx_zstd"]
        scx_v = None
        for sf in scx_fmts:
            v = pivot.get(sf, {}).get(ds_key)
            if v is not None:
                if scx_v is None or v < scx_v:
                    scx_v = v
        zarr_v = pivot.get("zarr_zstd", {}).get(ds_key) or pivot.get("zarr_lz4", {}).get(ds_key)
        if scx_v is not None and zarr_v is not None and scx_v > 0:
            out[f"scx_vs_zarr_{ds_key}"] = zarr_v / scx_v

    return out


def derive_ml_loader_headlines(
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Derive headline ML loader metrics from raw data.

    Returns a dict with keys:
      - ``scx_best_bps``: best SCX batches/sec value
      - ``scx_best_dataset``: dataset for best value
      - ``scx_vs_soma``: speedup over TileDB-SOMA-ML
    """
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="ml_loader")
    out: dict[str, Any] = {}

    # Build per-format per-dataset medians
    bps_pivot: dict[str, dict[str, float]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        hvg_vals: list[float] = []
        for run in r.get("runs", []):
            if run.get("extra", {}).get("scenario") == "hvg_norm":
                bps = run["extra"].get("batches_per_sec")
                if bps is not None:
                    hvg_vals.append(bps)
        if not hvg_vals:
            for run in r.get("runs", []):
                if run.get("extra", {}).get("scenario") == "raw":
                    bps = run["extra"].get("batches_per_sec")
                    if bps is not None:
                        hvg_vals.append(bps)
        if hvg_vals:
            bps_pivot.setdefault(fmt, {})[ds] = statistics.median(hvg_vals)

    # Best SCX batches/sec
    scx_best = 0.0
    scx_best_ds = ""
    for fmt in bps_pivot:
        if not fmt.startswith("scx_"):
            continue
        for ds, val in bps_pivot[fmt].items():
            if val > scx_best:
                scx_best = val
                scx_best_ds = ds
    if scx_best > 0:
        out["scx_best_bps"] = scx_best
        out["scx_best_dataset"] = SHORT_NAMES.get(scx_best_ds, scx_best_ds)

    # SCX vs TileDB-SOMA speedup (same dataset)
    soma_bps = bps_pivot.get("tiledb_soma", {})
    for fmt in bps_pivot:
        if not fmt.startswith("scx_"):
            continue
        for ds in bps_pivot[fmt]:
            soma_v = soma_bps.get(ds)
            if soma_v is not None and soma_v > 0:
                speedup = bps_pivot[fmt][ds] / soma_v
                if "scx_vs_soma" not in out or speedup > out["scx_vs_soma"]:
                    out["scx_vs_soma"] = speedup

    return out


def derive_memory_headlines(
    datasets: list[str] | None = None,
) -> dict[str, Any]:
    """Derive headline memory efficiency metrics from raw data.

    Returns a dict with keys:
      - ``scx_vs_zarr_ratio``: ratio of Zarr RSS / SCX RSS on best dataset
    """
    if datasets is None:
        datasets = [d for d in MAIN_DATASETS if d not in ("pbmc3k", "pbmc10k", "smartseq2")]
    results = load_all_results(benchmark="memory")
    out: dict[str, Any] = {}

    rss_pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        rss = r.get("metadata", {}).get("median_delta_rss_mb_read_full")
        if rss is not None:
            rss_pivot.setdefault(fmt, {})[ds] = rss

    # SCX vs Zarr
    for ds in reversed(datasets):
        scx_v = rss_pivot.get("scx_auto", {}).get(ds) or rss_pivot.get("scx_zstd", {}).get(ds)
        zarr_v = rss_pivot.get("zarr_zstd", {}).get(ds) or rss_pivot.get("zarr_lz4", {}).get(ds)
        if scx_v is not None and zarr_v is not None and scx_v > 0:
            out["scx_vs_zarr_ratio"] = zarr_v / scx_v
            break

    return out


def collect_manual_sources(store: Any = None) -> list[dict[str, str]]:
    """Collect all manual and external source references from tables.

    Returns a list of dicts with keys ``source_kind``, ``path``,
    ``reason``, and ``table``.
    """
    if store is None:
        store = get_store()
    sources: list[dict[str, str]] = []

    # Check tables that carry SourceRef
    table_funcs: dict[str, Any] = {
        "harmony_validation": harmony_validation_table,
        "harmony_lisi_correctness": harmony_lisi_correctness_summary_table,
    }
    for name, func in table_funcs.items():
        try:
            block = func()
            if isinstance(block, TableBlock) and block.source:
                sources.append({
                    "table": name,
                    "source_kind": block.source.kind.value,
                    "path": block.source.path or "",
                    "reason": block.source.reason or "",
                })
        except Exception:
            pass

    # External-source rows from the store
    for row in store.external_sources():
        sources.append({
            "table": f"external:{row.benchmark}",
            "source_kind": row.source.kind.value,
            "path": row.source.path or "",
            "reason": row.source.reason or "",
        })

    return sources


# ---------------------------------------------------------------------------
# Public table generators
# ---------------------------------------------------------------------------

def compression_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate compression matrix: Format x Dataset showing file size."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="compression")
    pivot = _build_pivot(results, "file_size_bytes", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_size, bold_best="min",
        caption="File sizes by format and dataset",
    )


def compression_ratio_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate compression ratio matrix: Format x Dataset showing ratio vs h5ad_none."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="compression")

    # Get h5ad_none sizes as baseline
    baseline: dict[str, float] = {}
    for r in results:
        if r.get("format") == "h5ad_none" and r.get("dataset") in datasets:
            baseline[r["dataset"]] = r.get("file_size_bytes", 0)

    # Build ratio pivot
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets or fmt == "h5ad_none":
            continue
        size = r.get("file_size_bytes")
        base = baseline.get(ds)
        if size and base and size > 0:
            ratio = base / size
        else:
            ratio = None
        pivot.setdefault(fmt, {})[ds] = ratio

    def fmt_ratio(v):
        if v is None:
            return "—"
        return f"{v:.2f}x"

    return _pivot_to_tableblock(
        pivot, datasets, fmt_ratio, bold_best="max",
        caption="Compression ratio vs uncompressed h5ad by format and dataset",
    )


def read_speed_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate read speed matrix: Format x Dataset showing median read time."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="read_full")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min",
        caption="Median full-read wall time by format and dataset",
    )


def read_selective_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate selective read table: Format x Dataset showing column projection time."""
    if datasets is None:
        datasets = [d for d in MAIN_DATASETS if d != "pbmc3k" and d != "pbmc10k"]
    results = load_all_results(benchmark="read_selective")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min",
        caption="Median selective-read (column projection) wall time by format and dataset",
    )


def write_speed_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate write speed matrix: Format x Dataset showing conversion time."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="write")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min",
        caption="Median write/conversion wall time by format and dataset",
    )


def write_conversion_table(datasets: list[str] | None = None) -> TableBlock:
    """Write/conversion pipeline wall time — reads h5ad then writes target format.

    This is identical to ``write_speed_table`` but explicitly labeled as
    the *full conversion pipeline* (h5ad read → encode → write) to
    distinguish it from a write-only benchmark that starts from in-memory
    data. The I/O chapter splits so readers know which timing
    includes h5ad read overhead.
    """
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="write")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min",
        caption="Median conversion-pipeline wall time (h5ad read + encode + write)",
    )


def write_only_table(datasets: list[str] | None = None) -> TableBlock | TextBlock:
    """Write-only wall time — encoding from in-memory AnnData (1 thread).

    Extracts the single-threaded (``"1"``) timing from the
    ``parallel_write_scaling`` benchmark's ``write_only`` mode, giving an
    apples-to-apples comparison of codec/write cost without h5ad read
    overhead.
    """
    if datasets is None:
        datasets = ["census_500k", "census_1m"]
    results = load_all_results(benchmark="parallel_write_scaling")
    if not results:
        return TextBlock("*No parallel_write_scaling results with write_only mode available yet.*")

    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        scaling = (r.get("metadata") or {}).get("scaling_wall_s", {}).get("write_only", {})
        t1 = scaling.get("1")
        if t1 is not None:
            pivot.setdefault(fmt, {})[ds] = t1

    if not pivot:
        return TextBlock("*No write_only timing data found in parallel_write_scaling results.*")

    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min",
        caption="Median write-only wall time (in-memory AnnData, 1 thread)",
    )


def memory_by_mode_tables(datasets: list[str] | None = None) -> list[Block]:
    """Memory tables grouped by measurement mode (full read vs subset).

    Returns separate ``TableBlock``s for each memory-measurement mode
    found in the raw results (``read_full``, ``read_subset_1k``, etc.)
    so that narrative takeaways can reference a specific mode without
    mixing definitions.
    """
    if datasets is None:
        datasets = [d for d in MAIN_DATASETS if d not in ("pbmc3k", "pbmc10k", "smartseq2")]
    results = load_all_results(benchmark="memory")

    # Collect per-mode pivots: mode -> format -> dataset -> delta_rss_mb
    mode_pivots: dict[str, dict[str, dict[str, float | None]]] = {}

    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        md = r.get("metadata", {}) or {}

        # Full-read RSS
        full_rss = md.get("median_delta_rss_mb_read_full")
        if full_rss is not None:
            mode_pivots.setdefault("Full read (materialized)", {}).setdefault(fmt, {})[ds] = full_rss

        # Subset/query RSS
        subset_rss = md.get("median_delta_rss_mb_read_subset_1k")
        if subset_rss is not None:
            mode_pivots.setdefault("Subset read (1K cells)", {}).setdefault(fmt, {})[ds] = subset_rss

        # Fallback: parse per-run operations
        if not full_rss and not subset_rss:
            for run in r.get("runs", []):
                op = (run.get("extra") or {}).get("operation", "read_full")
                delta = (run.get("extra") or {}).get("delta_rss_mb")
                if delta is None:
                    delta = run.get("peak_rss_mb")
                if delta is not None:
                    label = "Full read (materialized)" if op == "read_full" else f"{op}"
                    mode_pivots.setdefault(label, {}).setdefault(fmt, {})[ds] = delta

    blocks: list[Block] = []
    for mode_label in sorted(mode_pivots.keys()):
        pivot = mode_pivots[mode_label]
        blocks.append(_pivot_to_tableblock(
            pivot, datasets, _fmt_mem, bold_best="min",
            caption=f"Peak RSS (delta) — {mode_label}",
        ))
    return blocks or [TextBlock("*No per-mode memory results available.*")]


def memory_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate memory matrix: Format x Dataset showing peak RSS."""
    if datasets is None:
        datasets = [d for d in MAIN_DATASETS if d not in ("pbmc3k", "pbmc10k", "smartseq2")]
    results = load_all_results(benchmark="memory")

    # Use median peak_rss_mb from read_full operations
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        # Use the metadata summary if available, otherwise compute from runs
        rss = r.get("metadata", {}).get("median_delta_rss_mb_read_full")
        if rss is None:
            # Fallback: use peak RSS from read_full runs
            read_full_runs = [
                run for run in r.get("runs", [])
                if run.get("extra", {}).get("operation") == "read_full"
            ]
            if read_full_runs:
                rss = statistics.median(run["peak_rss_mb"] for run in read_full_runs)
        pivot.setdefault(fmt, {})[ds] = rss

    return _pivot_to_tableblock(
        pivot, datasets, _fmt_mem, bold_best="min",
        caption="Median peak RSS (delta) during full read by format and dataset",
    )


def parallel_scaling_table(datasets: list[str] | None = None) -> list[Block]:
    """Generate parallel scaling table: Format x Thread count showing wall time and speedup.

    Returns a list of blocks — one ``TextBlock`` header + ``TableBlock`` per dataset.
    """
    if datasets is None:
        datasets = ["census_500k", "census_1m", "census_5m"]
    results = load_all_results(benchmark="parallel_scaling")

    thread_cols = ["1 thread", "2 threads", "4 threads", "8 threads",
                   "16 threads", "32 threads", "Max speedup"]
    blocks: list[Block] = []
    for ds in datasets:
        ds_results = [r for r in results if r.get("dataset") == ds]
        if not ds_results:
            continue

        sorted_results = sorted(
            ds_results,
            key=lambda r: FORMAT_ORDER.index(r.get("format", ""))
            if r.get("format", "") in FORMAT_ORDER else 999,
        )

        rows: list[list[str]] = []
        for r in sorted_results:
            fmt = r.get("format", "")
            display = FORMAT_DISPLAY.get(fmt, fmt)
            scaling = r.get("metadata", {}).get("scaling_wall_s", {})
            speedup = r.get("metadata", {}).get("speedup", {})

            cells = [display]
            for t in ["1", "2", "4", "8", "16", "32"]:
                wall = scaling.get(t)
                spd = speedup.get(t)
                if wall is not None:
                    if spd is not None and t != "1":
                        cells.append(f"{_fmt_time(wall)} ({spd:.1f}x)")
                    else:
                        cells.append(_fmt_time(wall))
                else:
                    cells.append("—")

            max_spd = max(speedup.values()) if speedup else None
            cells.append(f"{max_spd:.1f}x" if max_spd else "—")
            rows.append(cells)

        blocks.append(TableBlock(
            headers=["Format"] + thread_cols, rows=rows,
            caption=f"Parallel read scaling — {SHORT_NAMES.get(ds, ds)}",
        ))

    return blocks or [TextBlock("*No parallel scaling results available.*")]


def scx_parallel_write_callout_table(
    datasets: list[str] | None = None,
) -> TableBlock | TextBlock:
    """Compact SCX-only 1T → 32T → speedup table for the Write Performance section.

    Pulled from the `parallel_write_scaling` results, `full` mode only (the
    h5ad-read + SCX-write pipeline that `write_speed_table()` measures
    single-threaded). Used to reframe §3 so readers don't walk away with
    the misleading impression that SCX writes are slow at every thread
    count — the single-threaded figure in `write_speed_table()` is the
    worst case for a shard-parallel codec.
    """
    if datasets is None:
        datasets = ["census_500k", "census_1m"]
    results = load_all_results(benchmark="parallel_write_scaling")
    if not results:
        return TextBlock("*No parallel write scaling results available yet.*")

    # Keep SCX rows only; other formats do not parallelise at all.
    scx_formats = [f for f in FORMAT_ORDER if f.startswith("scx_")]

    headers = ["Format"]
    for ds in datasets:
        headers += [
            f"{SHORT_NAMES.get(ds, ds)} @1T",
            f"{SHORT_NAMES.get(ds, ds)} @32T",
            f"{SHORT_NAMES.get(ds, ds)} Δ",
        ]

    rows: list[list[str]] = []
    for fmt in scx_formats:
        row = [FORMAT_DISPLAY.get(fmt, fmt)]
        any_cell = False
        for ds in datasets:
            r = next(
                (x for x in results
                 if x.get("format") == fmt and x.get("dataset") == ds),
                None,
            )
            scaling = (r or {}).get("metadata", {}).get("scaling_wall_s", {}).get("full", {})
            t1 = scaling.get("1")
            t32 = scaling.get("32")
            if t1 is not None and t32 is not None:
                any_cell = True
                speedup = t1 / t32 if t32 else None
                row += [
                    _fmt_time(t1),
                    _fmt_time(t32),
                    f"{speedup:.1f}x" if speedup else "—",
                ]
            else:
                row += ["—", "—", "—"]
        if any_cell:
            rows.append(row)

    return TableBlock(headers=headers, rows=rows,
                      caption="SCX parallel write scaling (1T vs 32T, full pipeline)")


def parallel_write_scaling_table(datasets: list[str] | None = None) -> list[Block]:
    """Generate parallel write scaling table: Format x Thread count showing wall time and speedup.

    Returns a list of blocks — grouped by mode (full pipeline / write-only)
    and dataset, with one ``TableBlock`` per (mode, dataset) combination.
    """
    if datasets is None:
        datasets = ["census_500k", "census_1m", "census_5m"]
    results = load_all_results(benchmark="parallel_write_scaling")

    if not results:
        return [TextBlock("*No parallel write scaling results available yet.*")]

    # Collect all modes present in results.
    all_modes: set[str] = set()
    for r in results:
        meta = r.get("metadata", {})
        speedup = meta.get("speedup", {})
        all_modes.update(speedup.keys())

    mode_labels = {"full": "Full pipeline (h5ad read + write)",
                   "write_only": "Write only (in-memory AnnData)"}
    thread_cols = ["1 thread", "2 threads", "4 threads", "8 threads",
                   "16 threads", "32 threads", "Max speedup"]

    blocks: list[Block] = []
    for mode in ["full", "write_only"]:
        if mode not in all_modes:
            continue

        for ds in datasets:
            ds_results = [r for r in results if r.get("dataset") == ds]
            if not ds_results:
                continue

            sorted_results = sorted(
                ds_results,
                key=lambda r: FORMAT_ORDER.index(r.get("format", ""))
                if r.get("format", "") in FORMAT_ORDER else 999,
            )

            rows: list[list[str]] = []
            for r in sorted_results:
                fmt = r.get("format", "")
                display = FORMAT_DISPLAY.get(fmt, fmt)
                meta = r.get("metadata", {})
                scaling = meta.get("scaling_wall_s", {}).get(mode, {})
                speedup = meta.get("speedup", {}).get(mode, {})

                if not scaling:
                    continue

                cells = [display]
                for t in ["1", "2", "4", "8", "16", "32"]:
                    wall = scaling.get(t)
                    spd = speedup.get(t)
                    if wall is not None:
                        if spd is not None and t != "1":
                            cells.append(f"{_fmt_time(wall)} ({spd:.1f}x)")
                        else:
                            cells.append(_fmt_time(wall))
                    else:
                        cells.append("—")

                max_spd = max(speedup.values()) if speedup else None
                cells.append(f"{max_spd:.1f}x" if max_spd is not None else "—")
                rows.append(cells)

            if rows:
                label = mode_labels.get(mode, mode)
                blocks.append(TableBlock(
                    headers=["Format"] + thread_cols, rows=rows,
                    caption=f"Parallel write scaling — {label} — {SHORT_NAMES.get(ds, ds)}",
                ))

    return blocks or [TextBlock("*No parallel write scaling data to render.*")]


def fragment_ops_table(datasets: list[str] | None = None) -> TableBlock | TextBlock:
    """Fragment/manifest operation throughput — Operation x Dataset.

    Each cell shows median wall-clock with a secondary metric in
    parentheses (MB/s for append/compact, rows/s for delete, — for
    rollback). SCX-only benchmark.
    """
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="fragment_ops")
    if not results:
        return TextBlock("*No fragment-ops results available yet.*")

    # Operation -> dataset -> (median_wall, median_secondary, secondary_unit)
    ops_order = ["append", "delete", "compact", "rollback"]
    pivot: dict[str, dict[str, tuple[float | None, float | None, str]]] = {
        op: {} for op in ops_order
    }
    for r in results:
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        per_op = r.get("metadata", {}).get("per_op_medians", {}) or {}
        for op, bucket in per_op.items():
            wall = bucket.get("walls_median")
            if op in ("append", "compact"):
                pivot.setdefault(op, {})[ds] = (
                    wall, bucket.get("throughput_mb_s_median"), "MB/s"
                )
            elif op == "delete":
                pivot.setdefault(op, {})[ds] = (
                    wall, bucket.get("rows_per_sec_median"), "rows/s"
                )
            elif op == "rollback":
                pivot.setdefault(op, {})[ds] = (wall, None, "")

    headers = ["Operation"] + [SHORT_NAMES.get(d, d) for d in datasets]
    rows: list[list[str]] = []
    for op in ops_order:
        row_data = pivot.get(op, {})
        cells = [f"`{op}`"]
        for ds in datasets:
            entry = row_data.get(ds)
            if entry is None:
                cells.append("—")
                continue
            wall, secondary, unit = entry
            cell = _fmt_time(wall)
            if secondary is not None and unit:
                if unit == "rows/s":
                    cell += f" ({secondary:,.0f} rows/s)"
                else:
                    cell += f" ({secondary:,.1f} {unit})"
            cells.append(cell)
        rows.append(cells)

    return TableBlock(headers=headers, rows=rows,
                      caption="Fragment operation throughput")


def grouped_sharding_table(datasets: list[str] | None = None) -> TableBlock | TextBlock:
    """Grouped sharding (``scx sort --group-by`` / ``scx convert --group-by``).

    Metric × dataset. Shows the convert-time grouping one-pass-vs-two-pass
    wall/RSS (the density auto-routing story), the grouped-sort wall, group
    count, source matrix format, and the read-back correctness flag. SCX-only.
    """
    if datasets is None:
        datasets = ["pert_synth_10k", "nb_glm_synth", "replogle_k562", "tahoe_c38"]
    results = load_all_results(benchmark="grouped_sort")
    if not results:
        return TextBlock("*No grouped-sharding results available yet.*")

    by_ds = {r.get("dataset", ""): r for r in results if r.get("dataset", "") in datasets}
    present = [d for d in datasets if d in by_ds]
    if not present:
        return TextBlock("*No grouped-sharding results for the selected datasets.*")

    def _scn(r: dict, scenario: str, key: str) -> float | None:
        meds = r.get("metadata", {}).get("per_scenario_medians", {}) or {}
        return meds.get(scenario, {}).get(key)

    def _correct(r: dict) -> str:
        for run in r.get("runs", []):
            ex = run.get("extra", {})
            if ex.get("scenario") == "correctness":
                ok = ex.get("correctness_passed_int", 0) == 1
                ref = ex.get("reference_isolated_int", 1) == 1
                return "✓" if (ok and ref) else "✗"
        return "—"

    headers = ["Metric"] + [SHORT_NAMES.get(d, d) for d in present]

    def _row(label: str, fn) -> list[str]:
        return [label] + [fn(by_ds[d]) for d in present]

    def _wall_rss(scenario: str):
        def f(r: dict) -> str:
            w = _scn(r, scenario, "wall_s_median")
            rss = _scn(r, scenario, "peak_rss_mb_median")
            if w is None:
                return "—"
            return f"{_fmt_time(w)} ({rss:,.0f} MB)" if rss else _fmt_time(w)

        return f

    def _groups(r: dict) -> str:
        n = r.get("metadata", {}).get("n_groups")
        return f"{n:,}" if isinstance(n, int) else "—"

    rows = [
        _row(
            "source X format",
            lambda r: r.get("metadata", {}).get("source_matrix_format", "—"),
        ),
        _row("`sort --group-by` wall", _wall_rss("sort_group")),
        _row("`convert --group-by` one-pass", _wall_rss("convert_one")),
        _row("`convert --group-by` two-pass", _wall_rss("convert_two")),
        _row("groups", _groups),
        _row("read-back correct", _correct),
    ]
    return TableBlock(
        headers=headers,
        rows=rows,
        caption="Grouped sharding — convert one-pass vs two-pass + grouped sort",
    )


def grouped_read_table(datasets: list[str] | None = None) -> list[Block]:
    """Cross-format grouped-sharding head-to-head — scx vs shardad.

    One table per integer-count grouping dataset, with Format columns (SCX vs
    Shardad). Shows the grouped-write wall + file size, the per-perturbation
    ``read_group`` median + throughput, the ``read_reference`` wall, group count,
    and the read-back correctness flag. Sourced from ``grouped_read`` raw JSONs.
    """
    if datasets is None:
        datasets = ["nb_glm_synth", "replogle_k562", "tahoe_c38"]
    results = load_all_results(benchmark="grouped_read")
    if not results:
        return [TextBlock("*No grouped-read (scx vs shardad) results available yet.*")]

    _FMT_LABEL = {"scx_auto": "SCX (auto)", "shardad": "Shardad"}
    fmt_order = ["scx_auto", "shardad"]

    pivot: dict[str, dict[str, dict]] = {}
    for r in results:
        ds = r.get("dataset", "")
        fmt = r.get("format", "")
        if ds in datasets:
            pivot.setdefault(ds, {})[fmt] = r
    present = [d for d in datasets if d in pivot]
    if not present:
        return [TextBlock("*No grouped-read results for the selected datasets.*")]

    def _scn(r: dict, scenario: str, key: str) -> float | None:
        meds = r.get("metadata", {}).get("per_scenario_medians", {}) or {}
        return meds.get(scenario, {}).get(key)

    def _correct(r: dict) -> str:
        for run in r.get("runs", []):
            ex = run.get("extra", {})
            if ex.get("scenario") == "correctness":
                ok = ex.get("correctness_passed_int", 0) == 1
                ref = ex.get("reference_isolated_int", 1) == 1
                return "✓" if (ok and ref) else "✗"
        return "—"

    def _rg_throughput(r: dict) -> float | None:
        tot_cells = 0
        tot_wall = 0.0
        for run in r.get("runs", []):
            ex = run.get("extra", {})
            if ex.get("scenario") == "read_group":
                tot_cells += int(ex.get("cells_read", 0) or 0)
                tot_wall += float(run.get("wall_s", 0.0) or 0.0)
        if tot_wall > 0 and tot_cells > 0:
            return tot_cells / tot_wall
        return None

    def _groups(r: dict) -> str:
        n = r.get("metadata", {}).get("n_groups")
        return f"{n:,}" if isinstance(n, int) else "—"

    blocks: list[Block] = []
    for ds in present:
        by_fmt = pivot[ds]
        fmts = [f for f in fmt_order if f in by_fmt]
        headers = ["Metric"] + [_FMT_LABEL.get(f, f) for f in fmts]
        rows = [
            ["source X format"]
            + [by_fmt[f].get("metadata", {}).get("source_matrix_format", "—") for f in fmts],
            ["grouped write wall"]
            + [_fmt_time(_scn(by_fmt[f], "grouped_write", "wall_s_median")) for f in fmts],
            ["grouped file size"]
            + [
                _fmt_size(
                    _scn(by_fmt[f], "grouped_write", "output_size_bytes_median")
                    or by_fmt[f].get("file_size_bytes")
                )
                for f in fmts
            ],
            ["read_group median"]
            + [_fmt_time(_scn(by_fmt[f], "read_group", "wall_s_median")) for f in fmts],
            ["read_group cells/s"]
            + [
                (_fmt_num(_rg_throughput(by_fmt[f]), 0) if _rg_throughput(by_fmt[f]) else "—")
                for f in fmts
            ],
            ["read_reference wall"]
            + [_fmt_time(_scn(by_fmt[f], "read_reference", "wall_s_median")) for f in fmts],
            ["groups"] + [_groups(by_fmt[f]) for f in fmts],
            ["read-back correct"] + [_correct(by_fmt[f]) for f in fmts],
        ]
        blocks.append(
            TableBlock(
                headers=headers,
                rows=rows,
                caption=f"Grouped read/write head-to-head — {SHORT_NAMES.get(ds, ds)} (scx vs shardad)",
            )
        )
    return blocks


def capability_matrix_table() -> TableBlock:
    """Static scx-vs-shardad capability matrix.

    The head-to-head perf tables cover the axes both formats support; this grid
    records the *capability* differences that a race can't (they're not "slower",
    they're absent on one side). Hand-authored from the scx-vs-shardad comparison and the
    two projects' feature surfaces. ``✓`` = supported, ``✗`` = not supported,
    ``~`` = partial; ``✓✓`` = a notable strength.
    """
    rows = [
        ["Condition grouping (`read_group`)", "✓", "✓"],
        ["Reference isolation (`read_reference`)", "✓", "✓"],
        ["Streaming grouped iteration (`iter_group_shards`)", "✓", "✓"],
        ["Backed / out-of-core reads (bounded RSS)", "✓", "✗ (materializes full matrix)"],
        ["Lazy query engine (predicate pushdown)", "✓", "✗"],
        ["Gene / column projection on read", "✓", "✗ (row-only; read-then-slice)"],
        ["CSC / gene-major sidecar", "✓", "✗ (CSR only)"],
        ["In-decode dtype / density knobs", "~ (at convert time)", "✓"],
        ["Integer-count compression", "✓", "✓✓ (stronger on raw counts)"],
        ["ML training loader (native, shuffled batches)", "✓", "✗ (row-slice random access)"],
        ["Analysis accelerators (PCA/kNN/UMAP/DE)", "✓", "✗"],
        ["GPU acceleration (analysis + decode)", "✓", "✗ (GPU read unsupported this release)"],
        ["Multimodal (CITE-seq / Multiome / TEA-seq)", "✓", "✗"],
        ["Cloud-native I/O (S3 / GCS / Azure)", "✓", "✗"],
        ["Mutation ops (append/delete/compact/merge/rollback)", "✓", "✗ (metadata-tail only)"],
        ["R bindings (Seurat / SCE)", "✓", "✗ (Python only)"],
    ]
    return TableBlock(
        headers=["Capability", "SCX", "Shardad"],
        rows=rows,
        caption="Format capability matrix — scx vs shardad",
        notes=[
            "✓ supported · ✗ not supported · ~ partial · ✓✓ notable strength. "
            "shardad is a deep, narrow counts + condition-grouping store; scx is a "
            "broad platform (query engine, ML loaders, accelerators, cloud, "
            "multimodal, R). Perf on the shared axes is in the comparison tables above."
        ],
    )


def ooc_rss_boundary_table(datasets: list[str] | None = None) -> TableBlock | TextBlock:
    """Out-of-core peak-RSS boundary — scx streaming vs shardad materialize.

    Per dataset (rising n_obs): true peak RSS for scx bounded streaming, scx full
    materialize, and shardad full materialize. scx-stream stays ~flat while the
    materialize columns grow with n_obs — the out-of-core moat.
    """
    if datasets is None:
        datasets = ["census_500k", "census_1m", "census_5m"]
    results = load_all_results(benchmark="ooc_rss_boundary")
    if not results:
        return TextBlock("*No out-of-core RSS-boundary results available yet.*")

    # (dataset, format) -> per_scenario_medians
    by: dict[tuple[str, str], dict] = {}
    n_obs_of: dict[str, int] = {}
    for r in results:
        ds = r.get("dataset", "")
        by[(ds, r.get("format", ""))] = r.get("metadata", {}).get("per_scenario_medians", {}) or {}
        no = r.get("metadata", {}).get("n_obs")
        if isinstance(no, int):
            n_obs_of[ds] = no
    present = [d for d in datasets if (d, "scx_auto") in by or (d, "shardad") in by]
    if not present:
        return TextBlock("*No out-of-core RSS-boundary results for the selected datasets.*")

    def _peak(ds: str, fmt: str, scenario: str) -> str:
        meds = by.get((ds, fmt), {})
        v = meds.get(scenario, {}).get("peak_rss_mb_median")
        return _fmt_mem(v) if v is not None else "—"

    headers = ["Dataset", "n_obs", "scx stream (peak)", "scx materialize (peak)", "shardad materialize (peak)"]
    rows = []
    for ds in present:
        n = n_obs_of.get(ds)
        rows.append([
            SHORT_NAMES.get(ds, ds),
            f"{n:,}" if isinstance(n, int) else "—",
            _peak(ds, "scx_auto", "scx_stream"),
            _peak(ds, "scx_auto", "scx_materialize"),
            _peak(ds, "shardad", "shardad_materialize"),
        ])
    return TableBlock(
        headers=headers,
        rows=rows,
        caption="Out-of-core peak RSS — scx streaming (bounded) vs materialize",
        notes=[
            "True high-water-mark RSS. scx streaming stays ~flat as n_obs grows; "
            "the materialize columns grow with the matrix — shardad has no "
            "streaming path, so full materialize is its only read mode (and the "
            "capability boundary at atlas scale)."
        ],
    )


def cloud_filtered_table(datasets: list[str] | None = None) -> list[Block]:
    """Cloud filtered-query parity table — Format × Query.

    Each cell shows median wall-clock over the ``(dataset, query, format)``
    triple with p95 in parentheses, sourced from ``cloud_filtered`` raw JSONs.
    Only formats that declared ``"cloud_filtered"`` appear; runners that
    silently skipped (no capability) are omitted rather than rendered as —.

    Returns a list of blocks — one ``TableBlock`` per dataset.
    """
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="cloud_filtered")
    if not results:
        return [TextBlock("*No cloud_filtered results available yet.*")]

    # format -> dataset -> predicate -> (median, p95, n_runs)
    pivot: dict[str, dict[str, dict[str, tuple[float, float, int]]]] = {}
    predicates_seen: set[str] = set()
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        per_pred = r.get("metadata", {}).get("per_predicate_summary", {}) or {}
        for pname, bucket in per_pred.items():
            predicates_seen.add(pname)
            pivot.setdefault(fmt, {}).setdefault(ds, {})[pname] = (
                bucket.get("median_s"),
                bucket.get("p95_s"),
                bucket.get("n_runs", 0),
            )

    if not predicates_seen:
        return [TextBlock("*No cloud_filtered per-predicate summaries present.*")]

    predicate_order = sorted(predicates_seen)
    format_order = [f for f in FORMAT_ORDER if f in pivot] + [
        f for f in sorted(pivot) if f not in FORMAT_ORDER
    ]

    blocks: list[Block] = []
    for ds in datasets:
        ds_any = any(ds in pivot.get(f, {}) for f in format_order)
        if not ds_any:
            continue
        rows: list[list[str]] = []
        for fmt in format_order:
            row_data = pivot.get(fmt, {}).get(ds)
            if row_data is None:
                continue
            cells = [FORMAT_DISPLAY.get(fmt, fmt)]
            for pname in predicate_order:
                entry = row_data.get(pname)
                if entry is None:
                    cells.append("—")
                    continue
                median, p95, n = entry
                cell = _fmt_time(median)
                if p95 is not None and p95 != median:
                    cell += f" ({_fmt_time(p95)})"
                if n:
                    cell += f" ×{n}"
                cells.append(cell)
            rows.append(cells)
        blocks.append(TableBlock(
            headers=["Format"] + predicate_order, rows=rows,
            caption=f"Cloud filtered query — {SHORT_NAMES.get(ds, ds)} (median wall-clock, p95 in parens)",
        ))
    return blocks or [TextBlock("*No cloud_filtered rows to render.*")]


def cloud_reader_vs_pull_table(datasets: list[str] | None = None) -> list[Block]:
    """CloudReader vs full-pull table — per (dataset, scenario, method).

    Shows median wall-clock and bytes-downloaded across the four scenarios
    (``metadata_only``, ``selective_{5,20,80}pct``) and the two methods
    each scenario compares (``open_cloud`` / ``pull_full`` for metadata;
    ``pull_filtered`` / ``pull_full`` for selectivities).

    Returns a list of blocks — one ``TableBlock`` per dataset.
    """
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="cloud_reader_vs_pull")
    if not results:
        return [TextBlock("*No cloud_reader_vs_pull results available yet.*")]

    # dataset -> scenario -> method -> (median_wall, median_bytes, n_runs)
    pivot: dict[str, dict[str, dict[str, tuple[float, int, int]]]] = {}
    for r in results:
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        per_scen = r.get("metadata", {}).get("per_scenario_summary", {}) or {}
        for scen, methods in per_scen.items():
            for method, bucket in methods.items():
                pivot.setdefault(ds, {}).setdefault(scen, {})[method] = (
                    bucket.get("median_wall_s"),
                    int(bucket.get("median_bytes_downloaded", 0)),
                    int(bucket.get("n_runs", 0)),
                )
    if not pivot:
        return [TextBlock("*No per-scenario cloud_reader_vs_pull summaries present.*")]

    headers = ["Scenario", "Method", "Median wall", "Bytes downloaded", "n"]
    blocks: list[Block] = []
    for ds in datasets:
        per_scen = pivot.get(ds)
        if not per_scen:
            continue
        rows: list[list[str]] = []
        for scen in sorted(per_scen):
            for method, (wall, byts, n) in sorted(per_scen[scen].items()):
                rows.append([
                    scen, f"`{method}`", _fmt_time(wall),
                    _fmt_size(byts), str(n),
                ])
        blocks.append(TableBlock(
            headers=headers, rows=rows,
            caption=f"CloudReader vs full pull — {SHORT_NAMES.get(ds, ds)}",
        ))
    return blocks or [TextBlock("*No cloud_reader_vs_pull rows to render.*")]


def cost_model_table(datasets: list[str] | None = None) -> list[Block]:
    """Cost model table — USD per 1M cells queried × (layout, scenario).

    Pulled from ``cost_model`` raw JSONs' ``per_layout_scenario_median_usd
    _per_million`` metadata bucket. Scenarios run across the four canonical
    points (metadata, selective_5pct, selective_20pct, full_read) for
    every cloud layout declared in the benchmark's ``_LAYOUTS`` list.

    Returns a list of blocks — one ``TableBlock`` per dataset.
    """
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="cost_model")
    if not results:
        return [TextBlock("*No cost_model results available yet.*")]

    scenarios = ["metadata", "selective_5pct", "selective_20pct", "full_read"]
    # (dataset, layout, scenario) -> median_usd_per_million
    pivot: dict[tuple[str, str, str], float] = {}
    layouts_seen: set[str] = set()
    for r in results:
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        per = r.get("metadata", {}).get(
            "per_layout_scenario_median_usd_per_million", {}
        ) or {}
        for combo, val in per.items():
            if "::" not in combo:
                continue
            layout, scen = combo.split("::", 1)
            layouts_seen.add(layout)
            pivot[(ds, layout, scen)] = float(val)

    if not layouts_seen:
        return [TextBlock("*No cost_model per-(layout, scenario) medians present.*")]

    headers = ["Layout"] + scenarios
    blocks: list[Block] = []
    for ds in datasets:
        ds_any = any((ds, layout, scen) in pivot
                     for layout in layouts_seen for scen in scenarios)
        if not ds_any:
            continue
        rows: list[list[str]] = []
        for layout in sorted(layouts_seen):
            cells = [f"`{layout}`"]
            for scen in scenarios:
                v = pivot.get((ds, layout, scen))
                cells.append(f"${v:.6f}" if v is not None else "—")
            rows.append(cells)
        blocks.append(TableBlock(
            headers=headers, rows=rows,
            caption=f"Cost model — {SHORT_NAMES.get(ds, ds)} (USD per 1M cells, GCS same-region)",
        ))
    return blocks or [TextBlock("*No cost_model rows to render.*")]


def gcp_matrix_table(datasets: list[str] | None = None) -> TableBlock | TextBlock:
    """GCP compute-node matrix — Instance × Format × Dataset.

    Pivots ``cloud_read`` results that carry ``system.gcp.instance_type``
    (set by ``submit_gcp_matrix.py`` via ``SCX_BENCH_GCP_INSTANCE``) into
    a table of median wall-clock with p95 in parentheses. Rows collapse
    the instance and format, columns span the selected datasets.

    Runs without a GCP instance label are excluded so on-cluster results
    don't contaminate the matrix view.
    """
    from benchmarks.comprehensive.config import GCP_INSTANCE_TYPES

    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="cloud_read")
    results = [
        r for r in results
        if r.get("system", {}).get("gcp", {}).get("instance_type")
    ]
    if not results:
        return TextBlock("*No GCP-matrix cloud_read results available yet.*")

    # (instance, format) -> dataset -> (median, p95)
    pivot: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    instances_seen: set[str] = set()
    for r in results:
        instance = r["system"]["gcp"]["instance_type"]
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        runs = r.get("runs", []) or []
        walls = [x.get("wall_s") for x in runs if x.get("wall_s") is not None]
        if not walls:
            continue
        walls_sorted = sorted(walls)
        median = walls_sorted[len(walls_sorted) // 2]
        p95 = walls_sorted[max(0, int(0.95 * len(walls_sorted)) - 1)]
        pivot.setdefault((instance, fmt), {})[ds] = (median, p95)
        instances_seen.add(instance)

    if not pivot:
        return TextBlock("*No GCP-matrix cloud_read rows match the selected datasets.*")

    instance_order = [i for i in GCP_INSTANCE_TYPES if i in instances_seen] + [
        i for i in sorted(instances_seen) if i not in GCP_INSTANCE_TYPES
    ]
    headers = ["Instance", "Format"] + [SHORT_NAMES.get(d, d) for d in datasets]
    rows: list[list[str]] = []
    for instance in instance_order:
        fmt_rows = sorted(
            {fmt for (i, fmt) in pivot if i == instance},
            key=lambda f: (
                FORMAT_ORDER.index(f) if f in FORMAT_ORDER else len(FORMAT_ORDER)
            ),
        )
        for fmt in fmt_rows:
            per_ds = pivot.get((instance, fmt), {})
            cells = [f"`{instance}`", FORMAT_DISPLAY.get(fmt, fmt)]
            for ds in datasets:
                entry = per_ds.get(ds)
                if entry is None:
                    cells.append("—")
                    continue
                median, p95 = entry
                cell = _fmt_time(median)
                if p95 != median:
                    cell += f" ({_fmt_time(p95)})"
                cells.append(cell)
            rows.append(cells)

    # Egress class annotation goes into TableBlock.notes
    egress_note = "*Egress class per instance:* " + ", ".join(
        f"`{i}`={GCP_INSTANCE_TYPES[i]['egress_gbps']} Gbps"
        for i in instance_order if i in GCP_INSTANCE_TYPES
    )
    return TableBlock(
        headers=headers, rows=rows,
        caption="GCP compute-node matrix (median wall-clock, p95 in parens)",
        notes=[egress_note],
    )


def ml_loader_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate ML loader comparison table: Format x Dataset showing b/s and TTFB."""
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="ml_loader")

    # Collect batches/sec for hvg_norm scenario (the primary comparison scenario)
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue

        # Find the hvg_norm runs
        hvg_norm_bps = []
        for run in r.get("runs", []):
            if run.get("extra", {}).get("scenario") == "hvg_norm":
                bps = run["extra"].get("batches_per_sec")
                if bps is not None:
                    hvg_norm_bps.append(bps)

        # Fallback to raw scenario if no hvg_norm
        if not hvg_norm_bps:
            for run in r.get("runs", []):
                if run.get("extra", {}).get("scenario") == "raw":
                    bps = run["extra"].get("batches_per_sec")
                    if bps is not None:
                        hvg_norm_bps.append(bps)

        if hvg_norm_bps:
            pivot.setdefault(fmt, {})[ds] = statistics.median(hvg_norm_bps)

    def fmt_bps(v):
        if v is None:
            return "—"
        return f"{v:,.1f}"

    return _pivot_to_tableblock(
        pivot, datasets, fmt_bps, bold_best="max", title="Loader",
        caption="ML training loader throughput (batches/sec, hvg_norm scenario)",
    )


def multimodal_compression_table(datasets: list[str] | None = None) -> TableBlock:
    """File-size + compression ratio for multimodal_compression rows.

    Compares SCX v2 multimodal (per-modality auto + uniform auto) vs
    h5mu (uncompressed + gzip) and zarr-mudata (zstd) on the registered
    multimodal datasets. Value: ``file_size_bytes`` rendered as a
    human-readable size.
    """
    if datasets is None:
        datasets = ["cite_seq_pbmc_5k", "multiome_pbmc_10k"]
    results = load_all_results(benchmark="multimodal_compression")
    pivot = _build_pivot(results, "file_size_bytes", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_size, bold_best="min", title="Format",
        caption="Multimodal file sizes by format and dataset",
    )


def multimodal_compression_ratio_table(datasets: list[str] | None = None) -> TableBlock:
    """Compression ratio relative to h5mu uncompressed for the same dataset.

    Higher is better. Pulled from each row's
    ``metadata.compression_ratio_vs_h5mu``.
    """
    if datasets is None:
        datasets = ["cite_seq_pbmc_5k", "multiome_pbmc_10k"]
    results = load_all_results(benchmark="multimodal_compression")
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        fmt = r.get("format", "")
        ratio = (r.get("metadata") or {}).get("compression_ratio_vs_h5mu")
        if ratio is not None:
            pivot.setdefault(fmt, {})[ds] = float(ratio)

    def fmt_ratio(v):
        return "—" if v is None else f"{v:.2f}x"

    return _pivot_to_tableblock(
        pivot, datasets, fmt_ratio, bold_best="max", title="Format",
        caption="Multimodal compression ratio vs h5mu uncompressed",
    )


def multimodal_training_table(datasets: list[str] | None = None) -> TableBlock:
    """Multimodal training-loader throughput (batches/sec) — median across runs.

    Pulls ``batches_per_sec`` from each row's ``runs[].extra``. Higher is
    better. Mirrors the ``ml_loader_table`` shape.
    """
    if datasets is None:
        datasets = ["cite_seq_pbmc_5k", "multiome_pbmc_10k"]
    results = load_all_results(benchmark="multimodal_training")
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        fmt = r.get("format", "")
        bps_vals: list[float] = []
        for run in r.get("runs", []) or []:
            extra = run.get("extra") or {}
            v = extra.get("batches_per_sec")
            if v is not None:
                try:
                    bps_vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if bps_vals:
            pivot.setdefault(fmt, {})[ds] = statistics.median(bps_vals)

    def fmt_bps(v):
        return "—" if v is None else f"{v:,.1f}"

    return _pivot_to_tableblock(
        pivot, datasets, fmt_bps, bold_best="max", title="Format",
        caption="Multimodal training loader throughput (batches/sec)",
    )


def multimodal_training_ttfb_table(datasets: list[str] | None = None) -> TableBlock:
    """Time-to-first-batch (seconds) for multimodal_training. Lower is better."""
    if datasets is None:
        datasets = ["cite_seq_pbmc_5k", "multiome_pbmc_10k"]
    results = load_all_results(benchmark="multimodal_training")
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        fmt = r.get("format", "")
        ttfbs: list[float] = []
        for run in r.get("runs", []) or []:
            extra = run.get("extra") or {}
            v = extra.get("time_to_first_batch_s")
            if v is not None:
                try:
                    ttfbs.append(float(v))
                except (TypeError, ValueError):
                    pass
        if ttfbs:
            pivot.setdefault(fmt, {})[ds] = statistics.median(ttfbs)

    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min", title="Format",
        caption="Multimodal training time-to-first-batch (seconds)",
    )


def bench_csc_dispatch_table(datasets: list[str] | None = None) -> TableBlock:
    """CSC vs CSR dispatch perf for qc_metrics / hvg / de / pseudobulk.

    Format keys are ``bench_csc__<op>_<axis>``; we surface the median
    wall-time per (format, dataset). Aimed at making the Phase L.3
    sweep visible in the report.
    """
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="bench_csc_dispatch")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_tableblock(
        pivot, datasets, _fmt_time, bold_best="min", title="Variant",
        caption="CSC vs CSR dispatch wall time by variant and dataset",
    )


def _classify_test_status(test: dict[str, Any]) -> str:
    """Classify a single test result into Pass / Fail / Skipped / Not applicable.

    Fixes the original bug where dependency-skipped tests (e.g. pseudobulk
    with ``"pydeseq2 not installed"``) were rendered as failures.
    """
    error = test.get("error", "")
    passed_raw = test.get("passed")

    # Dependency-skipped tests: marked passed=false but have a skip-like error
    if error and any(kw in error.lower() for kw in ("not installed", "skipped", "not available")):
        return "Skipped"

    if passed_raw in (True, "True"):
        return "Pass"
    if passed_raw in (False, "False"):
        # True failure — only if there's no skip-like error
        return "Fail"
    return "Not run"


def _recount_correctness(results_list: list[dict[str, Any]]) -> tuple[int, int, int]:
    """Recount pass/fail/skipped using proper status classification.

    Returns (n_passed, n_failed, n_skipped).
    """
    n_pass = n_fail = n_skip = 0
    for test in results_list:
        status = _classify_test_status(test)
        if status == "Pass":
            n_pass += 1
        elif status == "Fail":
            n_fail += 1
        else:  # Skipped, Not applicable, Not run
            n_skip += 1
    return n_pass, n_fail, n_skip


def correctness_table() -> TableBlock | TextBlock:
    """Generate correctness validation summary table.

    Uses proper status classification so that dependency-skipped tests
    (e.g. pseudobulk_dex when pydeseq2 is missing) count as Skipped
    rather than Failed. The ``Status`` column shows Pass/Fail/Mixed
    for the overall harness result.
    """
    results = load_all_results()  # Load all, filter correctness
    correctness = [
        r for r in results
        if r.get("harness") or r.get("benchmark", "").startswith("correctness")
    ]

    if not correctness:
        return TextBlock("_No correctness results found._")

    headers = ["Test", "Dataset", "Passed", "Failed", "Skipped", "Status", "Duration"]
    rows: list[list[str]] = []

    for r in correctness:
        harness = r.get("harness", r.get("benchmark", "unknown"))
        dataset = r.get("dataset", "—")
        duration = r.get("total_duration_s")

        # Recount using proper classification if per-test results exist
        per_test = r.get("results", [])
        if per_test:
            n_passed, n_failed, n_skipped = _recount_correctness(per_test)
        else:
            # Fall back to raw JSON counts for schema_version 2 results
            # that store counts in runs[].extra
            runs = r.get("runs", [])
            if runs and runs[0].get("extra", {}).get("n_passed") is not None:
                extra = runs[0]["extra"]
                n_passed = extra.get("n_passed", 0)
                n_failed = extra.get("n_failed", 0)
                n_skipped = extra.get("n_skipped", 0)
            else:
                n_passed = r.get("n_passed", 0)
                n_failed = r.get("n_failed", 0)
                n_skipped = r.get("n_skipped", 0)

        if n_failed > 0:
            status = "**Fail**"
        elif n_passed > 0 and n_skipped == 0:
            status = "Pass"
        elif n_passed > 0:
            status = "Pass (partial)"
        else:
            status = "Not run"

        rows.append([
            harness, dataset, str(n_passed), str(n_failed),
            str(n_skipped), status, _fmt_time(duration),
        ])

    return TableBlock(headers=headers, rows=rows,
                      caption="Correctness validation summary")


def correctness_detail_table(dataset: str = "pbmc3k") -> TableBlock | TextBlock:
    """Generate per-function correctness detail table for a given dataset.

    Uses proper status classification: ``Pass``, ``Fail``, ``Skipped``
    (for dependency-missing tests like pseudobulk), ``Not run``.
    Skipped rows are clearly labeled and not rendered as ``**No**``.
    """
    results = load_all_results()
    scanpy_equiv = [
        r for r in results
        if r.get("harness") == "scanpy_equivalence" and r.get("dataset") == dataset
    ]

    if not scanpy_equiv:
        return TextBlock(f"_No scanpy equivalence results for {dataset}._")

    r = scanpy_equiv[-1]  # Most recent
    headers = ["Function", "Status", "Key Metric", "Value", "Threshold", "Duration", "Notes"]
    rows: list[list[str]] = []

    for test in r.get("results", []):
        name = test.get("name", "unknown")
        status = _classify_test_status(test)
        error = test.get("error", "")
        metrics = test.get("metrics", {})
        thresholds = test.get("thresholds", {})
        duration = test.get("duration_s")

        # Status display with appropriate formatting
        if status == "Pass":
            status_str = "Pass"
        elif status == "Fail":
            status_str = "**Fail**"
        elif status == "Skipped":
            status_str = "_Skipped_"
        else:
            status_str = "_Not run_"

        # Notes: surface skip reason or error
        notes = ""
        if status == "Skipped" and error:
            notes = error
        elif status == "Fail" and error:
            notes = error

        # Pick the most representative metric.
        # CAVEAT: this picks the first metric by dict-insertion order, which
        # may not be the metric that triggered a failure. e.g. on
        # tabula_sapiens_100k, `rank_genes_groups` reports both
        # `min_top100_overlap_pct` (passes at 100%) and `min_pval_spearman_r`
        # (the brittle per-cluster check that actually fails near 0.80) — and
        # the table currently shows the passing one. The `rank_genes_groups`
        # spearman check itself is also fragile: it correlates p-values at
        # the same RANK between methods (not the same gene) and has zero
        # margin against the 0.80 threshold, so any numerical jitter in BH
        # tie structure for a single noisy cluster can flip the test.
        # TODO: prefer the failing metric when status == "Fail".
        if metrics:
            metric_name = next(iter(metrics))
            metric_val = metrics[metric_name]
            threshold_val = thresholds.get(metric_name)
            if isinstance(metric_val, float):
                metric_str = f"{metric_val:.6f}" if metric_val < 1 else f"{metric_val:.4f}"
            else:
                metric_str = str(metric_val)
            threshold_str = str(threshold_val) if threshold_val is not None else "—"
        else:
            metric_name = "—"
            metric_str = "—"
            threshold_str = "—"

        rows.append([name, status_str, metric_name, metric_str,
                     threshold_str, _fmt_time(duration), notes])

    return TableBlock(headers=headers, rows=rows,
                      caption=f"Scanpy equivalence detail ({dataset})")


def correctness_detail_all_datasets() -> list[Block]:
    """Generate scanpy equivalence detail tables for every dataset with data.

    Returns one ``TableBlock`` per dataset that has ``scanpy_equivalence``
    harness results, enabling Chapter 3 to show per-dataset breakdowns
    rather than only pbmc3k.
    """
    results = load_all_results()
    datasets_seen: list[str] = []
    for r in results:
        if r.get("harness") == "scanpy_equivalence":
            ds = r.get("dataset", "")
            if ds and ds not in datasets_seen:
                datasets_seen.append(ds)

    if not datasets_seen:
        return [TextBlock("_No scanpy equivalence results found._")]

    blocks: list[Block] = []
    for ds in datasets_seen:
        tbl = correctness_detail_table(dataset=ds)
        blocks.append(tbl)
    return blocks


def correctness_dataset_summary_table() -> TableBlock | TextBlock:
    """Dataset-level correctness summary across all harness types.

    One row per (dataset, harness) with properly classified pass/fail/skip
    counts and overall status. This provides the at-a-glance view for
    Chapter 3 and the executive summary.
    """
    results = load_all_results()
    correctness = [
        r for r in results
        if r.get("harness") or r.get("benchmark", "").startswith("correctness")
    ]
    if not correctness:
        return TextBlock("_No correctness results found._")

    headers = ["Dataset", "Harness", "Passed", "Failed", "Skipped", "Status"]
    rows: list[list[str]] = []

    # Group by dataset
    by_ds: dict[str, list[dict]] = {}
    for r in correctness:
        ds = r.get("dataset", "unknown")
        by_ds.setdefault(ds, []).append(r)

    for ds in sorted(by_ds.keys()):
        for r in by_ds[ds]:
            harness = r.get("harness", r.get("benchmark", "unknown"))
            per_test = r.get("results", [])
            if per_test:
                n_p, n_f, n_s = _recount_correctness(per_test)
            else:
                runs = r.get("runs", [])
                if runs and runs[0].get("extra", {}).get("n_passed") is not None:
                    extra = runs[0]["extra"]
                    n_p = extra.get("n_passed", 0)
                    n_f = extra.get("n_failed", 0)
                    n_s = extra.get("n_skipped", 0)
                else:
                    n_p = r.get("n_passed", 0)
                    n_f = r.get("n_failed", 0)
                    n_s = r.get("n_skipped", 0)

            if n_f > 0:
                status = "**Fail**"
            elif n_p > 0 and n_s == 0:
                status = "Pass"
            elif n_p > 0:
                status = "Pass (partial)"
            else:
                status = "Not run"

            rows.append([
                SHORT_NAMES.get(ds, ds), harness,
                str(n_p), str(n_f), str(n_s), status,
            ])

    return TableBlock(headers=headers, rows=rows,
                      caption="Dataset-level correctness summary")


def pipeline_agreement_table() -> list[Block]:
    """Pipeline-level biological agreement tables.

    Pulls from ``preprocessing_paths`` harness results which compare
    three preprocessing pathways (A: scanpy, B: pyscx eager,
    C: pyscx lazy) on metrics like Leiden ARI, PCA cosine similarity,
    and DE overlap.

    Returns one ``TableBlock`` per dataset.
    """
    results = load_all_results()
    preproc = [
        r for r in results
        if r.get("harness") == "preprocessing_paths"
    ]
    if not preproc:
        return [TextBlock("_No preprocessing path comparison results found._")]

    blocks: list[Block] = []
    for r in sorted(preproc, key=lambda x: x.get("dataset", "")):
        ds = r.get("dataset", "unknown")
        per_test = r.get("results", [])
        if not per_test:
            continue

        headers = ["Pipeline Test", "Status", "Key Metric", "Value", "Threshold"]
        rows: list[list[str]] = []
        for test in per_test:
            name = test.get("name", "unknown")
            status = _classify_test_status(test)
            status_str = "Pass" if status == "Pass" else (
                "**Fail**" if status == "Fail" else f"_{status}_"
            )
            metrics = test.get("metrics", {})
            thresholds = test.get("thresholds", {})
            if metrics:
                # Show the most important metric
                metric_name = next(iter(thresholds)) if thresholds else next(iter(metrics))
                val = metrics.get(metric_name)
                thr = thresholds.get(metric_name)
                if isinstance(val, float):
                    val_str = f"{val:.6f}" if val < 1 else f"{val:.4f}"
                else:
                    val_str = str(val)
                thr_str = str(thr) if thr is not None else "—"
            else:
                metric_name = "—"
                val_str = "—"
                thr_str = "—"
            rows.append([name, status_str, metric_name, val_str, thr_str])

        blocks.append(TableBlock(
            headers=headers, rows=rows,
            caption=f"Pipeline agreement — {SHORT_NAMES.get(ds, ds)}",
        ))
    return blocks


def accelerator_parity_table() -> TableBlock | TextBlock:
    """Accelerator parity summary — SCX vs scanpy/baseline per operation.

    Compares accelerator benchmark results across implementations for the
    same (operation, dataset) by extracting parity metrics from the
    ``runs[].extra`` fields (e.g. cosine_sim_mean for PCA, recall for kNN).

    This table belongs adjacent to accelerator timing tables.
    """
    store = get_store()
    accel_benchmarks = [
        "accel_pca", "accel_knn", "accel_umap", "accel_leiden",
        "accel_preprocess", "accel_hvg",
    ]

    # Collect all rows grouped by (benchmark, dataset)
    # Each group has multiple implementations (pyscx_cpu, scanpy_cpu, etc.)
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for bench in accel_benchmarks:
        for row in store.by_benchmark(bench):
            ds = row.dataset
            impl = row.format  # format field holds impl like "accel_pca__pyscx_cpu_auto"
            # Extract the implementation suffix
            impl_short = impl.replace(f"{bench}__", "")
            by_key.setdefault((bench, ds), {})[impl_short] = row

    if not by_key:
        return TextBlock("_No accelerator benchmark results found._")

    headers = ["Operation", "Dataset", "SCX impl", "SCX time", "Baseline", "Baseline time",
               "Speedup", "Parity metric", "Parity value"]
    rows: list[list[str]] = []

    op_labels = {
        "accel_pca": "PCA", "accel_knn": "kNN", "accel_umap": "UMAP",
        "accel_leiden": "Leiden", "accel_preprocess": "Preprocess",
        "accel_hvg": "HVG",
    }
    parity_keys = {
        "accel_pca": "cosine_sim_min",
        "accel_knn": "recall_vs_scanpy",
        "accel_umap": "trustworthiness",
        "accel_leiden": "ari_vs_leidenalg",
        "accel_preprocess": "max_abs_diff_vs_scanpy",
        "accel_hvg": "hvg_overlap_vs_scanpy",
    }

    for (bench, ds), impls in sorted(by_key.items()):
        op_label = op_labels.get(bench, bench)
        # Find SCX CPU impl and scanpy baseline
        scx_impl = None
        baseline_impl = None
        for impl_name, row in impls.items():
            if "pyscx_cpu" in impl_name:
                scx_impl = (impl_name, row)
            elif "scanpy" in impl_name or "leidenalg" in impl_name:
                baseline_impl = (impl_name, row)

        if not scx_impl:
            continue

        scx_name, scx_row = scx_impl
        scx_time = scx_row.median_wall_s

        if baseline_impl:
            base_name, base_row = baseline_impl
            base_time = base_row.median_wall_s
            speedup = (base_time / scx_time) if scx_time and base_time else None
            speedup_str = f"{speedup:.1f}x" if speedup else "—"
        else:
            base_name = "—"
            base_time = None
            speedup_str = "—"

        # Extract parity metric from SCX runs; fall back to the result-level
        # metadata (some runners populate metadata but not every run's extra).
        parity_key = parity_keys.get(bench, "")
        parity_val = None
        if scx_row.runs:
            extra = scx_row.runs[0].get("extra", {})
            parity_val = extra.get(parity_key)
        if parity_val is None and parity_key:
            parity_val = scx_row.metadata.get(parity_key)
        parity_str = "—"
        if parity_val is not None:
            if isinstance(parity_val, float):
                parity_str = f"{parity_val:.4f}" if parity_val < 100 else f"{parity_val:.1f}"
            else:
                parity_str = str(parity_val)

        rows.append([
            op_label, SHORT_NAMES.get(ds, ds),
            scx_name, _fmt_time(scx_time),
            base_name if baseline_impl else "—",
            _fmt_time(base_time),
            speedup_str,
            parity_key or "—", parity_str,
        ])

    return TableBlock(headers=headers, rows=rows, wide=True,
                      caption="Accelerator parity — SCX vs baseline (CPU)")


def accelerator_gpu_vs_rapids_comparison_table() -> TableBlock | TextBlock:
    """Surviving native GPU paths vs rapids-singlecell, per (operation, dataset).

    Post rapids transition, in-VRAM PCA / kNN / UMAP / preprocess / HVG route to
    rapids-singlecell, so SCX's default GPU path for those *is* rapids (a
    head-to-head ratio is ~1.0 by construction). This table therefore compares
    only the native GPU kernels that survive because they cover a regime rapids
    does not — PCA randomized/streaming (>VRAM), HVG seurat_v3, Leiden cuGraph,
    preprocess streaming/ML-loader — against rapids, to motivate the routing
    decision. The native impl is chosen by explicit name (see
    ``surviving_native_impl``) so the removed Phase-3 variants (`pyscx_gpu_cov`,
    `pyscx_gpu_cagra`, native UMAP) and the `pyscx_gpu_no_rapids` diagnostic
    fallback are never selected, even if stale JSONs linger in results/raw/.
    The ratio is *surfaced* (not gated — rapids version drift must not fail the
    build); correctness is gated separately.
    """
    store = get_store()
    # Surviving native GPU paths after the rapids transition.
    # In-VRAM PCA / kNN / UMAP / preprocess / HVG route to rapids-singlecell, so
    # SCX's *default* GPU path for those IS rapids (ratio ~1.0 by construction,
    # not worth a row). A distinct native GPU kernel survives only where it
    # covers a regime rapids does not; this table compares THOSE against rapids
    # to motivate the routing decision. Each entry lists the accepted impl keys
    # in preference order — selecting by explicit name means the removed Phase-3
    # variants (`pyscx_gpu_cov`, `pyscx_gpu_cagra`, the native `pyscx_gpu` UMAP)
    # and the rapids-disabled diagnostic fallback (`pyscx_gpu_no_rapids`) are
    # never picked, even if stale JSONs for them still sit in results/raw/.
    # kNN and UMAP have no surviving standalone native path → intentionally
    # omitted (in-VRAM delegates fully to rapids).
    surviving_native_impl: dict[str, tuple[str, ...]] = {
        "accel_pca": ("pyscx_gpu_rand_hh", "pyscx_gpu_rand_chol"),
        "accel_hvg": ("pyscx_gpu",),
        "accel_leiden": ("pyscx_gpu",),
        "accel_preprocess": ("pyscx_gpu",),
    }
    accel_benchmarks = list(surviving_native_impl)
    # Labels name the surviving native path so the ratio is not misread as
    # "SCX's GPU PCA is 13x slower" — it is the >VRAM randomized path, kept
    # precisely because in-VRAM routes to rapids.
    op_labels = {
        "accel_pca": "PCA (randomized / >VRAM)",
        "accel_leiden": "Leiden (cuGraph)",
        "accel_preprocess": "Preprocess (streaming)",
        "accel_hvg": "HVG (seurat_v3)",
    }
    parity_keys = {
        "accel_pca": "subspace_cos_min",
        "accel_knn": "recall_vs_scanpy",
        "accel_umap": "trustworthiness",
        "accel_leiden": "ari_vs_leidenalg",
        "accel_preprocess": "max_abs_diff_vs_scanpy",
        "accel_hvg": "hvg_overlap_vs_scanpy",
    }
    # Polarity marker for the "Metric" column. Most accel parity metrics are
    # higher-is-better (cosine / recall / trustworthiness / overlap / ARI), but
    # preprocess's `max_abs_diff_vs_scanpy` is an error term (lower-is-better).
    # Annotate the metric name so the side-by-side SCX/rapids values in one
    # column aren't read with the wrong polarity.
    lower_is_better = {"max_abs_diff_vs_scanpy"}

    def _fmt_metric_name(key: str) -> str:
        if not key:
            return "—"
        return f"{key} (↓)" if key in lower_is_better else f"{key} (↑)"

    def _metric(row: Any, key: str) -> Any:
        if row.runs:
            v = row.runs[0].get("extra", {}).get(key)
            if v is not None:
                return v
        return row.metadata.get(key)

    def _fmt_metric(v: Any) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
        return str(v)

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for bench in accel_benchmarks:
        for row in store.by_benchmark(bench):
            impl_short = row.format.replace(f"{bench}__", "")
            by_key.setdefault((bench, row.dataset), {})[impl_short] = row

    headers = ["Operation", "Dataset", "SCX GPU time", "rapids time",
               "SCX/rapids", "Metric", "SCX", "rapids"]
    rows: list[list[str]] = []
    # Track, per benchmark, whether rapids data existed and whether a
    # surviving-native impl actually matched — so a variant-key rename that
    # silently empties an op's rows surfaces a warning instead of vanishing.
    rapids_benches: set[str] = set()
    native_matched_benches: set[str] = set()
    for (bench, ds), impls in sorted(by_key.items()):
        rapids = impls.get("rapids_singlecell_gpu")
        if rapids is None:
            continue
        rapids_benches.add(bench)
        # Select the surviving native GPU impl by explicit name (preference
        # order), so removed variants / the no_rapids fallback are never picked.
        scx = next(
            (impls[k] for k in surviving_native_impl.get(bench, ()) if k in impls),
            None,
        )
        if scx is None:
            continue
        native_matched_benches.add(bench)
        scx_t = scx.median_wall_s
        rap_t = rapids.median_wall_s
        ratio = (scx_t / rap_t) if (scx_t and rap_t) else None
        mkey = parity_keys.get(bench, "")
        rows.append([
            op_labels.get(bench, bench), SHORT_NAMES.get(ds, ds),
            _fmt_time(scx_t), _fmt_time(rap_t),
            f"{ratio:.2f}x" if ratio else "—",
            _fmt_metric_name(mkey), _fmt_metric(_metric(scx, mkey)),
            _fmt_metric(_metric(rapids, mkey)),
        ])

    # A configured op with rapids data but no native match means its expected
    # impl key(s) drifted (rename / removal) — warn so the row isn't silently
    # dropped from the report unnoticed.
    for bench in sorted(rapids_benches - native_matched_benches):
        logger.warning(
            "native-vs-rapids: no surviving-native impl %s found for %s "
            "(rapids data present) — variant-key drift? row(s) omitted.",
            surviving_native_impl.get(bench, ()),
            bench,
        )

    if not rows:
        return TextBlock(
            "_No rapids-singlecell comparison data — run the "
            "`accel_*__rapids_singlecell_gpu` variants on a GPU host with "
            "rapids-singlecell installed._"
        )
    return TableBlock(headers=headers, rows=rows, wide=True,
                      caption="Surviving native GPU paths vs rapids-singlecell — "
                              "wall-time ratio (>1 = rapids faster) + accuracy")


# ---------------------------------------------------------------------------
# Differential expression (CPU + GPU): per-cell (accel_de) + NB-GLM
# ---------------------------------------------------------------------------

# Benchmark keys that carry DE results.
_DE_BENCHMARKS = ["accel_de", "accel_de_nb_glm"]

# Deterministic gate signals surfaced by de_route_correctness_table (1.0 = ✓).
_DE_ROUTE_SIGNALS = [
    "nb_glm_route_gpu_correct",
    "nb_glm_cpu_gpu_concordant",
    "nb_glm_pdex_ref_concordant",
    "wilcoxon_route_gpu_correct",
    "wilcoxon_route_csc_direct",
    "de_route_csc_direct",
    "de_route_resident_csr",
]


def _de_method_device(bench: str, fmt: str) -> tuple[str, str]:
    """(human method label, device) from a DE format key."""
    tail = fmt[len(bench) + 2:] if fmt.startswith(bench + "__") else fmt
    device = "GPU" if tail.endswith("_gpu") else ("CPU" if tail.endswith("_cpu") else "—")
    core = tail
    for suf in ("_gpu", "_cpu"):
        if core.endswith(suf):
            core = core[: -len(suf)]
    # Key on (bench, core) so a bare core like "pyscx" can't collide across
    # benchmarks (e.g. a future accel_de__pyscx_cpu vs accel_de_nb_glm__pyscx_cpu);
    # fall back to the core-only key, then the raw core.
    label_map = {
        ("accel_de", "scanpy_wilcoxon"): "scanpy Wilcoxon",
        ("accel_de", "pyscx_wilcoxon"): "pyscx Wilcoxon",
        ("accel_de", "pyscx_pdex_ref"): "pyscx pdex_ref",
        ("accel_de_nb_glm", "pyscx"): "pyscx NB-GLM",
    }
    label = label_map.get((bench, core)) or {
        "scanpy_wilcoxon": "scanpy Wilcoxon",
        "pyscx_wilcoxon": "pyscx Wilcoxon",
        "pyscx_pdex_ref": "pyscx pdex_ref",
        "pyscx": "pyscx NB-GLM",
    }.get(core, core or bench)
    return label, device


def _de_median_extra(row, metric: str) -> float | None:
    """Median numeric ``runs[].extra[metric]`` across a row's runs."""
    vals: list[float] = []
    for r in row.runs:
        v = (r.get("extra") or {}).get(metric)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append(float(v))
    return statistics.median(vals) if vals else None


def _de_first_extra(row, metric: str):
    """First non-None ``runs[].extra[metric]`` (for string fields like route)."""
    for r in row.runs:
        v = (r.get("extra") or {}).get(metric)
        if v is not None:
            return v
    return row.metadata.get(metric)


def _de_peak_rss_mb(row) -> float | None:
    vals = [
        r.get("peak_rss_mb")
        for r in row.runs
        # `> 0` rather than truthiness so a legitimate 0.0 isn't silently dropped.
        if isinstance(r.get("peak_rss_mb"), (int, float)) and r.get("peak_rss_mb") > 0
    ]
    return float(statistics.median(vals)) if vals else None


def de_performance_table() -> TableBlock | TextBlock:
    """DE wall time + peak RSS per (method × dataset × device).

    Pulls both per-cell DE (``accel_de__*``: scanpy/pyscx Wilcoxon, pdex_ref)
    and pseudobulk NB-GLM (``accel_de_nb_glm__*``) rows.
    """
    store = get_store()
    rows_data: list[tuple[str, str, str, float | None, float | None, str | None]] = []
    for bench in _DE_BENCHMARKS:
        for row in store.by_benchmark(bench):
            if row.missing_reason is not None:
                continue
            method, device = _de_method_device(bench, row.format)
            rows_data.append((
                method, row.dataset, device,
                row.median_wall_s, _de_peak_rss_mb(row), row.source.path,
            ))

    if not rows_data:
        return TextBlock(
            "_No DE results — run `accel_de` / `accel_de_nb_glm` "
            "(CPU on any host; GPU on a `scx-bench-gpu` host)._"
        )

    headers = ["Method", "Dataset", "Device", "Median wall", "Peak RSS"]
    rows: list[list[str]] = []
    for method, ds, device, wall, rss, _ in sorted(
        rows_data, key=lambda x: (x[0], x[1], x[2]),
    ):
        rows.append([
            method, SHORT_NAMES.get(ds, ds), device,
            _fmt_time(wall), _fmt_mem(rss),
        ])
    src_path = next((d[5] for d in rows_data if d[5]), None)
    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption="Differential expression — wall time + peak RSS per method × dataset × device",
        source=SourceRef(kind=SourceKind.raw_json, path=src_path,
                         reason="accel_de / accel_de_nb_glm runs[].median_wall_s + peak_rss_mb"),
    )


def de_nb_glm_gpu_table() -> TableBlock | TextBlock:
    """Pseudobulk NB-GLM CPU vs GPU: speedup, kernel throughput, CPU↔GPU agreement.

    One row per dataset, from the ``accel_de_nb_glm__pyscx_gpu`` variant (which
    runs both devices + the pdex_ref anchor). The end-to-end speedup is
    same-machine and **node-core-count dependent** (surfaced, not gated); the
    GPU-fit throughput is the node-independent kernel quantity.
    """
    store = get_store()
    rows: list[list[str]] = []
    src_path = None
    for row in store.by_benchmark("accel_de_nb_glm"):
        if row.missing_reason is not None or not row.format.endswith("_gpu"):
            continue
        src_path = src_path or row.source.path
        # Both wall times come from the GPU variant's runs[].extra (it times both
        # devices on one fixture for an apples-to-apples ratio); gpu_s falls back
        # to the row's top-level median_wall_s, which is the same timed GPU call.
        cpu_s = _de_median_extra(row, "nb_glm_cpu_wall_s")
        gpu_s = _de_median_extra(row, "nb_glm_gpu_wall_s")
        if gpu_s is None:
            gpu_s = row.median_wall_s
        speedup = _de_median_extra(row, "nb_glm_gpu_speedup")
        genes_per_s = _de_median_extra(row, "nb_glm_gpu_fit_genes_per_s")
        rho = _de_median_extra(row, "nb_glm_cpu_gpu_log2fc_spearman")
        max_rel = _de_median_extra(row, "nb_glm_cpu_gpu_max_rel_log2fc")
        ref_ok = _de_median_extra(row, "nb_glm_pdex_ref_concordant")
        route = _de_first_extra(row, "gpu_dispatch_route")
        rows.append([
            SHORT_NAMES.get(row.dataset, row.dataset),
            _fmt_time(cpu_s), _fmt_time(gpu_s),
            f"{speedup:.2f}×" if speedup is not None else "—",
            f"{genes_per_s:,.0f}" if genes_per_s is not None else "—",
            f"{rho:.4f}" if rho is not None else "—",
            f"{max_rel:.2e}" if max_rel is not None else "—",
            ("✓" if ref_ok and ref_ok >= 1.0 else "✗") if ref_ok is not None else "—",
            f"`{route}`" if route else "—",
        ])

    if not rows:
        return TextBlock(
            "_No GPU NB-GLM results — run `accel_de_nb_glm` on a `scx-bench-gpu` "
            "host (`--datasets nb_glm_synth`)._"
        )

    headers = [
        "Dataset", "CPU", "GPU", "Speedup", "GPU-fit genes/s",
        "ρ(log2FC) CPU↔GPU", "max rel log2FC", "vs pdex_ref", "Route",
    ]
    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption="Pseudobulk NB-GLM CPU vs GPU — speedup is same-machine and "
                "node-core-count dependent (surfaced, not gated). It is also "
                "CONSERVATIVE here: nb_glm_synth uses a moderate density (0.3) so "
                "the correctness anchors have signal, which raises the shared "
                "host-aggregation share and understates the GPU-fit win vs the "
                "ultra-sparse regime — see the standalone bench_nb_glm.py --gpu "
                "sweep for that. GPU-fit genes/s is the node-independent kernel "
                "throughput.",
        source=SourceRef(kind=SourceKind.raw_json, path=src_path,
                         reason="accel_de_nb_glm__pyscx_gpu runs[].extra"),
    )


def de_route_correctness_table() -> TableBlock | TextBlock:
    """Deterministic DE route / correctness gate signals as ✓/✗ per variant×dataset.

    Surfaces the binary signals the gate hard-floors (NB-GLM route + CPU↔GPU
    concordance + pdex_ref anchor; per-cell Wilcoxon/pdex_ref route) so a reader
    sees route health at a glance.
    """
    store = get_store()
    rows: list[list[str]] = []
    src_path = None
    for bench in _DE_BENCHMARKS:
        for row in store.by_benchmark(bench):
            if row.missing_reason is not None:
                continue
            method, device = _de_method_device(bench, row.format)
            for sig in _DE_ROUTE_SIGNALS:
                val = _de_median_extra(row, sig)
                if val is None:
                    continue
                src_path = src_path or row.source.path
                rows.append([
                    f"{method} ({device})", SHORT_NAMES.get(row.dataset, row.dataset),
                    sig, "✓" if val >= 1.0 else "✗",
                ])

    if not rows:
        return TextBlock(
            "_No DE route/correctness signals — these populate from the GPU DE "
            "variants (`accel_de` / `accel_de_nb_glm`) on a GPU host._"
        )

    headers = ["Variant", "Dataset", "Signal", "Result"]
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption="DE route + correctness gate signals (✓ = 1.0 = pass). "
                "Hard-gated in thresholds.yaml on the GPU variants.",
        source=SourceRef(kind=SourceKind.raw_json, path=src_path,
                         reason="accel_de / accel_de_nb_glm runs[].extra route signals"),
    )


def cell_eval_correctness_summary_table() -> TableBlock | TextBlock:
    """Cell-eval / arc-bench correctness summary (before performance table).

    Shows per-operation parity status: whether SCX reproduces the reference
    metric values within acceptable thresholds.
    """
    results = load_all_results(benchmark="cell_eval_parity_perf")
    if not results:
        return TextBlock("_No cell_eval_parity_perf results found._")

    headers = ["Dataset", "n_obs", "Operation", "Status", "Notes"]
    rows: list[list[str]] = []

    def _n_obs(r: dict[str, Any]) -> int:
        return r.get("metadata", {}).get("n_obs", 0)

    for r in sorted(results, key=_n_obs):
        dataset = r.get("dataset", "—")
        md = r.get("metadata", {}) or {}
        n_obs = md.get("n_obs", 0)
        for op in md.get("operations", []) or []:
            name = op.get("name", "—")
            if op.get("skipped"):
                status = "_Skipped_"
                notes = op.get("skipped_reason", "")
            elif "scx_error" in op or "ref_error" in op:
                status = "**Error**"
                notes = op.get("scx_error") or op.get("ref_error") or ""
            else:
                status = "Pass"
                notes = ""
            rows.append([dataset, f"{n_obs:,}", name, status, notes])

    return TableBlock(headers=headers, rows=rows,
                      caption="Cell-eval parity correctness summary")


def harmony_lisi_correctness_summary_table() -> TableBlock | TextBlock:
    """Harmony / LISI validation correctness summary.

    Shows per-dataset validation status for Harmony2 (Pearson r vs R harmony)
    and LISI (relative delta). This table precedes the scaling tables.

    Harmony rows are sourced from external ``harmony_integrate`` JSON when
    available, falling back to static diagnostic values with explicit
    ``SourceRef(kind=manual)``.
    """
    blocks_data: list[list[str]] = []
    source_kind = SourceKind.manual
    source_reason = "Harmony rows from static diagnostic; LISI from live data"

    # Try to source Harmony validation from harmony_integrate runs
    harmony_runs = _load_harmony_runs("harmony_integrate")
    harmony_by_ds: dict[str, dict] = {}
    for r in harmony_runs:
        ds = r.get("dataset", "")
        run_data = r.get("run", {})
        if run_data.get("ok") and run_data.get("min_per_pc_r") is not None:
            harmony_by_ds[ds] = {
                "n_obs": r.get("n_obs", 0),
                "min_r": run_data.get("min_per_pc_r"),
            }

    # Static fallback data from diagnostic (pyscx/tests/test_harmony_validation.py)
    _STATIC_HARMONY = [
        ("pbmc_small", 2_700, 0.9986),
        ("cell_lines", 9_478, 0.9789),
        ("hlca_subset", 50_000, 0.9979),
    ]

    if harmony_by_ds:
        source_kind = SourceKind.external_report
        source_reason = "Harmony from harmony_integrate JSON runs; LISI from live data"
        for ds, info in sorted(harmony_by_ds.items(), key=lambda x: x[1].get("n_obs", 0)):
            n_obs = info["n_obs"]
            min_r = info["min_r"]
            status = "Pass" if min_r >= 0.97 else "**Fail**"
            blocks_data.append([
                f"Harmony ({ds})", f"{n_obs:,}", status,
                f"min per-PC r={min_r:.4f}", "",
            ])
    else:
        # Fall back to static values with manual source
        for ds_name, n_obs, min_r in _STATIC_HARMONY:
            blocks_data.append([
                f"Harmony ({ds_name})", f"{n_obs:,}", "Pass",
                f"min per-PC r={min_r:.4f}", "",
            ])

    # LISI from live data
    runs = _load_harmony_runs("lisi")
    by_ds: dict[str, dict[str, dict]] = {}
    for r in runs:
        ds = r["dataset"]
        impl = r["impl"]
        run = r.get("run", {})
        by_ds.setdefault(ds, {})[impl] = {
            "mean_lisi": run.get("mean_lisi"),
        }

    for ds in MAIN_DATASETS:
        if ds not in by_ds:
            continue
        scx = by_ds[ds].get("scx_accel", {})
        ref = by_ds[ds].get("r_lisi", {})
        scx_mean = scx.get("mean_lisi")
        ref_mean = ref.get("mean_lisi")
        if scx_mean is not None and ref_mean is not None and ref_mean != 0:
            rel_delta = abs(scx_mean - ref_mean) / ref_mean * 100
            status = "Pass" if rel_delta < 5.0 else "**Fail**"
            metric = f"|Δ|/R = {rel_delta:.2f}%"
        else:
            status = "_Not run_"
            metric = "—"
        blocks_data.append([
            f"LISI ({SHORT_NAMES.get(ds, ds)})",
            f"{DATASETS[ds].n_obs:,}", status, metric, "",
        ])

    headers = ["Validation", "n_obs", "Status", "Key metric", "Notes"]
    return TableBlock(
        headers=headers, rows=blocks_data,
        caption="Harmony / LISI validation correctness summary",
        source=SourceRef(kind=source_kind, reason=source_reason),
    )


def system_info_table() -> TableBlock | TextBlock:
    """Generate system configuration table from the most recent result."""
    results = load_all_results()
    if not results:
        return TextBlock("_No results found._")

    # Get system info from the most recent result that has it
    sys_info = None
    for r in reversed(results):
        if r.get("system"):
            sys_info = r["system"]
            break

    if not sys_info:
        return TextBlock("_No system information found._")

    libs = sys_info.get("library_versions", {})
    key_libs = ", ".join(
        f"{k} {v}" for k, v in libs.items()
        if k in ("anndata", "scanpy", "zarr", "scipy", "tiledbsoma", "torch", "pyscx")
        and v != "not installed"
    )

    storage = sys_info.get("storage", {})
    storage_desc = storage.get("scratch_fstype", "unknown")
    if storage.get("has_nvme"):
        storage_desc += " (NVMe-backed)"

    headers = ["Property", "Value"]
    rows = [
        ["CPU", str(sys_info.get('cpu', 'unknown'))],
        ["Cores", str(sys_info.get('cpu_cores_physical', 'unknown'))],
        ["RAM", f"{sys_info.get('ram_gb', 'unknown')} GB"],
        ["OS", f"{sys_info.get('os', 'unknown')} ({sys_info.get('arch', '')})"],
        ["Storage", storage_desc],
        ["Python", str(sys_info.get('python_version', 'unknown'))],
        ["Rust", str(sys_info.get('rust_version', 'unknown'))],
        ["Key Libraries", key_libs],
    ]

    return TableBlock(headers=headers, rows=rows,
                      caption="System configuration")


def datasets_table(datasets: list[str] | None = None) -> TableBlock:
    """Generate datasets metadata table."""
    if datasets is None:
        datasets = MAIN_DATASETS

    headers = ["ID", "Name", "Cells", "Genes", "Protocol", "Source", "h5ad Size"]
    rows: list[list[str]] = []

    for ds_name in datasets:
        cfg = DATASETS.get(ds_name)
        if cfg is None:
            continue
        rows.append([
            cfg.id, cfg.name, f"{cfg.n_obs:,}", f"{cfg.n_vars:,}",
            cfg.protocol, cfg.source,
            _fmt_size(cfg.approx_h5ad_mb * 1024 * 1024),
        ])

    return TableBlock(headers=headers, rows=rows,
                      caption="Benchmark datasets")


def cell_eval_parity_perf_table() -> TableBlock | TextBlock:
    """SCX vs cell-eval / arc-bench perturbation-metric performance.

    One row per (dataset, operation). Columns: dataset size, operation,
    SCX median wall, reference (cell-eval / arc-bench) median wall, speedup,
    SCX peak RSS, reference peak RSS. Skipped ops are shown with the reason.
    """
    results = load_all_results(benchmark="cell_eval_parity_perf")
    if not results:
        return TextBlock("_No cell_eval_parity_perf results found._")

    headers = ["Dataset", "n_obs", "Operation", "SCX", "cell-eval",
               "Speedup", "SCX RSS", "ref RSS", "Notes"]
    rows: list[list[str]] = []

    # Order datasets by ascending n_obs where possible
    def _n_obs(r: dict[str, Any]) -> int:
        return r.get("metadata", {}).get("n_obs", 0)

    for r in sorted(results, key=_n_obs):
        dataset = r.get("dataset", "—")
        md = r.get("metadata", {}) or {}
        n_obs = md.get("n_obs", 0)
        for op in md.get("operations", []) or []:
            name = op.get("name", "—")
            if op.get("skipped"):
                note = f"_skipped: {op.get('skipped_reason', '—')}_"
                rows.append([
                    dataset, f"{n_obs:,}", name,
                    "—", "—", "—", "—", "—", note,
                ])
                continue
            if "scx_error" in op or "ref_error" in op:
                err = op.get("scx_error") or op.get("ref_error") or "unknown"
                rows.append([
                    dataset, f"{n_obs:,}", name,
                    "—", "—", "—", "—", "—", f"**err**: {err}",
                ])
                continue

            speedup = op.get("speedup")
            speedup_str = f"{speedup:.1f}x" if speedup is not None else "—"
            rows.append([
                dataset, f"{n_obs:,}", name,
                _fmt_time(op.get('scx_median_s')),
                _fmt_time(op.get('ref_median_s')),
                speedup_str,
                _fmt_mem(op.get('scx_peak_rss_mb')),
                _fmt_mem(op.get('ref_peak_rss_mb')),
                "",
            ])

    return TableBlock(headers=headers, rows=rows,
                      caption="SCX vs cell-eval perturbation-metric performance")


# ---------------------------------------------------------------------------
# Harmony2 + LISI
# ---------------------------------------------------------------------------
#
# Harmony and LISI results live outside `RAW_RESULTS_DIR` — they are produced
# by `benchmarks/scripts/benchmark_harmony.py` / `benchmark_lisi.py` and land
# in `benchmarks/results/harmony/runs/` as a flat set of JSONs.  They are
# loaded by the ``ResultStore`` alongside raw results.


def _load_harmony_runs(bench: str) -> list[dict[str, Any]]:
    """Load JSON results for a harmony benchmark type via the ResultStore."""
    return [
        row.raw
        for row in get_store().by_benchmark(bench)
    ]


def harmony_scaling_table() -> list[Block]:
    """Harmony2 wall-time + peak-RSS scaling across D1–D7.

    Pivot: rows = (impl, device), columns = MAIN_DATASETS. Returns three
    ``TableBlock``s (wall / RSS / iterations). Only includes the canonical
    d=30, K=100 sweep — secondary PC/K sweeps on D4 render separately.
    """
    runs = _load_harmony_runs("harmony_integrate")
    if not runs:
        return [TextBlock("_No harmony_integrate results found._")]

    # (impl, device) -> dataset -> {wall_s, peak_rss_mb, n_iters, ok}
    pivot: dict[tuple[str, str], dict[str, dict]] = {}
    for r in runs:
        params = r.get("params", {})
        if params.get("n_pcs") != 30 or params.get("n_clusters") != 100:
            continue
        run = r.get("run", {})
        key = (r["impl"], r.get("device", "cpu"))
        pivot.setdefault(key, {})[r["dataset"]] = {
            "wall_s": run.get("wall_s"),
            "peak_rss_mb": run.get("peak_rss_mb"),
            "n_iters": run.get("n_iterations"),
            "ok": run.get("ok", False),
        }

    if not pivot:
        return [TextBlock("_No harmony_integrate d=30/K=100 scaling runs found._")]

    header_cells = [f"{SHORT_NAMES[d]} ({DATASETS[d].n_obs:,})" for d in MAIN_DATASETS]
    headers = ["impl / device"] + header_cells

    def _make_table(caption: str, field: str, fmt) -> TableBlock:
        rows: list[list[str]] = []
        for (impl, dev), cols in sorted(pivot.items()):
            cells = [f"`{impl}` / {dev}"]
            for ds in MAIN_DATASETS:
                cell = cols.get(ds)
                if cell and cell.get(field) is not None and cell.get("ok"):
                    cells.append(fmt(cell[field]))
                elif cell and not cell.get("ok"):
                    cells.append("_OOM_")
                else:
                    cells.append("—")
            rows.append(cells)
        return TableBlock(headers=headers, rows=rows, caption=caption)

    return [
        _make_table("Harmony scaling — wall time (s)", "wall_s", lambda v: _fmt_time(v)),
        _make_table("Harmony scaling — peak RSS", "peak_rss_mb", lambda v: _fmt_mem(v)),
        _make_table("Harmony scaling — iterations", "n_iters", lambda v: f"{int(v)}"),
    ]


def lisi_comparison_table() -> TableBlock | TextBlock:
    """LISI: scx-accel vs R lisi on D1–D4.

    Columns: dataset, impl, wall_s, peak_rss_mb, mean_lisi, |Δ|/R_mean.
    """
    runs = _load_harmony_runs("lisi")
    if not runs:
        return TextBlock("_No LISI results found._")

    # Group by dataset so we can compute the relative-delta column.
    by_ds: dict[str, dict[str, dict]] = {}
    for r in runs:
        ds = r["dataset"]
        impl = r["impl"]
        run = r.get("run", {})
        by_ds.setdefault(ds, {})[impl] = {
            "wall_s": run.get("wall_s"),
            "peak_rss_mb": r.get("peak_rss_mb"),
            "mean_lisi": run.get("mean_lisi"),
            "median_lisi": run.get("median_lisi"),
        }

    headers = ["Dataset", "n_obs", "Impl", "wall", "peak RSS",
               "mean LISI", "median", "|Δ| / R mean"]
    rows: list[list[str]] = []
    for ds in MAIN_DATASETS:
        if ds not in by_ds:
            continue
        n_obs = DATASETS[ds].n_obs
        row_dict = by_ds[ds]
        ref = row_dict.get("r_lisi", {}).get("mean_lisi")
        for impl in ("scx_accel", "r_lisi"):
            if impl not in row_dict:
                continue
            m = row_dict[impl]
            scx_m = m.get("mean_lisi")
            if impl == "scx_accel" and ref and scx_m is not None and ref != 0:
                rel = f"{abs(scx_m - ref) / ref * 100:.2f}%"
            else:
                rel = "—"
            rows.append([
                SHORT_NAMES[ds], f"{n_obs:,}", f"`{impl}`",
                _fmt_time(m.get('wall_s')),
                _fmt_mem(m.get('peak_rss_mb')),
                _fmt_num(m.get('mean_lisi'), 3),
                _fmt_num(m.get('median_lisi'), 3),
                rel,
            ])
    return TableBlock(headers=headers, rows=rows,
                      caption="LISI: scx-accel vs R lisi")


def harmony_validation_table() -> TableBlock:
    """Per-PC Pearson r vs R harmony on the three validation fixtures.

    Attempts to source from ``harmony_integrate`` external JSON files
    first.  Falls back to static diagnostic values with explicit
    ``SourceRef(kind=manual)``.
    """
    headers = ["Dataset", "N", "Batches", "d", "K",
               "min per-PC r", "mean per-PC r", "iter (scx / R)"]

    # Try to source from harmony_integrate runs
    runs = _load_harmony_runs("harmony_integrate")
    live_rows: list[list[str]] = []
    for r in runs:
        run_data = r.get("run", {})
        params = r.get("params", {})
        if not run_data.get("ok"):
            continue
        min_r = run_data.get("min_per_pc_r")
        mean_r = run_data.get("mean_per_pc_r")
        if min_r is None or mean_r is None:
            continue
        scx_iters = run_data.get("n_iterations", "—")
        r_iters = run_data.get("r_iterations", "—")
        live_rows.append([
            r.get("dataset", "—"),
            f"{r.get('n_obs', 0):,}",
            str(params.get("n_batches", "—")),
            str(params.get("n_pcs", "—")),
            str(params.get("n_clusters", "—")),
            f"{min_r:.4f}",
            f"{mean_r:.4f}",
            f"{scx_iters} / {r_iters}",
        ])

    if live_rows:
        return TableBlock(
            headers=headers, rows=live_rows,
            caption="Harmony validation — per-PC Pearson r vs R harmony",
            source=SourceRef(kind=SourceKind.external_report,
                             reason="from harmony_integrate JSON runs"),
        )

    # Static fallback from diagnostic (pyscx/tests/test_harmony_validation.py).
    rows = [
        ["pbmc_small (D1)", "2,700", "3", "30", "100",
         "0.9986", "0.9992", "5 / 4"],
        ["cell_lines (smartseq2)", "9,478", "47", "20", "100",
         "0.9789", "0.9885", "10 / 8"],
        ["hlca_subset (tabula)", "50,000", "118", "30", "100",
         "0.9979", "0.9991", "10 / 5"],
    ]
    return TableBlock(
        headers=headers, rows=rows,
        caption="Harmony validation — per-PC Pearson r vs R harmony",
        source=SourceRef(kind=SourceKind.manual,
                         reason="static diagnostic (pyscx/tests/test_harmony_validation.py)"),
    )


# ---------------------------------------------------------------------------
# Community analytical workflows
#
# Four benchmarks that measure SCX against what a scanpy/MuData user actually
# runs, rather than against another storage format. Every metric below lives in
# `runs[].extra` under a suffixed name — `wall_s` / `peak_rss_mb` are reserved
# `add_run` parameters and never reach `extra`, so each arm owns a distinctly
# named RSS key (`_rss_key` in `multimodal_atlas_streaming`) and the timing
# keys carry a `qc_` / `score_` / `total_` prefix.
# ---------------------------------------------------------------------------

#: Human labels for the arms, keyed by the tail after `<benchmark>__`.
_COMMUNITY_ARM_LABELS = {
    # accel_qc_filter
    "pyscx_cpu": "pyscx (backed)",
    "pyscx_inmem": "pyscx (in-memory)",
    "scanpy_cpu": "scanpy",
    # accel_score_genes
    "pyscx_cpu_scanpy": "pyscx control (scanpy parity)",
    "pyscx_cpu_mean": "pyscx mean",
    "pyscx_cpu_zscore": "pyscx zscore",
    # pipeline_ooc_constrained
    "pyscx_16g": "pyscx @ 16 GB",
    "pyscx_32g": "pyscx @ 32 GB",
    "scanpy_16g": "scanpy @ 16 GB",
    "scanpy_32g": "scanpy @ 32 GB",
    # multimodal_atlas_streaming
    "scx_stream": "SCX backed stream",
    "scx_eager_u16": "SCX eager uint16",
    "scx_eager_f32": "SCX eager f32",
    "mudata_h5mu": "MuData eager",
    "mudata_backed": "MuData backed",
    "scx_query_mod": "SCX modality pushdown",
}


def _community_arm(benchmark: str, format_key: str) -> str:
    """Strip the `<benchmark>__` prefix and label the arm."""
    tail = (
        format_key[len(benchmark) + 2:]
        if format_key.startswith(benchmark + "__")
        else format_key
    )
    return _COMMUNITY_ARM_LABELS.get(tail, tail)


def _community_rows(benchmark: str) -> list:
    """Measured rows for a community benchmark, dataset axis derived from data.

    Deriving the dataset axis from the rows rather than a literal list is
    load-bearing: the tier registry keys are `cite_seq_pbmc` / `multiome_pbmc`
    while the result files carry `DatasetConfig.name` — `cite_seq_pbmc_5k` /
    `multiome_pbmc_10k`. A hardcoded list renders those two silently empty.
    """
    store = get_store()
    # `r.runs` and not just `r.missing_reason`: `_try_missing_reason` maps an
    # unrecognised string to `None`, and these modules write typed gaps the
    # `MissingReason` enum does not carry (`no_selective_obs_column`,
    # `scanpy_private_api_drift`). A row with no runs has nothing to render
    # either way, and an all-dash row is what the empty-cell lint exists for.
    return [
        r for r in store.by_benchmark(benchmark)
        if r.missing_reason is None and r.runs
    ]


def _community_src(rows) -> str | None:
    return next((r.source.path for r in rows if r.source and r.source.path), None)


def community_qc_filter_table() -> TableBlock | TextBlock:
    """`accel_qc_filter` — fused QC metrics + atomic filtering vs scanpy.

    Three arms over the identical call sequence. Only `pyscx (backed)` is
    native end to end: `accel.filter_cells` / `filter_genes` delegate to
    `sc.pp.*` for an in-memory scipy `X`, so the in-memory arm is a native QC
    pass followed by a scanpy filter.
    """
    rows_data = _community_rows("accel_qc_filter")
    if not rows_data:
        return TextBlock(
            "_No `accel_qc_filter` results — run it from `run_parallel.py` "
            "on any CPU host._"
        )

    headers = [
        "Dataset", "Arm", "Total", "qc_metrics", "filter_cells",
        "filter_genes", "Peak RSS", "Max abs diff", "Shapes match",
    ]
    rows: list[list[str]] = []
    for row in sorted(rows_data, key=lambda r: (r.dataset, r.format)):
        diff = _de_median_extra(row, "qc_metrics_max_abs_diff")
        match = _de_median_extra(row, "filtered_shape_match_int")
        rows.append([
            SHORT_NAMES.get(row.dataset, row.dataset),
            _community_arm("accel_qc_filter", row.format),
            _fmt_time(_de_median_extra(row, "qc_wall_s")),
            _fmt_time(_de_median_extra(row, "qc_wall_s__qc_metrics")),
            _fmt_time(_de_median_extra(row, "qc_wall_s__filter_cells")),
            _fmt_time(_de_median_extra(row, "qc_wall_s__filter_genes")),
            _fmt_mem(_de_median_extra(row, "qc_peak_rss_mb")),
            "—" if diff is None else f"{diff:.1e}",
            "—" if match is None else ("yes" if match >= 1.0 else "**no**"),
        ])

    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption=(
            "accel_qc_filter — calculate_qc_metrics → filter_cells → "
            "filter_genes, per arm. Parity columns are measured against an "
            "untimed scanpy reference in its own process; the scanpy arm is "
            "that reference and carries none."
        ),
        source=SourceRef(
            kind=SourceKind.raw_json, path=_community_src(rows_data),
            reason="accel_qc_filter runs[].extra qc_wall_s* / qc_peak_rss_mb*",
        ),
    )


def community_score_genes_table() -> TableBlock | TextBlock:
    """`accel_score_genes` — streaming gene-set scoring across three panel sizes.

    The parity arm scores a **backed** `X` that `sc.tl.score_genes` refuses
    outright, and it passes scanpy's own control set via `ctrl_genes=` so the
    comparison isolates the numerical kernel from control-sampling variance.
    """
    rows_data = _community_rows("accel_score_genes")
    if not rows_data:
        return TextBlock(
            "_No `accel_score_genes` results — run it from `run_parallel.py` "
            "on any CPU host._"
        )

    ks = (25, 100, 500)
    headers = (
        ["Dataset", "Arm"]
        + [f"K={k}" for k in ks]
        + ["Peak RSS", "Cells/s (K=100)", "Spearman vs scanpy", "Max abs diff"]
    )
    rows: list[list[str]] = []
    for row in sorted(rows_data, key=lambda r: (r.dataset, r.format)):
        rhos = [_de_median_extra(row, f"score_spearman_vs_scanpy__k{k}") for k in ks]
        diffs = [_de_median_extra(row, f"score_max_abs_diff__k{k}") for k in ks]
        rhos = [v for v in rhos if v is not None]
        diffs = [v for v in diffs if v is not None]
        cps = _de_median_extra(row, "score_cells_per_sec__k100")
        rows.append(
            [
                SHORT_NAMES.get(row.dataset, row.dataset),
                _community_arm("accel_score_genes", row.format),
            ]
            + [_fmt_time(_de_median_extra(row, f"score_wall_s__k{k}")) for k in ks]
            + [
                _fmt_mem(_de_median_extra(row, "score_peak_rss_mb")),
                "—" if cps is None else f"{cps:,.0f}",
                # Worst across the three panels, not the best.
                "—" if not rhos else f"{min(rhos):.4f}",
                "—" if not diffs else f"{max(diffs):.1e}",
            ]
        )

    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption=(
            "accel_score_genes — wall time per signature size. Parity columns "
            "report the worst of the three panels and exist only on the "
            "control-method arm; `mean` and `zscore` are different statistics "
            "with no scanpy counterpart."
        ),
        source=SourceRef(
            kind=SourceKind.raw_json, path=_community_src(rows_data),
            reason="accel_score_genes runs[].extra score_wall_s__k* / score_spearman_vs_scanpy__k*",
        ),
    )


def community_pipeline_ooc_table() -> TableBlock | TextBlock:
    """`pipeline_ooc_constrained` — the laptop test.

    The headline column is **completion under the ceiling**, not wall time. A
    cell that OOMs is a recorded `pipeline_completed_int = 0.0`, not a missing
    result, and reports the *budget* as its peak because SIGKILL leaves the
    sampler no final reading.
    """
    rows_data = _community_rows("pipeline_ooc_constrained")
    if not rows_data:
        return TextBlock(
            "_No `pipeline_ooc_constrained` results — run it from "
            "`run_parallel.py`; the ceiling comes from the SLURM cgroup._"
        )

    headers = [
        "Dataset", "Arm", "Budget", "Completed", "Outcome",
        "Total wall", "Peak RSS", "Clusters",
    ]
    rows: list[list[str]] = []
    for row in sorted(rows_data, key=lambda r: (r.dataset, r.format)):
        done = _de_median_extra(row, "pipeline_completed_int")
        budget = row.metadata.get("budget_gb")
        clusters = _de_median_extra(row, "n_clusters")
        rows.append([
            SHORT_NAMES.get(row.dataset, row.dataset),
            _community_arm("pipeline_ooc_constrained", row.format),
            "—" if budget is None else f"{budget} GB",
            "—" if done is None else ("yes" if done >= 1.0 else "**no**"),
            str(_de_first_extra(row, "outcome_reason") or "—"),
            _fmt_time(_de_median_extra(row, "total_wall_s")),
            _fmt_mem(_de_median_extra(row, "pipeline_peak_rss_mb")),
            # NaN is the module's "not reached" sentinel for a killed run.
            "—" if clusters is None or clusters != clusters else f"{clusters:.0f}",
        ])

    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption=(
            "pipeline_ooc_constrained — nine stages under a fixed memory "
            "ceiling. `Peak RSS` on a non-completing arm is the budget, not an "
            "observation."
        ),
        source=SourceRef(
            kind=SourceKind.raw_json, path=_community_src(rows_data),
            reason="pipeline_ooc_constrained runs[].extra pipeline_completed_int / total_wall_s",
        ),
    )


def community_pipeline_ooc_stage_table() -> TableBlock | TextBlock:
    """Per-stage wall time for `pipeline_ooc_constrained`.

    This is where the engines actually differ: the aggregate hides that SCX
    wins `neighbors` decisively and loses `rank_genes_groups`.
    """
    rows_data = _community_rows("pipeline_ooc_constrained")
    if not rows_data:
        return TextBlock("_No `pipeline_ooc_constrained` stage records._")

    from benchmarks.comprehensive.benchmarks.pipeline_ooc_constrained import STAGES

    # `Notes` is load-bearing, not decoration: an arm killed by the ceiling
    # legitimately has no wall time for the stages it never reached, and the
    # empty-cell lint asks for a reason on any row that is mostly blank. The
    # reason is the outcome, which is also the most useful thing in the row.
    headers = ["Dataset", "Arm"] + [s.replace("_", " ") for s in STAGES] + ["Notes"]
    rows: list[list[str]] = []
    for row in sorted(rows_data, key=lambda r: (r.dataset, r.format)):
        reason = str(_de_first_extra(row, "outcome_reason") or "—")
        failed = _de_first_extra(row, "failed_stage")
        note = f"{reason} @ {failed}" if reason != "completed" and failed else reason
        rows.append(
            [
                SHORT_NAMES.get(row.dataset, row.dataset),
                _community_arm("pipeline_ooc_constrained", row.format),
            ]
            + [_fmt_time(_de_median_extra(row, f"stage_wall_s__{s}")) for s in STAGES]
            + [note]
        )

    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption=(
            "pipeline_ooc_constrained — per-stage wall time. A killed arm "
            "shows the stages it completed before the ceiling and `—` after."
        ),
        source=SourceRef(
            kind=SourceKind.raw_json, path=_community_src(rows_data),
            reason="pipeline_ooc_constrained runs[].extra stage_wall_s__<stage>",
        ),
    )


def community_multimodal_atlas_table() -> TableBlock | TextBlock:
    """`multimodal_atlas_streaming` — atlas-scale multiome / CITE-seq reads.

    Each arm owns a distinctly named RSS key, so the peak column is assembled
    per arm rather than read from one shared metric.
    """
    rows_data = _community_rows("multimodal_atlas_streaming")
    if not rows_data:
        return TextBlock(
            "_No `multimodal_atlas_streaming` results — the two atlas fixtures "
            "are staged by `benchmarks/scripts/build_multimodal_atlas.py`._"
        )

    from benchmarks.comprehensive.benchmarks.multimodal_atlas_streaming import (
        _ARMS, _rss_key,
    )

    headers = [
        "Dataset", "Arm", "Wall", "Peak RSS", "Cells/s",
        "f32/u16 peak ratio", "Sums match",
    ]
    rows: list[list[str]] = []
    for row in sorted(rows_data, key=lambda r: (r.dataset, r.format)):
        # The arm's *mode* is not its format tail — `mudata_h5mu` is the mode
        # `mudata_eager`, `scx_query_mod` is `scx_query` — and `_rss_key`
        # raises on anything else, so go through `_ARMS` rather than guessing.
        arm = _ARMS.get(row.format) or {}
        rss_key = _rss_key(arm["mode"]) if arm.get("mode") else None
        rss = _de_median_extra(row, rss_key) if rss_key else None
        if rss is None:
            rss = _de_peak_rss_mb(row)
        cps = _de_median_extra(row, "stream_cells_per_sec")
        ratio = _de_median_extra(row, "eager_peak_rss_ratio_f32_over_u16")
        match = _de_median_extra(row, "modality_sums_match_int")
        rows.append([
            row.dataset,
            _community_arm("multimodal_atlas_streaming", row.format),
            _fmt_time(row.median_wall_s),
            _fmt_mem(rss),
            "—" if cps is None else f"{cps:,.0f}",
            "—" if ratio is None else f"{ratio:.3f}x",
            "—" if match is None else ("yes" if match >= 1.0 else "**no**"),
        ])

    return TableBlock(
        headers=headers, rows=rows, wide=True,
        caption=(
            "multimodal_atlas_streaming — six arms over the same file: "
            "backed streaming, eager f32, eager uint16, MuData eager, MuData "
            "backed, and a modality-scoped query. The f32/u16 ratio is "
            "measured on the narrowing arm, which runs its own f32 control in "
            "a second process; `Sums match` compares every modality's "
            "float64-accumulated total across arms."
        ),
        source=SourceRef(
            kind=SourceKind.raw_json, path=_community_src(rows_data),
            reason="multimodal_atlas_streaming runs[].extra per-arm *_peak_rss_mb",
        ),
    )


def generate_all_tables() -> dict[str, Block | list[Block]]:
    """Generate all summary tables, returning a dict of table_name -> block(s).

    Most entries are a single ``Block``; ``parallel_scaling`` and
    ``parallel_write_scaling`` return ``list[Block]`` because they
    produce one sub-table per dataset/mode combination.
    """
    return {
        "system_info": system_info_table(),
        "community_qc_filter": community_qc_filter_table(),
        "community_score_genes": community_score_genes_table(),
        "community_pipeline_ooc": community_pipeline_ooc_table(),
        "community_pipeline_ooc_stages": community_pipeline_ooc_stage_table(),
        "community_multimodal_atlas": community_multimodal_atlas_table(),
        "datasets": datasets_table(),
        "compression": compression_table(),
        "compression_ratio": compression_ratio_table(),
        "write_speed": write_speed_table(),
        "write_conversion": write_conversion_table(),
        "write_only": write_only_table(),
        "scx_parallel_write_callout": scx_parallel_write_callout_table(),
        "read_speed": read_speed_table(),
        "read_selective": read_selective_table(),
        "parallel_scaling": parallel_scaling_table(),
        "parallel_write_scaling": parallel_write_scaling_table(),
        "memory": memory_table(),
        "memory_by_mode": memory_by_mode_tables(),
        "fragment_ops": fragment_ops_table(),
        "capability_matrix": capability_matrix_table(),
        "cloud_filtered": cloud_filtered_table(),
        "gcp_matrix": gcp_matrix_table(),
        "cloud_reader_vs_pull": cloud_reader_vs_pull_table(),
        "cost_model": cost_model_table(),
        "ml_loader": ml_loader_table(),
        "multimodal_compression": multimodal_compression_table(),
        "multimodal_compression_ratio": multimodal_compression_ratio_table(),
        "multimodal_training": multimodal_training_table(),
        "multimodal_training_ttfb": multimodal_training_ttfb_table(),
        "bench_csc_dispatch": bench_csc_dispatch_table(),
        "correctness_summary": correctness_table(),
        "correctness_detail": correctness_detail_table(),
        "correctness_detail_all": correctness_detail_all_datasets(),
        "correctness_dataset_summary": correctness_dataset_summary_table(),
        "pipeline_agreement": pipeline_agreement_table(),
        "accelerator_parity": accelerator_parity_table(),
        "accelerator_gpu_vs_rapids": accelerator_gpu_vs_rapids_comparison_table(),
        "cell_eval_correctness": cell_eval_correctness_summary_table(),
        "cell_eval_parity_perf": cell_eval_parity_perf_table(),
        "harmony_lisi_correctness": harmony_lisi_correctness_summary_table(),
        "harmony_scaling": harmony_scaling_table(),
        "lisi_comparison": lisi_comparison_table(),
        "harmony_validation": harmony_validation_table(),
    }
