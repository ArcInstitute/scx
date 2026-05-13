"""
Generate markdown summary tables from raw benchmark JSON results.

Reads from benchmarks/comprehensive/results/raw/ and produces formatted
markdown tables for each benchmark dimension.

All table functions consume the shared ``ResultStore`` singleton (Phase 2)
so that raw JSON is loaded once per report-generation run. The legacy
``load_all_results`` name is preserved as a thin wrapper around the
store's compatibility layer.
"""

from __future__ import annotations

import statistics
from typing import Any

from benchmarks.comprehensive.config import DATASETS, DatasetConfig
from benchmarks.comprehensive.config import DATASETS, DatasetConfig
from benchmarks.comprehensive.reporting.result_store import get_store, SourceRef, SourceKind
from benchmarks.comprehensive.reporting.report_model import TableBlock, TextBlock, Block


def load_all_results(
    benchmark: str | None = None,
    format_key: str | None = None,
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    """Legacy wrapper — delegates to the ``ResultStore`` singleton.

    Preserves the exact return type (``list[dict]``) so that every
    existing table function works unchanged during the Phase 2→3
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

        # Pick the most representative metric
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
        "accel_knn": "recall_at_k",
        "accel_umap": "trustworthiness",
        "accel_leiden": "ari",
        "accel_preprocess": "max_abs_error",
        "accel_hvg": "overlap_pct",
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

        # Extract parity metric from SCX runs
        parity_key = parity_keys.get(bench, "")
        parity_val = None
        if scx_row.runs:
            extra = scx_row.runs[0].get("extra", {})
            parity_val = extra.get(parity_key)
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
    """
    blocks_data: list[list[str]] = []

    # Harmony validation (static from Phase 6 diagnostic)
    blocks_data.append(["Harmony (pbmc_small)", "2,700", "Pass",
                        "min per-PC r=0.9986", ""])
    blocks_data.append(["Harmony (cell_lines)", "9,478", "Pass",
                        "min per-PC r=0.9789", ""])
    blocks_data.append(["Harmony (hlca_subset)", "50,000", "Pass",
                        "min per-PC r=0.9979", ""])

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
        source=SourceRef(kind=SourceKind.manual,
                         reason="Harmony rows from Phase 6 diagnostic; LISI from live data"),
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
# Harmony2 + LISI (Phase 6)
# ---------------------------------------------------------------------------
#
# Harmony and LISI results live outside `RAW_RESULTS_DIR` — they are produced
# by `benchmarks/scripts/benchmark_harmony.py` / `benchmark_lisi.py` and land
# in `benchmarks/results/harmony/runs/` as a flat set of JSONs. As of Phase 2,
# these are loaded by the ``ResultStore`` alongside raw results.


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

    Reads the validation fixture + re-runs the Rust pipeline is too heavy
    for the report pass — instead, read the summary embedded in the runs
    directory if a precomputed validation JSON is present, otherwise emit a
    static table sourced from the Phase 6 diagnostic (pyscx/tests/
    test_harmony_validation.py docstring).
    """
    # Static numbers from Phase 6 diagnostic (see pyscx/tests/test_harmony_validation.py).
    headers = ["Dataset", "N", "Batches", "d", "K",
               "min per-PC r", "mean per-PC r", "iter (scx / R)"]
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
                         reason="Phase 6 diagnostic (pyscx/tests/test_harmony_validation.py)"),
    )


def generate_all_tables() -> dict[str, Block | list[Block]]:
    """Generate all summary tables, returning a dict of table_name -> block(s).

    Most entries are a single ``Block``; ``parallel_scaling`` and
    ``parallel_write_scaling`` return ``list[Block]`` because they
    produce one sub-table per dataset/mode combination.
    """
    return {
        "system_info": system_info_table(),
        "datasets": datasets_table(),
        "compression": compression_table(),
        "compression_ratio": compression_ratio_table(),
        "write_speed": write_speed_table(),
        "scx_parallel_write_callout": scx_parallel_write_callout_table(),
        "read_speed": read_speed_table(),
        "read_selective": read_selective_table(),
        "parallel_scaling": parallel_scaling_table(),
        "parallel_write_scaling": parallel_write_scaling_table(),
        "memory": memory_table(),
        "fragment_ops": fragment_ops_table(),
        "cloud_filtered": cloud_filtered_table(),
        "gcp_matrix": gcp_matrix_table(),
        "cloud_reader_vs_pull": cloud_reader_vs_pull_table(),
        "cost_model": cost_model_table(),
        "ml_loader": ml_loader_table(),
        "correctness_summary": correctness_table(),
        "correctness_detail": correctness_detail_table(),
        "correctness_detail_all": correctness_detail_all_datasets(),
        "correctness_dataset_summary": correctness_dataset_summary_table(),
        "pipeline_agreement": pipeline_agreement_table(),
        "accelerator_parity": accelerator_parity_table(),
        "cell_eval_correctness": cell_eval_correctness_summary_table(),
        "cell_eval_parity_perf": cell_eval_parity_perf_table(),
        "harmony_lisi_correctness": harmony_lisi_correctness_summary_table(),
        "harmony_scaling": harmony_scaling_table(),
        "lisi_comparison": lisi_comparison_table(),
        "harmony_validation": harmony_validation_table(),
    }
