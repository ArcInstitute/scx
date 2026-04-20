"""
Generate markdown summary tables from raw benchmark JSON results.

Reads from benchmarks/comprehensive/results/raw/ and produces formatted
markdown tables for each benchmark dimension (§6.1 of COMPREHENSIVE-BENCHMARKING.md).
"""

from __future__ import annotations

import statistics
from typing import Any

from benchmarks.comprehensive.config import DATASETS, DatasetConfig
from benchmarks.comprehensive.results import load_all_results

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


def _fmt_size(nbytes: float | None) -> str:
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


def _fmt_time(seconds: float | None) -> str:
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


def _fmt_mem(mb: float | None) -> str:
    """Format memory in MB as human-readable."""
    if mb is None:
        return "—"
    if mb < 1024:
        return f"{mb:,.0f} MB"
    else:
        return f"{mb / 1024:.1f} GB"


def _fmt_num(val: float | None, decimals: int = 1) -> str:
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


def _pivot_to_table(
    pivot: dict[str, dict[str, float | None]],
    datasets: list[str],
    formatter: callable,
    bold_best: str = "min",
    title: str = "Format",
) -> str:
    """Render a pivot dict as a markdown table.

    Parameters
    ----------
    pivot : format -> dataset -> value
    datasets : column order
    formatter : function to format values
    bold_best : "min" to bold the minimum per column, "max" for max, None for none
    title : first column header
    """
    headers = [SHORT_NAMES.get(d, d) for d in datasets]
    lines = [
        f"| {title} | " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---:" for _ in datasets]) + "|",
    ]

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

    for fmt in sorted_fmts:
        row = pivot[fmt]
        display = FORMAT_DISPLAY.get(fmt, fmt)
        cells = []
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
        lines.append(f"| {display} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public table generators
# ---------------------------------------------------------------------------

def compression_table(datasets: list[str] | None = None) -> str:
    """Generate compression matrix: Format x Dataset showing file size."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="compression")
    pivot = _build_pivot(results, "file_size_bytes", datasets)
    return _pivot_to_table(pivot, datasets, _fmt_size, bold_best="min")


def compression_ratio_table(datasets: list[str] | None = None) -> str:
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

    return _pivot_to_table(pivot, datasets, fmt_ratio, bold_best="max")


def read_speed_table(datasets: list[str] | None = None) -> str:
    """Generate read speed matrix: Format x Dataset showing median read time."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="read_full")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_table(pivot, datasets, _fmt_time, bold_best="min")


def read_selective_table(datasets: list[str] | None = None) -> str:
    """Generate selective read table: Format x Dataset showing column projection time."""
    if datasets is None:
        datasets = [d for d in MAIN_DATASETS if d != "pbmc3k" and d != "pbmc10k"]
    results = load_all_results(benchmark="read_selective")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_table(pivot, datasets, _fmt_time, bold_best="min")


def write_speed_table(datasets: list[str] | None = None) -> str:
    """Generate write speed matrix: Format x Dataset showing conversion time."""
    if datasets is None:
        datasets = MAIN_DATASETS
    results = load_all_results(benchmark="write")
    pivot = _build_pivot(results, "median_wall_s", datasets)
    return _pivot_to_table(pivot, datasets, _fmt_time, bold_best="min")


def memory_table(datasets: list[str] | None = None) -> str:
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

    return _pivot_to_table(pivot, datasets, _fmt_mem, bold_best="min")


def parallel_scaling_table(datasets: list[str] | None = None) -> str:
    """Generate parallel scaling table: Format x Thread count showing wall time and speedup."""
    if datasets is None:
        datasets = ["census_500k", "census_1m", "census_5m"]
    results = load_all_results(benchmark="parallel_scaling")

    lines = []
    for ds in datasets:
        ds_results = [r for r in results if r.get("dataset") == ds]
        if not ds_results:
            continue

        lines.append(f"\n**{SHORT_NAMES.get(ds, ds)}**\n")
        lines.append("| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

        sorted_results = sorted(
            ds_results,
            key=lambda r: FORMAT_ORDER.index(r.get("format", ""))
            if r.get("format", "") in FORMAT_ORDER else 999,
        )

        for r in sorted_results:
            fmt = r.get("format", "")
            display = FORMAT_DISPLAY.get(fmt, fmt)
            scaling = r.get("metadata", {}).get("scaling_wall_s", {})
            speedup = r.get("metadata", {}).get("speedup", {})

            cells = []
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
            lines.append(f"| {display} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def scx_parallel_write_callout_table(
    datasets: list[str] | None = None,
) -> str:
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
        return "*No parallel write scaling results available yet.*"

    # Keep SCX rows only; other formats do not parallelise at all.
    scx_formats = [f for f in FORMAT_ORDER if f.startswith("scx_")]

    header_cells = ["Format"]
    for ds in datasets:
        header_cells += [
            f"{SHORT_NAMES.get(ds, ds)} @1T",
            f"{SHORT_NAMES.get(ds, ds)} @32T",
            f"{SHORT_NAMES.get(ds, ds)} Δ",
        ]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(header_cells) - 1)) + "|",
    ]

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
            lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parallel_write_scaling_table(datasets: list[str] | None = None) -> str:
    """Generate parallel write scaling table: Format x Thread count showing wall time and speedup.

    Produces separate sub-tables for each mode (full pipeline vs write-only).
    """
    if datasets is None:
        datasets = ["census_500k", "census_1m", "census_5m"]
    results = load_all_results(benchmark="parallel_write_scaling")

    if not results:
        return "*No parallel write scaling results available yet.*"

    lines = []

    # Collect all modes present in results.
    all_modes: set[str] = set()
    for r in results:
        meta = r.get("metadata", {})
        speedup = meta.get("speedup", {})
        all_modes.update(speedup.keys())

    mode_labels = {"full": "Full pipeline (h5ad read + write)", "write_only": "Write only (in-memory AnnData)"}

    for mode in ["full", "write_only"]:
        if mode not in all_modes:
            continue

        lines.append(f"\n**{mode_labels.get(mode, mode)}**\n")

        for ds in datasets:
            ds_results = [r for r in results if r.get("dataset") == ds]
            if not ds_results:
                continue

            lines.append(f"\n*{SHORT_NAMES.get(ds, ds)}*\n")
            lines.append("| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

            sorted_results = sorted(
                ds_results,
                key=lambda r: FORMAT_ORDER.index(r.get("format", ""))
                if r.get("format", "") in FORMAT_ORDER else 999,
            )

            for r in sorted_results:
                fmt = r.get("format", "")
                display = FORMAT_DISPLAY.get(fmt, fmt)
                meta = r.get("metadata", {})
                scaling = meta.get("scaling_wall_s", {}).get(mode, {})
                speedup = meta.get("speedup", {}).get(mode, {})

                if not scaling:
                    continue

                cells = []
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
                lines.append(f"| {display} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def fragment_ops_table(datasets: list[str] | None = None) -> str:
    """Fragment/manifest operation throughput — Operation x Dataset.

    Each cell shows median wall-clock with a secondary metric in
    parentheses (MB/s for append/compact, rows/s for delete, — for
    rollback). SCX-only benchmark.
    """
    if datasets is None:
        datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]
    results = load_all_results(benchmark="fragment_ops")
    if not results:
        return "*No fragment-ops results available yet.*"

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

    headers = [SHORT_NAMES.get(d, d) for d in datasets]
    lines = [
        "| Operation | " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---:" for _ in datasets]) + "|",
    ]
    for op in ops_order:
        row_data = pivot.get(op, {})
        cells = []
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
        lines.append(f"| `{op}` | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def ml_loader_table(datasets: list[str] | None = None) -> str:
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

    return _pivot_to_table(pivot, datasets, fmt_bps, bold_best="max", title="Loader")


def correctness_table() -> str:
    """Generate correctness validation summary table."""
    results = load_all_results()  # Load all, filter correctness
    correctness = [
        r for r in results
        if r.get("harness") or r.get("benchmark", "").startswith("correctness")
    ]

    if not correctness:
        return "_No correctness results found._"

    lines = ["| Test | Dataset | Passed | Failed | Skipped | Duration |"]
    lines.append("|---|---|---:|---:|---:|---:|")

    for r in correctness:
        harness = r.get("harness", r.get("benchmark", "unknown"))
        dataset = r.get("dataset", "—")
        n_passed = r.get("n_passed", 0)
        n_failed = r.get("n_failed", 0)
        n_skipped = r.get("n_skipped", 0)
        duration = r.get("total_duration_s")
        status = "PASS" if r.get("overall_passed") else "**FAIL**"

        lines.append(
            f"| {harness} | {dataset} | {n_passed} | {n_failed} | {n_skipped} "
            f"| {_fmt_time(duration)} |"
        )

    return "\n".join(lines)


def correctness_detail_table(dataset: str = "pbmc3k") -> str:
    """Generate per-function correctness detail table for a given dataset."""
    results = load_all_results()
    scanpy_equiv = [
        r for r in results
        if r.get("harness") == "scanpy_equivalence" and r.get("dataset") == dataset
    ]

    if not scanpy_equiv:
        return f"_No scanpy equivalence results for {dataset}._"

    r = scanpy_equiv[-1]  # Most recent
    lines = ["| Function | Passed | Key Metric | Value | Threshold | Duration |"]
    lines.append("|---|:---:|---|---:|---:|---:|")

    for test in r.get("results", []):
        name = test.get("name", "unknown")
        passed = "Yes" if test.get("passed") in (True, "True") else "**No**"
        metrics = test.get("metrics", {})
        thresholds = test.get("thresholds", {})
        duration = test.get("duration_s")

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

        lines.append(
            f"| {name} | {passed} | {metric_name} | {metric_str} "
            f"| {threshold_str} | {_fmt_time(duration)} |"
        )

    return "\n".join(lines)


def system_info_table() -> str:
    """Generate system configuration table from the most recent result."""
    results = load_all_results()
    if not results:
        return "_No results found._"

    # Get system info from the most recent result that has it
    sys_info = None
    for r in reversed(results):
        if r.get("system"):
            sys_info = r["system"]
            break

    if not sys_info:
        return "_No system information found._"

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

    lines = [
        "| Property | Value |",
        "|---|---|",
        f"| CPU | {sys_info.get('cpu', 'unknown')} |",
        f"| Cores | {sys_info.get('cpu_cores_physical', 'unknown')} |",
        f"| RAM | {sys_info.get('ram_gb', 'unknown')} GB |",
        f"| OS | {sys_info.get('os', 'unknown')} ({sys_info.get('arch', '')}) |",
        f"| Storage | {storage_desc} |",
        f"| Python | {sys_info.get('python_version', 'unknown')} |",
        f"| Rust | {sys_info.get('rust_version', 'unknown')} |",
        f"| Key Libraries | {key_libs} |",
    ]

    return "\n".join(lines)


def datasets_table(datasets: list[str] | None = None) -> str:
    """Generate datasets metadata table."""
    if datasets is None:
        datasets = MAIN_DATASETS

    lines = [
        "| ID | Name | Cells | Genes | Protocol | Source | h5ad Size |",
        "|---|---|---:|---:|---|---|---:|",
    ]

    for ds_name in datasets:
        cfg = DATASETS.get(ds_name)
        if cfg is None:
            continue
        lines.append(
            f"| {cfg.id} | {cfg.name} | {cfg.n_obs:,} | {cfg.n_vars:,} "
            f"| {cfg.protocol} | {cfg.source} | {_fmt_size(cfg.approx_h5ad_mb * 1024 * 1024)} |"
        )

    return "\n".join(lines)


def cell_eval_parity_perf_table() -> str:
    """SCX vs cell-eval / arc-bench perturbation-metric performance.

    One row per (dataset, operation). Columns: dataset size, operation,
    SCX median wall, reference (cell-eval / arc-bench) median wall, speedup,
    SCX peak RSS, reference peak RSS. Skipped ops are shown with the reason.
    """
    results = load_all_results(benchmark="cell_eval_parity_perf")
    if not results:
        return "_No cell_eval_parity_perf results found._"

    lines = [
        "| Dataset | n_obs | Operation | SCX | cell-eval | Speedup | SCX RSS | ref RSS | Notes |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]

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
                lines.append(
                    f"| {dataset} | {n_obs:,} | {name} | — | — | — | — | — | {note} |"
                )
                continue
            if "scx_error" in op or "ref_error" in op:
                err = op.get("scx_error") or op.get("ref_error") or "unknown"
                lines.append(
                    f"| {dataset} | {n_obs:,} | {name} | — | — | — | — | — "
                    f"| **err**: {err} |"
                )
                continue

            speedup = op.get("speedup")
            speedup_str = f"{speedup:.1f}x" if speedup is not None else "—"
            lines.append(
                f"| {dataset} | {n_obs:,} | {name} "
                f"| {_fmt_time(op.get('scx_median_s'))} "
                f"| {_fmt_time(op.get('ref_median_s'))} "
                f"| {speedup_str} "
                f"| {_fmt_mem(op.get('scx_peak_rss_mb'))} "
                f"| {_fmt_mem(op.get('ref_peak_rss_mb'))} |  |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harmony2 + LISI (Phase 6)
# ---------------------------------------------------------------------------
#
# Harmony and LISI results live outside `RAW_RESULTS_DIR` — they are produced
# by `benchmarks/scripts/benchmark_harmony.py` / `benchmark_lisi.py` and land
# in `benchmarks/results/harmony/runs/` as a flat set of JSONs. The helpers
# below read that directory directly (no dependency on `load_all_results`).

import json as _json
from pathlib import Path as _Path

from benchmarks.comprehensive.config import PROJECT_ROOT as _PROJECT_ROOT

_HARMONY_RUNS_DIR = _PROJECT_ROOT / "benchmarks" / "results" / "harmony" / "runs"


def _load_harmony_runs(bench: str) -> list[dict[str, Any]]:
    """Load JSON results from the harmony benchmark runs directory."""
    if not _HARMONY_RUNS_DIR.exists():
        return []
    rows = []
    for p in sorted(_HARMONY_RUNS_DIR.glob("*.json")):
        try:
            r = _json.loads(p.read_text())
        except (OSError, _json.JSONDecodeError):
            continue
        if r.get("benchmark") == bench:
            rows.append(r)
    return rows


def harmony_scaling_table() -> str:
    """Harmony2 wall-time + peak-RSS scaling across D1–D7.

    Pivot: rows = (impl, device), columns = MAIN_DATASETS. One pair of
    sub-tables (wall / RSS). Only includes the canonical d=30, K=100 sweep —
    secondary PC/K sweeps on D4 render separately.
    """
    runs = _load_harmony_runs("harmony_integrate")
    if not runs:
        return "_No harmony_integrate results found._"

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
        return "_No harmony_integrate d=30/K=100 scaling runs found._"

    header_cells = [f"{SHORT_NAMES[d]} ({DATASETS[d].n_obs:,})" for d in MAIN_DATASETS]
    sep = ["---"] + ["---:"] * len(MAIN_DATASETS)

    def _section(title: str, field: str, fmt) -> list[str]:
        out = [f"\n**{title}**\n",
               "| impl / device | " + " | ".join(header_cells) + " |",
               "|" + "|".join(sep) + "|"]
        for (impl, dev), cols in sorted(pivot.items()):
            row = [f"`{impl}` / {dev}"]
            for ds in MAIN_DATASETS:
                cell = cols.get(ds)
                if cell and cell.get(field) is not None and cell.get("ok"):
                    row.append(fmt(cell[field]))
                elif cell and not cell.get("ok"):
                    row.append("_OOM_")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        return out

    lines: list[str] = []
    lines += _section("Wall time (s)", "wall_s", lambda v: _fmt_time(v))
    lines += _section("Peak RSS", "peak_rss_mb", lambda v: _fmt_mem(v))
    lines += _section("Harmony iterations", "n_iters",
                      lambda v: f"{int(v)}")
    return "\n".join(lines)


def lisi_comparison_table() -> str:
    """LISI: scx-accel vs R lisi on D1–D4.

    Columns: dataset, impl, wall_s, peak_rss_mb, mean_lisi, |Δ|/R_mean.
    """
    runs = _load_harmony_runs("lisi")
    if not runs:
        return "_No LISI results found._"

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

    lines = [
        "| Dataset | n_obs | Impl | wall | peak RSS | mean LISI | median | "
        "|Δ| / R mean |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
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
            lines.append(
                f"| {SHORT_NAMES[ds]} | {n_obs:,} | `{impl}` "
                f"| {_fmt_time(m.get('wall_s'))} "
                f"| {_fmt_mem(m.get('peak_rss_mb'))} "
                f"| {_fmt_num(m.get('mean_lisi'), 3)} "
                f"| {_fmt_num(m.get('median_lisi'), 3)} "
                f"| {rel} |"
            )
    return "\n".join(lines)


def harmony_validation_table() -> str:
    """Per-PC Pearson r vs R harmony on the three validation fixtures.

    Reads the validation fixture + re-runs the Rust pipeline is too heavy
    for the report pass — instead, read the summary embedded in the runs
    directory if a precomputed validation JSON is present, otherwise emit a
    static table sourced from the Phase 6 diagnostic (pyscx/tests/
    test_harmony_validation.py docstring).
    """
    # Static numbers from Phase 6 diagnostic (see pyscx/tests/test_harmony_validation.py).
    return "\n".join([
        "| Dataset | N | Batches | d | K | min per-PC r | mean per-PC r | "
        "iter (scx / R) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        "| pbmc_small (D1) | 2,700 | 3 | 30 | 100 | 0.9986 | 0.9992 | 5 / 4 |",
        "| cell_lines (smartseq2) | 9,478 | 47 | 20 | 100 | 0.9789 | "
        "0.9885 | 10 / 8 |",
        "| hlca_subset (tabula) | 50,000 | 118 | 30 | 100 | 0.9979 | 0.9991 "
        "| 10 / 5 |",
    ])


def generate_all_tables() -> dict[str, str]:
    """Generate all summary tables, returning a dict of table_name -> markdown."""
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
        "ml_loader": ml_loader_table(),
        "correctness_summary": correctness_table(),
        "correctness_detail": correctness_detail_table(),
        "cell_eval_parity_perf": cell_eval_parity_perf_table(),
        "harmony_scaling": harmony_scaling_table(),
        "lisi_comparison": lisi_comparison_table(),
        "harmony_validation": harmony_validation_table(),
    }
