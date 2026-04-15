"""
Generate publication-quality plots from raw benchmark JSON results.

All charts from §6.2 of COMPREHENSIVE-BENCHMARKING.md. Outputs PNG + PDF to
benchmarks/comprehensive/results/reports/figures/.

Requires: matplotlib, seaborn, numpy.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

from benchmarks.comprehensive.config import (
    DATASETS,
    FIGURES_DIR,
)
from benchmarks.comprehensive.results import load_all_results

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

# Datasets for the main tables (exclude lognorm variants)
MAIN_DATASETS = [
    "pbmc3k", "pbmc10k", "smartseq2", "tabula_sapiens_100k",
    "census_500k", "census_1m", "census_5m",
]

SHORT_NAMES = {
    "pbmc3k": "pbmc3k\n(2.7K)",
    "pbmc10k": "pbmc10k\n(12K)",
    "smartseq2": "smartseq2\n(50K)",
    "tabula_sapiens_100k": "tabula_100k\n(100K)",
    "census_500k": "census_500k\n(500K)",
    "census_1m": "census_1m\n(1M)",
    "census_5m": "census_5m\n(5M)",
}

# Color palette: SCX variants in blues, competitors in grays/oranges
FORMAT_COLORS = {
    "scx_auto": "#1f77b4",
    "scx_scx1": "#2ca02c",
    "scx_zstd": "#17becf",
    "scx_lz4": "#9467bd",
    "scx_pcodec": "#7f7f7f",
    "scx_none": "#d3d3d3",
    "zarr_zstd": "#ff7f0e",
    "zarr_lz4": "#ffbb78",
    "tiledb_soma": "#d62728",
    "h5ad_gzip": "#8c564b",
    "h5ad_lzf": "#c49c94",
    "h5ad_none": "#e377c2",
}

FORMAT_LABELS = {
    "scx_auto": "SCX auto",
    "scx_scx1": "SCX scx1",
    "scx_zstd": "SCX zstd",
    "scx_lz4": "SCX lz4",
    "scx_pcodec": "SCX pcodec",
    "scx_none": "SCX none",
    "zarr_zstd": "Zarr zstd",
    "zarr_lz4": "Zarr lz4",
    "tiledb_soma": "TileDB-SOMA",
    "h5ad_gzip": "h5ad gzip",
    "h5ad_lzf": "h5ad lzf",
    "h5ad_none": "h5ad none",
}

FORMAT_ORDER = [
    "scx_auto", "scx_scx1", "scx_zstd", "scx_lz4", "scx_pcodec", "scx_none",
    "zarr_zstd", "zarr_lz4", "tiledb_soma",
    "h5ad_gzip", "h5ad_lzf", "h5ad_none",
]

# Compact subset for busy plots
KEY_FORMATS = [
    "scx_auto", "scx_zstd", "scx_lz4",
    "zarr_zstd", "zarr_lz4",
    "tiledb_soma", "h5ad_gzip", "h5ad_none",
]


def _setup_style():
    """Apply consistent plot styling."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    })


def _save_fig(fig: plt.Figure, name: str, output_dir: Path | None = None):
    """Save figure as PNG and PDF."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {name}.png and {name}.pdf")


def _build_pivot(
    results: list[dict],
    value_key: str,
    datasets: list[str],
) -> dict[str, dict[str, float | None]]:
    """Pivot results into format -> dataset -> value."""
    pivot: dict[str, dict[str, float | None]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        val: Any = r
        for part in value_key.split("."):
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = None
                break
        pivot.setdefault(fmt, {})[ds] = val
    return pivot


# ---------------------------------------------------------------------------
# 1. Compression bar chart
# ---------------------------------------------------------------------------

def plot_compression(output_dir: Path | None = None):
    """Grouped bar chart: file size per format, grouped by dataset."""
    _setup_style()
    results = load_all_results(benchmark="compression")
    datasets = MAIN_DATASETS
    formats = KEY_FORMATS

    pivot = _build_pivot(results, "file_size_bytes", datasets)

    # Normalize to h5ad_none (uncompressed baseline)
    baseline: dict[str, float] = {}
    for ds in datasets:
        val = pivot.get("h5ad_none", {}).get(ds)
        baseline[ds] = val if val else 1.0

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    n_fmts = len(formats)
    width = 0.8 / n_fmts

    for i, fmt in enumerate(formats):
        vals = []
        for ds in datasets:
            raw = pivot.get(fmt, {}).get(ds)
            if raw is not None:
                vals.append(raw / baseline[ds])
            else:
                vals.append(0)
        offset = (i - n_fmts / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=FORMAT_LABELS.get(fmt, fmt),
            color=FORMAT_COLORS.get(fmt, "#999999"),
        )

    ax.set_ylabel("File size (normalized to h5ad uncompressed)")
    ax.set_title("Compression: File Size by Format and Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(d, d) for d in datasets], fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

    _save_fig(fig, "compression_bar", output_dir)


# ---------------------------------------------------------------------------
# 2. Read speed bar chart
# ---------------------------------------------------------------------------

def plot_read_speed(output_dir: Path | None = None):
    """Grouped bar chart: read time per format, grouped by dataset (log scale)."""
    _setup_style()
    results = load_all_results(benchmark="read_full")
    datasets = MAIN_DATASETS
    formats = KEY_FORMATS

    pivot = _build_pivot(results, "median_wall_s", datasets)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    n_fmts = len(formats)
    width = 0.8 / n_fmts

    for i, fmt in enumerate(formats):
        vals = [pivot.get(fmt, {}).get(ds) or 0 for ds in datasets]
        offset = (i - n_fmts / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=FORMAT_LABELS.get(fmt, fmt),
            color=FORMAT_COLORS.get(fmt, "#999999"),
        )

    ax.set_ylabel("Read time (seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("Full Read Performance by Format and Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(d, d) for d in datasets], fontsize=9)
    ax.legend(fontsize=8, ncol=4, loc="upper left")

    _save_fig(fig, "read_speed_bar", output_dir)


# ---------------------------------------------------------------------------
# 3. Scaling curves (read time vs cell count, log-log)
# ---------------------------------------------------------------------------

def plot_scaling_curves(output_dir: Path | None = None):
    """Line plot: read time vs cell count (log-log), one line per format."""
    _setup_style()
    results = load_all_results(benchmark="read_full")
    datasets = MAIN_DATASETS
    formats = KEY_FORMATS

    pivot = _build_pivot(results, "median_wall_s", datasets)

    fig, ax = plt.subplots(figsize=(10, 7))

    for fmt in formats:
        row = pivot.get(fmt, {})
        xs, ys = [], []
        for ds in datasets:
            cfg = DATASETS.get(ds)
            val = row.get(ds)
            if cfg and val:
                xs.append(cfg.n_obs)
                ys.append(val)
        if xs:
            ax.plot(
                xs, ys, "o-",
                label=FORMAT_LABELS.get(fmt, fmt),
                color=FORMAT_COLORS.get(fmt, "#999999"),
                markersize=5, linewidth=1.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Read time (seconds)")
    ax.set_title("Read Time Scaling with Dataset Size")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    _save_fig(fig, "scaling_curves", output_dir)


# ---------------------------------------------------------------------------
# 4. Parallel scaling plot
# ---------------------------------------------------------------------------

def plot_parallel_scaling(output_dir: Path | None = None):
    """Line plot: speedup vs thread count, one line per format×dataset."""
    _setup_style()
    results = load_all_results(benchmark="parallel_scaling")
    datasets = ["census_500k", "census_1m", "census_5m"]
    formats = ["scx_auto", "scx_pcodec", "scx_zstd", "zarr_lz4", "tiledb_soma"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_results = [r for r in results if r.get("dataset") == ds]

        for r in ds_results:
            fmt = r.get("format", "")
            if fmt not in formats:
                continue
            speedup = r.get("metadata", {}).get("speedup", {})
            if not speedup:
                continue
            threads = sorted(int(k) for k in speedup.keys())
            spds = [speedup[str(t)] for t in threads]
            ax.plot(
                threads, spds, "o-",
                label=FORMAT_LABELS.get(fmt, fmt),
                color=FORMAT_COLORS.get(fmt, "#999999"),
                markersize=4, linewidth=1.5,
            )

        # Ideal scaling line
        max_t = 32
        ax.plot([1, max_t], [1, max_t], "k--", alpha=0.3, label="Ideal")

        ds_short = SHORT_NAMES.get(ds, ds).replace("\n", " ")
        ax.set_title(ds_short)
        ax.set_xlabel("Threads")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    axes[0].set_ylabel("Speedup vs 1 thread")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Parallel Read Scaling", fontsize=13)
    fig.tight_layout()

    _save_fig(fig, "parallel_scaling", output_dir)


# ---------------------------------------------------------------------------
# 4b. Parallel write scaling
# ---------------------------------------------------------------------------

def plot_parallel_write_scaling(output_dir: Path | None = None):
    """Line plot: write speedup vs thread count, one line per format x dataset.

    Plots the "write_only" mode (in-memory AnnData → format) when available,
    as it isolates the parallel encoding benefit. Falls back to "full" mode.
    """
    _setup_style()
    results = load_all_results(benchmark="parallel_write_scaling")
    datasets = ["census_500k", "census_1m", "census_5m"]
    formats = ["scx_auto", "scx_pcodec", "scx_zstd", "zarr_lz4", "tiledb_soma"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        ds_results = [r for r in results if r.get("dataset") == ds]

        for r in ds_results:
            fmt = r.get("format", "")
            if fmt not in formats:
                continue
            meta = r.get("metadata", {})
            # Prefer write_only mode for SCX; fall back to full.
            speedup_by_mode = meta.get("speedup", {})
            speedup = speedup_by_mode.get("write_only") or speedup_by_mode.get("full", {})
            if not speedup:
                continue
            threads = sorted(int(k) for k in speedup.keys())
            spds = [speedup[str(t)] for t in threads]
            ax.plot(
                threads, spds, "o-",
                label=FORMAT_LABELS.get(fmt, fmt),
                color=FORMAT_COLORS.get(fmt, "#999999"),
                markersize=4, linewidth=1.5,
            )

        # Ideal scaling line
        max_t = 32
        ax.plot([1, max_t], [1, max_t], "k--", alpha=0.3, label="Ideal")

        ds_short = SHORT_NAMES.get(ds, ds).replace("\n", " ")
        ax.set_title(ds_short)
        ax.set_xlabel("Threads")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32])
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    axes[0].set_ylabel("Speedup vs 1 thread")
    axes[-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Parallel Write Scaling", fontsize=13)
    fig.tight_layout()

    _save_fig(fig, "parallel_write_scaling", output_dir)


# ---------------------------------------------------------------------------
# 5. Memory vs cell count
# ---------------------------------------------------------------------------

def plot_memory_scaling(output_dir: Path | None = None):
    """Scatter + line: peak RSS vs cell count."""
    _setup_style()
    results = load_all_results(benchmark="memory")
    datasets = [
        "tabula_sapiens_100k", "census_500k", "census_1m", "census_5m",
    ]
    formats = KEY_FORMATS

    fig, ax = plt.subplots(figsize=(10, 7))

    for fmt in formats:
        fmt_results = [r for r in results if r.get("format") == fmt]
        xs, ys = [], []
        for ds in datasets:
            cfg = DATASETS.get(ds)
            dr = [r for r in fmt_results if r.get("dataset") == ds]
            if cfg and dr:
                r = dr[0]
                read_full_runs = [
                    run for run in r.get("runs", [])
                    if run.get("extra", {}).get("operation") == "read_full"
                ]
                if read_full_runs:
                    peak = statistics.median(run["peak_rss_mb"] for run in read_full_runs)
                    xs.append(cfg.n_obs)
                    ys.append(peak)
        if xs:
            ax.plot(
                xs, ys, "o-",
                label=FORMAT_LABELS.get(fmt, fmt),
                color=FORMAT_COLORS.get(fmt, "#999999"),
                markersize=5, linewidth=1.5,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Memory Usage Scaling with Dataset Size")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    _save_fig(fig, "memory_scaling", output_dir)


# ---------------------------------------------------------------------------
# 6. Lazy vs materialized RSS comparison (Phase 4d)
# ---------------------------------------------------------------------------

def plot_lazy_vs_materialized_rss(output_dir: Path | None = None):
    """Bar chart: lazy vs materialized memory comparison from COMPREHENSIVE-BENCHMARKING.md data."""
    _setup_style()

    # Data from §Lazy Preprocessing Benchmarks in COMPREHENSIVE-BENCHMARKING.md
    operations = [
        "Lazy preprocess\n(normalize+log1p)",
        "Full pipeline\n(open→Leiden)",
        "Materialized\n(scanpy in-place)",
    ]
    rss_mb = [3491, 10923, 22343]
    colors = ["#1f77b4", "#17becf", "#d62728"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(operations, rss_mb, color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, rss_mb):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
            f"{val / 1024:.1f} GB",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Lazy vs Materialized Memory Usage (census_1m)")
    ax.set_ylim(0, max(rss_mb) * 1.15)

    # Add reduction annotations
    ax.annotate(
        "84% reduction", xy=(0, rss_mb[0]), xytext=(0.5, 15000),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=10, color="green", fontweight="bold",
    )

    _save_fig(fig, "lazy_vs_materialized_rss", output_dir)


# ---------------------------------------------------------------------------
# 7. Column-projected aggregation latency (Phase 4d)
# ---------------------------------------------------------------------------

def plot_column_projection_latency(output_dir: Path | None = None):
    """Bar chart: projected vs unprojected aggregation latency."""
    _setup_style()

    # Data from COMPREHENSIVE-BENCHMARKING.md
    modes = ["Unprojected\n(61K genes)", "Projected\n(500 genes)"]
    latency = [25.2, 46.8]
    colors = ["#1f77b4", "#ff7f0e"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(modes, latency, color=colors, width=0.4, edgecolor="white")

    for bar, val in zip(bars, latency):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Median Latency (seconds)")
    ax.set_title("Column-Projected Streaming Aggregation (census_1m)")
    ax.set_ylim(0, max(latency) * 1.2)

    _save_fig(fig, "column_projection_latency", output_dir)


# ---------------------------------------------------------------------------
# 8. Out-of-core pipeline RSS time-series (Phase 4d)
# ---------------------------------------------------------------------------

def plot_ooc_pipeline_rss(output_dir: Path | None = None):
    """Stacked bar chart: per-stage RSS for the out-of-core pipeline."""
    _setup_style()

    # Per-stage data from COMPREHENSIVE-BENCHMARKING.md (census_1m, Phase 4e)
    stages = ["normalize\n+log1p", "HVG", "PCA", "kNN", "UMAP", "Leiden", "DE"]

    # Out-of-core pipeline times (census_1m, Phase 4e)
    ooc_times = [4.80, 39.39, 18.25, 108.17, 668.66, 2938.54, 193.16]
    scanpy_times = [54.82, 72.62, 8.65, 119.26, 827.95, 2830.84, 92.90]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(stages))
    width = 0.35

    bars1 = ax.bar(x - width / 2, ooc_times, width, label="SCX out-of-core",
                   color="#1f77b4", edgecolor="white")
    bars2 = ax.bar(x + width / 2, scanpy_times, width, label="scanpy in-memory",
                   color="#d62728", edgecolor="white")

    ax.set_ylabel("Time (seconds, log scale)")
    ax.set_yscale("log")
    ax.set_title("Pipeline Stage Timing: SCX Out-of-Core vs Scanpy (census_1m)")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.legend(fontsize=10)

    _save_fig(fig, "ooc_pipeline_stages", output_dir)


# ---------------------------------------------------------------------------
# 9. Pareto frontier (compression ratio vs read speed)
# ---------------------------------------------------------------------------

def plot_pareto_frontier(output_dir: Path | None = None):
    """Scatter: compression ratio vs read speed, one point per format×dataset."""
    _setup_style()

    comp_results = load_all_results(benchmark="compression")
    read_results = load_all_results(benchmark="read_full")
    datasets = ["census_500k", "census_1m", "census_5m"]
    formats = KEY_FORMATS

    # Get h5ad_none sizes as baseline
    baseline: dict[str, float] = {}
    for r in comp_results:
        if r.get("format") == "h5ad_none":
            ds = r.get("dataset", "")
            baseline[ds] = r.get("file_size_bytes", 1)

    # Build compression ratio lookup
    comp_ratios: dict[str, dict[str, float]] = {}
    for r in comp_results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        size = r.get("file_size_bytes")
        base = baseline.get(ds)
        if size and base and size > 0:
            comp_ratios.setdefault(fmt, {})[ds] = base / size

    # Build read time lookup
    read_times: dict[str, dict[str, float]] = {}
    for r in read_results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        wall = r.get("median_wall_s")
        if wall:
            read_times.setdefault(fmt, {})[ds] = wall

    fig, ax = plt.subplots(figsize=(10, 7))

    for fmt in formats:
        for ds in datasets:
            ratio = comp_ratios.get(fmt, {}).get(ds)
            rt = read_times.get(fmt, {}).get(ds)
            if ratio and rt:
                marker = {"census_500k": "o", "census_1m": "s", "census_5m": "D"}.get(ds, "o")
                ax.scatter(
                    ratio, rt,
                    color=FORMAT_COLORS.get(fmt, "#999999"),
                    marker=marker, s=60, alpha=0.8,
                    label=f"{FORMAT_LABELS.get(fmt, fmt)} ({ds})" if ds == "census_1m" else "_nolegend_",
                )

    # Identify Pareto front for census_1m
    pareto_points = []
    for fmt in formats:
        ratio = comp_ratios.get(fmt, {}).get("census_1m")
        rt = read_times.get(fmt, {}).get("census_1m")
        if ratio and rt:
            pareto_points.append((ratio, rt, fmt))

    if pareto_points:
        # Sort by compression ratio ascending
        pareto_points.sort(key=lambda p: p[0])
        # Find Pareto frontier (higher ratio, lower time is better)
        front = []
        min_time = float("inf")
        for ratio, rt, fmt in sorted(pareto_points, key=lambda p: -p[0]):
            if rt <= min_time:
                front.append((ratio, rt, fmt))
                min_time = rt
        if len(front) >= 2:
            front.sort(key=lambda p: p[0])
            ax.plot(
                [p[0] for p in front], [p[1] for p in front],
                "k--", alpha=0.5, linewidth=1, label="Pareto front (1M)",
            )

    ax.set_xlabel("Compression ratio (vs h5ad uncompressed)")
    ax.set_ylabel("Read time (seconds)")
    ax.set_yscale("log")
    ax.set_title("Pareto Frontier: Compression vs Read Speed")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Add marker legend for dataset sizes
    from matplotlib.lines import Line2D
    ds_legend = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=6, label="census_500k"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None", markersize=6, label="census_1m"),
        Line2D([0], [0], marker="D", color="gray", linestyle="None", markersize=6, label="census_5m"),
    ]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=ds_legend, fontsize=8, loc="lower right", title="Dataset")

    _save_fig(fig, "pareto_frontier", output_dir)


# ---------------------------------------------------------------------------
# 10. ML loader throughput bar chart
# ---------------------------------------------------------------------------

def plot_ml_loader(output_dir: Path | None = None):
    """Bar chart: batches/sec per loader."""
    _setup_style()
    results = load_all_results(benchmark="ml_loader")
    datasets = ["pbmc3k", "tabula_sapiens_100k", "census_1m"]

    # Collect batches/sec for hvg_norm scenario
    data: dict[str, dict[str, float]] = {}
    for r in results:
        fmt = r.get("format", "")
        ds = r.get("dataset", "")
        if ds not in datasets:
            continue
        hvg_norm_bps = []
        for run in r.get("runs", []):
            if run.get("extra", {}).get("scenario") == "hvg_norm":
                bps = run["extra"].get("batches_per_sec")
                if bps is not None:
                    hvg_norm_bps.append(bps)
        if hvg_norm_bps:
            data.setdefault(fmt, {})[ds] = statistics.median(hvg_norm_bps)

    if not data:
        logger.warning("No ML loader results found, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    formats = sorted(data.keys(), key=lambda f: FORMAT_ORDER.index(f) if f in FORMAT_ORDER else 999)
    x = np.arange(len(datasets))
    n_fmts = len(formats)
    width = 0.8 / max(n_fmts, 1)

    for i, fmt in enumerate(formats):
        vals = [data.get(fmt, {}).get(ds, 0) for ds in datasets]
        offset = (i - n_fmts / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width,
            label=FORMAT_LABELS.get(fmt, fmt),
            color=FORMAT_COLORS.get(fmt, "#999999"),
        )

    ax.set_ylabel("Batches/sec (hvg_norm)")
    ax.set_yscale("log")
    ax.set_title("ML Loader Throughput (batch=1024, HVG=2000, normalize+log1p)")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES.get(d, d) for d in datasets], fontsize=9)
    ax.legend(fontsize=9)

    _save_fig(fig, "ml_loader_throughput", output_dir)


# ---------------------------------------------------------------------------
# 11. Accelerator speedup bar chart
# ---------------------------------------------------------------------------

def plot_accelerator_speedup(output_dir: Path | None = None):
    """Grouped bar chart: SCX vs scanpy per-stage speedup."""
    _setup_style()

    # Data from Phase 5 / Phase 4e results in COMPREHENSIVE-BENCHMARKING.md
    # tabula_sapiens_100k (100K cells)
    stages_100k = ["PCA", "kNN", "UMAP", "Leiden", "DE"]
    scx_100k = [4.49, 21.86, 47.20, 3.95, 3.58]
    scanpy_100k = [3.97, 8.03, 57.32, 132.04, 14.32]

    # census_1m (1M cells) — partial data
    stages_1m = ["PCA", "kNN", "UMAP", "Leiden", "DE"]
    scx_1m = [21.29, 103.32, 573.32, 54.96, 27.34]
    scanpy_1m = [8.37, 123.78, 784.16, 2226.0, 17.49]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # tabula_100k
    x = np.arange(len(stages_100k))
    width = 0.35
    ax1.bar(x - width / 2, scx_100k, width, label="SCX", color="#1f77b4")
    ax1.bar(x + width / 2, scanpy_100k, width, label="scanpy", color="#d62728")
    ax1.set_ylabel("Time (seconds, log scale)")
    ax1.set_yscale("log")
    ax1.set_title("tabula_sapiens_100k (100K cells)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages_100k)
    ax1.legend()

    # Add speedup annotations
    for i, (s, sc) in enumerate(zip(scx_100k, scanpy_100k)):
        spd = sc / s
        color = "green" if spd > 1 else "red"
        ax1.text(i, max(s, sc) * 1.5, f"{spd:.1f}x", ha="center", fontsize=8,
                 color=color, fontweight="bold")

    # census_1m
    x = np.arange(len(stages_1m))
    ax2.bar(x - width / 2, scx_1m, width, label="SCX", color="#1f77b4")
    ax2.bar(x + width / 2, scanpy_1m, width, label="scanpy", color="#d62728")
    ax2.set_ylabel("Time (seconds, log scale)")
    ax2.set_yscale("log")
    ax2.set_title("census_1m (1M cells)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(stages_1m)
    ax2.legend()

    for i, (s, sc) in enumerate(zip(scx_1m, scanpy_1m)):
        spd = sc / s
        color = "green" if spd > 1 else "red"
        ax2.text(i, max(s, sc) * 1.5, f"{spd:.1f}x", ha="center", fontsize=8,
                 color=color, fontweight="bold")

    fig.suptitle("Accelerator Benchmarks: SCX vs scanpy", fontsize=13)
    fig.tight_layout()

    _save_fig(fig, "accelerator_speedup", output_dir)


# ---------------------------------------------------------------------------
# 12. Full pipeline stacked bar chart
# ---------------------------------------------------------------------------

def plot_pipeline_comparison(output_dir: Path | None = None):
    """Stacked bar chart: pipeline total time broken down by stage."""
    _setup_style()

    # Phase 4e data from COMPREHENSIVE-BENCHMARKING.md
    # tabula_sapiens_100k
    stages = ["normalize+log1p", "HVG", "PCA", "kNN", "UMAP", "Leiden", "DE"]
    pipelines_100k = {
        "SCX out-of-core": [0.84, 5.39, 4.49, 21.86, 47.20, 3.95, 3.58],
        "SCX preprocess": [0, 5.52, 4.10, 17.52, 48.78, 4.08, 3.46],
        "scanpy": [0.74, 5.16, 3.97, 8.03, 57.32, 132.04, 14.32],
    }
    # Note: SCX preprocess includes preprocess_write=11.70 but we show stages

    pipelines_1m = {
        "SCX out-of-core": [4.80, 39.39, 18.25, 108.17, 668.66, 2938.54, 193.16],
        "scanpy": [54.82, 72.62, 8.65, 119.26, 827.95, 2830.84, 92.90],
    }

    stage_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 100K
    pipe_names_100k = list(pipelines_100k.keys())
    x = np.arange(len(pipe_names_100k))
    bottoms = [0] * len(pipe_names_100k)
    for j, stage in enumerate(stages):
        vals = [pipelines_100k[p][j] for p in pipe_names_100k]
        ax1.bar(x, vals, 0.5, bottom=bottoms, label=stage if j < len(stages) else "",
                color=stage_colors[j])
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    # Add total labels
    for i, p in enumerate(pipe_names_100k):
        total = sum(pipelines_100k[p])
        ax1.text(i, total + 2, f"{total:.0f}s", ha="center", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(pipe_names_100k, fontsize=9)
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("tabula_sapiens_100k (100K)")
    ax1.legend(fontsize=7, loc="upper right")

    # 1M
    pipe_names_1m = list(pipelines_1m.keys())
    x = np.arange(len(pipe_names_1m))
    bottoms = [0] * len(pipe_names_1m)
    for j, stage in enumerate(stages):
        vals = [pipelines_1m[p][j] for p in pipe_names_1m]
        ax2.bar(x, vals, 0.4, bottom=bottoms, label=stage, color=stage_colors[j])
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    for i, p in enumerate(pipe_names_1m):
        total = sum(pipelines_1m[p])
        ax2.text(i, total + 50, f"{total:.0f}s", ha="center", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(pipe_names_1m, fontsize=9)
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("census_1m (1M)")

    fig.suptitle("End-to-End Pipeline: Stage Breakdown", fontsize=13)
    fig.tight_layout()

    _save_fig(fig, "pipeline_comparison", output_dir)


# ---------------------------------------------------------------------------
# 13. Streaming preprocessing memory scaling
# ---------------------------------------------------------------------------

def plot_streaming_preprocess_memory(output_dir: Path | None = None):
    """Bar chart: OOC pipeline peak RSS vs scanpy for different datasets."""
    _setup_style()

    # Data from COMPREHENSIVE-BENCHMARKING.md
    datasets_names = ["tabula_100k\n(100K)", "census_1m\n(1M)"]

    # Phase 4e data
    scx_ooc_rss = [1108, 2399]  # MB
    scanpy_rss = [1605, 9871]  # MB

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets_names))
    width = 0.3

    bars1 = ax.bar(x - width / 2, scx_ooc_rss, width, label="SCX out-of-core",
                   color="#1f77b4")
    bars2 = ax.bar(x + width / 2, scanpy_rss, width, label="scanpy in-memory",
                   color="#d62728")

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            label = f"{h / 1024:.1f} GB" if h >= 1024 else f"{h:.0f} MB"
            ax.text(bar.get_x() + bar.get_width() / 2, h + 100, label,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add reduction labels
    for i in range(len(datasets_names)):
        reduction = (1 - scx_ooc_rss[i] / scanpy_rss[i]) * 100
        mid_x = x[i]
        mid_y = max(scx_ooc_rss[i], scanpy_rss[i]) * 1.15
        ax.text(mid_x, mid_y, f"{reduction:.0f}% less", ha="center",
                fontsize=9, color="green", fontweight="bold")

    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Pipeline Memory: SCX Out-of-Core vs Scanpy")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_names)
    ax.legend()
    ax.set_ylim(0, max(scanpy_rss) * 1.3)

    _save_fig(fig, "streaming_preprocess_memory", output_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_all_plots(output_dir: Path | None = None) -> list[str]:
    """Generate all plots. Returns list of saved file stems."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fns = [
        ("compression_bar", plot_compression),
        ("read_speed_bar", plot_read_speed),
        ("scaling_curves", plot_scaling_curves),
        ("parallel_scaling", plot_parallel_scaling),
        ("parallel_write_scaling", plot_parallel_write_scaling),
        ("memory_scaling", plot_memory_scaling),
        ("lazy_vs_materialized_rss", plot_lazy_vs_materialized_rss),
        ("column_projection_latency", plot_column_projection_latency),
        ("ooc_pipeline_stages", plot_ooc_pipeline_rss),
        ("pareto_frontier", plot_pareto_frontier),
        ("ml_loader_throughput", plot_ml_loader),
        ("accelerator_speedup", plot_accelerator_speedup),
        ("pipeline_comparison", plot_pipeline_comparison),
        ("streaming_preprocess_memory", plot_streaming_preprocess_memory),
    ]

    generated = []
    for name, fn in plot_fns:
        try:
            fn(output_dir)
            generated.append(name)
        except Exception:
            logger.exception(f"Failed to generate plot: {name}")

    return generated
