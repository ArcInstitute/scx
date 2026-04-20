"""
Standardized plot aesthetics (Phase I.7).

Single source of truth for colors, markers, and axis formatting so every
plot emitted by ``reporting/plots.py`` (and future landing-page trend
charts) matches. Importing this module registers the palette with
matplotlib if available; no-op when matplotlib isn't installed.
"""

from __future__ import annotations

from typing import Mapping

# Primary palette — colorblind-safe, sorted by typical benchmark winner order.
# SCX variants share a hue family (blues/teals) so reports group visually.
FORMAT_COLORS: Mapping[str, str] = {
    "scx_auto":           "#1f77b4",  # blue
    "scx_scx1":           "#2a9fd6",
    "scx_zstd":           "#17becf",
    "scx_lz4":            "#1abc9c",
    "scx_pcodec":         "#11a579",
    "scx_none":           "#6b8e9e",
    "zarr_zstd":          "#ff7f0e",  # orange
    "zarr_lz4":           "#f39c12",
    "tiledb_soma":        "#2ca02c",  # green
    "slaf":               "#9467bd",  # purple
    "h5ad_gzip":          "#d62728",  # red
    "h5ad_lzf":           "#e74c3c",
    "h5ad_none":          "#8c564b",
    "bpcells":            "#7f7f7f",  # grey
    "parquet_zstd":       "#c7c7c7",
    "anndata_zarr_backed": "#e377c2",  # pink (variant of zarr family)
}

# Consistent marker per family so grayscale prints stay readable.
FORMAT_MARKERS: Mapping[str, str] = {
    "scx_auto":           "o",
    "scx_scx1":           "s",
    "scx_zstd":           "^",
    "scx_lz4":            "v",
    "scx_pcodec":         "D",
    "scx_none":           "p",
    "zarr_zstd":          "X",
    "zarr_lz4":           "P",
    "tiledb_soma":        "*",
    "slaf":               "H",
    "h5ad_gzip":          "d",
    "h5ad_lzf":           "<",
    "h5ad_none":          ">",
    "bpcells":            "8",
    "parquet_zstd":       "h",
    "anndata_zarr_backed": "X",
}


def color_for(format_key: str) -> str:
    """Return the canonical color for a format key; fall back to grey."""
    return FORMAT_COLORS.get(format_key, "#7f7f7f")


def marker_for(format_key: str) -> str:
    return FORMAT_MARKERS.get(format_key, "o")


def apply_rcparams() -> None:
    """Apply the repo-wide matplotlib rcParams. No-op when matplotlib is
    absent. Callers that produce figures invoke this once before plotting.
    """
    try:
        import matplotlib as mpl
    except ImportError:
        return
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def humanize_seconds(seconds: float) -> str:
    """Y-axis formatter: seconds → human-friendly string."""
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def humanize_bytes(n: float) -> str:
    for unit, div in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= div:
            return f"{n / div:.1f} {unit}"
    return f"{n:.0f} B"
