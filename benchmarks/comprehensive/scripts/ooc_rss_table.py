#!/usr/bin/env python3
"""Phase A.7 — side-by-side peak-RSS table.

Aggregates ``memory`` benchmark JSONs from
``benchmarks/comprehensive/results/raw/`` into a markdown + JSON table
comparing the SLAF lazy path against SCX lazy, AnnData-on-Zarr (backed),
and Zarr full-load on the same dataset.

Rows: one per dataset.
Cols: each format's median ``peak_rss_mb`` for ``read_full`` and
``read_subset_1k``, plus the median ``delta_rss_mb`` stored in ``metadata``.

Usage::

    python benchmarks/comprehensive/scripts/ooc_rss_table.py \\
        --datasets census_1m \\
        --formats slaf scx_auto h5ad_none zarr_zstd anndata_zarr_backed \\
        --out benchmarks/comprehensive/results/reports/phase5A_ooc_rss.md
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.config import RAW_RESULTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_FORMATS = [
    "slaf",
    "scx_auto",
    "h5ad_none",
    "zarr_zstd",
]

_FORMAT_LABELS = {
    "slaf": "SLAF (lazy)",
    "scx_auto": "SCX (lazy)",
    "h5ad_none": "h5ad (uncompressed)",
    "h5ad_gzip": "h5ad (gzip)",
    "zarr_zstd": "Zarr (zstd)",
    "zarr_lz4": "Zarr (blosc-lz4)",
    "anndata_zarr_backed": "AnnData-on-Zarr (backed)",
    "tiledb_soma": "TileDB-SOMA",
}


def _load(dataset: str, format_key: str) -> dict | None:
    path = RAW_RESULTS_DIR / f"memory__{format_key}__{dataset}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _op(run: dict) -> str | None:
    """``operation`` is written into ``run["extra"]`` by ``memory.py``."""
    extra = run.get("extra") or {}
    return extra.get("operation")


def _median_peak_rss(result: dict, operation: str) -> float | None:
    runs = [r for r in result.get("runs", []) if _op(r) == operation]
    if not runs:
        return None
    return statistics.median(r["peak_rss_mb"] for r in runs)


def _median_delta_rss(result: dict, operation: str) -> float | None:
    runs = [r for r in result.get("runs", []) if _op(r) == operation]
    if not runs:
        return None
    vals = [
        (r.get("extra") or {}).get("delta_rss_mb") for r in runs
    ]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return statistics.median(vals)


def _fmt_mb(v: float | None) -> str:
    return "—" if v is None else f"{v:,.1f}"


def render(datasets: list[str], formats: list[str]) -> tuple[str, dict]:
    labels = [_FORMAT_LABELS.get(k, k) for k in formats]

    lines: list[str] = []
    lines.append("# Phase A.7 — OOC Peak-RSS Comparison")
    lines.append("")
    lines.append(
        "Per-operation median peak RSS (MB) from the `memory` benchmark. "
        "`read_full` materializes the full expression matrix; `read_subset_1k` "
        "reads a 1,000-cell slice. `Δ` is the delta RSS attributable to the "
        "operation (baseline subtracted, captured in `metadata`)."
    )
    lines.append("")

    rows_payload: list[dict] = []

    for operation in ("read_full", "read_subset_1k"):
        lines.append(f"## {operation}")
        lines.append("")
        header = "| Dataset | " + " | ".join(labels) + " |"
        sep = "|---" * (1 + len(formats)) + "|"
        lines.append(header)
        lines.append(sep)
        for ds in datasets:
            cells = [ds]
            row_payload: dict = {"dataset": ds, "operation": operation, "formats": {}}
            for fmt in formats:
                result = _load(ds, fmt)
                if result is None:
                    cells.append("—")
                    row_payload["formats"][fmt] = None
                    continue
                peak = _median_peak_rss(result, operation)
                delta = _median_delta_rss(result, operation)
                cells.append(
                    f"{_fmt_mb(peak)} (Δ {_fmt_mb(delta)})"
                    if peak is not None else "—"
                )
                row_payload["formats"][fmt] = {
                    "median_peak_rss_mb": peak,
                    "median_delta_rss_mb": delta,
                    "n_runs": len([
                        r for r in result.get("runs", [])
                        if r.get("operation") == operation
                    ]),
                }
            lines.append("| " + " | ".join(cells) + " |")
            rows_payload.append(row_payload)
        lines.append("")

    payload = {
        "datasets": datasets,
        "formats": formats,
        "rows": rows_payload,
    }
    return "\n".join(lines) + "\n", payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Dataset names to include as rows.")
    p.add_argument("--formats", nargs="+", default=_DEFAULT_FORMATS,
                   help=f"Format keys (default: {_DEFAULT_FORMATS})")
    p.add_argument("--out", type=Path, default=None,
                   help="Write markdown to this path (default: stdout).")
    p.add_argument("--out-json", type=Path, default=None,
                   help="Also write JSON payload to this path.")
    args = p.parse_args()

    md, payload = render(args.datasets, args.formats)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        logger.info("Wrote markdown table to %s", args.out)
    else:
        print(md)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Wrote JSON payload to %s", args.out_json)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
