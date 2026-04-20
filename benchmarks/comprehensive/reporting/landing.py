"""
Aggregated landing page (Phase I.7).

Builds a single ``index.md`` + ``index.html`` that summarizes the latest
snapshot for every benchmark dimension — one-screen cross-phase overview
of compression, read, selective, ML, memory, cloud, and cost.

The landing page is a thin aggregator over ``reporting/tables.py``; each
dimension section calls into an existing ``*_table()`` so the content
stays automatically in sync with the per-phase report. Cross-dimension
joins (e.g. cost × query latency) live here since they span phases that
don't otherwise see each other.
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path

from benchmarks.comprehensive.config import REPORTS_DIR
from benchmarks.comprehensive.reporting import dashboard, tables

logger = logging.getLogger(__name__)


_SECTIONS = [
    ("Compression", tables.compression_table),
    ("Full Read", tables.read_speed_table),
    ("Selective Read", tables.read_selective_table),
    ("Write", tables.write_speed_table),
    ("Parallel Scaling", tables.parallel_scaling_table),
    ("Memory (peak RSS)", tables.memory_table),
    ("Fragment Operations", tables.fragment_ops_table),
    ("Cloud Query Parity", tables.cloud_filtered_table),
    ("GCP Compute-Node Matrix", tables.gcp_matrix_table),
    ("CloudReader vs Full Pull", tables.cloud_reader_vs_pull_table),
    ("Cost Model (GCS)", tables.cost_model_table),
    ("ML Loader", tables.ml_loader_table),
    ("Correctness", tables.correctness_table),
]


def build_index_markdown() -> str:
    now = _dt.datetime.now().isoformat(timespec="seconds")
    chunks: list[str] = []
    chunks.append(f"# SCX Benchmark Landing Page\n\n")
    chunks.append(f"Generated {now}. See `BENCHMARK_REPORT.md` for full detail.\n\n")
    chunks.append("## Quick navigation\n\n")
    for title, _fn in _SECTIONS:
        slug = title.lower().replace(" ", "-").replace("(", "").replace(")", "")
        chunks.append(f"- [{title}](#{slug})\n")
    chunks.append("\n")

    for title, fn in _SECTIONS:
        chunks.append(f"## {title}\n\n")
        try:
            body = fn()
        except Exception as exc:  # noqa: BLE001
            body = f"*Section failed to render: {exc}*"
        chunks.append(body if body.strip() else f"*no data for {title}*")
        chunks.append("\n\n")

    # Cross-dimension join: cost × cloud query parity. Only meaningful
    # when both phase F (cost) and phase D (query) have landed for the
    # same datasets; empty otherwise.
    chunks.append("## Cost × Query Latency (joined)\n\n")
    chunks.append(_build_cost_x_query_table())
    chunks.append("\n")

    return "".join(chunks)


def _build_cost_x_query_table() -> str:
    """Cross-phase: cost-per-1M-cells × format × query (best-effort join).

    Combines the median wall-clock from Phase D's ``cloud_filtered`` with
    the USD/1M-cells from Phase F's ``cost_model``. One row per
    ``(format, scenario)`` that has both. When either side is missing,
    the column shows ``—``.
    """
    from benchmarks.comprehensive.results import load_all_results

    cost = load_all_results(benchmark="cost_model")
    filtered = load_all_results(benchmark="cloud_filtered")
    if not cost and not filtered:
        return "*No cost_model or cloud_filtered data — populate via run_parallel.py.*"

    # Pivot cost_model: (format, scenario) -> usd_per_million_median
    cost_pivot: dict[tuple[str, str], float] = {}
    for r in cost:
        fmt = r.get("format", "")
        per = (r.get("metadata", {}) or {}).get(
            "per_layout_scenario_median_usd_per_million", {}
        ) or {}
        for combo, val in per.items():
            if "::" not in combo:
                continue
            _layout, scen = combo.split("::", 1)
            cost_pivot[(fmt, scen)] = float(val)

    # Pivot cloud_filtered: (format, predicate_name) -> median_s
    wall_pivot: dict[tuple[str, str], float] = {}
    for r in filtered:
        fmt = r.get("format", "")
        per = (r.get("metadata", {}) or {}).get("per_predicate_summary", {}) or {}
        for pname, bucket in per.items():
            wall_pivot[(fmt, pname)] = float(bucket.get("median_s", 0.0))

    # Union of formats present on either side.
    formats = sorted({k[0] for k in cost_pivot} | {k[0] for k in wall_pivot})
    if not formats:
        return "*No comparable (format, scenario) pairs yet.*"

    lines = [
        "| Format | Scenario | Median wall (s) | USD/1M cells |",
        "|---|---|---:|---:|",
    ]
    # Rows: pick scenarios from cost_model where available, otherwise
    # fall back to cloud_filtered predicate names.
    scenarios = sorted({k[1] for k in cost_pivot} | {k[1] for k in wall_pivot})
    for fmt in formats:
        for scen in scenarios:
            wall = wall_pivot.get((fmt, scen))
            usd = cost_pivot.get((fmt, scen))
            if wall is None and usd is None:
                continue
            wall_cell = f"{wall:.3f}" if wall is not None else "—"
            usd_cell = f"${usd:.6f}" if usd is not None else "—"
            lines.append(f"| `{fmt}` | {scen} | {wall_cell} | {usd_cell} |")

    if len(lines) <= 2:
        return "*No overlapping (format, scenario) rows between cost_model and cloud_filtered.*"
    return "\n".join(lines)


def write_landing(output_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ``index.md`` + ``index.html`` under ``REPORTS_DIR`` (or override).

    HTML is generated by wrapping the markdown body with
    ``dashboard.render_html`` so the per-snapshot navigation + TOC shell
    is consistent with the detailed BENCHMARK_REPORT.html.
    """
    output_dir = output_dir or REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    body = build_index_markdown()
    md_path = output_dir / "index.md"
    md_path.write_text(body)
    logger.info("Wrote landing markdown to %s", md_path)

    prev_url = dashboard.previous_url(output_dir)
    html = dashboard.render_html(
        body, title="SCX Benchmark Landing", prev_url=prev_url,
    )
    html_path = output_dir / "index.html"
    html_path.write_text(html)
    logger.info("Wrote landing HTML to %s", html_path)

    return md_path, html_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    md, html = write_landing()
    print(f"Landing page: {md} / {html}")
