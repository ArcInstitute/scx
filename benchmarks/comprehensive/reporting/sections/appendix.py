"""Appendix (Phase 7 — external/manual sources appendix).

Phase 7 changes:
- Add an "External and Manual Data Sources" appendix that automatically
  lists every table and result row sourced from manual or external data.
- Keep the glossary from Phase 4.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, TableBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Appendix")

    # ── Glossary ──────────────────────────────────────────────────────
    c.sections.append(Section(title="Glossary", blocks=[
        TextBlock(
            "- **SCX:** Sparse Cell eXpression System.\n"
            "- **h5ad:** AnnData's HDF5-based serialization format.\n"
            "- **TileDB-SOMA:** Single-cell Open Matrix Architecture by TileDB.\n"
            "- **Zarr:** Cloud-native multidimensional array format."
        ),
    ]))

    # ── External and manual data sources ──────────────────────────────
    sources = tables.collect_manual_sources(store)
    if sources:
        headers = ["Table / Source", "Kind", "Path", "Reason"]
        rows: list[list[str]] = []
        for s in sources:
            rows.append([
                s.get("table", "—"),
                s.get("source_kind", "—"),
                s.get("path", "—") or "—",
                s.get("reason", "—") or "—",
            ])
        c.sections.append(Section(
            title="External and Manual Data Sources",
            blocks=[
                TextBlock(
                    "The following tables and data rows in this report are "
                    "sourced from manual curation, external benchmark scripts, "
                    "or standalone reports outside the comprehensive harness. "
                    "Each entry lists its provenance so that report freshness "
                    "can be verified."
                ),
                TableBlock(
                    headers=headers, rows=rows,
                    caption="External and manual data sources used in this report",
                ),
            ],
        ))
    else:
        c.sections.append(Section(
            title="External and Manual Data Sources",
            blocks=[
                TextBlock(
                    "*All data in this report is sourced from raw JSON results "
                    "in the comprehensive harness. No manual or external sources "
                    "were used.*"
                ),
            ],
        ))

    return c
