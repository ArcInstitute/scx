"""Chapter 11: SCX-specific Format Operations (Phase 6 — focused).

Phase 6 changes:
- CSC/CSR dispatch moved to the accelerator chapter (Ch 9).
- This chapter is now purely about fragment/manifest operations
  (append, delete, compact, rollback) — SCX-specific file operations
  that have no direct equivalent in competing formats.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="SCX-specific Format Operations")

    # ── Fragment & manifest operations ────────────────────────────────
    c.sections.append(Section(title="Fragment & Manifest Operations", blocks=[
        TextBlock(
            "Time to execute append/subset operations on SCX fragments.  "
            "These operations have no direct equivalent in h5ad, Zarr, or "
            "TileDB-SOMA."
        ),
        tables.fragment_ops_table(),
    ]))

    return c
