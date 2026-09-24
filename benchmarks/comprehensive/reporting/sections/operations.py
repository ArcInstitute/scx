"""Chapter 11: SCX-specific Format Operations.

Includes:
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

    # ── Grouped sharding (sort / convert --group-by) ──────────────────
    c.sections.append(Section(title="Grouped Sharding", blocks=[
        TextBlock(
            "Reference-first, group-clustered CSR layout via `scx sort "
            "--group-by` (re-shard an existing file) and `scx convert "
            "--group-by` (write the grouped layout during ingest). The "
            "convert path auto-routes by source density: CSR sources take the "
            "one-pass streaming gather (cheaper); dense sources fall back to a "
            "two-pass plain-convert-then-sort. `read-back correct` verifies "
            "`read_group(label)` partitions the obs axis and the reference "
            "label is isolated. SCX-only."
        ),
        tables.grouped_sharding_table(),
    ]))

    return c
