"""Chapter 9: Accelerators (Phase 6 — CSC/CSR dispatch moved here).

Phase 6 changes:
- Move CSC/CSR dispatch from the operations chapter into the
  accelerator chapter, since CSC vs CSR is an accelerator storage-layout
  concern rather than a file-operation concern.
- Keep parity tables adjacent to timing tables (Phase 5 carry-forward).
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Accelerators (PCA, kNN, UMAP, Leiden)")

    # ── Parity overview ──────────────────────────────────────────────
    c.sections.append(Section(
        title="Accelerator Parity",
        blocks=[
            TextBlock(
                "Before interpreting speedups, confirm that SCX's Rust-native "
                "accelerators reproduce scanpy/leidenalg reference results.  "
                "The table below shows per-operation parity metrics extracted "
                "from the same benchmark runs that produce timing data.  "
                "Full correctness details are in Chapter 3."
            ),
            tables.accelerator_parity_table(),
        ],
    ))

    # ── CPU accelerator performance ──────────────────────────────────
    c.sections.append(Section(
        title="CPU Accelerator Performance",
        blocks=[
            TextBlock(
                "All benchmarks use SCX's Rust-native implementations vs "
                "scanpy's standard Python stack.  Timing results are backed "
                "by raw JSON from the comprehensive harness."
            ),
        ],
    ))

    # ── CSC vs CSR dispatch (moved from operations chapter) ──────────
    c.sections.append(Section(
        title="CSC vs CSR Accelerator Dispatch",
        blocks=[
            TextBlock(
                "Time to compute column-oriented operations (column variance, "
                "HVG, DE, QC metrics) on a CSR-native file vs CSC sidecar.  "
                "This is an accelerator storage-layout concern — CSC sidecars "
                "can accelerate gene-major traversals."
            ),
            tables.bench_csc_dispatch_table(),
        ],
    ))

    return c
