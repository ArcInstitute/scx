"""Chapter 9: Accelerators (Phase 5 — parity tables adjacent to timing).

Adds accelerator parity tables alongside performance data so readers
see correctness and speed side by side.
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
            tables.bench_csc_dispatch_table(),
        ],
    ))

    return c
