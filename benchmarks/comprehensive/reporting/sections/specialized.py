"""Chapter 10: Specialized Workloads (Phase 5 — correctness before perf).

Adds correctness/validation summaries before the performance tables for
cell-eval and Harmony/LISI, so readers confirm parity before interpreting
speedups.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Specialized Workloads")

    # ── Multimodal ────────────────────────────────────────────────────
    c.sections.append(Section(
        title="Multimodal (CITE-seq / Multiome)",
        blocks=[
            tables.multimodal_compression_table(),
            tables.multimodal_compression_ratio_table(),
            tables.multimodal_training_table(),
            tables.multimodal_training_ttfb_table(),
        ],
    ))

    # ── Harmony & LISI — validation first, then scaling ──────────────
    harmony_blocks: list = [
        TextBlock(
            "Harmony/LISI validation status (full details in Chapter 3):"
        ),
        tables.harmony_lisi_correctness_summary_table(),
        tables.harmony_validation_table(),
        TextBlock("**Scaling performance:**"),
    ]
    harmony_blocks.extend(tables.harmony_scaling_table())
    harmony_blocks.append(tables.lisi_comparison_table())
    c.sections.append(Section(title="Harmony & LISI", blocks=harmony_blocks))

    # ── Cell-Eval — correctness first, then performance ──────────────
    c.sections.append(Section(
        title="Cell-Eval / Arc-Bench Parity",
        blocks=[
            TextBlock(
                "Cell-eval correctness status (full details in Chapter 3):"
            ),
            tables.cell_eval_correctness_summary_table(),
            TextBlock("**Performance:**"),
            tables.cell_eval_parity_perf_table(),
        ],
    ))

    return c
