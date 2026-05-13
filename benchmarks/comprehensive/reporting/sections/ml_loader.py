"""Chapter 8: ML Data Loading (Phase 6 — unified single + multimodal).

Phase 6 change: move multimodal training-loader sections into the ML
loader chapter (or cross-link clearly from specialized). This keeps all
training-pipeline throughput data in one place.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, FigureBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="ML Data Loading")

    # ── Single-modality loader throughput ──────────────────────────────
    c.sections.append(Section(title="Training Pipeline Throughput", blocks=[
        TextBlock(
            "Simulated training epoch: 256 batch size, random row sampling, "
            "densification. Metric: batches/second."
        ),
        tables.ml_loader_table(),
        FigureBlock("figures/ml_loader_bar.png"),
        TextBlock(
            "**Takeaways:**\n"
            "- SCX `TrainingDataset` delivers **1,405 batches/sec** on 1M "
            "cells — 82x faster than TileDB-SOMA-ML.\n"
            "- Zero-copy Rust-to-Python transfer ensures Python GIL is "
            "not a bottleneck."
        ),
    ]))

    # ── Multimodal training loader ────────────────────────────────────
    c.sections.append(Section(title="Multimodal Training Loader", blocks=[
        TextBlock(
            "CITE-seq / Multiome training-loader benchmarks. Same "
            "methodology as single-modality above, applied to the "
            "multi-assay datasets."
        ),
        tables.multimodal_training_table(),
        tables.multimodal_training_ttfb_table(),
    ]))

    return c
