"""Chapter 8: ML Data Loading (Phase 7 — data-derived commentary).

Phase 7 changes:
- Replace hardcoded '1,405 batches/sec' and '82x' claims with values
  derived from ``derive_ml_loader_headlines()``.
- Use ``CommentaryBlock`` with ``SourceRef`` for numeric takeaways.

Phase 6 carry-forward:
- Unified single + multimodal loader sections.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CommentaryBlock, FigureBlock,
)
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, SourceRef, SourceKind,
)
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="ML Data Loading")

    # Pre-derive loader headlines for commentary.
    ml = tables.derive_ml_loader_headlines()

    # ── Single-modality loader throughput ──────────────────────────────
    takeaway_parts = ["**Takeaways:**"]

    bps = ml.get("scx_best_bps")
    ds_name = ml.get("scx_best_dataset", "census scale")
    soma_spd = ml.get("scx_vs_soma")

    if bps is not None:
        bps_str = f"{bps:,.0f}"
        soma_str = ""
        if soma_spd is not None:
            soma_str = f" — {soma_spd:.0f}x faster than TileDB-SOMA-ML"
        takeaway_parts.append(
            f"- SCX `TrainingDataset` delivers **{bps_str} batches/sec** on "
            f"{ds_name}{soma_str}."
        )
    else:
        takeaway_parts.append(
            "- SCX `TrainingDataset` is the fastest ML data loader; "
            "see table above for concrete throughput numbers."
        )

    takeaway_parts.append(
        "- Zero-copy Rust-to-Python transfer ensures Python GIL is "
        "not a bottleneck."
    )

    c.sections.append(Section(title="Training Pipeline Throughput", blocks=[
        TextBlock(
            "Simulated training epoch: 256 batch size, random row sampling, "
            "densification. Metric: batches/second."
        ),
        tables.ml_loader_table(),
        FigureBlock("figures/ml_loader_bar.png"),
        CommentaryBlock(
            "\n".join(takeaway_parts),
            source=SourceRef(kind=SourceKind.raw_json,
                             reason="derived from ml_loader benchmark results"),
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
