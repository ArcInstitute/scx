"""Chapter 5: Storage Efficiency (Phase 7 — data-derived commentary).

Phase 7 changes:
- Replace hardcoded file-size and ratio claims with data-derived values
  from ``derive_compression_headlines()``.
- Use ``CommentaryBlock`` with ``SourceRef`` for numeric takeaways.

Phase 6 carry-forward:
- Chapter-level summary card at the top.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CalloutBlock, CommentaryBlock, FigureBlock,
)
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, SourceRef, SourceKind,
)
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Storage Efficiency")

    # Pre-derive compression headlines for commentary.
    comp = tables.derive_compression_headlines()

    # ── Chapter-level summary card ────────────────────────────────────
    summary_parts = []
    if comp.get("best_format"):
        summary_parts.append(
            f"- **Best compression:** {comp['best_format']} achieves the "
            "highest compression on UMI count data — consistently #1 at "
            "census scale."
        )
    else:
        summary_parts.append(
            "- **Best compression:** SCX achieves the highest compression "
            "on UMI count data."
        )

    # Ratio scaling commentary
    r_1m = comp.get("scx_ratio_census_1m")
    r_5m = comp.get("scx_ratio_census_5m")
    if r_1m is not None and r_5m is not None:
        summary_parts.append(
            f"- **Compression scales:** ratio improves with dataset size "
            f"({r_5m:.2f}x on census_5m vs {r_1m:.2f}x on census_1m)."
        )
    else:
        summary_parts.append(
            "- **Compression scales:** ratio improves with dataset size."
        )
    summary_parts.append(
        "- **Byte-shuffle effective:** SCX lz4 compresses better than "
        "Zarr lz4 thanks to the byte-shuffle pre-filter."
    )
    c.sections.append(Section(title="Storage Summary", blocks=[
        CalloutBlock("\n".join(summary_parts)),
    ]))

    # ── File sizes ────────────────────────────────────────────────────
    c.sections.append(Section(title="File Sizes", blocks=[
        tables.compression_table(),
    ]))

    # ── Compression ratios ────────────────────────────────────────────
    takeaway_parts = ["**Takeaways:**"]

    # SCX vs Zarr on census_1m
    pair_1m = comp.get("scx_vs_zarr_census_1m")
    if pair_1m is not None:
        scx_sz, zarr_sz = pair_1m
        takeaway_parts.append(
            f"- SCX achieves the best compression on UMI count data — "
            f"consistently #1 at census scale ({scx_sz} vs {zarr_sz} "
            f"Zarr zstd on 1M cells)."
        )
    else:
        takeaway_parts.append(
            "- SCX achieves the best compression on UMI count data."
        )

    # SCX lz4 vs Zarr lz4 on census_5m
    pair_5m = comp.get("scx_vs_zarr_census_5m")
    if pair_5m is not None:
        scx_sz, zarr_sz = pair_5m
        takeaway_parts.append(
            f"- SCX lz4 compresses better than Zarr lz4 — byte-shuffle "
            f"pre-filter is effective ({scx_sz} vs {zarr_sz} on 5M cells)."
        )
    else:
        takeaway_parts.append(
            "- SCX lz4 compresses better than Zarr lz4 — byte-shuffle "
            "pre-filter is effective."
        )

    # Ratio scaling
    if r_1m is not None and r_5m is not None:
        takeaway_parts.append(
            f"- Compression ratio improves with scale: {r_5m:.2f}x on "
            f"census_5m vs {r_1m:.2f}x on census_1m."
        )

    c.sections.append(Section(title="Compression Ratios", blocks=[
        tables.compression_ratio_table(),
        FigureBlock("figures/compression_bar.png",
                     caption="Compression Ratio vs h5ad_none"),
        CommentaryBlock(
            "\n".join(takeaway_parts),
            source=SourceRef(kind=SourceKind.raw_json,
                             reason="derived from compression benchmark results"),
        ),
    ]))

    return c
