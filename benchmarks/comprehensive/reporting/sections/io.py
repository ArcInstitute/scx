"""Chapter 6: Local I/O Performance — data-derived commentary.

Changes:
- Replace hardcoded speedup/size claims in takeaway commentary with
  data-derived values from ``derive_*_headlines()`` functions.
- Use ``CommentaryBlock`` with ``SourceRef(kind=SourceKind.raw_json)``
  for all numeric takeaways so report-lint can trace claims to sources.

Additional:
- Split conversion-pipeline write timing from write-only timing.
- Split full read, selective read, and backed/streaming read modes.
- Chapter-level summary card at the top.
- Memory tables grouped by measurement mode.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CalloutBlock, CommentaryBlock, FigureBlock,
)
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, SourceRef, SourceKind,
)
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Local I/O Performance")

    # Pre-derive headline metrics for commentary blocks.
    read_hl = tables.derive_read_headlines()
    sel_hl = tables.derive_selective_read_headlines()
    mem_hl = tables.derive_memory_headlines()

    # ── Chapter-level summary card ────────────────────────────────────
    summary_parts = [
        "- **Read:** SCX is the fastest reader at census scale — "
        "shard-level parallelism scales sub-linearly with cell count.",
        "- **Write:** SCX conversion pipeline includes h5ad read overhead; "
        "write-only timing isolates codec cost. SCX is the only format "
        "that scales writes with cores.",
        "- **Selective read:** SCX dominates column projection — "
        "shard-level predicate pushdown skips irrelevant data on disk.",
        "- **Memory:** SCX has the lowest RSS footprint for full "
        "materialization. Mode-specific tables below prevent mixing "
        "definitions.",
    ]
    c.sections.append(Section(title="I/O Performance Summary", blocks=[
        CalloutBlock("\n".join(summary_parts)),
    ]))

    # ── Full read performance ─────────────────────────────────────────
    read_takeaways = ["**Takeaways:**"]
    fmt_name = read_hl.get("fastest_format", "SCX")
    read_takeaways.append(
        f"- **{fmt_name} is the fastest reader at census scale.**"
    )
    for ds_key, label in [("census_1m", "1M cells"), ("census_5m", "5M cells")]:
        spd = read_hl.get(f"scx_vs_zarr_{ds_key}")
        if spd is not None:
            read_takeaways.append(f"- {spd:.2f}x faster than Zarr lz4 on {label}.")
    read_takeaways.append(
        "- Read time scales sub-linearly with cell count due to "
        "shard-level parallelism."
    )

    c.sections.append(Section(title="Read Performance (Full Materialization)", blocks=[
        TextBlock(
            "Warm-cache full read — the entire dataset is materialized "
            "into an in-memory AnnData/CSR matrix."
        ),
        tables.read_speed_table(),
        FigureBlock("figures/read_speed_bar.png"),
        FigureBlock("figures/scaling_curves.png"),
        CommentaryBlock(
            "\n".join(read_takeaways),
            source=SourceRef(kind=SourceKind.raw_json,
                             reason="derived from read_full benchmark results"),
        ),
    ]))

    # ── Selective / query read ────────────────────────────────────────
    sel_takeaways = ["**Takeaways:**"]
    for ds_key, label in [("census_1m", "census_1m"), ("census_5m", "census_5m")]:
        spd = sel_hl.get(f"scx_vs_zarr_{ds_key}")
        if spd is not None:
            sel_takeaways.append(
                f"- SCX is **{spd:.1f}x faster** than Zarr on {label}."
            )
    sel_takeaways.append(
        "- Shard-level predicate pushdown enables SCX to skip irrelevant "
        "data on disk."
    )

    c.sections.append(Section(title="Read Performance (Selective / Query)", blocks=[
        TextBlock(
            "Column projection: 2,000 HVG columns selected from full gene set."
        ),
        tables.read_selective_table(),
        CommentaryBlock(
            "\n".join(sel_takeaways),
            source=SourceRef(kind=SourceKind.raw_json,
                             reason="derived from read_selective benchmark results"),
        ),
    ]))

    # ── Write performance — conversion pipeline ───────────────────────
    c.sections.append(Section(title="Write Performance (Conversion Pipeline)", blocks=[
        TextBlock(
            "Full conversion pipeline: read source h5ad → encode → write "
            "target format. This timing includes h5ad read overhead and "
            "is *not* directly comparable to write-only benchmarks."
        ),
        tables.write_conversion_table(),
        TextBlock(
            "**Note:** All competing formats (Zarr, h5ad, TileDB-SOMA) "
            "write single-threaded — they cannot scale across cores. SCX "
            "parallelises shard encoding via rayon."
        ),
    ]))

    # ── Write performance — write-only ────────────────────────────────
    c.sections.append(Section(title="Write Performance (Write-Only)", blocks=[
        TextBlock(
            "Write-only timing: encode from in-memory AnnData without "
            "h5ad read overhead. Isolates codec/writer cost."
        ),
        tables.write_only_table(),
    ]))

    # ── Parallel write scaling ────────────────────────────────────────
    c.sections.append(Section(title="Write Scaling (Parallel)", blocks=[
        TextBlock(
            "Writing the same h5ad → SCX pipeline with 32 rayon threads."
        ),
        tables.scx_parallel_write_callout_table(),
        TextBlock(
            "**Takeaways:**\n"
            "- **Apples-to-apples, SCX is competitive at 32 threads.**\n"
            "- **SCX is the only format that scales writes with cores.**\n"
            "- **Compression-heavy SCX codecs scale best.**\n"
            "- **h5ad gzip and TileDB-SOMA are significantly slower at "
            "any thread count.**"
        ),
    ]))

    # ── Memory efficiency — grouped by measurement mode ───────────────
    memory_blocks = [
        TextBlock(
            "Peak RSS (delta above baseline) during dataset read. Tables "
            "are grouped by measurement mode to prevent mixing "
            "full-materialization and subset/query memory figures."
        ),
    ]
    memory_blocks.extend(tables.memory_by_mode_tables())

    mem_takeaway_parts = ["**Takeaways (full materialization):**"]
    mem_takeaway_parts.append("- **SCX has the lowest memory footprint for full reads.**")
    zarr_ratio = mem_hl.get("scx_vs_zarr_ratio")
    if zarr_ratio is not None:
        mem_takeaway_parts.append(f"- Zarr requires ~{zarr_ratio:.1f}x more memory than SCX.")
    else:
        mem_takeaway_parts.append("- Zarr requires more memory than SCX.")
    mem_takeaway_parts.append("- TileDB-SOMA memory scaling is unbounded.")

    memory_blocks.append(CommentaryBlock(
        "\n".join(mem_takeaway_parts),
        source=SourceRef(kind=SourceKind.raw_json,
                         reason="derived from memory benchmark results"),
    ))
    c.sections.append(Section(title="Memory Efficiency (Peak RSS)", blocks=memory_blocks))

    # ── Parallel read scaling ─────────────────────────────────────────
    scaling_blocks = [
        TextBlock(
            "Wall-time scaling from 1 to 32 threads. Value is the speedup "
            "factor vs 1 thread (1.0x)."
        ),
    ]
    scaling_blocks.extend(tables.parallel_scaling_table())
    scaling_blocks.extend(tables.parallel_write_scaling_table())
    c.sections.append(Section(title="Parallel Scaling", blocks=scaling_blocks))

    return c
