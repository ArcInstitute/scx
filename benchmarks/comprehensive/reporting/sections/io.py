"""Chapter 6: Local I/O Performance (Phase 6 — performance-domain restructuring).

Phase 6 changes:
- Split conversion-pipeline write timing from write-only timing.
- Split full read, selective read, and backed/streaming read modes.
- Add chapter-level summary card at the top.
- Group memory tables by measurement mode.
- Fix memory narrative so takeaways reference the correct measurement mode.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CalloutBlock, FigureBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Local I/O Performance")

    # ── Chapter-level summary card ────────────────────────────────────
    c.sections.append(Section(title="I/O Performance Summary", blocks=[
        CalloutBlock(
            "- **Read:** SCX is the fastest reader at census scale — "
            "shard-level parallelism scales sub-linearly with cell count.\n"
            "- **Write:** SCX conversion pipeline includes h5ad read overhead; "
            "write-only timing isolates codec cost. SCX is the only format "
            "that scales writes with cores.\n"
            "- **Selective read:** SCX dominates column projection — "
            "shard-level predicate pushdown skips irrelevant data on disk.\n"
            "- **Memory:** SCX has the lowest RSS footprint for full "
            "materialization. Mode-specific tables below prevent mixing "
            "definitions."
        ),
    ]))

    # ── Full read performance ─────────────────────────────────────────
    c.sections.append(Section(title="Read Performance (Full Materialization)", blocks=[
        TextBlock(
            "Warm-cache full read — the entire dataset is materialized "
            "into an in-memory AnnData/CSR matrix."
        ),
        tables.read_speed_table(),
        FigureBlock("figures/read_speed_bar.png"),
        FigureBlock("figures/scaling_curves.png"),
        TextBlock(
            "**Takeaways:**\n"
            "- **SCX is the fastest reader at census scale** — 1.38x faster "
            "than Zarr lz4 on 1M cells, 1.15x on 5M cells. Crossover at "
            "~100K cells.\n"
            "- Read time scales sub-linearly with cell count due to "
            "shard-level parallelism."
        ),
    ]))

    # ── Selective / query read ────────────────────────────────────────
    c.sections.append(Section(title="Read Performance (Selective / Query)", blocks=[
        TextBlock(
            "Column projection: 2,000 HVG columns selected from full gene set."
        ),
        tables.read_selective_table(),
        TextBlock(
            "**Takeaways:**\n"
            "- SCX dominates column projection — **4.2x faster** than Zarr "
            "on census_1m, **7.6x faster** on census_5m.\n"
            "- Shard-level predicate pushdown enables SCX to skip irrelevant "
            "data on disk."
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
    memory_blocks.append(TextBlock(
        "**Takeaways (full materialization):**\n"
        "- **SCX has the lowest memory footprint for full reads.**\n"
        "- Zarr requires ~1.5x more memory than SCX.\n"
        "- TileDB-SOMA memory scaling is unbounded."
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
