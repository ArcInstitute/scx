from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="I/O Performance")
    c.sections.append(Section(title="Read Performance (Full)", blocks=[
        tables.read_speed_table(),
        FigureBlock("figures/read_speed_bar.png"),
        FigureBlock("figures/scaling_curves.png"),
        TextBlock("**Takeaways:**\n- **SCX is the fastest reader at census scale** — 1.38x faster than Zarr lz4 on 1M cells, 1.15x on 5M cells. Crossover at ~100K cells.\n- Read time scales sub-linearly with cell count due to shard-level parallelism.")
    ]))
    c.sections.append(Section(title="Read Performance (Selective / Query)", blocks=[
        TextBlock("Column projection: 2,000 HVG columns selected from full gene set."),
        tables.read_selective_table(),
        TextBlock("**Takeaways:**\n- SCX dominates column projection — **4.2x faster** than Zarr on census_1m, **7.6x faster** on census_5m.\n- Shard-level predicate pushdown enables SCX to skip irrelevant data on disk.")
    ]))
    c.sections.append(Section(title="Write Performance", blocks=[
        TextBlock("All competing formats (Zarr, h5ad, TileDB-SOMA) write single-threaded — they cannot scale across cores. SCX parallelises shard encoding via rayon."),
        tables.write_speed_table(),
        TextBlock("### With parallel shard encoding (SCX only)\nWriting the same h5ad → SCX pipeline with 32 rayon threads."),
        tables.scx_parallel_write_callout_table(),
        TextBlock("**Takeaways:**\n- **Apples-to-apples, SCX is competitive at 32 threads.**\n- **SCX is the only format that scales writes with cores.**\n- **Compression-heavy SCX codecs scale best.**\n- **h5ad gzip and TileDB-SOMA are significantly slower at any thread count.**")
    ]))
    c.sections.append(Section(title="Memory Efficiency (Peak RSS)", blocks=[
        TextBlock("Memory footprint during full dataset read into memory."),
        tables.memory_table(),
        TextBlock("**Takeaways:**\n- **SCX has the lowest memory footprint.**\n- Zarr requires ~1.5x more memory than SCX.\n- TileDB-SOMA memory scaling is unbounded.")
    ]))
    scaling_blocks = [
        TextBlock("Wall-time scaling from 1 to 32 threads. Value is the speedup factor vs 1 thread (1.0x)."),
    ]
    scaling_blocks.extend(tables.parallel_scaling_table())
    scaling_blocks.extend(tables.parallel_write_scaling_table())
    c.sections.append(Section(title="Parallel Scaling", blocks=scaling_blocks))
    return c

