from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Storage Efficiency")
    c.sections.append(Section(title="File Sizes", blocks=[
        tables.compression_table()
    ]))
    c.sections.append(Section(title="Compression Ratios", blocks=[
        tables.compression_ratio_table(),
        FigureBlock("figures/compression_bar.png", caption="Compression Ratio vs h5ad_none"),
        TextBlock("**Takeaways:**\n- SCX pcodec/zstd achieve the best compression on UMI count data — consistently #1 at census scale (2.35 GB vs 2.60 GB Zarr zstd on 1M cells).\n- SCX lz4 compresses better than Zarr lz4 — byte-shuffle pre-filter is effective (13.85 GB vs 17.06 GB on 5M cells).\n- Compression ratio improves with scale: 7.30x on census_5m vs 4.90x on census_1m.")
    ]))
    return c

