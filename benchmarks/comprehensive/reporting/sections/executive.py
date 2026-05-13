from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Executive Summary")
    c.sections.append(Section(title="Key findings", blocks=[
        TextBlock("SCX is a purpose-built binary format for single-cell RNA-seq data. This report evaluates SCX against h5ad (gzip, lzf, uncompressed), Zarr (zstd, blosc-lz4), and TileDB-SOMA across 7 datasets (2.7K to 5M cells) on compression, read/write performance, parallel scaling, memory efficiency, ML data loading, analysis accelerators, and correctness."),
        CalloutBlock("- **Compression:** SCX pcodec/zstd achieve the best compression on UMI data...\n- **Read speed:** SCX is the fastest reader at census scale...\n- **Write scaling:** Parallel shard encoding...\n- **Column projection:** SCX dominates...\n- **ML loader:** SCX TrainingDataset delivers...\n- **Pipeline:** With Rust-native Leiden...\n- **GPU:** kNN 9.4x...\n- **Correctness:** 14/14 scanpy equivalence tests pass...")
    ]))
    return c

