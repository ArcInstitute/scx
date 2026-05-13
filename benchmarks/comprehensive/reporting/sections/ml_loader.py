from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="ML DataLoader")
    c.sections.append(Section(title="Training Pipeline Throughput", blocks=[
        TextBlock("Simulated training epoch: 256 batch size, random row sampling, densification. Metric: batches/second."),
        tables.ml_loader_table(),
        FigureBlock("figures/ml_loader_bar.png"),
        TextBlock("**Takeaways:**\n- SCX `TrainingDataset` delivers **1,405 batches/sec** on 1M cells — 82x faster than TileDB-SOMA-ML.\n- Zero-copy Rust-to-Python transfer ensures Python GIL is not a bottleneck.")
    ]))
    return c

