from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Accelerators (PCA, kNN, UMAP, Leiden)")
    c.sections.append(Section(title="Native vs Python Performance", blocks=[
        TextBlock("All benchmarks use SCX's Rust-native or GPU-accelerated implementations vs scanpy's standard Python stack."),
        TextBlock("This section is backed by raw JSON results but currently hardcoded in markdown.py. (Will be fixed in Phase 7).")
    ]))
    return c

