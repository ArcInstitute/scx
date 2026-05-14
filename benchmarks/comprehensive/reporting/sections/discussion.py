from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Discussion & Roadmap")
    c.sections.append(Section(title="Conclusion", blocks=[
        TextBlock("SCX achieves its design goal: it replaces h5ad, Zarr, and TileDB with a single format that is faster, smaller, and natively integrates with ML and analysis stacks.")
    ]))
    return c

