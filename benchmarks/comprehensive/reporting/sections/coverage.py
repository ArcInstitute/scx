from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Coverage")
    c.sections.append(Section(title="API Coverage", blocks=[
        TextBlock("API Coverage matrices will be implemented in Phase 4.")
    ]))
    return c

