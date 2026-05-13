from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="File Operations & Dispatch")
    c.sections.append(Section(title="Fragment & Manifest Operations", blocks=[
        TextBlock("Time to execute append/subset operations on SCX fragments."),
        tables.fragment_ops_table()
    ]))
    c.sections.append(Section(title="CSC / CSR Dispatch", blocks=[
        TextBlock("Time to compute column variance (requires CSC traversal) on a CSR-native file vs CSC-native file."),
        tables.bench_csc_dispatch_table()
    ]))
    return c

