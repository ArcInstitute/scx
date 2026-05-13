from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Correctness & Equivalence")
    c.sections.append(Section(title="Format Validation", blocks=[
        tables.correctness_table(),
        TextBlock("**Takeaways:**\n- SCX is 14/14 on equivalence tests.\n- Zarr fails to preserve sparse matrices (CSC format coercion).")
    ]))
    c.sections.append(Section(title="Pipeline Agreement (pbmc3k)", blocks=[
        tables.correctness_detail_table(),
        TextBlock("**Takeaways:**\n- Leiden ARI > 0.80 across all pipelines.\n- Preprocessing max error < 1e-6.")
    ]))
    return c

