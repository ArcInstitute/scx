from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Specialized Workloads")
    c.sections.append(Section(title="Multimodal (CITE-seq / Multiome)", blocks=[
        tables.multimodal_compression_table(),
        tables.multimodal_compression_ratio_table(),
        tables.multimodal_training_table(),
        tables.multimodal_training_ttfb_table(),
    ]))
    harmony_blocks: list = []
    harmony_blocks.extend(tables.harmony_scaling_table())
    harmony_blocks.append(tables.lisi_comparison_table())
    harmony_blocks.append(tables.harmony_validation_table())
    c.sections.append(Section(title="Harmony & LISI", blocks=harmony_blocks))
    c.sections.append(Section(title="Cell-Eval / Arc-Bench Parity", blocks=[
        tables.cell_eval_parity_perf_table()
    ]))
    return c

