from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Appendix")
    c.sections.append(Section(title="Glossary", blocks=[
        TextBlock("- **SCX:** Sparse Cell eXpression System.\n- **h5ad:** AnnData's HDF5-based serialization format.\n- **TileDB-SOMA:** Single-cell Open Matrix Architecture by TileDB.\n- **Zarr:** Cloud-native multidimensional array format.")
    ]))
    return c

