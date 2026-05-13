from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Methodology")
    c.sections.append(Section(title="Datasets", blocks=[
        tables.datasets_table(),
        TextBlock("Dataset paths configured via `SCX_WORK_DIR` / `SCX_DATA_DIR` environment variables. All benchmarks use warm-cache (1 warm-up read discarded) unless otherwise noted.")
    ]))
    c.sections.append(Section(title="Test Environment", blocks=[
        tables.system_info_table(),
        TextBlock("All benchmarks run on Arc Institute's Chimera HPC cluster. Intel Xeon Platinum 8468, 48 cores / 96 threads per socket, 1007–2015 GB RAM, WekaFS NVMe-backed parallel filesystem. GPU benchmarks on NVIDIA H100 80GB HBM3.")
    ]))
    return c

