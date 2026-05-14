"""Chapter 4: Coverage — benchmark and API coverage summary."""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock,
)
from benchmarks.comprehensive.reporting.result_store import ResultStore


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Coverage")

    # Coverage matrix data is not yet emitted by the harness; provide a
    # qualitative summary until the automated matrix is available.
    c.sections.append(Section(title="API Coverage", blocks=[
        TextBlock(
            "API coverage matrices (format × operation) will be generated "
            "automatically once the harness emits per-operation coverage "
            "records.  In the meantime, correctness and equivalence results "
            "in Chapter 3 serve as the primary coverage signal."
        ),
    ]))
    return c

