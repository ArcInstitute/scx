from benchmarks.comprehensive.reporting.report_model import Chapter, Section, TextBlock, TableBlock, CalloutBlock, FigureBlock
from benchmarks.comprehensive.reporting.result_store import ResultStore
from benchmarks.comprehensive.reporting import tables

def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Cloud Storage (GCS / S3)")

    filtered_blocks = [
        TextBlock("Filtering 5% of cells directly from object storage without downloading the full dataset."),
    ]
    filtered_blocks.extend(tables.cloud_filtered_table())
    filtered_blocks.append(TextBlock("**Takeaways:**\n- SCX is the only format that can push down query predicates to the cloud storage layer."))
    c.sections.append(Section(title="Cloud Filtered Read", blocks=filtered_blocks))

    pull_blocks = [
        TextBlock("Reading from object storage directly vs `aws s3 cp` to local disk first."),
    ]
    pull_blocks.extend(tables.cloud_reader_vs_pull_table())
    pull_blocks.append(TextBlock("**Takeaways:**\n- Direct cloud reading with SCX is faster than pulling the file to local NVMe and reading locally."))
    c.sections.append(Section(title="Cloud Read vs Pull", blocks=pull_blocks))

    c.sections.append(Section(title="Cloud Operations Matrix", blocks=[
        tables.gcp_matrix_table()
    ]))

    cost_blocks: list = []
    cost_blocks.extend(tables.cost_model_table())
    c.sections.append(Section(title="Cost Model", blocks=cost_blocks))

    return c

