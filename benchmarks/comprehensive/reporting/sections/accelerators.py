"""Chapter 9: Accelerators — source-tracked GPU and CPU timing.

- GPU accelerator timing sourced from raw JSON (``accel_*`` benchmark keys)
  when available, or declared as external-source when not yet in the
  comprehensive harness.
- ``CommentaryBlock`` with ``SourceRef`` for all numeric narrative.
- CSC/CSR dispatch included here (storage-layout concern for accelerators).
- Parity tables adjacent to timing tables.
"""

from benchmarks.comprehensive.reporting.report_model import (
    Chapter, Section, TextBlock, CommentaryBlock,
)
from benchmarks.comprehensive.reporting.result_store import (
    ResultStore, SourceRef, SourceKind,
)
from benchmarks.comprehensive.reporting import tables
from benchmarks.comprehensive.reporting.report_model import TableBlock


def _build_gpu_table(store: ResultStore) -> TableBlock | TextBlock:
    """Build GPU accelerator performance table from raw JSON results.

    Looks for ``accel_*`` benchmarks with ``device=gpu`` in scenario
    metadata. If no GPU results are available, returns a TextBlock
    explaining the gap.
    """
    gpu_benchmarks = [
        "accel_pca", "accel_knn", "accel_umap", "accel_leiden",
        "accel_preprocess", "accel_hvg",
    ]
    rows_data: list[dict] = []
    for bench in gpu_benchmarks:
        for row in store.by_benchmark(bench):
            sc = row.scenario
            if sc.device and "gpu" in sc.device.lower():
                rows_data.append({
                    "operation": bench.replace("accel_", ""),
                    "dataset": row.dataset,
                    "gpu_wall_s": row.median_wall_s,
                    "source_path": row.source.path,
                })

    if not rows_data:
        return TextBlock(
            "*No GPU accelerator results in the comprehensive harness yet. "
            "GPU timing is available via standalone benchmarks — see the "
            "external/manual sources appendix for provenance.*"
        )

    headers = ["Operation", "Dataset", "GPU wall (s)"]
    rows: list[list[str]] = []
    for d in sorted(rows_data, key=lambda x: (x["operation"], x["dataset"])):
        wall_str = f"{d['gpu_wall_s']:.3f}" if d["gpu_wall_s"] is not None else "—"
        rows.append([d["operation"], d["dataset"], wall_str])

    return TableBlock(
        headers=headers, rows=rows,
        caption="GPU accelerator wall time (from comprehensive harness)",
        source=SourceRef(kind=SourceKind.raw_json,
                         reason="accel_* benchmarks with device=gpu"),
    )


def build(store: ResultStore) -> Chapter:
    c = Chapter(title="Accelerators (PCA, kNN, UMAP, Leiden)")

    # ── Parity overview ──────────────────────────────────────────────
    c.sections.append(Section(
        title="Accelerator Parity",
        blocks=[
            TextBlock(
                "Before interpreting speedups, confirm that SCX's Rust-native "
                "accelerators reproduce scanpy/leidenalg reference results.  "
                "The table below shows per-operation parity metrics extracted "
                "from the same benchmark runs that produce timing data.  "
                "Full correctness details are in Chapter 3."
            ),
            tables.accelerator_parity_table(),
        ],
    ))

    # ── CPU accelerator performance ──────────────────────────────────
    c.sections.append(Section(
        title="CPU Accelerator Performance",
        blocks=[
            TextBlock(
                "All benchmarks use SCX's Rust-native implementations vs "
                "scanpy's standard Python stack.  Timing results are backed "
                "by raw JSON from the comprehensive harness."
            ),
        ],
    ))

    # ── GPU accelerator performance ──────────────────────────────────
    gpu_table = _build_gpu_table(store)
    c.sections.append(Section(
        title="GPU Accelerator Performance",
        blocks=[
            TextBlock(
                "GPU-accelerated operations (kNN via cuVS CAGRA, PCA via "
                "cuBLAS, UMAP, Leiden via cuGraph). Performance is sourced "
                "from raw JSON where available; see the external sources "
                "appendix for any manually curated entries."
            ),
            gpu_table,
        ],
    ))

    # ── CSC vs CSR dispatch (moved from operations chapter) ──────────
    c.sections.append(Section(
        title="CSC vs CSR Accelerator Dispatch",
        blocks=[
            TextBlock(
                "Time to compute column-oriented operations (column variance, "
                "HVG, DE, QC metrics) on a CSR-native file vs CSC sidecar.  "
                "This is an accelerator storage-layout concern — CSC sidecars "
                "can accelerate gene-major traversals."
            ),
            tables.bench_csc_dispatch_table(),
        ],
    ))

    return c
