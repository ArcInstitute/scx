# SCX API Reference (moved)

The API reference now lives in [`docs/api/`](api/README.md), split into one page
per surface. Start with the [index](api/README.md) or the
[API quick start](api/quickstart.md).

Where each section of the old single-page reference went:

| Old section | New page |
|-------------|----------|
| Section Types | [section-types.md](api/section-types.md) |
| ScxReader, ScxWriter, Codec Selection, Provenance, BackedCsrReader, ShardSource, BackedCscReader, ColumnShardSource | [rust-format-io.md](api/rust-format-io.md) |
| Multimodal API | [multimodal.md](api/multimodal.md) |
| Conversion warnings, Round-trip fidelity, `adata.raw`, `pyscx.from_anndata` — backed and lazy `X` | [conversion.md](api/conversion.md) |
| Memory budgets | [memory-budgets.md](api/memory-budgets.md) |
| Conversion-time predicate indexes and detection bitmaps | [indexes.md](api/indexes.md) |
| scx-ops — File Operations | [rust-ops.md](api/rust-ops.md) |
| scx-engine — Query Engine | [rust-engine.md](api/rust-engine.md) |
| scx-loader, scx-cloud | [rust-loader-cloud.md](api/rust-loader-cloud.md) |
| scx-gpu — GPU Analysis | [rust-gpu.md](api/rust-gpu.md) |
| Python API (`pyscx`) intro, Restricted-exec (sandbox) safety | [python.md](api/python.md) |
| Module-level functions, File operations, Cloud operations | [python-functions.md](api/python-functions.md) |
| Experiment, Handles and files that change underneath them, Container and dtype materialization, `uns` serialization | [python-experiment.md](api/python-experiment.md) |
| PyQueryPipeline, PyQueryResult, CloudExperiment, Per-surface capability matrix | [python-query.md](api/python-query.md) |
| pyscx.accel — Rust-Native Accelerators (incl. Accelerator route metadata) | [python-accel.md](api/python-accel.md) |
| ScxBackedSparseDataset, ScxBackedLayerDataset, ScxComparisonResult, ScxLazyTransformedDataset | [python-datasets.md](api/python-datasets.md) |
| scVI Integration, TrainingDataset, IndexPlanDataset, Fork safety, SparseCellSetDataset, Tokenisation kernels, Neighbourhood plans | [python-training.md](api/python-training.md) |
| CLI (`scx`) | [cli.md](api/cli.md) |
