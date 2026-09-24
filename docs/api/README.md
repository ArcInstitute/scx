# SCX API Reference

The API reference for every SCX surface: the Python bindings (`pyscx`), the
Rust crates, the R bindings (`rscx`), and the `scx` CLI. It is split into pages
by surface. New here? Start with the [API quick start](quickstart.md).

For task-oriented guides rather than reference material, see
[docs/scanpy/](../scanpy/README.md) (analysis with scanpy),
[docs/training.md](../training.md) (ML training), and
[docs/quickstart.md](../quickstart.md) (a 5-minute end-to-end pipeline). The
[auto-generated Python API reference](../python_api.rst) carries the canonical
per-function docstrings.

## Getting started

- [API quick start](quickstart.md) — the handful of calls most sessions need,
  in Python, Rust, and on the CLI, with links to the full reference for each.

## Python (`pyscx`)

| Page | What it covers |
|------|----------------|
| [Python API overview](python.md) | Where canonical docstrings live, and the restricted-exec (sandbox) guarantee |
| [Module-level functions](python-functions.md) | `pyscx.open` / `read` / `write`, conversion (`from_anndata`, `from_h5ad`, `to_h5ad`, …), annotation import, preprocessing, file operations, and cloud operations |
| [`Experiment`](python-experiment.md) | The file handle: `to_anndata`, `to_gpu_anndata`, `to_mudata`, handle staleness, container / dtype materialization, `uns` serialization |
| [Queries and cloud handles](python-query.md) | `PyQueryPipeline`, `PyQueryResult`, `CloudExperiment`, and the per-surface capability matrix |
| [`pyscx.accel`](python-accel.md) | Accelerator signatures, `prefer_format`, `gene_chunk_size`, axis subsetting, and route metadata |
| [Backed and lazy datasets](python-datasets.md) | `ScxBackedSparseDataset`, `ScxBackedLayerDataset`, `ScxComparisonResult`, `ScxLazyTransformedDataset` |
| [ML training and tokenisation](python-training.md) | scVI integration, `TrainingDataset`, `IndexPlanDataset`, fork safety, `SparseCellSetDataset`, `pyscx.tokenize`, neighbourhood plans |

## Conversion

| Page | What it covers |
|------|----------------|
| [Conversion and round-trip fidelity](conversion.md) | `ConvertWarning`, the round-trip fidelity table, `adata.raw`, and `from_anndata` with backed / lazy `X` |
| [Memory budgets](memory-budgets.md) | `--memory-budget`, the allocation table, and ops that bound themselves |
| [Predicate indexes and detection bitmaps](indexes.md) | Conversion-time `--index-*` and `--bitmap` |

## Rust crates

| Page | What it covers |
|------|----------------|
| [Section types](section-types.md) | The on-disk section type IDs |
| [Format I/O](rust-format-io.md) | `ScxReader`, `ScxWriter`, codec selection, provenance, `BackedCsrReader` / `BackedCscReader`, and the `ShardSource` traits |
| [Multimodal API](multimodal.md) | Modality types and accessors across Rust, Python, R, and the CLI |
| [File operations (`scx-ops`)](rust-ops.md) | `append`, `mark_deleted`, `compact`, `optimize`, `rollback`, `merge`, `sort`, metadata edits |
| [Query engine (`scx-engine`)](rust-engine.md) | `QueryPipeline`, predicate pushdown, grouped reads |
| [Training loader and cloud](rust-loader-cloud.md) | `TrainingPipeline` and `scx-cloud` operations |
| [GPU (`scx-gpu`)](rust-gpu.md) | cuSPARSE, cuSOLVER, cuRAND, GPU PCA / kNN / preprocessing, `GpuError` |

## CLI

- [CLI (`scx`)](cli.md) — every `scx` subcommand and its flags.
