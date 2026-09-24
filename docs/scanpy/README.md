# Using SCX with scanpy

SCX integrates directly with [scanpy](https://scanpy.readthedocs.io/) and the
[scverse](https://scverse.org/) ecosystem through `pyscx`. Every `pyscx` method
that returns data produces a standard `anndata.AnnData` object, so any scanpy
function works out of the box with zero glue code.

> Migrating an existing h5ad workflow? Start with
> [docs/migrating-from-h5ad.md](../migrating-from-h5ad.md) — it has the loader
> decision tree, the round-trip fidelity table, and the handful of
> scanpy-divergence gotchas consolidated in one place.

This guide is split into pages by task. New to SCX? Start with the
[quick start](quickstart.md), then pick a loader in
[Choosing the right approach](choosing-an-approach.md).

## Getting started

| Page | What it covers |
|------|----------------|
| [Quick start](quickstart.md) | Install, convert an h5ad, and run an in-memory or backed scanpy pipeline in a few minutes |
| [Choosing the right approach](choosing-an-approach.md) | Decision tree, feature comparison, and pros / cons of in-memory, backed, and query loading |
| [Common scanpy workflows](workflows.md) | End-to-end recipes: clustering, accelerated and GPU pipelines, DE, batch integration |

## Getting data in and out

| Page | What it covers |
|------|----------------|
| [Converting existing data to SCX](conversion.md) | `from_anndata`, streaming `from_h5ad`, 10x HDF5, Cell Ranger MTX, `adata.raw`, and exporting back to MTX / h5ad / h5mu |
| [Loading SCX data into AnnData](loading.md) | `to_anndata()` in full: signature, memory cost, dtype narrowing, gene / cell / layer / modality selection, and the query pipeline |
| [Backed mode and out-of-core iteration](backed-mode.md) | `to_anndata(backed=True)`, indexing patterns, which scanpy ops work backed, caching, and chunk iteration |
| [Lazy preprocessing](lazy-preprocessing.md) | Lazy `normalize_total` / `log1p` chains, the complete out-of-core pipeline, and writing preprocessed data back to SCX |
| [Landing external annotations](external-annotations.md) | Importing doublet-caller output and other per-cell / per-gene tables with `obs_import` / `var_import` |
| [File operations](file-operations.md) | Inspecting and validating files, removing doublets, appending batches, and merging datasets |

## Rust-native accelerators (`pyscx.accel`)

| Page | What it covers |
|------|----------------|
| [Accelerators overview](accelerators.md) | The `device=` selector, axis-subsetting ops, the compatibility matrix, and mixing accelerators with scanpy |
| [Column-major (CSC) dispatch](accel-csc.md) | `prefer_format="auto"\|"csr"\|"csc"` and when the CSC route pays off |
| [GPU acceleration](accel-gpu.md) | GPU-supported vs GPU-fast, data layout for device decode, GPU vs CPU numerical differences, tolerance thresholds, and checking which backend ran |
| [PCA, kNN, UMAP, and Leiden](accel-embedding-clustering.md) | Embedding and clustering accelerators, including the fused PCA → kNN (→ UMAP) paths |
| [Batch integration and LISI](accel-integration.md) | Harmony2 and local batch-mixing scores |
| [Gene-set scoring and PFlog](accel-scoring-normalization.md) | `score_genes` and PFlog normalization |
| [Differential expression](accel-differential-expression.md) | Wilcoxon `rank_genes_groups`, `pdex_ref`, pseudobulk DE, stratified DE, and the NB-GLM backend |
| [Perturbation evaluation metrics](accel-perturbation-metrics.md) | cell-eval-parity metrics: pseudobulk means, discrimination, energy distance, knockdown efficiency, clustering agreement |
| [Multithreading](threading.md) | Per-function threading, controlling parallelism, and GPU DE device residency |

## Beyond scanpy

- [ML training data loading](ml-training.md) — a short tour of the training
  loaders; the full guide is [docs/training.md](../training.md).
- [docs/api/](../api/README.md) — the complete Python and Rust API reference.
- [docs/migrating-from-h5ad.md](../migrating-from-h5ad.md) — moving an existing
  h5ad workflow to SCX.
- [docs/gpu-setup.md](../gpu-setup.md) — CUDA, RAPIDS, and SLURM setup for
  `device="gpu"`.
- [docs/multimodal.md](../multimodal.md) — CITE-seq / Multiome / spatial files and
  MuData.
