# Using SCX with scanpy (moved)

This guide now lives in [`docs/scanpy/`](scanpy/README.md), split into one page
per task. Start with the [index](scanpy/README.md) or the
[quick start](scanpy/quickstart.md).

Where each section of the old single-page guide went:

| Old section | New page |
|-------------|----------|
| Choosing the right approach | [choosing-an-approach.md](scanpy/choosing-an-approach.md) |
| Quick start: in-memory / backed mode, Dependencies | [quickstart.md](scanpy/quickstart.md) |
| Converting existing data to SCX (incl. exporting to MTX / h5ad / h5mu) | [conversion.md](scanpy/conversion.md) |
| Landing external per-cell annotations (doublet detection) | [external-annotations.md](scanpy/external-annotations.md) |
| Understanding `to_anndata()`, Querying subsets before loading | [loading.md](scanpy/loading.md) |
| Backed mode (lazy loading), Out-of-core chunk iteration | [backed-mode.md](scanpy/backed-mode.md) |
| Lazy preprocessing in backed mode, Preprocessing pipeline (write-back) | [lazy-preprocessing.md](scanpy/lazy-preprocessing.md) |
| Rust-native accelerators (overview, compatibility matrix) | [accelerators.md](scanpy/accelerators.md) |
| `prefer_format` column-major dispatch | [accel-csc.md](scanpy/accel-csc.md) |
| GPU-supported vs GPU-fast, GPU data layout, GPU helpers, numerical differences, tolerances, checking the backend | [accel-gpu.md](scanpy/accel-gpu.md) |
| PCA, kNN, fused PCA → kNN (→ UMAP), UMAP, Leiden | [accel-embedding-clustering.md](scanpy/accel-embedding-clustering.md) |
| Harmony2, LISI | [accel-integration.md](scanpy/accel-integration.md) |
| Gene-set scoring, PFlog normalization | [accel-scoring-normalization.md](scanpy/accel-scoring-normalization.md) |
| Differential expression, pseudobulk, stratified DE, NB-GLM | [accel-differential-expression.md](scanpy/accel-differential-expression.md) |
| Perturbation evaluation metrics (cell-eval parity) | [accel-perturbation-metrics.md](scanpy/accel-perturbation-metrics.md) |
| Multithreading | [threading.md](scanpy/threading.md) |
| Common scanpy workflows | [workflows.md](scanpy/workflows.md) |
| Validating files, File operations with scanpy, File inspection | [file-operations.md](scanpy/file-operations.md) |
| ML training data loading | [ml-training.md](scanpy/ml-training.md) |
