# pyscx — Python bindings for SCX

`pyscx` is the Python binding for **SCX** (Sparse Cell eXpression System), a
purpose-built binary format, query engine, and ML loader for single-cell
RNA-seq data. It is a PyO3 extension built with maturin, exporting a
scverse-compatible API: every `pyscx` reader returns a standard
`anndata.AnnData`, so existing `scanpy` / `scVI` code keeps working unchanged.

For the overall project, file format, and benchmarks see the
[top-level README](../README.md) and [docs/architecture.md](../docs/architecture.md).

## What `pyscx` provides

- **Readers and writers** — `pyscx.open(path)` and the streaming
  `pyscx.from_h5ad` / `pyscx.from_10x` / `pyscx.from_mudata`
  ingestion helpers, plus `to_h5ad` / `to_h5mu` for export.
- **Backed `AnnData`** — `open(...).to_anndata(backed=True)` returns an
  `AnnData` whose `X` stays on disk; only the shards you touch are
  decoded. See `ScxBackedSparseDataset` in
  [python/pyscx/__init__.py](python/pyscx/__init__.py).
- **Rust-native accelerators (`pyscx.accel.*`)** — drop-in replacements
  for `sc.pp.*` / `sc.tl.*`: `normalize_total`, `log1p`,
  `highly_variable_genes`, `pca`, `neighbors`, `umap`, `leiden`, `rank_genes_groups`,
  `calculate_qc_metrics`, `harmony2`, and more. Every op takes
  `device="auto" | "cpu" | "gpu"` and falls back to CPU when no GPU
  is available.
- **Lazy query pipeline** — `open(path).query().filter_obs(...).select_genes(...).with_normalize(...).with_log1p().collect()`
  pushes predicates into the Rust engine and skips non-matching
  shards entirely.
- **Cloud-native reads** — `pyscx.open_cloud("gs://…/atlas.scx")` and
  the matching query pipeline run directly over `object_store`
  range reads (no full download).
- **Training loader** — `TrainingPipeline` feeds a triple-buffered
  tokio → rayon → Python pipeline with zero Python on the hot path.

## Choosing an access pattern

| Pattern | Use when | Reference |
|---|---|---|
| **In-memory** — `open().to_anndata()` + `sc.pp.*` | Dataset fits in RAM (< ~500K cells) | [docs/scanpy.md § In-memory](../docs/scanpy.md) |
| **Backed + lazy** — `to_anndata(backed=True)` + `pyscx.accel.*` | Atlas-scale (500K–10M+ cells) on modest RAM | [docs/scanpy.md § Backed + lazy](../docs/scanpy.md) |
| **Query pipeline** — `.query().filter_obs().collect()` | Pull a subset out of a large file | [docs/scanpy.md § Query pipeline](../docs/scanpy.md) |

[docs/scanpy.md](../docs/scanpy.md) walks through each pattern end-to-end,
including the lazy preprocess → PCA → kNN → UMAP → Leiden pipeline and the
backed-mode scanpy compatibility table.

## Quick start

```python
import pyscx
import scanpy as sc

# Convert h5ad → scx once (streaming; no full materialisation)
pyscx.from_h5ad("pbmc10k.h5ad", "pbmc10k.scx")

# Read back as a standard AnnData
adata = pyscx.open("pbmc10k.scx").to_anndata()

# Use any scanpy function unchanged
sc.pp.normalize_total(adata)
sc.tl.pca(adata)

# Or use the Rust accelerators — same shape, faster, GPU-capable
from pyscx import accel
accel.pca(adata, n_comps=50, device="auto")
accel.neighbors(adata, n_neighbors=15)
accel.umap(adata)
accel.leiden(adata, resolution=0.5)
```

## Installation

`pyscx` is a maturin-built extension. Pick one of:

**From the workspace checkout (developer install).**

```bash
# From the repository root. The shared uv venv lives at ../.venv.
uv venv  # if not yet created
cd pyscx
../.venv/bin/maturin develop                              # CPU-only
../.venv/bin/maturin develop --features hdf5              # + h5ad ingest
../.venv/bin/maturin develop --features hdf5,cloud,gpu    # full
```

Always use the shared `uv` venv at `../.venv/` — do not mix in system
`pip`. See [docs/development.md](../docs/development.md) for the full
build matrix (CPU-only, HDF5, cloud, GPU, Python, R).

**Pre-built wheels** (published to [GitHub Releases](https://github.com/ArcInstitute/scx/releases))
ship the `hdf5-static` feature so libhdf5 is bundled — no system library
required. The `cloud` and `gpu` extras remain opt-in — attach them to the
resolved wheel filename (`pip install "$(ls ./pyscx-*.whl)[cloud,gpu]"`; a
quoted `*` glob reaches pip verbatim and fails).

### Feature flags

| Cargo feature | Python extras | Enables |
|---|---|---|
| (default) | (none) | Core reader / writer, accelerators, query pipeline |
| `hdf5` / `hdf5-static` | — | `pyscx.from_h5ad`, `pyscx.from_mudata`, backed-AnnData auto-routing |
| `cloud` | `cloud` (boto3, gcsfs, azure) | `pyscx.open_cloud`, cloud query pipeline |
| `gpu` | `gpu` (cupy) | `device="gpu"` / `"auto"` GPU paths for accelerators |
| — | `mudata` | Multimodal h5mu round-trip via `pyscx.from_mudata` / `to_mudata` |
| — | `10x` | `pyscx.from_10x` (pulls scanpy for `read_10x_h5`) |

Without `hdf5`, h5ad ingest raises `NotImplementedError` directing the
caller to rebuild with `--features hdf5`. Other paths are unaffected.

## Documentation map

- **[../docs/scanpy.md](../docs/scanpy.md)** — scanpy / scverse integration:
  in-memory, backed-lazy, and query-pipeline patterns; accelerator
  reference; backed-mode compatibility table. **Start here for everyday
  analysis recipes.**
- **[../docs/api.md](../docs/api.md)** — full API reference for the
  Rust seam and the Python surface (`Experiment`, `from_h5ad`,
  `from_anndata`, query pipeline kwargs, `TrainingPipeline`).
- **[../docs/multimodal.md](../docs/multimodal.md)** — CITE-seq / 10x
  Multiome / TEA-seq layout, `from_mudata` / `to_mudata`,
  `MultimodalTrainingDataset`.
- **[../docs/cloud.md](../docs/cloud.md)** — `open_cloud`, auth, S3 /
  GCS / Azure tuning.
- **[../docs/gpu-setup.md](../docs/gpu-setup.md)** — CUDA / RAPIDS /
  conda / container / SLURM setup for the GPU accelerators.
- **[../docs/compatibility-matrix.md](../docs/compatibility-matrix.md)**
  — tested vs. declared Python / numpy / scipy / pyarrow / anndata /
  scanpy combinations.
- **[../docs/performance.md](../docs/performance.md)** — benchmark
  results and memory / throughput characteristics.
- **[../docs/architecture.md](../docs/architecture.md)** — workspace
  crate graph, feature flags, format / codec overview.
- **[../docs/development.md](../docs/development.md)** — developer
  build guide, test matrix, fuzzing.

## Testing

```bash
cd pyscx
../.venv/bin/maturin develop --features hdf5
../.venv/bin/pytest tests/ -v
```

Cloud and GPU paths gate behind their respective extras / hardware;
see [../docs/testing.md](../docs/testing.md) for the full matrix.
