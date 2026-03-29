# SCX — Sparse Cell eXpression System

A purpose-built binary file format for single-cell RNA-seq data. SCX replaces h5ad with **3-5× smaller files**, **4-44× less memory**, a **GPU-saturating training loader**, and a **lazy query engine** — with native bindings for both **Python** and **R**, fully compatible with the [scverse](https://scverse.org/) ecosystem (scanpy, scVI, AnnData) and [Seurat v5](https://satijalab.org/seurat/).

**Python** — works with scanpy, scVI, and any scverse tool:

```python
import pyscx

adata = pyscx.open("experiment.scx").to_anndata()

import scanpy as sc
sc.pp.normalize_total(adata)
sc.tl.pca(adata)
sc.tl.leiden(adata)
```

**R** — works with Seurat v5 and SingleCellExperiment:

```r
library(rscx)

exp <- scx_open("experiment.scx")
seurat_obj <- exp$to_seurat()    # Seurat v5 assay
sce <- exp$to_sce()              # SingleCellExperiment
```

## Why SCX?

### Your h5ad files are bigger than they need to be

h5ad stores integer UMI counts as 32-bit floats. SCX detects this and uses the narrowest
integer type that fits (uint8/uint16), then applies domain-specific codecs designed for
the statistical properties of count data. Result:

| Dataset | Cells | h5ad | SCX | Compression |
|---------|-------|------|-----|-------------|
| PBMC 3K | 2,700 | 21.5 MB | 4.4 MB | **4.9×** |
| Smart-seq2 | 50,000 | 1.07 GB | 370 MB | **2.9×** |
| Tabula Sapiens | 100,000 | 1.59 GB | 428 MB | **3.7×** |
| CELLxGENE Census | 1,000,000 | 11.4 GB | 2.47 GB | **4.6×** |

### Your atlas doesn't fit in memory? SCX does.

SCX uses memory-mapped I/O and zero-copy transfers. Peak memory is the size of the
decompressed matrix, not the entire file. On a 1M-cell dataset:

| | h5ad | SCX | Savings |
|--|------|-----|---------|
| Peak memory | 11.6 GB | 264 MB | **44×** |

This means you can work with atlas-scale datasets on a laptop.

### Too big to load at all? Use backed mode.

SCX supports **backed mode** — data stays on disk and loads on demand, one shard at a
time. Open a 10M-cell atlas without allocating the full matrix:

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# X stays on disk — only the accessed shard is decoded
subset = adata[adata.obs["cell_type"] == "T cell"].copy()

# Now subset is a regular AnnData — preprocess normally
# Or use SCX accelerators for compute-heavy steps (3-10× faster at scale)
sc.pp.normalize_total(subset, target_sum=1e4)
sc.pp.log1p(subset)
sc.pp.pca(subset)
```

Layers, deletion vectors, and `anndata.abc.CSRDataset` registration all work
transparently. See [`docs/scanpy.md`](docs/scanpy.md#backed-mode-lazy-loading)
for details.

### Your training loop is bottlenecked on data loading

SCX includes a **triple-buffered Rust pipeline** (tokio I/O → rayon decode → GPU) that
keeps your GPU fed. Zero Python on the hot path — all I/O, decompression, shuffling,
sparse-to-dense conversion, and normalization happen in compiled Rust.

| | SCX | AnnData | TileDB-SOMA-ML |
|--|-----|---------|----------------|
| 1M cells (batches/sec) | **38.4** | 17.6 | 14.2 |
| vs SCX | — | 2.2× slower | 2.7× slower |

```python
# GPU-saturating training loader — no num_workers needed
dataset = pyscx.TrainingDataset(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_array,   # decode only HVGs → 15× less data
    normalize=True,
    log1p=True,
)
for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
    # model.forward(), loss.backward(), ...
```

### You want to query without loading everything

SCX includes a lazy query engine with two-level predicate pushdown. Filter by cell type,
tissue, donor — SCX skips entire shards that can't match, reading only the data you need.

```python
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")
    .select_genes(hvg_indices)
    .with_normalize(1e4)
    .with_log1p()
    .collect())

adata = result.to_anndata()   # only matching cells, zero-copy
print(f"Skipped {result.skipped_shards}/{result.total_shards} shards")
```

Average shard skip rate: **55%** on realistic queries. Selective query latency: **<5 ms**.

### You need to update your dataset without rewriting it

SCX supports append, delete, compact, merge, and rollback — no need to rewrite
the entire file when adding new cells or removing doublets.

```python
# Append new cells (writes at EOF, O(new_cells) not O(total_cells))
pyscx.append("atlas.scx", "new_batch.scx")

# Logical deletion (instant, no data rewrite)
pyscx.mark_deleted("atlas.scx", doublet_indices)

# Reclaim space when convenient
pyscx.compact("atlas.scx", "atlas_clean.scx")

# Oops? Roll back to the previous version (header-only update)
pyscx.rollback("atlas.scx")
```

### Your data lives in the cloud

SCX provides streaming push/pull with S3, GCS, and Azure — no intermediate files.
Selective pull downloads only matching shards, saving bandwidth on atlas-scale data.

```python
# Stream from GCS → local packed file (parallel downloads)
pyscx.pull("gs://bucket/atlas.scxd/", "atlas.scx")

# Selective pull — only download T cells
pyscx.pull("gs://bucket/atlas.scxd/", "t_cells.scx",
           filter="cell_type == 'T cell'")

# Direct cloud reads (no full download needed)
exp = pyscx.open_cloud("gs://bucket/atlas.scxd/")
print(exp.n_obs, exp.n_vars)
```

### Your analysis pipeline is too slow for atlas-scale

At >1M cells, even optimized CPU code for PCA, kNN, and UMAP takes minutes.
SCX provides GPU-accelerated analysis via CUDA — PCA through cuSPARSE SpMM +
cuSOLVER QR, kNN through cuVS CAGRA, and UMAP through a native CUDA SGD kernel.
All accessed through the same Python API with a single `device="gpu"` parameter:

```python
import pyscx

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# GPU-accelerated pipeline — up to 16× faster per-op, 3.8× end-to-end on 1M cells
pyscx.accel.pca(adata, n_comps=50, device="gpu")
pyscx.accel.neighbors(adata, n_neighbors=15, device="gpu")
pyscx.accel.umap(adata, device="gpu")

import scanpy as sc
sc.tl.leiden(adata)  # downstream scanpy works identically
sc.pl.umap(adata, color="leiden")
```

SCX streams shards from disk → GPU via cuSPARSE SpMM — no full matrix
materialization in CPU memory. This enables GPU analysis on datasets larger
than VRAM. When no GPU is available, every operation falls back to CPU
automatically with a warning.

### No more file locking headaches

HDF5 acquires **mandatory POSIX file locks** on every open — even for reads. On shared
and network filesystems (NFS, Lustre, GPFS) this causes the dreaded:

```
OSError: Unable to open file (unable to lock file, errno = 37, error message = 'No locks available')
```

Common workarounds (`HDF5_USE_FILE_LOCKING=FALSE`, rebuilding with `--disable-file-locking`)
disable data integrity checks entirely. Jupyter notebooks that hold an h5ad open will
block other processes from reading the same file.

SCX takes a different approach:

| | h5ad (HDF5) | SCX |
|--|-------------|-----|
| Read locking | Mandatory — blocks other readers/writers | **None** — reads never lock |
| Write locking | Mandatory — blocks all other access | **Advisory** `flock()` — only during append/delete/compact |
| Concurrent reads | ❌ Blocked if any writer is active | ✅ Always allowed, even during writes |
| Network filesystems | Frequently broken (`errno 37`) | Works — advisory locks degrade gracefully |
| Workaround needed? | `HDF5_USE_FILE_LOCKING=FALSE` | No workaround needed |

SCX's immutable-fragment design (append-only sections + atomic header update) means readers
always see a consistent snapshot without any locking. Multiple notebooks, pipeline stages,
or training jobs can read the same `.scx` file simultaneously — no coordination required.

### Sharding for parallel I/O and selective access

SCX splits the expression matrix into **CSR shards** — fixed-size, independently
decompressible chunks of rows (default: 10,000–16,384 cells per shard). Sharding
enables parallel decoding, memory-bounded reads, shard-level predicate pushdown,
append-without-rewrite, and selective cloud downloads. Control shard size via
`--shard-size` (CLI) or `shard_size=` (Python/R).

See [`docs/sharding.md`](docs/sharding.md) for a full guide including sizing
guidelines, CLI/Python/R commands, and how sharding powers each SCX feature.

### Multithreading for parallel decode and overlapped I/O

SCX exploits multicore CPUs at every stage. Rayon decodes shards in parallel
during reads and queries. The training loader runs a triple-buffered pipeline
(tokio I/O → rayon decode → Python/GPU) so that I/O, decompression, and GPU
transfer all overlap. Cloud downloads run as concurrent async tasks. File
mutations are serialized with advisory `flock()` locks — reads never lock.

| Component | Threading model | Runtime |
|-----------|----------------|---------|
| Shard decode | Data parallelism | Rayon `par_iter` |
| Query engine | Parallel shard decode + filter | Rayon |
| Training loader | Triple-buffered pipeline | tokio + rayon + std::thread |
| Cloud I/O | Parallel async downloads/uploads | tokio |
| File mutations | Advisory file locks | `fs4` `flock()` |
| GPU decode | Massively parallel kernels | CUDA |

See [`docs/multithreading.md`](docs/multithreading.md) for a full guide
including pipeline architecture, thread safety of key types, and how to
control parallelism.

## Installation

### Python (recommended)

```bash
# Create a virtual environment
uv venv .venv

# Install pyscx and dependencies
uv pip install maturin numpy scipy pyarrow anndata

# Build from source
cd pyscx && ../.venv/bin/maturin develop --release

# With cloud support (S3/GCS/Azure):
cd pyscx && ../.venv/bin/maturin develop --release --features cloud

# With GPU acceleration (requires CUDA Toolkit ≥ 12.0):
cd pyscx && ../.venv/bin/maturin develop --release --features gpu

# With both cloud and GPU:
cd pyscx && ../.venv/bin/maturin develop --release --features cloud,gpu
```

### Rust CLI

```bash
# Build the CLI tool
cargo build -p scx-cli --release

# With h5ad conversion support:
cargo build -p scx-cli --release --features hdf5

# With cloud operations:
cargo build -p scx-cli --release --features hdf5,cloud
```

### R (rscx)

Requires a Rust toolchain (`rustc` ≥ 1.78 and `cargo`). Install via [rustup](https://rustup.rs/).

```bash
# From the repository root:
R CMD INSTALL rscx/

# Or, from within R:
devtools::install_local("rscx/")
```

The package compiles the Rust workspace during installation (handled by `src/Makevars`).
No pre-built binaries are distributed — Cargo builds all SCX crates from source.

**System requirements:**
- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- R ≥ 4.2.0
- R packages: `Matrix`, `methods` (required); `Seurat` ≥ 5.0, `SingleCellExperiment` (optional)

## Quick Start

### Convert your data to SCX

SCX supports **roundtrip conversion** with h5ad, 10x HDF5, and Cell Ranger MTX formats —
convert in, work with SCX, convert back out.

```bash
# h5ad ↔ SCX (roundtrip)
scx convert experiment.h5ad experiment.scx
scx convert --to h5ad experiment.scx experiment.h5ad

# Cell Ranger MTX ↔ SCX (roundtrip)
scx convert /path/to/filtered_feature_bc_matrix/ experiment.scx
scx convert --to mtx experiment.scx /path/to/output_dir/

# 10x HDF5 → SCX
scx convert filtered_feature_bc_matrix.h5 experiment.scx
```

```python
import pyscx

# From AnnData object
pyscx.from_anndata(adata, "experiment.scx")

# From 10x HDF5
pyscx.from_10x("filtered_feature_bc_matrix.h5", "experiment.scx")

# From Cell Ranger MTX directory
pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "experiment.scx")

# Export back to MTX
pyscx.to_mtx("experiment.scx", "/path/to/output_dir")
```

### Read into AnnData

```python
adata = pyscx.open("experiment.scx").to_anndata()
# → standard AnnData with X, obs, var, obsm, layers, uns
```

### GPU-accelerated analysis

```python
import pyscx

adata = pyscx.open("atlas.scx").to_anndata()

# Auto-detect GPU (falls back to CPU if unavailable)
pyscx.accel.pca(adata, n_comps=50)               # device="auto" by default
pyscx.accel.neighbors(adata, n_neighbors=15)      # device="auto" by default
pyscx.accel.umap(adata)                           # device="auto" by default

# Force GPU
pyscx.accel.pca(adata, n_comps=50, device="gpu")

# Multi-GPU selection
pyscx.accel.pca(adata, n_comps=50, device="gpu:1")
# → standard AnnData with X, obs, var, obsm, layers, uns
```

### Query and filter

```python
result = (pyscx.open("experiment.scx")
    .query()
    .filter_obs("cell_type == 'B cell'")
    .collect())
adata = result.to_anndata()
```

### CLI

```bash
scx info experiment.scx                  # file metadata
scx validate experiment.scx              # verify checksums
scx query experiment.scx "tissue == 'lung'" --count
scx append atlas.scx --input batch2.scx
scx merge batch1.scx batch2.scx --output atlas.scx
```

## Benchmarks

All benchmarks on Intel Xeon Platinum 8468, 1 TB RAM. Full results in [`benchmarks/results/`](benchmarks/results/).

### Compression

| Dataset | Cells | h5ad → SCX | vs Zarr+Zstd |
|---------|-------|-----------|-------------|
| PBMC 3K | 2,700 | **4.9×** smaller | 2% smaller |
| Smart-seq2 | 50,000 | **2.9×** smaller | 5% smaller |
| Tabula Sapiens | 100,000 | **3.7×** smaller | 10% smaller |
| Census 1M | 1,000,000 | **4.6×** smaller | 6% smaller |

### Memory

| Dataset | h5ad Peak | SCX Peak | Reduction |
|---------|-----------|----------|-----------|
| PBMC 3K | 24 MB | 5.5 MB | **4×** |
| Smart-seq2 | 1.1 GB | 32 MB | **33×** |
| Tabula Sapiens | 1.6 GB | 42 MB | **38×** |
| Census 1M | 11.6 GB | 264 MB | **44×** |

### Training Loader (batches/sec, batch_size=1024)

| Dataset | SCX | AnnData | TileDB-SOMA-ML | SCX/SOMA |
|---------|-----|---------|----------------|----------|
| Census 1M | **38.4** | 17.6 | 14.2 | **2.7×** |
| Tabula Sapiens | 19.9 | 13.9 | 20.8 | 1.0× |
| Smart-seq2 | 19.7 | 8.7 | 19.6 | 1.0× |

SCX's advantage grows with dataset size — at atlas scale (1M+ cells), the compressed-shard
streaming pipeline outperforms random-access approaches.

### GPU Acceleration (NVIDIA H100)

The `scx-gpu` crate provides CUDA-accelerated codec decoding, sparse-to-dense
conversion, and a full GPU analysis pipeline (PCA, kNN, UMAP). Benchmarked on
H100 80GB HBM3:

#### Codec Decode & Training Pipeline

| Operation | Size | CPU (μs) | GPU (μs) | Speedup |
|-----------|------|----------|----------|---------|
| FOR-BP index decode | 16K rows, 33M nnz | 102,900 | 4,133 | **24.9×** |
| Sparse → dense | 16K rows × 30K cols | 433,252 | 7,711 | **56.2×** |
| Sparse → dense (HVG 2K) | 16K rows × 2K output | 110,416 | 897 | **123.1×** |

#### GPU Analysis Pipeline (Phase 4c)

GPU-accelerated analysis via cuSPARSE, cuSOLVER, cuVS CAGRA,
native CUDA UMAP kernel, and cuGraph Leiden. Benchmarked on H100 80GB
with 1M cells (CELLxGENE Census):

| Operation | CPU (s) | GPU (s) | Speedup | Backend |
|-----------|---------|---------|---------|---------|
| kNN (k=15, 50 PCs) | 288 | 31 | **9.4×** | cuVS CAGRA |
| UMAP (2D) | 560 | 74 | **7.6×** | native CUDA SGD |
| Leiden | 45 | 3 | **16.0×** | cuGraph |
| PCA (50 PCs, 2K HVGs) | 22 | 24 | 0.9× | cuSPARSE SpMM |
| **End-to-end pipeline** | **1077** | **286** | **3.8×** | all above |

The GPU PCA pipeline streams shards from disk → GPU SpMM shard-by-shard
without materializing the full matrix — enabling PCA on datasets larger
than VRAM. kNN uses NVIDIA's CAGRA algorithm (cuVS) for up to 9.4×
throughput over CPU HNSW on 1M cells.

Full GPU benchmark details in [`benchmarks/results/gpu_pipeline_benchmark.md`](benchmarks/results/gpu_pipeline_benchmark.md).

### Query Engine

| Metric | Result |
|--------|--------|
| Shard skip rate | **55%** average |
| Selective query | **4.2 ms** |
| vs AnnData subsetting | **2.1×** faster |

### File Operations

| Operation | Speed |
|-----------|-------|
| Append 10K cells | **1 ms** |
| Merge 3 files | **342 MB/s** |
| Compact (after 3 appends) | 0.98× fresh-write size |

## Architecture

SCX is a Rust workspace with 14 crates:

| Crate | Purpose |
|-------|---------|
| `scx-format` | File layout, reader/writer, catalog, checksums |
| `scx-codec` | Domain-specific codecs: Rice, FOR-BP, Delta-Golomb, Zstd |
| `scx-sparse` | CSR matrix type (scipy-compatible) |
| `scx-ops` | Append, delete, compact, merge, rollback |
| `scx-engine` | Lazy query engine with predicate pushdown |
| `scx-loader` | Triple-buffered ML training data loader |
| `scx-accel` | Rust-native analysis accelerators: PCA, kNN, UMAP, DE, pseudobulk |
| `scx-gpu` | CUDA-accelerated codec decoding, cuSPARSE interop, GPU sparse-to-dense |
| `scx-cloud` | Cloud access: push, pull, explode, pack, CloudReader |
| `scx-mtx` | Matrix Market (MTX) I/O: Cell Ranger directory read/write |
| `scx-cli` | CLI tool |
| `pyscx` | Python bindings (PyO3) |
| `rscx` | R bindings (extendr) |

For technical details, see [`docs/architecture.md`](docs/architecture.md), [`docs/api.md`](docs/api.md), [`docs/sharding.md`](docs/sharding.md), [`docs/multithreading.md`](docs/multithreading.md), [`docs/scanpy.md`](docs/scanpy.md), and [`SPEC.md`](SPEC.md).

## License

MIT
