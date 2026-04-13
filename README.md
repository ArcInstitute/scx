# SCX — Sparse Cell eXpression System

A purpose-built binary file format for single-cell RNA-seq data. SCX replaces h5ad with **3-7× smaller files**, **fastest reads at census scale** (1.4× faster than Zarr on 1M cells, up to 7× with parallel decode), **4-44× less memory**, a **GPU-saturating training loader**, and a **lazy query engine** — with native bindings for both **Python** and **R**, fully compatible with the [scverse](https://scverse.org/) ecosystem (scanpy, scVI, AnnData) and [Seurat v5](https://satijalab.org/seurat/).

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

| Dataset | Cells | h5ad | SCX (best) | Best ratio | Read time (SCX vs Zarr) |
|---------|-------|------|------------|------------|------------------------|
| PBMC 3K | 2,700 | 21.5 MB | 4.4 MB | **4.9×** | 0.04s vs 0.01s |
| Smart-seq2 | 50,000 | 1.07 GB | 350 MB | **3.1×** | 0.99s vs 0.37s |
| Tabula Sapiens | 100,000 | 1.59 GB | 322 MB | **4.9×** | 0.62s vs 0.56s |
| CELLxGENE Census 1M | 1,000,000 | 11.4 GB | 2.35 GB | **4.8×** | **2.9s** vs 4.0s |
| CELLxGENE Census 5M | 5,000,000 | 91.4 GB | 12.5 GB | **7.3×** | **35s** vs 41s |

### Your atlas doesn't fit in memory? SCX does.

SCX uses memory-mapped I/O with `MADV_DONTNEED` page release. During streaming
aggregation (row_sums, col_sums), only one decoded shard is resident at a time.
On a 1M-cell dataset (2.7 GB SCX file):

| | h5ad (full load) | SCX (streaming) | Savings |
|--|-------------------|-----------------|---------|
| Peak memory | 11.6 GB | 1.1 GB | **10×** |

This means you can stream through atlas-scale datasets without materializing.

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

| | SCX | AnnData | TileDB-SOMA-ML | scDataLoader |
|--|-----|---------|----------------|--------------|
| 1M cells (batches/sec) | **1,405** | 16.3 | 17.1 | 4.4 |
| vs SCX | — | 86× slower | 82× slower | 319× slower |

> _batch_size=1024, HVG=2000, normalize+log1p (hvg_norm scenario). See [detailed results](#training-loader-batchessec-batch_size1024-hvg2000-normalizelog1p) below._

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

### How SCX compares to existing formats

Every existing single-cell format has significant trade-offs that SCX was designed to avoid:

#### h5ad (HDF5)

The scverse standard. Ubiquitous but showing its age at atlas scale.

- **File locking breaks on HPC.** HDF5 uses mandatory POSIX `flock()`, which fails on NFS, Lustre, and GPFS — the most common HPC parallel filesystems. Users must set `HDF5_USE_FILE_LOCKING=FALSE`, disabling integrity checks entirely.
- **Single-threaded reads in Python.** h5py holds a global lock and does not release the GIL during HDF5 calls, so multithreaded reads gain zero parallelism.
- **Integer counts stored as float32.** AnnData stores UMI counts as 32-bit floats by default, wasting 2-4× space for data that fits in uint8/uint16.
- **Slow or weak compression.** Default gzip is slow to decompress; lzf is fast but achieves poor ratios. Zstd requires the third-party `hdf5plugin` and is not natively supported.
- **No cloud-native access.** HDF5 metadata is scattered throughout the file, requiring many small range requests on S3/GCS. The S3 VFD is read-only and limited.
- **No append without rewrite.** Adding cells or modifying obs/var requires rewriting the entire file. Backed mode (`r+`) only supports updating X values in-place.

#### Zarr

Cloud-native array storage. Good for object stores, problematic on local/HPC filesystems.

- **Multi-file directory structure.** Each chunk is a separate file on disk. A 1M-cell dataset produces tens of thousands of files, hitting inode quotas on Lustre/GPFS and causing heavy metadata server load.
- **No atomic writes.** A Zarr store is a directory tree — interrupted writes leave partial/corrupt state with no rollback mechanism.
- **No integrity verification.** No built-in checksums, tree hashing, or corruption detection. There is no way to validate a Zarr store's integrity after transfer or filesystem errors.
- **v2/v3 ecosystem fragmentation.** Zarr v3 is a breaking change (new metadata format, new codec pipeline). The anndata Zarr backend still defaults to v2; v3 sharding support is immature and significantly slower in practice.
- **No query or filter capability.** Zarr provides array-level chunk access only — no predicate pushdown, no cell/gene filtering without reading full chunks.
- **Filesystem overhead on HPC.** The one-file-per-chunk design suits object stores (S3) but penalizes local and HPC filesystems with per-file open/close syscall costs, directory traversal, and block alignment waste.

#### TileDB-SOMA

CELLxGENE Census standard. Powerful for cloud queries, heavy for everything else.

- **Multi-file directory structure.** Like Zarr, each TileDB array is a directory with many internal fragment files. On HPC shared filesystems, metadata operations (open, stat, list) are slow due to inode pressure. Fragment proliferation after repeated writes requires periodic consolidation — an operational burden absent from single-file formats.
- **Deep dependency stack.** TileDB-SOMA depends on TileDB Core (C++), libtiledbsoma, PyArrow, and multiple SOMA API layers (~5 layers deep). Building from source requires CMake and C++17. Conda packages frequently lag or conflict with RAPIDS/CUDA environments.
- **Slow for simple operations.** Opening a SOMA experiment requires listing and reading all fragment metadata — cold opens on networked storage take 5-30 seconds for large experiments. Simple "read all X into memory" is slower than h5ad for datasets under ~500K cells.
- **Complex API.** Reading a matrix requires navigating Experiment → Collection → Measurement → X["raw"] with Arrow table intermediaries, vs a single `sc.read_h5ad(path)` call.
- **Storage bloat before consolidation.** Fragment-based writes cause 1.5-3× storage bloat until consolidated. Each mutation creates a new fragment rather than updating in place.
- **Limited scanpy integration.** Converting SOMA to AnnData for scanpy/scVI typically materializes the full dataset, negating lazy-read benefits.

#### SCX addresses all of these

| Issue | h5ad | Zarr | TileDB-SOMA | SCX |
|-------|------|------|-------------|-----|
| Single file | Yes | No (directory) | No (directory) | **Yes** |
| HPC filesystem friendly | No (flock) | No (inode flood) | No (inode flood) | **Yes** (mmap, advisory locks) |
| Atomic writes | No | No | Fragment-based | **Yes** (atomic rename) |
| Integrity verification | Partial | None | Per-fragment | **Full** (BLAKE3 checksums) |
| Parallel reads | No (GIL) | Chunk-level | Tile-level | **Shard-level** (rayon) |
| Append without rewrite | No | No | Yes (fragments) | **Yes** (append sections) |
| Cloud-native access | No | Yes | Yes | **Yes** (explode/pack, selective pull) |
| Built-in query engine | No | No | Yes | **Yes** (predicate pushdown) |
| Domain-specific compression | No | No | No | **Yes** (Scx1 codec, 2-5× better) |
| Integer-aware storage | No (float32) | No (float32) | No (float64) | **Yes** (uint8/uint16 auto-detect) |

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
```

#### GPU acceleration

GPU support requires the CUDA Toolkit (≥ 12.0) and, for kNN/Leiden, the
RAPIDS libraries (cuVS, cuGraph). There are three ways to set this up —
conda is recommended as it handles the full CUDA + RAPIDS dependency tree.

**Option A: conda (recommended)** — resolves CUDA version matching automatically:

```bash
# Create a dedicated GPU environment
conda create -n scx-gpu python=3.13
conda activate scx-gpu

# Install RAPIDS (cuVS for kNN, cuGraph for Leiden) — pin cuda-version to match your driver
# Run `nvidia-smi` to check your driver's max CUDA version
conda install -c rapidsai -c conda-forge cuvs cugraph cuda-version=12.2

# Install Python deps + build pyscx with GPU support
pip install maturin numpy scipy pyarrow anndata scanpy scikit-learn leidenalg
cd pyscx && maturin develop --release --features gpu
```

**Option B: system CUDA Toolkit** — if you only need PCA/UMAP (no cuVS kNN or cuGraph Leiden):

```bash
# 1. Install CUDA Toolkit ≥ 12.0
#    Ubuntu/Debian:
#      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
#      sudo dpkg -i cuda-keyring_1.1-1_all.deb
#      sudo apt update && sudo apt install cuda-toolkit-12-2
#    Or see: https://developer.nvidia.com/cuda-downloads

# 2. Ensure nvcc is on PATH
export PATH=/usr/local/cuda/bin:$PATH
nvcc --version  # should print CUDA 12.x

# 3. Build pyscx with GPU support
uv venv .venv
uv pip install maturin numpy scipy pyarrow anndata
cd pyscx && ../.venv/bin/maturin develop --release --features gpu
```

This gives you GPU-accelerated PCA (cuSPARSE SpMM) and UMAP (native CUDA SGD).
kNN and Leiden will fall back to CPU since cuVS/cuGraph are not installed.

**Option C: container** — for reproducible environments or CI:

```bash
# Build the GPU image (multi-stage: compiles Rust + CUDA kernels, then slim runtime)
docker build -f Dockerfile.gpu -t scx-gpu .

# Run with GPU access
docker run --gpus all -it scx-gpu
docker run --gpus all -v /data:/data scx-gpu python my_analysis.py
```

**Verifying the installation:**

```python
import pyscx

# Check GPU availability
print(pyscx.accel.gpu_available())  # True if CUDA device found

# Run with explicit GPU — warns and falls back to CPU if unavailable
pyscx.accel.pca(adata, n_comps=50, device="gpu")
```

See [`docs/gpu-setup.md`](docs/gpu-setup.md) for troubleshooting, SLURM
configuration, and driver compatibility details.

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

All benchmarks on Intel Xeon Platinum 8468, 32 cores, 1–2 TB RAM. Full results in [`benchmarks/results/`](benchmarks/results/) and [`benchmarks/comprehensive/reporting/phase3_report.md`](benchmarks/comprehensive/reporting/phase3_report.md).

### Compression

| Dataset | Cells | h5ad → SCX | vs Zarr+Zstd |
|---------|-------|-----------|-------------|
| PBMC 3K | 2,700 | **4.9×** smaller | 2% smaller |
| Smart-seq2 | 50,000 | **2.9×** smaller | 5% smaller |
| Tabula Sapiens | 100,000 | **4.9×** smaller | 11% smaller |
| Census 1M | 1,000,000 | **4.8×** smaller | 10% smaller |
| Census 5M | 5,000,000 | **7.3×** smaller | 7% smaller |

### Read Speed (full load to AnnData)

| Dataset | SCX (auto) | h5ad (none) | h5ad (gzip) | Zarr (lz4) | TileDB-SOMA |
|---------|-----------|-------------|-------------|------------|-------------|
| PBMC 10K | 0.31s | 0.12s | 0.94s | **0.08s** | 0.44s |
| Tabula Sapiens 100K | **0.58s** | 1.41s | 7.56s | 1.20s | 3.79s |
| Census 1M | **2.74s** | 5.89s | 48.5s | 3.99s | 12.9s |
| Census 5M | **35.4s** | 43.7s | 291s | 40.6s | 80.6s |

SCX is the fastest reader at scale — **1.5× faster than Zarr**, **2.1× faster than uncompressed h5ad**, and **17.7× faster than gzip h5ad** on 1M cells. Four codecs available: `auto` (default), `scx1`, `zstd`, `lz4`.

### Column Projection (2000 HVGs)

| Dataset | SCX | h5ad (none) | Zarr (lz4) | TileDB-SOMA |
|---------|-----|-------------|------------|-------------|
| Tabula Sapiens 100K | **0.55s** | 0.86s | 0.94s | 1.31s |
| Census 1M | **3.53s** | 33.6s | 7.24s | 10.0s |
| Census 5M | **9.79s** | 94.1s | 63.9s | 66.3s |

SCX excels at gene selection — **2× faster than Zarr** and **9.6× faster than h5ad** on 1M+ cells.

### Memory

Peak RSS during full read (lower is better):

| Dataset | h5ad (none) | SCX (auto) | Zarr (zstd) |
|---------|-------------|------------|-------------|
| PBMC 10K | 0.48 GB | 1.51 GB | 0.76 GB |
| Tabula Sapiens 100K | 0.53 GB | 2.27 GB | 2.08 GB |
| Census 1M | 0.72 GB | 6.64 GB | 11.5 GB |
| Census 5M | 1.04 GB | 18.5 GB | 87.7 GB |

For streaming aggregation (row_sums, col_sums), `MADV_DONTNEED` reduces SCX peak RSS by **67%** — from 3.5 GB to 1.1 GB on Census 1M. h5ad has lowest peak RSS (lazy/backed mode). SCX uses less memory than Zarr at scale (18.5 GB vs 87.7 GB on Census 5M).

### Training Loader (batches/sec, batch_size=1024, HVG=2000, normalize+log1p)

| Dataset | SCX | AnnData | TileDB-SOMA-ML | scDataLoader | SCX/SOMA |
|---------|-----|---------|----------------|--------------|----------|
| Census 1M | **1,405** | 16.3 | 17.1 | 4.4 | **82×** |
| Tabula Sapiens 100K | **1,060** | 14.5 | 16.1 | 4.0 | **66×** |
| PBMC 3K | **168** | 14.5 | 5.0 | 6.3 | **34×** |

SCX's triple-buffered pipeline (tokio I/O → rayon decode → Python) with native HVG projection and fused normalize+log1p delivers **34–82× higher throughput** than TileDB-SOMA-ML at scale. TTFB (time to first batch): 16 ms on PBMC 3K, 603 ms on Census 1M.

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

#### GPU Analysis Pipeline

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
