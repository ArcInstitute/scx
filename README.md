# SCX — Sparse Cell eXpression System

A purpose-built binary file format for single-cell RNA-seq data. SCX replaces h5ad with **3-7× smaller files**, **fastest reads at census scale** (1.4× faster than Zarr on 1M cells, up to 7× with parallel decode, up to 3.2× parallel write scaling), **4-44× less memory**, a **GPU-saturating training loader**, and a **lazy query engine** — with native bindings for both **Python** and **R**, fully compatible with the [scverse](https://scverse.org/) ecosystem (scanpy, scVI, AnnData) and [Seurat v5](https://satijalab.org/seurat/).

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

## Main features

- **Fast at every scale** — on 1M cells SCX is **17× faster** than the gzipped h5ad most researchers ship, 1.4× faster than Zarr, and produces a file ~1.2–1.7× smaller than gzipped h5ad (4–5× smaller than anndata's default uncompressed h5ad). Single file, BLAKE3-checksummed, mmap-friendly, and HPC-safe: no `HDF5_USE_FILE_LOCKING=FALSE` workaround on NFS / Lustre / GPFS.
- **Scales on CPU and GPU** — shard-level parallelism via rayon delivers up to **7× read** and **3.2× write** scaling. The CUDA path (cuSPARSE · cuSOLVER · cuVS CAGRA · cuGraph) gives **3.8× end-to-end** on PCA → kNN → UMAP → Leiden at 1M cells; the training loader hits **1,405 batches/s** — 82× faster than TileDB-SOMA-ML.
- **Atlas-scale memory footprint** — backed mode + `MADV_DONTNEED` streaming. A full 1M-cell preprocess-to-cluster pipeline (open → QC → normalize → log1p → HVG → PCA → kNN → UMAP → Leiden) runs at **~11 GB peak RSS** vs ~22 GB materialised (51% less; lazy preprocessing alone peaks at ~3.5 GB). Backed mode lets you open a 10M-cell atlas without allocating the full matrix.
- **Rust-native analysis accelerators** — drop-in replacements for `sc.pp.*` / `sc.tl.*`: PCA, kNN, UMAP, Leiden, differential expression, pseudobulk, and [Harmony2 batch integration](https://www.biorxiv.org/content/10.64898/2026.03.16.711825v1). Same scanpy-shaped API, 3–40× faster; every op has a `device="auto"` switch that picks GPU when available.
- **Mutable without rewriting** — append new cells, mark-delete doublets, compact, merge, or roll back in milliseconds. Append writes new matrix shards in O(new cells); obs metadata is rewritten as a merged Arrow IPC covering all cells (see [docs/operations.md](docs/operations.md)). Delete is a logical mask, not a data rewrite.
- **Lazy query engine** — predicate pushdown skips ~55% of shards on realistic queries; selective reads land in under **5 ms**. Filter by cell type, tissue, donor, etc. before paying to read.
- **Drop-in for scverse and Seurat** — works with AnnData, scanpy, and scVI (Python), and Seurat v5 + SingleCellExperiment (R). Round-trips cleanly with h5ad, 10x HDF5, and Cell Ranger MTX.
- **Cloud-native** — streaming push/pull to S3, GCS, and Azure with selective download (only the shards you need) and a direct `open_cloud()` path that skips the full download.

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

For ML workloads where each batch is a list of `(perturbed_cell, control_cell)`
pairs (perturbation training, contrastive learning, donor-matched designs),
`pyscx.IndexPlanDataset` is the sibling row-source: consumer-supplied plans,
paired dense `{X, X_paired, pairs, obs, obs_paired}` batches, optional
shard prefetch lookahead. Plan-driven access is intentionally random — at
1M cells it reaches **20K cells/s** (3.7× slower than the sequential
`TrainingDataset` ceiling) but is **106× faster** than the cell-load-scx
`ScxBackedSparseDataset` Python-loop baseline. See
[`docs/api.md` § IndexPlanDataset](docs/api.md#indexplandataset).

#### Fork safety under PyTorch `DataLoader(num_workers > 0)`

`pyscx.TrainingDataset` and `pyscx.IndexPlanDataset` are fork-safe under
`torch.utils.data.DataLoader(num_workers > 0, start_method="fork")` —
the Linux PyTorch default — when the dataset is **constructed lazily
inside the worker's `__iter__`** (the pattern `cell-load-scx` and
`state-scx` already use). Both the per-pipeline tokio runtime (current-
thread, per-epoch) and per-pipeline `rayon::ThreadPool` (lazily built on
first iteration) are constructed inside the worker process, so a forked
child inherits no fork-hostile state from the parent.

```python
# Recommended IterableDataset wrapper for DataLoader(num_workers=2)
import torch.utils.data as data

class TrainingShim(data.IterableDataset):
    def __init__(self, scx_path):
        self.scx_path = scx_path  # paths only — no inner dataset yet

    def __iter__(self):
        # Construct the inner dataset HERE (in the worker process, post-fork).
        ds = pyscx.TrainingDataset(self.scx_path, batch_size=1024, hvg_indices=...)
        try:
            yield from ds
        finally:
            ds.close()  # release the per-pipeline rayon pool / tokio runtime

loader = torch.utils.data.DataLoader(
    TrainingShim("atlas.scx"),
    batch_size=None,
    num_workers=2,
    persistent_workers=False,
)
```

**Do**: construct lazily inside `__iter__`; call `dataset.close()` (or
register `weakref.finalize(dataset, dataset.close)`) before process exit;
prefer `multiprocessing.set_start_method("spawn")` if your workload
allows — spawn re-execs Python in the child and is genuinely fork-safe
because there is no fork.

**Don't**: construct a `TrainingDataset` / `IndexPlanDataset` in the
parent and share it across forked workers — the PID check in `__next__`
raises `RuntimeError`. Don't pickle a constructed dataset across
processes either; it owns thread handles that don't survive transfer.

> [!IMPORTANT]
> Calling **any** `rayon::par_*`-using pyscx API in the parent before
> fork (e.g., `pyscx.from_anndata(...)` to write the fixture) initialises
> rayon's process-global pool. Pre-fix this prerequisite was sufficient
> to wedge `DataLoader(num_workers=2)` indefinitely; the per-pipeline
> rayon pool in `scx-loader` removed that hazard. See
> `pyscx/tests/test_fork_safety.py` for the durable regression test.

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
# Append new cells (new CSR shards at EOF; obs metadata rewritten for all cells)
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

# Selective pull — only download shards containing T cells
# (shard-granular: output may include extra cells from partially matching shards)
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

### Your perturb-seq evaluation pipeline is slow

SCX ships Rust-accelerated equivalents of the metrics in
[`cell-eval`](https://github.com/arcinstitute/cell-eval) and
[`arc-bench`](https://github.com/arcinstitute/arc-bench) — pseudobulk means,
bundled bulk metrics (pearson_delta / mse / mae / mse_delta / mae_delta),
discrimination score, energy distance, knockdown efficiency, and clustering
agreement. Output is numerically equivalent to the Python references
(32/32 parity tests pass), so you can swap in `pyscx.accel.*` without changing
the rest of your pipeline.

```python
# One call replaces cell-eval's pearson_delta + mse + mae + mse_delta + mae_delta
results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)

# Per-cell knockdown efficiency vs control (arc-bench equivalent)
pyscx.accel.knockdown_efficiency(adata, pert_col="perturbation", control="control")
# → adata.obs["KnockDownEfficiency"], adata.obs["KnockDownGeneFC"]

# Energy distance: faer-gemm + f32 default (use backend="scalar" / dtype="f64"
# for legacy bit-exact reproduction).
corr = pyscx.accel.energy_distance(adata_real, adata_pred)

# Clustering agreement: native-Rust HNSW + Leiden (no scanpy under the hood).
score = pyscx.accel.clustering_agreement(adata_real, adata_pred, metric="ami")
```

| Operation | 20K × 2K × 50 ² | 100K cells | 500K cells | 1M cells |
|---|---:|---:|---:|---:|
| Pseudobulk means | 11.8× | **11.6×** | **13.8×** | **19.4×** |
| Bulk metrics (5 bundled) | 10.5× | **12.1×** | **13.6×** | **21.9×** |
| Discrimination score | 11.8× | **12.0×** | **12.9×** | **20.1×** |
| Energy distance (gemm + f32, default) | **52.1×** | re-bench TBD ² | — ¹ | — ¹ |
| Energy distance (scalar + f64, legacy) | 10.2× | **14.4×** | — ¹ | — ¹ |
| Clustering agreement (native Rust Leiden) | 3.0× ³ | **10–13×** | **24.6×** | **10.0×** |

¹ Reference's `sklearn.metrics.pairwise_distances` doesn't scale above 100K.
² 20K column captured 2026-04-27 with the Phase 1+2 `(backend, dtype)` matrix; 100K–1M columns are pre-Phase-1 historical baselines pending a re-run.
³ Speedup grows with `n_perts` × embedding-dim; at 200 perts × 300 genes the ratio is 12.8×.

See [`docs/scanpy.md`](docs/scanpy.md#perturbation-evaluation-metrics-cell-eval--arc-bench-parity)
for the full API and [`docs/performance.md`](docs/performance.md#perturbation-metrics-cell-eval--arc-bench-parity)
for the benchmark methodology.

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
during reads (up to 7× at 32 threads) and encodes shards in parallel during
writes (up to 3.2× at 32 threads). The training loader runs a triple-buffered
pipeline (tokio I/O → rayon decode → Python/GPU) so that I/O, decompression,
and GPU transfer all overlap. Cloud downloads run as concurrent async tasks.
File mutations are serialized with advisory `flock()` locks — reads never lock.

| Component | Threading model | Runtime |
|-----------|----------------|---------|
| Shard decode (read) | Data parallelism | Rayon `par_iter` |
| Shard encoding (write) | Data parallelism | Rayon `par_iter` |
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

#### SLAF

SQL-native lazy format backed by Lance columnar files + a DuckDB query layer.
Compelling design idea for obs-driven analytics, but heavy at realistic read
sizes in our benchmarks.

- **Slow full-materialize path.** `LazyAnnData.compute()` goes through Polars
  fragment processors and builds the CSR in Python-owned memory. Full read of
  Census 1M takes **53 s** vs 2.7 s for SCX and 4.0 s for Zarr (lz4); peak RSS
  during the materialization reached **35 GB**, an order of magnitude more than
  any other format we tested.
- **String `cell_id` / `gene_id` is load-bearing.** The public `get_submatrix`
  API returns long-form `(cell_id, gene_id, value)` with string ids. In
  real-world datasets (census, cell barcodes, etc.) those strings are not
  unique across the file's Lance fragments, so joining back to integer
  positions explodes row counts — consumers must go through the internal
  `expression(cell_integer_id, gene_integer_id, value)` SQL table to get a
  scipy CSR correctly.
- **Non-standard selector semantics.** `get_submatrix(cell_selector=[…])`
  treats an integer list as positional but returns string ids that don't
  always match the list (e.g. cell at position 201469 has `cell_id = "1469"`
  in census_1m). Integer numpy arrays, string lists, and mixed-type
  selectors all fail in different ways.
- **PyTorch loader stalls at atlas scale.** `SLAFDataLoader`'s
  Mixture-of-Scanners prefetcher produced **0 batches** on Census 10M in our
  benchmark with the default config (90 s TTFB, then timeout). On Census 1M
  it ran cleanly at **~4 batches/sec** across raw/hvg/norm/hvg_norm — **roughly
  340× slower than SCX**, and about 4× slower than `AnnData` in-memory
  iteration.
- **Heavy on-disk, small-file directory layout.** A 1M-cell dataset expands
  to 4.0 GB vs 2.4 GB for SCX, and the `.slaf` directory holds hundreds of
  Lance fragment + statistics files — same HPC-filesystem inode pressure
  problem as Zarr / TileDB-SOMA.
- **SQL predicate pushdown is a real strength.** For `cell_type == 'T cell'`
  on Census 1M, SLAF's SQL path returns the matching expression records in
  10 s — same order of magnitude as SCX's catalog pushdown.

#### SCX addresses all of these

| Issue | h5ad | Zarr | TileDB-SOMA | SLAF | SCX |
|-------|------|------|-------------|------|-----|
| Single file | Yes | No (directory) | No (directory) | No (directory) | **Yes** |
| HPC filesystem friendly | No (flock) | No (inode flood) | No (inode flood) | No (inode flood) | **Yes** (mmap, advisory locks) |
| Atomic writes | No | No | Fragment-based | Fragment-based | **Yes** (atomic rename) |
| Integrity verification | Partial | None | Per-fragment | Per-fragment | **Full** (BLAKE3: catalog verified on open; `validate()` re-hashes all section payloads) |
| Parallel reads | No (GIL) | Chunk-level | Tile-level | Fragment-level | **Shard-level** (rayon) |
| Append without rewrite | No | No | Yes (fragments) | Yes (fragments) | **Yes** (append sections) |
| Cloud-native access | No | Yes | Yes | Yes | **Yes** (explode/pack, selective pull) |
| Built-in query engine | No | No | Yes | **Yes (SQL)** | **Yes** (predicate pushdown) |
| Domain-specific compression | No | No | No | No | **Yes** (Scx1 codec, 2-5× better) |
| Integer-aware storage | No (float32) | No (float32) | No (float64) | u16 per-cell | **Yes** (uint8/uint16 auto-detect) |

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

**Pre-built binaries (recommended)** — published on each `scx-cli-v*` tag at [GitHub Releases](https://github.com/ArcInstitute/scx/releases). Linux only (x86_64 and arm64), glibc ≥ 2.35 (Ubuntu 22.04+, Debian 13+, RHEL 10+). Bundles `hdf5` (h5ad conversion) and `cloud` (S3/GCS/Azure); libhdf5 is statically linked so no system libraries are required at runtime.

```bash
VERSION=0.2.0
# Pick the matching target for your platform:
#   linux x86_64 → x86_64-unknown-linux-gnu
#   linux arm64  → aarch64-unknown-linux-gnu
TARGET=x86_64-unknown-linux-gnu

# Requires the GitHub CLI (https://cli.github.com/) and `gh auth login`
# while the repo is private.
gh release download "scx-cli-v${VERSION}" -R ArcInstitute/scx \
  -p "scx-cli-${VERSION}-${TARGET}.tar.gz"
tar xzf "scx-cli-${VERSION}-${TARGET}.tar.gz"
./scx-cli-${VERSION}-${TARGET}/scx --version

# Move onto your PATH:
install -m 0755 "scx-cli-${VERSION}-${TARGET}/scx" ~/.local/bin/scx
```

Once the repository is public, the asset can also be fetched without `gh`:

```bash
curl -L "https://github.com/ArcInstitute/scx/releases/download/scx-cli-v${VERSION}/scx-cli-${VERSION}-${TARGET}.tar.gz" | tar xz
```

**Build from source** — for macOS, Windows, musl, or custom feature sets:

```bash
# Build the CLI tool
cargo build -p scx-cli --release

# With h5ad conversion support (requires libhdf5-dev):
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

For an end-to-end walkthrough, see the [scanpy tutorial notebook](notebooks/scx_scanpy_tutorial.ipynb).

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

Headline numbers at Census 1M (CELLxGENE Census, 1M cells):

| Area | SCX | Next best | SCX advantage |
|------|-----|-----------|---------------|
| File size vs gzip h5ad | 2.35 GB | 11.4 GB | **4.8× smaller** |
| Read (full load to AnnData) | **2.74 s** | 3.99 s (Zarr lz4) | **1.5× faster** |
| Column projection (2K HVGs) | **3.53 s** | 7.24 s (Zarr lz4) | **2.0× faster** |
| Parallel read (32 threads) | **3.0 s** | — (no other format scales) | **6.1× vs 1 thread** |
| Parallel write (32 threads, pcodec) | 11.4 s | — | **3.2× vs 1 thread** |
| Out-of-core pipeline peak RSS | **~11 GB** | ~22 GB (materialized) | **51% reduction** |
| Training loader (batches/s) | **1,405** | 17.1 (TileDB-SOMA-ML) | **82× faster** |
| GPU end-to-end pipeline (H100) | **286 s** | 1,077 s (CPU) | **3.8× faster** |
| Selective query (55% shard skip) | **4.2 ms** | — | — |
| Append 10K cells | **1 ms** | — | — |

Full benchmark suite in [`docs/performance.md`](docs/performance.md): compression and read/write/conversion timings across h5ad / Zarr / TileDB-SOMA / SLAF, parallel read and write scaling, column projection, memory (peak RSS and out-of-core), CPU analysis accelerators (PCA / DE / Leiden), Harmony2 + LISI scaling, perturbation metrics (cell-eval / arc-bench parity), GPU codec and pipeline breakdowns, training loader across datasets, query engine, and file operations.

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

For technical details, see [`docs/architecture.md`](docs/architecture.md), [`docs/format.md`](docs/format.md), [`docs/codec.md`](docs/codec.md), [`docs/api.md`](docs/api.md), [`docs/sharding.md`](docs/sharding.md), [`docs/multithreading.md`](docs/multithreading.md), [`docs/cloud.md`](docs/cloud.md), and [`docs/scanpy.md`](docs/scanpy.md).

## License

MIT
