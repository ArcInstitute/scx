# SCX Architecture

SCX (Sparse Cell eXpression System) is a co-designed **file format**, **compression codec**,
**query engine**, and **ML data loader** for single-cell RNA-seq data. It replaces AnnData/h5ad,
scipy.sparse, and the scanpy I/O layer with a unified Rust-native stack.

This document describes the high-level architecture, crate structure, and data flow.
For the full binary format specification, see [format.md](format.md) and [codec.md](codec.md).
For the API reference, see [api.md](api.md).

---

## Crate Dependency Graph

The workspace contains 14 crates (including `scx-integration-tests`). Dependencies flow bottom-up:

```
                        ┌──────────┐
                        │  pyscx   │  Python bindings (PyO3 + maturin)
                        └────┬─────┘
                             │ depends on all below
                        ┌────┴─────┐
                        │ scx-cli  │  CLI tool
                        └────┬─────┘
                             │ depends on all below
       ┌──────────────┬──────┼──────────────┬─────────────┬────────────-─┐
       │              │      │              │             │              │
┌──────┴──────┐ ┌─────┴────┐ │    ┌─────────┴──┐   ┌──────┴──────┐  ┌────┴──────┐
│ scx-engine  │ │ scx-ops  │ │    │ scx-cloud  │   │  scx-mtx    │  │ scx-accel │
│ query engine│ │ file ops │ │    │ cloud ops  │   │ MTX I/O     │  │ PCA/kNN/  │
└──────┬──────┘ └─────┬────┘ │    └─────────┬──┘   └──────┬──────┘  └──┬─┬──────┘
       │              │      │              │             │            │ │ [gpu]
       │    ┌─────────┘  ┌───┴────────┐     │             │            │ │
       │    │            │ scx-loader │     │             │            │ │
       │    │            │ ML loader  │     │             │            │ │
       │    │            └───┬────────┘     │             │            │ │
       │    │                │              │             │            │ │
       └────┴────────────────┼──────────────┴─────────────┴────────────┘ │
                             │                         ┌───────────┐     │
                       ┌──────┴──────┐                 │  scx-gpu  │◀────┘
                       │ scx-format  │                 │ GPU codec │
                       └──────┬──────┘                 │ + analysis│
                  ┌───────────┼────────────────────────┴────┬──────┘
                  │                     │
           ┌──────┴──────┐        ┌─────┴──────┐
           │  scx-codec  │        │ scx-sparse │
           │ compression │        │ CSR types  │
           └─────────────┘        └────────────┘

rscx (R bindings via extendr, depends on scx-format, scx-codec, scx-sparse, scx-engine, scx-ops)
```

### Crate Summary

| Crate | Role | Key modules |
|-------|------|-------------|
| **scx-codec** | Compression codecs (standalone, no I/O) | `rice`, `forbp`, `delta_golomb`, `bitstream`, `dispatch` |
| **scx-sparse** | CSR matrix type with scipy-compatible dtypes | `csr` (`ScxCsr`), `convert` (CSR ↔ dense) |
| **scx-format** | File layout, reading, and writing | `header`, `catalog`, `shard`, `reader`, `writer`, `codec_select`, `provenance`, `deletion_vectors` |
| **scx-ops** | File lifecycle operations | `append`, `delete`, `compact`, `merge`, `rollback`, `flock` |
| **scx-engine** | Lazy query engine with predicate pushdown | `pipeline`, `predicate`, `pushdown`, `projection`, `fused_ops`, `index`, `collect` |
| **scx-loader** | ML training data loader (triple-buffered) | `pipeline`, `io_stage`, `decode_stage`, `shuffle`, `projection`, `normalize`, `batch`, `python` |
| **scx-cloud** | Cloud access operations (S3, GCS, Azure) | `backend`, `cloud_optimize`, `explode`, `pack`, `pull`, `push`, `coalesce`, `cloud_reader` |
| **scx-mtx** | Matrix Market (MTX) I/O (always-on, no feature gate) | `read` (COO→CSR, TSV parsers, gzip), `write` (CSR→COO, gzipped output) |
| **scx-accel** | Rust-native analysis accelerators (opt. GPU via `gpu` feature) | `pca` (covariance eigh + randomized SVD, auto-routed), `neighbors` (HNSW kNN), `umap` (SGD embedding), `diffexp` (Wilcoxon with pre-ranking), `leiden` (Rust-native Leiden community detection), `harmony` (Harmony2 batch integration — soft k-means + ridge regression), `lisi` (exact-kNN Local Inverse Simpson Index), `pseudobulk`. GPU dispatch when `gpu` feature enabled. |
| **scx-gpu** | CUDA-accelerated codec decoding, GPU analysis, and GPU interop | `rice_decode`, `forbp_decode`, `sparse_to_dense`, `cusparse` (SpMM), `cusolver` (QR), `curand` (random matrix), `gpu_pca`, `gpu_knn` (CAGRA), `gpu_umap` (CUDA SGD), `gpu_harmony` (distance / softmax+penalty / L2-normalize / batched correction kernels), `gpu_preprocess` (fused normalize+log1p), `gds` |
| **scx-cli** | Command-line interface | `convert`, `info`, `validate`, `query`, `append`, `delete`, `compact`, `merge`, `rollback`, `benchmark`, cloud ops |
| **pyscx** | Python bindings via PyO3 | `experiment`, `anndata`, `ops`, `query`, `cloud`, `backed`, `accel`, `preprocess`, `lazy_transform`, `projected_agg` |
| **rscx** | R bindings via extendr | Seurat v5 + SingleCellExperiment interop, pipe-friendly query API |

> [!NOTE]
> `scx-loader` does **not** depend on `scx-engine` — it has its own streaming-optimized
> gene projection and fused normalization, designed for the hot-path requirements
> of ML training. `scx-cloud` does **not** depend on `scx-loader` — they are siblings.
> `scx-cloud` reuses `scx-engine` for predicate parsing (selective pull).
> `scx-mtx` is **always-on** (no feature gate) since MTX is pure text I/O with no
> HDF5 dependency. Both `scx-cli` and `pyscx` depend on it.
> `scx-accel` depends only on `scx-format` and `scx-sparse` — no engine/loader
> dependency. It uses `faer` for dense linear algebra (QR, SVD, eigendecomposition),
> `instant-distance` for HNSW kNN, `rand_chacha` for deterministic Leiden seeding,
> and `libc` for `malloc_trim` in the Leiden optimizer.
> With the `gpu` feature enabled, `scx-accel` gains an optional dependency on `scx-gpu`
> for GPU-accelerated PCA (cuSPARSE SpMM + cuSOLVER QR), kNN (cuVS CAGRA), and
> UMAP (native CUDA SGD kernel).

---

## File Format Overview

An SCX file (`.scx`) is a single packed binary file. One file contains
the expression matrix, all metadata, embeddings, graphs, and an internal
catalog for O(1) random access to any component.

```
┌────────────────────────────────────────────────────-─┐
│ FILE HEADER           (256 bytes, offset 0)          │  Magic b"SCX\x01", dimensions, codec, flags
├───────────────────────────────────────────────────-──┤
│ ROOT CATALOG          (offset 256, max 4096 bytes)   │  Compact index of section groups
├──────────────────────────────────────────────────-───┤
│ SECTIONS              (8-byte aligned)               │
│   obs metadata        (Arrow IPC)                    │
│   var metadata        (Arrow IPC)                    │
│   predicate indexes                                  │
│   X/csr/000000..N-1   (CSR shards — expression data) │
│   layers, obsm, obsp, uns, provenance                │
│   deletion vectors    (optional, Roaring Bitmap)     │
├───────────────────────────────────────────────────-──┤
│ FULL CATALOG          (at EOF)                       │  Per-section checksums + shard statistics
└──────────────────────────────────────────────────-───┘
```

**Key design properties:**
- **Single file** — easy to copy, stage, and manage
- **CSR-native** — row-major sparse storage matches 60-80% of scRNA-seq access patterns
- **Sharded** — expression matrix is split into shards of ~10K cells each, enabling parallel I/O and selective reads
- **Immutable fragments** — sections are never overwritten; appends write new data at EOF and update the catalog pointer atomically
- **Dual catalog** — root catalog (fixed position) for fast open; full catalog (at EOF) for random access

For the complete binary layout, see [format.md](format.md).

---

## Codec System

SCX includes domain-specific codecs that exploit the statistical properties of
UMI count data (55-65% ones, near-geometric tail, ~2.0 bits/value entropy).

### Codec Pipeline (codec_id = 1, "Scx1")

Each CSR shard's three arrays are encoded independently:

```
CSR Shard
  ├── indptr   ──→  Delta-Golomb-Rice    (monotonic u64 pointers)
  ├── indices  ──→  FOR-BP               (sorted column indices per row)
  └── values   ──→  Adaptive Rice        (non-zero UMI counts, ~2.2 bits/value)
```

| Codec | Array | Technique |
|-------|-------|-----------|
| **Delta-Golomb** | indptr | Delta encoding + Rice coding of deltas |
| **FOR-BP** | indices | Frame-of-Reference + Bit-Packing, 128-row blocks (SIMD BitPacker4x for ≥128 NNZ rows) |
| **Adaptive Rice** | values | Per-block (256 values) Rice coding with adaptive *k* parameter |

All codecs use **LSB-first** bit packing. The bitstream module (`scx-codec/src/bitstream.rs`)
provides the shared reader/writer.

### Codec Selection

The file header stores a default `codec_id`, but each shard header may **override** it:

| codec_id | Name | When used |
|----------|------|-----------|
| 0 | None | Raw LE arrays. Fast for GDS bypass. |
| 1 | Scx1 | Integer counts with median ≤ 8 (typical 10x UMI data). Uses SIMD BitPacker4x for FOR-BP index decode. |
| 2 | Zstd | Integer data with median > 8. General-purpose fallback. |
| 3 | Lz4Shuffle | Byte-shuffle pre-filter + LZ4 frame compression. Speed-optimized. Matches Zarr/Blosc style. |
| 4 | Pcodec | Pcodec (pco) lossless numerical compression. Optimal for float layers (log-normalized, embeddings). 7–16% better compression than Zstd on float data, with ~20–40% slower encode/decode. |

Auto-codec selection (`scx-format/src/codec_select.rs`) samples up to 10K non-zero
values per shard: integer data uses the median heuristic to choose Scx1 vs Zstd,
float data routes to Pcodec. LZ4+shuffle is available via `codec="lz4"` but not auto-selected.

---

## Data Model

### On-disk types vs In-memory types

| Component | On-disk | In-memory (`ScxCsr`) |
|-----------|---------|---------------------|
| indptr | `u64` | `i64` (matches scipy) |
| indices | `u16` or `u32` | `i32` (matches scipy) |
| values | `u8`, `u16`, `u32`, `f32`, `f16` | `f32` (always) |
| obs/var metadata | Arrow IPC | Arrow RecordBatch → pandas |

The `ScxCsr` struct (in `scx-sparse`) uses the exact memory layout scipy expects —
`i64` indptr, `i32` indices, `f32` data — enabling **zero-copy** transfer to Python
via PyO3's buffer protocol.

---

## Reader/Writer Architecture

### Writer (`scx-format/src/writer.rs`)

Uses an **atomic rename** strategy for crash safety:

```
1. Create temp file (experiment.scx.tmp.{pid})
2. Write sections sequentially starting at offset 4352
3. Write full catalog at EOF
4. pwrite() root catalog at offset 256
5. pwrite() header at offset 0
6. fsync()
7. rename() temp → final path  (atomic commit point)
```

### Reader (`scx-format/src/reader.rs`)

Opens a file via `mmap` and validates magic/version/checksums:

```
1. mmap the file (MADV_NORMAL default)
2. Parse 256-byte header
3. Parse root catalog (offset 256)
4. Parse full catalog (from header's full_catalog_offset)
5. Individual sections accessed via catalog offsets
```

All section access is via offset+length from the catalog — no sequential scanning.

**madvise hints** (Unix only, `#[cfg(unix)]`):
- `MADV_SEQUENTIAL` on the shard byte range during `assemble_shards_parallel()` — tells kernel to readahead aggressively for full reads
- `MADV_WILLNEED` on next N shards in `BackedCsrReader::read_shard_cached()` — prefetches upcoming shards after a cache miss
- `MADV_DONTNEED` after `read_shard_uncached()` — releases page cache for decoded shards during streaming aggregation (67% RSS reduction on 1M cells)

---

## Fragment/Manifest Model (scx-ops)

SCX uses an immutable-fragment model inspired by Lance and Delta Lake.
Sections are **never overwritten** — operations append new data and update the catalog pointer.

```
                       ┌────────────────────────────────────┐
   scx-ops             │                                    │
  ┌────────────┐       │     .scx file                      │
  │ append     │──────▶│  [original sections] [new shards]  │
  │            │       │  [new catalog v1 → all sections]   │
  │ delete     │──────▶│  [deletion vectors (Roaring)]      │
  │            │       │  [new catalog v2]                   │
  │ compact    │──────▶│  Clean rewrite → new file          │
  │            │       │                                    │
  │ rollback   │──────▶│  Header update → prev catalog      │
  │            │       │                                    │
  │ merge      │──────▶│  Streaming merge → new file        │
  └────────────┘       └────────────────────────────────────┘
```

| Operation | What it does | Writes |
|-----------|-------------|--------|
| **append** | Add new shards at EOF | New sections + new catalog |
| **delete** | Logical deletion via Roaring Bitmap | Deletion vectors section + new catalog |
| **compact** | Reclaim space, merge small shards | Entire new file (atomic rename) |
| **rollback** | Revert to previous catalog | Header-only update (pwrite) |
| **merge** | Combine multiple SCX files | New file with merged data |

Advisory `flock()` (`scx-ops/src/flock.rs`) prevents concurrent writers.
Multiple concurrent readers are always safe.

---

## Query Engine (scx-engine)

The query engine provides a **lazy pipeline** with Polars-style evaluation.
No data is read until `.collect()` is called.

### Pipeline Stages

```
open("file.scx")
  │
  ▼
filter_obs("tissue == 'lung'")     ← predicate parsing
  │
  ▼
select_genes(hvg_indices)          ← gene projection
  │
  ▼
normalize(target_sum=1e4)          ← fused with log1p when possible
  │
  ▼
.collect()                         ← parallel execution
  │
  ├── Catalog-level pushdown       (shard pruning via CategoryBitset stats)
  ├── Index-level pushdown         (row-level filtering via predicate indexes)
  ├── Parallel shard decode        (rayon thread pool)
  ├── Gene projection              (CSR column subset)
  └── Fused normalize + log1p     (single row scan)
```

### Key optimizations

- **Predicate pushdown (Level 1):** Per-shard statistics in the catalog (min/max for numerics,
  `CategoryBitset` for categoricals) enable skipping entire shards without reading any data.
- **Predicate pushdown (Level 2):** Predicate indexes (§3.5) provide row-level mappings
  within qualifying shards.
- **Operation fusion:** `normalize(1e4) + log1p()` → single fused CSR row scan
  (`scx-engine/src/fused_ops.rs`).
- **Parallel collection:** Qualifying shards are decoded and filtered in parallel
  via rayon (`scx-engine/src/collect.rs`).

---

## ML Training Loader (scx-loader)

The loader is a **triple-buffered Rust pipeline** designed to keep the GPU
saturated during model training. It does **not** go through `to_anndata()`.

### Pipeline Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Stage 1: I/O   │     │ Stage 2: Decode │     │  Stage 3:    │
│  (tokio async)  │────▶│ (rayon threads) │────▶│  GPU/Python  │
│                 │     │                 │     │              │
│  Read shard     │     │  Row shuffle    │     │  model.fwd() │
│  groups from    │     │  Gene project   │     │  loss.bwd()  │
│  .scx file      │     │  Sparse→dense   │     │  optim.step()│
│                 │     │  Normalize+log1p│     │              │
└─────────────────┘     └─────────────────┘     └──────────────┘
         bounded channel          bounded channel
         (back-pressure)          (back-pressure)
```

**"Zero Python on the hot path"** — all I/O, decompression, shuffling, sparse-to-dense
conversion, and normalization happen in Rust. Python touches only the training loop
and the forward/backward pass.

### Key Components

| Module | Responsibility |
|--------|---------------|
| `pipeline.rs` | `TrainingPipeline` coordinator, `LoaderConfig`, `MemoryBudget` |
| `io_stage.rs` | Async shard group reads, deletion vector filtering |
| `decode_stage.rs` | Parallel row scatter, obs metadata extraction |
| `shuffle.rs` | `ShardShuffler` (shard order) + `RowShuffler` (within-buffer Fisher-Yates) |
| `projection.rs` | `HvgProjection` — gene subset at decode time (~15× data reduction) |
| `normalize.rs` | Dense-row fused normalize + log1p |
| `batch.rs` | `Batch` struct — dense f32 matrix + obs columns |
| `python.rs` | `TrainingDataset` PyO3 class (implements `__iter__`/`__next__`) |

### Memory Budget

The loader auto-tunes `shard_group_size`, `prefetch_batches`, and `batch_size`
to fit within a configurable memory budget (default 512 MB):

```
total ≈ shard_buffer + batch_buffer + 50 MB overhead
  shard_buffer = (shard_group_size + 1) × decoded_shard_bytes
  batch_buffer = (prefetch_batches + 1) × batch_size × n_genes × 4
```

---

## Python Bindings (pyscx)

Built with PyO3 + maturin. Exposes two main interfaces:

### AnnData Bridge

```python
import pyscx

# Read
exp = pyscx.open("experiment.scx")    # → PyExperiment (lazy handle)
adata = exp.to_anndata()              # → AnnData (zero-copy CSR + Arrow→pandas)

# Write
pyscx.from_anndata(adata, "output.scx", codec="auto")
pyscx.from_10x("matrix.h5", "output.scx")
pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "output.scx")

# Export to Cell Ranger MTX directory
pyscx.to_mtx("output.scx", "/path/to/mtx_dir")
```

The `to_anndata()` path is **zero-copy** for the expression matrix — `ScxCsr`'s
`i64/i32/f32` arrays are handed directly to scipy via numpy buffer protocol.
Arrow metadata goes to pandas via pyarrow.

### Training Dataset

```python
dataset = pyscx.TrainingDataset(
    "experiment.scx",
    batch_size=1024,
    hvg_indices=hvg_array,
    normalize=True,
    log1p=True,
)

for batch in dataset:
    x = batch["X"]          # dense f32 numpy array
    obs = batch["obs"]      # dict of obs columns
```

`TrainingDataset` wraps the Rust `TrainingPipeline` and implements Python's
iterator protocol. It detects `num_workers > 0` and raises an error to prevent
CUDA fork deadlocks.

---

## Cloud Operations (scx-cloud)

The cloud crate provides cloud-native access to SCX data on S3, GCS, and Azure:

### Operations

| Operation | What it does | Direction |
|-----------|-------------|----------|
| **cloud-optimize** | Rewrite with front-of-file catalog | Local → local |
| **explode** | Packed `.scx` → exploded `.scxd/` directory | Local → local |
| **pack** | Exploded `.scxd/` → packed `.scx` | Local → local |
| **pull** | Stream from cloud → local packed file | Cloud → local |
| **push** | Stream from local → cloud exploded directory | Local → cloud |
| **CloudReader** | Direct cloud reads without download | Cloud → memory |

### Exploded Directory Layout

```
experiment.scxd/
├── _catalog.bin              # Full catalog
├── _header.bin               # 256-byte file header
├── obs.arrow                 # Obs metadata (Arrow IPC)
├── var.arrow                 # Var metadata
├── X/
│   ├── 000000.shard          # CSR shards (byte-identical to packed)
│   └── ...
├── obsm/                     # Optional
├── layers/                   # Optional
├── _provenance.bin           # Optional
├── _deletion_vectors.bin     # Optional
└── uns.json                  # Optional
```

`_catalog.bin` is uploaded last (atomic-publish semantics).

### Pull Pipeline

```
1. GET _catalog.bin + _header.bin
2. Parse catalog → know all sections
3. Download shard files in parallel (N=8 async tasks)
4. Reorder buffer → sequential writer thread
5. Write full catalog + front catalog + header
6. fsync + rename
```

**Selective pull**: `pull --filter "cell_type == 'T cell'"` downloads only matching
shards using catalog-level pushdown, reducing bandwidth by up to 20×.

### Authentication

Relies on `object_store`'s built-in credential chains:
- **GCS**: `GOOGLE_APPLICATION_CREDENTIALS` or instance metadata
- **S3**: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or instance profile
- **Azure**: `AZURE_STORAGE_ACCOUNT`/`AZURE_STORAGE_KEY` or managed identity

---

## CLI (scx-cli)

The CLI binary provides format conversion, inspection, validation, query, file
operations, and cloud access:

```bash
# Convert h5ad/10x to SCX (requires --features hdf5)
scx convert input.h5ad output.scx --codec auto --shard-size 10000
scx convert --to h5ad output.scx output.h5ad

# Convert Cell Ranger MTX ↔ SCX (always available, no feature flag)
scx convert /path/to/filtered_feature_bc_matrix/ output.scx
scx convert --to mtx output.scx /path/to/mtx_output/

# Inspect file metadata
scx info experiment.scx --json --history

# Verify all checksums
scx validate experiment.scx --verbose

# Query engine
scx query experiment.scx "cell_type == 'T cell'" --count
scx query experiment.scx "tissue == 'lung'" --output subset.scx --normalize 1e4 --log1p

# File operations
scx append atlas.scx --input new_batch.scx
scx delete experiment.scx --filter "is_doublet == True" --dry-run
scx compact experiment.scx --output compacted.scx
scx rollback experiment.scx --to-seq 3
scx merge batch1.scx batch2.scx batch3.scx --output atlas.scx

# Benchmarks
scx benchmark experiment.scx --compare-h5ad data.h5ad --runs 5 --json

# Cloud operations (requires --features cloud)
scx cloud-optimize experiment.scx --output cloud_ready.scx
scx explode experiment.scx experiment.scxd/
scx pack experiment.scxd/ experiment.scx
scx pull gs://bucket/experiment.scxd/ local.scx --filter "tissue == 'lung'"
scx push experiment.scx gs://bucket/experiment.scxd/ --parallelism 16
```

Feature flags: `hdf5` (h5ad/10x conversion, opt-in), `cloud` (cloud operations, opt-in).
MTX conversion is **always available** — no feature flag required.

---

## Data Flow Diagrams

### Conversion: h5ad → SCX

```
  h5ad file
      │
      ▼
  hdf5-rust          Read sparse matrix (CSR/CSC), obs, var, layers, uns
      │
      ▼
  CSC → CSR?         Transpose if stored as CSC (streaming scatter)
      │
      ▼
  Integer detect      Float32 counts → uint8/uint16/uint32 if lossless
      │
      ▼
  Auto-codec select   Per-shard: median ≤ 8 → Scx1, else → Zstd
      │
      ▼
  ScxWriter           Shard → encode → write sections → full catalog → atomic rename
      │
      ▼
  experiment.scx
```

### Conversion: MTX ↔ SCX

Cell Ranger MTX conversion uses the `scx-mtx` crate (always-on, no HDF5 dependency):

```
  filtered_feature_bc_matrix/
  ├── matrix.mtx[.gz]             MTX → SCX
  ├── barcodes.tsv[.gz]     ────────────────▶  experiment.scx
  └── features.tsv[.gz]     scx-mtx::read       scx-mtx::write
                            COO→CSR, gzip       CSR→COO, gzip
                                                   │
  output_dir/                                      │
  ├── matrix.mtx.gz        ◀────────────────       │
  ├── barcodes.tsv.gz           SCX → MTX          │
  └── features.tsv.gz      ◀───────────────────────┘
```

### Query: filter → collect

```
  scx.open()           Read header + catalogs (< 4 KB)
      │
      ▼
  .filter_obs()        Parse predicate, validate against schema
      │
      ▼
  .collect()
      │
      ├── Catalog pushdown     Skip non-matching shards via CategoryBitset
      ├── Index pushdown       Row-level pruning within qualifying shards
      ├── Parallel decode      rayon: decode + decompress qualifying shards
      ├── Row filtering        Apply predicate to obs metadata rows
      ├── Gene projection      CSR column subset for selected genes
      └── Fused ops            normalize + log1p in single pass
      │
      ▼
  QueryResult          ScxCsr + Arrow obs/var → AnnData
```

### Training: SCX → GPU

```
  experiment.scx
      │
      ▼
  ShardShuffler        Permute shard order per epoch
      │
      ▼
  I/O stage (tokio)    Read shard groups → bounded channel
      │
      ▼
  Decode stage (rayon) Decompress → RowShuffler → HvgProjection
      │                scatter into dense batch → normalize + log1p
      ▼
  Batch channel        bounded channel → Python iterator
      │
      ▼
  PyTorch              batch["X"].to(device) → model.forward()
```

### Analysis: SCX → GPU Accelerators

When `device="gpu"` is set (or `device="auto"` with a CUDA GPU present),
the analysis pipeline runs entirely on GPU:

```
  experiment.scx
      │
      ▼
  BackedCsrReader      Read shard bytes from disk
      │
      ▼
  GPU PCA (streaming)   For each shard:
      │                   1. htod copy raw shard bytes
      │                   2. decode_shard_gpu() → GpuCsr
      │                   3. GpuCsr → CusparseSpMatDescr (zero-copy)
      │                   4. cusparseSpMM: Y_shard += A_shard @ Ω
      │                   5. mean_correct_kernel on Y_shard rows
      │                 Then:
      │                   cuSOLVER QR → Power iteration → CPU SVD → GPU GEMM
      ▼
  GPU kNN (CAGRA)       cuVS CAGRA index build + all-queries search
      │                 on PCA embeddings (stays on GPU)
      ▼
  GPU UMAP              Native CUDA SGD kernel (edge-parallel)
      │                 or cuML fallback if available
      ▼
  AnnData               dtoh copy embeddings, kNN graph, UMAP coords
                        → adata.obsm["X_pca"], obsp, obsm["X_umap"]
```

Peak GPU memory: ~500 MB for 1M cells (dominated by Y and Q matrices
during PCA). kNN and UMAP operate on the (n_obs × n_components) dense
embeddings, which are small relative to the full expression matrix.

---

## Integrity and Safety

| Mechanism | Scope | Algorithm |
|-----------|-------|-----------|
| Per-shard checksum | Shard content (after header) | BLAKE3 truncated to 64 bits |
| Per-section checksum | Each catalog entry | BLAKE3 (32 bytes) |
| Catalog checksum | All catalog bytes | BLAKE3 (32 bytes) |
| File checksum | Header field | BLAKE3 truncated to 64 bits |
| Atomic writes | New files | temp file → fsync → rename |
| Concurrent safety | Reads during append | Immutable fragments + header pwrite |
| Advisory locking | Concurrent writers | `flock()` via `fs4` crate |

---

## Error Handling

Each crate defines its own error type via `thiserror`:

| Crate | Error type | Covers |
|-------|-----------|--------|
| `scx-format` | `ScxError` | I/O, format validation, checksum failures |
| `scx-codec` | `CodecError` | Bitstream exhaustion, malformed encoded data |
| `scx-engine` | `EngineError` | Schema validation, predicate parsing, pipeline errors |
| `scx-ops` | `OpsError` | Append/delete/compact/merge/rollback failures |
| `scx-loader` | `LoaderError` | Pipeline errors, memory budget, configuration |
| `scx-cloud` | `CloudError` | Object store errors, auth failures, missing sections |
| `scx-mtx` | `MtxError` | MTX parse errors, missing sidecar files, I/O |
| `scx-accel` | `AccelError` | PCA, kNN, UMAP, differential expression computation errors |
| `scx-gpu` | `GpuError` | CUDA runtime errors, kernel launch failures, GDS errors |
| `scx-sparse` | `CsrError` | Invalid CSR dimensions |

Readers return errors (never panic) on malformed input, including bitstream
exhaustion, invalid magic bytes, and unsupported format versions.

---

## Further Reading

- [format.md](format.md) — Binary format reference: header, catalogs, CSR shards, fragment/manifest, checksums
- [codec.md](codec.md) — Bit-level codec specification
- [api.md](api.md) — API reference for Rust, Python, and CLI
- [scanpy.md](scanpy.md) — Scanpy integration, backed mode, and accelerator usage
- [performance.md](performance.md) — Benchmark results and performance characteristics
- [testing.md](testing.md) — Test infrastructure and benchmarks
