# SCX API Reference

## Section Types

15 section types are defined in `scx-format/src/section.rs`:

```
ObsMetadata (0)        — Arrow IPC metadata for observations
ObsIndex (1)           — Arrow IPC index for observations
VarMetadata (2)        — Arrow IPC metadata for variables
VarIndex (3)           — Arrow IPC index for variables
CsrShard (4)           — Main expression matrix (row-major)
CscShard (5)           — Gene-major view (opt-in, Phase 3)
BitmapShard (6)        — Detection bitmap (Phase 3)
LayerCsrShard (7)      — Alternative expression layers
ObsmEmbedding (8)      — Embeddings (obsm)
ObspCsrShard (9)       — Cell-cell graphs (obsp)
UnsBlob (10)           — Unstructured metadata (JSON)
Provenance (11)        — Operation history
DeletionVectors (12)   — Logical deletion tracking (Roaring Bitmap)
ObsPredicateIndex (13) — Obs predicate index for query pushdown
VarPredicateIndex (14) — Var predicate index for query pushdown
```

## ScxReader (`scx-format/src/reader.rs`)

- `open(path)` — Open and validate file (mmap-based, validates magic/version/minimum size)
- `header()`, `root_catalog()`, `catalog()` — Access file metadata
- `n_obs()`, `n_vars()`, `nnz()` — Quick dimension access
- `read_obs()`/`read_var()` — Arrow RecordBatch metadata
- `read_csr_shard(idx)` — Single shard as `(Vec<i64>, Vec<i32>, Vec<f32>)`
- `read_all_csr_shards()` — Full matrix as `ScxCsr` (parallel via rayon)
- `read_layer(name)` — Named layer as `ScxCsr`
- `layer_names()` — List available layer names
- `read_obsm(name)` / `read_all_obsm()` — Embeddings as Arrow RecordBatch
- `read_uns()` — Unstructured metadata as `serde_json::Value`
- `read_provenance()` — Operation history
- `validate()` — Check all section BLAKE3 checksums
- `section_bytes(entry)` — Direct byte access to a section
- `read_obs_schema()` / `read_var_schema()` — Arrow schema (without data)
- `read_obs_predicate_index_bytes()` / `read_var_predicate_index_bytes()` — Predicate index raw bytes
- `read_deletion_vectors()` — Roaring Bitmap deletion vectors
- `read_all_csr_shards_filtered()` — Full matrix with deletion vector filtering
- `read_shard_header()` / `read_raw_shard_bytes()` / `read_shard_from_entry()` — Low-level shard access
- `mmap()` — Direct mmap access to the underlying file

## ScxWriter (`scx-format/src/writer.rs`)

- `new(path, header)` — Create writer (writes to temp file)
- `write_obs(batch)`/`write_var(batch)` — Arrow IPC metadata
- `write_csr_shard(indptr, indices, values, ...)` — CSR expression data
- `write_layer_csr_shard(name, ...)` — Named layer CSR data
- `write_obsp_shard(name, ...)` — Cell-cell graph CSR data
- `write_obsm(name, batch)` — Embeddings
- `write_uns(json)` — JSON metadata
- `write_provenance(operations)` — Operation history
- `write_csc_shard(...)` — CSC (column-major) expression data
- `write_obs_predicate_index(data)` / `write_var_predicate_index(data)` — Predicate indexes for pushdown
- `write_deletion_vectors(dv)` — Roaring Bitmap deletion vectors
- `write_raw_shard(raw_bytes, section_type, name, stats, nnz)` — Pre-encoded shard passthrough
- `set_shard_column_stats(column_stats)` — Per-column shard statistics for catalog
- `finish()` — Atomic write: full catalog at EOF -> pwrite root catalog -> pwrite header -> fsync -> rename

## Codec Selection (`scx-format/src/codec_select.rs`)

- `select_codec(values, encoding)` — Auto-select Scx1 vs Zstd based on median value
  - Integer values with median ≤ 8 → Scx1 (Rice coding)
  - Integer values with median > 8 → Zstd
  - Float values → always Zstd

## Provenance

- `ProvenanceEntry`: timestamp, action, tool, params_json, input_checksums
- Auto-populated on `ScxWriter::finish()` with operation info
- `scx info` displays full provenance history
- Read/write via `ScxReader::read_provenance()` / `ScxWriter::write_provenance()`

## BackedCsrReader (`scx-format/src/backed.rs`)

- `new(reader, cache_shards)` — Create backed reader from `ScxReader` with LRU shard cache
- `read_rows(start, end)` → `ScxCsr` — Decode and concatenate rows from relevant shards
- `read_row_indices(indices)` → `ScxCsr` — Decode specific rows by index (fancy indexing)
- `read_shard_cached(idx)` → `ScxCsr` — Read shard through LRU cache (clones on hit)
- `read_shard_uncached(idx)` → `ScxCsr` — Read shard bypassing cache (preferred for streaming)
- `row_sums()` / `col_sums()` — Streaming per-row/column sums
- `row_nnz()` / `col_nnz()` — Streaming per-row/column NNZ
- `row_var()` / `col_var()` — Streaming per-row/column variance
- `row_max()` / `col_max()` / `row_min()` / `col_min()` — Streaming extrema
- `total_nnz()` — Total NNZ across all shards
- `col_means_and_sum_sq(zero_center)` — Single-pass column statistics for PCA
- Masked variants (deletion-vector aware): `col_sums_masked(kept_rows)`, `col_nnz_masked(kept_rows)`, `col_max_masked(kept_rows)`, `col_min_masked(kept_rows)`, `col_var_masked(kept_rows)`

## ShardSource Trait (`scx-format/src/shard_source.rs`)

A uniform interface for streaming CSR data shard-by-shard, enabling algorithms like PCA to process data without materializing the full matrix. Defined in `scx-format`, available to both `scx-accel` and `pyscx`.

```rust
pub trait ShardSource {
    fn n_shards(&self) -> usize;
    fn n_obs(&self) -> usize;
    fn n_vars(&self) -> usize;
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }
    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr>;
    fn col_means_and_sum_sq(&self, zero_center: bool)
        -> Result<(Option<Vec<f64>>, Vec<f64>)>; // default impl provided
}
```

**Implementations:**

| Type | Crate | Behavior |
|------|-------|---------|
| `BackedCsrReader` | `scx-format` | Reads raw decoded shards via `read_shard_uncached()` |
| `LazyShardSource` | `pyscx` (internal) | Applies per-shard transforms (NormalizeTotal, Log1p, RowScale), column projection (`project_csr`), and deletion vector filtering before returning |

**Used by:** `scx-accel::randomized_pca()`, `streaming_spmm_forward()`, `streaming_spmm_transpose()` — all generic over `<S: ShardSource>` for zero-cost abstraction across the crate boundary.

**Design note:** The trait is defined in `scx-format` (not `scx-accel`) so that `pyscx`'s `LazyShardSource` can implement it without creating a dependency on `scx-accel`. PCA functions in `scx-accel` are generic (`<S: ShardSource>`) rather than using `&dyn ShardSource` to allow monomorphization.

## scx-ops — File Operations

### `scx_ops::append(path, obs, indptr, indices, values, ...) → Result<()>`
Append new cells at EOF. Advisory flock for concurrent safety.

### `scx_ops::mark_deleted(path, cell_indices) → Result<u64>`
Logical deletion via Roaring Bitmap deletion vectors. Returns total deleted count.

### `scx_ops::compact(input, output) → Result<()>`
Rewrite file reclaiming deleted/orphaned space.

### `scx_ops::rollback(path) → Result<()>` / `rollback_to(path, seq) → Result<()>`
Revert to previous (or specific) manifest version — header-only update.

### `scx_ops::merge(inputs, output) → Result<()>`
Streaming merge of multiple SCX files into one.

## scx-engine — Query Engine

### `QueryPipeline`
Lazy pipeline builder — no I/O until `.collect()`:

```rust
QueryPipeline::open("file.scx")?
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")?
    .select_genes(hvg_indices)
    .with_normalize(1e4)
    .with_log1p()
    .limit(1000)
    .collect()?   // → QueryResult { x: ScxCsr, obs, var, skipped_shards, total_shards }
```

- Schema validation at construction time (fail-fast)
- Two-level predicate pushdown: catalog shard pruning + index row pruning
- Fused normalize+log1p in single CSR row scan
- Parallel shard processing via rayon

## scx-loader — Training Data Loader

### `TrainingPipeline`
Triple-buffered Rust pipeline: tokio I/O → rayon decode → Python/GPU.

```rust
let pipeline = TrainingPipeline::new("file.scx", LoaderConfig {
    batch_size: 1024,
    hvg_indices: Some(hvg_array),
    normalize: true,
    log1p: true,
    ..Default::default()
})?;
pipeline.start_epoch();
while let Some(batch) = pipeline.next_batch() {
    // batch.x: dense f32 matrix, batch.obs_columns: Vec<Vec<f32>>
}
```

## scx-cloud — Cloud Operations

### `cloud_optimize(input, output) → Result<()>`
Rewrite file with front-of-file catalog for single-read cloud opens.

### `explode(input, output_dir) → Result<()>`
Packed `.scx` → exploded `.scxd` directory (one file per section).

### `pack(input_dir, output) → Result<()>`
Exploded `.scxd` directory → packed `.scx` (cloud-ready by default).

### `pull(source, dest, options) → Result<PullStats>`
Streaming cloud → local packed file. Supports parallel downloads and selective filter.

### `push(source, dest, options) → Result<PushStats>`
Streaming local → cloud exploded directory. Parallel uploads.

### `CloudReader::open_cloud(url) → Result<CloudReader>`
Direct cloud reads without full download. Supports `.scxd/`, cloud-ready `.scx`, and non-cloud-ready `.scx`.

## scx-gpu — GPU Analysis

GPU-accelerated analysis APIs. Requires CUDA Toolkit ≥ 12.0 at build time.

### cuSPARSE SpMM

#### `spmm_csr(handle, stream, a, b, c, m, k, n, alpha, beta) → Result<(), GpuError>`
Sparse × dense matrix multiply: C = α·A·B + β·C. A is GPU-resident CSR, B/C are dense column-major f32.

#### `spmm_csr_transpose(handle, stream, a, b, c, m, k, n, alpha, beta) → Result<(), GpuError>`
Transposed SpMM: C = α·A^T·B + β·C.

### cuSOLVER Dense Operations

#### `gpu_qr_q(handle, stream, a, m, n) → Result<CudaSlice<f32>, GpuError>`
Economy QR decomposition on GPU: A = Q·R. Returns Q (m × n). Uses `cusolverDnSgeqrf` + `cusolverDnSorgqr`.

### cuRAND

#### `random_gaussian_gpu(stream, rows, cols, seed) → Result<CudaSlice<f32>, GpuError>`
Generate a random Gaussian matrix directly on GPU via cuRAND XORWOW generator.

### GPU PCA Pipeline

#### `gpu_randomized_pca(dev, reader, n_components, n_oversamples, n_power_iterations, zero_center, seed) → Result<GpuPcaResult, GpuError>`
Complete GPU-accelerated randomized PCA. Streams SpMM shard-by-shard via cuSPARSE, QR via cuSOLVER, SVD via CPU faer, final projection via GPU GEMM. Returns `GpuPcaResult { embeddings, components, variance_explained, variance_ratio, mean }`.

#### `mean_correct_gpu(dev, y, mc, n_obs, k) → Result<(), GpuError>`
Mean-centering correction kernel: Y[i,j] -= mc[j] for all rows.

### GPU kNN

#### `gpu_knn_cagra(dev, embeddings, n_obs, n_dims, n_neighbors) → Result<GpuKnnResult, GpuError>`
Build kNN graph on GPU using NVIDIA CAGRA (cuVS). L2 distance, optimized for PCA embeddings. Returns `GpuKnnResult { indices, distances, n_obs, n_neighbors }`.

#### `cuvs_available() → bool`
Check if `libcuvs.so` is available at runtime.

### GPU UMAP

#### `gpu_umap_native(dev, knn, n_components, min_dist, spread, n_epochs, seed) → Result<GpuUmapResult, GpuError>`
GPU UMAP via native CUDA SGD optimization kernel. Edge-parallel with `atomicAdd` for concurrent embedding updates.

### GPU Preprocessing

#### `gpu_normalize_log1p(dev, csr, target_sum) → Result<(), GpuError>`
Fused per-row normalize_total + log1p on GPU-resident CSR (in-place).

#### `gpu_normalize(dev, csr, target_sum) → Result<(), GpuError>`
Per-row normalize_total on GPU-resident CSR (in-place).

#### `gpu_log1p(dev, csr) → Result<(), GpuError>`
Element-wise log1p on GPU-resident CSR (in-place).

#### `gpu_apply_fused_ops(dev, csr, normalize, log1p, target_sum) → Result<(), GpuError>`
Apply configurable fused preprocessing ops on GPU-resident CSR.

### GpuError Variants

| Variant | Description |
|---------|-------------|
| `CudaError(String)` | CUDA runtime/driver error |
| `KernelLaunchFailed(String)` | Kernel launch failure |
| `GdsUnavailable(String)` | GDS not available |
| `DeviceNotFound(usize)` | GPU device not found |
| `InvalidShard(String)` | Malformed shard data |
| `ShapeMismatch { expected, got }` | Matrix dimension mismatch |
| `CodecError(CodecError)` | Codec decode error |
| `CuSparseError(String)` | cuSPARSE API error |
| `CuSolverError(String)` | cuSOLVER QR/SVD error |
| `CuRandError(String)` | cuRAND generation error |
| `StreamError(String)` | CUDA stream error |
| `OutOfMemory(String)` | GPU OOM |
| `ModuleLoadError(String)` | PTX/CUDA module load failure |
| `CuVsError(String)` | cuVS/CAGRA error |
| `LibraryNotFound(String)` | Runtime library missing (libcuvs.so) |

## Python API (`pyscx`)

### Module-level functions

- `pyscx.open(path) -> PyExperiment` — Open SCX file (local)
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None)` — Write AnnData to SCX
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None)` — 10x HDF5 to SCX
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None)` — Cell Ranger MTX directory (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`) to SCX. Default shard size is 16384.
- `pyscx.to_mtx(scx_path, output_dir)` — SCX to Cell Ranger–style MTX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`).
- `pyscx.iter_chunks(adata, chunk_size="shard")` — Shard-aligned or fixed-size chunk iterator
- `pyscx.preprocess(source, target, ops, target_sum=None)` — Streaming shard-by-shard preprocessing
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — Save transformed data as layer

### File operations
- `pyscx.append(target, input, codec=None, shard_size=None)` — Append from SCX file
- `pyscx.append_from_anndata(target, adata, codec=None, shard_size=None)` — Append from AnnData
- `pyscx.mark_deleted(path, cell_indices)` — Logical deletion
- `pyscx.compact(input, output)` — Rewrite reclaiming space
- `pyscx.rollback(path, to_seq=None)` — Revert to previous manifest
- `pyscx.merge(inputs, output)` — Merge multiple files

### Cloud operations (requires `--features cloud`)
- `pyscx.pull(source, dest, filter=None, parallelism=None)` — Streaming cloud → local
- `pyscx.push(source, dest, parallelism=None)` — Streaming local → cloud
- `pyscx.cloud_optimize(input, output=None)` — Front-of-file catalog
- `pyscx.explode(input, output)` — Packed → exploded directory
- `pyscx.pack(input, output)` — Exploded → packed
- `pyscx.open_cloud(url) -> PyCloudExperiment` — Direct cloud reads

### PyExperiment

- `to_anndata(backed=False, cache_shards=4, var_names=None, obs_filter=None, layers=None)` — Convert to AnnData
  - `var_names`: list of gene names to project (column subset)
  - `obs_filter`: predicate string for cell filtering (uses query engine with pushdown in non-backed mode)
  - `layers`: list of layer names to load (default: all)
  - `backed`: when True, X and layers are lazy `ScxBackedSparseDataset` instances
- `query() -> PyQueryPipeline` — Start lazy query pipeline
- `mark_deleted(mask)` — Delete cells matching boolean array
- `validate()` — Check checksums, returns list of `(section_name, passed)`
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `layer_names`

### PyQueryPipeline

- `filter_obs(expr)` / `filter_var(expr)` — Predicate filtering
- `select_genes(indices)` — Gene projection
- `with_normalize(target_sum=1e4)` — Total-count normalization
- `with_log1p()` — Log1p transformation
- `limit(n)` — Row limit
- `collect() -> PyQueryResult` — Execute pipeline
- `count() -> int` — Convenience: matching cell count

### PyQueryResult

- `to_anndata()` — Convert result to AnnData (zero-copy CSR)
- `to_csr()` — Return just the scipy CSR matrix
- Properties: `n_obs`, `n_vars`, `nnz`, `skipped_shards`, `total_shards`

### PyCloudExperiment

Returned by `pyscx.open_cloud()`. Metadata-only handle for cloud-hosted SCX files.
Does **not** support `to_anndata()`, `query()`, or `validate()` — use `pyscx.pull()` to
download the file first for full data access.

- `n_obs` `→ int` — Number of observations (cells)
- `n_vars` `→ int` — Number of variables (genes)
- `nnz` `→ int` — Total non-zero entries
- `shard_count` `→ int` — Number of CSR shards in the file
- `format_version` `→ int` — SCX format version
- `codec_id` `→ int` — Default codec ID

### pyscx.accel — Rust-Native Accelerators

All accelerators write results to standard AnnData slots (same as scanpy), so downstream functions work identically.

- `pyscx.accel.pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto")` — Randomized SVD PCA with streaming SpMM. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`. On GPU: cuSPARSE SpMM + cuSOLVER QR (f32).
- `pyscx.accel.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — kNN graph + UMAP-style connectivities. CPU: HNSW. GPU: CAGRA (cuVS). Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `pyscx.accel.umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — Spectral-init SGD UMAP. GPU: native CUDA kernel or cuML fallback. Writes `obsm["X_umap"]`.
- `pyscx.accel.rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, log_transformed=False, stratify_by=None, min_cells_per_stratum=50)` — Parallel Wilcoxon rank-sum with BH correction. Writes `uns["rank_genes_groups"]`, or returns DataFrame when `stratify_by` is set.
- `pyscx.accel.pseudobulk_dex(adata, groupby, test_col, reference, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50)` — Streaming pseudobulk aggregation (Rust) + pydeseq2 testing. Returns DataFrame. Requires optional `pydeseq2` dependency.
- `pyscx.accel.leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=-1, device="auto")` — Leiden community detection on kNN graph. Reads `obsp["connectivities"]` (from `neighbors()`). GPU: cuGraph Leiden (up to 47× faster). CPU fallback: leidenalg via igraph. Writes `adata.obs[key_added]` (categorical) and `adata.uns["leiden"]` (params + backend metadata). GPU and CPU may produce different partitions due to algorithmic differences; compare via ARI/NMI.
- `pyscx.accel.normalize_total(adata, target_sum=10000.0)` — Materialization-free row normalization. On `ScxBackedSparseDataset`: computes row sums via streaming, creates `ScxLazyTransformedDataset` wrapper. On `ScxLazyTransformedDataset`: appends `NormalizeTotal` transform to chain. On scipy CSR: delegates to `sc.pp.normalize_total()`.
- `pyscx.accel.log1p(adata)` — Materialization-free log1p. On `ScxBackedSparseDataset`: creates `ScxLazyTransformedDataset` with `Log1p` transform. On `ScxLazyTransformedDataset`: appends `Log1p` to chain (fuses with preceding `NormalizeTotal` when possible). On scipy CSR: delegates to `sc.pp.log1p()`.
- `pyscx.accel.gpu_info() → dict` — Query GPU device info: `{'device': ..., 'total_vram_gb': ..., 'free_vram_gb': ...}`. Returns `None` if no GPU available.
- `pyscx.accel.estimate_gpu_memory(adata, operation, **kwargs) → dict` — Estimate GPU VRAM required for an operation. Returns `{'required_gb': float, 'fits_in_vram': bool}`. Supported operations:
  - `"pca"`: kwargs `n_components` (default 50), `n_oversamples` (default 10), `shard_size` (default 16384)
  - `"knn"`: kwargs `n_neighbors` (default 15), `n_dims` (default 50)
  - `"umap"`: kwargs `n_components` (default 2)
  - `"leiden"`: no additional kwargs

  Raises `ValueError` for unknown operations. Note: estimates are approximate — cuSOLVER QR workspace may be undercounted by ~1.5×.
- `pyscx.accel.calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True)` — Streaming QC metrics for backed/lazy data without materialization. Computes per-cell `n_genes_by_counts`, `total_counts` and per-gene `n_cells_by_counts`, `total_counts`. Supports `qc_vars` for gene subsets (e.g., `["mt"]` for mitochondrial percentage). When `inplace=True`, writes to `adata.obs`/`adata.var`; when `False`, returns `(obs_df, var_df)`. Falls back to `sc.pp.calculate_qc_metrics()` for scipy/dense.
- `pyscx.accel.filter_cells(adata, min_genes=None, max_genes=None, min_counts=None, max_counts=None)` — Non-materializing cell QC filter for backed/lazy data. Computes row NNZ and/or row sums via streaming, builds a boolean mask, and updates the deletion vector (`kept_to_global`) on `ScxBackedSparseDataset` or `ScxLazyTransformedDataset`. Also slices `adata.obs`, `adata.obsm`, and updates `adata.layers` with the new deletion vector. Falls back to `sc.pp.filter_cells()` for scipy/dense.
- `pyscx.accel.filter_genes(adata, min_cells=None, max_cells=None, min_counts=None, max_counts=None)` — Non-materializing gene QC filter for backed/lazy data. Computes column NNZ and/or column sums via streaming, builds a boolean mask, and sets `col_projection` on `ScxBackedSparseDataset` or `ScxLazyTransformedDataset`. Slices `adata.var` to match. Composes with existing column projections. Falls back to `sc.pp.filter_genes()` for scipy/dense.
- `pyscx.accel.subset_obs(adata, mask_or_indices)` — Subset observations (cells) without materializing. Accepts a **boolean numpy mask** or **integer index array**. Creates a new deletion vector (`kept_to_global`) on the backing dataset, slices `adata.obs` and `adata.obsm`, and updates `adata.layers`. Composes correctly with existing deletion vectors. Falls back to numpy slicing for non-SCX data.

  > [!WARNING]
  > **Integer index semantics differ from NumPy.** When `mask_or_indices` is an integer array, it is internally converted to a boolean mask. This means:
  > - **Duplicate indices are silently collapsed** — `[0, 0, 5]` is equivalent to `[0, 5]`
  > - **Order is not preserved** — `[5, 0, 10]` produces the same result as `[0, 5, 10]`
  >
  > This differs from NumPy's fancy indexing where `a[[5, 0, 10]]` returns rows in the order `[5, 0, 10]` with duplicates preserved. Use a boolean mask for unambiguous results.

### ScxBackedSparseDataset

PyO3 class for backed-mode lazy access to the main expression matrix (`adata.X`). Data stays on disk; only requested shards are decoded on access. Registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`, adjusted for deletion vectors and column projection
- `dtype` `→ numpy.dtype` — Always `float32`
- `format` `→ str` — Always `"csr"`
- `backend` `→ str` — Always `"scx"`
- `ndim` `→ int` — Always `2`
- `non_negative` `→ bool` — Whether the data is known to be non-negative (enables `(X > 0).sum() → getnnz()` short-circuit)
- `nnz` `→ int` — Total non-zero count
- `n_shards` `→ int` — Number of CSR shards in the backing file

**Column projection:**
- `set_col_projection(col_indices)` — Restrict all access and aggregation to a subset of columns. Used internally by `to_anndata(var_names=...)` and streaming QC with gene subsets (`qc_vars`).

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, return scipy CSR.
- `__getitem__(row_slice, col_slice)` — Row decode + column post-filter.
- Supports integer, slice, boolean mask, and fancy indexing.

**Aggregation (streaming, no materialization):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums via native Rust streaming.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance (two-pass).
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts per column or row.
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards → full CSR.
- `copy()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()`.
- `toarray()` `→ numpy.ndarray` — Dense array.
- `tocsr()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()` (scipy compat).
- `tocsc()` `→ scipy.sparse.csc_matrix` — Materialize and convert to CSC.
- `.A` `→ numpy.ndarray` — Dense array property (scipy compat).

**Comparison operators:**
- `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` — Return `ScxComparisonResult` for lazy boolean operations.

**Arithmetic:**
- `__truediv__(other)` — If `other` is a per-row vector, returns `ScxLazyTransformedDataset` with `RowScale(1/factors)` (lazy). Otherwise materializes.
- `__mul__(other)` — If `other` is a per-row vector, returns `ScxLazyTransformedDataset` with `RowScale(factors)` (lazy). Otherwise materializes.
- `__add__(other)` — Materializes and adds.
- `__sub__(other)` — Materializes and subtracts.
- `__matmul__(other)` — Matrix multiply (materializes).
- `multiply(other)` — Element-wise Hadamard product (materializes).
- `power(n)` — Element-wise power (materializes).

**Introspection:**
- `shard_boundaries()` `→ list[(int, int)]` — List of `(row_start, row_end)` tuples per shard.

### ScxBackedLayerDataset

PyO3 class for backed-mode layer access (e.g., `adata.layers["raw_counts"]`). Wraps a `ScxBackedSparseDataset` for a named layer. Registered with `anndata.abc.CSRDataset`.

- Same interface as `ScxBackedSparseDataset` (`shape`, `dtype`, `format`, `backend`, `ndim`, `__getitem__`, `to_memory`, `toarray`, `tocsr`, `tocsc`, `copy`, `sum`, `mean`, `var`, `getnnz`, `max`, `min`)
- `layer_name` `→ str` — Name of the backing layer

### ScxComparisonResult

Lazy comparison result returned by `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` on `ScxBackedSparseDataset` and `ScxLazyTransformedDataset`. Exposed as `_ComparisonResult` in Python.

**Key optimization:** `(X > 0).sum(axis)` short-circuits to `getnnz(axis)` without materializing the full boolean matrix. This is the critical path for `sc.pp.calculate_qc_metrics()`, `sc.pp.filter_cells()`, and `sc.pp.filter_genes()`. The short-circuit only activates for non-negative data (raw counts, normalized, log1p).

- `shape` `→ (int, int)`, `dtype` `→ numpy.dtype` (bool), `ndim` `→ int` (2)
- `sum(axis=None)` — Short-circuits `(X > 0).sum()` → `getnnz()` for non-negative data; otherwise materializes.
- `getnnz(axis=None)`, `mean(axis=None)` — Materialize and delegate.
- `toarray()`, `tocsr()`, `tocsc()` — Materialize to dense/sparse.
- `multiply(other)` — Element-wise product (materializes).
- `.A` `→ numpy.ndarray` — Dense array property.

### ScxLazyTransformedDataset

PyO3 class wrapping `ScxBackedSparseDataset` with chained per-row transforms. Created by `pyscx.accel.normalize_total()` and `pyscx.accel.log1p()`. Implements the same interface as `ScxBackedSparseDataset` and is registered with `anndata.abc.CSRDataset`.

**Properties:**
- `shape` `→ (int, int)` — `(n_obs, n_vars)`
- `dtype` `→ numpy.dtype` — Always `float32`
- `format` `→ str` — Always `"csr"`
- `ndim` `→ int` — Always `2`
- `backend` `→ str` — Always `"scx-lazy"` (distinguishes from `ScxBackedSparseDataset.backend` which is `"scx"`)
- `non_negative` `→ bool` — Whether the transformed data is non-negative

**Slicing:**
- `__getitem__(row_slice)` `→ scipy.sparse.csr_matrix` — Decode requested shards, apply all transforms in order, return scipy CSR. Peak memory = 1 shard.
- `__getitem__(row_slice, col_slice)` — Row decode + transform + column projection.

**Aggregation (streaming through transforms):**
- `sum(axis=0|1)` `→ numpy.ndarray` — Column or row sums of transformed data.
- `mean(axis=0|1)` `→ numpy.ndarray` — Column or row means of transformed data.
- `var(axis=0|1)` `→ numpy.ndarray` — Column or row variance (two-pass streaming).
- `getnnz(axis=0|1)` `→ numpy.ndarray` — Non-zero counts (unchanged by normalize/log1p).
- `max(axis=0|1)` `→ numpy.ndarray` — Column or row max of transformed data.
- `min(axis=0|1)` `→ numpy.ndarray` — Column or row min of transformed data.

**Materialization:**
- `to_memory()` `→ scipy.sparse.csr_matrix` — Decode all shards + apply transforms → full CSR.
- `copy()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()`.
- `toarray()` `→ numpy.ndarray` — Dense array (via `to_memory().toarray()`).
- `tocsr()` `→ scipy.sparse.csr_matrix` — Same as `to_memory()` (scipy compat).
- `tocsc()` `→ scipy.sparse.csc_matrix` — Materialize and convert to CSC.
- `.A` `→ numpy.ndarray` — Dense array property (scipy compat, same as `toarray()`).

**Comparison operators:**
- `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` — Return `ScxComparisonResult` for lazy boolean operations.

**Arithmetic:**
- `__truediv__(other)` — If `other` is a per-row vector, appends `RowScale` transform (lazy). Otherwise materializes.
- `__mul__(other)` — If `other` is a per-row vector, appends `RowScale` transform (lazy). Otherwise materializes.
- `__add__(other)` — Materializes and adds.
- `__sub__(other)` — Materializes and subtracts.
- `__matmul__(other)` — Matrix multiply (materializes).
- `multiply(other)` — Element-wise Hadamard product (materializes).
- `power(n)` — Element-wise power (materializes).

**Introspection:**
- `nnz` `→ int` — Total non-zero count in backing file.
- `n_shards` `→ int` — Number of CSR shards in backing file.
- `shard_boundaries()` `→ list[(int, int)]` — List of `(row_start, row_end)` tuples per shard.

**Transform chain:**
- Backed data → `NormalizeTotal` → `Log1p` is fused into `ln(x × target_sum / row_sum + 1)` in a single pass.
- `repr()` shows the transform chain: `ScxLazyTransformedDataset(shape=(1000000, 33694), transforms=[NormalizeTotal, Log1p])`

### scVI Integration (`pyscx.scx_integrations.scvi`)

Pure-Python PyTorch Lightning DataModule wrapping `TrainingDataset` for scVI model training. Requires `lightning` or `pytorch_lightning`.

- `ScxDataModule(scx_path, batch_size=1024, hvg_indices=None, normalize=True, log1p=True, target_sum=1e4, seed=42, **kwargs)` — Creates a PyTorch Lightning `LightningDataModule`.
  - `scx_path` — Path to the `.scx` file.
  - `batch_size` — Mini-batch size (default: 1024).
  - `hvg_indices` — Gene indices for HVG projection. `None` = all genes.
  - `normalize` — Apply total-count normalization (default: `True`).
  - `log1p` — Apply log1p transformation (default: `True`).
  - `target_sum` — Normalization target sum (default: `1e4`).
  - `seed` — RNG seed for reproducibility (default: `42`).
  - `**kwargs` — Additional keyword arguments passed to `TrainingDataset`.
- Properties: `n_obs`, `n_vars`, `n_output_genes`
- `train_dataloader()` `→ DataLoader` — Uses `batch_size=None` and `num_workers=0` (Rust handles batching and threading internally).
- `val_dataloader()` `→ None` — Validation not currently supported.

```python
from pyscx.scx_integrations.scvi import ScxDataModule
import scvi

dm = ScxDataModule("atlas.scx", batch_size=1024, hvg_indices=hvg_array)
model = scvi.model.SCVI(dm.adata_manager)
model.train(datamodule=dm)
```

### TrainingDataset

```python
dataset = pyscx.TrainingDataset("file.scx", batch_size=1024,
    hvg_indices=hvg_array, normalize=True, log1p=True)
for batch in dataset:
    x = batch["X"]      # dense f32 numpy array
    obs = batch["obs"]   # dict of obs columns
```

## CLI (`scx-cli`)

### Core
- `scx convert <input> <output> [--from h5ad|10x] [--to h5ad] [--codec auto|none|scx1|zstd] [--shard-size N]`
- `scx info <file> [--json] [--history]`
- `scx validate <file> [--verbose]`
- `scx benchmark <file> [--compare-h5ad <path>] [--runs N] [--json]`

### File operations
- `scx append <target> --input <source> [--codec auto|none|scx1|zstd] [--shard-size N]`
- `scx delete <file> --filter <expr> [--dry-run]`
- `scx compact <input> --output <path> [--force]`
- `scx rollback <file> [--to-seq N]`
- `scx merge <file1> <file2> [<...>] --output <path>`
- `scx query <file> <filter> [--count] [--output <path>] [--select-genes <path>] [--normalize N] [--log1p] [--limit N] [--json]`
- `scx subset <input> [--output <path>] [--filter <expr>] [--genes <path>] [--dry-run] [--shard-size N] [--codec auto|none|scx1|zstd]` — Extract a subset of cells and/or genes into a new SCX file
- `scx build-csc <input> <output> [--memory-limit 4G] [--force]` — Build CSC (column-major) shards from existing CSR data
- `scx upgrade <input> [output] [--in-place]` — Upgrade an SCX file to the latest format version

### Cloud operations (`--features cloud`)
- `scx cloud-optimize <input> [--output <path>]`
- `scx explode <input> <output>`
- `scx pack <input> <output>`
- `scx pull <source-url> <dest> [--parallelism N] [--no-cloud-ready] [--filter <expr>]`
- `scx push <source> <dest-url> [--parallelism N]`

Note: `scx-cli` has optional `hdf5` and `cloud` feature flags. HDF5 support is opt-in (`--features hdf5`). Cloud operations are opt-in (`--features cloud`).
