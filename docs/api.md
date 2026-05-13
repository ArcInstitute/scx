# SCX API Reference

## Section Types

20 section types are defined in `scx-format/src/section.rs`:

```
ObsMetadata (0)        — Arrow IPC metadata for observations
ObsIndex (1)           — Arrow IPC index for observations
VarMetadata (2)        — Arrow IPC metadata for variables
VarIndex (3)           — Arrow IPC index for variables
CsrShard (4)           — Main expression matrix (row-major)
CscShard (5)           — Column-major (gene-major) sparse shard. Used
                         for column-axis analytical workloads (DE, HVG,
                         per-gene QC). Optional sidecar; written via
                         `scx convert --csc=always` or `scx build-csc`.
BitmapShard (6)        — Reserved; not produced by the current writer.
                         Detection-presence bitmap was deferred to a
                         future release.
LayerCsrShard (7)      — Alternative expression layers
ObsmEmbedding (8)      — Embeddings (obsm)
ObspCsrShard (9)       — Reserved (legacy); current obsp persistence uses
                         ObspEmbedding (18).
UnsBlob (10)           — Unstructured metadata (JSON)
Provenance (11)        — Operation history
DeletionVectors (12)   — Logical deletion tracking (Roaring Bitmap)
ObsPredicateIndex (13) — Obs predicate index for query pushdown
VarPredicateIndex (14) — Var predicate index for query pushdown
ModalityTable (15)     — v2; ordered list of named modalities (CITE-seq,
                         10x Multiome, …). See docs/format.md § 13.
LayerCscShard (16)     — v2; per-modality CSC sidecar for a named layer
                         (parallel to LayerCsrShard).
VarmEmbedding (17)     — Dense var embeddings (varm); same wire format
                         as ObsmEmbedding but indexed by var.
ObspEmbedding (18)     — Sparse obs×obs pairwise matrices (e.g.
                         kNN connectivities/distances). COO Arrow IPC
                         with schema metadata n_rows / n_cols; data is
                         stored as float32.
VarpEmbedding (19)     — Sparse var×var pairwise matrices. Same wire
                         format as ObspEmbedding.
```

## ScxReader (`scx-format/src/reader.rs`)

- `open(path)` — Open and validate file (mmap-based, validates magic/version/minimum size)
- `header()`, `root_catalog()`, `catalog()` — Access file metadata
- `n_obs()`, `n_vars()`, `nnz()` — Quick dimension access
- `read_obs()`/`read_var()` — Arrow RecordBatch metadata
- `read_csr_shard(idx)` — Single shard as `(Vec<i64>, Vec<i32>, Vec<f32>)`
- `read_all_csr_shards()` — Full matrix as `ScxCsr` (parallel via rayon)
- `read_csc_shard(idx)` — Single CSC sidecar shard as `ScxCsc`
- `read_all_csc_shards()` — Concatenated CSC matrix as `ScxCsc`
- `read_csc_columns(col_range)` — CSC columns covering a half-open
  `Range<u32>`; only shards overlapping the range are decoded
- `read_csc_columns_subset(cols)` — CSC columns gathered from a sorted
  unique `&[u32]`; contiguous runs are read in a single decode
- `csc_shard_count()` — Number of CSC sidecar shards (0 when
  `header.has_csc()` is false)
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
- `write_csc_shard(indptr, indices, values, codec_id, value_encoding, col_start)`
  — Write a CSC sidecar shard. Fully supported. The catalog
  `section_type` is `CscShard (5)`; the on-disk shard header carries
  `shard_type = 1` going forward (readers also accept the legacy
  `shard_type = 0` when the catalog `section_type` is `CscShard`).
  `finish()` updates `header.n_csc_shards` from the writer's
  internal counter and sets the `has_csc` flag bit.
- `write_obs_predicate_index(data)` / `write_var_predicate_index(data)` — Predicate indexes for pushdown
- `write_deletion_vectors(dv)` — Roaring Bitmap deletion vectors
- `write_raw_shard(raw_bytes, section_type, name, stats, nnz)` — Pre-encoded shard passthrough
- `set_shard_column_stats(column_stats)` — Per-column shard statistics for catalog
- `finish()` — Atomic write: full catalog at EOF -> pwrite root catalog -> pwrite header -> fsync -> rename

## Multimodal API

v2 SCX files carry multiple modalities (RNA + ADT + ATAC + …) routed
via a 1-byte `modality_id` stamped on each catalog entry. See
[docs/format.md § 13](format.md#13-multimodal-extension) for the
on-disk layout and [docs/multimodal.md](multimodal.md) for the
end-to-end usage guide.

### `ModalityType` enum (`scx-format/src/modality.rs`)

```
ModalityType::Rna           = 0
ModalityType::Protein       = 1   // ADT
ModalityType::Atac          = 2
ModalityType::Spatial       = 3
ModalityType::Methylation   = 4
ModalityType::Custom        = 255
```

Drives per-modality codec selection (see
[docs/codec.md § Per-modality codec defaults](codec.md#8a-per-modality-codec-defaults)).

### `ModalityInfo` struct

Per-modality record carried on disk by the `ModalityTable` section:

```rust
pub struct ModalityInfo {
    pub name: String,                  // UTF-8, ≤ 64 bytes, unique
    pub modality_type: ModalityType,
    pub default_codec_id: u8,          // CodecId u8 repr
    pub default_value_encoding: u8,    // ValueEncoding u8 repr
    pub n_vars: u64,                   // Per-modality variable count
    pub nnz: u64,                      // Per-modality non-zeros
    pub n_csr_shards: u32,
    pub n_csc_shards: u32,
    pub flags: ModalityFlags,          // HAS_CSC, HAS_OBSM, HAS_OBSP, HAS_LAYERS, HAS_UNS
}
```

### `ScxReader` — multimodal accessors

- `is_multimodal()` / `n_modalities()` / `has_modalities()` — capability checks.
- `modality_table()` → `Option<&ModalityTable>` — full parsed table.
- `modality_names()` → `Vec<&str>` — names in registration order.
- `modality_id(name)` → `Option<u8>` — name → id resolution.
- `modality_info(id)` → `Option<&ModalityInfo>` — per-modality record.
- `read_var_for(modality_id)` — per-modality var as `RecordBatch`.
- `read_csr_shard_for(modality_id, shard_idx)` — single shard from
  the modality's CSR shard list.
- `read_all_csr_shards_for(modality_id)` — concatenated CSR for the
  modality. Patches `n_cols` from `modality_info(id).n_vars` so the
  resulting `ScxCsr` has the modality's per-modality `n_vars` rather
  than the file-wide `header.n_vars` (which is the max across modalities).
- `csr_shard_count_for(modality_id)` — CSR shard count.
- `read_all_csc_shards_for(modality_id)` / `csc_shard_count_for(modality_id)`
  — same shape on the column-major axis.
- `read_obsm_for(modality_id, key)` — per-modality embeddings.

### `ScxWriter` — multimodal writers

- `add_modality(name, modality_type, default_codec, default_value_encoding)` →
  `Result<u8>` — register a modality (1-based id). Validates the name
  (≤ 64 bytes, UTF-8, unique).
- `set_modality_n_vars(modality_id, n_vars)` — record the per-modality
  variable count (called once per modality, before the first per-modality
  shard write).
- `modality_id(name)` → `Option<u8>` / `modality_name_for(id)` — lookups.
- `write_var_for(modality_id, batch)` — per-modality var.
- `write_csr_shard_for(modality_id, indptr, indices, values, codec_id, encoding, row_start)`
  — per-modality CSR shard. Uses the `X/{name}/shard_{i}` naming convention.
  Updates the modality's `n_csr_shards` / `nnz` automatically.
- `write_csc_shard_for(modality_id, indptr, indices, values, codec_id, encoding, col_start)`
  — per-modality CSC shard (`X_csc/{name}/shard_{i}`). Section type is
  `CscShard` (5).
- `write_layer_for(modality_id, layer_name, ...)` /
  `write_obsm_for(modality_id, key, batch)` /
  `write_obsp_for(modality_id, key, ...)` /
  `write_uns_for(modality_id, json)` — per-modality variants of the
  legacy section writers. Stamp the catalog entry with the chosen
  `modality_id`.
- `finish()` — emits the `ModalityTable` section automatically when
  any modality was registered. Updates header `n_modalities` /
  `modality_table_offset` / `modality_table_length` and sets the
  `has_modalities` flag bit.

### Codec selection — `select_codec_for_modality`

`select_codec_for_modality(raw_values, value_encoding, modality_type)`
extends `select_codec` with per-modality routing (Phase E):
RNA / Custom / Methylation / Spatial → delegate to `select_codec`;
Protein/ADT → Zstd for integers, Pcodec for floats; ATAC → Zstd for
binary peak presence (sample max ≤ 1) else Lz4Shuffle, Pcodec for
floats. See [docs/codec.md § Per-modality codec defaults](codec.md#8a-per-modality-codec-defaults).

### Per-modality `BackedCscReader`

`BackedCscReader::for_modality(reader, modality_id, cache_shards)` —
builds a column-major reader scoped to one modality so multimodal
training (e.g. totalVI) can hold separate caches per modality without
LRU thrashing across modalities.

### Python (`pyscx`)

- `pyscx.from_mudata(mu, path, codec="auto", ...)` — write a MuData
  object as a multimodal SCX file. Per-modality CSR shards stamped
  with `modality_id` derived from the registration order; per-modality
  codec resolved via `select_codec_for_modality`.
- `pyscx.open(path)` returns `PyExperiment`. New attrs / methods:
  - `is_multimodal: bool`, `n_modalities: int`, `modality_names: list[str]`.
  - `modality_id(name) -> int | None`, `modality_info(id) -> dict | None`.
  - `to_mudata() -> mudata.MuData` — round-trips back to MuData.
- `pyscx.MultimodalTrainingDataset(path, modalities=[…], …)` — yields
  per-batch dicts `{"X": {modality_name: ndarray}, "obs": {...},
  "cell_indices": ndarray}` (or tuples in `return_dict=False` mode).
- Backward compat: `pyscx.TrainingDataset(path)` on a multimodal file
  emits `UserWarning` and falls back to the alphabetically-first
  modality. Pass `modality="rna"` explicitly to suppress.

### R (`rscx`)

- `from_seurat(seu, path)` detects Seurat v5 multi-assay objects
  (`length(seu@assays) > 1`) and routes to a multimodal write path
  with one modality per assay. Single-assay objects keep the legacy
  path.
- `from_mae(mae, path)` writes a Bioconductor `MultiAssayExperiment`
  as multimodal SCX. Cells must align across experiments
  (`colnames(experiments[[i]])` identical); on misalignment, raises
  with a clear error directing to `intersectColumns(mae)`.
- `scx_open(path)$to_seurat()` builds a Seurat v5 multi-assay object
  on multimodal files (one `Assay5` per modality, shared `meta.data`).
- `scx_open(path)$to_mae()` builds a `MultiAssayExperiment` (one
  `SingleCellExperiment` per modality, shared `colData`).
- `scx_open(path)$is_multimodal()` / `$modality_names()` — capability
  checks.

### CLI surface

```
scx info path.scx               # Modalities (N): name, type, n_vars, nnz, csr/csc, codec
scx validate path.scx           # ModalityTable checksum + n_modalities cross-check
scx convert --from h5mu in.h5mu --to scx out.scx
scx convert --to h5ad out.scx out.h5ad --modality rna
scx append target.scx --input new.scx --modality rna
scx subset in.scx --modality rna --output rna_only.scx
```

## Codec Selection (`scx-format/src/codec_select.rs`)

- `select_codec(values, encoding)` — Auto-select best codec per shard based on data type and distribution:
  - Integer values with median ≤ 8 → Scx1 (Rice coding, optimal for typical 10x UMI counts)
  - Integer values with median > 8 → Zstd (LZ77 dictionary wins for larger values)
  - Float values (Float32, Float16) → Pcodec (7–16% better compression than Zstd on log-normalized data)
- LZ4+shuffle (`codec="lz4"`) and Pcodec (`codec="pcodec"`) also available as explicit overrides

**Codec tradeoffs:**

| Codec | Best for | Compression | Read speed | Write speed |
|-------|----------|-------------|------------|-------------|
| `auto` | General use (recommended default) | Best per-shard | Best per-shard | Best per-shard |
| `scx1` | Small UMI counts (median ≤ 8) | Best for 10x data (~4.8×) | Fastest (SIMD decode) | Moderate |
| `zstd` | Large integers, general fallback | Good (~4.3× UMI, ~3.8× float) | Fast | Fast |
| `pcodec` | Log-normalized, PCA embeddings, float layers | Best for floats (~4.1–4.7×) | Moderate (19–39% slower than Zstd) | Slower (35–40% slower than Zstd) |
| `lz4` | Speed-critical pipelines | Lower (~2.2–3.1×) | Fast | Fastest compressed |
| `none` | GDS bypass, debugging | 1× (no compression) | Fastest (I/O bound) | Fastest |

For raw count data (integer-valued), `auto` selects Scx1 or Zstd — Pcodec falls through to Zstd internally since its advantage is specific to float values. For storage-constrained workflows with normalized float data, explicitly selecting `pcodec` gives the best compression. For latency-sensitive pipelines, `zstd` or `lz4` are better choices.

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

## BackedCscReader (`scx-format/src/backed.rs`)

Column-major counterpart to `BackedCsrReader`. Streams CSC sidecar
shards from disk with an LRU shard cache, parallel to the CSR side.

- `new(reader, cache_shards)` — Create from `ScxReader`. `cache_shards = 0` disables caching (one decode per call)
- `index()` — Per-shard column-range index, sorted by `col_start`
- `n_shards()` / `n_obs()` / `n_vars()` — Dimensions
- `read_shard_uncached(idx)` → `ScxCsc` — Single CSC shard, bypass cache
- `read_shard_cached(idx)` → `Arc<ScxCsc>` — Single CSC shard, through cache
- `read_csc_columns(col_range)` → `ScxCsc` — Decode only shards overlapping the half-open range; partial-overlap shards are sliced post-decode
- `read_csc_columns_subset(cols)` → `ScxCsc` — Gather columns from a sorted unique `&[u32]`; contiguous runs share a single decode
- `enable_metrics()` / `metrics()` — Per-call hits / misses / decoded-bytes counters

## ColumnShardSource Trait (`scx-format/src/shard_source.rs`)

Sibling to `ShardSource` for column-major streaming. Defined in
`scx-format` so consumers in `scx-accel` and `pyscx` can take the
bound generically without depending on each other.

```rust
pub trait ColumnShardSource {
    fn n_csc_shards(&self) -> usize;
    fn n_obs(&self) -> usize;
    fn n_vars(&self) -> usize;
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }
    fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc>;
    fn read_csc_columns(&self, col_range: Range<u32>) -> Result<ScxCsc>;
    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)>;
}
```

**Implementations:**

| Type | Crate | Behavior |
|------|-------|---------|
| `BackedCscReader` | `scx-format` | Reads raw decoded CSC shards from disk (cache + index-driven shard skip) |
| `LazyShardSource` | `pyscx` (internal) | Applies the column-local subset of transforms (Log1p only) post-decode; honors column projection |

**Capability gate:** the trait is *not* a sub-trait of `ShardSource`.
Consumers that want CSC dispatch take the bound explicitly (`fn
require_csc<S: ColumnShardSource>(...)`); the runtime "does this
dataset support CSC?" question is answered exactly once at
`ScxBackedSparseDataset::as_column_source()` /
`ScxLazyTransformedDataset::as_column_source()`. Returns `Some` iff
the file has a CSC sidecar AND the transform chain is column-local
AND no row deletion vector is active. See `pyscx.accel.*
prefer_format` below.

**Used by:** `scx-accel::csc::{streaming_mean_var_csc,
streaming_clip_square_sum_csc, wilcoxon_rank_sum_streaming_csc,
pseudobulk_aggregate_csc}` and the `pyscx::projected_agg::*_csc`
column-aggregation kernels.

## scx-ops — File Operations

### `scx_ops::append(path, obs, indptr, indices, values, value_encoding, options: &AppendOptions) → Result<()>`
Append new cells at EOF from raw CSR arrays. `AppendOptions` bundles codec selection, shard sizing, and modality routing. Advisory flock for concurrent safety.

### `scx_ops::append_from_reader(target_path, source_reader, options: &AppendOptions, source_modality_id) → Result<()>`
Streaming SCX → SCX append. Reads one source CSR shard at a time (or copies raw bytes verbatim when codec / value encoding / index dtype / per-modality `n_vars` all match), avoiding materializing the entire source matrix in memory. Supports multimodal targets via `options.modality_id` / `source_modality_id` routing. CSC sidecars are dropped on append.

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

> [!NOTE]
> For the most up-to-date function signatures and type annotations, see the
> [auto-generated Python API reference](python_api.rst) built from the Rust
> docstrings via autodoc.

### Module-level functions

- `pyscx.open(path) -> PyExperiment` — Open SCX file (local)
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None)` — Write AnnData to SCX.
  Persists `X`, `obs`, `var`, `layers`, `obsm`, `varm`, `uns`, and the sparse
  pairwise slots `obsp` / `varp`. Pairwise matrices are stored as float32 COO
  Arrow IPC; higher-precision inputs are downcast on write.
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None)` — 10x HDF5 to SCX
- `pyscx.from_mtx(mtx_dir, scx_path, codec=None, shard_size=None)` — Cell Ranger MTX directory (`matrix.mtx[.gz]`, `barcodes.tsv[.gz]`, `features.tsv[.gz]`) to SCX. Default shard size is 16384.
- `pyscx.to_mtx(scx_path, output_dir)` — SCX to Cell Ranger–style MTX directory (`matrix.mtx.gz`, `barcodes.tsv.gz`, `features.tsv.gz`).
- `pyscx.iter_chunks(adata, chunk_size="shard")` — Shard-aligned or fixed-size chunk iterator
- `pyscx.preprocess(source, target, ops, target_sum=None)` — Streaming shard-by-shard preprocessing
- `pyscx.save_layer(source, target, layer_name, ops, target_sum=None)` — Save transformed data as layer

### File operations
- `pyscx.append(target, input, codec=None, shard_size=None)` — Streaming append from SCX file (reads one shard at a time; raw-copy fast path when codec/encoding match)
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
  - Returns `obsm` (dense), `varm` (dense), `obsp` (scipy CSR), and
    `varp` (scipy CSR) when present in the file. `obsp` / `varp` are
    not subject to deletion-vector row filtering — when cells are
    logically deleted, the pairwise matrices still cover the full
    original axis; `compact` resolves this by rebuilding from scratch.
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

#### `prefer_format="csr"|"csc"` kwarg

Several accelerators take an explicit `prefer_format` kwarg that selects
between the row-major CSR path (default) and the column-major CSC sidecar
path. See [scanpy.md § prefer_format](scanpy.md#prefer_formatcsrcsc-explicit-column-major-dispatch)
for the full dispatch rules and requirements.

- `pyscx.accel.pca(adata, n_comps=50, zero_center=True, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto")` — Randomized SVD PCA with streaming SpMM. Writes `obsm["X_pca"]`, `varm["PCs"]`, `uns["pca"]`. On GPU: cuSPARSE SpMM + cuSOLVER QR (f32).
- `pyscx.accel.neighbors(adata, n_neighbors=15, use_rep="X_pca", random_state=0, ef_construction=200, ef_search=200, device="auto")` — kNN graph + UMAP-style connectivities. CPU: HNSW. GPU: CAGRA (cuVS). Writes `obsp["distances"]`, `obsp["connectivities"]`, `uns["neighbors"]`.
- `pyscx.accel.umap(adata, n_components=2, n_epochs=200, min_dist=0.1, spread=1.0, negative_sample_rate=5, learning_rate=1.0, random_state=0, device="auto")` — Spectral-init SGD UMAP. GPU: native CUDA kernel or cuML fallback. Writes `obsm["X_umap"]`.
- `pyscx.accel.rank_genes_groups(adata, groupby, reference="rest", n_genes=None, method="wilcoxon", gene_chunk_size=None, log_transformed=False, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr")` — Parallel Wilcoxon rank-sum with BH correction. Writes `uns["rank_genes_groups"]`, or returns DataFrame when `stratify_by` is set. `prefer_format="csc"` routes per-chunk reads through the column-major sidecar (see kwarg docs above).
- `pyscx.accel.pseudobulk_dex(adata, groupby, test_col, reference, design=None, aggr_method="sum", min_cells_per_group=10, stratify_by=None, min_cells_per_stratum=50, prefer_format="csr", gene_indices=None)` — Streaming pseudobulk aggregation (Rust) + pydeseq2 testing. Returns DataFrame. Requires optional `pydeseq2` dependency. `prefer_format="csc"` requires a gene subset (either an explicit `gene_indices` argument or a `col_projection` already set on `adata.X`); full-gene CSC pseudobulk has no measurable speed-up.
- `pyscx.accel.rank_genes_groups_df(adata, groupby, reference="rest", n_genes=None, gene_chunk_size=None, rankby_abs=False, tie_correct=False) → polars.DataFrame` — Same Wilcoxon as `rank_genes_groups()` but returns a polars DataFrame in cell-eval's `DEResults` schema: `(target, feature, fold_change, p_value, fdr, log2_fold_change, abs_log2_fold_change)`. Ready to feed into `cell_eval.initialize_de_comparison()`.
- `pyscx.accel.pseudobulk_means(adata, groupby, min_cells_per_group=1) → (ndarray, list[str])` — Group-by mean on sparse X, streaming shard-by-shard (works on backed, lazy, scipy CSR, or dense). Returns `(means[P, G] float64, sorted group names)`. Foundation for the perturbation evaluation metrics below.
- `pyscx.accel.perturbation_metrics(adata_real, adata_pred, pert_col="perturbation", control="control", metrics=None, min_cells_per_group=1) → dict[str, dict[str, float]]` — Bundled bulk metrics `{pearson_delta, mse, mae, mse_delta, mae_delta}` between paired real/pred AnnData. Matches cell-eval's metrics within atol=1e-6.
- `pyscx.accel.energy_distance(adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None) → float` — Pearson correlation of per-perturbation e-distance vectors (real vs pred). Avoids `[N, N]` distance materialization (per-row sum reduction even on the gemm path); precomputes control self-distance once; rayon-parallel across perturbations. `backend ∈ {"auto" (default), "gemm", "scalar"}` — `"auto"` picks faer-dispatched gemm for euclidean/cosine and the scalar row-by-row path for L1; `"gemm" + metric="l1"` raises `RuntimeError` (no decomposition exists). `dtype ∈ {"f32" (default), "f64"}` controls only the matmul / per-pair arithmetic precision; reductions always accumulate in `f64`. f32 + gemm matches f64 + scalar within `atol=1e-4` correlation / `atol=1e-3` per-pert.
- `pyscx.accel.energy_distance_details(...)` — Same signature (including `backend` / `dtype`) as `energy_distance` but returns `{"correlation": float, "d_real": {pert: float}, "d_pred": {pert: float}, "pert_names": [...]}`.
- `pyscx.accel.discrimination_score(adata_real, adata_pred, pert_col="perturbation", control="control", metric="l1", exclude_target_gene=True, embed_key=None, min_cells_per_group=1) → dict[str, float]` — Per-perturbation normalized rank of the predicted perturbation effect's distance to the correct real effect. `metric ∈ {"l1", "l2"/"euclidean", "cosine"}`. `exclude_target_gene=True` drops the gene matching each perturbation's name from the distance (matches cell-eval's default).
- `pyscx.accel.knockdown_efficiency(adata, pert_col="perturbation", control="control", eps=1e-8)` — Per-cell knockdown efficiency + log-fold change vs control baseline. Input must be normalized (NOT log1p'd); log1p is applied internally. Writes `adata.obs["KnockDownEfficiency"]` and `adata.obs["KnockDownGeneFC"]` (both float32, NaN for control cells and cells whose perturbation name isn't in `var_names`). Matches `arc_bench.tools.normalize_transform.core` within atol=1e-6.
- `pyscx.accel.clustering_agreement(adata_real, adata_pred, pert_col="perturbation", control="control", metric="ami", real_resolution=1.0, pred_resolutions=None, n_neighbors=15, embed_key=None, min_cells_per_group=1) → float` — Builds perturbation-centroid kNN graphs, sweeps Leiden resolutions, scores best real-vs-pred agreement via AMI / NMI / ARI. **All-native-Rust** post-Phase-3 — no scanpy / anndata / igraph dispatch; uses `scx_accel::neighbors::build_knn_graph` (HNSW via `instant-distance`, `ef_construction=200, ef_search=50, seed=0`) plus `scx_accel::leiden` sequential mode (`max_iterations=2, parallel=false, seed=0` — matches scanpy's `flavor="igraph", n_iterations=2`). Pred-side kNN graph built once and reused across the resolution sweep; whole hot path runs under `py.allow_threads`. Matches cell-eval's `ClusteringAgreement` within `atol=0.15` aggregate (stochastic Leiden; exact score match not expected — algorithms agree exactly on graphs with `n_perts ≥ 16`).
- `pyscx.accel.adjusted_mutual_info(labels_a, labels_b) → float` — AMI on integer label arrays (arithmetic-mean convention). Matches `sklearn.metrics.adjusted_mutual_info_score` within atol=1e-10.
- `pyscx.accel.normalized_mutual_info(labels_a, labels_b) → float` — NMI (arithmetic-mean). Matches `sklearn.metrics.normalized_mutual_info_score` within atol=1e-10.
- `pyscx.accel.adjusted_rand_index(labels_a, labels_b) → float` — ARI rescaled to `[0, 1]` via `(ARI + 1) / 2` (cell-eval convention). For the raw sklearn ARI (in `[-0.5, 1]`), compute `2 * adjusted_rand_index(a, b) - 1`.
- `pyscx.accel.leiden(adata, resolution=1.0, key_added="leiden", random_state=0, n_iterations=-1, device="auto")` — Leiden community detection on kNN graph. Reads `obsp["connectivities"]` (from `neighbors()`). GPU: cuGraph Leiden (up to 47× faster). CPU fallback: leidenalg via igraph. Writes `adata.obs[key_added]` (categorical) and `adata.uns["leiden"]` (params + backend metadata). GPU and CPU may produce different partitions due to algorithmic differences; compare via ARI/NMI.
- `pyscx.accel.harmony_integrate(adata, key, *, basis="X_pca", adjusted_basis=None, n_clusters=None, theta=None, sigma=0.1, lamb=None, alpha=0.2, max_iter=10, max_iter_kmeans=4, epsilon_harmony=1e-2, epsilon_kmeans=1e-3, block_size=0.05, batch_prop_cutoff=1e-5, tau=0.0, random_state=0, device="auto")` — Clean-room Rust implementation of Harmony2 (Korsunsky et al., 2019). Soft k-means clustering with a diversity penalty, followed by ridge-regression correction of `adata.obsm[basis]` (`"X_pca"` by default). `key` accepts a single `obs` column name or a list for multi-covariate batch correction; each is factorised via `pandas.factorize(sort=False)`. Writes corrected embedding (f32, N × d) to `adata.obsm[adjusted_basis or basis]` and convergence metadata to `adata.uns["harmony"]` (`params`, `converged`, `n_iterations`, `objective_harmony`, `backend`). Parameter names match `scanpy.external.pp.harmony_integrate`, so existing scanpy pipelines can swap in. GPU path (when built with `--features gpu`) accelerates distance / L2-norm / batched scatter-subtract kernels; k-means++ init and covariance inversion stay on CPU. Numerical parity vs R `harmony` v2.x: mean per-PC Pearson r 0.989–0.999 on the three validation fixtures.
- `pyscx.accel.compute_lisi(adata, key, *, basis="X_pca", perplexity=30.0, n_neighbors=None) → np.ndarray` — Local Inverse Simpson Index on an `obsm` embedding. Exact brute-force kNN (matches R `FNN::get.knn`) + per-cell Gaussian-bandwidth search (t-SNE Hbeta routine) + Simpson index over kernel-weighted neighbour category probabilities. Returns LISI vector of length N and also writes to `adata.obs[f"lisi_{key}"]`. Values near 1 → poor mixing (neighbourhoods dominated by one category); values approaching the number of categories → uniform mixing. `n_neighbors` defaults to `ceil(3 × perplexity)`. ~10× faster than R `lisi::compute_lisi` on D1–D4 with mean-LISI agreement within 0.8–2.4 %.
- `pyscx.accel.normalize_total(adata, target_sum=10000.0)` — Materialization-free row normalization. On `ScxBackedSparseDataset`: computes row sums via streaming, creates `ScxLazyTransformedDataset` wrapper. On `ScxLazyTransformedDataset`: appends `NormalizeTotal` transform to chain. On scipy CSR: delegates to `sc.pp.normalize_total()`.
- `pyscx.accel.log1p(adata)` — Materialization-free log1p. On `ScxBackedSparseDataset`: creates `ScxLazyTransformedDataset` with `Log1p` transform. On `ScxLazyTransformedDataset`: appends `Log1p` to chain (fuses with preceding `NormalizeTotal` when possible). On scipy CSR: delegates to `sc.pp.log1p()`.
- `pyscx.accel.gpu_info() → dict` — Query GPU device info: `{'device': ..., 'total_vram_gb': ..., 'free_vram_gb': ...}`. Returns `None` if no GPU available.
- `pyscx.accel.estimate_gpu_memory(adata, operation, **kwargs) → dict` — Estimate GPU VRAM required for an operation. Returns `{'required_gb': float, 'fits_in_vram': bool}`. Supported operations:
  - `"pca"`: kwargs `n_components` (default 50), `n_oversamples` (default 10), `shard_size` (default 16384)
  - `"knn"`: kwargs `n_neighbors` (default 15), `n_dims` (default 50)
  - `"umap"`: kwargs `n_components` (default 2)
  - `"leiden"`: no additional kwargs

  Raises `ValueError` for unknown operations. Note: estimates are approximate — cuSOLVER QR workspace may be undercounted by ~1.5×.
- `pyscx.accel.calculate_qc_metrics(adata, qc_vars=None, log1p=True, inplace=True, prefer_format="csr")` — Streaming QC metrics for backed/lazy data without materialization. Computes per-cell `n_genes_by_counts`, `total_counts` and per-gene `n_cells_by_counts`, `total_counts`. Supports `qc_vars` for gene subsets (e.g., `["mt"]` for mitochondrial percentage). When `inplace=True`, writes to `adata.obs`/`adata.var`; when `False`, returns `(obs_df, var_df)`. `prefer_format="csc"` routes the gene-axis aggregation through the CSC sidecar (cell-axis stays CSR — row aggregations have no CSC win). Falls back to `sc.pp.calculate_qc_metrics()` for scipy/dense.
- `pyscx.accel.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", batch_key=None, span=0.3, subset=False, n_bins=20, device="auto", prefer_format="csr")` — Streaming HVG selection. Default CPU + CSR streams `mean_var` and clipped sums shard-by-shard via `ShardSource`; multi-batch runs CSR. `prefer_format="csc"` routes single-batch seurat_v3 through `streaming_mean_var_csc` and `streaming_clip_square_sum_csc` (multi-batch + GPU + non-seurat_v3 flavors raise on CSC). Writes `var["highly_variable"]`, `var["means"]`, `var["variances"]`, `var["variances_norm"]`, `var["highly_variable_rank"]`.
- `pyscx.accel.col_sums(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column sums on `ScxBackedSparseDataset`. Honors `col_projection` and `kept_to_global` on the CSR path; CSC dispatch requires no row deletion vector and (currently) only supports `ScxBackedSparseDataset` and `ScxLazyTransformedDataset` (CSR scipy / dense raises a helpful message — use the array-protocol `dataset.sum(axis=0)` for those).
- `pyscx.accel.col_nnz(dataset, prefer_format="csr") → np.ndarray (i64)` — Streaming per-column NNZ. Same dispatch as `col_sums`.
- `pyscx.accel.col_min(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column min. Implicit-zero correction (`mins[c] = min(mins[c], 0.0)` when `col_nnz[c] < n_obs`) applied on both paths.
- `pyscx.accel.col_max(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column max. Symmetric implicit-zero correction.
- `pyscx.accel.col_var(dataset, prefer_format="csr") → np.ndarray (f64)` — Streaming per-column variance. CSR uses a two-pass formulation; CSC uses single-pass `(sum_x² - n·mean²) / n`. Numerically equivalent within f64 epsilon (verified by `pyscx/tests/test_csc_dispatch.py`).
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

### IndexPlanDataset

Plan-driven paired-batch reader for ML workloads where each batch is a list
of `(perturbed_cell, control_cell)` index pairs (perturbation training,
contrastive learning, donor-matched designs). Sibling to `TrainingDataset`:
`TrainingDataset` streams shards in catalog order for the highest possible
sequential throughput; `IndexPlanDataset` consumes a Python iterator of
plans and yields paired dense batches, trading sequential streaming for
per-cell pairing flexibility.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p`). |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target sum. |
| `cache_shards` | `128` | LRU shard cache budget. Auto-tuned downward to fit `max_memory_mb`; check via `effective_cache_shards()`. |
| `sort_by_shard` | `True` | Reorder each plan by `min(shard_of(p), shard_of(c))` so the returned `X`/`X_paired` rows land in shard locality order. Disable to preserve caller's input pair order. |
| `lookahead` | `4` | Default lookahead for `iter_with_plans` when not overridden. `0` disables shard prefetching; auto-tuned downward to fit `max_memory_mb`; check via `effective_lookahead()`. |
| `max_plan_size` | `16384` | Upper bound on rows-per-batch for the memory budget calculation. |
| `max_memory_mb` | `512` | On overflow, `lookahead` is reduced first (down to 1), then `cache_shards` (down to 1); construction fails with `RuntimeError` if neither fits. |

**Batch dict schema** (yielded by `iter_with_plans`):

```python
{
    "X":          np.ndarray[B, n_output_genes, float32],   # perturbed rows
    "X_paired":   np.ndarray[B, n_output_genes, float32],   # control rows
    "pairs":      list[tuple[int, int]],                    # post-sort plan
    "obs":        dict[str, np.ndarray | {"codes", "categories"}],
    "obs_paired": dict[str, np.ndarray | {"codes", "categories"}],
}
```

`pairs[i]` always corresponds to `X[i]` and `X_paired[i]`. Categorical obs
columns encode as `{"codes": ndarray[i32], "categories": list[str]}` —
schema matches `TrainingDataset`.

**Iterator semantics**

- `plans` is any Python iterable yielding `list[tuple[int, int]]` (plain lists,
  generators, queues all work).
- Plan iteration is lazy: the loader pulls the next plan only when it is
  ready to schedule a prefetch for it.
- The loader keeps `lookahead` plans in flight at once: the head plan is
  decoding while shards for the next `lookahead - 1` are being warmed via
  `tokio::task::spawn_blocking` calls into `BackedCsrReader::read_shard_cached_arc`.
- `StopIteration` from `plans` ends the batch stream cleanly. Other Python
  exceptions from `plans` propagate as `RuntimeError("plan iterator raised: ...")`.
- Empty plans inside a stream are silently skipped.
- Out-of-range row indices raise `IndexError` immediately when validating
  the offending plan; missing obs columns raise `KeyError` at construction.
- `os.fork()` after construction raises `RuntimeError` with `num_workers=0`
  guidance — the shard cache and mmap state are not fork-safe; consumers
  must lazily construct the dataset post-fork in each DataLoader worker.

**Example — bare iterator**

```python
import pyscx

ds = pyscx.IndexPlanDataset(
    "atlas.scx",
    hvg_indices=hvg_array,
    obs_columns=["cell_type", "perturbation"],
    normalize=True,
    target_sum=1e4,
)

plans = [
    [(0, 5), (2, 7)],
    [(10, 100), (50, 75)],
]
for batch in ds.iter_with_plans(iter(plans)):
    X, X_paired = batch["X"], batch["X_paired"]
    pairs = batch["pairs"]
    # ... training step ...
```

**Example — paired with a `BaseMappingStrategy`-style plan generator** (the
cell-load-scx pattern):

```python
def plan_generator(strategy, perturbed_indices, batch_size):
    """Wrap any pairing policy that exposes `get_control_index(idx) -> int`."""
    buf = []
    for p_idx in perturbed_indices:
        c_idx = strategy.get_control_index(p_idx)
        if c_idx is not None:
            buf.append((p_idx, c_idx))
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

ds = pyscx.IndexPlanDataset("atlas.scx", obs_columns=["cell_type"])
for batch in ds.iter_with_plans(plan_generator(my_strategy, perm, 1024),
                                lookahead=4):
    ...
```

**Memory budget surfaces**

```python
ds = pyscx.IndexPlanDataset("atlas.scx", cache_shards=128, lookahead=4,
                            max_plan_size=16384, max_memory_mb=256)
print(ds.effective_cache_shards(), ds.effective_lookahead())
# Detects when auto-tuning kicked in.
```

## CLI (`scx-cli`)

### Core
- `scx convert <input> <output> [--from h5ad|10x] [--to h5ad] [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N]`
- `scx info <file> [--json] [--history]`
- `scx validate <file> [--verbose]`
- `scx benchmark <file> [--compare-h5ad <path>] [--runs N] [--json]`

### File operations
- `scx append <target> --input <source> [--codec auto|none|scx1|zstd|lz4|pcodec] [--shard-size N]` — Streaming append (reads source one shard at a time)
- `scx delete <file> --filter <expr> [--dry-run]`
- `scx compact <input> --output <path> [--force]`
- `scx rollback <file> [--to-seq N]`
- `scx merge <file1> <file2> [<...>] --output <path>`
- `scx query <file> <filter> [--count] [--output <path>] [--select-genes <path>] [--normalize N] [--log1p] [--limit N] [--json]`
- `scx subset <input> [--output <path>] [--filter <expr>] [--genes <path>] [--dry-run] [--shard-size N] [--codec auto|none|scx1|zstd|lz4|pcodec]` — Extract a subset of cells and/or genes into a new SCX file
- `scx build-csc <input> <output> [--memory-limit 4G] [--force]` — Build CSC (column-major) shards from existing CSR data
- `scx upgrade <input> [output] [--in-place]` — Upgrade an SCX file to the latest format version

### Cloud operations (`--features cloud`)
- `scx cloud-optimize <input> [--output <path>]`
- `scx explode <input> <output>`
- `scx pack <input> <output>`
- `scx pull <source-url> <dest> [--parallelism N] [--no-cloud-ready] [--filter <expr>]`
- `scx push <source> <dest-url> [--parallelism N]`

Note: `scx-cli` has optional `hdf5` and `cloud` feature flags. HDF5 support is opt-in (`--features hdf5`). Cloud operations are opt-in (`--features cloud`).
