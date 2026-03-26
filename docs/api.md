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

## ScxWriter (`scx-format/src/writer.rs`)

- `new(path, header)` — Create writer (writes to temp file)
- `write_obs(batch)`/`write_var(batch)` — Arrow IPC metadata
- `write_csr_shard(indptr, indices, values, ...)` — CSR expression data
- `write_layer_csr_shard(name, ...)` — Named layer CSR data
- `write_obsp_shard(name, ...)` — Cell-cell graph CSR data
- `write_obsm(name, batch)` — Embeddings
- `write_uns(json)` — JSON metadata
- `write_provenance(operations)` — Operation history
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

## Python API (`pyscx`)

### Module-level functions

- `pyscx.open(path) -> PyExperiment` — Open SCX file (local)
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None)` — Write AnnData to SCX
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None)` — 10x HDF5 to SCX
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

### Cloud operations (`--features cloud`)
- `scx cloud-optimize <input> [--output <path>]`
- `scx explode <input> <output>`
- `scx pack <input> <output>`
- `scx pull <source-url> <dest> [--parallelism N] [--no-cloud-ready] [--filter <expr>]`
- `scx push <source> <dest-url> [--parallelism N]`

Note: `scx-cli` has optional `hdf5` and `cloud` feature flags. HDF5 support is opt-in (`--features hdf5`). Cloud operations are opt-in (`--features cloud`).
