# SCX API Reference

## Section Types

13 section types are defined in `scx-format/src/section.rs`:

```
ObsMetadata (0)     — Arrow IPC metadata for observations
ObsIndex (1)        — Arrow IPC index for observations
VarMetadata (2)     — Arrow IPC metadata for variables
VarIndex (3)        — Arrow IPC index for variables
CsrShard (4)        — Main expression matrix (row-major)
CscShard (5)        — Gene-major view (opt-in, Phase 3)
BitmapShard (6)     — Detection bitmap (Phase 3)
LayerCsrShard (7)   — Alternative expression layers
ObsmEmbedding (8)   — Embeddings (obsm)
ObspCsrShard (9)    — Cell-cell graphs (obsp)
UnsBlob (10)        — Unstructured metadata (JSON)
Provenance (11)     — Operation history
DeletionVectors (12)— Logical deletion tracking (Phase 2)
```

## ScxReader (`scx-format/src/reader.rs`)

- `open(path)` — Open and validate file (mmap-based, validates magic/version/minimum size)
- `header()`, `root_catalog()`, `catalog()` — Access file metadata
- `n_obs()`, `n_vars()`, `nnz()` — Quick dimension access
- `read_obs()`/`read_var()` — Arrow RecordBatch metadata
- `read_csr_shard(idx)` — Single shard as `(Vec<i64>, Vec<i32>, Vec<f32>)`
- `read_all_csr_shards()` — Full matrix as `ScxCsr`
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

## Provenance

- `ProvenanceEntry`: timestamp, action, tool, params_json, input_checksums
- Auto-populated on `ScxWriter::finish()` with operation info
- `scx info` displays full provenance history
- Read/write via `ScxReader::read_provenance()` / `ScxWriter::write_provenance()`

## Python API (`pyscx`)

### Module-level functions

- `pyscx.open(path) -> PyExperiment` — Open SCX file
- `pyscx.from_anndata(adata, path, codec=None, shard_size=None)` — Write AnnData to SCX
- `pyscx.from_10x(h5_path, scx_path, codec=None, shard_size=None)` — 10x HDF5 to SCX

### PyExperiment

- `to_anndata()` — Convert to AnnData (zero-copy where possible)
- `validate()` — Check checksums, returns list of `(section_name, passed)`
- Properties: `n_obs`, `n_vars`, `nnz`, `shard_count`, `format_version`, `codec_id`, `layer_names`

## CLI (`scx-cli`)

- `scx convert --from <h5ad|10x> --to <path> [--shard-size N]` — Convert h5ad/10x to SCX
- `scx info <path>` — Display file metadata + provenance history
- `scx validate <path> [--verbose]` — Verify all section checksums

Note: `scx-cli` has an optional `hdf5` feature flag (`hdf5 = ["dep:hdf5", "dep:ndarray"]`). HDF5 support is opt-in.
