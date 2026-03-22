# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. Replaces AnnData/h5ad with a unified Rust-native stack.

**Phase 1 is complete.** Phase 2 is complete (Steps 1-5 done, cloud operations implemented). See [ROADMAP.md](ROADMAP.md) for the full phased plan and [Phase2.md](Phase2.md) / [Phase2-CLOUD.md](Phase2-CLOUD.md) for current work.

## Key Documents

- **SPEC.md** — Format specification (v0.5). Authoritative reference for binary layouts, codecs, and section types.
- **ROADMAP.md** — Phased implementation plan (Phases 1-4).
- **Phase1.md** — Phase 1 implementation plan (all tasks complete).
- **Phase2.md** — Phase 2 implementation plan (Steps 1-5 complete).
- **Phase3.md** — Phase 3 specification (GPU, R bindings, multimodal — future work).
- **docs/api.md** — API reference. **docs/testing.md** — Test/benchmark details. **docs/architecture.md** — Crate architecture.
- **tasks/** — Completed specs (e.g., `Phase2-Step4.md`, `Phase2-Step5.md`).

## Build and Test

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check

# With cloud features:
cargo test --workspace --features cloud

# Python bindings (always use uv venv at .venv/):
cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v

# Python bindings with cloud support:
cd pyscx && ../.venv/bin/maturin develop --features cloud && ../.venv/bin/pytest tests/ -v
```

**Always use the uv venv** at `.venv/` for all Python work. Do NOT use system Python or pip directly.

## Architecture

### Crate Dependency Order

```
scx-codec (standalone)
    └─> scx-format (depends on scx-codec)
            ├─> scx-sparse (standalone, used by scx-format)
            ├─> scx-ops (depends on scx-format; append/delete/compact/merge/rollback)
            ├─> scx-engine (depends on scx-format, scx-sparse, scx-ops)
            ├─> scx-loader (depends on scx-format, scx-codec, scx-sparse)
            ├─> scx-cloud (depends on scx-format, scx-codec, scx-engine)
            ├─> scx-cli (depends on all above)
            └─> pyscx (depends on all above)
```

9 workspace members total. `scx-cli` has optional `hdf5` and `cloud` feature flags. `scx-loader` does NOT depend on `scx-engine` — it has its own streaming-optimized gene projection and fused ops. `scx-cloud` does NOT depend on `scx-loader` — they are siblings.

### File Format (SPEC.md §3)

- **File header**: 256 bytes, LE, magic `b"SCX\x01"`. Includes `front_catalog_offset`/`length` (populated by `cloud-optimize`, zero otherwise).
- **Root catalog**: At offset 256, max 4096 bytes.
- **Sections**: 8-byte aligned. 15 types defined (0-14, see [docs/api.md](docs/api.md#section-types)).
- **Full catalog**: At EOF. Per-entry checksums + shard statistics (`CategoryBitset` for pushdown).
- **CSR shard header**: 76 bytes (NOT 64 — spec diagram discrepancy), magic `b"SCXS"`.

### Codec System (SPEC.md §4)

- `None (0)` — Raw LE arrays
- `Scx1 (1)` — Delta-Golomb (indptr) + FOR-BP (indices) + Rice (values). **Integer only.**
- `Zstd (2)` — Zstd per-section. Fallback for float layers.

**Auto-codec** (Phase 2): `codec_select.rs` chooses Scx1 vs Zstd based on median value (threshold ≤ 8). Default for `from_anndata()` and `scx convert` is now `"auto"`.

Per-shard codec override: readers MUST use the shard header's `codec_id`, not the file header's.

### Key Types

- **On-disk**: `u64` indptr, `u16/u32` indices, `u8/u16/u32` integer values
- **In-memory (ScxCsr)**: `i64` indptr, `i32` indices, `f32` data — matching scipy CSR for zero-copy
- **Value encodings**: uint8 (0), uint16 (1), uint32 (2), float32 (3), float16 (4)
- **Index dtype**: u16 if n_vars <= 65535, else u32

### Phase 2 Crates

**scx-ops** — File lifecycle operations (SPEC §3.6):
- `append.rs` — append new shards at EOF, advisory flock
- `delete.rs` — logical deletion via Roaring Bitmap deletion vectors
- `compact.rs` — rewrite file reclaiming deleted/orphaned space
- `rollback.rs` — header-only revert to previous manifest
- `merge.rs` — streaming merge of multiple SCX files
- `flock.rs` — advisory file locking via `fs4`

**scx-engine** — Lazy query engine (SPEC §7):
- `pipeline.rs` — lazy builder: `open → filter → select → normalize → collect`
- `predicate.rs` — predicate parsing/evaluation (==, !=, <, >, <=, >=, in, and/or/not)
- `pushdown.rs` — catalog-level shard pruning (CategoryBitset) + index-level row pruning
- `projection.rs` — gene projection (CSR column subset)
- `fused_ops.rs` — fused normalize+log1p (single CSR row scan)
- `index.rs` — predicate index construction/read (SPEC §3.5)
- `collect.rs` — parallel pipeline execution

**scx-loader** — ML training data loader (SPEC §8):
- Triple-buffered pipeline: tokio I/O → rayon decode → Python/GPU
- All stages implemented: `pipeline.rs`, `io_stage.rs`, `decode_stage.rs`
- `shuffle.rs` — `ShardShuffler` (shard order) + `RowShuffler` (within-buffer Fisher-Yates)
- `projection.rs` — `HvgProjection` (gene subset at decode time)
- `normalize.rs` — dense-row fused normalize + log1p
- `python.rs` — `TrainingDataset` PyO3 class (implements `__iter__`/`__next__`)

**scx-cloud** — Cloud access operations (SPEC §12):
- `backend.rs` — `object_store` URL parsing + backend instantiation (S3, GCS, Azure, local)
- `cloud_optimize.rs` — front-of-file catalog writer for single-read cloud opens
- `explode.rs` — packed `.scx` → exploded `.scxd` directory
- `pack.rs` — exploded `.scxd` directory → packed `.scx`
- `pull.rs` — streaming cloud → local packed (parallel downloads, reorder buffer, selective filter)
- `push.rs` — streaming local → cloud exploded (parallel uploads)
- `coalesce.rs` — shard coalescing for efficient range reads
- `cloud_reader.rs` — `CloudReader` for direct cloud reads without full download

## Coding Conventions

### Serialization

- Do NOT use `#[repr(C)]` for on-disk structs. Serialize field-by-field with `byteorder::WriteBytesExt`/`ReadBytesExt` (little-endian).
- Every section starts at an 8-byte-aligned offset. Insert zero padding as needed.

### Error Handling

- Use `thiserror` for error enums: `ScxError` (scx-format), `EngineError` (scx-engine), `OpsError` (scx-ops), `LoaderError` (scx-loader), `CloudError` (scx-cloud).
- Readers must return errors (not panic) on malformed input, especially bitstream exhaustion.
- Validate magic bytes, endianness, and format version on file/shard open.

### Checksums

- BLAKE3 everywhere. Per-shard: BLAKE3 truncated to 64 bits. Catalog: full 32-byte BLAKE3.
- Shard checksum covers everything after the shard header.
- Full catalog ends with a 32-byte BLAKE3 of all preceding catalog bytes.

### Codec Details

See SPEC.md §4 for full codec specifications. Key parameters:

- **Rice**: Block size 256, `k = max(0, floor(log2(0.6931 * median)))`, clamped 0-15. Shift: `values[i] - 1`. No zero values.
- **FOR-BP**: Block size 128 rows. LEB128 nnz counts. Per-row: frame_min + frame_bits + bit-packed deltas.
- **Delta-Golomb**: Single stream, first value raw LE u64, then k byte, then Rice-encoded deltas.
- **Bitstream**: All codecs use **LSB-first** bit packing.

### Writer (Atomic Rename)

- Write to temp file, then `fsync()` + `rename()` to final path
- Sections start at offset 4352 (256 header + 4096 root catalog placeholder)
- `finish()`: write full catalog at EOF → pwrite root catalog at 256 → pwrite header at 0 → fsync → rename

### Python Bindings (pyscx)

- PyO3 with `Bound<'py, T>` API (not deprecated `&PyAny`)
- Pin `pyo3` and `numpy` crate to same minor version (currently 0.23)
- `PyArray::from_vec()` for zero-copy (moves Rust Vec to numpy)
- ScxCsr `i64/i32/f32` matches scipy exactly — avoids copy
- Arrow → pandas via pyarrow's `to_pandas()` for obs/var metadata

### Feature Flags

- `scx-cli`: `hdf5` (h5ad conversion), `cloud` (cloud operations) — both opt-in
- `pyscx`: `cloud` (cloud operations) — opt-in
- Build with: `cargo build --features hdf5,cloud` or `maturin develop --features cloud`

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained. Only needed for conversion. Fallback: Python subprocess with h5py.
- **h5ad files are messy**: missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns, varying categoricals. Handle gracefully.
- **Integer dtype detection**: h5ad often stores integer counts as float32. Detect and use Rice (not Zstd).
- **CSC → CSR transpose** is memory-intensive. Use streaming scatter approach.
- **tokio + rayon interaction** (scx-loader): Keep tokio for I/O only, rayon for CPU work. Bounded channels for back-pressure. No shared mutable state between runtimes.
- **Cloud auth**: `object_store` handles credentials via environment variables and instance metadata. No custom auth code.

## Known SPEC Discrepancies

1. **FileHeader**: `front_catalog_offset`/`length` fields used for cloud layout (SPEC §12.2). Non-cloud-optimized files write zero. Total: 256 bytes.
2. **ShardHeader**: Spec diagram says 64 bytes, actual fields sum to **76 bytes**. SPEC v0.5 clarifies.
3. **SPEC §16 roadmap**: Fixed in v0.5. Authoritative roadmap is ROADMAP.md.

## Out of Scope (Current Phase)

- GPU/CUDA/GDS, R bindings, CSC, multimodal, SIMD, detection bitmap (Phase 3)
- Rust-native PCA/kNN/UMAP accelerators (Phase 4 — optional)
