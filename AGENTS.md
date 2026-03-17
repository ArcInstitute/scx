# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. Replaces AnnData/h5ad with a unified Rust-native stack.

**Phase 1 (Format + Codec + AnnData Bridge) is feature-complete.** Next step: Go/No-Go gate evaluation before Phase 2. See [README.md](README.md) for project overview, repo structure, and quick start. See [docs/api.md](docs/api.md) for full API reference and [docs/testing.md](docs/testing.md) for test/benchmark details.

## Key Documents

- **SPEC.md** — Format specification (v0.5). Authoritative reference for binary layouts, codecs, and section types.
- **ROADMAP.md** — Phased implementation plan (Phases 1-4).
- **Phase1.md** — Phase 1 implementation plan (all tasks complete).

## Build and Test

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check

# Python bindings (always use uv venv at .venv/):
cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v
```

**Always use the uv venv** at `.venv/` for all Python work. Do NOT use system Python or pip directly.

## Architecture

### Crate Dependency Order

```
scx-codec (standalone)
    └─> scx-format (depends on scx-codec)
            └─> scx-sparse (standalone, used by scx-format reader)
                    └─> scx-cli (depends on scx-format, scx-codec, scx-sparse)
                    └─> pyscx (depends on scx-format, scx-codec, scx-sparse)
```

`scx-cli` has an optional `hdf5` feature flag for h5ad conversion (opt-in).

### File Format (SPEC.md §3)

- **File header**: 256 bytes, LE, magic `b"SCX\x01"`. Includes `front_catalog_offset`/`length` (zero in Phase 1, reserved for cloud layout).
- **Root catalog**: At offset 256, max 4096 bytes.
- **Sections**: 8-byte aligned. 13 types defined (see [docs/api.md](docs/api.md#section-types)).
- **Full catalog**: At EOF. Complete section index with BLAKE3 checksums.
- **CSR shard header**: 76 bytes (NOT 64 — spec diagram discrepancy), magic `b"SCXS"`.

### Codec System (SPEC.md §4)

- `None (0)` — Raw LE arrays
- `Scx1 (1)` — Delta-Golomb (indptr) + FOR-BP (indices) + Rice (values). **Integer only.**
- `Zstd (2)` — Zstd per-section. Fallback for float layers.

Per-shard codec override: readers MUST use the shard header's `codec_id`, not the file header's.

### Key Types

- **On-disk**: `u64` indptr, `u16/u32` indices, `u8/u16/u32` integer values
- **In-memory (ScxCsr)**: `i64` indptr, `i32` indices, `f32` data — matching scipy CSR for zero-copy
- **Value encodings**: uint8 (0), uint16 (1), uint32 (2), float32 (3), float16 (4)
- **Index dtype**: u16 if n_vars <= 65535, else u32

## Coding Conventions

### Serialization

- Do NOT use `#[repr(C)]` for on-disk structs. Serialize field-by-field with `byteorder::WriteBytesExt`/`ReadBytesExt` (little-endian).
- Every section starts at an 8-byte-aligned offset. Insert zero padding as needed.

### Error Handling

- Use `thiserror` for the `ScxError` enum in `scx-format/src/error.rs`.
- Readers must return errors (not panic) on malformed input, especially bitstream exhaustion.
- Validate magic bytes, endianness, and format version on file/shard open.

### Checksums

- BLAKE3 everywhere. Per-shard: BLAKE3 truncated to 64 bits. Catalog: full 32-byte BLAKE3.
- Shard checksum covers everything after the shard header.
- Full catalog ends with a 32-byte BLAKE3 of all preceding catalog bytes.

### Codec Details

**Rice (SPEC §4.3)**: Block size 256 values. Shift: `values[i] - 1`. Parameter: `k = max(0, floor(log2(0.6931 * median)))`, clamped 0-15. Floor median. Byte-padded blocks. No zero values.

**FOR-BP (SPEC §4.2)**: Block size 128 rows. LEB128 nnz counts. Per-row: frame_min + frame_bits + bit-packed deltas. Empty rows: varint 0 only. `delta[0]` always 0.

**Delta-Golomb (SPEC §4.1)**: Single stream (no blocks). First value raw LE u64, then k byte, then Rice-encoded deltas. Reconstruct via prefix sum.

**Bitstream**: All codecs use **LSB-first** bit packing.

### Writer (Atomic Rename)

- Write to temp file, then `fsync()` + `rename()` to final path
- Sections start at offset 4352 (256 header + 4096 root catalog placeholder)
- `finish()`: write full catalog at EOF -> pwrite root catalog at 256 -> pwrite header at 0 -> fsync -> rename

### Python Bindings (pyscx)

- PyO3 with `Bound<'py, T>` API (not deprecated `&PyAny`)
- Pin `pyo3` and `numpy` crate to same minor version (currently 0.23)
- `PyArray::from_vec()` for zero-copy (moves Rust Vec to numpy)
- ScxCsr `i64/i32/f32` matches scipy exactly — avoids copy
- Arrow -> pandas via pyarrow's `to_pandas()` for obs/var metadata

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained. Only needed for conversion. Fallback: Python subprocess with h5py.
- **h5ad files are messy**: missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns, varying categoricals. Handle gracefully.
- **Integer dtype detection**: h5ad often stores integer counts as float32. Detect and use Rice (not Zstd).
- **CSC -> CSR transpose** is memory-intensive. Use streaming scatter approach.

## Known SPEC Discrepancies

1. **FileHeader**: `front_catalog_offset`/`length` fields added for cloud layout (SPEC §12.2). Phase 1 writes zero. Total: 256 bytes.
2. **ShardHeader**: Spec diagram says 64 bytes, actual fields sum to **76 bytes**. SPEC v0.5 clarifies.
3. **SPEC §16 roadmap**: Fixed in v0.5. Authoritative roadmap is ROADMAP.md.

## Out of Scope (Phase 1)

- Training data loader, query engine, predicate pushdown (Phase 2)
- Append/delete/compact/rollback (Phase 2)
- Fused normalize+log1p (Phase 2)
- Cloud operations: pull/push/explode/pack (Phase 2+)
- GPU/CUDA/GDS, R bindings, CSC, multimodal, SIMD, detection bitmap (Phase 3)
