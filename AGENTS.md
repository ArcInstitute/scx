# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. It replaces AnnData/h5ad with a unified Rust-native stack.

The project is currently in **Phase 1** (Format + Codec + AnnData Bridge). The goal is: `h5ad -> scx convert -> scx.open().to_anndata() -> scanpy works`.

## Repository Structure

```
scx/
├── Cargo.toml              # workspace root
├── scx-format/             # file layout, header, catalog, shard I/O
├── scx-codec/              # compression codecs (Rice, FOR-BP, Delta-Golomb)
├── scx-sparse/             # CSR operations
├── scx-cli/                # CLI tool (convert, info, validate)
├── pyscx/                  # Python bindings (PyO3 + maturin)
├── tests/                  # integration tests + reference files
└── benchmarks/             # benchmark scripts + results
```

## Key Specification Documents

- **SPEC.md** — Full format specification (v0.5). The authoritative reference for all binary layouts, codec algorithms, and section types.
- **ROADMAP.md** — Phased implementation plan (Phases 1-4).
- **Phase1.md** — Detailed implementation plan for Phase 1 with code sketches, test plans, and pitfall notes.

## Build and Test

```bash
cargo test --workspace          # run all tests
cargo clippy --workspace -- -D warnings
cargo fmt --check

# Python bindings
cd pyscx && maturin develop && pytest
```

## Architecture

### Crate Dependency Order

```
scx-codec (standalone)
    └─> scx-format (depends on scx-codec for encode/decode)
            └─> scx-sparse (standalone, used by scx-format reader)
                    └─> scx-cli (depends on scx-format, scx-codec, scx-sparse)
                    └─> pyscx (depends on scx-format, scx-sparse)
```

### File Format (SPEC.md §3)

- **File header**: 256 bytes, little-endian, starts with magic `b"SCX\x01"`
- **Root catalog**: At offset 256, max 4096 bytes. Compact summary for fast open.
- **Sections**: 8-byte aligned. Types include obs/var metadata (Arrow IPC), CSR shards, layers, obsm, obsp, uns (JSON), provenance.
- **Full catalog**: At end of file. Complete index of all sections with BLAKE3 checksums.
- **CSR shard header**: 76 bytes (NOT 64 — known spec diagram discrepancy), starts with magic `b"SCXS"`.

### Codec System (SPEC.md §4)

Three codec IDs:
- `None (0)` — Raw LE arrays, no compression
- `Scx1 (1)` — Domain-specific: Delta-Golomb for indptr + FOR-BP for indices + Rice for values. **Integer value encodings only.**
- `Zstd (2)` — Zstd per-section. Fallback for float layers.

Per-shard codec override: readers MUST use the shard header's `codec_id`, not the file header's.

### Bitstream Convention

All codecs use **LSB-first** bit packing. When writing value `0b1101` as 4 bits, bit 0 (value `1`) goes into the least significant available bit of the current byte.

### Key Types

- **On-disk**: `u64` indptr, `u16/u32` indices, `u8/u16/u32` integer values
- **In-memory (ScxCsr)**: `i64` indptr, `i32` indices, `f32` data — matching scipy CSR layout for zero-copy interop
- **Value encodings**: uint8 (0), uint16 (1), uint32 (2), float32 (3), float16 (4)
- **Index dtype**: u16 if n_vars <= 65535, else u32

## Coding Conventions

### Serialization

- Do NOT use `#[repr(C)]` for on-disk structs. Serialize/deserialize field-by-field with `byteorder::WriteBytesExt`/`ReadBytesExt` to guarantee little-endian regardless of platform.
- All multi-byte integers in the file format are little-endian.
- Every section starts at an 8-byte-aligned offset. Insert zero padding as needed.

### Error Handling

- Use `thiserror` for the `ScxError` enum in `scx-format/src/error.rs`.
- Readers must return errors (not panic) on malformed input, especially bitstream exhaustion.
- Validate magic bytes, endianness, and format version on file/shard open.

### Checksums

- BLAKE3 everywhere. Per-shard checksums are BLAKE3 truncated to 64 bits. Catalog checksums are full 32-byte BLAKE3.
- Shard checksum covers everything after the shard header (indptr + indices + values + block index).
- Full catalog ends with a 32-byte BLAKE3 of all preceding catalog bytes.

### Rice Codec Details (SPEC §4.3)

- Block size `B_val = 256` values
- Values are non-zero counts (always >= 1); shift: `shifted[i] = values[i] - 1`
- Rice parameter: `k = max(0, floor(log2(0.6931 * median)))`, clamped to 0-15
- Median: use floor median (lower of two middle values for even-length blocks) — deterministic, no float
- Each block's bitstream padded to byte boundary
- Debug assert that no zero values enter the encoder

### FOR-BP Codec Details (SPEC §4.2)

- Block size `B_idx = 128` rows
- LEB128 varints for per-row nnz counts
- Per-row: frame_min (u16 or u32 per index_dtype) + frame_bits (u8) + bit-packed deltas
- Empty rows (nnz=0): emit varint 0, no frame_min/bits/deltas
- `delta[0]` is always 0 (included per spec for simplicity)

### Delta-Golomb Codec Details (SPEC §4.1)

- Single stream (no blocks), unlike Rice
- First value: raw LE u64 (8 bytes), then k byte, then Rice-encoded deltas
- Reconstruct via prefix sum

### Writer (Atomic Rename)

- Write to temp file, then `fsync()` + `rename()` to final path (POSIX atomic)
- Sections start at offset 4352 (256 header + 4096 root catalog placeholder)
- `finish()` sequence: write full catalog at EOF -> pwrite root catalog at 256 -> pwrite header at 0 -> fsync -> rename

### Python Bindings (pyscx)

- PyO3 with `Bound<'py, T>` API (not deprecated `&PyAny`)
- Pin `pyo3` and `numpy` crate to same minor version
- `PyArray::from_vec()` for zero-copy transfer (moves Rust Vec ownership to numpy)
- ScxCsr uses `i64/i32/f32` to match scipy's expected dtypes exactly — avoids copy
- Arrow -> pandas via pyarrow's `to_pandas()` for obs/var metadata
- Build with maturin

## Phase 1 Go/No-Go Gate

These criteria must be met before proceeding to Phase 2:
1. h5ad -> scx -> h5ad round-trip is bit-exact for integer counts
2. SCX file < 60% the size of h5ad for typical datasets
3. `scx.open().to_anndata()` -> full scanpy pipeline (QC -> PCA -> Leiden -> DE) works

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained (last release Nov 2021). Only needed for conversion, not the native reader. Fallback: Python subprocess with h5py.
- **h5ad files in the wild are messy**: missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns objects, varying categorical encodings. Handle all gracefully.
- **Integer dtype detection**: h5ad often stores integer counts as float32. Detect and use Rice coding (not Zstd) for compression advantage.
- **CSC -> CSR transpose** is memory-intensive for large matrices. Use streaming scatter approach.
- **60% compression target** is for the expression matrix. Metadata may not compress as well. Report per-component sizes honestly.

## Out of Scope (Phase 1)

- Training data loader, query engine, predicate pushdown (Phase 2)
- GPU/CUDA/GDS (Phase 3)
- R bindings (Phase 3)
- CSC / build-csc (Phase 3)
- Multimodal support (Phase 3)
- SIMD codec optimizations (Phase 3)
