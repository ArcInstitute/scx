# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. Replaces AnnData/h5ad with a unified Rust-native stack.

**Phase 1 complete.** Phase 2 complete. Phase 3 in progress (GPU, R bindings, CLI extensions). See [ROADMAP.md](ROADMAP.md) for the full phased plan and [Phase3.md](Phase3.md) for current work.

## Key Documents

- **[SPEC.md](SPEC.md)** — Format specification (v0.5). Authoritative reference for binary layouts, codecs, and section types.
- **[ROADMAP.md](ROADMAP.md)** — Phased implementation plan (Phases 1–4).
- **[Phase3.md](Phase3.md)** — Phase 3 implementation plan (GPU, R bindings, CLI extensions, multimodal — in progress).
- **[docs/architecture.md](docs/architecture.md)** — Crate architecture and dependency details.
- **[docs/api.md](docs/api.md)** — API reference and section type documentation.
- **[docs/testing.md](docs/testing.md)** — Test and benchmark details.
- **[docs/multithreading.md](docs/multithreading.md)** — Multithreading architecture across crates.
- **[docs/sharding.md](docs/sharding.md)** — Sharding design and usage.
- **[2026-03-25_CODE-REVIEW.md](2026-03-25_CODE-REVIEW.md)** — Latest codebase review with known issues and fix priorities.
- **[tasks/](tasks/)** — Completed phase specs (Phase1.md, Phase2.md, Phase2-Step3–7.md, Phase2-CLOUD.md, Phase3-Step2–5.md).

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

# R bindings:
cd rscx && R CMD INSTALL .
```

**Always use the uv venv** at `.venv/` for all Python work. Do NOT use system Python or pip directly.

## Architecture

### Crate Dependency Order

```
scx-codec (standalone)
    └─> scx-format (depends on scx-codec)
            ├─> scx-sparse (standalone, used by scx-format)
            ├─> scx-mtx (depends on scx-format; MTX format I/O)
            ├─> scx-ops (depends on scx-format; append/delete/compact/merge/rollback)
            ├─> scx-engine (depends on scx-format, scx-sparse, scx-ops)
            ├─> scx-loader (depends on scx-format, scx-codec, scx-sparse)
            ├─> scx-cloud (depends on scx-format, scx-codec, scx-engine)
            ├─> scx-gpu (depends on scx-format, scx-codec, scx-sparse)
            ├─> scx-cli (depends on all above)
            ├─> pyscx (depends on all above)
            └─> rscx (depends on scx-format, scx-codec, scx-sparse, scx-engine, scx-ops)
```

12 workspace members total. See [docs/architecture.md](docs/architecture.md) for detailed crate descriptions.

Key isolation rules:
- `scx-loader` does NOT depend on `scx-engine` — it has its own streaming-optimized gene projection and fused ops.
- `scx-cloud` does NOT depend on `scx-loader` — they are siblings.

### Feature Flags

- `scx-cli`: `hdf5` (h5ad conversion), `cloud` (cloud operations) — both opt-in
- `pyscx`: `cloud` (cloud operations) — opt-in
- `scx-gpu`: `gds` (GPUDirect Storage) — opt-in, requires nvidia-fs drivers
- Build with: `cargo build --features hdf5,cloud` or `maturin develop --features cloud`

### File Format Summary

See [SPEC.md](SPEC.md) §3 for full details.

- **File header**: 256 bytes, LE, magic `b"SCX\x01"`. Includes `front_catalog_offset`/`length` (populated by `cloud-optimize`, zero otherwise).
- **Root catalog**: At offset 256, max 4096 bytes.
- **Sections**: 8-byte aligned. 15 types defined (0–14, see [docs/api.md](docs/api.md#section-types)).
- **Full catalog**: At EOF. Per-entry checksums + shard statistics (`CategoryBitset` for pushdown).
- **CSR shard header**: 76 bytes (NOT 64 — spec diagram discrepancy), magic `b"SCXS"`.

### Codec System

See [SPEC.md](SPEC.md) §4 for full codec specifications and parameters.

- `None (0)` — Raw LE arrays
- `Scx1 (1)` — Delta-Golomb (indptr) + FOR-BP (indices) + Rice (values). **Integer only.**
- `Zstd (2)` — Zstd per-section. Fallback for float layers.

**Auto-codec**: `codec_select.rs` chooses Scx1 vs Zstd based on median value (threshold ≤ 8). Default for `from_anndata()` and `scx convert` is `"auto"`.

Per-shard codec override: readers MUST use the shard header's `codec_id`, not the file header's.

### Key Types

- **On-disk**: `u64` indptr, `u16/u32` indices, `u8/u16/u32` integer values
- **In-memory (ScxCsr)**: `i64` indptr, `i32` indices, `f32` data — matching scipy CSR for zero-copy
- **Value encodings**: uint8 (0), uint16 (1), uint32 (2), float32 (3), float16 (4)
- **Index dtype**: u16 if n_vars <= 65535, else u32

### Phase 3 Status

- **Step 1 (SIMD)**: Skipped — scalar implementations benchmarked as near-optimal. See [Phase3.md](Phase3.md) §Step 1.
- **Step 2 (scx-gpu)**: In progress — CUDA Rice/FOR-BP decoders, cuSPARSE interop, sparse-to-dense conversion implemented. GDS pending nvidia-fs module load.
- **Step 3 (rscx)**: Implemented — R bindings via extendr with Seurat v5 and SingleCellExperiment interop, pipe-friendly query API.
- **Step 4 (Multimodal)**: Planned — CITE-seq, spatial transcriptomics, h5mu conversion.
- **Step 5 (CLI extensions)**: Partially done — `build-csc`, `subset`, `upgrade`, `benchmark`, `query` subcommands implemented.

## Coding Conventions

### Serialization

- Do NOT use `#[repr(C)]` for on-disk structs. Serialize field-by-field with `byteorder::WriteBytesExt`/`ReadBytesExt` (little-endian).
- Every section starts at an 8-byte-aligned offset. Insert zero padding as needed.

### Error Handling

- Use `thiserror` for error enums: `ScxError` (scx-format), `EngineError` (scx-engine), `OpsError` (scx-ops), `LoaderError` (scx-loader), `CloudError` (scx-cloud), `GpuError` (scx-gpu).
- Readers must return errors (not panic) on malformed input, especially bitstream exhaustion.
- Validate magic bytes, endianness, and format version on file/shard open.

### Checksums

- BLAKE3 everywhere. Per-shard: BLAKE3 truncated to 64 bits. Catalog: full 32-byte BLAKE3.
- Shard checksum covers everything after the shard header.
- Full catalog ends with a 32-byte BLAKE3 of all preceding catalog bytes.

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

### R Bindings (rscx)

- `extendr` v0.8.x for Rust↔R FFI
- SCX CSR (row-major) must be transposed to dgCMatrix (CSC, column-major) for R/Matrix interop
- R has no unsigned integers — use `i32` for all integer arguments from R, convert internally

### GPU (scx-gpu)

- Uses `cudarc` for CUDA runtime/driver API
- CUDA kernels compiled via `cc` build script (`build.rs`)
- GPU decoders must produce bit-identical output to scalar CPU reference
- GDS requires local NVMe + nvidia-fs drivers + ext4/XFS filesystem; always falls back to CPU path

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained. Only needed for conversion. Fallback: Python subprocess with h5py.
- **h5ad files are messy**: missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns. Handle gracefully.
- **tokio + rayon interaction** (scx-loader): Keep tokio for I/O only, rayon for CPU work. Bounded channels for back-pressure. No shared mutable state between runtimes.
- **Cloud auth**: `object_store` handles credentials via environment variables and instance metadata. No custom auth code.
- **Known bugs**: See [2026-03-25_CODE-REVIEW.md](2026-03-25_CODE-REVIEW.md) for critical issues (merge/compact codec selection, encode_value duplication, missing bounds checks).

## Known SPEC Discrepancies

1. **FileHeader**: `front_catalog_offset`/`length` fields for cloud layout (SPEC §12.2). Non-cloud-optimized files write zero. Total: 256 bytes.
2. **ShardHeader**: Spec diagram says 64 bytes, actual fields sum to **76 bytes**. SPEC v0.5 clarifies.
3. **SPEC §16 roadmap**: Fixed in v0.5. Authoritative roadmap is ROADMAP.md.

## Out of Scope (Current Phase)

- CSC as primary storage, multimodal (Phase 3, Steps 4–5 — planned)
- Rust-native PCA/kNN/UMAP accelerators (Phase 4 — optional)
