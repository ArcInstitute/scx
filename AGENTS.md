# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. Replaces AnnData/h5ad with a unified Rust-native stack.

**Phase 1 complete.** Phase 2 complete. Phase 3 partially complete (GPU, R bindings, CLI extensions). Phase 4a complete (scanpy parity — selective loading, chunk iteration, preprocessing pipeline). Phase 4b complete (Rust-native accelerators — PCA, kNN, UMAP, DE, pseudobulk). Phase 4c complete (GPU accelerators — benchmarked, 3/4 Go/No-Go gates pass). Phase 4d complete (lazy preprocessing — materialization-free normalize_total/log1p, streaming PCA via ShardSource trait, non-materializing filter_cells/filter_genes; 71% RSS reduction on 1M cells). Phase 4e complete (PCA/DE/Leiden optimization — covariance PCA for HVG data, optimized Wilcoxon DE with pre-ranking, Rust-native Leiden; pipeline 4.6× faster on 1M cells). See [ROADMAP.md](ROADMAP.md), [Phase3.md](Phase3.md), [Phase4.md](Phase4.md), [Phase4-GPU.md](Phase4-GPU.md), [Phase4-ACC-ALL.md](Phase4-ACC-ALL.md), and [PCA-DE-FIX.md](tasks/PCA-DE-FIX.md).

## Key Documents

- **[SPEC.md](SPEC.md)** — Format specification (v0.5). Authoritative reference for binary layouts, codecs, and section types.
- **[ROADMAP.md](ROADMAP.md)** — Phased implementation plan (Phases 1–4).
- **[Phase3.md](Phase3.md)** — Phase 3 plan (GPU, R bindings, CLI extensions — partially done).
- **[Phase4.md](./tasks/Phase4.md)** — Phase 4 plan (scanpy parity + Rust-native accelerators — 4a/4b complete).
- **[Phase4-GPU.md](./tasks/Phase4-GPU.md)** — Phase 4c GPU accelerator spec and benchmark results (cuSPARSE, cuVS/CAGRA, CUDA UMAP).
- **[Phase4-ACC-ALL.md](./tasks/Phase4-ACC-ALL.md)** — Phase 4d spec: materialization-free lazy preprocessing (ScxLazyTransformedDataset, column-projected streaming aggregation).
- **[PCA-DE-FIX.md](tasks/PCA-DE-FIX.md)** — Phase 4e spec: PCA covariance method, DE pre-ranking optimization, Rust-native Leiden. Root cause analysis, algorithm details, and benchmark results.
- **[COMPREHENSIVE-BENCHMARKING.md](COMPREHENSIVE-BENCHMARKING.md)** — Benchmark specs for accelerators and preprocessing.
- **[benchmarks/README.md](benchmarks/README.md)** — Practical guide to running benchmarks: SLURM job submission, dataset preparation, script reference. **Always use parallel SLURM job submission** (one job per benchmark×dataset pair) rather than sequential single-job scripts.
- **[docs/architecture.md](docs/architecture.md)** — Crate architecture and dependency details.
- **[docs/api.md](docs/api.md)** — API reference and section type documentation.
- **[docs/scanpy.md](docs/scanpy.md)** — Scanpy integration guide and accelerator usage.
- **[docs/gpu-setup.md](docs/gpu-setup.md)** — GPU setup guide: CUDA, RAPIDS, conda, container, SLURM, troubleshooting.
- **[docs/testing.md](docs/testing.md)** — Test and benchmark details.
- **[docs/multithreading.md](docs/multithreading.md)** — Multithreading architecture across crates.
- **[docs/sharding.md](docs/sharding.md)** — Sharding design and usage.
- **[tasks/](tasks/)** — Completed phase specs and code reviews.

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
            ├─> scx-accel (depends on scx-format, scx-sparse; PCA/kNN/UMAP/DE/Leiden; optional gpu dep on scx-gpu)
            ├─> scx-cli (depends on all above)
            ├─> pyscx (depends on all above)
            └─> rscx (depends on scx-format, scx-codec, scx-sparse, scx-engine, scx-ops)
```

14 workspace members total (including `scx-integration-tests`). See [docs/architecture.md](docs/architecture.md) for details.

Key isolation rules:
- `scx-loader` does NOT depend on `scx-engine` — it has its own streaming-optimized gene projection and fused ops.
- `scx-cloud` does NOT depend on `scx-loader` — they are siblings.
- `scx-accel` depends only on `scx-format` and `scx-sparse` — no engine/loader dependency.

### Feature Flags

- `scx-cli`: `hdf5` (h5ad conversion), `cloud` (cloud operations) — both opt-in
- `pyscx`: `cloud` (cloud operations), `gpu` (GPU-accelerated analysis) — both opt-in
- `scx-gpu`: `gds` (GPUDirect Storage) — opt-in, requires nvidia-fs drivers
- `scx-accel`: `gpu` (GPU dispatch via `scx-gpu`) — opt-in, requires CUDA Toolkit ≥ 12.0
- Build with: `cargo build --features hdf5,cloud` or `maturin develop --features cloud,gpu`

### File Format Summary

See [SPEC.md](SPEC.md) §3 for full details.

- **File header**: 256 bytes, LE, magic `b"SCX\x01"`. Includes `front_catalog_offset`/`length` (populated by `cloud-optimize`, zero otherwise).
- **Root catalog**: At offset 256, max 4096 bytes.
- **Sections**: 8-byte aligned. 15 types defined (0–14, see [docs/api.md](docs/api.md#section-types)).
- **Full catalog**: At EOF. Per-entry checksums + shard statistics (`CategoryBitset` for pushdown).
- **CSR shard header**: 76 bytes (NOT 64 — spec diagram discrepancy), magic `b"SCXS"`.

### Codec System

- `None (0)` — Raw LE arrays
- `Scx1 (1)` — Delta-Golomb (indptr) + FOR-BP (indices, SIMD BitPacker4x) + Rice (values). **Integer only.**
- `Zstd (2)` — Zstd per-section. Fallback for float layers.
- `Lz4Shuffle (3)` — Byte-shuffle pre-filter + LZ4 frame compression. Works with any value encoding. Matches Zarr/Blosc compression style.

**Auto-codec**: `codec_select.rs` chooses Scx1 vs Zstd based on median value (threshold ≤ 8). Default for `from_anndata()` and `scx convert` is `"auto"`.

Per-shard codec override: readers MUST use the shard header's `codec_id`, not the file header's.

### Key Types

- **On-disk**: `u64` indptr, `u16/u32` indices, `u8/u16/u32` integer values
- **In-memory (ScxCsr)**: `i64` indptr, `i32` indices, `f32` data — matching scipy CSR for zero-copy
- **Value encodings**: uint8 (0), uint16 (1), uint32 (2), float32 (3), float16 (4)
- **Index dtype**: u16 if n_vars <= 65535, else u32

### Phase Status Summary

- **Phase 1 (Format + Codec + AnnData Bridge)**: Complete.
- **Phase 2 (Training Loader + Query Engine)**: Complete.
- **Phase 3**: Steps 1–3 complete (SIMD FOR-BP, scx-gpu, rscx). Steps 4–5 partially done. Comprehensive benchmarking complete (D1–D7, 12 formats, 6 benchmarks).
- **Sprint 2 (Read Gap Closure)**: Complete — parallel shard decode (2A), catalog seek (2B), format cleanup (2C), LZ4+shuffle codec (2D), SIMD FOR-BP (2E), madvise hints (2F), integration testing (2G).
- **Sprint 3 (Comprehensive Benchmarks)**: Complete — SCX is now the fastest reader at census scale (1.38× faster than Zarr lz4 on 1M cells, 1.15× on 5M cells), achieves up to 7× parallel scaling at 32 threads, and writes are only 2–3× slower than Zarr (down from 10–60×). Column projection is 4–8× faster than all competitors. See `benchmarks/comprehensive/reporting/phase3_report.md`.
- **Phase 4a (Scanpy Integration)**: Complete — backed mode aggregation, comparison optimization, streaming preprocess, chunk iterator, selective loading.
- **Phase 4b (Rust-Native Accelerators)**: Complete — PCA, kNN, UMAP, Wilcoxon DE (in-memory + streaming), pseudobulk DE, stratified DE. All in `scx-accel` crate with `pyscx.accel.*` Python API.
- **Phase 4c (GPU Accelerators)**: Implementation complete, benchmarked — cuSPARSE SpMM, cuSOLVER QR, cuRAND (GPU PCA), cuVS CAGRA (GPU kNN, 9.4× standalone on 1M cells), native CUDA UMAP SGD (7.7× on 1M cells), cuGraph Leiden (16× on 1M cells), fused GPU preprocessing (normalize+log1p). All accessible via `device="gpu"` parameter in `pyscx.accel.*`. Go/No-Go: 3/4 pass (PCA correctness, kNN recall, graceful fallback); 10× pipeline target not met (3.8× median, 8.8× best run). Requires `scx-gpu` conda env for RAPIDS compatibility. See `benchmarks/results/gpu_gonogo.json`.
- **Phase 4d (Lazy Preprocessing)**: Complete — `ScxLazyTransformedDataset` wraps backed reader with chained transforms (NormalizeTotal, Log1p, RowScale). `ShardSource` trait enables streaming PCA through transforms without materialization. `pyscx.accel.normalize_total()`, `log1p()`, `filter_cells()`, `filter_genes()`, `subset_obs()`, `calculate_qc_metrics()` all work without materializing. Fused NormalizeTotal+Log1p optimization. Full out-of-core pipeline: open → QC filter → normalize → log1p → PCA → kNN → UMAP → Leiden with ~11 GB peak RSS on 1M cells (71% reduction from 38 GB materialized baseline). Benchmarked on census_1m (SLURM job 1935465). See [Phase4-ACC-ALL.md](Phase4-ACC-ALL.md).
- **Phase 4e (PCA/DE/Leiden Optimization)**: Complete — three major accelerator improvements. (1) **Covariance PCA**: exact eigendecomposition via sparse outer product accumulation for HVG-scale data (n_vars ≤ 5000), auto-routed in `pyscx.accel.pca()`; 4.2s on 1M cells (was 21s, **5× faster**, 1.9× faster than scanpy). (2) **Optimized Wilcoxon DE**: pre-ranking approach ranks all cells once per gene and reuses across groups (10× fewer sorts); `rankby_abs` parameter matches scanpy's default signed-score ranking; 5.4s on 1M cells (was 27s, **5× faster**, 3.2× faster than scanpy). (3) **Rust-native Leiden**: full Leiden algorithm (Traag et al. 2019) with RB configuration model, sequential and parallel modes, adapted from single-clustering (BSD 3-Clause); 55s on 1M cells (was 2,226s via Python leidenalg in same-conditions comparison, **40× faster**); ARI 0.92 vs Python leidenalg. Pipeline total: 870s on 1M cells (was 3,971s, **4.6× faster**). See [PCA-DE-FIX.md](tasks/PCA-DE-FIX.md).
- **Phase 4 ML Loader Benchmarks (§3.6)**: Complete — comprehensive ML data loader throughput benchmarks comparing SCX TrainingDataset vs AnnData, TileDB-SOMA-ML, and scDataLoader across pbmc3k, tabula_sapiens_100k, census_1m. SCX achieves 1,405 batches/sec on census_1m (hvg_norm scenario) vs 17.1 for TileDB-SOMA-ML (**82× faster**). GPU training scenario benchmarked with scVI-equivalent VAE. See `benchmarks/comprehensive/benchmarks/ml_loader.py` and results in `benchmarks/comprehensive/results/raw/ml_loader__*`.

## Coding Conventions

### Serialization

- Do NOT use `#[repr(C)]` for on-disk structs. Serialize field-by-field with `byteorder::WriteBytesExt`/`ReadBytesExt` (little-endian).
- Every section starts at an 8-byte-aligned offset. Insert zero padding as needed.

### Error Handling

- Use `thiserror` for error enums: `ScxError` (scx-format), `EngineError` (scx-engine), `OpsError` (scx-ops), `LoaderError` (scx-loader), `CloudError` (scx-cloud), `GpuError` (scx-gpu), `AccelError` (scx-accel).
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
- Accelerators exposed via `pyscx.accel.*` — results written to standard AnnData slots
- Optional Python deps (`pydeseq2`) imported at runtime with clear `ImportError` if missing

### R Bindings (rscx)

- `extendr` v0.8.x for Rust↔R FFI
- SCX CSR (row-major) must be transposed to dgCMatrix (CSC, column-major) for R/Matrix interop
- R has no unsigned integers — use `i32` for all integer arguments from R, convert internally

### GPU (scx-gpu)

- Uses `cudarc` for CUDA runtime/driver API
- CUDA kernels compiled via `cc` build script (`build.rs`)
- GPU decoders must produce bit-identical output to scalar CPU reference
- GDS requires local NVMe + nvidia-fs drivers + ext4/XFS filesystem; always falls back to CPU path

### Accelerators (scx-accel)

- Uses `faer` for dense linear algebra (QR, SVD, eigendecomposition in PCA)
- Uses `instant-distance` for HNSW-based approximate kNN
- PCA: two methods auto-routed by `n_vars`:
  - **Covariance PCA** (n_vars ≤ 5000): sparse outer product accumulation (`X^T @ X` directly from CSR nonzeros with symmetry exploitation), eigendecomposition. Exact results, faster for HVG-selected data.
  - **Randomized SVD** (n_vars > 5000): streaming SpMM shard-by-shard with zero-copy `MatRef::from_row_major_slice` views. No full matrix materialization.
- DE: pre-ranking approach ranks all cells once per gene, reuses ranks across groups (10× fewer sorts). `rankby_abs` parameter (default `false`) matches scanpy's signed-score ranking.
- Leiden: Rust-native implementation (Traag et al. 2019) with RB configuration model. Sequential mode (default) matches C++ leidenalg convergence; parallel mode uses conflict-free graph coloring. Adapted from `single-clustering` (BSD 3-Clause). Uses `rand_chacha` for deterministic seeding and `libc` for `malloc_trim` on Linux.
- Pseudobulk aggregation streams via `BackedCsrReader`, statistical testing delegated to `pydeseq2`

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained. Only needed for conversion. Fallback: Python subprocess with h5py.
- **h5ad files are messy**: missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns. Handle gracefully.
- **tokio + rayon interaction** (scx-loader): Keep tokio for I/O only, rayon for CPU work. Bounded channels for back-pressure.
- **Cloud auth**: `object_store` handles credentials via environment variables and instance metadata. No custom auth code.
- **Known bugs**: See [Phase4_CODE-REVIEW.md](tasks/Phase4_CODE-REVIEW.md) for latest issues and fix priorities.

## Known SPEC Discrepancies

1. **FileHeader**: `front_catalog_offset`/`length` fields for cloud layout (SPEC §12.2). Non-cloud-optimized files write zero. Total: 256 bytes.
2. **ShardHeader**: Spec diagram says 64 bytes, actual fields sum to **76 bytes**. SPEC v0.5 clarifies.
3. **SPEC §16 roadmap**: Fixed in v0.5. Authoritative roadmap is ROADMAP.md.

## Out of Scope (Current Phase)

- CSC as primary storage, multimodal (Phase 3, Steps 4–5 — planned)
- GPU-accelerated analysis: 10× pipeline target not met (3.8× achieved on 1M cells; kNN 9.4×, UMAP 7.6×, Leiden 16×)
