# AGENTS.md

## Project Overview

SCX (Sparse Cell eXpression System) is a purpose-built binary file format, compression codec, query engine, and ML data loader for single-cell RNA-seq data. It replaces AnnData/h5ad with a unified Rust-native stack providing 3-7x smaller files, fastest reads at census scale, 4-44x less memory, a GPU-saturating training loader, lazy query engine, and Rust-native analysis accelerators — with native bindings for Python and R, fully compatible with the scverse ecosystem and Seurat v5.

## Key Documents

- **[ROADMAP.md](ROADMAP.md)** — Historical phased implementation plan (Phases 1-4).
- **[docs/architecture.md](docs/architecture.md)** — Crate architecture and dependency details.
- **[docs/format.md](docs/format.md)** — Binary format reference: file header, catalogs, CSR shard layout, fragment/manifest model, checksums.
- **[docs/codec.md](docs/codec.md)** — Bit-level codec specification: Delta-Golomb-Rice, FOR-BP, Rice, LZ4+shuffle, auto-selection.
- **[docs/api.md](docs/api.md)** — API reference and section type documentation.
- **[docs/scanpy.md](docs/scanpy.md)** — Scanpy integration guide and accelerator usage.
- **[docs/performance.md](docs/performance.md)** — Benchmark results and performance characteristics.
- **[docs/gpu-setup.md](docs/gpu-setup.md)** — GPU setup guide: CUDA, RAPIDS, conda, container, SLURM, troubleshooting.
- **[docs/testing.md](docs/testing.md)** — Test, benchmark, and correctness validation details.
- **[docs/multithreading.md](docs/multithreading.md)** — Multithreading architecture across crates.
- **[docs/sharding.md](docs/sharding.md)** — Sharding design and usage.
- **[docs/cloud.md](docs/cloud.md)** — Using SCX in cloud environments: auth, layouts, tuning, provider-specific notes.
- **[benchmarks/README.md](benchmarks/README.md)** — Practical guide to running benchmarks: SLURM job submission, dataset preparation, script reference. **Always use parallel SLURM job submission** (one job per benchmark x dataset pair) rather than sequential single-job scripts.
- **[tasks/](tasks/)** — Historical phase specs and code reviews.

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
- `scx-accel`: `gpu` (GPU dispatch via `scx-gpu`) — opt-in, requires CUDA Toolkit >= 12.0
- Build with: `cargo build --features hdf5,cloud` or `maturin develop --features cloud,gpu`

### File Format Summary

See [docs/format.md](docs/format.md) for full details.

- **File header**: 256 bytes, LE, magic `b"SCX\x01"`. Includes `front_catalog_offset`/`length` (populated by `cloud-optimize`, zero otherwise).
- **Root catalog**: At offset 256, max 4096 bytes.
- **Sections**: 8-byte aligned. 15 types defined (0-14, see [docs/api.md](docs/api.md#section-types)).
- **Full catalog**: At EOF. Per-entry checksums + shard statistics (`CategoryBitset` for pushdown).
- **CSR shard header**: 76 bytes, magic `b"SCXS"`.

### Codec System

- `None (0)` — Raw LE arrays
- `Scx1 (1)` — Delta-Golomb (indptr) + FOR-BP (indices, SIMD BitPacker4x) + Rice (values). **Integer only.**
- `Zstd (2)` — Zstd per-section. Fallback for float layers.
- `Lz4Shuffle (3)` — Byte-shuffle pre-filter + LZ4 frame compression. Works with any value encoding. Matches Zarr/Blosc compression style.

**Auto-codec**: `codec_select.rs` chooses Scx1 vs Zstd based on median value (threshold <= 8). Default for `from_anndata()` and `scx convert` is `"auto"`.

Per-shard codec override: readers MUST use the shard header's `codec_id`, not the file header's.

### Key Types

- **On-disk**: `u64` indptr, `u16/u32` indices, `u8/u16/u32` integer values
- **In-memory (ScxCsr)**: `i64` indptr, `i32` indices, `f32` data — matching scipy CSR for zero-copy
- **Value encodings**: uint8 (0), uint16 (1), uint32 (2), float32 (3), float16 (4)
- **Index dtype**: u16 if n_vars <= 65535, else u32

## Capabilities

### Analysis Accelerators (scx-accel)

Rust-native accelerators exposed via `pyscx.accel.*`, writing results to standard AnnData slots:

- **PCA**: Two methods auto-routed by `n_vars`:
  - *Covariance PCA* (n_vars <= 5000): sparse outer product accumulation (`X^T @ X` directly from CSR nonzeros with symmetry exploitation), eigendecomposition via `faer`. Exact results, optimal for HVG-selected data.
  - *Randomized SVD* (n_vars > 5000): streaming SpMM shard-by-shard with zero-copy `MatRef::from_row_major_slice` views. No full matrix materialization.
- **kNN**: HNSW-based approximate nearest neighbors via `instant-distance`.
- **UMAP**: SGD-based layout optimization.
- **Differential expression**: Pre-ranking Wilcoxon test — ranks all cells once per gene, reuses across groups (10x fewer sorts). `rankby_abs` parameter matches scanpy's signed-score ranking.
- **Leiden clustering**: Rust-native implementation (Traag et al. 2019) with RB configuration model. Sequential mode (default) matches C++ leidenalg convergence; parallel mode uses conflict-free graph coloring. Uses `rand_chacha` for deterministic seeding.
- **HVG**: Streaming `highly_variable_genes()` via `ShardSource` — `streaming_mean_var()` and `streaming_clip_square_sum()`, loess fit via Python `skmisc.loess`, seurat_v3 and seurat flavors. `subset=True` uses column projection (no materialization).
- **Pseudobulk**: Streaming aggregation via `BackedCsrReader`, statistical testing delegated to `pydeseq2`.
- **Perturbation evaluation metrics**: Rust-accelerated equivalents of the `cell-eval` and `arc-bench` metric pipelines (`pseudobulk_means`, `perturbation_metrics` bundling pearson_delta/mse/mae/mse_delta/mae_delta, `discrimination_score` with target-gene exclusion, `energy_distance` + `energy_distance_details` with streaming pairwise distance, `knockdown_efficiency` writing `obs["KnockDownEfficiency"]`/`obs["KnockDownGeneFC"]`, `clustering_agreement` with AMI/NMI/ARI scoring, `rank_genes_groups_df` bridge to cell-eval's DE format). Parity verified in `pyscx/tests/test_cell_eval_parity.py` (30/30 tests) within tolerances documented in `docs/scanpy.md` (perturbation evaluation metrics section). 10–20× faster than the Python references at 100K–1M cells (see `docs/performance.md`).

### GPU Acceleration (scx-gpu)

All accessible via `device="gpu"` parameter in `pyscx.accel.*`. Requires `scx-gpu` conda env with RAPIDS. See [docs/gpu-setup.md](docs/gpu-setup.md).

- **GPU PCA**: cuSPARSE SpMM + cuSOLVER QR + cuRAND.
- **GPU kNN**: cuVS CAGRA (9.4x standalone on 1M cells).
- **GPU UMAP**: Native CUDA SGD (7.7x on 1M cells).
- **GPU Leiden**: cuGraph (16x on 1M cells).
- **GPU preprocessing**: Fused normalize_total + log1p.
- **Fallback**: Graceful CPU fallback when GPU unavailable.

### Lazy Preprocessing

`ScxLazyTransformedDataset` wraps the backed reader with chained transforms (NormalizeTotal, Log1p, RowScale) — no materialization. `ShardSource` trait enables streaming PCA through transforms.

Materialization-free operations via `pyscx.accel.*`:
- `normalize_total()`, `log1p()` — lazy transforms, fused when chained
- `filter_cells()`, `filter_genes()` — non-materializing filters
- `subset_obs()`, `calculate_qc_metrics()` — streaming
- `highly_variable_genes()` — streaming HVG (seurat_v3 + seurat flavors, 99.5% overlap with scanpy on pbmc3k)
- `X[:, col_array]` — returns backed/lazy dataset with column projection

Full out-of-core pipeline: open -> QC filter -> normalize -> log1p -> HVG -> PCA -> kNN -> UMAP -> Leiden with **5.1 GB peak RSS** on 1M cells (down from 43.6 GB — **88% reduction**).

### ML Training Loader

Triple-buffered Rust pipeline (tokio I/O -> rayon decode -> Python consumer). Zero Python on the hot path — all I/O, decompression, shuffling, sparse-to-dense conversion, and normalization in compiled Rust. See [docs/api.md](docs/api.md) for `TrainingPipeline` API.

## Performance

See [docs/performance.md](docs/performance.md) for detailed benchmark data.

| Area | Metric | Result |
|------|--------|--------|
| Read | vs Zarr lz4, 1M cells | **1.38x faster** |
| Read | Parallel read scaling (32 threads) | **Up to 7x** |
| Write | Parallel write scaling (32 threads) | **Up to 3.2x** |
| Column projection | vs all competitors | **4-8x faster** |
| Memory (OOC pipeline) | Peak RSS, 1M cells | **5.1 GB** (88% reduction) |
| PCA | vs scanpy, 1M cells (HVG) | **1.9x faster** (4.2s) |
| DE (Wilcoxon) | vs scanpy, 1M cells | **3.2x faster** (5.4s) |
| Leiden | vs Python leidenalg, 1M cells | **40x faster** (55s) |
| Training loader | batches/sec, 1M cells | **1,405** (82x vs TileDB-SOMA-ML) |
| GPU kNN | vs CPU, 1M cells | **9.4x faster** |
| GPU UMAP | vs CPU, 1M cells | **7.7x faster** |
| GPU Leiden | vs CPU, 1M cells | **16x faster** |

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
- `finish()`: write full catalog at EOF -> pwrite root catalog at 256 -> pwrite header at 0 -> fsync -> rename

### Python Bindings (pyscx)

- PyO3 with `Bound<'py, T>` API (not deprecated `&PyAny`)
- Pin `pyo3` and `numpy` crate to same minor version (currently 0.23)
- `PyArray::from_vec()` for zero-copy (moves Rust Vec to numpy)
- ScxCsr `i64/i32/f32` matches scipy exactly — avoids copy
- Arrow -> pandas via pyarrow's `to_pandas()` for obs/var metadata
- Accelerators exposed via `pyscx.accel.*` — results written to standard AnnData slots
- Optional Python deps (`pydeseq2`) imported at runtime with clear `ImportError` if missing

### R Bindings (rscx)

- `extendr` v0.8.x for Rust<->R FFI
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
- HVG: streaming `streaming_mean_var()` and `streaming_clip_square_sum()` in `hvg.rs`, loess via `skmisc.loess`
- Adapted Leiden from `single-clustering` (BSD 3-Clause). Default `n_iterations=2` (matches leidenalg package default). Uses `rand_chacha` for deterministic seeding and `libc` for `malloc_trim` on Linux.
- Pseudobulk aggregation streams via `BackedCsrReader`, statistical testing delegated to `pydeseq2`

## Known Risks and Pitfalls

- **HDF5 crate (`hdf5-rust`)** is unmaintained. Only needed for conversion. Fallback: Python subprocess with h5py.
- **h5ad files are messy**: missing encoding-type attrs, CSC instead of CSR, dense X, pickled uns. Handle gracefully.
- **tokio + rayon interaction** (scx-loader): Keep tokio for I/O only, rayon for CPU work. Bounded channels for back-pressure.
- **Cloud auth**: `object_store` handles credentials via environment variables and instance metadata. No custom auth code.

## Known Limitations

- **GPU pipeline speedup**: 3.8x median achieved on 1M cells (10x target not met). Best individual: kNN 9.4x, UMAP 7.7x, Leiden 16x.
- **GPU Leiden correctness**: ARI 0.92 vs Python leidenalg (Go/No-Go: 3/4 gates pass).
- **CSC storage**: Gene-major (CscShard) not yet implemented — CSR only.
- **Multimodal**: CITE-seq, spatial transcriptomics not yet supported.
- **GDS**: GPUDirect Storage requires local NVMe + nvidia-fs drivers + ext4/XFS; always falls back to CPU path.
