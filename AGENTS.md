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
- **[benchmarks/README.md](benchmarks/README.md)** — Practical guide to running benchmarks: SLURM job submission, dataset preparation, script reference. **Always use parallel SLURM job submission** (one job per benchmark x dataset pair) rather than sequential single-job scripts. See [Regression Gating](benchmarks/README.md#regression-gating) for the canonical local gate (`python benchmarks/comprehensive/scripts/gate_candidate.py` against `comprehensive/results/baselines/LATEST`) — covers format + accel CPU + accel GPU by default; opt out with `--no-gpu` / `--no-accel` / `--accel-only`. There is no CI-side gate (the prior `accel-gate.yml` workflow was removed because the self-hosted GPU runner queue made it unworkable). [GPU accelerator regression workflow](benchmarks/README.md#gpu-accelerator-regression-workflow) covers the routing details (which accel jobs land on which partitions).
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
            ├─> scx-accel (depends on scx-format, scx-sparse, scx-engine; PCA/kNN/UMAP/DE/Leiden/Harmony2/LISI; optional gpu dep on scx-gpu)
            ├─> scx-cli (depends on all above)
            ├─> pyscx (depends on all above)
            └─> rscx (depends on scx-format, scx-codec, scx-sparse, scx-engine, scx-ops)
```

14 workspace members total (including `scx-integration-tests`). See [docs/architecture.md](docs/architecture.md) for details.

Key isolation rules:
- `scx-loader` does NOT depend on `scx-engine` — it has its own streaming-optimized gene projection and fused ops.
- `scx-cloud` does NOT depend on `scx-loader` — they are siblings.
- `scx-accel` depends on `scx-format`, `scx-sparse`, and `scx-engine` (for `project_csr` in `diffexp.rs`) — no loader dependency.

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
- **Leiden clustering**: Rust-native implementation (Traag et al. 2019) with RB configuration model on the CPU path; cuGraph on the GPU path. Selected by `device` (`"cpu"` / `"gpu"` / `"gpu:N"` / `"auto"`); no silent cross-backend fallback, no Python `leidenalg` shim. Sequential mode (default) matches C++ leidenalg convergence; `parallel=True` uses conflict-free graph coloring (Rust-native only — ignored with `UserWarning` on the cuGraph path). Uses `rand_chacha` for deterministic seeding.
- **Harmony2 batch integration**: Clean-room Rust port of Harmony2 (Korsunsky et al. 2019) — soft k-means with diversity penalty + ridge-regression correction on PCA embeddings. Exposed as `pyscx.accel.harmony_integrate` (scanpy-compatible signature) and `rscx::scx_harmony_integrate`. GPU path accelerates distance / L2-normalize / batched scatter-subtract kernels. Mean per-PC Pearson r 0.989–0.999 vs R `harmony` v2.x on the validation fixtures (full parity limited by RNG stream divergence — see `pyscx/tests/test_harmony_validation.py`).
- **LISI**: Local Inverse Simpson Index via exact brute-force kNN + t-SNE-style Gaussian-bandwidth search + Simpson reduction. Exposed as `pyscx.accel.compute_lisi` and `rscx::scx_compute_lisi`. ~10× faster than the R `lisi` reference with mean-LISI agreement within 0.8–2.4 % on D1–D4.
- **HVG**: Streaming `highly_variable_genes()` via `ShardSource` — `streaming_mean_var()` and `streaming_clip_square_sum()`, loess fit via Python `skmisc.loess`, seurat_v3 and seurat flavors. `subset=True` uses column projection (no materialization).
- **Pseudobulk**: Streaming aggregation via `BackedCsrReader`, statistical testing delegated to `pydeseq2`.
- **Perturbation evaluation metrics**: Rust-accelerated equivalents of the `cell-eval` and `arc-bench` metric pipelines (`pseudobulk_means`, `perturbation_metrics` bundling pearson_delta/mse/mae/mse_delta/mae_delta, `discrimination_score` with target-gene exclusion, `energy_distance` + `energy_distance_details` with streaming pairwise distance — generic over `dtype ∈ {"f32" (default), "f64"}` and `backend ∈ {"auto" (default), "gemm", "scalar"}`; faer-dispatched gemm path for euclidean/cosine accumulates reductions in f64 even when input is f32, `knockdown_efficiency` writing `obs["KnockDownEfficiency"]`/`obs["KnockDownGeneFC"]`, `clustering_agreement` running native-Rust `scx_accel::neighbors::build_knn_graph` + `scx_accel::leiden` under `py.allow_threads` (no scanpy / igraph dispatch) with AMI/NMI/ARI scoring, `rank_genes_groups_df` bridge to cell-eval's DE format). Parity verified in `pyscx/tests/test_cell_eval_parity.py` (32/32 tests, including `dtype ∈ {"f32","f64"}` parametrisation on edistance) within tolerances documented in `docs/scanpy.md` (perturbation evaluation metrics section). Default `gemm + f32` path on the cell-eval-parity-perf bench is 52× faster than cell-eval at 20K × 2K × 50 perts; clustering_agreement is 12.83× at 200 perts (see `docs/performance.md`).

### GPU Acceleration (scx-gpu)

All accessible via the `device=` parameter on `pyscx.accel.*` ops. Accepted values: `"auto"` (GPU 0 if available else CPU), `"cpu"`, `"gpu"` (= `"gpu:0"`), or `"gpu:N"` to target a specific CUDA device on multi-GPU systems. Out-of-range indices and non-integer suffixes raise (`RuntimeError` / `ValueError` respectively). Requires `scx-gpu` conda env with RAPIDS. See [docs/gpu-setup.md](docs/gpu-setup.md).

- **GPU PCA**: Two dispatch paths auto-routed by `n_vars`:
  - *Covariance* (`n_vars ≤ GPU_COVARIANCE_PCA_THRESHOLD = 8000`): cuSPARSE + cuBLAS `sgemm`/`sger` builds the Gram matrix on-device; cuSOLVER `syevd` eigendecomposes; embeddings via a second streaming `matmat`. No host round-trip.
  - *Randomized SVD* (`n_vars > 8000`): cuSPARSE SpMM + cuSOLVER QR + cuRAND + cuBLAS. Fully GPU-resident final embedding (cuBLAS `sgemm` + broadcast-scale).
  - Override via `method="auto"|"covariance"|"randomized"` on `pyscx.accel.pca`.
  - Randomized path accepts `qr_method="householder"` (default, stable) or `"cholesky"` (CholeskyQR2 — `potrf` + `strsm`, ~3× faster on well-conditioned inputs; surfaces `RuntimeError` on non-SPD Gram so callers can retry with Householder).
- **GPU kNN**: cuVS CAGRA (9.4x standalone on 1M cells).
- **GPU UMAP**: Native CUDA SGD (7.7x on 1M cells).
- **GPU Leiden**: cuGraph (16x on 1M cells). Reached directly via `device="gpu"` / `"gpu:N"` (post dispatch reframe — Rust-native no longer runs first). `theta` kwarg is cuGraph-only; `parallel` kwarg is Rust-native-only; mismatched kwargs emit `UserWarning`. `gpu:N` pins the call to CUDA device `N` via `cupy.cuda.Device(N)`.
- **GPU preprocessing**: `pyscx.accel.normalize_total`, `log1p`, `highly_variable_genes` accept `device="auto"|"cpu"|"gpu"|"gpu:N"`. **Eager on GPU** — materializes `adata.X` to scipy CSR, breaking the lazy chain (emits `UserWarning`). The `normalize_total → log1p` chain fuses on GPU via an `adata.uns` marker (single fused pass over the original backed source). `log1p(device="gpu")` on an *already-materialised* scipy/dense X warns and falls back to `sc.pp.log1p` because the H→D / D→H round-trip dominates log1p's trivial math; the fast path requires the fusion marker or a backed/lazy source. HVG GPU dispatch is narrowed to single-batch seurat_v3; batched/seurat flavors fall back to CPU.
- **Async shard I/O**: `DoubleBufferedShardLoader` overlaps shard decode (worker thread) with GPU kernel launches (compute stream). Feeds the randomized / covariance PCA paths, preprocessing, and HVG.
- **Fallback**: Graceful CPU fallback when GPU unavailable (`device="auto"`).

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
| Read | vs SLAF, 1M cells | **19.3x faster** |
| Read | Parallel read scaling (32 threads) | **Up to 7x** |
| Write | Parallel write scaling (32 threads) | **Up to 3.2x** |
| Column projection | vs all competitors | **4-8x faster** |
| Memory (OOC pipeline) | Peak RSS, 1M cells | **5.1 GB** (88% reduction) |
| PCA | vs scanpy, 1M cells (HVG) | **1.9x faster** (4.2s) |
| DE (Wilcoxon) | vs scanpy, 1M cells | **3.2x faster** (5.4s) |
| Leiden | vs Python leidenalg, 1M cells | **40x faster** (55s) |
| Training loader | batches/sec, 1M cells | **1,405** (82x vs TileDB-SOMA-ML, ~340x vs SLAFDataLoader) |
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
- **SLAF runner quirks** (for future agents extending `slaf_runner.py`): `SLAFArray.get_submatrix` treats integer selectors positionally and returns string `cell_id` / `gene_id` columns that are NOT unique across Lance fragments in census-scale files — joining back to `cell_integer_id` explodes row counts. The runner goes through the internal `expression(cell_integer_id, gene_integer_id, value)` SQL table instead. `get_submatrix` also rejects numpy arrays (`_normalize_selector_indices` only accepts python list/slice/bool). `SLAFDataLoader`'s Mixture-of-Scanners prefetcher returns 0 batches on Census 10M with the default config — flag as upstream, not harness.

## Known Limitations

- **GPU pipeline speedup**: 6.9x median achieved on 1M cells post-Phase-7 (up from 3.8x pre-Phase-1; 10x target still not met). Best individual after Phase 1-7: UMAP 18.8x on census_1m (21.9x standalone), kNN 5.4x in-pipeline (4.8x standalone). PCA at 2K HVGs × 1M is 1.0x — CPU covariance PCA already ~3s, no GPU headroom; at n_vars=100K PCA reaches 1.7x. Leiden post-dispatch-reframe reaches cuGraph directly via `device="gpu"` (~16× on 1M cells); end-to-end pipeline gain from this is bounded by Leiden's share of total wall-time (~50 s of ~120 s pre-reframe). Full regression report at `benchmarks/results/phases_1_7_gpu_regression_report.md` from Phase 8 cluster run (job 2211369, 2026-04-23). See `GPU-ACC-SPEED-UP.md` Phase 8 and [benchmarks/README.md § GPU accelerator regression workflow](benchmarks/README.md#gpu-accelerator-regression-workflow).
- **GPU Leiden correctness**: ARI ≈ 0.92 vs Python leidenalg — documented behavior of `device="gpu"` (and of `device="auto"` on GPU hosts), not a regression. cuGraph uses a different refinement strategy than leidenalg. Pin `device="cpu"` to preserve label stability for downstream DE / annotation transfer.
- **Benchmark baseline scope**: the canonical gated baseline at `benchmarks/comprehensive/results/baselines/LATEST` covers format-level benchmarks (compression / read / write / parallel scaling / memory / cloud) and the `pyscx.accel.*` GPU accelerator surface via the `accel_*` modules in `benchmarks/comprehensive/benchmarks/`. The Phase-8 stopgap wrappers (`benchmarks/scripts/gpu_regression_*`, `slurm_gpu_regression*.sh`) have been deleted; use `gate_candidate.py` for accelerator regression checks. `ml_loader`, `correctness`, and `cell_eval_parity_perf` are now wired into `ALL_BENCHMARKS` with floors covering training-loader throughput, scanpy/backed/preprocessing parity, and the marquee `energy_distance_blas_f32` speedup. Remaining gaps tracked in `2026-04-29_SCX-BENCH-REVIEW.md`: SLAF floors for `ml_loader` (needs a multi-env orchestrator pass — `slafpy` lives in `scx-bench-slaf`), `avg_gpu_util_pct__gpu_train` floors (1 Hz `nvidia-smi dmon` misses sub-second epochs; needs an in-process sampler), and `census_1m` cell_eval coverage (energy_distance variants skip at n_obs >= 500K).
- **CSC storage**: Gene-major (CscShard) not yet implemented — CSR only.
- **Multimodal**: CITE-seq, spatial transcriptomics not yet supported.
- **GDS**: GPUDirect Storage requires local NVMe + nvidia-fs drivers + ext4/XFS; always falls back to CPU path.
