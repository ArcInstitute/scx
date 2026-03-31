# SCX Implementation Roadmap

**Last updated**: 2026-03-30

## Strategy: AnnData-First, Not Scanpy-Replacement

SCX does **not** need to reimplement scanpy, scVI, Harmony, or any other
scverse analysis tool. The entire scverse ecosystem operates on AnnData
objects backed by scipy sparse matrices and pandas/Arrow DataFrames.
SCX's `to_anndata()` produces exactly this via zero-copy (SPEC.md Section
6.2), so every existing tool works unmodified:

```python
adata = scx.open("experiment.scx").to_anndata()
sc.pp.normalize_total(adata)    # scanpy, unmodified
sc.tl.pca(adata)                # scanpy, unmodified
scvi.model.SCVI.setup_anndata(adata)  # scVI, unmodified
```

**What SCX must build**: the file format, codec, I/O layer, AnnData bridge,
query engine, and ML training data loader. These are things no existing tool
provides.

**What SCX may optionally build later**: Rust-native implementations of
performance-critical operations (fused normalize+log1p, PCA, kNN) that are
faster than scanpy. These are optimizations, not prerequisites.

---

## Phase 1: Format + Codec + AnnData Bridge (Months 1-4) — COMPLETE

**Status**: All tasks complete. Go/No-Go gate passed 2026-03-17.
**Implementation plan**: [Phase1.md](Phase1.md)
**Benchmark results**: [benchmarks/results/benchmark_results.md](benchmarks/results/benchmark_results.md)

**Goal**: Build the SCX file format, compression codec, and AnnData bridge.
Validate the thesis end-to-end: `h5ad → scx convert → scx.open().to_anndata()
→ scanpy pipeline works`. Publish compression and I/O benchmarks.

### 1.1 scx-format
- [x] File header read/write (256 bytes, all fields from SPEC.md Section 3.1)
- [x] Root catalog read/write (Section 3.2)
- [x] Full catalog read/write with per-entry checksums and shard statistics
- [x] Section alignment (8-byte) and padding
- [x] Atomic rename write path (Section 3.6.1)
- [x] `mmap` read path for local files
- [x] BLAKE3 checksums (per-section and file-level)

### 1.2 scx-codec
- [x] Rice encoder/decoder (Section 4.3) with per-block adaptive k
- [x] FOR-BP encoder/decoder for indices (Section 4.2)
- [x] Delta-Golomb encoder/decoder for indptr (Section 4.1)
- [x] Codec dispatch by `codec_id` (none, scx1, zstd)
- [x] Per-shard codec override (shard header overrides file header)
- [x] Conformance test vectors: known input → exact encoded bytes
- [x] Scalar reference implementation (normative)

### 1.3 scx-sparse
- [x] `ScxCsr` struct: construct from indptr/indices/data
- [x] CSR row slicing (single row, row range)
- [x] CSR → scipy.sparse.csr_matrix zero-copy (via buffer protocol)
- [x] CSR → dense matrix conversion

### 1.4 scx-cli (minimal)
- [x] `scx convert --from h5ad input.h5ad output.scx`
- [x] `scx convert --from 10x matrix.h5 output.scx`
- [x] `scx convert --to h5ad input.scx output.h5ad`
- [x] `scx info experiment.scx` (header summary, shard count, manifest history)
- [x] `scx validate experiment.scx` (BLAKE3 verification of all sections)

### 1.5 pyscx (AnnData bridge)
- [x] `scx.open("experiment.scx")` → lazy handle
- [x] `exp.to_anndata()` → AnnData (zero-copy CSR, Arrow→pandas obs/var)
- [x] `scx.from_anndata(adata, "output.scx")`
- [x] `scx.from_10x("matrix.h5", "output.scx")`
- [x] PyO3 bindings with maturin build
- [x] Layers, obsm, obsp, uns round-trip through AnnData

### 1.6 Testing and Benchmarks
- [x] Round-trip tests: h5ad → scx → h5ad, bit-exact for integer counts
- [x] Round-trip tests: 10x h5 → scx → h5ad
- [ ] Fuzz targets for shard decoder and catalog parser *(deferred — not blocking)*
- [ ] Property-based tests: random CSR matrices survive encode/decode *(deferred)*
- [x] Verify: `scx.open().to_anndata()` produces valid AnnData that scanpy
  accepts for all standard operations (QC through DE)
- [x] Benchmark: compression ratio vs h5ad and Zarr
- [x] Benchmark: read throughput (time to `to_anndata()`) vs h5ad
- [x] Benchmark datasets: PBMC 3K, CELLxGENE Census 100K lung cells,
  Smart-seq2 50K cells

### Deliverable
`scx convert` works end-to-end. Round-trip tests pass. Users can convert
h5ad → scx, open in Python, get an AnnData, and run their existing scanpy
pipeline unmodified. Published compression and I/O benchmarks show SCX files
are smaller than h5ad with dramatically lower memory usage.

### Go/No-Go Gate — PASSED
- [x] h5ad → scx → h5ad round-trip is bit-exact for integer counts
- [x] SCX file < 60% the size of h5ad for typical UMI datasets
  - PBMC 3K: 0.477 (47.7%), Lung 100K: 0.501 (50.1%)
  - With Scx1 codec: 0.207, 0.270 respectively
- [x] `scx.open().to_anndata()` → full scanpy pipeline (QC → PCA → Leiden → DE) works

### Phase 1 Benchmark Summary

| Dataset | Cells | h5ad | SCX (None) | SCX/h5ad | SCX (Scx1) | SCX (Zstd) |
|---------|-------|------|-----------|----------|-----------|-----------|
| PBMC 3K | 2,700 | 21.5 MB | 10.3 MB | 0.477 | 4.4 MB | 5.0 MB |
| Smart-seq2 | 50,000 | 1.07 GB | 799 MB | 0.746 | 912 MB* | 370 MB |
| Lung 100K | 100,000 | 1.59 GB | 795 MB | 0.501 | 428 MB | 322 MB |

*Rice codec increases size for non-UMI data — auto-codec selection needed (Phase 2).

**Key finding**: Default codec=None gives good compression via integer dtype detection
alone (float32→uint8/uint16). With codecs enabled, ratios reach 20-35%. SCX reads are
7-23× slower than h5ad (sequential decode, no parallelism) but use 4-38× less memory
(mmap + zero-copy). See `benchmarks/results/benchmark_results.md` for full analysis.

**Lessons for Phase 2**:
1. Auto-codec selection is critical (Rice hurts non-UMI data)
2. Parallel shard decode is the highest-impact read performance fix
3. Memory efficiency (4-38× less than h5ad) is a major selling point
4. Training loader bypasses to_anndata() entirely — read perf is less relevant there

### Phase 1 Pitfalls and Risks

- **HDF5 crate (`hdf5-rust`) is unmaintained.** Last release November 2021. It wraps
  the C HDF5 library and still works, but receives no bug fixes. Mitigation: HDF5 is
  only needed for conversion; the native SCX reader is independent. If the crate breaks,
  fall back to a Python subprocess calling `h5py`, or write a minimal h5ad reader (the
  h5ad layout is relatively simple: CSR arrays + HDF5 groups).
- **h5ad files in the wild are messy.** Not all h5ad files follow the spec cleanly:
  some store X as dense arrays, some use CSC instead of CSR, some have
  `encoding-type` attributes missing, some embed pickled Python objects in `uns`.
  The converter must handle all these gracefully (convert dense→CSR, transpose CSC→CSR,
  skip unpicklable `uns` entries with warnings).
- **Compression ratio target (< 60% of h5ad) may not hold for all datasets.** The
  Rice codec's advantage is largest for typical 10x Chromium data. Smart-seq2, deeply
  sequenced, or protein-level (CITE-seq) data may not hit the 60% target. Benchmark
  on diverse datasets and be honest about where the codec helps and where it doesn't.
- **The 60% target includes only the expression matrix.** Metadata (obs/var) stored
  as Arrow IPC may be similar in size to HDF5. The compression advantage is entirely
  in the sparse matrix encoding. For metadata-heavy files (many annotation columns),
  the overall ratio may be closer to 70-80%.

---

## Phase 2: Training Loader + Query Engine (Months 4-7)

**Implementation plan**: [Phase2.md](Phase2.md), [Phase2-CLOUD.md](Phase2-CLOUD.md)

**Goal**: The ML training data loader (the primary performance thesis) and
the lazy query engine for efficient subsetting. Also: auto-codec selection
and parallel shard decode (highest-impact fixes from Phase 1 benchmarks).

### 2.0 Phase 1 Fixes (immediate, before new features)
- [x] Auto-codec selection: choose Scx1 vs Zstd based on value distribution
- [x] Parallel shard decode via rayon (address 7-23× read slowdown)
- [x] Default `from_anndata()` codec from None to "auto"

### 2.1 Training Loader
- [x] Triple-buffered Rust pipeline (Section 8.2):
  - Stage 1: tokio async I/O reads shard groups from .scx
  - Stage 2: rayon thread pool shuffles + densifies batches
  - Stage 3: pinned memory handoff to PyTorch via buffer protocol
- [x] Pipeline coordinator with back-pressure (bounded channels)
- [x] Gene projection at decode time (HVG bitmap, Section 8.4)
- [x] Sparse-to-dense direct write into pinned tensors (Section 8.5)
- [x] Quasi-random shard shuffle (Section 8.3)
- [x] scVI DataModule integration (obs covariates in batch)
- [ ] scGPT DataModule integration
- [x] Configurable memory budget (`max_loader_memory_mb`)

### 2.2 scx-engine (query)
- [x] Lazy pipeline builder: `open → filter → select → collect`
- [x] Schema validation at pipeline construction time (fail-fast)
- [x] Predicate pushdown level 1: catalog-level shard pruning via stats
- [x] Predicate pushdown level 2: predicate index lookups (Section 3.5)
- [x] Projection pushdown: skip unreferenced sections
- [x] Parallel shard processing with rayon
- [x] Result as AnnData (filtered subset)

### 2.3 scx-engine (predicate indexes)
- [x] Categorical predicate index: sorted value → shard ranges
- [x] Numeric predicate index: B+ tree for range queries
- [x] High-cardinality hash index (>10K unique values)
- [x] Auto-indexing for low-cardinality columns (<1K unique values)
- [ ] `scx-cli` flag to specify indexed columns during conversion

### 2.4 Fragment/Manifest Operations
- [x] `scx append` — append new shards + updated catalog (Section 3.6.2)
- [x] `scx delete --filter` — logical deletion via deletion vectors (Section 3.6.3)
- [x] `scx compact` — rewrite file reclaiming space (Section 3.6.4)
- [x] `scx rollback` — revert to previous manifest (Section 3.6.5)
- [x] `scx merge` — streaming merge of multiple .scx files
- [x] Advisory `flock()` for concurrent append safety
- [x] Python API: `scx.open("file.scx", mode="append")`

### 2.5 Cloud Operations (SPEC §12)
- [x] `scx cloud-optimize` — rewrite with front-of-file catalog (SPEC §12.2)
- [x] `scx explode` / `scx pack` — packed ↔ exploded directory (SPEC §12.5)
- [x] `scx pull` — streaming cloud → packed with on-the-fly repackaging (SPEC §12.8)
- [x] `scx push` — streaming packed → cloud with on-the-fly explode (SPEC §12.8)
- [x] `scx pull --filter` — selective pull with predicate pushdown (SPEC §12.8)
- [x] `object_store` integration (S3, GCS, Azure backends)
- [x] Python API: `scx.pull()`, `scx.push()`, `scx.open_cloud("gs://...")`
- [x] `CloudReader` for direct cloud reads without full download

### 2.6 Fused Operations (performance, not analysis reimplementation)
- [x] Fused normalize + log1p (single CSR row scan, Section 7.2)
- [x] HVG selection via CSR column aggregation (faster than scanpy for large data)
- [x] These run inside the SCX query pipeline; results are written into
  the AnnData so downstream scanpy operations see the expected slots

### 2.7 Benchmarks
- [x] Benchmark dataset: 10M cells, 30K genes
- [x] Training throughput: batches/sec, GPU utilization, time-to-first-batch
- [x] Compare against TileDB-SOMA-ML (latest release, recommended config)
- [x] Query engine: measure shard skip rate on filtered queries
- [x] Memory footprint validation (~330 MB per Section 8.8)
- [x] Publish reproducible benchmark scripts and results

### Deliverable
Training loader reading native `.scx` files with published throughput
benchmarks. Query engine filters and subsets data efficiently. scVI and
scGPT train end-to-end on atlas-scale SCX data.

### Go/No-Go Gate
- scVI trains on 10M-cell SCX dataset with GPU utilization >85%
- Training throughput >2x TileDB-SOMA-ML on same hardware and data
- Predicate pushdown skips >50% of shards on filtered queries

### Phase 2 Pitfalls and Risks

- **TileDB-SOMA-ML is a moving target.** The `tiledbsoma-ml` package (alpha, March
  2025) has a 4-stage pipeline with eager prefetching and DDP support. By the time
  SCX Phase 2 ships, SOMA-ML may be stable with C++ acceleration. The >2× throughput
  target must be validated against the latest release, not an old version. If SOMA-ML
  closes the gap, SCX's value proposition shifts from "faster loader" to "better
  compression + single file + operation fusion."
- **BPCells is a dark horse competitor.** Seurat v5 with BPCells demonstrates 44M-cell
  PCA on a laptop via bitpacked on-disk sparse matrices. BPCells is R-only, but if
  it gets Python bindings or inspires a Python equivalent, it could address some of
  the same performance gaps SCX targets — without requiring a new file format.
- **PyTorch DataLoader integration is tricky.** `num_workers=0` is required because
  the Rust pipeline manages its own threads (SPEC §8.7). This means the standard
  PyTorch multiprocessing prefetch doesn't apply — the Rust pipeline must provide
  equivalent or better prefetching. Users familiar with `num_workers>0` patterns
  may be confused. Document this clearly.
- **CUDA fork safety.** If a user accidentally sets `num_workers>0`, forking after
  CUDA initialization causes deadlocks or crashes. The `TrainingDataset` should
  detect this and raise a clear error rather than silently deadlocking.
- **io_uring on Linux.** Consider `tokio-uring` for Stage 1 shard reads on Linux
  instead of thread-pool async I/O. Eliminates thread overhead for I/O. Fall back
  to `tokio::fs` on macOS. This is a backend swap, not an architecture change.
- **Predicate pushdown benchmark depends on data distribution.** The ">50% shard
  skip rate" target assumes queries filter on columns with non-uniform distribution
  across shards (e.g., cell_type). For uniformly distributed columns (e.g.,
  n_counts), skip rates will be much lower. Benchmark with realistic query workloads.
- **Memory budget enforcement.** The 330 MB projection (SPEC §8.8) assumes specific
  configuration. Make `max_loader_memory_mb` a configurable parameter and auto-tune
  `shard_group_size` and `prefetch_batches` to fit within it. Users on
  memory-constrained systems (shared HPC nodes) need this.

---

## Phase 3: GPU Path + Ecosystem (Months 7-10) — PARTIALLY COMPLETE

**Goal**: GPU-accelerated I/O, GDS, R bindings, and production polish.

### 3.1 scx-gpu — IN PROGRESS
- [x] CUDA codec decoders (Rice, FOR-BP) — warp-level parallel decode
- [x] cuSPARSE CSR interop (zero-copy from decoded shards)
- [ ] GDS path: NVMe → GPU VRAM bypass via `cuFileRead()`
- [x] GPU sparse-to-dense conversion for training batches

### 3.2 scx-cli (extended) — PARTIALLY COMPLETE
- [x] `scx build-csc input.scx output.scx` — streaming transpose
- [x] `scx benchmark experiment.scx` — I/O + pipeline benchmarks
- [x] `scx subset` — extract cell/gene subsets to new file
- [x] `scx upgrade input.scx output.scx` — rewrite to latest format version (SPEC §3.9)

### 3.3 rscx (R Bindings) — COMPLETE
- [x] extendr-based R package
- [x] `scx_open()`, `to_seurat()`, `to_sce()`, `from_seurat()`, `from_sce()`
- [x] R pipe-friendly API: `scx_open() |> filter_obs() |> collect()`
- [x] Seurat v5 assay integration
- [x] SingleCellExperiment interop

### 3.4 Multimodal Support
- [ ] CITE-seq (RNA + protein): multi-feature-space layout (Section 11.1)
- [ ] Spatial transcriptomics: spatial coordinates + optional R-tree index
- [ ] `scx convert --from h5mu` (MuData format)
- [ ] Round-trip with MuData/MuOn objects

### 3.5 Quality + Polish
- [ ] Full conformance test suite with reference .scx files
- [ ] Fuzz targets in CI for all decoders and parsers
- [ ] SIMD codec optimizations (AVX2, NEON) with runtime dispatch
- [ ] Detection bitmap layer (Roaring Bitmap, Section 5)
- [ ] Documentation: API reference, tutorials, migration guide from h5ad
- [ ] Benchmark suite: automated regression testing of throughput

### Deliverable
Production-ready v1.0 release. GPU-accelerated training loader with GDS.
R bindings. Multimodal support. Full documentation.

### Go/No-Go Gate
- Format spec frozen (no breaking changes after v1.0)
- R and Python bindings both pass conformance suite
- GDS path benchmarked: >2x CPU path throughput on NVMe

### Phase 3 Pitfalls and Risks

- **GDS has strict deployment prerequisites.** GPUDirect Storage requires local NVMe
  (not network-attached), nvidia-fs drivers, and a compatible filesystem (ext4, XFS —
  NOT GPFS or Lustre). On HPC clusters, this means staging the `.scx` file to local
  NVMe scratch before using GDS. GPFS GDS support is in technical preview with
  restrictions. The CPU path (`pread()`) must always work as a fallback and should be
  the default unless GDS is explicitly requested.
- **CUDA codec decoders are hard to debug.** Warp-level parallel decode of Rice/FOR-BP
  (SPEC §4.4) is a non-trivial CUDA kernel. The scalar Rust reference is normative;
  the GPU decoder must produce bit-identical results. Invest in extensive
  cross-validation between scalar and GPU decode paths before trusting the GPU path.
- **extendr (R bindings) is less mature than PyO3.** extendr works but has fewer
  contributors and less documentation than PyO3. R's memory model (SEXP protection,
  garbage collection) interacts differently with Rust than Python's. Expect more
  edge cases in R bindings. Consider allocating extra time for R-specific issues.
- **Seurat v5 is also a moving target.** Seurat v5 introduced BPCells, layers, and
  a new assay model. The `to_seurat()` bridge must target Seurat v5's current API,
  which may change. Pin to a specific Seurat version and test against it.
- **CITE-seq protein counts break Rice codec assumptions.** ADT (Antibody-Derived Tag)
  counts have wider distributions and less sparsity than RNA UMI counts. Per-shard
  codec override (SPEC §4.5) allows Zstd fallback, but the multimodal extension
  should default to Zstd for protein modalities. Benchmark Rice vs Zstd on real
  CITE-seq data before choosing.

---

## Phase 4: Rust-Native Analysis Accelerators — 4a/4b COMPLETE

**Implementation plan**: [Phase4.md](Phase4.md), [Phase4-GPU.md](Phase4-GPU.md), [Phase4-ACC-ALL.md](Phase4-ACC-ALL.md)

**Goal**: For operations where scanpy is a bottleneck at scale, provide
faster Rust implementations. These are **optional optimizations** — the
full scverse pipeline works via AnnData from Phase 1.

### Phase 4a: Scanpy Integration — COMPLETE
- [x] Remaining backed mode aggregation ops (`var`, `max`, `min` per axis with deletion vectors)
- [x] Comparison optimization (`(X > 0).sum()` → `getnnz()` short-circuit)
- [x] Streaming preprocessing pipeline (`pyscx.preprocess`, `pyscx.save_layer`)
- [x] Chunk iterator (`pyscx.iter_chunks`)
- [x] Selective loading (`var_names`, `obs_filter`, `layers` parameters)

### Phase 4b: Rust-Native Accelerators — COMPLETE
- [x] **PCA** (randomized SVD): streaming SpMM from backed mode, `faer` for QR/SVD
- [x] **kNN graph**: HNSW via `instant-distance`, UMAP-style connectivities
- [x] **UMAP**: SGD embedding with spectral initialization
- [x] **DE (Wilcoxon)**: parallel rank-sum with rayon, in-memory + gene-chunked streaming
- [x] **Pseudobulk DE**: streaming aggregation via `BackedCsrReader` + `pydeseq2`
- [x] **Stratified DE**: per-stratum execution for both Wilcoxon and pseudobulk

### Phase 4c: GPU Accelerators — COMPLETE (benchmarked, 3/4 Go/No-Go gates pass)
- [x] GPU SpMM for PCA (cuSPARSE + cuSOLVER QR + cuRAND)
- [x] GPU kNN via CAGRA (cuVS) — 9.4× on 1M cells
- [x] GPU UMAP via native CUDA SGD kernel — 7.7× on 1M cells
- [x] GPU Leiden via cuGraph — 16× on 1M cells
- [x] Fused GPU preprocessing (normalize+log1p)
- [x] Graceful fallback to CPU when GPU unavailable

See [Phase4-GPU.md](Phase4-GPU.md) for detailed specification and benchmark results.

### Phase 4d: Eliminating Materialization — COMPLETE
- [x] Column-projected streaming aggregation (`sum`, `var`, `nnz`, `max`, `min` with gene subsets)
- [x] Lazy transform wrappers (`ScxLazyTransformedDataset` for normalize_total + log1p)
- [x] `pyscx.accel.normalize_total()` — lazy, no materialization
- [x] `pyscx.accel.log1p()` — appends to lazy transform chain
- [x] Fused NormalizeTotal+Log1p optimization (single-pass per shard)
- [x] Optional `__truediv__` / `__mul__` interception for scanpy compatibility
- [x] `ShardSource` trait + `LazyShardSource` for streaming PCA through transforms
- [x] `pyscx.accel.filter_cells()` / `filter_genes()` — non-materializing QC filters
- [x] `pyscx.accel.subset_obs()` — deletion vector construction from Python
- [x] `pyscx.accel.calculate_qc_metrics()` — streaming QC metrics

See [Phase4-ACC-ALL.md](Phase4-ACC-ALL.md) for detailed specification.

---

## What Users Get at Each Phase

| Phase | Months | User Experience |
|-------|--------|----------------|
| **1** | 1-4 | **COMPLETE.** Convert to SCX for 50-80% smaller files and 4-38× lower memory. Run scanpy/scVI/everything as usual via `to_anndata()`. Read speed is slower than h5ad in Phase 1 (no parallelism, no codec auto-select). |
| **2** | 4-7 | Auto-codec selection + parallel decode fix read performance. Fast training loader saturates GPUs. Query/filter large datasets without loading everything. Append/merge/delete without full rewrites. |
| **3** | 7-10 | GPU-accelerated I/O via GDS. R support. Multimodal (CITE-seq, spatial). Production-ready v1.0. |
| **4a** | 10-12 | **COMPLETE.** Full scanpy backed mode parity: native aggregation, comparison optimization, streaming preprocess, chunk iteration, selective loading. |
| **4b** | 12-15 | **COMPLETE.** Rust-native PCA/kNN/UMAP/DE/pseudobulk accelerators (3-10× faster at scale). |
| **4c** | 15+ | **COMPLETE (benchmarked).** GPU-accelerated PCA/kNN/UMAP/Leiden via cuSPARSE/cuVS/cuGraph. Per-op speedups: kNN 9.4×, UMAP 7.7×, Leiden 16×. 3.8× end-to-end on 1M cells. |
| **4d** | 16+ | **COMPLETE.** Eliminate materialization: lazy normalize/log1p, column-projected streaming aggregation, streaming PCA through transforms via `ShardSource` trait, non-materializing `filter_cells`/`filter_genes`. Full out-of-core pipeline from open → QC → preprocess → PCA → kNN → UMAP → Leiden with ~11 GB peak RSS at 1M cells (vs ~38 GB materialized; 71% reduction). |

---

## Dependencies and External Libraries

| Component | Rust Crate / Library | Purpose |
|-----------|---------------------|---------|
| HDF5 reading | `hdf5` (crates.io; repo: `aldanor/hdf5-rust`) | h5ad conversion |
| Arrow IPC | `arrow-rs` | Metadata read/write |
| Async I/O | `tokio` | Stage 1 of loader pipeline |
| Parallelism | `rayon` | CPU-parallel shard processing |
| Python bindings | `pyo3` + `maturin` | pyscx |
| R bindings | `extendr` | rscx |
| Checksums | `blake3` | Integrity verification |
| Roaring Bitmaps | `roaring-rs` | Deletion vectors, detection bitmap |
| Compression | `zstd` | Fallback codec for float layers |
| Cloud I/O | `object_store` | S3, GCS, Azure backends for pull/push/open |
| CUDA | `cudarc` or raw FFI | GPU codec, cuSPARSE, GDS |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Adoption barrier: new format | High | `to_anndata()` means zero workflow disruption; users keep scanpy |
| Rice codec complexity | Medium | Scalar reference first, SIMD later; Zstd fallback always available |
| GPU driver/GDS compatibility | Medium | CPU path always functional; GDS is opt-in |
| AnnData zero-copy edge cases | Medium | Extensive round-trip testing; fallback to copy for problematic dtypes |
| HDF5 crate stability | Low | Only needed for conversion; SCX native path takes over |
| Scope creep into analysis tools | Medium | AnnData bridge means users keep their existing tools; Phase 4 is optional |
| Incumbents improve faster than expected | High | If AnnData Zarr v3 + SOMA-ML close the gap, pivot: contribute codec/loader back into existing formats rather than pushing a new format |
| Compression claims don't generalize | Medium | Benchmark on diverse datasets (10x, Smart-seq2, CITE-seq, spatial) before publishing claims; be honest about where Rice underperforms |
| PyO3 breaking changes | Low | Pin pyo3 + numpy crate versions; budget time for migration if needed |
| extendr R binding immaturity | Medium | Allocate extra testing time; consider R subprocess fallback for edge cases |

---

## Competitive Landscape (as of March 2026)

Key developments to monitor that affect SCX's value proposition:

| Project | What to watch | Impact on SCX |
|---------|--------------|---------------|
| **AnnData + Zarr v3** | zarr-python 3 (released Jan 2025) with sharding + async I/O. AnnData migration to Zarr v3 as primary backend | If AnnData-on-Zarr closes the cloud access gap, SCX's HPC advantage must be larger to justify adoption |
| **TileDB-SOMA-ML** | Alpha → stable release. C++ acceleration. Performance improvements | If SOMA-ML achieves competitive throughput, SCX's training loader advantage narrows |
| **BPCells** | Bitpacked on-disk sparse matrices for Seurat v5. 44M cells on a laptop. Potential Python bindings | A Python BPCells could address similar pain points without requiring a new format |
| **rapids-singlecell** | GPU-accelerated scanpy replacements via cupy/cuml | If GPU analysis becomes mainstream via rapids, SCX's GPU path is less novel |
| **scverse governance** | Consolidation around h5ad/Zarr. Community standards for new formats | SCX may face community resistance if it's seen as fragmenting the ecosystem |
| **CELLxGENE Census** | 125M+ cells on TileDB-SOMA. Growing API adoption | Census standardization on SOMA creates network effects that SCX must overcome |

**Strategic implication**: SCX's highest-risk scenario is not that it fails technically,
but that incumbents improve fast enough to close the gaps SCX targets. The phased
roadmap mitigates this — Phase 1 validates the format thesis before committing to
the full ecosystem. If Phase 1 benchmarks show only modest improvements over improving
incumbents, the honest response is to contribute the codec and loader innovations to
existing tools (e.g., a Rice codec plugin for Zarr, a Rust training loader for
AnnData/h5ad) rather than pushing full format adoption.
