# SCX Implementation Roadmap

**Last updated**: 2026-09-17

## Strategy: AnnData-First, Not Scanpy-Replacement

SCX does **not** need to reimplement scanpy, scVI, Harmony, or any other
scverse analysis tool. The entire scverse ecosystem operates on AnnData
objects backed by scipy sparse matrices and pandas/Arrow DataFrames.
SCX's `to_anndata()` produces exactly this via zero-copy (see
[docs/format.md](docs/format.md) and [docs/api.md](docs/api.md)),
so every existing tool works unmodified:

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

## 1. Format + Codec + AnnData Bridge (Months 1-4) — COMPLETE

**Status**: All tasks complete. Go/No-Go gate passed 2026-03-17.
**Benchmark results**: [benchmarks/results/benchmark_results.md](benchmarks/results/benchmark_results.md)

**Goal**: Build the SCX file format, compression codec, and AnnData bridge.
Validate the thesis end-to-end: `h5ad → scx convert → scx.open().to_anndata()
→ scanpy pipeline works`. Publish compression and I/O benchmarks.

### 1.1 scx-format
- [x] File header read/write (256 bytes, all fields; see docs/format.md §File Header)
- [x] Root catalog read/write (docs/format.md §Dual Catalog)
- [x] Full catalog read/write with per-entry checksums and shard statistics
- [x] Section alignment (8-byte) and padding
- [x] Atomic rename write path (docs/format.md §Initial file creation)
- [x] `mmap` read path for local files
- [x] BLAKE3 checksums (per-section and file-level)

### 1.2 scx-codec
- [x] Rice encoder/decoder (docs/codec.md §Values) with per-block adaptive k
- [x] FOR-BP encoder/decoder for indices (docs/codec.md §Indices)
- [x] Delta-Golomb encoder/decoder for indptr (docs/codec.md §Indptr)
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
- [x] `scx convert --from h5ad input.h5ad output.scx` (parallel streaming reader; `--reader-threads N`/`--writer-queue-depth N` opt-in tunables, auto-derived from `RAYON_NUM_THREADS`/CPU count by default, byte-identical to sequential output)
- [x] `scx convert --from 10x matrix.h5 output.scx`
- [x] `scx convert --to h5ad input.scx output.h5ad` (streams by default; parallel decoder pool via `--reader-threads N`/`--writer-queue-depth N`, auto-derived from `RAYON_NUM_THREADS`/CPU count by default, byte-identical to sequential output; `--stream=false` for the legacy materialising path; `--to h5mu` and `--modality NAME` for multimodal export)
- [x] `scx info experiment.scx` (header summary, shard count, manifest history)
- [x] `scx validate experiment.scx` (BLAKE3 verification of all sections)

### 1.5 pyscx (AnnData bridge)
- [x] `scx.open("experiment.scx")` → lazy handle
- [x] `exp.to_anndata()` → AnnData (zero-copy CSR, Arrow→pandas obs/var)
- [x] `scx.from_anndata(adata, "output.scx")` (also accepts a backed AnnData over an SCX file or a `ScxLazyTransformedDataset`; streams shard-by-shard with byte-passthrough when source and target shard layouts agree)
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

### Benchmark Summary

| Dataset | Cells | h5ad | SCX (None) | SCX/h5ad | SCX (Scx1) | SCX (Zstd) |
|---------|-------|------|-----------|----------|-----------|-----------|
| PBMC 3K | 2,700 | 21.5 MB | 10.3 MB | 0.477 | 4.4 MB | 5.0 MB |
| Smart-seq2 | 50,000 | 1.07 GB | 799 MB | 0.746 | 912 MB* | 370 MB |
| Lung 100K | 100,000 | 1.59 GB | 795 MB | 0.501 | 428 MB | 322 MB |

*Rice codec increases size for non-UMI data — auto-codec selection needed.

**Key finding**: Default codec=None gives good compression via integer dtype detection
alone (float32→uint8/uint16). With codecs enabled, ratios reach 20-35%. Original SCX reads
were 7-23× slower than h5ad (sequential decode, no parallelism) but used 4-38× less memory
(mmap + zero-copy). **Sprint 2–3 resolved the read gap and then some**: SCX is now 1.15–1.38× faster
than Zarr lz4 on census-scale datasets (500K–5M cells), achieves up to 7× parallel scaling
at 32 threads, and dominates column projection (4–8× faster than all competitors).

**Lessons learned** (all addressed in Sprint 2):
1. Auto-codec selection is critical (Rice hurts non-UMI data) → added LZ4+shuffle
2. Parallel shard decode is the highest-impact read performance fix → rayon par_iter
3. Memory efficiency (4-38× less than h5ad) is a major selling point
4. Training loader bypasses to_anndata() entirely — read perf is less relevant there

### Pitfalls and Risks

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

## 2. Training Loader + Query Engine (Months 4-7)

**Goal**: The ML training data loader (the primary performance thesis) and
the lazy query engine for efficient subsetting. Also: auto-codec selection
and parallel shard decode (highest-impact fixes from earlier benchmarks).

### 2.0 Fixes from earlier work (immediate, before new features)
- [x] Auto-codec selection: choose Scx1 vs Zstd based on value distribution
- [x] Parallel shard decode via rayon (addressed 7-23× read slowdown → now 1.5× faster than competitors)
- [x] Default `from_anndata()` codec from None to "auto"

### 2.1 Training Loader
- [x] Triple-buffered Rust pipeline (see docs/architecture.md §Training Data Loader):
  - Stage 1: tokio async I/O reads shard groups from .scx
  - Stage 2: rayon thread pool shuffles + densifies batches
  - Stage 3: pinned memory handoff to PyTorch via buffer protocol
- [x] Pipeline coordinator with back-pressure (bounded channels)
- [x] Gene projection at decode time (HVG bitmap)
- [x] Sparse-to-dense direct write into pinned tensors
- [x] Quasi-random shard shuffle
- [x] Global pre-shuffle — `scx sort --shuffle --seed N` / `pyscx.shuffle`. The two-level per-epoch shuffle is bounded by physical layout (level 2 only mixes rows already sharing a shard group), so a clustered file caps batch composition at `shard_group_size`. A seeded permutation of the obs axis, applied once on disk, removes that ceiling: measured on `tabula_sapiens_100k`, per-shard label divergence from the corpus mix drops 0.562 → 0.013 while cache-cold `TrainingDataset` throughput is unchanged (1.03×). Reuses the `scx sort` engine wholesale — one new pass-0 order producer. See [`docs/sharding.md` § Shuffling for training](docs/sharding.md#shuffling-for-training-scx-sort---shuffle).
- [x] scVI DataModule integration (obs covariates in batch)
- [ ] scGPT DataModule integration — **DEFERRED**
- [x] Configurable memory budget (`max_loader_memory_mb`)
- [x] Per-cell control pairing — `pyscx.IndexPlanDataset` (sibling row-source for perturbation training, contrastive learning, donor-matched designs). Plan-driven paired-batch reads via consumer-supplied `(pert_idx, ctrl_idx)` plan iterators; built on `BackedCsrReader::read_rows_with` (zero-allocation dense gather) + `HvgProjection::scatter_row` + `fused_normalize_log1p_dense`. Memory-budget auto-tune surfaces `effective_lookahead()` / `effective_cache_shards()`. **106× faster than the cell-load-scx `ScxBackedSparseDataset` Python-loop baseline at 1M cells** (20K vs 189 cells/s). See [`docs/api.md` § IndexPlanDataset](docs/api.md#indexplandataset).

### 2.2 scx-engine (query)
- [x] Lazy pipeline builder: `open → filter → select → collect`
- [x] Schema validation at pipeline construction time (fail-fast)
- [x] Predicate pushdown level 1: catalog-level shard pruning via stats
- [x] Predicate pushdown level 2: predicate index lookups (docs/format.md §Predicate Indexes)
- [x] Projection pushdown: skip unreferenced sections
- [x] Parallel shard processing with rayon
- [x] Result as AnnData (filtered subset)

### 2.3 scx-engine (predicate indexes)
- [x] Categorical predicate index: sorted value → shard ranges
- [x] Numeric predicate index: per-shard `[min, max]` bounds driving Level-1 shard pruning (numeric operators stay residual at Level 2 — see docs/format.md §Predicate Indexes)
- [x] High-cardinality hash index (>10K unique values)
- [x] Auto-indexing for low-cardinality columns (<1K unique values)
- [x] `scx-cli` flag to specify indexed columns during conversion — `scx convert --index-obs` / `--index-var` / `--index-preset` (also on `append`/`compact`)

### 2.4 Fragment/Manifest Operations
- [x] `scx append` — append new shards + updated catalog (docs/format.md §Append)
- [x] `scx delete --filter` — logical deletion via deletion vectors (docs/format.md §Deletion vectors)
- [x] `scx compact` — rewrite file reclaiming space (docs/format.md §Compaction)
- [x] `scx rollback` — revert to previous manifest (docs/format.md §Rollback)
- [x] `scx merge` — streaming merge of multiple .scx files
- [x] In-place metadata modification (`scx set-uns` / `scx modify-metadata`, `pyscx.set_uns` / `pyscx.modify_metadata`, `scx_ops::set_uns` / `modify_metadata`) — replace `uns`/`obs`/`var`/`obsm`/`varm` without re-encoding `X`; CSC sidecar + `data_generation` preserved (docs/operations.md §Modify Metadata Complexity). Multimodal deferred
- [x] Streaming merge: obs/layers/obsm merged shard-by-shard (no full obs materialization)
- [x] `ObsMetadataShard` / `VarMetadataShard` section types (ids 24–25) for row-sharded metadata emitted by merge, append, and from_anndata when n_obs exceeds shard_target_rows
- [x] `pyscx.merge` var identity validation (index, column names, values) — `assume_identical_var=False` default (breaking change from unchecked)
- [x] `pyscx.merge` `uns_policy` kwarg (`"first"` / `"require_equal"` / `"namespace"` / `"summary"`)
- [x] `pyscx.merge` `shard_target_rows` kwarg for override
- [x] Predicate indexes built incrementally from shard stream during merge
- [x] `pyscx.from_anndata` `force_legacy_metadata` / `memory_budget` / `shard_target_rows` kwargs
- [x] `Experiment.to_anndata` `memory_budget` kwarg with `EagerAssemblyMemoryHigh` warning
- [x] Append writes new obs as `ObsMetadataShard` sections (no rewrite of existing obs)
- [x] Advisory `flock()` for concurrent append safety
- [x] Python API: `scx.open("file.scx", mode="append")`

### 2.5 Cloud Operations (docs/cloud.md)
- [x] `scx cloud-optimize` — rewrite with front-of-file catalog (docs/cloud.md (Cloud-optimized layout))
- [x] `scx explode` / `scx pack` — packed ↔ exploded directory (docs/cloud.md (Exploded layout))
- [x] `scx pull` — streaming cloud → packed with on-the-fly repackaging (docs/cloud.md (Streaming pull/push))
- [x] `scx push` — streaming packed → cloud with on-the-fly explode (docs/cloud.md (Streaming pull/push))
- [x] `scx pull --filter` — selective pull with predicate pushdown (docs/cloud.md (Streaming pull/push))
- [x] `object_store` integration (S3, GCS, Azure backends)
- [x] Python API: `scx.pull()`, `scx.push()`, `scx.open_cloud("gs://...")`
- [x] `CloudReader` for direct cloud reads without full download
- [x] Cloud reader support for sharded obs (`ObsMetadataShard` / `VarMetadataShard`) — `CloudReader::read_obs`/`read_var` assemble row-sharded metadata over parallel range reads via the shared `scx_format::assemble_sharded_metadata`, so files from streaming merge/append/from_anndata open over `open_cloud`. The single file-scope `ObsPredicateIndex` is read unchanged. Caveat: a *stale* file-scope index after `scx append --rebuild-index=false` is not auto-merged across appended shards (the incremental delta-index design remains deferred); rebuild the index or expect a full obs scan over appended rows

### 2.6 Fused Operations (performance, not analysis reimplementation)
- [x] Fused normalize + log1p (single CSR row scan)
- [x] HVG selection via CSR column aggregation (faster than scanpy for large data)
- [x] These run inside the SCX query pipeline; results are written into
  the AnnData so downstream scanpy operations see the expected slots

### 2.7 Benchmarks
- [x] Benchmark dataset: 10M cells, 30K genes
- [x] Training throughput: batches/sec, GPU utilization, time-to-first-batch
- [x] Compare against TileDB-SOMA-ML (latest release, recommended config)
- [x] Query engine: measure shard skip rate on filtered queries
- [x] Memory footprint validation (~330 MB)
- [x] Publish reproducible benchmark scripts and results

### Deliverable
Training loader reading native `.scx` files with published throughput
benchmarks. Query engine filters and subsets data efficiently. scVI and
scGPT train end-to-end on atlas-scale SCX data.

### Go/No-Go Gate
- scVI trains on 10M-cell SCX dataset with GPU utilization >85%
- Training throughput >2x TileDB-SOMA-ML on same hardware and data
- Predicate pushdown skips >50% of shards on filtered queries

### Pitfalls and Risks

- **TileDB-SOMA-ML is a moving target.** The `tiledbsoma-ml` package (alpha, March
  2025) has a 4-stage pipeline with eager prefetching and DDP support. By the time
  the training loader work ships, SOMA-ML may be stable with C++ acceleration. The >2× throughput
  target must be validated against the latest release, not an old version. If SOMA-ML
  closes the gap, SCX's value proposition shifts from "faster loader" to "better
  compression + single file + operation fusion."
- **BPCells is a dark horse competitor.** Seurat v5 with BPCells demonstrates 44M-cell
  PCA on a laptop via bitpacked on-disk sparse matrices. BPCells is R-only, but if
  it gets Python bindings or inspires a Python equivalent, it could address some of
  the same performance gaps SCX targets — without requiring a new file format.
- **PyTorch DataLoader integration is tricky.** — **RESOLVED.** `TrainingDataset` is now fork-safe under `num_workers > 0` via per-pipeline rayon pool + lazy tokio runtime construction. The PID check in `__next__` catches the eager-construct-then-fork case. See `docs/multithreading.md` § Fork safety.
- **CUDA fork safety.** — **RESOLVED.** The PID check in `TrainingDataset.__next__` raises a clear `RuntimeError` when a dataset constructed in the parent is used in a forked child. `pyscx/tests/test_fork_safety.py` is the durable regression test.
- **io_uring on Linux.** Consider `tokio-uring` for Stage 1 shard reads on Linux
  instead of thread-pool async I/O. Eliminates thread overhead for I/O. Fall back
  to `tokio::fs` on macOS. This is a backend swap, not an architecture change.
- **Predicate pushdown benchmark depends on data distribution.** The ">50% shard
  skip rate" target assumes queries filter on columns with non-uniform distribution
  across shards (e.g., cell_type). For uniformly distributed columns (e.g.,
  n_counts), skip rates will be much lower. Benchmark with realistic query workloads.
- **Memory budget enforcement.** — **RESOLVED.** `max_loader_memory_mb` is implemented and auto-tunes `shard_group_size` and `prefetch_batches` to fit within the budget. `LoaderConfig::validate()` rejects too-small budgets up front.

---

## 3. GPU Path + Ecosystem (Months 7-10) — PARTIALLY COMPLETE

**Goal**: GPU-accelerated I/O, GDS, R bindings, and production polish.

### 3.1 scx-gpu — PARTIALLY COMPLETE
- [x] CUDA codec decoders (Rice, FOR-BP) — warp-level parallel decode
- [x] cuSPARSE CSR interop (zero-copy from decoded shards)
- [ ] GDS path: NVMe → GPU VRAM bypass via `cuFileRead()` — **DEFERRED** (strict deployment prerequisites: local NVMe + nvidia-fs + ext4/XFS; CPU path remains default)
- [x] GPU sparse-to-dense (CSR→dense) scatter kernel — serves the **analysis** paths (GPU DE gene-major scatter, `to_gpu_anndata` device handoff). **Not** wired into the CPU training loader, whose decode + dense-scatter stay on host (`scx-loader` `io_stage`/`decode_stage`). Loader-side device decode (Phase D) was profiled (D0, 2026-07-09) and **deferred**: post-SIMD ShufDeltaZstd CPU decode is only ~9% over Scx1, loader back-pressure ≈ 0, and GPU util ≈ 0% — host-bounce is not the measured training ceiling. A `gpu`-gated `scx-loader → scx-gpu` feature edge (D1) exists as scaffolding.

### 3.2 scx-cli (extended) — COMPLETE
- [x] `scx build-csc input.scx output.scx` — streaming transpose. **Fully integrated** end-to-end: write-time CSC at `scx convert --csc=always` / `pyscx.from_anndata(csc="always")` / 10x / MTX import; multi-shard CSC (`--csc-cols-per-shard`, default 5000); column-major streaming via `BackedCscReader` + `ColumnShardSource` trait; consumer dispatch via `prefer_format="csc"` on HVG / DE / pseudobulk / QC / col_aggs; `scx info` per-shard layout; `scx validate` BLAKE3 coverage; mutating ops drop CSC by default with `--rebuild-csc` opt-in. See [docs/sharding.md § CSC sharding](docs/sharding.md#csc-sharding) and [docs/scanpy.md § prefer_format](docs/scanpy.md#prefer_formatautocsrcsc-column-major-dispatch).
- [x] `scx benchmark experiment.scx` — I/O + pipeline benchmarks
- [x] `scx subset` — extract cell/gene subsets to new file
- [x] `scx upgrade input.scx output.scx` — rewrite to latest format version (docs/format.md (Versioning))

### 3.3 rscx (R Bindings) — COMPLETE
- [x] extendr-based R package
- [x] `scx_open()`, `to_seurat()`, `to_sce()`, `from_seurat()`, `from_sce()`
- [x] R pipe-friendly API: `scx_open() |> filter_obs() |> collect()`
- [x] Seurat v5 assay integration
- [x] SingleCellExperiment interop
- [x] Harmony batch correction in R (see `rscx/R/harmony.R`)
- [x] Codec intent axis (`auto`/`fast`/`compact` + explicit codecs) on
  `from_seurat` / `from_sce` / `from_mae` via the shared
  `scx_format::resolve_codec` (single-modality **and** multimodal write paths),
  plus a `row_group_rows` framing knob. R defaults to v4-framed adaptive `auto`,
  matching pyscx/CLI.

**R parity — intentional gaps (defer):** the following pyscx/CLI write-side knobs
are not exposed in R and are tracked here rather than implemented:
- Streaming-convert threading (`reader_threads` / `writer_queue_depth`): R has no
  streaming-convert entry (the `from_*` importers materialize in memory).
- Grouped-write args (`group_by` / `reference`): R exposes the F2 *reads*
  (`read_group` / `read_reference` / `group_labels`) but not the write side.
- Accel `device=` selector: `rscx` accelerators are CPU-only. (Route *metadata*
  is no longer a gap: since ORG-10.16-3b every rscx accelerator with a pyscx
  route stamp records the same `scx-accel` planner record pyscx writes to
  `uns["scx_accel"]` — `object@misc$scx_accel[[op]]` on Seurat input, a
  `scx_accel` list element / attribute on matrix input — gated by the
  `accel_r_route` benchmark floors. The one exception is `scx_pseudobulk`:
  pyscx stamps no plain `"pseudobulk"` op either, deliberately.)
- `scx_append(codec=)` accepts `Auto`/explicit only (not the `fast`/`compact`
  intent axis, which is a rewrite-path concern in `scx-ops`).
- Framing granularity: `from_*` expose `row_group_rows` but hardcode
  `target_nnz = None` (no `row_group_target_nnz` byte/nnz-aware sizing knob that
  pyscx/CLI carry).
- `positional=` on `scx_attach_obs` / `scx_attach_var` (`AxisJoinKey::Positional`):
  no in-process positional consumer exists in R yet, and for external tables
  positional is exactly the footgun the key join prevents.
- `uns=` payloads on `scx_attach_obs` / `scx_attach_var`: both accept `uns_key`
  and neither sends a payload, so the argument is inert on the R side.
- `obsm` embeddings on `pyscx.attach_obs_columns`: deferred on the Python side
  too (`build_obsm` materializes at `n_obs` scale; no caller needs it).

### 3.4 Multimodal Support — SHIPPED
- [x] Format v2 bump; carve `n_modalities` /
  `modality_table_offset` / `modality_table_length` out of the header
  reserved tail; add `has_modalities` flag (bit 7); add explicit
  `col_start` / `col_end` to `ShardStats` (catalog v2); v2-strict
  CSC `shard_type=1` validation with `ScxError::InvalidShardType`.
- [x] `ModalityTable` section (id 15), `LayerCscShard`
  (id 16), per-modality reader / writer methods, `BackedCscReader`
  per-modality scoping.
- [x] Catalog `modality_id: u8` field, per-modality
  filtering helpers (`csr_shards_for_modality`, etc.), bumped
  `catalog_version = 2`. `FullCatalog::write_to` auto-upgrades
  `catalog_version` to ≥2 on serialise so the on-disk header matches
  the v2 stats layout that `ShardStats::write_to` always emits —
  closes the 2026-05-10 symmetry bug where push-side re-serialisation
  of a v1 catalog produced "v1 header / v2 stats" hybrids unreadable
  by any code path (regression test
  `catalog::tests::write_upgrades_v1_catalog_to_v2`).
- [x] CITE-seq / 10x Multiome / TEA-seq layout via
  `pyscx.from_mudata` / `to_mudata` and `scx convert --from h5mu`.
- [x] Per-modality auto-codec via
  `select_codec_for_modality` (RNA→Scx1/Zstd, ADT→Zstd, ATAC→Zstd
  or Lz4Shuffle).
- [x] `scx info` per-modality table, `scx validate`
  modality cross-checks, `scx append --modality`, `scx subset
  --modality NAME` extract.
- [x] Cloud header preservation (`cloud_optimize` /
  `pack` / `pull` rewrite the modality table at the new layout
  offset; `pull_filtered` recomputes per-modality counts from the
  filtered catalog), exploded `_modality_table.bin` + per-modality
  `X/{name}/` directories.
- [x] `pyscx.MultimodalTrainingDataset` with
  triple-buffered per-modality pipelines and aligned `cell_indices`
  validation; `TrainingDataset(path)` backward-compat warning on
  multimodal files.
- [x] rscx Seurat v5 multi-assay (`from_seurat` /
  `scx_open(...)$to_seurat()`) and Bioconductor MAE
  (`from_mae` / `$to_mae()`) interop.
- [x] Multimodal `scx merge` / `scx compact`.
  `scx-ops::merge_multimodal` concatenates per-modality CSR shards in
  input order with `row_start` adjusted for the cumulative global obs
  offset and preserves per-modality var/obsm/uns from the first
  input. `scx-ops::compact_multimodal` applies the deletion-vector
  keep mask across every modality's CSR shards and per-modality
  layers; the modality table and per-modality var/obsm/uns are
  preserved; CSC sidecars are dropped (rebuild via `--rebuild-csc`).
- [x] Per-modality CSC sidecar preservation on `scx append`.
  `scx append --modality M` invalidates `HAS_CSC` only on the target
  modality; other modalities' CSC sections are preserved verbatim.
  The file header `has_csc()` for v2 means "at least one modality
  still owns a CSC sidecar"; `scx info` shows the per-modality state.
- [x] Compose `scx subset --modality NAME` with `--filter` / `--genes`.
  `extract_modality_with_filter` reads the modality CSR, applies the
  obs predicate against the global obs, projects to the chosen genes,
  and writes a single-modality v2 SCX in one pass.
- [ ] Spatial transcriptomics R-tree index — **DEFERRED** (separate
  spec; spatial coordinates already work via standard `obs`/`obsm`).
- [x] Multimodal backed / out-of-core reads. `to_anndata`
  takes `modality=` and routes `backed=True` through
  `BackedCsrReader::for_modality`. `to_mudata(backed=True)` assembles
  a `mudata.MuData` of per-modality backed `AnnData` sharing one
  global obs DataFrame. Modality scoping flows transparently through
  the wrapper's `Arc<BackedCsrReader>`, so `pp.normalize_total →
  pp.log1p` produces a modality-scoped `ScxLazyTransformedDataset`
  with no new field on the lazy wrapper. A latent bug in
  `BackedCsrReader::read_shard_uncached` / `decode_and_cache` (X path
  dispatched by global CSR index rather than the per-instance
  filtered table) was fixed en route — required for the per-modality
  path to return the right shard. Cloud
  `open_cloud(...).to_mudata(backed=True)` and multimodal predicate
  pushdown (`var_names` / `obs_filter` with `modality=`) remain
  follow-ons.

**Status**: shipped end-to-end. CITE-seq /
10x Multiome / TEA-seq round-trip through `pyscx.from_mudata` /
`pyscx.MultimodalTrainingDataset` and Seurat v5 / MAE via rscx. See
[docs/multimodal.md](docs/multimodal.md) for the user-facing guide
and [docs/format.md § 13](docs/format.md#13-multimodal-extension-optional) for
the on-disk layout. Section ids 15 = `ModalityTable`, 16 = `LayerCscShard`, 17–25 =
embedding / sharded metadata extensions are now allocated; id 26 is
reserved (formerly `DecodeMetadataShard`, removed); ids 27–31 are reserved for further multimodal/spatial
extensions; ids 32–239 are reserved for future use; ids 240–255 are
vendor / private. The `has_modalities`
header flag (bit 7) is wired through writers and readers.

### 3.5 Quality + Polish — PARTIALLY COMPLETE
- [ ] Full conformance test suite with reference .scx files — **PARTIAL**: round-trip and per-codec correctness tests run in CI; no frozen reference-file vectors yet.
- [x] Fuzz targets for codec decoders — `scx-codec/fuzz/fuzz_targets/` covers bitstream, Rice, FOR-BP, Delta-Golomb. Scaffolded `scx-format/fuzz/` exists but has no targets yet.
- [ ] Run fuzz targets in CI on a schedule — **DEFERRED** (currently manual)
- [x] SIMD FOR-BP decode (BitPacker4x, 44% faster index decode) — done in Sprint 2
- [ ] SIMD Rice/Delta-Golomb optimizations (AVX2, NEON) with runtime dispatch — **DEFERRED** (SIMD FOR-BP already dominates end-to-end gain; Rice is <20% of decode cost)
- [x] Detection bitmap layer. Per-shard
  gene → local-row roaring bitmap sidecars (`SCXB` wire format,
  `BitmapShard` section id 6) emitted by `scx convert --bitmap
  auto|always` and consumed by `Experiment.detection_counts` /
  `cells_expressing`. Auto policy: sparse X, `n_vars ≤ 1_000_000`,
  bitmap size ≤ 15 % of encoded CSR (ATAC always-on). `has_bitmap`
  header flag and per-modality `ModalityFlags::HAS_BITMAP` are wired
  end-to-end. See [docs/format.md § 12](docs/format.md#12-detection-bitmap-optional).
- [x] Documentation: API reference, architecture, format spec, codec spec, sharding, multithreading, cloud, scanpy integration, performance, testing, GPU setup — see `docs/`
- [x] Benchmark suite: comprehensive multi-format regression harness in `benchmarks/comprehensive/` — see §5

### 3.6 External-tool obs interop — SHIPPED

Land per-cell annotations computed *outside* SCX back onto a file. Plumbing,
not algorithms: SCX reimplements none of these tools, it removes the reason to
round-trip a whole atlas through h5ad to use them. The layer-side sibling is
`cellbender-import` (`scx_ops::attach_external_layer`), which lands a corrected
count *matrix*; this is the obs-column half.

- [x] `scx_ops::attach_external_obs` — in place via
  `prepare_in_place`/`commit_in_place`, so X, layers, var, the CSC sidecar,
  `.raw`, deletion vectors and bitmaps are preserved and `scx rollback` undoes
  the whole import. Joins **by key string, never row position**; uncovered
  target rows get `null`, never a fabricated `0.0`. Predicate indexes survive a
  pure column add (`obs_index_would_go_stale` decides precisely) and drop only
  when `overwrite` rewrites an indexed column.
- [x] `scx_convert::read_annotation_table` (CSV/TSV, **ungated** — a delimited
  reader has no business needing libhdf5) and `read_h5ad_obs` (`hdf5` feature),
  behind one `read_obs_source` asserted to produce identical obs from identical
  values.
- [x] `pyscx.obs_import` / `pyscx.diagnose_obs_key` and `scx obs-import`. The
  key diagnosis is load-bearing rather than a nicety: on a real 1M-cell merged
  atlas the obs index is a 10×-duplicated stringified `RangeIndex`, no batch
  composite resolves it, and the only unique column is one no fallback list
  would guess. Composite keys (`--key sample_id,barcode`) cover the rest.
- [x] Doublet wrapper — `pyscx.doublet_import` / `scx doublet-import --tool`
  over a seven-profile table (`scdblfinder`, `scrublet`, `doubletfinder`,
  `doubletdetection`, `solo`, `scds`, `generic`), normalising each caller's
  spellings onto `<K>_score` / `<K>_predicted` / `<K>_status` + `uns["<K>"]`. A
  score-only tool gets no `<K>_predicted`: thresholding is a scientific
  decision the importer does not make.
- [x] `pyscx.attach_obs_columns` — an in-memory pandas `DataFrame` straight
  onto the file, on the same seam (key-joined by default, `positional=True`
  for frames computed row-for-row from the file's own `read_obs()` — the
  `ObsJoinKey::Positional` mode in `scx_ops`). `doublet_consensus`'s default
  write path, retiring its whole-frame `modify_metadata` route for pure adds
  (review §10.5 / ORG-10.16-2).
- [x] `rscx::scx_attach_obs` — an R `data.frame` straight onto the file, so
  scDblFinder / scds need no intermediate file in either direction.
- [x] Categorical fidelity through in-place obs edits — `attach_obs_columns`,
  `scx_attach_obs` (R factors), `obs_import`, `doublet_import`,
  `cellbender_import` and `modify_metadata(obs=…)` write a categorical column
  back as the dictionary it arrived as, `scx.categorical.ordered` stamp
  included, so `read_obs()` returns `category` with declared order, unused
  levels and `ordered` intact. Previously every one of them demoted every
  categorical obs column to plain strings on the way out. The
  sharded-obs assembler also stopped losing declared-but-unused levels on small
  files (arrow's dictionary merge pruned them once the summed per-shard
  vocabularies reached the row count) — for every categorical value type, not
  just strings — and `filter_obs(...).collect()` prunes to the surviving
  categories deterministically on both obs layouts.
- [x] `attach_var_columns` — the var-axis twin of the obs attach family
  (REC-13, pyscx 0.18): `pyscx.var_import` (a delimited table, **ungated**, or
  an `.h5ad`'s `/var` on an `hdf5` build), `pyscx.attach_var_columns` (an
  in-memory DataFrame), `pyscx.diagnose_var_key`, `scx var-import` and
  `rscx::scx_attach_var`, all over one `scx_ops::attach_external_var` seam.
  Key-joined by `var_names` by default with `positional=True` opt-in, in place
  through the same harness `append` uses, `scx rollback`-able. It replaces
  `modify_metadata(var=<whole frame>)` for the add-one-column case, which
  required reading var, joining in pandas, and being trusted with every column.
  Three properties differ from the obs twin **by decision**: whatever layout
  var arrived in it leaves in (a sharded var keeps its shard boundaries, a
  single section stays one — nothing on the ingest path creates var shards, and
  the rewrite ops collapse them, so an attach is the wrong place to change a
  layout); a stale var predicate index is **rebuilt** rather than dropped
  (var's index is one batch-mode build over `[(0, n_vars)]` and the new table
  is already in memory, so there is no reason to lose it — and there are no
  per-shard var column stats, so nothing is cleared); and there is no
  live/physical row-space dispatch, because deletion vectors are obs-only.
  A **multimodal** target is refused, as on `attach_external_layer`. Two
  defects on `attach_external_layer` — the other op that writes var — were
  fixed with it: it silently collapsed a sharded var into one section, and it
  never asked whether the *var* predicate index had gone stale. A third, on the
  source readers and reachable on obs too: a source frame's index field was
  imported as an ordinary annotation whenever the key was some other column,
  landing a column literally named `__index_level_0__`. **Rust API break in the
  same release** (`scx-ops`, no pyscx or CLI surface): `ObsJoinKey` is renamed
  `AxisJoinKey`, with `pub type ObsJoinKey = AxisJoinKey` kept so existing
  callers compile; `KeyDiagnosis` gains `axis` and renames `n_obs` to `n_rows`
  (the pyscx dict still keys `n_obs` on the obs surface and `n_vars` on the
  var one); and `AttachLayerSummary` gains `var_index_rebuilt`,
  `var_index_dropped` and `var_columns_not_carried`, since
  `cellbender_import` writes var columns and had been reporting nothing about
  what that did to the var index. A stale var index is *retired* either way —
  "rebuilt" now means a replacement section was written and "dropped" that
  every column it covered became unindexable, which the planning decision
  alone cannot predict; both are decided before any write, so `--dry-run`
  previews the real outcome. Not done: `varm` payloads on the attach — `varm` has no header
  flag, so a first-ever in-place `varm` has a lifecycle question to settle
  first. See [docs/operations.md § External var import](docs/operations.md#external-var-import).
- [x] Dictionary output from `append` / `merge` / `merge_sorted` — the last
  three writers that decoded categoricals to plain strings for the rows they
  add now write them back as the dictionaries they arrived as, declared levels,
  declared order and the `scx.categorical.ordered` stamp included. So a `merge`
  output and a legacy-layout `append` read back as `category` rather than
  `object`, a filtered `collect()` drawn entirely from appended shards carries
  its surviving categories, and an `append` onto a sharded dictionary base no
  longer leaves a Dictionary/plain shard mix for the reader to reconcile. The
  ops stay **representation-preserving** in the other direction too: a plain
  source column is written back plain, never promoted, so
  `reconcile_dictionary_representations` remains live for files older scx
  versions wrote and for a plain-source append onto a dictionary base.
  `scx_format_io::concat_metadata_batches` (upcast → widen keys → reconcile →
  share values → concat → unify → downcast, the read side's own pipeline) is
  the primitive `merge --sort-by` reuses for the one concat that spans inputs;
  the other seven sites needed no unification at all, since each batch is one
  self-contained Arrow IPC section. Two things shipped with it: `merge`'s
  obs-identity check now compares **logical** types, because a merge output's
  shards carry the minimal key width and two of them could otherwise differ on
  `Int8` vs `Int16` and refuse to merge; and `rscx` unpacks a non-string
  categorical to its plain vector instead of erroring. Known cost: an arrow
  slice of a dictionary keeps the whole values array, so each output shard
  carries the full declared vocabulary — the price of not pruning declared
  levels per shard. Not done: `scx sort --memory-budget`'s spill path
  re-encodes obs categoricals per spilled shard and rebuilds each shard's
  vocabulary from its own rows, so it still prunes declared levels and gives
  each shard a different declared list; the in-memory sort path is correct.
- [x] `pyscx.export_batches` — one h5ad per batch without materialising the
  pool, guarding **both** identities a tool and the import rely on (`obs_names`
  and the resolved key) for uniqueness *within* each batch.
- [x] `pyscx.doublet_consensus` — `majority` / `any` / `all` / `mean_rank`
  across N imported tools, null-aware throughout: a tool that never saw a cell
  does not vote, and a cell nobody voted on stays `null` rather than `False`.
- [x] `doublet_interop` in `benchmarks/comprehensive/` — runs the callers in
  their own conda envs, imports and scores them against injected truth, and
  gates the round trip. **Truth is computational injection**, which is exact but
  does not reproduce capture or ambient-RNA artifacts; it establishes that the
  plumbing did not corrupt the science, not agreement with published rates.
- [ ] Scoring against a **hashing- or genotype-labelled** dataset — the
  manifest fields exist for one to drop in; none is on hand.

### Deliverable
Production-ready v1.0 release. GPU-accelerated training loader (CPU path;
GDS deferred). R bindings. Full documentation. Multimodal deferred.

### Go/No-Go Gate
- [x] Format spec frozen (no breaking changes after v1.0) — docs/format.md is authoritative
- [x] R and Python bindings both pass functional test suites
- [ ] GDS path benchmarked: >2x CPU path throughput on NVMe — **GATE DROPPED** (GDS deferred; CPU path is the v1 shipping configuration)

### Pitfalls and Risks

- **GDS has strict deployment prerequisites.** GPUDirect Storage requires local NVMe
  (not network-attached), nvidia-fs drivers, and a compatible filesystem (ext4, XFS —
  NOT GPFS or Lustre). On HPC clusters, this means staging the `.scx` file to local
  NVMe scratch before using GDS. GPFS GDS support is in technical preview with
  restrictions. The CPU path (`pread()`) must always work as a fallback and should be
  the default unless GDS is explicitly requested.
- **CUDA codec decoders are hard to debug.** Warp-level parallel decode of Rice/FOR-BP
  (docs/codec.md (SIMD and GPU Decode)) is a non-trivial CUDA kernel. The scalar Rust reference is normative;
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
  codec override (docs/codec.md (Codec IDs)) allows Zstd fallback, but the multimodal extension
  should default to Zstd for protein modalities. Benchmark Rice vs Zstd on real
  CITE-seq data before choosing.

---

## 4. Rust-Native Analysis Accelerators — 4a/4b COMPLETE

**Goal**: For operations where scanpy is a bottleneck at scale, provide
faster Rust implementations. These are **optional optimizations** — the
full scverse pipeline works via AnnData from the format / codec / bridge work.

### 4a. Scanpy Integration — COMPLETE
- [x] Remaining backed mode aggregation ops (`var`, `max`, `min` per axis with deletion vectors)
- [x] Comparison optimization (`(X > 0).sum()` → `getnnz()` short-circuit)
- [x] Streaming preprocessing pipeline (`pyscx.preprocess`, `pyscx.save_layer`)
- [x] Chunk iterator (`pyscx.iter_chunks`) — shard-aligned on plain, layer and lazily transformed `X`
- [x] Bounded row gather (REC-1): `handle[rows]` / `handle[:]` / `Experiment.gather_rows_sparse(layer=, logical=)` assemble the result once — peak = result + the shard cache + up to `cache_shards` shards decoding in flight (a warm gather or a bulk `handle[:]` on a full cache holds at most 2 × `cache_shards` decoded shards beside the result; it was a second copy of the result). `gather_rows_sparse` defaults to logical rows since 0.17
- [x] Handle selectors and introspection (REC-7): every column selector form (`int`, `list`, `range`, `slice`, any-order ndarray, mask) on `X` / a layer returns a projected handle with no decode, and so does an ascending-unique selection on a lazy `X`; a reorder on a lazy `X`, and repeats on any handle, materialise only the projected unique columns, never the whole matrix (`to_memory()` on a projected handle assembles shard by shard); a layer keeps its wrapper; `np.asarray(handle)` raises `TypeError` instead of returning a 0-d object array; `stored_dtype` / `cache_shards` on the handles and `Experiment.value_encoding` / `is_integer` / `max_value` (+ `info()` tokens) report what is on disk without decoding. Not done: a gather-capable presentation (which would let `X[:, [3, 1, 3]]` be a handle); cloud `Experiment` getters for the three
- [x] Zero-row / zero-var files (REC-5): `pyscx.from_anndata` writes an AnnData with `n_obs == 0` or `n_vars == 0` (a 0-row frame crosses the pandas → Arrow boundary as a real 0-row batch, so declared categories and `ordered` survive; empty `object` columns and the index are stored as string), and every reader — `to_anndata()` eager and backed, `read_obs`, `query`, `to_h5ad`, `scx info` — answers `(0, n_vars)` / `(n_obs, 0)` with the schema. Layers and `raw` exist on disk only as CSR shards and are dropped from a 0-row write with a `UserWarning`; no CSC sidecar or predicate index is built over an empty matrix / a 0-row axis (whatever `csc=` / `index_*` say — silently on the in-memory path; the SCX-backed / lazy rewrite has never taken `index_*` for any row count and now raises a `UserWarning` instead of ignoring them); `merge` tolerates 0-row inputs (and two empty inputs still yield an obs section), `pyscx.append` of a 0-row source is a no-op like the CLI, `build-csc` on an empty matrix writes no sidecar (a verbatim copy, or a rewrite without the stale sidecar an older writer left). Not done: `from_mudata` still rejects `n_obs == 0`; `adata.X is None` is unhandled; `append` does not extend `obsm` on any target (pre-existing).
- [x] `read_obs()` returns logical rows (REC-6, **behaviour flip, pyscx 0.17**): `Experiment.read_obs(columns=None, *, logical=True)` and `obs_categorical(_many)(…, logical=True)` return the live rows — deletion vectors applied, so `len(read_obs()) == n_obs == len(to_anndata(backed=True).obs)` and pyscx agrees with `query().collect()`, `gather_rows_sparse` and rscx's `$obs()`; `logical=False` is the physical table (`n_obs_physical` rows). `CloudExperiment` mirrors, and its `n_obs` / `shape` / `repr` are the live count too. Pre-1.0 clean break, no `FutureWarning` cycle. So that `read_obs() → compute → land` keeps working, `attach_obs_columns(positional=True)` and `modify_metadata(obs=)` accept a frame in either row space, told apart by length (a live-length frame scatters through the keep mask: deleted rows `null`, the obs index barcode kept; `obsm` stays physical-length); `export_batches` and `doublet_consensus` read physical on purpose and are byte-identical to 0.16. `to_h5ad(obs_mask=)` and `Experiment.mark_deleted(mask)` take a mask in either row space too (a live-length mask is expanded through the keep mask; a pandas Series with a labelled index is checked for order). Not done: the module-level `pyscx.mark_deleted(path, indices)` addresses physical rows by definition
- [x] Accel API, engine half (REC-10): `rank_genes_groups(pts=True)` writes scanpy's `uns[key]["pts"]` / `["pts_rest"]` (fraction of cells with a nonzero value, `genes × groups`, every gene) from one route-independent counting pass in `scx-accel`, and `rank_genes_groups_df` appends `pct_nz_group` / `pct_nz_reference`; `groups=` restricts the *reported* groups without changing "rest" (numerically identical to scanpy's `groups=`); `corr_method` accepts only `"benjamini-hochberg"` and is recorded in `params`. `pdex_ref(groups=)` restricts the tested targets before the kernel (exact, since a target is only compared with the reference; CPU work and GPU memory shrink with the request). `pseudobulk_means` / `pseudobulk_dex` take `groupby: str | list[str]` through one coercion; `pseudobulk_means` keys several columns by the per-cell tuple (tuple group names) and gained its `accel.pyi` stub; a stub-signature guard compares every `accel.pyi` `def` against the runtime signature. Not done: kernel-level test-group restriction for Wilcoxon (the GPU per-group slabs still run for every group); Bonferroni.
- [x] `uns` `pandas.DataFrame` envelope (X6): `pyscx.from_anndata` / `set_uns` / `update_uns` / `modify_metadata(uns=)` write a `pandas.DataFrame` as a `__scx_type__: "pandas.DataFrame"` envelope (a `pandas.Index` envelope for the index, an explicit ordered `columns` list, one `ndarray` / `categorical` envelope per column), and `to_anndata` / `read_uns` / `open_cloud(...).read_uns()` rebuild it with index name, column order and per-column dtypes — ordered categoricals and their unused levels included. h5ad ingest maps an `encoding-type: "dataframe"` `uns` group onto the same envelope instead of flattening it to a dict of columns + `_index` (`FlattenedUnsDataframe` retired), and h5ad export writes the anndata dataframe group directly, keeping every column's exact dtype (an `int8` stays `int8`, a `bool` stays a plain `bool` dataset). So `rank_genes_groups(pts=True)` output — scanpy's own included — survives a write/read cycle. After an **SCX-native** round trip it still drives `sc.tl.filter_rank_genes_groups`; after an **h5ad** round trip it does not, because that needs `names`, a compound array skipped on ingest and exported as a raw subgroup (pre-existing, unrelated to this envelope). Refused on write, each naming the column: a `MultiIndex` on either axis, non-string or duplicated column names, any pandas extension dtype but `category` on a column — and on the **index**, any extension dtype at all, `CategoricalIndex` included; `bytes` elements in an object column, the index, or a categorical's categories; and a frame under `uns_format="plain"`. Export is all-or-nothing per frame: a frame h5ad cannot spell (a null in an object column, a column named like the index, a name HDF5 cannot carry as a single member, a non-string index name) is demoted whole to a raw envelope subgroup with an `uns_exported_as_raw_envelope` warning, which keeps every value whose key HDF5 can carry — dropping the column would keep less. A key HDF5 cannot carry at all is dropped from the fallback too, with `skipped_uns_key`. Ingest drops per column with `unsupported_uns_dataframe_column`, or errors under `strict_uns=true`. Not done: a zero-column frame's `columns` index type is not carried; anndata's genuinely-nullable column encodings have no lossless `uns` form and are dropped on ingest; scanpy's compound `names` / `scores` / `pvals` arrays remain skipped on h5ad ingest and exported as raw envelope subgroups (pre-existing, unrelated to this envelope).
- [x] Singlet-group guard on every path, and scanpy's "rest" (X8 / X9, **numerical change, pyscx 0.17**): `rank_genes_groups` raises scanpy's `Could not calculate statistics for groups <g> since they only contain one sample.` for any **participating** group with fewer than two cells — every level when `groups=` is omitted (the default, which was the unguarded branch and returned finite, plausible scores from a one-cell group), the named ones plus a named reference when it is given; an unused category counts as zero cells and raises too, as in scanpy, with `remove_unused_categories()` named in the message, and a missing `groupby` value is now decided by `pandas.isna` rather than by how it prints, so a non-categorical column no longer mints a phantom level out of `NaN` / `None` / `pd.NA` (which, once the guard is universal, failed the whole call) and no longer steals a group genuinely named `"nan"` / `"None"` / `""` — the same rule on `pdex_ref`, whose targets had the same problem. A group named `""` also reaches the `uns` recarray now: both structured-array builders moved to numpy's dict dtype spelling, since the list-of-tuples form silently renames an empty field to `"f0"`. And `reference="rest"` now keeps cells with no `groupby` label in the rank pool and in every group's "rest" — scanpy 1.12's `X[~mask_g]`, which `pts_rest` already used — so one `uns["rank_genes_groups"]` no longer holds a fraction-expressing table and a p-value describing different reference populations. **DE on a partially labelled `obs` column returns different numbers than ≤0.16**, and so does a fully labelled one whose groups are *named* `"nan"` or `""` — those spellings used to be read as missing, so those cells were in no group at all and are now tested like any other. Every other fully labelled input is unchanged bit for bit, and a pairwise run against a named reference is unchanged either way. The scipy / scanpy reference tables in `scx-accel` were regenerated on the unfiltered fixture, so the oracle pins scanpy's NaN-bearing behaviour rather than a pre-filtered stand-in. **Rust API break in the same release** (`scx-accel`, no pyscx or CLI surface): `diffexp::GroupPartition` loses `labelled`, `pool_pos` and `is_total()` — with the pool always `0..n_obs` they were `0..n_obs` and `group_indices`, so a caller reaching for one of them wanted a pool and would now silently get every cell; `n_labelled()` stays, since the count still means what it did. Not done: the same guard on `pdex_ref(groups=)`, which is pairwise and has no scanpy counterpart.
- [x] `to_anndata` slot filters and the `raw=` opt-out (REC-8, pyscx 0.18): `obsp=` / `varp=` / `varm=` join `layers=` / `obsm=`, with the same `None` = every key / `[]` = none / list = that subset contract and a `KeyError` on an unknown key — on the eager, backed, `preserve_slots` and `to_gpu_anndata` paths. Worth having because per-key laziness is not per-key *access*: anndata's `AlignedMappingProperty` builds an `AlignedActual` on the first `adata.obsp` read and validates every entry of the slot, so an `n_obs × n_obs` kNN graph is decoded whether or not the caller wanted it. An empty or fully-excluding list builds no lazy bridge at all, so nothing is left that could decode. `raw=True` (default) keeps today's rebuild-or-`DroppedRaw`-notice behaviour; `raw=False` does neither, which is the opt-out for workloads that never want raw and should survive the backed raw handle landing later. `to_anndata(modality=…)` refuses all four rather than accepting and ignoring them, as it already did for `var_names` / `obs_filter` / `layers` / `obsm`. Every default call is unchanged. Making the drop warning filter-aware forced a second change: `slot_has_selected` cannot tell "the caller excluded this slot" from "the caller misspelled a key", so key validation moved to the entry points and now covers **all five** slot filters on every path. `obsm=` already documented `KeyError` but enforced it only inside the eager assembler, which the obs-filtered query branch never reaches; `layers=` had no such contract at all and silently returned an empty slot for an unknown name on any path — **`to_anndata(layers=["typo"])` now raises `KeyError` rather than quietly loading nothing**. Two things fell out: the obs-filtered query path's "these slots are not loaded" warning no longer names a slot the caller excluded, and deciding whether to emit it stopped calling `read_all_obsm()` / `read_all_varm()` / `read_all_obsp()` / `read_all_varp()` — it decoded all four slots just to test emptiness, where a catalog listing answers the same question. Not done: per-modality slot filters on `to_mudata`, which exposes neither `layers=` nor `obsm=` today
- [x] Eager `var_names` projects while assembling (REC-9, pyscx 0.18): `to_anndata(var_names=[...])` without an `obs_filter` used to assemble the whole `X`, every selected layer and `.raw` at full width and then hand the lot to anndata to slice — so asking for three genes of two thousand cost *more* than reading the file (74.0 MB peak against 40.6 MB for a plain `to_anndata()`, since `var_names=` also forces layers eager). `X` and each selected layer now stream shard by shard through the same projecting reader the backed handles use, narrowed while each shard is still one shard wide: 74.0 MB → 8.9 MB on a 40-shard 20 480 × 2 000 fixture with one layer, 48.8 → 8.8 MB for `container="dense"`, and 61.6 → 20.5 MB on a raw-bearing file (raw is no longer copied twice on the way through anndata's view). The metadata half is still built full width and sliced by anndata, deliberately: `adata[:, idx]` prunes unused `var` categories and reindexes or deletes `uns["<col>_colors"]`, and reproducing that by hand is how a rewrite changes output nobody was pinning. A purely sequential assembly cost ~6× the wall time of the parallel decode it replaced, so the shard loop runs a bounded window (`rayon::current_num_threads()`, derated so the in-flight full-width shards stay under 256 MiB); with it, wall time is unchanged. The `preserve_slots=True` + `obs_filter` + `var_names` branch takes the same route. The memory-budget estimate is scaled by the selected gene fraction, so the `EagerAssemblyMemoryHigh` warning stops recommending advice the caller has already taken. **Not done:** `obs`, `obsm` and `obsp` are cell-axis members and still come back at full size and get copied — `obsp=[]` / `varp=[]` / `varm=[]` (0.18, REC-8) is the escape hatch, and the tight bound is only claimed for files without them. `.raw` is **not** gene-projected, because anndata does not slice it on that axis; `raw=False` drops it. A narrow `data_dtype=` on a file whose `value_max` exceeds 2²⁴ keeps the old full-width path, since the projected assembler is f32 and the typed reader casts from the native `u32` stream — unreachable through pyscx's own write doors, which route `X` through f32. That stand-down is specific to this *var_names-only* route: the same projection reached through `obs_filter` goes via the query engine's native decode (REC-15 below) and is exact. Row-only projection (`obs_filter` without `var_names`) on the `preserve_slots` branch is untouched.
- [x] Accel API, rest (REC-11 + `score_genes(ctrl_genes=)`, pyscx 0.18): `calculate_qc_metrics` had two implementations behind one name — a backed or lazy `X` ran the native streaming kernel, a scipy/dense `X` was handed to `sc.pp.calculate_qc_metrics` — and they wrote **different column sets**: the streaming route never produced `log1p_n_genes_by_counts`, `mean_counts`, `log1p_mean_counts` or `pct_dropout_by_counts`, and only the scanpy route could produce `pct_counts_in_top_<n>_genes`. Every kind of `X` now runs one kernel — backed, lazy, a backed *layer* handle, or an in-memory matrix copied into an owned CSR — so the schema is equal by construction rather than by agreement. Three user-visible bugs fell out with it: a scipy `X` with fewer than 500 genes raised `IndexError: Positions outside range of features.` (the delegation never set `percent_top`, so scanpy's default applied); `open(p).to_anndata()` followed by `qc_vars=["mt"]` raised `AttributeError: 'BooleanArray' object has no attribute 'nonzero'`, because SCX round-trips a boolean `var` column as pandas' nullable dtype and scipy cannot be indexed with one; and `highly_variable_genes(layer=)` on a backed file raised `ValueError: unrecognized csr_matrix constructor input`, since a backed layer is `ScxBackedLayerDataset` and the dispatch cast only matched `ScxBackedSparseDataset` (both QC and HVG now unwrap it). New kwargs: `layer=` on `calculate_qc_metrics`, and `percent_top=` implemented natively inside the existing row pass, so it costs no extra shard scan; it defaults to `None` rather than scanpy's `(50, 100, 200, 500)`, which is the default that raises on narrow files. `score_genes(ctrl_genes=)` takes the control set as an argument and skips the expression-matched sampling entirely — the exact-scanpy-parity route, and one that works on a backed `X` where `sc.tl.score_genes` raises `NotImplementedError`. The divergence it closes was understated in our own docs: measured against scanpy 1.12 at all defaults, Spearman 0.958 and a maximum difference of 11% of the score range, not "near-perfect". **Not done:** `score_genes(use_raw=True)` (waits on the backed `.raw` handle); a pure-Python mirror of scanpy's control selection. Shard-skipping under a row projection shipped separately: a projection that empties a shard now skips its decode, in the masked column kernels and — via `ShardSource::visible_shard_indices` — for every `as_shard_source()` consumer that goes through those drivers (PCA, HVG, `score_genes`, `pflog`). Row-axis kernels are excluded by design; the streaming DE kernels followed, and were the last family running their own shard loop — a one-of-five-shard row window took them from 5 decodes to 1, and from 20 to 4 at four gene chunks, since Wilcoxon's and pdex's shard walk is inner to the gene-chunk walk (the `pts` counting pass is one whole-matrix walk, so it pays `visited shards` once and moved onto the same ordered driver rather than the shared gene-chunk fill). Routing them through the drivers also gave them the bounded decode-prefetch they never had; that is a wall-clock claim expressible as a benchmark triple (`bench_csc__de_csr` / `bench_csc__pdex_ref_csr` on `tabula_sapiens_100k` run CPU DE on a backed file), so per [docs/benchmark_manifest.md](docs/benchmark_manifest.md) it waits on a manifested capture rather than being quoted from a local run. The GPU CSR staging plan learned the same skip, and the GPU DE and pseudobulk passes gained the row-coverage guards that plan-trust needs — an over-covering plan there is an out-of-bounds *device* read. **Rust API break in the same release** (`scx-accel`, no pyscx or CLI surface): `diffexp::{wilcoxon_rank_sum_streaming, pdex_ref_streaming, group_nonzero_counts_streaming}` require `ShardSource + Sync + ?Sized`, which the decode-prefetch drivers need; every in-tree source already satisfies it, but a downstream `ShardSource` built on `Rc`, `Cell` or `RefCell` stops compiling. See [docs/sharding.md](docs/sharding.md#row-projected-reads-skip-whole-shards). `rank_genes_groups(layer=)` / `pdex_ref(layer=)` also work on a backed file now: a backed layer is `ScxBackedLayerDataset` and every DE dispatch site cast the inner type only, so the kwarg raised on exactly the files it exists for — and `rank_genes_groups_df` gained the `preserve_var_order` refusal it was missing, where it had been returning gene-labelled rows against a sorted projection. Consequences worth knowing: an in-memory `X` is now copied (`nnz × 8` bytes) where handing it to scanpy copied nothing, values accumulate in f64 rather than scanpy's f32, `pct_counts_<v>` for a cell with no counts is `0.0` rather than `NaN`, and an all-false `qc_var` mask now publishes `log1p_total_counts_<v>` as `0.0` instead of omitting the column — the schema had depended on the data rather than on the call.
- [x] The query path decodes `X` at the requested dtype (REC-15, pyscx 0.18): `data_dtype=` was accepted on `PyQueryResult.to_anndata()` / `to_csr()` and rejected by the guard one line above the decode, so a caller holding a file whose surviving shards' `value_max` exceeds 2²⁴ had two options and neither was correct — be refused, or pass `allow_lossy=True` and accept f32 rounding. Asking for a dtype that represents the value exactly was reachable through the signature and unreachable in fact. Found in production on a Perturb-Sapiens atlas (`value_max = 55_963_804`), where a downstream cell-index search caught the `ValueError` and reported "no cells matched". The cause was not the guard: `QueryResult.x` is an `ScxCsr`, whose `data` is `Vec<f32>`, so the values had already rounded by the time any dtype was named — making the guard dtype-aware on its own would have returned f32-rounded values labelled `float64`. `QueryPipeline::collect_typed` decodes each shard to its native stream instead (integer encodings stay `u32`) and narrows once, and the dtype is declared where the decode happens: `query().collect(data_dtype="float64")`, or `to_anndata(obs_filter=…, data_dtype=…)`, which resolves the plan before it collects and so needs no new kwarg. A dtype named after a plain `collect()` is a cast of already-decoded values and fails loud, naming `collect()`. `uint32` / `int64` / `float64` are exact; `float16` / `uint16` / the `float32` default still refuse; `allow_lossy=True` still overrides. Row filtering and gene projection are **fused** into one native kernel — the f32 path builds a whole projected shard that the row filter then re-allocates a subset of — while the merge scan itself stays single-source (`project_csr_row` is generic over the index and value element types). The cloud reader implements the native decode too, so `open_cloud(url).query()` gets the same lossless read. **Declared Rust API change** (`scx-engine`): `QueryResult` is now `QueryResult<X = ScxCsr>` — defaulted, so every existing caller compiles untouched — and `SectionReader` gains a required `read_shard_from_entry_native`; a downstream `SectionReader` impl must add it, deliberately, since a default would have to round through f32. **Not done:** `with_normalize()` / `with_log1p()` with a non-f32 dtype is *refused* rather than served, because those replace the counts with floats and an internal fallback would change which guard ran under the caller's feet; the eager `var_names`-only projection (no `obs_filter`) still assembles f32, so its `>2²⁴` stand-down above is unchanged; eager `layers` still cast post-assembly, awaiting the typed layer reader; and `rscx`'s query path stays f32, having no dtype concept. The load-bearing test uses an **odd** value above 2²⁴ — an even one such as `20_000_000` is f32-exact and passes whether or not the decode is typed, which is why every Python fixture (all write doors cast `X` through f32) needs a Rust companion.
- [x] The eager-layers decode-loss guard folds over the **selected** layers (pyscx 0.18): `layers=` has been a filter since before 0.17, but all three eager-layer guard sites folded the catalog's `value_max` over *every* layer in the file, so a `> 2²⁴` count in a layer the caller never asked for refused a read that does not touch it — `to_anndata(layers=["narrow"])` raising because a sibling `wide` layer holds 20,000,000. Same class as REC-15 above, and reproduced at each of the three sites (the bridge branch, the `var_names=` projected assemble, and the post-assembly narrow a non-default `data_dtype=` triggers); all three now scope the same way, so none of them disagrees about when a read raises. Naming the wide layer, naming both, and an unfiltered `eager=True` read all still refuse. Underneath sat a second defect with a wider blast radius: `FullCatalog::layer_csr_max_value(m, Some(name))` resolved shards through a needle matching only the per-modality `layer/{modality}/{layer}/shard_{idx}` naming, never the single-modality legacy `{layer}_shard_{idx}` that `layer_names()` parses and every non-multimodal writer emits — so the name-scoped fold answered `0` on every single-modality file, and `0` is also what "this layer stores floats, nothing to guard" looks like. The one pre-existing caller of that form was **rscx's `$layer()` decode-loss guard, which was therefore dead**: measured by reverting only the catalog change, `exp$layer("wide")` on a file whose layer holds 20,000,000 returned the matrix and now raises (`allow_lossy = TRUE` is the opt-out) — a **behaviour change for rscx in this release**, and the direction `docs/api.md` already claimed. `pyscx`'s `decode_window` had the same defect one function over, so a legacy-named layer listed no shards, could not be sized, and its parallel decode window collapsed to 1 — silently serializing the bounded layer decode the `var_names` projection work added. One predicate (`layer_entry_is_named`) now backs both the fold and the shard list so a guard and the read it guards cannot drift, and the fold's `None` arm is the cheap whole-modality scan: naming every layer instead costs a substring search per layer per entry, measured at 1.3 µs against 100.8 µs on an atlas-shaped catalog (1240 entries, 5 layers), so the branch lives in the catalog where a caller cannot forget it. Three review rounds; the loop ended on its cap. **Not done:** the typed layer reader, so a `> 2²⁴` layer still cannot be delivered exactly at any dtype — the guard remains the honest gate there
- [x] The eager-assembly warning sizes something, and the int64 upcast is skipped (pyscx 0.19): `EagerAssemblyMemoryHigh` divided by 1024³ and printed "GB", covered `X` only, and priced every nonzero at a flat 16 B. Measured on a 960,195 × 6,143 file with 2,650,704,199 nonzeros: it quoted ~39.7 GiB for a process that peaked at **84.76 GiB** — the estimate is accurate for *assembly* (measured transient 43.21 GiB against a computed 39.5), but an operator reads it as a job size, requests ~48 GB and is OOM-killed, because scanpy's per-group copies (+44.5 GiB over steady, in one `_basic_stats` call) and the write-back come afterwards and cannot be in it. The message now prints GiB, says it covers assembly only and is a floor on process RSS rather than a job size, names the nonzero count and the per-nonzero cost, names the int64 promotion when it applies (the larger half of a default-f32 wide matrix's footprint, and nothing surfaced it), and offers `container="dense"` **only** when dense is genuinely smaller — never switching container automatically, since callers branch on `issparse(adata.X)`. Underneath: the estimate prices the plan the read will use rather than the default — `container="dense"` is `n_obs × n_vars × value_width` with no index array (pricing a dense request from `nnz` let a sparse matrix allocate tens of GiB without tripping the budget), the value width comes from `data_dtype`, and the index width is scipy's rather than the caller's (4 B below `i32::MAX` nonzeros, 8 above). `to_gpu_anndata`'s device path assembles no host `X` and is no longer charged for one. `adata.raw` is counted when the read will assemble it (it was invisible, so a raw-bearing file was under-counted by raw's whole footprint), and each matrix decides its index width separately because scipy decides per `csr_matrix`. **Behaviour change:** a file between roughly 0.5 and 1 billion nonzeros no longer trips the default 8 GiB budget — the flat 16 was 2× conservative for every matrix that never paid an upcast. And the default eager read now **decodes int64 column indices directly** above `i32::MAX` nonzeros instead of letting scipy copy the int32 array it was handed while that array is still alive: identical scipy object, no transient (16.3 → 11.9 B/nnz measured). The gate matters in both directions — at or below the line scipy *downcasts* int64 inputs, so widening a matrix that does not need it would add a copy rather than remove one, and it is off entirely on a file with **deletion vectors**, whose catalog nnz is only an upper bound on what scipy is handed (those are applied after assembly, so a matrix physically over the line but logically under it would pay an oversized index buffer *and* the downcast); `to_anndata(index_dtype="int64")` has always been a no-op for the returned CSR on a file that fits in int32, and the pre-existing test said so with an `or` clause that also accepted int32 for the int64 arm, i.e. passed whether or not anything widened. An explicit `index_dtype=` is never overridden; `SCX_EAGER_INT64_NNZ_THRESHOLD` lowers the threshold so a small fixture can exercise the widened decode. That route was the serial one — `assemble_shards_typed` filled its buffers in a `for` loop while the `f32` assembler used rayon — so it was parallelized first, on the same `RowMajorStrategy` and the same carve-up (`IndexBuffer`/`ValueBuffer` gained `chunks_mut`, which resolves the runtime dtype for the carve and hands each shard a typed slice); every existing `data_dtype=` / `index_dtype=` caller gets that too. **Declared Rust API change** (`scx-convert`): `ConvertWarning::EagerAssemblyMemoryHigh` gains `nnz` / `value_bytes` / `index_bytes` / `dense_shape` / `dense_bytes`, so every figure the message prints is derived from the plan the estimate was priced with and the shape it used. **Not done:** eagerly-materialized `layers` are still uncounted in the estimate (they assemble `f32` and cast afterwards, and which are eager depends on a filter the estimator cannot see), and the warning still cannot report a peak including a downstream consumer — there is no typical consumer, and a wrong second number is worse than one honest one
- [x] Selective loading (`var_names`, `obs_filter`, `layers` parameters)

### 4b. Rust-Native Accelerators — COMPLETE
- [x] **PCA** (randomized SVD): streaming SpMM from backed mode, `faer` for QR/SVD
- [x] **kNN graph**: HNSW via `instant-distance`, UMAP-style connectivities
- [x] **UMAP**: SGD embedding with spectral initialization
- [x] **DE (Wilcoxon rank-sum)**: parallel rank-sum with rayon, in-memory + gene-chunked streaming
- [x] **Pseudobulk DE**: streaming aggregation via `BackedCsrReader` + the
      Rust-native NB-GLM (default) or `pydeseq2` (`backend="pydeseq2"`)
- [x] **Stratified DE**: per-stratum execution for both Wilcoxon rank-sum and pseudobulk

### 4c. GPU Accelerators — COMPLETE (rapids-singlecell transition done)
- [x] GPU PCA — in-VRAM routes to `rsc.pp.pca`; streaming/randomized PCA survives natively (>VRAM moat)
- [x] GPU kNN — in-VRAM routes to `rsc.pp.neighbors`; device-resident CAGRA (cuVS) retained in fused pipeline only. Standalone native kNN CAGRA dispatch removed in Phase 3
- [x] GPU UMAP — routes to `rsc.tl.umap`. Native CUDA SGD kernel (`umap_sgd.cu`, `umap_edges.cu`, `gpu_umap.rs`, `fuzzy_simplicial_set.cu`, `gpu_fuzzy.rs`) removed in Phase 3
- [x] GPU Leiden via cuGraph — 16× on 1M cells (native, unchanged)
- [x] GPU preprocessing — routes to `rsc.pp.normalize_total` / `rsc.pp.log1p`; native ML-loader + streaming kernels survive
- [x] GPU HVG — `seurat_v3` stays native (1.2× at 1M); extra flavors route to `rsc.pp.highly_variable_genes`
- [x] Fused GPU pipelines — `pca_neighbors_umap` and `pca_neighbors` re-pointed to full rapids pipeline
- [x] `to_gpu_anndata()` — minimal-copy device handoff returning GPU-resident AnnData with `cupyx.scipy.sparse.csr_matrix` X
- [x] Graceful fallback to CPU when GPU unavailable or rapids absent (`FallbackReason::NoRapids`)
- [x] `SCX_FORCE_NATIVE_GPU=1` pins surviving native paths; `SCX_DISABLE_RAPIDS=1` forces CPU fallback for testing

See [docs/gpu-setup.md](docs/gpu-setup.md) and [docs/performance.md](docs/performance.md) for setup and benchmark results.

### 4d. Eliminating Materialization — COMPLETE
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

### 4e. Rapids-singlecell GPU Compute Transition

In-VRAM GPU analysis ops now route to [rapids-singlecell](https://github.com/scverse/rapids_singlecell)
for PCA, kNN, UMAP, preprocessing, and HVG (extra flavors). Native GPU code
is retained only where it has a structural or >VRAM moat. rapids-singlecell
is a detected runtime dependency (not a pip extra); install story is in
`docs/gpu-setup.md` and `envs/scx-gpu-analysis.yml`.

- ✅ **Phase 0 — Evidence**: >VRAM benchmarks, transfer/decode profiling, cuPy min-copy PoC, packaging spike
- ✅ **Phase 1 — Routing**: rapids routes added (`AccelRoute::RapidsSinglecell`), `to_gpu_anndata()`, fused pipeline re-point, `FallbackReason::NoRapids`, `pyscx/src/accel/rapids.rs` runtime probe, one-shot `no_rapids` `UserWarning`
- ✅ **Phase 2 — Gate**: cross-tier rapids route + correctness gates (`*_route_rapids_correct`), no-rapids fallback gates (`*_fallback_no_rapids_correct`), baseline promoted to `v0.6.5-accel-gpu-rapids-floors`
- ✅ **Phase 3 — Removal**: deleted native UMAP (`umap_sgd.cu`, `umap_edges.cu`, `gpu_umap.rs`, `fuzzy_simplicial_set.cu`, `gpu_fuzzy.rs`), covariance PCA (`gpu_pca_covariance.rs`), standalone kNN CAGRA dispatch, PCA SpMM graph-capture (`pca_spmm_capture_opt_in`). Dead-kernel audit complete; FOR-BP kept
- ✅ **Phase 4 — Format**: decode-metadata sidecar (implemented in F5, then removed — superseded by the codec-agnostic row-group `BlockIndex` framing), sorted+deduped CSR invariant, decode→device fast path

**What survives natively**: streaming/randomized PCA (>VRAM moat), HVG `seurat_v3` (1.2× at 1M), Leiden (Rust-native CPU + cuGraph GPU), DE Wilcoxon rank-sum/pdex (CSC/CSR-direct, structural moat), Harmony, preprocessing kernels (ML loader + streaming), device-resident CAGRA kNN (fused pipeline only), codec decode (`rice_decode.cu`, `forbp_decode.cu`), `colmajor_ops.cu`, `gpu_graph.rs`, `shard_decode.rs`.

---

## 5. Comprehensive Benchmarking + Cloud Validation

**Goal**: Establish a reproducible, multi-format benchmark harness that
validates SCX's performance claims on representative single-cell workloads
across local HPC (Chimera) and cloud (GCP) environments, then keep it
running as a regression gate.

**Why this is its own workstream**: §1–§4 built features and shipped
ad-hoc benchmarks per feature. This effort consolidates everything into a single
harness, adds cross-format competitors (h5ad, Zarr v3, TileDB-SOMA, BPCells,
Parquet, [SLAF](https://github.com/slaf-project/slaf)), and extends
coverage to cloud object storage. This is what the public performance
claims in `docs/performance.md` cite.

**Location**: [`benchmarks/comprehensive/`](benchmarks/comprehensive/) (the
gated suite). [`benchmarks/scripts/`](benchmarks/scripts/) holds dataset
prep, the standalone ML training loader, and active-development GPU /
Harmony benches; the legacy wrappers and one-off
`benchmark_*.py` entrypoints have been deleted.
Practical operator guide: [`benchmarks/README.md`](benchmarks/README.md).

### 5.1 Benchmark Harness Infrastructure — COMPLETE
- [x] `FormatRunner` abstraction — one runner per format (SCX, h5ad, Zarr v3, TileDB-SOMA, BPCells, Parquet)
- [x] **SLAF `FormatRunner`** ([slaf-project/slaf](https://github.com/slaf-project/slaf), `slafdb` on PyPI) — SQL-native sparse lazy format with Scanpy-compatible API and PyTorch dataloaders. Direct competitor across compression, selective read, lazy AnnData ops, and ML loader dimensions. Lives at `benchmarks/comprehensive/runners/slaf_runner.py` with isolated `scx-bench-slaf` env; back-end goes through SLAF's `expression(cell_integer_id, gene_integer_id, value)` SQL table (SLAF's public `get_submatrix` API treats integer lists positionally and its `cell_id` column is non-unique across Lance fragments in census-scale files).
- [x] Shared conversion cache: convert each dataset once per format, reuse across benchmarks
- [x] Deterministic `BenchmarkResult` schema → one JSON per (benchmark × format × dataset)
- [x] System info collector (CPU, RAM, OS, disk, driver, CUDA version) recorded with every run
- [x] Conda environments pinned: `scx-bench.yml` (CPU), `scx-bench-gpu.yml` (CUDA + RAPIDS), `scx-bench-r.yml` (BPCells / Seurat)
- [x] Parallel SLURM submission via submitit — one job per (benchmark × format × dataset) pair (see `benchmarks/README.md`; always prefer parallel submission over sequential)
- [x] Report generation: markdown + matplotlib plots (`benchmarks/comprehensive/reporting/`)

### 5.2 Benchmark Dimensions — COMPLETE
- [x] **Compression** — file size vs h5ad/Zarr/TileDB/BPCells/Parquet across codec variants (SCX auto/none/scx1/zstd/pcodec/lz4)
- [x] Add SLAF to compression comparison — SLAF stores cells as rows in DuckDB-queryable tables; measure on-disk size for identical datasets. Census 1M: SLAF 4.0 GB vs SCX 2.35 GB (SCX ~1.7× smaller); see `benchmarks/comprehensive/results/raw/compression__slaf__census_1m.json`.
- [x] **Read (full)** — `to_anndata()` wall-clock, peak RSS, mmap resident
- [x] **Read (selective)** — column/row projection, predicate pushdown, shard skip rate
- [x] Add SLAF selective-read comparison — SLAF exposes SQL `SELECT ... WHERE` predicates; compare against SCX catalog-level pushdown on the same query. SLAF SQL pushdown on Census 1M: `cell_type == 'T cell'` 10.5s, random 1% sample 9.2s. Canonical predicate set lives at `benchmarks/comprehensive/queries.py`; runners declare the capability via `FormatRunner.capabilities`.
- [x] **Write** — end-to-end write throughput with each codec
- [x] **Parallel scaling** — read/write throughput as a function of thread count (1 → 32 threads)
- [x] **ML loader** — batches/sec, time-to-first-batch, GPU utilization; SCX vs TileDB-SOMA-ML
- [x] Extend ML loader comparison to include **SLAF's PyTorch tokenizer/dataloader** — direct head-to-head for foundation-model training workloads on the same dataset + model. Census 1M: SLAF 4.1 batches/s (~340× slower than SCX); on Census 10M SLAF's Mixture-of-Scanners prefetcher returns 0 batches with the default config (90 s TTFB then timeout).
- [x] **Correctness** — round-trip parity of X / obs / var / obsm / obsp / uns vs source h5ad
- [x] Extend correctness to SLAF round-trip (h5ad → SLAF → AnnData vs h5ad → SCX → AnnData) — `benchmarks/comprehensive/scripts/validate_slaf_equivalence.py`, wired into the `correctness` benchmark module. Lossy fields (currently `uns`) are documented in a per-runner allowlist.
- [x] **Memory** — peak RSS for the out-of-core pipeline (open → QC → preprocess → PCA → kNN → UMAP → Leiden)
- [x] Run the same out-of-core pipeline on SLAF's lazy Scanpy-compatible API to compare peak RSS end-to-end — aggregated table at `benchmarks/comprehensive/results/reports/phase5A_ooc_rss.md`. Census 1M `read_full` peak RSS: SCX 345 MB vs SLAF 34.7 GB (SLAF's `LazyAnnData.compute()` builds the CSR through Polars fragment processors).
- [x] **Perturbation / cell-eval parity** — Rust accelerator vs Python reference speed and tolerance (see `cell_eval_parity_perf.py`)
- [x] **Append / delete / compact / rollback** — throughput and correctness across fragment-manifest operations. `benchmarks/comprehensive/benchmarks/fragment_ops.py` runs all four ops (SCX-only, gated on `scx_auto`) with per-op throughput metrics. Smoke on pbmc3k: append ~38 MB/s, delete ~155 k rows/s (logical — independent of n_obs), compact ~54 MB/s, rollback ~3 ms. Reported in §8b of the benchmark report.

### 5.3 Datasets
- [x] PBMC 3K (small reference, for correctness + fast iteration)
- [x] Tabula Sapiens 100K (medium, used for cloud benchmarks)
- [x] Lung 100K (medium)
- [x] CELLxGENE Census 500K, 1M, 5M subsets (large)
- [x] 10M-cell synthetic build for training loader (`build_census_*.py`)
- [x] Smart-seq2 50K (non-UMI protocol — validates codec selection heuristic)
- [ ] CITE-seq reference dataset — multimodal support (§3.4) landed (`pyscx.from_mudata` + `MultimodalTrainingDataset` ship) and multimodal compression benchmarks are live (`benchmarks/comprehensive/benchmarks/multimodal_compression.py`). Still TODO: extend the comprehensive suite with multimodal training loader throughput rows.

### 5.4 Local HPC Benchmarking (Chimera SLURM) — COMPLETE
- [x] Parallel SLURM submission via `benchmarks/scripts/submit_benchmarks.py`
- [x] GPU benchmarks on H100 / A100 nodes (PCA, kNN, UMAP, Leiden, fused preprocess)
- [x] R / BPCells benchmarks via isolated conda env (`scx-bench-r.yml`)
- [x] Published benchmark report: `benchmarks/comprehensive/reporting/phase3_report.md`
- [x] Results snapshot archived (`benchmarks/comprehensive/results/raw_archive_*/`)

### 5.5 Cloud Benchmarking (GCP) — COMPLETE (S3/Azure deferred)

Scope: validate SCX's cloud story (pull / push / selective pull / CloudReader /
cloud-optimize / exploded `.scxd`) on a real object store, and confirm that
the cost and latency models in `docs/cloud.md` hold end-to-end.

**Reference**: [`docs/cloud.md`](docs/cloud.md) for operator guidance on layouts, auth,
and tuning knobs. The full harness now lives under
[`benchmarks/comprehensive/benchmarks/cloud_*.py`](benchmarks/comprehensive/benchmarks/) with
one-time fixture staging via [`benchmarks/comprehensive/scripts/setup_cloud_test_data.sh`](benchmarks/comprehensive/scripts/setup_cloud_test_data.sh).
The legacy `benchmarks/scripts/benchmark_cloud.py` shim has been deleted.

- [x] GCS bucket + service-account setup (`setup_cloud_test_data.sh`) + `check_gcp_auth.py` preflight (env var + JSON identity + healthcheck round-trip)
- [x] `scx push` — local `.scx` → `gs://…/.scxd/` throughput (`cloud_push.py`)
- [x] `scx pull` — full-dataset download + on-the-fly pack throughput (`cloud_pull.py`)
- [x] `scx pull --filter` — selective pull latency, shard skip rate, bytes saved (via `cloud_filtered.py` + `cloud_reader_vs_pull.py` at 5%/20%/80% selectivity)
- [x] `scx cloud-optimize` — rewrite overhead (time, size delta)
- [x] `scx explode` + `scx pack` round-trip correctness
- [x] Streaming `scx pull` vs naive `gsutil cp` + `scx pack` wall-clock comparison
- [x] `scx info` / metadata-only latency on cloud-ready `.scx` (`cloud_metadata.py`)
- [x] **Promote cloud benchmarks into `benchmarks/comprehensive/`** — first-class `FormatRunner` cloud dimension. Every format is now compared end-to-end on GCS through the same orchestrator + JSON schema + reporting pipeline as the local benchmarks.
- [x] **GCP compute-node matrix** — `benchmarks/comprehensive/scripts/submit_gcp_matrix.py` provisions n2-standard-8 / c3-standard-8 / a3-highgpu-1g VMs in the bucket region, runs the cloud suite with `SCX_BENCH_GCP_INSTANCE` env-stamped on every result, collects JSONs back, tears down. `--dry-run` is safe; `--yes-spend` required for live execution.
- [x] **Cloud competitor parity** — Zarr v3 (consolidated metadata + `anndata_zarr_backed` variant), TileDB-SOMA (native GCS + `AxisQuery(value_filter=…)`), and SLAF (cloud-backed DuckDB). `cloud_filtered.py` iterates the canonical predicate set and pivots `(dataset, query, format)` with median + p95 over ≥3 reps.
- [x] **CloudReader vs pull** — `cloud_reader_vs_pull.py` compares `pyscx.open_cloud` (metadata-only) vs full `pyscx.pull`, and `pyscx.pull(filter=…)` vs full `pull` at ~5%/20%/80% selectivity with bytes-downloaded + GET-count proxies.
- [x] **Request-cost accounting** — `GCS_PRICING` pinned in `config.py`; `CloudIOCounters` helper in `cloud_fixtures.py`; `cost_model.py` records `usd_per_million_cells_queried` per `(layout, scenario)`. Reporting pivots as "Cost Model (GCS pricing)".
- [ ] **S3 + Azure parity** — DEFERRED indefinitely. GCP is the only validated cloud target. The `provider="gcs"` ValueError gate across every cloud benchmark module keeps the provider-parameterization in place for a future un-defer without rewrites.
- [x] **Resumable pull regression test** — idempotent-retry contract documented in `docs/cloud.md`; `CloudError::Interrupted` variant + `cleanup_stale_tmp_files` orphan sweep in `scx-cloud/src/pull.rs`. Three Rust unit tests + four Python tests at `pyscx/tests/test_resumable_pull.py`.
- [x] **Large atlas (50 GB+) streaming pull** — `cloud_large_atlas.py` runs `pyscx.pull` with a background RSS sampler (100 ms cadence); asserts `peak_rss_mb <= 240` per the `docs/cloud.md` performance model. Fails loudly on violation.

### 5.6 Regression Gating — COMPLETE (on-demand, not scheduled)
- [x] On-demand gate runs: `scripts/gate_candidate.py` one-shot captures a candidate snapshot and runs the gate against `results/baselines/LATEST`. Recommended trigger points (pre-PR / pre-merge / pre-release / on-suspicion) documented in `benchmarks/README.md`. Cron-based scheduling was explicitly deferred — no wasted compute when nothing changed, every gate result ties to a specific commit.
- [x] JSON result diff against the canonical baseline via `compare_against_baseline.py --gate` (3% wall / 10% RSS / 1% size tolerances; absolute floors from `thresholds.yaml`; disappearing-benchmark detection; justification-markdown suppression with expiry dates).
- [x] Rolling performance dashboard: `reporting/dashboard.py` emits `BENCHMARK_REPORT.html` alongside the markdown; `dashboard_history.json` threads "← previous snapshot" navigation. `publish_dashboard.py` rsyncs to a configurable static-hosting target (no-op when unset).
- [x] **Variance-aware per-row timing tolerance** — when the baseline's `summary.json` carries `wall_s_iqr` per row, the gate widens the row's effective tolerance to `max(--timing-tolerance, --iqr-k * baseline_iqr / baseline_median)`. Replaces the prior flat 3% floor on rows whose intrinsic CV is higher than the floor (avoids false positives). Coverage and fallback counts surface in the markdown report header (`Timing IQR coverage: X/Y rows have wall_s_iqr; Z fall back to fixed P% tolerance`). Captured automatically when `capture_baseline.py` sees `n_runs >= 3`; `n_runs < 3` rows fall back to the fixed tolerance by design (IQR is unreliable on 2 samples).
- [x] **Justification suppression covers floors** — `(benchmark, format, dataset)` entries in `results/justifications/*.md` suppress both `is_regression` deltas and `absolute_floor` violations on the same triple (previously only filtered regressions; floors silently failed the gate even when the triple was justified). Floors marked `| suppressed |` in the markdown report rather than dropped, so operators see what was waived. Deferred floors documented in `thresholds.yaml` § "Deferred floors".
- [x] **TIMEOUT-handling robustness** — `run_parallel.py`'s wait loop falls back to `sacct -X -j <id>` when `submitit.Job.state` returns empty (post-squeue-reap), so a job that timed out before writing its result pickle no longer hangs `.result()` indefinitely. 12 hermetic tests in `test_run_parallel_timeout.py` cover the head-state normalisation + reap-sequence handling.
- [x] **Cloud preflight** — `gate_candidate.py --probe-cloud` (opt-in) does one `fsspec.filesystem('gs').ls(<bucket>)` + one `pyscx.open_cloud(<probe_url>)` in <30 s before any sbatch. Catches missing/mismatched `gcsfs` and catalog-format bugs (the two recurring pre-run failure modes from the 2026-05 firefight: 24 zarr cloud FAST_FAILs and 158 SCX cloud FAST_FAILs respectively) before they cascade into the live gate.
- [x] **Fixture re-conversion workflow** — `benchmarks/scripts/reconvert_fixtures.py` is the canonical entry point for rebuilding derived SCX / Zarr / TileDB local files (and their cloud-pushed copies) after a source h5ad fixture change. Default matrix: all datasets × `{scx_auto, tiledb_soma, zarr_zstd}`. `--cloud-push` invalidates the prior `.blake3` completion sidecar before re-uploading so `ensure_cloud_fixture`'s short-circuit doesn't skip fresh data. Replaces the ad-hoc `/tmp/reconvert_stale.py` from the 2026-05 campaign.

### Go/No-Go Gate
- [x] Comprehensive cloud benchmark suite runs to completion on GCS via the unified launcher (`run_parallel.py --benchmarks cloud_*`).
- [x] Cross-format cloud comparison (SCX vs Zarr v3 vs TileDB-SOMA vs SLAF) published with reproducible scripts (`cloud_read`, `cloud_filtered`, `cloud_metadata`; launched via the documented entry points in `benchmarks/README.md` and `docs/cloud.md`).
- [x] Published cost model ($/1M cells read) — `cost_model.py` benchmark + `GCS_PRICING` rate-card pin. ±20% on three GCP instance sizes requires live `submit_gcp_matrix.py --yes-spend` runs — infrastructure is in place, numbers land when operators spend the budget.
- [x] Regression gate catches a synthetic 15% slowdown before release — enforced by `tests/test_gate_self_test.py` (5 hermetic pytest cases, all passing).

### Pitfalls and Risks
- **Cloud benchmark drift is expensive.** Every GCS / S3 read / write is billed. Cache aggressively, keep a shared test bucket (`gs://arc-ctc-nextflow/scx-test`), and never re-upload benchmark fixtures per-run.
- **Network noise.** Cloud throughput is inherently variable (shared tenancy). Run ≥3 repetitions, report median and p95, and always pin the GCP region to the bucket's region to avoid silent cross-region egress.
- **Credentials in CI.** Never embed service-account keys in scripts or commits. Use workload identity or short-lived tokens for automated runs; the existing scripts expect `GOOGLE_APPLICATION_CREDENTIALS` to be set out-of-band.
- **Comparing unlike things.** Zarr v3, TileDB-SOMA, and SCX have different cloud access models (exploded objects vs fragments vs range reads). Always report *the same end-user query*, not *the same read pattern* — the query is what users care about.
- **Moving competitors.** zarr-python 3, `tiledbsoma-ml`, and `slafdb` all ship frequently. Pin exact versions in the conda envs; re-bench against the latest whenever the risk register flags a move.
- **SLAF architectural mismatch.** SLAF is a table/SQL-native format (DuckDB + Polars) and exposes a different query surface than SCX's binary shards. Benchmark comparisons must be driven by the *user-facing query* (e.g. "load all T cells", "train scVI for one epoch") not the underlying read pattern — otherwise each format is compared on a strawman.

---

## What Users Get at Each Tier

| Tier | Months | User Experience |
|-------|--------|----------------|
| **1** | 1-4 | **COMPLETE.** Convert to SCX for 50-80% smaller files and 4-38× lower memory. Run scanpy/scVI/everything as usual via `to_anndata()`. Read speed is slower than h5ad in Tier 1 (no parallelism, no codec auto-select). |
| **2** | 4-7 | Auto-codec selection + parallel decode fix read performance. Fast training loader saturates GPUs. Query/filter large datasets without loading everything. Append/merge/delete without full rewrites. |
| **3** | 7-10 | **PARTIALLY COMPLETE.** R bindings, extended CLI (`build-csc`, `subset`, `upgrade`), complete documentation. CITE-seq / Multiome / TEA-seq multimodal + Seurat v5 / MAE interop shipped, including multimodal `merge` / `compact` / `append` / `subset --filter`. Detection bitmap shipped. GDS and spatial transcriptomics R-tree remain **DEFERRED**. |
| **4a** | 10-12 | **COMPLETE.** Full scanpy backed mode parity: native aggregation, comparison optimization, streaming preprocess, chunk iteration, selective loading. |
| **4b** | 12-15 | **COMPLETE.** Rust-native PCA/kNN/UMAP/DE/pseudobulk accelerators (3-10× faster at scale). |
| **4c** | 15+ | **COMPLETE.** GPU analysis routes to rapids-singlecell for in-VRAM PCA/kNN/UMAP/preprocess/HVG. Leiden stays cuGraph-native (16×). Native UMAP, covariance PCA, standalone kNN removed (Phase 3). Streaming/randomized PCA, HVG `seurat_v3`, DE, Harmony survive natively. `to_gpu_anndata()` for minimal-copy device handoff. |
| **4e** | 15+ | **COMPLETE.** Rapids-singlecell GPU compute transition. Phases 0–3 (evidence → routing → gate → removal) done. Phase 4 (format): the decode-metadata sidecar was implemented in F5 then removed — superseded by the codec-agnostic row-group `BlockIndex` framing (decode→device fast path, sorted-CSR invariant). |
| **4d** | 16+ | **COMPLETE.** Eliminate materialization: lazy normalize/log1p, column-projected streaming aggregation, streaming PCA through transforms via `ShardSource` trait, non-materializing `filter_cells`/`filter_genes`. Full out-of-core pipeline from open → QC → preprocess → PCA → kNN → UMAP → Leiden with ~11 GB peak RSS at 1M cells (vs ~22 GB materialized; 51% reduction). |
| **5** | ongoing | **COMPLETE** (S3/Azure deferred; CITE-seq multimodal depends on §3.4). Comprehensive multi-format benchmark harness validated on Chimera HPC + GCS, including SLAF parity across compression / read / selective / ML / memory, fragment-ops throughput, cloud push/pull/read/metadata/filtered-query + cost model + GCP instance matrix, on-demand regression gate with justification workflow + rolling HTML dashboard, and honest idempotent-retry contract for interrupted pulls. |

---

## Dependencies and External Libraries

| Component | Rust Crate / Library | Purpose |
|-----------|---------------------|---------|
| HDF5 reading | `hdf5-metno` (metno fork of `aldanor/hdf5-rust`; v0.9.4) | h5ad conversion |
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
| Rice codec complexity | Medium | **RESOLVED.** Scalar reference is normative; SIMD FOR-BP shipped (44% faster index decode). Rice is <20% of total decode cost. Zstd fallback always available. |
| GPU driver/GDS compatibility | Medium | CPU path always functional; GDS is opt-in and currently deferred |
| AnnData zero-copy edge cases | Medium | Extensive round-trip testing; fallback to copy for problematic dtypes |
| HDF5 crate stability | Low | Only needed for conversion; SCX native path takes over |
| Scope creep into analysis tools | Medium | AnnData bridge means users keep their existing tools; the analysis accelerators (§4) are optional |
| Incumbents improve faster than expected | High | If AnnData Zarr v3 + SOMA-ML close the gap, pivot: contribute codec/loader back into existing formats rather than pushing a new format |
| Compression claims don't generalize | Medium | Benchmark on diverse datasets (10x, Smart-seq2, CITE-seq, spatial) before publishing claims; be honest about where Rice underperforms |
| PyO3 breaking changes | Low | Pin pyo3 + numpy crate versions; budget time for migration if needed |
| extendr R binding immaturity | Medium | Allocate extra testing time; consider R subprocess fallback for edge cases |
| Cloud benchmark cost runaway | Medium | Shared test bucket with object lifecycle rules; pin region; cap runs per CI job; report median + p95 over ≥3 repetitions |
| GCP / S3 / Azure API drift | Low | Delegate to `object_store` crate; re-bench on upgrade; keep provider-specific env-var docs in `docs/cloud.md` current |
| `gcsfs` / `fsspec` minor-version bumps silently shift cloud read latency | Medium | Exact versions pinned in `scx-bench.yml` (`gcsfs=2025.9.0`, `fsspec=2025.9.0`); regression gate catches the drift via `cloud_read` timing tolerance |
| SLAF `SLAFDataLoader` Mixture-of-Scanners prefetcher returns 0 batches at 10M cells with default config | Medium | Flagged as SLAF-upstream tuning issue, not a harness fix; ML-loader benchmark records TTFB + batches/s so regressions on smaller datasets still surface |
| `scx-cloud` pull is idempotent-retry only, not resumable-from-checkpoint — interrupted pulls must re-download all shards | Low | Deliberate design (atomic-rename safety invariant; shards are independent and bounded). Documented in `docs/cloud.md`; `CloudError::Interrupted` variant + stale-`.tmp.*` sweep land the contract |
| `GCS_PRICING` table in `config.py` drifts silently from the GCS rate card | Low | Pricing table is explicit (not scraped); `docs/performance.md` "Cost model" section dates the capture. Re-check on release and on any significant-egress CI alert |
| SCX cloud filtered query benchmark scope (`cloud_filtered`) still measures pull-then-local-filter | Low | Native `open_cloud(...).query()` shipped — `SectionReader` + `QueryPipeline::from_reader` + `scx query <url>` are wired end-to-end. `cloud_filtered` will adopt the native variant alongside `CloudQueryOptions` (parallelism / max-inflight / cache-dir) and a batched async fetcher in a follow-on PR |

---

## Competitive Landscape (as of March 2026)

Key developments to monitor that affect SCX's value proposition:

| Project | What to watch | Impact on SCX |
|---------|--------------|---------------|
| **AnnData + Zarr v3** | zarr-python 3 (released Jan 2025) with sharding + async I/O. AnnData migration to Zarr v3 as primary backend | If AnnData-on-Zarr closes the cloud access gap, SCX's HPC advantage must be larger to justify adoption |
| **TileDB-SOMA-ML** | Alpha → stable release. C++ acceleration. Performance improvements | If SOMA-ML achieves competitive throughput, SCX's training loader advantage narrows |
| **BPCells** | Bitpacked on-disk sparse matrices for Seurat v5. 44M cells on a laptop. Potential Python bindings | A Python BPCells could address similar pain points without requiring a new format |
| **rapids-singlecell** | GPU-accelerated scanpy replacements via cupy/cuml. SCX now routes in-VRAM PCA/kNN/UMAP/preprocess/HVG to rapids-singlecell (Phases 0–3 complete) | SCX leverages rapids as a compute backend rather than competing. Risk shifts to rapids API stability and conda packaging friction |
| **scverse governance** | Consolidation around h5ad/Zarr. Community standards for new formats | SCX may face community resistance if it's seen as fragmenting the ecosystem |
| **CELLxGENE Census** | 125M+ cells on TileDB-SOMA. Growing API adoption | Census standardization on SOMA creates network effects that SCX must overcome |
| **SLAF** ([slaf-project/slaf](https://github.com/slaf-project/slaf)) | SQL-native sparse lazy format: DuckDB + Polars backend, Scanpy-compatible lazy API, PyTorch tokenizers/dataloaders for foundation models, `slafdb` on PyPI | Overlaps directly with SCX across lazy AnnData, selective queries (SQL pushdown), and ML training loader. A SQL-first approach sidesteps a binary-format learning curve; if SLAF matches SCX on throughput, the pitch shifts to "binary format + domain codec + single-file portability" |

**Strategic implication**: SCX's highest-risk scenario is not that it fails technically,
but that incumbents improve fast enough to close the gaps SCX targets. The tiered
roadmap mitigates this — the format / codec / bridge tier validates the format thesis
before committing to the full ecosystem. If those benchmarks show only modest improvements
over improving incumbents, the honest response is to contribute the codec and loader
innovations to existing tools (e.g., a Rice codec plugin for Zarr, a Rust training loader
for AnnData/h5ad) rather than pushing full format adoption.
