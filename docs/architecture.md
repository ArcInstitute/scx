# SCX Architecture

SCX (Sparse Cell eXpression System) is a co-designed **file format**, **compression codec**,
**query engine**, and **ML data loader** for single-cell RNA-seq data. It replaces AnnData/h5ad,
scipy.sparse, and the scanpy I/O layer with a unified Rust-native stack.

This document describes the high-level architecture, crate structure, and data flow.
For the full binary format specification, see [format.md](format.md) and [codec.md](codec.md).
For the API reference, see [api.md](api.md).

---

## Crate Dependency Graph

The workspace contains 15 crates plus an integration-test crate (`scx-integration-tests`). Dependencies flow bottom-up:

```
                        ┌──────────┐
                        │  pyscx   │  Python bindings (PyO3 + maturin)
                        └────┬─────┘
                             │ depends on all below
                        ┌────┴─────┐
                        │ scx-cli  │  CLI tool
                        └────┬─────┘
                             │ depends on all below
       ┌──────────────┬──────┼──────────────┬─────────────┬────────────-─┐
       │              │      │              │             │              │
┌──────┴──────┐ ┌─────┴────┐ │    ┌─────────┴──┐   ┌──────┴──────┐  ┌────┴──────┐
│ scx-engine  │ │ scx-ops  │ │    │ scx-cloud  │   │  scx-mtx    │  │ scx-accel │
│ query engine│ │ file ops │ │    │ cloud ops  │   │ MTX I/O     │  │ PCA/kNN/  │
└──────┬──────┘ └─────┬────┘ │    └─────────┬──┘   └──────┬──────┘  └──┬─┬──────┘
       │              │      │              │             │            │ │ [gpu]
       │    ┌─────────┘  ┌───┴────────┐     │             │            │ │
       │    │            │ scx-loader │     │             │            │ │
       │    │            │ ML loader  │     │             │            │ │
       │    │            └───┬────────┘     │             │            │ │
       │    │                │              │             │            │ │
       └────┴────────────────┼──────────────┴─────────────┴────────────┘ │
                             │                         ┌───────────┐     │
                    ┌────────┴───────┐                 │  scx-gpu  │◀────┘
                    │ scx-format-io  │                 │ GPU codec │
                    │ runtime I/O:   │                 │ + analysis│
                    │ reader/writer/ │◀────────────────┴─────┬─────┘
                    │ backed/decode  │                       │
                    └────────┬───────┘                       │
                       ┌─────┴───────┐                        (scx-gpu also
                       │ scx-format  │  pure on-disk           depends on
                       │ layout/spec │  layout (no I/O)        scx-format-io)
                       └──────┬──────┘
                  ┌───────────┼────────────┐
                  │                        │
           ┌──────┴──────┐         ┌───────┴────┐
           │  scx-codec  │         │ scx-sparse │
           │ compression │         │ CSR types  │
           └─────────────┘         └────────────┘

rscx (R bindings via extendr, depends on scx-format-io, scx-codec, scx-sparse, scx-engine, scx-ops)
```

### Crate Summary

| Crate | Role | Key modules |
|-------|------|-------------|
| **scx-codec** | Compression codecs (standalone, no I/O) | `rice`, `forbp`, `delta_golomb`, `bitstream`, `dispatch` |
| **scx-sparse** | CSR/CSC matrix types with scipy-compatible dtypes | `csr` (`ScxCsr`), `csc` (`ScxCsc`), `transpose` (streaming CSR→CSC), `convert` (CSR ↔ dense) |
| **scx-format** | Pure on-disk layout/spec — no `std::fs`, no `memmap2` (the surface conformance vectors verify against) | `header`, `catalog`, `catalog_view`, `shard`, `section`, `modality`, `codec_select`, `provenance`, `csc_policy`, `error`, `checksum` |
| **scx-format-io** | Runtime reader/writer, backed/streaming access, shard codec dispatch, sidecars; re-exports the full `scx-format` surface | `reader`, `writer`, `backed`, `shard_decode`, `shard_source`, `encoder`, `decode_sidecar`, `csc_sidecar`, `bitmap`, `deletion_vectors`, `mem`, `arrow_compat` |
| **scx-ops** | File lifecycle operations | `append`, `append_from_reader` (streaming SCX→SCX), `delete`, `compact`, `optimize` (in-place sidecar + canonical v3 upgrade), `merge`, `rollback`, `flock` |
| **scx-engine** | Lazy query engine with predicate pushdown | `pipeline`, `predicate`, `pushdown`, `projection`, `fused_ops`, `index`, `collect` |
| **scx-loader** | ML training data loader (triple-buffered) | `pipeline`, `io_stage`, `decode_stage`, `shuffle`, `projection`, `normalize`, `batch`, `python` |
| **scx-cloud** | Cloud access operations (S3, GCS, Azure) | `backend`, `cloud_optimize`, `explode`, `pack`, `pull`, `push`, `coalesce`, `cloud_reader` |
| **scx-mtx** | Matrix Market (MTX) I/O (always-on, no feature gate) | `read` (COO→CSR, TSV parsers, gzip), `write` (CSR→COO, gzipped output) |
| **scx-accel** | Rust-native analysis accelerators (opt. GPU via `gpu` feature) | `route` (accelerator execution planner + rapids probe), `pca` (streaming/randomized SVD, auto-routed; in-VRAM routes to `rsc.pp.pca`), `neighbors` (HNSW kNN; in-VRAM routes to `rsc.pp.neighbors`), `umap` (routes to `rsc.tl.umap`), `hvg` (streaming `seurat_v3`; extra flavors route to `rsc.pp.highly_variable_genes`), `fused` (fused `pca_neighbors_umap` / `pca_neighbors` pipelines via rapids), `diffexp` (Wilcoxon with pre-ranking), `leiden` (Rust-native CPU + cuGraph GPU), `harmony` (Harmony2 batch integration — soft k-means + ridge regression), `lisi` (exact-kNN Local Inverse Simpson Index), `pseudobulk`. GPU dispatch when `gpu` feature enabled; rapids-singlecell detected at runtime (not a pip extra). |
| **scx-gpu** | CUDA-accelerated codec decoding, GPU analysis, and GPU interop | `rice_decode`, `forbp_decode`, `sparse_to_dense`, `cusparse` (SpMM), `cusolver` (QR), `curand` (random matrix), `gpu_pca`, `gpu_knn` (CAGRA, device-resident fused path), `gpu_harmony` (distance / softmax+penalty / L2-normalize / batched correction kernels), `gpu_preprocess` (fused normalize+log1p), `gpu_matrix_source` (unified `GpuMatrixSource` capability trait over the row-major `GpuShardSource` (CSR) and column-major `GpuCscShardSource` (CSC) device shard sources, with G3-shaped pinned-ring staging), `gds` |
| **scx-cli** | Command-line interface | `convert`, `info`, `validate`, `query`, `append`, `delete`, `compact`, `optimize`, `merge`, `rollback`, `benchmark`, cloud ops |
| **pyscx** | Python bindings via PyO3 | `experiment`, `anndata`, `ops`, `query`, `cloud`, `backed`, `accel`, `preprocess`, `lazy_transform`, `projected_agg` |
| **rscx** | R bindings via extendr | Seurat v5 + SingleCellExperiment interop, pipe-friendly query API |

> [!NOTE]
> `scx-loader` does **not** depend on `scx-engine` — it has its own streaming-optimized
> gene projection and fused normalization, designed for the hot-path requirements
> of ML training. `scx-cloud` does **not** depend on `scx-loader` — they are siblings.
> `scx-cloud` reuses `scx-engine` for predicate parsing (selective pull).
> `scx-mtx` is **always-on** (no feature gate) since MTX is pure text I/O with no
> HDF5 dependency. Both `scx-cli` and `pyscx` depend on it.
> `scx-accel` depends on `scx-format-io`, `scx-sparse`, and `scx-engine` (for
> `project_csr` in `diffexp.rs`) — no loader dependency.
> It uses `faer` for dense linear algebra (QR, SVD, eigendecomposition),
> `instant-distance` for HNSW kNN, `rand_chacha` for deterministic Leiden seeding,
> and `libc` for `malloc_trim` in the Leiden optimizer.
> With the `gpu` feature enabled, `scx-accel` gains an optional dependency on `scx-gpu`
> and routes GPU analysis through **rapids-singlecell** (`rsc.*`) when available:
> PCA → `rsc.pp.pca`, kNN → `rsc.pp.neighbors`, UMAP → `rsc.tl.umap`,
> preprocessing → `rsc.pp.*`, HVG extra flavors → `rsc.pp.highly_variable_genes`.
> rapids-singlecell is a detected runtime dependency (conda `envs/scx-gpu-analysis.yml`);
> `pyscx/src/accel/rapids.rs` provides a one-shot import probe and emits a
> `no_rapids` `UserWarning` if the package is absent.
> `SCX_FORCE_NATIVE_GPU=1` pins surviving native paths;
> `SCX_DISABLE_RAPIDS=1` forces the no-rapids fallback for testing.
>
> **What survives natively** (not routed to rapids):
> streaming / randomized PCA (>VRAM moat), HVG `seurat_v3` (1.2× at 1M),
> Leiden (Rust-native CPU + cuGraph GPU), DE Wilcoxon/pdex (CSC/CSR-direct),
> Harmony, preprocessing kernels (ML loader + streaming), device-resident
> CAGRA kNN (fused pipeline only), codec decode (`rice_decode.cu`,
> `forbp_decode.cu`), and `gpu_graph.rs` (generic graph capture).
>
> Native GPU UMAP (`umap_sgd.cu`, `gpu_umap.rs`), in-VRAM covariance PCA
> (`gpu_pca_covariance.rs`), standalone kNN CAGRA dispatch, and PCA SpMM
> graph-capture (`pca_spmm_capture_opt_in`) were removed in Phase 3.

---

## File Format Overview

An SCX file (`.scx`) is a single packed binary file. One file contains
the expression matrix, all metadata, embeddings, graphs, and an internal
catalog for O(1) random access to any component.

```
┌────────────────────────────────────────────────────-─┐
│ FILE HEADER           (256 bytes, offset 0)          │  Magic b"SCX\x01", dimensions, codec, flags
├───────────────────────────────────────────────────-──┤
│ ROOT CATALOG          (offset 256, max 4096 bytes)   │  Compact index of section groups
├──────────────────────────────────────────────────-───┤
│ SECTIONS              (8-byte aligned)               │
│   obs metadata        (Arrow IPC, single or sharded) │
│   var metadata        (Arrow IPC, single or sharded) │
│   predicate indexes                                  │
│   X/csr/000000..N-1   (CSR shards — expression data) │
│   X_csc_shard_0..M-1  (CSC sidecar — optional)       │
│   layers, obsm, obsp, uns, provenance                │
│   deletion vectors    (optional, Roaring Bitmap)     │
├───────────────────────────────────────────────────-──┤
│ FULL CATALOG          (at EOF)                       │  Per-section checksums + shard statistics
└──────────────────────────────────────────────────-───┘
```

**Key design properties:**
- **Single file** — easy to copy, stage, and manage
- **CSR-native** — row-major sparse storage matches 60-80% of scRNA-seq access patterns
- **CSC sidecar** — optional column-major view for gene-centric analytics (DE, HVG, per-gene QC)
- **Sharded** — expression matrix is split into shards of ~10K cells each, enabling parallel I/O and selective reads
- **Immutable fragments** — sections are never overwritten; appends write new data at EOF and update the catalog pointer atomically
- **Dual catalog** — root catalog (fixed position) for fast open; full catalog (at EOF) for random access

For the complete binary layout, see [format.md](format.md).

---

## Codec System

SCX includes domain-specific codecs that exploit the statistical properties of
UMI count data (55-65% ones, near-geometric tail, ~2.0 bits/value entropy).

### Codec Pipeline (codec_id = 1, "Scx1")

Each CSR shard's three arrays are encoded independently:

```
CSR Shard
  ├── indptr   ──→  Delta-Golomb-Rice    (monotonic u64 pointers)
  ├── indices  ──→  FOR-BP               (sorted column indices per row)
  └── values   ──→  Adaptive Rice        (non-zero UMI counts, ~2.2 bits/value)
```

| Codec | Array | Technique |
|-------|-------|-----------|
| **Delta-Golomb** | indptr | Delta encoding + Rice coding of deltas |
| **FOR-BP** | indices | Frame-of-Reference + Bit-Packing, 128-row blocks (SIMD BitPacker4x for ≥128 NNZ rows) |
| **Adaptive Rice** | values | Per-block (256 values) Rice coding with adaptive *k* parameter |

All codecs use **LSB-first** bit packing. The bitstream module (`scx-codec/src/bitstream.rs`)
provides the shared reader/writer.

### Codec Selection

The file header stores a default `codec_id`, but each shard header may **override** it:

| codec_id | Name | When used |
|----------|------|-----------|
| 0 | None | Raw LE arrays. Fast for GDS bypass. |
| 1 | Scx1 | Integer counts with median ≤ 8 (typical 10x UMI data). Uses SIMD BitPacker4x for FOR-BP index decode. |
| 2 | Zstd | Integer data with median > 8. General-purpose fallback. |
| 3 | Lz4Shuffle | Byte-shuffle pre-filter + LZ4 frame compression. Speed-optimized. Matches Zarr/Blosc style. |
| 4 | Pcodec | Pcodec (pco) lossless numerical compression. Optimal for float layers (log-normalized, embeddings). 7–16% better compression than Zstd on float data, with ~20–40% slower encode/decode. |

Auto-codec selection (`scx-format/src/codec_select.rs`) samples up to 10K non-zero
values per shard: integer data uses the median heuristic to choose Scx1 vs Zstd,
float data routes to Pcodec. LZ4+shuffle is available via `codec="lz4"` but not auto-selected.

---

## CSC Sidecar Architecture

The CSC (Compressed Sparse Column) sidecar is an **optional, additive column-major
view** of the same expression matrix data that the primary CSR shards hold. CSR
shards stay on disk unchanged — the CSC shards add roughly the same compressed
bytes (same nnz, just laid out column-major; codec compression ratios are similar
under Scx1 / Zstd / Pcodec).

### Design rationale

SCX's primary storage is CSR — optimized for row-major (cell-centric) access
patterns: PCA, ML training, per-cell QC, cell subsetting. But several key
analysis operations are inherently **column-major** (gene-centric):

| Operation | Why CSC helps |
|-----------|---------------|
| Differential expression (small gene subsets) | Reads each gene's column as a single CSC slab instead of decoding every CSR row and projecting |
| Highly variable genes (single-batch seurat_v3) | Single-pass per-column accumulators with no O(n_vars) row-wise scratch |
| Per-gene QC metrics | Gene-axis aggregations route through CSC |
| Filtered pseudobulk | Only the requested gene columns are decoded |

Without a CSC sidecar, these operations must decode every CSR shard and project
out columns — wasting I/O proportional to the full matrix rather than the
queried gene set.

Operations that are row-axis-only (PCA, full-pass HVG, per-cell QC, ML
training) should **not** use CSC. Both PCA methods (covariance and randomized
SVD) explicitly reject `prefer_format="csc"` on the pyscx side.

### On-disk format

CSC shards reuse the **exact same 76-byte shard header** as CSR shards
(see [format.md §4](format.md#4-csr-shard-internal-layout)). The structural
layout is identical — header + encoded `indptr` / `indices` / `values` +
block index. The column-major semantics live entirely in field interpretation:

| Header field | CSR semantics | CSC semantics |
|---|---|---|
| `shard_type` | `0` | `1` (authoritative) |
| `n_major` | rows in this shard | **columns** in this shard |
| `n_minor` | columns in full matrix | **rows** in full matrix (`n_obs`) |
| `global_offset` | first row index | first **column** index |
| `indptr` (len `n_major + 1`) | row-pointer | **column-pointer** |
| `indices` (len `nnz`) | column indices | **global row** indices (NOT shard-local) |

The catalog's `section_type = CscShard (5)` is the authoritative discriminator;
the in-shard `shard_type` byte exists for self-contained shard validation
(e.g. exploded `.scxd` files). Catalog `ShardStats.row_start` / `row_end`
fields are reused for the **major-axis** range (i.e. `col_start..col_end`
for CSC); the accessors `ShardStats::major_start()` / `major_end()` handle
the dispatch. See [format.md §4.1](format.md#41-csc-shard-internal-layout)
for the full field-level spec.

### Multi-shard column layout

CSC sidecars are split by column range (default: 5000 columns per shard via
`--csc-cols-per-shard`). Each shard covers a contiguous half-open
`[col_start, col_end)` range, non-overlapping and sorted by `col_start`:

```
n_vars = 36000, --csc-cols-per-shard 5000
                       ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬───┐
CSC shards (8 total):  │ 0..5K│5..10K│10..15│15..20│20..25│25..30│30..35│..36│
                       └──────┴──────┴──────┴──────┴──────┴──────┴──────┴───┘
```

Two reasons to split by column range rather than emitting one giant shard:

1. **Column-range pushdown.** `BackedCscIndex::shards_for_col_range` binary-searches
   the sorted ranges and skips non-overlapping shards entirely. With 5000 cols/shard
   on a 36K-gene matrix, a single-gene DE query touches 1 shard out of 8.
2. **Bounded transpose memory.** The streaming CSR→CSC transpose chunks by column
   range, so peak memory during `build-csc` scales with
   `csc_cols_per_shard × n_obs × 8 bytes` rather than the full matrix.

Pass `--csc-cols-per-shard 0` for no cap (single CSC shard).

### Key types and data flow

```
                   CREATION                                    CONSUMPTION
 ─────────────────────────────────────────   ──────────────────────────────────────

 scx-sparse/src/transpose.rs                scx-format/src/backed.rs
 ┌──────────────────────────────────┐        ┌─────────────────────────────────┐
 │ streaming_csr_to_csc_iter_with   │        │ BackedCscIndex                  │
 │ _cap()                           │        │   shard_ranges: Vec<(col_start, │
 │   - iterates CSR shards          │        │     col_end, sorted_idx)>       │
 │   - yields CscArrays per col     │        │   shards_for_col_range(lo,hi)   │
 │     chunk (memory-bounded)       │        │     → Vec<usize>  (binary srch) │
 └────────────┬─────────────────────┘        └────────────┬────────────────────┘
              │                                           │
              ▼                                           ▼
 scx-format/src/writer.rs                   ┌─────────────────────────────────┐
 ┌──────────────────────────────────┐        │ BackedCscReader                 │
 │ ScxWriter::write_csc_shard()    │        │   reader: ScxReader             │
 │   section_type = CscShard(5)    │        │   index:  BackedCscIndex        │
 │   codec, value_encoding per     │        │   cache:  LRU<usize, ScxCsc>    │
 │   shard                         │        │                                 │
 └──────────────────────────────────┘        │   read_csc_columns(col_range)   │
                                            │   read_csc_columns_subset(cols) │
                                            │   read_shard_cached(idx)        │
                                            └────────────┬────────────────────┘
                                                         │
                                                         │ impl ColumnShardSource
                                                         ▼
                                            scx-accel/src/csc/
                                            ┌─────────────────────────────────┐
                                            │ wilcoxon_rank_sum_streaming_csc │
                                            │ streaming_mean_var_csc          │
                                            │ pdex_ref_streaming_csc          │
                                            │ pseudobulk_aggregate_csc        │
                                            │                                 │
                                            │ All generic over                │
                                            │   S: ColumnShardSource          │
                                            └─────────────────────────────────┘
```

**`BackedCscIndex`** (`scx-format/src/backed.rs`): A sorted vector of
`(col_start, col_end, sorted_shard_idx)` ranges built from the catalog at
construction time. Provides O(log n) column lookups via `partition_point`:
`shard_for_col(col)` for single-column lookups and
`shards_for_col_range(c_lo, c_hi)` for range queries.

**`BackedCscReader`** (`scx-format/src/backed.rs`): The primary CSC consumer.
Wraps an `ScxReader` + `BackedCscIndex` + a count-only LRU cache for decoded
`ScxCsc` shards (simpler than the CSR reader's byte-budgeted / singleflight
cache — CSC analytical workloads access shards in column-range order with
limited reuse). Key method: `read_csc_columns(col_range)` skips non-overlapping
shards, `col_slice`s partial-overlap shards post-decode, and concatenates
results. Implements `ColumnShardSource`.

**`ColumnShardSource`** (`scx-format/src/shard_source.rs`): The trait that
abstracts CSC access. Both `BackedCscReader` (raw on-disk) and pyscx's
`LazyShardSource` (transform-aware) implement it. All `scx-accel` CSC kernels
are generic over this trait — no concrete type dependency.

**`PreferFormat` + `require_csc()`** (`scx-accel/src/csc/dispatch.rs`):
Explicit opt-in dispatch. There is intentionally no `Auto` variant — every
CSC dispatch is explicit at the call site. Callers pass `prefer_format="csc"`
through pyscx kwargs; `require_csc()` either returns the `ColumnShardSource`
or a clean error explaining why CSC is unavailable.

### Creation pipeline

CSC sidecars are created by a streaming CSR→CSC transpose in
`scx-sparse/src/transpose.rs`:

1. All CSR shards for the matrix are decoded (or read from the existing file).
2. `streaming_csr_to_csc_iter_with_cap()` iterates column chunks bounded by
   `min(memory_budget, csc_cols_per_shard)` — each `next()` call transposes a
   `[col_start, col_end)` slice across all CSR shards.
3. Each yielded `CscArrays` is written via `ScxWriter::write_csc_shard()` as
   `section_type = CscShard(5)` with independent per-shard codec selection.

Entry points:

| Entry point | When |
|-------------|------|
| `scx build-csc` | Post-hoc addition to an existing file |
| `scx convert --csc=always` | During h5ad/h5mu → SCX conversion |
| `pyscx.from_anndata(csc="always")` | During Python-side conversion |
| `--rebuild-csc` on mutating ops | Re-emit after append/compact/merge/subset |

Typical throughput: ~10–20 seconds for `build-csc` on a 1M-cell × 30K-gene
file (single core, dominated by codec encoding).

### Multimodal support

`BackedCscReader::for_modality(reader, modality_id, cache_shards)` scopes the
CSC index and LRU cache to a specific modality, filtering by
`(SectionType::CscShard, modality_id)`. Each modality gets its own cache to
avoid thrashing under interleaved access (e.g. totalVI touching RNA + ADT in
the same step). `BackedCscReader::for_layer()` similarly scopes to a layer's
CSC sidecar.

### Mutating ops and CSC lifecycle

Mutating operations (`append`, `compact`, `merge`, `subset`) change the row
layout or column index space, making existing CSC `indices` arrays reference
stale rows/columns. Each op therefore **drops the CSC sidecar by default**
with a `log::warn!` message. Pass `--rebuild-csc` to re-emit the sidecar
against the post-op output.

`scx upgrade` is the exception — it preserves CSC sidecars by re-emitting
them through `catalog.csc_shards_sorted()` into the new file.

See [sharding.md § CSC sharding](sharding.md#csc-sharding) for the
detailed design rationale and [format.md §4.1](format.md#41-csc-shard-internal-layout)
for the full on-disk specification.

---

## Data Model

### On-disk types vs In-memory types

| Component | On-disk | In-memory (`ScxCsr`) |
|-----------|---------|---------------------|
| indptr | `u64` | `i64` (matches scipy) |
| indices | `u16` or `u32` | `i32` (matches scipy) |
| values | `u8`, `u16`, `u32`, `f32`, `f16` | `f32` (always) |
| obs/var metadata | Arrow IPC (single section or row-sharded) | Arrow RecordBatch → pandas |

The `ScxCsr` struct (in `scx-sparse`) uses the exact memory layout scipy expects —
`i64` indptr, `i32` indices, `f32` data — enabling **zero-copy** transfer to Python
via PyO3's buffer protocol.

---

## Reader/Writer Architecture

### Writer (`scx-format/src/writer.rs`)

Uses an **atomic rename** strategy for crash safety:

```
1. Create temp file (experiment.scx.tmp.{pid})
2. Write sections sequentially starting at offset 4352
3. Write full catalog at EOF
4. pwrite() root catalog at offset 256
5. pwrite() header at offset 0
6. fsync()
7. rename() temp → final path  (atomic commit point)
```

### Reader (`scx-format/src/reader.rs`)

Opens a file via `mmap` and validates magic/version/checksums:

```
1. mmap the file (MADV_NORMAL default)
2. Parse 256-byte header
3. Parse root catalog (offset 256)
4. Parse full catalog (from header's full_catalog_offset)
5. Individual sections accessed via catalog offsets
```

All section access is via offset+length from the catalog — no sequential scanning.

**madvise hints** (Unix only, `#[cfg(unix)]`):
- `MADV_SEQUENTIAL` on the shard byte range during `assemble_shards_parallel()` — tells kernel to readahead aggressively for full reads
- `MADV_WILLNEED` on next N shards in `BackedCsrReader::read_shard_cached()` — prefetches upcoming shards after a cache miss
- `MADV_DONTNEED` after `read_shard_uncached()` — releases page cache for decoded shards during streaming aggregation (67% RSS reduction on 1M cells)

### Catalog representations (`FullCatalog`, `CatalogView`, `LazyShardStats`)

`ScxReader` exposes three catalog representations that trade completeness against per-entry cost:

| Type | Location | When to use |
|------|----------|-------------|
| `FullCatalog` | `scx-format/src/catalog.rs` | Validation, mutation (append/compact/delete/rollback), metadata inspection, the writer round-trip, and all `scx-ops` / `scx-engine` tooling that needs the full per-entry record (`name: String`, `checksum: [u8; 32]`, `ShardStats` with all scalar + indexed-column fields). This is what `ScxReader::open` parses eagerly; the value is held internally as `Arc<FullCatalog>`. |
| `CatalogView` | `scx-format/src/catalog_view.rs` | Reader/open hot path. Drops the 32-byte `checksum`, the diagnostic `value_*` fields, and `col_*` on row-major shards; drops the `name: String` on `CsrShard` / `CscShard` / `ObspCsrShard` entries (looked up by `(section_type, major_start)` instead). Stats collapse to `ShardStatsLite { major_start, major_end, nnz }` — dispatched on `section_type` so the consumer reads the right axis without branching per access. Designed to wrap in `Arc<CatalogView>`. |
| `LazyShardStats` | `scx-format/src/catalog.rs` | Sibling of `ShardStats`. Eagerly decodes every fixed-width scalar field (`row_*`, `col_*`, `nnz`, `value_*`, `n_indexed_columns`) but retains the variable-length per-column stats payload as `Box<[u8]>` until `decode_column_stats()` (or `to_full()`) is called. Files with `n_indexed_columns == 0` never allocate the tail. |

**Which reader paths use which representation:**

- `ScxReader::open` and `ScxReader::open_unchecked` parse and retain `Arc<FullCatalog>` — the tooling-facing path. Compatibility-preserving for `scx-engine` predicate pushdown (needs `column_stats`), `scx-ops` mutations (needs per-entry checksums), and any caller of `ScxReader::catalog()`.
- `ScxReader::open_with_shared_catalog(path, Arc<FullCatalog>)` skips the catalog parse entirely; used by `pyscx::to_anndata_backed` to amortise a single parse across the N+3 sibling readers (main + X CSR + CSC sidecar + per-layer) it opens against the same file. The reused `Arc<FullCatalog>` is byte-identical to what an independent open would produce.
- `BackedCsrReader::new*` constructors (and `for_modality` / `new_for_layer_*`) build a `CatalogView` from the reader's `FullCatalog`, then walk the view's `csr_shards_sorted` / `csr_shards_for_modality` / `layer_csr_shards_sorted_with_prefix` helpers to construct a `Vec<ShardEntryLite>` per-shard table. They never clone `FullCatalogEntry`. Per-shard retained footprint drops from ~250 B → 32 B (with alignment), which compounds at the ~16K entries census-scale shards produce.
- `BackedCsrIndex::from_view_sorted(&[&CatalogViewEntry])` zips the major-axis range into `ShardRange` rows in a single pass over the pre-sorted view; it never re-walks the catalog.
- `scx-engine` predicate pushdown still reads `ShardStats` via `FullCatalog` — `LazyShardStats` is wired up but the pushdown evaluator has not been migrated. The shape of `LazyShardStats` is the eventual landing site for the cold-column-stats deferral.

**Fork safety:** `Arc<FullCatalog>` and `Arc<CatalogView>` have no interior mutability. `FullCatalog::reconcile_v1_csr_col_range` mutates entries once during `read_from` *before* the `Arc` wrap; after that point every reader path treats both types as frozen. A `fork()` from a Python DataLoader worker COW-duplicates the parent's catalog into each child — no shared mutex, no shared singleflight table, no atomic refcount contention across processes. See [docs/multithreading.md § Fork safety](multithreading.md#fork-safety).

### Sharded obs/var metadata reader APIs

Files produced by streaming `scx merge`, `scx append`, and `pyscx.from_anndata` (when `n_obs > shard_target_rows`) store obs/var metadata as row-sharded Arrow IPC sections (types 24/25) instead of a single monolithic section. `ScxReader` exposes a parallel set of shard-aware accessors alongside the legacy single-section API:

| Method | Returns | Notes |
|--------|---------|-------|
| `obs_shard_count()` | `usize` | Catalog-only; no payload read. Returns 0 for legacy single-section files. |
| `read_obs_shard(idx)` | `RecordBatch` | Returns on-disk wide types (e.g. `LargeUtf8`) as-is — no downcast. |
| `obs_shards()` | `impl Iterator<Item = RecordBatch>` | Iterator over all shards in shard-index order. |
| `read_obs_assembled()` | `RecordBatch` | Reassembles all shards into a single batch. **Explicit memory hazard** at atlas scale. |
| `read_obs_schema_physical()` | `Schema` | On-disk Arrow schema without reading payload. |
| `read_obs_schema_logical_lossy()` | `Schema` | Downcast schema (e.g. `LargeUtf8 → Utf8`) without reading payload. |
| `read_obs()` | `RecordBatch` | Returns the single-section batch. **Errors on sharded files** with a diagnostic directing callers to the shard APIs above. |

Mirror `var_shard_count`, `read_var_shard`, `var_shards`, `read_var_assembled`, `read_var_schema_physical`, `read_var_schema_logical_lossy`, and `read_var` accessors exist for var metadata, with identical semantics on the var axis.

---

## Fragment/Manifest Model (scx-ops)

SCX uses an immutable-fragment model inspired by Lance and Delta Lake.
Sections are **never overwritten** — operations append new data and update the catalog pointer.

```
                       ┌────────────────────────────────────┐
   scx-ops             │                                    │
  ┌────────────┐       │     .scx file                      │
  │ append     │──────▶│  [original sections] [new shards]  │
  │            │       │  [new catalog v1 → all sections]   │
  │ delete     │──────▶│  [deletion vectors (Roaring)]      │
  │            │       │  [new catalog v2]                   │
  │ compact    │──────▶│  Clean rewrite → new file          │
  │            │       │                                    │
  │ rollback   │──────▶│  Header update → prev catalog      │
  │            │       │                                    │
  │ merge      │──────▶│  Streaming merge → new file        │
  └────────────┘       └────────────────────────────────────┘
```

| Operation | What it does | Writes |
|-----------|-------------|--------|
| **append** | Add new shards at EOF | New sections + new catalog |
| **delete** | Logical deletion via Roaring Bitmap | Deletion vectors section + new catalog |
| **compact** | Reclaim space, merge small shards | Entire new file (atomic rename) |
| **optimize** | Re-encode + canonicalize CSR shards → add decode sidecars, stamp v3 | Entire new file (atomic rename) |
| **rollback** | Revert to previous catalog | Header-only update (pwrite) |
| **merge** | Combine multiple SCX files | New file with merged data |

Advisory `flock()` (`scx-ops/src/flock.rs`) prevents concurrent writers.
Multiple concurrent readers are always safe.

---

## Query Engine (scx-engine)

The query engine provides a **lazy pipeline** with Polars-style evaluation.
No data is read until `.collect()` is called.

### Pipeline Stages

```
open("file.scx")
  │
  ▼
filter_obs("tissue == 'lung'")     ← predicate parsing
  │
  ▼
select_genes(hvg_indices)          ← gene projection
  │
  ▼
normalize(target_sum=1e4)          ← fused with log1p when possible
  │
  ▼
.collect()                         ← parallel execution
  │
  ├── Catalog-level pushdown       (shard pruning via CategoryBitset stats)
  ├── Index-level pushdown         (row-level filtering via predicate indexes)
  ├── Parallel shard decode        (rayon thread pool)
  ├── Gene projection              (CSR column subset)
  └── Fused normalize + log1p     (single row scan)
```

### Key optimizations

- **Predicate pushdown (Level 1):** Per-shard statistics in the catalog (min/max for numerics,
  `CategoryBitset` for categoricals) enable skipping entire shards without reading any data.
- **Predicate pushdown (Level 2):** Predicate indexes (§3.5) provide row-level mappings
  within qualifying shards.
- **Operation fusion:** `normalize(1e4) + log1p()` → single fused CSR row scan
  (`scx-engine/src/fused_ops.rs`).
- **Parallel collection:** Qualifying shards are decoded and filtered in parallel
  via rayon (`scx-engine/src/collect.rs`).

### `SectionReader` — local + cloud unification

`QueryPipeline` is generic over a `SectionReader` trait that abstracts
how catalog sections are fetched. Two implementations ship today:

| Reader | Section fetch | Construction |
| --- | --- | --- |
| `ScxReader` (`scx-format`) | mmap / `pread` over a local `.scx` file | `QueryPipeline::open(path)` |
| `CloudSectionReader` (`scx-cloud`) | `object_store` range reads over `gs://` / `s3://` / `az://` / exploded `.scxd/` directories | `QueryPipeline::from_reader(reader)` |

`PyExperiment.query()` opens a local mmap-backed pipeline;
`PyCloudExperiment.query()` opens a cloud-backed pipeline (see
[docs/cloud.md § Cloud-native query](cloud.md#cloud-native-query)).
Both share the same predicate planning, shard pruning, gene
projection, and decoding code paths — only the bytes-by-section
implementation differs.

The CLI surface mirrors this: `scx query <input> "<predicate>"`
auto-detects the input as a local file, an exploded `.scxd/`
directory, or a cloud URL and constructs the matching `SectionReader`.

Cloud reads on the engine path currently issue per-shard
`block_on(read)` calls from rayon workers. A batched async
section-fetch stage is deferred to a follow-on (tracked alongside
`CloudQueryOptions`).

---

## ML Training Loader (scx-loader)

The loader is a **triple-buffered Rust pipeline** designed to keep the GPU
saturated during model training. It does **not** go through `to_anndata()`.

### Two row-source models

`scx-loader` exposes two parallel row-source types for ML training. They
share lower-level decode / projection / normalize primitives but use
different I/O orchestration tuned for different access patterns:

- **`TrainingDataset`** — catalog-order streaming. Sequential triple-buffered
  pipeline (tokio I/O → rayon decode → Python). Optimised for highest
  per-thread throughput (peak ~75K cells/s on 1M-cell synthetic; 82× faster
  than TileDB-SOMA-ML at scale). Right primitive for single-cell pretraining,
  classification, embedding extraction.

- **`IndexPlanDataset`** — consumer-supplied plans. Each batch is a
  `list[tuple[pert_idx, ctrl_idx]]` and the loader gathers paired rows via
  `BackedCsrReader::read_rows_with` — a zero-allocation dense-gather API
  that scatters directly from cached shards into the dense output without
  materialising an intermediate `ScxCsr`. Optimised for plan-driven pairing
  (perturbation training, contrastive learning, donor-matched designs).
  Random by definition — breaks the catalog-order I/O optimisation in
  exchange for per-cell pairing flexibility. Reaches ~20K cells/s at 1M
  cells (~3.7× slower than the sequential ceiling), 106× faster than the
  current cell-load-scx `ScxBackedSparseDataset` Python-loop baseline.

The two types share LRU shard caching (`BackedCsrReader::read_shard_cached`),
`HvgProjection::scatter_row` / `scatter_row_full`,
`fused_normalize_log1p_dense`, and `extract_obs_columns`. They do **not**
share `pipeline.rs` — the streaming pipeline's I/O stage sorts shard groups
by file offset, an optimisation that doesn't apply to plan-driven access.
See `docs/api.md` § `IndexPlanDataset` for details.

### Pipeline Architecture (TrainingDataset)

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Stage 1: I/O   │     │ Stage 2: Decode │     │  Stage 3:    │
│  (tokio async)  │────▶│ (rayon threads) │────▶│  GPU/Python  │
│                 │     │                 │     │              │
│  Read shard     │     │  Row shuffle    │     │  model.fwd() │
│  groups from    │     │  Gene project   │     │  loss.bwd()  │
│  .scx file      │     │  Sparse→dense   │     │  optim.step()│
│                 │     │  Normalize+log1p│     │              │
└─────────────────┘     └─────────────────┘     └──────────────┘
         bounded channel          bounded channel
         (back-pressure)          (back-pressure)
```

**"Zero Python on the hot path"** — all I/O, decompression, shuffling, sparse-to-dense
conversion, and normalization happen in Rust. Python touches only the training loop
and the forward/backward pass.

### Key Components

| Module | Responsibility |
|--------|---------------|
| `pipeline.rs` | `TrainingPipeline` coordinator, `LoaderConfig`, `MemoryBudget` |
| `io_stage.rs` | Async shard group reads, deletion vector filtering |
| `decode_stage.rs` | Parallel row scatter, obs metadata extraction |
| `shuffle.rs` | `ShardShuffler` (shard order) + `RowShuffler` (within-buffer Fisher-Yates) |
| `projection.rs` | `HvgProjection` — gene subset at decode time (~15× data reduction) |
| `normalize.rs` | Dense-row fused normalize + log1p |
| `batch.rs` | `Batch` struct — dense f32 matrix + obs columns |
| `index_plan.rs` | `IndexPlanLoader` + `IndexPlanIter` — plan-driven paired-batch reader (sibling row-source model; see above) |
| `python.rs` | `TrainingDataset` + `IndexPlanDataset` PyO3 classes (both implement `__iter__`/`__next__`) |

### Memory Budget

The loader auto-tunes `shard_group_size`, `prefetch_batches`, and `batch_size`
to fit within a configurable memory budget (default 512 MB):

```
total ≈ shard_buffer + batch_buffer + 50 MB overhead
  shard_buffer = (shard_group_size + 1) × decoded_shard_bytes
  batch_buffer = (prefetch_batches + 1) × batch_size × n_genes × 4
```

---

## Python Bindings (pyscx)

Built with PyO3 + maturin. Exposes two main interfaces:

### AnnData Bridge

```python
import pyscx

# Read
exp = pyscx.open("experiment.scx")    # → PyExperiment (lazy handle)
adata = exp.to_anndata()              # → AnnData (zero-copy CSR + Arrow→pandas)

# Write
pyscx.from_anndata(adata, "output.scx", codec="auto")
pyscx.from_10x("matrix.h5", "output.scx")
pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "output.scx")

# Export to Cell Ranger MTX directory
pyscx.to_mtx("output.scx", "/path/to/mtx_dir")
```

The `to_anndata()` path is **zero-copy** for the expression matrix — `ScxCsr`'s
`i64/i32/f32` arrays are handed directly to scipy via numpy buffer protocol.
Arrow metadata goes to pandas via pyarrow.

### Training Dataset

```python
dataset = pyscx.TrainingDataset(
    "experiment.scx",
    batch_size=1024,
    hvg_indices=hvg_array,
    normalize=True,
    log1p=True,
)

for batch in dataset:
    x = batch["X"]          # dense f32 numpy array
    obs = batch["obs"]      # dict of obs columns
```

`TrainingDataset` wraps the Rust `TrainingPipeline` and implements Python's
iterator protocol. The pipeline is constructed eagerly in `__new__`, so direct
use requires `num_workers=0` — a PID check in `__next__` raises `RuntimeError`
if a `TrainingDataset` constructed in the parent is reused from a forked
child. For `num_workers>0` with the fork start-method, wrap `TrainingDataset`
in a thin Python `IterableDataset` that constructs it inside the worker's
`__iter__` (the pattern used by `cell-load-scx`'s `ScxTrainingDataset` and
`state-scx`'s `ScxStateAdapter`); the tokio current-thread runtime and rayon
pool are then built inside the worker process and never inherit fork-hostile
parent state.

---

## Cloud Operations (scx-cloud)

The cloud crate provides cloud-native access to SCX data on S3, GCS, and Azure:

### Operations

| Operation | What it does | Direction |
|-----------|-------------|----------|
| **cloud-optimize** | Rewrite with front-of-file catalog | Local → local |
| **explode** | Packed `.scx` → exploded `.scxd/` directory | Local → local |
| **pack** | Exploded `.scxd/` → packed `.scx` | Local → local |
| **pull** | Stream from cloud → local packed file | Cloud → local |
| **push** | Stream from local → cloud exploded directory | Local → cloud |
| **CloudReader** | Direct cloud reads without download | Cloud → memory |

### Exploded Directory Layout

```
experiment.scxd/
├── _catalog.bin              # Full catalog
├── _header.bin               # 256-byte file header
├── obs.arrow                 # Obs metadata (Arrow IPC)
├── var.arrow                 # Var metadata
├── X/
│   ├── 000000.shard          # CSR shards (byte-identical to packed)
│   └── ...
├── obsm/                     # Optional
├── layers/                   # Optional
├── _provenance.bin           # Optional
├── _deletion_vectors.bin     # Optional
└── uns.json                  # Optional
```

`_catalog.bin` is uploaded last (atomic-publish semantics).

### Pull Pipeline

```
1. GET _catalog.bin + _header.bin
2. Parse catalog → know all sections
3. Download shard files in parallel (N=8 async tasks)
4. Reorder buffer → sequential writer thread
5. Write full catalog + front catalog + header
6. fsync + rename
```

**Selective pull**: `pull --filter "cell_type == 'T cell'"` downloads only matching
shards using catalog-level pushdown, reducing bandwidth by up to 20×.

### Authentication

Relies on `object_store`'s built-in credential chains:
- **GCS**: `GOOGLE_APPLICATION_CREDENTIALS` or instance metadata
- **S3**: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or instance profile
- **Azure**: `AZURE_STORAGE_ACCOUNT`/`AZURE_STORAGE_KEY` or managed identity

---

## CLI (scx)

The CLI binary is named `scx` (built from the `scx-cli` crate). It provides
format conversion, inspection, validation, query, file operations, and cloud
access:

```bash
# Convert h5ad/10x to SCX (requires --features hdf5)
scx convert input.h5ad output.scx --codec auto --shard-size 10000
scx convert --to h5ad output.scx output.h5ad

# Convert Cell Ranger MTX ↔ SCX (always available, no feature flag)
scx convert /path/to/filtered_feature_bc_matrix/ output.scx
scx convert --to mtx output.scx /path/to/mtx_output/

# Inspect file metadata
scx info experiment.scx --json --history

# Verify all checksums
scx validate experiment.scx --verbose

# Query engine
scx query experiment.scx "cell_type == 'T cell'" --count
scx query experiment.scx "tissue == 'lung'" --output subset.scx --normalize 1e4 --log1p

# File operations
scx append atlas.scx --input new_batch.scx
scx delete experiment.scx --filter "is_doublet == True" --dry-run
scx compact experiment.scx --output compacted.scx
scx optimize experiment.scx --output optimized.scx   # add decode sidecars + upgrade to v3
scx rollback experiment.scx --to-seq 3
scx merge batch1.scx batch2.scx batch3.scx --output atlas.scx

# Benchmarks
scx benchmark experiment.scx --compare-h5ad data.h5ad --runs 5 --json

# Cloud operations (requires --features cloud)
scx cloud-optimize experiment.scx --output cloud_ready.scx
scx explode experiment.scx experiment.scxd/
scx pack experiment.scxd/ experiment.scx
scx pull gs://bucket/experiment.scxd/ local.scx --filter "tissue == 'lung'"
scx push experiment.scx gs://bucket/experiment.scxd/ --parallelism 16
```

Feature flags: `hdf5` (h5ad/10x conversion, opt-in), `cloud` (cloud operations, opt-in).
MTX conversion is **always available** — no feature flag required.

---

## Data Flow Diagrams

### Conversion: h5ad → SCX

```
  h5ad file
      │
      ▼
  hdf5-rust          Read sparse matrix (CSR/CSC), obs, var, layers, uns
      │
      ▼
  CSC → CSR?         Transpose if stored as CSC (streaming scatter)
      │
      ▼
  Integer detect      Float32 counts → uint8/uint16/uint32 if lossless
      │
      ▼
  Auto-codec select   Per-shard: median ≤ 8 → Scx1, else → Zstd
      │
      ▼
  ScxWriter           Shard → encode → write sections → full catalog → atomic rename
      │
      ▼
  experiment.scx
```

The conversion code lives in the **`scx-convert`** crate (workspace
member 15, opt-in `hdf5` feature). Both `scx-cli` and `pyscx` depend on
it; the previous in-line `scx-cli/src/convert/` module was extracted
when streaming conversion landed so `pyscx` could share the pipeline
without depending on the binary-only `scx-cli`.

**Streaming ingestion** (`scx convert --stream`, `pyscx.from_h5ad`,
auto-routing on backed AnnData): `scx_convert::h5ad_to_scx_streaming`
loads the full `indptr` then iterates `XStreamReader::next_shard`,
running `sort_csr_rows_in_place` + `drop_explicit_zeros_inplace` per
shard before `encode_one_shard` → `ScxWriter::write_preencoded_shard`.
`obsm` / `varm` / `obsp` / `varp` are also hyperslab-read one
row-range at a time (`read_dense_mapping_shard` /
`read_sparse_mapping_shard`) and emitted as row-sharded sections
(`<section>/<name>_shard_<idx>`, types 20–23) so peak memory per
metadata matrix matches one X shard's worth. For pairwise
`obsp` / `varp`, a CSR input is streamed straight to COO and a **dense**
input is densified to nonzero COO one row-shard at a time; a CSC or
otherwise unsupported pairwise layout is dropped with a `DroppedObsp`
warning rather than silently skipped. Peak memory is therefore
bounded by **one X shard + one row-shard per `obsm` / `varm` / `obsp`
/ `varp` matrix**, plus the resident indptr — independent of total
dataset size or per-key embedding dimension. The pyscx backed-routing
path detects per-section mutation via top-level key comparison
against the source h5ad; clean sections route through the disk
streamer, mutated sections are extracted from Python and partitioned
into the same sharded layout on the way out. `csc="always"` triggers a
post-`finish()` `scx_ops::rebuild_csc_inplace` pass (transient disk
~2× the output size during the rebuild).

**Streaming export** (`scx convert --to h5ad/h5mu`, `pyscx.to_h5ad`,
`pyscx.to_h5mu`): the inverse path. `scx_convert::
scx_to_h5ad_streaming` (and `scx_to_h5mu_streaming` / `scx_modality_to_h5ad_streaming`)
iterate SCX CSR shards in row order
via `ScxReader::read_csr_shard_for` / `read_layer_csr_shard*` and
write hyperslab slices into pre-allocated `/X/{indptr,indices,data}`
HDF5 datasets. The total `nnz` is computed up front from catalog
`ShardStats` (single pre-scan decode when deletion vectors are
active) so the on-disk layout is deterministic — no extendable HDF5
datasets. Obs and var metadata stream through the symmetric
`h5ad::write::write_dataframe_group_streaming`: schema comes from
`read_obs_schema_logical_lossy()` (catalog-only), HDF5 datasets are
pre-allocated to `n_rows_kept`, then `obs_shards()` / `var_shards()`
are drained shard-by-shard with kept-row hyperslab writes per column.
Categorical columns unify disjoint per-shard vocabularies via a
single-pass running global dictionary (codes remapped on the fly,
final `categories` dataset emitted at finalize). Legacy single-section
obs/var sources transparently fall through to the eager
`write_dataframe_group_at` writer. Obsm / varm / uns / mappings still
reuse the materialising helpers. Peak memory is bounded by one
shard's worth of CSR per matrix written, plus one shard's worth of
obs/var per column when sharded. `--stream=false` falls back to the
legacy materialising `scx_to_h5ad` / `scx_to_h5mu` /
`scx_modality_to_h5ad` paths; sharded obs/var still stream there
because the alternative is to materialise the full obs table.

### Conversion: MTX ↔ SCX

Cell Ranger MTX conversion uses the `scx-mtx` crate (always-on, no HDF5 dependency):

```
  filtered_feature_bc_matrix/
  ├── matrix.mtx[.gz]             MTX → SCX
  ├── barcodes.tsv[.gz]     ────────────────▶  experiment.scx
  └── features.tsv[.gz]     scx-mtx::read       scx-mtx::write
                            COO→CSR, gzip       CSR→COO, gzip
                                                   │
  output_dir/                                      │
  ├── matrix.mtx.gz        ◀────────────────       │
  ├── barcodes.tsv.gz           SCX → MTX          │
  └── features.tsv.gz      ◀───────────────────────┘
```

On read, `scx-mtx::read` detects matrix orientation by matching the size-line
dimensions against the `barcodes.tsv` / `features.tsv` lengths. Cell Ranger's
native **features × barcodes** layout is transposed (via `scx_sparse::csr_to_csc`
reinterpretation) to the cells × genes CSR SCX stores; an already-cells×genes
matrix is kept; a square matrix defaults to Cell Ranger's layout with a warning;
and a dimension mismatch is a hard `MtxError::OrientationMismatch`.

### Query: filter → collect

```
  scx.open()           Read header + catalogs (< 4 KB)
      │
      ▼
  .filter_obs()        Parse predicate, validate against schema
      │
      ▼
  .collect()
      │
      ├── Catalog pushdown     Skip non-matching shards via CategoryBitset
      ├── Index pushdown       Row-level pruning within qualifying shards
      ├── Parallel decode      rayon: decode + decompress qualifying shards
      ├── Row filtering        Apply predicate to obs metadata rows
      ├── Gene projection      CSR column subset for selected genes
      └── Fused ops            normalize + log1p in single pass
      │
      ▼
  QueryResult          ScxCsr + Arrow obs/var → AnnData
```

### Training: SCX → GPU

```
  experiment.scx
      │
      ▼
  ShardShuffler        Permute shard order per epoch
      │
      ▼
  I/O stage (tokio)    Read shard groups → bounded channel
      │
      ▼
  Decode stage (rayon) Decompress → RowShuffler → HvgProjection
      │                scatter into dense batch → normalize + log1p
      ▼
  Batch channel        bounded channel → Python iterator
      │
      ▼
  PyTorch              batch["X"].to(device) → model.forward()
```

### Analysis: SCX → GPU Accelerators (rapids-singlecell)

When `device="gpu"` is set (or `device="auto"` with a CUDA GPU present),
the analysis pipeline routes through **rapids-singlecell** for UMAP, in-VRAM
PCA, and kNN, while streaming/randomized PCA and codec decode remain native.

```
  experiment.scx
      │
      ▼
  AccelRoute planner    Detect rapids (one-shot import probe)
      │                 → RapidsSinglecell | NativeGpu | FallbackCpu
      │                 SCX_FORCE_NATIVE_GPU=1 pins native paths
      │                 SCX_DISABLE_RAPIDS=1 forces no-rapids fallback
      ▼
  to_gpu_anndata()      Minimal-copy device handoff:
      │                 PyExperiment → GPU-resident AnnData
      │                 X as cupyx.scipy.sparse.csr_matrix
      ▼
  GPU PCA               Two paths:
      │                 (a) In-VRAM: rsc.pp.pca (via rapids)
      │                 (b) Streaming/randomized (native, >VRAM moat):
      │                     shard-streaming cuSPARSE SpMM + cuBLAS,
      │                     auto-routed by dataset size
      ▼
  GPU kNN               Two paths:
      │                 (a) In-VRAM: rsc.pp.neighbors (via rapids)
      │                 (b) Device-resident CAGRA (fused pipeline only)
      ▼
  GPU UMAP              rsc.tl.umap (via rapids-singlecell)
      │                 Native CUDA SGD kernel removed in Phase 3
      ▼
  AnnData               dtoh copy embeddings, kNN graph, UMAP coords
                        → adata.obsm["X_pca"], obsp, obsm["X_umap"]
```

Fused pipelines (`pca_neighbors_umap`, `pca_neighbors`) route end-to-end
through rapids when available, minimising device↔host copies.

Peak GPU memory: ~500 MB for 1M cells (dominated by Y and Q matrices
during streaming PCA). kNN and UMAP operate on the (n_obs × n_components)
dense embeddings, which are small relative to the full expression matrix.

---

## Integrity and Safety

| Mechanism | Scope | Algorithm |
|-----------|-------|-----------|
| Per-shard checksum | Shard content (after header) | BLAKE3 truncated to 64 bits |
| Per-section checksum | Each catalog entry | BLAKE3 (32 bytes) |
| Catalog checksum | All catalog bytes | BLAKE3 (32 bytes) |
| File checksum | Header field | BLAKE3 truncated to 64 bits |
| Atomic writes | New files | temp file → fsync → rename |
| Concurrent safety | Reads during append | Immutable fragments + header pwrite |
| Advisory locking | Concurrent writers | `flock()` via `fs4` crate |

---

## Error Handling

Each crate defines its own error type via `thiserror`:

| Crate | Error type | Covers |
|-------|-----------|--------|
| `scx-format` | `ScxError` | I/O, format validation, checksum failures |
| `scx-codec` | `CodecError` | Bitstream exhaustion, malformed encoded data |
| `scx-engine` | `EngineError` | Schema validation, predicate parsing, pipeline errors |
| `scx-ops` | `OpsError` | Append/delete/compact/merge/rollback failures |
| `scx-loader` | `LoaderError` | Pipeline errors, memory budget, configuration |
| `scx-cloud` | `CloudError` | Object store errors, auth failures, missing sections |
| `scx-mtx` | `MtxError` | MTX parse errors, missing sidecar files, I/O |
| `scx-accel` | `AccelError` | PCA, kNN, UMAP, differential expression computation errors |
| `scx-gpu` | `GpuError` | CUDA runtime errors, kernel launch failures, GDS errors |
| `scx-sparse` | `CsrError` | Invalid CSR dimensions |

Readers return errors (never panic) on malformed input, including bitstream
exhaustion, invalid magic bytes, and unsupported format versions.

---

## Environment variables

The canonical list of `SCX_*` environment variables read by the workspace.
Every entry below is read in Rust code; feature-area docs (e.g.
[gpu-setup.md](gpu-setup.md), [sharding.md](sharding.md)) link here rather than
re-describe them. All are optional — defaults apply when unset.

| Variable | Crate | Default | Effect |
|----------|-------|---------|--------|
| `SCX_FORCE_NATIVE_GPU` | `pyscx` | unset | Any non-empty, non-`0` value pins the surviving native GPU kernels instead of routing in-VRAM ops to rapids-singlecell. |
| `SCX_DISABLE_RAPIDS` | `pyscx` | unset | Any non-empty, non-`0` value treats rapids-singlecell as unavailable, forcing the CPU fallback (testing). |
| `SCX_PCA_COV_MEMORY_BUDGET` | `scx-accel` | `2 GiB` | Bytes; caps concurrent `n_vars × n_vars` f64 covariance-PCA accumulators (worker count). |
| `SCX_CSC_AUTO_OBS_THRESHOLD` | `scx-format` | `50000` | `CscPolicy::Auto` builds a CSC sidecar only when `n_obs ≥` this. |
| `SCX_CSC_AUTO_VARS_THRESHOLD` | `scx-format` | `5000` | `CscPolicy::Auto` builds a CSC sidecar only when `n_vars ≥` this. |
| `SCX_GPU_DE_GENE_CHUNK_SIZE` | `scx-gpu` | VRAM heuristic | Overrides the streaming GPU-DE gene-chunk size (rounded to a multiple of 64, min 64). |
| `SCX_CUVS_TRUST_LAYOUT` | `scx-gpu` | unset | `=1` downgrades a cuVS version/layout-compatibility mismatch from a hard error to a warning (kNN results may be wrong). |
| `SCX_DISABLE_CUDA_GRAPHS` | `scx-gpu` | unset | `=1`/`true` bypasses CUDA-graph capture at every call site. |
| `SCX_GPU_PROFILE` | `scx-gpu` | unset | Any non-empty, non-`0` value emits GPU profiling output. |
| `SCX_LOADER_PROFILE` | `scx-loader` | unset | `=1`/`true` emits ML-loader memory-budget profiling on drop. |

Benchmark-harness shell/Python variables (`SCX_WORK_DIR`, `SCX_DATA_DIR`,
`SCX_BENCH_HIGH_MEM_PARTITION`) are consumed by `benchmarks/` scripts, not by
the Rust crates — see [benchmarks/README.md](../benchmarks/README.md).

---

## Further Reading

- [format.md](format.md) — Binary format reference: header, catalogs, CSR shards, fragment/manifest, checksums
- [codec.md](codec.md) — Bit-level codec specification
- [api.md](api.md) — API reference for Rust, Python, and CLI
- [scanpy.md](scanpy.md) — Scanpy integration, backed mode, and accelerator usage
- [performance.md](performance.md) — Benchmark results and performance characteristics
- [testing.md](testing.md) — Test infrastructure and benchmarks
