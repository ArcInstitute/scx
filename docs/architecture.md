# SCX Architecture

SCX (Sparse Cell eXpression System) is a co-designed **file format**, **compression codec**,
**query engine**, and **ML data loader** for single-cell RNA-seq data. It replaces AnnData/h5ad,
scipy.sparse, and the scanpy I/O layer with a unified Rust-native stack.

This document describes the high-level architecture, crate structure, and data flow.
For the full binary format specification, see [format.md](format.md) and [codec.md](codec.md).
For the API reference, see [api.md](api.md).

---

## Crate Dependency Graph

The workspace contains 15 code crates plus two test-only members — `scx-testkit`
(the output-identity harness) and `scx-integration-tests` — 17 in total. Neither
test-only crate appears in the graph below: both are `publish = false` and are
reached only through `[dev-dependencies]`, so nothing shipped depends on them.
Dependencies flow bottom-up:

```
                        ┌──────────┐
                        │  pyscx   │  Python bindings (PyO3 + maturin)
                        └────┬─────┘
                             │ depends on all below
                        ┌────┴─────┐
                        │ scx-cli  │  CLI tool
                        └────┬─────┘
                             │ depends on all below
       ┌──────────────┬──────┼────────┬─────┬─────────────┬────────────-─┐
       │              │      │        │     │             │              │
┌──────┴──────┐ ┌─────┴────┐ │  ┌─────┴──────┐ ┌──────────┴─┐  ┌──────┴──────┐  ┌────┴──────┐
│ scx-engine  │ │ scx-ops  │ │  │scx-convert │ │ scx-cloud  │  │  scx-mtx    │  │ scx-accel │
│ query engine│ │ file ops │ │  │ conversion │ │ cloud ops  │  │ MTX I/O     │  │ PCA/kNN/  │
└──────┬──────┘ └─────┬────┘ │  └─────┬──────┘ └──────────┬─┘  └──────┬──────┘  └──┬─┬──────┘
       │              │      │        │                   │           │            │ │ [gpu]
       │    ┌─────────┘  ┌───┴────────┐                   │           │            │ │
       │    │            │ scx-loader │                   │           │            │ │
       │    │            │ ML loader  │                   │           │            │ │
       │    │            └───┬────────┘                   │           │            │ │
       │    │                │                            │           │            │ │
       └────┴────────────────┼────────────────────────────┴───────────┴────────────┘ │
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
| **scx-codec** | Compression codecs (standalone, no I/O) | `rice`, `forbp`, `delta_golomb`, `bitstream`, `dispatch` (driver), `codecs/*` (per-codec), `codec_id`, `guards`, `raw` |
| **scx-sparse** | CSR/CSC matrix types with scipy-compatible dtypes | `csr` (`ScxCsr`), `csc` (`ScxCsc`), `transpose` (streaming CSR→CSC), `convert` (CSR ↔ dense) |
| **scx-format** | Pure on-disk layout/spec — no `std::fs`, no `memmap2` (the surface conformance vectors verify against) | `header`, `catalog`, `catalog_view`, `shard`, `section`, `modality`, `codec_select`, `provenance`, `csc_policy`, `error`, `checksum` |
| **scx-format-io** | Runtime reader/writer, backed/streaming access, shard codec dispatch, sidecars; re-exports the full `scx-format` surface | `reader`, `writer`, `backed`, `shard_decode`, `shard_source`, `encoder`, `csc_sidecar`, `bitmap`, `deletion_vectors`, `mem`, `arrow_compat` |
| **scx-ops** | File lifecycle operations | `append`, `append_from_reader` (streaming SCX→SCX), `delete`, `compact`, `optimize` (in-place re-encode + row-group-frame → v4), `merge`, `rollback`, `flock` |
| **scx-engine** | Lazy query engine with predicate pushdown | `pipeline`, `predicate`, `pushdown`, `projection`, `fused_ops`, `index`, `collect` |
| **scx-loader** | ML training data loader (triple-buffered) | `pipeline`, `io_stage`, `decode_stage`, `shuffle`, `projection`, `normalize`, `batch`, `python` |
| **scx-cloud** | Cloud access operations (S3, GCS, Azure) | `backend`, `cloud_optimize`, `explode`, `pack`, `pull`, `push`, `coalesce`, `cloud_reader` |
| **scx-mtx** | Matrix Market (MTX) I/O (always-on, no feature gate) | `read` (COO→CSR, TSV parsers, gzip), `write` (CSR→COO, gzipped output) |
| **scx-convert** | h5ad / h5mu / 10x / CellBender ↔ SCX streaming conversion (opt. `hdf5` feature). Note: depends on **scx-ops** (`sort` for sort-on-convert, `external_layer` for the CellBender seam type) — an edge the ASCII graph above omits. | `pipeline/` (entry points, coordinators, shard + mapping writers — see the module table below), `options` + `budget` (direction-specific options; what a memory budget buys), `h5ad/` (`read`, `write`, `columns`, `column_stream`, `categorical`, `uns`, `stream_write`, `dense_stream`, `csc_stream`), `h5mu` (multimodal pipeline), `tenx_read`, `cellbender` (remove-background output reader), `export_filter` (streaming `min_counts` obs mask) |
| **scx-accel** | Rust-native analysis accelerators (opt. GPU via `gpu` feature) | `route` (accelerator execution planner + rapids probe), `pca` (streaming/randomized SVD, auto-routed; in-VRAM routes to `rsc.pp.pca`), `neighbors` (HNSW kNN; in-VRAM routes to `rsc.pp.neighbors`), `umap` (routes to `rsc.tl.umap`), `hvg` (streaming `seurat_v3`; extra flavors route to `rsc.pp.highly_variable_genes`), `fused` (fused `pca_neighbors_umap` / `pca_neighbors` pipelines via rapids), `diffexp` (Wilcoxon rank-sum with pre-ranking), `leiden` (Rust-native CPU + cuGraph GPU), `harmony` (Harmony2 batch integration — soft k-means + ridge regression), `lisi` (exact-kNN Local Inverse Simpson Index), `pseudobulk`. GPU dispatch when `gpu` feature enabled; rapids-singlecell detected at runtime (not a pip extra). |
| **scx-gpu** | CUDA-accelerated codec decoding, GPU analysis, and GPU interop | `rice_decode`, `forbp_decode`, `sparse_to_dense`, `cusparse` (SpMM), `cusolver` (QR), `curand` (random matrix), `gpu_pca`, `gpu_knn` (CAGRA, device-resident fused path), `gpu_harmony` (distance / softmax+penalty / L2-normalize / batched correction kernels), `gpu_preprocess` (fused normalize+log1p), `gpu_matrix_source` (unified `GpuMatrixSource` capability trait over the row-major `GpuShardSource` (CSR) and column-major `GpuCscShardSource` (CSC) device shard sources, with G3-shaped pinned-ring staging), `gds` |
| **scx-cli** | Command-line interface | `convert`, `info`, `validate`, `query`, `append`, `delete`, `compact`, `optimize`, `merge`, `rollback`, `cellbender-import`, `obs-import`, `doublet-import`, `benchmark`, cloud ops |
| **pyscx** | Python bindings via PyO3 | `experiment`, `anndata`, `ops`, `query`, `cloud`, `backed`, `accel`, `preprocess`, `lazy_transform`, `projected_agg` |
| **rscx** | R bindings via extendr | Seurat v5 + SingleCellExperiment interop, pipe-friendly query API |
| **scx-testkit** | Test-only (`publish = false`, dev-dependency): asserts a refactor did not change what SCX writes. Hashes **per catalog section** rather than the whole file, because every mutating write path stamps `SystemTime::now()` into `Provenance` (and `file_checksum` covers it), so a whole-file BLAKE3 differs between two runs of the same op. Covers each section's payload **plus** its `ShardStats` (`row_start`, `nnz`, `column_stats`) and the catalog's `data_generation` / `csc_build_generation` — those live in the catalog, not in any section's bytes, and readers act on them, so a payload-only hash would call two behaviourally different files equal. `Strictness::Layout` additionally pins section offsets and the header's catalog pointers. The 4096-byte root catalog is deliberately excluded: it has no production readers and every writer rebuilds it. | `digest` (`FileDigest` / `Strictness::{Content,Layout}` / `assert_matches_golden`), `ab` (`OpDigestManifest` — several files' digests under one label each, dumped to JSON in one worktree and asserted against in another; plus the run-it-twice-across-a-second helpers), `fixtures` (`mixed_codec_file` — one unframed, one row-group-framed integer and one `Pcodec` float shard, so the digest covers every encoder path) |

> [!NOTE]
> `scx-loader` has its own streaming-optimized gene projection and fused
> normalization for the hot-path requirements of ML training. It depends on
> `scx-accel` for one thing — the v4 PFlog α estimator (`estimate_alpha`) used
> when a `pflog=True` loader is constructed with `pflog_alpha=None` — and so
> **transitively** pulls in `scx-accel`'s dependencies (`scx-engine`, `faer`,
> HNSW, …). This edge is cycle-free (`scx-accel` depends only on
> `scx-format-io` + `scx-sparse`). `scx-cloud` does **not** depend on
> `scx-loader` — they are siblings; `scx-cloud` reuses `scx-engine` for
> predicate parsing (selective pull).
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
> Leiden (Rust-native CPU + cuGraph GPU), DE Wilcoxon rank-sum/pdex (CSC/CSR-direct),
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
float data routes to Pcodec. LZ4+shuffle is available via `codec="lz4"`, and is
also auto-selected by the *per-modality* heuristic
(`select_codec_for_modality`) for non-binary integer ATAC counts; the
modality-blind heuristic above never picks it.

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
2. **Bounded staging working set.** `CscBuilder` routes each nonzero into a
   column bucket and spills past the budget's bucket share, so the staging is
   bounded by a maintained counter rather than by a chunk-width estimate, and
   the op holds one decoded source shard instead of all of them.

   `--memory-limit` still does **not** cap the whole op. The source shard's
   re-encode, the writer, and the buckets that are still resident during the
   emit sit outside the bucket share. Measured before/after on one node against
   a declared 4 GiB budget: `tabula_sapiens_100k` 3580 → 3775 MB (0.92x),
   `census_500k` 8998 → 5512 MB (1.35x), `census_1m` 15411 → 7138 MB (3.76x →
   **1.74x**). Closing the rest is a separate open item; the re-encode is the
   next piece to go.

Pass `--csc-cols-per-shard 0` for no cap (single CSC shard).

### Key types and data flow

```
                   CREATION                                    CONSUMPTION
 ─────────────────────────────────────────   ──────────────────────────────────────

 scx-sparse/src/csc_builder.rs              scx-format-io/src/backed/csc.rs
 ┌──────────────────────────────────┐        ┌─────────────────────────────────┐
 │ CscBuilder                       │        │ BackedCscIndex                  │
 │   push_shard(row_start, &ScxCsr) │        │   shard_ranges: Vec<(col_start, │
 │   - routes nnz into column       │        │     col_end, sorted_idx)>       │
 │     buckets, spilling past the   │        │   shards_for_col_range(lo,hi)   │
 │     budget; finish() emits       │        │     → Vec<usize>  (binary srch) │
 └────────────┬─────────────────────┘        └────────────┬────────────────────┘
              │                                           │
              ▼                                           ▼
 scx-format-io/src/writer.rs                   ┌─────────────────────────────────┐
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

**`BackedCscIndex`** (`scx-format-io/src/backed/csc.rs`): A sorted vector of
`(col_start, col_end, sorted_shard_idx)` ranges built from the catalog at
construction time. Provides O(log n) column lookups via `partition_point`:
`shard_for_col(col)` for single-column lookups and
`shards_for_col_range(c_lo, c_hi)` for range queries.

**`BackedCscReader`** (`scx-format-io/src/backed/csc.rs`): The primary CSC
consumer. Wraps an `ScxReader` + `BackedCscIndex` + a `ShardCache` of decoded
`ScxCsc` shards — the same byte-budgeted, singleflighted cache the CSR and dense
readers use. It was a separate count-only implementation with neither, on the
reasoning that CSC workloads access shards in column-range order with limited
reuse; the reuse argument holds, but it did not justify a third copy of the
eviction loop, and concurrent readers of one cold shard each decoded their own.
`BackedCscReader::new` still opens count-only (`usize::MAX` bytes);
`with_byte_budget` bounds it. Key method: `read_csc_columns(col_range)` skips non-overlapping
shards, `col_slice`s partial-overlap shards post-decode, and concatenates
results. Implements `ColumnShardSource`.

**`ColumnShardSource`** (`scx-format-io/src/shard_source.rs`): The trait that
abstracts CSC access. Both `BackedCscReader` (raw on-disk) and pyscx's
`LazyShardSource` (transform-aware) implement it. All `scx-accel` CSC kernels
are generic over this trait — no concrete type dependency.

**`PreferFormat` + `require_csc()`** (`scx-accel/src/csc/dispatch.rs`):
Explicit dispatch at the kernel boundary. There is intentionally no `Auto`
variant *in the enum* — by the time a kernel is called the choice has been
made. Callers pass `prefer_format="csc"` through pyscx kwargs; `require_csc()`
either returns the `ColumnShardSource` or a clean error explaining why CSC is
unavailable. `"auto"` lives one level up, as a pyscx-side policy
(`accel::de::resolve_de_format`, the default for `rank_genes_groups` and
`pdex_ref`): it probes the dataset's CSC capability per call and resolves to
`Csr` or `Csc` before crossing the boundary.

### Creation pipeline

CSC sidecars are created by a one-pass bucketed transpose in
`scx-sparse/src/csc_builder.rs`:

1. Each CSR shard is decoded and pushed into a `CscBuilder` **in row order**,
   one at a time. The builder routes every nonzero into a contiguous column
   bucket and drops the shard; buckets stay in RAM until they cross the
   budget's bucket share, then spill whole blocks to a `SpillStore`.
2. `finish()` returns a `CscEmitter` whose plan — the exact column range and
   nnz of every shard it will produce — is known before a byte is read back,
   because the per-column counts were maintained during the push.
3. Each emitted shard is written via `ScxWriter::write_csc_shard()` as
   `section_type = CscShard(5)` with independent per-shard codec selection.

Total work is `2 * nnz`, against the `2 * nnz * n_chunks` of the chunked
transpose this replaced (which rescanned every nonzero of every shard, twice,
per column chunk). Shard **boundaries** are unchanged: `shard_cols` still comes
from `compute_chunk_cols_with_cap`, so the emitted layout is byte-identical and
only the cost of producing it moved.

A caller that already holds the whole CSR — the eager h5ad/h5mu/`from_anndata`
ingest paths — uses `ResidentCscSource` instead, which scatters straight out of
the resident shards rather than making a second copy of them in buckets. Both
sources feed one `csc_sidecar::emit_csc_shards` writer loop.

A writer that is producing the X shards itself — streaming ingest and the
rewrite ops — skips step 1's decode: `ScxWriter::enable_csc_sidecar` feeds
every X CSR shard it writes (layers and `adata.raw` excluded) to the builder as
it goes, and `emit_csc_sidecar()` writes the CSC shards right after X. The
output is byte-identical to building the sidecar over the finished file, with
no second read and no staged copy (`scx-format-io/src/csc_sink.rs`).

Entry points:

| Entry point | When |
|-------------|------|
| `scx build-csc` | Post-hoc addition to an existing file |
| `scx convert --csc=always` | During h5ad/h5mu → SCX conversion |
| `pyscx.from_anndata(csc="always")` | During Python-side conversion |
| `--csc carry\|always\|off` on rewrite ops | Same-pass rebuild in compact/merge/optimize/sort/subset (`carry` is the default) |
| `append --rebuild-csc` | In-place rebuild after an append |

Measured throughput, from the `build_csc` rows of the current benchmark
baseline (`benchmarks/comprehensive/results/baselines/LATEST`, `scx_auto`,
`--memory-limit 4G`): 0.22 s on `pbmc3k` (2.3M non-zeros), 2.6 s on `pbmc10k`
(24.8M), 19.4 s on `smartseq2` (131M), 29.3 s on `tabula_sapiens_100k` (195M),
201 s on `census_500k` (747M) and **1308 s on `census_1m`** (1.40B). Cost tracks
non-zeros, not cells.

Those are the figures for the **chunked transpose** `CscBuilder` replaced, which
rescanned every nonzero of every shard twice per column chunk. The saving
therefore grows with the chunk count, and a branch A/B measured 1.40x on
`pbmc10k` (7 CSC shards) up to 9.36x on `census_1m` (173). Those ratios are not
quoted as baseline figures here — see
[benchmark_manifest.md](benchmark_manifest.md): a number in a user-visible doc
needs a capture whose `git_sha` is an ancestor of `main`, which a PR-branch
capture never becomes — so this table keeps the last promoted capture until the
next one lands.

### Multimodal support

`BackedCscReader::for_modality(reader, modality_id, cache_shards)` scopes the
CSC index and LRU cache to a specific modality, filtering by
`(SectionType::CscShard, modality_id)`. Each modality gets its own cache to
avoid thrashing under interleaved access (e.g. totalVI touching RNA + ADT in
the same step). `BackedCscReader::for_layer()` similarly scopes to a layer's
CSC sidecar.

### Mutating ops and CSC lifecycle

Mutating operations change the row layout, column index space or `nnz`,
making existing CSC `indices` arrays reference stale rows/columns, so the
input's sidecar is never copied. The rewrite ops (`compact`, `merge`,
`optimize`, `sort`, `subset`) **carry it by default**: they build a fresh one
from the output's X in the same pass iff an input had one (`--csc
carry|always|off`, `scx_ops::CscCarryOptions`). A multimodal input's sidecars
are dropped with a warning (the builder is single-modality). `append` still
drops the sidecar with a `log::warn!` message; `append --rebuild-csc` rebuilds
it in place afterwards.

`scx upgrade` differs — it carries the input's CSC sidecars rather than
rebuilding them, re-emitting them through `catalog.csc_shards_sorted()` into the new file.

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

### Writer (`scx-format-io/src/writer.rs`)

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

### Reader (`scx-format-io/src/reader/`)

| Module | Holds |
|---|---|
| `mod.rs` | the `ScxReader` struct, accessors, `section_bytes` (the single mmap-slicing chokepoint), madvise helpers, freshness, modality lookups |
| `open.rs` | the two constructors |
| `metadata.rs` | Arrow IPC, sharded-metadata assembly, dictionary unification — reaches the mapping only through `section_bytes` |
| `matrix.rs` | CSR/CSC shard decode, whole-matrix assembly, layers, `adata.raw` |
| `integrity.rs` | checksum and canonical-CSR validation |
| `filtered.rs` | deletion-vector and detection-bitmap reads (`deletion-vectors` feature) |

Opens a file via `mmap` and validates magic/version/checksums:

```
1. mmap the file (MADV_NORMAL default), with the path echoed in any I/O error
2. Parse 256-byte header
3. Parse root catalog (offset 256, cursor bounded to ROOT_CATALOG_MAX_SIZE)
4. Parse full catalog (from header's full_catalog_offset)
5. Parse the ModalityTable if the header points at one
6. Individual sections accessed via catalog offsets
```

`open_with_shared_catalog` reuses a sibling reader's `Arc<FullCatalog>` and runs
the same steps except 4, cross-checking `manifest_sequence` instead. `open()`
finishes by backfilling v1 `col_start`/`col_end`, which the shared path cannot
do — it holds an `Arc` it cannot mutate — so it *verifies* that backfill has
happened rather than assuming it: a v1 catalog whose row-major entries still
report `col_end == 0` is refused. Note `reconcile_v1_csr_col_range` does not bump
`catalog_version`, so a reconciled catalog still reports 1; refusing on the
version instead of the invariant would reject every v1 file `open()` accepts.

All section access is via offset+length from the catalog — no sequential scanning.

**Whole-matrix assembly.** One `assemble_row_major` serves `X`, a modality's
`X`, layers, and `adata.raw`; the raw matrix passes its own `n_cols` and its own
empty-result shape. `plan_row_major_layout` and `check_decoded_lengths` hold the
per-shard stats arithmetic and the decoded-vs-catalog checks, and
`typed_read.rs`'s narrowing assembler calls both, and runs on the same
`RowMajorStrategy`. Output buffers are carved into per-shard exclusive slices
with `split_at_mut` before decode, so the parallel path needs no `unsafe`; the
typed assembler carves the same way through `IndexBuffer::chunks_mut` /
`ValueBuffer::chunks_mut`, which resolve the runtime dtype once for the carve;
the per-shard fill matches the slice enum again, as the range-taking form always
did.

**madvise hints** (Unix only, `#[cfg(unix)]`):
- `MADV_SEQUENTIAL` on the shard byte range during `assemble_row_major()` — tells kernel to readahead aggressively for full reads
- `MADV_WILLNEED` on next N shards in `BackedCsrReader::read_shard_cached()` — prefetches upcoming shards after a cache miss
- `MADV_DONTNEED` after `read_shard_uncached()` — releases page cache for decoded shards during streaming aggregation (67% RSS reduction on 1M cells)

### Backed readers (`scx-format-io/src/backed/`)

| Module | Holds |
|---|---|
| `mod.rs` | the re-exports every `backed::<name>` import path resolves through, `scatter_block_index_enabled`, `ROW_RANGE_WINDOW_DIVISOR` |
| `index.rs` | `BackedCsrIndex` + `ShardEntryLite` — the row-range → shard binary search, shared by the CSR and dense readers |
| `cache.rs` | `CacheMetrics`, `SizeHint`, `WeightedLruCache`, `ShardCache` |
| `csr.rs` | `BackedCsrReader` and its `ShardSource` impl |
| `aggregate.rs` | the native shard-by-shard statistics kernels, a second `impl BackedCsrReader` |
| `csc.rs` | `BackedCscIndex`, `BackedCscReader`, the CSC sidecar freshness guard |
| `dense.rs` | `BackedDenseReader` — row gather over an `obsm` mapping |

**One cache for all three readers.** `ShardCache<K, V>` is a byte-budgeted LRU
(`WeightedLruCache`) plus a singleflight table, generic over the decoded payload
via `SizeHint` — `ScxCsr` and `ScxCsc` by their component sizes (`indptr×8 +
indices×4 + data×4`, the model `IndexPlanLoader`'s memory-budget auto-tune also
uses), `DenseShard` by `RecordBatch::get_array_memory_size()`. `SharedShardCache`
is the CSR instantiation, keyed `(file_id, shard_id)` so several readers of one
multi-file run share a budget; CSC and dense key by shard index alone.

`ShardCache::get_or_decode` is the only place that takes both the `in_flight` and
`cache` locks, and it takes them in that order — which is what makes the lock
ordering a property of the code rather than a contract three separate
transcriptions each had to honour.

### Catalog representations (`FullCatalog`, `CatalogView`)

Both **byte parsers** are one parser. `scx-format/src/catalog_cursor.rs` holds the single walk over the catalog's entry list; it yields `RawEntry`, whose `section_type_raw` is an unresolved `u8` and whose `name_bytes` / `stats_bytes` are borrowed and undecoded. The two representations below differ only in what they materialise from it, and each validates exactly what it materialises. That shape is load-bearing rather than tidy: when two hand-written parsers existed they drifted on the order of two steps, and `FullCatalog` decoded an entry's stats before resolving its section type — so a file carrying a future section type with a short stats blob failed to open at all.

Note the scope: this is about parsing *bytes*. In production today only `FullCatalog` is built from bytes — every `CatalogView` comes from `CatalogView::from_full` in the backed-reader constructors, and `read_from_bytes` has no production caller (see the bullets below).

`ScxReader` exposes two catalog representations that trade completeness against per-entry cost:

| Type | Location | When to use |
|------|----------|-------------|
| `FullCatalog` | `scx-format/src/catalog.rs` | Validation, mutation (append/compact/delete/rollback), metadata inspection, the writer round-trip, and all `scx-ops` / `scx-engine` tooling that needs the full per-entry record (`name: String`, `checksum: [u8; 32]`, `ShardStats` with all scalar + indexed-column fields). This is what `ScxReader::open` parses eagerly; the value is held internally as `Arc<FullCatalog>`. |
| `CatalogView` | `scx-format/src/catalog_view.rs` | Reader/open hot path. Drops the 32-byte `checksum`, the diagnostic `value_*` fields, and `col_*` on row-major shards; drops the `name: String` on `CsrShard` / `CscShard` / `ObspCsrShard` entries (looked up by `(section_type, major_start)` instead). Stats collapse to `ShardStatsLite { major_start, major_end, nnz }` — dispatched on `section_type` so the consumer reads the right axis without branching per access. Also carries the v4 `data_generation` / `csc_build_generation` counters, so a freshness check made through the view cannot read a stale CSC sidecar as `0 == 0`. |

**Which reader paths use which representation:**

- `ScxReader::open` and `ScxReader::open_unchecked` parse and retain `Arc<FullCatalog>` — the tooling-facing path. Compatibility-preserving for `scx-engine` predicate pushdown (needs `column_stats`), `scx-ops` mutations (needs per-entry checksums), and any caller of `ScxReader::catalog()`.
- `ScxReader::open_with_shared_catalog(path, Arc<FullCatalog>)` skips the catalog parse entirely; used by `pyscx::to_anndata_backed` to amortise a single parse across the N+3 sibling readers (main + X CSR + CSC sidecar + per-layer) it opens against the same file. The reused `Arc<FullCatalog>` is byte-identical to what an independent open would produce.
- `BackedCsrReader::new*` constructors (and `for_modality` / `new_for_layer_*`) build a `CatalogView` from the reader's `FullCatalog`, then walk the view's `csr_shards_sorted` / `csr_shards_for_modality` / `layer_csr_shards_sorted_with_prefix` helpers to construct a `Vec<ShardEntryLite>` per-shard table. They never clone `FullCatalogEntry`. Per-shard retained footprint drops from ~250 B → 32 B (with alignment), which compounds at the ~16K entries census-scale shards produce.
- `BackedCsrIndex::from_view_sorted(&[&CatalogViewEntry])` zips the major-axis range into `ShardRange` rows in a single pass over the pre-sorted view; it never re-walks the catalog.
- `scx-engine` predicate pushdown reads `ShardStats` via `FullCatalog`, which decodes `column_stats` eagerly for every entry. A caller that wants the cold-path deferral instead holds `RawEntry.stats_bytes` — the undecoded payload, borrowed out of the catalog bytes — and decodes on demand. (A `LazyShardStats` type existed for this and had no production caller; it was removed rather than kept as a third representation of one blob.)

**Fork safety:** `Arc<FullCatalog>` has no interior mutability, and neither does `CatalogView` (the reader paths above build views as stack temporaries; an `Arc<CatalogView>` is supported by the type but not a pattern anything currently uses). `FullCatalog::reconcile_v1_csr_col_range` mutates entries once during `read_from` *before* the `Arc` wrap; after that point every reader path treats the catalog as frozen. A `fork()` from a Python DataLoader worker COW-duplicates the parent's catalog into each child — no shared mutex, no shared singleflight table, no atomic refcount contention across processes. See [docs/multithreading.md § Fork safety](multithreading.md#fork-safety).

### Sharded obs/var metadata reader APIs

Files produced by streaming `scx merge`, `scx append`, `pyscx.from_anndata`, `scx optimize --shard-obs` and every `scx convert` / `pyscx.from_h5ad` / `pyscx.from_h5mu` / `pyscx.from_mtx` ingest (all on the same `n_obs > shard_target_rows` boundary; the convert paths take `--shard-obs off|auto|always`) store obs metadata as row-sharded Arrow IPC sections (types 24/25) instead of a single monolithic section. Var is sharded only by merge / append / `from_anndata` — convert ingest always writes one `var_metadata` section. `ScxReader` exposes a parallel set of shard-aware accessors alongside the legacy single-section API:

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
| **optimize** | Re-encode + canonicalize CSR shards → row-group-frame, stamp v4 | Entire new file (atomic rename) |
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
  within qualifying shards. Disable per pipeline with
  `QueryPipeline::rowset_pushdown(false)` (or the `SCX_DISABLE_ROWSET_PUSHDOWN`
  environment variable, which seeds the default at construction) to force the
  legacy full-decode path. This is a **diagnostic** knob: both paths must return
  the same rows for every predicate, and `scx-engine/tests/rowset_differential.rs`
  asserts exactly that.
- **One tree walk, two interpretations:** both evaluators recurse through a
  single `walk` over the `PredicateAlgebra` trait
  (`scx-engine/src/predicate.rs`). `MaskAlgebra` evaluates leaves as Arrow
  comparison kernels into a Kleene `BooleanArray`; `RowSetAlgebra` resolves them
  straight from the predicate index into an `Option<RowSet>`, where `None` means
  *residual*, not *no rows*. They share the descent only — each keeps its own
  leaves and its own combinator kernels, since `or_kleene` and `RowSet::union`
  are not interchangeable. Agreement between the two is enforced by the
  differential oracle, not by the type system.
- **Operation fusion:** `normalize(1e4) + log1p()` → single fused CSR row scan
  (`scx-engine/src/fused_ops.rs`).
- **Parallel collection:** Qualifying shards are decoded and filtered in parallel
  via rayon (`scx-engine/src/collect/`).

### Query execution (`scx-engine/src/collect/`)

Three layers, bottom up. The split exists so that `mask` is the **only** module
that decides row-set-pushdown vs legacy-full-decode; `execute` consumes a
`MaskResult` without knowing which arm produced it, which makes that fork an
interface rather than a branch buried mid-file.

| Module | Holds |
|---|---|
| `mod.rs` | the re-exports every `collect::<name>` import path resolves through, and `rowset_pushdown_disabled_by_env` |
| `retry.rs` | `par_map_with_shard_retry` — generic resilient parallel map, nothing query-specific |
| `rows.rs` | `filter_csr_rows` / `filter_csr_rows_owned` — row selection inside one decoded shard; the owned form returns an all-kept shard by move |
| `native.rs` | the fused native decode + filter + project for the dtype-selected (`collect_typed`) route |
| `plan.rs` | `ExecutionPlan`: Level-1 catalog shard pruning and the category dictionaries both levels share |
| `mask.rs` | the row-set fast path, the legacy full-decode fallback, and `compute_mask`, the fork between them |
| `execute.rs` | `execute` / `count` / `exists`, `plan_and_mask` and `materialize` |

### Predicate indexes (`scx-engine/src/index/`)

| Module | Holds |
|---|---|
| `mod.rs` | the nine on-disk structs, `IndexKind`, and the re-exports every `index::<name>` path resolves through |
| `wire.rs` | the [format.md](format.md) binary layout — `write_to` / `read_from`, v1 and v2 |
| `lookup.rs` | query-time lookups: what the row-set fast path asks an index |
| `build.rs` | eager construction, plus the categorical / numeric primitives `stream` shares |
| `stream.rs` | `ObsPredicateIndexBuilder`, for callers that never hold the whole obs `RecordBatch` |
| `derive.rs` | per-shard catalog `column_stats` (Level 1) derived from a finished index |
| `diagnostics.rs` | CLI message rendering, index presets, and `resolve_csc_policy` |
| `values.rs` | Arrow value extraction and type classification, the leaf layer |

`diagnostics.rs` holds code that reads no index at all. It lives in `scx-engine`
because `pyscx` depends on `scx-engine` unconditionally but on `scx-convert`
only under the `hdf5` feature, so a helper shared by the `scx convert` CLI and
the Python conversion entry points cannot live in `scx-convert`. Moving it
within the crate is fine; moving it out is not.

### `SectionReader` — local + cloud unification

`QueryPipeline` is generic over a `SectionReader` trait that abstracts
how catalog sections are fetched. Two implementations ship today:

| Reader | Section fetch | Construction |
| --- | --- | --- |
| `ScxReader` (`scx-format-io`) | mmap / `pread` over a local `.scx` file | `QueryPipeline::open(path)` |
| `CloudSectionReader` (`scx-cloud`) | `object_store` range reads over `gs://` / `s3://` / `az://` / exploded `.scxd/` directories | `QueryPipeline::from_reader(reader)` |

`Experiment.query()` opens a local mmap-backed pipeline;
`CloudExperiment.query()` opens a cloud-backed pipeline (see
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
| `index_plan.rs` | `IndexPlanLoader` — plan-driven paired-batch reader (sibling row-source model; see above). Its `IndexPlanIter` is a thin adapter over `plan_engine`'s iterator, not a second implementation |
| `plan_engine.rs` | `PrefetchEngine` + `PlanPrefetchIter` — the one plan-driven prefetch iterator (plan-pull thread, bounded lookahead, per-plan shard prefetch, `IterMetrics`). Both the paired and the sparse cell-set loaders run on it; each exposes it through a thin named adapter (`IndexPlanIter`, `SparseCellSetIter`) because the engine iterator's closure types are unnameable |
| `python/` | `TrainingDataset` + `IndexPlanDataset` PyO3 classes (both implement `__iter__`/`__next__`), one module per binding |

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
exp = pyscx.open("experiment.scx")    # → Experiment (lazy handle)
adata = exp.to_anndata()              # → AnnData (zero-copy CSR + Arrow→pandas)

# Write
pyscx.from_anndata(adata, "output.scx", codec="auto")
pyscx.from_10x("matrix.h5", "output.scx")   # materializes — see note below
pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "output.scx")

# Export to Cell Ranger MTX directory
pyscx.to_mtx("output.scx", "/path/to/mtx_dir")
```

`from_10x` reads through `scanpy.read_10x_h5` and hands the in-memory AnnData to
`from_anndata`, so peak memory scales with the whole matrix. For a *raw*
all-droplet `raw_feature_bc_matrix.h5`, use `scx convert --from 10x`, which
streams by default — see [docs/scanpy.md § From 10x HDF5](scanpy.md#from-10x-hdf5).


The `to_anndata()` path is **zero-copy** for the expression matrix — `ScxCsr`'s
`i64/i32/f32` arrays are handed directly to scipy via numpy buffer protocol.
Arrow metadata goes to pandas via pyarrow. The `container` / `data_dtype` /
`index_dtype` / `allow_lossy` read kwargs opt out of this default: they
materialize `X` into a chosen container (scipy CSR or dense ndarray) and numeric
dtype via a read-then-convert (with a fail-loud cast gate), trading the zero-copy
move for a narrower / dense output. See
[docs/api.md § Container and dtype materialization](api.md#container-and-dtype-materialization).

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
scx append atlas.scx new_batch.scx
scx delete experiment.scx --filter "is_doublet == True" --dry-run
scx compact experiment.scx compacted.scx
scx optimize experiment.scx optimized.scx   # re-encode + row-group-frame → upgrade to v4
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

The conversion code lives in the **`scx-convert`** crate (opt-in `hdf5`
feature; see the crate summary above). Both `scx-cli` and `pyscx` depend on
it; the previous in-line `scx-cli/src/convert/` module was extracted
when streaming conversion landed so `pyscx` could share the pipeline
without depending on the binary-only `scx-cli`.

### Conversion pipeline (`scx-convert/src/pipeline/`)

`pipeline.rs` was 3727 lines, the largest production file in the workspace, holding
nine concerns with no section markers and no module doc comment — line 1 was a bare
`use`. The split exists so that the dependency runs one way: the entry points
sequence, and each thing they sequence lives in a module that does not know about
them.

| Module | Holds |
|---|---|
| `mod.rs` | the re-exports for every `pipeline::<name>` path with a **production** caller; of five without one, three were removed and two narrowed to `cfg(test)` for the attached test files (see below). Plus `write_ingest_obs` and the `BitmapPolicy` / `CscPolicy` / `IngestOptions` / `codec_selection_json` re-exports |
| `error.rs` | `ConvertError` and its `parallel_drain::DrainFailure` conversion |
| `bitmap.rs` | detection-bitmap `auto` eligibility (density, `n_vars` cap, size budget) and the per-shard writer |
| `threads.rs` | reader-thread and queue-depth derating — the *consumer* of `budget.rs`, not a second copy of it |
| `index.rs` | when to build a convert-time predicate index, and its per-column outcomes → `ConvertWarning` |
| `entry.rs` | the eager entry points: `h5ad_to_scx`, `tenx_to_scx`, `scx_to_h5ad` |
| `entry_streaming.rs` | `h5ad_to_scx_streaming` and `tenx_to_scx_streaming`, the bounded-memory sequencers, and `convert_then_sort_grouped` |
| `coordinator.rs` | the four shard coordinators and `encode_one_shard_worker`; the pool, channel and reorder buffer are in `parallel_drain` |
| `shards.rs` | X / `raw/X` / layer / CSC-sidecar shard writers |
| `mappings.rs` | `obsm` / `varm` / `obsp` / `varp` row-sharded section writers |

Two things deliberately are **not** submodules here, for different reasons.
`IngestOptions` and `ExportOptions` live in `scx-convert/src/options.rs`, which is
`hdf5`-gated exactly like `pipeline/` — it is a sibling because keeping the
ingest/export pair in one file is the point of ORG-11.16-3, not because the
default build names it. What a memory budget buys lives in
`scx-convert/src/budget.rs`, which **is** ungated: its arithmetic invariants run in
the default test job, and `csc_sidecar_bytes` has to stay reachable from the
non-`hdf5` `pyscx.from_anndata` sidecar path. (`mtx_pipeline.rs` names neither.)

The carve also **narrowed** the internal surface, which is worth stating because
it is not what "pure move" implies. Five items were reachable in production at
`crate::pipeline::*` before and are not now: `BitmapBuildOutcome`,
`maybe_build_bitmap_shard` and `ensure_shard_fits_budget` have no re-export, and
`compute_shard_row_ranges` / `process_predicate_index_outcomes` are re-exported
only under `cfg(test)`. No caller outside `pipeline` used any of them, so nothing
broke — but the `pub`/`pub(crate)` *declaration* set being unchanged is not the
same claim as every old path still resolving, and only the first was measured.

### h5ad export (`scx-convert/src/h5ad/`)

The export side was one 2908-line `write.rs`. Two thirds of it was the dataframe
column writer, which is why `columns` / `column_stream` / `categorical` are three
modules rather than one: the layout pre-pass and the write pass are separate
passes over the same shards, and the categorical decode is the only thing both
need.

| Module | Holds |
|---|---|
| `write.rs` | the SCX → h5ad drivers, `/X`, `/raw`, `/obsm`, `/obsp`, `/varp` |
| `columns.rs` | the dataframe layout / schema pre-pass: nullable columns, any-shard-dictionary promotion, used-category scan |
| `column_stream.rs` | `write_dataframe_group_from_shards` and the nine column encodings, plus the whole-batch driver over it |
| `categorical.rs` | Arrow dictionary → h5ad categorical: decode, widen, cross-shard vocabulary (`CatAccum`) |
| `uns.rs` | `/uns`: JSON → HDF5, including the `__scx_type__` envelope decode |
| `read.rs` | h5ad ingest reads: dataframes, `X` shape, `uns`, mapping shards |
| `stream_write.rs` | the streaming SCX → h5ad / h5mu writer and its rayon shard-decode pool |
| `stream.rs`, `dense_stream.rs`, `csc_stream.rs`, `csc_transpose.rs` | the three source layouts and the external-memory CSC → CSR transpose |

**Streaming ingestion** (`scx convert --stream`, `pyscx.from_h5ad`,
auto-routing on backed AnnData): `scx_convert::h5ad_to_scx_streaming`
— and `tenx_to_scx_streaming`, which reuses the same reader over a 10x
`/matrix` group (a CSC of genes×cells *is* a CSR of cells×genes) —
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
into the same sharded layout on the way out. `csc="always"` builds the
sidecar in the same pass as X (`ScxWriter::enable_csc_sidecar`), so there is no
extra read pass and no second copy of the file; under `--memory-budget` the
builder's buckets take a quarter of the budget (the `IngestWithCscPush` phase in
`scx-convert/src/budget.rs`) and ingest sizes itself against the rest.

**Streaming export** (`scx convert --to h5ad/h5mu`, `pyscx.to_h5ad`,
`pyscx.to_h5mu`): the inverse path. `scx_convert::
scx_to_h5ad_streaming` (and `scx_to_h5mu_streaming` / `scx_modality_to_h5ad_streaming`)
iterate SCX CSR shards in row order
via `ScxReader::read_shard_from_entry` (keyed by the catalog entry the
shard walk already holds, so `X`, layers and raw share one read path
rather than three index-based dispatches that each re-derive the shard
ordering) and write hyperslab slices into pre-allocated `/X/{indptr,indices,data}`
HDF5 datasets. The total `nnz` is computed up front from catalog
`ShardStats` (single pre-scan decode when deletion vectors are
active) so the on-disk layout is deterministic — no extendable HDF5
datasets. Obs and var metadata stream through the symmetric
`h5ad::column_stream::write_dataframe_group_from_shards`: schema comes from
`read_obs_schema_logical_lossy()` (catalog-only), HDF5 datasets are
pre-allocated to `n_rows_kept`, then `obs_shards()` / `var_shards()`
are drained shard-by-shard with kept-row hyperslab writes per column.
Categorical columns intern each shard's **declared** vocabulary in
declared order and union across shards, so a `pd.Categorical`'s
category list and its `ordered` bit survive the export unchanged; a
level no row uses is dropped only when a row filter is active, which
is anndata's `remove_unused_categories`-on-subset rule. That prune is
decided by a second decode pass over the metadata shards
(`scan_used_categories`) that runs *only* when a keep mask is present,
because the `codes` dataset is pre-allocated and a category dropped at
finalize would need every already-written code renumbered.
A legacy single-section obs/var source is not a second writer: it goes
through `write_dataframe_group_at`, the whole-batch **driver** over the
same function (one shard, and the deletion-vector mask passed through
rather than applied first). `adata.raw` streams too, via `stream_raw_at`
— the same `stream_csr_to_group_at` driver on raw's own (usually wider)
gene axis, resolved from `ScxReader::raw_n_vars()`; only `raw/var` stays
whole-batch, because var is never sharded. Obsm / varm / obsp / varp /
uns still reuse the materialising helpers and are the remaining
unbounded term. Peak memory is bounded by one
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
      │                 Experiment → GPU-resident AnnData
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
| `SCX_PCA_COV_MEMORY_BUDGET` | `scx-accel` | — | **Removed after v0.13.0; warns once if set.** Capped concurrent `n_vars × n_vars` f64 covariance-PCA accumulators back when there was one per worker. The build now partitions one shared accumulator, so peak no longer scales with the thread count and there is nothing to cap. |
| `SCX_ACCEL_DETERMINISTIC_LINALG` | `scx-accel` | unset | `1`/`true`/`yes`/`on` pins faer's dense decompositions to sequential execution, making **CPU PCA / PFlog** bit-identical across thread counts. Opt-in: ~2.3× slower on the covariance route's eigendecomposition. Read once, at the first CPU PCA/PFlog call. The setting it writes is a faer process-global, so every *implicit* faer decomposition (Harmony, NB-GLM, native-GPU PCA's host SVD) also runs sequentially once it is pinned; the *guarantee*, however, is PCA/PFlog-scoped, and kNN's and eval-metrics' gemms pass an explicit `Par` and ignore it entirely. See [scanpy.md § PCA reproducibility](scanpy.md#reproducibility). |
| `SCX_ACCEL_WILCOXON_NNZ` | `scx-accel` | on | `0`/`false`/`FALSE`/`off` turns the **CPU CSC** Wilcoxon kernel's exact sparse-nnz path (§5.3) **off**, falling back to the densify kernel; unset or any other value leaves it on (the default since 0.20 — before, it was opt-in with `1`). The nnz kernel ranks each gene's nonzeros plus an analytic implicit-zero tie block instead of a dense `n_obs`-length per-gene sort, and is bit-identical to densify (pinned at zero tolerance). 1-vs-rest only — an explicit `reference=` or `rankby_abs=True` keeps the densify path whatever this is set to. The route records as **`cpu_csc_nnz`** when it runs and `cpu_csc` otherwise, so a benchmark can tell which kernel it timed. Read once, through a `OnceLock`, so it must be set before the process's first CSC DE call. |
| `SCX_ACCEL_PAIRWISE_MEMORY_BUDGET` | `scx-accel` | `268435456` (256 MiB) | Byte ceiling for **one** pairwise-distance Gram block on the `gemm` backend (`pyscx.accel.energy_distance` / `energy_distance_details`, i.e. `backend="auto"` on euclidean or cosine). The Gram is built one row block of `a` at a time and reduced before the next overwrites it, rather than as `n_a · n_b · itemsize`. An input whose whole Gram already fits is a single block, i.e. the unblocked computation, so lowering this can only add blocks and raising it can only remove them — the same pairs are summed either way, though a different block width repanels the gemm and moves Gram ulps (inside the `atol=1e-4` parity bound; at one block the **cross** result is bit-identical to pre-0.13.1 — the self path changed convention independently of this knob, so `energy_distance` as a whole is not). **Three things it does not bound:** the block is `max(budget, one Gram row)`, since the planner never returns a zero-row block; `metric="cosine"` separately materialises `n_a · n_dims` (+ `n_b · n_dims` when the sides differ) normalized copies outside it — ~800 MB / ~1.6 GB at 100K × 2000 f32; and `energy_distance` runs perturbations on a rayon `par_iter`, so all of the above is `RAYON_NUM_THREADS ×` over, on top of each task's own dense copy of the perturbation and control rows. Unparseable / zero falls back to the default. Read once. |
| `SCX_CSC_AUTO_OBS_THRESHOLD` | `scx-format` | `50000` | `CscPolicy::Auto` builds a CSC sidecar only when `n_obs ≥` this. |
| `SCX_CSC_AUTO_VARS_THRESHOLD` | `scx-format` | `5000` | `CscPolicy::Auto` builds a CSC sidecar only when `n_vars ≥` this. |
| `SCX_GPU_DE_GENE_CHUNK_SIZE` | `scx-gpu` | VRAM heuristic | Overrides the streaming GPU-DE gene-chunk size (rounded to a multiple of 64, min 64). |
| `SCX_GPU_DE_REQUIRE_DETERMINISTIC` | `scx-gpu` | unset | `=1` turns the CSC pseudobulk fold's fall-through to the global-atomic kernel into an error instead of a silent downgrade. The deterministic shared-memory tree-reduce is taken only while `n_groups × block_dim × 8 B` fits the device's opt-in shared memory (908 groups on sm_90, 652 on sm_80, at the smallest block dimension), so the same file can be reproducible on one card and not on another — set this when the numbers must be reproducible and you would rather be told. Which kernel ran is recorded either way as `reduction` on `uns["scx_accel"][<op>]`. Read once per process, so set it before the first DE op. |
| `SCX_GPU_DE_PSEUDOBULK_FORCE_ATOMIC` | `scx-gpu` | unset | Test-only inverse of the above: any value forces the global-atomic CSC pseudobulk kernel regardless of `n_groups`, so a small fixture can exercise the non-reproducible arm. Read once per process. |
| `SCX_CUVS_TRUST_LAYOUT` | `scx-gpu` | unset | `=1` downgrades a cuVS version/layout-compatibility mismatch from a hard error to a warning (kNN results may be wrong). |
| `SCX_DISABLE_CUDA_GRAPHS` | `scx-gpu` | unset | `=1`/`true` bypasses CUDA-graph capture at every call site. **Not an isolated capture A/B**: call sites also read it to pick a stream. Harmony (the only capture site) runs its k-means sub-iter kernels, order upload and sync on the per-thread stream when graphs are enabled and on the device's own stream when they are not; three sites in `scx-accel/src/diffexp/gpu.rs` read it purely as a stream selector with no capture involved. |
| `SCX_GPU_PCA_RESIDENT` | `scx-gpu` | unset | `=0` forces the streaming GPU PCA power loop instead of the device-resident one (the whole matrix re-decoded and re-uploaded per multiply). Mirrors `SCX_GPU_DE_RESIDENT`; the path taken is recorded on `uns["scx_accel"]["pca"]["resident_csr"]`. Read once per process. |
| `SCX_GPU_PROFILE` | `scx-gpu` | unset | Any non-empty, non-`0` value emits GPU profiling output. |
| `SCX_LOADER_PROFILE` | `scx-loader` | unset | `=1`/`true` emits ML-loader profiling: per-stage timing (I/O-stage decode + `tx.send` back-pressure wait, decode-stage scatter) and the memory-budget breakdown on drop. |

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
