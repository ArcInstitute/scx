# Rust: format I/O (`scx-format`, `scx-format-io`)

> Part of the [SCX API reference](README.md). The on-disk layout these types read and write is specified in
[docs/format.md](../format.md).

## ScxReader (`scx-format-io/src/reader/`)

- `open(path)` — Open and validate file (mmap-based, validates magic/version/minimum size)
- `header()`, `root_catalog()`, `catalog()` — Access file metadata
- `n_obs()`, `n_vars()`, `nnz()` — Quick dimension access
- `read_obs()`/`read_var()` — Arrow RecordBatch metadata
- `read_csr_shard(idx)` — Single shard as `(Vec<i64>, Vec<i32>, Vec<f32>)`
- `read_all_csr_shards()` — Full matrix as `ScxCsr` (parallel via rayon)
- `read_all_csr_shards_typed(plan)` — Full matrix as a `TypedCsr` at the plan's value / index dtypes, assembled directly at that width (also parallel via rayon)

  > **These three address the flattened, all-modality CSR shard list**, so they
  > are defined only on a file that presents one tiling of the obs axis. On a
  > multimodal file each modality independently tiles `[0, n_obs)`, and they
  > return `ScxError::MultimodalRequiresModality` naming the `_for(modality_id)`
  > sibling rather than an `n_obs x n_modalities`-row matrix over mixed column
  > spaces. `FullCatalog::single_tiling_csr_shards` is the seam; the predicate
  > is shard geometry, never the modality table, so a file with a one-entry
  > modality table (what a single-modality h5mu ingest writes, stamping
  > `modality_id = 1`) still reads. `BackedCsrReader::read_all` — `to_memory()`
  > on the Python side — inherits the refusal when it was built unscoped, and
  > reads its own modality when built with `for_modality`.

- `read_csc_shard(idx)` — Single CSC sidecar shard as `ScxCsc`
- `read_all_csc_shards()` — Concatenated CSC matrix as `ScxCsc`
- `read_csc_columns(col_range)` — CSC columns covering a half-open
  `Range<u32>`; only shards overlapping the range are decoded
- `read_csc_columns_subset(cols)` — CSC columns gathered from a sorted
  unique `&[u32]`; contiguous runs are read in a single decode
- `csc_shard_count()` — Number of CSC sidecar shards (0 when
  `header.has_csc()` is false)
- `read_layer(name)` — Named layer as `ScxCsr`
- `layer_names()` — List available layer names
- `read_obsm(name)` / `read_all_obsm()` — Embeddings as Arrow RecordBatch
- `read_uns()` — Unstructured metadata as `serde_json::Value`
- `read_provenance()` — Operation history
- `validate()` — Check all section BLAKE3 checksums
- `section_bytes(entry)` — Direct byte access to a section
- `read_obs_schema()` / `read_var_schema()` — Arrow schema (without data)
- `read_obs_schema_physical()` / `read_var_schema_physical()` — Physical Arrow schema from the first on-disk section (without data)
- `read_obs_schema_logical_lossy()` / `read_var_schema_logical_lossy()` — Logical schema assembled from all shards (field union; lossy because cross-shard type conflicts are resolved by first-seen-wins)
- `obs_shard_count()` / `var_shard_count()` — Number of `ObsMetadataShard` / `VarMetadataShard` sections (0 on legacy single-section files)
- `read_obs_shard(idx)` / `read_var_shard(idx)` — Single metadata shard as Arrow RecordBatch
- `obs_shards()` / `var_shards()` — Iterator over all metadata shards
- `read_obs_assembled()` / `read_var_assembled()` — Reassemble all metadata shards into one Arrow RecordBatch (transparent on legacy single-section files)
- `obs_categorical(col)` / `obs_categorical_many(&[cols])` — `(codes: Vec<i32>, categories: Vec<String>)` for string/categorical obs columns, folded **one shard at a time** into a running global dictionary (never concatenates the column, unlike `read_obs_keys`). Null → `-1` (pandas convention), so a literal `"NaN"` string stays a real category; category order is first-seen. Accepts both `Dictionary(_, Utf8|LargeUtf8)` (as `from_anndata`, the in-place obs writers and — since this change — `append` / `merge` write a categorical) and plain `Utf8`/`LargeUtf8` (as an older `append` / `merge` wrote one, and as a plain source column still lands), including a file mixing both across shards. `_many` costs **one** projected read per shard for N columns. See [Obs categorical codes](../performance/loader-data-load.md#obs-categorical-codes-without-pandas-data-load-phase-1-1c).
- `debug_counts()` — `ReaderDebugCounts` with `AtomicU64` I/O counters (`cfg(debug_assertions)` only). `read_obs_shard_projected` counts column-scoped shard reads separately from `read_obs_shard`, so a test can assert the cheap path was *taken* rather than only that the materialising ones were avoided.
- `read_obs_predicate_index_bytes()` / `read_var_predicate_index_bytes()` — Predicate index raw bytes
- `read_deletion_vectors()` — Roaring Bitmap deletion vectors
- `deletion_keep_mask()` / `deletion_keep_mask_for(modality_id)` — `Option<Vec<bool>>` (`true` = keep); `None` when nothing is deleted
- `read_all_csr_shards_filtered()` / `read_all_csr_shards_for_filtered(modality_id)` / `read_layer_filtered(name)` — Matrix reads with deletion vectors applied
- `read_obs_filtered()` / `read_obs_keys_filtered(&[cols])` / `obs_categorical_filtered(col)` / `obs_categorical_many_filtered(&[cols])` — The obs half of the above: the whole frame, the projected read and the codes fold with the deletion keep mask applied. `read_obs()` / `read_obs_keys()` / `obs_categorical()` are *physical* (`header.n_obs` rows regardless of deletions), so any caller materialising a matrix for a user needs both halves and must move them together — an obs frame longer than its matrix is worse than either being stale alone. (`pyscx`'s `read_obs()` / `obs_categorical()` route to the filtered twins by default since 0.17.) `scatter_batch_to_physical(batch, keep)` is the inverse of `filter_batch_by_keep_mask`: a live-length batch back onto the physical axis, null at deleted rows — what the in-place obs writers use to accept a live-length frame
- `read_shard_header()` / `read_raw_shard_bytes()` / `read_shard_from_entry()` — Low-level shard access
- `mmap()` — Direct mmap access to the underlying file

## ScxWriter (`scx-format-io/src/writer.rs`)

- `new(path, header)` — Create writer (writes to temp file)
- `write_obs(batch)`/`write_var(batch)` — Arrow IPC metadata
- `write_csr_shard(indptr, indices, values, ...)` — CSR expression data
- `write_layer_csr_shard(name, ...)` — Named layer CSR data
- `write_obsp_shard(name, ...)` — Cell-cell graph CSR data. The shard's
  minor extent is stamped from `header.n_obs` (an obsp graph is obs x obs), so
  the per-shard index width follows the cell axis and not the gene axis; on a
  multimodal file `write_obsp_shard_for` stamps the same global `n_obs`, never
  the modality's `n_vars`. Canonical CSR is a precondition — this writer does
  not canonicalize.
- `write_obsm(name, batch)` — Embeddings
- `write_uns(json)` — JSON metadata
- `write_provenance(operations)` — Operation history
- `write_csc_shard(indptr, indices, values, codec_id, value_encoding, col_start)`
  — Write a CSC sidecar shard. Fully supported. The catalog
  `section_type` is `CscShard (5)`; the on-disk shard header carries
  `shard_type = 1` going forward (readers also accept the legacy
  `shard_type = 0` when the catalog `section_type` is `CscShard`).
  `finish()` updates `header.n_csc_shards` from the writer's
  internal counter and sets the `has_csc` flag bit.
- `write_obs_predicate_index(data)` / `write_var_predicate_index(data)` — Predicate indexes for pushdown
- `write_deletion_vectors(dv)` — Roaring Bitmap deletion vectors
- `write_raw_shard(raw_bytes, section_type, name, stats, nnz)` — Pre-encoded shard passthrough
- `set_shard_column_stats(column_stats)` — Per-column shard statistics for catalog
- `finish()` — Atomic write: full catalog at EOF -> pwrite root catalog -> pwrite header -> fsync -> rename

## Codec Selection (`scx-format/src/codec_select.rs`)

The user-facing `codec=` argument is a single **intent axis** with three profiles,
resolved by `resolve_codec`:

- **`auto`** (default) — cost-aware adaptive. Per framed integer shard, dual-encode
  the heuristic vs ShufDeltaZstd and adopt ShufDeltaZstd only when it is smaller by
  at least `ADOPT_MARGIN` (5%), so a marginal size win never pays the ShufDeltaZstd
  decode tax. The heuristic itself is `select_codec`: integer median ≤ 8 → Scx1 (Rice,
  optimal for 10x UMI counts), median > 8 → Zstd, float → Pcodec. Unframed writes fall
  back to the heuristic single-encode. `auto` files are typically **mixed-codec**;
  `scx info` prints the per-shard breakdown.
- **`fast`** — decode-speed-max: the heuristic single-encode (Scx1/Zstd), never
  ShufDeltaZstd. This is the pre-flip default; pin it for latency-critical CPU training.
- **`compact`** — size-max: adopt ShufDeltaZstd on ties (framed only).

Explicit forces (`none`/`scx1`/`zstd`/`lz4`/`pcodec`/`shufdelta`) and `compact-trial`
remain available. (The prior `auto_v2` profile + `decode_target` knob were removed as
a pre-1.0 clean break — use `auto`/`compact`.)

**Codec / profile tradeoffs:**

| Codec | Best for | Compression | Read speed | Write speed |
|-------|----------|-------------|------------|-------------|
| `auto` | General use (recommended default) — size-optimizing adaptive | Best per-shard (adopts ShufDeltaZstd where it wins) | Near-best (~6% ShufDeltaZstd tax on adopted shards) | Moderate (dual-encode) |
| `fast` | Latency-critical CPU training | Heuristic per-shard | Fastest (SIMD Scx1 decode) | Best (single-encode) |
| `compact` | Storage/egress-bound archival | Highest (adopts on ties) | ~6% ShufDeltaZstd tax | Moderate (dual-encode) |
| `scx1` | Small UMI counts (median ≤ 8) | Best for 10x data (~4.8×) | Fastest (SIMD decode) | Moderate |
| `zstd` | Large integers, general fallback | Good (~4.3× UMI, ~3.8× float) | Fast | Fast |
| `shufdelta` | Force the compact integer codec | ~1.5–2.5× smaller than Scx1 | ~6% slower than Scx1 | Moderate |
| `pcodec` | Log-normalized, PCA embeddings, float layers | Best for floats (~4.1–4.7×) | Moderate (19–39% slower than Zstd) | Slower (35–40% slower than Zstd) |
| `lz4` | Speed-critical pipelines | Lower (~2.2–3.1×) | Fast | Fastest compressed |
| `none` | GDS bypass, debugging | 1× (no compression) | Fastest (I/O bound) | Fastest |

For storage-constrained workflows with normalized float data, explicitly selecting `pcodec` gives the best compression. For latency-sensitive pipelines, `fast`, `zstd`, or `lz4` are better choices.

## Provenance

- `ProvenanceEntry`: timestamp, action, tool, params_json, input_checksums
- Auto-populated on `ScxWriter::finish()` with operation info
- `scx info` displays full provenance history
- Read/write via `ScxReader::read_provenance()` / `ScxWriter::write_provenance()`
- Conversion runs include a `params_json.warnings` summary (per-category
  counts emitted via [`WarningSink::summary_json`](conversion.md#conversion-warnings-convertwarning)),
  along with `stream`, `source_format`, `source_matrix_format`,
  `indexed_obs`, `indexed_var`, `bitmap`, `memory_budget_mb`,
  `modalities`, and modality-type overrides where applicable.

## BackedCsrReader (`scx-format-io/src/backed/csr.rs`)

- `new(reader, cache_shards)` — Create backed reader from `ScxReader` with LRU shard cache
- `cache_shards()` → `usize` — The requested LRU count cap this reader was built with (`0` = no cache; the cache clamps its own capacity to ≥ 1 internally). Read back by the pyscx handles' `cache_shards` getter.
- `stored_value_encoding()` → `Result<Option<ValueEncoding>>` — The on-disk value encoding of this reader's shard family (X, the layer, or the modality it is scoped to — it walks the same `shard_entry` table every read does): one 76-byte header read per shard, no payload decode, memoised. A uniform family reports its own encoding; a mixed one the widest via `ValueEncoding::widest` (any float ⇒ `Float32`, else the widest integer); `None` for no shards. The encoding lives only in the shard header — `ShardStats` has no encoding field, and a `value_max` of 0 cannot tell a float shard from an all-zero one — so this is the one place a caller learns whether the stored values are integer counts without decoding.
- `read_rows(start, end)` → `ScxCsr` — Decode rows `[start, end)` from the overlapping shards into one pre-sized result. A range that fits the LRU is warmed and copied from the cache; a bulk range (more full shards than `cache_shards`, e.g. the whole matrix) decodes uncached in parallel chunks of `cache_shards`, keeping the LRU's entries as they were (the residents it copies are promoted, as any hit is) — peak = result + up to `cache_shards` shards in flight, on top of whatever the LRU already holds (itself capped at `cache_shards`). `end > n_obs` is an error, and so is a catalog whose shards do not tile the range exactly (a gap, or the overlapping modalities of an unscoped reader on a multimodal file).
- `read_row_indices(indices)` → `ScxCsr` — Decode specific rows by index (fancy indexing), in request order, duplicates allowed. Assembles the result once: an indptr-only prescan of each touched shard sizes the output exactly, then the `read_rows_with` scatter copies each row into place — peak = result + the shard cache + up to `cache_shards` shards decoding in flight while `warm_shards` fills it (at most `2 × cache_shards` decoded shards beside the result on a full cache) + the block-index transient, which is one pool-width **chunk** of row groups (they are decoded a chunk at a time and each chunk is scattered before the next is decoded, so an over-budget gather holds one chunk rather than all its groups). A sparse request on a row-group-framed shard that is not resident whole decodes only the touched row groups (block index) and retains **those groups** in the same LRU under the same byte budget — the whole shard is never inserted — so a repeated small gather over one region is served from cache (`row_group_hits`). This method decides admission for itself: its groups are retained only if all of them fit the byte budget (sized from the block index before any decode); over budget, it decodes and drops. `read_row_indices_with_admission(indices, admit_row_groups)` is the same read taking the verdict from the caller instead — the pair is exactly `read_rows_with` / `read_rows_with_admission` below, and `scx-loader`'s cell-set gather reads a whole plan through it so the L1 gather and the L2 warm act on one verdict. Whether the read takes the block-index route or decodes the shard whole is decided from the **distinct** rows it touches, not from the request positions, so a plan that repeats rows does not read as a dense request. The prescan reads a resident shard's indptr rather than decoding it again, which is invisible except as time: it counts no hit and does not touch LRU recency. An out-of-range row is an error (it used to be dropped silently).
- `read_rows_with(rows, scatter)` / `read_rows_with_admission(rows, admit_row_groups, scatter)` → `()` — The scatter primitive under `read_row_indices` and the plan loaders: `scatter(i, indices, data)` once per requested row with zero-copy views into the decoded shard or row group. `read_rows_with` decides row-group admission per call, as above; `read_rows_with_admission` takes the verdict from the caller — `scx-loader`'s prefetch engine decides once per plan, over every gather the plan will make and every file it touches, against `budget / (lookahead + 1)`, so the L1 gathers and the L2 warm act on one admission verdict (the warm pre-decodes the eligible subset of what the gathers may retain) and a plan of individually-fitting gathers whose union is over budget cannot retain that union. The verdict sizes the plan's whole footprint in the shared LRU — its row groups and the shards it takes whole. A plan that fits its share is retained whole; a plan that does not keeps only the row groups another plan of the lookahead window also touches, hottest-first and only while they fit **the room that share leaves after the shards the plan takes whole** (`share - whole_shard_bytes` — those shards are inserted regardless), with its cold tail decoded and dropped. A resident group is served as a hit either way; `Admit::None` stops every miss from being inserted and `Admit::Groups` all but the named keys. ⚠️ **Scatter order is per pass, not sorted by row**: every full-shard fallback fires before every block-index row group, so on a mixed request a later full-shard row precedes an earlier block-index one. The `i` argument — the position in `rows` — is the only ordering guarantee, and every in-tree consumer indexes by it. `read_rows(start, end)` applies the same idea to its own edge windows: one verdict over every row-range window of the read.
- `read_shard_cached(idx)` → `ScxCsr` — Read shard through LRU cache (clones on hit)
- `read_shard_uncached(idx)` → `ScxCsr` — Read shard bypassing cache (preferred for streaming)
- `row_sums()` / `col_sums()` — Streaming per-row/column sums
- `row_nnz()` / `col_nnz()` — Streaming per-row/column NNZ
- `row_var()` / `col_var()` — Streaming per-row/column variance
- `row_max()` / `col_max()` / `row_min()` / `col_min()` — Streaming extrema

- `total_nnz()` — Total NNZ across all shards
- `col_means_and_sum_sq(zero_center)` — Single-pass column statistics for PCA.
  Decodes one shard at a time on the calling thread; a caller that can name a
  `Sync` source should prefer `scx_format_io::col_means_and_sum_sq_prefetched`,
  which overlaps decode across shards and is bit-identical to it (the trait
  default is that function's test oracle, so the two cannot drift). Both CPU and
  GPU PCA take the prefetched form.
- Masked variants (deletion-vector aware): `col_sums_masked(kept_rows)`, `col_nnz_masked(kept_rows)`, `col_max_masked(kept_rows)`, `col_min_masked(kept_rows)`, `col_var_masked(kept_rows)`

### Overfull-axis rejection in the aggregations

`row_var` / `col_var` and the four extrema fold implicit zeros into their result,
and so have to know how many there are: `extent − nnz`, where `extent` is
`n_cols` on the row axis and `n_obs` (or the kept-row count) on the column axis.
That is only valid on a **canonical** CSR — `docs/format.md` § "v3 canonical CSR
invariant" requires per-row column indices to be strictly increasing, so an axis
of `extent` cells holds at most `extent` entries. Ordinary reads do not verify
that: the decode seam bounds index *values*, and only `scx validate --deep`
checks ordering. They therefore return
`ScxError::Csr(CsrError::NonCanonicalAxis { extent, nnz })` when **an axis holds
more stored entries than it has cells**, rather than reporting a number derived
from a wrapped count. Surfaced through pyscx as a `RuntimeError` from
`X.var(axis=0)` / `X.var(axis=1)` / `X.max(axis=…)` / `X.min(axis=…)` on a backed
matrix, with or without a column projection.

⚠️ **That predicate is narrower than "the shard is canonical", and the
difference matters.** `nnz > extent` proves a repeated coordinate, but the
converse does not hold: a sparse row with a couple of repeats stays under its
extent and is *not* detected — a 1×3 row storing indices `[0, 0]` yields a
variance computed from both entries plus one implicit zero, with no error. So
these are overfull-axis guards, not uniqueness guards.

Which routes reject, and which answer:

| Route | On an overfull axis |
|---|---|
| `X.var(axis=0)` / `X.var(axis=1)`, CSR or CSC, projected or not | `RuntimeError` |
| `X.max(axis=0)` / `X.min(axis=0)`, projected or not | `RuntimeError` |
| `X.max(axis=1)` / `X.min(axis=1)`, **unprojected** | `RuntimeError` |
| `X.max(axis=1)` / `X.min(axis=1)`, **with a column projection** | **answers** — that branch materializes via `to_memory()` and defers to scipy, so no guard runs |
| CSC Wilcoxon DE (`scx-accel`, nnz fast path) | `AccelError::InvalidInput` when the **labelled-nonzero** count overruns; explicit stored zeros are excluded from that count, so a column overfull purely with duplicate zeros is not caught |
| `X.var(axis=None)` scalar, unprojected | clamped to `0.0` — no per-column count exists on that path |
| `scx-accel` PCA total variance | `saturating_sub`, absorbed |
| A duplicate leaving the axis under its extent | not detected anywhere |
| An **unsorted** overfull row seen through a column projection | not detected — `project_csr` drops indices smaller than one already seen, so the projected row is no longer overfull |

Variance results preserve `NaN`: the clamps use `if v < 0.0 { 0.0 } else { v }`
rather than `v.max(0.0)`, which would return `0.0` for a NaN input because
Rust's `f64::max` ignores NaN.

Use `scx validate --deep` when you need an actual canonicality verdict on a file.
None of the read-path aggregations is a substitute for it.

## ShardSource Trait (`scx-format-io/src/shard_source.rs`)

A uniform interface for streaming CSR data shard-by-shard, enabling algorithms like PCA to process data without materializing the full matrix. Defined in `scx-format`, available to both `scx-accel` and `pyscx`.

```rust
pub trait ShardSource {
    fn n_shards(&self) -> usize;
    fn n_obs(&self) -> usize;
    fn n_vars(&self) -> usize;
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }
    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr>;
    fn col_means_and_sum_sq(&self, zero_center: bool)
        -> Result<(Option<Vec<f64>>, Vec<f64>)>; // default impl provided
}
```

**Implementations:**

| Type | Crate | Behavior |
|------|-------|---------|
| `BackedCsrReader` | `scx-format` | Reads raw decoded shards via `read_shard_uncached()` |
| `LazyShardSource` | `pyscx` (internal) | Applies per-shard transforms (NormalizeTotal, Log1p, RowScale), column projection (`project_csr`), and deletion vector filtering before returning |

**Used by:** `scx-accel::randomized_pca()`, `streaming_spmm_forward()`, `streaming_spmm_transpose()` — all generic over `<S: ShardSource>` for zero-cost abstraction across the crate boundary.

**Design note:** The trait is defined in `scx-format` (not `scx-accel`) so that `pyscx`'s `LazyShardSource` can implement it without creating a dependency on `scx-accel`. PCA functions in `scx-accel` are generic (`<S: ShardSource>`) rather than using `&dyn ShardSource` to allow monomorphization.

## BackedCscReader (`scx-format-io/src/backed/csc.rs`)

Column-major counterpart to `BackedCsrReader`. Streams CSC sidecar
shards from disk through the same `ShardCache` the CSR and dense readers use —
a byte-budgeted LRU plus a singleflight table, so concurrent readers of one cold
shard decode it once.

- `new(reader, cache_shards)` — Create from `ScxReader`. `cache_shards = 0` disables caching (one decode per call). Count-only: the byte budget is `usize::MAX`, so only the shard count bounds the cache
- `with_byte_budget(reader, cache_shards, bytes)` — As `new`, but also caps the cache in bytes. A decoded `ScxCsc` is measured by its components (`indptr.len()*8 + indices.len()*4 + data.len()*4`), the same model `IndexPlanLoader`'s memory-budget auto-tune uses
- `index()` — Per-shard column-range index, sorted by `col_start`
- `n_shards()` / `n_obs()` / `n_vars()` — Dimensions
- `read_shard_uncached(idx)` → `ScxCsc` — Single CSC shard, bypass cache
- `read_shard_cached(idx)` → `Arc<ScxCsc>` — Single CSC shard, through cache and singleflight
- `read_csc_columns(col_range)` → `ScxCsc` — Decode only shards overlapping the half-open range; partial-overlap shards are sliced post-decode
- `read_csc_columns_subset(cols)` → `ScxCsc` — Gather columns from a sorted unique `&[u32]`; contiguous runs share a single decode
- `enable_metrics()` / `metrics()` — Hits / misses / evictions / decoded-bytes counters. **Idempotent**: repeat calls return the same accumulating handle rather than resetting. To measure an interval, snapshot and subtract — the same contract `BackedCsrReader` and `BackedDenseReader` have

## ColumnShardSource Trait (`scx-format-io/src/shard_source.rs`)

Sibling to `ShardSource` for column-major streaming. Defined in
`scx-format` so consumers in `scx-accel` and `pyscx` can take the
bound generically without depending on each other.

```rust
pub trait ColumnShardSource {
    fn n_csc_shards(&self) -> usize;
    fn n_obs(&self) -> usize;
    fn n_vars(&self) -> usize;
    fn shape(&self) -> (usize, usize) { (self.n_obs(), self.n_vars()) }
    fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc>;
    fn read_csc_columns(&self, col_range: Range<u32>) -> Result<ScxCsc>;
    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)>;
}
```

**Implementations:**

| Type | Crate | Behavior |
|------|-------|---------|
| `BackedCscReader` | `scx-format` | Reads raw decoded CSC shards from disk (cache + index-driven shard skip) |
| `LazyShardSource` | `pyscx` (internal) | Presents a handle's **view**: applies the full transform chain post-decode — the row-indexed `NormalizeTotal` / `RowScale` included, read at the global row `ScxCsc::indices` carries — then renumbers the slab's rows onto the live row space under a row filter, then remaps its columns into the projected axis. Serves both handle kinds; a backed one reaches it with an empty chain |

**Capability gate:** the trait is *not* a sub-trait of `ShardSource`.
Consumers that want CSC dispatch take the bound explicitly (`fn
require_csc<S: ColumnShardSource>(...)`); the runtime "does this
dataset support CSC?" question is answered exactly once at
`ScxBackedSparseDataset::as_column_source()` /
`ScxLazyTransformedDataset::as_column_source()` — one predicate with one
condition, and the two now share an implementation. Returns `Some` iff the
file has a CSC sidecar. A transform chain is not a disqualifier (every
`Transform` is CSC-applicable), and neither is a row filter or a column
projection: both gates hand back a `LazyShardSource` that renumbers the
slab's rows onto the live space and remaps its columns into the projected
axis, where the backed gate used to hand back the full-axis
`BackedCscReader` and so had to refuse. **A consumer must therefore address
the projected axis** — `n_vars()` is the visible width and a requested
column range is mapped *through* the projection, so passing global column
ids back in applies it twice. See `pyscx.accel.* prefer_format` below.

**Used by:** `scx-accel::csc::{streaming_mean_var_csc,
streaming_clip_square_sum_csc, wilcoxon_rank_sum_streaming_csc,
pseudobulk_aggregate_csc}` and the `pyscx::projected_agg::*_csc`
column-aggregation kernels.
