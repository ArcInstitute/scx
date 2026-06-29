# Sharding in SCX

SCX partitions the expression matrix into **CSR shards** — fixed-size, independently
decompressible chunks of rows. Sharding is the core mechanism enabling parallel I/O,
predicate pushdown, memory-bounded reads, and append-without-rewrite.

## How it works

When an SCX file is written, the sparse CSR matrix is split into consecutive groups
of rows. Each group becomes a self-contained shard with its own header, compressed
indptr/indices/values arrays, and block index.

```
Expression matrix (n_obs × n_vars)
┌─────────────────────────────┐
│  rows 0 – 16383             │ → CSR Shard 0
│  rows 16384 – 32767         │ → CSR Shard 1
│  rows 32768 – 49151         │ → CSR Shard 2
│  ...                        │ → ...
│  rows K – n_obs             │ → CSR Shard N-1
└─────────────────────────────┘
```

Each shard is referenced in the **full catalog** at the end of the file, which stores
per-shard checksums, row ranges, nnz counts, and — critically — **per-column category
bitsets** that power predicate pushdown.

### Default shard size

| Context | Default rows per shard |
|---------|------------------------|
| All entry points (CLI, Python, R) | **16,384** (`DEFAULT_SHARD_TARGET_ROWS`) |

> [!TIP]
> The default is a good starting point for most datasets. Smaller shards
> (e.g., 1,000–5,000) improve query selectivity at the cost of slightly larger
> files. Larger shards (e.g., 50,000) reduce catalog overhead for very large
> datasets.

### Shard sizing guidelines

At 5% density with 30K genes, a 16,384-row shard contains ~24M non-zero values and
compresses to ~50–100 MB. Key considerations:

- **Query engine**: Smaller shards → more granular predicate pushdown → fewer cells
  read for selective queries. Diminishing returns below ~1,000 rows.
- **Training loader**: The triple-buffered pipeline (see [architecture.md §Training Dataset](architecture.md#training-dataset)) reads full shards
  sequentially. Very small shards add per-shard overhead; very large shards delay
  shuffling. The default of 16,384 rows is a good balance.
- **Cloud access**: The exploded `.scxd` layout stores each shard as a separate
  object. Shard coalescing (merging adjacent range reads) works best when shards
  are large enough to amortize per-request latency (~30–100 MB).
- **Append**: New cells are written as new shards at EOF. The `shard_target_rows`
  controls how many rows go into each appended shard.

## Sorting for read locality (`scx sort`)

Sharding decides *how many* rows live in a block; **sorting** decides *which*
rows. `scx sort` globally reorders the obs (cell) axis by a chosen obs key and
re-shards into the canonical row-major layout, so cells that share a key value
land in the same few contiguous shards. The X matrix, layers, obsm, and obsp are
reordered to match; var/varm/varp are untouched; the predicate index is rebuilt
so each category's `shard_ranges` become a single contiguous range.

**What it buys you — read locality, not filter cost.** After sorting by
`cell_type`, "fetch all macrophages" touches a handful of adjacent shards instead
of being scattered across the whole file, and the catalog-dict shard pruning the
query engine already does (`prune_shards_by_catalog_with_dict`) collapses the
scan to that contiguous range. This is the one thing the engine-side **row-set
predicate pushdown** (the O(result) obs-filter path) *cannot* do — pushdown
decodes matching shards in catalog order but never reorders them, so physical
layout is the only lever for X-read locality. The two are **complementary**:
pushdown is the workload-agnostic filter-cost fix (no rebuild); sort is an opt-in
layout tool for *"I know my dominant query axis and run large/unbounded scans on
it."* Only the leading sort key benefits — a shifting query axis should rely on
the pushdown instead.

Sorting is **stable** (equal keys keep their original order) and **deterministic**
(same input + flags ⇒ byte-identical output, modulo the provenance timestamp).
Deletions are materialized away first (the output is dense and deletion-free).
The CSC sidecar and detection bitmap are dropped by default; pass `--rebuild-csc`
/ `--bitmap auto|always` to re-emit them on the sorted output.

**Output size is not guaranteed neutral.** A sort is a row permutation, not a
recompression pass: `scx1`-coded X is size-neutral (each row's gene indices are
coded independently of row order), but `zstd`-coded shards and per-shard
auto-codec re-selection make the sorted X a few percent larger *or* smaller
because regrouping which cells share a shard changes cross-row compressibility.
On the 149M-cell `drug.scx` (`mixed scx1/zstd`, uint32), sorting by `cell_type`
grew the file ~6% — all in the X matrix (value encoding, predicate index, and
CSC were unchanged). Pin `--codec` or run a follow-up `scx compact` if size
matters; the point of `sort` is read locality, not compression. See
[performance.md § Sort (physical layout)](performance.md#sort-physical-layout).

### Three ways to sort

The reorder ships as three entry points — prefer the build-time forms, which are
spill-free:

| form | when | mechanism |
|------|------|-----------|
| `scx convert --sort-by` / `from_anndata(sort_by=…)` | sorting at ingest from h5ad | spill-free hyperslab gather from the random-access source |
| `scx merge --sort-by` / `merge(sort_by=…)` | building an atlas from many `.scx` | sorted **k-way merge** of (pre-sorted) inputs; spill-free streaming |
| `scx sort` / `pyscx.sort(…)` | re-sorting an already-built file | external partition sort (bounded memory; the recovery path) |

```bash
# Recovery: sort an existing file by a categorical key, indexing it
scx sort --by cell_type --index-obs cell_type input.scx sorted.scx

# Composite (lexicographic), descending, custom shard size, bounded memory
scx sort --by cell_type,donor_id input.scx out.scx
scx sort --by n_genes --reverse input.scx out.scx
scx sort --by cell_type --memory-budget 8G --temp-dir /scratch in.scx out.scx
scx sort --by cell_type --rebuild-csc --bitmap auto in.scx out.scx
```

```python
import pyscx
# Build-time (preferred): bake the order into the conversion / merge
pyscx.from_anndata(adata, "out.scx", sort_by=["cell_type"])
pyscx.merge(inputs, "atlas.scx", sort_by=["cell_type"])
# Recovery: sort an already-built file
pyscx.sort("input.scx", "sorted.scx", by=["cell_type"], memory_budget="8G")
```

The standalone engine auto-selects a strategy from `(n_obs, n_vars, density, key
cardinality, --memory-budget)`: an in-memory argsort for files that fit the
budget, a zero-spill K-pass-by-category for a low-cardinality categorical key,
or a bounded-memory external partition sort otherwise. See
[performance.md § Sort (physical layout)](performance.md#sort-physical-layout)
for measured compression and locality numbers.

## CLI commands

### Setting shard size during conversion

```bash
# Convert h5ad → SCX with 16,384 rows per shard (default)
scx convert experiment.h5ad experiment.scx

# Use a custom shard size
scx convert experiment.h5ad experiment.scx --shard-size 5000

# Convert MTX → SCX with custom shard size
scx convert /path/to/filtered_feature_bc_matrix/ experiment.scx --shard-size 20000
```

### Appending with shard size control

```bash
# Append new cells (uses default shard size for new shards)
scx append atlas.scx new_batch.scx

# Custom shard size for appended shards
scx append atlas.scx new_batch.scx --shard-size 5000
```

### Subsetting with shard size control

```bash
# Extract T cells with a custom shard size in the output file
scx subset experiment.scx t_cells.scx \
    --filter "cell_type == 'T cell'" --shard-size 8000
```

### Inspecting shard information

```bash
# View file metadata including shard count and shard_target_rows
scx info experiment.scx

# JSON output includes n_csr_shards, shard_target_rows
scx info experiment.scx --json
```

### Compaction (re-sharding)

After multiple appends, files may contain many small shards. Compaction merges
them into optimally-sized shards matching `shard_target_rows`:

```bash
scx compact experiment.scx compacted.scx
```

### Merging (re-sharding)

Merge also writes optimally-sized shards:

```bash
scx merge batch1.scx batch2.scx batch3.scx --output atlas.scx
```

## Python API

### Writing with shard size

```python
import pyscx

# From AnnData (default: 16,384 rows per shard)
pyscx.from_anndata(adata, "experiment.scx")

# Custom shard size
pyscx.from_anndata(adata, "experiment.scx", shard_size=8192)

# From 10x HDF5
pyscx.from_10x("filtered_feature_bc_matrix.h5", "experiment.scx", shard_size=10000)

# From MTX directory
pyscx.from_mtx("/path/to/matrix/", "experiment.scx", shard_size=10000)
```

### Appending with shard size

```python
# Append from another SCX file
pyscx.append("atlas.scx", "new_batch.scx", shard_size=10000)

# Append directly from an AnnData object
pyscx.append_from_anndata("atlas.scx", new_adata, shard_size=10000)
```

### Querying (shard-level pushdown)

The query engine automatically prunes shards that cannot match a predicate. The
result reports how many shards were skipped:

```python
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")
    .collect())

adata = result.to_anndata()
print(f"Skipped {result.skipped_shards}/{result.total_shards} shards")
```

On row-sharded files (obs/var metadata sharding, below) the query engine
evaluates obs predicates **one metadata shard at a time** into a bounded
`n_obs` mask, and skips decoding obs shards that don't overlap a surviving
CSR shard when per-shard catalog row-range stats are present. Peak obs
memory is ~`(rayon width × one shard) + n_obs` bytes rather than the whole
obs table, so `query().filter_obs(...).count()` / `.collect()` stay
bounded at atlas scale (9 000+ shards). Files written before row-range
stats existed still get the memory bound (every shard is streamed) but not
the I/O skip; re-running `pyscx.compact` re-stamps the stats and restores
shard skipping. This bound is **specific to the query path** — `compact`,
`merge`, streaming export, `subset`, and `to_anndata` still assemble the
full obs table.

Filtered categorical obs columns in a `collect()` result carry only the
categories present in the surviving rows (AnnData/pandas convention), not
the full parent dictionary.

### Training loader (shard-level streaming)

The training data loader reads shards sequentially with inter-epoch
shuffling at the shard level:

```python
dataset = pyscx.TrainingDataset(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_array,
    normalize=True,
    log1p=True,
)
for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
```

### Cloud operations (shard-level selectivity)

Selective pull downloads only shards matching a predicate:

```python
# Download only T cell shards from cloud
stats = pyscx.pull(
    "gs://bucket/atlas.scxd/",
    "t_cells.scx",
    filter="cell_type == 'T cell'"
)
print(f"Downloaded {stats['downloaded_shards']}/{stats['total_shards']} shards")

# CloudExperiment exposes shard count metadata
exp = pyscx.open_cloud("gs://bucket/atlas.scxd/")
print(exp.shard_count)  # number of CSR shards
```

## R API

### Writing with sharding

```r
library(rscx)

# From Seurat v5 object (default: 16,384 rows per shard)
scx_from_seurat(seurat_obj, "experiment.scx")

# From SingleCellExperiment
scx_from_sce(sce, "experiment.scx")
```

### Reading shard metadata

```r
exp <- scx_open("experiment.scx")
exp$shard_count   # number of CSR shards
```

### Querying with shard stats

```r
result <- scx_open("experiment.scx") |>
  filter_obs(cell_type == "T cell") |>
  collect()

result$skipped_shards  # shards pruned by pushdown
result$total_shards    # total shards in file
```

### File operations

```r
# Append new cells
scx_append("atlas.scx", "new_batch.scx")

# Compact fragmented shards
scx_compact("atlas.scx", "compacted.scx")

# Merge multiple files (re-shards output)
scx_merge(c("batch1.scx", "batch2.scx"), "atlas.scx")

# Inspect shard metadata
scx_info("experiment.scx")  # returns list with n_shards
```

## How sharding enables key features

| Feature | How shards help |
|---------|-----------------|
| **Predicate pushdown** | Per-shard category bitsets let the query engine skip entire shards that can't match a filter, without reading any cell data. |
| **Parallel decode** | Rayon decodes multiple shards in parallel on multicore CPUs. |
| **Append-without-rewrite** | New cells are written as new shards at EOF; only the catalog is updated. |
| **Logical deletion** | Deletion vectors are per-shard Roaring Bitmaps — no data rewrite needed. |
| **Memory-bounded reads** | Only one shard needs to be resident in memory at a time. |
| **GPU decode** | Each shard is independently decompressible on GPU. |
| **Cloud selectivity** | In the exploded `.scxd` layout, each shard is a separate cloud object — only matching shards are downloaded. |
| **Training loader** | Shards are the shuffle and prefetch unit. The triple-buffered pipeline reads, decodes, and transfers shards without blocking. |

## Internal shard layout

Each CSR shard is a 76-byte header followed by three compressed sections:

```
┌──────────────────────────────────────────┐
│ Shard Header (76 bytes)                  │
│   magic, codec_id, value_encoding,       │
│   n_major (rows), n_minor (cols), nnz,   │
│   global_offset (first row index),       │
│   section offsets/lengths, checksum      │
├──────────────────────────────────────────┤
│ Indptr    (Delta-Golomb encoded)         │
├──────────────────────────────────────────┤
│ Indices   (FOR-BP encoded)               │
├──────────────────────────────────────────┤
│ Values    (Adaptive Rice encoded)        │
├──────────────────────────────────────────┤
│ Block Index (O(1) row-range access)      │
└──────────────────────────────────────────┘
```

Each shard can independently override the file-level `codec_id` and `value_encoding`,
enabling per-shard adaptive codec selection and mixed integer/float layers.

For the full binary specification, see [format.md §CSR Shard Internal Layout](format.md#4-csr-shard-internal-layout).

## Obs/var metadata sharding

Streaming `scx merge`, `scx append`, and `pyscx.from_anndata` (when
`n_obs > shard_size`) write obs and var metadata as row-sharded
Arrow IPC sections (section types 24/25) instead of a single monolithic
`obs_metadata` / `var_metadata` section. This bounds peak memory at one
shard's worth of metadata during the merge/append/ingest hot path and
avoids Arrow IPC's 2 GB narrow-offset ceiling for string columns at atlas
scale.

Obs/var **metadata** sharding is decided independently of X sharding and
applies only to the paths listed above. The streaming h5ad / MTX convert
path — `pyscx.from_h5ad`, `pyscx.from_mtx`, and `scx convert` — always
writes a single-section `obs_metadata` / `var_metadata`, regardless of
`n_obs` (the X matrix is still row-sharded at `shard_target_rows`). To
obtain sharded metadata from an h5ad source, either ingest with
`pyscx.from_anndata`, or migrate an existing single-section file with
`pyscx.compact(src, dst, reshape_obs=True)` (`scx compact --reshape-obs`).

Each shard covers rows `[row_start, row_start + n_shard_rows)` of the
logical metadata table and carries schema metadata (`shard_idx`,
`row_start`, `n_shard_rows`, `n_rows_total`). The same row range is also
recorded in the catalog as row-range-only `ShardStats` (no nnz/value
summary — metadata shards have no CSR semantics), so the query engine can
map a metadata shard to its global rows, and skip decoding shards that
don't overlap a surviving CSR shard, without reading the shard payload.
These stats are additive (~57 B/shard, no format-version bump) and written
by every producer (convert / merge / append / compact); files written
before they existed are still read correctly (the query engine falls back
to streaming every shard). Readers reassemble shards transparently via the
shard-aware APIs (`obs_shard_count`, `read_obs_shard`, `obs_shards`,
`read_obs_assembled`); the legacy `read_obs` accessor errors on sharded
files with a diagnostic directing callers to the shard APIs.

A file MUST NOT carry both a single-section `obs_metadata` (type 0) and
sharded `obs_metadata_shard` (type 24) entries — the writer enforces
this. Legacy single-section files remain fully readable. See
[format.md § Sharded metadata layout](format.md#sharded-metadata-layout-section-types-2425)
for the binary layout.

### Streaming export over sharded obs/var

`pyscx.to_h5ad` / `pyscx.to_h5mu` (and `scx convert --to h5ad/h5mu`)
stream sharded obs/var through `write_dataframe_group_streaming` in
`scx-convert/src/h5ad/write.rs`: HDF5 datasets are pre-allocated to
`n_rows_kept` (computed catalog-only from `obs_metadata_shard_count`
+ catalog stats, minus deletion-vector kept-count when active), then
`obs_shards()` / `var_shards()` are drained one shard at a time and
hyperslab-written per column. Categorical (`Dictionary<_, Utf8>`)
columns maintain a running global dictionary across shards — disjoint
per-shard vocabularies (which `scx append` can produce when categories
diverge) are unified on the fly without a second pass over the data.
Peak RSS during export is bounded to one shard per matrix plus one
shard per obs/var column, even at multi-million-row scale. Legacy
single-section obs/var sources transparently fall back to the eager
`write_dataframe_group_at` writer.

## CSC sharding

A CSC sidecar is an optional **column-major view** of the same matrix
data the CSR shards hold. The on-disk shard layout is the same as a
CSR shard (76-byte header + indptr/indices/values + block index); only
the field interpretation flips axes (`n_major` is columns,
`global_offset` is `col_start`, `indices` are global row indices). See
[format.md §4.1 CSC Shard Internal Layout](format.md#41-csc-shard-internal-layout).

### When to add a CSC sidecar

Add a CSC sidecar when your workload is **column-axis-heavy**:

- **Differential expression** with small gene subsets (`pyscx.accel.rank_genes_groups(prefer_format="csc")`) — the kernel reads each chunk's columns as a single CSC slab instead of decoding every CSR row and projecting.
- **Highly variable genes** at scale (`pyscx.accel.highly_variable_genes(prefer_format="csc")` for single-batch seurat_v3) — single-pass per-column accumulators with no `O(n_vars)` row-wise scratch.
- **Per-gene QC** (`pyscx.accel.calculate_qc_metrics(prefer_format="csc")`) — gene-axis aggregations route through CSC.
- **Filtered pseudobulk** (`pyscx.accel.pseudobulk_dex(prefer_format="csc", gene_indices=...)`) — only the requested gene columns are decoded.

Skip the sidecar when the workload is **row-axis-only** (PCA — CSR
already streams shards row-major; full-pass HVG without `prefer_format`
arg; per-cell QC; subsetting by cells; ML training loaders). Both PCA
methods (covariance and randomized SVD) explicitly reject
`prefer_format="csc"` on the pyscx side; see
[api.md §pyscx.accel](api.md#pyscxaccel--rust-native-accelerators).

### Build policy: `off` / `auto` / `always`

The `csc` knob on the conversion entry points (`pyscx.from_anndata` /
`from_h5ad` / `from_10x`, `scx convert --csc`) is a three-state policy:

- **`off`** (default) — never emit a CSC sidecar; CSR-only output.
- **`always`** — always emit a CSC sidecar regardless of dataset size.
- **`auto`** — emit a sidecar only when the dataset is large enough that the
  column-axis acceleration pays for the extra write-time transpose and
  storage: `n_obs ≥ 50000` **and** `n_vars ≥ 5000`. Both thresholds are
  tunable via the `SCX_CSC_AUTO_OBS_THRESHOLD` and `SCX_CSC_AUTO_VARS_THRESHOLD`
  environment variables (set either to `0` to force a build on any shape) — see
  [architecture.md § Environment variables](architecture.md#environment-variables).

`auto` is resolved against the matrix shape at write time, so a (unimodal)
streaming conversion picks it up from the X reader's reported dimensions.

Multimodal (h5mu / MuData) inputs cannot build per-modality CSC while
streaming, and `scx build-csc` is unimodal-only (it would collapse all
modalities into one CSC transpose). So per-modality CSC is built only by
the **non-streaming** path:

- `pyscx.from_h5mu(..., stream=False)` / `scx convert --from h5mu --stream=false`
  — honors `csc="auto"`/`"always"` per modality.
- `pyscx.from_h5mu(..., csc=...)` with the default `stream=True`,
  `scx convert --from h5mu` (default streaming), and the in-memory
  `pyscx.from_mudata` all **reject** `csc="always"` and **degrade**
  `csc="auto"` to no-CSC (the streaming h5mu path emits a `UserWarning`
  when a modality would have qualified).

### `--csc-cols-per-shard`

Multi-shard CSC is the default. Each emitted CSC shard covers a
contiguous half-open `[col_start, col_end)` column range:

```
n_vars = 36000, --csc-cols-per-shard 5000 (default)
                       ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬────┐
CSC shards (8 total):  │ 0..5K│5..10K│10..15│15..20│20..25│25..30│30..35│..36│
                       └──────┴──────┴──────┴──────┴──────┴──────┴──────┴────┘
```

`scx build-csc --csc-cols-per-shard N` and the matching kwargs on
`pyscx.from_anndata`, `scx convert --csc-cols-per-shard`, and the
`--rebuild-csc` flag on mutating ops all default to **5000 columns
per shard**. Pass `0` for no cap (single CSC shard, memory permitting
— the streaming transpose will still chunk internally to respect the
`--memory-limit` budget).

To add a CSC sidecar to a file you already have, use the standalone
`pyscx.build_csc(input, output, memory_limit="4G", force=False,
csc_cols_per_shard=5000)` — the Python equivalent of `scx build-csc`. It
reads `input`'s CSR shards and writes both the CSR shards and the new CSC
sidecar to `output`. For an in-place rebuild use
`pyscx.sort(..., rebuild_csc=True)`; to emit the sidecar at write time use
`pyscx.from_anndata(..., csc="always")`.

### Why multi-shard CSC

Two reasons to split CSC by column range rather than emitting one
giant shard:

1. **Column-range pushdown.** `BackedCscReader::read_csc_columns(c_lo..c_hi)` consults `BackedCscIndex::shards_for_col_range(c_lo, c_hi)` and skips non-overlapping shards entirely; partial-overlap shards are sliced post-decode. With 5000 cols/shard on a 36K-gene matrix, a single-gene DE query touches one shard out of eight — a 7/8 I/O reduction even before catalog-level pushdown via `ShardStats.col_range`.
2. **Bounded transpose memory.** The streaming CSR→CSC transpose chunks emitted shards by column range, so peak memory during `build-csc` (and convert-time CSC) scales with `csc_cols_per_shard × n_obs × 8 bytes` rather than the full matrix.

### Cost: write-time transpose, ~equal storage

CSC is an **additive** sidecar — CSR shards stay on disk unchanged,
and the CSC shards add roughly the same compressed bytes (the same
nnz, just laid out column-major; codec compression ratios are similar
under Scx1 / Zstd / Pcodec).

Write-time cost is one full-matrix transpose. The streaming transpose
in `scx-sparse::transpose::streaming_csr_to_csc_iter_with_cap` keeps
peak RAM bounded by the `--memory-limit` budget (default 4G).
Throughput on a typical 1M-cell × 30K-gene file: ~10–20 seconds for
build-csc on a single core, dominated by codec encoding.

### Inspecting the CSC layout

`scx info` shows the CSC shard count on the Shards line and prints a
per-shard `cols a..b ({n} cols), nnz N` block when there are multiple
CSC shards. The JSON output (`scx info --json`) gains a `csc_layout`
array with `name`, `col_start`, `col_end`, `nnz` per shard.

### Mutating ops drop CSC by default

`scx append`, `scx compact`, `scx merge`, and `scx subset` change the
row layout (or the column index space, in subset's case), so the
existing CSC `indices` arrays would silently reference stale rows /
columns. Each op therefore drops the CSC sidecar by default and emits
a `log::warn!` message. Pass `--rebuild-csc` to re-emit the sidecar
against the post-op output via `scx build-csc` + atomic rename.

### CSC lifecycle

The sidecar moves through four stages over a file's life:

1. **Creation.** At conversion time via the `csc` policy (`auto` / `always`
   on `pyscx.from_anndata` / `from_h5ad` / `from_10x` / `scx convert`), or
   after the fact with `scx build-csc input.scx output.scx`. Both paths run
   the memory-bounded streaming CSR→CSC transpose and set the `has_csc`
   header flag.
2. **Consumption.** Column algorithms opt into the sidecar with
   `prefer_format="csc"` (CPU) or, for GPU `pdex_ref` / `rank_genes_groups`,
   the `gpu_csc_v3` route (the default GPU DE route when a CSC sidecar is
   present). `BackedCscReader` serves column-range
   reads with shard-level pushdown. See
   [scanpy.md § GPU-supported vs GPU-fast](scanpy.md#gpu-supported-vs-gpu-fast).
3. **Mutation drop.** Any row/column-layout-changing op (`append`,
   `compact`, `merge`, `subset`) drops the sidecar with a warning, because
   its `indices` would otherwise reference stale rows/columns.
4. **Rebuild.** Re-emit with `--rebuild-csc` on the mutating op, or run
   `scx build-csc` against the post-op file. The rebuild reads the current
   CSR shards, transposes, and writes a fresh CSC sidecar + updated catalog
   to a temp file that is atomically renamed.
