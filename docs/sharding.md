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
CSC were unchanged). Leave `--codec` at `auto` (it re-runs the same adaptive
selection `scx convert` does) and pin one only if you want a specific encoding;
the point of `sort` is read locality, not compression. Note `scx compact` is not
a size remedy for this — it re-encodes under the same intent axis, so it will
land where `sort` did. See
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

### Shuffling for training (`scx sort --shuffle`)

The same engine also runs the **opposite** reorder: a seeded random permutation
of the obs axis, for training rather than for queries.

```bash
scx sort --shuffle --seed 42 atlas.scx atlas.shuffled.scx
```

```python
pyscx.shuffle("atlas.scx", "atlas.shuffled.scx", seed=42)
```

**What it buys you.** `TrainingDataset` randomizes in two levels — it permutes
shard order, then Fisher-Yates shuffles rows *within* each shard group (see
[training.md § Epoch and shuffling](training.md#epoch-and-shuffling)).
Both levels are bounded by physical layout, so on a file whose rows arrived
clustered — by donor, plate, or cell type — batch composition is capped by
`shard_group_size`, and widening it costs memory linearly. Permuting the rows
once, on disk, moves that cost off the training loop: after a shuffle even
`shard_group_size=1` yields batches that look like the corpus.

**It is the inverse of a sort, and you cannot have both.** Sorting collapses
each category's predicate-index `shard_ranges` into one contiguous run;
shuffling scatters every category across every shard. `--shuffle` is therefore
mutually exclusive with `--by`, `--group-by` and `--reverse` — the engine
rejects the combination rather than silently preferring one. Shuffle the copy
you train from; keep a sorted or indexed copy for querying.

**Reproducibility.** The seed (default `42`, matching `TrainingDataset`) is
recorded in the output's provenance and is the *only* record of the
permutation — there is no key to re-derive it from. The same seed on the same
input always reproduces the same file. Two caveats: the permutation runs over
**live** rows, so a file carrying deletion vectors shuffles differently from the
same file without them (deletions are materialized away, as in `sort`); and the
row order is not stable across a `rand` crate upgrade, which is pinned by a test
rather than promised by the format.

**Output size.** Measured on `tabula_sapiens_100k` (100,000 cells, 7 CSR
shards), shuffling each per-codec fixture *at its own codec*, so the only
variable is row order:

| codec | X before | X after | delta |
|-------|----------|---------|-------|
| `scx1` | 413.3 MB | 413.3 MB | **1.000×** |
| `lz4` | 342.6 MB | 343.2 MB | 1.002× |
| `shufdelta` | 199.2 MB | 200.2 MB | 1.005× |
| `zstd` | 314.2 MB | 333.0 MB | **1.060×** |
| `pcodec` | 314.2 MB | 333.0 MB | 1.060× |

Reading it, in order of what dominates:

1. **`zstd` genuinely loses ~6%.** This is the cross-row redundancy a
   whole-shard-stream codec really does depend on, and a random permutation
   really does destroy some of it. Small, but real, and inherent to shuffling.
2. **`scx1` is exactly neutral**, as its design implies: each row's gene indices
   are coded independently of row order, so a permutation just relocates
   identically-sized blocks. `lz4` and `shufdelta` are near-neutral for the same
   reason.

**Historical note — the `auto` row is gone because it was a bug.** This table
used to carry an `auto` row reading 197.9 MB → 413.3 MB (**2.088×**), described
as a codec *flip* (`shufdelta ×7 → scx1 ×7`) that you were told to avoid by
pinning the input's own codec. That flip was not a property of shuffling: every
derived-file op built a framing config whose `decode_target` was `None` — which
is precisely the `fast` profile — so `auto` on a rewrite silently ran the
single-encode heuristic and never re-adopted `ShufDeltaZstd`. The same bug made
`scx compact` *grow* a file it was asked to shrink. `auto` now runs the same
adaptive per-shard selection on `sort`/`subset`/`merge`/`compact` that
`scx convert` does, so **leave `--codec` at `auto`** — pin a codec only when you
want that specific encoding (e.g. `scx1` for a permutation-invariant layout or
the GPU device-decode route), not to hold the file's size.

**A rewrite still costs ~1.3× on a mixed-width file, for a different reason.**
Measured on `census_500k` (500,000 cells, 782,470,575 nnz, 863.7 MiB, input
`shufdelta` with `mixed (uint16, uint32)` value encoding), `sort --by cell_type`:

| `--codec` | output | ratio | output codec |
|-----------|--------|-------|--------------|
| `auto` | 1175.6 MB | **1.298×** | `shufdelta` |
| `shufdelta` (explicit pin) | 1175.6 MB | **1.298×** | `shufdelta` |
| `fast` | 1867.2 MB | 2.062× | `scx1` |
| `auto`, sorting by an already-sorted key | 1177.8 MB | 1.301× | `shufdelta` |

Three things to read off it. `auto` is now byte-identical to an explicit
`shufdelta` pin, so the codec axis is doing exactly what it should. `fast`
reproduces the old 2× flip, which is what the bug above was. And the residual
1.3× is **not** reordering: sorting by a key the file is *already* sorted by
costs the same 1.301×, so it is the rewrite itself. The cause is the value
encoding — the input's `mixed (uint16, uint32)` comes back as a uniform
`uint32`, because the ops widen to a single file-wide encoding
(`scx_ops::helpers::widest_value_encoding`) instead of preserving each shard's.
That is a separate, still-open issue from the codec flip; if output size on a
mixed-width file matters, budget for it. See
[performance.md § Global pre-shuffle](performance.md#global-pre-shuffle-data-load-phase-1-1d)
for the second dataset and the batch-mixing numbers.

**Shard geometry is not preserved by default.** `--shard-size` defaults to 16,384
(inherited from `sort`), so shuffling a file written with a different shard size
re-shards it as well as reordering it — and shard size is exactly what quantises
batch composition. Pass `--shard-size <input's value>` to reorder only; the
command warns when the two differ.

Everything else behaves exactly as a key sort: deletions are materialized away,
the CSC sidecar and detection bitmap are dropped unless `--rebuild-csc` /
`--bitmap` is passed, `adata.raw` is not preserved, obsm/obsp/layers are
remapped, and multimodal inputs reorder every modality in lockstep.

## CLI commands

### Setting shard size during conversion

```bash
# Convert h5ad → SCX with 16,384 rows per shard (default)
scx convert experiment.h5ad experiment.scx

# Use a custom shard size
scx convert experiment.h5ad experiment.scx --shard-size 5000

# Convert MTX → SCX with custom shard size
scx convert /path/to/filtered_feature_bc_matrix/ experiment.scx --shard-size 20000

# Obs metadata is sharded on the same threshold by default (`--shard-obs auto`,
# i.e. when n_obs > --shard-size). `off` keeps one legacy obs_metadata section.
scx convert experiment.h5ad experiment.scx --shard-obs off
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

# Keep the subset query-ready: the input's predicate index cannot be carried
# over (row/column projection invalidates its shard ranges), so rebuild it
# against the output. Without an --index-* flag the subset has no predicate
# index and `filter_obs` pushdown falls back to a full obs scan.
scx subset experiment.scx t_cells.scx \
    --filter "cell_type == 'T cell'" --index-preset cellxgene
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

Categorical obs columns in a `collect()` whose rows the caller narrowed
(`filter_obs` or `limit`) carry only the categories present in the surviving
rows (AnnData/pandas convention), not the full parent dictionary —
deterministically, on both obs layouts; an unfiltered `collect()` and a full
`read_obs()` keep the declared list, unused levels included. (Exception until
`append` writes dictionaries: a filtered result drawn entirely from appended,
plain-encoded shards comes back as plain strings.)

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

## Row-projected reads skip whole shards

A row projection — a deletion vector, `adata[mask]`, an `adata[:n]` window — is
carried as `kept_to_global`, a strictly ascending list of the physical rows still
visible. The streaming aggregation kernels intersect it with each shard's row
range and visit only the rows inside; a shard the projection empties therefore
contributes nothing, and is not read at all.

Two mechanisms, because the row set arrives two ways:

- **Kernels that take the row set as an argument** (`col_sums_masked`,
  `col_nnz_masked`, `col_max_masked`, `col_min_masked`, `col_var_masked`,
  `col_sums_and_nnz_masked`, and their column-projected twins) ask
  `BackedCsrIndex::shards_with_kept_rows` for the shard list up front. That is
  catalog arithmetic — two `partition_point`s per shard, no I/O.
- **Kernels that stream a `ShardSource`** (PCA, HVG, `score_genes`, `pflog`,
  and the DE kernels — Wilcoxon, pdex and the `pts` counting pass) cannot see
  the row set: it belongs to the source. Those go through
  `ShardSource::visible_shard_indices`, which the lazy source overrides and the
  decode-prefetch drivers consult before scheduling a read. The default is
  `None` — "visit every shard" — so a source without a row filter is unaffected.
  An adapter that wraps such a source must forward the hook or it discards the
  plan; `ProjectedShardSource` (a column projection, which changes a shard's
  width and never which shards hold a visible row) does — without it,
  `pca(mask_var=…)`, and so the HVG → PCA pipeline, decoded all five shards of
  the fixture below instead of one. The tolerant reduction arm
  (`SCX_ACCEL_REDUCTION_MODE=parallel_tolerant`, which HVG `flavor="seurat"`
  uses) consults the plan too, so the saving does not depend on the mode.

Measured on a 120 x 200 file in 5 shards of 24 rows, counting shard decodes:

| request | shards decoded |
|---|---|
| `X.sum(axis=0)` / `X.getnnz(axis=0)`, no projection | 5 |
| the same, over a one-shard window | 1 |
| the same, over a mask covering shards 0 and 4 | 2 |
| `col_var` (two passes), one-shard window | 2 of 10 |
| `score_genes` / `pca`, one-shard window | 1 |
| `pca(mask_var=…)`, one-shard window (5 without the adapter forward) | 1 |
| `highly_variable_genes(flavor="seurat_v3")` (two passes), one-shard window | 2 of 10 |
| `normalize_total` + `log1p` chain, two-shard mask | 2 |
| `rank_genes_groups` / `pdex_ref`, one-shard window | 1 of 5 |
| `rank_genes_groups(pts=True)` (two passes), one-shard window | 2 of 10 |
| `rank_genes_groups(gene_chunk_size=…)`, 4 chunks, one-shard window | 4 of 20 |
| `rank_genes_groups`, 4 chunks, two-shard mask | 8 of 20 |

A plan can also be **empty** — a projection that keeps no row in any shard, as
`adata[:0]` does. That is where skipping needed care rather than just wiring: the
per-shard read is what checks that the file has not changed since the handle was
opened, so an empty plan would have answered zeros from a possibly-obsolete
mapping while `shape` on the same handle raised. Every kernel that builds a plan
therefore checks freshness up front, unconditionally — the six masked column
kernels, their six column-projected twins, and the three transform-aware ones —
and the lazy source reports "no plan" on a stale file so a driver consumer falls
back to reading and raises. (Two of those checks also close a hole that predated
the skip: the variance kernels' `n_kept == 0` early return never read either.)

The saving scales with how much of the file the projection excludes, so it is
largest exactly where it matters: a per-batch or per-condition view of an atlas.
Results are unchanged — a skipped shard's rows were already contributing nothing
— with one subtlety that is *not* free: a transform chain is applied per shard at
a global row offset, which now comes from the shard's own `row_start` rather
than a running count of rows seen, since a running count under-counts once a
shard is skipped.

One family is **not** covered, by design: row-*axis* kernels.
`calculate_qc_metrics`' fused row pass writes into `n_obs`-length vectors and
ends by checking that the shards it saw tile `[0, n_obs)`; it applies its row
projection after the pass, so its shard count is unchanged. That check is also
why `BackedCsrReader` deliberately does not override `visible_shard_indices`
even though it knows the file's deletion keep-mask. Everything else is covered —
the streaming DE kernels were the last family running their own shard loop.

The DE rows in the table above are the largest saving, because DE's shard walk
is **inner** to its gene-chunk walk — every visited shard is read once per gene
chunk, so the cost is `n_gene_chunks x visited shards`, not `n_shards`. That is
also why the per-chunk factor is only paid in *decodes* when the shard LRU
cannot hold the shards being revisited: with `cache_shards` sized to the file,
the four-chunk run above costs the same 5 decodes as a one-chunk run, and the
undersized-cache warning measures the cache against the shards a call visits
rather than against the file.

Going through the drivers also gave the DE kernels the bounded decode-prefetch
they never had — a separate effect from the skip, measured separately, and not
quoted here: it is a wall-clock claim of a shape
[docs/benchmark_manifest.md](benchmark_manifest.md) requires a manifest entry
for. Overlap helps where decode is repeated, so it is largest with no shard
cache and vanishes on a one-shard window. Depth is clamped
by the DE prefetch clamp (`scx-accel`, crate-internal) against whatever the dense
`n_obs x chunk` workspace left of `SCX_ACCEL_DE_MEMORY_BUDGET`, so a
budget-bound file falls back to depth 1 rather than growing: the pipeline holds
`depth` decoded shards where the loop held one, and on a plain backed handle
those are the same `Arc`s the LRU already holds.

## Row-group framing & scattered reads

A shard's `Block Index` is what lets a scattered read decode only the row-groups it
touches instead of the whole 16,384-row shard — the key lever for random-row training
loaders. Two shard layouts exist:

- **Unframed** (legacy, `shard_format_version = 1`, in a `format_version = 3` file): the
  block index is a single entry (or an oversized-row split) with **zero byte offsets**.
  A scattered gather has no sub-shard offsets to seek to, so it **full-shard-decodes**.
- **Framed** (`shard_format_version = 2`, in a `format_version = 4` file): the block index
  has **multiple entries with real per-group byte offsets**, so `read_rows_with` decodes
  only the touched groups via `resolve_block_index`. Framing is **codec-agnostic** (works
  for Scx1/Zstd/Pcodec/ShufDeltaZstd) and adds no extra encode cost.

**Framing is on by default at `DEFAULT_ROW_GROUP_ROWS = 256`** — the write paths
(`convert`, `from_anndata`/`from_h5ad`/`from_10x`, `scx optimize`) frame every re-encoded
CSR shard unless you pass `--row-group-rows 0` (the legacy unframed v3 opt-out). G=256 is
the scatter-friendly middle: compression ratio is flat across G (±0.3%), while finer groups
cut worst-case scattered-gather latency on multi-shard files — e.g. smartseq2 p50 `G=128`
2.18 s vs `G=512` 3.05 s (1.4×), tabula `G=128` 1.70 s vs `G=512` 3.50 s (2.1×). Prefer
G=128–256 for scatter-heavy training; coarser G only pays off for sequential full-shard
scans. (See [codec.md §7b](codec.md) for the bit-level layout and the full G sweep.)

**Loader adoption.** `IndexPlanDataset` defaults `scatter_block_index=True` and its
prefetcher never warms a block-index-eligible framed shard *whole*, so the gather reaches the
group-level path; `SparseCellSetDataset` takes the same kwarg and defaults it off, because
its regime is cache-friendly (see [performance.md](performance.md) for the measurement —
taken before the row-group LRU below existed). The process-wide kill-switch is
`SCX_SCATTER_BLOCK_INDEX=0`. Adoption is observable on **both** classes via
`cache_metrics()["block_index_groups"]` (> 0 ⇒ framed path taken; `full_shard_groups` is the
fallback).

**Row-group LRU (OPT-FORMATIO-1).** The groups a block-index gather decodes are retained in
the same LRU as whole shards, keyed `(file_id, shard, group)`, under the same byte budget
(`max_memory_mb`); the `cache_shards` count cap applies to whole shards only. The per-shard
framing layout (header scalars, sub-stream ranges, resolved block index) is memoized once
per shard. A repeated gather over a hot region is served as
`cache_metrics()["row_group_hits"]`, and the L2 prefetcher pre-decodes a plan's groups when
they fit `budget / (lookahead + 1)`. `SCX_ROW_GROUP_CACHE=0` disables retention (the
same-build A/B arm). Output is byte-identical either way. Opening **all-unframed** data with
`scatter_block_index=True` emits a one-shot `UserWarning` on **both** classes — the fast
path is inert there, so reframe with `scx optimize --row-group-rows 256 <file>`. On
`SparseCellSetDataset` the check is *any file in the set*, since one framed file means the
route is live for that file's rows; a mixed set therefore does not warn, and
`cache_metrics()["block_index_groups"] > 0` remains the way to confirm a gather actually
took the route.

## Condition/label-grouped sharding (F1) + grouped reads (F2)

By default shards are cut by a fixed row count (`--shard-size`). For
perturbation screens it pays to instead cluster cells by condition so that
reading one perturbation touches one shard, and to isolate reference cells
(e.g. `non-targeting`) in their own leading shard. `scx sort --group-by`
does exactly that:

```bash
# --reference takes a comma-separated label list, or `col:<name>` (the
# `column:<name>` spelling is also accepted) for a boolean obs column.
# --group-target-bytes is optional; omit it to pack by --shard-size rows
# instead of a byte budget.
scx sort screen.scx grouped.scx \
  --group-by target_gene \
  --reference non-targeting \
  --group-target-bytes 256M
```

The sort forces `--group-by` to be the leading sort key (so the column is
also predicate-indexed), computes a **reference-first, group-clustered**
global order, then runs an offline byte-budget bin-packer that:

- packs all reference rows first into the leading shard(s) (`reference_shard`,
  always shard 0), never mixed with groups;
- bin-packs the remaining groups greedily to the byte (or row) budget,
  **the planner cuts only at group edges** — the group *plan* never splits a group;
- gives any single oversized group its own planned shard.

> **F6 — oversized groups & the in-memory fast path.** The *emitter* may
> sub-flush a single oversized group across **multiple output shards** once its
> buffered CSR reaches `--group-write-block-bytes` (default 256 MB), so grouped
> write no longer OOMs on a huge reference group (e.g. a 127K-cell
> `non-targeting`). Such a group then has **several `group_index` records sharing
> one label** (each still within one shard); the read side unions them, so
> `read_group` / `read_reference` return the whole group transparently. In the
> default in-memory path (no `--memory-budget`), grouped X is gathered from the
> resident CSR and encoded **in parallel** (rayon) — byte-identical to the
> single-threaded path but ~2–4× faster (see [performance.md](performance.md)).
> The parallel gather trades peak RSS for speed; cap it with `RAYON_NUM_THREADS`,
> or set `SCX_SORT_NO_INMEM_FAST=1` to force the memory-lean single-threaded
> emitter. `--group-write-block-bytes 0` disables the sub-flush entirely.

`--reverse` is ignored in grouped mode (reference must sort first). The full
group contract is persisted in a self-describing `group_index` sidecar
(section id 29, JSON: `{group_by, reference_shard, reference_labels,
records[]}`; each record `{label, shard, row_start, row_stop, role}` with
**global** output-row ranges). Pre-F1 readers skip the unknown section, so
grouped archives stay readable.

> `bytes_per_nnz` is a packing *estimate* (in-memory width), not the
> on-disk size, so tools assign groups to shards
> differently while agreeing on order, roles, and per-label ranges.

CSC sidecars are dropped by the reorder as usual — pass
`--group-by … --rebuild-csc`. Grouping is single-modality only in v1.

Convert-time grouping (`scx convert --group-by`) picks a one-pass or two-pass
route: `--group-pass auto` (default) streams CSR X in one pass and routes dense
X — **or any input that carries `obsp`** — through the two-pass path (plain
convert then `scx sort --group-by`) so `obsp` is remapped and preserved rather
than dropped. This means a CSR h5ad that carries `obsp` (e.g. `connectivities`)
takes the heavier two-pass route (a transient temp file) by default; forcing
`--group-pass one` on such an input errors and points at `--group-pass two`.

### Grouped reads

The sidecar powers a grouped-read API that defaults to slicing the recorded
global row range directly (decoding only the covered shards):

```python
exp = pyscx.open("grouped.scx")
exp.read_group("MYC")              # -> AnnData (just the MYC cells)
exp.read_reference()               # -> AnnData | None (the full reference region)
exp.group_labels()                 # -> list[str]
for gs in exp.iter_group_shards(): # streaming, ~one shard resident
    ad = gs.to_anndata()           # gs.labels, gs.shard_index, gs.global_start/stop
```

`read_group` raises `KeyError` (with close-match suggestions) for an unknown
label and `ValueError` if the archive is not grouped. On a label that splits
across the reference / non-reference boundary (only possible with
`--reference col:<name>`), `read_group` returns the non-reference record and
`read_reference` serves the reference rows. The Rust seam is
`scx_engine::QueryPipeline::{read_group, read_reference, group_labels,
iter_group_shards, read_row_range}`.

> **F6 note.** `read_group` / `read_reference` are safe under the F6 sub-flush:
> they union all of a label's records via `GroupIndex::label_range` /
> `reference_range`, so a group split across shards reads back in full. The
> lower-level **streaming** surface (`iter_group_shards` / `GroupShardHandle`)
> yields one handle **per shard**, so an oversized group now appears in
> **several** handles (each covering its shard-local slice) — a streaming
> consumer must accumulate across handles rather than stop at the first handle
> mentioning a label.

### Limitations & staleness

The `group_index` sidecar is **write-once** — produced only by `scx sort
--group-by`. Mutating an archive afterward affects it as follows:

- **`scx append` drops the sidecar** (with a warning), because its records hold
  global row ranges over the pre-append row universe. After an append the file
  is ungrouped and grouped reads raise `ValueError` (`NotGrouped`); re-run `scx
  sort --group-by` to regroup.
- **`scx merge` / `scx subset` produce fresh, ungrouped files** (no sidecar is
  written) — re-sort the output to regroup.
- **`scx compact` does not propagate the sidecar**, so a compacted file is
  ungrouped.
- **Deletion vectors are honored** by grouped reads: if you `mark_deleted` /
  `delete_cells` on a grouped file, `read_group` / `read_reference` /
  `iter_group_shards` drop the deleted rows, staying equivalent to
  `query().filter_obs(...).collect()`. (A fresh grouped output has no deletion
  vectors — the sort materializes them away.)
- **Grouped reads are *raw*.** They return the cells of the range only — builder
  state on a `QueryPipeline` (`filter_obs` / `filter_var` / `select_genes` /
  `with_normalize` / `with_log1p` / `limit`) is **not** applied. The pyscx
  `Experiment.read_*` methods are structurally safe (each opens a fresh
  pipeline); compose transforms via `query().collect()` instead.
- **Cloud, `rscx`, and native `pyscx.sort` are all supported.** `open_cloud(...)`
  exposes the same grouped-read methods over range reads (the cloud
  `SectionReader` implements `read_group_index_bytes`); `rscx` has grouped
  bindings; and `pyscx.sort(..., group_by=..., reference=...)` produces a grouped
  file natively (in addition to the `scx sort --group-by` CLI and convert-time
  `scx convert --group-by` / `pyscx.from_h5ad(group_by=...)`).

> On row-sharded-obs files (streaming merge/append/`from_anndata` at
> `n_obs > shard_size`, or convert-time grouping), `read_group` reads only the
> `ObsMetadataShard`s overlapping the group's range, so obs is not materialized
> in full. Legacy single-section obs falls back to a full read then slice.

## Obs/var metadata sharding

Streaming `scx merge`, `scx append`, and `pyscx.from_anndata` (when
`n_obs > shard_size`) write obs and var metadata as row-sharded
Arrow IPC sections (section types 24/25) instead of a single monolithic
`obs_metadata` / `var_metadata` section. This bounds peak memory at one
shard's worth of metadata during the merge/append hot path and avoids
Arrow IPC's 2 GB narrow-offset ceiling for string columns at atlas scale.

Obs/var **metadata** sharding is decided independently of X sharding.

**The obs axis on ingest.** `scx convert` (h5ad, h5mu, 10x and mtx alike),
`pyscx.from_h5ad`, `pyscx.from_h5mu` and `pyscx.from_mtx` shard obs under
`--shard-obs off|auto|always` / `shard_obs=`, default `auto` — the same tri-state, the same
`ObsShardPolicy::parse`, and the same `n_obs > shard_target_rows` boundary
`scx optimize --shard-obs` and `pyscx.from_anndata` use, so the four
producers converge on one layout for a given `n_obs`. Shard boundaries come
from the shared `scx_format_io::write_obs_section`, not a per-producer copy of the
loop.

> This is a **layout** choice, not a memory one. Ingest reads obs whole
> (`read_dataframe_group`) and `RecordBatch::slice` is zero-copy, so
> conversion peak RSS is identical either way — and a `>2 GB` obs string
> column still overflows `i32` offsets in the *reader*, before any of this
> runs. What sharding buys is the bounded layout for downstream streaming /
> cloud / bounded-memory readers, and putting converted files on the
> streaming h5ad export writer (see [Streaming export over sharded
> obs/var](#streaming-export-over-sharded-obsvar) below), which is otherwise reachable only from `from_anndata` output.

**The var axis on ingest is never sharded.** Every ingest path writes a
single `var_metadata` section regardless of `n_vars`; `ObsShardPolicy` is
obs-scoped, matching `scx optimize`. (`pyscx.from_anndata` *does* shard var
above the threshold — a divergence between the in-memory and on-disk ingest
routes, not yet reconciled.)

**Other ingest routes.** `pyscx.from_mudata` (the in-memory MuData sibling of
`from_anndata`) still writes a single-section obs at any scale. To shard it,
migrate the output with
`pyscx.compact(src, dst, reshape_obs=True)` (`scx compact --reshape-obs`) or
`pyscx.optimize(src, dst, shard_obs=...)` (`scx optimize --shard-obs`).

⚠️ **One observable difference between the two layouts**: a categorical read
back from sharded obs carries a dictionary key narrowed to the minimal fit
(e.g. `Dictionary(Int8, Utf8)` for four categories), because
`assemble_sharded_metadata` widens every shard's key to `Int32` for the
concat and `unify_dictionary_columns` narrows it again after dedup. A
single-section read has nothing to unify and keeps whatever key the source
reader built (`Int32` from an h5ad categorical group). Values, the declared
category list and the `ordered` bit are identical, and pandas does not
observe key width — but Rust code that downcasts to
`DictionaryArray<Int32Type>` does.

`scx optimize --shard-obs off|auto|always` (default `auto`) migrates a
**single-section** obs table to the sharded layout while it modernizes the
file (row-group-framed v4 canonical CSR). `auto` shards only when
`n_obs > shard_target_rows` — the same threshold `from_anndata` uses — so
small files stay single-section and byte-faithful while atlas-scale files get
sharded obs in the same pass; `always` shards unconditionally and `off`
preserves the single section (the historical 1:1 copy). An already-sharded
obs is always stream-preserved regardless of the policy (optimize never
collapses or re-sizes existing obs shards — use `compact` to re-shard).
optimize's peak memory is unchanged: the single-section input is materialized
whole by `read_obs()` either way, so the benefit accrues to downstream
streaming / cloud / bounded-memory readers of the output, not to optimize
itself.

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
stream sharded obs/var through `write_dataframe_group_from_shards` in
`scx-convert/src/h5ad/column_stream.rs`: HDF5 datasets are pre-allocated to
`n_rows_kept` (computed catalog-only from `obs_metadata_shard_count`
+ catalog stats, minus deletion-vector kept-count when active), then
`obs_shards()` / `var_shards()` are drained one shard at a time and
hyperslab-written per column. Categorical columns union each shard's
declared vocabulary in declared order — disjoint per-shard vocabularies
(which `scx append` can produce when categories diverge) are unified on
the fly, and a shard that arrives as a plain array rather than a
dictionary contributes its distinct values in first-appearance order,
which is all such a shard carries.
Peak RSS during export is bounded to one shard per matrix plus one
shard per obs/var column, even at multi-million-row scale. A legacy
single-section obs/var source goes through the whole-batch driver
`write_dataframe_group_at`, which is the same writer over a one-shard
iterator — not a separate implementation.

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

- **`off`** — never emit a CSC sidecar; CSR-only output. This is what an
  unset `csc` resolves to *unless* an accel-ready `index_preset` upgrades
  it (see below) — the default is deferred, not a pinned `"off"`.
- **`always`** — always emit a CSC sidecar regardless of dataset size.
- **`auto`** — emit a sidecar only when the dataset is large enough that the
  column-axis acceleration pays for the extra write-time transpose and
  storage: `n_obs ≥ 50000` **and** `n_vars ≥ 5000`. Both thresholds are
  tunable via the `SCX_CSC_AUTO_OBS_THRESHOLD` and `SCX_CSC_AUTO_VARS_THRESHOLD`
  environment variables (set either to `0` to force a build on any shape) — see
  [architecture.md § Environment variables](architecture.md#environment-variables).

`auto` is resolved against the matrix shape at write time, so a (unimodal)
streaming conversion picks it up from the X reader's reported dimensions.

On the pyscx entry points and `scx convert`, the default is a *deferred*
`off`: leaving `csc` unset lets an accel-ready `--index-preset` /
`index_preset=` (`training` or `perturbseq`, whose substrate is the
column-major sidecar) upgrade it to `auto`. `cellxgene` is query-oriented
and does not. Passing any explicit value — including `off` — always wins.
The rule lives in `scx_engine::index::resolve_csc_policy` so the CLI and
pyscx cannot drift apart.

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
`pyscx.build_csc(input, output=None, memory_limit="4G", force=False,
csc_cols_per_shard=5000)` — the Python equivalent of `scx build-csc`. It reads
`input`'s CSR shards and re-emits them alongside the new CSC sidecar.
`output=None` (the default) does that **in place** via a temp file + atomic
rename; passing a path writes a copy and leaves `input` alone. `force` applies
only to the copy-out form. Either form preserves the input's row-group framing.
To emit the sidecar at write time use `pyscx.from_anndata(..., csc="always")`.

### Why multi-shard CSC

Two reasons to split CSC by column range rather than emitting one
giant shard:

1. **Column-range pushdown.** `BackedCscReader::read_csc_columns(c_lo..c_hi)` consults `BackedCscIndex::shards_for_col_range(c_lo, c_hi)` and skips non-overlapping shards entirely; partial-overlap shards are sliced post-decode. With 5000 cols/shard on a 36K-gene matrix, a single-gene DE query touches one shard out of eight — a 7/8 I/O reduction even before catalog-level pushdown via `ShardStats.col_range`.
2. **Bounded transpose working set.** The streaming CSR→CSC transpose chunks emitted shards by column range, so the *transpose's own* working set scales with `csc_cols_per_shard × n_obs × 8 bytes` rather than with the full matrix. That is not the same as bounding the op — see [Cost](#cost-write-time-transpose-equal-storage) below.

### Cost: write-time transpose, ~equal storage

CSC is an **additive** sidecar — CSR shards stay on disk unchanged,
and the CSC shards add roughly the same compressed bytes (the same
nnz, just laid out column-major; codec compression ratios are similar
under Scx1 / Zstd / Pcodec).

Write-time cost is one full-matrix transpose. The streaming transpose
in `scx-sparse::transpose::streaming_csr_to_csc_iter_with_cap` keeps
its own working set inside the `--memory-limit` budget (default 4G) —
but the op holds every decoded CSR shard resident alongside it, so the
**process** peak is not bounded by that budget: measured at 2.0x the
declared 4 GiB on `census_500k` and 3.6x on `census_1m`.

Measured throughput, from the `build_csc` rows of
`benchmarks/comprehensive/results/baselines/LATEST` (`scx_auto`,
`--memory-limit 4G`): 0.22 s on `pbmc3k` (2.3M non-zeros), 19.4 s on
`smartseq2` (131M), 29.3 s on `tabula_sapiens_100k` (195M), 201 s on
`census_500k` (747M) and 1308 s on `census_1m` (1.40B). Cost tracks
non-zeros, not cells. Encoding and the transpose dominate — removing one of
the two full CSR decode passes the op used to perform, plus `n + 1` standalone
shard-header reads, took roughly a tenth off the wall. That split comes from a
branch A/B rather than a promoted capture, so it is not quoted as a figure; see
[CSC Sidecar Architecture](architecture.md#csc-sidecar-architecture).

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
   after the fact with `scx build-csc input.scx` (in place) or
   `scx build-csc input.scx output.scx` (copy out). All paths run the
   memory-bounded streaming CSR→CSC transpose and set the `has_csc` header
   flag.
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
