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
│  rows 0 – 9999              │ → CSR Shard 0
│  rows 10000 – 19999         │ → CSR Shard 1
│  rows 20000 – 29999         │ → CSR Shard 2
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
| CLI (`scx convert`, `scx append`, `scx subset`) | **10,000** |
| Python (`pyscx.from_anndata`, `pyscx.append`, etc.) | **16,384** |
| R (`rscx::scx_from_seurat`, `rscx::scx_from_sce`) | **16,384** |

> [!TIP]
> The default is a good starting point for most datasets. Smaller shards
> (e.g., 1,000–5,000) improve query selectivity at the cost of slightly larger
> files. Larger shards (e.g., 50,000) reduce catalog overhead for very large
> datasets.

### Shard sizing guidelines

At 5% density with 30K genes, a 10K-row shard contains ~15M non-zero values and
compresses to ~30–60 MB. Key considerations:

- **Query engine**: Smaller shards → more granular predicate pushdown → fewer cells
  read for selective queries. Diminishing returns below ~1,000 rows.
- **Training loader**: The triple-buffered pipeline (§8 of SPEC) reads full shards
  sequentially. Very small shards add per-shard overhead; very large shards delay
  shuffling. The default of 10K–16K rows is a good balance.
- **Cloud access**: The exploded `.scxd` layout stores each shard as a separate
  object. Shard coalescing (merging adjacent range reads) works best when shards
  are large enough to amortize per-request latency (~30–100 MB).
- **Append**: New cells are written as new shards at EOF. The `shard_target_rows`
  controls how many rows go into each appended shard.

## CLI commands

### Setting shard size during conversion

```bash
# Convert h5ad → SCX with 10,000 rows per shard (default)
scx convert experiment.h5ad experiment.scx

# Use a custom shard size
scx convert experiment.h5ad experiment.scx --shard-size 5000

# Convert MTX → SCX with custom shard size
scx convert /path/to/filtered_feature_bc_matrix/ experiment.scx --shard-size 20000
```

### Appending with shard size control

```bash
# Append new cells (uses default shard size for new shards)
scx append atlas.scx --input new_batch.scx

# Custom shard size for appended shards
scx append atlas.scx --input new_batch.scx --shard-size 5000
```

### Subsetting with shard size control

```bash
# Extract T cells with a custom shard size in the output file
scx subset experiment.scx --output t_cells.scx \
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
scx compact experiment.scx --output compacted.scx
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
│   section offsets/lengths, checksum       │
├──────────────────────────────────────────┤
│ Indptr    (Delta-Golomb encoded)         │
├──────────────────────────────────────────┤
│ Indices   (FOR-BP encoded)              │
├──────────────────────────────────────────┤
│ Values    (Adaptive Rice encoded)        │
├──────────────────────────────────────────┤
│ Block Index (O(1) row-range access)      │
└──────────────────────────────────────────┘
```

Each shard can independently override the file-level `codec_id` and `value_encoding`,
enabling per-shard adaptive codec selection and mixed integer/float layers.

For the full binary specification, see [format.md §CSR Shard Internal Layout](format.md#4-csr-shard-internal-layout).

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

### `--csc-cols-per-shard`

Multi-shard CSC is the default. Each emitted CSC shard covers a
contiguous half-open `[col_start, col_end)` column range:

```
n_vars = 36000, --csc-cols-per-shard 5000 (default)
                       ┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬───┐
CSC shards (8 total):  │ 0..5K│5..10K│10..15│15..20│20..25│25..30│30..35│..36│
                       └──────┴──────┴──────┴──────┴──────┴──────┴──────┴───┘
```

`scx build-csc --csc-cols-per-shard N` and the matching kwargs on
`pyscx.from_anndata`, `scx convert --csc-cols-per-shard`, and the
`--rebuild-csc` flag on mutating ops all default to **5000 columns
per shard**. Pass `0` for no cap (single CSC shard, memory permitting
— the streaming transpose will still chunk internally to respect the
`--memory-limit` budget).

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
