# Lazy preprocessing

> Part of the [SCX + scanpy guide](README.md).

## Lazy preprocessing in backed mode

SCX supports **materialization-free preprocessing** for backed mode.
Instead of loading the entire expression matrix into RAM, `pyscx.accel.normalize_total()`
and `pyscx.accel.log1p()` create a **lazy transform wrapper** that applies
transformations on-read — data stays on disk.

### Lazy vs eager preprocessing

| `device=` | Behaviour |
|-----------|-----------|
| `"cpu"` / `"auto"` (no GPU) | **Lazy**: wraps `adata.X` in an `ScxLazyTransformedDataset` (see below). No materialization. |
| `"gpu"` / `"auto"` (GPU available) | **Eager**: streams shards through GPU kernels (`gpu_preprocess_to_csr`) and replaces `adata.X` with a materialized scipy CSR. A `UserWarning` is emitted when the prior X was already lazy so the broken chain is visible. |

**Fusion detection** — `normalize_total(device="gpu")` stashes a marker on
`adata.uns["__scx_gpu_pending_normalize__"]`. A subsequent
`log1p(device="gpu")` on the same AnnData consumes the marker and re-runs a
**single fused normalize+log1p pass over the original backed source** rather
than reading the already-materialized scipy CSR. Any intervening op (PCA,
kNN, …) silently forfeits the fusion, producing correct — but 1× redundant —
results.

**Scipy/dense fallback (symmetric for `normalize_total` and `log1p`)** — when
`device="gpu"` is passed but X is already a materialized scipy/dense matrix,
GPU dispatch on these per-row ops is dominated by H→D + D→H copies and
makes no sense. Both `pyscx.accel.normalize_total(device="gpu")` and
`pyscx.accel.log1p(device="gpu")` detect this, emit a `UserWarning` naming
the gating condition (X must be `ScxBackedSparseDataset` or
`ScxLazyTransformedDataset`), and delegate to `sc.pp.normalize_total` /
`sc.pp.log1p` on the host. To get the GPU fast path — including the
single-pass `normalize+log1p` fusion via the marker on
`adata.uns["__scx_gpu_pending_normalize__"]` — open via
`pyscx.open(...).to_anndata(backed=True)` (or keep the source as
`ScxLazyTransformedDataset`) and avoid `.copy()` between `normalize_total`
and `log1p` (that materialises X to scipy CSR and unreachably forfeits the
fast path for the rest of the pipeline).

**HVG input — backed, lazy, or materialized X.** `flavor` in `seurat_v3` /
`seurat_v3_paper` / `seurat` runs the scx-native streaming kernel whether
`adata.X` is an `ScxBackedSparseDataset`, an `ScxLazyTransformedDataset`, or a
plain materialized scipy/dense matrix (a materialized `X` is wrapped in a
single-shard `ShardSource`). So the common `pyscx.open(...).query()...collect()
.to_anndata()` (eager) idiom gets the same numerics — and the same per-batch
LOESS-singularity tolerance — as the backed path. Only flavors scx does not
implement natively (today `cell_ranger`) delegate to
`scanpy.pp.highly_variable_genes`, with a one-shot `UserWarning`.

> **`batch_key` cardinality is the usual LOESS-singularity trigger.** `seurat_v3`
> fits one `skmisc.loess` per batch; a high-cardinality key such as CELLxGENE
> `dataset_id` produces many tiny batches whose log-mean / log-variance
> regression is singular. The native path catches each such fit, warns naming
> the batch, and drops it from the ranking — so HVG completes. If many batches
> drop, prefer a coarser `batch_key` (or none). `filter_genes(min_cells=10)`
> only helps the single global fit, not the per-batch case.

**HVG on GPU** — `pyscx.accel.highly_variable_genes(device="gpu")` routes
through GPU atomicAdd kernels for `streaming_mean_var` and
`streaming_clip_square_sum`, including per-batch variants when `batch_key`
is set (materialized X uses the same single-shard `ShardSource`). GPU dispatch
is active for any `seurat_v3` configuration regardless of `batch_key`;
`flavor="seurat"` still falls back to CPU with a `UserWarning`. The per-batch
loess fits run on CPU via `skmisc.loess` — a batch whose log-mean / log-variance
regression is too degenerate to fit (small batch sizes, near-collinear inputs)
is caught, surfaced as a `UserWarning` naming the batch, and excluded from the
per-batch ranking; other batches proceed normally.


### Quick example

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# These do NOT materialize — they create/extend a lazy wrapper
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)

# adata.X is now an ScxLazyTransformedDataset (still on disk)
print(type(adata.X))  # <class 'pyscx.ScxLazyTransformedDataset'>

# Slicing decodes, transforms, and returns a scipy CSR
chunk = adata.X[0:1000]  # normalized + log1p'd scipy CSR (1000 × n_vars)

# PCA streams through the transforms shard-by-shard
pyscx.accel.pca(adata, n_comps=50)
```

### Memory savings

| Step | `sc.pp.*` (materializes) | `pyscx.accel.*` (lazy) |
|------|--------------------------|------------------------|
| After `normalize_total()` | ~8 GB (1M cells) | ~8 MB (cached row sums) |
| After `log1p()` | ~8 GB (in-place on CSR) | ~0 bytes (transform enum) |
| PCA peak | ~8 GB + working matrices | ~128 MB (1 shard) + working matrices |

### How it works: ScxLazyTransformedDataset

When you call `pyscx.accel.normalize_total(adata)` on backed data, it:

1. **Computes row sums** via streaming (no materialization — `BackedCsrReader::row_sums()`)
2. **Creates an `ScxLazyTransformedDataset`** wrapping the backed reader with a `NormalizeTotal` transform
3. **Replaces `adata.X`** with this lazy wrapper

When you subsequently call `pyscx.accel.log1p(adata)`, it:

1. **Detects** that `adata.X` is already an `ScxLazyTransformedDataset`
2. **Appends** a `Log1p` transform to the existing chain

When data is accessed (e.g., `adata.X[100:200]` or by `pyscx.accel.pca()`),
each shard is decoded → normalized → log1p'd on the fly. Peak memory is
one shard (~128 MB), not the full matrix.

### Transform types

| Transform | Created by | Effect |
|-----------|------------|--------|
| `NormalizeTotal` | `pyscx.accel.normalize_total()` | Divides each row by its precomputed sum, multiplies by `target_sum` |
| `Log1p` | `pyscx.accel.log1p()` | Element-wise `ln(x + 1)` on non-zero values |
| `RowScale` | `__truediv__` interception (experimental) | Multiplies each row by a per-row scalar |

When `NormalizeTotal` + `Log1p` are chained (the common case), they are
automatically **fused** into a single pass: `ln(x × target_sum / row_sum + 1)`,
avoiding an intermediate normalized array.

### Aggregation through transforms

The lazy wrapper supports streaming aggregation *after* transforms:

```python
# These stream shards, apply transforms, then aggregate — no materialization
adata.X.sum(axis=0)    # column sums of normalized+log1p'd data
adata.X.sum(axis=1)    # row sums of normalized+log1p'd data
adata.X.var(axis=0)    # column variance of transformed data
```

> [!NOTE]
> Aggregation through transforms is ~2× slower than raw aggregation
> (each shard must be decoded, transformed, then aggregated). However,
> peak memory remains O(shard_size) — ~128 MB vs ~8+ GB for full
> materialization.

### Materializing when needed

To get a full scipy CSR matrix (e.g., for a function that requires it):

```python
# Explicit materialization
csr = adata.X.to_memory()  # scipy CSR with all transforms applied
# or
csr = adata.X.copy()       # same as to_memory()
```

### Complete out-of-core pipeline

This pipeline processes 1M+ cells with ~11 GB peak RSS (vs ~22 GB materialised — 51% reduction). The lazy preprocessing path (normalize + log1p + streaming PCA) peaks at ~3.5 GB; the remaining RSS is dominated by kNN graph construction and UMAP:

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC — fully streaming, including gene subsets (column-projected streaming aggregation)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# HVG — streaming variance
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Preprocessing — lazy, no materialization
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)

# Dimensionality reduction — streams through lazy transforms
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)

sc.pl.umap(adata, color="leiden")
```

### `__truediv__` interception (experimental)

As an optional convenience, `ScxBackedSparseDataset.__truediv__()` detects
when it receives a per-row scaling vector (as produced by scanpy's
`normalize_total` internals) and returns a lazy `ScxLazyTransformedDataset`
instead of materializing. This means `sc.pp.normalize_total(adata)` may
avoid materialization in some cases.

> [!WARNING]
> **This interception is experimental.** Modern scanpy (≥1.10) uses a
> numba-jitted `_normalize_csr()` function that accesses `.data`/`.indptr`
> attributes directly for CSR matrices. Since `ScxBackedSparseDataset`
> doesn't expose raw CSR attributes, scanpy falls back to the non-CSR path
> where `__truediv__` interception *does* work — but this behavior depends
> on scanpy's internal code paths and may change between versions.
>
> **Recommendation:** Always use `pyscx.accel.normalize_total()` and
> `pyscx.accel.log1p()` for reliable, version-independent lazy preprocessing.

## Preprocessing pipeline (write-back)

For in-place modifications that need to persist, use the streaming
preprocessing pipeline. This reads shards one at a time, applies fused
operations in Rust, and writes the result to a new SCX file:

```python
import pyscx

# Normalize + log1p → new file (no full materialization)
pyscx.preprocess(
    "raw.scx",
    "preprocessed.scx",
    ops=["normalize_total", "log1p"],
    target_sum=1e4,
)

# Save as a named layer in an existing file
pyscx.save_layer(
    "experiment.scx",
    "experiment_with_norm.scx",
    layer_name="normalized",
    ops=["normalize_total", "log1p"],
    target_sum=1e4,
)

# Then use the preprocessed file
adata = pyscx.open("preprocessed.scx").to_anndata()
sc.pp.pca(adata)  # Already normalized — skip normalize/log1p
```

The preprocessing pipeline uses fused `normalize_total + log1p` in a
single pass over each shard, which is faster than the equivalent scanpy
calls on large datasets.

> [!TIP]
> **Choosing between lazy preprocessing and write-back:**
> - **Lazy** (`pyscx.accel.*`): Best for interactive analysis — no disk I/O,
>   instant, transforms applied on-read. Original data preserved.
> - **Write-back** (`pyscx.preprocess()`): Best when you need a persistent
>   preprocessed file (e.g., for repeated analysis or sharing). Writes a new
>   SCX file with transforms baked in.

## See also

- [Backed mode and out-of-core iteration](backed-mode.md).
- [Common scanpy workflows § Accelerated pipeline for large datasets](workflows.md#accelerated-pipeline-for-large-datasets).
- [Gene-set scoring and PFlog](accel-scoring-normalization.md) — PFlog, the
  other streaming normalization.
