# Backed mode and out-of-core iteration

> Part of the [SCX + scanpy guide](README.md).

## Backed mode (lazy loading)

For atlas-scale datasets where full materialization is impractical, SCX
supports **backed mode** — data stays on disk and is loaded on demand, one
shard at a time:

```python
import pyscx
import scanpy as sc

# Open in backed mode — X stays on disk
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

print(type(adata.X))  # <class 'pyscx.ScxBackedSparseDataset'>
print(adata.X.shape)   # (1000000, 33694) — no data in memory yet

# Slicing loads only the needed shards
subset = adata.X[100:200]  # returns scipy.sparse.csr_matrix, ~1 shard decode
```

### How it works

When `backed=True`:

- `X` is an `ScxBackedSparseDataset` (not a materialized CSR matrix)
- **Layers** are wrapped in `ScxBackedLayerDataset` — also lazy
- **obs, var, uns** are loaded eagerly (same as non-backed — these are
  small relative to X)
- **obsm** is loaded eagerly by default, but `obsm=[...]` (without
  `eager=True` / `obs_filter`) makes each selected key a lazy
  `ScxBackedObsmDataset` row-gather dataset — see [Selective + lazy
  `obsm`](loading.md#selective--lazy-obsm-obsm) above
- The dataset is **read-only** (matching AnnData's `backed="r"` semantics)

Each access to `adata.X[rows, cols]` decompresses only the CSR shards that
overlap the requested rows. For a 1M-cell file with 64 shards, slicing 1K
cells touches ≤ 1 shard.

### Testing whether a matrix is lazy: `pyscx.is_backed_handle`

```python
if pyscx.is_backed_handle(adata.X):
    adata.X = adata.X.to_memory()   # scipy CSR
```

True for all four handle classes — `ScxBackedSparseDataset`,
`ScxBackedLayerDataset`, `ScxBackedObsmDataset`,
`ScxLazyTransformedDataset` — and False for anything already in memory.

**Do not reach for `scipy.sparse.issparse` here.** It returns `False` for a
handle and cannot be made to return `True`: scipy's `sparray` / `spmatrix` are
concrete classes rather than ABCs, so there is no `register()` seam. The usual
guard therefore takes the wrong arm, silently:

```python
sp.issparse(adata.X)          # False  -> the usual guard takes the DENSE arm
np.asarray(adata.X)           # TypeError: use to_memory() / toarray() / a row window
```

The dense arm is the wrong arm; the `np.asarray` it typically leads to is now
a loud `TypeError` on the three sparse handles rather than the 0-d object array
it used to return (which surfaced much later as "setting an array element with
a sequence") — decoding an atlas because a guard misrouted is the worse
failure. The dense `ScxBackedObsmDataset` still materialises under
`np.asarray`. A **row** slice of a handle *does* yield genuine scipy sparse, so
`sp.issparse(adata.X[0:10])` is `True`; a **column** selector (`X[:, 5]`,
`X[:, genes]`, `X[:, 10:20]`) returns another handle, for which `issparse` is
again `False` and `np.asarray` again raises — test a row slice, or use this
predicate on the matrix.

### Indexing patterns

| Access | Behavior |
|--------|----------|
| `X[100:200]` | Row slice → decodes 1 shard |
| `X[[0, 5, 10]]` | Fancy index → one decode per touched shard, result assembled once in request order (duplicates and negative indices allowed) |
| `X[mask]` | Boolean mask (length must equal `n_obs`) → one decode per touched shard, result assembled once; peak memory = result + the shard cache + up to `cache_shards` shards decoding in flight while it fills (never a second copy of the result) |
| `X[:]` | Whole matrix → exact-size result; without deletion vectors the shards decode uncached in parallel (costs what `to_memory()` costs), with deletion vectors it is the row gather over the kept rows |
| `X[100:200, :500]` | Row slice + column filter → 1 shard + post-filter |
| `X[:, hvg_idx]` / `X[:, mask]` | Column projection → a new **handle**, no decode; the projection is applied shard by shard on later reads / aggregations |
| `X[:, 5]` / `X[:, [7, 2, 11]]` / `X[:, 10:20]` | Same — `int`, `list`, `range`, `slice` and any-order ndarray all project; a reordered request keeps its order (`sum(axis=0)` too). Before 0.17 these decoded the whole matrix |
| `X[:, [3, 1, 3]]` | Repeated columns → scipy built from the 2 projected columns, never the whole matrix (a lazily transformed `X` does the same for any reorder) |
| `X[0, 5]` | Scalar → returns `float` |
| `X[:, [0, 10**9]]` / `X[:, np.array([1.5])]` | Out-of-range or float column selector → `IndexError` (float used to slip through and decode everything) |
| `X[[0, 10**9]]` | Out-of-range row → `IndexError` (never a shorter matrix) |

The same rows are reachable without an AnnData through
`Experiment.gather_rows_sparse(rows, layer=None, logical=True)` — see
[api/python-experiment.md](../api/python-experiment.md#experiment). `adata.layers[name][rows]` gathers from a layer.

### scanpy operations in backed mode

| Operation | Works? | Notes |
|-----------|--------|-------|
| `sc.pp.filter_cells()` | ✅ | Native Rust row sums/NNZ |
| `sc.pp.filter_genes()` | ✅ | Native Rust column sums/NNZ |
| `sc.pp.calculate_qc_metrics()` | ✅ | `(X > 0).sum()` short-circuits to `getnnz()` |
| `sc.pp.calculate_qc_metrics(qc_vars=)` | ✅ | Column-projected streaming aggregation (non-materializing) |
| `sc.pp.highly_variable_genes()` | ✅ | Native Rust column variance |
| `adata[mask].copy()` | ✅ | Subset → materialize → preprocess |
| `sc.pp.pca()` | ✅ | Forces materialization of HVG columns |
| `pyscx.accel.normalize_total()` | ✅ | Lazy — no materialization |
| `pyscx.accel.log1p()` | ✅ | Lazy — no materialization |
| `sc.pp.normalize_total()` | ⚠️ | Use `pyscx.accel.normalize_total()` instead — see [Lazy preprocessing](lazy-preprocessing.md#lazy-preprocessing-in-backed-mode) |
| `sc.pp.log1p()` | ⚠️ | Use `pyscx.accel.log1p()` instead — see [Lazy preprocessing](lazy-preprocessing.md#lazy-preprocessing-in-backed-mode) |

> [!TIP]
> For `normalize_total` and `log1p` in backed mode, use `pyscx.accel.*`
> instead of `sc.pp.*`. The accelerator functions create lazy transform
> wrappers that keep data on disk — no materialization needed. See the
> [Lazy preprocessing](lazy-preprocessing.md#lazy-preprocessing-in-backed-mode) section below.

**Traditional workflow** (materialize → preprocess):

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Subset to cells of interest, then materialize
adata_sub = adata[adata.obs["cell_type"] == "T cell"].copy()

# Now X is a regular scipy CSR — in-place ops work
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)
sc.pp.pca(adata_sub)
sc.pp.neighbors(adata_sub)
sc.tl.leiden(adata_sub)
```

**Zero-materialization workflow** (recommended for large datasets):

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Preprocessing stays lazy — data never leaves disk
pyscx.accel.normalize_total(adata, target_sum=1e4)  # → ScxLazyTransformedDataset
pyscx.accel.log1p(adata)                             # → appends transform

# PCA streams through the lazy transforms
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)
```

> **HVG masking without subsetting.** `pyscx.accel.pca(adata, mask_var=...)`
> restricts PCA to the selected genes in-place (scanpy semantics): pass a
> var-column name or a boolean array, or leave `mask_var=None` to auto-consume
> `adata.var["highly_variable"]` when present. It projects columns on the fly
> (backed/lazy/in-memory) — no HVG subset copy — and `varm["PCs"]` stays aligned
> to the full `var` axis (excluded genes filled with 0), so you never need the
> `adata[:, adata.var["highly_variable"]]` slice before PCA.
>
> **Behavior change (v0.11.6+):** matching scanpy, `mask_var=None` now
> **auto-consumes** `adata.var["highly_variable"]` when that column exists — so a
> PCA that previously ran on all genes will run on the HVG subset if you have
> flagged HVGs. `uns["pca"]["params"]["use_highly_variable"]` records whether a
> mask was applied. To force all genes, pass an all-`True` `mask_var`.

### Backed layers

Layers are also lazy in backed mode:

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

print(type(adata.layers["raw"]))  # ScxBackedLayerDataset
raw_slice = adata.layers["raw"][0:1000]  # decodes only needed shards
```

### Deletion vector support

Backed mode transparently handles files with deleted rows. If cells have
been marked deleted via `mark_deleted()`, the backed dataset automatically:

- Excludes deleted rows from `shape`
- Remaps row indices so user-visible indices are contiguous
- Filters obs metadata to match

```python
exp = pyscx.open("experiment.scx")
exp.mark_deleted(doublet_mask)

# Backed mode sees only non-deleted cells
adata = exp.to_anndata(backed=True)
assert adata.X.shape[0] == exp.n_obs - doublet_mask.sum()
```

### Cache configuration

Decoded shards are optionally cached in an LRU cache to speed up repeated
access to the same rows:

```python
# Default: 4-shard LRU cache
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Larger cache for repeated access patterns
adata = pyscx.open("atlas.scx").to_anndata(backed=True, cache_shards=16)

# No cache (minimum memory footprint)
adata = pyscx.open("atlas.scx").to_anndata(backed=True, cache_shards=0)

# Read the setting back — it is fixed per to_anndata() call (there is no
# cache_shards on pyscx.open); X and each layer get their own reader, built
# with the same count
adata.X.cache_shards            # 0
adata.layers["counts"].cache_shards
```

`cache_shards=0` is a true uncached read: every access decodes afresh and
retains nothing, so the streaming footprint is one shard plus the result.
`IndexPlanDataset(cache_shards=0)` is different — it raises on purpose, because
the training loader's prefetcher needs a cache to prefetch into.

While you are inspecting a handle: `adata.X.stored_dtype` is the on-disk value
encoding (`uint16` for counts, `float32` after normalisation was saved), and
`pyscx.open(path).is_integer` / `.max_value` answer "are these counts, and how
large?" from the shard headers and catalog stats without decoding a value.

### Comparison with h5ad backed mode

| | h5ad `backed="r"` | SCX `backed=True` |
|--|---|---|
| File open | ~50 ms | ~5 ms (mmap + catalog parse) |
| Row slice 1K | ~20 ms | ~5 ms (1 shard decode) |
| Idle memory (1M cells) | ~8 MB | ~2 MB |
| Column slice (all rows) | ~2 s | ~800 ms (parallel shard decode) |
| `isinstance(X, CSRDataset)` | ✅ | ✅ (ABC registered) |

## Out-of-core chunk iteration

For workflows that need to process data shard-by-shard without loading the
entire matrix, use `iter_chunks()`:

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# Iterate shard-aligned chunks (default)
for chunk in pyscx.iter_chunks(adata):
    print(f"Chunk: {chunk.n_obs} cells, {chunk.n_vars} genes")
    # Each chunk is a fully materialized AnnData with obs/var/obsm sliced
    sc.pp.normalize_total(chunk, target_sum=1e4)
    sc.pp.log1p(chunk)
    # ... accumulate results ...

# Fixed-size chunks
for chunk in pyscx.iter_chunks(adata, chunk_size=5000):
    # Each chunk has at most 5000 cells
    pass
```

Shard-aligned chunking (default) avoids decoding any shard twice. Each
chunk's `obs`, `var`, and `obsm` are correctly sliced to match the rows
in that chunk.

## See also

- [Lazy preprocessing](lazy-preprocessing.md) — `normalize_total` / `log1p`
  on a backed `X` without materializing it.
- [Loading SCX data into AnnData](loading.md) — the `to_anndata()` options
  that also apply in backed mode.
- [Accelerators overview](accelerators.md) — which `pyscx.accel` ops stream
  over a backed `X`.
