# Using SCX with scanpy

SCX integrates directly with [scanpy](https://scanpy.readthedocs.io/) and the
[scverse](https://scverse.org/) ecosystem through `pyscx`. Every `pyscx` method
that returns data produces a standard `anndata.AnnData` object — so any scanpy
function works out of the box with zero glue code.

## Quick start

```python
import pyscx
import scanpy as sc

# Open an SCX file and convert to AnnData
adata = pyscx.open("experiment.scx").to_anndata()

# Standard scanpy pipeline — nothing changes
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.umap(adata, color="leiden")
```

## Converting existing data to SCX

### From AnnData / h5ad

```python
import scanpy as sc
import pyscx

# From an AnnData object in memory
adata = sc.read_h5ad("dataset.h5ad")
pyscx.from_anndata(adata, "dataset.scx")

# Optional: control codec and shard size
pyscx.from_anndata(adata, "dataset.scx", codec="auto", shard_size=8192)
```

The `codec` parameter accepts `"auto"` (default — selects best codec per shard),
`"scx1"` (domain-specific integer codec), `"zstd"`, or `"none"`.

### From 10x HDF5

```python
pyscx.from_10x("filtered_feature_bc_matrix.h5", "dataset.scx")
```

### From Cell Ranger MTX directory

```python
pyscx.from_mtx("/path/to/filtered_feature_bc_matrix", "dataset.scx")
```

### Exporting back to MTX

```python
pyscx.to_mtx("dataset.scx", "/path/to/output_dir")
```

## Understanding `to_anndata()`

Calling `to_anndata()` performs a **full read** of the SCX file. Here is
exactly what happens:

1. **Reads and decompresses every CSR shard** from disk. Each shard is decoded
   using its per-shard codec (Scx1, Zstd, or raw), then all shards are
   assembled into a single contiguous CSR matrix.
2. **Applies deletion vectors.** If any cells have been logically deleted via
   `mark_deleted()`, those rows are excluded from both the matrix and the obs
   metadata. You always get a clean view — no manual filtering needed.
3. **Transfers the CSR matrix to scipy via zero-copy.** The Rust `Vec`s for
   indptr (`i64`), indices (`i32`), and data (`f32`) are moved directly into
   numpy arrays with no memory copy. These dtypes match what scipy expects, so
   `scipy.sparse.csr_matrix` wraps them without conversion.
4. **Reads obs/var metadata** as Arrow RecordBatches, converts to pandas
   DataFrames via `pyarrow.to_pandas()`.
5. **Reads obsm, uns, and layers** if present in the file.

The returned `anndata.AnnData` is fully populated:

| Slot | Source | Type |
|------|--------|------|
| `X` | CSR shards | `scipy.sparse.csr_matrix` (zero-copy) |
| `obs` | Obs metadata section | pandas DataFrame |
| `var` | Var metadata section | pandas DataFrame |
| `obsm` | Obsm sections | dict of numpy arrays (e.g. `X_pca`, `X_umap`) |
| `uns` | Uns section | dict (JSON round-tripped) |
| `layers` | Layer shards | dict of `scipy.sparse.csr_matrix` |

**Memory implications:** Because `to_anndata()` loads the entire decompressed
matrix into memory, peak memory equals the size of the sparse CSR
representation (not the file size on disk). For a 1M-cell dataset this is
typically ~264 MB — far less than h5ad's ~11.6 GB — but it is still a full
materialization. If you only need a subset of cells or genes, use the
[query pipeline](#querying-subsets-before-loading) instead to avoid loading
data you don't need.

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
- **obs, var, obsm, uns** are loaded eagerly (same as non-backed — these are
  small relative to X)
- The dataset is **read-only** (matching AnnData's `backed="r"` semantics)

Each access to `adata.X[rows, cols]` decompresses only the CSR shards that
overlap the requested rows. For a 1M-cell file with 64 shards, slicing 1K
cells touches ≤ 1 shard.

### Indexing patterns

| Access | Behavior |
|--------|----------|
| `X[100:200]` | Row slice → decodes 1 shard |
| `X[[0, 5, 10]]` | Fancy index → decodes needed shards only |
| `X[mask]` | Boolean mask → decodes matching shards |
| `X[100:200, :500]` | Row slice + column filter → 1 shard + post-filter |
| `X[:, hvg_idx]` | Column-only → must decode all shards (CSR is row-major) |
| `X[0, 5]` | Scalar → returns `float` |

### scanpy operations in backed mode

| Operation | Works? | Notes |
|-----------|--------|-------|
| `sc.pp.filter_cells()` | ✅ | Row-wise sum, read-only |
| `sc.pp.filter_genes()` | ✅ | Column-wise sum, read-only |
| `sc.pp.highly_variable_genes()` | ✅ | Column statistics |
| `adata[mask].copy()` | ✅ | Subset → materialize → preprocess |
| `sc.pp.pca()` | ✅ | Forces materialization of HVG columns |
| `sc.pp.normalize_total()` | ❌ | Modifies X in-place — materialize first |
| `sc.pp.log1p()` | ❌ | Modifies X in-place — materialize first |

**In-place operations** (normalize, log1p, scale) fail on read-only backed
data. The typical workflow is: subset → copy → preprocess:

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
```

### Comparison with h5ad backed mode

| | h5ad `backed="r"` | SCX `backed=True` |
|--|---|---|
| File open | ~50 ms | ~5 ms (mmap + catalog parse) |
| Row slice 1K | ~20 ms | ~5 ms (1 shard decode) |
| Idle memory (1M cells) | ~8 MB | ~2 MB |
| Column slice (all rows) | ~2 s | ~800 ms (parallel shard decode) |
| `isinstance(X, CSRDataset)` | ✅ | ✅ (ABC registered) |

## Querying subsets before loading

For large datasets, you don't need to load everything into memory. The SCX
query engine filters at the shard level, skipping data that can't match:

```python
# Load only T cells from lung tissue
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'T cell' and tissue == 'lung'")
    .collect())

adata = result.to_anndata()
print(f"Loaded {adata.n_obs} cells, skipped {result.skipped_shards}/{result.total_shards} shards")

# Continue with scanpy as usual
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.pca(adata)
```

### Query pipeline options

The query pipeline supports chaining multiple operations:

```python
result = (pyscx.open("atlas.scx")
    .query()
    .filter_obs("cell_type == 'B cell'")       # filter cells
    .filter_var("highly_variable == True")      # filter genes
    .select_genes([0, 1, 2, 100, 200])         # or select by index
    .with_normalize(1e4)                        # normalize in Rust (faster)
    .with_log1p()                               # log1p in Rust
    .limit(5000)                                # cap returned cells
    .collect())

adata = result.to_anndata()
```

When you use `with_normalize()` and `with_log1p()` in the query pipeline,
normalization runs in compiled Rust — significantly faster than the Python
equivalent on large datasets. The resulting AnnData is ready for downstream
analysis (PCA, clustering, etc.) without calling `sc.pp.normalize_total()`
or `sc.pp.log1p()` again.

## Common scanpy workflows

### Clustering and visualization

```python
import pyscx
import scanpy as sc

adata = pyscx.open("experiment.scx").to_anndata()

# QC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
adata = adata[adata.obs["pct_counts_mt"] < 20].copy()

# Normalize and select HVGs
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var["highly_variable"]].copy()

# Dimensionality reduction and clustering
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Visualization
sc.pl.umap(adata, color=["leiden", "cell_type"])
```

### Differential expression

```python
adata = pyscx.open("experiment.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=20)
```

### Batch integration with scVI

```python
import pyscx
import scvi

adata = pyscx.open("multi_batch.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, batch_key="batch")

scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
model = scvi.model.SCVI(adata)
model.train()
adata.obsm["X_scVI"] = model.get_latent_representation()
```

## File operations with scanpy

### Removing doublets and saving back

```python
import pyscx
import scanpy as sc
import scrublet

adata = pyscx.open("experiment.scx").to_anndata()

# Run doublet detection
scrub = scrublet.Scrublet(adata.X)
scores, predicted = scrub.scrub_doublets()

# Mark doublets as deleted (instant, no data rewrite)
exp = pyscx.open("experiment.scx")
exp.mark_deleted(predicted)  # boolean numpy array

# Or save the filtered result as a new file
adata_clean = adata[~predicted].copy()
pyscx.from_anndata(adata_clean, "experiment_clean.scx")
```

### Appending new batches

```python
# Append cells from another SCX file
pyscx.append("atlas.scx", "new_batch.scx")

# Append cells from an AnnData object
new_adata = sc.read_h5ad("new_batch.h5ad")
pyscx.append_from_anndata("atlas.scx", new_adata)
```

### Merging datasets

```python
# Merge multiple SCX files (must have same n_vars)
pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")

# Then analyze the merged atlas
adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.combat(adata, key="batch")  # batch correction
```

## GPU-accelerated training with scVI / scANVI

For large-scale model training, SCX provides a high-performance data loader
that bypasses Python I/O entirely:

```python
import pyscx
import torch

dataset = pyscx.TrainingDataset(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_array,   # decode only HVGs → less data
    normalize=True,
    log1p=True,
)

for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
    # model.forward(), loss.backward(), ...
```

The training loader uses a triple-buffered Rust pipeline (I/O → decode → GPU)
with zero Python on the hot path.

## File inspection

```python
exp = pyscx.open("experiment.scx")
print(exp)           # PyExperiment(n_obs=10000, n_vars=33694, ...)
print(exp.n_obs)     # 10000
print(exp.n_vars)    # 33694
print(exp.nnz)       # 5234891
print(exp.codec_id)  # 1 (Scx1)
print(exp.layer_names)  # ["raw_counts", "spliced"]

# Validate checksums
results = exp.validate()
for section, passed in results:
    print(f"  {section}: {'✓' if passed else '✗'}")
```

## Dependencies

```
pyscx          # SCX Python bindings
scanpy         # scverse analysis toolkit
anndata        # AnnData data structure
scipy          # sparse matrix support
numpy          # array support
pyarrow        # metadata transfer (Arrow IPC)
```

Install with:

```bash
uv pip install scanpy anndata scipy numpy pyarrow
cd pyscx && maturin develop --release
```
