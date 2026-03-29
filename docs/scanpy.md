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

### Full signature

```python
exp.to_anndata(
    backed=False,         # True for lazy loading (X stays on disk)
    cache_shards=4,       # LRU cache size for backed mode
    var_names=None,       # List of gene names to project (column subset)
    obs_filter=None,      # Predicate string to filter cells (e.g. "cell_type == 'T cell'")
    layers=None,          # List of layer names to load (default: all)
)
```

### Default behavior (no extra params)

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
materialization. If you only need a subset of cells or genes, use
[selective loading](#selective-loading) or the
[query pipeline](#querying-subsets-before-loading) instead.

### Selective loading

All selective loading parameters work in both non-backed and backed modes.

#### Gene projection (`var_names`)

Load only specific genes, reducing memory and computation:

```python
# Load only marker genes
adata = pyscx.open("atlas.scx").to_anndata(
    var_names=["CD3E", "CD4", "CD8A", "MS4A1", "NCAM1"]
)
print(adata.n_vars)  # 5
```

In non-backed mode, this applies column projection to the materialized CSR.
In backed mode, projection is applied lazily — full rows are decoded from
disk, but only the requested columns are retained in the returned CSR.

#### Cell filtering (`obs_filter`)

Filter cells using a predicate string. In non-backed mode, this leverages
the query engine with predicate pushdown (shard skipping). In backed mode,
it evaluates the predicate on the obs DataFrame:

```python
# Load only T cells from lung tissue
adata = pyscx.open("atlas.scx").to_anndata(
    obs_filter="cell_type == 'T cell' and tissue == 'lung'"
)
```

#### Layer selection (`layers`)

Load only specific layers instead of all:

```python
# Load raw counts layer only
adata = pyscx.open("atlas.scx").to_anndata(layers=["raw_counts"])

# Load no layers at all (X only)
adata = pyscx.open("atlas.scx").to_anndata(layers=[])
```

#### Combining parameters

All parameters can be combined:

```python
adata = pyscx.open("atlas.scx").to_anndata(
    backed=True,
    var_names=["CD3E", "CD4", "CD8A"],
    obs_filter="cell_type == 'T cell'",
    layers=["raw_counts"],
)

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
| `sc.pp.filter_cells()` | ✅ | Native Rust row sums/NNZ |
| `sc.pp.filter_genes()` | ✅ | Native Rust column sums/NNZ |
| `sc.pp.calculate_qc_metrics()` | ✅ | `(X > 0).sum()` short-circuits to `getnnz()` |
| `sc.pp.highly_variable_genes()` | ✅ | Native Rust column variance |
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

## Rust-native accelerators

SCX includes optional Rust-native implementations of PCA, kNN graph
construction, and UMAP embedding via `pyscx.accel`. These accelerators are
2–10× faster than their scanpy equivalents at scale (>100K cells) while
writing results to the same AnnData slots — so downstream scanpy functions
(leiden, plotting, DE) work identically.

All accelerators support a `device` parameter for GPU acceleration:
- `device="auto"` (default) — use GPU if available, fall back to CPU
- `device="cpu"` — force CPU
- `device="gpu"` — force GPU (raises error if unavailable)
- `device="gpu:1"` — select a specific GPU on multi-GPU systems

### PCA (`pyscx.accel.pca`)

Randomized SVD with streaming shard-by-shard SpMM. Can run directly on
backed mode without materializing the full matrix.

```python
import pyscx

adata = pyscx.open("atlas.scx").to_anndata(backed=True)
pyscx.accel.pca(adata, n_comps=50)

# Results written to standard scanpy slots:
#   adata.obsm["X_pca"]           — (n_obs × n_comps) float32
#   adata.varm["PCs"]             — (n_vars × n_comps) float32
#   adata.uns["pca"]["variance"]  — explained variance per PC
#   adata.uns["pca"]["variance_ratio"]
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_comps` | 50 | Number of principal components |
| `zero_center` | True | Mean-center data (True = standard PCA, False = TruncatedSVD) |
| `random_state` | 0 | Random seed |
| `n_oversamples` | 10 | Extra dimensions for accuracy |
| `n_power_iterations` | 2 | Power iterations for spectral accuracy |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

**Key advantage:** In backed mode, PCA streams SpMM shard-by-shard. On GPU,
the pipeline uses cuSPARSE SpMM + cuSOLVER QR. Benchmarked at ~1× on 1M cells
(GPU overhead offsets SpMM gains at this scale; larger datasets benefit more).
Peak memory is one shard plus working matrices.

### kNN graph (`pyscx.accel.neighbors`)

Approximate nearest neighbors via HNSW (Hierarchical Navigable Small
World), followed by UMAP-style fuzzy set connectivities.

```python
pyscx.accel.neighbors(adata, n_neighbors=15)

# Results written to standard scanpy slots:
#   adata.obsp["distances"]        — sparse CSR (n_obs × n_obs)
#   adata.obsp["connectivities"]   — sparse CSR (n_obs × n_obs)
#   adata.uns["neighbors"]         — metadata dict
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | 15 | Number of nearest neighbors |
| `use_rep` | `"X_pca"` | Key in `adata.obsm` to use as input |
| `random_state` | 0 | Random seed |
| `ef_construction` | 200 | HNSW build parameter (higher = more accurate) |
| `ef_search` | 200 | HNSW search parameter (higher = more accurate) |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

On CPU, uses HNSW (instant-distance) with Euclidean distance.
On GPU, uses NVIDIA CAGRA (cuVS) — benchmarked at 4.4× on 100K cells and 9.4× on 1M cells.

### UMAP (`pyscx.accel.umap`)

SGD-based UMAP embedding with spectral initialization and negative
sampling. Takes the kNN connectivity graph as input.

```python
pyscx.accel.umap(adata)

# Result written to:
#   adata.obsm["X_umap"]  — (n_obs × 2) float32
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 2 | Output dimensions |
| `n_epochs` | 200 | SGD epochs (more = better quality, slower) |
| `min_dist` | 0.1 | Minimum distance in embedding |
| `spread` | 1.0 | Spread of embedded points |
| `negative_sample_rate` | 5 | Negative samples per positive edge |
| `learning_rate` | 1.0 | Initial learning rate |
| `random_state` | 0 | Random seed |
| `device` | `"auto"` | Device selection: `"auto"`, `"cpu"`, `"gpu"`, `"gpu:N"` |

On GPU, uses a native CUDA SGD kernel (edge-parallel with `atomicAdd`).
Falls back to cuML UMAP if available for maximum performance.

### Differential Expression (`pyscx.accel.rank_genes_groups`)

Parallel Wilcoxon rank-sum test with rayon. Compares each cluster against
the rest (or a specific reference group) and applies Benjamini–Hochberg
correction. Results are written to the same `adata.uns["rank_genes_groups"]`
format as scanpy, so `sc.pl.rank_genes_groups()` and
`sc.get.rank_genes_groups_df()` work identically.

```python
pyscx.accel.rank_genes_groups(adata, "leiden")

# Results written to:
#   adata.uns["rank_genes_groups"]["names"]           — structured array
#   adata.uns["rank_genes_groups"]["scores"]           — z-scores
#   adata.uns["rank_genes_groups"]["pvals"]             — raw p-values
#   adata.uns["rank_genes_groups"]["pvals_adj"]         — BH-adjusted
#   adata.uns["rank_genes_groups"]["logfoldchanges"]    — log2 FC

# Downstream scanpy works identically:
sc.pl.rank_genes_groups(adata, n_genes=20)
df = sc.get.rank_genes_groups_df(adata, group="0")
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | Column in `adata.obs` to group cells by |
| `reference` | `"rest"` | Compare against a specific group or `"rest"` (1-vs-rest) |
| `n_genes` | all | Number of top genes to report per group |
| `method` | `"wilcoxon"` | Statistical method (currently only `"wilcoxon"`) |

### Pseudobulk Differential Expression (`pyscx.accel.pseudobulk_dex`)

Streaming pseudobulk aggregation in Rust + negative binomial GLM testing via
[pydeseq2](https://pydeseq2.readthedocs.io/). Designed for perturbation
sequencing (Perturb-seq) experiments with biological replicates.

The aggregation phase streams shards from `BackedCsrReader` without
materializing the full matrix — peak memory is one shard plus the pseudobulk
count matrix (n_groups × n_vars).

```python
import pyscx

adata = pyscx.open("perturb_seq.scx").to_anndata(backed=True)

# Run pseudobulk DE: drug vs control, grouped by (perturbation, donor)
result = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference="control",
)

# result is a pandas DataFrame:
#   gene | baseMean | log2FoldChange | lfcSE | stat | pvalue | padj | target | reference
print(result.sort_values("padj").head(20))
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `groupby` | (required) | List of obs columns to group cells by |
| `test_col` | (required) | Column in `groupby` containing the condition variable |
| `reference` | (required) | Reference level in `test_col` (e.g., `"control"`) |
| `design` | `"~ test_col"` | DESeq2 design formula (auto-generated if not specified) |
| `aggr_method` | `"sum"` | Aggregation method: `"sum"` or `"mean"` |
| `min_cells_per_group` | 10 | Groups with fewer cells are excluded |

### Stratified Differential Expression

Both `rank_genes_groups()` and `pseudobulk_dex()` support automatic
stratification via the `stratify_by` parameter. DE is run independently
within each stratum and the results are concatenated into a single
DataFrame with stratum columns appended.

```python
import pyscx

adata = pyscx.open("perturb_seq.scx").to_anndata()

# Single-cell DE stratified by cell type:
result = pyscx.accel.rank_genes_groups(
    adata, "perturbation",
    stratify_by=["cell_type"],
    min_cells_per_stratum=50,
)
# Returns a DataFrame with columns:
#   gene | scores | pvals | pvals_adj | logfoldchanges | group | cell_type

# Multi-column stratification (composite strata):
result = pyscx.accel.rank_genes_groups(
    adata, "perturbation",
    stratify_by=["cell_type", "tissue"],
    min_cells_per_stratum=30,
)
# Returns DataFrame with both cell_type and tissue columns

# Pseudobulk DE stratified by cell type:
result = pyscx.accel.pseudobulk_dex(
    adata,
    groupby=["perturbation", "donor"],
    test_col="perturbation",
    reference="control",
    stratify_by=["cell_type"],
    min_cells_per_stratum=50,
)
# Returns DataFrame with cell_type column added
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stratify_by` | `None` | Column(s) in `adata.obs` to stratify by. Single string or list of strings. |
| `min_cells_per_stratum` | 50 | Strata with fewer cells are skipped (with warning). |

Strata with insufficient cells are skipped with a `UserWarning`. If all
strata are filtered, a `ValueError` is raised. `stratify_by` columns must
not collide with `groupby` or `test_col`.

> [!NOTE]
> When `stratify_by` is provided, `rank_genes_groups()` returns a pandas
> DataFrame instead of writing to `adata.uns`. Without `stratify_by`, it
> writes to `adata.uns["rank_genes_groups"]` as usual and returns `None`.

> [!NOTE]
> `pydeseq2` is an **optional** runtime dependency. Install with
> `pip install pydeseq2` before calling `pseudobulk_dex()`.


The accelerators write to the same AnnData slots as scanpy, so they are
fully interchangeable. You can mix and match:

```python
import pyscx
import scanpy as sc

adata = pyscx.open("atlas.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
adata = adata[:, adata.var["highly_variable"]].copy()

# Use SCX accelerators for compute-heavy steps
pyscx.accel.pca(adata, n_comps=50)       # faster than sc.pp.pca
pyscx.accel.neighbors(adata)              # faster than sc.pp.neighbors
pyscx.accel.umap(adata)                   # faster than sc.tl.umap

# Downstream scanpy works identically
sc.tl.leiden(adata)                       # uses adata.obsp["connectivities"]
sc.tl.rank_genes_groups(adata, "leiden")  # standard DE
sc.pl.umap(adata, color="leiden")         # uses adata.obsm["X_umap"]
```

Or use scanpy for everything — no changes needed:

```python
sc.pp.pca(adata)        # works fine with SCX data
sc.pp.neighbors(adata)  # uses pynndescent
sc.tl.umap(adata)       # uses umap-learn
```

The choice is purely about performance. At <50K cells, the difference is
negligible. At >100K cells, the Rust accelerators provide meaningful
speedups. At >1M cells, GPU acceleration (`device="gpu"`) transforms
interactive exploration from "go get coffee" to "instant."

### GPU helper functions

```python
# Query GPU availability and memory
info = pyscx.accel.gpu_info()
# {'device': 'NVIDIA A100-SXM4-80GB', 'total_vram_gb': 80.0, 'free_vram_gb': 72.3}

# Estimate GPU memory requirements
est = pyscx.accel.estimate_gpu_memory(adata, operation="pca", n_comps=50)
# {'required_gb': 2.1, 'fits_in_vram': True}
```

### GPU vs CPU numerical differences

GPU and CPU accelerators may produce slightly different results due to:

| Factor | Impact | When it matters |
|--------|--------|-----------------|
| **PCA precision** | GPU uses f32 throughout; CPU uses f64 | Cosine similarity per PC > 0.99 — no biological impact |
| **kNN algorithm** | GPU uses CAGRA (graph-based ANN); CPU uses HNSW | Both are approximate; recall@k > 0.95 |
| **UMAP non-determinism** | GPU uses `atomicAdd` (race conditions are intentional) | Embedding coordinates differ; cluster structure preserved |
| **Leiden** | cuGraph vs leidenalg may produce different partitions | ARI > 0.90; biological conclusions equivalent |

For reproducibility notes and tolerance thresholds, see
[Phase4-GPU.md §7](../Phase4-GPU.md).

### Checking which backend was used

All accelerators record the backend in `adata.uns`:

```python
pyscx.accel.pca(adata, device="gpu")
print(adata.uns["pca"]["backend"])          # "scx-gpu-cusparse"
print(adata.uns["pca"]["device"])           # "NVIDIA A100-SXM4-80GB"
print(adata.uns["pca"]["gpu_time_ms"])      # 1234.5

pyscx.accel.neighbors(adata, device="gpu")
print(adata.uns["neighbors"]["backend"])    # "scx-gpu-cagra"

pyscx.accel.umap(adata, device="gpu")
print(adata.uns["umap"]["backend"])         # "scx-gpu-cuda" or "cuml"
```

If cuML or cuGraph are importable at runtime, they are used as optimized
backends for UMAP and Leiden respectively. Otherwise, SCX's native CUDA
kernels (UMAP) or CPU fallbacks (Leiden via leidenalg) are used.

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

# Dimensionality reduction and clustering (standard scanpy)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)

# Visualization
sc.pl.umap(adata, color=["leiden", "cell_type"])
```

### Accelerated pipeline for large datasets

For datasets with >100K cells, use the Rust accelerators for the
compute-heavy steps:

```python
import pyscx
import scanpy as sc

# Open in backed mode for memory-efficient QC
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC works natively in backed mode (no materialization)
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# HVG selection also works natively (streaming variance)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Subset to HVGs and materialize for preprocessing
adata_sub = adata[:, adata.var["highly_variable"]].copy()
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# Use Rust accelerators for speed
pyscx.accel.pca(adata_sub, n_comps=50)
pyscx.accel.neighbors(adata_sub, n_neighbors=15)
pyscx.accel.umap(adata_sub)
sc.tl.leiden(adata_sub)

sc.pl.umap(adata_sub, color="leiden")
```

### GPU-accelerated analysis pipeline

When a CUDA GPU is available, use `device="gpu"` for GPU-accelerated
analysis (up to 16× per-op, 3.8× end-to-end on 1M cells). See
[`docs/gpu-setup.md`](gpu-setup.md) for installation instructions (conda,
system CUDA, or container).

```python
import pyscx
import scanpy as sc

# Open in backed mode for memory-efficient QC
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC works natively in backed mode
sc.pp.calculate_qc_metrics(adata, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Subset to HVGs and materialize
adata_sub = adata[:, adata.var["highly_variable"]].copy()
sc.pp.normalize_total(adata_sub, target_sum=1e4)
sc.pp.log1p(adata_sub)

# GPU-accelerated pipeline (falls back to CPU if no GPU)
pyscx.accel.pca(adata_sub, n_comps=50, device="gpu")
pyscx.accel.neighbors(adata_sub, n_neighbors=15, device="gpu")
pyscx.accel.umap(adata_sub, device="gpu")
sc.tl.leiden(adata_sub)  # CPU Leiden; use device="gpu" for cuGraph

sc.pl.umap(adata_sub, color="leiden")

# Check which backends were used
print(adata_sub.uns["pca"]["backend"])       # "scx-gpu-cusparse"
print(adata_sub.uns["neighbors"]["backend"]) # "scx-gpu-cagra"
print(adata_sub.uns["umap"]["backend"])      # "scx-gpu-cuda"
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
