# Common scanpy workflows

> Part of the [SCX + scanpy guide](README.md). New here? The [quick start](quickstart.md) is shorter.

## Clustering and visualization

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

## Accelerated pipeline for large datasets

For datasets with >100K cells, use `pyscx.accel.*` for a fully
out-of-core pipeline — no materialization from file open through
clustering:

```python
import pyscx
import scanpy as sc

# Open in backed mode — X stays on disk
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC works natively in backed mode (no materialization)
# Gene subsets (e.g., mitochondrial) use column-projected streaming
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# HVG selection — streaming variance
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Lazy preprocessing — NO materialization
pyscx.accel.normalize_total(adata, target_sum=1e4)   # → lazy wrapper
pyscx.accel.log1p(adata)                              # → appends to chain

# PCA streams through lazy transforms shard-by-shard
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)

sc.pl.umap(adata, color="leiden")
```

> Peak RSS for this pipeline on census_1m (1M cells × 61K genes) is ~11 GB
> (vs ~22 GB with the traditional materialize-then-preprocess approach — a
> 51% reduction). The lazy preprocessing path streams shard-by-shard through
> transforms without materializing the full matrix. The remaining RSS is
> dominated by kNN graph construction and UMAP embedding, not preprocessing.
> The Python interpreter + library baseline is ~450 MB.

## GPU-accelerated analysis pipeline

When a CUDA GPU is available, use `device="gpu"` for GPU-accelerated
analysis (up to 16× per-op, 3.8× end-to-end on 1M cells). See
[`docs/gpu-setup.md`](../gpu-setup.md) for installation instructions (conda,
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
pyscx.accel.leiden(adata_sub, device="gpu")  # cuGraph on GPU (or device="cpu" for Rust-native)

sc.pl.umap(adata_sub, color="leiden")

# Check which backends were used
print(adata_sub.uns["pca"]["backend"])       # "rapids_singlecell_gpu"
print(adata_sub.uns["neighbors"]["backend"]) # "rapids_singlecell_gpu"
print(adata_sub.uns["umap"]["backend"])      # "rapids_singlecell_gpu"
print(adata_sub.uns["leiden"]["backend"])    # "cugraph"
```

## Differential expression

```python
adata = pyscx.open("experiment.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Using scanpy directly (works fine):
sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon")
sc.pl.rank_genes_groups(adata, n_genes=20)

# Or using pyscx accelerator (3× faster, supports GPU):
pyscx.accel.rank_genes_groups(adata, "cell_type")
sc.pl.rank_genes_groups(adata, n_genes=20)  # scanpy plotting works identically

# Perturbation DE (Perturb-seq experiments):
df = pyscx.accel.pdex_ref(adata, "perturbation", reference="non-targeting")

# Pseudobulk DE with biological replicates:
result = pyscx.accel.pseudobulk_dex(
    adata, groupby=["perturbation", "donor"],
    test_col="perturbation", reference="control",
    backend="nb_glm",  # Rust-native NB-GLM, no pydeseq2 needed
)
```

## Batch integration

Two integrated batch-correction paths, neither of which requires leaving
the pyscx stack:

**Harmony2 (fast, linear).** Clean-room Rust port of Harmony2. Operates
on the PCA embedding — correct once, feed the result into kNN / UMAP /
Leiden as if it were raw PCA. Drop-in replacement for
`scanpy.external.pp.harmony_integrate`.

```python
import pyscx
import scanpy as sc

adata = pyscx.open("multi_batch.scx").to_anndata()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="batch")

pyscx.accel.pca(adata, n_comps=30)
pyscx.accel.harmony_integrate(adata, "batch")  # writes obsm["X_pca_harmony"], keeps "X_pca"

# Use corrected embedding for downstream:
pyscx.accel.neighbors(adata, use_rep="X_pca_harmony")
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)

# QC: batch mixing before / after — both embeddings are available by default
lisi_pre = pyscx.accel.compute_lisi(adata, "batch", basis="X_pca")
lisi = pyscx.accel.compute_lisi(adata, "batch", basis="X_pca_harmony")
print(f"mean LISI batch = {lisi.mean():.2f}  (ideal: ≈ n_batches)")
```

**scVI (deep-learning, nonlinear).** When you need nonlinear
integration or want to learn a latent space for transfer tasks.

```python
import pyscx
import scanpy as sc
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

Harmony runs in seconds-to-minutes on 1M cells (see
[`docs/performance/accel-qc-de-integration.md`](../performance/accel-qc-de-integration.md#harmony2-batch-integration--lisi));
scVI adds a GPU-minutes training phase but captures nonlinear
batch effects Harmony cannot. For most routine integration tasks start
with Harmony; reach for scVI when Harmony underperforms on a
`compute_lisi` gate.

## See also

- [Accelerators overview](accelerators.md) and the per-op accelerator pages.
- [Lazy preprocessing](lazy-preprocessing.md) — the out-of-core pipeline in
  detail.
