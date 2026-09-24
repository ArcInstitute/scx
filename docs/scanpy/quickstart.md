# Quick start: SCX with scanpy

> Part of the [SCX + scanpy guide](README.md).

From an h5ad file to clusters in a few minutes. Every `pyscx` loader returns a
standard `anndata.AnnData`, so the scanpy calls below are unchanged — SCX only
changes how the data gets into memory (or stays out of it). For a CLI-first
end-to-end run with QC and marker ranking, see the repository-wide
[docs/quickstart.md](../quickstart.md).

## Install

Pre-built `pyscx` wheels and the `scx` CLI are on GitHub Releases — see
[docs/quickstart.md § Install](../quickstart.md#install) and
[skills/scx-usage/reference/installation.md](../../skills/scx-usage/reference/installation.md)
for the exact filenames. GPU support (`device="gpu"`) needs extra setup —
see [docs/gpu-setup.md](../gpu-setup.md).

The dependencies a scanpy session uses:

```
pyscx          # SCX Python bindings
scanpy         # scverse analysis toolkit
anndata        # AnnData data structure
scipy          # sparse matrix support
numpy          # array support
pyarrow        # metadata transfer (Arrow IPC)
```

To build `pyscx` from a source checkout instead:

```bash
uv pip install scanpy anndata scipy numpy pyarrow
cd pyscx && maturin develop --release
```

## Convert your data

```python
import pyscx

# Streams from disk shard by shard — safe for files larger than RAM.
pyscx.from_h5ad("experiment.h5ad", "experiment.scx")
```

Or from the shell: `scx convert experiment.h5ad experiment.scx`. An AnnData
already in memory converts with `pyscx.from_anndata(adata, "experiment.scx")`.
10x HDF5, Cell Ranger MTX, `adata.raw`, and the export direction are covered in
[Converting existing data to SCX](conversion.md).

## Pick a loader

| Your situation | Loader | Page |
|----------------|--------|------|
| The matrix fits comfortably in RAM (roughly < 500K cells) | `pyscx.open(path).to_anndata()` | [Quick start: in-memory](#quick-start-in-memory) below |
| Atlas-scale data on modest RAM | `to_anndata(backed=True)` + `pyscx.accel.*` | [Quick start: backed mode](#quick-start-backed-mode-out-of-core) below |
| You only need a subset of cells or genes | `pyscx.open(path).query()...collect()` | [Querying subsets before loading](loading.md#querying-subsets-before-loading) |
| Training a model | `pyscx.TrainingDataset` | [ML training data loading](ml-training.md#ml-training-data-loading) |

The full trade-off table is in
[Choosing the right approach](choosing-an-approach.md#choosing-the-right-approach).

## Quick start: in-memory

The simplest approach. `to_anndata()` loads the entire expression matrix into
memory as a scipy CSR matrix (via **zero-copy** transfer from Rust). This is
ideal for datasets that fit comfortably in RAM, since the full standard scanpy
API works without modification.

```python
import pyscx
import scanpy as sc

# Open an SCX file and load everything into memory
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

> [!IMPORTANT]
> `to_anndata()` **materializes the full matrix** into a standard scipy CSR.
> The resulting AnnData is identical in memory whether loaded from SCX or
> h5ad — SCX's advantage is a smaller file on disk, not a smaller in-memory
> object. For a 1M-cell dataset with 30K genes at 5% density, the scipy
> CSR requires ~12 GB (see [memory formula](loading.md#default-behavior-no-extra-params)
> below). If this exceeds your available memory, use
> [backed mode](backed-mode.md#backed-mode-lazy-loading) or the
> [query pipeline](loading.md#querying-subsets-before-loading) instead.

## Quick start: backed mode (out-of-core)

For large datasets, backed mode keeps data on disk and preprocesses
lazily — peak memory is one shard (~128 MB) rather than the full matrix:

```python
import pyscx
import scanpy as sc

# Open in backed mode — X stays on disk
adata = pyscx.open("atlas.scx").to_anndata(backed=True)

# QC and filtering — fully streaming, no materialization
pyscx.accel.filter_cells(adata, min_genes=200)
pyscx.accel.filter_genes(adata, min_cells=3)

# Preprocessing — lazy, data stays on disk
pyscx.accel.normalize_total(adata, target_sum=1e4)
pyscx.accel.log1p(adata)

# Analysis — PCA streams through lazy transforms
sc.pp.highly_variable_genes(adata)
pyscx.accel.pca(adata, n_comps=50)
pyscx.accel.neighbors(adata, n_neighbors=15)
pyscx.accel.umap(adata)
pyscx.accel.leiden(adata)                    # Rust-native, 40× faster than leidenalg
sc.pl.umap(adata, color="leiden")
```

> [!NOTE]
> In backed mode, use `pyscx.accel.*` for preprocessing functions
> (`normalize_total`, `log1p`, `filter_cells`, `filter_genes`, `pca`,
> `neighbors`, `umap`, `leiden`). These are designed for out-of-core data and avoid
> materializing the full matrix. Standard `sc.pp.*` functions work for
> operations that don't modify X (e.g., `highly_variable_genes`), but
> `sc.pp.normalize_total()` and `sc.pp.log1p()` will force full
> materialization. See the [compatibility table](backed-mode.md#scanpy-operations-in-backed-mode)
> for details.


## Check which backend ran

Every accelerator records the route it actually took, so a silent GPU → CPU or
CSC → CSR fallback is visible:

```python
pyscx.accel.pca(adata, n_comps=50)
print(adata.uns["scx_accel"]["pca"]["route"])
```

See [Checking which backend was used](accel-gpu.md#checking-which-backend-was-used) and
[docs/api/python-accel.md § Accelerator route metadata](../api/python-accel.md#accelerator-route-metadata).

## Save your results

```python
# Write the analysed AnnData back to SCX ...
pyscx.from_anndata(adata, "experiment_analysed.scx")

# ... or export an SCX file to h5ad for tools that need it (streams by default).
pyscx.to_h5ad("experiment.scx", "experiment.h5ad")
```

## Next steps

- [Loading SCX data into AnnData](loading.md) — load only the genes, cells,
  layers, or modality you need, and what each choice costs in memory.
- [Backed mode](backed-mode.md) and [lazy preprocessing](lazy-preprocessing.md)
  — the out-of-core path in depth.
- [Accelerators overview](accelerators.md) — the `pyscx.accel` compatibility
  matrix and GPU routing.
- [Common scanpy workflows](workflows.md) — longer recipes, including
  differential expression and batch integration.
- [docs/migrating-from-h5ad.md](../migrating-from-h5ad.md) — scanpy-divergence
  gotchas when porting an existing workflow.
