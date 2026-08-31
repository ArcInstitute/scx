# Quickstart

> **Status:** pre-1.0 (v0.7.x). Install from
> [GitHub Releases](https://github.com/ArcInstitute/scx/releases); PyPI + conda
> planned at public release.

A 5-minute end-to-end pipeline: convert an h5ad file to SCX, run QC, normalize,
find HVGs, embed, cluster, and rank markers. Uses the [`pyscx.accel`](scanpy.md#rust-native-accelerators)
Rust-native pipeline.

For the design and architecture, see [docs/architecture.md](architecture.md).
For the full scanpy integration story, see [docs/scanpy.md](scanpy.md).
Coming from h5ad? See [docs/migrating-from-h5ad.md](migrating-from-h5ad.md) for
the round-trip fidelity table and the scanpy-divergence gotchas.

## Which loader do I use?

```
                         Is your dataset small enough
                         to fit in memory (~500K cells)?
                                    │
                        ┌───yes─────┴──────no───┐
                        ▼                       ▼
                  pyscx.read(path)        Do you need
              (or .to_anndata();          the full dataset?
               simplest, full                   │
               scanpy compat)      ┌───yes──────┴──────no───┐
                                   ▼                        ▼
                          to_anndata(backed=True)     .query().filter_obs()
                          + pyscx.accel.*             .collect()  (subset,
                          (out-of-core)               then in-memory)
```

Training a model? Use the `TrainingDataset` loader instead — see
[docs/training.md](training.md). Full trade-off table:
[docs/scanpy.md § Choosing the right approach](scanpy.md#choosing-the-right-approach).

## Install

pyscx and the `scx` CLI are distributed as pre-built artifacts on
[GitHub Releases](https://github.com/ArcInstitute/scx/releases) (not yet on
PyPI / crates.io). Set `VERSION` to the latest release tag.

```bash
# Pick the latest release version (see the Releases page above):
VERSION=0.7.1

# Python bindings — download the wheel for your architecture from GitHub
# Releases (pyscx-v* tags). One abi3 wheel per arch (Linux x86_64 and aarch64)
# installs on every CPython >= 3.11. See
# skills/scx-usage/reference/installation.md for resolving the exact filename.
uv pip install ./pyscx-${VERSION}-cp311-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# CLI — download the pre-built binary (Linux x86_64 / arm64):
TARGET=x86_64-unknown-linux-gnu   # or: aarch64-unknown-linux-gnu
gh release download "scx-cli-v${VERSION}" -R ArcInstitute/scx \
  -p "scx-cli-${VERSION}-${TARGET}.tar.gz"
tar xzf "scx-cli-${VERSION}-${TARGET}.tar.gz"
install -m 0755 "scx-cli-${VERSION}-${TARGET}/scx" ~/.local/bin/scx
```

To build the CLI from source instead (macOS, Windows, custom features), see the
[README install section](../README.md#rust-cli).

## Get a dataset

This walkthrough uses the standard scanpy PBMC fixture — no external download
needed:

```python
import scanpy as sc
sc.datasets.pbmc3k().write("pbmc3k.h5ad")
```

## Convert an h5ad to SCX

```bash
scx convert pbmc3k.h5ad pbmc3k.scx --stream --index-preset cellxgene
scx info pbmc3k.scx
```

`--stream` keeps peak RSS bounded by one shard at a time. `--index-preset cellxgene`
materialises predicate indexes at write time so `pyscx.open(...).query()` can push
filters down on the resulting file. (PBMC3k is small enough that `--stream` is not
required; it is shown here because it is the recommended default at scale.)

## Analyze in Python

```python
import pyscx
from pyscx import accel

exp = pyscx.open("pbmc3k.scx")
adata = exp.to_anndata()           # CSR float32, exactly what scanpy expects
# Or the one-liner, mirroring sc.read_h5ad:
#   adata = pyscx.read("pbmc3k.scx")

# QC. MT genes must be tagged explicitly (same contract as scanpy) — without
# `qc_vars=["mt"]` the `pct_counts_mt` column is not produced. `pyscx.accel`
# emits a UserWarning when MT-prefixed symbols are present but `qc_vars=None`.
adata.var["mt"] = adata.var_names.str.startswith("MT-")
accel.calculate_qc_metrics(adata, qc_vars=["mt"])
adata = adata[
    (adata.obs["n_genes_by_counts"] >= 200)
    & (adata.obs["pct_counts_mt"] < 20)
].copy()

# Stash raw counts so we can run HVG on integer-valued X after log-normalizing.
adata.layers["counts"] = adata.X.copy()
accel.normalize_total(adata, target_sum=1e4)
accel.log1p(adata)

# HVG with seurat_v3 reads from the raw-counts layer; it would warn on the
# normalised X otherwise.
accel.highly_variable_genes(
    adata, n_top_genes=2000, flavor="seurat_v3", layer="counts"
)
adata = adata[:, adata.var["highly_variable"]].copy()

accel.pca(adata, n_comps=50)
accel.neighbors(adata, n_neighbors=15, use_rep="X_pca")
accel.umap(adata)
accel.leiden(adata, resolution=1.0, device="cpu")   # pin CPU for label stability
accel.rank_genes_groups(adata, groupby="leiden")

# Persist processed results. `pyscx.to_h5ad(exp, ...)` re-exports the
# unmodified on-disk SCX file — to save the analysis you just ran, write
# `adata` directly. Use `pyscx.from_anndata(adata, "out.scx")` instead to
# round-trip the processed AnnData back to SCX.
adata.write_h5ad("pbmc3k_processed.h5ad")
```

## Train a model

SCX includes a high-performance training data loader that replaces PyTorch
`DataLoader` workers with a triple-buffered Rust pipeline. No `num_workers`
needed — I/O, decompression, and normalization all happen in compiled Rust.

```python
import numpy as np
import pyscx
import torch

hvg_indices = np.where(adata.var["highly_variable"])[0].astype(np.uint32)

dataset = pyscx.TrainingDataset(
    "pbmc3k.scx",
    batch_size=1024,
    hvg_indices=hvg_indices,
    normalize=True,
    log1p=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
    cell_indices = batch["cell_indices"]
    # model.forward(x), loss.backward(), optimizer.step()

dataset.close()
```

For perturbation training with `(perturbed, control)` cell pairs, use
`pyscx.IndexPlanDataset`. For multimodal data (CITE-seq, Multiome), use
`pyscx.MultimodalTrainingDataset`. See the full
[ML Training Guide](training.md) for end-to-end examples, PyTorch Lightning
integration, and migration from h5ad-based training loops.

## What's different from scanpy

A few places where pyscx behaves slightly differently from scanpy. These are
intentional and documented:

- **`pyscx.to_h5ad` is a free function**, not a method on the Experiment handle.
  It accepts `str | os.PathLike | pyscx.Experiment` as the source. Same for
  `pyscx.from_h5ad` (file source only, no Experiment).
- **`accel.calculate_qc_metrics` does not auto-tag MT genes.** Same contract as
  `scanpy.pp.calculate_qc_metrics`: tag `adata.var["mt"]` yourself and pass
  `qc_vars=["mt"]`. pyscx will emit a UserWarning if it sees MT-prefixed symbols
  in `var_names` and `qc_vars` is omitted.
- **`accel.leiden(device="cpu")` recommended for label stability.** The GPU
  Leiden path is faster but has a documented label-stability divergence vs
  `leidenalg`. Pin `device="cpu"` (the default for the CPU path) when downstream
  cares about cluster identity (DE, annotation transfer).
- **HVG with `flavor="seurat_v3"` expects raw counts.** Same contract as scanpy.
  Pass `layer="counts"` (or run HVG before `normalize_total` / `log1p`) to avoid
  the "non-integers were found" warning and statistically wrong HVGs.
- **GPU dispatch via `device=`.** Most ops accept `device="auto" | "cpu" | "gpu" | "gpu:N"`.
  See [the compatibility matrix in docs/scanpy.md](scanpy.md#compatibility-matrix)
  for which ops have GPU implementations.

## Next steps

- [docs/training.md](training.md) — ML training data loading guide (TrainingDataset, IndexPlanDataset, PyTorch Lightning).
- [docs/scanpy.md](scanpy.md) — full scanpy integration guide and accelerator reference.
- [docs/gpu-setup.md](gpu-setup.md) — CUDA / RAPIDS / SLURM setup for `device="gpu"`.
- [docs/architecture.md](architecture.md) — crate graph, file format, codec system.
- [docs/api.md](api.md) — API reference for the Python and Rust surfaces.
