---
name: scx-usage
description: How to USE scx (pyscx + scx-cli) to get real work done — installing pyscx (GitHub Release wheels vs source, optional features, common install failures), converting data into/out of SCX (h5ad/h5mu/10x/mtx), processing single-cell data (in-memory / backed-lazy / query pipeline, the pyscx.accel.* accelerators), and ML data loading (TrainingDataset, IndexPlanDataset, MultimodalTrainingDataset). Trigger when installing, writing, or debugging pyscx/scx-cli code for conversion, preprocessing/QC/clustering, or training loaders.
---

# Using scx (pyscx + scx-cli)

SCX is a binary file format + query engine + ML loader for single-cell RNA-seq
data, a Rust-native alternative to AnnData/h5ad. `pyscx` is the Python binding;
`scx` is the CLI. This skill is a task-oriented reference for the three things
users do with it: **convert** data, **process** it, and **load it for ML
training**.

This file is the entry point — enough to write correct code for the common
cases, plus the gotchas that are easy to get wrong. For exhaustive signatures
and edge cases, read the bundled reference files (self-contained, in this skill
directory):

- `reference/installation.md` — GitHub Release wheels vs source install, optional extras, verify
  steps, and troubleshooting (missing `.so`, HDF5, patchelf/rpath, GPU fallback,
  venv/conda conflicts).
- `reference/conversion.md` — every ingest/export entry point + all kwargs.
- `reference/processing.md` — the three approaches in depth + the full `pyscx.accel.*` catalog.
- `reference/ml-loading.md` — TrainingDataset / IndexPlanDataset / multimodal / Lightning, full constructor refs.

---

## 0. Installing pyscx

**End users** — install the wheel; no Rust toolchain needed:

```bash
# Download the wheel for your Python version from GitHub Releases:
# https://github.com/ArcInstitute/scx/releases (look for pyscx-v* tags)
pip install ./pyscx-0.6.3-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -c "import pyscx; print(pyscx.__version__)"
```

Add extras only when needed: `'./pyscx-*.whl[mudata]'`, `'./pyscx-*.whl[10x]'`,
`'./pyscx-*.whl[gpu]'` (cupy only — GPU kernels need a source build). Pre-built
wheels bundle libhdf5 **and** cloud I/O (Linux x86_64, py ≥ 3.11); they do **not**
include GPU.

**Developers** — clone the repo and compile the extension (a bare clone does
*not* work like `pip install`):

```bash
uv venv .venv
uv pip install -e "./pyscx[dev]"     # includes maturin[patchelf]
export PATH="$(pwd)/.venv/bin:$PATH"
cd pyscx && ../.venv/bin/maturin develop --release && cd ..
```

Use the repo `.venv/` only — not system Python. Source builds need
`libhdf5-dev` on Linux for h5ad ingest. Cloud is **not** in the default dev
build — pass `--features cloud` when you need `open_cloud()`. Rebuild after
branch switches or feature changes (`--features gpu`, etc.).

**Sanity check after any install:**

```python
import pyscx
import pyscx.pyscx as native
from pyscx import accel
print(native.__file__)             # must be a .so / .pyd
print(accel.gpu_available())       # False is fine on CPU-only / pre-built wheels
```

**Common gotchas** (full troubleshooting in `reference/installation.md`):

| Symptom | Likely cause | Fix |
|---|---|---|
| `No module named 'pyscx'` | Wheel not installed, or dev build skipped | `pip install ./pyscx-*.whl` or run `maturin develop` |
| `from_h5ad` → `NotImplementedError` | Built without HDF5 | Pre-built wheel: reinstall from GitHub Releases; source: `libhdf5-dev` + `maturin develop` |
| "Failed to set rpath" on every build | Missing patchelf | `uv pip install -e "./pyscx[dev]"` + `export PATH=.venv/bin:$PATH` |
| GPU feels like CPU | Pre-built wheel and/or no GPU build/device | Build from source with `--features gpu`; check `accel.gpu_info()` / `nvidia-smi` |
| `open_cloud` missing | Source build without cloud feature | `maturin develop --features cloud` (included in pre-built wheels) |
| maturin errors with venv + conda | Both `VIRTUAL_ENV` and `CONDA_PREFIX` set | `unset VIRTUAL_ENV` or deactivate conda before building |

The CLI binary `scx` is separate from pyscx (`cargo install --features default-bin scx-cli`).

---

## 1. Converting data

All h5ad/h5mu ingest **and** export entry points **stream by default**
(`stream=True`) — peak RSS is bounded by one shard's worth of CSR per matrix,
independent of total file size. Pass `stream=False` only to force the legacy
materializing path.

### Into SCX

| Source | Python | CLI |
|---|---|---|
| h5ad file on disk | `pyscx.from_h5ad(path, out, ...)` | `scx convert in.h5ad out.scx --stream` |
| AnnData in memory | `pyscx.from_anndata(adata, out, ...)` | — |
| h5mu (multimodal) | `pyscx.from_h5mu(path, out, modalities=..., modality_types=...)` | `scx convert in.h5mu out.scx --modalities rna,adt` |
| 10x HDF5 | `pyscx.from_10x(h5, out)` | `scx convert in.h5 out.scx --from 10x` |
| Cell Ranger MTX dir | `pyscx.from_mtx(mtx_dir, out)` | `scx convert mtx_dir out.scx --from mtx` |

**Prefer `from_h5ad(path, ...)` over `from_anndata(sc.read_h5ad(path), ...)`**
for files larger than RAM — `from_h5ad` reads obs/var/uns via pure-Rust HDF5 and
never triggers anndata's eager `obsm` allocation. `from_anndata` also accepts a
*backed* AnnData (`sc.read_h5ad(p, backed='r')`) and auto-routes to streaming.

Most-used kwargs (shared across ingest entry points):
- `codec="auto"` (default; per-shard) — also `"scx1"` (integer-only), `"zstd"`,
  `"pcodec"` (best for float layers), `"lz4"`, `"none"`.
- `index_obs=[...]` / `index_var=[...]` / `index_preset="cellxgene"|"perturbseq"|"training"`
  — **materialize predicate indexes at write time** so a later
  `pyscx.open(...).query().filter_obs(...)` pushes the filter down. Without
  them, pushdown silently regresses to a full obs scan. Set these if the file
  will be queried.
- `csc="off"|"auto"|"always"` — write a column-major sidecar (needed for
  `prefer_format="csc"` accel paths + the GPU `gpu_csc_v3` DE route; two-pass,
  transient disk ~2× output). `"auto"` builds it only when the dataset is large
  enough to benefit (`n_obs ≥ 50000` and `n_vars ≥ 5000`, env-tunable via
  `SCX_CSC_AUTO_OBS_THRESHOLD` / `SCX_CSC_AUTO_VARS_THRESHOLD`).
- `memory_budget="4G"` (bare bytes or a binary-prefixed size: `K`/`M`/`G`/`T`
  or `KiB`/`MiB`/`GiB`/`TiB`, powers of 1024; decimal `KB`/`MB`/`GB`/`TB`
  rejected), `strict_uns=True`, `shard_size`. See `reference/conversion.md` for the rest
  (`reader_threads`, `bitmap`, `dense_zero_epsilon`, `temp_dir`, overrides).

### Out of SCX
- `pyscx.to_h5ad(path_or_experiment, out, stream=True, modality=None)` — a
  **free function**, not a method. Source can be a path or a `pyscx.Experiment`.
  Deletion vectors are honored (only kept rows written). Multimodal files need
  `modality="rna"` to extract one modality.
- `pyscx.to_h5mu(path, out)` (multimodal), `pyscx.to_mtx(scx, out_dir)`.

> `to_h5ad(exp, ...)` re-exports the **unmodified on-disk** file. To persist an
> in-memory analysis, use `adata.write_h5ad(...)` or
> `pyscx.from_anndata(adata, "out.scx")`.

### Inspect / file ops (CLI)
`scx info <file> [--json --history]`, `scx validate <file>`,
`scx subset <in> --filter <expr> --genes <path>`, `scx merge`, `scx append`,
`scx compact`. Full flag lists in `reference/conversion.md`.

---

## 2. Processing data

**Decide the approach first** — by two questions: **(a)** do you need only a
*subset* (a filter)? **(b)** does the working set *fit in RAM*? In-memory CSR ≈
`8·n_obs + 8·nnz` bytes (i32 indices + f32 data); e.g. 1M cells × 30K genes @ 5%
density ≈ 12 GB. (Details + tradeoff table in `reference/processing.md`.)

| Scenario | Approach | API | Peak mem |
|---|---|---|---|
| Whole dataset, fits in RAM | **In-memory** | `to_anndata()` then `sc.pp.*` / `sc.tl.*` | full matrix |
| Whole dataset, too big | **Backed + lazy** | `to_anndata(backed=True)` then `pyscx.accel.*` | ~1 shard (~128 MB) |
| Subset, fits in RAM | **Query → collect** | `.query().filter_obs(...).collect().to_anndata()` then `sc.pp.*` | subset (materialized) |
| Subset, still too big | **Backed + filter** | `to_anndata(backed=True, obs_filter="...")` then `pyscx.accel.*` | ~1 shard |

`query().collect()` **always materializes** the subset into scipy — pick it only
when the *filtered* result fits in RAM. For a lazy, out-of-core filtered view
(filter, then `accel.*`), use `to_anndata(backed=True, obs_filter=...)`; there is
no `query().collect(backed=True)` today.

**Two predicate entry points, two grammars.** `query().filter_obs(expr)` is
evaluated by the scx engine and does *shard pushdown* — but only skips shards if
`index_obs=` / `--index-preset` was set at convert time (otherwise a full obs
scan, still correct, just not faster). The backed `obs_filter="expr"` kwarg is
evaluated by **pandas `.query()`** (richer grammar, no pushdown). Same intent,
different engines — don't assume an expression behaves identically in both.

### In-memory (full scanpy compatibility)
```python
import pyscx, scanpy as sc
adata = pyscx.open("experiment.scx").to_anndata()   # zero-copy CSR float32
sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata); sc.pp.pca(adata)
sc.pp.neighbors(adata); sc.tl.umap(adata); sc.tl.leiden(adata)
```

### Backed + lazy (out-of-core) — use `pyscx.accel.*` for anything touching X
```python
import pyscx
from pyscx import accel
adata = pyscx.open("atlas.scx").to_anndata(backed=True)
accel.filter_cells(adata, min_genes=200)         # updates deletion vector, no materialize
accel.filter_genes(adata, min_cells=3)            # sets column projection
accel.normalize_total(adata, target_sum=1e4)      # lazy — appends a transform
accel.log1p(adata)                                # lazy — fuses with normalize
accel.pca(adata, n_comps=50)                      # STREAMS through lazy transforms
accel.neighbors(adata, n_neighbors=15, use_rep="X_pca")
accel.umap(adata)
accel.leiden(adata, resolution=1.0, device="cpu") # CPU for label stability
accel.rank_genes_groups(adata, groupby="leiden")
```
In backed mode `sc.pp.normalize_total` / `sc.pp.log1p` **force full
materialization** — always use the `accel` versions. `sc.tl.*` / `sc.pl.*` work
normally. `accel.*` write to the standard AnnData slots (`obsm["X_pca"]`,
`obsp["distances"]`/`["connectivities"]`, `obsm["X_umap"]`, `obs[leiden]`), so
downstream scanpy is unchanged.

### Query pipeline (predicate pushdown → subset)
```python
adata = (pyscx.open("atlas.scx").query()
         .filter_obs("tissue == 'lung'").select_genes(idx)
         .with_normalize(1e4).with_log1p()
         .collect().to_anndata())
sc.pp.pca(adata)   # regular scanpy from here
```
`collect()` materializes the subset (see the decision table above): use this
only when the filtered result fits in RAM, else switch to
`to_anndata(backed=True, obs_filter="tissue == 'lung'")`.

### Critical processing gotchas
- **MT genes are not auto-tagged.** Set `adata.var["mt"] =
  adata.var_names.str.startswith("MT-")` (mouse: `"mt-"`) and pass
  `qc_vars=["mt"]` to `calculate_qc_metrics`, else `pct_counts_mt` is never
  produced. On CELLxGENE Census, gene symbols live in `var["feature_name"]`
  (var_names are integer strings) — tag from `feature_name` or the mask is all-False.
- **`highly_variable_genes(flavor="seurat_v3")` expects raw counts.** Run it
  before `normalize_total`/`log1p`, or stash `adata.layers["counts"]` and pass
  `layer="counts"`. Otherwise → statistically wrong HVGs + a warning.
- **`leiden(device="cpu")` for label stability.** GPU Leiden diverges from
  `leidenalg` (documented, not a bug). Pin CPU when downstream cares about
  cluster identity (DE, annotation transfer).
- `accel.subset_obs` with an integer array becomes a boolean mask — **order not
  preserved, duplicates collapsed** (unlike NumPy fancy indexing).

The `pyscx.accel.*` catalog (PCA/neighbors/UMAP/leiden, DE & perturbation
metrics, HVG/QC, harmony, LISI, streaming column stats), the `device=` and
`prefer_format=` dispatch, and the full backed-mode scanpy-compat table are in
`reference/processing.md`.

---

## 3. ML data loading

Rust triple-buffered pipeline (I/O → decode → Python). **No `num_workers`** —
threading is in compiled Rust. Each `for batch in dataset:` is one epoch; shards
reshuffle between epochs.

### TrainingDataset — sequential streaming (the common case)
```python
import numpy as np, pyscx, torch
hvg = np.where(adata.var["highly_variable"])[0].astype(np.uint32)
ds = pyscx.TrainingDataset(
    "experiment.scx", batch_size=1024, hvg_indices=hvg,
    normalize=True, log1p=True, target_sum=1e4,
    obs_columns=["cell_type", "batch"],   # included in each batch dict
    max_memory_mb=512,                     # auto-tunes shard grouping/prefetch/batch
)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(n_epochs):
    for batch in ds:
        x = torch.from_numpy(batch["X"]).to(dev)   # [B, n_output_genes] float32 dense
        idx = batch["cell_indices"]                 # [B] int64 global rows
        # categorical obs → {"codes": int32[], "categories": [str]}; numeric → int64/float64
ds.close()   # idempotent; call before process exit
```
Compute HVGs once on raw counts and save the `uint32` indices for reuse.

### IndexPlanDataset — paired (perturbed, control) batches
For perturbation / contrastive / donor-matched training. Consumes a Python
iterator of plans (`list[tuple[int, int]]`); yields `X` (perturbed) + `X_paired`
(control) + `pairs`.
```python
ds = pyscx.IndexPlanDataset("atlas.scx", hvg_indices=hvg,
                            obs_columns=["cell_type", "perturbation"], normalize=True)
for batch in ds.iter_with_plans(iter(plans), lookahead=4):
    X, X_paired, pairs = batch["X"], batch["X_paired"], batch["pairs"]
```

### Other
- **Multimodal:** `pyscx.MultimodalTrainingDataset(path, modalities=[...])`.
- **PyTorch Lightning:** `from pyscx.scx_integrations.scvi import ScxDataModule`
  (wraps TrainingDataset internally), or a generic DataModule.
- **`DataLoader(num_workers > 0)`:** fork-safe **only if the dataset is
  constructed lazily inside the worker's `__iter__`**, not in the parent.
  `num_workers=0` is the intended mode — the Rust pipeline already parallelizes.

Full constructor kwargs, batch schemas, shuffling semantics, train/val/test
splitting, and Lightning examples are in `reference/ml-loading.md`.

---

## Quick gotcha checklist
- **Install:** end users → `pip install ./pyscx-*.whl` (from GitHub Releases); devs → `uv pip install -e "./pyscx[dev]"` then `maturin develop`. Rebuild after pulling Rust changes.
- Pick the approach by *subset?* × *fits in RAM?* — `query().collect()` materializes, so a too-big subset needs `to_anndata(backed=True, obs_filter=...)`.
- Predicate grammar differs: `query().filter_obs()` (engine + pushdown) vs backed `obs_filter=` (pandas `.query()`).
- Convert with `index_obs=`/`--index-preset` if the file will be queried, else pushdown is a full scan.
- `pyscx.to_h5ad` / `from_h5ad` are **free functions**; `to_anndata` / `query` are Experiment methods.
- Backed mode: `pyscx.accel.normalize_total/log1p`, **not** `sc.pp.*` (which materialize).
- `qc_vars=["mt"]` + tag `var["mt"]` yourself; HVG seurat_v3 on raw counts; `leiden(device="cpu")` for stable labels.
- Training loaders: `num_workers=0`; HVG indices as `np.uint32`; `close()` when done.
- GPU ops fall back to CPU silently — check `pyscx.accel.gpu_info()` / `nvidia-smi` if you expected GPU.
