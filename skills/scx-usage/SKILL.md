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
# Download the wheel for your Python version + arch (x86_64 / aarch64) from
# GitHub Releases: https://github.com/ArcInstitute/scx/releases (pyscx-v* tags).
# Replace <version> with the release you downloaded (e.g. 0.7.1).
pip install ./pyscx-<version>-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -c "import pyscx; print(pyscx.__version__)"
```

Add extras only when needed — attach them to the resolved wheel filename via
`pip install "$(ls ./pyscx-*.whl)[mudata]"` (a quoted `*` glob reaches pip
verbatim and fails); e.g. `[mudata]`, `[10x]`, `[gpu]` (cupy only — GPU kernels
need a source build). Pre-built wheels bundle libhdf5 **and** cloud I/O (Linux
x86_64, py ≥ 3.11); they do **not** include GPU.

**Developers** — clone the repo and compile the extension (a bare clone does
*not* work like `pip install`):

```bash
uv venv .venv
uv pip install -e "./pyscx[dev]"     # includes maturin[patchelf]
export PATH="$(pwd)/.venv/bin:$PATH"
cd pyscx && ../.venv/bin/maturin develop --release && cd ..
```

Use the repo `.venv/` only for *building* — not system Python. Source builds need
`libhdf5-dev` on Linux for h5ad ingest. Cloud is **not** in the default dev
build — pass `--features cloud` when you need `open_cloud()`. Rebuild after
branch switches or feature changes (`--features gpu`, etc.).

> **Before running, check existing conda envs for the deps your task needs —
> don't assume `.venv`, and don't create a new env without first looking.**
> Task-specific dependencies often live in only one environment: GPU analysis
> needs `rapids-singlecell` (+ `cupy`/`cugraph`/`cuml`), multimodal needs
> `mudata`, HVG `seurat_v3` needs `skmisc`, SLAF benchmarks need their own env,
> etc. A plain pip/uv `.venv` usually has none of these. List the available
> environments and inspect what they actually carry, then run from the one that
> already has what the task requires:
> ```bash
> conda env list
> conda list -n <env> | grep -iE 'rapids-singlecell|cupy|cugraph|mudata|scikit-misc|scanpy'
> ```
> The maturin editable `.so` is shared across environments (each carries a
> `pyscx.pth` pointing at the repo), so a single `--features gpu` build is
> importable from whichever env you select — you almost never need a *new* env,
> just the right *existing* one.

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

The CLI binary `scx` is separate from pyscx — install the pre-built binary from
[GitHub Releases](https://github.com/ArcInstitute/scx/releases) (`scx-cli-v*` tags),
or build from a clone with `cargo install --path scx-cli --features default-bin`
(the crate is not on crates.io). See `reference/installation.md`.

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
| MuData in memory | `pyscx.from_mudata(mdata, out, ...)` | — |
| 10x HDF5 | `pyscx.from_10x(h5, out)` | `scx convert in.h5 out.scx --from 10x` |
| Cell Ranger MTX dir | `pyscx.from_mtx(mtx_dir, out)` | `scx convert mtx_dir out.scx --from mtx` |

**Prefer `from_h5ad(path, ...)` over `from_anndata(sc.read_h5ad(path), ...)`**
for files larger than RAM — `from_h5ad` reads obs/var/uns via pure-Rust HDF5 and
never triggers anndata's eager `obsm` allocation. `from_anndata` also accepts a
*backed* AnnData (`sc.read_h5ad(p, backed='r')`) and auto-routes to streaming.

Most-used kwargs (shared across ingest entry points):
- `codec="auto"` (default; per-shard) — also `"scx1"` (integer-only), `"zstd"`,
  `"pcodec"` (best for float layers), `"lz4"`, `"none"`.
  Also `"compact-trial"` (trial-encode each shard with both heuristic winner
  and ShufDeltaZstd, keep the smaller; requires `row_group_rows=N`, e.g. 256)
  and `"shufdelta"` (force ShufDeltaZstd on all integer shards). **Codec
  choice guide:** use the default `"auto"` for GPU ML training and interactive
  CPU analysis (Scx1 has GPU in-VRAM decode + ~1.3–1.8× faster CPU decode);
  use `"compact-trial"` for storage-constrained archival or cloud hosting
  (1.3–2.1× smaller on integer counts, retains random access via framing).
  See [docs/codec.md § Codec tradeoff summary](../docs/codec.md#codec-tradeoff-summary--scx1-vs-shufdeltazstd)
  for the full comparison.
- `index_obs=[...]` / `index_var=[...]` / `index_preset="cellxgene"|"perturbseq"|"training"`
  — **materialize predicate indexes at write time** so a later
  `pyscx.open(...).query().filter_obs(...)` pushes the filter down. Without
  them, pushdown silently regresses to a full obs scan. Set these if the file
  will be queried.
- `csc="off"|"auto"|"always"` — write a column-major sidecar (needed for
  `prefer_format="csc"` accel paths + the CSC-direct `gpu_csc_v3` DE route; two-pass,
  transient disk ~2× output). `"auto"` builds it only when the dataset is large
  enough to benefit (`n_obs ≥ 50000` and `n_vars ≥ 5000`, env-tunable via
  `SCX_CSC_AUTO_OBS_THRESHOLD` / `SCX_CSC_AUTO_VARS_THRESHOLD`). To get the
  **GPU-fast** CSC-direct DE route (`gpu_csc_v3`), call `rank_genes_groups`/`pdex_ref`
  with the **default `prefer_format="csr"`** and `device="gpu"` (or `"auto"`) on a
  file that has the sidecar — the planner picks `gpu_csc_v3` automatically.
  `prefer_format="csc"` is the **CPU** column-major path (no GPU kernel);
  combining it with `device="gpu"` raises.
- `sort_by=[...]` / `reverse=True` — globally reorder the cell axis by obs
  columns at convert time (forces `stream=True`). CLI: `--sort-by CSV
  [--sort-reverse]`. Also available on `from_anndata`.
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
`scx compact`, `scx optimize` (in-place re-encode + row-group-frame → v4;
`--shard-obs off|auto|always`, default `auto`, migrates a legacy single-section
obs to the sharded layout when `n_obs > shard_target_rows`),
`scx build-csc <in> <out>` (add a CSC sidecar),
`scx sort <in> <out> --by CSV` (reorder cells by obs key for query locality),
`scx sort <in> <out> --shuffle --seed N` / `pyscx.shuffle` (seeded random reorder for training-batch diversity — the inverse of a sort; pin `--codec`),
`scx set-uns <file> --uns JSON` (replace uns in place),
`scx modify-metadata <file> --obs PARQUET --var PARQUET --obsm NAME=NPY ...
--index-*` (replace metadata sections in place).
Full flag lists in `reference/conversion.md`.

**Selective read / count from the CLI:** `scx query <file> <filter> [--count]
[--output OUT] [--explain]`. Worth knowing:
- The predicate is a **positional argument** here (`scx query f.scx "disease == 'normal'"`),
  or alternatively via `--filter EXPR` (matching `scx subset`/`scx delete`;
  provide one form, not both).
- `--count` is fast on a multi-shard, indexed file (sub-second on 31–62 shards);
  `--explain` prints the pushdown plan (Level-1 shard pruning + Level-2 row mask)
  so you can confirm the index is doing work. `scx query ... --output OUT.scx` is
  the local "write the matching cells to a new file" path.

**Cloud subcommands are build-gated.** `scx pull`/`push`/`explode`/`pack`/`cloud-optimize`
exist only in a `--features cloud` build; an hdf5-only `scx` reports
`error: unrecognized subcommand 'pull'`. Likewise `scx convert` from h5ad needs the
`hdf5` feature. Check `scx --help` for the subcommands your binary actually has.

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

**GPU rapids-singlecell routing:** When `rapids-singlecell` is installed
(conda — not pip), `device="gpu"` routes PCA → `rsc.pp.pca`, kNN →
`rsc.pp.neighbors`, UMAP → `rsc.tl.umap`, and preprocessing → `rsc.pp.*`.
Native GPU paths survive for streaming/randomized PCA (>VRAM), HVG
`seurat_v3`, Leiden, DE, and Harmony. If rapids is absent, GPU ops fall
back to CPU with `FallbackReason::NoRapids`.

> **rapids lives in a conda env, and the GPU build is what you import.** Pip/uv
> `.venv`s typically don't have `rapids-singlecell`, so `device="gpu"` there
> silently CPU-falls-back. The maturin editable `.so` is shared across envs, so
> the working setup is: build pyscx once with `--features gpu`, then run from the
> conda env that *already* has rapids-singlecell — **check before assuming or
> creating one**:
> ```bash
> conda env list
> # pick the env whose list includes rapids-singlecell (it pulls cupy/cugraph/cuml):
> conda list -n <env> | grep -iE 'rapids-singlecell|cupy|cugraph'
> conda run -n <env> python -c "import pyscx, rapids_singlecell; print('ok')"
> ```
> Verified GPU is real and fast — PCA was ~22× CPU and bit-identical on 1 M
> cells, with route metadata correctly naming the backend. **Always confirm via
> `adata.uns["scx_accel"][op]["route"]`** (`rapids_singlecell_gpu` / `gpu_csr` /
> `cpu_*`), since silent fallback is the common failure. `harmony_integrate`
> stamps the same envelope (`gpu_dense` / `cpu_dense`).

`to_gpu_anndata()` on an `Experiment` returns a GPU-resident AnnData
(`X` = `cupyx.scipy.sparse.csr_matrix`). It records the path in
`uns["scx_accel"]["to_gpu_anndata"]["transfer_mode"]`: `scx_device_decode_gpu`
(in-VRAM decode, only the indptr uploaded — needs Scx1 count shards + decode
sidecars) vs `scx_device_handoff_streamed` (host-bounce, used for mixed-codec /
sidecar-less files). It's **≤VRAM only** (raises if the matrix won't fit) and may
emit a debug-formatted `eager_assembly_memory_high` warning on large files —
that's non-fatal; pass `var_names=`/`obs_filter=` to shrink the handoff.
Env vars: `SCX_FORCE_NATIVE_GPU=1` pins surviving native GPU paths;
`SCX_DISABLE_RAPIDS=1` forces the no-rapids CPU fallback for testing.

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
- **`leiden(device="cpu")` for label stability.** GPU (cuGraph) and CPU now give
  comparable cluster *counts* (the old `n_iterations`→`max_iter` unit bug that
  produced ~116k degenerate clusters is fixed), but their labels still differ by
  design (ARI ≈ 0.72, not 1.0). Pin CPU whenever downstream cares about exact
  cluster identity (DE, annotation transfer).
- `accel.subset_obs` with an integer array becomes a boolean mask — **order not
  preserved, duplicates collapsed** (unlike NumPy fancy indexing).

The `pyscx.accel.*` catalog (PCA/neighbors/UMAP/leiden, DE & perturbation
metrics, HVG/QC, PFlog (v4) shifted-log normalization, gene-set scoring
(`score_genes`), pseudobulk NB GLM (`nb_glm`/`pdex_nb_glm`), harmony, LISI,
streaming column stats, fused GPU pipelines (`pca_neighbors`,
`pca_neighbors_umap`)), the `device=` and `prefer_format=` dispatch, and the
full backed-mode scanpy-compat table are in `reference/processing.md`.

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

### Grouped sharding — per-perturbation reads
Physical co-location of groups on disk. Convert or sort with `group_by=`:
```python
pyscx.from_h5ad("screen.h5ad", "screen.scx",
                group_by="target_gene", reference=["non-targeting"])
```
Then read one perturbation at a time (1–2 shard reads, not ~120):
```python
exp = pyscx.open("screen.scx")
adata = exp.read_group("MYC")            # AnnData, 1-2 shard reads
ref   = exp.read_reference()             # AnnData | None (shard 0)
for gs in exp.iter_group_shards():       # streaming, ~one shard resident
    ad = gs.to_anndata()
```
Cloud: `pyscx.open_cloud(url).read_group("MYC")` — same API, range reads.
`append` drops the group index — re-sort to restore.

Full constructor kwargs, batch schemas, shuffling semantics, train/val/test
splitting, grouped sharding details, and Lightning examples are in
`reference/ml-loading.md`.

---

## Quick gotcha checklist
- **Install:** end users → `pip install ./pyscx-*.whl` (from GitHub Releases); devs → `uv pip install -e "./pyscx[dev]"` then `maturin develop`. Rebuild after pulling Rust changes.
- Pick the approach by *subset?* × *fits in RAM?* — `query().collect()` materializes, so a too-big subset needs `to_anndata(backed=True, obs_filter=...)`.
- Predicate grammar differs: `query().filter_obs()` (engine + pushdown) vs backed `obs_filter=` (pandas `.query()`).
- Convert with `index_obs=`/`--index-preset` if the file will be queried, else pushdown is a full scan.
- `pyscx.to_h5ad` / `from_h5ad` are **free functions**; `to_anndata` / `query` are Experiment methods.
- Backed mode: `pyscx.accel.normalize_total/log1p`, **not** `sc.pp.*` (which materialize).
- `qc_vars=["mt"]` + tag `var["mt"]` yourself; HVG seurat_v3 on raw counts; `leiden(device="cpu")` for stable labels.
- Training loaders: `num_workers=0`; HVG indices as `np.uint32`; `close()` when done. `pflog=True` for PFlog (v4) normalization (replaces `normalize`/`log1p`; `pflog_alpha=None` estimates α once, or pin a float).
- GPU ops fall back to CPU silently — check `pyscx.accel.gpu_info()` / `nvidia-smi` / `adata.uns["scx_accel"]` route. PCA/kNN/UMAP/preprocess route to rapids-singlecell when installed; `SCX_DISABLE_RAPIDS=1` forces CPU fallback for testing. Build pyscx `--features gpu` once, then **run from the conda env that has rapids** (a plain `.venv` usually doesn't).
- GPU-fast CSC-direct DE (`gpu_csc_v3`): call `rank_genes_groups`/`pdex_ref` with the **default `prefer_format="csr"`** + `device="gpu"` (or `"auto"`) on a file with a `csc=` sidecar — the planner picks it automatically. `prefer_format="csc"` is the **CPU** path; `prefer_format="csc"` + `device="gpu"` raises.
- CLI predicate: `scx query <file> <filter>` accepts the filter positionally **or** via `--filter EXPR` (matching `scx subset`/`delete`; one form, not both); cloud subcommands (`scx pull`/`push`/…) only exist in a `--features cloud` build.
- `compute_lisi` defaults to O(N²) exact kNN — minutes-to-tens-of-minutes at ≥1M cells; pass `approximate_knn=True` for an HNSW kNN (~10× faster, small drift) or subsample to evaluate integration.
