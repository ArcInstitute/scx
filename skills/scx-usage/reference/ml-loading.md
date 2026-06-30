# ML data loading reference (pyscx)

Self-contained reference for SCX training data loaders. They wrap a Rust
triple-buffered pipeline (tokio I/O → rayon decode → Python/GPU consumer) so the
hot path has **no Python and needs no `num_workers`** — threading is in compiled
Rust. Each `for batch in dataset:` loop is one epoch; shards reshuffle between
epochs.

Compute HVG indices once on **raw counts** and reuse them across runs:
```python
import numpy as np, pyscx
from pyscx import accel
adata = pyscx.open("experiment.scx").to_anndata(backed=True)
accel.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
hvg = np.where(adata.var["highly_variable"].to_numpy())[0].astype(np.uint32)
np.save("hvg.npy", hvg)   # reload with np.load for later runs
```

## TrainingDataset — sequential streaming (common case)

```python
ds = pyscx.TrainingDataset("experiment.scx", batch_size=1024, hvg_indices=hvg,
                           normalize=True, log1p=True, target_sum=1e4,
                           obs_columns=["cell_type", "batch"], max_memory_mb=512)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(n_epochs):
    for batch in ds:
        x   = torch.from_numpy(batch["X"]).to(dev)   # [B, n_output_genes] float32 dense
        obs = batch["obs"]
        idx = batch["cell_indices"]                   # [B] int64 global rows
ds.close()
```

**Constructor kwargs:**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx`. |
| `batch_size` | `1024` | Auto-tuned downward only if an **explicit** `max_memory_mb` is exceeded; the adaptive default budget preserves it. |
| `hvg_indices` | `None` | `np.ndarray[uint32]` of gene indices; `None` = all genes. |
| `obs_columns` | `[]` | Obs column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in one CSR row scan). |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target. |
| `shard_group_size` | `8` | Shards per I/O group (sequential reads within a group). |
| `prefetch_batches` | `4` | Ring buffer depth. |
| `seed` | `42` | Deterministic shuffle via `(seed, epoch)`. |
| `max_memory_mb` | adaptive (≥512) | Omit → adaptive budget scales to fit a full-width file (floor 512 MB, cap 4096 MB), so the requested `batch_size` survives. Pass a value → hard ceiling that auto-tunes `shard_group_size`/`prefetch_batches`/`batch_size` down. |
| `modality` | `None` | For multimodal v2 files; ignored on single-modality. |
| `pflog1ppf` | `False` | Apply PFlog1pPF normalization (Booeshaghi et al. 2026) instead of normalize+log1p. Mutually exclusive with `normalize`/`log1p` (those are ignored when `pflog1ppf=True`). |
| `pflog1ppf_c` | `1.0` | PFlog1pPF shift / pseudocount `c`. Only used when `pflog1ppf=True`. |

**Properties:** `n_obs`, `n_vars`, `n_output_genes` (HVG count if projected else
`n_vars`), `effective_batch_size`.
**Methods:** `close()` (idempotent; recommended before process exit),
`memory_budget()` (diagnostics dict).

**PFlog1pPF normalization:** Pass `pflog1ppf=True` (optionally with `pflog1ppf_c`)
to apply PFlog1pPF (shifted-CLR) normalization in the Rust pipeline instead of
the default `normalize_total → log1p`. When enabled, `normalize` and `log1p`
are ignored. Available on `TrainingDataset`, `IndexPlanDataset`, and
`MultimodalTrainingDataset`.

**Batch dict schema:**
```python
{
  "X":            np.ndarray[B, n_output_genes, float32],   # dense
  "obs":          dict[str, np.ndarray | {"codes": int32[], "categories": [str]}],
  "cell_indices": np.ndarray[B, int64],                      # global row indices
}
```
Categorical obs → `{"codes": int32[], "categories": [str]}`; numeric obs →
`int64[]` or `float64[]`.

**Shuffling:** Level 1 permutes shard indices via a deterministic RNG seeded from
`(seed, epoch)`, grouped into `shard_group_size` I/O groups for disk-sequential
reads; Level 2 Fisher-Yates shuffles cell indices within each group. Same
`(seed, epoch)` always yields the identical ordering.

## IndexPlanDataset — paired (perturbed, control) batches

For perturbation, contrastive, or donor-matched training. Consumes a Python
iterator of plans (`list[tuple[int, int]]`) and yields paired dense batches.

```python
ds = pyscx.IndexPlanDataset("atlas.scx", hvg_indices=hvg,
                            obs_columns=["cell_type", "perturbation"],
                            normalize=True, target_sum=1e4)
plans = [[(0, 5), (2, 7)], [(10, 100), (50, 75)]]
for batch in ds.iter_with_plans(iter(plans), lookahead=4):
    X, X_paired, pairs = batch["X"], batch["X_paired"], batch["pairs"]
```

**Constructor kwargs:** `path`, `hvg_indices=None`, `obs_columns=[]`,
`normalize=True`, `log1p=True`, `target_sum=1e4`, `pflog1ppf=False`,
`pflog1ppf_c=1.0`, `cache_shards=128` (LRU; auto-tuned to fit `max_memory_mb`;
check `effective_cache_shards()`), `sort_by_shard=True` (reorder each plan by
shard locality; disable to preserve caller order), `lookahead=4` (in-flight
plans / shard prefetch; `0` disables; check `effective_lookahead()`),
`max_plan_size=16384`, `max_memory_mb=512`.

**Batch dict** (from `iter_with_plans`): `"X"` (perturbed rows), `"X_paired"`
(control rows), `"pairs"` (post-sort plan), `"obs"`, `"obs_paired"`. `pairs[i]`
corresponds to `X[i]` / `X_paired[i]`.

**Iterator semantics:** `plans` is any iterable yielding `list[tuple[int,int]]`
(lists, generators, queues). Lazy pull; `lookahead` plans in flight. Empty plans
skipped. `StopIteration` ends cleanly; other exceptions propagate as
`RuntimeError("plan iterator raised: ...")`. Out-of-range row indices raise
`IndexError`; missing obs columns raise `KeyError` at construction.

Plan generator pattern (wrap any pairing policy):
```python
def plan_generator(strategy, perturbed_indices, batch_size):
    buf = []
    for p_idx in perturbed_indices:
        c_idx = strategy.get_control_index(p_idx)
        if c_idx is not None:
            buf.append((p_idx, c_idx))
        if len(buf) == batch_size:
            yield buf; buf = []
    if buf:
        yield buf
```

## MultimodalTrainingDataset — CITE-seq / Multiome / TEA-seq

```python
ds = pyscx.MultimodalTrainingDataset("citeseq.scx", modalities=["rna", "adt"],
                                     batch_size=1024, normalize=True, log1p=True,
                                     obs_columns=["cell_type"])
for batch in ds:
    x_rna = torch.from_numpy(batch["X"]["rna"]).to(device)
    x_adt = torch.from_numpy(batch["X"]["adt"]).to(device)
    # cell axes aligned: x_rna[i] and x_adt[i] are the same cell
```
Differences from `TrainingDataset`: categorical obs are decoded `list[str]` (not
the `{"codes","categories"}` dict); `n_vars` is a `dict[str,int]` (per-modality
counts); `return_dict=False` yields tuples of X arrays only (no obs /
cell_indices).

## Train / val / test splits

- **Query-based (recommended):** if obs has a `split` column, materialize each
  split into its own `.scx`:
  ```python
  exp = pyscx.open("atlas.scx")
  pyscx.from_anndata(exp.query().filter_obs("split == 'train'").collect().to_anndata(), "train.scx")
  pyscx.from_anndata(exp.query().filter_obs("split == 'val'").collect().to_anndata(),   "val.scx")
  train_ds = pyscx.TrainingDataset("train.scx", batch_size=1024)
  val_ds   = pyscx.TrainingDataset("val.scx",   batch_size=1024)
  ```
- **Index-based:** mask on `batch["cell_indices"]` against a precomputed
  `train_idx` set (simple, but reads all rows each epoch).
- **Convert-time:** slice the AnnData by `obs["split"]` before `from_anndata`.

## PyTorch Lightning

- **scVI:** `from pyscx.scx_integrations.scvi import ScxDataModule` — wraps
  `TrainingDataset` (num_workers=0, Rust threading); exposes `n_obs`, `n_vars`,
  `n_output_genes`.
- **Generic:** wrap `TrainingDataset` in a `LightningDataModule`; the dataloader
  is `DataLoader(ds, batch_size=None, num_workers=0)` — `batch_size=None`
  because SCX already batches.

## DataLoader with `num_workers > 0` (fork safety)

`TrainingDataset` and `IndexPlanDataset` are fork-safe under
`DataLoader(num_workers>0, start_method="fork")` (the Linux default) **only when
the dataset is constructed lazily inside the worker's `__iter__`**, not in the
parent. The per-pipeline tokio runtime and rayon pool are then built inside the
child, so no fork-hostile state is inherited. Constructing in the parent and
forking raises `RuntimeError`. `num_workers=0` is the intended mode — the Rust
pipeline already parallelizes.

## `to_anndata(backed=True)` vs TrainingDataset

Backed AnnData is for **analysis** (random slicing, scanpy/`accel` ops).
TrainingDataset is for **training throughput** (sequential dense batches,
normalize+log1p fused in Rust, two-level epoch shuffle). Don't drive a training
loop off backed slicing.

### Random-access loaders that read an `obsm` embedding (`embed_key`)

If you drive a `DataLoader` off backed `X` *and* an obsm embedding per cell
(`adata.obsm[embed_key][cell_indices]`), pass `obsm=[embed_key]`:

```python
adata = pyscx.open("atlas.scx").to_anndata(backed=True, obsm=["X_state"])
emb = adata.obsm["X_state"]        # ScxBackedObsmDataset (shard-aware, dense)
batch_emb = emb[cell_indices]      # reads only the touched obsm shards
```

Without `obsm=[...]`, `to_anndata` materialises **every** obsm key × **all**
rows into a dense numpy dict at open time — paid per key (even unused ones) and
**per DataLoader worker** (each re-opens the reader), which OOMs at multi-million
cell scale. `obsm=[embed_key]` loads only the key you use and (in backed mode)
gathers rows lazily (`O(batch)` memory, per-key LRU = `cache_shards`). See
[Selective + lazy `obsm`](../../../docs/scanpy.md#selective--lazy-obsm-obsm).

## Troubleshooting
- `RuntimeError: scx.TrainingDataset requires num_workers=0` — see fork-safety above.
- `RuntimeError: Must call __iter__ before __next__` — iterate with `for batch in ds:`, don't call `next()` on the dataset directly.
- Memory budget exceeded — lower `max_memory_mb`, `batch_size`, or `shard_group_size`; check `ds.memory_budget()`.
