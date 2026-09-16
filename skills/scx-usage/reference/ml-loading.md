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
| `hvg_indices` | `None` | `np.ndarray[uint32]` of gene indices; `None` = all genes. Every index must be `< n_vars`. Sorted and deduplicated, so batch columns come back in ascending gene-index order whatever order you pass — `np.unique(hvg_indices)` is the column order. |
| `obs_columns` | `[]` | Obs column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in one CSR row scan). |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target. |
| `shard_group_size` | `8` | Shards per I/O group (sequential reads within a group). |
| `prefetch_batches` | `4` | Ring buffer depth. |
| `seed` | `42` | Deterministic shuffle via `(seed, epoch)`. |
| `max_memory_mb` | adaptive (≥512) | Omit → adaptive budget scales to fit a full-width file (floor 512 MB, cap 4096 MB), so the requested `batch_size` survives. Pass a value → hard ceiling that auto-tunes `shard_group_size`/`prefetch_batches`/`batch_size` down. |
| `modality` | `None` | For multimodal v2 files; ignored on single-modality. |
| `pflog` | `False` | Apply PFlog (v4) shifted-log normalization on raw counts (Booeshaghi et al.) instead of normalize+log1p. Mutually exclusive with `normalize`/`log1p` (those are ignored when `pflog=True`). |
| `pflog_alpha` | `None` | PFlog NB overdispersion `α` (matrix-wide pseudocount `1/(4α)`). `None` estimates `α` once at construction from the raw counts (single-modality only); a float pins it. Only used when `pflog=True`. |

**Properties:** `n_obs`, `n_vars`, `n_output_genes` (HVG count if projected else
`n_vars`), `effective_batch_size`.
**Methods:** `close()` (idempotent; recommended before process exit),
`memory_budget()` (diagnostics dict).

**PFlog normalization:** Pass `pflog=True` (optionally with `pflog_alpha`) to
apply PFlog (v4, shifted-log on raw counts) normalization in the Rust pipeline
instead of the default `normalize_total → log1p`. `pflog_alpha=None` estimates the
NB overdispersion `α` once at loader construction (single-modality only; a
modality-scoped loader must pin `pflog_alpha`); a float pins a reference `α`.
When enabled, `normalize` and `log1p` are ignored. Available on
`TrainingDataset`, `IndexPlanDataset`, and `MultimodalTrainingDataset`.

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
`normalize=True`, `log1p=True`, `target_sum=1e4`, `pflog=False`,
`pflog_alpha=None`, `cache_shards=128` (LRU; auto-tuned to fit `max_memory_mb`;
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

## Grouped sharding — per-perturbation reads

Physical co-location of perturbation groups on disk. `scx sort --group-by` (or
`from_h5ad(group_by=)`) sorts rows by group label, bin-packs groups into shards
cutting only at group edges, and isolates reference cells in shard 0. Result:
`read_group("MYC")` touches 1–2 shards instead of the full file — 100–1000× I/O
reduction per group read.

**Write-time:**
```python
# From h5ad — sort + convert in one step
pyscx.from_h5ad("screen.h5ad", "screen.scx",
                group_by="target_gene", reference=["non-targeting"])

# Re-sort an existing .scx
pyscx.sort("screen.scx", "screen_grouped.scx",
           group_by="target_gene", reference=["non-targeting"])
```

`reference` accepts `list[str]` (label values treated as reference) **or**
`{"column": "is_control"}` (boolean obs column — rows where that column is
`True` become reference cells).

**Read-time:**
```python
exp = pyscx.open("screen_grouped.scx")
adata     = exp.read_group("MYC")       # AnnData, 1-2 shard reads
ref_adata = exp.read_reference()         # AnnData | None
labels    = exp.group_labels()           # list[str]

for gs in exp.iter_group_shards():       # streaming, ~one shard resident
    ad       = gs.to_anndata()
    label_ad = gs.read_group("MYC")      # per-label slice from this shard
```

Cloud: `pyscx.open_cloud(url).read_group("MYC")` etc. — same API, range reads.

**Staleness:** `pyscx.append` drops the group index. Re-sort with `pyscx.sort`
to restore grouping after mutations.

**Relation to training loaders:** grouped sharding is the physical layout
optimisation that makes random-access patterns efficient — paired (perturbed,
control) reads hit the shard LRU cache when groups are co-located.
`SparseCellSetDataset` benefits most: without grouping a perturbation batch
scatters across 30–50 shards (large `cache_shards` + high RSS); with grouping
each perturbation is in 1–2 shards.

## Tokenisation kernels (`pyscx.tokenize`)

The per-cell numeric steps a transformer-class tokeniser is built from, over a
gathered CSR batch. Use these instead of a Python loop over cells:

```python
import pyscx, pyscx.tokenize as tok

ds = pyscx.SparseCellSetDataset(["atlas.scx"])
b = ds.gather(file_ids, rows, role_tags, set_offsets)
ip, ix, dt = b["indptr"], b["indices"], b["data"]

tok.top_k(ip, ix, dt, k=2048, n_genes_total=n_genes)        # fixed-length crop
tok.rank_tokens(ip, ix, dt, gene_stats, 2048, "corpus-v2")  # Geneformer-style
tok.bin_values(ip, ix, dt, n_bins=51)                       # scGPT-style
tok.sample_genes(ip, ix, dt, n=1024, seed=0,                # UCE-style
                 file_identity=pyscx.downsample_file_identity(path),
                 rows=b["cell_indices"])
```

Each takes the whole batch, so the row loop runs in Rust with the GIL released,
and returns numpy arrays moved rather than copied. Assert
`pyscx.tokenize.CONTRACT_VERSION` at setup.

**Read `docs/tokenize.md` before wiring one into a training run.** Three of the
reference tokenisers are stochastic or underdetermined — scGPT randomises at bin
edges from numpy's global RNG, Geneformer's tie order is `np.argsort`'s unstable
default, UCE samples *with* replacement — so these kernels diverge from them in
stated ways rather than silently. Two gotchas that bite first:

- Pass real row ids in `rows` and the file's `downsample_file_identity`, or the
  seeded kernels key on batch position and are reproducible only for that batch.
- `library_size` sums the row **as given**. On a panel-projected batch that is
  not the cell's sequencing depth, and a model normalised against whole-cell
  depth must not be fed it.

## Neighbourhood plans (spatial / graph context)

A neighbourhood is a cell set with the centre role-tagged, so no new gather is
needed — only a plan. Build one from whichever relationship the file already
stores:

```python
import pyscx

exp = pyscx.open("tissue.scx")

# From a stored obsp graph. `weight_order` says which end `k` keeps:
# "desc" for an affinity (connectivities), "asc" for a distance graph.
plans, centers = pyscx.neighborhood_plans_from_graph(
    exp, "connectivities", file_id=0, k=8, weight_order="desc")

# Or straight from obsm["spatial"], with no graph and no index
plans, centers = pyscx.neighborhood_plans_from_coords(exp, file_id=0, k=8)
plans, centers = pyscx.neighborhood_plans_from_coords(exp, file_id=0, radius=50.0)

ds = pyscx.SparseCellSetDataset(["tissue.scx"])
for batch in ds.iter_with_plans(pyscx.batch_plans(plans, sets_per_batch=64)):
    ...   # role_tags: 0 for each set's centre, 1 for its neighbours
```

The plans feed `gather` / `iter_with_plans` and the tokenisation kernels above
unchanged. Four things that bite first:

- **`file_id` is a manifest position, not a file identity** — the index of this
  file in the `SparseCellSetDataset` you will gather with. Get it wrong and you
  gather a different file's rows with nothing raising, which is why it is
  required rather than defaulting to 0.
- **`weight_order` is not inferred from the key's name.** `k=8` with the
  default `"desc"` on `obsp["distances"]` returns each cell's eight *farthest*
  neighbours. Use `"asc"` for a distance graph.
- **The coordinate builder takes 1-D, 2-D or 3-D only.** `obsm_key="X_pca"` on
  a wide embedding is refused; write a kNN into `obsp` and use the graph
  builder instead.
- **Rows are physical, and deletions are dropped.** A deleted centre yields no
  set at all (so `centers` is shorter than `n_obs` and tells you which survived)
  and a deleted neighbour is dropped from every set rather than backfilled.
- **Layout, not the builder, decides what this costs.** A converted spatial file
  is usually in barcode order, where a 7-cell neighbourhood is scattered across
  the whole row axis and every batch touches every shard. `scx sort` on a key
  that tracks position is the lever; measure before and after.
- **`radius` and `k` are mutually exclusive** on the coordinate builder, and
  passing neither or both raises rather than picking one.

`Experiment.read_obsp_rows(key, start, stop)` is the bounded graph read the
builder uses, and is public — use it instead of `to_anndata().obsp[k]`, which
materialises the whole matrix. It is only bounded on a *sharded* graph; `scx
sort` re-emits obsp unsharded.

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
