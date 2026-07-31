# ML Training Data Loading

SCX replaces Python-side `DataLoader(num_workers=N)` + `sc.read_h5ad()` +
`.toarray()` with a **triple-buffered Rust pipeline** that streams shards
from disk, decodes, normalizes, and batches entirely in compiled code.
Zero Python on the hot path — your GPU stays fed.

| | SCX | AnnData | TileDB-SOMA-ML | scDataLoader |
|--|-----|---------|----------------|--------------|
| 1M cells (batches/sec) | **1,405** | 16.3 | 17.1 | 4.4 |
| vs SCX | — | 86× slower | 82× slower | 319× slower |

> _batch_size=1024, HVG=2000, normalize+log1p. See
> [performance.md](performance.md#training-loader) for the full suite._

Three dataset types cover different ML patterns:

- **[`TrainingDataset`](#trainingdataset)** — sequential streaming for standard
  training loops (autoencoder, scVI, Geneformer). Highest throughput.
- **[`IndexPlanDataset`](#indexplandataset)** — paired `(perturbed, control)` cell
  reads for perturbation training, contrastive learning, donor-matched designs.
- **[`MultimodalTrainingDataset`](#multimodaltrainingdataset)** — cell-aligned
  multi-assay batches (RNA + ADT + ATAC) from a single multimodal SCX file.


## Converting your data to SCX

```bash
# One-time conversion from h5ad (streaming is the default)
scx convert experiment.h5ad experiment.scx --index-preset training

# Or from Python
import pyscx
pyscx.from_h5ad("experiment.h5ad", "experiment.scx", index_preset="training")

# Merge multiple h5ad files into one SCX (eliminates multi-file management)
scx merge batch1.scx batch2.scx batch3.scx --output atlas.scx
```

`--index-preset training` materialises predicate indexes on `cell_type`,
`donor`, `batch`, `dataset_id`, `split`, `organism`, and `tissue` — enabling
fast query-based train/val splitting later.

For perturbation screens, add `--group-by target_gene --reference non-targeting`
to co-locate each perturbation's cells into contiguous shards — see
[Grouped sharding for perturbation training](#grouped-sharding-for-perturbation-training).


## Computing HVG indices

Most ML models train on a subset of highly variable genes. Compute HVG
indices once and pass them to the training dataset:

```python
import pyscx
from pyscx import accel
import numpy as np

# Load data (backed mode — doesn't materialize X)
exp = pyscx.open("atlas.scx")
adata = exp.to_anndata(backed=True)

# Compute HVGs on raw counts (stash counts layer if needed)
accel.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                            layer="counts")

# Extract HVG indices as uint32 for the loader
hvg_indices = np.where(adata.var["highly_variable"])[0].astype(np.uint32)

# Save for reuse across training runs
np.save("hvg_indices.npy", hvg_indices)
```


## Grouped sharding for perturbation training

Perturbation screens have a natural group axis — `target_gene`. A typical
CRISPR screen has ~5 k perturbations spread across ~120 shards. Without
grouping, reading one perturbation's cells touches nearly every shard
(scatter read). **Grouped sharding** physically co-locates each
perturbation's cells into 1–2 contiguous shards, turning a ~120-shard
scatter into a 1-shard sequential read (~120× I/O reduction).

### Creating a grouped SCX file

```python
import pyscx

# One-pass: convert and group in a single step
pyscx.from_h5ad(
    "screen.h5ad", "screen.scx",
    group_by="target_gene",
    reference=["non-targeting"],   # isolate control cells in shard 0
)
```

```bash
# CLI equivalent
scx convert screen.h5ad screen.scx \
    --group-by target_gene --reference non-targeting

# Two-pass fallback: convert first, then sort
scx convert screen.h5ad screen_unsorted.scx
scx sort screen_unsorted.scx screen.scx \
    --group-by target_gene --reference non-targeting
```

Rows are sorted so reference (control) cells come first (shard 0), then by
group label. Groups are bin-packed into shards cutting only at group
boundaries — a group never straddles a shard boundary.

### Reading per-perturbation data

```python
exp = pyscx.open("screen.scx")

# Single perturbation — touches only 1–2 shards
adata = exp.read_group("MYC")

# Control cells (shard 0)
ref = exp.read_reference()           # -> AnnData | None

# All group labels
labels = exp.group_labels()          # -> list[str]

# Streaming iteration — ~one shard resident at a time
for gs in exp.iter_group_shards():
    ad = gs.to_anndata()
    label_ad = gs.read_group("MYC")  # slice one label within this shard
```

### Connection to IndexPlanDataset

Grouped sharding and `IndexPlanDataset` are complementary. Grouped sharding
optimises the **physical layout** (which cells share a shard); `IndexPlanDataset`
handles the **logical pairing** (perturbed, control) batch reads via plan-driven
I/O. For perturbation training pipelines (State, CPA, GEARS) the recommended
workflow is:

1. Convert with `group_by="target_gene", reference=["non-targeting"]`.
2. Use `read_group` / `read_reference` for analysis and QC.
3. Use `TrainingDataset`, `IndexPlanDataset`, or `SparseCellSetDataset` for
   training.

Grouped sharding also reduces **shard cache pressure** in random-access loaders
like `SparseCellSetDataset`. Without grouping, a perturbation batch scatters
across 30–50 shards, requiring a large `cache_shards` setting and ~20+ GB RSS
to avoid thrashing. With grouping each perturbation is in 1–2 shards, so the
cache stays warm with far fewer entries.

### Sizing the shard cache

Don't guess `cache_shards` — measure what your plans touch:

```python
probe = pyscx.SparseCellSetDataset(paths)
need = max(probe.suggested_cache_shards(fids, rows) for fids, rows, _, _ in plans[:64])
ds = pyscx.SparseCellSetDataset(paths, cache_shards=need)
```

`suggested_cache_shards` counts the distinct `(file_id, shard)` pairs a plan
touches, from the catalog's shard row ranges — no I/O and no decode.
`IndexPlanDataset.suggested_cache_shards(plan)` is the paired-plan equivalent.

Two things to know:

- **Raise the byte budget too.** `cache_shards` is a count cap; `max_memory_mb`
  is a byte cap, and both are enforced. A file with large shards can want more
  bytes than the adaptive default affords (census_500k wants 31 shards while the
  4 GB adaptive cap holds 22), so `cache_shards=31` alone under-delivers.
  `memory_budget()` reports `affordable_cache_shards` so you can see it.
- **On framed files (the v4 default) this mostly doesn't matter.** A scattered
  gather routes through the row-group block-index path, which decodes only the
  touched row-groups and never populates the whole-shard LRU — so `cache_shards`
  is not on the critical path at all. It becomes load-bearing on unframed/legacy
  layouts and when you pass `scatter_block_index=False`, where sizing the cache
  correctly is worth **2,486×** and halves peak RSS (cold capture; an earlier warm
  probe of the same comparison gave 269× — see
  [performance.md § Shard-cache sizing](performance.md#shard-cache-sizing-on-the-gather-path-data-load-phase-1-1a)).

Both gather loaders sample their cache counters while iterating and emit a
one-shot `UserWarning` if the observed miss/eviction pattern indicates the
working set exceeds the cache. `max_memory_mb=None` resolves to a **bounded**
adaptive budget on all three loader classes (512 MB floor → 4 GB cap), so peak
RSS is capped by default rather than open-ended.

> [!TIP]
> On cloud storage (S3 / GCS), each shard access is a separate HTTP
> range-read request. Grouped sharding reduces ~120 round-trips per
> perturbation to 1–2 — a significant latency win on high-latency backends.

See [api.md § Grouped reads](api.md#grouped-reads-conditionlabel-grouped-sharding) and
[sharding.md § Grouped sharding](sharding.md#conditionlabel-grouped-sharding-f1--grouped-reads-f2)
for the full API reference and on-disk layout details.


## TrainingDataset

Sequential streaming dataset for standard training loops. Each
`for batch in dataset:` is one epoch with deterministic reshuffling.

```python
import pyscx
import torch
import numpy as np

hvg_indices = np.load("hvg_indices.npy")

dataset = pyscx.TrainingDataset(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_indices,
    obs_columns=["cell_type", "batch"],
    normalize=True,          # normalize and log1p are applied in Rust
    log1p=True,              # (all 4 True/False combinations are valid)
    target_sum=1e4,
    seed=42,                 # deterministic shuffle
)

print(dataset)
# TrainingDataset(n_obs=1000000, n_vars=33694, n_output_genes=2000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(10):
    for batch in dataset:       # each loop = one epoch, reshuffled
        x = torch.from_numpy(batch["X"]).to(device)  # [B, 2000] float32
        cell_types = batch["obs"]["cell_type"]        # {"codes": int32, "categories": [...]}
        indices = batch["cell_indices"]               # [B] int64

        # Your training step:
        # logits = model(x)
        # loss = criterion(logits, labels)
        # loss.backward()
        # optimizer.step()

dataset.close()  # release Rust threads before process exit
```

### Why `num_workers=0`

SCX's Rust pipeline already manages I/O, decode, and prefetch threads
internally. Adding PyTorch `DataLoader` workers on top would double-buffer
without benefit and risk fork-safety issues. Use:

```python
# Wrap in DataLoader only if your training framework requires it
loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
```

### Constructor reference

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `batch_size` | `1024` | Mini-batch size. Auto-tuned downward if `max_memory_mb` is exceeded; check via `effective_batch_size`. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. |
| `obs_columns` | `None` | Obs metadata column names included in each batch dict. `None` = no obs columns. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in a single CSR row scan). The per-cell depth is the **full transcriptome** total count even under `hvg_indices` projection (matching scanpy's normalize-then-subset), not the panel-local sum. **scVI and other count-likelihood models need `normalize=False, log1p=False`.** |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target sum. |
| `pflog` | `False` | Apply PFlog (v4) / shifted-log normalization on raw counts (Booeshaghi et al.) instead of normalize/log1p. Mutually exclusive with `normalize`/`log1p` (takes precedence when `True`). The centering denominator is computed over the full transcriptome even under `hvg_indices` projection (no per-cell depth in v4). |
| `pflog_alpha` | `None` | PFlog NB overdispersion `α` (matrix-wide pseudocount `1/(4α)`; only used when `pflog=True`). `None` estimates `α` once at loader construction from the raw counts (single-modality only; a modality-scoped loader must pin it); a float pins a reference `α`. |
| `shard_group_size` | `8` | Shards per I/O group. Sequential I/O within each group for disk efficiency. |
| `prefetch_batches` | `4` | Ring buffer depth — number of pre-built batches to buffer ahead. |
| `seed` | `42` | RNG seed. Deterministic shuffle via `(seed, epoch_number)`. |
| `max_memory_mb` | adaptive | When omitted, the budget is adaptive: scales up to fit the file's configuration (floor 512 MB, cap 4096 MB) so a full-width ~33k-gene file keeps its requested `batch_size`. Pass an explicit value to pin a hard ceiling — the pipeline then auto-tunes `shard_group_size`, `prefetch_batches`, and `batch_size` down to fit. |
| `modality` | `None` | For multimodal v2 files: name of the modality to load (e.g. `"rna"`). Ignored on single-modality files. |

### Batch dict schema

```python
{
    "X":            np.ndarray[B, n_output_genes, float32],   # dense expression
    "obs":          dict[str, np.ndarray | {"codes", "categories"}],
    "cell_indices": np.ndarray[B, int64],   # global row indices
}
```

- **Categorical** obs columns: `{"codes": ndarray[int32], "categories": list[str]}`
- **Numeric** obs columns: `ndarray[int64]` or `ndarray[float64]`
- **`cell_indices`**: global row indices into the SCX file — useful for
  looking up metadata or matching predictions to cells.

### Epoch and shuffling

Each `for batch in dataset:` loop is one epoch. On each epoch:

1. **Level 1 (shard order)**: Shard indices are randomly permuted using a
   deterministic RNG seeded from `(seed, epoch_number)`.
2. **Level 2 (row shuffle)**: Within each shard group, cell indices are
   Fisher-Yates shuffled and sliced into batches.

This two-level shuffle provides training randomization without random I/O.
The same `seed` always produces the identical ordering for reproducibility.


## IndexPlanDataset

Plan-driven paired-batch reader for ML workloads where each batch is a list
of `(perturbed_cell, control_cell)` pairs. Use this for perturbation training
(GEARS, CPA, scGEN), contrastive learning, or donor-matched designs.

```python
import pyscx
import numpy as np
from collections.abc import Iterator

# Build pairing plans from your strategy
def plan_generator(
    perturbed_indices: list[int],
    control_map,
    batch_size: int,
) -> Iterator[list[tuple[int, int]]]:
    """Yield batches of (perturbed_idx, control_idx) pairs."""
    buf: list[tuple[int, int]] = []
    for p_idx in perturbed_indices:
        c_idx = control_map.get_control(p_idx)
        if c_idx is not None:
            buf.append((p_idx, c_idx))
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

ds = pyscx.IndexPlanDataset(
    "atlas.scx",
    hvg_indices=hvg_indices,
    obs_columns=["cell_type", "perturbation"],
    normalize=True,
    target_sum=1e4,
    cache_shards=128,       # LRU shard cache for random access
    lookahead=4,            # prefetch shards for upcoming plans
)

for batch in ds.iter_with_plans(plan_generator(perturbed, controls, 1024)):
    X_pert = torch.from_numpy(batch["X"]).to(device)
    X_ctrl = torch.from_numpy(batch["X_paired"]).to(device)
    pairs = batch["pairs"]  # post-sort plan — pairs[i] ↔ X[i], X_paired[i]
    # ... training step ...
```

`IndexPlanDataset` is **106× faster** than Python-loop `ScxBackedSparseDataset`
access (20K vs 189 cells/s). See [api.md § IndexPlanDataset](api.md#indexplandataset)
for the full constructor reference, batch schema, and iterator semantics.


## MultimodalTrainingDataset

For CITE-seq, 10x Multiome, or TEA-seq data stored in a single multimodal
SCX file:

```python
ds = pyscx.MultimodalTrainingDataset(
    "citeseq.scx",
    modalities=["rna", "adt"],
    batch_size=1024,
    normalize=True,
    log1p=True,
    obs_columns=["cell_type"],
)

for batch in ds:
    x_rna = torch.from_numpy(batch["X"]["rna"]).to(device)
    x_adt = torch.from_numpy(batch["X"]["adt"]).to(device)
    # Cell axes are aligned: x_rna[i] and x_adt[i] are the same cell
```

> [!NOTE]
> `MultimodalTrainingDataset` encodes categorical obs columns as decoded
> `list[str]` values, **not** the `{"codes", "categories"}` dict format used
> by `TrainingDataset`. `n_vars` returns a `dict[str, int]` mapping modality
> names to per-modality variable counts. With `return_dict=False`, batches
> are tuples of X arrays only (no obs or cell_indices).

See [multimodal.md](multimodal.md) for the full API.


## Train/val/test splits

### Option A: Query-based splitting (recommended)

If your SCX file has a `split` column in obs metadata (or any other
split indicator), use the query engine to create separate datasets:

```python
import pyscx

# Query for train and val splits
exp = pyscx.open("atlas.scx")

train_result = exp.query().filter_obs("split == 'train'").collect()
train_adata = train_result.to_anndata()
pyscx.from_anndata(train_adata, "train.scx")

val_result = exp.query().filter_obs("split == 'val'").collect()
val_adata = val_result.to_anndata()
pyscx.from_anndata(val_adata, "val.scx")

# Create separate datasets
train_ds = pyscx.TrainingDataset("train.scx", batch_size=1024, ...)
val_ds = pyscx.TrainingDataset("val.scx", batch_size=1024, ...)
```

### Option B: Index-based splitting

For random splits, compute indices and use `cell_indices` to filter:

```python
import numpy as np

exp = pyscx.open("atlas.scx")
n_obs = exp.n_obs
indices = np.random.permutation(n_obs)
train_idx = set(indices[:int(0.8 * n_obs)])

dataset = pyscx.TrainingDataset("atlas.scx", batch_size=1024, ...)
for batch in dataset:
    mask = np.isin(batch["cell_indices"], list(train_idx))
    x_train = batch["X"][mask]
    # ... use x_train for training
```

### Option C: Separate files at conversion time

```python
import anndata as ad

adata = ad.read_h5ad("experiment.h5ad")
train_adata = adata[adata.obs["split"] == "train"]
val_adata = adata[adata.obs["split"] == "val"]

pyscx.from_anndata(train_adata, "train.scx")
pyscx.from_anndata(val_adata, "val.scx")
```


## PyTorch Lightning integration

### scVI with ScxDataModule

SCX includes a built-in Lightning DataModule for scVI:

```python
from pyscx.scx_integrations.scvi import ScxDataModule

dm = ScxDataModule(
    "atlas.scx",
    batch_size=1024,
    hvg_indices=hvg_indices,
    normalize=False,      # scVI requires raw counts — disable normalize + log1p
    log1p=False,
)

# dm wraps TrainingDataset internally — num_workers=0, Rust handles threading
# dm exposes n_obs, n_vars, n_output_genes properties
print(f"Dataset: {dm.n_obs} cells, {dm.n_output_genes} genes")
```

> [!WARNING]
> `ScxDataModule` inherits `normalize=True` and `log1p=True` from
> `TrainingDataset`. **scVI and other count-likelihood models require raw
> integer counts** — always pass `normalize=False, log1p=False` as shown
> above.

### Generic Lightning DataModule

For models other than scVI, wrap `TrainingDataset` in a
`LightningDataModule`:

```python
import pyscx
import lightning as L
import torch
from torch.utils.data import DataLoader

class ScxDataModule(L.LightningDataModule):
    def __init__(self, train_path, val_path=None, batch_size=1024,
                 hvg_indices=None, **kwargs):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.batch_size = batch_size
        self.hvg_indices = hvg_indices
        self.kwargs = kwargs

    def train_dataloader(self):
        ds = pyscx.TrainingDataset(
            self.train_path,
            batch_size=self.batch_size,
            hvg_indices=self.hvg_indices,
            **self.kwargs,
        )
        # batch_size=None because SCX handles batching; num_workers=0
        return DataLoader(ds, batch_size=None, num_workers=0)

    def val_dataloader(self):
        if self.val_path is None:
            return None
        ds = pyscx.TrainingDataset(
            self.val_path,
            batch_size=self.batch_size,
            hvg_indices=self.hvg_indices,
            **self.kwargs,
        )
        return DataLoader(ds, batch_size=None, num_workers=0)
```


## PyTorch DataLoader with `num_workers > 0`

If you must use `num_workers > 0` (e.g., framework requirement), construct
the dataset **lazily inside the worker's `__iter__`**:

```python
import torch.utils.data as data

class ScxIterableDataset(data.IterableDataset):
    def __init__(self, scx_path, **kwargs):
        self.scx_path = scx_path
        self.kwargs = kwargs  # paths only — no inner dataset yet

    def __iter__(self):
        # Construct the dataset HERE (in the worker process, post-fork)
        ds = pyscx.TrainingDataset(self.scx_path, **self.kwargs)
        try:
            yield from ds
        finally:
            ds.close()  # release rayon pool / tokio runtime

loader = torch.utils.data.DataLoader(
    ScxIterableDataset("atlas.scx", batch_size=1024, hvg_indices=hvg_indices),
    batch_size=None,
    num_workers=2,
    persistent_workers=False,
)
```

**Don't** construct a `TrainingDataset` in the parent process and share it
across forked workers — the PID check in `__next__` raises `RuntimeError`.

> [!TIP]
> Use `multiprocessing.set_start_method("spawn")` if your workload allows.
> Spawn re-execs Python in the child and eliminates fork hazards entirely.


## Migrating from h5ad-based training

### Before (typical h5ad pattern)

```python
import scanpy as sc
import torch
from torch.utils.data import DataLoader, Dataset

# Slow: full file load into memory
adata = sc.read_h5ad("experiment.h5ad")            # ← slow, high memory

# In-memory preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)         # ← doubles memory
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable].copy()    # ← copies again

# Dense conversion — triples memory
X = torch.from_numpy(adata.X.toarray().astype(np.float32))

class MyDataset(Dataset):
    def __init__(self, X, obs):
        self.X = X
        self.labels = torch.from_numpy(obs["cell_type"].cat.codes.values)

    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.labels[idx]

loader = DataLoader(MyDataset(X, adata.obs),
                    batch_size=1024, shuffle=True,
                    num_workers=4, pin_memory=True)
```

### After (SCX)

```python
import pyscx
import torch
import numpy as np

# One-time conversion (do once, not per run)
# pyscx.from_h5ad("experiment.h5ad", "experiment.scx")

# Compute HVGs once (or load saved indices)
hvg_indices = np.load("hvg_indices.npy")

# Streaming — no full file load, no .toarray(), no DataLoader workers
dataset = pyscx.TrainingDataset(
    "experiment.scx",
    batch_size=1024,
    hvg_indices=hvg_indices,        # HVG projection in Rust
    obs_columns=["cell_type"],      # metadata in each batch
    normalize=True,                 # normalize + log1p in Rust
    log1p=True,
)

for batch in dataset:
    x = torch.from_numpy(batch["X"]).to(device)
    cell_type = batch["obs"]["cell_type"]  # {"codes": int32, "categories": [...]}
    labels = torch.from_numpy(cell_type["codes"]).to(device)
    # ... training step ...

dataset.close()
```

### What you can remove

| h5ad pattern | SCX replacement | Notes |
|---|---|---|
| `sc.read_h5ad(path)` | `TrainingDataset(path)` | No full file load |
| `.toarray()` / `.todense()` | Automatic | SCX does sparse→dense in Rust |
| `sc.pp.normalize_total()` | `normalize=True` | Fused in Rust, one CSR scan |
| `sc.pp.log1p()` | `log1p=True` | Fused with normalize |
| `adata[:, hvg_mask].copy()` | `hvg_indices=` | Column projection in Rust |
| `torch.utils.data.Dataset` | Not needed | SCX is already iterable |
| `DataLoader(num_workers=4)` | `num_workers=0` | Rust manages I/O threads |
| `shuffle=True` | Automatic | Two-level shuffle per epoch |
| Custom h5py shims | Not needed | Single-file format, no h5py dependency |
| LRU file handle caches | Not needed | Single `.scx` file, mmap'd |
| `worker_init_fn` for fork safety | Not needed | Rust pipeline is self-contained |

### Multi-file to single-file

If your training code manages multiple h5ad files with per-file handle
caches and gene vocabulary alignment:

```bash
# Convert each h5ad to SCX, then merge into one file
for f in *.h5ad; do scx convert "$f" "${f%.h5ad}.scx"; done
scx merge *.scx --output atlas.scx

# Or merge directly from Python
pyscx.merge(["batch1.scx", "batch2.scx", "batch3.scx"], "atlas.scx")
```

One file = one `TrainingDataset` call. No multi-file management, no gene
index alignment, no catalog cold-start.


## `to_anndata(backed=True)` vs `TrainingDataset`

If you're already using `pyscx.open(...).to_anndata(backed=True)` for
training, you're using SCX for storage but still doing per-cell Python-side
reads through AnnData's interface. Switching to `TrainingDataset` gives you
**80–100× higher throughput** because:

| | `to_anndata(backed=True)` | `TrainingDataset` |
|---|---|---|
| Read pattern | Per-cell random access | Shard-sequential streaming |
| Decode | Python → Rust per cell | Bulk Rust decode (rayon) |
| Normalize | Python per batch | Fused in Rust |
| Threading | Python DataLoader workers | Triple-buffered Rust pipeline |
| Memory | One shard cached | Ring buffer of pre-built batches |

The backed AnnData path is designed for interactive analysis (random access
to arbitrary cells). `TrainingDataset` is designed for training (sequential
streaming with maximum throughput).


## Perturbation evaluation

After training, use SCX's Rust-accelerated perturbation metrics (numerically
equivalent to [cell-eval](https://github.com/arcinstitute/cell-eval) and
[arc-bench](https://github.com/arcinstitute/arc-bench)):

```python
from pyscx import accel

# Replaces cell-eval's pearson_delta + mse + mae + mse_delta + mae_delta
results = accel.perturbation_metrics(adata_real, adata_pred)

# Per-cell knockdown efficiency
accel.knockdown_efficiency(adata, pert_col="perturbation", control="control")

# Energy distance and clustering agreement
corr = accel.energy_distance(adata_real, adata_pred)
score = accel.clustering_agreement(adata_real, adata_pred, metric="ami")
```

See [scanpy.md § Perturbation evaluation metrics](scanpy.md#perturbation-evaluation-metrics-cell-eval--arc-bench-parity)
for the full API.


## Troubleshooting

### `RuntimeError: scx.TrainingDataset requires num_workers=0`

The dataset was constructed in the parent process and then used from a
forked DataLoader worker. Either:
- Set `num_workers=0` (recommended), or
- Use the [lazy `IterableDataset` wrapper](#pytorch-dataloader-with-num_workers--0)
  to construct the dataset inside each worker.

### `RuntimeError: Must call __iter__ before __next__`

You're calling `next(dataset)` without first calling `iter(dataset)`. Use
`for batch in dataset:` which handles this automatically.

### Memory budget exceeded

If you see `budget_exceeded: True` in `dataset.memory_budget()`, the
pipeline auto-tuned `shard_group_size`, `prefetch_batches`, and/or
`batch_size` downward. Check `effective_batch_size` and increase
`max_memory_mb` if needed:

```python
ds = pyscx.TrainingDataset("atlas.scx", max_memory_mb=1024, ...)
print(ds.memory_budget())
print(f"Effective batch size: {ds.effective_batch_size}")
```

### GIL and CUDA overlap

The GIL is released while waiting for the next batch from the Rust
pipeline, allowing PyTorch CUDA operations (e.g., GPU kernels from the
previous batch's backward pass) to run concurrently.
