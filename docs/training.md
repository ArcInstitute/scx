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
need = max(probe.suggested_cache_shards(p) for p in plans[:64])
ds = pyscx.SparseCellSetDataset(paths, cache_shards=need)
```

`suggested_cache_shards` counts the distinct `(file_id, shard)` pairs a plan
touches, from the catalog's shard row ranges — no I/O and no decode. Both
plan-driven classes take **one plan**, in whatever shape that class's
`iter_with_plans` consumes: `[(pert, ctrl), ...]` on `IndexPlanDataset`,
`(file_ids, rows, role_tags, set_offsets)` here (the last two are ignored, since
residency depends only on which rows of which files are read). So the line above
is the same on both.

Two things to know:

- **Raise the byte budget too.** `cache_shards` is a count cap; `max_memory_mb`
  is a byte budget, and both are enforced. A file with large shards can want more
  bytes than the adaptive default affords (census_500k wants 31 shards while the
  4 GB adaptive cap holds 22), so `cache_shards=31` alone under-delivers.
  `memory_budget()` reports `effective_cache_shards` so you can see it — the
  same key `IndexPlanDataset` uses, and the same
  `loader.effective_cache_shards()` behind both.
  Since ORG-9.10-5 the constant interpreter/numpy/Arrow overhead (~50 MB) is
  subtracted before the cache is sized, on every loader class — so
  `max_memory_mb` is no longer a bare cache cap. It is **not** a hard ceiling on
  process RSS either: on `SparseCellSetDataset` the batch's transients are never
  charged and the batch itself only if you pass `max_plan_rows` (below), and the
  LRU keeps one oversize shard rather than refusing to cache it. Sizing a budget
  by hand therefore means `cache_shards × shard_decoded_bytes + non-cache
  terms`; both are in `memory_budget()`, and the sizing `UserWarning` already
  quotes the total in its "pass `max_memory_mb>=…`" advice.
- **`lookahead` does not bound decode concurrency.** It bounds in-flight
  *plans*; each plan issues one blocking task per distinct `(file, shard)` it
  touches, which plan width controls. `memory_budget()["max_blocking_threads"]`
  reports the cap on how many of those run at once — on both plan-driven
  classes, since it lives on the shared prefetch engine.
- **Charge the batch to stop the budget ignoring it entirely.**
  `SparseCellSetDataset(paths, max_memory_mb=…, max_plan_rows=N)` costs one
  gathered CSR batch of `N` rows at the manifest's mean density and subtracts it
  before sizing the cache; `memory_budget()["breakdown"]["batch_buffer_bytes"]`
  reports it and `effective_cache_shards` falls accordingly, and a plan wider
  than `max_plan_rows` is **refused** — a cache sized for N rows while the
  gather accepts 100N would not be a bound at all. What it bounds is the plan's
  **row count**, not its bytes: the charge uses the manifest's mean density, so
  a plan of denser-than-average rows can still exceed the charged figure. Treat
  `batch_buffer_bytes` as a sized estimate that the row count keeps honest, not
  as a ceiling on process RSS. It is **opt-in** and defaults to uncharged and
  unenforced: this class has no `max_plan_size`, so plan width is
  yours to declare, and a default guess would shrink the cache — the lever worth
  2,486× below — on every existing caller.
- **Bound the manifest itself when it is very large.**
  `SparseCellSetDataset(paths, reader_limit=N)` keeps at most `N` of the
  manifest's files open at a time, reopening on demand. Default `None` opens
  every file and never closes one, which is what this class has always done.
  See [Very large manifests](#very-large-manifests) — the resource it bounds is
  **resident memory**, not file descriptors, and none of it is charged to
  `max_memory_mb`.
- **On `SparseCellSetDataset` this is load-bearing by default.** The class
  defaults `scatter_block_index=False` (and no route wins everywhere — the two
  cross near 1M cells, see [Cell-set scatter routes](performance.md#cell-set-scatter-routes-re-measured-after-the-row-group-lru-phase-0-gate)),
  so a scattered gather decodes whole
  shards into the LRU and serves reuse from cache — sizing it correctly is worth
  **2,486×** and halves peak RSS (cold capture; an earlier warm probe of the same
  comparison gave 269× — see
  [performance.md § Shard-cache sizing](performance.md#shard-cache-sizing-on-the-gather-path-data-load-phase-1-1a)).
  Pass `scatter_block_index=True` and the picture changes: on a framed (v4) file
  the gather then routes through the row-group block-index path, which decodes
  only the touched row-groups and never inserts a whole shard, so
  `cache_shards` as a *count* drops off the critical path. The touched groups
  are retained in the same LRU under the same byte budget (`max_memory_mb`),
  reported as `cache_metrics()["row_group_hits"]` / `["row_group_misses"]`, so
  the budget still decides how much of a hot region the next batch gets for
  free — before that retention existed the row-group path re-decoded every
  group every batch. `IndexPlanDataset` defaults the other way, so that is its
  normal regime rather than its opt-in.

Both gather loaders sample their cache counters while iterating and emit a
one-shot `UserWarning` if the observed miss/eviction pattern indicates the
working set exceeds the cache. `max_memory_mb=None` resolves to a **bounded**
adaptive budget on all three loader classes (512 MB floor → 4 GB cap), so the
budget is bounded by default rather than open-ended — a bound on what the
loaders *size themselves to*, not a hard cap on process RSS.

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
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. Every index must be `< n_vars` — an out-of-range index is rejected at construction, because it matches no column and would otherwise yield an output feature that is silently always zero. **The panel is sorted and deduplicated**, so batch columns are in ascending gene-index order regardless of the order you pass, and duplicates shrink the batch width (check `n_output_genes`). Passing a panel that is not already ascending-unique emits a `UserWarning`; `np.unique(hvg_indices)` reproduces the column order the batches use. |
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

> [!NOTE]
> **The `seed` → ordering mapping changed in v0.13.1.** The two levels used to
> compose `seed` and `epoch` by addition, which collides: `(seed, epoch)` and
> `(seed + φ, epoch − 1)` drove the *same* stream, so a seed sweep over
> `s, s + φ, s + 2φ` silently replayed orderings from neighbouring epochs, and
> the row shuffle at seed `s` was identical to the shard shuffle at
> `s + 0xDEADBEEF`. Both levels now chain their components through SplitMix64
> with distinct domain tags, matching what `downsample` and `scx sort --shuffle`
> already did.
>
> Determinism is unchanged — the same seed still gives the same ordering within
> a version — but a given seed produces a *different* ordering than it did
> before. Files written by `scx sort --shuffle` are unaffected; that permutation
> is on disk and its derivation did not change.

#### When two levels aren't enough: pre-shuffle the file

Both levels are bounded by **physical layout**. Level 2 only mixes rows that
already share a shard group, so if your file arrived clustered — cells grouped
by donor, plate, or cell type, which is the normal case for a concatenated
atlas — every batch is drawn from a handful of adjacent shards and inherits
their composition. Raising `shard_group_size` widens the pool but costs memory
linearly.

The fix is to pay once, on disk:

```bash
scx sort --shuffle --seed 42 atlas.scx atlas.train.scx
```
```python
pyscx.shuffle("atlas.scx", "atlas.train.scx", seed=42)
```

After that the row order carries no residual structure, so even
`shard_group_size=1` gives batches that look like the corpus, and the loader's
cheap level-1 shard permutation is all the per-epoch randomization you need.

Two things to know before you run it: the output is the **inverse** of a sorted
file for query purposes (it maximally scatters predicate-index shard ranges),
and a permutation inherently costs some cross-row redundancy for codecs whose
compression spans rows (~6–12% for `zstd`, under 1% for `lz4`/`shufdelta`).
Leave `--codec` at `auto`: it runs the same adaptive per-shard selection
`scx convert` does. (This used to say to pass the input's own codec because
`auto` grew the file ~2× — that was a bug in every derived-file op, since fixed.)
Pass `--shard-size <input's value>` if the input is not at the 16,384 default, so
the rewrite reorders without also re-sharding. Both are
covered in
[sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).


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

### Feedback and curriculum plan generators

`lookahead` is a prefetch *budget*, not a contract the generator has to meet.
The loader waits for the first plan of each batch and then tops its queue up
with whatever the generator has already produced, so a generator that computes
plan *i+1* from batch *i* — curriculum sampling, hard-negative mining, anything
with a feedback signal — is supported directly:

Two properties of the loader shape the code you have to write:

1. It advances the generator with `next()`, never `.send()` — so
   `batch = yield plan` binds `None`, not the batch.
2. The plan-pull thread does **not** wait for the consumer. After `yield`, it
   sends the plan and immediately asks for the next one, up to the channel's
   capacity — so reading a shared variable straight after `yield` reads the
   *previous* iteration's value, or `None` on the first.

So the feedback has to be an explicit rendezvous: publish the batch, then
signal, and have the generator block on that signal before answering.

```python
import threading

published = threading.Semaphore(0)
state = {"batch": None}

def curriculum(model):
    plan = initial_plan()
    while True:
        yield plan
        # Block until the consumer has published the batch for the plan just
        # yielded. Without this the loader asks for the next plan immediately
        # and `state["batch"]` is still the previous one.
        published.acquire()
        plan = next_plan_from(model, state["batch"])

for batch in ds.iter_with_plans(curriculum(model)):
    state["batch"] = batch     # publish first…
    published.release()        # …then signal
    ...
```

`pyscx/tests/test_index_plan_dataset.py::TestFeedbackPlanGenerator` is the
executable version of this.

Such a generator simply runs un-prefetched (effectively `lookahead=0`) while it
is the bottleneck, and regains depth whenever it runs ahead. It is never
required to stay `lookahead` plans in front of the consumer.


## SparseCellSetDataset and the native collation kernel

`SparseCellSetDataset` gathers multi-file, role-tagged **cell sets** as sparse
CSR — the shape set-transformer models over cell populations consume. Pass
per-file `remap_tables` to have the gather emit global-vocab CSR, in which case
it owns the whole per-row finalisation: gene remap, dropping unmapped genes,
sorting, coalescing duplicate mappings, the non-negativity clip, and — when
configured — a seeded count downsample.

`pyscx.collate_cellset_gathered` then turns a gathered batch into stacked
encoder/target tensors: per-cell preprocessing (`pass_through` / `log1p_raw` /
`pflog_raw` / `normalize_log1p`), the top-K encoder crop, drop-to-PAD encoder
masking, the target gather at query positions, and `library_size`. The kernel is
RNG-free by design — the decoder query gene ids and per-cell masks are supplied
by the caller — which is what makes it cheap to hold byte-exact against a Python
reference implementation.

`pyscx.COLLATE_CELLSET_CONTRACT_VERSION` (currently `2`) pins that contract:
the crop/mask/target semantics, the accepted preprocess-mode strings, and the
gather stage's value contract. A consumer mirroring the kernel should assert it
at setup so version skew fails loudly rather than mid-training.

Encoder masking is the kernel's one non-obvious cost, and it is now paid per
*set* rather than per cell: the decoder query panel is sorted once for the set
that shares it, and each row's withheld genes are found by binary-searching that
panel and testing the row's own mask bits. The panel is not deduplicated, so a
gene id appearing at several query positions is withheld when **any** of them is
flagged. Output is unchanged — this is a throughput change, not a contract one.

### One plan at a time

`iter_with_plans` is for driving an epoch. For a single plan — an interactive
probe, a unit test, a neighbourhood lookup — `gather` returns the same batch
dict synchronously:

```python
batch = ds.gather(*plan)          # (file_ids, rows, role_tags, set_offsets)
```

The two differ only in row-group admission: the iterator takes one verdict
per plan against its divided share (`budget / (lookahead + 1)`), `gather` decides once over the plan
against the whole byte budget. That changes what the shard cache *retains*, not
what is read, so the batches are identical — but `gather` runs on the calling
thread with no prefetch, so it is not the way to stream.

### Very large manifests

`SparseCellSetDataset` takes a list of paths and, by default, opens all of them
in the constructor and holds them for the dataset's lifetime. Manifests in this
regime reach tens of thousands of files, so it is worth being precise about what
that costs — the intuitive answer is wrong.

**It is not file descriptors.** Opening an SCX file mmaps it and closes the
descriptor; the loader's readers do not watch their files, so they retain none.
Measured on this constructor: a 5,000-file manifest constructs *and* gathers
under `ulimit -n 1024` with the process's descriptor count unchanged.

**It is resident memory**, almost all of it the parsed catalog — one owned entry
per catalog entry, so it scales with shards per file rather than cells:

| manifest | per open reader | 26,453 files |
|---|---|---|
| `tabula_sapiens_100k` | ~106 kB | ~2.8 GB |
| `census_1m` | ~121 kB | ~3.2 GB |

Per process, before multiplying by DataLoader workers and ranks.

`reader_limit=N` caps how many readers are resident; the rest are reopened on
demand, and what a vacated slot keeps is the path, the row count, the shard
index and the file's identity — a few kB, not a hundred.

```python
ds = pyscx.SparseCellSetDataset(paths, reader_limit=64)
m = ds.cache_metrics()
m["reader_hwm"]        # peak simultaneously-open readers
m["reader_opens"]      # > len(paths) once plans revisit evicted files
m["reader_evictions"]
```

Four things to know before setting it:

- **A reopen re-parses the catalog** (0.09–20 ms per file). That is not an
  oversight — the catalog is over 90 % of what the eviction reclaims, so
  retaining it to make reopens cheap would make the whole thing pointless.
  Match `reader_limit` to how many files your plans actually touch together;
  a manifest of 26k files whose plans each touch three is the good case, and a
  plan that fans across thousands of files every batch is the bad one.
- **It bounds handles the loader is free to drop, not handles in existence.** A
  plan that needs more files at once than the limit exceeds it rather than
  blocking — blocking would deadlock against a caller already holding readers
  from the same plan. `reader_hwm` is what reports the truth.
- **A file replaced at its path between gathers is refused, not served.** A
  reopen compares the file's inode and header catalog pointer against what the
  constructor scanned and raises if either moved, because the decoded-shard
  cache is keyed by manifest position with no notion of file generation. At the
  default `reader_limit=None` nothing is ever reopened, so nothing checks, and
  the behaviour is exactly what it was before this option existed.
- **None of it is charged to `max_memory_mb`.** That budget sizes the decoded
  shard cache; an open reader's catalog is not one of its terms.
  `memory_budget()["reader_limit"]` reports the cap, deliberately outside
  `breakdown`, rather than pretending the readers are priced.

`IndexPlanDataset` takes a single path and has no manifest, so it has no
`reader_limit` and its `cache_metrics()` carries no `reader_*` keys.

### Count-depth downsampling

Downsampling every cell to a common library size is a standard depth
augmentation. It runs **inside the gather**, not in the collator:

```python
ds = pyscx.SparseCellSetDataset(
    paths,
    remap_tables=remap_tables,
    downsample_target_library_size=2000,
    downsample_method="multinomial",   # or "binomial"
    downsample_seed=42,
)
```

| Argument | Default | Meaning |
|---|---|---|
| `downsample_target_library_size` | `None` (off) | Counts per cell to keep. Cells already at or below it are not sampled. |
| `downsample_method` | `"multinomial"` | `"multinomial"` hits the target **exactly**; `"binomial"` hits it in expectation (each gene drawn independently at `p = target / library_size`). |
| `downsample_seed` | `0` | Seeds the per-row draw. |

Passing a method or seed without a target is an error, not a silent no-op.

Three properties worth knowing:

- **The draw is keyed on `(seed, method, resolved file path, row)`**, not on
  manifest position. So reordering `paths`, or running on a subset of them,
  reproduces the same counts for the same cells — and because the key is derived
  per row rather than drawn from a shared stream, the result is also invariant to
  I/O and thread scheduling. `pyscx.downsample_file_identity(path)` exposes the
  path component.
- **Enabling it rounds every cell's counts to integers** (ties-to-even), not just
  the cells above target — the counts are trial counts for a discrete sampler.
- **It happens before the batch is returned**, which is deliberate. Callers
  sample the decoder query from the gathered counts (`counts > 0`) and then hand
  the same arrays to the collator; downsampling later would draw the query from
  pre-downsample expressed genes while the numerics used post-downsample counts.

`pyscx.downsample_counts_csr(...)` applies the same primitive to a CSR batch you
gathered yourself. Prefer the loader arguments when the loader is doing the
gather. Across several files, pass one `downsample_file_identity` value per row;
the loader refuses a multi-file downsample without them rather than key on
manifest position.

> **`SparseCellSetBatch.data` is no longer a passthrough for signed or NaN
> values.** Independently of downsampling, the gather now clips negatives (and
> NaN) to zero, because the collate kernel was already doing so lazily on every
> read and the two therefore disagreed about what a row contained. The number of
> stored nonzeros is unchanged — a clipped entry stays as an explicit zero — but
> the values are. Code relying on negatives reaching the consumer needs to read
> them before the gather.

The dense loaders (`TrainingDataset`, `MultimodalTrainingDataset`,
`IndexPlanDataset`) clip **only where a log is applied** — `log1p=True` and
`pflog=True`. `ln` is undefined below `-1`, so an unclipped `log1p` put `NaN`
into the batch for any stored value `<= -1`; under `pflog` a single such value
made the whole cell's baseline `NaN`, and with it every gene in that row. As on
the sparse path, `NaN` clips to `0` too.

The clip reaches the **normalization denominator** too, not just the values.
Under `normalize=True, log1p=True` the per-cell depth is the sum of the clipped
values, so a negative that is about to be discarded cannot inflate the genes
that survive it (`[-1, 3]` normalizes by 3, not 2), and a stored `NaN` cannot
make the depth `NaN` and skip normalization for the whole cell.

With `normalize=False, log1p=False` — or `normalize=True` alone, which scales
rather than logs — negative values still reach the batch unchanged, and the
denominator stays the signed sum. That is deliberate: nothing is undefined
there, and a pre-centered or scaled matrix stored in `X` is a legitimate thing
to stream.

Reproducibility caveat: this is a Rust-native ChaCha8 sampler, so runs
downsampled by a numpy-based implementation are **not** bit-reproducible under it.
What is contractual is the semantics above plus the golden fixture at
`scx-loader/tests/data/downsample_golden.json`.


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
> by `TrainingDataset`, and its `cell_indices` is **uint64** where
> `TrainingDataset`'s is int64. `n_vars` returns a `dict[str, int]` mapping
> modality names to per-modality variable counts. With `return_dict=False`,
> batches are tuples of X arrays only (no obs or cell_indices).
>
> Each modality's `X`, the numeric obs columns and `cell_indices` are **moved**
> into their numpy arrays rather than copied, so handing a batch to Python
> performs **no bulk payload copy**. It is not free per modality: one array is
> still constructed and reshaped and inserted into the `X` dict for each one,
> so the wrapper cost still scales with the modality count — what does not
> scale is the bytes. Those arrays own the loader's decode buffers; they stay
> valid for as long as you hold them, but writing into one writes into nothing
> else's memory. A **categorical** obs column is the exception to the no-copy
> half — its codes are decoded into a Python list of strings, which allocates
> per batch and is not a numpy array at all.

> [!WARNING]
> **`hvg_indices` is not range-checked here — this class only.** A single panel
> is applied to *every* selected modality, and modalities have different feature
> counts, so the check the other loaders apply — reject any index `>= n_vars` —
> would reject an RNA-sized panel outright on a CITE-seq file whose ADT modality
> has ~100 features. The panel is therefore passed through unvalidated, and an
> index beyond a given modality's width yields a column that is **silently
> always zero** for that modality. Check panels yourself against
> `ds.n_vars[modality]`.
>
> This applies to `MultimodalTrainingDataset` and nothing else. A
> *modality-scoped* `TrainingDataset` — `TrainingDataset(path, modality="rna")`,
> or the implicit alphabetically-first fallback on a multimodal file — has one
> panel and one unambiguous `n_vars`, so it **is** range-checked like any
> single-modality loader.
>
> The *other* panel semantics are unchanged here: the panel is still sorted and
> deduplicated, so every modality's columns come back in ascending gene-index
> order, and a non-canonical panel still emits the `UserWarning` (once — one
> panel, one warning).

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

The same construct-in-the-worker rule applies to `IndexPlanDataset` and
`SparseCellSetDataset`. Both spawn CPU work on a pool that
`scx_loader::pool::cpu_pool()` rebuilds whenever the PID changes, so a forked
worker never dispatches to rayon's inherited global registry (whose worker
threads `fork()` does not duplicate — a dispatch to it never returns).

Size the pools with `SCX_LOADER_CPU_THREADS` (default: physical cores capped at
8). It applies *per worker process*, so `num_workers` multiplies it — and a
`TrainingDataset` worker holds **two** pools, the shared `cpu_pool()` plus its
own decode pool, so budget `num_workers × 2 × threads` there. `IndexPlanDataset`
and `SparseCellSetDataset` hold one. See
[multithreading.md § Per-worker thread footprint](multithreading.md#per-worker-thread-footprint).

### Closing a dataset

All four dataset classes have `close()`, and all four release the GIL around
teardown — a `#[pyclass]` is dropped with the GIL held, and tearing down a
tokio runtime blocks until every already-started shard decode returns, which
would otherwise stall every other Python thread (CUDA stream callbacks, the
logging thread) for that window. Dropping the object does the same thing, so
`close()` is for explicitness and determinism, not correctness:

```python
def __iter__(self):
    ds = pyscx.IndexPlanDataset(self.path, **self.kwargs)
    try:
        yield from ds.iter_with_plans(self.plans())
    finally:
        ds.close()
```

One difference worth knowing:

| | `TrainingDataset`, `MultimodalTrainingDataset` | `IndexPlanDataset`, `SparseCellSetDataset` |
|---|---|---|
| after `close()` | re-usable — the next `__iter__` rebuilds the pool and runtime | **terminal** — every *other* method raises `RuntimeError`; construct a new dataset |
| why | its pool and runtime are rebuilt per epoch anyway | the runtime is built exactly once, so a forked child can never inherit live tokio threads; that also means it cannot be rebuilt |
| `closed` means | torn down right now; back to `False` after the next `__iter__` | closed for good |

`repr()` never raises on any of them, and neither does `closed`, which all four
expose. It is deliberately not the same predicate on both halves: on the
re-usable pair it answers "is this torn down right now", so
`if not ds.closed: ds.close()` works everywhere while
`assert ds.closed` after an epoch does not.

Whichever object ends up holding the last reference does the teardown, and every
one of them detaches the GIL first — the dataset's `close()`/`Drop`, and the
batch iterator's `Drop` *and* its end-of-stream branch in `__next__`. So
`ds.close()` followed by draining an outstanding iterator is safe: `close()`
releases only the dataset's reference, and the iterator's own teardown finishes
the job under the same guarantees.

The **bound** is not one number, though it is often quoted as one. The
plan-driven pair (and their iterators) drop a tokio runtime under a single
`SHUTDOWN_DEADLINE` of 5 s. `TrainingDataset` bound-joins its I/O and decode
threads *sequentially*, so ~2 × that, and `MultimodalTrainingDataset` repeats
the whole shutdown per modality.

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
| `adata[:, hvg_mask].copy()` | `hvg_indices=` | Column projection in Rust. Unlike `adata[:, idx]`, the panel is sorted and deduplicated — see `hvg_indices` above |
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

If you see `budget_exceeded: True` in `dataset.memory_budget()`, the pipeline
auto-tuned `prefetch_batches`, `shard_group_size` and `batch_size` all the way
to their minimums and **still** does not fit — peak RSS will exceed the budget
you asked for. A `UserWarning` says so at construction. Check
`effective_batch_size` and increase `max_memory_mb`:

The budget covers anonymous memory only. The mmap'd file is kernel page cache,
evictable under pressure, so it is reported as `mmap_mb` and never budgeted —
on any loader class. That means a file far larger than `max_memory_mb` is
normal and does not by itself shrink the batch.

```python
ds = pyscx.TrainingDataset("atlas.scx", max_memory_mb=1024, ...)
print(ds.memory_budget())
print(f"Effective batch size: {ds.effective_batch_size}")
```

### GIL and CUDA overlap

The GIL is released while waiting for the next batch from the Rust
pipeline, allowing PyTorch CUDA operations (e.g., GPU kernels from the
previous batch's backward pass) to run concurrently.
