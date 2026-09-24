# Python: ML training and tokenisation

> Part of the [SCX API reference](README.md). For the usage guide, see [docs/training.md](../training.md).

## scVI Integration (`pyscx.scx_integrations.scvi`)

Pure-Python PyTorch Lightning DataModule wrapping `TrainingDataset` for scVI model training. Requires `lightning` or `pytorch_lightning`.

> **scVI needs raw counts.** `normalize`/`log1p` default to `True` (inherited from `TrainingDataset`), so batches are log-normalized by default. scVI and other count-likelihood models require raw integer counts — construct with `ScxDataModule(..., normalize=False, log1p=False)`.

- `ScxDataModule(scx_path, batch_size=1024, hvg_indices=None, normalize=True, log1p=True, target_sum=1e4, seed=42, **kwargs)` — Creates a PyTorch Lightning `LightningDataModule`.
  - `scx_path` — Path to the `.scx` file.
  - `batch_size` — Mini-batch size (default: 1024).
  - `hvg_indices` — Gene indices for HVG projection. `None` = all genes.
  - `normalize` — Apply total-count normalization (default: `True`).
  - `log1p` — Apply log1p transformation (default: `True`).
  - `target_sum` — Normalization target sum (default: `1e4`).
  - `seed` — RNG seed for reproducibility (default: `42`).
  - `**kwargs` — Additional keyword arguments passed to `TrainingDataset`.
- Properties: `n_obs`, `n_vars`, `n_output_genes`
- `train_dataloader()` `→ DataLoader` — Uses `batch_size=None` and `num_workers=0` (Rust handles batching and threading internally).
- `val_dataloader()` `→ None` — Validation not currently supported.

```python
from pyscx.scx_integrations.scvi import ScxDataModule
import scvi

dm = ScxDataModule("atlas.scx", batch_size=1024, hvg_indices=hvg_array)
model = scvi.model.SCVI(dm.adata_manager)
model.train(datamodule=dm)
```

## TrainingDataset

High-throughput sequential streaming dataset. Wraps the triple-buffered
Rust pipeline (tokio I/O → rayon decode → Python/GPU). Each `for batch in dataset:`
loop is one epoch; shards are reshuffled between epochs for training randomization.

> **Transforms are ON by default.** `normalize` **and** `log1p` both default to
> `True`, so batches are total-count normalized (`target_sum=1e4`) and
> `log1p`-transformed even though the file stores raw counts — the yielded `X` is
> log-normalized, **not** raw counts. Count-likelihood models (scVI, scANVI,
> count autoencoders, NB/ZINB decoders) need raw counts: pass
> `normalize=False, log1p=False`.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `batch_size` | `1024` | Mini-batch size. Auto-tuned downward if `max_memory_mb` is exceeded. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. Every index must be `< n_vars` — an out-of-range index is rejected at construction, because it matches no column and would otherwise yield an output feature that is silently always zero. **The panel is sorted and deduplicated**, so batch columns are in ascending gene-index order regardless of the order you pass, and duplicates shrink the batch width (check `n_output_genes`). Passing a panel that is not already ascending-unique emits a `UserWarning`; `np.unique(hvg_indices)` reproduces the column order the batches use. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p` in a single CSR row scan). **On by default** — set `normalize=False` (with `log1p=False`) for raw-count output. |
| `log1p` | `True` | Apply `log1p` after normalize. **On by default** — set `log1p=False` for raw-count output. |
| `target_sum` | `1e4` | Normalization target sum. |
| `pflog` | `False` | Apply PFlog (v4) / shifted-log normalization on raw counts (Booeshaghi et al.) instead of `normalize`/`log1p`. Mutually exclusive with them — when `True` it takes precedence and those flags are ignored. |
| `pflog_alpha` | `None` | PFlog NB overdispersion `α` (matrix-wide pseudocount `1/(4α)`). `None` estimates `α` once at loader construction from the raw counts (single-modality only); a float pins it. Only used when `pflog=True`. |
| `shard_group_size` | `8` | Shards per I/O group. Sequential I/O within each group for disk efficiency. |
| `prefetch_batches` | `4` | Ring buffer depth — number of pre-built batches to buffer ahead. |
| `seed` | `42` | RNG seed for reproducibility. Deterministic shuffle via `(seed, epoch)`. |
| `max_memory_mb` | adaptive (≥512) | Memory budget. **On every loader class it excludes the mmap'd file** — kernel page cache, evictable under pressure, reported (`mmap_mb`) but never budgeted. What each class *does* charge differs: this one costs the shard buffers and the batch ring, while `SparseCellSetDataset` costs only its shard cache (see its row for the two terms it leaves out). **When omitted**, the budget is *adaptive*: it scales up to fit the file's requested configuration (so a full-width ~33k-gene file keeps its requested `batch_size` instead of silently shrinking), floored at 512 MB and capped at 4096 MB. Pass an explicit value to pin a **hard ceiling** — then the pipeline auto-tunes `prefetch_batches` (to 2), then `shard_group_size` (to 1), then `batch_size` (halved, to 64), in that order. If it still does not fit, `budget_exceeded` is set and a `UserWarning` is raised. |
| `modality` | `None` | For multimodal v2 files: name of the modality to load (e.g. `"rna"`). Ignored on single-modality files. |

**Properties**

- `n_obs` → `int` — Total number of observations (cells) in the dataset.
- `n_vars` → `int` — Total number of variables (genes) in the dataset.
- `n_output_genes` → `int` — Genes per batch (HVG count if projection active, else `n_vars`).
- `effective_batch_size` → `int` — Actual batch size after memory budget auto-tuning.

**Methods**

- `close()` — Explicitly shut the pipeline down (join I/O + decode threads, release rayon pool). Idempotent, and **not** terminal: the next `__iter__` rebuilds and starts a fresh epoch. Recommended before process exit; see [Fork safety](#fork-safety-under-pytorch-dataloadernum_workers--0) and **Lifecycle — `close()` and `closed`** under [IndexPlanDataset](#indexplandataset).
- `memory_budget()` → `dict` — `breakdown` (the six-key per-component estimate every class that reports a budget uses) plus this class's own `shard_group_size`, `prefetch_batches`, `batch_size`, `estimated_mb`, `mmap_mb`, `budget_exceeded`. `mmap_mb` appears here and nowhere else, but it is **not** budgeted anywhere: `estimated_mb` equals `breakdown["total_bytes"]`. `budget_exceeded` means the auto-tune could not fit even at its minimums, and also raises a `UserWarning` at construction.

**Properties (lifecycle)**

- `closed` → `bool` — True from `close()` until the next `__iter__` rebuilds. Never raises.

**Batch dict schema**

```python
{
    "X":            np.ndarray[B, n_output_genes, float32],   # dense expression
    "obs":          dict[str, np.ndarray | {"codes", "categories"}],
    "cell_indices":  np.ndarray[B, int64],   # global row indices
}
```

Categorical obs columns encode as `{"codes": ndarray[int32], "categories": list[str]}`.
Numeric obs columns are `ndarray[int64]` or `ndarray[float64]`.

**Epoch and shuffling semantics**

Each `for batch in dataset:` loop is one epoch. On each epoch:
- **Level 1 (shard order)**: Shard indices `[0..n_shards)` are randomly permuted using a
  deterministic RNG seeded from `(seed, epoch_number)`. Permuted shards are grouped into
  contiguous I/O groups of `shard_group_size` for disk-sequential reads.
- **Level 2 (row shuffle)**: Within each shard group, cell indices are Fisher-Yates shuffled
  and sliced into `batch_size`-sized batches.

This two-level shuffle provides training randomization without random I/O. The same `seed`
and epoch always produce the identical ordering.

> [!NOTE]
> The `seed` → ordering mapping changed in v0.13.1: both levels now chain
> `(seed, domain tag, epoch)` through SplitMix64 instead of adding them, which
> removes a collision where `(seed, epoch)` aliased `(seed + φ, epoch − 1)`.
> Determinism within a version is unchanged; a given seed produces a different
> ordering than it did before. See [docs/training.md](../training.md#epoch-and-shuffling).

```python
dataset = pyscx.TrainingDataset("file.scx", batch_size=1024,
    hvg_indices=hvg_array, normalize=True, log1p=True,
    obs_columns=["cell_type", "batch"])

for epoch in range(n_epochs):
    for batch in dataset:
        x = batch["X"]              # [B, n_output_genes] float32
        obs = batch["obs"]          # {"cell_type": {"codes": ..., "categories": ...}, ...}
        idx = batch["cell_indices"]  # [B] int64 — global row indices

dataset.close()
```

## IndexPlanDataset

Plan-driven paired-batch reader for ML workloads where each batch is a list
of `(perturbed_cell, control_cell)` index pairs (perturbation training,
contrastive learning, donor-matched designs). Sibling to `TrainingDataset`:
`TrainingDataset` streams shards in catalog order for the highest possible
sequential throughput; `IndexPlanDataset` consumes a Python iterator of
plans and yields paired dense batches, trading sequential streaming for
per-cell pairing flexibility.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `path` | — | Path to `.scx` file. |
| `hvg_indices` | `None` | `np.ndarray[u32]` of gene indices for HVG projection; `None` = all genes. Every index must be `< n_vars` — an out-of-range index is rejected at construction, because it matches no column and would otherwise yield an output feature that is silently always zero. **The panel is sorted and deduplicated**, so batch columns are in ascending gene-index order regardless of the order you pass, and duplicates shrink the batch width (check `n_output_genes`). Passing a panel that is not already ascending-unique emits a `UserWarning`; `np.unique(hvg_indices)` reproduces the column order the batches use. |
| `obs_columns` | `[]` | Obs metadata column names included in each batch. |
| `normalize` | `True` | Total-count normalize (fused with `log1p`). |
| `log1p` | `True` | Apply `log1p` after normalize. |
| `target_sum` | `1e4` | Normalization target sum. |
| `cache_shards` | `128` | LRU shard cache budget. Auto-tuned downward to fit `max_memory_mb`; check via `effective_cache_shards()`. |
| `sort_by_shard` | `True` | Reorder each plan by `min(shard_of(p), shard_of(c))` so the returned `X`/`X_paired` rows land in shard locality order. Disable to preserve caller's input pair order. |
| `lookahead` | `4` | Default lookahead for `iter_with_plans` when not overridden. `0` disables shard prefetching; auto-tuned downward to fit `max_memory_mb`; check via `effective_lookahead()`. |
| `max_plan_size` | `16384` | Upper bound on rows-per-batch for the memory budget calculation. |
| `max_memory_mb` | `512` | On overflow, `lookahead` is reduced first (down to 1), then `cache_shards` (down to 1); construction fails with `RuntimeError` if neither fits. |

**Batch dict schema** (yielded by `iter_with_plans`):

```python
{
    "X":          np.ndarray[B, n_output_genes, float32],   # perturbed rows
    "X_paired":   np.ndarray[B, n_output_genes, float32],   # control rows
    "pairs":      list[tuple[int, int]],                    # post-sort plan
    "obs":        dict[str, np.ndarray | {"codes", "categories"}],
    "obs_paired": dict[str, np.ndarray | {"codes", "categories"}],
}
```

`pairs[i]` always corresponds to `X[i]` and `X_paired[i]`. Categorical obs
columns encode as `{"codes": ndarray[i32], "categories": list[str]}` —
schema matches `TrainingDataset`.

**Iterator semantics**

- `plans` is any Python iterable yielding `list[tuple[int, int]]` (plain lists,
  generators, queues all work).
- Plan iteration is lazy: the loader pulls the next plan only when it is
  ready to schedule a prefetch for it.
- The loader keeps *up to* `lookahead` plans in flight at once: the head plan is
  decoding while shards for the next `lookahead - 1` are being warmed via
  `tokio::task::spawn_blocking` calls into `BackedCsrReader::read_shard_cached_arc`.
- **`lookahead` is a budget, not an obligation on the generator.** The loader
  waits for the first plan of a batch and fills the remaining slots only from
  plans the generator has already produced. A generator that yields plan *i+1*
  only after inspecting batch *i* — curriculum sampling, hard-negative mining —
  therefore makes progress rather than deadlocking; it simply runs
  un-prefetched. (Before v0.13.1 it hung: the generator waited for the batch,
  the loader waited for `lookahead` plans.)
- The plan-pull thread is eager and never waits for the consumer, so such a
  generator must **block on an explicit feedback signal** after `yield` — the
  loader will otherwise ask for the next plan before the current batch exists.
  See [training.md § Feedback and curriculum plan generators](../training.md#feedback-and-curriculum-plan-generators)
  for the rendezvous.
- `StopIteration` from `plans` ends the batch stream cleanly. Other Python
  exceptions from `plans` propagate as `RuntimeError("plan iterator raised: ...")`.
- Empty plans inside a stream are silently skipped.
- Out-of-range row indices raise `IndexError` immediately when validating
  the offending plan; missing obs columns raise `KeyError` at construction.
- `os.fork()` after construction raises `RuntimeError` with `num_workers=0`
  guidance — the shard cache and mmap state are not fork-safe; consumers
  must lazily construct the dataset post-fork in each DataLoader worker.

**Example — bare iterator**

```python
import pyscx

ds = pyscx.IndexPlanDataset(
    "atlas.scx",
    hvg_indices=hvg_array,
    obs_columns=["cell_type", "perturbation"],
    normalize=True,
    target_sum=1e4,
)

plans = [
    [(0, 5), (2, 7)],
    [(10, 100), (50, 75)],
]
for batch in ds.iter_with_plans(iter(plans)):
    X, X_paired = batch["X"], batch["X_paired"]
    pairs = batch["pairs"]
    # ... training step ...
```

**Example — paired with a `BaseMappingStrategy`-style plan generator** (the
cell-load-scx pattern):

```python
def plan_generator(strategy, perturbed_indices, batch_size):
    """Wrap any pairing policy that exposes `get_control_index(idx) -> int`."""
    buf = []
    for p_idx in perturbed_indices:
        c_idx = strategy.get_control_index(p_idx)
        if c_idx is not None:
            buf.append((p_idx, c_idx))
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

ds = pyscx.IndexPlanDataset("atlas.scx", obs_columns=["cell_type"])
for batch in ds.iter_with_plans(plan_generator(my_strategy, perm, 1024),
                                lookahead=4):
    ...
```

**Memory budget surfaces**

```python
ds = pyscx.IndexPlanDataset("atlas.scx", cache_shards=128, lookahead=4,
                            max_plan_size=16384, max_memory_mb=256)
print(ds.effective_cache_shards(), ds.effective_lookahead())
# Detects when auto-tuning kicked in.
print(ds.memory_budget()["max_blocking_threads"])
```

`max_blocking_threads` is the cap on simultaneously-running shard decodes. It
lives on the prefetch engine this class shares with `SparseCellSetDataset`, and
is sized from the **constructor** `lookahead` — which bounds in-flight *plans*,
not the blocking task a plan spawns per distinct `(file, shard)` it touches.

**Lifecycle — `close()` and `closed`**

All four dataset classes expose `close()` and a `closed` property, and all four
release the GIL around teardown, so tearing a dataset down cannot stall other
Python threads while an in-flight shard decode finishes. The bound is not one
number: the plan-driven pair drop a tokio runtime under a single 5 s deadline,
while the training pair bound-join two threads *sequentially* (`~2 ×
SHUTDOWN_DEADLINE`), and `MultimodalTrainingDataset` repeats that per modality.
Dropping the object does the same, so `close()` buys determinism rather than
correctness. `closed` and `repr()` never raise on any of them.

`close()` is **terminal on `IndexPlanDataset` and `SparseCellSetDataset`**:
their tokio runtime is built exactly once so that a forked child can never
inherit live tokio threads, which also means it cannot be rebuilt. Every other
method raises `RuntimeError` afterwards; construct a new dataset. On
`TrainingDataset` and `MultimodalTrainingDataset` it is not — their pool and
runtime are per-epoch anyway, so the next `__iter__` rebuilds them.

`closed` follows that split rather than papering over it: on the terminal pair
it means closed for good, on the re-usable pair it means torn down *right now*
and returns to `False` after the next `__iter__`. So `if not ds.closed:
ds.close()` is portable across all four; `assert ds.closed` after an epoch is
not.

A still-alive `IndexPlanBatchIter` / `SparseCellSetBatchIter` holds its own
reference, so `close()` releases only the dataset's. The iterator then does the
teardown itself — under the same 5 s bound and with the GIL detached — on
whichever comes first, its `Drop` or the end-of-stream branch of `__next__`. So
this ordering, which the API supports, is safe:

```python
it = ds.iter_with_plans(plans)
ds.close()      # releases the dataset's reference only
list(it)        # the iterator's own teardown, bounded and off-GIL
```

**Iterator metrics — `metrics()`**

Both `IndexPlanBatchIter` and `SparseCellSetBatchIter` expose `metrics()`,
returning `{"cache": {...}, "prefetch": {...}}`. The `cache` half is
loader-cumulative and identical to the dataset's `cache_metrics()`; the
`prefetch` half is per-iter and resets on each `iter_with_plans` call:

| key | meaning |
|---|---|
| `prefetch_tasks_spawned` | shards warmed into the LRU ahead of the gather |
| `prefetch_skipped_cache_hit` | already resident |
| `prefetch_skipped_in_flight` | a peer was already decoding it |
| `prefetch_skipped_block_index` | not warmed *whole*, so the gather takes the row-group path; its touched row groups are pre-decoded into the row-group LRU instead when the plan fits `budget / (lookahead + 1)` |
| `prefetch_skipped_reader_limit` | the plan touches more distinct files than `reader_limit` can keep resident, so it was not prefetched at all — the prefetcher would have had to pin the plan's whole width. Structurally 0 on `IndexPlanDataset` (single-file) and on any dataset that did not set `reader_limit`. The plan still gets a real admission verdict. |

Both handles are cloned when the iterator is built, so `metrics()` is safe to
call after the iterator has been drained.

`prefetch_skipped_block_index` counts the **L2 prefetch-time** decision: shards
not warmed whole so the gather could take the row-group path. The name predates
the row-group LRU and is a contract; such a shard is no longer left cold. The
engine takes **one row-group admission verdict per plan** — the row-group bytes
of every framed shard the plan touches, over every file, sized from the block
index with no decode, against `max_memory_mb`'s cache share
`budget / (lookahead + 1)` — and that verdict drives both sides: an admitted
plan's eligible shards are pre-decoded into the row-group half of the LRU and
the gather reports them as `cache_metrics()["row_group_hits"]`; an over-share
plan is not warmed, and retains only the groups **another plan of the lookahead
window also touches**, hottest-first and only while they fit the same share —
its cold tail still decodes and drops, so `row_group_misses` keeps growing while
`row_group_bytes_inserted` moves only by the reused groups.
`cache_metrics()["reuse_admissions"]` counts the plans that got such a partial
verdict, and `admitted_group_bytes` / `rejected_group_bytes` is the split it
decided. Before that, an over-share plan retained nothing at all —
`SCX_ROW_GROUP_ADMIT=plan` restores it as the same-build A/B arm. The bound on
the partial verdict is load-bearing, not belt-and-braces: a wide random plan can
touch nearly every row group a file has, at which point every key is "reused"
and an unbounded rule degenerates into admitting everything. It is a useful
confirmation that `scatter_block_index=True` had an effect — it stays 0 against
an unframed file. Both classes now warn at construction when the kwarg is set
and no shard is framed, so this is a confirmation rather than the only signal;
on `SparseCellSetDataset` the warning tests *any file in the set*, so a mixed
set is silent and these counters are still what tell you the route ran.

It is **not** the same signal as `cache_metrics()["block_index_groups"]`, and
the two can legitimately differ. `block_index_groups` is the route the *gather*
actually took; the prefetch counter only exists when prefetching runs. At
`lookahead=0` no prefetch runs at all, so every counter here is 0 while
`block_index_groups` is still positive — and with concurrent iterators sharing
one cache, a peer can warm a shard between the prefetch decision and the
gather. **Use `block_index_groups` as the route signal**; use these counters to
see what the prefetcher decided.

```python
ds = pyscx.IndexPlanDataset("atlas.scx")
try:
    for batch in ds.iter_with_plans(plans):
        ...
finally:
    ds.close()
assert ds.closed
```

## Fork safety under PyTorch `DataLoader(num_workers > 0)`

`pyscx.TrainingDataset`, `pyscx.IndexPlanDataset` and
`pyscx.SparseCellSetDataset` are fork-safe under
`torch.utils.data.DataLoader(num_workers > 0, start_method="fork")` —
the Linux PyTorch default — when the dataset is **constructed lazily
inside the worker's `__iter__`** (the pattern `cell-load-scx` and
`state-scx` already use). The per-pipeline tokio runtime (current-thread,
per-epoch), the per-pipeline `rayon::ThreadPool` (lazily built on first
iteration) and the process-wide `scx_loader::pool::cpu_pool()` used during
construction are all created inside the worker process, so a forked child
inherits no fork-hostile state from the parent. `cpu_pool()` is keyed on the
PID and rebuilt when it changes, which is what makes "created in the worker"
true even if the parent touched a loader path first. Construction therefore
does own worker threads — they are simply the child's own. See
[multithreading.md § Per-worker thread footprint](../multithreading.md#per-worker-thread-footprint)
for how many.

```python
# Recommended IterableDataset wrapper for DataLoader(num_workers=2)
import torch.utils.data as data

class TrainingShim(data.IterableDataset):
    def __init__(self, scx_path):
        self.scx_path = scx_path  # paths only — no inner dataset yet

    def __iter__(self):
        # Construct the inner dataset HERE (in the worker process, post-fork).
        ds = pyscx.TrainingDataset(self.scx_path, batch_size=1024, hvg_indices=...)
        try:
            yield from ds
        finally:
            ds.close()  # release the per-pipeline rayon pool / tokio runtime

loader = torch.utils.data.DataLoader(
    TrainingShim("atlas.scx"),
    batch_size=None,
    num_workers=2,
    persistent_workers=False,
)
```

**Do**: construct lazily inside `__iter__`; call `dataset.close()` (or
register `weakref.finalize(dataset, dataset.close)`) before process exit;
prefer `multiprocessing.set_start_method("spawn")` if your workload
allows — spawn re-execs Python in the child and is genuinely fork-safe
because there is no fork.

**Don't**: construct a `TrainingDataset` / `IndexPlanDataset` in the
parent and share it across forked workers — the PID check in `__next__`
raises `RuntimeError`. Don't pickle a constructed dataset across
processes either; it owns thread handles that don't survive transfer.

> [!IMPORTANT]
> Calling **any** `rayon::par_*`-using pyscx API in the parent before
> fork (e.g., `pyscx.from_anndata(...)` to write the fixture) initialises
> rayon's process-global pool. Pre-fix this prerequisite was sufficient
> to wedge `DataLoader(num_workers=2)` indefinitely; the per-pipeline
> rayon pool in `scx-loader` removed that hazard. See
> `pyscx/tests/test_fork_safety.py` for the durable regression test.

## SparseCellSetDataset

Plan-driven **sparse** reader for role-tagged cell sets spanning several `.scx`
files — the shape perturbation screens and contrastive set-based models want.
Where `IndexPlanDataset` yields a dense paired batch from one file,
`SparseCellSetDataset` gathers flat CSR rows across files and delimits each set
with `set_offsets`.

**Constructor kwargs**

| Argument | Default | Notes |
|---|---|---|
| `paths` | — | List of `.scx` paths. `file_id` in a plan is the index into this list. |
| `cache_shards` | `None` → 128 | Shard-cache count cap. Size it with `suggested_cache_shards`, not by guessing — see [Sizing the shard cache](../training.md#sizing-the-shard-cache). |
| `max_memory_mb` | `None` | Memory budget. Since ORG-9.10-5 the constant interpreter/numpy/Arrow overhead is subtracted before the cache is sized, so this is no longer a bare cache cap — but it is **not** a hard ceiling on the process either: the batch's per-row transients are never charged and the batch itself only when `max_plan_rows` declares how wide plans get (this path has no `max_plan_size`, so plan output size is otherwise caller-controlled), and one above-average shard can sit above the byte cap because the LRU keeps an oversize entry rather than refusing to cache it. `None` resolves adaptively (never unbounded). Both the count cap and the byte cap are enforced, and on large-shard files the byte cap binds first. |
| `remap_tables` / `n_global_genes` | `None` | Per-file `local → global` gene tables. Required for a cell **set** that spans files: raw-local indices from different files are not comparable. |
| `normalize` / `log1p` / `target_sum` | `None` → off | Off by default here, unlike the training classes. |
| `lookahead` | `None` → tuned | Prefetch depth; `0` disables prefetch. |
| `scatter_block_index` | `False` | Opt into the row-group block-index scattered route. Defaulted **off**, and there is no route that wins everywhere: full-shard beats block-index 117–167× on 100k-cell corpora but **loses 1.53×** on census_1m with grouped plans, so the two cross near 1M cells. The default serves the common case; pick per dataset. See [Cell-set scatter routes, re-measured after the row-group LRU](../performance/loader-index-plan.md#cell-set-scatter-routes-re-measured-after-the-row-group-lru-phase-0-gate) and [sharding.md § Row-group framing & scattered reads](../sharding.md#row-group-framing--scattered-reads). |
| `max_plan_rows` | `None` | Upper bound on rows per plan, **enforced**: a wider plan raises `RuntimeError`, because a cache sized for `N` rows while the gather accepts any width is not a bound. **Uncharged and unenforced by default** — the resolved cache is then exactly what it would be without the argument. Declared, it charges one gathered CSR batch — `presize_nnz(rows, mean_nnz_per_row) × 8` (the mean plus a ⅛ bias) + `(rows + 1) × 8` — against `max_memory_mb` before the cache is sized, reported as `breakdown["batch_buffer_bytes"]`, and on the configurations whose gather holds a second buffer (a remap, a downsample, or more than one file) the same figure again as `breakdown["transient_bytes"]`. The gather itself no longer allocates the biased figure — it sizes the batch exactly from an indptr prescan — so the ⅛ is headroom on an estimate rather than a model of the allocation. It bounds the plan's **row count**, not its bytes: the density is a manifest-wide mean, so a denser-than-average plan can exceed the charge. Opt-in because this class has no `max_plan_size`: plan width is the caller's, and a guessed default would shrink the shard cache on every existing dataset. |
| `reader_limit` | `None` | Cap on simultaneously-open readers. `None` opens every path and never closes one — today's behaviour, byte-identical in sizing, throughput and gather output. What it bounds is **resident memory, not file descriptors**: opening an SCX file mmaps it and closes the descriptor, and these readers do not watch their files, so a 5,000-file manifest constructs and gathers under `ulimit -n 1024` with the descriptor count flat. An open reader costs ~104 kB (tabula) to ~121 kB (census_1m) resident, over 90% of it the parsed catalog, so a 26k-file manifest is ~2.8-3.2 GB per process. A reopen re-parses that catalog (0.09-20 ms/file), which is exactly why the saving is real. Caps handles the loader may drop, not handles in existence — a plan needing more files at once exceeds it rather than blocking, and `cache_metrics()["reader_hwm"]` reports the truth. On the streaming path the figure to size against is roughly `files-per-plan x (lookahead + 1)`, since each in-flight plan leases every file it touches; a plan wider than the limit is **not prefetched** (counted by `metrics()["prefetch"]["prefetch_skipped_reader_limit"]` on the iterator, not by `cache_metrics()` — it is a per-iterator counter) and falls back to the one-file-at-a-time gather. Not charged to `max_memory_mb`. See [Very large manifests](../training.md#very-large-manifests). |
| `downsample_target_library_size` / `downsample_method` / `downsample_seed` | `None` | Seeded per-row count downsample, applied before the batch leaves Rust. |

**Plans.** Each plan is a tuple `(file_ids, rows, role_tags, set_offsets)`:
`file_ids` and `rows` are parallel per-row arrays, `role_tags` labels each row
(perturbed / control / …), and `set_offsets` has length `n_sets + 1` and
delimits each set's row range in the flat batch.

**Properties**

- `n_files` → `int`, `n_cols` → `int` — reader count and CSR column count.
- `closed` → `bool` — True once `close()` has run. Terminal on this class; never raises.

**Methods**

- `iter_with_plans(plans, lookahead=None)` → `SparseCellSetBatchIter` — stream plans into §4.4 sparse batch dicts.
- `gather(file_ids, rows, role_tags, set_offsets)` → `dict` — gather **one** plan synchronously, returning the same batch dict the iterator yields (`ds.gather(*plan)`). Admission is decided once over the whole plan against the whole byte budget, where `iter_with_plans` takes one verdict per plan against its divided share (`budget / (lookahead + 1)`); that changes what the cache *retains*, not what is read, and the output is identical. Gathered on the calling thread, so it is not the way to drive an epoch. Raises exactly as the iterator does.
- `suggested_cache_shards(plan)` → `int` — distinct `(file_id, shard)` pairs one plan touches. Takes the same four-tuple `iter_with_plans` consumes; `role_tags` / `set_offsets` are ignored.
- `cache_metrics()` → `dict` — cumulative shard-cache counters, including `full_shard_groups` / `block_index_groups`, which report the scattered-read route the gathers actually took, and the `row_group_*` set (`hits`, `misses`, `evictions`, `bytes_inserted`, `duplicate_waiters`) for the decoded row groups a framed `scatter_block_index=True` gather retains. `hits` / `misses` / `evictions` / `bytes_inserted` keep meaning **whole-shard** entries; both kinds share one LRU and one byte budget, and `peak_bytes_in_cache` gauges both, so `peak <= bytes_inserted + row_group_bytes_inserted`. `SCX_ROW_GROUP_CACHE=0` disables row-group retention process-wide. Four further keys report the **reader registry** rather than the cache — `reader_opens`, `reader_evictions`, `reader_resident`, `reader_hwm`. They are absent from `IndexPlanDataset.cache_metrics()`, which is single-file and would report them as permanently zero. At the default `reader_limit=None`, `reader_opens` equals the manifest size and the other three never move. `reader_opens` counts the registry's opens — what it was handed, plus every reopen since — not the constructor's manifest scan, which opens every file once whatever the limit.
- `memory_budget()` → `dict` — `breakdown` plus `max_memory_mb`, `cache_shards`, `effective_cache_shards`, `shard_decoded_bytes`, `max_plan_rows`, `mean_nnz_per_row`, `max_blocking_threads`, `reader_limit`, `budget_exceeded`. `cache_bytes` and `python_overhead_bytes` are the non-zero terms by default; `max_plan_rows` adds `batch_buffer_bytes`, and `transient_bytes` beside it when the loader's configuration forces the gather to assemble the batch from a second buffer. `budget_exceeded` means even a one-shard cache does not fit. It is a statement about the **cache this loader sizes**, not a guarantee about process RSS — see the `max_memory_mb` row. `max_blocking_threads` is the cap on simultaneously-running shard decodes: `lookahead` bounds in-flight *plans*, not the tasks a plan spawns (one per distinct `(file, shard)` it touches). `reader_limit` is reported but deliberately **outside** `breakdown`: the breakdown is the byte model the shard cache is sized against, and an open reader's catalog is not one of its terms, so charging readers there would shrink the cache by something the tuner has never accounted for.
- `close()` — release the prefetch engine's tokio runtime, GIL detached, 5 s bound. Idempotent and **terminal** — see **Lifecycle — `close()` and `closed`** under [IndexPlanDataset](#indexplandataset).

## Tokenisation kernels (`pyscx.tokenize`)

Numeric kernels over a gathered CSR batch — the per-cell steps a transformer-class model's tokeniser is built from. Each takes the whole batch as `(indptr, indices, data)`, runs its row loop in Rust with the GIL released, and returns numpy arrays **moved** (not copied) out of Rust. Full semantics, parameter-by-parameter, and the divergences from each reference implementation: [docs/tokenize.md](../tokenize.md).

`pyscx.tokenize.CONTRACT_VERSION` → `int` pins the kernels' numeric semantics, the parameters each requires, and the seed derivation for the RNG-driven ones. A module constant, not also a function. Assert it at setup. It does not cover array shapes or dict key names.

Gene ids must be non-negative and strictly ascending within each row — what a gathered batch always is, and **enforced** at these entries, since a `scipy` CSR is not index-sorted until `sort_indices()` and an unsorted or negative id otherwise reached a panic or a silently wrong mask.

| Kernel | Signature | Returns |
|---|---|---|
| `top_k` | `(indptr, indices, data, k, n_genes_total)` | `{ids, values, mask, pad}`, each `[n_rows * k]`, plus `n_rows` / `k`. Value descending, gene id ascending on ties; unfilled slots PAD; a row with no positive value gets one GENE_MASK token. The withheld-gene masking `collate_cellset_gathered` layers on this kernel is reachable only there. |
| `rank_tokens` | `(indptr, indices, data, gene_stats, l_max, vocabulary_version, target_sum=1e4)` | `{ids, lengths, norm_identity, n_rows, l_max}`. `gene_stats` is indexed by global gene id, every entry finite and strictly positive. Slots past a row's length are **undefined** — this kernel reports a length, the consumer owns padding. `norm_identity` is a blake3-64 stamp over the statistics and the vocabulary version; record it with the run. ⚠️ Ties break by gene id ascending; Geneformer's `np.argsort` default is unstable and follows no rule. |
| `bin_values` | `(indptr, indices, data, n_bins, edges=None, tie="left", seed=0, file_identity=0, rows=None)` | `{bins, n_rows, n_bins}`; `bins` is `[nnz]`, parallel to `data`. Zeros stay at bin 0. `edges=None` recomputes per-cell quantile edges (scGPT's `Preprocessor`); explicit `edges` must be exactly `n_bins - 1` finite non-decreasing values (checked before the all-zero early return) and become part of the tokeniser's identity. ⚠️ `tie` — `"left"` / `"right"` are `np.digitize`'s two deterministic bounds; `"seeded"` is scGPT's randomisation keyed on content instead of numpy's global RNG, which reproduces its distribution but never its draws. |
| `sample_genes` | `(indptr, indices, data, n, seed, file_identity, weight="log1p", rows=None)` | `{ids, lengths, n_rows, n}`. Draws **with replacement** (UCE's `replace=True`), weights `log1p(count)` renormalised. Pass real row ids in `rows` and the file's `downsample_file_identity`, or the draw is keyed on batch position and is reproducible only for that batch. |
| `transform_values` | `(indptr, data, mode, target_sum=1e4, pflog_alpha=None, n_measured=None)` | a new `data` array. `mode` is the collate kernel's: `pass_through` / `log1p_raw` / `normalize_log1p` / `pflog_raw`. `pflog_raw` is PFlog **v4** and requires both extra parameters. |
| `library_size` | `(indptr, data)` | `[n_rows]` `float64`. ⚠️ The sum of the row **as given** — on a panel-projected batch that is the library size after feature filtering, not the cell's sequencing depth. |
| `measured_mask` | `(indptr, indices, panel)` | `[n_rows * len(panel)]` `uint8`. "Measured", not "non-zero": a gene the row does not carry is absent from the CSR, which on a heterogeneous panel is a different claim from a zero count. |
| `gene_mask_id` / `pad_id` | `(n_genes_total)` | the two crop sentinels, `n` and `n + 1`. `top_k` rejects any gene id `>= n_genes_total` for exactly this reason. |

## Neighbourhood plans

Turn a stored `obsp` graph, or `obsm` coordinates, into the plans
`SparseCellSetDataset.gather` / `.iter_with_plans` already take. Each plan is
**one set**: the centre at position 0 with `role_tag` 0, then its neighbours
with `role_tag` 1. See [docs/training.md § Neighbourhood
plans](../training.md#neighbourhood-plans) for the worked example.

| Function | Signature | Returns |
|---|---|---|
| `neighborhood_plans_from_graph` | `(path, key="connectivities", *, file_id, k=None, weight_order=None, include_center=True, drop_deleted=True, chunk_rows=65536)` | `(plans, centers)`. `plans` is a list of `(file_ids, rows, role_tags, set_offsets)` numpy tuples, one per surviving centre; `centers` is those centres' physical rows. `k` keeps `k` edges from the `weight_order` end (required with `k`), ties by column ascending — a rule this function declares because a stored graph carries none; `k=None` keeps every stored edge in column order. Non-finite weights sort last either way, so a distance graph's `inf` is never picked as a near neighbour. The graph is read one `chunk_rows` range at a time. |
| `neighborhood_plans_from_coords` | `(path, obsm_key="spatial", *, file_id, k=None, radius=None, include_center=True, drop_deleted=True)` | `(plans, centers)`, as above. Exactly one of `k` / `radius` is required; neither or both raises rather than resolving to one. A uniform grid is built at call time (O(n), no on-disk index) and searched ring by ring, so the answer is **exact**; ties order by squared distance then row ascending. **1-D, 2-D or 3-D only** — the search is exponential in the dimensionality, so a wide key raises rather than not returning. Integer and float64 columns narrow to float32. |
| `batch_plans` | `(plans, sets_per_batch, *, shuffle_seed=None)` | Single-set plans concatenated into batch plans. `shuffle_seed` reorders the **sets**, never a set's members. The last batch is short, not dropped. |

- `path` accepts a `str`, an `os.PathLike`, or an open `Experiment`.
- ⚠️ **`file_id` is a manifest position, not a file identity** — the index of
  this file in the `SparseCellSetDataset` the plans will be gathered with.
  Plans built from file A and fed to a dataset whose manifest puts A third
  gather the *first* file's rows, silently. Keyword-only and **required**: a
  default of `0` would make the documented hazard the quiet path.
- ⚠️ **`weight_order` is required whenever `k` is given, and has no default.**
  `"desc"` suits an affinity graph (`connectivities`: larger = closer); `"asc"`
  suits a distance graph (`distances`: larger = farther). A default of `"desc"`
  would be right for the default key and silently wrong the moment a caller
  changed only the key — returning each cell's `k` **farthest** neighbours —
  so the direction is stated rather than documented.
- ⚠️ **The two builders order a set's neighbours differently.**
  `_from_graph` emits them **column-ascending** — a stored graph carries no
  other order, and re-sorting by weight would make which edges `k` picked also
  change where they land. `_from_coords` emits them **nearest first**, the
  order it computed. Position 1 is therefore the nearest neighbour on the
  coordinate path and simply the lowest-numbered one on the graph path.
- With `k`, a deleted row is **not a candidate**, so the `k` best of the *live*
  neighbours are taken and the set is still `k` wide. Without `k`, a set is
  simply short by whatever is gone. Different answers, both intended.
- A `Float64` COO `data` column is narrowed to `f32` before ranking, so `k`
  over a float64 distance graph can order two very close weights differently
  from `to_anndata().obsp[key]`, which keeps them wide.
- Rows are **physical**. A deleted centre yields no set (so `centers` is
  shorter than `n_obs`); a deleted neighbour is never emitted. What that costs
  a set is the `k` / no-`k` split above. `drop_deleted=False` opts out.
- A NaN or infinite coordinate on a kept cell **raises**. That differs from the
  tokenisation kernels, which clip a NaN value to zero: zero is a meaningful
  expression level, and a NaN coordinate has no defensible grid cell.
- Sets never span files, so these plans never need `remap_tables`.
- `scx-accel`'s kNN is deliberately not used: it dispatches to approximate HNSW
  above 5,000 cells and has no radius mode.
