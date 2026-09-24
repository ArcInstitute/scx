# Training loader: data-load phase 1

> Part of [SCX performance](README.md). See also the [training loader overview](loader.md).

## Shard-cache sizing on the gather path (data-load Phase 1, 1A)

The pathology that motivated this work: STATE3 measured **143 s/batch** on a scattered
perturbation gather, and fixed it entirely by raising a *consumer-side* `scx_cache_shards`
from 16 to 48 — "config-only; the prefetch threads were never the cap". SCX's own default is
128 and would have been fine. What SCX lacked was any way to *see* it: `cache_metrics()`
already carried hits / misses / evictions and nothing interpreted them, so diagnosis was a
bisection.

**The headline result reframes the problem.** On **framed** files — row-group framing is the
v4 default since F5 — a scattered cell-set gather **never touches the whole-shard LRU at
all**. Captured cold on `census_500k_auto.scx` (31 CSR shards):

| `cache_shards` | sets/s | peak RSS | hits+misses | `full_shard_groups` | `block_index_groups` |
|---|---|---|---|---|---|
| 16 (undersized) | 6.7 | 1,181 MB | **0** | 0 | 1,307 |
| 31 (suggested) | 6.7 | 1,181 MB | **0** | 0 | 1,307 |

Every request goes through the codec-agnostic block-index path, which decodes only the
touched row-groups. `cache_shards` cannot matter, and a 16-vs-31 comparison returns 1.00×
**by construction** — which is why the benchmark records `read_path.lru_consulted` alongside
the ratio. A bare 1.00× would read as "sizing doesn't help"; the correct reading is "the
cache was bypassed".

**With the block-index path disabled** (`SCX_SCATTER_BLOCK_INDEX=0` — the legacy
full-shard-decode regime STATE3 was actually in), the same file and the same plans, cold:

| `cache_shards` | sets/s | peak RSS | `lru_consulted` | `full_shard_groups` |
|---|---|---|---|---|
| 16 (undersized) | **0.2** | 19,762 MB | yes (2,013 reads) | 1,307 |
| 31 (suggested, `max_memory_mb=6415`) | **546.4** | 10,557 MB | yes | 1,307 |

**2,486× throughput and half the peak RSS.** Thrash is not a speed/memory trade — it is
worse on both axes, because a cache that cannot hold the working set spends the run
allocating, decoding and freeing 193 MB shard buffers. The 19.8 GB figure is also close to
the ~23 GB STATE3 reported, which is corroboration that this reproduces their regime rather
than a synthetic one. (An earlier warm interactive probe of the same comparison gave 269×;
the cold capture is the citable number — a warm smoke is not a measurement, least of all for
a ratio.)

The signature is exactly what the detector keys on: at 16 shards nearly every miss
*displaces a live entry* (0.98–0.99 evictions/miss), whereas the well-sized cache evicts
nothing. This is why the predicate requires `evictions ≈ misses` and not merely a high miss
rate — a cold sequential scan also misses on ~100% of reads while evicting nothing, and is
perfectly healthy. `suggested_cache_shards` derived the right number (31) from the plan
alone, with **zero I/O**, via the pre-existing `BackedCsrIndex::shards_for_indices`.

One more thing this shows, which was not the point of the experiment: at
`cache_shards=31` the full-shard path reaches **546 sets/s against the block-index path's
6.7** on the same file. Once the whole file is resident and decoded, every subsequent batch is
served from RAM, whereas the block-index path — as captured here, before OPT-FORMATIO-1 — re-decoded row-groups on each visit. That is
**conditional on the working set fitting** — 31 × 193 MB fits the 6.4 GB budget granted here
and would not at atlas scale — so it is not an argument for changing the default. It does mean
the block-index path is not universally faster, and a caller with a small file and repeated
scattered access has a real reason to pass `scatter_block_index=False` with a
plan-sized cache.

Two conclusions worth stating plainly:

1. **F5 row-group framing already designed this pathology out of the default path.** The 1A
   diagnostic's domain is unframed/legacy layouts and callers who pass
   `scatter_block_index=False` — not modern default files.
2. Following the advice means raising **both** knobs. `suggested_cache_shards` returns a
   count, but the byte budget must also hold that many shards: census_500k wants 31 while
   the adaptive 4 GB cap affords 22, so `cache_shards=31` alone under-delivers. The warning
   text says so, and the benchmark's "sized correctly" arm sets both.

**Capture provenance.** Two serialized SLURM jobs on one `cpu` node (16 CPU / 200 GB),
`cold_fadvise` before every timed run, 3 runs per arm, chained
`sbatch --dependency=afterany` so no two arms of a timing A/B were ever co-scheduled:
`candidate_2026_07_30_dataload_phase1_{blockidx,fullshard}`. The `fullshard` job took
**4 h 32 m** against the `blockidx` job's **35 m** — the wall-clock gap is itself the thrash
signal. Not floored in `thresholds.yaml`: these arms deliberately *misconfigure* one side, so a
floor on `cache_undersized` would gate on a number the code is trying to make impossible.

**One caveat on the capture, stated rather than buried:** the runtime thrash *warning* emitted
**zero** times during that 4.5-hour thrashing run, because the sampler checked every 32 batches
while `cellset_gather` runs 30 — the diagnostic was silently dead on exactly the workload it
was built for. Fixed (cadence 8, plus a 30-batch regression test) and re-verified interactively
on the same census_500k fullshard configuration, where it now fires once with the correct
diagnosis. The throughput A/B above is unaffected — it measures the two configurations, not the
warning — but the warning's own evidence is the test suite plus that targeted re-run, not this
capture. A second finding from the same re-run: the suggested size was derived from shard
*requests* per batch, which overcounts under thrash (it advised 174 on a 31-shard file), and is
now capped at the file's shard count.

**Budget-default reconciliation.** The three loader classes had three policies —
`TrainingDataset` adaptive (512 MB floor → 4 GB cap), `IndexPlanDataset` a hard 512 MB that
silently shrank `cache_shards` toward 1, and `SparseCellSetDataset` no byte cap at all
(`usize::MAX`; STATE3 observed ~23 GB RSS). All three now resolve `max_memory_mb=None`
through the adaptive policy, so a `None` budget is bounded everywhere. The auto-tune's
*behaviour* is deliberately unchanged — it may still reach `cache_shards=1` — because
refusing would turn configurations that work today into hard errors. What changed is that a
reduction is now reported.

That report is deliberately quiet under an adaptive budget unless the surviving cache falls
below 8 shards. A preflight on pbmc10k (≈194 MB shards) showed the first version warning that
"max_memory_mb=4096 affords only 21 of 128" on a run whose observed peak RSS was 0.7 GB —
i.e. firing on a healthy default configuration, which is how a diagnostic gets filtered and
stops working. An *explicit* `max_memory_mb` that conflicts with an explicit `cache_shards`
is always reported: the caller asked for two things that don't fit, and only they can decide
which gives.

**One budget model (ORG-9.10-5).** Resolving `None` the same way left the three classes
still *meaning* three different things by `max_memory_mb`, because each kept its own tune
loop. They now share one `BudgetModel` trait and one `tune()` driver, which unified three
things that were quietly different:

- **The mmap'd file is no longer budgeted anywhere.** The sequential model counted the whole
  file against the budget while the two plan-driven models never did. Since the adaptive cap
  is 4 GB, *any* file above it exceeded its budget on that term alone and collapsed to
  `batch_size=64, shard_group_size=1, prefetch_batches=2` with `budget_exceeded` set,
  whatever the caller asked for — `benchmarks/comprehensive/benchmarks/ml_loader.py` carries
  a SLURM-memory-scaling workaround written for exactly that. Page cache is evictable under
  pressure; it is now reported (`mmap_mb`) and never budgeted.

  **The cost, measured.** A two-arm A/B on the small tier (`ml_loader`, `scx_auto`, 4
  datasets, main-arm capture as the baseline). Both captures are checked in —
  `results/mainarm_9e_20260830/` and `results/branch_9e_20260830/` — per
  [benchmark_manifest.md](../benchmark_manifest.md); they are an **A/B against each other** on a
  partial-fixture Lambda node, not a promoted `baselines/LATEST` capture, and their
  `environment.json` records `git_dirty: true`. What those two summaries report:

  | `ml_loader` / `scx_auto` | main arm | branch arm | Δ peak RSS |
  |---|---:|---:|---:|
  | `tabula_sapiens_100k` | 595.53 MB | 657.55 MB | **+10.4%** |
  | `pbmc3k` | 395.28 MB | 449.01 MB | **+13.6%** |
  | `pbmc10k` | 389.44 MB | 394.25 MB | +1.2% |
  | `smartseq2` | 498.50 MB | 483.22 MB | −3.1% |

  **0 timing regressions, 0 file-size, 0 fingerprint mismatches** across all four.

  The `tabula_sapiens_100k` rise is the intended mechanism, visible in the tuned config: the
  mmap term no longer forces a reduction, so `shard_group_size` goes 3 → 4 (default) and
  4 → 5 (`hvg_indices`), holding more decoded shard buffers resident.

  The `pbmc3k` row is **not** attributable to the change: at 4 MB the mmap term never bound,
  and both arms tune to an identical `(shard_group_size, prefetch_batches, batch_size)` of
  `(1, 4, 1024)`. A separate controlled measurement — three fresh processes per arm on one
  node — put the two within ±1% (422.3 → 422.9 MB default, 189.8 → 188.0 MB with
  `hvg_indices`). ⚠️ **That control is not one of the checked-in captures**: it is a local
  measurement, not a manifest entry, and +13.6% is what the committed A/B says.

  So the trade is explicit: the loader stops shrinking the buffers a caller asked for in
  order to pay for evictable page cache, and uses more anonymous memory for it. A caller who
  wants the old footprint sets `max_memory_mb` to the value they actually want enforced,
  which is now what that argument means.

- **`SparseCellSetDataset` now budgets the interpreter constant it reports.** Its tuner
  passed `non_cache_bytes: 0`, so `memory_budget()["breakdown"]["total_bytes"]` could exceed
  the budget the tuner had just checked, and it handed the cache the raw request and raw
  byte budget rather than the tuned ones. The cache is correspondingly smaller. Note what
  this does **not** buy: the gathered batch and its transients are still uncharged (this
  path has no `max_plan_size`), and `WeightedLruCache` keeps a single oversize shard rather
  than refusing to cache it, so `max_memory_mb` bounds the cache this loader sizes, not
  process RSS.
- **`MultimodalTrainingDataset` reports its per-modality split.** The nnz-proportional
  division has a 64 MB floor (a share below it fails validation), so two modalities at
  `max_memory_mb=64` budget 128 MB between them. That was always true and never reported;
  `memory_budget()["effective_total_mb"]` now says so, and an explicit request that gets
  rounded up warns.

The exhaustion *policy* stays per class, deliberately: the sequential path flags
`budget_exceeded` and continues (and now raises a `UserWarning`, where it used to only write
a log line), `IndexPlanLoader` refuses construction, and `SparseCellSetLoader` bottoms out at
a one-shard cache and warns.

## Obs categorical codes without pandas (data-load Phase 1, 1C)

The Phase-0 cold breakdown put **96–99.8% of per-file obs cost in obs reading**, not in
`File::open` / catalog parse / BLAKE3 (0.2–3.8% and at-or-below-noise respectively). The
accessor aimed at that term is `obs_categorical(col) -> (codes, categories)`: what every
model's vocabulary/one-hot setup actually wants, returned as an `int32` numpy array plus a
list of strings.

It differs from `read_obs(columns=[col])` in two ways that both matter at manifest scale.
It skips the Arrow-IPC-bytes → pyarrow → `to_pandas()` round trip entirely; and it folds
**one shard at a time** into a running global dictionary, so the column is never
concatenated — peak memory is `n_obs × 4 B` plus the vocabulary, against a full materialised
Arrow column for `read_obs_keys` → `concat_batches` → `unify_dictionary_columns`.
`obs_categorical_many` runs N accumulators over **one** shard pass, so resolving four
covariate columns across a 26k-file manifest costs 26k projected reads rather than 104k.

**Per-file cost, cold, seconds/file** (one obs column; `read_obs` = `read_obs(columns=[col])`
→ pandas, `obs_categorical` = the numpy accessor). h5ad's column is
`adata.obs[col].cat.codes`, i.e. what a consumer writes today.

| dataset | scale | SCX `read_obs` | SCX `obs_categorical` | speedup | h5ad `read_obs` | h5ad `.cat.codes` |
|---|---|---|---|---|---|---|
| census_500k | 1 file | 0.5546 | **0.3060** | **1.81×** | 0.6593 | 0.6962 |
| tabula_sapiens_100k | 1 file | 0.0576 | **0.0242** | **2.38×** | 0.1995 | 0.1972 |
| tabula_sapiens_100k | 256-file manifest | 0.00666 | **0.00576** | 1.16× | 0.1117 | 0.1128 |

**Multi-column, cold** — the shape state3's `_setup_global_maps` actually issues, and the
measurement that validates the "one shard pass for N columns" claim rather than asserting it.
4 columns, `obs_categorical_many` against N separate `obs_categorical` calls:

| dataset | 4 × separate | `obs_categorical_many` | speedup |
|---|---|---|---|
| census_500k | 0.658 s | **0.245 s** | **2.69×** |
| tabula_sapiens_100k | 0.015 s | **0.005 s** | 3.16× |

Sub-linear in the column count, as intended: the per-shard projected read is paid once, not
per column. (tabula's absolute numbers are small enough to be near the noise floor on a
100k-cell file; census_500k is the solid figure.)

**1.8–2.4× at single-file scale** for one column, and cross-format `obs_categorical` is
**8.2×** h5ad's `.cat.codes` at tabula_sapiens_100k (0.0242 vs 0.1972) and 2.2× at
census_500k. Two honest qualifications:

- **The h5ad column shows ~1.0× because pandas is pandas.** Taking `.cat.codes` off an
  already-materialised `adata.obs` costs nothing extra — the expense was materialising it. The
  SCX win comes from never building the frame at all, which is a choice only the SCX side can
  make.
- **The win shrinks to 1.16× at manifest scale**, where per-file fixed cost dominates: the
  256-file fixture shards one 100k-cell file, so each file's obs is tiny and 0.0058 s/file is
  mostly open + catalog. The accessor is aimed at the *per-column assembly* term, and that term
  is only large when a file's obs is large. At census_500k — one file, 500k rows — it is 1.81×.

Peak RSS moves modestly on these fixtures (751 vs 780 MB at census_500k; 480 vs 488 MB at
tabula), because a single narrow column is not where the concatenation transient bites. The
memory argument is structural rather than demonstrated here: the fold never holds more than one
shard's projected column plus the running vocabulary, so it does not scale with `n_obs × width`
the way `concat_batches` does.

One contract worth stating loudly: `codes` follows `read_obs()`'s row space. By default
(`logical=True`, since 0.17) it is indexed in the **live** obs row space, `len(codes) == n_obs`,
aligned with `read_obs()` and `to_anndata(backed=True).obs`; `logical=False` gives the
**physical** space, `len(codes) == n_obs_physical`, deleted rows in place. On a file with
deletion vectors those differ, and indexing one space's codes by the other's row ids addresses
the wrong cell — a correctly *shaped* array of wrong rows — so both are pinned by a test
(`test_obs_categorical_row_space`) rather than left to the docstring.

Semantics are pandas-compatible by choice: null → code `-1`, so a literal `"NaN"` *string*
stays a real category. This deliberately differs from `scx_loader`'s training-internal
`CategoryDict`, which appends a synthetic trailing `"NaN"` level and documents that it is
not `pandas.Categorical.codes`-stable. Category order is **first-seen**, not lexicographic.
Both on-disk encodings are accepted, including a file that carries both across its shards.
Every write door now writes a categorical as a dictionary, but a file an older `append`
or `merge` grew is mixed, and so is one appended to from a plain-obs source (these ops
preserve the representation they are handed rather than promoting a plain column).

On the cloud path the same pass lands, plus the projection half that was missing:
`CloudExperiment.read_obs(columns=)` previously assembled the entire obs table and then
projected it in memory (its own docstring conceded "the network cost is the full obs metadata
regardless"). It is now a genuine per-shard pushdown, so the network cost is the requested
columns' bytes.

## Count-depth downsampling in the gather (data-load Phase 1, 1B)

**The headline is a null result, and it is the useful one: moving the per-cell downsample draw
from numpy into Rust is a wash.** Captured cold on `cellset_gather` / `scx_auto`, S=64 random
plans, `target_library_size=2000`, multinomial, `n_runs=3` (per-run values were identical to
the tenth across all three runs in every arm):

| dataset | baseline, no downsample | gather + numpy per-cell draw | draw inside the gather | ratio |
|---|---|---|---|---|
| census_500k | 6.7 sets/s | 6.5 sets/s | 6.4 sets/s | **0.99×** |
| tabula_sapiens_100k | 5.0–5.1 sets/s | 5.3 sets/s | 5.2–5.3 sets/s | **1.00×** |

Both arms move the same cells over the same plans, so the ratio isolates the draw. The draw is
not where the time goes: a scattered cell-set gather is decode-bound at 5–7 sets/s, and a
per-cell multinomial over a ~1.4k-nonzero row is ~1% of that in either language. Downsampling
*at all* costs about 4% (6.7 → 6.4 at census_500k); which implementation does it is noise-level,
and the Rust side is if anything a hair slower — plausibly the per-row `Vec<u64>` trial buffer
and a fresh ChaCha8 rekey per row against numpy's vectorised multinomial. **That 1% was not
chased**, because a 1% term on a decode-bound path does not justify the complexity.

**So what 1B buys is capability, not throughput** — and that is the claim to carry. Before it,
a consumer using the native collate kernel could not downsample at all: the kernel's consumer
owns query sampling, which reads the gathered counts, so the draw has to land upstream of it or
the query is sampled from pre-downsample expressed genes while the numerics use post-downsample
counts. state3 refused the combination outright rather than get that wrong. The number above
says the refusal cost nothing to lift.

Note what the table is *not*: it is not "what a consumer pays today". A consumer that wants
downsampling today cannot use the native kernel at all and falls back to a Python collator,
which is far more expensive than the draw. This arm deliberately isolates only the draw, so the
comparison is honest about the one thing it measures.

**The unconditional negative clip did not regress the gather.** 1B also clips negatives in the
emitted CSR (previously they passed through while the collate kernel clipped them lazily per
read, so the two disagreed about a row's contents). Against the Phase-0 numbers on the same
fixture and plan shape: census_500k random S=64 **6.6 → 6.7 sets/s** (423 → 427 cells/s),
grouped S=64 **16.1 → 18.0** (1,028 → 1,150 cells/s). Neither direction is claimed as a change
— the point is that a per-nonzero pass added to every gathered row is not measurable beside the
decode, and the existing `cellsets_per_sec__gather_random` floor (min 5.61 at census_500k)
already gates that path if it ever becomes measurable.

**No new `thresholds.yaml` floor**, deliberately, and the reason matters more than the omission:
the ratio must never gate, because the `downsample_python` arm is a deliberate strawman whose
job is to be slower; and the absolute `downsample_rust` rate has exactly one capture, so any
threshold would be guesswork. The gather floors already cover the path the clip touches. This
follows the Phase-0 convention of leaving a metric unfloored rather than setting a floor that
fires on noise and trains operators to ignore the gate.

The 1A cache-sizing arm re-ran in the same wave and reproduced its earlier result exactly
(census_500k 6.7 vs 6.7 = 1.00×, `lru_consulted: false`); tabula_sapiens_100k recorded
`applicable: false` because a scattered plan there touches 7 shards against the 16-shard
undersized arm, so both arms hold the whole working set — the fixture cannot demonstrate the
effect, which the metadata now says out loud rather than reporting a bare 1.00×.

Reproducibility, for the record: the draw is keyed on
`(seed, method, blake3-64(canonical path), row)`, so it is invariant to manifest order, to a
manifest subset, and to I/O and thread scheduling — but it is **not** bit-reproducible against a
numpy-based implementation (ChaCha8 vs PCG64 + BTPE), so counts from a previously downsampled
run change when it moves onto this path. Distributions and target semantics do not.

## Global pre-shuffle (data-load Phase 1, 1D)

**`scx sort --shuffle` does what it exists to do and costs nothing to read: per-shard label
divergence from the corpus mix collapses by 27–43×, while cache-cold `TrainingDataset`
throughput is unchanged (1.01–1.03×). The one real cost is on disk, and on the default
`codec="auto"` it is large — 1.86–2.09× on X.** Captured cold on `shuffle_layout` / `scx_auto`,
`seed=42`, `n_runs=3` for the epochs and 2 for the rewrite, one job at a time
(`--dependency=afterany`) on a 16-core / 200 GB `cpu` node.

### Batch mixing — the thing the feature is for

`TrainingDataset` randomizes in two levels and both are bounded by physical layout, so on a
clustered file batch composition is capped by `shard_group_size`. The metric below is the
**total-variation distance between one CSR shard's `cell_type` mix and the corpus mix** — the
composition a `shard_group_size=1` batch would inherit. It is computed analytically from
`obs_categorical` codes plus the shard row ranges: no X decode, no RNG, no sampling error.

| dataset | levels | block | per-shard TV | pooled 8-shard TV |
|---|---|---|---|---|
| tabula_sapiens_100k | 33 | 14,286 rows | 0.562 → **0.013** (43×) | n/a — 7 shards |
| census_500k | 885 | 16,130 rows | 0.625 → **0.023** (27×) | 0.257 → **0.007** (36×) |

The pooled figure is deliberately **withheld** for `tabula_sapiens_100k`: with 7 shards,
pooling 8 blocks *is* the whole corpus, so the answer is 0.0 by construction on any row order —
including a maximally clustered one. Reporting it would have said "perfectly mixed" about the
file the feature had not yet touched.

The `census_500k` pair is the more informative one. Even at the loader's default
`shard_group_size=8`, an unshuffled file's groups sit at TV 0.257 from the corpus; after the
pre-shuffle they sit at 0.007, i.e. at the sampling floor. That is the ceiling the pre-shuffle
removes.

### Read throughput — a neutrality check, not a speedup

| dataset | unshuffled | shuffled | ratio |
|---|---|---|---|
| tabula_sapiens_100k | 11 911 samples/s | 12 296 samples/s | 1.032× |
| census_500k | 14 515 samples/s | 14 650 samples/s | 1.009× |

**Read ~1.0× as the arm passing.** A shuffled file is still streamed sequentially, so
neutrality is the expected and desired result; what this arm exists to catch is a *regression*,
evidence that the permutation cost the sequential read something. It did not. The throughput a
pre-shuffle actually buys is downstream — a consumer can drop `shard_group_size` and its memory
budget without losing batch diversity — and is not measured here.

### Output size — pin your codec

X-section bytes only (obs, var and the predicate index all move under a reorder too; isolating
X is what makes this a statement about *compression*). Each variant is shuffled **at its own
codec and its own shard geometry**, so the only variable is row order:

| codec | tabula_sapiens_100k | census_500k | codec flipped? |
|---|---|---|---|
| `scx1` | 1.000× | 1.001× | no |
| `lz4` | 1.002× | 1.008× | no |
| `shufdelta` | 1.005× | 1.011× | no |
| `zstd` | **1.060×** | **1.121×** | no |
| `pcodec` | 1.060× | 1.121× | no |
| **`auto`** (the default) | **2.088×** | **1.857×** | **yes — `shufdelta` → `scx1`** |

Three separate findings, and conflating them is the easy mistake:

1. **The `auto` row measured a bug, since fixed — it is no longer reproducible.** The
   per-shard codec histogram is recorded before and after, and it moved `shufdelta ×N →
   scx1 ×N` on both datasets, the entire 1.86–2.09× being that switch. The cause was not
   the reorder: every derived-file op built a `FramingConfig` whose `decode_target` was
   `None`, which *is* the `fast` profile, so `auto` on any rewrite ran the single-encode
   heuristic and never re-adopted `ShufDeltaZstd`. The same bug made `scx compact` grow a
   file it was asked to shrink. Post-fix, `auto` on `sort`/`subset`/`merge`/`compact` runs
   the same adaptive per-shard selection `scx convert` does; a re-measurement on
   `census_500k` is in [sharding.md § Shuffling for training](../sharding.md#shuffling-for-training-scx-sort---shuffle).
   The rows below were captured before that fix and are kept as the record of it.
2. **`zstd` genuinely loses cross-row redundancy — 6% at 100k cells, 12% at 500k.** No flip;
   this is the real effect, and it grows with shard occupancy, which is what the mechanism
   predicts. It is also an order of magnitude smaller than the flip.
3. **`scx1` is exactly neutral**, as its design implies: each row's gene indices are coded
   independently of row order, so a permutation relocates identically-sized blocks.

**The remediation, twice revised, is now "no pin at all".** The first version said to pin
`scx1`, which is neutral only *relative to an `scx1` input*; on the `shufdelta` files above it
is precisely what `auto` flipped to, so that advice reproduced the 2.09× rewrite rather than
avoiding it. The second said to pin the input's own codec — correct against the bug, but
obsolete once the bug was fixed, and worse than `auto` on a `zstd` input that `auto` would now
re-encode smaller. Leave `--codec` at `auto`; pin a codec only to *choose* an encoding (`scx1`
for a permutation-invariant layout or the GPU device-decode route), never to hold size. The
pre-rewrite warning was narrowed to match: it now fires only for codecs whose compression
spans rows, states the inherent 6–12% zstd cost, and recommends nothing.

Reproduce with `benchmarks/comprehensive/benchmarks/shuffle_layout.py` (SCX-only, `scx_auto`
trigger, `_NO_CONVERSION`); the capture harness is `/scratch/…/scx-1d-capture/`.

The methodology matters more than usual here, because two earlier versions of this measurement
were wrong in ways that looked plausible. Shuffling at the writer's *default* shard size turned
a 6-shard input into a 1-shard output, so the "size ratio" was measuring re-sharding and the
throughput arm reported a spurious 5.97×. And leaving the output codec at `auto` re-encoded
every variant identically — all six landed on byte-identical output — so the "codec sweep"
measured auto-reselection rather than the codec. Both are now controlled and asserted
(`shard_geometry_matched`, `codec_flipped` in the recorded metadata).

### Rewrite cost

| dataset | wall | throughput | peak RSS | output |
|---|---|---|---|---|
| tabula_sapiens_100k | 5.6 s | ~18.0k cells/s | 3.2 GB | 446.8 MB |
| census_500k | 21.1 s | ~23.4k cells/s | 9.5 GB | 1 833.1 MB |

One-off, and it is the in-memory strategy (no `--memory-budget`), which gathers the whole X.
Pass `--memory-budget` for the bounded external-partition path on files that do not fit.

**No new `thresholds.yaml` floor, and the reasons differ per arm** — recorded in that file's
deferred-floors block rather than left implicit. The size ratios are a property of the *data
and codec*, not of the implementation, so a floor would fire on a fixture refresh rather than
on a regression. The throughput arm must never gate on its ratio, since the interesting
direction is a regression and `ooc_loader`'s `samples_per_sec__raw` floors already cover
`TrainingDataset` throughput on these same datasets. `label_tv_block__after` is the honest
candidate — a shuffle that stopped mixing would show up there and nowhere else — but one
capture cannot separate the sampling-noise floor (a function of shard size and label
cardinality) from a real regression; floor it after a second capture on the same fixtures.

Reproducibility, for the record: the permutation is a Fisher–Yates shuffle over the **live**
rows seeded by `splitmix64(splitmix64(seed) ^ SHUFFLE_DOMAIN_TAG)`, and the seed is written to
provenance because it is the only record of the order — there is no key to re-derive it from.
The domain tag exists so a `seed=42` shuffle is uncorrelated with the training loader's
`(seed=42, epoch=0)` shard permutation, `42` being the default on both surfaces. Deletions are
materialized away first, so the same seed yields a different order on a file with deletion
vectors than on the same file without them. The order is **not** stable across a `rand` crate
upgrade; that is pinned by a literal-permutation test rather than promised by the format.
