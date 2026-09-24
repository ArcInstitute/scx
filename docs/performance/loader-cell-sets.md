# Training loader: cell-set gathers

> Part of [SCX performance](README.md). See also the [training loader overview](loader.md).

## Tier-1 loader fixes: gather pre-sizing and the crop mask (phase 1)

Two changes to `SparseCellSetDataset`, measured as a two-build A/B
(SLURM `2949908`, host `GPU71BA`, `main` `0870ac39` against
`phase1-tier1-loader` `19ec5256`; raw rows under
`results/raw/phase1_tier1_loader_ab.json`).

The measured `after` build carries **all** of phase 1, W4 included, so the
blocking-thread cap is in the timed arm even though the two changes below are
what the table is about. `IndexPlanDataset`, the other consumer of that shared
cap, was not timed at all.

- **The gather pre-sizes its CSR outputs.** `indices` and `data` used to grow
  from empty, reallocating and copying at every doubling. They are now sized up
  front from `rows x mean_nnz_per_row` (catalog stats, no I/O), biased up by an
  eighth.
- **The collate kernel's withheld-gene test is per set.** `collate_cell` rebuilt
  a `HashSet` of withheld ids for every cell out of the `k_dec` ids its whole
  *set* shares; it now sorts that panel once per set and binary-searches it.

| dataset | metric | `main` | phase 1 | ratio | rounds won | p |
|---|---|---|---|---|---|---|
| pbmc3k | `cellsets_per_sec__gather_random` | 4692 | 10210 | **2.20x** | 12/12 | <0.001 |
| pbmc3k | `cellsets_per_sec__gather_random_s512` | 558.0 | 1063 | **1.93x** | 12/12 | <0.001 |
| pbmc3k | `cellsets_per_sec__gather_grouped_s512` | 506.9 | 821.7 | **1.59x** | 11/12 | 0.006 |
| pbmc3k | `cellsets_per_sec__gather_grouped` | 4961 | 7051 | **1.56x** | 12/12 | <0.001 |
| pbmc3k | `us_per_cell__collate` | 13.69 | 12.84 | 1.06x | 12/12 | <0.001 |
| tabula_sapiens_100k | `cellsets_per_sec__gather_grouped` | 661.6 | 839.5 | **1.27x** | 12/12 | <0.001 |
| tabula_sapiens_100k | `cellsets_per_sec__gather_random` | 646.4 | 837.1 | **1.26x** | 12/12 | <0.001 |
| tabula_sapiens_100k | `cellsets_per_sec__gather_random_s512` | 84.55 | 93.45 | 1.10x | 10/12 | 0.039 |
| tabula_sapiens_100k | `cellsets_per_sec__gather_grouped_s512` | 82.85 | 86.75 | 1.05x | 8/12 | 0.388 |
| tabula_sapiens_100k | `us_per_cell__collate` | 22.30 | 21.75 | 1.02x | 10/12 | 0.039 |

Ratios are oriented so `>1` always means faster. The last tabula row is **not a
result** — 8 of 12 rounds is what a coin does — and is listed so the one metric
without a reliable difference is visible rather than omitted.

### The crop mask is reliable and small, which is not what was predicted

The per-set panel was expected to be worth ~25%, on the strength of an earlier
measurement that collating with an empty mask ran at 78.7 us/cell against 120.1
with 25% of query positions withheld — ~34% of collate wall. It is worth **6% on
pbmc3k and 2% on tabula**.

The earlier figure was sound but measured the wrong thing for this purpose: it
covered the `HashSet` **build and its probe together**, and only the build is
removed. The probe became a binary search — about `log2(k_dec)` ≈ 10 comparisons
per surviving gene against one hash — so on cells with many genes it costs more
than it saves, which is why tabula (~1950 non-zeros per cell) gains less than
pbmc3k. Removing the per-row allocation is the durable part.

The change that would collect the rest is a merge: walk the cell's own sorted
`gene_ids` against the sorted panel once, `O(n + k_dec)` with no per-gene search.
That needs a per-row scratch buffer to mark the withheld genes in, and
allocating one per row would reintroduce exactly the allocation this removed —
so it belongs with the caller-supplied scratch the tokenisation kernels
introduce, not here.

### Why the numbers are paired, and what the first attempt got wrong

The arms alternate **per round**: each round gathers once on each build, back to
back on the same dataset, and the statistic is the median of the within-round
ratios plus a sign test over them. The within-pair order flips on alternate
rounds.

That is not ceremony. The first version of this capture compared arm-sized
blocks and produced two mutually contradictory answers depending on which node
SLURM picked. On an idle node the within-arm spread was ~1% and the signal was
clean; on a node shared with a dev shell and a co-tenant array job it reached
7-34%, and the two metrics that cleared the noise bar disagreed with the quiet
run — one of them reporting a 1.85x speedup as a 0.81x regression. Neighbour
load drifts on the timescale of a block, so a block design lays it directly on
top of whichever arm was running.

Pairing cancels drift that is slow compared to one round, because it moves both
members of a pair together. The sign test then answers the question that
survives heavy noise — did the change win more rounds than chance allows —
rather than the one that does not, which is by how much. It is also what makes
the 6% collate figure reportable at all: a 6% median ratio is exactly the
magnitude a block design discards as noise, and 12 wins out of 12 is not.

### What this does not say

- **The committed rows are reduced records, not `BenchmarkResult`s.** Each
  round's real `BenchmarkResult` was written inside the arm's scratch worktree,
  which the job removes on exit; what is committed is the per-round metric
  scalars the driver copied out. The table recomputes exactly from them — medians,
  ratios, sign counts and p-values — but the file does not carry the
  `schema_version` / `system` / `runs` envelope
  [docs/benchmark_manifest.md](../benchmark_manifest.md) describes. The driver now
  preserves each round's full record, but the folded envelope is still not
  manifest-shaped at the top level — a conforming artifact needs one result per
  (arm, dataset) with the paired rounds in `runs[*].extra`. This is an open gap,
  not compliance.
- **Peak RSS was not captured** in this A/B. The pre-size bias adds 12.5% of a
  batch's CSR payload — on the pbmc3k arm, a measured capacity of 975,737
  elements against 870,886 used, so 0.84 MB per batch. That is arithmetic from
  the capacities, not a measurement of process RSS.
- **No census.** A `cellset_gather` census cell does not finish: `census_500k`
  was killed at 205 minutes still on run 1 of 3 of its `cache_undersized`
  scenario. That is the same reason the collate floors were prescribed on
  tabula alone.
- Single node, single rank, one format (`scx_auto`).

## Bounded reader registry: what a large manifest actually costs (phase 2)

`SparseCellSetDataset` takes a list of paths. Before phase 2 it opened all of
them in the constructor and held them for the dataset's lifetime; `reader_limit`
now caps how many are resident and reopens the rest on demand. Default `None`
keeps the old behaviour.

**The resource being bounded is resident memory, not file descriptors.** That is
worth stating first because the opposite is the intuitive answer and it is
wrong. Opening an SCX file mmaps it and closes the descriptor, and the loader's
readers do not watch their files, so they retain none. Every arm below records
the process's descriptor count before and after; it is **7 in all twelve**, at
every manifest size and every limit. A 5,000-file manifest constructs *and*
gathers under `ulimit -n 1024` on unmodified `main`.

What an open reader does cost is the parsed catalog — one owned entry per
catalog entry, so it scales with shards per file, not cells. Measured per open
reader, `n_files = 1000`:

| fixture | CSR shards | `None` | 256 | 64 | 16 |
|---|---|---|---|---|---|
| `tabula_sapiens_100k` | 7 | 102.0 MB (104.4 kB/file) | 24.7 MB | 6.2 MB | 1.6 MB |
| `census_1m_preprocessed` | 62 | 117.8 MB (120.6 kB/file) | 26.3 MB | 6.6 MB | 1.7 MB |
| synthetic, 100 shards | 100 | 151.7 MB (155.3 kB/file) | 31.9 MB | 8.1 MB | 2.0 MB |

`reader_limit=16` against the default is **66× / 71× / 76×** (release build).
Two runs of the capture agreed to 0.1 MB on every arm except the smallest, whose
~2 MB delta is near the resolution of a `VmRSS` reading and moved the third
ratio between 74× and 76× — read that column as "about 75×", not as three
significant figures. The mmap count
falls with it exactly — 1000 → 256 → 64 → 16 additional VMAs — which matters at
the second, looser wall: `vm.max_map_count` is 65,530 by default, so an
unbounded manifest also stops working somewhere near 64k files.

Splitting the ~105 kB: an `ScxReader` alone is 91.8 kB and the
`BackedCsrReader` wrapper adds ~9 kB. **Over 90% of the cost is the catalog**,
which is why an eviction drops it and a reopen re-parses it (0.09–20 ms per
file, from the per-file open measurements above). Retaining
`Arc<FullCatalog>` and reopening through `open_with_shared_catalog` — the
obvious way to make reopens cheap — would have reclaimed the 9 kB and left the
100.

The default arm carries ~1 kB/file more than a reader strictly needs: the shard
index is retained per file whatever the limit, because the alternative — reading
it back through the resident handle — puts a mutex acquisition on a path that
runs once per *row* of every plan. 0.9% of the per-file cost to keep plan
bucketing lock-free.

Extrapolated to the manifest size this exists for, 26,453 files: **~2.8 GB
(tabula-shaped) to ~3.2 GB (census-shaped) per process**, before multiplying by
DataLoader workers and ranks. That is arithmetic on the per-file figures, not a
26k-file measurement — no such capture was run.

Raw rows: `results/raw/phase2_reader_registry/manifest_rss.json`, produced by
`benchmarks/scripts/measure_reader_registry_rss.py`.

**What this does not say.** These are allocation counts around one constructor,
not throughput. Nothing here measures what a bounded limit costs a *gather*: a
plan that fans across more files than the limit reopens on every batch, and no
arm at a bounded limit was timed, so no floor is proposed for one.

The default `reader_limit=None` path raises a separate question these numbers do
not answer, and it is answered below.

### Does the leased-`Arc` seam cost anything? (phase 2)

The engine now hands out a leased `Arc` from a mutex-guarded map where it
previously indexed an array, and sizing a plan takes a lease per touched file.
Two-build A/B, SLURM `2951009`, host `GPU3694`, `main` `aba3c114` against
`phase2-reader-registry` `9833d075`, 12 interleaved rounds per dataset, median
of within-round ratios with an exact two-sided sign test. Raw rows under
`results/raw/phase2_reader_registry/cellset_gather_ab.json`.

⚠️ **The A/B measures `9833d075`, not the branch head.** Two later commits
changed lease ordering on the prefetch path — sizing now reuses the leases the
prefetcher already holds, and a plan touching more files than `reader_limit` is
not prefetched at all. Neither can fire in the arms measured here, which all run
`reader_limit=None`: an unbounded registry never evicts, so there is nothing to
reorder and no cap to exceed. The measurement stands for the seam it names; it
is not a measurement of the bounded path, which was never timed.

Every arm runs `reader_limit=None`, where nothing is ever evicted or reopened.
Flat was the expected result and the gate.

| metric | pbmc3k | tabula_sapiens_100k |
|---|---|---|
| `us_per_cell__collate` | 1.00× (6/12, p 1.000) | 1.03× (9/12, p 0.146) |
| `cellsets_per_sec__collate_rust` | 1.00× (6/12, p 1.000) | 1.03× (9/12, p 0.146) |
| `cellsets_per_sec__gather_random` | 0.996× (6/12, p 1.000) | 0.992× (5/12, p 0.774) |
| `cellsets_per_sec__gather_grouped` | 0.987× (5/12, p 0.774) | 0.999× (6/12, p 1.000) |
| `cellsets_per_sec__gather_random_s512` | 1.02× (8/12, p 0.388) | 0.971× (4/12, p 0.388) |
| `cellsets_per_sec__gather_grouped_s512` | 1.01× (8/12, p 0.388) | 1.03× (9/12, p 0.146) |

Ratios are oriented so > 1 means the change helped. **All twelve cells are "no
reliable difference"** — every sign test is p ≥ 0.146, and no median ratio
leaves 0.97–1.03.

Read that as a resolution, not as proof of zero. Twelve paired rounds can rule
out an effect of roughly the size the spread here admits; they cannot
distinguish 1.00× from 1.01×. The claim this supports is the one the gate asked
for — the seam costs nothing a consumer would notice at the default — not that
the two builds are identical instruction for instruction.

## Tokenisation kernels: what each per-cell step costs (phase 3)

The W6 kernel set (`pyscx.tokenize`) is the per-cell numerics every
transformer-class model re-implements in Python — rank, bin, top-K crop,
expression-weighted sampling. These are their first captured numbers.

Two-build A/B, SLURM `2954036`, host `GPU70DC`, `main` `e8d1f9a9` against
`phase3-tokenize-kernels` `8cab0e3c`, 12 interleaved rounds per dataset,
`RAYON_NUM_THREADS=16`. Raw rows under
`results/raw/phase3_tokenize/tokenize_ab.json`.

### The existing collate arm, which had to stay flat

The top-K crop moved out of `collate_cell` into `tokenize::crop::top_k`, the
kernel now takes reusable scratch instead of allocating three `Vec`s per cell,
and a per-row CSR invariant check was later folded into the same loop. The crop
is byte-identical — the cross-repo `encoder_crop_golden.json` is unchanged and
green — so only the wall could move, in either direction. Flat was the expected
result; a move in either direction was to be reported, not budgeted for.

| metric | pbmc3k | tabula_sapiens_100k |
|---|---|---|
| `us_per_cell__collate` | **1.09× (11/12, p 0.006)** | 1.01× (9/12, p 0.146) |
| `cellsets_per_sec__collate_rust` | **1.09× (11/12, p 0.006)** | 1.01× (9/12, p 0.146) |
| `cellsets_per_sec__gather_random` | 1.03× (11/12, p 0.006) | 0.98× (5/12, p 0.774) |
| `cellsets_per_sec__gather_grouped` | 1.06× (8/12, p 0.388) | 1.03× (9/12, p 0.146) |
| `cellsets_per_sec__gather_random_s512` | 1.06× (9/12, p 0.146) | 0.996× (6/12, p 1.000) |
| `cellsets_per_sec__gather_grouped_s512` | 0.991× (5/12, p 0.774) | 0.988× (5/12, p 0.774) |

Ratios are oriented so > 1 means the change helped. The collate arm came out
**better than flat on pbmc3k** — 13.09 → 12.11 µs/cell, eleven of twelve rounds
— and within noise on tabula (21.33 → 20.94). The three removed allocations per
cell more than pay for both the new call boundary and the per-row validation, and
they are worth proportionally more where cells are shallow, which is what the
pbmc3k/tabula split shows.

`gather_random` on pbmc3k also reads 1.03× at p 0.006. Nothing in this change
touches the gather, so read that as this arm's floor on what twelve paired
rounds can resolve on a shared node, not as a result.

### Per-kernel cost, first capture

Each kernel timed alone over the same gathered batches, at the shapes the models
actually use: `k = 2048` (STATE3 / Geneformer v2), `l_max = 2048`,
`n_bins = 51` (scGPT), `n = 1024` (UCE's `sample_size`).

| kernel | pbmc3k µs/cell | tabula µs/cell |
|---|---|---|
| `crop` (top-K) | 8.15 (7.98–8.41) | 15.80 (15.28–16.18) |
| `rank` (Geneformer-style) | 11.38 (11.24–11.69) | 25.80 (24.93–26.52) |
| `bin` (scGPT-style) | 3.98 (3.84–4.28) | 6.78 (6.33–6.94) |
| `sample` (UCE-style) | 5.12 (4.94–5.36) | 7.23 (6.76–7.69) |
| `collate` (the whole STATE3 chain) | 12.11 | 20.94 |

Median over 12 rounds, min–max in brackets.

⚠️ **An earlier capture published pbmc3k's `sample` cell as "not applicable"**,
on the reasoning that its median cell carries 817 non-zeros against `n = 1024`
so "the draw would cover essentially the whole row". That is an argument about
sampling *without* replacement, and this kernel samples **with** replacement, as
UCE does — repeats are the expected output and a shallow cell is a real
workload. The gate suppressed a real number for a reason that did not apply; it
is deleted, and 5.12 µs/cell is what it was hiding.

**What these numbers are not.** They are a first capture, and **no floor is
proposed from them** — `thresholds.yaml`'s deferred item 22 records what would
have to be true to author one, and the same blocker that has kept the collate
arm's floors deferred since PR-02c applies: no baseline carries `cellset_gather`
rows at all. They are also not a claim about any model's end-to-end tokenise
time, which depends on that model's own sequence assembly, special tokens and
masking — all of which stay with the consumer.

One shape worth reading off the table rather than inferring: the crop's cost
tracks nnz, not `k`. pbmc3k fills 40 % of a 2048-wide crop and tabula 83 %, and
the costs are 8.15 and 15.80 µs — roughly proportional to the 817 and 1699
non-zeros being sorted, not to the constant output width. That is why the crop
arm does not assert that it truncates: truncation makes it cheaper, and at these
shapes it does not truncate at all.

## Neighbourhood plans: what a spatial gather costs (phase 4)

R6 — spatial and graph-context workloads — had no loader surface before this:
coordinates and graphs were stored and nothing turned either into a plan. These
are the first captured numbers for the two builders and for gathering what they
produce.

One-armed capture, SLURM `2961099`, host `GPU0F98`, partition
`cpu_batch_high_mem`, `phase4-neighborhood-plans` `5a69d7ad`, 8 rounds x 3
timed runs on `visium_lymph_node` (4,035 spots x 36,601 genes, 8 shards),
`RAYON_NUM_THREADS=16`, page cache dropped before every run. Raw rows under
`results/raw/phase4_neighborhood/neighborhood.json`.

⚠️ **This capture replaces two earlier ones, and what it does *not* reproduce
is worth stating.** The first (job `2959318`, 2 rounds) and second (`2959338`,
8 rounds) were taken at `ab156ab9`, before two review fix commits changed the
timed path — the graph open now compares every shard's schema and the COO
decode validates every row against its shard's stamped span. Publishing pre-fix
figures as the current builders' numbers is a provenance slip a reader cannot
detect, so the capture was re-run at the reviewed head rather than annotated,
and the superseded artifacts are not shipped.

Those earlier captures showed a first-touch effect — round 1 running 1.4x
slower than rounds 2–8 even though every run drops the page cache — and **this
one does not**: its eight rounds span 5.5 % and 1.6 %. So that effect was a
property of those runs, not of the code, and no claim is made about it here.
It is the reason the capture takes eight rounds rather than the two the phase
gate asks for.

### Steady state, at k = 6 and 146 sets per batch

| metric | graph-driven | coordinate-driven |
|---|---|---|
| sets/s | **15,474** (15,239–16,083) | **14,971** (14,845–15,084) |
| µs/cell | 9.96 (9.58–10.11) | 9.54 (9.47–9.62) |
| plan build, whole file | **2.9 ms** | **13 ms** |
| peak RSS | 1,813 MB (1,599–2,006) | 1,839 MB (1,596–2,005) |

Median over 8 rounds, min–max in brackets. Spread is 5.5 % and 1.6 % on the
rates — unlike the superseded capture, this one has no first-touch outlier, so
every round is reported rather than round 1 being split out. The two paths
agree to within 3.4 % on the gather, which they should: at k = 6 on a lattice
they select nearly the same cells, and the gather does not know which builder
produced the plan.

**Building the plans is not the cost.** 4,035 neighbourhoods come out of the
stored graph in 2.9 ms and out of the coordinates in 13 ms — against roughly
260 ms to gather them. The coordinate builder is ~4.5x the graph one and still
under 1 % of the round, which answers "should the grid search be parallel" with
a measurement rather than a guess. The per-row span validation added between
the two captures is invisible at this scale: 3.2 ms before, 2.9 ms after.

At Xenium scale (10⁵ cells) the build term is the one that grows, and this
capture does not reach it.

### What these numbers are and are not

**They are the pessimal scattered read.** The fixture's obs order is barcode
order, which has nothing to do with position: a 7-cell neighbourhood spans a
median of **3,024 row indices of 4,035** and every batch touches **all 8
shards**. `scx sort` on a key that tracks position is the lever that turns
these into contiguous reads, and this capture does not pull it — so read these
as the floor a spatial file gets for free, not as what the regime can do.

⚠️ **This capture cannot show that nothing else regressed.** Every run is a
*head* build — the driver never builds `main` — so the four pre-existing
`gather_*` metrics recorded on the same runs are within-build variance, not a
comparison. Three of the four are steady across all eight rounds (782.9 sets/s
`gather_random`, spread 5.4 %; 897.2 `gather_grouped`, 7.0 %; 85.6
`gather_random_s512`, 9.8 %), which is worth knowing and is not the same claim.
The fourth, `gather_grouped_s512`, is not steady and resolves nothing: its
median is 95.3 but one round reads 233.2. At 4,035 spots an S=512 grouped batch
is most of the file, so that arm runs a handful of very large plans and one
scheduling hiccup moves it 2.4x. Reported rather than trimmed.

**No floor is proposed**, and `thresholds.yaml` item 23 records three separate
blockers, one specific to this phase: these arms run on a single dataset that is
deliberately outside every `capture_baseline.TIERS` list, and a floor on a
triple the default gate never schedules reads as coverage while providing none.

**The reuse signal this regime was supposed to provide is not set overlap.** The
design called overlapping neighbourhoods "the reuse signal". Measured on the
graph arm at its own parameters — k = 6, 146 sets/batch, `shuffle_seed`
20260915 — the batch duplicate factor is 1.1023 in centre order, **1.1084**
shuffled (the value the committed artifact carries, under
`neighborhood.arms.graph.locality.batch_duplicate_factor`), and **1.1286 for a
random-plan control of the same set size and batch width**. Random sets
duplicate rows *more* than neighbourhoods do, because neighbourhoods partition
the tissue while random draws collide freely. The coordinate arm is the same
story: 1.113 / 1.1185 / 1.1286. The reuse that is real is shard locality, which
is a property of layout, not of the plan.

The ordered and random-control figures are reproducible from the fixture in
seconds with `pyscx.batch_plans` and carry no timing, so they need no capture;
the shuffled one is read off the committed artifact.

## The multi-set batch executor: what a cell-set gather costs (phase 6)

The cell-set gather walked a plan **one set at a time** — a per-set
`Vec<Option<(Vec<i32>, Vec<f32>)>>`, its own read call, two owned `Vec`s per row
inside the row transform, and a copy of every row into a batch pre-sized from a
catalog estimate. It is now one executor over the whole batch, in one of two
shapes: a raw-local single-file plan is read in plan order and **moved out as
the batch**, and a plan spanning files or carrying a length-changing transform
(a remap drops and coalesces, a downsample truncates) is **assembled** from a
deduplicated read.

Two-arm same-build A/B, one build with the executor selected by
`SCX_CELLSET_EXECUTOR` (`set` = the pre-change per-set walk, `plan` = shipped),
12 interleaved rounds with the arm order alternated by round, one timed run per
round, page cache dropped before every run. Ratios are the median of per-round
ratios, oriented so >1 means the shipped build helped; `p` is an exact two-sided
sign test with ties dropped.

SLURM **`2969830`**, host `GPU104C`, partition `cpu_batch_high_mem`,
`phase6-cellset-executor` `c7ca7a26`, `cellset_gather` on
tabula_sapiens_100k / `scx_auto`, `RAYON_NUM_THREADS=16`. Raw rows under
`results/raw/phase6_executor/executor_ab_tabula.json`.

| metric | `set` (med) | `plan` (med) | ratio | wins | p |
|---|---|---|---|---|---|
| `cellsets_per_sec__gather_random` | 830.2 | 765.8 | 0.973x | 3/12 | 0.146 |
| `cellsets_per_sec__gather_grouped` | 811 | 812 | 1.01x | 7/12 | 0.774 |
| `cellsets_per_sec__gather_random_s512` | 91.55 | 90.6 | 1.03x | 7/12 | 0.774 |
| `cellsets_per_sec__gather_grouped_s512` | 85.35 | 98.65 | **1.15x** | 12/12 | 0.000 |
| `cellsets_per_sec__gather_shared_control_per_set` | 343.3 | 373.8 | 1.1x | 8/12 | 0.388 |
| `cellsets_per_sec__gather_hot_control_cold_tail` | 33.8 | 262.9 | **7.76x** | 12/12 | 0.000 |
| `cellsets_per_sec__downsample_rust` | 107.8 | 423.1 | **3.94x** | 12/12 | 0.000 |
| `cellsets_per_sec__collate_rust` | 780.4 | 773 | 0.992x | 3/12 | 0.146 |
| `peak_rss_mb__gather_random` | 2310 | 2320 | 0.995x | 1/12 | 0.006 |
| `peak_rss_mb__gather_random_s512` | 2340 | 2357 | 0.994x | 2/12 | 0.039 |
| `peak_rss_mb__gather_shared_control_per_set` | 2570 | 2498 | 1.03x | 12/12 | 0.000 |
| `peak_rss_mb__gather_hot_control_cold_tail` | 1191 | 1109 | **1.07x** | 12/12 | 0.000 |
| `peak_rss_mb__tokenize_crop` | 1016 | 915.1 | **1.11x** | 12/12 | 0.000 |

**The two big wins are not the same win.** `gather_hot_control_cold_tail` is
7.76x because it is the one arm that asks for the block-index route
(`scatter_block_index=True`) at a budget its plans exceed: one read per plan
instead of one per set gives the chunked parallel group decode the whole plan's
run list to overlap, where before it saw one set's worth at a time. Its
time-to-first-set falls 5.77x with it. `downsample_rust` is 3.94x for an
unrelated reason: the row transform used to run **serially**, inside the read's
scatter callback, and now runs on the loader's rayon pool.

**`gather_grouped_s512` is the arm where the executor's own shape shows.** At
S=512 a covariate group is usually smaller than the set, so
`rng.choice(..., replace=True)` pads it and the plan repeats rows heavily; one
wide read plus an exact allocation is 1.15x at 12/12 there against 1.01x on the
S=64 grouped arm.

**Four of the twelve live `cellsets_per_sec` floors are on this dataset and
none moves down.** The one negative row that clears the sign test,
`cellsets_per_sec__gather_random` at 0.973x, does not (3/12, p = 0.146).

**Peak RSS falls almost everywhere**, most on the arms that gather and then run
a kernel over the batch (`tokenize_*` and `collate_rust`, 1,016 -> 915 MB,
12/12) — those hold the batch while they work, and the batch is now exactly
sized rather than an estimate biased up an eighth. ⚠️ Two rows go the other
way and clear the sign test: `peak_rss_mb__gather_random` 0.995x (1/12,
p = 0.006) and `__gather_random_s512` 0.994x (2/12, p = 0.039). That is
**10-17 MB on a 2.3 GB peak** — reported because it is reliable, not because it
is large, and not suppressed.

### The allocations, which the A/B cannot see

Counting-allocator test (`scx-loader/tests/gather_allocation.rs`), a 1,024-row
plan of 32,768 non-zeros, same fixture and same plan on both arms, measured
after an unmeasured warm-up so what it sees is the assembly rather than the
decode:

| arm | allocations | per row | peak live bytes | × the result |
|---|---|---|---|---|
| `set` | 2,698 | 2.63 | 421,192 | 1.56 |
| `plan` | **81** | **0.08** | **371,304** | **1.37** |

Forcing the assembled path on that plan takes the peak to 2.35× the result and
fails the test's budget, which is how the direct path is pinned.

### ⚠️ census_500k was attempted and abandoned, with its own counters as the reason

The phase's gate names census_500k as the second fixture. The cell was
submitted (`2969831`) and **cancelled after one arm of one round**, which is
shipped as `census_500k_single_run_probe.json` — not as a measurement of the
executor, but as the evidence for dropping it.

That one run took **1,267 s** across its 15 timed sub-runs, so the 12-round
two-arm design needed about **8.5 hours**. More to the point, it would not have
been measuring the gather. The four floored scenarios read a
`shard_cache_hit_rate` of **0.296** (`gather_random`), 0.343 (`gather_grouped`),
0.300 (`gather_random_s512`) and 0.730 (`gather_grouped_s512`), and the loader's
own sizing warning asks for `max_memory_mb >= 5704` to hold 31 shards of
~188 MB against the 21 the default budget affords. At a 0.30 hit rate the cell
prices the shard cache re-decoding shards it just evicted:
`cellsets_per_sec__gather_random` reads **2.7** there against ~766 on tabula.

Raising the budget would make it measurable and also make it a different
scenario from the one the floors are authored against, so it is recorded as not
measurable at this shape rather than measured at another. Phase 1's driver had
already excluded census from `cellset_gather` after a census_500k cell was
killed at 205 minutes; this is the same finding with the cell's own counters
attached.

### ⚠️ Three captures, and two of them are superseded

Both superseded artifacts ship, because each is the evidence for the change that
superseded it.

`2969631` put the executor at **0.811×** on `cellsets_per_sec__gather_random`
(0/12, p = 0.000) and 0.820× on `__gather_grouped`, while the duplicate- and
transform-heavy arms won (`gather_hot_control_cold_tail` 7.92×,
`downsample_rust` 3.71×, both 12/12). The cause was not the executor: the
`read_row_indices` prescan re-decoded a **resident** shard's indptr on every
call — deliberate, so that a prescan between planning and warming cannot change
what the block-index route admits, and pure waste once the shard is decoded. A
local probe on a 100k-row synthetic with every shard warm read 0.47 ms/batch on
the per-set walk against 2.73 ms on the executor; after the fix, 0.47 against
0.29.

`2969753`, with that fixed, left a residual **0.926× / 0.910×** — the dedup
itself. Deduplicating forces the batch to be assembled from the read, which
costs a second batch-sized allocation and a second full copy, to save one memcpy
per repeated row from an already-resident shard. That is all it ever saved on a
raw-local plan, whose whole row transform is an elementwise clip. So the shipped
rule deduplicates only where a repeat costs real work, and the phase-4
measurement above says how often that is: a neighbourhood batch's duplicate
factor is **1.108** and a random-plan control's is **1.129** — about a tenth of
the rows, nowhere near enough to pay for a second buffer.
