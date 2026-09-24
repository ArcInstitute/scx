# Training Loader

> Part of [SCX performance](README.md). More loader results:
[IndexPlanDataset](loader-index-plan.md),
[cell-set gathers](loader-cell-sets.md), and
[data-load phase 1](loader-data-load.md). Usage is in
[docs/training.md](../training.md).

Batches/sec, batch_size=1024, HVG=2000, normalize+log1p:

| Dataset | SCX | AnnData | TileDB-SOMA-ML | scDataLoader | SLAF | SCX/SOMA |
|---------|-----|---------|----------------|--------------|------|----------|
| Census 1M | **1,405** | 16.3 | 17.1 | 4.4 | 4.1 | **82x** |
| Tabula Sapiens 100K | **1,060** | 14.5 | 16.1 | 4.0 | — | **66x** |
| PBMC 3K | **168** | 14.5 | 5.0 | 6.3 | — | **34x** |

Triple-buffered pipeline (tokio I/O -> rayon decode -> Python) with native HVG projection and fused normalize+log1p delivers **34-82x higher throughput** than TileDB-SOMA-ML at scale (and **~340x higher throughput** than SLAF on Census 1M). TTFB (time to first batch): 16 ms on PBMC 3K, 603 ms on Census 1M.

SLAF numbers come from `SLAFDataLoader` with the Geneformer tokenizer
(max_genes=2,048). On Census 10M the default Mixture-of-Scanners prefetcher
returns 0 batches per scenario (TTFB 90.2 s then timeout) — flagged as a
SLAF-upstream tuning issue, not a harness defect. Source JSONs:
`benchmarks/comprehensive/results/raw/ml_loader__slaf__census_{1m,10m}.json`.

## Training loader on-disk size — `auto` vs `fast` codec (2026-07-10)

**On-disk fixture size (MB, lower is better):**

| Dataset | scx fast (Scx1) | scx auto (ShufDeltaZstd) |
|---|---|---|
| pbmc10k | 39.8 | 30.1 |
| smartseq2 | 552.7 | 262.7 |
| tabula_sapiens_100k | 448.9 | 233.5 |
| census_500k | 1841.8 | 1182.9 |
| census_1m | 3988.8 | 2800.1 |

**Storage: the default optimizes for size.** As of the 2026-07-12 codec-intent
flip, `codec="auto"` is **cost-aware adaptive** (predominantly ShufDeltaZstd) — the
column labeled `scx auto (ShufDeltaZstd)` above is what the new default `auto`
produces, and the `scx fast (Scx1)` column is `codec="fast"`. Adaptive
`auto` recovers ~30% of the disk vs `fast` (census_1m 3989→2800 MB) while staying
**within ≤~3% of `fast` on the realistic `hvg_norm` training scenario at every scale**.
Latency-critical CPU training that wants the old decode-max behavior pins
`codec="fast"`.

> **Naming (2026-07-12 flip).** These tables use the current codec-intent names;
> the numbers are the original pre-flip measurements. For reference, the old `auto`
> (Scx1, decode-max) is now **`fast`**, and the old opt-in `auto_v2` (ShufDeltaZstd,
> adaptive) is now the default **`auto`**. An exact recapture under the new names
> (and the tightened adaptive
> floors, incl. a **looser raw-path (G1b) throughput floor** — the `raw`
> full-width-scatter path carries the full ~9.6–20% ShufDeltaZstd CPU-decode tax, vs
> ~6% on `hvg_norm` and ~4% on `gpu_train`) is deferred to the comprehensive recapture.

*Methodology: all rows from the `benchmarks/comprehensive` ml_loader capture
(median-of-N, batch_size=1024, HVG=2000), 2026-07-10/11, consistent with the promoted
`v0.11.0-recapture` baseline. Emitting census scx rows required two harness
fixes: scaling the loader `memory_budget` to the SLURM allocation (`_scx_memory_budget_mb`,
0.6×`--mem`) so `batch_size` stays at 1024 — the default 4096 MB is too small for 1M×61,497
full width and the auto-tune otherwise collapses `batch_size` to 64 (a budget/estimator
interaction, not a fixture bug: standard 62×16k-shard geometry) — and skipping the
`pyscx_training_dataset_workers2` scenarios above 250k cells, which time out at census scale
and would otherwise discard the whole (dataset, format) result.*

## ShufDeltaZstd loader-decode cost (Phase-D D0 profiling)

The training loader decodes CSR shards on **CPU** (stage-1 `io_stage`
`read_shard_from_entry`, inside `spawn_blocking`; stage-2 `decode_stage` scatters
to dense) — the merged GPU ShufDeltaZstd decoder serves the `to_gpu_anndata`
*analysis* path, not the loader. D0 profiled whether moving loader decode onto the
GPU (Phase D) would lift training throughput. H100, pbmc10k (single 16,384-cap
shard = 1 row-group), `SCX_LOADER_PROFILE=1`, batch_size=1024:

| Regime | Codec | Stage-1 decode/epoch | `io_stage` send-wait | GPU util |
|--------|-------|----------------------|----------------------|----------|
| raw (null consumer) | `fast` (Scx1) | 325.8 ms | ~0 (4 µs) | — |
| raw (null consumer) | `auto` (ShufDeltaZstd) | 357.1 ms (**+9.6%**) | ~0 (4 µs) | — |
| gpu_train (scVI VAE) | `fast` (Scx1) | 262.6 ms | ~0 (5 µs) | 0% |
| gpu_train (scVI VAE) | `auto` (ShufDeltaZstd) | 286.3 ms (**+9.0%**) | ~0 (4 µs) | 0% |

The ShufDeltaZstd CPU-decode tax over Scx1 is only **~9%** (consistent with the
Phase-B whole-shard ~1.09×; SSE2 SIMD already closed the gap that once was
1.3–1.8×). Stage-1 send-wait is ≈0 (the loader never blocks handing batches to the
consumer) and GPU util is ≈0% (the model is too light to expose decode) — so the
"smaller PCIe payload → throughput" lever Phase D would exploit is not on the
critical path in any measured regime. **GPU-loader decode (Phase D D2–D5) is
therefore deferred**; a `gpu`-gated `scx-loader → scx-gpu` feature edge (D1) exists
as scaffolding. Storage is unaffected: `auto` is 24% smaller on this dataset
(30.1 vs 39.8 MB) — the egress win is real and orthogonal to loader throughput.

## Out-of-core loader — cold-cache measurements and the P-1 premise gate

Every loader number above this subsection is **page-cache-warm**. That matters: annbatch's
paper records the same methodological trap inflating BioNeMo-SCDL from 2.5k to 110k
samples/s purely through cache residency. The benchmarks here drop the page cache before
**every** timed epoch — per-file `posix_fadvise(POSIX_FADV_DONTNEED)`
(`benchmarks/comprehensive/cache_control.py`, unprivileged so it works on shared SLURM
nodes) — and every row records a `cache_policy` tag so a silently-warm run is visible
rather than assumed away. Census runs additionally execute under an
`SCX_BENCH_OOC_MEM_CAP_GB=96` allocation that under-sizes the node below the resident
footprint, forcing genuine misses.

Runners: `benchmarks/comprehensive/benchmarks/{ooc_loader,cellset_gather,obs_open}.py`.
Baseline `results/baselines/v0.11.5-dataload-phase0` (captured 2026-07-23, `cold_fadvise`,
96 GB cap), with **33** cold-cache floors under `absolute_floors` in `thresholds.yaml`.
BioNeMo-SCDL is **not** included — its NeMo/CUDA stack needs an isolated env, tracked as a
deferred floor rather than quietly dropped.

**Sequential-epoch throughput, cold, samples/s** (`raw` = no HVG/normalize; `hvg_norm` =
2k-HVG projection + `normalize_total` + `log1p`). Blank cells were not run in this wave,
not zero.

| dataset | scenario | SCX (auto) | h5ad full-RAM | annbatch | scDataset | AnnLoader |
|---|---|---|---|---|---|---|
| pbmc10k | raw | **26,088** | 13,957 | 11,501 | 7,510 | 5,784 |
| smartseq2 | raw | **20,338** | 9,050 | 8,485 | 4,333 | 4,315 |
| tabula_sapiens_100k | raw | **28,825** | 15,681 | 11,728 | 6,748 | 6,141 |
| census_500k | raw | **48,786** | 17,159 | 14,686 | 9,587 | 6,492 |
| census_1m | raw | **57,795** | — | — | 10,153 | 6,433 |
| census_5m | raw | **62,351** | — | — | — | — |
| census_500k | hvg_norm | **58,391** | 15,949 | 14,052 | 9,285 | 6,386 |
| census_1m | hvg_norm | **63,110** | — | — | 9,970 | 6,729 |

At census_500k cold, SCX is **2.8× h5ad-full-RAM, 3.3× annbatch, 5.1× scDataset, 7.5×
AnnLoader**; at census_1m, **5.7× scDataset and 9.0× AnnLoader**. Two honesty notes: SCX's
lead *grows* with size (26k → 62k samples/s from pbmc10k to census_5m) because the
competitors are I/O-bound where SCX's decode pipeline still has headroom; and annbatch's
paper figure (~35k on Tahoe, EBS-bound) is **not** comparable to its 14.7k here — different
hardware, dataset and filesystem (wekafs). The cross-format comparison in one column of one
table is the comparable quantity.

### ⚑ The P-1 premise gate

The plan of record makes every optimization past Phase 1 conditional on one question: **is
there a regime where SCX's loader is the bottleneck?** It exists because STATE3 measured its
own loader at **0.8 ms of a 130 ms compiled step — 0.6%, hidden ~160×** — and closed loader
parallelism as unwarranted. Four candidate regimes that measurement did not cover:

**(a) 26,453-file manifest startup — NO.** Measured directly on a 256-file manifest fixture,
cold: SCX `read_obs([col])` costs **6.13 ms/file** against **115.5 ms/file** for
`anndata.read_h5ad(backed='r')` + `.obs[col]` — **18.8×**. Extrapolated to STATE3's
`basecount_homo_sapiens_train_int.csv` (26,453 files) that is **~162 s vs ~51 min** per
process. 162 s of one-time catalog build is not a bottleneck worth Rust work, and it is
already the cheap side of a 19× gap. (Per-file cost here is *lower* than the single-file
numbers below because the fixture shards one 100k-cell file into 256 small ones — the
small-file regime, which is what a 26k-file manifest actually is.)

The sub-component breakdown settles which term is worth attacking. Per file, cold, on
dedicated SLURM allocations at three scales:

| dataset | open + mmap + header + catalog parse | BLAKE3 catalog verify | **obs read** | obs read share |
|---|---|---|---|---|
| tabula_sapiens_100k | 0.09 ms | 0.01 ms | **53.8 ms** | 99.8% |
| census_500k | 18.5 ms | 1.2 ms | **498.8 ms** | 96.2% |
| census_1m | 20.0 ms | *below noise* | **1251.2 ms** | 98.7% |

Measured as `open_only_unverified` / (`open_only − open_only_unverified`) /
(`open_1file − open_only`). The census_1m verify term came out **negative** (−3.8 ms), i.e.
BLAKE3 catalog verification is cheaper than the run-to-run spread — reported as "below noise"
rather than as a negative cost.

**The open is cheap and obs reading is essentially the whole cost — 96–99.8% at every
scale.** That settles P0.1b's internal argument in favour of its own proposal: its stated
non-goal was a separate `ScxObsReader` that "opens without mmap-ing X", on the grounds that
per-file cost is `File::open` + full-catalog parse + BLAKE3 verify. Measured, those three
total 0.1–21 ms against a 54–1251 ms obs read. So neither `open_unchecked`, nor catalog-size
reduction, nor a process-level catalog cache is worth building; the numpy
`(codes, categories)` accessor — skipping the pandas/pyarrow round-trip — is aimed at the
right term.

Neither measurement needed a new pyscx API: `pyscx.open(path, verify=False)` already exists
(`pyscx/src/lib.rs:144` → `ScxReader::open_unchecked`).

> **A correction worth recording, because it inverted the conclusion.** An interactive
> `pbmc10k` smoke of the same three scenarios read 13.8 ms open / 0.32 ms verify / 1.73 ms
> obs — "87% open", the exact opposite split — and was briefly published here. It was
> instrumentation: in a warm interactive shell the first `pyscx.open` absorbs interpreter and
> pyo3 init plus page first-touch that `posix_fadvise` does not evict, and 13.8 ms of "open"
> on a 30 MB file was never plausible beside 0.09 ms on a 233 MB one. **A small-fixture
> interactive smoke is not a measurement** — a cost *ratio* between a first-touch-contaminated
> term and a steady-state term can invert outright, and this one did.

For the h5ad baseline the split is **not separable**: `open_only` and `open_1file` agree to
within noise at every scale (the subtraction even goes slightly negative), because anndata
builds the obs index during `read_h5ad(backed='r')`. Materializing one obs column on top of
that costs nothing measurable — all of anndata's cost is the open.

**(b) Observational / pretraining at S=128–512 — NO.** STATE3's 0.6% figure is S=64, B=4,
GPU-memory-ceilinged, so the larger-set regime was genuinely untested. Measured cold, holding
cells/batch ≈ 1024 so only set granularity changes:

| dataset | plan | cells/s S=64 | cells/s S=512 | change | sets/s S=64 | sets/s S=512 |
|---|---|---|---|---|---|---|
| tabula_sapiens_100k | random | 329 | 426 | 1.3× | 5.1 | 0.8 |
| census_500k | random | 423 | 443 | 1.0× | 6.6 | 0.9 |
| census_1m | random | 446 | 467 | 1.0× | 7.0 | 0.9 |
| tabula_sapiens_100k | **grouped** | 435 | 733 | **1.7×** | 6.8 | 1.4 |
| census_500k | **grouped** | 1,028 | **4,446** | **4.3×** | 16.1 | 8.7 |
| census_1m | **grouped** | 1,012 | **3,139** | **3.1×** | 15.8 | 6.1 |

The two plan types separate cleanly, and the mechanism is legible. **Random scatter** scales
~linearly with cells: each of the S rows is an independent random row, so 8× the set size
means ~8× the work — cells/s is flat and sets/s falls ~7.8×. **Covariate-grouped** gather
amortizes shard decode across the set, because a group's cells cluster into few shards: 8×
the set size costs only ~2.6× the time, so cells/s *improves* 3–4×.

Real models issue grouped sets, not uniform-random ones. In that regime **S=512 is 3–4×
cheaper per cell than S=64**, so moving to the pretraining regime pushes the loader further
from the critical path, not closer.

> **Do not read these against STATE3's 4.55 steps/s.** These are cold-cache numbers with the
> page cache dropped before every run — the epoch-start worst case. STATE3's steady state had
> `scx_cache_shards=48` and a populated ~8 GB shard cache, which is precisely what amortizes
> the per-shard decode measured here. Taken naively, 1,012 cells/s would put S=64×B=4 = 256
> cells at 253 ms against a 130 ms step; that comparison is invalid in both directions. The
> cold table answers "how does gather cost scale with set size", not "what does a training
> step cost".

**(c) DDP multi-rank — NO.** Measured by running the same per-rank workload in N spawned
processes against the same file(s) and reporting `rank_scaling_efficiency` = median(per-rank
rate at N) ÷ (rate at 1 rank); 1.0 means a rank is as fast with siblings as alone, ~1/N means
the ranks serialise.

| arm | dataset | N | efficiency (2 runs) | aggregate sets/s | total peak RSS |
|---|---|---|---|---|---|
| `gather_random_r4` | census_1m | 4 | **1.013, 1.010** | 7.0 → **28.2** | 6.98 GB |
| `gather_random_r4` | census_500k | 4 | **1.001, 1.004** | 6.6 → **26.2** | 4.44 GB |
| `gather_random_r4` | tabula_sapiens_100k | 4 | 0.989, 0.925 | 4.9 → **19.2** | 2.43 GB |
| `open_manifest_r4` (SCX) | tabula, 256 files | 4 | 1.70, 1.87 | — | — |
| `open_manifest_r4` (h5ad) | tabula, 256 files | 4 | 1.03 | — | — |

**The gather arm is the contention-honest one** — each rank gets a distinct seed, so ranks
touch different shards and the page cache cannot flatter the result. At census scale
efficiency is **1.00–1.01**: four concurrent ranks are each exactly as fast as one alone, and
aggregate throughput scales ~4×. Peak RSS scales linearly (1.75 GB/rank at census_1m), which
was worth noting when `SparseCellSetDataset` had no byte cap on its shard cache — since the
Phase-1 1A work below it resolves `max_memory_mb=None` to a bounded adaptive budget, so
per-rank RSS is now capped rather than open-ended.

**The manifest arm's 1.70–1.87 is cache sharing, not an absence of contention**, and the
distinction matters. There every rank reads the *same* 256 files by design (each model process
builds a global obs vocabulary from every file, so per-rank cost does not shrink with world
size), so whichever rank touches a file first warms it for the other three. The h5ad arm's
1.03 corroborates the mechanism from the other side: anndata's manifest cost is CPU-bound
index building rather than I/O, so there is nothing for a sibling rank to inherit and it scales
exactly linearly.

Two further details make these numbers mean what they say: `spawn` rather than `fork` (a
parent-constructed dataset used post-fork trips pyscx's PID guard), and a barrier before the
timed region (otherwise interpreter-startup skew makes the "concurrent" window partly serial).

**The limit, stated rather than buried:** N processes on one node measures shared page-cache,
shared-filesystem and memory-bandwidth contention. It does **not** measure inter-node NCCL
interaction or per-rank sampler cost under a real `DistributedSampler`. This rules out the
loader as a *shared-resource* bottleneck on one node; it does not certify multi-node DDP.

**(d) STACK — NO, and for a more basic reason.** `arc-stack` v0.1.3 (`~/dev/python/stack`)
contains **zero `pyscx`/`scx` references**: it is not an SCX consumer. Its loader is pure
h5py with a hand-rolled block-coalescing CSR gather. STACK cannot be a regime where *SCX's*
loader is the bottleneck, and since it was the main driver of the global-pre-shuffle item,
that item's priority drops accordingly.

**Verdict: no on all four.** The plan's own gate condition — "a 'no' on all four ends the
plan at Phase 1" — is met. The measured picture: SCX's sequential loader is 3–9× faster than
every competitor cold and its lead grows with size; grouped cell-set gather gets *cheaper per
cell* as sets grow, so the untested pretraining regime is further from the critical path than
the one already measured at 0.6% of step time; four concurrent ranks each run at full speed;
and per-file obs cost is 96–99.8% obs read, so the one Phase-1 obs item is aimed correctly
while the open-side ideas are not worth building.

Optimization effort past the small Phase-1 residuals is therefore **not funded by
measurement**. The binding constraint is adoption — STATE3's integration is unmerged and
STACK has none — not loader capability.

**What this does not say.** These are cold-cache, single-node, CPU-side measurements. They do
not certify multi-node DDP (no NCCL in the picture), they do not measure a real
`DistributedSampler`'s per-rank sampler cost, and they say nothing about steady-state
warm-cache training throughput, which is what STATE3's own 4.55 steps/s figure covers. A
regime that shows up in any of those is not excluded by the table above — it is simply not
evidenced today.

## Data-wait fraction per regime (`p`) — the loader-work gate

Every throughput item in the ML-loader plan is conditional on one number per
access regime: `p`, the fraction of a training step spent waiting on data. The
arithmetic is unforgiving — accelerating an exposed fraction `p` by `s` is
worth `1 / [(1 - p) + p/s]` and no more, so at `p = 0.006` (STATE3's measured
figure) even `s = infinity` buys 1.006x. Tier-3 loader work is funded per
regime by what `p` actually is, not by how fast a decoder could be.

**Quote `data_wait_fraction_steady`, not `data_wait_fraction`.** The first
`next()` of an epoch pays tokio spin-up, the first shard decode and (on the
`DataLoader` path) worker spawn. Folding that into the fraction measures how
long the benchmark's epoch happens to be: a 98-step `gpu_train` epoch over
tabula reports **0.85** all-steps against a **0.009** steady figure, because
2.04 s of its 2.4 s "wait" is step 1. `ttfb_s` carries that startup separately.

| regime | consumer | dataset | cells | `p` (steady) | wait p50 | wait p95 | ttfb | steps |
|---|---|---|---|---|---|---|---|---|
| **R1** i.i.d. minibatches | scVI-equivalent VAE (2L/128h/128z), batch 1024, HVG+normalise+log1p | tabula_sapiens_100k | 100 k | **0.0086** / 0.0035 | 13 us | 16 us | 2.04 / 1.88 s | 98 |
| | | census_1m | 1 M | **0.759** / 0.749 | 13 us | 799 / 256 us | 2.02 / 1.93 s | 977 |
| **R2** grouped sets | STATE3 — consumer-side, not measured here | — | — | *(not measured)* | — | — | — | — |
| **R3** paired batches | `IndexPlanDataset` + a 25 ms fixed step | tabula_sapiens_100k | 100 k | **0.41** (0.401–0.421) | 9.7 ms | 29 ms | 0.76 s | 49 |

Two figures per R1 cell are `scx_auto` / `scx_fast`. **R3 ran `scx_auto`
only** — its figure is the median over two timed runs, with their range in
parentheses. R1 is `ml_loader`'s `gpu_train` scenario, R3 `index_plan`'s
`pyscx_index_plan_dataset_workers2`.

**R1 does not have one answer — it has two, and scale is the variable.** At
100 k cells the loader is nowhere near the critical path (`p` under 1 %,
per-batch wait 13 microseconds, agreeing with the D0 profile's ~4 microsecond
send-wait). At 1 M cells it is roughly three quarters of steady-state step
time. The single-fixture reading that "no model on record is data-starved on
SCX" does not survive the move to census scale with this consumer.

**And at census scale the wait is a tail, not a level — by arithmetic, not yet
by direct measurement.** p50 is 13 microseconds and p95 799 microseconds, yet
the steady fraction is 0.759. Over a ~12.6 s steady region, 95 % of 976 steps
at or below 799 us account for under 0.8 s, so the remaining ~49 steps must
carry ~9 s — of order 200 ms each. Both inputs to that (the steady fraction and
p95) are sound, but it is an inference: the metric that would show the stall
directly, `batch_wait_ms_max`, was **defective in the captures published here**
and is fixed but not yet re-captured (see the correction below). The work this
points at is deeper cross-batch prefetch, not faster decode: the median batch
is already free.

> ### ⚠️ Correction — the published `batch_wait_ms_max` is time-to-first-batch
>
> The captures behind this table computed the wait percentiles over the **full**
> per-step list, including the first `next()`. `batch_wait_ms_max` is therefore
> the startup cost restated in milliseconds — `ttfb_s` 1.933 against `max_ms`
> 1932.967 in `results/raw/phase0_p/ml_loader__scx_fast__census_1m.json` — and
> `p99` collapses onto it on short epochs, where nearest rank puts 0.99 at the
> last index. **The steady fraction and p50/p95 are unaffected** (p95 sits at
> index 927 of 977, far below the startup entry), so the `p` values and the tail
> arithmetic above stand.
>
> **Fixed** by moving the percentiles inside `steady_state_wait`, which computes
> them on the post-startup slice; `wait_percentiles` remains for the all-steps
> view and nothing publishes it. Guarded by
> `test_steady_percentiles_exclude_the_startup_batch` and an AST check that
> neither emitter can call it over the full list again. The `max`/`p99` columns
> are omitted from the table above rather than printed from the defective
> captures. Found by **codex - gpt-5.6-terra** and **Antigravity - Gemini 3.8
> Flash**.

> [!IMPORTANT]
> **`p` is a ratio, and the denominator here is a benchmark's model, not
> yours.** R1's consumer is a deliberately small VAE, so its step is cheap and
> `p` is correspondingly high; a heavier model lowers `p` without the loader
> changing at all. R3's step is a literal 25 ms `sleep`. Read the **absolute**
> wait, which is a property of the loader alone, and divide by your own step:
> R3's ~18 ms of wait per batch is `p = 0.40` against a 25 ms step but would be
> `p ≈ 0.12` against the 130 ms step STATE3 measured. The fractions above are
> reported because the gate's threshold is stated as a fraction, not because a
> fraction transfers between consumers.

**R3's `p` exists only because the scenario was given a step to have a
fraction of.** `pyscx_index_plan_dataset_workers2` counts batches and does no
model work, so its unmodified data-wait fraction is ~1.0 by construction and
says nothing. `SCX_BENCH_R3_NULL_MODEL_MS` buys a fixed-cost stand-in step
(default `0` = off, so registered captures are unchanged); the row above is a
separate, deliberately unpromoted capture at 25 ms/batch. Its two runs
disagree sharply on p50 (0.196 ms vs 19.29 ms) at the same p95 — DataLoader
worker scheduling, and a reason to read R3's median rather than either run.

**R2 is consumer-side.** STATE3 is the reference consumer and the only model
with a previously published `p` (0.8 ms of a 130 ms step, 0.6 %, at S=64/B=4
warm). Reproducing it on the backed and native paths is a STATE3-repository
measurement, not one this benchmark suite can make, so the cell is **empty
rather than zero** — the distinction the `None`-not-`0.0` rule in
`benchmarks/comprehensive/data_wait.py` exists to preserve.

> [!NOTE]
> Manifest entries, all force-added under
> `results/raw/phase0_p/`: `ml_loader__scx_{auto,fast}__{tabula_sapiens_100k,census_1m}.json`
> (R1) and `index_plan__scx_auto__tabula_sapiens_100k_nullmodel25ms.json` (R3).
> The R3 row carries a **distinct filename** so it can never overwrite the
> tracked `results/raw/index_plan__scx_auto__tabula_sapiens_100k.json`, whose
> 25.14 batches/s backs the OPT-FORMATIO-1 claims — a null-model capture did
> exactly that once. ⚠️ These captures predate the percentile fix above, so
> their `batch_wait_ms_p99` / `_max` keys (where present at all) are not
> trustworthy; the `p`, p50 and p95 values this section quotes are. The metric is **not**
> floored — see `thresholds.yaml`'s deferred item 21: it is `None` on most
> scenarios by construction, and a bound on it would pin the ratio of two
> unrelated things (it falls when the *model* slows down). These are cold-cache,
> single-node, single-rank numbers against the stated consumers; they do not
> certify multi-node DDP, a real `DistributedSampler`, or any other model.
