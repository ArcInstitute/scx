# IndexPlanDataset (plan-driven paired reads)

> Part of [SCX performance](README.md). See also the [training loader overview](loader.md).

`pyscx.IndexPlanDataset` is the sibling type for perturbation training and
other workloads where each batch is a list of `(perturbed_cell, control_cell)`
pairs. It consumes a Python iterator of plans, gathers rows via the cached
`BackedCsrReader`, and yields paired dense `{X, X_paired, pairs, obs, obs_paired}`
batches.

**1M-cell synthetic fixture** (Lambda HPC, `vci-steady-state-node-020`,
`--cpus-per-task=16 --mem=64G`; 1M × 2K × 5%-density × 8 nnz/row × HVG=2000 ×
1024 pairs/batch × 1000 batches.

| Scenario | batches/s | cells/s | Notes |
|---|---|---|---|
| `pyscx_index_plan_random` | 9.93 | 20,342 | uniformly-random pairs (Mode A) |
| `pyscx_index_plan_locality` | 9.81 | 20,089 | shard-locality keyed (Mode B) |
| `pyscx_backed_python_loop` | 0.09 | 189 | current cell-load-scx baseline (50 batches) |
| `pyscx_training_dataset` | 36.62 | 74,996 | sequential ceiling |

Headline: **`IndexPlanDataset` is 106× faster than the current
`ScxBackedSparseDataset` Python-loop path** (20,089 vs 189 cells/s) that
cell-load-scx consumes today. The `TrainingDataset` ceiling is 3.7× higher
because sequential reads can stream shards in catalog order; plan-driven
access is intentionally random and trades that for per-cell pairing
flexibility.

**Locality optimizations (1M cells, same fixture):**

| Lever | Delta | Notes |
|---|---|---|
| Shard sort | 0.96× | break-even (fixture fits in 1.4 TB RAM, page-cache hot) |
| Vectorised pair scatter | 1.07–1.08× | CPU-bound microbench, scale-independent |
| Lookahead 0→4 | **1.04×** | small but consistent at 1M scale |
| Lookahead 4→8 | 0.98× | beyond 4 doesn't help on this fixture |
| Zero-allocation dense gather | **1.36×** | tabula_sapiens_100k A/B, see below |

The shard-sort and lookahead deltas are scale-dependent — they should grow once the
fixture exceeds RAM and shard-cache misses start dominating wall time. The
default `lookahead=4` is justified by the 1M result; `cache_shards=128` is
the more important lever.

**Zero-allocation dense gather** (tabula_sapiens_100k, HVG=2000,
1024 pairs/batch × 300 batches, page-cache warm; A/B on Chimera CPU node).
Replaces the `read_row_indices` → per-row `ScxCsr` → `concatenate_csr`
pipeline with `BackedCsrReader::read_rows_with`, which scatters directly
from the LRU-cached shard into the dense output (no per-row CSR allocation,
no final concatenate). Per-shard request grouping uses `partition_point`
(O(log R) per shard) instead of the old O(R) inner scan.

| Scenario | before (b850e36^) | after (b850e36) | speedup |
|---|---|---|---|
| `pyscx_index_plan_random` | 19.42 batches/s | 26.47 batches/s | **1.36×** |
| `pyscx_index_plan_locality` | 21.38 batches/s | 29.01 batches/s | **1.36×** |

Peak RSS unchanged (~3.8 GB, dominated by HVG-projected output buffers).
The 1M-cell numbers in the table above predate this change and will refresh
on the next baseline capture. `read_row_indices` is unchanged — pyscx
scipy-interop callers in `pyscx/src/backed.rs` and `lazy_transform.rs`
that need a `ScxCsr` return shape continue to use it.

Comprehensive-suite module: `benchmarks/comprehensive/benchmarks/index_plan.py`
(SCX-only, gated on `scx_auto`). Run via
`python benchmarks/comprehensive/scripts/gate_candidate.py --benchmarks index_plan`.
Standalone driver for fast iteration: `benchmarks/index_plan_bench.py`. Cluster
sbatch script: `benchmarks/index_plan_bench_1m.sbatch`. Result JSONs from the
1M run live at
`benchmarks/comprehensive/results/index_plan_phase7/index_plan_1m_{indexed,backed_python}.json`.
Floors in `thresholds.yaml` set at 0.5× the 1M medians — re-pin if/when
fixtures grow beyond memory.

## Scattered pair-gather via the row-group BlockIndex (L1+L2)

For `streaming_mode=indexplan` (STATE_TX), each batch's `(pert, ctrl)` pairs are
scattered randomly across shards. The legacy gather **full-decodes every touched
shard** (`read_shard_cached_arc`) to extract a handful of rows; when the touched
shard set exceeds the byte-budgeted shard cache this thrashes — Census 1M
(62 shards) collapsed to **0.13 batches/s**. The block-index gather routes the
scattered rows through `BackedCsrReader::read_rows_with`, which decodes only the
row groups containing the touched rows **O(rows)** via the codec-agnostic
row-group `BlockIndex` (framing) instead of O(shard) (L1), and a block-index-aware
prefetch (L2) skips warming block-index-eligible cold shards so the O(rows) path
isn't negated by eager full-shard warms.

Measured (Lambda HPC, scx-bench pyscx 0.9.1, regenerated framed fixtures,
`lookahead=4`, `cache_shards=128`, `max_memory_mb=8192`; block-index on = the
`scatter_block_index` default, off = the L2 prefetch-skip disabled). Census 1M,
where shards exceed the cache budget → thrash:

| scenario | pairs/batch | block-index batches/s | legacy batches/s | speedup | p50 latency (block-index / legacy) |
|---|---|---|---|---|---|
| random   | 32  | 59.9 | 0.94 | **64×** | 11 ms / 256 ms |
| random   | 128 | 20.5 | 0.30 | **68×** | 44 ms / 4815 ms |
| locality | 2   | 170  | 17   | 10×   | 2.6 ms / 1.0 ms |
| locality | 128 | 20.1 | 0.58 | 35×   | 44 ms / 917 ms |

**Peak RSS drops** with the block-index path (no full-shard pool): Census 1M
`bs=2` **3.06 GB vs 14.9 GB** (~5× less); tabula_sapiens_100k `bs=2` 1.06 GB vs
3.40 GB (3× less) — the streaming memory ceiling is preserved.

**Regime.** The *throughput* win requires cache-thrash (touched shards > cache
budget). When the shard set fits the cache (e.g. tabula's 7 shards), the
full-shard path warms once and is comparable or marginally faster at tiny batch
sizes — there the win is **memory**, which is universal. The per-dataset
`scatter_block_index=false` kwarg (and the process-wide `SCX_SCATTER_BLOCK_INDEX=0`
kill-switch) restore the legacy path for the fits-cache regime.

**`SparseCellSetDataset` defaults the other way (`scatter_block_index=false`).**
The cell-set loader serves a cache-*friendly* regime (sorted data + a reused
control-pool cache → a small working set that fits the shard cache and is touched
on most batches). There the block-index "skip the warm, decode O(rows) each batch"
strategy *was* a net loss when measured: the hot shard's groups were re-decoded every
batch and nothing retained them (the row-group LRU that now does so landed later, in
OPT-FORMATIO-1 — these numbers predate it). So `SparseCellSetDataset` defaults to the full-shard warm+cache path,
recovering ≈ `.h5ad` parity on a 50-file Tahoe atlas (steps/s 2.80 → 4.55, gather
337 → ~5 ms, cache populated to ~8 GB). ⚠️ **Provenance**: those Tahoe figures were
measured on the *scx1 decode-sidecar* gather this knob replaced, not on the
block-index gather, and were relabelled onto `scatter_block_index` when the
sidecar was removed. The mechanism carries over exactly — both routes skip the
warm and key eligibility on `!cache.contains()` — and the block-index gather has
now been measured directly:

| tabula_sapiens_100k, reframed to v4, S=64 random cell sets, cold | cellsets/s (median of 3) | peak RSS (median) |
|---|---|---|
| `scatter_block_index=false` (the default) | 541.3 | 3233 MB |
| `scatter_block_index=true` | 2.63 | 977 MB |

**206×** on this fixture, with peak RSS running the other way — 977 MB against
3.2 GB, 3.3× less. A 1-shard `pbmc10k` reframed the same way gives 988.1 vs 5.26
cellsets/s (**188×**) at 462 vs 411 MB: the throughput gap holds while the memory
gap nearly vanishes, because the block-index route's bounded peak buys nothing
until a whole shard is large.

> [!WARNING]
> **These two ratios are superseded.** They predate the row-group LRU
> (OPT-FORMATIO-1), which is the whole reason they were re-measured: see
> [Cell-set scatter routes, re-measured after the row-group
> LRU](#cell-set-scatter-routes-re-measured-after-the-row-group-lru-phase-0-gate),
> where the comparison is redone as a 2×2 of corpus size × plan locality. The
> headline changes shape: there is no fixed winner. Full-shard wins 167× at
> 100 k cells, 1.6× at 1 M, and **loses** 1.53× at 1 M once the plan has the
> locality a real model's sets have.

Treat the ratio as an order of magnitude, not a constant — three captures of the
same tabula arm pair landed at 184×, 187× and 206×, since the `on` arm's absolute
rate (2.6–3.2 cellsets/s) is small enough that ordinary node variance moves the
quotient by ~10%. What is stable is the direction and the scale.

Read it as the *extreme* of the cache-friendly regime. tabula's 7 shards sit
inside the 128-shard default cache, so the working set is fully resident after
the first pass (hit rate 0.9987) — exactly where re-decoding row groups per batch
cost everything: at the time of this capture the block-index path retained nothing, so
the LRU never populated on it (OPT-FORMATIO-1 has since made the touched groups
resident under the same budget; this table is the pre-change measurement). Tahoe's
1.6× is the same mechanism where the working set does not trivially fit.

Captured 2026-08-29 on Lambda `standard`-partition node `vci-steady-state-node-022`
at commit `15bac541`, one SLURM job per dataset (188733, 188734), via

```bash
for f in tabula_sapiens_100k pbmc10k; do
  SOURCE=$SCX_DATA_DIR/${f}_auto.scx WORKDIR=$SCRATCH/$f OUT_DIR=$SCRATCH N_RUNS=3 \
      sbatch --job-name=cr-$f benchmarks/scripts/bench_cellset_scatter_routes.sbatch
done
```

> [!NOTE]
> Manifest entries are the six schema-v2 results
> `results/raw/cellset_gather_scatter_routes__scx_v4reframed_{default,off,on}__{tabula_sapiens_100k,pbmc10k}_v4reframed.json`
> — **one per arm**, because a `BenchmarkResult` pooling a 2.6 cellsets/s route
> with a 541 cellsets/s one yields a `median_wall_s` describing neither and a
> `wall_s_iqr` that is merely the gap between them. They are force-added
> (`results/raw/` is gitignored) and deliberately **not promoted**: the subject
> is a reframed *copy* of a registered fixture, so no `gate_candidate.py` run
> reproduces these rows. `system.provenance.git_dirty` is `true` and
> `provenance.dirty_tracked_paths` is `[]` — the only dirt was untracked scratch
> markdown in the repo root, which is the documented-as-acceptable case in
> [`docs/benchmark_manifest.md` § Workflow](../benchmark_manifest.md#workflow).
> Every sample records `cache_policy: cold_fadvise`; the driver refuses to
> publish a median labelled cold if any sample fell back to warm, and refuses to
> report a ratio at all unless the `on` arm reached the block-index route, the
> `off` arm took the full-shard route, and the default agreed with the explicit
> `false`.

Pass `scatter_block_index=true` for a
genuinely cache-hostile cell-set run (working set ≫ cache), where the
row-group-scoped decode's bounded peak RAM is the memory-safe choice. The gate is
per-reader: the `IndexPlanDataset` defaults above are unchanged, and
`SCX_SCATTER_BLOCK_INDEX=0` remains a hard master kill-switch over both.

## Row-group LRU on the scattered gather (OPT-FORMATIO-1)

Everything above in this section was measured before the row groups a
block-index gather decodes were retained anywhere: `decode_block_index_row_runs`
decoded the touched groups per call and dropped them, and the whole-shard LRU
could not populate on that path (its `!contains` clause is part of the
eligibility). Since OPT-FORMATIO-1 the groups live in the same LRU as whole
shards under `(file_id, shard, group)` keys, bounded by the loader's byte
budget (`cache_shards` caps whole shards only); each shard's framing layout is
resolved once; and retention is **one admission verdict per plan**, taken by
the prefetch engine over every file and shard the plan touches against the
plan's share of the budget, `budget / (lookahead + 1)`, sized from the block
index with no decode — an admitted plan is pre-decoded by the L2 prefetcher
and retained by its gathers, an over-share plan is neither (a scan larger than
the cache is the one pattern an LRU makes strictly worse, so it decodes and
drops instead; a standalone `read_rows_with` outside the loaders decides for
itself against the whole budget). `cache_metrics()` reports the new half as
`row_group_hits` / `row_group_misses`; `hits` / `misses` keep meaning whole
shards.

**Same-build A/B on `read_scattered`** (256 random `(pert, ctrl)` pairs per
batch, `cache_shards=128`, `max_memory_mb=8192`, `lookahead=4`,
`scatter_block_index=True`; the arm is `SCX_ROW_GROUP_CACHE=0` — the pre-change
regime — against the shipped default; one `.so`, same fixtures, same seeded
plans; medians over the timed runs):

| format | dataset | p50 gather, off → on | peak RSS off → on | row-group hit rate (on) |
|---|---|---|---|---|
| `scx_compact_trial_g512` | pbmc3k | 69.6 → 50.5 ms (**1.38×**) | 781 → 815 MB | 99.0 % |
| `scx_compact_trial_g256` | pbmc3k | 53.8 → 51.0 ms (1.05×) | 802 → 833 MB | 99.0 % |
| `scx_compact_trial_g256` | smartseq2 | 1224 → 1233 ms (0.99×) | 1183 → 1178 MB | 0 — bypassed |
| `scx_compact_trial_g512` | smartseq2 | 1291 → 1327 ms (0.97×) | 1172 → 1174 MB | 0 — bypassed |
| `scx_compact_trial_g256` | tabula_sapiens_100k | 1051 → 1043 ms (1.01×) | 1124 → 1158 MB | 0 — bypassed |
| `scx_compact_trial_g512` | tabula_sapiens_100k | 1401 → 1380 ms (1.02×) | 1140 → 1157 MB | 0 — bypassed |

The pbmc3k `on` arm sits at ≈50 ms in every capture of this series; its `off`
arm ran 53.8 ms on the G=256 cell here against 63–67 ms in the two earlier
captures of the same cell (jobs 2930275, 2930348), so the small-file gain reads
1.05–1.38× across captures at a constant 99 % hit rate — `cpu_preemptible`
cells are not pinned to a node.

Read the two regimes with the budget. `read_scattered` asks for 128 cached
shards under 8 GiB, and `IndexPlanDataset`'s auto-tune grants what is left
after the `max_plan_size` batch buffers — on tabula_sapiens_100k (≈217 MB
decoded per shard) that is `effective_cache_shards=2`, a ≈435 MB row-group
budget against ≈890 MB of groups per 512-row batch, and smartseq2 is the same
shape. There the plan is over its share, the admission rule bypasses the LRU,
and the arm is the pre-change behaviour within run-to-run noise (0.97–1.02×;
the OFF arm itself moved 1099 → 1224 ms across captures of the same cell).
Without the rule — the first capture of this series, job 2930275 — the same two datasets
measured a **0.0 % hit rate** with +350–500 MB of resident bytes and 2–4 %
slower batches: every group was inserted and evicted before the next gather
reached it. On pbmc3k the file's eleven (G=256) or six (G=512) groups fit,
and 99 % of lookups are hits. `block_index_adoption_rate` is 1.0 on both arms
and all six cells — a cache-served group is still the block-index route, so
the gate's route floors do not move.

**Where the working set fits, the win is the estimate the review made.** The
`index_plan` benchmark's scattered scenarios run on the `_auto.scx` fixtures,
which `scx info` reports as `format_version: 4` (framed), with the default
`IndexPlanDataset` budget; against `LATEST` (`v0.16.0-opt-instruments`,
captured 2026-09-03 at `cafeb2ce` — **not** a same-build A/B, though the
intervening commits are all write-side):

| dataset | scenario | LATEST | this branch |
|---|---|---|---|
| smartseq2 | `pyscx_index_plan_random` | 65.9 s | **1.80 s** (37×) |
| smartseq2 | `pyscx_index_plan_locality` | 61.8 s | **1.79 s** (35×) |
| smartseq2 | `pyscx_index_plan_dataset_workers2` | 64.8 s | **2.28 s** (28×) |
| tabula_sapiens_100k | `pyscx_index_plan_random` | 87.7 s | **1.58 s** (56×) |
| tabula_sapiens_100k | `pyscx_index_plan_locality` | 69.7 s | **1.46 s** (48×) |
| tabula_sapiens_100k | `pyscx_index_plan_dataset_workers2` | 75.0 s | **2.12 s** (35×) |
| tabula_sapiens_100k | peak RSS (pooled max) | 2,708 MB | 2,974 MB |

`pyscx_backed_python_loop` (a Python-bound per-row loop) and
`pyscx_training_dataset` (the sequential pipeline) are flat, as is every
`cellset_gather` scenario (`SparseCellSetDataset` defaults to the whole-shard
route). The extra resident bytes are the configured budget being used for the
first time on this path.

Captured 2026-09-10 on Chimera (`cpu_preemptible` cells, `cpu_batch`
orchestrator, SLURM job 2930656) at `0f86625b` — the PR's final head, after
the per-plan admission verdict replaced the per-gather rule the earlier
captures of this series (jobs 2930275, 2930348) measured — via
`sbatch benchmarks/scripts/_run_pr25_row_group_lru_ab.sh`: 2 timed runs per
cell on smartseq2 / tabula (3 on pbmc3k) after a 3-batch warm-up, 12–50
batches per run, `gate_candidate.py --skip-capture` against `LATEST` reporting
0 peak-RSS regressions and 0 floor violations (its 26 "timing regressions" are
`DISAPPEARED` rows for triples the narrowed capture did not run).

> [!NOTE]
> Manifest entries: the ON-arm rows are the canonical triples
> `results/raw/read_scattered__scx_compact_trial_g{256,512}__{pbmc3k,smartseq2,tabula_sapiens_100k}.json`
> and `results/raw/index_plan__scx_auto__{pbmc3k,smartseq2,tabula_sapiens_100k}.json`
> (reproducible by `gate_candidate.py --benchmarks read_scattered index_plan
> --formats scx_compact_trial_g256 scx_compact_trial_g512 scx_auto`); the six
> OFF-arm rows sit beside them under `results/raw/pr25_row_group_lru_off/`
> with the same names plus that snapshot's `environment.json`, whose
> `determinism_env.SCX_ROW_GROUP_CACHE` is `"0"` — **one file per arm**, for
> the reason the note above gives. Both snapshots' `environment.json` also
> record `pyscx.file`, so the build each row measured is legible from the
> file. Force-added (`results/raw/` is gitignored); `git_dirty` is `true` only
> for the untracked scratch markdown in the repo root.

**Framing is a write-time property** (v4 writers row-group-frame all shards by
default; codec-agnostic, so it covers every codec, not just Scx1). `.scx` files
written before framing (pre-v4) carry no `BlockIndex` and silently fall back to
full-shard — **regenerate fixtures** (or `scx optimize` in place; `scx info <f>
--json` reports framed shards) before expecting the win. Sweep results:
`benchmarks/comprehensive/results/phase5/T5_sidecar_sweep.md`; driver
`benchmarks/scripts/phase5_sidecar_sweep.py`.

## Cell-set scatter routes, re-measured after the row-group LRU (phase-0 gate)

The 206x / 188x table above is the **pre-LRU** measurement, and it asked the
wrong question: it compared the two routes on one fixture and one plan shape.
Re-measured across a 2x2 of corpus size and plan locality, cold, `N_RUNS=3`,
`N_BATCHES=50`, S=64, one SLURM job per cell:

Every figure below is `median_cellsets_per_sec` / median `peak_rss_mb` read
back out of the committed manifests under
`results/raw/phase0_scatter_routes/` — `default` / `off` is the arm pair, and
the ratio is that file's own `speedup_off_over_on`:

| fixture | cells / shards | plan | `default` | `off` | `on` | off/on | peak RSS `off` -> `on` |
|---|---|---|---|---|---|---|---|
| tabula_sapiens_100k | 100 k / 7 | random | 874.8 | 899.1 | 5.40 | full-shard **166.5x** | 2 216 -> 783 MB |
| tabula_sapiens_100k | 100 k / 7 | grouped | 805.6 | 812.4 | 6.94 | full-shard **117.1x** | 2 283 -> 818 MB |
| census_1m | 1 M / 62 | random | 11.13 | 11.41 | 6.93 | full-shard **1.65x** | 14 338 -> 3 853 MB |
| census_1m | 1 M / 62 | grouped | 280.9 | 299.7 | **458.8** | **block-index 1.53x** | 13 606 -> 12 063 MB |
| pbmc10k | 11.8 k / 1 | random | 1 166.9 | 1 222.3 | 1 161.5 | 1.05x | 619 -> 678 MB |

**There is no fixed winner — there is a crossover, and a real corpus sits on
the far side of it.** The two routes scale differently:

* **Full-shard** pays to keep shards resident. It is superb while the whole
  shard set fits the cache and is touched often — ~900 sets/s on tabula's seven
  shards at a 0.9985 hit rate — and it degrades as the corpus grows: 11.4
  sets/s on census_1m's 62 shards, holding 14.3 GB to do it.
* **Block-index** pays per touched row group and holds almost nothing. It is
  roughly flat in corpus size on a given plan shape (5.40 on tabula random,
  6.99 on census_1m random) and it tracks plan locality strongly: a grouped
  plan touches ~23x fewer groups than a random one (1 354 vs 31 878 over the
  same batch count), which is why grouped census runs at 458.8.

So full-shard's advantage is 117–167x at 100 k cells, 1.65x at 1 M, and
**already inverted** at 1 M once the plan has the locality a real model's sets
have.
Block-index also holds 3.8x less memory at census scale on random plans
(3.8 GB against 14.3 GB), which is the axis that decides whether a job fits a
node at all.

**What this means for the default.** `SparseCellSetDataset.scatter_block_index`
stays `false` for now — flipping it unconditionally would cost 117–173x on
100 k-cell corpora, which is where the registered fixtures and most current
users are. But "the default is correct" is not what these numbers say. They
say the correct route is **a function of corpus size and plan locality**, the
crossover is near 1 M cells, and datasets past it — many files, millions of
cells, covariate-grouped sets — are the case the default gets wrong. Choosing
the route per dataset (or estimating `planned` bytes against the budget at
plan time, which the loader already computes for admission) is the follow-on
this gate actually motivates, and it supersedes the "flip or do not flip"
framing the phase was written around.

**Admission is the other half, and it is off at scale by default.** Row-group
retention (`cache_metrics()`, shipped default budget, random plans):

> [!WARNING]
> This table is the **before** side of the reuse-signal admission change (W10)
> and is not re-measured here. The `0.000` rows below are what an
> all-or-nothing per-plan verdict produces; the shipped rule now admits, within
> the same share, the groups two plans of the lookahead window both touch.
> Re-capture before quoting these as current.

| fixture | shards | cells | `row_group_hits` | `row_group_misses` | hit rate |
|---|---|---|---|---|---|
| pbmc10k | 1 | 11 769 | 3 597 | 46 | **0.987** |
| tabula_sapiens_100k | 7 | 100 000 | **0** | 5 681 | 0.000 |
| census_500k | 31 | 500 000 | **0** | 6 038 | 0.000 |
| census_1m | 62 | 1 000 000 | **0** | 6 098 | 0.000 |

Every multi-shard fixture retains nothing on a random plan. The rule is
`admit_row_groups = planned <= budget / (lookahead + 1)`, taken once per plan
over every file and shard it touches (`plan_engine.rs`), and `planned` is the
decoded footprint of the row groups the plan's rows land in. Grouped plans
clear it where random ones do not (census_500k 0.421, census_1m 0.447), and
removing the share divisor lifts tabula to 0.931 and census_1m to 0.310 —
so most of what is forfeited is recoverable by admitting on a reuse signal
rather than all-or-nothing per plan.

**The file count is not the variable — the shard size is.** On the real
256-file manifest fixture (tabula split into 390-cell, 1-shard, 10.3 MB files)
retention is healthy at every file count that exercises the route: 64 files
0.667, 256 files 0.603. Spreading the same plans over 1 -> 32 readers on one
1-shard fixture moves the hit rate only 0.928 -> 0.888. A many-small-file
atlas admits and retains; a few-large-shard corpus does not. The untested
corner is many files that each carry census-sized shards.

> [!NOTE]
> The census_1m random row was captured twice. A first run shared its node with
> this session's interactive shell while diagnostic probes ran there, so its
> `default` / `off` arms were potentially depressed; it measured 12.00 / 11.21 /
> 6.99 cellsets/s (ratio 1.604). The clean re-run above measured 11.13 / 11.41 /
> 6.93 (ratio 1.646) — within 3 %, with an identical block-index arm. The
> contention was immaterial here, but the published row is the clean one.

> [!IMPORTANT]
> **Size the cache, or measure the wrong thing — and the loader will tell you
> which.** census_1m's adaptive budget resolves to ~4.0 GB inside a 128 GB
> allocation — 23 of its 62 shards — and a random plan touches nearly all of
> them every batch. Under that budget the full-shard arm runs at **0.14
> cellsets/s** at a 0.215 hit rate; sized to hold all 62 shards (12.1 GB) the
> same arm runs at **11.4** at 0.988, an **80x** difference that has nothing to
> do with routes.
>
> This is **not** a silent trap. `SparseCellSetDataset` emits a `UserWarning`
> naming the pathology, the measurement and the exact remedy — *"shard-cache
> thrash detected — 78% of 8358 shard reads missed ... raise max_memory_mb to
> >=10707 (enough for ~62 shards of ~176837 KB). Raising cache_shards alone
> cannot help."* — and the 12.1 GB used above is that recommendation rounded up.
> The warning goes to **stderr**, so a capture that splits stdout and stderr
> into separate files (as `bench_cellset_scatter_routes.sbatch` does) will hide
> it from anyone reading only the `.out`. Read the `.err`. `docs/performance.md`'s
> data-load 1A capture is the same pathology from the other side. Every row in
> the 2x2 above is measured with the cache sized to the file, via the driver's
> `_sized_budget`, which may only ever *raise* the budget: an earlier version
> sized tabula *down* from 4.23 GB to 1.73 GB, starving the full-shard arm
> (hit rate 0.9987 -> 0.5408, 936 -> 1.12 sets/s) and making block-index look
> 6.2x faster. That row was discarded, not published.

> [!NOTE]
> **The premise about fixture versions changed, but only for some fixtures.**
> The 2026-08-29 capture reframed v3 / `scx1` / pyscx-0.9.1 sources, and the
> driver's docstring justified the reframe by saying both routes collapse to
> full-shard without it. That is no longer true of the fixtures this A/B uses:
> `pbmc10k`, `tabula_sapiens_100k`, `census_500k`, `census_1m`, `pbmc3k`,
> `smartseq2` and `chemogenetic_rgfp` are all v4 / framed today, and the
> block-index route is reachable on them directly — a 256-row scattered plan
> over `pbmc10k_auto.scx` reports `block_index_groups=4` at
> `scatter_block_index=True` and `full_shard_groups=4` at `False`.
>
> It is **not** true of the fixture set as a whole, and the exceptions matter:
> `census_5m_auto.scx` is **format v1** (306 shards, 15.1 GB, mixed
> `scx1`/`zstd`), and `tahoe_c38`, `replogle_k562` and the three `_lognorm`
> fixtures are v3. None carries a `BlockIndex`, so on those the block-index
> route cannot fire at all and `SparseCellSetDataset(scatter_block_index=True)`
> warns and full-shard-decodes. That is why the scale probe above stops at
> census_1m: extending it to 5M cells would mean reframing a 15 GB v1 file
> first, not just pointing the probe at a bigger fixture. The perturbation
> fixtures closest to a real STATE-style workload (`tahoe_c38`,
> `replogle_k562`) are among the unframed ones.
>
> The reframe is kept for a reason unrelated to either: `scx optimize
> --row-group-rows 256` pins the row-group geometry the arms are compared at
> and keeps `_v4reframed` naming the same subject the 08-29 rows measured.

> [!NOTE]
> Manifest entries: the fifteen schema-v2 results
> `results/raw/phase0_scatter_routes/cellset_gather_scatter_routes__scx_v4reframed_{default,off,on}__{tabula_sapiens_100k,pbmc10k}_v4reframed.json`
> and `…__{tabula_sapiens_100k,census_1m}_v4reframed_grouped.json` and
> `…__census_1m_v4reframed.json`
> — **one per arm**, for the reason the 08-29 note gives, and in their own
> subdirectory so the tracked 08-29 rows that back the table above are not
> overwritten by a re-measure. Force-added (`results/raw/` is gitignored) and
> deliberately **not promoted**: the subject is a reframed *copy* of a
> registered fixture, so no `gate_candidate.py` run reproduces these rows.
> Every sample records `cache_policy: cold_fadvise`; the driver refuses a
> median labelled cold if any sample fell back to warm, refuses a ratio unless
> each arm reached its own route, and refuses to start at all against a pyscx
> whose `cache_metrics()` lacks the `row_group_*` counters — a build predating
> OPT-FORMATIO-1 would otherwise reproduce the pre-LRU numbers under this
> heading, which is what the first attempt at this capture did.
> `provenance.dirty_tracked_paths` names this PR's own benchmark-harness
> edits, which are the subject of the capture rather than a contaminant; no
> `scx-*` crate is modified, so the measured `.so` is `main`'s.
>
> ⚠️ **Capture vintage differs across the 2×2.** The tabula and pbmc *random*
> rows were captured before the driver's `_sized_budget` helper existed
> (`cache_sizing: null` in their metadata); the census rows after it. That does
> not move their values — tabula needs 1.73 GB and the adaptive tuner already
> grants 4.23 GB, so sizing is a no-op there, which is why the helper now
> refuses to lower a budget — but the artifacts are not one vintage and should
> not be read as a single campaign.
>
> **What is not manifested**: the `row_group_*`, budget-sweep, scale and
> file-count tables above come from read-only `cache_metrics()` probes over
> the same fixtures and plan generator, not from timed benchmark arms — they
> report counters, not wall clock, so they carry no `BenchmarkResult`. The two
> captures ran on the same node this session's probes ran on; the default
> arm reproduced to 0.1 % across them (873.8 / 874.8 sets/s), so contention
> was immaterial, but the timed rows above are the manifested ones and the
> counter tables are diagnostics.

## Reuse-signal admission (W10)

An over-share plan used to forfeit row-group retention outright — one verdict
per plan, all or nothing — which is what the `0.000` rows in the table above
are. It now keeps the groups **another plan of the lookahead window also
touches**, hottest-first and only while they fit that same
`budget / (lookahead + 1)` share, and still decode-and-drops its cold tail.

**Same-build A/B**, one build with the arm selected by `SCX_ROW_GROUP_ADMIT`
(`plan` = the pre-W10 rule, `reuse` = shipped): 12 rounds per dataset, the
within-pair order flipped on alternate rounds so a first-slot advantage cancels,
one timed run per invocation, `cellset_gather` driven in process on one host.
`p` is an exact two-sided sign test over the per-round ratios.

The new `gather_hot_control_cold_tail` arm is 8 control sets repeated verbatim in
every batch plus 56 drawn fresh, at S=64 under a 256 MB budget. It **does not run
on pbmc3k**: 56 × 64 draws from 2,700 rows make the "cold" tails overlap 0.55 of
their own rows, so the arm records `applicable: false` rather than timing a warm
tail under a cold name.

| tabula_sapiens_100k, 12 rounds | plan | reuse | ratio | reuse wins | p |
|---|---|---|---|---|---|
| `cellsets_per_sec__gather_hot_control_cold_tail` | 32.7 | 32.9 | 1.003× | 6/12 | 1.000 |
| `us_per_cell__gather_hot_control_cold_tail` | 477.3 | 475.5 | 1.002× | 7/12 | 0.774 |
| `peak_rss_mb__gather_hot_control_cold_tail` | 1,176 | 1,196 | **0.987×** | 0/12 | **0.000** |

| retention, same runs | plan | reuse |
|---|---|---|
| `row_group_hit_rate__gather_hot_control_cold_tail` | 0 | **0.0421** |
| `reuse_admissions__gather_hot_control_cold_tail` | 0 | 11 |
| `admitted_group_bytes__gather_hot_control_cold_tail` | 0 | 7.42 GB |
| `rejected_group_bytes__gather_hot_control_cold_tail` | 162.2 GB | 154.7 GB |

**The hit rate has no derivable ceiling here, and an earlier version of this
section claimed one.** `0.0421` is a fraction of group *lookups*; the arm's
control population is 8 of 64 *sets*. How many lookups a control set generates
depends on how many distinct row groups its rows land in — a property of the
layout, not of 8-of-64 — so "34 % of what any policy could reach" did not
follow and has been removed. The arm records `control_set_fraction` for context
and nothing is divided by it.

**Read it beside the rejection.** 154.7 GB of the verdict's lookups are still
refused, against 7.4 GB admitted: the cold tail is not being let in. A policy
that raised the hit rate by admitting it would show a higher rate *and* a
collapsed rejection, and neither number alone separates the two.

**The cost is +20 MB of peak RSS on the arm (1.7 %), and nothing else moved.**
That RSS row is the only one in the capture with a reliable verdict, at 0/12
rounds and p = 0.000 — it is the configured budget being used for retention that
did not previously happen, bounded by the share. Every other metric, on both
datasets, is "no reliable difference": the four pre-existing `gather_*`
scenarios and `us_per_cell__collate` on tabula (p 0.146–0.774) and on pbmc3k
(p 0.146–0.774). Throughput on the arm itself is flat — 1.003× at 6/12 rounds
and p = 1.000.

> [!WARNING]
> ⚠️ **An interactive probe said the opposite, and it was wrong.** Three probe
> pairs on this fixture, one run per arm on a shared node, read 20.7 / 23.1 /
> 23.0 sets/s for `plan` against 19.0 / 21.9 / 19.1 for `reuse` — a consistent
> ~10–17 % slowdown that does not survive 12 interleaved rounds. The probes are
> not in this table and are recorded only as the reason the capture was run
> before anything was tuned.

> [!NOTE]
> Manifest entry: `results/raw/phase5_admission/admission_ab.json` (48 rounds,
> each carrying its full `BenchmarkResult` envelope) beside `provenance.json`,
> written by the running job from `SLURM_JOB_PARTITION` rather than from the
> `#SBATCH` directive. SLURM job **2964125**, `cpu_batch_high_mem`, node
> GPU104C, at `b285ffe2`, via
> `sbatch --exclude=… benchmarks/scripts/_run_phase5_admission_ab.sh`;
> summarised by `_phase5_admission_summary.py`. Force-added (`results/raw/` is
> gitignored). Every figure above was checked programmatically against that
> file.
>
> **Not measured**: census at any scale (a `cellset_gather` census cell does not
> finish — census_500k was killed at 205 minutes still on run 1 of 3), and any
> workload whose hot set is a different fraction of its lookups than this arm's
> 8-of-64. The arm's own numbers bind to that shape and to the 256 MB budget
> that puts a plan over its share at all.

## W10's two factors, isolated (three-arm A/B)

⚠️ **The `cellset_gather` capture above measures only one of this phase's two
changes, and it is not the one that moved.** The chunked parallel group decode
is not gated by `SCX_ROW_GROUP_ADMIT`, so it is identical in the `plan` and
`reuse` arms and cancels out of every ratio they produce; and that benchmark's
hot/cold arm caps its hit rate at 8/64 = 0.125 by construction. The capture
below adds a third arm — one group per chunk, i.e. the pre-change serial
decode — and runs the benchmarks whose regime (R3) funded the work. That arm was
selected by `SCX_ROW_GROUP_DECODE_CHUNK=1`, which review renamed to the boolean
`SCX_ROW_GROUP_SERIAL_DECODE=1`; the committed `provenance.json` records the name
that actually ran and is left as it was.

Three arms on one build, 12 rounds per cell, arm order rotated by round so each
arm occupies each slot equally. `reuse` vs `plan` isolates the admission policy;
`reuse` vs `serial` isolates the decode. `p` is an exact two-sided sign test
over per-round ratios with ties dropped.

**`read_scattered` is the decisive cell.** It drives `IndexPlanDataset` at
`cache_shards=128` under an auto-tuned two-shard budget, which is the regime
`thresholds.yaml` records as sitting at a **0.000** row-group hit rate *by
design* — a plan over its budget share used to forfeit retention outright.

| tabula_sapiens_100k, g256 | serial | reuse | ratio | wins | p |
|---|---|---|---|---|---|
| `gather_latency_ms_p50` | 1,018.5 | 434.1 | **2.39×** | 12/12 | 0.000 |
| `gather_latency_ms_p99` | 1,121.8 | 456.4 | **2.43×** | 12/12 | 0.000 |

| smartseq2, g256 | serial | reuse | ratio | wins | p |
|---|---|---|---|---|---|
| `gather_latency_ms_p50` | 1,192.3 | 431.3 | **2.79×** | 12/12 | 0.000 |
| `gather_latency_ms_p99` | 1,232.6 | 471.6 | **2.60×** | 12/12 | 0.000 |

> [!NOTE]
> ⚠️ **The decode was also a reliable regression on the high-hit-rate path.
> It was fixed, and the fix was then re-captured rather than asserted.** In the
> capture above, `index_plan` / tabula random-plan `gather_latency_ms_p50` read
> 27.96 → 29.17 ms (1 of 12 rounds won, p = 0.006) and p99 29.91 → 31.84 ms
> (2/12, p = 0.039) against the serial arm. The cause was routing every run
> through rayon including LRU hits — ~38,500 hits against 391 misses on that
> path, where a "decode" is a mutex lookup. Residents are now probed and served
> serially and only misses are dispatched. **The whole three-arm capture was then
> re-run on the fixed build** (job 2967288, head `dabdbd32`, same script, same
> three cells, 12 rounds), and both regressed rows are neutral:

| index_plan / tabula, random plans | serial | reuse | ratio | wins | p |
|---|---|---|---|---|---|
| `gather_latency_ms_p50`, as first captured | 27.96 | 29.17 | 0.952× | 1/12 | **0.006** |
| `gather_latency_ms_p50`, re-captured | 28.03 | 27.76 | 1.01× | 8/12 | 0.388 |
| `gather_latency_ms_p99`, as first captured | 29.91 | 31.84 | 0.937× | 2/12 | **0.039** |
| `gather_latency_ms_p99`, re-captured | 30.19 | 29.57 | 1.01× | 9/12 | 0.146 |

The counter says the same thing and says it exactly:
`parallel_group_decodes` on that path is **19,441 → 0** between the two captures
— at a 0.9899 hit rate nothing reaches rayon any more — while `row_group_hits`
(38,495) and `row_group_misses` (391) are unchanged to the unit. On
`read_scattered`/tabula, where the misses are real, the shipped arm dispatches
2,903 of its 2,911 misses (the eight are groups a peer inserted between the
probe and the dispatch) against 3,427 in the `plan` arm.

The decode win reproduces on the same build, slightly larger than first
measured:

| re-captured decode (`serial` → `reuse`) | serial | reuse | ratio | wins | p |
|---|---|---|---|---|---|
| tabula g256 `gather_latency_ms_p50` | 1,011.1 | 397.7 | **2.56×** | 12/12 | 0.000 |
| tabula g256 `gather_latency_ms_p99` | 1,147.7 | 421.3 | **2.70×** | 12/12 | 0.000 |
| smartseq2 g256 `gather_latency_ms_p50` | 1,184.6 | 407.3 | **2.90×** | 12/12 | 0.000 |
| smartseq2 g256 `gather_latency_ms_p99` | 1,229.6 | 434.2 | **2.83×** | 12/12 | 0.000 |

> [!WARNING]
> ⚠️ **One regression the split did not remove: peak RSS on the
> triple-buffered loader.** `peak_rss_mb` on `pyscx_training_dataset` /
> `pyscx_index_plan_dataset_workers2` reads 3,232 → 3,460 MB against the serial
> arm (1/12, p = 0.006) on the re-capture, and 3,239 → 3,485 MB (1/12,
> p = 0.006) on the first — the same ~230 MB, reliable in both. That is the
> chunked decode holding a pool-width chunk of groups live plus rayon's
> per-worker scratch, and it is the cost the chunking bounds rather than
> removes. It stays far under the scenario's own 8,166 MB budget, and it is
> **not** a bound this capture claims is tight. The `index_plan` /
> `backed_python_loop` peak-RSS regression the first capture reported
> (2,914 → 2,971 MB, 2/12, p = 0.039) does **not** reproduce
> (2,999.6 → 2,912.1 MB, 8/12, p = 0.388).

**Every work counter is identical on all 12 rounds across those two arms** —
`block_index_groups`, `block_index_adoption_rate`, `row_group_hits`,
`row_group_misses`, `admitted_group_bytes`, `reuse_admissions`. The arms decode
the same groups and retain the same bytes; they differ only in whether the
decodes are batched a pool-width chunk at a time across shards. That identity
is the control, and it is why the latency ratio can be attributed to the
batching rather than to a changed working set.

And the admission policy, on the same runs:

| tabula_sapiens_100k, g256 | plan | reuse | ratio | wins | p |
|---|---|---|---|---|---|
| `gather_latency_ms_p50` | 445.8 | 434.1 | 1.05× | 10/12 | 0.039 |
| `row_group_misses` (fewer is better) | 3,427 | 2,911 | **1.18×** | 12/12 | 0.000 |
| `row_group_hit_rate` | **0** | **0.1506** | — | — | — |
| `reuse_admissions` | 0 | 11 | — | — | — |

18 % fewer row-group decodes on every one of 12 rounds. On smartseq2 the same
contrast gives 2,913 → 2,683 misses (−8 %). The retention is real; the latency
it buys is not separable from noise, and the decode is where the time went.

> [!WARNING]
> ⚠️ **The 1.05× p50 row above did not reproduce, and is withdrawn as a
> latency claim.** The re-capture (job 2967288, head `dabdbd32`) reads the same
> contrast at 416.1 → 397.7 ms, 1.05× but **8/12, p = 0.388**, and smartseq2 at
> 419.4 → 407.3 ms, 1.01×, 9/12, p = 0.146. A 10/12 at p = 0.039 is one round
> from the 0.05 line, so this is what a marginal sign test does on a second
> sample, not a changed build. **The counters reproduced to the unit** — 3,427
> → 2,911 misses, 0 → 0.1506 hit rate, 0 → 11 `reuse_admissions` on tabula and
> 2,913 → 2,683 / 0 → 0.079 / 0 → 15 on smartseq2, each identical across the
> two captures. What admission buys is measured as retention; on these two
> cells it does not buy measurable wall clock.

`index_plan` at its own 8 GB default budget shows the admission contrast at
noise — its plans **fit** their share, so both arms read a 0.99 hit rate and the
policy never engages. That is the expected result and worth stating: this policy
only acts where a plan is over its share.

> [!NOTE]
> Manifest entry: `results/raw/phase5_factors/factors_ab.json` (108 rounds, each
> carrying its full `BenchmarkResult` envelope) beside `provenance.json`,
> written by the running job. SLURM job **2966558**, `cpu_batch_high_mem`, at
> `c5d8a62a`, via `benchmarks/scripts/_run_phase5_factors_ab.sh`; summarised by
> `_phase5_factors_summary.py`. Force-added. Every figure above was checked
> programmatically against that file.
>
> The re-capture on the fixed build is
> `results/raw/phase5_factors_recapture/factors_ab.json` (108 rounds) beside its
> own `provenance.json`: SLURM job **2967288**, same partition, same node
> (GPU0F98), same driver, at `dabdbd32`. Every re-captured figure above was
> checked programmatically against that file.
>
> **Not measured**: census at any scale, and `cellset_gather` under the third
> arm (its hot/cold arm's hit-rate ceiling makes it the weakest witness of
> either factor).
