# Analysis Accelerators (CPU)

> Part of [SCX performance](README.md). The per-op pages are
[PCA and pseudobulk](accel-pca-pseudobulk.md) and
[QC, DE, scoring, integration](accel-qc-de-integration.md); GPU DE device
residency is on the [GPU page](gpu.md#gpu-de-device-residency--gene-chunk-windowing-phase-4-task-45).

Benchmarked on 1M cells (CELLxGENE Census), HVG-selected (2000 genes):

| Operation | SCX (s) | scanpy (s) | Speedup vs scanpy |
|-----------|---------|------------|-------------------|
| PCA (covariance, 50 PCs, 2K HVGs) | **4.2** | 8.0 | **1.9x** |
| Wilcoxon rank-sum DE (pre-ranking) | **5.4** | 17.3 | **3.2x** |
| Leiden (Rust-native) | **55** | 2,226 (leidenalg) | **40x** |

Full pipeline (PCA -> kNN -> UMAP -> Leiden -> DE) on 1M cells: **870s** (vs 3,971s — **4.6x faster**).

## CPU stage profile — the decode/reduction ranking oracle (Phase-2 task 2.0)

Before any Phase-2 `×` speedup claim, the CPU per-stage profiler
(`SCX_CPU_PROFILE=1`, `scx_format_io::profile`) breaks a streaming accelerator op
into **io / decode / reduction / marshalling** so we know which stage bounds it.
Captured on the **backed streaming** path (`pyscx.open(scx).to_anndata(backed=True)`
→ the out-of-core shard source the Phase-2 §5.1 work targets) via
`benchmarks/scripts/profile_cpu_stages_backed.py`. Times are median-of-runs ms;
`bound` is the larger of decode vs reduction:

| Dataset | Op | wall ms | decode ms | reduction ms | marshalling ms | Bound |
|---|---|--:|--:|--:|--:|:--|
| pbmc3k | pca | 765 | 18 | 192 | 0.5 | **reduction** |
| pbmc3k | hvg | 80 | 34 | 14 | 0.0 | **decode** |
| pbmc3k | normalize | 81 | 40 | 0 | 0.0 | **decode** |
| pbmc10k | pca | 2819 | 276 | 1826 | 1.8 | **reduction** |
| pbmc10k | hvg | 743 | 538 | 147 | 0.0 | **decode** |
| pbmc10k | normalize | 821 | 535 | 0 | 0.0 | **decode** |
| smartseq2 | pca | 6395 | 1499 | 3446 | 7.1 | **reduction** |
| smartseq2 | hvg | 3951 | 2924 | 772 | 0.0 | **decode** |
| smartseq2 | normalize | 3878 | 3020 | 0 | 0.0 | **decode** |
| tabula_sapiens_100k | pca | 8147 | 1756 | 4253 | 13.4 | **reduction** |
| tabula_sapiens_100k | hvg | 4671 | 3329 | 1033 | 0.0 | **decode** |
| tabula_sapiens_100k | normalize | 5023 | 3510 | 0 | 0.0 | **decode** |
| census_500k | pca | 32801 | 7259 | 17223 | 74.6 | **reduction** |
| census_500k | hvg | 17470 | 12474 | 3905 | 0.0 | **decode** |
| census_500k | normalize | 20714 | 13527 | 0 | 0.0 | **decode** |
| census_1m | pca | 138311 | 88764 | 30587 | 183.6 | **decode** |
| census_1m | hvg | 27888 | 18553 | 7677 | 0.0 | **decode** |
| census_1m | normalize | 39315 | 22247 | 0 | 0.0 | **decode** |

**What the oracle says (ranks the Phase-2 work):**
- **HVG is decode-bound at every scale** — decode > reduction by a *measured* margin
  (2.4× at 1M; Σ(buckets)/wall ≈ 85–94 %, so both stages are instrumented) → §5.1
  bounded-ordered decode-prefetch attacks its dominant cost directly; highest-value
  Phase-2 target.
- **`normalize_total` is decode-heavy** — decode is the single largest *measured*
  stage (~57 % of wall at 1M). Its `reduction` reads 0 **by construction, not by
  measurement**: the lazy row-sum + scale path has no `reduction_guard`, so ~43 % of
  wall is unattributed. Treat it as "decode-heavy, reduction not instrumented on this
  path" — *not* as a decode-vs-reduction ratio peer of HVG. Decode-prefetch still
  helps (decode dominates the accounted stages), but confirm with a guarded row-sum
  pass before ranking preprocess against HVG.
- **PCA is reduction-bound up to ~500k but FLIPS to decode-bound at census_1m**
  (decode 88.8 s vs reduction 30.6 s) — the streaming shard decode overtakes the
  randomized-PCA compute at atlas scale. So decode-prefetch pays off for PCA *only*
  at ≥~1M cells; below that, PCA gains come from compute (Phase-2 task 2.7 fusions) /
  marshalling (Phase-2 task 2.4), not decode.
- `marshalling` stays sub-0.2 s even at 1M-cell embeddings — Phase-2 task 2.4 flat
  marshalling is a low-priority micro-win on this path.

`io` is ~0 because these are local mmap reads (page-fault cost folds into `decode`);
the `decode` bucket is the §5.1 oracle signal. The `reduction` bucket covers per-shard
kernel accumulation; PCA's post-streaming dense SVD and `normalize_total`'s row-sum
pass fall outside it (so Σ(buckets) < wall for those). **Instrumentation gaps** (not in
the `decode` bucket): the *framed/block-index* CSC route (`decode_block_index_row_runs`)
and the typed-dtype path (`read_shard_from_entry_native`) — a *full-shard* CSC or CSR
decode is captured; the cloud range-read readers that bypass the local `ScxReader`
inner. Non-`auto` codec / shard-size / cold-vs-warm-storage axes are staged, not yet
captured. Source: backed-streaming capture
(`benchmarks/scripts/profile_cpu_stages_backed.py`, `SCX_CPU_PROFILE=1`), 3 runs,
median across runs, `results/raw/accel_cpu_profile_backed__*`.

## Bounded ordered decode-prefetch (Phase-2 task 2.1)

The 2.0 oracle above ranked HVG (and `normalize_total`) as **decode-bound at every
scale**. Task 2.1 attacks that directly with a **bounded, ordered decode-prefetch**
pipeline (`scx_accel::prefetch::for_each_shard_ordered`): up to `depth` shards decode
concurrently on the rayon pool while the reduction runs on the calling thread **in
strict shard order**. Because consumption stays single-threaded and ordered, the
reduction sees exactly the sequential accumulation order — the result is **bit-identical**
to the pre-2.1 loop (unit-proven: `prefetch::tests::ordered_accumulation_is_bit_identical_to_sequential`),
while decode now overlaps reduction and runs across shards. Applied to the decode-bound
streaming kernels: HVG (`streaming_mean_var{,_expm1}`, `streaming_clip_square_sum`, and
the batched variants), `score_genes`, PFlog baselines, and pseudobulk aggregation.

Two knobs (both read once per process via `OnceLock`, so set them in the environment
before the first accelerator call; both default to the safe/bit-exact behaviour):
- **`SCX_ACCEL_PREFETCH_DEPTH`** — max shards decoded-but-unconsumed (default 4, capped
  by the rayon pool size). `0`/`1` disables prefetch (sequential fallback). New shards are
  spawned only as one drains in shard order, so the decoded-but-unconsumed set — and hence
  extra peak RSS — stays bounded to `depth` decoded shards **even under a head-of-line
  stall** (regression-tested). This is a fixed small cap, not a per-shard byte budget.
- **`SCX_ACCEL_REDUCTION_MODE`** — `stable` (default; ordered, bit-exact) vs `parallel`
  (per-worker accumulators merged at the end, worker count derated by the CPU memory
  budget; **tolerance-only** because float summation reorders). The parallel mode is opt-in
  and reached **only** by the offset-independent per-column reductions (unbatched HVG
  moments, clipped-square-sum); the offset-dependent kernels (batched HVG, `score_genes`,
  PFlog, pseudobulk) always use the ordered path and ignore `parallel`.

`for_each_shard_ordered` must not be called from within a rayon parallel region (the drain
blocks the calling thread on the channel; a saturated pool worker would deadlock) — it
detects a worker-thread caller and falls back to sequential decode, and this is why PCA
(which owns inner rayon pools) is a deferred follow-up rather than wired here.

**The win is multi-shard-gated.** A single-shard file (≤ the 16 384-row shard target,
e.g. `pbmc3k`/`pbmc10k`) hits the `n_shards == 1` guard and runs the sequential path
unchanged — verified no-regression locally (pbmc10k HVG 802 ms sequential vs 797 ms
prefetch, within noise; the file has 1 CSR shard). The overlap only pays off where
there are many shards to decode ahead — the atlas tier (`tabula_sapiens_100k` ~6 shards,
`census_500k`/`census_1m` tens of shards).

**Measured before/after** (same host + `.so`; prefetch off = `SCX_ACCEL_PREFETCH_DEPTH=1`
vs default depth 4), backed-streaming wall-clock (median of runs):

| Dataset | Op | wall off (ms) | wall on (ms) | Speedup |
|---|---|--:|--:|--:|
| tabula_sapiens_100k | hvg | 4930.8 | 2179.4 | **2.26×** |
| census_500k | hvg | 16714.1 | 7255.7 | **2.30×** |
| census_1m | hvg | 29150.5 | 12735.3 | **2.29×** |
| tabula_sapiens_100k | normalize | 5013.6 | 4610.6 | 1.09× |
| census_500k / 1m | normalize | 20477 / 38584 | 20817 / 38423 | ~1.0× |
| tabula/500k/1m | pca | 8364 / 31368 / 138220 | 8222 / 31613 / 135041 | ~1.0× |

**HVG — the fully-prefetched kernel — is ~2.3× faster at every multi-shard scale**, the
first measured Phase-2 win. In the profiler the `decode` bucket now *exceeds* wall
(**Σ/wall** — the profiler's summed per-shard decode time divided by wall-clock, so 100 %
means "decode accounted for the entire run" and anything above it means decode threads
overlapped — 220–275 %) because it sums each worker's decode time and those run concurrently;
the wall drop is the real gain. `normalize_total` (a pyscx lazy-transform path, not one of
the wired `scx-accel` streaming kernels) and `pca` (streaming decode-prefetch deferred —
see above) are flat by construction, confirming the change is scoped to the kernels it
touched and introduces no regression elsewhere. Peak-RSS held flat (bounded to `depth`
decoded shards). Source: `pf21_cap` capture on `cpu_preemptible`,
`benchmarks/scripts/profile_cpu_stages_backed.py` before/after.

**Deferred to a measured Phase-2 follow-up:** PCA streaming decode-prefetch (its
covariance/transpose passes already own inner rayon parallelism — wrapping them adds a
nested-pool interaction that needs its own measurement; PCA is decode-bound only at
≥~1M) and the CSC mean/var kernels (reached via `&dyn ColumnShardSource`, which would
need a `+ Sync` dyn boundary change through the pyscx capability-detection layer). The
`for_each_csc_shard_ordered` sibling primitive ships and is tested, ready for that
follow-up.

## Decode-prefetch beyond `scx-accel` (Phase-4 task 4.2)

2.1 applied the pipeline only to the `scx-accel` streaming kernels, because it *lived*
in `scx-accel` — above two of its three natural consumers. `scx-format-io` and `scx-gpu`
sit below that crate in the dependency graph, so neither could reach it, and both still
decoded one shard at a time on the consuming thread. Task 4.2 moved the module to
`scx-format-io` (where the `ShardSource` / `ColumnShardSource` traits it is generic over
are already defined) and wired up the loops that could not previously see it:

| Consumer | What now prefetches |
|---|---|
| `scx-format-io/src/backed/aggregate.rs` | the 15 aggregation kernels behind QC, filtering, `col_*` and `normalize_total`'s row sums |
| `pyscx/src/projected_agg.rs` | the 17 column-projected CSR twins |
| `pyscx/src/lazy_transform/dataset.rs` | the 8 `streaming_*` lazy/transformed kernels |
| `scx-gpu/src/gpu_shard_source.rs` | GPU staging — replaces a single scoped decode thread + `sync_channel(1)` |

`scx_accel::prefetch` re-exports the surface with the error type pinned to `AccelError`,
so the eight 2.1 call sites and both `SCX_ACCEL_*` knobs above are unchanged; the knobs
now govern these loops too, including GPU staging.

**Bit-identity is tested, not asserted.** `scx-format-io`'s
`prefetched_aggregations_are_bit_identical_to_a_sequential_loop` compares every touched
aggregation against a hand-written sequential reference on f64 bit patterns, and
`pyscx/tests/test_prefetch_equivalence.py` compares 19 arrays across the three **CPU**
consumers — backed, projected and lazy — between two subprocesses at
`SCX_ACCEL_PREFETCH_DEPTH=1` and the default (the depth is a process-wide `OnceLock`, so
one interpreter cannot hold both arms). It does **not** reach `gpu_shard_source`; GPU
equivalence is discharged by the cargo suites and the failure-set A/B described below, not
by hex bit-identity.

Both fixtures needed a **cancelling ±1e16 pair** to be worth anything. The kernels reduce
per-shard *partial* sums, so what has to be order-sensitive is the merge of a handful of
numbers — and the obvious "cycle through three magnitudes" fixture is completely blind to
it: 0 of the 119 non-identity permutations of a 5-shard visit order changed any column
sum. Companion tests (`fixture_is_sensitive_to_shard_order`,
`test_fixture_would_expose_a_reordering`) pin the property so the equivalence assertions
cannot quietly go vacuous.

**Note on peak RSS.** Decoded-but-unconsumed shards go from one (two on the GPU staging
path) to `depth`, default 4. That is the one way this change can regress, so the capture
below reports peak RSS alongside wall-clock. `prefetch::clamp_prefetch_depth` plus the
exact per-shard `nnz` in `ShardEntryLite` is the derating lever if it ever needs one.

**Measured before/after** (SLURM job 2708500, `cpu` partition, one host, 24 cores,
`RAYON_NUM_THREADS=24`, `--release`, **one build** with `SCX_ACCEL_PREFETCH_DEPTH=1` vs
the default 4, 3 runs, median wall):

| Dataset | shards | `qc` | `filter_genes` | `filter_genes_real` | `filter_cells` | `normalize` |
|---|--:|--:|--:|--:|--:|--:|
| pbmc10k | 2 | 1.02× | 1.04× | 0.98× | 0.99× | 0.97× |
| smartseq2 | 8 | **2.18×** | **2.07×** | **2.03×** | **2.11×** | 1.20× |
| tabula_sapiens_100k | 14 | **2.13×** | **2.00×** | **2.13×** | **2.18×** | 1.24× |
| census_500k | 62 | **2.09×** | 1.86× | 1.80× | 1.78× | 1.16× |
| census_1m | 124 | **2.70×** | **2.30×** | **2.35×** | **2.29×** | 1.35× |

Arm order is `off` then `on`, which if anything warms the page cache *for* the `on` arm —
the opposite of a safeguard. What actually controls for it is
`profile_cpu_stages_backed.py`'s per-(op, dataset) warm-up run, which precedes the timed
runs in both arms.

`qc` at census_1m goes 33.2 s → 12.3 s. The profiler shows why: the decode bucket is
26.1 s sequential and 28.7 s prefetched — **the same work, ~10 % concurrency overhead** —
but Σ/wall moves from 79 % to **234 %**, i.e. it is now summing workers that run at the
same time. That is overlap, not less decoding.

`normalize` gains least (1.16–1.35×) and its Σ/wall only reaches 84 %, because the
lazy/transformed kernels still apply the transform chain on the *consumer* thread; only
the decode ahead of it overlaps. Moving transforms into the workers is possible — the
position each shard needs is `shard_range(idx).0`, not a running cursor — but it is a
separate change with its own equivalence argument.

**Two things the table is not.** `pbmc10k` is 2 shards, so the pipeline mostly no-ops and
those columns are noise, exactly as 2.1 found — the win is multi-shard-gated. And `hvg`
is **not** a control here even though 2.1 already prefetched it: `SCX_ACCEL_PREFETCH_DEPTH`
is global, so the `off` arm disables 2.1's HVG prefetch too, and HVG's 1.93–2.62× in this
capture is a re-measurement of the 2.1 result, not a 4.2 gain. The genuine control is
**`pca`**, whose prefetch is still deferred and which stays flat at 1.01–1.04× at the three
largest scales (`smartseq2` reads 0.90×, a single outlier in a median-of-3 on a 4-shard
file — reported rather than dropped).

**Peak RSS: flat at the ceiling, visible in the middle.** The extra memory is
`(depth − 1)` decoded shards, and the capture shows exactly that. Final process high-water
per arm is **26 437 MB off vs 26 346 MB on at census_1m (−0.3 %)** — the ceiling is set by
the obs frame and the largest decode buffers, not by the in-flight count. But partway
through, on `smartseq2`, the prefetch arm sits at 2 116 MB against 1 426 MB (**+48 %**,
≈ 3 × one decoded shard): a workload whose baseline is small enough for three extra shards
to matter *will* see it. `prefetch::clamp_prefetch_depth` plus the exact per-shard `nnz` in
`ShardEntryLite` is the lever if that ever needs bounding; it is deliberately not wired
yet.

Raw JSON under the job's `raw-off` / `raw-on` directories; `results/raw/` is gitignored, so
the numbers above cite the job and the two arms rather than a checked-in artifact, as 4.0b
and 4.1 do.

**GPU staging.** (SLURM job 2709055, H100, `benchmarks/scripts/_run_4_2_gpu_main_vs_branch.sh`
→ `profile_gpu_staging.py`, 3 runs, median wall, `SCX_DISABLE_CUDA_GRAPHS=1`.) The CPU table
above does not cover this path; it has its own capture.

| Op | Dataset | `main` | branch | Speedup | host-decode Σ/wall | peak RSS |
|---|---|--:|--:|--:|--:|--:|
| HVG | tabula_sapiens_100k | 5 590.8 ms | 4 487.9 ms | **1.25×** | 72.7 % → 92.8 % | 2 049 → 2 596 MB |
| HVG | census_500k | 17 878.0 ms | 11 773.1 ms | **1.52×** | 86.4 % → 133.1 % | 3 129 → 3 394 MB |
| HVG | census_1m | 31 180.5 ms | 18 443.8 ms | **1.69×** | 89.1 % → 160.8 % | 4 219 → 4 297 MB |
| DE | tabula_sapiens_100k | 255 726.0 ms | 127 819.2 ms | **2.00×** | 96.0 % → 208.6 % | 2 164 → 2 650 MB |
| DE | census_500k | 1 019 749.4 ms | 391 682.1 ms | **2.60×** | 98.6 % → 239.9 % | 3 113 → 3 557 MB |

Both arms are two builds in one job on one node, each at its own defaults — `main` ignores
`SCX_ACCEL_PREFETCH_DEPTH` for staging, the branch uses its default of 4.

§9.12's diagnosis holds, and `main`'s arm quantifies it: with a single one-ahead decode
thread, GPU DE is **96–99 % host-decode-bound** and GPU HVG 73–89 %. The host side is
essentially the whole ceiling, which is why widening it pays. census_500k DE goes from 17
minutes to 6.5.

**An earlier revision of this section reported 1.73–2.28× and 2.59–3.00×.** Those came from
comparing `SCX_ACCEL_PREFETCH_DEPTH=1` against `4` on one branch build, and depth 1 is *not*
`main`: at depth 1 the pipeline declines to engage and staging decodes with **zero** decode
threads, whereas `main` unconditionally spawned a one-ahead `std::thread` +
`sync_channel(1)` for multi-shard input. The depth-1 arm is slower than `main` — by ~12 % on
census_500k DE — so the ratios were inflated. The table above is the two-build
`main`-vs-branch measurement (SLURM job 2709055) that replaced them.

**Nothing device-side changed, but the `htod` column is not the evidence for that.** The
bucket is recorded around `PinnedCsrSlot::stage()` — a host memcpy into the pinned buffer —
and the asynchronous `upload_to` runs after the timer closes, so its flatness (±4.4 % here)
says a single-threaded host memcpy did not change, which it could hardly do. The real
argument is structural and checkable from the diff: the pinned 2-slot ring, the host-side
`pinned_events` gate and both device event gates are untouched, because the pipeline
already delivers to the calling thread in shard order. Host-decode Σ/wall exceeding 100 %
is the pipeline working: it sums concurrent workers.

**Peak RSS rises by +78 to +547 MB (+2 % to +27 %)**, largest on the smallest fixture. It is
the `(depth − 1)` decoded-shards model, but measured against `main` rather than against a
zero-thread arm the increment is smaller than it first appeared: `main` already held ~2
shards, so the true delta is roughly two extra shards, not three. (An earlier revision
quoted +19–46 % from the depth-1 comparison.) GPU staging otherwise keeps host RSS at
1.7–4.5 GB, so nothing hides it — worth knowing before raising
`SCX_ACCEL_PREFETCH_DEPTH` on a memory-tight GPU host.

Correctness on GPU is separately verified: `cargo test -p scx-gpu` 193/0 and
`-p scx-accel --features gpu` 19/0, plus a failure-set A/B of the pyscx GPU suites against
`2055f74f` — **11 failures on each arm, branch-only list empty** — so rewriting the staging
loop every GPU streaming op shares introduced no regression.

Where the pipeline declines to engage — `RAYON_NUM_THREADS=1`, or a caller that is itself
a rayon worker — the GPU staging path is now fully sequential, where the old dedicated
`std::thread` overlapped one shard ahead unconditionally. Accepted: both are an explicit
"no ambient parallelism" configuration, and the worker-thread guard is what keeps a
nested call from deadlocking.

## Decode-prefetch reaches the DE kernels

Task 4.2 wired every loop that could reach the pipeline. Three streaming DE
kernels could not: `wilcoxon_rank_sum_streaming`, `pdex_ref_streaming` and the
`pts` counting pass each ran their own `for shard_idx in 0..n_shards`, so neither
knob above governed them and neither did the row-projection shard skip the
`ShardSource::visible_shard_indices` hook gives every driver consumer.

They are two different shapes, and only the first is a per-gene-chunk walk.
Wilcoxon and pdex now share one `fill_gene_chunk_dense` over
`for_each_shard_ordered`; `pts` makes a single whole-matrix counting pass and
moved that one pass onto the same ordered driver. DE is where this matters most
because *its* shard walk is **inner** to its gene-chunk walk — every visited
shard is read once per gene chunk, so the cost is
`n_gene_chunks × visited shards`, where `pts` pays `visited shards` once. Two
separable effects, measured separately.

**The row-projection skip**, counted in shard decodes (`SCX_CPU_PROFILE=1`, a
120 × 200 file in 5 shards of 24, `cache_shards=0` so every read is a decode):

| request | before | after |
|---|--:|--:|
| `rank_genes_groups` / `pdex_ref`, one-shard row window | 5 | **1** |
| `rank_genes_groups(pts=True)` (two passes), one-shard window | 10 | **2** |
| `rank_genes_groups(gene_chunk_size=64)` → 4 chunks, one-shard window | 20 | **4** |
| `rank_genes_groups`, 4 chunks, two-shard mask | 20 | **8** |
| any of them, no row projection | 5 / 10 / 20 | unchanged |

**The decode-prefetch** is a wall-clock claim, and it is deliberately **not
published here**. It could be expressed as a `benchmark × format × dataset`
triple — `bench_csc_dispatch`'s `bench_csc__de_csr` / `bench_csc__pdex_ref_csr`
on `tabula_sapiens_100k` run CPU DE on a backed file, which *is* the streaming
kernel — and [docs/benchmark_manifest.md](../benchmark_manifest.md) is explicit
that a claim of that shape must be manifested, with the inline-disclosure tier
reserved for kernel measurements that genuinely cannot take it. The local
depth-1-vs-4 numbers are in the pull request that made the change; the durable
figure waits on a capture of those two arms, neither of which carries a floor in
`thresholds.yaml` today.

(For the record on the *shape* of the win, which the counts above already
establish: overlap only helps where decode is repeated, so it is largest with no
shard cache and smallest once the LRU holds the file — and flat on a one-shard
row window, which has nothing to prefetch ahead of.)

Peak memory is bounded rather than assumed: the pipeline holds `depth` decoded
shards where the loop held one, and the DE prefetch clamp (`scx-accel`, crate-internal)
grants only what the dense `n_obs × chunk` workspace left of
`SCX_ACCEL_DE_MEMORY_BUDGET`. A budget-bound file therefore resolves to depth 1
and keeps its pre-change footprint; on a plain backed handle the in-flight
`Arc`s are the same ones the LRU already holds, so there is no second copy at
all. The GPU CSR staging plan learned the same skip (`StagingPlan::for_source`).

## Low-risk marshalling & fusions (Phase-2 tasks 2.4 + 2.7)

These are **correctness-neutral** clean-ups — the 2.0 oracle rated marshalling negligible
and none of the 2.7 items were bottlenecks — so they carry no `×` speed claim; the point
is lower allocation counts and one shared thread-control knob, with no regression.

- **2.4 — flat NumPy marshalling.** The PCA / PFlog / UMAP result writers built a
  `Vec<Vec<f32>>` (one small allocation per obs row → ~N allocations at N cells) before
  `PyArray2::from_vec2`. They now assemble one flat row-major `Vec<f32>` and hand it to
  numpy via the shared `accel::util::flat_pyarray2` helper — `PyArray1::from_vec` takes
  ownership of the buffer (**no copy**) and `reshape([rows, cols])` returns a row-major view
  (no `unsafe`; a size mismatch fails loud as a Python error). Output is **byte-identical**.
  Measured with `SCX_CPU_PROFILE=1` on an in-memory `pca(n_comps=50)` over a 300 000 × 2 000
  matrix (`benchmarks/scripts/bench_marshalling.py`): the `marshalling` bucket is **~31 ms**
  for the 60 MB written (300k×50 `X_pca` + 2k×50 `PCs`) — well under 1 % of the ~7 s `pca()`
  wall (and about half the ~66 ms an allocate-then-`copy_from_slice` path took, since the
  zero-copy `from_vec` avoids the second buffer). The win is the allocation-count drop (one
  moved buffer vs ~300k per-row `Vec`s), not wall time; this confirms the oracle's ranking.
- **2.7a — `score_genes` weight fusion.** The control-set score fused the two weight
  vectors (`w_list`, `w_ctrl`) into a single `w = w_list − w_ctrl`, halving the per-nonzero
  inner iterations in `streaming_weighted_row_sums`. Numerically it matches the previous
  two-accumulator `Σw_list·v − Σw_ctrl·v` up to f64 re-association (single accumulator vs
  two subtracted once) — within the scanpy parity tolerance, unit-bounded to 1e-12.
- **2.7b/2.7c — allocation hygiene.** NB-GLM's no-shrink branch moves `mle` instead of
  cloning the whole per-gene state vector; the PCA transpose-spMM fold reuses one `q_row`
  scratch buffer per rayon task instead of allocating per row. Both **bit-identical**.
- **2.7d — `SCX_ACCEL_NUM_THREADS`.** A shared thread-ceiling policy knob (read once via
  `OnceLock`). When set to a positive integer it caps the accelerators' private rayon work:
  Harmony's integration pool and **both** PCA covariance-accumulator paths (the streaming
  `accumulate_covariance_streaming` pool and the in-memory
  `sparse_outer_product_accumulate_par` fold-segment count), whose memory-derived worker
  cap still applies — the env only lowers it further. Unset (the default) → behaviour is
  identical to before. *(Both named PCA functions are internal, and were removed after v0.13.0 along with the
  private pool and the memory-derived cap; the knob now bounds how many column blocks the
  PCA reductions split their output into. Still speed-and-memory only — more so than
  before, since the block count provably cannot change the numbers.)* It does **not** resize the ambient global rayon pool the many
  `current_num_threads()` callers use — that stays governed by `RAYON_NUM_THREADS`.

## Marshalling & GIL hygiene at the Python boundary (Phase-4 task 4.3)

2.4 converted the 2-D result writers; 4.3 converts the 1-D ones on the `uns` / `obsp`
result paths and fixes a concurrency defect next door. It is **not** exhaustive: the DE
DataFrame builders (`de.rs`'s `rank_genes_groups_df` path and three siblings) still push
`n_groups × n_genes` f64 columns through Python lists, some of them larger than sites that
were converted. They were left because they feed pandas/polars constructors rather than a
numpy array, so the conversion is a different shape — and, per the measurement below, not
one worth reaching for on performance grounds. Both halves are **output-neutral** and carry no `×` claim.

- **Flat buffers for every remaining `numpy.array(<Rust Vec>)`.** That spelling cloned
  the Rust buffer, had pyo3 build a Python `list` (one `PyLong`/`PyFloat` per element),
  then made numpy re-parse it. The two that mattered: `write_neighbors_to_adata` built
  **six** CSR arrays that way — `n_obs × k` each, so tens of millions of transient Python
  objects at 1M cells × k=15, on a path shared by CPU kNN, GPU kNN, `pca_neighbors` and
  `pca_neighbors_umap` — and the DE structured-array builder did it per group per field,
  so a 30-group × 60k-gene run materialised millions more. Both now use
  `PyArray1::from_vec` / `from_slice`; `write_neighbors_to_adata` takes its `KnnResult` by
  value so the buffers are *moved*, not copied. Smaller instances converted alongside in
  `pca.rs`, `harmony.rs`, `pseudobulk.rs`, `nb_glm.rs`, `de.rs` and `eval_metrics.rs`. The
  `rank_genes_groups` `names` field keeps its `PyList` path — its object dtype (scanpy's,
  so no name is truncated and no long name widens every cell) genuinely needs Python strings.

  **Dtype was the risk, not values.** `np.array(list[int])` is int64 whatever the Rust
  width, so the kNN `Vec<i32>` indices used to arrive as int64 and now arrive as int32 —
  absorbed, because `scipy.sparse.csr_matrix` re-derives its index dtype through
  `get_index_dtype(..., check_contents=True)`. One site was *not* absorbed:
  `eval_metrics.rs`'s group-reorder indices are `Vec<usize>`, which `from_vec` would land
  as **uint64** where the list round-trip gave int64, so the kernel collects `i64`
  explicitly. `pyscx/tests/test_marshalling_dtypes.py` asserts the observable dtype at
  every converted site rather than assuming the absorption.

- **`col_*` release the GIL.** `pyscx.accel.{col_sums,col_nnz,col_min,col_max,col_var}`
  ran a full-matrix streaming decode **holding the GIL** (`run_csr_f64` had a
  `let _ = py;` where the release belonged), so any one of them blocked every other Python
  thread for the duration and concurrent use from a dataloader or server thread pool
  serialised completely — while every sibling heavy entry point already released it. A
  `PyRef` cannot cross `py.detach(...)`, so each entry now snapshots what the scan needs
  into an owned handle (`Arc` clones + owned index vectors; no matrix data copied), runs
  the kernel detached, and re-acquires only to build the array. The CSC dispatchers needed
  an owned accessor for the same reason; `as_column_source()` is now that accessor on both
  handle kinds — it returns an owned view rather than a borrow, so the separate
  `as_column_source_owned` it once needed is gone.

  `pyscx/tests/test_col_aggs_gil.py` measures the property directly rather than as a
  throughput ratio, which would be flaky on a loaded host: a monitor thread stamps
  `perf_counter()` every ~1 ms during the scan and the assertion is on the largest gap.
  Against `2055f74f` all three ops starve the monitor for **100 % of the scan**; after the
  change the gap is a small fraction. A 4-thread concurrency test pins that overlapping
  `col_sums` calls on one reader — newly possible, since they now share its mmap and LRU
  concurrently — still agree exactly.

**Measured — and the headline prediction did not hold.** Finding §9.5 expected the kNN
path's "tens of millions of transient Python objects" to be a real cost at atlas scale. It
is a real *allocation* cost, but it does not show up in either metric that would justify
the change on performance grounds (SLURM jobs 2708544 and 2708786, `cpu` partition,
synthetic 500k/1M × 2 000 at density 0.05, k=15):

| | peak RSS | wall |
|---|--:|--:|
| `neighbors`, 500k, `2055f74f` | 9 212 MB | 219.8 s |
| `neighbors`, 500k, this change | 9 211 MB | 215.1 s |

**Peak RSS is unchanged (−1 MB, 0.01 %), and the 2 % wall difference is noise on a single
220 s run.** The transient list for a 14M-element `Vec<f64>` really is ~560 MB, but the
process high-water at this scale is set elsewhere (the HNSW build), and CPython's allocator
serves the churn from arenas it already holds — so `ru_maxrss` never sees it.

The bucket itself, on the new path: 533 MB of CSR arrays marshalled in **4.4 ms** at 1M
cells (0.00 % of a 754 s wall) — a memory-move rate, consistent with `from_vec` handing the
buffer to numpy rather than copying it. `pca` writes 200 MB in 106 ms (0.49 % of wall, and
a genuine copy via `from_slice`); DE writes 128 KB in 0.09 ms. The `main` arm reports a
zero bucket because the instrumentation is part of this change, so the two bucket numbers
are **not** a before/after — only the branch column is meaningful.

So this lands as **hygiene, not a speedup**, and is described that way deliberately: it
removes a full Rust-side buffer clone (533 MB at 1M cells) and tens of millions of
transient object allocations, it is byte-identical in output, and it is simpler. The 2.0
oracle ranked marshalling negligible and both 2.4 and this capture agree; §9.5's
"largest remaining marshalling cost" was right about the *ordering* among marshalling
sites and wrong that the absolute mattered.

## Owned snapshots at the numpy boundary (finding §10.1)

Seven `py.detach(...)` call paths in `pyscx` read Rust `&[T]` slices *borrowed* from live,
Python-reachable numpy buffers. rust-numpy borrows are not GIL-bound, do not clear numpy's
`WRITEABLE` flag, and carry no synchronization; `astype(..., copy=False)` and
`scipy.sparse.csr_matrix(A)` on an already-CSR `A` are identity operations, so those slices
*were* `adata.X.data` / `adata.X.indices`. Each now takes an owned snapshot under the GIL
first. Sites: `pseudobulk_means` (sparse and dense in-memory arms, and the lazy arm, which
now skips the scipy round-trip entirely by using the `ScxCsr` `to_memory_py` already builds),
`knockdown_efficiency`, `Experiment.gather_rows_sparse`, `from_anndata`'s CSC-sidecar
block, and — found in PR review — `decompose_scipy_csr_with`, which already copied
`indptr` / `indices` but handed `data` through as a borrow on its canonical fast path
while all three of its callers wrapped the callback in `py.detach`. The same change folded
three near-duplicate scipy→`ScxCsr` materializers into one, `convert::owned_csr`.

**Measured.** Single node, `cargo` release build, medians of 3–5 runs; sparse fixture
200k × 600 at density 0.08 (9.6M nnz), dense fixture 40k × 1 200 f32 (192 MB). The middle
column is the naive form of the fix — a serial `to_vec()` — and is shown because it is where
this nearly landed:

| op | before | serial copy | **parallel copy** | net |
|---|--:|--:|--:|---|
| `pca`, in-memory CSR, `n_comps=10` | 1 213 ms | 342 ms | **307 ms** | **4.0× faster** |
| `highly_variable_genes`, in-memory, `seurat_v3` | 1 090 ms | 208 ms | **189 ms** | **5.8× faster** |
| `pseudobulk_means`, in-memory sparse | 38 ms | 86 ms | **47 ms** | 1.2× slower |
| `pseudobulk_means`, in-memory dense f64 | 128 ms | 248 ms | **138 ms** | 1.08× slower |
| `pseudobulk_means`, in-memory dense f32 | 23 ms | 132 ms | **37 ms** | 1.6× slower |

The speedups are the fold, and they are the larger effect. `extract_materialized_csr` — on
the in-memory `pca` / `hvg` / `score_genes` / `pflog` / fused paths — reached numpy through
`extract::<Vec<i64>>()`, which is **not** a memcpy: pyo3's `Vec<T>` extraction fast-paths
only `u8` from bytes (`pyo3-0.28.3/src/conversions/std/vec.rs:74-84`) and otherwise falls to
`extract_sequence`, one Python object per element. Those paths also paid an `astype(copy=True)`
on top. One memcpy replaces both.

**The defensive copy is page-fault bound, not memcpy bound, and that is the whole story of
the third column.** A serial `to_vec()` of the 192 MB dense array cost 109 ms — matching the
119 ms `np.copy()` measures on the same array, i.e. ~1.6 GB/s, an order of magnitude below
the machine's memcpy rate. The destination is a fresh allocation, so the cost is first-touch
faulting one 4 KB page at a time, and that parallelizes: `convert::interop::par_to_vec` fills
from a rayon pool above a 4 MB threshold and takes the same copy to ~14 ms. Rayon is safe to
call with the GIL held here — the closure is pure Rust and never re-enters the interpreter.

What remains is small enough to leave alone. Dense f64 is within 8 % of baseline (its
`astype` was always the dominant copy); sparse is within 23 %; dense f32 pays the most at
1.6×, and is the one case where nothing was being copied before —
`astype`/`ascontiguousarray` are both no-ops on an already-f32 C-contiguous array, which is
precisely why the borrow aliased `adata.X` there. Passing a sparse `X`, or a backed / lazy
handle (which streams and never materializes), avoids the copy entirely;
`pseudobulk_means` emits a `UserWarning` naming both escape hatches once the copy exceeds
1 GB dense / 2 GB sparse.

Two follow-ons are recorded rather than taken, both now marginal: skipping the copy when the
coercion already produced a provably-unaliased array (`np.may_share_memory(coerced, x) is
False`), worth ~10 ms on the dense-f64 row; and a chunked dense accumulator in `scx-accel`,
which would bound the extra peak memory rather than the time. `from_anndata`'s CSC block
needed neither — the transpose already allocated all three owned buffers as its first act,
inside the detached region, so splitting it into `csc_input_from_csr_slices` (GIL held, copy
only) + `write_csc_shards_from_owned` (detached, sort + write) is byte-for-byte RSS-neutral.

`pyscx/tests/test_numpy_buffer_race.py` measures the property rather than the symptom: a
mutator thread flips `X` between two states for the duration of the op, and a correct
implementation must return exactly one of the two results, never a blend. "Mutate and assert
the answer is unchanged" is unsound in both directions — it passes on broken code by timing
luck, and *fails on correct code* when the mutator lands before the snapshot. All four tests
were observed red on 8 of 8 runs against unmodified source, and green on 8 of 8 after.

Two properties of the *mutator* turn out to be load-bearing, both found by watching a test
fail on correct code:

- **numpy releases the GIL inside a large array assignment.** Measured with a concurrent
  reader comparing the two ends of the buffer, a 9.6M-element `arr[:] = other` is observed
  torn in **3 076 100 of 11 357 806** checks (27 %); a 480-element strided write is torn
  **0 times in 3 347 462**. A torn mutation is a third state, so the two-state premise fails
  and a correct implementation looks broken. The mutation has to stay under numpy's
  threading threshold — and strided, so the touched elements are spread across the read
  order rather than sitting in a handful of adjacent rows.
- **The op has to consult enough independent values for a blend to be possible at all.**
  `knockdown_efficiency` reduces each targeted gene's control cells to one baseline scalar;
  with four target genes and roughly one spiked value per gene, each baseline lands wholly
  in one state and the result matches A or B by construction. That version passed on the
  unfixed build. Two hundred target genes make an all-one-way outcome vanishingly unlikely.

## Graph-layout refactors (Phase-2 task 2.5)

These are **result-preserving** layout/allocation refactors of the UMAP connectivity
builder and the Rust-native Leiden — validated by parity gates before any timing, so the
value is reduced allocation (fewer HashMaps) + parallelism headroom, not a large `×`.

- **UMAP connectivities** (`neighbors/cpu.rs::compute_connectivities`): the per-point
  bandwidth (σ) search now runs on `into_par_iter` (each point is independent and
  `find_sigma` is deterministic → order-preserving, **byte-identical**), and the
  fuzzy-simplicial-set symmetrization replaced its `HashMap<(i,j)>` + `HashSet` with a
  counting-scatter CSR transpose + sorted row-merge. Output is **byte-identical** to the
  prior path — the symmetrization is a fixed `μ(i,k)+μ(k,i)−μ(i,k)·μ(k,i)` per pair —
  proven by a new independent dense O(n²) reference test (`connectivities_match_dense_reference_*`).
- **Leiden** (`leiden.rs`): `aggregate` replaced its two per-collapse `HashMap`s with a
  stable-sort + merge-sum over dense group ids (stable sort preserves the graph-visitation
  accumulation order → byte-identical collapsed weights); per-node self-loop weights are
  precomputed once (was a binary search per move candidate); the redundant `node_strengths`
  copy and the dead `degrees` field were dropped. **Partition-identical** — guarded by
  `test_deterministic_with_seed`, the quality goldens, a new `aggregate` reference test, and
  the Python `test_ari_vs_scanpy_leiden_cpu` ARI floor.

**Measured** (`benchmarks/scripts/bench_graph_layouts.py`, 50 000 × 50 synthetic PCA, 8
blobs, CPU): kNN+connectivity end-to-end **flat within run-to-run noise** (~17.8–18.8 s;
the HNSW kNN build dominates, so the parallelized connectivity sub-phase does not move the
end-to-end wall), Leiden **~1.1×** (1186 ms → ~1060 ms), peak RSS slightly lower
(651 → 643 MB). The allocation/fragmentation win scales with graph size — the Leiden
`aggregate` HashMaps were the source of the heap fragmentation the outer-loop `malloc_trim`
was added to fight on 100 K+-node graphs.

## UMAP determinism + invariant hoist (Phase-2 task 2.6)

The CPU UMAP SGD stays **serial and deterministic** by design — a single seeded
`ChaCha8Rng` stream plus in-order edge iteration make two same-seed runs byte-identical.
Task 2.6 locks that in with a full-output determinism regression test
(`umap::tests::test_compute_umap_deterministic`) and hoists the loop-invariant `b - 1` out
of the per-edge gradient (`grad_coeff`). The hoist is **bit-identical by construction**
(constant subexpression elimination of a loop-invariant — no reassociation; `b` is bound
once from `find_ab_params`), guarded (not proven) by the determinism + cluster-separation
regressions, and **perf-neutral**: measured UMAP wall is
unchanged within noise (~19.07 s → ~19.04 s on 40 000 × 50, `n_epochs=200`), because the two
`powf` calls in `grad_coeff` dominate the single hoisted subtraction — so no `×` claim.

The **opt-in parallel (Hogwild) UMAP** mode from the audit is **deferred**: CPU UMAP SGD is
not a profiled bottleneck (it has a rapids GPU path), and a correct implementation needs a
per-thread deterministic RNG design plus a trustworthiness / kNN-overlap quality gate (none
exists yet). When revisited it should keep the serial path as the deterministic default and
mirror Leiden's opt-in `parallel` flag.
