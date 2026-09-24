# CPU accelerators: PCA and pseudobulk

> Part of [SCX performance](README.md). Headline CPU numbers are on
[the streaming pipeline page](accel-cpu-pipeline.md).

## PCA streaming decode-prefetch (Phase-4)

Task 4.2 wired the bounded ordered decode-prefetch pipeline into the backed aggregation
kernels, the column-projected twins, the lazy/transformed kernels and GPU staging — but
not PCA, which owns two inner rayon pools and was deferred on a deadlock concern. `pca`
was that capture's *control* for exactly that reason and came out flat at 1.02× while
everything around it moved ~2×, leaving it the single most expensive CPU op in the tier.
`scx-codec` contains no rayon, so a shard decode is genuinely single-threaded: more than
half of census_1m `pca` was one core decoding while the other 23 waited.

**The deferral rested on the wrong nesting.** What deadlocks `for_each_shard_ordered` is
being *called from* a saturated pool worker — the drain blocks on the channel while its own
decode spawns queue behind it. Entering a parallel region *from inside* `consume` is the
opposite, and is safe: outstanding decode tasks never exceed `depth` and the channel
capacity **is** `depth`, so a decode worker always completes its `tx.send` rather than
parking, and PCA's private covariance pool is a disjoint set of OS threads from the global
pool the decodes run on. *(Removed after v0.13.0: with one shared accumulator
there is nothing left to bound, and the covariance build now enters the global pool from
inside `consume` exactly as the embeddings pass already did.)* Streaming CPU PCA is only ever entered from `py.detach` on the
calling Python thread, never from a worker.

Six loops now prefetch: the fused column-means pass (`col_means_and_sum_sq_prefetched`,
added to `scx-format-io` beside the pipeline rather than to `scx-accel`, so that GPU PCA's
identical serial loop could adopt it without another move — which it since has, in
`scx_gpu::gpu_pca::randomized_pca_core`), the covariance build, the covariance embeddings pass, both streaming SpMM
passes, and the rare centered-variance re-stream. The covariance embeddings pass also
stopped hand-inlining `spmm_forward_into` — same accumulation, same order, same mean
correction, but serial, where the randomized route already called the shared parallel
kernel.

**Measured** (SLURM job 2709414, `cpu` partition, one host, 24 cores,
`RAYON_NUM_THREADS=24`, `--release`, **one build** with `SCX_ACCEL_PREFETCH_DEPTH=1` vs the
default 4, 3 runs, median wall):

| Dataset | CSR shards | `pca` (randomized) | `pca_hvg` (covariance) | `qc` ⁺ | `subset_obs` ⁻ |
|---|--:|--:|--:|--:|--:|
| pbmc10k | 1 | 1.31× ‡ | 0.98× | 1.01× | 1.06× |
| smartseq2 | 4 | 1.11× | **1.63×** | 1.57× | 1.01× |
| tabula_sapiens_100k | 7 | 1.20× | **1.80×** | 1.75× | 1.08× |
| census_500k | 31 | 1.27× | **2.55×** | 1.97× | 0.96× |
| census_1m | 62 | **2.29×** | **2.91×** | 2.13× | 1.01× |

⁺ positive control — wired for prefetch in 4.2, so the knob *must* move it; it re-measures
that result and confirms the knob binds. ⁻ negative control — decodes no shards at all, so
the knob must *not* move it; its 0.96–1.08× spread is the host's noise floor.
‡ **discount this one**: pbmc10k is a single-shard file, where `for_each_ordered` declines
to engage by construction, so 1.31× on a 2.5 s op is page-cache asymmetry between the
first and second arm, not prefetch.

**The win tracks how much the LRU misses, and the decode counts say so exactly.** They are
**identical between arms** at every point — 434/434 at census_1m `pca`, 124/124 for
`pca_hvg`, 31/31 at census_500k — so prefetch changes *when* decoding happens, never how
much. Read against the shard count they also explain the shape of the table:

- census_1m `pca`: 434 decodes over **62** shards = **exactly 7 passes**, which is
  randomized PCA's pass count (means + forward + 2 × (transpose + forward) + final
  transpose). The default 8 GiB shard LRU cannot hold the ~11 GB decoded working set, so
  it serves *nothing* and every pass re-decodes. Everything is available to overlap → 2.29×.
- census_500k `pca`: 31 decodes over 31 shards = **1 pass**. The LRU holds the whole matrix,
  six of the seven passes are cache hits, and there is almost nothing left to overlap →
  1.27×. Not a disappointing result; a different regime.
- census_1m `pca_hvg`: 124 / 62 = **2 passes**, covariance's exact count.

Σ decode ÷ wall crossing 100 % is the overlap signal (it sums concurrent workers):
census_1m `pca` goes 65 % → **170 %**, `pca_hvg` 51 % → **156 %**.

**The wall drops by more than the decode bucket accounts for, and the reason is worth
knowing.** At census_1m `pca` the sequential model closes exactly — decode 99.3 s +
reduction 29.7 s + ~24 s of QR/SVD = 153.5 s measured. `pca_hvg` does not: 28.7 + 2.0 leaves
25.6 s unexplained in the off arm but only ~10 s in the on arm. The missing term is
`ProjectedShardSource::read_shard_arc`, which calls `project_csr` **after**
`inner.read_shard_arc()` returns — outside `record_decode_since`, which wraps only the codec
in `reader/matrix.rs`. Projecting 61,497 → 2,000 columns over every shard's nonzeros is real
row-scale work, it is untimed, and because it sits inside the pipeline's read closure it is
**also** overlapped. So the `decode` bucket *understates* what prefetch moves off the
critical path for any projected or transformed source, and `pca_hvg` beats `pca` at 500k
and 1m despite doing 3.5× fewer decode passes.

**The one-build protocol's premise was confirmed, not argued** (SLURM job 2709415, two
worktree builds on one host, census_1m, 3 runs). Its whole validity rests on
`SCX_ACCEL_PREFETCH_DEPTH=1` being what `main` actually did — the claim #373 got wrong for
GPU staging, where the "off" arm had zero decode threads and `main` had one. Here `main`
and the depth-1 arm land within host noise, on **different nodes**:

| census_1m | `main` (two-build) | depth-1 arm (one-build) | agreement | two-build speedup | one-build speedup |
|---|--:|--:|--:|--:|--:|
| `pca` | 145.8 s | 153.5 s | 5.0 % | **2.27×** | 2.29× |
| `pca_hvg` | 56.8 s | 56.3 s | 1.0 % | **3.03×** | 2.91× |

**And it retires a claim.** The two-build arms differ by prefetch *and* the
`spmm_forward_into` swap; the one-build arms differ by prefetch alone. The gap between them
— 3.03× vs 2.91× — is the swap's entire contribution, and at ~4 % it is not separable from
the 1–5 % host spread. The reduction bucket says why: the covariance build **and** the
embeddings scatter together are 2.0 s of a 56 s op, so parallelising the scatter can save
about a second. The framing that motivated including it (that it was "the covariance
route's dominant cost") was wrong — it is dominated by the eigendecomposition and by the
untimed projection above. It is kept as hygiene, deleting ~20 lines of a duplicated kernel,
with **no `×` claimed** — the same disposition as 4.3's marshalling and 4.5's parallel
validation.

**Peak RSS is flat or lower** — −716 MB and −564 MB at census_1m, +9 to +53 MB elsewhere.
Prefetch keeps up to `depth` decoded shards alive, and `pca(memory_budget=…)` is documented
as the RAM ceiling for out-of-core PCA, so those are reserved *out of* that ceiling rather
than added on top, sized from the catalog-backed `shard_size_hint()` (no decode). The
reserve is `depth − 1` shards, not `depth`: the pre-prefetch loop already held one decoded
shard outside the cache, so worst-case live bytes go from `B + s` to
`(B − (depth−1)s) + depth·s = B + s`, memory-neutral by construction — and zero at
`depth = 1`, which is what keeps the off arm byte-for-byte the old behaviour. That the
decode counts did not move is the direct evidence the smaller LRU cost no hit rate.

**Equivalence splits, and the split was a finding — since fixed.** The streaming PCA
reductions used to fold into `ThreadLocal` accumulators whose row→thread assignment was
decided by work-stealing and whose merge order was `ThreadLocal::iter_mut()`. Five
consecutive `method="covariance"` runs on a cancelling-pair fixture produced five different
results; `method="randomized"` produced one. The diff showed both the accumulator
declaration and the merge loop unchanged, so it predated decode-prefetch — this capture
found it, it did not cause it.

Both reductions now partition their **output** instead of their input rows, so the schedule
cannot reach the result and exact f64 bit patterns are claimed everywhere on the CPU PCA
paths, not only where the pre-fix code happened to earn them. See
[docs/scanpy/accel-embedding-clustering.md § PCA reproducibility](../scanpy/accel-embedding-clustering.md#reproducibility). The rewrite was also
**2.0–2.5× faster** — see [PCA reduction partitioning](#pca-reduction-partitioning-covariance-route) below.

The fixture rule inverts between the two, which is worth stating because it reads as a
contradiction. A pure *reduction* needs a cancelling ±1e16 pair or a lost ordering shows up
in no bit at all and the assertion is vacuous — the trap 4.2 hit, where the obvious fixture
was blind in 0 of 119 permutations. A *row-scatter* pass needs a well-conditioned one,
because there a wiring bug misplaces whole rows rather than perturbing a sum, and a
cancelling fixture would only drown that signal.

Because "the pipeline silently declined to engage" is invisible in every result, three
tests measure it rather than infer it: `GaugedSource` counts concurrent decodes and demands
more than one for the covariance build, for both SpMM passes and for the two whole ops,
while `depth_one_never_overlaps` pins the other side.

## PCA reduction partitioning (covariance route)

Making the CPU PCA reductions deterministic made them faster, which was not the point but is
the larger effect. The covariance build used to give each worker a private `n_vars × n_vars`
accumulator and merge them at the end; it now gives each worker a disjoint *column range* of
one shared accumulator. Two things follow: the per-worker merge disappears, and each worker's
write set drops from the whole matrix (32 MB at 2K vars) to its own slice (~2 MB), which fits
cache.

`cargo bench -p scx-accel --bench covariance_pca`, 12 cores, 8 shards × 25 K rows,
before/after on the same host in one sitting (Criterion medians):

| Fixture | Before | After | Speedup |
|---|---|---|---|
| 2 000 vars, 5 % dense | 2.589 s | **1.057 s** | **2.45×** |
| 5 000 vars, 3 % dense | 14.769 s | **7.230 s** | **2.04×** |

> [!NOTE]
> These are **Criterion microbenchmark** medians from the in-repo bench, not a captured
> entry under `benchmarks/comprehensive/results/`. `docs/benchmark_manifest.md` asks for a
> manifest behind every number here; the manifest system is shaped for the SLURM
> comprehensive suite and has no covariance-PCA microbenchmark triple, so the reproduction
> recipe above stands in for one. `benchmarks/scripts/check_readme_manifests.py` does not
> flag these claims.

Peak memory falls with it: one accumulator rather than one per worker, which is why
`SCX_PCA_COV_MEMORY_BUDGET` no longer has anything to cap. The transpose SpMM got the same
treatment, dropping `n_vars × k` per worker to one shared buffer.

The remaining thread-count sensitivity is faer's, not SCX's: its dense QR and
eigendecomposition block by the ambient rayon width. They are stable run to run at a fixed
width — the contract numpy/scipy give — and `SCX_ACCEL_DETERMINISTIC_LINALG=1` pins them for
callers who need identity across thread counts, measured at ~2.3× slower on the covariance
route's eigendecomposition (n_vars = 2000) and ~1.65× *faster* on the randomized route's thin
QR (200 K × 60).

## Pseudobulk aggregation partitioning (OPT-ACCEL-4)

The pseudobulk scatter — `counts[g · n_vars + col] += v` once per nonzero, behind
`pyscx.accel.pseudobulk_means`, `perturbation_metrics` / cell-eval, `pseudobulk_dex` (whose
default backend since v0.13 is the native `nb_glm`) and rscx's pseudobulk entry points — ran serially on the
calling thread at all four of its sites (the streaming shard loop, the two in-memory paths and
the CSC projected path) while the decode-prefetch pool idled. It now partitions its **output**
the way the PCA reductions do, and merges nothing, so every `(group, gene)` sum is formed from
the same f32 operands in the same ascending-cell order as before: **bit-identical** to the serial
loop on any thread count, which the crate's tests pin against a float fixture whose sums are
order-sensitive (integer counts sum exactly in any order and would hide a reordering). Two
partitions, chosen per shard on the streaming path and once for an in-memory matrix: one
group row per task while the largest group holds at most two pool-shares of the nonzeros (it
reads every nonzero once and needs no sorted columns, so it is also where an unsorted scipy
CSR goes — still parallel, where PCA falls back to serial); otherwise a column block of every
group's row, with each row's block windows planned once in a parallel pass rather than
binary-searched per block, and only when a block owns at least 64 nonzeros of the average row
— on 100-nonzero rows every partition reads more cache lines than the memory-bound serial loop
streams, so a two-group HVG-subset matrix keeps the serial walk and pays nothing.
`SCX_ACCEL_NUM_THREADS` caps the column-block count as it caps PCA's; the group partition is
one task per group row on the ambient pool (a fixed number of coarse tasks measured 40 % slower
on the streaming arms below). The CSC projected path parallelises only a contiguous run of
requested columns wider than one, into a run-local scratch — a scattered gene subset keeps the
serial loop.

`cargo bench -p scx-accel --bench pseudobulk` on the Chimera worker `GPUCACE`, 12 cores,
2026-09-11, both arms in one build at commit `a4d09aa7` of the PR-16 branch (the `before` arm is
the replaced serial loop, replicated inline; both arms include the identical group-mapping step,
so the scatter-only ratio is higher than shown). `Mean`, Criterion medians:

| Fixture | Groups | In-memory before → after | Streaming before → after |
|---|--:|---|---|
| 200k × 2 000 genes, 5 % (100 nnz/row) | 2 | 26.4 → 26.7 ms (0.99×, serial walk kept) | 43.0 → 32.2 ms (1.34×) |
| | 64 | 32.5 → 17.5 ms (**1.86×**) | 51.9 → 19.9 ms (**2.61×**) |
| | 2 048 | 86.0 → 19.5 ms (**4.42×**) | 104.9 → 29.2 ms (**3.60×**) |
| 20k × 20 000 genes, 10 % (2 000 nnz/row) | 2 | 43.9 → 25.9 ms (**1.70×**, column blocks) | 76.6 → 50.1 ms (**1.53×**) |
| | 64 | 114.6 → 11.8 ms (**9.75×**) | 145.7 → 36.5 ms (**3.99×**) |

The gains scale with how badly the serial loop was missing cache: with two group rows resident
in L1 it was already memory-bound near the node's bandwidth, and no partition can read the
input faster than that; with 64 or more groups the dependent adds were the cost, and those
divide across the pool. The streaming arms carry the per-shard decode (a clone here) on both
sides and the ordered consumer keeps the pool partly idle during it, which is why they trail the
in-memory arms at the high end.

> [!NOTE]
> These are **Criterion microbenchmark** medians from the in-repo bench, not a captured entry
> under `benchmarks/comprehensive/results/`. `docs/benchmark_manifest.md` asks for a manifest
> behind every number here; the manifest system is shaped for the SLURM comprehensive suite and
> has no pseudobulk-aggregation triple, so the reproduction recipe above stands in for one.
> `benchmarks/scripts/check_readme_manifests.py` does not flag these claims.

The comprehensive suite's pseudobulk-bearing cells — `bench_csc_dispatch` ×
`bench_csc__pseudobulk_{csr,csc}` on tabula_sapiens_100k, which time `pseudobulk_dex` end to
end with the pydeseq2 fit dominating — are the **no-regression** check for this change, not its
measurement. One job per arm (`benchmarks/scripts/_run_pr16_pseudobulk_gate.sh`; the job logs the
extension's sha256 and any dirty tracked paths), medians of three; the harness's `cpu_preemptible`
cells land on whichever node is free, and the node class moves `pseudobulk_dex`'s wall more than
this change does, so the node is part of the row:

| Arm (job) | Cell node | `pseudobulk_csr` wall / peak RSS | `pseudobulk_csc` wall / peak RSS | `csc_dispatch_correct` |
|---|---|---|---|---|
| `main` `9e628f38` (2932673) | CPUDFDC84 | 13.70 s (runs 17.1 / 13.5 / 13.7) / 3 579 MB | 0.74 s / 3 414 MB | 1.0 |
| branch, round-1 kernel, dirty tree at `9e628f38` (2932638) | CPUDFDE34 | 13.51 s / 3 553 MB | 0.78 s / 3 433 MB | 1.0 |
| branch, fix commit `9664b85e`, tracked tree clean (2933595) | GPU1298 / GPU726E | 15.43 s / 3 601 MB | 0.79 s / 3 482 MB | 1.0 |

The two CPU-node captures are the like-for-like pair: wall within 1.5 % and RSS within 30 MB of
`main`. The clean-commit recapture is the provenance-correct row and sits on a different node
class, where the serial pydeseq2 fit runs slower; its RSS is within 70 MB of `main`. Manifest rows:
the clean recapture at `benchmarks/comprehensive/results/raw/bench_csc_dispatch__bench_csc__pseudobulk_{csr,csc}__tabula_sapiens_100k.json`,
the `main` arm under `results/raw/pr16_pseudobulk_base/`, the round-1 capture under
`results/raw/pr16_pseudobulk_r1_dirty_tree/`. Every row carries `git_dirty: true`: the harness
flags any `git status --porcelain` output, and this checkout keeps four untracked scratch notes
at the repo root (not named here — tracked files do not cite them — and read by neither the
harness nor the build); the job log records every porcelain line, the tracked-tree state and the
extension's sha256. The rows themselves carry neither, which would take a harness change.
Against `LATEST` (captured 2026-09-03 at `33d52cd0`) the gate reports both cells ~550 MB higher in
peak RSS; the `main` arm shows the same figure, so that delta belongs to the merges between the two
snapshots, not to this change.
