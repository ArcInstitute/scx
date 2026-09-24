# CPU accelerators: QC, DE, scoring, and integration

> Part of [SCX performance](README.md). Headline CPU numbers are on
[the streaming pipeline page](accel-cpu-pipeline.md).

## QC / filtering pass fusion (Phase-4 task 4.1)

`calculate_qc_metrics` used to decode every shard once **per statistic**: per-cell
sums, per-cell nnz, per-gene sums, per-gene nnz, and one further full scan for each
`qc_var`. The standard analyst call — `qc_vars=["mt", "ribo", "hb"]` — therefore read
the whole matrix **seven** times. Every one of those quantities is an additive
accumulator over the same nonzeros, so they collapse into one row-axis pass (sums +
nnz + all subset sums, dispatched through a per-visible-column `u64` bitmask) and one
column-axis pass. `filter_genes` with both a cell and a count threshold likewise drops
from two column scans to one, and the CSC gene axis now walks the sidecar once.

Measured on one `cpu`-partition host, `--release`, both arms built from isolated git
worktrees of the two commits (`25479830` = projection fix only / unfused, `7d278365` =
fused), SLURM job 2706142, 3 runs each, median wall / max peak RSS
(`benchmarks/scripts/profile_cpu_stages_backed.py --ops qc filter_genes`, `SCX_CPU_PROFILE=1`):

| Dataset | op | passes | wall before | wall after | speedup | peak RSS before → after |
|---|---|--:|--:|--:|--:|--:|
| pbmc10k (12K) | `qc` | 7 → 2 | 2.34 s | 0.83 s | **2.80×** | 553 → 549 MB |
| smartseq2 (18K) | `qc` | 7 → 2 | 13.55 s | 4.14 s | **3.27×** | 900 → 885 MB |
| tabula_sapiens_100k | `qc` | 7 → 2 | 17.15 s | 5.11 s | **3.36×** | 904 → 899 MB |
| census_500k | `qc` | 7 → 2 | 65.76 s | 19.44 s | **3.38×** | 1508 → 1481 MB |
| census_1m | `qc` | 7 → 2 | 120.01 s | 35.89 s | **3.34×** | 3566 → 3481 MB |
| pbmc10k | `filter_genes` | 2 → 1 | 0.61 s | 0.43 s | 1.40× | 553 → 551 MB |
| smartseq2 | `filter_genes` | 2 → 1 | 3.59 s | 2.08 s | 1.73× | 904 → 899 MB |
| tabula_sapiens_100k | `filter_genes` | 2 → 1 | 4.56 s | 2.59 s | 1.76× | 904 → 899 MB |
| census_500k | `filter_genes` | 2 → 1 | 17.48 s | 10.02 s | 1.74× | 1560 → 1542 MB |
| census_1m | `filter_genes` | 2 → 1 | 31.99 s | 18.29 s | 1.75× | 3578 → 3488 MB |

The speedups sit just under the pass ratios (3.5× and 2×) because the decode bucket is
78–88 % of wall, not 100 % — the per-nonzero accumulation and the fixed AnnData/pandas
handoff don't shrink. They converge on the ratio as the matrix grows (2.80× at 12K cells
→ 3.34× at 1M), which is the signature of a fixed cost being amortized rather than a
scale-dependent win. Peak RSS is marginally **lower** in every cell: the fused kernels
allocate a handful of extra accumulator vectors (`n_qc × n_obs` f64 ≈ 24 MB at 1M cells ×
3 subsets) but churn far fewer transient decode buffers.

Both figures are for **3** `qc_vars`. The row pass carries up to 64 subsets in one
scan; past that it repeats once per additional 64, so the pass count is
`ceil(n_qc / 64) + 1`. Peak memory scales with the subsets held at once — the
accumulator is `n_qc x n_obs` f64 (~24 MB at 3 subsets x 1M cells, ~1.5 GB at the
64-subset ceiling x 3M cells), sized by the **physical** row count, and the
deletion-filtering step transiently doubles each row vector. The "RSS marginally
lower" result above characterises the ordinary handful-of-subsets call, not the
ceiling.

Output is unchanged. Each fused kernel keeps the existing left-to-right f64 accumulation
over an ascending-column walk, so its sums are bit-identical to the per-statistic kernels
it replaces; `pyscx/tests/test_qc_metrics_fused.py` asserts exact equality against the
array-protocol dunders (which still drive the unfused path) across backed / lazy ×
projection / none × deletions / none × 0, 1, 3 `qc_vars`, and a
`cpu_profile_snapshot()` decode-count test pins the pass count so a later refactor cannot
silently re-split it.

**Task 4.0a (axis-subsetting correctness) does not change these numbers.** It adds
per-call bookkeeping over the aligned members — no matrix work and no I/O — and the
decode-pass counts above are unchanged, which `test_qc_metrics_fused.py`'s
`cpu_profile_snapshot()` test pins. Lazy `obsp` / `varp` / `varm` entries are not decoded
by a subset at all; the subset is recorded and applied if and when the key is read.

## Fused QC + filtering vs scanpy (`accel_qc_filter`)

The section above measures the fusion against SCX's own unfused predecessor.
This one measures the fused kernel against **scanpy**, on the call sequence
that opens essentially every workflow — `calculate_qc_metrics(qc_vars=["mt",
"ribo"])` → `filter_cells(min_genes=200)` → `filter_genes(min_cells=3)`.
The two are different comparands and give different numbers (3.36× there,
2.3–4.1× here); neither contradicts the other.

`percent_top=(50, 100, 200, 500)` is passed **explicitly on both sides**,
clamped to the gene axis. pyscx defaults it to `None` and scanpy to that
tuple, so leaving the defaults alone would have the scanpy arm computing
four extra order statistics the SCX arm skips.

| Dataset | pyscx backed | pyscx in-memory | scanpy | speedup | RSS ratio |
|---|---:|---:|---:|---:|---:|
| pbmc3k (2.7K) | 0.13 s / 341 MB | 1.09 s / 514 MB | 1.21 s / 550 MB | 9.6× | 1.6× |
| pbmc10k (11.5K) | 1.48 s / 554 MB | 2.23 s / 1,045 MB | 2.55 s / 1,082 MB | 1.7× | 2.0× |
| smartseq2 (50K) | 4.00 s / 1,510 MB | 12.88 s / 3,521 MB | 14.47 s / 3,553 MB | 3.6× | 2.4× |
| tabula_sapiens_100k | 3.82 s / 1,525 MB | 16.44 s / 5,019 MB | 15.52 s / 5,047 MB | 4.1× | 3.3× |
| census_500k | 13.54 s / 1,871 MB | 24.50 s / 17,916 MB | 31.04 s / 17,959 MB | 2.3× | 9.6× |
| census_1m | **23.16 s / 2,196 MB** | 44.54 s / 33,311 MB | **52.47 s / 33,366 MB** | **2.3×** | **15.2×** |

Speedup and RSS ratio are the backed arm against scanpy. The memory ratio is
the one that grows: at 1M cells scanpy holds the whole CSR at 33 GB while the
backed arm streams in 2.2 GB, and that gap is what decides whether the work
runs on the machine in front of you.

Only **pyscx backed** is native end to end. `accel.filter_cells` /
`accel.filter_genes` delegate to `sc.pp.*` when `X` is an in-memory scipy
matrix, so the in-memory column is a native QC pass followed by a scanpy
filter — read it as the cost of the QC kernel alone, not as a
native-vs-scanpy result.

> [!NOTE]
> **On `smartseq2`, SCX and scanpy disagree — and SCX is right.** Its
> `qc_metrics_max_abs_diff` is 9.0 where every other fixture measures
> exactly 0.0. scanpy reduces `X` in the array's own dtype and every fixture
> stores float32; smartseq2's per-cell totals are 2–3 × 10⁷, past float32's
> 2²⁴ exact-integer range, where consecutive integers are no longer
> representable. 37 of 50,000 cells differ, and against the exact `int64`
> sum of the raw data SCX matches every time (22,989,693 / 20,484,705 /
> 34,427,105) while scanpy is one out. SCX accumulates in f64. The benchmark
> therefore judges each column against four ULPs of its own magnitude
> (`accel_qc_filter.float32_sum_tolerance`) rather than one flat tolerance,
> and `thresholds.yaml` carries no parity floor for that fixture.

Source: `benchmarks/comprehensive/results/raw/accel_qc_filter__accel_qc_filter__{pyscx_cpu,pyscx_inmem,scanpy_cpu}__{pbmc3k,pbmc10k,smartseq2,tabula_sapiens_100k,census_500k,census_1m}.json`,
capture `candidate_community_20260920` (`--tier full`, pyscx 0.18.0 release,
median of 3 runs at ≥100K cells and 5 below it). Each arm runs in its own
process and samples its own peak RSS; parity is computed against an untimed
scanpy reference in a fourth process.
 Manifest provenance for every community-benchmark number is recorded
once under [The laptop test](memory.md#the-laptop-test-a-full-pipeline-under-a-fixed-ceiling).

## Axis subsetting — what delegating to anndata costs (Phase-4 task 4.0b)

Task 4.0b stopped reimplementing anndata's axis bookkeeping: `filter_cells`,
`filter_genes`, `subset_obs`, `subset_var` and HVG `subset=True` now let **anndata**
perform the subset, so `obs` / `var` / `uns` / `raw` / unused categorical levels /
every aligned member behave exactly as they do in scanpy, and the operation is atomic.
The cost is that anndata builds a whole replacement `AnnData` and swaps it in
(`_mutated_copy` deep-copies `uns` and copies `obs` / `var` / every non-handle aligned
member), so for a moment both the old and new frames are live. This capture answers the
two questions that raises: does the matrix still stay on disk, and what does the rebuild
cost at census scale?

Measured on one `cpu`-partition host (24 cores, `RAYON_NUM_THREADS=24`), `--release`,
both arms built from isolated git worktrees of the two commits (`59783883` = 4.0a /
hand-rolled, `adfe70fc` = 4.0b / delegated), SLURM job 2707573, 3 runs each, median wall
/ max peak RSS
(`benchmarks/scripts/_run_4_0b_axis_subset_rss.sh`, which drives
`profile_cpu_stages_backed.py --ops filter_cells filter_genes_real subset_obs`,
`SCX_CPU_PROFILE=1`). Both arms report `git_dirty` — the current profile script is copied
into both worktrees so the harness is identical, since the ops did not exist at the
"before" commit; that is a benchmark-only change, per
[docs/benchmark_manifest.md](../benchmark_manifest.md).

`subset_obs` is the isolated probe: a fixed 50 % mask, no threshold scan, **0 shard
decodes**, so its wall and RSS are open + rebuild and nothing else. The two filters are
the realistic ops, where a full streaming scan dominates.

| Dataset | op | kept obs/vars | wall before | wall after | Δ | peak RSS before → after |
|---|---|:-:|--:|--:|--:|--:|
| pbmc10k (12K) | `subset_obs` | 0.50/1.00 | 18 ms | 24 ms | +30 % | 692 → 696 MB |
| smartseq2 (18K) | `subset_obs` | 0.50/1.00 | 112 ms | 156 ms | +39 % | 1051 → 1070 MB |
| tabula_sapiens_100k | `subset_obs` | 0.50/1.00 | 160 ms | 212 ms | +32 % | 1061 → 1083 MB |
| census_500k | `subset_obs` | 0.50/1.00 | 0.97 s | 1.23 s | +27 % | 1878 → 1993 MB |
| census_1m | `subset_obs` | 0.50/1.00 | 1.69 s | 2.19 s | +30 % | 3938 → 4088 MB |
| pbmc10k | `filter_cells` | 0.98/1.00 | 371 ms | 382 ms | +3.0 % | 692 → 690 MB |
| smartseq2 | `filter_cells` | 1.00/1.00 | 1.81 s | 1.94 s | +7.3 % | 1035 → 1042 MB |
| tabula_sapiens_100k | `filter_cells` | 1.00/1.00 | 2.28 s | 2.27 s | −0.6 % | 1061 → 1083 MB |
| census_500k | `filter_cells` | 0.98/1.00 | 8.64 s | 9.28 s | +7.5 % | 1715 → 1752 MB |
| census_1m | `filter_cells` | 0.99/1.00 | 16.63 s | 17.36 s | +4.4 % | 3829 → 3899 MB |
| pbmc10k | `filter_genes` | 1.00/0.61 | 391 ms | 406 ms | +3.9 % | 692 → 696 MB |
| smartseq2 | `filter_genes` | 1.00/0.95 | 1.93 s | 2.04 s | +6.0 % | 1051 → 1070 MB |
| tabula_sapiens_100k | `filter_genes` | 1.00/0.41 | 2.41 s | 2.54 s | +5.4 % | 1061 → 1083 MB |
| census_500k | `filter_genes` | 1.00/0.59 | 9.32 s | 9.79 s | +5.1 % | 1847 → 1915 MB |
| census_1m | `filter_genes` | 1.00/0.64 | 17.46 s | 17.90 s | +2.5 % | 3935 → 4046 MB |

`filter_genes` here is `min_cells=3` (scanpy's canonical default), not the permissive
`min_cells=1, min_counts=1.0` of the 4.1 table above — that one exists to time the *scan*
and its cut is incidental. The `kept` column is reported for every row precisely because
a subset that keeps everything is now correctly skipped, and would otherwise read as a
speedup: `filter_cells(min_genes=200)` is a no-op on smartseq2 and tabula_sapiens_100k,
and those two rows measure only the threshold scan.

**The matrix never materialises.** census_1m is 1,000,000 × 61,497 with 1.40 G nonzeros;
a materialised scipy CSR (f32 data + i32 indices, 8 B/nnz) is **~10.4 GiB**. The process
high-water across the whole four-op sequence is **4.0 GiB** — and that figure is
dominated by the obs frame and the decode buffers, not by `X`, which stays a projection
update on a handle. `test_in_place_ops_keep_x_lazy` pins the property itself
(`type(adata.X)` is unchanged by every in-place op), and
`test_in_place_ops_do_not_decode_the_lazy_bridges` pins that a subset never pulls
`varm` / `obsp` / `varp` off disk.

**What the rebuild costs.** ~0.5 s and ~150 MB at 1M cells, which is +30 % on the
isolated probe and **+2.5 % to +7.5 % on the realistic ops**, where the streaming scan —
77–91 % of wall in the decode bucket — dominates. The RSS delta is the transient second copy of `obs` and the
aligned members; it does not scale with nnz. Peak RSS here is a process high-water mark
(`ru_maxrss`) over an identical op sequence in both arms, so the per-row deltas are
comparable but the absolute values accumulate across the ops above them in the table.

That regression buys correctness that the hand-rolled path could not reach: `adata.raw`
follows an obs subset, unused categorical levels drop, a failure part-way leaves the
object untouched, and members nobody enumerated are handled by construction. It also
removes a silent pessimisation in the other direction — a filter that keeps every element
no longer installs an identity deletion vector, which used to close the CSC capability
gate and permanently downgrade the `gpu_csc_v3` CSC-direct DE route on a file that was
never really filtered.

## Differential expression (CPU, full-matrix)

The Wilcoxon rank-sum DE row above is from an HVG-projected (2K genes) 1M-cell fixture. The dedicated `accel_de` benchmark sweeps the raw count matrix (no HVG projection) across the full dataset tier — scanpy's per-gene rank pass becomes the bottleneck and times out on census-scale:

| Dataset | scanpy `rank_genes_groups` | `pyscx.accel.rank_genes_groups` (CPU) | Speedup |
|---------|---:|---:|---:|
| pbmc3k (2.7K) | 1.16 s | 0.69 s | 1.7× |
| pbmc10k (12K) | 9.66 s | 5.85 s | 1.7× |
| smartseq2 (18K) | 45.96 s | 19.59 s | **2.3×** |
| tabula_sapiens_100k (62K) | 318.77 s | 42.28 s | **7.5×** |
| census_500k | timeout (≥ 55 min) | 106.13 s | **≥ 31×** |
| census_1m | timeout (≥ 55 min) | 156.51 s | **≥ 21×** |

`pyscx.accel.pdex_ref` (perturbation-screen Mann–Whitney U + pseudobulk geometric-mean log fold change, pinned bit-for-bit to upstream [`pdex`](https://github.com/ArcInstitute/pdex)) tracks similarly: 0.59 s on pbmc3k, 5.52 s on pbmc10k, 42.04 s on tabula_sapiens_100k, 119.86 s on census_500k, 268.28 s on census_1m. The CPU path uses gene-chunked dense materialisation (default `gene_chunk_size=500`) with rayon-parallel per-gene rank tests — peak RSS is `O(n_obs × gene_chunk_size)`, not `O(n_obs × n_vars)`. CPU numbers improved 20-40% vs the prior `v0.4.3-g1-gpu-de` baseline after the `pdex-unsorted-csr` fix (commit b423a2f).

**CPU DE routing (Phase-2 §5.2/§5.3).** `rank_genes_groups` and `pdex_ref` now
default to `prefer_format="auto"`: on CPU they take the CSC-direct kernel when the
file has a CSC sidecar **and** the handle's row window still spans at least half the
CSR shards, else the CSR streamer. Neither a lazy transform chain, nor a row filter,
nor a column projection is a *disqualifier* any more — the
row-indexed `normalize_total` / row-scale transforms are served column-major by
looking up their per-row factor at the global row CSC `indices` carries
(bit-identically), and a filtered or gene-subset handle has its slab's rows
renumbered onto the live row space and its columns remapped into the projected
one on the way out of the reader. What a window does still decide is the *policy*:
a CSC column shard spans the whole row axis, so a slice confined to a few shards
would read every physical cell of the columns it asks for where the CSR path skips
the shards the slice empties — hence the half-of-the-shards cut, which an explicit
`prefer_format="csc"` bypasses. On GPU they stay CSR so the planner can route
`gpu_csc_v3`, which a windowed or transformed handle now also reaches under the same
condition. A route declined by that policy records `csc_available=true` with
`fallback_reason=perf_policy`, not `no_csc_sidecar` — the sidecar was there.
The route + `csc_available` flag are recorded on `adata.uns["scx_accel"][<op>]`.
This is a compatibility change (DE previously defaulted to `"csr"`); pin
`prefer_format="csr"` for the old behaviour. An **exact sparse-nnz Wilcoxon**
kernel (the default for 1-vs-rest since 0.20; `SCX_ACCEL_WILCOXON_NNZ=0` falls
back to densify) ranks only each gene's
nonzeros plus an analytic implicit-zero tie-block — `O(nnz·log nnz)`/gene instead
of an `O(n_obs·log n_obs)` dense sort — bit-identical to the densify kernel:
scores, p-values, adjusted p-values, log fold changes and gene order are pinned at
zero tolerance on a tie-heavy count matrix
(`nnz_matches_the_densify_kernel_exactly_on_tie_heavy_counts`), besides the
1e-9 property test. It records as its **own** route, `cpu_csc_nnz`; until
0.20 both CSC kernels stamped `cpu_csc`, so a benchmark could not tell from
`uns["scx_accel"]` which of the two it had timed (review §7.17).

**Measured DE-route benchmark** (`rank_genes_groups`, 1-vs-rest, backed streaming;
a CSC sidecar built with `scx build-csc`; median of 2 runs, CPU). Routes confirmed
via `uns["scx_accel"]` (`cpu_csr` / `cpu_csc` / `cpu_csc_nnz`). Every arm opens its
handle with `to_anndata(backed=True)`'s **default 4-shard cache**, which is fewer
than the 7 CSR shards the pass visits — so the CSR arm re-decodes every shard for
every gene chunk, and most of the ratio below is that re-decode rather than the
column-major layout as such. With the cache sized to the whole file (what
`bench_csc_dispatch`'s `de_csr` / `de_csc` arms do) the same comparison is 1.95×
here and 1.45× at census_1m, at the cost of holding every decoded shard (83 GB of
RSS for census_1m's CSR arm):

| Dataset | Route | wall (s) | peak RSS (MB) | vs CSR | vs CSC-densify |
|---|---|--:|--:|--:|--:|
| tabula_100k (100K × 61.5K, 33 groups) | CSR streaming (`csr`) | 351.8 | 3454 | 1.0× | — |
| tabula_100k | CSC-direct densify (`csc`, §5.2) | 25.6 | 2749 | **13.7×** | 1.0× |
| tabula_100k | CSC-direct nnz (`csc`, the default since 0.20; §5.3) | 16.2 | 2552 | **21.7×** | **1.58×** |

**§5.2 — at the default shard cache, CSR → CSC-direct is ~13.7× faster with lower
peak RSS.** The CSR streamer re-decodes every shard for each gene-chunk (here ~123
chunks × 7 shards on the full 61.5K-gene matrix, cache-bound), while the CSC-direct
route reads each column-chunk exactly once — so on a sidecar file, in the bounded
cache a backed handle opens with, the `auto` default's CPU routing is a large,
measured win. Against a CSR route allowed to cache the whole file it is the
1.95× / 1.45× above. (For CSR-*only* files the deferred loop-inversion / a
larger shard cache addresses the same re-decode; `auto` sidesteps it when a sidecar
exists.) **§5.3 — the exact sparse-nnz kernel adds ~1.58×** over CSC-densify by
ranking only nonzeros + an analytic zero block instead of an `n_obs` dense sort,
for **~21.7× end-to-end** over the old CSR default at the same default cache, at
lower peak RSS. Bit-identical to the dense kernel (pinned at zero tolerance, above). Source: `bench_de_csc_routes.py`
on `cpu_preemptible`, when the nnz kernel was still opt-in (`SCX_ACCEL_WILCOXON_NNZ=1`).
It became the 1-vs-rest default in 0.20; `bench_csc_dispatch`'s `de_csc` arm now
times it, and `de_csc_densify` (`SCX_ACCEL_WILCOXON_NNZ=0`) keeps the densify
kernel measured as the control.

Source: 2026-05-25 full-tier gate (post-G10 graph capture + bench env-routing fix), candidate `candidate_2623788_20260525`. Benchmark module: `benchmarks/comprehensive/benchmarks/accel_de.py` — picks the best obs column from `cell_type`/`leiden`/`louvain`/`cluster`/`perturbation`/`target` or falls back to a deterministic 50/50 synthetic split, restricts to top-4 test groups + reference, and records the chosen `groupby` in `metadata`. Each SLURM bench job is allocated 16 CPUs; `pyscx_cpu`'s `user_s/wall_s` ratio shows ~3-5 effective cores per run.

## Gene-set scoring vs scanpy (`accel_score_genes`)

`pyscx.accel.score_genes` scores a per-cell signature by streaming the
matrix shard by shard, so it runs on a **backed** `X` — which
`sc.tl.score_genes` refuses outright with `NotImplementedError`. The
benchmark scores three panel sizes (K = 25 / 100 / 500), derived
deterministically per fixture by ranking `var_names` on total counts,
keeping the top 2,000 and taking each K by a **stride** through that pool
(a head would land every gene of a K=500 panel in one bin of the
expression-matched control sampler).

| Dataset | pyscx (backed, control) | scanpy (eager) | speedup | RSS ratio |
|---|---:|---:|---:|---:|
| pbmc3k (2.7K) | 0.11 s / 323 MB | 0.13 s / 484 MB | 1.2× | 1.5× |
| pbmc10k (11.5K) | 1.28 s / 327 MB | 0.94 s / 848 MB | **0.7×** | 2.6× |
| smartseq2 (50K) | 3.05 s / 453 MB | 4.96 s / 2,602 MB | 1.6× | 5.7× |
| tabula_sapiens_100k | 3.21 s / 487 MB | 22.09 s / 3,666 MB | 6.9× | 7.5× |
| census_500k | 10.55 s / 675 MB | 29.05 s / 12,732 MB | 2.8× | 18.9× |
| census_1m | **18.76 s / 821 MB** | **55.76 s / 23,531 MB** | **3.0×** | **28.7×** |

SCX is **slower than scanpy on pbmc10k** (1.28 s vs 0.94 s). At that size
the matrix fits in cache and the streaming decode is overhead the eager path
does not pay; the crossover is somewhere above 10K cells. The memory side
never crosses: 327 MB against 848 MB even where SCX loses on time, and
28.7× at 1M cells, where the whole point is that scanpy's path needs 23 GB
resident and this one needs 0.8.

The two other methods trade accuracy for speed on the same fixtures:
`method="mean"` is 18.34 s at census_1m (no control set) and
`method="zscore"` is 34.21 s.

**Parity is exact.** Spearman is 1.0 and the maximum absolute score
difference is 0.0 at all three panel sizes on pbmc3k, pbmc10k,
tabula_sapiens_100k, census_500k and census_1m. That requires passing
scanpy's own control genes to SCX via `ctrl_genes=`, which the benchmark
reconstructs through scanpy's private binning helpers — scanpy 1.12 keeps
`control_genes` as a function local and writes nothing to `uns`. The route
is load-bearing rather than ceremonial: SCX's own sampler is seeded
independently of numpy and draws a different set, scoring Spearman
0.92 / 0.73 / 0.81 at K = 25 / 100 / 500 on pbmc3k. A silent fallback to it
would leave the parity column comparing two different control sets and
reporting agreement, so the module records a typed
`scanpy_private_api_drift` gap instead of falling back.

On `smartseq2` the absolute difference is 0.017–0.038 rather than 0.0, for
the same float32-accumulator reason described under
[Fused QC + filtering](#fused-qc--filtering-vs-scanpy-accel_qc_filter). Its
Spearman is still exactly 1.0.

Source: `benchmarks/comprehensive/results/raw/accel_score_genes__accel_score_genes__{pyscx_cpu_scanpy,scanpy_cpu}__{pbmc3k,pbmc10k,smartseq2,tabula_sapiens_100k,census_500k,census_1m}.json`
and `…__{pyscx_cpu_mean,pyscx_cpu_zscore}__census_1m.json`,
capture `candidate_community_20260920` (pyscx 0.18.0 release). Each arm
primes with a discarded scoring call before the timed loop — the first
`sc.tl.score_genes` in a process pays a fixed 1.0 s cost the rest do not,
and panels are scored smallest-first.
 Manifest provenance for every community-benchmark number is recorded
once under [The laptop test](memory.md#the-laptop-test-a-full-pipeline-under-a-fixed-ceiling).

## Harmony2 batch integration + LISI

Rust-native re-implementation of the Harmony2 algorithm (Korsunsky et al., 2019) and the Local Inverse Simpson Index (LISI). Exposed via `pyscx.accel.harmony_integrate` and `pyscx.accel.compute_lisi`; R wrappers are `rscx::scx_harmony_integrate` and `rscx::scx_compute_lisi`. GPU path available behind the `gpu` feature (`pyscx.accel.harmony_integrate(adata, ..., device="gpu")`).

**Numerical parity** is pinned against **harmonypy 0.2.0** at the level of the clustering primitives, in `scx-accel/src/harmony/harmony_reference_values.rs`, so `cargo test` gates it with no Python installed: the M-step, the cosine-distance kernel, the ridge correction against `torch.linalg.inv`, and `update_R`'s softmax half. End-to-end agreement is gated as a mean per-PC Pearson correlation floor by the `accel_harmony` benchmark (`benchmarks/comprehensive/thresholds.yaml`). Correlation is the strongest claim available end to end: SCX seeds k-means++ from `rand_chacha` where harmonypy uses `sklearn.KMeans` and R uses Mersenne Twister, so the runs start from different cluster geometry.

> **Withdrawn, pending re-measurement.** A table here previously reported *mean per-PC Pearson r* of **0.999 / 0.989 / 0.999** against R `harmony` "v2.x" on three fixtures. Three problems: the numbers pre-date the soft k-means M-step (`scx-accel/src/harmony/cpu.rs::update_y`), which changes the corrected embedding; the fixtures they were measured on are `.npz` files under `benchmarks/results/harmony/reference/` that are **gitignored**, so `pyscx/tests/test_harmony_validation.py` skips for every contributor and for CI and nobody can reproduce them; and the installed R package is **1.2.4**, not v2.x (the *algorithm* is Harmony2 — the version string was wrong). The figures are removed rather than restated, and will return when they are measured on current code against a fixture that ships.

> **⚠️ Withdrawn, pending re-measurement — the Harmony wall-time table below and
> every figure derived from it.** These were captured before the soft k-means
> M-step (`scx-accel/src/harmony/cpu.rs::update_y`). The M-step adds a centroid
> gemm, a column normalization and a full distance recomputation on **every**
> k-means sub-iteration — up to `max_iter × max_iter_kmeans` = 60 per run — so
> the `scx-accel CPU` and `scx-accel GPU` rows, the α wall exponents fitted from
> them, and the "~13M cells in ~2 h" / "~23M cells in ~3 h" capacity
> extrapolations do not describe the current code. The accuracy table above was
> withdrawn for the same reason; these were left standing by mistake and are
> withdrawn on the same grounds.
>
> First aligned measurement on the gate fixtures, for scale — census_1m, 1M
> cells, 2000 seurat_v3 HVGs, 618 real donors, K=100, all arms pinned to the
> same seed, cluster count, tolerances and dynamic-lambda policy (job 2843742):
> **harmonypy 1083 s, scx-accel CPU 423 s, scx-accel GPU 121 s** — 2.6× and
> 8.9× against harmonypy. A single point, not a replacement for the sweep.
>
> (An earlier revision of this note quoted 1697 / 565 / 125 s. Those came from a
> run that skipped the HVG subset and left `lamb` unaligned, so it timed Harmony
> over 32k genes under a different ridge policy — a different computation, not a
> noisier measurement of this one.)
>
> The **peak-RSS** table is retained: the M-step allocates no new host memory
> (`update_y` writes into the existing `y` buffer, and the device gemm targets
> the already-allocated `d_y`), so those figures are unaffected by this change.

Scaling sweep (d=30, K=100, theta=2, max_iter=10) — wall time in seconds per dataset size:

| Impl / device | D1 (2.7K) | D2 (11.8K) | D3 (50K) | D4 (100K) | D5 (500K) | D6 (1M) | D7 (5M) | α (wall) |
|---------------|---:|---:|---:|---:|---:|---:|---:|---:|
| scx-accel CPU | 5.7 | 22.1 | 69.7 | 19.7 | 100.0 | 236.8 | 2,249.2 | **0.67** |
| scx-accel GPU | — | 4.4 | 10.6 | 20.8 | 109.7 | 209.3 | 1,868.3 | **1.00** |
| harmonypy (CPU) | 5.2 | 7.0 | 12.4 | 53.4 | 77.0 | 166.3 | 1,344.6 | **0.73** |
| R harmony (CPU) | — | 8.7 | 36.7 | 67.8 | 312.3 | 626.8 | 4,831.2 | **1.02** |

Peak RSS in MB (host; GPU VRAM not counted):

| Impl / device | D1 | D2 | D3 | D4 | D5 | D6 | D7 | β (RSS) |
|---------------|---:|---:|---:|---:|---:|---:|---:|---:|
| scx-accel CPU | 455 | 6,771 | 47,798 | 784 | 2,187 | 3,945 | 174,779 | **+0.21** |
| scx-accel GPU | — | 530 | 678 | 890 | 12,218 | 22,376 | 174,666 | **+1.04** |
| harmonypy (CPU) | 563 | 6,773 | 47,797 | 1,164 | 2,521 | 4,733 | 174,778 | **+0.44** |
| R harmony (CPU) | — | 6,659 | 47,788 | 86 | 294 | 552 | 2,623 | **−0.35** |

Peak-RSS anomalies at D3 reflect the in-process PCA-cache build (densifies a 50K×2K float32 scaled matrix) rather than Harmony itself; R harmony dodges the spike because it receives a pre-built NumPy matrix from a child `Rscript` process. Scaling exponents α/β fit `log(y) = α·log(N) + b` over the points above; full per-PC correlations, log-log plots, and secondary PC/cluster-count sweeps live in `benchmarks/results/harmony/REPORT.md`.

### Extrapolated capacity (500 GB / 1000 GB RAM)

Power-law extrapolation of the **D5–D6–D7** points (`log y = α log N + b`,
i.e. large-N regime only) gives a rough read on the largest dataset each
implementation can process for a given memory budget, and how long it would
take. Peak RSS in these rows includes the scanpy `normalize → PCA` cache build
that runs inside the benchmark driver — for scx-accel CPU/GPU and harmonypy
that is the dominant allocation at D7. Supplying a precomputed PCA (skipping
`_build_pca_cache`) shifts their RSS scaling onto the R-harmony curve
(β≈0.95 — memory-proportional to N), which dramatically raises the capacity.

| Impl / device | β (RSS) | α (wall) | @ 500 GB: N (M cells), wall | @ 1000 GB: N (M cells), wall |
|---|---:|---:|---:|---:|
| scx-accel CPU | 1.98 | 1.36 |  9.1M, 1.4 h |  12.9M, 2.2 h |
| scx-accel GPU | 1.18 | 1.25 | 12.6M, 1.6 h |  22.7M, 3.3 h |
| harmonypy (CPU) | 1.91 | 1.25 |  9.2M, 0.8 h |  13.3M, 1.2 h |
| R harmony 1.2.4 (CPU) | 0.95 | 1.20 |   compute-bound¹ |  compute-bound¹ |

¹ R harmony's RSS scales ~linearly with N (β≈0.95), so a 500 GB budget
would technically fit >1B cells, but the α≈1.20 wall-time curve puts even
50M cells at ~1.5 days of wall time. Memory is not the binding constraint;
throughput is.

**Practical takeaways**

- With the default benchmark driver (scanpy PCA cache + Harmony), a
  1000 GB node supports ~13M cells in ~2 h on scx-accel CPU, ~23M cells in
  ~3 h on scx-accel GPU, and ~13M in ~1.2 h with harmonypy.
- The memory ceiling for scx-accel CPU/GPU and harmonypy sits on the
  scanpy `normalize → PCA` cache build, not Harmony itself. Feeding
  Harmony a pre-computed PCA (a real-world pattern — scanpy pipelines
  usually persist `X_pca` once) should shift each implementation's RSS
  curve onto roughly the R-harmony line (β≈0.95), moving the bottleneck
  onto compute. An isolated Harmony-only RSS measurement is not in the
  current sweep; see `benchmarks/results/harmony/REPORT.md` for the
  raw per-run RSS time-series.
- GPU wins on both axes above D6: at 1000 GB it clears ~23M cells in
  ~3 h, versus 13M cells / 2 h on CPU.

Extrapolations assume d=30 PCs, K=100 clusters, single-covariate batch.
Increasing d or K shifts wall time (see the D4 PC/K secondary sweeps in
`benchmarks/results/harmony/REPORT.md`) but leaves memory roughly
unchanged for the Harmony core.

LISI: `pyscx.accel.compute_lisi` is **~10× faster** than R `lisi::compute_lisi` on D1–D4 (e.g. smartseq2 3.85 s vs 43.11 s; tabula_sapiens_100k 12 s vs 110 s). (The previously reported mean-LISI agreement of 0.8–2.4 % vs the R reference predates the 2026-07 raw-distance kernel fix — §2.3 of the accelerator review — and is pending a benchmark recapture.)

## See also

- [Perturbation metrics](perturbation-metrics.md).
- [docs/scanpy/accel-differential-expression.md](../scanpy/accel-differential-expression.md)
  and [docs/scanpy/accel-integration.md](../scanpy/accel-integration.md) — usage.
