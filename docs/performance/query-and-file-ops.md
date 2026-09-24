# Query engine and file operations

> Part of [SCX performance](README.md).

## Query Engine

| Metric | Result |
|--------|--------|
| Shard skip rate | **55%** average |
| Selective query | **4.2 ms** |
| vs AnnData subsetting | **2.1x** faster |

### Modality-scoped query pushdown vs `scx subset` (multimodal)

Getting a filtered single-modality slice out of a multimodal (CITE-seq–shaped)
file two ways — both the **same `scx` release CLI** (apples-to-apples, no
Python-interpreter RSS baseline):

- `scx query f --modality rna --filter P --output q.scx` — Level-1 catalog
  pruning + per-shard decode-and-filter; only the **matching** cells' rows of
  the modality are ever assembled (`QueryPipeline::collect` never holds the
  whole modality in memory).
- `scx subset f out.scx --modality rna --filter P` — the pre-pushdown
  workaround: `extract_modality` reads the **entire** modality X into memory
  (`read_all_csr_shards_for`), masks rows, then rewrites the file.

Synthetic CITE-seq fixture (RNA 3000 vars @ 5% density + ADT 30 vars), filter
`cell_type == 'rare'` selecting ~5% of cells (spread across all shards, so no
shards are Level-1-skipped — the win is purely avoided materialization). Peak
RSS = `/usr/bin/time -v` Maximum RSS; min-of-3 wall. Reproduce with
[`benchmarks/multimodal_query_bench.py`](../../benchmarks/multimodal_query_bench.py)
(`benchmarks/multimodal_query_bench.sbatch`).

| Cells | `scx query` (pushdown) | `scx subset` (materialize) | Peak-RSS reduction | Wall |
|------:|------------------------:|---------------------------:|-------------------:|-----:|
| 100K | 158 MB · 0.081 s | 258 MB · 0.098 s | **1.63×** | 1.2× |
| 300K | 302 MB · 0.221 s | 532 MB · 0.263 s | **1.76×** | 1.19× |

The peak-RSS **delta** (subset − query ≈ 100 MB @ 100K, 230 MB @ 300K) tracks the
full-modality CSR that `subset` holds resident and pushdown never assembles, so
the reduction grows with cell count; at atlas scale (10–100M cells) `subset`'s
whole-modality materialization runs to many GB while the pushdown query stays
bounded by one shard + the matched slice. Wall is modestly faster (pushdown skips
the mask-after-full-decode). The obs predicate is evaluated once against the
shared global obs axis and applied per modality (see
[docs/multimodal.md § 3.4](../multimodal.md#34-modality-scoped-queries--querymodality)).

## File Operations

| Operation | Speed |
|-----------|-------|
| Append 10K cells | **1 ms** |
| Merge 3 files | **342 MB/s** |
| Compact (after 3 appends) | 0.98x fresh-write size |

### Sort (physical layout)

`scx sort` (see [sharding.md § Sorting for read locality](../sharding.md#sorting-for-read-locality-scx-sort))
globally reorders the obs axis for X-read locality. Measured on
`tabula_sapiens_100k` (100,000 cells × 61,497 genes, 194.9M nnz, 7 CSR shards,
`scx1`/uint16), sorting by `cell_type` (33 categories), in-memory strategy:

| metric | unsorted | sorted by `cell_type` | delta |
|--------|----------|-----------------------|-------|
| X matrix (CSR) size | 394.1 MB | 394.1 MB | **size-neutral** |
| obs predicate index (`cell_type`) | 724.3 KB | 1.3 KB | **~557× smaller** |
| whole file | 427.6 MB | 426.7 MB | −0.2% |
| shards touched per category (mean) | 2.94 | 1.18 | **2.6× fewer** |
| shards touched per category (max) | 4 / 7 | 2 / 7 | — |

**Reading the numbers.** Compression: the X matrix is *size-neutral* here because
**`scx1` codes each row's gene indices independently of row order**, so a
permutation just relocates identically-sized per-row blocks (394.1 MB → 394.1 MB
above). The real on-disk win is the **predicate index**, which collapses from
per-shard scattered row lists (724 KB) to a handful of contiguous ranges (1.3 KB)
once each category occupies one shard range.

This neutrality is **specific to `scx1`** and does **not** generalize to
`zstd`-coded shards or to auto-codec re-selection. `zstd` compresses the whole
shard byte stream, so regrouping *which* cells share a shard changes cross-row
redundancy, and the auto-codec `scx1`-vs-`zstd` median heuristic can flip per
shard under the reorder. On a mixed-codec atlas this shifts X size by a few
percent in **either** direction. Measured: sorting the 149M-cell `drug.scx`
(380.1 GB X, `mixed scx1/zstd`, uint32) by `cell_type` **grew** the X matrix by
~30 GB — that is ~8% of the 380 GB X matrix (equivalently ~6% of the 530.7 GB
whole file) — with value encoding (uint32 → uint32),
predicate index (2.2 GB over the same columns, present in both), CSC (none), and
obs (plain `LargeUtf8`, order-independent) all ruled out by the input catalog, so
the delta is entirely X re-compression. So whole-file size after a sort is
data- and codec-dependent and not guaranteed to shrink — the headline value is
**locality**, not compression. Pin `--codec zstd` or run a follow-up
`scx compact` if output size matters.

Locality: the comparison is against an isolated
unsorted baseline produced by the *same* writer (`scx compact --reshape-obs
--index-obs cell_type`), so the same format-version features (sharded obs) are
present in both and don't confound the delta. The dominant cell types
drop from 3–4 shards to 1–2 contiguous shards; the 2.6× figure is conservative
because real atlas data is already partially clustered by cell type. The
synthetic worst-case (`bench_ops_sort_locality`: 40k cells, 8 categories cycled
across every shard) brackets the upper end at **10 → 2 shards (5×)**, and the gap
widens with shard count at atlas scale. In-memory sort of the 100k file ran in
~73 s at 2.7 GB peak RSS (debug build); the bounded-memory external strategy
(`--memory-budget`) caps peak RSS to one `new_pos`-range partition for atlas-scale
files.

### Grouped sharding (`scx sort --group-by` / `scx convert --group-by`)

Grouped sharding writes a **reference-first, group-clustered** CSR layout — each
`group_by` label occupies a contiguous, never-split shard range, with reference
cells (e.g. `non-targeting`) isolated in shard 0 — so `read_group(label)` /
`read_reference()` touch only that group's byte range
(see [sharding.md](../sharding.md) and [docs/api/](../api/README.md)). It is produced two
ways: `scx sort --group-by` re-shards an existing `.scx`, and `scx convert
--group-by` (Phase 7.4) writes the grouped layout **directly during ingest** —
byte-equivalent to convert-then-sort, but in one pass. Measured by the
`grouped_sort` comprehensive benchmark
(`benchmarks/comprehensive/results/raw/grouped_sort__scx_auto__{dataset}.json`).

**Convert-time grouping: one-pass vs two-pass, and the density auto-route.**
Convert-time grouping does a random-access *gather* of the source whose cost
scales with bytes-read-per-row — `nnz` for CSR, the full `n_vars` for dense — so
`--group-pass auto` (the default) routes **CSR → one-pass** streaming gather and
**dense → two-pass** (plain convert + `scx sort`). Real Perturb-seq fixtures,
`grouped_sort` benchmark (release pyscx, 16 threads; wall = median, RSS **sampled
post-op** — see the note under the table):

| source (X format, cells × genes) | one-pass wall / RSS | two-pass wall / RSS | `auto` route |
|---|---|---|---|
| `tahoe_c38` (CSR, 69,245 × 62,710, by `drug`) | **16.9 s / 1.0 GB** | 27.9 s / 1.4 GB | one-pass (1.7× faster) |
| `replogle_k562` (dense, 68,729 × 6,546, by `gene`) | 3 m 39 s / 6.8 GB | **39 s / 4.7 GB** | two-pass (5.6× faster) |

For a **CSR** source the one-pass gather reads only each row's non-zeros and wins
outright; for a **dense** source it reads full-width rows per gathered cell and is
a ~5× loss, so `auto` falls back to the two-pass path (which costs a transient
~2× output disk for the intermediate file). Output is byte-identical regardless of
route. Both synthetic CSR fixtures confirm the ordering (`pert_synth_10k`:
one-pass 1.6 s vs two-pass 3.4 s; `nb_glm_synth`: 3.6 s vs 6.5 s).

**The RSS column above understates true peak, and is kept as captured.** Those
figures come from a run in which `grouped_sort` sampled `current_rss_mb()` after
each op finished, so a gather buffer that was allocated and freed inside the op
is not in them; a `/usr/bin/time` true-peak measurement of the release `scx` CLI
puts the dense one-pass at ~11.5 GB against ~6.5 GB for two-pass — the memory
motivation for the dense → two-pass route. The benchmark now brackets each op
with `PeakRssSampler` and reports a genuine in-region high-water mark, so a
re-capture will read higher than the table; the numbers here are not restated
because they are the ones the tracked
`results/raw/grouped_sort__scx_auto__*.json` actually contain.

**Grouped sort + reference isolation.** On `replogle_k562` (415 perturbation
groups), `sort --group-by gene --reference non-targeting` isolates all 10,691
`non-targeting` cells into shard 0 and clusters each remaining gene into its own
contiguous shard range; `read_group("MYC")` / `read_reference()` then decode only
that range rather than scanning all shards. The read-locality mechanics (predicate
index collapse, shards-touched-per-group) match the generic "Sort (physical
layout)" numbers above — grouping is the never-split, reference-first special case.
The benchmark's read-back parity check (`read_group(label)` partitions the obs axis
and the reference is isolated) is a hard regression gate
(`thresholds.yaml`: `correctness_passed_int` / `reference_isolated_int` /
`n_group_records`).

**F6 — in-memory grouped-write fast path.** The pre-F6 path streamed the reorder
row-by-row through a single-threaded encoder. F6 fixed both halves: **Phase 0** caps
the emitter's per-shard buffer at `--group-write-block-bytes` (default 256 MB),
sub-flushing an oversized group across shards so grouped write no longer OOMs on a
huge reference group; **Phase 1** added an in-memory **parallel** fast path (`scx sort
--group-by` / `pyscx.sort(group_by=)` with no `--memory-budget`) that gathers the
resident CSR and encodes blocks across rayon threads — **byte-identical** to the
single-threaded path. A/B on real fixtures (release pyscx, 16-core `cpu`,
`/usr/bin/time -v` true peak; "legacy" = `SCX_SORT_NO_INMEM_FAST=1`, the
Phase-0 single-threaded emitter):

| dataset | scx fast | scx legacy | speedup | fast peak RSS |
|---|---|---|---|---|
| `chemogenetic_rgfp` (136K × 18K, 909M nnz) | **6.5 s** | 24.2 s | 3.7× | 18.7 GB |
| `replogle_k562` (69K × 6.5K) | **12.8 s** | 28.0 s | 2.2× | 6.3 GB |
| `tahoe_c38` (69K × 63K) | **9.0 s** | 17.4 s | 1.9× | 5.0 GB |

The parallel gather trades peak RSS for speed — on
`chemogenetic_rgfp` (127K-cell reference group) the 16-way gather peaks at
~18.7 GB vs the emitter's ~11.9 GB, still far under the historical 48–72 GB OOM;
neutral on the other two. Bound it with `RAYON_NUM_THREADS`, or set
`SCX_SORT_NO_INMEM_FAST=1` to force the memory-lean single-threaded emitter. A
`grouped_sort` `grouped_peak_rss_mb` gross-regression ceiling gates each dataset
in `thresholds.yaml` (wall time stays measured-not-gated — hardware-sensitive).
It is spelled `grouped_peak_rss_mb`, not `peak_rss_mb`, because the latter is a
reserved `add_run` parameter that never reaches `runs[].extra` — the only place
a threshold reads. The ceilings carried the reserved spelling until 2026-09 and
could not resolve a value on any run; the silence went unnoticed because none of
the three datasets is in a capture tier, so the triple never ran and the
resulting violation was skipped rather than reported.

**Beyond grouped sharding (full benchmark campaign, release build).** Further
results from the comprehensive suite:

- **Compression.** Codec choice matters more than which format you use: the
  default `scx_auto` codec (`census_1m` 3.99 GB; `tabula_100k` 449 MB) is not
  scx's most compact integer codec — `scx_compact_trial` (framed ShufDeltaZstd)
  brings `census_1m` to 2.80 GB, `tabula_100k` to 234 MB, and
  `chemogenetic_rgfp` to 968 MB, and holds ≈parity on log-normalized/float
  data. See [codec.md § Codec tradeoff summary](../codec.md#codec-tradeoff-summary--scx1-vs-shufdeltazstd)
  for the full Scx1-vs-ShufDeltaZstd comparison.

  > **Why isn't ShufDeltaZstd the default?** Despite better compression,
  > ShufDeltaZstd has **no in-VRAM decode via BitPacker4x** (Scx1 decodes its
  > indices in VRAM; ShufDeltaZstd's GPU path uploads planes / host-bounces).
  > CPU decode is **≈ Scx1 parity (~1.09× at a 16 K-row shard, at/below Scx1
  > for smaller shards)** after the Phase-B SSE2 byte-transforms (was 1.3–1.8×
  > slower); the remaining reason to keep Scx1 as the `auto` default is the GPU
  > analysis path, not CPU decode. `compact-trial`
  > gives the best of both by trial-encoding each shard and keeping the
  > smaller, but doubles encode time and frames all output (no in-VRAM Scx1
  > decode). See [codec.md § "Codec tradeoff summary"](../codec.md#codec-tradeoff-summary--scx1-vs-shufdeltazstd)
  > for a full comparison and guidance on when to use which codec.

  **GPU ShufDeltaZstd decode — Phase 0 profiling (go/no-go).** Before building
  a GPU decode kernel for ShufDeltaZstd, Phase 0
  quantified the host-bounce ceiling. Measured with
  `benchmarks/scripts/bench_gpu_codec.py` — `to_gpu_anndata(device="gpu")`,
  median of 3 runs on one H100, byte-exact parity vs the host CSR decode
  verified for every codec:

  | dataset | codec | route | wall (s) | throughput (Mnnz/s) | host→device upload |
  |---|---|---|---|---|---|
  | census_1m (1.40 B nnz) | `scx1` | `scx_device_decode_gpu` | 10.05 | 139.5 | 8 MB (indptr only) |
  | census_1m | `compact_trial` | `scx_device_handoff_streamed` | 27.18 | 51.6 | 11.2 GB |
  | census_1m | `shufdelta` | `scx_device_handoff_streamed` | 28.23 | 49.7 | 11.2 GB |
  | census_500k (0.75 B nnz) | `scx1` | `scx_device_decode_gpu` | 5.35 | 139.6 | 4 MB |
  | census_500k | `compact_trial` | `scx_device_handoff_streamed` | 14.08 | 53.1 | 6.0 GB |
  | census_500k | `shufdelta` | `scx_device_handoff_streamed` | 14.02 | 53.3 | 6.0 GB |

  Host-bounce throughput is **0.36–0.38× of Scx1's in-VRAM decode** — far below
  the 90% exit criterion, so Phase 1 is justified (**GO**). The gap is
  **CPU-decode-bound, not PCIe-bound**: the 11.2 GB decoded-CSR upload crosses
  PCIe 4.0 in <1 s, yet the host-bounce wall is ~27 s for 1.4 B nnz. Scx1
  instead uploads only the ~8 MB indptr and decodes indices+values in VRAM.

  A CPU per-stage micro-bench (task 0b, `scx-codec/benches/codec_bench.rs ::
  bench_shufdelta_decode_stages`, 16 K-row × ~2000-nnz shard) shows *where* that
  CPU decode goes, and **inverts the spec's `§8` assumption that zstd
  dominates**:

  | stage | u16 scalar | u16 SSE2 | u32 scalar | u32 SSE2 |
  |---|---|---|---|---|
  | zstd-decompress | 93.5 ms | 91.4 ms | 124.3 ms | 124.2 ms |
  | byte-undelta (prefix scan) | 156.7 ms | **16.3 ms** (9.6×) | 313.8 ms | **32.8 ms** (9.6×) |
  | byte-unshuffle (transpose) | 102.9 ms | **26.7 ms** (3.9×) | 181.1 ms | **53.7 ms** (3.4×) |

  The original scalar `byte_undelta_planes` / `byte_unshuffle` ran at ~0.4–0.7
  GB/s and together were **63–73% of the per-shard CPU decode**, with zstd only
  27–37% — inverting the spec's `§8` assumption that zstd dominates.

  **Phase B result — SSE2 SIMD byte-transforms (`scx-codec/src/simd.rs`).** The
  two transforms are 128-bit SSE2 kernels (baseline-guaranteed on x86_64;
  scalar fallback on other arches; bit-identical, `simd == scalar` proptested):
  a log-step (Hillis–Steele) wrapping-`u8` prefix scan for undelta and an
  `unpack`-cascade transpose (width-specialized for 2/4/8) for unshuffle. undelta
  drops ~9.6× and unshuffle ~3.4–3.9× (SSE2 column above), so zstd becomes the
  dominant decode stage again. Whole-shard `decode_shard` (16 384-row × ~2000-nnz,
  u16 indices) is **ShufDeltaZstd 274.6 ms vs Scx1 251.7 ms — ~1.09× (within
  ~9%)**, down from the previous 1.3–1.8×; on the smaller 2048-row shard
  ShufDeltaZstd is at/below Scx1 (5.4 vs 7.7 ms). This removes ShufDeltaZstd's
  CPU-decode training tax and is the Phase-C adaptive-`auto` enabler. (Phase 1's GPU
  decode below offloads the *same* transforms for the in-VRAM analysis path.)

  **Phase 1 result — GPU ShufDeltaZstd decode lands (H100).** Phase 1 added a
  GPU decode path for framed/unframed ShufDeltaZstd shards: the CPU still runs
  zstd, but the dominant undelta/unshuffle/convert transforms move to two CUDA
  kernels (`scx-gpu/kernels/shufdelta.cu`), and only the narrower pre-convert
  plane bytes cross PCIe. Re-running the same `bench_gpu_codec.py` (median of 3,
  byte-exact parity verified for every codec):

  | dataset | codec | Phase 0 (host-bounce) | Phase 1 (GPU) | wall (s) |
  |---|---|---|---|---|
  | census_1m | compact_trial | 0.37× Scx1 | **1.18× Scx1** | 8.48 (vs Scx1 9.98) |
  | census_1m | shufdelta | 0.36× | 0.86× | 11.64 |
  | census_500k | compact_trial | 0.38× | **1.20× Scx1** | 4.57 (vs Scx1 5.49) |
  | census_500k | shufdelta | 0.38× | **1.21× Scx1** | 4.54 |

  Framed ShufDeltaZstd decodes at **~parity-to-faster than Scx1 (0.86–1.21×
  across runs), a ~2.4–3.2× jump over the Phase-0 host-bounce** — vindicating the
  0b finding (offloading the transform-dominated CPU cost was the lever, not the
  spec's ~1.1× estimate). Two corroborating signals: host→device upload fell
  from ~11.2 GB to ~5.6 GB per file (narrower u16/u8 plane bytes instead of the
  decoded i32+f32 CSR), and `uns["scx_accel"]["to_gpu_anndata"]
  ["n_shards_shufdelta_gpu"]` reports 62 (census_1m) / 31 (census_500k) shards
  on the GPU path. `transfer_mode` stays `scx_device_handoff_streamed` (Phase 1
  uploads decompressed bytes; the compact_trial file's 62 X-shards all chose
  ShufDeltaZstd, so it and the pure-shufdelta file carry identical X content and
  their wall gap is run-to-run variance). Phase 2 (nvcomp GPU zstd) remains a
  follow-on.

  **Phase 1.5 result — pipelined decode (parallel zstd + multi-stream, H100).**
  Phase 1.5 pipelines the per-group CPU zstd across worker threads (bounded
  channel) and overlaps it with async H2D on a dedicated copy stream + GPU
  kernels (event-gated 2-slot ring), mirroring `gpu_shard_source.rs`. A/B vs the
  Phase-1 sequential path (`SCX_SHUFDELTA_GPU_SEQUENTIAL=1`), median of 3:

  | dataset | codec | seq wall | pipelined wall | pipeline | vs Scx1 |
  |---|---|---|---|---|---|
  | census_1m | compact_trial | 8.53 s | **7.90 s** | 1.08× | 1.26× |
  | census_1m | shufdelta | 11.23 s | 11.20 s | ~1.0× | 0.89× |
  | census_500k | compact_trial | 4.51 s | **4.02 s** | 1.12× | 1.33× |
  | census_500k | shufdelta | 4.50 s | **4.05 s** | 1.11× | 1.32× |

  The pipeline is **1.08–1.12× over sequential and never slower** (parity holds;
  a GPU test asserts pipelined output is byte-identical to both the sequential
  path and the source CSR), reaching up to **1.33× Scx1**. The gain is modest —
  and the profiler shows why: on census_1m the GPU-side critical path is
  `htod ≈ 778 ms + gpu_decode ≈ 83 ms + indptr ≈ 43 ms ≈ 0.9 s`, a fraction of
  the ~7.9 s wall. Most of `to_gpu_anndata`'s wall is **fixed metadata assembly +
  cupy handoff** (the same for every codec — Scx1 is 9.9 s), so optimizing the
  X-decode moves the *total* only ~1.1×. The pipeline is the default for framed
  ShufDeltaZstd (≥2 groups); `SCX_SHUFDELTA_GPU_SEQUENTIAL=1` forces the
  sequential path. The largest remaining GPU-side cost is the ~5.6 GB H2D of
  decompressed plane bytes — which Phase 2 (nvcomp, upload compressed instead)
  would attack.

  **Phase 2 result — nvcomp full in-VRAM decode (opt-in, H100).** Phase 2 adds a
  GPU-zstd path: upload the **compressed** per-group frames and decompress them
  on-device via NVIDIA **nvcomp 5.1** (runtime-`dlopen`ed from the conda env; no
  build dependency), then finish with the existing kernels. It achieves the
  architectural goal — `transfer_mode == scx_device_decode_gpu` (Scx1 route
  parity) with **~3.9× less host→device transfer** (compressed vs decompressed
  planes) — and is byte-exact. A/B on framed compact_trial (median of 3):

  | dataset | path | wall | transfer_mode | upload |
  |---|---|---|---|---|
  | census_1m | pipeline (default) | **8.21 s** | handoff_streamed | 5.64 GB |
  | census_1m | nvcomp (opt-in) | 11.60 s | scx_device_decode_gpu | **1.46 GB** |
  | census_1m | Scx1 | 9.67 s | scx_device_decode_gpu | 8 MB |
  | census_500k | pipeline (default) | **4.10 s** | handoff_streamed | 3.03 GB |
  | census_500k | nvcomp (opt-in) | 5.97 s | scx_device_decode_gpu | **0.77 GB** |

  **nvcomp is ~1.4× slower end-to-end** than the Phase-1.5 pipeline despite the
  PCIe win: `to_gpu_anndata` is metadata-bound, and nvcomp adds per-shard
  overhead (temp alloc + host pointer arrays + a stream sync per shard, across
  62/31 shards) that the pipeline's parallel CPU zstd avoids. A standalone spike
  confirmed nvcomp batched zstd is competitive-to-2× faster than parallel CPU
  **per batch** (256 frames: 6.9 vs 3.4 GB/s) — the loss is the per-shard call
  overhead, not the kernel. So Phase 2 is **opt-in** (`SCX_SHUFDELTA_NVCOMP=1`);
  the pipeline stays the default. nvcomp is the right choice only when the
  device-decode route or the ~4× smaller PCIe transfer matters more than wall
  time (PCIe- or CPU-constrained hosts). Reducing the per-shard overhead
  (cross-shard batching, fewer syncs) is the natural follow-on.

  **Phase 2.x result — nvcomp cross-shard batching (H100, post-T4 refresh).**
  Phase 2.x collapses the per-shard nvcomp overhead: instead of two
  `nvcompBatchedZstdDecompressAsync` calls *per shard* (each with its own temp
  alloc + host pointer arrays + stream sync + status readback), a uniform run of
  framed ShufDeltaZstd shards is decoded in **2 batched calls total** (all indices
  frames, then all values) — removing ~124→2 syncs on a 62-shard file. Full A/B
  matrix (median of 3), still opt-in via `SCX_SHUFDELTA_NVCOMP=1`, per-shard forced
  with `SCX_NVCOMP_NO_BATCH=1`. The framed `compact_trial` fixture is uniform
  ShufDeltaZstd (62 / 31 shards); `scx1` is the in-VRAM baseline:

  | dataset | path | wall | transfer_mode | upload |
  |---|---|---|---|---|
  | census_1m (62 shards) | scx1 (in-VRAM baseline) | 9.24 s | scx_device_decode_gpu | 8 MB |
  | census_1m | pipeline (default) | **7.73 s** | handoff_streamed | 5.64 GB |
  | census_1m | nvcomp **batched** (2.x) | 8.01 s | scx_device_decode_gpu | **1.46 GB** |
  | census_1m | nvcomp per-shard (Phase 2) | 10.84 s | scx_device_decode_gpu | 1.46 GB |
  | census_500k (31 shards) | pipeline (default) | **3.91 s** | handoff_streamed | 3.03 GB |
  | census_500k | nvcomp **batched** (2.x) | 4.23 s | scx_device_decode_gpu | **0.77 GB** |
  | census_500k | nvcomp per-shard (Phase 2) | 5.80 s | scx_device_decode_gpu | 0.77 GB |

  Batching makes nvcomp **1.35× faster than the per-shard path at 62 shards**
  (10.84 → 8.01 s) and **1.37× at 31 shards** (5.80 → 4.23 s), cutting the nvcomp
  penalty vs the pipeline from ~1.46× to **~1.04× (62 shards) / ~1.08× (31)** while
  keeping the `scx_device_decode_gpu` route and 3.9× smaller PCIe transfer. It is
  still **slightly slower than the pipeline** — `to_gpu_anndata` is
  metadata-assembly-bound, so the PCIe win does not translate to wall time — so
  nvcomp **stays opt-in**; batching is applied automatically *within* the opt-in.
  (`census_1m_shufdelta.scx` is v3-unframed and never enters the nvcomp path; the
  62-shard case is measured on the framed `census_1m_compact_trial.scx`.)

  **T4 result — parallel obs/var metadata decode (H100 + 32-core CPU).** The
  obs/var sharded-metadata decode (`reader::read_sharded_layout_by_prefix`) was
  single-threaded; T4 fans the per-shard zstd + Arrow-IPC decode across rayon
  (byte-identical; serial fallback via `SCX_METADATA_DECODE_SERIAL=1`). This is the
  codec-agnostic lever on the metadata-bound wall Phase 2.x identified. Isolated
  `read_obs()` (median of 5, `read_obs_timing` harness):

  | fixture | obs shards | serial | parallel (32t) | speedup |
  |---|---|---|---|---|
  | census_1m_scx1 | 62 (~1.1 GB obs) | 1.37 s | **0.97 s** | **1.40×** |
  | census_500k_scx1 | 31 (~0.55 GB) | 0.55 s | 0.48 s | ~1.17× |

  Scaling plateaus by ~8 threads (census_1m: 1.05 s @4t → 1.03 s @8t → 0.97 s @32t):
  the shard *decode* parallelizes, but the downstream `assemble_sharded_metadata`
  (concat + single-pass global-dictionary unification) is serial and sets a ~1 s
  floor. End-to-end `to_gpu_anndata` (serial vs parallel metadata, median of 3):
  census_1m_scx1 9.41 → **9.22 s** (1.02×), census_1m_compact_trial 8.07 → **7.22 s**
  (1.12×), census_500k_scx1 5.38 → **5.19 s** (1.04×). So T4 is a real but partial
  win: it removes the serial *decode* of the metadata wall (0.2–0.85 s off
  `to_gpu_anndata`), but the serial *assemble* + cupy handoff remain the larger
  residual — the next metadata lever.

- **Out-of-core peak RSS** (true high-water mark, full-data pass): scx streaming stays
  ~flat while a full materialize scales with dataset size —

  | dataset | scx streaming | scx materialize |
  |---|---|---|
  | `census_500k` | 5.5 GB | 8.4 GB |
  | `census_1m` | 6.0 GB | 16.1 GB |
  | `census_5m` | **19.0 GB** | 126.7 GB |

  Streaming stays bounded by one shard's footprint at every scale, which is the
  capability that matters at atlas scale.
- **Parallel read scaling** — scx scales 3.6× (`tabula_100k`) / 4.5× (`census_1m`) to 32
  threads.
- **ML training loader** — scx `TrainingDataset` sustains 45.5 (`tabula_100k`) / 767–1,193
  (`census_1m`) batches/s.
- **Fidelity** — scx's dtype/materialization knobs
  (`to_anndata(container="dense", data_dtype="float16", allow_lossy=True)`) verify
  losslessly on `pbmc3k` / `nb_glm_synth` / `tabula_100k`.

Net: scx's durable edges are **per-group reads, out-of-core, parallel scaling, ML
throughput, and breadth** (query engine, accelerators, cloud, multimodal, R).

---

## See also

- [docs/operations.md](../operations.md) — the semantics of each op.
- [Storage and reads](storage-and-reads.md) — predicate-pushdown reads.
