# SCX Performance

Benchmark results for SCX across compression, read/write, memory, analysis accelerators, GPU, training loader, and query engine.

All benchmarks on Intel Xeon Platinum 8468, 32 cores, 1-2 TB RAM unless noted otherwise. GPU benchmarks on NVIDIA H100 80GB HBM3.

---

## Compression

| Dataset | Cells | h5ad -> SCX | vs Zarr+Zstd | vs SLAF |
|---------|-------|-------------|--------------|---------|
| PBMC 3K | 2,700 | **4.9x** smaller | 2% smaller | — |
| Smart-seq2 | 50,000 | **2.9x** smaller | 5% smaller | — |
| Tabula Sapiens | 100,000 | **4.9x** smaller | 11% smaller | — |
| Census 1M | 1,000,000 | **4.8x** smaller | 10% smaller | **1.7x** smaller |
| Census 5M | 5,000,000 | **7.3x** smaller | 7% smaller | — |

SLAF on-disk size is measured across the `.slaf/` Lance + statistics directory
via `_dir_size`. See `benchmarks/comprehensive/results/raw/compression__slaf__census_1m.json`.

## Read Speed (full load to AnnData)

| Dataset | SCX (auto) | h5ad (none) | h5ad (gzip) | Zarr (lz4) | TileDB-SOMA | SLAF |
|---------|-----------|-------------|-------------|------------|-------------|------|
| PBMC 10K | 0.31s | 0.12s | 0.94s | **0.08s** | 0.44s | — |
| Tabula Sapiens 100K | **0.58s** | 1.41s | 7.56s | 1.20s | 3.79s | — |
| Census 1M | **2.74s** | 5.89s | 48.5s | 3.99s | 12.9s | 53.0s |
| Census 5M | **35.4s** | 43.7s | 291s | 40.6s | 80.6s | — |

SCX is the fastest reader at census scale — **1.5x faster than Zarr**, **2.1x faster than uncompressed h5ad**, **17.7x faster than gzip h5ad**, and **19.3x faster than SLAF** on 1M cells. Parallel read scaling: up to **7x** at 32 threads. The SLAF full-read path (`LazyAnnData.compute()`) goes through Polars fragment processing to build the CSR — competitive for predicate-selective reads but heavy for "load everything" at census scale.

## Read Scaling (parallel shard decode)

SCX parallelizes shard decoding via rayon. Full load, 32 threads vs 1 thread:

| Dataset | SCX (auto) | Speedup | Zarr (lz4) | h5ad |
|---------|-----------|---------|------------|------|
| Census 500K | 1.5s | **6.3x** | 2.0s (1.0x) | no scaling |
| Census 1M | 3.0s | **6.1x** | 3.7s (1.0x) | no scaling |
| Census 5M | 37.5s | **3.1x** | 89.6s (1.0x) | no scaling |

No other format scales full-read throughput with threads: h5py holds a global lock, Zarr's chunk reads are I/O-bound on local/HPC filesystems, and TileDB-SOMA's fragment-based reads don't benefit from additional threads at these sizes.

## Conversion (h5ad → format)

End-to-end write time (in-memory AnnData → target format) and peak RSS during the write. Single-threaded, 3 runs median, median wall / max peak RSS reported.

| Dataset | SCX (auto) | h5ad (none) | h5ad (gzip) | Zarr (lz4) | TileDB-SOMA |
|---------|-----------|-------------|-------------|------------|-------------|
| PBMC 10K | 1.78s / 0.4 GB | 0.33s / 0.4 GB | 4.14s / 0.4 GB | **0.33s** / 0.3 GB | 7.12s / 1.1 GB |
| Smart-seq2 50K | 7.31s / 1.3 GB | **1.37s** / 1.2 GB | 29.4s / 1.2 GB | 1.48s / 0.3 GB | 26.7s / 2.3 GB |
| Tabula Sapiens 100K | 4.70s / 1.9 GB | **2.06s** / 1.7 GB | 31.7s / 1.7 GB | 2.09s / 0.3 GB | 44.2s / 2.8 GB |
| Census 500K | 13.7s / 6.7 GB | **6.83s** / 6.0 GB | 123s / 6.0 GB | 7.74s / 0.4 GB | 173s / 7.0 GB |
| Census 1M | 30.0s / 12.3 GB | **11.8s** / 11.1 GB | 224s / 11.1 GB | 14.6s / 0.5 GB | 255s / 12.1 GB |

Takeaways:
- **h5ad (none) and Zarr (lz4) are fastest at writing** because they do the least work — no compression (h5ad none) or minimal LZ4 (Zarr). They pay for it on the read side (Zarr lz4 files are ~4–7× larger than SCX; see Compression).
- **SCX writes are 5–8× faster than h5ad (gzip)** while producing smaller files.
- **SCX writes are 8–10× faster than TileDB-SOMA** across all sizes tested. TileDB's fragment-based write path has significant per-row overhead.
- **SCX and Zarr have similar peak RSS characteristics for writes** — both stream compressed output incrementally. h5ad materializes each chunk in memory before compressing, explaining its higher RSS on large datasets.
- Census 5M write benchmarks are not yet available; parallel write scaling data at 500K is in the next section.

Source: `benchmarks/comprehensive/results/raw/write__{format}__{dataset}.json`.

## Write Scaling (parallel shard encoding)

SCX parallelizes shard encoding via rayon — compression, checksumming, and statistics run on separate threads. Benchmarks cover two modes:

- **`write_only`** — in-memory AnnData → SCX (isolates the SCX encoder).
- **`full`** — end-to-end h5ad → SCX (h5ad read + SCX write; what most users actually do).

### `write_only`: in-memory AnnData → SCX, Census 500K

| Codec | 1 thread | 32 threads | Speedup |
|-------|---------|-----------|---------|
| SCX (pcodec) | 36.1s | 11.4s | **3.2x** |
| SCX (zstd) | 35.9s | 11.4s | **3.2x** |
| SCX (scx1) | 32.3s | 12.0s | 2.7x |
| SCX (auto) | 32.3s | 12.7s | 2.5x |
| SCX (none) | 21.3s | 12.5s | 1.7x |

### `full`: h5ad → SCX (auto codec), dataset sweep

| Dataset | 1t | 2t | 4t | 8t | 16t | 32t | Speedup |
|---------|---:|---:|---:|---:|----:|----:|--------:|
| PBMC 10K | 1.78s | 1.78s | 1.78s | 1.77s | 1.77s | 1.78s | 1.0x |
| Smart-seq2 50K | 11.95s | 9.76s | 7.86s | 7.77s | 7.68s | **7.71s** | 1.6x |
| Tabula Sapiens 100K | 9.40s | 6.40s | 5.57s | 4.74s | 4.64s | **4.68s** | 2.0x |
| Census 500K | 34.4s | 23.3s | 18.1s | 15.3s | 14.4s | **14.0s** | 2.5x |
| Census 1M | 72.4s | 46.5s | 35.3s | 32.1s | 32.8s | **33.5s** | 2.2x |

### `full`: h5ad → SCX codec sweep, Census 500K

| Codec | 1 thread | 32 threads | Speedup |
|-------|---------:|-----------:|--------:|
| SCX (pcodec) | 39.1s | 13.6s | **2.9x** |
| SCX (zstd) | 37.7s | 14.4s | 2.6x |
| SCX (auto) | 34.4s | 14.0s | 2.5x |
| SCX (scx1) | 35.6s | 16.9s | 2.1x |
| SCX (none) | 23.2s | 14.7s | 1.6x |

Takeaways:
- **Small datasets (≤10K cells) don't benefit from threading** — write finishes before rayon's fork/join amortizes. Use fewer threads to avoid overhead.
- **Speedup plateaus at 8–16 threads** — sequential h5ad read, output I/O, and metadata serialization bound further scaling. Full-mode speedups are slightly lower than write-only because the h5ad read is single-threaded via `h5py`.
- **Heavier codecs (pcodec, zstd) parallelize best** — more CPU work per shard gives rayon more to schedule. `none` parallelizes least because the hot path is I/O-bound.
- **`auto` picks `scx1` for UMI data**, so its scaling profile matches `scx1` (median-value-based heuristic — see [docs/codec.md](codec.md#8-automatic-codec-selection)).

Source: `benchmarks/comprehensive/results/raw/parallel_write_scaling__{codec}__{dataset}.json` (`metadata.scaling_wall_s.full` and `metadata.scaling_wall_s.write_only`).

### Rewrite ops: the parallel shard encode (2026-09-10)

`scx compact` / `scx optimize` re-encode every shard, and since PR #524 that
encode is parallel *inside* one shard. PR #526 measured what that left behind — both ops stop scaling well before the
core count, with a substantial part of the wall on the serial per-shard path —
and made `optimize` re-encode several shards at once under a byte-bounded
in-flight cap.

**Those wall and peak-RSS figures are not reproduced here.** They came from a
two-worktree CLI A/B (one binary per commit), which is not a shape the
comprehensive harness can produce, so they cannot be backed by a
`benchmark × format × dataset` manifest entry — and
[benchmark_manifest.md](benchmark_manifest.md) admits no exception for a claim
of that shape in this file. They live in
[multithreading.md § Across shards, in a rewrite op](multithreading.md#across-shards-in-a-rewrite-op-scx-optimize)
instead, with their provenance, where they document the architecture rather than
standing as captured results.

What *will* appear here is the `fragment_ops` `wall_s__optimize` /
`peak_rss_mb__optimize` arm added alongside them, once a capture runs — that one
is a real triple.

### Streaming conversion (h5ad → SCX)

`scx convert --stream` and `pyscx.from_h5ad(path, out)` use a
shard-at-a-time pipeline (`scx_convert::h5ad_to_scx_streaming`) that
holds only one shard's worth of CSR in memory plus the always-resident
`indptr`. `pyscx.from_anndata(backed_adata, out)` auto-routes to the
same pipeline. Recommended whenever the input doesn't comfortably fit
in node RAM.

Peak RSS bound: `shard_target_rows × n_vars × density × ~16` bytes +
`(n_obs + 1) × 8` bytes for indptr. At default
`shard_target_rows = 16384`, ~5% density, ~20 000 vars → ~130 MB per
shard plus ~80 MB indptr at 10M cells (~800 MB at 100M cells).

**Measured on `census_1m`** (1 000 000 × 61 497, 1.57 B nnz,
12.7 GB h5ad → 3.04 GB streaming SCX vs 3.11 GB materialise SCX;
median of 3 paired runs on a 16-core Lambda `standard` node):

| Path | Median wall | Median peak RSS |
|---|---:|---:|
| streaming (`pyscx.from_h5ad`), `reader_threads=4`   |  26.31 s | **2 401 MB** |
| streaming, default parallelism                      |  15.90 s |  10 352 MB |
| materialise (`pyscx.from_anndata`)                  |  30.57 s |  32 120 MB |

Re-measured 2026-09-03 (the `v0.16.0-opt-instruments` capture, median of 3 runs
per arm);
every figure is the median from the tracked JSON cited below. Streaming uses
~11× less peak RSS than materialising at the pinned thread count, and is *faster*
as well — the earlier table recorded the sequential-era 98.2 s. Both paths emit
byte-equivalent SCX (identical n_obs / n_vars / nnz / 62-shard catalog).

> **The default-parallelism arm's peak RSS moved, and it is not explained.**
> It reads 10 352 MB here against 5 893 MB in the 2026-08-22 capture, a +76%
> change while its wall *improved* (17.24 s → 15.90 s) — the shape of a
> parallelism change rather than a leak. Two cautions for whoever chases it.
> The arm records `reader_threads: None` and never stores the **resolved**
> count, so the earlier "(16 readers)" label in this table was an inference
> about that capture's node rather than a recorded fact, and has been dropped;
> the resolved value is `RAYON_NUM_THREADS` or `available_parallelism()`, which
> follows the cgroup, so it is per-allocation. And do **not** compare against
> `baselines/v0.14.0-phase5c-streaming-floors`'s `summary.json`: its 5 893 is
> `peak_rss_mb_median` pooled over all nine runs of the three arms, and with
> three equal-sized arms the pooled median is arithmetically the middle arm's,
> so it coincides with the default-threads figure and invites the comparison it
> cannot support. The floored arm (`reader_threads=4`) is stable across the two
> captures — 2 342 → 2 401 MB — so no floor catches this.
>
> **RSS is quoted in MB, deliberately.** An earlier version of this table gave GB
> figures computed as `MB / 1000` — so 5 893 MB was published as "5.79 GB" (it is
> 5.75 GiB), 26 919 MB as "26.9 GB" (26.29 GiB), and export's 13 988 MB as
> "13.98 GB" (13.66 GiB). Two of the three were also the *minimum* of the three
> runs rather than the median, and they came from an earlier capture (2834726)
> than the JSON now tracked to back them (2834737). Reporting the same unit the
> results file and `thresholds.yaml` use removes the conversion step that
> produced all of it.

> **These numbers replace an earlier table that read 98.2 s / 851 MB streaming
> and 60.2 s / 13.6 GB materialising, and the RSS figures there were wrong** —
> not stale. The benchmark reported `max(rss_before, rss_after)` of the
> *instantaneous* RSS rather than a peak, so a transient spike inside the call was
> invisible: the materialise arm's true peak is 26.9 GB, not 13.6. It also ran
> both arms in one process, so the streaming figure inherited the materialise
> arm's freed-but-unreturned heap. Both are fixed
> (`benchmarks/comprehensive/benchmarks/conversion_streaming.py` now uses
> `PeakRssSampler` and one arm per subprocess).
>
> Peak RSS scales with **reader threads**, not with file size — that is the
> bounded-memory contract. One 16384-row shard of census_1m is 351 MB
> (1402 nnz/cell × 16 B), and the reader holds
> `reader_threads + writer_queue_depth` of them. Hence the two streaming rows:
> `thresholds.yaml` gates the pinned one, because a floor whose value depends on
> the runner's core count is not a floor.

**Thread scaling on the same fixture** (median wall in seconds across
3 paired runs, `RAYON_NUM_THREADS` set per subprocess):

| Threads | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| streaming   | 87.6 | 87.8 | 88.4 | 87.6 | 87.9 | 88.9 |
| materialise | 92.2 | 72.6 | 61.8 | 58.0 | 55.4 | 54.6 |

The streaming pipeline supports a **parallel reader coordinator**
(`--reader-threads N` / `reader_threads=N`; default auto-derived from
`RAYON_NUM_THREADS` or `available_parallelism()`), which fans shard
reads across a rayon worker pool with a bounded crossbeam reorder
buffer when libhdf5 is threadsafe. Output is byte-identical to the
sequential path. The benchmark numbers above predate the parallel
reader and show the sequential-era profile — wall was flat (±1.5 %)
across the whole range. Materialise scales sub-linearly (1.69× peak
at 32 threads, efficiency 5 %) because the upstream
`anndata.read_h5ad` is single-threaded HDF5 and the rayon-parallel
encode pool runs into Amdahl's law on top of that. Materialise peak
RSS is thread-independent — measured 32 120 MB in the 2026-09-03 capture; the
~13.6 GB this line used to quote came from the pre-`PeakRssSampler` metric and
was roughly the *residual* after the call, not the peak during it.

**Crossover regime**: streaming now wins on wall *and* memory at census_1m
(26.31 s / 2 401 MB pinned, 15.90 s / 10 352 MB at default parallelism, against
30.57 s / 32 120 MB) — the parallel reader closed the wall gap the sequential-era
table above records. Materialise remains viable only while the in-memory CSR
triplet fits: census_1m's is ~27 GB resident at peak, and `census_5m` /
`census_10m` push it past most workstation memory outright. Streaming also fits the
backed-AnnData path used by `pyscx.from_anndata(adata)` when
`adata.isbacked` is true.

Source data (tracked, per
[docs/benchmark_manifest.md](benchmark_manifest.md#for-readmedocs-authors)):
`benchmarks/comprehensive/results/raw/conversion_streaming__scx_streaming_vs_materialize__census_1m.json`
— the per-arm runs quoted above live there, and only there.
`promote_baseline.py` does **not** copy `raw/*.json`, so the companion snapshot at
`benchmarks/comprehensive/results/baselines/v0.14.0-phase5c-streaming-floors/`
carries the aggregate `summary.json` (which medians all nine runs into one figure
and therefore cannot back a per-arm claim), plus `environment.json`,
`MANIFEST.sha256` and the fingerprint summary. `LATEST` is now
`v0.16.0-opt-instruments` (1833 rows / 52 benchmark families, captured
2026-09-03 at `git_sha cafeb2ce`, threads unpinned), which is where the figures
above come from. The absolute
floor is `streaming_peak_rss_mb max: 4096` on `census_1m`, declared in
`benchmarks/comprehensive/thresholds.yaml`, and it is measured on the
`reader_threads=4` arm — see the note above on why the gated arm is pinned.

> `git_dirty` is `true` on the 2026-09-03 capture too, which
> [docs/benchmark_manifest.md](benchmark_manifest.md#for-readmedocs-authors)
> permits when the dirtiness is documented — and here it is structural rather
> than incidental. The four `results/raw/*.json` files that back the numbers
> above are git-tracked (force-added past `.gitignore`, so that a fresh
> checkout can audit the claims), and a capture **overwrites them**. So any
> capture that produces these figures dirties its own tree by producing them,
> and stamps its own baseline "will not be reproducible". Nothing else was
> modified: the tracked tree was otherwise at `cafeb2ce`. Worth fixing in the
> manifest design — a results file that is both an input to the audit and an
> output of the run cannot be clean — but not by weakening the audit.

### Streaming export (SCX → h5ad / h5mu)

Streaming export closes the asymmetry with the ingestion path. `scx convert
--to h5ad` / `--to h5mu` (default `--stream=true`) and
`pyscx.to_h5ad` / `pyscx.to_h5mu` (default `stream=True`) walk SCX
CSR shards in row order via
`scx_convert::scx_to_h5ad_streaming` /
`scx_to_h5mu_streaming` / `scx_modality_to_h5ad_streaming` and write
hyperslab slices into pre-allocated `/X/{indptr,indices,data}` HDF5
datasets. Total `nnz` is summed from catalog `ShardStats` up front
(or computed via a single pre-scan decode when deletion vectors are
active) so the on-disk layout is deterministic — no extendable HDF5
datasets.

Peak RSS bound: one shard's worth of CSR per matrix written
(~`stats.nnz × 16` bytes for indices + data), plus the
always-resident kept-row indptr (`(n_obs_kept + 1) × 8` bytes — same
floor the ingestion side carries). When the source SCX file carries
`ObsMetadataShard` / `VarMetadataShard` sections, obs and var also
stream through `h5ad::column_stream::write_dataframe_group_from_shards` (pre-
allocated HDF5 datasets per column, hyperslab writes per shard,
declared-order categorical vocabularies unioned across shards) so the
bound covers metadata too — atlas-scale obs no longer materialises during
export. Multimodal h5mu export reuses the same writer per modality
so the bound applies per-modality, not cumulatively. Pass
`--stream=false` / `stream=False` to opt into the legacy materialising
path (`scx_to_h5ad` / `scx_to_h5mu`); sharded obs/var still stream
on that fallback because the alternative would re-introduce the full
obs materialisation step.

For regression coverage, see
`benchmarks/comprehensive/benchmarks/export_streaming.py` (paired
`pyscx.to_h5ad` / `pyscx.to_h5mu` with `stream=True` vs
`stream=False`, run via
`benchmarks/comprehensive/scripts/run_slurm_export_streaming.sh`).
The streaming row's `streaming_peak_rss_mb` is gated on `census_1m` in
`benchmarks/comprehensive/thresholds.yaml`, measured on the arm pinned to
`reader_threads=4`; a cross-shard accumulator leak in
`scx_to_h5ad_streaming` / `scx_to_h5mu_streaming` trips the floor without
needing a wall-clock signal.

Measured 2026-09-05 (median of 3, census_1m → h5ad, `reader_threads=4`):

| Arm | Peak RSS |
| --- | --- |
| streaming (`pyscx.to_h5ad`), `reader_threads=4` | **3 212 MB** |
| materialise (`stream=False`) | **14 048 MB** |
| streaming, `_full` fixture (`.raw` + `obsm` + a layer) | **7 547 MB** |

The materialise arm calls `read_all_csr_shards_filtered()` and holds the whole
CSR triplet, so the ~4.4× separation at the pinned thread count is the contract
working. Before the harness was fixed that arm reported 913 MB — under-stated
by 15×, which made the two arms look indistinguishable and the comparison this
benchmark exists for vacuous.

The `_full` row is the one that reaches `.raw`, `obsm` and layers, since no
source h5ad in the suite carries any of the three. It was **20 198 MB** in the
`v0.16.0-opt-instruments` capture, when `/raw` was read whole by
`write_raw_to_h5ad`; OPT-CONVERT-1 made raw stream through the same shard walk
as `/X`, taking it to 7 547 MB — 2.68×, 12.7 GB freed. The same change moved
tabula_sapiens_100k from 3 799 MB to 2 308 MB (1.65×), and left the plain arm
essentially unchanged — 1.015× / 1.010× / 1.011× across the three datasets,
which is cross-capture variance (this capture ran on a different node from the
`v0.16.0-opt-instruments` baseline), not a shift. That arm is the control:
those fixtures have no raw, so a change scoped to raw must not move them.

Both sides are backed by tracked results: the after values by
`results/raw/export_streaming__*.json` (three datasets), the before values by
`results/candidate_unpinned_20260903/raw/export_streaming__*.json`. Run counts
differ by dataset — three for census_1m and tabula_sapiens_100k, five for
pbmc3k. One caveat on the before side: that capture's recorded `git_sha`
(`3ed7861f`) was on a development branch that has since been squash-merged
away, so the *values* are auditable from the tracked JSON but the exact commit
is no longer fetchable. The after capture (`868cf6d6`) is reachable from this
PR's head.

The residual over the plain arm (4 335 MB at census, 508 MB at tabula) is not
one shard's worth, and is not fully attributed: `obsm` is still read whole
(OPT-CONVERT-4, ~208 MB at census scale), and peak RSS over a multi-matrix
export is not the max of independent peaks — buffers freed after `/X` are not
returned to the OS before the layer and raw are written.

Backed by the tracked
`results/raw/export_streaming__scx_streaming_vs_materialize__census_1m.json`.

## Column Projection (2000 HVGs)

| Dataset | SCX | h5ad (none) | Zarr (lz4) | TileDB-SOMA | SLAF |
|---------|-----|-------------|------------|-------------|------|
| Tabula Sapiens 100K | **0.55s** | 0.86s | 0.94s | 1.31s | — |
| Census 1M | **3.53s** | 33.6s | 7.24s | 10.0s | 16.8s |
| Census 5M | **9.79s** | 94.1s | 63.9s | 66.3s | — |

SCX excels at gene selection — **2x faster than Zarr**, **4.8x faster than SLAF**, and **9.6x faster than h5ad** on 1M+ cells. Column projection returns a backed/lazy dataset without materializing.

## Selective Read — Predicate Pushdown (Census 1M, SLAF)

SLAF's strongest dimension in our suite. These numbers come from the
`read_selective` benchmark's `filtered_query` scenarios and the
`benchmarks/comprehensive/queries.py` canonical predicate set.

| Predicate | SLAF (SQL WHERE) | Native mechanism |
|-----------|-----------------:|------------------|
| `cell_type == 'T cell'` | 10.5s | `slaf_sql` |
| `random 1% sample` | 9.2s | `slaf_stride_hash` |

The numbers are competitive with SCX's catalog pushdown at this scale; SLAF
pays the cost on full materialization, not on predicate-selective reads.
The `slaf_stride_hash` mechanism is a deterministic congruence-class filter
(`cell_integer_id % N == k`), i.e. every Nth cell at a fixed offset — not
Bernoulli sampling. It is the fastest obs-only scan SLAF exposes through
`SLAFArray.query`, but it is *not* comparable to the `rng.choice`-based
random-index path used by the SCX and h5ad runners for the same predicate
name; treat the `random_1pct` scenario as a different workload per format.

## Memory

Peak RSS during full read (lower is better):

| Dataset | h5ad (none) | SCX (auto) | Zarr (zstd) | SLAF |
|---------|-------------|------------|-------------|------|
| PBMC 10K | 0.48 GB | 1.51 GB | 0.76 GB | — |
| Tabula Sapiens 100K | 0.53 GB | 2.27 GB | 2.08 GB | — |
| Census 1M | 0.72 GB | 6.64 GB | 11.5 GB | 34.7 GB |
| Census 5M | 1.04 GB | 18.5 GB | 87.7 GB | — |

SLAF peak RSS includes the Polars fragment accumulator used during
`LazyAnnData.compute()`. See `benchmarks/comprehensive/results/reports/phase5A_ooc_rss.md`
for the full side-by-side table generated by `ooc_rss_table.py`.

For streaming aggregation (row_sums, col_sums), `MADV_DONTNEED` reduces SCX peak RSS by **67%** — from 3.5 GB to 1.1 GB on Census 1M.

### Out-of-Core Pipeline Memory

Full lazy pipeline (open -> QC filter -> normalize -> log1p -> HVG -> PCA -> kNN -> UMAP -> Leiden) on 1M cells:

| | Full materialization | SCX lazy pipeline | Reduction |
|--|----------------------|-------------------|-----------:|
| Lazy preprocess only (normalize + log1p) | — | **3.5 GB** | — |
| Full pipeline through Leiden | 22.3 GB | **10.9 GB** | **51%** |

The lazy preprocessing peak (~3.5 GB) covers QC through streaming PCA; kNN graph
construction and UMAP dominate the remaining RSS in the full pipeline (~10.9 GB).
Source: `benchmarks/comprehensive/results/reports/phase5A_ooc_rss.md` and `benchmarks/comprehensive/results/raw/memory__scx_auto__census_1m.json`.

### Read iteration: streaming vs in-memory

`pyscx.open(path).to_anndata(backed=True)` and `to_mudata(backed=True)` walk
shards on demand; the eager `to_anndata()` / `to_mudata()` materialise the
full X up front. Direct head-to-head on a row-by-row iteration workload
(``read_streaming_vs_inmemory`` benchmark, 65 536-row chunks, ``chunk.sum()``
per chunk, **cold cache** via ``posix_fadvise(POSIX_FADV_DONTNEED)`` between
runs; Lambda HPC ``preemptible`` partition, single-modality SCX (auto)):

| Dataset | SCX on disk | Streaming wall | In-memory wall | Streaming peak RSS | In-memory peak RSS | Wall ratio | RSS savings |
|---|---:|---:|---:|---:|---:|---:|---:|
| census_500k (500K × 61K) | 1.56 GB | 48.5 s | 50.5 s | 3.2 GB | 7.9 GB | **0.96× (streaming faster)** | **2.4×** |
| census_1m (1M × 61K) | 3.13 GB | 94.4 s | 99.0 s | 5.2 GB | 15.6 GB | **0.95× (streaming faster)** | **3.0×** |

Two takeaways:

- Under cold cache, streaming actually edges out in-memory by ~4-5 % on
  wall time. Streaming interleaves shard I/O with decode; the in-memory
  path reads everything before iteration begins, and at cold cache that
  serialise-then-iterate pattern doesn't amortise. (Warm-cache repeats —
  pre-`POSIX_FADV_DONTNEED` — invert the ratio to ~1.23× *slower* for
  streaming, since the in-memory path can re-read from RAM. Production
  workloads on census-scale files don't fit in RAM, so cold cache is the
  load-bearing regime.)
- Streaming peak RSS scales with one shard's footprint
  (``chunk_size × n_vars × ~16`` bytes plus indptr); in-memory scales
  with the full CSR. Both modes produce identical per-chunk matrix sums
  (correctness gate in the benchmark module). The RSS ratio grows
  linearly with dataset size — extrapolating from these two points,
  ``census_5m`` (5M cells × 61K vars) lands at roughly 6-10× and
  ``census_10m`` pushes the in-memory path past the headroom of typical
  workstation hardware.

**Multimodal** counterpart (``multimodal_read_streaming_vs_inmemory``)
routes through ``to_mudata(backed=True)`` and iterates each modality. On
the available real fixtures the in-memory ceiling fits comfortably in
node RAM, so RSS is baseline-bound and the two modes look identical;
the streaming read path is verified correct via matching per-modality
matrix sums across both modes:

| Dataset | Modalities | SCX size | Streaming wall | In-memory wall | Streaming peak RSS | In-memory peak RSS |
|---|---|---:|---:|---:|---:|---:|
| cite_seq_pbmc_5k (5.2K × 33K + 32) | rna + adt | 16 MB | 0.61 s | 0.59 s | 312 MB | 315 MB |
| multiome_pbmc_10k (11.9K × 36K + 144K) | rna + atac | 228 MB | 25.4 s | 24.5 s | 1.58 GB | 1.59 GB |

The RSS-saving regime activates at census-scale CITE-seq / Multiome
(not staged on the current Lambda fleet); the pattern at single-modality
``census_500k`` / ``census_1m`` is the load-bearing extrapolation.

#### Per-modality in-decode narrow (eager `to_mudata`)

`to_mudata` narrows each modality's `X` **in-decode** to a caller-chosen dtype
(`data_dtype={"rna": "uint16", "atac": "uint16"}`, scalar or per-modality dict),
assembling directly at the target width via `read_all_csr_shards_for_typed` —
never building the intermediate f32 CSR. The returned MuData's value-buffer
footprint drops deterministically: on `multiome_pbmc_10k` (RNA 23.5M nnz + ATAC
104M nnz), the two `X` value buffers go from **510 MB f32 → 255 MB uint16 (2×)**,
lossless (RNA max 5 585, ATAC max 762 both fit `uint16`). Whole-process peak RSS
for the eager `to_mudata()` call, `/usr/bin/time -v`, multi-shard file
(2 048-row shards, 6 shards/modality):

| `to_mudata` call | peak RSS |
|---|---:|
| default (all f32) | **1.98 GB** |
| `data_dtype={"rna":"uint16","atac":"uint16"}` | **1.52 GB** |

≈0.46 GB (~24%) lower, driven by the narrower value buffers *and* the in-decode
path avoiding the f32 assembly/concat transients. The process-level win scales
with **shard count**: on a single-shard-per-modality file (the default 16 384-row
shards → 1 shard each here) the whole ATAC `u32` stream is resident during the
cast, so the transient advantage collapses (~44 MB delta) even though the
returned buffers still halve — atlas-scale multimodal files (many shards) sit in
the multi-shard regime. A modality that binarizes / stays shallow (`uint8`, e.g.
peak-called ATAC or small ADT panels) reaches **4×** on the value buffer; the
available Multiome fixture's ATAC counts exceed 255 (max 762), so `uint8` there
needs `allow_lossy=True` and is not lossless. Default (no override) modalities
keep the byte-identical zero-copy f32 path. A dict key naming no modality raises
`ValueError`; `container="dense"` is not yet supported for `to_mudata`.

Source: ``benchmarks/comprehensive/results/raw/read_streaming_vs_inmemory__scx_auto__{census_500k,census_1m}.json``
and ``benchmarks/comprehensive/results/raw/multimodal_read_streaming_vs_inmemory__scx_multimodal_per_modality_auto__{cite_seq_pbmc_5k,multiome_pbmc_10k}.json``.
SLURM wrapper: ``benchmarks/comprehensive/scripts/slurm_read_streaming_vs_inmemory.sh``.

## Analysis Accelerators (CPU)

Benchmarked on 1M cells (CELLxGENE Census), HVG-selected (2000 genes):

| Operation | SCX (s) | scanpy (s) | Speedup vs scanpy |
|-----------|---------|------------|-------------------|
| PCA (covariance, 50 PCs, 2K HVGs) | **4.2** | 8.0 | **1.9x** |
| Wilcoxon rank-sum DE (pre-ranking) | **5.4** | 17.3 | **3.2x** |
| Leiden (Rust-native) | **55** | 2,226 (leidenalg) | **40x** |

Full pipeline (PCA -> kNN -> UMAP -> Leiden -> DE) on 1M cells: **870s** (vs 3,971s — **4.6x faster**).

### CPU stage profile — the decode/reduction ranking oracle (Phase-2 task 2.0)

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

### Bounded ordered decode-prefetch (Phase-2 task 2.1)

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

### Decode-prefetch beyond `scx-accel` (Phase-4 task 4.2)

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

### Decode-prefetch reaches the DE kernels

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
kernel — and [docs/benchmark_manifest.md](benchmark_manifest.md) is explicit
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

### GPU DE device residency + gene-chunk windowing (Phase-4 task 4.5)

4.2 widened GPU staging's decode; it did not reduce how much decoding there was.
The GPU CSR DE route (`gpu_csr_v3` — the mandatory route for any file **without** a
CSC sidecar) has no column-range prefilter, so each of the two v3 CSR drivers runs a
full `for_each_gpu_csr_shard` pass **per gene chunk**. Cost is
`n_gene_chunks × n_shards` host decodes and H→D uploads. At census_500k — 61 497 genes
over a 500-gene chunk is 123 chunks, across 31 CSR shards — that is 123 complete passes
over a 747 M-nnz matrix. Pre-4.2, with staging decoding on one thread, that made GPU DE
there **98.6 % host-decode-bound** (1 005.6 s of a 1 019.7 s wall).

The arithmetic closes exactly, which is what made the diagnosis actionable rather than
plausible: 1 005.6 s ÷ 123 chunks = **8.2 s**, one full decode pass. (That prediction is
what the 4.5 capture below then hit, at 8 578 ms.)

**Decode was only half of it.** All four CSR row-scan kernels in `diffexp.cu` are
one-block-per-row and stride the row's *entire* nonzero range, testing
`col >= c0 && col < c1` per element — so the per-chunk *kernel* cost was O(nnz) too,
another 123× over. 4.2 widened the decode but left that untouched: bounding its arm from
wall (383.9 s as measured below) and Σ host-decode (≈ 941 s summed over depth-4 workers)
puts kernels + sync somewhere in **[84, 319] s**. Residency alone could not have been
shown to fix that, so both halves ship together.

Note the two baselines in play. Everything above quoting 1 019.7 s is **pre-4.2**
(`2055f74f`); the capture below is against **post-4.2** `main` (`2d1fe16b`), which is the
383.9 s arm. 4.2's 2.60× on this op is already banked in that baseline and is not counted
again here.

**1. Device residency.** `scx_gpu::ResidentGpuCsrSource` drains the inner
`GpuMatrixSource` once, retains every shard in its own device-resident `GpuCsrSlot`, and
serves each later chunk from VRAM. Decode and H→D collapse from `n_chunks × n_shards` to
`n_shards`.

Shards are **retained separately, not concatenated**. Two builders in `scx-gpu` already
concatenate (`decode_csr_shards_to_device` for the `to_gpu_anndata` handoff,
`gpu_pca_resident::try_build_resident_csr` for the PCA power loop) because their consumers
need one cuSPARSE descriptor spanning the matrix. DE does not — its kernels take a
per-shard view plus a `global_row` offset. Collapsing 31 shards into one 500 000-row shard
would change every kernel's grid shape and, for the f64 `atomicAdd` pseudobulk fold, the
accumulation interleaving. Keeping them separate means the callback sees byte-for-byte
what it saw while streaming: same shard indices, shapes, launch geometry, arguments. Total
VRAM is the same either way (~6 GB at census_500k, 8 B/nnz).

**2. Gene-chunk windowing.** Every one of those kernels already requires
strictly-increasing per-row column indices — `shard_validate::validate_shard` enforces it
release-active, because a duplicate column races the scatter. Sorted indices make a
chunk's columns a contiguous sub-range of the row, so two `lower_bound` searches replace
the linear scan and the per-chunk term becomes `O(nnz / n_chunks + log(row_len))`.

`[lower_bound(c0), lower_bound(c1))` selects exactly the elements the predicate selected.
For the three scatter kernels each output cell has a single writer, so this is
bit-identical. `csr_shard_pseudobulk_kernel` folds with f64 `atomicAdd`, whose ordering
across rows is **already** run-to-run nondeterministic; narrowing the loop changes the
interleaving but not the character, and the equivalence test compares means and fold
changes to tolerance while holding statistics and p-values exact.

**What residency costs, per tier** (8 B/nnz — one f32 value + one i32 index; all three fit
inside half an 80 GB H100 many times over, and all three run 123 gene chunks at the default
500-gene chunk over 61 497 genes):

| Dataset | nnz | CSR shards | resident VRAM | shard decodes before → after |
|---|--:|--:|--:|--:|
| tabula_sapiens_100k | 194.9 M | 7 | ~1.6 GB | 861 → 7 |
| census_500k | 747.0 M | 31 | ~6.0 GB | 3 813 → 31 |
| census_1m | 1 402.4 M | 62 | ~11.2 GB | 7 626 → 62 |

**Knobs.** `SCX_GPU_DE_RESIDENT_MAX_FRAC` (default `0.5`) caps residency at half the free
card, leaving the rest for the per-chunk gene slabs; `SCX_GPU_DE_RESIDENT=0` is the kill
switch. Residency is declined for a single gene chunk (streaming would run one pass
anyway) and when the matrix does not fit the budget — checked before every shard, aborting
the drain at the offending one rather than retaining more than it checked for. If the
per-chunk budget then cannot fit *alongside* the resident matrix, residency is released
and the clamp retried: an optimisation must never be the reason a call errors.

**A declined run is invisible in the output** — it produces the same numbers, slowly. So
the decision is stamped on `uns["scx_accel"][<op>]["resident_csr"]` and floored by the
`de_route_resident_csr` gate, the same reasoning behind `de_route_csc_direct`.
`shards_decoded` is *not* the signal: it counts slab passes, which residency does not
change.

**Also in 4.5.** The staging validator's O(nnz) scans go parallel above 65 536
nnz — they were amortised into irrelevance when the same shard was re-validated 123 times
beside 123 re-decodes, and are on the critical path once each shard is decoded once. The
parallel form reduces by *minimum row index* rather than first-hit, so the error names the
same offending position it always did rather than one that varies with load. And
`ShardSource::shard_size_hint()` (catalog-backed, no decode) finally lets GPU staging
pre-size its pinned/device slots — `RawGpuShardSource::with_max_shard_rows` had been dead
code — and gives `clamp_prefetch_depth` a real per-shard byte estimate, so
`SCX_GPU_STAGING_MEMORY_BUDGET` can bound the decoded-but-unconsumed set. Unset, nothing
derates and the depth is unchanged.

**Measured** (SLURM job 2709095, H100, **two builds** — `main` at `2d1fe16b` (i.e. post-4.2)
vs the branch, both at their own defaults, `benchmarks/scripts/profile_gpu_de_resident.py`,
3 runs, median wall, `SCX_DISABLE_CUDA_GRAPHS=1`):

| Op | Dataset | `main` (4.2) | branch | Speedup | Σ host-decode | pinned HTOD | VRAM peak |
|---|---|--:|--:|--:|--:|--:|--:|
| pdex_ref | tabula_sapiens_100k | 114 902 ms | 3 154 ms | **36.4×** | 246 868 → 2 105 ms | 17 313 → 203 ms | 1 775 → 2 984 MB |
| wilcoxon | tabula_sapiens_100k | 113 185 ms | 3 661 ms | **30.9×** | 247 348 → 2 120 ms | 17 351 → 200 ms | 2 127 → 3 334 MB |
| pdex_ref | census_500k | 383 890 ms | 12 536 ms | **30.6×** | 941 404 → 8 578 ms | 64 656 → 641 ms | 3 658 → 9 128 MB |
| wilcoxon | census_500k | 385 877 ms | 14 064 ms | **27.4×** | 945 281 → 8 146 ms | 64 477 → 631 ms | 5 162 → 10 632 MB |
| hvg *(control)* | 100k / 500k / 1m | 4 409 / 10 559 / 16 030 ms | 4 447 / 11 232 / 17 255 ms | 0.99 / 0.94 / 0.93× | — | — | 1 192 → 1 192 MB |

**The mechanism is confirmed three ways, not just by the wall.** At census_500k `pdex_ref`
the summed host-decode falls **110×** (predicted 123×: one pass instead of one per gene
chunk) and lands on **8 578 ms** against the 8 200 ms predicted *before the run* from
1 005.6 s ÷ 123. The pinned-staging bucket falls **101×**. And the VRAM delta between arms
is **5 470 MB** against a predicted 5 976 MB of resident CSR. Host peak RSS *falls* 14 %
(3 643 → 3 127 MB) — one decode pass churns far less host memory than 123.

**Residency alone would not have done this.** Its own predicted range was 1.2–4.4× (the
[84, 319] s kernel bound above). Post-change, kernels + sync are ~10 s of the 12.5 s wall,
so the windowing cut the per-chunk scan by roughly 8–32×. The capture measures the two
**together** and cannot attribute between them: there is a knob to disable residency but
none to disable the windowing, and adding one was not judged worth the API surface.

> [!NOTE]
> **The hvg control's 0.93–0.99× is noise, and that was measured rather than assumed.**
> Consistently-below-1.0 across three datasets looked like a real cost — plausibly the
> parallel shard validation, which runs on the consuming thread and so competes with the
> prefetch workers on a decode-bound op. Job 2709123 tested exactly that with one build and
> two arms (`SCX_GPU_VALIDATE_PAR_MIN_NNZ` pinned high takes the unchanged serial branch),
> 5 runs each. The result scattered in **both** directions — 1.037× / 1.001× / 0.944× on
> hvg, 0.977×–1.014× on DE — i.e. no effect. The job-to-job spread is the explanation: the
> *same* branch build measured census_500k hvg at 11 232, 12 822 and 12 839 ms across two
> jobs on the same node, a 14 % swing that swallows the 7 % being chased. Single-job control
> deltas below ~15 % on this node are not interpretable.

**Output equivalence is checked at the scale the change was built for**, not only on unit
fixtures (SLURM 2709125, census_500k, `SCX_GPU_DE_RESIDENT=0` vs default, same file and
groups, both arms confirmed on `gpu_csr_v3` at 123 gene chunks):

| Op | names / feature | p-values | statistics | pseudobulk means / log2FC |
|---|---|---|---|---|
| `rank_genes_groups` | exact | exact | exact | **exact** (streaming self-spread also 0) |
| `pdex_ref` | exact | exact | exact | ≤ 1.7 × 10⁻¹³ |

`pdex_ref`'s means are the only figures that are not bit-identical, and the bar they are
judged against is measured rather than chosen: the streaming path was run **twice**, and
its own run-to-run spread (1.0 × 10⁻¹³ — f64 `atomicAdd` ordering across rows is already
nondeterministic) is what residency has to come in under. It does, at the same order of
magnitude and ~5 decades inside the 2.9 × 10⁻⁸ relative floor. Wilcoxon's pseudobulk fold
happened to be reproducible on this run, and residency matched it exactly.

**Parallel shard validation (§9.13) is a measured no-op at these scales, and is kept as
hygiene with no `×` claimed** — the same disposition as task 4.3's marshalling. The reason
it does not show up is worth stating: validation runs on the consumer thread *while* the
prefetch workers decode ahead, so it is hidden behind decode entirely. What actually made
validation cheap was residency, which cut it from once-per-shard-per-chunk to once per
shard — 123× fewer invocations. Parallelising what remains is correct and free, not a win.

### Low-risk marshalling & fusions (Phase-2 tasks 2.4 + 2.7)

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

### Marshalling & GIL hygiene at the Python boundary (Phase-4 task 4.3)

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
  a new owned accessor (`as_column_source_owned`) under the identical deletion-vector gate.

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

### Owned snapshots at the numpy boundary (finding §10.1)

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

### Graph-layout refactors (Phase-2 task 2.5)

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

### UMAP determinism + invariant hoist (Phase-2 task 2.6)

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

### PCA streaming decode-prefetch (Phase-4)

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
[docs/scanpy.md § PCA reproducibility](scanpy.md#reproducibility). The rewrite was also
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

### PCA reduction partitioning (covariance route)

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

### Pseudobulk aggregation partitioning (OPT-ACCEL-4)

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

### QC / filtering pass fusion (Phase-4 task 4.1)

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

### Axis subsetting — what delegating to anndata costs (Phase-4 task 4.0b)

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
[docs/benchmark_manifest.md](benchmark_manifest.md).

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

### Differential expression (CPU, full-matrix)

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
file has a valid CSC sidecar (no active deletion vector, column-local transforms),
else the CSR streamer; on GPU they stay CSR so the planner can route `gpu_csc_v3`.
The route + `csc_available` flag are recorded on `adata.uns["scx_accel"][<op>]`.
This is a compatibility change (DE previously defaulted to `"csr"`); pin
`prefer_format="csr"` for the old behaviour. An **exact sparse-nnz Wilcoxon**
kernel (opt-in `SCX_ACCEL_WILCOXON_NNZ=1`, 1-vs-rest) ranks only each gene's
nonzeros plus an analytic implicit-zero tie-block — `O(nnz·log nnz)`/gene instead
of an `O(n_obs·log n_obs)` dense sort — numerically equivalent to the dense kernel
(property-tested to 1e-9).

**Measured DE-route benchmark** (`rank_genes_groups`, 1-vs-rest, backed streaming;
a CSC sidecar built with `scx build-csc`; median of 2 runs, CPU). Routes confirmed
via `uns["scx_accel"]` (`cpu_csr` / `cpu_csc`):

| Dataset | Route | wall (s) | peak RSS (MB) | vs CSR | vs CSC-densify |
|---|---|--:|--:|--:|--:|
| tabula_100k (100K × 61.5K, 33 groups) | CSR streaming (`csr`) | 351.8 | 3454 | 1.0× | — |
| tabula_100k | CSC-direct densify (`csc`, §5.2) | 25.6 | 2749 | **13.7×** | 1.0× |
| tabula_100k | CSC-direct nnz (`csc`+`SCX_ACCEL_WILCOXON_NNZ`, §5.3) | 16.2 | 2552 | **21.7×** | **1.58×** |

**§5.2 — CSR → CSC-direct is ~13.7× faster with lower peak RSS.** The CSR streamer
re-decodes every shard for each gene-chunk (here ~123 chunks × 7 shards on the
full 61.5K-gene matrix, cache-bound), while the CSC-direct route reads each
column-chunk exactly once — so on a sidecar file the `auto` default's CPU routing
is a large, measured win. (For CSR-*only* files the deferred loop-inversion / a
larger shard cache addresses the same re-decode; `auto` sidesteps it when a sidecar
exists.) **§5.3 — the exact sparse-nnz kernel adds ~1.58×** over CSC-densify by
ranking only nonzeros + an analytic zero block instead of an `n_obs` dense sort,
for **~21.7× end-to-end** over the old CSR default, at lower peak RSS. Numerically
identical to the dense kernel (property-tested). Source: `bench_de_csc_routes.py`
on `cpu_preemptible`; the nnz kernel stays opt-in pending a promotion decision.

Source: 2026-05-25 full-tier gate (post-G10 graph capture + bench env-routing fix), candidate `candidate_2623788_20260525`. Benchmark module: `benchmarks/comprehensive/benchmarks/accel_de.py` — picks the best obs column from `cell_type`/`leiden`/`louvain`/`cluster`/`perturbation`/`target` or falls back to a deterministic 50/50 synthetic split, restricts to top-4 test groups + reference, and records the chosen `groupby` in `metadata`. Each SLURM bench job is allocated 16 CPUs; `pyscx_cpu`'s `user_s/wall_s` ratio shows ~3-5 effective cores per run.

### Harmony2 batch integration + LISI

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

#### Extrapolated capacity (500 GB / 1000 GB RAM)

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

## Perturbation Metrics (cell-eval / arc-bench parity)

Rust-accelerated perturbation evaluation metrics exposed via `pyscx.accel.*` are numerically equivalent to the Python reference implementations in `cell-eval` (v0.7) and `arc-bench` (32/32 parity tests pass within the tolerances documented in [`docs/scanpy.md`](scanpy.md#perturbation-evaluation-metrics-cell-eval--arc-bench-parity)). Wall-clock speedup vs the Python reference on synthetic perturbation datasets (N cells × 2K genes × 50 perturbations, 3 runs median, reference reconstructs a cold `PerturbationAnndataPair` per op for fair comparison):

| Operation | 10K | 20K ⁴ | 100K | 500K | 1M |
|-----------|----:|-----:|-----:|-----:|----:|
| Pseudobulk means | 7.8x | 11.8x | **11.6x** | **13.8x** | **19.4x** |
| Bulk metrics (pearson_delta + mse + mae + mse_delta + mae_delta, bundled) | 9.1x | 10.5x | **12.1x** | **13.6x** | **21.9x** |
| Discrimination score (L1) | 8.1x | 11.8x | **12.0x** | **12.9x** | **20.1x** |
| Energy distance (gemm + f32, default) | 30–40x ² | **52.1x** | not yet captured ² | skipped¹ | skipped¹ |
| Energy distance (gemm + f64) | ~25–35x ² | **33.0x** | not yet captured ² | skipped¹ | skipped¹ |
| Energy distance (scalar + f64, legacy alias) | 13.4x | 10.2x | **14.4x** | skipped¹ | skipped¹ |
| Clustering agreement (AMI, native Rust Leiden) | 4–5x ³ | 3.0x ³ | **10–13x** ³ | **24.6x** | **10.0x** |
| Knockdown efficiency + log deviation | 0.6x | 1.4x | 0.9x | **1.3x** | 0.7x |

¹ The cell-eval reference's `sklearn.metrics.pairwise_distances` path allocates an O(N²) distance matrix per perturbation and runs ~18 s/pert × 49 perts at 100K already (941 s/run observed); ≥ 500K would take hours for the reference alone. SCX's fused-gemm Rust kernel remains feasible at 1M+ — kernel-level scaling is tracked by the standalone criterion microbench at `scx-accel/benches/distances.rs`.

² `pyscx.accel.energy_distance` exposes `backend ∈ {"scalar", "gemm"}` and `dtype ∈ {"f32", "f64"}` kwargs. Default is `backend="auto"` (gemm for euclidean / cosine, scalar for L1) and `dtype="f32"`. The four combinations are reported as separate ops in `cell_eval_parity_perf.py`; the legacy `energy_distance` op alias preserves the `scalar + f64` (slowest) numbers for back-compat with historical baselines. Stand-alone matmul-vs-scalar speedup at 102K × 2K × 50 is 3.72× (scalar f64: 121.2 s vs gemm f64: 32.6 s); f32 vs f64 at 204K × 1K × 50 is 2.24×. Combined, the headline `gemm + f32` cuts ~7 s of cell-eval-side reference wall to a few hundred ms of SCX-side wall — speedup ratio is reference-bound, so the absolute SCX time is the more useful number for scaling decisions.

³ The scanpy `pp.neighbors` + `tl.leiden` path inside `clustering_agreement` was replaced with native-Rust `scx_accel::neighbors::build_knn_graph` + `scx_accel::leiden`, runnable under `py.allow_threads`. End-to-end on a synthetic n_perts=200 (10K cells × 300 genes), SCX takes 229 ms vs 2942 ms for the cell-eval scanpy reference (12.83× speedup; AMI score within 0.019 of the reference at `atol=0.15`). Speedup ratio varies with the centroid graph's modular structure — at small n_perts the Rust-native Leiden's RB-modularity tie-break can pick a different number of communities than scanpy's `flavor="igraph"`; the parity test was bumped from `n_perts=8 → 30` because at n_perts ≥ 16 the algorithms agree exactly on the test scaffolding. The 3.0× number at 20K cells × 50 perts is dominated by Leiden iteration count on a 49-node centroid graph; speedup grows with both centroid count and per-centroid embedding dimension.

⁴ The 20K column was captured on 2026-04-27 with the native-Rust `clustering_agreement` code path; the 100K / 500K / 1M columns are earlier measurements preserved as historical baselines for back-compat trending. New `energy_distance_*` ops are exercised at the 20K size since the cell-eval reference's O(N²) work makes the larger sizes infeasible for it (see footnote ¹). Re-running the comprehensive parity-perf suite at 100K–1M with the gemm + f32 default is queued as a follow-up SLURM job.

### Pairwise-distance kernels: Gram blocking and the `with_min_len` fix

Captured 2026-08-10 on Chimera `cpu`-partition nodes via
`cargo bench -p scx-accel --bench distances` (criterion median), plus one end-to-end
`/usr/bin/time -v` run. `d2k` = 2000 dims. `before` is `db537021`.

> [!NOTE]
> These are **Criterion microbenchmark** medians and a single `/usr/bin/time -v`
> run, with **no manifest entry** under `benchmarks/comprehensive/results/`. They
> are the second tier of
> [`docs/benchmark_manifest.md` § Scope](benchmark_manifest.md#scope-which-claims-this-covers):
> a `benchmark`/`format`/`dataset` triple is not a shape a `cargo bench` kernel id
> can take, so the command, node, date and commit above stand in for one.
> `benchmarks/scripts/check_readme_manifests.py` enforces the first tier over
> `README.md` only and does not parse this file.

**Restoring parallelism** (16-CPU node). All four `with_min_len` calls in
`eval_metrics/distances.rs` exceeded their iterator length, so rayon never split them —
`(0..n).with_min_len(n * 16)` on the scalar self path is unsatisfiable for every `n`. Those
loops ran single-threaded at any pool size:

| Benchmark | before | after | |
|---|---:|---:|---:|
| `mean_pairwise_distance/f32/2000x2000/d2k/euclidean/scalar` | 6.970 s | 873.8 ms | **7.98×** |
| `mean_pairwise_distance_self/f32/2000/d2k/euclidean/scalar` | 3.481 s | 436.3 ms | **7.98×** |
| `mean_pairwise_distance_self/f32/2000/d2k/cosine/scalar` | 5.388 s | 665.8 ms | **8.09×** |
| `mean_pairwise_distance_self/f64/2000/d2k/euclidean/scalar` | 2.558 s | 174.3 ms | **14.67×** |
| `mean_pairwise_distance_self/f64/2000/d2k/cosine/scalar` | 2.619 s | 280.0 ms | **9.35×** |
| `mean_pairwise_distance/f32/2000x2000/d2k/euclidean/gemm` | 26.77 ms | 19.36 ms | **1.38×** |
| `mean_pairwise_distance_self/f32/2000/d2k/euclidean/gemm` | 26.79 ms | 19.07 ms | **1.40×** |
| `mean_pairwise_distance_self/f64/2000/d2k/cosine/gemm` | 47.25 ms | 41.58 ms | **1.14×** |

The gemm rows are the same fix applied to the Gram-expansion loop, **not** the upper-triangle
change: the cross benchmark, which has no triangle, gains the same 1.2–1.4×.

**What blocking costs, and what the triangle buys** (32-CPU node, same shape both arms,
`2000 × 2000 × d2k` f32 euclidean gemm). Every shape in the bench grid is a single block at the
256 MiB default — the grid tops out at 200 MB — so the triangle's effect is invisible there and
had to be measured against a forced 2 MB budget (8 blocks):

| | 1 block (default) | 8 blocks (2 MB budget) | cost of blocking |
|---|---:|---:|---:|
| full square (`mean_pairwise_distance(a, b)`) | 13.28 ms | 26.79 ms | 2.02× |
| upper triangle (`…_self(a)`) | 12.96 ms | 19.46 ms | 1.50× |
| **triangle vs. square** | 1.03× | **1.38×** | |

At one block the triangle narrows only the expansion loop — the columns operand is still all of
`b` — so it is worth ~3 %. Once blocked, later blocks take a narrower `b`, total gemm work drops
to `≈ (k+1)/2k` of the square at `k` blocks, and the triangle is what makes blocking nearly free
on the self path (1.50× vs 2.02×).

**End to end** (`pyscx.accel.energy_distance`, 30 K control + 20 × 500 perturbation cells × 50
dims, `RAYON_NUM_THREADS=16`). The A/B lever is the budget knob itself: a budget larger than the
whole Gram reproduces the pre-fix single allocation, so both arms are the same binary.

| | peak RSS | kernel wall |
|---|---:|---:|
| one block (budget ≫ Gram) | 3.85 GB | 3.9 s |
| default 256 MiB budget | **1.76 GB** | **1.0 s** |

Identical correlation to 9 decimal places. Blocking is *faster* here, the opposite of the 2000-dim
microbench: at 50 dims the gemm is memory-bound, and a block that survives in cache between the
matmul and the expansion beats streaming a 3.6 GB Gram through DRAM. So the throughput cost of
blocking is real but `n_dims`-dependent — it bites on raw-gene inputs and pays on the
`embed_key="X_pca"` inputs the metric is normally run on.

Note that 1.76 GB is far above the 256 MiB budget, and that is expected rather than a miss: the
budget bounds **one block**, and `energy_distance` runs perturbations on a rayon `par_iter`, so
the Gram term is `RAYON_NUM_THREADS × budget` on top of each task's own dense copy of its group's
rows (`extract_group_rows_indexed`, untouched here).

Speedups grow with cell count for the pseudobulk-driven metrics (pseudobulk, bulk_metrics, discrimination_l1) — single-pass streaming aggregation in Rust wins harder as the per-cell work scales. `knockdown_efficiency` is within ±40% of arc-bench's tight NumPy column-access loop and is not currently a speedup target.

The table above is the SCX-Rust-vs-Python-reference speedup on the **CPU**. `perturbation_metrics` and `energy_distance` (euclidean / cosine) also accept `device="gpu"` (pyscx ≥ 0.11.2) — CPU-vs-GPU wall times are in [GPU perturbation-evaluation metrics](#gpu-perturbation-evaluation-metrics) below.

Full per-operation results (wall time + peak RSS) are tracked in `benchmarks/comprehensive/results/raw/cell_eval_parity_perf__scx_auto__pert_synth_*.json` and rendered in the "Cell-eval / arc-bench Parity Performance" section of the comprehensive benchmark report. Kernel-level distance-kernel microbenchmarks live in `scx-accel/benches/distances.rs` (run via `cargo bench -p scx-accel --bench distances`; see [`benchmarks/README.md`](../benchmarks/README.md#rust-microbenchmarks-criterion)).

## GPU Acceleration (NVIDIA H100)

### Codec Decode and Training Pipeline

| Operation | Size | CPU (us) | GPU (us) | Speedup |
|-----------|------|----------|----------|---------|
| FOR-BP index decode | 16K rows, 33M nnz | 102,900 | 4,133 | **24.9x** |
| Sparse -> dense | 16K rows x 30K cols | 433,252 | 7,711 | **56.2x** |
| Sparse -> dense (HVG 2K) | 16K rows x 2K output | 110,416 | 897 | **123.1x** |

### GPU Analysis Pipeline

GPU-accelerated analysis via **rapids-singlecell** (`rsc.pp.pca`, `rsc.pp.neighbors`, `rsc.tl.umap`, `rsc.pp.*`) for in-VRAM ops, plus native Rust/CUDA paths for streaming PCA, HVG `seurat_v3`, Leiden (Rust-native CPU + cuGraph GPU), DE Wilcoxon rank-sum/pdex (CSC/CSR-direct), Harmony, and codec decode. Benchmarked on H100 80GB (driver 560.35.05, CUDA 12.6, scx-bench-gpu conda env).

**GPU VRAM usage.** `to_gpu_anndata()` preserves sparse CSR on the GPU — VRAM
for `X` scales with NNZ, not N×M. A 1M-cell × 2K-gene HVG-selected matrix at
5% density occupies ~800 MB as sparse CSR (vs ~8 GB dense). Peak VRAM during
an operation also includes per-op working memory (PCA dense working matrices,
kNN embeddings, DE per-chunk intermediates). rapids-singlecell ops may allocate
additional dense working buffers internally. Use
`pyscx.accel.estimate_gpu_memory(adata, operation=...)` for pre-flight sizing;
`to_gpu_anndata()` includes a VRAM pre-flight guard (1.2× headroom) that raises
`ValueError` if insufficient. See
[gpu-setup.md § GPU memory model](gpu-setup.md#gpu-memory-model) for the full
sizing model. Formal per-op peak-VRAM benchmarks are planned.

Numbers below are from the full-tier gate run on 2026-05-25 (post-G10 graph capture for GPU DE + bench env-routing fix). The per-job conda-env routing fix (`run_parallel.py::_env_for_format`) unlocked real GPU coverage for Leiden + kNN that prior baselines silently missed (workers were running on the orchestrator's env which lacked cuGraph + cuVS — see `benchmarks/README.md` § Environment notes for the routing details).

#### Per-operation timing

| Operation | Dataset | CPU (s) | GPU (s) | Speedup | Backend |
|-----------|---------|---------|---------|---------|---------|
| PCA (50 PCs, auto-routed) | tabula_sapiens_100k | 2.79 (`pyscx_cpu_auto`) | 0.38 | **7.3×** | rapids `rsc.pp.pca` (in-VRAM) |
| PCA (50 PCs, auto-routed) | census_500k | 1.89 | 0.79 | 2.4× | rapids `rsc.pp.pca` (in-VRAM) |
| PCA (50 PCs, auto-routed) | census_1m | 2.77 | 1.56 | 1.8× | rapids `rsc.pp.pca` (in-VRAM) |
| PCA correctness (cos sim vs scanpy, top-50) | pbmc3k | — | — | **min=0.999911** | — |
| PCA correctness (cos sim vs scanpy, top-50) | census_1m | — | — | **min=1.0** | — |
| kNN (k=15, 50 PCs) | tabula_sapiens_100k | 5.68 (`scanpy_cpu`) | 4.28 | 1.3× | rapids `rsc.pp.neighbors` |
| kNN (k=15, 50 PCs) | census_500k | 36.36 | 12.75 | **2.9×** | rapids `rsc.pp.neighbors` |
| kNN (k=15, 50 PCs) | census_1m | 91.97 | 26.67 | **3.4×** | rapids `rsc.pp.neighbors` |
| UMAP (2D) | tabula_sapiens_100k | 50.82 (`scanpy_cpu`) | 2.58 | **20×** | rapids `rsc.tl.umap` |
| UMAP (2D) | census_500k | 364.51 | 10.61 | **34×** | rapids `rsc.tl.umap` |
| UMAP (2D) | census_1m | 846.89 | 27.91 | **30×** | rapids `rsc.tl.umap` |
| UMAP trustworthiness | pbmc3k | 0.9238 | 0.9233 | — | vs PCA space |
| Leiden (`device="cpu"`, Rust-native) | tabula_sapiens_100k | 3.41 | — | — | `scx_accel::leiden` |
| Leiden (`device="gpu"`, cuGraph) | tabula_sapiens_100k | 100.73 (`leidenalg_cpu`) | 0.54 | **187×** | cuGraph |
| Leiden (`device="gpu"`) | census_500k | 838.94 (`leidenalg_cpu`) | 1.59 | **528×** | cuGraph |
| Leiden (`device="gpu"`) | census_1m | 659.0 (`leidenalg_cpu`, prior baseline) | 3.06 | 215× | cuGraph |
| Wilcoxon rank-sum (vs `pyscx_cpu` reference) | pbmc10k | 5.85 | 12.60 | 0.47× | CUB block sort + searchsorted + tie + p-value |
| Wilcoxon rank-sum (vs `pyscx_cpu`) | tabula_sapiens_100k | 42.28 | 89.77 | 0.47× | (same) |
| Wilcoxon rank-sum (vs `pyscx_cpu`) | census_500k | 106.13 | 213.07 | 0.50× | (same) |
| Wilcoxon rank-sum (vs `pyscx_cpu`) | census_1m | 156.51 | 365.15 | 0.43× | (same) |
| Wilcoxon rank-sum (vs `scanpy_cpu`) | tabula_sapiens_100k | 318.77 | 89.77 | **3.6×** | (same) |
| pdex_ref (vs `pyscx_cpu`) | pbmc10k | 5.52 | 12.58 | 0.44× | (same) |
| pdex_ref (vs `pyscx_cpu`) | tabula_sapiens_100k | 42.04 | 93.81 | 0.45× | (same) |
| pdex_ref (vs `pyscx_cpu`) | census_500k | 119.86 | 212.37 | 0.56× | (same) |
| pdex_ref (vs `pyscx_cpu`) | census_1m | 268.28 | 356.34 | 0.75× | (same) |

The accel_de Wilcoxon rank-sum/pdex_ref GPU rows above are **slower than `pyscx_cpu`** (CPU's rayon-parallel implementation effectively uses ~3-5 of the 16 SLURM-allocated CPUs and is highly tuned). G10's graph capture closed ~7-10% of the gap but the GPU implementation is bottlenecked by the per-chunk `[n_obs × chunk_size]` dense materialization step. The GPU paths are still **3-5× faster than `scanpy_cpu`** — for users replacing scanpy directly, GPU is the clear win; for users who already have `pyscx.accel.rank_genes_groups(device="cpu")` working, the default GPU variant is a draw or worse.

**`pdex_ref` GPU v3-CSC (default, requires CSC sidecar).** An SCX file with a CSC sidecar (`pyscx.from_anndata(..., csc="always")` or `scx convert --csc=always`) routes the GPU `pdex_ref` through a CSC-direct driver (`pdex_ref_gpu_chunked_v3_csc` in `scx-accel/src/diffexp/gpu.rs`) — v3 is the default GPU DE route (the `SCX_GPU_DE_V3` opt-in gate was removed when v3 became the default). The driver drops the dense intermediate entirely and replaces the per-chunk pseudobulk with a block-per-(gene, group) shared-memory tree-reduce (`csc_shard_pseudobulk_kernel`) that avoids `atomicAdd` contention. The shard source (`RawGpuCscShardSource`) ships full G3-shape pipelining (2-slot pinned ring + dedicated copy stream + scoped worker pre-decode + dual event handshake) and uses cheap catalog metadata to skip CSC shards whose `[col_start, col_end)` doesn't overlap the current gene chunk — so non-overlapping shards never get decoded or uploaded. Measured 2026-05-27 against backed-AnnData fixtures (`pyscx.open(path).to_anndata(backed=True)`):

| Dataset | v2-CSR GPU (former) | **v3-CSC (default)** | v3-CSC vs v2 |
|---|---:|---:|---:|
| pbmc3k | 1.17 s | **0.42 s** | **−64%** |
| pbmc10k | 16.36 s | **0.97 s** | **−94%** (17×) |
| smartseq2 | 107.28 s | **4.49 s** | **−96%** (24×) |
| tabula_sapiens_100k | 134.04 s | **8.91 s** | **−93%** (15×) |
| census_500k | 218.35 s | **11.90 s** | **−95%** (18×) |
| census_1m | 352.16 s | **16.02 s** | **−96%** (22×) |

n_runs=5 (3 for tabula/census). v1 default and v2-CSR baselines are unchanged because neither code path was modified. CPU `pdex_ref` numbers in the previous table (272 s at census_1m) also remain the reference point: at atlas scale v3-CSC is **17–22× faster than the rayon CPU implementation**, which is a real wall-time difference for Perturb-seq screens.

In-memory inputs (`pyscx.accel.pdex_ref(scipy_csr_adata, device="gpu")`) and files without a CSC sidecar automatically fall back to the v3-CSR-direct path — same algorithm, no CSC sidecar needed, slightly slower than v3-CSC because it loses the atomicAdd-avoidance win but still drops the dense intermediate. v3 was promoted to the unconditional default in Phase V1b after a route-marked soak confirmed v3 ≥ CPU at every tier and 13–28× faster than the former v1 GPU path at medium+large scale. Parity tests (`scx-accel/src/csc/pdex.rs::tests::test_pdex_ref_gpu_v3_csc_matches_cpu_streaming` and `_csr_fallback_*`) pin v3 to the CPU oracle to fp32 tolerance.

**Which route ran is recorded, and the gate asserts it.** Every `pdex_ref` call stamps its execution route on `adata.uns["scx_accel"]["pdex_ref"]` (`route ∈ {gpu_csc_v3, gpu_csr_v3, …}`, `fallback_reason`, `csc_available`), decided by the single planner `scx_accel::route::plan_de_route`. The `accel_de` benchmark reads this back into `runs[].extra` as `gpu_dispatch_route` (human-readable) and `de_route_csc_direct` (numeric: `0.0` only when a CSC fixture was built yet a non-`gpu_csc_v3` route ran — v3 being the unconditional default since Phase V1b). `thresholds.yaml` floors `de_route_csc_direct ≥ 1.0` for the GPU pdex_ref triple, so a *silent fallback to CSR while CSC-direct was intended* — exactly the prior benchmark misread — is a hard gate failure rather than an invisible footgun. The structured `adata.uns` metadata is the signal (the former ad-hoc stderr trace was removed).

**Headline finding from the 2026-05-25 routing fix:** GPU Leiden at census scale (`pyscx_gpu` cuGraph, 1.59-3.06s on census_500k/_1m) was completely missing from prior LATEST baselines because the gate's worker jobs were activating `scx-bench` (no cugraph), failing every Leiden GPU run silently. With per-job routing → `scx-bench-gpu`, the 200-500× speedup over `leidenalg_cpu` is now visible. Same correction for kNN — the prior bench's "cuVS missing → CPU HNSW fallback" was disguising real GPU CAGRA wall times under scanpy-CPU speeds.

#### Choosing a Leiden backend

The two Leiden backends produce different partitions by design — they are not interchangeable. `device` is authoritative; there is no silent cross-backend fallback.

| Backend | `device` | Wall on census_1m | ARI vs leidenalg | Pick when |
|---|---|---:|---:|---|
| Rust-native (`scx_accel::leiden`) | `"cpu"` | ~56 s | ≈ 0.97 | Cluster IDs feed a downstream pipeline (marker-gene DE, annotation transfer, anything keyed on specific labels). Reproducibility against the CPU reference matters more than ~50 s on a 1M-cell graph. |
| cuGraph | `"gpu"` / `"gpu:N"` | ~3.5 s | **0.92** | Throughput-bound exploratory work — resolution sweeps, clustering under many random seeds, one-shot visualizations — where ARI 0.92 parity is acceptable. |

`device="auto"` (default) follows the rest of `pyscx.accel.*`: cuGraph if a CUDA device is visible and `cugraph` imports cleanly, else Rust-native. **Migration**: this differs from the pre-spec dispatcher, which always tried Rust-native first. Pin `device="cpu"` to preserve pre-spec cluster IDs. The cluster-assignment shift (ARI 0.97 → 0.92 vs leidenalg) is real for any user on a host with cuGraph installed.

cuGraph's Leiden uses a different refinement step and seed-handling scheme from leidenalg; the Rust-native implementation is a direct port of Traag et al. 2019 with the RB configuration model. The divergence is not an implementation bug — see `CLAUDE.md` § Known Limitations.

`device="gpu:N"` pins the cuGraph call to CUDA device `N` via `cupy.cuda.Device(N)`. Bare `"gpu"` is `"gpu:0"`. Out-of-range indices are rejected by `resolve_device`'s validation against `cudarc::GpuDevice::count()`. The Python `leidenalg` shim has been removed — callers who want it run `scanpy.tl.leiden(flavor="leidenalg")` directly.

#### Preprocessing device dispatch

`pyscx.accel.{normalize_total, log1p, highly_variable_genes}` accept `device="cpu|gpu|auto"`. The GPU path is eager (materializes to scipy CSR). **`log1p(device="gpu")` on a materialised scipy/dense X warns and falls back to CPU** — the H→D + kernel + D→H round-trip dominates log1p's trivial math. The pre-fallback measurement (retained as motivation):

| Op | pbmc3k CPU / GPU | tabula_sapiens_100k CPU / GPU | census_1m CPU / GPU |
|---|---|---|---|
| normalize_total | 0.004s / 0.004s (1.0×) | 0.61s / 0.43s (**1.4×**) | 3.27s / 3.00s (**1.1×**) |
| log1p (pre-fallback) | 0.003s / 0.41s (**0.01×**) | 0.20s / 9.23s (**0.02×**) | 1.46s / 63.78s (**0.02×**) |
| fused normalize+log1p | 0.006s / 0.41s (0.01×) | 0.83s / 9.73s (0.09×) | 4.50s / 67.45s (0.07×) |
| highly_variable_genes (seurat_v3) | 0.06s / 0.07s (0.9×) | 3.42s / 3.40s (1.0×) | 25.77s / 28.31s (0.9×) |

Practical recommendation: **use the GPU preprocessing path only via the `normalize_total → log1p` fusion-marker chain on backed SCX data, and only when the downstream consumer is also GPU**. The fused-chain optimization is the only case where GPU preprocessing doesn't round-trip through the host. Standalone `log1p(device="gpu")` on materialised X emits a `UserWarning` and runs `sc.pp.log1p` instead; the GPU fast path is preserved when log1p sees the fusion marker planted by `normalize_total(device="gpu")`, or when X is still backed/lazy.

**Dispatch logic:** for an in-memory `X`, in-VRAM `pyscx.accel.pca(device="gpu")` routes to rapids-singlecell (`rsc.pp.pca`). The native GPU PCA path (backed/lazy/streaming inputs, or `SCX_FORCE_NATIVE_GPU=1`) is **always randomized** — the in-VRAM covariance core was removed, so `method="covariance"` / `"auto"` resolve to randomized on GPU (covariance is still honored on the CPU path). The randomized path accepts `qr_method="householder"` (default, always-stable) or `"cholesky"` (CholeskyQR2 — opt-in, surfaces `RuntimeError` on non-SPD Gram so callers can retry with Householder).

**Correctness.** On pbmc3k + census_1m, GPU PCA's 50 leading PCs match scanpy's reference to cosine ≥ 0.9999 sign-agnostic (`gpu_pca_validation.json`). kNN via rapids `rsc.pp.neighbors` matches scanpy-neighbors at recall = 1.0 on pbmc3k and ARI 0.91 against a downstream Leiden on tabula_sapiens_100k. UMAP via rapids `rsc.tl.umap` trustworthiness 0.9233 (vs CPU 0.9238) on pbmc3k.

Native GPU PCA (streaming/randomized path, used for backed/lazy/streaming inputs or `SCX_FORCE_NATIVE_GPU=1`) streams shards from disk → GPU kernels shard-by-shard without materializing the full matrix — enabling PCA on datasets larger than VRAM. In-VRAM PCA routes to rapids `rsc.pp.pca`.

#### Differential expression

`pyscx.accel.pdex_ref(..., device=…)` and `pyscx.accel.rank_genes_groups(..., device=…)` both gained a `device="auto"|"cpu"|"gpu"[:N]"` selector. The GPU path uses per-gene CUB `BlockRadixSort` of the reference column once per gene chunk, batched warp-cooperative `searchsorted` to derive U₁ for every test group, merge-walk combined tie correction, and on-device `erfc` p-value matching the CPU formula bit-for-bit.

| Operation | Dataset | n_pool | CPU | GPU | Speedup | Notes |
|---|---|---:|---:|---:|---:|---|
| `pdex_ref` | pbmc3k (2.7K) | n_ref ≈ 540 | 0.62 s | 0.90 s | 0.69× | launch-overhead bound |
| `pdex_ref` | pbmc10k (12K) | n_ref ≈ 2.4K | 3.6 s | 3.4 s | 1.07× | ~tied |
| `pdex_ref` | smartseq2 (18K) | n_ref ≈ 3.5K | 21.4 s | 18.5 s | **1.16×** | searchsorted starts winning |
| `pdex_ref` | tabula_100k → census_1m | n_ref > 8192 | 54 → 317 s | **skip** | — | v1 capacity cap |
| Wilcoxon rank-sum (1-vs-rest) | pbmc3k (2.7K) | n_obs = 2.7K | 0.74 s | 0.92 s | 0.80× | launch-overhead bound |
| Wilcoxon rank-sum (1-vs-rest) | pbmc10k → census_1m | n_obs > 8192 | 3.4 → 210 s | **skip** | — | v1 capacity cap |

The headline speedup is modest because v1 caps the per-gene sort pool at `GPU_DE_BLOCK_SORT_CAPACITY = 8192` cells — the CUB `BlockRadixSort` is one block per gene, holding the whole row in registers + shared memory. Above that, the dispatch returns `AccelError::InvalidInput("…use device='cpu' or subsample the reference")` and the caller falls back to the rayon-parallel CPU path. Where the GPU does run (small + medium datasets, mid-size reference groups), launch overhead and chunked-upload latency dominate the on-device sort + searchsorted work. The spec-anticipated **10–50× win** lives at Perturb-seq scale (≥ 50K cells × hundreds of perturbation groups, `n_ref` typically a few thousand non-targeting controls) — none of the dataset-tier fixtures match that group structure with the synthetic 2-way `groupby` the benchmark falls back to. Lifting the 8192 cap via a tiled merge-sort upgrade is deferred to PR series G4.

**Default v3-CSC path (PR series G4.3, 2026-05-27; promoted to default in Phase V1b).** A backed AnnData over an SCX file with a CSC sidecar (built via `pyscx.from_anndata(..., csc="always")` or `scx convert --csc=always`) routes `pdex_ref(device="gpu")` through a CSC-direct pseudobulk + scatter-to-gene-major kernel pair that drops the dense intermediate entirely. The CSC shard source is fully pipelined (2-slot pinned ring + dedicated copy stream + worker pre-decode) and pre-filters CSC shards by gene-chunk overlap using cheap catalog metadata, so non-overlapping shards are never decoded or uploaded. Bench (backed-AnnData, n_runs=3–5, median wall_s): **0.42 / 0.97 / 4.49 / 8.91 / 11.90 / 16.02 s** on pbmc3k / pbmc10k / smartseq2 / tabula_sapiens_100k / census_500k / census_1m respectively — **13–24× faster than v2-CSR GPU** and 17–22× faster than the rayon CPU implementation at atlas scale. In-memory inputs (no SCX file) automatically fall back to the v3-CSR-direct path. See the [`pdex_ref` GPU v3-CSC table](#per-operation-timing) above for the full numbers and the disposition. v3 is the unconditional default GPU DE route since Phase V1b.

**Correctness signal** (gated via `runs[].extra` in the `v0.4.3-g1-gpu-de` baseline):

| Metric | Threshold | Observed |
|---|---:|---:|
| `de_pval_agreement_vs_cpu` (mean Spearman ρ over shared (group × gene) p-values) | ≥ 0.999999 | 1.0 (pbmc3k, pbmc10k), 0.999999 (smartseq2) |
| `de_top_gene_overlap_vs_cpu` (median top-200 Jaccard per group) | ≥ 0.95 | 1.0 (pbmc3k, pbmc10k), 0.985 (smartseq2) |

Tolerance-based parity for p-values / FDR (not exact) because of `erfc` and sort-order numerics; U statistics agree exactly in f64. The CPU path itself is pinned bit-for-bit to upstream `pdex` via `pyscx/tests/test_pdex_ref_parity.py`, so CPU↔GPU parity here transitively pins the GPU path to the upstream oracle.

GPU Wilcoxon rank-sum (`rank_genes_groups(device="gpu")`) routes through `plan_de_route` — when a CSC sidecar is present, it takes the `gpu_csc_v3` CSC-direct path (same as `pdex_ref`); otherwise it falls back to `gpu_csr_v3`. `prefer_format="csc"` on the CPU path uses `CpuCsc`.

#### GPU perturbation-evaluation metrics

`pyscx.accel.perturbation_metrics` and `pyscx.accel.energy_distance` gained a `device=` selector (pyscx ≥ 0.11.2). GPU `perturbation_metrics` runs the per-group pseudobulk means on the device (reusing the DE CSR pseudobulk kernels; route `gpu_csr`) with the five bulk metrics on the host; GPU `energy_distance` runs a gemm-based pairwise-distance mean (`‖x−y‖² = ‖x‖² + ‖y‖² − 2·xyᵀ`; route `gpu_dense`) for euclidean/cosine at f32 with f64 reductions. `discrimination_score` has no GPU kernel — exact-rank parity is not f32-safe, so it is deferred. CPU↔GPU parity: `perturbation_metrics` `atol ≈ 1e-6`, `energy_distance` `atol = 1e-4` (it is a Pearson correlation).

CPU-vs-GPU wall time on H100 (synthetic paired real/pred, 2K genes × 50 perturbations, 3-run median, pyscx 0.11.2; via `benchmarks/scripts/gpu_cpu_bench.py` in the cell-eval-scx fork):

| Metric | n_obs | CPU (s) | GPU (s) | Speedup | Route |
|---|---:|---:|---:|---:|---|
| `perturbation_metrics` | 10K | 0.50 | 1.31 | 0.38× | `gpu_csr` |
| `perturbation_metrics` | 100K | 4.89 | 4.90 | 1.00× | `gpu_csr` |
| `perturbation_metrics` | 1M | 40.35 | 44.26 | 0.91× | `gpu_csr` |
| `energy_distance` (euclidean) | 10K | 0.91 | 0.97 | 0.94× | `gpu_dense` |
| `energy_distance` (euclidean) | 100K | 7.44 | 4.35 | **1.71×** | `gpu_dense` |

GPU helps the compute-bound metric: `energy_distance` (an O(N²)-per-perturbation pairwise-distance gemm) reaches **1.71× at 100K** and widens with cell count — and it is what makes the metric feasible at atlas scale, where the CPU O(N²) reference is skipped (the bench caps `energy_distance` at ~200K cells; the CPU baseline is infeasible beyond — see the [Perturbation Metrics](#perturbation-metrics-cell-eval--arc-bench-parity) footnote ¹). `perturbation_metrics` is a cheap pseudobulk mean (O(nnz), memory / host-transfer-bound), so GPU ≈ CPU across sizes (small data even regresses on kernel-launch + host→device overhead) — its GPU kernel exists for uniform `device=` dispatch, not a speedup. All GPU runs took a `gpu_*` route (no silent CPU fallback); numeric parity is gated by `pyscx/tests/test_eval_metrics_gpu_parity.py` and the fork's `tests/test_scx_parity.py`. End-to-end, `cell-eval run --device gpu` matches `--device cpu` within the documented per-metric tolerances (fork `benchmarks/scripts/gpu_e2e_parity.py`).

#### Canonical baseline

Two baselines live side-by-side under `benchmarks/comprehensive/results/baselines/`. **Accel** PRs (PCA / kNN / UMAP / Leiden / preprocess / HVG / DE) gate against `LATEST` (currently accel-only); **format / cloud / multimodal** PRs must pin the multi-surface baseline `v0.6.2-n_counts-augmentation` explicitly (not `LATEST`). The split exists because the multi-surface baseline captures `accel_*` rows but doesn't produce gate signal against them — see [benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating).

| Use | Baseline | Date | Coverage |
|---|---|---|---|
| Format / cloud / multimodal | `v0.6.2-n_counts-augmentation` (pin explicitly — not `LATEST`) | 2026-05-11 | 806 rows × 8 datasets (`pbmc3k` → `census_1m`, `cite_seq_pbmc`, `multiome_pbmc`) |
| Accel (incl. `accel_de`) | `LATEST` → `v0.6.5-accel-gpu-to-gpu-anndata` | 2026-06-10 | rapids-routed accel rows; cross-tier rapids route + correctness gates (`*_route_rapids_correct`, `*_fallback_no_rapids_correct`); `to_gpu_anndata` promotion |

Per-run correctness metrics (`cosine_sim_min`/`mean`, `recall_vs_scanpy`, `trustworthiness`, `ari_vs_leidenalg`, `max_abs_diff_vs_scanpy`, `hvg_overlap_vs_scanpy`, plus `de_pval_agreement_vs_cpu` / `de_top_gene_overlap_vs_cpu` added in G1) flow through `runs[].extra` so the floor checks in `thresholds.yaml` evaluate real observed values, not `missing` placeholders.

```bash
# Accel PRs (PCA / kNN / UMAP / Leiden / preprocess / HVG / DE) — default LATEST:
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only

# Format / cloud / multimodal — pin the multi-surface baseline:
python benchmarks/comprehensive/scripts/gate_candidate.py --no-accel \
    --baseline benchmarks/comprehensive/results/baselines/v0.6.2-n_counts-augmentation
```

Older accel-only baselines (`v0.6.0-gpu-phase1-7`, `v0.6.0-gpu-phase1-7-multidataset`) remain in-tree for historical bisects but are no longer the gate targets. The earlier stop-gap wrappers (`benchmarks/scripts/gpu_regression_{diff,driver}.py` and `slurm_gpu_regression*.sh`) have been deleted; use `gate_candidate.py` for accelerator regression runs.

#### Changes vs previous version

- **Covariance-PCA dispatch path** on GPU (threshold `n_vars ≤ 8000`) — *historical, removed.* The native in-VRAM covariance PCA core (`gpu_pca_covariance.rs`, `covariance_pca_gpu`, `GPU_COVARIANCE_PCA_THRESHOLD`) was deleted; in-VRAM PCA routes to rapids `rsc.pp.pca`. The numbers below are from the pre-removal baseline: on tabula_sapiens_100k (HVG-shaped input) GPU PCA ran 1.7× vs CPU, up from 0.9× in the earlier baseline. On census_1m at the same n_vars, the speedup remained 0.9×. Native streaming/randomized PCA survives for >VRAM workloads.
- **Randomized PCA's critical path** fully GPU-resident — the prior `Q → host → f64` SVD tail and per-iteration `d_m` download round-trip are gone (cuBLAS `sgemv` + `sgemm`). Correctness preserved (cosine ≥ 0.9999 on real data).
- **Opt-in CholeskyQR2** (`qr_method="cholesky"`) for the randomized path; benchmark-suite variants `gpu_randomized_pca_chol` vs `gpu_randomized_pca_householder` pending from the current cluster run.
- **Standalone GPU preprocessing ops** (`normalize_total`, `log1p`, `highly_variable_genes`) gain a `device` kwarg. In isolation they are slower than the CPU path (see table above — `log1p` is ~40× slower on tabula due to H2D/D2H round-trips); the `normalize_total → log1p` fusion marker is the only fast path.
- **cuGraph Leiden** exposes the `theta` knob via `pyscx.accel.leiden(theta=...)`.

### Go/No-Go Status

| Gate | Criterion | Result |
|------|-----------|--------|
| PCA correctness | cosine similarity > 0.99 | **Pass** |
| kNN recall | recall@15 > 0.95 | **Pass** |
| Graceful fallback | CPU fallback when no GPU | **Pass** |
| 10x pipeline speedup | end-to-end 10x vs CPU | **Fail** (3.8x achieved) |


## Training Loader

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

### scx vs shardad vs cellstream — random-gather training loader (2026-07-10)

Head-to-head against the two custom per-cell stores built for scatter-heavy
training: **shardad** (`.shad`, condition-grouped zstd CSR shards) and
**cellstream** (`~/dev/python/cellstream` — a scatter-immune per-cell store over
shardad's codec, `gather_rows(row_ids)→CSR`). Neither has a native batched
DataLoader, so each is driven as the honest "loader you'd build on it": permute
cell ids, slice into batches, gather each batch and apply the same HVG/normalize as
scx. batch_size=1024; median batches/sec over the shuffled epoch.

**`raw` scenario** (random gather, no HVG/normalize):

| Dataset | **scx (auto)** | cellstream | shardad |
|---|---|---|---|
| pbmc3k | **35.2** | 14.6 | 8.1 |
| pbmc10k | **27.7** | 11.7 | 2.7 |
| smartseq2 | **18.1** | 8.2 | 0.6 |
| tabula_sapiens_100k | **36.0** | 2.7 | 0.5 |
| census_500k | **48.1** | 13.0 | — |
| census_1m | **57.2** | 12.7 | — |

**`hvg_norm` scenario** (HVG-2000 + normalize + log1p):

| Dataset | **scx (auto)** | cellstream | shardad |
|---|---|---|---|
| pbmc3k | **49.2** | 12.5 | 8.2 |
| pbmc10k | **36.6** | 11.3 | 2.6 |
| smartseq2 | **23.0** | 7.5 | 0.6 |
| tabula_sapiens_100k | **40.9** | 12.7 | 0.5 |
| census_500k | **57.3** | 11.7 | — |
| census_1m | **64.1** | 11.3 | — |

**On-disk fixture size (MB, lower is better):**

| Dataset | scx fast (Scx1) | scx auto (ShufDeltaZstd) | cellstream | shardad |
|---|---|---|---|---|
| pbmc10k | 39.8 | 30.1 | 34.6 | 31.7 |
| smartseq2 | 552.7 | 262.7 | 255.7 | 253.2 |
| tabula_sapiens_100k | 448.9 | 233.5 | 234.5 | 222.2 |
| census_500k | 1841.8 | 1182.9 | 920.6 | 859.2 |
| census_1m | 3988.8 | 2800.1 | 1747.0 | 1622.0 |

**Findings.**
- **Throughput: scx is the fastest training loader at every scale** — ~2.4× cellstream
  and ~10× shardad on pbmc10k random gather, widening to ~13× cellstream on tabula.
  cellstream clearly beats shardad (shardad's per-call subset read collapses to
  ~0.5 batches/sec at ≥50k cells; its census fixtures exist but the loader runs
  did not complete).
- **Storage: the default optimizes for size.** As of the 2026-07-12 codec-intent
  flip, `codec="auto"` is **cost-aware adaptive** (predominantly ShufDeltaZstd) — the
  column labeled `scx auto (ShufDeltaZstd)` above is what the new default `auto`
  produces, and the `scx fast (Scx1)` column is `codec="fast"`. Adaptive
  `auto` recovers ~30% of the disk vs `fast` (census_1m 3989→2800 MB) while staying
  **within ≤~3% of `fast` on the realistic `hvg_norm` training scenario at every scale**
  (see the head-to-head section below), narrowing but not closing the gap to
  cellstream/shardad's aggressive zstd+dictionary codec. Latency-critical CPU training
  that wants the old decode-max behavior pins `codec="fast"`.

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
`v0.11.0-cellstream-recapture` baseline. Emitting census scx rows required two harness
fixes: scaling the loader `memory_budget` to the SLURM allocation (`_scx_memory_budget_mb`,
0.6×`--mem`) so `batch_size` stays at 1024 — the default 4096 MB is too small for 1M×61,497
full width and the auto-tune otherwise collapses `batch_size` to 64 (a budget/estimator
interaction, not a fixture bug: standard 62×16k-shard geometry) — and skipping the
`pyscx_training_dataset_workers2` scenarios above 250k cells, which time out at census scale
and would otherwise discard the whole (dataset, format) result.*

### scx auto vs cellstream — the size-vs-throughput head-to-head

The fairest apples-to-apples: scx's **adaptive default codec** (`auto` =
ShufDeltaZstd, framed) vs **cellstream**, the storage-optimized scatter competitor.
Both target smaller-on-disk random-access training. batch_size=1024; 2026-07-10/11.

**Compression + on-disk size** (ratio vs source h5ad; MB):

| Dataset | ratio auto | ratio cellstream | MB auto | MB cellstream |
|---|---|---|---|---|
| pbmc10k | **6.73** | 5.86 | 30 | 35 |
| smartseq2 | 4.08 | 4.19 | 263 | 256 |
| tabula_100k | 6.79 | 6.77 | 234 | 235 |
| census_500k | 5.14 | **6.60** | 1183 | 921 |
| census_1m | 4.07 | **6.52** | 2800 | **1747** |

**Training throughput** (median batches/sec; higher = better):

| Dataset | raw auto | raw cellstream | hvg_norm auto | hvg_norm cellstream | throughput edge |
|---|---|---|---|---|---|
| pbmc10k | 24.4 | 11.7 | 37.2 | 11.3 | **auto 2–3×** |
| smartseq2 | 18.8 | 8.2 | 23.3 | 7.5 | **auto 2–3×** |
| tabula_100k | 29.8 | 2.7 | 39.5 | 12.7 | **auto 3–11×** |
| census_500k | 52.5 | 13.0 | 58.3 | 11.7 | **auto 4–5×** |
| census_1m | 57.3 | 12.7 | 63.6 | 11.3 | **auto 4.5–5.6×** |

**Tradeoff.** `auto` delivers **2–11× cellstream's training throughput at every
scale**; the two are **comparable on compression through tabula**, and cellstream pulls
ahead only at **census** (~6.5× vs ~4.1×, ~1.6× smaller on disk) via its trained zstd
dictionary. cellstream also opens faster (pbmc10k TTFB ~0.035 s vs `auto`'s ~0.4 s —
mmap store vs pipeline spin-up). Net: `auto` dominates on throughput; cellstream only
edges it on atlas-scale footprint, at a 4–6× throughput cost.

*Methodology: cellstream from the `benchmarks/comprehensive` ml_loader capture; `auto`
throughput from a direct `_run_scx_epoch` measurement (warmup + timed epoch) that
bypasses the `pyscx_training_dataset_workers2` scenario (which times out at scale in the
full harness). Both use batch_size=1024, HVG=2000. The single-epoch direct `auto` figures
run a few percent above the harness median-of-N (e.g. pbmc10k 24.4/37.2 here vs the baseline
harness row 22.2/34.3); the at-scale `auto` harness capture is deferred, so these are the
best available `auto` numbers at census scale.*

### ShufDeltaZstd loader-decode cost (Phase-D D0 profiling)

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

### IndexPlanDataset (plan-driven paired reads)

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

#### Scattered pair-gather via the row-group BlockIndex (L1+L2)

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
> [`docs/benchmark_manifest.md` § Workflow](benchmark_manifest.md#workflow).
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

#### Row-group LRU on the scattered gather (OPT-FORMATIO-1)

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

#### Cell-set scatter routes, re-measured after the row-group LRU (phase-0 gate)

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

### Out-of-core loader — cold-cache measurements and the P-1 premise gate

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

#### ⚑ The P-1 premise gate

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

### Data-wait fraction per regime (`p`) — the loader-work gate

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

### Tier-1 loader fixes: gather pre-sizing and the crop mask (phase 1)

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

#### The crop mask is reliable and small, which is not what was predicted

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

#### Why the numbers are paired, and what the first attempt got wrong

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

#### What this does not say

- **The committed rows are reduced records, not `BenchmarkResult`s.** Each
  round's real `BenchmarkResult` was written inside the arm's scratch worktree,
  which the job removes on exit; what is committed is the per-round metric
  scalars the driver copied out. The table recomputes exactly from them — medians,
  ratios, sign counts and p-values — but the file does not carry the
  `schema_version` / `system` / `runs` envelope
  [docs/benchmark_manifest.md](benchmark_manifest.md) describes. The driver now
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

### Bounded reader registry: what a large manifest actually costs (phase 2)

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

#### Does the leased-`Arc` seam cost anything? (phase 2)

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

### Tokenisation kernels: what each per-cell step costs (phase 3)

The W6 kernel set (`pyscx.tokenize`) is the per-cell numerics every
transformer-class model re-implements in Python — rank, bin, top-K crop,
expression-weighted sampling. These are their first captured numbers.

Two-build A/B, SLURM `2954036`, host `GPU70DC`, `main` `e8d1f9a9` against
`phase3-tokenize-kernels` `8cab0e3c`, 12 interleaved rounds per dataset,
`RAYON_NUM_THREADS=16`. Raw rows under
`results/raw/phase3_tokenize/tokenize_ab.json`.

#### The existing collate arm, which had to stay flat

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

#### Per-kernel cost, first capture

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

### Neighbourhood plans: what a spatial gather costs (phase 4)

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

#### Steady state, at k = 6 and 146 sets per batch

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

#### What these numbers are and are not

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

### Shard-cache sizing on the gather path (data-load Phase 1, 1A)

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
  [benchmark_manifest.md](benchmark_manifest.md); they are an **A/B against each other** on a
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

### Obs categorical codes without pandas (data-load Phase 1, 1C)

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
Both on-disk encodings are accepted, including a file that carries both across its shards —
`from_anndata` writes dictionary-encoded obs while `append` decodes to plain strings, so any
appended file is mixed.

On the cloud path the same pass lands, plus the projection half that was missing:
`CloudExperiment.read_obs(columns=)` previously assembled the entire obs table and then
projected it in memory (its own docstring conceded "the network cost is the full obs metadata
regardless"). It is now a genuine per-shard pushdown, so the network cost is the requested
columns' bytes.

### Count-depth downsampling in the gather (data-load Phase 1, 1B)

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

### Global pre-shuffle (data-load Phase 1, 1D)

**`scx sort --shuffle` does what it exists to do and costs nothing to read: per-shard label
divergence from the corpus mix collapses by 27–43×, while cache-cold `TrainingDataset`
throughput is unchanged (1.01–1.03×). The one real cost is on disk, and on the default
`codec="auto"` it is large — 1.86–2.09× on X.** Captured cold on `shuffle_layout` / `scx_auto`,
`seed=42`, `n_runs=3` for the epochs and 2 for the rewrite, one job at a time
(`--dependency=afterany`) on a 16-core / 200 GB `cpu` node.

#### Batch mixing — the thing the feature is for

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

#### Read throughput — a neutrality check, not a speedup

| dataset | unshuffled | shuffled | ratio |
|---|---|---|---|
| tabula_sapiens_100k | 11 911 samples/s | 12 296 samples/s | 1.032× |
| census_500k | 14 515 samples/s | 14 650 samples/s | 1.009× |

**Read ~1.0× as the arm passing.** A shuffled file is still streamed sequentially, so
neutrality is the expected and desired result; what this arm exists to catch is a *regression*,
evidence that the permutation cost the sequential read something. It did not. The throughput a
pre-shuffle actually buys is downstream — a consumer can drop `shard_group_size` and its memory
budget without losing batch diversity — and is not measured here.

#### Output size — pin your codec

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
   `census_500k` is in [sharding.md § Shuffling for training](sharding.md#shuffling-for-training-scx-sort---shuffle).
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

#### Rewrite cost

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
[`benchmarks/multimodal_query_bench.py`](../benchmarks/multimodal_query_bench.py)
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
[docs/multimodal.md § 3.4](multimodal.md#34-modality-scoped-queries--querymodality)).

## File Operations

| Operation | Speed |
|-----------|-------|
| Append 10K cells | **1 ms** |
| Merge 3 files | **342 MB/s** |
| Compact (after 3 appends) | 0.98x fresh-write size |

### Sort (physical layout)

`scx sort` (see [sharding.md § Sorting for read locality](sharding.md#sorting-for-read-locality-scx-sort))
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
(see [sharding.md](sharding.md) and [docs/api.md](api.md)). It is produced two
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

**Head-to-head vs shardad.** Grouped sharding is also the axis where scx most
directly overlaps [shardad](https://github.com/ArcInstitute/shardad) — a
counts-oriented single-file (`.shad`) format whose distinguishing feature is
native condition grouping (`write_sharded(group_by=, reference=)` +
`read_group()` / `read_reference()`). The cross-format `grouped_read`
comprehensive benchmark
(`benchmarks/comprehensive/benchmarks/grouped_read.py`) runs both stacks over the
same integer-count perturbation fixtures (`nb_glm_synth`, `replogle_k562`,
`tahoe_c38`), timing the grouped write (wall + on-disk size) and the
per-perturbation `read_group` / `read_reference` reads, and asserting both
formats partition the obs axis and isolate the reference (a hard gate for each
arm). The float paired fixture `pert_synth_10k` is scx-only above — shardad is a
counts format and rejects float `X` on grouped write. Both formats share the
`scx-bench` conda env.

Measured head-to-head (release build, `ctc_cpu_priority`; median wall, `.scx`/`.shad`
file size; `read_group` is the per-perturbation read):

| dataset (X) | grouped write (scx→shardad) | file size | `read_group` (scx→shardad) |
|---|---|---|---|
| `nb_glm_synth` (CSR counts) | 3.6 → **3.0 s** | 125 → **73 MB** | **0.60** → 1.02 s |
| `replogle_k562` (dense, float) | **45.6** → 53.2 s | 2330 → **2238 MB** | **0.71** → 9.24 s |
| `tahoe_c38` (CSR, float) | 18.4 → **9.5 s** | 1427 → **1403 MB** | **1.27** → 4.02 s |
| `chemogenetic_rgfp` (CSR counts) | 34.2 → **21.0 s** | 1250 → **803 MB** | **0.61** → 0.86 s |

**Reading it.** shardad wins **grouped-write** speed on in-RAM-sized data (it loads an
in-memory CSR then encodes; scx streams from the h5ad) and **integer-count compression**
(≈1.5–1.7× smaller on raw counts `nb_glm_synth`/`chemogenetic_rgfp`; ≈parity on the
float fixtures). scx wins **`read_group`** on every dataset — decisively where the
reference/group spans many shards (`replogle_k562` **13×**, `tahoe_c38` **3×**) — because
its predicate-index byte-range read decodes only the group's rows. scx additionally offers
`query().filter_obs(...)` pushdown (≈`read_group` latency) and out-of-core reads shardad
lacks (see the "Out-of-Core Peak RSS" report section: at census_5m scx streaming peaks at
~19 GB vs shardad's ~90 GB full materialize). Both formats pass the read-back correctness +
reference-isolation gates on all fixtures. The live tables render in the comprehensive
report's "Grouped Read/Write — scx vs shardad", "Out-of-Core Peak RSS", and "Format
Capability Matrix" sections.

**F6 — in-memory grouped-write fast path (closes the write-speed gap).** The
grouped-write times in the head-to-head table above are the pre-F6 path (scx
streamed the reorder row-by-row through a single-threaded encoder, losing to
shardad's in-RAM encode). F6 fixed both halves: **Phase 0** caps the emitter's
per-shard buffer at `--group-write-block-bytes` (default 256 MB), sub-flushing an
oversized group across shards so grouped write no longer OOMs on a huge reference
group; **Phase 1** added an in-memory **parallel** fast path (`scx sort
--group-by` / `pyscx.sort(group_by=)` with no `--memory-budget`) that gathers the
resident CSR and encodes blocks across rayon threads — **byte-identical** to the
single-threaded path. A/B on real fixtures (release pyscx, 16-core `cpu`,
`/usr/bin/time -v` true peak; "legacy" = `SCX_SORT_NO_INMEM_FAST=1`, the
Phase-0 single-threaded emitter):

| dataset | scx fast | scx legacy | speedup | shardad grouped write | fast peak RSS |
|---|---|---|---|---|---|
| `chemogenetic_rgfp` (136K × 18K, 909M nnz) | **6.5 s** | 24.2 s | 3.7× | 21.0 s | 18.7 GB |
| `replogle_k562` (69K × 6.5K) | **12.8 s** | 28.0 s | 2.2× | 53.2 s | 6.3 GB |
| `tahoe_c38` (69K × 63K) | **9.0 s** | 17.4 s | 1.9× | 9.5 s | 5.0 GB |

scx **beats shardad on grouped write** for in-RAM-sized data (chemogenetic
6.5 s ≪ 21 s; tahoe 9.0 s < 9.5 s), while keeping the streaming/`--memory-budget`
path as the atlas-scale moat. The parallel gather trades peak RSS for speed — on
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

**Beyond grouped sharding (full cross-format campaign, release build).** shardad also
runs as a first-class format in the comprehensive suite:

- **Compression** — shardad is smaller than scx on **integer counts**, but the margin
  depends entirely on which scx codec you compare. Against the **default `scx_auto`** the
  gap looks large (`census_1m` 1.62 vs 3.99 GB ≈2.5×; `tabula_100k` 222 vs 449 MB), but
  `scx_auto` is not scx's best integer codec — against **`scx_compact_trial`** (framed
  ShufDeltaZstd, the codec to compare) the gap narrows to **~1.0–1.7×** (`census_1m` 1.62
  vs 2.80 GB ≈1.7×; `tabula_100k` 222 vs 234 MB ≈parity; `chemogenetic_rgfp` 802 vs 968 MB
  ≈1.2×), and scx wins the small datasets (`pbmc3k` 4.5 vs 6.2 MB). ≈parity on
  **log-normalized/float** data. shardad's byte-filter is tuned for integer UMI streams;
  scx's ShufDeltaZstd/pcodec/zstd competes closely. See the detailed
  [scx vs shardad — full feature parity](#scx-vs-shardad--full-feature-parity) section
  below for the full per-dataset tables.

  > **Why isn't ShufDeltaZstd the default?** Despite better compression,
  > ShufDeltaZstd has **no in-VRAM decode via BitPacker4x** (Scx1 decodes its
  > indices in VRAM; ShufDeltaZstd's GPU path uploads planes / host-bounces).
  > CPU decode is **≈ Scx1 parity (~1.09× at a 16 K-row shard, at/below Scx1
  > for smaller shards)** after the Phase-B SSE2 byte-transforms (was 1.3–1.8×
  > slower); the remaining reason to keep Scx1 as the `auto` default is the GPU
  > analysis path, not CPU decode. `compact-trial`
  > gives the best of both by trial-encoding each shard and keeping the
  > smaller, but doubles encode time and frames all output (no in-VRAM Scx1
  > decode). See [codec.md § "Codec tradeoff summary"](codec.md#codec-tradeoff-summary--scx1-vs-shufdeltazstd)
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
  ~flat while shardad must materialize the whole matrix —

  | dataset | scx streaming | scx materialize | shardad materialize |
  |---|---|---|---|
  | `census_500k` | 5.5 GB | 8.4 GB | 6.2 GB |
  | `census_1m` | 6.0 GB | 16.1 GB | 11.3 GB |
  | `census_5m` | **19.0 GB** | 126.7 GB | **87.4 GB** |

  shardad has no streaming path, so full materialize is its only read mode — the
  capability boundary at atlas scale.
- **Parallel read scaling** — scx scales 3.6× (`tabula_100k`) / 4.5× (`census_1m`) to 32
  threads; shardad is faster single-threaded but scales only ~1.0×/2.1× (its
  multiprocessing read is materialization-bound). Net at 32 threads scx is faster on both.
- **ML training loader** — scx `TrainingDataset` sustains 45.5 (`tabula_100k`) / 767–1,193
  (`census_1m`) batches/s; a shardad random-access row-slice loader (shardad has no native
  batched loader) manages ~0.6 batches/s — the ML-throughput gap is decisive.
- **Fidelity** — shardad round-trips counts losslessly (`shardad_fidelity` gate: 0 value
  mismatches on `pbmc3k` / `nb_glm_synth` / `tabula_100k`), and its dtype/materialization
  knobs (`to_anndata(container="dense", data_dtype="float16", allow_lossy=True)`) verify.

Net: shardad's durable edges are **integer-count file size** and single-shot in-RAM write;
scx's are **per-group reads, out-of-core, parallel scaling, ML throughput, and breadth**
(query engine, accelerators, cloud, multimodal, R — see the capability matrix).

---

## Doublet-Caller Interop (`doublet_interop`)

External doublet callers run in their own conda environments, their results are
imported to canonical obs columns, and everything is scored — against each
other and against labelled truth. Captured 2026-08-03 on a Chimera CPU node;
`scDblFinder` 1.24.0 (R, `rscx` env) and `Scrublet` via `sc.pp.scrublet`
(scanpy 1.12, `scx-bench` env). Solo is an optional comparator and is skipped
here — no environment on this cluster ships scvi-tools.

**The runs are bit-identical across repeat invocations** (all 19 emitted
metrics), because the injection seed, the tool seeds and scDblFinder's
`set.seed` carry end to end. That is what makes the science numbers below
gateable rather than indicative.

### Round-trip fidelity — the plumbing

The tool's own output compared with what comes back off the SCX file, **joined
by key**. A positional import passes every row-count and column-presence check
and fails only here.

| Dataset | Cells | Tool | Max score delta | Call disagreements | Matched |
|---|---:|---|---:|---:|---:|
| `pbmc3k` | 2,916 | scDblFinder | 5.0e-16 | 0 | 2,916 / 2,916 |
| `pbmc3k` | 2,916 | Scrublet | 7.2e-09 | 0 | 2,916 / 2,916 |
| `tabula_sapiens_100k` (3 donors) | 1,914 | scDblFinder | 5.0e-16 | 0 | 1,914 / 1,914 |
| `tabula_sapiens_100k` (3 donors) | 1,914 | Scrublet | 6.9e-09 | 0 | 1,914 / 1,914 |

scDblFinder's 5.0e-16 is f64→f32 rounding; Scrublet's 7e-09 is the same plus
its CSV text round trip.

### Accuracy against injected truth

Truth is **computational injection** — pairs of real cells summed, at an 8%
rate. This is `DOUBLET-DETECTION.md` **Category D**: exact labels, but summed
count vectors do not reproduce capture or ambient-RNA artifacts. Do not read
these as accuracy against real doublets, and do not pool them with
hashing- or genotype-labelled results.

| Dataset | Tool | AUROC | Precision | Recall | Recall (heterotypic) | Recall (homotypic) |
|---|---|---:|---:|---:|---:|---:|
| `pbmc3k` | scDblFinder | 0.979 | 0.780 | 0.690 | — | — |
| `pbmc3k` | Scrublet | 0.785 | 0.667 | 0.269 | — | — |
| `tabula_sapiens_100k` | scDblFinder | 0.912 | 0.821 | 0.451 | **0.484** | **0.188** |
| `tabula_sapiens_100k` | Scrublet | 0.841 | 0.750 | 0.254 | **0.286** | **0.000** |

The stratified columns are the point. Pooled recall says Scrublet finds a
quarter of the doublets; split by kind it finds **none of the homotypic ones**
on this fixture. `pbmc3k` has no `cell_type` column so its injections are
recorded as kind `unknown` rather than guessed — hence the dashes.

### Between-tool agreement

| Dataset | Raw agreement | Cohen's κ | Score Spearman | Consensus AUROC |
|---|---:|---:|---:|---:|
| `pbmc3k` | 0.953 | 0.482 | 0.280 | 0.842 |
| `tabula_sapiens_100k` | 0.966 | 0.459 | 0.369 | 0.744 |

Raw agreement is the number to distrust: at a ~7% doublet rate two tools that
both call almost nothing agree ~95% by construction. κ ≈ 0.47 is the honest
reading — moderate. Consensus AUROC sits *below* either tool alone because
`method="majority"` over two voters is an AND, and it ranks by an integer vote
count with heavy ties.

Per-donor called rates on the atlas slice show scDblFinder steady (3.6–4.5%
across three donors) and Scrublet spread 4× (1.1–4.8%) — the kind of
per-library miscalibration a pooled rate hides.

### Coverage caveat

The atlas run is the **first three donors** of `tabula_sapiens_100k`
(`THD0001`, `THD0002`, `THD0005` — 1,772 real cells), capped by
`max_batches=3`; **116 donors are dropped** and the result records them.
The cap is a runtime bound, not a sample: read these as a fixed regression
fixture, not as a characterisation of the atlas.

---

## Comprehensive Benchmarking + Cloud Validation

Closes SLAF parity, cloud validation on GCS, fragment-ops
throughput, and the regression gate. Numbers below come from live
benchmark runs on a Chimera CPU node against
`gs://arc-ctc-nextflow/scx-test/` across all four primary cloud
formats (SCX, Zarr v3, TileDB-SOMA, SLAF). See `docs/cloud.md` for
cloud-specific operational notes. The original post-ship known issues
are all resolved: the zarr `cloud_read` decompression failure (a
fixture-upload race — fixed with fcntl-serialized uploads + BLAKE3
sidecars), the SLAF cloud-path probe failure (missing
`smart_open[gcs]` dep — fixed by pinning `google-cloud-storage` in
`scx-bench-slaf.yml`), and the missing selective-predicate coverage
(no `n_counts` obs column on the staged h5ads — fixed by
`benchmarks/scripts/augment_obs_n_counts.py` populating
`obs["n_counts"] = X.sum(axis=1)` during dataset prep).

### SLAF parity

SLAF (`slafdb==0.5.2`) is a first-class competitor across every
comprehensive-suite dimension — compression, full read, selective read,
filtered-query pushdown (SQL via its DuckDB engine), correctness
round-trip, ML loader, and out-of-core memory. Key results on census_1m:

| Metric | SCX | SLAF | Zarr (zstd) | h5ad (backed) |
|---|---:|---:|---:|---:|
| Full-read peak RSS | ~345 MB | ~34 GB | ~11 GB | ~345 MB |
| ML-loader batches/s | 1,405 | ~4.1 | n/a | n/a |
| Selective `cell_type == "T cell"` | scx_pushdown | slaf_sql | skipped | h5ad_load_and_mask |

SLAF's Mixture-of-Scanners prefetcher returns 0 batches at 10M scale
with the default config — flagged as a SLAF-upstream tuning issue, not
a harness fix.

### Cloud push / pull throughput (SCX → GCS)

`pyscx.push` / `pyscx.pull` streaming throughput on the default
`.scxd/` layout. Per-request overhead dominates on tiny files; the
100K-cell dataset is where bandwidth matters:

| Dataset | Size | Push | Pull |
|---|---:|---:|---:|
| pbmc3k | 4 MB | 12.4 MB/s | 0.7 MB/s |
| tabula_sapiens_100k | 428 MB | **114.7 MB/s** | **181.1 MB/s** |

The 50 MB/s absolute floor in `thresholds.yaml` is keyed on
tabula_sapiens_100k (pbmc3k is deliberately below the bandwidth
regime). Comfortable ~2× headroom vs the floor.

### Cloud full-dataset read (cross-format)

Full-dataset cloud read — pull from GCS + materialize to in-memory
AnnData via each format's native cloud read path:

| Dataset | SCX | Zarr (zstd) | Zarr (lz4) | TileDB-SOMA | SLAF |
|---|---:|---:|---:|---:|---:|
| pbmc3k | **0.165s** | 0.627s | 0.556s | 0.944s | 1.873s |
| tabula_sapiens_100k | **2.815s** | 4.593s | 5.336s | 4.570s | 5.306s |

SCX leads on both datasets. At 100K cells, SCX is **1.6× faster than
TileDB-SOMA**, **1.6× faster than zarr (zstd)**, and **1.9× faster
than SLAF**. Mechanisms: `scx_pull_and_load`, `zarr_cloud_open`,
`soma_open_gs`, `slaf_cloud`.

### Cloud metadata open — single-GET catalog parse

Time to open the cloud-hosted fixture and surface obs / var schema
(no X materialization):

| Dataset | SCX (`open_cloud`) | Zarr (zstd) | Zarr (lz4) | TileDB-SOMA | SLAF |
|---|---:|---:|---:|---:|---:|
| pbmc3k | **0.097s** | 0.120s | 0.113s | 0.274s | 0.800s |
| tabula_sapiens_100k | **0.114s** | 0.110s | 0.112s | 0.338s | 0.754s |

SCX and zarr metadata latency are essentially dataset-size-independent
(SCX 0.097 → 0.114s going from 2.7K to 100K cells), as expected for a
single-GET catalog fetch against the exploded `.scxd/` front catalog.
TileDB-SOMA's open path does a handful of extra directory listings;
SLAF's metadata open includes loading cells/genes Lance fragments.

### Cloud filtered query (predicate pushdown)

Per-predicate median wall across the canonical predicate set
(`cell_type == "T cell"`, `n_counts > 1000`, `random_1pct`). `pbmc3k`
has no `cell_type` obs column, so the eq-predicate is excluded at
runtime by `_applicable_predicates` (`cloud_filtered.py`):

**pbmc3k** (no `cell_type`):

| Format | `n_counts > 1000` | `random_1pct` |
|---|---:|---:|
| SCX (scx_pull_and_filter) | **0.539s** | **0.530s** |
| TileDB (tiledb_cloud_value_filter) | 0.983s | 0.907s |
| SLAF (slaf_cloud_sql / slaf_cloud_stride_hash) | 1.418s | 1.216s |

**tabula_sapiens_100k**:

| Format | `cell_type == 'T cell'` | `n_counts > 1000` | `random_1pct` |
|---|---:|---:|---:|
| SCX | **2.07s** | 5.39s | 13.59s |
| TileDB | 0.80s | 9.27s | **3.04s** |
| SLAF | 1.00s | **4.97s** | 3.16s |

At 100K cells the winner rotates by predicate — TileDB's categorical
enum index wins `cell_type` equality (0.80s vs SCX 2.07s), SLAF's
DuckDB streaming WHERE wins `n_counts > 1000`, and TileDB wins
`random_1pct` via its cell-id coordinate sampler. SCX leads on
pbmc3k across the board. The current SCX cloud-filtered mechanism
is `scx_pull_and_filter` (pull full shard + local filter); a native
range-read pushdown variant is a roadmap item — numeric-range
predicates like `n_counts > X` don't currently skip shards because
catalog pushdown keys on `CategoryBitset` indices only. Zarr cloud
fixtures don't persist obs (the raw-CSR converter in `zarr_runner`
writes `indptr` / `indices` / `data` only), so zarr rows are absent
from this table by design.

### Cost model — USD per 1M cells queried (GCS same-region pricing)

Priced against the pinned `GCS_PRICING` table
(Class-B $0.004/10k GETs, same-region egress $0.00/GB on intra-region
GCE ↔ GCS). Cost is dominated by full-read egress at large scale;
metadata opens are effectively free in the committed regime. Selective
scenarios use the `n_counts > quantile_cutoff` predicate synthesized
by `_n_counts_threshold_predicates` against the augmented obs column:

| Dataset | Metadata | selective 5% | selective 20% | Full read |
|---|---:|---:|---:|---:|
| pbmc3k | $0.000000 | $0.011852 | $0.002963 | $0.000593 |
| tabula_sapiens_100k | $0.000000 | $0.000800 | $0.000200 | $0.000040 |

Per-1M-cells cost *decreases* with dataset size because the per-GET
overhead amortizes over more cells. Note that selective rows are
currently **more expensive** per-million-cells than full-read: the
denominator (matching cells) shrinks but the byte count stays roughly
constant because SCX's current catalog pushdown doesn't skip shards
on numeric-range predicates like `n_counts > X` (catalog pushdown
keys on `CategoryBitset`-indexed columns only). This is the honest
measurement the cost model is designed to surface — the selective
pull downloads the same bytes as a full pull but is accounted against
the matching-cell subset. Row-group skipping on numeric ranges is a
candidate roadmap item; it would shift the selective columns below
the full-read column.

### Cloud reader vs full pull (metadata workloads)

`open_cloud` streams only the front catalog; `pull_full` fetches the
entire fixture. Bytes transferred reflect what the underlying GCS
reads actually download:

| Dataset | `open_cloud` wall | `pull_full` wall | `pull_full` bytes |
|---|---:|---:|---:|
| pbmc3k | 0.115s | 0.315s | 4.4 MB |
| tabula_sapiens_100k | **0.117s** | **2.31s** | **408 MB** |

On tabula_sapiens_100k, `open_cloud` is **~20× faster** than a full
pull and avoids transferring 408 MB — the core "cloud-aware access"
win that justifies the exploded `.scxd/` layout.

### GCP compute-node matrix

Cloud-read throughput characterized across `n2-standard-8`,
`c3-standard-8`, and `a3-highgpu-1g`. Per-VM egress bandwidth class
(16 / 23 / 200 Gbps) is the dominant predictor for full-read wall
clock on atlases that fit the streaming-pull envelope. The launcher
(`submit_gcp_matrix.py`) pins every VM to the bucket region so
cross-region egress is impossible by construction. Results in §8d of
the benchmark report; raw numbers require `--yes-spend` to generate.

### Fragment operations throughput

`pyscx.append` / `mark_deleted` / `compact` / `rollback` throughput on
pbmc3k (see §8b):

| Operation | Median wall | Dominant throughput |
|---|---:|---:|
| append | streaming (one shard at a time for SCX→SCX; raw-copy fast path when codec matches) | ~38 MB/s |
| delete (logical) | independent of n_obs | ~155 k rows/s |
| compact | base-file read + re-encode bandwidth | ~54 MB/s |
| rollback | single root-catalog pwrite | ~3 ms |

### Regression gating

All benchmark results carry a `schema_version=1` stamp + full
provenance (git SHA, thread pinning, run_id) in their
`system.provenance` block. The on-demand gate
(`scripts/gate_candidate.py` + `scripts/compare_against_baseline.py
--gate`) evaluates relative tolerances (3% wall / 10% RSS / 1% size),
absolute floors from `thresholds.yaml` (e.g. cloud throughput ≥ 50
MB/s, keyed on tabula_sapiens_100k), and disappeared-benchmark
detection. Justification markdown files under
`results/justifications/` suppress accepted regressions with an
optional expiry date. The dashboard (`reporting/dashboard.py`) emits a
browsable HTML snapshot alongside the markdown report, threaded with
"← previous snapshot" navigation via `dashboard_history.json`.

Two baselines coexist: `LATEST` (currently
`v0.6.5-accel-gpu-to-gpu-anndata`, accel-only) is the default gate
target for accelerator PRs; format / cloud / multimodal PRs pin
`v0.6.2-n_counts-augmentation` explicitly. See the [Canonical
baseline](#canonical-baseline) table above for coverage details.
`gate_candidate.py` with no flags gates against `LATEST`.

## scx vs shardad — full feature parity

Direct head-to-head against [shardad](https://github.com/ArcInstitute/shardad)
(`0.5.1`), the condition-grouped sharded `.h5ad` replacement for perturbation
screens, across **every** benchmarkable shardad feature — not just compression.
Measured on a Chimera CPU node (scx `0.11.0`, always-framed; shardad `0.5.1`) via
the comprehensive harness (`benchmarks/comprehensive/benchmarks/{compression,
write,read_full,read_selective,parallel_scaling,parallel_write_scaling,memory,
grouped_read,ooc_rss_boundary,shardad_fidelity}.py`). scx is shown as `scx_auto`
(default per-shard codec) and `scx_compact_trial` (the per-shard smaller of the
heuristic winner vs ShufDeltaZstd — scx's best integer codec). Datasets span
tiny→census_1m plus the perturbation/grouped set (replogle_k562, tahoe_c38,
chemogenetic_rgfp — shardad's home turf).

**Headline:** the two projects trade wins by axis. **shardad** leads on
compression of large integer counts, write speed, and small/medium read peak-RSS
(its uint16 in-decode materialization). **scx** leads on out-of-core streaming RAM
at scale, grouped query latency, and carries a vastly broader capability set (GPU,
query engine, R, cloud, multimodal, ML loader). The old self-reported "shardad
1.9–3.2× smaller than scx 0.8.7 on integer counts" has narrowed to **~1.0–1.7×**
against scx `0.11.0`'s framed ShufDeltaZstd.

### Compression — on-disk size (MB, lower is better) + ratio vs source h5ad

| dataset | scx_auto | scx_compact_trial | shardad | shardad vs best-scx |
|---|---:|---:|---:|---|
| pbmc3k | 4.8 (4.5×) | **4.5 (4.8×)** | 6.2 (3.5×) | scx 1.38× smaller |
| pbmc10k | 39.8 | **30.1 (6.7×)** | 31.7 (6.4×) | ~parity (scx 1.05×) |
| smartseq2 | 552.7 | 262.9 (4.1×) | **253.2 (4.2×)** | shardad 4% smaller |
| tabula_100k | 448.9 | 233.9 (6.8×) | **222.2 (7.1×)** | shardad 5% smaller |
| census_1m | 3989 | 2802 (4.1×) | **1622 (7.0×)** | **shardad 1.73× smaller** |
| replogle_k562 | 2345 | 2313 | **2237 (1.7×)** | shardad 3% smaller |
| tahoe_c38 | 1563 | 1516 | **1403 (1.6×)** | shardad 8% smaller |
| chemogenetic_rgfp | 1305 | 968 (7.6×) | **802 (9.1×)** | shardad 1.21× smaller |

shardad leads compression on 6/8 (by 3–42%); scx wins the two smallest. The
largest gap is census_1m (shardad 1.62 GB vs scx 2.80 GB). scx `compact_trial`
consistently beats `scx_auto` on integer counts (it is the codec to compare).

### Write speed (wall s, lower is better)

| dataset | scx_auto | scx_compact_trial | shardad |
|---|---:|---:|---:|
| pbmc10k | 1.82 | 2.38 | **0.86** |
| smartseq2 | 5.48 | 8.29 | **4.55** |
| tabula_100k | **4.99** | 5.47 | 6.30 |
| census_1m | 31.3 | 31.8 | **24.6** |
| replogle_k562 | 29.3 | 28.2 | **13.6** |
| tahoe_c38 | 13.2 | 13.6 | **9.6** |
| chemogenetic_rgfp | 21.5 | 22.6 | **16.3** |

shardad writes faster on 6/7 (multiprocessing + a single simple codec). **Caveat:**
scx write peak-RSS is *not* directly comparable — scx materializes the source
in-process — so only size + wall are reported here.

### Full read → AnnData (wall s / peak RSS MB)

| dataset | scx_auto | scx_compact_trial | shardad |
|---|---|---|---|
| pbmc10k | 0.38 / 338 | 0.67 / 340 | **0.24** / 334 |
| smartseq2 | **1.22** / 432 | 1.65 / 451 | 1.29 / **346** |
| tabula_100k | **0.87** / 514 | 1.14 / 569 | 1.74 / **347** |
| census_1m | 5.45 / 2317 | **4.80** / 2732 | 5.31 / **551** |
| replogle_k562 | 5.74 / 2904 | 5.84 / 2962 | **4.26 / 345** |
| tahoe_c38 | **3.39** / 1537 | 3.35 / 1660 | 3.71 / **428** |
| chemogenetic_rgfp | **2.78** / 772 | 3.70 / 1081 | 4.74 / **392** |

Read **speed** is roughly parity, dataset-dependent. The peak-RSS figures in this
table are the `read_full` 2-sample sampler's steady-state numbers (they miss the
transient assembly high-water mark; the true peak is in the `ooc_rss_boundary`
table below). scx's `to_anndata` now narrows **in-decode**: a
`data_dtype="uint16"` request assembles `X` (and `adata.raw`) directly at the
target width (2 B/nnz), never building the intermediate float32 CSR, so a narrow
read **lowers** peak RSS instead of raising it. Measured on the true-peak
`ooc_rss_boundary` boundary (below), the narrow `scx_materialize_u16` read
matches or beats shardad's uint16 materialize on pbmc10k, tabula, and
chemogenetic, and closes most of the remaining gap on census_1m (12.6 vs
11.6 GB — shardad additionally narrows its column indices, which scx keeps at
i32). scx's structural answer at the largest scale remains its streaming
accelerators, which never build full X — see below.

### Out-of-core streaming vs materialize — peak RSS (MB), the at-scale story

`ooc_rss_boundary`: scx streaming (bounded) vs scx full-materialize (f32) vs scx
full-materialize narrowed in-decode (uint16) vs shardad full-materialize. True
peak RSS via the background `PeakRssSampler`. Refreshed on the Phase-4 in-decode
narrow build (`scx_materialize_u16` = `to_anndata(data_dtype="uint16")`).

| dataset | scx_stream | scx_materialize (f32) | scx_materialize_u16 | shardad_materialize |
|---|---:|---:|---:|---:|
| pbmc10k | 924 | 556 | **508** | 527 |
| tabula_100k | 3567 | 2230 | **1887** | 1892 |
| tahoe_c38 | 4732 | 4238 | 4047 | **3844** |
| chemogenetic_rgfp | 14763 | 8869 | **7178** | 7471 |
| **census_1m** | **5630** | 15268 | 12605 | 11608 |

The **in-decode narrow** (`scx_materialize_u16`) lowers peak RSS vs the f32
materialize on every dataset (the value buffer drops from 4 B/nnz to 2 B/nnz with
no f32 intermediate) — enough to **match or beat** shardad's own uint16
materialize on pbmc10k, tabula, and chemogenetic. shardad stays lower on tahoe and
census because it also narrows its column indices (scx keeps i32) and uses a
tighter materialize layout; scx closes most of the census gap (12.6 vs 11.6 GB).

**At census_1m scale the picture inverts entirely: scx streaming (5.6 GB) is the
lowest by far — below shardad's materialize (11.6 GB) and well under either scx
materialize (15.3 GB f32 / 12.6 GB uint16).** This is scx's structural
out-of-core advantage: full-matrix approaches (shardad, and both scx materialize
modes) grow with dataset size, while scx streaming stays bounded — the gap widens
further at census_5m/10m (not run here).

> **Note — why `scx_materialize` here (15.3 GB) ≠ the "Full read → AnnData" peak
> above (~2.3 GB) for census_1m:** the two rows measure different things. The
> `read_full` benchmark's 2-sample sampler catches the *steady-state* resident
> AnnData; the `ooc_rss_boundary` `PeakRssSampler` catches the true high-water
> mark, which includes the transient decode/assembly buffers a narrow sampler
> misses (and the residue of the preceding `scx_stream` pass, run back-to-back in
> one process). Compare each column *within* its own table, not across the two.

### Selective / row-subset read (wall s)

Roughly parity; shardad edges ahead on census_1m (5.6 s vs scx_auto 13.1 s) via
whole-covering-row reads, scx competitive on the perturbation datasets. scx's
`filtered_query` predicate-pushdown sub-scenarios have **no shardad equivalent**
(see capability gaps).

### Grouped / condition-sharded reads — shardad's differentiator

shardad physically groups cells by an `obs` label; scx approximates it via `scx
sort --group-by` + query-time shard pruning. `grouped_read` (self-materialized
grouped fixtures):

| op | dataset | scx | shardad |
|---|---|---|---|
| `read_group` (one label) | replogle_k562 | **0.7 s / 1530 MB** | 4.1 s / 4179 MB |
| `read_group` | tahoe_c38 | **1.1 s / 1355 MB** | 3.9 s / 3238 MB |
| `read_group` | chemogenetic_rgfp | 0.8 s / 11744 MB | 0.9 s / **810 MB** |
| `iter_group_shards` (stream) | replogle_k562 | 3.2 s / **2368 MB** | 4.0 s / 9624 MB |
| `iter_group_shards` | tahoe_c38 | 5.3 s / **4580 MB** | 3.8 s / 6272 MB |
| grouped_write | replogle_k562 | 40.6 s / 2338 MB | **19.1 s** / 2238 MB |
| grouped_write | chemogenetic_rgfp | 33.8 s / 1176 MB | **21.9 s / 803 MB** |

scx's query-pruned targeted read is competitive-to-faster on latency (replogle,
tahoe) with lower RAM, but its RAM can spike on some layouts (chemogenetic
`read_group`); shardad writes grouped archives faster. Mixed — genuinely
dataset-dependent, with neither dominating.

### Round-trip fidelity

shardad round-trips correctly (`shardad_fidelity`: exact CSR match + `dense` /
`float16` in-decode materialization) on pbmc3k and tabula_100k — parity with scx's
own correctness suite.

### Capability comparison (documented — not head-to-head perf races)

Features with no meaningful two-sided benchmark, reported as capability presence:

| capability | scx | shardad |
|---|---|---|
| GPU on-device decode / analysis | ✅ (framed Scx1 decodes in VRAM; rapids) | ❌ unimplemented ([#40]; CPU-decode+upload only) |
| Lazy query engine / predicate pushdown | ✅ | ❌ (no query engine) |
| ML training loader | ✅ (1,405 batches/s) | ❌ |
| Streaming analysis accelerators (HVG/PCA/…) | ✅ (bounded RAM) | ❌ (materialize → scanpy) |
| Cloud-native reads (S3/GCS/Azure) | ✅ | ❌ |
| Multimodal (CITE-seq / Multiome) | ✅ | ❌ |
| R bindings (Seurat / SCE) | ✅ | ❌ (Python only) |
| In-place mutation (append/delete/compact/merge) | ✅ | metadata-tail only (`update_obs`) |
| In-decode dtype/density materialization | ✅ true in-decode narrow on the eager path (`data_dtype=`/`container=`/`index_dtype=`): `X`/`raw` assemble directly at target dtype, no f32 intermediate → lowers peak RSS (matches/beats shardad on 3/5 datasets); `>2²⁴` integer reads exact. (layers still post-assembly) | ✅ (true in-decode narrow, direct to target dtype; also narrows indices) |
| Physical group-aligned (condition) sharding | approximated (sort + query pruning) | ✅ (native) |

### Takeaways

- **Compression:** shardad still leads on large integer-count data (up to 1.7×),
  but scx `compact_trial` has closed the old 1.9–3.2× gap to ~1.0–1.7× and wins on
  small datasets.
- **Write:** shardad faster (simpler codec + multiprocessing).
- **Read speed:** parity, dataset-dependent.
- **Memory:** scx's in-decode narrow (`to_anndata(data_dtype="uint16")`) now
  matches or beats shardad's uint16 materialize on 3/5 datasets and closes most of
  the gap on the rest (shardad edges ahead where it also narrows indices); **scx
  streaming wins outright at census scale** (census_1m: 5.6 vs 11.6 GB) and is the
  only path that stays bounded as datasets grow.
- **Grouped access:** shardad's native physical grouping vs scx's query-pruning —
  mixed, neither dominates.
- **Breadth:** scx carries GPU, a query engine, an ML loader, streaming
  accelerators, cloud, multimodal, and R — none of which shardad targets.

Two honest conclusions: scx has reached rough **compression/read parity** with a
deep-narrow specialist on its home turf, and the projects' real difference is
scope (a broad platform vs a focused perturbation-screen tool), not a one-sided
performance gap. The clearest scx-specific perf win is **bounded out-of-core RAM
at scale**. On whole-matrix integer reads, scx's eager `to_anndata` now performs a
*true* in-decode narrow — each shard assembles directly into the target dtype's
full-matrix buffer, never allocating the intermediate f32 matrix — so a
`data_dtype="uint16"` read genuinely lowers peak RSS (matching or beating
shardad's uint16 materialize on 3/5 datasets above). shardad retains a small edge
where it also narrows column indices and uses a tighter materialize layout; scx
keeps i32 indices. Integer→integer narrows additionally decode from the native
`u32` stream, so a `>2²⁴` count read into `uint32`/`int64`/`float64` is now exact
(previously a hard error).

[#40]: https://github.com/ArcInstitute/shardad/issues/40
