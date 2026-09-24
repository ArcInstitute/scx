# Conversion and write scaling

> Part of [SCX performance](README.md).

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
- **`auto` picks `scx1` for UMI data**, so its scaling profile matches `scx1` (median-value-based heuristic — see [docs/codec.md](../codec.md#8-automatic-codec-selection)).

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
[benchmark_manifest.md](../benchmark_manifest.md) admits no exception for a claim
of that shape in this file. They live in
[multithreading.md § Across shards, in a rewrite op](../multithreading.md#across-shards-in-a-rewrite-op-scx-optimize)
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
[docs/benchmark_manifest.md](../benchmark_manifest.md#for-readmedocs-authors)):
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
> [docs/benchmark_manifest.md](../benchmark_manifest.md#for-readmedocs-authors)
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

## See also

- [Storage and reads](storage-and-reads.md) — the size and read speed of the
  files these conversions produce.
- [docs/scanpy/conversion.md](../scanpy/conversion.md) — how to convert.
