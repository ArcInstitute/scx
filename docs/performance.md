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

| Path | Median wall | Peak RSS |
|---|---:|---:|
| streaming (`pyscx.from_h5ad`)        |  98.2 s | **851 MB** |
| materialise (`pyscx.from_anndata`)   |  60.2 s |  13.6 GB |

Streaming uses ~16× less peak RSS at ~63% wall-clock premium on
this size. Both paths emit byte-equivalent SCX (identical n_obs /
n_vars / nnz / 62-shard catalog).

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
RSS is thread-independent at ~13.6 GB.

**Crossover regime**: materialise wins on wall whenever the
in-memory CSR triplet fits comfortably; streaming becomes the only
viable path once that working set exceeds node RAM (somewhere above
the `census_1m` 13.6 GB working set; `census_5m` / `census_10m`
push it well past most workstation memory). Streaming also fits the
backed-AnnData path used by `pyscx.from_anndata(adata)` when
`adata.isbacked` is true.

Source data:
`benchmarks/comprehensive/results/raw/conversion_streaming__scx_streaming_vs_materialize__census_1m.json`
(absolute floor: `streaming_peak_rss_mb max: 2048` on `census_1m`,
declared in `benchmarks/comprehensive/thresholds.yaml`).

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
stream through `h5ad::write::write_dataframe_group_streaming` (pre-
allocated HDF5 datasets per column, hyperslab writes per shard,
single-pass running global dictionary for categoricals) so the bound
covers metadata too — atlas-scale obs no longer materialises during
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
The streaming row's `streaming_peak_rss_mb` is gated on `census_1m`
in `benchmarks/comprehensive/thresholds.yaml` at the same `2048 MB`
ceiling as the ingestion floor; a cross-shard accumulator leak in
`scx_to_h5ad_streaming` / `scx_to_h5mu_streaming` trips the floor
without needing a wall-clock signal.

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
Source: `BENCHMARK_REPORT.md` §12 (Lazy Preprocessing & Out-of-Core Pipeline).

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
(Σ/wall 220–275 %) because it sums each worker's decode time and those run concurrently;
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
  identical to before. It does **not** resize the ambient global rayon pool the many
  `current_num_threads()` callers use — that stays governed by `RAYON_NUM_THREADS`.

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

Numerical parity against R `harmony` v2.x (clean-room Rust implementation; validation fixtures + thresholds in `pyscx/tests/test_harmony_validation.py`):

| Dataset | N | Batches | d | K | mean per-PC Pearson r vs R | mean LISI agreement |
|---------|---:|---:|---:|---:|---:|---:|
| pbmc_small (D1) | 2,700 | 3 | 30 | 100 | **0.999** | within 5% |
| cell_lines (smartseq2, D3) | 9,478 | 47 | 20 | 100 | **0.989** | within 5% |
| hlca_subset (tabula_sapiens, D4) | 50,000 | 118 | 30 | 100 | **0.999** | within 5% |

The Rust RNG (`rand_chacha`) draws differ from R's Mersenne Twister, so tail PCs can deviate by up to ~2% on high-batch-count inputs (see `benchmarks/results/harmony/REPORT.md` for per-PC curves and wall/RSS scaling across D1–D7 for CPU scx-accel vs harmonypy vs R harmony).

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
| R harmony (CPU) | 0.95 | 1.20 |   compute-bound¹ |  compute-bound¹ |

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
strategy is a net loss: the hot shard is re-decoded every batch and the LRU never
populates. So `SparseCellSetDataset` defaults to the full-shard warm+cache path,
recovering ≈ `.h5ad` parity on a 50-file Tahoe atlas (steps/s 2.80 → 4.55, gather
337 → ~5 ms, cache populated to ~8 GB). Pass `scatter_block_index=true` for a
genuinely cache-hostile cell-set run (working set ≫ cache), where the
row-group-scoped decode's bounded peak RAM is the memory-safe choice. The gate is
per-reader: the `IndexPlanDataset` defaults above are unchanged, and
`SCX_SCATTER_BLOCK_INDEX=0` remains a hard master kill-switch over both.

**Framing is a write-time property** (v4 writers row-group-frame all shards by
default; codec-agnostic, so it covers every codec, not just Scx1). `.scx` files
written before framing (pre-v4) carry no `BlockIndex` and silently fall back to
full-shard — **regenerate fixtures** (or `scx optimize` in place; `scx info <f>
--json` reports framed shards) before expecting the win. Sweep results:
`benchmarks/comprehensive/results/phase5/T5_sidecar_sweep.md`; driver
`benchmarks/scripts/phase5_sidecar_sweep.py`.

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
`grouped_sort` benchmark (release pyscx, 16 threads; wall = median, RSS sampled
post-op):

| source (X format, cells × genes) | one-pass wall / RSS | two-pass wall / RSS | `auto` route |
|---|---|---|---|
| `tahoe_c38` (CSR, 69,245 × 62,710, by `drug`) | **16.9 s / 1.0 GB** | 27.9 s / 1.4 GB | one-pass (1.7× faster) |
| `replogle_k562` (dense, 68,729 × 6,546, by `gene`) | 3 m 39 s / 6.8 GB | **39 s / 4.7 GB** | two-pass (5.6× faster) |

For a **CSR** source the one-pass gather reads only each row's non-zeros and wins
outright; for a **dense** source it reads full-width rows per gathered cell and is
a ~5× loss, so `auto` falls back to the two-pass path (which costs a transient
~2× output disk for the intermediate file). Output is byte-identical regardless of
route. Both synthetic CSR fixtures confirm the ordering (`pert_synth_10k`:
one-pass 1.6 s vs two-pass 3.4 s; `nb_glm_synth`: 3.6 s vs 6.5 s). The post-op RSS
above understates true peak (the suite samples after the op, not a high-water
mark); a `/usr/bin/time` true-peak measurement of the release `scx` CLI puts the
dense one-pass at ~11.5 GB vs ~6.5 GB for two-pass — the memory motivation for the
dense → two-pass route.

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
`grouped_sort` `peak_rss_mb` gross-regression ceiling gates each dataset in
`thresholds.yaml` (wall time stays measured-not-gated — hardware-sensitive).

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
| In-decode dtype/density materialization | ✅ true in-decode narrow on the eager path (`data_dtype=`/`container=`/`index_dtype=`): `X`/`raw` assemble directly at target dtype, no f32 intermediate → lowers peak RSS (matches/beats shardad on 3/5 datasets); `>2²⁴` integer reads exact. (query path + layers still post-assembly) | ✅ (true in-decode narrow, direct to target dtype; also narrows indices) |
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
