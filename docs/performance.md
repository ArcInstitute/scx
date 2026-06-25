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

The streaming pipeline is currently sequential by design — wall is
flat (±1.5 %) across the whole range. Materialise scales sub-linearly
(1.69× peak at 32 threads, efficiency 5 %) because the upstream
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

Source: ``benchmarks/comprehensive/results/raw/read_streaming_vs_inmemory__scx_auto__{census_500k,census_1m}.json``
and ``benchmarks/comprehensive/results/raw/multimodal_read_streaming_vs_inmemory__scx_multimodal_per_modality_auto__{cite_seq_pbmc_5k,multiome_pbmc_10k}.json``.
SLURM wrapper: ``benchmarks/comprehensive/scripts/slurm_read_streaming_vs_inmemory.sh``.

## Analysis Accelerators (CPU)

Benchmarked on 1M cells (CELLxGENE Census), HVG-selected (2000 genes):

| Operation | SCX (s) | scanpy (s) | Speedup vs scanpy |
|-----------|---------|------------|-------------------|
| PCA (covariance, 50 PCs, 2K HVGs) | **4.2** | 8.0 | **1.9x** |
| Wilcoxon DE (pre-ranking) | **5.4** | 17.3 | **3.2x** |
| Leiden (Rust-native) | **55** | 2,226 (leidenalg) | **40x** |

Full pipeline (PCA -> kNN -> UMAP -> Leiden -> DE) on 1M cells: **870s** (vs 3,971s — **4.6x faster**).

### Differential expression (CPU, full-matrix)

The Wilcoxon DE row above is from an HVG-projected (2K genes) 1M-cell fixture. The dedicated `accel_de` benchmark sweeps the raw count matrix (no HVG projection) across the full dataset tier — scanpy's per-gene rank pass becomes the bottleneck and times out on census-scale:

| Dataset | scanpy `rank_genes_groups` | `pyscx.accel.rank_genes_groups` (CPU) | Speedup |
|---------|---:|---:|---:|
| pbmc3k (2.7K) | 1.16 s | 0.69 s | 1.7× |
| pbmc10k (12K) | 9.66 s | 5.85 s | 1.7× |
| smartseq2 (18K) | 45.96 s | 19.59 s | **2.3×** |
| tabula_sapiens_100k (62K) | 318.77 s | 42.28 s | **7.5×** |
| census_500k | timeout (≥ 55 min) | 106.13 s | **≥ 31×** |
| census_1m | timeout (≥ 55 min) | 156.51 s | **≥ 21×** |

`pyscx.accel.pdex_ref` (perturbation-screen Mann–Whitney U + pseudobulk geometric-mean log fold change, pinned bit-for-bit to upstream [`pdex`](https://github.com/ArcInstitute/pdex)) tracks similarly: 0.59 s on pbmc3k, 5.52 s on pbmc10k, 42.04 s on tabula_sapiens_100k, 119.86 s on census_500k, 268.28 s on census_1m. The CPU path uses gene-chunked dense materialisation (default `gene_chunk_size=500`) with rayon-parallel per-gene rank tests — peak RSS is `O(n_obs × gene_chunk_size)`, not `O(n_obs × n_vars)`. CPU numbers improved 20-40% vs the prior `v0.4.3-g1-gpu-de` baseline after the `pdex-unsorted-csr` fix (commit b423a2f).

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

LISI: `pyscx.accel.compute_lisi` is **~10× faster** than R `lisi::compute_lisi` on D1–D4 (e.g. smartseq2 3.85 s vs 43.11 s; tabula_sapiens_100k 12 s vs 110 s), with mean-LISI agreement within 0.8–2.4 % of the R reference.

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

Full per-operation results (wall time + peak RSS) are tracked in `benchmarks/comprehensive/results/raw/cell_eval_parity_perf__scx_auto__pert_synth_*.json` and rendered in the "Cell-eval / arc-bench Parity Performance" section of the comprehensive benchmark report. Kernel-level distance-kernel microbenchmarks live in `scx-accel/benches/distances.rs` (run via `cargo bench -p scx-accel --bench distances`; see [`benchmarks/README.md`](../benchmarks/README.md#rust-microbenchmarks-criterion)).

## GPU Acceleration (NVIDIA H100)

### Codec Decode and Training Pipeline

| Operation | Size | CPU (us) | GPU (us) | Speedup |
|-----------|------|----------|----------|---------|
| FOR-BP index decode | 16K rows, 33M nnz | 102,900 | 4,133 | **24.9x** |
| Sparse -> dense | 16K rows x 30K cols | 433,252 | 7,711 | **56.2x** |
| Sparse -> dense (HVG 2K) | 16K rows x 2K output | 110,416 | 897 | **123.1x** |

### GPU Analysis Pipeline

GPU-accelerated analysis via **rapids-singlecell** (`rsc.pp.pca`, `rsc.pp.neighbors`, `rsc.tl.umap`, `rsc.pp.*`) for in-VRAM ops, plus native Rust/CUDA paths for streaming PCA, HVG `seurat_v3`, Leiden (Rust-native CPU + cuGraph GPU), DE Wilcoxon/pdex (CSC/CSR-direct), Harmony, and codec decode. Benchmarked on H100 80GB (driver 560.35.05, CUDA 12.6, scx-bench-gpu conda env).

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
| Wilcoxon (vs `pyscx_cpu` reference) | pbmc10k | 5.85 | 12.60 | 0.47× | CUB block sort + searchsorted + tie + p-value |
| Wilcoxon (vs `pyscx_cpu`) | tabula_sapiens_100k | 42.28 | 89.77 | 0.47× | (same) |
| Wilcoxon (vs `pyscx_cpu`) | census_500k | 106.13 | 213.07 | 0.50× | (same) |
| Wilcoxon (vs `pyscx_cpu`) | census_1m | 156.51 | 365.15 | 0.43× | (same) |
| Wilcoxon (vs `scanpy_cpu`) | tabula_sapiens_100k | 318.77 | 89.77 | **3.6×** | (same) |
| pdex_ref (vs `pyscx_cpu`) | pbmc10k | 5.52 | 12.58 | 0.44× | (same) |
| pdex_ref (vs `pyscx_cpu`) | tabula_sapiens_100k | 42.04 | 93.81 | 0.45× | (same) |
| pdex_ref (vs `pyscx_cpu`) | census_500k | 119.86 | 212.37 | 0.56× | (same) |
| pdex_ref (vs `pyscx_cpu`) | census_1m | 268.28 | 356.34 | 0.75× | (same) |

The accel_de wilcoxon/pdex_ref GPU rows above are **slower than `pyscx_cpu`** (CPU's rayon-parallel implementation effectively uses ~3-5 of the 16 SLURM-allocated CPUs and is highly tuned). G10's graph capture closed ~7-10% of the gap but the GPU implementation is bottlenecked by the per-chunk `[n_obs × chunk_size]` dense materialization step. The GPU paths are still **3-5× faster than `scanpy_cpu`** — for users replacing scanpy directly, GPU is the clear win; for users who already have `pyscx.accel.rank_genes_groups(device="cpu")` working, the default GPU variant is a draw or worse.

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

**Which route ran is now recorded, and the gate asserts it.** Every `pdex_ref` call stamps its execution route on `adata.uns["scx_accel"]["pdex_ref"]` (`route ∈ {gpu_csc_v3, gpu_csr_v3, …}`, `fallback_reason`, `csc_available`), decided by the single planner `scx_accel::route::plan_de_route`. The `accel_de` benchmark reads this back into `runs[].extra` as `gpu_dispatch_route` (human-readable) and `de_route_csc_direct` (numeric: `0.0` only when a CSC fixture was built yet a non-`gpu_csc_v3` route ran — v3 being the unconditional default since Phase V1b). `thresholds.yaml` floors `de_route_csc_direct ≥ 1.0` for the GPU pdex_ref triple, so a *silent fallback to CSR while CSC-direct was intended* — exactly the prior benchmark misread — is now a hard gate failure rather than an invisible footgun. The structured `adata.uns` metadata is the signal (the former ad-hoc stderr trace was removed).

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

`pyscx.accel.{normalize_total, log1p, highly_variable_genes}` now accept `device="cpu|gpu|auto"`. The GPU path is eager (materializes to scipy CSR). **`log1p(device="gpu")` on a materialised scipy/dense X warns and falls back to CPU** — the H→D + kernel + D→H round-trip dominates log1p's trivial math. The pre-fallback measurement (retained as motivation):

| Op | pbmc3k CPU / GPU | tabula_sapiens_100k CPU / GPU | census_1m CPU / GPU |
|---|---|---|---|
| normalize_total | 0.004s / 0.004s (1.0×) | 0.61s / 0.43s (**1.4×**) | 3.27s / 3.00s (**1.1×**) |
| log1p (pre-fallback) | 0.003s / 0.41s (**0.01×**) | 0.20s / 9.23s (**0.02×**) | 1.46s / 63.78s (**0.02×**) |
| fused normalize+log1p | 0.006s / 0.41s (0.01×) | 0.83s / 9.73s (0.09×) | 4.50s / 67.45s (0.07×) |
| highly_variable_genes (seurat_v3) | 0.06s / 0.07s (0.9×) | 3.42s / 3.40s (1.0×) | 25.77s / 28.31s (0.9×) |

Practical recommendation: **use the GPU preprocessing path only via the `normalize_total → log1p` fusion-marker chain on backed SCX data, and only when the downstream consumer is also GPU**. The fused-chain optimization is the only case where GPU preprocessing doesn't round-trip through the host. Standalone `log1p(device="gpu")` on materialised X now emits a `UserWarning` and runs `sc.pp.log1p` instead; the GPU fast path is preserved when log1p sees the fusion marker planted by `normalize_total(device="gpu")`, or when X is still backed/lazy.

**Dispatch logic:** for an in-memory `X`, in-VRAM `pyscx.accel.pca(device="gpu")` routes to rapids-singlecell (`rsc.pp.pca`). The native GPU PCA path (backed/lazy/streaming inputs, or `SCX_FORCE_NATIVE_GPU=1`) is **always randomized** — the in-VRAM covariance core was removed in ACC-RUST-OPT-V4 Phase 3.2, so `method="covariance"` / `"auto"` resolve to randomized on GPU (covariance is still honored on the CPU path). The randomized path accepts `qr_method="householder"` (default, always-stable) or `"cholesky"` (CholeskyQR2 — opt-in, surfaces `RuntimeError` on non-SPD Gram so callers can retry with Householder).

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
| Wilcoxon (1-vs-rest) | pbmc3k (2.7K) | n_obs = 2.7K | 0.74 s | 0.92 s | 0.80× | launch-overhead bound |
| Wilcoxon (1-vs-rest) | pbmc10k → census_1m | n_obs > 8192 | 3.4 → 210 s | **skip** | — | v1 capacity cap |

The headline speedup is modest because v1 caps the per-gene sort pool at `GPU_DE_BLOCK_SORT_CAPACITY = 8192` cells — the CUB `BlockRadixSort` is one block per gene, holding the whole row in registers + shared memory. Above that, the dispatch returns `AccelError::InvalidInput("…use device='cpu' or subsample the reference")` and the caller falls back to the rayon-parallel CPU path. Where the GPU does run (small + medium datasets, mid-size reference groups), launch overhead and chunked-upload latency dominate the on-device sort + searchsorted work. The spec-anticipated **10–50× win** lives at Perturb-seq scale (≥ 50K cells × hundreds of perturbation groups, `n_ref` typically a few thousand non-targeting controls) — none of the dataset-tier fixtures match that group structure with the synthetic 2-way `groupby` the benchmark falls back to. Lifting the 8192 cap via a tiled merge-sort upgrade is deferred to PR series G4.

**Default v3-CSC path (PR series G4.3, 2026-05-27; promoted to default in Phase V1b).** A backed AnnData over an SCX file with a CSC sidecar (built via `pyscx.from_anndata(..., csc="always")` or `scx convert --csc=always`) routes `pdex_ref(device="gpu")` through a CSC-direct pseudobulk + scatter-to-gene-major kernel pair that drops the dense intermediate entirely. The CSC shard source is fully pipelined (2-slot pinned ring + dedicated copy stream + worker pre-decode) and pre-filters CSC shards by gene-chunk overlap using cheap catalog metadata, so non-overlapping shards are never decoded or uploaded. Bench (backed-AnnData, n_runs=3–5, median wall_s): **0.42 / 0.97 / 4.49 / 8.91 / 11.90 / 16.02 s** on pbmc3k / pbmc10k / smartseq2 / tabula_sapiens_100k / census_500k / census_1m respectively — **13–24× faster than v2-CSR GPU** and 17–22× faster than the rayon CPU implementation at atlas scale. In-memory inputs (no SCX file) automatically fall back to the v3-CSR-direct path. See the [`pdex_ref` GPU v3-CSC table](#per-operation-timing) above for the full numbers and the disposition. v3 is the unconditional default GPU DE route since Phase V1b.

**Correctness signal** (gated via `runs[].extra` in the `v0.4.3-g1-gpu-de` baseline):

| Metric | Threshold | Observed |
|---|---:|---:|
| `de_pval_agreement_vs_cpu` (mean Spearman ρ over shared (group × gene) p-values) | ≥ 0.999999 | 1.0 (pbmc3k, pbmc10k), 0.999999 (smartseq2) |
| `de_top_gene_overlap_vs_cpu` (median top-200 Jaccard per group) | ≥ 0.95 | 1.0 (pbmc3k, pbmc10k), 0.985 (smartseq2) |

Tolerance-based parity for p-values / FDR (not exact) because of `erfc` and sort-order numerics; U statistics agree exactly in f64. The CPU path itself is pinned bit-for-bit to upstream `pdex` via `pyscx/tests/test_pdex_ref_parity.py`, so CPU↔GPU parity here transitively pins the GPU path to the upstream oracle.

`pyscx.accel.rank_genes_groups(device="gpu", prefer_format="csc")` returns `RuntimeError` — CSC dispatch is CPU-only in v1.

#### Canonical baseline

Two baselines live side-by-side under `benchmarks/comprehensive/results/baselines/`. **Format / cloud / multimodal** PRs gate against the default `LATEST` symlink; **accel** PRs (PCA / kNN / UMAP / Leiden / preprocess / HVG / DE) pin the accel-only baseline explicitly. The split exists because the multi-surface baseline captures `accel_*` rows but doesn't produce gate signal against them — see [benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating).

| Use | Baseline | Date | Coverage |
|---|---|---|---|
| Format / cloud / multimodal | `LATEST` → `v0.6.2-n_counts-augmentation` | 2026-05-11 | 806 rows × 8 datasets (`pbmc3k` → `census_1m`, `cite_seq_pbmc`, `multiome_pbmc`) |
| Accel (incl. `accel_de`) | `v0.6.5-accel-gpu-rapids-floors` | 2026-06-XX | rapids-routed accel rows; cross-tier rapids route + correctness gates (`*_route_rapids_correct`, `*_fallback_no_rapids_correct`); Phase 2 promotion |

Per-run correctness metrics (`cosine_sim_min`/`mean`, `recall_vs_scanpy`, `trustworthiness`, `ari_vs_leidenalg`, `max_abs_diff_vs_scanpy`, `hvg_overlap_vs_scanpy`, plus `de_pval_agreement_vs_cpu` / `de_top_gene_overlap_vs_cpu` added in G1) flow through `runs[].extra` so the floor checks in `thresholds.yaml` evaluate real observed values, not `missing` placeholders.

```bash
# Format / cloud / multimodal — default LATEST:
python benchmarks/comprehensive/scripts/gate_candidate.py --no-accel

# Accel (incl. DE) — pin the accel-only baseline:
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only \
    --baseline benchmarks/comprehensive/results/baselines/v0.6.5-accel-gpu-rapids-floors
```

Older accel-only baselines (`v0.6.0-gpu-phase1-7`, `v0.6.0-gpu-phase1-7-multidataset`) remain in-tree for historical bisects but are no longer the gate targets. The earlier stop-gap wrappers (`benchmarks/scripts/gpu_regression_{diff,driver}.py` and `slurm_gpu_regression*.sh`) have been deleted; use `gate_candidate.py` for accelerator regression runs.

#### Changes vs previous version

- **Covariance-PCA dispatch path** on GPU (threshold `n_vars ≤ 8000`) — *historical, removed in ACC-RUST-OPT-V4 Phase 3.2.* The native in-VRAM covariance PCA core (`gpu_pca_covariance.rs`, `covariance_pca_gpu`, `GPU_COVARIANCE_PCA_THRESHOLD`) was deleted; in-VRAM PCA now routes to rapids `rsc.pp.pca`. The numbers below are from the pre-removal baseline: on tabula_sapiens_100k (HVG-shaped input) GPU PCA ran 1.7× vs CPU, up from 0.9× in the earlier baseline. On census_1m at the same n_vars, the speedup remained 0.9×. Native streaming/randomized PCA survives for >VRAM workloads.
- **Randomized PCA's critical path** now fully GPU-resident — the prior `Q → host → f64` SVD tail and per-iteration `d_m` download round-trip are gone (cuBLAS `sgemv` + `sgemm`). Correctness preserved (cosine ≥ 0.9999 on real data).
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

**Locality optimisations (1M cells, same fixture):**

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

#### Scattered pair-gather via the per-row decode sidecar (L1+L2)

For `streaming_mode=indexplan` (STATE_TX), each batch's `(pert, ctrl)` pairs are
scattered randomly across shards. The legacy gather **full-decodes every touched
shard** (`read_shard_cached_arc`) to extract a handful of rows; when the touched
shard set exceeds the byte-budgeted shard cache this thrashes — Census 1M
(62 shards) collapsed to **0.13 batches/s**. The sidecar gather routes the
scattered rows through `BackedCsrReader::read_rows_with`, which decodes the
touched rows **O(rows)** from the scx1 per-row **decode sidecar** instead of
O(shard) (L1), and a sidecar-aware prefetch (L2) skips warming sidecar-eligible
cold shards so the O(rows) path isn't negated by eager full-shard warms.

Measured (Lambda HPC, scx-bench pyscx 0.9.1, regenerated sidecar fixtures,
`lookahead=4`, `cache_shards=128`, `max_memory_mb=8192`; sidecar on = the
`scatter_sidecar` default, off = the L2 prefetch-skip disabled). Census 1M, where
shards exceed the cache budget → thrash:

| scenario | pairs/batch | sidecar batches/s | legacy batches/s | speedup | p50 latency (sidecar / legacy) |
|---|---|---|---|---|---|
| random   | 32  | 59.9 | 0.94 | **64×** | 11 ms / 256 ms |
| random   | 128 | 20.5 | 0.30 | **68×** | 44 ms / 4815 ms |
| locality | 2   | 170  | 17   | 10×   | 2.6 ms / 1.0 ms |
| locality | 128 | 20.1 | 0.58 | 35×   | 44 ms / 917 ms |

**Peak RSS drops** with the sidecar (no full-shard pool): Census 1M `bs=2`
**3.06 GB vs 14.9 GB** (~5× less); tabula_sapiens_100k `bs=2` 1.06 GB vs 3.40 GB
(3× less) — the streaming memory ceiling is preserved.

**Regime.** The *throughput* win requires cache-thrash (touched shards > cache
budget). When the shard set fits the cache (e.g. tabula's 7 shards), the
full-shard path warms once and is comparable or marginally faster at tiny batch
sizes — there the win is **memory**, which is universal. The per-dataset
`scatter_sidecar=false` kwarg (and the process-wide `SCX_SCATTER_SIDECAR=0`
kill-switch) restore the legacy path for the fits-cache regime.

**`SparseCellSetDataset` defaults the other way (`scatter_sidecar=false`).** The
cell-set loader serves a cache-*friendly* regime (sorted data + a reused
control-pool cache → a small working set that fits the shard cache and is touched
on most batches). There the sidecar's "skip the warm, decode O(rows) each batch"
strategy is a net loss: the hot shard is re-decoded every batch and the LRU never
populates. So `SparseCellSetDataset` defaults to the full-shard warm+cache path,
recovering ≈ `.h5ad` parity on a 50-file Tahoe atlas (steps/s 2.80 → 4.55, gather
337 → ~5 ms, cache populated to ~8 GB). Pass `scatter_sidecar=true` for a
genuinely cache-hostile cell-set run (working set ≫ cache), where the per-row
sidecar's bounded peak RAM is the memory-safe choice. The gate is per-reader: the
`IndexPlanDataset` defaults above are unchanged, and `SCX_SCATTER_SIDECAR=0`
remains a hard master kill-switch over both.

**Sidecars are a write-time property** (emitted by default for Scx1 integer-CSR
shards within a 25% overhead budget). `.scx` files written before the sidecar
writer (pre-0.9.x) carry none and silently fall back to full-shard — **regenerate
fixtures** (`scx info <f> --json` → `decode/*` sections should equal `n_csr_shards`
for integer-count files) before expecting the win. Sweep results:
`benchmarks/comprehensive/results/phase5/T5_sidecar_sweep.md`; driver
`benchmarks/scripts/phase5_sidecar_sweep.py`.

## Query Engine

| Metric | Result |
|--------|--------|
| Shard skip rate | **55%** average |
| Selective query | **4.2 ms** |
| vs AnnData subsetting | **2.1x** faster |

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
--index-obs cell_type`), so the format-version sidecars (decode metadata, sharded
obs) are present in both and don't confound the delta. The dominant cell types
drop from 3–4 shards to 1–2 contiguous shards; the 2.6× figure is conservative
because real atlas data is already partially clustered by cell type. The
synthetic worst-case (`bench_ops_sort_locality`: 40k cells, 8 categories cycled
across every shard) brackets the upper end at **10 → 2 shards (5×)**, and the gap
widens with shard count at atlas scale. In-memory sort of the 100k file ran in
~73 s at 2.7 GB peak RSS (debug build); the bounded-memory external strategy
(`--memory-budget`) caps peak RSS to one `new_pos`-range partition for atlas-scale
files.

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

SLAF (`slafdb==0.5.2`) is now a first-class competitor across every
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

All benchmark results now carry a `schema_version=1` stamp + full
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

The canonical baseline sits at
`benchmarks/comprehensive/results/baselines/v0.5.0-phase5/` (371 raw
JSONs archived, manifest + environment committed). `LATEST` symlink
makes on-demand gate runs (`gate_candidate.py`) work with no flags.
