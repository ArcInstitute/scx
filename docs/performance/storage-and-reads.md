# Storage and reads

> Part of [SCX performance](README.md).


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

## See also

- [Conversion and write scaling](conversion.md).
- [Memory](memory.md) — what a read costs in RSS, eager vs streaming.
