# SCX Performance

Benchmark results for SCX across compression, read/write, memory, analysis accelerators, GPU, training loader, and query engine.

All benchmarks on Intel Xeon Platinum 8468, 32 cores, 1-2 TB RAM unless noted otherwise. GPU benchmarks on NVIDIA H100 80GB HBM3. Full raw results in [`benchmarks/results/`](../benchmarks/results/) and [`benchmarks/comprehensive/reporting/phase3_report.md`](../benchmarks/comprehensive/reporting/phase3_report.md).

---

## Compression

| Dataset | Cells | h5ad -> SCX | vs Zarr+Zstd |
|---------|-------|-------------|--------------|
| PBMC 3K | 2,700 | **4.9x** smaller | 2% smaller |
| Smart-seq2 | 50,000 | **2.9x** smaller | 5% smaller |
| Tabula Sapiens | 100,000 | **4.9x** smaller | 11% smaller |
| Census 1M | 1,000,000 | **4.8x** smaller | 10% smaller |
| Census 5M | 5,000,000 | **7.3x** smaller | 7% smaller |

## Read Speed (full load to AnnData)

| Dataset | SCX (auto) | h5ad (none) | h5ad (gzip) | Zarr (lz4) | TileDB-SOMA |
|---------|-----------|-------------|-------------|------------|-------------|
| PBMC 10K | 0.31s | 0.12s | 0.94s | **0.08s** | 0.44s |
| Tabula Sapiens 100K | **0.58s** | 1.41s | 7.56s | 1.20s | 3.79s |
| Census 1M | **2.74s** | 5.89s | 48.5s | 3.99s | 12.9s |
| Census 5M | **35.4s** | 43.7s | 291s | 40.6s | 80.6s |

SCX is the fastest reader at census scale — **1.5x faster than Zarr**, **2.1x faster than uncompressed h5ad**, and **17.7x faster than gzip h5ad** on 1M cells. Parallel read scaling: up to **7x** at 32 threads.

## Write Scaling (parallel shard encoding)

Write-only mode (in-memory AnnData → SCX, 500K cells):

| Codec | 1 thread | 32 threads | Speedup |
|-------|---------|-----------|---------|
| SCX (pcodec) | 36.1s | 11.4s | **3.2x** |
| SCX (zstd) | 35.9s | 11.4s | **3.2x** |
| SCX (scx1) | 32.3s | 12.0s | 2.7x |
| SCX (auto) | 32.3s | 12.7s | 2.5x |
| SCX (none) | 21.3s | 12.5s | 1.7x |

SCX parallelizes shard encoding via rayon — compression, checksumming, and statistics run on separate threads. Heavier codecs (pcodec, zstd) benefit most from parallel encoding. Write scaling plateaus around 8–16 threads due to sequential I/O.

## Column Projection (2000 HVGs)

| Dataset | SCX | h5ad (none) | Zarr (lz4) | TileDB-SOMA |
|---------|-----|-------------|------------|-------------|
| Tabula Sapiens 100K | **0.55s** | 0.86s | 0.94s | 1.31s |
| Census 1M | **3.53s** | 33.6s | 7.24s | 10.0s |
| Census 5M | **9.79s** | 94.1s | 63.9s | 66.3s |

SCX excels at gene selection — **2x faster than Zarr** and **9.6x faster than h5ad** on 1M+ cells. Column projection returns a backed/lazy dataset without materializing.

## Memory

Peak RSS during full read (lower is better):

| Dataset | h5ad (none) | SCX (auto) | Zarr (zstd) |
|---------|-------------|------------|-------------|
| PBMC 10K | 0.48 GB | 1.51 GB | 0.76 GB |
| Tabula Sapiens 100K | 0.53 GB | 2.27 GB | 2.08 GB |
| Census 1M | 0.72 GB | 6.64 GB | 11.5 GB |
| Census 5M | 1.04 GB | 18.5 GB | 87.7 GB |

For streaming aggregation (row_sums, col_sums), `MADV_DONTNEED` reduces SCX peak RSS by **67%** — from 3.5 GB to 1.1 GB on Census 1M.

### Out-of-Core Pipeline Memory

Full lazy pipeline (open -> QC filter -> normalize -> log1p -> HVG -> PCA -> kNN -> UMAP -> Leiden) on 1M cells:

| | Full materialization | SCX lazy pipeline | Reduction |
|--|----------------------|-------------------|-----------|
| Peak RSS | 43.6 GB | **5.1 GB** | **88%** |

## Analysis Accelerators (CPU)

Benchmarked on 1M cells (CELLxGENE Census), HVG-selected (2000 genes):

| Operation | SCX (s) | scanpy (s) | Speedup vs scanpy |
|-----------|---------|------------|-------------------|
| PCA (covariance, 50 PCs, 2K HVGs) | **4.2** | 8.0 | **1.9x** |
| Wilcoxon DE (pre-ranking) | **5.4** | 17.3 | **3.2x** |
| Leiden (Rust-native) | **55** | 2,226 (leidenalg) | **40x** |

Full pipeline (PCA -> kNN -> UMAP -> Leiden -> DE) on 1M cells: **870s** (vs 3,971s — **4.6x faster**).

## GPU Acceleration (NVIDIA H100)

### Codec Decode and Training Pipeline

| Operation | Size | CPU (us) | GPU (us) | Speedup |
|-----------|------|----------|----------|---------|
| FOR-BP index decode | 16K rows, 33M nnz | 102,900 | 4,133 | **24.9x** |
| Sparse -> dense | 16K rows x 30K cols | 433,252 | 7,711 | **56.2x** |
| Sparse -> dense (HVG 2K) | 16K rows x 2K output | 110,416 | 897 | **123.1x** |

### GPU Analysis Pipeline

GPU-accelerated analysis via cuSPARSE, cuSOLVER, cuVS CAGRA, native CUDA UMAP kernel, and cuGraph Leiden. Benchmarked on H100 80GB with 1M cells:

| Operation | CPU (s) | GPU (s) | Speedup | Backend |
|-----------|---------|---------|---------|---------|
| kNN (k=15, 50 PCs) | 288 | 31 | **9.4x** | cuVS CAGRA |
| UMAP (2D) | 560 | 74 | **7.6x** | native CUDA SGD |
| Leiden | 45 | 3 | **16.0x** | cuGraph |
| PCA (50 PCs, 2K HVGs) | 22 | 24 | 0.9x | cuSPARSE SpMM |
| **End-to-end pipeline** | **1077** | **286** | **3.8x** | all above |

GPU PCA streams shards from disk -> GPU SpMM shard-by-shard without materializing the full matrix — enabling PCA on datasets larger than VRAM.

### Go/No-Go Status

| Gate | Criterion | Result |
|------|-----------|--------|
| PCA correctness | cosine similarity > 0.99 | **Pass** |
| kNN recall | recall@15 > 0.95 | **Pass** |
| Graceful fallback | CPU fallback when no GPU | **Pass** |
| 10x pipeline speedup | end-to-end 10x vs CPU | **Fail** (3.8x achieved) |

Full GPU benchmark details in [`benchmarks/results/gpu_pipeline_benchmark.md`](../benchmarks/results/gpu_pipeline_benchmark.md).

## Training Loader

Batches/sec, batch_size=1024, HVG=2000, normalize+log1p:

| Dataset | SCX | AnnData | TileDB-SOMA-ML | scDataLoader | SCX/SOMA |
|---------|-----|---------|----------------|--------------|----------|
| Census 1M | **1,405** | 16.3 | 17.1 | 4.4 | **82x** |
| Tabula Sapiens 100K | **1,060** | 14.5 | 16.1 | 4.0 | **66x** |
| PBMC 3K | **168** | 14.5 | 5.0 | 6.3 | **34x** |

Triple-buffered pipeline (tokio I/O -> rayon decode -> Python) with native HVG projection and fused normalize+log1p delivers **34-82x higher throughput** than TileDB-SOMA-ML at scale. TTFB (time to first batch): 16 ms on PBMC 3K, 603 ms on Census 1M.

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
