# SCX Performance

Benchmark results for SCX across compression, read/write, memory, analysis accelerators, GPU, training loader, and query engine.

All benchmarks on Intel Xeon Platinum 8468, 32 cores, 1-2 TB RAM unless noted otherwise. GPU benchmarks on NVIDIA H100 80GB HBM3.

Results are split into pages by area. New here? The
[performance quick start](quickstart.md) maps common questions to the page
that answers them and explains how to reproduce a number.

## Getting started

- [Performance quick start](quickstart.md) — where to find a figure, and how
  every figure is captured, backed, and gated.

## Data on disk and conversion

| Page | What it covers |
|------|----------------|
| [Storage and reads](storage-and-reads.md) | Compression, full-load read speed, parallel read scaling, column projection, and predicate-pushdown selective reads |
| [Conversion and write scaling](conversion.md) | h5ad → SCX conversion, parallel shard encoding, rewrite ops, streaming conversion and streaming export |
| [Memory](memory.md) | Out-of-core pipeline memory and streaming vs in-memory read iteration |

## Analysis

| Page | What it covers |
|------|----------------|
| [CPU accelerators: the streaming pipeline](accel-cpu-pipeline.md) | Headline CPU accelerator numbers, the stage profile, decode-prefetch, marshalling and GIL hygiene, graph layout, UMAP determinism |
| [CPU accelerators: PCA and pseudobulk](accel-pca-pseudobulk.md) | PCA streaming decode-prefetch, the covariance-route reduction, and pseudobulk aggregation partitioning |
| [CPU accelerators: QC, DE, scoring, integration](accel-qc-de-integration.md) | QC / filtering fusion, axis subsetting, differential expression, gene-set scoring, Harmony2 + LISI — mostly vs scanpy |
| [Perturbation metrics](perturbation-metrics.md) | cell-eval parity metrics and the pairwise-distance kernels |
| [GPU acceleration](gpu.md) | Codec decode, the GPU analysis pipeline, GPU DE device residency, and go / no-go status |

## ML training loader

| Page | What it covers |
|------|----------------|
| [Training loader](loader.md) | Random-gather loader head-to-heads, ShufDeltaZstd decode cost, out-of-core cold-cache measurements, and data-wait fractions |
| [IndexPlanDataset](loader-index-plan.md) | Plan-driven paired reads |
| [Cell-set gathers](loader-cell-sets.md) | Gather pre-sizing, the bounded reader registry, tokenisation kernels, neighbourhood plans, and the multi-set batch executor |
| [Data-load phase 1](loader-data-load.md) | Shard-cache sizing, obs categorical codes, count-depth downsampling, and global pre-shuffle |

## Operations, cloud, and comparisons

| Page | What it covers |
|------|----------------|
| [Query engine and file operations](query-and-file-ops.md) | Modality-scoped pushdown, sort, and grouped sharding |
| [Doublet-caller interop](doublet-interop.md) | Round-trip fidelity, accuracy, and agreement of imported doublet calls |
| [Comprehensive benchmarking and cloud](comprehensive-and-cloud.md) | SLAF parity, cloud throughput and queries, cost model, fragment ops, and regression gating |

## Related

- [docs/benchmark_manifest.md](../benchmark_manifest.md) — how every performance
  claim is backed by a reproducible capture.
- [benchmarks/README.md](../../benchmarks/README.md) — running benchmarks on
  SLURM and gating a candidate against the baseline.
- [docs/testing.md](../testing.md) — test and correctness-validation details.
