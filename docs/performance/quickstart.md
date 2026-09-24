# Performance quick start

> Part of [SCX performance](README.md).

How to find a number on these pages, and how to trust or reproduce it.

## Where is the number I want?

| Question | Page |
|----------|------|
| How much smaller is an SCX file than h5ad? | [Compression](storage-and-reads.md#compression) |
| How fast does a whole file load into AnnData? | [Read speed](storage-and-reads.md#read-speed-full-load-to-anndata) |
| How long does converting my h5ad take, and how much memory? | [Conversion and write scaling](conversion.md) |
| Can I run the whole scanpy pipeline on a machine smaller than the data? | [Out-of-core pipeline memory](memory.md#out-of-core-pipeline-memory) |
| How much faster are the `pyscx.accel` ops than scanpy? | [Analysis accelerators (CPU)](accel-cpu-pipeline.md#analysis-accelerators-cpu) and [QC, DE, scoring, integration](accel-qc-de-integration.md) |
| What does a GPU buy me? | [GPU acceleration](gpu.md) |
| Will the training loader keep my GPU busy? | [Training loader](loader.md#training-loader) |
| What does a filtered query over GCS cost? | [Cost model](comprehensive-and-cloud.md#cost-model--usd-per-1m-cells-queried-gcs-same-region-pricing) |
| How does SCX compare with shardad? | [scx vs shardad](vs-shardad.md) |

## How the numbers are produced

- **Hardware.** Unless a section says otherwise: Intel Xeon Platinum 8468,
  32 cores, 1-2 TB RAM; GPU results on an NVIDIA H100 80GB HBM3.
- **Every published claim is backed by a capture.** A figure on these pages
  must trace to a tracked raw result under
  `benchmarks/comprehensive/results/`; the rules are in
  [docs/benchmark_manifest.md](../benchmark_manifest.md). Two tables (streaming
  conversion and streaming export) are pinned to their raw JSON by tests in
  `benchmarks/comprehensive/tests/test_floor_reachability.py`, so a stale figure
  there fails the suite.
- **Medians, not minima.** Where a section reports repeated runs, the published
  figure is the median; a few sections say explicitly when they differ.

## Reproducing a number

Benchmarks run on SLURM, one job per benchmark × dataset pair. The practical
guide — dataset preparation, conda environments, submission scripts — is
[benchmarks/README.md](../../benchmarks/README.md). To check a change against the
canonical baseline:

```bash
python benchmarks/comprehensive/scripts/gate_candidate.py --tier small
```

See [benchmarks/README.md § Regression Gating](../../benchmarks/README.md#regression-gating)
and [Regression gating](comprehensive-and-cloud.md#regression-gating) for what the gate enforces
(relative tolerances, absolute floors, disappeared-benchmark detection).
