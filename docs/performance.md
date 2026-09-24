# SCX Performance (moved)

The benchmark results now live in [`docs/performance/`](performance/README.md),
split into one page per area. Start with the [index](performance/README.md) or
the [performance quick start](performance/quickstart.md).

Where each section of the old single page went:

| Old section | New page |
|-------------|----------|
| Compression, Read Speed, Read Scaling, Column Projection, Selective Read | [storage-and-reads.md](performance/storage-and-reads.md) |
| Conversion, Write Scaling (incl. streaming conversion and export) | [conversion.md](performance/conversion.md) |
| Memory | [memory.md](performance/memory.md) |
| Analysis Accelerators (CPU): headline, stage profile, decode-prefetch, marshalling, graph layout, UMAP | [accel-cpu-pipeline.md](performance/accel-cpu-pipeline.md) |
| PCA decode-prefetch and reduction, pseudobulk aggregation | [accel-pca-pseudobulk.md](performance/accel-pca-pseudobulk.md) |
| QC / filtering, axis subsetting, DE, gene-set scoring, Harmony2 + LISI | [accel-qc-de-integration.md](performance/accel-qc-de-integration.md) |
| Perturbation Metrics | [perturbation-metrics.md](performance/perturbation-metrics.md) |
| GPU Acceleration, GPU DE device residency | [gpu.md](performance/gpu.md) |
| Training Loader (head-to-heads, out-of-core loader, data-wait) | [loader.md](performance/loader.md) |
| IndexPlanDataset | [loader-index-plan.md](performance/loader-index-plan.md) |
| Tier-1 loader fixes through the multi-set batch executor (phases 1–6) | [loader-cell-sets.md](performance/loader-cell-sets.md) |
| Data-load Phase 1 (1A–1D) | [loader-data-load.md](performance/loader-data-load.md) |
| Query Engine, File Operations | [query-and-file-ops.md](performance/query-and-file-ops.md) |
| Doublet-Caller Interop | [doublet-interop.md](performance/doublet-interop.md) |
| Comprehensive Benchmarking + Cloud Validation | [comprehensive-and-cloud.md](performance/comprehensive-and-cloud.md) |
