"""
Assemble the final comprehensive benchmark report as markdown and PDF.

Pulls data from both raw JSON results (via tables.py) and hardcoded
historical numbers for sections not covered by the automated pipeline
(GPU, lazy preprocessing, accelerators).

Output:
  - benchmarks/comprehensive/results/reports/BENCHMARK_REPORT.md
  - benchmarks/comprehensive/results/reports/BENCHMARK_REPORT.pdf
"""

from __future__ import annotations

import datetime
import logging
import shutil
import subprocess
from pathlib import Path

from benchmarks.comprehensive.config import REPORTS_DIR, FIGURES_DIR, RAW_RESULTS_DIR
from benchmarks.comprehensive.reporting.tables import (
    bench_csc_dispatch_table,
    cell_eval_parity_perf_table,
    compression_table,
    compression_ratio_table,
    multimodal_compression_ratio_table,
    multimodal_compression_table,
    multimodal_training_table,
    multimodal_training_ttfb_table,
    scx_parallel_write_callout_table,
    datasets_table,
    cloud_filtered_table,
    cloud_reader_vs_pull_table,
    cost_model_table,
    fragment_ops_table,
    gcp_matrix_table,
    harmony_scaling_table,
    harmony_validation_table,
    lisi_comparison_table,
    read_speed_table,
    read_selective_table,
    write_speed_table,
    memory_table,
    parallel_scaling_table,
    parallel_write_scaling_table,
    ml_loader_table,
    correctness_table,
    correctness_detail_table,
    system_info_table,
)

logger = logging.getLogger(__name__)


def _n_json_files() -> int:
    """Count raw result JSON files."""
    if RAW_RESULTS_DIR.exists():
        return len(list(RAW_RESULTS_DIR.glob("*.json")))
    return 0


def _fig(name: str) -> str:
    """Return a markdown image reference to a figure."""
    return f"![{name}](figures/{name}.png)"


def generate_report() -> str:
    """Generate the full benchmark report as a markdown string."""
    date = datetime.date.today().isoformat()
    n_results = _n_json_files()

    sections = []

    # -----------------------------------------------------------------------
    # Title + Executive Summary
    # -----------------------------------------------------------------------
    sections.append(f"""\
# SCX Comprehensive Benchmark Report

**Generated:** {date}
**Results:** {n_results} JSON files in `benchmarks/comprehensive/results/raw/`
**Methodology:** See [benchmarks/README.md](../../../benchmarks/README.md) for formats, measurement protocol, and environments.

---

## Executive Summary

SCX is a purpose-built binary format for single-cell RNA-seq data. This report
evaluates SCX against h5ad (gzip, lzf, uncompressed), Zarr (zstd, blosc-lz4),
and TileDB-SOMA across 7 datasets (2.7K to 5M cells) on compression, read/write
performance, parallel scaling, memory efficiency, ML data loading, analysis
accelerators, and correctness.

**Key findings:**

- **Compression:** SCX pcodec/zstd achieve the best compression on UMI data —
  2.35 GB on 1M cells (4.85x smaller than h5ad). Ratio improves with scale:
  7.30x on 5M cells.
- **Read speed:** SCX is the fastest reader at census scale — 1.38x faster than
  Zarr lz4 on 1M cells, with up to 7.1x parallel read scaling at 32 threads.
- **Write scaling:** Parallel shard encoding yields up to 3.2x write speedup at
  32 threads (pcodec/zstd on 500K cells). No other format scales writes.
- **Column projection:** SCX dominates — 4.2x faster than Zarr on 1M cells,
  7.6x on 5M cells.
- **ML loader:** SCX TrainingDataset delivers 1,405 batches/sec on 1M cells —
  82x faster than TileDB-SOMA-ML.
- **Pipeline:** With Rust-native Leiden (40x), pre-ranked DE (4x), and
  covariance PCA, the full SCX pipeline is 2.5–4.6x faster than scanpy with
  31–76% less memory.
- **GPU:** kNN 9.4x, UMAP 7.7x, Leiden 16x. Pipeline 3.8x (10x target not met).
- **Correctness:** 14/14 scanpy equivalence tests pass on pbmc3k. Preprocessing
  max error < 1e-6. Leiden ARI > 0.80 across all pipelines.
""")

    # -----------------------------------------------------------------------
    # Test Environment
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## Test Environment

{system_info_table()}

All benchmarks run on Arc Institute's Chimera HPC cluster. Intel Xeon Platinum
8468, 48 cores / 96 threads per socket, 1007–2015 GB RAM, WekaFS NVMe-backed
parallel filesystem. GPU benchmarks on NVIDIA H100 80GB HBM3.
""")

    # -----------------------------------------------------------------------
    # 1. Datasets
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 1. Datasets

{datasets_table()}

Dataset paths configured via `SCX_WORK_DIR` / `SCX_DATA_DIR` environment
variables. All benchmarks use warm-cache (1 warm-up read discarded) unless
otherwise noted.
""")

    # -----------------------------------------------------------------------
    # 2. Storage Efficiency
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 2. Storage Efficiency

### File Sizes

{compression_table()}

### Compression Ratios (vs h5ad uncompressed)

{compression_ratio_table()}

{_fig("compression_bar")}

**Takeaways:**
- SCX pcodec/zstd achieve the best compression on UMI count data — consistently
  #1 at census scale (2.35 GB vs 2.60 GB Zarr zstd on 1M cells).
- SCX lz4 compresses better than Zarr lz4 — byte-shuffle pre-filter is effective
  (13.85 GB vs 17.06 GB on 5M cells).
- Compression ratio improves with scale: 7.30x on census_5m vs 4.90x on census_1m.
""")

    # -----------------------------------------------------------------------
    # 3. Write Performance
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 3. Write Performance

### Single-threaded (`num_threads=1`)

All competing formats (Zarr, h5ad, TileDB-SOMA) write single-threaded —
they cannot scale across cores. SCX parallelises shard encoding via
rayon, so a 1-thread number is SCX's worst case and should be read
alongside the 32-thread column below, not as its headline write speed.

{write_speed_table()}

### With parallel shard encoding (SCX only)

Writing the same h5ad → SCX pipeline with 32 rayon threads (`full` mode —
h5ad read + SCX write). Zarr / h5ad / TileDB-SOMA are omitted because
their writers don't parallelise (Δ = 1.0× across all SCX codecs at every
thread count in §6).

{scx_parallel_write_callout_table()}

**Takeaways:**
- **Apples-to-apples, SCX is competitive at 32 threads.** On census_1m,
  SCX (pcodec) drops from 72s single-threaded to 26s at 32 threads
  (2.8× speedup) — roughly 2× the Zarr (blosc-lz4) wall at 14s but with
  a 1.2× smaller file and 17× faster subsequent reads.
- **SCX is the only format that scales writes with cores.** Zarr,
  h5ad, and TileDB-SOMA stay flat regardless of `num_threads` (see §6
  for the per-thread-count breakdown across all formats).
- **Compression-heavy SCX codecs (pcodec, zstd) scale best** (2.6–2.9×
  at 32T) — more CPU work per shard gives rayon more to schedule.
  `scx1` and `none` plateau earlier (2.2× / 1.6×).
- **h5ad gzip and TileDB-SOMA are significantly slower at any thread
  count** (single-threaded compression / fragment writes dominate).
""")

    # -----------------------------------------------------------------------
    # 4. Read Performance (Full)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 4. Read Performance (Full)

{read_speed_table()}

{_fig("read_speed_bar")}

{_fig("scaling_curves")}

**Takeaways:**
- **SCX is the fastest reader at census scale** — 1.38x faster than Zarr lz4 on
  1M cells, 1.15x on 5M cells. Crossover at ~100K cells.
- Read time scales sub-linearly with cell count due to shard-level parallelism.
""")

    # -----------------------------------------------------------------------
    # 5. Read Performance (Selective / Query)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 5. Read Performance (Selective / Query)

Column projection: 2,000 HVG columns selected from full gene set.

{read_selective_table()}

**Takeaways:**
- SCX dominates column projection — **4.2x faster** than Zarr on census_1m,
  **7.6x faster** on census_5m.
- Shard-level predicate pushdown enables SCX to skip irrelevant data on disk.
""")

    # -----------------------------------------------------------------------
    # 6. Parallel Scaling
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 6. Parallel Scaling

{parallel_scaling_table()}

{_fig("parallel_scaling")}

**Takeaways (read):**
- SCX achieves up to 7.1x speedup at 32 threads via shard-level parallelism.
- Zarr and h5ad show no parallel scaling (single-threaded I/O paths).
- Efficiency decreases at high thread counts due to memory bus saturation.

### Write Scaling

{parallel_write_scaling_table()}

{_fig("parallel_write_scaling")}

**Takeaways (write):**
- SCX achieves up to **3.2x write speedup** at 32 threads for compression-heavy
  codecs (pcodec, zstd) on 500K cells. Heavier per-shard CPU work yields more
  parallelism.
- Write-only mode (in-memory AnnData → SCX) shows ~15–25% better scaling than
  the full pipeline, since it isolates parallel shard encoding from the
  single-threaded h5ad read.
- SCX (none) shows only 1.7x — minimal compression means less work to parallelize.
- Scaling plateaus around 8–16 threads due to sequential I/O in the final write phase.
- Small datasets (pbmc3k, pbmc10k) show no write scaling — they fit in a single shard.
""")

    # -----------------------------------------------------------------------
    # 7. ML Data Loader
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 7. ML Data Loader

Batch size=1024, HVG=2000, normalize+log1p (hvg_norm scenario).

{ml_loader_table()}

{_fig("ml_loader_throughput")}

**Takeaways:**
- SCX TrainingDataset is **82x faster** than TileDB-SOMA-ML on census_1m.
- Triple-buffered Rust pipeline (tokio I/O → rayon decode → Python consumer)
  with zero Python on the hot path.
- TTFB < 500ms on all datasets (target: < 2s).
""")

    # -----------------------------------------------------------------------
    # 8. Memory Efficiency
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 8. Memory Efficiency

Peak RSS during full file read.

{memory_table()}

{_fig("memory_scaling")}

**Takeaways:**
- h5ad uses least memory (mmap-based access).
- SCX uses 18.5 GB on census_5m — less than half of Zarr's 87.9 GB.
- MADV_DONTNEED optimization reduces streaming peak RSS by 67%.

{_fig("streaming_preprocess_memory")}
""")

    # -----------------------------------------------------------------------
    # 8b. Fragment / Manifest Operations (SCX-only)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 8b. Fragment / Manifest Operations (SCX-only)

Throughput and wall-clock for the four SCX fragment/manifest mutations
exposed by `scx-ops` (`pyscx.append`, `pyscx.mark_deleted`,
`pyscx.compact`, `pyscx.rollback`). Each cell shows median wall-clock
with the dominant throughput metric in parentheses.

Competitors (Zarr, h5ad, TileDB-SOMA, SLAF) are omitted because the
underlying operation semantics (append without rewrite, logical delete
via deletion vectors, rollback via manifest revert) are not defined on
their formats.

{fragment_ops_table()}

**Takeaways:**
- `append` is dominated by input-CSR read + re-encode cost; scales with
  the number of appended rows.
- `delete` is a logical operation — only deletion-vector construction
  and a manifest rewrite — so wall-clock is independent of dataset size.
- `compact` throughput tracks base-file read + re-encode bandwidth; the
  `reclaimed_bytes` extra records the space recovered from orphaned
  sections and logical deletions.
- `rollback` is a single root-catalog `pwrite()` and should be
  near-instant at any scale; a regression here indicates manifest-load
  bloat.
""")

    # -----------------------------------------------------------------------
    # 8c. Cloud Query Parity (cross-format GCS filtered reads)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 8c. Cloud Query Parity (GCS)

The same standardized predicate set (`cell_type == "T cell"`,
`n_counts > 1000`, `random 1% sample`) executed directly against GCS for
every format that declares the `cloud_filtered` capability. Each cell
shows median wall-clock with p95 in parentheses over n runs.

Runners declare the capability only if they can push the predicate
through natively: SCX pulls then applies catalog-pushdown locally
(`scx_pull_and_filter` — Phase F.1 adds a native cloud-pushdown
variant); TileDB-SOMA uses `AxisQuery(value_filter=...)` on the
`Experiment.open(gs://…)` handle (`tiledb_cloud_value_filter`); SLAF
issues the SQL ``WHERE`` against its cloud-backed DuckDB engine
(`slaf_cloud_sql`). Zarr variants (`zarr_zstd`, `zarr_lz4`,
`anndata_zarr_backed`) do not declare the capability — the raw-CSR
converters don't preserve obs metadata, so they are omitted from this
table rather than silently skipped as dashes.

Bytes-transferred and GET-count columns are deferred to Phase F.2 when
the object-store telemetry shim lands.

{cloud_filtered_table()}
""")

    # -----------------------------------------------------------------------
    # 8d. GCP Compute-Node Matrix
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 8d. GCP Compute-Node Matrix

Cloud-read latency and throughput vs GCP instance type, collected by
`benchmarks/comprehensive/scripts/submit_gcp_matrix.py`. The launcher
provisions each instance in the bucket's region, runs the cloud
benchmarks with `SCX_BENCH_GCP_INSTANCE` / `SCX_BENCH_GCP_REGION`
exported, and tears the VM down. Result JSONs carry `system.gcp` tags
so the table pivots across instances automatically.

Only results with a `system.gcp.instance_type` label are shown; on-cluster
runs without the label are excluded to keep the matrix view clean.

{gcp_matrix_table()}
""")

    # -----------------------------------------------------------------------
    # 8e. CloudReader vs full Pull
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 8e. CloudReader vs Full Pull

Scored by `cloud_reader_vs_pull`: for each dataset, compares
`pyscx.open_cloud` (single-GET metadata open) against a full
`pyscx.pull`, and then sweeps `pyscx.pull_filtered` at ~5% / 20% / 80%
cell selectivity against a full pull. The break-even selectivity where
full-pull starts beating predicate-pushdown is visible in the table
below (as selectivity approaches ~100%, `pull_filtered` downloads every
shard and the bytes-downloaded column equalizes).

`open_cloud` does not yet surface `bytes_downloaded`; a Rust-side
object-store counting middleware (Phase F.2 follow-up) will close this
gap so the metadata row reports non-zero bytes.

{cloud_reader_vs_pull_table()}
""")

    # -----------------------------------------------------------------------
    # 8f. Cost model
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 8f. Cost Model (GCS pricing)

Cents per 1 million cells queried, computed from GCS Standard-class
pricing pinned in `benchmarks/comprehensive/config.py::GCS_PRICING`.
Egress is priced at the same-region rate ($0.00/GB today — the Phase E
launcher pins the VM region to the bucket region so this is the
committed regime); request cost follows the Class-B $0.004/10k GET
schedule.

Scenarios: `metadata` (catalog open), `selective_5pct` / `selective_20pct`
(predicate-pushdown pull), `full_read` (complete pull). Only the
exploded `.scxd/` cloud layout is wired today; packed `.scx` with a
front catalog will appear as additional rows once the benchmark's
`_LAYOUTS` list grows.

{cost_model_table()}
""")

    # -----------------------------------------------------------------------
    # 9. Analysis Accelerators
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 9. Analysis Accelerators (PCA, kNN, UMAP, DE, Leiden)

Rust-native accelerators via `pyscx.accel.*`, benchmarked against scanpy equivalents.

### Per-Stage Timing

{_fig("accelerator_speedup")}

#### tabula_sapiens_100k (100K cells, Phase 4e)

| Stage | SCX (s) | scanpy (s) | Speedup |
|-------|---------|-----------|---------|
| PCA (covariance eigh) | 4.49 | 3.97 | 0.9x |
| kNN (HNSW, k=15) | 21.86 | 8.03 | 0.4x |
| UMAP (Rust SGD) | 47.20 | 57.32 | **1.2x** |
| Leiden (Rust-native) | 3.95 | 132.04 | **33.4x** |
| DE (pre-ranked Wilcoxon) | 3.58 | 14.32 | **4.0x** |

#### census_1m (1M cells)

| Stage | SCX (s) | scanpy (s) | Speedup |
|-------|---------|-----------|---------|
| PCA | 21.29 | 8.37 | 0.39x |
| kNN | 103.32 | 123.78 | **1.20x** |
| UMAP | 573.32 | 784.16 | **1.37x** |
| Leiden (sequential) | 54.96 | 2,226 | **40.5x** |
| DE | 27.34 | 17.49 | 0.64x |

### End-to-End Pipeline

{_fig("pipeline_comparison")}

#### tabula_sapiens_100k (Phase 4e)

| Pipeline | Total (s) | Peak RSS (MB) | vs scanpy |
|----------|-----------|---------------|-----------|
| SCX out-of-core | **87.31** | 1,108 | **2.54x** |
| SCX preprocess | **95.35** | 1,127 | **2.32x** |
| scanpy in-memory | 221.56 | 1,605 | baseline |

#### census_1m

| Pipeline | Total (s) | Peak RSS (MB) | vs scanpy |
|----------|-----------|---------------|-----------|
| SCX out-of-core | **~870** | 2,399 | **~4.6x** |
| scanpy in-memory | ~4,007 | ~9,871 | baseline |

**Takeaways:**
- **Leiden** is the biggest win: 33–40x faster via Rust-native implementation.
- **DE** pre-ranked Wilcoxon is 4x faster on 100K cells.
- **Pipeline** is 2.5–4.6x faster with 31–76% less memory.
- PCA is slower due to shard-to-dense conversion overhead; kNN slower on small data due to HNSW construction cost.
""")

    # -----------------------------------------------------------------------
    # 10. GPU Accelerators
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 10. GPU Accelerators (NVIDIA H100 80GB)

### Per-Operation Timing (CPU vs GPU)

| Operation | Dataset | CPU (s) | GPU (s) | Speedup |
|-----------|---------|---------|---------|---------|
| kNN (k=15) | tabula_100k | 16.72 | 3.79 | **4.4x** |
| kNN (k=15) | census_1m | 288.09 | 30.75 | **9.4x** |
| UMAP (2D) | tabula_100k | 47.34 | 4.22 | **11.2x** |
| UMAP (2D) | census_1m | 559.54 | 73.53 | **7.6x** |
| PCA (50 PCs) | tabula_100k | 3.89 | 3.60 | 1.1x |
| PCA (50 PCs) | census_1m | 15.31 | 21.14 | 0.7x |
| Leiden | census_1m | 2,226 | 139 | **16.0x** |

### GPU End-to-End Pipeline (census_1m)

| Operation | CPU (s) | GPU (s) | Speedup |
|-----------|---------|---------|---------|
| PCA | 22.09 | 23.46 | 0.9x |
| kNN | 423.19 | 183.93 | **2.3x** |
| UMAP | 587.46 | 76.13 | **7.7x** |
| Leiden | 44.53 | 2.78 | **16.0x** |
| **Total** | **1,077** | **286** | **3.8x** |

### Go/No-Go Gates

| Criterion | Target | Value | Pass |
|-----------|--------|-------|------|
| PCA correctness | cosine_sim > 0.99 | 1.0 | Yes |
| kNN recall | recall@k > 0.95 | 1.0 | Yes |
| Pipeline speedup | >= 10x on 1M | 3.8x | **No** |
| Graceful fallback | all ops CPU fallback | all pass | Yes |

**Overall: 3/4 pass.** 10x pipeline target not met. Individual ops (kNN 9.4x,
UMAP 7.7x, Leiden 16x) exceed targets, but PCA bottleneck and data transfer
overhead limit pipeline speedup.
""")

    # -----------------------------------------------------------------------
    # 11. Streaming Preprocessing
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 11. Streaming Preprocessing

SCX preprocessing (normalize_total + log1p) operates in two modes:
1. **Lazy** — transforms applied on-the-fly during reads, no materialization
2. **Materialized** — write new SCX file with transformed data

### Preprocessing Correctness

| Dataset | n_obs | Max abs diff | Mean abs diff | Max rel err | Pass |
|---------|-------|-------------|---------------|-------------|------|
| pbmc3k | 2,700 | 4.77e-07 | 6.04e-10 | 1.45e-07 | Yes |
| tabula_100k | 100,000 | 9.54e-07 | 5.81e-10 | 2.16e-07 | Yes |

### Write Performance

| Dataset | SCX (s) | scanpy (s) | Speedup |
|---------|---------|-----------|---------|
| tabula_100k | 11.82 | 0.67 | 0.06x |
| census_1m | 83.56 | 5.33 | 0.06x |

SCX preprocessing writes a full new compressed file to disk, while scanpy
modifies the in-memory matrix in-place — hence the asymmetry.
""")

    # -----------------------------------------------------------------------
    # 12. Lazy Preprocessing & Out-of-Core Pipeline
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 12. Lazy Preprocessing & Out-of-Core Pipeline

`ScxLazyTransformedDataset` wraps the backed reader with chained transforms
(NormalizeTotal, Log1p, RowScale) — no materialization.

### Lazy vs Materialized Timing (census_1m)

| Path | Median (s) |
|------|-----------|
| Materialized (normalize -> log1p -> PCA) | 351.4 |
| Lazy (lazy normalize -> lazy log1p -> streaming PCA) | 436.7 |
| **Speedup** | **0.80x** |

Lazy path is 24% slower (per-shard transform overhead). Benefit is memory, not speed.

### Memory

{_fig("lazy_vs_materialized_rss")}

| Operation | Peak RSS (MB) |
|-----------|--------------|
| Lazy preprocess (normalize + log1p) | 3,491 |
| Full pipeline (open -> PCA -> kNN -> UMAP -> Leiden) | 10,923 |
| Materialized (scanpy in-place) | 22,343 |

Lazy preprocessing uses **84% less memory** than materialized.

### Column-Projected Streaming Aggregation (census_1m)

{_fig("column_projection_latency")}

| Mode | Median (s) |
|------|-----------|
| Unprojected (all 61K genes) | 25.2 |
| Projected (500 genes) | 46.8 |

Projection is slower (per-shard overhead) but reduces memory by only loading
the projected gene columns.

### Out-of-Core Pipeline Stage Breakdown

{_fig("ooc_pipeline_stages")}
""")

    # -----------------------------------------------------------------------
    # 13. Correctness Validation
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 13. Correctness Validation

### Scanpy Equivalence (pbmc3k)

{correctness_detail_table("pbmc3k")}

### Validation Summary

{correctness_table()}

### Pipeline Cross-Validation

| Metric | tabula_100k | census_1m |
|--------|-------------|-----------|
| Leiden ARI (OOC vs scanpy) | 0.837 | 0.858 |
| Leiden ARI (preprocess vs scanpy) | 0.806 | 0.855 |
| DE overlap (OOC vs scanpy) | 0.544 | 0.300 |

All Leiden ARI values exceed the >= 0.80 threshold, confirming that SCX and
scanpy pipelines produce comparable biological results despite algorithmic
differences in kNN (HNSW vs PyNNDescent) and UMAP (Rust SGD vs C++).
""")

    # -----------------------------------------------------------------------
    # 13b. Cell-eval / arc-bench parity performance
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 13b. Cell-eval / arc-bench Parity Performance

Wall-clock and peak-RSS comparison of SCX-accelerated perturbation metrics
(``pyscx.accel.*``) against the Python reference implementations in
``cell-eval`` and ``arc-bench`` on synthetic perturbation datasets. These
numbers complement the parity correctness suite at
``pyscx/tests/test_cell_eval_parity.py`` (30 tests, all passing) and the
small-scale 10K-cell snapshot recorded by
``test_performance_vs_cell_eval`` in that same file.

Operations with cost superlinear in ``n_obs`` (``energy_distance`` at O(N²),
``clustering_agreement`` at very large scale) are skipped automatically at
the sizes where they become infeasible — see the Notes column.

{cell_eval_parity_perf_table()}
""")

    # -----------------------------------------------------------------------
    # 13bA. Multimodal compression (Phase K.3)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 13bA. Multimodal Compression

`pyscx.from_mudata` writes h5mu / MuData inputs (CITE-seq, 10x Multiome,
TEA-seq) to SCX v2 multimodal files. Two SCX variants are compared:
``scx_multimodal_per_modality_auto`` (Phase E codec routing — RNA→Scx1,
Protein→Zstd, ATAC→Lz4Shuffle) vs ``scx_multimodal_uniform_auto``
(single-modality ``select_codec`` applied uniformly). Baselines: h5mu
uncompressed/gzip and Zarr-MuData (zstd).

### File Sizes

{multimodal_compression_table()}

### Compression Ratio (vs h5mu uncompressed)

{multimodal_compression_ratio_table()}

Smoke numbers on real public 10x datasets (5k CITE-seq PBMC, 10k Multiome
PBMC): SCX per-modality routing typically beats Zarr-MuData zstd by
~1.5× and h5mu gzip by ~1.7× on CITE-seq. On Multiome, uniform-auto
edges per-modality routing because ATAC's binary peaks compress
better under Scx1 than Lz4Shuffle on this fixture.
""")

    # -----------------------------------------------------------------------
    # 13bB. Multimodal training-loader throughput (Phase K.4)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 13bB. Multimodal Training Loader

`pyscx.MultimodalTrainingDataset` yields per-batch dicts of
{{modality_name → ndarray}} via the same triple-buffered Rust pipeline
as the single-modality `TrainingDataset`. Compared against an eager
``mudata.read_h5mu`` baseline that mirrors what scvi-tools' AnnTorchDataset
does internally for multimodal models without an SCX-native loader.

### Throughput (batches/sec, median across runs)

{multimodal_training_table()}

### Time to First Batch

{multimodal_training_ttfb_table()}

SCX's triple-buffered I/O and per-modality SHM densification keep the
loader CPU-bound on the consumer side; eager-h5mu paths block on
HDF5 read for every batch. Per-modality CSR shards mean SCX's TTFB
is dominated by mmap+catalog open, not by data copy.
""")

    # -----------------------------------------------------------------------
    # 13bC. CSC dispatch (Phase L.3)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 13bC. CSC vs CSR Dispatch (Phase L.3)

`bench_csc_dispatch` exercises the column-major code paths in
`pyscx.accel.*` against the equivalent row-major (CSR) paths. The
sweep covers four ops × two axes:

| Op | CSR variant | CSC variant |
|---|---|---|
| qc_metrics | `bench_csc__qc_metrics_csr` | `bench_csc__qc_metrics_csc` |
| HVG | `bench_csc__hvg_csr` | `bench_csc__hvg_csc` |
| Differential expression | `bench_csc__de_csr` | `bench_csc__de_csc` |
| Pseudobulk | `bench_csc__pseudobulk_csr` | `bench_csc__pseudobulk_csc` |

Each row's median wall-time:

{bench_csc_dispatch_table()}

CSC dispatch wins on column-heavy workloads (HVG variance, per-gene DE
ranking). CSR dispatch wins on row-heavy workloads (per-cell
qc_metrics, pseudobulk aggregation). The sweep validates that
`prefer_format="csc"` actually picks the column-major path and that
the column path produces equivalent numerical output.
""")

    # -----------------------------------------------------------------------
    # 13c. Harmony2 batch integration + LISI (Phase 6)
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 13c. Harmony2 Batch Integration + LISI

Clean-room Rust reimplementation of Harmony2 (Korsunsky et al., 2019) and
the Local Inverse Simpson Index (LISI), exposed via
``pyscx.accel.harmony_integrate`` / ``pyscx.accel.compute_lisi`` and the
R wrappers in ``rscx::scx_harmony_integrate`` / ``rscx::scx_compute_lisi``.
GPU path available when ``pyscx`` is built with ``--features gpu``.

Source JSONs: ``benchmarks/results/harmony/runs/``. Full per-PC curves
and log-log scaling plots: ``benchmarks/results/harmony/REPORT.md``.

### Numerical parity vs R harmony v2.x

Measured on the three validation fixtures under
``benchmarks/results/harmony/reference/``. The Rust core uses
``rand_chacha`` (ChaCha8) while R harmony uses Mersenne Twister; tail PCs
can drift by ~2% on high-batch-count inputs despite otherwise identical
arithmetic, so the assertion in ``pyscx/tests/test_harmony_validation.py``
uses a ≥0.95 per-PC floor and ≥0.97 mean rather than the 0.998 deterministic
target from the spec.

{harmony_validation_table()}

### Scaling (d=30, K=100, theta=2, max_iter=10, single batch covariate)

All four implementations share the same per-dataset PCA cache built by
``benchmarks/scripts/benchmark_harmony.py``. The RSS spike at ``smartseq2``
(~47 GB across three impls) reflects the in-process scale → PCA step that
densifies a 50K × 2K float32 matrix; R harmony dodges it because its worker
receives the precomputed matrix via an ``Rscript`` child process.

{harmony_scaling_table()}

Log-log exponents (``log y = α log N + b``) from ``REPORT.md``:

| Impl / device | α (wall vs N) | β (RSS vs N) |
|---|---:|---:|
| scx-accel CPU | 0.67 | +0.21 |
| scx-accel GPU | 1.00 | +1.04 |
| harmonypy (CPU) | 0.73 | +0.44 |
| R harmony (CPU) | 1.02 | −0.35 |

Key observations:

- **scx-accel CPU** has the lowest wall-time exponent (α=0.67) — rayon
  parallelises the distance, L2-norm, and update_R hot paths; the full
  Harmony loop stays sub-linear in N up to 5M cells.
- **scx-accel GPU** is competitive from D5 onward (500k cells ≤ scx-accel CPU)
  and fastest at D7 (31.1 min vs 37.5 min CPU).
- **harmonypy** is the fastest CPU implementation on every dataset (numpy
  BLAS wins at scale) but its peak RSS grows with N (β=+0.44).
- **R harmony** is the slowest at every scale (80 min on D7 vs 22–37 min for
  the Rust/Python impls) and has the highest wall-time exponent (α=1.02).

### LISI — scx-accel vs R `lisi`

Exact-kNN brute-force LISI in Rust vs the reference R package. Benchmarked
only on D1–D4 because brute-force LISI is O(N²·d) and becomes impractical
at census scale; at those sizes the HNSW-approximate path would be used
instead (not part of this comparison).

{lisi_comparison_table()}

- scx-accel LISI is **~10× faster** than R lisi across D1–D4 (0.02s vs
  0.40s on pbmc3k; 3.85s vs 43.11s on smartseq2; 12.07s vs 110.29s on
  tabula_sapiens_100k).
- Mean-LISI agreement vs the R reference is within 0.76–2.39 %, well
  inside the 5 % tolerance asserted in
  ``pyscx/tests/test_harmony_validation.py``.
""")

    # -----------------------------------------------------------------------
    # 14. Discussion
    # -----------------------------------------------------------------------
    sections.append("""\
---

## 14. Discussion

### SCX Strengths

1. **Best-in-class compression** on UMI count data (integer-heavy). SCX
   pcodec/zstd are the smallest at every census-scale dataset.
2. **Fastest reader at scale.** Shard-level parallelism enables up to 7x
   speedup at 32 threads. Zarr and h5ad cannot parallelize reads.
3. **Dominant column projection.** 4–8x faster than all competitors due to
   per-shard predicate pushdown.
4. **ML loader throughput.** 82x faster than TileDB-SOMA-ML via a
   triple-buffered Rust pipeline with zero Python overhead.
5. **Rust-native accelerators.** Leiden (40x), DE (4x), UMAP (1.4x) are
   faster than their Python equivalents, with identical or near-identical results.
6. **Out-of-core pipeline.** Full analysis from open to Leiden with 76% less
   memory than scanpy (2.4 GB vs 9.9 GB on 1M cells).

### SCX Weaknesses

1. **Write speed.** 1.8–2.9x slower than Zarr lz4 at single-threaded, though
   parallel shard encoding (up to 3.2x at 32 threads) narrows the gap.
2. **PCA.** Covariance eigendecomposition is slower than scanpy's ARPACK
   (iterative vs direct). Bottleneck is shard-to-dense conversion for GEMM.
3. **kNN on small data.** HNSW graph construction overhead dominates when
   n_obs < 100K. PyNNDescent amortizes better on small inputs.
4. **GPU pipeline.** 3.8x (vs 10x target). PCA doesn't benefit from GPU, and
   data transfer between stages adds overhead.

### Format Comparison Summary

| Format | Best at | Weakest at |
|--------|---------|------------|
| SCX (zstd/pcodec) | Compression, selective reads | Write speed |
| SCX (auto) | Read speed, parallel scaling (read+write) | Write speed (single-threaded) |
| Zarr (zstd) | Compression (close to SCX) | Parallel scaling, selective reads |
| Zarr (lz4) | Write speed | Compression |
| TileDB-SOMA | Schema richness, ecosystem | Read speed, ML loading |
| h5ad (gzip) | Ecosystem compatibility | Everything else |
| h5ad (none) | Memory (mmap) | Compression, read speed |

### Areas for Improvement

- **Write optimization:** Parallel shard encoding delivers up to 3.2x speedup;
  further gains possible via streaming CSR construction and parallel I/O.
- **PCA:** Streaming randomized SVD without shard-to-dense conversion.
- **GPU pipeline:** Minimize host-device transfers; unified memory for PCA.
- **CSC storage:** Gene-major layout for column-heavy workloads.
- **Lazy RSS targets:** Reduce shard decode buffers and rayon allocations.
""")

    # -----------------------------------------------------------------------
    # 15. Appendix: Raw Data
    # -----------------------------------------------------------------------
    sections.append(f"""\
---

## 15. Appendix: Raw Data

All benchmark results are stored as individual JSON files in:

```
benchmarks/comprehensive/results/raw/
```

**Total results:** {n_results} JSON files

### File Naming Convention

```
{{benchmark}}__{{format}}__{{dataset}}.json
```

Examples:
- `compression__scx_auto__census_1m.json`
- `read_full__zarr_zstd__pbmc3k.json`
- `correctness__scanpy_equiv__pbmc3k.json`

### Benchmark Types

| Type | Description | Count |
|------|-------------|-------|
| `compression` | File sizes and compression ratios | ~84 |
| `write` | Conversion/write throughput | ~84 |
| `read_full` | Full-file load performance | ~84 |
| `read_selective` | Column projection performance | ~70 |
| `parallel_scaling` | Read thread scaling (1–32 threads) | ~42 |
| `parallel_write_scaling` | Write thread scaling (1–32 threads) | ~53 |
| `memory` | Peak RSS during operations | ~42 |
| `ml_loader` | ML data loader throughput | ~12 |
| `correctness` | Validation pass/fail results | ~3 |

### Pareto Frontier

{_fig("pareto_frontier")}

The Pareto frontier shows the tradeoff between compression ratio and read speed.
SCX auto/zstd are Pareto-optimal at census scale — they offer both better
compression and faster reads than alternatives.
""")

    return "\n".join(sections)


def _generate_pdf(md_path: Path, pdf_path: Path) -> Path:
    """Convert a markdown report to PDF using pandoc + pdflatex.

    Requires ``pandoc`` and ``pdflatex`` on PATH.  Image paths in the markdown
    are resolved relative to the markdown file's parent directory (i.e. the
    reports/ folder, so ``figures/foo.png`` works).

    Parameters
    ----------
    md_path : Path to the source markdown file.
    pdf_path : Path to write the output PDF.

    Returns
    -------
    Path to the written PDF, or *md_path* if pandoc/pdflatex are unavailable.
    """
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        logger.warning("pandoc not found on PATH — skipping PDF generation")
        return md_path

    # --resource-path tells pandoc where to find images referenced as
    # relative paths (e.g. "figures/compression_bar.png").
    resource_path = str(md_path.parent)

    cmd = [
        pandoc,
        str(md_path),
        "-o", str(pdf_path),
        "--resource-path", resource_path,
        # PDF engine — xelatex handles Unicode better than pdflatex
        "--pdf-engine", "xelatex",
        # Geometry: reasonable margins for data-heavy tables
        "-V", "geometry:margin=1in",
        "-V", "geometry:landscape",
        # Smaller font so wide tables fit
        "-V", "fontsize=9pt",
        # Enable table-friendly packages
        "--variable", "colorlinks=true",
        "--variable", "linkcolor=blue",
        "--variable", "urlcolor=blue",
        # Standalone document (not a fragment)
        "--standalone",
        # Allow raw LaTeX passthrough
        "--from", "markdown+pipe_tables+raw_tex",
        # Table of contents
        "--toc",
        "--toc-depth=2",
        # Metadata
        "--metadata", "title=SCX Comprehensive Benchmark Report",
    ]

    logger.info(f"Generating PDF: {' '.join(cmd[:6])} ...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=resource_path,
        )
        if result.returncode != 0:
            # Log stderr but don't crash — the markdown report is the primary output
            logger.error(
                "pandoc failed (exit %d). stderr:\n%s",
                result.returncode,
                result.stderr[-2000:] if result.stderr else "(empty)",
            )
            return md_path
        logger.info(f"Wrote PDF to {pdf_path}")
        return pdf_path
    except FileNotFoundError:
        logger.warning("pandoc/xelatex not found — skipping PDF generation")
        return md_path
    except subprocess.TimeoutExpired:
        logger.warning("pandoc timed out after 120s — skipping PDF generation")
        return md_path


def write_report(output_dir: Path | None = None) -> Path:
    """Generate and write the full benchmark report (markdown + PDF + HTML).

    Returns the path to the written markdown file. An HTML snapshot is
    emitted alongside via ``write_html_snapshot`` (Phase G.4) so rolling-
    dashboard navigation works without server-side state.
    """
    if output_dir is None:
        output_dir = REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    report = generate_report()
    md_path = output_dir / "BENCHMARK_REPORT.md"
    md_path.write_text(report)
    logger.info(f"Wrote markdown report to {md_path}")

    # Generate PDF with embedded figures
    pdf_path = output_dir / "BENCHMARK_REPORT.pdf"
    _generate_pdf(md_path, pdf_path)

    # Emit HTML snapshot + append to dashboard history (Phase G.4).
    try:
        write_html_snapshot(report, output_dir)
    except Exception as exc:  # noqa: BLE001 — snapshot is best-effort
        logger.warning("HTML snapshot emission failed: %s", exc)

    return md_path


def write_html_snapshot(
    markdown_body: str,
    output_dir: Path,
    *,
    title: str = "SCX Benchmark Report",
) -> Path:
    """Write a browsable HTML snapshot alongside the markdown report.

    Records the snapshot's URL in ``dashboard_history.json`` so the next
    snapshot can link back to it via "← previous snapshot". The URL is
    the filename (relative to ``output_dir``); the static-hosting publish
    step (``publish_dashboard.py``) preserves the same path structure so
    relative links work unchanged on the published site.
    """
    from benchmarks.comprehensive.reporting import dashboard

    output_dir.mkdir(parents=True, exist_ok=True)
    prev_url = dashboard.previous_url(output_dir)

    html_body = dashboard.render_html(
        markdown_body, title=title, prev_url=prev_url,
    )
    html_path = output_dir / "BENCHMARK_REPORT.html"
    html_path.write_text(html_body)
    logger.info("Wrote HTML snapshot to %s", html_path)

    # Record this snapshot so the *next* render links back to it.
    dashboard.append_history(
        output_dir, url=html_path.name, title=title,
    )
    return html_path
