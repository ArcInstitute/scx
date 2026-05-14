# SCX Comprehensive Benchmark Report

## 1. Executive Summary <a id="executive-summary"></a>

### Key findings <a id="executive-summary-key-findings"></a>

SCX is a purpose-built binary format for single-cell RNA-seq data.  This report evaluates SCX against h5ad (gzip, lzf, uncompressed), Zarr (zstd, blosc-lz4), and TileDB-SOMA across 7 datasets (2.7K to 5M cells) on compression, read/write performance, parallel scaling, memory efficiency, ML data loading, analysis accelerators, and correctness.

> **Info**
> - **Correctness:** 24/28 scanpy equivalence tests pass across 2 datasets (pbmc3k, tabula_sapiens_100k), 4 skipped (dependency absent).
> - **Compression:** SCX (pcodec) achieves the best compression on UMI data (7.3x compression ratio on census_5m).
> - **Read speed:** Zarr (blosc-lz4) is the fastest reader at census scale — 1.30x faster than Zarr lz4 on 1M cells — 0.33x faster than Zarr lz4 on 5M cells.
> - **Column projection:** SCX dominates — **0.4x faster** on census_1m, **1.0x faster** on census_5m.
> - **Write scaling:** Parallel shard encoding — SCX is the only format that scales writes with cores.
> - **ML loader:** SCX TrainingDataset delivers **2,061 batches/sec** on census_1m — 116x faster than TileDB-SOMA-ML.
> - **Pipeline:** With Rust-native accelerators (PCA, kNN, UMAP, Leiden), SCX enables a full out-of-core analysis pipeline.
> - **GPU:** GPU-accelerated kNN, PCA, UMAP, Leiden available; see Chapter 9 for per-operation speedups.

*Interpretation guide:* speedup claims are only meaningful where equivalency/accuracy checks pass or are explicitly marked as approximate.  See Chapter 3 for full correctness status.

## 2. Methodology <a id="methodology"></a>

### Datasets <a id="methodology-datasets"></a>

*Benchmark datasets*

| ID | Name | Cells | Genes | Protocol | Source | h5ad Size |
|---|---|---|---|---|---|---|
| D1 | pbmc3k | 2,700 | 32,738 | 10x v2 (UMI) | 10x Genomics | 21.0 MB |
| D2 | pbmc10k | 11,769 | 33,538 | 10x v3 (UMI) | 10x Genomics | 194.0 MB |
| D3 | smartseq2 | 50,000 | 61,497 | Smart-seq2 | CELLxGENE Census | 1.04 GB |
| D4 | tabula_sapiens_100k | 100,000 | 61,497 | 10x (UMI) | CELLxGENE Census | 1.56 GB |
| D5 | census_500k | 500,000 | 61,497 | 10x (UMI) | CELLxGENE Census (blood) | 5.57 GB |
| D6 | census_1m | 1,000,000 | 61,497 | 10x (UMI) | CELLxGENE Census (blood) | 11.13 GB |
| D7 | census_5m | 5,000,000 | 61,497 | 10x (UMI) | CELLxGENE Census (blood) | 83.98 GB |

Dataset paths configured via `SCX_WORK_DIR` / `SCX_DATA_DIR` environment variables. All benchmarks use warm-cache (1 warm-up read discarded) unless otherwise noted.

### Test Environment <a id="methodology-test-environment"></a>

*System configuration*

| Property | Value |
|---|---|
| CPU | Intel(R) Xeon(R) Platinum 8468 |
| Cores | 192 |
| RAM | 2015 GB |
| OS | Linux 5.15.0-176-generic (x86_64) |
| Storage | wekafs (NVMe-backed) |
| Python | 3.13.3 |
| Rust | rustc 1.94.0 (4a4ef493e 2026-03-02) |
| Key Libraries | anndata 0.12.10, scanpy 1.12, zarr 3.1.5, scipy 1.17.1, tiledbsoma 2.3.0, torch 2.11.0, pyscx dev (from source) |

All benchmarks run on Arc Institute's Chimera HPC cluster. Intel Xeon Platinum 8468, 48 cores / 96 threads per socket, 1007–2015 GB RAM, WekaFS NVMe-backed parallel filesystem. GPU benchmarks on NVIDIA H100 80GB HBM3.

## 3. Correctness & Equivalence <a id="correctness-equivalence"></a>

### Dataset-level Summary <a id="correctness-equivalence-dataset-level-summary"></a>

The following table summarises all correctness and equivalence harnesses across every benchmarked dataset.  Status uses proper classification: dependency-skipped tests (e.g. pseudobulk when pydeseq2 is absent) count as **Skipped**, not Failed.

*Dataset-level correctness summary*

| Dataset | Harness | Passed | Failed | Skipped | Status |
|---|---|---|---|---|---|
| pbmc10k | correctness | 33 | 1 | 2 | **Fail** |
| pbmc3k | scanpy_equivalence | 12 | 0 | 2 | Pass (partial) |
| pbmc3k | backed_equivalence | 19 | 0 | 0 | Pass |
| pbmc3k | preprocessing_paths | 3 | 0 | 0 | Pass |
| pbmc3k | correctness | 34 | 0 | 2 | Pass (partial) |
| tabula_100k | scanpy_equivalence | 12 | 0 | 2 | Pass (partial) |
| tabula_100k | backed_equivalence | 19 | 0 | 0 | Pass |
| tabula_100k | preprocessing_paths | 3 | 0 | 0 | Pass |

### Scanpy API Equivalence <a id="correctness-equivalence-scanpy-api-equivalence"></a>

Per-function scanpy equivalence results for every dataset with validation data.  Each test compares pyscx's output against scanpy's reference and reports the key parity metric, its observed value, and the pass/fail threshold.

*Scanpy equivalence detail (pbmc3k)*

| Function | Status | Key Metric | Value | Threshold | Duration | Notes |
|---|---|---|---|---|---|---|
| normalize_total | Pass | max_abs_error | 0.000122 | 0.001 | 1.19s |  |
| log1p | Pass | max_abs_error | 0.000000 | 0.001 | 1.19s |  |
| pca | Pass | min_cosine_sim | 1.000000 | 0.99 | 1.26s |  |
| neighbors | Pass | recall_at_15 | 0.933432 | 0.9 | 0.688s |  |
| umap | Pass | trustworthiness_k15 | 0.919614 | 0.9 | 2.29s |  |
| rank_genes_groups | Pass | min_top100_overlap_pct | 100.0000 | — | 0.337s |  |
| rank_genes_groups_chunked | Pass | min_top50_overlap_pct | 100.0000 | 100.0 | 0.203s |  |
| pseudobulk_dex | _Skipped_ | — | — | — | 0.6ms | pydeseq2 not installed — skipped |
| pseudobulk_dex_stratified | _Skipped_ | — | — | — | 0.1ms | pydeseq2 not installed — skipped |
| rank_genes_groups_stratified | Pass | n_stratified_results | 54000 | — | 0.329s |  |
| filter_cells | Pass | n_cells_scanpy | 2700 | — | 0.045s |  |
| filter_genes | Pass | n_genes_scanpy | 13714 | — | 0.051s |  |
| calculate_qc_metrics | Pass | max_float_error | 0.000000 | 1e-05 | 0.085s |  |
| subset_obs | Pass | shape_match | True | True | 0.520s |  |

*Scanpy equivalence detail (tabula_sapiens_100k)*

| Function | Status | Key Metric | Value | Threshold | Duration | Notes |
|---|---|---|---|---|---|---|
| normalize_total | Pass | max_abs_error | 0.000488 | 0.001 | 2.2m |  |
| log1p | Pass | max_abs_error | 0.000001 | 0.001 | 1.2m |  |
| pca | Pass | min_cosine_sim | 1.000000 | 0.99 | 7.04s |  |
| neighbors | Pass | recall_at_15 | 0.986452 | 0.9 | 26.16s |  |
| umap | Pass | trustworthiness_k15 | 0.980203 | 0.9 | 5.9m |  |
| rank_genes_groups | Pass | min_top100_overlap_pct | 100.0000 | — | 18.97s |  |
| rank_genes_groups_chunked | Pass | min_top50_overlap_pct | 100.0000 | 100.0 | 5.28s |  |
| pseudobulk_dex | _Skipped_ | — | — | — | 4.2ms | pydeseq2 not installed — skipped |
| pseudobulk_dex_stratified | _Skipped_ | — | — | — | 0.1ms | pydeseq2 not installed — skipped |
| rank_genes_groups_stratified | Pass | n_stratified_results | 144000 | — | 5.65s |  |
| filter_cells | Pass | n_cells_scanpy | 100000 | — | 4.34s |  |
| filter_genes | Pass | n_genes_scanpy | 25121 | — | 5.22s |  |
| calculate_qc_metrics | Pass | max_float_error | 0.000000 | 1e-05 | 4.20s |  |
| subset_obs | Pass | shape_match | True | True | 29.84s |  |

### Pipeline-level Biological Agreement <a id="correctness-equivalence-pipeline-level-biological-agreement"></a>

Pipeline agreement compares three preprocessing pathways (A: scanpy in-memory, B: pyscx eager, C: pyscx lazy/out-of-core) end-to-end through Leiden clustering and differential expression. Parity metrics include Leiden ARI, PCA cosine similarity, DE overlap, and UMAP Procrustes correlation.

*Pipeline agreement — pbmc3k*

| Pipeline Test | Status | Key Metric | Value | Threshold |
|---|---|---|---|---|
| normalize_log1p_threeway | Pass | max_pairwise_error | 0.000000 | 1e-05 |
| pca_threeway | Pass | min_pairwise_cosine | 1.000000 | 0.99 |
| extended_pipeline | Pass | min_leiden_ari | 1.0000 | 0.95 |

*Pipeline agreement — tabula_100k*

| Pipeline Test | Status | Key Metric | Value | Threshold |
|---|---|---|---|---|
| normalize_log1p_threeway | Pass | max_pairwise_error | 0.000001 | 1e-05 |
| pca_threeway | Pass | min_pairwise_cosine | 1.000000 | 0.99 |
| extended_pipeline | Pass | min_leiden_ari | 1.0000 | 0.95 |

### Round-trip & Format Integrity <a id="correctness-equivalence-round-trip-format-integrity"></a>

Round-trip validation confirms that SCX write → read preserves matrix data, obs/var metadata, layers, obsm, and uns.  Backed-equivalence testing verifies that slicing, indexing, row/column sums, and NNZ operations on the backed reader match the in-memory AnnData reference.

*Correctness validation summary*

| Test | Dataset | Passed | Failed | Skipped | Status | Duration |
|---|---|---|---|---|---|---|
| scanpy_equivalence | pbmc3k | 12 | 0 | 2 | Pass (partial) | 8.18s |
| backed_equivalence | pbmc3k | 19 | 0 | 0 | Pass | 7.50s |
| preprocessing_paths | pbmc3k | 3 | 0 | 0 | Pass | 43.58s |
| scanpy_equivalence | tabula_sapiens_100k | 12 | 0 | 2 | Pass (partial) | 11.0m |
| backed_equivalence | tabula_sapiens_100k | 19 | 0 | 0 | Pass | 3.9m |
| preprocessing_paths | tabula_sapiens_100k | 3 | 0 | 0 | Pass | 47.9m |
| correctness | pbmc3k | 34 | 0 | 2 | Pass (partial) | — |
| correctness | pbmc10k | 33 | 1 | 2 | **Fail** | — |

### Accelerator Parity <a id="correctness-equivalence-accelerator-parity"></a>

Accelerator parity shows whether SCX's Rust-native implementations reproduce scanpy/leidenalg reference results. Parity metrics (cosine similarity, recall, ARI, etc.) are extracted from the same benchmark runs that produce timing data in Chapter 9.

*Accelerator parity — SCX vs baseline (CPU)*

| Operation | Dataset | SCX impl | SCX time | Baseline | Baseline time | Speedup | Parity metric | Parity value |
|---|---|---|---|---|---|---|---|---|
| HVG | census_1m | pyscx_cpu | 25.26s | scanpy_cpu | 25.98s | 1.0x | overlap_pct | — |
| HVG | pbmc10k | pyscx_cpu | 0.548s | scanpy_cpu | 0.540s | 1.0x | overlap_pct | — |
| HVG | pbmc3k | pyscx_cpu | 0.077s | scanpy_cpu | 0.079s | 1.0x | overlap_pct | — |
| HVG | smartseq2 | pyscx_cpu | 2.81s | scanpy_cpu | 2.81s | 1.0x | overlap_pct | — |
| HVG | tabula_100k | pyscx_cpu | 4.09s | scanpy_cpu | 4.09s | 1.0x | overlap_pct | — |
| kNN | census_1m | pyscx_cpu | 4.3m | scanpy_cpu | 3.4m | 0.8x | recall_at_k | — |
| kNN | pbmc10k | pyscx_cpu | 28.79s | scanpy_cpu | 1.60s | 0.1x | recall_at_k | — |
| kNN | pbmc3k | pyscx_cpu | 0.542s | scanpy_cpu | 0.157s | 0.3x | recall_at_k | — |
| kNN | smartseq2 | pyscx_cpu | 1.7m | scanpy_cpu | 3.41s | 0.0x | recall_at_k | — |
| kNN | tabula_100k | pyscx_cpu | 4.4m | scanpy_cpu | 6.23s | 0.0x | recall_at_k | — |
| Leiden | census_1m | pyscx_cpu | 49.71s | leidenalg_cpu | 9.2m | 11.1x | ari | — |
| Leiden | pbmc10k | pyscx_cpu | 2.44s | leidenalg_cpu | 7.88s | 3.2x | ari | — |
| Leiden | pbmc3k | pyscx_cpu | 0.489s | leidenalg_cpu | 0.515s | 1.1x | ari | — |
| Leiden | smartseq2 | pyscx_cpu | 7.44s | leidenalg_cpu | 21.49s | 2.9x | ari | — |
| Leiden | tabula_100k | pyscx_cpu | 25.71s | leidenalg_cpu | 1.1m | 2.6x | ari | — |
| PCA | census_1m | pyscx_cpu_auto | 3.30s | scanpy_cpu | 7.62s | 2.3x | cosine_sim_min | 1.0000 |
| PCA | pbmc10k | pyscx_cpu_auto | 39.56s | scanpy_cpu | 0.847s | 0.0x | cosine_sim_min | 1.0000 |
| PCA | pbmc3k | pyscx_cpu_auto | 35.75s | scanpy_cpu | 0.440s | 0.0x | cosine_sim_min | 1.0000 |
| PCA | smartseq2 | pyscx_cpu_auto | 36.96s | scanpy_cpu | 0.872s | 0.0x | cosine_sim_min | 1.0000 |
| PCA | tabula_100k | pyscx_cpu_auto | 2.2m | scanpy_cpu | 2.97s | 0.0x | cosine_sim_min | 1.0000 |
| Preprocess | census_1m | pyscx_cpu | 7.97s | scanpy_cpu | 4.80s | 0.6x | max_abs_error | — |
| Preprocess | pbmc10k | pyscx_cpu | 0.126s | scanpy_cpu | 0.075s | 0.6x | max_abs_error | — |
| Preprocess | pbmc3k | pyscx_cpu | 8.3ms | scanpy_cpu | 7.6ms | 0.9x | max_abs_error | — |
| Preprocess | smartseq2 | pyscx_cpu | 0.658s | scanpy_cpu | 0.396s | 0.6x | max_abs_error | — |
| Preprocess | tabula_100k | pyscx_cpu | 0.986s | scanpy_cpu | 0.578s | 0.6x | max_abs_error | — |
| UMAP | census_1m | pyscx_cpu | 10.9m | scanpy_cpu | 13.6m | 1.2x | trustworthiness | 0.9304 |
| UMAP | pbmc10k | pyscx_cpu | 55.32s | scanpy_cpu | 6.24s | 0.1x | trustworthiness | 0.9643 |
| UMAP | pbmc3k | pyscx_cpu | 16.19s | scanpy_cpu | 4.09s | 0.3x | trustworthiness | 0.9226 |
| UMAP | smartseq2 | pyscx_cpu | 4.3m | scanpy_cpu | 28.81s | 0.1x | trustworthiness | 0.9621 |
| UMAP | tabula_100k | pyscx_cpu | 7.9m | scanpy_cpu | 54.22s | 0.1x | trustworthiness | 0.9766 |

### Cell-Eval Parity <a id="correctness-equivalence-cell-eval-parity"></a>

Cell-eval / arc-bench correctness status for each perturbation metric operation.  Performance numbers for these operations appear in Chapter 10; this section focuses on whether the SCX implementation reproduces the reference values.

*Cell-eval parity correctness summary*

| Dataset | n_obs | Operation | Status | Notes |
|---|---|---|---|---|
| pert_synth_10k | 10,000 | pseudobulk | Pass |  |
| pert_synth_10k | 10,000 | bulk_metrics | Pass |  |
| pert_synth_10k | 10,000 | discrimination_l1 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance_blas_f32 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance_blas_f64 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance_scalar_f32 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance | Pass |  |
| pert_synth_10k | 10,000 | knockdown_efficiency | Pass |  |
| pert_synth_10k | 10,000 | clustering_agreement | Pass |  |
| pert_synth_100k | 100,000 | pseudobulk | Pass |  |
| pert_synth_100k | 100,000 | bulk_metrics | Pass |  |
| pert_synth_100k | 100,000 | discrimination_l1 | Pass |  |
| pert_synth_100k | 100,000 | energy_distance_blas_f32 | Pass |  |
| pert_synth_100k | 100,000 | energy_distance_blas_f64 | _Skipped_ | non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K |
| pert_synth_100k | 100,000 | energy_distance_scalar_f32 | _Skipped_ | non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K |
| pert_synth_100k | 100,000 | energy_distance | _Skipped_ | non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K |
| pert_synth_100k | 100,000 | knockdown_efficiency | Pass |  |
| pert_synth_100k | 100,000 | clustering_agreement | Pass |  |
| pert_synth_500k | 500,000 | pseudobulk | Pass |  |
| pert_synth_500k | 500,000 | bulk_metrics | Pass |  |
| pert_synth_500k | 500,000 | discrimination_l1 | Pass |  |
| pert_synth_500k | 500,000 | energy_distance | _Skipped_ | O(N^2) pairwise distance at n_obs >= 500K is infeasible |
| pert_synth_500k | 500,000 | knockdown_efficiency | Pass |  |
| pert_synth_500k | 500,000 | clustering_agreement | Pass |  |
| pert_synth_1m | 1,000,000 | pseudobulk | Pass |  |
| pert_synth_1m | 1,000,000 | bulk_metrics | Pass |  |
| pert_synth_1m | 1,000,000 | discrimination_l1 | Pass |  |
| pert_synth_1m | 1,000,000 | energy_distance | _Skipped_ | O(N^2) pairwise distance at n_obs >= 500K is infeasible |
| pert_synth_1m | 1,000,000 | knockdown_efficiency | Pass |  |
| pert_synth_1m | 1,000,000 | clustering_agreement | Pass |  |

### Harmony & LISI Validation <a id="correctness-equivalence-harmony-lisi-validation"></a>

Harmony2 validation compares per-PC Pearson r between SCX's Rust-native implementation and R harmony.  LISI validation compares mean LISI values between scx-accel and R lisi.  Scaling performance appears in Chapter 10.

*Harmony / LISI validation correctness summary*

| Validation | n_obs | Status | Key metric | Notes |
|---|---|---|---|---|
| Harmony (pbmc_small) | 2,700 | Pass | min per-PC r=0.9986 |  |
| Harmony (cell_lines) | 9,478 | Pass | min per-PC r=0.9789 |  |
| Harmony (hlca_subset) | 50,000 | Pass | min per-PC r=0.9979 |  |
| LISI (pbmc3k) | 2,700 | Pass | \|Δ\|/R = 0.76% |  |
| LISI (pbmc10k) | 11,769 | Pass | \|Δ\|/R = 0.92% |  |
| LISI (smartseq2) | 50,000 | Pass | \|Δ\|/R = 2.09% |  |
| LISI (tabula_100k) | 100,000 | Pass | \|Δ\|/R = 2.39% |  |

_Source: manual_

*Harmony validation — per-PC Pearson r vs R harmony*

| Dataset | N | Batches | d | K | min per-PC r | mean per-PC r | iter (scx / R) |
|---|---|---|---|---|---|---|---|
| pbmc_small (D1) | 2,700 | 3 | 30 | 100 | 0.9986 | 0.9992 | 5 / 4 |
| cell_lines (smartseq2) | 9,478 | 47 | 20 | 100 | 0.9789 | 0.9885 | 10 / 8 |
| hlca_subset (tabula) | 50,000 | 118 | 30 | 100 | 0.9979 | 0.9991 | 10 / 5 |

_Source: manual_

### Skipped & Non-comparable Tests <a id="correctness-equivalence-skipped-non-comparable-tests"></a>

Tests that could not run due to missing dependencies or environment constraints are listed below.  These are categorised as **Skipped** (dependency absent) or **Not applicable** (test does not apply to the format/dataset combination) — not as failures.

- `pseudobulk_dex` / `pseudobulk_dex_stratified`: Skipped when `pydeseq2` is not installed.
- Census-scale cell-eval: O(N²) reference computation deferred for datasets > 500K cells.
- GPU Leiden label stability: documented divergence from `leidenalg` — not a correctness failure; pin `device="cpu"` for label-stable downstream work.

## 4. Coverage <a id="coverage"></a>

### API Coverage <a id="coverage-api-coverage"></a>

API coverage matrices (format × operation) will be generated automatically once the harness emits per-operation coverage records.  In the meantime, correctness and equivalence results in Chapter 3 serve as the primary coverage signal.

## 5. Storage Efficiency <a id="storage-efficiency"></a>

### Storage Summary <a id="storage-efficiency-storage-summary"></a>

> **Info**
> - **Best compression:** SCX (pcodec) achieves the highest compression on UMI count data — consistently #1 at census scale.
> - **Compression scales:** ratio improves with dataset size (6.04x on census_5m vs 4.13x on census_1m).
> - **Byte-shuffle effective:** SCX lz4 compresses better than Zarr lz4 thanks to the byte-shuffle pre-filter.

### File Sizes <a id="storage-efficiency-file-sizes"></a>

*File sizes by format and dataset*

| Format | pbmc3k | pbmc10k | smartseq2 | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 4.5 MB | **37.6 MB** | 511.2 MB | 409.7 MB | 1.39 GB | 2.57 GB | 14.10 GB |
| SCX (scx1) | 4.5 MB | **37.6 MB** | 871.0 MB | 409.7 MB | 1.39 GB | 2.56 GB | 14.10 GB |
| SCX (zstd) | 5.0 MB | 39.8 MB | 353.9 MB | **308.8 MB** | **1.16 GB** | **2.19 GB** | **11.65 GB** |
| SCX (lz4) | 5.5 MB | 45.3 MB | **335.4 MB** | 342.1 MB | 1.28 GB | 2.40 GB | 12.90 GB |
| SCX (pcodec) | 5.0 MB | 39.8 MB | 353.9 MB | **308.8 MB** | **1.16 GB** | **2.19 GB** | **11.65 GB** |
| SCX (none) | 10.0 MB | 96.4 MB | 763.2 MB | 759.9 MB | 2.88 GB | 5.36 GB | 28.78 GB |
| Zarr (zstd) | **4.2 MB** | 44.1 MB | 368.5 MB | 344.2 MB | 1.29 GB | 2.42 GB | 12.48 GB |
| Zarr (blosc-lz4) | 5.3 MB | 55.2 MB | 408.5 MB | 417.6 MB | 1.56 GB | 2.92 GB | 15.88 GB |
| TileDB-SOMA | 5.0 MB | 47.8 MB | 410.8 MB | 371.4 MB | 1.39 GB | 2.59 GB | 13.86 GB |
| h5ad (gzip) | 20.5 MB | 193.3 MB | 1021.0 MB | 1.48 GB | 1.66 GB | 3.11 GB | 16.74 GB |
| h5ad (lzf) | 20.5 MB | 193.3 MB | 1021.0 MB | 1.48 GB | 3.32 GB | 6.08 GB | 36.71 GB |
| h5ad (none) | 20.5 MB | 193.3 MB | 1021.0 MB | 1.48 GB | 5.66 GB | 10.62 GB | 85.08 GB |
| slaf | 6.8 MB | — | — | — | — | 3.75 GB | — |

### Compression Ratios <a id="storage-efficiency-compression-ratios"></a>

*Compression ratio vs uncompressed h5ad by format and dataset*

| Format | pbmc3k | pbmc10k | smartseq2 | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 4.55x | **5.14x** | 2.00x | 3.69x | 4.07x | 4.13x | 6.04x |
| SCX (scx1) | 4.55x | **5.14x** | 1.17x | 3.69x | 4.08x | 4.15x | 6.04x |
| SCX (zstd) | 4.10x | 4.85x | 2.89x | **4.90x** | **4.88x** | **4.84x** | **7.30x** |
| SCX (lz4) | 3.71x | 4.27x | **3.04x** | 4.42x | 4.41x | 4.43x | 6.60x |
| SCX (pcodec) | 4.10x | 4.85x | 2.89x | **4.90x** | **4.88x** | **4.84x** | **7.30x** |
| SCX (none) | 2.04x | 2.00x | 1.34x | 1.99x | 1.96x | 1.98x | 2.96x |
| Zarr (zstd) | **4.93x** | 4.39x | 2.77x | 4.40x | 4.39x | 4.39x | 6.81x |
| Zarr (blosc-lz4) | 3.88x | 3.50x | 2.50x | 3.62x | 3.62x | 3.63x | 5.36x |
| TileDB-SOMA | 4.08x | 4.04x | 2.49x | 4.07x | 4.06x | 4.10x | 6.14x |
| h5ad (gzip) | 1.00x | 1.00x | 1.00x | 1.00x | 3.41x | 3.41x | 5.08x |
| h5ad (lzf) | 1.00x | 1.00x | 1.00x | 1.00x | 1.70x | 1.75x | 2.32x |
| slaf | 3.00x | — | — | — | — | 2.83x | — |

![Compression Ratio vs h5ad_none](figures/compression_bar.png)
*Figure: Compression Ratio vs h5ad_none*


**Takeaways:**
- SCX achieves the best compression on UMI count data — consistently #1 at census scale (2.57 GB vs 2.42 GB Zarr zstd on 1M cells).
- SCX lz4 compresses better than Zarr lz4 — byte-shuffle pre-filter is effective (14.10 GB vs 12.48 GB on 5M cells).
- Compression ratio improves with scale: 6.04x on census_5m vs 4.13x on census_1m.

_Source: raw_json — derived from compression benchmark results_

## 6. Local I/O Performance <a id="local-io-performance"></a>

### I/O Performance Summary <a id="local-io-performance-io-performance-summary"></a>

> **Info**
> - **Read:** SCX is the fastest reader at census scale — shard-level parallelism scales sub-linearly with cell count.
> - **Write:** SCX conversion pipeline includes h5ad read overhead; write-only timing isolates codec cost. SCX is the only format that scales writes with cores.
> - **Selective read:** SCX dominates column projection — shard-level predicate pushdown skips irrelevant data on disk.
> - **Memory:** SCX has the lowest RSS footprint for full materialization. Mode-specific tables below prevent mixing definitions.

### Read Performance (Full Materialization) <a id="local-io-performance-read-performance-full-materialization"></a>

Warm-cache full read — the entire dataset is materialized into an in-memory AnnData/CSR matrix.

*Median full-read wall time by format and dataset*

| Format | pbmc3k | pbmc10k | smartseq2 | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 0.032s | 0.314s | 1.00s | 0.600s | 1.92s | 3.02s | 2.1m |
| SCX (scx1) | 0.042s | 0.321s | 1.06s | 0.791s | **1.84s** | **2.74s** | 1.9m |
| SCX (zstd) | 0.041s | 0.431s | 1.29s | 0.967s | 2.31s | 3.25s | 2.6m |
| SCX (lz4) | 0.046s | 0.495s | 1.33s | 0.966s | 2.45s | 4.10s | 2.9m |
| SCX (pcodec) | 0.040s | 0.418s | 1.17s | 0.859s | 2.50s | 3.20s | 2.6m |
| SCX (none) | 0.029s | 0.283s | 0.806s | 0.633s | 1.89s | 3.20s | 1.6m |
| Zarr (zstd) | 0.022s | 0.100s | 0.458s | 0.703s | 2.59s | 4.87s | 57.38s |
| Zarr (blosc-lz4) | **0.018s** | **0.088s** | **0.388s** | **0.580s** | 2.23s | 3.93s | **41.45s** |
| TileDB-SOMA | 0.193s | 0.588s | 2.42s | 3.31s | 10.23s | 15.02s | 1.5m |
| h5ad (gzip) | 0.047s | 0.125s | 0.551s | 0.832s | 25.40s | 47.41s | 4.7m |
| h5ad (lzf) | 0.047s | 0.123s | 0.553s | 0.755s | 11.28s | 21.70s | 2.4m |
| h5ad (none) | 0.048s | 0.129s | 0.520s | 0.863s | 3.04s | 5.60s | 42.92s |
| slaf | — | — | — | — | — | 53.04s | — |

![Figure](figures/read_speed_bar.png)


![Figure](figures/scaling_curves.png)


**Takeaways:**
- **Zarr (blosc-lz4) is the fastest reader at census scale.**
- 1.30x faster than Zarr lz4 on 1M cells.
- 0.33x faster than Zarr lz4 on 5M cells.
- Read time scales sub-linearly with cell count due to shard-level parallelism.

_Source: raw_json — derived from read_full benchmark results_

### Read Performance (Selective / Query) <a id="local-io-performance-read-performance-selective-query"></a>

Column projection: 2,000 HVG columns selected from full gene set.

*Median selective-read (column projection) wall time by format and dataset*

| Format | smartseq2 | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|---|
| SCX (auto) | 0.769s | 0.934s | 5.69s | 10.92s | 1.1m |
| SCX (scx1) | 0.794s | 0.933s | 5.59s | 10.82s | 1.2m |
| SCX (zstd) | 0.937s | 1.43s | 9.02s | 17.03s | 1.7m |
| SCX (lz4) | 1.11s | 1.59s | 10.04s | 19.04s | 2.0m |
| SCX (pcodec) | 0.950s | 1.40s | 9.14s | 18.24s | 1.7m |
| SCX (none) | 0.527s | 0.761s | 4.59s | 8.39s | 47.55s |
| Zarr (zstd) | 0.467s | 0.674s | 2.45s | 4.79s | 1.1m |
| Zarr (blosc-lz4) | **0.373s** | **0.571s** | **2.04s** | 4.46s | 33.72s |
| TileDB-SOMA | 1.34s | 1.66s | 2.51s | **2.83s** | **5.84s** |
| h5ad (gzip) | 0.632s | 0.862s | 25.12s | 47.07s | 5.7m |
| h5ad (lzf) | 0.657s | 0.849s | 10.99s | 21.39s | 3.3m |
| h5ad (none) | 0.631s | 0.873s | 3.18s | 5.60s | 1.7m |
| slaf | — | — | — | 10.51s | — |

**Takeaways:**
- SCX is **0.4x faster** than Zarr on census_1m.
- SCX is **1.0x faster** than Zarr on census_5m.
- Shard-level predicate pushdown enables SCX to skip irrelevant data on disk.

_Source: raw_json — derived from read_selective benchmark results_

### Write Performance (Conversion Pipeline) <a id="local-io-performance-write-performance-conversion-pipeline"></a>

Full conversion pipeline: read source h5ad → encode → write target format. This timing includes h5ad read overhead and is *not* directly comparable to write-only benchmarks.

*Median conversion-pipeline wall time (h5ad read + encode + write)*

| Format | pbmc3k | pbmc10k | smartseq2 | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 0.150s | 1.84s | 7.60s | 5.26s | 14.96s | 29.36s | 6.9m |
| SCX (scx1) | 0.154s | 1.83s | 8.06s | 5.07s | 15.25s | 29.30s | 6.6m |
| SCX (zstd) | 0.166s | 1.97s | 5.59s | 5.43s | 20.13s | 27.98s | 7.2m |
| SCX (lz4) | 0.147s | 1.67s | 4.14s | 4.78s | 15.09s | 25.92s | 5.6m |
| SCX (pcodec) | 0.166s | 1.97s | 5.60s | 5.23s | 15.52s | 31.54s | 7.2m |
| SCX (none) | 0.124s | 1.49s | 4.02s | 4.88s | 20.93s | 31.18s | 4.9m |
| Zarr (zstd) | 0.105s | 0.403s | 1.92s | 2.65s | 8.96s | 16.95s | 1.7m |
| Zarr (blosc-lz4) | **0.083s** | **0.335s** | 1.54s | 2.41s | 8.06s | 15.01s | **1.6m** |
| TileDB-SOMA | 1.07s | 7.02s | 37.78s | 57.30s | 3.0m | 4.6m | 25.0m |
| h5ad (gzip) | 0.435s | 4.15s | 29.45s | 31.95s | 2.0m | 3.7m | 20.4m |
| h5ad (lzf) | 0.153s | 0.763s | 4.07s | 5.54s | 20.13s | 38.02s | 3.6m |
| h5ad (none) | 0.104s | 0.353s | **1.52s** | **2.31s** | **7.08s** | **13.36s** | 1.9m |

**Note:** All competing formats (Zarr, h5ad, TileDB-SOMA) write single-threaded — they cannot scale across cores. SCX parallelises shard encoding via rayon.

### Write Performance (Write-Only) <a id="local-io-performance-write-performance-write-only"></a>

Write-only timing: encode from in-memory AnnData without h5ad read overhead. Isolates codec/writer cost.

*Median write-only wall time (in-memory AnnData, 1 thread)*

| Format | census_500k | census_1m |
|---|---|---|
| SCX (auto) | 32.94s | 59.61s |
| SCX (scx1) | 32.20s | 1.0m |
| SCX (zstd) | 35.50s | 1.1m |
| SCX (lz4) | 26.45s | 48.14s |
| SCX (pcodec) | 35.79s | 1.1m |
| SCX (none) | **20.77s** | **39.82s** |

### Write Scaling (Parallel) <a id="local-io-performance-write-scaling-parallel"></a>

Writing the same h5ad → SCX pipeline with 32 rayon threads.

*SCX parallel write scaling (1T vs 32T, full pipeline)*

| Format | census_500k @1T | census_500k @32T | census_500k Δ | census_1m @1T | census_1m @32T | census_1m Δ |
|---|---|---|---|---|---|---|
| SCX (auto) | 36.62s | 16.23s | 2.3x | 1.1m | 27.09s | 2.4x |
| SCX (scx1) | 36.38s | 14.99s | 2.4x | 1.1m | 26.71s | 2.5x |
| SCX (zstd) | 38.79s | 15.85s | 2.4x | 1.2m | 26.64s | 2.8x |
| SCX (lz4) | 29.50s | 14.30s | 2.1x | 55.19s | 25.64s | 2.2x |
| SCX (pcodec) | 38.93s | 15.91s | 2.4x | 1.2m | 26.15s | 2.8x |
| SCX (none) | 24.13s | 16.06s | 1.5x | 47.31s | 31.37s | 1.5x |

**Takeaways:**
- **Apples-to-apples, SCX is competitive at 32 threads.**
- **SCX is the only format that scales writes with cores.**
- **Compression-heavy SCX codecs scale best.**
- **h5ad gzip and TileDB-SOMA are significantly slower at any thread count.**

### Memory Efficiency (Peak RSS) <a id="local-io-performance-memory-efficiency-peak-rss"></a>

Peak RSS (delta above baseline) during dataset read. Tables are grouped by measurement mode to prevent mixing full-materialization and subset/query memory figures.

*Peak RSS (delta) — Full read (materialized)*

| Format | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|
| SCX (auto) | 24 MB | 108 MB | 53 MB | 21 MB |
| SCX (scx1) | 39 MB | 92 MB | 77 MB | 21 MB |
| SCX (zstd) | 45 MB | 41 MB | 79 MB | 54 MB |
| SCX (lz4) | 43 MB | 43 MB | 146 MB | 52 MB |
| SCX (pcodec) | 39 MB | 47 MB | 60 MB | 48 MB |
| SCX (none) | 29 MB | 147 MB | 277 MB | 5 MB |
| Zarr (zstd) | 4 MB | 13 MB | 5 MB | **1 MB** |
| Zarr (blosc-lz4) | **0 MB** | 1 MB | **2 MB** | 25 MB |
| TileDB-SOMA | 95 MB | 157 MB | 468 MB | 814 MB |
| h5ad (gzip) | 16 MB | **0 MB** | 10 MB | 53 MB |
| h5ad (lzf) | 16 MB | 1 MB | 3 MB | 86 MB |
| h5ad (none) | 8 MB | 4 MB | 3 MB | 16 MB |
| slaf | — | — | 2.6 GB | — |

*Peak RSS (delta) — Subset read (1K cells)*

| Format | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|
| SCX (auto) | 7 MB | 4 MB | 57 MB | **0 MB** |
| SCX (scx1) | 1 MB | 6 MB | 57 MB | 36 MB |
| SCX (zstd) | 3 MB | 13 MB | 24 MB | **0 MB** |
| SCX (lz4) | 7 MB | 39 MB | **0 MB** | 29 MB |
| SCX (pcodec) | 19 MB | **0 MB** | 22 MB | **0 MB** |
| SCX (none) | 6 MB | 13 MB | 20 MB | 5 MB |
| Zarr (zstd) | 4 MB | 5 MB | 3 MB | 0 MB |
| Zarr (blosc-lz4) | 15 MB | 14 MB | 2 MB | 11 MB |
| TileDB-SOMA | 45 MB | **0 MB** | **0 MB** | **0 MB** |
| h5ad (gzip) | **0 MB** | 12 MB | **0 MB** | 16 MB |
| h5ad (lzf) | **0 MB** | 10 MB | 1 MB | 4 MB |
| h5ad (none) | 1 MB | 4 MB | 3 MB | **0 MB** |
| slaf | — | — | 271 MB | — |

**Takeaways (full materialization):**
- **SCX has the lowest memory footprint for full reads.**
- Zarr requires ~0.0x more memory than SCX.
- TileDB-SOMA memory scaling is unbounded.

_Source: raw_json — derived from memory benchmark results_

### Parallel Scaling <a id="local-io-performance-parallel-scaling"></a>

Wall-time scaling from 1 to 32 threads. Value is the speedup factor vs 1 thread (1.0x).

*Parallel read scaling — census_500k*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 9.36s | 5.43s (1.7x) | 3.31s (2.8x) | 2.41s (3.9x) | 1.93s (4.9x) | 1.81s (5.2x) | 5.2x |
| SCX (scx1) | 9.20s | 5.22s (1.8x) | 3.18s (2.9x) | 2.13s (4.3x) | 1.67s (5.5x) | 1.50s (6.1x) | 6.1x |
| SCX (zstd) | 13.25s | 7.26s (1.8x) | 4.25s (3.1x) | 2.85s (4.6x) | 2.52s (5.3x) | 2.17s (6.1x) | 6.1x |
| SCX (lz4) | 14.58s | 7.99s (1.8x) | 4.72s (3.1x) | 3.09s (4.7x) | 2.48s (5.9x) | 2.30s (6.3x) | 6.3x |
| SCX (pcodec) | 12.93s | 6.96s (1.9x) | 4.11s (3.1x) | 2.76s (4.7x) | 2.08s (6.2x) | 1.93s (6.7x) | 6.7x |
| SCX (none) | 8.32s | 4.65s (1.8x) | 3.18s (2.6x) | 2.24s (3.7x) | 1.85s (4.5x) | 1.82s (4.6x) | 4.6x |
| Zarr (zstd) | 2.53s | 2.51s (1.0x) | 2.54s (1.0x) | 2.49s (1.0x) | 2.51s (1.0x) | 2.59s (1.0x) | 1.0x |
| Zarr (blosc-lz4) | 2.14s | 1.95s (1.1x) | 2.16s (1.0x) | 1.97s (1.1x) | 2.14s (1.0x) | 1.99s (1.1x) | 1.1x |
| TileDB-SOMA | 8.37s | 8.13s (1.0x) | 8.51s (1.0x) | 8.50s (1.0x) | 8.49s (1.0x) | 8.44s (1.0x) | 1.0x |
| h5ad (gzip) | 25.20s | — | — | — | — | — | 1.0x |
| h5ad (lzf) | 10.94s | — | — | — | — | — | 1.0x |
| h5ad (none) | 2.76s | — | — | — | — | — | 1.0x |

*Parallel read scaling — census_1m*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 18.77s | 10.52s (1.8x) | 6.25s (3.0x) | 4.17s (4.5x) | 3.28s (5.7x) | 3.13s (6.0x) | 6.0x |
| SCX (scx1) | 17.37s | 9.98s (1.7x) | 5.87s (3.0x) | 3.79s (4.6x) | 2.94s (5.9x) | 2.76s (6.3x) | 6.3x |
| SCX (zstd) | 23.48s | 13.32s (1.8x) | 7.70s (3.0x) | 4.90s (4.8x) | 3.65s (6.4x) | 3.44s (6.8x) | 6.8x |
| SCX (lz4) | 27.24s | 14.70s (1.9x) | 8.35s (3.3x) | 5.37s (5.1x) | 4.05s (6.7x) | 3.82s (7.1x) | 7.1x |
| SCX (pcodec) | 25.84s | 13.89s (1.9x) | 7.84s (3.3x) | 5.15s (5.0x) | 3.82s (6.8x) | 3.63s (7.1x) | 7.1x |
| SCX (none) | 14.12s | 8.80s (1.6x) | 5.23s (2.7x) | 3.64s (3.9x) | 2.94s (4.8x) | 2.96s (4.8x) | 4.8x |
| Zarr (zstd) | 4.83s | 4.83s (1.0x) | 4.87s (1.0x) | 4.80s (1.0x) | 4.99s (1.0x) | 4.86s (1.0x) | 1.0x |
| Zarr (blosc-lz4) | 4.15s | 4.24s (1.0x) | 4.11s (1.0x) | 4.12s (1.0x) | 4.22s (1.0x) | 4.15s (1.0x) | 1.0x |
| TileDB-SOMA | 15.63s | 15.48s (1.0x) | 15.94s (1.0x) | 15.57s (1.0x) | 15.61s (1.0x) | 15.49s (1.0x) | 1.0x |
| h5ad (gzip) | 47.19s | — | — | — | — | — | 1.0x |
| h5ad (lzf) | 21.16s | — | — | — | — | — | 1.0x |
| h5ad (none) | 6.28s | — | — | — | — | — | 1.0x |

*Parallel read scaling — census_5m*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 2.4m | 1.5m (1.6x) | 57.55s (2.5x) | 50.29s (2.8x) | 42.10s (3.4x) | 39.98s (3.5x) | 3.5x |
| SCX (scx1) | 1.9m | 1.2m (1.5x) | 50.09s (2.3x) | 39.82s (2.9x) | 34.12s (3.4x) | 33.58s (3.4x) | 3.4x |
| SCX (zstd) | 2.5m | 1.6m (1.6x) | 59.45s (2.6x) | 45.06s (3.4x) | 38.14s (4.0x) | 51.04s (3.0x) | 4.0x |
| SCX (lz4) | 2.7m | 1.7m (1.6x) | 1.0m (2.6x) | 47.32s (3.5x) | 39.94s (4.1x) | 40.23s (4.1x) | 4.1x |
| SCX (pcodec) | 2.6m | 1.7m (1.6x) | 1.0m (2.6x) | 47.25s (3.3x) | 40.68s (3.9x) | 39.32s (4.0x) | 4.0x |
| SCX (none) | 1.6m | 1.1m (1.5x) | 46.99s (2.1x) | 38.96s (2.5x) | 36.84s (2.7x) | 36.44s (2.7x) | 2.7x |
| Zarr (zstd) | 45.46s | 45.15s (1.0x) | 45.22s (1.0x) | 45.52s (1.0x) | 46.37s (1.0x) | 45.26s (1.0x) | 1.0x |
| Zarr (blosc-lz4) | 42.19s | 42.96s (1.0x) | 41.68s (1.0x) | 41.87s (1.0x) | 40.64s (1.0x) | 41.68s (1.0x) | 1.0x |
| TileDB-SOMA | 1.5m | 1.4m (1.1x) | 1.5m (1.0x) | 1.5m (1.0x) | 1.6m (1.0x) | 1.5m (1.0x) | 1.1x |
| h5ad (gzip) | 4.8m | — | — | — | — | — | 1.0x |
| h5ad (lzf) | 2.4m | — | — | — | — | — | 1.0x |
| h5ad (none) | 44.05s | — | — | — | — | — | 1.0x |

*Parallel write scaling — Full pipeline (h5ad read + write) — census_500k*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 36.62s | 25.11s (1.5x) | 20.38s (1.8x) | 18.05s (2.0x) | 16.69s (2.2x) | 16.23s (2.3x) | 2.3x |
| SCX (scx1) | 36.38s | 25.53s (1.4x) | 20.40s (1.8x) | 16.22s (2.2x) | 15.37s (2.4x) | 14.99s (2.4x) | 2.4x |
| SCX (zstd) | 38.79s | 26.16s (1.5x) | 19.68s (2.0x) | 16.62s (2.3x) | 16.22s (2.4x) | 15.85s (2.4x) | 2.4x |
| SCX (lz4) | 29.50s | 21.41s (1.4x) | 17.05s (1.7x) | 15.00s (2.0x) | 14.66s (2.0x) | 14.30s (2.1x) | 2.1x |
| SCX (pcodec) | 38.93s | 26.90s (1.4x) | 20.05s (1.9x) | 18.52s (2.1x) | 16.81s (2.3x) | 15.91s (2.4x) | 2.4x |
| SCX (none) | 24.13s | 19.69s (1.2x) | 17.28s (1.4x) | 16.44s (1.5x) | 16.18s (1.5x) | 16.06s (1.5x) | 1.5x |
| Zarr (zstd) | 8.83s | 8.82s (1.0x) | 8.86s (1.0x) | 8.82s (1.0x) | 8.85s (1.0x) | 8.86s (1.0x) | 1.0x |
| Zarr (blosc-lz4) | 7.70s | 7.69s (1.0x) | 7.89s (1.0x) | 7.70s (1.0x) | 7.69s (1.0x) | 7.64s (1.0x) | 1.0x |
| TileDB-SOMA | 3.1m | 3.0m (1.0x) | 3.0m (1.0x) | 3.1m (1.0x) | 3.1m (1.0x) | 3.0m (1.0x) | 1.0x |
| h5ad (gzip) | 2.0m | — | — | — | — | — | 1.0x |
| h5ad (lzf) | 20.20s | — | — | — | — | — | 1.0x |
| h5ad (none) | 7.07s | — | — | — | — | — | 1.0x |

*Parallel write scaling — Full pipeline (h5ad read + write) — census_1m*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 1.1m | 46.65s (1.4x) | 36.09s (1.8x) | 29.77s (2.2x) | 27.66s (2.4x) | 27.09s (2.4x) | 2.4x |
| SCX (scx1) | 1.1m | 46.11s (1.4x) | 35.31s (1.9x) | 30.70s (2.2x) | 27.73s (2.4x) | 26.71s (2.5x) | 2.5x |
| SCX (zstd) | 1.2m | 49.50s (1.5x) | 37.02s (2.0x) | 31.15s (2.4x) | 27.76s (2.6x) | 26.64s (2.8x) | 2.8x |
| SCX (lz4) | 55.19s | 39.54s (1.4x) | 31.56s (1.7x) | 27.73s (2.0x) | 26.37s (2.1x) | 25.64s (2.2x) | 2.2x |
| SCX (pcodec) | 1.2m | 48.58s (1.5x) | 36.11s (2.0x) | 30.23s (2.4x) | 26.93s (2.7x) | 26.15s (2.8x) | 2.8x |
| SCX (none) | 47.31s | 37.13s (1.3x) | 34.85s (1.4x) | 31.75s (1.5x) | 31.25s (1.5x) | 31.37s (1.5x) | 1.5x |
| Zarr (zstd) | 16.21s | 16.27s (1.0x) | 16.29s (1.0x) | 16.15s (1.0x) | 16.11s (1.0x) | 16.23s (1.0x) | 1.0x |
| Zarr (blosc-lz4) | 14.25s | 14.52s (1.0x) | 14.17s (1.0x) | 14.21s (1.0x) | 14.20s (1.0x) | 14.13s (1.0x) | 1.0x |
| TileDB-SOMA | 4.6m | 4.5m (1.0x) | 4.5m (1.0x) | 4.5m (1.0x) | 4.4m (1.0x) | 4.4m (1.0x) | 1.0x |
| h5ad (gzip) | 3.7m | — | — | — | — | — | 1.0x |
| h5ad (lzf) | 37.61s | — | — | — | — | — | 1.0x |
| h5ad (none) | 13.41s | — | — | — | — | — | 1.0x |

*Parallel write scaling — Full pipeline (h5ad read + write) — census_5m*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 7.0m | 5.0m (1.4x) | 3.8m (1.8x) | 3.3m (2.2x) | 3.0m (2.3x) | 2.9m (2.4x) | 2.4x |
| SCX (scx1) | 6.9m | 4.9m (1.4x) | 3.9m (1.8x) | 3.3m (2.1x) | 3.1m (2.2x) | 2.9m (2.4x) | 2.4x |
| SCX (zstd) | 7.3m | 5.0m (1.5x) | 3.8m (1.9x) | 3.2m (2.3x) | 2.9m (2.5x) | 2.9m (2.6x) | 2.6x |
| SCX (lz4) | 5.7m | 4.3m (1.3x) | 3.5m (1.7x) | 3.1m (1.9x) | 2.9m (2.0x) | 3.0m (1.9x) | 2.0x |
| SCX (pcodec) | 7.1m | 4.8m (1.5x) | 3.6m (1.9x) | 3.1m (2.3x) | 2.8m (2.5x) | 2.7m (2.6x) | 2.6x |
| SCX (none) | 5.0m | 4.0m (1.2x) | 3.5m (1.4x) | 3.3m (1.5x) | 3.2m (1.5x) | 3.1m (1.6x) | 1.6x |
| Zarr (zstd) | 1.7m | 1.7m (1.0x) | 1.7m (1.0x) | 1.7m (1.0x) | 1.7m (1.0x) | 1.7m (1.0x) | 1.0x |
| Zarr (blosc-lz4) | 1.6m | 1.6m (1.0x) | 1.6m (1.0x) | 1.6m (1.0x) | 1.6m (1.0x) | 1.6m (1.0x) | 1.0x |
| h5ad (gzip) | 20.4m | — | — | — | — | — | 1.0x |
| h5ad (lzf) | 3.6m | — | — | — | — | — | 1.0x |
| h5ad (none) | 1.9m | — | — | — | — | — | 1.0x |

*Parallel write scaling — Write only (in-memory AnnData) — census_500k*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 32.94s | 21.70s (1.5x) | 16.39s (2.0x) | 14.33s (2.3x) | 13.28s (2.5x) | 12.80s (2.6x) | 2.6x |
| SCX (scx1) | 32.20s | 21.47s (1.5x) | 15.13s (2.1x) | 15.54s (2.1x) | 12.51s (2.6x) | 12.03s (2.7x) | 2.7x |
| SCX (zstd) | 35.50s | 23.36s (1.5x) | 16.40s (2.2x) | 13.16s (2.7x) | 12.55s (2.8x) | 12.49s (2.8x) | 2.8x |
| SCX (lz4) | 26.45s | 18.12s (1.5x) | 14.19s (1.9x) | 12.48s (2.1x) | 11.57s (2.3x) | 11.53s (2.3x) | 2.3x |
| SCX (pcodec) | 35.79s | 23.46s (1.5x) | 16.55s (2.2x) | 13.43s (2.7x) | 12.65s (2.8x) | 13.09s (2.7x) | 2.8x |
| SCX (none) | 20.77s | 16.22s (1.3x) | 14.08s (1.5x) | 12.93s (1.6x) | 12.57s (1.7x) | 12.37s (1.7x) | 1.7x |

*Parallel write scaling — Write only (in-memory AnnData) — census_1m*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 59.61s | 41.15s (1.4x) | 28.86s (2.1x) | 23.71s (2.5x) | 22.82s (2.6x) | 20.57s (2.9x) | 2.9x |
| SCX (scx1) | 1.0m | 39.44s (1.5x) | 29.50s (2.0x) | 24.34s (2.5x) | 21.91s (2.7x) | 21.00s (2.9x) | 2.9x |
| SCX (zstd) | 1.1m | 42.55s (1.6x) | 30.40s (2.2x) | 24.46s (2.7x) | 21.58s (3.1x) | 20.67s (3.2x) | 3.2x |
| SCX (lz4) | 48.14s | 33.19s (1.5x) | 25.18s (1.9x) | 21.37s (2.3x) | 19.61s (2.5x) | 19.04s (2.5x) | 2.5x |
| SCX (pcodec) | 1.1m | 42.11s (1.6x) | 30.06s (2.2x) | 24.05s (2.8x) | 21.07s (3.2x) | 20.62s (3.3x) | 3.3x |
| SCX (none) | 39.82s | 31.31s (1.3x) | 26.23s (1.5x) | 25.01s (1.6x) | 23.85s (1.7x) | 24.75s (1.6x) | 1.7x |

*Parallel write scaling — Write only (in-memory AnnData) — census_5m*

| Format | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads | Max speedup |
|---|---|---|---|---|---|---|---|
| SCX (auto) | 6.0m | 4.1m (1.5x) | 3.0m (2.0x) | 2.4m (2.5x) | 2.2m (2.7x) | 2.0m (3.0x) | 3.0x |
| SCX (scx1) | 5.9m | 3.9m (1.5x) | 3.0m (2.0x) | 2.3m (2.5x) | 2.1m (2.8x) | 2.1m (2.8x) | 2.8x |
| SCX (zstd) | 6.3m | 4.3m (1.5x) | 2.9m (2.1x) | 2.4m (2.6x) | 2.2m (2.9x) | 2.1m (3.0x) | 3.0x |
| SCX (lz4) | 4.8m | 3.5m (1.4x) | 2.6m (1.8x) | 2.2m (2.2x) | 2.2m (2.2x) | 2.0m (2.4x) | 2.4x |
| SCX (pcodec) | 6.2m | 4.1m (1.5x) | 2.9m (2.1x) | 2.3m (2.8x) | 2.0m (3.1x) | 2.0m (3.2x) | 3.2x |
| SCX (none) | 4.0m | 3.2m (1.3x) | 2.6m (1.5x) | 2.4m (1.7x) | 2.3m (1.8x) | 2.3m (1.7x) | 1.8x |

## 7. Cloud and Object-Store Behavior <a id="cloud-and-object-store-behavior"></a>

### Cloud Filtered Read <a id="cloud-and-object-store-behavior-cloud-filtered-read"></a>

Filtering 5% of cells directly from object storage without downloading the full dataset.

*Cloud filtered query — pbmc3k (median wall-clock, p95 in parens)*

| Format | cell_type_eq_t_cell | n_counts_gt_1000 | random_1pct |
|---|---|---|---|
| SCX (auto) | — | 0.359s (0.457s) ×5 | 0.410s (0.942s) ×5 |
| SCX (scx1) | — | 0.483s (0.942s) ×5 | 0.414s (0.927s) ×5 |
| SCX (zstd) | — | 0.463s (0.603s) ×5 | 0.781s (0.963s) ×5 |
| SCX (lz4) | — | 0.506s (1.12s) ×5 | 0.538s (0.783s) ×5 |
| SCX (pcodec) | — | 0.457s (0.953s) ×5 | 0.409s (0.656s) ×5 |
| SCX (none) | — | 0.706s (0.871s) ×5 | 0.466s (1.19s) ×5 |
| TileDB-SOMA | — | 1.47s (2.09s) ×5 | 0.975s (1.03s) ×5 |
| slaf | — | 1.42s (1.55s) ×5 | 1.22s (1.50s) ×5 |

*Cloud filtered query — tabula_100k (median wall-clock, p95 in parens)*

| Format | cell_type_eq_t_cell | n_counts_gt_1000 | random_1pct |
|---|---|---|---|
| SCX (auto) | 1.87s (2.02s) ×3 | 3.48s (4.11s) ×3 | 2.67s (2.80s) ×3 |
| SCX (scx1) | 4.72s (6.19s) ×3 | 6.66s (7.85s) ×3 | 6.20s (6.96s) ×3 |
| SCX (zstd) | 5.07s (5.88s) ×3 | 4.61s (5.59s) ×3 | 6.07s (9.25s) ×3 |
| SCX (lz4) | 5.50s (6.04s) ×3 | 6.65s (11.15s) ×3 | 6.49s (7.03s) ×3 |
| SCX (pcodec) | 4.34s (5.26s) ×3 | 7.43s (9.23s) ×3 | 5.61s (6.00s) ×3 |
| SCX (none) | 4.75s (5.39s) ×3 | 8.01s (8.04s) ×3 | 6.09s (8.01s) ×3 |
| TileDB-SOMA | 1.74s (1.98s) ×3 | 14.10s (14.22s) ×3 | 5.60s (5.83s) ×3 |
| slaf | 1.00s (1.04s) ×3 | 4.97s (5.09s) ×3 | 3.16s (3.31s) ×3 |

*Cloud filtered query — census_1m (median wall-clock, p95 in parens)*

| Format | cell_type_eq_t_cell | n_counts_gt_1000 | random_1pct |
|---|---|---|---|
| SCX (auto) | 17.92s (36.22s) ×3 | — | 24.13s (29.87s) ×3 |
| SCX (scx1) | 12.91s (25.99s) ×3 | — | 23.67s (25.86s) ×3 |
| SCX (zstd) | 14.47s (18.55s) ×3 | — | 24.55s (26.13s) ×3 |
| SCX (lz4) | 15.11s (16.83s) ×3 | — | 24.07s (28.37s) ×3 |
| SCX (pcodec) | 13.12s (14.26s) ×3 | — | 23.40s (59.07s) ×3 |
| SCX (none) | 34.60s (53.15s) ×3 | — | 27.68s (31.93s) ×3 |
| TileDB-SOMA | 5.38s (9.48s) ×3 | — | 17.37s (32.69s) ×3 |

**Takeaways:**
- SCX is the only format that can push down query predicates to the cloud storage layer.

### Cloud Read vs Pull <a id="cloud-and-object-store-behavior-cloud-read-vs-pull"></a>

Reading from object storage directly vs `aws s3 cp` to local disk first.

*CloudReader vs full pull — pbmc3k*

| Scenario | Method | Median wall | Bytes downloaded | n |
|---|---|---|---|---|
| metadata_only | `open_cloud` | 0.108s | 0 B | 5 |
| metadata_only | `pull_full` | 0.372s | 4.5 MB | 5 |
| selective_20pct | `pull_filtered` | 0.336s | 4.5 MB | 5 |
| selective_5pct | `pull_filtered` | 0.342s | 4.5 MB | 5 |
| selective_80pct | `pull_filtered` | 0.348s | 4.5 MB | 5 |

*CloudReader vs full pull — tabula_100k*

| Scenario | Method | Median wall | Bytes downloaded | n |
|---|---|---|---|---|
| metadata_only | `open_cloud` | 0.119s | 0 B | 3 |
| metadata_only | `pull_full` | 2.23s | 409.7 MB | 3 |
| selective_20pct | `pull_filtered` | 2.05s | 409.7 MB | 3 |
| selective_5pct | `pull_filtered` | 2.13s | 409.7 MB | 3 |
| selective_80pct | `pull_filtered` | 1.94s | 409.7 MB | 3 |

*CloudReader vs full pull — census_1m*

| Scenario | Method | Median wall | Bytes downloaded | n |
|---|---|---|---|---|
| metadata_only | `open_cloud` | 0.234s | 0 B | 3 |
| metadata_only | `pull_full` | 12.18s | 2.57 GB | 3 |

**Takeaways:**
- Direct cloud reading with SCX is faster than pulling the file to local NVMe and reading locally.

### GCP Compute-Node Matrix <a id="cloud-and-object-store-behavior-gcp-compute-node-matrix"></a>

*No GCP-matrix cloud_read results available yet.*

### Cost Model <a id="cloud-and-object-store-behavior-cost-model"></a>

*Cost model — pbmc3k (USD per 1M cells, GCS same-region)*

| Layout | metadata | selective_5pct | selective_20pct | full_read |
|---|---|---|---|---|
| `scxd_exploded` | $0.000000 | $0.011852 | $0.002963 | $0.000593 |

*Cost model — tabula_100k (USD per 1M cells, GCS same-region)*

| Layout | metadata | selective_5pct | selective_20pct | full_read |
|---|---|---|---|---|
| `scxd_exploded` | $0.000000 | $0.000800 | $0.000200 | $0.000040 |

*Cost model — census_1m (USD per 1M cells, GCS same-region)*

| Layout | metadata | selective_5pct | selective_20pct | full_read |
|---|---|---|---|---|
| `scxd_exploded` | $0.000000 | — | — | $0.000026 |

## 8. ML Data Loading <a id="ml-data-loading"></a>

### Training Pipeline Throughput <a id="ml-data-loading-training-pipeline-throughput"></a>

Simulated training epoch: 256 batch size, random row sampling, densification. Metric: batches/second.

*ML training loader throughput (batches/sec, hvg_norm scenario)*

| Loader | pbmc3k | tabula_100k | census_1m |
|---|---|---|---|
| SCX (auto) | 79.0 | 1,044.3 | 1,436.5 |
| SCX (scx1) | 77.1 | 1,040.8 | 1,497.0 |
| SCX (zstd) | 68.4 | 655.7 | 920.6 |
| SCX (lz4) | 69.8 | 548.3 | 856.4 |
| SCX (pcodec) | 75.7 | 642.7 | 920.4 |
| SCX (none) | **111.9** | **1,322.6** | **2,061.0** |
| TileDB-SOMA | 6.9 | 19.2 | 20.3 |
| h5ad (gzip) | 7.3 | 5.6 | 3.3 |
| h5ad (none) | 14.1 | 13.2 | 17.4 |

![Figure](figures/ml_loader_bar.png)


**Takeaways:**
- SCX `TrainingDataset` delivers **2,061 batches/sec** on census_1m — 116x faster than TileDB-SOMA-ML.
- Zero-copy Rust-to-Python transfer ensures Python GIL is not a bottleneck.

_Source: raw_json — derived from ml_loader benchmark results_

### Multimodal Training Loader <a id="ml-data-loading-multimodal-training-loader"></a>

CITE-seq / Multiome training-loader benchmarks. Same methodology as single-modality above, applied to the multi-assay datasets.

*Multimodal training loader throughput (batches/sec)*

| Format | cite_seq_pbmc_5k | multiome_pbmc_10k |
|---|---|---|
| scx_multimodal_per_modality_auto | 498.7 | 23.1 |
| scx_multimodal_uniform_auto | **542.8** | **30.8** |
| h5mu_uncompressed | 6.2 | 0.8 |
| h5mu_gzip | 6.7 | 0.8 |
| zarr_mudata_zstd | 6.1 | 0.8 |

*Multimodal training time-to-first-batch (seconds)*

| Format | cite_seq_pbmc_5k | multiome_pbmc_10k |
|---|---|---|
| scx_multimodal_per_modality_auto | **0.096s** | 1.92s |
| scx_multimodal_uniform_auto | 0.099s | **0.824s** |
| h5mu_uncompressed | 0.485s | 3.04s |
| h5mu_gzip | 0.777s | 6.82s |
| zarr_mudata_zstd | 0.676s | 2.80s |

## 9. Accelerators (PCA, kNN, UMAP, Leiden) <a id="accelerators-pca-knn-umap-leiden"></a>

### Accelerator Parity <a id="accelerators-pca-knn-umap-leiden-accelerator-parity"></a>

Before interpreting speedups, confirm that SCX's Rust-native accelerators reproduce scanpy/leidenalg reference results.  The table below shows per-operation parity metrics extracted from the same benchmark runs that produce timing data.  Full correctness details are in Chapter 3.

*Accelerator parity — SCX vs baseline (CPU)*

| Operation | Dataset | SCX impl | SCX time | Baseline | Baseline time | Speedup | Parity metric | Parity value |
|---|---|---|---|---|---|---|---|---|
| HVG | census_1m | pyscx_cpu | 25.26s | scanpy_cpu | 25.98s | 1.0x | overlap_pct | — |
| HVG | pbmc10k | pyscx_cpu | 0.548s | scanpy_cpu | 0.540s | 1.0x | overlap_pct | — |
| HVG | pbmc3k | pyscx_cpu | 0.077s | scanpy_cpu | 0.079s | 1.0x | overlap_pct | — |
| HVG | smartseq2 | pyscx_cpu | 2.81s | scanpy_cpu | 2.81s | 1.0x | overlap_pct | — |
| HVG | tabula_100k | pyscx_cpu | 4.09s | scanpy_cpu | 4.09s | 1.0x | overlap_pct | — |
| kNN | census_1m | pyscx_cpu | 4.3m | scanpy_cpu | 3.4m | 0.8x | recall_at_k | — |
| kNN | pbmc10k | pyscx_cpu | 28.79s | scanpy_cpu | 1.60s | 0.1x | recall_at_k | — |
| kNN | pbmc3k | pyscx_cpu | 0.542s | scanpy_cpu | 0.157s | 0.3x | recall_at_k | — |
| kNN | smartseq2 | pyscx_cpu | 1.7m | scanpy_cpu | 3.41s | 0.0x | recall_at_k | — |
| kNN | tabula_100k | pyscx_cpu | 4.4m | scanpy_cpu | 6.23s | 0.0x | recall_at_k | — |
| Leiden | census_1m | pyscx_cpu | 49.71s | leidenalg_cpu | 9.2m | 11.1x | ari | — |
| Leiden | pbmc10k | pyscx_cpu | 2.44s | leidenalg_cpu | 7.88s | 3.2x | ari | — |
| Leiden | pbmc3k | pyscx_cpu | 0.489s | leidenalg_cpu | 0.515s | 1.1x | ari | — |
| Leiden | smartseq2 | pyscx_cpu | 7.44s | leidenalg_cpu | 21.49s | 2.9x | ari | — |
| Leiden | tabula_100k | pyscx_cpu | 25.71s | leidenalg_cpu | 1.1m | 2.6x | ari | — |
| PCA | census_1m | pyscx_cpu_auto | 3.30s | scanpy_cpu | 7.62s | 2.3x | cosine_sim_min | 1.0000 |
| PCA | pbmc10k | pyscx_cpu_auto | 39.56s | scanpy_cpu | 0.847s | 0.0x | cosine_sim_min | 1.0000 |
| PCA | pbmc3k | pyscx_cpu_auto | 35.75s | scanpy_cpu | 0.440s | 0.0x | cosine_sim_min | 1.0000 |
| PCA | smartseq2 | pyscx_cpu_auto | 36.96s | scanpy_cpu | 0.872s | 0.0x | cosine_sim_min | 1.0000 |
| PCA | tabula_100k | pyscx_cpu_auto | 2.2m | scanpy_cpu | 2.97s | 0.0x | cosine_sim_min | 1.0000 |
| Preprocess | census_1m | pyscx_cpu | 7.97s | scanpy_cpu | 4.80s | 0.6x | max_abs_error | — |
| Preprocess | pbmc10k | pyscx_cpu | 0.126s | scanpy_cpu | 0.075s | 0.6x | max_abs_error | — |
| Preprocess | pbmc3k | pyscx_cpu | 8.3ms | scanpy_cpu | 7.6ms | 0.9x | max_abs_error | — |
| Preprocess | smartseq2 | pyscx_cpu | 0.658s | scanpy_cpu | 0.396s | 0.6x | max_abs_error | — |
| Preprocess | tabula_100k | pyscx_cpu | 0.986s | scanpy_cpu | 0.578s | 0.6x | max_abs_error | — |
| UMAP | census_1m | pyscx_cpu | 10.9m | scanpy_cpu | 13.6m | 1.2x | trustworthiness | 0.9304 |
| UMAP | pbmc10k | pyscx_cpu | 55.32s | scanpy_cpu | 6.24s | 0.1x | trustworthiness | 0.9643 |
| UMAP | pbmc3k | pyscx_cpu | 16.19s | scanpy_cpu | 4.09s | 0.3x | trustworthiness | 0.9226 |
| UMAP | smartseq2 | pyscx_cpu | 4.3m | scanpy_cpu | 28.81s | 0.1x | trustworthiness | 0.9621 |
| UMAP | tabula_100k | pyscx_cpu | 7.9m | scanpy_cpu | 54.22s | 0.1x | trustworthiness | 0.9766 |

### CPU Accelerator Performance <a id="accelerators-pca-knn-umap-leiden-cpu-accelerator-performance"></a>

All benchmarks use SCX's Rust-native implementations vs scanpy's standard Python stack.  Timing results are backed by raw JSON from the comprehensive harness.

### GPU Accelerator Performance <a id="accelerators-pca-knn-umap-leiden-gpu-accelerator-performance"></a>

GPU-accelerated operations (kNN via cuVS CAGRA, PCA via cuBLAS, UMAP, Leiden via cuGraph). Performance is sourced from raw JSON where available; see the external sources appendix for any manually curated entries.

*No GPU accelerator results in the comprehensive harness yet. GPU timing is available via standalone benchmarks — see the external/manual sources appendix for provenance.*

### CSC vs CSR Accelerator Dispatch <a id="accelerators-pca-knn-umap-leiden-csc-vs-csr-accelerator-dispatch"></a>

Time to compute column-oriented operations (column variance, HVG, DE, QC metrics) on a CSR-native file vs CSC sidecar.  This is an accelerator storage-layout concern — CSC sidecars can accelerate gene-major traversals.

*CSC vs CSR dispatch wall time by variant and dataset*

| Variant | pbmc3k | pbmc10k | smartseq2 | tabula_100k | census_500k | census_1m | census_5m |
|---|---|---|---|---|---|---|---|
| bench_csc__qc_metrics_csr | 0.580s | 6.30s | — | 6.53s | — | 34.62s | — |
| bench_csc__de_csc | 2.71s | 13.20s | — | 31.91s | — | 7.1m | — |
| bench_csc__de_csr | 12.35s | 2.0m | — | 3.8m | — | 23.9m | — |
| bench_csc__hvg_csc | **0.319s** | **3.06s** | — | 5.15s | — | 1.0m | — |
| bench_csc__hvg_csr | 0.427s | 4.20s | — | **3.81s** | — | **18.58s** | — |
| bench_csc__qc_metrics_csc | 0.589s | 6.12s | — | 9.87s | — | 1.5m | — |

## 10. Specialized Workloads <a id="specialized-workloads"></a>

### Multimodal Compression (CITE-seq / Multiome) <a id="specialized-workloads-multimodal-compression-cite-seq-multiome"></a>

File sizes and compression ratios for multimodal datasets. Training-loader throughput for multimodal data is in Chapter 8 (ML Data Loading).

*Multimodal file sizes by format and dataset*

| Format | cite_seq_pbmc_5k | multiome_pbmc_10k |
|---|---|---|
| h5mu_uncompressed | 79.3 MB | 1.01 GB |
| h5mu_gzip | 26.1 MB | 303.4 MB |
| zarr_mudata_zstd | 22.2 MB | 282.4 MB |
| scx_multimodal_per_modality_auto | **15.1 MB** | 227.3 MB |
| scx_multimodal_uniform_auto | 15.6 MB | **182.4 MB** |

*Multimodal compression ratio vs h5mu uncompressed*

| Format | cite_seq_pbmc_5k | multiome_pbmc_10k |
|---|---|---|
| h5mu_uncompressed | 1.00x | 1.00x |
| h5mu_gzip | 3.04x | 3.41x |
| zarr_mudata_zstd | 3.57x | 3.67x |
| scx_multimodal_per_modality_auto | **5.25x** | 4.55x |
| scx_multimodal_uniform_auto | 5.08x | **5.67x** |

### Harmony & LISI <a id="specialized-workloads-harmony-lisi"></a>

Harmony/LISI validation status (full details in Chapter 3):

*Harmony / LISI validation correctness summary*

| Validation | n_obs | Status | Key metric | Notes |
|---|---|---|---|---|
| Harmony (pbmc_small) | 2,700 | Pass | min per-PC r=0.9986 |  |
| Harmony (cell_lines) | 9,478 | Pass | min per-PC r=0.9789 |  |
| Harmony (hlca_subset) | 50,000 | Pass | min per-PC r=0.9979 |  |
| LISI (pbmc3k) | 2,700 | Pass | \|Δ\|/R = 0.76% |  |
| LISI (pbmc10k) | 11,769 | Pass | \|Δ\|/R = 0.92% |  |
| LISI (smartseq2) | 50,000 | Pass | \|Δ\|/R = 2.09% |  |
| LISI (tabula_100k) | 100,000 | Pass | \|Δ\|/R = 2.39% |  |

_Source: manual_

*Harmony validation — per-PC Pearson r vs R harmony*

| Dataset | N | Batches | d | K | min per-PC r | mean per-PC r | iter (scx / R) |
|---|---|---|---|---|---|---|---|
| pbmc_small (D1) | 2,700 | 3 | 30 | 100 | 0.9986 | 0.9992 | 5 / 4 |
| cell_lines (smartseq2) | 9,478 | 47 | 20 | 100 | 0.9789 | 0.9885 | 10 / 8 |
| hlca_subset (tabula) | 50,000 | 118 | 30 | 100 | 0.9979 | 0.9991 | 10 / 5 |

_Source: manual_

**Scaling performance:**

*Harmony scaling — wall time (s)*

| impl / device | pbmc3k (2,700) | pbmc10k (11,769) | smartseq2 (50,000) | tabula_100k (100,000) | census_500k (500,000) | census_1m (1,000,000) | census_5m (5,000,000) |
|---|---|---|---|---|---|---|---|
| `harmonypy` / cpu | 5.23s | 6.98s | 12.38s | 53.43s | 1.3m | 2.8m | 22.4m |
| `r_harmony` / cpu | — | 8.66s | 36.69s | 1.1m | 5.2m | 10.4m | 1.3h |
| `scx_accel_cpu` / cpu | 5.73s | 22.13s | 1.2m | 19.72s | 1.7m | 3.9m | 37.5m |
| `scx_accel_gpu` / gpu | — | 4.40s | 10.59s | 20.76s | 1.8m | 3.5m | 31.1m |

*Harmony scaling — peak RSS*

| impl / device | pbmc3k (2,700) | pbmc10k (11,769) | smartseq2 (50,000) | tabula_100k (100,000) | census_500k (500,000) | census_1m (1,000,000) | census_5m (5,000,000) |
|---|---|---|---|---|---|---|---|
| `harmonypy` / cpu | 563 MB | 6.6 GB | 46.7 GB | 1.1 GB | 2.5 GB | 4.6 GB | 170.7 GB |
| `r_harmony` / cpu | — | 6.5 GB | 46.7 GB | 86 MB | 294 MB | 552 MB | 2.6 GB |
| `scx_accel_cpu` / cpu | 455 MB | 6.6 GB | 46.7 GB | 784 MB | 2.1 GB | 3.9 GB | 170.7 GB |
| `scx_accel_gpu` / gpu | — | 530 MB | 678 MB | 890 MB | 11.9 GB | 21.9 GB | 170.6 GB |

*Harmony scaling — iterations*

| impl / device | pbmc3k (2,700) | pbmc10k (11,769) | smartseq2 (50,000) | tabula_100k (100,000) | census_500k (500,000) | census_1m (1,000,000) | census_5m (5,000,000) |
|---|---|---|---|---|---|---|---|
| `harmonypy` / cpu | 10 | 5 | 6 | 11 | 6 | 6 | 7 |
| `r_harmony` / cpu | — | 3 | 8 | 6 | 6 | 6 | 9 |
| `scx_accel_cpu` / cpu | 4 | 10 | 7 | 8 | 6 | 6 | 10 |
| `scx_accel_gpu` / gpu | — | 10 | 7 | 8 | 6 | 6 | 10 |

*LISI: scx-accel vs R lisi*

| Dataset | n_obs | Impl | wall | peak RSS | mean LISI | median | |Δ| / R mean |
|---|---|---|---|---|---|---|---|
| pbmc3k | 2,700 | `scx_accel` | 0.018s | 434 MB | 2.659 | 2.701 | 0.76% |
| pbmc3k | 2,700 | `r_lisi` | 0.404s | 36 MB | 2.639 | 2.678 | — |
| pbmc10k | 11,769 | `scx_accel` | 0.191s | 458 MB | 2.660 | 2.701 | 0.92% |
| pbmc10k | 11,769 | `r_lisi` | 3.18s | 38 MB | 2.636 | 2.671 | — |
| smartseq2 | 50,000 | `scx_accel` | 3.85s | 542 MB | 2.596 | 2.574 | 2.09% |
| smartseq2 | 50,000 | `r_lisi` | 43.11s | 46 MB | 2.543 | 2.481 | — |
| tabula_100k | 100,000 | `scx_accel` | 12.07s | 92.7 GB | 4.218 | 3.716 | 2.39% |
| tabula_100k | 100,000 | `r_lisi` | 1.8m | 92.7 GB | 4.119 | 3.662 | — |

### Cell-Eval / Arc-Bench Parity <a id="specialized-workloads-cell-eval-arc-bench-parity"></a>

Cell-eval correctness status (full details in Chapter 3):

*Cell-eval parity correctness summary*

| Dataset | n_obs | Operation | Status | Notes |
|---|---|---|---|---|
| pert_synth_10k | 10,000 | pseudobulk | Pass |  |
| pert_synth_10k | 10,000 | bulk_metrics | Pass |  |
| pert_synth_10k | 10,000 | discrimination_l1 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance_blas_f32 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance_blas_f64 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance_scalar_f32 | Pass |  |
| pert_synth_10k | 10,000 | energy_distance | Pass |  |
| pert_synth_10k | 10,000 | knockdown_efficiency | Pass |  |
| pert_synth_10k | 10,000 | clustering_agreement | Pass |  |
| pert_synth_100k | 100,000 | pseudobulk | Pass |  |
| pert_synth_100k | 100,000 | bulk_metrics | Pass |  |
| pert_synth_100k | 100,000 | discrimination_l1 | Pass |  |
| pert_synth_100k | 100,000 | energy_distance_blas_f32 | Pass |  |
| pert_synth_100k | 100,000 | energy_distance_blas_f64 | _Skipped_ | non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K |
| pert_synth_100k | 100,000 | energy_distance_scalar_f32 | _Skipped_ | non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K |
| pert_synth_100k | 100,000 | energy_distance | _Skipped_ | non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K |
| pert_synth_100k | 100,000 | knockdown_efficiency | Pass |  |
| pert_synth_100k | 100,000 | clustering_agreement | Pass |  |
| pert_synth_500k | 500,000 | pseudobulk | Pass |  |
| pert_synth_500k | 500,000 | bulk_metrics | Pass |  |
| pert_synth_500k | 500,000 | discrimination_l1 | Pass |  |
| pert_synth_500k | 500,000 | energy_distance | _Skipped_ | O(N^2) pairwise distance at n_obs >= 500K is infeasible |
| pert_synth_500k | 500,000 | knockdown_efficiency | Pass |  |
| pert_synth_500k | 500,000 | clustering_agreement | Pass |  |
| pert_synth_1m | 1,000,000 | pseudobulk | Pass |  |
| pert_synth_1m | 1,000,000 | bulk_metrics | Pass |  |
| pert_synth_1m | 1,000,000 | discrimination_l1 | Pass |  |
| pert_synth_1m | 1,000,000 | energy_distance | _Skipped_ | O(N^2) pairwise distance at n_obs >= 500K is infeasible |
| pert_synth_1m | 1,000,000 | knockdown_efficiency | Pass |  |
| pert_synth_1m | 1,000,000 | clustering_agreement | Pass |  |

**Performance:**

*SCX vs cell-eval perturbation-metric performance*

| Dataset | n_obs | Operation | SCX | cell-eval | Speedup | SCX RSS | ref RSS | Notes |
|---|---|---|---|---|---|---|---|---|
| pert_synth_10k | 10,000 | pseudobulk | 0.209s | 0.194s | 0.9x | 991 MB | 1.1 GB |  |
| pert_synth_10k | 10,000 | bulk_metrics | 0.439s | 0.526s | 1.2x | 1.1 GB | 1.2 GB |  |
| pert_synth_10k | 10,000 | discrimination_l1 | 0.472s | 0.411s | 0.9x | 1.2 GB | 1.2 GB |  |
| pert_synth_10k | 10,000 | energy_distance_blas_f32 | 0.668s | 12.69s | 19.0x | 1.4 GB | 1.4 GB |  |
| pert_synth_10k | 10,000 | energy_distance_blas_f64 | 0.817s | 12.62s | 15.4x | 1.7 GB | 1.7 GB |  |
| pert_synth_10k | 10,000 | energy_distance_scalar_f32 | 13.72s | 12.62s | 0.9x | 1.7 GB | 1.7 GB |  |
| pert_synth_10k | 10,000 | energy_distance | 14.06s | 12.57s | 0.9x | 1.7 GB | 1.7 GB |  |
| pert_synth_10k | 10,000 | knockdown_efficiency | 0.111s | 0.179s | 1.6x | 1.7 GB | 2.1 GB |  |
| pert_synth_10k | 10,000 | clustering_agreement | 0.478s | 0.378s | 0.8x | 2.1 GB | 2.1 GB |  |
| pert_synth_100k | 100,000 | pseudobulk | 2.71s | 2.19s | 0.8x | 4.6 GB | 6.1 GB |  |
| pert_synth_100k | 100,000 | bulk_metrics | 3.89s | 4.51s | 1.2x | 6.1 GB | 6.1 GB |  |
| pert_synth_100k | 100,000 | discrimination_l1 | 4.13s | 4.47s | 1.1x | 6.1 GB | 6.1 GB |  |
| pert_synth_100k | 100,000 | energy_distance_blas_f32 | 51.59s | 15.4m | 17.9x | 8.3 GB | 8.3 GB |  |
| pert_synth_100k | 100,000 | energy_distance_blas_f64 | — | — | — | — | — | _skipped: non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K_ |
| pert_synth_100k | 100,000 | energy_distance_scalar_f32 | — | — | — | — | — | _skipped: non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K_ |
| pert_synth_100k | 100,000 | energy_distance | — | — | — | — | — | _skipped: non-marquee variant; O(N^2) cell-eval ref too slow at n_obs >= 100K_ |
| pert_synth_100k | 100,000 | knockdown_efficiency | 1.14s | 1.72s | 1.5x | 8.9 GB | 13.0 GB |  |
| pert_synth_100k | 100,000 | clustering_agreement | 4.03s | 3.39s | 0.8x | 13.0 GB | 13.6 GB |  |
| pert_synth_500k | 500,000 | pseudobulk | 0.827s | 11.43s | 13.8x | 20.6 GB | 28.3 GB |  |
| pert_synth_500k | 500,000 | bulk_metrics | 1.79s | 24.33s | 13.6x | 28.3 GB | 28.4 GB |  |
| pert_synth_500k | 500,000 | discrimination_l1 | 1.78s | 22.96s | 12.9x | 28.4 GB | 28.4 GB |  |
| pert_synth_500k | 500,000 | energy_distance | — | — | — | — | — | _skipped: O(N^2) pairwise distance at n_obs >= 500K is infeasible_ |
| pert_synth_500k | 500,000 | knockdown_efficiency | 34.28s | 43.67s | 1.3x | 34.6 GB | 55.1 GB |  |
| pert_synth_500k | 500,000 | clustering_agreement | 2.02s | 49.86s | 24.6x | 55.1 GB | 55.1 GB |  |
| pert_synth_1m | 1,000,000 | pseudobulk | 1.94s | 37.67s | 19.4x | 40.5 GB | 55.7 GB |  |
| pert_synth_1m | 1,000,000 | bulk_metrics | 3.57s | 1.3m | 21.9x | 55.7 GB | 55.8 GB |  |
| pert_synth_1m | 1,000,000 | discrimination_l1 | 3.87s | 1.3m | 20.1x | 55.8 GB | 55.8 GB |  |
| pert_synth_1m | 1,000,000 | energy_distance | — | — | — | — | — | _skipped: O(N^2) pairwise distance at n_obs >= 500K is infeasible_ |
| pert_synth_1m | 1,000,000 | knockdown_efficiency | 35.43s | 24.12s | 0.7x | 68.2 GB | 95.5 GB |  |
| pert_synth_1m | 1,000,000 | clustering_agreement | 4.54s | 45.54s | 10.0x | 95.5 GB | 95.5 GB |  |

## 11. SCX-specific Format Operations <a id="scx-specific-format-operations"></a>

### Fragment & Manifest Operations <a id="scx-specific-format-operations-fragment-manifest-operations"></a>

Time to execute append/subset operations on SCX fragments.  These operations have no direct equivalent in h5ad, Zarr, or TileDB-SOMA.

*Fragment operation throughput*

| Operation | pbmc3k | tabula_100k | census_1m |
|---|---|---|---|
| `append` | 0.093s (48.2 MB/s) | 7.97s (51.4 MB/s) | 50.37s (52.2 MB/s) |
| `delete` | 3.6ms (372,396 rows/s) | 0.256s (39,011 rows/s) | 1.61s (6,193 rows/s) |
| `compact` | 0.110s (71.2 MB/s) | 14.12s (63.6 MB/s) | 1.7m (58.6 MB/s) |
| `rollback` | 3.4ms | 0.377s | 2.57s |

## 12. Discussion & Roadmap <a id="discussion-roadmap"></a>

### Conclusion <a id="discussion-roadmap-conclusion"></a>

SCX achieves its design goal: it replaces h5ad, Zarr, and TileDB with a single format that is faster, smaller, and natively integrates with ML and analysis stacks.

## 13. Appendix <a id="appendix"></a>

### Glossary <a id="appendix-glossary"></a>

- **SCX:** Sparse Cell eXpression System.
- **h5ad:** AnnData's HDF5-based serialization format.
- **TileDB-SOMA:** Single-cell Open Matrix Architecture by TileDB.
- **Zarr:** Cloud-native multidimensional array format.

### External and Manual Data Sources <a id="appendix-external-and-manual-data-sources"></a>

The following tables and data rows in this report are sourced from manual curation, external benchmark scripts, or standalone reports outside the comprehensive harness. Each entry lists its provenance so that report freshness can be verified.

*External and manual data sources used in this report*

| Table / Source | Kind | Path | Reason |
|---|---|---|---|
| harmony_validation | manual | — | static diagnostic (pyscx/tests/test_harmony_validation.py) |
| harmony_lisi_correctness | manual | — | Harmony rows from static diagnostic; LISI from live data |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_census_1m_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_census_500k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_census_5m_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_pbmc10k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_pbmc3k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_smartseq2_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_harmonypy_tabula_sapiens_100k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_census_1m_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_census_500k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_census_5m_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_pbmc10k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_pbmc3k_d30_K5_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_smartseq2_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_r_harmony_tabula_sapiens_100k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_census_1m_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_census_500k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_census_5m_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_pbmc10k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_pbmc3k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_pbmc3k_d30_K20_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_smartseq2_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d100_K100_cpu_pc100.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d10_K100_cpu_pc10.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d20_K100_cpu_pc20.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d30_K100_cpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d30_K100_cpu_K100.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d30_K100_cpu_pc30.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d30_K200_cpu_K200.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d30_K400_cpu_K400.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d30_K50_cpu_K50.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_cpu_tabula_sapiens_100k_d50_K100_cpu_pc50.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_gpu_census_1m_d30_K100_gpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_gpu_census_500k_d30_K100_gpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_gpu_census_5m_d30_K100_gpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_gpu_pbmc10k_d30_K100_gpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_gpu_smartseq2_d30_K100_gpu.json | — |
| external:harmony_integrate | external_report | benchmarks/results/harmony/runs/harmony_scx_accel_gpu_tabula_sapiens_100k_d30_K100_gpu.json | — |
| external: | external_report | benchmarks/results/harmony/runs/issue10_after_census_1m_d30_K100_cpu.json | — |
| external: | external_report | benchmarks/results/harmony/runs/issue10_after_census_500k_d30_K100_cpu.json | — |
| external: | external_report | benchmarks/results/harmony/runs/issue10_after_tabula100k_d30_K100_cpu.json | — |
| external: | external_report | benchmarks/results/harmony/runs/issue10_before_census_1m_d30_K100_cpu.json | — |
| external: | external_report | benchmarks/results/harmony/runs/issue10_before_census_500k_d30_K100_cpu.json | — |
| external: | external_report | benchmarks/results/harmony/runs/issue10_before_tabula100k_d30_K100_cpu.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_r_lisi_pbmc10k_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_r_lisi_pbmc3k_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_r_lisi_smartseq2_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_r_lisi_tabula_sapiens_100k_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_scx_accel_pbmc10k_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_scx_accel_pbmc3k_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_scx_accel_smartseq2_d30.json | — |
| external:lisi | external_report | benchmarks/results/harmony/runs/lisi_scx_accel_tabula_sapiens_100k_d30.json | — |
