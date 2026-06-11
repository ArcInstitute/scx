# SCX Testing & Benchmarks

## Rust Tests

**Run**: `cargo test --workspace` (all crates) or `cargo test --workspace --features cloud` (with cloud features)

| Crate | Location | Key tests |
|-------|----------|-----------|
| `scx-format` | `tests/integration.rs` | Round-trip validation, minimum file size, checksum verification |
| `scx-codec` | Per-module unit tests | Rice, FOR-BP (scalar + SIMD BitPacker4x), Delta-Golomb, LZ4+shuffle, byte-shuffle encode/decode round-trips. Reference vector tests for all codecs. |
| `scx-sparse` | Unit tests | CSR construction, row slicing, dense conversion |
| `scx-ops` | `tests/` | Append, delete, compact, rollback, merge, flock concurrency |
| `scx-engine` | Unit tests | Predicate parsing, pipeline validation, pushdown, fused ops |
| `scx-loader` | Unit tests | Pipeline lifecycle, batch format, shuffle, projection, normalize |
| `scx-cloud` | `tests/` | Explode/pack round-trip, cloud-optimize, pull/push (local backend) |
| `scx-accel` | Unit tests | PCA round-trip, kNN recall, UMAP trustworthiness, DE p-values, pseudobulk aggregation |
| `scx-gpu` | Unit tests | CUDA Rice/FOR-BP decode, sparse-to-dense, cuSPARSE SpMM, cuSOLVER QR, cuRAND, GPU PCA pipeline, GPU UMAP SGD, GPU preprocessing (normalize+log1p). GPU parity with CPU reference. |
| `scx-integration-tests` | `tests/golden_files.rs` | Golden file regression: 15 golden files (None/Scx1/Zstd/Lz4Shuffle × value encodings), BLAKE3 manifest, CSR bit-exact match, metadata match, backward compatibility, unknown codec rejection |
| `rscx` | via `R CMD check` | Seurat/SCE round-trip, query, CSR transpose |

## Python Test Suite

**Location**: `pyscx/tests/`
**Run**: `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`
**With cloud**: `../.venv/bin/maturin develop --features cloud && ../.venv/bin/pytest tests/ -v`
**With GPU**: `../.venv/bin/maturin develop --features gpu && ../.venv/bin/pytest tests/ -v`

| Test file | Purpose |
|-----------|---------|
| `test_round_trip.py` | AnnData round-trip validation |
| `test_scanpy_pipeline.py` | Full scanpy workflow (QC → PCA → Leiden → DE) |
| `test_zero_copy.py` | Zero-copy verification (`np.shares_memory`) |
| `test_go_no_go.py` | Go/No-Go gate criteria validation |
| `test_metadata.py` | Metadata preservation tests |
| `test_auto_codec.py` | Auto-codec selection (Scx1 vs Zstd) |
| `test_ops.py` | File operations (append, delete, compact, rollback, merge) |
| `test_query_pipeline.py` | Query engine Python bindings |
| `test_training_loader.py` | TrainingDataset iteration, batch format |
| `test_e2e_pipeline.py` | End-to-end loader pipeline (shuffle, HVG, normalize, memory) |
| `test_training_integration.py` | Training integration with scVI-style loops |
| `test_scvi_integration.py` | scVI DataModule integration |
| `test_python_lifecycle.py` | Python object lifecycle (GC, refcount) |
| `test_cloud.py` | Cloud operations (explode, pack, pull, push, cloud-optimize) |
| `test_backed.py` | Backed mode indexing, slicing, layer access |
| `test_backed_aggregation.py` | Backed mode streaming aggregation (sum, mean, var, max, min, nnz per axis) |
| `test_comparison_optimization.py` | `(X > 0).sum()` → `getnnz()` short-circuit optimization |
| `test_chunk_iterator.py` | Shard-aligned and fixed-size chunk iteration |
| `test_preprocess.py` | Streaming preprocessing pipeline (normalize, log1p, save_layer) |
| `test_to_anndata_integration.py` | Selective loading (var_names, obs_filter, layers) |
| `test_h5ad_scx_equivalence.py` | h5ad ↔ SCX equivalence validation |
| `test_accel.py` | Rust accelerators: PCA, kNN, UMAP, DE (Wilcoxon + streaming), pseudobulk, stratified DE |
| `test_lazy_transform.py` | Lazy transform wrapper: slicing, aggregation through transforms, chaining |
| `test_normalize_total.py` | `pyscx.accel.normalize_total()` correctness vs scanpy |
| `test_log1p.py` | `pyscx.accel.log1p()` correctness vs scanpy |
| `test_normalize_col_projection.py` | NormalizeTotal with column projection active |
| `test_truediv_interception.py` | `__truediv__` lazy RowScale interception for backed/lazy datasets |
| `test_mul_interception.py` | `__mul__` lazy RowScale interception |
| `test_col_projected_agg.py` | Column-projected streaming aggregation (sum, nnz, var, max, min) |
| `test_index_space_regression.py` | Deletion vector + transform index space correctness regression |
| `test_b2_col_projection.py` | Column projection bug regression (streaming col aggregation) |
| `conftest.py` | Shared pytest fixtures |

## Correctness Validation Suite

The correctness validation suite verifies that SCX's Rust-native accelerators, backed-mode data access, and preprocessing pipelines produce results equivalent to their scanpy/scipy counterparts. Unlike the unit tests in `pyscx/tests/`, the validation suite runs on real datasets with structured JSON reporting and quantitative pass/fail thresholds.

**Location**: `benchmarks/comprehensive/scripts/validate_*.py`
**Results**: `benchmarks/comprehensive/results/raw/correctness__*.json`

### Running

```bash
# Run all three suites on pbmc3k (~5 min)
SCX_DATA_DIR=/path/to/datasets \
  .venv/bin/python benchmarks/comprehensive/scripts/validate_scanpy_equivalence.py --dataset pbmc3k
SCX_DATA_DIR=/path/to/datasets \
  .venv/bin/python benchmarks/comprehensive/scripts/validate_backed_equivalence.py --dataset pbmc3k
SCX_DATA_DIR=/path/to/datasets \
  .venv/bin/python benchmarks/comprehensive/scripts/validate_preprocessing_paths.py --dataset pbmc3k

# Or run on SLURM (pbmc3k + tabula_sapiens_100k, ~1 hr, 200 GB)
bash benchmarks/comprehensive/scripts/slurm_validation_suite.sh --scale --mem 200G

# Or run via the benchmark module system
.venv/bin/python benchmarks/comprehensive/scripts/run_all.py \
  --benchmarks correctness --datasets pbmc3k --formats scx_auto
```

### Scanpy Equivalence (`validate_scanpy_equivalence.py`)

Runs every `pyscx.accel.*` function and its scanpy equivalent side-by-side on the same data (14 checks).

| Function | Comparison | Threshold | pbmc3k Result |
|----------|-----------|-----------|---------------|
| `normalize_total()` | Max abs error vs `sc.pp.normalize_total()` | < 1e-3 | 1.2e-4 |
| `log1p()` | Max abs error vs `sc.pp.log1p()` | < 1e-3 | 4.8e-7 |
| `pca()` | Cosine similarity per PC vs `sc.pp.pca()` | > 0.99 | > 0.999 |
| `pca()` | Variance ratio Pearson r | > 0.99 | > 0.999 |
| `neighbors()` | Recall@15 vs `sc.pp.neighbors()` | > 0.90 | 0.93 |
| `neighbors()` | Downstream Leiden ARI | > 0.80 | 0.98 |
| `umap()` | Trustworthiness at k=15 | > 0.90 | 0.92 |
| `rank_genes_groups()` | Mean top-100 gene overlap per group | > 60% | 72% |
| `rank_genes_groups()` | Min p-value Spearman r | > 0.80 | 0.91 |
| `rank_genes_groups(gene_chunk_size)` | Top-50 gene overlap vs in-memory | 100% | 100% |
| `pseudobulk_dex()` | Runs without error, produces results | Finite LFCs | 49,902 |
| `pseudobulk_dex(stratify_by)` | Per-stratum consistency | Runs | 3 strata |
| `rank_genes_groups(stratify_by)` | Per-stratum consistency | Runs | OK |
| `filter_cells()` / `filter_genes()` | Exact mask vs scanpy | 100% | 100% |
| `calculate_qc_metrics()` | Float max abs error; int exact | < 1e-5; exact | 0.0; exact |
| `subset_obs()` | Shape + data vs `adata[mask].copy()` | Exact | Exact |

### Backed-Mode Equivalence (`validate_backed_equivalence.py`)

Verifies that every operation on backed-mode (on-disk) data matches the same operation on fully materialized data (19 checks).

| Operation | Comparison | Threshold | pbmc3k Result |
|-----------|-----------|-----------|---------------|
| Row slice, fancy index, bool mask, 2D slice | Exact CSR equality | Exact | Exact |
| `sum(axis=1)`, `sum(axis=0)` | Max abs error | < 1e-6 | 0.0 |
| `var(axis=0)` | Max rel error | < 1e-3 | 1.3e-4 |
| `getnnz(axis=0)`, `getnnz(axis=1)` | Exact match | Exact | Exact |
| PCA (backed vs non-backed) | Cosine similarity per PC | > 0.99 | > 0.999 |
| QC metrics | Max abs error | < 1e-5 | 6.6e-7 |
| `filter_cells()`, `filter_genes()`, `subset_obs()` | Shape + data agreement | Exact | Exact |
| `X / row_sums`, `X * factors` (operator interception) | Max abs error vs scipy | < 1e-6 | 0.0 |
| Lazy aggregation (sum through transforms) | Max abs error vs materialized | < 0.1 | 0.025 |
| Shape, obs, var metadata | Exact match | Exact | Exact |
| `issparse()` / `format` checks | True / "csr" for backed and lazy | True | True |

### Preprocessing Path Cross-Validation (`validate_preprocessing_paths.py`)

Three-way comparison (3 checks): **A** scanpy in-memory, **B** SCX write-back (`pyscx.preprocess()`), **C** SCX lazy (backed + `pyscx.accel.*`, then materialize).

| Check | Comparison | Threshold | pbmc3k Result |
|-------|-----------|-----------|---------------|
| normalize + log1p | Pairwise max abs error on X | < 1e-5 | 4.8e-7 |
| PCA | Pairwise cosine similarity per PC | > 0.99 | > 0.999 |
| Extended pipeline (Leiden/DE/UMAP) | ARI > 0.95, DE overlap > 90%, Procrustes > 0.95 | See thresholds | 1.0 / 100% / 0.96 |

### Precision Expectations

SCX uses f32 throughout its Rust pipeline (matching scipy CSR's `float32`), while scanpy often uses f64 intermediates:

| Source | Typical magnitude | Affected checks |
|--------|-------------------|-----------------|
| f32 accumulation in normalize_total | ~1e-4 max abs error | normalize_total, log1p |
| Streaming f32 variance (two-pass) | ~1e-4 relative error | col_var |
| Cumulative error through chained transforms | ~0.01-0.03 abs error on sums | lazy_aggregation |
| HNSW vs sklearn kNN | ~93% recall@15 | neighbors |
| Randomized SVD vs ARPACK PCA | >0.999 cosine on HVG-subset data | pca |

### Adding New Checks

All three scripts follow the same pattern — write a function returning `ValidationCheck`, add it to `run_all_checks()`:

```python
from benchmarks.comprehensive.scripts.validation_helpers import (
    ValidationCheck, run_check, max_abs_error
)

def check_my_function(adata) -> ValidationCheck:
    err = max_abs_error(result_pyscx, result_scanpy)
    threshold = 1e-5
    return ValidationCheck(
        name="my_function", passed=err < threshold,
        metrics={"max_abs_error": err}, thresholds={"max_abs_error": threshold},
    )

# In run_all_checks():
checks.append(run_check("my_function", check_my_function, adata))
```

Helper utilities in `validation_helpers.py`: `max_abs_error`, `max_rel_error`, `cosine_similarity_columns`, `pearson_r`, `spearman_r`, `recall_at_k`, `gene_overlap_pct`, `csr_equal`, `to_dense`.

## Benchmarks

**Location**: `benchmarks/scripts/`
**Results**: `benchmarks/results/`

Compression, read, write, query, and file-ops benchmarks run through the
comprehensive suite (`benchmarks/comprehensive/scripts/run_parallel.py
--benchmarks compression read_full read_selective write ...`), not standalone
scripts — see [Running Benchmarks](#running-benchmarks) below. The standalone
scripts under `benchmarks/scripts/` cover the training loader, bindings, and
GPU/analysis paths:

| Script | Purpose |
|--------|---------|
| `benchmark_loader.py` | Training loader throughput, SOMA comparison |
| `benchmark_python_bindings.py` | Python bindings performance |
| `benchmark_cli.py` | CLI operation timing |
| `benchmark_bpcells.R` | BPCells R comparison |
| `benchmark_harmony.py` | Harmony2 batch-integration throughput |
| `benchmark_lisi.py` | LISI batch/cell-type mixing metric |
| `benchmark_gpu_pca.py` | GPU PCA throughput |
| `benchmark_gpu_knn.py` | GPU kNN throughput |
| `benchmark_gpu_umap.py` | GPU UMAP throughput |
| `benchmark_gpu_pipeline.py` | End-to-end GPU PCA → kNN → UMAP → Leiden |
| `benchmark_gpu_preprocess.py` | GPU normalize/log1p preprocessing |
| `benchmark_gpu_decode.py` | GPU codec decode throughput |
| `benchmark_gpu_scvi.py` | scVI GPU utilization on 10M-cell dataset |
| `download_datasets.sh` | Fetches test data (PBMC 3K, Tabula Sapiens, CELLxGENE Census, Smart-seq2) |
| `setup_cloud_test_data.sh` | Set up cloud test data in GCS bucket |
| `submit_benchmarks.py` | Submit and manage benchmark SLURM jobs |

### Running Benchmarks

```bash
# Download test datasets first:
bash benchmarks/scripts/download_datasets.sh

# Run the comprehensive suite (one submitit job per benchmark × format × dataset):
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k

# Narrow to specific benchmarks / formats:
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --benchmarks compression read_full read_selective \
    --formats scx_auto h5ad_gzip zarr_zstd \
    --datasets pbmc3k

# Run the standalone ML training-loader benchmark (not in the gate):
.venv/bin/python benchmarks/scripts/submit_benchmarks.py
```

### Key Benchmark Results

| Benchmark | Key Result |
|-----------|------------|
| Compression ratio (PBMC 3K) | 0.207 (Scx1), 0.231 (Zstd) |
| Compression ratio (Lung 100K) | 0.270 (Scx1), 0.203 (Zstd) |
| Parallel read speedup (8 threads) | 2.19–3.40× vs single-thread |
| Shard skip rate (filtered query) | 55.1% average |
| Query vs AnnData subset | 2.08× speedup |
| Append overhead | 1 ms for 10K cells |
| Merge throughput | 424 MB/s |

Full results in `benchmarks/results/*.md`.

### Benchmark Regression Gating

`gate_candidate.py` validates a candidate commit against the baseline
snapshot (`results/baselines/LATEST`). The gate checks two things:

1. **Relative-tolerance regression** — each metric must not regress beyond
   its per-metric tolerance versus the baseline.
2. **Absolute-floor violations** — metrics declared in `thresholds.yaml`
   must meet their hard bounds regardless of the baseline.

Route-specific absolute floors turn silent dispatch fallbacks (GPU→CPU, CSC→CSR)
into hard gate failures. Every GPU accelerator benchmark emits an
`<op>_route_gpu_correct` signal (1.0 = correct route, 0.0 = silent fallback);
`bench_csc_dispatch.py` emits `csc_dispatch_correct`; and `accel_de.py` emits
`de_route_csc_direct` for the pdex_ref CSC-direct path. The floor entries
cover PCA, kNN, UMAP, Leiden, HVG, DE (pdex_ref + Wilcoxon), and CSC dispatch.

See [benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating)
for the full gate table and workflow.

## GPU Tests

### Running GPU Tests

```bash
# Rust-level GPU tests (require CUDA device)
cargo test -p scx-gpu --features bench

# Python-level GPU accelerator tests
cd pyscx && ../.venv/bin/maturin develop --features gpu && \
  ../.venv/bin/pytest tests/test_accel.py -v -k "gpu"
```

### GPU Test Skip Behavior

All GPU tests use a `require_gpu!()` macro that gracefully skips when no CUDA
device is available. This macro:

1. Attempts to create a `GpuDevice` (CUDA context + stream)
2. If CUDA initialization fails (no GPU, no driver, wrong CUDA version), the test returns `Ok(())` silently
3. Tests run normally in CI environments with GPUs, and are safely skipped in CPU-only CI

The macro is defined per-module (e.g., `cusparse.rs`, `cusolver.rs`, `gpu_pca.rs`,
`gpu_umap.rs`, `gpu_preprocess.rs`) to avoid cross-module test coupling.

### GPU Correctness Thresholds

| Test | Metric | Threshold |
|------|--------|-----------|
| GPU SpMM vs CPU SpMM | Max relative error | < 1e-5 |
| GPU PCA vs CPU PCA | Cosine similarity per PC | > 0.99 (sign-invariant) |
| GPU kNN vs CPU HNSW | Recall@k | > 0.95 |
| GPU UMAP | Trustworthiness | > 0.95 |
| GPU Leiden vs CPU Leiden | ARI | > 0.90 |
| GPU normalize+log1p | Match CPU reference | rtol=1e-7 |

### GPU Benchmarks

```bash
# GPU analysis benchmarks (PCA, kNN, UMAP throughput)
.venv/bin/python benchmarks/scripts/benchmark_gpu_pca.py
.venv/bin/python benchmarks/scripts/benchmark_gpu_knn.py
.venv/bin/python benchmarks/scripts/benchmark_gpu_umap.py

# GPU preprocessing benchmarks
.venv/bin/python benchmarks/scripts/benchmark_gpu_preprocess.py
```

| Benchmark | Datasets | Compare against |
|-----------|----------|----------------|
| GPU PCA throughput | 100K, 1M, 5M cells × 2K HVGs | scx-accel CPU, scanpy, rapids-singlecell |
| GPU kNN throughput | 100K, 1M, 5M cells × 50 PCs | scx-accel CPU, PyNNDescent, rapids |
| GPU UMAP throughput | 100K, 1M cells | scx-accel CPU, umap-learn, rapids |
| GPU memory footprint | 1M, 5M, 10M cells | Peak VRAM per operation |
| End-to-end GPU pipeline | 1M cells | Full PCA → kNN → UMAP time |

### Hardware Requirements

| GPU | VRAM | Expected max dataset |
|-----|------|---------------------|
| RTX 3090 / 4090 | 24 GB | ~2M cells |
| A100 / H100 (40 GB) | 40 GB | ~5M cells |
| A100 / H100 (80 GB) | 80 GB | ~10M cells |
