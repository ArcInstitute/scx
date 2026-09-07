# SCX Testing & Benchmarks

## Rust Tests

**Run**: `cargo test --workspace` (all crates) or `cargo test --workspace --features cloud` (with cloud features)

⚠️ That runs at **default** features, and nothing in the workspace enables
`scx-convert/hdf5` by default — so it runs none of the h5ad / h5mu ingest,
export or dataframe tests in `scx-convert` and `scx-cli`. Those need their own
invocation (CI job `Test (hdf5 features)`); see
[development.md § The hdf5-gated suites are not in that run](development.md#the-hdf5-gated-suites-are-not-in-that-run)
for the commands and why the `scx-cli` half must be serialised.

| Crate | Location | Key tests |
|-------|----------|-----------|
| `scx-format` | `tests/` proptests + per-module unit tests | Modality round-trip, header, catalog, checksum verification |
| `scx-codec` | Per-module unit tests | Rice, FOR-BP (scalar + SIMD BitPacker4x), Delta-Golomb, LZ4+shuffle, Pcodec, byte-shuffle encode/decode round-trips. Reference vector tests for all codecs. |
| `scx-sparse` | Unit tests | CSR construction, row slicing, dense conversion |
| `scx-format-io` | Unit tests + `tests/` | Reader/writer round-trip, backed I/O, Arrow compat, bitmap, CSC sidecar |
| `scx-convert` | Per-module unit tests | h5ad/h5mu streaming ingest, parallel determinism, HDF5 thread-safety probe (all `--features hdf5` only); the shared parallel drain's bound / panic / early-return invariants are feature-free and do run in the default job |
| `scx-ops` | `tests/` | Append, delete, compact, rollback, merge, flock concurrency, predicate-index rewrite, streaming merge/append |
| `scx-engine` | Unit tests | Predicate parsing, pipeline validation, pushdown, fused ops |
| `scx-loader` | Unit tests | Pipeline lifecycle, batch format, shuffle, projection, normalize |
| `scx-cloud` | `tests/` | Explode/pack round-trip, cloud-optimize, pull/push (local backend), cloud query |
| `scx-accel` | Unit tests | PCA round-trip, kNN recall, UMAP trustworthiness, DE p-values, pseudobulk aggregation, NB-GLM, Harmony, LISI, HVG, gene scoring, CSC dispatch, eval metrics |
| `scx-gpu` | Unit tests | CUDA Rice/FOR-BP decode, sparse-to-dense, cuSPARSE SpMM, cuSOLVER QR, cuRAND, GPU PCA pipeline, GPU UMAP SGD, GPU preprocessing (normalize+log1p). GPU parity with CPU reference. |
| `scx-integration-tests` | `tests/` | Golden file regression (19 golden files: None/Scx1/Zstd/Lz4Shuffle/Pcodec × value encodings), conformance vectors (layers, obsm, uns, CSC, predicate indexes, deletion vectors, multimodal, bitmap, cloud layouts), doc-drift guards, integration lifecycle (write → append → delete → compact → query cross-crate), Harmony pipeline end-to-end |
| `rscx` | via `R CMD check` | Seurat/SCE round-trip, query, CSR transpose |

## Python Test Suite

**Location**: `pyscx/tests/`
**Run**: `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`
**With cloud**: `../.venv/bin/maturin develop --features hdf5,cloud && ../.venv/bin/pytest tests/ -v`
**With GPU**: `../.venv/bin/maturin develop --features hdf5,gpu && ../.venv/bin/pytest tests/ -v`

The test suite has 155+ test files. Key categories:

| Category | Test files | Purpose |
|----------|------------|---------|
| **Core round-trip** | `test_round_trip.py`, `test_dtype_round_trip.py`, `test_raw_round_trip.py`, `test_golden_files.py` | AnnData round-trip, dtype preservation, raw layer, golden file parity |
| **Scanpy pipeline** | `test_scanpy_pipeline.py`, `test_go_no_go.py` | Full scanpy workflow (QC → PCA → Leiden → DE), gate criteria |
| **File operations** | `test_ops.py`, `test_metadata.py`, `test_modify_metadata.py`, `test_auto_codec.py` | Append, delete, compact, rollback, merge; metadata; auto-codec |
| **Query engine** | `test_query_pipeline.py`, `test_predicate_index.py`, `test_predicate_index_rewrite.py` | Query pipeline, predicate index build + rewrite |
| **Training loader** | `test_training_loader.py`, `test_training_loader_pflog.py`, `test_e2e_pipeline.py`, `test_training_integration.py`, `test_scvi_integration.py`, `test_multimodal_training.py`, `test_loader_stub_coverage.py` | Dataset iteration, batch format, PFlog transforms, scVI DataModule; `__init__.pyi` ↔ runtime parity for the six loader pyclasses |
| **Backed mode** | `test_backed.py`, `test_backed_aggregation.py`, `test_chunk_iterator.py`, `test_lazy_transform.py`, `test_from_anndata_scx_backed.py`, `test_shard_skipping.py` | Indexing, slicing, streaming aggregation, chunk iteration, lazy transforms, backed→SCX rewrite, row-projected reads skipping the shards they empty (decode counts through `SCX_CPU_PROFILE`, plus the transform row-offset that skipping could shift) |
| **Operator interception** | `test_truediv_interception.py`, `test_mul_interception.py`, `test_comparison_optimization.py` | `__truediv__`/`__mul__` lazy RowScale, `(X>0).sum()` → `getnnz()` |
| **Column projection** | `test_col_projected_agg.py`, `test_col_projection_getitem.py`, `test_normalize_col_projection.py`, `test_b2_col_projection.py`, `test_lazy_row_axis_projection.py` | Column-projected aggregation, getitem, normalize, regressions; every `sum`/`mean`/`getnnz`/`var` axis on backed **and** lazy `X`, against a pure-numpy oracle |
| **Accelerators (CPU)** | `test_accel.py`, `test_accel_fused.py`, `test_accel_leiden.py`, `test_accel_pflog.py`, `test_accel_score_genes.py`, `test_accel_route_metadata.py`, `test_accel_stub_coverage.py`, `test_accel_doc_signatures.py` | PCA, kNN, UMAP, DE, Leiden, PFlog, gene scoring, route metadata, accel stub coverage, accel docs-vs-runtime signature drift |
| **Accelerators (GPU)** | `test_accel_gpu_device.py`, `test_accel_pca_gpu.py`, `test_accel_pipeline_gpu.py`, `test_to_gpu_anndata_e2e.py` | GPU device selection, GPU PCA, fused GPU pipeline, GPU AnnData handoff |
| **HVG** | `test_hvg.py`, `test_hvg_csc.py`, `test_hvg_gpu_batch_key.py`, `test_hvg_inmemory_native.py`, `test_hvg_layer_kwarg.py`, `test_hvg_loess_singularity.py`, `test_hvg_scanpy_fallback_warning.py` | HVG (seurat_v3/seurat), CSC path, GPU batch_key, loess singularity handling |
| **Differential expression** | `test_rank_genes_groups_cpu_parity.py`, `test_rank_genes_groups_gpu_parity.py`, `test_rank_genes_groups_gpu_csc_parity.py`, `test_pdex_ref_parity.py`, `test_pdex_ref_gpu_parity.py`, `test_pdex_ref_gpu_csc_parity.py`, `test_pdex_nb_glm.py`, `test_nb_glm.py`, `test_de_use_raw_layer.py`, `test_de_backed_layer.py` | Wilcoxon rank-sum CPU/GPU/CSC parity, pdex_ref parity, NB-GLM, `use_raw=` / `layer=` on in-memory and on backed input (incl. the presentation-ordered gene-axis refusal) |
| **CSC sidecar** | `test_csc_convert.py`, `test_csc_dispatch.py`, `test_csc_dispatch_lazy.py`, `test_csc_filtered_genes.py`, `test_csc_lifecycle.py`, `test_csc_capability_gate.py`, `test_csr_bypass.py` | CSC build, dispatch routing, lazy CSC, lifecycle, capability gates |
| **Preprocessing** | `test_preprocess.py`, `test_normalize_total.py`, `test_normalize_total_fallback_warning.py`, `test_log1p.py` | Streaming normalize/log1p, correctness vs scanpy, fallback warnings |
| **QC / filtering** | `test_calculate_qc_metrics_empty_qc_var.py`, `test_calculate_qc_metrics_mt_warning.py`, `test_qc_metrics_projection.py`, `test_qc_metrics_fused.py`, `test_qc_metrics_schema_parity.py`, `test_axis_subset.py` | QC edge cases (empty qc_var, MT warning); one obs/var column set across the in-memory, backed and lazy routes and against scanpy, `layer=` and `percent_top=` on each; column-projection correctness on both axes and both routes; fused-vs-unfused bit-identity and a `cpu_profile_snapshot()` pass-count pin; every aligned member (`layers`/`obsm`/`varm`/`obsp`/`varp`) staying in step across `filter_cells`/`filter_genes`/`subset_obs`/HVG-subset, including the no-decode contract for lazy members |
| **Axis subsetting via anndata** | `test_anndata_hooks_compat.py`, `test_anndata_subset_hooks.py` | The private anndata surface the subset path registers with (`as_view` / `_subset` / `to_memory` singledispatch, `_mutated_copy` / `_init_as_actual`, the raw aligned stores) — one assertion per name so an upgrade names its casualty; and the behaviour it buys: `adata[:, mask]` on a backed `X`, `.copy()` still materializing, `raw` and unused categoricals following a subset, atomicity on failure, and the in-place ops keeping `X` lazy and the mapping bridges un-decoded |
| **Harmony / LISI** | `test_harmony.py`, `test_harmony_validation.py`, `test_lisi.py` | Harmony2 batch integration, LISI metric. **The parity claims are gated Rust-side**, not here: `scx-accel/src/harmony/harmony_reference_{values,tests}.rs` and `scx-accel/src/lisi_reference_{values,tests}.rs` pin harmonypy 0.2.0 output under plain `cargo test`. `test_harmony_validation.py` needs gitignored `.npz` fixtures and skips without them — including in CI. |
| **Conversion / export** | `test_h5ad_scx_equivalence.py`, `test_streaming_conversion.py`, `test_to_h5ad.py`, `test_to_h5ad_dispatch.py`, `test_to_h5ad_heterogeneous_categorical.py`, `test_to_h5mu.py`, `test_read_h5ad_metadata.py`, `test_from_10x.py` | h5ad/h5mu streaming ingest/export, metadata read, 10x mtx |
| **Multimodal** | `test_mudata.py`, `test_multimodal_ops_rejection.py`, `test_multimodal_training.py` | MuData round-trip, ops rejection on multimodal, multimodal training |
| **Selective loading** | `test_to_anndata_integration.py`, `test_obsm_loading.py`, `test_obsp_to_h5ad.py` | var_names, obs_filter, layers, obsm, obsp |
| **Fork safety** | `test_fork_safety.py`, `test_fork_deadlock.py`, `test_fork_deadlock_dataloader.py` | Fork-after-init safety, deadlock regression, DataLoader fork |
| **Lifecycle / misc** | `test_python_lifecycle.py`, `test_zero_copy.py`, `test_import_smoke.py`, `test_import_paths.py`, `test_version.py`, `test_ordered_categorical.py`, `test_uns_dataframe_warning.py`, `test_phase3_ergonomics.py`, `test_phase4_streaming.py`, `test_phase5_coo_v2.py` | GC/refcount, zero-copy, import paths, version, categoricals |
| **Index / bitmap** | `test_index_space_regression.py`, `test_bitmap.py`, `test_predicate_index.py`, `test_predicate_index_rewrite.py`, `test_index_plan_dataset.py` | Deletion vector index space, detection bitmap, predicate index |
| **Cloud** | `test_cloud.py`, `test_resumable_pull.py` | Cloud operations, resumable pull |
| **Eval metrics** | `test_eval_metrics.py`, `test_cell_eval_parity.py`, `test_knockdown_efficiency_cpu_parity.py` | Perturbation evaluation, cell-eval parity, knockdown efficiency |
| **Determinism** | `test_parallel_determinism.py`, `test_conformance_vectors.py`, `test_cli_pyscx_parity.py` | Parallel reproducibility, conformance vectors, CLI↔pyscx parity |
| **Misc regression** | `test_obs_shard_oom.py`, `test_t2_warnings.py`, `test_gather_rows_sparse.py`, `test_sparse_cellset_dataset.py` | OOM regression, tier-2 warnings, sparse gather, CellSet |
| `conftest.py`, `_pdex_fixtures.py` | — | Shared fixtures |

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
| `calculate_qc_metrics()` | Max **relative** error over every column both sides write; column-set match | < 1e-5; sets equal | 8.6e-8; equal |
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
| `benchmark_gather_rows_sparse.py` | Sparse row-gather throughput |
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
cover PCA, kNN, UMAP, Leiden, HVG, DE (pdex_ref + Wilcoxon rank-sum), and CSC dispatch.

See [benchmarks/README.md § Regression Gating](../benchmarks/README.md#regression-gating)
for the full gate table and workflow.

## GPU Tests

### Running GPU Tests

```bash
# Rust-level GPU tests. They are #[ignore]d, so --include-ignored is required
# and SCX_REQUIRE_GPU=1 makes a missing device a failure rather than a skip.
SCX_REQUIRE_GPU=1 cargo test -p scx-gpu --lib --tests -- --include-ignored
SCX_REQUIRE_GPU=1 cargo test -p scx-accel --features gpu --lib --tests -- --include-ignored

# On a Chimera GPU node, both suites with the preflight and skip summary:
bash benchmarks/scripts/slurm_scx_gpu_tests.sh

# Python-level GPU accelerator tests
cd pyscx && ../.venv/bin/maturin develop --features hdf5,gpu && \
  ../.venv/bin/pytest tests/test_accel.py -v -k "gpu"
```

### GPU Test Skip Behavior

A GPU test that cannot run is reported as **ignored**, never as passed. Every
one carries two things, and needs both:

| | what it does |
|---|---|
| `#[ignore = "requires a CUDA GPU"]` | libtest does not select it by default, so a plain run counts and names it as ignored |
| `require_gpu!()` (scx-gpu) / `require_gpu_or_skip!()` (scx-accel) | acquires device 0 once the test *is* selected, and returns early if there is none |

Both macros delegate to `scx_gpu::test_gate`, which is a normal `pub` module
rather than `#[cfg(test)]` so that `scx-accel` shares the identical decision.
`require_gpu_cap!(nvcomp)` / `require_gpu_cap!(cuvs)` gate on an optional CUDA
library *below* the device gate.

```bash
cargo test -p scx-gpu                    # 42 passed; 170 ignored   ← honest on a CPU host
cargo test -p scx-gpu --lib --tests -- --include-ignored   # selects them; they skip, printing SCX_GPU_TEST_SKIPPED
SCX_REQUIRE_GPU=1 cargo test -p scx-gpu --lib --tests -- --include-ignored   # 170 FAILED — no device, and one was required
```

⚠️ **`--lib --tests` is load-bearing, not decoration.** Cargo forwards
`--include-ignored` to *every* harness it starts, including rustdoc's — where an
` ```ignore ` fence means the same flag. Three scx-gpu module-doc examples
(`cusolver`, `cusparse`, `gpu_preprocess`) are illustrative pseudo-code fenced
that way, so the unscoped form additionally fails on them. Run doctests
separately with `cargo test -p scx-gpu --doc`.

`SCX_REQUIRE_GPU=1` turns "no device" from a skip into a hard failure;
`SCX_REQUIRE_NVCOMP=1` / `SCX_REQUIRE_CUVS=1` do the same per optional library,
and `SCX_REQUIRE_LARGE_VRAM=1` does it for the free-VRAM floor that gates the
tests needing a multi-GiB allocation to reach a 32-bit index boundary (today:
`colmajor_kernels_do_their_work_past_2_31_elements`, which needs ~10 GiB).
`benchmarks/scripts/_run_scx_gpu_tests.sh` sets `SCX_REQUIRE_GPU=1`, **defaults
`SCX_REQUIRE_LARGE_VRAM=1`** (override with `=0` on a smaller or shared GPU),
passes `--include-ignored`, asserts the result line reports `0 ignored`, and
prints every `SCX_GPU_TEST_SKIPPED` line as a closing summary — so a node
missing nvcomp shows up as listed coverage it did not provide, not as tests that
silently stopped existing. The VRAM default is on for the same reason the device
one is: the job holds a whole GPU via `--gres=gpu:1`, and left opt-in the only
test that can observe a 2³¹ kernel-index overflow could quietly skip while the
suite reported green.

**Why the pairing is enforced.** Before this, `require_gpu!()` expanded to
`eprintln! + return` with no `#[ignore]`, so 170 tests here and 34 in
`scx-accel` were guaranteed no-ops on every CPU host — counted as passed, with
the message swallowed by libtest's output capture. `scx-gpu/tests/gpu_test_gating.rs`
scans both crates' sources and fails if a test has one half without the other,
or decides for itself whether to run (a bare `GpuDevice::new(0)` probe, the
shape 11 of `scx-accel`'s GPU tests used via helper functions). It needs no GPU,
so it runs in CI — where the drift happens.

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
