# SCX Testing & Benchmarks

## Rust Tests

**Run**: `cargo test --workspace` (all crates) or `cargo test --workspace --features cloud` (with cloud features)

| Crate | Location | Key tests |
|-------|----------|-----------|
| `scx-format` | `tests/integration.rs` | Round-trip validation, minimum file size, checksum verification |
| `scx-codec` | Per-module unit tests | Rice, FOR-BP, Delta-Golomb encode/decode round-trips |
| `scx-sparse` | Unit tests | CSR construction, row slicing, dense conversion |
| `scx-ops` | `tests/` | Append, delete, compact, rollback, merge, flock concurrency |
| `scx-engine` | Unit tests | Predicate parsing, pipeline validation, pushdown, fused ops |
| `scx-loader` | Unit tests | Pipeline lifecycle, batch format, shuffle, projection, normalize |
| `scx-cloud` | `tests/` | Explode/pack round-trip, cloud-optimize, pull/push (local backend) |
| `scx-accel` | Unit tests | PCA round-trip, kNN recall, UMAP trustworthiness, DE p-values, pseudobulk aggregation |
| `scx-gpu` | Unit tests | CUDA Rice/FOR-BP decode, sparse-to-dense, GPU parity with CPU reference |
| `rscx` | via `R CMD check` | Seurat/SCE round-trip, query, CSR transpose |

## Python Test Suite

**Location**: `pyscx/tests/`
**Run**: `cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v`
**With cloud**: `../.venv/bin/maturin develop --features cloud && ../.venv/bin/pytest tests/ -v`

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
| `conftest.py` | Shared pytest fixtures |

## Benchmarks

**Location**: `benchmarks/scripts/`
**Results**: `benchmarks/results/`

| Script | Purpose |
|--------|---------|
| `benchmark_all.py` | Runs all Phase 1 benchmarks together |
| `benchmark_compression.py` | Compression ratio vs h5ad (target: < 60%) |
| `benchmark_read.py` | Read performance (scx vs h5ad) |
| `benchmark_write.py` | h5ad → scx conversion speed (MB/s) |
| `benchmark_auto_codec.py` | Auto-codec selection performance |
| `benchmark_parallel_read.py` | Parallel shard decode scaling (1-8 threads) |
| `benchmark_compressed_h5ad.py` | SCX vs gzip/lzf-compressed h5ad |
| `benchmark_ops.py` | File operations (append, compact, delete, merge) |
| `benchmark_query.py` | Query engine: shard skip rate, latency, pushdown |
| `benchmark_loader.py` | Training loader throughput, SOMA comparison |
| `benchmark_gpu_scvi.py` | scVI GPU utilization on 10M-cell dataset |
| `benchmark_cli.py` | CLI operation timing |
| `benchmark_python_bindings.py` | Python bindings performance |
| `benchmark_bpcells.R` | BPCells R comparison |
| `download_datasets.sh` | Fetches test data (PBMC 3K, Tabula Sapiens, CELLxGENE Census, Smart-seq2) |
| `setup_cloud_test_data.sh` | Set up cloud test data in GCS bucket |
| `run_benchmarks_slurm.sh` | SLURM job script for HPC benchmarks |
| `submit_benchmarks.py` | Submit and manage benchmark SLURM jobs |

### Running Benchmarks

```bash
# Download test datasets first:
bash benchmarks/scripts/download_datasets.sh

# Run all Phase 1 benchmarks:
.venv/bin/python benchmarks/scripts/benchmark_all.py

# Run individual benchmarks:
.venv/bin/python benchmarks/scripts/benchmark_compression.py
.venv/bin/python benchmarks/scripts/benchmark_read.py
.venv/bin/python benchmarks/scripts/benchmark_query.py
.venv/bin/python benchmarks/scripts/benchmark_loader.py

# Run on HPC (SLURM):
python benchmarks/scripts/submit_benchmarks.py
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
