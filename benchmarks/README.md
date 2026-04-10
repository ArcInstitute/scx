# SCX Benchmarks

This directory contains the benchmarking infrastructure for SCX — scripts, SLURM job definitions, results, and logs for evaluating compression, read/write performance, ML data loader throughput, GPU accelerators, and lazy preprocessing.

For the full benchmarking specification (formats under test, dataset matrix, methodology, metrics), see [COMPREHENSIVE-BENCHMARKING.md](../COMPREHENSIVE-BENCHMARKING.md).

---

## Directory Layout

```
benchmarks/
├── README.md              ← You are here
├── scripts/               # Benchmark scripts and SLURM job definitions
│   ├── run_benchmarks_slurm.sh          # All-in-one SLURM submission (3 sub-jobs)
│   ├── submit_benchmarks.py             # Programmatic SLURM submission via submitit
│   ├── benchmark_all.py                 # Orchestrator: compression + write + read
│   ├── benchmark_*.py                   # Individual Python benchmark scripts
│   ├── benchmark_bpcells.R              # BPCells benchmark (R)
│   ├── slurm_*.sh                       # Standalone SLURM job scripts
│   ├── download_datasets.sh             # Download & convert benchmark datasets
│   ├── download_*.py                    # Dataset-specific downloaders
│   ├── build_census_*.py                # Large dataset builders (5M, 10M cells)
│   ├── build_release.py                 # Helper: ensure pyscx release build
│   ├── verify_datasets.py               # Validate datasets & record metadata
│   └── setup_cloud_test_data.sh         # Cloud benchmark test data setup
├── results/               # Benchmark output (JSON + Markdown reports)
└── logs/                  # SLURM job logs (.out, .err, .log)
```

---

## Prerequisites

1. **Python venv** — All scripts use the project's uv virtualenv at `.venv/`:
   ```bash
   # From repo root:
   uv venv .venv
   uv pip install scanpy anndata cellxgene-census tiledbsoma tiledbsoma-ml submitit
   ```

2. **pyscx release build** — Benchmark scripts automatically rebuild pyscx in release mode. You can also do this manually:
   ```bash
   cd pyscx && ../.venv/bin/maturin develop --release && cd ..
   ```

3. **Rust toolchain** — Required for Rust-level benchmarks (ops, query engine, GPU decode):
   ```bash
   cargo build --release --workspace
   ```

4. **Datasets** — Download benchmark datasets to the data directory (default: `/scratch/ctc/nickyoungblut/scx/`). See [Dataset Preparation](#dataset-preparation) below.

5. **GPU environment** (for GPU benchmarks only) — See [docs/gpu-setup.md](../docs/gpu-setup.md) for CUDA/RAPIDS setup.

---

## Quick Start

### Run all CPU benchmarks via SLURM

```bash
# Submits 3 sub-jobs: (A-C) compression/write/read, (D-E) ops/query, (F) ML loader
bash benchmarks/scripts/run_benchmarks_slurm.sh
```

### Run locally (no SLURM)

```bash
# Smoke test on pbmc3k
.venv/bin/python benchmarks/scripts/submit_benchmarks.py --local --smoke

# Full suite
.venv/bin/python benchmarks/scripts/benchmark_all.py --all
```

---

## Dataset Preparation

Benchmarks expect datasets on local scratch storage. Prepare them before running benchmarks.

### Download & convert datasets D1–D6

```bash
# Interactive (runs directly):
bash benchmarks/scripts/download_datasets.sh [DATA_DIR]

# Via SLURM (8 CPUs, 64 GB, ~4 hours):
sbatch benchmarks/scripts/slurm_prep_datasets.sh
```

### Build large datasets D7–D8 (high-memory nodes)

```bash
# D7: census_5m (500 GB RAM required)
sbatch benchmarks/scripts/slurm_build_census_5m.sh

# D8: census_10m (1.5 TB RAM required)
sbatch benchmarks/scripts/slurm_build_census_10m.sh
```

### Verify datasets

```bash
.venv/bin/python benchmarks/scripts/verify_datasets.py
```

| ID | Name | Cells | Protocol | Source |
|----|------|-------|----------|--------|
| D1 | `pbmc3k` | 2,700 | 10x v2 (UMI) | 10x Genomics |
| D2 | `pbmc10k` | 10,000 | 10x v3 (UMI) | 10x Genomics |
| D3 | `smartseq2` | ~50,000 | Smart-seq2 | CELLxGENE Census |
| D4 | `tabula_sapiens_100k` | 100,000 | 10x (UMI) | CELLxGENE Census |
| D5 | `census_500k` | 500,000 | 10x (UMI) | CELLxGENE Census |
| D6 | `census_1m` | 1,000,000 | 10x (UMI) | CELLxGENE Census |
| D7 | `census_5m` | 5,000,000 | 10x (UMI) | CELLxGENE Census |
| D8 | `census_10m` | 10,000,000 | Mixed | CELLxGENE Census |

By default, datasets are stored in `/scratch/ctc/nickyoungblut/scx/` (override with `SCX_DATA_DIR` env var).

---

## SLURM Job Reference

### Dataset Preparation

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `slurm_prep_datasets.sh` | `cpu` | 8 CPUs, 64 GB | 4 h | Prepare D1–D6 |
| `slurm_build_census_5m.sh` | `cpu_high_mem` | 16 CPUs, 500 GB | 8 h | Build D7 from chunks |
| `slurm_build_census_10m.sh` | `cpu_high_mem` | 16 CPUs, 1.5 TB | 12 h | Build D8 from chunks |

### CPU Benchmarks

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `run_benchmarks_slurm.sh` | `cpu` | 16 CPUs, 80 GB | 4 h | All-in-one (compression + write + read + ops + query + ML loader; submits 3 sub-jobs) |
| `slurm_phase0_baseline.sh` | `cpu` | 32 CPUs, 80 GB | 2 h/job | Phase 0 regression baseline D1–D4 (submits 24 parallel jobs + archival) |
| `slurm_phase0_baseline_large.sh` | varies | 32 CPUs, 80–500 GB | 2–4 h/job | Phase 0 regression baseline D5–D7 (submits 18 parallel jobs + archival) |
| `slurm_phase3_small.sh` | `cpu` | 32 CPUs, 80 GB | 6 h | Phase 3 benchmarks D1–D4 (sequential, single job) |
| `slurm_phase3_large.sh` | `cpu_high_mem` | 32 CPUs, 500 GB | 12 h | Phase 3 benchmarks D5–D7 (sequential, single job) |
| `slurm_phase3_parallel_large.sh` | varies | 32 CPUs, 80–500 GB | 4–12 h/job | Phase 3 D5–D7 parallel submission (one job per benchmark×dataset) |
| `slurm_phase3_parallel_scaling_d5d7.sh` | varies | 32 CPUs, 80–500 GB | 6–16 h/job | Phase 3 parallel_scaling only, D5–D7 |
| `slurm_fused_bench.sh` | `cpu` | 8 CPUs, 32 GB | 30 min | Fused normalize+log1p microbenchmarks |
| `slurm_lazy_preprocess_bench.sh` | `cpu` | 8 CPUs, 64 GB | 2 h | Phase 4d lazy preprocessing benchmarks |
| `slurm_phase4_ml_loader.sh` | `cpu`+`gpu` | 16–32 CPUs, 32–200 GB | 1–6 h/job | Phase 4 ML loader throughput (submits parallel CPU + GPU jobs per dataset) |
| `slurm_phase5_accel_bench.sh` | `cpu` | 16 CPUs, 16–256 GB | 1–16 h/job | Phase 5 accelerator benchmarks (PCA, kNN, UMAP, DE, pipeline; submits ~20 parallel jobs) |

### GPU Benchmarks

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `slurm_gpu_bench.sh` | `gpu` | 1 GPU, 8 CPUs, 32 GB | 30 min | GPU decode microbenchmarks |
| `slurm_gpu_knn_bench.sh` | `gpu` | 1 GPU, 16 CPUs, 64 GB | 1 h | GPU kNN (CAGRA) validation + benchmark |
| `slurm_gpu_pca_opt_bench.sh` | `gpu` | 1 GPU, 16 CPUs, 128 GB | 2 h | GPU PCA optimization benchmark |
| `slurm_gpu_analysis_bench.sh` | `gpu` | 1 GPU, 16 CPUs, 128 GB | 4 h | Full GPU suite: PCA + kNN + UMAP + preprocessing + pipeline |

### Programmatic Submission (submitit)

```bash
# Smoke test (pbmc3k, CPU-only)
.venv/bin/python benchmarks/scripts/submit_benchmarks.py --smoke

# All datasets, CPU
.venv/bin/python benchmarks/scripts/submit_benchmarks.py --all-datasets

# Single dataset with GPU
.venv/bin/python benchmarks/scripts/submit_benchmarks.py --dataset tabula_sapiens_100k --n-gpus 1

# GPU scaling sweep (0, 1, 2, 4 GPUs)
.venv/bin/python benchmarks/scripts/submit_benchmarks.py --all-datasets --gpu-sweep

# Run locally without SLURM (for debugging)
.venv/bin/python benchmarks/scripts/submit_benchmarks.py --local --smoke
```

---

## Submitting SLURM Jobs

> [!IMPORTANT]
> **Always prefer parallel job submission over sequential single-job scripts.** Each independent (benchmark, dataset) combination should be submitted as a separate SLURM job so they run concurrently across cluster nodes. This dramatically reduces wall-clock time (e.g., 24 parallel jobs finishing in ~2h vs one sequential job taking ~4h). Use `--dependency=afterok:$JOB1:$JOB2:...` for any post-processing that must wait for all benchmarks to complete (e.g., archiving results). Scale memory per dataset — not every job needs the largest allocation. See `slurm_phase0_baseline.sh` and `slurm_phase0_baseline_large.sh` for the reference pattern.

### Basic submission

```bash
sbatch benchmarks/scripts/slurm_gpu_analysis_bench.sh
```

### Adding exclusive node access (recommended for reproducibility)

```bash
sbatch --exclusive benchmarks/scripts/slurm_gpu_analysis_bench.sh
```

### Recommended execution order

```
1. Prepare datasets (run first — benchmarks depend on these):
   sbatch benchmarks/scripts/slurm_prep_datasets.sh
   sbatch benchmarks/scripts/slurm_build_census_5m.sh    # (optional, high-memory)
   sbatch benchmarks/scripts/slurm_build_census_10m.sh   # (optional, high-memory)

2. CPU benchmarks (after datasets are ready):
   bash benchmarks/scripts/run_benchmarks_slurm.sh
   sbatch benchmarks/scripts/slurm_fused_bench.sh
   sbatch benchmarks/scripts/slurm_lazy_preprocess_bench.sh

3. GPU benchmarks (require gpu partition):
   sbatch benchmarks/scripts/slurm_gpu_analysis_bench.sh
   sbatch benchmarks/scripts/slurm_gpu_bench.sh
   sbatch benchmarks/scripts/slurm_gpu_knn_bench.sh
   sbatch benchmarks/scripts/slurm_gpu_pca_opt_bench.sh
```

> [!IMPORTANT]
> Always run dataset preparation jobs first. The benchmark scripts assume datasets exist at the `SCX_DATA_DIR` path (default: `/scratch/ctc/nickyoungblut/scx/`).

---

## Monitoring Jobs

```bash
# Check job queue
squeue -u $USER

# Watch a running job's log
tail -f benchmarks/logs/<script>_<JOBID>.log

# Check completed job output
cat benchmarks/logs/<script>_<JOBID>.log
```

---

## Environment Notes

**CPU benchmarks** use the project's uv virtualenv (`.venv/`). The scripts reference it automatically.

**GPU benchmarks** use different environments depending on the script:

| Script | Environment | Reason |
|--------|-------------|--------|
| `slurm_gpu_analysis_bench.sh` | conda `scx-gpu` env | Full CUDA + RAPIDS stack (cuVS, cuGraph) |
| `slurm_gpu_pca_opt_bench.sh` | conda `scx-gpu` env | Same as above |
| `slurm_gpu_knn_bench.sh` | uv `.venv/` + `LD_LIBRARY_PATH` | pip-installed RAPIDS, manual lib paths |
| `slurm_gpu_bench.sh` | Rust-only (cargo) | No Python RAPIDS deps |

All GPU scripts rebuild pyscx with `--features gpu` before running:

```bash
cd pyscx && maturin develop --release --features gpu && cd ..
```

See [docs/gpu-setup.md](../docs/gpu-setup.md) for full GPU environment setup instructions.

---

## Results

Benchmark outputs are organized into two directories:

- **`benchmarks/results/`** — Per-script benchmark results (JSON + Markdown reports, GPU Go/No-Go gates)
- **`benchmarks/comprehensive/results/raw/`** — Phase 3 comprehensive benchmarks (500+ JSON files, one per benchmark×format×dataset)
- **`benchmarks/comprehensive/reporting/phase3_report.md`** — Phase 3 comprehensive report (D1–D7, 12 formats, 6 benchmarks)

To list comprehensive results:

```bash
ls benchmarks/comprehensive/results/raw/*.json | wc -l   # ~500 results
ls benchmarks/comprehensive/results/raw/*census_1m*       # D6 results
```

---

## Individual Benchmark Scripts

| Script | What it measures |
|--------|-----------------|
| `benchmark_all.py` | Orchestrator for compression + write + read (A-C) |
| `benchmark_compression.py` | On-disk sizes across codecs and formats |
| `benchmark_write.py` | h5ad → SCX conversion throughput |
| `benchmark_read.py` | Full-file read performance |
| `benchmark_query.py` | Query engine (selective reads, metadata filters) |
| `benchmark_ops.py` | File operations (append, compact, merge) |
| `benchmark_loader.py` | ML data loader throughput vs SOTA baselines |
| `benchmark_parallel_read.py` | Parallel read scaling (1–32 threads) |
| `benchmark_auto_codec.py` | Auto-codec selection accuracy and performance |
| `benchmark_cli.py` | CLI command performance |
| `benchmark_python_bindings.py` | Python bindings overhead |
| `benchmark_cloud.py` | Cloud storage read performance |
| `benchmark_compressed_h5ad.py` | Compressed h5ad baseline (gzip, lzf) |
| `benchmark_gpu_decode.py` | GPU decode microbenchmarks (cuSPARSE, bitstream) |
| `benchmark_gpu_pca.py` | GPU PCA validation + timing |
| `benchmark_gpu_knn.py` | GPU kNN (CAGRA) validation + timing |
| `benchmark_gpu_umap.py` | GPU UMAP validation + timing |
| `benchmark_gpu_preprocess.py` | GPU fused preprocessing (normalize+log1p) |
| `benchmark_gpu_pipeline.py` | End-to-end GPU pipeline + Go/No-Go gate |
| `benchmark_gpu_scvi.py` | GPU scVI training benchmark |
| `benchmark_lazy_preprocess.py` | Phase 4d lazy transforms, memory, column projection |
| `benchmark_accelerators.py` | Phase 4b CPU accelerators: PCA, kNN, UMAP, DE vs scanpy |
| `benchmark_accel_pipeline.py` | Full pipeline (3 variants: SCX OOC, SCX preprocess, scanpy) |
| `benchmark_accel_preprocessing.py` | pyscx.preprocess() vs scanpy normalize+log1p |
| `benchmark_bpcells.R` | BPCells comparison (R) |
