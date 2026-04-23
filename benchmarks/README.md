# SCX Benchmarks

This directory contains the benchmarking infrastructure for SCX — scripts, SLURM job definitions, results, and logs for evaluating compression, read/write performance, ML data loader throughput, GPU accelerators, and lazy preprocessing.

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
├── comprehensive/         # Comprehensive benchmark suite (Phase 3+)
│   ├── config.py                        # Dataset paths, format configs, constants
│   ├── sysinfo.py                       # System info collector (CPU, RAM, OS, disk)
│   ├── results.py                       # BenchmarkResult schema + JSON writer
│   ├── convert.py                       # Shared conversion cache (convert once, reuse)
│   ├── envs/                            # Conda environment definitions
│   │   ├── scx-bench.yml                # CPU benchmark environment
│   │   ├── scx-bench-gpu.yml            # GPU benchmark environment (CUDA + RAPIDS)
│   │   ├── scx-bench-r.yml              # R / BPCells benchmark environment
│   │   └── scx-bench-slaf.yml           # SLAF (slafdb) benchmark environment
│   ├── runners/                         # Per-format benchmark runners
│   │   ├── base.py                      # Abstract FormatRunner interface
│   │   ├── h5ad_runner.py               # h5ad (uncompressed, gzip, lzf)
│   │   ├── zarr_runner.py               # Zarr (zstd, blosc-lz4)
│   │   ├── tiledb_runner.py             # TileDB-SOMA
│   │   ├── scx_runner.py                # SCX (auto, none, scx1, zstd, pcodec, lz4)
│   │   ├── bpcells_runner.py            # BPCells (R subprocess)
│   │   ├── parquet_runner.py            # Parquet (pyarrow)
│   │   └── slaf_runner.py               # SLAF (slafdb — Lance + DuckDB)
│   ├── benchmarks/                      # Benchmark modules (one per dimension)
│   ├── reporting/                       # Report generation (markdown, plots, tables)
│   ├── scripts/                         # Orchestrators and SLURM launchers
│   │   ├── run_all.py                   # Serial benchmark orchestrator
│   │   ├── run_parallel.py              # Parallel SLURM launcher via submitit
│   │   ├── run_slurm.sh                 # SLURM submission script
│   │   ├── install_dependencies.sh      # Create conda environments
│   │   ├── validate_*.py                # Correctness validation scripts
│   │   ├── slurm_*.sh                   # SLURM job scripts
│   │   │
│   │   │  # Regression gating (on-demand)
│   │   ├── gate_candidate.sh            # One-shot: capture + gate against LATEST baseline
│   │   ├── capture_baseline.py          # Freeze one snapshot (raw/ + summary.json + manifest)
│   │   ├── promote_baseline.py          # Promote a snapshot to results/baselines/<version>/
│   │   ├── compare_against_baseline.py  # Relative + absolute-floor + justification gate
│   │   ├── publish_dashboard.py         # Rsync HTML snapshot to DASHBOARD_PUBLISH_TARGET
│   │   │
│   │   │  # Cloud infrastructure (GCP)
│   │   ├── check_gcp_auth.py            # Credentials + bucket round-trip preflight
│   │   ├── setup_cloud_test_data.sh     # Idempotent fixture staging via BLAKE3 sidecars
│   │   ├── cloud_fixtures_doctor.py     # Dry-run probe: local vs cloud fixture drift
│   │   ├── submit_gcp_matrix.py         # Instance-type × benchmark launcher (--yes-spend)
│   │   ├── gcs_lifecycle.json           # Committed GCS object-lifecycle rules
│   │   │
│   │   │  # Observability + migrations
│   │   ├── watch.py                     # Rich TUI tailing run_manifest.json + submitit logs
│   │   └── migrate_results.py           # Back-stamp schema_version on legacy raw JSONs
│   ├── r_scripts/                       # BPCells R benchmark scripts
│   └── results/                         # Raw JSON results + generated reports
│       ├── raw/                         # One JSON per benchmark×format×dataset
│       └── reports/                     # Generated markdown + plots
├── results/               # Per-script benchmark output (JSON + Markdown reports)
└── logs/                  # SLURM job logs (.out, .err, .log)
```

---

## Formats Under Test

The benchmark suite compares SCX against all relevant single-cell data formats:

### Primary Competitors

| Format | Variant | Library / Tool | Notes |
|--------|---------|---------------|-------|
| **h5ad** (uncompressed) | CSR in HDF5, no filter | `anndata` + `h5py` | Default of `adata.write_h5ad()` (`compression=None`) |
| **h5ad** (gzip) | CSR in HDF5, gzip level 4 | `anndata` + `h5py` | Commonly used by researchers |
| **h5ad** (lzf) | CSR in HDF5, lzf filter | `anndata` + `h5py` | Faster alternative to gzip |
| **Zarr** (zstd) | CSR arrays, Zarr v3, zstd level 3 | `zarr` >= 3.0 | Chunked, cloud-native |
| **Zarr** (blosc-lz4) | CSR arrays, Zarr v3, blosc-lz4 | `zarr` >= 3.0 | Fast decompression variant |
| **TileDB-SOMA** | SOMAExperiment | `tiledbsoma` >= 2.3 + `tiledbsoma_ml` | CELLxGENE Census native format |
| **SLAF** | SLAF directory (Lance + DuckDB) | `slafdb` >= 0.5.2 | SQL-native lazy format. Isolated `scx-bench-slaf` env (DuckDB/Lance conflict with TileDB/Zarr pins) |
| **SCX** | auto, none, scx1, zstd, pcodec, lz4 | `pyscx` / `scx-cli` | System under test (multiple codec variants) |

### Additional Competitors

| Format | Library / Tool | Notes |
|--------|---------------|-------|
| **BPCells** | `BPCells` R package | Bitpacking, disk-backed streaming. R-only; benchmarked via `Rscript` subprocess. |
| **Parquet** (zstd) | `pyarrow` | Columnar; store CSR arrays as columns. |

### Out of Scope

| Format | Reason |
|--------|--------|
| Lance (raw) | Covered through SLAF, which wraps Lance |
| DuckDB / AnnSQL | SQL query engine, not a storage format |
| Loom | Deprecated in favor of h5ad |
| 10x HDF5 (.h5) | Legacy input format, not used for analysis storage |

---

## Benchmark Methodology

### Benchmark Dimensions

The suite measures seven core dimensions, plus accelerator, GPU, lazy preprocessing, and correctness benchmarks:

| Dimension | Script(s) | What it measures |
|-----------|-----------|------------------|
| **Compression** (3.1) | `compression.py` | On-disk file size for every format x dataset |
| **Write** (3.2) | `write.py` | h5ad -> target format conversion time, throughput, peak RSS |
| **Read Full** (3.3) | `read_full.py` | Full expression matrix read into in-memory CSR |
| **Selective Read** (3.4) | `read_selective.py` | Row slices, column projection (2K HVGs), filtered queries |
| **Parallel Scaling** (3.5) | `parallel_scaling.py`, `parallel_write_scaling.py` | Read/write throughput vs thread count (1, 2, 4, 8, 16, 32) |
| **ML Loader** (3.6) | `ml_loader.py` | Batched iteration throughput (batches/sec, TTFB, peak RSS) |
| **Memory** (3.7) | `memory.py` | Peak RSS during common operations |
| **Cell-eval parity perf** (3.15) | `cell_eval_parity_perf.py` | SCX `pyscx.accel.*` perturbation metrics vs cell-eval / arc-bench reference, on synthetic perturbation datasets at 100K–1M cells |

### Measurement Protocol

- **Timing**: `time.perf_counter()` for wall-clock. Median of 3 runs (large datasets) or 5 runs (small datasets).
- **Cold start**: Fresh Python subprocess per run to avoid warm-up artifacts.
- **Cache control**: Warm-cache = 3 warm-up reads discarded. Cold-cache = `sync; echo 3 > /proc/sys/vm/drop_caches` between runs (requires root).
- **Memory**: Peak RSS via `/proc/self/status` or `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
- **Parallel scaling**: Each thread count runs in a separate subprocess so rayon/thread pools are created fresh. SCX parallelism controlled via `RAYON_NUM_THREADS` env var.
- **Reproducibility**: Use SLURM `--exclusive` for CPU binding. Record `uname -a`, CPU model, RAM, and storage device for every run (via `sysinfo.py`). All dependencies pinned in conda `environment.yml` files.
- **Directory sizes**: For multi-file formats (Zarr, TileDB-SOMA, BPCells), measure with `du -sb` on the full directory.

### Output Format

All results are stored as structured JSON in `comprehensive/results/raw/`:

```json
{
  "benchmark": "read_full",
  "format": "scx_auto",
  "dataset": "census_1m",
  "timestamp": "2026-03-25T10:00:00",
  "system": { "hostname": "...", "cpu": "...", "ram_gb": 2113 },
  "runs": [
    { "wall_s": 12.491, "user_s": 11.2, "sys_s": 1.1, "peak_rss_mb": 264.2 },
    ...
  ],
  "median_wall_s": 12.491,
  "file_size_bytes": 2470000000
}
```

---

## Benchmark Environments

The comprehensive benchmark suite uses **isolated conda environments** for reproducibility, separate from the development `.venv/`. Environment definitions are in `comprehensive/envs/`:

| Environment | Purpose | Key additions |
|-------------|---------|---------------|
| `scx-bench` | CPU benchmarks: format comparisons, accelerators, lazy preprocessing, correctness validation, ML loaders | All Python deps + PyTorch (CPU) |
| `scx-bench-gpu` | GPU benchmarks: CUDA-accelerated PCA, kNN, UMAP, Leiden, fused preprocessing | Extends CPU deps with `cuda-version`, `cuvs`, `cugraph`, PyTorch (CUDA) |
| `scx-bench-r` | BPCells benchmarks | R, Matrix, HDF5, BPCells (from GitHub) |
| `scx-bench-slaf` | SLAF (slafdb) benchmarks — DuckDB/Lance backend conflicts with the main env's pins, so SLAF runs alone | `slafdb`, Polars, DuckDB, Lance (pip), PyTorch (CPU) for SLAFDataLoader |

### Setup

```bash
# Create all environments (one-time)
bash benchmarks/comprehensive/scripts/install_dependencies.sh --all

# Or create individually
bash benchmarks/comprehensive/scripts/install_dependencies.sh          # CPU only (default)
bash benchmarks/comprehensive/scripts/install_dependencies.sh --gpu    # GPU + RAPIDS
bash benchmarks/comprehensive/scripts/install_dependencies.sh --r      # R + BPCells

# Verify installations
bash benchmarks/comprehensive/scripts/install_dependencies.sh --check

# Rebuild pyscx inside an environment
bash benchmarks/comprehensive/scripts/install_dependencies.sh --rebuild
bash benchmarks/comprehensive/scripts/install_dependencies.sh --rebuild --gpu
```

> [!IMPORTANT]
> The `scx-bench-gpu` environment pins `cuda-version` to match the NVIDIA driver. The default is `12.2` (for driver 535.x). Edit `benchmarks/comprehensive/envs/scx-bench-gpu.yml` to adjust for your driver version. See [docs/gpu-setup.md](../docs/gpu-setup.md) for the driver compatibility table.

### Which environment to use

**Comprehensive benchmark suite** (`comprehensive/`):

| Script | Environment | Activation |
|--------|-------------|------------|
| `comprehensive/scripts/run_all.py` | `scx-bench` | `conda activate scx-bench` |
| `comprehensive/scripts/run_parallel.py` | `scx-bench` | `conda activate scx-bench` |
| `comprehensive/scripts/validate_*.py` | `scx-bench` | `conda activate scx-bench` |
| GPU benchmarks | `scx-bench-gpu` | `conda activate scx-bench-gpu` |
| BPCells benchmarks | `scx-bench-r` | `conda activate scx-bench-r` |
| SLAF benchmarks | `scx-bench-slaf` | `conda activate scx-bench-slaf` |

**Legacy scripts** (`scripts/`) still reference the dev `.venv/` (CPU) and the `scx-gpu` conda env (GPU), and are preserved as-is.

SLURM scripts in `comprehensive/scripts/` auto-detect the correct environment. Override with `--conda-env`:

```bash
bash benchmarks/comprehensive/scripts/run_slurm.sh --conda-env scx-bench-gpu
```

---

## Parallel Benchmark Execution (run_parallel.py)

The serial orchestrator (`run_all.py`) processes benchmarks sequentially within a single SLURM job. For faster execution, `run_parallel.py` uses two-phase parallel execution via `submitit`:

**Phase A — Convert once.** Each (dataset, format) pair is converted exactly once and written to a persistent path. Conversions run as independent parallel SLURM jobs. Existing files are skipped automatically (`--overwrite` to force).

**Phase B — Benchmark in parallel.** Each (benchmark, dataset, format) triple is submitted as an independent SLURM job reading from the pre-converted file. All jobs run concurrently.

```
Serial (run_all.py):     420 tasks x avg 3 min = ~21 hours wall time
Parallel (run_parallel.py): conversions + benchmarks, all concurrent
                            Wall time ~ max(single slowest job) ~ 50 min
```

### Usage

```bash
# Small datasets (D1-D4)
python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k

# Large datasets — high-memory partition
python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets census_500k census_1m census_5m \
    --partition cpu_preemptible --mem-gb 500 --timeout 480

# Specific benchmarks and formats only
python benchmarks/comprehensive/scripts/run_parallel.py \
    --benchmarks read_full read_selective \
    --formats scx_auto zarr_zstd h5ad_gzip \
    --datasets census_1m

# Dry run — show job count without submitting
python benchmarks/comprehensive/scripts/run_parallel.py --dry-run

# Skip conversion phase (reuse existing pre-converted files)
python benchmarks/comprehensive/scripts/run_parallel.py --skip-convert
```

submitit logs are written to `comprehensive/logs/submitit/`. Each SLURM job writes its result JSON independently to `comprehensive/results/raw/` — filenames are unique per triple, so concurrent writes are safe.

---

## Cloud Benchmarks (GCP)

The comprehensive suite validates cloud behavior against GCP only — AWS S3 and Azure Blob coverage is deferred until a second-provider requirement lands. Cloud benchmarks live under `comprehensive/benchmarks/cloud_*.py` and run through the same `run_parallel.py` launcher as every other benchmark:

| Benchmark | Coverage | What it measures |
|-----------|----------|------------------|
| `cloud_push`     | SCX-only | Local `.scx` → `gs://…/.scxd/` upload throughput |
| `cloud_pull`     | SCX-only | `gs://…/.scxd/` → local `.scx` download throughput |
| `cloud_read`     | Cross-format | Full in-memory read directly from GCS (SCX pull-then-read; Zarr / SOMA / SLAF via their native GCS paths) |
| `cloud_metadata` | Cross-format | Metadata-only open latency (`pyscx.open_cloud`, `zarr.open`, `Experiment.open`, `SLAFArray(url)`) |

**Prerequisites**:
1. GCP service account `scx-bench@c-tc-429521.iam.gserviceaccount.com` with bucket-scoped `roles/storage.objectAdmin` on `gs://arc-ctc-nextflow/`. One-time bootstrap (requires `roles/iam.serviceAccountAdmin` + `roles/resourcemanager.projectIamAdmin`):
   ```bash
   gcloud config set project c-tc-429521
   gcloud iam service-accounts create scx-bench \
       --display-name "SCX Benchmark Runner" --project c-tc-429521
   gcloud storage buckets add-iam-policy-binding gs://arc-ctc-nextflow \
       --member=serviceAccount:scx-bench@c-tc-429521.iam.gserviceaccount.com \
       --role=roles/storage.objectAdmin
   gcloud iam service-accounts keys create ~/.gcp/scx-bench.json \
       --iam-account=scx-bench@c-tc-429521.iam.gserviceaccount.com
   chmod 600 ~/.gcp/scx-bench.json
   ```
   Rotate the key every 90 days; never commit it or paste it into chat.
2. Point the harness at the key via the repo-root `.env` file — it's
   auto-loaded by `benchmarks/scripts/bench_env.py` through
   `python-dotenv`, so no shell export is required. Tildes are expanded.
   `cloud_fixtures.require_gcp_credentials` falls back to `~/.gcp/scx-bench.json`
   automatically if the env var is unset.
   ```bash
   echo 'GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/scx-bench.json' >> .env
   ```
   An uncommented template line lives in `.env.example`. Operators who
   prefer shell exports can still use them — process env wins over `.env`.
3. Bucket knobs are env-configurable (defaults in parens): `GCS_TEST_BUCKET` (`gs://arc-ctc-nextflow/scx-test`), `GCP_PROJECT` (`c-tc-429521`), `GCP_BUCKET_REGION` (`us-central1`).

**Usage**:

```bash
# GOOGLE_APPLICATION_CREDENTIALS resolved from .env; see step 2 above.
python benchmarks/comprehensive/scripts/run_parallel.py \
    --benchmarks cloud_push cloud_pull cloud_read cloud_metadata \
    --datasets pbmc3k tabula_sapiens_100k \
    --formats scx_auto zarr_zstd tiledb_soma slaf
```

Fixtures self-heal on first run: if the expected cloud object doesn't exist, `cloud_fixtures.ensure_cloud_fixture` uploads it from the local converted file via `pyscx.push` (SCX) or `gsutil -m rsync` (other formats). SCX-only benchmarks return `None` for non-SCX formats so the orchestrator silently skips those triples.

The legacy `benchmarks/scripts/benchmark_cloud.py` is a thin deprecation shim — it prints a banner and forwards to the launcher above.

---

## Known Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| OOM when creating 10M cell h5ad | Use SCX `merge` to build directly from chunk h5ad files. For h5ad baseline, use incremental h5py writes or run on a >= 500 GB RAM SLURM node. |
| TileDB-SOMA-ML import path | The ML wrapper is a separate package (`pip install tiledbsoma-ml`). Import as `from tiledbsoma_ml import ExperimentDataset`. |
| BPCells requires R | Use the `scx-bench-r` conda environment. Run via `Rscript` subprocess. Parse JSON output from R scripts. |
| h5ad gzip/lzf conversion at scale | Pre-generate compressed h5ad variants for all datasets. `anndata.write_h5ad()` defaults to `compression=None` (uncompressed); gzip/lzf must be set explicitly. |
| Warm vs cold cache inconsistency | Warm-cache: 3 warm-up reads discarded. Cold-cache: `drop_caches` between each run (requires root). Report both. |
| Non-deterministic timing on shared nodes | Use SLURM `--exclusive` flag. If not available, run at least 5 repeats and report median + IQR. |
| Formats without native subsetting | For h5ad: read full file then subset in memory. Record total time and note in results. |

---

## Prerequisites

1. **Environment config** — Copy `.env.example` to `.env` and set `SCX_WORK_DIR` to your base data directory:
   ```bash
   cp .env.example .env
   # Edit .env: set SCX_WORK_DIR=/path/to/your/scx/workdir
   ```
   `SCX_DATA_DIR` defaults to `$SCX_WORK_DIR/benchmarks/datasets` if not set.

2. **Python venv** — All scripts use the project's uv virtualenv at `.venv/`:
   ```bash
   # From repo root:
   uv venv .venv
   uv pip install scanpy anndata cellxgene-census tiledbsoma tiledbsoma-ml submitit python-dotenv
   ```

3. **pyscx release build** — Benchmark scripts automatically rebuild pyscx in release mode. You can also do this manually:
   ```bash
   cd pyscx && ../.venv/bin/maturin develop --release && cd ..
   ```

4. **Rust toolchain** — Required for Rust-level benchmarks (ops, query engine, GPU decode):
   ```bash
   cargo build --release --workspace
   ```

5. **Datasets** — Download benchmark datasets to the data directory. Paths are configured via `SCX_WORK_DIR` and `SCX_DATA_DIR` environment variables (see `.env` at repo root). See [Dataset Preparation](#dataset-preparation) below.

6. **GPU environment** (for GPU benchmarks only) — See [docs/gpu-setup.md](../docs/gpu-setup.md) for CUDA/RAPIDS setup.

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

Dataset paths are configured via the `.env` file at the repo root (or environment variables). `SCX_WORK_DIR` sets the base working directory; `SCX_DATA_DIR` defaults to `$SCX_WORK_DIR/benchmarks/datasets`.

---

## SLURM Job Reference

### Dataset Preparation

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `slurm_prep_datasets.sh` | `cpu_preemptible` | 8 CPUs, 64 GB | 4 h | Prepare D1–D6 |
| `slurm_build_census_5m.sh` | `cpu_high_mem` | 16 CPUs, 500 GB | 8 h | Build D7 from chunks |
| `slurm_build_census_10m.sh` | `cpu_high_mem` | 16 CPUs, 1.5 TB | 12 h | Build D8 from chunks |

### CPU Benchmarks

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `run_benchmarks_slurm.sh` | `cpu_preemptible` | 16 CPUs, 80 GB | 4 h | All-in-one (compression + write + read + ops + query + ML loader; submits 3 sub-jobs) |
| `slurm_phase0_baseline.sh` | `cpu_preemptible` | 32 CPUs, 80 GB | 2 h/job | Phase 0 regression baseline D1–D4 (submits 24 parallel jobs + archival) |
| `slurm_phase0_baseline_large.sh` | varies | 32 CPUs, 80–500 GB | 2–4 h/job | Phase 0 regression baseline D5–D7 (submits 18 parallel jobs + archival) |
| `slurm_phase3_small.sh` | `cpu_preemptible` | 32 CPUs, 80 GB | 6 h | Phase 3 benchmarks D1–D4 (sequential, single job) |
| `slurm_phase3_large.sh` | `cpu_high_mem` | 32 CPUs, 500 GB | 12 h | Phase 3 benchmarks D5–D7 (sequential, single job) |
| `slurm_phase3_parallel_large.sh` | varies | 32 CPUs, 80–500 GB | 4–12 h/job | Phase 3 D5–D7 parallel submission (one job per benchmark×dataset) |
| `slurm_phase3_parallel_scaling_d5d7.sh` | varies | 32 CPUs, 80–500 GB | 6–16 h/job | Phase 3 parallel read scaling only, D5–D7 |
| `slurm_parallel_write_scaling.sh` | varies | 32 CPUs, 80–500 GB | 2–24 h/job | Parallel write scaling (§3.5.2), D1–D7 (one job per dataset) |
| `slurm_fused_bench.sh` | `cpu_preemptible` | 8 CPUs, 32 GB | 30 min | Fused normalize+log1p microbenchmarks |
| `slurm_lazy_preprocess_bench.sh` | `cpu_preemptible` | 16–32 CPUs, 80–256 GB | 6–16 h/job | Phase 5c lazy preprocessing benchmarks (§3.13.1–3.13.6; submits ~15 parallel jobs) |
| `slurm_phase4_ml_loader.sh` | `cpu_preemptible`+`preemptible` | 16–32 CPUs, 32–200 GB | 1–6 h/job | Phase 4 ML loader throughput (submits parallel CPU + GPU jobs per dataset) |
| `slurm_phase5_accel_bench.sh` | `cpu_preemptible` | 16 CPUs, 16–256 GB | 1–16 h/job | Phase 5 accelerator benchmarks (PCA, kNN, UMAP, DE, pipeline; submits ~20 parallel jobs) |

### Correctness Validation

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `comprehensive/scripts/slurm_validation_suite.sh` | `cpu_preemptible` | 16 CPUs, 80 GB | 2 h | Correctness validation on pbmc3k (fast gate) |
| `comprehensive/scripts/slurm_validation_suite.sh --scale` | `cpu_preemptible` | 16 CPUs, 200 GB | 6 h | Correctness validation on pbmc3k + tabula_sapiens_100k |

See the "Correctness Validation Suite" section in [docs/testing.md](../docs/testing.md) for details and threshold rationale.

### GPU Benchmarks

| Script | Partition | Resources | Time | Purpose |
|--------|-----------|-----------|------|---------|
| `slurm_gpu_bench.sh` | `preemptible` | 1 GPU, 8 CPUs, 32 GB | 30 min | GPU decode microbenchmarks |
| `slurm_gpu_knn_bench.sh` | `preemptible` | 1 GPU, 16 CPUs, 64 GB | 1 h | GPU kNN (CAGRA) validation + benchmark |
| `slurm_gpu_pca_opt_bench.sh` | `preemptible` | 1 GPU, 16 CPUs, 128 GB | 2 h | GPU PCA optimization benchmark |
| `slurm_gpu_analysis_bench.sh` | `preemptible` | 1 GPU, 16 CPUs, 128 GB | 4 h | Full GPU suite: PCA + kNN + UMAP + preprocessing + pipeline |

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

## SLURM Partition Policy

Benchmarks use the following partitions:

| Workload | Primary Partition | Fallback Partition |
|----------|------------------|--------------------|
| Standard CPU jobs | `cpu_preemptible` | `cpu_batch` |
| High-memory CPU jobs (≥500 GB) | `cpu_high_mem` | — |
| GPU jobs | `preemptible` | `gpu_batch` |

Preemptible partitions offer faster scheduling and lower cost. If a job is preempted, resubmit it — benchmark scripts are idempotent. Use the fallback (`cpu_batch` / `gpu_batch`) partitions only when preemptible queues are too congested or when you need guaranteed completion (e.g., multi-day runs).

To override the partition for any script that takes a `--partition` flag:

```bash
# Example: use cpu_batch instead of cpu_preemptible
bash benchmarks/comprehensive/scripts/run_slurm.sh --partition cpu_batch
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

The comprehensive harness is the committed entrypoint. Legacy wrappers
under `benchmarks/scripts/slurm_*.sh` are deprecated (see
`benchmarks/scripts/README.md` for the migration table).

```
1. Prepare datasets (run first — benchmarks depend on these):
   sbatch benchmarks/scripts/slurm_prep_datasets.sh
   sbatch benchmarks/scripts/slurm_build_census_5m.sh    # (optional, high-memory)
   sbatch benchmarks/scripts/slurm_build_census_10m.sh   # (optional, high-memory)

2. Run benchmarks via the unified comprehensive launcher
   (one submitit job per benchmark × format × dataset triple):
   python benchmarks/comprehensive/scripts/run_parallel.py \
       --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k
   python benchmarks/comprehensive/scripts/run_parallel.py \
       --datasets census_500k census_1m census_5m --tier large

3. Cloud benchmarks (GCP — requires GOOGLE_APPLICATION_CREDENTIALS;
   see check_gcp_auth.py preflight):
   python benchmarks/comprehensive/scripts/check_gcp_auth.py
   python benchmarks/comprehensive/scripts/run_parallel.py \
       --benchmarks cloud_push cloud_pull cloud_read cloud_metadata \
                    cloud_filtered cloud_reader_vs_pull cost_model \
       --datasets pbmc3k tabula_sapiens_100k

4. Regression gate (on-demand — per PR, pre-release, on-suspicion):
   bash benchmarks/comprehensive/scripts/gate_candidate.sh
```

> [!IMPORTANT]
> Always run dataset preparation jobs first. The comprehensive harness
> expects datasets at `SCX_DATA_DIR` (configured via `.env`; defaults to
> `$SCX_WORK_DIR/benchmarks/datasets`). Cloud benchmarks additionally
> require one-time fixture staging via
> `benchmarks/comprehensive/scripts/setup_cloud_test_data.sh`.

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

**Comprehensive benchmark suite** (`comprehensive/`) uses isolated conda environments (`scx-bench`, `scx-bench-gpu`, `scx-bench-r`, `scx-bench-slaf`) for reproducibility. See [Benchmark Environments](#benchmark-environments) above for setup.

**Legacy scripts** (`scripts/`):

**CPU benchmarks** use the project's uv virtualenv (`.venv/`). The scripts reference it automatically.

**GPU benchmarks** use different environments depending on the script:

| Script | Partition | Environment | Reason |
|--------|-----------|-------------|--------|
| `slurm_gpu_analysis_bench.sh` | `preemptible` | conda `scx-gpu` env | Full CUDA + RAPIDS stack (cuVS, cuGraph) |
| `slurm_gpu_pca_opt_bench.sh` | `preemptible` | conda `scx-gpu` env | Same as above |
| `slurm_gpu_knn_bench.sh` | `preemptible` | uv `.venv/` + `LD_LIBRARY_PATH` | pip-installed RAPIDS, manual lib paths |
| `slurm_gpu_bench.sh` | `preemptible` | Rust-only (cargo) | No Python RAPIDS deps |

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

## Regression Gating

Regression gating is **on-demand, not scheduled**. No nightly cron runs —
operators invoke the gate per PR, pre-release, or on suspicion of
regression. No wasted compute when nothing has changed, and every gate
result is tied to a specific commit the operator cares about.

### On-demand workflow (one command)

```bash
bash benchmarks/comprehensive/scripts/gate_candidate.sh
```

That captures a snapshot named `candidate_<git-sha>_<YYYYMMDD>` at the
`small` tier and runs the gate against `results/baselines/LATEST`
(maintained by `promote_baseline.py`). Exit code bubbles up: `0` = pass,
`1` = unjustified regression / floor violation / fingerprint mismatch,
`2` = missing inputs.

Common options:

```bash
# Larger tier (full dataset set)
bash benchmarks/comprehensive/scripts/gate_candidate.sh --tier full

# Reuse an already-captured candidate (skips the capture step)
bash benchmarks/comprehensive/scripts/gate_candidate.sh \
    --skip-capture --name candidate_abc1234_20260420

# Pin a specific historical baseline instead of LATEST
bash benchmarks/comprehensive/scripts/gate_candidate.sh \
    --baseline benchmarks/comprehensive/results/baselines/v0.4.0

# Anything after known flags is passed through to compare_against_baseline.py
bash benchmarks/comprehensive/scripts/gate_candidate.sh \
    --timing-tolerance 0.05 --report-json /tmp/gate.json
```

**Recommended trigger points:**

- Pre-PR: run on your topic branch before opening the PR.
- Pre-merge: re-run on the merge candidate if CPU / memory-sensitive code changed.
- Pre-release: run at the `xl` tier; promote the candidate as the new
  baseline if the gate passes (see "Promoting" below).
- On-suspicion: after a suspicious benchmark result, landed profiler
  change, or upstream dependency bump.

### Direct gate invocation (advanced)

The wrapper is a thin convenience layer over
`compare_against_baseline.py`; invoke it directly when you need finer
control:

```bash
python benchmarks/comprehensive/scripts/compare_against_baseline.py \
    --current benchmarks/comprehensive/results/candidate_$(date +%Y_%m_%d) \
    --gate
```

Under `--gate`, these flags auto-default and can be omitted:
- `--baseline` → `results/baselines/LATEST`
- `--justifications` → `results/justifications/`
- `--thresholds` → `benchmarks/comprehensive/thresholds.yaml`

Override any of them by passing the flag explicitly. Without `--gate` the
script stays in pure-diff mode — no justification or floor logic, no
disappeared-benchmark flagging.

Exit codes: `0` = pass, `1` = unjustified regression / floor violation /
fingerprint mismatch, `2` = baseline or current directory missing or no
canonical baseline promoted yet.

### Promoting a canonical baseline

Snapshots land in `benchmarks/comprehensive/results/<name>/` from
`capture_baseline.py` (or via the `gate_candidate.sh` wrapper). To
promote one as the canonical release baseline:

```bash
python benchmarks/comprehensive/scripts/promote_baseline.py \
    --snapshot benchmarks/comprehensive/results/candidate_2026_04_18_batch_d_t3 \
    --version v0.5.0-phase5
```

This copies only `summary.json` + `environment.json` + `MANIFEST.sha256`
into `results/baselines/v0.5.0-phase5/`, and updates `results/baselines/LATEST`
to point at the new version (relative symlink, pointer-file fallback on
filesystems that reject symlinks). The gate's `--baseline` auto-resolves
to `LATEST`, so no follow-up configuration is needed — next
`gate_candidate.sh` run compares against the newly-promoted baseline.

Raw per-run JSONs stay gitignored; the manifest provides tamper-evidence.
Pass `--no-latest` to promote without touching the `LATEST` pointer
(useful for backfilling historical baselines out-of-order).

### Justification markdown format

Add a new file under `benchmarks/comprehensive/results/justifications/`
whenever a flagged regression has been investigated and deliberately
accepted:

```markdown
---
triples:
  - benchmark: cloud_push
    format: scx_auto
    dataset: pbmc3k
reason: Upstream gcsfs 2025.10.0 HTTP/2 header canonicalization (~4%).
expires: 2026-06-01
---

One or more paragraphs of prose explaining the tradeoff. Shown in the
gate's failure summary so reviewers see the reason inline.
```

`expires` is optional. When present and past `date.today()`, the
justification stops suppressing and the gate fails again — forces
periodic review. Multiple triples per file are fine; one file per PR is
typical.

### Rolling dashboard

`write_report()` emits `BENCHMARK_REPORT.html` alongside the markdown
report, with a "← previous snapshot" link threaded through
`dashboard_history.json`. Publish to a static-hosting target via:

```bash
export DASHBOARD_PUBLISH_TARGET="user@host:/var/www/scx-bench/"   # or gs://bucket/path/
python benchmarks/comprehensive/scripts/publish_dashboard.py
```

With `DASHBOARD_PUBLISH_TARGET` unset, the publish script is a no-op
and exits 0 — safe for unconditional CI invocation.

### Gate self-test

A hermetic pytest suite validates every transition (regression fails,
justification suppresses, expired justification stops suppressing,
disappearing benchmarks flagged, absolute-floor violations fail):

```bash
.venv/bin/pytest benchmarks/comprehensive/tests/test_gate_self_test.py -v
```

Run it locally before opening a PR that touches the gate.

---

## GPU accelerator regression workflow

> **Scope.** The canonical baseline at
> `comprehensive/results/baselines/LATEST` gates **format-level**
> benchmarks — compression, read/write, parallel scaling, memory,
> ml_loader, cloud. It does **not** currently cover the
> `pyscx.accel.*` GPU accelerator surface (PCA / kNN / UMAP / Leiden /
> preprocessing / HVG). Those benchmarks are maintained by the
> standalone `scripts/benchmark_gpu_*.py` family and dump to
> `results/gpu_*.json`. This section documents the parallel workflow
> for GPU-accelerator regression checks until the comprehensive gate
> learns those dimensions.

### Tooling

Two purpose-built helpers drive this flow end-to-end:

- `benchmarks/scripts/gpu_regression_driver.sh` — orchestrator that
  rebuilds pyscx, runs GPU correctness tests, submits SLURM benchmarks
  in parallel, captures the Phase-7.2 / 7.3 sub-metrics, and calls
  the diff tool. **Run this on the cluster head-node;** it handles
  sbatch + wait + diff.
- `benchmarks/scripts/gpu_regression_diff.py` — Python diff tool that
  reads a PRE and a POST directory, classifies every row via a
  configurable tolerance + a hard-floor table, and emits Markdown + JSON
  reports. Safe to run standalone from any host (CPU-only too — that's
  how `--diff-only` mode on the driver works).

A frozen pre-Phases-1-7 snapshot lives at
`benchmarks/results/pre_phases_1_7_baseline_2026_03/` (13 JSONs + 6
Markdown reports + `MANIFEST.md` + `MANIFEST.sha256`). This is the
default `--pre` for the driver.

### Quick start — one-shot run

```bash
# From the cluster head-node:
bash benchmarks/scripts/gpu_regression_driver.sh
# → benchmarks/results/phases_1_7_gpu_regression_report.{md,json}
#   + benchmarks/logs/gpu_regression_driver_<stamp>.log
```

Exit codes: `0` pass, `1` regression or hard-floor flag, `2` infra
error (build failure, missing pre, sbatch failure).

Common driver flags:

| Flag | Effect |
|---|---|
| `--diff-only` | Skip everything except the final diff. Use when the bench files are already up-to-date. Runs on any host. |
| `--skip-tests` | Skip step 8.3 (cargo + pytest) when already run. |
| `--inline` | Run the bench scripts directly on the current node (no sbatch). Prefer sbatch unless iterating on a small fixture. |
| `--pre DIR` | Alternative baseline directory. Default: `benchmarks/results/pre_phases_1_7_baseline_2026_03`. |
| `--dataset NAME` | Dataset for the `benchmark_gpu_pipeline.py --attribute-preprocessing --pca-variants` capture. Default: `census_1m`. |
| `--conda-prefix DIR` | Conda env for Python / maturin / pytest. Default: `/home/nickyoungblut/miniforge3/envs/scx-gpu`. |

### When to run

Run the GPU accelerator workflow whenever a PR touches any of:

- `scx-gpu/src/**` — kernels, cuSPARSE / cuSOLVER / cuBLAS wrappers,
  shard pipeline, linear operator.
- `scx-accel/src/{pca,neighbors,umap,leiden,hvg,harmony}.rs` — Rust-side
  accelerator entry points (CPU paths too — they gate the CPU baseline).
- `pyscx/src/accel/**` — Python dispatch, device resolution, preprocessing
  materialization, fusion markers.
- `pyscx/Cargo.toml` / `scx-gpu/Cargo.toml` — dep bumps, feature
  toggles (especially `cudarc` minor-version bumps).

CPU-only PRs that don't touch these paths do not need this workflow;
the comprehensive `gate_candidate.sh` is sufficient.

### Step 1 — Capture the pre-change snapshot

Before your post-change runs overwrite `results/gpu_*.json`, copy the
current state into a dated directory so the diff is unambiguous:

```bash
PRE=benchmarks/results/pre_$(git rev-parse --short HEAD)_$(date +%Y%m%d)
mkdir -p "$PRE"
cp benchmarks/results/gpu_*.json benchmarks/results/gpu_*.md "$PRE/" 2>/dev/null || true
# When no prior run exists at all, capture an empty-but-timestamped dir —
# the diff script treats missing files as "new benchmark" rather than "regressed".
```

**If the previous GPU run pre-dates your branch point.** Check the file
modification times (`stat -c '%y %n' benchmarks/results/gpu_*.json`).
If the most recent GPU run is older than the branch point of the work
under review, run the pre-change baseline on the cluster *first* by
checking out the parent commit:

```bash
git stash --include-untracked
git checkout <parent-sha>
# rebuild pyscx with GPU feature (see Step 2), then run the suite.
cp benchmarks/results/gpu_*.json "$PRE/"
git checkout -
git stash pop
```

### Step 2 — Rebuild pyscx with `--features gpu` on the cluster

The dev `.venv/` is typically built without the `gpu` feature. Every
benchmark script imports the compiled `pyscx` wheel, so rebuild on the
cluster host before benchmarking:

```bash
/home/nickyoungblut/miniforge3/envs/scx-gpu/bin/maturin develop \
    --manifest-path pyscx/Cargo.toml --release --features gpu
```

`--release` matters — debug builds produce meaningless bench numbers.

### Step 3 — Run the GPU correctness tests first

Fast signal on any CUDA runtime / driver / kernel-launch issue. If any
of these fail, fix the root cause before running the (slower, more
expensive) benchmarks:

```bash
cargo test --workspace --features gpu -- --nocapture 2>&1 \
    | tee benchmarks/logs/gpu_tests_$(date +%Y%m%d).log

/home/nickyoungblut/miniforge3/envs/scx-gpu/bin/pytest \
    pyscx/tests/test_accel_pca_gpu.py \
    pyscx/tests/test_accel_pipeline_gpu.py \
    -v 2>&1 | tee benchmarks/logs/pytest_gpu_$(date +%Y%m%d).log
```

Expect every `require_gpu!()`-gated Rust test to run (not skip), and
every `@gpu_only` Python test to pass.

### Step 4 — Submit GPU benchmarks in parallel

One SLURM job per GPU benchmark (per repo-level guidance: parallel
submission is the default). Each SLURM wrapper rebuilds pyscx
internally, so parallel submission is safe:

```bash
sbatch benchmarks/scripts/slurm_gpu_analysis_bench.sh   # PCA + kNN + UMAP + preprocessing + pipeline
sbatch benchmarks/scripts/slurm_gpu_pca_opt_bench.sh    # PCA sweep
sbatch benchmarks/scripts/slurm_gpu_knn_bench.sh        # kNN only
sbatch benchmarks/scripts/slurm_gpu_bench.sh            # decode microbenchmarks
```

Monitor with `squeue -u $USER` or
`.venv/bin/python benchmarks/comprehensive/scripts/watch.py`.

Capture the per-PCA-variant rows and the CPU-vs-GPU device-dispatch
preprocessing rows (Phase 7.2 / 7.3 additions):

```bash
/home/nickyoungblut/miniforge3/envs/scx-gpu/bin/python \
    benchmarks/scripts/benchmark_gpu_pipeline.py \
    --dataset census_1m --attribute-preprocessing --pca-variants

/home/nickyoungblut/miniforge3/envs/scx-gpu/bin/python \
    benchmarks/scripts/benchmark_gpu_preprocess.py --mode device
```

These two invocations produce `gpu_pipeline_timing.json` with
`cpu_preprocessing_*` / `gpu_preprocessing_*` / `gpu_pca_variants`
sub-keys, and `gpu_preprocess_device.json` with per-op cpu/gpu
median_s + speedup.

### Step 5 — Diff post vs pre

The driver calls `gpu_regression_diff.py` as its final step. To run the
diff standalone (e.g., after tweaking thresholds, or for post-hoc
analysis of an older result set):

```bash
python benchmarks/scripts/gpu_regression_diff.py \
    --pre  benchmarks/results/pre_phases_1_7_baseline_2026_03 \
    --post benchmarks/results \
    --report    benchmarks/results/phases_1_7_gpu_regression_report.md \
    --json-out  benchmarks/results/phases_1_7_gpu_regression_report.json \
    --timing-tolerance 0.10        # ±10% default; use --strict for ±5%
```

The tool handles every `gpu_*.json` schema produced by the standalone
benchmark scripts (dict-shaped `gpu_pipeline_timing.json`, list-shaped
`gpu_{pca,knn,umap,preprocess}_timing.json`, and the Phase-7-only
`gpu_preprocess_device.json`). Phase-7.2 additions like
`gpu_pca_variants` are classified as `new` (post-only) rather than
regressions.

The Markdown report is structured as:

1. Summary header: pre/post paths, tolerance, row counts.
2. **Hard-floor flags** (from §8.6 threshold table) — these fire
   regardless of the tolerance and are the first thing to read.
3. Regressions, improvements, new, disappeared — in that order.
4. Full row table for exhaustive audits.

Raw side-by-side comparison of the Markdown reports is also useful:

```bash
diff -u benchmarks/results/pre_phases_1_7_baseline_2026_03/gpu_*.md \
        benchmarks/results/gpu_*.md | less
```

### Step 6 — Regression thresholds

Tolerance-based classification (configurable via `--timing-tolerance`,
default ±10 %):

- `regression` — a higher-is-better metric (speedup, throughput)
  dropped more than the tolerance below pre, or a lower-is-better metric
  (wall time) rose more than the tolerance above pre.
- `improvement` — symmetric in the other direction.
- `new` — present only in post (e.g. Phase-7.2 `gpu_pca_variants`
  dispatch rows).
- `disappeared` — present only in pre (possibly a renamed or removed
  benchmark).

Hard-floor flags (fire regardless of the tolerance):

| Signal | Hard-floor check |
|---|---|
| Covariance PCA on 2K HVGs | `pipeline:pca` post speedup < 1.5× |
| End-to-end pipeline | post speedup < pre by > 0.1× |
| CholeskyQR2 vs Householder | Cholesky median > Householder × 1.1 |
| Unmodified path regression | kNN / UMAP / Leiden regressed past tolerance |

Expected regressions (e.g. a refactor trades throughput for memory or
correctness) should be documented the same way the comprehensive gate
handles them — a markdown file under
`comprehensive/results/justifications/` with frontmatter triples and
an expiry date. The GPU diff script doesn't auto-suppress justified
regressions today; manual review remains authoritative. Thresholds
live in `gpu_regression_diff.py` → `apply_hard_floors()` and should be
tuned once the first few cluster runs establish real variance bands.

### Step 7 — Update docs and promote (optional)

On a clean pass:

1. Fill in any `_pending bench_` placeholders in `docs/performance.md`
   with the post-change numbers. Update the "GPU Analysis Pipeline"
   table and the "Changes vs previous version" note.
2. Freeze the post-change results as the *next* bisect reference. The
   standard naming convention is
   `benchmarks/results/pre_<spec-name>_<YYYY_MM>/`:

   ```bash
   NEXT=benchmarks/results/pre_next_gpu_phase_$(date +%Y_%m)
   mkdir -p "$NEXT"
   cp benchmarks/results/gpu_*.json benchmarks/results/gpu_*.md "$NEXT/"
   (cd "$NEXT" && sha256sum *.json > MANIFEST.sha256)
   # Add a MANIFEST.md following the pattern of
   # benchmarks/results/pre_phases_1_7_baseline_2026_03/MANIFEST.md
   ```
3. When the comprehensive gate gains GPU-accelerator coverage, promote
   the post-change snapshot via `promote_baseline.py --version
   v0.X.Y-gpu` — same flow as the format baseline.

### Long-term direction: Phase 9 comprehensive-framework integration

The workflow above is a **stopgap**. The strategic home for every
benchmark — format and accelerator — is the comprehensive framework at
`benchmarks/comprehensive/`, which already handles parallel SLURM
submission (`run_parallel.py` / submitit), per-cell result schema
(`summary.json`), canonical baselines
(`results/baselines/LATEST → v0.X.Y`), and PR gating
(`gate_candidate.sh`). Phase 9 of `GPU-ACC-SPEED-UP.md` migrates the
GPU accelerator benchmarks into this framework. Current status:

- `accel_pca.py` — **landed** as the reference module with 5 variants
  (scanpy CPU, pyscx CPU, pyscx GPU covariance, pyscx GPU randomized
  Householder, pyscx GPU randomized Cholesky). Discoverable via
  `config.accel_formats()` and registered in `ALL_BENCHMARKS`.
- `accel_{knn,umap,leiden,preprocess,hvg}.py` — **pending**; follow
  the exact same pattern as `accel_pca.py`.
- `thresholds.yaml` / `v0.6.0-gpu-phase1-7` baseline — **pending**
  on post-Phase-7 cluster run.

Once Phase 9 completes, this §GPU-accelerator-regression-workflow
section collapses into a pointer at the standard `gate_candidate.sh`
flow — no separate scripts, no separate baselines.

### Known limitations (during the Phase-9 transition)

- **No automated gating for GPU benchmarks.** The
  `compare_against_baseline.py` gate operates on the comprehensive
  `summary.json`; the standalone GPU JSONs (and the one `accel_pca`
  module that's landed so far) don't feed into it. The diff above is
  manual. Tracking this as Phase 9 Task 9.5 / 9.6 in
  `GPU-ACC-SPEED-UP.md`.
- **Reproducibility.** The GPU benchmarks don't record the exact cudarc
  version, RAPIDS commit, or driver version in the JSON. When diffing,
  check `environment.json` next to the raw JSON for the comprehensive
  runs; for the standalone scripts, rely on the SLURM log header
  (`nvidia-smi`, `nvcc --version`, `conda list`).
- **Census 1M wall-time variance.** Preemptible GPU jobs can restart
  mid-run. Every preempted job re-runs from scratch; budget accordingly.

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
| `benchmark_cloud.py` | **DEPRECATED** — forwards to `comprehensive/benchmarks/cloud_*.py` (see Phase C below) |
| `benchmark_compressed_h5ad.py` | Compressed h5ad baseline (gzip, lzf) |
| `benchmark_gpu_decode.py` | GPU decode microbenchmarks (cuSPARSE, bitstream) |
| `benchmark_gpu_pca.py` | GPU PCA validation + timing |
| `benchmark_gpu_knn.py` | GPU kNN (CAGRA) validation + timing |
| `benchmark_gpu_umap.py` | GPU UMAP validation + timing |
| `benchmark_gpu_preprocess.py` | GPU fused preprocessing (normalize+log1p) |
| `benchmark_gpu_pipeline.py` | End-to-end GPU pipeline + Go/No-Go gate |
| `benchmark_gpu_scvi.py` | GPU scVI training benchmark |
| `benchmark_lazy_preprocess.py` | Phase 4d/5c lazy transforms, memory, column projection, fused opt, E2E OOC pipeline |
| `benchmark_lazy_preprocess_rss_worker.py` | RSS time-series subprocess worker for lazy preprocessing benchmarks |
| `benchmark_accelerators.py` | Phase 4b CPU accelerators: PCA, kNN, UMAP, DE vs scanpy |
| `benchmark_accel_pipeline.py` | Full pipeline (3 variants: SCX OOC, SCX preprocess, scanpy) |
| `benchmark_accel_preprocessing.py` | pyscx.preprocess() vs scanpy normalize+log1p |
| `benchmark_bpcells.R` | BPCells comparison (R) |
| `comprehensive/benchmarks/cell_eval_parity_perf.py` | cell-eval / arc-bench parity perf (pseudobulk, perturbation metrics, energy distance, discrimination score, knockdown efficiency, clustering agreement) at synthetic 100K–1M scale |
