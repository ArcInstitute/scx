# SCX Benchmarks

This directory contains the benchmarking infrastructure for SCX — scripts, SLURM job definitions, results, and logs for evaluating compression, read/write performance, ML data loader throughput, GPU accelerators, and lazy preprocessing.

---

## Directory Layout

```
benchmarks/
├── README.md              ← You are here
├── scripts/               # Dataset prep + ML loader + standalone GPU/Harmony benches
│   │   # Dataset preparation
│   ├── download_datasets.sh             # Download & convert benchmark datasets
│   ├── download_*.py                    # Dataset-specific downloaders
│   ├── build_census_*.py                # Large dataset builders (500K, 5M, 10M cells)
│   ├── slurm_prep_datasets.sh           # SLURM wrapper for D1–D6 prep
│   ├── slurm_build_census_*.sh          # High-mem SLURM wrappers for D7 / D8
│   ├── verify_datasets.py               # Validate datasets & record metadata
│   ├── generate_compressed_h5ad.py      # Pre-generate gzip/lzf h5ad variants
│   ├── prep_lognorm_datasets.py         # Log-normalized dataset prep
│   ├── augment_obs_n_counts.py          # Backfill obs.n_counts for legacy fixtures
│   ├── setup_cloud_test_data.sh         # Initial GCS test-data upload (one-time bootstrap)
│   │   # ML training loader (only path covering pyscx.TrainingDataset)
│   ├── submit_benchmarks.py             # submitit launcher for the loader benchmark
│   ├── benchmark_loader.py              # Loader throughput vs SOTA baselines
│   ├── benchmark_bpcells.R              # Driver invoked by benchmark_loader.py
│   │   # Active dev surfaces — GPU
│   ├── slurm_gpu_*_bench.sh             # Standalone GPU benchmark drivers
│   ├── benchmark_gpu_{decode,pca,knn,umap,preprocess,pipeline,scvi}.py  # Backing implementations
│   │   # Active dev surfaces — Harmony / LISI
│   ├── slurm_harmony_bench.sh           # Harmony2 driver
│   ├── benchmark_{harmony,lisi}.py      # Harmony2 / LISI performance
│   ├── harmony_bench_worker.py          # Per-config worker
│   ├── report_harmony.py                # Markdown / plot generator
│   ├── build_harmony_validation_fixtures.py  # Test-fixture builder (used by pyscx tests)
│   ├── generate_harmony_reference.R     # R-harmony reference
│   │   # Helpers
│   ├── benchmark_cli.py                 # CLI command latency
│   ├── benchmark_python_bindings.py     # PyO3 binding overhead
│   ├── build_release.py                 # Helper: ensure pyscx release build
│   └── run_with_rayon_limit.sh          # Wrapper that pins RAYON_NUM_THREADS
│   #
│   # Phase wrappers (slurm_phase*_*.sh), the original benchmark_*.py one-off
│   # entrypoints, gpu_regression_*, and benchmark_cloud.py have been deleted —
│   # use comprehensive/scripts/run_parallel.py instead. See scripts/README.md.
├── r_scripts/             # Top-level R benchmark scripts
│   └── benchmark_bpcells.R              # BPCells benchmark driver (called via Rscript)
├── comprehensive/         # Comprehensive benchmark suite
│   ├── config.py                        # Dataset paths, format configs, constants
│   ├── sysinfo.py                       # System info collector (CPU, RAM, OS, disk)
│   ├── results.py                       # BenchmarkResult schema + JSON writer
│   ├── convert.py                       # Shared conversion cache (convert once, reuse)
│   ├── cloud_fixtures.py                # GCP fixture self-healing + credential resolution
│   ├── queries.py                       # Shared query fixtures (HVG lists, filters, row/col slices)
│   ├── profile.py                       # Per-run peak-RSS / timing profiler helpers
│   ├── provenance.py                    # Git SHA / env capture for summary.json
│   ├── thresholds.yaml                  # Hard-floor + tolerance config for the regression gate
│   ├── requirements.txt                 # pip fallback pins (conda envs are authoritative)
│   ├── envs/                            # Conda environment definitions
│   │   ├── scx-bench.yml                # CPU benchmark environment
│   │   ├── scx-bench-gpu.yml            # GPU benchmark environment (CUDA + RAPIDS)
│   │   ├── scx-bench-r.yml              # R / BPCells benchmark environment
│   │   ├── scx-bench-slaf.yml           # SLAF (slafdb) benchmark environment
│   │   └── scx-bench-eval.yml           # Perturbation / cell-eval parity environment
│   ├── runners/                         # Per-format benchmark runners
│   │   ├── base.py                      # Abstract FormatRunner interface
│   │   ├── h5ad_runner.py               # h5ad (uncompressed, gzip, lzf)
│   │   ├── zarr_runner.py               # Zarr (zstd, blosc-lz4)
│   │   ├── tiledb_runner.py             # TileDB-SOMA
│   │   ├── scx_runner.py                # SCX (auto, none, scx1, zstd, pcodec, lz4)
│   │   ├── bpcells_runner.py            # BPCells (R subprocess)
│   │   ├── parquet_runner.py            # Parquet (pyarrow)
│   │   ├── slaf_runner.py               # SLAF (slafdb — Lance + DuckDB)
│   │   └── accel_runner.py              # pyscx.accel.* accelerator runner (CPU + GPU dispatch)
│   ├── benchmarks/                      # Benchmark modules (one per dimension)
│   │   ├── compression.py, write.py, read_full.py, read_selective.py
│   │   ├── parallel_scaling.py, parallel_write_scaling.py, memory.py, ml_loader.py
│   │   ├── read_streaming_vs_inmemory.py            # Backed iteration vs eager materialise (Phase 6b)
│   │   ├── fragment_ops.py, correctness.py
│   │   ├── accel_{pca,knn,umap,leiden,preprocess,hvg}.py   # Accelerator surface
│   │   ├── cloud_{push,pull,read,metadata,filtered,large_atlas,reader_vs_pull}.py, cost_model.py
│   │   └── cell_eval_parity_perf.py, _pert_synth.py        # perturbation / cell-eval parity
│   ├── reporting/                       # Report generation (markdown, plots, tables, dashboard)
│   │   └── report_cli.py, markdown.py, plots.py, tables.py, dashboard.py, lint.py, landing.py, style.py
│   ├── scripts/                         # Orchestrators and SLURM launchers
│   │   ├── run_all.py                   # Serial benchmark orchestrator
│   │   ├── run_parallel.py              # Parallel SLURM launcher via submitit
│   │   ├── run_slurm.sh                 # SLURM submission script
│   │   ├── install_dependencies.sh      # Create conda environments
│   │   ├── validate_*.py                # Correctness validation scripts
│   │   ├── validation_helpers.py        # Shared helpers for validate_*.py
│   │   ├── smoke_test_runners.py        # Fast per-runner smoke test
│   │   ├── slurm_*.sh                   # SLURM job scripts (validation suite, capture, parity, ...)
│   │   │
│   │   │  # Regression gating (on-demand)
│   │   ├── gate_candidate.py            # One-shot: capture + gate against LATEST baseline
│   │   ├── capture_baseline.py          # Freeze one snapshot (raw/ + summary.json + manifest)
│   │   ├── promote_baseline.py          # Promote a snapshot to results/baselines/<version>/
│   │   ├── compare_against_baseline.py  # Relative + absolute-floor + justification + flakiness gate
│   │   ├── _justifications.py           # Justification frontmatter loader used by the gate
│   │   ├── _flakiness.py                # Flakiness-ledger frontmatter loader (per-row tolerance overrides)
│   │   ├── fingerprint_accelerators.py  # Record accelerator library versions into summary.json
│   │   ├── ooc_rss_table.py             # OOC pipeline peak-RSS table generator
│   │   ├── publish_dashboard.py         # Rsync HTML snapshot to DASHBOARD_PUBLISH_TARGET
│   │   │
│   │   │  # Cloud infrastructure (GCP)
│   │   ├── check_gcp_auth.py            # Credentials + bucket round-trip preflight
│   │   ├── setup_cloud_test_data.sh     # Idempotent fixture staging via BLAKE3 sidecars
│   │   ├── cloud_fixtures_doctor.py     # Dry-run probe: local vs cloud fixture drift
│   │   ├── submit_gcp_matrix.py         # Instance-type × benchmark launcher (--yes-spend)
│   │   ├── gcs_lifecycle.json           # Committed GCS object-lifecycle rules
│   │   ├── repro_slaf_cloud_read.py     # Standalone SLAF GCS-read repro
│   │   ├── repro_zarr_cloud_read.py     # Standalone Zarr GCS-read repro
│   │   │
│   │   │  # Observability + migrations
│   │   ├── watch.py                     # Rich TUI tailing run_manifest.json + submitit logs
│   │   └── migrate_results.py           # Back-stamp schema_version on legacy raw JSONs
│   ├── tests/                           # Gate self-test + cloud-mechanism pytest suite
│   ├── r_scripts/                       # BPCells R benchmark scripts (used by bpcells_runner)
│   ├── logs/                            # submitit / SLURM job logs
│   └── results/                         # Raw JSON results, candidate snapshots, and baselines
│       ├── raw/                         # One JSON per benchmark×format×dataset
│       ├── reports/                     # Generated markdown + plots
│       ├── baselines/                   # Promoted baselines + LATEST pointer (v0.5.0-phase5, v0.6.0-gpu-phase1-7, ...)
│       ├── justifications/              # Markdown justifications with frontmatter triples
│       ├── flakiness/                   # Markdown flakiness ledger (per-row relaxed tolerances)
│       └── candidate_*/, tier*_*/, baseline_*/  # Dated capture snapshots
├── results/               # Per-script benchmark output (JSON + Markdown reports)
│   ├── pre_phases_1_7_baseline_2026_03/ # Frozen pre-GPU-accel baseline (see §GPU workflow)
│   └── harmony/                         # Harmony2 validation fixtures + run outputs
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
| **Streaming vs in-memory** (3.7b) | `read_streaming_vs_inmemory.py` | Backed row-chunk iteration vs eager-materialise-then-iterate. Wall time and peak RSS for both modes; gated on `"backed_mode"` capability — SCX-only today |
| **Multimodal streaming vs in-memory** (3.7c) | `multimodal_read_streaming_vs_inmemory.py` | Phase 6b — `to_mudata(backed=True)` per-modality chunked iteration vs eager `to_mudata()`. Gated on `dataset.multimodal == True` AND multimodal-SCX format keys |
| **Streaming export** (3.7d) | `export_streaming.py` | Phase 8 — paired `pyscx.to_h5ad` / `pyscx.to_h5mu` with `stream=True` vs `stream=False`. Wall time and peak RSS for both paths; gates on the streaming row's `streaming_peak_rss_mb` floor at `census_1m`. Multimodal datasets auto-dispatch to `to_h5mu`. SCX-only |
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

> [!CAUTION]
> **The comprehensive benchmark orchestrators (`run_parallel.py`,
> `capture_baseline.py`, `gate_candidate.py`) MUST be launched from
> within the `scx-bench` conda env, not the dev `.venv/`.** This
> isn't a soft preference — it's load-bearing. Launched from `.venv/`,
> every SLURM job's PATH falls back to `.venv/bin` — which does
> **not** include `mudata` or other format-runner deps that live in
> the conda env stack. The cascade failure mode is recurring +
> expensive:
>
> 1. Convert phase: many `convert_from_h5ad` jobs ImportError at runtime
>    because the dependency isn't on `.venv/bin/python`.
> 2. Benchmark phase: hundreds of dependent bench jobs queue as
>    `DependencyNeverSatisfied`, pinning the QOS-cap throttle.
> 3. Post-submit: orchestrator's `job.result()` loop spends
>    ~15 s per cancelled job, multiplying into hour-long drains.
>
> **Always:**
>
> ```bash
> conda activate scx-bench
> python benchmarks/comprehensive/scripts/gate_candidate.py --tier small
> ```
>
> `run_parallel.py` then **automatically routes each SLURM worker to
> the right env per its format key** (see `_env_for_format`):
> `_gpu*` → `scx-bench-gpu`, `slaf*` → `scx-bench-slaf`,
> `bpcells*` → `scx-bench-r`, everything else → `scx-bench`. The
> operator does not need to launch from a different env or pass
> `--formats` to scope away from "missing deps in current env" — the
> per-job routing handles it. Workers use `srun python` (via
> `slurm_python="python"` on the `AutoExecutor`) so the `conda
> activate <env>` in `slurm_setup` actually determines which
> interpreter the worker runs under.
>
> **Build pyscx with `--features gpu` always.** The editable .so
> from `maturin develop` is written to the source tree (one .so per
> repo, shared across all conda envs); a build without
> `--features gpu` from any env silently disables GPU support
> everywhere — `pyscx.accel.gpu_info()` then returns `None`, every
> bench module's `_HAS_PYSCX_GPU` evaluates to `False` at import
> time, and GPU bench variants are filtered out of submission
> entirely (jobs "complete successfully" producing no JSON output).
> The canonical sequence:
>
> ```bash
> cd pyscx
> conda activate scx-bench-gpu   # any env; only one shared .so
> maturin develop --release --features gpu
> ```
>
> Then verify: `for env in scx-bench scx-bench-gpu scx-bench-slaf; do
> conda activate $env && python -c "import pyscx;
> print(bool(pyscx.accel.gpu_info()))" && conda deactivate; done`
> — all three should print `True` on a host with CUDA visible.

**Comprehensive benchmark suite** (`comprehensive/`):

| Script | Environment | Activation |
|--------|-------------|------------|
| `comprehensive/scripts/gate_candidate.py` | **`scx-bench`** | `conda activate scx-bench` |
| `comprehensive/scripts/capture_baseline.py` | **`scx-bench`** | `conda activate scx-bench` |
| `comprehensive/scripts/run_parallel.py` | **`scx-bench`** | `conda activate scx-bench` |
| `comprehensive/scripts/run_all.py` | `scx-bench` | `conda activate scx-bench` |
| `comprehensive/scripts/validate_*.py` | `scx-bench` | `conda activate scx-bench` |
| GPU benchmark workers (auto-routed) | `scx-bench-gpu` | per-job, via `_env_for_format` |
| BPCells benchmark workers (auto-routed) | `scx-bench-r` | per-job, via `_env_for_format` |
| SLAF benchmark workers (auto-routed) | `scx-bench-slaf` | per-job, via `_env_for_format` |

**Legacy scripts** (`scripts/`) still reference the dev `.venv/` (CPU) and the `scx-gpu` conda env (GPU), and are preserved as-is.

SLURM scripts in `comprehensive/scripts/` auto-detect the correct environment. Override with `--conda-env`:

```bash
bash benchmarks/comprehensive/scripts/run_slurm.sh --conda-env scx-bench-gpu
```

---

## Parallel Benchmark Execution (run_parallel.py)

> [!IMPORTANT]
> **Orchestrator vs. workers.** `run_parallel.py` and `gate_candidate.py`
> are *job submitters* — they run on the host you invoke them from
> (login node, `sh_dev`, or any CPU-only machine that can reach SLURM)
> and submit SLURM jobs that execute on the actual compute nodes.
> **You do not need a GPU on the host running the orchestrator** —
> GPU-bearing benchmarks request GPUs from SLURM (`partition=preemptible
> slurm_gres=gpu:1`) and run on H100 worker nodes. This is the standard
> Chimera workflow: orchestrate from `sh_dev` (or login), benchmarks
> execute under `sbatch` on GPU nodes. The orchestrator script just
> needs to stay alive long enough to submit; it can exit immediately
> after submission (jobs continue independently) or follow the
> submitted jobs through completion (default).

The serial orchestrator (`run_all.py`) processes benchmarks sequentially within a single SLURM job. For faster execution, `run_parallel.py` uses two-phase parallel execution via `submitit`:

**Phase A — Conversions (individual jobs).** Each (dataset, format) pair is converted exactly once and written to a persistent path. Conversions run as independent parallel SLURM jobs submitted via `executor.submit(...)`. Existing files are skipped automatically (`--overwrite` to force). The conversion population is small (typically 20–40 jobs) and resource-heterogeneous, so individual submissions are appropriate.

**Phase B — Benchmarks as resource-cohort SLURM Job Arrays.** Each (benchmark, dataset, format) triple is grouped into a *resource cohort* keyed by `(dataset, format_key, partition, gres, needs_conversion)`, and every cohort is submitted as a single SLURM Job Array via `executor.map_array(...)`. Because SLURM requires every task within an array to share identical resource parameters (partition, gres, mem, time, cpu), the 5-tuple key guarantees uniformity while still letting the orchestrator separate:

- **GPU vs. CPU** — `ml_loader` on `scx_auto` routes to GPU; `read_full` on the same `(dataset, format)` stays on CPU and lands in a different array.
- **High-memory vs. standard** — `partition_for_memory` promotes cohorts past `MEM_HIGH_MEM_THRESHOLD_GB` (200 GB) to `cpu_high_mem`. When an estimate exceeds `MEM_CEILING_GB` (1000 GB) the request is clamped and a `logger.warning("...clamped to MEM_CEILING_GB...")` line lands in the orchestrator log so the truncation is greppable.
- **Dependent vs. independent** — `_NO_CONVERSION` benchmarks (`write`, `parallel_write_scaling`, `cell_eval_parity_perf`) sit in cohorts with `needs_conversion=False` and skip the `afterok` edge entirely.

Per-bench compatibility — `_bench_format_compatible` applies three layered filters at cohort-build time so non-applicable (bench, format) combinations never get submitted:

1. **Accel/CSC pairing** — `accel_pca` only groups with `accel_pca__*` formats; `bench_csc_dispatch` only groups with `bench_csc__*` formats; non-accel benchmarks skip both.
2. **`SUPPORTED_FORMATS`** — benches that only run on a fixed format set (e.g. `cloud_push`, `correctness`, `roundtrip`, `ml_loader`) expose a module-level `SUPPORTED_FORMATS: frozenset[str]` and the cohort builder enforces it. Adding a new such bench is a single-line declaration next to the runtime guard.
3. **`REQUIRED_CAPABILITIES`** — benches that need a specific runner capability (e.g. `cloud_read` needs `"cloud_read"`, `read_streaming_vs_inmemory` needs `"backed_mode"`) expose `REQUIRED_CAPABILITIES: frozenset[str]`. The cohort builder reads `runner.capabilities` via a cached `make_runner(fmt)` (cheap, side-effect-free) and filters incompatible cells.

Cohorts that need conversion attach an `--dependency=afterok:<convert_jobid>` so their array waits for Phase A to finish for the matching `(dataset, format)`. Cohorts that don't need conversion start immediately.

Throttling is layered:
1. **Native cluster-side**: `slurm_array_parallelism` caps how many tasks within an array run concurrently (`sbatch --array=0-N%M`).
2. **QOS-aware client-side**: SLURM counts each array task individually against Chimera's `QOSMaxSubmitJobPerUserLimit`. The documented cap is ~500 on `cpu_preemptible`, but the May 2026 tier-xl run observed rejections at queue depth 80–110, so the default is conservative. Before each `map_array` call, `run_parallel.py` blocks on `_wait_under_pending_cap(args.max_pending_jobs)` (default 75). On a transient `QOSMaxSubmitJobPerUserLimit` rejection (race between squeue and SLURM's internal counter), the cohort submission retries up to 3× with a 60s drain.

Result: ~30–40 conversion submits + ~40–80 cohort array submits replaces ~500 individual `sbatch` calls — an >80% reduction in scheduler load with no per-task client-side polling between successful submissions. The three-layer compatibility filter additionally eliminates ~800 phantom `missing_result` entries per full-tier run (cells where a bench would otherwise have been submitted, returned `None` for an incompatible format, and produced no JSON).

```
Serial (run_all.py):        420 tasks × ~3 min = ~21 hours wall time
Parallel (run_parallel.py): Phase A conversions + Phase B cohort arrays
                            Wall time ≈ max(single slowest job)
```

`watch.py` consumes `comprehensive/logs/submitit/run_manifest.json` to render a live status table. Submitit names per-task files `<SLURM_jobid>_<task_idx>_*` for both individual jobs (`2306028_0_result.pkl`) and array tasks (`2374101_0_0_result.pkl` for array `2374101` task `0`) — the array-task ID is part of the SLURM job ID, so `watch.py` always appends the `_0` task-index suffix uniformly.

### Usage

`run_parallel.py` takes an explicit `--datasets` list — tier selection (`--tier small|full|xl`) lives on the higher-level entry points `gate_candidate.py` and `capture_baseline.py`, which translate the tier into the appropriate dataset list and forward it to `run_parallel.py`.

```bash
# Recommended: go through gate_candidate.py for the canonical tier flow
python benchmarks/comprehensive/scripts/gate_candidate.py --tier small  # ~1h
python benchmarks/comprehensive/scripts/gate_candidate.py --tier full   # ~4-6h
python benchmarks/comprehensive/scripts/gate_candidate.py --tier xl     # adds census_5m

# Direct run_parallel.py invocation — small tier
python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k \
        cite_seq_pbmc multiome_pbmc

# Specific benchmarks and formats only
python benchmarks/comprehensive/scripts/run_parallel.py \
    --benchmarks read_full read_selective \
    --formats scx_auto zarr_zstd h5ad_gzip \
    --datasets census_1m

# Dry run — print cohort plan + total task count without submitting
python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k --benchmarks read_full --dry-run

# Skip conversion phase (reuse existing pre-converted files)
python benchmarks/comprehensive/scripts/run_parallel.py --skip-convert

# Override the conservative QOS throttle (default --max-pending-jobs 75
# is sized for Chimera's observed limit; raise it on clusters with a
# higher QOSMaxSubmitJobPerUserLimit)
python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k --max-pending-jobs 200
```

submitit logs are written to `comprehensive/logs/submitit/{convert,bench}/`. Each SLURM job writes its result JSON independently to `comprehensive/results/raw/` — filenames are unique per triple, so concurrent writes are safe.

> [!TIP]
> For long-running orchestrators (full / xl tiers), wrap `gate_candidate.py` or
> `run_parallel.py` in an ``sbatch`` sentinel rather than running it interactively
> on the login node: the sentinel survives ssh drops and can be monitored via
> ``squeue`` and ``watch.py``. The orchestrator itself needs only a few CPUs
> and modest RAM; the heavy work happens in the cohort arrays it spawns.

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
   auto-loaded by `benchmarks/comprehensive/bench_env.py` through
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

Fixtures self-heal on first run: if the expected cloud object doesn't exist, `cloud_fixtures.ensure_cloud_fixture` uploads it from the local converted file via `pyscx.push` (SCX) or `gsutil -m rsync` (other formats).

**Skip behavior** — when GCP credentials can't be resolved (no env var, no `.env` entry, no `~/.gcp/scx-bench.json`), `cloud_fixtures.ensure_gcp_credentials_or_skip` writes a typed `{"missing_reason": "no_gcp_credentials"}` stub JSON via `results.write_missing_result` and the bench returns `None`. The reporting pipeline classifies the cell as a credential-absence skip instead of an undifferentiated `missing_result`. Format-incompatible (bench, format) combinations (e.g. `cloud_push` on `h5ad_*`) are filtered at cohort-build time via `SUPPORTED_FORMATS` / `REQUIRED_CAPABILITIES` and never submitted at all.

The legacy `benchmarks/scripts/benchmark_cloud.py` shim has been deleted; use the launcher above.

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

### Run all CPU benchmarks (parallel, one job per triple)

```bash
# Small datasets (D1–D4)
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k

# Large datasets — high-mem partition
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets census_500k census_1m --partition cpu_preemptible
```

### One-shot regression gate

```bash
# Capture a candidate snapshot and diff against results/baselines/LATEST
.venv/bin/python benchmarks/comprehensive/scripts/gate_candidate.py
```

### Smoke test (pbmc3k only)

```bash
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k --benchmarks read_full --formats scx_auto h5ad_gzip
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

The phase-numbered SLURM wrappers (`slurm_phase*_*.sh`,
`run_benchmarks_slurm.sh`, `slurm_parallel_write_scaling.sh`,
`slurm_lazy_preprocess_bench.sh`, `slurm_fused_bench.sh`) have been
deleted. Submit benchmarks through the comprehensive launcher:

```bash
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k

# Large datasets — the launcher auto-routes to cpu_high_mem when
# estimate_memory_gb exceeds 200 GB:
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets census_500k census_1m census_5m
```

The launcher submits (benchmark × format × dataset) triples as
resource-cohort SLURM Job Arrays via `submitit.AutoExecutor.map_array`.
Each task is sized by `estimate_memory_gb` / `estimate_time_minutes`
(`comprehensive/config.py`); the cohort takes the per-task `max(mem, time)`.
`partition_for_memory` auto-routes to `cpu_high_mem` past 200 GB; estimates
above `MEM_CEILING_GB` (1000 GB) are clamped with a logged warning.

#### Phase 6b — streaming vs in-memory sweep

A thin pre-canned wrapper submits both
`read_streaming_vs_inmemory` (on `census_500k` / `census_1m` /
`census_5m`) and `multimodal_read_streaming_vs_inmemory` (on
`cite_seq_pbmc` / `multiome_pbmc`) with the right per-partition
sizing for Lambda HPC:

```bash
sbatch benchmarks/comprehensive/scripts/slurm_read_streaming_vs_inmemory.sh
```

The wrapper sets `SCX_BENCH_HIGH_MEM_PARTITION=large_batch` so jobs
sized above the `MEM_HIGH_MEM_THRESHOLD_GB` floor (the census_5m
conversion lands at ~864 GB) auto-route to Lambda's `large_batch`
partition instead of the missing `cpu_high_mem`. Pass-through
overrides go after `--`:

```bash
sbatch slurm_read_streaming_vs_inmemory.sh -- --datasets census_1m
sbatch slurm_read_streaming_vs_inmemory.sh -- --partition standard
```

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

### ML training loader (standalone — not in the gate)

`scripts/submit_benchmarks.py` is the only path that exercises
`pyscx.TrainingDataset` against competitor loaders (TileDB-SOMA-ML,
scDataLoader, BPCells). It produces a stand-alone report at
`benchmarks/results/training_loader_benchmark.{md,json}` and is **not**
part of the regression gate — the gate's `ml_loader` benchmark
exercises only the SCX-side loader and its scenario throughput floors.

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

Preemptible partitions offer faster scheduling and lower cost. **Preempted jobs on Chimera are automatically requeued back into the preemptible partition by SLURM** — no manual resubmission needed. Because benchmark scripts are idempotent, requeued jobs pick up cleanly (existing converted files are skipped, per-triple result JSONs are unique). Use the fallback (`cpu_batch` / `gpu_batch`) partitions only when preemptible queues are too congested or when you need guaranteed completion without requeue churn (e.g., multi-day runs).

To override the partition for any script that takes a `--partition` flag:

```bash
# Example: use cpu_batch instead of cpu_preemptible
bash benchmarks/comprehensive/scripts/run_slurm.sh --partition cpu_batch
```

---

## Submitting SLURM Jobs

> [!IMPORTANT]
> **Always prefer parallel job submission over sequential single-job scripts.** Each independent (benchmark, dataset) combination should run concurrently across cluster nodes. This dramatically reduces wall-clock time (e.g., 24 parallel jobs finishing in ~2h vs one sequential job taking ~4h). Use `--dependency=afterok:$JOB1:$JOB2:...` for any post-processing that must wait for all benchmarks to complete (e.g., archiving results). Scale memory per dataset — not every job needs the largest allocation. The reference pattern is `comprehensive/scripts/run_parallel.py` — Phase A conversions submit individually, Phase B benchmarks submit as resource-cohort SLURM Job Arrays (`executor.map_array`) sized via `estimate_memory_gb` / `estimate_time_minutes`.

### Basic submission

```bash
sbatch benchmarks/scripts/slurm_gpu_analysis_bench.sh
```

### Adding exclusive node access (recommended for reproducibility)

```bash
sbatch --exclusive benchmarks/scripts/slurm_gpu_analysis_bench.sh
```

### Recommended execution order

The comprehensive harness is the committed entrypoint. The phase-numbered
legacy wrappers under `benchmarks/scripts/` have been deleted; only
dataset-prep, ML-loader, and standalone GPU/Harmony scripts remain
(see [`scripts/README.md`](scripts/README.md)).

```
1. Prepare datasets (run first — benchmarks depend on these):
   sbatch benchmarks/scripts/slurm_prep_datasets.sh
   sbatch benchmarks/scripts/slurm_build_census_5m.sh    # (optional, high-memory)
   sbatch benchmarks/scripts/slurm_build_census_10m.sh   # (optional, high-memory)

2. Run benchmarks via the unified comprehensive launcher
   (Phase A: per-(dataset, format) conversion jobs; Phase B: cohort-array
   benchmark jobs grouped by (dataset, format, partition, gres, needs_conv)):
   python benchmarks/comprehensive/scripts/run_parallel.py \
       --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k
   python benchmarks/comprehensive/scripts/run_parallel.py \
       --datasets census_500k census_1m census_5m

3. Cloud benchmarks (GCP — requires GOOGLE_APPLICATION_CREDENTIALS;
   see check_gcp_auth.py preflight):
   python benchmarks/comprehensive/scripts/check_gcp_auth.py
   python benchmarks/comprehensive/scripts/run_parallel.py \
       --benchmarks cloud_push cloud_pull cloud_read cloud_metadata \
                    cloud_filtered cloud_reader_vs_pull cost_model \
       --datasets pbmc3k tabula_sapiens_100k

4. Regression gate (on-demand — per PR, pre-release, on-suspicion):
   python benchmarks/comprehensive/scripts/gate_candidate.py
```

> [!IMPORTANT]
> Always run dataset preparation jobs first. The comprehensive harness
> expects datasets at `SCX_DATA_DIR` (configured via `.env`; defaults to
> `$SCX_WORK_DIR/benchmarks/datasets`). Cloud benchmarks additionally
> require one-time fixture staging via
> `benchmarks/comprehensive/scripts/setup_cloud_test_data.sh`.

### Refreshing derived fixtures after an h5ad change

When the source h5ad fixtures change (e.g. an obs-column augmentation),
the derived `scx_auto`, `tiledb_soma`, and `zarr_zstd` local files —
and their cloud-pushed copies — go stale. Use `reconvert_fixtures.py`
to drive the full re-conversion + push pipeline:

```bash
# Default matrix: all datasets × {scx_auto, tiledb_soma, zarr_zstd}, local only
python benchmarks/scripts/reconvert_fixtures.py

# Cherry-pick datasets + formats and re-push to GCS
python benchmarks/scripts/reconvert_fixtures.py \
    --datasets pbmc3k pbmc10k tabula_sapiens_100k \
    --formats scx_auto tiledb_soma \
    --cloud-push

# Dry-run to see the plan without converting or uploading
python benchmarks/scripts/reconvert_fixtures.py --dry-run --cloud-push

# List known datasets / format keys
python benchmarks/scripts/reconvert_fixtures.py --list
```

`--cloud-push` invalidates the existing GCS copy (main directory +
`.blake3` sidecar) before uploading so `ensure_cloud_fixture`'s
completion short-circuit doesn't skip the fresh data. Requires
`GOOGLE_APPLICATION_CREDENTIALS` (or ADC) and `gsutil` on PATH.

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
- **`benchmarks/comprehensive/results/raw/`** — Comprehensive benchmarks (500+ JSON files, one per benchmark×format×dataset)

To list comprehensive results:

```bash
ls benchmarks/comprehensive/results/raw/*.json | wc -l   # ~500 results
ls benchmarks/comprehensive/results/raw/*census_1m*       # D6 results
```

---

## Report Generation

The benchmark report is generated from raw JSON results via a structured
pipeline (report model → section builders → lint → renderers). All report
generation uses the `report_cli.py` entry point:

```bash
# Generate the full report (markdown + HTML + PDF) from default results
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli generate

# Generate with strict lint (errors on manual-source numeric claims)
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli generate --strict-lint

# Generate with public profile (blocks internal phase labels)
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli generate --profile public

# Point at a different results directory
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli generate \
    --results-dir /path/to/results/raw --output-dir /path/to/reports

# Save a timestamped snapshot alongside the report
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli generate \
    --snapshot-dir benchmarks/comprehensive/results/reports/snapshot_$(date +%Y%m%d)

# Choose specific output formats (md, html, pdf, json)
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli generate \
    --format md --format html --format json

# Lint-only mode (no files written)
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli lint
PYTHONPATH=. python -m benchmarks.comprehensive.reporting.report_cli lint --strict
```

### Profiles

| Profile | Description |
|---------|-------------|
| `default` | Full engineering report with all sections. |
| `public` | Public-facing report. Internal phase labels in headings produce lint errors. |
| `engineering` | Engineering report with internal phase labels allowed. |

### Output Artifacts

| File | Description |
|------|-------------|
| `BENCHMARK_REPORT.md` | Primary report (GitHub Flavored Markdown). |
| `BENCHMARK_REPORT.html` | Semantic HTML snapshot with ToC and styling. |
| `BENCHMARK_REPORT.pdf` | PDF via pandoc (requires `pandoc` + `xelatex`). |
| `BENCHMARK_REPORT.json` | Machine-readable JSON manifest of the report AST. |
| `LINT_WARNINGS.json` | Lint findings from the most recent generation. |
| `dashboard_history.json` | Rolling dashboard snapshot history. |
| `figures/*.png` | Publication-quality plot PNGs. |
| `figures/*.provenance.json` | Per-figure provenance sidecars (source benchmark, timestamp, SHA-256). |

### Legacy compatibility

The `write_reports()` function in `reporting/markdown.py` remains as a
compatibility wrapper. New integrations should use `report_cli.py` or
the `build_report()` function in `reporting/report_cli.py` directly.

---

## Regression Gating

Regression gating is **on-demand, not scheduled**. No nightly cron runs —
operators invoke the gate per PR, pre-release, or on suspicion of
regression. No wasted compute when nothing has changed, and every gate
result is tied to a specific commit the operator cares about.

### Which script? `gate_candidate.py` is the canonical benchmark entry point

`gate_candidate.py` (which orchestrates `capture_baseline.py` →
`run_parallel.py` → `compare_against_baseline.py`) is **the
comprehensive SCX benchmark**. It covers every aspect of SCX
performance currently under regression governance:

| Surface | Benchmarks |
|---|---|
| **Codec / file-format size** | `compression` |
| **I/O — bulk and projected reads, writes** | `write`, `read_full`, `read_selective` |
| **Parallel scaling** | `parallel_scaling`, `parallel_write_scaling` |
| **Memory** | `memory` |
| **ML training loader (sequential)** | `ml_loader` |
| **Plan-driven paired reads (perturbation training)** | `index_plan` |
| **SCX-only fragment / manifest ops** | `fragment_ops` |
| **Correctness parity (scanpy / backed / preprocessing)** | `correctness` |
| **Codec round-trip parity** | `roundtrip` |
| **Cell-eval / arc-bench parity perf** | `cell_eval_parity_perf` |
| **Cloud (GCP) — push, pull, read, metadata, query, large-atlas, cost model** | `cloud_push`, `cloud_pull`, `cloud_read`, `cloud_metadata`, `cloud_filtered`, `cloud_reader_vs_pull`, `cost_model`, `cloud_large_atlas` |
| **Analysis accelerators (CPU + GPU)** | `accel_pca`, `accel_knn`, `accel_umap`, `accel_leiden`, `accel_preprocess`, `accel_hvg` |
| **Multimodal — h5mu compression and training-loader throughput** | `multimodal_compression`, `multimodal_training` |

That is 29 benchmarks across 12 distinct domains, each expanded across the
relevant format variants (h5ad / zarr / scx / tiledb / parquet / bpcells
plus accelerator-implementation variants like `accel_pca__pyscx_gpu_cov`)
and the tier's dataset list (pbmc3k → census_10m). The canonical list
lives in [`benchmarks/comprehensive/benchmarks/__init__.py::ALL_BENCHMARKS`](comprehensive/benchmarks/__init__.py).
Adding a new benchmark to the suite is a one-line edit there — every
capture run picks it up automatically.

> **Capture vs gate coverage are not the same.** The capture phase runs
> every entry in `ALL_BENCHMARKS`. The gate phase only flags regressions
> on rows that exist in the chosen baseline. Inspect what your baseline
> actually contains with:
>
> ```bash
> .venv/bin/python -c "import json; \
>     keys=set(k.split('__')[0] for k in \
>         json.load(open('benchmarks/comprehensive/results/baselines/LATEST/summary.json'))['rows']); \
>     print(sorted(keys))"
> ```
>
> The current `LATEST` symlink points at `v0.6.2-n_counts-augmentation`
> (captured 2026-05-11 from `b1629ea`, tier `full`, 806 rows across all
> 8 datasets — `pbmc3k`, `pbmc10k`, `smartseq2`, `tabula_sapiens_100k`,
> `census_500k`, `census_1m`, `cite_seq_pbmc_5k`, `multiome_pbmc_10k`).
> It gates **format** (`compression`, `read_full`, `read_selective`,
> `write`, `memory`, `parallel_scaling`, `parallel_write_scaling`,
> `roundtrip`), **cloud** (`cloud_metadata`, `cloud_read`,
> `cloud_filtered`, `cloud_pull`, `cloud_push`, `cloud_large_atlas`,
> `cloud_reader_vs_pull`, `cost_model`), **`ml_loader`**, **multimodal**
> (`multimodal_compression`, `multimodal_training`), `index_plan`,
> `fragment_ops`, and `correctness`. The remaining gap is the `accel_*`
> family (`accel_pca`, `accel_knn`, `accel_umap`, `accel_leiden`,
> `accel_preprocess`, `accel_hvg`) — those benchmarks capture cleanly
> but produce no gate signal against `LATEST`. Pin
> `--baseline benchmarks/comprehensive/results/baselines/v0.6.0-gpu-phase1-7`
> when you need accelerator gating, until the next multi-surface
> baseline that re-includes `accel_*` is promoted as `LATEST`.

**Accelerator route gates.** Every GPU accelerator benchmark records the
execution route each call actually took (read back from
`adata.uns["scx_accel"]`, written by pyscx) and emits binary gate signals in
`runs[].extra`. `thresholds.yaml` declares absolute floors (`min: 1.0`) on
these signals so a silent GPU→CPU fallback or a CSC→CSR fallback becomes a
hard gate failure via the existing absolute-floor machinery.

| Gate metric | Benchmark module | What it catches |
|------------|------------------|-----------------|
| `de_route_csc_direct` | `accel_de` (pdex_ref GPU) | CSC sidecar + `SCX_GPU_DE_V3=1` active, yet a non-CSC route ran. Gated on `pbmc3k`, `tabula_sapiens_100k`, and `census_1m`. |
| `wilcoxon_route_gpu_correct` | `accel_de` (Wilcoxon GPU) | `device="gpu"` requested but a `cpu_*` route ran. Gated on `pbmc3k` and `tabula_sapiens_100k`. |
| `csc_dispatch_correct` | `bench_csc_dispatch` | A `_csc`-labelled variant ran a non-CSC route (or vice versa). Gated on `tabula_sapiens_100k` for `qc_metrics`, `hvg`, `de`, and `pdex_ref` CSC variants. |
| `hvg_route_gpu_correct` | `accel_hvg` | GPU HVG dispatch silently fell back to CPU. Gated on `pbmc3k`. |
| `pca_route_gpu_correct` | `accel_pca` | GPU PCA dispatch silently fell back to CPU. Gated on `pbmc3k` and `tabula_sapiens_100k`. |
| `knn_route_gpu_correct` | `accel_knn` | GPU kNN dispatch silently fell back to CPU. Gated on `pbmc3k` and `tabula_sapiens_100k`. |
| `umap_route_gpu_correct` | `accel_umap` | GPU UMAP dispatch silently fell back to CPU. Gated on `pbmc3k` and `tabula_sapiens_100k`. |
| `leiden_route_gpu_correct` | `accel_leiden` | GPU Leiden dispatch silently fell back to CPU. Gated on `pbmc3k`. |

Each gate metric is `1.0` when the expected route ran (or wasn't applicable —
e.g. no GPU host), `0.0` on a silent fallback. All GPU benchmark modules also
emit `gpu_dispatch_route` (the stable wire identifier, e.g. `gpu_csc_v3`) and
`gpu_dispatch_fallback` (e.g. `no_csc_sidecar`) in `runs[].extra` for
diagnostics. Benchmark provenance (`provenance.py`) captures the route-
affecting env vars `SCX_GPU_DE_V2`, `SCX_GPU_DE_V3`, and
`SCX_GPU_DE_V3_TRACE` so a result can always be attributed to its dispatch
configuration.

**`scripts/submit_benchmarks.py` is a narrow specialty tool**, not a
peer of `gate_candidate.py`. It exclusively drives
[`benchmark_loader.py`](scripts/benchmark_loader.py), which targets the
one performance surface the comprehensive suite doesn't yet cover: the
SCX **ML training loader** (`pyscx.TrainingDataset` throughput, latency,
peak RSS, HVG-projection impact, GPU utilization under an scVI-style
VAE loop, multi-GPU scaling, and side-by-side comparisons against
AnnData / SOMA / scdataloader / BPCells loaders). It produces a
stand-alone report (`benchmarks/results/training_loader_benchmark.{md,json}`)
with no baseline diff and no gate semantics. There is a placeholder
`ml_loader` time-budget slot in [`comprehensive/config.py`](comprehensive/config.py)
hinting at future integration, but the benchmarks live outside the
gate's scope today.

**Choosing between them:**

| | `comprehensive/scripts/gate_candidate.py` | `scripts/submit_benchmarks.py` |
|---|---|---|
| **What it benchmarks** | Everything in the canonical SCX surface — codec, I/O, parallel scaling, memory, cloud, analysis accelerators | One surface only: the ML training loader and its competitive baselines |
| **Compares against a baseline?** | Yes — diffs the captured snapshot against `results/baselines/LATEST` via `compare_against_baseline.py --gate`, applies justifications and absolute floors | No baseline diff; produces a stand-alone report each run |
| **Exit codes** | Gate-shaped: `0` pass / `1` regression / `2` missing inputs / `130` SIGINT | Per-job loader-benchmark exit codes only |
| **Output** | `benchmarks/comprehensive/results/<candidate>/` (snapshot, gitignored) + `comprehensive/logs/gate_candidate_<sha>_<TS>.{log,summary.json}` | `benchmarks/results/training_loader_benchmark.{md,json}` (overwrites in place) |
| **GPU pre-flight** | Submits a SLURM probe job to validate `nvidia-smi` / `cupy` / `pyscx.accel` on a real GPU node before scheduling work | None — assumes the partition the operator picked has GPUs |
| **Use when** | Pre-PR / pre-merge / pre-release validation of any SCX change; characterising codec / I/O / cloud / accel performance against the canonical baseline | Iterating on a `TrainingDataset` change, sweeping `n_gpus` for loader scaling curves, or producing a loader-vs-competitor comparison report |

If you're unsure which to run, the answer is `gate_candidate.py`. It is
the only tool that sees the whole SCX performance surface.
`submit_benchmarks.py` is a focused profiler for the training loader
specifically — useful when iterating on loader internals, but not a
substitute for running the gate before opening a PR.

### On-demand workflow (one command)

```bash
python benchmarks/comprehensive/scripts/gate_candidate.py
```

That captures a snapshot named `candidate_<git-sha>_<YYYYMMDD>` at the
`small` tier and runs the gate against `results/baselines/LATEST`
(maintained by `promote_baseline.py`). The default coverage matrix is
**format + accel CPU + accel GPU**. Exit code bubbles up: `0` = pass,
`1` = unjustified regression / floor violation / fingerprint mismatch,
`2` = pre-flight failure / missing inputs, `130` = SIGINT.

Each invocation writes a timestamped log to
`benchmarks/comprehensive/logs/gate_candidate_<sha>_<TS>.log` plus a
sidecar `*.summary.json` with the phase / exit / elapsed / coverage
plan for downstream tooling.

The script is also executable directly:
`benchmarks/comprehensive/scripts/gate_candidate.py`.

Common options:

```bash
# Larger tier (full dataset set)
python benchmarks/comprehensive/scripts/gate_candidate.py --tier full

# Reuse an already-captured candidate (skips the capture step)
python benchmarks/comprehensive/scripts/gate_candidate.py \
    --skip-capture --name candidate_abc1234_20260420

# Pin a specific historical baseline instead of LATEST
python benchmarks/comprehensive/scripts/gate_candidate.py \
    --baseline benchmarks/comprehensive/results/baselines/v0.4.0

# Anything after `--` is forwarded to compare_against_baseline.py
python benchmarks/comprehensive/scripts/gate_candidate.py \
    -- --timing-tolerance 0.05 --report-json /tmp/gate.json
```

#### CPU / GPU / accel modes

The gate covers three benchmark axes — pick a subset when iterating
locally to shorten turnaround. Pre-flight checks fail fast if the host
can't satisfy the requested mode (e.g. `--no-gpu` was not set but
`nvidia-smi` reports no GPUs).

```bash
# CPU-only (skip accel GPU variants — useful on a GPU host validating
# a CPU-only refactor, or on a CPU-only laptop):
python benchmarks/comprehensive/scripts/gate_candidate.py --no-gpu

# Accel only — fast iteration on PCA / kNN / UMAP / Leiden kernels:
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only

# Accel CPU only (CPU-only laptop, no GPU coverage at all):
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only --no-gpu

# Format only (skip accel — older default; useful for codec / sharding work):
python benchmarks/comprehensive/scripts/gate_candidate.py --no-accel
```

`--no-accel` and `--accel-only` are mutually exclusive; `--no-gpu` is
orthogonal and composes with either. The coverage banner the script
prints before capture (`format benchmarks ✓ / accel CPU ✓ / accel GPU ✗`)
is the source of truth for what was actually exercised.

**Recommended trigger points:**

- Pre-PR: run on your topic branch before opening the PR.
- Pre-merge: re-run on the merge candidate if CPU / memory-sensitive code changed.
- Pre-release: run at the `xl` tier; promote the candidate as the new
  baseline if the gate passes (see "Promoting" below).
- On-suspicion: after a suspicious benchmark result, landed profiler
  change, or upstream dependency bump.

There is no CI-side gate — the prior `.github/workflows/accel-gate.yml`
was removed because the self-hosted GPU runner queue made the loop
unworkable. Run the gate locally before merging any PR that touches
accelerator paths.

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
- `--flakiness` → `results/flakiness/`
- `--thresholds` → `benchmarks/comprehensive/thresholds.yaml`

Override any of them by passing the flag explicitly. Without `--gate` the
script stays in pure-diff mode — no justification or floor logic, no
disappeared-benchmark flagging.

Exit codes: `0` = pass, `1` = unjustified regression / floor violation /
fingerprint mismatch, `2` = baseline or current directory missing or no
canonical baseline promoted yet.

### Ad-hoc A/B for a single benchmark (no baseline rows yet)

When the benchmark you care about isn't in `LATEST` (e.g. `index_plan` and
`ml_loader` against the current accel-only baseline), the gate produces
no signal. Use a manual A/B instead — capture the metric on both branches
and diff. Two pitfalls show up reliably:

1. **Page-cache state dominates the first run.** A 2-3 GB SCX file that
   isn't in `/proc/sys/vm/drop_caches` makes the first run 3× slower than
   the second. Always do an untimed warm-up read before timing, and run
   the same scenarios on both branches in alternating order so neither
   branch gets the cold-cache run. The two-batch warm-up loop in the
   benchmark module's `_run_index_plan` (`benchmarks/comprehensive/benchmarks/index_plan.py`)
   is the pattern to copy for ad-hoc scripts.
2. **Wheel state must match the checkout.** `maturin develop --release`
   leaves the previously-built `.so` installed if you only `git checkout`
   the source. Rebuild after every checkout — both the format crate and
   any pyscx changes have to land in the wheel before the python harness
   sees them. A safe template:

   ```bash
   # On candidate branch
   (cd pyscx && ../.venv/bin/maturin develop --release)
   .venv/bin/python my_bench.py --label after  > after.json

   # Switch and rebuild — DO NOT skip the rebuild
   git checkout <baseline-sha>
   (cd pyscx && ../.venv/bin/maturin develop --release)
   .venv/bin/python my_bench.py --label before > before.json

   # Rebuild back to head when done so the working tree's .so matches HEAD
   git checkout -
   (cd pyscx && ../.venv/bin/maturin develop --release)
   ```

   Run each side at least twice (after a warm-up); take the second number.

### Promoting a canonical baseline

Snapshots land in `benchmarks/comprehensive/results/<name>/` from
`capture_baseline.py` (or via the `gate_candidate.py` wrapper). To
promote one as the canonical release baseline:

```bash
python benchmarks/comprehensive/scripts/promote_baseline.py \
    --snapshot benchmarks/comprehensive/results/candidate_2026_04_18_batch_d_t3 \
    --version v0.5.0-phase5
```

> Change `--version v0.5.0-phase5` as needed

This copies only `summary.json` + `environment.json` + `MANIFEST.sha256`
into `results/baselines/v0.5.0-phase5/`, and updates `results/baselines/LATEST`
to point at the new version (relative symlink, pointer-file fallback on
filesystems that reject symlinks). The gate's `--baseline` auto-resolves
to `LATEST`, so no follow-up configuration is needed — next
`gate_candidate.py` run compares against the newly-promoted baseline.

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

### Variance-aware timing tolerance (IQR widening)

The gate widens the per-row `median_wall_s` tolerance automatically when
the *baseline's own* run-to-run dispersion exceeds the global
`--timing-tolerance` floor. Effective tolerance per timing row is

```
max(--timing-tolerance, --iqr-k * baseline_iqr / baseline_median)
```

where `baseline_iqr` is the IQR of `runs[].wall_s` recorded at capture
time (persisted on every `summary.json` row as `wall_s_iqr`). With the
defaults (`--timing-tolerance 0.03`, `--iqr-k 1.5`), a rock-stable row
stays gated at 3 %, while a row whose baseline IQR/median is 5 % auto-
widens to 7.5 % — the bare 3 % gate's false-positive rate on noisy
shared-node samples drops without operator intervention.
The header line "Timing rows widened by IQR" reports how many rows
crossed the floor; the "Tol" column suffixes widened rows with `~` and
override-relaxed rows with `⚠`. Set `--iqr-k 0` to disable.

When the baseline predates this mechanism (no `wall_s_iqr` field), the
gate logs a single WARN and falls back to the fixed `--timing-tolerance`.
Re-capture and re-promote with `capture_baseline.py` + `promote_baseline.py`
to enable variance-aware gating.

### Flakiness ledger

Use a flakiness override **only** when a row's noise floor is genuinely
not represented by its baseline IQR — bimodal wall-time across runs,
periodic GC pauses that fall outside the captured sample window, or a
benchmark that's known to be unstable beyond what `--iqr-k 1.5` will
absorb. For ordinary "shared-node noise" the IQR-widening above already
handles it; reach for the ledger only when that mechanism is itself
insufficient. The override raises the per-row tolerance above the
IQR-widened bound; a real regression that crosses the relaxed bound
still trips. Reach for this only after exhausting `--iqr-k` (and only
for the rare rows that need it) — never as a substitute for a global
`--timing-tolerance` bump.

Each entry lives in `benchmarks/comprehensive/results/flakiness/`:

```markdown
---
overrides:
  - benchmark: cloud_pull
    format: scx_auto
    dataset: tabula_sapiens_100k
    metric: median_wall_s   # optional; defaults to median_wall_s
    tolerance: 0.08         # required; relaxed bound for this row
reason: "Shared SLURM node — 7% wall-time RSD across 10 runs (issue #1234)."
expires: 2026-07-01
---

Optional prose explaining what would let the override expire (e.g.
``--exclusive`` SLURM allocation, NUMA pinning, switching the queue).
```

Set `metric:` explicitly to `peak_rss_mb_median` or `file_size_bytes`
to relax those rows; otherwise the entry applies to `median_wall_s`
only. `tolerance` must be non-negative — overrides relax, they don't
tighten. Two entries for the same `(benchmark, format, dataset, metric)`
quad keep the **stricter** tolerance and log a warning, so overlapping
files should be collapsed during cleanup.

The gate report annotates every `median_wall_s` row with its observed
coefficient of variation (`stdev/mean`) computed from the candidate's
`runs[].wall_s`, and surfaces a "High-variance rows" section listing
rows with CV > 5% — those are the natural candidates for a ledger
entry.

|                  | Justification              | IQR widening (`--iqr-k`)                  | Flakiness override                        |
|------------------|----------------------------|-------------------------------------------|-------------------------------------------|
| Effect           | Drops row from regression tally entirely | Auto-widens timing tolerance using baseline's own dispersion | Raises the row's per-metric tolerance |
| Use when         | Real regression accepted   | Default — handles ordinary measurement noise transparently | Bimodal / pathological noise IQR misses |
| Trigger          | Manual (markdown)          | Automatic (per `summary.json` row)        | Manual (markdown)                         |
| Real regression beyond the bound | Hidden        | Still trips the gate                      | Still trips the gate                      |
| Per-metric scope | All metrics for the triple | `median_wall_s` only                      | One metric (defaults to `median_wall_s`)  |

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
disappearing benchmarks flagged, absolute-floor violations fail,
flakiness override relaxes per-row tolerance):

```bash
.venv/bin/pytest benchmarks/comprehensive/tests/test_gate_self_test.py -v
```

Run it locally before opening a PR that touches the gate.

---

## GPU accelerator regression workflow

The comprehensive baseline at
`comprehensive/results/baselines/LATEST` (currently
`v0.6.0-gpu-phase1-7-multidataset`) covers both format-level **and**
GPU-accelerator benchmarks (`accel_pca`, `accel_knn`, `accel_umap`,
`accel_leiden`, `accel_preprocess`, `accel_hvg`) across pbmc3k,
tabula_sapiens_100k, and census_1m. Per-run correctness metrics
(`cosine_sim_min`/`mean`, `recall_vs_scanpy`, `trustworthiness`,
`ari_vs_leidenalg`, `max_abs_diff_vs_scanpy`,
`hvg_overlap_vs_scanpy`) flow into `runs[].extra` so the absolute
floors in `thresholds.yaml` evaluate real values, not
`missing` placeholders.

**You should not need separate GPU tooling.** The standard
[Regression Gating workflow](#regression-gating) handles accelerator
PRs end-to-end. The stop-gap wrappers
(`gpu_regression_driver.sh`, `gpu_regression_diff.py`,
`slurm_gpu_regression*.sh`) have been deleted — use
`gate_candidate.py` for everything below.

### When this workflow applies

Run the full accel sweep when a PR touches any of:

- `scx-gpu/src/**` — kernels, cuSPARSE / cuSOLVER / cuBLAS wrappers,
  shard pipeline, linear operator.
- `scx-accel/src/{pca,neighbors,umap,leiden,hvg,harmony}.rs` — Rust-side
  accelerator entry points (CPU paths gate the CPU baseline).
- `pyscx/src/accel/**` — Python dispatch, device resolution,
  preprocessing materialization, fusion markers.
- `pyscx/Cargo.toml` / `scx-gpu/Cargo.toml` — dep bumps, feature
  toggles (especially `cudarc` minor-version bumps).

PRs that don't touch these paths can pass `--no-accel` to keep the gate
to format benchmarks only.

### One-shot accel + format gate

```bash
# Default coverage = format + accel CPU + accel GPU. Captures
# candidate_<sha>_<date> at the small tier and gates against LATEST.
python benchmarks/comprehensive/scripts/gate_candidate.py

# Accel-only iteration (skip format benchmarks, full GPU coverage):
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only

# Accel CPU only (CPU-only host or CPU-only refactor on a GPU box):
python benchmarks/comprehensive/scripts/gate_candidate.py --accel-only --no-gpu
```

The script writes a structured log under
`benchmarks/comprehensive/logs/gate_candidate_<sha>_<TS>.log` with the
full subprocess output and a sidecar `*.summary.json` recording the
phase, exit code, elapsed time, and coverage matrix. Pre-flight checks
fail fast (exit 2) if `nvidia-smi`, `cupy`, or `pyscx.accel` isn't
reachable when GPU/accel coverage was requested — the error message
points at the missing piece.

Exit codes match the standard gate: `0` pass, `1` regression / floor
violation, `2` pre-flight failure / missing inputs.

### Routing details (no operator action needed)

`run_parallel.py` automatically:

- Routes `accel_*__*_gpu*` formats to `partition=preemptible
  slurm_gres=gpu:1` and clamps memory at 128 GB to fit the GPU QOS cap.
- Routes `accel_*__*_cpu*` / `__scanpy_cpu` / `__leidenalg_cpu` formats
  to `cpu_preemptible`, with `partition_for_memory()` auto-promoting to
  `cpu_high_mem` when an estimate exceeds 200 GB (e.g.
  `accel_preprocess` on `census_1m`).
- Sizes per-cell wall-time budgets via `estimate_time_minutes` (umap
  120 min base, leiden 90, knn 60, preprocess 60, hvg 45, pca 30) plus
  a +8 min/M-cells slope. CPU UMAP on census_1m lands at ~130 min.
- Filters cross-products so `accel_<bench>` only pairs with
  `accel_<bench>__*` formats (one accel module never schedules another
  module's variants).

### Promoting a new baseline

When the post-change run shows expected gains (and any threshold
recalibration in `thresholds.yaml` has landed), promote the snapshot
the same way as format baselines:

```bash
python benchmarks/comprehensive/scripts/promote_baseline.py \
    --snapshot benchmarks/comprehensive/results/$CAND \
    --version v0.X.Y-gpu-<descriptor>
```

This copies `summary.json` + `environment.json` + `MANIFEST.sha256`
into `comprehensive/results/baselines/<version>/` and repoints
`LATEST`. Add a row to `comprehensive/results/baselines/README.md`'s
versions table.

### Historical baselines

The frozen pre-GPU-accel snapshot at
`benchmarks/results/pre_phases_1_7_baseline_2026_03/` is retained for
historical bisects but is not the gate target.


## Individual Benchmark Scripts

The legacy `benchmarks/scripts/benchmark_*.py` one-off entrypoints have
been deleted; their measurements live under
`benchmarks/comprehensive/benchmarks/`. The remaining scripts in
`benchmarks/scripts/` are dataset prep, the standalone ML training
loader, and active-development GPU / Harmony surfaces — see
[`scripts/README.md`](scripts/README.md).

| Script | What it measures |
|--------|-----------------|
| `scripts/benchmark_loader.py` | ML data loader throughput vs SOTA baselines (TileDB-SOMA-ML, scDataLoader, BPCells). **Not in the gate** — the gated `ml_loader` benchmark covers only the SCX-side loader |
| `scripts/benchmark_cli.py` | CLI command performance |
| `scripts/benchmark_python_bindings.py` | Python bindings overhead |
| `scripts/benchmark_gpu_decode.py` | GPU decode microbenchmarks (cuSPARSE, bitstream) |
| `scripts/benchmark_gpu_pca.py` | GPU PCA validation + timing (kernel-level optimization) |
| `scripts/benchmark_gpu_knn.py` | GPU kNN (CAGRA) validation + timing |
| `scripts/benchmark_gpu_umap.py` | GPU UMAP validation + timing |
| `scripts/benchmark_gpu_preprocess.py` | GPU fused preprocessing (normalize+log1p) |
| `scripts/benchmark_gpu_pipeline.py` | End-to-end GPU pipeline + Go/No-Go gate |
| `scripts/benchmark_gpu_scvi.py` | GPU scVI training benchmark |
| `scripts/benchmark_harmony.py` | Harmony2 batch correction performance |
| `scripts/benchmark_lisi.py` | LISI metric performance |
| `scripts/benchmark_bpcells.R` | BPCells comparison (R) — driver for `benchmark_loader.py --include-bpcells` |
| `comprehensive/benchmarks/cell_eval_parity_perf.py` | cell-eval / arc-bench parity perf (pseudobulk, perturbation metrics, energy distance, discrimination score, knockdown efficiency, clustering agreement) at synthetic 100K–1M scale |

## Rust microbenchmarks (criterion)

Kernel-level benches live alongside the crates they exercise. Run via
`cargo bench`:

| Bench | Crate | What it measures |
|-------|-------|-----------------|
| `codec_bench` | `scx-codec` | Encode/decode throughput per codec (Rice, FOR-BP, Delta-Golomb, LZ4-shuffle, Zstd) |
| `catalog_parse` | `scx-format` | Round-trip cost of `FullCatalog::read_from` at census scale. Synthesises a catalog with `n_shards ∈ {64, 1024, 16384}` v2 CSR entries + the usual obs/var/index/provenance/uns entries, serialises it to bytes, then measures the parse-back cycle. Used to baseline the per-entry `String` / stats-payload allocation reductions; the 16K shard size matches the worst case the `index_plan/scx_auto/census_1m` cells hit when the workers2 path opens one `BackedCsrReader` per worker. |
| `distances` | `scx-accel` | `mean_pairwise_distance` and `mean_pairwise_distance_self` across `(n_a, n_b, n_dims)` shapes × `metric ∈ {euclidean, l1, cosine}` × `backend ∈ {scalar, gemm}` × `dtype ∈ {f32, f64}`. Filter by criterion regex, e.g. `cargo bench -p scx-accel --bench distances -- 'gemm/cosine'`. |

```bash
# Catalog parse: full sweep (64 / 1024 / 16384 shards)
cargo bench -p scx-format --bench catalog_parse

# Just the 16K-entry workload (matches census_1m amplification)
cargo bench -p scx-format --bench catalog_parse -- 'parse/16384'

# All distance microbenches (full grid takes ~30 minutes)
cargo bench -p scx-accel --bench distances

# Just the f32 + gemm + euclidean cross combos (a few minutes)
cargo bench -p scx-accel --bench distances -- 'mean_pairwise_distance/f32/.*/euclidean/gemm'

# Ad-hoc: print without measuring (target enumeration)
cargo bench -p scx-accel --bench distances -- --list
```

Sample budgets are tuned per shape so the full grid stays under tens of
minutes; the heaviest combos (5000×2000×18000 scalar f64) drop to
`sample_size=10` / `measurement_time=30s` to stay tractable.

`catalog_parse` baselines the `FullCatalog::read_from` path only — the
lightweight `CatalogView::read_from_bytes` and the `Arc<FullCatalog>`
sharing across the N+3 `to_anndata_backed` opens are not exercised by this
bench. Comparing those paths requires a dedicated `backed_open` microbench
against a 16K-entry catalog (flagged as a follow-up; not landed). The
ML-loader-level signal lives in `comprehensive/results/baselines/LATEST`'s
`ml_loader/pyscx_index_plan_dataset_*` floors.
