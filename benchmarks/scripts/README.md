# benchmarks/scripts — Active dev surfaces and dataset prep

This directory holds scripts that are **not** part of the comprehensive
benchmark suite (`benchmarks/comprehensive/`). Each falls into one of
four categories:

1. **Dataset preparation** — downloaders, converters, large-dataset builders
   that produce the `*.h5ad` fixtures the comprehensive suite consumes.
2. **ML training loader** — the only path that exercises
   `pyscx.TrainingDataset` against competitor loaders. The comprehensive
   suite does not yet cover this surface.
3. **Active dev surfaces (GPU / Harmony / LISI)** — standalone benches
   for surfaces that pre-date the comprehensive harness or that produce
   dedicated reports rather than gateable timing rows. Will be folded
   into the comprehensive suite when the per-area unified launcher lands.
4. **Test fixtures** — generators called by the pyscx test suite
   (e.g. Harmony validation fixtures).

The previous deprecation tree under this directory (Phase wrappers
`slurm_phase*_*.sh`, the original `benchmark_*.py` one-off entrypoints,
`gpu_regression_*` stop-gaps, the `benchmark_cloud.py` shim) has been
deleted. For format-level / accel benchmarks use
`benchmarks/comprehensive/scripts/run_parallel.py` (or the
`gate_candidate.py` wrapper) — see [`benchmarks/README.md`](../README.md).

---

## What lives here

### Dataset preparation

| File | Purpose |
|---|---|
| `download_datasets.sh` | Driver for D1–D6 download + h5ad conversion |
| `download_pbmc10k.py` | D2 specifically |
| `download_census_500k.py` | D5 specifically |
| `build_census_500k.py` / `build_census_5m.py` / `build_census_10m.py` | Build large CELLxGENE Census slabs from chunks |
| `slurm_prep_datasets.sh` | SLURM wrapper for D1–D6 prep |
| `slurm_build_census_5m.sh` / `slurm_build_census_10m.sh` | High-mem SLURM wrappers for D7 / D8 |
| `verify_datasets.py` | Validate every dataset file + record metadata |
| `generate_compressed_h5ad.py` | Pre-generate `_gzip.h5ad` / `_lzf.h5ad` variants |
| `prep_lognorm_datasets.py` | Materialize log-normalized variants for Pcodec float benches |
| `augment_obs_n_counts.py` | Backfill `obs.n_counts` for legacy fixtures |
| `setup_cloud_test_data.sh` | Initial GCS test-data upload (one-time bootstrap) |

### ML training loader

| File | Purpose |
|---|---|
| `submit_benchmarks.py` | submitit launcher for the loader benchmark |
| `benchmark_loader.py` | Loader throughput vs SOTA (TileDB-SOMA-ML, scDataLoader, …) |
| `benchmark_bpcells.R` | Driver invoked by `benchmark_loader.py --include-bpcells` |

The comprehensive suite's `ml_loader.py` module is now wired into
`ALL_BENCHMARKS` and gated via `thresholds.yaml`, so the comprehensive
harness covers SCX/h5ad/SOMA loader throughput. The standalone scripts
here remain because they drive the SOTA loader comparison
(TileDB-SOMA-ML, scDataLoader, BPCells) which the comprehensive suite
does not currently host. Output:
`benchmarks/results/training_loader_benchmark.{md,json}`.

### Active dev surfaces — GPU benches

| File | Purpose |
|---|---|
| `slurm_gpu_bench.sh` + `benchmark_gpu_decode.py` | GPU decode microbenchmark |
| `slurm_gpu_knn_bench.sh` + `benchmark_gpu_knn.py` | CAGRA kNN validation |
| `slurm_gpu_pca_opt_bench.sh` + `benchmark_gpu_pca.py` | GPU PCA tuning |
| `slurm_gpu_analysis_bench.sh` + `benchmark_gpu_{umap,preprocess,pipeline,scvi}.py` | End-to-end GPU pipeline + Go/No-Go gate |

The comprehensive suite covers `accel_pca` / `accel_knn` / `accel_umap`
/ `accel_preprocess` more uniformly; these standalone benches remain
for kernel-level optimization work that doesn't map onto the
gate-shaped `accel_*__<impl>` variant matrix.

### Active dev surfaces — Harmony / LISI

| File | Purpose |
|---|---|
| `slurm_harmony_bench.sh` | Driver |
| `benchmark_harmony.py` | Harmony2 batch correction performance |
| `benchmark_lisi.py` | LISI metric performance |
| `harmony_bench_worker.py` | Per-config worker |
| `report_harmony.py` | Markdown / plot generator |
| `build_harmony_validation_fixtures.py` | Builds the fixtures `pyscx/tests/test_harmony_validation.py` consumes |
| `generate_harmony_reference.R` | R-harmony reference (called from the fixture builder) |

### Helpers (not benchmarks)

| File | Purpose |
|---|---|
| `bench_env.py` | **Moved** — now `benchmarks/comprehensive/bench_env.py` |
| `build_release.py` | Ensure `pyscx` is built in release mode |
| `run_with_rayon_limit.sh` | Wrapper that pins `RAYON_NUM_THREADS` |
| `benchmark_cli.py` | CLI command latency |
| `benchmark_python_bindings.py` | PyO3 binding overhead |

---

## Adding a new benchmark

Don't add it here. New benchmarks belong in
`benchmarks/comprehensive/benchmarks/`, registered via
`comprehensive/benchmarks/__init__.py::ALL_BENCHMARKS`, so they land in
the gate automatically. See `comprehensive/benchmarks/accel_pca.py` for
the reference pattern. The exception is dataset prep, which stays here
because it is intentionally outside the per-triple gate scheduler.
