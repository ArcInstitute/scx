# benchmarks/scripts — DEPRECATED WRAPPERS

> **The scripts under this directory are legacy wrappers retained for
> muscle-memory compatibility.** New benchmark work lands under
> `benchmarks/comprehensive/scripts/` only. See Phase I.1 of the roadmap.

## What lives here

Three categories of files, all deprecated:

### `benchmark_*.py` — one-off Python entrypoints

Single-benchmark Python scripts from the pre-comprehensive era. The
comprehensive harness (`benchmarks/comprehensive/scripts/run_parallel.py`)
supersedes every one of them — it runs the same benchmarks with
per-triple SLURM submission, uniform JSON output, and the regression
gate. `benchmark_cloud.py` is already a thin shim forwarding to the
comprehensive launcher; other `benchmark_*.py` scripts retain their
original shape.

### `slurm_*.sh` — sequential single-job wrappers

Wrap one or more `benchmark_*.py` scripts into a single SLURM job.
Sequential and not retried per-triple, so they occupy a full node for
the slowest benchmark even when most have already completed. Replace
by the submitit one-job-per-triple launcher:

| Legacy wrapper | Replacement |
|---|---|
| `slurm_phase3_small.sh` / `slurm_phase3_large.sh` | `run_parallel.py --datasets …` |
| `slurm_phase3_parallel_*.sh` | `run_parallel.py --benchmarks parallel_scaling parallel_write_scaling …` |
| `slurm_parallel_write_scaling.sh` | `run_parallel.py --benchmarks parallel_write_scaling …` |
| `slurm_phase0_baseline*.sh` | `capture_baseline.py --tier small` |
| `slurm_phase4_ml_loader.sh` | `run_parallel.py --benchmarks ml_loader …` |
| `slurm_phase3c_pcodec*.sh` | `run_parallel.py --formats scx_pcodec …` |
| `slurm_phase8[bc]_leiden_bench.sh` | direct `pyscx.accel.leiden` invocation |
| `slurm_phase9_pipeline_bench.sh` | `run_parallel.py --benchmarks memory` |
| `slurm_gpu_*_bench.sh`, `slurm_harmony_bench.sh` | standalone acceleration benchmarks — these are active dev surfaces and will remain until a unified acceleration launcher lands |
| `slurm_build_census_*.sh` / `slurm_prep_datasets.sh` | dataset prep; kept as-is (not a benchmark run) |

### `bench_env.py`

Used by the comprehensive harness — NOT deprecated. Sets
`DATA_DIR` / `SCX_WORK_DIR` conventions.

## Retirement plan

Scripts tagged "deprecated" in the table above will be deleted one
release after Phase 5 ships. Please migrate to the comprehensive
launcher if you have active workflows using them:

```bash
# Instead of: sbatch benchmarks/scripts/slurm_phase3_small.sh
python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k pbmc10k smartseq2 tabula_sapiens_100k
```

For regression gating, use the on-demand wrapper:

```bash
bash benchmarks/comprehensive/scripts/gate_candidate.sh
```

See `benchmarks/README.md` for the full workflow docs.
