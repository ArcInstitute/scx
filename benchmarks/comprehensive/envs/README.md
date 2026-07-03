# Benchmark conda environments

The comprehensive benchmark suite uses isolated conda envs so each runner
has its native dependency stack without cross-contamination. The
orchestrator (`run_parallel.py`, `gate_candidate.py`) activates exactly
one env per gate run via the `CONDA_PREFIX` it inherits — pick the env
whose runners you want exercised, then launch from there.

## Environments

| YAML | Env name | Use case | Created by |
|---|---|---|---|
| `scx-bench.yml` | `scx-bench` | CPU benchmarks: format runners, accelerators, lazy preprocessing, correctness, ML loader | `install_dependencies.sh` (default) |
| `scx-bench-gpu.yml` | `scx-bench-gpu` | GPU benchmarks: CUDA + RAPIDS (cuVS, cuGraph). Extends `scx-bench` with GPU-side deps. | `install_dependencies.sh --gpu` |
| `scx-bench-r.yml` | `scx-bench-r` | R / BPCells benchmarks. Isolated R interpreter; pyscx not built here. | `install_dependencies.sh --r` |
| `scx-bench-slaf.yml` | `scx-bench-slaf` | SLAF runner only — slafpy's deps conflict with the main `scx-bench` env. Active only for the `slaf_runner` row of the matrix. | manual: `conda env create -f scx-bench-slaf.yml` |
| `scx-bench-eval.yml` | `scx-bench-eval` | `cell-eval` / `arc-bench` parity validation (cell_eval_parity_perf benchmark). | `install_dependencies.sh --eval` |

Create everything at once: `install_dependencies.sh --all`.

## Shardad

The `shardad` runner (`shardad_runner`) and the `grouped_read` head-to-head
share the `scx-bench` env — shardad's deps (anndata/h5py/hdf5plugin/numpy/
scipy/pandas/pyarrow/bitshuffle/zstandard) don't conflict with the main stack,
so unlike SLAF it needs no isolated env. It's not on PyPI/conda; `scx-bench.yml`
installs it editable from the local repo:

```yaml
- -e /home/nickyoungblut/dev/python/shardad
```

That build compiles shardad's Rust core in release (needs a Rust toolchain; the
pure-Python fallback is byte-identical but ~3× slower — set
`SHARDAD_DISABLE_RUST=1` to force it). If your checkout lives elsewhere, edit the
path. Verify:

```bash
conda activate scx-bench
python -c "import shardad, shardad.v2._rust; print('shardad', shardad.__version__, 'rust-core OK')"
```

Check what's installed: `install_dependencies.sh --check` (does not modify anything).

Rebuild just pyscx without reinstalling deps: `install_dependencies.sh --rebuild` (or `--rebuild --gpu`).

## Cloud object-store dependencies (gcsfs)

The cloud benchmarks (`cloud_read`, `cloud_push`, `cloud_pull`, `cloud_filtered`,
`cloud_metadata`, `cloud_reader_vs_pull`, `cloud_large_atlas`, `cost_model`)
need `gcsfs` registered with `fsspec` so the `gs://` protocol resolves at
import time. **Both packages must come from the same channel and version
pair** — if conda's solver drops one but keeps the other, the protocol
registration silently breaks. The failure mode is downstream: every
zarr-backed cloud cell errors out with `ValueError: Please install gcsfs
to access Google Storage` even though `gcsfs` is technically importable.

The 2026-05-10 tier-full gate caught this — a worker env had `fsspec`
without `gcsfs`, surfacing as 24 confusing FAST_FAILs across the
`zarr_*` / `cloud_*` matrix. The yaml now pins:

```yaml
- fsspec=2026.2.0
- gcsfs=2026.2.0
```

(see `scx-bench.yml`). The matched-version pair preserves the protocol
registration through future env rebuilds.

To verify the registration works in your env:

```bash
conda activate scx-bench
python -c "import fsspec; fs = fsspec.filesystem('gs'); print(fs.__class__)"
# Expected: <class 'gcsfs.core.GCSFileSystem'> (or ExtendedGcsFileSystem)
```

`install_dependencies.sh --check` performs the same probe automatically
and reports `❌ fsspec gs proto` when the registration is broken.

For cloud benchmarks you also need GCP credentials reachable from the
SLURM worker — either `GOOGLE_APPLICATION_CREDENTIALS` pointing at a
service-account JSON, or `gcloud auth application-default login` for
interactive use. The orchestrator does not inject credentials; the env
hands them off via `.env` (loaded with `set -a; source .env; set +a`).

## Multiple-env captures

A single `gate_candidate.py` invocation runs from the orchestrator's own
`CONDA_PREFIX` and propagates that env to every SLURM worker. To capture
metrics that span two envs (e.g. SLAF rows + the rest of the matrix in
one snapshot), you currently have to:

1. Capture from `scx-bench`: `python gate_candidate.py ... --formats <all-non-slaf>`
2. Capture from `scx-bench-slaf` against the same candidate dir:
   `python gate_candidate.py ... --formats slaf --name <same-as-step-1>`
   — the second invocation appends its rows to `<candidate>/raw/`.

This is a known limitation and the reason SLAF floors are currently
deferred (see `thresholds.yaml` "Deferred floors" section). A future
orchestrator change could activate per-runner envs inside one capture.

## Editing a yaml

Bump pins in the yaml, then re-create:

```bash
conda env remove -n scx-bench -y
bash install_dependencies.sh             # re-reads scx-bench.yml
cd pyscx && maturin develop --release && cd ..
```

The yaml is authoritative; `--rebuild` does not re-solve the env, so a
yaml change requires a full env recreate for the new pins to land.
