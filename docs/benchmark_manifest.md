# Benchmark Manifest Format

Every performance claim in `README.md` and `docs/performance.md` must be
backed by a **benchmark manifest entry** — a JSON result file checked into
`benchmarks/comprehensive/results/` (either under `raw/` or promoted into
`baselines/LATEST/summary.json`).

## Schema

Each raw result JSON is written by `BenchmarkResult.to_dict()` (see
[`benchmarks/comprehensive/results.py`](../benchmarks/comprehensive/results.py))
and contains the following fields:

| Field | Type | Purpose |
|-------|------|---------|
| `schema_version` | `int` | Currently 2. Readers reject mismatches. |
| `benchmark` | `str` | Benchmark type (e.g. `read_full`, `compression`, `ml_loader`). |
| `format` | `str` | Format key (e.g. `scx_auto`, `zarr_zstd`, `h5ad_gzip`). |
| `dataset` | `str` | Dataset name (e.g. `census_1m`, `pbmc3k`, `tabula_sapiens_100k`). |
| `timestamp` | `str` | ISO 8601 capture time. |
| `system.hostname` | `str` | Machine hostname. |
| `system.cpu` | `str` | CPU model string. |
| `system.cpu_cores_physical` | `int` | Physical core count. |
| `system.ram_gb` | `int` | Total RAM in GiB. |
| `system.os` | `str` | OS and kernel version. |
| `system.rust_version` | `str` | Rust compiler version. |
| `system.library_versions` | `dict` | Python package versions (anndata, scanpy, numpy, etc.). |
| `system.storage` | `dict` | Scratch device, filesystem type, NVMe presence. |
| `system.gpu` | `dict` | CUDA runtime version, nvidia-fs status. |
| `system.provenance.run_id` | `str` | ULID-style run identifier (groups results from one `run_parallel.py` invocation). |
| `system.provenance.git_sha` | `str` | Exact commit SHA. |
| `system.provenance.git_branch` | `str` | Branch name. |
| `system.provenance.git_dirty` | `bool` | Whether the worktree had uncommitted changes. |
| `system.provenance.git_describe` | `str` | `git describe --tags --always --dirty` output. |
| `system.provenance.rayon_threads` | `str` | `RAYON_NUM_THREADS` if set. |
| `system.provenance.conda_env` | `str` | Active conda environment. |
| `runs` | `list` | Per-run measurements (see below). |
| `median_wall_s` | `float` | Median wall-clock time across runs. |
| `wall_s_iqr` | `float?` | Inter-quartile range of `wall_s` (null if <3 runs). |
| `n_runs` | `int` | Number of timed runs (excludes warmup). |
| `file_size_bytes` | `int?` | On-disk file size (for compression benchmarks). |
| `metadata` | `dict` | Benchmark-specific metadata (e.g. `cold_cache`, `n_warmup`). |

Each entry in `runs`:

| Field | Type | Purpose |
|-------|------|---------|
| `wall_s` | `float` | Wall-clock seconds. |
| `user_s` | `float` | User CPU seconds. |
| `sys_s` | `float` | System CPU seconds. |
| `peak_rss_mb` | `float` | Peak resident set size in MiB. |
| `extra` | `dict` | Run-specific extra data. |

## File naming convention

Raw results follow the naming pattern:

```
{benchmark}__{format}__{dataset}.json
```

For example: `read_full__scx_auto__census_1m.json`.

## Promoted baselines

`benchmarks/comprehensive/results/baselines/LATEST/summary.json` is the
canonical baseline for regression gating. It contains rows keyed by
`{benchmark}__{format}__{dataset}` with `median_wall_s`, `wall_s_iqr`,
`n_runs`, `peak_rss_mb_median`, `file_size_bytes`, and `source_file`.

## Workflow

### For benchmark operators

1. Run the benchmark suite via `run_parallel.py` or a targeted benchmark
   script. Results are automatically written to `results/raw/`.
2. Promote a snapshot to baseline via `capture_baseline.py`.
3. Gate candidates via `gate_candidate.py`.

### For README/docs authors

Every number that appears in a user-visible document must satisfy:

1. A corresponding raw JSON result exists in `benchmarks/comprehensive/results/raw/`
   **or** the number appears in `baselines/LATEST/summary.json`.
2. The result's `system.provenance.git_sha` is an ancestor of `main`.
3. The result's `system.provenance.git_dirty` is `false` (or the dirtiness
   is documented as acceptable — e.g., benchmark-only changes).

### Automated verification

Run `benchmarks/scripts/check_readme_manifests.py` to verify that every
numeric benchmark claim in `README.md` has a backing manifest entry:

```bash
python benchmarks/scripts/check_readme_manifests.py
```

The script exits with code 0 if all claims are backed, 1 if any are
unbacked. See the script's `--help` for options.

## Release checklist item

Before every tagged release:

```
- [ ] Every README-visible benchmark number has a corresponding manifest
      entry in `benchmarks/comprehensive/results/`. Verified by:
      `python benchmarks/scripts/check_readme_manifests.py`
```
