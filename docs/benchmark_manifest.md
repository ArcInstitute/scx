# Benchmark Manifest Format

Every performance claim in `README.md` and `docs/performance.md` must be
backed by a **benchmark manifest entry** — a JSON result file checked into
`benchmarks/comprehensive/results/` (either under `raw/` or promoted into
`baselines/LATEST/summary.json`).

## Scope: which claims this covers

The manifest schema is a `benchmark` × `format` × `dataset` triple produced by
the SLURM comprehensive suite (`benchmarks/comprehensive/`). That is the right
shape for a claim of the form "SCX reads `census_1m` N× faster than h5ad", and
it is what the rule above governs. **Those claims must be manifested. No
exceptions.**

It is not a shape a `cargo bench` kernel microbenchmark can take — there is no
format and no dataset, only a criterion id and a machine. `docs/performance.md`
carries such numbers (covariance PCA, the pairwise distance kernels), and
inventing a synthetic triple for them would put un-reproducible rows into the
baseline that `gate_candidate.py` compares against. So they are **disclosed
inline instead**, and a disclosure is only adequate if it carries all of:

- the exact command, including the criterion filter;
- the machine (partition, core count) and the date;
- the commit or branch point each arm was measured at;
- an explicit `[!NOTE]` saying it is a microbenchmark with no manifest entry, so
  a reader never mistakes it for a captured result.

If a number could be expressed as a triple, it must be — this tier is for
kernel-level measurements that genuinely cannot, not a way around the capture
harness.

> [!IMPORTANT]
> `benchmarks/scripts/check_readme_manifests.py` enforces only the first tier,
> and only over `README.md` — it does not parse `docs/performance.md` at all.
> The second tier is a review-time convention, not a checked one.

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

### CPU stage profile (`extra["cpu_profile_*"]`)

The `extra` dict is the free-form per-run channel. When the CPU per-stage
profiler is enabled (`SCX_CPU_PROFILE=1` exported **before** the worker imports
pyscx), the accel/read benchmark modules (`accel_pca`, `accel_hvg`,
`accel_preprocess`, `accel_de`, `accel_format_pipeline`) record a
decode / I-O / reduction / marshalling breakdown of the timed op into `extra`
via `runners.accel_runner.cpu_profile_capture`. This is the Phase-2 **ranking
oracle** — it tells you whether a streaming CPU accelerator op is decode/I-O-bound
or reduction/marshalling-bound at a given scale, so no `×` speedup claim is
trusted without it. Keys (all `float`; absent when the profiler is disabled):

| Key | Meaning |
|-----|---------|
| `cpu_profile_enabled` | `1.0` iff the profiler was active for the run. |
| `cpu_profile_io_ms` / `_count` / `_bytes` | Raw shard byte-fetch (mmap slice / cloud range read). ~0 for a warm local mmap (page-fault cost lands in `decode`). |
| `cpu_profile_decode_scx1_ms` / `_count` / `_bytes` | Host decode of Scx1 shards. |
| `cpu_profile_decode_generic_ms` / `_count` / `_bytes` | Host decode of None/Zstd/Lz4Shuffle/Pcodec/ShufDeltaZstd shards. |
| `cpu_profile_reduction_ms` / `_count` | Per-shard kernel accumulation (disjoint from decode; includes the DE ranking pass). Its `_bytes` is always `0` — **not tracked** (no natural byte count for a kernel accumulation), not "zero bytes processed". |
| `cpu_profile_marshalling_ms` / `_count` / `_bytes` | Python result assembly (`Vec<Vec<f32>>` → `PyArray2`). |

The buckets are disjoint, so `io + decode + reduction + marshalling ≈ wall` for a
streaming op. The Rust counters live in `scx_format_io::profile` (mirroring the
GPU-path `scx_gpu::profile` / `SCX_GPU_PROFILE`) and are surfaced to Python via
`pyscx.accel.cpu_profile_snapshot()` / `cpu_profile_reset()`. **Known gaps** (not
captured in the `decode` bucket): the *framed/block-index* decode route
(`decode_block_index_row_runs`) and the typed-dtype path
(`read_shard_from_entry_native`) — a *full-shard* CSR **or CSC** decode routes
through the central hook and **is** captured; and the cloud range-read readers that
bypass the local `ScxReader` (current captures are local mmap only).

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
