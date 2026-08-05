# Development Guide

This guide covers how to build, test, and develop SCX across all supported
configurations. For project architecture and coding conventions, see
[architecture.md](architecture.md) and [conventions.md](conventions.md).

## Prerequisites

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Rust toolchain | stable ≥ 1.78 | Core workspace build |
| Python | 3.11+ | pyscx bindings, benchmarks, validation |
| uv | latest | Python virtualenv management (`.venv/`) |
| R | 4.2+ (optional) | rscx bindings |
| CUDA Toolkit | 12.0+ (optional) | GPU kernel compilation for scx-gpu |

Install Rust via [rustup](https://rustup.rs/):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    --default-toolchain stable
```

## Build Configurations

### CPU-only build (default)

The default workspace build compiles all crates — including `scx-gpu` — on
CPU-only machines. `cudarc` (the CUDA binding crate) compiles without a CUDA
toolkit installed; GPU functionality is only exercised at runtime. GPU tests
gracefully skip via the `require_gpu!()` macro when no CUDA device is present.

```bash
# Build all crates except rscx (which requires the R toolchain).
# Note: --workspace includes all `members`, not just `default-members`,
# so --exclude rscx is needed on machines without R.
cargo build --workspace --exclude rscx

# Run all tests
cargo test --workspace --exclude rscx

# Lint
cargo clippy --workspace --exclude rscx -- -D warnings

# Format check
cargo fmt --check
```

`rscx` is excluded from `default-members` because it requires the R toolchain
and extendr. Build it explicitly with `cargo build -p rscx` (see
[R bindings](#r-binding-build-rscx) below).

### HDF5 feature build

The `hdf5` feature on `scx-cli` enables `scx convert --from h5ad` using the
Rust `hdf5` crate (unmaintained upstream — fallback is a Python subprocess
via h5py). Requires `libhdf5-dev` system headers.

```bash
# Install system dependency (Ubuntu/Debian)
sudo apt-get install -y libhdf5-dev

# Build with HDF5 support
cargo build -p scx-cli --features hdf5

# Build with statically-linked HDF5 (no system libhdf5 required at runtime;
# pulls in CMake at build time)
cargo build -p scx-cli --features hdf5-static
```

**Parallel streaming reader and libhdf5 thread-safety.** The parallel
streaming convert path (`scx convert --reader-threads N`, `pyscx.from_h5ad(...,
reader_threads=N)`, both with `N > 1`; default `None` auto-resolves to
`RAYON_NUM_THREADS` or CPU count) requires a libhdf5 build with
`--enable-threadsafe`. Conda-forge `hdf5=1.12.*=nompi*` and the Ubuntu
system package ship this option by default. The runtime probe
(`H5is_library_threadsafe`) is cached in a `OnceLock`; non-threadsafe
builds fall back to the sequential coordinator with a one-shot
`Hdf5NotThreadsafe` warning. Output is byte-identical on both paths.

### Cloud feature build

The `cloud` feature enables `object_store`-backed GCS/S3/Azure I/O in
`scx-cli` and `pyscx`. No special system libraries are required.

```bash
# Build CLI with cloud support
cargo build -p scx-cli --features cloud

# Build and test with cloud features workspace-wide
cargo test --workspace --features cloud
```

### GPU feature build

GPU support compiles CUDA kernels to PTX via `nvcc` and enables cuSPARSE,
cuSOLVER, and cuBLAS bindings. See [gpu-setup.md](gpu-setup.md) for full
environment setup.

The GPU feature gate is layered:
- `scx-gpu` depends unconditionally on `cudarc` but compiles on CPU-only
  machines (cudarc is build-safe without CUDA)
- `scx-accel` has an opt-in `gpu` feature that pulls in `scx-gpu`
- `pyscx` has an opt-in `gpu` feature that enables `scx-accel/gpu`

```bash
# Check that GPU crate compiles (works on CPU-only machines)
cargo check -p scx-gpu

# Build scx-accel with GPU support (requires CUDA toolkit for full functionality)
cargo build -p scx-accel --features gpu

# Run GPU tests (skips gracefully if no CUDA device is present)
cargo test -p scx-gpu --features bench
```

### Python editable install (pyscx)

**Always use the uv venv** at `.venv/` for all Python work. Do not use system
Python or pip directly.

```bash
# Create the venv (one-time)
uv venv .venv

# Install Python dependencies
# `maturin[patchelf]` bundles the `patchelf` binary; without it, every
# `maturin develop` prints "Failed to set rpath for libpyscx.so" as a
# non-fatal warning.
uv pip install 'maturin[patchelf]' pytest numpy scipy pyarrow anndata scanpy \
    scikit-learn leidenalg python-dotenv

# Put the venv's bin/ on PATH so `maturin develop` (invoked below via the
# explicit `.venv/bin/maturin` path) can find the `patchelf` binary that
# `maturin[patchelf]` just installed. Alternative: `source .venv/bin/activate`
# before any maturin call. Without this, maturin's subprocess for patchelf
# fails and prints the harmless rpath warning.
export PATH="$(pwd)/.venv/bin:$PATH"

# Build and install pyscx in development mode. The pyscx `[tool.maturin]`
# default feature set includes `hdf5`, so `pyscx.from_h5ad` / `to_h5ad`
# / `from_h5mu` / `to_h5mu` are available out of the box (requires
# `libhdf5-dev` system headers).
#
# IMPORTANT: maturin's `--features` REPLACES this default set — it is NOT
# additive. Whenever you pass `--features`, re-list `hdf5` explicitly
# (e.g. `--features hdf5,gpu`), or the four h5ad/h5mu wrappers will raise
# `NotImplementedError`. Because the editable `.so` is shared across every
# env pointing at this checkout, an hdf5-less rebuild in one env silently
# disables h5ad I/O in all of them.
cd pyscx && ../.venv/bin/maturin develop && cd ..

# CPU-only / no-libhdf5 build (the four h5ad/h5mu wrappers still import
# but raise `NotImplementedError` at call time).
cd pyscx && ../.venv/bin/maturin develop --no-default-features \
    --features pyo3/extension-module && cd ..

# With cloud support
cd pyscx && ../.venv/bin/maturin develop --features hdf5,cloud && cd ..

# With GPU support (requires CUDA toolkit)
cd pyscx && ../.venv/bin/maturin develop --features hdf5,gpu && cd ..

# With both
cd pyscx && ../.venv/bin/maturin develop --features hdf5,cloud,gpu && cd ..

# Run Python tests
cd pyscx && ../.venv/bin/pytest tests/ -v && cd ..
```

### R binding build (rscx)

Requires R ≥ 4.2, the `Matrix` package, and a Rust toolchain. `rscx` uses
[extendr](https://extendr.github.io/) (v0.8.x) for Rust ↔ R FFI.

```bash
# Install R dependencies
Rscript -e 'install.packages(c("Matrix", "testthat"), repos="https://cloud.r-project.org")'

# Build and install
cd rscx && R CMD INSTALL . && cd ..

# Run R tests
cd rscx && Rscript -e 'testthat::test_dir("tests/testthat")' && cd ..
```

> **Conda R:** if R came from conda, build from an **activated** env
> (`conda activate <env>`), or put `$CONDA_PREFIX/bin` on `PATH`. R's `Makeconf`
> references the conda C compiler (e.g. `x86_64-conda-linux-gnu-cc`), which is
> only on `PATH` when the env is active — otherwise the link step fails with
> `x86_64-conda-linux-gnu-cc: not found`. Invoking the interpreter by full path
> (`<env>/bin/R CMD INSTALL rscx/`) without activating is **not** enough.

`rscx` is excluded from `default-members` because:
- The R toolchain and extendr are not always available (CI, Rust-only dev)
- Build it explicitly: `cargo build -p rscx`

## Test Matrix

### Rust tests

`cargo test --workspace --exclude rscx` covers all crates except `rscx`. Key test locations:

| Crate | Location | What it covers |
|-------|----------|----------------|
| `scx-format` | `tests/integration.rs` | Round-trip, checksums, minimum file size |
| `scx-codec` | Per-module unit tests | All codec encode/decode round-trips |
| `scx-sparse` | Unit tests | CSR construction, slicing, dense conversion |
| `scx-ops` | `tests/` | Append, delete, compact, rollback, merge, flock |
| `scx-engine` | Unit tests | Predicate parsing, pipeline, fused ops |
| `scx-loader` | Unit tests | Pipeline lifecycle, batch format, normalize |
| `scx-cloud` | `tests/` | Explode/pack, cloud-optimize, pull/push |
| `scx-accel` | Unit tests | PCA, kNN, UMAP, DE, pseudobulk |
| `scx-gpu` | Unit tests | CUDA decode, SpMM, GPU PCA/UMAP/preprocess |
| `scx-integration-tests` | `tests/golden_files.rs` | Golden file regression (15 fixtures) |

### GPU test skip behavior

All GPU tests use a `require_gpu!()` macro that:
1. Attempts to create a `GpuDevice` (CUDA context + stream)
2. If initialization fails (no GPU, no driver, wrong CUDA version), returns
   `Ok(())` silently
3. Tests run normally on GPU nodes; safely skipped on CPU-only machines

### Python test suite

Located in `pyscx/tests/` (30+ test files). Run with:

```bash
cd pyscx && ../.venv/bin/maturin develop && ../.venv/bin/pytest tests/ -v
```

See [testing.md](testing.md) for the full test file inventory and correctness
validation suite.

### Feature-gated CI checks

The CI runs `cargo check` and `cargo clippy` for individual feature
combinations that are not covered by the default workspace build:

| Feature combination | What it validates |
|--------------------|-------------------|
| `scx-cli --features cloud` | Cloud-only CLI build |
| `scx-cli --features hdf5,cloud` | HDF5 + cloud CLI build |
| `scx-accel --features gpu` | GPU accelerator build (cudarc stubs, no CUDA required) |
| `pyscx --features gpu` | Python GPU bindings (rapids-absent lane) |
| `pyscx --features hdf5,gpu` | Python GPU bindings + h5ad I/O (the documented GPU build combo) |
| `pyscx --features cloud` | Python cloud bindings |

## Workspace Structure

### Crate membership

| Scope | Crates |
|-------|--------|
| **`members`** (all) | scx-format, scx-format-io, scx-codec, scx-sparse, scx-cli, scx-convert, scx-ops, scx-engine, scx-loader, scx-cloud, scx-gpu, scx-mtx, scx-accel, pyscx, rscx, tests/scx-integration-tests |
| **`default-members`** | All except `rscx` (R toolchain dependency) |

`scx-gpu` is deliberately included in `default-members` because `cudarc`
compiles without CUDA installed. This ensures that:
- `cargo build --workspace --exclude rscx` works on any machine with a Rust toolchain
- `cargo test --workspace --exclude rscx` runs all non-GPU tests everywhere
- GPU tests gracefully skip via `require_gpu!()`
- CI validates the CPU-only build contract on every PR

### Feature flags

All feature flags are opt-in, with one exception: `pyscx/hdf5` is in the
default `[tool.maturin] features` set so a bare `maturin develop`
matches the published pre-built wheel. Opt out with
`maturin develop --no-default-features --features pyo3/extension-module`.

| Crate | Feature | What it enables |
|-------|---------|----------------|
| `scx-cli` | `hdf5` | h5ad conversion via Rust hdf5 crate |
| `scx-cli` | `hdf5-static` | Statically-linked HDF5 (no system lib at runtime) |
| `scx-cli` | `cloud` | GCS/S3/Azure cloud I/O |
| `scx-gpu` | `gds` | GPUDirect Storage (requires nvidia-fs drivers) |
| `scx-accel` | `gpu` | GPU-accelerated PCA/kNN/UMAP/Leiden/preprocessing |
| `pyscx` | `hdf5` | `pyscx.from_h5ad` / `to_h5ad` / `from_h5mu` / `to_h5mu` (default-on; see above) |
| `pyscx` | `hdf5-static` | Bundles libhdf5 + zlib statically (wheel builds) |
| `pyscx` | `cloud` | Python cloud operations |
| `pyscx` | `gpu` | Python GPU accelerators |

## Fuzzing

Three fuzz target suites exist, all using `cargo-fuzz` + `libfuzzer-sys`. Each lives in its own cargo workspace under `<crate>/fuzz/` (the standard `cargo-fuzz` layout), so the top-level `cargo build/test --workspace` walks past them.

### scx-codec fuzzing

Targets malformed codec bitstreams:

```bash
# Install cargo-fuzz (one-time)
cargo install --locked cargo-fuzz

cd scx-codec/fuzz
cargo fuzz list

# Run a specific target (runs indefinitely; Ctrl-C to stop)
cargo +nightly fuzz run fuzz_bitstream
cargo +nightly fuzz run fuzz_rice
cargo +nightly fuzz run fuzz_forbp
cargo +nightly fuzz run fuzz_delta_golomb
```

### scx-format fuzzing

Targets malformed file structures (headers, catalogs, shards, sidecars):

```bash
cd scx-format/fuzz
cargo fuzz list

cargo +nightly fuzz run fuzz_header
cargo +nightly fuzz run fuzz_shard
cargo +nightly fuzz run fuzz_catalog
cargo +nightly fuzz run fuzz_csc_shard
cargo +nightly fuzz run fuzz_bitmap            # Phase 5b BitmapShard
cargo +nightly fuzz run fuzz_modality_table    # Phase 9: ModalityTable
cargo +nightly fuzz run fuzz_provenance        # Phase 9: Provenance.params_json
```

### scx-engine fuzzing

Predicate-index deserialisation lives in `scx_engine::PredicateIndex::read_from`, not `scx-format`, so it gets its own fuzz crate:

```bash
cd scx-engine/fuzz
cargo fuzz list

cargo +nightly fuzz run fuzz_predicate_index   # Phase 9: categorical + numeric
```

### Bounded runs and CI

For a fixed-duration smoke test (e.g. 2 minutes) rather than indefinite fuzzing:

```bash
cd scx-format/fuzz
cargo +nightly fuzz run fuzz_modality_table -- -max_total_time=120
```

The `.github/workflows/fuzz.yml` workflow exposes both modes:

- **Build check** runs on every PR and push to `main`: `cargo +nightly fuzz build` in each of the three fuzz crates. This catches API bitrot — if a refactor breaks a fuzz target's call site, the PR fails before merging.
- **Run on demand** runs only via `workflow_dispatch`. Trigger from the GitHub Actions tab → Fuzz → Run workflow, with inputs `duration_seconds` (default 120) and an optional `target` filter (e.g. `fuzz_modality_table` to run a single target). Crash inputs are uploaded as workflow artifacts when a target fails.

> **Note**: `cargo fuzz` requires the nightly toolchain. Install with `rustup toolchain install nightly`. Corpus files accumulate in `fuzz/corpus/<target>/`; check in interesting specimens as regression fixtures.

## Benchmarks

The benchmark infrastructure lives in `benchmarks/`. See
[benchmarks/README.md](../benchmarks/README.md) for:

- Dataset preparation (D1–D8, from PBMC 3K to Census 10M cells)
- Parallel SLURM job submission via `run_parallel.py`
- Conda benchmark environments (`scx-bench`, `scx-bench-gpu`, `scx-bench-r`,
  `scx-bench-slaf`)
- Regression gating via `gate_candidate.py`
- Cloud benchmark setup (GCP)

Quick smoke test:

```bash
.venv/bin/python benchmarks/comprehensive/scripts/run_parallel.py \
    --datasets pbmc3k --benchmarks read_full --formats scx_auto h5ad_gzip
```

## CI Overview

The GitHub Actions CI (`.github/workflows/ci.yml`) runs on every push and PR
to `main`:

| Job | What it does |
|-----|-------------|
| `build-cpu-only` | `cargo build --workspace --exclude rscx` — pins the CPU-only build contract |
| `test` | `cargo test --workspace --exclude rscx` |
| `clippy` | `cargo clippy --workspace --exclude rscx -- -D warnings` |
| `fmt` | `cargo fmt --check` |
| `feature-check` | Matrix of `cargo check`/`clippy` for feature combos (cloud, hdf5, gpu) |
| `python` | maturin develop + pytest (with cloud features, fork-safety tests) |

The `build-cpu-only` job explicitly validates that the entire workspace
(including `scx-gpu`) compiles on machines without CUDA installed. This is
the CI-side guarantee that the CPU-only developer experience is never broken.
