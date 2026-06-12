# Installation reference (pyscx)

Self-contained guide for getting `pyscx` installed and working. Most install
pain comes from mixing up **end-user** vs **developer** installs, missing
**optional features**, or building against the wrong Python.

---

## Two install paths — pick one

| Who | What you want | How |
|---|---|---|
| **End user** | Use pyscx on data you already have | Download `.whl` from [GitHub Releases](https://github.com/ArcInstitute/scx/releases) (pyscx-v* tags), then `pip install ./pyscx-*.whl` |
| **Developer** | Hack on the Rust/Python repo | Clone + `maturin develop` from `pyscx/` |

Do **not** assume a git checkout works like a wheel install. A fresh clone
has no compiled extension until you run `maturin develop`.

---

## End user: pre-built wheel from GitHub Releases

pyscx is **not** published on PyPI. The `pypi.org/project/pyscx/` package is
**unrelated**. Pre-built wheels are attached to GitHub Releases tagged `pyscx-v*`
at <https://github.com/ArcInstitute/scx/releases>. Wheels are Linux x86_64 and
aarch64, Python 3.11–3.14.

```bash
python -m venv .venv
source .venv/bin/activate   # or: . .venv/bin/activate
# Download the wheel for your Python version + architecture from GitHub Releases:
# https://github.com/ArcInstitute/scx/releases (look for pyscx-v* tags).
# Replace <version> with the release you downloaded (e.g. 0.7.1).
pip install ./pyscx-<version>-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
python -c "import pyscx; print(pyscx.__version__)"
```

**What pre-built wheels include** (see `.github/workflows/pyscx-release.yml`):

- **hdf5-static** — libhdf5 bundled; `from_h5ad` / `to_h5ad` / `from_h5mu` /
  `to_h5mu` work without system HDF5 libraries.
- **cloud** (Rust) — `open_cloud`, cloud query pipeline, and related native
  cloud I/O are compiled in.
- **Platform** — manylinux x86_64 and aarch64 (Linux). Python 3.11–3.14.
- **Not included** — GPU acceleration. For `device="gpu"` you must build from
  source with `--features gpu` (see below).

**Optional Python extras** (install only what you need). Extras must attach to
the **resolved** wheel filename — `pip` does not expand globs, and quoting a
`*` pattern (needed so the shell leaves `[extra]` alone) blocks shell expansion
too, so `'./pyscx-*.whl[cloud]'` reaches pip verbatim and fails. Wrap the glob
in `$(ls ...)` so the shell resolves it to a single filename before appending
the extra (assumes exactly one matching wheel in the directory):

```bash
pip install "$(ls ./pyscx-*.whl)[cloud]"      # boto3, google-cloud-storage, azure-storage-blob
pip install "$(ls ./pyscx-*.whl)[gpu]"        # cupy-cuda12x (Linux); needs a GPU-enabled build
pip install "$(ls ./pyscx-*.whl)[mudata]"     # in-memory MuData round-trip (from_mudata / to_mudata)
pip install "$(ls ./pyscx-*.whl)[10x]"        # from_10x (pulls scanpy)
pip install "$(ls ./pyscx-*.whl)[eval]"       # polars for pyscx.eval helpers
pip install "$(ls ./pyscx-*.whl)[scvi]"       # scvi-tools integration helpers
pip install "$(ls ./pyscx-*.whl)[cloud,gpu]"  # combine as needed
```

Or just spell out the full filename, e.g.
`pip install './pyscx-0.7.0-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl[cloud]'`.

Notes on extras:

- **`[cloud]`** adds Python SDK packages (per `docs/compatibility-matrix.md`).
  Cloud credentials for `open_cloud()` still come from the usual env vars /
  instance metadata (`docs/cloud.md`) — SCX does not ship custom auth code.
- **`[gpu]`** installs CuPy but **does not** magically enable GPU kernels.
  Pre-built wheels are CPU-only; you still need a source build with
  `maturin develop --features gpu`.
- **`[mudata]`** is for in-memory `mudata.MuData` objects (`from_mudata`,
  `Experiment.to_mudata()`). File-based h5mu ingest/export (`from_h5mu`,
  `to_h5mu`) is covered by the base wheel's HDF5 support — no `[mudata]`
  extra required for path-based h5mu workflows.

**Python version:** `>=3.11` (`requires-python` in `pyproject.toml`).
`docs/development.md` mentions Python 3.10+ for the wider workspace, but
**pyscx itself requires 3.11+**. Pin numpy/scipy/anndata inside the declared
bounds in `docs/compatibility-matrix.md` if you hit resolver conflicts.

---

## Developer: build from the repo

Use the repo-root `.venv/` and **never** mix system pip with the project venv
(`docs/development.md`).

```bash
# From repository root
uv venv .venv

# Dev tools + runtime deps. [dev] pulls maturin[patchelf], pytest, scanpy,
# python-dotenv. For a typical scanpy workflow also install leidenalg etc.
# (development.md lists the full set).
uv pip install -e "./pyscx[dev]"
uv pip install scikit-learn leidenalg   # not in [dev], needed for many recipes

# patchelf must be on PATH for maturin to silence the rpath warning
export PATH="$(pwd)/.venv/bin:$PATH"

# Build the extension (default maturin features include hdf5)
cd pyscx && ../.venv/bin/maturin develop --release && cd ..

python -c "import pyscx; import pyscx.pyscx; print(pyscx.pyscx.__file__)"
```

Why `[dev]` and `PATH`? `[dev]` includes `maturin[patchelf]`, which bundles a
`patchelf` binary so `maturin develop` can set the extension rpath. Without it
you get a harmless but noisy **"Failed to set rpath"** warning on every build.
Exporting `.venv/bin` on `PATH` lets maturin find that binary without
activating the venv.

**System dependency for h5ad ingest (source builds only):**

```bash
# Ubuntu/Debian
sudo apt-get install -y libhdf5-dev
```

Default `[tool.maturin] features` is `["pyo3/extension-module", "hdf5"]`, so a
normal `maturin develop` enables the h5ad/h5mu entry points. Pre-built wheels use
`hdf5-static` instead (no system lib needed).

**Feature builds from source:**

```bash
cd pyscx
../.venv/bin/maturin develop --release                      # default: hdf5 on
../.venv/bin/maturin develop --release --features cloud     # + cloud I/O (NOT in dev default)
../.venv/bin/maturin develop --release --features gpu       # + CUDA accel (needs toolkit)
../.venv/bin/maturin develop --release --features cloud,gpu # full stack
../.venv/bin/maturin develop --no-default-features \
    --features pyo3/extension-module                        # no hdf5 (from_h5ad raises)
```

Cloud is compiled into pre-built wheels but **not** into the default dev build —
source installs that need `open_cloud()` must pass `--features cloud`.

GPU builds need a CUDA toolkit (≥ 12.0) and a compatible driver. For kNN/Leiden
GPU paths you also want RAPIDS (cuVS, cuGraph). Conda is usually easier than pip
for the full GPU stack — see `docs/gpu-setup.md`.

---

## Verify the install

Run these after install or rebuild:

```python
import pyscx
import pyscx.pyscx as native

print("version:", pyscx.__version__)
print("native ext:", native.__file__)       # should end in .so (Linux) or .pyd (Windows)
print("hdf5 ingest:", hasattr(pyscx, "from_h5ad"))
print("cloud I/O:", hasattr(pyscx, "open_cloud"))   # False if built without cloud feature
from pyscx import accel
print("gpu build:", accel.gpu_available())  # False is OK on CPU-only builds
print("gpu info:", accel.gpu_info())        # None = no GPU device or CPU-only build
```

`import pyscx` alone is a good smoke test — it imports the native `pyscx.pyscx`
module and fails immediately if the extension was never built.

Quick functional smoke (needs a `.scx` file or h5ad to convert first):

```python
import pyscx
exp = pyscx.open("some.scx")
adata = exp.to_anndata()
print(adata.shape)
```

---

## Common problems

### `ModuleNotFoundError: No module named 'pyscx'`

- End user: download the wheel from [GitHub Releases](https://github.com/ArcInstitute/scx/releases) and run `pip install ./pyscx-*.whl` in the env you actually use (`which python`).
- Developer: you skipped `maturin develop`. Run it from `pyscx/` and retry.
- Wrong platform: there are no macOS/Windows pyscx wheels today — build from
  source on those platforms.

### `ImportError: ... pyscx.pyscx ...` / missing `.so`

- Extension was built for a different Python than the one running. Rebuild with
  that Python's venv: `cd pyscx && ../.venv/bin/maturin develop`.
- After switching branches or pulling big Rust changes, rebuild — stale `.so`
  files cause obscure import or segfault errors.

### `NotImplementedError: pyscx.from_h5ad requires the hdf5 feature`

Your build lacks HDF5 support.

- **Pre-built wheel:** reinstall from [GitHub Releases](https://github.com/ArcInstitute/scx/releases) (`pip install --force-reinstall ./pyscx-*.whl`).
- **Source:** install `libhdf5-dev`, then `maturin develop` (default features).
- **Intentional no-hdf5 build:** use the `scx` CLI or open existing `.scx`
  files instead of `from_h5ad`.

### `AttributeError: module 'pyscx' has no attribute 'open_cloud'`

Built without the cloud Rust feature. On source installs run
`maturin develop --features cloud`. Pre-built wheels include cloud by default.

### "Failed to set rpath for libpyscx.so"

Non-fatal. Fix for dev builds:

```bash
uv pip install 'maturin[patchelf]'
export PATH="$(pwd)/.venv/bin:$PATH"
cd pyscx && ../.venv/bin/maturin develop --release
```

Or install the dev extra: `uv pip install -e "./pyscx[dev]"`.

Do **not** add `maturin` to runtime dependencies in `pyproject.toml` — it
belongs in the `[dev]` extra only.

### `maturin` refuses to run / venv conflict

Maturin errors when **both** `VIRTUAL_ENV` and `CONDA_PREFIX` are set. Unset
one before building:

```bash
unset VIRTUAL_ENV    # when using conda for the build
# or deactivate conda when using .venv/
```

### GPU ops feel slow / `gpu_available()` is False

- **Pre-built wheel:** GPU kernels are not shipped — rebuild from source with
  `maturin develop --release --features gpu`.
- **Source without GPU feature:** same rebuild with `--features gpu`.
- **GPU build but no CUDA device:** ops **silently fall back to CPU**. Check
  `nvidia-smi` and `pyscx.accel.gpu_info()`.
- **Missing cuVS/cuGraph:** kNN and Leiden fall back to CPU even with GPU
  PCA/UMAP. See `docs/gpu-setup.md` (conda path recommended).
- Pin `device="cpu"` on Leiden when label stability matters — GPU Leiden
  differs from `leidenalg` by design.

### Missing APIs after installing the pre-built wheel

| Need | Install |
|---|---|
| h5ad / h5mu file I/O | Base pre-built wheel (hdf5-static) |
| Cloud URLs (`open_cloud`) | Base wheel (cloud compiled in); `[cloud]` for Python SDK extras |
| In-memory MuData (`from_mudata`, `to_mudata`) | `pip install "$(ls ./pyscx-*.whl)[mudata]"` |
| 10x HDF5 (`from_10x`) | `pip install "$(ls ./pyscx-*.whl)[10x]"` (scanpy) |
| GPU accelerators | Source build with `--features gpu` + `pip install "$(ls ./pyscx-*.whl)[gpu]"` (cupy) |
| cell-eval parity helpers | `pip install "$(ls ./pyscx-*.whl)[eval]"` (polars) |

### Dependency version conflicts

`pyscx` pins upper bounds on numpy, scipy, pyarrow, anndata. If pip/uv
complains, create a fresh venv, install pyscx first, then add scanpy/scvi on
top. See `docs/compatibility-matrix.md` for tested combos.

### `scx` CLI not found

The CLI is a separate Rust binary — pyscx Python APIs work without it.

**Pre-built binaries (recommended on Linux)** — published on each `scx-cli-v*`
tag at [GitHub Releases](https://github.com/ArcInstitute/scx/releases). Linux
x86_64 and arm64, glibc ≥ 2.35. Bundles hdf5 + cloud; libhdf5 statically linked.

```bash
# Set VERSION to the latest release — see the Releases page above.
VERSION=0.7.1
TARGET=x86_64-unknown-linux-gnu   # or: aarch64-unknown-linux-gnu
gh release download "scx-cli-v${VERSION}" -R ArcInstitute/scx \
  -p "scx-cli-${VERSION}-${TARGET}.tar.gz"
tar xzf "scx-cli-${VERSION}-${TARGET}.tar.gz"
install -m 0755 "scx-cli-${VERSION}-${TARGET}/scx" ~/.local/bin/scx
# See README.md for the curl-based download (once the repo is public).
```

**Build from source** (macOS, custom features, etc.):

```bash
cargo build -p scx-cli --release --features hdf5,cloud
# hdf5 conversion needs libhdf5-dev at build time unless using hdf5-static

# Or install from a clone onto PATH (the crate is not on crates.io, so plain
# `cargo install scx-cli` does not work — install from the local checkout):
cargo install --path scx-cli --features default-bin
```

`default-bin` bundles h5ad/h5mu/10x conversion; without it the CLI omits h5ad
conversion (SCX → SCX ops only).

---

## Rebuild checklist (when in doubt)

1. Confirm Python: `which python` → should be `.venv/bin/python` (3.11+).
2. Reinstall dev tools: `uv pip install -e "./pyscx[dev]"`.
3. `export PATH="$(pwd)/.venv/bin:$PATH"`.
4. `cd pyscx && ../.venv/bin/maturin develop --release` (+ `--features` as needed).
5. Verify: `python -c "import pyscx.pyscx; from pyscx import accel; print(accel.gpu_available())"`.

For the full maintainer build matrix (R bindings, fuzz, CI parity), see
`docs/development.md`. For GPU conda/container/SLURM setup, see
`docs/gpu-setup.md`.
