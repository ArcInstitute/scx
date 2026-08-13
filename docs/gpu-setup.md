# GPU Setup Guide

This guide covers installing and configuring GPU acceleration for SCX.
Most GPU-accelerated analysis — PCA, kNN, UMAP, preprocessing, and extra
HVG flavors — routes to
[rapids-singlecell](https://rapids-singlecell.readthedocs.io/) as the in-VRAM
GPU compute layer. Native SCX GPU kernels remain for Leiden (cuGraph),
streaming/out-of-VRAM PCA, seurat_v3 HVG, CSC-direct / pdex DE, Harmony,
and the ML training loader. All ops are accessed through the same
`pyscx.accel.*` API with a `device="gpu"` parameter.

## Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| NVIDIA GPU | Volta (compute 7.0) | Ampere / Hopper | PTX compiled for compute_70, forward-compatible |
| NVIDIA driver | 525+ | 535+ | Must support CUDA ≥ 12.0 (`nvidia-smi` shows max CUDA version) |
| CUDA Toolkit | 12.0 | 12.2–12.6 | Provides `nvcc` for kernel compilation |
| rapids-singlecell | 0.12+ | latest | In-VRAM PCA/kNN/UMAP/preprocess — **most important GPU dep** |
| cuGraph | 25.10+ | 25.12+ | GPU Leiden clustering (optional — CPU fallback available) |
| cuVS | 25.10+ | 25.12+ | Device-resident CAGRA kNN in fused `pca_neighbors` only (optional) |
| Rust toolchain | 1.78+ | stable | For building scx-gpu crate |
| Python | 3.11+ | 3.12–3.13 | For pyscx bindings |

**What requires what:**

| SCX operation | rapids-singlecell | CUDA Toolkit (`nvcc`) | cuVS | cuGraph |
|---------------|-------------------|----------------------|------|---------|
| In-VRAM PCA (`device="gpu"`, in-memory `X`) | **required** | — | — | — |
| In-VRAM kNN / neighbors (`device="gpu"`, in-memory `X`) | **required** | — | — | — |
| In-VRAM UMAP (`device="gpu"`, in-memory `X`) | **required** | — | — | — |
| In-VRAM preprocessing (`normalize_total` / `log1p`, in-memory `X`) | **required** | — | — | — |
| In-VRAM extra HVG flavors (`seurat`, `cell_ranger`, `pearson_residuals`, `poisson_gene_selection`) | **required** | — | — | — |
| Streaming / out-of-VRAM PCA (backed / lazy `X`) | — | required | — | — |
| Native `seurat_v3` / `seurat_v3_paper` HVG | — | required | — | — |
| Streaming preprocessing (ML loader, lazy `X`) | — | required | — | — |
| Device-resident CAGRA kNN (fused `pca_neighbors` only) | — | required | required | — |
| GPU Leiden clustering | — | — | — | required |
| CSC-direct / pdex DE (`device="gpu"`) | — | required | — | — |
| Harmony (`device="gpu"`) | — | required | — | — |
| NB-GLM pseudobulk DE (`device="gpu"`) | — | required | — | — |

Operations without their required dependencies fall back to CPU automatically
with a warning — no crashes. **For most in-VRAM GPU analysis, rapids-singlecell
is the critical dependency** — without it, PCA, kNN, UMAP, and preprocessing
fall back to CPU (the native in-VRAM covariance PCA, standalone CAGRA kNN, and
CUDA-SGD UMAP kernels were removed; CAGRA survives only inside the
device-resident fused `pca_neighbors` path).

**Note on cuBLAS:** randomized PCA, GPU-resident final-embedding
multiply, CholeskyQR2, and preprocessing Gram correction depend on **cuBLAS**.
`libcublas.so` ships alongside `libcusparse.so` / `libcusolver.so` inside
every CUDA Toolkit 12.x install, so no new runtime library path or env-var
setup is required — any working `scx-gpu` env from prior releases continues
to work out of the box.

> **Build note — always keep `hdf5` in `--features`.** maturin's `--features`
> flag **replaces** the `[tool.maturin]` default feature set (which includes
> `hdf5`); it is *not* additive. Every build command below uses
> `--features hdf5,gpu` for this reason. If you drop `hdf5` (e.g.
> `--features gpu`), `pyscx.from_h5ad` / `to_h5ad` raise `NotImplementedError`
> — and because the editable `.so` is shared across every env pointing at the
> checkout, that one rebuild disables h5ad I/O for all of them.

## Option A: conda (recommended)

Conda is the easiest way to get a working GPU environment because it resolves
the full CUDA + RAPIDS dependency tree and matches library versions to your
driver automatically.

> **The GPU-built pyscx and `rapids-singlecell` must live in the *same*
> environment.** This is the most common GPU-setup mistake: people build
> `pyscx --features gpu` into a pip/uv `.venv` that has no `rapids-singlecell`,
> run on a GPU node, and get a **mostly-CPU** pipeline — in-VRAM ops
> (PCA/kNN/UMAP/preprocess) route to rapids-singlecell when present and
> *silently CPU-fall-back* (`FallbackReason::NoRapids`) when absent. Build into,
> and run from, the **one** conda env that has both. The recipe below produces
> exactly that env.

> **Already have GPU conda environments? Check them before creating a new one.**
> A suitable env may already exist. List them and inspect what they carry —
> reuse the one that already has `rapids-singlecell` rather than building another:
> ```bash
> conda env list
> conda list -n <env> | grep -iE 'rapids-singlecell|cupy|cugraph'
> conda run -n <env> python -c "import pyscx, rapids_singlecell; print('ok')"
> ```
> Because a `maturin develop` editable install drops a `pyscx.pth` pointing at
> the repo, the compiled `.so` is shared across environments — one
> `--features gpu` build is importable from every env that has the `.pth`.

```bash
# 1. Create a dedicated environment
conda create -n scx-gpu python=3.13   # pyscx requires >=3.11
conda activate scx-gpu

# 2. Install the RAPIDS stack — pin cuda-version to match your driver.
#    Check your driver's max CUDA version:  nvidia-smi
#    Driver 535.x → cuda-version=12.2
#    Driver 550.x → cuda-version=12.4
#    Driver 560.x → cuda-version=12.6
#    The <26.04 band pin is the last cuda12 RAPIDS band; 26.04+ is cuda13-only.
conda install -c rapidsai -c conda-forge \
    cuml'>=25.10,<26.04' cuvs'>=25.10,<26.04' cugraph'>=25.10,<26.04' \
    cuda-version=12.6

# 3. Install rapids-singlecell — ON A GPU NODE (>=0.12 ships CUDA kernels
#    built with -arch=native, so nvcc must see a real GPU at build time).
pip install --no-deps 'rapids-singlecell>=0.12'
pip install docrep scikit-image

# 4. Install Python dependencies
pip install maturin numpy scipy pyarrow anndata scanpy scikit-learn leidenalg

# 5. Build pyscx with GPU support — INTO this env (rapids-singlecell must be
#    importable from the same interpreter that imports pyscx)
cd pyscx && maturin develop --release --features hdf5,gpu

# 6. Verify — both halves must be present, not just the GPU build
python -c "import pyscx; print('GPU build:', pyscx.accel.gpu_available())"
python -c "import rapids_singlecell as rsc; print('rapids-singlecell:', rsc.__version__)"
```

**Why conda?** pip-installed RAPIDS packages may pull CUDA 12.9+ runtime
libraries that are incompatible with older drivers (e.g., driver 535 supports
CUDA ≤ 12.2). Conda pins `cuda-version` and resolves compatible builds for
cuVS, cuGraph, RAFT, and RMM together.

After setup, **confirm the GPU path actually engages** at runtime via the route
metadata — `adata.uns["scx_accel"][op]["route"]` should read
`rapids_singlecell_gpu` (or `gpu_csr` for native paths), not `cpu_*`. A `cpu_*`
route with `fallback_reason="NoRapids"` means pyscx and rapids-singlecell are
not in the same environment.

## Option B: system CUDA Toolkit (no RAPIDS)

> **Important:** without rapids-singlecell, **in-VRAM PCA, kNN, and UMAP all
> fall back to CPU.** The native in-VRAM kernels for those ops were removed.
> Use this option only if you need the **native streaming/out-of-VRAM GPU
> paths** (randomized PCA on backed/lazy `X`, seurat_v3 HVG, streaming
> preprocessing, Harmony, CSC-direct DE) or GPU Leiden (cuGraph, installed
> separately). For the full GPU-accelerated analysis stack, use
> [Option A](#option-a-conda-recommended) with rapids-singlecell.

### Ubuntu / Debian

```bash
# Install CUDA Toolkit 12.2 (adjust version to match your driver)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-2

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}

# Verify
nvcc --version
```

### RHEL / Rocky / CentOS

```bash
sudo dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf install cuda-toolkit-12-2

export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}
```

### Build pyscx

```bash
# Confirm nvcc is available
nvcc --version

# Create venv and build
uv venv .venv
uv pip install maturin numpy scipy pyarrow anndata
cd pyscx && ../.venv/bin/maturin develop --release --features hdf5,gpu
```

## Option C: container

For reproducible builds, CI, or environments where you cannot install system
packages.

### Docker (full RAPIDS — recommended)

The repository includes a multi-stage `Dockerfile.gpu` that compiles Rust +
CUDA kernels in a builder stage, then produces a slim runtime image with pyscx,
the `scx` CLI, and the full RAPIDS stack (cuVS, cuGraph).

```bash
# Build the image (from the repository root)
docker build -f Dockerfile.gpu -t scx-gpu .

# Run interactively with GPU access
docker run --gpus all -it scx-gpu

# Mount data and run a script
docker run --gpus all -v /data:/data scx-gpu python /data/my_analysis.py

# Use the CLI
docker run --gpus all scx-gpu scx info /data/atlas.scx
```

### Docker (CUDA-only, no RAPIDS)

For a lighter image with only the native GPU paths (streaming PCA, seurat_v3
HVG, Leiden via cuGraph, CSC-direct DE — in-VRAM PCA/kNN/UMAP fall back to CPU
without rapids), use the NVIDIA CUDA devel base image directly:

```bash
docker run --gpus all -it nvidia/cuda:12.2.2-devel-ubuntu22.04

# Inside the container:
apt update && apt install -y python3 python3-pip curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
pip install maturin numpy scipy pyarrow anndata
cd pyscx && maturin develop --release --features hdf5,gpu
```

### Apptainer / Singularity (HPC)

```bash
# Build from the SCX GPU Docker image
docker build -f Dockerfile.gpu -t scx-gpu .
apptainer build scx-gpu.sif docker-daemon://scx-gpu:latest

# Or build directly from RAPIDS base (requires manual pyscx install inside)
apptainer build scx-gpu.sif docker://rapidsai/base:24.12-cuda12.2-py3.12

# Run with GPU passthrough
apptainer exec --nv scx-gpu.sif python -c "import pyscx; print(pyscx.accel.gpu_available())"
```

## rapids-singlecell — the GPU compute layer

rapids-singlecell is the **primary GPU compute layer** for in-VRAM analysis ops.
Most in-VRAM `device="gpu"` analysis ops
delegate to rapids-singlecell rather than native SCX CUDA kernels. The surviving
native GPU paths (streaming PCA, seurat_v3 HVG, Leiden, CSC-direct DE, Harmony,
NB-GLM pseudobulk DE, streaming preprocessing, ML loader) work with the CUDA
setup above and do **not** require rapids.

rapids-singlecell is a **detected runtime dependency**, not a hard requirement:

- When rapids-singlecell **is** importable, in-VRAM PCA / kNN / UMAP /
  preprocessing / extra HVG flavors route to it; SCX hands it a GPU-resident
  matrix so there is no host round-trip.
- When it is **absent**, those ops fall back to CPU with a one-shot diagnostic
  naming this install path and `fallback_reason="no_rapids"`. SCX still imports
  and runs (CPU + native streaming + the ML loader) with rapids absent.

The lightweight native GPU bits (cudarc + cuVS-C: loader, streaming, the cuPy
device handoff) ship under `pyscx[gpu]` with **no** RAPIDS dependency.

### Install

A dedicated, slimmer environment spec is published at
[`benchmarks/comprehensive/envs/scx-gpu-analysis.yml`](../benchmarks/comprehensive/envs/scx-gpu-analysis.yml):

```bash
conda env create -f benchmarks/comprehensive/envs/scx-gpu-analysis.yml
conda activate scx-gpu-analysis
cd pyscx && maturin develop --release --features hdf5,gpu && cd ..
```

For most users the simplest route is the conda-forge package, as in
[Option A](#option-a-conda-recommended) above —
`conda install -c rapidsai -c conda-forge rapids-singlecell` pulls a pre-built
backend that needs no GPU at install time. The slim benchmark spec below instead
leaves rapids-singlecell **unpinned** and installs it **manually on a GPU node**,
because that path resolves it against a specific pinned RAPIDS/CUDA-arch stack
(`>=0.12` source builds with `-arch=native` need a GPU visible so nvcc resolves
`sm_90` on H100):

```bash
# Inside an sbatch/srun GPU allocation, in the scx-gpu-analysis env,
# isolated from any .venv whose pip RAPIDS libs would shadow conda's:
pip install --no-deps 'rapids-singlecell>=0.12'
pip install docrep scikit-image
```

**Packaging verdict (Phase 0.3):** the `-arch=native` GPU-node build is the
practical ceiling today — a fully reproducible conda/wheel resolution of
rapids-singlecell is not available, so SCX treats it as a detected runtime
dependency with a documented manual install rather than declaring a brittle hard
extra. The rapids-absent path is exercised by a dedicated CI lane.

### What routes to rapids (in-VRAM `device="gpu"`)

With rapids present, in-memory (≤VRAM) `device="gpu"` ops route to rapids:
`pca`, `neighbors`, `umap`, `normalize_total`/`log1p`, the fused
`pca_neighbors`/`pca_neighbors_umap`, and the HVG flavors SCX has no native GPU
kernel for (`seurat`, `cell_ranger`, `pearson_residuals`,
`poisson_gene_selection`). The route is recorded as `rapids_singlecell_gpu` on
`adata.uns["scx_accel"][<op>]` with the detected rapids/cuML/cuPy versions and a
`transfer_mode`: `scx_device_handoff` when `X` is already device-resident — e.g.
from `pyscx.open(...).to_gpu_anndata()`, regardless of which mode that call itself
stamped — or `anndata_to_gpu` when rapids uploads a host `X`. (`to_gpu_anndata`
stamps its *own*, more specific transfer mode on the `to_gpu_anndata` op key —
`scx_device_decode_gpu` for a fully-in-VRAM Scx1 sidecar decode (incl. dense
≥128-nnz rows via the BitPacker4x kernel), or `scx_device_handoff_streamed`
when a non-Scx1 / sidecar-less shard host-bounces; see
[scanpy.md § Data layout for fast GPU decode](scanpy.md#data-layout-for-fast-gpu-decode-to_gpu_anndata--device-resident-analysis).)

**`X` residency contract.** When the op uploads a host `X` (`anndata_to_gpu`),
the result slots (`obsm`/`obsp`) **and** `X` are brought back to host afterwards
and the device buffers freed — so `pyscx.accel.<op>(adata, device="gpu")` on an
in-memory AnnData leaves `adata.X` host-resident, exactly as the native path
does. When `X` arrives already device-resident (`scx_device_handoff`, from
`to_gpu_anndata()`), it is **left on the GPU** so a chain of `pyscx.accel.*`
calls runs without re-uploading — that path is the way to keep data GPU-resident
across ops.

**Lay out data so the handoff is decode-light.** The device handoff's cost is
dominated by per-shard decode, which is set by the on-disk codec. Prefer storing
raw integer counts (Scx1 codec, decode-sidecar-accelerated) and deriving
`normalize_total` / `log1p` in VRAM, rather than persisting a log-normalized
float `X` (Pcodec → host decode, no sidecar). See
[scanpy.md § Data layout for fast GPU decode](scanpy.md#data-layout-for-fast-gpu-decode-to_gpu_anndata--device-resident-analysis).

These stay **native** (SCX wins or is structurally unique): `seurat_v3` /
`seurat_v3_paper` HVG, Leiden, CSC-direct / pdex DE, Harmony, NB-GLM pseudobulk
DE, and every **out-of-VRAM** path — a backed/lazy `X`
(`ScxBackedSparseDataset` / `ScxLazyTransformedDataset`) always uses SCX's
native streaming kernels, never rapids (which would OOM).

### Forcing the native GPU kernels

`SCX_FORCE_NATIVE_GPU=1` keeps the **surviving** native SCX GPU kernels for
in-VRAM ops instead of routing to rapids. After the native kernel
removals, these are the streaming preprocess kernels and randomized PCA; the
in-VRAM native UMAP, covariance PCA, and CAGRA kNN kernels were deleted (rapids
supersedes them), so for those ops the override now falls through to CPU. When
rapids is absent and this override is **not** set, in-VRAM GPU-analysis ops fall
back to CPU (`fallback_reason="no_rapids"`) with a one-shot `UserWarning`.

`SCX_DISABLE_RAPIDS=1` forces that rapids-absent CPU fallback **even on a host
where rapids is installed** — it makes the dispatcher treat rapids as
unimportable. It exists so the no-rapids fallback contract can be exercised
(tests / the Phase 2 benchmark gate) without uninstalling rapids; it takes
precedence over `SCX_FORCE_NATIVE_GPU`.

See [architecture.md § Environment variables](architecture.md#environment-variables)
for the full canonical list of `SCX_*` variables.

## SLURM / HPC configuration

On HPC clusters, GPU nodes typically require module loads or conda activation
before `nvcc` and CUDA libraries are available.

### Using conda on SLURM

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Activate conda env (avoid `conda activate` in non-interactive shells)
export CONDA_PREFIX="/path/to/miniforge3/envs/scx-gpu"
export PATH="${CONDA_PREFIX}/bin:$PATH"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd /path/to/scx
cd pyscx && maturin develop --release --features hdf5,gpu && cd ..

python my_analysis.py
```

### Using environment modules

```bash
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load cuda/12.2
module load python/3.12

# Verify nvcc is available
nvcc --version

cd /path/to/scx
cd pyscx && maturin develop --release --features hdf5,gpu && cd ..

python my_analysis.py
```

## Driver compatibility

The NVIDIA driver determines the maximum CUDA version you can use. The CUDA
Toolkit version must not exceed the driver's supported CUDA version.

| Driver version | Max CUDA | Recommended `cuda-version=` |
|---------------|----------|----------------------------|
| 525.x | 12.0 | 12.0 |
| 535.x | 12.2 | 12.2 |
| 545.x | 12.3 | 12.2 |
| 550.x | 12.4 | 12.4 |
| 555.x | 12.5 | 12.4 |
| 560.x | 12.6 | 12.6 |

Check your driver version and max CUDA version:

```bash
nvidia-smi
# Look for "CUDA Version: 12.x" in the top-right corner
```

## How GPU dispatch works

SCX's GPU acceleration is split between **rapids-singlecell** (in-VRAM compute
layer for most analysis ops) and **native Rust CUDA kernels** (streaming,
out-of-VRAM, and structurally-unique paths).

### Architecture layers

1. **rapids-singlecell** (Python, runtime-detected) — the primary in-VRAM GPU
   compute layer. When present and `device="gpu"`, PCA / kNN / UMAP /
   preprocessing / extra HVG flavors delegate to `rsc.pp.*` / `rsc.tl.*`.
   SCX hands it a GPU-resident `cupyx.sparse.csr_matrix` (from
   `to_gpu_anndata()` or via `rsc.get.anndata_to_GPU`), so there is no host
   round-trip for chained ops.

2. **scx-gpu** crate (Rust, compiled PTX) — streaming/randomized PCA
   (cuSPARSE SpMM + cuSOLVER QR + cuBLAS), streaming preprocessing kernels,
   CSC-direct DE, NB-GLM pseudobulk DE, Harmony, and the ML loader
   decode-to-device path. These run for backed/lazy `X` (out-of-VRAM),
   under `SCX_FORCE_NATIVE_GPU=1`, or on ops where rapids has no equivalent.

3. **scx-accel** crate — analysis accelerator routing with optional `gpu`
   feature. The **execution planner** (`scx_accel::route`) decides which
   backend receives each op based on input type, device request, and rapids
   availability.

4. **pyscx** Python bindings — `pyscx.accel.*` functions pass `device=`
   through to scx-accel. rapids-singlecell, cuVS (fused CAGRA kNN), and
   cuGraph (Leiden) are loaded at runtime from Python — they don't need to
   be present at Rust compile time.

5. **Fallback** — every operation falls back to CPU if the GPU path is
   unavailable (no device, missing library, CUDA error). A `UserWarning` is
   emitted but no exception is raised (except `device="gpu"` on a host with
   no CUDA GPU, which errors up front; `device="auto"` falls back quietly).

### Dispatch examples

```
# In-VRAM with rapids (the common case)
pyscx.accel.pca(adata, device="gpu")
    → rapids-singlecell: rsc.pp.pca()
    → route: rapids_singlecell_gpu

pyscx.accel.neighbors(adata, device="gpu")
    → rapids-singlecell: rsc.pp.neighbors()
    → route: rapids_singlecell_gpu
    → Falls back to CPU HNSW if rapids absent

pyscx.accel.umap(adata, device="gpu")
    → rapids-singlecell: rsc.tl.umap()
    → route: rapids_singlecell_gpu
    → Falls back to cuML then CPU SGD if rapids absent

# Native GPU paths (no rapids needed)
pyscx.accel.pca(adata_backed, device="gpu")
    → scx-gpu: streaming randomized PCA (cuSPARSE SpMM + cuSOLVER QR)
    → route: gpu_csr

pyscx.accel.leiden(adata, device="gpu")
    → Python-side: import cugraph → GPU Leiden
    → Falls back to leidenalg (CPU) if cugraph absent

pyscx.accel.pdex_ref(adata_backed, device="gpu")
    → scx-gpu: CSC-direct DE (with CSC sidecar → gpu_csc_v3)
    → Falls back to gpu_csr_v3 without sidecar
```

### What routes where

| Operation | In-VRAM (in-memory `X`) | Out-of-VRAM (backed/lazy `X`) |
|-----------|------------------------|-------------------------------|
| PCA | rapids (`rsc.pp.pca`) | Native streaming randomized (cuSPARSE) |
| kNN / neighbors | rapids (`rsc.pp.neighbors`) | CPU HNSW |
| UMAP | rapids (`rsc.tl.umap`) | cuML fallback → CPU SGD |
| Preprocessing | rapids (`rsc.pp.normalize_total`/`log1p`) | Native streaming kernels |
| HVG (seurat_v3) | Native (atomic-CSR) | Native streaming |
| HVG (other flavors) | rapids (`rsc.pp.highly_variable_genes`) | CPU |
| Leiden | cuGraph (native) | cuGraph (native) |
| DE (pdex/Wilcoxon rank-sum) | Native CSC-direct / CSR | Native CSC-direct / CSR |
| NB-GLM pseudobulk DE | Native (`gpu_nb_glm_csr`) | Native (`gpu_nb_glm_csr`) |
| Harmony | Native | Native |
| Fused PCA→kNN | rapids pipeline (in-memory) | Native streaming PCA + device-resident CAGRA (cuVS) |

## Troubleshooting

### `nvcc` not found during build

The scx-gpu build script requires `nvcc` to compile CUDA kernels to PTX. If
`nvcc` is not on PATH, the build succeeds but writes empty PTX stubs — GPU
kernels will not be functional at runtime.

```bash
# Check if nvcc is available
which nvcc

# If not, add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH

# Or load the module (HPC)
module load cuda/12.2

# Then rebuild
cargo clean -p scx-gpu
cd pyscx && maturin develop --release --features hdf5,gpu
```

### A GPU op runs for minutes with no output / the GPU shows ~0% util — is it hung?

Usually **not hung** — it is the expected behavior of the streaming, out-of-VRAM
GPU path on **backed / atlas-scale** input. Ops like `highly_variable_genes`
(`flavor="seurat_v3"`) and streaming PCA decode the SCX shards on the **CPU**
(many threads → high `%CPU`) and feed batches to the GPU kernel; on a
1M-cell / billion-nnz backed matrix the CPU-side decode dominates, so the GPU
can sit near 0% utilization for the bulk of the run while real work proceeds.

To see the chosen route the instant an op starts — so you can tell "GPU route,
decode-bound" from "hung" — raise the `pyscx` logger to INFO:

```python
import logging
logging.basicConfig(level=logging.INFO)          # or:
logging.getLogger("pyscx.accel").setLevel(logging.INFO)

import pyscx
adata = pyscx.open("census_1m.scx").to_anndata(backed=True)
pyscx.accel.highly_variable_genes(adata, flavor="seurat_v3", device="gpu")
# INFO pyscx.accel: highly_variable_genes: route=gpu_csr device=gpu fallback=none
# INFO pyscx.accel: highly_variable_genes: seurat_v3 over 1000000×61497 on gpu —
#                   streaming mean/var; large/backed input is CPU-decode-bound
#                   (GPU may show low utilization), not hung
```

The final route is also recorded in `adata.uns["scx_accel"]["<op>"]` (`route`,
`fallback_reason`) after the call returns. If an explicit `device="gpu"` request
silently lands on CPU (e.g. the input layout has no GPU kernel), pyscx now emits
a `UserWarning` naming the `fallback_reason` rather than failing silently. (A
`device="gpu"` request on a host with no CUDA GPU still errors up front; `"auto"`
falls back to CPU quietly by design.)

### What `device="auto"` does and does not do

`device="auto"` resolves to GPU or CPU **once, before the op starts**, from the
pre-flight conditions `fallback_reason` enumerates: is CUDA present, does this
op have a kernel for this input layout, are the dimensions in range, is rapids
importable. That decision is then final.

It is **not** a safety net for a GPU that is present but fails while running.
An out-of-memory error, a driver fault, or a kernel launch failure **raises**;
`auto` does not quietly re-run the op on CPU. That is deliberate: a CPU re-run
of an atlas-scale DE or PCA is not a degradation you can ignore, it is hours of
work you did not ask for, discoverable only after the fact — and the run that
already failed has consumed its time either way. The error names the shortfall
and the remedy instead. When you want the CPU kernel, ask for it:

```python
try:
    pyscx.accel.harmony_integrate(adata, key="batch")          # device="auto"
except RuntimeError as e:
    print(e)
    # harmony_integrate_gpu: GPU out of memory: Harmony needs ≥11.3 GB of device
    # memory for 8000000 cells × 50 PCs × 100 clusters, but only 6.1 GB of
    # 79.1 GB is free on GPU 0. Re-run with device="cpu", lower n_clusters, or
    # free VRAM — device="auto" resolves the device before the op starts and
    # does not fall back to CPU on a runtime GPU failure.
    pyscx.accel.harmony_integrate(adata, key="batch", device="cpu")
```

Because a failed op raises, it also leaves **no** `uns["scx_accel"]` entry — the
stamp is rolled back rather than rewritten to claim a CPU run that never
happened. See [api.md § Accelerator route metadata](api.md#accelerator-route-metadata).

The one place a runtime failure is *survived* rather than raised is
`Experiment.to_gpu_anndata`, and it is not a GPU→CPU fallback: when the in-VRAM
shard decode fails, the host-assemble path uploads the same matrix to the same
device, so the result is unchanged. It warns and records
`fallback_reason="gpu_runtime_error"`, because a device decode that stopped
working — a build whose PTX did not compile, a driver mismatch — is worth
knowing about even though the call succeeded.

### `pyscx.accel.gpu_available()` returns `False`

Possible causes:
- No NVIDIA GPU present
- NVIDIA driver not loaded (`nvidia-smi` fails)
- pyscx was built without `--features gpu`
- CUDA runtime initialization failed (check `nvidia-smi` for GPU errors)

### `cusparseBsrSetStridedBatch undefined symbol` panic at GPU PCA

**Symptom:**

```
thread '<unnamed>' panicked at .../cudarc-0.19.x/src/cusparse/sys/mod.rs:...:
Expected symbol in library: DlSym { source:
  "/usr/lib/x86_64-linux-gnu/libcusparse.so: undefined symbol:
   cusparseBsrSetStridedBatch" }
pyo3_runtime.PanicException: ...
```

(Or another `cusparse*` symbol — the exact missing function may differ
across cudarc versions.)

**Cause:** Ubuntu's `libcusparse-dev` package ships cuSPARSE 12.0.1.140
(2023-01) at `/usr/lib/x86_64-linux-gnu/libcusparse.so`. cudarc 0.19+
requires cuSPARSE 12.5+ (CUDA Toolkit 12.5, mid-2024). If the toolkit's
newer libcusparse at `/usr/local/cuda*/lib64/libcusparse.so` is not
earlier on `LD_LIBRARY_PATH`, the dynamic loader picks the older system
version and `cusparseCreate` (or the first SpMM-related call) panics
during cudarc's lazy `dlsym`.

**Fix:** prepend the toolkit's lib path:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# or the specific version:
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH
```

Verify with `ldd path/to/libpyscx.so | grep libcusparse` — should
resolve to `/usr/local/cuda*/lib64`, not `/usr/lib/x86_64-linux-gnu`.

**Behaviour as of pyscx 0.4.3+:** the runtime probes for
`cusparseBsrSetStridedBatch` at the first `accel.pca(device="gpu")` call.
If the symbol is missing, pyscx emits a one-shot `UserWarning` naming
this exact fix and routes PCA to CPU instead of panicking. The CPU path
still produces correct results; restore the GPU path by fixing
`LD_LIBRARY_PATH` and re-running.

### cuVS / cuGraph import errors

```
ImportError: libraft.so: cannot open shared object file
```

The RAPIDS libraries can't find their shared objects. Fix:

```bash
# If using conda:
conda activate scx-gpu  # ensures LD_LIBRARY_PATH includes conda lib/

# If using pip-installed RAPIDS (not recommended):
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/lib:$LD_LIBRARY_PATH
```

### CUDA version mismatch

```
CUDA driver version is insufficient for CUDA runtime version
```

Your CUDA Toolkit version exceeds what the driver supports. Either:
- Upgrade your NVIDIA driver, or
- Install a lower CUDA Toolkit version matching your driver (see driver
  compatibility table above)

### GPU memory model

**Sparse CSR is preserved on GPU.** `to_gpu_anndata()` always produces a
`cupyx.scipy.sparse.csr_matrix` — data is never densified during the
host→device transfer. VRAM usage for `X` scales with the number of non-zero
elements (NNZ), not the full N×M dense dimensions:

```
VRAM(X) ≈ NNZ × 4 (f32 values) + NNZ × 4 (i32 indices) + (N + 1) × 8 (i64 indptr)
        = NNZ × 8 + (N + 1) × 8 bytes
```

For example, a 1M-cell × 30K-gene matrix at 5% density (~1.5B NNZ) requires
~12 GB as sparse CSR, versus ~120 GB if densified. SCX's native GPU PCA uses
cuSPARSE `SpMM` (sparse × dense multiply) without densifying `X` — only the
dense working matrices (`n_obs × n_comps`) are allocated alongside the sparse
input.

**Shard-by-shard GPU assembly.** `to_gpu_anndata()` pre-allocates combined
device buffers for the full sparse matrix, then decodes each shard into the
combined buffer one at a time. Peak device memory during transfer is the
combined buffer plus **one shard** — not all shards simultaneously. Only the
indptr array (tiny: `N + 1` elements) round-trips to the host. The result is
the **complete** sparse matrix in VRAM — this is not a streaming/partial
representation.

> [!NOTE]
> **rapids-singlecell may allocate additional dense working memory.** SCX
> preserves sparsity when handing data to rapids, but rapids ops may
> internally allocate dense working buffers. For example, `rsc.pp.pca()` uses
> cuBLAS dense matmul internally, so peak VRAM during GPU PCA can be
> substantially higher than the sparse `X` footprint alone. Use
> `estimate_gpu_memory()` (below) to pre-flight sizing before launching an op.

#### VRAM pre-flight guard

`to_gpu_anndata()` checks available VRAM before transferring. It computes
`device_bytes = NNZ × 8 + (N + 1) × 8`, multiplies by a **1.2× headroom
factor**, and compares against free device memory. If VRAM is insufficient, it
raises a `ValueError` with an actionable message pointing to the backed /
streaming workflows — it never silently OOMs.

#### Estimating VRAM requirements

Use `estimate_gpu_memory()` and `gpu_info()` to plan GPU resource allocation
before launching an operation:

```python
import pyscx

# Check available VRAM
info = pyscx.accel.gpu_info()
# {'device': 'NVIDIA A100-SXM4-80GB', 'total_vram_gb': 80.0, 'free_vram_gb': 72.3}

# Estimate VRAM for a specific operation
est = pyscx.accel.estimate_gpu_memory(adata, operation="pca", n_comps=50)
# {'required_gb': 2.1, 'fits_in_vram': True}
```

Per-op VRAM sizing (approximate):

| Component | Formula | Example (1M × 2K) |
|-----------|---------|-------------------|
| Sparse `X` on device | `NNZ × 8 + (N + 1) × 8` | ~800 MB at 5% density |
| PCA working matrices | `Y: N × k × 4` + `Ω+B: 2 × D × k × 4` + `shard_dense: shard_rows × D × 4` + `QR: 2 × N × k × 4` | ~600 MB (k=50) |
| kNN input | `N × D × 4` (float32 embeddings) | 200 MB (50 PCs) |
| UMAP working memory | ~`N × 12` bytes for graph + embeddings | ~12 MB |
| DE dense chunk (CSR path) | `N × gene_chunk_size × 4` | ~2 GB (chunk=500) |

> [!WARNING]
> Estimates are approximate — cuSOLVER QR workspace may be undercounted
> by ~1.5×. When NNZ is not available from the input data, the estimator
> falls back to a ~10% density assumption.

#### Backed / lazy `X` — automatic streaming

For backed / lazy `X` (`ScxBackedSparseDataset` / `ScxLazyTransformedDataset`),
SCX automatically uses native streaming kernels instead of uploading the full
matrix. GPU PCA streams shards to avoid full-matrix materialization; backed
`X` never routes to rapids (which would OOM). This means:

- `pyscx.open(...).to_anndata(backed=True)` + `pyscx.accel.pca(adata, device="gpu")`
  streams shard-by-shard — peak VRAM is one shard plus working matrices.
- `pyscx.open(...).to_gpu_anndata()` + rapids ops loads the full sparse `X`
  into VRAM — use only when `X` fits.

If VRAM is insufficient for in-memory ops, use `backed=True` to take the
streaming path. To force a specific GPU on multi-GPU systems:

```python
pyscx.accel.pca(adata, device="gpu:0")   # first GPU
pyscx.accel.pca(adata, device="gpu:1")   # second GPU
```
