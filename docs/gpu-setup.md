# GPU Setup Guide

This guide covers installing and configuring GPU acceleration for SCX.
GPU support enables CUDA-accelerated PCA, kNN, UMAP, Leiden clustering,
and fused preprocessing — accessed through the same `pyscx.accel.*` API
with a `device="gpu"` parameter.

## Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| NVIDIA GPU | Volta (compute 7.0) | Ampere / Hopper | PTX compiled for compute_70, forward-compatible |
| NVIDIA driver | 525+ | 535+ | Must support CUDA ≥ 12.0 (`nvidia-smi` shows max CUDA version) |
| CUDA Toolkit | 12.0 | 12.2–12.6 | Provides `nvcc` for kernel compilation |
| cuVS | 24.10+ | 24.12+ | GPU kNN via CAGRA (optional — CPU fallback available) |
| cuGraph | 24.10+ | 24.12+ | GPU Leiden clustering (optional — CPU fallback available) |
| Rust toolchain | 1.78+ | stable | For building scx-gpu crate |
| Python | 3.10+ | 3.12–3.13 | For pyscx bindings |

**What requires what:**

| SCX operation | CUDA Toolkit (`nvcc`) | cuVS | cuGraph |
|---------------|----------------------|------|---------|
| GPU PCA — randomized (cuSPARSE SpMM + cuSOLVER QR + cuBLAS) | required | — | — |
| GPU PCA — covariance (cuSPARSE + cuSOLVER `syevd` + cuBLAS) | required | — | — |
| GPU CholeskyQR2 (cuSOLVER `potrf` + cuBLAS `strsm`) | required | — | — |
| GPU UMAP (native CUDA SGD kernel) | required | — | — |
| GPU preprocessing (`normalize_total` / `log1p` / `highly_variable_genes` with `device="gpu"`) | required | — | — |
| GPU kNN (CAGRA) | required | required | — |
| GPU Leiden clustering | — | — | required |

Operations without their required dependencies fall back to CPU automatically
with a warning — no crashes.

**Note on cuBLAS:** covariance PCA, GPU-resident final-embedding
multiply, CholeskyQR2, and preprocessing Gram correction depend on **cuBLAS**.
`libcublas.so` ships alongside `libcusparse.so` / `libcusolver.so` inside
every CUDA Toolkit 12.x install, so no new runtime library path or env-var
setup is required — any working `scx-gpu` env from prior releases continues
to work out of the box.

## Option A: conda (recommended)

Conda is the easiest way to get a working GPU environment because it resolves
the full CUDA + RAPIDS dependency tree and matches library versions to your
driver automatically.

```bash
# 1. Create a dedicated environment
conda create -n scx-gpu python=3.13
conda activate scx-gpu

# 2. Install RAPIDS packages — pin cuda-version to match your driver
#    Check your driver's max CUDA version:  nvidia-smi
#    Driver 535.x → cuda-version=12.2
#    Driver 550.x → cuda-version=12.4
#    Driver 560.x → cuda-version=12.6
conda install -c rapidsai -c conda-forge \
    cuvs cugraph cuda-version=12.2

# 3. Install Python dependencies
pip install maturin numpy scipy pyarrow anndata scanpy scikit-learn leidenalg

# 4. Build pyscx with GPU support
cd pyscx && maturin develop --release --features gpu

# 5. Verify
python -c "import pyscx; print('GPU available:', pyscx.accel.gpu_available())"
```

**Why conda?** pip-installed RAPIDS packages may pull CUDA 12.9+ runtime
libraries that are incompatible with older drivers (e.g., driver 535 supports
CUDA ≤ 12.2). Conda pins `cuda-version` and resolves compatible builds for
cuVS, cuGraph, RAFT, and RMM together.

## Option B: system CUDA Toolkit (no RAPIDS)

Use this if you only need GPU PCA and UMAP and want to avoid conda. kNN and
Leiden will fall back to CPU since cuVS/cuGraph are not installed.

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
cd pyscx && ../.venv/bin/maturin develop --release --features gpu
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

For a lighter image with only GPU PCA and UMAP (kNN/Leiden fall back to CPU),
use the NVIDIA CUDA devel base image directly:

```bash
docker run --gpus all -it nvidia/cuda:12.2.2-devel-ubuntu22.04

# Inside the container:
apt update && apt install -y python3 python3-pip curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
pip install maturin numpy scipy pyarrow anndata
cd pyscx && maturin develop --release --features gpu
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

## rapids-singlecell analysis backend (GPU compute layer)

SCX's native GPU accelerators (PCA, kNN, UMAP, Leiden, HVG, preprocessing, DE)
work with the conda/CUDA setup above and **do not** require RAPIDS. Separately,
SCX can route `device="gpu"` analysis ops to
[rapids-singlecell](https://rapids-singlecell.readthedocs.io/) as a GPU *compute
layer* (the destination of the ACC-RUST-OPT-V4 transition). This backend is an
**optional, detected runtime dependency**, not a hard requirement:

- When rapids-singlecell **is** importable, supported GPU-analysis ops route to
  it; SCX hands it a GPU-resident matrix so there is no host round-trip.
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
cd pyscx && maturin develop --release --features gpu && cd ..
```

rapids-singlecell itself is **not** pinned in that spec and must be installed
**manually on a GPU node**, because it cannot be resolved reproducibly by conda
or a pip extra (it is CUDA-arch + RAPIDS-version coupled; `>=0.12` ships CUDA
kernels built with `-arch=native`, which needs a GPU visible at build time so
nvcc resolves `sm_90` on H100):

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
`transfer_mode` (`scx_device_handoff` when `X` is already device-resident — e.g.
from `pyscx.open(...).to_gpu_anndata()` — or `anndata_to_gpu` when rapids
uploads a host `X`).

**`X` residency contract.** When the op uploads a host `X` (`anndata_to_gpu`),
the result slots (`obsm`/`obsp`) **and** `X` are brought back to host afterwards
and the device buffers freed — so `pyscx.accel.<op>(adata, device="gpu")` on an
in-memory AnnData leaves `adata.X` host-resident, exactly as the native path
does. When `X` arrives already device-resident (`scx_device_handoff`, from
`to_gpu_anndata()`), it is **left on the GPU** so a chain of `pyscx.accel.*`
calls runs without re-uploading — that path is the way to keep data GPU-resident
across ops.

These stay **native** (SCX wins or is structurally unique): `seurat_v3` /
`seurat_v3_paper` HVG, Leiden, CSC-direct / pdex DE, Harmony, and every
**out-of-VRAM** path — a backed/lazy `X` (`ScxBackedSparseDataset` /
`ScxLazyTransformedDataset`) always uses SCX's native streaming kernels, never
rapids (which would OOM).

### Forcing the native GPU kernels

`SCX_FORCE_NATIVE_GPU=1` keeps the native SCX GPU kernels for in-VRAM ops
instead of routing to rapids — a transition-only A/B + rollback switch (removed
once the rapids routes are gated and the superseded native kernels are deleted).
When rapids is absent and this override is **not** set, in-VRAM GPU-analysis ops
fall back to CPU (`fallback_reason="no_rapids"`) with a one-shot `UserWarning`.

`SCX_DISABLE_RAPIDS=1` forces that rapids-absent CPU fallback **even on a host
where rapids is installed** — it makes the dispatcher treat rapids as
unimportable. It exists so the no-rapids fallback contract can be exercised
(tests / the Phase 2 benchmark gate) without uninstalling rapids; it takes
precedence over `SCX_FORCE_NATIVE_GPU`.

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
cd pyscx && maturin develop --release --features gpu && cd ..

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
cd pyscx && maturin develop --release --features gpu && cd ..

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

SCX's GPU support is layered:

1. **scx-gpu** crate — Rust CUDA kernels (compiled to PTX via `nvcc` at build
   time) and cuSPARSE/cuSOLVER bindings via `cudarc`. Provides GPU PCA
   (streaming SpMM), UMAP (CUDA SGD), and fused preprocessing.

2. **scx-accel** crate — analysis accelerators with optional `gpu` feature.
   When enabled, PCA/kNN/UMAP functions accept `device="gpu"` and dispatch
   to scx-gpu.

3. **pyscx** Python bindings — `pyscx.accel.*` functions pass `device=`
   through to scx-accel. cuVS (kNN) and cuGraph (Leiden) are loaded at
   runtime from Python — they don't need to be present at Rust compile time.

4. **Fallback** — every operation falls back to CPU if the GPU path is
   unavailable (no device, missing library, CUDA error). A warning is emitted
   but no exception is raised.

```
pyscx.accel.pca(adata, device="gpu")
    → scx-accel (Rust, gpu feature)
        → scx-gpu: cuSPARSE SpMM + cuSOLVER QR + cuRAND
            → CUDA kernels (PTX, compiled from scx-gpu/kernels/*.cu)

pyscx.accel.neighbors(adata, device="gpu")
    → scx-accel (Rust, gpu feature)
        → Python-side: import cuvs → CAGRA index build + search
            → Falls back to CPU HNSW if cuvs not installed

pyscx.accel.leiden(adata, device="gpu")
    → Python-side: import cugraph → GPU Leiden
        → Falls back to leidenalg (CPU) if cugraph not installed
```

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
cd pyscx && maturin develop --release --features gpu
```

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

### GPU out of memory

GPU PCA streams shards to avoid full matrix materialization, but kNN (CAGRA)
and UMAP load the full embedding matrix into VRAM.

For a dataset with N cells and D dimensions:
- kNN input: N × D × 4 bytes (float32) — 1M cells × 50 PCs ≈ 200 MB
- UMAP working memory: ~N × 12 bytes for graph + embeddings

If VRAM is insufficient, operations fall back to CPU. To force a specific GPU
on multi-GPU systems:

```python
pyscx.accel.pca(adata, device="gpu:0")   # first GPU
pyscx.accel.pca(adata, device="gpu:1")   # second GPU
```
