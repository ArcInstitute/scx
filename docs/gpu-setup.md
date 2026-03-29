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
| GPU PCA (cuSPARSE SpMM + cuSOLVER) | required | — | — |
| GPU UMAP (native CUDA SGD kernel) | required | — | — |
| GPU fused preprocessing (normalize+log1p) | required | — | — |
| GPU kNN (CAGRA) | required | required | — |
| GPU Leiden clustering | — | — | required |

Operations without their required dependencies fall back to CPU automatically
with a warning — no crashes.

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
