#!/usr/bin/env bash
# =============================================================================
# SCX Comprehensive Benchmark Suite — Environment Setup
# =============================================================================
#
# Creates isolated conda environments for benchmarking. Four environments
# are available:
#
#   scx-bench       CPU benchmarks (all format comparisons, accelerators,
#                   lazy preprocessing, correctness validation, ML loaders)
#   scx-bench-gpu   GPU benchmarks (extends CPU with CUDA + RAPIDS: cuVS, cuGraph)
#   scx-bench-r     R / BPCells benchmarks (isolated R environment)
#   scx-bench-eval  cell-eval / arc-bench parity validation
#
# Usage:
#     # Create the CPU benchmark environment (default)
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh
#
#     # Create the GPU benchmark environment
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --gpu
#
#     # Create the R / BPCells benchmark environment
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --r
#
#     # Create the cell-eval / arc-bench parity validation environment
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --eval
#
#     # Create all four environments
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --all
#
#     # Check what's installed in each environment
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --check
#
#     # Rebuild pyscx inside the CPU environment
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --rebuild
#
#     # Rebuild pyscx with GPU features inside the GPU environment
#     bash benchmarks/comprehensive/scripts/install_dependencies.sh --rebuild --gpu
#
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
ENVS_DIR="${REPO_ROOT}/benchmarks/comprehensive/envs"

# Detect conda
CONDA=""
for candidate in conda mamba micromamba; do
    if command -v "$candidate" &>/dev/null; then
        CONDA="$candidate"
        break
    fi
done

# Also check miniforge3 in home directory
if [[ -z "$CONDA" ]] && [[ -f "$HOME/miniforge3/condabin/conda" ]]; then
    CONDA="$HOME/miniforge3/condabin/conda"
fi

if [[ -z "$CONDA" ]]; then
    echo "ERROR: conda/mamba/micromamba not found."
    echo "Install miniforge3: https://github.com/conda-forge/miniforge"
    exit 1
fi

# Parse flags
CREATE_CPU=false
CREATE_GPU=false
CREATE_R=false
CREATE_EVAL=false
CHECK_ONLY=false
REBUILD_ONLY=false

if [[ $# -eq 0 ]]; then
    CREATE_CPU=true  # Default: create CPU env
fi

for arg in "$@"; do
    case "$arg" in
        --cpu)      CREATE_CPU=true ;;
        --gpu)      CREATE_GPU=true ;;
        --r)        CREATE_R=true ;;
        --eval)     CREATE_EVAL=true ;;
        --all)      CREATE_CPU=true; CREATE_GPU=true; CREATE_R=true; CREATE_EVAL=true ;;
        --check)    CHECK_ONLY=true ;;
        --rebuild)  REBUILD_ONLY=true ;;
        -h|--help)
            head -37 "$0" | tail -33
            exit 0
            ;;
        *)
            echo "Unknown flag: $arg"
            echo "Use --help for usage."
            exit 1
            ;;
    esac
done

# Default --rebuild to CPU env unless --gpu is also passed
if $REBUILD_ONLY && ! $CREATE_GPU; then
    CREATE_CPU=true
fi

echo "=============================================="
echo "SCX Benchmark Suite — Environment Setup"
echo "=============================================="
echo "Repo:    ${REPO_ROOT}"
echo "Conda:   ${CONDA}"
echo ""

# ---------------------------------------------------------------------------
# Helper: get conda prefix for an env name
# ---------------------------------------------------------------------------
get_env_prefix() {
    local env_name="$1"
    "$CONDA" info --envs 2>/dev/null | grep "^${env_name} " | awk '{print $NF}' || true
}

# ---------------------------------------------------------------------------
# Helper: activate a conda env (non-interactive shell compatible)
# ---------------------------------------------------------------------------
activate_env() {
    local env_name="$1"
    local prefix
    prefix=$(get_env_prefix "$env_name")
    if [[ -z "$prefix" ]]; then
        echo "ERROR: Environment '$env_name' not found."
        return 1
    fi
    export CONDA_PREFIX="$prefix"
    export PATH="${prefix}/bin:$PATH"
    export LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}"
}

# ---------------------------------------------------------------------------
# Helper: build pyscx inside an activated environment
# ---------------------------------------------------------------------------
build_pyscx() {
    local features="${1:-}"
    echo "  Building pyscx (release mode${features:+, features: $features})..."
    cd "${REPO_ROOT}/pyscx"
    if [[ -n "$features" ]]; then
        maturin develop --release --features "$features" 2>&1 | tail -5
    else
        maturin develop --release 2>&1 | tail -5
    fi
    cd "${REPO_ROOT}"
    echo "  pyscx built successfully."
}

# ---------------------------------------------------------------------------
# Check mode: verify installed packages in each environment
# ---------------------------------------------------------------------------
if $CHECK_ONLY; then
    for env_name in scx-bench scx-bench-gpu scx-bench-r scx-bench-eval; do
        prefix=$(get_env_prefix "$env_name")
        if [[ -z "$prefix" ]]; then
            echo "❌ ${env_name}: NOT CREATED"
            echo ""
            continue
        fi
        echo "✅ ${env_name}: ${prefix}"

        if [[ "$env_name" == "scx-bench-r" ]]; then
            # R environment — check R and BPCells
            echo "  R:       $("${prefix}/bin/R" --version 2>/dev/null | head -1 || echo 'not found')"
            echo "  BPCells: $("${prefix}/bin/Rscript" -e 'cat(as.character(packageVersion("BPCells")))' 2>/dev/null || echo 'not installed')"
        elif [[ "$env_name" == "scx-bench-eval" ]]; then
            # Eval environment — check cell-eval / arc-bench specific packages
            "${prefix}/bin/python" -c "
import importlib.metadata
packages = [
    'anndata', 'scanpy', 'scipy', 'numpy', 'pandas', 'h5py',
    'scikit-learn', 'python-igraph', 'leidenalg',
    'pyarrow', 'maturin', 'psutil',
    'cell-eval', 'arc-bench', 'pdex', 'polars', 'tqdm',
]
for pkg in packages:
    try:
        v = importlib.metadata.version(pkg)
        print(f'  ✅ {pkg:20s} {v}')
    except importlib.metadata.PackageNotFoundError:
        print(f'  ❌ {pkg:20s} NOT INSTALLED')

# pyscx
try:
    import pyscx
    print(f'  ✅ {\"pyscx\":20s} (from source)')
except ImportError:
    print(f'  ⚠️  {\"pyscx\":20s} NOT BUILT — run: install_dependencies.sh --rebuild --eval')
" 2>/dev/null || echo "  (failed to query packages)"
        else
            # Python environment — check key packages
            "${prefix}/bin/python" -c "
import importlib.metadata
packages = [
    'anndata', 'scanpy', 'zarr', 'h5py', 'scipy', 'numpy', 'pandas',
    'pyarrow', 'tiledbsoma', 'matplotlib', 'seaborn', 'scikit-learn',
    'umap-learn', 'leidenalg', 'psutil', 'maturin',
]
for pkg in packages:
    try:
        v = importlib.metadata.version(pkg)
        print(f'  ✅ {pkg:20s} {v}')
    except importlib.metadata.PackageNotFoundError:
        print(f'  ❌ {pkg:20s} NOT INSTALLED')

# pyscx
try:
    import pyscx
    print(f'  ✅ {\"pyscx\":20s} (from source)')
except ImportError:
    print(f'  ⚠️  {\"pyscx\":20s} NOT BUILT — run: install_dependencies.sh --rebuild')

# torch
try:
    import torch
    gpu_str = f', CUDA {torch.version.cuda}' if torch.cuda.is_available() else ', CPU-only'
    print(f'  ✅ {\"torch\":20s} {torch.__version__}{gpu_str}')
except ImportError:
    print(f'  ❌ {\"torch\":20s} NOT INSTALLED')
" 2>/dev/null || echo "  (failed to query packages)"

            if [[ "$env_name" == "scx-bench-gpu" ]]; then
                # Check GPU-specific packages
                "${prefix}/bin/python" -c "
try:
    import cuvs; print(f'  ✅ {\"cuvs\":20s} (available)')
except ImportError:
    print(f'  ❌ {\"cuvs\":20s} NOT INSTALLED')
try:
    import cugraph; print(f'  ✅ {\"cugraph\":20s} (available)')
except ImportError:
    print(f'  ❌ {\"cugraph\":20s} NOT INSTALLED')
" 2>/dev/null || true
            fi
        fi
        echo ""
    done

    # System info
    echo "--- System Info ---"
    echo "  Hostname: $(hostname)"
    echo "  Kernel:   $(uname -r)"
    echo "  CPU:      $(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || echo 'unknown')"
    echo "  RAM:      $(awk '/MemTotal/ {printf "%.0f GB", $2/1048576}' /proc/meminfo 2>/dev/null || echo 'unknown')"
    echo "  Rust:     $(rustc --version 2>/dev/null || echo 'not found')"
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU:      $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'unavailable')"
    fi
    exit 0
fi

# ---------------------------------------------------------------------------
# Rebuild mode: just rebuild pyscx
# ---------------------------------------------------------------------------
if $REBUILD_ONLY; then
    if $CREATE_GPU; then
        echo "--- Rebuilding pyscx in scx-bench-gpu ---"
        activate_env "scx-bench-gpu"
        build_pyscx "gpu"
    elif $CREATE_EVAL; then
        echo "--- Rebuilding pyscx in scx-bench-eval ---"
        activate_env "scx-bench-eval"
        build_pyscx
    else
        echo "--- Rebuilding pyscx in scx-bench ---"
        activate_env "scx-bench"
        build_pyscx
    fi
    echo ""
    echo "Done!"
    exit 0
fi

# ---------------------------------------------------------------------------
# Create environments
# ---------------------------------------------------------------------------

if $CREATE_CPU; then
    echo "=============================================="
    echo "Creating scx-bench (CPU benchmark environment)"
    echo "=============================================="
    if get_env_prefix "scx-bench" | grep -q .; then
        echo "  Environment 'scx-bench' already exists. Updating..."
        "$CONDA" env update -f "${ENVS_DIR}/scx-bench.yml" --prune 2>&1 | tail -10
    else
        "$CONDA" env create -f "${ENVS_DIR}/scx-bench.yml" 2>&1 | tail -10
    fi
    echo ""

    # Build pyscx
    echo "--- Building pyscx in scx-bench ---"
    activate_env "scx-bench"
    build_pyscx
    echo ""
fi

if $CREATE_GPU; then
    echo "=============================================="
    echo "Creating scx-bench-gpu (GPU benchmark environment)"
    echo "=============================================="
    echo ""
    echo "NOTE: Adjust cuda-version in envs/scx-bench-gpu.yml to match your driver."
    echo "      Check with: nvidia-smi (look for 'CUDA Version: 12.x')"
    echo ""
    if get_env_prefix "scx-bench-gpu" | grep -q .; then
        echo "  Environment 'scx-bench-gpu' already exists. Updating..."
        "$CONDA" env update -f "${ENVS_DIR}/scx-bench-gpu.yml" --prune 2>&1 | tail -10
    else
        "$CONDA" env create -f "${ENVS_DIR}/scx-bench-gpu.yml" 2>&1 | tail -10
    fi
    echo ""

    # Build pyscx with GPU features
    echo "--- Building pyscx (with GPU) in scx-bench-gpu ---"
    activate_env "scx-bench-gpu"
    build_pyscx "gpu"
    echo ""
fi

if $CREATE_R; then
    echo "=============================================="
    echo "Creating scx-bench-r (R / BPCells environment)"
    echo "=============================================="
    if get_env_prefix "scx-bench-r" | grep -q .; then
        echo "  Environment 'scx-bench-r' already exists. Updating..."
        "$CONDA" env update -f "${ENVS_DIR}/scx-bench-r.yml" --prune 2>&1 | tail -10
    else
        "$CONDA" env create -f "${ENVS_DIR}/scx-bench-r.yml" 2>&1 | tail -10
    fi
    echo ""

    # Install BPCells from GitHub
    echo "--- Installing BPCells from GitHub ---"
    activate_env "scx-bench-r"
    Rscript -e 'if (!requireNamespace("remotes", quietly=TRUE)) install.packages("remotes", repos="https://cloud.r-project.org"); remotes::install_github("bnprks/BPCells/r", quiet=TRUE)' 2>&1 | tail -5
    echo ""
fi

if $CREATE_EVAL; then
    echo "=============================================="
    echo "Creating scx-bench-eval (cell-eval / arc-bench parity validation)"
    echo "=============================================="
    if get_env_prefix "scx-bench-eval" | grep -q .; then
        echo "  Environment 'scx-bench-eval' already exists. Updating..."
        "$CONDA" env update -f "${ENVS_DIR}/scx-bench-eval.yml" --prune 2>&1 | tail -10
    else
        "$CONDA" env create -f "${ENVS_DIR}/scx-bench-eval.yml" 2>&1 | tail -10
    fi
    echo ""

    # Build pyscx (release mode, no GPU features)
    echo "--- Building pyscx in scx-bench-eval ---"
    activate_env "scx-bench-eval"
    build_pyscx
    echo ""
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo "=============================================="
echo "Setup complete! Verifying..."
echo "=============================================="
echo ""
bash "$0" --check

echo ""
echo "=============================================="
echo "Usage:"
echo "  CPU benchmarks:  conda activate scx-bench"
echo "  GPU benchmarks:  conda activate scx-bench-gpu"
echo "  R benchmarks:    conda activate scx-bench-r"
echo "  Eval parity:     conda activate scx-bench-eval"
echo ""
echo "  Run orchestrator:"
echo "    conda activate scx-bench"
echo "    python benchmarks/comprehensive/scripts/run_all.py --smoke"
echo "=============================================="
