#!/usr/bin/env bash
# =============================================================================
# Streaming conversion benchmark — smoke-test submission for Lambda HPC.
#
# Validates the end-to-end harness (build pyscx with hdf5, generate a
# small synthetic h5ad, run conversion_streaming, write results)
# against a tiny synthetic fixture so the wiring is exercised even
# when canonical census_*m fixtures aren't provisioned locally.
# Production runs against the census fixtures use
# `run_slurm_conversion_streaming.sh` with the dataset name and a
# real h5ad on disk.
#
# Differences from run_slurm_conversion_streaming.sh:
#   1. The job script installs `hdf5` and `maturin` into the
#      activated conda env (a no-op when they're already present),
#      because the cluster's stock `arcbench-scx-bench` env hasn't
#      historically needed them. Without these, `maturin develop
#      --features hdf5` fails at `hdf5-sys` build.
#   2. The job generates a small synthetic h5ad on `/scratch` rather
#      than expecting a real fixture under `$SCX_DATA_DIR`.
#   3. The job registers the synthetic fixture into the harness via
#      a dataset override env var.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
LOGS_DIR="${REPO_ROOT}/benchmarks/comprehensive/logs"
mkdir -p "$LOGS_DIR"

CONDA_ENV="${CONDA_ENV:-arcbench-scx-bench}"
CONDA_BASE="${HOME}/miniforge3"
CONDA_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV}"
if [[ ! -x "${CONDA_BASE}/bin/conda" ]]; then
    echo "ERROR: conda not at ${CONDA_BASE}/bin/conda" >&2
    exit 1
fi
if [[ ! -d "$CONDA_PREFIX" ]]; then
    echo "ERROR: Conda env '${CONDA_ENV}' not at ${CONDA_PREFIX}" >&2
    exit 1
fi

# Lambda HPC sizing:
#   * standard partition, single node, 16 CPUs, 32 GB RAM, 1 h —
#     the synthetic fixture is tiny (~20 MB h5ad), both paths
#     comfortably under the materialise ceiling.
PARTITION="${PARTITION:-standard}"
CPUS="${CPUS:-16}"
MEM="${MEM:-32G}"
TIME="${TIME:-01:00:00}"
JOB_NAME="${JOB_NAME:-scx-conv-stream-smoke}"

# Synthetic-fixture shape — small enough to fit on /scratch while
# still hitting both code paths (multiple shards). The benchmark
# reads `dataset.h5ad_path`; the synthetic generator emits there.
SYNTH_N_OBS="${SYNTH_N_OBS:-50000}"
SYNTH_N_VARS="${SYNTH_N_VARS:-5000}"
SYNTH_DENSITY="${SYNTH_DENSITY:-0.05}"

JOB_SCRIPT="${LOGS_DIR}/${JOB_NAME}.sh"

cat > "${JOB_SCRIPT}" << EOFJOB
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT}"
export CONDA_PREFIX="${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\${HOME}/.cargo/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

# Point hdf5-sys at the conda env's headers + lib.
export HDF5_DIR="\${CONDA_PREFIX}"
# scx-convert's hdf5 build needs the dev headers; without
# RUSTFLAGS the link step can't find libhdf5.so at runtime either.
export RUSTFLAGS="\${RUSTFLAGS:-} -C link-args=-Wl,-rpath,\${CONDA_PREFIX}/lib"

cd "\${REPO_ROOT}"

echo "=============================================="
echo "Streaming conversion smoke — SLURM Job"
echo "=============================================="
echo "Job ID:    \${SLURM_JOB_ID:-local}"
echo "Node:      \$(hostname)"
echo "Conda env: ${CONDA_ENV} (\${CONDA_PREFIX})"
echo "Started:   \$(date -Is)"
echo ""

# Step 1: ensure hdf5 + maturin are present. `conda` lives in
# ${CONDA_BASE}/bin, not in the per-env bin/ — use the base
# binary and pass --prefix to target our env. Idempotent.
#
# hdf5-sys 0.8.1 (the dep that scx-convert's --features hdf5 pulls
# in) parses H5_VERSION at build time and panics on 1.14.x with
# "Invalid H5_VERSION". Pin to 1.12.x — the highest 1.x the crate
# accepts. Upstream hdf5-rust 0.9 tracks 1.14 but pyscx hasn't
# moved to it yet.
echo "--- Bootstrapping conda env (hdf5 + maturin) ---"
"${CONDA_BASE}/bin/conda" install -y --prefix "\${CONDA_PREFIX}" \\
    -c conda-forge "hdf5=1.12.*=nompi*" maturin 2>&1 | tail -10
echo ""

# Step 2: build pyscx in release mode with hdf5.
# Clean pyscx + maturin staging before building. Without this, a
# prior interrupted build can leave both `target/release/libpyscx.so`
# and `target/maturin/libpyscx.so` truncated to 0 bytes; cargo's
# fingerprint cache then declares the crate "up to date" and skips
# the rebuild, after which maturin tries to parse the empty file as
# ELF and dies with "Malformed entity: Object is too small". Forcing
# a clean rebuild of pyscx ensures a fresh .so on every smoke run.
"\${HOME}/.cargo/bin/cargo" clean -p pyscx --release 2>&1 | tail -3 || true
rm -rf "\${REPO_ROOT}/target/maturin"
echo "--- Building pyscx (--features hdf5, release) ---"
cd "\${REPO_ROOT}/pyscx"
"\${CONDA_PREFIX}/bin/maturin" develop --release 2>&1 | tail -10
cd "\${REPO_ROOT}"
echo ""

# Step 3: smoke-validate that the new entry points are reachable.
echo "--- pyscx smoke ---"
"\${CONDA_PREFIX}/bin/python" -c "import pyscx; assert hasattr(pyscx, 'from_h5ad'); print('from_h5ad OK')"
echo ""

# Step 4: generate a synthetic h5ad in a writable per-node directory.
# Prefer /scratch (per-node SSD; documented at /scratch/<uid>) but
# fall back to /tmp if it isn't accessible — observed on some
# preemptible_low workers where /scratch isn't writable.
if [[ -w /scratch ]]; then
    SCRATCH_DIR="/scratch/\$(id -u)/scx_conv_smoke_\${SLURM_JOB_ID:-local}"
else
    SCRATCH_DIR="/tmp/scx_conv_smoke_\$(id -u)_\${SLURM_JOB_ID:-local}"
fi
mkdir -p "\${SCRATCH_DIR}/datasets"
"\${CONDA_PREFIX}/bin/python" "\${REPO_ROOT}/benchmarks/comprehensive/scripts/_synth_h5ad.py" \\
    --out "\${SCRATCH_DIR}/datasets/streaming_smoke.h5ad" \\
    --n-obs ${SYNTH_N_OBS} --n-vars ${SYNTH_N_VARS} --density ${SYNTH_DENSITY}
echo "Synthetic h5ad written: \$(ls -lh \${SCRATCH_DIR}/datasets/streaming_smoke.h5ad)"
echo ""

# Step 5: run the conversion_streaming benchmark against the synthetic.
# `SCX_WORK_DIR` / `SCX_DATA_DIR` redirect the harness so it picks up
# the synthetic fixture rather than the canonical one.
export SCX_WORK_DIR="\${SCRATCH_DIR}"
export SCX_DATA_DIR="\${SCRATCH_DIR}/datasets"

echo "--- Running conversion_streaming benchmark ---"
"\${CONDA_PREFIX}/bin/python" -m benchmarks.comprehensive.scripts.run_all \\
    --benchmarks conversion_streaming \\
    --datasets streaming_smoke \\
    --formats scx_auto

# Surface the result JSON the harness just wrote. `run_all.py`
# writes to `benchmarks/comprehensive/results/raw/`; print the most
# recent matching file in full so the smoke run's numbers land in
# the SLURM stdout.
RESULTS_RAW="\${REPO_ROOT}/benchmarks/comprehensive/results/raw"
echo ""
echo "--- Results summary ---"
LATEST_JSON="\$(find "\${RESULTS_RAW}" -maxdepth 1 -name 'conversion_streaming__*.json' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | awk '{print \$2}')"
if [[ -n "\${LATEST_JSON}" ]]; then
    echo "Latest result: \${LATEST_JSON}"
    cat "\${LATEST_JSON}"
else
    echo "No conversion_streaming result JSON found under \${RESULTS_RAW}"
fi

echo ""
echo "=============================================="
echo "Completed: \$(date -Is)"
echo "=============================================="
EOFJOB
chmod +x "${JOB_SCRIPT}"

SBATCH_FLAGS=(
    --job-name="${JOB_NAME}"
    --partition="${PARTITION}"
    --cpus-per-task="${CPUS}"
    --mem="${MEM}"
    --time="${TIME}"
    --output="${LOGS_DIR}/%x_%j.out"
    --error="${LOGS_DIR}/%x_%j.err"
)

JOB_ID=$(sbatch "${SBATCH_FLAGS[@]}" "${JOB_SCRIPT}" | awk '{print $NF}')
echo "Submitted streaming smoke: job ${JOB_ID}"
echo "  Log: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
echo "  Err: ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.err"
echo ""
echo "Monitor: squeue -u \$USER -j ${JOB_ID}"
echo "Tail:    tail -f ${LOGS_DIR}/${JOB_NAME}_${JOB_ID}.out"
