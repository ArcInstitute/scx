#!/bin/bash
# GPU verification for the GPU pseudobulk NB-GLM:
#   1. scx-gpu kernel smoke + scx-accel CPU↔GPU parity (cargo, release).
#   2. pyscx CPU↔GPU agreement + route-stamp pytest (native CUDA; no rapids).
#   3. Stage-A speedup measurement (bench_nb_glm.py --gpu on).
#
# Invoked under sbatch --wrap by the submitter. NOT `set -e`: one failing step
# must not abort the rest (so the speedup bench still runs even if a tolerance
# assert trips), mirroring the loose-failure stance of the GPU test harness.

set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export SCX_DISABLE_CUDA_GRAPHS=1   # NB-GLM uses single launches; harmless guard.

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
echo

rc=0

echo "=== [1a] scx-gpu gpu_nb_glm smoke (release) ==="
cargo test -p scx-gpu --release gpu_nb_glm -- --nocapture --test-threads=1 || rc=1
echo

echo "=== [1b] scx-accel CPU<->GPU NB-GLM parity (release) ==="
cargo test -p scx-accel --features gpu --release nb_glm::gpu -- --nocapture --test-threads=1 || rc=1
echo

echo "=== [2a] build pyscx (release, hdf5,gpu) ==="
# maturin --features REPLACES the default set, so hdf5 must be listed.
( cd pyscx && "${SCX_DIR}/.venv/bin/maturin" develop --release --features hdf5,gpu ) || rc=1
echo

echo "=== [2b] pyscx CPU<->GPU agreement + route pytest (native CUDA) ==="
"${SCX_DIR}/.venv/bin/python" -m pytest -v \
    pyscx/tests/test_nb_glm.py::test_nb_glm_cpu_gpu_agreement \
    pyscx/tests/test_pdex_nb_glm.py::test_pdex_nb_glm_gpu_route_stamped \
    pyscx/tests/test_pdex_nb_glm.py::test_pdex_nb_glm_cpu_gpu_agreement || rc=1
echo

echo "=== [3] Stage-A speedup measurement (bench_nb_glm.py --gpu on) ==="
"${SCX_DIR}/.venv/bin/python" pyscx/benchmarks/bench_nb_glm.py \
    --gpu on --no-pydeseq2 --out-name nb_glm_stage_a_gpu || rc=1
echo

echo "=== done (rc=${rc}) ==="
exit ${rc}
