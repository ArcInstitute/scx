#!/bin/bash
# Stage B0 profiling run: build pyscx (release) and run the GPU-vs-CPU
# pdex_nb_glm sweep with the per-phase profiler on, on the realistic SPARSE
# fixture. Produces the gpu_mle_fit / gpu_shrink_fit timings + the per-phase
# split that decides which Stage-B lever (if any) is worth building.

set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export SCX_DISABLE_CUDA_GRAPHS=1
export SCX_NBGLM_PROFILE=1

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
echo

rc=0
echo "=== build pyscx (release, hdf5,gpu) ==="
( cd pyscx && "${SCX_DIR}/.venv/bin/maturin" develop --release --features hdf5,gpu ) || rc=1
echo

echo "=== GPU correctness after L1 refactor (parity + route pytest) ==="
"${SCX_DIR}/.venv/bin/python" -m pytest -q \
    pyscx/tests/test_nb_glm.py::test_nb_glm_cpu_gpu_agreement \
    pyscx/tests/test_pdex_nb_glm.py::test_pdex_nb_glm_cpu_gpu_agreement \
    pyscx/tests/test_pdex_nb_glm.py::test_pdex_nb_glm_gpu_route_stamped || rc=1
echo

echo "=== profiled GPU-vs-CPU sweep (sparse fixture, SCX_NBGLM_PROFILE=1) ==="
"${SCX_DIR}/.venv/bin/python" pyscx/benchmarks/bench_nb_glm.py \
    --gpu on --no-pydeseq2 --reps 3 --out-name nb_glm_stage_b_profile || rc=1

echo "=== done (rc=${rc}) ==="
exit ${rc}
