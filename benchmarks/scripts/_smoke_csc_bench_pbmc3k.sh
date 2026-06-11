#!/bin/bash
# Smoke test that the bench fixture actually exercises the CSC-direct
# dispatch path on a single small dataset (pbmc3k). Runs the
# accel-only gate restricted to pbmc3k with:
#   - SCX_BENCH_REBUILD_CSC=1 — force a fresh build of the temp CSC
#                              SCX fixture (cached otherwise)
#
# GPU DE v3 is the unconditional default route, so a CSC-sidecar fixture takes
# the gpu_csc_v3 route. Success/failure is read from the recorded route
# metadata in the result JSON (`runs[].extra.gpu_dispatch_route` /
# `de_route_csc_direct`), not from a stderr trace.
#   Success: gpu_dispatch_route == gpu_csc_v3  (de_route_csc_direct == 1.0)
#   Failure: any other route (e.g. gpu_csr_v3 → CSC sidecar not used)

set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

source /home/nickyoungblut/miniforge3/etc/profile.d/conda.sh

# Rebuild pyscx so the worker sees the freshly-compiled scx-accel. Editable
# .so is shared across envs; build under scx-bench-gpu for clarity.
conda activate scx-bench-gpu
echo "=== rebuild pyscx --features gpu in scx-bench-gpu ==="
cd "${SCX_DIR}/pyscx"
"${SCX_DIR}/.venv/bin/maturin" develop --release --features gpu --quiet
cd "${SCX_DIR}"
python -c "
import pyscx
info = pyscx.accel.gpu_info()
assert info, 'pyscx missing GPU features'
print('pyscx gpu_info OK:', info['device'])
"

# Orchestrator runs from scx-bench per workspace convention.
conda deactivate
conda activate scx-bench
set -a
source .env
set +a

export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo
echo "=== run accel-only gate on pbmc3k (v3 CSC-direct is the default route) ==="
START_TS=$(date +%s)
SCX_BENCH_REBUILD_CSC=1 \
python "${SCX_DIR}/benchmarks/comprehensive/scripts/gate_candidate.py" \
    --name "g4_de_v3_csc_smoke_pbmc3k" \
    --tier small \
    --accel-only \
    --skip-preflight \
    --datasets pbmc3k \
    --benchmarks accel_de \
    2>&1 | tee "${SCX_DIR}/benchmarks/scripts/g4_de_v3_csc_smoke_pbmc3k.log"

echo
echo "=== checking recorded dispatch route in the pdex_ref GPU result JSON ==="
# The accel_de pdex_ref GPU bench stamps the route it actually took onto the
# result JSON (runs[].extra.gpu_dispatch_route / de_route_csc_direct), written
# to the canonical results/raw/ dir. Read that instead of grepping stderr.
RAW_DIR="${SCX_DIR}/benchmarks/comprehensive/results/raw"
python - "${RAW_DIR}" "${START_TS}" <<'PY'
import glob, json, os, sys

raw_dir, start_ts = sys.argv[1], int(sys.argv[2])
pattern = os.path.join(raw_dir, "accel_de__*pdex_ref_gpu*__pbmc3k.json")
# Only accept results written by this run (mtime at/after the gate started).
fresh = [p for p in glob.glob(pattern) if os.path.getmtime(p) >= start_ts - 1]
if not fresh:
    print(f"FAIL: no fresh pdex_ref GPU result JSON matching {pattern}")
    print("      (did the GPU accel_de job run and write a result?)")
    sys.exit(1)

path = max(fresh, key=os.path.getmtime)
with open(path) as f:
    result = json.load(f)
runs = result.get("runs", [])
routes = sorted({r.get("extra", {}).get("gpu_dispatch_route") for r in runs} - {None})
csc_direct = [r.get("extra", {}).get("de_route_csc_direct") for r in runs]

print(f"result : {os.path.basename(path)}")
print(f"route(s): {routes}")
print(f"de_route_csc_direct: {csc_direct}")

csc_ok = csc_direct and all(v == 1.0 for v in csc_direct if v is not None)
if "gpu_csc_v3" in routes and csc_ok:
    print("PASS: pdex_ref took the CSC-direct route (gpu_csc_v3)")
    sys.exit(0)
print("FAIL: pdex_ref did NOT take the CSC-direct route (CSC sidecar not used)")
sys.exit(1)
PY
RC=$?

echo
echo "=== done (route check exit=${RC}) ==="
exit "${RC}"
