#!/bin/bash
# Phase 8e: does the capture guard cost anything? main vs branch, same node.
#
# The Phase 8e gate (job 2858457) reported six timing regressions against the
# `v0.11.2-multimodal-loader-fix` baseline, the largest being
# `accel_to_gpu_anndata__scx1_gpu` on tabula at +368%. That cell walks the decode
# path PR #473 touched, so it cannot be waved away — but the same report also
# regressed `accel_pca__scanpy_cpu` (+4.1%) and `accel_pca__rapids_singlecell_gpu`
# (+6.8%), which are scanpy and rapids-singlecell code that no commit in this
# series touches. That pattern points at a baseline captured on different
# hardware and library versions, not at the diff.
#
# Pointing is not measuring. This runs the SHARED variants — the ones that exist
# on both trees — on one node, in one job, with two builds, and prints the
# medians side by side. Nothing else may be submitted alongside: both arms
# rebuild the in-tree `.so` that every pyscx-importing job resolves through.
#
#SBATCH --job-name=scx8e-ab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/phase8e-ab-%j.out

set -uo pipefail

SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-8e
WT="${WORK}/wt-main"
BASE=${BASE_REF:-github/main}
BRANCH_DIR="${SCX_DIR}"

mkdir -p "${WORK}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export SCX_GPU_REQUIRE_NVCC=1
unset VIRTUAL_ENV
unset PYTHONHOME PYTHONPATH
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu

# The `main` arm runs from a worktree, which has no `.env` of its own —
# `bench_env` then raises before a single measurement. Export the paths the
# primary checkout declares, so both arms read the same fixtures.
set -a
# shellcheck disable=SC1091
source "${SCX_DIR}/.env"
set +a
export SCX_WORK_DIR="${SCX_WORK_DIR:?SCX_WORK_DIR not found in ${SCX_DIR}/.env}"
export SCX_DATA_DIR="${SCX_DATA_DIR:-${SCX_WORK_DIR}/benchmarks/datasets}"
echo "=== SCX_WORK_DIR=${SCX_WORK_DIR}"
echo "=== SCX_DATA_DIR=${SCX_DATA_DIR}"

# A worktree for the `main` arm, so the branch checkout is never disturbed.
# Its own CARGO_TARGET_DIR per arm, cleared WHOLESALE: clearing only
# $TARGET/maturin lets cargo re-link a 0-byte artifact from release/ and maturin
# dies on "Object is too small".
cd "${SCX_DIR}"
git fetch github main --quiet || true
rm -rf "${WT}"
git worktree prune
git worktree add --detach "${WT}" "${BASE}" || { echo "worktree add failed" >&2; exit 1; }
echo "=== main arm:   $(cd "${WT}" && git rev-parse --short HEAD)"
echo "=== branch arm: $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# The measurement, run identically on both arms. Deliberately NOT the SLURM
# fan-out: this is a controlled A/B on one device, and the benchmark modules'
# own `run()` is the same code the gate calls, so the numbers are comparable to
# the snapshot's while the confounders (node, driver, day) are held fixed.
read -r -d '' PROBE <<'PY'
import json, os, sys
sys.path.insert(0, os.getcwd())

from benchmarks.comprehensive.config import DATASETS, FormatVariant
from benchmarks.comprehensive.benchmarks import accel_to_gpu_anndata as tga
from benchmarks.comprehensive.benchmarks import accel_pca as pca

# Only variants that exist on BOTH trees. The shufdelta and streaming arms are
# new, so `main` has nothing to compare them against — that is the point of
# them, and it is also why they cannot appear here.
SHARED = [
    (tga, "accel_to_gpu_anndata__scx1_gpu"),
    (pca, "accel_pca__pyscx_gpu_rand_hh"),
    (pca, "accel_pca__rapids_singlecell_gpu"),
    (pca, "accel_pca__scanpy_cpu"),
]
out = {}
for mod, key in SHARED:
    for ds_name in ("pbmc3k", "tabula_sapiens_100k"):
        ds = DATASETS[ds_name]
        fv = FormatVariant(name=key, key=key, category="accel", runner="accel_runner")
        try:
            res = mod.run(dataset=ds, format_variant=fv, n_runs=5)
        except Exception as e:                                    # noqa: BLE001
            out[f"{key}__{ds_name}"] = {"error": repr(e)[:200]}
            continue
        if res is None or not getattr(res, "runs", None):
            out[f"{key}__{ds_name}"] = {"error": "no runs"}
            continue
        out[f"{key}__{ds_name}"] = {
            "median_wall_s": res.median_wall_s,
            "n": len(res.runs),
        }
print("PROBE_JSON " + json.dumps(out))
PY

run_arm() {
    local name="$1" dir="$2"
    local target="/home/nickyoungblut/.cargo-target-8eab-${name}"
    echo ""
    echo "########## ARM ${name} (${dir}) ##########"
    rm -rf "${target}"
    ( cd "${dir}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${target}" \
        "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -3
    ( cd "${dir}" && CARGO_TARGET_DIR="${target}" cargo build --release -p scx-cli ) 2>&1 | tail -2
    export SCX_CLI_BIN="${target}/release/scx"
    test -x "${SCX_CLI_BIN}" || { echo "FATAL: no scx CLI on arm ${name}" >&2; return 1; }
    cd "${dir}" || return 1
    python -c "import pyscx; print('so:', pyscx.__file__)"
    python -c "$PROBE" 2>&1 | tee "${WORK}/${name}.log" | grep -v PROBE_JSON
    grep -h PROBE_JSON "${WORK}/${name}.log" | sed 's/^PROBE_JSON //' > "${WORK}/${name}.json"
}

run_arm main   "${WT}"          || echo "main arm failed"
run_arm branch "${BRANCH_DIR}"  || echo "branch arm failed"

echo ""
echo "########## A/B ##########"
python - "${WORK}/main.json" "${WORK}/branch.json" <<'PY'
import json, sys
main = json.load(open(sys.argv[1]))
branch = json.load(open(sys.argv[2]))
keys = sorted(set(main) | set(branch))
w = max(len(k) for k in keys)
print(f"{'cell'.ljust(w)}  {'main':>10}  {'branch':>10}  {'delta':>9}")
worst = 0.0
for k in keys:
    m, b = main.get(k, {}), branch.get(k, {})
    if "error" in m or "error" in b:
        print(f"{k.ljust(w)}  {m.get('error', m.get('median_wall_s'))!s:>10}  "
              f"{b.get('error', b.get('median_wall_s'))!s:>10}  {'—':>9}")
        continue
    mm, bb = m["median_wall_s"], b["median_wall_s"]
    d = (bb - mm) / mm * 100.0
    worst = max(worst, d)
    print(f"{k.ljust(w)}  {mm:10.4f}  {bb:10.4f}  {d:+8.1f}%")
print(f"\nworst branch-vs-main delta: {worst:+.1f}%")
PY

cd "${SCX_DIR}"
git worktree remove --force "${WT}" 2>/dev/null || true
echo "=== done"
