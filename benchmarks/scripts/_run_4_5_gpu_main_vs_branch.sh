#!/bin/bash
# Phase-4 task 4.5 — the GPU DE number that matters: main vs branch, two builds.
#
# Both arms run at their own defaults. Deliberately **not** an env-knob A/B:
# `SCX_GPU_DE_RESIDENT=0` really is a genuine baseline here — residency simply
# is not taken and the pre-4.5 loop runs unchanged, unlike #373's depth-1 arm
# which had *zero* decode threads where main had one. But the lesson from #373
# is to prove that with two builds instead of arguing it, so the knob stays a
# kill switch and the headline comes from main vs branch.
#
# Reads three things per (dataset, op):
#
#   * wall, and the `gpu_profile` host-decode / htod / compute buckets;
#   * `resident_csr` from the route stamp — the *only* way to tell a resident
#     run from a streaming one, since the outputs are supposed to match;
#   * **VRAM**, the cost side of this change (~6 GB at census_500k). The 4.2
#     harness recorded none, which is why this uses a different profiler.
#
# `hvg` is the control: 4.5 touches neither the HVG kernels nor its two-pass
# structure (§9.14 is deferred to 4.4), so it should not move. It shares the
# staging driver, though, so a *large* hvg change would mean commit D's
# pre-sizing did something unintended rather than nothing.
#
# **Run alone** — see `_run_4_5_gpu_verify.sh`'s header for why the in-tree
# `.so` makes that a hard rule rather than a preference.
#SBATCH --job-name=scx-4.5-gpu-mainab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.5/gpu_mainab_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.5/gpu_mainab_%j.out

set -uo pipefail
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
WORK=/home/nickyoungblut/scx-bench-4.5
OUT="${WORK}/gpu_mainab_${SLURM_JOB_ID:-manual}"
WT="${WORK}/wt-gpu-mainab"
BASE=2d1fe16b
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8

run_arm() {
    local name="$1" dir="$2"
    local target="/home/nickyoungblut/.cargo-target-45ab-${name}"
    echo ""
    echo "########## ARM ${name} (${dir}) ##########"
    # Wholesale, not just $TARGET/maturin: cargo re-links a 0-byte artifact
    # from release/ and maturin dies on "Object is too small".
    rm -rf "${target}"
    ( cd "${dir}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${target}" \
        "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -2
    cd "${dir}" || return 1
    python -c "import pyscx; print('so:', pyscx.__file__)"
    # Never spend GPU hours on a binary that cannot do the measurement.
    python - <<'PY' || { echo "FATAL: gpu feature missing on arm"; return 1; }
import sys
import numpy as np, scipy.sparse as sp, anndata as ad, pyscx
a = ad.AnnData(X=sp.random(64, 16, density=0.5, format="csr", dtype=np.float32))
try:
    pyscx.accel.pca(a, n_comps=4, device="gpu")
except RuntimeError as e:
    if "without the 'gpu' feature" in str(e):
        print("PREFLIGHT FAILED:", e); sys.exit(1)
PY
    set -a; . ./.env; set +a

    # DE at the two mid tiers plus the hvg control. census_1m DE is left out of
    # the default sweep: at 123 gene chunks x 62 shards the *main* arm alone is
    # ~35 min per run, and three runs x two ops x two arms would not fit the
    # allocation. Add it back with GPU_DE_DATASETS once the mid tiers confirm
    # the shape.
    SCX_GPU_PROFILE=1 GPU_DE_RUNS=3 \
        GPU_DE_OPS=pdex_ref,wilcoxon \
        GPU_DE_DATASETS=tabula_sapiens_100k,census_500k \
        GPU_DE_OUT="${OUT}/de_${name}.json" \
        python benchmarks/scripts/profile_gpu_de_resident.py \
        2>&1 | tee "${OUT}/de_${name}.txt" | grep -vE "warn|^\s*$"

    SCX_GPU_PROFILE=1 GPU_DE_RUNS=3 \
        GPU_DE_OPS=hvg \
        GPU_DE_DATASETS=tabula_sapiens_100k,census_500k,census_1m \
        GPU_DE_OUT="${OUT}/hvg_${name}.json" \
        python benchmarks/scripts/profile_gpu_de_resident.py \
        2>&1 | tee "${OUT}/hvg_${name}.txt" | grep -vE "warn|^\s*$"
}

rm -rf "${WT}"; git -C "${SCX_DIR}" worktree prune
git -C "${SCX_DIR}" worktree add --detach "${WT}" "${BASE}" >/dev/null 2>&1 || {
    echo "FATAL: worktree add failed"; exit 1; }
# main predates this profiler; pin the harness into both arms so the *only*
# difference between them is the library.
cp "${SCX_DIR}/benchmarks/scripts/profile_gpu_de_resident.py" "${WT}/benchmarks/scripts/"
cp "${SCX_DIR}/.env" "${WT}/.env"

run_arm main   "${WT}"
run_arm branch "${SCX_DIR}"

git -C "${SCX_DIR}" worktree remove --force "${WT}" 2>/dev/null; rm -rf "${WT}"

echo ""
echo "########## SUMMARY ##########"
python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])


def load(name):
    rows = []
    for kind in ("de", "hvg"):
        p = out / f"{kind}_{name}.json"
        if p.exists():
            rows.extend(json.loads(p.read_text()))
    return {(r["dataset"], r["op"]): r for r in rows}


main, branch = load("main"), load("branch")
keys = sorted(set(main) & set(branch))
if not keys:
    print("no comparable rows — check the arm logs")
    raise SystemExit(0)

hdr = f"{'dataset':<22} {'op':<9} {'main ms':>11} {'branch ms':>11} {'x':>6}  " \
      f"{'main dec%':>9} {'br dec%':>8} {'VRAM main':>10} {'VRAM br':>9}  resident"
print(hdr)
print("-" * len(hdr))
for k in keys:
    m, b = main[k], branch[k]
    speedup = m["wall_ms"] / b["wall_ms"] if b["wall_ms"] else float("nan")
    print(
        f"{k[0]:<22} {k[1]:<9} {m['wall_ms']:11.1f} {b['wall_ms']:11.1f} "
        f"{speedup:6.2f}  {m['host_decode_over_wall'] * 100:8.1f}% "
        f"{b['host_decode_over_wall'] * 100:7.1f}% "
        f"{m['vram_peak_mb']:10.0f} {b['vram_peak_mb']:9.0f}  "
        f"{b.get('resident_csr')}"
    )
print()
print("resident_csr must be True for every DE row on the branch arm and "
      "None/False on main; a False on the branch means residency declined and "
      "the row measures something other than this change.")
PY

echo ""
echo "=== done; artifacts under ${OUT} ==="
