#!/bin/bash
# The gate owed by Organization Phases 8c and 8d, in one run.
#
# Both were recorded as owed with the same instruction — "run both arms and
# label every number with its arm" — and neither was runnable as written:
#
#   * 8d asked for the nvcomp arm alongside the pipelined one. But
#     `accel_to_gpu_anndata` had ONE variant, built by `scx optimize --codec
#     scx1`, and Scx1 shards never enter the ShufDeltaZstd decode paths where
#     `nvcomp_enabled()` is consulted. `SCX_SHUFDELTA_NVCOMP=1` against it is a
#     literal no-op: both "arms" would have reported byte-identical work under
#     different names. The decode path is a property of the FILE, so the two
#     ShufDeltaZstd arms now carry their own fixture.
#   * 8c asked for the streaming PCA arm. Residency is decided against FREE
#     VRAM and every gate dataset fits an 80 GB H100, so `accel_pca` measured
#     the resident arm on all three tiers and the streaming power loop — the one
#     8c's `PcaOperator` seam rewrote — had no coverage at all.
#
# ONE job, and nothing else submitted alongside: `maturin develop` rewrites the
# in-tree `.so` that every pyscx-importing job resolves through, so a second scx
# job overlapping this one would measure a binary neither of us intended.
#
#SBATCH --job-name=scx8e-gate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nickyoungblut/dev/rust/scx/benchmarks/comprehensive/logs/phase8e-gate-%j.out

set -euo pipefail

# `sbatch` copies the batch script into slurmd's spool and runs it from there, so
# `${BASH_SOURCE[0]}` is /var/spool/slurmd/job<N>/slurm_script. `dirname/../..`
# is then /var/spool — which EXISTS, so `cd` succeeds and a naive `||` fallback
# never fires. Validate the candidate rather than trusting the exit status.
REPO_FALLBACK=/home/nickyoungblut/dev/rust/scx
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd || true)
if [ ! -f "${REPO:-/nonexistent}/pyscx/Cargo.toml" ]; then
  REPO=$REPO_FALLBACK
fi
if [ ! -f "$REPO/pyscx/Cargo.toml" ]; then
  echo "cannot locate the scx checkout (tried \"$REPO\"); set REPO_FALLBACK" >&2
  exit 1
fi
cd "$REPO"
echo "=== repo: $REPO"

# sbatch exports the submitting shell's environment, and this repo's `.venv`
# sets VIRTUAL_ENV. maturin refuses outright when both VIRTUAL_ENV and
# CONDA_PREFIX are set ("Please unset one of them") and the job dies in seconds,
# before the build. Unset here rather than trusting the submitting shell.
unset VIRTUAL_ENV
unset PYTHONHOME PYTHONPATH

CONDA_SH=$( { conda info --base 2>/dev/null || echo /home/nickyoungblut/miniforge3; } )/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate scx-bench-gpu

echo "=== HEAD: $(git rev-parse HEAD) on $(git rev-parse --abbrev-ref HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- build, release: a debug .so runs 4-10x slower and poisons every timing ---
#
# Whether nvcc exists is NOT part of scx-gpu's build-script fingerprint, so a
# target directory that once saw a CPU-only build replays the stub branch on a
# machine that has nvcc. `touch` makes the build script rerun; the env var makes
# a missing nvcc a 20-second build failure instead of a runtime symptom four
# steps from its cause.
export SCX_GPU_REQUIRE_NVCC=1
export PATH=/usr/local/cuda/bin:${PATH}
touch scx-gpu/build.rs
echo "=== maturin develop --release --features hdf5,gpu"
( cd pyscx && maturin develop --release --features hdf5,gpu )

# The `scx` CLI, because `accel_to_gpu_anndata` prepares every arm's fixture with
# `scx optimize --codec <arm>`. Without a binary at target/release/scx it falls
# back to an h5ad self-convert, which produces an UNFRAMED shufdelta file with no
# block index — the GPU framed decode is then never reached and both shufdelta
# arms measure the host bounce. The floors would catch that, hours later.
echo "=== cargo build --release -p scx-cli"
cargo build --release -p scx-cli
test -x target/release/scx || { echo "scx CLI did not build" >&2; exit 1; }

# --- preflight the arms, not the ops -----------------------------------------
#
# The usual rule is to run one real GPU op before spending hours. Here the thing
# that can silently not-happen is not an op, it is an ARM: nvcomp is
# runtime-dlopen'd and falls back to the CPU-zstd pipeline without a word when
# libnvcomp.so.5 is absent, and SCX_GPU_PCA_RESIDENT is read once per process.
# Either failure produces a complete, plausible, green gate in which both arms
# are the same arm. Assert the separation here, in minutes.
echo "=== preflight: do the arms actually separate?"
python - <<'PY'
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata
import numpy as np
import scipy.sparse as sp

import pyscx

fail = []
REPO = Path.cwd()
SCX = REPO / "target" / "release" / "scx"

# ---------------------------------------------------------------- nvcomp arm
# Build a framed ShufDeltaZstd file the same way the benchmark does, then decode
# it twice: once with the env off (Phase 1.5 pipelined) and once on (Phase 2
# nvcomp). `transfer_mode` must DIFFER. If it does not, either nvcomp is not
# loadable here or the arm wiring is broken, and every shufdelta number this job
# is about to produce would be the same arm twice.
with tempfile.TemporaryDirectory() as td:
    rng = np.random.default_rng(8)
    X = rng.poisson(1.2, size=(3000, 400)).astype(np.float32)
    src = Path(td) / "src.scx"
    framed = Path(td) / "framed.scx"
    pyscx.from_anndata(anndata.AnnData(X=sp.csr_matrix(X)), str(src))
    subprocess.run(
        [str(SCX), "optimize", str(src), str(framed),
         "--codec", "shufdelta", "--row-group-rows", "256"],
        check=True, capture_output=True, timeout=600,
    )

    def decode(nvcomp: bool) -> dict:
        prev = os.environ.get("SCX_SHUFDELTA_NVCOMP")
        os.environ["SCX_SHUFDELTA_NVCOMP"] = "1" if nvcomp else "0"
        try:
            ad = pyscx.open(str(framed)).to_gpu_anndata(device="gpu")
            return dict(ad.uns["scx_accel"]["to_gpu_anndata"])
        finally:
            if prev is None:
                os.environ.pop("SCX_SHUFDELTA_NVCOMP", None)
            else:
                os.environ["SCX_SHUFDELTA_NVCOMP"] = prev

    try:
        off = decode(False)
        on = decode(True)
    except Exception as e:  # noqa: BLE001
        fail.append(f"shufdelta decode raised: {e!r}")
        off = on = {}

    print(f"preflight: pipelined arm transfer_mode={off.get('transfer_mode')!r} "
          f"n_shards_shufdelta_gpu={off.get('n_shards_shufdelta_gpu')}")
    print(f"preflight: nvcomp    arm transfer_mode={on.get('transfer_mode')!r} "
          f"n_shards_shufdelta_gpu={on.get('n_shards_shufdelta_gpu')}")

    if not (off.get("n_shards_shufdelta_gpu") or 0) >= 1:
        fail.append(
            "the pipelined arm took no GPU ShufDeltaZstd path at all "
            "(n_shards_shufdelta_gpu == 0) — the fixture is probably unframed, "
            "so both arms would measure the host bounce"
        )
    if off.get("transfer_mode") == on.get("transfer_mode"):
        fail.append(
            f"BOTH ARMS ARE THE SAME ARM: transfer_mode={off.get('transfer_mode')!r} "
            "with SCX_SHUFDELTA_NVCOMP off and on. Either libnvcomp.so.5 is not "
            "loadable in this env (check CONDA_PREFIX/lib) or the arm wiring is "
            "broken. Every shufdelta number below would be a duplicate."
        )
    elif on.get("transfer_mode") != "scx_device_decode_gpu":
        fail.append(
            f"the nvcomp arm reports {on.get('transfer_mode')!r}, expected "
            "'scx_device_decode_gpu' (fully_device_decoded)"
        )

# ------------------------------------------------------------- PCA arm
# In a SUBPROCESS, because SCX_GPU_PCA_RESIDENT is read once per process and this
# one has already run GPU work. The benchmark's own arms are separate SLURM jobs
# for the same reason; here the subprocess is what makes the probe honest.
probe = r'''
import numpy as np, scipy.sparse as sp, anndata, pyscx, json, sys
rng = np.random.default_rng(11)
X = rng.poisson(1.0, size=(4000, 300)).astype(np.float32)
ad = anndata.AnnData(X=sp.csr_matrix(X))
pyscx.accel.pca(ad, n_comps=20, device="gpu", method="randomized",
                qr_method="householder", random_state=0)
info = ad.uns.get("scx_accel", {}).get("pca", {})
print(json.dumps({"route": info.get("route"),
                  "resident_csr": info.get("resident_csr")}))
'''
arms = {}
for label, resident in (("resident", "1"), ("streaming", "0")):
    env = dict(os.environ)
    env["SCX_GPU_PCA_RESIDENT"] = resident
    env["SCX_FORCE_NATIVE_GPU"] = "1"
    p = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                       text=True, env=env, timeout=1200)
    if p.returncode != 0:
        fail.append(f"PCA {label} arm probe failed: {p.stderr[-300:]}")
        continue
    import json as _json
    arms[label] = _json.loads(p.stdout.strip().splitlines()[-1])
    print(f"preflight: PCA {label:9s} arm route={arms[label]['route']!r} "
          f"resident_csr={arms[label]['resident_csr']}")

if len(arms) == 2:
    if arms["resident"]["resident_csr"] is not True:
        fail.append(
            f"the resident arm reports resident_csr="
            f"{arms['resident']['resident_csr']!r}, expected True — the arm this "
            "gate compares against is not the resident one"
        )
    if arms["streaming"]["resident_csr"] is not False:
        fail.append(
            f"BOTH PCA ARMS MAY BE THE SAME ARM: SCX_GPU_PCA_RESIDENT=0 gave "
            f"resident_csr={arms['streaming']['resident_csr']!r}, expected False"
        )
    for label in arms:
        route = str(arms[label]["route"] or "")
        if not route.startswith("gpu_"):
            fail.append(f"PCA {label} arm routed to {route!r}, not a native GPU route")

if fail:
    print("\nPREFLIGHT FAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("\npreflight OK: both arm pairs separate, on this host, with this build")
PY

# --- the gate ----------------------------------------------------------------
#
# One invocation covering both owed gates. `--accel-only` skips the format
# benchmarks; `--skip-smoke` skips the additional-format smoke test, which needs
# no coverage here.
#
# ⚠️ These three variants have NO rows in LATEST, so compare_against_baseline
# treats them as appearing benchmarks and reports them informationally. This gate
# therefore checks ABSOLUTE FLOORS ONLY and says nothing about regressions. Read
# the `[baseline] archived N result files` line before drawing any conclusion —
# a snapshot that archived nothing skips every floor and exits 0.
#
# ⚠️ `--probe-partition gpu`, not the `preemptible` default. The first attempt
# (job 2858357) died here: the pre-flight `gpu_probe` sat PENDING on
# `preemptible` for its full 600 s budget and the gate exited 2 having measured
# nothing, after paying for the wheel and CLI builds. The preemptible GPU queue
# can starve for the better part of a day; `gpu` is the partition for short
# jobs, and this probe is a ~30 s one.
echo "=== gate_candidate.py"
set +e
python benchmarks/comprehensive/scripts/gate_candidate.py \
    --accel-only \
    --benchmarks accel_to_gpu_anndata accel_pca \
    --datasets pbmc3k tabula_sapiens_100k \
    --name phase8e-arms-gate \
    --probe-partition gpu \
    --probe-timeout 900 \
    --skip-smoke \
    -v
GATE_RC=$?
set -e
echo "=== gate exit code: ${GATE_RC}"

# --- did it measure anything? ------------------------------------------------
SNAP=$(ls -dt benchmarks/comprehensive/results/phase8e-arms-gate* 2>/dev/null | head -1 || true)
if [ -n "$SNAP" ]; then
  echo "=== snapshot: $SNAP"
  echo "=== raw/ result files: $(find "$SNAP/raw" -name '*.json' 2>/dev/null | wc -l)"
  echo "=== per-variant rows:"
  find "$SNAP/raw" -name '*.json' 2>/dev/null | sed 's#.*/##' | sort | sed 's/^/    /'
else
  echo "=== NO SNAPSHOT DIRECTORY — the gate measured nothing" >&2
fi

exit "${GATE_RC}"
