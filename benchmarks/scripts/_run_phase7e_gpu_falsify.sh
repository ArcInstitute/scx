#!/bin/bash
# Phase 7e: falsify the claim that `test_gpu_vs_cpu_per_pc_correlation` catches
# an M-step present on ONE arm only.
#
# That claim was asserted in a commit message before it was checked. It is the
# reason `gpu_m_step_matches_harmonypy` pins the device kernels *directly*
# rather than leaning on CPU/GPU parity — but the claim itself still has to be
# true, or the argument for where the M-step lives is built on nothing.
#
# Runs in a throwaway git worktree with its own CARGO_TARGET_DIR so the primary
# checkout is untouched: this job may overlap with the author editing the tree.
# It runs NO `maturin develop`, so it does not disturb the in-tree `.so`.

set -uo pipefail
export SCX_REQUIRE_GPU=1 SCX_GPU_REQUIRE_NVCC=1
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

SCX_DIR=/home/nickyoungblut/dev/rust/scx
WT=$(mktemp -d /tmp/scx7e-falsify-XXXXXX)
export CARGO_TARGET_DIR="$WT/target"
trap 'cd /; git -C "$SCX_DIR" worktree remove --force "$WT/src" 2>/dev/null; rm -rf "$WT"' EXIT

echo "=== node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
git -C "$SCX_DIR" worktree add -q --detach "$WT/src" HEAD || exit 1
cd "$WT/src" || exit 1
echo "=== worktree at $(git rev-parse --short HEAD)"

T="harmony::cpu::tests::test_gpu_vs_cpu_per_pc_correlation"

echo "=== [baseline] both arms have the M-step -- must PASS"
cargo test -p scx-accel --features gpu --release --lib "$T" \
    -- --include-ignored --nocapture --test-threads=1 2>&1 | tail -4
base=${PIPESTATUS[0]}

echo
echo "=== [mutant] remove the M-step from the GPU arm ONLY -- must FAIL"
python3 - <<'PY'
import sys, pathlib
p = pathlib.Path("scx-accel/src/harmony/gpu.rs")
s = p.read_text()
old = """            gpu_harmony_update_y(&dev, &cublas, &d_z_cos, &d_r, &mut d_y, d, k, n)
                .map_err(|e| AccelError::LinAlg(format!("GPU M-step (Y = Z_cos R'): {e}")))?;
            gpu_harmony_l2_normalize_cols(&dev, &mut d_y, d, k)
                .map_err(|e| AccelError::LinAlg(format!("GPU L2 normalize Y: {e}")))?;
            dispatch_harmony_distances(&dev, &cublas, &d_y, &d_z_cos, &mut d_dist, d, k, n)?;
"""
if old not in s:
    sys.exit("MUTATION DID NOT APPLY -- the GPU M-step block is not where this "
             "script expects it; the falsification would be meaningless")
p.write_text(s.replace(old, "", 1))
print("mutation applied: GPU M-step removed, CPU M-step intact")
PY
[ $? -ne 0 ] && exit 1

cargo test -p scx-accel --features gpu --release --lib "$T" \
    -- --include-ignored --nocapture --test-threads=1 2>&1 | tail -6
mut=${PIPESTATUS[0]}

echo
echo "=== VERDICT: baseline_rc=$base mutant_rc=$mut"
if [ "$base" -eq 0 ] && [ "$mut" -ne 0 ]; then
    echo "CONFIRMED: CPU/GPU per-PC correlation reds when the M-step is on one arm only."
    exit 0
fi
if [ "$base" -ne 0 ]; then
    echo "INCONCLUSIVE: the baseline itself failed; nothing can be concluded from the mutant."
    exit 1
fi
echo "REFUTED: the mutant PASSED. CPU/GPU parity does NOT catch a one-armed M-step,"
echo "so the commit message claiming it does is wrong and must be corrected."
exit 1
