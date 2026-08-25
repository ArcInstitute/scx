#!/bin/bash
# Phase 7e follow-up: can ANY per-PC correlation bar separate a GPU arm that
# has the M-step from one that does not?
#
# Job 2840979 refuted the claim that `test_gpu_vs_cpu_per_pc_correlation`
# catches a one-armed M-step: the mutant passed at r >= 0.95. This measures the
# actual correlations in both configurations so the answer is a number rather
# than a second guess. Two possible outcomes, both worth having:
#
#   * a gap exists  -> tighten the bar, and the test becomes a real guard;
#   * no gap exists -> say so. A parity test between two SCX arms is then
#     structurally unable to see this, which is precisely why
#     `gpu_m_step_matches_harmonypy` pins the device kernels against harmonypy.
#
# Throwaway worktree + private CARGO_TARGET_DIR; no `maturin develop`.

set -uo pipefail
export SCX_REQUIRE_GPU=1 SCX_GPU_REQUIRE_NVCC=1
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

SCX_DIR=/home/nickyoungblut/dev/rust/scx
WT=$(mktemp -d /tmp/scx7e-measure-XXXXXX)
export CARGO_TARGET_DIR="$WT/target"
trap 'cd /; git -C "$SCX_DIR" worktree remove --force "$WT/src" 2>/dev/null; rm -rf "$WT"' EXIT

echo "=== node: $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader
git -C "$SCX_DIR" worktree add -q --detach "$WT/src" HEAD || exit 1
cd "$WT/src" || exit 1

# A temporary test that PRINTS the per-PC correlations instead of asserting a
# bar. Appended to the existing test module so it sees the same helpers and the
# same fixture as `test_gpu_vs_cpu_per_pc_correlation`.
cat >> scx-accel/src/harmony/tests.rs <<'RS'

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn zz_measure_per_pc_correlation() {
    require_gpu_or_skip!();
    let n_per = 80;
    let d = 5;
    let (emb, labels) = batched_gaussian(n_per, d, 123);
    let n = emb.len() / d;
    let cov = BatchCovariate { labels, n_levels: 2, name: None };
    let config = HarmonyConfig {
        n_clusters: Some(4), max_iter: 3, random_state: 11, ..Default::default()
    };
    let cpu = harmony_integrate(&emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    let gpu = harmony_integrate_gpu(0, &emb, n, d, std::slice::from_ref(&cov), &config).unwrap();
    let mut worst = 1.0f64;
    for pc in 0..d {
        let a: Vec<f64> = (0..n).map(|i| cpu.z_corrected[i * d + pc]).collect();
        let b: Vec<f64> = (0..n).map(|i| gpu.z_corrected[i * d + pc]).collect();
        let ma = a.iter().sum::<f64>() / n as f64;
        let mb = b.iter().sum::<f64>() / n as f64;
        let (mut sab, mut sa, mut sb) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let (x, y) = (a[i] - ma, b[i] - mb);
            sab += x * y; sa += x * x; sb += y * y;
        }
        let r = sab / (sa.sqrt() * sb.sqrt());
        println!("PERPC pc={pc} r={r:.9}");
        if r < worst { worst = r; }
    }
    println!("PERPC worst={worst:.9}");
}
RS

T="harmony::cpu::tests::zz_measure_per_pc_correlation"
echo
echo "=== [A] both arms have the M-step"
cargo test -p scx-accel --features gpu --release --lib "$T" \
    -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "PERPC|test result"

echo
echo "=== [B] GPU M-step removed, CPU M-step intact"
python3 - <<'PY'
import sys, pathlib
p = pathlib.Path("scx-accel/src/harmony/gpu.rs"); s = p.read_text()
old = """            gpu_harmony_update_y(&dev, &cublas, &d_z_cos, &d_r, &mut d_y, d, k, n)
                .map_err(|e| AccelError::LinAlg(format!("GPU M-step (Y = Z_cos R'): {e}")))?;
            gpu_harmony_l2_normalize_cols(&dev, &mut d_y, d, k)
                .map_err(|e| AccelError::LinAlg(format!("GPU L2 normalize Y: {e}")))?;
            dispatch_harmony_distances(&dev, &cublas, &d_y, &d_z_cos, &mut d_dist, d, k, n)?;
"""
if old not in s:
    sys.exit("MUTATION DID NOT APPLY -- measurement B would be a duplicate of A")
p.write_text(s.replace(old, "", 1)); print("mutation applied")
PY
[ $? -ne 0 ] && exit 1
cargo test -p scx-accel --features gpu --release --lib "$T" \
    -- --include-ignored --nocapture --test-threads=1 2>&1 | grep -E "PERPC|test result"
echo
echo "=== compare the two 'worst=' lines. The current bar is r >= 0.95."
