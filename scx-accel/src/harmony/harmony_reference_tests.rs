//! Harmony pinned against harmonypy 0.2.0 (§7.4, ORG-7.21-4).
//!
//! Values and provenance live in [`super::harmony_reference_values`]; this file
//! holds only the assertions. Every expected number was produced by harmonypy —
//! see that module's `# Provenance` table — and every fixture value is
//! harmonypy's own `init_cluster` output, so there is no step where SCX supplies
//! an input that SCX also grades.
//!
//! The load-bearing arm is `the_clustering_sub_loop_runs_the_m_step`, not the
//! unit arm beside it. Before Phase 7e `update_y` did not exist:
//! `cluster_iteration` called `update_r` in a loop with `y` and `dist_mat`
//! frozen, so soft k-means could never move a centroid off a batch-driven mode.
//! Falsification settled which arm carries that: deleting `self.update_y()`
//! from the sub-loop while keeping the function leaves every *unit* arm green,
//! because they call it directly. §7.4 is stated as a number by the sub-loop
//! arm and by nothing else.
//!
//! (Plain backticks rather than intra-doc links throughout: rustdoc compiles
//! `#[test]` items out, so a link to one can never resolve — and this whole
//! module is `#[cfg(test)]`, which rustdoc does not process at all.)

use super::harmony_reference_values as r;
use super::*;

/// Build a state whose clustering fields are harmonypy's `init_cluster` output.
///
/// `HarmonyState::new` is used for everything derived from the *inputs* —
/// layout, `cell_to_gb`, `n_b`, `pr_b`, the theta expansion, `lambda_fixed` —
/// and then the five clustering buffers are overwritten with the fixture. That
/// split is deliberate: re-deriving the layout here would put a second copy of
/// it in the test, which is the mistake this phase exists to remove.
fn fixture_state(theta: f64, block_size: f64, max_iter_kmeans: usize) -> HarmonyState {
    let emb: Vec<f32> = r::HP_Z
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    let cov = BatchCovariate {
        labels: r::HP_LABELS.to_vec(),
        n_levels: r::HP_N_BATCHES,
        name: None,
    };
    let config = HarmonyConfig {
        n_clusters: Some(r::HP_N_CLUSTERS),
        theta: Some(vec![theta]),
        sigma: r::HP_SIGMA,
        lambda: Some(vec![r::HP_LAMB]),
        block_size,
        max_iter_kmeans,
        tau: 0.0,
        ..Default::default()
    };
    let mut s =
        HarmonyState::new(&emb, r::HP_N_CELLS, r::HP_N_PCS, &[cov], &config).expect("fixture");

    // y: (K rows of d) -> column-major d x K, i.e. y[ku*d + t].
    s.y = r::HP_FIX_Y.iter().flat_map(|c| c.iter().copied()).collect();
    // r / dist_mat: (K rows of N) -> row-major K x N, f32 as SCX stores them.
    s.r = r::HP_FIX_R
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    s.dist_mat = r::HP_FIX_DIST
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    // o / e: (K rows of B) -> row-major K x B.
    s.o = r::HP_FIX_O
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    s.e = r::HP_FIX_E
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    s
}

/// max |got - want| over a flattened pair, reported with the offending index.
fn max_abs(got: &[f64], want: &[f64]) -> (f64, usize) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    let mut worst = 0.0;
    let mut at = 0;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    (worst, at)
}

fn flat(m: &[[f64; r::HP_N_CELLS]]) -> Vec<f64> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

// ─── Premises ────────────────────────────────────────────────────────

/// The shapes the tables are declared at must be the shapes the fixture claims.
/// A silent shape change here would make every arm below compare the wrong
/// elements while still passing on length.
#[test]
fn every_table_has_the_shape_pinned_beside_it() {
    assert_eq!(r::HP_Z.len(), r::HP_N_CELLS);
    assert_eq!(r::HP_Z[0].len(), r::HP_N_PCS);
    assert_eq!(r::HP_LABELS.len(), r::HP_N_CELLS);
    assert_eq!(r::HP_FIX_Y.len(), r::HP_N_CLUSTERS);
    assert_eq!(r::HP_FIX_Y[0].len(), r::HP_N_PCS);
    assert_eq!(r::HP_FIX_R.len(), r::HP_N_CLUSTERS);
    assert_eq!(r::HP_FIX_O.len(), r::HP_N_CLUSTERS);
    assert_eq!(r::HP_FIX_O[0].len(), r::HP_N_BATCHES);
    assert_eq!(r::HP_MSTEP_Y.len(), r::HP_N_CLUSTERS);
    assert_eq!(r::HP_RIDGE_Z_CORR.len(), r::HP_N_CELLS);
    assert_eq!(r::HP_RIDGE_Z_CORR[0].len(), r::HP_N_PCS);
}

/// The M-step is only a test if the fixture's centroids are *not* already the
/// R-weighted means. The generator refuses a fixture that fails this; assert it
/// on the emitted literals too, because the generator's check and the literals
/// can drift if someone edits the file by hand.
#[test]
fn the_m_step_target_is_not_where_the_fixture_already_is() {
    let fix: Vec<f64> = r::HP_FIX_Y.iter().flat_map(|c| c.iter().copied()).collect();
    let want: Vec<f64> = r::HP_MSTEP_Y
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let (moved, _) = max_abs(&fix, &want);
    assert!(
        moved > 1e-2,
        "harmonypy's M-step moved the fixture centroids by only {moved:.3e}; \
         an implementation with no M-step would pass the arm below"
    );
}

/// SCX never materializes a `Z_cos` buffer — `compute_inv_norms` normalizes per
/// cell inside `compute_distances` and `update_y` (the H9 note in `cpu.rs`). So
/// the cosine normalization every arm below depends on is implicit, and nothing
/// checked it against the reference's. harmonypy's `_Z_cos` is
/// `Z / ||Z||` per column, computed in f32.
#[test]
fn the_implicit_cosine_normalization_matches_harmonypys() {
    let s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    let inv = compute_inv_norms(&s.z_corr, s.d, s.n);
    let mut got = Vec::with_capacity(s.n * s.d);
    for i in 0..s.n {
        for t in 0..s.d {
            got.push(s.z_corr[i * s.d + t] * inv[i]);
        }
    }
    let want: Vec<f64> = r::HP_FIX_Z_COS
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let (d, at) = max_abs(&got, &want);
    assert!(
        d <= r::HP_MSTEP_Y_ATOL,
        "Z_cos element {at}: |delta| {d:.3e} > {:.3e}",
        r::HP_MSTEP_Y_ATOL
    );
}

// The device bar must not be TIGHTER than the host bar. cuBLAS accumulates the
// M-step's product in f32 where the CPU arm accumulates in f64, so a GPU bar
// below the CPU one would be claiming the device is more accurate than the
// host — and would red on hardware nobody runs in CI, which is the worst place
// to discover a mis-set tolerance.
//
// A compile-time assertion, not a `#[test]`: both sides are `const`, so clippy
// rightly calls a runtime `assert!` on them constant-valued. This fails the
// BUILD, which is stronger and also keeps `HP_GPU_MSTEP_Y_ATOL` used in a
// non-`gpu` build, where nothing else reads it.
const _: () = assert!(r::HP_GPU_MSTEP_Y_ATOL >= r::HP_MSTEP_Y_ATOL);

// ─── Arm 1: the M-step, and the distances it invalidates ─────────────

/// **This is §7.4, and it is the arm that pins the CALL SITE.**
///
/// `cluster_iteration` at `max_iter_kmeans = 1` is exactly harmonypy's
/// `cluster()` at the same setting: M-step, then the distances it invalidates,
/// then `update_R`, then the objective. `update_r` mutates `r` / `o` / `e` but
/// never `y` or `dist_mat`, so both are still the M-step's output when the call
/// returns.
///
/// The unit arm below pins `update_y` itself. This one pins that the sub-loop
/// *runs* it — deleting the call and keeping the function would leave the unit
/// arm green, which is the gap that made 7d's first `distances.rs` falsification
/// pass against reverted code.
#[test]
fn the_clustering_sub_loop_runs_the_m_step() {
    let mut s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    s.cluster_iteration(true);

    let want_y: Vec<f64> = r::HP_MSTEP_Y
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let (dy, at_y) = max_abs(&s.y, &want_y);
    assert!(
        dy <= r::HP_MSTEP_Y_ATOL,
        "after one sub-iteration, centroid {at_y}: |delta| {dy:.3e} > {:.3e} — \
         the sub-loop is not applying the M-step",
        r::HP_MSTEP_Y_ATOL
    );

    let got: Vec<f64> = s.dist_mat.iter().map(|&v| v as f64).collect();
    let (dd, at_d) = max_abs(&got, &flat(&r::HP_MSTEP_DIST));
    assert!(
        dd <= r::HP_MSTEP_DIST_ATOL,
        "after one sub-iteration, distance {at_d}: |delta| {dd:.3e} > {:.3e} — \
         the sub-loop is not recomputing distances from the moved centroids",
        r::HP_MSTEP_DIST_ATOL
    );
}

/// The M-step in isolation. `update_y` did not exist before Phase 7e.
#[test]
fn the_m_step_moves_centroids_onto_the_r_weighted_means() {
    let mut s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    s.update_y();
    let want: Vec<f64> = r::HP_MSTEP_Y
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let (d, at) = max_abs(&s.y, &want);
    assert!(
        d <= r::HP_MSTEP_Y_ATOL,
        "M-step centroid {at}: |delta| {d:.3e} > {:.3e}",
        r::HP_MSTEP_Y_ATOL
    );
}

/// The cosine-distance kernel the M-step feeds, on the M-step's own output.
/// This is the accept side: it was already correct, and must stay so.
#[test]
fn the_distances_follow_the_moved_centroids() {
    let mut s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    s.update_y();
    let got = compute_distances(&s.y, &s.z_corr, s.d, s.k, s.n);
    let got64: Vec<f64> = got.iter().map(|&v| v as f64).collect();
    let (d, at) = max_abs(&got64, &flat(&r::HP_MSTEP_DIST));
    assert!(
        d <= r::HP_MSTEP_DIST_ATOL,
        "distance {at}: |delta| {d:.3e} > {:.3e}",
        r::HP_MSTEP_DIST_ATOL
    );
}

// ─── Arm 2: the ridge correction ─────────────────────────────────────

/// SCX's arrowhead inverse against harmonypy's `torch.linalg.inv`, on a fixture
/// where no batch is pruned (the generator checks that), so the two are solving
/// the same system rather than SCX solving a smaller one.
#[test]
fn the_ridge_correction_matches_moe_correct_ridge() {
    let mut s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    s.correct().expect("ridge solve");
    let want: Vec<f64> = r::HP_RIDGE_Z_CORR
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    let (d, at) = max_abs(&s.z_corr, &want);
    assert!(
        d <= r::HP_RIDGE_ATOL,
        "z_corr {at}: |delta| {d:.3e} > {:.3e}",
        r::HP_RIDGE_ATOL
    );
}

/// `moe_correct_ridge` computes `W` and zeroes its intercept row; it never
/// writes `Y`. SCX used to copy that intercept row into `y` before zeroing it —
/// the stand-in for the missing M-step, a centroid written once per *outer*
/// iteration as a by-product of the correction solve.
///
/// This arm exists because falsification found the gap: restoring that write
/// left all nine other arms green. `correct()` runs *after* `cluster_iteration`
/// in `run()`, so the clobber only shows up through the next outer iteration's
/// `cold_start_r` — no arm that stops at the sub-loop can see it. Asserting
/// `correct()` leaves `y` alone states the contract directly.
#[test]
fn the_ridge_correction_does_not_touch_the_centroids() {
    let mut s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    let before = s.y.clone();
    s.correct().expect("ridge solve");
    for (i, (&a, &b)) in before.iter().zip(s.y.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "correct() moved centroid element {i}: {a} -> {b}. The ridge solve \
             owns z_corr; update_y owns y."
        );
    }
}

// ─── Arm 3: update_R's softmax half ──────────────────────────────────

/// At `theta = 0` the diversity penalty is exactly 1 on both sides for any
/// O/E, and at `block_size = 1.0` both are permutation-invariant — so this arm
/// survives the two implementations drawing their shuffle from different RNGs.
/// It pins the softmax, not the penalty; see the values module for why the
/// penalty cannot be pinned against harmonypy at all.
#[test]
fn update_r_matches_harmonypy_with_the_penalty_switched_off() {
    let mut s = fixture_state(r::HP_SOFTMAX_ARM_THETA, r::HP_BLOCK_SIZE, 1);
    s.update_r();
    let got: Vec<f64> = s.r.iter().map(|&v| v as f64).collect();
    let (d, at) = max_abs(&got, &flat(&r::HP_SOFTMAX_R));
    assert!(
        d <= r::HP_SOFTMAX_R_ATOL,
        "R {at}: |delta| {d:.3e} > {:.3e}",
        r::HP_SOFTMAX_R_ATOL
    );
}

// ─── Arm 4: the objective, decomposed ────────────────────────────────

/// Recompute the three objective components the way `compute_objective` does,
/// but separately, so the parity halves and the divergence half can be
/// asserted apart. The formulas are read off `compute_objective`; they are not
/// a second implementation of it, because the two parity components are pinned
/// against harmonypy and would red if this drifted from the real one.
fn objective_parts(s: &HarmonyState) -> (f64, f64, f64) {
    let (k, n, b) = (s.k, s.n, s.layout.b);
    let norm_const = 2000.0 / n as f64;
    let mut kmeans_err = 0f64;
    let mut entropy = 0f64;
    for ku in 0..k {
        let off = ku * n;
        let sg = s.sigma[ku];
        for i in 0..n {
            let rv = s.r[off + i] as f64;
            kmeans_err += rv * s.dist_mat[off + i] as f64;
            if rv > 0.0 {
                entropy += rv * rv.ln() * sg;
            }
        }
    }
    let mut cross = 0f64;
    for ku in 0..k {
        let sg = s.sigma[ku];
        for gb in 0..b {
            let o = s.o[ku * b + gb];
            let e = s.e[ku * b + gb];
            cross += sg * o * s.theta[gb] * ((o + e + 1.0) / (2.0 * e + 1.0)).ln();
        }
    }
    (
        kmeans_err * norm_const,
        entropy * norm_const,
        cross * norm_const,
    )
}

/// Two thirds of the objective is genuine parity: `Σ R·dist` and
/// `Σ σ · R ln R` are the same formula in both implementations.
#[test]
fn the_objectives_distance_and_entropy_terms_match_harmonypy() {
    let s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    let (dist, ent, _) = objective_parts(&s);
    assert!(
        (dist - r::HP_OBJ_DIST).abs() <= r::HP_OBJ_ATOL,
        "objective distance term {dist} vs harmonypy {}",
        r::HP_OBJ_DIST
    );
    assert!(
        (ent - r::HP_OBJ_ENTROPY).abs() <= r::HP_OBJ_ATOL,
        "objective entropy term {ent} vs harmonypy {}",
        r::HP_OBJ_ENTROPY
    );
}

/// The remaining third is a **divergence**, asserted as one. SCX's
/// cross-entropy is `log((O+E+1)/(2E+1))` where harmonypy 0.2.0 uses
/// `log((O+E)/E)`; up to the `+1` smoothing that is harmonypy's value minus
/// `log(2)·Σ σ O θ`. Both are self-consistent and GPU-matched, so the gap is
/// recorded rather than removed.
///
/// Without the third assertion someone widens `HP_OBJ_ATOL`, points this arm at
/// `HP_OBJ_CROSS`, and the flag silently stops meaning anything — the same
/// non-collapse premise 7c's `tie_correct` pins carry.
#[test]
fn the_objectives_cross_entropy_is_a_stated_divergence_not_parity() {
    let s = fixture_state(r::HP_THETA, r::HP_BLOCK_SIZE, 1);
    let (_, _, cross) = objective_parts(&s);
    let gap = (cross - r::HP_OBJ_CROSS).abs();
    assert!(
        (gap - r::HP_OBJ_CROSS_GAP).abs() <= r::HP_OBJ_CROSS_GAP_ATOL,
        "cross-entropy divergence is {gap:.6}, pinned at {:.6}",
        r::HP_OBJ_CROSS_GAP
    );
    assert!(
        gap > r::HP_OBJ_ATOL * 1000.0,
        "the divergence ({gap:.3e}) is within 1000x the parity bar ({:.3e}); \
         one atol widening would make the two arms indistinguishable",
        r::HP_OBJ_ATOL
    );

    // And it is the divergence we say it is, not an unexplained gap that
    // happens to be large. Ignoring the `+1` smoothing,
    //   log((O+E)/(2E)) = log((O+E)/E) - log 2,
    // so SCX's term should sit below harmonypy's by exactly
    //   log(2) * (2000/N) * Σ_k Σ_b σ_k · O[k,b] · θ_b.
    // Deriving that here is not a second oracle — `HP_OBJ_CROSS` is still
    // harmonypy's number, and this only explains the distance to it.
    let (k, n, b) = (s.k, s.n, s.layout.b);
    let mut weighted_o = 0f64;
    for ku in 0..k {
        for gb in 0..b {
            weighted_o += s.sigma[ku] * s.o[ku * b + gb] * s.theta[gb];
        }
    }
    let predicted = std::f64::consts::LN_2 * (2000.0 / n as f64) * weighted_o;
    let rel = (gap - predicted).abs() / predicted;
    assert!(
        rel < 0.02,
        "the divergence ({gap:.4}) is not the log-2 term ({predicted:.4}); \
         relative miss {rel:.4}. Something other than the documented formula \
         difference is moving the cross-entropy."
    );
}

// ─── GPU arm ─────────────────────────────────────────────────────────
// `#[ignore]`d and gated, so a machine without a CUDA driver reports these as
// ignored rather than as passes that did nothing. The GPU harness re-selects
// them with `--include-ignored` under `SCX_REQUIRE_GPU=1`.

/// The device M-step, against the same harmonypy literals the CPU arm uses.
///
/// §7.4 was present on **both** arms: `harmony/gpu.rs` had its own copy of the
/// ridge-intercept centroid write, and its sub-loop passed `d_dist` immutably
/// into a captured graph. Fixing only the CPU would have made
/// `test_gpu_vs_cpu_per_pc_correlation` the thing that caught it — a parity
/// test between two SCX arms, which is the evidence shape this phase replaces.
///
/// This pins the device kernels directly, and that is the ONLY thing covering
/// them.
///
/// An earlier version of this comment said the call site was covered by
/// `test_gpu_vs_cpu_per_pc_correlation` — "with the M-step on one arm only, the
/// two embeddings stop correlating". **That was asserted, and it is false.**
/// Measured on an H100 (job 2840979): remove `gpu_harmony_update_y` from the
/// sub-loop, keep the CPU M-step, and the parity test still passes at its
/// `r >= 0.95` bar. Harmony's corrected embedding is dominated by the ridge
/// solve, which is identical on both arms, so a correlation between two SCX
/// arms cannot see a clustering-loop divergence — the same reason the gate's
/// between-batch-variance probe could not separate the two builds either.
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a CUDA GPU"]
fn gpu_m_step_matches_harmonypy() {
    require_gpu_or_skip!();
    use scx_gpu::{gpu_harmony_l2_normalize_cols, gpu_harmony_update_y, CublasHandle, GpuDevice};

    let dev = GpuDevice::new(0).expect("device");
    let cublas = CublasHandle::new().expect("cublas");
    let (d, k, n) = (r::HP_N_PCS, r::HP_N_CLUSTERS, r::HP_N_CELLS);

    // Z_cos column-major (d x N) — harmonypy's own normalization, so the
    // device arm's *input* is on the reference side too.
    let z_cos: Vec<f32> = r::HP_FIX_Z_COS
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    let r_fix: Vec<f32> = r::HP_FIX_R
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();

    let d_z_cos = dev.htod_copy(&z_cos).expect("upload Z_cos");
    let d_r = dev.htod_copy(&r_fix).expect("upload R");
    let mut d_y = dev.alloc_zeros::<f32>(d * k).expect("alloc Y");

    gpu_harmony_update_y(&dev, &cublas, &d_z_cos, &d_r, &mut d_y, d, k, n).expect("M-step");
    gpu_harmony_l2_normalize_cols(&dev, &mut d_y, d, k).expect("normalize");
    dev.synchronize().expect("sync");

    let got: Vec<f64> = dev
        .dtoh_copy(&d_y)
        .expect("download Y")
        .iter()
        .map(|&v| v as f64)
        .collect();
    let want: Vec<f64> = r::HP_MSTEP_Y
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    let (delta, at) = max_abs(&got, &want);
    assert!(
        delta <= r::HP_GPU_MSTEP_Y_ATOL,
        "GPU M-step centroid {at}: |delta| {delta:.3e} > {:.3e}",
        r::HP_GPU_MSTEP_Y_ATOL
    );
}
