//! Does faer's parallelism setting change PCA's answer?
//!
//! This is the **premise** for `SCX_ACCEL_DETERMINISTIC_LINALG` existing at all.
//! If faer's dense QR and eigendecomposition gave the same bits at every
//! parallelism setting, the knob would be dead weight and the default would
//! already be bit-identical across machines with different core counts.
//!
//! It lives in its own integration binary because it mutates faer's
//! **process-wide** parallelism. `scx-accel`'s unit tests share one process and
//! run concurrently, so doing this there would silently change what every other
//! test measured — and the knob test in `pca_deterministic_linalg_knob.rs` pins
//! `Par::Seq` for its whole process, which would defeat this file.
//!
//! What this file does *not* claim: that SCX's own reductions are affected. They
//! are not — they partition their output, so the schedule cannot reach the
//! result, and `pca::cpu::tests` asserts that at the kernel level with faer out
//! of the picture.

use faer::Par;
use scx_accel::{covariance_pca_inmemory, randomized_pca_inmemory};
use scx_sparse::ScxCsr;

/// Rows dense over `n_vars`, values spanning ~10 decades with non-trivial
/// mantissas. A well-conditioned fixture is bit-identical under every blocking,
/// which would make this file report "no divergence" for the wrong reason.
fn wide_range_csr(n_rows: usize, n_vars: usize) -> ScxCsr {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for r in 0..n_rows {
        for c in 0..n_vars {
            let bits = next();
            let mant = 1.0 + (bits >> 40) as f32 / 16_777_216.0;
            let exp = 10f32.powi((((r + c) % 11) as i32) - 5);
            let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
            indices.push(c as i32);
            data.push(sign * mant * exp);
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr::new_unchecked((n_rows, n_vars), indptr, indices, data)
}

fn differing(a: &[f64], b: &[f64]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .filter(|(x, y)| x.to_bits() != y.to_bits())
        .count()
}

/// Run `f` under an explicit faer parallelism setting.
///
/// Setting the global directly is the only way to vary this: faer's solver
/// constructors take no `Par` argument, they all read `get_global_parallelism()`.
/// Note that `install`ing a rayon pool of a given width is *not* equivalent — it
/// changes `rayon::current_num_threads()`, but faer reaches its parallel path
/// differently from inside a worker thread, and the divergence does not show.
fn under<T>(par: Par, f: impl FnOnce() -> T) -> T {
    faer::set_global_parallelism(par);
    let out = f();
    faer::set_global_parallelism(Par::rayon(0));
    out
}

/// The randomized route's thin QR reblocks with faer's parallelism.
///
/// If this ever fails, faer became parallelism-stable for this shape — good news
/// that would make `SCX_ACCEL_DETERMINISTIC_LINALG` unnecessary for this route.
/// Check before deleting it: the covariance twin below may still need the knob.
#[test]
fn faer_parallelism_changes_the_randomized_route() {
    let csr = wide_range_csr(4_800, 12);
    let run = || {
        randomized_pca_inmemory(&csr, 2, 8, 2, true, 11)
            .unwrap()
            .embeddings
    };

    let seq = under(Par::Seq, run);
    let wide = under(Par::rayon(12), run);
    assert!(
        differing(&seq, &wide) > 0,
        "premise: faer's parallelism no longer changes the randomized route's bits, \
         so SCX_ACCEL_DETERMINISTIC_LINALG buys nothing here"
    );

    // And the knob's setting is stable in itself: sequential twice is sequential.
    assert_eq!(
        differing(&seq, &under(Par::Seq, run)),
        0,
        "sequential faer must be reproducible — if this fails the knob cannot \
         deliver what it promises"
    );
}

/// Same question for the covariance route, whose dense step is the
/// `n_vars × n_vars` self-adjoint eigendecomposition rather than a QR.
#[test]
fn faer_parallelism_changes_the_covariance_route() {
    let csr = wide_range_csr(1_500, 500);
    let run = || covariance_pca_inmemory(&csr, 5, true).unwrap().embeddings;

    let seq = under(Par::Seq, run);
    let wide = under(Par::rayon(12), run);
    assert!(
        differing(&seq, &wide) > 0,
        "premise: faer's parallelism no longer changes the covariance route's bits \
         at n_vars=500. Either faer became parallelism-stable, or this fixture fell \
         below the size at which its eigendecomposition parallelizes at all"
    );

    assert_eq!(
        differing(&seq, &under(Par::Seq, run)),
        0,
        "sequential faer must be reproducible"
    );
}
