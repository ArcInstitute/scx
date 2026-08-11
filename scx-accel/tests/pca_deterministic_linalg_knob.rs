//! `SCX_ACCEL_DETERMINISTIC_LINALG=1` must actually pin faer.
//!
//! Its own integration binary for two reasons: the pin is a process-wide
//! `Once`, so it can only be observed in a process that has not yet run a PCA
//! call; and once it fires, every other test in that process sees sequential
//! faer — which is exactly what `pca_linalg_parallelism.rs` must *not* see.
//!
//! Why the knob is opt-in rather than the default is documented on
//! `pin_linalg_if_requested` in `pca/cpu.rs`: faer is stable run-to-run at a
//! fixed configuration, which is all the reported defect needed, and pinning
//! costs roughly 2.3x on the covariance route's eigendecomposition.

use faer::Par;
use scx_accel::covariance_pca_inmemory;
use scx_sparse::ScxCsr;

#[test]
fn the_env_knob_pins_faer_to_sequential() {
    // Must happen before the first PCA call in this process — the pin is a
    // `Once` read at the top of every entry point.
    std::env::set_var("SCX_ACCEL_DETERMINISTIC_LINALG", "1");
    assert!(
        matches!(faer::get_global_parallelism(), Par::Rayon(_)),
        "premise: faer must start out parallel, or this test cannot tell that \
         the knob did anything"
    );

    let csr = ScxCsr::new_unchecked(
        (4, 3),
        vec![0, 3, 6, 9, 12],
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        vec![1.0, 2.0, 3.0, 4.0, 1.5, 2.5, 0.5, 3.5, 1.0, 2.0, 0.25, 4.5],
    );
    covariance_pca_inmemory(&csr, 2, true).unwrap();

    assert!(
        matches!(faer::get_global_parallelism(), Par::Seq),
        "SCX_ACCEL_DETERMINISTIC_LINALG=1 did not pin faer to sequential; without \
         that the knob promises cross-thread-count reproducibility it cannot give"
    );
}
