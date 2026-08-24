//! Property tests for the NB-GLM public entry point (spec §11.5).
//!
//! The fitter must never panic on arbitrary in-shape input — it either returns a
//! fitted result or a typed [`AccelError`]. Malformed input (wrong shape, empty
//! dims, negative / non-finite counts, too few samples) must be rejected.
//!
//! # This file's name is not a reference comparison
//!
//! Despite `_reference`, nothing here compares against pydeseq2 or DESeq2 — the
//! review named exactly that gap: a file called "reference" holding only
//! never-panic properties, with no pinned external numbers anywhere in the Rust
//! suite. The name is kept (moving it would churn nothing useful) and the
//! comparison now exists next to the fitter, where it can reuse crate-internal
//! primitives: `scx-accel/src/nb_glm/pydeseq2_reference_tests.rs` pins a real
//! pydeseq2 0.5.4 run at the bar `docs/pseudobulk_nb_glm.md` claims — ranking,
//! effect sign and significance, explicitly not numerical equality.
//!
//! Both files are load-bearing and neither substitutes for the other: a
//! never-panic property holds over arbitrary input and says nothing about
//! correctness; a pinned reference says everything about one input and nothing
//! about the rest.

use proptest::prelude::*;
use scx_accel::{pseudobulk_nb_glm, AccelError, NbGlmContrast, NbGlmOptions};

/// Two-condition design (intercept + treatment) for `n` samples (half treated).
fn two_condition_design(n_samples: usize) -> Vec<f64> {
    let mut d = Vec::with_capacity(n_samples * 2);
    for s in 0..n_samples {
        let trt = if s >= n_samples / 2 { 1.0 } else { 0.0 };
        d.extend_from_slice(&[1.0, trt]);
    }
    d
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Random non-negative counts on a valid design either fit or return a typed
    /// error — never panic.
    #[test]
    fn random_counts_never_panic(
        n_genes in 1usize..6,
        n_samples in 4usize..12,
        seed in any::<u64>(),
    ) {
        // Simple LCG for deterministic per-case counts in [0, 200).
        let mut state = seed | 1;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) % 200) as f64
        };
        let counts: Vec<f64> = (0..n_genes * n_samples).map(|_| next()).collect();
        let design = two_condition_design(n_samples);
        let res = pseudobulk_nb_glm(
            &counts,
            n_genes,
            n_samples,
            &design,
            2,
            None,
            NbGlmContrast::Coefficient { index: 1 },
            NbGlmOptions::default(),
        );
        // Either Ok with correctly-shaped vectors, or a typed error (acceptable).
        if let Ok(r) = res {
            prop_assert_eq!(r.log2_fold_change.len(), n_genes);
            prop_assert_eq!(r.p_value.len(), n_genes);
            prop_assert_eq!(r.dispersion.len(), n_genes);
        }
    }

    /// A wrong-length count buffer is always a typed ShapeError.
    #[test]
    fn shape_mismatch_rejected(
        n_genes in 1usize..6,
        n_samples in 4usize..12,
        extra in 1usize..5,
    ) {
        let counts = vec![1.0; n_genes * n_samples + extra];
        let design = two_condition_design(n_samples);
        let res = pseudobulk_nb_glm(
            &counts, n_genes, n_samples, &design, 2, None,
            NbGlmContrast::Coefficient { index: 1 }, NbGlmOptions::default(),
        );
        prop_assert!(matches!(res, Err(AccelError::ShapeError(_))));
    }

    /// Negative or non-finite counts are rejected as InvalidInput.
    #[test]
    fn bad_count_values_rejected(
        n_samples in 4usize..12,
        bad_idx in 0usize..40,
        use_nan in any::<bool>(),
    ) {
        let n_genes = 4;
        let total = n_genes * n_samples;
        let mut counts = vec![1.0; total];
        counts[bad_idx % total] = if use_nan { f64::NAN } else { -3.0 };
        let design = two_condition_design(n_samples);
        let res = pseudobulk_nb_glm(
            &counts, n_genes, n_samples, &design, 2, None,
            NbGlmContrast::Coefficient { index: 1 }, NbGlmOptions::default(),
        );
        prop_assert!(matches!(res, Err(AccelError::InvalidInput(_))));
    }

    /// n_samples <= n_features is rejected (no residual df).
    #[test]
    fn too_few_samples_rejected(n_features in 2usize..6) {
        let n_samples = n_features; // == ⇒ rejected
        let n_genes = 2;
        let counts = vec![3.0; n_genes * n_samples];
        let design = vec![1.0; n_samples * n_features]; // rank-1 but caught by n<=p first
        let res = pseudobulk_nb_glm(
            &counts, n_genes, n_samples, &design, n_features, None,
            NbGlmContrast::Coefficient { index: 0 }, NbGlmOptions::default(),
        );
        prop_assert!(matches!(res, Err(AccelError::InvalidInput(_))));
    }
}

/// Empty dimensions are rejected (outside `proptest!` — fixed input).
#[test]
fn empty_dims_rejected() {
    let res = pseudobulk_nb_glm(
        &[],
        0,
        4,
        &[],
        2,
        None,
        NbGlmContrast::Coefficient { index: 0 },
        NbGlmOptions::default(),
    );
    assert!(matches!(res, Err(AccelError::InvalidInput(_))));
}
