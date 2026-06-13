//! Unit tests for Wald inference (spec §11.1, §7.7).

use super::*;

fn close(a: f64, b: f64, rel: f64, abs: f64) {
    let diff = (a - b).abs();
    let tol = abs.max(rel * a.abs().max(b.abs()));
    assert!(diff <= tol, "expected {a} ≈ {b} (|Δ|={diff} > tol={tol})");
}

#[test]
fn contrast_vector_forms() {
    let c = contrast_vector(&NbGlmContrast::Coefficient { index: 1 }, 3);
    assert_eq!(c, vec![0.0, 1.0, 0.0]);
    let v = contrast_vector(
        &NbGlmContrast::Vector {
            weights: vec![0.5, -0.5],
        },
        2,
    );
    assert_eq!(v, vec![0.5, -0.5]);
}

#[test]
fn wald_hand_checked() {
    // fisher = diag(4, 16) ⇒ cov = diag(0.25, 0.0625). beta = [ln10, 0.5].
    // Contrast on coef 1: effect=0.5, se=0.25, stat=2.0, p=2·Φ̄(2)=0.04550,
    // log2fc = 0.5/ln2.
    let beta = vec![10.0_f64.ln(), 0.5];
    let fisher = vec![4.0, 0.0, 0.0, 16.0];
    let c = contrast_vector(&NbGlmContrast::Coefficient { index: 1 }, 2);
    let (out, cov) = wald_stat(&beta, &fisher, 2, &c);
    assert!(
        cov.is_some(),
        "covariance should be returned on a PD fisher"
    );
    close(out.standard_error, 0.25, 1e-12, 1e-12);
    close(out.wald_stat, 2.0, 1e-12, 1e-12);
    close(out.p_value, 0.045_500_263_896_358_4, 1e-9, 1e-12);
    close(
        out.log2_fold_change,
        0.5 / std::f64::consts::LN_2,
        1e-12,
        1e-12,
    );
}

#[test]
fn ill_conditioned_is_conservative() {
    // Singular fisher [[1,1],[1,1]] ⇒ no usable covariance ⇒ p=1, stat=0, se=inf.
    let beta = vec![1.0, 1.0];
    let fisher = vec![1.0, 1.0, 1.0, 1.0];
    let c = contrast_vector(&NbGlmContrast::Coefficient { index: 1 }, 2);
    let (out, _cov) = wald_stat(&beta, &fisher, 2, &c);
    assert_eq!(out.p_value, 1.0);
    assert_eq!(out.wald_stat, 0.0);
    assert!(out.standard_error.is_infinite());
    // log2fc is still reported from beta.
    close(
        out.log2_fold_change,
        1.0 / std::f64::consts::LN_2,
        1e-12,
        1e-12,
    );
}
