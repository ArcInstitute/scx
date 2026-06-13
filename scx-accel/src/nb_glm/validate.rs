//! Input validation for the NB-GLM fit (spec §7.2, §12).
//!
//! All failures map onto existing [`AccelError`] variants — no NB-GLM-specific
//! error type (spec §12). Shape mismatches are `ShapeError`; bad values,
//! rank-deficient designs, and `n_samples ≤ n_features` are `InvalidInput`.

use faer::MatRef;

use crate::error::{AccelError, Result};
use crate::nb_glm::types::NbGlmContrast;

/// Validate the count matrix, design, and size factors before fitting.
///
/// * `counts_gene_major` is `[n_genes × n_samples]` row-major (gene-major).
/// * `design_row_major` is `[n_samples × n_features]` row-major.
/// * `size_factors`, when supplied, is length `n_samples`.
///
/// Checks: dimensional consistency; counts finite & non-negative; size factors
/// finite & strictly positive; `n_samples > n_features` (a fittable design);
/// design full column rank.
pub fn validate_inputs(
    counts_gene_major: &[f64],
    n_genes: usize,
    n_samples: usize,
    design_row_major: &[f64],
    n_features: usize,
    size_factors: Option<&[f64]>,
) -> Result<()> {
    if n_genes == 0 || n_samples == 0 || n_features == 0 {
        return Err(AccelError::InvalidInput(format!(
            "empty dimension: n_genes={n_genes}, n_samples={n_samples}, n_features={n_features}"
        )));
    }
    if counts_gene_major.len() != n_genes * n_samples {
        return Err(AccelError::ShapeError(format!(
            "counts length {} != n_genes*n_samples = {}*{} = {}",
            counts_gene_major.len(),
            n_genes,
            n_samples,
            n_genes * n_samples
        )));
    }
    if design_row_major.len() != n_samples * n_features {
        return Err(AccelError::ShapeError(format!(
            "design length {} != n_samples*n_features = {}*{} = {}",
            design_row_major.len(),
            n_samples,
            n_features,
            n_samples * n_features
        )));
    }
    if let Some(sf) = size_factors {
        if sf.len() != n_samples {
            return Err(AccelError::ShapeError(format!(
                "size_factors length {} != n_samples {}",
                sf.len(),
                n_samples
            )));
        }
        for (s, &v) in sf.iter().enumerate() {
            if !v.is_finite() || v <= 0.0 {
                return Err(AccelError::InvalidInput(format!(
                    "size_factors[{s}] = {v} must be finite and > 0"
                )));
            }
        }
    }

    // Counts: finite & non-negative. Report the first offender.
    for (i, &c) in counts_gene_major.iter().enumerate() {
        if !c.is_finite() || c < 0.0 {
            let g = i / n_samples;
            let s = i % n_samples;
            return Err(AccelError::InvalidInput(format!(
                "counts[gene={g}, sample={s}] = {c} must be finite and >= 0"
            )));
        }
    }

    // A pseudobulk NB-GLM needs more samples than coefficients to estimate
    // dispersion (spec §12). n_samples == n_features leaves zero residual df.
    if n_samples <= n_features {
        return Err(AccelError::InvalidInput(format!(
            "n_samples ({n_samples}) <= n_features ({n_features}): not enough \
             residual degrees of freedom to fit a pseudobulk NB-GLM. For \
             no-replicate layouts use de_method=\"pdex_ref\" or \"wilcoxon\"."
        )));
    }

    validate_design_rank(design_row_major, n_samples, n_features)?;
    Ok(())
}

/// Verify the design matrix is full column rank via the QR R-factor diagonal.
/// Rank-deficient designs (aliased / collinear columns) are rejected naming the
/// deficiency count (spec §7.2); a future mode may drop aliased columns instead.
pub fn validate_design_rank(
    design_row_major: &[f64],
    n_samples: usize,
    n_features: usize,
) -> Result<()> {
    let view = MatRef::from_row_major_slice(design_row_major, n_samples, n_features);
    let qr = view.qr();
    let r = qr.thin_R();
    // Relative tolerance scaled by the largest pivot (LAPACK-style rank test).
    let mut max_diag = 0.0_f64;
    for i in 0..n_features {
        max_diag = max_diag.max(r[(i, i)].abs());
    }
    if max_diag == 0.0 {
        return Err(AccelError::InvalidInput(
            "design matrix is entirely zero (rank 0)".to_string(),
        ));
    }
    let tol = max_diag * (n_samples.max(n_features) as f64) * f64::EPSILON;
    let n_deficient = (0..n_features).filter(|&i| r[(i, i)].abs() <= tol).count();
    if n_deficient > 0 {
        return Err(AccelError::InvalidInput(format!(
            "design matrix is rank-deficient: {n_deficient} of {n_features} \
             columns are linearly dependent (collinear / aliased). Drop or \
             combine the offending columns before fitting."
        )));
    }
    Ok(())
}

/// Validate a contrast against the number of design columns (spec §4.2).
pub fn validate_contrast(contrast: &NbGlmContrast, n_features: usize) -> Result<()> {
    match contrast {
        NbGlmContrast::Coefficient { index } => {
            if *index >= n_features {
                return Err(AccelError::InvalidInput(format!(
                    "contrast coefficient index {index} out of range for \
                     n_features {n_features}"
                )));
            }
        }
        NbGlmContrast::Vector { weights } => {
            if weights.len() != n_features {
                return Err(AccelError::InvalidInput(format!(
                    "contrast vector length {} != n_features {n_features}",
                    weights.len()
                )));
            }
            if weights.iter().any(|w| !w.is_finite()) {
                return Err(AccelError::InvalidInput(
                    "contrast vector contains non-finite weights".to_string(),
                ));
            }
        }
    }
    Ok(())
}
