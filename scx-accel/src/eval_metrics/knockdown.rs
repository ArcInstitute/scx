//! Knockdown efficiency and log deviation metrics.
//!
//! Implements per-cell knockdown efficiency and log fold change metrics from
//! `arc-bench`. These metrics measure how effectively each perturbation knocks
//! down its target gene, using efficient single-column extraction from CSR
//! sparse matrices.
//!
//! ## Knockdown Efficiency
//!
//! For each perturbed cell, computes:
//! ```text
//! KD = 1.0 - x_target / (μ_control[gene_idx] + ε)
//! ```
//! where `x_target` is the cell's expression of its target gene (perturbation
//! name → gene name), and `μ_control` is the mean control expression for that
//! gene. Operates on **normalized** (not log-transformed) data.
//!
//! ## Log Deviation
//!
//! For each perturbed cell, computes:
//! ```text
//! FC = x_log - log1p(μ_control[gene_idx])
//! ```
//! where `x_log` is the already-log1p-transformed expression. Operates on
//! **log1p-transformed** data.
//!
//! ## CSR Column Extraction
//!
//! Both metrics require extracting a single gene column per perturbation from
//! a CSR sparse matrix. In CSR format, finding column `j` in row `i` requires
//! scanning `indices[indptr[i]..indptr[i+1]]`. Since SCX CSR shards store
//! indices in sorted order, we use binary search for O(log nnz_per_row) lookup
//! instead of linear scan.

use std::collections::HashMap;

/// Compute mean expression of control cells per gene from CSR slices.
///
/// # Arguments
/// * `indptr` — CSR indptr array (length = n_obs + 1).
/// * `indices` — CSR column indices.
/// * `data` — CSR values.
/// * `pert_labels` — Per-cell perturbation labels.
/// * `ctrl_label` — Label identifying control cells.
/// * `n_vars` — Number of genes (columns).
///
/// # Returns
/// Dense vector of length `n_vars` with mean control expression per gene.
///
/// # Index bounds
/// Column indices `>= n_vars` are silently skipped. This provides defensive
/// tolerance for edge cases (e.g., scipy CSR with stale indices after column
/// slicing), consistent with scipy's own behavior of ignoring out-of-range
/// entries. Use [`compute_knockdown_efficiency`] or [`compute_log_deviation`]
/// for the per-cell metrics — those functions use binary search via
/// [`csr_get_value`] which naturally returns 0.0 for missing columns.
pub fn compute_control_baseline(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    pert_labels: &[String],
    ctrl_label: &str,
    n_vars: usize,
) -> crate::Result<Vec<f64>> {
    let n_obs = pert_labels.len();
    if indptr.len() != n_obs + 1 {
        return Err(crate::AccelError::InvalidInput(format!(
            "indptr length {} != n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        )));
    }

    let mut sums = vec![0.0f64; n_vars];
    let mut n_control: usize = 0;

    for (row, label) in pert_labels.iter().enumerate() {
        if label != ctrl_label {
            continue;
        }
        n_control += 1;
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for k in start..end {
            let col = indices[k] as usize;
            if col < n_vars {
                // f32 → f64 promotion: intentional precision gain when
                // accumulating sums over many control cells.
                sums[col] += data[k] as f64;
            }
        }
    }

    if n_control == 0 {
        return Err(crate::AccelError::InvalidInput(format!(
            "no cells found with label '{ctrl_label}'"
        )));
    }

    let n = n_control as f64;
    for s in &mut sums {
        *s /= n;
    }

    Ok(sums)
}

/// Extract the value at column `col_idx` from a CSR row using binary search.
///
/// CSR indices within each row are sorted, so binary search gives O(log nnz).
/// Returns 0.0 if the column is not present (implicit zero).
#[inline]
fn csr_get_value(
    indices: &[i32],
    data: &[f32],
    row_start: usize,
    row_end: usize,
    col_idx: i32,
) -> f32 {
    let row_indices = &indices[row_start..row_end];
    match row_indices.binary_search(&col_idx) {
        Ok(pos) => data[row_start + pos],
        Err(_) => 0.0,
    }
}

/// Compute per-cell knockdown efficiency from CSR sparse matrix.
///
/// For each non-control cell whose perturbation name matches a gene name:
/// ```text
/// KD[cell] = 1.0 - X[cell, gene_idx] / (baseline[gene_idx] + eps)
/// ```
///
/// Cells that are control or whose perturbation name doesn't match any gene
/// name will have NaN in the output.
///
/// # Arguments
/// * `indptr` — CSR indptr array (length = n_obs + 1).
/// * `indices` — CSR column indices (sorted within each row).
/// * `data` — CSR values (normalized, NOT log-transformed).
/// * `pert_labels` — Per-cell perturbation labels.
/// * `ctrl_label` — Label identifying control cells.
/// * `gene_names` — Gene names (length = n_vars), used to find target gene index.
/// * `baseline` — Mean control expression per gene (from `compute_control_baseline`).
/// * `eps` — Small constant for numerical stability (default: 1e-8).
///
/// # Returns
/// Vec<f32> of length n_obs with per-cell knockdown efficiency.
#[allow(clippy::too_many_arguments)]
pub fn compute_knockdown_efficiency(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    pert_labels: &[String],
    ctrl_label: &str,
    gene_names: &[String],
    baseline: &[f64],
    eps: f64,
) -> crate::Result<Vec<f32>> {
    let n_obs = pert_labels.len();
    let n_vars = gene_names.len();

    if indptr.len() != n_obs + 1 {
        return Err(crate::AccelError::InvalidInput(format!(
            "indptr length {} != n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        )));
    }
    if baseline.len() != n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "baseline length {} != n_vars {}",
            baseline.len(),
            n_vars
        )));
    }

    // Build gene name → column index map.
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut efficiency = vec![f32::NAN; n_obs];

    for (row, label) in pert_labels.iter().enumerate() {
        if label == ctrl_label {
            continue;
        }

        // Look up the gene index matching this perturbation name.
        let gene_idx = match gene_to_idx.get(label.as_str()) {
            Some(&idx) => idx,
            None => continue, // perturbation name doesn't match any gene → NaN
        };

        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let x_target = csr_get_value(indices, data, start, end, gene_idx as i32) as f64;
        let mu_control = baseline[gene_idx];

        efficiency[row] = (1.0 - x_target / (mu_control + eps)) as f32;
    }

    Ok(efficiency)
}

/// Compute per-cell log fold change from CSR sparse matrix.
///
/// For each non-control cell whose perturbation name matches a gene name:
/// ```text
/// FC[cell] = X_log[cell, gene_idx] - baseline_log[gene_idx]
/// ```
///
/// When `apply_log1p` is true, `data` is treated as raw (non-log-transformed)
/// values and `log1p` is applied on-the-fly to each extracted entry. This
/// avoids allocating a second copy of the full data array when the caller
/// already has raw data in hand.
///
/// Expects already log1p-transformed input (matches arc-bench pipeline order).
///
/// # Arguments
/// * `indptr` — CSR indptr array (length = n_obs + 1).
/// * `indices` — CSR column indices (sorted within each row).
/// * `data` — CSR values. If `apply_log1p` is false, expected to be already
///   log1p-transformed; if true, expected to be raw (non-log) and the kernel
///   applies `log1p` to each extracted entry.
/// * `pert_labels` — Per-cell perturbation labels.
/// * `ctrl_label` — Label identifying control cells.
/// * `gene_names` — Gene names (length = n_vars).
/// * `baseline_log` — log1p of mean control expression per gene.
/// * `apply_log1p` — When true, apply `log1p` on-the-fly to `data` entries.
///
/// # Returns
/// Vec<f32> of length n_obs with per-cell log fold change.
#[allow(clippy::too_many_arguments)]
pub fn compute_log_deviation(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    pert_labels: &[String],
    ctrl_label: &str,
    gene_names: &[String],
    baseline_log: &[f64],
    apply_log1p: bool,
) -> crate::Result<Vec<f32>> {
    let n_obs = pert_labels.len();
    let n_vars = gene_names.len();

    if indptr.len() != n_obs + 1 {
        return Err(crate::AccelError::InvalidInput(format!(
            "indptr length {} != n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        )));
    }
    if baseline_log.len() != n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "baseline_log length {} != n_vars {}",
            baseline_log.len(),
            n_vars
        )));
    }

    // Build gene name → column index map.
    let gene_to_idx: HashMap<&str, usize> = gene_names
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();

    let mut log_fc = vec![f32::NAN; n_obs];

    for (row, label) in pert_labels.iter().enumerate() {
        if label == ctrl_label {
            continue;
        }

        let gene_idx = match gene_to_idx.get(label.as_str()) {
            Some(&idx) => idx,
            None => continue,
        };

        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let raw = csr_get_value(indices, data, start, end, gene_idx as i32) as f64;
        let x_log = if apply_log1p { raw.ln_1p() } else { raw };
        let mu_log = baseline_log[gene_idx];

        log_fc[row] = (x_log - mu_log) as f32;
    }

    Ok(log_fc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small CSR matrix from dense for testing.
    /// Returns (indptr, indices, data).
    fn dense_to_csr(dense: &[f32], n_rows: usize, n_cols: usize) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        let mut indptr = vec![0i64; n_rows + 1];
        let mut indices = Vec::new();
        let mut data = Vec::new();

        for row in 0..n_rows {
            for col in 0..n_cols {
                let val = dense[row * n_cols + col];
                if val != 0.0 {
                    indices.push(col as i32);
                    data.push(val);
                }
            }
            indptr[row + 1] = indices.len() as i64;
        }

        (indptr, indices, data)
    }

    #[test]
    fn test_control_baseline_simple() {
        // 4 cells × 3 genes
        // cells 0,1 are control; cells 2,3 are perturbation
        #[rustfmt::skip]
        let dense = vec![
            1.0, 2.0, 3.0,  // control
            3.0, 4.0, 5.0,  // control
            10.0, 0.0, 0.0, // pert "gene_0"
            0.0, 20.0, 0.0, // pert "gene_1"
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 4, 3);
        let labels = vec![
            "control".to_string(),
            "control".to_string(),
            "gene_0".to_string(),
            "gene_1".to_string(),
        ];

        let baseline =
            compute_control_baseline(&indptr, &indices, &data, &labels, "control", 3).unwrap();

        // Mean of control rows: [(1+3)/2, (2+4)/2, (3+5)/2] = [2, 3, 4]
        assert!((baseline[0] - 2.0).abs() < 1e-10);
        assert!((baseline[1] - 3.0).abs() < 1e-10);
        assert!((baseline[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_control_baseline_no_control() {
        let (indptr, indices, data) = dense_to_csr(&[1.0, 2.0], 1, 2);
        let labels = vec!["not_control".to_string()];

        let result = compute_control_baseline(&indptr, &indices, &data, &labels, "control", 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_knockdown_efficiency_simple() {
        // 4 cells × 3 genes (gene_0, gene_1, gene_2)
        // Cells 0,1 are control; cell 2 is "gene_0", cell 3 is "gene_1"
        // After normalize, let's say:
        //   gene_0: control mean = 2.0
        //   gene_1: control mean = 3.0
        //   cell 2 (pert "gene_0"): x[gene_0] = 0.5
        //   cell 3 (pert "gene_1"): x[gene_1] = 1.5
        #[rustfmt::skip]
        let dense = vec![
            1.0, 2.0, 3.0,   // control
            3.0, 4.0, 5.0,   // control
            0.5, 0.0, 0.0,   // pert "gene_0"
            0.0, 1.5, 0.0,   // pert "gene_1"
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 4, 3);
        let labels = vec![
            "control".to_string(),
            "control".to_string(),
            "gene_0".to_string(),
            "gene_1".to_string(),
        ];
        let gene_names = vec![
            "gene_0".to_string(),
            "gene_1".to_string(),
            "gene_2".to_string(),
        ];

        let baseline =
            compute_control_baseline(&indptr, &indices, &data, &labels, "control", 3).unwrap();
        let eps = 1e-8;

        let eff = compute_knockdown_efficiency(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline,
            eps,
        )
        .unwrap();

        assert_eq!(eff.len(), 4);
        // Control cells should be NaN
        assert!(eff[0].is_nan());
        assert!(eff[1].is_nan());

        // Cell 2 (pert "gene_0"): KD = 1 - 0.5 / (2.0 + eps) ≈ 0.75
        let expected_0 = 1.0 - 0.5 / (2.0 + eps);
        assert!((eff[2] as f64 - expected_0).abs() < 1e-5, "got {}", eff[2]);

        // Cell 3 (pert "gene_1"): KD = 1 - 1.5 / (3.0 + eps) ≈ 0.5
        let expected_1 = 1.0 - 1.5 / (3.0 + eps);
        assert!((eff[3] as f64 - expected_1).abs() < 1e-5, "got {}", eff[3]);
    }

    #[test]
    fn test_knockdown_missing_gene() {
        // Perturbation name "unknown_gene" doesn't match any gene → NaN
        #[rustfmt::skip]
        let dense = vec![
            1.0, 2.0,   // control
            3.0, 4.0,   // pert "unknown_gene"
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 2, 2);
        let labels = vec!["control".to_string(), "unknown_gene".to_string()];
        let gene_names = vec!["gene_0".to_string(), "gene_1".to_string()];
        let baseline = vec![1.0, 2.0];

        let eff = compute_knockdown_efficiency(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline,
            1e-8,
        )
        .unwrap();

        assert!(eff[0].is_nan()); // control
        assert!(eff[1].is_nan()); // unmatched pert
    }

    #[test]
    fn test_log_deviation_simple() {
        // After log1p, the data is in log space.
        // Say control baseline (pre-log) = [2.0, 3.0]
        // baseline_log = [log1p(2.0), log1p(3.0)] = [1.0986, 1.3863]
        // Cell 2 (pert "gene_0"): x_log[gene_0] = 0.5  → FC = 0.5 - 1.0986 = -0.5986
        // Cell 3 (pert "gene_1"): x_log[gene_1] = 2.0  → FC = 2.0 - 1.3863 = 0.6137
        let baseline_log = vec![
            (3.0f64).ln_1p(), // log1p(2.0) = ln(3) ≈ 1.0986
            (4.0f64).ln_1p(), // log1p(3.0) = ln(4) ≈ 1.3863
        ];

        #[rustfmt::skip]
        let dense = vec![
            1.0, 1.5, // control (log-space values)
            1.2, 1.8, // control
            0.5, 0.0, // pert "gene_0"
            0.0, 2.0, // pert "gene_1"
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 4, 2);
        let labels = vec![
            "control".to_string(),
            "control".to_string(),
            "gene_0".to_string(),
            "gene_1".to_string(),
        ];
        let gene_names = vec!["gene_0".to_string(), "gene_1".to_string()];

        let fc = compute_log_deviation(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline_log,
            false,
        )
        .unwrap();

        assert_eq!(fc.len(), 4);
        assert!(fc[0].is_nan()); // control
        assert!(fc[1].is_nan()); // control

        // Cell 2: x_log=0.5, baseline_log[0]=ln(3)≈1.0986 → FC ≈ -0.5986
        let expected_2 = 0.5 - baseline_log[0];
        assert!((fc[2] as f64 - expected_2).abs() < 1e-5, "got {}", fc[2]);

        // Cell 3: x_log=2.0, baseline_log[1]=ln(4)≈1.3863 → FC ≈ 0.6137
        let expected_3 = 2.0 - baseline_log[1];
        assert!((fc[3] as f64 - expected_3).abs() < 1e-5, "got {}", fc[3]);
    }

    #[test]
    fn test_log_deviation_missing_gene() {
        let baseline_log = vec![1.0, 2.0];
        #[rustfmt::skip]
        let dense = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 2, 2);
        let labels = vec!["control".to_string(), "unknown_gene".to_string()];
        let gene_names = vec!["gene_0".to_string(), "gene_1".to_string()];

        let fc = compute_log_deviation(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline_log,
            false,
        )
        .unwrap();

        assert!(fc[0].is_nan()); // control
        assert!(fc[1].is_nan()); // unmatched
    }

    #[test]
    fn test_knockdown_zero_control_expression() {
        // When control baseline is ~0, KD ≈ 1 - x/(0+eps) → very large negative
        // This tests numerical stability with eps guard.
        #[rustfmt::skip]
        let dense = vec![
            0.0, 1.0,   // control (gene_0 = 0)
            5.0, 0.0,   // pert "gene_0" (high expression)
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 2, 2);
        let labels = vec!["control".to_string(), "gene_0".to_string()];
        let gene_names = vec!["gene_0".to_string(), "gene_1".to_string()];
        let baseline = vec![0.0, 1.0]; // gene_0 baseline is 0

        let eff = compute_knockdown_efficiency(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline,
            1e-8,
        )
        .unwrap();

        // KD = 1 - 5.0 / (0.0 + 1e-8) = very large negative number
        assert!(eff[1].is_finite(), "eps guard should prevent NaN/Inf");
    }

    #[test]
    fn test_knockdown_sparse_zero_value() {
        // When the target gene has zero expression (not stored in CSR),
        // csr_get_value should return 0.0, giving KD = 1 - 0/(baseline+eps) ≈ 1.0
        #[rustfmt::skip]
        let dense = vec![
            2.0, 0.0,   // control
            0.0, 5.0,   // pert "gene_0" (gene_0 is zero — not stored in sparse)
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 2, 2);
        let labels = vec!["control".to_string(), "gene_0".to_string()];
        let gene_names = vec!["gene_0".to_string(), "gene_1".to_string()];
        let baseline = vec![2.0, 0.0];

        let eff = compute_knockdown_efficiency(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline,
            1e-8,
        )
        .unwrap();

        // KD = 1 - 0 / (2.0 + eps) ≈ 1.0 (perfect knockdown)
        assert!((eff[1] as f64 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_validation_errors() {
        let labels = vec!["control".to_string(), "gene_0".to_string()];
        let gene_names = vec!["gene_0".to_string()];
        let baseline = vec![1.0];

        // Wrong indptr length
        let result = compute_knockdown_efficiency(
            &[0, 1], // length 2, should be 3 (n_obs + 1 = 3)
            &[0],
            &[1.0],
            &labels,
            "control",
            &gene_names,
            &baseline,
            1e-8,
        );
        assert!(result.is_err());

        // Wrong baseline length
        let result = compute_knockdown_efficiency(
            &[0, 1, 2],
            &[0, 0],
            &[1.0, 2.0],
            &labels,
            "control",
            &gene_names,
            &[1.0, 2.0], // length 2, should be 1 (n_vars)
            1e-8,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_perturbations_with_shared_gene() {
        // Test with multiple perturbation types, ensuring isolation.
        // 6 cells × 4 genes
        #[rustfmt::skip]
        let dense = vec![
            2.0, 4.0, 6.0, 8.0,  // control
            2.0, 4.0, 6.0, 8.0,  // control
            1.0, 0.0, 0.0, 0.0,  // pert "gene_0"
            1.0, 0.0, 0.0, 0.0,  // pert "gene_0"
            0.0, 2.0, 0.0, 0.0,  // pert "gene_1"
            0.0, 0.0, 3.0, 0.0,  // pert "gene_2"
        ];
        let (indptr, indices, data) = dense_to_csr(&dense, 6, 4);
        let labels = vec![
            "control".to_string(),
            "control".to_string(),
            "gene_0".to_string(),
            "gene_0".to_string(),
            "gene_1".to_string(),
            "gene_2".to_string(),
        ];
        let gene_names = vec![
            "gene_0".to_string(),
            "gene_1".to_string(),
            "gene_2".to_string(),
            "gene_3".to_string(),
        ];

        let baseline =
            compute_control_baseline(&indptr, &indices, &data, &labels, "control", 4).unwrap();
        // baseline = [2, 4, 6, 8]
        assert!((baseline[0] - 2.0).abs() < 1e-10);
        assert!((baseline[1] - 4.0).abs() < 1e-10);
        assert!((baseline[2] - 6.0).abs() < 1e-10);
        assert!((baseline[3] - 8.0).abs() < 1e-10);

        let eps = 1e-8;
        let eff = compute_knockdown_efficiency(
            &indptr,
            &indices,
            &data,
            &labels,
            "control",
            &gene_names,
            &baseline,
            eps,
        )
        .unwrap();

        assert_eq!(eff.len(), 6);
        assert!(eff[0].is_nan()); // control
        assert!(eff[1].is_nan()); // control

        // Cell 2: pert "gene_0", x=1.0, baseline[0]=2.0
        // KD = 1 - 1/2 = 0.5
        assert!((eff[2] as f64 - 0.5).abs() < 1e-5);
        // Cell 3: same perturbation, same gene value
        assert!((eff[3] as f64 - 0.5).abs() < 1e-5);
        // Cell 4: pert "gene_1", x=2.0, baseline[1]=4.0
        // KD = 1 - 2/4 = 0.5
        assert!((eff[4] as f64 - 0.5).abs() < 1e-5);
        // Cell 5: pert "gene_2", x=3.0, baseline[2]=6.0
        // KD = 1 - 3/6 = 0.5
        assert!((eff[5] as f64 - 0.5).abs() < 1e-5);
    }
}
