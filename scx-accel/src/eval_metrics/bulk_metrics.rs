//! Bulk perturbation metrics on pseudobulk means.
//!
//! Given `means_real[P, G]` and `means_pred[P, G]` (pseudobulk means for
//! P perturbations × G genes), computes per-perturbation metrics:
//!
//! - **pearson_delta**: Pearson correlation of perturbation effects
//!   (delta from control) between real and predicted.
//! - **mse**: Mean squared error of pseudobulk means.
//! - **mae**: Mean absolute error of pseudobulk means.
//! - **mse_delta**: MSE of perturbation effects (delta from control).
//! - **mae_delta**: MAE of perturbation effects (delta from control).

use std::collections::HashMap;

/// Which bulk metrics to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BulkMetric {
    PearsonDelta,
    Mse,
    Mae,
    MseDelta,
    MaeDelta,
}

impl BulkMetric {
    /// Parse from string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "pearson_delta" => Some(Self::PearsonDelta),
            "mse" => Some(Self::Mse),
            "mae" => Some(Self::Mae),
            "mse_delta" => Some(Self::MseDelta),
            "mae_delta" => Some(Self::MaeDelta),
            _ => None,
        }
    }

    /// Canonical name for the metric.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PearsonDelta => "pearson_delta",
            Self::Mse => "mse",
            Self::Mae => "mae",
            Self::MseDelta => "mse_delta",
            Self::MaeDelta => "mae_delta",
        }
    }
}

/// Result of bulk perturbation metric computation.
#[derive(Debug, Clone)]
pub struct BulkMetricsResult {
    /// Metric name → per-perturbation values (length = n_perts, excluding control).
    pub metrics: HashMap<String, Vec<f64>>,
    /// Perturbation names in order (excluding control).
    pub pert_names: Vec<String>,
}

/// Compute Pearson correlation between two slices.
///
/// Uses the canonical two-pass algorithm for numerical stability:
/// first pass computes the means, second pass computes deviations.
/// Returns NaN for constant vectors (zero variance) or length < 2.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 || n != y.len() {
        return f64::NAN;
    }

    let n_f = n as f64;
    let mean_x = x.iter().sum::<f64>() / n_f;
    let mean_y = y.iter().sum::<f64>() / n_f;

    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
        sum_xy += dx * dy;
    }

    let denom = (sum_xx * sum_yy).sqrt();
    if denom == 0.0 {
        return f64::NAN;
    }

    sum_xy / denom
}

/// Compute bulk perturbation metrics in a single fused pass.
///
/// # Arguments
/// * `means_real` — `[n_perts × n_genes]` row-major pseudobulk means (real).
/// * `means_pred` — `[n_perts × n_genes]` row-major pseudobulk means (predicted).
/// * `ctrl_idx` — Index of the control perturbation row.
/// * `n_perts` — Total number of perturbations (including control).
/// * `n_genes` — Number of genes.
/// * `pert_names` — Names of all perturbations (length = n_perts, including control).
/// * `metrics` — Which metrics to compute.
///
/// # Returns
/// `BulkMetricsResult` with per-perturbation values for each requested metric.
/// The control perturbation is excluded from the output.
pub fn compute_bulk_metrics(
    means_real: &[f64],
    means_pred: &[f64],
    ctrl_idx: usize,
    n_perts: usize,
    n_genes: usize,
    pert_names: &[String],
    metrics: &[BulkMetric],
) -> crate::Result<BulkMetricsResult> {
    if means_real.len() != n_perts * n_genes {
        return Err(crate::AccelError::InvalidInput(format!(
            "means_real length {} doesn't match n_perts={} × n_genes={}",
            means_real.len(),
            n_perts,
            n_genes
        )));
    }
    if means_pred.len() != n_perts * n_genes {
        return Err(crate::AccelError::InvalidInput(format!(
            "means_pred length {} doesn't match n_perts={} × n_genes={}",
            means_pred.len(),
            n_perts,
            n_genes
        )));
    }
    if ctrl_idx >= n_perts {
        return Err(crate::AccelError::InvalidInput(format!(
            "ctrl_idx {} >= n_perts {}",
            ctrl_idx, n_perts
        )));
    }
    if pert_names.len() != n_perts {
        return Err(crate::AccelError::InvalidInput(format!(
            "pert_names length {} != n_perts {}",
            pert_names.len(),
            n_perts
        )));
    }

    // Extract control rows.
    let ctrl_real = &means_real[ctrl_idx * n_genes..(ctrl_idx + 1) * n_genes];
    let ctrl_pred = &means_pred[ctrl_idx * n_genes..(ctrl_idx + 1) * n_genes];

    // Determine which metric types are needed.
    let need_pearson_delta = metrics.contains(&BulkMetric::PearsonDelta);
    let need_mse = metrics.contains(&BulkMetric::Mse);
    let need_mae = metrics.contains(&BulkMetric::Mae);
    let need_mse_delta = metrics.contains(&BulkMetric::MseDelta);
    let need_mae_delta = metrics.contains(&BulkMetric::MaeDelta);
    let need_delta = need_pearson_delta || need_mse_delta || need_mae_delta;

    // Pre-allocate result vectors (excluding control).
    let n_output = n_perts - 1;
    let mut pearson_delta_vals = if need_pearson_delta {
        Vec::with_capacity(n_output)
    } else {
        Vec::new()
    };
    let mut mse_vals = if need_mse {
        Vec::with_capacity(n_output)
    } else {
        Vec::new()
    };
    let mut mae_vals = if need_mae {
        Vec::with_capacity(n_output)
    } else {
        Vec::new()
    };
    let mut mse_delta_vals = if need_mse_delta {
        Vec::with_capacity(n_output)
    } else {
        Vec::new()
    };
    let mut mae_delta_vals = if need_mae_delta {
        Vec::with_capacity(n_output)
    } else {
        Vec::new()
    };
    let mut output_pert_names = Vec::with_capacity(n_output);

    // Temporary buffers for delta computation.
    let mut delta_real = if need_delta {
        vec![0.0; n_genes]
    } else {
        Vec::new()
    };
    let mut delta_pred = if need_delta {
        vec![0.0; n_genes]
    } else {
        Vec::new()
    };

    let n_genes_f = n_genes as f64;

    for p in 0..n_perts {
        if p == ctrl_idx {
            continue;
        }

        output_pert_names.push(pert_names[p].clone());

        let row_real = &means_real[p * n_genes..(p + 1) * n_genes];
        let row_pred = &means_pred[p * n_genes..(p + 1) * n_genes];

        // Compute direct MSE/MAE if needed.
        if need_mse || need_mae {
            let mut sum_sq = 0.0;
            let mut sum_abs = 0.0;
            for g in 0..n_genes {
                let diff = row_real[g] - row_pred[g];
                if need_mse {
                    sum_sq += diff * diff;
                }
                if need_mae {
                    sum_abs += diff.abs();
                }
            }
            if need_mse {
                mse_vals.push(sum_sq / n_genes_f);
            }
            if need_mae {
                mae_vals.push(sum_abs / n_genes_f);
            }
        }

        // Compute delta-based metrics if needed.
        if need_delta {
            for g in 0..n_genes {
                delta_real[g] = row_real[g] - ctrl_real[g];
                delta_pred[g] = row_pred[g] - ctrl_pred[g];
            }

            if need_pearson_delta {
                pearson_delta_vals.push(pearson_correlation(&delta_real, &delta_pred));
            }

            if need_mse_delta || need_mae_delta {
                let mut sum_sq = 0.0;
                let mut sum_abs = 0.0;
                for g in 0..n_genes {
                    let diff = delta_real[g] - delta_pred[g];
                    if need_mse_delta {
                        sum_sq += diff * diff;
                    }
                    if need_mae_delta {
                        sum_abs += diff.abs();
                    }
                }
                if need_mse_delta {
                    mse_delta_vals.push(sum_sq / n_genes_f);
                }
                if need_mae_delta {
                    mae_delta_vals.push(sum_abs / n_genes_f);
                }
            }
        }
    }

    // Build result map.
    let mut result_map = HashMap::new();
    if need_pearson_delta {
        result_map.insert("pearson_delta".to_string(), pearson_delta_vals);
    }
    if need_mse {
        result_map.insert("mse".to_string(), mse_vals);
    }
    if need_mae {
        result_map.insert("mae".to_string(), mae_vals);
    }
    if need_mse_delta {
        result_map.insert("mse_delta".to_string(), mse_delta_vals);
    }
    if need_mae_delta {
        result_map.insert("mae_delta".to_string(), mae_delta_vals);
    }

    Ok(BulkMetricsResult {
        metrics: result_map,
        pert_names: output_pert_names,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_known_values() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "expected 1.0, got {}", r);

        // Perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r_neg = pearson_correlation(&x, &y_neg);
        assert!(
            (r_neg - (-1.0)).abs() < 1e-10,
            "expected -1.0, got {}",
            r_neg
        );

        // No correlation (orthogonal)
        let a = vec![1.0, 0.0, -1.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, -1.0];
        let r_zero = pearson_correlation(&a, &b);
        assert!(r_zero.abs() < 1e-10, "expected ~0.0, got {}", r_zero);
    }

    #[test]
    fn test_pearson_zero_variance() {
        let x = vec![5.0, 5.0, 5.0];
        let y = vec![1.0, 2.0, 3.0];
        let r = pearson_correlation(&x, &y);
        assert!(r.is_nan(), "expected NaN for zero-variance input");
    }

    #[test]
    fn test_pearson_single_element() {
        let x = vec![1.0];
        let y = vec![2.0];
        let r = pearson_correlation(&x, &y);
        assert!(r.is_nan(), "expected NaN for n < 2");
    }

    #[test]
    fn test_mse_mae_known_values() {
        // 3 perturbations (indices 0=ctrl, 1=A, 2=B), 2 genes
        let means_real = vec![
            1.0, 2.0, // ctrl
            3.0, 4.0, // A
            5.0, 6.0, // B
        ];
        let means_pred = vec![
            1.0, 2.0, // ctrl (same)
            4.0, 5.0, // A (pred differs by 1.0 each gene)
            7.0, 8.0, // B (pred differs by 2.0 each gene)
        ];
        let pert_names: Vec<String> = vec!["ctrl", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();

        let result = compute_bulk_metrics(
            &means_real,
            &means_pred,
            0, // ctrl_idx
            3,
            2,
            &pert_names,
            &[BulkMetric::Mse, BulkMetric::Mae],
        )
        .unwrap();

        assert_eq!(result.pert_names, vec!["A", "B"]);

        let mse = &result.metrics["mse"];
        // A: mean((3-4)^2 + (4-5)^2) / 2 = (1 + 1) / 2 = 1.0
        assert!(
            (mse[0] - 1.0).abs() < 1e-10,
            "A MSE expected 1.0, got {}",
            mse[0]
        );
        // B: mean((5-7)^2 + (6-8)^2) / 2 = (4 + 4) / 2 = 4.0
        assert!(
            (mse[1] - 4.0).abs() < 1e-10,
            "B MSE expected 4.0, got {}",
            mse[1]
        );

        let mae = &result.metrics["mae"];
        // A: mean(|3-4| + |4-5|) / 2 = (1 + 1) / 2 = 1.0
        assert!(
            (mae[0] - 1.0).abs() < 1e-10,
            "A MAE expected 1.0, got {}",
            mae[0]
        );
        // B: mean(|5-7| + |6-8|) / 2 = (2 + 2) / 2 = 2.0
        assert!(
            (mae[1] - 2.0).abs() < 1e-10,
            "B MAE expected 2.0, got {}",
            mae[1]
        );
    }

    #[test]
    fn test_delta_metrics() {
        // 3 perturbations (0=ctrl, 1=A, 2=B), 3 genes
        let means_real = vec![
            1.0, 1.0, 1.0, // ctrl
            3.0, 2.0, 4.0, // A: delta = [2, 1, 3]
            5.0, 3.0, 2.0, // B: delta = [4, 2, 1]
        ];
        let means_pred = vec![
            1.0, 1.0, 1.0, // ctrl
            3.0, 2.0, 4.0, // A: delta = [2, 1, 3] (same as real)
            6.0, 4.0, 3.0, // B: delta = [5, 3, 2] (differs from real)
        ];
        let pert_names: Vec<String> = vec!["ctrl", "A", "B"]
            .into_iter()
            .map(String::from)
            .collect();

        let result = compute_bulk_metrics(
            &means_real,
            &means_pred,
            0,
            3,
            3,
            &pert_names,
            &[
                BulkMetric::PearsonDelta,
                BulkMetric::MseDelta,
                BulkMetric::MaeDelta,
            ],
        )
        .unwrap();

        // A: delta_real == delta_pred → pearson = 1.0
        let pearson = &result.metrics["pearson_delta"];
        assert!(
            (pearson[0] - 1.0).abs() < 1e-10,
            "A pearson_delta expected 1.0, got {}",
            pearson[0]
        );

        // B: delta_real = [4, 2, 1], delta_pred = [5, 3, 2]
        // diff = [-1, -1, -1]
        // MSE = (1 + 1 + 1) / 3 = 1.0
        let mse_d = &result.metrics["mse_delta"];
        assert!(
            (mse_d[1] - 1.0).abs() < 1e-10,
            "B mse_delta expected 1.0, got {}",
            mse_d[1]
        );

        // MAE = (1 + 1 + 1) / 3 = 1.0
        let mae_d = &result.metrics["mae_delta"];
        assert!(
            (mae_d[1] - 1.0).abs() < 1e-10,
            "B mae_delta expected 1.0, got {}",
            mae_d[1]
        );

        // B: pearson of [4,2,1] vs [5,3,2] — both are perfectly linearly related (shifted by 1)
        // pearson should be 1.0
        assert!(
            (pearson[1] - 1.0).abs() < 1e-10,
            "B pearson_delta expected 1.0, got {}",
            pearson[1]
        );
    }

    #[test]
    fn test_single_perturbation() {
        // Only 2 groups: ctrl + one perturbation
        let means_real = vec![0.0, 0.0, 1.0, 2.0];
        let means_pred = vec![0.0, 0.0, 1.5, 2.5];
        let pert_names: Vec<String> = vec!["ctrl", "A"].into_iter().map(String::from).collect();

        let result = compute_bulk_metrics(
            &means_real,
            &means_pred,
            0,
            2,
            2,
            &pert_names,
            &[
                BulkMetric::Mse,
                BulkMetric::Mae,
                BulkMetric::PearsonDelta,
                BulkMetric::MseDelta,
                BulkMetric::MaeDelta,
            ],
        )
        .unwrap();

        assert_eq!(result.pert_names.len(), 1);
        assert_eq!(result.pert_names[0], "A");
    }

    #[test]
    fn test_all_metrics_in_one_call() {
        let means_real = vec![
            0.0, 0.0, // ctrl
            2.0, 4.0, // A
        ];
        let means_pred = vec![
            0.0, 0.0, // ctrl
            3.0, 5.0, // A
        ];
        let pert_names: Vec<String> = vec!["ctrl", "A"].into_iter().map(String::from).collect();

        let result = compute_bulk_metrics(
            &means_real,
            &means_pred,
            0,
            2,
            2,
            &pert_names,
            &[
                BulkMetric::PearsonDelta,
                BulkMetric::Mse,
                BulkMetric::Mae,
                BulkMetric::MseDelta,
                BulkMetric::MaeDelta,
            ],
        )
        .unwrap();

        // All 5 metric keys should be present
        assert_eq!(result.metrics.len(), 5);
        assert!(result.metrics.contains_key("pearson_delta"));
        assert!(result.metrics.contains_key("mse"));
        assert!(result.metrics.contains_key("mae"));
        assert!(result.metrics.contains_key("mse_delta"));
        assert!(result.metrics.contains_key("mae_delta"));
    }

    #[test]
    fn test_validation_errors() {
        let pert_names: Vec<String> = vec!["ctrl", "A"].into_iter().map(String::from).collect();

        // Wrong means_real length
        let err = compute_bulk_metrics(
            &[0.0; 3], // wrong: should be 4
            &[0.0; 4],
            0,
            2,
            2,
            &pert_names,
            &[BulkMetric::Mse],
        );
        assert!(err.is_err());

        // ctrl_idx out of range
        let err = compute_bulk_metrics(
            &[0.0; 4],
            &[0.0; 4],
            5, // out of range
            2,
            2,
            &pert_names,
            &[BulkMetric::Mse],
        );
        assert!(err.is_err());
    }
}
