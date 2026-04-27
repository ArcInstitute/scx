//! Energy distance computation for perturbation evaluation.
//!
//! Implements the e-distance metric from cell-eval: for each perturbation,
//! computes the energy distance between perturbation and control cells on
//! both real and predicted sides, then returns the Pearson correlation
//! of per-perturbation e-distances.
//!
//! The key optimization over cell-eval's sklearn-based approach:
//! - No `[N, N]` pairwise distance matrix allocation.
//! - Control self-distances precomputed once and reused.
//! - Per-perturbation computation parallelized with rayon.

use std::collections::HashMap;

use super::distances::{mean_pairwise_distance, mean_pairwise_distance_self, DistanceBackend};
use super::DistanceMetric;
use crate::eval_metrics::bulk_metrics::pearson_correlation;
use rayon::prelude::*;

/// Result of energy distance computation.
#[derive(Debug, Clone)]
pub struct EDistanceResult {
    /// Per-perturbation e-distances (real side).
    pub d_real: Vec<f64>,
    /// Per-perturbation e-distances (predicted side).
    pub d_pred: Vec<f64>,
    /// Pearson correlation between d_real and d_pred.
    pub correlation: f64,
    /// Perturbation names in order.
    pub pert_names: Vec<String>,
}

/// Compute fused e-distance: `2*D(X,Y) - D(X,X) - sigma_y` where `sigma_y`
/// is the precomputed mean self-distance of Y (control cells).
///
/// This avoids computing the control self-distance repeatedly across
/// perturbations.
#[allow(clippy::too_many_arguments)]
pub fn fused_edistance(
    x: &[f64],
    y: &[f64],
    n_x: usize,
    n_y: usize,
    n_dims: usize,
    sigma_y: f64,
    metric: DistanceMetric,
    backend: DistanceBackend,
) -> crate::Result<f64> {
    let sigma_x = mean_pairwise_distance_self(x, n_x, n_dims, metric, backend)?;
    let delta = mean_pairwise_distance(x, y, n_x, n_y, n_dims, metric, backend)?;
    Ok(2.0 * delta - sigma_x - sigma_y)
}

/// Compute energy distances for all perturbations, returning per-perturbation
/// e-distances on both real and predicted sides plus their Pearson correlation.
///
/// # Arguments
/// * `real_cells` — `[N_real × D]` row-major dense matrix (all real cells).
/// * `pred_cells` — `[N_pred × D]` row-major dense matrix (all predicted cells).
/// * `real_groups` — Per-cell group index for real cells (length = `N_real`).
///   Each value is an index into `pert_names` (or `ctrl_group_idx`).
/// * `pred_groups` — Per-cell group index for predicted cells (length = `N_pred`).
/// * `ctrl_group_idx` — Group index of the control group.
/// * `pert_names` — Names of non-control perturbations to evaluate.
/// * `pert_group_indices` — Group indices corresponding to `pert_names`.
/// * `n_dims` — Dimensionality `D`.
/// * `metric` — Distance metric.
///
/// # Returns
/// `EDistanceResult` with per-perturbation e-distances and their correlation.
#[allow(clippy::too_many_arguments)]
pub fn compute_energy_distance(
    real_cells: &[f64],
    pred_cells: &[f64],
    real_groups: &[u32],
    pred_groups: &[u32],
    ctrl_group_idx: u32,
    pert_names: &[String],
    pert_group_indices: &[u32],
    n_dims: usize,
    metric: DistanceMetric,
    backend: DistanceBackend,
) -> crate::Result<EDistanceResult> {
    if pert_names.len() != pert_group_indices.len() {
        return Err(crate::AccelError::InvalidInput(
            "pert_names and pert_group_indices must have the same length".to_string(),
        ));
    }
    if pert_names.is_empty() {
        return Err(crate::AccelError::InvalidInput(
            "no perturbation groups provided".to_string(),
        ));
    }

    let n_real = real_groups.len();
    let n_pred = pred_groups.len();
    if real_cells.len() != n_real * n_dims {
        return Err(crate::AccelError::InvalidInput(format!(
            "real_cells length {} doesn't match n_real={} × n_dims={}",
            real_cells.len(),
            n_real,
            n_dims
        )));
    }
    if pred_cells.len() != n_pred * n_dims {
        return Err(crate::AccelError::InvalidInput(format!(
            "pred_cells length {} doesn't match n_pred={} × n_dims={}",
            pred_cells.len(),
            n_pred,
            n_dims
        )));
    }

    // ── 1. Pre-index all groups (avoids repeated O(N) scans inside par_iter) ──
    let real_index = build_group_index(real_groups);
    let pred_index = build_group_index(pred_groups);

    // ── 2. Extract control cells ────────────────────────────────────
    let ctrl_real = extract_group_rows_indexed(real_cells, real_index.get(&ctrl_group_idx), n_dims);
    let ctrl_pred = extract_group_rows_indexed(pred_cells, pred_index.get(&ctrl_group_idx), n_dims);

    let n_ctrl_real = ctrl_real.len() / n_dims;
    let n_ctrl_pred = ctrl_pred.len() / n_dims;

    if n_ctrl_real == 0 || n_ctrl_pred == 0 {
        return Err(crate::AccelError::InvalidInput(
            "control group has no cells".to_string(),
        ));
    }

    // ── 3. Precompute control self-distances (once each) ────────────
    let sigma_ctrl_real =
        mean_pairwise_distance_self(&ctrl_real, n_ctrl_real, n_dims, metric, backend)?;
    let sigma_ctrl_pred =
        mean_pairwise_distance_self(&ctrl_pred, n_ctrl_pred, n_dims, metric, backend)?;

    // ── 4. Compute per-perturbation e-distances in parallel ─────────
    let results: Vec<crate::Result<(f64, f64)>> = pert_group_indices
        .par_iter()
        .map(|&gi| -> crate::Result<(f64, f64)> {
            let pert_real = extract_group_rows_indexed(real_cells, real_index.get(&gi), n_dims);
            let pert_pred = extract_group_rows_indexed(pred_cells, pred_index.get(&gi), n_dims);

            let n_pert_real = pert_real.len() / n_dims;
            let n_pert_pred = pert_pred.len() / n_dims;

            let e_real = if n_pert_real > 0 {
                fused_edistance(
                    &pert_real,
                    &ctrl_real,
                    n_pert_real,
                    n_ctrl_real,
                    n_dims,
                    sigma_ctrl_real,
                    metric,
                    backend,
                )?
            } else {
                f64::NAN
            };

            let e_pred = if n_pert_pred > 0 {
                fused_edistance(
                    &pert_pred,
                    &ctrl_pred,
                    n_pert_pred,
                    n_ctrl_pred,
                    n_dims,
                    sigma_ctrl_pred,
                    metric,
                    backend,
                )?
            } else {
                f64::NAN
            };

            Ok((e_real, e_pred))
        })
        .collect();
    let results: Vec<(f64, f64)> = results.into_iter().collect::<crate::Result<Vec<_>>>()?;

    let d_real: Vec<f64> = results.iter().map(|(r, _)| *r).collect();
    let d_pred: Vec<f64> = results.iter().map(|(_, p)| *p).collect();

    // ── 5. Pearson correlation of e-distance vectors ────────────────
    // Filter out NaN pairs before computing correlation.
    let valid: Vec<(f64, f64)> = d_real
        .iter()
        .zip(d_pred.iter())
        .filter(|(r, p)| !r.is_nan() && !p.is_nan())
        .map(|(&r, &p)| (r, p))
        .collect();

    let correlation = if valid.len() >= 2 {
        let vr: Vec<f64> = valid.iter().map(|(r, _)| *r).collect();
        let vp: Vec<f64> = valid.iter().map(|(_, p)| *p).collect();
        pearson_correlation(&vr, &vp)
    } else {
        f64::NAN
    };

    Ok(EDistanceResult {
        d_real,
        d_pred,
        correlation,
        pert_names: pert_names.to_vec(),
    })
}

/// Build a group index: maps each group ID to the list of row indices
/// belonging to that group. This is done once before the parallel loop
/// to avoid repeated O(N) scans inside `par_iter`.
fn build_group_index(groups: &[u32]) -> HashMap<u32, Vec<usize>> {
    let mut index: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, &g) in groups.iter().enumerate() {
        index.entry(g).or_default().push(i);
    }
    index
}

/// Extract rows from a `[N × D]` matrix using a pre-built group index.
///
/// Returns a new dense `Vec<f64>` containing the selected rows contiguously.
fn extract_group_rows_indexed(
    data: &[f64],
    row_indices: Option<&Vec<usize>>,
    n_dims: usize,
) -> Vec<f64> {
    let indices = match row_indices {
        Some(v) => v,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(indices.len() * n_dims);
    for &i in indices {
        let start = i * n_dims;
        out.extend_from_slice(&data[start..start + n_dims]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_data() -> (Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        // 3 groups: 0=control, 1=pert_A, 2=pert_B
        // 2 dimensions for simplicity
        // Real: ctrl at origin, A shifted right, B shifted up
        let real_cells = vec![
            // ctrl cells (group 0)
            0.0, 0.0, // cell 0
            0.1, 0.1, // cell 1
            // pert_A cells (group 1)
            5.0, 0.0, // cell 2
            5.1, 0.1, // cell 3
            // pert_B cells (group 2)
            0.0, 5.0, // cell 4
            0.1, 5.1, // cell 5
        ];
        let real_groups = vec![0, 0, 1, 1, 2, 2];

        // Pred: similar but with some noise
        let pred_cells = vec![
            // ctrl cells (group 0)
            0.0, 0.0, // cell 0
            0.2, 0.2, // cell 1
            // pert_A cells (group 1)
            4.8, 0.2, // cell 2
            5.2, -0.1, // cell 3
            // pert_B cells (group 2)
            0.2, 4.8, // cell 4
            -0.1, 5.2, // cell 5
        ];
        let pred_groups = vec![0, 0, 1, 1, 2, 2];

        (real_cells, pred_cells, real_groups, pred_groups)
    }

    #[test]
    fn test_fused_edistance_basic() {
        // X = [[5, 0]], Y = [[0, 0]] (single cell each)
        // sigma_y = 0 (single cell)
        // D(X,Y) = 5.0, D(X,X) = 0
        // E = 2*5 - 0 - 0 = 10.0
        let x = vec![5.0, 0.0];
        let y = vec![0.0, 0.0];
        let sigma_y = 0.0;
        let e = fused_edistance(
            &x,
            &y,
            1,
            1,
            2,
            sigma_y,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        assert!((e - 10.0).abs() < 1e-12, "expected 10.0, got {e}");
    }

    #[test]
    fn test_fused_edistance_identical() {
        // When X == Y, e-distance should be ~0
        let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0];
        let n = 3;
        let d = 2;
        let sigma = mean_pairwise_distance_self(
            &data,
            n,
            d,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        let e = fused_edistance(
            &data,
            &data,
            n,
            n,
            d,
            sigma,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();
        assert!(e.abs() < 1e-12, "identical distributions → e ≈ 0, got {e}");
    }

    #[test]
    fn test_compute_energy_distance_basic() {
        let (real_cells, pred_cells, real_groups, pred_groups) = make_test_data();
        let pert_names = vec!["A".to_string(), "B".to_string()];
        let pert_indices = vec![1, 2];

        let result = compute_energy_distance(
            &real_cells,
            &pred_cells,
            &real_groups,
            &pred_groups,
            0, // ctrl group
            &pert_names,
            &pert_indices,
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();

        assert_eq!(result.pert_names, vec!["A", "B"]);
        assert_eq!(result.d_real.len(), 2);
        assert_eq!(result.d_pred.len(), 2);
        // E-distances should be positive (perturbations are far from control)
        assert!(result.d_real[0] > 0.0, "A real e-dist should be positive");
        assert!(result.d_real[1] > 0.0, "B real e-dist should be positive");
        assert!(result.d_pred[0] > 0.0, "A pred e-dist should be positive");
        assert!(result.d_pred[1] > 0.0, "B pred e-dist should be positive");
    }

    #[test]
    fn test_identical_real_pred_high_correlation() {
        // When pred == real, e-distances are identical → correlation = 1.0.
        // Need ≥3 perturbations with *different* e-distances to avoid zero-variance.
        // 1D data: ctrl=0..1, pertA=10..11, pertB=20..21, pertC=50..51
        let cells = vec![
            0.0, 1.0, // ctrl (group 0)
            10.0, 11.0, // A (group 1) — moderate distance
            20.0, 21.0, // B (group 2) — large distance
            50.0, 51.0, // C (group 3) — very large distance
        ];
        let groups = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
        let pert_names = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let pert_indices = vec![1u32, 2, 3];
        let n_dims = 1;

        let result = compute_energy_distance(
            &cells,
            &cells, // identical
            &groups,
            &groups,
            0,
            &pert_names,
            &pert_indices,
            n_dims,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();

        // d_real should equal d_pred element-wise
        for i in 0..3 {
            assert!(
                (result.d_real[i] - result.d_pred[i]).abs() < 1e-12,
                "d_real[{i}]={} != d_pred[{i}]={}",
                result.d_real[i],
                result.d_pred[i]
            );
        }

        assert!(
            (result.correlation - 1.0).abs() < 1e-10,
            "identical data → correlation 1.0, got {}",
            result.correlation
        );
    }

    #[test]
    fn test_energy_distance_validation() {
        // Empty pert_names
        let result = compute_energy_distance(
            &[0.0; 4],
            &[0.0; 4],
            &[0, 0],
            &[0, 0],
            0,
            &[],
            &[],
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        );
        assert!(result.is_err());

        // Mismatched pert_names / indices
        let result = compute_energy_distance(
            &[0.0; 4],
            &[0.0; 4],
            &[0, 0],
            &[0, 0],
            0,
            &["A".to_string()],
            &[1, 2], // length mismatch
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_energy_distance_no_ctrl_cells() {
        // ctrl_group_idx doesn't appear in groups → error
        let result = compute_energy_distance(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[1, 1],
            &[1, 1],
            0, // no cells with group 0
            &["A".to_string()],
            &[1],
            2,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_energy_distance_with_manual_computation() {
        // Small manual test: 1D, 2 ctrl cells, 2 pert cells
        // Real: ctrl=[0, 1], pert=[10, 11]
        // Pred: ctrl=[0, 1], pert=[10, 11] (identical to real)
        let real_cells = vec![0.0, 1.0, 10.0, 11.0]; // 1D data, 4 cells
        let pred_cells = vec![0.0, 1.0, 10.0, 11.0];
        let real_groups = vec![0, 0, 1, 1];
        let pred_groups = vec![0, 0, 1, 1];
        let n_dims = 1;

        // Manual computation for real side:
        // ctrl = [0, 1], pert = [10, 11]
        // sigma_ctrl = mean_pairwise_self([0, 1], 2, 1) = 2*1/(2*2) = 0.5
        // D(pert, ctrl) = mean([|10-0|, |10-1|, |11-0|, |11-1|]) = (10+9+11+10)/4 = 10.0
        // D(pert, pert) = mean_pairwise_self([10, 11], 2, 1) = 2*1/(2*2) = 0.5
        // e = 2*10.0 - 0.5 - 0.5 = 19.0
        let result = compute_energy_distance(
            &real_cells,
            &pred_cells,
            &real_groups,
            &pred_groups,
            0,
            &["pert".to_string()],
            &[1],
            n_dims,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();

        assert!(
            (result.d_real[0] - 19.0).abs() < 1e-10,
            "expected e_real=19.0, got {}",
            result.d_real[0]
        );
        assert!(
            (result.d_pred[0] - 19.0).abs() < 1e-10,
            "expected e_pred=19.0, got {}",
            result.d_pred[0]
        );
    }

    #[test]
    fn test_negative_edistance_overlapping_distributions() {
        // E-distance can be negative when perturbation cells are embedded
        // within the control cloud (within-group variance > between-group).
        //
        // e = 2*D(pert,ctrl) - D(pert,pert) - D(ctrl,ctrl)
        //
        // Strategy: single pert cell at the control centroid (5.0).
        // D(pert,pert) = 0 (single cell), D(pert,ctrl) = mean |5 - c_i|,
        // D(ctrl,ctrl) = mean pairwise distance of ctrl (large).
        //
        // For ctrl = [0, 5, 10]:
        //   D(ctrl,ctrl) = (|0-5|+|0-10|+|5-10|)*2/9 = (5+10+5)*2/9 = 40/9 ≈ 4.44
        //   D(pert,ctrl) = (|5-0|+|5-5|+|5-10|)/3 = (5+0+5)/3 ≈ 3.33
        //   e = 2*3.33 - 0 - 4.44 = 6.67 - 4.44 = 2.22 (still positive!)
        //
        // For ctrl = [0, 2, 4, 6, 8, 10], pert = [5.0]:
        //   D(ctrl,ctrl) = 2*sum_upper/36, sum_upper = 2+4+6+8+10+2+4+6+8+2+4+6+2+4+2 = 60
        //   D(ctrl,ctrl) = 120/36 ≈ 3.33
        //   D(pert,ctrl) = (5+3+1+1+3+5)/6 = 18/6 = 3.0
        //   e = 2*3.0 - 0 - 3.33 = 2.67 (still positive)
        //
        // For negative e-distance, we need D(pert,pert) + D(ctrl,ctrl) > 2*D(pert,ctrl).
        // Use 2 pert cells at exactly the same spot (D(pert,pert)=0)
        // and very spread ctrl: ctrl = [-100, 0, 100], pert = [0, 0]
        //   D(ctrl,ctrl) = 2*(100+200+100)/9 = 800/9 ≈ 88.89
        //   D(pert,ctrl) = (100+0+100 + 100+0+100)/6 = 400/6 ≈ 66.67
        //   e = 2*66.67 - 0 - 88.89 = 133.33 - 88.89 = 44.44 (still positive!)
        //
        // The math shows e-distance is always >= 0 for metric distances.
        // Actually, the energy distance IS non-negative for true metric distances.
        // The "e-distance can be negative" from the review is incorrect for metric
        // distances — it only applies to non-metric kernels or the sample estimate
        // correction term. Let's verify that overlapping distributions give
        // a SMALL positive e-distance instead.
        let cells = vec![
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0, // ctrl (group 0) — wide spread
            4.9, 5.0, 5.1, // pert (group 1) — tight cluster inside ctrl
        ];
        let groups = vec![0u32, 0, 0, 0, 0, 0, 1, 1, 1];
        let n_dims = 1;

        let result = compute_energy_distance(
            &cells,
            &cells,
            &groups,
            &groups,
            0,
            &["pert".to_string()],
            &[1],
            n_dims,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();

        // E-distance should be small but non-negative for overlapping distributions
        // (energy distance is a proper metric and is >= 0).
        assert!(
            result.d_real[0] >= 0.0,
            "e-distance should be non-negative for metric distances, got {}",
            result.d_real[0]
        );
        // Should be much smaller than for well-separated distributions
        assert!(
            result.d_real[0] < 5.0,
            "overlapping distributions should give small e-distance, got {}",
            result.d_real[0]
        );
    }

    #[test]
    fn test_many_perturbations_parallel() {
        // Stress test with 50 perturbations to exercise the rayon par_iter
        // codepath and verify no race conditions.
        let n_perts = 50;
        let cells_per_group = 10;
        let n_dims = 3;
        let n_total = (n_perts + 1) * cells_per_group; // +1 for control

        // Build data: control at origin, each pert shifted by (group_id, 0, 0)
        let mut cells = Vec::with_capacity(n_total * n_dims);
        let mut groups = Vec::with_capacity(n_total);

        // Control cells (group 0)
        for _ in 0..cells_per_group {
            cells.extend_from_slice(&[0.0, 0.0, 0.0]);
            groups.push(0u32);
        }
        // Perturbation cells
        for p in 1..=n_perts {
            for _ in 0..cells_per_group {
                cells.extend_from_slice(&[p as f64 * 2.0, 0.0, 0.0]);
                groups.push(p as u32);
            }
        }

        let pert_names: Vec<String> = (1..=n_perts).map(|p| format!("pert_{p}")).collect();
        let pert_indices: Vec<u32> = (1..=n_perts as u32).collect();

        let result = compute_energy_distance(
            &cells,
            &cells,
            &groups,
            &groups,
            0,
            &pert_names,
            &pert_indices,
            n_dims,
            DistanceMetric::Euclidean,
            DistanceBackend::Scalar,
        )
        .unwrap();

        assert_eq!(result.d_real.len(), n_perts);
        assert_eq!(result.d_pred.len(), n_perts);

        // All e-distances should be positive (perts far from control)
        for (i, &d) in result.d_real.iter().enumerate() {
            assert!(
                d > 0.0,
                "pert_{} e-distance should be positive, got {d}",
                i + 1
            );
        }

        // d_real should equal d_pred (identical data)
        for i in 0..n_perts {
            assert!(
                (result.d_real[i] - result.d_pred[i]).abs() < 1e-10,
                "d_real[{i}] != d_pred[{i}]"
            );
        }

        // E-distances should increase with distance from control
        for i in 1..n_perts {
            assert!(
                result.d_real[i] > result.d_real[i - 1],
                "e-distances should be monotonically increasing"
            );
        }

        // Correlation should be 1.0 (identical data)
        assert!(
            (result.correlation - 1.0).abs() < 1e-10,
            "correlation should be 1.0 for identical data, got {}",
            result.correlation
        );
    }
}
