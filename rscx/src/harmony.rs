//! Harmony2 batch integration — R bindings via extendr.
//!
//! Thin wrapper around [`scx_accel::harmony_integrate`]. Accepts an
//! `N x d` numeric matrix (R is column-major) and a character vector of
//! batch labels, factorises the labels to contiguous integer codes, and
//! returns a named list with the corrected embeddings, convergence flag,
//! iteration count, and per-iteration objective.

use extendr_api::prelude::*;

use scx_accel::{harmony_integrate, BatchCovariate, HarmonyConfig};

use crate::util::factorize_chars;

/// Run Harmony2 batch integration on PCA-style embeddings.
///
/// @param embeddings Numeric matrix (N rows × d columns). R passes this as
///   a column-major flat buffer; the binding repacks into the row-major
///   `f32` layout expected by `scx-accel`.
/// @param batch Character vector of length N with batch labels. Factorised
///   internally into contiguous integer codes.
/// @param n_clusters Integer or NULL. Defaults to `min(N/30, 100)`.
/// @param theta Numeric scalar — diversity penalty strength (default 2.0).
/// @param sigma Numeric scalar — kernel bandwidth (default 0.1).
/// @param lambda Numeric scalar or NULL — ridge penalty. NULL enables
///   dynamic estimation (recommended).
/// @param alpha Numeric — dynamic lambda scale factor (default 0.2).
/// @param max_iter Integer — maximum Harmony iterations (default 10).
/// @param max_iter_kmeans Integer — k-means sub-iterations (default 6; must be
///   >= 2*window_size so the k-means convergence check can fire).
/// @param epsilon_harmony Numeric — Harmony convergence tolerance
///   (default 1e-2).
/// @param epsilon_kmeans Numeric — k-means convergence tolerance
///   (default 1e-3).
/// @param block_size Numeric — stochastic block size (fraction of N)
///   (default 0.05).
/// @param batch_prop_cutoff Numeric — minimum batch proportion per
///   cluster (default 1e-5).
/// @param tau Numeric — overcorrection protection (default 0.0).
/// @param random_state Integer — RNG seed (default 0).
///
/// @return A named list: `$embeddings` (N × d numeric matrix in R's
///   column-major layout), `$converged` (logical), `$n_iterations`
///   (integer), `$n_clusters` (integer), `$objective` (numeric vector).
/// Returns `Robj` and throws a clean R error via `throw_on_err`: a fallible
/// `#[extendr]` fn would otherwise `unwrap()`-panic in extendr 0.8.0, masking
/// the real message behind "User function panicked". See B3/B7.
#[extendr]
#[allow(clippy::too_many_arguments)]
fn scx_harmony_integrate(
    embeddings: RMatrix<f64>,
    batch: Strings,
    n_clusters: Nullable<i32>,
    theta: f64,
    sigma: f64,
    lambda: Nullable<f64>,
    alpha: f64,
    max_iter: i32,
    max_iter_kmeans: i32,
    epsilon_harmony: f64,
    epsilon_kmeans: f64,
    block_size: f64,
    batch_prop_cutoff: f64,
    tau: f64,
    random_state: i32,
) -> Robj {
    crate::util::throw_on_err(scx_harmony_integrate_impl(
        embeddings,
        batch,
        n_clusters,
        theta,
        sigma,
        lambda,
        alpha,
        max_iter,
        max_iter_kmeans,
        epsilon_harmony,
        epsilon_kmeans,
        block_size,
        batch_prop_cutoff,
        tau,
        random_state,
    ))
}

#[allow(clippy::too_many_arguments)]
fn scx_harmony_integrate_impl(
    embeddings: RMatrix<f64>,
    batch: Strings,
    n_clusters: Nullable<i32>,
    theta: f64,
    sigma: f64,
    lambda: Nullable<f64>,
    alpha: f64,
    max_iter: i32,
    max_iter_kmeans: i32,
    epsilon_harmony: f64,
    epsilon_kmeans: f64,
    block_size: f64,
    batch_prop_cutoff: f64,
    tau: f64,
    random_state: i32,
) -> Result<Robj> {
    let n_obs = embeddings.nrows();
    let n_pcs = embeddings.ncols();
    if n_obs == 0 || n_pcs == 0 {
        return Err(Error::Other(format!(
            "embeddings matrix is empty ({n_obs}x{n_pcs})",
        )));
    }

    let batch_strs: Vec<String> = batch.iter().map(|s| s.to_string()).collect();
    if batch_strs.len() != n_obs {
        return Err(Error::Other(format!(
            "batch vector length {} != n_obs {}",
            batch_strs.len(),
            n_obs
        )));
    }
    let (labels, levels) = factorize_chars(&batch_strs);
    let n_levels = levels.len();
    if n_levels < 2 {
        return Err(Error::Other(format!(
            "batch has only {n_levels} level(s); Harmony requires >= 2",
        )));
    }

    // R passes `embeddings` in column-major layout: element (i, j) is at
    // index `j * n_obs + i`. scx-accel expects row-major f32 of shape N x d
    // where row i has d contiguous PCs. Repack and cast in one pass.
    let col_major = embeddings.data();
    let mut emb_f32 = vec![0f32; n_obs * n_pcs];
    for i in 0..n_obs {
        for j in 0..n_pcs {
            let v = col_major[j * n_obs + i];
            if !v.is_finite() {
                return Err(Error::Other(format!(
                    "embeddings[{}, {}] is NaN or Inf",
                    i + 1,
                    j + 1
                )));
            }
            emb_f32[i * n_pcs + j] = v as f32;
        }
    }

    let covariate = BatchCovariate {
        labels,
        n_levels,
        name: Some("batch".into()),
    };

    // Validate signed i32 inputs before casting to usize/u64.
    if max_iter < 1 || max_iter_kmeans < 1 {
        return Err(Error::Other(
            "max_iter and max_iter_kmeans must be >= 1".into(),
        ));
    }
    if random_state < 0 {
        return Err(Error::Other("random_state must be >= 0".into()));
    }
    let n_clusters_opt = match n_clusters {
        Nullable::NotNull(k) if k >= 2 => Some(k as usize),
        Nullable::NotNull(k) => {
            return Err(Error::Other(format!("n_clusters must be >= 2 (got {k})")));
        }
        Nullable::Null => None,
    };
    let lambda_opt = match lambda {
        Nullable::NotNull(v) if v >= 0.0 => Some(vec![v]),
        Nullable::NotNull(v) => {
            return Err(Error::Other(format!("lambda must be >= 0 (got {v})")));
        }
        Nullable::Null => None,
    };

    let config = HarmonyConfig {
        n_clusters: n_clusters_opt,
        theta: Some(vec![theta]),
        sigma,
        lambda: lambda_opt,
        alpha,
        max_iter: max_iter as usize,
        max_iter_kmeans: max_iter_kmeans as usize,
        epsilon_harmony,
        epsilon_kmeans,
        window_size: 3,
        block_size,
        batch_prop_cutoff,
        tau,
        random_state: random_state as u64,
        n_threads: None,
    };

    let result = harmony_integrate(&emb_f32, n_obs, n_pcs, &[covariate], &config)
        .map_err(|e| Error::Other(format!("harmony_integrate: {e}")))?;

    // Repack N×d row-major f64 output back to R's column-major layout.
    let mut col_major_out = vec![0f64; n_obs * n_pcs];
    for i in 0..n_obs {
        for j in 0..n_pcs {
            col_major_out[j * n_obs + i] = result.z_corrected[i * n_pcs + j];
        }
    }

    let n_obs_i = n_obs as i32;
    let n_pcs_i = n_pcs as i32;
    let converged = result.converged;
    let n_iterations = result.n_iterations as i32;
    let n_clusters_out = result.n_clusters as i32;
    let objective = result.objective_harmony.clone();

    R!("list(
        embeddings = matrix({{col_major_out}}, nrow = {{n_obs_i}}, ncol = {{n_pcs_i}}),
        converged = {{converged}},
        n_iterations = {{n_iterations}},
        n_clusters = {{n_clusters_out}},
        objective = {{objective}}
    )")
    .map_err(|e| Error::Other(e.to_string()))
}

extendr_module! {
    mod harmony;
    fn scx_harmony_integrate;
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize_stable_order() {
        let labels: Vec<String> = ["B", "A", "A", "B", "C", "A"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (codes, levels) = factorize_chars(&labels);
        assert_eq!(levels.len(), 3);
        // First-seen order: B→0, A→1, C→2.
        assert_eq!(levels, vec!["B", "A", "C"]);
        assert_eq!(codes, vec![0, 1, 1, 0, 2, 1]);
    }
}
