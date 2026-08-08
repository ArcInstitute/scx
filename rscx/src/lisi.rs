//! Local Inverse Simpson Index (LISI) — R bindings via extendr.
//!
//! Thin wrapper around [`scx_accel::compute_lisi`]. Accepts an `N x d`
//! numeric matrix (column-major per R convention) and a character vector
//! of categorical labels; returns the per-cell LISI as a numeric vector.

use std::collections::HashMap;

use extendr_api::prelude::*;

use scx_accel::{compute_lisi, LisiConfig};

/// Factorise a character vector into contiguous `u32` level codes
/// (first-seen order). Matches the helper in `harmony.rs`.
fn factorize_chars(labels: &[String]) -> (Vec<u32>, usize) {
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut codes: Vec<u32> = Vec::with_capacity(labels.len());
    let mut next: u32 = 0;
    for s in labels {
        let code = match map.get(s) {
            Some(&c) => c,
            None => {
                let c = next;
                map.insert(s.clone(), c);
                next += 1;
                c
            }
        };
        codes.push(code);
    }
    (codes, next as usize)
}

/// Compute per-cell Local Inverse Simpson Index.
///
/// @param embeddings Numeric matrix (N rows × d columns).
/// @param labels Character vector of length N with categorical labels.
/// @param perplexity Gaussian-kernel target perplexity (default 30).
/// @param n_neighbors Number of neighbours to use. NULL means
///   `ceil(3 * perplexity)`.
///
/// @return Numeric vector of length N with LISI values.
///
/// Returns `Robj` and throws a clean R error via `throw_on_err`: a fallible
/// `#[extendr]` fn would otherwise `unwrap()`-panic in extendr 0.8.0, masking
/// the real message behind "User function panicked". See B3/B7.
#[extendr]
fn scx_compute_lisi(
    embeddings: RMatrix<f64>,
    labels: Strings,
    perplexity: f64,
    n_neighbors: Nullable<i32>,
) -> Robj {
    crate::util::throw_on_err(scx_compute_lisi_impl(
        embeddings,
        labels,
        perplexity,
        n_neighbors,
    ))
}

fn scx_compute_lisi_impl(
    embeddings: RMatrix<f64>,
    labels: Strings,
    perplexity: f64,
    n_neighbors: Nullable<i32>,
) -> Result<Robj> {
    let n_obs = embeddings.nrows();
    let n_dims = embeddings.ncols();
    if n_obs == 0 || n_dims == 0 {
        return Err(Error::Other(format!(
            "embeddings matrix is empty ({n_obs}x{n_dims})",
        )));
    }
    // `is_finite` first: NaN compares false against everything, so a lone
    // `perplexity <= 0.0` passed it through and the computation degenerated to
    // an all-1.0 result with no error (see `scx_accel::lisi::compute_lisi`,
    // which carries the load-bearing copy of this guard). Kept here too so the
    // message names the R-facing argument rather than a downstream crate's.
    if !perplexity.is_finite() || perplexity <= 0.0 {
        return Err(Error::Other(format!(
            "perplexity must be a finite positive number (got {perplexity})"
        )));
    }

    let label_strs: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
    if label_strs.len() != n_obs {
        return Err(Error::Other(format!(
            "labels length {} != n_obs {}",
            label_strs.len(),
            n_obs
        )));
    }
    let (labels_u32, _n_levels) = factorize_chars(&label_strs);

    // R column-major → row-major f32 repack.
    let col_major = embeddings.data();
    let mut emb_f32 = vec![0f32; n_obs * n_dims];
    for i in 0..n_obs {
        for j in 0..n_dims {
            let v = col_major[j * n_obs + i];
            if !v.is_finite() {
                return Err(Error::Other(format!(
                    "embeddings[{}, {}] is NaN or Inf",
                    i + 1,
                    j + 1
                )));
            }
            emb_f32[i * n_dims + j] = v as f32;
        }
    }

    let n_neighbors = match n_neighbors {
        Nullable::NotNull(k) if k >= 1 => k as usize,
        Nullable::NotNull(k) => {
            return Err(Error::Other(format!("n_neighbors must be >= 1 (got {k})")));
        }
        Nullable::Null => (perplexity * 3.0).ceil() as usize,
    };

    let config = LisiConfig {
        perplexity,
        n_neighbors,
        ..Default::default()
    };

    let result = compute_lisi(&emb_f32, n_obs, n_dims, &labels_u32, &config)
        .map_err(|e| Error::Other(format!("compute_lisi: {e}")))?;

    Ok(Robj::from(result.lisi))
}

extendr_module! {
    mod lisi;
    fn scx_compute_lisi;
}
