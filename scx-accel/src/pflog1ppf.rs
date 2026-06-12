//! PFlog1pPF / shifted centered-log-ratio normalization (Booeshaghi et al. 2026).
//!
//! Per cell, counts are converted to within-cell proportions, shifted by a
//! positive pseudocount `c`, log-transformed, and centered within the cell:
//!
//! ```text
//! z_ij = log(x_ij / s_i + c) − (1/D) Σ_k log(x_ik / s_i + c)
//! ```
//!
//! where `s_i = Σ_j x_ij` is the cell depth and `D = n_vars`. For `c = 1` this
//! is `z_ij = log1p(x_ij / s_i) − mean_j log1p(x_ij / s_i)`.
//!
//! ## Compact exact representation
//!
//! The exact `Z` is **dense** — every original zero in cell `i` collapses to a
//! shared value — so materializing it is `O(n_obs · n_vars)`. Instead `Z`
//! decomposes exactly into a sparse part plus a per-cell baseline:
//!
//! ```text
//! Z = delta + baseline · 1ᵀ
//! delta_ij = log1p(x_ij / (c · s_i))   for x_ij > 0, else 0   (sparse, X's pattern)
//! baseline_i = −(1/D) · Σ_j delta_ij                          (dense, length n_obs)
//! ```
//!
//! `delta` is exactly the existing lazy transform chain
//! `NormalizeTotal{target_sum = 1/c} → Log1p` applied per shard, so this module
//! only computes `baseline` (here) and teaches PCA about the rank-1 baseline
//! offset (see [`crate::pca::pflog1ppf_pca`]). No new codec, no format change.
//!
//! ## Numerics
//!
//! Per-element `delta` rides on `f32` CSR values, but the baseline reduction
//! accumulates in `f64`, so `baseline` stays accurate while the stored
//! deviations match the rest of the streaming kernels (~`1e-6` floor).

use crate::error::{AccelError, Result};
use scx_format_io::ShardSource;

/// Reject non-finite values at the normalization boundary (mirrors the
/// gene-scoring / HVG checks): a NaN/Inf would silently poison the baseline.
fn ensure_finite(data: &[f32]) -> Result<()> {
    if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
        return Err(AccelError::InvalidInput(format!(
            "PFlog1pPF input contains a non-finite value ({}) at nonzero index {pos}; \
             PFlog1pPF requires finite raw counts — filter/QC NaN and Inf first",
            data[pos]
        )));
    }
    Ok(())
}

/// Exact PFlog1pPF representation: the per-cell `baseline` plus the shift `c`
/// and shape. `delta` is **not** stored — it is the lazy `NormalizeTotal→Log1p`
/// source, reproduced on demand — keeping this struct `O(n_obs)`.
#[derive(Debug, Clone)]
pub struct PFlog1pPF {
    /// Per-cell baseline `b_i = −(1/D) Σ_j delta_ij` (length `n_obs`).
    pub baseline: Vec<f64>,
    /// Positive shift / pseudocount.
    pub c: f64,
    /// Number of cells (rows).
    pub n_obs: usize,
    /// Number of features (columns).
    pub n_vars: usize,
}

impl PFlog1pPF {
    /// Construct from a precomputed `baseline`, validating its invariants:
    /// `baseline.len() == n_obs`, `c` positive and finite, and `n_vars > 0`
    /// (the centering denominator). Returns [`AccelError::ShapeError`] /
    /// [`AccelError::InvalidInput`] on violation rather than constructing an
    /// inconsistent value.
    pub fn new(baseline: Vec<f64>, c: f64, n_obs: usize, n_vars: usize) -> Result<Self> {
        if c <= 0.0 || c.is_nan() || c.is_infinite() {
            return Err(AccelError::InvalidInput(format!(
                "PFlog1pPF shift c must be positive and finite, got {c}"
            )));
        }
        if n_vars == 0 {
            return Err(AccelError::InvalidInput(
                "PFlog1pPF requires n_vars > 0 for centering".into(),
            ));
        }
        if baseline.len() != n_obs {
            return Err(AccelError::ShapeError(format!(
                "PFlog1pPF baseline has length {} but n_obs={n_obs}",
                baseline.len()
            )));
        }
        Ok(Self {
            baseline,
            c,
            n_obs,
            n_vars,
        })
    }
}

/// Per-cell raw count depth `s_i = Σ_j x_ij`, streamed shard-by-shard.
///
/// `source` must carry **raw counts** (this is the depth used to form
/// within-cell proportions). Returns a length-`n_obs` vector.
pub fn pflog1ppf_cell_depths<S: ShardSource>(source: &S) -> Result<Vec<f64>> {
    let n_obs = source.n_obs();
    let mut depths = vec![0.0f64; n_obs];
    let mut row_base = 0usize;
    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        ensure_finite(&csr.data)?;
        let rows = csr.n_rows();
        for (r, s) in csr.row_sums().into_iter().enumerate() {
            depths[row_base + r] = s;
        }
        row_base += rows;
    }
    if row_base != n_obs {
        return Err(AccelError::ShapeError(format!(
            "PFlog1pPF depth pass streamed {row_base} rows but source reports n_obs={n_obs}"
        )));
    }
    Ok(depths)
}

/// Per-cell baseline `b_i = −(1/D) Σ_j delta_ij`, streamed shard-by-shard.
///
/// `source` yields **raw-count** CSR shards; `cell_depths` are the row sums
/// `s_i` (see [`pflog1ppf_cell_depths`]). For each stored nonzero this
/// accumulates `log1p(value / (c · s_i))` in `f64`; original zeros contribute
/// nothing (`delta = 0`). `O(M)` time, `O(n_obs)` memory.
///
/// # Raw-count guard
///
/// This kernel takes raw X by contract — running it on an already-normalized
/// matrix produces nonsense. It rejects `c ≤ 0`, any non-positive depth,
/// negative counts, and non-finite values rather than silently proceeding.
pub fn pflog1ppf_baseline<S: ShardSource>(
    source: &S,
    cell_depths: &[f64],
    c: f64,
) -> Result<Vec<f64>> {
    if c <= 0.0 || c.is_nan() || c.is_infinite() {
        return Err(AccelError::InvalidInput(format!(
            "PFlog1pPF shift c must be positive and finite, got {c}"
        )));
    }
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();
    if n_vars == 0 {
        return Err(AccelError::InvalidInput(
            "PFlog1pPF requires n_vars > 0 for centering".into(),
        ));
    }
    if cell_depths.len() != n_obs {
        return Err(AccelError::ShapeError(format!(
            "cell_depths has length {} but source reports n_obs={n_obs}",
            cell_depths.len()
        )));
    }

    let mut row_sum_delta = vec![0.0f64; n_obs];
    let mut row_base = 0usize;
    for shard_idx in 0..source.n_shards() {
        let csr = source.read_shard(shard_idx)?;
        ensure_finite(&csr.data)?;
        let rows = csr.n_rows();
        if row_base + rows > n_obs {
            return Err(AccelError::ShapeError(format!(
                "PFlog1pPF: shard {shard_idx} has {rows} rows, exceeding n_obs={n_obs} \
                 at row_base={row_base}"
            )));
        }
        for r in 0..rows {
            let cell = row_base + r;
            let depth = cell_depths[cell];
            if depth <= 0.0 || depth.is_nan() {
                return Err(AccelError::InvalidInput(format!(
                    "PFlog1pPF: cell {cell} has non-positive depth {depth}; \
                     filter empty cells before normalizing"
                )));
            }
            let start = csr.indptr[r] as usize;
            let end = csr.indptr[r + 1] as usize;
            let mut acc = 0.0f64;
            for nz in start..end {
                let v = csr.data[nz] as f64;
                if v < 0.0 {
                    return Err(AccelError::InvalidInput(format!(
                        "PFlog1pPF: negative count {v} at cell {cell}; counts must be non-negative"
                    )));
                }
                acc += (v / (c * depth)).ln_1p();
            }
            row_sum_delta[cell] += acc;
        }
        row_base += rows;
    }
    if row_base != n_obs {
        return Err(AccelError::ShapeError(format!(
            "PFlog1pPF baseline pass streamed {row_base} rows but source reports n_obs={n_obs}"
        )));
    }

    let inv_d = 1.0 / n_vars as f64;
    Ok(row_sum_delta.into_iter().map(|s| -s * inv_d).collect())
}

/// Per-cell baseline `b_i = −(1/D) · Σ_j delta_ij` from an already-built
/// `delta` source (the lazy `NormalizeTotal{1/c}→Log1p` chain).
///
/// Equivalent to [`pflog1ppf_baseline`] but reads pre-transformed `delta`
/// values directly (`b_i = −rowsum(delta_i)/D`), so the result aligns to the
/// source's **visible** row order — the natural form for the pyscx binding,
/// where the lazy source already handles column projection and deletion
/// filtering. Use [`pflog1ppf_baseline`] (raw counts + depths) when you need
/// the raw-count validation guard.
pub fn pflog1ppf_baseline_from_delta<S: ShardSource>(delta_source: &S) -> Result<Vec<f64>> {
    let n_obs = delta_source.n_obs();
    let n_vars = delta_source.n_vars();
    if n_vars == 0 {
        return Err(AccelError::InvalidInput(
            "PFlog1pPF requires n_vars > 0 for centering".into(),
        ));
    }
    let inv_d = 1.0 / n_vars as f64;
    let mut baseline = vec![0.0f64; n_obs];
    let mut row_base = 0usize;
    for shard_idx in 0..delta_source.n_shards() {
        let csr = delta_source.read_shard(shard_idx)?;
        ensure_finite(&csr.data)?;
        let rows = csr.n_rows();
        for (r, s) in csr.row_sums().into_iter().enumerate() {
            baseline[row_base + r] = -s * inv_d;
        }
        row_base += rows;
    }
    if row_base != n_obs {
        return Err(AccelError::ShapeError(format!(
            "PFlog1pPF baseline pass streamed {row_base} rows but source reports n_obs={n_obs}"
        )));
    }
    Ok(baseline)
}

#[cfg(test)]
#[path = "pflog1ppf_tests.rs"]
mod tests;
