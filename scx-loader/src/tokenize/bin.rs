//! Value binning — per-cell quantile or fixed-edge bins over expressed values.
//!
//! Consumers: scGPT and its lineage (Tahoe-x1), CellFM-class. The cell's values
//! become small integers so the model can embed them as tokens.
//!
//! # Semantics, from scGPT's `binning` at `cebd6fae`
//!
//! Zeros stay in bin 0. The non-zero values get quantile edges
//! `np.quantile(non_zero, np.linspace(0, 1, n_bins - 1))` — `n_bins - 1` edges
//! spanning the observed range, so `edges[0]` is the minimum and `edges[-1]` the
//! maximum — and are then digitized against them into `[1, n_bins - 1]`.
//!
//! Edges are computed in `f64` with numpy's default `linear` interpolation.
//! That is not a choice: `np.quantile` on a `float32` array **returns float64**
//! (checked on numpy 2.4.4), and the subsequent `np.digitize` therefore widens
//! each value to `f64` to compare. Computing the edges in `f32` gives different
//! answers — `[0.1, 0.16, 0.22, 0.28, 0.46, 0.7]` where numpy gives
//! `[0.1, 0.16, 0.22, 0.28000001, 0.46, 0.69999999]`.
//!
//! # ⚠️ The reference is stochastic at bin edges, and cannot be matched
//!
//! scGPT's `binning` calls `_digitize(x, bins)`, whose default is `side="both"`:
//!
//! ```text
//! left_digits  = np.digitize(x, bins)
//! right_digits = np.digitize(x, bins, right=True)
//! rands        = np.random.rand(len(x))
//! digits       = np.ceil(rands * (right_digits - left_digits) + left_digits)
//! ```
//!
//! A value that sits exactly on an edge lands anywhere between the two bounds,
//! chosen from numpy's **global** RNG. There is no seed to pass and no stream a
//! `ChaCha8Rng` can reproduce, so no golden taken from the reference can be
//! matched exactly.
//!
//! What this kernel offers instead is the pair of deterministic bounds the
//! reference interpolates between ([`BinTie::Left`], [`BinTie::Right`]) plus a
//! seeded equivalent of the randomisation ([`BinTie::SeededUniform`]) keyed on
//! content identity. The strongest available claim about the reference is then a
//! **bracketing** one — every draw the reference can produce lies in
//! `[Right, Left]` — and that is what the tests assert.
//!
//! The degenerate case is worth naming because it is not rare. A row whose
//! non-zero values are all equal collapses every quantile edge onto that value,
//! giving `left = n_bins - 1` and `right = 0`; the reference then assigns a
//! **uniformly random bin over the whole range** (measured: `left=[50]`,
//! `right=[0]` at `n_bins=51`). [`BinTie::Left`] pins it at the top bin, which is
//! at least a rule.
//!
//! # Negatives and NaN
//!
//! Values are clipped with `max(0.0)` first, as everywhere else in this module
//! and as the gather stage does — so a negative and a NaN both become zeros and
//! land in bin 0. scGPT's `nonzero()` would instead have binned a negative. On
//! count and log-count data, where both references operate, the two agree.

use crate::error::{LoaderError, Result};
use crate::seed::{row_seed, TOKENIZE_BIN_TAG};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Where the bin edges come from.
#[derive(Clone, Copy, Debug)]
pub enum BinEdges<'a> {
    /// Caller-supplied edges, non-decreasing. Use this when the model pins bin
    /// edges corpus-wide rather than recomputing them per cell — then the edges
    /// are part of the tokeniser's identity and must be recorded with it.
    Fixed(&'a [f64]),
    /// `np.quantile(non_zero, np.linspace(0, 1, n_bins - 1))`, recomputed for
    /// every row. This is what scGPT's `Preprocessor` does, and it means a
    /// cell's bin for a given value depends on the rest of that cell.
    PerCellQuantile,
}

/// How a value sitting exactly on an edge is placed.
#[derive(Clone, Copy, Debug)]
pub enum BinTie {
    /// `np.digitize(x, bins)` — the upper of the two bounds.
    Left,
    /// `np.digitize(x, bins, right=True)` — the lower of the two bounds.
    Right,
    /// The reference's randomised interpolation, but seeded on content identity
    /// rather than on numpy's global RNG: `ceil(u * (right - left) + left)` with
    /// `u` drawn from `ChaCha8Rng` keyed by `(seed, file, row)`.
    ///
    /// This is a **different stream** from the reference's. It reproduces the
    /// reference's *distribution*, not its draws, and it is reproducible across
    /// runs and machines, which numpy's global RNG is not.
    SeededUniform {
        seed: u64,
        file_identity: u64,
        row: u64,
    },
}

/// Bin one row's values into `out`, which must be the same length as `values`.
///
/// `buf` is caller-owned scratch reused across rows; on the
/// [`BinEdges::PerCellQuantile`] path it holds the row's edges.
pub fn bin_values(
    values: &[f32],
    edges: BinEdges<'_>,
    n_bins: usize,
    tie: BinTie,
    buf: &mut Vec<f64>,
    out: &mut [i64],
) -> Result<()> {
    if out.len() != values.len() {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "bin_values: out len {} != values len {}",
                out.len(),
                values.len()
            ),
        });
    }
    if n_bins < 3 {
        // `n_bins - 1` edges spanning `linspace(0, 1, n_bins - 1)` needs at
        // least two edge positions to be a range rather than a point.
        return Err(LoaderError::ConfigError {
            reason: format!("bin_values: n_bins must be >= 3, got {n_bins}"),
        });
    }

    for o in out.iter_mut() {
        *o = 0;
    }

    // Clip first, then "non-zero" means "positive" — see the module docs.
    buf.clear();
    buf.extend(values.iter().filter_map(|&v| {
        let c = v.max(0.0);
        (c > 0.0).then_some(c as f64)
    }));
    if buf.is_empty() {
        // scGPT's `row.max() == 0` early return: an all-zero row is all bin 0.
        return Ok(());
    }

    // On the quantile path the edges are APPENDED to `buf` behind the sorted
    // values rather than collected into a fresh `Vec`: this runs once per row,
    // and a 50-element allocation per cell is exactly the cost the caller-owned
    // scratch exists to remove.
    let edges_at = match edges {
        BinEdges::Fixed(e) => {
            if e.is_empty() {
                return Err(LoaderError::ConfigError {
                    reason: "bin_values: Fixed edges are empty".to_string(),
                });
            }
            if e.windows(2).any(|w| w[1] < w[0]) || e.iter().any(|v| !v.is_finite()) {
                return Err(LoaderError::ConfigError {
                    reason: "bin_values: Fixed edges must be finite and non-decreasing".to_string(),
                });
            }
            None
        }
        BinEdges::PerCellQuantile => {
            buf.sort_by(|a, b| a.partial_cmp(b).expect("clipped values are finite"));
            let m = buf.len();
            push_quantile_edges(buf, m, n_bins - 1);
            Some(m)
        }
    };
    let edges: &[f64] = match (edges_at, edges) {
        (Some(m), _) => &buf[m..],
        (None, BinEdges::Fixed(e)) => e,
        (None, BinEdges::PerCellQuantile) => unreachable!("quantile path always appends its edges"),
    };

    let mut rng = match tie {
        BinTie::SeededUniform {
            seed,
            file_identity,
            row,
        } => Some(ChaCha8Rng::seed_from_u64(row_seed(
            seed,
            TOKENIZE_BIN_TAG,
            file_identity,
            row,
        ))),
        _ => None,
    };

    for (o, &v) in out.iter_mut().zip(values) {
        let c = v.max(0.0);
        if c <= 0.0 {
            continue;
        }
        let x = c as f64;
        let left = digitize_left(edges, x);
        *o = match tie {
            BinTie::Left => left,
            BinTie::Right => digitize_right(edges, x),
            BinTie::SeededUniform { .. } => {
                let right = digitize_right(edges, x);
                let u: f64 = rng.as_mut().expect("seeded tie has an rng").gen();
                (u * (right - left) as f64 + left as f64).ceil() as i64
            }
        };
    }
    Ok(())
}

/// Append `np.quantile(buf[..m], np.linspace(0, 1, n))` — numpy's `linear`
/// method — to `buf`, behind the `m` ascending values it reads.
///
/// `buf[..m]` must already be ascending. `n == 1` gives the single point
/// `q = 0`, matching `np.linspace(0, 1, 1) == [0.0]`.
///
/// Appending rather than returning a `Vec` is what keeps the kernel free of
/// per-row allocations: at `n_bins = 51` a returned vector is a 50-element
/// allocation per cell. The two reads are copied into locals before the push,
/// so a reallocation mid-loop is harmless; each iteration re-indexes.
fn push_quantile_edges(buf: &mut Vec<f64>, m: usize, n: usize) {
    for j in 0..n {
        let q = if n <= 1 {
            0.0
        } else {
            j as f64 / (n - 1) as f64
        };
        let h = (m - 1) as f64 * q;
        let lo = h.floor() as usize;
        let hi = h.ceil() as usize;
        let (a, b) = (buf[lo], buf[hi]);
        buf.push(a + (h - lo as f64) * (b - a));
    }
}

/// `np.digitize(x, bins)` for increasing `bins` = `searchsorted(bins, x, "right")`.
#[inline]
fn digitize_left(edges: &[f64], x: f64) -> i64 {
    edges.partition_point(|&e| e <= x) as i64
}

/// `np.digitize(x, bins, right=True)` = `searchsorted(bins, x, "left")`.
#[inline]
fn digitize_right(edges: &[f64], x: f64) -> i64 {
    edges.partition_point(|&e| e < x) as i64
}

#[cfg(test)]
#[path = "bin_tests.rs"]
mod tests;
