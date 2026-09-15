//! Expression-weighted gene sampling.
//!
//! Consumers: UCE-class models, which build a "cell sentence" by drawing genes
//! with probability proportional to a transform of their expression. Assembly of
//! the sentence — chromosome grouping, CLS/PAD tokens, sorting within a
//! chromosome by genomic start — is the consumer's; this kernel draws the genes.
//!
//! # ⚠️ Three corrections to how this is usually described
//!
//! 1. **The draw is with replacement.** UCE's `sample_cell_sentences`
//!    (`eval_data.py`) calls
//!    `np.random.choice(np.arange(len(weights)), size=args.sample_size, p=weights, replace=True)`.
//!    A highly expressed gene appears several times in one sentence, by design.
//! 2. **The weights are `log1p` of the counts, renormalised** — not the counts
//!    themselves. `torch.log1p(counts)` then `weights / weights.sum()`.
//! 3. **No golden can be taken from the reference.** `np.random.choice` draws
//!    from numpy's global RNG. This kernel uses `ChaCha8Rng` keyed on content
//!    identity, so it reproduces the reference's *algorithm and distribution*,
//!    never its draws.
//!
//! # What is reproduced exactly
//!
//! numpy's `choice` with `p` and `replace=True` is inverse-CDF sampling:
//!
//! ```text
//! cdf = p.cumsum(); cdf /= cdf[-1]
//! idx = cdf.searchsorted(random_sample(size), side="right")
//! ```
//!
//! This kernel does the same three steps, so the algorithm and the distribution
//! match, which is why the parity test is distributional (expected counts at
//! large N) rather than element-wise.
//!
//! ⚠️ It does **not** follow that the same uniforms select the same genes. UCE
//! normalises its weights in **float32** (torch's `log1p` on an int64 tensor
//! returns float32, and numpy widens `p` only for the cumsum) while this kernel
//! accumulates in `f64`, so a uniform landing on a CDF boundary can select an
//! adjacent gene. An earlier version of this comment claimed the stronger
//! property; `docs/tokenize.md` carries the measurement.
//!
//! # Seeding
//!
//! Keyed on `(seed, file identity, row)` through the same four-component
//! derivation the count downsampler uses (`crate::seed::row_seed`), under its own
//! domain tag so the two draw independent streams for the same cell. Keying on
//! **content identity** rather than on epoch or worker index is §3's rule: a
//! resumed run, or one with a different worker count, must not silently switch
//! streams.

use crate::error::{LoaderError, Result};
use crate::seed::{row_seed, TOKENIZE_SAMPLE_TAG};
use crate::tokenize::CsrRow;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// What the sampling weight is a function of.
///
/// Part of the tokeniser's identity: two runs that differ only here draw
/// different sentences from the same cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightTransform {
    /// `log1p(count)`, then renormalised — UCE's choice. Compresses the dynamic
    /// range, so a gene with 1000 counts is ~3x as likely as one with 10, not
    /// 100x.
    Log1p,
    /// The count itself, renormalised — "probability proportional to expression"
    /// read literally.
    Linear,
}

impl WeightTransform {
    #[inline]
    fn apply(self, v: f32) -> f64 {
        let c = v.max(0.0) as f64;
        match self {
            Self::Log1p => c.ln_1p(),
            Self::Linear => c,
        }
    }
}

/// Draw `out.len()` genes from `row`, with replacement, and return how many were
/// written.
///
/// Returns `0` — leaving `out` untouched — when the row is empty or every weight
/// is zero. An all-zero row has no distribution to sample from, and the
/// reference would raise on the resulting NaN probabilities; a cell with no
/// counts is an ordinary cell, so this reports a length instead.
///
/// `buf` is caller-owned scratch reused across rows; it holds the CDF.
pub fn sample_genes(
    row: CsrRow,
    weight: WeightTransform,
    seed: u64,
    file_identity: u64,
    row_index: u64,
    buf: &mut Vec<f64>,
    out: &mut [i64],
) -> Result<usize> {
    if row.gene_ids.len() != row.values.len() {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "sample_genes: gene_ids len {} != values len {}",
                row.gene_ids.len(),
                row.values.len()
            ),
        });
    }
    if row.is_empty() || out.is_empty() {
        return Ok(0);
    }

    // cumsum of the weights, then normalise by the last entry — numpy's own two
    // steps, in that order. Normalising each weight first and then accumulating
    // would give a different final entry and a CDF that need not reach 1.0.
    buf.clear();
    let mut acc = 0.0f64;
    for &v in row.values {
        acc += weight.apply(v);
        buf.push(acc);
    }
    let total = *buf.last().expect("row is non-empty");
    if total <= 0.0 || !total.is_finite() {
        return Ok(0);
    }
    for c in buf.iter_mut() {
        *c /= total;
    }

    let mut rng = ChaCha8Rng::seed_from_u64(row_seed(
        seed,
        TOKENIZE_SAMPLE_TAG,
        file_identity,
        row_index,
    ));
    let last = buf.len() - 1;
    for slot in out.iter_mut() {
        let u: f64 = rng.gen();
        // `searchsorted(cdf, u, side="right")`. The clamp cannot fire for a
        // uniform in [0, 1) against a CDF whose last entry is exactly 1.0, and
        // is kept because a rounding-induced out-of-range index here would be an
        // out-of-bounds read rather than a wrong gene.
        let i = buf.partition_point(|&c| c <= u).min(last);
        *slot = row.gene_ids[i] as i64;
    }
    Ok(out.len())
}

#[cfg(test)]
#[path = "sample_tests.rs"]
mod tests;
