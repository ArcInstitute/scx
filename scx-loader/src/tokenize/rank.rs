//! Rank tokens — per-cell gene order by normalised expression.
//!
//! Consumers: Geneformer, C2S, TranscriptFormer-class. The cell becomes a
//! sequence of gene ids ordered by how enriched each gene is *in this cell
//! relative to the corpus*, truncated to the model's context length. Special
//! tokens (`<cls>`, `<eos>`) and the vocabulary mapping are the consumer's; this
//! kernel emits global gene ids in rank order and their count.
//!
//! # Semantics
//!
//! Per Geneformer's `tokenize_cell` / `rank_genes`, at a pinned revision:
//!
//! 1. divide the row by the cell's library size and multiply by `target_sum`
//!    (Geneformer uses 10,000);
//! 2. divide each gene by its corpus statistic — the non-zero median expression
//!    across Genecorpus, supplied here as [`PerGeneNorm`];
//! 3. drop non-positive values;
//! 4. order descending, gene id ascending on ties;
//! 5. truncate to `l_max`.
//!
//! # ⚠️ The reference has no tie rule, so this kernel declares one
//!
//! Geneformer ranks with `gene_tokens[np.argsort(-gene_vector)]`, and
//! `np.argsort`'s default `kind` is quicksort — **not stable**. Measured on
//! numpy 2.4.4, `np.argsort(-v)` on a vector with a run of equal values returns
//! `[0 5 2 1 3 4 ...]` where `kind="stable"` returns `[0 5 1 2 3 4 ...]`: the
//! order within an equal-value run is an artefact of introsort's partitioning,
//! reproducible for a given input but derived from no rule.
//!
//! Ties are not rare there. Every gene whose corpus median is 1.0 and whose count
//! in this cell is 1 normalises to the same value, and single-count genes are the
//! bulk of a droplet cell.
//!
//! So this kernel sorts ties by **gene id ascending**, which is the same rule the
//! crop uses, and the divergence is declared rather than hidden: a golden taken
//! from the reference agrees with this kernel exactly on tie-free cases, and
//! agrees up to the order within equal-value runs otherwise. A consumer that
//! needs Geneformer's exact permutation must use Geneformer.
//!
//! # Identity is an input, never inferred
//!
//! [`PerGeneNorm`] carries the statistics vector, a caller-declared vocabulary
//! version, and a 64-bit identity over both. "Rank order under statistics S at
//! vocabulary V" is only a well-defined artifact if S and V are named, and a
//! kernel that derived S from the data it is ranking would make every run's
//! tokens depend on that run's cells.

use std::sync::Arc;

use crate::error::{LoaderError, Result};
use crate::tokenize::{transform, CsrRow};

/// Per-gene normalisation statistics plus the identity that names them.
///
/// `stat` is indexed by **global gene id**, so its length is the vocabulary size
/// and a row's ids index into it directly.
#[derive(Clone, Debug)]
pub struct PerGeneNorm {
    stat: Arc<[f32]>,
    vocabulary_version: String,
    identity: u64,
}

impl PerGeneNorm {
    /// Validate and stamp a statistics vector.
    ///
    /// Every entry must be finite and strictly positive: it is a divisor, and a
    /// zero or NaN would turn one gene into `inf`/`NaN` and silently sort it to
    /// the front of every cell. Geneformer's median file is positive by
    /// construction (it stores *non-zero* medians), so a violation means the
    /// vector is not the one the model expects.
    pub fn new(stat: Arc<[f32]>, vocabulary_version: impl Into<String>) -> Result<Self> {
        let vocabulary_version = vocabulary_version.into();
        if stat.is_empty() {
            return Err(LoaderError::ConfigError {
                reason: "PerGeneNorm: statistics vector is empty".to_string(),
            });
        }
        if let Some((i, v)) = stat
            .iter()
            .enumerate()
            .find(|(_, v)| !v.is_finite() || **v <= 0.0)
        {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "PerGeneNorm: statistic at gene {i} is {v}; every entry must be finite and \
                     strictly positive (it is a divisor)"
                ),
            });
        }
        // blake3 truncated-64 over the raw statistic bytes plus the version
        // string, the way `downsample::file_identity` stamps a path. The version
        // is folded in so two corpora that happen to share a median vector but
        // not a vocabulary do not collide.
        let mut hasher = blake3::Hasher::new();
        for v in stat.iter() {
            hasher.update(&v.to_le_bytes());
        }
        hasher.update(vocabulary_version.as_bytes());
        let identity = u64::from_le_bytes(
            hasher.finalize().as_bytes()[..8]
                .try_into()
                .expect("blake3 >= 8 bytes"),
        );
        Ok(Self {
            stat,
            vocabulary_version,
            identity,
        })
    }

    /// 64-bit identity over the statistics vector and the vocabulary version.
    ///
    /// Echo this into whatever records a run's token provenance: it is half of
    /// what makes a rank order reproducible, the other half being the row's
    /// content.
    #[inline]
    pub fn identity(&self) -> u64 {
        self.identity
    }

    #[inline]
    pub fn vocabulary_version(&self) -> &str {
        &self.vocabulary_version
    }

    /// Vocabulary size — the length of the statistics vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.stat.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.stat.is_empty()
    }
}

/// Rank one row's genes, writing up to `out.len()` ids and returning how many.
///
/// `order` is caller-owned scratch reused across rows. `out` is written from
/// index 0; entries past the returned length are **left untouched**, so a caller
/// reusing a buffer must read only the prefix — there is no PAD fill, because
/// unlike the crop this kernel reports a length and the consumer owns the
/// padding token.
///
/// An empty row, an all-zero row, or a row whose library size is zero yields
/// length 0 rather than an error: those are ordinary cells, and the reference
/// would divide by zero on them.
pub fn rank_tokens(
    row: CsrRow,
    norm: &PerGeneNorm,
    target_sum: f64,
    order: &mut Vec<usize>,
    values: &mut Vec<f32>,
    out: &mut [i64],
) -> Result<usize> {
    if row.gene_ids.len() != row.values.len() {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "rank_tokens: gene_ids len {} != values len {}",
                row.gene_ids.len(),
                row.values.len()
            ),
        });
    }
    // EVERY id, not just the last one. An earlier version checked
    // `gene_ids.last()` on the reasoning that the row is sorted ascending, so
    // the last id is the maximum — true, and not sufficient. `[-1, 2]` has a
    // last id in range and `-1i32 as usize` wraps to `usize::MAX`; an unsorted
    // `[7, 1]` has a last id in range and `stat[7]` is out of bounds. Both are
    // ordinary numpy CSR arriving at a `pub` entry, and both reached an
    // index-out-of-bounds panic across the FFI.
    //
    // Before the zero-library early return, not after: a row of all-zero counts
    // with a malformed id would otherwise report `Ok(0)` and never look at the
    // ids at all. `bin_values` validates before its own early return for the
    // same reason; the two were inconsistent.
    for &g in row.gene_ids {
        if g < 0 || g as usize >= norm.stat.len() {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "rank_tokens: gene id {g} is outside the normalisation vocabulary of size {}",
                    norm.stat.len()
                ),
            });
        }
    }

    let lib = transform::library_size(row.values);
    if lib <= 0.0 {
        return Ok(0);
    }
    let factor = target_sum / lib;

    values.clear();
    values.reserve(row.len());
    for (&g, &v) in row.gene_ids.iter().zip(row.values) {
        let normalised = (v.max(0.0) as f64) * factor / (norm.stat[g as usize] as f64);
        values.push(normalised as f32);
    }

    order.clear();
    order.extend((0..row.len()).filter(|&i| values[i] > 0.0));
    let ids = row.gene_ids;
    order.sort_by(|&a, &b| {
        // Finite by construction: `max(0.0)` removes NaN, `factor` is finite
        // because `lib > 0`, and every statistic was validated positive.
        values[b]
            .partial_cmp(&values[a])
            .unwrap()
            .then(ids[a].cmp(&ids[b]))
    });

    let take = out.len().min(order.len());
    for (slot, &i) in order.iter().take(take).enumerate() {
        out[slot] = ids[i] as i64;
    }
    Ok(take)
}

#[cfg(test)]
#[path = "rank_tests.rs"]
mod tests;
