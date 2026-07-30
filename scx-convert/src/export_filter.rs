//! Caller-supplied obs row filters for the SCX → h5ad/h5mu export path.
//!
//! The streaming exporter has always been able to drop observations — it just
//! had exactly one source for the decision, the on-disk deletion vector (see
//! [`crate::h5ad::stream_write::build_keep_mask`]). Everything downstream of
//! that already takes a plain `Option<&[bool]>`, so widening the *producer* is
//! all it takes to support an arbitrary caller filter.
//!
//! The one thing that must not drift is the coordinate system. Every mask here
//! is indexed in the **global / physical** obs row space — length equals the
//! file header's `n_obs`, counting logically deleted rows. That matches
//! [`ScxReader::deletion_keep_mask`] and [`BackedCsrReader::row_sums`], so the
//! masks compose with a plain elementwise AND.

use std::path::Path;

use scx_format_io::{BackedCsrReader, ScxReader};

use crate::pipeline::ConvertError;

/// Shard-cache depth for the row-sum pre-pass. The pass is a single ordered
/// sweep with no revisits, so a cache exists only to satisfy the reader's
/// constructor; 2 keeps the footprint at roughly one decoded shard.
const ROW_SUM_CACHE_SHARDS: usize = 2;

/// Streaming per-cell UMI totals → a global obs keep mask.
///
/// One ordered pass over the modality's CSR shards via
/// [`BackedCsrReader::row_sums`], which decodes at most the prefetch depth of
/// shards at a time — the matrix is never materialised. Row `i` is kept when
/// `row_sum[i] >= min_counts`; `>=` matches `pyscx.accel.filter_cells` and
/// `sc.pp.filter_cells`.
///
/// The returned vector is **global-length and not deletion-filtered**, so it
/// can be ANDed directly with the deletion-vector mask.
///
/// Note this sums `X`, not a layer and not `/raw`. On a file whose `X` has
/// already been normalised the threshold is meaningless — the intended input
/// is a raw count matrix.
pub fn min_counts_obs_mask(
    scx_path: &Path,
    modality_id: u8,
    min_counts: f64,
) -> Result<Vec<bool>, ConvertError> {
    if !min_counts.is_finite() || min_counts < 0.0 {
        return Err(ConvertError::Other(format!(
            "min_counts must be a finite non-negative number; got {min_counts}"
        )));
    }

    let reader = ScxReader::open(scx_path)?;
    let n_obs = reader.n_obs() as usize;

    let backed = if modality_id == 0 {
        BackedCsrReader::new(reader, ROW_SUM_CACHE_SHARDS)
    } else {
        BackedCsrReader::for_modality(reader, modality_id, ROW_SUM_CACHE_SHARDS)
    };

    let sums = backed.row_sums()?;

    // A shard-coverage gap would silently truncate the mask and drop the tail
    // of the obs axis, so treat a short vector as corruption rather than
    // padding it out.
    if sums.len() != n_obs {
        return Err(ConvertError::Other(format!(
            "min_counts pre-pass produced {} row sums but the file header \
             declares n_obs = {n_obs} (modality_id = {modality_id}); the CSR \
             shards do not cover the full obs axis",
            sums.len()
        )));
    }

    Ok(sums.into_iter().map(|s| s >= min_counts).collect())
}

#[cfg(test)]
#[path = "export_filter_tests.rs"]
mod tests;
