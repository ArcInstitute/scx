//! Row filtering within a single decoded shard.

use crate::error::{EngineError, Result};

/// Extract only the rows where `keep_mask[row]` is true from decoded CSR data.
///
/// Returns new `(indptr, indices, data)` arrays for the filtered subset.
/// Operates on pre-decoded scipy-compatible types (i64/i32/f32).
///
/// A caller that owns the decoded arrays should prefer
/// [`filter_csr_rows_owned`], which can hand an all-true mask's rows straight
/// back instead of copying them.
pub fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    keep_mask: &[bool],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let n_rows = indptr.len().saturating_sub(1);
    check_keep_mask_len(keep_mask.len(), n_rows)?;
    let mask_len = n_rows;

    // Sized from the mask, not from the shard: `mask_len + 1` reserved the
    // *unfiltered* row count, and the value buffers reserved nothing at all and
    // grew by doubling — ~8 B/nnz of copying plus reallocation churn on every
    // shard of every query. Both totals are exact and free: the kept row count
    // is a fold over the mask, and the kept nnz a fold over the indptr, neither
    // of which touches a value. (The typed kernel's identical fold in
    // `collect::native` is an *upper* bound rather than exact, because there the
    // projection is fused into the same pass and can only drop entries. Here the
    // projection has already happened.)
    let kept = keep_mask.iter().filter(|&&k| k).count();
    let kept_nnz: usize = (0..mask_len)
        .filter(|&row| keep_mask[row])
        .map(|row| (indptr[row + 1] - indptr[row]) as usize)
        .sum();

    let mut new_indptr = Vec::with_capacity(kept + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::with_capacity(kept_nnz);
    let mut new_data = Vec::with_capacity(kept_nnz);

    for row in 0..mask_len {
        if !keep_mask[row] {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        new_indices.extend_from_slice(&indices[start..end]);
        new_data.extend_from_slice(&data[start..end]);
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + (end - start) as i64);
    }

    Ok((new_indptr, new_indices, new_data))
}

/// [`filter_csr_rows`] for a caller that owns the decoded arrays: an all-true
/// mask returns them **by move**, with no copy at all.
///
/// The decoders guarantee `indptr[0] == 0` and `indptr.last() == indices.len()`
/// (`scx_codec::guards::check_decoded_shape` on the unframed arm, construction
/// on the framed one, and `decode_shard_projected`'s own `push(0)` /
/// `push(new_indices.len())`), so the rebase this function's slow path performs
/// is the identity when every row is kept — which is the case for an unfiltered
/// query and for any shard a row-set fully covers. Returning the input is
/// therefore bit-identical, not merely equivalent.
pub(crate) fn filter_csr_rows_owned(
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
    keep_mask: &[bool],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    // Length first, and not as a formality: `all()` over a mask *shorter* than
    // the CSR is vacuously true, so testing before the length check would return
    // every row of a shard the caller believes it truncated — the exact silent
    // truncation `check_keep_mask_len` exists to reject.
    check_keep_mask_len(keep_mask.len(), indptr.len().saturating_sub(1))?;
    if keep_mask.iter().all(|&k| k) {
        return Ok((indptr, indices, data));
    }
    filter_csr_rows(&indptr, &indices, &data, keep_mask)
}

/// Reject a keep mask that does not cover the decoded shard exactly.
///
/// A mismatch used to be a `debug_assert!` followed by
/// `min(keep_mask.len(), n_rows)`, which means release builds — the ones users
/// run — silently dropped every row past the shorter of the two and returned a
/// truncated query result. The assertion documented the invariant and then the
/// next line worked around it.
///
/// Same failure the reader-side deletion filters had: the mask and the CSR come
/// from independent places, and the direction that does *not* panic is the
/// dangerous one, because a wrong answer looks like an answer. Reject both
/// directions instead of clamping.
///
/// Shared with the native (typed-collect) kernel in `collect::native` so the two
/// row filters cannot end up disagreeing about what a valid mask is.
pub(crate) fn check_keep_mask_len(keep_len: usize, n_rows: usize) -> Result<()> {
    if keep_len != n_rows {
        return Err(EngineError::FormatError(
            scx_format_io::ScxError::InvalidCatalog(format!(
                "filter_csr_rows: keep mask covers {keep_len} rows but the decoded CSR has \
                 {n_rows} (truncated or corrupt shard)",
            )),
        ));
    }
    Ok(())
}
