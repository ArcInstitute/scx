//! Row filtering within a single decoded shard.

use crate::error::{EngineError, Result};

/// Extract only the rows where `keep_mask[row]` is true from decoded CSR data.
///
/// Returns new `(indptr, indices, data)` arrays for the filtered subset.
/// Operates on pre-decoded scipy-compatible types (i64/i32/f32).
pub fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    keep_mask: &[bool],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let n_rows = indptr.len().saturating_sub(1);
    check_keep_mask_len(keep_mask.len(), n_rows)?;
    let mask_len = n_rows;

    let mut new_indptr = Vec::with_capacity(mask_len + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

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
