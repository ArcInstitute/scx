//! Native (non-`f32`) per-shard decode for the typed collect.
//!
//! The default collect decodes each shard to scipy types — `f32` values — and a
//! caller who later asks for `data_dtype="float64"` gets those rounded values
//! widened, which is exactly the loss the fail-loud guard exists to prevent. The
//! typed collect instead decodes to the shard's **native** stream (integer
//! encodings stay `u32`, float encodings stay `f32`) and narrows once, into the
//! caller's dtype. A count above 2²⁴ survives that route and does not survive
//! the other.
//!
//! Row filtering and gene projection are **fused** into one pass here, unlike the
//! `f32` path where [`decode_shard_projected`](crate::projection::decode_shard_projected)
//! builds a whole projected shard that [`filter_csr_rows`](super::rows::filter_csr_rows)
//! then re-allocates a subset of. Same output, one allocation and one pass over
//! the kept rows instead of two over all of them. The hazardous part — the
//! monotonic merge scan — is not duplicated: it stays in
//! [`project_csr_row`](crate::projection::project_csr_row), which is generic over
//! the index and value element types so both decode shapes share it.

use scx_codec::ShardValuesNative;
use scx_format_io::catalog::FullCatalogEntry;

use super::rows::check_keep_mask_len;
use crate::error::Result;
use crate::projection::project_csr_row_into;
use crate::reader::SectionReader;

/// One shard's kept rows, decoded to native types.
pub(crate) struct NativeShardRows {
    pub(crate) indptr: Vec<i64>,
    pub(crate) indices: Vec<u32>,
    pub(crate) values: ShardValuesNative,
}

impl NativeShardRows {
    /// Rows retained by the keep mask.
    pub(crate) fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Number of stored non-zeros.
    pub(crate) fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Keep only the first `rows` rows.
    ///
    /// How the typed collect applies `limit`: the f32 path merges the whole
    /// decoded prefix and then `row_slice`s the assembled matrix, but the typed
    /// merge sizes its buffers from the per-shard totals, so trimming here
    /// yields the same first-`limit` rows without allocating the rows it is
    /// about to drop. Shard results are in ascending global row order, so
    /// truncating the concatenation and truncating each shard in turn agree.
    pub(crate) fn truncate_rows(&mut self, rows: usize) {
        if rows >= self.n_rows() {
            return;
        }
        self.indptr.truncate(rows + 1);
        let nnz = *self.indptr.last().unwrap_or(&0) as usize;
        self.indices.truncate(nnz);
        match &mut self.values {
            ShardValuesNative::U32(v) => v.truncate(nnz),
            ShardValuesNative::F32(v) => v.truncate(nnz),
        }
    }
}

/// Trim a decoded prefix to `limit` rows in global order, dropping whole shards
/// past the cutoff.
pub(crate) fn truncate_to_limit(results: &mut Vec<NativeShardRows>, limit: usize) {
    let mut budget = limit;
    let mut keep_shards = 0usize;
    for r in results.iter_mut() {
        if budget == 0 {
            break;
        }
        r.truncate_rows(budget);
        budget -= r.n_rows();
        keep_shards += 1;
    }
    results.truncate(keep_shards);
}

/// Decode one CSR shard to native types, keeping only the rows `keep_mask`
/// selects and — when `gene_set` is `Some` — only the requested columns,
/// remapped to `0..gene_set.len()`.
///
/// `gene_set` must be ascending and unique (`PlanAndMask::effective_gene_indices`
/// already is; the merge scan silently drops columns otherwise, which
/// `project_csr_row`'s own `debug_assert!` documents).
pub(crate) fn decode_shard_native_filtered(
    reader: &dyn SectionReader,
    entry: &FullCatalogEntry,
    keep_mask: &[bool],
    gene_set: Option<&[u32]>,
) -> Result<NativeShardRows> {
    let (indptr, indices, values) = reader.read_shard_from_entry_native(entry)?;
    // Shared with the f32 row filter, so the two cannot end up disagreeing about
    // what a valid mask is — and so a truncated shard is rejected here too
    // rather than silently yielding fewer rows than the obs half will carry.
    check_keep_mask_len(keep_mask.len(), indptr.len().saturating_sub(1))?;

    Ok(match values {
        ShardValuesNative::U32(v) => {
            let (ip, ix, out) = filter_project_rows(&indptr, &indices, &v, keep_mask, gene_set);
            NativeShardRows {
                indptr: ip,
                indices: ix,
                values: ShardValuesNative::U32(out),
            }
        }
        ShardValuesNative::F32(v) => {
            let (ip, ix, out) = filter_project_rows(&indptr, &indices, &v, keep_mask, gene_set);
            NativeShardRows {
                indptr: ip,
                indices: ix,
                values: ShardValuesNative::F32(out),
            }
        }
    })
}

/// Keep `keep_mask` rows and (optionally) project their columns, in one pass.
///
/// Generic over the value element rather than written once per
/// [`ShardValuesNative`] arm: a shard is uniformly integer- or float-encoded, but
/// a *file* may hold both, so the arm is chosen per shard and the body must not
/// be.
fn filter_project_rows<V: Copy>(
    indptr: &[i64],
    indices: &[u32],
    values: &[V],
    keep_mask: &[bool],
    gene_set: Option<&[u32]>,
) -> (Vec<i64>, Vec<u32>, Vec<V>) {
    let n_rows = indptr.len().saturating_sub(1);
    let kept = keep_mask.iter().filter(|&&k| k).count();

    let mut new_indptr = Vec::with_capacity(kept + 1);
    new_indptr.push(0i64);
    // Exact when there is no projection: sum the kept rows' lengths, which the
    // indptr already gives without touching the values. Under a projection it is
    // an **upper** bound (the merge scan can only drop entries), so it is still
    // a safe reservation — the first version reserved zero for every filtered
    // query and mis-described the unprojected count as a lower bound.
    let kept_nnz: usize = (0..n_rows)
        .filter(|&row| keep_mask[row])
        .map(|row| (indptr[row + 1] - indptr[row]) as usize)
        .sum();
    let mut new_indices = Vec::with_capacity(kept_nnz);
    let mut new_values = Vec::with_capacity(kept_nnz);

    for row in 0..n_rows {
        if !keep_mask[row] {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;

        match gene_set {
            Some(gs) => {
                // Appends into the shard's buffers: the returning form allocated
                // and freed two `Vec`s per kept row.
                project_csr_row_into(
                    &mut new_indices,
                    &mut new_values,
                    &indices[start..end],
                    &values[start..end],
                    gs,
                );
            }
            None => {
                new_indices.extend_from_slice(&indices[start..end]);
                new_values.extend_from_slice(&values[start..end]);
            }
        }
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(new_indices.len() as i64);
        debug_assert!(prev <= *new_indptr.last().unwrap());
    }

    (new_indptr, new_indices, new_values)
}

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;
