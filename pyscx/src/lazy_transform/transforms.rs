// Free transform-application helpers over CSR shards.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use scx_sparse::ScxCsr;

use super::*;

/// Apply all transforms in-place on a decoded CSR shard.
///
/// `global_row_offset` is the starting global row index for this shard,
/// used to look up per-row parameters (row_sums, factors).
///
/// Shared by both `ScxLazyTransformedDataset` and `LazyShardSource`.
pub(crate) fn apply_transforms_to_csr(
    transforms: &[Transform],
    csr: &mut ScxCsr,
    global_row_offset: usize,
) {
    // Detect fused NormalizeTotal + Log1p pattern for the first two transforms
    if transforms.len() >= 2 {
        if let (
            Transform::NormalizeTotal {
                row_sums,
                target_sum,
            },
            Transform::Log1p,
        ) = (&transforms[0], &transforms[1])
        {
            // Fused path: ln(x * target_sum / row_sum + 1) in one pass
            for row in 0..csr.n_rows() {
                let g = global_row_offset + row;
                let sum = row_sums[g];
                if sum > 0.0 {
                    let factor = *target_sum / sum;
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    for v in &mut csr.data[start..end] {
                        *v = ((*v as f64 * factor) as f32).ln_1p();
                    }
                } else {
                    // Non-positive row total: `NormalizeTotal` skips the row
                    // (scanpy's rule — an empty cell stays empty rather than
                    // being divided by zero), but `Log1p` still applies. This
                    // branch used to do neither, on the premise that such a row
                    // stores only zeros and `ln_1p(0) == 0` makes the omission
                    // invisible; it asserted that premise in debug builds.
                    //
                    // The premise does not hold for signed data, which
                    // `from_anndata` accepts: a row `[0.5, -0.5]` sums to 0 and
                    // a row `[-2.0, 1.0]` to -1. On those this branch left the
                    // values untouched while the unfused path below,
                    // `apply_transforms_to_csc`, and `sc.pp.normalize_total` +
                    // `sc.pp.log1p` all applied `ln_1p`, so the fusion was the
                    // outlier and the assertion fired on input the public API
                    // does not reject.
                    //
                    // `dataset_index.rs::apply_transforms_per_row` carries a
                    // second copy of this fusion, for fancy indexing and for
                    // any read once a deletion vector is set. It had the same
                    // bug and carries the same repair — the two must stay in
                    // step, or one matrix answers differently depending on how
                    // it is addressed.
                    //
                    // Applying `ln_1p` here makes all four agree. Counts data
                    // is unaffected: a sum of non-negative f32 values is 0 only
                    // when every stored value is 0, and `ln_1p(0) == 0`.
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    for v in &mut csr.data[start..end] {
                        *v = v.ln_1p();
                    }
                }
            }
            // Apply remaining transforms (index 2+)
            for transform in &transforms[2..] {
                apply_single_transform(csr, transform, global_row_offset);
            }
            return;
        }
    }

    // General path: apply each transform sequentially
    for transform in transforms {
        apply_single_transform(csr, transform, global_row_offset);
    }
}

/// Apply a single transform in-place.
fn apply_single_transform(csr: &mut ScxCsr, transform: &Transform, global_row_offset: usize) {
    match transform {
        Transform::NormalizeTotal {
            row_sums,
            target_sum,
        } => {
            for row in 0..csr.n_rows() {
                let g = global_row_offset + row;
                let sum = row_sums[g];
                if sum > 0.0 {
                    let factor = *target_sum / sum;
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    for v in &mut csr.data[start..end] {
                        *v = (*v as f64 * factor) as f32;
                    }
                }
            }
        }
        Transform::Log1p => {
            for v in &mut csr.data {
                *v = v.ln_1p();
            }
        }
        Transform::RowScale { factors } => {
            for row in 0..csr.n_rows() {
                let g = global_row_offset + row;
                let factor = factors[g] as f32;
                let start = csr.indptr[row] as usize;
                let end = csr.indptr[row + 1] as usize;
                for v in &mut csr.data[start..end] {
                    *v *= factor;
                }
            }
        }
        Transform::Scale { factor } => {
            for v in &mut csr.data {
                *v = (*v as f64 * *factor) as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LazyShardSource — ShardSource impl for streaming PCA through transforms
// ---------------------------------------------------------------------------
