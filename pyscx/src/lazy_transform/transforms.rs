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
                    // Zero-sum row: all stored values must be zero for CSR
                    // from count data. In the unfused path, NormalizeTotal
                    // skips the row and Log1p applies ln(0+1)=0, so both
                    // paths produce identical results when this invariant
                    // holds. Assert to catch upstream data corruption.
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    debug_assert!(
                        csr.data[start..end].iter().all(|&v| v == 0.0),
                        "Fused NormalizeTotal+Log1p: zero-sum row {} has non-zero values",
                        g
                    );
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
    }
}

// ---------------------------------------------------------------------------
// LazyShardSource — ShardSource impl for streaming PCA through transforms
// ---------------------------------------------------------------------------
