//! HVG (Highly Variable Gene) projection at decode time.
//!
//! During CSR row decode, only values whose column index is in the HVG set
//! are written into the dense output tensor. Non-HVG values are skipped.
//!
//! **Difference from `scx-engine`'s projection**: `scx-engine/src/projection.rs`
//! operates on materialized `ScxCsr` matrices (post-decode). This module's
//! projection operates during the decode→densify step: as each CSR row is
//! scattered into the dense output tensor, only HVG columns get a write.
//! This avoids materializing intermediate projected CSR data.

const DENSE_REMAP_MAX_BYTES: usize = 8 * 1024 * 1024;

/// HVG gene projection for the training data loader.
///
/// Maps original column indices to projected output positions. During scatter,
/// only genes in the HVG set are written to the dense output row.
#[derive(Clone)]
pub struct HvgProjection {
    /// Sorted, deduplicated HVG gene indices (original column indices).
    gene_indices: Vec<u32>,
    /// Dense lookup table: original column index -> projected output position.
    /// Non-HVG genes store `-1`. Built only when the index space is small
    /// enough to keep the extra memory bounded; otherwise `scatter_row` falls
    /// back to the merge-scan over `gene_indices`.
    dense_remap: Option<Vec<i32>>,
    /// Number of output columns.
    n_output_cols: usize,
}

impl HvgProjection {
    /// Create a new HVG projection from a list of gene indices.
    ///
    /// The input `gene_indices` are sorted and deduplicated. Each unique gene
    /// index is mapped to a contiguous output position `[0..n_hvg)`.
    pub fn new(gene_indices: Vec<u32>) -> Self {
        Self::new_with_dense_remap_limit(gene_indices, DENSE_REMAP_MAX_BYTES)
    }

    fn new_with_dense_remap_limit(
        mut gene_indices: Vec<u32>,
        max_dense_remap_bytes: usize,
    ) -> Self {
        gene_indices.sort_unstable();
        gene_indices.dedup();

        let n_output_cols = gene_indices.len();
        let dense_remap = Self::build_dense_remap(&gene_indices, max_dense_remap_bytes);
        HvgProjection {
            gene_indices,
            dense_remap,
            n_output_cols,
        }
    }

    fn build_dense_remap(gene_indices: &[u32], max_dense_remap_bytes: usize) -> Option<Vec<i32>> {
        let max_gene = gene_indices.last().copied()?;
        let len = usize::try_from(max_gene).ok()?.checked_add(1)?;
        let bytes = len.checked_mul(std::mem::size_of::<i32>())?;
        if bytes > max_dense_remap_bytes {
            return None;
        }

        let mut dense_remap = vec![-1; len];
        for (pos, &gene_idx) in gene_indices.iter().enumerate() {
            let pos = i32::try_from(pos).ok()?;
            let idx = usize::try_from(gene_idx).ok()?;
            dense_remap[idx] = pos;
        }
        Some(dense_remap)
    }

    /// Number of output columns (projected gene count).
    pub fn n_output_cols(&self) -> usize {
        self.n_output_cols
    }

    /// Scatter a CSR row into a dense output row, applying HVG projection.
    ///
    /// Uses a dense original-column-to-output-column remap when the selected
    /// gene index space is small enough, so sparse rows only pay O(row nnz).
    /// Falls back to a merge-scan over sorted CSR/HVG indices when the dense
    /// remap would exceed the memory cap.
    ///
    /// # Arguments
    /// - `csr_indices`: Column indices from the CSR row (sorted, i32 per scipy).
    /// - `csr_data`: Values from the CSR row (parallel to `csr_indices`).
    /// - `output_row`: Pre-zeroed dense output row of length `n_output_cols`.
    ///   Only HVG positions are written; non-HVG values are skipped.
    pub fn scatter_row(&self, csr_indices: &[i32], csr_data: &[f32], output_row: &mut [f32]) {
        debug_assert_eq!(output_row.len(), self.n_output_cols);

        if let Some(dense_remap) = &self.dense_remap {
            for (&col_idx, &value) in csr_indices.iter().zip(csr_data.iter()) {
                if col_idx < 0 {
                    continue;
                }
                let idx = col_idx as usize;
                let Some(&out_idx) = dense_remap.get(idx) else {
                    continue;
                };
                if out_idx >= 0 {
                    output_row[out_idx as usize] = value;
                }
            }
            return;
        }

        // Merge-scan: two pointers over sorted csr_indices and gene_indices
        let mut gi = 0; // pointer into self.gene_indices
        for (&col_idx, &value) in csr_indices.iter().zip(csr_data.iter()) {
            let col = col_idx as u32;

            // Advance gene pointer to catch up with current CSR column
            while gi < self.gene_indices.len() && self.gene_indices[gi] < col {
                gi += 1;
            }

            // If match, write to output at the projected position. CSR indices
            // are unique per row in well-formed data; if a duplicate slipped
            // in, the last write wins (we don't advance gi).
            if gi < self.gene_indices.len() && self.gene_indices[gi] == col {
                output_row[gi] = value;
            }
        }
    }

    #[cfg(test)]
    fn uses_dense_remap(&self) -> bool {
        self.dense_remap.is_some()
    }
}

/// Scatter a CSR row into a dense output row without projection (all genes).
///
/// Used when `hvg_indices` is `None` in the loader config. Scatters all CSR
/// values into the dense row at their original column positions.
///
/// # Arguments
/// - `csr_indices`: Column indices from the CSR row (i32 per scipy).
/// - `csr_data`: Values from the CSR row (parallel to `csr_indices`).
/// - `output_row`: Pre-zeroed dense output row of length `n_vars`.
pub fn scatter_row_full(
    csr_indices: &[i32],
    csr_data: &[f32],
    output_row: &mut [f32],
) -> Result<(), crate::error::LoaderError> {
    for (&col_idx, &value) in csr_indices.iter().zip(csr_data.iter()) {
        if col_idx < 0 {
            return Err(crate::error::LoaderError::ConfigError {
                reason: format!("negative CSR column index {col_idx}"),
            });
        }
        let idx = col_idx as usize;
        if idx >= output_row.len() {
            return Err(crate::error::LoaderError::ConfigError {
                reason: format!(
                    "CSR column index {idx} out of bounds for output row of length {}",
                    output_row.len()
                ),
            });
        }
        output_row[idx] = value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_3_of_30k() {
        // Project 3 genes out of a 30K gene space
        let proj = HvgProjection::new(vec![100, 500, 29999]);
        assert_eq!(proj.n_output_cols(), 3);

        // CSR row with values at various columns, including the 3 HVG genes
        let csr_indices: Vec<i32> = vec![50, 100, 200, 500, 1000, 29999];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 3];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        // Only the 3 HVG genes should have values
        assert_eq!(output[0], 2.0); // gene 100 → output position 0
        assert_eq!(output[1], 4.0); // gene 500 → output position 1
        assert_eq!(output[2], 6.0); // gene 29999 → output position 2
    }

    #[test]
    fn test_non_hvg_values_not_written() {
        let proj = HvgProjection::new(vec![5, 10]);
        let csr_indices: Vec<i32> = vec![0, 3, 5, 7, 10, 15];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 2];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output[0], 3.0); // gene 5 → position 0
        assert_eq!(output[1], 5.0); // gene 10 → position 1
    }

    #[test]
    fn test_output_column_indices_correctly_remapped() {
        let proj = HvgProjection::new(vec![20, 5, 100, 50]);
        // After sort+dedup: [5, 20, 50, 100]
        assert_eq!(proj.n_output_cols(), 4);
        assert!(proj.uses_dense_remap());

        let csr_indices: Vec<i32> = vec![5, 20, 50, 100];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 4];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_full_scatter_no_projection() {
        let csr_indices: Vec<i32> = vec![1, 3, 7];
        let csr_data: Vec<f32> = vec![2.0, 4.0, 6.0];
        let mut output = vec![0.0f32; 10];

        scatter_row_full(&csr_indices, &csr_data, &mut output).unwrap();

        assert_eq!(output[0], 0.0);
        assert_eq!(output[1], 2.0);
        assert_eq!(output[2], 0.0);
        assert_eq!(output[3], 4.0);
        assert_eq!(output[7], 6.0);
        // All other positions remain zero
        assert_eq!(output[4], 0.0);
        assert_eq!(output[5], 0.0);
        assert_eq!(output[6], 0.0);
        assert_eq!(output[8], 0.0);
        assert_eq!(output[9], 0.0);
    }

    #[test]
    fn test_empty_hvg_set_all_zeros() {
        let proj = HvgProjection::new(vec![]);
        assert_eq!(proj.n_output_cols(), 0);

        let csr_indices: Vec<i32> = vec![0, 1, 2];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut output: Vec<f32> = vec![];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_duplicate_gene_indices_deduplicated() {
        let proj = HvgProjection::new(vec![5, 5, 10, 10, 10, 20]);
        assert_eq!(proj.n_output_cols(), 3); // only 3 unique: [5, 10, 20]

        let csr_indices: Vec<i32> = vec![5, 10, 20];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_no_matching_genes_in_csr_row() {
        let proj = HvgProjection::new(vec![100, 200, 300]);
        // CSR row has no genes matching the HVG set
        let csr_indices: Vec<i32> = vec![0, 1, 50, 99];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 3];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_all_csr_genes_are_hvg() {
        let proj = HvgProjection::new(vec![0, 1, 2, 3, 4]);
        let csr_indices: Vec<i32> = vec![0, 1, 2, 3, 4];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0f32; 5];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_dense_remap_handles_sparse_row_and_out_of_range_non_hvg() {
        let mut genes: Vec<u32> = (0..2000).collect();
        genes.push(29_999);
        let proj = HvgProjection::new(genes);
        assert_eq!(proj.n_output_cols(), 2001);
        assert!(proj.uses_dense_remap());

        let csr_indices: Vec<i32> = vec![7, 29_999, 40_000];
        let csr_data: Vec<f32> = vec![1.5, 2.5, 3.5];
        let mut output = vec![0.0f32; proj.n_output_cols()];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output[7], 1.5);
        assert_eq!(output[2000], 2.5);
        assert_eq!(output.iter().filter(|&&v| v != 0.0).count(), 2);
    }

    #[test]
    fn test_merge_scan_fallback_when_dense_remap_exceeds_limit() {
        let proj = HvgProjection::new_with_dense_remap_limit(vec![5, 20, 100], 0);
        assert_eq!(proj.n_output_cols(), 3);
        assert!(!proj.uses_dense_remap());

        let csr_indices: Vec<i32> = vec![1, 5, 7, 20, 100, 200];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 3];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output, vec![2.0, 4.0, 5.0]);
    }

    #[test]
    fn test_scatter_row_full_empty_row() {
        let csr_indices: Vec<i32> = vec![];
        let csr_data: Vec<f32> = vec![];
        let mut output = vec![0.0f32; 5];

        scatter_row_full(&csr_indices, &csr_data, &mut output).unwrap();

        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_scatter_row_full_negative_index_returns_error() {
        let csr_indices: Vec<i32> = vec![1, -5, 3];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 10];

        let result = scatter_row_full(&csr_indices, &csr_data, &mut output);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("negative"), "expected 'negative' in: {msg}");
    }

    #[test]
    fn test_scatter_row_full_oob_index_returns_error() {
        let csr_indices: Vec<i32> = vec![1, 3, 100];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 10];

        let result = scatter_row_full(&csr_indices, &csr_data, &mut output);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of bounds"),
            "expected 'out of bounds' in: {msg}"
        );
    }
}
