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

use std::collections::HashMap;

/// HVG gene projection for the training data loader.
///
/// Maps original column indices to projected output positions. During scatter,
/// only genes in the HVG set are written to the dense output row.
#[derive(Clone)]
pub struct HvgProjection {
    /// Sorted, deduplicated HVG gene indices (original column indices).
    gene_indices: Vec<u32>,
    /// Lookup table: original column index → position in projected output.
    /// Returns `None` for non-HVG genes. Kept for validation and future use.
    #[allow(dead_code)]
    remap: HashMap<u32, u32>,
    /// Number of output columns.
    n_output_cols: usize,
}

impl HvgProjection {
    /// Create a new HVG projection from a list of gene indices.
    ///
    /// The input `gene_indices` are sorted and deduplicated. Each unique gene
    /// index is mapped to a contiguous output position `[0..n_hvg)`.
    pub fn new(mut gene_indices: Vec<u32>) -> Self {
        gene_indices.sort_unstable();
        gene_indices.dedup();

        let mut remap = HashMap::with_capacity(gene_indices.len());
        for (pos, &gene_idx) in gene_indices.iter().enumerate() {
            remap.insert(gene_idx, pos as u32);
        }

        let n_output_cols = gene_indices.len();
        HvgProjection {
            gene_indices,
            remap,
            n_output_cols,
        }
    }

    /// Number of output columns (projected gene count).
    pub fn n_output_cols(&self) -> usize {
        self.n_output_cols
    }

    /// Scatter a CSR row into a dense output row, applying HVG projection.
    ///
    /// Uses a merge-scan (two-pointer) algorithm for O(n+m) performance with
    /// good cache locality, since both `csr_indices` and `gene_indices` are
    /// sorted.
    ///
    /// # Arguments
    /// - `csr_indices`: Column indices from the CSR row (sorted, i32 per scipy).
    /// - `csr_data`: Values from the CSR row (parallel to `csr_indices`).
    /// - `output_row`: Pre-zeroed dense output row of length `n_output_cols`.
    ///   Only HVG positions are written; non-HVG values are skipped.
    pub fn scatter_row(&self, csr_indices: &[i32], csr_data: &[f32], output_row: &mut [f32]) {
        debug_assert_eq!(output_row.len(), self.n_output_cols);

        // Merge-scan: two pointers over sorted csr_indices and gene_indices
        let mut gi = 0; // pointer into self.gene_indices
        for (ci, (&col_idx, &value)) in csr_indices.iter().zip(csr_data.iter()).enumerate() {
            let col = col_idx as u32;

            // Advance gene pointer to catch up with current CSR column
            while gi < self.gene_indices.len() && self.gene_indices[gi] < col {
                gi += 1;
            }

            // If match, write to output at the projected position
            if gi < self.gene_indices.len() && self.gene_indices[gi] == col {
                output_row[gi] = value;
                // Don't advance gi here — CSR indices should be unique per row,
                // but if there were duplicates, the last value wins. In normal
                // CSR data, indices are unique so this is fine.
            }
            let _ = ci; // suppress unused variable warning
        }
    }

    /// Scatter two CSR rows (perturbed + control) into two dense outputs in a
    /// single pass over the HVG gene indices.
    ///
    /// Pair-aware variant of [`scatter_row`]: walks `gene_indices` once and
    /// advances both CSR pointers in parallel, sharing the gene-index lookup
    /// cost between the two outputs. Bit-identical to two `scatter_row` calls.
    ///
    /// # Arguments
    /// - `p_csr_indices`, `p_csr_data`: CSR row for the perturbed cell.
    /// - `c_csr_indices`, `c_csr_data`: CSR row for the control cell.
    /// - `p_out`, `c_out`: pre-zeroed dense outputs of length `n_output_cols`.
    pub fn scatter_pair_rows(
        &self,
        p_csr_indices: &[i32],
        p_csr_data: &[f32],
        c_csr_indices: &[i32],
        c_csr_data: &[f32],
        p_out: &mut [f32],
        c_out: &mut [f32],
    ) {
        debug_assert_eq!(p_out.len(), self.n_output_cols);
        debug_assert_eq!(c_out.len(), self.n_output_cols);
        debug_assert_eq!(p_csr_indices.len(), p_csr_data.len());
        debug_assert_eq!(c_csr_indices.len(), c_csr_data.len());

        let mut pi = 0usize;
        let mut ci = 0usize;

        // Walk gene_indices once. For each HVG gene, advance both CSR
        // pointers to the first column >= gene; write the value if equal.
        for (gi, &gene) in self.gene_indices.iter().enumerate() {
            let gene_i = gene as i32;

            while pi < p_csr_indices.len() && p_csr_indices[pi] < gene_i {
                pi += 1;
            }
            if pi < p_csr_indices.len() && p_csr_indices[pi] == gene_i {
                p_out[gi] = p_csr_data[pi];
            }

            while ci < c_csr_indices.len() && c_csr_indices[ci] < gene_i {
                ci += 1;
            }
            if ci < c_csr_indices.len() && c_csr_indices[ci] == gene_i {
                c_out[gi] = c_csr_data[ci];
            }
        }
    }

    /// Get the remap HashMap (for validation/debugging).
    #[cfg(test)]
    fn remap(&self) -> &HashMap<u32, u32> {
        &self.remap
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
        assert_eq!(proj.remap()[&5], 0);
        assert_eq!(proj.remap()[&20], 1);
        assert_eq!(proj.remap()[&50], 2);
        assert_eq!(proj.remap()[&100], 3);
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
        assert_eq!(proj.remap()[&5], 0);
        assert_eq!(proj.remap()[&10], 1);
        assert_eq!(proj.remap()[&20], 2);
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

    // ---------------------------------------------------------------------
    // Phase 3 — scatter_pair_rows
    // ---------------------------------------------------------------------

    /// Helper: run scatter_row twice, return the (p_out, c_out) tuple. Used
    /// as the parity reference for scatter_pair_rows.
    fn ref_two_scatter_row(
        proj: &HvgProjection,
        p_idx: &[i32],
        p_dat: &[f32],
        c_idx: &[i32],
        c_dat: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut p = vec![0f32; proj.n_output_cols()];
        let mut c = vec![0f32; proj.n_output_cols()];
        proj.scatter_row(p_idx, p_dat, &mut p);
        proj.scatter_row(c_idx, c_dat, &mut c);
        (p, c)
    }

    #[test]
    fn test_pair_scatter_basic_parity() {
        let proj = HvgProjection::new(vec![100, 500, 29999]);
        let p_idx: Vec<i32> = vec![50, 100, 200, 500, 1000, 29999];
        let p_dat: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c_idx: Vec<i32> = vec![100, 300, 500];
        let c_dat: Vec<f32> = vec![10.0, 20.0, 30.0];

        let (p_ref, c_ref) = ref_two_scatter_row(&proj, &p_idx, &p_dat, &c_idx, &c_dat);
        let mut p_out = vec![0f32; 3];
        let mut c_out = vec![0f32; 3];
        proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
        assert_eq!(p_out, p_ref);
        assert_eq!(c_out, c_ref);
        assert_eq!(p_out, vec![2.0, 4.0, 6.0]);
        assert_eq!(c_out, vec![10.0, 30.0, 0.0]);
    }

    #[test]
    fn test_pair_scatter_empty_pert_row() {
        let proj = HvgProjection::new(vec![1, 2, 3]);
        let p_idx: Vec<i32> = vec![];
        let p_dat: Vec<f32> = vec![];
        let c_idx: Vec<i32> = vec![1, 2, 3];
        let c_dat: Vec<f32> = vec![10.0, 20.0, 30.0];

        let (p_ref, c_ref) = ref_two_scatter_row(&proj, &p_idx, &p_dat, &c_idx, &c_dat);
        let mut p_out = vec![0f32; 3];
        let mut c_out = vec![0f32; 3];
        proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
        assert_eq!(p_out, p_ref);
        assert_eq!(c_out, c_ref);
        assert!(p_out.iter().all(|&v| v == 0.0));
        assert_eq!(c_out, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_pair_scatter_empty_ctrl_row() {
        let proj = HvgProjection::new(vec![1, 2, 3]);
        let p_idx: Vec<i32> = vec![1, 2, 3];
        let p_dat: Vec<f32> = vec![10.0, 20.0, 30.0];
        let c_idx: Vec<i32> = vec![];
        let c_dat: Vec<f32> = vec![];

        let (p_ref, c_ref) = ref_two_scatter_row(&proj, &p_idx, &p_dat, &c_idx, &c_dat);
        let mut p_out = vec![0f32; 3];
        let mut c_out = vec![0f32; 3];
        proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
        assert_eq!(p_out, p_ref);
        assert_eq!(c_out, c_ref);
    }

    #[test]
    fn test_pair_scatter_both_empty() {
        let proj = HvgProjection::new(vec![1, 2, 3]);
        let mut p_out = vec![0f32; 3];
        let mut c_out = vec![0f32; 3];
        proj.scatter_pair_rows(&[], &[], &[], &[], &mut p_out, &mut c_out);
        assert!(p_out.iter().all(|&v| v == 0.0));
        assert!(c_out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pair_scatter_identical_rows() {
        let proj = HvgProjection::new(vec![5, 10, 15]);
        let idx: Vec<i32> = vec![5, 10, 15];
        let dat: Vec<f32> = vec![1.5, 2.5, 3.5];

        let (p_ref, c_ref) = ref_two_scatter_row(&proj, &idx, &dat, &idx, &dat);
        let mut p_out = vec![0f32; 3];
        let mut c_out = vec![0f32; 3];
        proj.scatter_pair_rows(&idx, &dat, &idx, &dat, &mut p_out, &mut c_out);
        assert_eq!(p_out, p_ref);
        assert_eq!(c_out, c_ref);
        assert_eq!(p_out, c_out);
    }

    #[test]
    fn test_pair_scatter_non_overlapping_hvg_sets() {
        // Two CSR rows whose nonzero columns are disjoint from each other,
        // yet both partially overlap the HVG set.
        let proj = HvgProjection::new(vec![1, 5, 10, 100]);
        let p_idx: Vec<i32> = vec![1, 10]; // hits HVG positions 0 and 2
        let p_dat: Vec<f32> = vec![1.0, 2.0];
        let c_idx: Vec<i32> = vec![5, 100]; // hits HVG positions 1 and 3
        let c_dat: Vec<f32> = vec![5.0, 100.0];

        let (p_ref, c_ref) = ref_two_scatter_row(&proj, &p_idx, &p_dat, &c_idx, &c_dat);
        let mut p_out = vec![0f32; 4];
        let mut c_out = vec![0f32; 4];
        proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
        assert_eq!(p_out, p_ref);
        assert_eq!(c_out, c_ref);
        assert_eq!(p_out, vec![1.0, 0.0, 2.0, 0.0]);
        assert_eq!(c_out, vec![0.0, 5.0, 0.0, 100.0]);
    }

    #[test]
    fn test_pair_scatter_no_matching_genes() {
        let proj = HvgProjection::new(vec![1000, 2000, 3000]);
        let p_idx: Vec<i32> = vec![10, 20, 30];
        let p_dat: Vec<f32> = vec![1.0, 2.0, 3.0];
        let c_idx: Vec<i32> = vec![5, 15, 25];
        let c_dat: Vec<f32> = vec![10.0, 20.0, 30.0];

        let mut p_out = vec![0f32; 3];
        let mut c_out = vec![0f32; 3];
        proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
        assert!(p_out.iter().all(|&v| v == 0.0));
        assert!(c_out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_pair_scatter_empty_hvg_set() {
        let proj = HvgProjection::new(vec![]);
        let p_idx: Vec<i32> = vec![1, 2, 3];
        let p_dat: Vec<f32> = vec![1.0, 2.0, 3.0];
        let c_idx: Vec<i32> = vec![4, 5, 6];
        let c_dat: Vec<f32> = vec![4.0, 5.0, 6.0];

        let mut p_out: Vec<f32> = vec![];
        let mut c_out: Vec<f32> = vec![];
        proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
        assert!(p_out.is_empty());
        assert!(c_out.is_empty());
    }

    #[test]
    fn test_pair_scatter_random_parity() {
        // Stress: pseudo-random rows + HVG set, parity vs scatter_row x 2.
        // Deterministic LCG so the test is reproducible.
        let mut state: u32 = 0x9E3779B1;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            state
        };

        let n_vars: u32 = 5_000;
        let mut hvg: Vec<u32> = (0..400).map(|_| next() % n_vars).collect();
        hvg.sort_unstable();
        hvg.dedup();
        let proj = HvgProjection::new(hvg);

        for trial in 0..16 {
            let nnz_p = 100 + (next() as usize % 400);
            let nnz_c = 100 + (next() as usize % 400);
            let mut p_cols: Vec<u32> = (0..nnz_p).map(|_| next() % n_vars).collect();
            p_cols.sort_unstable();
            p_cols.dedup();
            let mut c_cols: Vec<u32> = (0..nnz_c).map(|_| next() % n_vars).collect();
            c_cols.sort_unstable();
            c_cols.dedup();
            let p_idx: Vec<i32> = p_cols.iter().map(|&x| x as i32).collect();
            let p_dat: Vec<f32> = (0..p_idx.len()).map(|i| (i + 1) as f32).collect();
            let c_idx: Vec<i32> = c_cols.iter().map(|&x| x as i32).collect();
            let c_dat: Vec<f32> = (0..c_idx.len()).map(|i| (i + 100) as f32).collect();

            let (p_ref, c_ref) = ref_two_scatter_row(&proj, &p_idx, &p_dat, &c_idx, &c_dat);
            let mut p_out = vec![0f32; proj.n_output_cols()];
            let mut c_out = vec![0f32; proj.n_output_cols()];
            proj.scatter_pair_rows(&p_idx, &p_dat, &c_idx, &c_dat, &mut p_out, &mut c_out);
            assert_eq!(p_out, p_ref, "trial {trial}: pert mismatch");
            assert_eq!(c_out, c_ref, "trial {trial}: ctrl mismatch");
        }
    }

    /// Phase 3.4 — Microbenchmark gate, run with:
    ///     cargo test --release -p scx-loader projection::tests::bench_scatter_pair_rows -- --ignored --nocapture
    /// Compares scatter_row × 2 vs scatter_pair_rows on a realistic-sparsity
    /// workload (n_vars=20K, ~5% density, 2K HVGs). 1024 pairs per repetition,
    /// 16 alternating reps to dampen scheduler / thermal noise; reports the
    /// median ratio. Spec gates: revert if median speedup < 5%.
    #[test]
    #[ignore]
    fn bench_scatter_pair_rows_vs_two_scatter_row() {
        use std::time::Instant;

        let n_vars: u32 = 20_000;
        let n_hvg: usize = 2_000;
        let nnz_per_row: usize = 1_000;
        let n_pairs: usize = 1024;
        let reps: usize = 16;

        let mut state: u64 = 0xDEADBEEFCAFEBABE;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            state
        };

        let mut hvg: Vec<u32> = (0..n_hvg).map(|_| (next() as u32) % n_vars).collect();
        hvg.sort_unstable();
        hvg.dedup();
        let proj = HvgProjection::new(hvg);
        let n_cols = proj.n_output_cols();

        let pairs: Vec<(Vec<i32>, Vec<f32>, Vec<i32>, Vec<f32>)> = (0..n_pairs)
            .map(|_| {
                let mut p_cols: Vec<u32> =
                    (0..nnz_per_row).map(|_| (next() as u32) % n_vars).collect();
                p_cols.sort_unstable();
                p_cols.dedup();
                let mut c_cols: Vec<u32> =
                    (0..nnz_per_row).map(|_| (next() as u32) % n_vars).collect();
                c_cols.sort_unstable();
                c_cols.dedup();
                let p_dat: Vec<f32> = (0..p_cols.len()).map(|i| i as f32 + 1.0).collect();
                let c_dat: Vec<f32> = (0..c_cols.len()).map(|i| i as f32 + 100.0).collect();
                (
                    p_cols.iter().map(|&x| x as i32).collect(),
                    p_dat,
                    c_cols.iter().map(|&x| x as i32).collect(),
                    c_dat,
                )
            })
            .collect();

        let mut p_out = vec![0f32; n_cols];
        let mut c_out = vec![0f32; n_cols];

        // Warm-up — caches, branch predictor.
        for (p_idx, p_dat, c_idx, c_dat) in &pairs[..32] {
            p_out.fill(0.0);
            c_out.fill(0.0);
            proj.scatter_row(p_idx, p_dat, &mut p_out);
            proj.scatter_row(c_idx, c_dat, &mut c_out);
            proj.scatter_pair_rows(p_idx, p_dat, c_idx, c_dat, &mut p_out, &mut c_out);
        }

        let mut sink = 0.0f32;
        let mut baseline_ns = Vec::with_capacity(reps);
        let mut pair_ns = Vec::with_capacity(reps);
        let mut ratios: Vec<f64> = Vec::with_capacity(reps);

        // Interleave A and B to share whatever transient state exists.
        for _ in 0..reps {
            let t0 = Instant::now();
            for (p_idx, p_dat, c_idx, c_dat) in &pairs {
                p_out.fill(0.0);
                c_out.fill(0.0);
                proj.scatter_row(p_idx, p_dat, &mut p_out);
                proj.scatter_row(c_idx, c_dat, &mut c_out);
                sink += p_out[0] + c_out[0];
            }
            let b = t0.elapsed().as_nanos() as f64;
            baseline_ns.push(b);

            let t1 = Instant::now();
            for (p_idx, p_dat, c_idx, c_dat) in &pairs {
                p_out.fill(0.0);
                c_out.fill(0.0);
                proj.scatter_pair_rows(p_idx, p_dat, c_idx, c_dat, &mut p_out, &mut c_out);
                sink += p_out[0] + c_out[0];
            }
            let p = t1.elapsed().as_nanos() as f64;
            pair_ns.push(p);

            ratios.push(b / p);
        }

        baseline_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pair_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let med_b = baseline_ns[reps / 2] / 1e6;
        let med_p = pair_ns[reps / 2] / 1e6;
        let med_r = ratios[reps / 2];
        let min_r = ratios[0];
        let max_r = ratios[reps - 1];

        eprintln!(
            "[bench_scatter_pair_rows] {n_pairs} pairs, n_vars={n_vars}, n_hvg={n_hvg}, ~{nnz_per_row} nnz/row, {reps} reps\n\
             scatter_row × 2   median: {med_b:.3} ms\n\
             scatter_pair_rows median: {med_p:.3} ms\n\
             speedup           median: {med_r:.3}× (min {min_r:.3}×, max {max_r:.3}×) (sink={sink})",
        );
    }
}
