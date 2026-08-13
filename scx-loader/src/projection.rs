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

use crate::error::LoaderError;

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

/// Construction-time verdict: building the projection changed the panel the
/// caller passed.
///
/// The panel is canonicalised to ascending-unique (see
/// [`HvgProjection::normalize_panel`]), so a caller who passed rank order —
/// `np.argsort(-variances)[:2000]`, a natural thing to write — gets output
/// columns in *gene-index* order instead, and duplicates shrink the batch
/// width. Neither is wrong, but both silently break the caller's own mapping
/// from column position back to gene, which is the only place that mapping
/// exists. `None` means the panel was already canonical and nothing needs
/// saying.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HvgPanelVerdict {
    /// Length of the panel as passed.
    pub requested_len: usize,
    /// Output columns after deduplication — the actual batch width.
    pub unique_len: usize,
    /// The panel was not in ascending order, so output columns are not in the
    /// order it was written in.
    pub was_reordered: bool,
}

/// Returns a verdict iff building the projection would change the panel.
///
/// Pure, so it stays unit-testable without a file, and so the two loader
/// constructors can share one definition of "worth warning about".
pub fn assess_hvg_panel(gene_indices: &[u32]) -> Option<HvgPanelVerdict> {
    let requested_len = gene_indices.len();
    let unique_len = HvgProjection::output_cols_for(gene_indices);
    let was_reordered = gene_indices.windows(2).any(|w| w[0] > w[1]);
    if unique_len == requested_len && !was_reordered {
        return None;
    }
    Some(HvgPanelVerdict {
        requested_len,
        unique_len,
        was_reordered,
    })
}

impl HvgProjection {
    /// Create a new HVG projection, validating every index against the file's
    /// gene count.
    ///
    /// The input `gene_indices` are sorted and deduplicated. Each unique gene
    /// index is mapped to a contiguous output position `[0..n_hvg)`.
    ///
    /// # Why this is the only checked constructor
    ///
    /// An index `>= n_vars` matches no CSR column on either scatter path — the
    /// dense remap takes `dense_remap.get(idx) -> None -> continue`, the
    /// merge-scan simply never finds it — so the projected batch carries a
    /// column that is **always exactly zero**. Nothing downstream can tell that
    /// apart from a gene that happens to be silent, so it becomes a dead input
    /// feature that trains to a zero weight and is never diagnosed. That is why
    /// the check lives here and not in the callers: `IndexPlanLoader` had it and
    /// `TrainingPipeline` did not, and the divergence was invisible from either
    /// side.
    ///
    /// An empty panel is valid and yields `n_output_cols() == 0`.
    ///
    /// # Errors
    ///
    /// [`LoaderError::ConfigError`] when any index is `>= n_vars`, or when
    /// `n_vars` itself exceeds `u32::MAX` (HVG indices are `u32`).
    pub fn new(gene_indices: Vec<u32>, n_vars: u64) -> Result<Self, LoaderError> {
        let n_vars_u32: u32 = u32::try_from(n_vars).map_err(|_| LoaderError::ConfigError {
            reason: format!("n_vars={n_vars} exceeds u32::MAX; HVG indices use u32"),
        })?;
        if let Some(&bad) = gene_indices.iter().find(|&&i| i >= n_vars_u32) {
            return Err(LoaderError::ConfigError {
                reason: format!("HVG index {bad} is out of range (n_vars={n_vars})"),
            });
        }
        Ok(Self::new_unchecked(gene_indices))
    }

    /// Build a projection **without** the `n_vars` range check.
    ///
    /// Two callers, both deliberate:
    ///
    /// 1. the geometry unit tests below, which have no file behind them and so
    ///    no meaningful `n_vars`;
    /// 2. the `LoaderConfig::shared_hvg_panel` branch of `TrainingPipeline::new`,
    ///    for the one caller that fans a single panel across modalities of
    ///    differing widths — see that field's docs for why.
    ///
    /// Everything else must go through [`HvgProjection::new`]. Keeping the
    /// bypass a *named* function rather than an `if` around the check is the
    /// point: every site that skips validation is one grep away.
    pub(crate) fn new_unchecked(gene_indices: Vec<u32>) -> Self {
        Self::new_with_dense_remap_limit(gene_indices, DENSE_REMAP_MAX_BYTES)
    }

    /// Canonicalise a panel to ascending order with no duplicates.
    ///
    /// Not a convenience: the merge-scan scatter uses its cursor into
    /// `gene_indices` **as** the output column index, so ascending order is
    /// what makes that path agree with the dense-remap path, and a duplicate
    /// would leave one of the two columns permanently unwritten. Output columns
    /// therefore come back in gene-index order whatever order the caller passed.
    ///
    /// The single definition of a panel's width and column order — see
    /// [`Self::output_cols_for`] for the reason it is factored out.
    fn normalize_panel(gene_indices: &mut Vec<u32>) {
        gene_indices.sort_unstable();
        gene_indices.dedup();
    }

    /// Number of output columns a panel would produce, without building the
    /// projection.
    ///
    /// The memory model has to size the batch buffer before there is a
    /// projection to ask (`pipeline::compute_memory_budget` takes only a
    /// `LoaderConfig`), and answering with `hvg_indices.len()` was wrong for any
    /// panel containing duplicates: the budget costed the raw length while the
    /// batch was allocated at the deduplicated one, so a 2000-entry panel with
    /// 200 duplicates reserved 10 % more than it could ever use — and could
    /// auto-tune `batch_size` down to afford memory that was never needed.
    pub fn output_cols_for(gene_indices: &[u32]) -> usize {
        let mut panel = gene_indices.to_vec();
        Self::normalize_panel(&mut panel);
        panel.len()
    }

    fn new_with_dense_remap_limit(
        mut gene_indices: Vec<u32>,
        max_dense_remap_bytes: usize,
    ) -> Self {
        Self::normalize_panel(&mut gene_indices);

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

    /// Scatter a CSR row as **PFlog (v4)** into a projected dense output row.
    ///
    /// Mirrors [`scatter_row`](Self::scatter_row) (dense-remap fast path +
    /// merge-scan fallback), but writes the exact PFlog transform
    /// `z = log1p(4α·x) + baseline` instead of the raw value, and pre-fills
    /// every output column with `baseline` (so projected columns that were zero
    /// in the original row carry the per-cell baseline, as the exact transform
    /// requires).
    ///
    /// The centering denominator `n_vars_full` is taken over the **full
    /// pre-projection row** (`csr_indices`/`csr_data` span the whole
    /// transcriptome), NOT the projected panel — see
    /// [`crate::normalize::pflog_baseline_row`]. v4 has no per-cell depth; an
    /// empty cell yields `baseline = 0` (the output row is all zeros).
    ///
    /// # Arguments
    /// - `csr_indices` / `csr_data`: the full CSR row (sorted, i32 columns).
    /// - `four_alpha`: `4α` (the matrix-wide Anscombe scale; pseudocount `1/(4α)`).
    /// - `n_vars_full`: full feature count `D` (NOT `n_output_cols`).
    /// - `output_row`: pre-zeroed dense output row of length `n_output_cols`.
    pub fn scatter_pflog_row(
        &self,
        csr_indices: &[i32],
        csr_data: &[f32],
        four_alpha: f64,
        n_vars_full: usize,
        output_row: &mut [f32],
    ) {
        debug_assert_eq!(output_row.len(), self.n_output_cols);

        let Some(baseline) =
            crate::normalize::pflog_baseline_row(csr_data, four_alpha, n_vars_full)
        else {
            // Degenerate (n_vars == 0); leave the row zeroed.
            for v in output_row.iter_mut() {
                *v = 0.0;
            }
            return;
        };
        let baseline_f32 = baseline as f32;
        for v in output_row.iter_mut() {
            *v = baseline_f32;
        }

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
                    output_row[out_idx as usize] =
                        ((four_alpha * value.max(0.0) as f64).ln_1p() + baseline) as f32;
                }
            }
            return;
        }

        // Merge-scan: two pointers over sorted csr_indices and gene_indices.
        let mut gi = 0;
        for (&col_idx, &value) in csr_indices.iter().zip(csr_data.iter()) {
            if col_idx < 0 {
                continue;
            }
            let col = col_idx as u32;
            while gi < self.gene_indices.len() && self.gene_indices[gi] < col {
                gi += 1;
            }
            if gi < self.gene_indices.len() && self.gene_indices[gi] == col {
                output_row[gi] = ((four_alpha * value.max(0.0) as f64).ln_1p() + baseline) as f32;
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

/// Scatter a CSR row as **PFlog (v4)** without projection (all genes).
///
/// No-projection analog of [`HvgProjection::scatter_pflog_row`]: writes the
/// exact transform `z = log1p(4α·x) + baseline` at each original column,
/// pre-filling every column with `baseline` so original zeros carry it. `n_vars`
/// is both the output width and the centering denominator `D`. v4 has no depth;
/// an empty cell yields `baseline = 0` (the output row is all zeros).
pub fn pflog_row_full(
    csr_indices: &[i32],
    csr_data: &[f32],
    four_alpha: f64,
    n_vars: usize,
    output_row: &mut [f32],
) -> Result<(), crate::error::LoaderError> {
    // No-projection contract: the dense output row spans the full transcriptome,
    // so its width is the centering denominator `n_vars`.
    debug_assert_eq!(output_row.len(), n_vars);
    let Some(baseline) = crate::normalize::pflog_baseline_row(csr_data, four_alpha, n_vars) else {
        for v in output_row.iter_mut() {
            *v = 0.0;
        }
        return Ok(());
    };
    let baseline_f32 = baseline as f32;
    for v in output_row.iter_mut() {
        *v = baseline_f32;
    }
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
        output_row[idx] = ((four_alpha * value.max(0.0) as f64).ln_1p() + baseline) as f32;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- Range validation (`new` vs `new_unchecked`) --------------------
    //
    // An out-of-range index is invisible downstream — `scatter_row` skips it on
    // both paths and the batch simply carries an always-zero column — so these
    // assert on the constructor, which is the only place the difference exists.

    /// The boundary is `>= n_vars`, not `> n_vars`: on a 50-gene file, index 50
    /// is already out of range. This is the exact off-by-one that reached the
    /// `TrainingPipeline` path as a silent dead feature column.
    #[test]
    fn new_rejects_an_index_equal_to_n_vars() {
        // `let Err(..) else` rather than `unwrap_err()`: the Ok type holds a
        // dense remap that can reach 8 MiB, and unwrap_err would need Debug on
        // it and dump the whole thing on failure.
        let Err(err) = HvgProjection::new(vec![0, 1, 50], 50) else {
            panic!("index 50 must be rejected on a 50-gene file");
        };
        let msg = err.to_string();
        // The wording is load-bearing: `IndexPlanLoader`'s pre-existing Rust and
        // Python tests assert against this exact string, and they were left
        // unmodified so that they witness the hoist preserving behaviour.
        assert!(
            msg.contains("HVG index 50")
                && msg.contains("out of range")
                && msg.contains("n_vars=50"),
            "unexpected message: {msg}"
        );
        assert!(matches!(err, LoaderError::ConfigError { .. }));
    }

    #[test]
    fn new_rejects_an_index_far_past_n_vars() {
        let Err(err) = HvgProjection::new(vec![0, 1, 99_999], 50) else {
            panic!("index 99999 must be rejected on a 50-gene file");
        };
        assert!(err.to_string().contains("HVG index 99999"));
    }

    /// The last valid index must still be accepted — a check written as `>`
    /// instead of `>=` passes the rejection tests above while silently costing
    /// the file its final gene, so both halves of the boundary are pinned.
    #[test]
    fn new_accepts_the_last_valid_index() {
        let proj = HvgProjection::new(vec![0, 49], 50).unwrap();
        assert_eq!(proj.n_output_cols(), 2);
    }

    /// An empty panel is valid on both loader paths today (`n_output_genes` is
    /// 0 on each), so hoisting the check must not start rejecting it.
    #[test]
    fn new_accepts_an_empty_panel_even_on_an_empty_file() {
        assert_eq!(HvgProjection::new(vec![], 0).unwrap().n_output_cols(), 0);
        assert_eq!(HvgProjection::new(vec![], 50).unwrap().n_output_cols(), 0);
    }

    /// `n_vars` beyond `u32::MAX` cannot be compared against `u32` indices, so
    /// it is refused rather than silently truncated into a wrong bound.
    #[test]
    fn new_rejects_an_n_vars_past_u32_max() {
        let Err(err) = HvgProjection::new(vec![0], u32::MAX as u64 + 1) else {
            panic!("an n_vars past u32::MAX must be rejected");
        };
        assert!(
            err.to_string().contains("exceeds u32::MAX"),
            "unexpected message: {err}"
        );
    }

    /// Validation is the *only* thing `new` adds: sort, dedup and the resulting
    /// output width must be identical to the unchecked path, since both loaders
    /// already agreed on that behaviour before the hoist.
    #[test]
    fn new_matches_new_unchecked_apart_from_the_range_check() {
        let panel = vec![20u32, 5, 100, 5, 50];
        let checked = HvgProjection::new(panel.clone(), 101).unwrap();
        let unchecked = HvgProjection::new_unchecked(panel);
        assert_eq!(checked.n_output_cols(), unchecked.n_output_cols());
        assert_eq!(checked.gene_indices, unchecked.gene_indices);
        assert_eq!(checked.dense_remap, unchecked.dense_remap);
    }

    #[test]
    fn test_projection_3_of_30k() {
        // Project 3 genes out of a 30K gene space
        let proj = HvgProjection::new_unchecked(vec![100, 500, 29999]);
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
        let proj = HvgProjection::new_unchecked(vec![5, 10]);
        let csr_indices: Vec<i32> = vec![0, 3, 5, 7, 10, 15];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = vec![0.0f32; 2];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output[0], 3.0); // gene 5 → position 0
        assert_eq!(output[1], 5.0); // gene 10 → position 1
    }

    #[test]
    fn test_output_column_indices_correctly_remapped() {
        let proj = HvgProjection::new_unchecked(vec![20, 5, 100, 50]);
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
        let proj = HvgProjection::new_unchecked(vec![]);
        assert_eq!(proj.n_output_cols(), 0);

        let csr_indices: Vec<i32> = vec![0, 1, 2];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut output: Vec<f32> = vec![];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);
        assert!(output.is_empty());
    }

    #[test]
    fn test_duplicate_gene_indices_deduplicated() {
        let proj = HvgProjection::new_unchecked(vec![5, 5, 10, 10, 10, 20]);
        assert_eq!(proj.n_output_cols(), 3); // only 3 unique: [5, 10, 20]

        let csr_indices: Vec<i32> = vec![5, 10, 20];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0f32; 3];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert_eq!(output, vec![1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------
    // Panel width and the canonicalisation verdict
    // -----------------------------------------------------------------

    /// `output_cols_for` must agree with the built projection on every shape,
    /// because the memory model uses the former to size what the latter
    /// allocates. Disagreement is the bug: the budget used to cost
    /// `hvg_indices.len()`.
    #[test]
    fn output_cols_for_agrees_with_the_built_projection() {
        for panel in [
            vec![],
            vec![7],
            vec![0, 1, 2, 3],
            vec![5, 5, 10, 10, 10, 20],
            vec![20, 5, 100, 50],
            vec![9, 9, 9, 9],
        ] {
            let expected = HvgProjection::new_unchecked(panel.clone()).n_output_cols();
            assert_eq!(
                HvgProjection::output_cols_for(&panel),
                expected,
                "width disagreement on panel {panel:?}"
            );
        }
    }

    #[test]
    fn assess_hvg_panel_is_silent_on_an_already_canonical_panel() {
        assert_eq!(assess_hvg_panel(&[]), None);
        assert_eq!(assess_hvg_panel(&[42]), None);
        // `np.where(...)[0]` — the recipe every doc uses — is exactly this.
        assert_eq!(assess_hvg_panel(&[0, 3, 7, 900]), None);
    }

    #[test]
    fn assess_hvg_panel_reports_a_reordered_panel() {
        // `np.argsort(-variances)[:4]` shape: rank order, all unique.
        let v = assess_hvg_panel(&[900, 3, 7, 0]).expect("reordering must be reported");
        assert!(v.was_reordered);
        assert_eq!(v.requested_len, 4);
        assert_eq!(v.unique_len, 4, "nothing was dropped, only reordered");
    }

    #[test]
    fn assess_hvg_panel_reports_a_deduplicated_panel() {
        // Ascending, so the only change is the dedup.
        let v = assess_hvg_panel(&[5, 5, 10, 10, 10, 20]).expect("dedup must be reported");
        assert!(!v.was_reordered);
        assert_eq!(v.requested_len, 6);
        assert_eq!(v.unique_len, 3);
    }

    #[test]
    fn assess_hvg_panel_reports_both_at_once() {
        let v = assess_hvg_panel(&[20, 5, 5, 10]).expect("both changes must be reported");
        assert!(v.was_reordered);
        assert_eq!((v.requested_len, v.unique_len), (4, 3));
    }

    #[test]
    fn test_no_matching_genes_in_csr_row() {
        let proj = HvgProjection::new_unchecked(vec![100, 200, 300]);
        // CSR row has no genes matching the HVG set
        let csr_indices: Vec<i32> = vec![0, 1, 50, 99];
        let csr_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut output = vec![0.0f32; 3];

        proj.scatter_row(&csr_indices, &csr_data, &mut output);

        assert!(output.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_all_csr_genes_are_hvg() {
        let proj = HvgProjection::new_unchecked(vec![0, 1, 2, 3, 4]);
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
        let proj = HvgProjection::new_unchecked(genes);
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

    // ----- PFlog (v4) kernel tests ----------------------------------------
    //
    // Shared single-row reference (exact f64): given a dense raw-count row,
    //   z_j = log1p(4α·x_j) − mean_k log1p(4α·x_k).
    // No depth (v4). Kernel paths ride on f32, so assert to ~1e-5.

    /// Exact PFlog (v4) for one dense row over the FULL transcriptome.
    fn reference_dense_row(full: &[f32], four_alpha: f64) -> Vec<f64> {
        let logs: Vec<f64> = full
            .iter()
            .map(|&v| (four_alpha * v as f64).ln_1p())
            .collect();
        let mean = logs.iter().sum::<f64>() / logs.len() as f64;
        logs.iter().map(|&l| l - mean).collect()
    }

    /// Sparse (indices, data) for a dense row, dropping zeros.
    fn sparsify(full: &[f32]) -> (Vec<i32>, Vec<f32>) {
        let mut idx = Vec::new();
        let mut data = Vec::new();
        for (c, &v) in full.iter().enumerate() {
            if v != 0.0 {
                idx.push(c as i32);
                data.push(v);
            }
        }
        (idx, data)
    }

    #[test]
    fn test_pflog_row_full_matches_reference() {
        let full = vec![0.0f32, 1.0, 3.0, 0.0, 2.0];
        let (idx, data) = sparsify(&full);
        let four_alpha = 4.0; // α = 1
        let mut out = vec![0.0f32; full.len()];
        pflog_row_full(&idx, &data, four_alpha, full.len(), &mut out).unwrap();

        let expected = reference_dense_row(&full, four_alpha);
        for (j, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got as f64 - want).abs() <= 1e-5,
                "col {j}: got {got}, want {want}"
            );
        }
        // Row of the exact transform sums to ~0.
        let s: f64 = out.iter().map(|&v| v as f64).sum();
        assert!(s.abs() <= 1e-5, "row sum {s} not ~0");
        // Original-zero columns share the per-row baseline.
        assert!((out[0] - out[3]).abs() <= 1e-7);
    }

    #[test]
    fn test_pflog_row_full_various_alpha() {
        let full = vec![4.0f32, 0.0, 1.0, 0.0, 7.0, 2.0];
        let (idx, data) = sparsify(&full);
        for &four_alpha in &[0.4f64, 2.0, 4.0, 8.0] {
            let mut out = vec![0.0f32; full.len()];
            pflog_row_full(&idx, &data, four_alpha, full.len(), &mut out).unwrap();
            let expected = reference_dense_row(&full, four_alpha);
            for (&got, &want) in out.iter().zip(expected.iter()) {
                assert!(
                    (got as f64 - want).abs() <= 1e-5,
                    "four_alpha={four_alpha}: {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn test_pflog_row_full_empty_cell_zeroed() {
        // v4: an empty cell yields baseline 0 → the whole row is zero.
        let full = vec![0.0f32, 0.0, 0.0];
        let (idx, data) = sparsify(&full);
        let mut out = vec![9.0f32; full.len()]; // pre-dirtied
        pflog_row_full(&idx, &data, 4.0, full.len(), &mut out).unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "empty cell must zero the row"
        );
    }

    /// 🔴 The load-bearing property (review item A): projected PFlog must equal
    /// the corresponding columns of FULL-transcriptome PFlog — the centering
    /// denominator D and baseline sum are over all genes, not the panel.
    /// Exercises the dense-remap path.
    #[test]
    fn test_pflog_projection_equals_full_columns_dense_remap() {
        let full = vec![0.0f32, 1.0, 3.0, 0.0, 2.0, 5.0, 0.0, 4.0];
        let (idx, data) = sparsify(&full);
        let four_alpha = 4.0;
        let panel = vec![1u32, 2, 5, 7]; // strict subset
        let proj = HvgProjection::new_unchecked(panel.clone());
        assert!(proj.uses_dense_remap());

        let mut out = vec![0.0f32; proj.n_output_cols()];
        proj.scatter_pflog_row(&idx, &data, four_alpha, full.len(), &mut out);

        let full_ref = reference_dense_row(&full, four_alpha);
        for (pos, &g) in panel.iter().enumerate() {
            assert!(
                (out[pos] as f64 - full_ref[g as usize]).abs() <= 1e-5,
                "panel pos {pos} (gene {g}): got {}, want {} (full-transcriptome)",
                out[pos],
                full_ref[g as usize]
            );
        }
    }

    /// Same property via the merge-scan fallback (forced by a tiny dense-remap
    /// byte cap), so both projection code paths are guarded.
    #[test]
    fn test_pflog_projection_equals_full_columns_merge_scan() {
        let full = vec![0.0f32, 1.0, 3.0, 0.0, 2.0, 5.0, 0.0, 4.0];
        let (idx, data) = sparsify(&full);
        let four_alpha = 4.0;
        let panel = vec![1u32, 2, 5, 7];
        // max_dense_remap_bytes = 0 forces the merge-scan path.
        let proj = HvgProjection::new_with_dense_remap_limit(panel.clone(), 0);
        assert!(!proj.uses_dense_remap());

        let mut out = vec![0.0f32; proj.n_output_cols()];
        proj.scatter_pflog_row(&idx, &data, four_alpha, full.len(), &mut out);

        let full_ref = reference_dense_row(&full, four_alpha);
        for (pos, &g) in panel.iter().enumerate() {
            assert!(
                (out[pos] as f64 - full_ref[g as usize]).abs() <= 1e-5,
                "merge-scan panel pos {pos} (gene {g}): {} vs {}",
                out[pos],
                full_ref[g as usize]
            );
        }
    }

    /// A panel-LOCAL transform (centering D and baseline sum over the panel
    /// only) still differs from the full-transcriptome result in v4 — the
    /// per-element `log1p(4α·x)` matches, but the baseline (over full-D / full
    /// row) does not — so the property test above is non-trivial.
    #[test]
    fn test_pflog_full_differs_from_panel_local() {
        let full = vec![0.0f32, 1.0, 3.0, 0.0, 2.0, 5.0, 0.0, 4.0];
        let (idx, data) = sparsify(&full);
        let four_alpha = 4.0;
        let panel = vec![1u32, 2, 5, 7];
        let proj = HvgProjection::new_unchecked(panel.clone());

        let mut out = vec![0.0f32; proj.n_output_cols()];
        proj.scatter_pflog_row(&idx, &data, four_alpha, full.len(), &mut out);

        // Panel-local reference: D & baseline sum over the 4 panel genes only.
        let panel_vals: Vec<f32> = panel.iter().map(|&g| full[g as usize]).collect();
        let panel_local = reference_dense_row(&panel_vals, four_alpha);

        let max_diff = out
            .iter()
            .zip(panel_local.iter())
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff > 1e-3,
            "full-transcriptome and panel-local PFlog should differ, max_diff={max_diff}"
        );
    }
}
