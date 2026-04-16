//! Pseudobulk aggregation for differential expression.
//!
//! Aggregates single-cell counts into pseudobulk samples by grouping cells
//! according to metadata columns (e.g., `["perturbation", "donor"]`).
//! The resulting count matrix is fed to `pydeseq2` on the Python side for
//! negative binomial GLM testing.
//!
//! Supports both streaming (shard-by-shard via `BackedCsrReader`) and
//! in-memory (`ScxCsr`) paths.

use std::collections::HashMap;

use crate::Result;

/// Aggregation method for pseudobulk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Sum of counts per group (default for DESeq2).
    Sum,
    /// Mean of counts per group.
    Mean,
}

/// Result of pseudobulk aggregation.
#[derive(Debug, Clone)]
pub struct PseudobulkResult {
    /// Aggregated count matrix `[n_groups × n_vars]`, row-major.
    pub counts: Vec<f64>,
    /// Group labels: `group_labels[i]` is a Vec of column values for group `i`.
    /// E.g., for groupby `["perturbation", "donor"]`, group_labels[0] might be
    /// `["drug_A", "donor_1"]`.
    pub group_labels: Vec<Vec<String>>,
    /// Column names from groupby (e.g., `["perturbation", "donor"]`).
    pub groupby_columns: Vec<String>,
    /// Number of cells contributing to each group.
    pub cell_counts: Vec<usize>,
    /// Gene names.
    pub gene_names: Vec<String>,
    /// Number of groups.
    pub n_groups: usize,
    /// Number of variables (genes).
    pub n_vars: usize,
}

/// Build a group-key → group-index mapping from per-cell obs column vectors.
///
/// Each cell's group key is the concatenation of its values across all groupby
/// columns, joined by `"\x1F"` (unit separator, safe for any string value).
///
/// Returns:
/// - `cell_to_group`: group index for each cell (length = n_obs)
/// - `group_labels`: per-group label vectors (each Vec has len = n_groupby_cols)
/// - ordered deterministically (sorted by group key)
fn build_group_mapping(obs_groups: &[Vec<String>], n_obs: usize) -> (Vec<usize>, Vec<Vec<String>>) {
    let n_cols = obs_groups.len();

    // Build composite keys for each cell.
    let mut key_to_index: HashMap<String, usize> = HashMap::new();
    let mut group_labels: Vec<Vec<String>> = Vec::new();
    let mut cell_to_group = Vec::with_capacity(n_obs);

    for cell in 0..n_obs {
        // Build composite key.
        let mut key = String::new();
        for (col_idx, col) in obs_groups.iter().enumerate() {
            if col_idx > 0 {
                key.push('\x1F');
            }
            key.push_str(&col[cell]);
        }

        let group_idx = if let Some(&idx) = key_to_index.get(&key) {
            idx
        } else {
            let idx = group_labels.len();
            key_to_index.insert(key, idx);
            let labels: Vec<String> = (0..n_cols).map(|c| obs_groups[c][cell].clone()).collect();
            group_labels.push(labels);
            idx
        };

        cell_to_group.push(group_idx);
    }

    // Sort groups deterministically by their composite key.
    let mut sorted_indices: Vec<usize> = (0..group_labels.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        let key_a: String = group_labels[a].join("\x1F");
        let key_b: String = group_labels[b].join("\x1F");
        key_a.cmp(&key_b)
    });

    // Build remapping: old index → new index.
    let mut remap = vec![0usize; group_labels.len()];
    for (new_idx, &old_idx) in sorted_indices.iter().enumerate() {
        remap[old_idx] = new_idx;
    }

    // Apply remapping.
    let sorted_labels: Vec<Vec<String>> = sorted_indices
        .iter()
        .map(|&old| group_labels[old].clone())
        .collect();
    let remapped_cells: Vec<usize> = cell_to_group.iter().map(|&old| remap[old]).collect();

    (remapped_cells, sorted_labels)
}

/// Streaming pseudobulk aggregation from `BackedCsrReader`.
///
/// Iterates shards one at a time, accumulating per-group sums without
/// materializing the full matrix.
///
/// # Arguments
/// * `reader` — Backed CSR reader for shard-by-shard iteration.
/// * `obs_groups` — Per-cell group labels for each groupby column.
///   `obs_groups[col_idx][cell_idx]` is the label for cell `cell_idx` in column `col_idx`.
/// * `groupby_columns` — Column names from obs (e.g., `["perturbation", "donor"]`).
/// * `gene_names` — Gene names (length = n_vars).
/// * `method` — Aggregation method (Sum or Mean).
/// * `min_cells_per_group` — Groups with fewer cells are excluded from the result.
pub fn pseudobulk_aggregate(
    reader: &scx_format::backed::BackedCsrReader,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = reader.n_obs();
    let n_vars = reader.n_vars();

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    // Accumulate counts and cell counts.
    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];

    // Count cells per group.
    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }

    // Stream shards.
    let n_shards = reader.index().n_shards();
    let mut global_row = 0usize;

    for shard_idx in 0..n_shards {
        let shard_csr = reader
            .read_shard_uncached(shard_idx)
            .map_err(crate::AccelError::Scx)?;

        let shard_n_rows = shard_csr.n_rows();
        for row in 0..shard_n_rows {
            let cell_idx = global_row + row;
            let group_idx = cell_to_group[cell_idx];

            let start = shard_csr.indptr[row] as usize;
            let end = shard_csr.indptr[row + 1] as usize;
            for j in start..end {
                let col = shard_csr.indices[j] as usize;
                counts[group_idx * n_vars + col] += shard_csr.data[j] as f64;
            }
        }
        global_row += shard_n_rows;
    }

    // Apply mean if requested.
    if method == AggregationMethod::Mean {
        for g in 0..n_groups {
            if cell_counts[g] > 0 {
                let cc = cell_counts[g] as f64;
                for v in 0..n_vars {
                    counts[g * n_vars + v] /= cc;
                }
            }
        }
    }

    // Filter by min_cells_per_group.
    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// In-memory pseudobulk aggregation from a pre-loaded `ScxCsr`.
///
/// Same algorithm as `pseudobulk_aggregate()` but operates on a single
/// already-decoded CSR matrix instead of streaming shards.
pub fn pseudobulk_aggregate_inmemory(
    csr: &scx_sparse::ScxCsr,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = csr.shape;

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];

    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }

    // Iterate all rows of the CSR.
    for (row, &group_idx) in cell_to_group.iter().enumerate() {
        let start = csr.indptr[row] as usize;
        let end = csr.indptr[row + 1] as usize;
        for j in start..end {
            let col = csr.indices[j] as usize;
            counts[group_idx * n_vars + col] += csr.data[j] as f64;
        }
    }

    if method == AggregationMethod::Mean {
        for g in 0..n_groups {
            if cell_counts[g] > 0 {
                let cc = cell_counts[g] as f64;
                for v in 0..n_vars {
                    counts[g * n_vars + v] /= cc;
                }
            }
        }
    }

    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// In-memory pseudobulk aggregation from borrowed CSR slices.
///
/// Same algorithm as `pseudobulk_aggregate_inmemory()` but operates on
/// borrowed slices (`&[i64]`, `&[i32]`, `&[f32]`) instead of requiring
/// an owning `ScxCsr`. This enables zero-copy aggregation from numpy
/// arrays via `PyReadonlyArray1` without cloning the data.
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_aggregate_from_slices(
    shape: (usize, usize),
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];

    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }

    // Iterate all rows of the CSR using borrowed slices.
    for (row, &group_idx) in cell_to_group.iter().enumerate() {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for j in start..end {
            let col = indices[j] as usize;
            counts[group_idx * n_vars + col] += data[j] as f64;
        }
    }

    if method == AggregationMethod::Mean {
        for g in 0..n_groups {
            if cell_counts[g] > 0 {
                let cc = cell_counts[g] as f64;
                for v in 0..n_vars {
                    counts[g * n_vars + v] /= cc;
                }
            }
        }
    }

    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// Validate common inputs for both streaming and in-memory paths.
fn validate_inputs(
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    n_obs: usize,
    n_vars: usize,
) -> Result<()> {
    if obs_groups.is_empty() {
        return Err(crate::AccelError::InvalidInput(
            "obs_groups must not be empty".to_string(),
        ));
    }
    if obs_groups.len() != groupby_columns.len() {
        return Err(crate::AccelError::InvalidInput(format!(
            "obs_groups has {} columns but groupby_columns has {}",
            obs_groups.len(),
            groupby_columns.len()
        )));
    }
    for (i, col) in obs_groups.iter().enumerate() {
        if col.len() != n_obs {
            return Err(crate::AccelError::InvalidInput(format!(
                "obs_groups[{}] has {} entries but n_obs = {}",
                i,
                col.len(),
                n_obs
            )));
        }
    }
    if gene_names.len() != n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "gene_names has {} entries but n_vars = {}",
            gene_names.len(),
            n_vars
        )));
    }
    Ok(())
}

/// Filter groups by min_cells and build the final `PseudobulkResult`.
#[allow(clippy::too_many_arguments)]
fn filter_and_build_result(
    counts: Vec<f64>,
    group_labels: Vec<Vec<String>>,
    groupby_columns: &[String],
    cell_counts: Vec<usize>,
    gene_names: &[String],
    n_groups: usize,
    n_vars: usize,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    // Identify groups that pass the filter.
    let kept: Vec<usize> = (0..n_groups)
        .filter(|&g| cell_counts[g] >= min_cells_per_group)
        .collect();

    if kept.len() == n_groups {
        // No filtering needed.
        return Ok(PseudobulkResult {
            counts,
            group_labels,
            groupby_columns: groupby_columns.to_vec(),
            cell_counts,
            gene_names: gene_names.to_vec(),
            n_groups,
            n_vars,
        });
    }

    let new_n_groups = kept.len();
    let mut new_counts = Vec::with_capacity(new_n_groups * n_vars);
    let mut new_labels = Vec::with_capacity(new_n_groups);
    let mut new_cell_counts = Vec::with_capacity(new_n_groups);

    for &g in &kept {
        new_counts.extend_from_slice(&counts[g * n_vars..(g + 1) * n_vars]);
        new_labels.push(group_labels[g].clone());
        new_cell_counts.push(cell_counts[g]);
    }

    Ok(PseudobulkResult {
        counts: new_counts,
        group_labels: new_labels,
        groupby_columns: groupby_columns.to_vec(),
        cell_counts: new_cell_counts,
        gene_names: gene_names.to_vec(),
        n_groups: new_n_groups,
        n_vars,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_csr() -> scx_sparse::ScxCsr {
        // 6 cells × 4 genes
        // Cell 0: gene0=1, gene1=2
        // Cell 1: gene0=3, gene2=4
        // Cell 2: gene1=5, gene3=6
        // Cell 3: gene0=7, gene1=8
        // Cell 4: gene2=9, gene3=10
        // Cell 5: gene0=11
        let indptr = vec![0i64, 2, 4, 6, 8, 10, 11];
        let indices = vec![0i32, 1, 0, 2, 1, 3, 0, 1, 2, 3, 0];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        scx_sparse::ScxCsr::new_unchecked((6, 4), indptr, indices, data)
    }

    #[test]
    fn test_pseudobulk_sum_inmemory() {
        let csr = make_test_csr();
        // Groups: cells 0,1,2 → "A", cells 3,4,5 → "B"
        let obs_groups = vec![vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
        ]];
        let groupby = vec!["group".to_string()];
        let genes = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];

        let result = pseudobulk_aggregate_inmemory(
            &csr,
            &obs_groups,
            &groupby,
            &genes,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        assert_eq!(result.n_groups, 2);
        assert_eq!(result.n_vars, 4);
        assert_eq!(result.cell_counts, vec![3, 3]);

        // Group A (cells 0,1,2): g0=1+3=4, g1=2+5=7, g2=4, g3=6
        let a_idx = result
            .group_labels
            .iter()
            .position(|l| l[0] == "A")
            .unwrap();
        let a_row = &result.counts[a_idx * 4..(a_idx + 1) * 4];
        assert_eq!(a_row, &[4.0, 7.0, 4.0, 6.0]);

        // Group B (cells 3,4,5): g0=7+11=18, g1=8, g2=9, g3=10
        let b_idx = result
            .group_labels
            .iter()
            .position(|l| l[0] == "B")
            .unwrap();
        let b_row = &result.counts[b_idx * 4..(b_idx + 1) * 4];
        assert_eq!(b_row, &[18.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_pseudobulk_mean_inmemory() {
        let csr = make_test_csr();
        let obs_groups = vec![vec![
            "A".to_string(),
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
            "B".to_string(),
        ]];
        let groupby = vec!["group".to_string()];
        let genes = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];

        let result = pseudobulk_aggregate_inmemory(
            &csr,
            &obs_groups,
            &groupby,
            &genes,
            AggregationMethod::Mean,
            0,
        )
        .unwrap();

        let a_idx = result
            .group_labels
            .iter()
            .position(|l| l[0] == "A")
            .unwrap();
        let a_row = &result.counts[a_idx * 4..(a_idx + 1) * 4];
        // Mean of group A: sum / 3
        assert!((a_row[0] - 4.0 / 3.0).abs() < 1e-10);
        assert!((a_row[1] - 7.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_min_cells_filter() {
        let csr = make_test_csr();
        // 3 groups: A (cells 0,1), B (cell 2), C (cells 3,4,5)
        let obs_groups = vec![vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "C".to_string(),
            "C".to_string(),
        ]];
        let groupby = vec!["group".to_string()];
        let genes = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];

        // min_cells=2 → B (1 cell) should be excluded
        let result = pseudobulk_aggregate_inmemory(
            &csr,
            &obs_groups,
            &groupby,
            &genes,
            AggregationMethod::Sum,
            2,
        )
        .unwrap();

        assert_eq!(result.n_groups, 2);
        let labels: Vec<&str> = result.group_labels.iter().map(|l| l[0].as_str()).collect();
        assert!(labels.contains(&"A"));
        assert!(labels.contains(&"C"));
        assert!(!labels.contains(&"B"));
    }

    #[test]
    fn test_multi_column_groupby() {
        let csr = make_test_csr();
        // Two groupby columns: perturbation and donor
        let obs_groups = vec![
            vec![
                "drug".to_string(),
                "drug".to_string(),
                "ctrl".to_string(),
                "ctrl".to_string(),
                "drug".to_string(),
                "drug".to_string(),
            ],
            vec![
                "d1".to_string(),
                "d1".to_string(),
                "d1".to_string(),
                "d2".to_string(),
                "d2".to_string(),
                "d2".to_string(),
            ],
        ];
        let groupby = vec!["perturbation".to_string(), "donor".to_string()];
        let genes = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];

        let result = pseudobulk_aggregate_inmemory(
            &csr,
            &obs_groups,
            &groupby,
            &genes,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        // Groups: (ctrl, d1)→cell2, (ctrl, d2)→cell3, (drug, d1)→cells0,1, (drug, d2)→cells4,5
        assert_eq!(result.n_groups, 4);
        assert_eq!(result.groupby_columns, vec!["perturbation", "donor"]);

        // Check (drug, d1): cells 0,1 → g0=1+3=4, g1=2, g2=4, g3=0
        let drug_d1_idx = result
            .group_labels
            .iter()
            .position(|l| l[0] == "drug" && l[1] == "d1")
            .unwrap();
        let row = &result.counts[drug_d1_idx * 4..(drug_d1_idx + 1) * 4];
        assert_eq!(row, &[4.0, 2.0, 4.0, 0.0]);
        assert_eq!(result.cell_counts[drug_d1_idx], 2);
    }

    #[test]
    fn test_validation_errors() {
        let csr = make_test_csr();
        let genes = vec![
            "g0".to_string(),
            "g1".to_string(),
            "g2".to_string(),
            "g3".to_string(),
        ];

        // Empty obs_groups
        let err = pseudobulk_aggregate_inmemory(&csr, &[], &[], &genes, AggregationMethod::Sum, 0);
        assert!(err.is_err());

        // Wrong number of cells
        let bad_groups = vec![vec!["A".to_string(), "B".to_string()]]; // only 2 cells, need 6
        let err = pseudobulk_aggregate_inmemory(
            &csr,
            &bad_groups,
            &["group".to_string()],
            &genes,
            AggregationMethod::Sum,
            0,
        );
        assert!(err.is_err());

        // Wrong number of genes
        let obs = vec![vec!["A".to_string(); 6]];
        let bad_genes = vec!["g0".to_string(), "g1".to_string()]; // only 2, need 4
        let err = pseudobulk_aggregate_inmemory(
            &csr,
            &obs,
            &["group".to_string()],
            &bad_genes,
            AggregationMethod::Sum,
            0,
        );
        assert!(err.is_err());
    }
}
