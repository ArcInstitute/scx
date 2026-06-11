//! CSC pseudobulk aggregation restricted to a gene subset.
//!
//! The full-gene CSR pseudobulk path already streams shards row-major
//! and accumulates per-group sums; CSC offers no measurable win
//! against it. Where CSC pays off is the *projected* case: a caller
//! has narrowed the gene axis (`col_indices`) and wants to read only
//! those columns. With CSC we skip non-overlapping shards entirely
//! and decode just the requested columns; with CSR we'd decode every
//! row and project per-shard.
//!
//! The kernel signature reflects this: `col_indices` is required.
//! Passing the full gene set works but defeats the purpose; pyscx
//! gates on this and rejects empty `col_indices` for the CSC path.

use crate::error::{AccelError, Result};
use crate::pseudobulk::{AggregationMethod, PseudobulkResult};
use scx_format_io::ColumnShardSource;

/// Aggregate counts per group on a CSC source, restricted to the
/// `col_indices` gene subset.
///
/// `cell_to_group[cell] -> group_idx in 0..n_groups`. Groups absent
/// from `cell_to_group` (no contributing cell) end up at zero, but are
/// filtered out by `min_cells_per_group` if requested.
///
/// Output `PseudobulkResult.counts` has shape `[n_groups × col_indices.len()]`
/// (row-major). `gene_names` should match the *projected* gene set —
/// the caller is responsible for slicing the original gene name vector
/// by `col_indices` before passing it in.
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_aggregate_csc<S: ColumnShardSource + ?Sized>(
    source: &S,
    cell_to_group: &[usize],
    n_groups: usize,
    group_labels: Vec<Vec<String>>,
    groupby_columns: &[String],
    gene_names: &[String],
    col_indices: &[u32],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = source.n_obs();
    let n_proj = col_indices.len();

    if cell_to_group.len() != n_obs {
        return Err(AccelError::InvalidInput(format!(
            "cell_to_group length {} != source.n_obs() {}",
            cell_to_group.len(),
            n_obs
        )));
    }
    if n_proj == 0 {
        return Err(AccelError::InvalidInput(
            "col_indices must be non-empty for the CSC pseudobulk path".to_string(),
        ));
    }
    if gene_names.len() != n_proj {
        return Err(AccelError::ShapeError(format!(
            "gene_names length {} != col_indices length {}",
            gene_names.len(),
            n_proj
        )));
    }
    if group_labels.len() != n_groups {
        return Err(AccelError::ShapeError(format!(
            "group_labels has {} entries but n_groups = {}",
            group_labels.len(),
            n_groups
        )));
    }

    let mut counts = vec![0.0f64; n_groups * n_proj];
    let mut cell_counts = vec![0usize; n_groups];
    for &g in cell_to_group {
        if g < n_groups {
            cell_counts[g] += 1;
        }
    }

    // `read_csc_columns_subset` is on `BackedCscReader` only, not on
    // the trait. Walk `col_indices` in sorted contiguous-run order and
    // call the trait's `read_csc_columns(Range<u32>)` per run.
    //
    // Sort col_indices and remember their original output positions.
    let mut sorted_with_pos: Vec<(u32, usize)> = col_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();
    sorted_with_pos.sort_by_key(|(c, _)| *c);

    // Walk contiguous runs of sorted columns.
    let mut i = 0;
    while i < sorted_with_pos.len() {
        let mut j = i + 1;
        while j < sorted_with_pos.len() && sorted_with_pos[j].0 == sorted_with_pos[j - 1].0 + 1 {
            j += 1;
        }
        let run_start = sorted_with_pos[i].0;
        let run_end = sorted_with_pos[j - 1].0 + 1;

        let csc_run = source
            .read_csc_columns(run_start..run_end)
            .map_err(AccelError::Scx)?;

        // For each col in the run, accumulate into counts[g, output_col].
        let run_n_cols = (run_end - run_start) as usize;
        for local_col in 0..run_n_cols {
            let output_col = sorted_with_pos[i + local_col].1;
            let s = csc_run.indptr[local_col] as usize;
            let e = csc_run.indptr[local_col + 1] as usize;
            for k in s..e {
                let row = csc_run.indices[k] as usize;
                if row >= n_obs {
                    continue;
                }
                let group_idx = cell_to_group[row];
                if group_idx >= n_groups {
                    continue;
                }
                counts[group_idx * n_proj + output_col] += csc_run.data[k] as f64;
            }
        }

        i = j;
    }

    if method == AggregationMethod::Mean {
        for g in 0..n_groups {
            if cell_counts[g] > 0 {
                let cc = cell_counts[g] as f64;
                for v in 0..n_proj {
                    counts[g * n_proj + v] /= cc;
                }
            }
        }
    }

    // Filter groups by min_cells_per_group inline (the public helper
    // in `pseudobulk` is private to that module; replicate the small
    // amount of bookkeeping here).
    if min_cells_per_group <= 1 {
        return Ok(PseudobulkResult {
            counts,
            group_labels,
            groupby_columns: groupby_columns.to_vec(),
            cell_counts,
            gene_names: gene_names.to_vec(),
            n_groups,
            n_vars: n_proj,
        });
    }

    let kept: Vec<usize> = (0..n_groups)
        .filter(|&g| cell_counts[g] >= min_cells_per_group)
        .collect();

    if kept.len() == n_groups {
        return Ok(PseudobulkResult {
            counts,
            group_labels,
            groupby_columns: groupby_columns.to_vec(),
            cell_counts,
            gene_names: gene_names.to_vec(),
            n_groups,
            n_vars: n_proj,
        });
    }

    let new_n = kept.len();
    let mut new_counts = Vec::with_capacity(new_n * n_proj);
    let mut new_labels = Vec::with_capacity(new_n);
    let mut new_cell_counts = Vec::with_capacity(new_n);
    for &g in &kept {
        new_counts.extend_from_slice(&counts[g * n_proj..(g + 1) * n_proj]);
        new_labels.push(group_labels[g].clone());
        new_cell_counts.push(cell_counts[g]);
    }

    Ok(PseudobulkResult {
        counts: new_counts,
        group_labels: new_labels,
        groupby_columns: groupby_columns.to_vec(),
        cell_counts: new_cell_counts,
        gene_names: gene_names.to_vec(),
        n_groups: new_n,
        n_vars: n_proj,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csc::test_helpers::{deterministic_dense, write_csr_csc_test_file};
    use crate::pseudobulk::pseudobulk_aggregate;
    use scx_format_io::{BackedCscReader, BackedCsrReader, ScxReader};
    use tempfile::tempdir;

    #[test]
    fn pseudobulk_csc_matches_csr_on_projected_subset() {
        let dir = tempdir().unwrap();
        let n_obs = 12usize;
        let n_vars = 8usize;
        let dense = deterministic_dense(n_obs, n_vars);
        let path = write_csr_csc_test_file(dir.path(), "pb", n_obs, n_vars, &dense, 4);

        let groupby_columns = vec!["batch".to_string()];
        let labels: Vec<String> = (0..n_obs)
            .map(|i| {
                if i < n_obs / 2 {
                    "A".to_string()
                } else {
                    "B".to_string()
                }
            })
            .collect();
        let obs_groups = vec![labels];
        let gene_names: Vec<String> = (0..n_vars).map(|j| format!("g{j}")).collect();

        let csr_reader = BackedCsrReader::new(ScxReader::open(&path).unwrap(), 0);
        let csr_full = pseudobulk_aggregate(
            &csr_reader,
            &obs_groups,
            &groupby_columns,
            &gene_names,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        // Project to a subset of genes.
        let col_indices: Vec<u32> = vec![1, 3, 5, 7];
        let projected_gene_names: Vec<String> = col_indices
            .iter()
            .map(|&c| gene_names[c as usize].clone())
            .collect();

        // Build cell_to_group/group_labels matching the CSR full result's
        // group ordering.
        let n_groups = csr_full.n_groups;
        let group_labels = csr_full.group_labels.clone();
        // Build cell_to_group from obs_groups using the same key->index
        // map the CSR path produced. We already know n_groups == 2 with
        // group_labels[[A]], [[B]] in lexicographic order.
        let cell_to_group: Vec<usize> = (0..n_obs)
            .map(|i| {
                let label = if i < n_obs / 2 { "A" } else { "B" };
                group_labels.iter().position(|gl| gl[0] == label).unwrap()
            })
            .collect();

        let csc_reader = BackedCscReader::new(ScxReader::open(&path).unwrap(), 0).unwrap();
        let csc_proj = pseudobulk_aggregate_csc(
            &csc_reader,
            &cell_to_group,
            n_groups,
            group_labels.clone(),
            &groupby_columns,
            &projected_gene_names,
            &col_indices,
            AggregationMethod::Sum,
            0,
        )
        .unwrap();

        // Compare slice-by-slice: CSR full result projected to the
        // selected columns should match the CSC projected result.
        for g in 0..n_groups {
            for (out_col, &orig_col) in col_indices.iter().enumerate() {
                let csr_val = csr_full.counts[g * csr_full.n_vars + orig_col as usize];
                let csc_val = csc_proj.counts[g * csc_proj.n_vars + out_col];
                assert!(
                    (csr_val - csc_val).abs() < 1e-9,
                    "g={g} out_col={out_col} (orig {orig_col}): csr={csr_val} csc={csc_val}"
                );
            }
        }
    }
}
