//! Pseudobulk aggregation for differential expression.
//!
//! Aggregates single-cell counts into pseudobulk samples by grouping cells
//! according to metadata columns (e.g., `["perturbation", "donor"]`).
//! The resulting count matrix is fed to `pydeseq2` on the Python side for
//! negative binomial GLM testing.
//!
//! Supports both streaming (shard-by-shard over any CSR `ShardSource`) and
//! in-memory (`ScxCsr`) paths. Streaming callers holding a subset SCX handle
//! must pass that handle's *view* (`as_shard_source()`), not the reader
//! underneath it — `obs_groups` is indexed by visible cell.

use std::collections::HashMap;

use scx_format_io::ShardSource;

use crate::Result;

/// Aggregation method for pseudobulk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Sum of counts per group (default for DESeq2).
    Sum,
    /// Mean of counts per group.
    Mean,
}

/// pdex-style pseudobulk mode controlling per-cell and per-group-mean transforms.
///
/// Encodes the four `(geometric_mean × is_log1p)` combinations from
/// `pdex._math.pseudobulk`. The per-cell transform `f(x)` is applied to each
/// cell's expression value before averaging; the per-group-mean transform
/// `g(y)` is applied to the resulting mean. Both `f(0) = 0` and `g(0) = 0`
/// hold for all four modes, so CSR aggregation only visits non-zero entries.
///
/// | mode             | `geometric_mean` | `is_log1p` | `f(x)`    | `g(y)`     |
/// |------------------|------------------|------------|-----------|------------|
/// | `ArithRaw`       | false            | false      | `x`       | `y`        |
/// | `ArithLog1pExpand` | false          | true       | `expm1(x)`| `y`        |
/// | `GeomRaw`        | true             | false      | `log1p(x)`| `expm1(y)` |
/// | `GeomLog1p`      | true             | true       | `x`       | `expm1(y)` |
///
/// The output mean is always in **natural (count) space**, matching pdex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomMeanMode {
    /// `geometric_mean=False, is_log1p=False`: arithmetic mean of raw counts.
    ArithRaw,
    /// `geometric_mean=False, is_log1p=True`: arithmetic mean of `expm1(X)`.
    ArithLog1pExpand,
    /// `geometric_mean=True, is_log1p=False`: `expm1(mean(log1p(X)))`.
    GeomRaw,
    /// `geometric_mean=True, is_log1p=True`: `expm1(mean(X))`.
    GeomLog1p,
}

impl GeomMeanMode {
    /// Per-cell-value transform `f(x)`. All four modes satisfy `f(0) = 0`,
    /// so sparse-aggregation paths can skip explicit zeros.
    #[inline]
    pub fn pre(self, x: f64) -> f64 {
        match self {
            Self::ArithRaw | Self::GeomLog1p => x,
            Self::ArithLog1pExpand => x.exp_m1(),
            Self::GeomRaw => x.ln_1p(),
        }
    }

    /// Per-(group, gene) mean transform `g(y)`. Applied after dividing the
    /// sum of `f(x_i)` by the group's cell count.
    #[inline]
    pub fn post(self, y: f64) -> f64 {
        match self {
            Self::ArithRaw | Self::ArithLog1pExpand => y,
            Self::GeomRaw | Self::GeomLog1p => y.exp_m1(),
        }
    }

    /// Convenience constructor from the `(geometric_mean, is_log1p)` pair
    /// pdex's Python API exposes.
    #[inline]
    pub fn from_flags(geometric_mean: bool, is_log1p: bool) -> Self {
        match (geometric_mean, is_log1p) {
            (false, false) => Self::ArithRaw,
            (false, true) => Self::ArithLog1pExpand,
            (true, false) => Self::GeomRaw,
            (true, true) => Self::GeomLog1p,
        }
    }

    /// The count-space **arithmetic** counterpart of this mode, preserving its
    /// `is_log1p` interpretation. This is what pdex's `cpm_bulk` uses for the
    /// `cpm_filter` decision — a per-gene arithmetic mean in count space,
    /// independent of whether the reported mean is geometric.
    #[inline]
    pub fn arith(self) -> Self {
        match self {
            Self::ArithRaw | Self::GeomRaw => Self::ArithRaw,
            Self::ArithLog1pExpand | Self::GeomLog1p => Self::ArithLog1pExpand,
        }
    }
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
/// The group key is the cell's tuple of values across all groupby columns
/// (e.g. `("drug_A", "donor_1")`). Strings are interned once per column,
/// giving each distinct label a `u32` id; the per-cell hot path then builds
/// a `Vec<u32>` of ids and looks it up by borrowed slice, so lookups of
/// already-seen groups allocate nothing. This replaces the old `"\x1F"`
/// joined-string scheme (which collided on values containing the separator)
/// without re-introducing the per-cell `String` allocations that scheme paid.
///
/// Returns:
/// - `cell_to_group`: group index for each cell (length = n_obs)
/// - `group_labels`: per-group label vectors (each Vec has len = n_groupby_cols)
/// - ordered deterministically (sorted lexicographically by the label tuple)
pub fn build_group_mapping(
    obs_groups: &[Vec<String>],
    n_obs: usize,
) -> (Vec<usize>, Vec<Vec<String>>) {
    let n_cols = obs_groups.len();

    // Per-column string interners. Keyed by `&str` borrowed from obs_groups;
    // the backing Vec<String>s live for the full call, so the borrow is sound.
    let mut col_interners: Vec<HashMap<&str, u32>> = (0..n_cols).map(|_| HashMap::new()).collect();
    let mut col_vocab: Vec<Vec<&str>> = vec![Vec::new(); n_cols];

    // Group table: `Vec<u32>` of interned ids -> group index.
    let mut key_to_index: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut group_label_ids: Vec<Vec<u32>> = Vec::new();
    let mut cell_to_group = Vec::with_capacity(n_obs);

    // Reused per-cell buffer — avoids n_obs * n_cols * u32 reallocations.
    let mut key_buf: Vec<u32> = Vec::with_capacity(n_cols);

    // Parallel-index across `n_cols` columns — idiomatic `for cell in ...` is
    // clearer than the clippy-suggested iterator chain over the first column.
    #[allow(clippy::needless_range_loop)]
    for cell in 0..n_obs {
        key_buf.clear();
        for col_idx in 0..n_cols {
            let s: &str = obs_groups[col_idx][cell].as_str();
            let id = match col_interners[col_idx].get(s) {
                Some(&id) => id,
                None => {
                    let id = col_vocab[col_idx].len() as u32;
                    col_interners[col_idx].insert(s, id);
                    col_vocab[col_idx].push(s);
                    id
                }
            };
            key_buf.push(id);
        }

        // Borrowed-slice lookup: `Vec<u32>: Borrow<[u32]>`, so we hit without
        // allocating a key. Only new groups pay a single clone on insert.
        let group_idx = if let Some(&idx) = key_to_index.get(key_buf.as_slice()) {
            idx
        } else {
            let idx = group_label_ids.len();
            key_to_index.insert(key_buf.clone(), idx);
            group_label_ids.push(key_buf.clone());
            idx
        };

        cell_to_group.push(group_idx);
    }

    // Materialize owned group labels from interned ids (once per unique group).
    let group_labels: Vec<Vec<String>> = group_label_ids
        .iter()
        .map(|ids| {
            ids.iter()
                .enumerate()
                .map(|(col_idx, &id)| col_vocab[col_idx][id as usize].to_string())
                .collect()
        })
        .collect();

    // Sort groups deterministically by their label tuple (lexicographic).
    let mut sorted_indices: Vec<usize> = (0..group_labels.len()).collect();
    sorted_indices.sort_by(|&a, &b| group_labels[a].cmp(&group_labels[b]));

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

/// Streaming pseudobulk aggregation over a CSR [`ShardSource`].
///
/// Iterates shards one at a time, accumulating per-group sums without
/// materializing the full matrix.
///
/// Generic over the source rather than taking a `BackedCsrReader`, because
/// `obs_groups` is indexed by *visible* cell: a caller holding a subset SCX
/// handle must pass that handle's view (`as_shard_source()`), not the reader
/// underneath it, or the group labels line up against the wrong rows.
///
/// # Arguments
/// * `source` — CSR shard source for shard-by-shard iteration.
/// * `obs_groups` — Per-cell group labels for each groupby column.
///   `obs_groups[col_idx][cell_idx]` is the label for cell `cell_idx` in column `col_idx`.
/// * `groupby_columns` — Column names from obs (e.g., `["perturbation", "donor"]`).
/// * `gene_names` — Gene names (length = n_vars).
/// * `method` — Aggregation method (Sum or Mean).
/// * `min_cells_per_group` — Groups with fewer cells are excluded from the result.
pub fn pseudobulk_aggregate<S: ShardSource + Sync>(
    source: &S,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();

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

    // Stream shards with ordered decode-prefetch (2.1). Counts land in
    // per-group bins keyed by the global cell index, so shards must be consumed
    // in order for the `global_row` cursor to map cells correctly — StableOrder
    // is the only valid mode. Pseudobulk is a **single pass**, so use the
    // *uncached* prefetch variant: `read_shard` (not the LRU `read_shard_arc`),
    // preserving the pre-2.1 `read_shard_uncached` behaviour so this pass does
    // not warm/evict the shared shard cache (review feedback).
    let mut global_row = 0usize;
    crate::prefetch::for_each_shard_ordered_uncached(
        source,
        crate::prefetch::prefetch_depth(),
        |_shard_idx, shard_csr| {
            let _r = scx_format_io::reduction_guard();
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
            Ok(())
        },
    )?;

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

/// In-memory pseudobulk aggregation from a row-major dense `f32` matrix.
///
/// Avoids the `scipy.sparse.csr_matrix(dense_array)` densification round-trip
/// that the CSR-based paths pay when the caller's `X` is already dense.
/// Builds the same `PseudobulkResult` shape as `pseudobulk_aggregate_inmemory`
/// / `pseudobulk_aggregate_from_slices`.
///
/// At Replogle scale (n_obs ≈ 24K cells × n_vars ≈ 18K genes) the dense path
/// is ~50–100× faster than going through `scipy.sparse.csr_matrix`, because
/// the CSR conversion scans every f32 looking for non-zeros and materialises
/// 1.7 GB of `(indices, data)` arrays just to be summed back into a dense
/// per-group means matrix. The dense path skips that intermediate altogether.
///
/// Parallelisation is over groups (each thread writes to its own contiguous
/// row of `means`, so no shared-accumulator contention or thread-local
/// `n_groups × n_vars` blow-up — the per-thread allocator pattern would have
/// allocated ~11 GB across 32 threads for typical Replogle shapes).
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_aggregate_dense(
    data: &[f32],
    shape: (usize, usize),
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    use rayon::prelude::*;

    let (n_obs, n_vars) = shape;
    if data.len() != n_obs * n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "data length {} != n_obs ({}) × n_vars ({}) = {}",
            data.len(),
            n_obs,
            n_vars,
            n_obs * n_vars,
        )));
    }

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    // Invert cell_to_group → per-group list of cell row indices. One pass,
    // O(n_obs) time + O(n_obs + n_groups) memory.
    let mut cells_by_group: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    for (cell, &g) in cell_to_group.iter().enumerate() {
        cells_by_group[g].push(cell as u32);
    }
    let cell_counts: Vec<usize> = cells_by_group.iter().map(|c| c.len()).collect();

    // Allocate the result `[n_groups × n_vars]` matrix once, then have each
    // group sum (and optionally mean-normalise) its own cells into its
    // dedicated row in parallel. No shared mutable state across threads:
    // each thread owns a disjoint row range. Folding the divide into the
    // same loop avoids a sequential `n_groups × n_vars` post-pass (~432M
    // divisions at Replogle scale).
    let mut counts = vec![0.0f64; n_groups * n_vars];
    let want_mean = method == AggregationMethod::Mean;
    counts
        .par_chunks_mut(n_vars)
        .zip(cells_by_group.par_iter())
        .with_min_len(1)
        .for_each(|(dst, cells)| {
            for &cell in cells {
                let src = &data[cell as usize * n_vars..(cell as usize + 1) * n_vars];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s as f64;
                }
            }
            if want_mean && !cells.is_empty() {
                let cc = cells.len() as f64;
                for d in dst.iter_mut() {
                    *d /= cc;
                }
            }
        });

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

// ──────────────────────────────────────────────────────────────────────────
// GPU pseudobulk means.
//
// GPU-accelerate ONLY the aggregation (the cell-count-scaling step): produce
// per-group means `[n_groups × n_vars]` f64 by wrapping the scx-gpu DE
// pseudobulk primitives, then reuse `filter_and_build_result` so the
// `PseudobulkResult` is identical to the CPU `pseudobulk_aggregate*` output
// (same lexicographic group ordering, min-cells filtering, and cell counts).
// The five bulk metrics run downstream on the host. GPU sums accumulate in f64
// with the identity pre-transform, matching the CPU f32→f64 accumulate.
// ──────────────────────────────────────────────────────────────────────────

/// Sorted group mapping shared by the GPU means paths: per-cell group id (i32,
/// in the same lexicographic order as [`build_group_mapping`]), the sorted
/// group labels, and per-group cell counts.
#[cfg(feature = "gpu")]
fn gpu_group_plan(
    obs_groups: &[Vec<String>],
    n_obs: usize,
) -> (Vec<i32>, Vec<Vec<String>>, Vec<usize>) {
    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();
    let mut cell_counts = vec![0usize; n_groups];
    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }
    let cell_to_group_i32 = cell_to_group.iter().map(|&g| g as i32).collect();
    (cell_to_group_i32, group_labels, cell_counts)
}

/// GPU pseudobulk **means** over a CSR [`ShardSource`]. Mirrors
/// [`pseudobulk_aggregate`] with `AggregationMethod::Mean`, but streams the
/// shards in-VRAM and folds them with the DE pseudobulk kernel on the GPU.
///
/// Generic for the same reason as [`pseudobulk_aggregate`]: `obs_groups` is
/// per *visible* cell, so a subset SCX handle must hand over its view.
#[cfg(feature = "gpu")]
pub fn pseudobulk_means_gpu_streaming<S: ShardSource + Sync>(
    dev: &scx_gpu::GpuDevice,
    source: &S,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();
    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels, cell_counts) = gpu_group_plan(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let means = scx_gpu::gpu_pseudobulk_means_csr(
        dev,
        source,
        &cell_to_group,
        n_groups,
        n_vars,
        &cell_counts,
    )
    .map_err(|e| crate::AccelError::LinAlg(format!("GPU pseudobulk means (streaming): {e}")))?;

    filter_and_build_result(
        means,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// GPU pseudobulk means from in-memory CSR slices (scipy CSR). Mirrors
/// [`pseudobulk_aggregate_from_slices`] with `Mean`; wraps the borrowed CSR in
/// a single-shard source for the streaming kernel.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_means_gpu_from_slices(
    dev: &scx_gpu::GpuDevice,
    shape: (usize, usize),
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;
    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels, cell_counts) = gpu_group_plan(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let csr =
        scx_sparse::ScxCsr::new_unchecked(shape, indptr.to_vec(), indices.to_vec(), data.to_vec());
    let source = scx_format_io::shard_source::SingleShardSource { csr: &csr };

    let means = scx_gpu::gpu_pseudobulk_means_csr(
        dev,
        &source,
        &cell_to_group,
        n_groups,
        n_vars,
        &cell_counts,
    )
    .map_err(|e| crate::AccelError::LinAlg(format!("GPU pseudobulk means (csr): {e}")))?;

    filter_and_build_result(
        means,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// GPU pseudobulk means from a dense row-major `[n_obs × n_vars]` f32 matrix
/// (in-memory dense `X` or an `obsm` embedding). Mirrors
/// [`pseudobulk_aggregate_dense`] with `Mean`.
#[cfg(feature = "gpu")]
pub fn pseudobulk_means_gpu_dense(
    dev: &scx_gpu::GpuDevice,
    data: &[f32],
    shape: (usize, usize),
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;
    if data.len() != n_obs * n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "data length {} != n_obs ({}) × n_vars ({})",
            data.len(),
            n_obs,
            n_vars
        )));
    }
    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels, cell_counts) = gpu_group_plan(obs_groups, n_obs);
    let n_groups = group_labels.len();

    // Concatenated per-group cell lists + prefix offsets that the dense kernel
    // consumes (built in the sorted group order from `gpu_group_plan`).
    let mut group_offsets = vec![0i32; n_groups + 1];
    for g in 0..n_groups {
        group_offsets[g + 1] = group_offsets[g] + cell_counts[g] as i32;
    }
    let mut cursor: Vec<i32> = group_offsets[..n_groups].to_vec();
    let mut all_group_cells = vec![0i32; n_obs];
    for (cell, &g) in cell_to_group.iter().enumerate() {
        let g = g as usize;
        all_group_cells[cursor[g] as usize] = cell as i32;
        cursor[g] += 1;
    }

    let means = scx_gpu::gpu_pseudobulk_means_dense(
        dev,
        data,
        n_obs,
        n_vars,
        &all_group_cells,
        &group_offsets,
        n_groups,
        &cell_counts,
    )
    .map_err(|e| crate::AccelError::LinAlg(format!("GPU pseudobulk means (dense): {e}")))?;

    filter_and_build_result(
        means,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
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

    #[test]
    fn test_geom_mean_mode_transforms() {
        // Each row: (mode, x, expected_pre, mean (sum/2), expected_post(mean))
        let cases: &[(GeomMeanMode, f64)] = &[
            (GeomMeanMode::ArithRaw, 2.5),
            (GeomMeanMode::ArithLog1pExpand, 1.5),
            (GeomMeanMode::GeomRaw, 1.5),
            (GeomMeanMode::GeomLog1p, 0.5),
        ];

        for &(mode, x) in cases {
            // f(0) == 0 invariant — required for CSR aggregation correctness.
            assert!(
                mode.pre(0.0).abs() < 1e-15,
                "{:?}.pre(0.0) must equal 0 (got {})",
                mode,
                mode.pre(0.0)
            );

            // Match pdex's _math.pseudobulk reference behavior on a single value.
            let pre = mode.pre(x);
            let post = mode.post(pre);
            let expected = match mode {
                GeomMeanMode::ArithRaw => x,
                GeomMeanMode::ArithLog1pExpand => x.exp_m1(),
                GeomMeanMode::GeomRaw => x.ln_1p().exp_m1(), // = x for x > -1
                GeomMeanMode::GeomLog1p => x.exp_m1(),
            };
            assert!(
                (post - expected).abs() < 1e-12,
                "{:?} round-trip: post(pre({})) = {} != {}",
                mode,
                x,
                post,
                expected
            );
        }
    }

    #[test]
    fn test_geom_mean_mode_from_flags() {
        assert_eq!(
            GeomMeanMode::from_flags(false, false),
            GeomMeanMode::ArithRaw
        );
        assert_eq!(
            GeomMeanMode::from_flags(false, true),
            GeomMeanMode::ArithLog1pExpand
        );
        assert_eq!(GeomMeanMode::from_flags(true, false), GeomMeanMode::GeomRaw);
        assert_eq!(
            GeomMeanMode::from_flags(true, true),
            GeomMeanMode::GeomLog1p
        );
    }

    #[test]
    fn test_pseudobulk_dense_matches_csr_inmemory() {
        // The dense kernel should produce bit-identical sums to the CSR
        // kernel when given the dense expansion of the same matrix.
        let csr = make_test_csr();
        let (n_obs, n_vars) = csr.shape;
        let mut dense = vec![0.0f32; n_obs * n_vars];
        for row in 0..n_obs {
            let s = csr.indptr[row] as usize;
            let e = csr.indptr[row + 1] as usize;
            for j in s..e {
                let col = csr.indices[j] as usize;
                dense[row * n_vars + col] = csr.data[j];
            }
        }

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

        for method in [AggregationMethod::Sum, AggregationMethod::Mean] {
            let csr_res =
                pseudobulk_aggregate_inmemory(&csr, &obs_groups, &groupby, &genes, method, 0)
                    .unwrap();
            let dense_res = pseudobulk_aggregate_dense(
                &dense,
                (n_obs, n_vars),
                &obs_groups,
                &groupby,
                &genes,
                method,
                0,
            )
            .unwrap();
            assert_eq!(csr_res.n_groups, dense_res.n_groups);
            assert_eq!(csr_res.cell_counts, dense_res.cell_counts);
            assert_eq!(csr_res.group_labels, dense_res.group_labels);
            // f64 sums of the exact same f32 values; bit-identical.
            for (a, b) in csr_res.counts.iter().zip(dense_res.counts.iter()) {
                assert!(
                    (a - b).abs() < 1e-12,
                    "method={method:?} mismatch: csr={a} dense={b}"
                );
            }
        }
    }
}
